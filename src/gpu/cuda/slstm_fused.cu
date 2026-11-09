// Eager fused-gate sLSTM: one launch per timestep, gate pre-activation assembled
// in-kernel from the input half, the recurrent half and the bias.

// Fused-gate forward step. Same recurrence as slstm_cell_step, but a gate
// pre-activation is assembled here from three pieces instead of arriving as one
// tensor: the input half `g` (a [B, T, 4H] buffer holding x·Wx for every
// timestep), the recurrent half `gh` (a [B, 4H] scratch holding h_{t-1}·Wh for
// this timestep) and the bias. The saved tensors are [B, T, H] slabs indexed by
// (b·T + t)·H + j rather than one small tensor per timestep.
extern "C" __global__ void slstm_gate_matvec(
        const float* __restrict__ h_state, const float* __restrict__ wh,
        float* __restrict__ gh, int H, int B, int cols_per_thread) {
    extern __shared__ float sh[];   // [H] — this block's copy of h for batch `b`
    int H4 = 4 * H;
    int cols_per_block = blockDim.x * cols_per_thread;
    long long first = (long long)blockIdx.x * cols_per_block;
    long long total = (long long)B * H4;
    if (first >= total) return;
    int b = (int)(first / H4);
    for (int r = threadIdx.x; r < H; r += blockDim.x) sh[r] = h_state[(long long)b * H + r];
    __syncthreads();

    for (int u = 0; u < cols_per_thread; ++u) {
        long long i = first + (long long)u * blockDim.x + threadIdx.x;
        if (i >= total) return;
        int c = (int)(i - (long long)b * H4);
        if (c < 0 || c >= H4) continue;  // block straddles a batch boundary
        float acc = 0.0f;
        for (int r = 0; r < H; ++r) acc = fmaf(sh[r], wh[(long long)r * H4 + c], acc);
        gh[i] = acc;
    }
}

// Backward twin: dh[b, r] = sum_c dg[b, c] * Wh[r, c]  (i.e. dg · Whᵀ).
extern "C" __global__ void slstm_gate_matvec_t(
        const float* __restrict__ dg, const float* __restrict__ wh,
        float* __restrict__ dh, int H, int B) {
    int H4 = 4 * H;
    int warps = (blockDim.x * gridDim.x) >> 5;
    int wid = ((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    int lane = threadIdx.x & 31;
    for (int i = wid; i < B * H; i += warps) {
        int b = i / H, r = i - b * H;
        const float* dgrow = dg + (long long)b * H4;
        const float* wrow = wh + (long long)r * H4;
        float acc = 0.0f;
        for (int c = lane; c < H4; c += 32) acc = fmaf(dgrow[c], wrow[c], acc);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if (lane == 0) dh[i] = acc;
    }
}

extern "C" __global__ void slstm_step_fused(
        float* g, const float* gh, const float* bcat, slab_t* h_prev,
        float* c_state, float* n_state, float* m_state, float* h_state,
        slab_t* h_narrow, state_t* c_entry, state_t* n_entry, slab_t* zt, slab_t* ot,
        state_t* i_prime, state_t* f_prime, state_t* c_out, state_t* n_out,
        float* out, int t, int T, int H, int BH, int first) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= BH) return;
    int b = k / H, j = k % H;
    long long go = (long long)(b * T + t) * 4 * H + j;
    long long s = (long long)(b * T + t) * H + j;
    int ho = b * 4 * H + j; // gh row for this batch element

    float hp = first ? 0.0f : h_state[k];
    float mp = first ? 0.0f : m_state[k];
    float np = first ? 0.0f : n_state[k];
    float cp = first ? 0.0f : c_state[k];

    float z_pre = g[go]         + (first ? 0.0f : gh[ho])           + bcat[j];
    float i_pre = g[go + H]     + (first ? 0.0f : gh[ho + H])       + bcat[H + j];
    float f_pre = g[go + 2 * H] + (first ? 0.0f : gh[ho + 2 * H])   + bcat[2 * H + j];
    float o_pre = g[go + 3 * H] + (first ? 0.0f : gh[ho + 3 * H])   + bcat[3 * H + j];
    g[go + 2 * H] = f_pre; // biased forget pre-activation, saved for backward

    slab_st(h_prev, s, hp);

    float z = tanhf(z_pre);
    float o = stable_sigmoid(o_pre);
    float log_f = log_sigmoid(f_pre);
    float fm = log_f + mp;
    float m = (np == 0.0f) ? i_pre : fmaxf(fm, i_pre);
    float ip = fminf(1.0f, expf(i_pre - m));
    float fp = fminf(1.0f, expf(fm - m));
    float c = fp * cp + ip * z;
    float n = fp * np + ip;

    if (t == 0) {
        state_st(c_entry, k, cp);
        state_st(n_entry, k, np);
    }
    slab_st(zt, s, z); slab_st(ot, s, o);
    state_st(i_prime, s, ip); state_st(f_prime, s, fp);
    state_st(c_out, s, c);  state_st(n_out, s, n);
    c_state[k] = c; n_state[k] = n; m_state[k] = m;

    // h = o·c/n — the exp(−m) in c and n cancels. See `slstm_cell_step`.
    float hh = o * c / n;
    h_state[k] = hh;
    slab_st(h_narrow, k, hh);
    out[s] = hh;
}

// Backward of slstm_step_fused: the four gate deltas are written back into `g`
// (whose forward contents are dead by now), so one buffer carries the gate
// pre-activations forward and the gate deltas backward. Reads the biased forget
// pre-activation out of g's forget block before overwriting it.
extern "C" __global__ void slstm_step_fused_bwd(
        const float* d_out, float* gates, slab_t* d_gates_flat, const float* d_h_recur,
        const slab_t* o_act, const state_t* c_t, const state_t* n_t,
        const state_t* c_entry, const state_t* n_entry, const slab_t* z_act,
        const state_t* i_gate, const state_t* f_gate,
        float* d_c_recur, float* d_n_recur, int t, int T, int H, int BH, int reset) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= BH) return;
    int b = k / H, j = k % H;
    long long gate_off = (long long)(b * T + t) * 4 * H + j; // row base in [B,T,4H]
    long long s = (long long)(b * T + t) * H + j;            // row base in [B,T,H]
    int flat_off = b * 4 * H + j;                            // row base in [B,4H]

    float f_pre = gates[gate_off + 2 * H];
    // Total grad on h_t: from the layer above plus from step t+1.
    float d_h = d_out[s] + d_h_recur[k];
    float o = slab_ld(o_act, s);
    float c = state_ld(c_t, s);
    float n = state_ld(n_t, s);
    // h = o·c/n — no clamp, so dn has no branch: ∂h/∂n = −o·c/n².
    float d_o_pre = d_h * (c / n) * o * (1.0f - o);
    float d_c = d_h * o / n + d_c_recur[k];
    float d_n = d_h * o * (-c) / (n * n) + d_n_recur[k];
    float f = state_ld(f_gate, s);
    float i = state_ld(i_gate, s);
    float z = slab_ld(z_act, s);
    float c_pr = reset ? 0.0f : (t == 0) ? state_ld(c_entry, k) : state_ld(c_t, s - H);
    float n_pr = reset ? 0.0f : (t == 0) ? state_ld(n_entry, k) : state_ld(n_t, s - H);
    float d_f_gate = d_c * c_pr + d_n * n_pr;
    float d_i_gate = d_c * z + d_n;
    float d_z_act = d_c * i;

    float d_z_pre = d_z_act * (1.0f - z * z);
    float d_i_pre = d_i_gate * i;
    float d_f_pre = d_f_gate * f * (1.0f - stable_sigmoid(f_pre));

    gates[gate_off]         = d_z_pre;  slab_st(d_gates_flat, flat_off,         d_z_pre);
    gates[gate_off + H]     = d_i_pre;  slab_st(d_gates_flat, flat_off + H,     d_i_pre);
    gates[gate_off + 2 * H] = d_f_pre;  slab_st(d_gates_flat, flat_off + 2 * H, d_f_pre);
    gates[gate_off + 3 * H] = d_o_pre;  slab_st(d_gates_flat, flat_off + 3 * H, d_o_pre);

    // Carry to step t−1: both paths are scaled by the forget gate.
    d_c_recur[k] = reset ? 0.0f : d_c * f;
    d_n_recur[k] = reset ? 0.0f : d_n * f;
}
