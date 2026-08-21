// Eager fused-gate sLSTM: one launch per timestep, gate pre-activation assembled
// in-kernel from the input half, the recurrent half and the bias.
//
// Writes the same slabs as slstm_coop.cu, so both must be built at the same
// `slab_t` width.

// Fused-gate forward step. Same recurrence as slstm_cell_step, but a gate
// pre-activation is assembled here from three pieces instead of arriving as one
// tensor: the input half `g` (a [B, T, 4H] buffer holding x·Wx for every
// timestep), the recurrent half `gh` (a [B, 4H] scratch holding h_{t-1}·Wh for
// this timestep) and the bias. The saved tensors are [B, T, H] slabs indexed by
// (b·T + t)·H + j rather than one small tensor per timestep.
//
// Keeping the recurrent half in its own *contiguous* [B, 4H] scratch is what lets
// its GEMM stay one dense matmul at any batch size — accumulating straight into
// g's strided [:, t, :] rows would force a batched GEMM of one-row matrices, which
// is fine at batch 1 (the backbone) and disastrous at batch 2047 (the encoder).
//
// `g`'s forget block is overwritten in place with the *biased* forget
// pre-activation: backward needs it, and the slot is dead once this step is done.
// `h_prev` records h_{t-1} for the deferred dWh GEMM.
// Recurrent gate half for ONE timestep: gh[b, c] = sum_r h[b, r] * Wh[r, c].
//
// This replaces a cuBLAS call inside the sLSTM's per-step loop, and the reason is
// host overhead rather than device throughput. Measured at the backbone's shape
// (B=1, H=1024, so M=1 x K=1024 x N=4096): the matvec itself is ~5.6 us of GPU work,
// but the cuBLAS call costs **41 us per invocation** back-to-back, while a bare
// kernel launch is 1.7 us. Across T=1024 timesteps that overhead IS the T-loop.
//
// The kernel is bandwidth-bound: arithmetic intensity is 2*H*4H / (H*4H*4) = 0.5
// FLOP/byte, so every byte of Wh read is the cost. Two things follow, and getting
// either wrong loses more than the cuBLAS overhead saved (both were measured):
//
//   * `h` goes in SHARED MEMORY, staged once per block. It is the operand reused by
//     every column, so leaving it in global costs an extra H-length read per column.
//   * the Wh read must COALESCE. Lanes hold consecutive columns `c` and walk rows
//     together, so `wh[r*H4 + c]` at fixed r is one contiguous 128-byte transaction
//     per warp. A warp-per-column arrangement (each lane striding `r`) reads the same
//     bytes with a stride of H4 and measured 79.9 ms against cuBLAS's 44.5 ms.
//
// Wh itself cannot be staged: it is [H, 4H] = 16 MB at this width. Only `h` fits,
// which is why this is a shared-memory *broadcast* rather than a tiled GEMM.
extern "C" __global__ void slstm_gate_matvec(
        const float* __restrict__ h_state, const float* __restrict__ wh,
        float* __restrict__ gh, int H, int B, int cols_per_thread) {
    extern __shared__ float sh[];   // [H] — this block's copy of h for batch `b`
    int H4 = 4 * H;
    // One block serves a contiguous run of columns from ONE batch element, so a
    // single staged `h` row serves the whole block. Each thread takes
    // `cols_per_thread` columns, strided by blockDim so the warp stays coalesced.
    int cols_per_block = blockDim.x * cols_per_thread;
    long long first = (long long)blockIdx.x * cols_per_block;
    long long total = (long long)B * H4;
    if (first >= total) return;
    int b = (int)(first / H4);
    for (int r = threadIdx.x; r < H; r += blockDim.x) sh[r] = h_state[(long long)b * H + r];
    __syncthreads();

    // Accumulate `cols_per_thread` columns at once: `h[r]` is read from shared ONCE
    // per row and reused across them, so the inner loop is pure coalesced Wh traffic.
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
//
// Same reasoning as `slstm_gate_matvec` — M=1, so cuBLAS's per-call cost dwarfs the
// work. One warp per output ROW here, reducing over the 4H gate columns; that walk
// is contiguous in `wh`, so these reads coalesce better than the forward's do.
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
        slab_t* h_narrow, float* c_prev, float* n_prev, slab_t* zt, slab_t* ot,
        float* i_prime, float* f_prime, float* c_out, float* n_out,
        float* out, int t, int T, int H, int BH, int first) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= BH) return;
    int b = k / H, j = k % H;
    long long go = (long long)(b * T + t) * 4 * H + j;
    long long s = (long long)(b * T + t) * H + j;
    int ho = b * 4 * H + j; // gh row for this batch element

    // `first` is the start of a sequence that carries nothing in: h_{-1} is zero, so
    // `gh = h_{-1}·Wh` is zero and the whole carried state is zero. Substituting the
    // zeros here is what lets the host skip both that GEMM and the state's memset —
    // exactly equivalent, since the values it would have read are these.
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
    // See `slstm_cell_step`: n == 0 is the first step of a sequence, where m must be
    // ĩ so that i' is exactly 1 and h = c/n cannot become 0/0.
    float m = (np == 0.0f) ? i_pre : fmaxf(fm, i_pre);
    // Both gates are <= 1 by construction (m >= i_pre and m >= fm), so the clamp
    // is a no-op in exact arithmetic; it only catches a rounded exp() landing a
    // hair above 1. In the n == 0 branch fp is unbounded, but it multiplies
    // np == 0 here and its BPTT carry is discarded at a sequence start, so
    // clamping it there is equally invisible. NOTE: `n` itself must NOT be
    // clamped -- it is the stabilized normalizer and exp(-m) cancels in c/n.
    float ip = fminf(1.0f, expf(i_pre - m));
    float fp = fminf(1.0f, expf(fm - m));
    float c = fp * cp + ip * z;
    float n = fp * np + ip;

    c_prev[s] = cp;
    n_prev[s] = np;
    slab_st(zt, s, z); slab_st(ot, s, o);
    i_prime[s] = ip; f_prime[s] = fp;
    c_out[s] = c; n_out[s] = n;
    c_state[k] = c; n_state[k] = n; m_state[k] = m;

    // h = o·c/n — the exp(−m) in c and n cancels. See `slstm_cell_step`.
    float hh = o * c / n;
    h_state[k] = hh;
    // The narrowed twin the next step's `h·Wh` GEMM reads. Writing it here is what
    // keeps that GEMM's operand bf16 without a cast launch inside the loop.
    slab_st(h_narrow, k, hh);
    out[s] = hh;
}

// Backward of slstm_step_fused: the four gate deltas are written back into `g`
// (whose forward contents are dead by now), so one buffer carries the gate
// pre-activations forward and the gate deltas backward. Reads the biased forget
// pre-activation out of g's forget block before overwriting it.
//
// The deltas are also written to the contiguous [B, 4H] scratch `dgh`, the mirror
// of the forward's `gh`: this timestep's dh = dgh·Whᵀ is the one thing BPTT cannot
// defer, and going through the scratch keeps that a dense GEMM at any batch size.
// Parameters (all device pointers; one thread per (batch, hidden-unit) pair):
//
//   d_out        [B, T, H]   in   incoming grad of this layer's output h_t
//   gates        [B, T, 4H]  both in: biased forget pre-activation (block 2);
//                                 out: the four gate deltas (z, i, f, o)
//   d_gates_flat [B, 4H]     out  this step's gate deltas, contiguous, for the
//                                 dh = d_gates·Whᵀ GEMM (mirror of forward `gh`)
//   d_h_recur    [B, H]      in   grad arriving from step t+1 through h
//   o_act        [B, T, H]   in   saved sigmoid(o_pre)
//   c_t, n_t     [B, T, H]   in   saved cell / normalizer AFTER this step
//   c_prev,n_prev[B, T, H]   in   saved cell / normalizer BEFORE this step
//   z_act        [B, T, H]   in   saved tanh(z_pre)
//   i_gate,f_gate[B, T, H]   in   saved stabilized exp() gate values i', f'
//   d_c_recur    [B, H]      both cell grad carried backward across steps
//   d_n_recur    [B, H]      both normalizer grad carried backward across steps
//   t, T, H, BH              in   current step, sequence length, width, B*H
extern "C" __global__ void slstm_step_fused_bwd(
        const float* d_out, float* gates, float* d_gates_flat, const float* d_h_recur,
        const slab_t* o_act, const float* c_t, const float* n_t,
        const float* c_prev, const float* n_prev, const slab_t* z_act,
        const float* i_gate, const float* f_gate,
        float* d_c_recur, float* d_n_recur, int t, int T, int H, int BH) {
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
    float c = c_t[s];
    float n = n_t[s];
    // h = o·c/n — no clamp, so dn has no branch: ∂h/∂n = −o·c/n².
    float d_o_pre = d_h * (c / n) * o * (1.0f - o);
    float d_c = d_h * o / n + d_c_recur[k];
    float d_n = d_h * o * (-c) / (n * n) + d_n_recur[k];
    float f = f_gate[s];
    float i = i_gate[s];
    float z = slab_ld(z_act, s);
    // Grads w.r.t. the stabilized gate values i', f', before their exp/sigmoid.
    float d_f_gate = d_c * c_prev[s] + d_n * n_prev[s];
    float d_i_gate = d_c * z + d_n;
    float d_z_act = d_c * i;

    // Through the activations: tanh' = 1−z², exp' = the gate itself,
    // and f' = exp(log σ(f_pre) + …) contributes σ'(f_pre)/σ(f_pre) = 1−σ(f_pre).
    float d_z_pre = d_z_act * (1.0f - z * z);
    float d_i_pre = d_i_gate * i;
    float d_f_pre = d_f_gate * f * (1.0f - stable_sigmoid(f_pre));

    gates[gate_off]         = d_z_pre;  d_gates_flat[flat_off]         = d_z_pre;
    gates[gate_off + H]     = d_i_pre;  d_gates_flat[flat_off + H]     = d_i_pre;
    gates[gate_off + 2 * H] = d_f_pre;  d_gates_flat[flat_off + 2 * H] = d_f_pre;
    gates[gate_off + 3 * H] = d_o_pre;  d_gates_flat[flat_off + 3 * H] = d_o_pre;

    // Carry to step t−1: both paths are scaled by the forget gate.
    d_c_recur[k] = d_c * f;
    d_n_recur[k] = d_n * f;
}
