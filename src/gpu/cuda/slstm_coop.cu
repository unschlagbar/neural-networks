// Cooperative-launch sLSTM: the whole time loop in one grid-synchronising kernel.
// Behind COOP_KERNELS; drags in libcu++ via <cooperative_groups.h>, so it has the
// largest include requirement of any module. FUSED_BF16 selects bf16 gate staging,
// SLSTM_H / SLSTM_B are the shape-specialisation constants (see Kernels::specialized).

#ifdef COOP_KERNELS
#include <cooperative_groups.h>
#ifdef FUSED_BF16
#include <cuda_bf16.h>
#endif
namespace cg = cooperative_groups;

// Time-fused sLSTM forward: the ENTIRE T-loop in ONE launch (FlashRNN's
// `cuda_fused` idea, arXiv 2412.07752). At B=1 the per-step path is a GEMV plus a
// pointwise kernel, i.e. two launches whose combined ~6.6us is almost all launch
// latency and Wh re-reads -- the math itself is trivial. Fusing time removes both:
//
//   * Wh is loaded ONCE into shared memory, before the loop, instead of streaming
//     from HBM at every timestep. Each block owns a contiguous slice of Wh's 4H
//     output columns (`cols_per_block`), so the whole matrix lives on-chip across
//     the grid and no block needs more than its slice.
//   * the two per-step launches become one `grid.sync()` (measured ~0.74us here,
//     against ~6.6us for the launch pair).
//
// Cross-block consistency is the reason this must be a *cooperative* launch: block
// k computes gate columns for the whole batch but needs the FULL h_t vector from
// the previous step, which other blocks produced. `grid.sync()` after the state
// update is what makes h_t visible everywhere before step t+1 reads it.
//
// Shape contract: gridDim.x * cols_per_block >= 4H, blockDim.x threads per block,
// B*H <= blockDim.x * gridDim.x for the state update. `wh` is [H, 4H] row-major.
// Under FUSED_BF16 the staged operands are bf16 and the accumulators fp32; above
// H=640 that is what keeps the grid co-resident at all (see `fused_bf16_enabled`).
// SPECIALIZATION (FlashRNN's `-DFLASHRNN_HIDDEN_SIZE=... -DFLASHRNN_BATCH_SIZE=...`
// in `flashrnn.py`): when SLSTM_H / SLSTM_B are defined, H and B become compile-time
// constants and the runtime arguments are ignored. That lets the compiler give the
// reduction a static trip count, turn the /32 and %32 into shifts, and — at B == 1,
// which is what the backbone always runs — delete the batch indexing entirely.
//
// Sequence length is deliberately NOT specialized, exactly as upstream: `steps` is a
// runtime argument to their `Run()` too. T is the outer loop's bound, walked once
// with a data dependency between iterations, so there is nothing to unroll or size
// by it — and here windows never cross a document border, so T varies per window and
// specializing on it would mean an NVRTC compile per window length.
#ifdef SLSTM_H
#define FUSED_H SLSTM_H
#else
#define FUSED_H H
#endif
#ifdef SLSTM_B
#define FUSED_B SLSTM_B
#else
#define FUSED_B B
#endif

extern "C" __global__ void slstm_fused_time(
        const float* __restrict__ wh, float* g, const float* __restrict__ bcat,
        slab_t* h_prev, float* c_state, float* n_state, float* m_state, float* h_state,
        float* c_prev, float* n_prev, slab_t* zt, slab_t* ot,
        float* i_prime, float* f_prime, float* c_out, float* n_out,
        float* out, int T, int H_rt, int B_rt, int cols_per_block) {
    extern __shared__ float smem[];
    cg::grid_group grid = cg::this_grid();

#ifdef SLSTM_H
    (void)H_rt;
#else
    const int H = H_rt;
#endif
#ifdef SLSTM_B
    (void)B_rt;
#else
    const int B = B_rt;
#endif

    // Each block owns a contiguous range of HIDDEN UNITS [j0, j0+nj), and for each
    // it owns all four gate columns j, H+j, 2H+j, 3H+j. That pairing is what lets
    // the pointwise phase below read gate values this same block just wrote, so
    // only ONE grid.sync() per step is needed (for h_t) instead of two: `cols_per_block`
    // counts hidden units here, not raw Wh columns.
    const int H4 = 4 * FUSED_H;
    const int j0 = blockIdx.x * cols_per_block;
    const int nj = max(0, min(cols_per_block, FUSED_H - j0));
    const int ncol = 4 * nj;                            // Wh columns staged by this block

    // Shared layout. With FUSED_BF16 the Wh slice is bf16 (half the footprint, so a
    // block can own twice the units at the same shared budget) and is followed by a
    // bf16 mirror of h_state; the fp32 gate scratch comes last and stays fp32
    // because it is what the recurrence consumes.
    //
    // This is FlashRNN's split (`R_shared` is FLASHRNN_DTYPE_R = bf16 while
    // ACC_DTYPE stays float): bf16 is the STORAGE dtype for the operands that get
    // re-read every timestep, fp32 is the ARITHMETIC dtype. Their kernel converts
    // too — `type2float` on the way into the accumulator, `float2type` on the way
    // out — but both sides are already in registers or shared memory, so a
    // conversion is a couple of ALU ops. The earlier standalone `SLSTM_BF16` path
    // lost because it converted through a separate kernel over HBM, paying a whole
    // launch and a round-trip per timestep to save a read it never got to reuse.
#ifdef FUSED_BF16
    __nv_bfloat16* wsh = (__nv_bfloat16*)smem;
    __nv_bfloat16* hsh = wsh + (long long)FUSED_H * ncol;
    float* gacc = (float*)(hsh + (long long)FUSED_B * FUSED_H);
#else
    float* wsh = smem;
    float* gacc = smem + (long long)FUSED_H * ncol;
#endif

    // Stage this block's Wh slice into shared memory, once for all T, TRANSPOSED:
    // wsh[c * H + r] = wh[r * H4 + col0 + c], i.e. column-major so that one
    // column is contiguous. The reduction below has the 32 lanes of a warp stride
    // consecutive `r` for a fixed column; storing row-major would make that a
    // stride-`ncol` walk (a bank conflict on every access), while this makes it a
    // fully coalesced read of consecutive elements.
    // Local column c in [0, 4*nj) maps to gate g = c / nj and unit j = j0 + c % nj,
    // i.e. global Wh column g * H + j.
    if (ncol > 0) {
        for (int i = threadIdx.x; i < FUSED_H * ncol; i += blockDim.x) {
            int r = i / ncol, c = i - r * ncol;
            int gidx = c / nj, j = j0 + (c - gidx * nj);
            float v = wh[(long long)r * H4 + gidx * FUSED_H + j];
#ifdef FUSED_BF16
            wsh[(long long)c * FUSED_H + r] = __float2bfloat16(v);
#else
            wsh[(long long)c * FUSED_H + r] = v;
#endif
        }
    }
    grid.sync(); // Wh staged everywhere AND initial h_state visible to all blocks

#ifdef FUSED_BF16
    // Seed the mirror with h_0 (the loop refreshes it at the end of every step).
    for (int i = threadIdx.x; i < FUSED_B * FUSED_H; i += blockDim.x) {
        hsh[i] = __float2bfloat16(h_state[i]);
    }
    __syncthreads();
#endif

    for (int t = 0; t < T; ++t) {
        // --- recurrent half: gh[b, col] = sum_r h_state[b, r] * Wh[r, col] ---
        // ONE WARP PER COLUMN, reducing over the H rows. Assigning one *thread* per
        // column instead would leave almost every thread idle at the shape that
        // matters: the backbone runs B=1 with ~49 columns per block, so a
        // thread-per-column loop fills 49 of 256 lanes and then makes each of them
        // walk all H rows serially. Here all 256 threads work on 8 columns at a
        // time, each lane striding the reduction, finished by a warp shuffle.
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        const int warps = blockDim.x >> 5;
        for (int i = warp; i < FUSED_B * ncol; i += warps) {
            int b = i / ncol, c = i - b * ncol;
            float acc = 0.0f;
#ifdef FUSED_BF16
            // Both operands bf16, accumulator fp32 — FlashRNN's ACC_DTYPE split.
            const __nv_bfloat16* hrow = hsh + b * FUSED_H;
            const __nv_bfloat16* wcol = wsh + (long long)c * FUSED_H;
            for (int r = lane; r < FUSED_H; r += 32)
                acc = fmaf(__bfloat162float(hrow[r]), __bfloat162float(wcol[r]), acc);
#else
            const float* hrow = h_state + b * FUSED_H;
            const float* wcol = wsh + (long long)c * FUSED_H; // column c, contiguous
            for (int r = lane; r < FUSED_H; r += 32) acc = fmaf(hrow[r], wcol[r], acc);
#endif
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
            // Land in the block-local staging area right after the Wh slice: the
            // pointwise phase below consumes these WITHOUT a grid-wide sync, because
            // this block owns every gate column of every unit it is about to update.
            if (lane == 0) gacc[i] = acc;
        }
        __syncthreads(); // block-local only: gacc written before this block reads it

        // --- pointwise recurrence: one thread per (batch, hidden unit) ---
        // Identical math to slstm_step_fused; see it for the stabilizer notes.
        for (int i = threadIdx.x; i < FUSED_B * nj; i += blockDim.x) {
            int b = i / nj, jl = i - b * nj;   // jl = local unit index
            int j = j0 + jl;                   // global hidden unit
            int k = b * FUSED_H + j;
            long long go = ((long long)b * T + t) * H4 + j;
            long long s = ((long long)b * T + t) * FUSED_H + j;

            // Gate pre-activation = input half (already in g) + recurrent half
            // (this block's gacc) + bias. gacc is laid out [b][gate][unit].
            const float* ga = gacc + (long long)b * ncol + jl;
            float z_pre = g[go]         + ga[0]          + bcat[j];
            float i_pre = g[go + FUSED_H]     + ga[nj]         + bcat[FUSED_H + j];
            float f_pre = g[go + 2 * FUSED_H] + ga[2 * nj]     + bcat[2 * FUSED_H + j];
            float o_pre = g[go + 3 * FUSED_H] + ga[3 * nj]     + bcat[3 * FUSED_H + j];
            g[go + 2 * FUSED_H] = f_pre; // biased forget pre-activation, saved for backward

            slab_st(h_prev, s, h_state[k]);

            float z = tanhf(z_pre);
            float o = stable_sigmoid(o_pre);
            float log_f = log_sigmoid(f_pre);
            float fm = log_f + m_state[k];
            float np = n_state[k];
            float m = (np == 0.0f) ? i_pre : fmaxf(fm, i_pre);
            float ip = fminf(1.0f, expf(i_pre - m));
            float fp = fminf(1.0f, expf(fm - m));
            float cp = c_state[k];
            float c = fp * cp + ip * z;
            float n = fp * np + ip;

            c_prev[s] = cp;
            n_prev[s] = np;
            slab_st(zt, s, z); slab_st(ot, s, o);
            i_prime[s] = ip; f_prime[s] = fp;
            c_out[s] = c; n_out[s] = n;
            c_state[k] = c; n_state[k] = n; m_state[k] = m;

            float hh = o * c / n;
            h_state[k] = hh;
            out[s] = hh;
        }
        grid.sync(); // h_t complete before step t+1's GEMV reads it

#ifdef FUSED_BF16
        // Refresh this block's bf16 mirror of the FULL h vector. It cannot be
        // filled in-register above: a block computes h only for the units it owns,
        // while the reduction contracts over all H of them, so every block needs
        // every other block's h — which is exactly what the grid.sync() above just
        // made visible. This is a shared-memory write of B*H bf16 values per step,
        // not a global round-trip, and no extra launch.
        for (int i = threadIdx.x; i < FUSED_B * FUSED_H; i += blockDim.x) {
            hsh[i] = __float2bfloat16(h_state[i]);
        }
        __syncthreads(); // mirror complete before this block's next reduction
#endif
    }
}
// Time-fused sLSTM BACKWARD: the whole reverse T-loop in one cooperative launch,
// the mirror of slstm_fused_time.
//
// The ownership is transposed relative to the forward. Forward, a block owning
// hidden units [j0, j0+nj) needed those units' gate COLUMNS of Wh; backward, the
// contraction is dh[r] = sum_c dgates[c] * Wh[r, c], so a block producing dh for
// units [j0, j0+nj) needs those ROWS of Wh -- all 4H entries of each. That is what
// this kernel stages: `wh_rows[nj][4H]`, read once before the loop.
//
// Unlike the forward, TWO grid syncs per step are structural and cannot be merged
// away. The forward got to one because a unit's pointwise update needed only that
// unit's own gates; here the pointwise update at step t needs dh_recur[j], which
// is a full contraction over every OTHER unit's gate deltas. So: pointwise (needs
// dh_recur from t+1) -> sync -> contraction (needs all gate deltas) -> sync. At
// T=2047 the pair costs ~2.3us*T total, against ~19.8ms of per-layer backward, so
// it is well worth paying to delete the per-step launches.
//
// `dgates_all` is a [B, 4H] grid-visible scratch: the pointwise phase writes this
// step's deltas there for the contraction phase to read across blocks. The kernel
// also writes them into `g` (whose forward contents are dead), exactly as the
// per-step path does, so the post-loop dWx/dWh/db GEMMs are unchanged.
extern "C" __global__ void slstm_fused_time_bwd(
        const float* __restrict__ wh, const float* __restrict__ dy, float* g,
        float* dgates_all, float* dh_recur, float* dc_recur, float* dn_recur,
        const slab_t* __restrict__ ot, const float* __restrict__ c_t,
        const float* __restrict__ n_t, const float* __restrict__ c_prev,
        const float* __restrict__ n_prev, const slab_t* __restrict__ zt,
        const float* __restrict__ i_gate, const float* __restrict__ f_gate,
        int T, int H_rt, int B_rt, int units_per_block) {
    extern __shared__ float smem[];
    cg::grid_group grid = cg::this_grid();

#ifdef SLSTM_H
    (void)H_rt;
#else
    const int H = H_rt;
#endif
#ifdef SLSTM_B
    (void)B_rt;
#else
    const int B = B_rt;
#endif

    const int H4 = 4 * FUSED_H;
    const int j0 = blockIdx.x * units_per_block;
    const int nj = max(0, min(units_per_block, FUSED_H - j0));

    // Stage this block's Wh rows: wsh[u * H4 + c] = wh[(j0 + u) * H4 + c].
    // Row-major keeps the contraction's lane stride contiguous in c.
    //
    // Under FUSED_BF16 the staged rows are bf16, halving a block's footprint so it
    // can own twice the units. That is what decides whether this path exists at all
    // above H=640: a row costs 4H, so at fp32 the grid needs more blocks than the
    // device has SMs and the cooperative launch is declined. Storage is bf16, the
    // accumulator below stays fp32 — the same split as the forward.
#ifdef FUSED_BF16
    __nv_bfloat16* wsh = (__nv_bfloat16*)smem;
    float* dgsh = (float*)(wsh + (long long)nj * H4);
#else
    float* wsh = smem;
    float* dgsh = wsh + (long long)nj * H4;
#endif
    for (int i = threadIdx.x; i < nj * H4; i += blockDim.x) {
        int u = i / H4, c = i - u * H4;
        float v = wh[(long long)(j0 + u) * H4 + c];
#ifdef FUSED_BF16
        wsh[i] = __float2bfloat16(v);
#else
        wsh[i] = v;
#endif
    }
    grid.sync();

    for (int t = T - 1; t >= 0; --t) {
        // --- pointwise: one thread per (batch, owned unit) ---
        // Identical math to slstm_step_fused_bwd; see it for the derivation.
        for (int i = threadIdx.x; i < FUSED_B * nj; i += blockDim.x) {
            int b = i / nj, u = i - b * nj;
            int j = j0 + u;
            int k = b * FUSED_H + j;
            long long go = ((long long)b * T + t) * H4 + j;
            long long s = ((long long)b * T + t) * FUSED_H + j;

            float f_pre = g[go + 2 * FUSED_H];
            float d_h = dy[s] + dh_recur[k];
            float o = slab_ld(ot, s);
            float c = c_t[s];
            float n = n_t[s];
            float d_o_pre = d_h * (c / n) * o * (1.0f - o);
            float d_c = d_h * o / n + dc_recur[k];
            float d_n = d_h * o * (-c) / (n * n) + dn_recur[k];
            float f = f_gate[s];
            float ig = i_gate[s];
            float z = slab_ld(zt, s);
            float d_f_gate = d_c * c_prev[s] + d_n * n_prev[s];
            float d_i_gate = d_c * z + d_n;
            float d_z_act = d_c * ig;

            float d_z_pre = d_z_act * (1.0f - z * z);
            float d_i_pre = d_i_gate * ig;
            float d_f_pre = d_f_gate * f * (1.0f - stable_sigmoid(f_pre));

            g[go]          = d_z_pre;
            g[go + FUSED_H]      = d_i_pre;
            g[go + 2 * FUSED_H]  = d_f_pre;
            g[go + 3 * FUSED_H]  = d_o_pre;

            // Grid-visible copy for the contraction below.
            long long fo = (long long)b * H4 + j;
            dgates_all[fo]          = d_z_pre;
            dgates_all[fo + FUSED_H]      = d_i_pre;
            dgates_all[fo + 2 * FUSED_H]  = d_f_pre;
            dgates_all[fo + 3 * FUSED_H]  = d_o_pre;

            dc_recur[k] = d_c * f;
            dn_recur[k] = d_n * f;
        }
        grid.sync(); // all gate deltas visible before any block contracts them

        // Stage the whole [B, 4H] delta vector into shared memory once. Every warp
        // below reduces against it, and without this each would re-read it from
        // global — the same 4H floats fetched by all `gridDim.x` blocks on the
        // critical path of every timestep.
        for (int i = threadIdx.x; i < FUSED_B * H4; i += blockDim.x) {
            dgsh[i] = dgates_all[i];
        }
        __syncthreads();

        // --- contraction: dh_recur[b, j] = sum_c dgates_all[b, c] * Wh[j, c] ---
        // One warp per (batch, owned unit), lanes striding the 4H reduction. The
        // staged row is contiguous in c, so the lane reads are coalesced.
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        const int warps = blockDim.x >> 5;
        for (int i = warp; i < FUSED_B * nj; i += warps) {
            int b = i / nj, u = i - b * nj;
            const float* dg = dgsh + (long long)b * H4;
            float acc = 0.0f;
#ifdef FUSED_BF16
            const __nv_bfloat16* wrow = wsh + (long long)u * H4;
            for (int c = lane; c < H4; c += 32)
                acc = fmaf(dg[c], __bfloat162float(wrow[c]), acc);
#else
            const float* wrow = wsh + (long long)u * H4;
            for (int c = lane; c < H4; c += 32) acc = fmaf(dg[c], wrow[c], acc);
#endif
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
            if (lane == 0) dh_recur[b * FUSED_H + j0 + u] = acc;
        }
        grid.sync(); // dh_recur complete before step t-1's pointwise reads it
    }
}
#endif // COOP_KERNELS
