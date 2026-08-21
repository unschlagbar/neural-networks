// Cooperative-launch sLSTM: the whole time loop in one grid-synchronising kernel.
// Behind COOP_KERNELS; drags in libcu++ via <cooperative_groups.h>, so it has the
// largest include requirement of any module. FUSED_BF16 selects bf16 gate staging
// for the backward, SLSTM_H / SLSTM_B are the shape-specialisation constants (see
// Kernels::specialized).

#ifdef COOP_KERNELS
#include <cooperative_groups.h>
#include <cuda_bf16.h>
namespace cg = cooperative_groups;

// Eight bf16 in one 128-bit shared/global access — the widest either supports.
union bf16x8 {
    int4 raw;
    __nv_bfloat162 pair[4];
};

// Time-fused sLSTM forward: the ENTIRE T-loop in ONE launch (FlashRNN's
// `cuda_fused` idea, arXiv 2412.07752). At B=1 the per-step path is a GEMV plus a
// pointwise kernel, i.e. two launches whose combined ~6.6us is almost all launch
// latency and Wh re-reads -- the math itself is trivial. Fusing time removes both:
//
//   * Wh is loaded ONCE into shared memory, before the loop, instead of streaming
//     from HBM at every timestep. Each block owns a contiguous slice of Wh's 4H
//     output columns (`units_per_block`), so the whole matrix lives on-chip across
//     the grid and no block needs more than its slice.
//   * the two per-step launches become one `grid.sync()` (measured ~0.9us here,
//     against ~6.6us for the launch pair).
//
// Cross-block consistency is the reason this must be a *cooperative* launch: block
// k computes gate columns for the whole batch but needs the FULL h_t vector from
// the previous step, which other blocks produced. `grid.sync()` after the state
// update is what makes h_t visible everywhere before step t+1 reads it.
//
// Precision follows FlashRNN's split: bf16 is the STORAGE dtype for everything the
// loop re-reads (the staged `Wh` slice, the `h` mirror), fp32 is the ARITHMETIC
// dtype — every accumulator and the whole stabilizer group stay wide. A conversion
// costs a couple of ALU ops because both sides are already in registers or shared
// memory; the earlier standalone bf16 path lost only because it converted through a
// separate kernel over HBM.
//
// SPECIALIZATION (FlashRNN's `-DFLASHRNN_HIDDEN_SIZE=... -DFLASHRNN_BATCH_SIZE=...`
// in `flashrnn.py`): H and B are compile-time constants here, which is what lets the
// reduction below hold its slice of `h` in a REGISTER ARRAY. There is deliberately no
// generic build of this kernel — a runtime H would put that array in local memory and
// give back the whole win — so `ops::slstm_fused_time` declines when the specialized
// build is unavailable and the caller takes the per-step loop.
//
// Sequence length is NOT specialized, exactly as upstream: `steps` is a runtime
// argument to their `Run()` too. T is the outer loop's bound, walked once with a data
// dependency between iterations, so there is nothing to unroll or size by it — and
// here windows never cross a document border, so T varies per window and specializing
// on it would mean an NVRTC compile per window length.
#if defined(SLSTM_H) && defined(SLSTM_B)

#define FUSED_H SLSTM_H
#define FUSED_B SLSTM_B

// Rows of one Wh column a lane pulls per 128-bit access, and the rows a whole warp
// covers per pass. The staged slice is padded to a multiple of FUSED_RPP and the tail
// zero-filled, so the reduction has a static trip count and no tail branch — the
// padding contributes exact zeros.
#define FUSED_RPL 8
#define FUSED_RPP (32 * FUSED_RPL)
#define FUSED_HP (((FUSED_H) + FUSED_RPP - 1) / FUSED_RPP * FUSED_RPP)
#define FUSED_PASSES (FUSED_HP / FUSED_RPP)
#define FUSED_HSLICE (FUSED_HP / 32)

// Load a lane's slice of `src` (bf16, FUSED_HP-strided) into `dst` fp32 registers.
__device__ __forceinline__ void fused_load_slice(float* dst, const __nv_bfloat16* src,
                                                 int lane) {
    #pragma unroll
    for (int p = 0; p < FUSED_PASSES; ++p) {
        bf16x8 v;
        v.raw = *(const int4*)(src + p * FUSED_RPP + lane * FUSED_RPL);
        #pragma unroll
        for (int q = 0; q < FUSED_RPL / 2; ++q) {
            float2 f = __bfloat1622float2(v.pair[q]);
            dst[p * FUSED_RPL + 2 * q] = f.x;
            dst[p * FUSED_RPL + 2 * q + 1] = f.y;
        }
    }
}

extern "C" __global__ void slstm_fused_time(
        const float* __restrict__ wh, float* g, const float* __restrict__ bcat,
        slab_t* h_prev, float* c_state, float* n_state, float* m_state, float* h_state,
        float* hmir, float* c_prev, float* n_prev, slab_t* zt, slab_t* ot,
        float* i_prime, float* f_prime, float* c_out, float* n_out,
        float* out, int T, int units_per_block, int carry) {
    extern __shared__ __align__(16) float smem[];
    cg::grid_group grid = cg::this_grid();

    // Each block owns a contiguous range of HIDDEN UNITS [j0, j0+nj), and for each
    // it owns all four gate columns j, H+j, 2H+j, 3H+j. That pairing is what lets
    // the pointwise phase below read gate values this same block just wrote, so only
    // ONE grid.sync() per step is needed (for h_t) instead of two.
    const int H4 = 4 * FUSED_H;
    const int j0 = blockIdx.x * units_per_block;
    const int nj = min(units_per_block, FUSED_H - j0);
    const int ncol = 4 * nj;                       // Wh columns staged by this block

    // Shared: this block's Wh columns, then the fp32 gate scratch the recurrence
    // consumes. The gate scratch stays wide — it is an accumulator.
    __nv_bfloat16* wsh = (__nv_bfloat16*)smem;
    float* gacc = (float*)(wsh + (long long)ncol * FUSED_HP);

    // Stage the Wh slice, once for all T, TRANSPOSED: wsh[c * HP + r], i.e. column-
    // major so one column is contiguous and a lane's 8 rows are one 128-bit access.
    // Local column c in [0, 4*nj) maps to gate g = c / nj and unit j = j0 + c % nj,
    // i.e. global Wh column g * H + j. The traversal is row-major so the strided
    // global read still coalesces across `c`.
    const __nv_bfloat16 bzero = __float2bfloat16(0.0f);
    for (int i = threadIdx.x; i < FUSED_HP * ncol; i += blockDim.x) {
        int r = i / ncol, c = i - r * ncol;
        int gidx = c / nj, j = j0 + (c - gidx * nj);
        wsh[(long long)c * FUSED_HP + r] =
            (r < FUSED_H) ? __float2bfloat16(wh[(long long)r * H4 + gidx * FUSED_H + j])
                          : bzero;
    }

    // The bf16 mirror of h the reduction reads: TWO [B, HP] planes packed into the
    // caller's fp32 [B, HP] buffer (two bf16 to a float). Two, because a block writes
    // its own units of step t while another may still be reading step t-1 — a
    // cross-block write-after-read that would otherwise cost a second barrier per
    // step. Writing into the other plane removes the hazard itself.
    __nv_bfloat16* hb_cur = (__nv_bfloat16*)hmir;
    __nv_bfloat16* hb_nxt = hb_cur + (long long)FUSED_B * FUSED_HP;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < FUSED_B * FUSED_HP;
         i += blockDim.x * gridDim.x) {
        int b = i / FUSED_HP, r = i - b * FUSED_HP;
        // Both planes' row padding must be a real zero: it multiplies Wh padding that
        // is also zero, and an uninitialised NaN there would poison the sum. `carry`
        // off means this call starts a sequence, so h_{-1} is zero by definition and
        // `h_state` holds whatever the previous call left — the host no longer zeroes
        // it, this does.
        hb_cur[i] = (carry && r < FUSED_H) ? __float2bfloat16(h_state[b * FUSED_H + r]) : bzero;
        hb_nxt[i] = bzero;
    }
    grid.sync(); // Wh staged and the h mirror complete everywhere

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps = blockDim.x >> 5;

    // Pointwise ownership is FIXED for the whole loop: thread `p` owns exactly one
    // (batch row, hidden unit), which `geometry` guarantees by keeping the block at
    // least `B * units_per_block` wide. That is what lets the recurrent state and the
    // four biases live in REGISTERS — they touch global memory once at each end of
    // the loop instead of four loads and four stores per timestep.
    const int pw = threadIdx.x;
    const bool owner = pw < FUSED_B * nj;
    const int b_pw = owner ? pw / nj : 0;
    const int jl_pw = pw - b_pw * nj;
    const int j_pw = j0 + jl_pw;
    const int k_pw = b_pw * FUSED_H + j_pw;
    float c_reg = 0.0f, n_reg = 0.0f, m_reg = 0.0f, h_reg = 0.0f;
    float bz = 0.0f, bi = 0.0f, bf = 0.0f, bo = 0.0f;
    // Same as the mirror above: without a carry the sequence starts from zero, so
    // there is nothing to read and nothing for the host to have zeroed.
    if (owner && carry) {
        c_reg = c_state[k_pw];
        n_reg = n_state[k_pw];
        m_reg = m_state[k_pw];
        h_reg = h_state[k_pw];
    }
    if (owner) {
        bz = bcat[j_pw];
        bi = bcat[FUSED_H + j_pw];
        bf = bcat[2 * FUSED_H + j_pw];
        bo = bcat[3 * FUSED_H + j_pw];
    }

    for (int t = 0; t < T; ++t) {
        // This step's input half, issued BEFORE the reduction: it depends on `t`
        // alone, so its global latency lands under the reduction instead of in front
        // of the pointwise phase, where only the owning threads are awake to hide it.
        long long go = ((long long)b_pw * T + t) * H4 + j_pw;
        long long s = ((long long)b_pw * T + t) * FUSED_H + j_pw;
        float gz = 0.0f, gi = 0.0f, gf = 0.0f, go4 = 0.0f;
        if (owner) {
            gz = g[go];
            gi = g[go + FUSED_H];
            gf = g[go + 2 * FUSED_H];
            go4 = g[go + 3 * FUSED_H];
        }

        // --- recurrent half: gacc[b, col] = sum_r h[b, r] * Wh[r, col] ---
        // ONE WARP PER COLUMN, lanes striding the H reduction. A thread per column
        // instead would leave almost every lane idle at the shape that matters (B=1,
        // ~40 columns per block) and then walk all H rows serially in each.
        //
        // `h` is pulled into REGISTERS once per batch row and reused by every column
        // the warp takes. Every warp needs the whole h vector, so re-reading it per
        // column made h roughly as much shared traffic as Wh itself — and Wh is the
        // operand that cannot be reused at all.
        for (int b = 0; b < FUSED_B; ++b) {
            float hr[FUSED_HSLICE];
            fused_load_slice(hr, hb_cur + (long long)b * FUSED_HP, lane);
            for (int c = warp; c < ncol; c += warps) {
                const __nv_bfloat16* wcol = wsh + (long long)c * FUSED_HP + lane * FUSED_RPL;
                float acc = 0.0f;
                #pragma unroll
                for (int p = 0; p < FUSED_PASSES; ++p) {
                    bf16x8 v;
                    v.raw = *(const int4*)(wcol + p * FUSED_RPP);
                    #pragma unroll
                    for (int q = 0; q < FUSED_RPL / 2; ++q) {
                        float2 f = __bfloat1622float2(v.pair[q]);
                        acc = fmaf(hr[p * FUSED_RPL + 2 * q], f.x, acc);
                        acc = fmaf(hr[p * FUSED_RPL + 2 * q + 1], f.y, acc);
                    }
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
                // Block-local staging: the pointwise phase below consumes these
                // WITHOUT a grid-wide sync, because this block owns every gate column
                // of every unit it is about to update.
                if (lane == 0) gacc[b * ncol + c] = acc;
            }
        }
        __syncthreads(); // gacc written before this block reads it

        // --- pointwise recurrence ---
        // Identical math to slstm_step_fused; see it for the stabilizer notes.
        if (owner) {
            // Gate pre-activation = input half + recurrent half (this block's gacc)
            // + bias. gacc is laid out [b][gate][unit].
            const float* ga = gacc + (long long)b_pw * ncol + jl_pw;
            float z_pre = gz + ga[0] + bz;
            float i_pre = gi + ga[nj] + bi;
            float f_pre = gf + ga[2 * nj] + bf;
            float o_pre = go4 + ga[3 * nj] + bo;
            g[go + 2 * FUSED_H] = f_pre; // biased forget pre-activation, for backward

            slab_st(h_prev, s, h_reg);

            float z = tanhf(z_pre);
            float o = stable_sigmoid(o_pre);
            float fm = log_sigmoid(f_pre) + m_reg;
            float m = (n_reg == 0.0f) ? i_pre : fmaxf(fm, i_pre);
            float ip = fminf(1.0f, expf(i_pre - m));
            float fp = fminf(1.0f, expf(fm - m));
            float c = fp * c_reg + ip * z;
            float n = fp * n_reg + ip;

            c_prev[s] = c_reg;
            n_prev[s] = n_reg;
            slab_st(zt, s, z); slab_st(ot, s, o);
            i_prime[s] = ip; f_prime[s] = fp;
            c_out[s] = c; n_out[s] = n;
            c_reg = c; n_reg = n; m_reg = m;

            h_reg = o * c / n;
            hb_nxt[(long long)b_pw * FUSED_HP + j_pw] = __float2bfloat16(h_reg);
            out[s] = h_reg;
        }
        grid.sync(); // h_t complete before step t+1's reduction reads the mirror

        __nv_bfloat16* tmp = hb_cur; hb_cur = hb_nxt; hb_nxt = tmp;
    }
    // The state the next call carries in, written once rather than every timestep.
    if (owner) {
        c_state[k_pw] = c_reg;
        n_state[k_pw] = n_reg;
        m_state[k_pw] = m_reg;
        h_state[k_pw] = h_reg;
    }
}

#endif // SLSTM_H && SLSTM_B

#ifndef FUSED_H
#define FUSED_H H
#endif
#ifndef FUSED_B
#define FUSED_B B
#endif

// Time-fused sLSTM BACKWARD: the whole reverse T-loop in one cooperative launch,
// the mirror of slstm_fused_time.
//
// The ownership is transposed relative to the forward. Forward, a block owning
// hidden units [j0, j0+nj) needed those units' gate COLUMNS of Wh; backward, the
// contraction is dh[r] = sum_c dgates[c] * Wh[r, c], so a block producing dh for
// units [j0, j0+nj) needs those ROWS of Wh -- all 4H entries of each. That is what
// this kernel stages: `wh_rows[nj][4H]`, read once before the loop.
//
// Unlike the forward, the pointwise/contraction split needs a grid sync between the
// two phases. The forward got to one because a unit's pointwise update needed only that
// unit's own gates; here the pointwise update at step t needs dh_recur[j], which
// is a full contraction over every OTHER unit's gate deltas. So: pointwise (needs
// dh_recur from t+1) -> sync -> contraction (needs all gate deltas). The trailing
// sync the contraction used to need is gone whenever the deltas are staged in shared
// memory — see the `stage_dg` note at the swap below — which is the case for the
// backbone, where this kernel spends nearly all of its time.
//
// `dgates_all` is a [B, 4H] grid-visible scratch: the pointwise phase writes this
// step's deltas there for the contraction phase to read across blocks. The kernel
// also writes them into `g` (whose forward contents are dead), exactly as the
// per-step path does, so the post-loop dWx/dWh/db GEMMs are unchanged.
extern "C" __global__ void slstm_fused_time_bwd(
        const float* __restrict__ wh, const float* __restrict__ dy, float* g,
        float* dgates_all, float* dgates_alt, float* dh_recur, float* dh_alt,
        float* dc_recur, float* dn_recur,
        const slab_t* __restrict__ ot, const float* __restrict__ c_t,
        const float* __restrict__ n_t, const float* __restrict__ c_prev,
        const float* __restrict__ n_prev, const slab_t* __restrict__ zt,
        const float* __restrict__ i_gate, const float* __restrict__ f_gate,
        int T, int H_rt, int B_rt, int units_per_block, int stage_dg) {
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

    // Delta ping-pong, for the same reason as the forward's h buffers: the staging
    // loop below reads the whole [B, 4H] vector while the pointwise phase writes
    // only this block's slice of it.
    float* dg_cur = dgates_all;
    float* dg_nxt = dgates_alt;
    // `dh_recur` ping-pongs for the same reason: the pointwise phase READS step t+1's
    // dh while the contraction below WRITES step t's. Sharing one buffer makes that a
    // write-after-read across the grid, which is what the second barrier used to close
    // — and a barrier here costs ~0.6us x T, the single largest item in this kernel.
    // Writing into the other buffer removes the hazard itself, so the barrier goes.
    // Each block only ever touches its own [j0, j0+nj) units of dh, so no block can
    // observe another's half-written slice.
    float* dh_cur = dh_recur;
    float* dh_nxt = dh_alt;

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
            float d_h = dy[s] + dh_cur[k];
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
            dg_cur[fo]          = d_z_pre;
            dg_cur[fo + FUSED_H]      = d_i_pre;
            dg_cur[fo + 2 * FUSED_H]  = d_f_pre;
            dg_cur[fo + 3 * FUSED_H]  = d_o_pre;

            dc_recur[k] = d_c * f;
            dn_recur[k] = d_n * f;
        }
        grid.sync(); // all gate deltas visible before any block contracts them

        // Stage the whole [B, 4H] delta vector into shared memory once, so the warps
        // below reduce against shared rather than each re-reading the same floats from
        // global on every timestep. The cache costs `B * 4H * 4` bytes, which only fits
        // at small B; past that `stage_dg` is off and the contraction reads `dg_cur`
        // directly (lane-contiguous either way, so the reads stay coalesced and L2
        // absorbs the reuse across blocks).
        if (stage_dg) {
            for (int i = threadIdx.x; i < FUSED_B * H4; i += blockDim.x) {
                dgsh[i] = dg_cur[i];
            }
            __syncthreads();
        }

        // --- contraction: dh_recur[b, j] = sum_c dgates_all[b, c] * Wh[j, c] ---
        // One warp per (batch, owned unit), lanes striding the 4H reduction. The
        // staged row is contiguous in c, so the lane reads are coalesced.
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        const int warps = blockDim.x >> 5;
        for (int i = warp; i < FUSED_B * nj; i += warps) {
            int b = i / nj, u = i - b * nj;
            const float* dg = (stage_dg ? dgsh : dg_cur) + (long long)b * H4;
            float acc = 0.0f;
            // Four columns per lane per trip, keeping the scalar loop's column set:
            // lane L still covers L, L+32, L+64, ... in that order, so the sum is
            // unchanged bit for bit, but the loop runs a quarter of the trips and the
            // four loads issue together instead of each waiting on the previous.
            // H4 is a multiple of 4 but not always of 128, hence the tail guards.
#ifdef FUSED_BF16
            const __nv_bfloat16* wrow = wsh + (long long)u * H4;
            for (int c = lane; c < H4; c += 128) {
                acc = fmaf(dg[c], __bfloat162float(wrow[c]), acc);
                if (c + 32 < H4) acc = fmaf(dg[c + 32], __bfloat162float(wrow[c + 32]), acc);
                if (c + 64 < H4) acc = fmaf(dg[c + 64], __bfloat162float(wrow[c + 64]), acc);
                if (c + 96 < H4) acc = fmaf(dg[c + 96], __bfloat162float(wrow[c + 96]), acc);
            }
#else
            const float* wrow = wsh + (long long)u * H4;
            for (int c = lane; c < H4; c += 128) {
                acc = fmaf(dg[c], wrow[c], acc);
                if (c + 32 < H4) acc = fmaf(dg[c + 32], wrow[c + 32], acc);
                if (c + 64 < H4) acc = fmaf(dg[c + 64], wrow[c + 64], acc);
                if (c + 96 < H4) acc = fmaf(dg[c + 96], wrow[c + 96], acc);
            }
#endif
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
            if (lane == 0) dh_nxt[b * FUSED_H + j0 + u] = acc;
        }
        // The barrier after the contraction is only needed when the contraction reads
        // `dg` straight from global: there a straggler is still reading the buffer that
        // the next step's pointwise phase overwrites, and no rotation depth fixes it
        // (measured — a deeper buffer rotation does not help; the barrier is what the
        // straggler needs).
        //
        // When the deltas are staged, each block copies its `dg` into shared memory and
        // stops touching the global buffer at the `__syncthreads()` above, so the
        // write-after-read window closes inside the block and the barrier is dead
        // weight — worth ~0.6us x T, the largest single cost in this kernel. The
        // backbone (B=1) always stages, which is where essentially all of the time is.
        if (!stage_dg) grid.sync();
        float* tmp = dg_cur; dg_cur = dg_nxt; dg_nxt = tmp;
        tmp = dh_cur; dh_cur = dh_nxt; dh_nxt = tmp;
    }

    // The caller reads dh_recur as the sequence's incoming state gradient, so it must
    // hold the newest values. After T swaps the newest dh is in `dh_cur`, which is
    // `dh_alt` for odd T — mirror it back, exactly as the forward does for h_state.
    if (dh_cur != dh_recur) {
        // The last step's contraction is no longer followed by a barrier, so the grid
        // must be squared up before one block copies units another block wrote. This
        // is one barrier per LAUNCH, not per timestep.
        grid.sync();
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < FUSED_B * FUSED_H;
             i += blockDim.x * gridDim.x) {
            dh_recur[i] = dh_cur[i];
        }
    }
}

#endif // COOP_KERNELS
