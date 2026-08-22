// Cooperative-launch sLSTM: the whole time loop in one grid-synchronising kernel.
// Behind COOP_KERNELS; drags in libcu++ via <cooperative_groups.h>, so it has the
// largest include requirement of any module. Both kernels exist only in their
// shape-specialised builds — SLSTM_H / SLSTM_B for the forward, plus SLSTM_NJ /
// SLSTM_TH for the backward (see Kernels::specialized).

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

// Time-fused sLSTM BACKWARD: the whole reverse T-loop in ONE cooperative launch,
// the mirror of slstm_fused_time.
//
// A block owns a contiguous range of hidden units [j0, j0 + NJ). Everything else
// follows from that range:
//
//   * the POINTWISE update of those units is entirely block-local. Thread `p` owns
//     (batch row b, unit u) for the whole loop, so `dc`/`dn` live in registers and
//     reach global memory once at each end instead of twice per timestep, and the
//     `dh` a step consumes is produced by this same block's contraction — it never
//     leaves shared memory.
//   * the CONTRACTION dh[j] = sum_c dgates[c] * Wh[j, c] runs over EVERY gate delta
//     in the grid. That is the one cross-block dependency, and the only reason for a
//     grid.sync() at all — one per timestep. `g` is the channel: the pointwise phase
//     writes this step's deltas there for the post-loop dWx/dWh/db GEMMs anyway, and
//     `g[:, t, :]` is exactly the [B, 4H] vector the contraction wants. No separate
//     grid-visible scratch, and no write-after-read hazard to rotate buffers around,
//     because a timestep only ever touches its own slice of it.
//
// The contraction is ONE WARP PER (batch row, owned unit), with that unit's whole
// `Wh` row held across the warp's lanes in REGISTERS — bf16 storage (FlashRNN's
// `DTYPE_R`, arXiv 2412.07752), fp32 accumulation (their `ACC_DTYPE`), the conversion
// a register op. Two things follow, and both are the point:
//
//   * a row is read once per LAUNCH instead of once per timestep, so `Wh` never
//     reaches shared memory or HBM inside the loop;
//   * the reduction is a single shuffle tree whose result lands complete in lane 0 —
//     no cross-warp fold, and nothing for the pointwise phase to sum. The obvious
//     alternative, spreading every unit over every thread so each gate delta is read
//     once, costs `threads * NJ` shuffle steps per timestep against this one's
//     `threads`; measured at the backbone's shape that reduction alone was a third of
//     the kernel.
//
// SPECIALIZATION: H, B, the units per block and the block width are all compile-time
// constants (SLSTM_H / SLSTM_B / SLSTM_NJ / SLSTM_TH), which is what lets the `Wh`
// slice be a register ARRAY — a runtime bound would put it in local memory and give
// back the whole win. There is deliberately no generic build;
// `ops::slstm_fused_time_bwd` declines when the specialized one is unavailable and
// the caller takes the per-step loop. T stays a runtime argument, as it is upstream:
// it is the outer loop's bound, with a data dependency between iterations, so there
// is nothing to unroll by it.
#if defined(SLSTM_H) && defined(SLSTM_B) && defined(SLSTM_NJ) && defined(SLSTM_TH)

#define BW_H4 (4 * SLSTM_H)
#define BW_VEC 4                                  // columns one LANE holds per pass
#define BW_SPAN (32 * BW_VEC)                     // columns one warp covers per pass
#define BW_PASSES ((BW_H4 + BW_SPAN - 1) / BW_SPAN)
#define BW_OWNERS (SLSTM_B * SLSTM_NJ)
// Warps sharing a unit, each taking a stride of the batch. One at B = 1 (the
// backbone); wider batches split the rows rather than leaving warps idle.
#define BW_WPU (SLSTM_TH / (32 * SLSTM_NJ))

extern "C" __global__ __launch_bounds__(SLSTM_TH) void slstm_fused_time_bwd(
        const float* __restrict__ wh, const float* __restrict__ dy, float* g,
        float* dh_recur, float* dc_recur, float* dn_recur,
        const slab_t* __restrict__ ot, const float* __restrict__ c_t,
        const float* __restrict__ n_t, const float* __restrict__ c_prev,
        const float* __restrict__ n_prev, const slab_t* __restrict__ zt,
        const float* __restrict__ i_gate, const float* __restrict__ f_gate, int T) {
    cg::grid_group grid = cg::this_grid();

    // The block's whole dh vector: written by the contraction's lane 0, read by the
    // owning thread at the top of the next timestep. Seeded with the gradient arriving
    // from the chunk to the right (zero for an unchunked sweep), so a timestep reads
    // dh from exactly one place and the first one needs no special case.
    __shared__ float dh_sh[BW_OWNERS];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int j0 = blockIdx.x * SLSTM_NJ;

    // Contraction ownership: warp `warp` reduces unit `u_w` for batch rows
    // `b0_w, b0_w + BW_WPU, ...`.
    const int u_w = warp / BW_WPU;
    const int b0_w = warp - u_w * BW_WPU;

    // Pointwise ownership, fixed for the whole loop. The last block's range runs past
    // H when NJ does not divide it; those threads sit out and their Wh row below is
    // zero, so nothing they touch contributes.
    const int b_pw = tid / SLSTM_NJ;
    const int j_pw = j0 + (tid - b_pw * SLSTM_NJ);
    const int k_pw = b_pw * SLSTM_H + j_pw;
    const bool owner = tid < BW_OWNERS && j_pw < SLSTM_H;

    if (owner) dh_sh[tid] = dh_recur[k_pw];

    // This warp's Wh row, spread over its lanes four columns at a time. Columns past
    // 4H and rows past H are exact zeros, so the reduction has a static trip count and
    // no tail branch.
    __nv_bfloat162 wr[BW_PASSES][BW_VEC / 2];
    #pragma unroll
    for (int p = 0; p < BW_PASSES; ++p) {
        const int c0 = (p * 32 + lane) * BW_VEC;
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (c0 < BW_H4 && j0 + u_w < SLSTM_H) {
            v = *(const float4*)(wh + (long long)(j0 + u_w) * BW_H4 + c0);
        }
        wr[p][0] = __floats2bfloat162_rn(v.x, v.y);
        wr[p][1] = __floats2bfloat162_rn(v.z, v.w);
    }

    float dc_reg = 0.0f, dn_reg = 0.0f;
    if (owner) {
        dc_reg = dc_recur[k_pw];
        dn_reg = dn_recur[k_pw];
    }

    // Everything the pointwise phase reads at a timestep depends on `t` alone, so the
    // NEXT step's loads are issued before the grid.sync() and land under it and the
    // contraction. At B = 1 only a handful of lanes are awake in that phase, which is
    // far too few to hide an HBM round trip any other way.
    float f_dy = 0.0f, f_fpre = 0.0f, f_o = 0.0f, f_c = 0.0f, f_n = 0.0f;
    float f_cp = 0.0f, f_np = 0.0f, f_z = 0.0f, f_i = 0.0f, f_f = 0.0f;
    auto fetch = [&](int t) {
        if (!owner) return;
        const long long s = ((long long)b_pw * T + t) * SLSTM_H + j_pw;
        const long long go = ((long long)b_pw * T + t) * BW_H4 + j_pw;
        f_dy = dy[s];
        f_fpre = g[go + 2 * SLSTM_H]; // biased forget pre-activation, from the forward
        f_o = slab_ld(ot, s);
        f_c = c_t[s];
        f_n = n_t[s];
        f_cp = c_prev[s];
        f_np = n_prev[s];
        f_z = slab_ld(zt, s);
        f_i = i_gate[s];
        f_f = f_gate[s];
    };
    fetch(T - 1);
    __syncthreads(); // dh seeded before the first timestep reads it

    for (int t = T - 1; t >= 0; --t) {
        // --- pointwise: one thread per (batch row, owned unit) ---
        // Identical math to slstm_step_fused_bwd; see it for the derivation.
        if (owner) {
            const float d_h = f_dy + dh_sh[tid];
            const float d_o_pre = d_h * (f_c / f_n) * f_o * (1.0f - f_o);
            const float d_c = d_h * f_o / f_n + dc_reg;
            const float d_n = d_h * f_o * (-f_c) / (f_n * f_n) + dn_reg;
            const float d_f_gate = d_c * f_cp + d_n * f_np;
            const float d_i_gate = d_c * f_z + d_n;
            const float d_z_act = d_c * f_i;

            const float d_z_pre = d_z_act * (1.0f - f_z * f_z);
            const float d_i_pre = d_i_gate * f_i;
            const float d_f_pre = d_f_gate * f_f * (1.0f - stable_sigmoid(f_fpre));

            const long long go = ((long long)b_pw * T + t) * BW_H4 + j_pw;
            g[go] = d_z_pre;
            g[go + SLSTM_H] = d_i_pre;
            g[go + 2 * SLSTM_H] = d_f_pre;
            g[go + 3 * SLSTM_H] = d_o_pre;

            // Carry to step t-1: both paths are scaled by the forget gate.
            dc_reg = d_c * f_f;
            dn_reg = d_n * f_f;
        }
        if (t > 0) fetch(t - 1);
        grid.sync(); // this step's gate deltas visible in `g` across the grid

        // --- contraction: dh[b, j] = sum_c g[b, t, c] * Wh[j, c] ---
        #pragma unroll 1
        for (int b = b0_w; b < SLSTM_B; b += BW_WPU) {
            const float* dg = g + ((long long)b * T + t) * BW_H4;
            float acc = 0.0f;
            #pragma unroll
            for (int p = 0; p < BW_PASSES; ++p) {
                const int c0 = (p * 32 + lane) * BW_VEC;
                float4 d = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (c0 < BW_H4) d = *(const float4*)(dg + c0);
                const float2 w0 = __bfloat1622float2(wr[p][0]);
                const float2 w1 = __bfloat1622float2(wr[p][1]);
                acc = fmaf(d.x, w0.x, acc);
                acc = fmaf(d.y, w0.y, acc);
                acc = fmaf(d.z, w1.x, acc);
                acc = fmaf(d.w, w1.y, acc);
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                acc += __shfl_down_sync(0xffffffff, acc, off);
            }
            if (lane == 0) dh_sh[b * SLSTM_NJ + u_w] = acc;
        }
        __syncthreads(); // dh complete before the next timestep reads it
    }

    // What the chunk to the left carries in, written once rather than per timestep.
    if (owner) {
        dh_recur[k_pw] = dh_sh[tid];
        dc_recur[k_pw] = dc_reg;
        dn_recur[k_pw] = dn_reg;
    }
}

#endif // SLSTM_H && SLSTM_B && SLSTM_NJ && SLSTM_TH

#endif // COOP_KERNELS
