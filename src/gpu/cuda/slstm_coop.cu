// Cooperative-launch sLSTM: the whole time loop in one grid-synchronising kernel.

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
// latency and Wh re-reads; fusing time removes both.
#if defined(SLSTM_H) && defined(SLSTM_B) && defined(SLSTM_RS)

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

// Rows of its Wh columns a block keeps in SHARED memory; the remaining FUSED_TAIL
// rows sit in a bf16 GLOBAL scratch in the same column-major layout and are re-read
// once per timestep.
#define FUSED_RS SLSTM_RS
#define FUSED_TAIL (FUSED_HP - FUSED_RS)
#define FUSED_SPASSES (FUSED_RS / FUSED_RPP)
#define FUSED_TPASSES (FUSED_TAIL / FUSED_RPP)

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
        float* hmir, float* wtail, state_t* c_entry, state_t* n_entry, slab_t* zt, slab_t* ot,
        state_t* i_prime, state_t* f_prime, state_t* c_out, state_t* n_out,
        float* out, int T, int units_per_block, int carry, const int* __restrict__ reset) {
    extern __shared__ __align__(16) float smem[];
    cg::grid_group grid = cg::this_grid();

    const int H4 = 4 * FUSED_H;
    const int j0 = blockIdx.x * units_per_block;
    const int nj = min(units_per_block, FUSED_H - j0);
    const int ncol = 4 * nj;                       // Wh columns staged by this block

    __nv_bfloat16* wsh = (__nv_bfloat16*)smem;
    float* gacc = (float*)(wsh + (long long)ncol * FUSED_RS);
    __nv_bfloat16* wtl = (__nv_bfloat16*)wtail;

    const __nv_bfloat16 bzero = __float2bfloat16(0.0f);
    for (int i = threadIdx.x; i < FUSED_HP * ncol; i += blockDim.x) {
        int r = i / ncol, c = i - r * ncol;
        int gidx = c / nj, j = j0 + (c - gidx * nj);
        __nv_bfloat16 v =
            (r < FUSED_H) ? __float2bfloat16(wh[(long long)r * H4 + gidx * FUSED_H + j])
                          : bzero;
        if (r < FUSED_RS) {
            wsh[(long long)c * FUSED_RS + r] = v;
        }
#if FUSED_TPASSES > 0
        else {
            wtl[(long long)(gidx * FUSED_H + j) * FUSED_TAIL + (r - FUSED_RS)] = v;
        }
#endif
    }

    __nv_bfloat16* hb_cur = (__nv_bfloat16*)hmir;
    __nv_bfloat16* hb_nxt = hb_cur + (long long)FUSED_B * FUSED_HP;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < FUSED_B * FUSED_HP;
         i += blockDim.x * gridDim.x) {
        int b = i / FUSED_HP, r = i - b * FUSED_HP;
        hb_cur[i] = (carry && r < FUSED_H) ? __float2bfloat16(h_state[b * FUSED_H + r]) : bzero;
        hb_nxt[i] = bzero;
    }
    grid.sync(); // Wh staged and the h mirror complete everywhere

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps = blockDim.x >> 5;

    const int pw = threadIdx.x;
    const bool owner = pw < FUSED_B * nj;
    const int b_pw = owner ? pw / nj : 0;
    const int jl_pw = pw - b_pw * nj;
    const int j_pw = j0 + jl_pw;
    const int k_pw = b_pw * FUSED_H + j_pw;
    float c_reg = 0.0f, n_reg = 0.0f, m_reg = 0.0f, h_reg = 0.0f;
    float bz = 0.0f, bi = 0.0f, bf = 0.0f, bo = 0.0f;
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
        long long go = ((long long)b_pw * T + t) * H4 + j_pw;
        long long s = ((long long)b_pw * T + t) * FUSED_H + j_pw;
        float gz = 0.0f, gi = 0.0f, gf = 0.0f, go4 = 0.0f;
        if (owner) {
            gz = g[go];
            gi = g[go + FUSED_H];
            gf = g[go + 2 * FUSED_H];
            go4 = g[go + 3 * FUSED_H];
        }

        for (int b = 0; b < FUSED_B; ++b) {
            float hr[FUSED_HSLICE];
            fused_load_slice(hr, hb_cur + (long long)b * FUSED_HP, lane);
            for (int c = warp; c < ncol; c += warps) {
                const __nv_bfloat16* wcol = wsh + (long long)c * FUSED_RS + lane * FUSED_RPL;
                float acc = 0.0f;
                #pragma unroll
                for (int p = 0; p < FUSED_SPASSES; ++p) {
                    bf16x8 v;
                    v.raw = *(const int4*)(wcol + p * FUSED_RPP);
                    #pragma unroll
                    for (int q = 0; q < FUSED_RPL / 2; ++q) {
                        float2 f = __bfloat1622float2(v.pair[q]);
                        acc = fmaf(hr[p * FUSED_RPL + 2 * q], f.x, acc);
                        acc = fmaf(hr[p * FUSED_RPL + 2 * q + 1], f.y, acc);
                    }
                }
#if FUSED_TPASSES > 0
                const int gidx_t = c / nj;
                const __nv_bfloat16* wtc =
                    wtl + (long long)(gidx_t * FUSED_H + j0 + (c - gidx_t * nj)) * FUSED_TAIL
                        + lane * FUSED_RPL;
                #pragma unroll
                for (int p = 0; p < FUSED_TPASSES; ++p) {
                    bf16x8 v;
                    v.raw = *(const int4*)(wtc + p * FUSED_RPP);
                    const int hb0 = (FUSED_SPASSES + p) * FUSED_RPL;
                    #pragma unroll
                    for (int q = 0; q < FUSED_RPL / 2; ++q) {
                        float2 f = __bfloat1622float2(v.pair[q]);
                        acc = fmaf(hr[hb0 + 2 * q], f.x, acc);
                        acc = fmaf(hr[hb0 + 2 * q + 1], f.y, acc);
                    }
                }
#endif
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
                if (lane == 0) gacc[b * ncol + c] = acc;
            }
        }
        __syncthreads(); // gacc written before this block reads it

        if (owner) {
            const bool rs = reset != nullptr && reset[t] != 0;
            if (rs) {
                c_reg = 0.0f; n_reg = 0.0f; m_reg = 0.0f; h_reg = 0.0f;
            }
            const float* ga = gacc + (long long)b_pw * ncol + jl_pw;
            float z_pre = gz + (rs ? 0.0f : ga[0]) + bz;
            float i_pre = gi + (rs ? 0.0f : ga[nj]) + bi;
            float f_pre = gf + (rs ? 0.0f : ga[2 * nj]) + bf;
            float o_pre = go4 + (rs ? 0.0f : ga[3 * nj]) + bo;
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

            if (t == 0) {
                state_st(c_entry, (long long)b_pw * FUSED_H + j_pw, c_reg);
                state_st(n_entry, (long long)b_pw * FUSED_H + j_pw, n_reg);
            }
            slab_st(zt, s, z); slab_st(ot, s, o);
            state_st(i_prime, s, ip); state_st(f_prime, s, fp);
            state_st(c_out, s, c);  state_st(n_out, s, n);
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

#endif // SLSTM_H && SLSTM_B && SLSTM_RS

// Time-fused sLSTM BACKWARD: the whole reverse T-loop in ONE cooperative launch,
// the mirror of slstm_fused_time.
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
        const slab_t* __restrict__ ot, const state_t* __restrict__ c_t,
        const state_t* __restrict__ n_t, const state_t* __restrict__ c_entry,
        const state_t* __restrict__ n_entry, const slab_t* __restrict__ zt,
        const state_t* __restrict__ i_gate, const state_t* __restrict__ f_gate, int T,
        const int* __restrict__ reset) {
    cg::grid_group grid = cg::this_grid();

    __shared__ float dh_sh[BW_OWNERS];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int j0 = blockIdx.x * SLSTM_NJ;

    const int u_w = warp / BW_WPU;
    const int b0_w = warp - u_w * BW_WPU;

    const int b_pw = tid / SLSTM_NJ;
    const int j_pw = j0 + (tid - b_pw * SLSTM_NJ);
    const int k_pw = b_pw * SLSTM_H + j_pw;
    const bool owner = tid < BW_OWNERS && j_pw < SLSTM_H;

    if (owner) dh_sh[tid] = dh_recur[k_pw];

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

    float f_dy = 0.0f, f_fpre = 0.0f, f_o = 0.0f, f_c = 0.0f, f_n = 0.0f;
    float f_cp = 0.0f, f_np = 0.0f, f_z = 0.0f, f_i = 0.0f, f_f = 0.0f;
    auto fetch = [&](int t) {
        if (!owner) return;
        const long long s = ((long long)b_pw * T + t) * SLSTM_H + j_pw;
        const long long go = ((long long)b_pw * T + t) * BW_H4 + j_pw;
        // A reset row's forward ran from zero state, so its predecessor is zero.
        const bool rs = reset != nullptr && reset[t] != 0;
        f_dy = dy[s];
        f_fpre = g[go + 2 * SLSTM_H]; // biased forget pre-activation, from the forward
        f_o = slab_ld(ot, s);
        f_c = state_ld(c_t, s);
        f_n = state_ld(n_t, s);
        f_cp = rs ? 0.0f
             : (t == 0) ? state_ld(c_entry, (long long)b_pw * SLSTM_H + j_pw)
                        : state_ld(c_t, s - SLSTM_H);
        f_np = rs ? 0.0f
             : (t == 0) ? state_ld(n_entry, (long long)b_pw * SLSTM_H + j_pw)
                        : state_ld(n_t, s - SLSTM_H);
        f_z = slab_ld(zt, s);
        f_i = state_ld(i_gate, s);
        f_f = state_ld(f_gate, s);
    };
    fetch(T - 1);
    __syncthreads(); // dh seeded before the first timestep reads it

    for (int t = T - 1; t >= 0; --t) {
        const bool rs = reset != nullptr && reset[t] != 0;
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
            dc_reg = rs ? 0.0f : d_c * f_f;
            dn_reg = rs ? 0.0f : d_n * f_f;
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
            if (lane == 0) dh_sh[b * SLSTM_NJ + u_w] = rs ? 0.0f : acc;
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
