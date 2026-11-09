// Batch-parallel time-fused sLSTM: the whole T-loop in ONE cooperative launch, with
// the recurrent product `h·Wh` on the bf16 MMA unit.
#if defined(SLSTM_MMA) && defined(SLSTM_H) && defined(SB_NJ) && defined(SB_RPT)
#include <cooperative_groups.h>
namespace cg_b = cooperative_groups;

// Shared by both kernels: they own the same units and the same batch rows, and only
// the staging width (SB_WSR / SB_WSC) tells the two builds apart.
#define SB_H     SLSTM_H
#define SB_H4    (4 * SB_H)
#define SB_NCOL  (4 * SB_NJ)              // gate columns a block owns
#define SB_TH    (16 * SB_NJ)             // one warp per 8 gate columns
#define SB_BR    (16 * SB_RPT)            // batch rows a block owns
#define SB_KT    (SB_H / 16)              // k-tiles of the recurrent contraction
#define SB_LD    BF16_LD(SB_H)            // smem row stride of the `h` tile, elements

#ifdef SB_WSR
#define SB_WPASS (SB_H / SB_WSR)          // passes the `Wh` staging takes

extern "C" __global__ __launch_bounds__(SB_TH) void slstm_batched_fwd(
        const float* __restrict__ wh, float* g, const float* __restrict__ bcat,
        slab_t* h_prev, float* c_state, float* n_state, float* m_state, float* h_state,
        float* hmir, state_t* c_entry, state_t* n_entry, slab_t* zt, slab_t* ot,
        state_t* i_prime, state_t* f_prime, state_t* c_out, state_t* n_out,
        float* out, int T, int B, int row0, int carry) {
    extern __shared__ __align__(16) char sb_smem[];
    float* gacc = (float*)sb_smem;                       // [16, SB_NCOL]
    bf16s_t* htile = (bf16s_t*)(gacc + 16 * SB_NCOL);    // [SB_BR, SB_LD]
    cg_b::grid_group grid = cg_b::this_grid();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n0 = warp * 8;                             // local gate columns of this warp
    const int j0 = blockIdx.x * SB_NJ;
    const int rowbase = row0 + (int)blockIdx.y * SB_BR;

    unsigned whb[SB_KT][2];
    const int gq = lane >> 2, cq = (lane & 3) << 1;
    if (!(T == 1 && !carry)) { // a single step from a zero state never reads Wh
        bf16s_t* wst = (bf16s_t*)sb_smem;      // [SB_WSR, SB_NCOL], dead once loaded
        #pragma unroll
        for (int p = 0; p < SB_WPASS; ++p) {
            __syncthreads();
            for (int i = threadIdx.x; i < SB_WSR * SB_NCOL; i += SB_TH) {
                const int r = i / SB_NCOL, lc = i - r * SB_NCOL;
                const int gate = lc / SB_NJ;
                wst[i] = to_bf16(wh[(long long)(p * SB_WSR + r) * SB_H4 + gate * SB_H + j0
                                    + (lc - gate * SB_NJ)]);
            }
            __syncthreads();
            const bf16s_t* w = wst + n0 + gq;
            #pragma unroll
            for (int kl = 0; kl < SB_WSR / 16; ++kl) {
                const int k0 = kl * 16 + cq;
                unsigned* dst = whb[p * (SB_WSR / 16) + kl];
                dst[0] = bf16_pair(w[k0 * SB_NCOL], w[(k0 + 1) * SB_NCOL]);
                dst[1] = bf16_pair(w[(k0 + 8) * SB_NCOL], w[(k0 + 9) * SB_NCOL]);
            }
        }
        __syncthreads(); // the staging area is `gacc` and the `h` tile from here on
    }

    // Pointwise ownership, fixed for the whole loop.
    const int row_l = threadIdx.x / SB_NJ;               // row within a 16-row M-tile
    const int ul = threadIdx.x - row_l * SB_NJ;          // unit within the block's range
    const int j_pw = j0 + ul;
    const float bz = bcat[j_pw], bi = bcat[SB_H + j_pw];
    const float bf_ = bcat[2 * SB_H + j_pw], bo = bcat[3 * SB_H + j_pw];

    float c_reg[SB_RPT], n_reg[SB_RPT], m_reg[SB_RPT], h_reg[SB_RPT];
    #pragma unroll
    for (int m = 0; m < SB_RPT; ++m) {
        const int b = rowbase + m * 16 + row_l;
        c_reg[m] = n_reg[m] = m_reg[m] = h_reg[m] = 0.0f;
        if (carry && b < B) {
            const long long k = (long long)b * SB_H + j_pw;
            c_reg[m] = c_state[k];
            n_reg[m] = n_state[k];
            m_reg[m] = m_state[k];
            h_reg[m] = h_state[k];
        }
    }

    bf16s_t* hb_cur = (bf16s_t*)hmir;
    bf16s_t* hb_nxt = hb_cur + (long long)B * SB_H;

    for (int t = 0; t < T; ++t) {
        const bool zero_h = t == 0 && !carry;
        if (!zero_h) {
            #pragma unroll 1
            for (int i = threadIdx.x; i < SB_BR * (SB_H / 8); i += SB_TH) {
                const int lr = i / (SB_H / 8), r = i - lr * (SB_H / 8);
                const int b = rowbase + lr;
                int4 v = make_int4(0, 0, 0, 0);
                if (b < B) {
                    if (t > 0) {
                        v = ((const int4*)hb_cur)[(long long)b * (SB_H / 8) + r];
                    } else {
                        const float4* hp = (const float4*)(h_state + (long long)b * SB_H) + 2 * r;
                        const float4 lo = hp[0], hi = hp[1];
                        v.x = bf16_pair(to_bf16(lo.x), to_bf16(lo.y));
                        v.y = bf16_pair(to_bf16(lo.z), to_bf16(lo.w));
                        v.z = bf16_pair(to_bf16(hi.x), to_bf16(hi.y));
                        v.w = bf16_pair(to_bf16(hi.z), to_bf16(hi.w));
                    }
                }
                ((int4*)htile)[lr * (SB_LD / 8) + r] = v;
            }
        }

        float gz[SB_RPT], gin[SB_RPT], gfo[SB_RPT], gou[SB_RPT];
        #pragma unroll
        for (int m = 0; m < SB_RPT; ++m) {
            const int b = rowbase + m * 16 + row_l;
            gz[m] = gin[m] = gfo[m] = gou[m] = 0.0f;
            if (b < B) {
                const long long go = ((long long)b * T + t) * SB_H4 + j_pw;
                gz[m] = g[go];
                gin[m] = g[go + SB_H];
                gfo[m] = g[go + 2 * SB_H];
                gou[m] = g[go + 3 * SB_H];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int m = 0; m < SB_RPT; ++m) {
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (!zero_h) {
                float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                #pragma unroll
                for (int kt = 0; kt < SB_KT; kt += 2) {
                    unsigned a[4];
                    ldm_a_mk(a, htile, SB_LD, m * 16, kt * 16);
                    mma_bf16(acc, a, whb[kt]);
                    ldm_a_mk(a, htile, SB_LD, m * 16, (kt + 1) * 16);
                    mma_bf16(acc1, a, whb[kt + 1]);
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) acc[i] += acc1[i];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                gacc[mma_row(i) * SB_NCOL + n0 + mma_col(i)] = acc[i];
            }
            __syncthreads(); // gacc written before the pointwise phase reads it

            const int b = rowbase + m * 16 + row_l;
            if (b < B) {
                const long long go = ((long long)b * T + t) * SB_H4 + j_pw;
                const long long s = ((long long)b * T + t) * SB_H + j_pw;
                const float* ga = gacc + row_l * SB_NCOL + ul;
                const float z_pre = gz[m] + ga[0] + bz;
                const float i_pre = gin[m] + ga[SB_NJ] + bi;
                const float f_pre = gfo[m] + ga[2 * SB_NJ] + bf_;
                const float o_pre = gou[m] + ga[3 * SB_NJ] + bo;
                g[go + 2 * SB_H] = f_pre; // biased forget pre-activation, for backward

                slab_st(h_prev, s, h_reg[m]);

                const float z = tanhf(z_pre);
                const float o = stable_sigmoid(o_pre);
                const float fm = log_sigmoid(f_pre) + m_reg[m];
                const float mx = (n_reg[m] == 0.0f) ? i_pre : fmaxf(fm, i_pre);
                const float ip = fminf(1.0f, expf(i_pre - mx));
                const float fp = fminf(1.0f, expf(fm - mx));
                const float cc = fp * c_reg[m] + ip * z;
                const float nn = fp * n_reg[m] + ip;

                if (t == 0) {
                    state_st(c_entry, (long long)b * SB_H + j_pw, c_reg[m]);
                    state_st(n_entry, (long long)b * SB_H + j_pw, n_reg[m]);
                }
                slab_st(zt, s, z);
                slab_st(ot, s, o);
                state_st(i_prime, s, ip);
                state_st(f_prime, s, fp);
                state_st(c_out, s, cc);
                state_st(n_out, s, nn);
                c_reg[m] = cc;
                n_reg[m] = nn;
                m_reg[m] = mx;

                h_reg[m] = o * cc / nn;
                hb_nxt[(long long)b * SB_H + j_pw] = to_bf16(h_reg[m]);
                out[s] = h_reg[m];
            }
            if (SB_RPT > 1) {
                __syncthreads();
            }
        }
        if (t + 1 < T) {
            grid.sync(); // h_t complete before step t+1 reads the mirror
            bf16s_t* tmp = hb_cur;
            hb_cur = hb_nxt;
            hb_nxt = tmp;
        }
    }

    // The state the next call carries in, written once rather than every timestep.
    #pragma unroll
    for (int m = 0; m < SB_RPT; ++m) {
        const int b = rowbase + m * 16 + row_l;
        if (b < B) {
            const long long k = (long long)b * SB_H + j_pw;
            c_state[k] = c_reg[m];
            n_state[k] = n_reg[m];
            m_state[k] = m_reg[m];
            h_state[k] = h_reg[m];
        }
    }
}

#endif // SB_WSR

// Batch-parallel time-fused sLSTM BACKWARD: the whole reverse T-loop in ONE
// cooperative launch, the mirror of `slstm_batched_fwd`.
#ifdef SB_WSC

#define SBB_H4     (4 * SB_H)
#define SBB_TH     (16 * SB_NJ)
#define SBB_WARPS  (SB_NJ / 2)
#define SBB_WN     (SB_NJ / 8)              // warps along the unit axis
#define SBB_WK     (SBB_WARPS / SBB_WN)     // warps splitting the reduction
#define SBB_BR     (16 * SB_RPT)
#define SBB_KC     (SBB_H4 / SB_RPT)        // reduction columns per staged chunk
#define SBB_KTW    (SBB_KC / SBB_WK / 16)   // k-tiles a warp owns per chunk
#define SBB_KTOT   (SBB_KTW * SB_RPT)       // k-tiles a warp owns in all
#define SBB_LD     (SBB_KC + 8)             // smem row stride of the `dg` tile
#define SBB_WPASS  (SBB_H4 / SB_WSC)       // passes the `Wh` staging takes

// Two blocks per SM at SB_RPT = 1: left to itself ptxas takes 126 registers, which at
// this block width is one block and a third of the warp slots — and the kernel is
// latency-bound, not register-bound. Capped it fits 64 with no spill. Past one row
// tile it does not: the per-row state and accumulators scale with SB_RPT and `ptxas
// -v` reports a stack frame, which would give back more than the occupancy is worth.
extern "C" __global__ __launch_bounds__(SBB_TH, SB_RPT == 1 ? 2 : 1) void slstm_batched_bwd(
        const float* __restrict__ wh, const float* __restrict__ dy, float* g,
        float* dh_recur, float* dc_recur, float* dn_recur, float* dgmir,
        const slab_t* __restrict__ ot, const state_t* __restrict__ c_t,
        const state_t* __restrict__ n_t, const state_t* __restrict__ c_entry,
        const state_t* __restrict__ n_entry, const slab_t* __restrict__ zt,
        const state_t* __restrict__ i_gate, const state_t* __restrict__ f_gate,
        int T, int B, int row0) {
    extern __shared__ __align__(16) char sb_smem[];
    bf16s_t* dgt = (bf16s_t*)sb_smem;                        // [SBB_BR, SBB_LD]
    float* red = (float*)(dgt + SBB_BR * SBB_LD);            // [SBB_WK, SBB_BR, SB_NJ]
    float* dh_sh = red + SBB_WK * SBB_BR * SB_NJ;            // [SBB_BR, SB_NJ]
    cg_b::grid_group grid = cg_b::this_grid();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n0 = (warp % SBB_WN) * 8;      // this warp's 8 units, within the block
    const int wk = warp / SBB_WN;            // its share of the reduction
    const int j0 = blockIdx.x * SB_NJ;
    const int rowbase = row0 + (int)blockIdx.y * SBB_BR;
    const int gq = lane >> 2, cq = (lane & 3) << 1;

    unsigned whb[SBB_KTOT][2];
    {
        bf16s_t* wst = (bf16s_t*)sb_smem; // [SB_NJ, SB_WSC], dead once loaded
        #pragma unroll
        for (int p = 0; p < SBB_WPASS; ++p) {
            __syncthreads();
            for (int i = threadIdx.x; i < SB_NJ * SB_WSC; i += SBB_TH) {
                const int u = i / SB_WSC;
                wst[i] = to_bf16(wh[(long long)(j0 + u) * SBB_H4 + p * SB_WSC
                                    + (i - u * SB_WSC)]);
            }
            __syncthreads();
            const bf16s_t* w = wst + (n0 + gq) * SB_WSC - p * SB_WSC + cq;
            #pragma unroll
            for (int ch = 0; ch < SB_RPT; ++ch) {
                #pragma unroll
                for (int kt = 0; kt < SBB_KTW; ++kt) {
                    const int k0 = ch * SBB_KC + wk * (SBB_KC / SBB_WK) + kt * 16;
                    if (k0 >= p * SB_WSC && k0 < (p + 1) * SB_WSC) {
                        unsigned* dst = whb[ch * SBB_KTW + kt];
                        dst[0] = bf16_pair(w[k0], w[k0 + 1]);
                        dst[1] = bf16_pair(w[k0 + 8], w[k0 + 9]);
                    }
                }
            }
        }
        __syncthreads(); // the staging area is the `dg` tile from here on
    }

    // Pointwise ownership, fixed for the whole loop — the same map the forward uses.
    const int row_l = threadIdx.x / SB_NJ;
    const int ul = threadIdx.x - row_l * SB_NJ;
    const int j_pw = j0 + ul;

    float dc_reg[SB_RPT], dn_reg[SB_RPT];
    #pragma unroll
    for (int m = 0; m < SB_RPT; ++m) {
        const int b = rowbase + m * 16 + row_l;
        dc_reg[m] = dn_reg[m] = 0.0f;
        dh_sh[m * 16 * SB_NJ + threadIdx.x] = 0.0f;
        if (b < B) {
            const long long k = (long long)b * SB_H + j_pw;
            dc_reg[m] = dc_recur[k];
            dn_reg[m] = dn_recur[k];
            dh_sh[m * 16 * SB_NJ + threadIdx.x] = dh_recur[k];
        }
    }
    bf16s_t* dg_cur = (bf16s_t*)dgmir;
    bf16s_t* dg_nxt = dg_cur + (long long)B * SBB_H4;
    __syncthreads();

    for (int t = T - 1; t >= 0; --t) {
        // Pointwise: identical math to slstm_step_fused_bwd; see it for the derivation.
        #pragma unroll
        for (int m = 0; m < SB_RPT; ++m) {
            const int b = rowbase + m * 16 + row_l;
            if (b >= B) {
                continue;
            }
            const long long s = ((long long)b * T + t) * SB_H + j_pw;
            const long long go = ((long long)b * T + t) * SBB_H4 + j_pw;
            const float f_o = slab_ld(ot, s);
            const float f_c = state_ld(c_t, s);
            const float f_n = state_ld(n_t, s);
            const float f_cp = (t == 0) ? state_ld(c_entry, (long long)b * SB_H + j_pw)
                                        : state_ld(c_t, s - SB_H);
            const float f_np = (t == 0) ? state_ld(n_entry, (long long)b * SB_H + j_pw)
                                        : state_ld(n_t, s - SB_H);
            const float f_z = slab_ld(zt, s);
            const float f_i = state_ld(i_gate, s);
            const float f_f = state_ld(f_gate, s);
            const float f_fpre = g[go + 2 * SB_H]; // biased forget pre-activation

            const float d_h = dy[s] + dh_sh[m * 16 * SB_NJ + threadIdx.x];
            const float d_o_pre = d_h * (f_c / f_n) * f_o * (1.0f - f_o);
            const float d_c = d_h * f_o / f_n + dc_reg[m];
            const float d_n = d_h * f_o * (-f_c) / (f_n * f_n) + dn_reg[m];
            const float d_f_gate = d_c * f_cp + d_n * f_np;
            const float d_i_gate = d_c * f_z + d_n;

            const float d_z_pre = (d_c * f_i) * (1.0f - f_z * f_z);
            const float d_i_pre = d_i_gate * f_i;
            const float d_f_pre = d_f_gate * f_f * (1.0f - stable_sigmoid(f_fpre));

            g[go] = d_z_pre;
            g[go + SB_H] = d_i_pre;
            g[go + 2 * SB_H] = d_f_pre;
            g[go + 3 * SB_H] = d_o_pre;
            bf16s_t* dm = dg_cur + (long long)b * SBB_H4 + j_pw;
            dm[0] = to_bf16(d_z_pre);
            dm[SB_H] = to_bf16(d_i_pre);
            dm[2 * SB_H] = to_bf16(d_f_pre);
            dm[3 * SB_H] = to_bf16(d_o_pre);

            // Carry to step t-1: both paths are scaled by the forget gate.
            dc_reg[m] = d_c * f_f;
            dn_reg[m] = d_n * f_f;
        }
        grid.sync(); // this step's gate deltas visible in the mirror across the grid

        float acc[SB_RPT][4] = {};
        #pragma unroll
        for (int ch = 0; ch < SB_RPT; ++ch) {
            __syncthreads();
            for (int i = threadIdx.x; i < SBB_BR * (SBB_KC / 8); i += SBB_TH) {
                const int lr = i / (SBB_KC / 8), r = i - lr * (SBB_KC / 8);
                const int b = rowbase + lr;
                ((int4*)dgt)[lr * (SBB_LD / 8) + r] =
                    (b < B) ? ((const int4*)dg_cur)[(long long)b * (SBB_H4 / 8)
                                                    + ch * (SBB_KC / 8) + r]
                            : make_int4(0, 0, 0, 0);
            }
            __syncthreads();
            #pragma unroll
            for (int m = 0; m < SB_RPT; ++m) {
                #pragma unroll
                for (int kt = 0; kt < SBB_KTW; ++kt) {
                    unsigned a[4];
                    ldm_a_mk(a, dgt, SBB_LD, m * 16, wk * (SBB_KC / SBB_WK) + kt * 16);
                    mma_bf16(acc[m], a, whb[ch * SBB_KTW + kt]);
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int m = 0; m < SB_RPT; ++m) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                red[(wk * SBB_BR + m * 16 + mma_row(i)) * SB_NJ + n0 + mma_col(i)] =
                    acc[m][i];
            }
        }
        __syncthreads();
        #pragma unroll
        for (int m = 0; m < SB_RPT; ++m) {
            float sum = 0.0f;
            #pragma unroll
            for (int w = 0; w < SBB_WK; ++w) {
                sum += red[(w * SBB_BR + m * 16) * SB_NJ + threadIdx.x];
            }
            dh_sh[m * 16 * SB_NJ + threadIdx.x] = sum;
        }
        __syncthreads();

        bf16s_t* tmp = dg_cur;
        dg_cur = dg_nxt;
        dg_nxt = tmp;
    }

    // What the chunk to the left carries in, written once rather than per timestep.
    #pragma unroll
    for (int m = 0; m < SB_RPT; ++m) {
        const int b = rowbase + m * 16 + row_l;
        if (b < B) {
            const long long k = (long long)b * SB_H + j_pw;
            dh_recur[k] = dh_sh[m * 16 * SB_NJ + threadIdx.x];
            dc_recur[k] = dc_reg[m];
            dn_recur[k] = dn_reg[m];
        }
    }
}

#endif // SB_WSC

#endif // SLSTM_MMA && SLSTM_H && SB_NJ && SB_RPT
