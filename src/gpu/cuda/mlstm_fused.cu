// mLSTM chunkwise, FUSED (TFLA — after nx-ai/mlstm_kernels): a whole sequence in
// three launches forward and two backward. The MMA_TF32 fences hold the tensor-core
// twins, compiled only for sm_80+ against the real device arch.
#define MLSTM_STRIDES_G                                                        \
    const long sG_S = 2L * H, sG_B = (long)T * sG_S;
#define MLSTM_STRIDES                                                          \
    const long sX_S = (long)H * (2 * dqk + 2 * dhv), sX_B = (long)T * sX_S;    \
    const long sY_S = (long)H * dhv, sY_H = dhv, sY_B = (long)T * sY_S;        \
    MLSTM_STRIDES_G

// Column offset of each part within its concatenated tensor.
#define MLSTM_OFF_K ((long)H * dqk)
#define MLSTM_OFF_V (2L * (long)H * dqk)
#define MLSTM_OFF_O (2L * (long)H * dqk + (long)H * dhv)
#define MLSTM_OFF_IG (0L)
#define MLSTM_OFF_FG ((long)H)

// Base offset of the (b, h) a block owns, from its flat `bh = b*H + h`. Each tensor
// group (q/k, v/h, gates) passes its own stride pair.
__device__ __forceinline__ long bhBase(int bh, int H, long sB, long sH) {
    return (long)(bh / H) * sB + (long)(bh % H) * sH;
}

// Where head `bh % H` of each part starts. q/k are `dqk` apart per head, v `dhv`,
// and the gates one element; the part's own column offset is added on top.
#define Q_BASE(bh) (bhBase((bh), H, sX_B, (long)dqk))
#define K_BASE(bh) (bhBase((bh), H, sX_B, (long)dqk) + MLSTM_OFF_K)
#define V_BASE(bh) (bhBase((bh), H, sX_B, (long)dhv) + MLSTM_OFF_V)
#define Y_BASE(bh) (bhBase((bh), H, sY_B, sY_H))
#define IG_BASE(bh) (bhBase((bh), H, sG_B, 1L) + MLSTM_OFF_IG)
#define FG_BASE(bh) (bhBase((bh), H, sG_B, 1L) + MLSTM_OFF_FG)

// Sum across a warp, every lane left holding the total. The order is the shuffle
// tree's, which the shape alone decides — unlike an atomic, whose order is the
// scheduler's and so is not reproducible run to run.
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

// SHAPE CONSTANTS. Built with -DMLSTM_SPEC the shape parameters are literals from
// NVRTC, matching how the reference passes them as `tl.constexpr`; the kernels then
// fold their padded dims, give every strided loop a static trip count, and drop the
// pad branches. Without it they stay the runtime arguments they always were, so the
// generic kernels are unchanged.
#if MLSTM_SPEC
#define MLSTM_SHAPE_ARGS \
    int T, int arg_L, int NC, int arg_dqk, int arg_dhv, int CARRY, int arg_H
#define MLSTM_SHAPE_BIND                                                       \
    (void)arg_L; (void)arg_dqk; (void)arg_dhv; (void)arg_H;                    \
    const int L = MLSTM_L, dqk = MLSTM_DQK, dhv = MLSTM_DHV, H = MLSTM_H;      \
    MLSTM_STRIDES
#define MLSTM_NTHREADS MLSTM_THREADS
// `sRed` is a static array, so it must be sized at compile time; specialized it is
// exactly the launch width instead of the generic build's worst case.
#define MLSTM_NTHREADS_MAX MLSTM_THREADS
#else
#define MLSTM_SHAPE_ARGS \
    int T, int L, int NC, int dqk, int dhv, int CARRY, int H
#define MLSTM_SHAPE_BIND MLSTM_STRIDES
#define MLSTM_NTHREADS blockDim.x
#define MLSTM_NTHREADS_MAX 1024
#endif

// Head-dim slice width for the two parallel kernels (the reference's `siz_b_DHQK`).
#ifndef MLSTM_KT
#define MLSTM_KT 32
#endif

// The `dhv` twin of MLSTM_KT (the reference's `siz_b_DHHV`). In the FORWARD this
// slices the output columns rather than a contraction, so no slice needs
// accumulating across iterations; in the backward `dhv` IS a contraction (phases 4
// and 5), which is why this must be a multiple of 16 and not just of the mma's N.
#ifndef MLSTM_VT
#define MLSTM_VT 64
#endif

// Gate preprocessing for a whole sequence: one block per (b, h). Everything the
// chunk state needs that is a pure function of the gates, so neither heavy kernel
// carries any stabilizer logic of its own. Within chunk k of length `len`:
//   fc[k][j] = Σ_{j'<=j} logsigmoid(f[k*L + j'])           (their vecB)
extern "C" __global__ void mlstm_fw_gates(
    const float* gates,
    float* fcb, float* avec, float* gvec, float* mst,
    int T, int L, int NC, int CARRY, int H) {
    MLSTM_STRIDES_G
    const int bh = blockIdx.x, tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5, nwarps = blockDim.x >> 5;
    const long igbase = IG_BASE(bh), fgbase = FG_BASE(bh);
    const long kbase = (long)bh * NC;
    float* mrow = mst + (long)bh * (NC + 1);

    extern __shared__ float sh[];
    float* sFcLast = sh;          // [NC]
    float* sMloc = sFcLast + NC;  // [NC]  the chunk-local row max, before the scan

    for (int k = warp; k < NC; k += nwarps) {
        const int c0 = k * L, len = min(L, T - c0);
        float fc = (lane < len) ? log_sigmoid(gates[fgbase + (long)(c0 + lane) * sG_S]) : 0.0f;
        for (int off = 1; off < 32; off <<= 1) {
            const float v = __shfl_up_sync(0xffffffffu, fc, off);
            if (lane >= off) fc += v;
        }
        const float fc_last = __shfl_sync(0xffffffffu, fc, len - 1);
        const float igv = (lane < len) ? gates[igbase + (long)(c0 + lane) * sG_S] : 0.0f;
        const float d = (lane < len) ? (fc_last - fc + igv) : -1e30f;
        float mx = d;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
        if (lane < L) {
            fcb[(kbase + k) * L + lane] = fc;
            avec[(kbase + k) * L + lane] = d;
        }
        if (lane == 0) {
            sFcLast[k] = fc_last;
            sMloc[k] = mx;
        }
    }
    __syncthreads();

    if (tid == 0) {
        float m = CARRY ? mrow[0] : 0.0f;
        mrow[0] = m;
        for (int k = 0; k < NC; ++k) {
            const float mn = fmaxf(sMloc[k], sFcLast[k] + m);
            gvec[kbase + k] = expf(sFcLast[k] + m - mn);
            m = mn;
            mrow[k + 1] = m;
        }
    }
    __syncthreads();

    for (int k = warp; k < NC; k += nwarps) {
        if (lane < L) {
            float* p = avec + (kbase + k) * L + lane;
            *p = expf(*p - mrow[k + 1]);
        }
    }
}

// The chunk-state recurrence, and all that is left of it: fold the per-chunk
// contributions `mlstm_fw_dC` / `mlstm_bw_dC` wrote into the running state.
extern "C" __global__ void mlstm_state_scan(
    float* cst, float* nst, const float* gvec,
    int NC, int dqk, int dhv, int REV) {
    const int bh = blockIdx.y;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    const int cn = dhv * dqk;
    if (e >= cn + dqk) return;

    float* p;
    long stride;
    if (e < cn) {
        p = cst + (long)bh * (NC + 1) * cn + e;
        stride = cn;
    } else {
        p = nst + (long)bh * (NC + 1) * dqk + (e - cn);
        stride = dqk;
    }

    const float* g = gvec + (long)bh * NC;
    const int dir = REV ? -1 : 1;
    float acc = p[(long)(REV ? NC : 0) * stride];
    // The slot this step writes, and the chunk whose `g` decays into it.
    float* d = p + (long)(REV ? NC - 1 : 1) * stride;
    int k = REV ? NC - 1 : 0;
    for (int i = 0; i < NC; ++i) {
        acc = g[k] * acc + *d;
        *d = acc;
        k += dir;
        d += (long)dir * stride;
    }
}

// dψ, for every timestep, once. Both backward kernels need it and the reduction it
// costs is over the whole `dhv` row, so it is a launch of its own rather than the
// same sum computed twice.
extern "C" __global__ void mlstm_bw_dqn(
    const float* dytil, const slab_t* ytil, const float* psiv,
    const float* qnv, const float* msv, float* dqnv,
    int N, int T, int dhv, int H) {
    const int w = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    if (w >= N) return;
    const int lane = threadIdx.x & 31;
    const long gy = (long)w * dhv;

    float red = 0.0f;
    for (int v = lane; v < dhv; v += 32) red += dytil[gy + v] * slab_ld(ytil, gy + v);
    red = warp_sum(red);
    if (lane == 0) {
        // The scalars are `[BH, T]`, so the (b, t, h) this warp owns has to be rebuilt.
        const int h = w % H, bt = w / H, t = bt % T;
        const long gt = (long)((bt / T) * H + h) * T + t;
        const float qn = qnv[gt];
        const float dpsi = -red / psiv[gt];
        dqnv[gt] = (fabsf(qn) > expf(-msv[gt])) ? ((qn > 0.0f ? 1.0f : -1.0f) * dpsi) : 0.0f;
    }
}

// Tensor-core dots (sm_80+): the m16n8k16 bf16 primitives these kernels contract
// on live in `mma.cu`, which the batched sLSTM shares.
#if MMA_TF32

// The chunk-LOCAL contribution to the state, one block per (chunk, bh):
//   ΔC_k = Σ_j a_k[j]·V_k[j] ⊗ K_k[j]      [dhv, dqk]
//   Δn_k = Σ_j a_k[j]·K_k[j]               [dqk]
// written into slot k+1 of `cst`/`nst`, where `mlstm_state_scan` folds them into the
// running state. The split is exact rather than an approximation: `a` already
// carries the stabilizer (see `mlstm_fw_gates`), so nothing here depends on the
// state to its left and EVERY chunk runs at once.
extern "C" __global__ void mlstm_fw_dC(
    const slab_t* qkv, const float* avec,
    float* cst, float* nst,
    MLSTM_SHAPE_ARGS) {
    MLSTM_SHAPE_BIND
    const int k = blockIdx.x, bh = blockIdx.y;
    const int tid = threadIdx.x, nthreads = blockDim.x;
    const int warp = tid >> 5, nwarps = nthreads >> 5;
    const int c0 = k * L, len = min(L, T - c0);
    const int LP = (L + 15) & ~15;
    const int VP = (dhv + 15) & ~15;
    const int KP = (dqk + 7) & ~7;
    const int LV = BF16_LD(VP), LK = BF16_LD(KP);

    extern __shared__ float sh[];
    float* sA = sh;                                  // [L]
    bf16s_t* sV = (bf16s_t*)(sA + L);                // [LP, LV]  V with `a` folded in
    bf16s_t* sK = sV + (long)LP * LV;                // [LP, LK]

    const long kB = K_BASE(bh);
    const long vB = V_BASE(bh);
    const float* ap = avec + ((long)bh * NC + k) * L;
    for (int j = tid; j < L; j += nthreads) sA[j] = ap[j];
    __syncthreads();

    for (int e = tid; e < LP * VP; e += nthreads) {
        const int j = e / VP, v = e - j * VP;
        sV[j * LV + v] = to_bf16((j < len && v < dhv)
            ? sA[j] * slab_ld(qkv, vB + (long)(c0 + j) * sX_S + v) : 0.0f);
    }
    for (int e = tid; e < LP * KP; e += nthreads) {
        const int j = e / KP, q = e - j * KP;
        sK[j * LK + q] = to_bf16((j < len && q < dqk)
            ? slab_ld(qkv, kB + (long)(c0 + j) * sX_S + q) : 0.0f);
    }
    __syncthreads();

    if (k == 0 && !CARRY) {
        float* c0out = cst + (long)bh * (NC + 1) * dhv * dqk;
        for (int e = tid; e < dhv * dqk; e += nthreads) c0out[e] = 0.0f;
        float* n0out = nst + (long)bh * (NC + 1) * dqk;
        for (int e = tid; e < dqk; e += nthreads) n0out[e] = 0.0f;
    }

    float* cout = cst + ((long)bh * (NC + 1) + k + 1) * dhv * dqk;
    const int ntile = KP >> 3;
    for (int tile = warp; tile < (VP >> 4) * ntile; tile += nwarps) {
        const int m0 = (tile / ntile) << 4, n0 = (tile % ntile) << 3;
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        unsigned a[4], b[2];
        for (int k0 = 0; k0 < LP; k0 += 16) {
            ldb_a_km(a, sV, LV, m0, k0);
            ldb_b_kn(b, sK, LK, k0, n0);
            mma_bf16(d, a, b);
        }
        for (int i = 0; i < 4; ++i) {
            const int v = m0 + mma_row(i), q = n0 + mma_col(i);
            if (v < dhv && q < dqk) cout[(long)v * dqk + q] = d[i];
        }
    }
    // Δn is a matrix-VECTOR product, so there is no dot for the tensor cores here.
    float* nout = nst + ((long)bh * (NC + 1) + k + 1) * dqk;
    for (int q = tid; q < dqk; q += nthreads) {
        float acc = 0.0f;
        for (int j = 0; j < len; ++j) acc += sA[j] * from_bf16(sK[j * LK + q]);
        nout[q] = acc;
    }
}

// One block per (chunk, bh) — all chunks at once. Intra-chunk attention plus the
// read-out of the incoming state:
//   num[t] = Σ_j (D̄⊙S)[t][j]·V[j] + b[t]·(Q[t]·C_prevᵀ)
extern "C" __global__ void mlstm_fw_parallel(
    const slab_t* qkv, const float* gates, const float* fcb,
    const float* cst, const float* nst, const float* mst,
    slab_t* ytil, float* msv, float* psiv, float* qnv,
    MLSTM_SHAPE_ARGS) {
    MLSTM_SHAPE_BIND
    int k = blockIdx.x, bh = blockIdx.y;
    int tid = threadIdx.x, nthreads = blockDim.x;
    int warp = tid >> 5, nwarps = nthreads >> 5;
    int c0 = k * L;
    int len = min(L, T - c0);
    const int has_state = (k > 0) || CARRY;

    const int LP = (L + 15) & ~15;    // rows -> multiple of the mma M
    const int KP = (dqk + 15) & ~15;  // dqk  -> multiple of the mma K
    const int VP = (dhv + 7) & ~7;    // dhv  -> multiple of the mma N
    const int KT = (MLSTM_KT < KP) ? MLSTM_KT : KP;
    const int VT = (MLSTM_VT < VP) ? MLSTM_VT : VP;
    const int NKT = (KP + KT - 1) / KT;
    const int NVT = (VP + VT - 1) / VT;
    const int LS = LP + 1, LA = VT + 1;                       // +1: fp32 bank pad
    const int LQ = BF16_LD(KT), LV = BF16_LD(VT), LD = BF16_LD(LP);

    extern __shared__ float sh[];
    float* sDS = sh;                  // [LP, LS]
    float* sN  = sDS + LP * LS;       // [KT]
    float* sFc = sN + KT;             // [LP]
    float* sIg = sFc + LP;            // [LP]
    float* sM  = sIg + LP;            // [LP]
    float* sB  = sM + LP;             // [LP]
    float* sQn = sB + LP;             // [LP]
    float* sQi = sQn + LP;            // [LP]  Q·n_prev, accumulated over dqk slices
    float* sAcc = sQi + LP;           // [LP, LA] the output tile's running sum
    bf16s_t* sQ  = (bf16s_t*)(sAcc + LP * LA);  // [LP, LQ]  one dqk slice
    bf16s_t* sK  = sQ + (long)LP * LQ;          // [LP, LQ]  one dqk slice
    bf16s_t* sV  = sK + (long)LP * LQ;          // [LP, LV]  one dhv slice
    bf16s_t* sC  = sV + (long)LP * LV;          // [VT, LQ]  one (dhv, dqk) tile
    bf16s_t* sDSb = sC + (long)VT * LQ;         // [LP, LD]

    for (int j = tid; j < LP; j += nthreads) {
        sFc[j] = (j < L)   ? fcb[((long)bh * NC + k) * L + j] : 0.0f;
        sIg[j] = (j < len) ? gates[IG_BASE(bh) + (long)(c0 + j) * sG_S]                   : 0.0f;
        sQi[j] = 0.0f;
    }
    float m_prev = mst[(long)bh * (NC + 1) + k];
    __syncthreads();

    for (int t = tid; t < LP; t += nthreads) {
        float mx = 0.0f, b = 0.0f;
        if (t < len) {
            float fct = sFc[t];
            mx = fct + m_prev;
            for (int j = 0; j <= t; ++j) mx = fmaxf(mx, fct - sFc[j] + sIg[j]);
            b = expf(fct + m_prev - mx);
        }
        sM[t] = mx;
        sB[t] = b;
    }

    int mtile = LP >> 4, ntile = LP >> 3;

    for (int kt = 0; kt < NKT; ++kt) {
        int q0 = kt * KT;
        __syncthreads();
        for (int e = tid; e < LP * KT; e += nthreads) {
            int t = e / KT, qs = e - t * KT, q = q0 + qs;
            int ok = (t < len) && (q < dqk);
            long off = Q_BASE(bh) + (long)(c0 + t) * sX_S + q;
            sQ[t * LQ + qs] = to_bf16(ok ? slab_ld(qkv, off) : 0.0f);
            sK[t * LQ + qs] = to_bf16(ok ? slab_ld(qkv, off + MLSTM_OFF_K) : 0.0f);
        }
        for (int e = tid; e < KT; e += nthreads) {
            int q = q0 + e;
            sN[e] = (has_state && q < dqk) ? nst[((long)bh * (NC + 1) + k) * dqk + q] : 0.0f;
        }
        __syncthreads();

        for (int tile = warp; tile < mtile * ntile; tile += nwarps) {
            int m0 = (tile / ntile) << 4, n0 = (tile % ntile) << 3;
            float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            unsigned a[4], b[2];
            for (int k0 = 0; k0 < KT; k0 += 16) {
                ldb_a_mk(a, sQ, LQ, m0, k0);
                ldb_b_nk(b, sK, LQ, k0, n0);
                mma_bf16(d, a, b);
            }
            for (int i = 0; i < 4; ++i) {
                int t = m0 + mma_row(i), j = n0 + mma_col(i);
                float prev = (kt == 0) ? 0.0f : sDS[t * LS + j];
                float val = prev + d[i];
                if (kt == NKT - 1) {
                    val = (t < len && j <= t)
                        ? expf(sFc[t] - sFc[j] + sIg[j] - sM[t]) * val : 0.0f;
                    sDSb[t * LD + j] = to_bf16(val);
                }
                sDS[t * LS + j] = val;
            }
        }
        for (int t = tid; t < len; t += nthreads) {
            float qi = 0.0f;
            for (int qs = 0; qs < KT; ++qs) qi += from_bf16(sQ[t * LQ + qs]) * sN[qs];
            sQi[t] += qi;
        }
    }
    __syncthreads();

    for (int t = tid; t < len; t += nthreads) {
        float acc = 0.0f;
        for (int j = 0; j <= t; ++j) acc += sDS[t * LS + j];
        acc += sB[t] * sQi[t];
        sQn[t] = acc;
        long gt = (long)bh * T + c0 + t;
        qnv[gt] = acc;
        msv[gt] = sM[t];
        psiv[gt] = fmaxf(fabsf(acc), expf(-sM[t]));
    }

    int vtile = VT >> 3;
    for (int vt = 0; vt < NVT; ++vt) {
        int v0 = vt * VT;
        __syncthreads();
        for (int e = tid; e < LP * VT; e += nthreads) {
            int t = e / VT, vs = e - t * VT, v = v0 + vs;
            sV[t * LV + vs] = to_bf16(((t < len) && (v < dhv))
                ? slab_ld(qkv, V_BASE(bh) + (long)(c0 + t) * sX_S + v) : 0.0f);
        }
        __syncthreads();

        for (int tile = warp; tile < mtile * vtile; tile += nwarps) {
            int m0 = (tile / vtile) << 4, n0 = (tile % vtile) << 3;
            float dintra[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            unsigned a[4], b[2];
            for (int k0 = 0; k0 < LP; k0 += 16) {   // (D̄⊙S)·V, contracting j
                ldb_a_mk(a, sDSb, LD, m0, k0);
                ldb_b_kn(b, sV, LV, k0, n0);
                mma_bf16(dintra, a, b);
            }
            for (int i = 0; i < 4; ++i) {
                int t = m0 + mma_row(i), vs = n0 + mma_col(i);
                sAcc[t * LA + vs] = dintra[i];
            }
        }
        __syncthreads();

        for (int kt = 0; has_state && kt < NKT; ++kt) {
            int q0 = kt * KT;
            __syncthreads();
            for (int e = tid; e < LP * KT; e += nthreads) {
                int t = e / KT, qs = e - t * KT, q = q0 + qs;
                sQ[t * LQ + qs] = to_bf16(((t < len) && (q < dqk))
                    ? slab_ld(qkv, Q_BASE(bh) + (long)(c0 + t) * sX_S + q) : 0.0f);
            }
            for (int e = tid; e < VT * KT; e += nthreads) {
                int vs = e / KT, qs = e - vs * KT, v = v0 + vs, q = q0 + qs;
                sC[vs * LQ + qs] = to_bf16(((v < dhv) && (q < dqk))
                    ? cst[((long)bh * (NC + 1) + k) * dhv * dqk + (long)v * dqk + q] : 0.0f);
            }
            __syncthreads();
            for (int tile = warp; tile < mtile * vtile; tile += nwarps) {
                int m0 = (tile / vtile) << 4, n0 = (tile % vtile) << 3;
                float dinter[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                unsigned a[4], b[2];
                for (int k0 = 0; k0 < KT; k0 += 16) {
                    ldb_a_mk(a, sQ, LQ, m0, k0);
                    ldb_b_nk(b, sC, LQ, k0, n0);
                    mma_bf16(dinter, a, b);
                }
                for (int i = 0; i < 4; ++i) {
                    int t = m0 + mma_row(i), vs = n0 + mma_col(i);
                    if (t < len && v0 + vs < dhv) sAcc[t * LA + vs] += sB[t] * dinter[i];
                }
            }
        }
        __syncthreads();
        for (int e = tid; e < LP * VT; e += nthreads) {
            int t = e / VT, vs = e - t * VT, v = v0 + vs;
            if (t < len && v < dhv)
                slab_st(ytil, Y_BASE(bh) + (long)(c0 + t) * sY_S + v,
                        sAcc[t * LA + vs] / fmaxf(fabsf(sQn[t]), expf(-sM[t])));
        }
    }
}

// The chunk-LOCAL contribution to the BPTT state, one block per (chunk, bh) — the
// exact mirror of `mlstm_fw_dC`:
//   ΔdC_k[v][q] = Σ_t b_k[t]·d_num_k[t][v]·Q_k[t][q]      [dhv, dqk]
//   Δdn_k[q]    = Σ_t b_k[t]·d_qn_k[t]·Q_k[t][q]          [dqk]
// written into slot k of `dcst`/`dnst`, where `mlstm_state_scan` (REV) folds them
// into the running gradient `dcst[k] = g_k·dcst[k+1] + ΔdC_k`.
extern "C" __global__ void mlstm_bw_dC(
    const slab_t* qkv, const float* dytil, const float* psiv, const float* dqnv,
    const float* fcb, const float* mst, const float* msv,
    float* dcst, float* dnst,
    MLSTM_SHAPE_ARGS) {
    MLSTM_SHAPE_BIND
    const int k = blockIdx.x, bh = blockIdx.y;
    const int tid = threadIdx.x, nthreads = MLSTM_NTHREADS;
    const int warp = tid >> 5, nwarps = nthreads >> 5;
    const int c0 = k * L, len = min(L, T - c0);
    const int LP = (L + 15) & ~15;
    const int VP = (dhv + 15) & ~15;
    const int KP = (dqk + 7) & ~7;
    const int LV = BF16_LD(VP), LQ = BF16_LD(KP);

    extern __shared__ float sh[];
    float* sB = sh;                                  // [L]  b[t]/ψ[t]
    float* sBD = sB + L;                             // [L]  b[t]·d_qn[t]
    bf16s_t* sDN = (bf16s_t*)(sBD + L);              // [LP, LV]  d_num, `b` folded in
    bf16s_t* sQ = sDN + (long)LP * LV;               // [LP, LQ]

    const long qB = Q_BASE(bh);
    const long yB = Y_BASE(bh);
    const float m_prev = mst[(long)bh * (NC + 1) + k];
    for (int t = tid; t < len; t += nthreads) {
        const long gt = (long)bh * T + c0 + t;
        const float b = expf(fcb[((long)bh * NC + k) * L + t] + m_prev - msv[gt]);
        sB[t] = b / psiv[gt];
        sBD[t] = b * dqnv[gt];
    }

    if (k == 0 && !CARRY) {
        float* dcin = dcst + ((long)bh * (NC + 1) + NC) * dhv * dqk;
        for (int e = tid; e < dhv * dqk; e += nthreads) dcin[e] = 0.0f;
        float* dnin = dnst + ((long)bh * (NC + 1) + NC) * dqk;
        for (int e = tid; e < dqk; e += nthreads) dnin[e] = 0.0f;
    }
    for (int e = tid; e < LP * KP; e += nthreads) {
        const int t = e / KP, q = e - t * KP;
        sQ[t * LQ + q] = to_bf16((t < len && q < dqk)
            ? slab_ld(qkv, qB + (long)(c0 + t) * sX_S + q) : 0.0f);
    }
    __syncthreads();

    for (int e = tid; e < LP * VP; e += nthreads) {
        const int t = e / VP, v = e - t * VP;
        sDN[t * LV + v] = to_bf16((t < len && v < dhv)
            ? sB[t] * dytil[yB + (long)(c0 + t) * sY_S + v] : 0.0f);
    }
    __syncthreads();

    float* dcout = dcst + ((long)bh * (NC + 1) + k) * dhv * dqk;
    const int ntile = KP >> 3;
    for (int tile = warp; tile < (VP >> 4) * ntile; tile += nwarps) {
        const int m0 = (tile / ntile) << 4, n0 = (tile % ntile) << 3;
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        unsigned a[4], b[2];
        for (int k0 = 0; k0 < LP; k0 += 16) {
            ldb_a_km(a, sDN, LV, m0, k0);
            ldb_b_kn(b, sQ, LQ, k0, n0);
            mma_bf16(d, a, b);
        }
        for (int i = 0; i < 4; ++i) {
            const int v = m0 + mma_row(i), q = n0 + mma_col(i);
            if (v < dhv && q < dqk) dcout[(long)v * dqk + q] = d[i];
        }
    }
    // Δdn is a matrix-VECTOR product, so there is no dot for the tensor cores here.
    float* dnout = dnst + ((long)bh * (NC + 1) + k) * dqk;
    for (int q = tid; q < dqk; q += nthreads) {
        float acc = 0.0f;
        for (int t = 0; t < len; ++t) acc += sBD[t] * from_bf16(sQ[t * LQ + q]);
        dnout[q] = acc;
    }
}

// The intra-chunk backward, one block per (chunk, bh) — the twin of
// `mlstm_fw_parallel`; every contraction in it runs on the tensor cores.
extern "C" __global__ void mlstm_bw_parallel(
    const slab_t* qkv, const float* gates, const float* fcb,
    const float* cst, const float* nst, const float* mst,
    const float* dcst, const float* dnst,
    const float* dytil, const float* psiv, const float* dqnv, const float* msv,
    float* dqkv, float* dgates,
    MLSTM_SHAPE_ARGS) {
    MLSTM_SHAPE_BIND
    (void)CARRY;
    const int k = blockIdx.x, bh = blockIdx.y;
    const int tid = threadIdx.x, nthreads = MLSTM_NTHREADS;
    const int warp = tid >> 5, nwarps = nthreads >> 5, lane = tid & 31;
    const int c0 = k * L;
    const int len = min(L, T - c0);

    const int LP = (L + 15) & ~15;
    const int KP = (dqk + 15) & ~15;
    const int VP = (dhv + 15) & ~15;
    const int KT = (MLSTM_KT < KP) ? MLSTM_KT : KP;
    const int NKT = (KP + KT - 1) / KT;
    const int VT = (MLSTM_VT < VP) ? MLSTM_VT : VP;
    const int NVT = (VP + VT - 1) / VT;
    const int LS = LP + 1, LV = VT + 1, LK = KT + 1;      // +1: fp32 bank pad
    const int LQb = BF16_LD(KT), LVb = BF16_LD(VT), LDb = BF16_LD(LP);

    extern __shared__ float sh[];
    float* sDS  = sh;                  // [LP, LS]   DS, then dS
    float* sDds = sDS + LP * LS;       // [LP, LS]   phase 4's dhv contraction
    float* sSt  = sDds + LP * LS;      // [LP, LV]   Σ_q dC·K, summed over dqk slices
    float* sDqi = sSt + LP * LV;       // [LP, LK]   phase 5's dQ tile
    float* sDki = sDqi + LP * LK;      // [LP, LK]   phase 5's dK tile
    float* sDqx = sDki + LP * LK;      // [LP, LK]   Σ_v dN·C, summed over dhv slices
    float* sN   = sDqx + LP * LK;      // [KT]
    float* sdN  = sN + KT;             // [KT]
    float* sFc  = sdN + KT;            // [LP]
    float* sIg  = sFc + LP;            // [LP]
    float* sM   = sIg + LP;            // [LP]
    float* sB   = sM + LP;             // [LP]
    float* sA   = sB + LP;             // [LP]
    float* sDQn = sA + LP;             // [LP]
    float* sDfc = sDQn + LP;           // [LP]
    float* sDig = sDfc + LP;           // [LP]
    float* sDa  = sDig + LP;           // [LP]
    float* sDb  = sDa + LP;            // [LP]
    bf16s_t* sQ   = (bf16s_t*)(sDb + LP);   // [LP, LQb]  one dqk slice
    bf16s_t* sK   = sQ + (long)LP * LQb;    // [LP, LQb]  one dqk slice
    bf16s_t* sV   = sK + (long)LP * LQb;    // [LP, LVb]  one dhv slice
    bf16s_t* sDN  = sV + (long)LP * LVb;    // [LP, LVb]  d_num, one dhv slice
    bf16s_t* sC   = sDN + (long)LP * LVb;   // [VT, LQb]  C_{k-1}, one (dhv, dqk) tile
    bf16s_t* sdC  = sC + (long)VT * LQb;    // [VT, LQb]  dC_k,    one (dhv, dqk) tile
    bf16s_t* sDSb = sdC + (long)VT * LQb;   // [LP, LDb]  the narrowed DS / dS
    __shared__ float sRed[MLSTM_NTHREADS_MAX];

    #define STAGE_QK(q0)                                                              \
        for (int e = tid; e < LP * KT; e += nthreads) {                               \
            int t = e / KT, q = e - t * KT;                                           \
            int ok = (t < len) && ((q0) + q < dqk);                                   \
            long base = Q_BASE(bh) + (long)(c0 + t) * sX_S + (q0) + q;                \
            sQ[t * LQb + q] = to_bf16(ok ? slab_ld(qkv, base) : 0.0f);                \
            sK[t * LQb + q] = to_bf16(ok ? slab_ld(qkv, base + MLSTM_OFF_K) : 0.0f);  \
        }
    #define STAGE_STATE(dst, src, slot, v0, q0)                                       \
        for (int e = tid; e < VT * KT; e += nthreads) {                               \
            int vs = e / KT, q = e - vs * KT;                                         \
            int ok = ((v0) + vs < dhv) && ((q0) + q < dqk);                           \
            long off = (long)((v0) + vs) * dqk + (q0) + q;                            \
            dst[vs * LQb + q] = to_bf16(                                              \
                ok ? src[((long)bh * (NC + 1) + (slot)) * dhv * dqk + off] : 0.0f);    \
        }
    #define STAGE_N(q0)                                                               \
        for (int e = tid; e < KT; e += nthreads) {                                    \
            int ok = (q0) + e < dqk;                                                  \
            sN[e] = ok ? nst[((long)bh * (NC + 1) + k) * dqk + (q0) + e] : 0.0f;       \
            sdN[e] = ok ? dnst[((long)bh * (NC + 1) + (k + 1)) * dqk + (q0) + e] : 0.0f; \
        }

    // The `v0` slice of V and of d_num.
    #define STAGE_V(v0)                                                               \
        for (int e = tid; e < LP * VT; e += nthreads) {                               \
            int t = e / VT, vs = e - t * VT, v = (v0) + vs;                           \
            int ok = (t < len) && (v < dhv);                                          \
            long gv = V_BASE(bh) + (long)(c0 + t) * sX_S + v;                         \
            long gy = Y_BASE(bh) + (long)(c0 + t) * sY_S + v;                         \
            sV[t * LVb + vs] = to_bf16(ok ? slab_ld(qkv, gv) : 0.0f);                 \
            sDN[t * LVb + vs] = to_bf16(                                              \
                ok ? dytil[gy] / psiv[(long)bh * T + c0 + t] : 0.0f);                 \
        }

    for (int j = tid; j < LP; j += nthreads) {
        sFc[j] = (j < L)   ? fcb[((long)bh * NC + k) * L + j] : 0.0f;
        sIg[j] = (j < len) ? gates[IG_BASE(bh) + (long)(c0 + j) * sG_S]                   : 0.0f;
        sM[j] = 0.0f; sB[j] = 0.0f; sA[j] = 0.0f; sDQn[j] = 0.0f;
        sDfc[j] = 0.0f; sDig[j] = 0.0f; sDa[j] = 0.0f; sDb[j] = 0.0f;
    }
    float m_prev = mst[(long)bh * (NC + 1) + k];
    __syncthreads();

    float fc_last = sFc[len - 1];
    float m_last = msv[(long)bh * T + c0 + len - 1];
    float gsca = expf(fc_last + m_prev - m_last);

    for (int t = tid; t < len; t += nthreads) {
        long gt = (long)bh * T + c0 + t;
        sM[t] = msv[gt];
        sB[t] = expf(sFc[t] + m_prev - sM[t]);
        sA[t] = expf(fc_last - sFc[t] + sIg[t] - m_last);
        sDQn[t] = dqnv[gt];
    }
    __syncthreads();

    int mtile = LP >> 4, ltile = LP >> 3, vtile = VT >> 3, ktile = KT >> 3;

    for (int e = tid; e < LP * LS; e += nthreads) sDS[e] = 0.0f;
    float dgloc = 0.0f;
    for (int kt = 0; kt < NKT; ++kt) {
        __syncthreads();
        STAGE_QK(kt * KT);
        __syncthreads();

        // DS += Q·Kᵀ over this slice. Each (t, j) belongs to one warp — no atomic.
        for (int tile = warp; tile < mtile * ltile; tile += nwarps) {
            int m0 = (tile / ltile) << 4, n0 = (tile % ltile) << 3;
            float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            unsigned a[4], b[2];
            for (int k0 = 0; k0 < KT; k0 += 16) {
                ldb_a_mk(a, sQ, LQb, m0, k0);
                ldb_b_nk(b, sK, LQb, k0, n0);
                mma_bf16(d, a, b);
            }
            for (int i = 0; i < 4; ++i) {
                int t = m0 + mma_row(i), j = n0 + mma_col(i);
                sDS[t * LS + j] += d[i];
            }
        }

        STAGE_N(kt * KT);
        __syncthreads();
        for (int j = warp; j < len; j += nwarps) {
            float acc = 0.0f, pre_qn = 0.0f;
            for (int q = lane; q < KT; q += 32) {
                acc += sdN[q] * from_bf16(sK[j * LQb + q]);
                pre_qn += from_bf16(sQ[j * LQb + q]) * sN[q];
            }
            acc = warp_sum(acc);
            pre_qn = warp_sum(pre_qn);
            if (lane == 0) {
                sDa[j] += acc;
                sDb[j] += sDQn[j] * pre_qn;
            }
        }
        for (int e = tid; e < KT; e += nthreads) dgloc += sdN[e] * sN[e];
    }
    __syncthreads();
    for (int e = tid; e < LP * LS; e += nthreads) {
        int t = e / LS, j = e - t * LS;
        float val = 0.0f;
        if (t < len && j <= t) val = expf(sFc[t] - sFc[j] + sIg[j] - sM[t]) * sDS[e];
        sDS[e] = val;
        sDSb[t * LDb + j] = to_bf16(val);
    }
    __syncthreads();

    for (int e = tid; e < LP * LS; e += nthreads) sDds[e] = 0.0f;
    __syncthreads();
    for (int vt = 0; vt < NVT; ++vt) {
        int v0 = vt * VT;
        __syncthreads();
        STAGE_V(v0);
        for (int e = tid; e < LP * LV; e += nthreads) sSt[e] = 0.0f;
        __syncthreads();

        for (int kt = 0; kt < NKT; ++kt) {
            __syncthreads();
            STAGE_QK(kt * KT);
            STAGE_STATE(sdC, dcst, k + 1, v0, kt * KT);
            __syncthreads();
            for (int tile = warp; tile < mtile * vtile; tile += nwarps) {
                int m0 = (tile / vtile) << 4, n0 = (tile % vtile) << 3;
                float dst[4] = {0.0f, 0.0f, 0.0f, 0.0f};   // Σ_q dC[v][q]·K[j][q]
                unsigned a[4], b[2];
                for (int k0 = 0; k0 < KT; k0 += 16) {
                    ldb_a_mk(a, sK, LQb, m0, k0);
                    ldb_b_nk(b, sdC, LQb, k0, n0);
                    mma_bf16(dst, a, b);
                }
                for (int i = 0; i < 4; ++i) {
                    int r = m0 + mma_row(i), vs = n0 + mma_col(i);
                    sSt[r * LV + vs] += dst[i];
                }
            }
        }
        __syncthreads();

        // dnum, then the epilogue phase 2 owed: dV and the v-side of da/db.
        for (int tile = warp; tile < mtile * vtile; tile += nwarps) {
            int m0 = (tile / vtile) << 4, n0 = (tile % vtile) << 3;
            float dnum[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Σ_t DS[t][j]·dN[t][v]
            unsigned a[4], b[2];
            // DS[t][j] is zero for j > t, so contracting over ALL t is the same as t >= j.
            for (int k0 = 0; k0 < LP; k0 += 16) {
                ldb_a_km(a, sDSb, LDb, m0, k0);  // Aᵀ in memory: DS is [t, j], we want [j, t]
                ldb_b_kn(b, sDN, LVb, k0, n0);
                mma_bf16(dnum, a, b);
            }
            for (int i = 0; i < 4; ++i) {
                int r = m0 + mma_row(i), vs = n0 + mma_col(i);
                int v = v0 + vs;             // r is `j` for dv, `t` for db
                if (r < len && v < dhv) {
                    float st = sSt[r * LV + vs];
                    dqkv[V_BASE(bh) + (long)(c0 + r) * sX_S + v] = dnum[i] + sA[r] * st;
                }
            }
        }

        for (int r = warp; r < len; r += nwarps) {
            float da = 0.0f;
            for (int vs = lane; vs < VT; vs += 32)
                da += from_bf16(sV[r * LVb + vs]) * sSt[r * LV + vs];
            da = warp_sum(da);
            if (lane == 0) sDa[r] += da;
        }

        // Phase 4's dhv contraction, accumulated across the slices.
        for (int tile = warp; tile < mtile * ltile; tile += nwarps) {
            int m0 = (tile / ltile) << 4, n0 = (tile % ltile) << 3;
            float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};   // Σ_v dN[t][v]·V[j][v]
            unsigned a[4], b[2];
            for (int k0 = 0; k0 < VT; k0 += 16) {
                ldb_a_mk(a, sDN, LVb, m0, k0);
                ldb_b_nk(b, sV, LVb, k0, n0);
                mma_bf16(d, a, b);
            }
            for (int i = 0; i < 4; ++i) {
                int t = m0 + mma_row(i), j = n0 + mma_col(i);
                sDds[t * LS + j] += d[i];
            }
        }
    }
    __syncthreads();

    for (int tile = warp; tile < mtile * ltile; tile += nwarps) {
        int m0 = (tile / ltile) << 4, n0 = (tile % ltile) << 3;
        for (int i = 0; i < 4; ++i) {
            int t = m0 + mma_row(i), j = n0 + mma_col(i);
            float out = 0.0f, p = 0.0f;
            if (t < len && j <= t) {
                float dds = sDds[t * LS + j] + sDQn[t];
                p = dds * sDS[t * LS + j];
                out = dds * expf(sFc[t] - sFc[j] + sIg[j] - sM[t]);
            }
            sDds[t * LS + j] = p;
            sDSb[t * LDb + j] = to_bf16(out);
        }
    }
    __syncthreads();
    for (int r = warp; r < len; r += nwarps) {
        float row = 0.0f, col = 0.0f;
        for (int j = lane; j <= r; j += 32) row += sDds[r * LS + j];
        for (int t = r + lane; t < len; t += 32) col += sDds[t * LS + r];
        row = warp_sum(row);
        col = warp_sum(col);
        if (lane == 0) {
            sDfc[r] += row - col;
            sDig[r] += col;
        }
    }
    __syncthreads();

    for (int kt = 0; kt < NKT; ++kt) {
        __syncthreads();
        STAGE_QK(kt * KT);
        STAGE_N(kt * KT);
        for (int e = tid; e < LP * LK; e += nthreads) sDqx[e] = 0.0f;
        __syncthreads();
        for (int tile = warp; tile < mtile * ktile; tile += nwarps) {
            int m0 = (tile / ktile) << 4, n0 = (tile % ktile) << 3;
            float dqi[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Σ_j dS[t][j]·K[j][q]
            float dki[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Σ_t dS[t][j]·Q[t][q]
            unsigned a[4], b[2];
            for (int k0 = 0; k0 < LP; k0 += 16) {
                ldb_a_mk(a, sDSb, LDb, m0, k0);   // dS as [t, j], contracting j
                ldb_b_kn(b, sK, LQb, k0, n0);
                mma_bf16(dqi, a, b);
                ldb_a_km(a, sDSb, LDb, m0, k0);   // dSᵀ, contracting t
                ldb_b_kn(b, sQ, LQb, k0, n0);
                mma_bf16(dki, a, b);
            }
            for (int i = 0; i < 4; ++i) {
                int r = m0 + mma_row(i), qs = n0 + mma_col(i);
                sDqi[r * LK + qs] = dqi[i];
                sDki[r * LK + qs] = dki[i];
            }
        }
        for (int vt = 0; vt < NVT; ++vt) {
            int v0 = vt * VT;
            __syncthreads();
            STAGE_V(v0);
            STAGE_STATE(sC, cst, k, v0, kt * KT);
            STAGE_STATE(sdC, dcst, k + 1, v0, kt * KT);
            __syncthreads();
            for (int tile = warp; tile < mtile * ktile; tile += nwarps) {
                int m0 = (tile / ktile) << 4, n0 = (tile % ktile) << 3;
                float dqx[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Σ_v dN[t][v]·C[v][q]
                float dks[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Σ_v V[j][v]·dC[v][q]
                unsigned a[4], b[2];
                for (int k0 = 0; k0 < VT; k0 += 16) {
                    ldb_a_mk(a, sDN, LVb, m0, k0);
                    ldb_b_kn(b, sC, LQb, k0, n0);
                    mma_bf16(dqx, a, b);
                    ldb_a_mk(a, sV, LVb, m0, k0);
                    ldb_b_kn(b, sdC, LQb, k0, n0);
                    mma_bf16(dks, a, b);
                }
                for (int i = 0; i < 4; ++i) {
                    int r = m0 + mma_row(i), qs = n0 + mma_col(i);
                    sDqx[r * LK + qs] += dqx[i];
                    sDki[r * LK + qs] += sA[r] * dks[i];
                }
            }
            for (int e = tid; e < VT * KT; e += nthreads) {
                int vs = e / KT, q = e - vs * KT;
                dgloc += from_bf16(sdC[vs * LQb + q]) * from_bf16(sC[vs * LQb + q]);
            }
        }
        __syncthreads();
        for (int r = warp; r < len; r += nwarps) {
            float db = 0.0f;
            for (int qs = lane; qs < KT; qs += 32)
                db += from_bf16(sQ[r * LQb + qs]) * sDqx[r * LK + qs];
            db = warp_sum(db);
            if (lane == 0) sDb[r] += db;
        }
        for (int e = tid; e < LP * KT; e += nthreads) {
            int r = e / KT, qs = e - r * KT, q = kt * KT + qs;
            if (r < len && q < dqk) {
                long base = Q_BASE(bh) + (long)(c0 + r) * sX_S + q;
                dqkv[base] =
                    sDqi[r * LK + qs] + sB[r] * (sDqx[r * LK + qs] + sDQn[r] * sN[qs]);
                dqkv[base + MLSTM_OFF_K] =
                    sDki[r * LK + qs] + sA[r] * sdN[qs];
            }
        }
    }
    __syncthreads();

    sRed[tid] = dgloc;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sRed[tid] += sRed[tid + s];
        __syncthreads();
    }

    for (int j = tid; j < len; j += nthreads) {
        float pa = sDa[j] * sA[j];
        sDig[j] += pa;
        sDfc[j] += sDb[j] * sB[j] - pa;
    }
    __syncthreads();

    if (warp == 0) {
        const int lane = tid;
        const float dfc = (lane < len) ? sDfc[lane] : 0.0f;
        float da_a = (lane < len) ? sDa[lane] * sA[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            da_a += __shfl_xor_sync(0xffffffffu, da_a, off);
        float acc = dfc + ((lane == len - 1) ? (da_a + sRed[0] * gsca) : 0.0f);
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const float v = __shfl_down_sync(0xffffffffu, acc, off);
            if (lane + off < 32) acc += v;
        }
        if (lane < len) {
            long gi = IG_BASE(bh) + (long)(c0 + lane) * sG_S;
            long gf = FG_BASE(bh) + (long)(c0 + lane) * sG_S;
            dgates[gf] = acc * (1.0f - stable_sigmoid(gates[gf]));
            dgates[gi] = sDig[lane];
        }
    }
    #undef STAGE_QK
    #undef STAGE_STATE
    #undef STAGE_V
}

#endif // MMA_TF32
