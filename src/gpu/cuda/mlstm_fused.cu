// mLSTM chunkwise, FUSED (TFLA — after nx-ai/mlstm_kernels): a whole sequence in
// three launches forward and two backward. The MMA_TF32 fences hold the tensor-core
// twins, compiled only for sm_80+ against the real device arch.
//
// The op-at-a-time path in `gpu::mlstm` runs this same math as ~25 launches per
// chunk inside a host-side chunk loop: ~600 launches for one fwd+bwd at the
// backbone's shape, which is ~1 ms of arithmetic stretched over 14 ms of driver
// latency (see `examples/mlstm_stage_prof.rs`). These kernels do a whole sequence
// in three launches forward and two backward, and the [L, L] decay matrix never
// reaches HBM — it lives in shared memory for the lifetime of a block.
//
// Notation follows the reference. Within a chunk of length `len` (<= L):
//   fc[j]      = cumsum_{j'<=j} logsigmoid(f[j'])           (their vecB)
//   logD[t][j] = fc[t] - fc[j] + i[j]   for j <= t          (their matD)
//   m[t]       = max( max_{j<=t} logD[t][j], fc[t] + m_prev )
//   b[t]       = exp(fc[t] + m_prev - m[t])                 (their vecBbar)
//   a[j]       = exp(fc[last] - fc[j] + i[j] - m[last])     (their vecAbar)
//   g          = exp(fc[last] + m_prev - m[last])           (their scaGbar)
// which is exactly the (fc, m, bvec, avec) the op-at-a-time path builds, so the
// two agree elementwise — `mlstm_fused_matches_legacy` pins them together.
//
// The only sequential axis is the CHUNK STATE (C, n, m), and it is small. So the
// work splits into a serial-over-chunks kernel that carries a [dhv, dqk] state
// and does no per-timestep launch, and a parallel-over-chunks kernel that holds
// all the FLOPs and is embarrassingly parallel:
//   mlstm_fw_gates    -> fc, per chunk, independent
//   mlstm_fw_C        -> the chunk states, looping chunks INSIDE the kernel
//   mlstm_fw_parallel -> every chunk independently, one block each
// Backward mirrors it (mlstm_bw_dC walks chunks in reverse, mlstm_bw_parallel is
// per-chunk). The last chunk may be short and is masked by `len` everywhere.
//
// LAYOUT: q/k/v and the gates are addressed through explicit strides, not a fixed
// packing, following the reference (nx-ai/mlstm_kernels passes str_matQK_B_NH /
// _S / _DHQK and reads them off the tensor). The batch and head axes are strided
// SEPARATELY — the flat `bh = b*H + h` a block owns is split back into `b` and `h`,
// so element (b, h, t, c) of a q/k-shaped tensor sits at
// `b*sQK_B + h*sQK_H + t*sQK_S + c`, and of a v/h-shaped one at the `sHV_*` twin.
// The innermost stride is 1 on every layout used here and is not passed.
//
// Splitting b from h is what makes position-major work at B > 1: there the row
// index is `b*T + t`, so the batch and timestep strides are NOT proportional the
// way a single fused `bh` stride would force them to be.
//   head-major     [BH, T, W]        -> sB = H*T*W, sH = T*W, sS = W
//   position-major [N, H*W], N = B*T -> sB = T*H*W, sH = W,   sS = H*W
// The second is what the projections produce, so feeding them straight in costs no
// reorg pass. Loads stay coalesced either way: the fast axis `c` is contiguous, so
// a warp still covers consecutive floats of one timestep — only the distance
// between timesteps changes.

// Base offset of the (b, h) a block owns, from its flat `bh = b*H + h`. Each tensor
// group (q/k, v/h, gates) passes its own stride pair.
__device__ __forceinline__ long bhBase(int bh, int H, long sB, long sH) {
    return (long)(bh / H) * sB + (long)(bh % H) * sH;
}

// fc: the chunk-local cumulative log-forget. One thread per (bh, chunk) — the
// scan is serial but L is tiny and there are BH*NC of them. Positions past `len`
// hold the last valid prefix (they are always masked out by a `j < len` guard,
// but leaving them undefined would poison the exp()s below).
extern "C" __global__ void mlstm_fw_gates(const float* fg, float* fcb,
                                          int T, int L, int NC, int BH, int H,
                                          long sG_B, long sG_H, long sG_S) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BH * NC) return;
    int k = idx % NC, bh = idx / NC;
    int c0 = k * L;
    int len = min(L, T - c0);
    float acc = 0.0f;
    float* out = fcb + (long)(bh * NC + k) * L;
    const float* fgp = fg + bhBase(bh, H, sG_B, sG_H) + (long)c0 * sG_S;
    for (int j = 0; j < L; ++j) {
        if (j < len) acc += log_sigmoid(fgp[(long)j * sG_S]);
        out[j] = acc;
    }
}

// The chunk states. ONE block per (bh); the chunk loop is inside the kernel, so
// the serial dependency costs iterations, not launches.
//   C_k = g·C_{k-1} + Σ_j a[j]·V[j]⊗K[j]      (C is [dhv, dqk])
//   n_k = g·n_{k-1} + Σ_j a[j]·K[j]
//   m_k = max(g_exp + m_{k-1}, max_j a_exp[j])
// State k (the state ENTERING chunk k) is published at index k, so index 0 is the
// zero initial state and index NC is the final one.
//
// m is carried in a REGISTER, not shared memory: every thread derives it from the
// same block-wide max reduction, so they all hold the identical value and no
// broadcast is needed.
//
// Every 2D shared array is stored with its row stride PADDED by one float (LQ, LV,
// LS below). Without that, a row stride of dqk = 64 floats puts `s[row*64 + c]` on
// bank (row*64 + c) % 32 = c % 32 for every row — so the warps below, which walk
// `row` across threads at fixed `c`, would hit one bank 32 ways and serialize. The
// pad makes consecutive rows land on consecutive banks. This is the single biggest
// factor in these kernels; measured, it is worth ~4x.
// This kernel and `mlstm_bw_dC` are TILED over the value dimension: grid is
// (ceil(dhv/TV), BH), and a block owns the `v` slice [v0, v0+tv) of the state.
// Untiled, the grid would be BH alone — 8 blocks at the backbone's shape (B=1,
// 8 heads), i.e. 8 of the GPU's 48 SMs doing a 64x64 state update over every chunk
// in sequence. Slicing `v` is free: C's update is an outer product, so row v of C
// only ever needs column v of V, and only the `n` update (which does not depend on
// v at all) has to be assigned to a single tile — tile 0.
extern "C" __global__ void mlstm_fw_C(const slab_t* kk, const slab_t* vv, const float* ig,
                                      const float* fcb,
                                      float* cst, float* nst, float* mst,
                                      int T, int L, int NC, int dqk, int dhv, int TV,
                                      int CARRY, int H,
                                      long sQK_B, long sQK_H, long sQK_S,
                                      long sHV_B, long sHV_H, long sHV_S,
                                      long sG_B, long sG_H, long sG_S) {
    int v0 = blockIdx.x * TV, bh = blockIdx.y;
    int tv = min(TV, dhv - v0);
    int tid = threadIdx.x, nthreads = blockDim.x;
    int LQ = dqk + 1, LV = tv + 1;
    int lead = (blockIdx.x == 0); // the tile that also owns `n` and `m`

    extern __shared__ float sh[];
    float* sK  = sh;                  // [L, LQ]
    float* sV  = sK + L * LQ;         // [L, LV]   the v-slice only
    float* sC  = sV + L * LV;         // [tv, LQ]
    float* sN  = sC + tv * LQ;        // [dqk]
    float* sFc = sN + dqk;            // [L]
    float* sIg = sFc + L;             // [L]
    float* sA  = sIg + L;             // [L]
    __shared__ float sRed[256];

    // Initial state. Normally zero — a forward is a whole sequence. When CARRY is set
    // the caller has staged the state this call continues from in `cst`/`nst`/`mst` at
    // index 0 (the same slots the loop below publishes into), so a chunked sweep
    // reproduces the unchunked recurrence exactly instead of resetting at every chunk
    // border. `m` is the stabilizer: seeding it wrong does not crash, it silently
    // rescales the whole chunk, which is why `mlstm_chunked_carry_matches_whole`
    // checks it against a single-call reference.
    const float* cin = cst + ((long)bh * (NC + 1)) * dhv * dqk;
    const float* nin = nst + ((long)bh * (NC + 1)) * dqk;
    for (int e = tid; e < tv * dqk; e += nthreads) {
        int v = e / dqk, q = e - v * dqk;
        sC[v * LQ + q] = CARRY ? cin[(v0 + v) * dqk + q] : 0.0f;
    }
    for (int e = tid; e < dqk; e += nthreads) sN[e] = CARRY ? nin[e] : 0.0f;
    float m_run = CARRY ? mst[(long)bh * (NC + 1)] : 0.0f;
    __syncthreads();

    for (int k = 0; k < NC; ++k) {
        int c0 = k * L;
        int len = min(L, T - c0);

        // Publish the state entering chunk k. Each element is owned by one thread
        // for the whole kernel, so reading it here and updating it below is a
        // same-thread sequence — no barrier needed between the two.
        float* cout = cst + ((long)bh * (NC + 1) + k) * dhv * dqk;
        for (int e = tid; e < tv * dqk; e += nthreads) {
            int v = e / dqk, q = e - v * dqk;
            cout[(long)(v0 + v) * dqk + q] = sC[v * LQ + q];
        }
        if (lead) {
            float* nout = nst + ((long)bh * (NC + 1) + k) * dqk;
            for (int e = tid; e < dqk; e += nthreads) nout[e] = sN[e];
            if (tid == 0) mst[(long)bh * (NC + 1) + k] = m_run;
        }

        for (int j = tid; j < L; j += nthreads) {
            sFc[j] = fcb[((long)bh * NC + k) * L + j];
            sIg[j] = (j < len) ? ig[bhBase(bh, H, sG_B, sG_H) + (long)(c0 + j) * sG_S] : 0.0f;
        }
        for (int e = tid; e < len * dqk; e += nthreads) {
            int j = e / dqk, q = e - j * dqk;
            sK[j * LQ + q] = slab_ld(kk, bhBase(bh, H, sQK_B, sQK_H) + (long)(c0 + j) * sQK_S + q);
        }
        for (int e = tid; e < len * tv; e += nthreads) {
            int j = e / tv, v = e - j * tv;
            sV[j * LV + v] = slab_ld(vv, bhBase(bh, H, sHV_B, sHV_H) + (long)(c0 + j) * sHV_S + v0 + v);
        }
        __syncthreads();

        float fc_last = sFc[len - 1];

        // m_new = max( max_j (fc_last - fc_j + ig_j), fc_last + m_prev ). Every tile
        // recomputes it — it is a handful of flops, and the alternative is a grid-wide
        // dependency between blocks.
        float local = -1e30f;
        for (int j = tid; j < len; j += nthreads)
            local = fmaxf(local, fc_last - sFc[j] + sIg[j]);
        sRed[tid] = local;
        __syncthreads();
        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) sRed[tid] = fmaxf(sRed[tid], sRed[tid + s]);
            __syncthreads();
        }
        float m_new = fmaxf(sRed[0], fc_last + m_run);
        float gbar = expf(fc_last + m_run - m_new);

        for (int j = tid; j < L; j += nthreads)
            sA[j] = (j < len) ? expf(fc_last - sFc[j] + sIg[j] - m_new) : 0.0f;
        __syncthreads();

        for (int e = tid; e < tv * dqk; e += nthreads) {
            int v = e / dqk, q = e - v * dqk;
            float acc = 0.0f;
            for (int j = 0; j < len; ++j) acc += sA[j] * sV[j * LV + v] * sK[j * LQ + q];
            sC[v * LQ + q] = gbar * sC[v * LQ + q] + acc;
        }
        if (lead) {
            for (int q = tid; q < dqk; q += nthreads) {
                float acc = 0.0f;
                for (int j = 0; j < len; ++j) acc += sA[j] * sK[j * LQ + q];
                sN[q] = gbar * sN[q] + acc;
            }
        }
        m_run = m_new;
        __syncthreads();
    }

    float* cout = cst + ((long)bh * (NC + 1) + NC) * dhv * dqk;
    for (int e = tid; e < tv * dqk; e += nthreads) {
        int v = e / dqk, q = e - v * dqk;
        cout[(long)(v0 + v) * dqk + q] = sC[v * LQ + q];
    }
    if (lead) {
        float* nout = nst + ((long)bh * (NC + 1) + NC) * dqk;
        for (int e = tid; e < dqk; e += nthreads) nout[e] = sN[e];
        if (tid == 0) mst[(long)bh * (NC + 1) + NC] = m_run;
    }
}

// One block per (chunk, bh) — all chunks at once. Intra-chunk attention plus the
// read-out of the incoming state:
//   num[t] = Σ_j (D̄⊙S)[t][j]·V[j] + b[t]·(Q[t]·C_prevᵀ)
//   qn[t]  = Σ_j (D̄⊙S)[t][j]      + b[t]·(Q[t]·n_prev)
//   ỹ[t]   = num[t] / max(|qn[t]|, 1)
// Chunk 0 needs no special case: its incoming state is zero, so the inter terms
// vanish on their own.
extern "C" __global__ void mlstm_fw_parallel(
    const slab_t* qq, const slab_t* kk, const slab_t* vv, const float* ig, const float* fcb,
    const float* cst, const float* nst, const float* mst,
    slab_t* ytil, float* msv, float* psiv, float* qnv,
    int T, int L, int NC, int dqk, int dhv, int H,
    long sQK_B, long sQK_H, long sQK_S, long sHV_B, long sHV_H, long sHV_S,
    long sG_B, long sG_H, long sG_S) {
    int k = blockIdx.x, bh = blockIdx.y;
    int tid = threadIdx.x, nthreads = blockDim.x;
    int c0 = k * L;
    int len = min(L, T - c0);
    int LQ = dqk + 1, LV = dhv + 1, LS = L + 1;

    extern __shared__ float sh[];
    float* sQ  = sh;                  // [L, LQ]
    float* sK  = sQ + L * LQ;         // [L, LQ]
    float* sV  = sK + L * LQ;         // [L, LV]
    float* sC  = sV + L * LV;         // [dhv, LQ]
    float* sDS = sC + dhv * LQ;       // [L, LS]
    float* sN  = sDS + L * LS;        // [dqk]
    float* sFc = sN + dqk;            // [L]
    float* sIg = sFc + L;             // [L]
    float* sM  = sIg + L;             // [L]
    float* sB  = sM + L;              // [L]
    float* sQn = sB + L;              // [L]

    for (int e = tid; e < len * dqk; e += nthreads) {
        int t = e / dqk, q = e - t * dqk;
        long off = bhBase(bh, H, sQK_B, sQK_H) + (long)(c0 + t) * sQK_S + q;
        sQ[t * LQ + q] = slab_ld(qq, off);
        sK[t * LQ + q] = slab_ld(kk, off);
    }
    for (int e = tid; e < len * dhv; e += nthreads) {
        int t = e / dhv, v = e - t * dhv;
        sV[t * LV + v] = slab_ld(vv, bhBase(bh, H, sHV_B, sHV_H) + (long)(c0 + t) * sHV_S + v);
    }
    for (int e = tid; e < dhv * dqk; e += nthreads) {
        int v = e / dqk, q = e - v * dqk;
        sC[v * LQ + q] = cst[((long)bh * (NC + 1) + k) * dhv * dqk + e];
    }
    for (int e = tid; e < dqk; e += nthreads)
        sN[e] = nst[((long)bh * (NC + 1) + k) * dqk + e];
    for (int j = tid; j < L; j += nthreads) {
        sFc[j] = fcb[((long)bh * NC + k) * L + j];
        sIg[j] = (j < len) ? ig[bhBase(bh, H, sG_B, sG_H) + (long)(c0 + j) * sG_S] : 0.0f;
    }
    float m_prev = mst[(long)bh * (NC + 1) + k];
    __syncthreads();

    for (int t = tid; t < len; t += nthreads) {
        float fct = sFc[t];
        float mx = fct + m_prev;
        for (int j = 0; j <= t; ++j) mx = fmaxf(mx, fct - sFc[j] + sIg[j]);
        sM[t] = mx;
        sB[t] = expf(fct + m_prev - mx);
    }
    __syncthreads();

    // DS = D̄ ⊙ (Q·Kᵀ), the whole [len, len] block, kept in shared memory.
    for (int e = tid; e < len * len; e += nthreads) {
        int t = e / len, j = e - t * len;
        float val = 0.0f;
        if (j <= t) {
            float s = 0.0f;
            for (int q = 0; q < dqk; ++q) s += sQ[t * LQ + q] * sK[j * LQ + q];
            val = expf(sFc[t] - sFc[j] + sIg[j] - sM[t]) * s;
        }
        sDS[t * LS + j] = val;
    }
    __syncthreads();

    for (int t = tid; t < len; t += nthreads) {
        float acc = 0.0f;
        for (int j = 0; j <= t; ++j) acc += sDS[t * LS + j];
        float qi = 0.0f;
        for (int q = 0; q < dqk; ++q) qi += sQ[t * LQ + q] * sN[q];
        acc += sB[t] * qi;
        sQn[t] = acc;
        long gt = (long)bh * T + c0 + t;
        qnv[gt] = acc;
        msv[gt] = sM[t];
        // ψ = max(|qn|, exp(−m)) — qn is the stabilized normalizer. See `nn2::mlstm`.
        psiv[gt] = fmaxf(fabsf(acc), expf(-sM[t]));
    }
    __syncthreads();

    for (int e = tid; e < len * dhv; e += nthreads) {
        int t = e / dhv, v = e - t * dhv;
        float acc = 0.0f;
        for (int j = 0; j <= t; ++j) acc += sDS[t * LS + j] * sV[j * LV + v];
        float inter = 0.0f;
        for (int q = 0; q < dqk; ++q) inter += sQ[t * LQ + q] * sC[v * LQ + q];
        acc += sB[t] * inter;
        slab_st(ytil, bhBase(bh, H, sHV_B, sHV_H) + (long)(c0 + t) * sHV_S + v,
                acc / fmaxf(fabsf(sQn[t]), expf(-sM[t])));
    }
}
// Tensor-core dot (MMA_TF32, sm_80+)
//
// The scalar kernels above give every output element to one thread, which then
// walks the contraction with an FMA loop. That is the one thing our chunkwise core
// does differently from the reference (nx-ai/mlstm_kernels): every contraction
// there is a `tl.dot`, and Triton lowers `tl.dot` on fp32 inputs to the tensor
// cores in TF32 (`allow_tf32` defaults to true). Below is that same `dot`, written
// out as the PTX Triton would emit.
//
// The unit is `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`: a whole WARP
// cooperates to compute D(16x8) += A(16x8)·B(8x8), with the operands rounded to
// TF32 (fp32's 8 exponent bits, 10 mantissa bits) and the product accumulated in
// full fp32. Precision-wise it sits exactly where cuBLAS's TF32 math mode does,
// and where the reference already is.
//
// A warp's 32 lanes each hold a fixed slice of every fragment. With
//   g = lane / 4   (the "group")      c = lane % 4   (the index within it)
// the layouts the instruction requires are:
//   A (16x8, row): a0=(g, c)   a1=(g+8, c)   a2=(g, c+4)   a3=(g+8, c+4)
//   B (8x8, col):  b0=(c, g)   b1=(c+4, g)                  [(row=k, col=n)]
//   D (16x8):      d0=(g, 2c)  d1=(g, 2c+1)  d2=(g+8, 2c)  d3=(g+8, 2c+1)
// The `ld_*` helpers below are just those tables, applied to a shared-memory tile.
// Nothing here is allowed to diverge inside a warp — `mma.sync` is warp-wide.
#if MMA_TF32

__device__ __forceinline__ unsigned tf32_of(float x) {
    unsigned r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}

// D += A·B for one warp. Accumulates in place, so a K-loop just calls it again.
__device__ __forceinline__ void mma_16x8x8(float* d, const unsigned* a, const unsigned* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// A from a row-major [M, K] tile: A[m][k] = s[m*ld + k].
__device__ __forceinline__ void ld_a_mk(unsigned* a, const float* s, int ld, int m0, int k0) {
    int lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    a[0] = tf32_of(s[(m0 + g)     * ld + k0 + c]);
    a[1] = tf32_of(s[(m0 + g + 8) * ld + k0 + c]);
    a[2] = tf32_of(s[(m0 + g)     * ld + k0 + c + 4]);
    a[3] = tf32_of(s[(m0 + g + 8) * ld + k0 + c + 4]);
}

// A from a row-major [K, M] tile, i.e. Aᵀ is what is in memory: A[m][k] = s[k*ld + m].
// This is how `dS` is contracted over `t` for dK without ever transposing it.
__device__ __forceinline__ void ld_a_km(unsigned* a, const float* s, int ld, int m0, int k0) {
    int lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    a[0] = tf32_of(s[(k0 + c)     * ld + m0 + g]);
    a[1] = tf32_of(s[(k0 + c)     * ld + m0 + g + 8]);
    a[2] = tf32_of(s[(k0 + c + 4) * ld + m0 + g]);
    a[3] = tf32_of(s[(k0 + c + 4) * ld + m0 + g + 8]);
}

// B from a row-major [N, K] tile: B[k][n] = s[n*ld + k]. (`Q·Kᵀ`: K is stored
// [j, q] and is wanted as [q, j].)
__device__ __forceinline__ void ld_b_nk(unsigned* b, const float* s, int ld, int k0, int n0) {
    int lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    b[0] = tf32_of(s[(n0 + g) * ld + k0 + c]);
    b[1] = tf32_of(s[(n0 + g) * ld + k0 + c + 4]);
}

// B from a row-major [K, N] tile: B[k][n] = s[k*ld + n]. (`(D̄⊙S)·V`.)
__device__ __forceinline__ void ld_b_kn(unsigned* b, const float* s, int ld, int k0, int n0) {
    int lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    b[0] = tf32_of(s[(k0 + c)     * ld + n0 + g]);
    b[1] = tf32_of(s[(k0 + c + 4) * ld + n0 + g]);
}

// Where accumulator register `i` lands in the 16x8 output tile.
__device__ __forceinline__ int mma_row(int i) { return ((threadIdx.x & 31) >> 2) + ((i & 2) ? 8 : 0); }
__device__ __forceinline__ int mma_col(int i) { return (((threadIdx.x & 31) & 3) << 1) + (i & 1); }

// The tensor-core twin of `mlstm_fw_parallel`. Same algorithm, same shared-memory
// plan, same numbers up to TF32 rounding of the three contractions:
//   S    = Q·Kᵀ        (over dqk)   -> masked/decayed in the mma epilogue
//   H    = (D̄⊙S)·V    (over j)
//   Hinter = Q·C_prevᵀ (over dqk)   -> scaled by b[t] and added to H
// The decay mask is applied as the epilogue of the first mma rather than in a pass
// of its own, so `S` never lands in shared memory unmasked.
//
// Shapes are padded UP to the mma tile (rows to 16, contractions to 8, columns to
// 8) and the pad is zero-filled, so a short last chunk, an odd `dqk` or an odd
// `dhv` all fall out for free: a zero row contributes nothing to a dot, and the
// out-of-range outputs are simply not written. `LP`/`KP`/`VP` are those padded
// dims and must match `fused_smem("fw_parallel_mma", ..)` on the host exactly.
extern "C" __global__ void mlstm_fw_parallel_mma(
    const slab_t* qq, const slab_t* kk, const slab_t* vv, const float* ig, const float* fcb,
    const float* cst, const float* nst, const float* mst,
    slab_t* ytil, float* msv, float* psiv, float* qnv,
    int T, int L, int NC, int dqk, int dhv, int H,
    long sQK_B, long sQK_H, long sQK_S, long sHV_B, long sHV_H, long sHV_S,
    long sG_B, long sG_H, long sG_S) {
    int k = blockIdx.x, bh = blockIdx.y;
    int tid = threadIdx.x, nthreads = blockDim.x;
    int warp = tid >> 5, nwarps = nthreads >> 5;
    int c0 = k * L;
    int len = min(L, T - c0);

    int LP = (L + 15) & ~15;    // rows      -> multiple of the mma M
    int KP = (dqk + 7) & ~7;    // dqk       -> multiple of the mma K
    int VP = (dhv + 7) & ~7;    // dhv       -> multiple of the mma N
    int LQ = KP + 1, LV = VP + 1, LS = LP + 1;   // +1: the bank-conflict pad

    extern __shared__ float sh[];
    float* sQ  = sh;                  // [LP, LQ]
    float* sK  = sQ + LP * LQ;        // [LP, LQ]
    float* sV  = sK + LP * LQ;        // [LP, LV]
    float* sC  = sV + LP * LV;        // [VP, LQ]
    float* sDS = sC + VP * LQ;        // [LP, LS]
    float* sN  = sDS + LP * LS;       // [KP]
    float* sFc = sN + KP;             // [LP]
    float* sIg = sFc + LP;            // [LP]
    float* sM  = sIg + LP;            // [LP]
    float* sB  = sM + LP;             // [LP]
    float* sQn = sB + LP;             // [LP]

    for (int e = tid; e < LP * KP; e += nthreads) {
        int t = e / KP, q = e - t * KP;
        int ok = (t < len) && (q < dqk);
        sQ[t * LQ + q] = ok ? slab_ld(qq, bhBase(bh, H, sQK_B, sQK_H) + (long)(c0 + t) * sQK_S + q) : 0.0f;
        sK[t * LQ + q] = ok ? slab_ld(kk, bhBase(bh, H, sQK_B, sQK_H) + (long)(c0 + t) * sQK_S + q) : 0.0f;
    }
    for (int e = tid; e < LP * VP; e += nthreads) {
        int t = e / VP, v = e - t * VP;
        sV[t * LV + v] = ((t < len) && (v < dhv))
            ? slab_ld(vv, bhBase(bh, H, sHV_B, sHV_H) + (long)(c0 + t) * sHV_S + v) : 0.0f;
    }
    for (int e = tid; e < VP * KP; e += nthreads) {
        int v = e / KP, q = e - v * KP;
        sC[v * LQ + q] = ((v < dhv) && (q < dqk))
            ? cst[((long)bh * (NC + 1) + k) * dhv * dqk + (long)v * dqk + q] : 0.0f;
    }
    for (int e = tid; e < KP; e += nthreads)
        sN[e] = (e < dqk) ? nst[((long)bh * (NC + 1) + k) * dqk + e] : 0.0f;
    for (int j = tid; j < LP; j += nthreads) {
        sFc[j] = (j < L)   ? fcb[((long)bh * NC + k) * L + j] : 0.0f;
        sIg[j] = (j < len) ? ig[bhBase(bh, H, sG_B, sG_H) + (long)(c0 + j) * sG_S]        : 0.0f;
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
    __syncthreads();

    // S = Q·Kᵀ, with D̄ and the causal mask folded into the epilogue. Rows/cols in
    // the pad get 0, which is what the next dot needs anyway.
    int mtile = LP >> 4, ntile = LP >> 3;
    for (int tile = warp; tile < mtile * ntile; tile += nwarps) {
        int m0 = (tile / ntile) << 4, n0 = (tile % ntile) << 3;
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        unsigned a[4], b[2];
        for (int k0 = 0; k0 < KP; k0 += 8) {
            ld_a_mk(a, sQ, LQ, m0, k0);
            ld_b_nk(b, sK, LQ, k0, n0);
            mma_16x8x8(d, a, b);
        }
        for (int i = 0; i < 4; ++i) {
            int t = m0 + mma_row(i), j = n0 + mma_col(i);
            float val = 0.0f;
            if (t < len && j <= t) val = expf(sFc[t] - sFc[j] + sIg[j] - sM[t]) * d[i];
            sDS[t * LS + j] = val;
        }
    }
    __syncthreads();

    // qn: the row sum of D̄⊙S plus the b·(Q·n_prev) read-out. A matrix-VECTOR
    // product, so it stays scalar — there is no dot for the tensor cores here.
    for (int t = tid; t < len; t += nthreads) {
        float acc = 0.0f;
        for (int j = 0; j <= t; ++j) acc += sDS[t * LS + j];
        float qi = 0.0f;
        for (int q = 0; q < dqk; ++q) qi += sQ[t * LQ + q] * sN[q];
        acc += sB[t] * qi;
        sQn[t] = acc;
        long gt = (long)bh * T + c0 + t;
        qnv[gt] = acc;
        msv[gt] = sM[t];
        psiv[gt] = fmaxf(fabsf(acc), expf(-sM[t]));
    }
    __syncthreads();

    // The two output dots share an output tile, so they share a warp: intra over j,
    // inter over q, combined as num = intra + b[t]·inter in the epilogue.
    int vtile = VP >> 3;
    for (int tile = warp; tile < mtile * vtile; tile += nwarps) {
        int m0 = (tile / vtile) << 4, n0 = (tile % vtile) << 3;
        float dintra[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float dinter[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        unsigned a[4], b[2];
        for (int k0 = 0; k0 < LP; k0 += 8) {   // (D̄⊙S)·V, contracting j
            ld_a_mk(a, sDS, LS, m0, k0);
            ld_b_kn(b, sV, LV, k0, n0);
            mma_16x8x8(dintra, a, b);
        }
        for (int k0 = 0; k0 < KP; k0 += 8) {   // Q·C_prevᵀ, contracting q
            ld_a_mk(a, sQ, LQ, m0, k0);
            ld_b_nk(b, sC, LQ, k0, n0);
            mma_16x8x8(dinter, a, b);
        }
        for (int i = 0; i < 4; ++i) {
            int t = m0 + mma_row(i), v = n0 + mma_col(i);
            if (t < len && v < dhv) {
                float acc = dintra[i] + sB[t] * dinter[i];
                slab_st(ytil, bhBase(bh, H, sHV_B, sHV_H) + (long)(c0 + t) * sHV_S + v,
                        acc / fmaxf(fabsf(sQn[t]), expf(-sM[t])));
            }
        }
    }
}

#endif // MMA_TF32

// Backward over the chunk states: the mirror of `mlstm_fw_C`, walking chunks in
// reverse with the chunk loop inside the kernel. `dcst[k]` is the gradient wrt the
// state ENTERING chunk k (so it lines up index-for-index with `cst`):
//   dcst[k] = g_k · dcst[k+1] + Σ_t b_k[t]·d_num_k[t]⊗Q_k[t]
// dcst[NC] is zero — the state the last chunk produces is never read.
extern "C" __global__ void mlstm_bw_dC(
    const slab_t* qq, const float* dytil, const slab_t* ytil,
    const float* psiv, const float* qnv, const float* msv,
    const float* fcb, const float* mst,
    float* dcst, float* dnst,
    int T, int L, int NC, int dqk, int dhv, int TV, int CARRY, int H,
    long sQK_B, long sQK_H, long sQK_S, long sHV_B, long sHV_H, long sHV_S) {
    int v0 = blockIdx.x * TV, bh = blockIdx.y;
    int tv = min(TV, dhv - v0);
    int tid = threadIdx.x, nthreads = blockDim.x;
    int LQ = dqk + 1, LV = tv + 1;
    int lead = (blockIdx.x == 0); // the tile that also owns `dn`

    extern __shared__ float sh[];
    float* sQ   = sh;                 // [L, LQ]
    float* sDN  = sQ + L * LQ;        // [L, LV]   d_num, the v-slice only
    float* sdC  = sDN + L * LV;       // [tv, LQ]
    float* sdN  = sdC + tv * LQ;      // [dqk]
    float* sB   = sdN + dqk;          // [L]
    float* sDQn = sB + L;             // [L]

    // Incoming BPTT state. Normally zero — the state the last chunk produces is never
    // read. Under CARRY the caller has staged the gradient flowing back from the chunk
    // to the RIGHT into slot NC (the slot this kernel would otherwise publish first),
    // so a chunked backward matches the unchunked one. Mirrors `mlstm_fw_C`'s CARRY.
    const float* dcin = dcst + ((long)bh * (NC + 1) + NC) * dhv * dqk;
    const float* dnin = dnst + ((long)bh * (NC + 1) + NC) * dqk;
    for (int e = tid; e < tv * dqk; e += nthreads) {
        int v = e / dqk, q = e - v * dqk;
        sdC[v * LQ + q] = CARRY ? dcin[(long)(v0 + v) * dqk + q] : 0.0f;
    }
    for (int e = tid; e < dqk; e += nthreads) sdN[e] = CARRY ? dnin[e] : 0.0f;
    __syncthreads();

    for (int k = NC - 1; k >= 0; --k) {
        int c0 = k * L;
        int len = min(L, T - c0);

        // What is in sdC right now is the gradient wrt C_k — i.e. wrt the state
        // entering chunk k+1. Publish it there before folding chunk k in.
        float* dcout = dcst + ((long)bh * (NC + 1) + (k + 1)) * dhv * dqk;
        for (int e = tid; e < tv * dqk; e += nthreads) {
            int v = e / dqk, q = e - v * dqk;
            dcout[(long)(v0 + v) * dqk + q] = sdC[v * LQ + q];
        }
        if (lead) {
            float* dnout = dnst + ((long)bh * (NC + 1) + (k + 1)) * dqk;
            for (int e = tid; e < dqk; e += nthreads) dnout[e] = sdN[e];
        }

        float m_prev = mst[(long)bh * (NC + 1) + k];
        float fc_last = fcb[((long)bh * NC + k) * L + (len - 1)];
        float m_last = msv[(long)bh * T + c0 + len - 1];
        float gk = expf(fc_last + m_prev - m_last);

        for (int e = tid; e < len * dqk; e += nthreads) {
            int t = e / dqk, q = e - t * dqk;
            sQ[t * LQ + q] = slab_ld(qq, bhBase(bh, H, sQK_B, sQK_H) + (long)(c0 + t) * sQK_S + q);
        }

        // d_num = d_ytil/ψ and d_qn — the backward of ỹ = num/ψ, ψ = max(|qn|,1).
        // num is not saved: num = ỹ·ψ, so Σ_v d_ytil·num = ψ·Σ_v d_ytil·ỹ and the
        // ψ² cancels down to one division. d_qn contracts over ALL of dhv, so every
        // tile computes it (each reading the full d_ytil row) — only the v-slice of
        // d_num goes to shared memory.
        // One WARP per timestep rather than one thread: `len` is at most L (<= 32), so
        // a thread-per-t left all but `len` of the 256 threads idle while each of the
        // few active ones ran the whole `dhv` reduction alone. Splitting `red` across
        // the warp's lanes uses 32x more of the block and shortens the serial chain to
        // dhv/32 + a shuffle tree.
        {
            const int lane = tid & 31;
            const int warp = tid >> 5;
            const int nwarps = nthreads >> 5;
            for (int t = warp; t < len; t += nwarps) {
                long gt = (long)bh * T + c0 + t;
                long gy = bhBase(bh, H, sHV_B, sHV_H) + (long)(c0 + t) * sHV_S;
                float inv = 1.0f / psiv[gt];
                float red = 0.0f;
                for (int v = lane; v < dhv; v += 32)
                    red += dytil[gy + v] * slab_ld(ytil, gy + v);
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    red += __shfl_down_sync(0xffffffffu, red, off);
                red = __shfl_sync(0xffffffffu, red, 0);
                for (int v = lane; v < tv; v += 32)
                    sDN[t * LV + v] = dytil[gy + v0 + v] * inv;
                if (lane == 0) {
                    float dpsi = -red * inv;
                    float qn = qnv[gt];
                    // Grad flows through qn only where it, not the exp(-m) floor, won
                    // the max.
                    sDQn[t] = (fabsf(qn) > expf(-msv[gt]))
                                  ? ((qn > 0.0f ? 1.0f : -1.0f) * dpsi)
                                  : 0.0f;
                    sB[t] = expf(fcb[((long)bh * NC + k) * L + t] + m_prev - msv[gt]);
                }
            }
        }
        __syncthreads();

        for (int e = tid; e < tv * dqk; e += nthreads) {
            int v = e / dqk, q = e - v * dqk;
            float acc = 0.0f;
            for (int t = 0; t < len; ++t) acc += sB[t] * sDN[t * LV + v] * sQ[t * LQ + q];
            sdC[v * LQ + q] = gk * sdC[v * LQ + q] + acc;
        }
        if (lead) {
            for (int q = tid; q < dqk; q += nthreads) {
                float acc = 0.0f;
                for (int t = 0; t < len; ++t) acc += sB[t] * sDQn[t] * sQ[t * LQ + q];
                sdN[q] = gk * sdN[q] + acc;
            }
        }
        __syncthreads();
    }

    float* dcout = dcst + (long)bh * (NC + 1) * dhv * dqk;
    for (int e = tid; e < tv * dqk; e += nthreads) {
        int v = e / dqk, q = e - v * dqk;
        dcout[(long)(v0 + v) * dqk + q] = sdC[v * LQ + q];
    }
    if (lead) {
        float* dnout = dnst + (long)bh * (NC + 1) * dqk;
        for (int e = tid; e < dqk; e += nthreads) dnout[e] = sdN[e];
    }
}

// One block per (chunk, bh): everything a chunk owes its inputs, given the state
// gradients the two recurrent kernels produced. S and D̄ are RECOMPUTED here from
// (Q, K, fc, ig, m) rather than saved — that is what keeps the [L, L] matrices off
// HBM. The shared DS buffer is read as DS (for dV) and then overwritten in place
// with dS, since each (t, j) is owned by exactly one thread.
//
// The gate gradients close out in the same block: `a`, `b` and `g` are chunk-local,
// so dfc/dig need no cross-block reduction, and the reverse cumsum that turns dfc
// into the forget-logit grad is a within-chunk scan.
extern "C" __global__ void mlstm_bw_parallel(
    const slab_t* qq, const slab_t* kk, const slab_t* vv,
    const float* ig, const float* fg, const float* fcb,
    const float* cst, const float* nst, const float* mst,
    const float* dcst, const float* dnst,
    const slab_t* ytil, const float* dytil, const float* psiv,
    const float* qnv, const float* msv,
    float* dq, float* dk, float* dv, float* dig, float* dfg,
    int T, int L, int NC, int dqk, int dhv, int CARRY, int H,
    long sQK_B, long sQK_H, long sQK_S, long sHV_B, long sHV_H, long sHV_S,
    long sG_B, long sG_H, long sG_S) {
    int k = blockIdx.x, bh = blockIdx.y;
    int tid = threadIdx.x, nthreads = blockDim.x;
    int c0 = k * L;
    int len = min(L, T - c0);
    // The last chunk's outgoing state is only unread when this call IS the whole
    // sequence. Under CARRY it feeds the chunk to the right, whose gradient
    // `mlstm_bw_dC` has already staged in slot NC — so read it like any other.
    int is_last = (k == NC - 1) && !CARRY;
    int LQ = dqk + 1, LV = dhv + 1, LS = L + 1;

    extern __shared__ float sh[];
    float* sQ   = sh;                 // [L, LQ]
    float* sK   = sQ + L * LQ;        // [L, LQ]
    float* sV   = sK + L * LQ;        // [L, LV]
    float* sDN  = sV + L * LV;        // [L, LV]    d_num
    float* sDS  = sDN + L * LV;       // [L, LS]    DS, then dS
    float* sC   = sDS + L * LS;       // [dhv, LQ]  C_{k-1}
    float* sdC  = sC + dhv * LQ;      // [dhv, LQ]  dC_k
    float* sN   = sdC + dhv * LQ;     // [dqk]      n_{k-1}
    float* sdN  = sN + dqk;           // [dqk]      dn_k
    float* sFc  = sdN + dqk;          // [L]
    float* sIg  = sFc + L;            // [L]
    float* sM   = sIg + L;            // [L]
    float* sB   = sM + L;             // [L]
    float* sA   = sB + L;             // [L]
    float* sQn  = sA + L;             // [L]
    float* sDQn = sQn + L;            // [L]
    float* sDfc = sDQn + L;           // [L]
    float* sDig = sDfc + L;           // [L]
    float* sDa  = sDig + L;           // [L]
    float* sDb  = sDa + L;            // [L]
    __shared__ float sRed[512]; // must cover FUSED_THREADS_PAR

    for (int e = tid; e < len * dqk; e += nthreads) {
        int t = e / dqk, q = e - t * dqk;
        long off = bhBase(bh, H, sQK_B, sQK_H) + (long)(c0 + t) * sQK_S + q;
        sQ[t * LQ + q] = slab_ld(qq, off);
        sK[t * LQ + q] = slab_ld(kk, off);
    }
    for (int e = tid; e < len * dhv; e += nthreads) {
        int t = e / dhv, v = e - t * dhv;
        sV[t * LV + v] = slab_ld(vv, bhBase(bh, H, sHV_B, sHV_H) + (long)(c0 + t) * sHV_S + v);
    }
    for (int e = tid; e < dhv * dqk; e += nthreads) {
        int v = e / dqk, q = e - v * dqk;
        sC[v * LQ + q] = cst[((long)bh * (NC + 1) + k) * dhv * dqk + e];
        sdC[v * LQ + q] =
            is_last ? 0.0f : dcst[((long)bh * (NC + 1) + (k + 1)) * dhv * dqk + e];
    }
    for (int e = tid; e < dqk; e += nthreads) {
        sN[e] = nst[((long)bh * (NC + 1) + k) * dqk + e];
        sdN[e] = is_last ? 0.0f : dnst[((long)bh * (NC + 1) + (k + 1)) * dqk + e];
    }
    for (int j = tid; j < L; j += nthreads) {
        sFc[j] = fcb[((long)bh * NC + k) * L + j];
        sIg[j] = (j < len) ? ig[bhBase(bh, H, sG_B, sG_H) + (long)(c0 + j) * sG_S] : 0.0f;
        sDfc[j] = 0.0f;
        sDig[j] = 0.0f;
        sDa[j] = 0.0f;
        sDb[j] = 0.0f;
    }
    float m_prev = mst[(long)bh * (NC + 1) + k];
    __syncthreads();

    float fc_last = sFc[len - 1];
    float m_last = msv[(long)bh * T + c0 + len - 1];
    float gsca = expf(fc_last + m_prev - m_last);

    for (int t = tid; t < len; t += nthreads) {
        long gt = (long)bh * T + c0 + t;
        long gy = bhBase(bh, H, sHV_B, sHV_H) + (long)(c0 + t) * sHV_S;
        sM[t] = msv[gt];
        sB[t] = expf(sFc[t] + m_prev - sM[t]);
        sA[t] = expf(fc_last - sFc[t] + sIg[t] - m_last);
        sQn[t] = qnv[gt];

        float inv = 1.0f / psiv[gt];
        float red = 0.0f;
        for (int v = 0; v < dhv; ++v) {
            float dy = dytil[gy + v];
            sDN[t * LV + v] = dy * inv;
            red += dy * slab_ld(ytil, gy + v);
        }
        float dpsi = -red * inv;
        float qn = sQn[t];
        // Grad flows through qn only where it, not the exp(−m) floor, won the max.
        sDQn[t] = (fabsf(qn) > expf(-sM[t])) ? ((qn > 0.0f ? 1.0f : -1.0f) * dpsi) : 0.0f;
    }
    __syncthreads();

    // Recompute DS = D̄ ⊙ (Q·Kᵀ).
    for (int e = tid; e < len * len; e += nthreads) {
        int t = e / len, j = e - t * len;
        float val = 0.0f;
        if (j <= t) {
            float s = 0.0f;
            for (int q = 0; q < dqk; ++q) s += sQ[t * LQ + q] * sK[j * LQ + q];
            val = expf(sFc[t] - sFc[j] + sIg[j] - sM[t]) * s;
        }
        sDS[t * LS + j] = val;
    }
    __syncthreads();

    // dV: the num path (needs DS) plus the state-update path C_k = g·C + (a⊙V)ᵀ·K.
    //
    // `da` rides along here rather than in a loop of its own. da[j] contracts dC_k
    // over BOTH v and q — as its own loop that is `len` threads (32) each grinding
    // dhv·dqk (4096) iterations while the other 224 idle. But the inner q-contraction
    // is exactly the `st` this loop already computes, so accumulating V[j][v]·st into
    // a shared da[j] reuses it and spreads the work over all len·dhv elements.
    for (int e = tid; e < len * dhv; e += nthreads) {
        int j = e / dhv, v = e - j * dhv;
        float acc = 0.0f;
        for (int t = j; t < len; ++t) acc += sDS[t * LS + j] * sDN[t * LV + v];
        float st = 0.0f;
        for (int q = 0; q < dqk; ++q) st += sdC[v * LQ + q] * sK[j * LQ + q];
        atomicAdd(&sDa[j], sV[j * LV + v] * st);
        acc += sA[j] * st;
        dv[bhBase(bh, H, sHV_B, sHV_H) + (long)(c0 + j) * sHV_S + v] = acc;
    }

    // db[t] contracts d_num against the pre-b inter read-out Q[t]·C_prevᵀ, which is
    // an [len, dhv] product — so it parallelizes over the same len·dhv grid instead
    // of over `len` alone.
    for (int e = tid; e < len * dhv; e += nthreads) {
        int t = e / dhv, v = e - t * dhv;
        float pre = 0.0f;
        for (int q = 0; q < dqk; ++q) pre += sQ[t * LQ + q] * sC[v * LQ + q];
        atomicAdd(&sDb[t], sDN[t * LV + v] * pre);
    }
    __syncthreads();

    // The n-side of da/db: both contract over dqk only, so `len` threads is enough.
    for (int j = tid; j < len; j += nthreads) {
        float acc = 0.0f, pre_qn = 0.0f;
        for (int q = 0; q < dqk; ++q) {
            acc += sdN[q] * sK[j * LQ + q];
            pre_qn += sQ[j * LQ + q] * sN[q];
        }
        sDa[j] += acc;
        sDb[j] += sDQn[j] * pre_qn;
    }

    // dDS -> (dS, P). P = dDS⊙DS feeds the decay grads; dS overwrites DS in place
    // (each (t, j) is owned by exactly one thread, so the read-then-write is safe).
    // dfc[t] += Σ_{j<=t} P[t][j] is a row sum; dfc[j] -= Σ_t P[t][j] and
    // dig[j] += Σ_t P[t][j] are column sums, so they go through shared atomics.
    for (int e = tid; e < len * len; e += nthreads) {
        int t = e / len, j = e - t * len;
        if (j <= t) {
            float ds_val = sDS[t * LS + j];
            float dds = sDQn[t];
            for (int v = 0; v < dhv; ++v) dds += sDN[t * LV + v] * sV[j * LV + v];
            float p = dds * ds_val;
            atomicAdd(&sDfc[t], p);
            atomicAdd(&sDfc[j], -p);
            atomicAdd(&sDig[j], p);
            float dbar = expf(sFc[t] - sFc[j] + sIg[j] - sM[t]);
            sDS[t * LS + j] = dds * dbar;
        } else {
            sDS[t * LS + j] = 0.0f;
        }
    }
    __syncthreads();

    // dQ: the intra path (dS·K) plus the two inter read-outs of the incoming state.
    for (int e = tid; e < len * dqk; e += nthreads) {
        int t = e / dqk, q = e - t * dqk;
        float acc = 0.0f;
        for (int j = 0; j <= t; ++j) acc += sDS[t * LS + j] * sK[j * LQ + q];
        float inter = 0.0f;
        for (int v = 0; v < dhv; ++v) inter += sDN[t * LV + v] * sC[v * LQ + q];
        acc += sB[t] * (inter + sDQn[t] * sN[q]);
        dq[bhBase(bh, H, sQK_B, sQK_H) + (long)(c0 + t) * sQK_S + q] = acc;
    }

    // dK: the intra path (dSᵀ·Q) plus both state-update paths (C and n).
    for (int e = tid; e < len * dqk; e += nthreads) {
        int j = e / dqk, q = e - j * dqk;
        float acc = 0.0f;
        for (int t = j; t < len; ++t) acc += sDS[t * LS + j] * sQ[t * LQ + q];
        float st = 0.0f;
        for (int v = 0; v < dhv; ++v) st += sV[j * LV + v] * sdC[v * LQ + q];
        acc += sA[j] * (st + sdN[q]);
        dk[bhBase(bh, H, sQK_B, sQK_H) + (long)(c0 + j) * sQK_S + q] = acc;
    }
    __syncthreads();

    // dg = Σ dC_k⊙C_{k-1} + Σ dn_k⊙n_{k-1}, a block-wide reduction.
    float loc = 0.0f;
    for (int e = tid; e < dhv * dqk; e += nthreads) {
        int v = e / dqk, q = e - v * dqk;
        loc += sdC[v * LQ + q] * sC[v * LQ + q];
    }
    for (int e = tid; e < dqk; e += nthreads) loc += sdN[e] * sN[e];
    sRed[tid] = loc;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sRed[tid] += sRed[tid + s];
        __syncthreads();
    }
    float dg = sRed[0];

    // a/b/g -> (dfc, dig), accumulating onto the intra-chunk D̄ contribution
    // (m held constant, as everywhere).
    for (int j = tid; j < len; j += nthreads) {
        float pa = sDa[j] * sA[j];
        atomicAdd(&sDig[j], pa);
        atomicAdd(&sDfc[j], sDb[j] * sB[j] - pa);
    }
    __syncthreads();
    if (tid == 0) {
        float acc = dg * gsca;
        for (int j = 0; j < len; ++j) acc += sDa[j] * sA[j];
        sDfc[len - 1] += acc;
    }
    __syncthreads();

    // dfc -> d(forget logit): reverse cumsum within the chunk, times logσ'.
    // Serial over len on one thread — len is 32 and this is the tail of the kernel.
    if (tid == 0) {
        float acc = 0.0f;
        for (int j = len - 1; j >= 0; --j) {
            acc += sDfc[j];
            long gg = bhBase(bh, H, sG_B, sG_H) + (long)(c0 + j) * sG_S;
            dfg[gg] = acc * (1.0f - stable_sigmoid(fg[gg]));
            dig[gg] = sDig[j];
        }
    }
}

#if MMA_TF32

// The tensor-core twin of `mlstm_bw_parallel`. Every contraction in the backward
// is a dot, so every one of them goes to the tensor cores:
//
//   phase 1   S     = Q·Kᵀ         over dqk   -> masked+decayed into sDS (as fwd)
//   phase 2   dVnum = DSᵀ·dN       over t     |
//             st    = K·dCᵀ        over dqk   |- all three land on [L, dhv], so one
//             pre   = Q·Cᵀ         over dqk   |  warp pass computes all three
//   phase 4   dds   = dN·Vᵀ        over dhv   -> [L, L], epilogue makes dS in place
//   phase 5   dQint = dS·K         over j     |
//             dQinx = dN·C         over dhv   |- all four land on [L, dqk], so again
//             dKint = dSᵀ·Q        over t     |  one warp pass
//             dKst  = V·dC         over dhv   |
//
// Grouping by output tile is the point: an accumulator lives in registers for the
// whole K-loop, so two dots that write the same tile cost one tile's worth of
// epilogue and one set of shared-memory reads for the shared operand.
//
// `da`/`db` still reduce with shared atomics — they contract the SAME products the
// tiles already hold (`st` over v, `pre` over v), so they ride along in the epilogue
// instead of getting a pass of their own, exactly as in the scalar kernel.
//
// Everything else — the dg reduction, the a/b/g -> (dfc, dig) fold, the reverse
// cumsum tail — is elementwise or a scan, has no dot in it, and is unchanged.
extern "C" __global__ void mlstm_bw_parallel_mma(
    const slab_t* qq, const slab_t* kk, const slab_t* vv,
    const float* ig, const float* fg, const float* fcb,
    const float* cst, const float* nst, const float* mst,
    const float* dcst, const float* dnst,
    const slab_t* ytil, const float* dytil, const float* psiv,
    const float* qnv, const float* msv,
    float* dq, float* dk, float* dv, float* dig, float* dfg,
    int T, int L, int NC, int dqk, int dhv, int CARRY, int H,
    long sQK_B, long sQK_H, long sQK_S, long sHV_B, long sHV_H, long sHV_S,
    long sG_B, long sG_H, long sG_S) {
    int k = blockIdx.x, bh = blockIdx.y;
    int tid = threadIdx.x, nthreads = blockDim.x;
    int warp = tid >> 5, nwarps = nthreads >> 5;
    int c0 = k * L;
    int len = min(L, T - c0);
    // See `mlstm_bw_parallel`: under CARRY the last chunk's outgoing gradient is the
    // one staged in slot NC, not zero.
    int is_last = (k == NC - 1) && !CARRY;

    int LP = (L + 15) & ~15;
    int KP = (dqk + 7) & ~7;
    int VP = (dhv + 7) & ~7;
    int LQ = KP + 1, LV = VP + 1, LS = LP + 1;

    extern __shared__ float sh[];
    float* sQ   = sh;                  // [LP, LQ]
    float* sK   = sQ + LP * LQ;        // [LP, LQ]
    float* sV   = sK + LP * LQ;        // [LP, LV]
    float* sDN  = sV + LP * LV;        // [LP, LV]   d_num
    float* sDS  = sDN + LP * LV;       // [LP, LS]   DS, then dS
    float* sC   = sDS + LP * LS;       // [VP, LQ]   C_{k-1}
    float* sdC  = sC + VP * LQ;        // [VP, LQ]   dC_k
    float* sN   = sdC + VP * LQ;       // [KP]
    float* sdN  = sN + KP;             // [KP]
    float* sFc  = sdN + KP;            // [LP]
    float* sIg  = sFc + LP;            // [LP]
    float* sM   = sIg + LP;            // [LP]
    float* sB   = sM + LP;             // [LP]
    float* sA   = sB + LP;             // [LP]
    float* sQn  = sA + LP;             // [LP]
    float* sDQn = sQn + LP;            // [LP]
    float* sDfc = sDQn + LP;           // [LP]
    float* sDig = sDfc + LP;           // [LP]
    float* sDa  = sDig + LP;           // [LP]
    float* sDb  = sDa + LP;            // [LP]
    __shared__ float sRed[512]; // must cover FUSED_THREADS_PAR

    for (int e = tid; e < LP * KP; e += nthreads) {
        int t = e / KP, q = e - t * KP;
        int ok = (t < len) && (q < dqk);
        sQ[t * LQ + q] = ok ? slab_ld(qq, bhBase(bh, H, sQK_B, sQK_H) + (long)(c0 + t) * sQK_S + q) : 0.0f;
        sK[t * LQ + q] = ok ? slab_ld(kk, bhBase(bh, H, sQK_B, sQK_H) + (long)(c0 + t) * sQK_S + q) : 0.0f;
    }
    for (int e = tid; e < LP * VP; e += nthreads) {
        int t = e / VP, v = e - t * VP;
        sV[t * LV + v] = ((t < len) && (v < dhv))
            ? slab_ld(vv, bhBase(bh, H, sHV_B, sHV_H) + (long)(c0 + t) * sHV_S + v) : 0.0f;
        sDN[t * LV + v] = 0.0f; // filled below for t < len
    }
    for (int e = tid; e < VP * KP; e += nthreads) {
        int v = e / KP, q = e - v * KP;
        int ok = (v < dhv) && (q < dqk);
        sC[v * LQ + q] =
            ok ? cst[((long)bh * (NC + 1) + k) * dhv * dqk + (long)v * dqk + q] : 0.0f;
        // The last chunk's outgoing state is never read, so its gradient is zero.
        sdC[v * LQ + q] = (ok && !is_last)
            ? dcst[((long)bh * (NC + 1) + (k + 1)) * dhv * dqk + (long)v * dqk + q] : 0.0f;
    }
    for (int e = tid; e < KP; e += nthreads) {
        int ok = e < dqk;
        sN[e] = ok ? nst[((long)bh * (NC + 1) + k) * dqk + e] : 0.0f;
        sdN[e] = (ok && !is_last) ? dnst[((long)bh * (NC + 1) + (k + 1)) * dqk + e] : 0.0f;
    }
    for (int j = tid; j < LP; j += nthreads) {
        sFc[j] = (j < L)   ? fcb[((long)bh * NC + k) * L + j] : 0.0f;
        sIg[j] = (j < len) ? ig[bhBase(bh, H, sG_B, sG_H) + (long)(c0 + j) * sG_S]        : 0.0f;
        sM[j] = 0.0f; sB[j] = 0.0f; sA[j] = 0.0f; sQn[j] = 0.0f; sDQn[j] = 0.0f;
        sDfc[j] = 0.0f; sDig[j] = 0.0f; sDa[j] = 0.0f; sDb[j] = 0.0f;
    }
    float m_prev = mst[(long)bh * (NC + 1) + k];
    __syncthreads();

    float fc_last = sFc[len - 1];
    float m_last = msv[(long)bh * T + c0 + len - 1];
    float gsca = expf(fc_last + m_prev - m_last);

    for (int t = tid; t < len; t += nthreads) {
        long gt = (long)bh * T + c0 + t;
        long gy = bhBase(bh, H, sHV_B, sHV_H) + (long)(c0 + t) * sHV_S;
        sM[t] = msv[gt];
        sB[t] = expf(sFc[t] + m_prev - sM[t]);
        sA[t] = expf(fc_last - sFc[t] + sIg[t] - m_last);
        sQn[t] = qnv[gt];

        float inv = 1.0f / psiv[gt];
        float red = 0.0f;
        for (int v = 0; v < dhv; ++v) {
            float dy = dytil[gy + v];
            sDN[t * LV + v] = dy * inv;
            red += dy * slab_ld(ytil, gy + v);
        }
        float dpsi = -red * inv;
        float qn = sQn[t];
        // Grad flows through qn only where it, not the exp(−m) floor, won the max.
        sDQn[t] = (fabsf(qn) > expf(-sM[t])) ? ((qn > 0.0f ? 1.0f : -1.0f) * dpsi) : 0.0f;
    }
    __syncthreads();

    int mtile = LP >> 4, ltile = LP >> 3, vtile = VP >> 3, ktile = KP >> 3;

    // Phase 1: recompute DS = D̄ ⊙ (Q·Kᵀ), exactly as the forward built it.
    for (int tile = warp; tile < mtile * ltile; tile += nwarps) {
        int m0 = (tile / ltile) << 4, n0 = (tile % ltile) << 3;
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        unsigned a[4], b[2];
        for (int k0 = 0; k0 < KP; k0 += 8) {
            ld_a_mk(a, sQ, LQ, m0, k0);
            ld_b_nk(b, sK, LQ, k0, n0);
            mma_16x8x8(d, a, b);
        }
        for (int i = 0; i < 4; ++i) {
            int t = m0 + mma_row(i), j = n0 + mma_col(i);
            float val = 0.0f;
            if (t < len && j <= t) val = expf(sFc[t] - sFc[j] + sIg[j] - sM[t]) * d[i];
            sDS[t * LS + j] = val;
        }
    }
    __syncthreads();

    // Phase 2: dV, plus the `st` and `pre` products that `da`/`db` reduce over v.
    // Three dots, one [L, dhv] output tile, one warp pass.
    for (int tile = warp; tile < mtile * vtile; tile += nwarps) {
        int m0 = (tile / vtile) << 4, n0 = (tile % vtile) << 3;
        float dnum[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Σ_t DS[t][j]·dN[t][v]
        float dst[4]  = {0.0f, 0.0f, 0.0f, 0.0f};  // Σ_q dC[v][q]·K[j][q]
        float dpre[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Σ_q Q[t][q]·C[v][q]
        unsigned a[4], b[2];
        // DS[t][j] is zero for j > t, so contracting over ALL t is the same as t >= j.
        for (int k0 = 0; k0 < LP; k0 += 8) {
            ld_a_km(a, sDS, LS, m0, k0);   // Aᵀ in memory: DS is [t, j], we want [j, t]
            ld_b_kn(b, sDN, LV, k0, n0);
            mma_16x8x8(dnum, a, b);
        }
        for (int k0 = 0; k0 < KP; k0 += 8) {
            ld_a_mk(a, sK, LQ, m0, k0);
            ld_b_nk(b, sdC, LQ, k0, n0);
            mma_16x8x8(dst, a, b);
            ld_a_mk(a, sQ, LQ, m0, k0);
            ld_b_nk(b, sC, LQ, k0, n0);
            mma_16x8x8(dpre, a, b);
        }
        for (int i = 0; i < 4; ++i) {
            int r = m0 + mma_row(i), v = n0 + mma_col(i); // r is `j` for dv, `t` for db
            if (r < len && v < dhv) {
                float st = dst[i];
                dv[bhBase(bh, H, sHV_B, sHV_H) + (long)(c0 + r) * sHV_S + v] = dnum[i] + sA[r] * st;
                atomicAdd(&sDa[r], sV[r * LV + v] * st);
                atomicAdd(&sDb[r], sDN[r * LV + v] * dpre[i]);
            }
        }
    }
    __syncthreads();

    // The n-side of da/db: both contract over dqk only — a matrix-vector product, so
    // there is no dot here for the tensor cores and `len` threads is enough.
    for (int j = tid; j < len; j += nthreads) {
        float acc = 0.0f, pre_qn = 0.0f;
        for (int q = 0; q < dqk; ++q) {
            acc += sdN[q] * sK[j * LQ + q];
            pre_qn += sQ[j * LQ + q] * sN[q];
        }
        sDa[j] += acc;
        sDb[j] += sDQn[j] * pre_qn;
    }
    __syncthreads();

    // Phase 4: dDS -> (dS, P). dS overwrites DS in place; the warp that owns an
    // output tile is the only one that reads or writes those elements, and the
    // barrier above guarantees phase 2 is done reading DS.
    for (int tile = warp; tile < mtile * ltile; tile += nwarps) {
        int m0 = (tile / ltile) << 4, n0 = (tile % ltile) << 3;
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};   // Σ_v dN[t][v]·V[j][v]
        unsigned a[4], b[2];
        for (int k0 = 0; k0 < VP; k0 += 8) {
            ld_a_mk(a, sDN, LV, m0, k0);
            ld_b_nk(b, sV, LV, k0, n0);
            mma_16x8x8(d, a, b);
        }
        for (int i = 0; i < 4; ++i) {
            int t = m0 + mma_row(i), j = n0 + mma_col(i);
            float out = 0.0f;
            if (t < len && j <= t) {
                float dds = d[i] + sDQn[t];
                float p = dds * sDS[t * LS + j];
                atomicAdd(&sDfc[t], p);
                atomicAdd(&sDfc[j], -p);
                atomicAdd(&sDig[j], p);
                out = dds * expf(sFc[t] - sFc[j] + sIg[j] - sM[t]);
            }
            sDS[t * LS + j] = out;
        }
    }
    __syncthreads();

    // Phase 5: dQ and dK. Four dots, one [L, dqk] output tile, one warp pass.
    for (int tile = warp; tile < mtile * ktile; tile += nwarps) {
        int m0 = (tile / ktile) << 4, n0 = (tile % ktile) << 3;
        float dqi[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Σ_j dS[t][j]·K[j][q]
        float dqx[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Σ_v dN[t][v]·C[v][q]
        float dki[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Σ_t dS[t][j]·Q[t][q]
        float dks[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Σ_v V[j][v]·dC[v][q]
        unsigned a[4], b[2];
        for (int k0 = 0; k0 < LP; k0 += 8) {
            ld_a_mk(a, sDS, LS, m0, k0);   // dS as [t, j], contracting j
            ld_b_kn(b, sK, LQ, k0, n0);
            mma_16x8x8(dqi, a, b);
            ld_a_km(a, sDS, LS, m0, k0);   // dSᵀ, contracting t
            ld_b_kn(b, sQ, LQ, k0, n0);
            mma_16x8x8(dki, a, b);
        }
        for (int k0 = 0; k0 < VP; k0 += 8) {
            ld_a_mk(a, sDN, LV, m0, k0);
            ld_b_kn(b, sC, LQ, k0, n0);
            mma_16x8x8(dqx, a, b);
            ld_a_mk(a, sV, LV, m0, k0);
            ld_b_kn(b, sdC, LQ, k0, n0);
            mma_16x8x8(dks, a, b);
        }
        for (int i = 0; i < 4; ++i) {
            int r = m0 + mma_row(i), q = n0 + mma_col(i); // r is `t` for dq, `j` for dk
            if (r < len && q < dqk) {
                dq[bhBase(bh, H, sQK_B, sQK_H) + (long)(c0 + r) * sQK_S + q] =
                    dqi[i] + sB[r] * (dqx[i] + sDQn[r] * sN[q]);
                dk[bhBase(bh, H, sQK_B, sQK_H) + (long)(c0 + r) * sQK_S + q] =
                    dki[i] + sA[r] * (dks[i] + sdN[q]);
            }
        }
    }
    __syncthreads();

    // dg = Σ dC_k⊙C_{k-1} + Σ dn_k⊙n_{k-1}, a block-wide reduction. The pad is zero
    // in both operands, so it contributes nothing and needs no masking.
    float loc = 0.0f;
    for (int e = tid; e < VP * KP; e += nthreads) {
        int v = e / KP, q = e - v * KP;
        loc += sdC[v * LQ + q] * sC[v * LQ + q];
    }
    for (int e = tid; e < KP; e += nthreads) loc += sdN[e] * sN[e];
    sRed[tid] = loc;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sRed[tid] += sRed[tid + s];
        __syncthreads();
    }
    float dg = sRed[0];

    // a/b/g -> (dfc, dig), accumulating onto the intra-chunk D̄ contribution
    // (m held constant, as everywhere).
    for (int j = tid; j < len; j += nthreads) {
        float pa = sDa[j] * sA[j];
        atomicAdd(&sDig[j], pa);
        atomicAdd(&sDfc[j], sDb[j] * sB[j] - pa);
    }
    __syncthreads();
    if (tid == 0) {
        float acc = dg * gsca;
        for (int j = 0; j < len; ++j) acc += sDa[j] * sA[j];
        sDfc[len - 1] += acc;
    }
    __syncthreads();

    if (tid == 0) {
        float acc = 0.0f;
        for (int j = len - 1; j >= 0; --j) {
            acc += sDfc[j];
            long gg = bhBase(bh, H, sG_B, sG_H) + (long)(c0 + j) * sG_S;
            dfg[gg] = acc * (1.0f - stable_sigmoid(fg[gg]));
            dig[gg] = sDig[j];
        }
    }
}

#endif // MMA_TF32
