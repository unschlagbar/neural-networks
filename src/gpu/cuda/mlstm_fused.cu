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
// LAYOUT: q/k/v and the gate logits are POSITION-MAJOR `[B*T, H*W]` — the layout the
// projections write — so element (b, h, t, c) sits at `((b*T + t)*H + h)*W + c`, and
// `W` is `dqk` for q/k, `dhv` for v and the output, 1 for the gates. Feeding the
// projection output straight in costs no reorg pass, and loads stay coalesced: the
// fast axis `c` is contiguous, so a warp covers consecutive elements of one timestep.
//
// The reference (nx-ai/mlstm_kernels) passes str_matQK_B_NH / _S / _DHQK because
// PyTorch hands it whatever layout the caller had. Here there is one producer and
// one consumer, so the strides are derived from (B, H, T, dqk, dhv) below instead of
// travelling as nine more kernel arguments.
#define MLSTM_STRIDES_G                                                        \
    const long sG_S = H, sG_H = 1, sG_B = (long)T * H;
#define MLSTM_STRIDES                                                          \
    const long sQK_S = (long)H * dqk, sQK_H = dqk, sQK_B = (long)T * sQK_S;    \
    const long sHV_S = (long)H * dhv, sHV_H = dhv, sHV_B = (long)T * sHV_S;    \
    MLSTM_STRIDES_G

// Base offset of the (b, h) a block owns, from its flat `bh = b*H + h`. Each tensor
// group (q/k, v/h, gates) passes its own stride pair.
__device__ __forceinline__ long bhBase(int bh, int H, long sB, long sH) {
    return (long)(bh / H) * sB + (long)(bh % H) * sH;
}

// SHAPE CONSTANTS. Built with -DMLSTM_SPEC the shape parameters are literals from
// NVRTC, matching how the reference passes them as `tl.constexpr`; the kernels then
// fold their padded dims, give every strided loop a static trip count, and drop the
// pad branches. Without it they stay the runtime arguments they always were, so the
// generic kernels are unchanged.
//
// The specialized build takes each parameter under an `arg_`-prefixed name and
// immediately rebinds the plain name to the constant, so every use in the body folds
// without editing each site. Signatures are identical either way, so the launch side
// needs no second argument list — a specialized kernel is simply passed values it
// then ignores.
//
// `TV` and the block width matter as much as the head dims here: the two recurrent
// kernels stride `tv * dqk` elements over `nthreads`, and with both constant that
// becomes a fixed unrolled trip count instead of a loop with a runtime bound.
#if MLSTM_SPEC
#define MLSTM_SHAPE_ARGS \
    int T, int arg_L, int NC, int arg_dqk, int arg_dhv, int arg_H
#define MLSTM_SHAPE_ARGS_BW \
    int T, int arg_L, int NC, int arg_dqk, int arg_dhv, int CARRY, int arg_H
#define MLSTM_SHAPE_BIND                                                       \
    (void)arg_L; (void)arg_dqk; (void)arg_dhv; (void)arg_H;                    \
    const int L = MLSTM_L, dqk = MLSTM_DQK, dhv = MLSTM_DHV, H = MLSTM_H;      \
    MLSTM_STRIDES
// The recurrent pair additionally takes TV; `nthreads` is blockDim.x, which the
// launch pins to MLSTM_THREADS.
#define MLSTM_SHAPE_ARGS_REC                                                   \
    int T, int arg_L, int NC, int arg_dqk, int arg_dhv, int arg_TV,            \
    int CARRY, int arg_H
#define MLSTM_SHAPE_BIND_REC                                                   \
    (void)arg_L; (void)arg_dqk; (void)arg_dhv; (void)arg_TV; (void)arg_H;      \
    const int L = MLSTM_L, dqk = MLSTM_DQK, dhv = MLSTM_DHV, H = MLSTM_H;      \
    const int TV = MLSTM_TV;                                                   \
    MLSTM_STRIDES
#define MLSTM_NTHREADS MLSTM_THREADS
// `sRed` is a static array, so it must be sized at compile time; specialized it is
// exactly the launch width instead of the generic build's worst case.
#define MLSTM_NTHREADS_MAX MLSTM_THREADS
#else
#define MLSTM_SHAPE_ARGS \
    int T, int L, int NC, int dqk, int dhv, int H
#define MLSTM_SHAPE_ARGS_BW \
    int T, int L, int NC, int dqk, int dhv, int CARRY, int H
#define MLSTM_SHAPE_ARGS_REC \
    int T, int L, int NC, int dqk, int dhv, int TV, int CARRY, int H
#define MLSTM_SHAPE_BIND MLSTM_STRIDES
#define MLSTM_SHAPE_BIND_REC MLSTM_STRIDES
#define MLSTM_NTHREADS blockDim.x
#define MLSTM_NTHREADS_MAX 1024
#endif

// Head-dim slice width for the two parallel kernels (the reference's `siz_b_DHQK`).
// A fixed cap, never a divisor of any particular head dim: a wider head raises the
// loop's trip count and leaves shared memory where it is, which is what keeps the
// kernels usable as WORD_HIDDEN grows. Must be a multiple of 16, the bf16 mma's
// contraction depth. `MLSTM_KT=<n>` at compile time overrides for a sweep.
#ifndef MLSTM_KT
#define MLSTM_KT 32
#endif

// The `dhv` twin of MLSTM_KT (the reference's `siz_b_DHHV`). Slicing the value
// dimension partitions the OUTPUT columns of the parallel kernels rather than a
// contraction, so unlike `MLSTM_KT` no slice needs accumulating across iterations.
// Must be a multiple of the mma N (8). `MLSTM_VT=<n>` overrides for a sweep.
#ifndef MLSTM_VT
#define MLSTM_VT 64
#endif

// Gate preprocessing for a whole sequence: one block per (b, h). Everything the
// chunk state needs that is a pure function of the gates, so neither heavy kernel
// carries any stabilizer logic of its own. Within chunk k of length `len`:
//   fc[k][j] = Σ_{j'<=j} logsigmoid(f[k*L + j'])           (their vecB)
//   d[k][j]  = fc[k][len-1] - fc[k][j] + i[k*L + j]        the last row of logD
//   m[k+1]   = max( max_j d[k][j], fc[k][len-1] + m[k] )   the running stabilizer
//   g[k]     = exp(fc[k][len-1] + m[k] - m[k+1])           (their scaGbar)
//   a[k][j]  = exp(d[k][j] - m[k+1])                       (their vecAbar)
//
// The scan over `m` is the ONLY serial-over-chunks step left in the forward, and it
// is a handful of flops per chunk rather than a [dhv, dqk] state update — which is
// what lets `mlstm_fw_dC` take every chunk in parallel.
//
// One WARP per chunk, lane j owning position j: L never exceeds FUSED_MAX_L = 32, so
// the cumulative log-forget is a single warp-inclusive scan and the row max a single
// shuffle reduction. Lanes past `len` add nothing to the scan, so they come out
// holding the last valid prefix — which is what `mlstm_bw_parallel` expects to find
// in `fcb` there.
extern "C" __global__ void mlstm_fw_gates(
    const float* fg, const float* ig,
    float* fcb, float* avec, float* gvec, float* mst,
    int T, int L, int NC, int CARRY, int H) {
    MLSTM_STRIDES_G
    const int bh = blockIdx.x, tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5, nwarps = blockDim.x >> 5;
    const long gbase = bhBase(bh, H, sG_B, sG_H);
    const long kbase = (long)bh * NC;
    float* mrow = mst + (long)bh * (NC + 1);

    extern __shared__ float sh[];
    float* sFcLast = sh;          // [NC]
    float* sMloc = sFcLast + NC;  // [NC]  the chunk-local row max, before the scan

    for (int k = warp; k < NC; k += nwarps) {
        const int c0 = k * L, len = min(L, T - c0);
        float fc = (lane < len) ? log_sigmoid(fg[gbase + (long)(c0 + lane) * sG_S]) : 0.0f;
        for (int off = 1; off < 32; off <<= 1) {
            const float v = __shfl_up_sync(0xffffffffu, fc, off);
            if (lane >= off) fc += v;
        }
        const float fc_last = __shfl_sync(0xffffffffu, fc, len - 1);
        const float igv = (lane < len) ? ig[gbase + (long)(c0 + lane) * sG_S] : 0.0f;
        // Positions past `len` must contribute nothing to the contraction in
        // `mlstm_fw_dC`; -1e30 puts them at exactly zero once exponentiated below.
        const float d = (lane < len) ? (fc_last - fc + igv) : -1e30f;
        float mx = d;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
        if (lane < L) {
            fcb[(kbase + k) * L + lane] = fc;
            // `d` is parked where `a` will go: the second pass below only needs the
            // stabilizer subtracted, which is not known until the scan has run.
            avec[(kbase + k) * L + lane] = d;
        }
        if (lane == 0) {
            sFcLast[k] = fc_last;
            sMloc[k] = mx;
        }
    }
    __syncthreads();

    if (tid == 0) {
        // Slot 0 is the state the sequence starts from: under CARRY the caller staged
        // the incoming stabilizer there, otherwise the recurrence starts at zero.
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

// The chunk-state recurrence, and all that is left of it: fold the chunk-local
// contributions `mlstm_fw_dC` wrote into the running state.
//   C_{k+1} = g_k·C_k + ΔC_k ,   n_{k+1} = g_k·n_k + Δn_k
// In place, one thread per state element walking the chunks in a register. Every
// element of the state is independent, so what used to pin the grid at BH blocks
// (one per sequence, each carrying a whole [dhv, dqk] state through a serial chunk
// loop) now spreads over BH·(dhv·dqk + dqk) threads with the same serial depth.
//
// Elements past the [dhv, dqk] state are the `n` vector, which decays by the same
// `g`, so one loop covers both.
extern "C" __global__ void mlstm_fw_scan(
    float* cst, float* nst, const float* gvec,
    int NC, int dqk, int dhv, int CARRY) {
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
    float acc = CARRY ? p[0] : 0.0f;
    if (!CARRY) p[0] = 0.0f;
    // One slot read ahead, so the multiply-add of chunk k overlaps the load of k+1
    // instead of waiting on it.
    float nxt = p[stride];
    for (int k = 0; k < NC; ++k) {
        const float cur = nxt;
        p += stride;
        if (k + 1 < NC) nxt = p[stride];
        acc = g[k] * acc + cur;
        *p = acc;
    }
}

// Tensor-core dots (sm_80+)
//
// Every contraction in the reference (nx-ai/mlstm_kernels) is a `tl.dot`. Below is
// that same `dot`, written out as the PTX Triton would emit, in the two widths the
// kernels need — fp32 operands (TF32) and bf16 operands. Everything the forward
// contracts is bf16 in memory or is narrowed on the way into shared memory, so the
// forward uses `mma_bf16`; the backward still carries fp32 intermediates into its
// dots and uses the TF32 unit.
//
// The TF32 unit is `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`: a whole
// WARP cooperates to compute D(16x8) += A(16x8)·B(8x8), with the operands rounded
// to TF32 (fp32's 8 exponent bits, 10 mantissa bits) and the product accumulated in
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

// bf16 operands (mma.m16n8k16, sm_80+)
//
// The forward's operands are already bf16 where they came out of a projection, so
// rounding them to TF32 for the unit above is a no-op that costs a `cvt` per element
// and half the contraction depth. `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
// takes them as they are: a warp computes D(16x8) += A(16x16)·B(16x8) in one
// instruction where the TF32 unit needs two, and the staging tiles are half the
// shared memory — which is what lets a block hold the whole chunk without slicing.
// This is where the reference is too (`tl.dot` on DTYPE = bf16 tensors).
//
// The fragment register layout is the same table as above with the contraction
// twice as deep, so each register holds a PAIR of adjacent k: register a0 is
// A[g][2c] in its low half and A[g][2c+1] in its high half, with g = lane/4 and
// c = lane%4.
typedef unsigned short bf16s_t;

// Round-to-nearest-even into bf16. Written out rather than taken from
// <cuda_bf16.h>: NVRTC only finds that header when the CUDA include path was
// located, and this is four integer ops.
__device__ __forceinline__ bf16s_t to_bf16(float x) {
    unsigned u = __float_as_uint(x);
    return (bf16s_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ float from_bf16(bf16s_t x) {
    return __uint_as_float((unsigned)x << 16);
}

// The two halves of a fragment register, in contraction order.
__device__ __forceinline__ unsigned bf16_pair(bf16s_t lo, bf16s_t hi) {
    return (unsigned)lo | ((unsigned)hi << 16);
}

// Bank-conflict pad for a bf16 tile, in ELEMENTS. A row stride of `w + 8` rounded to
// a multiple of 16 puts consecutive rows 4 banks apart when a warp reads 32-bit
// pairs across `m` at fixed `k` — the access every A/B loader below makes — so the
// 8 rows of a fragment land on 8 different banks instead of piling onto one.
#define BF16_LD(w) ((((w) + 15) & ~15) + 8)

__device__ __forceinline__ void mma_bf16(float* d, const unsigned* a, const unsigned* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// A from a row-major [M, K] tile: A[m][k] = s[m*ld + k]. `ld` is even (BF16_LD), so
// the k-pair each register wants is one aligned 32-bit word.
__device__ __forceinline__ void ldb_a_mk(unsigned* a, const bf16s_t* s, int ld, int m0, int k0) {
    int lane = threadIdx.x & 31, g = lane >> 2, c = (lane & 3) << 1;
    const unsigned* p = (const unsigned*)s;
    int w = ld >> 1, k = (k0 + c) >> 1, k8 = (k0 + c + 8) >> 1;
    a[0] = p[(m0 + g)     * w + k];
    a[1] = p[(m0 + g + 8) * w + k];
    a[2] = p[(m0 + g)     * w + k8];
    a[3] = p[(m0 + g + 8) * w + k8];
}

// A from a row-major [K, M] tile, i.e. Aᵀ is what is in memory: A[m][k] = s[k*ld + m].
// The pair is along the SLOW axis here, so each register is two loads and a pack.
__device__ __forceinline__ void ldb_a_km(unsigned* a, const bf16s_t* s, int ld, int m0, int k0) {
    int lane = threadIdx.x & 31, g = lane >> 2, c = (lane & 3) << 1;
    a[0] = bf16_pair(s[(k0 + c)     * ld + m0 + g],     s[(k0 + c + 1) * ld + m0 + g]);
    a[1] = bf16_pair(s[(k0 + c)     * ld + m0 + g + 8], s[(k0 + c + 1) * ld + m0 + g + 8]);
    a[2] = bf16_pair(s[(k0 + c + 8) * ld + m0 + g],     s[(k0 + c + 9) * ld + m0 + g]);
    a[3] = bf16_pair(s[(k0 + c + 8) * ld + m0 + g + 8], s[(k0 + c + 9) * ld + m0 + g + 8]);
}

// B from a row-major [N, K] tile: B[k][n] = s[n*ld + k]. (`Q·Kᵀ`: K is stored
// [j, q] and is wanted as [q, j].)
__device__ __forceinline__ void ldb_b_nk(unsigned* b, const bf16s_t* s, int ld, int k0, int n0) {
    int lane = threadIdx.x & 31, g = lane >> 2, c = (lane & 3) << 1;
    const unsigned* p = (const unsigned*)s;
    int w = ld >> 1;
    b[0] = p[(n0 + g) * w + ((k0 + c) >> 1)];
    b[1] = p[(n0 + g) * w + ((k0 + c + 8) >> 1)];
}

// B from a row-major [K, N] tile: B[k][n] = s[k*ld + n]. (`(D̄⊙S)·V`.)
__device__ __forceinline__ void ldb_b_kn(unsigned* b, const bf16s_t* s, int ld, int k0, int n0) {
    int lane = threadIdx.x & 31, g = lane >> 2, c = (lane & 3) << 1;
    b[0] = bf16_pair(s[(k0 + c)     * ld + n0 + g], s[(k0 + c + 1) * ld + n0 + g]);
    b[1] = bf16_pair(s[(k0 + c + 8) * ld + n0 + g], s[(k0 + c + 9) * ld + n0 + g]);
}

// Where accumulator register `i` lands in the 16x8 output tile.
__device__ __forceinline__ int mma_row(int i) { return ((threadIdx.x & 31) >> 2) + ((i & 2) ? 8 : 0); }
__device__ __forceinline__ int mma_col(int i) { return (((threadIdx.x & 31) & 3) << 1) + (i & 1); }

// The chunk-LOCAL contribution to the state, one block per (chunk, bh):
//   ΔC_k = Σ_j a_k[j]·V_k[j] ⊗ K_k[j]      [dhv, dqk]
//   Δn_k = Σ_j a_k[j]·K_k[j]               [dqk]
// written into slot k+1 of `cst`/`nst`, where `mlstm_fw_scan` folds them into the
// running state. The split is exact rather than an approximation: `a` already
// carries the stabilizer (see `mlstm_fw_gates`), so nothing here depends on the
// state to its left and EVERY chunk runs at once.
//
// That is the whole point. The reference (nx-ai/mlstm_kernels) keeps this serial —
// one block per (b, h) walking the chunks — which is right when `B·NH` alone fills
// the machine. At the backbone's shape it is 8 blocks, so the state update was 43%
// of the forward: a grid of BH doing a [dhv, dqk] update per chunk in sequence.
// Here the grid is NC·BH and the only serial axis left is `mlstm_fw_scan`'s
// elementwise fold.
//
// ΔC is Vᵀ·K contracted over the chunk's timesteps, so it is one accumulator per
// output tile and two mma steps at L = 32. `a` is folded into V while staging it,
// which leaves K exactly the bf16 it already is in memory — as in the reference,
// which casts its `matKbar` and lets the accumulation stay fp32.
extern "C" __global__ void mlstm_fw_dC(
    const slab_t* kk, const slab_t* vv, const float* avec,
    float* cst, float* nst,
    MLSTM_SHAPE_ARGS) {
    MLSTM_SHAPE_BIND
    const int k = blockIdx.x, bh = blockIdx.y;
    const int tid = threadIdx.x, nthreads = blockDim.x;
    const int warp = tid >> 5, nwarps = nthreads >> 5;
    const int c0 = k * L, len = min(L, T - c0);
    // Padded to the mma tile — rows to M=16, the contraction to K=16, columns to
    // N=8 — and zero-filled, so a short last chunk and an odd head dim need no
    // special case: a zero row contributes nothing to a dot.
    const int LP = (L + 15) & ~15;
    const int VP = (dhv + 15) & ~15;
    const int KP = (dqk + 7) & ~7;
    const int LV = BF16_LD(VP), LK = BF16_LD(KP);

    extern __shared__ float sh[];
    float* sA = sh;                                  // [L]
    bf16s_t* sV = (bf16s_t*)(sA + L);                // [LP, LV]  V with `a` folded in
    bf16s_t* sK = sV + (long)LP * LV;                // [LP, LK]

    const long qkB = bhBase(bh, H, sQK_B, sQK_H);
    const long hvB = bhBase(bh, H, sHV_B, sHV_H);
    const float* ap = avec + ((long)bh * NC + k) * L;
    for (int j = tid; j < L; j += nthreads) sA[j] = ap[j];
    __syncthreads();

    for (int e = tid; e < LP * VP; e += nthreads) {
        const int j = e / VP, v = e - j * VP;
        sV[j * LV + v] = to_bf16((j < len && v < dhv)
            ? sA[j] * slab_ld(vv, hvB + (long)(c0 + j) * sHV_S + v) : 0.0f);
    }
    for (int e = tid; e < LP * KP; e += nthreads) {
        const int j = e / KP, q = e - j * KP;
        sK[j * LK + q] = to_bf16((j < len && q < dqk)
            ? slab_ld(kk, qkB + (long)(c0 + j) * sQK_S + q) : 0.0f);
    }
    __syncthreads();

    // sV is [L, dhv] and sK is [L, dqk], so the A loader consumes V transposed and
    // neither operand is ever materialised the other way round.
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
//   qn[t]  = Σ_j (D̄⊙S)[t][j]      + b[t]·(Q[t]·n_prev)
//   ỹ[t]   = num[t] / max(|qn[t]|, exp(-m[t]))
// Chunk 0 needs no special case: its incoming state is zero, so the inter terms
// vanish on their own.
//
// All three contractions run on the tensor cores:
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
// dims and must match `fused_smem("fw_parallel", ..)` on the host exactly.
extern "C" __global__ void mlstm_fw_parallel(
    const slab_t* qq, const slab_t* kk, const slab_t* vv, const float* ig, const float* fcb,
    const float* cst, const float* nst, const float* mst,
    slab_t* ytil, float* msv, float* psiv, float* qnv,
    MLSTM_SHAPE_ARGS) {
    MLSTM_SHAPE_BIND
    int k = blockIdx.x, bh = blockIdx.y;
    int tid = threadIdx.x, nthreads = blockDim.x;
    int warp = tid >> 5, nwarps = nthreads >> 5;
    int c0 = k * L;
    int len = min(L, T - c0);

    // `const int` of a literal folds exactly like an enum under -DMLSTM_SPEC, and
    // stays an ordinary runtime value without it.
    const int LP = (L + 15) & ~15;    // rows -> multiple of the mma M
    const int KP = (dqk + 15) & ~15;  // dqk  -> multiple of the mma K
    const int VP = (dhv + 7) & ~7;    // dhv  -> multiple of the mma N
    // Both head dims are walked in fixed-width slices, so shared memory is flat in
    // dqk/dhv and only the trip counts grow. KT slices a CONTRACTION (accumulated
    // into sDS across slices); VT slices OUTPUT columns (each slice is independent).
    const int KT = (MLSTM_KT < KP) ? MLSTM_KT : KP;
    const int VT = (MLSTM_VT < VP) ? MLSTM_VT : VP;
    const int NKT = (KP + KT - 1) / KT;
    const int NVT = (VP + VT - 1) / VT;
    const int LS = LP + 1, LA = VT + 1;                       // +1: fp32 bank pad
    const int LQ = BF16_LD(KT), LV = BF16_LD(VT), LD = BF16_LD(LP);

    // The mma operands are staged bf16 and everything that is summed, compared or
    // exponentiated stays fp32. `sDS` is both: the fp32 copy is what the row sum and
    // the mask epilogue work on, `sDSb` the narrowed operand the second dot reads —
    // as in the reference, which casts `matSbar` to DTYPE for exactly that dot.
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
        sIg[j] = (j < len) ? ig[bhBase(bh, H, sG_B, sG_H) + (long)(c0 + j) * sG_S]        : 0.0f;
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

    // S = Q·Kᵀ contracts dqk, so it accumulates across the dqk slices; the D̄ mask
    // is the epilogue of the LAST slice only, applied once to the finished sum.
    // Q·n_prev contracts dqk too and rides along in sQi.
    for (int kt = 0; kt < NKT; ++kt) {
        int q0 = kt * KT;
        __syncthreads();
        for (int e = tid; e < LP * KT; e += nthreads) {
            int t = e / KT, qs = e - t * KT, q = q0 + qs;
            int ok = (t < len) && (q < dqk);
            long off = bhBase(bh, H, sQK_B, sQK_H) + (long)(c0 + t) * sQK_S + q;
            sQ[t * LQ + qs] = to_bf16(ok ? slab_ld(qq, off) : 0.0f);
            sK[t * LQ + qs] = to_bf16(ok ? slab_ld(kk, off) : 0.0f);
        }
        for (int e = tid; e < KT; e += nthreads) {
            int q = q0 + e;
            sN[e] = (q < dqk) ? nst[((long)bh * (NC + 1) + k) * dqk + q] : 0.0f;
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
        // sQi accumulates Q·n_prev over the same slices. Guarded by `t < len`, and
        // the pad columns of sQ/sN are zero, so the slice bound needs no `q < dqk`.
        for (int t = tid; t < len; t += nthreads) {
            float qi = 0.0f;
            for (int qs = 0; qs < KT; ++qs) qi += from_bf16(sQ[t * LQ + qs]) * sN[qs];
            sQi[t] += qi;
        }
    }
    __syncthreads();

    // qn: the row sum of D̄⊙S plus the b·(Q·n_prev) read-out. A matrix-VECTOR
    // product, so it stays scalar — there is no dot for the tensor cores here.
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

    // The two output dots share an output tile, so they share a warp: intra over j,
    // inter over q, combined as num = intra + b[t]·inter in the epilogue.
    //
    // `v` is an OUTPUT index here, so a dhv slice owns its columns outright and
    // nothing carries between slices. The inter dot still contracts dqk, but sC is
    // only [VT, KT], so it walks the dqk slices INSIDE the tile loop, re-staging Q.
    int vtile = VT >> 3;
    for (int vt = 0; vt < NVT; ++vt) {
        int v0 = vt * VT;
        __syncthreads();
        for (int e = tid; e < LP * VT; e += nthreads) {
            int t = e / VT, vs = e - t * VT, v = v0 + vs;
            sV[t * LV + vs] = to_bf16(((t < len) && (v < dhv))
                ? slab_ld(vv, bhBase(bh, H, sHV_B, sHV_H) + (long)(c0 + t) * sHV_S + v) : 0.0f);
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

        // Q·C_prevᵀ contracts dqk, so it needs its own slice loop over the same
        // output tile; the partial sums live in sAcc between the two.
        for (int kt = 0; kt < NKT; ++kt) {
            int q0 = kt * KT;
            __syncthreads();
            for (int e = tid; e < LP * KT; e += nthreads) {
                int t = e / KT, qs = e - t * KT, q = q0 + qs;
                sQ[t * LQ + qs] = to_bf16(((t < len) && (q < dqk))
                    ? slab_ld(qq, bhBase(bh, H, sQK_B, sQK_H) + (long)(c0 + t) * sQK_S + q) : 0.0f);
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
                slab_st(ytil, bhBase(bh, H, sHV_B, sHV_H) + (long)(c0 + t) * sHV_S + v,
                        sAcc[t * LA + vs] / fmaxf(fabsf(sQn[t]), expf(-sM[t])));
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
    MLSTM_SHAPE_ARGS_REC) {
    MLSTM_SHAPE_BIND_REC
    int v0 = blockIdx.x * TV, bh = blockIdx.y;
    int tv = min(TV, dhv - v0);
    int tid = threadIdx.x;
    const int nthreads = MLSTM_NTHREADS;
    const int LQ = dqk + 1, LV = tv + 1;
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
extern "C" __global__ void mlstm_bw_parallel(
    const slab_t* qq, const slab_t* kk, const slab_t* vv,
    const float* ig, const float* fg, const float* fcb,
    const float* cst, const float* nst, const float* mst,
    const float* dcst, const float* dnst,
    const slab_t* ytil, const float* dytil, const float* psiv,
    const float* qnv, const float* msv,
    float* dq, float* dk, float* dv, float* dig, float* dfg,
    MLSTM_SHAPE_ARGS_BW) {
    MLSTM_SHAPE_BIND
    int k = blockIdx.x, bh = blockIdx.y;
    int tid = threadIdx.x, nthreads = blockDim.x;
    int warp = tid >> 5, nwarps = nthreads >> 5;
    int c0 = k * L;
    int len = min(L, T - c0);
    // See `mlstm_bw_parallel`: under CARRY the last chunk's outgoing gradient is the
    // one staged in slot NC, not zero.
    int is_last = (k == NC - 1) && !CARRY;

    const int LP = (L + 15) & ~15;
    const int KP = (dqk + 7) & ~7;
    const int VP = (dhv + 7) & ~7;
    // The `dqk` axis is staged one KT-wide slice at a time rather than whole. Every
    // buffer that spans it — Q, K, and the two `[dhv, dqk]` states, which alone are
    // 57% of the untiled footprint — is sized to the slice, so shared memory stops
    // scaling with the head dim: the block loops `NKT` slices instead of holding
    // them. This is the reference's `siz_b_DHQK` (nx-ai/mlstm_kernels caps it with
    // `get_head_dim_block_size`, min(64, ..), so a wider head raises the trip count
    // and never the footprint). KT >= dqk degenerates to one full-width pass, i.e.
    // exactly the untiled kernel.
    const int KT = (MLSTM_KT < dqk) ? MLSTM_KT : KP;
    const int NKT = (dqk + KT - 1) / KT;
    // The `dhv` twin of KT. Unlike `dqk`, this axis is a CONTRACTION in phases 4/5
    // and an OUTPUT index in phase 2, so the two need opposite treatment: phase 2's
    // slice owns its columns outright, while phases 4 and 5 accumulate across slices.
    const int VT = (MLSTM_VT < dhv) ? MLSTM_VT : VP;
    const int NVT = (dhv + VT - 1) / VT;
    const int LQ = KT + 1, LV = VT + 1, LS = LP + 1;
    const int LK = KT + 1;

    extern __shared__ float sh[];
    float* sQ   = sh;                  // [LP, LQ]   one dqk slice
    float* sK   = sQ + LP * LQ;        // [LP, LQ]   one dqk slice
    float* sV   = sK + LP * LQ;        // [LP, LV]   one dhv slice
    float* sDN  = sV + LP * LV;        // [LP, LV]   d_num,   one dhv slice
    float* sDS  = sDN + LP * LV;       // [LP, LS]   DS, then dS
    float* sC   = sDS + LP * LS;       // [VT, LQ]   C_{k-1}, one (dhv, dqk) tile
    float* sdC  = sC + VT * LQ;        // [VT, LQ]   dC_k,    one (dhv, dqk) tile
    float* sSt  = sdC + VT * LQ;       // [LP, LV]   Σ_q dC·K, summed over dqk slices
    float* sPre = sSt + LP * LV;       // [LP, LV]   Σ_q Q·C,  summed over dqk slices
    float* sN   = sPre + LP * LV;      // [KT]
    float* sdN  = sN + KT;             // [KT]
    float* sFc  = sdN + KT;            // [LP]
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
    float* sDds = sDb + LP;            // [LP, LS]  phase 4's dhv contraction
    float* sDqi = sDds + LP * LS;      // [LP, LK]  phase 5's dQ tile
    float* sDki = sDqi + LP * LK;      // [LP, LK]  phase 5's dK tile
    __shared__ float sRed[512]; // must cover FUSED_THREADS_PAR

    // Stage the `q0`-based dqk slice of Q/K and of the two states. Called once per
    // tile by every phase that walks the dqk axis; each is followed by the barrier
    // its reader needs, so the helper does not sync itself.
    #define STAGE_QK(q0)                                                              \
        for (int e = tid; e < LP * KT; e += nthreads) {                               \
            int t = e / KT, q = e - t * KT;                                           \
            int ok = (t < len) && ((q0) + q < dqk);                                   \
            long base = bhBase(bh, H, sQK_B, sQK_H) + (long)(c0 + t) * sQK_S + (q0) + q; \
            sQ[t * LQ + q] = ok ? slab_ld(qq, base) : 0.0f;                           \
            sK[t * LQ + q] = ok ? slab_ld(kk, base) : 0.0f;                           \
        }
    // The (v0, q0) tile of the two `[dhv, dqk]` states. `n`/`dn` span dqk only, so
    // they are staged by the v0 == 0 pass and left alone by the others.
    #define STAGE_STATE(v0, q0)                                                       \
        for (int e = tid; e < VT * KT; e += nthreads) {                               \
            int vs = e / KT, q = e - vs * KT;                                         \
            int ok = ((v0) + vs < dhv) && ((q0) + q < dqk);                           \
            long off = (long)((v0) + vs) * dqk + (q0) + q;                            \
            sC[vs * LQ + q] =                                                         \
                ok ? cst[((long)bh * (NC + 1) + k) * dhv * dqk + off] : 0.0f;          \
            sdC[vs * LQ + q] = (ok && !is_last)                                       \
                ? dcst[((long)bh * (NC + 1) + (k + 1)) * dhv * dqk + off] : 0.0f;      \
        }                                                                             \
        for (int e = tid; e < KT; e += nthreads) {                                    \
            int ok = (q0) + e < dqk;                                                  \
            sN[e] = ok ? nst[((long)bh * (NC + 1) + k) * dqk + (q0) + e] : 0.0f;       \
            sdN[e] = (ok && !is_last)                                                 \
                ? dnst[((long)bh * (NC + 1) + (k + 1)) * dqk + (q0) + e] : 0.0f;       \
        }

    // The `v0` slice of V and of d_num. d_num needs `psi`/`ytil` reductions that run
    // over the WHOLE dhv row, so the reduction they feed (`sDQn`) is done once up
    // front and only the staging is per slice.
    #define STAGE_V(v0)                                                               \
        for (int e = tid; e < LP * VT; e += nthreads) {                               \
            int t = e / VT, vs = e - t * VT, v = (v0) + vs;                           \
            int ok = (t < len) && (v < dhv);                                          \
            sV[t * LV + vs] = ok                                                      \
                ? slab_ld(vv, bhBase(bh, H, sHV_B, sHV_H) + (long)(c0 + t) * sHV_S + v) : 0.0f; \
            sDN[t * LV + vs] = ok                                                     \
                ? dytil[bhBase(bh, H, sHV_B, sHV_H) + (long)(c0 + t) * sHV_S + v]     \
                    * (1.0f / psiv[(long)bh * T + c0 + t]) : 0.0f;                    \
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
        // Contracts the whole dhv row, so it cannot be confined to a dhv slice;
        // sDN itself is staged per slice by STAGE_V.
        float red = 0.0f;
        for (int v = 0; v < dhv; ++v) red += dytil[gy + v] * slab_ld(ytil, gy + v);
        float dpsi = -red * inv;
        float qn = sQn[t];
        // Grad flows through qn only where it, not the exp(−m) floor, won the max.
        sDQn[t] = (fabsf(qn) > expf(-sM[t])) ? ((qn > 0.0f ? 1.0f : -1.0f) * dpsi) : 0.0f;
    }
    __syncthreads();

    int mtile = LP >> 4, ltile = LP >> 3, vtile = VT >> 3, ktile = KT >> 3;

    // ONE slice loop for every dqk-contracting accumulation. Each of these was its
    // own pass at first, and each pass re-read Q/K/C/dC from global memory — 5 loops
    // × NKT slices, against the single staging the untiled kernel needed. They are
    // merged because none of them reads `sDS`: they only accumulate into buffers of
    // their own, so one staging per slice serves all four.
    //
    //   DS   = Q·Kᵀ            -> sDS   (phase 1)
    //   st   = K·dCᵀ           -> sSt   (phase 2)
    //   pre  = Q·Cᵀ            -> sPre  (phase 2)
    //   da/db n-side           -> sDa/sDb
    //   dg   = ΣdC⊙C + ΣdN⊙N   -> `dgloc`, reduced after the loop
    //
    // Phase 5 cannot join: it consumes the FINAL dS, which phase 4 derives from the
    // completed DS, so it keeps a second slice loop of its own.
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
            for (int k0 = 0; k0 < KT; k0 += 8) {
                ld_a_mk(a, sQ, LQ, m0, k0);
                ld_b_nk(b, sK, LQ, k0, n0);
                mma_16x8x8(d, a, b);
            }
            for (int i = 0; i < 4; ++i) {
                int t = m0 + mma_row(i), j = n0 + mma_col(i);
                sDS[t * LS + j] += d[i];
            }
        }

        // The n-side of da/db: a matrix-vector product, so no dot for the tensor
        // cores and `len` threads is enough. `n`/`dn` span dqk only, so the v0 == 0
        // staging of STAGE_STATE is all this needs.
        STAGE_STATE(0, kt * KT);
        __syncthreads();
        {
            int qlim = min(KT, dqk - kt * KT);
            for (int j = tid; j < len; j += nthreads) {
                float acc = 0.0f, pre_qn = 0.0f;
                for (int q = 0; q < qlim; ++q) {
                    acc += sdN[q] * sK[j * LQ + q];
                    pre_qn += sQ[j * LQ + q] * sN[q];
                }
                sDa[j] += acc;
                sDb[j] += sDQn[j] * pre_qn;
            }
            for (int e = tid; e < KT; e += nthreads) dgloc += sdN[e] * sN[e];
        }
    }
    __syncthreads();
    // The D̄ epilogue phase 1 owed, applied once the whole contraction is in.
    for (int e = tid; e < LP * LS; e += nthreads) {
        int t = e / LS, j = e - t * LS;
        float val = 0.0f;
        if (t < len && j <= t) val = expf(sFc[t] - sFc[j] + sIg[j] - sM[t]) * sDS[e];
        sDS[e] = val;
    }
    __syncthreads();

    // Phase 2: dV, plus the `st` and `pre` products that `da`/`db` reduce over v,
    // and phase 4's dhv contraction, which shares the same V/dN staging.
    //
    // `v` is an OUTPUT index for dV, so a dhv slice owns its columns outright. But
    // `st`/`pre` contract dqk INSIDE that slice, so this is a nested loop: the outer
    // pass stages a dhv slice, the inner one sweeps dqk into `sSt`/`sPre`, which are
    // therefore only [LP, VT] and are re-zeroed per outer pass.
    //
    // `dds` (phase 4) contracts dhv, so it cannot finish inside one slice — it
    // accumulates into sDds across the outer loop and its epilogue runs after.
    for (int e = tid; e < LP * LS; e += nthreads) sDds[e] = 0.0f;
    __syncthreads();
    for (int vt = 0; vt < NVT; ++vt) {
        int v0 = vt * VT;
        __syncthreads();
        STAGE_V(v0);
        for (int e = tid; e < LP * LV; e += nthreads) { sSt[e] = 0.0f; sPre[e] = 0.0f; }
        __syncthreads();

        for (int kt = 0; kt < NKT; ++kt) {
            __syncthreads();
            STAGE_QK(kt * KT);
            STAGE_STATE(v0, kt * KT);
            __syncthreads();
            for (int tile = warp; tile < mtile * vtile; tile += nwarps) {
                int m0 = (tile / vtile) << 4, n0 = (tile % vtile) << 3;
                float dst[4]  = {0.0f, 0.0f, 0.0f, 0.0f};  // Σ_q dC[v][q]·K[j][q]
                float dpre[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Σ_q Q[t][q]·C[v][q]
                unsigned a[4], b[2];
                for (int k0 = 0; k0 < KT; k0 += 8) {
                    ld_a_mk(a, sK, LQ, m0, k0);
                    ld_b_nk(b, sdC, LQ, k0, n0);
                    mma_16x8x8(dst, a, b);
                    ld_a_mk(a, sQ, LQ, m0, k0);
                    ld_b_nk(b, sC, LQ, k0, n0);
                    mma_16x8x8(dpre, a, b);
                }
                for (int i = 0; i < 4; ++i) {
                    int r = m0 + mma_row(i), vs = n0 + mma_col(i);
                    sSt[r * LV + vs] += dst[i];
                    sPre[r * LV + vs] += dpre[i];
                }
            }
            // dg's C-term: this (v0, q0) tile of dC⊙C. The pad is zero in both
            // operands, so it contributes nothing and needs no masking.
            for (int e = tid; e < VT * KT; e += nthreads) {
                int vs = e / KT, q = e - vs * KT;
                dgloc += sdC[vs * LQ + q] * sC[vs * LQ + q];
            }
        }
        __syncthreads();

        // dnum, then the epilogue phase 2 owed: dV and the v-side of da/db.
        for (int tile = warp; tile < mtile * vtile; tile += nwarps) {
            int m0 = (tile / vtile) << 4, n0 = (tile % vtile) << 3;
            float dnum[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Σ_t DS[t][j]·dN[t][v]
            unsigned a[4], b[2];
            // DS[t][j] is zero for j > t, so contracting over ALL t is the same as t >= j.
            for (int k0 = 0; k0 < LP; k0 += 8) {
                ld_a_km(a, sDS, LS, m0, k0);   // Aᵀ in memory: DS is [t, j], we want [j, t]
                ld_b_kn(b, sDN, LV, k0, n0);
                mma_16x8x8(dnum, a, b);
            }
            for (int i = 0; i < 4; ++i) {
                int r = m0 + mma_row(i), vs = n0 + mma_col(i);
                int v = v0 + vs;             // r is `j` for dv, `t` for db
                if (r < len && v < dhv) {
                    float st = sSt[r * LV + vs];
                    dv[bhBase(bh, H, sHV_B, sHV_H) + (long)(c0 + r) * sHV_S + v] = dnum[i] + sA[r] * st;
                    atomicAdd(&sDa[r], sV[r * LV + vs] * st);
                    atomicAdd(&sDb[r], sDN[r * LV + vs] * sPre[r * LV + vs]);
                }
            }
        }

        // Phase 4's dhv contraction, accumulated across the slices.
        for (int tile = warp; tile < mtile * ltile; tile += nwarps) {
            int m0 = (tile / ltile) << 4, n0 = (tile % ltile) << 3;
            float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};   // Σ_v dN[t][v]·V[j][v]
            unsigned a[4], b[2];
            for (int k0 = 0; k0 < VT; k0 += 8) {
                ld_a_mk(a, sDN, LV, m0, k0);
                ld_b_nk(b, sV, LV, k0, n0);
                mma_16x8x8(d, a, b);
            }
            for (int i = 0; i < 4; ++i) {
                int t = m0 + mma_row(i), j = n0 + mma_col(i);
                sDds[t * LS + j] += d[i];
            }
        }
    }
    __syncthreads();

    // Phase 4's epilogue: dDS -> (dS, P), once the dhv contraction is complete. dS
    // overwrites DS in place; the warp that owns an output tile is the only one that
    // reads or writes those elements.
    for (int tile = warp; tile < mtile * ltile; tile += nwarps) {
        int m0 = (tile / ltile) << 4, n0 = (tile % ltile) << 3;
        for (int i = 0; i < 4; ++i) {
            int t = m0 + mma_row(i), j = n0 + mma_col(i);
            float out = 0.0f;
            if (t < len && j <= t) {
                float dds = sDds[t * LS + j] + sDQn[t];
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
    //
    // Here `q` is an OUTPUT index, not a contraction: a slice owns its own columns of
    // dQ/dK outright, so the loop needs no cross-slice accumulation — each pass
    // stages its slice and writes the columns it owns.
    // The two dS dots contract `j`/`t` and are done once per dqk slice; `dqx`/`dks`
    // contract dhv, so they need the dhv slices nested inside — the accumulator sits
    // in registers across that inner loop and only the epilogue is per (t, q).
    for (int kt = 0; kt < NKT; ++kt) {
        __syncthreads();
        STAGE_QK(kt * KT);
        __syncthreads();
        for (int tile = warp; tile < mtile * ktile; tile += nwarps) {
            int m0 = (tile / ktile) << 4, n0 = (tile % ktile) << 3;
            float dqi[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Σ_j dS[t][j]·K[j][q]
            float dki[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Σ_t dS[t][j]·Q[t][q]
            unsigned a[4], b[2];
            for (int k0 = 0; k0 < LP; k0 += 8) {
                ld_a_mk(a, sDS, LS, m0, k0);   // dS as [t, j], contracting j
                ld_b_kn(b, sK, LQ, k0, n0);
                mma_16x8x8(dqi, a, b);
                ld_a_km(a, sDS, LS, m0, k0);   // dSᵀ, contracting t
                ld_b_kn(b, sQ, LQ, k0, n0);
                mma_16x8x8(dki, a, b);
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
            STAGE_STATE(v0, kt * KT);
            __syncthreads();
            for (int tile = warp; tile < mtile * ktile; tile += nwarps) {
                int m0 = (tile / ktile) << 4, n0 = (tile % ktile) << 3;
                float dqx[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Σ_v dN[t][v]·C[v][q]
                float dks[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Σ_v V[j][v]·dC[v][q]
                unsigned a[4], b[2];
                for (int k0 = 0; k0 < VT; k0 += 8) {
                    ld_a_mk(a, sDN, LV, m0, k0);
                    ld_b_kn(b, sC, LQ, k0, n0);
                    mma_16x8x8(dqx, a, b);
                    ld_a_mk(a, sV, LV, m0, k0);
                    ld_b_kn(b, sdC, LQ, k0, n0);
                    mma_16x8x8(dks, a, b);
                }
                for (int i = 0; i < 4; ++i) {
                    int r = m0 + mma_row(i), qs = n0 + mma_col(i);
                    sDqi[r * LK + qs] += sB[r] * dqx[i];
                    sDki[r * LK + qs] += sA[r] * dks[i];
                }
            }
        }
        __syncthreads();
        // `n`/`dn` span dqk only, so the last STAGE_STATE left them correct.
        for (int e = tid; e < LP * KT; e += nthreads) {
            int r = e / KT, qs = e - r * KT, q = kt * KT + qs;
            if (r < len && q < dqk) {
                dq[bhBase(bh, H, sQK_B, sQK_H) + (long)(c0 + r) * sQK_S + q] =
                    sDqi[r * LK + qs] + sB[r] * sDQn[r] * sN[qs];
                dk[bhBase(bh, H, sQK_B, sQK_H) + (long)(c0 + r) * sQK_S + q] =
                    sDki[r * LK + qs] + sA[r] * sdN[qs];
            }
        }
    }
    __syncthreads();

    // dg was accumulated per-thread in the merged slice loop; only the block-wide
    // reduction of those partials is left.
    sRed[tid] = dgloc;
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
