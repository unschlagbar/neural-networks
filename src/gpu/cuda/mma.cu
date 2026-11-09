// Tensor-core primitives shared by every kernel that contracts on the bf16 MMA
// unit. Compiled whenever some consumer asks for them (MMA_TF32 for the fused
// mLSTM, SLSTM_MMA for the batched sLSTM); `mma.sync` does not exist at NVRTC's
// default target, so a build that wants neither must not see this at all.
#if MMA_TF32 || defined(SLSTM_MMA)

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

// A from a row-major [M, K] tile in ONE instruction, the hardware's own version of
// `ldb_a_mk`: `ldmatrix` gathers the four 8x8 sub-matrices an A fragment is made of and
// distributes them across the warp in exactly the layout `mma.sync` wants, so a lane
// supplies one row address instead of issuing four loads and assembling the pairs.
__device__ __forceinline__ void ldm_a_mk(unsigned* a, const bf16s_t* s, int ld, int m0, int k0) {
    const int lane = threadIdx.x & 31;
    const int row = m0 + (lane & 7) + ((lane >> 3) & 1) * 8;
    const int col = k0 + ((lane >> 4) & 1) * 8;
    const unsigned addr =
        (unsigned)__cvta_generic_to_shared(s + (long long)row * ld + col);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(addr));
}

// Where accumulator register `i` lands in the 16x8 output tile.
__device__ __forceinline__ int mma_row(int i) { return ((threadIdx.x & 31) >> 2) + ((i & 2) ? 8 : 0); }
__device__ __forceinline__ int mma_col(int i) { return (((threadIdx.x & 31) & 3) << 1) + (i & 1); }

#endif // MMA_TF32 || SLSTM_MMA
