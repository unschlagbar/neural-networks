// File-scope prelude: MUST be first in the concatenation.

// STORAGE dtype of the sLSTM's saved-for-backward slabs. With SLAB_BF16 the slabs
// that do NOT carry the stabilizer (`zt`, `ot`, `h_prev`) live in global memory as
// bf16: half the bytes, and since each is written once by the forward and read once
// by the backward, narrowing them costs one convert on either side and saves a full
// [B, T, H] round-trip's worth of bandwidth.
#ifdef SLAB_BF16
#include <cuda_bf16.h>
typedef __nv_bfloat16 slab_t;
__device__ __forceinline__ float slab_ld(const slab_t* p, long long i) {
    return __bfloat162float(p[i]);
}
__device__ __forceinline__ void slab_st(slab_t* p, long long i, float v) {
    p[i] = __float2bfloat16(v);
}
#else
typedef float slab_t;
__device__ __forceinline__ float slab_ld(const slab_t* p, long long i) { return p[i]; }
__device__ __forceinline__ void slab_st(slab_t* p, long long i, float v) { p[i] = v; }
#endif

// STORAGE dtype of the sLSTM's STABILIZER-carrying saved tensors: `c`, `n`, their
// entry values, and `i_prime`/`f_prime`. Separate from `slab_t` because the two are
// different bets. The `slab_t` group is bounded by construction (a tanh, a sigmoid, a
// normalized ratio); this group carries the exp(-m) scale, which is what keeps the
// recurrence from overflowing, and an absolute error in an exponent is a
// multiplicative error in the value it guards.
#ifdef STATE_BF16
typedef __nv_bfloat16 state_t;
__device__ __forceinline__ float state_ld(const state_t* p, long long i) {
    return __bfloat162float(p[i]);
}
__device__ __forceinline__ void state_st(state_t* p, long long i, float v) {
    p[i] = __float2bfloat16(v);
}
#else
typedef float state_t;
__device__ __forceinline__ float state_ld(const state_t* p, long long i) { return p[i]; }
__device__ __forceinline__ void state_st(state_t* p, long long i, float v) { p[i] = v; }
#endif
