// File-scope prelude: MUST be first in the concatenation.

// STORAGE dtype of the sLSTM's saved-for-backward slabs. With SLAB_BF16 the slabs
// that do NOT carry the stabilizer (`zt`, `ot`, `h_prev`) live in global memory as
// bf16: half the bytes, and since each is written once by the forward and read once
// by the backward, narrowing them costs one convert on either side and saves a full
// [B, T, H] round-trip's worth of bandwidth.
//
// Arithmetic is unaffected. `slab_ld` widens to fp32 on load and `slab_st` narrows
// on store, so every recurrence below computes in fp32 exactly as it did before;
// only the bits that sit in HBM between the two passes are narrower.
//
// `c`, `n`, `c_entry`, `n_entry`, `i_prime` and `f_prime` are deliberately NOT in
// this set. They all carry the exp(-m) stabilizer factor (i' and f' *are*
// exp(·-m)), and an absolute error eps in an exponent becomes a multiplicative
// exp(eps) error in the value it guards. The reference kernels (NX-AI
// mlstm_kernels) pin exactly this group to fp32 — `vecM`/`vecN` are stored
// `.to(tl.float32)` even where Q/K/V are bf16. See the table in `gpu::bf16`.
//
// This must sit at file scope, ahead of every kernel: both the cooperative
// (`slstm_fused_time`) and the eager (`slstm_step_fused`) paths write the SAME
// slab buffers, so they have to agree on the dtype, and they are compiled into
// different modules.
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
//
// FlashRNN (NX-AI) stores exactly this group in bf16 by default — `FLASHRNN_DTYPE_S`
// — and computes the pointwise in bf16 intrinsics besides, guarded by an
// unconditional `n = max(n, 1)`. `STATE_BF16` is here to measure that claim against
// our fp32 reference rather than argue about it. Arithmetic stays fp32 either way:
// `state_ld` widens on load and `state_st` narrows on store, so only the bits that sit
// in HBM between the forward and the backward change.
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
