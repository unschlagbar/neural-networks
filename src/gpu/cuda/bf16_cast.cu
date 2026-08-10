// fp32 <-> bf16 casts, behind BF16_CAST. Needs only <cuda_bf16.h>, which is why
// it is its own module: a machine that cannot build the cooperative kernels can
// still store activations in bf16.

#ifdef BF16_CAST
#include <cuda_bf16.h>

// fp32 -> bf16 and back, for activations that are SAVED for backward but whose
// stored precision does not need fp32's 24-bit mantissa (see `gpu::bf16`).
//
// Round-to-nearest-even, not truncation: `__float2bfloat16` is the RNE intrinsic.
// Truncating instead would bias every stored value toward zero, and the sLSTM's
// saved slabs feed a product chain that is summed over T — a systematic bias there
// accumulates over the sequence, where symmetric rounding noise cancels.
//
// Four elements per thread through `float4`/`ushort4`. A scalar thread has only
// 4 B loaded and 2 B stored in flight, so the launch needs very high occupancy to
// keep enough bytes outstanding to saturate DRAM, and these launches are mostly
// small (p50 ~2 us) — they never get there. The loop is grid-stride so the launch
// stays correct either way: the host sizes the grid for the vector body, and a
// misaligned buffer that falls back to the scalar path still covers all n.
extern "C" __global__ void cast_f32_to_bf16(const float* src, __nv_bfloat16* dst, int n) {
    int stride = gridDim.x * blockDim.x;
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (((size_t)src % 16 == 0) && ((size_t)dst % 8 == 0)) {
        int n4 = n >> 2;
        for (int i = i0; i < n4; i += stride) {
            float4 v = reinterpret_cast<const float4*>(src)[i];
            ushort4 o;
            o.x = __bfloat16_as_ushort(__float2bfloat16(v.x));
            o.y = __bfloat16_as_ushort(__float2bfloat16(v.y));
            o.z = __bfloat16_as_ushort(__float2bfloat16(v.z));
            o.w = __bfloat16_as_ushort(__float2bfloat16(v.w));
            reinterpret_cast<ushort4*>(dst)[i] = o;
        }
        for (int t = (n4 << 2) + i0; t < n; t += stride) dst[t] = __float2bfloat16(src[t]);
    } else {
        for (int i = i0; i < n; i += stride) dst[i] = __float2bfloat16(src[i]);
    }
}

extern "C" __global__ void cast_bf16_to_f32(const __nv_bfloat16* src, float* dst, int n) {
    int stride = gridDim.x * blockDim.x;
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (((size_t)src % 8 == 0) && ((size_t)dst % 16 == 0)) {
        int n4 = n >> 2;
        for (int i = i0; i < n4; i += stride) {
            ushort4 v = reinterpret_cast<const ushort4*>(src)[i];
            float4 o;
            o.x = __bfloat162float(__ushort_as_bfloat16(v.x));
            o.y = __bfloat162float(__ushort_as_bfloat16(v.y));
            o.z = __bfloat162float(__ushort_as_bfloat16(v.z));
            o.w = __bfloat162float(__ushort_as_bfloat16(v.w));
            reinterpret_cast<float4*>(dst)[i] = o;
        }
        for (int t = (n4 << 2) + i0; t < n; t += stride) dst[t] = __bfloat162float(src[t]);
    } else {
        for (int i = i0; i < n; i += stride) dst[i] = __bfloat162float(src[i]);
    }
}
#endif
