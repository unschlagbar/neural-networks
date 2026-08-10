//! bf16 **storage** for saved activations.
//!
//! Training memory on this model is dominated not by weights but by the tensors a
//! forward pass saves for its backward. At the backbone's shape (`T` = a window's
//! word count, `H` = `WORD_HIDDEN`) one sLSTM holds nine `[B, T, H]` slabs plus a
//! `[B, T, 4H]` gate buffer, and there are `WORD_BLOCKS` of them. Those slabs are
//! written once, read once, and never enter a reduction in their stored form —
//! which makes their *storage* precision a free parameter, independent of the
//! precision the math runs at.
//!
//! A [`BTensor`] is such a slab held as bf16: same shape, half the bytes. Kernels
//! that read one convert to fp32 on load and compute in fp32 exactly as before, so
//! the arithmetic is unchanged and only the round-trip through global memory is
//! narrowed.
//!
//! # Why bf16 and not fp16
//!
//! bf16 keeps fp32's 8 exponent bits and drops the mantissa to 7 stored bits
//! (fp16 does the reverse: 5 exponent bits, 10 of mantissa). Nothing stored here
//! can overflow
//! or flush to zero that would not have in fp32 — the dynamic range is identical —
//! so no loss scaling is needed. What is lost is precision: ~3 decimal digits
//! instead of ~7. That is the trade, and it is the same one every mixed-precision
//! framework makes for activations.
//!
//! # What must NOT be stored here
//!
//! The rule is taken from the reference implementation (NX-AI `mlstm_kernels`,
//! the Triton kernels behind the xLSTM paper), which is explicit about its dtype
//! split: `matQ`/`matK`/`matV`/`matC` and the output `matH` are loaded and stored
//! at the kernel's `DTYPE` (bf16 under autocast), while **every** accumulator is
//! `tl.zeros(..., dtype=tl.float32)` and — the part that matters here — the gate
//! vectors and the stabilizer are pinned to fp32:
//!
//! ```text
//! vecB_val        = tl.load(...).to(tl.float32)      # cumulative log-forget
//! vecI_val        = tl.load(...).to(tl.float32)      # input-gate logit
//! scaMinter_km1   = tl.load(...).to(tl.float32)      # inter-chunk stabilizer
//! vecM_combine    = tl.maximum(vecB + scaMinter, vecM_intra)
//! tl.store(vecMout_ptr, vecM_combine_val.to(tl.float32))
//! tl.store(vecNout_ptr, vecH_denom_val.to(tl.float32))
//! ```
//!
//! and its inter-chunk recurrent state is accumulated *and stored* in fp32
//! (`matC_k_val = tl.zeros(..., dtype=tl.float32)`, stored `.to(tl.float32)`)
//! even though the same `matC` is *read back* as bf16 by the parallel kernels.
//!
//! The reason is structural, not conservatism. The stabilizer `m` is an exponent:
//! every value it guards is formed as `exp(x - m)`, so an absolute error `eps` in
//! `m` multiplies the result by `exp(eps)`. bf16's mantissa near a typical `m` of
//! order 10 quantizes to steps of ~0.06, i.e. a ~6% multiplicative error injected
//! into the very quantity whose job is to keep the recurrence bounded. `c` and `n`
//! carry the matching `exp(-m)` factor and cancel it in the ratio `c/n`, so they
//! must be stored at the same precision as each other and as `m`.
//!
//! Concretely, in this codebase that means:
//!
//! | fp32 (pinned)                                   | bf16 (storage)                       |
//! |-------------------------------------------------|--------------------------------------|
//! | sLSTM `c`, `n`, `c_prev`, `n_prev`, `m_state`   | sLSTM `zt`, `ot`, `h_prev`, `x_saved` |
//! | mLSTM `mst`, `msv`, `psiv`, `qnv`, `cst`, `nst` | mLSTM `qh`, `kh`, `vh`, `o`, `yhat`, `ytil` |
//! | gate logits `igh`, `fgh`, all cumulative sums   | the sLSTM gate buffer `g`            |
//! | every weight, gradient and optimizer moment     |                                      |
//!
//! `i_prime`/`f_prime` sit on the fp32 side too: they are `exp(i - m)` and
//! `exp(fm - m)`, i.e. the stabilizer's own outputs, and the backward multiplies a
//! chain of `f_prime` together across timesteps where relative error compounds.
//!
//! # The other half: bf16 GEMMs
//!
//! Storage is only one of the two places the reference uses bf16. The other is the
//! matmuls themselves: its projections run under
//! `@custom_fwd(cast_inputs=autocast_kernel_dtype)` with `autocast_kernel_dtype =
//! "bfloat16"`, so the operands entering cuBLAS are bf16 while the accumulator
//! stays fp32. [`ops::matmul_bf16_into`](super::ops::matmul_bf16_into) is that
//! path here — `cublasGemmEx` with `CUDA_R_16BF` operands and
//! `CUBLAS_COMPUTE_32F` — and [`ops::GemmBf16`](super::ops::GemmBf16) owns the
//! per-layer staging that narrows an fp32 master weight per call.
//!
//! Measured on an RTX 5080 at 4096³: **10.9 → 92.8 TFLOP/s**, an 8.5x speedup on
//! the GEMM in isolation. End-to-end it is worth ~4% of a training step, because
//! the backbone is ~89% of the step and is bound by the sLSTM's serial recurrence
//! and the fused mLSTM kernels rather than by cuBLAS.
//!
//! Logit heads are excluded and stay fp32: their output is exponentiated by the
//! softmax/cross-entropy, so a bf16 wobble on a logit becomes a multiplicative
//! error on a probability, and there is one such GEMM per step against the
//! backbone's many.
//!
//! # Enabling
//!
//! On by default; `GPU_NO_BF16=1` forces fp32 everywhere — storage *and* GEMMs —
//! which is the A/B baseline and the escape hatch. Also silently unavailable, and
//! therefore fp32, when `<cuda_bf16.h>` was not found at startup (see
//! `Kernels::has_bf16`).
//!
//! # What this did NOT buy
//!
//! VRAM, which is what prompted the work. At the production config the largest
//! window that fits is 2048 words either way. The saved recurrent tensors are
//! simply not where the memory is: only 4 of 16 backbone blocks are sLSTMs
//! (`hierarchical.rs` builds one when `i % 4 == 0`), the backbone runs B = 1, and
//! the dominant consumers are the per-block `Act` pool and the SwiGLU activations
//! at `[N, U]`, both of which are fp32 temporaries rather than slabs. Note also
//! that `mem_get_info` reports the CUDA async pool's *reserved* size, which does
//! not shrink when blocks are freed — measure capacity (the largest window that
//! fits) rather than reported usage.

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use super::{DTensor, Gpu};
use crate::tensor::MAX_RANK;

/// Whether saved activations are stored as bf16. On unless `GPU_NO_BF16` is set.
///
/// Read through a `OnceLock`: the answer must not change within a process, since a
/// forward and its backward have to agree on the layout of what was saved.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("GPU_NO_BF16").is_err())
}

/// Whether bf16 storage is usable on this `Gpu` — the flag above *and* the cast
/// kernels having compiled. Every allocation site goes through this.
pub fn active(gpu: &Gpu) -> bool {
    enabled() && gpu.kernels.has_bf16
}

/// A dense, row-major, contiguous **bf16** tensor in device memory.
///
/// Shape metadata mirrors [`DTensor`]; the buffer is `u16` because that is the
/// storage width and Rust has no native bf16 — nothing on the host interprets
/// these bits, they only travel between the cast kernels and the consumers.
pub struct BTensor {
    pub shape: [usize; MAX_RANK],
    pub rank: usize,
    pub buf: CudaSlice<u16>,
}

impl BTensor {
    /// Allocate an **uninitialized** bf16 tensor. Like [`DTensor::uninit`], the
    /// caller must write every element before reading it.
    pub fn uninit(gpu: &Gpu, dims: &[usize]) -> Self {
        assert!(dims.len() <= MAX_RANK, "rank {} exceeds MAX_RANK", dims.len());
        let n: usize = dims.iter().product();
        // SAFETY: same contract as `DTensor::uninit` — fully written before read.
        let buf = unsafe { gpu.stream.alloc::<u16>(n) }.expect("bf16 device alloc");
        let mut shape = [0usize; MAX_RANK];
        shape[..dims.len()].copy_from_slice(dims);
        Self { shape, rank: dims.len(), buf }
    }

    /// Narrow `src` into this tensor (fp32 -> bf16, round-to-nearest-even).
    pub fn store(&mut self, gpu: &Gpu, src: &DTensor) {
        let n = src.len();
        assert!(n <= self.capacity(), "store: {n} elements into a {} buffer", self.capacity());
        let f = gpu.kernels.get("cast_f32_to_bf16");
        let n_i32 = n as i32;
        let mut b = gpu.stream.launch_builder(&f);
        b.arg(&src.buf).arg(&mut self.buf).arg(&n_i32);
        unsafe { b.launch(LaunchConfig::for_num_elems(n.div_ceil(4) as u32)) }
            .expect("cast to bf16");
    }

    /// Widen this tensor into `dst` (bf16 -> fp32, exact: every bf16 value is a
    /// representable fp32). `dst` is presented at this tensor's shape.
    pub fn load(&self, gpu: &Gpu, dst: &mut DTensor) {
        let n = self.len();
        assert!(n <= dst.capacity(), "load: {n} elements into a {} buffer", dst.capacity());
        dst.shrink_to(self.dims());
        let f = gpu.kernels.get("cast_bf16_to_f32");
        let n_i32 = n as i32;
        let mut b = gpu.stream.launch_builder(&f);
        b.arg(&self.buf).arg(&mut dst.buf).arg(&n_i32);
        unsafe { b.launch(LaunchConfig::for_num_elems(n.div_ceil(4) as u32)) }
            .expect("cast from bf16");
    }

    /// Present this tensor at `dims`, which must fit within its allocation — the
    /// bf16 twin of [`DTensor::shrink_to`], with the same pooled-buffer rationale.
    pub fn shrink_to(&mut self, dims: &[usize]) {
        assert!(dims.len() <= MAX_RANK, "shrink rank {} exceeds MAX_RANK", dims.len());
        let n: usize = dims.iter().product();
        assert!(n <= self.buf.len(), "shrink_to {n} exceeds the {} allocated", self.buf.len());
        self.shape = [0usize; MAX_RANK];
        self.shape[..dims.len()].copy_from_slice(dims);
        self.rank = dims.len();
    }

    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.shape[..self.rank]
    }

    /// Elements the current shape describes (not the allocation — see
    /// [`capacity`](Self::capacity)).
    #[inline]
    pub fn len(&self) -> usize {
        self.shape[..self.rank].iter().product()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buf.len() == 0
    }
}

/// A saved activation slab, held as bf16 when that is available and fp32 when it
/// is not.
///
/// This is what a layer stores instead of a bare [`DTensor`]. The two variants are
/// deliberately not a runtime choice per call: [`Slab::new`] picks once from
/// [`active`], and forward and backward then agree by construction.
///
/// Reading is always through [`get`](Self::get), which hands back an fp32 view —
/// either the tensor itself (fp32 variant, zero cost) or a widened copy into a
/// caller-provided scratch buffer (bf16 variant). Consumers therefore need no
/// knowledge of which is in play.
pub enum Slab {
    F32(DTensor),
    Bf16(BTensor),
}

impl Slab {
    /// An uninitialized slab of the given shape, bf16 where available.
    pub fn new(gpu: &Gpu, dims: &[usize]) -> Self {
        if active(gpu) {
            Slab::Bf16(BTensor::uninit(gpu, dims))
        } else {
            Slab::F32(DTensor::uninit(gpu, dims))
        }
    }

    /// Elements the slab currently presents.
    pub fn len(&self) -> usize {
        match self {
            Slab::F32(t) => t.len(),
            Slab::Bf16(t) => t.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dims(&self) -> &[usize] {
        match self {
            Slab::F32(t) => t.dims(),
            Slab::Bf16(t) => t.dims(),
        }
    }

    /// Elements allocated, for reuse checks.
    pub fn capacity(&self) -> usize {
        match self {
            Slab::F32(t) => t.capacity(),
            Slab::Bf16(t) => t.capacity(),
        }
    }

    /// Present the slab at `dims` (must fit the allocation), so a steady-state loop
    /// reuses one buffer across the varying window shapes.
    pub fn shrink_to(&mut self, dims: &[usize]) {
        match self {
            Slab::F32(t) => t.shrink_to(dims),
            Slab::Bf16(t) => t.shrink_to(dims),
        }
    }

    /// Bytes this slab occupies — what the whole exercise is about.
    pub fn bytes(&self) -> usize {
        match self {
            Slab::F32(t) => t.capacity() * 4,
            Slab::Bf16(t) => t.capacity() * 2,
        }
    }

    /// Narrow `src` into this slab. For the fp32 variant this is a device copy; for
    /// bf16 it is the cast kernel.
    pub fn store(&mut self, gpu: &Gpu, src: &DTensor) {
        match self {
            Slab::F32(t) => t.copy_from(gpu, src),
            Slab::Bf16(t) => t.store(gpu, src),
        }
    }

    /// True when reading requires widening into scratch — i.e. the bf16 variant.
    pub fn needs_widen(&self) -> bool {
        matches!(self, Slab::Bf16(_))
    }

    /// An fp32 view of this slab.
    ///
    /// `scratch` is only touched for the bf16 variant, where it receives the widened
    /// copy; the fp32 variant ignores it and returns the slab directly, so the
    /// no-bf16 path costs exactly what it did before this module existed.
    pub fn get<'a>(&'a self, gpu: &Gpu, scratch: &'a mut DTensor) -> &'a DTensor {
        match self {
            Slab::F32(t) => t,
            Slab::Bf16(t) => {
                t.load(gpu, scratch);
                scratch
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    /// A bf16 round-trip must preserve value to bf16's ~3 decimal digits, and must
    /// be unbiased — the mean error over many values has to sit near zero, which is
    /// what distinguishes round-to-nearest-even from truncation. Truncation would
    /// pull every magnitude toward zero and show up here as a one-sided mean.
    #[test]
    fn bf16_roundtrip_is_close_and_unbiased() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        if !gpu.kernels.has_bf16 {
            eprintln!("skipping: bf16 kernels unavailable");
            return;
        }
        let t = Tensor::random(&[64, 64], 3.0);
        let d = DTensor::from_host(&gpu, &t);
        let mut b = BTensor::uninit(&gpu, &[64, 64]);
        b.store(&gpu, &d);
        let mut back = DTensor::uninit(&gpu, &[64, 64]);
        b.load(&gpu, &mut back);
        let got = back.to_host(&gpu);

        let mut worst = 0.0;
        let mut signed = 0.0;
        for (a, g) in t.data.iter().zip(got.data.iter()) {
            let rel = (a - g).abs() / a.abs().max(1e-6);
            worst = f32::max(worst, rel);
            signed += a - g;
        }
        // bf16 stores 7 explicit mantissa bits (+1 implicit), so round-to-nearest
        // has a half-ulp bound of 2^-8 = 3.91e-3 relative.
        assert!(worst < 3.91e-3, "bf16 round-trip relative error {worst} too large");
        let mean_bias = signed / t.data.len() as f32;
        assert!(
            mean_bias.abs() < 1e-3,
            "bf16 rounding is biased ({mean_bias}); expected round-to-nearest-even"
        );
    }

    /// The vectorized cast handles 4 elements per thread and leaves `n % 4` to a
    /// scalar tail. `bf16_roundtrip_is_close_and_unbiased` uses 4096 elements — a
    /// multiple of 4 — so it never exercises that tail. Every remainder class must
    /// round-trip, including sizes below one vector.
    #[test]
    fn bf16_roundtrip_covers_the_vector_tail() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        if !gpu.kernels.has_bf16 {
            return;
        }
        for n in [1, 2, 3, 4, 5, 7, 8, 17, 63, 255, 257, 1023, 4097] {
            let t = Tensor::random(&[n], 3.0);
            let d = DTensor::from_host(&gpu, &t);
            let mut b = BTensor::uninit(&gpu, &[n]);
            b.store(&gpu, &d);
            let mut back = DTensor::uninit(&gpu, &[n]);
            b.load(&gpu, &mut back);
            let got = back.to_host(&gpu);
            for (i, (a, g)) in t.data.iter().zip(got.data.iter()).enumerate() {
                let rel = (a - g).abs() / a.abs().max(1e-6);
                assert!(rel < 3.91e-3, "n={n} element {i}: {a} round-tripped to {g}");
            }
        }
    }

    /// The point of the exercise: a bf16 slab must occupy half the bytes of the
    /// fp32 one it replaces.
    #[test]
    fn bf16_slab_is_half_the_bytes() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        if !gpu.kernels.has_bf16 {
            return;
        }
        let dims = [2, 512, 256];
        let f32_bytes = Slab::F32(DTensor::uninit(&gpu, &dims)).bytes();
        let bf_bytes = Slab::Bf16(BTensor::uninit(&gpu, &dims)).bytes();
        assert_eq!(f32_bytes, 2 * bf_bytes);
    }
}
