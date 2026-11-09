//! bf16 **storage** for saved activations.

use cudarc::driver::PushKernelArg;

use super::{GTensor, Gpu};

/// Whether saved activations are stored as bf16. On unless `GPU_NO_BF16` is set.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("GPU_NO_BF16").is_err())
}

/// Whether bf16 storage is usable on this `Gpu` — the flag above *and* the cast
/// kernels having compiled. Every allocation site goes through this.
pub fn active(gpu: &Gpu) -> bool {
    enabled() && gpu.kernels.has_bf16
}

/// The bf16 storage width: `u16`, because that is what bf16 occupies and Rust has no
/// native type for it — nothing on the host interprets these bits, they only travel
/// between the cast kernels and the consumers. Shape handling, pooling and arena
/// windows are inherited from the generic [`GTensor`]; only the two casts below are
/// specific to this width.
impl GTensor<u16> {
    /// Narrow `src` into this tensor (fp32 -> bf16, round-to-nearest-even).
    pub fn store(&mut self, gpu: &Gpu, src: &GTensor<f32>) {
        let n = src.len();
        assert!(
            n <= self.capacity(),
            "store: {n} elements into a {} buffer",
            self.capacity()
        );
        let f = gpu.kernels.get("cast_f32_to_bf16");
        let n_i32 = n as i32;
        let mut b = gpu.stream.launch_builder(&f);
        b.arg(&src.buf).arg(&mut self.buf).arg(&n_i32);
        unsafe { b.launch(super::ops::elem_cfg(gpu, n.div_ceil(4) as u32)) }
            .expect("cast to bf16");
    }

    /// Widen this tensor into `dst` (bf16 -> fp32, exact: every bf16 value is a
    /// representable fp32). `dst` is presented at this tensor's shape.
    pub fn load(&self, gpu: &Gpu, dst: &mut GTensor<f32>) {
        let n = self.len();
        assert!(
            n <= dst.capacity(),
            "load: {n} elements into a {} buffer",
            dst.capacity()
        );
        dst.shrink_to(self.dims());
        let f = gpu.kernels.get("cast_bf16_to_f32");
        let n_i32 = n as i32;
        let mut b = gpu.stream.launch_builder(&f);
        b.arg(&self.buf).arg(&mut dst.buf).arg(&n_i32);
        unsafe { b.launch(super::ops::elem_cfg(gpu, n.div_ceil(4) as u32)) }
            .expect("cast from bf16");
    }
}

/// A saved activation slab, held as bf16 when that is available and fp32 when it
/// is not.
pub enum Slab {
    F32(GTensor<f32>),
    Bf16(GTensor<u16>),
}

impl Slab {
    /// An uninitialized slab of the given shape, bf16 where available.
    pub fn new(gpu: &Gpu, dims: &[usize]) -> Self {
        if active(gpu) {
            Slab::Bf16(GTensor::uninit(gpu, dims))
        } else {
            Slab::F32(GTensor::uninit(gpu, dims))
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
    pub fn store(&mut self, gpu: &Gpu, src: &GTensor<f32>) {
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
    pub fn get<'a>(&'a self, gpu: &Gpu, scratch: &'a mut GTensor<f32>) -> &'a GTensor<f32> {
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
        let d = GTensor::from_host(&gpu, &t);
        let mut b = GTensor::uninit(&gpu, &[64, 64]);
        b.store(&gpu, &d);
        let mut back = GTensor::uninit(&gpu, &[64, 64]);
        b.load(&gpu, &mut back);
        let got = back.to_host(&gpu);

        let mut worst = 0.0;
        let mut signed = 0.0;
        for (a, g) in t.data.iter().zip(got.data.iter()) {
            let rel = (a - g).abs() / a.abs().max(1e-6);
            worst = f32::max(worst, rel);
            signed += a - g;
        }
        assert!(
            worst < 3.91e-3,
            "bf16 round-trip relative error {worst} too large"
        );
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
            let d = GTensor::from_host(&gpu, &t);
            let mut b = GTensor::uninit(&gpu, &[n]);
            b.store(&gpu, &d);
            let mut back = GTensor::uninit(&gpu, &[n]);
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
        let f32_bytes = Slab::F32(GTensor::uninit(&gpu, &dims)).bytes();
        let bf_bytes = Slab::Bf16(GTensor::uninit(&gpu, &dims)).bytes();
        assert_eq!(f32_bytes, 2 * bf_bytes);
    }
}
