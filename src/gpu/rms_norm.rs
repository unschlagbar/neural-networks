//! Device-resident RMSNorm, the GPU counterpart of
//! [`nn2::rms_norm::RmsNorm`](crate::nn2::rms_norm::RmsNorm).
//!
//! Pure normalization (`y = γ ⊙ x / rms(x)`, row-wise) with a learned scale that
//! trains on undecayed AdamW (norm scales are never weight-decayed). Wraps the
//! grouped `rms_norm_forward`/`backward` ops with the single-group (plain) config
//! `group == size`. Scale, grad, moments and the saved `inv_rms` all live on the
//! device; `x̂` is never stored — backward rebuilds it from the forward output.

use super::arena::{self, ParamKind, ParamSlot, TrainingCache};
use super::ops::{self, GpuRmsForward};
use super::{GTensor, Gpu};
use crate::nn2::optim::AdamCfg;
use crate::tensor::Tensor;

/// Matches `nn2::rms_norm::EPS` so the two systems normalize identically.
const EPS: f32 = 1e-6;

pub struct RmsNorm {
    pub gamma: GTensor<f32>,  // [F]
    pub dgamma: GTensor<f32>, // [F]
    m: GTensor<f32>,
    v: GTensor<f32>,
    size: usize,
    /// Normalization group width: `== size` for plain RMSNorm, `== dhv` for the
    /// head-wise variant (`F/group` independent groups per row, one γ slice each).
    group: usize,
    /// Saved `inv_rms`, one per normalization group, reused across calls.
    ///
    /// The only thing the forward keeps. `x̂` is NOT stored: backward is handed the
    /// forward output and rebuilds `x̂ = y/γ`, which is what Apex's `memory_efficient`
    /// path does and Liger's equivalent from the input side. Storing it costs `[N, F]`
    /// against this `[N]` — at the backbone's shape three norms per block over a chunked
    /// sweep came to 1.15 GB of device memory that nothing else could use.
    fwd: Option<GpuRmsForward>,
    /// Earlier chunks' `inv_rms`, oldest first. The single `fwd` slot holds one chunk's,
    /// so without this chunk c+1's forward overwrites what chunk c's backward reads.
    /// Empty unless [`set_carry`](Self::set_carry) is on.
    chunk_saved: Vec<GpuRmsForward>,
    /// Whether this norm is inside a chunked sweep. See [`set_carry`](Self::set_carry).
    carry: bool,
}

impl RmsNorm {
    /// Build from a host scale `[F]` (uploaded). Plain (single-group) norm.
    pub fn from_parts(gpu: &Gpu, gamma: &Tensor) -> Self {
        let size = gamma.len();
        Self::from_parts_grouped(gpu, gamma, size)
    }

    /// Head-wise variant: `group` is the per-head width (`dhv`); `F` must be a
    /// multiple of it. Matches `nn2` head-wise RMSNorm (γ is `[F]`, group `grp`
    /// uses `γ[grp*group ..]`).
    pub fn from_parts_grouped(gpu: &Gpu, gamma: &Tensor, group: usize) -> Self {
        let size = gamma.len();
        assert!(
            size.is_multiple_of(group),
            "RmsNorm: size {size} not divisible by group {group}"
        );
        Self {
            gamma: GTensor::from_host(gpu, gamma),
            dgamma: GTensor::zeros(gpu, &[size]),
            m: GTensor::zeros(gpu, &[size]),
            v: GTensor::zeros(gpu, &[size]),
            size,
            group,
            fwd: None,
            chunk_saved: Vec::new(),
            carry: false,
        }
    }

    /// Fresh RMSNorm with `γ = 1` (matches `nn2::RmsNorm::new`).
    pub fn new(gpu: &Gpu, size: usize) -> Self {
        Self::from_parts(gpu, &Tensor::new(&[size], vec![1.0; size]))
    }

    /// `y = γ ⊙ (x / rms(x))`, row-wise, into the caller's `out` `[B, F]`. Saves
    /// `inv_rms` for backward, and nothing else.
    ///
    /// `out` may alias `x` (the kernel reads each row before writing it), which
    /// is what lets a caller normalize a buffer in place.
    pub fn forward(&mut self, gpu: &Gpu, x: &GTensor<f32>, out: &mut GTensor<f32>) {
        self.fit_saved(gpu, x);
        let Self { gamma, group, fwd, .. } = self;
        let saved = fwd.as_mut().expect("fit_saved filled it");
        ops::rms_norm_forward_into(gpu, x, gamma, *group, EPS, out, saved);
    }

    /// [`forward`](Self::forward) writing a slab, for a caller whose only readers of
    /// `y` take it narrow — the block's two pre-norms, whose output goes straight into
    /// a GEMM. `x` stays fp32: it is the residual stream.
    ///
    /// Pair it with [`backward_slab`](Self::backward_slab); mixing the two widths
    /// across a forward/backward pair reads the wrong bits.
    pub fn forward_slab(&mut self, gpu: &Gpu, x: &GTensor<f32>, out: &mut ops::SlabBuf) {
        self.fit_saved(gpu, x);
        let Self { gamma, group, fwd, .. } = self;
        let saved = fwd.as_mut().expect("fit_saved filled it");
        ops::rms_norm_forward_into_slab(gpu, x, gamma, *group, EPS, out, saved);
    }

    /// Present `inv_rms` at the shape this call needs, setting the previous chunk's
    /// aside first when the sweep is chunked.
    fn fit_saved(&mut self, gpu: &Gpu, x: &GTensor<f32>) {
        // Position-wise: any rank is accepted and folded to `[N, F]` over the last
        // axis, so a caller holding `[B, T, H]` need not reshape.
        let (b, f) = x.as_2d();
        assert_eq!(f, self.size, "RmsNorm::forward — width mismatch");
        let total_groups = b * (f / self.group);
        // Chunked sweep: the previous chunk's `inv_rms` is still owed a backward, so set
        // it aside rather than letting the refit below reuse its buffer.
        if self.carry {
            if let Some(prev) = self.fwd.take() {
                self.chunk_saved.push(prev);
            }
        }
        match &self.fwd {
            Some(s) if s.inv_rms.len() == total_groups => {}
            _ => {
                self.fwd = Some(GpuRmsForward {
                    inv_rms: gpu
                        .stream
                        .alloc_zeros::<f32>(total_groups)
                        .expect("alloc inv_rms"),
                })
            }
        }
    }

    /// Forward into a freshly allocated `[B, F]` — the by-value companion to
    /// [`forward`](Self::forward), for call sites that still compose by value.
    pub fn forward_alloc(&mut self, gpu: &Gpu, x: &GTensor<f32>) -> GTensor<f32> {
        let mut out = GTensor::uninit(gpu, &[x.rows(), x.cols()]);
        self.forward(gpu, x, &mut out);
        out
    }

    /// Backward into a freshly allocated `dX` `[B, F]`.
    pub fn backward_alloc(&mut self, gpu: &Gpu, dy: &GTensor<f32>, y: &GTensor<f32>, cache: &TrainingCache) -> GTensor<f32> {
        let mut dx = GTensor::uninit(gpu, &[dy.rows(), dy.cols()]);
        self.backward(gpu, dy, y, &mut dx, cache);
        dx
    }

    /// Given `dY` `[B, F]`, accumulate `dγ` and write `dX` `[B, F]` into `dx`.
    ///
    /// `y` is this norm's own forward OUTPUT, which backward divides by γ to recover
    /// `x̂`. Keeping the caller's `y` alive is the whole reason the forward can get away
    /// with saving only `inv_rms`; every caller here holds it for another reason anyway.
    pub fn backward(
        &mut self,
        gpu: &Gpu,
        dy: &GTensor<f32>,
        y: &GTensor<f32>,
        dx: &mut GTensor<f32>,
            cache: &TrainingCache,
) {
        self.backward_wos(gpu, dy, ops::WideOrSlab::F32(y), dx, cache);
    }

    /// [`backward`](Self::backward) where `y` is the slab this norm's
    /// [`forward_slab`](Self::forward_slab) wrote. Both readers of `y` — the kernel
    /// and the `dγ` reduction — take it at that width.
    pub fn backward_slab(
        &mut self,
        gpu: &Gpu,
        dy: &GTensor<f32>,
        y: &ops::SlabBuf,
        dx: &mut GTensor<f32>,
            cache: &TrainingCache,
) {
        self.backward_wos(gpu, dy, ops::WideOrSlab::Slab(y), dx, cache);
    }

    fn backward_wos(
        &mut self,
        gpu: &Gpu,
        dy: &GTensor<f32>,
        y: ops::WideOrSlab<'_>,
        dx: &mut GTensor<f32>,
            cache: &TrainingCache,
) {
        let (_, f) = dy.as_2d();
        assert_eq!(f, self.size, "RmsNorm::backward — width mismatch");
        assert_eq!(y.as_2d(), dy.as_2d(), "RmsNorm::backward — y shape");
        let fwd = self.fwd.as_ref().expect("RmsNorm::backward before forward");
        ops::rms_norm_backward_into(
            gpu,
            dy,
            fwd,
            y,
            &self.gamma,
            &mut self.dgamma,
            self.group,
            dx,
            &cache.temps,
        );
        // Chunks unwind right to left, so hand the slot to the chunk on the left.
        if let Some(prev) = self.chunk_saved.pop() {
            self.fwd = Some(prev);
        }
    }

    /// Every learnable tensor, in a fixed order (used by checkpoint save/load).
    /// The norm scale with its gradient and AdamW moments. Never decayed.
    pub fn param_slots(&mut self) -> Vec<ParamSlot<'_>> {
        vec![ParamSlot::new(
            &mut self.gamma,
            &mut self.dgamma,
            &mut self.m,
            &mut self.v,
            ParamKind::NoDecay,
        )]
    }

    pub fn params_mut(&mut self) -> Vec<&mut GTensor<f32>> {
        vec![&mut self.gamma]
    }

    pub fn zero_grad(&mut self, gpu: &Gpu) {
        self.dgamma.zero_(gpu);
    }

    /// Device bytes held, split `(params, activations)`. Diagnostic — see
    /// [`Hierarchical::retained_report`](super::hierarchical::Hierarchical::retained_report).
    ///
    /// The params are four `[F]` vectors — negligible, and so is the activation side:
    /// `inv_rms` is one float per normalization group, not per element.
    pub fn retained_bytes(&self) -> (usize, usize) {
        let params = [&self.gamma, &self.dgamma, &self.m, &self.v]
            .iter()
            .map(|t| t.capacity() * 4)
            .sum();
        let act = self
            .fwd
            .as_ref()
            .map_or(0, |s| s.inv_rms.len() * 4);
        (params, act)
    }

    /// Keep one saved `inv_rms` per chunk, for a sweep whose chunks all forward
    /// before any of them unwinds. Off means the single slot is reused per call, which
    /// is what every unchunked caller wants.
    pub fn set_carry(&mut self, carry: bool) {
        self.carry = carry;
        if !carry {
            self.chunk_saved.clear();
        }
    }

    /// Release the saved `x̂` / `inv_rms`. The next forward reallocates them.
    pub fn drop_saved_act(&mut self) {
        self.fwd = None;
        self.chunk_saved.clear();
    }

    /// AdamW step (norm scale is never decayed). Clears the grad.
    pub fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        arena::step_slots(gpu, &mut self.param_slots(), cfg);
    }
}

#[cfg(test)]
mod tests {

    /// One temp cache per test, sized past every shape this module presents.
    fn test_cache(gpu: &Gpu) -> TrainingCache {
        TrainingCache::new(gpu, 1 << 20, 1 << 16, 1 << 20)
    }
    use super::*;
    use crate::gpu::GTensor;

    /// The slab path must agree with the fp32 one to bf16's precision, and only to
    /// that: `y` is the sole tensor whose width changes, so the gap is one rounding
    /// of the forward output propagated through both readers of it.
    ///
    /// Worth pinning separately from the CPU parity tests because those run the fp32
    /// entry points — a `_slab` kernel could be wrong in every element and they would
    /// not notice.
    #[test]
    fn slab_path_matches_fp32_within_bf16() {
        let Some(gpu) = crate::gpu::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        // A block-norm shape (group == width) and a head-norm one (group << width),
        // which take different launch geometries.
        for (rows, size, group) in [(64usize, 256usize, 256usize), (128, 128, 16)] {
            let g = Tensor::random(&[size], 0.4);
            let x = GTensor::from_host(&gpu, &Tensor::random(&[rows, size], 0.7));
            let dy = GTensor::from_host(&gpu, &Tensor::random(&[rows, size], 0.9));

            let mut wide = RmsNorm::from_parts_grouped(&gpu, &g, group);
            let mut y_w = GTensor::uninit(&gpu, &[rows, size]);
            wide.forward(&gpu, &x, &mut y_w);
            let mut dx_w = GTensor::uninit(&gpu, &[rows, size]);
            wide.backward(&gpu, &dy, &y_w, &mut dx_w, &tc);

            let mut narrow = RmsNorm::from_parts_grouped(&gpu, &g, group);
            let mut y_n = ops::SlabBuf::new(&gpu, &[rows, size]);
            narrow.forward_slab(&gpu, &x, &mut y_n);
            let mut dx_n = GTensor::uninit(&gpu, &[rows, size]);
            narrow.backward_slab(&gpu, &dy, &y_n, &mut dx_n, &tc);

            // bf16 keeps 8 mantissa bits, so a single rounding is ~4e-3 relative. The
            // fp32 build makes the two paths the same kernel, hence the tighter bound.
            let tol = if gpu.kernels.slab_bf16 { 1e-2 } else { 1e-6 };
            for (name, a, b) in [
                ("y", y_w.to_host(&gpu).data, {
                    let mut s = GTensor::uninit(&gpu, &[rows, size]);
                    y_n.as_f32(&gpu, &mut s).to_host(&gpu).data
                }),
                ("dx", dx_w.to_host(&gpu).data, dx_n.to_host(&gpu).data),
                (
                    "dgamma",
                    wide.dgamma.to_host(&gpu).data,
                    narrow.dgamma.to_host(&gpu).data,
                ),
            ] {
                let scale = a.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
                for (i, (p, q)) in a.iter().zip(&b).enumerate() {
                    assert!(
                        (p - q).abs() <= tol * scale,
                        "{name}[{i}] at ({rows},{size},{group}): {p} vs {q}"
                    );
                }
            }
        }
    }
}
