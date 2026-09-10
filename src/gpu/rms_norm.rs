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
use super::{Frame, GTensor, Gpu};
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
    /// Saved `inv_rms` of a standalone [`forward`](Self::forward), reused across calls.
    ///
    /// The only thing the forward keeps. `x̂` is NOT stored: backward is handed the
    /// forward output and rebuilds `x̂ = y/γ`, which is what Apex's `memory_efficient`
    /// path does and Liger's equivalent from the input side. Storing it costs `[N, F]`
    /// against this `[N]`. A norm inside a block saves into the block's frame instead —
    /// see [`carve`](Self::carve).
    fwd: Option<GpuRmsForward>,
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
            m: arena::unbacked(gpu),
            v: arena::unbacked(gpu),
            size,
            group,
            fwd: None,
        }
    }

    /// Fresh RMSNorm with `γ = 1` (matches `nn2::RmsNorm::new`).
    pub fn new(gpu: &Gpu, size: usize) -> Self {
        Self::from_parts(gpu, &Tensor::new(&[size], vec![1.0; size]))
    }

    /// This norm's saved `inv_rms` for `rows` rows, carved from `f`.
    pub fn carve(&self, gpu: &Gpu, f: &mut Frame, rows: usize) -> GpuRmsForward {
        GpuRmsForward {
            inv_rms: f.f32(gpu, &[rows * (self.size / self.group)]),
        }
    }

    /// `y = γ ⊙ (x / rms(x))`, row-wise, into the caller's `out` `[B, F]`, saving
    /// `inv_rms` into `saved` and nothing else.
    ///
    /// `out` may alias `x` (the kernel reads each row before writing it), which
    /// is what lets a caller normalize a buffer in place.
    pub fn forward_saved(
        &self,
        gpu: &Gpu,
        x: &GTensor<f32>,
        out: &mut GTensor<f32>,
        saved: &mut GpuRmsForward,
    ) {
        assert_eq!(x.as_2d().1, self.size, "RmsNorm::forward — width mismatch");
        ops::rms_norm_forward_into(gpu, x, &self.gamma, self.group, EPS, out, saved);
    }

    /// [`forward_saved`](Self::forward_saved) writing a slab, for a caller whose only
    /// readers of `y` take it narrow — the block's two pre-norms, whose output goes
    /// straight into a GEMM. `x` stays fp32: it is the residual stream.
    ///
    /// Pair it with [`backward_slab_saved`](Self::backward_slab_saved); mixing the two
    /// widths across a forward/backward pair reads the wrong bits.
    ///
    /// With `add = Some((x2, z))` the norm's input is the residual sum `z = x + x2`,
    /// which is written to `z` in the same pass.
    pub fn forward_slab_saved(
        &self,
        gpu: &Gpu,
        x: &GTensor<f32>,
        add: Option<(&GTensor<f32>, &mut GTensor<f32>)>,
        out: &mut ops::SlabBuf,
        saved: &mut GpuRmsForward,
    ) {
        assert_eq!(x.as_2d().1, self.size, "RmsNorm::forward — width mismatch");
        ops::rms_norm_forward_into_slab(gpu, x, add, &self.gamma, self.group, EPS, out, saved);
    }

    /// [`forward_saved`](Self::forward_saved) into this norm's own slot, for a norm
    /// used on its own.
    pub fn forward(&mut self, gpu: &Gpu, x: &GTensor<f32>, out: &mut GTensor<f32>) {
        let mut saved = self.take_slot(gpu, x);
        self.forward_saved(gpu, x, out, &mut saved);
        self.fwd = Some(saved);
    }

    /// [`forward_slab_saved`](Self::forward_slab_saved) into this norm's own slot.
    pub fn forward_slab(&mut self, gpu: &Gpu, x: &GTensor<f32>, out: &mut ops::SlabBuf) {
        let mut saved = self.take_slot(gpu, x);
        self.forward_slab_saved(gpu, x, None, out, &mut saved);
        self.fwd = Some(saved);
    }

    /// The own slot at `x`'s row count, reusing its buffer when it already fits.
    fn take_slot(&mut self, gpu: &Gpu, x: &GTensor<f32>) -> GpuRmsForward {
        let groups = x.as_2d().0 * (self.size / self.group);
        match self.fwd.take() {
            Some(s) if s.inv_rms.len() == groups => s,
            _ => GpuRmsForward {
                inv_rms: GTensor::uninit(gpu, &[groups]),
            },
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
    pub fn backward_alloc(
        &mut self,
        gpu: &Gpu,
        dy: &GTensor<f32>,
        y: &GTensor<f32>,
        cache: &TrainingCache,
    ) -> GTensor<f32> {
        let mut dx = GTensor::uninit(gpu, &[dy.rows(), dy.cols()]);
        self.backward(gpu, dy, y, &mut dx, cache);
        dx
    }

    /// Given `dY` `[B, F]`, accumulate `dγ` and write `dX` `[B, F]` into `dx`, reading
    /// the `inv_rms` a standalone [`forward`](Self::forward) saved.
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
        let saved = self.fwd.take().expect("RmsNorm::backward before forward");
        self.backward_saved(gpu, dy, y, &saved, None, dx, cache);
        self.fwd = Some(saved);
    }

    /// [`backward`](Self::backward) where `y` is the slab
    /// [`forward_slab`](Self::forward_slab) wrote.
    pub fn backward_slab(
        &mut self,
        gpu: &Gpu,
        dy: &GTensor<f32>,
        y: &ops::SlabBuf,
        dx: &mut GTensor<f32>,
        cache: &TrainingCache,
    ) {
        let saved = self.fwd.take().expect("RmsNorm::backward before forward");
        self.backward_slab_saved(gpu, dy, y, &saved, None, dx, cache);
        self.fwd = Some(saved);
    }

    /// [`backward`](Self::backward) reading the `inv_rms` a
    /// [`forward_saved`](Self::forward_saved) wrote. `resid`, when given, is added to
    /// `dx` — the gradient of a residual branch that bypasses the norm.
    pub fn backward_saved(
        &mut self,
        gpu: &Gpu,
        dy: &GTensor<f32>,
        y: &GTensor<f32>,
        saved: &GpuRmsForward,
        resid: Option<&GTensor<f32>>,
        dx: &mut GTensor<f32>,
        cache: &TrainingCache,
    ) {
        self.backward_wos(gpu, dy, ops::WideOrSlab::F32(y), saved, resid, dx, cache);
    }

    /// [`backward_saved`](Self::backward_saved) where `y` is the slab
    /// [`forward_slab_saved`](Self::forward_slab_saved) wrote. Both readers of `y` — the
    /// kernel and the `dγ` reduction — take it at that width.
    pub fn backward_slab_saved(
        &mut self,
        gpu: &Gpu,
        dy: &GTensor<f32>,
        y: &ops::SlabBuf,
        saved: &GpuRmsForward,
        resid: Option<&GTensor<f32>>,
        dx: &mut GTensor<f32>,
        cache: &TrainingCache,
    ) {
        self.backward_wos(gpu, dy, ops::WideOrSlab::Slab(y), saved, resid, dx, cache);
    }

    #[allow(clippy::too_many_arguments)]
    fn backward_wos(
        &mut self,
        gpu: &Gpu,
        dy: &GTensor<f32>,
        y: ops::WideOrSlab<'_>,
        saved: &GpuRmsForward,
        resid: Option<&GTensor<f32>>,
        dx: &mut GTensor<f32>,
        cache: &TrainingCache,
    ) {
        let (_, f) = dy.as_2d();
        assert_eq!(f, self.size, "RmsNorm::backward — width mismatch");
        assert_eq!(y.as_2d(), dy.as_2d(), "RmsNorm::backward — y shape");
        ops::rms_norm_backward_into(
            gpu,
            dy,
            saved,
            y,
            &self.gamma,
            &mut self.dgamma,
            self.group,
            resid,
            dx,
            &cache.temps,
        );
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
        let act = self.fwd.as_ref().map_or(0, |s| s.inv_rms.len() * 4);
        (params, act)
    }

    /// Release the own slot's `inv_rms`. The next forward reallocates it.
    pub fn drop_saved_act(&mut self) {
        self.fwd = None;
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
            let mut saved = ops::GpuRmsForward {
                inv_rms: GTensor::uninit(&gpu, &[rows * (size / group)]),
            };
            narrow.forward_slab_saved(&gpu, &x, None, &mut y_n, &mut saved);
            let mut dx_n = GTensor::uninit(&gpu, &[rows, size]);
            narrow.backward_slab_saved(&gpu, &dy, &y_n, &saved, None, &mut dx_n, &tc);

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
