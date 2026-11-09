//! Device-resident `Linear` (`Y = X · W + b`), the GPU counterpart of
//! [`cpu::linear::Linear`](crate::cpu::linear::Linear).

use super::arena::{self, ParamKind, ParamSlot, TrainingCache};
use super::{GTensor, Gpu, ops};
use crate::cpu::optim::AdamCfg;
use crate::tensor::Tensor;

pub struct Linear {
    pub w: GTensor<f32>, // [in, out]
    pub b: GTensor<f32>, // [out]
    pub dw: GTensor<f32>,
    pub db: GTensor<f32>,
    /// Saved forward input `[B, in]` (device copy) for the weight gradient.
    x: GTensor<f32>,
    /// AdamW moments for `w` and `b`.
    mw: GTensor<f32>,
    vw: GTensor<f32>,
    mb: GTensor<f32>,
    vb: GTensor<f32>,
    input: usize,
    output: usize,
    /// bf16 staging for this layer's three GEMMs (see `ops::GemmBf16`). Holds the
    /// narrowed operand copies across calls so the cast never allocates.
    gemm: ops::GemmBf16,
    /// Whether this layer's GEMMs go through the bf16 path. Pinned at construction
    /// so forward and backward cannot disagree, and `false` for a layer that must
    /// stay exact (see `set_fp32`).
    bf16: bool,
    /// Whether `w` is weight-decayed. True by default — an interior projection —
    /// and cleared for an embedding-like table or a logit head (`set_no_decay`).
    decay: bool,
    /// Whether `b` is left at its init and never stepped (`freeze_bias`).
    bias_frozen: bool,
}

impl Linear {
    /// Build from host weight `[in, out]` and bias `[out]` tensors (uploads them).
    pub fn from_parts(gpu: &Gpu, w: &Tensor, b: &Tensor) -> Self {
        let (input, output) = (w.rows(), w.cols());
        assert_eq!(b.len(), output, "Linear::from_parts — bias len != out");
        Self {
            w: GTensor::from_host(gpu, w),
            b: GTensor::from_host(gpu, b),
            dw: GTensor::zeros(gpu, &[input, output]),
            db: GTensor::zeros(gpu, &[output]),
            x: GTensor::zeros(gpu, &[0, input]),
            mw: arena::unbacked(gpu),
            vw: arena::unbacked(gpu),
            mb: arena::unbacked(gpu),
            vb: arena::unbacked(gpu),
            input,
            output,
            gemm: ops::GemmBf16::new(),
            bf16: ops::gemm_bf16_enabled(gpu),
            decay: true,
            bias_frozen: false,
        }
    }

    /// Device bytes held, split `(params, activations)`. Diagnostic — see
    /// [`Hierarchical::retained_report`](super::hierarchical::Hierarchical::retained_report).
    pub fn retained_bytes(&self) -> (usize, usize) {
        let params = [
            &self.w, &self.b, &self.dw, &self.db, &self.mw, &self.vw, &self.mb, &self.vb,
        ]
        .iter()
        .map(|t| t.capacity() * 4)
        .sum();
        (params, self.x.capacity() * 4 + self.gemm.retained_bytes())
    }

    /// Release the saved forward input and the bf16 staging of the *activations*.
    pub fn drop_saved_act(&mut self, gpu: &Gpu) {
        self.x = GTensor::zeros(gpu, &[0, self.input]);
        self.gemm.clear_operands();
    }

    /// Force this layer's GEMMs back to fp32.
    pub fn set_fp32(&mut self) {
        self.bf16 = false;
    }

    /// Take `w` off the weight-decay path, for an embedding-like table or a logit
    /// head (the project's optimizer convention — see `gpu::arena::ParamKind`).
    pub fn set_no_decay(&mut self) {
        self.decay = false;
    }

    /// Never step `b`, leaving it at its zero init: a bias-free head.
    pub fn freeze_bias(&mut self) {
        self.bias_frozen = true;
    }

    #[inline]
    pub fn input_size(&self) -> usize {
        self.input
    }
    #[inline]
    pub fn output_size(&self) -> usize {
        self.output
    }
    /// Whether this layer's GEMMs run in bf16 (see [`Self::pin_f32`]).
    #[inline]
    pub fn is_bf16(&self) -> bool {
        self.bf16
    }

    /// `Y = X · W + b` into the caller's `y` `[B, out]`; `x` is `[B, in]`. Saves
    /// `x` for backward.
    pub fn forward(&mut self, gpu: &Gpu, x: &GTensor<f32>, y: &mut GTensor<f32>) {
        let b = x.rows();
        assert_eq!(
            x.cols(),
            self.input,
            "Linear::forward — input width mismatch"
        );
        assert_eq!(y.dims(), [b, self.output], "Linear::forward — output shape");
        if self.x.len() == b * self.input {
            self.x.reshape_to(&[b, self.input]);
            self.x.copy_from(gpu, x);
        } else {
            self.x = x.dup(gpu);
        }
        self.forward_raw(gpu, x, y);
    }

    /// The input [`forward`](Self::forward) saved, `[B, in]`.
    pub fn saved_x(&self) -> &GTensor<f32> {
        &self.x
    }

    /// `Y = X · W + b` for a caller-owned input, saving **nothing**.
    pub fn forward_shared(&mut self, gpu: &Gpu, x: &GTensor<f32>, y: &mut GTensor<f32>) {
        assert_eq!(
            x.cols(),
            self.input,
            "Linear::forward_shared — input width mismatch"
        );
        assert_eq!(
            y.dims(),
            [x.rows(), self.output],
            "Linear::forward_shared — output shape"
        );
        if !self.x.is_empty() {
            self.x = GTensor::uninit(gpu, &[0, self.input]);
        }
        self.forward_raw(gpu, x, y);
    }

    /// [`forward_shared`](Self::forward_shared) writing `y = resid + b + x·W`, for a
    /// projection whose output feeds a residual add. The add rides along in the bias
    /// seed the GEMM already needs, so it costs no kernel of its own — see
    /// [`ops::broadcast_row_resid`]. `resid` and `y` may not alias.
    pub fn forward_shared_resid(
        &mut self,
        gpu: &Gpu,
        x: &GTensor<f32>,
        resid: &GTensor<f32>,
        y: &mut GTensor<f32>,
    ) {
        assert_eq!(
            x.cols(),
            self.input,
            "Linear::forward_shared_resid — input width mismatch"
        );
        assert_eq!(
            y.dims(),
            [x.rows(), self.output],
            "Linear::forward_shared_resid — output shape"
        );
        if !self.x.is_empty() {
            self.x = GTensor::uninit(gpu, &[0, self.input]);
        }
        ops::broadcast_row_resid(gpu, y, resid, &self.b);
        if self.bf16 {
            self.gemm.run_wb(gpu, ops::MmForm::Nn, x, &self.w, y, 1.0);
        } else {
            ops::matmul_nn_into(gpu, x, &self.w, y, 1.0);
        }
    }

    /// [`forward_shared_resid`](Self::forward_shared_resid) with a slab `x` — the
    /// residual form of [`forward_slab_lhs`](Self::forward_slab_lhs), for the block's
    /// down-projection, whose input is written narrow by the SwiGLU.
    pub fn forward_slab_lhs_resid(
        &mut self,
        gpu: &Gpu,
        x: &ops::SlabBuf,
        resid: &GTensor<f32>,
        y: &mut GTensor<f32>,
    ) {
        match x {
            ops::SlabBuf::F32(t) => self.forward_shared_resid(gpu, t, resid, y),
            ops::SlabBuf::Bf16(b) => {
                assert!(
                    self.bf16,
                    "Linear::forward_slab_lhs_resid — layer is fp32-pinned"
                );
                assert_eq!(
                    y.dims(),
                    [b.dims()[0], self.output],
                    "Linear::forward_slab_lhs_resid — output shape"
                );
                if !self.x.is_empty() {
                    self.x = GTensor::uninit(gpu, &[0, self.input]);
                }
                ops::broadcast_row_resid(gpu, y, resid, &self.b);
                self.gemm
                    .run_staged_lhs(gpu, ops::MmForm::Nn, b, &self.w, y, 1.0);
            }
        }
    }

    /// The GEMM half of forward, without the input-saving policy. Shared by
    /// [`forward`](Self::forward) and [`forward_shared`](Self::forward_shared) so the
    /// two cannot drift apart.
    fn forward_raw(&mut self, gpu: &Gpu, x: &GTensor<f32>, y: &mut GTensor<f32>) {
        ops::broadcast_row(gpu, y, &self.b);
        if self.bf16 {
            self.gemm.run_wb(gpu, ops::MmForm::Nn, x, &self.w, y, 1.0);
        } else {
            ops::matmul_nn_into(gpu, x, &self.w, y, 1.0);
        }
    }

    /// [`forward_shared`](Self::forward_shared) where `x` has already been narrowed
    /// into bf16 by the caller — saves the per-layer cast when several
    /// layers project the same input. `x` is the fp32 original, used only when this
    /// layer is pinned to fp32.
    pub fn forward_staged(
        &mut self,
        gpu: &Gpu,
        x: &GTensor<f32>,
        x_b: &super::GTensor<u16>,
        out: &mut GTensor<f32>,
    ) {
        assert_eq!(
            x.cols(),
            self.input,
            "Linear::forward_staged — input width mismatch"
        );
        assert_eq!(
            out.dims(),
            [x.rows(), self.output],
            "Linear::forward_staged — output shape"
        );
        if !self.x.is_empty() {
            self.x = GTensor::uninit(gpu, &[0, self.input]);
        }
        ops::broadcast_row(gpu, out, &self.b);
        if self.bf16 {
            self.gemm
                .run_staged_lhs(gpu, ops::MmForm::Nn, x_b, &self.w, out, 1.0);
        } else {
            ops::matmul_nn_into(gpu, x, &self.w, out, 1.0);
        }
    }

    /// [`forward_shared`](Self::forward_shared) where `x` is a slab — already at the
    /// width the GEMM wants, so nothing is narrowed here at all.
    pub fn forward_slab_lhs(&mut self, gpu: &Gpu, x: &ops::SlabBuf, out: &mut GTensor<f32>) {
        match x {
            ops::SlabBuf::F32(t) => self.forward_shared(gpu, t, out),
            ops::SlabBuf::Bf16(b) => {
                assert!(self.bf16, "Linear::forward_slab_lhs — layer is fp32-pinned");
                assert_eq!(
                    out.dims(),
                    [b.dims()[0], self.output],
                    "Linear::forward_slab_lhs — output shape"
                );
                if !self.x.is_empty() {
                    self.x = GTensor::uninit(gpu, &[0, self.input]);
                }
                ops::broadcast_row(gpu, out, &self.b);
                self.gemm
                    .run_staged_lhs(gpu, ops::MmForm::Nn, b, &self.w, out, 1.0);
            }
        }
    }

    /// [`backward_with_x`](Self::backward_with_x) where the saved input is a slab.
    /// The backward twin of [`forward_slab_lhs`](Self::forward_slab_lhs).
    pub fn backward_slab_x(
        &mut self,
        gpu: &Gpu,
        x: &ops::SlabBuf,
        dy: &GTensor<f32>,
        dx: &mut GTensor<f32>,
        dx_beta: f32,
        cache: &TrainingCache,
    ) {
        match x {
            ops::SlabBuf::F32(t) => self.backward_with_x(gpu, t, dy, dx, dx_beta, cache),
            ops::SlabBuf::Bf16(b) => {
                assert!(self.bf16, "Linear::backward_slab_x — layer is fp32-pinned");
                self.gemm
                    .run_backward_staged_x(gpu, b, dy, &self.w, &mut self.dw, dx, dx_beta);
                ops::add_col_sum(gpu, &mut self.db, dy, &cache.temps);
            }
        }
    }

    /// [`forward_staged`](Self::forward_staged) writing a **bf16** output with the
    /// bias fused into the GEMM epilogue, for a consumer that reads bf16 anyway.
    pub fn forward_staged_bf16(
        &mut self,
        gpu: &Gpu,
        x_b: &super::GTensor<u16>,
        out: &mut super::GTensor<u16>,
    ) {
        assert!(
            self.bf16,
            "Linear::forward_staged_bf16 — layer is fp32-pinned"
        );
        assert_eq!(
            out.dims(),
            [x_b.dims()[0], self.output],
            "Linear::forward_staged_bf16 — output shape"
        );
        if !self.x.is_empty() {
            self.x = GTensor::uninit(gpu, &[0, self.input]);
        }
        self.gemm
            .run_staged_lhs_bias(gpu, x_b, &self.w, &self.b, out);
    }

    /// [`forward_staged`](Self::forward_staged) into a slab, at whatever width the
    /// kernels were built for: a bf16 slab is written straight out of the GEMM
    /// epilogue, an fp32 one goes through the ordinary staged path.
    pub fn forward_staged_slab(
        &mut self,
        gpu: &Gpu,
        x: &GTensor<f32>,
        x_b: &super::GTensor<u16>,
        out: &mut ops::SlabBuf,
    ) {
        match out {
            ops::SlabBuf::Bf16(o) => self.forward_staged_bf16(gpu, x_b, o),
            ops::SlabBuf::F32(o) => self.forward_staged(gpu, x, x_b, o),
        }
    }

    /// [`backward_slab_x`](Self::backward_slab_x) where `dy` is a slab too. With both
    /// narrow neither GEMM casts, and the bias sum reads the same bf16 `dy` they do.
    pub fn backward_slab_xy(
        &mut self,
        gpu: &Gpu,
        x: &ops::SlabBuf,
        dy: &ops::SlabBuf,
        dx: &mut GTensor<f32>,
        dx_beta: f32,
        cache: &TrainingCache,
    ) {
        match (x, dy) {
            (ops::SlabBuf::Bf16(xb), ops::SlabBuf::Bf16(dyb)) => {
                assert!(self.bf16, "Linear::backward_slab_xy — layer is fp32-pinned");
                self.gemm
                    .run_backward_narrow(gpu, xb, dyb, &self.w, &mut self.dw, dx, dx_beta);
                ops::add_col_sum_slab(gpu, &mut self.db, dy, &cache.temps);
            }
            (_, ops::SlabBuf::F32(d)) => self.backward_slab_x(gpu, x, d, dx, dx_beta, cache),
            (ops::SlabBuf::F32(_), ops::SlabBuf::Bf16(_)) => {
                panic!("Linear::backward_slab_xy — bf16 dy against an fp32 input")
            }
        }
    }

    /// [`forward_slab_lhs`](Self::forward_slab_lhs) into a slab: a bf16 output takes
    /// the bias in the GEMM epilogue, an fp32 one the ordinary seeded path.
    pub fn forward_slab_slab(&mut self, gpu: &Gpu, x: &ops::SlabBuf, out: &mut ops::SlabBuf) {
        match (x, out) {
            (ops::SlabBuf::Bf16(xb), ops::SlabBuf::Bf16(o)) => self.forward_staged_bf16(gpu, xb, o),
            (_, ops::SlabBuf::F32(o)) => self.forward_slab_lhs(gpu, x, o),
            (ops::SlabBuf::F32(_), ops::SlabBuf::Bf16(_)) => {
                panic!("Linear::forward_slab_slab — fp32 input into a bf16 output")
            }
        }
    }

    /// [`backward_with_x`](Self::backward_with_x) where the saved input is already
    /// narrowed into bf16 by the caller.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_staged_x(
        &mut self,
        gpu: &Gpu,
        x: &GTensor<f32>,
        x_b: &super::GTensor<u16>,
        dy: &GTensor<f32>,
        dx: &mut GTensor<f32>,
        dx_beta: f32,
        cache: &TrainingCache,
    ) {
        assert_eq!(
            dy.cols(),
            self.output,
            "Linear::backward_staged_x — grad width mismatch"
        );
        assert_eq!(
            x.rows(),
            dy.rows(),
            "Linear::backward_staged_x — saved input / grad row mismatch"
        );
        if self.bf16 {
            self.gemm
                .run_backward_staged_x(gpu, x_b, dy, &self.w, &mut self.dw, dx, dx_beta);
            ops::add_col_sum(gpu, &mut self.db, dy, &cache.temps);
        } else {
            Self::grad_w(gpu, &mut self.gemm, self.bf16, x, dy, &mut self.dw);
            ops::add_col_sum(gpu, &mut self.db, dy, &cache.temps);
            self.grad_x(gpu, dy, dx, dx_beta);
        }
    }

    /// `Y = X · W + b` into a freshly allocated `[B, out]`.
    pub fn forward_alloc(&mut self, gpu: &Gpu, x: &GTensor<f32>) -> GTensor<f32> {
        let mut y = GTensor::uninit(gpu, &[x.rows(), self.output]);
        self.forward(gpu, x, &mut y);
        y
    }

    /// Backward into a freshly allocated `dX` `[B, in]` — the by-value companion
    /// to [`backward`](Self::backward).
    pub fn backward_alloc(&mut self, gpu: &Gpu, dy: &GTensor<f32>, cache: &TrainingCache) -> GTensor<f32> {
        let mut dx = GTensor::uninit(gpu, &[dy.rows(), self.input]);
        self.backward(gpu, dy, &mut dx, cache);
        dx
    }

    /// [`backward_with_x`](Self::backward_with_x) into a freshly allocated `dX`.
    pub fn backward_alloc_with_x(
        &mut self,
        gpu: &Gpu,
        x: &GTensor<f32>,
        dy: &GTensor<f32>,
            cache: &TrainingCache,
) -> GTensor<f32> {
        let mut dx = GTensor::uninit(gpu, &[dy.rows(), self.input]);
        self.backward_with_x(gpu, x, dy, &mut dx, 0.0, cache);
        dx
    }

    /// Given `dY` `[B, out]`, accumulate `dW`/`db` and write `dX = dY · Wᵀ` into
    /// `dx` `[B, in]`.
    pub fn backward(&mut self, gpu: &Gpu, dy: &GTensor<f32>, dx: &mut GTensor<f32>, cache: &TrainingCache) {
        assert_eq!(
            self.x.rows(),
            dy.rows(),
            "Linear::backward — batch mismatch"
        );
        Self::grad_w(gpu, &mut self.gemm, self.bf16, &self.x, dy, &mut self.dw);
        ops::add_col_sum(gpu, &mut self.db, dy, &cache.temps);
        self.grad_x(gpu, dy, dx, 0.0);
    }

    /// Backward for a caller-owned input: the companion to
    /// [`forward_shared`](Self::forward_shared).
    pub fn backward_with_x(
        &mut self,
        gpu: &Gpu,
        x: &GTensor<f32>,
        dy: &GTensor<f32>,
        dx: &mut GTensor<f32>,
        dx_beta: f32,
        cache: &TrainingCache,
    ) {
        assert_eq!(
            dy.cols(),
            self.output,
            "Linear::backward — grad width mismatch"
        );
        assert_eq!(
            x.cols(),
            self.input,
            "Linear::backward — saved input width mismatch"
        );
        assert_eq!(
            x.rows(),
            dy.rows(),
            "Linear::backward — saved input / grad row mismatch"
        );
        assert_eq!(
            dx.dims(),
            [dy.rows(), self.input],
            "Linear::backward — dx shape"
        );
        if self.bf16 {
            self.gemm
                .run_backward(gpu, x, dy, &self.w, &mut self.dw, dx, dx_beta);
            ops::add_col_sum(gpu, &mut self.db, dy, &cache.temps);
        } else {
            Self::grad_w(gpu, &mut self.gemm, self.bf16, x, dy, &mut self.dw);
            ops::add_col_sum(gpu, &mut self.db, dy, &cache.temps);
            self.grad_x(gpu, dy, dx, dx_beta);
        }
    }

    /// `dW += Xᵀ · dY` (accumulating, `beta = 1`).
    fn grad_w(
        gpu: &Gpu,
        gemm: &mut ops::GemmBf16,
        bf16: bool,
        x: &GTensor<f32>,
        dy: &GTensor<f32>,
        dw: &mut GTensor<f32>,
    ) {
        if bf16 {
            gemm.run(gpu, ops::MmForm::Tn, x, dy, dw, 1.0);
        } else {
            ops::matmul_tn_into(gpu, x, dy, dw, 1.0);
        }
    }

    /// `dX = dY · Wᵀ + beta·dX`. cuBLAS transposes `W(in×out)` internally — no host
    /// transpose.
    fn grad_x(&mut self, gpu: &Gpu, dy: &GTensor<f32>, dx: &mut GTensor<f32>, beta: f32) {
        if self.bf16 {
            self.gemm.run_wb(gpu, ops::MmForm::Nt, dy, &self.w, dx, beta);
        } else {
            ops::matmul_nt_into(gpu, dy, &self.w, dx, beta);
        }
    }

    /// Freshly-initialised layer, matching `cpu::Linear::new`'s init exactly
    /// (built on the host, then uploaded).
    pub fn new_rand(gpu: &Gpu, input: usize, output: usize) -> Self {
        let cpu = crate::cpu::Linear::new(input, output);
        Self::from_parts(gpu, &cpu.w, &cpu.b)
    }

    /// Every parameter with its gradient and AdamW moments.
    pub fn param_slots(&mut self) -> Vec<ParamSlot<'_>> {
        self.gemm.invalidate_w();
        let w_kind = if self.decay {
            ParamKind::Decay
        } else {
            ParamKind::NoDecay
        };
        let b_kind = if self.bias_frozen {
            ParamKind::Frozen
        } else {
            ParamKind::NoDecay
        };
        vec![
            ParamSlot::new(
                &mut self.w,
                &mut self.dw,
                &mut self.mw,
                &mut self.vw,
                w_kind,
            ),
            ParamSlot::new(
                &mut self.b,
                &mut self.db,
                &mut self.mb,
                &mut self.vb,
                b_kind,
            ),
        ]
    }

    /// Every learnable tensor, in a fixed order (used by checkpoint save/load).
    pub fn params_mut(&mut self) -> Vec<&mut GTensor<f32>> {
        self.param_slots().into_iter().map(|s| s.param).collect()
    }

    /// Gradient accumulators, matching `params_mut`'s order. Diagnostic.
    pub fn grads(&mut self) -> Vec<&GTensor<f32>> {
        self.param_slots().into_iter().map(|s| &*s.grad).collect()
    }

    pub fn zero_grad(&mut self, gpu: &Gpu) {
        // In-place memset — reuse the existing allocations across steps.
        self.dw.zero_(gpu);
        self.db.zero_(gpu);
    }

    /// AdamW over this layer's own parameters, then clear the accumulators. A model
    /// steps its whole `ParamArena` in one launch instead.
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
    use crate::cpu::linear::Linear as CpuLinear;
    use crate::cpu::optim::AdamCfg;

    fn assert_close(got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < tol, "index {i}: gpu {g} vs cpu {w}");
        }
    }

    /// Tolerance against the all-fp32 CPU reference, widened when this layer's
    /// GEMMs run with bf16 operands. `gemm_bf16_matches_fp32_all_forms` in
    /// `gpu::ops` is the tighter, more direct check on that path; this test's job is
    /// the layer's *composition* (bias seeding, grad accumulation, the AdamW step),
    /// which the wider bound still pins.
    fn tol(gpu: &Gpu, base: f32) -> f32 {
        if ops::gemm_bf16_enabled(gpu) {
            base * 20.0
        } else {
            base
        }
    }

    /// Tolerance for a parameter compared AFTER an Adam step.
    fn step_tol(gpu: &Gpu) -> f32 {
        if ops::gemm_bf16_enabled(gpu) {
            2e-3
        } else {
            1e-5
        }
    }

    /// `drop_saved_act` releases the narrowed *activations* and keeps the narrowed
    /// weight — and the weight it keeps must still follow the optimizer.
    #[test]
    fn drop_saved_act_keeps_the_weight_and_still_follows_a_step() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        if !ops::gemm_bf16_enabled(&gpu) {
            return; // no weight cache to keep
        }
        let (b, i, o) = (4usize, 6usize, 5usize);
        let cpu = CpuLinear::new(i, o);
        let mut gl = Linear::from_parts(&gpu, &cpu.w, &cpu.b);
        gl.freeze_bias();
        let x = GTensor::from_host(&gpu, &Tensor::random(&[b, i], 0.5));
        let dy = GTensor::from_host(&gpu, &Tensor::random(&[b, o], 1.0));

        let _ = gl.forward_alloc(&gpu, &x);
        gl.drop_saved_act(&gpu);
        // The weight is [i, o] = 30 elements at 2 bytes; the activations are gone.
        assert_eq!(
            gl.retained_bytes().1,
            i * o * 2,
            "drop_saved_act kept something other than exactly the narrowed weight"
        );

        // ...and the survivor is still live, not frozen at its pre-step value.
        for t in 1..=3 {
            let before = gl.forward_alloc(&gpu, &x).to_host(&gpu).data;
            let _ = gl.backward_alloc(&gpu, &dy, &tc);
            let mut c = AdamCfg::new(1e-2, 0.0);
            c.t = t;
            gl.step(&gpu, &c);
            gl.drop_saved_act(&gpu);
            let after = gl.forward_alloc(&gpu, &x).to_host(&gpu).data;
            let changed = before
                .iter()
                .zip(&after)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                changed > 1e-5,
                "step {t}: the forward did not move — a weight kept across \
                 drop_saved_act went stale"
            );
        }
    }

    /// The bf16 weight cache must be dropped by every optimizer step.
    #[test]
    fn weight_cache_follows_optimizer_steps() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        let (b, i, o) = (4usize, 6usize, 5usize);
        let cpu = CpuLinear::new(i, o);
        let mut gl = Linear::from_parts(&gpu, &cpu.w, &cpu.b);
        gl.freeze_bias();
        let hx = Tensor::random(&[b, i], 0.5);
        let x = GTensor::from_host(&gpu, &hx);
        let hdy = Tensor::random(&[b, o], 1.0);
        let dy = GTensor::from_host(&gpu, &hdy);
        let cfg = AdamCfg::new(1e-2, 0.0);

        for t in 1..=3 {
            let before = gl.forward_alloc(&gpu, &x).to_host(&gpu).data;
            let w_before = gl.w.to_host(&gpu).data;

            let _ = gl.backward_alloc(&gpu, &dy, &tc);
            let mut c = cfg;
            c.t = t;
            gl.step(&gpu, &c);

            let w_after = gl.w.to_host(&gpu).data;
            let moved = w_before
                .iter()
                .zip(&w_after)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(moved > 1e-5, "step {t}: the optimizer did not move w");

            let after = gl.forward_alloc(&gpu, &x).to_host(&gpu).data;
            let changed = before
                .iter()
                .zip(&after)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                changed > 1e-5,
                "step {t}: w moved by {moved} but the forward output did not change — \
                 the bf16 weight cache was not invalidated"
            );
        }
    }

    /// GPU Linear must match the CPU `cpu::Linear` for a full
    /// forward → backward → AdamW-step cycle, starting from identical weights.
    #[test]
    fn linear_matches_cpu_layer() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        let (batch, input, output) = (12, 7, 5);

        let w = Tensor::xavier(input, output);
        let b = Tensor::random(&[output], 0.1);
        let mut cpu = CpuLinear::from_parts(w.clone(), b.clone());
        let mut dev = Linear::from_parts(&gpu, &w, &b);

        let x = Tensor::random(&[batch, input], 1.0);
        let dy = Tensor::random(&[batch, output], 1.0);

        // Forward
        let y_cpu = cpu.forward(&x);
        let y_dev = dev.forward_alloc(&gpu, &GTensor::from_host(&gpu, &x));
        assert_close(&y_dev.to_host(&gpu).data, &y_cpu.data, tol(&gpu, 1e-3));

        // Backward
        let dx_cpu = cpu.backward(&dy);
        let dx_dev = dev.backward_alloc(&gpu, &GTensor::from_host(&gpu, &dy), &tc);
        assert_close(&dx_dev.to_host(&gpu).data, &dx_cpu.data, tol(&gpu, 1e-3));
        assert_close(&dev.dw.to_host(&gpu).data, &cpu.dw.data, tol(&gpu, 1e-3));
        assert_close(&dev.db.to_host(&gpu).data, &cpu.db.data, 1e-3);

        // One AdamW step, then compare the updated parameters.
        let mut cfg = AdamCfg::new(1e-3, 0.01);
        cfg.t = 1;
        cpu.step(&cfg);
        dev.step(&gpu, &cfg);
        assert_close(&dev.w.to_host(&gpu).data, &cpu.w.data, step_tol(&gpu));
        assert_close(&dev.b.to_host(&gpu).data, &cpu.b.data, step_tol(&gpu));
    }

    /// The shared-input path must be numerically identical to the saving one, and
    /// must actually stop holding an `[N, in]` copy.
    #[test]
    fn forward_shared_matches_forward_and_saves_nothing() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        let (batch, input, output) = (12, 7, 5);
        let w = Tensor::xavier(input, output);
        let b = Tensor::random(&[output], 0.1);
        let x = GTensor::from_host(&gpu, &Tensor::random(&[batch, input], 1.0));
        let dy = GTensor::from_host(&gpu, &Tensor::random(&[batch, output], 1.0));

        let mut saving = Linear::from_parts(&gpu, &w, &b);
        let mut shared = Linear::from_parts(&gpu, &w, &b);

        let eq = |got: &[f32], want: &[f32], what: &str| {
            assert_eq!(
                got, want,
                "{what}: shared path diverged from the saving one"
            );
        };

        let y_saving = saving.forward_alloc(&gpu, &x);
        let mut y_shared = GTensor::uninit(&gpu, &[batch, output]);
        shared.forward_shared(&gpu, &x, &mut y_shared);
        eq(
            &y_shared.to_host(&gpu).data,
            &y_saving.to_host(&gpu).data,
            "y",
        );

        // The whole point: no retained input.
        assert!(shared.x.is_empty(), "forward_shared kept an [N, in] copy");
        assert!(!saving.x.is_empty(), "forward should keep its input");

        let dx_saving = saving.backward_alloc(&gpu, &dy, &tc);
        let mut dx_shared = GTensor::uninit(&gpu, &[batch, input]);
        shared.backward_with_x(&gpu, &x, &dy, &mut dx_shared, 0.0, &tc);

        eq(
            &dx_shared.to_host(&gpu).data,
            &dx_saving.to_host(&gpu).data,
            "dx",
        );
        eq(
            &shared.dw.to_host(&gpu).data,
            &saving.dw.to_host(&gpu).data,
            "dw",
        );

        eq(
            &shared.db.to_host(&gpu).data,
            &saving.db.to_host(&gpu).data,
            "db",
        );
    }

    /// `backward_with_x` over contiguous **row blocks** must sum to the whole-batch
    /// backward: `dW = Xᵀ·dY` and `db = Σ_rows dY` both contract over the row axis and
    /// accumulate at `beta = 1`, so splitting `N` is a reassociation. `dX = dY·Wᵀ`
    /// writes at `beta = 0` but each block owns disjoint rows, so it stays correct.
    #[test]
    fn backward_in_row_blocks_matches_whole_batch() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        let (batch, input, output, blk) = (14, 7, 5, 4); // 14 = 3·4 + 2 (ragged)
        let w = Tensor::xavier(input, output);
        let b = Tensor::random(&[output], 0.1);
        let x_h = Tensor::random(&[batch, input], 1.0);
        let dy_h = Tensor::random(&[batch, output], 1.0);
        let x = GTensor::from_host(&gpu, &x_h);
        let dy = GTensor::from_host(&gpu, &dy_h);

        let mut whole = Linear::from_parts(&gpu, &w, &b);
        let mut blocked = Linear::from_parts(&gpu, &w, &b);

        let dx_whole = whole.backward_alloc_with_x(&gpu, &x, &dy, &tc);

        let mut dx_blocked = Vec::with_capacity(batch * input);
        for r0 in (0..batch).step_by(blk) {
            let rows = blk.min(batch - r0);
            let xb = Tensor::new(
                &[rows, input],
                x_h.data[r0 * input..(r0 + rows) * input].to_vec(),
            );
            let dyb = Tensor::new(
                &[rows, output],
                dy_h.data[r0 * output..(r0 + rows) * output].to_vec(),
            );
            let part = blocked.backward_alloc_with_x(
                &gpu,
                &GTensor::from_host(&gpu, &xb),
                &GTensor::from_host(&gpu, &dyb),
                &tc,
            );
            dx_blocked.extend_from_slice(&part.to_host(&gpu).data);
        }

        assert_close(&dx_blocked, &dx_whole.to_host(&gpu).data, 1e-5);
        assert_close(
            &blocked.dw.to_host(&gpu).data,
            &whole.dw.to_host(&gpu).data,
            1e-4,
        );
        assert_close(
            &blocked.db.to_host(&gpu).data,
            &whole.db.to_host(&gpu).data,
            1e-4,
        );
    }
}
