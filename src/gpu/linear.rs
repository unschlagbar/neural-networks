//! Device-resident `Linear` (`Y = X · W + b`), the GPU counterpart of
//! [`nn2::linear::Linear`](crate::nn2::linear::Linear).
//!
//! Same weight layout (`W` is `[in, out]`, forward is `X · W`) and the same
//! AdamW convention (weight decays, bias does not), so a GPU layer built from a
//! CPU layer's `[w, b]` produces bit-comparable results — which the parity test
//! checks against `nn2::Linear` for forward, backward and one optimizer step.
//!
//! The whole layer lives on the GPU: weights, gradient accumulators, AdamW
//! moments and the saved forward input are all `DTensor`s, so a forward→backward→
//! step cycle touches host memory only if the caller downloads a result.

use super::{DTensor, Gpu, ops};
use crate::nn2::optim::AdamCfg;
use crate::tensor::Tensor;

pub struct Linear {
    pub w: DTensor, // [in, out]
    pub b: DTensor, // [out]
    pub dw: DTensor,
    pub db: DTensor,
    /// Saved forward input `[B, in]` (device copy) for the weight gradient.
    x: DTensor,
    /// AdamW moments for `w` and `b`.
    mw: DTensor,
    vw: DTensor,
    mb: DTensor,
    vb: DTensor,
    input: usize,
    output: usize,
    /// bf16 staging for this layer's three GEMMs (see `ops::GemmBf16`). Holds the
    /// narrowed operand copies across calls so the cast never allocates.
    ///
    /// This mirrors the reference's autocast boundary: its projections run under
    /// `@custom_fwd(cast_inputs="bfloat16")`, i.e. fp32 master weights narrowed per
    /// call, tensor-core matmul, fp32 accumulator. `w`/`b` above stay fp32 — the
    /// optimizer steps them there and the checkpoint stores them there.
    gemm: ops::GemmBf16,
    /// Whether this layer's GEMMs go through the bf16 path. Pinned at construction
    /// so forward and backward cannot disagree, and `false` for a layer that must
    /// stay exact (see `set_fp32`).
    bf16: bool,
}

impl Linear {
    /// Build from host weight `[in, out]` and bias `[out]` tensors (uploads them).
    pub fn from_parts(gpu: &Gpu, w: &Tensor, b: &Tensor) -> Self {
        let (input, output) = (w.rows(), w.cols());
        assert_eq!(b.len(), output, "Linear::from_parts — bias len != out");
        Self {
            w: DTensor::from_host(gpu, w),
            b: DTensor::from_host(gpu, b),
            dw: DTensor::zeros(gpu, &[input, output]),
            db: DTensor::zeros(gpu, &[output]),
            x: DTensor::zeros(gpu, &[0, input]),
            mw: DTensor::zeros(gpu, &[input, output]),
            vw: DTensor::zeros(gpu, &[input, output]),
            mb: DTensor::zeros(gpu, &[output]),
            vb: DTensor::zeros(gpu, &[output]),
            input,
            output,
            gemm: ops::GemmBf16::new(),
            bf16: ops::gemm_bf16_enabled(gpu),
        }
    }

    /// Force this layer's GEMMs back to fp32.
    ///
    /// For the few matmuls where the narrowed operands are not worth it: the logit
    /// head, whose output feeds the softmax/cross-entropy directly, and any layer a
    /// test needs bit-comparable against the CPU reference.
    pub fn set_fp32(&mut self) {
        self.bf16 = false;
    }

    #[inline]
    pub fn input_size(&self) -> usize {
        self.input
    }
    #[inline]
    pub fn output_size(&self) -> usize {
        self.output
    }

    /// `Y = X · W + b` into the caller's `y` `[B, out]`; `x` is `[B, in]`. Saves
    /// `x` for backward.
    pub fn forward(&mut self, gpu: &Gpu, x: &DTensor, y: &mut DTensor) {
        let b = x.rows();
        assert_eq!(
            x.cols(),
            self.input,
            "Linear::forward — input width mismatch"
        );
        assert_eq!(y.dims(), [b, self.output], "Linear::forward — output shape");
        // Reuse the saved-input buffer when the batch size is unchanged (steady
        // state), else reallocate — avoids a per-call device allocation.
        if self.x.len() == b * self.input {
            self.x.reshape_to(&[b, self.input]);
            self.x.copy_from(gpu, x);
        } else {
            self.x = x.dup(gpu);
        }
        self.forward_raw(gpu, x, y);
    }

    /// `Y = X · W + b` for a caller-owned input, saving **nothing**.
    ///
    /// [`forward`](Self::forward) deep-copies `x` into `self.x` so backward can form
    /// `dW = Xᵀ·dY`. When several layers share one input — mLSTM hands the same `xf`
    /// to all six projections — that is five identical `[N, in]` copies (21 MB per
    /// cell at H=1024, 252 MB across the backbone). Here the caller keeps `x` alive
    /// to its own backward instead and passes it to
    /// [`backward_with_x`](Self::backward_with_x).
    ///
    /// The saved-input buffer is released, not merely bypassed: a layer that has ever
    /// taken this path must not keep a stale `[N, in]` allocation resident. That frees
    /// at most once per layer — after the first shared forward `self.x` is already
    /// empty — so it is not an allocation on the steady-state path.
    pub fn forward_shared(&mut self, gpu: &Gpu, x: &DTensor, y: &mut DTensor) {
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
            self.x = DTensor::uninit(gpu, &[0, self.input]);
        }
        self.forward_raw(gpu, x, y);
    }

    /// The GEMM half of forward, without the input-saving policy. Shared by
    /// [`forward`](Self::forward) and [`forward_shared`](Self::forward_shared) so the
    /// two cannot drift apart.
    fn forward_raw(&mut self, gpu: &Gpu, x: &DTensor, y: &mut DTensor) {
        // Seed each output row with the bias (fills y entirely, so uninit is safe),
        // then accumulate X·W on top (beta=1).
        ops::broadcast_row(gpu, y, &self.b);
        if self.bf16 {
            self.gemm.run(gpu, ops::MmForm::Nn, x, &self.w, y, 1.0);
        } else {
            ops::matmul_nn_into(gpu, x, &self.w, y, 1.0);
        }
    }

    /// `Y = X · W + b` into a freshly allocated `[B, out]`.
    ///
    /// The allocation-free [`forward`](Self::forward) is the one to use on a hot
    /// path; this exists for call sites that still compose by value (mLSTM's
    /// projection shell), where the output feeds straight into the next op.
    pub fn forward_alloc(&mut self, gpu: &Gpu, x: &DTensor) -> DTensor {
        let mut y = DTensor::uninit(gpu, &[x.rows(), self.output]);
        self.forward(gpu, x, &mut y);
        y
    }

    /// Backward into a freshly allocated `dX` `[B, in]` — the by-value companion
    /// to [`backward`](Self::backward).
    pub fn backward_alloc(&mut self, gpu: &Gpu, dy: &DTensor) -> DTensor {
        let mut dx = DTensor::uninit(gpu, &[dy.rows(), self.input]);
        self.backward(gpu, dy, &mut dx);
        dx
    }

    /// [`backward_with_x`](Self::backward_with_x) into a freshly allocated `dX`.
    pub fn backward_alloc_with_x(&mut self, gpu: &Gpu, x: &DTensor, dy: &DTensor) -> DTensor {
        let mut dx = DTensor::uninit(gpu, &[dy.rows(), self.input]);
        self.backward_with_x(gpu, x, dy, &mut dx);
        dx
    }

    /// Given `dY` `[B, out]`, accumulate `dW`/`db` and write `dX = dY · Wᵀ` into
    /// `dx` `[B, in]`.
    pub fn backward(&mut self, gpu: &Gpu, dy: &DTensor, dx: &mut DTensor) {
        assert_eq!(
            self.x.rows(),
            dy.rows(),
            "Linear::backward — batch mismatch"
        );
        // `self.x` and `self.dw` are disjoint fields, but the borrow checker cannot
        // see that through a `&mut self` method call — so the shared body takes the
        // three tensors it touches, not `self`.
        Self::grad_w(gpu, &mut self.gemm, self.bf16, &self.x, dy, &mut self.dw);
        ops::add_col_sum(gpu, &mut self.db, dy);
        self.grad_x(gpu, dy, dx);
    }

    /// Backward for a caller-owned input: the companion to
    /// [`forward_shared`](Self::forward_shared).
    ///
    /// `x` must be the `[B, in]` this layer saw in forward — or, for a chunked
    /// caller, a contiguous **row block** of it paired with the matching row block of
    /// `dy`. Both weight terms contract over the row axis
    /// (`dW = Xᵀ·dY`, `db = Σ_rows dY`) and both accumulate at `beta = 1`, so
    /// summing over row blocks gives the identical result — splitting `N` is a
    /// reassociation, not a change of math. `dX = dY·Wᵀ` writes at `beta = 0`, which
    /// stays correct because each block owns disjoint rows of `dx`.
    pub fn backward_with_x(&mut self, gpu: &Gpu, x: &DTensor, dy: &DTensor, dx: &mut DTensor) {
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
        Self::grad_w(gpu, &mut self.gemm, self.bf16, x, dy, &mut self.dw);
        ops::add_col_sum(gpu, &mut self.db, dy);
        self.grad_x(gpu, dy, dx);
    }

    /// `dW += Xᵀ · dY` (accumulating, `beta = 1`).
    ///
    /// Free-standing in the three tensors it touches rather than a `&mut self`
    /// method, so a caller holding `self.x` borrowed can still reach `self.dw`.
    fn grad_w(
        gpu: &Gpu,
        gemm: &mut ops::GemmBf16,
        bf16: bool,
        x: &DTensor,
        dy: &DTensor,
        dw: &mut DTensor,
    ) {
        if bf16 {
            gemm.run(gpu, ops::MmForm::Tn, x, dy, dw, 1.0);
        } else {
            ops::matmul_tn_into(gpu, x, dy, dw, 1.0);
        }
    }

    /// `dX = dY · Wᵀ` (overwriting, `beta = 0`). cuBLAS transposes `W(in×out)`
    /// internally — no host transpose.
    fn grad_x(&mut self, gpu: &Gpu, dy: &DTensor, dx: &mut DTensor) {
        if self.bf16 {
            self.gemm.run(gpu, ops::MmForm::Nt, dy, &self.w, dx, 0.0);
        } else {
            ops::matmul_nt_into(gpu, dy, &self.w, dx, 0.0);
        }
    }

    /// Freshly-initialised layer, matching `nn2::Linear::new`'s init exactly
    /// (built on the host, then uploaded).
    pub fn new_rand(gpu: &Gpu, input: usize, output: usize) -> Self {
        let cpu = crate::nn2::Linear::new(input, output);
        Self::from_parts(gpu, &cpu.w, &cpu.b)
    }

    /// Every learnable tensor, in a fixed order (used by checkpoint save/load).
    pub fn params_mut(&mut self) -> Vec<&mut DTensor> {
        vec![&mut self.w, &mut self.b]
    }

    pub fn zero_grad(&mut self, gpu: &Gpu) {
        // In-place memset — reuse the existing allocations across steps.
        self.dw.zero_(gpu);
        self.db.zero_(gpu);
    }

    /// AdamW step (weight decay on `w`, none on `b`), then clears the accumulators.
    pub fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        self.step_wd(gpu, cfg, true);
    }

    /// AdamW step with explicit weight-decay control on `w` (pass `false` for an
    /// undecayed logit head).
    pub fn step_wd(&mut self, gpu: &Gpu, cfg: &AdamCfg, decay_w: bool) {
        ops::adamw(
            gpu,
            &mut self.w,
            &self.dw,
            &mut self.mw,
            &mut self.vw,
            cfg,
            decay_w,
        );
        ops::adamw(
            gpu,
            &mut self.b,
            &self.db,
            &mut self.mb,
            &mut self.vb,
            cfg,
            false,
        );
        self.zero_grad(gpu);
    }

    /// AdamW step on `w` only, leaving `b` untouched; clears both grads. Used for
    /// a bias-free head: `b` stays at its (zero) init so the layer is equivalent
    /// to `nn::LinearNBLayer` and exports to the no-bias `HIER` head faithfully.
    pub fn step_w_only(&mut self, gpu: &Gpu, cfg: &AdamCfg, decay_w: bool) {
        ops::adamw(
            gpu,
            &mut self.w,
            &self.dw,
            &mut self.mw,
            &mut self.vw,
            cfg,
            decay_w,
        );
        self.zero_grad(gpu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn2::linear::Linear as CpuLinear;
    use crate::nn2::optim::AdamCfg;

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
        if ops::gemm_bf16_enabled(gpu) { base * 20.0 } else { base }
    }

    /// Tolerance for a parameter compared AFTER an Adam step.
    ///
    /// Deliberately not `tol(gpu, 1e-5)`. Adam's update is `lr·ĝ/(√v̂+ε)`, which is
    /// scale-invariant in the gradient: a weight whose gradient is near zero has its
    /// bf16-sized difference divided by an equally small √v̂, so a ~1e-7 gradient
    /// wobble lands as ~1e-4 on the weight. Scaling the activation tolerance by the
    /// same factor is not enough — it leaves a bound that passes in isolation and
    /// fails on maybe one run in five, which is worse than no check at all.
    fn step_tol(gpu: &Gpu) -> f32 {
        if ops::gemm_bf16_enabled(gpu) { 2e-3 } else { 1e-5 }
    }

    /// GPU Linear must match the CPU `nn2::Linear` for a full
    /// forward → backward → AdamW-step cycle, starting from identical weights.
    #[test]
    fn linear_matches_cpu_layer() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let (batch, input, output) = (12, 7, 5);

        let w = Tensor::xavier(input, output);
        let b = Tensor::random(&[output], 0.1);
        let mut cpu = CpuLinear::from_parts(w.clone(), b.clone());
        let mut dev = Linear::from_parts(&gpu, &w, &b);

        let x = Tensor::random(&[batch, input], 1.0);
        let dy = Tensor::random(&[batch, output], 1.0);

        // Forward
        let y_cpu = cpu.forward(&x);
        let y_dev = dev.forward_alloc(&gpu, &DTensor::from_host(&gpu, &x));
        assert_close(&y_dev.to_host(&gpu).data, &y_cpu.data, tol(&gpu, 1e-3));

        // Backward
        let dx_cpu = cpu.backward(&dy);
        let dx_dev = dev.backward_alloc(&gpu, &DTensor::from_host(&gpu, &dy));
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
    ///
    /// This is what lets mLSTM keep one `xf` for all six projections instead of six
    /// copies of it (252 MB across the backbone).
    #[test]
    fn forward_shared_matches_forward_and_saves_nothing() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let (batch, input, output) = (12, 7, 5);
        let w = Tensor::xavier(input, output);
        let b = Tensor::random(&[output], 0.1);
        let x = DTensor::from_host(&gpu, &Tensor::random(&[batch, input], 1.0));
        let dy = DTensor::from_host(&gpu, &Tensor::random(&[batch, output], 1.0));

        let mut saving = Linear::from_parts(&gpu, &w, &b);
        let mut shared = Linear::from_parts(&gpu, &w, &b);

        // Both paths issue the identical GEMM over the identical data — the only
        // difference is whether `x` was copied first — so this is exact equality,
        // not a tolerance.
        let eq = |got: &[f32], want: &[f32], what: &str| {
            assert_eq!(got, want, "{what}: shared path diverged from the saving one");
        };

        let y_saving = saving.forward_alloc(&gpu, &x);
        let mut y_shared = DTensor::uninit(&gpu, &[batch, output]);
        shared.forward_shared(&gpu, &x, &mut y_shared);
        eq(&y_shared.to_host(&gpu).data, &y_saving.to_host(&gpu).data, "y");

        // The whole point: no retained input.
        assert!(shared.x.is_empty(), "forward_shared kept an [N, in] copy");
        assert!(!saving.x.is_empty(), "forward should keep its input");

        let dx_saving = saving.backward_alloc(&gpu, &dy);
        let mut dx_shared = DTensor::uninit(&gpu, &[batch, input]);
        shared.backward_with_x(&gpu, &x, &dy, &mut dx_shared);

        eq(&dx_shared.to_host(&gpu).data, &dx_saving.to_host(&gpu).data, "dx");
        eq(&shared.dw.to_host(&gpu).data, &saving.dw.to_host(&gpu).data, "dw");
        eq(&shared.db.to_host(&gpu).data, &saving.db.to_host(&gpu).data, "db");
    }

    /// `backward_with_x` over contiguous **row blocks** must sum to the whole-batch
    /// backward: `dW = Xᵀ·dY` and `db = Σ_rows dY` both contract over the row axis and
    /// accumulate at `beta = 1`, so splitting `N` is a reassociation. `dX = dY·Wᵀ`
    /// writes at `beta = 0` but each block owns disjoint rows, so it stays correct.
    ///
    /// This is the property the chunked-offload path rests on. The batch is
    /// deliberately **not** a multiple of the block size, to exercise a ragged last
    /// block.
    #[test]
    fn backward_in_row_blocks_matches_whole_batch() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let (batch, input, output, blk) = (14, 7, 5, 4); // 14 = 3·4 + 2 (ragged)
        let w = Tensor::xavier(input, output);
        let b = Tensor::random(&[output], 0.1);
        let x_h = Tensor::random(&[batch, input], 1.0);
        let dy_h = Tensor::random(&[batch, output], 1.0);
        let x = DTensor::from_host(&gpu, &x_h);
        let dy = DTensor::from_host(&gpu, &dy_h);

        let mut whole = Linear::from_parts(&gpu, &w, &b);
        let mut blocked = Linear::from_parts(&gpu, &w, &b);

        let dx_whole = whole.backward_alloc_with_x(&gpu, &x, &dy);

        // Row blocks are contiguous in a row-major [batch, ·] tensor, so a block is a
        // plain host sub-slice — no gather needed.
        let mut dx_blocked = Vec::with_capacity(batch * input);
        for r0 in (0..batch).step_by(blk) {
            let rows = blk.min(batch - r0);
            let xb = Tensor::new(&[rows, input], x_h.data[r0 * input..(r0 + rows) * input].to_vec());
            let dyb = Tensor::new(
                &[rows, output],
                dy_h.data[r0 * output..(r0 + rows) * output].to_vec(),
            );
            let part = blocked.backward_alloc_with_x(
                &gpu,
                &DTensor::from_host(&gpu, &xb),
                &DTensor::from_host(&gpu, &dyb),
            );
            dx_blocked.extend_from_slice(&part.to_host(&gpu).data);
        }

        assert_close(&dx_blocked, &dx_whole.to_host(&gpu).data, 1e-5);
        assert_close(&blocked.dw.to_host(&gpu).data, &whole.dw.to_host(&gpu).data, 1e-4);
        assert_close(&blocked.db.to_host(&gpu).data, &whole.db.to_host(&gpu).data, 1e-4);
    }
}
