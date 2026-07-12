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
        }
    }

    #[inline]
    pub fn input_size(&self) -> usize {
        self.input
    }
    #[inline]
    pub fn output_size(&self) -> usize {
        self.output
    }

    /// `Y = X · W + b`, `x` is `[B, in]`, result `[B, out]`. Saves `x` for backward.
    pub fn forward(&mut self, gpu: &Gpu, x: &DTensor) -> DTensor {
        let b = x.rows();
        assert_eq!(x.cols(), self.input, "Linear::forward — input width mismatch");
        // Reuse the saved-input buffer when the batch size is unchanged (steady
        // state), else reallocate — avoids a per-call device allocation.
        if self.x.rank == 2 && self.x.rows() == b {
            gpu.stream.memcpy_dtod(&x.buf, &mut self.x.buf).expect("save x");
        } else {
            self.x = x.dup(gpu);
        }
        // Seed each output row with the bias (fills y entirely, so uninit is safe),
        // then accumulate X·W on top (beta=1).
        let mut y = DTensor::uninit(gpu, &[b, self.output]);
        ops::broadcast_row(gpu, &mut y, &self.b);
        ops::matmul_nn_into(gpu, x, &self.w, &mut y, 1.0);
        y
    }

    /// Given `dY` `[B, out]`, accumulate `dW`/`db` and return `dX = dY · Wᵀ`.
    pub fn backward(&mut self, gpu: &Gpu, dy: &DTensor) -> DTensor {
        assert_eq!(dy.cols(), self.output, "Linear::backward — grad width mismatch");
        assert_eq!(self.x.rows(), dy.rows(), "Linear::backward — batch mismatch");
        // dW += Xᵀ · dY ; db += Σ_batch dY
        ops::matmul_tn_into(gpu, &self.x, dy, &mut self.dw, 1.0);
        ops::add_col_sum(gpu, &mut self.db, dy);
        // dX = dY · Wᵀ (cuBLAS transposes W(in×out) internally — no host transpose).
        ops::matmul_nt(gpu, dy, &self.w)
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
        ops::adamw(gpu, &mut self.w, &self.dw, &mut self.mw, &mut self.vw, cfg, decay_w);
        ops::adamw(gpu, &mut self.b, &self.db, &mut self.mb, &mut self.vb, cfg, false);
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

    /// GPU Linear must match the CPU `nn2::Linear` for a full
    /// forward → backward → AdamW-step cycle, starting from identical weights.
    #[test]
    fn linear_matches_cpu_layer() {
        let Some(gpu) = super::super::test_gpu() else { return };
        let (batch, input, output) = (12, 7, 5);

        let w = Tensor::xavier(input, output);
        let b = Tensor::random(&[output], 0.1);
        let mut cpu = CpuLinear::from_parts(w.clone(), b.clone());
        let mut dev = Linear::from_parts(&gpu, &w, &b);

        let x = Tensor::random(&[batch, input], 1.0);
        let dy = Tensor::random(&[batch, output], 1.0);

        // Forward
        let y_cpu = cpu.forward(&x);
        let y_dev = dev.forward(&gpu, &DTensor::from_host(&gpu, &x));
        assert_close(&y_dev.to_host(&gpu).data, &y_cpu.data, 1e-3);

        // Backward
        let dx_cpu = cpu.backward(&dy);
        let dx_dev = dev.backward(&gpu, &DTensor::from_host(&gpu, &dy));
        assert_close(&dx_dev.to_host(&gpu).data, &dx_cpu.data, 1e-3);
        assert_close(&dev.dw.to_host(&gpu).data, &cpu.dw.data, 1e-3);
        assert_close(&dev.db.to_host(&gpu).data, &cpu.db.data, 1e-3);

        // One AdamW step, then compare the updated parameters.
        let mut cfg = AdamCfg::new(1e-3, 0.01);
        cfg.t = 1;
        cpu.step(&cfg);
        dev.step(&gpu, &cfg);
        assert_close(&dev.w.to_host(&gpu).data, &cpu.w.data, 1e-5);
        assert_close(&dev.b.to_host(&gpu).data, &cpu.b.data, 1e-5);
    }
}
