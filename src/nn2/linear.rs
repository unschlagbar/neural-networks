//! Batched affine layer: `Y = X · W + b`.
//!
//! Weight layout matches the old `nn::linear::LinearLayer` (`W` is `[in, out]`,
//! forward is `X · W`) so a checkpoint could in principle move between the two
//! systems and so numeric comparisons line up.

use crate::nn2::ops;
use crate::nn2::optim::{AdamCfg, AdamState, ensure_states};
use crate::tensor::{Tensor, gemm};

pub struct Linear {
    pub w: Tensor, // [in, out]
    pub b: Tensor, // [out]

    /// Gradient accumulators (summed across the batch of windows, cleared per
    /// optimizer step). Same convention as the old system.
    pub dw: Tensor, // [in, out]
    pub db: Tensor, // [out]

    /// Saved forward input `[B, in]` for the weight gradient.
    x: Tensor,
    /// Pooled transpose of `w` (`[out, in]`), rebuilt each backward so `dX` can
    /// use the fast `gemm_nn` instead of the reduction-form `gemm_nt`.
    wt: Tensor,
    /// AdamW state for `[w, b]`.
    opt: Vec<AdamState>,
}

impl Linear {
    pub fn new(input: usize, output: usize) -> Self {
        Self {
            w: Tensor::xavier(input, output),
            b: Tensor::zeros(&[output]),
            dw: Tensor::zeros(&[input, output]),
            db: Tensor::zeros(&[output]),
            x: Tensor::zeros(&[0, input]),
            wt: Tensor::zeros(&[output, input]),
            opt: Vec::new(),
        }
    }

    pub fn from_parts(w: Tensor, b: Tensor) -> Self {
        let (input, output) = (w.rows(), w.cols());
        assert_eq!(b.len(), output, "Linear::from_parts — bias len != out");
        Self {
            w,
            b,
            dw: Tensor::zeros(&[input, output]),
            db: Tensor::zeros(&[output]),
            x: Tensor::zeros(&[0, input]),
            wt: Tensor::zeros(&[output, input]),
            opt: Vec::new(),
        }
    }

    #[inline]
    pub fn input_size(&self) -> usize {
        self.w.rows()
    }
    #[inline]
    pub fn output_size(&self) -> usize {
        self.w.cols()
    }

    /// `Y = X · W + b`, where `x` is `[B, in]` and the result is `[B, out]`.
    pub fn forward(&mut self, x: &Tensor) -> Tensor {
        let b = x.rows();
        let (input, output) = (self.input_size(), self.output_size());
        assert_eq!(x.cols(), input, "Linear::forward — input width mismatch");

        // Reuse the saved-input buffer across calls (no per-call clone); only
        // reallocate when the batch size changes.
        if self.x.rank() != 2 || self.x.rows() != b {
            self.x = Tensor::zeros(&[b, input]);
        }
        self.x.data.copy_from_slice(&x.data);

        // Seed each output row with the bias, then accumulate the matmul on top
        // (gemm beta=1) — avoids both the double-zero and a separate bias pass.
        let mut y = Tensor::zeros(&[b, output]);
        ops::broadcast_row(&mut y, &self.b);
        gemm::gemm_nn(b, input, output, &self.x.data, &self.w.data, &mut y.data, 1.0);
        y
    }

    /// Given `dY = dL/dY` `[B, out]`, accumulate `dW`, `db`, and return
    /// `dX = dY · Wᵀ` `[B, in]`.
    pub fn backward(&mut self, dy: &Tensor) -> Tensor {
        let b = dy.rows();
        let (input, output) = (self.input_size(), self.output_size());
        assert_eq!(dy.cols(), output, "Linear::backward — grad width mismatch");
        assert_eq!(self.x.rows(), b, "Linear::backward — batch mismatch with cached input");

        // dW += Xᵀ · dY   (accumulate, beta = 1.0)
        gemm::gemm_tn(input, b, output, &self.x.data, &dy.data, &mut self.dw.data, 1.0);

        // db += Σ_batch dY
        ops::add_col_sum(&mut self.db, dy);

        // dX = dY · Wᵀ. Transpose W (in×out → out×in) once, then use the fast
        // gemm_nn form: dX[B,in] = dY[B,out] · Wt[out,in].
        gemm::transpose(input, output, &self.w.data, &mut self.wt.data);
        let mut dx = Tensor::zeros(&[b, input]);
        gemm::gemm_nn(b, output, input, &dy.data, &self.wt.data, &mut dx.data, 0.0);
        dx
    }

    pub fn zero_grad(&mut self) {
        self.dw.zero_();
        self.db.zero_();
    }

    /// AdamW step: weight decays, bias does not. Clears the accumulators.
    pub fn step(&mut self, cfg: &AdamCfg) {
        self.step_wd(cfg, true);
    }

    /// AdamW step with explicit weight-decay control on `w` — pass `false` for
    /// logit heads (undecayed, per the project convention; the SoftCap bounds
    /// the logits so the head has no incentive to grow).
    pub fn step_wd(&mut self, cfg: &AdamCfg, decay_w: bool) {
        ensure_states(&mut self.opt, 2);
        self.opt[0].step(&mut self.w.data, &self.dw.data, cfg, decay_w);
        self.opt[1].step(&mut self.b.data, &self.db.data, cfg, false);
        self.zero_grad();
    }

    /// Plain SGD step, then clears the accumulators. Kept for the toy smoke test.
    pub fn sgd_step(&mut self, lr: f32) {
        for (p, g) in self.w.data.iter_mut().zip(&self.dw.data) {
            *p -= lr * g;
        }
        for (p, g) in self.b.data.iter_mut().zip(&self.db.data) {
            *p -= lr * g;
        }
        self.zero_grad();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Central finite-difference check: for a scalar loss L = Σ (Y ⊙ G), the
    /// analytic gradients dW, db, dX must match numeric perturbation.
    #[test]
    fn linear_backward_matches_finite_difference() {
        let (batch, input, output) = (4, 5, 3);
        let mut lin = Linear::new(input, output);

        // Deterministic-ish input and an arbitrary upstream grad G (dL/dY).
        let x = Tensor::random(&[batch, input], 1.0);
        let g = Tensor::random(&[batch, output], 1.0);

        // Analytic pass.
        let _y = lin.forward(&x);
        let dx = lin.backward(&g);
        let dw = lin.dw.clone();
        let db = lin.db.clone();

        // L(params) = Σ (forward(x) ⊙ G).
        let loss = |lin: &mut Linear, x: &Tensor| -> f32 {
            let y = lin.forward(x);
            y.data.iter().zip(&g.data).map(|(a, b)| a * b).sum()
        };

        let eps = 1e-3;
        let tol = 2e-2;

        // dW
        for i in 0..lin.w.data.len() {
            let orig = lin.w.data[i];
            lin.w.data[i] = orig + eps;
            let lp = loss(&mut lin, &x);
            lin.w.data[i] = orig - eps;
            let lm = loss(&mut lin, &x);
            lin.w.data[i] = orig;
            let num = (lp - lm) / (2.0 * eps);
            assert!((num - dw.data[i]).abs() < tol, "dW[{i}]: num {num} vs analytic {}", dw.data[i]);
        }

        // db
        for o in 0..output {
            let orig = lin.b.data[o];
            lin.b.data[o] = orig + eps;
            let lp = loss(&mut lin, &x);
            lin.b.data[o] = orig - eps;
            let lm = loss(&mut lin, &x);
            lin.b.data[o] = orig;
            let num = (lp - lm) / (2.0 * eps);
            assert!((num - db.data[o]).abs() < tol, "db[{o}]: num {num} vs analytic {}", db.data[o]);
        }

        // dX
        let mut xp = x.clone();
        for i in 0..x.data.len() {
            let orig = x.data[i];
            xp.data[i] = orig + eps;
            let lp = loss(&mut lin, &xp);
            xp.data[i] = orig - eps;
            let lm = loss(&mut lin, &xp);
            xp.data[i] = orig;
            let num = (lp - lm) / (2.0 * eps);
            assert!((num - dx.data[i]).abs() < tol, "dX[{i}]: num {num} vs analytic {}", dx.data[i]);
        }
    }
}
