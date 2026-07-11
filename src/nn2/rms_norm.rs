//! Batched RMSNorm — pure normalization (no residual, no inner layer).
//!
//! Unlike the old `nn::rms_norm::RMSNorm`, which bakes in a pre-norm residual
//! and an inner layer, this is just the normalization op:
//!
//!   rms   = sqrt( mean(xᵢ²) + ε )
//!   x̂ᵢ   = xᵢ / rms
//!   yᵢ   = γᵢ · x̂ᵢ
//!
//! applied independently to each row (feature vector) of a `[B, F]` tensor.
//! Residual connections are the block's responsibility, kept explicit.

use crate::nn2::ops;
use crate::nn2::optim::{AdamCfg, AdamState};
use crate::tensor::Tensor;

/// Matches the old `EPS` so the two systems normalize identically.
const EPS: f32 = 1e-6;

pub struct RmsNorm {
    pub gamma: Tensor,  // [F]
    pub dgamma: Tensor, // [F]

    /// Per-row `1/rms`, saved from forward for the backward pass. Length `B`.
    inv_rms: Vec<f32>,
    /// Normalized activations `x̂` `[B, F]`, saved for backward.
    x_hat: Tensor,
    size: usize,
    opt: AdamState,
}

impl RmsNorm {
    pub fn new(size: usize) -> Self {
        Self {
            gamma: Tensor::new(&[size], vec![1.0; size]),
            dgamma: Tensor::zeros(&[size]),
            inv_rms: Vec::new(),
            x_hat: Tensor::zeros(&[0, size]),
            size,
            opt: AdamState::new(),
        }
    }

    /// AdamW step (norm scale is never decayed). Clears the grad.
    pub fn step(&mut self, cfg: &AdamCfg) {
        self.opt.step(&mut self.gamma.data, &self.dgamma.data, cfg, false);
        self.zero_grad();
    }

    /// `y = γ ⊙ (x / rms(x))`, row-wise. `x` is `[B, F]`, result `[B, F]`.
    pub fn forward(&mut self, x: &Tensor) -> Tensor {
        assert_eq!(x.cols(), self.size, "RmsNorm::forward — width mismatch");
        // Plain RMSNorm is the single-group case (group == full width).
        let fwd = ops::rms_norm_forward(x, &self.gamma, self.size, EPS);
        self.x_hat = fwd.x_hat;
        self.inv_rms = fwd.inv_rms;
        fwd.out
    }

    /// Given `dY` `[B, F]`, accumulate `dγ` and return `dX` `[B, F]`.
    ///
    /// Per row:  S   = Σⱼ γⱼ · dYⱼ · x̂ⱼ
    ///           dXᵢ = inv_rms · ( γᵢ·dYᵢ − x̂ᵢ · S / F )
    pub fn backward(&mut self, dy: &Tensor) -> Tensor {
        assert_eq!(dy.cols(), self.size, "RmsNorm::backward — width mismatch");
        ops::rms_norm_backward(
            dy,
            &self.x_hat,
            &self.inv_rms,
            &self.gamma,
            &mut self.dgamma,
            self.size,
        )
    }

    pub fn zero_grad(&mut self) {
        self.dgamma.zero_();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backward_matches_finite_difference() {
        let (b, f) = (3, 6);
        let mut norm = RmsNorm::new(f);
        // Give gamma non-trivial values so its gradient is exercised.
        norm.gamma = Tensor::random(&[f], 0.5);
        for g in norm.gamma.data.iter_mut() {
            *g += 1.0;
        }

        let x = Tensor::random(&[b, f], 1.5);
        let g = Tensor::random(&[b, f], 1.0); // upstream dL/dY

        let _y = norm.forward(&x);
        let dx = norm.backward(&g);
        let dgamma = norm.dgamma.clone();

        let loss = |norm: &mut RmsNorm, x: &Tensor| -> f32 {
            let y = norm.forward(x);
            y.data.iter().zip(&g.data).map(|(a, b)| a * b).sum()
        };

        let eps = 1e-3;
        let tol = 2e-2;

        // dX
        let mut xp = x.clone();
        for i in 0..x.data.len() {
            let orig = x.data[i];
            xp.data[i] = orig + eps;
            let lp = loss(&mut norm, &xp);
            xp.data[i] = orig - eps;
            let lm = loss(&mut norm, &xp);
            xp.data[i] = orig;
            let num = (lp - lm) / (2.0 * eps);
            assert!((num - dx.data[i]).abs() < tol, "dX[{i}]: {num} vs {}", dx.data[i]);
        }

        // dgamma
        for i in 0..f {
            let orig = norm.gamma.data[i];
            norm.gamma.data[i] = orig + eps;
            let lp = loss(&mut norm, &x);
            norm.gamma.data[i] = orig - eps;
            let lm = loss(&mut norm, &x);
            norm.gamma.data[i] = orig;
            let num = (lp - lm) / (2.0 * eps);
            assert!((num - dgamma.data[i]).abs() < tol, "dgamma[{i}]: {num} vs {}", dgamma.data[i]);
        }
    }
}
