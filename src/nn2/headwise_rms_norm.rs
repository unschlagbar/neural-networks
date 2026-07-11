//! Batched head-wise RMSNorm: RMS normalization applied independently to each
//! head-slice of a `[B, d]` row.
//!
//! For each row split into `H` heads of size `dhv = d/H`:
//!   rms_h = sqrt(mean(x_h²) + ε),  out_h = γ_h ⊙ x_h / rms_h
//!
//! `γ` is a learnable per-element scale (`d` elements, init 1). Same math as
//! `nn::headwise_rms_norm::HeadwiseRMSNorm`, batched over the leading axis and
//! grouped over heads. (This is `RmsNorm` with an inner per-head group loop.)

use crate::nn2::ops;
use crate::tensor::Tensor;

const EPS: f32 = 1e-6;

pub struct HeadwiseRmsNorm {
    pub gamma: Tensor,  // [d]
    pub dgamma: Tensor, // [d]
    num_heads: usize,
    dhv: usize,

    /// Per-(row, head) `1/rms`, `[B, H]`, saved for backward.
    inv_rms: Vec<f32>,
    /// Normalized `x̂` `[B, d]`, saved for backward.
    x_hat: Tensor,
}

impl HeadwiseRmsNorm {
    pub fn new(d: usize, num_heads: usize) -> Self {
        assert_eq!(d % num_heads, 0, "HeadwiseRmsNorm: d must be divisible by heads");
        Self {
            gamma: Tensor::new(&[d], vec![1.0; d]),
            dgamma: Tensor::zeros(&[d]),
            num_heads,
            dhv: d / num_heads,
            inv_rms: Vec::new(),
            x_hat: Tensor::zeros(&[0, d]),
        }
    }

    #[inline]
    fn d(&self) -> usize {
        self.num_heads * self.dhv
    }

    /// `x` is `[B, d]`; returns `[B, d]`, each head-group normalized.
    pub fn forward(&mut self, x: &Tensor) -> Tensor {
        assert_eq!(x.cols(), self.d(), "HeadwiseRmsNorm::forward — width mismatch");
        // Head-wise RMSNorm is the grouped case with one group per head.
        let fwd = ops::rms_norm_forward(x, &self.gamma, self.dhv, EPS);
        self.x_hat = fwd.x_hat;
        self.inv_rms = fwd.inv_rms;
        fwd.out
    }

    /// Given `dY` `[B, d]`, accumulate `dγ` and return `dX` `[B, d]`.
    pub fn backward(&mut self, dy: &Tensor) -> Tensor {
        assert_eq!(dy.cols(), self.d(), "HeadwiseRmsNorm::backward — width mismatch");
        ops::rms_norm_backward(
            dy,
            &self.x_hat,
            &self.inv_rms,
            &self.gamma,
            &mut self.dgamma,
            self.dhv,
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
        let (b, d, heads) = (3, 8, 4); // dhv = 2
        let mut norm = HeadwiseRmsNorm::new(d, heads);
        norm.gamma = Tensor::random(&[d], 0.5);
        for g in norm.gamma.data.iter_mut() {
            *g += 1.0;
        }

        let x = Tensor::random(&[b, d], 1.5);
        let g = Tensor::random(&[b, d], 1.0);

        let _y = norm.forward(&x);
        let dx = norm.backward(&g);
        let dgamma = norm.dgamma.clone();

        let loss = |norm: &mut HeadwiseRmsNorm, x: &Tensor| -> f32 {
            let y = norm.forward(x);
            y.data.iter().zip(&g.data).map(|(a, b)| a * b).sum()
        };

        let eps = 1e-3;
        let tol = 2e-2;

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
        for i in 0..d {
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
