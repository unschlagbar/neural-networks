//! Batched logit soft-cap: `y = cap · tanh(x / cap)`, applied element-wise.
//!
//! Weightless. Bounds logits to (−cap, cap); backward is the exact derivative
//! `1 − (y/cap)²`. Same math as `nn::soft_cap::SoftCapLayer`, on `[B, F]`.

use crate::nn2::ops;
use crate::tensor::Tensor;

pub struct SoftCap {
    pub cap: f32,
    /// Capped output `[B, F]`, saved for the exact backward.
    out: Tensor,
}

impl SoftCap {
    pub fn new(cap: f32) -> Self {
        assert!(cap > 0.0, "SoftCap cap must be positive");
        Self {
            cap,
            out: Tensor::zeros(&[0, 0]),
        }
    }

    pub fn forward(&mut self, x: &Tensor) -> Tensor {
        let y = ops::softcap_forward(x, self.cap);
        self.out = y.clone();
        y
    }

    /// `dX = dY · (1 − (y/cap)²)`.
    pub fn backward(&self, dy: &Tensor) -> Tensor {
        ops::softcap_backward(dy, &self.out, self.cap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caps_and_backward_matches_finite_difference() {
        let cap = 30.0;
        let mut sc = SoftCap::new(cap);
        let x = Tensor::new(&[1, 4], vec![-45.0, -3.0, 0.5, 60.0]);
        let g = Tensor::new(&[1, 4], vec![1.0, -2.0, 0.7, 3.0]);

        let y = sc.forward(&x);
        for &v in &y.data {
            assert!(v.abs() < cap);
        }
        let dx = sc.backward(&g);

        let eps = 1e-2;
        for i in 0..4 {
            let mut xp = x.clone();
            xp.data[i] += eps;
            let lp: f32 = sc
                .forward(&xp)
                .data
                .iter()
                .zip(&g.data)
                .map(|(a, b)| a * b)
                .sum();
            xp.data[i] -= 2.0 * eps;
            let lm: f32 = sc
                .forward(&xp)
                .data
                .iter()
                .zip(&g.data)
                .map(|(a, b)| a * b)
                .sum();
            let num = (lp - lm) / (2.0 * eps);
            assert!(
                (dx.data[i] - num).abs() < 1e-3,
                "dx[{i}] {} vs {num}",
                dx.data[i]
            );
        }
    }
}
