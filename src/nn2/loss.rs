//! Fused softmax + cross-entropy over a batch of logit rows.
//!
//! Batched analogue of the old flat model's `Softmax` output: the forward
//! softmax and the cross-entropy gradient are fused, so backward returns the
//! standard `p - onehot` directly (no separate softmax Jacobian pass).

use crate::tensor::Tensor;

/// Row-wise softmax of a `[B, C]` logits tensor (numerically stabilized).
pub fn softmax_rows(logits: &Tensor) -> Tensor {
    let c = logits.cols();
    let mut out = Tensor::zeros(&[logits.rows(), c]);
    for (row_in, row_out) in logits.data.chunks(c).zip(out.data.chunks_mut(c)) {
        let max = row_in.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for (o, &z) in row_out.iter_mut().zip(row_in) {
            let e = (z - max).exp();
            *o = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        row_out.iter_mut().for_each(|o| *o *= inv);
    }
    out
}

/// Mean cross-entropy loss and its gradient w.r.t. the logits.
///
/// `logits` is `[B, C]`, `targets` holds `B` class indices. Returns
/// `(mean_loss, dlogits)` where `dlogits = (softmax − onehot) / B` — already
/// averaged over the batch, ready to feed straight into `Linear::backward`.
pub fn softmax_cross_entropy(logits: &Tensor, targets: &[usize]) -> (f32, Tensor) {
    let (b, c) = (logits.rows(), logits.cols());
    assert_eq!(
        targets.len(),
        b,
        "softmax_cross_entropy — targets len != batch"
    );

    let mut probs = softmax_rows(logits);
    let inv_b = 1.0 / b as f32;
    let mut loss = 0.0;

    for (row, &t) in probs.data.chunks_mut(c).zip(targets) {
        debug_assert!(t < c, "target {t} out of range for {c} classes");
        loss -= row[t].max(1e-30).ln();
        // p - onehot, then average over the batch.
        row[t] -= 1.0;
        row.iter_mut().for_each(|v| *v *= inv_b);
    }

    (loss * inv_b, probs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_rows_sum_to_one() {
        let logits = Tensor::new(&[2, 3], vec![1.0, 2.0, 3.0, -1.0, 0.0, 1.0]);
        let p = softmax_rows(&logits);
        for row in p.data.chunks(3) {
            let s: f32 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "row sums to {s}");
        }
    }

    /// dlogits from the fused path must match a finite-difference gradient of
    /// the mean cross-entropy loss.
    #[test]
    fn ce_grad_matches_finite_difference() {
        let (b, c) = (3, 4);
        let mut logits = Tensor::random(&[b, c], 1.0);
        let targets = [0usize, 2, 1];

        let (_loss, grad) = softmax_cross_entropy(&logits, &targets);

        let only_loss = |lg: &Tensor| softmax_cross_entropy(lg, &targets).0;
        let eps = 1e-3;
        for i in 0..logits.data.len() {
            let orig = logits.data[i];
            logits.data[i] = orig + eps;
            let lp = only_loss(&logits);
            logits.data[i] = orig - eps;
            let lm = only_loss(&logits);
            logits.data[i] = orig;
            let num = (lp - lm) / (2.0 * eps);
            assert!(
                (num - grad.data[i]).abs() < 1e-3,
                "grad[{i}]: num {num} vs {}",
                grad.data[i]
            );
        }
    }
}
