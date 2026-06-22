use iron_oxide::collections::Matrix;

use crate::optimizers::{
    GradMatrixOps, OptimizerGradTypes,
    adam::{AdamGradMatrix, AdamGradVec},
};

// Muon — MomentUm Orthogonalized by Newton-schulz (Keller Jordan, 2024).
// https://kellerjordan.github.io/posts/muon/
//
// For 2D hidden weight matrices only: the momentum-smoothed gradient is replaced
// by its nearest semi-orthogonal matrix (all singular values driven toward 1) via
// a fixed 5-step Newton-Schulz iteration, and that is used as the weight update.
//
// Embeddings (GradMatrixNoDecay) and 1D params / biases (GradVec) are NOT
// orthogonalized — they fall back to Adam. This is exactly the "MuonWithAuxAdam"
// hybrid recommended by the author. Note: the same scheduled `lr` is handed to
// both the Muon matrices and the aux-Adam params here; Muon wants a markedly
// larger lr (~0.02) than Adam, so retune the schedule when switching to Muon.

const MOMENTUM: f32 = 0.95;
const NESTEROV: bool = true;
const NS_STEPS: usize = 5;
/// Decoupled weight decay (λ), applied directly to the weights like AdamW.
/// Muon's reference default is 0; a small value (~0.1) is common for LM pretraining.
const WEIGHT_DECAY: f32 = 0.02;
/// Per-element gradient clip, guarding against blow-ups before orthogonalization.
const CLIP: f32 = 10.0;

// Newton-Schulz quintic coefficients (Jordan 2024). The polynomial
// f(x) = a·x + b·x³ + c·x⁵ is tuned so that 5 iterations push every singular
// value toward 1. It deliberately does NOT converge — it is optimised to behave
// well for exactly this fixed step count, with a steep slope at 0 to inflate
// small singular values fast.
const NS_A: f32 = 3.4445;
const NS_B: f32 = -4.7750;
const NS_C: f32 = 2.0315;

#[derive(Debug)]
pub struct Muon;

pub struct MuonGradMatrix {
    grads: Matrix,
    momentum: Matrix,
}

/// Orthogonalize `g` via the quintic Newton-Schulz iteration, returning a matrix
/// of the same shape whose singular values are ≈ 1 (the "zeroth power" U·Vᵀ of
/// the SVD G = U·Σ·Vᵀ, approximated without ever computing the SVD).
fn newton_schulz5(g: &Matrix) -> Matrix {
    // Iterate on the "wide" orientation (rows ≤ cols) so A = X·Xᵀ is the smaller
    // square; transpose the result back at the end.
    let transposed = g.rows() > g.cols();
    let mut x = if transposed { g.transpose() } else { g.clone() };

    // Normalize by the Frobenius norm so the largest singular value is ≤ 1, i.e.
    // start inside the basin where the iteration is well-behaved.
    let fro = x.as_slice().iter().map(|v| v * v).sum::<f32>().sqrt();
    x.scale(1.0 / (fro + 1e-7));

    for _ in 0..NS_STEPS {
        let a = x.mul(&x.transpose()); // A = X·Xᵀ
        let aa = a.mul(&a); // A²
        // B = b·A + c·A²
        let mut b = a;
        b.scale(NS_B);
        b.add_inplace_scaled(&aa, NS_C);
        // X = a·X + B·X
        let bx = b.mul(&x);
        x.scale(NS_A);
        x.add_inplace(&bx);
    }

    if transposed { x.transpose() } else { x }
}

impl GradMatrixOps for MuonGradMatrix {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            grads: Matrix::zeros(rows, cols),
            momentum: Matrix::zeros(rows, cols),
        }
    }

    fn apply_to(&mut self, weights: &mut Matrix, lr: f32) {
        debug_assert_eq!(weights.rows(), self.grads.rows());
        debug_assert_eq!(weights.cols(), self.grads.cols());

        // momentum ← β·momentum + (1-β)·g
        self.momentum.scale(MOMENTUM);
        self.momentum
            .add_inplace_scaled(&self.grads, 1.0 - MOMENTUM);

        // Nesterov: update = (1-β)·g + β·momentum;  plain: update = momentum.
        let update = if NESTEROV {
            let mut u = self.grads.clone();
            u.scale(1.0 - MOMENTUM);
            u.add_inplace_scaled(&self.momentum, MOMENTUM);
            u
        } else {
            self.momentum.clone()
        };

        // Orthogonalize, then rescale by max(1, rows/cols)^0.5 so the update has
        // a consistent magnitude for non-square matrices (the NS output has
        // singular values ≈ 1 regardless of shape).
        let ortho = newton_schulz5(&update);
        let scale = (self.grads.rows() as f32 / self.grads.cols() as f32)
            .max(1.0)
            .sqrt();

        // w ← w·(1 - lr·λ) − lr·scale·ortho
        if WEIGHT_DECAY != 0.0 {
            weights.scale(1.0 - lr * WEIGHT_DECAY);
        }
        weights.add_inplace_scaled(&ortho, -lr * scale);
    }

    fn clear(&mut self) {
        self.grads.clear();
    }

    fn clip(&mut self) {
        self.grads.clip(-CLIP, CLIP);
    }

    fn matrix(&mut self) -> &mut Matrix {
        &mut self.grads
    }
}

impl OptimizerGradTypes for Muon {
    type GradMatrix = MuonGradMatrix;
    // Embeddings and 1D params stay on Adam (no orthogonalization).
    type GradMatrixNoDecay = AdamGradMatrix;
    type GradVec = AdamGradVec;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Checks the two properties NS5 actually delivers, on the wide orientation
    /// (rows ≤ cols) where X·Xᵀ should approximate the identity:
    ///   • diagonal (= squared singular values) sits in a band around 1 — the
    ///     iteration inflates the small input singular values up toward 1 without
    ///     blowing up,
    ///   • off-diagonals are small — the rows are nearly orthogonal.
    /// Tolerances reflect what 5 fixed steps reach, not exact orthogonality.
    fn assert_approximately_orthogonal(o: &Matrix) {
        let wide = if o.rows() > o.cols() {
            o.transpose()
        } else {
            o.clone()
        };
        let g = wide.mul(&wide.transpose()); // rows × rows ≈ I
        let m = wide.rows();
        for i in 0..m {
            let sv2 = g[(i, i)];
            assert!(sv2 > 0.5 && sv2 < 1.6, "σ²[{i}] = {sv2} not in (0.5, 1.6)");
            for j in 0..m {
                if i != j {
                    let off = g[(i, j)];
                    assert!(
                        off.abs() < 0.35,
                        "off-diagonal [{i}][{j}] = {off} too large"
                    );
                }
            }
        }
    }

    /// Well-conditioned deterministic fill (xorshift-style integer hash → [-1, 1]),
    /// so the input is full-rank and the singular values are all sizeable — the
    /// regime Newton-Schulz is designed for.
    fn filled(rows: usize, cols: usize, seed: u32) -> Matrix {
        let mut g = Matrix::zeros(rows, cols);
        for (i, v) in g.iter_mut().enumerate() {
            let mut x = (i as u32)
                .wrapping_add(seed)
                .wrapping_mul(2654435761)
                .wrapping_add(1);
            x ^= x >> 15;
            x = x.wrapping_mul(2246822519);
            x ^= x >> 13;
            *v = (x as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }
        g
    }

    #[test]
    fn newton_schulz_square_is_orthogonal() {
        let (rows, cols) = (8, 8);
        let g = filled(rows, cols, 1);
        let o = newton_schulz5(&g);
        assert_eq!((o.rows(), o.cols()), (rows, cols));
        assert_approximately_orthogonal(&o);
    }

    #[test]
    fn newton_schulz_tall_keeps_shape_and_orthogonalizes() {
        // Tall input (rows > cols): exercises the internal transpose path.
        let (rows, cols) = (12, 4);
        let g = filled(rows, cols, 7);
        let o = newton_schulz5(&g);
        assert_eq!((o.rows(), o.cols()), (rows, cols));
        assert_approximately_orthogonal(&o);
    }
}
