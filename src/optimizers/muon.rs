use std::cell::RefCell;

use iron_oxide::collections::Matrix;

use crate::optimizers::{
    GradMatrixOps, OptimizerGradTypes, WEIGHT_DECAY,
    adam::{AdamGradMatrix, AdamGradVec},
};

// Muon — MomentUm Orthogonalized by Newton-schulz (Keller Jordan, 2024).
// https://kellerjordan.github.io/posts/muon/
//
// For 2D hidden weight matrices only: the momentum-smoothed gradient is replaced
// by its nearest semi-orthogonal matrix (all singular values driven toward 1) via
// a fixed 5-step Newton-Schulz iteration, and that is used as the weight update.
// Unlike Adam/AdamW, this step is NOT elementwise — it is real matrix math
// (X·Xᵀ etc.), which is why it needs matrix-product scratch space (see below).
//
// Embeddings (GradMatrixNoDecay) and 1D params / biases (GradVec) are NOT
// orthogonalized — they fall back to Adam. This is exactly the "MuonWithAuxAdam"
// hybrid recommended by the author. Note: the same scheduled `lr` is handed to
// both the Muon matrices and the aux-Adam params here; Muon wants a markedly
// larger lr (~0.02) than Adam, so retune the schedule when switching to Muon.

const MOMENTUM: f32 = 0.95;
const NESTEROV: bool = true;
const NS_STEPS: usize = 5;

/// Per-element gradient clip, guarding against blow-ups before orthogonalization.
const CLIP: f32 = 10.0;

// Newton-Schulz quintic coefficients from the reference Muon implementation
// (Jordan 2024). The polynomial f(x) = a·x + b·x³ + c·x⁵ is tuned so that 5
// iterations push every singular value toward 1. It deliberately does NOT
// converge — it is optimised to behave well for exactly this fixed step count,
// with a steep slope at 0 to inflate small singular values fast.
const NS_A: f32 = 3.4445;
const NS_B: f32 = -4.7750;
const NS_C: f32 = 2.0315;

#[derive(Debug)]
pub struct Muon;

pub struct MuonGradMatrix {
    grads: Matrix,
    momentum: Matrix,
}

/// Newton-Schulz works on matrix products (A = X·Xᵀ, B·X), and products need a
/// place to write their results. Instead of allocating that per call or storing
/// it per weight matrix, ONE process-wide scratch is shared by every layer: it
/// grows to the largest matrix seen during the first optimizer step and is then
/// reused forever — the hot path never allocates again. In the wide orientation
/// (m ≤ n): `x`/`x2` hold m×n, `a`/`b` hold m×m.
struct NsScratch {
    x: Vec<f32>,
    x2: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
}

thread_local! {
    static SCRATCH: RefCell<NsScratch> = const {
        RefCell::new(NsScratch {
            x: Vec::new(),
            x2: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
        })
    };
}

/// Grow-only resize: allocates during the first optimizer step, no-op afterwards.
fn grow(v: &mut Vec<f32>, len: usize) {
    if v.len() < len {
        v.resize(len, 0.0);
    }
}

/// Dot product with 16 independent accumulator lanes so LLVM can vectorize the
/// reduction despite float non-associativity (two AVX2 FMA chains in flight).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0; 16];
    let mut ca = a.chunks_exact(16);
    let mut cb = b.chunks_exact(16);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for l in 0..16 {
            acc[l] += x[l] * y[l];
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        sum += x * y;
    }
    sum
}

/// Copy the upper triangle of the square m×m matrix `v` into its lower triangle.
fn mirror_upper(v: &mut [f32], m: usize) {
    for i in 1..m {
        for j in 0..i {
            v[i * m + j] = v[j * m + i];
        }
    }
}

/// The quintic Newton-Schulz iteration on a wide (m ≤ n) m×n matrix in `x`,
/// approximating its "zeroth power" U·Vᵀ (all singular values ≈ 1) without ever
/// computing the SVD. `x2` (m×n) and `a`/`b` (m×m) are scratch. The result ends
/// up in one of the two m×n buffers; the returned slice is it.
fn newton_schulz5_core<'a>(
    mut x: &'a mut [f32],
    mut x2: &'a mut [f32],
    a: &mut [f32],
    b: &mut [f32],
    m: usize,
    n: usize,
) -> &'a [f32] {
    // Normalize by the Frobenius norm so the largest singular value is ≤ 1, i.e.
    // start inside the basin where the iteration is well-behaved.
    let fro = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let inv = 1.0 / (fro + 1e-7);
    for v in x.iter_mut() {
        *v *= inv;
    }

    for _ in 0..NS_STEPS {
        // A = X·Xᵀ — dot products of X's rows (contiguous, no transpose
        // materialized). Symmetric: compute the upper triangle, mirror the rest.
        for i in 0..m {
            let xi = &x[i * n..i * n + n];
            for j in i..m {
                a[i * m + j] = dot(xi, &x[j * n..j * n + n]);
            }
        }
        mirror_upper(a, m);

        // B = b·A + c·A² + a·I, fused so the X update below is a single matmul
        // (a·X + (b·A + c·A²)·X = B·X). A is symmetric, so A² = A·Aᵀ — again row
        // dot products — and B is symmetric too: upper triangle + mirror.
        for i in 0..m {
            for j in i..m {
                b[i * m + j] =
                    NS_C * dot(&a[i * m..i * m + m], &a[j * m..j * m + m]) + NS_B * a[i * m + j];
            }
            b[i * m + i] += NS_A;
        }
        mirror_upper(b, m);

        // X ← B·X (axpy kernel: stream rows of X scaled by B's entries).
        for i in 0..m {
            let row = &mut x2[i * n..i * n + n];
            row.fill(0.0);
            for (k, &c) in b[i * m..i * m + m].iter().enumerate() {
                for (o, &v) in row.iter_mut().zip(&x[k * n..k * n + n]) {
                    *o += c * v;
                }
            }
        }
        std::mem::swap(&mut x, &mut x2);
    }

    x
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

        let rows = self.grads.rows();
        let cols = self.grads.cols();
        // Newton-Schulz runs on the wide orientation (rows ≤ cols) so A = X·Xᵀ
        // is the smaller square; tall matrices are transposed on the fly while
        // writing into / reading out of the scratch — never materialized.
        let transposed = rows > cols;
        let (m, n) = (rows.min(cols), rows.max(cols));

        SCRATCH.with(|scratch| {
            let s = &mut *scratch.borrow_mut();
            grow(&mut s.x, m * n);
            grow(&mut s.x2, m * n);
            grow(&mut s.a, m * m);
            grow(&mut s.b, m * m);

            // momentum ← β·momentum + (1-β)·g and the update it feeds NS
            // (Nesterov: u = (1-β)·g + β·momentum; plain: u = momentum), fused
            // into one pass writing straight into the NS working buffer.
            {
                let g = self.grads.as_slice();
                let mom = self.momentum.as_slice_mut();
                let x = &mut s.x[..m * n];
                for r in 0..rows {
                    for c in 0..cols {
                        let i = r * cols + c;
                        mom[i] = MOMENTUM * mom[i] + (1.0 - MOMENTUM) * g[i];
                        let u = if NESTEROV {
                            (1.0 - MOMENTUM) * g[i] + MOMENTUM * mom[i]
                        } else {
                            mom[i]
                        };
                        x[if transposed { c * rows + r } else { i }] = u;
                    }
                }
            }

            let ortho = newton_schulz5_core(
                &mut s.x[..m * n],
                &mut s.x2[..m * n],
                &mut s.a[..m * m],
                &mut s.b[..m * m],
                m,
                n,
            );

            // Rescale by max(1, rows/cols)^0.5 so the update has a consistent
            // magnitude for non-square matrices (the NS output has singular
            // values ≈ 1 regardless of shape), applied together with decoupled
            // weight decay in one pass:  w ← w·(1 - lr·λ) − lr·scale·ortho.
            let scale = (rows as f32 / cols as f32).max(1.0).sqrt();
            let decay = 1.0 - lr * WEIGHT_DECAY;
            let step = -lr * scale;
            let w = weights.as_slice_mut();
            if transposed {
                for r in 0..rows {
                    for c in 0..cols {
                        w[r * cols + c] = w[r * cols + c] * decay + step * ortho[c * rows + r];
                    }
                }
            } else {
                for (w, &o) in w.iter_mut().zip(ortho) {
                    *w = *w * decay + step * o;
                }
            }
        });
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

    /// Test-only wrapper: run Newton-Schulz on a Matrix, handling orientation.
    fn newton_schulz5(g: &Matrix) -> Matrix {
        let transposed = g.rows() > g.cols();
        let wide = if transposed { g.transpose() } else { g.clone() };
        let (m, n) = (wide.rows(), wide.cols());

        let mut x = wide.as_slice().to_vec();
        let mut x2 = vec![0.0; m * n];
        let mut a = vec![0.0; m * m];
        let mut b = vec![0.0; m * m];
        let res = newton_schulz5_core(&mut x, &mut x2, &mut a, &mut b, m, n);

        let out = Matrix::from_box(res.to_vec().into_boxed_slice(), m, n);
        if transposed { out.transpose() } else { out }
    }

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

    /// The zero-allocation `apply_to` must produce the same update as the
    /// reference formulation (momentum, NS, scale and decay done step by step).
    #[test]
    fn apply_to_matches_reference() {
        for &(rows, cols) in &[(6, 10), (10, 6), (8, 8)] {
            let lr = 0.01;
            let g = filled(rows, cols, 3);

            let mut gm = MuonGradMatrix::zeros(rows, cols);
            gm.matrix().as_slice_mut().copy_from_slice(g.as_slice());
            let mut w = filled(rows, cols, 11);
            let mut w_ref = w.clone();

            gm.apply_to(&mut w, lr);

            // Reference: first step ⇒ momentum = (1-β)·g, Nesterov update
            // u = (1-β)·g + β·momentum = (1-β)(1+β)·g.
            let mut u = g.clone();
            u.scale((1.0 - MOMENTUM) * (1.0 + MOMENTUM));
            let ortho = newton_schulz5(&u);
            let scale = (rows as f32 / cols as f32).max(1.0).sqrt();
            w_ref.scale(1.0 - lr * WEIGHT_DECAY);
            w_ref.add_inplace_scaled(&ortho, -lr * scale);

            for (a, b) in w.as_slice().iter().zip(w_ref.as_slice()) {
                assert!((a - b).abs() < 1e-5, "{a} vs {b} ({rows}x{cols})");
            }
        }
    }
}
