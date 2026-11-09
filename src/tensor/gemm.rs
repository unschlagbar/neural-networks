//! Hand-written single-precision GEMM.
//!
//! Three transpose forms cover a Linear layer's forward and backward:
//!   - `NN`  C = A · B        forward           Y   = X · W
//!   - `NT`  C = A · Bᵀ       input gradient    dX  = dY · Wᵀ
//!   - `TN`  C = Aᵀ · B       weight gradient    dW  = Xᵀ · dY
//!
//! Each form is written with the loop order that keeps the innermost loop
//! contiguous (stride-1) over the output, so the compiler auto-vectorizes it.
//! Correctness first; cache tiling and explicit SIMD are a later pass — this is
//! the module we will tune when we start the performance comparison.
//!
//! `beta = 0.0` overwrites C, `beta = 1.0` accumulates into C (used to sum
//! gradients across a batch of windows without a separate add pass).
//!
//! A product of at least `PAR_MIN` multiply-adds runs on the rayon pool: split
//! over C's rows, or, when there are too few rows to go round, over K into
//! partial products that are summed afterwards.

use rayon::prelude::*;

use super::Tensor;

/// Smallest product (`m·k·n`) worth splitting across threads.
const PAR_MIN: usize = 1 << 18;

/// Rows of C per task, or `None` when the product stays on the calling thread.
fn row_split(m: usize, k: usize, n: usize) -> Option<usize> {
    let threads = rayon::current_num_threads();
    if threads < 2 || m * k * n < PAR_MIN || m < 2 * MR * threads {
        return None;
    }
    Some(m.div_ceil(threads).next_multiple_of(MR))
}

/// C(M×N) = A·B + beta·C, all row-major. A is M×K, B is K×N.
pub fn gemm_nn(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32], beta: f32) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    let threads = rayon::current_num_threads();
    if let Some(rows) = row_split(m, k, n) {
        c.par_chunks_mut(rows * n)
            .zip(a.par_chunks(rows * k))
            .for_each(|(cc, ac)| nn_serial(cc.len() / n, k, n, ac, b, cc, beta));
    } else if threads > 1 && m * k * n >= PAR_MIN && k >= 2 * threads {
        // Few rows: each task takes a slab of K (contiguous rows of B) into its own
        // partial C, so B is still read once and in order.
        let per = k.div_ceil(threads);
        let partial = (0..k.div_ceil(per))
            .into_par_iter()
            .map(|s| {
                let (p0, p1) = (s * per, ((s + 1) * per).min(k));
                let mut acc = vec![0.0; m * n];
                let mut a_slab = Vec::with_capacity(m * (p1 - p0));
                for i in 0..m {
                    a_slab.extend_from_slice(&a[i * k + p0..i * k + p1]);
                }
                nn_serial(m, p1 - p0, n, &a_slab, &b[p0 * n..p1 * n], &mut acc, 0.0);
                acc
            })
            .reduce_with(|mut x, y| {
                x.iter_mut().zip(&y).for_each(|(a, b)| *a += b);
                x
            })
            .unwrap();
        scale_beta(c, beta);
        c.iter_mut().zip(&partial).for_each(|(c, p)| *c += p);
    } else {
        nn_serial(m, k, n, a, b, c, beta);
    }
}

/// C(M×N) = A·Bᵀ + beta·C. A is M×K, B is N×K (i.e. Bᵀ is K×N).
pub fn gemm_nt(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32], beta: f32) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), n * k);
    debug_assert_eq!(c.len(), m * n);
    match row_split(m, k, n) {
        Some(rows) => c
            .par_chunks_mut(rows * n)
            .zip(a.par_chunks(rows * k))
            .for_each(|(cc, ac)| nt_serial(cc.len() / n, k, n, ac, b, cc, beta)),
        None => nt_serial(m, k, n, a, b, c, beta),
    }
}

/// C(M×N) = Aᵀ·B + beta·C. A is K×M, B is K×N.
pub fn gemm_tn(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32], beta: f32) {
    debug_assert_eq!(a.len(), k * m);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    match row_split(m, k, n) {
        Some(rows) => c
            .par_chunks_mut(rows * n)
            .enumerate()
            .for_each(|(t, cc)| tn_rows(t * rows, cc.len() / n, m, k, n, a, b, cc, beta)),
        None => tn_rows(0, m, m, k, n, a, b, c, beta),
    }
}

/// Number of C rows a micro-kernel updates together. Each B row loaded from
/// memory is reused across `MR` output rows, cutting weight-matrix bandwidth by
/// `MR×`. The contiguous inner `n` loop auto-vectorizes cleanly (AVX2 FMA); a
/// pure-Rust register tile was tried and regressed — LLVM spilled the
/// accumulator array to the stack instead of holding it in vector registers.
const MR: usize = 4;

#[inline]
fn scale_beta(c: &mut [f32], beta: f32) {
    if beta == 0.0 {
        c.iter_mut().for_each(|x| *x = 0.0);
    } else if beta != 1.0 {
        c.iter_mut().for_each(|x| *x *= beta);
    }
}

/// Single-threaded `gemm_nn`.
///
/// M-blocked by `MR`: for each block of `MR` rows and each k, one row of B is
/// loaded once and fused into all `MR` accumulator rows, so the weight row is
/// reused `MR` times from L1 instead of being re-streamed per output row. The
/// `MR` independent FMA chains give the vectorizer ILP.
fn nn_serial(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32], beta: f32) {
    scale_beta(c, beta);

    let mut i = 0;
    while i + MR <= m {
        // Disjoint mutable slices for the MR accumulator rows, held across the k loop.
        let (r0, rest) = c[i * n..(i + MR) * n].split_at_mut(n);
        let (r1, rest) = rest.split_at_mut(n);
        let (r2, r3) = rest.split_at_mut(n);
        for p in 0..k {
            let brow = &b[p * n..p * n + n];
            let a0 = a[i * k + p];
            let a1 = a[(i + 1) * k + p];
            let a2 = a[(i + 2) * k + p];
            let a3 = a[(i + 3) * k + p];
            for j in 0..n {
                let bj = brow[j];
                r0[j] += a0 * bj;
                r1[j] += a1 * bj;
                r2[j] += a2 * bj;
                r3[j] += a3 * bj;
            }
        }
        i += MR;
    }
    // Remainder rows (M not a multiple of MR).
    while i < m {
        let crow = &mut c[i * n..i * n + n];
        for p in 0..k {
            let aik = a[i * k + p];
            let brow = &b[p * n..p * n + n];
            for j in 0..n {
                crow[j] += aik * brow[j];
            }
        }
        i += 1;
    }
}

/// Single-threaded `gemm_nt`.
///
/// C[m,n] = Σ_k A[m,k]·B[n,k]: both operands are contiguous over k, so the
/// inner loop is a dot product.
fn nt_serial(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32], beta: f32) {
    // M-blocked by MR: each B row (length k) is loaded once and dotted against
    // MR different A rows, so the weight row is reused MR times from cache
    // instead of being re-streamed per output row.
    let mut i = 0;
    while i + MR <= m {
        let a0 = &a[i * k..i * k + k];
        let a1 = &a[(i + 1) * k..(i + 1) * k + k];
        let a2 = &a[(i + 2) * k..(i + 2) * k + k];
        let a3 = &a[(i + 3) * k..(i + 3) * k + k];
        for j in 0..n {
            let brow = &b[j * k..j * k + k];
            let (mut s0, mut s1, mut s2, mut s3) = (0.0, 0.0, 0.0, 0.0);
            for p in 0..k {
                let bp = brow[p];
                s0 += a0[p] * bp;
                s1 += a1[p] * bp;
                s2 += a2[p] * bp;
                s3 += a3[p] * bp;
            }
            for (r, s) in [s0, s1, s2, s3].into_iter().enumerate() {
                let idx = (i + r) * n + j;
                c[idx] = if beta == 0.0 { s } else { beta * c[idx] + s };
            }
        }
        i += MR;
    }
    while i < m {
        let arow = &a[i * k..i * k + k];
        for j in 0..n {
            let brow = &b[j * k..j * k + k];
            let mut acc = 0.0;
            for p in 0..k {
                acc += arow[p] * brow[p];
            }
            let idx = i * n + j;
            c[idx] = if beta == 0.0 {
                acc
            } else {
                beta * c[idx] + acc
            };
        }
        i += 1;
    }
}

/// Rows `i0..i0 + rows` of `gemm_tn`, written to `c` (those rows only).
///
/// C[m,n] = Σ_k A[k,m]·B[k,n]: loop k→m→n keeps the inner `n` loop contiguous
/// over a row of B and of C, with A[k,m] held as a scalar.
#[allow(clippy::too_many_arguments)]
fn tn_rows(
    i0: usize,
    rows: usize,
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    beta: f32,
) {
    scale_beta(c, beta);

    // M-blocked by MR (same reuse trick as gemm_nn): each B row is fused into
    // MR accumulator rows. A[k,m] is column-strided here (stride M), but that's
    // only MR scalar loads per k — the contiguous inner `n` loop dominates.
    let mut i = 0;
    while i + MR <= rows {
        let (r0, rest) = c[i * n..(i + MR) * n].split_at_mut(n);
        let (r1, rest) = rest.split_at_mut(n);
        let (r2, r3) = rest.split_at_mut(n);
        for p in 0..k {
            let brow = &b[p * n..p * n + n];
            let arow = &a[p * m + i0 + i..p * m + i0 + i + MR];
            let (a0, a1, a2, a3) = (arow[0], arow[1], arow[2], arow[3]);
            for j in 0..n {
                let bj = brow[j];
                r0[j] += a0 * bj;
                r1[j] += a1 * bj;
                r2[j] += a2 * bj;
                r3[j] += a3 * bj;
            }
        }
        i += MR;
    }
    while i < rows {
        let crow = &mut c[i * n..i * n + n];
        for p in 0..k {
            let akm = a[p * m + i0 + i];
            let brow = &b[p * n..p * n + n];
            for j in 0..n {
                crow[j] += akm * brow[j];
            }
        }
        i += 1;
    }
}

/// Transpose a row-major `rows×cols` matrix into `dst` (`cols×rows`).
/// Blocked to keep the strided writes cache-friendly.
pub fn transpose(rows: usize, cols: usize, src: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), rows * cols);
    debug_assert_eq!(dst.len(), rows * cols);
    const TILE: usize = 32;
    let mut i0 = 0;
    while i0 < rows {
        let i1 = (i0 + TILE).min(rows);
        let mut j0 = 0;
        while j0 < cols {
            let j1 = (j0 + TILE).min(cols);
            for i in i0..i1 {
                for j in j0..j1 {
                    dst[j * rows + i] = src[i * cols + j];
                }
            }
            j0 += TILE;
        }
        i0 += TILE;
    }
}

// Tensor-level convenience wrappers (fresh allocation)

/// Y = A · B for 2D tensors A(M×K), B(K×N).
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let (m, ka) = (a.rows(), a.cols());
    let (kb, n) = (b.rows(), b.cols());
    assert_eq!(ka, kb, "matmul: inner dims {ka} != {kb}");
    let mut out = Tensor::zeros(&[m, n]);
    gemm_nn(m, ka, n, &a.data, &b.data, &mut out.data, 0.0);
    out
}

/// Y = A · Bᵀ for 2D tensors A(M×K), B(N×K).
pub fn matmul_nt(a: &Tensor, b: &Tensor) -> Tensor {
    let (m, ka) = (a.rows(), a.cols());
    let (n, kb) = (b.rows(), b.cols());
    assert_eq!(ka, kb, "matmul_nt: inner dims {ka} != {kb}");
    let mut out = Tensor::zeros(&[m, n]);
    gemm_nt(m, ka, n, &a.data, &b.data, &mut out.data, 0.0);
    out
}

/// Y = Aᵀ · B for 2D tensors A(K×M), B(K×N).
pub fn matmul_tn(a: &Tensor, b: &Tensor) -> Tensor {
    let (ka, m) = (a.rows(), a.cols());
    let (kb, n) = (b.rows(), b.cols());
    assert_eq!(ka, kb, "matmul_tn: outer dims {ka} != {kb}");
    let mut out = Tensor::zeros(&[m, n]);
    gemm_tn(m, ka, n, &a.data, &b.data, &mut out.data, 0.0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Triple-nested reference multiply with explicit transpose flags.
    fn naive(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], at: bool, bt: bool) -> Vec<f32> {
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for p in 0..k {
                    let av = if at { a[p * m + i] } else { a[i * k + p] };
                    let bv = if bt { b[j * k + p] } else { b[p * n + j] };
                    acc += av * bv;
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    fn assert_close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-4, "{x} vs {y}");
        }
    }

    #[test]
    fn nn_matches_naive() {
        let (m, k, n) = (3, 4, 5);
        let a = Tensor::random(&[m, k], 1.0);
        let b = Tensor::random(&[k, n], 1.0);
        let got = matmul(&a, &b);
        assert_close(&got.data, &naive(m, k, n, &a.data, &b.data, false, false));
    }

    #[test]
    fn nt_matches_naive() {
        // A(M×K), B(N×K) -> A·Bᵀ (M×N)
        let (m, k, n) = (3, 4, 5);
        let a = Tensor::random(&[m, k], 1.0);
        let b = Tensor::random(&[n, k], 1.0);
        let got = matmul_nt(&a, &b);
        assert_close(&got.data, &naive(m, k, n, &a.data, &b.data, false, true));
    }

    #[test]
    fn tn_matches_naive() {
        // A(K×M), B(K×N) -> Aᵀ·B (M×N)
        let (m, k, n) = (3, 4, 5);
        let a = Tensor::random(&[k, m], 1.0);
        let b = Tensor::random(&[k, n], 1.0);
        let got = matmul_tn(&a, &b);
        assert_close(&got.data, &naive(m, k, n, &a.data, &b.data, true, false));
    }

    #[test]
    fn nn_blocked_and_remainder() {
        // M = 10 = 2·MR + 2: exercises both the MR-blocked body and the tail.
        let (m, k, n) = (10, 7, 6);
        let a = Tensor::random(&[m, k], 1.0);
        let b = Tensor::random(&[k, n], 1.0);
        assert_close(
            &matmul(&a, &b).data,
            &naive(m, k, n, &a.data, &b.data, false, false),
        );
    }

    #[test]
    fn nn_full_register_tiles() {
        // M = 19 (4·MR + tail), N = 43 (2·NR + tail): hits full MR×NR tiles plus
        // both remainder-column and remainder-row paths.
        let (m, k, n) = (19, 11, 43);
        let a = Tensor::random(&[m, k], 1.0);
        let b = Tensor::random(&[k, n], 1.0);
        assert_close(
            &matmul(&a, &b).data,
            &naive(m, k, n, &a.data, &b.data, false, false),
        );
    }

    #[test]
    fn tn_blocked_and_remainder() {
        let (m, k, n) = (10, 7, 6); // A is K×M
        let a = Tensor::random(&[k, m], 1.0);
        let b = Tensor::random(&[k, n], 1.0);
        assert_close(
            &matmul_tn(&a, &b).data,
            &naive(m, k, n, &a.data, &b.data, true, false),
        );
    }

    /// Large enough to take both parallel paths: the row split and, with two
    /// rows, the K split.
    #[test]
    fn parallel_paths_match_naive() {
        for (m, k, n) in [(203, 61, 47), (2, 4099, 97)] {
            let a = Tensor::random(&[m, k], 1.0);
            let b = Tensor::random(&[k, n], 1.0);
            let want = naive(m, k, n, &a.data, &b.data, false, false);
            let close = |got: &[f32]| {
                for (x, y) in got.iter().zip(&want) {
                    assert!((x - y).abs() < 1e-3 * y.abs().max(1.0), "{x} vs {y}");
                }
            };
            let mut c = vec![1.0; m * n];
            gemm_nn(m, k, n, &a.data, &b.data, &mut c, 0.0);
            close(&c);
            let bt = Tensor::random(&[n, k], 1.0);
            let mut c = vec![0.0; m * n];
            gemm_nt(m, k, n, &a.data, &bt.data, &mut c, 0.0);
            let want_nt = naive(m, k, n, &a.data, &bt.data, false, true);
            for (x, y) in c.iter().zip(&want_nt) {
                assert!((x - y).abs() < 1e-3 * y.abs().max(1.0), "nt {x} vs {y}");
            }
            let at = Tensor::random(&[k, m], 1.0);
            let mut c = vec![0.0; m * n];
            gemm_tn(m, k, n, &at.data, &b.data, &mut c, 0.0);
            let want_tn = naive(m, k, n, &at.data, &b.data, true, false);
            for (x, y) in c.iter().zip(&want_tn) {
                assert!((x - y).abs() < 1e-3 * y.abs().max(1.0), "tn {x} vs {y}");
            }
        }
    }

    #[test]
    fn beta_accumulates() {
        let (m, k, n) = (2, 3, 2);
        let a = Tensor::random(&[m, k], 1.0);
        let b = Tensor::random(&[k, n], 1.0);
        let mut c = Tensor::zeros(&[m, n]);
        gemm_nn(m, k, n, &a.data, &b.data, &mut c.data, 0.0);
        gemm_nn(m, k, n, &a.data, &b.data, &mut c.data, 1.0); // add a second time
        let single = matmul(&a, &b);
        let doubled: Vec<f32> = single.data.iter().map(|x| 2.0 * x).collect();
        assert_close(&c.data, &doubled);
    }
}
