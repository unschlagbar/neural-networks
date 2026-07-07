use iron_oxide::collections::Matrix;

pub mod activations;
pub mod causal_conv1d;
pub mod dropout;
pub mod embedding;
pub mod headwise_rms_norm;
pub mod linear;
pub mod linear_nb;
pub mod lstm;
pub mod mlstm;
pub mod mlstm_block;
pub mod rms_norm;
pub mod silu_dense;
pub mod slstm;
pub mod slstm_block;
pub mod soft_cap;
pub mod softmax;

pub fn add_vec_in_place(x: &mut [f32], y: &[f32]) {
    debug_assert_eq!(x.len(), y.len());
    x.iter_mut().zip(y).for_each(|(x, y)| *x += y);
}

/// Dot product with 8 independent accumulator lanes.
///
/// A plain `s += a[j] * b[j]` loop is a serial dependency chain that LLVM may
/// not vectorize (f32 addition is not associative), leaving the loop scalar.
/// The lane accumulators reassociate the sum explicitly, so the chunk loop
/// compiles to packed FMA. The result differs from the serial sum only in the
/// last ulps.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; 8];
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let mut tail = 0.0;
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        tail += x * y;
    }
    for (x8, y8) in ca.zip(cb) {
        for l in 0..8 {
            // mul_add compiles to a packed FMA under target-cpu=native;
            // rustc never contracts `a * b + c` on its own.
            acc[l] = x8[l].mul_add(y8[l], acc[l]);
        }
    }
    tail + ((acc[0] + acc[4]) + (acc[2] + acc[6])) + ((acc[1] + acc[5]) + (acc[3] + acc[7]))
}

/// out += Wᵀx, four input rows per sweep. Processing 4 rows per pass over
/// `out` quarters the out load/store traffic and gives the backend four
/// independent FMA streams; the zips are exact-length so no bounds checks.
#[inline]
pub fn matvec_acc(w: &Matrix, x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(w.rows(), x.len());
    debug_assert_eq!(w.cols(), out.len());
    let mut rows = x.chunks_exact(4);
    let mut i = 0;
    for x4 in rows.by_ref() {
        let (r0, r1, r2, r3) = (&w[i], &w[i + 1], &w[i + 2], &w[i + 3]);
        for ((((o, &a), &b), &c), &e) in out.iter_mut().zip(r0).zip(r1).zip(r2).zip(r3) {
            // mul_add chains compile to packed FMAs under target-cpu=native.
            *o = x4[0].mul_add(a, x4[1].mul_add(b, x4[2].mul_add(c, x4[3].mul_add(e, *o))));
        }
        i += 4;
    }
    for &xi in rows.remainder() {
        for (o, &wv) in out.iter_mut().zip(&w[i]) {
            *o = xi.mul_add(wv, *o);
        }
        i += 1;
    }
}

/// out = Wᵀx (see `matvec_acc`).
#[inline]
pub fn matvec_into(w: &Matrix, x: &[f32], out: &mut [f32]) {
    out.fill(0.0);
    matvec_acc(w, x, out);
}

/// How many backward steps of (xh, gate-delta) outer products are buffered
/// before being folded into the grad matrices. Amortizes the grad-matrix
/// read-modify-write traffic 1/GRAD_BLOCK; any leftover partial block is
/// flushed at the window-end hooks (accumulate_init_grad / reset_bptt_state)
/// and defensively before grads are read (add_grads / apply_grads).
pub const GRAD_BLOCK: usize = 16;

/// gw += Σ_b xh_b ⊗ d_b over the n pending steps. One gate at a time: the
/// pending delta rows (≤ GRAD_BLOCK·h floats) stay cache-resident while gw is
/// streamed exactly once. Four steps per sweep — same shape as `matvec_acc`:
/// four independent FMA streams, exact-length zips, no bounds checks.
pub fn outer_acc_block(gw: &mut [f32], xh: &[f32], d: &[f32], n: usize, r: usize, h: usize) {
    for i in 0..r {
        let gw_r = &mut gw[i * h..i * h + h];
        let mut b = 0;
        while b + 4 <= n {
            let x0 = xh[b * r + i];
            let x1 = xh[(b + 1) * r + i];
            let x2 = xh[(b + 2) * r + i];
            let x3 = xh[(b + 3) * r + i];
            let d0 = &d[b * h..(b + 1) * h];
            let d1 = &d[(b + 1) * h..(b + 2) * h];
            let d2 = &d[(b + 2) * h..(b + 3) * h];
            let d3 = &d[(b + 3) * h..(b + 4) * h];
            for ((((g, &a), &bb), &c), &e) in gw_r.iter_mut().zip(d0).zip(d1).zip(d2).zip(d3) {
                *g = x0.mul_add(a, x1.mul_add(bb, x2.mul_add(c, x3.mul_add(e, *g))));
            }
            b += 4;
        }
        while b < n {
            let xi = xh[b * r + i];
            let db = &d[b * h..(b + 1) * h];
            for (g, &v) in gw_r.iter_mut().zip(db) {
                *g = xi.mul_add(v, *g);
            }
            b += 1;
        }
    }
}

pub fn sub_in_place(a: &mut Matrix, b: &Matrix, lr: f32) {
    debug_assert_eq!(a.rows(), b.rows());
    debug_assert_eq!(a.cols(), b.cols());
    a.as_slice_mut()
        .iter_mut()
        .zip(b.as_slice())
        .for_each(|(a, b)| *a -= lr * b);
}

pub fn sub_vec_in_place(a: &mut [f32], b: &[f32], lr: f32) {
    a.iter_mut().zip(b).for_each(|(a, b)| *a -= lr * b);
}

pub fn one_hot(index: usize, size: usize) -> Vec<f32> {
    let mut out = vec![0.0; size];
    out[index] = 1.0;
    out
}
