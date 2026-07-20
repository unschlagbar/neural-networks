//! Backend op seam for the `nn2` layers.
//!
//! Every non-GEMM piece of elementwise / reduction / gather math that a layer
//! used to write as an inline `.data[i]` loop lives here instead, as a named
//! free function operating on `Tensor`s (never on raw host `&[f32]` — a `Tensor`
//! will later be able to carry a *device* buffer, and the whole point of this
//! seam is that no layer touches `.data` for math directly). Each function here
//! maps one-to-one onto a future CUDA kernel: to add the GPU backend you
//! implement these bodies against device buffers, one kernel at a time, and the
//! layer code above them does not change.
//!
//! GEMM (`tensor::gemm`), the fused softmax/cross-entropy (`nn2::loss`) and the
//! AdamW update (`nn2::optim`) are already single-choke-point free functions, so
//! they are already "kernels" in this sense and are not duplicated here.
//!
//! The CPU implementations below are behaviour-preserving copies of the loops
//! they replaced — same arithmetic, same order — so the layers' finite-
//! difference tests keep passing unchanged.

use crate::tensor::Tensor;

// Elementwise

/// SoftCap forward: `y = cap · tanh(x / cap)`, element-wise. Kernel: map.
pub fn softcap_forward(x: &Tensor, cap: f32) -> Tensor {
    let mut y = Tensor::zeros(x.dims());
    for (o, &v) in y.data.iter_mut().zip(&x.data) {
        *o = cap * (v / cap).tanh();
    }
    y
}

/// SoftCap backward: `dx = dy · (1 − (y/cap)²)`, using the saved output `y`.
/// Kernel: map over three aligned buffers.
pub fn softcap_backward(dy: &Tensor, y: &Tensor, cap: f32) -> Tensor {
    let mut dx = Tensor::zeros(dy.dims());
    for ((d, &g), &yv) in dx.data.iter_mut().zip(&dy.data).zip(&y.data) {
        let t = yv / cap;
        *d = g * (1.0 - t * t);
    }
    dx
}

// Linear helpers (the matmuls themselves stay in tensor::gemm)

/// Copy `bias` (`[N]`) into every row of `out` (`[B, N]`). Used to seed a
/// Linear's output before the `beta = 1` matmul accumulates on top. Kernel:
/// row broadcast.
pub fn broadcast_row(out: &mut Tensor, bias: &Tensor) {
    let n = bias.len();
    for row in out.data.chunks_mut(n) {
        row.copy_from_slice(&bias.data);
    }
}

/// Accumulate the column sum of `dy` (`[B, N]`) into `db` (`[N]`):
/// `db[o] += Σ_batch dy[·, o]`. This is the bias gradient. Kernel: column
/// reduction over the batch axis.
pub fn add_col_sum(db: &mut Tensor, dy: &Tensor) {
    let n = db.len();
    for row in dy.data.chunks(n) {
        for (o, v) in row.iter().enumerate() {
            db.data[o] += *v;
        }
    }
}

// Embedding gather / scatter

/// Gather rows of `table` (`[vocab, dim]`) by `ids` into a `[B, dim]` tensor.
/// Kernel: indexed row copy.
pub fn embedding_gather(table: &Tensor, ids: &[usize], dim: usize) -> Tensor {
    let mut out = Tensor::zeros(&[ids.len(), dim]);
    for (r, &id) in ids.iter().enumerate() {
        debug_assert!(id < table.rows(), "embedding id {id} out of range");
        out.data[r * dim..(r + 1) * dim].copy_from_slice(&table.data[id * dim..(id + 1) * dim]);
    }
    out
}

/// Scatter-add: `dtable[ids[r]] += dy[r]`. Ids may repeat, so contributions
/// sum. Kernel: indexed row scatter-add (atomicAdd on GPU when ids collide).
pub fn embedding_scatter_add(dtable: &mut Tensor, ids: &[usize], dy: &Tensor, dim: usize) {
    for (r, &id) in ids.iter().enumerate() {
        let src = &dy.data[r * dim..(r + 1) * dim];
        let dst = &mut dtable.data[id * dim..(id + 1) * dim];
        for (a, b) in dst.iter_mut().zip(src) {
            *a += *b;
        }
    }
}

// (Head-wise) RMSNorm

/// Saved intermediates from an RMSNorm forward, consumed by its backward.
pub struct RmsForward {
    /// Normalized-and-scaled output `[B, F]`.
    pub out: Tensor,
    /// Normalized (pre-γ) activations `x̂` `[B, F]`, saved for backward.
    pub x_hat: Tensor,
    /// Per-(row, group) `1/rms`, length `B · (F/group)`.
    pub inv_rms: Vec<f32>,
}

/// Grouped RMSNorm forward. Each row of `x` (`[B, F]`) is split into
/// `F / group` contiguous groups of size `group`; RMS is computed per group.
/// Plain RMSNorm is the `group == F` case; head-wise RMSNorm uses `group = dhv`
/// (the per-head width). `gamma` is a per-column (`[F]`) learnable scale applied
/// after normalization. Kernel: per-(row, group) reduction + scale.
pub fn rms_norm_forward(x: &Tensor, gamma: &Tensor, group: usize, eps: f32) -> RmsForward {
    let (b, f) = (x.rows(), x.cols());
    debug_assert_eq!(f % group, 0, "rms group {group} does not divide width {f}");
    debug_assert_eq!(gamma.len(), f, "rms gamma len != width");
    let groups_per_row = f / group;

    let mut out = Tensor::zeros(&[b, f]);
    let mut x_hat = Tensor::zeros(&[b, f]);
    let mut inv_rms = vec![0.0; b * groups_per_row];

    for r in 0..b {
        for g in 0..groups_per_row {
            let off = r * f + g * group;
            let g_off = g * group; // column offset for gamma
            let xin = &x.data[off..off + group];
            let ss: f32 = xin.iter().map(|&v| v * v).sum();
            let inv = 1.0 / (ss / group as f32 + eps).sqrt();
            inv_rms[r * groups_per_row + g] = inv;
            for i in 0..group {
                let xh = xin[i] * inv;
                x_hat.data[off + i] = xh;
                out.data[off + i] = gamma.data[g_off + i] * xh;
            }
        }
    }
    RmsForward {
        out,
        x_hat,
        inv_rms,
    }
}

/// Grouped RMSNorm backward. Accumulates the γ gradient into `dgamma` and
/// returns `dX` `[B, F]`. Uses the `x_hat` / `inv_rms` saved by the forward.
///
/// Per group:  S    = Σⱼ γⱼ · dYⱼ · x̂ⱼ
///             dXᵢ  = inv_rms · ( γᵢ·dYᵢ − x̂ᵢ · S / group )
/// Kernel: per-(row, group) reduction; `dgamma` accumulate is an atomicAdd on
/// GPU (or a batched reduction).
pub fn rms_norm_backward(
    dy: &Tensor,
    x_hat: &Tensor,
    inv_rms: &[f32],
    gamma: &Tensor,
    dgamma: &mut Tensor,
    group: usize,
) -> Tensor {
    let (b, f) = (dy.rows(), dy.cols());
    let groups_per_row = f / group;
    debug_assert_eq!(
        inv_rms.len(),
        b * groups_per_row,
        "rms inv_rms len mismatch"
    );

    let mut dx = Tensor::zeros(&[b, f]);
    for r in 0..b {
        for g in 0..groups_per_row {
            let off = r * f + g * group;
            let g_off = g * group;
            let inv = inv_rms[r * groups_per_row + g];
            let mut s = 0.0;
            for i in 0..group {
                let dyxh = dy.data[off + i] * x_hat.data[off + i];
                dgamma.data[g_off + i] += dyxh;
                s += gamma.data[g_off + i] * dyxh;
            }
            let s_over_g = s / group as f32;
            for i in 0..group {
                dx.data[off + i] = inv
                    * (gamma.data[g_off + i] * dy.data[off + i] - x_hat.data[off + i] * s_over_g);
            }
        }
    }
    dx
}
