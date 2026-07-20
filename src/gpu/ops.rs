//! GPU implementations of the backend op seam, operating on [`DTensor`]s.
//!
//! These are the device counterparts of `src/nn2/ops.rs` (+ `tensor::gemm`).
//! Each is checked against its CPU reference in the tests below, so the finite-
//! difference tests that already pin the CPU math transitively pin these too.
//!
//! GEMM goes through cuBLAS. cuBLAS is column-major while our tensors are
//! row-major, so we use the standard identity: a row-major `C = op(A)·op(B)` is
//! computed by asking cuBLAS for the column-major `Cᵀ`, which means swapping the
//! operands (pass B first, A second) and swapping `m`/`n`. Working this out per
//! transpose form is error-prone, hence the exhaustive parity tests.

use cudarc::cublas::{Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::cublas::sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use super::{DTensor, Gpu};
use crate::nn2::optim::AdamCfg;

/// Upload host indices (`usize`) as a `u32` device buffer for the gather /
/// scatter / CE kernels.
fn upload_ids(gpu: &Gpu, ids: &[usize]) -> CudaSlice<u32> {
    let u: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
    gpu.stream.memcpy_stod(&u).expect("upload ids")
}

/// `C = A · B + beta·C` for row-major `A(M×K)`, `B(K×N)`, writing into an
/// existing `C(M×N)`. `beta = 0` overwrites, `beta = 1` accumulates (bias-seeded
/// forward). Uses cuBLAS via the operand-swap trick: cuBLAS computes column-major
/// `Cᵀ(N×M) = Bᵀ·Aᵀ`, which is exactly our row-major `C` in memory.
pub fn matmul_nn_into(gpu: &Gpu, a: &DTensor, b: &DTensor, c: &mut DTensor, beta: f32) {
    let (m, ka) = (a.rows(), a.cols());
    let (kb, n) = (b.rows(), b.cols());
    assert_eq!(ka, kb, "matmul: inner dims {ka} != {kb}");
    assert_eq!((c.rows(), c.cols()), (m, n), "matmul: C shape mismatch");
    let cfg = GemmConfig {
        transa: CUBLAS_OP_N,
        transb: CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: ka as i32,
        alpha: 1.0,
        lda: n as i32,
        ldb: ka as i32,
        beta,
        ldc: n as i32,
    };
    unsafe { gpu.blas.gemm(cfg, &b.buf, &a.buf, &mut c.buf) }.expect("cublas gemm nn");
}

/// `C = A · Bᵀ + beta·C` for row-major `A(M×K)`, `B(N×K)` → `C(M×N)`. The
/// input-gradient form (`dX = dY · Wᵀ`).
pub fn matmul_nt_into(gpu: &Gpu, a: &DTensor, b: &DTensor, c: &mut DTensor, beta: f32) {
    let (m, ka) = (a.rows(), a.cols());
    let (n, kb) = (b.rows(), b.cols());
    assert_eq!(ka, kb, "matmul_nt: inner dims {ka} != {kb}");
    assert_eq!((c.rows(), c.cols()), (m, n), "matmul_nt: C shape mismatch");
    let cfg = GemmConfig {
        transa: CUBLAS_OP_T,
        transb: CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: ka as i32,
        alpha: 1.0,
        lda: ka as i32,
        ldb: ka as i32,
        beta,
        ldc: n as i32,
    };
    unsafe { gpu.blas.gemm(cfg, &b.buf, &a.buf, &mut c.buf) }.expect("cublas gemm nt");
}

/// `C = Aᵀ · B + beta·C` for row-major `A(K×M)`, `B(K×N)` → `C(M×N)`. The
/// weight-gradient form (`dW += Xᵀ · dY`, used with `beta = 1`).
pub fn matmul_tn_into(gpu: &Gpu, a: &DTensor, b: &DTensor, c: &mut DTensor, beta: f32) {
    let (ka, m) = (a.rows(), a.cols());
    let (kb, n) = (b.rows(), b.cols());
    assert_eq!(ka, kb, "matmul_tn: outer dims {ka} != {kb}");
    assert_eq!((c.rows(), c.cols()), (m, n), "matmul_tn: C shape mismatch");
    let cfg = GemmConfig {
        transa: CUBLAS_OP_N,
        transb: CUBLAS_OP_T,
        m: n as i32,
        n: m as i32,
        k: ka as i32,
        alpha: 1.0,
        lda: n as i32,
        ldb: m as i32,
        beta,
        ldc: n as i32,
    };
    unsafe { gpu.blas.gemm(cfg, &b.buf, &a.buf, &mut c.buf) }.expect("cublas gemm tn");
}

/// `C = A · B` (fresh allocation). Convenience wrapper over [`matmul_nn_into`].
pub fn matmul(gpu: &Gpu, a: &DTensor, b: &DTensor) -> DTensor {
    let mut c = DTensor::uninit(gpu, &[a.rows(), b.cols()]);
    matmul_nn_into(gpu, a, b, &mut c, 0.0);
    c
}

/// `C = A · Bᵀ` (fresh allocation). Convenience wrapper over [`matmul_nt_into`].
pub fn matmul_nt(gpu: &Gpu, a: &DTensor, b: &DTensor) -> DTensor {
    let mut c = DTensor::uninit(gpu, &[a.rows(), b.rows()]);
    matmul_nt_into(gpu, a, b, &mut c, 0.0);
    c
}

/// `C = Aᵀ · B` (fresh allocation). Convenience wrapper over [`matmul_tn_into`].
pub fn matmul_tn(gpu: &Gpu, a: &DTensor, b: &DTensor) -> DTensor {
    let mut c = DTensor::uninit(gpu, &[a.cols(), b.cols()]);
    matmul_tn_into(gpu, a, b, &mut c, 0.0);
    c
}

// ---------------------------------------------------------------------------
// Strided-batched GEMM (per-(batch,head) small matmuls for chunkwise mLSTM).
// Rank-3 device tensors `[batch, ·, ·]`, contiguous → per-batch stride is the
// matrix element count. Same row-major-over-column-major operand-swap as the
// single GEMMs (pass B first, A second, compute Cᵀ), applied per batch.
// ---------------------------------------------------------------------------

/// `C[g] = A[g] · B[g]` for `A[batch,M,K]`, `B[batch,K,N]` → `C[batch,M,N]`.
pub fn matmul_batched_nn(gpu: &Gpu, a: &DTensor, b: &DTensor) -> DTensor {
    let (batch, m, ka) = (a.shape[0], a.shape[1], a.shape[2]);
    let (kb, n) = (b.shape[1], b.shape[2]);
    assert_eq!(ka, kb, "matmul_batched_nn: inner dims {ka} != {kb}");
    assert_eq!(batch, b.shape[0], "matmul_batched_nn: batch mismatch");
    let mut c = DTensor::uninit(gpu, &[batch, m, n]);
    let gemm = GemmConfig {
        transa: CUBLAS_OP_N, transb: CUBLAS_OP_N,
        m: n as i32, n: m as i32, k: ka as i32,
        alpha: 1.0, lda: n as i32, ldb: ka as i32, beta: 0.0, ldc: n as i32,
    };
    let cfg = StridedBatchedConfig {
        gemm, batch_size: batch as i32,
        stride_a: (kb * n) as i64, stride_b: (m * ka) as i64, stride_c: (m * n) as i64,
    };
    unsafe { gpu.blas.gemm_strided_batched(cfg, &b.buf, &a.buf, &mut c.buf) }
        .expect("cublas gemm_strided_batched nn");
    c
}

/// `C[g] = A[g] · B[g]ᵀ` for `A[batch,M,K]`, `B[batch,N,K]` → `C[batch,M,N]`.
pub fn matmul_batched_nt(gpu: &Gpu, a: &DTensor, b: &DTensor) -> DTensor {
    let (batch, m, ka) = (a.shape[0], a.shape[1], a.shape[2]);
    let (n, kb) = (b.shape[1], b.shape[2]);
    assert_eq!(ka, kb, "matmul_batched_nt: inner dims {ka} != {kb}");
    assert_eq!(batch, b.shape[0], "matmul_batched_nt: batch mismatch");
    let mut c = DTensor::uninit(gpu, &[batch, m, n]);
    let gemm = GemmConfig {
        transa: CUBLAS_OP_T, transb: CUBLAS_OP_N,
        m: n as i32, n: m as i32, k: ka as i32,
        alpha: 1.0, lda: ka as i32, ldb: ka as i32, beta: 0.0, ldc: n as i32,
    };
    let cfg = StridedBatchedConfig {
        gemm, batch_size: batch as i32,
        stride_a: (n * kb) as i64, stride_b: (m * ka) as i64, stride_c: (m * n) as i64,
    };
    unsafe { gpu.blas.gemm_strided_batched(cfg, &b.buf, &a.buf, &mut c.buf) }
        .expect("cublas gemm_strided_batched nt");
    c
}

/// `C[g] = A[g]ᵀ · B[g]` for `A[batch,K,M]`, `B[batch,K,N]` → `C[batch,M,N]`.
pub fn matmul_batched_tn(gpu: &Gpu, a: &DTensor, b: &DTensor) -> DTensor {
    let (batch, ka, m) = (a.shape[0], a.shape[1], a.shape[2]);
    let (kb, n) = (b.shape[1], b.shape[2]);
    assert_eq!(ka, kb, "matmul_batched_tn: outer dims {ka} != {kb}");
    assert_eq!(batch, b.shape[0], "matmul_batched_tn: batch mismatch");
    let mut c = DTensor::uninit(gpu, &[batch, m, n]);
    let gemm = GemmConfig {
        transa: CUBLAS_OP_N, transb: CUBLAS_OP_T,
        m: n as i32, n: m as i32, k: ka as i32,
        alpha: 1.0, lda: n as i32, ldb: m as i32, beta: 0.0, ldc: n as i32,
    };
    let cfg = StridedBatchedConfig {
        gemm, batch_size: batch as i32,
        stride_a: (kb * n) as i64, stride_b: (ka * m) as i64, stride_c: (m * n) as i64,
    };
    unsafe { gpu.blas.gemm_strided_batched(cfg, &b.buf, &a.buf, &mut c.buf) }
        .expect("cublas gemm_strided_batched tn");
    c
}

// ---------------------------------------------------------------------------
// Elementwise / reduction / gather (NVRTC kernels, see gpu/kernels.rs)
// ---------------------------------------------------------------------------

/// In-place scale: `x *= s`. The mLSTM k-projection's `1/√dqk`.
pub fn scale_(gpu: &Gpu, x: &mut DTensor, s: f32) {
    let n = x.len();
    let n_i = n as i32;
    let f = gpu.kernels.get("scale_inplace");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut x.buf).arg(&s).arg(&n_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(n as u32)) }.expect("scale_inplace");
}

/// In-place numerically-stable sigmoid. The mLSTM o-gate projection.
pub fn sigmoid_(gpu: &Gpu, x: &mut DTensor) {
    let n = x.len();
    let n_i = n as i32;
    let f = gpu.kernels.get("sigmoid_inplace");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut x.buf).arg(&n_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(n as u32)) }.expect("sigmoid_inplace");
}

/// SoftCap forward: `y = cap · tanh(x / cap)`.
pub fn softcap_forward(gpu: &Gpu, x: &DTensor, cap: f32) -> DTensor {
    let n = x.len();
    let n_i = n as i32;
    let mut y = DTensor::uninit(gpu, x.dims());
    let f = gpu.kernels.get("softcap_forward");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&x.buf).arg(&mut y.buf).arg(&cap).arg(&n_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(n as u32)) }.expect("softcap_forward");
    y
}

/// SoftCap backward: `dx = dy · (1 − (y/cap)²)`, using the saved output `y`.
pub fn softcap_backward(gpu: &Gpu, dy: &DTensor, y: &DTensor, cap: f32) -> DTensor {
    let n = dy.len();
    let n_i = n as i32;
    let mut dx = DTensor::uninit(gpu, dy.dims());
    let f = gpu.kernels.get("softcap_backward");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&dy.buf).arg(&y.buf).arg(&mut dx.buf).arg(&cap).arg(&n_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(n as u32)) }.expect("softcap_backward");
    dx
}

/// Copy `bias` (`[N]`) into every row of `out` (`[rows, N]`).
pub fn broadcast_row(gpu: &Gpu, out: &mut DTensor, bias: &DTensor) {
    let (rows, n) = (out.rows(), out.cols());
    let (rows_i, n_i) = (rows as i32, n as i32);
    let f = gpu.kernels.get("broadcast_row");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut out.buf).arg(&bias.buf).arg(&rows_i).arg(&n_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems((rows * n) as u32)) }.expect("broadcast_row");
}

/// Accumulate the column sum of `dy` (`[rows, N]`) into `db` (`[N]`) — the bias
/// gradient.
pub fn add_col_sum(gpu: &Gpu, db: &mut DTensor, dy: &DTensor) {
    let (rows, n) = (dy.rows(), dy.cols());
    let (rows_i, n_i) = (rows as i32, n as i32);
    let f = gpu.kernels.get("add_col_sum");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut db.buf).arg(&dy.buf).arg(&rows_i).arg(&n_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(n as u32)) }.expect("add_col_sum");
}

/// Gather rows of `table` (`[vocab, dim]`) by `ids` into a `[ids.len(), dim]`
/// tensor.
pub fn embedding_gather(gpu: &Gpu, table: &DTensor, ids: &[usize], dim: usize) -> DTensor {
    let rows = ids.len();
    let (dim_i, rows_i) = (dim as i32, rows as i32);
    let dids = upload_ids(gpu, ids);
    let mut out = DTensor::uninit(gpu, &[rows, dim]);
    let f = gpu.kernels.get("embedding_gather");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&table.buf).arg(&dids).arg(&mut out.buf).arg(&dim_i).arg(&rows_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems((rows * dim) as u32)) }.expect("embedding_gather");
    out
}

/// Scatter-add: `dtable[ids[r]] += dy[r]` (atomic; ids may repeat).
pub fn embedding_scatter_add(gpu: &Gpu, dtable: &mut DTensor, ids: &[usize], dy: &DTensor, dim: usize) {
    let rows = ids.len();
    let (dim_i, rows_i) = (dim as i32, rows as i32);
    let dids = upload_ids(gpu, ids);
    let f = gpu.kernels.get("embedding_scatter_add");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut dtable.buf).arg(&dids).arg(&dy.buf).arg(&dim_i).arg(&rows_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems((rows * dim) as u32)) }
        .expect("embedding_scatter_add");
}

/// Saved intermediates from a GPU RMSNorm forward, consumed by its backward.
/// The forward *output* is returned separately (it flows onward), so this holds
/// only what backward needs.
pub struct GpuRmsForward {
    pub x_hat: DTensor,
    pub inv_rms: CudaSlice<f32>,
}

/// Grouped RMSNorm forward (plain: `group == F`; head-wise: `group == dhv`).
/// Returns `(out, saved)`.
pub fn rms_norm_forward(gpu: &Gpu, x: &DTensor, gamma: &DTensor, group: usize, eps: f32) -> (DTensor, GpuRmsForward) {
    let (b, f) = (x.rows(), x.cols());
    let groups_per_row = f / group;
    let total_groups = b * groups_per_row;
    let mut out = DTensor::uninit(gpu, &[b, f]);
    let mut x_hat = DTensor::uninit(gpu, &[b, f]);
    let mut inv_rms = gpu.stream.alloc_zeros::<f32>(total_groups).expect("alloc inv_rms");
    let (gpr_i, group_i, tg_i) = (groups_per_row as i32, group as i32, total_groups as i32);
    let func = gpu.kernels.get("rms_norm_forward");
    let mut lb = gpu.stream.launch_builder(&func);
    lb.arg(&x.buf)
        .arg(&gamma.buf)
        .arg(&mut out.buf)
        .arg(&mut x_hat.buf)
        .arg(&mut inv_rms)
        .arg(&gpr_i)
        .arg(&group_i)
        .arg(&eps)
        .arg(&tg_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(total_groups as u32)) }.expect("rms_norm_forward");
    (out, GpuRmsForward { x_hat, inv_rms })
}

/// Grouped RMSNorm backward. Accumulates γ grad into `dgamma`, returns `dX`.
pub fn rms_norm_backward(
    gpu: &Gpu,
    dy: &DTensor,
    fwd: &GpuRmsForward,
    gamma: &DTensor,
    dgamma: &mut DTensor,
    group: usize,
) -> DTensor {
    let (b, f) = (dy.rows(), dy.cols());
    let groups_per_row = f / group;
    let total_groups = b * groups_per_row;
    let mut dx = DTensor::uninit(gpu, &[b, f]);
    let (gpr_i, group_i, tg_i) = (groups_per_row as i32, group as i32, total_groups as i32);
    let func = gpu.kernels.get("rms_norm_backward");
    let mut lb = gpu.stream.launch_builder(&func);
    lb.arg(&dy.buf)
        .arg(&fwd.x_hat.buf)
        .arg(&fwd.inv_rms)
        .arg(&gamma.buf)
        .arg(&mut dgamma.buf)
        .arg(&mut dx.buf)
        .arg(&gpr_i)
        .arg(&group_i)
        .arg(&tg_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(total_groups as u32)) }.expect("rms_norm_backward");
    dx
}

/// Fused softmax + cross-entropy. Returns `(mean_loss, dlogits)` with
/// `dlogits = (softmax − onehot) / B`, matching `nn2::loss`.
pub fn softmax_cross_entropy(gpu: &Gpu, logits: &DTensor, targets: &[usize]) -> (f32, DTensor) {
    let (b, c) = (logits.rows(), logits.cols());
    assert_eq!(targets.len(), b, "softmax_cross_entropy — targets len != batch");
    let inv_b = 1.0 / b as f32;
    let (c_i, b_i) = (c as i32, b as i32);
    let dtargets = upload_ids(gpu, targets);
    let mut dlogits = DTensor::uninit(gpu, &[b, c]);
    let mut row_loss = gpu.stream.alloc_zeros::<f32>(b).expect("alloc row_loss");
    let f = gpu.kernels.get("softmax_ce");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&logits.buf)
        .arg(&dtargets)
        .arg(&mut dlogits.buf)
        .arg(&mut row_loss)
        .arg(&c_i)
        .arg(&inv_b)
        .arg(&b_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(b as u32)) }.expect("softmax_ce");
    let losses = gpu.stream.memcpy_dtov(&row_loss).expect("download row_loss");
    let loss = losses.iter().sum::<f32>() * inv_b;
    (loss, dlogits)
}

/// One AdamW step of `param` from `grad`, updating moments `m`/`v` in place.
/// `decay` toggles the decoupled weight-decay term. Mirrors `nn2::optim`.
pub fn adamw(
    gpu: &Gpu,
    param: &mut DTensor,
    grad: &DTensor,
    m: &mut DTensor,
    v: &mut DTensor,
    cfg: &AdamCfg,
    decay: bool,
) {
    let n = param.len();
    let n_i = n as i32;
    let bc1 = 1.0 - cfg.beta1.powi(cfg.t as i32);
    let bc2 = 1.0 - cfg.beta2.powi(cfg.t as i32);
    let wd = if decay { cfg.weight_decay } else { 0.0 };
    let (lr, b1, b2, eps) = (cfg.lr, cfg.beta1, cfg.beta2, cfg.eps);
    let f = gpu.kernels.get("adamw");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut param.buf)
        .arg(&grad.buf)
        .arg(&mut m.buf)
        .arg(&mut v.buf)
        .arg(&lr)
        .arg(&b1)
        .arg(&b2)
        .arg(&eps)
        .arg(&wd)
        .arg(&bc1)
        .arg(&bc2)
        .arg(&n_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(n as u32)) }.expect("adamw");
}

// ---------------------------------------------------------------------------
// sLSTM cell kernels (recurrent core, see gpu/slstm.rs). Each is the device
// counterpart of an inner step of `nn2::SLstm`; all state stays resident in
// `DTensor`s across the T-loop — the only host transfers are the layer's input
// and output.
// ---------------------------------------------------------------------------

/// Build `xh = concat(x[:, t, :], h_state)` into `xh` (`[B, rows]`), reading the
/// timestep-`t` slice of `x` (`[B, T, inp]`) and the recurrent state (`[B, H]`).
pub fn concat_xh(gpu: &Gpu, xh: &mut DTensor, x: &DTensor, h_state: &DTensor, t: usize) {
    let (b, rows) = (xh.rows(), xh.cols());
    let h = h_state.cols();
    let inp = rows - h;
    let big_t = x.shape[1];
    let br = b * rows;
    let (t_i, bigt_i, inp_i, h_i, br_i) = (t as i32, big_t as i32, inp as i32, h as i32, br as i32);
    let f = gpu.kernels.get("concat_xh");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut xh.buf).arg(&x.buf).arg(&h_state.buf)
        .arg(&t_i).arg(&bigt_i).arg(&inp_i).arg(&h_i).arg(&br_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(br as u32)) }.expect("concat_xh");
}

/// Split `dxh` (`[B, rows]`) into `dx[:, t, :]` (first `inp` cols) and `dh_bptt`
/// (`[B, H]`, last `H` cols). `dx` is `[B, T, inp]`.
pub fn split_dxh(gpu: &Gpu, dxh: &DTensor, dx: &mut DTensor, dh_bptt: &mut DTensor, t: usize) {
    let (b, rows) = (dxh.rows(), dxh.cols());
    let h = dh_bptt.cols();
    let inp = rows - h;
    let big_t = dx.shape[1];
    let br = b * rows;
    let (t_i, bigt_i, inp_i, h_i, br_i) = (t as i32, big_t as i32, inp as i32, h as i32, br as i32);
    let f = gpu.kernels.get("split_dxh");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&dxh.buf).arg(&mut dx.buf).arg(&mut dh_bptt.buf)
        .arg(&t_i).arg(&bigt_i).arg(&inp_i).arg(&h_i).arg(&br_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(br as u32)) }.expect("split_dxh");
}

/// Per-step saved tensors of one sLSTM forward step, consumed by the backward
/// step. Each is `[B, H]` (`xh` lives on the layer). Grouped so the layer can
/// hold a `Vec` of them across the T-loop.
pub struct SlstmSaved {
    pub c_prev: DTensor,
    pub n_prev: DTensor,
    pub zt: DTensor,
    pub ot: DTensor,
    pub i_prime: DTensor,
    pub f_prime: DTensor,
    pub c: DTensor,
    pub n: DTensor,
}

/// One forward sLSTM step: advances `(c,n,m,h)_state` in place from the four gate
/// pre-activations, fills `saved` for backward, and writes `out[:, t, :]`.
/// `ft_pre` is the (bias-added) forget pre-activation and is itself a saved
/// per-step buffer (reused in backward).
#[allow(clippy::too_many_arguments)]
pub fn slstm_cell_step(
    gpu: &Gpu,
    zt_pre: &DTensor,
    it_pre: &DTensor,
    ft_pre: &DTensor,
    ot_pre: &DTensor,
    c_state: &mut DTensor,
    n_state: &mut DTensor,
    m_state: &mut DTensor,
    h_state: &mut DTensor,
    saved: &mut SlstmSaved,
    out: &mut DTensor,
    t: usize,
) {
    let (b, h) = (c_state.rows(), c_state.cols());
    let bh = b * h;
    let big_t = out.shape[1];
    let (t_i, bigt_i, h_i, bh_i) = (t as i32, big_t as i32, h as i32, bh as i32);
    let f = gpu.kernels.get("slstm_cell_step");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&zt_pre.buf).arg(&it_pre.buf).arg(&ft_pre.buf).arg(&ot_pre.buf)
        .arg(&mut c_state.buf).arg(&mut n_state.buf).arg(&mut m_state.buf).arg(&mut h_state.buf)
        .arg(&mut saved.c_prev.buf).arg(&mut saved.n_prev.buf).arg(&mut saved.zt.buf).arg(&mut saved.ot.buf)
        .arg(&mut saved.i_prime.buf).arg(&mut saved.f_prime.buf).arg(&mut saved.c.buf)
        .arg(&mut saved.n.buf)
        .arg(&mut out.buf).arg(&t_i).arg(&bigt_i).arg(&h_i).arg(&bh_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(bh as u32)) }.expect("slstm_cell_step");
}

/// One backward sLSTM step: from `dy[:, t, :]` + the incoming BPTT channels,
/// produce the four gate deltas (`dz,di,df,dob`) and update `dc_bptt`/`dn_bptt`
/// in place for the earlier step. `dh_bptt` is read here (set by the later step's
/// `split_dxh`).
#[allow(clippy::too_many_arguments)]
pub fn slstm_cell_step_bwd(
    gpu: &Gpu,
    dy: &DTensor,
    dh_bptt: &DTensor,
    saved: &SlstmSaved,
    ft_pre: &DTensor,
    dc_bptt: &mut DTensor,
    dn_bptt: &mut DTensor,
    dz: &mut DTensor,
    di: &mut DTensor,
    df: &mut DTensor,
    dob: &mut DTensor,
    t: usize,
) {
    let (b, h) = (dc_bptt.rows(), dc_bptt.cols());
    let bh = b * h;
    let big_t = dy.shape[1];
    let (t_i, bigt_i, h_i, bh_i) = (t as i32, big_t as i32, h as i32, bh as i32);
    let f = gpu.kernels.get("slstm_cell_step_bwd");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&dy.buf).arg(&t_i).arg(&bigt_i).arg(&dh_bptt.buf)
        .arg(&saved.ot.buf).arg(&saved.c.buf).arg(&saved.n.buf)
        .arg(&saved.c_prev.buf).arg(&saved.n_prev.buf).arg(&saved.zt.buf)
        .arg(&saved.i_prime.buf).arg(&saved.f_prime.buf).arg(&ft_pre.buf)
        .arg(&mut dc_bptt.buf).arg(&mut dn_bptt.buf)
        .arg(&mut dz.buf).arg(&mut di.buf).arg(&mut df.buf).arg(&mut dob.buf)
        .arg(&h_i).arg(&bh_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(bh as u32)) }.expect("slstm_cell_step_bwd");
}

// ---------------------------------------------------------------------------
// Fused-gate sLSTM (the fast path; see gpu/slstm.rs). The four gates run as one
// [., 4H] block, so a timestep costs one GEMM + one kernel instead of four GEMMs
// plus four bias broadcasts plus a concat. The gate weights of record stay the
// four [rows, H] matrices; these pack them into the fused operands and unpack the
// gradients back, so the checkpoint layout is untouched.
// ---------------------------------------------------------------------------

/// Pack the four gate matrices `[rows, H]` into `wx [inp, 4H]` / `wh [H, 4H]` and
/// the four biases into `bcat [4H]`. Two launches, once per forward.
#[allow(clippy::too_many_arguments)]
pub fn slstm_pack(
    gpu: &Gpu,
    w: &[DTensor; 4],
    bias: &[DTensor; 4],
    wx: &mut DTensor,
    wh: &mut DTensor,
    bcat: &mut DTensor,
    inp: usize,
    h: usize,
) {
    let rows = inp + h;
    let (inp_i, h_i, rows_i) = (inp as i32, h as i32, rows as i32);
    let f = gpu.kernels.get("slstm_pack_w");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&w[0].buf).arg(&w[1].buf).arg(&w[2].buf).arg(&w[3].buf)
        .arg(&mut wx.buf).arg(&mut wh.buf)
        .arg(&inp_i).arg(&h_i).arg(&rows_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems((rows * 4 * h) as u32)) }.expect("slstm_pack_w");

    let f = gpu.kernels.get("slstm_pack_b");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&bias[0].buf).arg(&bias[1].buf).arg(&bias[2].buf).arg(&bias[3].buf)
        .arg(&mut bcat.buf).arg(&h_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems((4 * h) as u32)) }.expect("slstm_pack_b");
}

/// `dw[g] += ` the g-th column block of the fused `dwx` / `dwh` (the inverse of
/// [`slstm_pack`] for gradients — accumulating, so grads survive across windows).
pub fn slstm_unpack_dw(
    gpu: &Gpu,
    dwx: &DTensor,
    dwh: &DTensor,
    dw: &mut [DTensor; 4],
    inp: usize,
    h: usize,
) {
    let rows = inp + h;
    let (inp_i, h_i, rows_i) = (inp as i32, h as i32, rows as i32);
    let [dw0, dw1, dw2, dw3] = dw;
    let f = gpu.kernels.get("slstm_unpack_dw");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&dwx.buf).arg(&dwh.buf)
        .arg(&mut dw0.buf).arg(&mut dw1.buf).arg(&mut dw2.buf).arg(&mut dw3.buf)
        .arg(&inp_i).arg(&h_i).arg(&rows_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems((rows * 4 * h) as u32)) }
        .expect("slstm_unpack_dw");
}

/// Fill `t` with a constant.
pub fn fill(gpu: &Gpu, t: &mut DTensor, v: f32) {
    let n = t.len();
    let n_i = n as i32;
    let f = gpu.kernels.get("fill_const");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut t.buf).arg(&v).arg(&n_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(n as u32)) }.expect("fill_const");
}

/// `db[g] += ` column sums of the g-th block of the fused gate deltas `dg [N, 4H]`.
/// The sum over the N rows is a `ones[1, N] · dg` GEMM (cuBLAS reduces properly);
/// the kernel only scatters the reduced `[4H]` row into the four bias grads.
pub fn slstm_db_from_dg(
    gpu: &Gpu,
    dg: &DTensor,
    ones: &DTensor,
    dbcat: &mut DTensor,
    db: &mut [DTensor; 4],
    h: usize,
) {
    matmul_nn_into(gpu, ones, dg, dbcat, 0.0);
    let h_i = h as i32;
    let [db0, db1, db2, db3] = db;
    let f = gpu.kernels.get("slstm_unpack_db");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&dbcat.buf)
        .arg(&mut db0.buf).arg(&mut db1.buf).arg(&mut db2.buf).arg(&mut db3.buf)
        .arg(&h_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems((4 * h) as u32)) }.expect("slstm_unpack_db");
}

/// Saved forward tensors of a fused-gate sLSTM, each a `[B, T, H]` slab (indexed
/// `(b·T + t)·H + j`) rather than one small tensor per timestep.
pub struct SlstmSlabs {
    pub c_prev: DTensor,
    pub n_prev: DTensor,
    pub zt: DTensor,
    pub ot: DTensor,
    pub i_prime: DTensor,
    pub f_prime: DTensor,
    pub c: DTensor,
    pub n: DTensor,
    pub h_prev: DTensor,
}

/// One fused forward step: add the biases, advance `(c,n,m,h)_state`, fill the
/// saved slabs at `t` and write `out[:, t, :]`. `g`'s forget block is left holding
/// the biased forget pre-activation for backward.
#[allow(clippy::too_many_arguments)]
pub fn slstm_step_fused(
    gpu: &Gpu,
    g: &mut DTensor,
    gh: &DTensor,
    bcat: &DTensor,
    c_state: &mut DTensor,
    n_state: &mut DTensor,
    m_state: &mut DTensor,
    h_state: &mut DTensor,
    slabs: &mut SlstmSlabs,
    out: &mut DTensor,
    t: usize,
) {
    let (b, h) = (c_state.rows(), c_state.cols());
    let bh = b * h;
    let big_t = out.shape[1];
    let (t_i, bigt_i, h_i, bh_i) = (t as i32, big_t as i32, h as i32, bh as i32);
    let f = gpu.kernels.get("slstm_step_fused");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut g.buf).arg(&gh.buf).arg(&bcat.buf).arg(&mut slabs.h_prev.buf)
        .arg(&mut c_state.buf).arg(&mut n_state.buf).arg(&mut m_state.buf).arg(&mut h_state.buf)
        .arg(&mut slabs.c_prev.buf).arg(&mut slabs.n_prev.buf).arg(&mut slabs.zt.buf)
        .arg(&mut slabs.ot.buf).arg(&mut slabs.i_prime.buf).arg(&mut slabs.f_prime.buf)
        .arg(&mut slabs.c.buf).arg(&mut slabs.n.buf)
        .arg(&mut out.buf).arg(&t_i).arg(&bigt_i).arg(&h_i).arg(&bh_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(bh as u32)) }.expect("slstm_step_fused");
}

/// One fused backward step: writes the four gate deltas back into `g` (its
/// forward contents are dead once read) and carries `dc/dn_bptt` back a step.
#[allow(clippy::too_many_arguments)]
pub fn slstm_step_fused_bwd(
    gpu: &Gpu,
    dy: &DTensor,
    g: &mut DTensor,
    dgh: &mut DTensor,
    dh_bptt: &DTensor,
    slabs: &SlstmSlabs,
    dc_bptt: &mut DTensor,
    dn_bptt: &mut DTensor,
    t: usize,
) {
    let (b, h) = (dc_bptt.rows(), dc_bptt.cols());
    let bh = b * h;
    let big_t = dy.shape[1];
    let (t_i, bigt_i, h_i, bh_i) = (t as i32, big_t as i32, h as i32, bh as i32);
    let f = gpu.kernels.get("slstm_step_fused_bwd");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&dy.buf).arg(&mut g.buf).arg(&mut dgh.buf).arg(&dh_bptt.buf)
        .arg(&slabs.ot.buf).arg(&slabs.c.buf).arg(&slabs.n.buf)
        .arg(&slabs.c_prev.buf).arg(&slabs.n_prev.buf).arg(&slabs.zt.buf)
        .arg(&slabs.i_prime.buf).arg(&slabs.f_prime.buf)
        .arg(&mut dc_bptt.buf).arg(&mut dn_bptt.buf)
        .arg(&t_i).arg(&bigt_i).arg(&h_i).arg(&bh_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(bh as u32)) }.expect("slstm_step_fused_bwd");
}

// ---------------------------------------------------------------------------
// Residual block / SwiGLU kernels (see gpu/block.rs).
// ---------------------------------------------------------------------------

/// Elementwise `out = a + b` (fresh allocation). Used for residual adds and the
/// grad accumulations that are plain sums.
pub fn add(gpu: &Gpu, a: &DTensor, b: &DTensor) -> DTensor {
    let n = a.len();
    assert_eq!(n, b.len(), "add: length mismatch");
    let n_i = n as i32;
    let mut out = DTensor::uninit(gpu, a.dims());
    let f = gpu.kernels.get("add");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut out.buf).arg(&a.buf).arg(&b.buf).arg(&n_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(n as u32)) }.expect("add");
    out
}

/// SwiGLU forward: returns `(gate_act = SiLU(gate_pre), mixed = gate_act ⊙ value)`.
pub fn swiglu_forward(gpu: &Gpu, gate_pre: &DTensor, value: &DTensor) -> (DTensor, DTensor) {
    let n = gate_pre.len();
    assert_eq!(n, value.len(), "swiglu_forward: length mismatch");
    let n_i = n as i32;
    let mut gate_act = DTensor::uninit(gpu, gate_pre.dims());
    let mut mixed = DTensor::uninit(gpu, gate_pre.dims());
    let f = gpu.kernels.get("swiglu_forward");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&gate_pre.buf).arg(&value.buf).arg(&mut gate_act.buf).arg(&mut mixed.buf).arg(&n_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(n as u32)) }.expect("swiglu_forward");
    (gate_act, mixed)
}

/// SwiGLU backward: from `d_mixed` and the saved `gate_act`/`value`/`gate_pre`,
/// returns `(d_gate, d_value)`.
pub fn swiglu_backward(
    gpu: &Gpu,
    d_mixed: &DTensor,
    gate_act: &DTensor,
    value: &DTensor,
    gate_pre: &DTensor,
) -> (DTensor, DTensor) {
    let n = d_mixed.len();
    let n_i = n as i32;
    let mut d_gate = DTensor::uninit(gpu, d_mixed.dims());
    let mut d_value = DTensor::uninit(gpu, d_mixed.dims());
    let f = gpu.kernels.get("swiglu_backward");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&d_mixed.buf).arg(&gate_act.buf).arg(&value.buf).arg(&gate_pre.buf)
        .arg(&mut d_gate.buf).arg(&mut d_value.buf).arg(&n_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(n as u32)) }.expect("swiglu_backward");
    (d_gate, d_value)
}

// ---------------------------------------------------------------------------
// mLSTM parallel/chunkwise core (see gpu/mlstm.rs).
// ---------------------------------------------------------------------------

/// Position-major `[N, H*W]` (N=B·T) → head-major `[B*H, T, W]`.
pub fn head_gather(gpu: &Gpu, x: &DTensor, b: usize, h: usize, t: usize, w: usize) -> DTensor {
    let mut out = DTensor::uninit(gpu, &[b * h, t, w]);
    let (bi, hi, ti, wi) = (b as i32, h as i32, t as i32, w as i32);
    let f = gpu.kernels.get("head_gather");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&x.buf).arg(&mut out.buf).arg(&bi).arg(&hi).arg(&ti).arg(&wi);
    unsafe { lb.launch(LaunchConfig::for_num_elems((b * h * t * w) as u32)) }.expect("head_gather");
    out
}

/// Head-major `[B*H, T, W]` → position-major `[N, H*W]` (inverse of `head_gather`).
pub fn head_scatter(gpu: &Gpu, x: &DTensor, b: usize, h: usize, t: usize, w: usize) -> DTensor {
    let mut out = DTensor::uninit(gpu, &[b * t, h * w]);
    let (bi, hi, ti, wi) = (b as i32, h as i32, t as i32, w as i32);
    let f = gpu.kernels.get("head_scatter");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&x.buf).arg(&mut out.buf).arg(&bi).arg(&hi).arg(&ti).arg(&wi);
    unsafe { lb.launch(LaunchConfig::for_num_elems((b * h * t * w) as u32)) }.expect("head_scatter");
    out
}

/// Inclusive cumsum of logσ along T, per row of `f` `[BH, T]` → `fc` `[BH, T]`.
pub fn cumsum_logsig(gpu: &Gpu, f: &DTensor) -> DTensor {
    let (bh, t) = (f.rows(), f.cols());
    let mut fc = DTensor::uninit(gpu, &[bh, t]);
    let (ti, bhi) = (t as i32, bh as i32);
    let func = gpu.kernels.get("cumsum_logsig");
    let mut lb = gpu.stream.launch_builder(&func);
    lb.arg(&f.buf).arg(&mut fc.buf).arg(&ti).arg(&bhi);
    unsafe { lb.launch(LaunchConfig::for_num_elems(bh as u32)) }.expect("cumsum_logsig");
    fc
}

/// Per-row stabilizer `m` `[BH, T]` = `max(max_{j≤t}(fc_t−fc_j+ig_j), fc_t+m_prev)`.
pub fn mlstm_rowmax_m(gpu: &Gpu, fc: &DTensor, ig: &DTensor, m_prev: &DTensor) -> DTensor {
    let (bh, t) = (fc.rows(), fc.cols());
    let mut m = DTensor::uninit(gpu, &[bh, t]);
    let (ti, bht) = (t as i32, (bh * t) as i32);
    let f = gpu.kernels.get("mlstm_rowmax_m");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&fc.buf).arg(&ig.buf).arg(&m_prev.buf).arg(&mut m.buf).arg(&ti).arg(&bht);
    unsafe { lb.launch(LaunchConfig::for_num_elems((bh * t) as u32)) }.expect("mlstm_rowmax_m");
    m
}

/// `DS = D̄ ⊙ S` plus the decay weights `D̄`, row normalizer `qn` and `ψ`. `s` is
/// `[BH, T, T]`; returns `(Dbar, DS, qn, psi)`. `D̄` is retained for the backward.
pub fn mlstm_ds(gpu: &Gpu, s: &DTensor, fc: &DTensor, ig: &DTensor, m: &DTensor)
    -> (DTensor, DTensor, DTensor, DTensor) {
    let (bh, t) = (fc.rows(), fc.cols());
    let mut dbar = DTensor::uninit(gpu, &[bh, t, t]);
    let mut ds = DTensor::uninit(gpu, &[bh, t, t]);
    let mut qn = DTensor::uninit(gpu, &[bh, t]);
    let mut psi = DTensor::uninit(gpu, &[bh, t]);
    let (ti, bht) = (t as i32, (bh * t) as i32);
    let f = gpu.kernels.get("mlstm_ds");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&s.buf).arg(&fc.buf).arg(&ig.buf).arg(&m.buf)
        .arg(&mut dbar.buf).arg(&mut ds.buf).arg(&mut qn.buf).arg(&mut psi.buf).arg(&ti).arg(&bht);
    unsafe { lb.launch(LaunchConfig::for_num_elems((bh * t) as u32)) }.expect("mlstm_ds");
    (dbar, ds, qn, psi)
}

// ---------------------------------------------------------------------------
// mLSTM chunking (inter-chunk state carry; see gpu/mlstm.rs).
// ---------------------------------------------------------------------------

/// Extract the T-range `[c0, c0+len)` of a head-major `[BH, T, W]` tensor.
pub fn slice_t(gpu: &Gpu, x: &DTensor, c0: usize, len: usize) -> DTensor {
    let (bh, t, w) = (x.shape[0], x.shape[1], x.shape[2]);
    assert!(c0 + len <= t, "slice_t: chunk [{c0}, {}) out of range T={t}", c0 + len);
    let mut out = DTensor::uninit(gpu, &[bh, len, w]);
    let total = bh * len * w;
    let (ti, li, c0i, wi, total_i) = (t as i32, len as i32, c0 as i32, w as i32, total as i32);
    let f = gpu.kernels.get("slice_t");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&x.buf).arg(&mut out.buf).arg(&ti).arg(&li).arg(&c0i).arg(&wi).arg(&total_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(total as u32)) }.expect("slice_t");
    out
}

/// Write `src` `[BH, len, W]` back into the T-range `[c0, c0+len)` of `dst`
/// `[BH, T, W]`. Chunks partition T, so this is a store, not an accumulate.
pub fn unslice_t(gpu: &Gpu, dst: &mut DTensor, src: &DTensor, c0: usize) {
    let (bh, t, w) = (dst.shape[0], dst.shape[1], dst.shape[2]);
    let len = src.shape[1];
    assert!(c0 + len <= t, "unslice_t: chunk [{c0}, {}) out of range T={t}", c0 + len);
    let total = bh * len * w;
    let (ti, li, c0i, wi, total_i) = (t as i32, len as i32, c0 as i32, w as i32, total as i32);
    let f = gpu.kernels.get("unslice_t");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut dst.buf).arg(&src.buf).arg(&ti).arg(&li).arg(&c0i).arg(&wi).arg(&total_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(total as u32)) }.expect("unslice_t");
}

/// The chunk's two inter-chunk decay weight vectors, both `[BH, L]`:
/// `b_t = exp(fc_t + m_prev − m_t)` (carried state → row t) and
/// `a_j = exp(fc_last − fc_j + ig_j − m_last)` (row j → outgoing state).
pub fn mlstm_chunk_ab(gpu: &Gpu, fc: &DTensor, ig: &DTensor, m: &DTensor, m_prev: &DTensor)
    -> (DTensor, DTensor) {
    let (bh, l) = (fc.rows(), fc.cols());
    let mut b = DTensor::uninit(gpu, &[bh, l]);
    let mut a = DTensor::uninit(gpu, &[bh, l]);
    let (li, bhl) = (l as i32, (bh * l) as i32);
    let f = gpu.kernels.get("mlstm_chunk_ab");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&fc.buf).arg(&ig.buf).arg(&m.buf).arg(&m_prev.buf)
        .arg(&mut b.buf).arg(&mut a.buf).arg(&li).arg(&bhl);
    unsafe { lb.launch(LaunchConfig::for_num_elems((bh * l) as u32)) }.expect("mlstm_chunk_ab");
    (b, a)
}

/// `out = s ⊙ x` broadcast over the trailing width `w`: `out[i] = s[i/w]·x[i]`.
pub fn mul_rows(gpu: &Gpu, x: &DTensor, s: &DTensor, w: usize) -> DTensor {
    let total = x.len();
    assert_eq!(total, s.len() * w, "mul_rows: {total} != {} · {w}", s.len());
    let mut out = DTensor::uninit(gpu, x.dims());
    let (wi, total_i) = (w as i32, total as i32);
    let f = gpu.kernels.get("mul_rows");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut out.buf).arg(&x.buf).arg(&s.buf).arg(&wi).arg(&total_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(total as u32)) }.expect("mul_rows");
    out
}

/// `out += s ⊙ x` broadcast over the trailing width `w`. Doubles as the per-head
/// state decay (`C += g[head]·C_prev`, with `w` = the head's element count).
pub fn mul_rows_add(gpu: &Gpu, out: &mut DTensor, x: &DTensor, s: &DTensor, w: usize) {
    let total = x.len();
    assert_eq!(total, out.len(), "mul_rows_add: out/x length mismatch");
    assert_eq!(total, s.len() * w, "mul_rows_add: {total} != {} · {w}", s.len());
    let (wi, total_i) = (w as i32, total as i32);
    let f = gpu.kernels.get("mul_rows_add");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut out.buf).arg(&x.buf).arg(&s.buf).arg(&wi).arg(&total_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(total as u32)) }.expect("mul_rows_add");
}

/// `ψ = max(|qn|, 1)` — needed once the inter-chunk term has been folded into `qn`.
pub fn psi_from_qn(gpu: &Gpu, qn: &DTensor, m: &DTensor) -> DTensor {
    let n = qn.len();
    let n_i = n as i32;
    let mut psi = DTensor::uninit(gpu, qn.dims());
    let f = gpu.kernels.get("psi_from_qn");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&qn.buf).arg(&m.buf).arg(&mut psi.buf).arg(&n_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(n as u32)) }.expect("psi_from_qn");
    psi
}

/// `out[r] += Σ_w x[r,w]·y[r,w]` for `[R, W]` operands (`out` is `[R]`).
pub fn row_dot_add(gpu: &Gpu, out: &mut DTensor, x: &DTensor, y: &DTensor, w: usize) {
    let r = out.len();
    assert_eq!(x.len(), r * w, "row_dot_add: x length mismatch");
    assert_eq!(y.len(), r * w, "row_dot_add: y length mismatch");
    let (wi, ri) = (w as i32, r as i32);
    let f = gpu.kernels.get("row_dot_add");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut out.buf).arg(&x.buf).arg(&y.buf).arg(&wi).arg(&ri);
    unsafe { lb.launch(LaunchConfig::for_num_elems(r as u32)) }.expect("row_dot_add");
}

/// `out[g] += Σ_e x[g,e]·y[g,e]` for `[G, E]` operands (`out` is `[G]`) — the
/// per-head reduction behind `dg`.
pub fn group_dot_add(gpu: &Gpu, out: &mut DTensor, x: &DTensor, y: &DTensor) {
    let g = out.len();
    let e = x.len() / g;
    assert_eq!(x.len(), y.len(), "group_dot_add: x/y length mismatch");
    assert_eq!(x.len(), g * e, "group_dot_add: not divisible by group count");
    let (ei, gi) = (e as i32, g as i32);
    let f = gpu.kernels.get("group_dot_add");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut out.buf).arg(&x.buf).arg(&y.buf).arg(&ei).arg(&gi);
    unsafe { lb.launch(LaunchConfig::for_num_elems(g as u32)) }.expect("group_dot_add");
}

/// Backward of [`mlstm_chunk_ab`], accumulating into the `dfc`/`dig` that
/// [`mlstm_dfc_dig`] already wrote from the intra-chunk D̄.
pub fn mlstm_chunk_ab_bwd(
    gpu: &Gpu, db: &DTensor, da: &DTensor, b: &DTensor, a: &DTensor,
    dfc: &mut DTensor, dig: &mut DTensor,
) {
    let (bh, l) = (b.rows(), b.cols());
    let (li, bhl) = (l as i32, (bh * l) as i32);
    let f = gpu.kernels.get("mlstm_chunk_ab_bwd");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&db.buf).arg(&da.buf).arg(&b.buf).arg(&a.buf)
        .arg(&mut dfc.buf).arg(&mut dig.buf).arg(&li).arg(&bhl);
    unsafe { lb.launch(LaunchConfig::for_num_elems((bh * l) as u32)) }
        .expect("mlstm_chunk_ab_bwd");
}

/// Copy rows of `src` into arbitrary rows of `dst`: `dst[row_ids[i]] = src[i]`.
/// The inverse (pulling those rows back out) is [`embedding_gather`] with the same
/// row ids, treating the matrix as the "table".
pub fn scatter_rows(gpu: &Gpu, dst: &mut DTensor, src: &DTensor, row_ids: &[usize]) {
    let dim = src.cols();
    let rows = src.rows();
    assert_eq!(rows, row_ids.len(), "scatter_rows: row_ids len != src rows");
    let ids = upload_ids(gpu, row_ids);
    let (dim_i, rows_i) = (dim as i32, rows as i32);
    let f = gpu.kernels.get("scatter_rows");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut dst.buf).arg(&src.buf).arg(&ids).arg(&dim_i).arg(&rows_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems((rows * dim) as u32)) }.expect("scatter_rows");
}

/// Column-concatenate `a` `[R, C]` and `b` `[R, C]` into a fresh `[R, 2C]`
/// tensor `[a | b]`. Used by the bidirectional encoder to build
/// `[fwd_readout ; bwd_readout]` before the combine projection.
pub fn concat_cols(gpu: &Gpu, a: &DTensor, b: &DTensor) -> DTensor {
    let (r, c) = (a.rows(), a.cols());
    assert_eq!(b.rows(), r, "concat_cols: row mismatch");
    assert_eq!(b.cols(), c, "concat_cols: col mismatch");
    let mut out = DTensor::uninit(gpu, &[r, 2 * c]);
    let (r_i, c_i) = (r as i32, c as i32);
    let f = gpu.kernels.get("concat_cols");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut out.buf).arg(&a.buf).arg(&b.buf).arg(&r_i).arg(&c_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems((r * 2 * c) as u32)) }.expect("concat_cols");
    out
}

/// Inverse of [`concat_cols`]: split `d_out` `[R, 2C]` into `(da, db)`, each
/// `[R, C]` (the first and last `C` columns).
pub fn split_cols(gpu: &Gpu, d_out: &DTensor) -> (DTensor, DTensor) {
    let (r, cc) = (d_out.rows(), d_out.cols());
    debug_assert_eq!(cc % 2, 0, "split_cols: odd width");
    let c = cc / 2;
    let mut da = DTensor::uninit(gpu, &[r, c]);
    let mut db = DTensor::uninit(gpu, &[r, c]);
    let (r_i, c_i) = (r as i32, c as i32);
    let f = gpu.kernels.get("split_cols");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&d_out.buf).arg(&mut da.buf).arg(&mut db.buf).arg(&r_i).arg(&c_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems((r * 2 * c) as u32)) }.expect("split_cols");
    (da, db)
}

/// Masked softmax cross-entropy (the hierarchical decode loss). `mask[r] == false`
/// marks a padding row: zero loss, zero grad. Normalized by the number of valid
/// rows. Returns `(mean_loss, dlogits)`.
pub fn masked_softmax_cross_entropy(
    gpu: &Gpu, logits: &DTensor, targets: &[usize], mask: &[bool],
) -> (f32, DTensor) {
    let num_valid = mask.iter().filter(|&&m| m).count().max(1) as f32;
    masked_softmax_cross_entropy_scaled(gpu, logits, targets, mask, 1.0 / num_valid)
}

/// Masked CE with an explicit `1/N` normalizer. When one window is split into
/// several rectangles (the length-grouped word batches), every group must be
/// scaled by the window's TOTAL valid-row count — not its own — so the summed
/// losses and gradients equal the single-rectangle result.
pub fn masked_softmax_cross_entropy_scaled(
    gpu: &Gpu, logits: &DTensor, targets: &[usize], mask: &[bool], inv: f32,
) -> (f32, DTensor) {
    let (r, c) = (logits.rows(), logits.cols());
    assert_eq!(targets.len(), r, "masked CE — targets len != rows");
    assert_eq!(mask.len(), r, "masked CE — mask len != rows");
    let dtargets = upload_ids(gpu, targets);
    let mask_u: Vec<usize> = mask.iter().map(|&m| m as usize).collect();
    let dmask = upload_ids(gpu, &mask_u);
    let mut dlogits = DTensor::uninit(gpu, &[r, c]);
    let mut row_loss = gpu.stream.alloc_zeros::<f32>(r).expect("alloc row_loss");
    let (c_i, r_i) = (c as i32, r as i32);
    let f = gpu.kernels.get("masked_softmax_ce");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&logits.buf).arg(&dtargets).arg(&dmask)
        .arg(&mut dlogits.buf).arg(&mut row_loss).arg(&c_i).arg(&inv).arg(&r_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(r as u32)) }.expect("masked_softmax_ce");
    let losses = gpu.stream.memcpy_dtov(&row_loss).expect("download row_loss");
    (losses.iter().sum::<f32>() * inv, dlogits)
}

/// o-gate backward → `(do_pre, d_yhat)` from `d_hconcat` and saved `o`/`yhat`.
pub fn ogate_bwd(gpu: &Gpu, d_hconcat: &DTensor, o: &DTensor, yhat: &DTensor) -> (DTensor, DTensor) {
    let n = d_hconcat.len();
    let n_i = n as i32;
    let mut do_pre = DTensor::uninit(gpu, d_hconcat.dims());
    let mut d_yhat = DTensor::uninit(gpu, d_hconcat.dims());
    let f = gpu.kernels.get("ogate_bwd");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&d_hconcat.buf).arg(&o.buf).arg(&yhat.buf).arg(&mut do_pre.buf).arg(&mut d_yhat.buf).arg(&n_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(n as u32)) }.expect("ogate_bwd");
    (do_pre, d_yhat)
}

/// Backward of `ytil = num/ψ` → `(d_num, d_qn)`. `num` `[BH,T,dhv]`, `psi`/`qn` `[BH,T]`.
pub fn div_rows_bwd(
    gpu: &Gpu, d_ytil: &DTensor, num: &DTensor, psi: &DTensor, qn: &DTensor, m: &DTensor,
    dhv: usize,
) -> (DTensor, DTensor) {
    let (bh, t) = (psi.rows(), psi.cols());
    let mut d_num = DTensor::uninit(gpu, num.dims());
    let mut d_qn = DTensor::uninit(gpu, &[bh, t]);
    let (dhv_i, bht) = (dhv as i32, (bh * t) as i32);
    let f = gpu.kernels.get("div_rows_bwd");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&d_ytil.buf).arg(&num.buf).arg(&psi.buf).arg(&qn.buf).arg(&m.buf)
        .arg(&mut d_num.buf).arg(&mut d_qn.buf).arg(&dhv_i).arg(&bht);
    unsafe { lb.launch(LaunchConfig::for_num_elems((bh * t) as u32)) }.expect("div_rows_bwd");
    (d_num, d_qn)
}

/// Backward of `DS = D̄⊙S` + qn-sum → `(dS, P)`. `dds_num = d_num·Vᵀ` (num path),
/// `d_qn` the qn path; `dbar`/`ds` the saved forward weights.
pub fn mlstm_ds_bwd(gpu: &Gpu, dds_num: &DTensor, d_qn: &DTensor, dbar: &DTensor, ds: &DTensor)
    -> (DTensor, DTensor) {
    let (bh, t, _) = (dds_num.shape[0], dds_num.shape[1], dds_num.shape[2]);
    let total = bh * t * t;
    let mut d_s = DTensor::uninit(gpu, &[bh, t, t]);
    let mut p = DTensor::uninit(gpu, &[bh, t, t]);
    let (ti, total_i) = (t as i32, total as i32);
    let f = gpu.kernels.get("mlstm_ds_bwd");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&dds_num.buf).arg(&d_qn.buf).arg(&dbar.buf).arg(&ds.buf)
        .arg(&mut d_s.buf).arg(&mut p.buf).arg(&ti).arg(&total_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(total as u32)) }.expect("mlstm_ds_bwd");
    (d_s, p)
}

/// Reduce `P` `[BH,T,T]` into `(dfc, dig)` `[BH,T]` (see kernel).
pub fn mlstm_dfc_dig(gpu: &Gpu, p: &DTensor) -> (DTensor, DTensor) {
    let (bh, t) = (p.shape[0], p.shape[1]);
    let mut dfc = DTensor::uninit(gpu, &[bh, t]);
    let mut dig = DTensor::uninit(gpu, &[bh, t]);
    let (ti, bht) = (t as i32, (bh * t) as i32);
    let f = gpu.kernels.get("mlstm_dfc_dig");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&p.buf).arg(&mut dfc.buf).arg(&mut dig.buf).arg(&ti).arg(&bht);
    unsafe { lb.launch(LaunchConfig::for_num_elems((bh * t) as u32)) }.expect("mlstm_dfc_dig");
    (dfc, dig)
}

/// Reverse-cumsum + logσ' backward of `fc`: `df[BH,T]` from `dfc` and saved `f`.
pub fn revcumsum_dlogsig(gpu: &Gpu, dfc: &DTensor, f: &DTensor) -> DTensor {
    let (bh, t) = (dfc.rows(), dfc.cols());
    let mut df = DTensor::uninit(gpu, &[bh, t]);
    let (ti, bhi) = (t as i32, bh as i32);
    let func = gpu.kernels.get("revcumsum_dlogsig");
    let mut lb = gpu.stream.launch_builder(&func);
    lb.arg(&dfc.buf).arg(&f.buf).arg(&mut df.buf).arg(&ti).arg(&bhi);
    unsafe { lb.launch(LaunchConfig::for_num_elems(bh as u32)) }.expect("revcumsum_dlogsig");
    df
}

/// Row-normalize `num` `[BH, T, dhv]` by `psi` `[BH, T]` → `ytil` `[BH, T, dhv]`.
pub fn div_rows(gpu: &Gpu, num: &DTensor, psi: &DTensor, dhv: usize) -> DTensor {
    let total = num.len();
    let mut ytil = DTensor::uninit(gpu, num.dims());
    let (dhv_i, total_i) = (dhv as i32, total as i32);
    let f = gpu.kernels.get("div_rows");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&num.buf).arg(&psi.buf).arg(&mut ytil.buf).arg(&dhv_i).arg(&total_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(total as u32)) }.expect("div_rows");
    ytil
}

/// Elementwise product `out = a ⊙ b` (fresh allocation).
pub fn mul(gpu: &Gpu, a: &DTensor, b: &DTensor) -> DTensor {
    let n = a.len();
    assert_eq!(n, b.len(), "mul: length mismatch");
    let n_i = n as i32;
    let mut out = DTensor::uninit(gpu, a.dims());
    let f = gpu.kernels.get("mul");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut out.buf).arg(&a.buf).arg(&b.buf).arg(&n_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(n as u32)) }.expect("mul");
    out
}

// ---------------------------------------------------------------------------
// Fused chunkwise mLSTM (TFLA). See the kernel block in `gpu/kernels.rs`.
//
// Three launches forward, two backward, for a whole sequence — against ~25 per
// chunk on the op-at-a-time path. The [L, L] decay matrix stays in shared memory,
// so none of the per-chunk intermediates the old path allocated exist at all.
// ---------------------------------------------------------------------------

/// Threads per block for the two RECURRENT kernels (`fw_C`, `bw_dC`). They are
/// small and shared-memory-cheap; their problem is grid size, not warp count.
///
/// Must stay a power of two, and must not exceed the `__shared__ float sRed[256]`
/// that `fw_C`'s max-reduction indexes.
pub const FUSED_THREADS_REC: u32 = 256;

/// Threads per block for the two PARALLEL kernels (`fw_parallel`, `bw_parallel`).
///
/// 512, not 256, because shared memory — not registers — is what caps these. At
/// 71 KB a `bw_parallel` block is the only one that fits on an SM, so with 256
/// threads it had just 8 of the SM's 48 warps resident (17% occupancy) to hide the
/// latency of its shared-memory FMAs. Doubling the block doubles the resident warps
/// *for free*: the shared memory per block is unchanged, and 48 regs x 512 threads
/// is still well inside the 64 K register file. `gpu_occupancy` is how you check
/// that this is still true after any change to the kernels.
///
/// Must stay a power of two and not exceed `bw_parallel`'s `sRed[512]`.
pub const FUSED_THREADS_PAR: u32 = 512;

/// Longest chunk the fused kernels support. `L` bounds the shared-memory
/// footprint — the backward stages `[L, dqk]`, `[L, dhv]` and `[L, L]` blocks plus
/// two `[dhv, dqk]` states, which at L=32, dqk=dhv=64 is ~70 KB, inside the 99 KB
/// opt-in limit and out of reach at L=64. Chunk length is a pure blocking choice
/// (`mlstm_chunking_matches_single_chunk` pins every L to the same numbers), so
/// capping it costs accuracy nothing.
pub const FUSED_MAX_L: usize = 32;

/// What forward hands to backward. Everything here is `[BH, …]` head-major.
///
/// Deliberately absent: `D̄` and `D̄⊙S`. The op-at-a-time path saved both — two
/// `[BH, L, L]` slabs per chunk — and backward re-read them from HBM; the fused
/// backward recomputes them in shared memory from `(Q, K, fc, ig, m)` instead.
pub struct MlstmFused {
    pub l: usize,
    pub nc: usize,
    pub ytil: DTensor, // [BH, T, dhv]
    cst: DTensor,      // [BH, NC+1, dhv, dqk]  state entering each chunk
    nst: DTensor,      // [BH, NC+1, dqk]
    mst: DTensor,      // [BH, NC+1]
    fcb: DTensor,      // [BH, NC, L]  chunk-local cumulative log-forget
    msv: DTensor,      // [BH, T]      per-row stabilizer
    psiv: DTensor,     // [BH, T]
    qnv: DTensor,      // [BH, T]
}

/// Shared-memory floats each kernel needs, given the blocking. Kept next to the
/// kernels' `extern __shared__` layouts — the two must agree exactly.
///
/// The `+ 1`s are the bank-conflict pad: every 2D shared array is stored with its
/// row stride one float longer than its row, so that warps walking `row` at fixed
/// `col` spread across banks instead of piling onto one. See `mlstm_fw_C`.
fn fused_smem(kind: &str, l: usize, dqk: usize, dhv: usize) -> usize {
    let (lq, lv, ls) = (dqk + 1, dhv + 1, l + 1);
    let tv = fused_tv(dhv);
    let lvt = tv + 1; // the two recurrent kernels stage only a `tv`-wide slice of v
    match kind {
        "fw_C" => l * lq + l * lvt + tv * lq + dqk + 3 * l,
        "fw_parallel" => 2 * l * lq + l * lv + dhv * lq + l * ls + dqk + 5 * l,
        "bw_dC" => l * lq + l * lvt + tv * lq + dqk + 2 * l,
        "bw_parallel" => 2 * l * lq + 2 * l * lv + l * ls + 2 * dhv * lq + 2 * dqk + 11 * l,
        // The mma kernels pad every dim up to the tensor-core tile (rows to the
        // mma M=16, the contraction to K=8, the columns to N=8) and zero-fill the
        // pad, which is what lets one code path serve a short last chunk and an odd
        // dqk/dhv alike. Must match the `LP`/`KP`/`VP` the kernel recomputes.
        "fw_parallel_mma" => {
            let (lp, kp, vp) = (mma_pad(l, 16), mma_pad(dqk, 8), mma_pad(dhv, 8));
            let (lq, lv, ls) = (kp + 1, vp + 1, lp + 1);
            2 * lp * lq + lp * lv + vp * lq + lp * ls + kp + 5 * lp
        }
        "bw_parallel_mma" => {
            let (lp, kp, vp) = (mma_pad(l, 16), mma_pad(dqk, 8), mma_pad(dhv, 8));
            let (lq, lv, ls) = (kp + 1, vp + 1, lp + 1);
            2 * lp * lq + 2 * lp * lv + lp * ls + 2 * vp * lq + 2 * kp + 11 * lp
        }
        _ => unreachable!(),
    }
}

/// Round `n` up to a multiple of `to` (a power of two): the mma tile padding.
fn mma_pad(n: usize, to: usize) -> usize {
    n.div_ceil(to) * to
}

/// Whether the tensor-core kernels are *selected* (as opposed to merely compiled).
///
/// `MLSTM_NO_MMA=1` forces the scalar path: the A/B baseline for benchmarking, the
/// exact-fp32 oracle `mlstm_fused_mma_matches_scalar` pins the tensor-core path
/// against, and the escape hatch if a tensor-core kernel ever misbehaves.
static MMA_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
static MMA_INIT: std::sync::Once = std::sync::Once::new();

fn mma_enabled() -> bool {
    MMA_INIT.call_once(|| {
        let on = std::env::var("MLSTM_NO_MMA").is_err();
        MMA_ON.store(on, std::sync::atomic::Ordering::Relaxed);
    });
    MMA_ON.load(std::sync::atomic::Ordering::Relaxed)
}

/// Flip the tensor-core path on/off at runtime. Only for the A/B tests, which run
/// the same input down both paths in one process; production reads the env once.
#[cfg(test)]
pub fn set_mma_enabled(on: bool) {
    mma_enabled(); // run the Once first, so it cannot clobber us afterwards
    MMA_ON.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// The parallel forward kernel to run: the tensor-core one where the device has
/// tensor cores, else the scalar one.
///
/// They compute the same thing; only the three contractions differ, and only in
/// that the tensor-core path rounds their inputs to TF32 — as the reference
/// `mlstm_kernels` does, Triton's `tl.dot` being TF32 on fp32 inputs by default.
fn fw_parallel_kernel(gpu: &Gpu) -> (&'static str, &'static str) {
    if gpu.kernels.has_mma && mma_enabled() {
        ("mlstm_fw_parallel_mma", "fw_parallel_mma")
    } else {
        ("mlstm_fw_parallel", "fw_parallel")
    }
}

/// The parallel backward kernel to run. See [`fw_parallel_kernel`].
fn bw_parallel_kernel(gpu: &Gpu) -> (&'static str, &'static str) {
    if gpu.kernels.has_mma && mma_enabled() {
        ("mlstm_bw_parallel_mma", "bw_parallel_mma")
    } else {
        ("mlstm_bw_parallel", "bw_parallel")
    }
}

/// Value-dimension tile for the two recurrent kernels. Their grid is otherwise BH
/// alone — 8 blocks at the backbone's shape, on a 48-SM card — so `v` is split to
/// fill the machine. 16 gives 4 tiles at dhv = 64, i.e. 32 blocks, each with a
/// small enough state to keep several resident per SM.
fn fused_tv(dhv: usize) -> usize {
    dhv.min(16)
}

/// Fetch a fused kernel and opt it into the shared memory it needs.
///
/// A block gets 48 KB of shared memory by default; past that the driver requires
/// an explicit per-function opt-in, and the backward's parallel kernel is ~70 KB
/// at the backbone's shape. Raising the cap costs occupancy (one block per SM),
/// which is the trade the whole design makes: one big block that keeps the decay
/// matrix resident beats many small ones that stream it through HBM.
fn fused_kernel(gpu: &Gpu, name: &str, smem_floats: usize) -> (cudarc::driver::CudaFunction, u32) {
    let bytes = (smem_floats * std::mem::size_of::<f32>()) as u32;
    let f = gpu.kernels.get(name);
    if bytes > 48 * 1024 {
        f.set_attribute(
            cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            bytes as i32,
        )
        .unwrap_or_else(|e| panic!("{name}: cannot opt into {bytes} B of shared memory: {e:?}"));
    }
    (f, bytes)
}

fn fused_cfg(grid: (u32, u32, u32), threads: u32, smem: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: grid, block_dim: (threads, 1, 1), shared_mem_bytes: smem }
}

/// Threads a fused kernel launches with — recurrent kernels and parallel kernels
/// have different needs. See the two constants above.
pub fn fused_threads(name: &str) -> u32 {
    match name {
        "mlstm_fw_C" | "mlstm_bw_dC" => FUSED_THREADS_REC,
        _ => FUSED_THREADS_PAR,
    }
}

/// Whether the fused kernels can run this shape. The op-at-a-time path stays as
/// the fallback (and as the A/B baseline for `mlstm_fused_bench`).
pub fn mlstm_fused_supported(l: usize, dqk: usize, dhv: usize) -> bool {
    l >= 1 && l <= FUSED_MAX_L && dqk >= 1 && dhv >= 1
}

/// The four heavy fused kernels at a given blocking, each already opted into the
/// shared memory it launches with, plus that byte count and its grid at `(bh, t)`.
///
/// Exists so `examples/gpu_occupancy.rs` can ask the *driver* what these kernels
/// actually cost (registers, spills, blocks resident per SM) instead of anyone
/// guessing from the source.
pub fn mlstm_fused_kernels(
    gpu: &Gpu,
    l: usize,
    dqk: usize,
    dhv: usize,
    bh: usize,
    t: usize,
) -> Vec<(&'static str, cudarc::driver::CudaFunction, u32, (u32, u32))> {
    let nc = t.div_ceil(l) as u32;
    let ntv = dhv.div_ceil(fused_tv(dhv)) as u32;
    let bh = bh as u32;
    // The two parallel kernels are reported as whichever variant would actually be
    // launched — asking the driver about the scalar one while the tensor-core one
    // runs would be reporting the wrong kernel's registers and shared memory.
    let (fw_par, fw_kind) = fw_parallel_kernel(gpu);
    let (bw_par, bw_kind) = bw_parallel_kernel(gpu);
    [
        ("fw_C", "mlstm_fw_C", (ntv, bh)),
        (fw_kind, fw_par, (nc, bh)),
        ("bw_dC", "mlstm_bw_dC", (ntv, bh)),
        (bw_kind, bw_par, (nc, bh)),
    ]
    .into_iter()
    .map(|(kind, name, grid)| {
        let (f, smem) = fused_kernel(gpu, name, fused_smem(kind, l, dqk, dhv));
        (name, f, smem, grid)
    })
    .collect()
}

/// Chunkwise forward: `mlstm_fw_gates` → `mlstm_fw_C` → `mlstm_fw_parallel`.
#[allow(clippy::too_many_arguments)]
pub fn mlstm_fused_fw(
    gpu: &Gpu,
    qh: &DTensor,  // [BH, T, dqk]
    kh: &DTensor,  // [BH, T, dqk]  (already scaled by 1/√dqk)
    vh: &DTensor,  // [BH, T, dhv]
    igh: &DTensor, // [BH, T]
    fgh: &DTensor, // [BH, T]
    l: usize,
) -> MlstmFused {
    let (bh, t, dqk) = (qh.shape[0], qh.shape[1], qh.shape[2]);
    let dhv = vh.shape[2];
    let l = l.min(t);
    assert!(mlstm_fused_supported(l, dqk, dhv), "fused mLSTM: unsupported shape");
    let nc = t.div_ceil(l);
    let (t_i, l_i, nc_i, dqk_i, dhv_i, bh_i) =
        (t as i32, l as i32, nc as i32, dqk as i32, dhv as i32, bh as i32);

    let mut fcb = DTensor::uninit(gpu, &[bh, nc, l]);
    let mut cst = DTensor::uninit(gpu, &[bh, nc + 1, dhv, dqk]);
    let mut nst = DTensor::uninit(gpu, &[bh, nc + 1, dqk]);
    let mut mst = DTensor::uninit(gpu, &[bh, nc + 1]);
    let mut ytil = DTensor::uninit(gpu, &[bh, t, dhv]);
    let mut msv = DTensor::uninit(gpu, &[bh, t]);
    let mut psiv = DTensor::uninit(gpu, &[bh, t]);
    let mut qnv = DTensor::uninit(gpu, &[bh, t]);

    let f = gpu.kernels.get("mlstm_fw_gates");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&fgh.buf).arg(&mut fcb.buf).arg(&t_i).arg(&l_i).arg(&nc_i).arg(&bh_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems((bh * nc) as u32)) }.expect("mlstm_fw_gates");

    let tv = fused_tv(dhv);
    let tv_i = tv as i32;
    let ntv = dhv.div_ceil(tv) as u32;

    let (f, smem) = fused_kernel(gpu, "mlstm_fw_C", fused_smem("fw_C", l, dqk, dhv));
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&kh.buf).arg(&vh.buf).arg(&igh.buf).arg(&fcb.buf)
        .arg(&mut cst.buf).arg(&mut nst.buf).arg(&mut mst.buf)
        .arg(&t_i).arg(&l_i).arg(&nc_i).arg(&dqk_i).arg(&dhv_i).arg(&tv_i);
    unsafe { lb.launch(fused_cfg((ntv, bh as u32, 1), FUSED_THREADS_REC, smem)) }.expect("mlstm_fw_C");

    let (name, kind) = fw_parallel_kernel(gpu);
    let (f, smem) = fused_kernel(gpu, name, fused_smem(kind, l, dqk, dhv));
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&qh.buf).arg(&kh.buf).arg(&vh.buf).arg(&igh.buf).arg(&fcb.buf)
        .arg(&cst.buf).arg(&nst.buf).arg(&mst.buf)
        .arg(&mut ytil.buf).arg(&mut msv.buf).arg(&mut psiv.buf).arg(&mut qnv.buf)
        .arg(&t_i).arg(&l_i).arg(&nc_i).arg(&dqk_i).arg(&dhv_i);
    unsafe { lb.launch(fused_cfg((nc as u32, bh as u32, 1), FUSED_THREADS_PAR, smem)) }.expect("mlstm_fw_parallel");

    MlstmFused { l, nc, ytil, cst, nst, mst, fcb, msv, psiv, qnv }
}

/// Chunkwise backward: `mlstm_bw_dC` → `mlstm_bw_parallel`.
/// Returns `(dq, dk, dv, dig, dfg)`, all head-major like the inputs.
#[allow(clippy::too_many_arguments)]
pub fn mlstm_fused_bw(
    gpu: &Gpu,
    sv: &MlstmFused,
    qh: &DTensor,
    kh: &DTensor,
    vh: &DTensor,
    igh: &DTensor,
    fgh: &DTensor,
    d_ytil: &DTensor, // [BH, T, dhv]
) -> (DTensor, DTensor, DTensor, DTensor, DTensor) {
    let (bh, t, dqk) = (qh.shape[0], qh.shape[1], qh.shape[2]);
    let dhv = vh.shape[2];
    let (l, nc) = (sv.l, sv.nc);
    let (t_i, l_i, nc_i, dqk_i, dhv_i) =
        (t as i32, l as i32, nc as i32, dqk as i32, dhv as i32);

    let mut dcst = DTensor::uninit(gpu, &[bh, nc + 1, dhv, dqk]);
    let mut dnst = DTensor::uninit(gpu, &[bh, nc + 1, dqk]);

    let tv = fused_tv(dhv);
    let tv_i = tv as i32;
    let ntv = dhv.div_ceil(tv) as u32;

    let (f, smem) = fused_kernel(gpu, "mlstm_bw_dC", fused_smem("bw_dC", l, dqk, dhv));
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&qh.buf).arg(&d_ytil.buf).arg(&sv.ytil.buf)
        .arg(&sv.psiv.buf).arg(&sv.qnv.buf).arg(&sv.msv.buf)
        .arg(&sv.fcb.buf).arg(&sv.mst.buf)
        .arg(&mut dcst.buf).arg(&mut dnst.buf)
        .arg(&t_i).arg(&l_i).arg(&nc_i).arg(&dqk_i).arg(&dhv_i).arg(&tv_i);
    unsafe { lb.launch(fused_cfg((ntv, bh as u32, 1), FUSED_THREADS_REC, smem)) }.expect("mlstm_bw_dC");

    let mut dq = DTensor::uninit(gpu, &[bh, t, dqk]);
    let mut dk = DTensor::uninit(gpu, &[bh, t, dqk]);
    let mut dv = DTensor::uninit(gpu, &[bh, t, dhv]);
    let mut dig = DTensor::uninit(gpu, &[bh, t]);
    let mut dfg = DTensor::uninit(gpu, &[bh, t]);

    let (name, kind) = bw_parallel_kernel(gpu);
    let (f, smem) = fused_kernel(gpu, name, fused_smem(kind, l, dqk, dhv));
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&qh.buf).arg(&kh.buf).arg(&vh.buf).arg(&igh.buf).arg(&fgh.buf).arg(&sv.fcb.buf)
        .arg(&sv.cst.buf).arg(&sv.nst.buf).arg(&sv.mst.buf)
        .arg(&dcst.buf).arg(&dnst.buf)
        .arg(&sv.ytil.buf).arg(&d_ytil.buf).arg(&sv.psiv.buf)
        .arg(&sv.qnv.buf).arg(&sv.msv.buf)
        .arg(&mut dq.buf).arg(&mut dk.buf).arg(&mut dv.buf)
        .arg(&mut dig.buf).arg(&mut dfg.buf)
        .arg(&t_i).arg(&l_i).arg(&nc_i).arg(&dqk_i).arg(&dhv_i);
    unsafe { lb.launch(fused_cfg((nc as u32, bh as u32, 1), FUSED_THREADS_PAR, smem)) }.expect("mlstm_bw_parallel");

    (dq, dk, dv, dig, dfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor, gemm};

    fn assert_close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "length mismatch");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < 1e-3, "index {i}: gpu {g} vs cpu {w}");
        }
    }

    #[test]
    fn gemm_nn_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else { return };
        let (m, k, n) = (17, 23, 11);
        let a = Tensor::random(&[m, k], 1.0);
        let b = Tensor::random(&[k, n], 1.0);
        let want = gemm::matmul(&a, &b);
        let got = matmul(&gpu, &DTensor::from_host(&gpu, &a), &DTensor::from_host(&gpu, &b));
        assert_close(&got.to_host(&gpu).data, &want.data);
    }

    #[test]
    fn gemm_nt_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else { return };
        let (m, k, n) = (17, 23, 11);
        let a = Tensor::random(&[m, k], 1.0);
        let b = Tensor::random(&[n, k], 1.0);
        let want = gemm::matmul_nt(&a, &b);
        let got = matmul_nt(&gpu, &DTensor::from_host(&gpu, &a), &DTensor::from_host(&gpu, &b));
        assert_close(&got.to_host(&gpu).data, &want.data);
    }

    #[test]
    fn gemm_tn_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else { return };
        let (m, k, n) = (17, 23, 11);
        let a = Tensor::random(&[k, m], 1.0);
        let b = Tensor::random(&[k, n], 1.0);
        let want = gemm::matmul_tn(&a, &b);
        let got = matmul_tn(&gpu, &DTensor::from_host(&gpu, &a), &DTensor::from_host(&gpu, &b));
        assert_close(&got.to_host(&gpu).data, &want.data);
    }

    /// Per-batch CPU reference for the three strided-batched GEMM forms, checked
    /// against `tensor::gemm` looped over the batch axis.
    #[test]
    fn matmul_batched_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else { return };
        let (batch, m, k, n) = (5, 7, 6, 4);

        // nn: [batch,m,k]·[batch,k,n]
        let a = Tensor::random(&[batch, m, k], 1.0);
        let b = Tensor::random(&[batch, k, n], 1.0);
        let mut want = Tensor::zeros(&[batch, m, n]);
        for g in 0..batch {
            let (ao, bo, co) = (g * m * k, g * k * n, g * m * n);
            gemm::gemm_nn(m, k, n, &a.data[ao..ao + m * k], &b.data[bo..bo + k * n],
                &mut want.data[co..co + m * n], 0.0);
        }
        let got = matmul_batched_nn(&gpu, &DTensor::from_host(&gpu, &a), &DTensor::from_host(&gpu, &b));
        assert_close(&got.to_host(&gpu).data, &want.data);

        // nt: [batch,m,k]·[batch,n,k]ᵀ
        let bt = Tensor::random(&[batch, n, k], 1.0);
        let mut want_nt = Tensor::zeros(&[batch, m, n]);
        for g in 0..batch {
            let (ao, bo, co) = (g * m * k, g * n * k, g * m * n);
            gemm::gemm_nt(m, k, n, &a.data[ao..ao + m * k], &bt.data[bo..bo + n * k],
                &mut want_nt.data[co..co + m * n], 0.0);
        }
        let got_nt = matmul_batched_nt(&gpu, &DTensor::from_host(&gpu, &a), &DTensor::from_host(&gpu, &bt));
        assert_close(&got_nt.to_host(&gpu).data, &want_nt.data);

        // tn: [batch,k,m]ᵀ·[batch,k,n]
        let at = Tensor::random(&[batch, k, m], 1.0);
        let bn = Tensor::random(&[batch, k, n], 1.0);
        let mut want_tn = Tensor::zeros(&[batch, m, n]);
        for g in 0..batch {
            let (ao, bo, co) = (g * k * m, g * k * n, g * m * n);
            gemm::gemm_tn(m, k, n, &at.data[ao..ao + k * m], &bn.data[bo..bo + k * n],
                &mut want_tn.data[co..co + m * n], 0.0);
        }
        let got_tn = matmul_batched_tn(&gpu, &DTensor::from_host(&gpu, &at), &DTensor::from_host(&gpu, &bn));
        assert_close(&got_tn.to_host(&gpu).data, &want_tn.data);
    }

    #[test]
    fn scale_and_sigmoid_match_cpu() {
        let Some(gpu) = super::super::test_gpu() else { return };
        let x = Tensor::random(&[6, 5], 2.0);
        // scale
        let s = 0.37;
        let mut want_scale = x.clone();
        for v in want_scale.data.iter_mut() { *v *= s; }
        let mut dx = DTensor::from_host(&gpu, &x);
        scale_(&gpu, &mut dx, s);
        assert_close(&dx.to_host(&gpu).data, &want_scale.data);
        // sigmoid (matches the cell's stable_sigmoid)
        let sig = |v: f32| if v >= 0.0 { 1.0 / (1.0 + (-v).exp()) } else { let e = v.exp(); e / (1.0 + e) };
        let want_sig: Vec<f32> = x.data.iter().map(|&v| sig(v)).collect();
        let mut dx2 = DTensor::from_host(&gpu, &x);
        sigmoid_(&gpu, &mut dx2);
        assert_close(&dx2.to_host(&gpu).data, &want_sig);
    }

    #[test]
    fn softcap_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else { return };
        use crate::nn2::ops as cpu;
        let cap = 30.0;
        let x = Tensor::random(&[9, 13], 50.0);
        let dy = Tensor::random(&[9, 13], 1.0);
        let y_cpu = cpu::softcap_forward(&x, cap);
        let dx_cpu = cpu::softcap_backward(&dy, &y_cpu, cap);
        let dx = DTensor::from_host(&gpu, &x);
        let y_gpu = softcap_forward(&gpu, &dx, cap);
        let dx_gpu = softcap_backward(&gpu, &DTensor::from_host(&gpu, &dy), &y_gpu, cap);
        assert_close(&y_gpu.to_host(&gpu).data, &y_cpu.data);
        assert_close(&dx_gpu.to_host(&gpu).data, &dx_cpu.data);
    }

    #[test]
    fn linear_bias_helpers_match_cpu() {
        let Some(gpu) = super::super::test_gpu() else { return };
        use crate::nn2::ops as cpu;
        let (b, n) = (7, 5);
        let bias = Tensor::random(&[n], 1.0);
        let dy = Tensor::random(&[b, n], 1.0);
        // broadcast_row
        let mut out_cpu = Tensor::zeros(&[b, n]);
        cpu::broadcast_row(&mut out_cpu, &bias);
        let mut out_gpu = DTensor::zeros(&gpu, &[b, n]);
        broadcast_row(&gpu, &mut out_gpu, &DTensor::from_host(&gpu, &bias));
        assert_close(&out_gpu.to_host(&gpu).data, &out_cpu.data);
        // add_col_sum (start from a nonzero db to check accumulation)
        let mut db_cpu = Tensor::random(&[n], 1.0);
        let mut db_gpu = DTensor::from_host(&gpu, &db_cpu);
        cpu::add_col_sum(&mut db_cpu, &dy);
        add_col_sum(&gpu, &mut db_gpu, &DTensor::from_host(&gpu, &dy));
        assert_close(&db_gpu.to_host(&gpu).data, &db_cpu.data);
    }

    #[test]
    fn embedding_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else { return };
        use crate::nn2::ops as cpu;
        let (vocab, dim) = (11, 6);
        let table = Tensor::random(&[vocab, dim], 1.0);
        let ids = [3usize, 0, 3, 7, 3]; // repeats -> exercises scatter atomics
        let gathered_cpu = cpu::embedding_gather(&table, &ids, dim);
        let gathered_gpu = embedding_gather(&gpu, &DTensor::from_host(&gpu, &table), &ids, dim);
        assert_close(&gathered_gpu.to_host(&gpu).data, &gathered_cpu.data);
        // scatter_add from the gathered grads
        let mut dt_cpu = Tensor::zeros(&[vocab, dim]);
        cpu::embedding_scatter_add(&mut dt_cpu, &ids, &gathered_cpu, dim);
        let mut dt_gpu = DTensor::zeros(&gpu, &[vocab, dim]);
        embedding_scatter_add(&gpu, &mut dt_gpu, &ids, &gathered_gpu, dim);
        assert_close(&dt_gpu.to_host(&gpu).data, &dt_cpu.data);
    }

    #[test]
    fn rms_norm_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else { return };
        use crate::nn2::ops as cpu;
        let (b, f, group, eps) = (5, 12, 4, 1e-5); // head-wise case: 3 groups/row
        let x = Tensor::random(&[b, f], 1.0);
        let gamma = Tensor::random(&[f], 1.0);
        let dy = Tensor::random(&[b, f], 1.0);
        let fwd_cpu = cpu::rms_norm_forward(&x, &gamma, group, eps);
        let mut dg_cpu = Tensor::zeros(&[f]);
        let dx_cpu = cpu::rms_norm_backward(&dy, &fwd_cpu.x_hat, &fwd_cpu.inv_rms, &gamma, &mut dg_cpu, group);

        let dgamma_t = DTensor::from_host(&gpu, &gamma);
        let (out_gpu, fwd_gpu) = rms_norm_forward(&gpu, &DTensor::from_host(&gpu, &x), &dgamma_t, group, eps);
        let mut dg_gpu = DTensor::zeros(&gpu, &[f]);
        let dx_gpu = rms_norm_backward(&gpu, &DTensor::from_host(&gpu, &dy), &fwd_gpu, &dgamma_t, &mut dg_gpu, group);
        assert_close(&out_gpu.to_host(&gpu).data, &fwd_cpu.out.data);
        assert_close(&dx_gpu.to_host(&gpu).data, &dx_cpu.data);
        assert_close(&dg_gpu.to_host(&gpu).data, &dg_cpu.data);
    }

    #[test]
    fn softmax_ce_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else { return };
        use crate::nn2::loss;
        let (b, c) = (6, 9);
        let logits = Tensor::random(&[b, c], 2.0);
        let targets = [0usize, 8, 3, 5, 1, 7];
        let (loss_cpu, d_cpu) = loss::softmax_cross_entropy(&logits, &targets);
        let (loss_gpu, d_gpu) = softmax_cross_entropy(&gpu, &DTensor::from_host(&gpu, &logits), &targets);
        assert!((loss_cpu - loss_gpu).abs() < 1e-4, "loss {loss_cpu} vs {loss_gpu}");
        assert_close(&d_gpu.to_host(&gpu).data, &d_cpu.data);
    }

    #[test]
    fn adamw_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else { return };
        use crate::nn2::optim::{AdamCfg, AdamState};
        let n = 20;
        let param = Tensor::random(&[n], 1.0);
        let grad = Tensor::random(&[n], 1.0);
        let mut cfg = AdamCfg::new(1e-3, 0.01);
        cfg.t = 1;

        let mut param_cpu = param.clone();
        let mut st = AdamState::new();
        st.step(&mut param_cpu.data, &grad.data, &cfg, true);

        let mut p_gpu = DTensor::from_host(&gpu, &param);
        let mut m = DTensor::zeros(&gpu, &[n]);
        let mut v = DTensor::zeros(&gpu, &[n]);
        adamw(&gpu, &mut p_gpu, &DTensor::from_host(&gpu, &grad), &mut m, &mut v, &cfg, true);
        assert_close(&p_gpu.to_host(&gpu).data, &param_cpu.data);
    }
}
