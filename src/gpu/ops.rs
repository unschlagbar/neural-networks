//! GPU implementations of the backend op seam, operating on [`GTensor`]s.
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

use cudarc::cublas::sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};
use cudarc::cublas::{Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{CudaSlice, CudaView, LaunchConfig, PushKernelArg};

/// Launch geometry for a pure elementwise kernel over `n` items.
///
/// `LaunchConfig::for_num_elems` hardcodes 1024 threads per block, so anything with
/// `n <= 1024` runs as a SINGLE block — one multiprocessor busy and 83 idle. That is
/// the worst case for exactly the kernels that hit it: small casts, state copies and
/// the per-step sLSTM pointwise, all latency-bound, where the only lever is having
/// more SMs issuing loads at once.
///
/// Every kernel launched this way is elementwise — none use `__syncthreads`,
/// `__shfl` or shared memory — so the block width is free to choose. 256 is what they
/// want once there is enough work to fill the device; below that the width narrows
/// until the grid covers the SMs, never past a warp.
///
/// `GPU_WIDE_BLOCKS=1` restores the old fixed 1024 for an A/B.
pub(crate) fn elem_cfg(gpu: &Gpu, n: u32) -> LaunchConfig {
    use std::sync::OnceLock;
    static WIDE: OnceLock<bool> = OnceLock::new();
    if *WIDE.get_or_init(|| std::env::var("GPU_WIDE_BLOCKS").is_ok()) {
        return LaunchConfig::for_num_elems(n);
    }
    let sm = (gpu.sm_count as u32).max(1);
    let mut threads = 256u32;
    if n < sm * threads {
        threads = n.div_ceil(sm).next_power_of_two().clamp(32, 256);
    }
    LaunchConfig {
        grid_dim: (n.div_ceil(threads).max(1), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

use std::collections::HashMap;

use super::{GTensor, Gpu};
use crate::nn2::optim::AdamCfg;

/// Upload host indices (`usize`) as a `u32` device buffer for the gather /
/// scatter / CE kernels.
///
/// The kernels take `u32`, so a `usize` slice has to be narrowed into a
/// temporary first. Callers on a hot path build their ids as `u32` up front and
/// use the `*_u32` entry points below, which skip that copy entirely.
fn upload_ids(gpu: &Gpu, ids: &[usize]) -> CudaSlice<u32> {
    let u: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
    upload_ids_u32(gpu, &u)
}

/// Upload an already-narrowed id slice.
pub fn upload_ids_u32(gpu: &Gpu, ids: &[u32]) -> CudaSlice<u32> {
    gpu.stream.clone_htod(ids).expect("upload ids")
}

/// `C = A · B + beta·C` for row-major `A(M×K)`, `B(K×N)`, writing into an
/// existing `C(M×N)`. `beta = 0` overwrites, `beta = 1` accumulates (bias-seeded
/// forward). Uses cuBLAS via the operand-swap trick: cuBLAS computes column-major
/// `Cᵀ(N×M) = Bᵀ·Aᵀ`, which is exactly our row-major `C` in memory.
pub fn matmul_nn_into(
    gpu: &Gpu,
    a: &GTensor<f32>,
    b: &GTensor<f32>,
    c: &mut GTensor<f32>,
    beta: f32,
) {
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
pub fn matmul_nt_into(
    gpu: &Gpu,
    a: &GTensor<f32>,
    b: &GTensor<f32>,
    c: &mut GTensor<f32>,
    beta: f32,
) {
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
pub fn matmul_tn_into(
    gpu: &Gpu,
    a: &GTensor<f32>,
    b: &GTensor<f32>,
    c: &mut GTensor<f32>,
    beta: f32,
) {
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

// bf16 GEMM (`cublasGemmEx`, bf16 operands, fp32 accumulate)

/// Which transpose form a [`matmul_bf16_into`] call wants; the same three shapes
/// the fp32 wrappers above expose.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MmForm {
    /// `C = A · B` — forward.
    Nn,
    /// `C = A · Bᵀ` — input gradient (`dX = dY · Wᵀ`).
    Nt,
    /// `C = Aᵀ · B` — weight gradient (`dW += Xᵀ · dY`).
    Tn,
}

/// Whether GEMMs run with bf16 operands. Follows the same switch as slab storage
/// (`GPU_NO_BF16=1` disables) and additionally needs the cast kernels.
pub fn gemm_bf16_enabled(gpu: &Gpu) -> bool {
    super::bf16::active(gpu)
}

/// `C = op(A) · op(B) + beta·C` with **bf16 operands and an fp32 accumulator**,
/// via `cublasGemmEx`.
///
/// Narrowing the projections is **ours**, not the reference's. xLSTM-7B ships
/// `torch_dtype: float32` and its `autocast_kernel_dtype: bfloat16` casts the inputs
/// of the *mLSTM kernel* — `q, k, v, i, f` entering the chunkwise recurrence — not the
/// projection matmuls, which stay fp32. What is borrowed is only the shape of the
/// trade: narrow operands, `CUBLAS_COMPUTE_32F` accumulation, fp32 master weights.
///
/// Raw `result::gemm_ex` rather than cudarc's `Gemm<half::bf16>`, which does exist and
/// does reach `CUDA_R_16BF` + `CUBLAS_COMPUTE_32F`: the trait is homogeneous
/// (`GemmConfig<T>`, `C: DevicePtrMut<T>`), so it can only write a **bf16** `C`. The
/// point here is the wide accumulator landing in an fp32 `C`, which no `Gemm<T>` can
/// express.
///
/// bf16 rather than fp16 for the usual reason: it keeps fp32's 8 exponent bits, so
/// no activation or gradient here can overflow or flush that would not have in
/// fp32, and no loss scaling is needed. Only the multiplicand mantissas narrow to 8
/// bits — a strictly larger cut than the TF32 path in `gpu::set_tf32` (10 bits),
/// applied at the same place.
///
/// `a`/`b` are the bf16 operands, already narrowed by the caller (see
/// [`GemmBf16`]); `c` stays fp32, because it is both the accumulator's output type
/// and what every downstream op expects.
///
/// # Safety of the operand swap
///
/// Identical to the fp32 wrappers: cuBLAS is column-major, so we ask for `Cᵀ` by
/// passing `b` first and swapping `m`/`n`. The `ld*` values below therefore mirror
/// `matmul_{nn,nt,tn}_into` exactly — keep them in step if those ever change.
pub fn matmul_bf16_into(
    gpu: &Gpu,
    form: MmForm,
    a: &super::GTensor<u16>,
    b: &super::GTensor<u16>,
    c: &mut GTensor<f32>,
    beta: f32,
) {
    use cudarc::cublas::sys;
    use cudarc::driver::{DevicePtr, DevicePtrMut};

    let (m, ka, n, kb) = match form {
        MmForm::Nn => (a.dims()[0], a.dims()[1], b.dims()[1], b.dims()[0]),
        MmForm::Nt => (a.dims()[0], a.dims()[1], b.dims()[0], b.dims()[1]),
        MmForm::Tn => (a.dims()[1], a.dims()[0], b.dims()[1], b.dims()[0]),
    };
    assert_eq!(ka, kb, "matmul_bf16: inner dims {ka} != {kb}");
    assert_eq!(
        (c.rows(), c.cols()),
        (m, n),
        "matmul_bf16: C shape mismatch"
    );

    // Mirrors the fp32 wrappers' (transa, transb, lda, ldb) per form.
    let (transa, transb, lda, ldb) = match form {
        MmForm::Nn => (CUBLAS_OP_N, CUBLAS_OP_N, n, ka),
        MmForm::Nt => (CUBLAS_OP_T, CUBLAS_OP_N, ka, ka),
        MmForm::Tn => (CUBLAS_OP_N, CUBLAS_OP_T, n, m),
    };
    let (alpha, beta) = (1.0f32, beta);
    let (pa, _ra) = a.buf.device_ptr(&gpu.stream);
    let (pb, _rb) = b.buf.device_ptr(&gpu.stream);
    let (pc, _rc) = c.buf.device_ptr_mut(&gpu.stream);

    // SAFETY: the handle is live, the three pointers are device allocations of at
    // least the sizes the shape asserts above imply, and the leading dimensions are
    // the same ones the parity-tested fp32 wrappers use for this form.
    unsafe {
        cudarc::cublas::result::gemm_ex(
            *gpu.blas.handle(),
            transa,
            transb,
            n as i32, // swapped m/n: we are computing Cᵀ
            m as i32,
            ka as i32,
            (&alpha) as *const f32 as *const _,
            pb as *const _, // operand swap: B first
            sys::cudaDataType_t::CUDA_R_16BF,
            lda as i32,
            pa as *const _,
            sys::cudaDataType_t::CUDA_R_16BF,
            ldb as i32,
            (&beta) as *const f32 as *const _,
            pc as *mut _,
            sys::cudaDataType_t::CUDA_R_32F,
            n as i32,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    }
    .expect("cublas gemm_ex bf16");
}

/// `C = A·B + bias`, bf16 out, bias fused into the GEMM epilogue.
///
/// Saves the [`broadcast_row`] seed the legacy path needs (cuBLAS has no bias
/// argument) and the `SlabBuf::from_f32` narrowing pass a fp32 output would need.
/// Accumulation is still fp32, so the result is rounded once, at production.
///
/// A bf16 `bias` is what `F.linear` under `torch.autocast(bfloat16)` does: its
/// `gemm_and_bias` hands cuBLASLt the bias as `const Dtype*`, the operand type, and
/// sets `CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE` only in the fp8 `scaled_gemm`. cuBLASLt
/// itself would take an fp32 bias next to bf16 operands (`examples/bgrad_probe.rs`
/// runs one), and cudarc's `Matmul<T>` could not ask for it either way — but matching
/// the rounding is worth more than the half-ulp. Only [`MmForm::Nn`]. Operand swap as
/// in [`matmul_bf16_into`].
pub fn matmul_bf16_bias_into(
    gpu: &Gpu,
    a: &super::GTensor<u16>,
    b: &super::GTensor<u16>,
    bias: &super::GTensor<u16>,
    c: &mut super::GTensor<u16>,
) {
    use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};

    let (m, ka) = (a.dims()[0], a.dims()[1]);
    let (kb, n) = (b.dims()[0], b.dims()[1]);
    assert_eq!(ka, kb, "matmul_bf16_bias: inner dims {ka} != {kb}");
    assert_eq!(c.dims(), [m, n], "matmul_bf16_bias: C shape");
    assert_eq!(bias.len(), n, "matmul_bf16_bias: bias width");

    // Cᵀ = Bᵀ·Aᵀ, so `b` goes first and m/n swap. That puts the bias on the swapped
    // form's leading dimension, which is what cuBLASLt's row-wise bias expects.
    let cfg = MatmulConfig {
        transa: false,
        transb: false,
        m: n as u64,
        n: m as u64,
        k: ka as u64,
        alpha: 1.0,
        lda: n as i64,
        ldb: ka as i64,
        beta: 0.0,
        ldc: n as i64,
        stride_a: None,
        stride_b: None,
        stride_c: None,
        stride_bias: None,
        batch_size: None,
        transc: false,
    };

    // `GTensor<u16>` holds bf16 as `u16`; `Matmul<bf16>` wants `DevicePtr<bf16>`.
    //
    // SAFETY: both are plain 16-bit types with identical layout, and the slices are
    // device allocations of the sizes the shape asserts above imply.
    let (a_bf, b_bf, bias_bf) = unsafe {
        (
            a.buf
                .transmute::<half::bf16>(a.buf.len())
                .expect("bf16 view of a"),
            b.buf
                .transmute::<half::bf16>(b.buf.len())
                .expect("bf16 view of b"),
            bias.buf
                .transmute::<half::bf16>(bias.buf.len())
                .expect("bf16 view of bias"),
        )
    };
    let c_len = c.buf.len();
    let mut c_bf = unsafe {
        c.buf
            .transmute_mut::<half::bf16>(c_len)
            .expect("bf16 view of c")
    };

    // SAFETY: shapes and leading dimensions are asserted above; the buffers are live
    // allocations on `gpu.stream`, the stream the Lt handle was built on.
    unsafe {
        <CudaBlasLT as Matmul<half::bf16>>::matmul(
            &gpu.blas_lt,
            cfg,
            &b_bf,
            &a_bf,
            &mut c_bf,
            Some(&bias_bf),
            None,
        )
    }
    .expect("cublasLt matmul bf16 + bias");
}

/// Round `t` to bf16 precision in place, leaving it fp32.
///
/// Reproduces a cuBLAS accumulate with `Ctype = CUDA_R_16BF, beta = 1` — which is how
/// FlashRNN stores `dR` and `db` — for a buffer that has to stay fp32 because it is a
/// window into the fp32 [`ParamArena`](super::arena::ParamArena). See
/// `quantize_bf16_inplace` in `bf16_cast.cu` for why the two questions are separable.
pub fn quantize_bf16_(gpu: &Gpu, t: &mut GTensor<f32>) {
    let n = t.len();
    if n == 0 {
        return;
    }
    let f = gpu.kernels.get("quantize_bf16_inplace");
    let n_i = n as i32;
    let mut b = gpu.stream.launch_builder(&f);
    b.arg(&mut t.buf).arg(&n_i);
    unsafe { b.launch(elem_cfg(gpu, n as u32)) }.expect("quantize_bf16_inplace");
}


thread_local! {
    /// Resolved batched geometry per `(kernel, H, B)`. Choosing one costs an NVRTC
    /// lookup and a driver occupancy query per `rpt` candidate, and the encoder asks the
    /// same twenty questions every window — so the answer is remembered, including a
    /// remembered `None`.
    static BATCHED_GEOM: std::cell::RefCell<(Option<usize>, HashMap<(bool, usize, usize), Option<SbGeom>>)> =
        std::cell::RefCell::new((None, HashMap::new()));
}

/// Memoize `resolve` per `(bwd, h, b)` on this thread and stream.
fn cached_geom(
    gpu: &Gpu,
    bwd: bool,
    h: usize,
    b: usize,
    resolve: impl FnOnce() -> Option<SbGeom>,
) -> Option<SbGeom> {
    BATCHED_GEOM.with(|s| {
        let mut s = s.borrow_mut();
        let tag = std::sync::Arc::as_ptr(&gpu.stream) as usize;
        if s.0 != Some(tag) {
            s.1.clear();
            s.0 = Some(tag);
        }
        *s.1.entry((bwd, h, b)).or_insert_with(resolve)
    })
}








/// Reusable bf16 staging for a layer's GEMM operands.
///
/// [`matmul_bf16_into`] needs both operands already in bf16, but the tensors a
/// layer holds are fp32 (the optimizer steps them there, and the checkpoint stores
/// them there). This owns the narrowed copies and refills them per call, so the
/// conversion does not allocate on the hot path.
///
/// Keeping the fp32 master and narrowing per use is deliberate — it is what every
/// mixed-precision framework does, and what makes this safe: the weights that
/// accumulate small updates never lose precision, only the transient operands the
/// tensor cores read.
#[derive(Default)]
pub struct GemmBf16 {
    lhs: Option<super::GTensor<u16>>,
    rhs: Option<super::GTensor<u16>>,
    /// Narrowed copy of the layer's weight, reused until [`invalidate_w`](Self::invalidate_w).
    ///
    /// The weight is the one operand that does not change between GEMMs: a layer
    /// narrows it for its forward (`Y = X·W`) and again for `dX = dY·Wᵀ`, and only an
    /// optimizer step writes it in between. Re-narrowing it each time made
    /// `cast_f32_to_bf16` the single largest kernel on the profile (17% of GPU time).
    w: Option<super::GTensor<u16>>,
    /// Whether [`w`](Self::w) currently matches the fp32 weight.
    w_valid: bool,
    /// Narrowed copy of the layer's bias, for the epilogue path. Invalidated with
    /// `w` — an optimizer step writes both.
    b: Option<super::GTensor<u16>>,
}

/// Whether the bf16 weight cache is on (`GPU_NO_WCACHE=1` turns it off).
///
/// The cache costs one bf16 copy of every weight — half the weight bytes, ~273 MB at
/// the 240M config — in exchange for ~4% of step time. It scales with the parameter
/// count, not the window, so a much larger model can reclaim it here.
fn wcache_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("GPU_NO_WCACHE").as_deref() != Ok("1"))
}

impl GemmBf16 {
    pub const fn new() -> Self {
        Self {
            lhs: None,
            rhs: None,
            w: None,
            w_valid: false,
            b: None,
        }
    }

    /// Mark the cached bf16 weight stale. **Must** be called by anything that writes
    /// the fp32 weight — an optimizer step, a checkpoint load, a manual overwrite.
    /// Missing a call means the GEMMs keep reading a stale weight, which shows up as
    /// training silently not learning rather than as an error.
    pub fn invalidate_w(&mut self) {
        self.w_valid = false;
        self.b = None;
    }

    /// Narrow `w` once and reuse it until invalidated. See [`w`](Self::w).
    fn stage_w(&mut self, gpu: &Gpu, src: &GTensor<f32>) -> &super::GTensor<u16> {
        // A hit reuses the buffer *at the dims it was narrowed with*, so it is only
        // sound while the weight's shape is fixed — which it is: a layer's `w` is
        // allocated once at `[in, out]` and only ever stepped in place. A shape change
        // therefore means a different tensor, and is treated as a miss.
        let same_shape = self.w.as_ref().is_some_and(|t| t.dims() == src.dims());
        if !(self.w_valid && same_shape) {
            let n = src.len();
            match &mut self.w {
                Some(t) if t.capacity() >= n => t.shrink_to(src.dims()),
                _ => self.w = Some(super::GTensor::uninit(gpu, src.dims())),
            }
            self.w.as_mut().expect("just filled").store(gpu, src);
            self.w_valid = true;
        }
        self.w.as_ref().expect("staged")
    }

    /// Device bytes the staging buffers hold (2 bytes/element). Diagnostic.
    ///
    /// Counts **capacity**: `stage` reuses whenever the existing buffer is large
    /// enough, so these grow to the largest operand this layer has ever narrowed and
    /// stay there.
    pub fn retained_bytes(&self) -> usize {
        [&self.lhs, &self.rhs, &self.w]
            .iter()
            .filter_map(|s| s.as_ref())
            .map(|t| t.capacity() * 2)
            .sum()
    }

    /// Drop the staging buffers. The next `run` reallocates them.
    pub fn clear(&mut self) {
        self.clear_operands();
        self.w = None;
        self.w_valid = false;
        self.b = None;
    }

    /// Drop only the buffers sized to the *call* — the narrowed activations. The
    /// weight cache is sized to the weight, so it neither grows with a rectangle nor
    /// goes stale between two calls that step nothing: dropping it at a group boundary
    /// gives back parameter-sized memory the layer will want again immediately, and
    /// buys a re-narrow of every weight in the stack per group.
    pub fn clear_operands(&mut self) {
        self.lhs = None;
        self.rhs = None;
    }

    /// `Y = X·W + b` with a bf16 `Y`, bias in the epilogue. `x_b` is the caller's
    /// shared narrowed input; the weight and bias come from this cache.
    pub fn run_staged_lhs_bias(
        &mut self,
        gpu: &Gpu,
        x_b: &super::GTensor<u16>,
        w: &GTensor<f32>,
        b: &GTensor<f32>,
        y: &mut super::GTensor<u16>,
    ) {
        let stale = self.b.as_ref().is_none_or(|t| t.dims() != b.dims());
        if stale {
            let mut nb = super::GTensor::uninit(gpu, b.dims());
            nb.store(gpu, b);
            self.b = Some(nb);
        }
        // Not through `stage_w`: that returns a borrow of `self`, which would still be
        // live when the bias is read below.
        self.stage_w(gpu, w);
        let rhs = self.w.as_ref().expect("staged");
        let bias = self.b.as_ref().expect("staged");
        matmul_bf16_bias_into(gpu, x_b, rhs, bias, y);
    }

    /// Narrow `a` and `b` into the owned staging buffers and run the GEMM.
    pub fn run(
        &mut self,
        gpu: &Gpu,
        form: MmForm,
        a: &GTensor<f32>,
        b: &GTensor<f32>,
        c: &mut GTensor<f32>,
        beta: f32,
    ) {
        fn stage<'s>(
            gpu: &Gpu,
            slot: &'s mut Option<super::GTensor<u16>>,
            src: &GTensor<f32>,
        ) -> &'s super::GTensor<u16> {
            let n = src.len();
            match slot {
                Some(t) if t.capacity() >= n => t.shrink_to(src.dims()),
                _ => *slot = Some(super::GTensor::uninit(gpu, src.dims())),
            }
            let t = slot.as_mut().expect("just filled");
            t.store(gpu, src);
            t
        }
        // Two separate slots so the borrows do not overlap.
        stage(gpu, &mut self.lhs, a);
        stage(gpu, &mut self.rhs, b);
        let lhs = self.lhs.as_ref().expect("staged");
        let rhs = self.rhs.as_ref().expect("staged");
        matmul_bf16_into(gpu, form, lhs, rhs, c, beta);
    }

    /// [`run`](Self::run) where the **right** operand is the layer's weight, taken from
    /// the cache instead of being narrowed again.
    pub fn run_wb(
        &mut self,
        gpu: &Gpu,
        form: MmForm,
        a: &GTensor<f32>,
        w: &GTensor<f32>,
        c: &mut GTensor<f32>,
        beta: f32,
    ) {
        let n = a.len();
        match &mut self.lhs {
            Some(t) if t.capacity() >= n => t.shrink_to(a.dims()),
            _ => self.lhs = Some(super::GTensor::uninit(gpu, a.dims())),
        }
        self.lhs.as_mut().expect("just filled").store(gpu, a);
        if !wcache_enabled() {
            // Narrow through the shared `rhs` slot, so the dedicated cache buffer is
            // never allocated — the point of the switch is to give the memory back,
            // and keeping the buffer while skipping only the cast would not.
            let n2 = w.len();
            match &mut self.rhs {
                Some(t) if t.capacity() >= n2 => t.shrink_to(w.dims()),
                _ => self.rhs = Some(super::GTensor::uninit(gpu, w.dims())),
            }
            self.rhs.as_mut().expect("just filled").store(gpu, w);
            let lhs = self.lhs.as_ref().expect("staged");
            let rhs = self.rhs.as_ref().expect("staged");
            matmul_bf16_into(gpu, form, lhs, rhs, c, beta);
            return;
        }
        self.stage_w(gpu, w);
        let lhs = self.lhs.as_ref().expect("staged");
        let rhs = self.w.as_ref().expect("staged");
        matmul_bf16_into(gpu, form, lhs, rhs, c, beta);
    }

    /// `Y = X·W` where `X` is **already narrowed** to bf16 by the caller, so this
    /// launches no cast for it. The weight still comes from the cache.
    pub fn run_staged_lhs(
        &mut self,
        gpu: &Gpu,
        form: MmForm,
        lhs: &super::GTensor<u16>,
        w: &GTensor<f32>,
        c: &mut GTensor<f32>,
        beta: f32,
    ) {
        if !wcache_enabled() {
            let n2 = w.len();
            match &mut self.rhs {
                Some(t) if t.capacity() >= n2 => t.shrink_to(w.dims()),
                _ => self.rhs = Some(super::GTensor::uninit(gpu, w.dims())),
            }
            self.rhs.as_mut().expect("just filled").store(gpu, w);
            let rhs = self.rhs.as_ref().expect("staged");
            matmul_bf16_into(gpu, form, lhs, rhs, c, beta);
            return;
        }
        self.stage_w(gpu, w);
        let rhs = self.w.as_ref().expect("staged");
        matmul_bf16_into(gpu, form, lhs, rhs, c, beta);
    }

    /// [`run_backward`](Self::run_backward) where `x` is already narrowed into a
    /// bf16 by the caller — only `dy` is cast here.
    pub fn run_backward_staged_x(
        &mut self,
        gpu: &Gpu,
        x: &super::GTensor<u16>,
        dy: &GTensor<f32>,
        w: &GTensor<f32>,
        dw: &mut GTensor<f32>,
        dx: &mut GTensor<f32>,
    ) {
        let n = dy.len();
        match &mut self.rhs {
            Some(t) if t.capacity() >= n => t.shrink_to(dy.dims()),
            _ => self.rhs = Some(super::GTensor::uninit(gpu, dy.dims())),
        }
        self.rhs.as_mut().expect("just filled").store(gpu, dy);
        {
            let rhs = self.rhs.as_ref().expect("staged");
            matmul_bf16_into(gpu, MmForm::Tn, x, rhs, dw, 1.0);
        }
        // `dX = dY·Wᵀ`. `lhs` is free here — the shared `x` lives outside this struct.
        if !wcache_enabled() {
            let n2 = w.len();
            match &mut self.lhs {
                Some(t) if t.capacity() >= n2 => t.shrink_to(w.dims()),
                _ => self.lhs = Some(super::GTensor::uninit(gpu, w.dims())),
            }
            self.lhs.as_mut().expect("just filled").store(gpu, w);
        } else {
            self.stage_w(gpu, w);
        }
        let dy_b = self.rhs.as_ref().expect("staged");
        let w_b = if wcache_enabled() {
            self.w.as_ref().expect("staged")
        } else {
            self.lhs.as_ref().expect("staged")
        };
        matmul_bf16_into(gpu, MmForm::Nt, dy_b, w_b, dx, 0.0);
    }

    /// The sLSTM cell's THREE post-loop GEMMs, all driven from one narrowed copy of
    /// the gate deltas: `dWx += xᵀ·dg`, `dx = dg·Wxᵀ` and `dWh += h_prevᵀ·dg`.
    ///
    /// A `Linear` has only the first two, which is why this is not
    /// [`run_backward_staged_x`](Self::run_backward_staged_x): the recurrent half adds
    /// a third consumer of the same deltas. It used to sit outside this cache
    /// entirely — widening the bf16 `h_prev` slab back to fp32 and running an fp32
    /// SIMT GEMM — which at the backbone's shape cost 90 us against the 47 + 29 us of
    /// the two tensor-core GEMMs beside it.
    ///
    /// Both left operands arrive already narrowed: `x` is the cell's saved input slab
    /// and `h_prev` one of its forward caches, so neither costs a cast here.
    #[allow(clippy::too_many_arguments)]
    pub fn run_slstm_backward(
        &mut self,
        gpu: &Gpu,
        x: &super::GTensor<u16>,
        h_prev: &super::GTensor<u16>,
        dg: &GTensor<f32>,
        wx: &GTensor<f32>,
        dwx: &mut GTensor<f32>,
        dwhr: &mut GTensor<f32>,
        dx: &mut GTensor<f32>,
    ) {
        self.run_backward_staged_x(gpu, x, dg, wx, dwx, dx);
        // That left the narrowed gate deltas in the staging slot — the weight goes to
        // `w` or `lhs`, never `rhs` — so the recurrent grad reads the same copy.
        let dg_b = self.rhs.as_ref().expect("staged by run_backward_staged_x");
        matmul_bf16_into(gpu, MmForm::Tn, h_prev, dg_b, dwhr, 1.0);
    }

    /// A Linear's two backward GEMMs, `dW += Xᵀ·dY` and `dX = dY·Wᵀ`, driven from a
    /// **single** narrowed `dy`.
    ///
    /// Running them as separate `run`/`run_wb` calls narrows `dy` twice — the same
    /// values, through two different slots. The cast is launch-bound at these shapes
    /// (two thirds of them are under 1.5 µs), so the duplicate is nearly all overhead.
    pub fn run_backward(
        &mut self,
        gpu: &Gpu,
        x: &GTensor<f32>,
        dy: &GTensor<f32>,
        w: &GTensor<f32>,
        dw: &mut GTensor<f32>,
        dx: &mut GTensor<f32>,
    ) {
        // `dy` is the shared operand: right for `Tn`, left for `Nt`. It lives in `rhs`
        // so `lhs` stays free for `x`.
        let n = dy.len();
        match &mut self.rhs {
            Some(t) if t.capacity() >= n => t.shrink_to(dy.dims()),
            _ => self.rhs = Some(super::GTensor::uninit(gpu, dy.dims())),
        }
        self.rhs.as_mut().expect("just filled").store(gpu, dy);

        let nx = x.len();
        match &mut self.lhs {
            Some(t) if t.capacity() >= nx => t.shrink_to(x.dims()),
            _ => self.lhs = Some(super::GTensor::uninit(gpu, x.dims())),
        }
        self.lhs.as_mut().expect("just filled").store(gpu, x);

        {
            let lhs = self.lhs.as_ref().expect("staged");
            let rhs = self.rhs.as_ref().expect("staged");
            matmul_bf16_into(gpu, MmForm::Tn, lhs, rhs, dw, 1.0);
        }

        // `dX = dY·Wᵀ`. Without the weight cache the weight has to go somewhere, and
        // `lhs` (holding `x`, already consumed above) is the free slot.
        if !wcache_enabled() {
            let n2 = w.len();
            match &mut self.lhs {
                Some(t) if t.capacity() >= n2 => t.shrink_to(w.dims()),
                _ => self.lhs = Some(super::GTensor::uninit(gpu, w.dims())),
            }
            self.lhs.as_mut().expect("just filled").store(gpu, w);
        } else {
            self.stage_w(gpu, w);
        }
        let dy_b = self.rhs.as_ref().expect("staged");
        let w_b = if wcache_enabled() {
            self.w.as_ref().expect("staged")
        } else {
            self.lhs.as_ref().expect("staged")
        };
        matmul_bf16_into(gpu, MmForm::Nt, dy_b, w_b, dx, 0.0);
    }
}

/// `C = A · B` (fresh allocation). Convenience wrapper over [`matmul_nn_into`].
pub fn matmul(gpu: &Gpu, a: &GTensor<f32>, b: &GTensor<f32>) -> GTensor<f32> {
    let mut c = GTensor::uninit(gpu, &[a.rows(), b.cols()]);
    matmul_nn_into(gpu, a, b, &mut c, 0.0);
    c
}

/// `C = A · Bᵀ` (fresh allocation). Convenience wrapper over [`matmul_nt_into`].
pub fn matmul_nt(gpu: &Gpu, a: &GTensor<f32>, b: &GTensor<f32>) -> GTensor<f32> {
    let mut c = GTensor::uninit(gpu, &[a.rows(), b.rows()]);
    matmul_nt_into(gpu, a, b, &mut c, 0.0);
    c
}

/// `C = Aᵀ · B` (fresh allocation). Convenience wrapper over [`matmul_tn_into`].
pub fn matmul_tn(gpu: &Gpu, a: &GTensor<f32>, b: &GTensor<f32>) -> GTensor<f32> {
    let mut c = GTensor::uninit(gpu, &[a.cols(), b.cols()]);
    matmul_tn_into(gpu, a, b, &mut c, 0.0);
    c
}

// Strided-batched GEMM (per-(batch,head) small matmuls for chunkwise mLSTM).
// Rank-3 device tensors `[batch, ·, ·]`, contiguous → per-batch stride is the
// matrix element count. Same row-major-over-column-major operand-swap as the
// single GEMMs (pass B first, A second, compute Cᵀ), applied per batch.

/// Whether the batched GEMMs round their operands to TF32 (`GPU_NO_BATCHED_TF32=1`
/// turns it off).
///
/// On by default, unlike the handle-wide [`gpu::set_tf32`](super::set_tf32), and the
/// difference is which GEMMs each one reaches. These three serve only the chunkwise
/// mLSTM, whose *other* dots — the fused kernels' `mma.sync...f32.tf32.tf32.f32` —
/// are already TF32 by default, matching the reference (`mlstm_kernels`, where
/// Triton's `tl.dot` is TF32 on fp32 inputs). Leaving these on the CUDA cores made
/// one half of the same algorithm fp32 and the other TF32 for no stated reason.
///
/// The handle-wide switch stays opt-in because it also reaches the GEMMs the
/// `gemm_*_matches_cpu` tests use as an exact-fp32 oracle. These wrappers are not on
/// that path, so the oracle survives.
pub fn batched_tf32() -> bool {
    std::env::var("GPU_NO_BATCHED_TF32").as_deref() != Ok("1")
}

/// Strided-batched `Cᵀ = op(B)·op(A)` in fp32, optionally on the TF32 tensor cores.
///
/// The caller passes the already-swapped operands and leading dimensions — this only
/// chooses the compute type, so the layouts below stay identical to what the fp32
/// path used. `CUBLAS_COMPUTE_32F_FAST_TF32` keeps the buffers, the accumulator and
/// the result fp32 and narrows only the multiplicand mantissas to 10 bits.
#[allow(clippy::too_many_arguments)]
fn batched_gemm(
    gpu: &Gpu,
    cfg: StridedBatchedConfig<f32>,
    b: &GTensor<f32>,
    a: &GTensor<f32>,
    c: &mut GTensor<f32>,
    what: &str,
) {
    if !batched_tf32() {
        // SAFETY: shapes and leading dimensions are the caller's, which the shape
        // asserts above validate; all three buffers are live device allocations.
        unsafe {
            gpu.blas
                .gemm_strided_batched(cfg, &b.buf, &a.buf, &mut c.buf)
        }
        .unwrap_or_else(|e| panic!("cublas gemm_strided_batched {what}: {e:?}"));
        return;
    }

    use cudarc::cublas::sys;
    use cudarc::driver::{DevicePtr, DevicePtrMut};

    let g = cfg.gemm;
    let (pb, _rb) = b.buf.device_ptr(&gpu.stream);
    let (pa, _ra) = a.buf.device_ptr(&gpu.stream);
    let (pc, _rc) = c.buf.device_ptr_mut(&gpu.stream);

    // SAFETY: same operands, leading dimensions and strides the typed wrapper would
    // pass; only the compute type differs.
    unsafe {
        cudarc::cublas::result::gemm_strided_batched_ex(
            *gpu.blas.handle(),
            g.transa,
            g.transb,
            g.m,
            g.n,
            g.k,
            (&g.alpha) as *const f32 as *const _,
            pb as *const _,
            sys::cudaDataType_t::CUDA_R_32F,
            g.lda,
            cfg.stride_a,
            pa as *const _,
            sys::cudaDataType_t::CUDA_R_32F,
            g.ldb,
            cfg.stride_b,
            (&g.beta) as *const f32 as *const _,
            pc as *mut _,
            sys::cudaDataType_t::CUDA_R_32F,
            g.ldc,
            cfg.stride_c,
            cfg.batch_size,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    }
    .unwrap_or_else(|e| panic!("cublas gemm_strided_batched_ex {what}: {e:?}"));
}

/// `C[g] = A[g] · B[g]` for `A[batch,M,K]`, `B[batch,K,N]` → `C[batch,M,N]`.
pub fn matmul_batched_nn(gpu: &Gpu, a: &GTensor<f32>, b: &GTensor<f32>) -> GTensor<f32> {
    let (batch, m, ka) = (a.shape[0], a.shape[1], a.shape[2]);
    let (kb, n) = (b.shape[1], b.shape[2]);
    assert_eq!(ka, kb, "matmul_batched_nn: inner dims {ka} != {kb}");
    assert_eq!(batch, b.shape[0], "matmul_batched_nn: batch mismatch");
    let mut c = GTensor::uninit(gpu, &[batch, m, n]);
    let gemm = GemmConfig {
        transa: CUBLAS_OP_N,
        transb: CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: ka as i32,
        alpha: 1.0,
        lda: n as i32,
        ldb: ka as i32,
        beta: 0.0,
        ldc: n as i32,
    };
    let cfg = StridedBatchedConfig {
        gemm,
        batch_size: batch as i32,
        stride_a: (kb * n) as i64,
        stride_b: (m * ka) as i64,
        stride_c: (m * n) as i64,
    };
    batched_gemm(gpu, cfg, b, a, &mut c, "nn");
    c
}

/// `C[g] = A[g] · B[g]ᵀ` for `A[batch,M,K]`, `B[batch,N,K]` → `C[batch,M,N]`.
pub fn matmul_batched_nt(gpu: &Gpu, a: &GTensor<f32>, b: &GTensor<f32>) -> GTensor<f32> {
    let (batch, m, ka) = (a.shape[0], a.shape[1], a.shape[2]);
    let (n, kb) = (b.shape[1], b.shape[2]);
    assert_eq!(ka, kb, "matmul_batched_nt: inner dims {ka} != {kb}");
    assert_eq!(batch, b.shape[0], "matmul_batched_nt: batch mismatch");
    let mut c = GTensor::uninit(gpu, &[batch, m, n]);
    let gemm = GemmConfig {
        transa: CUBLAS_OP_T,
        transb: CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: ka as i32,
        alpha: 1.0,
        lda: ka as i32,
        ldb: ka as i32,
        beta: 0.0,
        ldc: n as i32,
    };
    let cfg = StridedBatchedConfig {
        gemm,
        batch_size: batch as i32,
        stride_a: (n * kb) as i64,
        stride_b: (m * ka) as i64,
        stride_c: (m * n) as i64,
    };
    batched_gemm(gpu, cfg, b, a, &mut c, "nt");
    c
}

/// `C[g] = A[g]ᵀ · B[g]` for `A[batch,K,M]`, `B[batch,K,N]` → `C[batch,M,N]`.
pub fn matmul_batched_tn(gpu: &Gpu, a: &GTensor<f32>, b: &GTensor<f32>) -> GTensor<f32> {
    let (batch, ka, m) = (a.shape[0], a.shape[1], a.shape[2]);
    let (kb, n) = (b.shape[1], b.shape[2]);
    assert_eq!(ka, kb, "matmul_batched_tn: outer dims {ka} != {kb}");
    assert_eq!(batch, b.shape[0], "matmul_batched_tn: batch mismatch");
    let mut c = GTensor::uninit(gpu, &[batch, m, n]);
    let gemm = GemmConfig {
        transa: CUBLAS_OP_N,
        transb: CUBLAS_OP_T,
        m: n as i32,
        n: m as i32,
        k: ka as i32,
        alpha: 1.0,
        lda: n as i32,
        ldb: m as i32,
        beta: 0.0,
        ldc: n as i32,
    };
    let cfg = StridedBatchedConfig {
        gemm,
        batch_size: batch as i32,
        stride_a: (kb * n) as i64,
        stride_b: (ka * m) as i64,
        stride_c: (m * n) as i64,
    };
    batched_gemm(gpu, cfg, b, a, &mut c, "tn");
    c
}

// Elementwise / reduction / gather (NVRTC kernels, see gpu/kernels.rs)

/// In-place scale: `x *= s`. The mLSTM k-projection's `1/√dqk`.
pub fn scale_(gpu: &Gpu, x: &mut GTensor<f32>, s: f32) {
    let n = x.len();
    let n_i = n as i32;
    let f = gpu.kernels.get("scale_inplace");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut x.buf).arg(&s).arg(&n_i);
    unsafe { lb.launch(elem_cfg(gpu, n as u32)) }.expect("scale_inplace");
}

/// In-place numerically-stable sigmoid. The mLSTM o-gate projection.
pub fn sigmoid_(gpu: &Gpu, x: &mut GTensor<f32>) {
    let n = x.len();
    let n_i = n as i32;
    let f = gpu.kernels.get("sigmoid_inplace");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut x.buf).arg(&n_i);
    unsafe { lb.launch(elem_cfg(gpu, n as u32)) }.expect("sigmoid_inplace");
}

/// SoftCap forward: `y = cap · tanh(x / cap)`.
pub fn softcap_forward(gpu: &Gpu, x: &GTensor<f32>, cap: f32) -> GTensor<f32> {
    let n = x.len();
    let n_i = n as i32;
    let mut y = GTensor::uninit(gpu, x.dims());
    let f = gpu.kernels.get("softcap_forward");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&x.buf).arg(&mut y.buf).arg(&cap).arg(&n_i);
    unsafe { lb.launch(elem_cfg(gpu, n as u32)) }.expect("softcap_forward");
    y
}

/// SoftCap backward: `dx = dy · (1 − (y/cap)²)`, using the saved output `y`.
pub fn softcap_backward(gpu: &Gpu, dy: &GTensor<f32>, y: &GTensor<f32>, cap: f32) -> GTensor<f32> {
    let n = dy.len();
    let n_i = n as i32;
    let mut dx = GTensor::uninit(gpu, dy.dims());
    let f = gpu.kernels.get("softcap_backward");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&dy.buf)
        .arg(&y.buf)
        .arg(&mut dx.buf)
        .arg(&cap)
        .arg(&n_i);
    unsafe { lb.launch(elem_cfg(gpu, n as u32)) }.expect("softcap_backward");
    dx
}

/// Copy `bias` (`[N]`) into every row of `out` (`[rows, N]`).
pub fn broadcast_row(gpu: &Gpu, out: &mut GTensor<f32>, bias: &GTensor<f32>) {
    let (rows, n) = (out.rows(), out.cols());
    let (rows_i, n_i) = (rows as i32, n as i32);
    let f = gpu.kernels.get("broadcast_row");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut out.buf).arg(&bias.buf).arg(&rows_i).arg(&n_i);
    // One block per row, grid-strided so a tall output does not need a huge grid.
    // Threads cover the width, which is where the coalescing is.
    let threads = n.clamp(32, 256).next_power_of_two().min(1024) as u32;
    let blocks = rows.min(65535).max(1) as u32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { lb.launch(cfg) }.expect("broadcast_row");
}

/// [`broadcast_row`] with a residual folded in: `out = resid + bias`, so a projection
/// feeding a residual seeds its output with the sum and the trailing add needs no
/// kernel. `resid` and `out` may not alias.
pub fn broadcast_row_resid(
    gpu: &Gpu,
    out: &mut GTensor<f32>,
    resid: &GTensor<f32>,
    bias: &GTensor<f32>,
) {
    let (rows, n) = (out.rows(), out.cols());
    assert_eq!(
        resid.dims(),
        out.dims(),
        "broadcast_row_resid — residual shape"
    );
    assert_eq!(bias.len(), n, "broadcast_row_resid — bias width");
    let (rows_i, n_i) = (rows as i32, n as i32);
    let f = gpu.kernels.get("broadcast_row_resid");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut out.buf)
        .arg(&resid.buf)
        .arg(&bias.buf)
        .arg(&rows_i)
        .arg(&n_i);
    let threads = n.clamp(32, 256).next_power_of_two().min(1024) as u32;
    let blocks = rows.min(65535).max(1) as u32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { lb.launch(cfg) }.expect("broadcast_row_resid");
}

/// Accumulate the column sum of `dy` (`[rows, N]`) into `db` (`[N]`) — the bias
/// gradient.
pub fn add_col_sum(
    gpu: &Gpu,
    db: &mut GTensor<f32>,
    dy: &GTensor<f32>,
    cache: &super::temp::TempCache,
) {
    col_sum_into(gpu, db, dy, None, None, cache);
}

/// [`add_col_sum`] over the elementwise product `dy ⊙ mul` — RMSNorm's `dgamma`,
/// which is the same reduction with one more operand.
pub fn add_col_sum_mul(
    gpu: &Gpu,
    db: &mut GTensor<f32>,
    dy: &GTensor<f32>,
    mul: &GTensor<f32>,
    cache: &super::temp::TempCache,
) {
    assert_eq!(dy.len(), mul.len(), "add_col_sum_mul: operand sizes");
    col_sum_into(gpu, db, dy, Some(WideOrSlab::F32(mul)), None, cache);
}

/// [`add_col_sum_mul`] with each column divided by `div[o]` after the reduction.
///
/// RMSNorm's `dγ` when backward rebuilds `x̂` from the norm's output: `x̂ = y/γ` and γ
/// is constant down a column, so `Σ_r dy·x̂ = (Σ_r dy·y)/γ` and the divide leaves the
/// inner loop. That is what lets the output-sourced path run without materializing
/// `x̂` anywhere. See [`crate::gpu::rms_norm::XHat`].
pub fn add_col_sum_mul_div(
    gpu: &Gpu,
    db: &mut GTensor<f32>,
    dy: &GTensor<f32>,
    mul: WideOrSlab<'_>,
    div: &GTensor<f32>,
    cache: &super::temp::TempCache,
) {
    assert_eq!(dy.as_2d(), mul.as_2d(), "add_col_sum_mul_div: operand shapes");
    assert_eq!(db.len(), div.len(), "add_col_sum_mul_div: divisor width");
    col_sum_into(gpu, db, dy, Some(mul), Some(div), cache);
}

/// `db[o] += Σ_r dy[r, o]` (times `mul[r, o]` if given), **deterministically**.
///
/// A block owns a column tile and a **band** of its rows, folding the band through
/// `threadIdx.y` and a fixed-order tree; `col_sum_merge` then folds the bands in
/// ascending order. Both splits are functions of the shape, never of scheduling —
/// float addition does not associate, so an `atomicAdd` across blocks would make the
/// last bits of every bias gradient depend on the order the blocks happened to run
/// in, and one optimizer step later that is a different model.
///
/// A single band (the whole row axis in one block) leaves the grid at `ceil(n / 32)`,
/// which at these layer widths is 8–24 blocks and reads at a tenth of the machine's
/// bandwidth. [`col_sum_bands`] decides when the second launch is worth paying for.

/// An operand that may be stored fp32 or narrow, borrowed for one launch.
///
/// RMSNorm's output is read by three different kernels (its own backward, the `dγ`
/// reduction, and whatever consumes it downstream), and once the forward is allowed to
/// write it narrow they all have to follow. This is what lets one launcher serve both
/// widths instead of a parallel set of functions per width.
#[derive(Clone, Copy)]
pub enum WideOrSlab<'a> {
    F32(&'a GTensor<f32>),
    Slab(&'a SlabBuf),
}

impl<'a> WideOrSlab<'a> {
    pub fn as_2d(&self) -> (usize, usize) {
        match self {
            WideOrSlab::F32(t) => t.as_2d(),
            WideOrSlab::Slab(SlabBuf::F32(t)) => t.as_2d(),
            WideOrSlab::Slab(SlabBuf::Bf16(t)) => t.as_2d(),
        }
    }

    /// Pick between a kernel's fp32 and `_slab` entry points.
    ///
    /// `SlabBuf::F32` takes the fp32 one: a slab in a build without bf16 IS an fp32
    /// tensor, and the narrow entry point would be the same kernel under a second
    /// name. Both names are `&'static str` so the choice costs no allocation on a
    /// path that runs per launch.
    fn pick(&self, wide: &'static str, narrow: &'static str) -> &'static str {
        match self {
            WideOrSlab::Slab(SlabBuf::Bf16(_)) => narrow,
            _ => wide,
        }
    }
}

/// Push a [`WideOrSlab`] as the next kernel argument.
macro_rules! push_wos {
    ($lb:expr, $v:expr) => {
        match $v {
            WideOrSlab::F32(t) => $lb.arg(&t.buf),
            WideOrSlab::Slab(SlabBuf::F32(t)) => $lb.arg(&t.buf),
            WideOrSlab::Slab(SlabBuf::Bf16(t)) => $lb.arg(&t.buf),
        }
    };
}

fn col_sum_into(
    gpu: &Gpu,
    db: &mut GTensor<f32>,
    dy: &GTensor<f32>,
    mul: Option<WideOrSlab<'_>>,
    div: Option<&GTensor<f32>>,
    cache: &super::temp::TempCache,
) {
    // `as_2d`, not `rows()`/`cols()`: RMSNorm hands this a `[B, T, d]` activation.
    let (rows, n) = dy.as_2d();
    assert_eq!(db.len(), n, "col_sum: db width");
    let (rows_i, n_i) = (rows as i32, n as i32);
    // A warp wants adjacent columns, so the block is as wide as the layer up to 32;
    // a narrower layer spends the freed threads on rows instead. Both extents stay
    // powers of two, which the kernel's reduction tree requires.
    const THREADS: usize = 512;
    let bx = n.next_power_of_two().min(32).max(1);
    let by = (THREADS / bx).min(rows.next_power_of_two()).max(1);
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(bx) as u32, 1, 1),
        block_dim: (bx as u32, by as u32, 1),
        shared_mem_bytes: (bx * by * std::mem::size_of::<f32>()) as u32,
    };
    // No second operand: hand the kernel `dy` again rather than a null it would have
    // to test per element. The divisor stands in the same way — `dy` is at least `n`
    // wide, so the unused read stays in bounds.
    let use_mul = mul.is_some() as i32;
    let mul = mul.unwrap_or(WideOrSlab::F32(dy));
    let use_div = div.is_some() as i32;
    let div = div.unwrap_or(dy);

    let bands = col_sum_bands(gpu, rows, by, cfg.grid_dim.0 as usize);
    if bands > 1 {
        col_sum_banded(
            gpu, db, dy, mul, use_mul, div, use_div, rows, n, bx, by, cfg, bands, cache,
        );
        return;
    }
    let f = gpu.kernels.get(mul.pick("add_col_sum", "add_col_sum_slab"));
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut db.buf).arg(&dy.buf);
    push_wos!(lb, mul);
    lb.arg(&use_mul)
        .arg(&div.buf)
        .arg(&use_div)
        .arg(&rows_i)
        .arg(&n_i);
    unsafe { lb.launch(cfg) }.expect("add_col_sum");
}

/// How many row bands to cut the reduction into — see `col_sum_part`.
///
/// One block per column tile leaves the grid at `ceil(n / 32)`, which at these layer
/// widths is a handful of blocks on an 84-SM part. Bands trade a second (tiny) launch
/// for a grid that fills the machine, so the split is worth it only once there are
/// enough rows to go round: each band still wants a few rows per `threadIdx.y`, or the
/// bands are pure overhead.
///
/// Depends on `rows`, the block shape and the device — never on scheduling — so the
/// summation order stays a property of the shape.
fn col_sum_bands(gpu: &Gpu, rows: usize, by: usize, grid_x: usize) -> usize {
    const ROWS_PER_THREAD: usize = 4;
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("GPU_NO_COLSUM_BANDS").as_deref() != Ok("1")) {
        return 1;
    }
    let want = (4 * gpu.sm_count).div_ceil(grid_x.max(1));
    let afford = rows / (by * ROWS_PER_THREAD).max(1);
    want.min(afford).max(1)
}

/// [`col_sum_into`] over `bands` row bands: a partial per band, then a fold.
fn col_sum_banded(
    gpu: &Gpu,
    db: &mut GTensor<f32>,
    dy: &GTensor<f32>,
    mul: WideOrSlab<'_>,
    use_mul: i32,
    div: &GTensor<f32>,
    use_div: i32,
    rows: usize,
    n: usize,
    bx: usize,
    by: usize,
    cfg: LaunchConfig,
    bands: usize,
    cache: &super::temp::TempCache,
) {
    let (rows_i, n_i) = (rows as i32, n as i32);
    let band_i = rows.div_ceil(bands) as i32;
    let bands_i = bands as i32;
    {
        let mut part = cache.get::<f32>(gpu, &[bands, n]);
        let part = &mut *part;
        let f = gpu.kernels.get(mul.pick("col_sum_part", "col_sum_part_slab"));
        let mut lb = gpu.stream.launch_builder(&f);
        lb.arg(&mut part.buf).arg(&dy.buf);
        push_wos!(lb, mul);
        lb.arg(&use_mul)
            .arg(&rows_i)
            .arg(&n_i)
            .arg(&band_i);
        let part_cfg = LaunchConfig {
            grid_dim: (cfg.grid_dim.0, bands as u32, 1),
            block_dim: (bx as u32, by as u32, 1),
            shared_mem_bytes: cfg.shared_mem_bytes,
        };
        unsafe { lb.launch(part_cfg) }.expect("col_sum_part");

        let f = gpu.kernels.get("col_sum_merge");
        let mut lb = gpu.stream.launch_builder(&f);
        lb.arg(&mut db.buf)
            .arg(&part.buf)
            .arg(&div.buf)
            .arg(&use_div)
            .arg(&bands_i)
            .arg(&n_i);
        let threads = n.clamp(32, 256).next_power_of_two().min(1024);
        unsafe {
            lb.launch(LaunchConfig {
                grid_dim: (n.div_ceil(threads) as u32, 1, 1),
                block_dim: (threads as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .expect("col_sum_merge");
    }
}




/// Gather rows of `table` (`[vocab, dim]`) by `ids` into a `[ids.len(), dim]`
/// tensor.
pub fn embedding_gather(
    gpu: &Gpu,
    table: &GTensor<f32>,
    ids: &[usize],
    dim: usize,
) -> GTensor<f32> {
    embedding_gather_u32(gpu, table, &upload_ids(gpu, ids).slice(..), ids.len(), dim)
}

/// [`embedding_gather`] against ids already resident on the device. `rows` is
/// the id count, which the caller knows and a `CudaSlice` may over-allocate.
pub fn embedding_gather_u32(
    gpu: &Gpu,
    table: &GTensor<f32>,
    dids: &CudaView<'_, u32>,
    rows: usize,
    dim: usize,
) -> GTensor<f32> {
    let mut out = GTensor::uninit(gpu, &[rows, dim]);
    embedding_gather_u32_into(gpu, &mut out, table, dids, rows, dim);
    out
}

/// [`embedding_gather_u32`] into a caller-owned `out`, which must already be at
/// least `[rows, dim]`. Lets a reused buffer take the gather instead of a fresh
/// allocation per call.
pub fn embedding_gather_u32_into(
    gpu: &Gpu,
    out: &mut GTensor<f32>,
    table: &GTensor<f32>,
    dids: &CudaView<'_, u32>,
    rows: usize,
    dim: usize,
) {
    assert!(
        rows * dim <= out.capacity(),
        "embedding_gather: {rows}x{dim} exceeds the {} allocated",
        out.capacity()
    );
    let (dim_i, rows_i) = (dim as i32, rows as i32);
    let f = gpu.kernels.get("embedding_gather");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&table.buf)
        .arg(dids)
        .arg(&mut out.buf)
        .arg(&dim_i)
        .arg(&rows_i);
    unsafe { lb.launch(elem_cfg(gpu, (rows * dim) as u32)) }
        .expect("embedding_gather");
}

/// Scatter-add: `dtable[ids[r]] += dy[r]`, deterministically (ids may repeat).
pub fn embedding_scatter_add(
    gpu: &Gpu,
    dtable: &mut GTensor<f32>,
    ids: &[usize],
    dy: &GTensor<f32>,
    dim: usize,
) {
    embedding_scatter_add_u32(
        gpu,
        dtable,
        &upload_ids(gpu, ids).slice(..),
        ids.len(),
        dy,
        dim,
    );
}

/// [`embedding_scatter_add`] against ids already resident on the device.
pub fn embedding_scatter_add_u32(
    gpu: &Gpu,
    dtable: &mut GTensor<f32>,
    dids: &CudaView<'_, u32>,
    rows: usize,
    dy: &GTensor<f32>,
    dim: usize,
) {
    let vocab = dtable.len() / dim;
    assert_eq!(
        vocab * dim,
        dtable.len(),
        "embedding_scatter_add: table shape"
    );
    // Row slices: one thread owns one (slice, column) and walks its rows in order, so
    // repeated ids accumulate in an order the shape fixes rather than the scheduler.
    // More slices means more parallelism and a bigger private table, so the count is
    // capped both ways; one slice needs no private table and no merge at all.
    const ROWS_PER_SLICE: usize = 256;
    const MAX_SLICES: usize = 16;
    const BLOCK: usize = 64;
    let slices = rows.div_ceil(ROWS_PER_SLICE).clamp(1, MAX_SLICES);
    let mut part = (slices > 1).then(|| GTensor::zeros(gpu, &[slices, vocab, dim]));
    let (dim_i, rows_i) = (dim as i32, rows as i32);
    let (vocab_i, slices_i) = (vocab as i32, slices as i32);
    let cfg = LaunchConfig {
        grid_dim: (dim.div_ceil(BLOCK) as u32, slices as u32, 1),
        block_dim: (BLOCK as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    {
        let target = part.as_mut().unwrap_or(dtable);
        let f = gpu.kernels.get("embedding_scatter_add");
        let mut lb = gpu.stream.launch_builder(&f);
        lb.arg(&mut target.buf)
            .arg(dids)
            .arg(&dy.buf)
            .arg(&dim_i)
            .arg(&rows_i)
            .arg(&vocab_i)
            .arg(&slices_i);
        unsafe { lb.launch(cfg) }.expect("embedding_scatter_add");
    }
    let Some(part) = part else { return };
    let n_i = (vocab * dim) as i32;
    let f = gpu.kernels.get("embedding_scatter_merge");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut dtable.buf)
        .arg(&part.buf)
        .arg(&n_i)
        .arg(&slices_i);
    unsafe { lb.launch(elem_cfg(gpu, (vocab * dim) as u32)) }
        .expect("embedding_scatter_merge");
}

/// The one intermediate a GPU RMSNorm forward saves: `1/rms(x)` per normalization
/// group, `[N]` for a plain norm.
///
/// Not `x̂`. Backward rebuilds that from the forward output as `x̂ = y/γ` — the same
/// trade every fast implementation makes (Apex's `memory_efficient` mode; Liger saves
/// the input and rebuilds `x̂ = x·inv_rms` from the other side). `x̂` is `[N, F]`, so
/// storing it costs the full activation width for a value two flops recover.
pub struct GpuRmsForward {
    pub inv_rms: CudaSlice<f32>,
}

/// Grouped RMSNorm forward (plain: `group == F`; head-wise: `group == dhv`).
/// Returns `(out, saved)`.
pub fn rms_norm_forward(
    gpu: &Gpu,
    x: &GTensor<f32>,
    gamma: &GTensor<f32>,
    group: usize,
    eps: f32,
) -> (GTensor<f32>, GpuRmsForward) {
    let (b, f) = (x.rows(), x.cols());
    let total_groups = b * (f / group);
    let mut out = GTensor::uninit(gpu, &[b, f]);
    let mut saved = GpuRmsForward {
        inv_rms: gpu
            .stream
            .alloc_zeros::<f32>(total_groups)
            .expect("alloc inv_rms"),
    };
    rms_norm_forward_into(gpu, x, gamma, group, eps, &mut out, &mut saved);
    (out, saved)
}

/// Grouped RMSNorm forward into caller-owned buffers — the no-allocation form of
/// [`rms_norm_forward`]. `out` and `saved.x_hat` must be `[B, F]`, and
/// `saved.inv_rms` must hold `B * F/group` elements; the kernel writes all three
/// in full, so their prior contents do not matter.
/// Threads per block for the RMSNorm kernels. Must match `RMSN_THREADS` in the
/// kernel source.
const RMSN_THREADS: u32 = 256;

/// Launch geometry for the RMSNorm kernels: one BLOCK per (row, group), which
/// reduces over the group cooperatively.
///
/// The kernels used to be thread-per-group, which was pathological at this model's
/// shape: a block's norms are ungrouped, so `group == hidden` and one thread walked
/// 1024 elements serially — 1024 threads for the whole launch on an 84-SM card.
/// Phase timing put the norms plus residual adds at 44.7% of a training step, more
/// than every recurrent cell combined; block-per-group took that component from
/// 213 ms to 4 ms and the step from 475 ms to 263 ms (1.8x).
///
/// One kernel, no crossover: block-per-group was measured faster at every group
/// size present here, including the mLSTM's head-wise `group == dhv` (64), where the
/// old kernel's launch was already wide. A sweep of the crossover found 257.8 ms
/// with the block kernel used everywhere against 262.8 ms with it used only above
/// 128, so the narrow case gains too and the second kernel earned no keep.
/// Launch shape for the RMSNorm kernels: one block per group, as wide as the group.
///
/// The width used to be [`RMSN_THREADS`] regardless, which is right for a block-level
/// norm — there `group` IS the hidden width — and badly wrong for a head norm. The
/// encoder's and decoder's mLSTM normalises groups of `dhv = 16`: 240 of every 256
/// threads sat idle, across 32768 blocks, and the cross-warp fold ran over seven
/// empty warps to combine one partial.
fn rms_norm_cfg(total_groups: usize, group: usize) -> LaunchConfig {
    let threads = (group.next_power_of_two() as u32).clamp(32, RMSN_THREADS);
    LaunchConfig {
        grid_dim: (total_groups as u32, 1, 1),
        block_dim: (threads, 1, 1),
        // One float per warp for the cross-warp fold.
        shared_mem_bytes: (threads / 32) * std::mem::size_of::<f32>() as u32,
    }
}

pub fn rms_norm_forward_into(
    gpu: &Gpu,
    x: &GTensor<f32>,
    gamma: &GTensor<f32>,
    group: usize,
    eps: f32,
    out: &mut GTensor<f32>,
    saved: &mut GpuRmsForward,
) {
    // No reshape: `out` may legitimately be `[B, T, H]` and the caller's next op
    // depends on its rank. `as_2d` in the launcher folds it for the shape check only.
    rms_fwd_launch(gpu, x, gamma, group, eps, RmsOut::F32(out), saved);
}

/// [`rms_norm_forward_into`] writing a slab.
///
/// `x` stays fp32 — it is the residual stream, which is the one tensor in the block
/// that must not be narrowed — but the output need not be: its consumer is a GEMM
/// that reads bf16 anyway, so writing narrow here both halves the store and removes
/// the cast pass that would otherwise read the fp32 result straight back out of HBM
/// to produce the very same bits.
pub fn rms_norm_forward_into_slab(
    gpu: &Gpu,
    x: &GTensor<f32>,
    gamma: &GTensor<f32>,
    group: usize,
    eps: f32,
    out: &mut SlabBuf,
    saved: &mut GpuRmsForward,
) {
    let (b, f) = x.as_2d();
    out.fit(gpu, &[b, f]);
    rms_fwd_launch(gpu, x, gamma, group, eps, RmsOut::Slab(out), saved);
}

/// The forward's output, at either width. Mutable, so it cannot reuse [`WideOrSlab`].
enum RmsOut<'a> {
    F32(&'a mut GTensor<f32>),
    Slab(&'a mut SlabBuf),
}

fn rms_fwd_launch(
    gpu: &Gpu,
    x: &GTensor<f32>,
    gamma: &GTensor<f32>,
    group: usize,
    eps: f32,
    out: RmsOut<'_>,
    saved: &mut GpuRmsForward,
) {
    // Position-wise over the last axis, so a `[B, T, H]` caller is served as-is —
    // see `GTensor::as_2d`.
    let (b, f) = x.as_2d();
    let groups_per_row = f / group;
    let total_groups = b * groups_per_row;
    assert_eq!(
        saved.inv_rms.len(),
        total_groups,
        "rms_norm_forward: inv_rms length"
    );
    let (gpr_i, group_i, tg_i) = (groups_per_row as i32, group as i32, total_groups as i32);
    let cfg = rms_norm_cfg(total_groups, group);
    let name = match &out {
        RmsOut::Slab(SlabBuf::Bf16(_)) => "rms_norm_forward_slab",
        _ => "rms_norm_forward",
    };
    let func = gpu.kernels.get(name);
    let mut lb = gpu.stream.launch_builder(&func);
    lb.arg(&x.buf).arg(&gamma.buf);
    match out {
        RmsOut::F32(t) => {
            assert_eq!(t.as_2d(), (b, f), "rms_norm_forward: out shape");
            lb.arg(&mut t.buf)
        }
        RmsOut::Slab(SlabBuf::F32(t)) => {
            assert_eq!(t.as_2d(), (b, f), "rms_norm_forward: out shape");
            lb.arg(&mut t.buf)
        }
        RmsOut::Slab(SlabBuf::Bf16(t)) => {
            assert_eq!(t.as_2d(), (b, f), "rms_norm_forward: out shape");
            lb.arg(&mut t.buf)
        }
    };
    lb.arg(&mut saved.inv_rms)
        .arg(&gpr_i)
        .arg(&group_i)
        .arg(&eps)
        .arg(&tg_i);
    unsafe { lb.launch(cfg) }.expect("rms_norm_forward");
}

/// Grouped RMSNorm backward. Accumulates γ grad into `dgamma`, returns `dX`.
pub fn rms_norm_backward(
    gpu: &Gpu,
    dy: &GTensor<f32>,
    fwd: &GpuRmsForward,
    y: &GTensor<f32>,
    gamma: &GTensor<f32>,
    dgamma: &mut GTensor<f32>,
    group: usize,
    cache: &super::temp::TempCache,
) -> GTensor<f32> {
    let mut dx = GTensor::uninit(gpu, &[dy.rows(), dy.cols()]);
    rms_norm_backward_into(
        gpu,
        dy,
        fwd,
        WideOrSlab::F32(y),
        gamma,
        dgamma,
        group,
        &mut dx,
        cache,
    );
    dx
}

/// Grouped RMSNorm backward into a caller-owned `dx` — the no-allocation form of
/// [`rms_norm_backward`]. `dgamma` is accumulated into (not overwritten); `dx` is
/// written in full.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_backward_into(
    gpu: &Gpu,
    dy: &GTensor<f32>,
    fwd: &GpuRmsForward,
    y: WideOrSlab<'_>,
    gamma: &GTensor<f32>,
    dgamma: &mut GTensor<f32>,
    group: usize,
    dx: &mut GTensor<f32>,
    cache: &super::temp::TempCache,
) {
    let (b, f) = dy.as_2d();
    let groups_per_row = f / group;
    let total_groups = b * groups_per_row;
    assert_eq!(dx.as_2d(), (b, f), "rms_norm_backward: dx shape");
    assert_eq!(y.as_2d(), (b, f), "rms_norm_backward: y shape");
    let (gpr_i, group_i, tg_i) = (groups_per_row as i32, group as i32, total_groups as i32);
    let cfg = rms_norm_cfg(total_groups, group);
    // `dy` and `dx` stay fp32 either way: `dx` continues into the residual chain.
    let func = gpu
        .kernels
        .get(y.pick("rms_norm_backward", "rms_norm_backward_slab"));
    let mut lb = gpu.stream.launch_builder(&func);
    lb.arg(&dy.buf);
    push_wos!(lb, y);
    lb.arg(&fwd.inv_rms)
        .arg(&gamma.buf)
        .arg(&mut dx.buf)
        .arg(&gpr_i)
        .arg(&group_i)
        .arg(&tg_i);
    unsafe { lb.launch(cfg) }.expect("rms_norm_backward");
    // dγ is a sum over ROWS of `dy ⊙ x̂`, so every block above would contribute to the
    // same slots. Its own deterministic reduction instead of an atomic there.
    //
    // `x̂ = y/γ` and γ is constant down a column, so the divide comes out of the sum
    // entirely — which is what lets this path run without materializing `x̂` anywhere.
    add_col_sum_mul_div(gpu, dgamma, dy, y, gamma, cache);
}

/// Fused softmax + cross-entropy. Returns `(mean_loss, dlogits)` with
/// `dlogits = (softmax − onehot) / B`, matching `nn2::loss`.
pub fn softmax_cross_entropy(
    gpu: &Gpu,
    logits: &GTensor<f32>,
    targets: &[usize],
) -> (f32, GTensor<f32>) {
    let (b, c) = (logits.rows(), logits.cols());
    assert_eq!(
        targets.len(),
        b,
        "softmax_cross_entropy — targets len != batch"
    );
    let inv_b = 1.0 / b as f32;
    let (c_i, b_i) = (c as i32, b as i32);
    let dtargets = upload_ids(gpu, targets);
    let mut dlogits = GTensor::uninit(gpu, &[b, c]);
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
    unsafe { lb.launch(elem_cfg(gpu, b as u32)) }.expect("softmax_ce");
    let losses = gpu.stream.clone_dtoh(&row_loss).expect("download row_loss");
    let loss = losses.iter().sum::<f32>() * inv_b;
    (loss, dlogits)
}

/// `MLSTM_HEAD_MAJOR=1` gathers q/k/v head-major before the fused kernels instead of
/// striding over the projection output where it lies.
///
/// The gather is a streaming pass the stride path does not need, but it buys the
/// kernels row locality: head-major puts a timestep `W` from the next, position-major
/// `H*W`, and the fused kernels re-read q/k/v once per chunk. Which wins is a
/// measurement — see `examples/mlstm_layout_ab.rs`.
pub fn mlstm_head_major() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MLSTM_HEAD_MAJOR").is_ok_and(|v| v != "0"))
}

/// One AdamW step of `param` from `grad`, updating moments `m`/`v` in place.
/// `decay` toggles the decoupled weight-decay term. Mirrors `nn2::optim`.
///
/// Per tensor: a whole model steps its `ParamArena` in one launch instead (see
/// `gpu::arena`), and this is what a standalone layer and the parity tests use.
pub fn adamw(
    gpu: &Gpu,
    param: &mut GTensor<f32>,
    grad: &GTensor<f32>,
    m: &mut GTensor<f32>,
    v: &mut GTensor<f32>,
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
    unsafe { lb.launch(elem_cfg(gpu, n as u32)) }.expect("adamw");
}

// sLSTM cell kernels (recurrent core, see gpu/slstm.rs). Each is the device
// counterpart of an inner step of `nn2::SLstm`; all state stays resident in
// `GTensor<f32>`s across the T-loop — the only host transfers are the layer's input
// and output.

/// Build `xh = concat(x[:, t, :], h_state)` into `xh` (`[B, rows]`), reading the
/// timestep-`t` slice of `x` (`[B, T, inp]`) and the recurrent state (`[B, H]`).
pub fn concat_xh(
    gpu: &Gpu,
    xh: &mut GTensor<f32>,
    x: &GTensor<f32>,
    h_state: &GTensor<f32>,
    t: usize,
) {
    let (b, rows) = (xh.rows(), xh.cols());
    let h = h_state.cols();
    let inp = rows - h;
    let big_t = x.shape[1];
    let br = b * rows;
    let (t_i, bigt_i, inp_i, h_i, br_i) = (t as i32, big_t as i32, inp as i32, h as i32, br as i32);
    let f = gpu.kernels.get("concat_xh");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut xh.buf)
        .arg(&x.buf)
        .arg(&h_state.buf)
        .arg(&t_i)
        .arg(&bigt_i)
        .arg(&inp_i)
        .arg(&h_i)
        .arg(&br_i);
    unsafe { lb.launch(elem_cfg(gpu, br as u32)) }.expect("concat_xh");
}

/// Split `dxh` (`[B, rows]`) into `dx[:, t, :]` (first `inp` cols) and `dh_bptt`
/// (`[B, H]`, last `H` cols). `dx` is `[B, T, inp]`.
pub fn split_dxh(
    gpu: &Gpu,
    dxh: &GTensor<f32>,
    dx: &mut GTensor<f32>,
    dh_bptt: &mut GTensor<f32>,
    t: usize,
) {
    let (b, rows) = (dxh.rows(), dxh.cols());
    let h = dh_bptt.cols();
    let inp = rows - h;
    let big_t = dx.shape[1];
    let br = b * rows;
    let (t_i, bigt_i, inp_i, h_i, br_i) = (t as i32, big_t as i32, inp as i32, h as i32, br as i32);
    let f = gpu.kernels.get("split_dxh");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&dxh.buf)
        .arg(&mut dx.buf)
        .arg(&mut dh_bptt.buf)
        .arg(&t_i)
        .arg(&bigt_i)
        .arg(&inp_i)
        .arg(&h_i)
        .arg(&br_i);
    unsafe { lb.launch(elem_cfg(gpu, br as u32)) }.expect("split_dxh");
}

/// Per-step saved tensors of one sLSTM forward step, consumed by the backward
/// step. Each is `[B, H]` (`xh` lives on the layer). Grouped so the layer can
/// hold a `Vec` of them across the T-loop.
pub struct SlstmSaved {
    pub c_prev: GTensor<f32>,
    pub n_prev: GTensor<f32>,
    pub zt: GTensor<f32>,
    pub ot: GTensor<f32>,
    pub i_prime: GTensor<f32>,
    pub f_prime: GTensor<f32>,
    pub c: GTensor<f32>,
    pub n: GTensor<f32>,
}

/// One forward sLSTM step: advances `(c,n,m,h)_state` in place from the four gate
/// pre-activations, fills `saved` for backward, and writes `out[:, t, :]`.
/// `ft_pre` is the (bias-added) forget pre-activation and is itself a saved
/// per-step buffer (reused in backward).
#[allow(clippy::too_many_arguments)]
pub fn slstm_cell_step(
    gpu: &Gpu,
    zt_pre: &GTensor<f32>,
    it_pre: &GTensor<f32>,
    ft_pre: &GTensor<f32>,
    ot_pre: &GTensor<f32>,
    c_state: &mut GTensor<f32>,
    n_state: &mut GTensor<f32>,
    m_state: &mut GTensor<f32>,
    h_state: &mut GTensor<f32>,
    saved: &mut SlstmSaved,
    out: &mut GTensor<f32>,
    t: usize,
) {
    let (b, h) = (c_state.rows(), c_state.cols());
    let bh = b * h;
    let big_t = out.shape[1];
    let (t_i, bigt_i, h_i, bh_i) = (t as i32, big_t as i32, h as i32, bh as i32);
    let f = gpu.kernels.get("slstm_cell_step");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&zt_pre.buf)
        .arg(&it_pre.buf)
        .arg(&ft_pre.buf)
        .arg(&ot_pre.buf)
        .arg(&mut c_state.buf)
        .arg(&mut n_state.buf)
        .arg(&mut m_state.buf)
        .arg(&mut h_state.buf)
        .arg(&mut saved.c_prev.buf)
        .arg(&mut saved.n_prev.buf)
        .arg(&mut saved.zt.buf)
        .arg(&mut saved.ot.buf)
        .arg(&mut saved.i_prime.buf)
        .arg(&mut saved.f_prime.buf)
        .arg(&mut saved.c.buf)
        .arg(&mut saved.n.buf)
        .arg(&mut out.buf)
        .arg(&t_i)
        .arg(&bigt_i)
        .arg(&h_i)
        .arg(&bh_i);
    unsafe { lb.launch(elem_cfg(gpu, bh as u32)) }.expect("slstm_cell_step");
}

/// One backward sLSTM step: from `dy[:, t, :]` + the incoming BPTT channels,
/// produce the four gate deltas (`dz,di,df,dob`) and update `dc_bptt`/`dn_bptt`
/// in place for the earlier step. `dh_bptt` is read here (set by the later step's
/// `split_dxh`).
#[allow(clippy::too_many_arguments)]
pub fn slstm_cell_step_bwd(
    gpu: &Gpu,
    dy: &GTensor<f32>,
    dh_bptt: &GTensor<f32>,
    saved: &SlstmSaved,
    ft_pre: &GTensor<f32>,
    dc_bptt: &mut GTensor<f32>,
    dn_bptt: &mut GTensor<f32>,
    dz: &mut GTensor<f32>,
    di: &mut GTensor<f32>,
    df: &mut GTensor<f32>,
    dob: &mut GTensor<f32>,
    t: usize,
) {
    let (b, h) = (dc_bptt.rows(), dc_bptt.cols());
    let bh = b * h;
    let big_t = dy.shape[1];
    let (t_i, bigt_i, h_i, bh_i) = (t as i32, big_t as i32, h as i32, bh as i32);
    let f = gpu.kernels.get("slstm_cell_step_bwd");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&dy.buf)
        .arg(&t_i)
        .arg(&bigt_i)
        .arg(&dh_bptt.buf)
        .arg(&saved.ot.buf)
        .arg(&saved.c.buf)
        .arg(&saved.n.buf)
        .arg(&saved.c_prev.buf)
        .arg(&saved.n_prev.buf)
        .arg(&saved.zt.buf)
        .arg(&saved.i_prime.buf)
        .arg(&saved.f_prime.buf)
        .arg(&ft_pre.buf)
        .arg(&mut dc_bptt.buf)
        .arg(&mut dn_bptt.buf)
        .arg(&mut dz.buf)
        .arg(&mut di.buf)
        .arg(&mut df.buf)
        .arg(&mut dob.buf)
        .arg(&h_i)
        .arg(&bh_i);
    unsafe { lb.launch(elem_cfg(gpu, bh as u32)) }.expect("slstm_cell_step_bwd");
}

// Fused-gate sLSTM (the fast path; see gpu/slstm.rs). The four gates run as one
// [., 4H] block, so a timestep costs one GEMM + one kernel instead of four GEMMs
// plus four bias broadcasts plus a concat. The gate weights of record stay the
// four [rows, H] matrices; these pack them into the fused operands and unpack the
// gradients back, so the checkpoint layout is untouched.

/// Pack the four gate matrices `[rows, H]` into `wx [inp, 4H]` / `wh [H, 4H]` and
/// the four biases into `bcat [4H]`. Two launches, once per forward.
#[allow(clippy::too_many_arguments)]
pub fn slstm_pack(
    gpu: &Gpu,
    w: &[GTensor<f32>; 4],
    bias: &[GTensor<f32>; 4],
    wx: &mut GTensor<f32>,
    wh: &mut GTensor<f32>,
    bcat: &mut GTensor<f32>,
    inp: usize,
    h: usize,
) {
    let rows = inp + h;
    let (inp_i, h_i, rows_i) = (inp as i32, h as i32, rows as i32);
    let f = gpu.kernels.get("slstm_pack_w");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&w[0].buf)
        .arg(&w[1].buf)
        .arg(&w[2].buf)
        .arg(&w[3].buf)
        .arg(&mut wx.buf)
        .arg(&mut wh.buf)
        .arg(&inp_i)
        .arg(&h_i)
        .arg(&rows_i);
    unsafe { lb.launch(elem_cfg(gpu, (rows * 4 * h) as u32)) }.expect("slstm_pack_w");

    let f = gpu.kernels.get("slstm_pack_b");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&bias[0].buf)
        .arg(&bias[1].buf)
        .arg(&bias[2].buf)
        .arg(&bias[3].buf)
        .arg(&mut bcat.buf)
        .arg(&h_i);
    unsafe { lb.launch(elem_cfg(gpu, (4 * h) as u32)) }.expect("slstm_pack_b");
}

/// `dw[g] += ` the g-th column block of the fused `dwx` / `dwh` (the inverse of
/// [`slstm_pack`] for gradients — accumulating, so grads survive across windows).
pub fn slstm_unpack_dw(
    gpu: &Gpu,
    dwx: &GTensor<f32>,
    dwh: &GTensor<f32>,
    dw: &mut [GTensor<f32>; 4],
    inp: usize,
    h: usize,
) {
    let rows = inp + h;
    let (inp_i, h_i, rows_i) = (inp as i32, h as i32, rows as i32);
    let [dw0, dw1, dw2, dw3] = dw;
    let f = gpu.kernels.get("slstm_unpack_dw");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&dwx.buf)
        .arg(&dwh.buf)
        .arg(&mut dw0.buf)
        .arg(&mut dw1.buf)
        .arg(&mut dw2.buf)
        .arg(&mut dw3.buf)
        .arg(&inp_i)
        .arg(&h_i)
        .arg(&rows_i);
    unsafe { lb.launch(elem_cfg(gpu, (rows * 4 * h) as u32)) }
        .expect("slstm_unpack_dw");
}

/// Fill `t` with a constant.
pub fn fill(gpu: &Gpu, t: &mut GTensor<f32>, v: f32) {
    let n = t.len();
    let n_i = n as i32;
    let f = gpu.kernels.get("fill_const");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut t.buf).arg(&v).arg(&n_i);
    unsafe { lb.launch(elem_cfg(gpu, n as u32)) }.expect("fill_const");
}

/// `db[g] += ` column sums of the g-th block of the fused gate deltas `dg [N, 4H]`.
/// The sum over the N rows is a `ones[1, N] · dg` GEMM (cuBLAS reduces properly);
/// the kernel only scatters the reduced `[4H]` row into the four bias grads.
pub fn slstm_db_from_dg(
    gpu: &Gpu,
    dg: &GTensor<f32>,
    ones: &GTensor<f32>,
    dbcat: &mut GTensor<f32>,
    db: &mut [GTensor<f32>; 4],
    h: usize,
) {
    matmul_nn_into(gpu, ones, dg, dbcat, 0.0);
    let h_i = h as i32;
    let [db0, db1, db2, db3] = db;
    let f = gpu.kernels.get("slstm_unpack_db");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&dbcat.buf)
        .arg(&mut db0.buf)
        .arg(&mut db1.buf)
        .arg(&mut db2.buf)
        .arg(&mut db3.buf)
        .arg(&h_i);
    unsafe { lb.launch(elem_cfg(gpu, (4 * h) as u32)) }.expect("slstm_unpack_db");
}

/// Saved forward tensors of a fused-gate sLSTM, each a `[B, T, H]` slab (indexed
/// `(b·T + t)·H + j`) rather than one small tensor per timestep.
pub struct SlstmSlabs {
    // Stabilizer-carrying. `c`/`n` hold the exp(-m)-scaled cell and normalizer and
    // `i_prime`/`f_prime` ARE exp(·-m), so an absolute error here is a multiplicative
    // error in the quantity that keeps the recurrence bounded. fp32 unless
    // `SLSTM_BF16_STATE=1` — see `Kernels::state_bf16` and `gpu::bf16`.
    /// The cell and normalizer *entering* the sweep, `[B, H]` — the `t = 0`
    /// predecessor and nothing more.
    ///
    /// Backward needs `c_{t-1}`/`n_{t-1}` at every step, but for `t > 0` that is
    /// `c`/`n` one timestep back in the very same buffer (`s - H`). Only the first
    /// step has no predecessor there. Storing the whole history separately cost two
    /// extra `[B, T, H]` tensors for one timestep's worth of information.
    pub c_entry: SlabBuf,
    pub n_entry: SlabBuf,
    pub i_prime: SlabBuf,
    pub f_prime: SlabBuf,
    pub c: SlabBuf,
    pub n: SlabBuf,
    // Plain activations: bf16 when the kernels were built for it. These are bounded
    // by construction (`zt` is a tanh, `ot` a sigmoid, `h_prev` their product over a
    // normalized ratio), written once and read once, and enter no reduction in their
    // stored form — so storage precision is free of the recurrence's error growth.
    //
    // Held as `SlabBuf` rather than `GTensor<f32>`: the width must match what the kernels
    // were compiled against (`Kernels::slab_bf16`), which is checked on construction.
    pub zt: SlabBuf,
    pub ot: SlabBuf,
    pub h_prev: SlabBuf,
}

impl SlstmSlabs {
    /// Device bytes this saved set holds, each slab at its real width. Diagnostic.
    pub fn retained_bytes(&self) -> usize {
        [
            &self.c_entry,
            &self.n_entry,
            &self.i_prime,
            &self.f_prime,
            &self.c,
            &self.n,
            &self.zt,
            &self.ot,
            &self.h_prev,
        ]
        .iter()
        .map(|t| t.retained_bytes())
        .sum()
    }
}

/// A saved slab whose element width follows the compiled kernels: bf16 when they
/// were built with `-DSLAB_BF16`, fp32 otherwise.
///
/// This exists rather than a bare `GTensor<f32>` because the kernels index these buffers
/// at a **compile-time** width. Handing a kernel built for `__nv_bfloat16` an fp32
/// buffer is not a type error anywhere — it is a silent stride mismatch that reads
/// half the tensor and writes past the end of it. Routing every allocation through
/// [`SlabBuf::new`], which reads `Kernels::slab_bf16`, makes the two agree by
/// construction.
pub enum SlabBuf {
    F32(GTensor<f32>),
    Bf16(super::GTensor<u16>),
}

impl SlabBuf {
    /// An uninitialized slab at an explicitly chosen width.
    ///
    /// For a value whose readers are not the fused kernels: `zn` is normalized by the
    /// RMSNorm kernels (which take either width) and read by GEMMs (which take bf16
    /// only under `gemm_bf16_enabled`), so it must narrow only when *both* switches
    /// are on. [`new`](Self::new) asks the kernels alone, which is right for a slab the
    /// fused kernels own.
    pub fn new_width(gpu: &Gpu, dims: &[usize], bf16: bool) -> Self {
        if bf16 {
            SlabBuf::Bf16(super::GTensor::uninit(gpu, dims))
        } else {
            SlabBuf::F32(GTensor::uninit(gpu, dims))
        }
    }

    /// An uninitialized slab at the width the kernels expect.
    pub fn new(gpu: &Gpu, dims: &[usize]) -> Self {
        if gpu.kernels.slab_bf16 {
            SlabBuf::Bf16(super::GTensor::uninit(gpu, dims))
        } else {
            SlabBuf::F32(GTensor::uninit(gpu, dims))
        }
    }

    /// Take ownership of an fp32 tensor as a slab, narrowing it when the kernels
    /// were built for bf16.
    ///
    /// This is the conversion point for a value that is *produced* in fp32 (a
    /// projection, a head-major reorg) but only *consumed* by the fused kernels.
    /// On the bf16 path the fp32 original is dropped here, so the cache holds only
    /// the narrow copy — which is the whole point: the wide buffer is transient,
    /// the slab is what lives across forward and backward.
    pub fn from_f32(gpu: &Gpu, t: GTensor<f32>) -> Self {
        if !gpu.kernels.slab_bf16 {
            return SlabBuf::F32(t);
        }
        let mut b = super::GTensor::uninit(gpu, t.dims());
        b.store(gpu, &t);
        SlabBuf::Bf16(b)
    }

    /// An fp32 view of this slab, for a consumer that cannot read bf16 — chiefly
    /// cuBLAS, which has no bf16 operand on these GEMMs.
    ///
    /// `scratch` receives the widened copy on the bf16 path and is left untouched on
    /// the fp32 path, where the slab is already what the caller wants. (A `Cow` would
    /// be the natural shape, but `GTensor<f32>` is deliberately not `Clone`.)
    pub fn as_f32<'a>(&'a self, gpu: &Gpu, scratch: &'a mut GTensor<f32>) -> &'a GTensor<f32> {
        match self {
            SlabBuf::F32(t) => t,
            SlabBuf::Bf16(b) => {
                b.load(gpu, scratch);
                scratch
            }
        }
    }

    pub fn dims(&self) -> &[usize] {
        match self {
            SlabBuf::F32(t) => t.dims(),
            SlabBuf::Bf16(t) => t.dims(),
        }
    }

    pub fn capacity(&self) -> usize {
        match self {
            SlabBuf::F32(t) => t.capacity(),
            SlabBuf::Bf16(t) => t.capacity(),
        }
    }

    /// Device bytes held, at this slab's actual element width. Diagnostic.
    pub fn retained_bytes(&self) -> usize {
        match self {
            SlabBuf::F32(t) => t.capacity() * 4,
            SlabBuf::Bf16(t) => t.capacity() * 2,
        }
    }

    pub fn shrink_to(&mut self, dims: &[usize]) {
        match self {
            SlabBuf::F32(t) => t.shrink_to(dims),
            SlabBuf::Bf16(t) => t.shrink_to(dims),
        }
    }

    /// Present this slab at `dims`, reallocating only when the current buffer is too
    /// small — the [`SlabBuf`] twin of `Buf::get`.
    pub fn fit(&mut self, gpu: &Gpu, dims: &[usize]) {
        let n: usize = dims.iter().product();
        if self.capacity() >= n {
            self.shrink_to(dims);
        } else {
            *self = SlabBuf::new(gpu, dims);
        }
    }

    /// Fill from an fp32 tensor of this slab's shape — a narrowing cast on the bf16
    /// path, a device-to-device copy on the fp32 one.
    pub fn store(&mut self, gpu: &Gpu, src: &GTensor<f32>) {
        let n = self.dims().iter().product::<usize>();
        assert!(n <= src.capacity(), "SlabBuf::store: source is too small");
        match self {
            SlabBuf::Bf16(t) => t.store(gpu, src),
            SlabBuf::F32(t) => gpu
                .stream
                .memcpy_dtod(&src.buf.slice(..n), &mut t.buf.slice_mut(..n))
                .expect("SlabBuf::store"),
        }
    }
}

/// Push a slab as a kernel argument at whichever width it holds.
///
/// A `LaunchArgs` builder takes `&CudaSlice<T>` for a concrete `T`, so the two
/// variants cannot be unified behind one `.arg()` call; this matches once and
/// pushes the right pointer. The kernel's parameter is `slab_t*`, which was
/// compiled to the same width by construction (see [`SlabBuf`]).
macro_rules! push_slab {
    ($lb:expr, $slab:expr) => {
        match &mut $slab {
            SlabBuf::F32(t) => $lb.arg(&mut t.buf),
            SlabBuf::Bf16(t) => $lb.arg(&mut t.buf),
        }
    };
}

/// [`push_slab`] for a slab the kernel only reads.
macro_rules! push_slab_ref {
    ($lb:expr, $slab:expr) => {
        match &$slab {
            SlabBuf::F32(t) => $lb.arg(&t.buf),
            SlabBuf::Bf16(t) => $lb.arg(&t.buf),
        }
    };
}

/// One fused forward step: add the biases, advance `(c,n,m,h)_state`, fill the
/// saved slabs at `t` and write `out[:, t, :]`. `g`'s forget block is left holding
/// the biased forget pre-activation for backward.
///
/// `h_narrow` receives the new `h` at the slab width — the left operand the *next*
/// step's `h·Wh` GEMM reads. Producing it here is what lets that GEMM run on bf16
/// operands without a narrowing launch of its own inside the T-loop.
#[allow(clippy::too_many_arguments)]
pub fn slstm_step_fused(
    gpu: &Gpu,
    g: &mut GTensor<f32>,
    gh: &GTensor<f32>,
    bcat: &GTensor<f32>,
    c_state: &mut GTensor<f32>,
    n_state: &mut GTensor<f32>,
    m_state: &mut GTensor<f32>,
    h_state: &mut GTensor<f32>,
    h_narrow: &mut SlabBuf,
    slabs: &mut SlstmSlabs,
    out: &mut GTensor<f32>,
    t: usize,
    first: bool,
) {
    let (b, h) = (c_state.rows(), c_state.cols());
    let bh = b * h;
    let big_t = out.shape[1];
    let (t_i, bigt_i, h_i, bh_i) = (t as i32, big_t as i32, h as i32, bh as i32);
    let first_i = i32::from(first);
    let f = gpu.kernels.get("slstm_step_fused");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut g.buf).arg(&gh.buf).arg(&bcat.buf);
    push_slab!(lb, slabs.h_prev);
    lb.arg(&mut c_state.buf)
        .arg(&mut n_state.buf)
        .arg(&mut m_state.buf)
        .arg(&mut h_state.buf);
    push_slab!(lb, *h_narrow);
    push_slab!(lb, slabs.c_entry);
    push_slab!(lb, slabs.n_entry);
    push_slab!(lb, slabs.zt);
    push_slab!(lb, slabs.ot);
    push_slab!(lb, slabs.i_prime);
    push_slab!(lb, slabs.f_prime);
    push_slab!(lb, slabs.c);
    push_slab!(lb, slabs.n);
    lb.arg(&mut out.buf)
        .arg(&t_i)
        .arg(&bigt_i)
        .arg(&h_i)
        .arg(&bh_i)
        .arg(&first_i);
    unsafe { lb.launch(elem_cfg(gpu, bh as u32)) }.expect("slstm_step_fused");
}

/// Grid width for the fused kernels (`SLSTM_BLOCKS`), overriding the geometry's own
/// choice. Both geometries re-derive their units per block from it and then re-check
/// every constraint, so an override can trade grid width against work per block but
/// cannot break the contract — a value that does not fit declines instead.
fn fused_blocks_override() -> Option<usize> {
    static N: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("SLSTM_BLOCKS")
            .ok()
            .and_then(|v| v.parse().ok())
    })
}

/// Block width for the fused kernels (`SLSTM_THREADS`), for A/B sweeps. The forward
/// takes it as the width outright — both its phases are strided loops, so any width is
/// correct and it only trades warps-in-flight against idle lanes. The backward's width
/// is fixed by its warp-per-unit ownership, so there it sets the warps per unit and the
/// width follows; see [`slstm_fused_time_bwd_geometry`].
///
/// 1024 is a cliff, not a continuation: the forward holds its `h` slice in registers,
/// and at 1024 threads the per-thread budget (64) is too small for it, so the array
/// spills to local memory and the call goes 2.09 -> 7.92 us/timestep.
fn fused_threads_override() -> Option<usize> {
    static N: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("SLSTM_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|t: &usize| *t >= 32 && *t <= MAX_BLOCK_THREADS && t % 32 == 0)
    })
}

/// Rows of `Wh` the fused forward stages in shared memory (`SLSTM_STAGED_ROWS`),
/// capped by what actually fits. Lowering it moves rows into the global tail, which
/// is how the cost of that tail is measured against the shared-memory it frees.
fn fused_staged_rows_override() -> Option<usize> {
    static N: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("SLSTM_STAGED_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n > 0)
    })
}

/// Threads the fused FORWARD asks for a block.
///
/// Measured best at the backbone shape (H=768, B=1, 77 blocks), swept against the
/// real step. The intuition that a narrow block would suit the pointwise phase —
/// which fills only `B*nj` = 10 lanes there — is backwards: that phase is a short
/// prologue, while the gate reduction that follows it is the whole cost, and the
/// reduction scales with warps.
///
/// A ceiling, not the launch width: [`fused_fwd_threads_for`] narrows it to what the
/// register file can place. The curve is flat over the top of its range — at H=1024
/// the widths 576/640/704 measure 2.65/2.67/2.68 us per timestep — so the narrowing
/// costs next to nothing where it bites.
const FUSED_FWD_THREADS: usize = 768;

/// Hardware ceiling on a block, and on the static `__shared__` a kernel may declare.
const MAX_BLOCK_THREADS: usize = 1024;
const MAX_STATIC_SHARED: usize = 48 * 1024;

/// Rows the fused forward's staged `Wh` slice is padded to. Its lanes read eight
/// bf16 per shared access, so a warp covers 256 rows per pass and the tail is
/// zero-filled rather than branched around.
const FUSED_ROW_PAD: usize = 256;

/// Padded row count of the fused forward's staged `Wh` slice and `h` mirror.
pub fn fused_hp(h: usize) -> usize {
    h.div_ceil(FUSED_ROW_PAD) * FUSED_ROW_PAD
}

/// Launch geometry for [`slstm_fused_time`]: `(blocks, threads, units_per_block,
/// staged_rows, shared_bytes)`, or `None` when the shape does not fit the kernel's
/// contract.
///
/// The constraint that drives everything is shared memory: each block stages a
/// `[4*units, staged_rows]` bf16 slice of `Wh` plus a `[B, 4*units]` fp32 gate
/// scratch, and the total must fit the device's opt-in limit. Blocks are then spread
/// over as many SMs as the grid may use, and the whole grid must be co-resident — a
/// cooperative launch deadlocks if it cannot be.
///
/// Those two pull against each other above `H ≈ 1000`: co-residency caps the grid at
/// one wave, which sets `units = ceil(H / SMs)` from below, while the opt-in limit
/// caps `units * HP`. `staged_rows` is what gives: the block stages as many rows of
/// its columns as fit and reads the rest from the global tail scratch (see
/// `FUSED_RS` in the kernel). It equals `HP` — no tail, byte-identical to the
/// all-shared kernel — wherever the whole slice fits.
pub fn slstm_fused_time_geometry(
    gpu: &Gpu,
    h: usize,
    b: usize,
) -> Option<(usize, usize, usize, usize, usize)> {
    let hp = fused_hp(h);
    let threads = fused_threads_override().unwrap_or(FUSED_FWD_THREADS);
    // Leave a little headroom under the opt-in cap for the driver's own use.
    let smem_cap = gpu.max_shared_optin.saturating_sub(1024);
    // Spread over as many SMs as the grid may use, not as few blocks as possible.
    // The fewest-blocks choice minimises `grid.sync()` cost but leaves most of the
    // device idle, and the gate reduction (not the sync) is what dominates — so
    // halving the parallelism costs far more than the extra sync saves. One wave is
    // the ceiling either way (co-residency), so this is simply as wide as it goes.
    //
    // `SLSTM_BLOCKS` re-opens that tradeoff to measurement: it is a choice, not a
    // derived optimum, and it moves with H. Swept at H=768 (60 -> 10.57ms,
    // 77 -> 9.90ms, 84 -> 9.92ms): wider still wins, default stands.
    let blocks = fused_blocks_override()
        .unwrap_or(gpu.sm_count)
        .min(h)
        .max(1);
    let units_per_block = h.div_ceil(blocks);
    // Recompute: rounding the slice up may need fewer blocks than requested.
    let blocks = h.div_ceil(units_per_block);
    // The kernel's pointwise phase is one thread per (batch row, owned unit), fixed
    // for the whole T-loop so that the recurrent state can live in registers. Widen
    // the block if the default is too narrow for that; a shape that needs more than
    // a block can hold takes the per-step path instead.
    let threads = threads.max(b * units_per_block).next_multiple_of(32);
    if threads > MAX_BLOCK_THREADS {
        return None;
    }
    // Rows of the slice that fit in shared, in whole reduction passes — the loop
    // reads FUSED_ROW_PAD rows per pass with no tail branch, so a partial pass is
    // not a shape the kernel has.
    let gate_scratch = b * 4 * 4 * units_per_block;
    let per_row = units_per_block * 4 * 2;
    let staged_rows = fused_staged_rows_override()
        .unwrap_or(hp)
        .min(smem_cap.saturating_sub(gate_scratch) / per_row)
        .min(hp)
        / FUSED_ROW_PAD
        * FUSED_ROW_PAD;
    if staged_rows == 0 {
        return None; // not even one pass of the slice fits
    }
    let shared_bytes = staged_rows * per_row + gate_scratch;
    // A cooperative grid must be co-resident, and at this shared footprint an SM holds
    // one block — so more blocks than SMs cannot work. This is the cheap bound callers
    // can use as a predicate; `coop_grid_fits` asks the driver for the real one, which
    // also accounts for registers, once the function is in hand.
    if shared_bytes > smem_cap || blocks > gpu.sm_count {
        return None;
    }
    Some((blocks, threads, units_per_block, staged_rows, shared_bytes))
}

/// Elements of the fused forward's global `Wh` tail scratch, as fp32 (two bf16 to a
/// float). Zero when the whole slice is staged in shared memory.
pub fn fused_tail_len(h: usize, staged_rows: usize) -> usize {
    4 * h * (fused_hp(h) - staged_rows) / 2
}

/// The fused forward's block width, narrowed until one block fits an SM.
///
/// [`FUSED_FWD_THREADS`] is a *preference* — the width the gate reduction measures
/// best at — and the register file is what can refuse it. The kernel's per-thread
/// demand grows with `H` (a lane's slice of `h` is `HP/32` fp32 registers), and past
/// `H ≈ 900` the fitted 768-wide block needs more registers than an SM has: 90 regs
/// x 768 threads = 69k against 64k at `H = 1024`. That is not slow, it is
/// *unschedulable* — `occupancy_max_active_blocks_per_multiprocessor` returns 0 and
/// the cooperative launch cannot place a single block, so the whole path declines and
/// the caller silently takes the per-step loop at ~4x the cost per timestep.
///
/// Narrowing is the cheap half of that trade: the reduction loses warps, the kernel
/// stays fused. Registers per thread do not change with the launch width, so the cap
/// is exact arithmetic and the descending scan below is only there to absorb the
/// driver's own allocation granularity.
///
/// Returns `None` when even the narrowest legal width cannot be placed — the block
/// can never go below `b * units_per_block`, since the pointwise phase's ownership is
/// one thread per (batch row, owned unit) and fixed for the whole T-loop.
pub fn fused_fwd_threads_for(
    gpu: &Gpu,
    f: &cudarc::driver::CudaFunction,
    threads: usize,
    min_threads: usize,
    blocks: usize,
    shared: usize,
) -> Option<usize> {
    let regs = f
        .get_attribute(cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS)
        .ok()
        .filter(|&r| r > 0)? as usize;
    let by_regs = gpu.regs_per_sm / regs / 32 * 32;
    let mut w = threads.min(by_regs);
    while w >= min_threads.max(32) {
        if coop_grid_fits(gpu, f, blocks, w, shared) {
            return Some(w);
        }
        w -= 32;
    }
    None
}

/// Whether the driver will schedule `blocks` of this kernel as a cooperative grid.
///
/// A cooperative launch requires the WHOLE grid to be co-resident — `grid.sync()`
/// blocks until every block arrives, so an unscheduled block would deadlock it — and
/// the driver enforces that with `maxActiveBlocksPerSM * smCount`. Shared memory is
/// not the only thing that sets that number — registers can take it to zero, and then
/// a grid the geometry thought was fine comes back as
/// `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE` at launch. Asking makes the decline a
/// decision the caller can fall back from instead of an error message.
fn coop_grid_fits(
    gpu: &Gpu,
    f: &cudarc::driver::CudaFunction,
    blocks: usize,
    threads: usize,
    shared: usize,
) -> bool {
    match f.occupancy_max_active_blocks_per_multiprocessor(threads as u32, shared, None) {
        Ok(per_sm) => blocks <= per_sm as usize * gpu.sm_count,
        // No answer is not a licence to try: a launch that is too large is a hard
        // error and the per-step path is always available.
        Err(e) => {
            eprintln!("slstm fused: occupancy query failed: {e:?}");
            false
        }
    }
}

/// Ceiling, in 32-bit registers (two bf16 to a register), on the `Wh` row a
/// fused-backward warp spreads over its lanes.
///
/// The row is a register ARRAY, and it stays in registers only while the block's whole
/// demand fits the file — past that NVRTC spills it to local memory and the kernel
/// gives back more than the fused path was ever worth. A shape that would need more is
/// declined and takes the per-step loop instead. A lane holds `4H / 32` entries, so
/// this is the real ceiling on H for the fused backward: 64 registers reaches H=1024.
const BWD_WH_REGS: usize = 64;

/// Launch geometry for [`slstm_fused_time_bwd`]: `(blocks, threads, units_per_block)`,
/// or `None` when the shape does not fit the kernel's contract.
///
/// Nothing like the forward's, because the backward stages no `Wh` in shared memory:
/// a warp keeps its unit's whole row in its lanes' registers, so shared memory bounds
/// neither the block nor the grid. What is left is:
///
///   * spread the units over the whole device, one block per SM;
///   * one warp per (batch row, owned unit), the batch split across
///     `warps_per_unit` of them so a wide batch fills the block instead of leaving
///     warps idle;
///   * at least one thread per (batch row, owned unit), because the pointwise phase's
///     ownership is fixed for the whole T-loop — that is what puts `dc`/`dn` in
///     registers — so a batch wider than the block can hold declines.
pub fn slstm_fused_time_bwd_geometry(
    gpu: &Gpu,
    h: usize,
    b: usize,
) -> Option<(usize, usize, usize)> {
    let blocks = fused_blocks_override().unwrap_or(gpu.sm_count).clamp(1, h);
    let units = h.div_ceil(blocks);
    // Rounding the slice up may need fewer blocks than requested.
    let blocks = h.div_ceil(units);
    // Warps sharing a unit, each taking a stride of the batch. `threads` FOLLOWS from
    // it: the kernel derives a warp's unit by dividing the block width back out, so a
    // width that is not exactly `32 * units * warps_per_unit` would alias warps onto
    // units that are not this block's. `SLSTM_THREADS` therefore sets the warps per
    // unit rather than the width — and at B = 1 there is only one, since the batch is
    // the only axis these warps split.
    let lane_cap = (MAX_BLOCK_THREADS / (32 * units).max(1)).max(1);
    let warps_per_unit = fused_threads_override()
        .map_or(b, |t| t / (32 * units).max(1))
        .clamp(1, b.min(lane_cap).max(1));
    let threads = 32 * units * warps_per_unit;
    if threads > MAX_BLOCK_THREADS || b * units > threads || blocks > gpu.sm_count {
        return None;
    }
    if (4 * h).div_ceil(32) > 2 * BWD_WH_REGS {
        return None;
    }
    // `dh_sh`, the kernel's only shared array.
    if b * units * 4 > MAX_STATIC_SHARED {
        return None;
    }
    Some((blocks, threads, units))
}

/// The whole backward T-loop as **one cooperative launch**: see
/// `slstm_fused_time_bwd` in `cuda/slstm_coop.cu`. Writes the gate deltas into `g`
/// exactly as the per-step path does, so the post-loop dWx/dWh/db GEMMs are
/// unaffected — and reads them back out of it, so there is no scratch to pass.
///
/// Returns `false` (having launched nothing) when unavailable, so the caller falls
/// back to the per-step loop.
#[allow(clippy::too_many_arguments)]
pub fn slstm_fused_time_bwd(
    gpu: &Gpu,
    wh: &GTensor<f32>,
    dy: &GTensor<f32>,
    g: &mut GTensor<f32>,
    dh_recur: &mut GTensor<f32>,
    dc_recur: &mut GTensor<f32>,
    dn_recur: &mut GTensor<f32>,
    slabs: &SlstmSlabs,
    t: usize,
) -> bool {
    if !gpu.kernels.has_coop {
        return false;
    }
    let (b, h) = (dc_recur.rows(), dc_recur.cols());
    let Some((blocks, threads, units)) = slstm_fused_time_bwd_geometry(gpu, h, b) else {
        return false;
    };
    // Specialized build only — the block's `Wh` rows and per-unit accumulators are
    // register arrays sized by the shape, which a runtime bound would put in local
    // memory. Decline rather than launch a shape-generic twin; the caller has the
    // per-step loop.
    let Some(f) = gpu.kernels.specialized(
        &gpu.context,
        "slstm_fused_time_bwd",
        &[
            ("SLSTM_H", h),
            ("SLSTM_B", b),
            ("SLSTM_NJ", units),
            ("SLSTM_TH", threads),
        ],
    ) else {
        return false;
    };
    if !coop_grid_fits(gpu, &f, blocks, threads, 0) {
        return false;
    }
    let cfg = LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let t_i = t as i32;
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&wh.buf)
        .arg(&dy.buf)
        .arg(&mut g.buf)
        .arg(&mut dh_recur.buf)
        .arg(&mut dc_recur.buf)
        .arg(&mut dn_recur.buf);
    push_slab_ref!(lb, slabs.ot);
    push_slab_ref!(lb, slabs.c);
    push_slab_ref!(lb, slabs.n);
    push_slab_ref!(lb, slabs.c_entry);
    push_slab_ref!(lb, slabs.n_entry);
    push_slab_ref!(lb, slabs.zt);
    push_slab_ref!(lb, slabs.i_prime);
    push_slab_ref!(lb, slabs.f_prime);
    lb.arg(&t_i);
    // SAFETY: the geometry above guarantees a co-resident grid.
    match unsafe { lb.launch_cooperative(cfg) } {
        Ok(_) => true,
        Err(e) => {
            eprintln!("slstm_fused_time_bwd: cooperative launch failed: {e:?}");
            false
        }
    }
}

/// Hidden units one `slstm_batched_fwd` block owns. Sets the block width outright
/// (one warp per 8 of the `4*SB_NJ` gate columns, so `threads = 16 * nj`) and, with
/// it, how many column blocks the grid needs: `H / nj`.
///
/// It also sets how many COLUMN blocks the grid has (`cb = H / nj`) and therefore how
/// many times the two kernels re-read their big operand: every column block reads the
/// whole `h` (forward) or `dg` (backward) row for its batch rows, every timestep. Both
/// pull the same way — a narrower block means more of them — and which wins turns
/// entirely on where the grid lands against the SM count. See [`sb_nj_for`].
///
/// Only 16 and 32 are ever worth taking. 8 loses at every batch even where its grid
/// fits (67.6 us against 54.0 at B=32), and 64 needs a 1024-thread block, whose
/// 64-register budget spills the backward's `Wh` fragments (`ptxas -v`) — 82.0 us
/// against 77.3 at B=186.
const SB_NJ_CANDIDATES: [usize; 2] = [16, 32];

/// Ceiling, in 32-bit registers, on the `Wh` B-fragments a batched thread holds.
///
/// A warp keeps its gate columns over its whole share of the reduction axis in
/// registers — one `mma` B-fragment pair per k-tile. Past this the array spills to local
/// memory and gives back the whole win, so a wider `H` declines and takes another path.
/// 64 registers reaches H = 512.
const SB_MAX_WH_REGS: usize = 64;

/// Units per block for a `[B, H]` sweep: the NARROWEST whose grid stays inside
/// `max_blocks`, else the widest.
///
/// A cooperative grid must be fully resident, so more blocks than the device holds is
/// not a second wave — it doubles up on some SMs while the rest hold one block, and
/// every `grid.sync()` then waits on the doubled ones. Up to a point that is worth
/// paying: the narrower block gives twice the blocks for the same work, which hides
/// latency better, and its extra operand re-reads stay in L2. Measured at H=256, T=16
/// (us per pass, `rpt` = 1, grid = `(H / nj) * ceil(B / 16)`):
///
/// | B | 48 | 64 | 80 | 96 | 128 | 144 | 160 |
/// |---|----|----|----|----|-----|-----|-----|
/// | fwd nj=16 | 39.9 | 40.5 | 41.5 | 49.0 | 49.8 | 50.4 | 51.6 |
/// | fwd nj=32 | 48.4 | 48.5 | 48.4 | 49.4 | 50.0 | 50.8 | 52.1 |
/// | bwd nj=16 | 55.7 | 57.7 | 59.9 | 72.0 | 77.6 | 82.7 | 92.1 |
/// | bwd nj=32 | 66.3 | 68.3 | 69.9 | 74.5 | 78.1 | 81.3 | 86.0 |
///
/// So the two halves want different ceilings — the forward stays ahead to twice the SM
/// count, the backward only to about 1.5x, because its operand is `[B, 4H]` and the
/// re-reads cost four times as much. Each geometry passes its own.
///
/// Only 16 and 32 are ever worth taking. 8 loses at every batch even where its grid
/// fits (67.6 us against 54.0 at B=32), and 64 needs a 1024-thread block, whose
/// 64-register budget spills the backward's `Wh` fragments (`ptxas -v`).
fn sb_nj_for(h: usize, b: usize, max_blocks: usize) -> usize {
    if let Some(n) = std::env::var("SLSTM_BATCH_NJ").ok().and_then(|v| v.parse().ok()) {
        return n;
    }
    // `rpt` is 1 for every shape this path takes, so a block owns 16 rows.
    let rows = b.div_ceil(16);
    let last = SB_NJ_CANDIDATES[SB_NJ_CANDIDATES.len() - 1];
    SB_NJ_CANDIDATES
        .into_iter()
        .find(|&nj| h % nj == 0 && h / nj * rows <= max_blocks)
        .unwrap_or(last)
}

/// Blocks the forward may put on the device before a narrower block stops paying, and
/// the same for the backward — see [`sb_nj_for`] for the measurements behind the two.
fn sb_fwd_max_blocks(gpu: &Gpu) -> usize {
    2 * gpu.sm_count
}

fn sb_bwd_max_blocks(gpu: &Gpu) -> usize {
    3 * gpu.sm_count / 2
}

/// Widest `cb * B` — column blocks times batch rows — the batched path takes.
///
/// Both kernels re-read their big operand once per COLUMN block: the forward reads
/// `h` (`[B, H]`) `cb` times per timestep, the backward `dg` (`[B, 4H]`) `cb` times.
/// The per-step path reads each exactly once and pays two launches instead, so the
/// trade turns on whether `(cb - 1)` extra copies cost more than those launches — and
/// that grows with the batch while the launches do not.
///
/// Measured over the encoder's twenty length groups at H=256, where `cb` = 8: the
/// batched path runs 1.03-1.67x at B <= 186, sits inside the noise (0.96-1.08x) from
/// 204 to 256 and loses from 292 up. The cap is the bottom of that neutral band rather
/// than the top of it — the groups inside it are worth nothing either way, and the ones
/// above are a real loss.
///
/// Written in `cb * B` rather than `B` because it is the re-read row count that
/// decides, so a wider `H` — which needs more column blocks — moves the crossover down
/// on its own.
const SB_MAX_REREAD_ROWS: usize = 1536;

/// Whether the batched path is worth taking at this shape — see
/// [`SB_MAX_REREAD_ROWS`]. Correctness never depends on it: both kernels compute the
/// same thing at any batch the geometry accepts, and `SLstm::force_batched` overrides
/// this so a benchmark or a test can measure the losing arm.
pub fn slstm_batched_pays(gpu: &Gpu, h: usize, b: usize) -> bool {
    let nj = sb_nj_for(h, b, sb_bwd_max_blocks(gpu));
    nj > 0 && h / nj.max(1) * b <= SB_MAX_REREAD_ROWS
}

/// Rows-per-thread candidates for [`slstm_batched_geometry`], smallest first.
///
/// A block owns `16 * rpt` batch rows, and the recurrent state of every one of them
/// lives in registers. Raising it is the only handle on how many ROW blocks the grid
/// has — and it is never worth pulling: the M-tiles inside a block run in sequence
/// with a barrier between them, so halving the grid also doubles the serial chain.
/// Measured at H=256, T=16 (us, forward): `rpt` = 1 / 2 / 4 costs 54.6 / 85.4 / 158.9
/// at B=64 and 58.4 / 89.8 / 157.7 at B=96. The search therefore takes the first that
/// the device can place, which is 1 wherever 1 fits, and lets the host loop row chunks
/// rather than widening the block to avoid a second launch.
const SB_RPT_CANDIDATES: [usize; 4] = [1, 2, 4, 8];

/// `SLSTM_BATCH_RPT` pins the rows per thread, for sweeping the grid shape: `rpt` is
/// the only handle on how many ROW blocks the grid has, and how the grid lands against
/// the SM count is worth more here than anything inside a block.
fn sb_rpt_override() -> Option<usize> {
    std::env::var("SLSTM_BATCH_RPT").ok().and_then(|v| v.parse().ok())
}

/// Resolved launch geometry for [`slstm_batched_fwd`]: the grid is `(cb, blocks_y)`
/// and one launch covers `blocks_y * 16 * rpt` batch rows.
#[derive(Clone, Copy, Debug)]
pub struct SbGeom {
    pub nj: usize,
    pub rpt: usize,
    pub cb: usize,
    pub blocks_y: usize,
    pub threads: usize,
    pub shared: usize,
    /// Rows (forward) or reduction columns (backward) of the `Wh` slice one staging
    /// pass holds. A specialization constant, so it belongs with the rest of them.
    pub stage: usize,
}

impl SbGeom {
    /// Batch rows one cooperative launch covers.
    pub fn rows_per_launch(&self) -> usize {
        self.blocks_y * 16 * self.rpt
    }
}

/// Launch geometry for [`slstm_batched_fwd`], or `None` when the shape does not fit.
///
/// `nj` fixes the block width and the column half of the grid (`cb = H / nj`); what is
/// left to choose is `rpt`, the batch rows a block owns. Bigger `rpt` means fewer row
/// blocks — which matters only when the grid would otherwise not be co-resident — and
/// costs registers, shared memory (the `h` tile is `[16*rpt, H]`) and a longer serial
/// chain of M-tiles inside every timestep. So the search takes the SMALLEST `rpt` whose
/// grid still covers the whole batch in one launch, and only falls back to a wider one
/// when it cannot.
///
/// How many blocks are co-resident is the driver's answer, not `sm_count`: at the
/// encoder's shapes a block is small enough that two or three share an SM, and reading
/// the ceiling as one block per SM would force `rpt` up — leaving most of the device
/// idle behind a grid too narrow to fill it.
pub fn slstm_batched_geometry(gpu: &Gpu, h: usize, b: usize) -> Option<SbGeom> {
    cached_geom(gpu, false, h, b, || batched_fwd_geometry(gpu, h, b))
}

fn batched_fwd_geometry(gpu: &Gpu, h: usize, b: usize) -> Option<SbGeom> {
    // `SB_KT` must be even (the k-loop runs two accumulators), and the `Wh` fragments
    // are a register array of `2 * H / 16` entries.
    if h % 32 != 0 || b == 0 || 2 * h / 16 > SB_MAX_WH_REGS {
        return None;
    }
    let nj = sb_nj_for(h, b, sb_fwd_max_blocks(gpu));
    if nj < 2 || nj % 2 != 0 || h % nj != 0 || 16 * nj > MAX_BLOCK_THREADS {
        return None;
    }
    let (cb, threads) = (h / nj, 16 * nj);
    let smem_cap = gpu.max_shared_optin.saturating_sub(1024);
    let ld = h.next_multiple_of(16) + 8; // BF16_LD, in elements; a multiple of 8
    for rpt in SB_RPT_CANDIDATES {
        if sb_rpt_override().is_some_and(|r| r != rpt) {
            continue;
        }
        let br = 16 * rpt;
        // What is live for the whole T-loop: the gate scratch and the `h` tile.
        let live = 16 * 4 * nj * 4 + br * ld * 2;
        // The `Wh` staging overlays that same block before either region exists, and
        // runs in passes so it never widens it — sizing the allocation by the whole
        // slice would cost occupancy for the entire loop to save a prologue.
        let wsr = stage_rows(h, 4 * nj * 2, live);
        let shared = live.max(wsr * 4 * nj * 2);
        if shared > smem_cap {
            break;
        }
        let Some(f) = batched_fwd_kernel(gpu, h, nj, rpt, wsr, shared) else {
            break;
        };
        let per_sm = f
            .occupancy_max_active_blocks_per_multiprocessor(threads as u32, shared, None)
            .ok()? as usize;
        let max_blocks = per_sm * gpu.sm_count;
        if max_blocks < cb {
            break; // not even one row block would be co-resident
        }
        let blocks_y = (max_blocks / cb).min(b.div_ceil(br));
        let geom = SbGeom {
            nj,
            rpt,
            cb,
            blocks_y,
            threads,
            shared,
            stage: wsr,
        };
        return Some(geom);
    }
    None
}

/// The batched forward specialized to `(h, nj, rpt)`, with its shared-memory carve-out
/// opted into. Both are cached by the kernel cache, so this is a lookup after the
/// first call at a shape.
/// Rows (or columns) of the `Wh` slice one staging pass holds: the largest halving of
/// `total` whose `stride` bytes fit `budget`, never below one MMA k-tile. `total` is a
/// multiple of 32, so every halving down to 16 divides it and the passes tile it
/// exactly.
fn stage_rows(total: usize, stride: usize, budget: usize) -> usize {
    let mut n = total;
    while n > 16 && n * stride > budget {
        n /= 2;
    }
    n
}

fn batched_fwd_kernel(
    gpu: &Gpu,
    h: usize,
    nj: usize,
    rpt: usize,
    wsr: usize,
    shared: usize,
) -> Option<cudarc::driver::CudaFunction> {
    let f = gpu.kernels.specialized(
        &gpu.context,
        "slstm_batched_fwd",
        &[
            ("SLSTM_MMA", 1),
            ("SLSTM_H", h),
            ("SB_NJ", nj),
            ("SB_RPT", rpt),
            ("SB_WSR", wsr),
        ],
    )?;
    // Without this the launch fails for any tile above the default 48 KB.
    if let Err(e) = f.set_attribute(
        cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared as i32,
    ) {
        eprintln!("slstm_batched_fwd: shared-memory opt-in failed: {e:?}");
        return None;
    }
    Some(f)
}

/// The whole forward T-loop as **one cooperative launch per row chunk**, with the
/// recurrent product on the bf16 MMA unit: see `slstm_batched_fwd` in
/// `cuda/slstm_batched.cu`. The batch-parallel twin of [`slstm_fused_time`], for the
/// encoder and decoder — wide batch, a handful of timesteps.
///
/// `g` must already hold the input half `x·Wx` for every timestep; the kernel adds the
/// recurrent half in place and runs the recurrence. Returns `false` (having launched
/// nothing) when the kernel is unavailable or the shape does not fit.
#[allow(clippy::too_many_arguments)]
pub fn slstm_batched_fwd(
    gpu: &Gpu,
    wh: &GTensor<f32>,
    g: &mut GTensor<f32>,
    bcat: &GTensor<f32>,
    c_state: &mut GTensor<f32>,
    n_state: &mut GTensor<f32>,
    m_state: &mut GTensor<f32>,
    h_state: &mut GTensor<f32>,
    slabs: &mut SlstmSlabs,
    out: &mut GTensor<f32>,
    t: usize,
    carry: bool,
    cache: &super::temp::TempCache,
) -> bool {
    if !gpu.kernels.has_coop {
        return false;
    }
    let (b, h) = (c_state.rows(), c_state.cols());
    let Some(geom) = slstm_batched_geometry(gpu, h, b) else {
        return false;
    };
    let SbGeom {
        nj,
        rpt,
        cb,
        blocks_y,
        threads,
        shared,
        stage,
    } = geom;
    let Some(f) = batched_fwd_kernel(gpu, h, nj, rpt, stage, shared) else {
        return false;
    };
    let rows_per_launch = geom.rows_per_launch();
    let (t_i, b_i, carry_i) = (t as i32, b as i32, i32::from(carry));
    {
        let mut hmir = cache.get::<f32>(gpu, &[b * h]);
        let hmir = &mut *hmir;
        let mut row0 = 0usize;
        while row0 < b {
            // The last chunk gets a narrower grid rather than blocks that would sit
            // out the whole T-loop while still having to reach every `grid.sync()`.
            let by = (b - row0).div_ceil(16 * rpt).min(blocks_y);
            let cfg = LaunchConfig {
                grid_dim: (cb as u32, by as u32, 1),
                block_dim: (threads as u32, 1, 1),
                shared_mem_bytes: shared as u32,
            };
            let row0_i = row0 as i32;
            let mut lb = gpu.stream.launch_builder(&f);
            lb.arg(&wh.buf).arg(&mut g.buf).arg(&bcat.buf);
            push_slab!(lb, slabs.h_prev);
            lb.arg(&mut c_state.buf)
                .arg(&mut n_state.buf)
                .arg(&mut m_state.buf)
                .arg(&mut h_state.buf)
                .arg(&mut hmir.buf);
            push_slab!(lb, slabs.c_entry);
            push_slab!(lb, slabs.n_entry);
            push_slab!(lb, slabs.zt);
            push_slab!(lb, slabs.ot);
            push_slab!(lb, slabs.i_prime);
            push_slab!(lb, slabs.f_prime);
            push_slab!(lb, slabs.c);
            push_slab!(lb, slabs.n);
            lb.arg(&mut out.buf)
                .arg(&t_i)
                .arg(&b_i)
                .arg(&row0_i)
                .arg(&carry_i);
            // SAFETY: `coop_grid_fits` above confirmed the driver will place this grid,
            // and the geometry keeps every block's shared slice inside the opt-in.
            if let Err(e) = unsafe { lb.launch_cooperative(cfg) } {
                eprintln!("slstm_batched_fwd: cooperative launch failed: {e:?}");
                return false;
            }
            row0 += rows_per_launch;
        }
        true
    }
}

/// The whole forward T-loop as **one cooperative launch**: see `slstm_fused_time`
/// in `kernels.rs`. `g` must already hold the input half `x·Wx` for every
/// timestep; the kernel adds the recurrent half in place and runs the recurrence.
///
/// Returns `false` (having launched nothing) when the kernel is unavailable or the
/// shape does not fit, so callers can fall back to the per-step path.
#[allow(clippy::too_many_arguments)]
pub fn slstm_fused_time(
    gpu: &Gpu,
    wh: &GTensor<f32>,
    g: &mut GTensor<f32>,
    bcat: &GTensor<f32>,
    c_state: &mut GTensor<f32>,
    n_state: &mut GTensor<f32>,
    m_state: &mut GTensor<f32>,
    h_state: &mut GTensor<f32>,
    slabs: &mut SlstmSlabs,
    out: &mut GTensor<f32>,
    t: usize,
    carry: bool,
    cache: &super::temp::TempCache,
) -> bool {
    if !gpu.kernels.has_coop {
        return false;
    }
    let (b, h) = (c_state.rows(), c_state.cols());
    let Some((blocks, threads, units_per_block, staged_rows, shared_bytes)) =
        slstm_fused_time_geometry(gpu, h, b)
    else {
        return false;
    };
    // Specialized build only — the kernel keeps its slice of `h` in a register array
    // sized by H, which a runtime H would put in local memory. Decline rather than
    // launch a slower shape-generic twin; the caller has the per-step loop.
    let Some(f) = gpu.kernels.specialized(
        &gpu.context,
        "slstm_fused_time",
        &[("SLSTM_H", h), ("SLSTM_B", b), ("SLSTM_RS", staged_rows)],
    ) else {
        return false;
    };
    // Opt into the larger shared-memory carve-out; without this the launch fails
    // for any slice above the default 48 KB.
    if let Err(e) = f.set_attribute(
        cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared_bytes as i32,
    ) {
        eprintln!("slstm_fused_time: shared-memory opt-in failed: {e:?}");
        return false;
    }
    // After the opt-in, which is what decides how many blocks an SM can hold.
    let Some(threads) =
        fused_fwd_threads_for(gpu, &f, threads, b * units_per_block, blocks, shared_bytes)
    else {
        return false;
    };
    let cfg = LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: shared_bytes as u32,
    };
    let (t_i, upb_i, carry_i) = (t as i32, units_per_block as i32, i32::from(carry));
    // The kernel's `h` mirror: two [B, HP] bf16 planes packed into one fp32 [B, HP]
    // scratch (two bf16 to a float), ping-ponged so a block writing step t cannot
    // race a block still reading step t-1. `wtail` holds the rows of `Wh` that did
    // not fit shared memory, in the same packing; it is empty at the widths where
    // the whole slice is staged.
    {
        // Zeroed: the kernel reads the alternate plane before it has written it on the
        // first step, and the tail before its first `grid.sync()`.
        let mut hmir = cache.get_zeroed::<f32>(gpu, &[b, fused_hp(h)]);
        let hmir = &mut *hmir;
        {
            let mut wtail = cache.get_zeroed::<f32>(gpu, &[fused_tail_len(h, staged_rows).max(1)]);
            let wtail = &mut *wtail;
            let mut lb = gpu.stream.launch_builder(&f);
            lb.arg(&wh.buf).arg(&mut g.buf).arg(&bcat.buf);
            push_slab!(lb, slabs.h_prev);
            lb.arg(&mut c_state.buf)
                .arg(&mut n_state.buf)
                .arg(&mut m_state.buf)
                .arg(&mut h_state.buf)
                .arg(&mut hmir.buf)
                .arg(&mut wtail.buf)
                ;
            push_slab!(lb, slabs.c_entry);
            push_slab!(lb, slabs.n_entry);
            push_slab!(lb, slabs.zt);
            push_slab!(lb, slabs.ot);
            push_slab!(lb, slabs.i_prime);
            push_slab!(lb, slabs.f_prime);
            push_slab!(lb, slabs.c);
            push_slab!(lb, slabs.n);
            lb.arg(&mut out.buf)
                .arg(&t_i)
                .arg(&upb_i)
                .arg(&carry_i);
            // SAFETY: the geometry above guarantees the grid is co-resident (a cooperative
            // launch deadlocks otherwise) and that every block's shared slice fits.
            match unsafe { lb.launch_cooperative(cfg) } {
                Ok(_) => true,
                Err(e) => {
                    eprintln!("slstm_fused_time: cooperative launch failed: {e:?}");
                    false
                }
            }
        }
    }
}

/// Launch geometry for [`slstm_batched_bwd`], or `None` when the shape does not fit.
///
/// The same `(cb, blocks_y)` grid as the forward's, chosen the same way, but its
/// shared footprint is a different sum: the `dg` tile (one k-chunk of `[16*rpt,
/// 4H/rpt]`, so constant in `rpt`), the cross-warp reduction buffer and the `dh`
/// exchange, against the forward's `h` tile.
pub fn slstm_batched_bwd_geometry(gpu: &Gpu, h: usize, b: usize) -> Option<SbGeom> {
    cached_geom(gpu, true, h, b, || batched_bwd_geometry(gpu, h, b))
}

fn batched_bwd_geometry(gpu: &Gpu, h: usize, b: usize) -> Option<SbGeom> {
    if h % 32 != 0 || b == 0 || 2 * h / 16 > SB_MAX_WH_REGS {
        return None;
    }
    let nj = sb_nj_for(h, b, sb_bwd_max_blocks(gpu));
    // A warp owns 8 units and the warps split the reduction four ways, so `nj` must be
    // a multiple of 8 — the forward only needs it even.
    if nj % 8 != 0 || h % nj != 0 || 16 * nj > MAX_BLOCK_THREADS {
        return None;
    }
    let (cb, threads) = (h / nj, 16 * nj);
    let wk = (nj / 2) / (nj / 8);
    let smem_cap = gpu.max_shared_optin.saturating_sub(1024);
    for rpt in SB_RPT_CANDIDATES {
        if sb_rpt_override().is_some_and(|r| r != rpt) {
            continue;
        }
        let (br, kc) = (16 * rpt, 4 * h / rpt);
        if kc % (16 * wk) != 0 {
            continue; // a warp's share of a chunk is not a whole number of k-tiles
        }
        let live = br * (kc + 8) * 2 + wk * br * nj * 4 + br * nj * 4;
        let wsc = stage_rows(4 * h, nj * 2, live);
        let shared = live.max(nj * wsc * 2);
        if shared > smem_cap {
            break;
        }
        let Some(f) = batched_bwd_kernel(gpu, h, nj, rpt, wsc, shared) else {
            break;
        };
        let per_sm = f
            .occupancy_max_active_blocks_per_multiprocessor(threads as u32, shared, None)
            .ok()? as usize;
        let max_blocks = per_sm * gpu.sm_count;
        if max_blocks < cb {
            break;
        }
        let geom = SbGeom {
            nj,
            rpt,
            cb,
            blocks_y: (max_blocks / cb).min(b.div_ceil(br)),
            threads,
            shared,
            stage: wsc,
        };
        return Some(geom);
    }
    None
}

fn batched_bwd_kernel(
    gpu: &Gpu,
    h: usize,
    nj: usize,
    rpt: usize,
    wsc: usize,
    shared: usize,
) -> Option<cudarc::driver::CudaFunction> {
    let f = gpu.kernels.specialized(
        &gpu.context,
        "slstm_batched_bwd",
        &[
            ("SLSTM_MMA", 1),
            ("SLSTM_H", h),
            ("SB_NJ", nj),
            ("SB_RPT", rpt),
            ("SB_WSC", wsc),
        ],
    )?;
    if let Err(e) = f.set_attribute(
        cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared as i32,
    ) {
        eprintln!("slstm_batched_bwd: shared-memory opt-in failed: {e:?}");
        return None;
    }
    Some(f)
}

/// The whole reverse T-loop as **one cooperative launch per row chunk**: see
/// `slstm_batched_bwd` in `cuda/slstm_batched.cu`. Writes the gate deltas into `g`
/// (the post-loop `dWx`/`dWh`/`db` GEMMs read them there) and carries the BPTT
/// channels in registers.
///
/// Returns `false` (having launched nothing) when the kernel is unavailable or the
/// shape does not fit.
#[allow(clippy::too_many_arguments)]
pub fn slstm_batched_bwd(
    gpu: &Gpu,
    wh: &GTensor<f32>,
    dy: &GTensor<f32>,
    g: &mut GTensor<f32>,
    dh_recur: &mut GTensor<f32>,
    dc_recur: &mut GTensor<f32>,
    dn_recur: &mut GTensor<f32>,
    slabs: &SlstmSlabs,
    t: usize,
    cache: &super::temp::TempCache,
) -> bool {
    if !gpu.kernels.has_coop {
        return false;
    }
    let (b, h) = (dc_recur.rows(), dc_recur.cols());
    let Some(geom) = slstm_batched_bwd_geometry(gpu, h, b) else {
        return false;
    };
    let SbGeom {
        nj,
        rpt,
        cb,
        blocks_y,
        threads,
        shared,
        stage,
    } = geom;
    let Some(f) = batched_bwd_kernel(gpu, h, nj, rpt, stage, shared) else {
        return false;
    };
    let rows_per_launch = geom.rows_per_launch();
    let (t_i, b_i) = (t as i32, b as i32);
    {
        let mut dgmir = cache.get::<f32>(gpu, &[b * 4 * h]);
        let dgmir = &mut *dgmir;
        let mut row0 = 0usize;
        while row0 < b {
            let by = (b - row0).div_ceil(16 * rpt).min(blocks_y);
            let cfg = LaunchConfig {
                grid_dim: (cb as u32, by as u32, 1),
                block_dim: (threads as u32, 1, 1),
                shared_mem_bytes: shared as u32,
            };
            let row0_i = row0 as i32;
            let mut lb = gpu.stream.launch_builder(&f);
            lb.arg(&wh.buf)
                .arg(&dy.buf)
                .arg(&mut g.buf)
                .arg(&mut dh_recur.buf)
                .arg(&mut dc_recur.buf)
                .arg(&mut dn_recur.buf)
                .arg(&mut dgmir.buf);
            push_slab_ref!(lb, slabs.ot);
            push_slab_ref!(lb, slabs.c);
            push_slab_ref!(lb, slabs.n);
            push_slab_ref!(lb, slabs.c_entry);
            push_slab_ref!(lb, slabs.n_entry);
            push_slab_ref!(lb, slabs.zt);
            push_slab_ref!(lb, slabs.i_prime);
            push_slab_ref!(lb, slabs.f_prime);
            lb.arg(&t_i).arg(&b_i).arg(&row0_i);
            // SAFETY: the geometry confirmed the driver will place this grid, and every
            // block's shared slice is inside the opt-in.
            if let Err(e) = unsafe { lb.launch_cooperative(cfg) } {
                eprintln!("slstm_batched_bwd: cooperative launch failed: {e:?}");
                return false;
            }
            row0 += rows_per_launch;
        }
        true
    }
}

/// One fused backward step of the sLSTM cell.
///
/// Reads the biased forget pre-activation out of `gates` and then overwrites all
/// four of its blocks with the gate *deltas* — one buffer carries the gate
/// pre-activations forward and their grads backward, since the forward contents
/// are dead once read. The same deltas also go to the contiguous `d_gates_flat`
/// scratch so `dh = d_gates·Whᵀ` stays a dense GEMM at any batch size.
///
/// * `d_out` — `[B, T, H]` grad of this layer's output `h_t`.
/// * `gates` — `[B, T, 4H]` gate pre-activations in, gate deltas out.
/// * `d_gates_flat` — `[B, 4H]` this step's deltas, contiguous and at the slab
///   width, so the dh GEMM reads a tensor-core operand without a cast of its own.
/// * `d_h_recur` — `[B, H]` grad arriving from step `t+1` through `h`.
/// * `slabs` — the forward's saved activations (`o`, `c`, `n`, `z`, `i'`, `f'`).
/// * `d_c_recur` / `d_n_recur` — `[B, H]` cell and normalizer grads, carried
///   back a step in place.
/// * `t` — the timestep being differentiated.
#[allow(clippy::too_many_arguments)]
pub fn slstm_step_fused_bwd(
    gpu: &Gpu,
    d_out: &GTensor<f32>,
    gates: &mut GTensor<f32>,
    d_gates_flat: &mut SlabBuf,
    d_h_recur: &GTensor<f32>,
    slabs: &SlstmSlabs,
    d_c_recur: &mut GTensor<f32>,
    d_n_recur: &mut GTensor<f32>,
    t: usize,
) {
    let (b, h) = (d_c_recur.rows(), d_c_recur.cols());
    let bh = b * h;
    let big_t = d_out.shape[1];
    let (t_i, bigt_i, h_i, bh_i) = (t as i32, big_t as i32, h as i32, bh as i32);
    let f = gpu.kernels.get("slstm_step_fused_bwd");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&d_out.buf).arg(&mut gates.buf);
    push_slab!(lb, *d_gates_flat);
    lb.arg(&d_h_recur.buf);
    push_slab_ref!(lb, slabs.ot);
    push_slab_ref!(lb, slabs.c);
    push_slab_ref!(lb, slabs.n);
    push_slab_ref!(lb, slabs.c_entry);
    push_slab_ref!(lb, slabs.n_entry);
    push_slab_ref!(lb, slabs.zt);
    push_slab_ref!(lb, slabs.i_prime);
    push_slab_ref!(lb, slabs.f_prime);
    lb.arg(&mut d_c_recur.buf)
        .arg(&mut d_n_recur.buf)
        .arg(&t_i)
        .arg(&bigt_i)
        .arg(&h_i)
        .arg(&bh_i);
    unsafe { lb.launch(elem_cfg(gpu, bh as u32)) }.expect("slstm_step_fused_bwd");
}

// Residual block / SwiGLU kernels (see gpu/block.rs).

/// Elementwise `out = a + b` (fresh allocation). Used for residual adds and the
/// grad accumulations that are plain sums.
pub fn add(gpu: &Gpu, a: &GTensor<f32>, b: &GTensor<f32>) -> GTensor<f32> {
    let mut out = GTensor::uninit(gpu, a.dims());
    add_into(gpu, a, b, &mut out);
    out
}

/// In-place `acc += b`.
///
/// The running sums in mLSTM's backward (`dkc`, `dqc`, `dxf`, …) were written as
/// `x = add(&x, &term)`, which allocates a buffer per term and drops the previous
/// one. This accumulates into `acc` instead, so a sum of k terms costs no
/// allocations at all.
pub fn add_assign(gpu: &Gpu, acc: &mut GTensor<f32>, b: &GTensor<f32>) {
    let n = acc.len();
    assert_eq!(n, b.len(), "add_assign: length mismatch");
    let n_i = n as i32;
    let f = gpu.kernels.get("add_assign");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut acc.buf).arg(&b.buf).arg(&n_i);
    unsafe { lb.launch(elem_cfg(gpu, n as u32)) }.expect("add_assign");
}

/// Elementwise `out = a + b` into a caller-owned buffer. The allocating [`add`]
/// is a thin wrapper over this; layers that own their buffers call this directly
/// so a residual add costs no allocation.
///
/// `out` may alias `a` or `b` (the kernel reads element `i` before writing it).
pub fn add_into(gpu: &Gpu, a: &GTensor<f32>, b: &GTensor<f32>, out: &mut GTensor<f32>) {
    let n = a.len();
    assert_eq!(n, b.len(), "add: length mismatch");
    assert_eq!(n, out.len(), "add: output length mismatch");
    let n_i = n as i32;
    let f = gpu.kernels.get("add");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut out.buf).arg(&a.buf).arg(&b.buf).arg(&n_i);
    unsafe { lb.launch(elem_cfg(gpu, n as u32)) }.expect("add");
}

/// SwiGLU forward: returns `(gate_act = SiLU(gate_pre), mixed = gate_act ⊙ value)`.
pub fn swiglu_forward(
    gpu: &Gpu,
    gate_pre: &GTensor<f32>,
    value: &GTensor<f32>,
) -> (GTensor<f32>, GTensor<f32>) {
    let mut gate_act = GTensor::uninit(gpu, gate_pre.dims());
    let mut mixed = GTensor::uninit(gpu, gate_pre.dims());
    swiglu_forward_into(gpu, gate_pre, value, &mut gate_act, &mut mixed);
    (gate_act, mixed)
}

/// SwiGLU forward into caller-owned buffers — the no-allocation form of
/// [`swiglu_forward`]. Both outputs are written in full.
pub fn swiglu_forward_into(
    gpu: &Gpu,
    gate_pre: &GTensor<f32>,
    value: &GTensor<f32>,
    gate_act: &mut GTensor<f32>,
    mixed: &mut GTensor<f32>,
) {
    let n = gate_pre.len();
    assert_eq!(n, value.len(), "swiglu_forward: length mismatch");
    assert_eq!(
        n,
        gate_act.len(),
        "swiglu_forward: gate_act length mismatch"
    );
    assert_eq!(n, mixed.len(), "swiglu_forward: mixed length mismatch");
    let n_i = n as i32;
    let f = gpu.kernels.get("swiglu_forward");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&gate_pre.buf)
        .arg(&value.buf)
        .arg(&mut gate_act.buf)
        .arg(&mut mixed.buf)
        .arg(&n_i);
    unsafe { lb.launch(elem_cfg(gpu, n as u32)) }.expect("swiglu_forward");
}

/// SwiGLU backward: from `d_mixed` and the saved `gate_act`/`value`/`gate_pre`,
/// returns `(d_gate, d_value)`.
pub fn swiglu_backward(
    gpu: &Gpu,
    d_mixed: &GTensor<f32>,
    gate_act: &GTensor<f32>,
    value: &GTensor<f32>,
    gate_pre: &GTensor<f32>,
) -> (GTensor<f32>, GTensor<f32>) {
    let mut d_gate = GTensor::uninit(gpu, d_mixed.dims());
    let mut d_value = GTensor::uninit(gpu, d_mixed.dims());
    swiglu_backward_into(
        gpu,
        d_mixed,
        gate_act,
        value,
        gate_pre,
        &mut d_gate,
        &mut d_value,
    );
    (d_gate, d_value)
}

/// SwiGLU backward into caller-owned buffers — the no-allocation form of
/// [`swiglu_backward`].
#[allow(clippy::too_many_arguments)]
pub fn swiglu_backward_into(
    gpu: &Gpu,
    d_mixed: &GTensor<f32>,
    gate_act: &GTensor<f32>,
    value: &GTensor<f32>,
    gate_pre: &GTensor<f32>,
    d_gate: &mut GTensor<f32>,
    d_value: &mut GTensor<f32>,
) {
    let n = d_mixed.len();
    assert_eq!(n, d_gate.len(), "swiglu_backward: d_gate length mismatch");
    assert_eq!(n, d_value.len(), "swiglu_backward: d_value length mismatch");
    let n_i = n as i32;
    let f = gpu.kernels.get("swiglu_backward");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&d_mixed.buf)
        .arg(&gate_act.buf)
        .arg(&value.buf)
        .arg(&gate_pre.buf)
        .arg(&mut d_gate.buf)
        .arg(&mut d_value.buf)
        .arg(&n_i);
    unsafe { lb.launch(elem_cfg(gpu, n as u32)) }.expect("swiglu_backward");
}

// mLSTM parallel/chunkwise core (see gpu/mlstm.rs).

/// Inclusive cumsum of logσ along T, per row of `f` `[BH, T]` → `fc` `[BH, T]`.
pub fn cumsum_logsig(gpu: &Gpu, f: &GTensor<f32>) -> GTensor<f32> {
    let (bh, t) = (f.rows(), f.cols());
    let mut fc = GTensor::uninit(gpu, &[bh, t]);
    let (ti, bhi) = (t as i32, bh as i32);
    if !no_block_scan() {
        // One block per row: the row is contiguous across lanes and the scan is a
        // shuffle tree. The thread-per-row form below reads a full row apart per
        // lane and fits the whole launch in one block (grid=1 at BH=1024).
        let func = gpu.kernels.get("cumsum_logsig_block");
        let mut lb = gpu.stream.launch_builder(&func);
        lb.arg(&f.buf).arg(&mut fc.buf).arg(&ti).arg(&bhi);
        let cfg = LaunchConfig {
            grid_dim: (bh as u32, 1, 1),
            block_dim: (scan_block_dim(t), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { lb.launch(cfg) }.expect("cumsum_logsig_block");
        return fc;
    }
    let func = gpu.kernels.get("cumsum_logsig");
    let mut lb = gpu.stream.launch_builder(&func);
    lb.arg(&f.buf).arg(&mut fc.buf).arg(&ti).arg(&bhi);
    unsafe { lb.launch(elem_cfg(gpu, bh as u32)) }.expect("cumsum_logsig");
    fc
}

/// `GPU_NO_BLOCK_SCAN=1` reverts the gate scans to the thread-per-row kernels.
fn no_block_scan() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("GPU_NO_BLOCK_SCAN").is_ok_and(|v| v != "0"))
}

/// Block width for the per-row scans: the row length rounded up to a warp, capped at
/// 1024. A row shorter than the block leaves the tail lanes idle but still coalesced.
fn scan_block_dim(t: usize) -> u32 {
    t.div_ceil(32).clamp(1, 32) as u32 * 32
}

// mLSTM chunking (inter-chunk state carry; see gpu/mlstm.rs).

/// [`slice_t`] with the `[BH, T, W]` interpretation given explicitly, so a rank-2
/// `[BH, T]` tensor can be sliced as `[BH, T, 1]` without being reshaped.
///
/// `reshaped` consumes `self`, so calling it on a borrow needs an owned tensor —
/// and the call sites reached for `.dup(gpu)`, copying a whole `[BH, T]` tensor,
/// once per chunk, purely to satisfy the borrow checker. Nothing about the copy
/// was load-bearing: the kernel takes the shape as parameters.
pub fn slice_t_as(
    gpu: &Gpu,
    x: &GTensor<f32>,
    bh: usize,
    t: usize,
    w: usize,
    c0: usize,
    len: usize,
) -> GTensor<f32> {
    assert_eq!(
        bh * t * w,
        x.len(),
        "slice_t_as: dims do not cover the tensor"
    );
    slice_t_inner(gpu, x, bh, t, w, c0, len)
}

/// Extract the T-range `[c0, c0+len)` of a head-major `[BH, T, W]` tensor.
pub fn slice_t(gpu: &Gpu, x: &GTensor<f32>, c0: usize, len: usize) -> GTensor<f32> {
    let (bh, t, w) = (x.shape[0], x.shape[1], x.shape[2]);
    slice_t_inner(gpu, x, bh, t, w, c0, len)
}

fn slice_t_inner(
    gpu: &Gpu,
    x: &GTensor<f32>,
    bh: usize,
    t: usize,
    w: usize,
    c0: usize,
    len: usize,
) -> GTensor<f32> {
    assert!(
        c0 + len <= t,
        "slice_t: chunk [{c0}, {}) out of range T={t}",
        c0 + len
    );
    let mut out = GTensor::uninit(gpu, &[bh, len, w]);
    let total = bh * len * w;
    let (ti, li, c0i, wi, total_i) = (t as i32, len as i32, c0 as i32, w as i32, total as i32);
    let f = gpu.kernels.get("slice_t");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&x.buf)
        .arg(&mut out.buf)
        .arg(&ti)
        .arg(&li)
        .arg(&c0i)
        .arg(&wi)
        .arg(&total_i);
    unsafe { lb.launch(elem_cfg(gpu, total as u32)) }.expect("slice_t");
    out
}

/// Most tensors one batched slice/unslice launch can carry. Must match
/// `SLICE_BATCH_MAX` in `mlstm_ops.cu`.
pub const SLICE_BATCH_MAX: usize = 8;

/// `GPU_NO_SLICE_BATCH=1` reverts slice/unslice to one launch per tensor.
fn no_slice_batch() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("GPU_NO_SLICE_BATCH").is_ok_and(|v| v != "0"))
}

/// Kernel-argument twin of `SliceBatch` in `mlstm_ops.cu`. Passed by value, so the
/// field order and types must match the `.cu` definition exactly.
#[repr(C)]
#[derive(Clone, Copy)]
struct SliceBatchArg {
    src: [u64; SLICE_BATCH_MAX],
    dst: [u64; SLICE_BATCH_MAX],
    w: [i32; SLICE_BATCH_MAX],
    off: [i32; SLICE_BATCH_MAX + 1],
    n: i32,
}

// SAFETY: a plain-old-data struct of pointers and integers, passed by value as a
// kernel argument exactly as `cudarc` does for the scalar types.
unsafe impl cudarc::driver::DeviceRepr for SliceBatchArg {}

/// Slice the same T-range out of several tensors in one launch.
///
/// The chunked mLSTM sweep slices q/k/v and the two gate rows (forward), and five
/// gradient tensors (backward), out of an identical time range back to back. Each is
/// a ~2 us copy, so separate launches are dominated by launch cost.
///
/// Each entry is `(tensor, W)` with the width given explicitly rather than read from
/// `shape[2]`: the gate tensors are rank-2 `[BH, T]` (width 1), so a rank-3 reading
/// would take a garbage width. `bh` and `t` are likewise explicit — this is the
/// batched [`slice_t_as`], and outputs come back in input order as `[BH, len, W]`.
pub fn slice_t_batch(
    gpu: &Gpu,
    srcs: &[(&GTensor<f32>, usize)],
    bh: usize,
    t: usize,
    c0: usize,
    len: usize,
) -> Vec<GTensor<f32>> {
    assert!(
        srcs.len() <= SLICE_BATCH_MAX,
        "slice_t_batch: {} tensors exceeds SLICE_BATCH_MAX",
        srcs.len()
    );
    assert!(
        c0 + len <= t,
        "slice_t_batch: chunk [{c0}, {}) out of range T={t}",
        c0 + len
    );

    // A/B toggle: fall back to one launch per tensor, so both paths can be measured
    // in the same process (thermal drift makes cross-process timings incomparable).
    if no_slice_batch() {
        return srcs
            .iter()
            .map(|&(x, w)| slice_t_as(gpu, x, bh, t, w, c0, len))
            .collect();
    }

    let mut outs: Vec<GTensor<f32>> = srcs
        .iter()
        .map(|&(_, w)| GTensor::uninit(gpu, &[bh, len, w]))
        .collect();
    let mut a = SliceBatchArg {
        src: [0; SLICE_BATCH_MAX],
        dst: [0; SLICE_BATCH_MAX],
        w: [0; SLICE_BATCH_MAX],
        off: [0; SLICE_BATCH_MAX + 1],
        n: srcs.len() as i32,
    };
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let mut total = 0usize;
    // The guards keep each buffer's pointer valid for this stream; they are scoped so
    // they release the borrow on `outs` right after the launch is queued.
    {
        let mut guards = Vec::with_capacity(srcs.len() * 2);
        for (i, (&(x, w), out)) in srcs.iter().zip(outs.iter_mut()).enumerate() {
            assert_eq!(
                bh * t * w,
                x.len(),
                "slice_t_batch: dims do not cover the tensor"
            );
            let (ps, gs) = x.buf.device_ptr(&gpu.stream);
            let (pd, gd) = out.buf.device_ptr_mut(&gpu.stream);
            a.src[i] = ps;
            a.dst[i] = pd;
            guards.push(gs);
            guards.push(gd);
            a.w[i] = w as i32;
            a.off[i] = total as i32;
            total += bh * len * w;
        }
        a.off[srcs.len()] = total as i32;

        let (ti, li, c0i, total_i) = (t as i32, len as i32, c0 as i32, total as i32);
        let f = gpu.kernels.get("slice_t_batch");
        let mut lb = gpu.stream.launch_builder(&f);
        lb.arg(&a).arg(&ti).arg(&li).arg(&c0i).arg(&total_i);
        unsafe { lb.launch(elem_cfg(gpu, total as u32)) }.expect("slice_t_batch");
    }
    outs
}

/// Write several chunks back into their full tensors in one launch — the
/// [`slice_t_batch`] inverse. Each entry is `(dst, src, W)`, with `W` explicit for
/// the same reason as in `slice_t_batch`: a source may be rank-2.
pub fn unslice_t_batch(
    gpu: &Gpu,
    pairs: &mut [(&mut GTensor<f32>, &GTensor<f32>, usize)],
    bh: usize,
    t: usize,
    c0: usize,
    len: usize,
) {
    assert!(
        pairs.len() <= SLICE_BATCH_MAX,
        "unslice_t_batch: {} tensors exceeds SLICE_BATCH_MAX",
        pairs.len()
    );
    assert!(
        c0 + len <= t,
        "unslice_t_batch: chunk [{c0}, {}) out of range T={t}",
        c0 + len
    );

    if no_slice_batch() {
        for (dst, src, w) in pairs.iter_mut() {
            unslice_t_inner(gpu, dst, src, bh, t, *w, len, c0);
        }
        return;
    }

    let mut a = SliceBatchArg {
        src: [0; SLICE_BATCH_MAX],
        dst: [0; SLICE_BATCH_MAX],
        w: [0; SLICE_BATCH_MAX],
        off: [0; SLICE_BATCH_MAX + 1],
        n: pairs.len() as i32,
    };
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let n_pairs = pairs.len();
    let mut guards = Vec::with_capacity(n_pairs * 2);
    let mut total = 0usize;
    for (i, (dst, src, w)) in pairs.iter_mut().enumerate() {
        let w = *w;
        assert_eq!(
            bh * t * w,
            dst.len(),
            "unslice_t_batch: dims do not cover the destination"
        );
        assert_eq!(
            src.len(),
            bh * len * w,
            "unslice_t_batch: source size mismatch"
        );
        let (ps, gs) = src.buf.device_ptr(&gpu.stream);
        let (pd, gd) = dst.buf.device_ptr_mut(&gpu.stream);
        a.src[i] = ps;
        a.dst[i] = pd;
        guards.push(gs);
        guards.push(gd);
        a.w[i] = w as i32;
        a.off[i] = total as i32;
        total += bh * len * w;
    }
    a.off[n_pairs] = total as i32;

    let (ti, li, c0i, total_i) = (t as i32, len as i32, c0 as i32, total as i32);
    let f = gpu.kernels.get("unslice_t_batch");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&a).arg(&ti).arg(&li).arg(&c0i).arg(&total_i);
    unsafe { lb.launch(elem_cfg(gpu, total as u32)) }.expect("unslice_t_batch");
}

/// Write `src` `[BH, len, W]` back into the T-range `[c0, c0+len)` of `dst`
/// `[BH, T, W]`. Chunks partition T, so this is a store, not an accumulate.
/// [`unslice_t`] with the source's `[BH, len, W]` interpretation given explicitly,
/// so a rank-2 source needs no reshape (and hence no `dup`). See [`slice_t_as`].
pub fn unslice_t_as(gpu: &Gpu, dst: &mut GTensor<f32>, src: &GTensor<f32>, len: usize, c0: usize) {
    let (bh, t, w) = (dst.shape[0], dst.shape[1], dst.shape[2]);
    unslice_t_inner(gpu, dst, src, bh, t, w, len, c0);
}

pub fn unslice_t(gpu: &Gpu, dst: &mut GTensor<f32>, src: &GTensor<f32>, c0: usize) {
    let (bh, t, w) = (dst.shape[0], dst.shape[1], dst.shape[2]);
    let len = src.shape[1];
    unslice_t_inner(gpu, dst, src, bh, t, w, len, c0);
}

fn unslice_t_inner(
    gpu: &Gpu,
    dst: &mut GTensor<f32>,
    src: &GTensor<f32>,
    bh: usize,
    t: usize,
    w: usize,
    len: usize,
    c0: usize,
) {
    assert!(
        c0 + len <= t,
        "unslice_t: chunk [{c0}, {}) out of range T={t}",
        c0 + len
    );
    let total = bh * len * w;
    let (ti, li, c0i, wi, total_i) = (t as i32, len as i32, c0 as i32, w as i32, total as i32);
    let f = gpu.kernels.get("unslice_t");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut dst.buf)
        .arg(&src.buf)
        .arg(&ti)
        .arg(&li)
        .arg(&c0i)
        .arg(&wi)
        .arg(&total_i);
    unsafe { lb.launch(elem_cfg(gpu, total as u32)) }.expect("unslice_t");
}

/// `out[r] += Σ_w x[r,w]·y[r,w]` for `[R, W]` operands (`out` is `[R]`).
pub fn row_dot_add(
    gpu: &Gpu,
    out: &mut GTensor<f32>,
    x: &GTensor<f32>,
    y: &GTensor<f32>,
    w: usize,
) {
    let r = out.len();
    assert_eq!(x.len(), r * w, "row_dot_add: x length mismatch");
    assert_eq!(y.len(), r * w, "row_dot_add: y length mismatch");
    let (wi, ri) = (w as i32, r as i32);
    let f = gpu.kernels.get("row_dot_add");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut out.buf)
        .arg(&x.buf)
        .arg(&y.buf)
        .arg(&wi)
        .arg(&ri);
    unsafe { lb.launch(row_block_cfg(r, w)) }.expect("row_dot_add");
}

/// `out[g] += Σ_e x[g,e]·y[g,e]` for `[G, E]` operands (`out` is `[G]`) — the
/// per-head reduction behind `dg`.
pub fn group_dot_add(gpu: &Gpu, out: &mut GTensor<f32>, x: &GTensor<f32>, y: &GTensor<f32>) {
    let g = out.len();
    let e = x.len() / g;
    assert_eq!(x.len(), y.len(), "group_dot_add: x/y length mismatch");
    assert_eq!(
        x.len(),
        g * e,
        "group_dot_add: not divisible by group count"
    );
    let (ei, gi) = (e as i32, g as i32);
    let f = gpu.kernels.get("group_dot_add");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut out.buf)
        .arg(&x.buf)
        .arg(&y.buf)
        .arg(&ei)
        .arg(&gi);
    unsafe { lb.launch(row_block_cfg(g, e)) }.expect("group_dot_add");
}

/// Backward of [`mlstm_chunk_ab`], accumulating into the `dfc`/`dig` that
/// [`mlstm_dfc_dig`] already wrote from the intra-chunk D̄.
pub fn mlstm_chunk_ab_bwd(
    gpu: &Gpu,
    db: &GTensor<f32>,
    da: &GTensor<f32>,
    b: &GTensor<f32>,
    a: &GTensor<f32>,
    dfc: &mut GTensor<f32>,
    dig: &mut GTensor<f32>,
) {
    let (bh, l) = (b.rows(), b.cols());
    let (li, bhl) = (l as i32, (bh * l) as i32);
    if !no_block_scan() {
        // One block per row: the `t == L-1` thread's serial row sum becomes a block
        // reduction, and the whole launch stops fitting in two blocks.
        let bh_i = bh as i32;
        let f = gpu.kernels.get("mlstm_chunk_ab_bwd_block");
        let mut lb = gpu.stream.launch_builder(&f);
        lb.arg(&db.buf)
            .arg(&da.buf)
            .arg(&b.buf)
            .arg(&a.buf)
            .arg(&mut dfc.buf)
            .arg(&mut dig.buf)
            .arg(&li)
            .arg(&bh_i);
        let cfg = LaunchConfig {
            grid_dim: (bh as u32, 1, 1),
            block_dim: (scan_block_dim(l), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { lb.launch(cfg) }.expect("mlstm_chunk_ab_bwd_block");
        return;
    }
    let f = gpu.kernels.get("mlstm_chunk_ab_bwd");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&db.buf)
        .arg(&da.buf)
        .arg(&b.buf)
        .arg(&a.buf)
        .arg(&mut dfc.buf)
        .arg(&mut dig.buf)
        .arg(&li)
        .arg(&bhl);
    unsafe { lb.launch(elem_cfg(gpu, (bh * l) as u32)) }.expect("mlstm_chunk_ab_bwd");
}

/// Copy rows of `src` into arbitrary rows of `dst`: `dst[row_ids[i]] = src[i]`.
/// The inverse (pulling those rows back out) is [`embedding_gather`] with the same
/// row ids, treating the matrix as the "table".
pub fn scatter_rows(gpu: &Gpu, dst: &mut GTensor<f32>, src: &GTensor<f32>, row_ids: &[usize]) {
    assert_eq!(
        src.rows(),
        row_ids.len(),
        "scatter_rows: row_ids len != src rows"
    );
    scatter_rows_u32(gpu, dst, src, &upload_ids(gpu, row_ids).slice(..));
}

/// [`scatter_rows`] against row ids already resident on the device. The row
/// count comes from `src`, so the id buffer may be a larger reused allocation.
pub fn scatter_rows_u32(
    gpu: &Gpu,
    dst: &mut GTensor<f32>,
    src: &GTensor<f32>,
    ids: &CudaView<'_, u32>,
) {
    let dim = src.cols();
    let rows = src.rows();
    let (dim_i, rows_i) = (dim as i32, rows as i32);
    let f = gpu.kernels.get("scatter_rows");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut dst.buf)
        .arg(&src.buf)
        .arg(ids)
        .arg(&dim_i)
        .arg(&rows_i);
    unsafe { lb.launch(elem_cfg(gpu, (rows * dim) as u32)) }.expect("scatter_rows");
}

/// `dst[dst_ids[i]] = src[src_ids[i]]` for `rows` rows of `dim` floats — a gather and a
/// scatter in one pass, with no temporary between them.
pub fn route_rows_u32(
    gpu: &Gpu,
    dst: &mut GTensor<f32>,
    src: &GTensor<f32>,
    src_ids: &CudaView<'_, u32>,
    dst_ids: &CudaView<'_, u32>,
    rows: usize,
) {
    let dim = src.cols();
    assert_eq!(dim, dst.cols(), "route_rows: width {dim} != {}", dst.cols());
    let (dim_i, rows_i) = (dim as i32, rows as i32);
    let f = gpu.kernels.get("route_rows");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut dst.buf)
        .arg(&src.buf)
        .arg(src_ids)
        .arg(dst_ids)
        .arg(&dim_i)
        .arg(&rows_i);
    unsafe { lb.launch(elem_cfg(gpu, (rows * dim) as u32)) }.expect("route_rows");
}

/// `dst[ids[i]] = src[i * stride + offset]` — the rectangle-to-window half of a
/// readout. A group holds one word length, so the rectangle side needs no id table.
pub fn pack_rows_u32(
    gpu: &Gpu,
    dst: &mut GTensor<f32>,
    src: &GTensor<f32>,
    ids: &cudarc::driver::CudaView<'_, u32>,
    stride: usize,
    offset: usize,
    rows: usize,
) {
    strided_rows(gpu, "pack_rows", dst, src, ids, stride, offset, rows);
}

/// `dst[i * stride + offset] = src[ids[i]]` — the window-to-rectangle direction of
/// [`pack_rows_u32`].
pub fn unpack_rows_u32(
    gpu: &Gpu,
    dst: &mut GTensor<f32>,
    src: &GTensor<f32>,
    ids: &cudarc::driver::CudaView<'_, u32>,
    stride: usize,
    offset: usize,
    rows: usize,
) {
    strided_rows(gpu, "unpack_rows", dst, src, ids, stride, offset, rows);
}

/// Both strided readouts take the same arguments and differ only in which side the
/// id table addresses, so the launch is written once.
fn strided_rows(
    gpu: &Gpu,
    kernel: &str,
    dst: &mut GTensor<f32>,
    src: &GTensor<f32>,
    ids: &cudarc::driver::CudaView<'_, u32>,
    stride: usize,
    offset: usize,
    rows: usize,
) {
    let dim = src.cols();
    assert_eq!(dim, dst.cols(), "{kernel}: width {dim} != {}", dst.cols());
    assert!(
        offset < stride,
        "{kernel}: offset {offset} outside stride {stride}"
    );
    let (stride_i, offset_i) = (stride as i32, offset as i32);
    let (dim_i, rows_i) = (dim as i32, rows as i32);
    let f = gpu.kernels.get(kernel);
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut dst.buf)
        .arg(&src.buf)
        .arg(ids)
        .arg(&stride_i)
        .arg(&offset_i)
        .arg(&dim_i)
        .arg(&rows_i);
    unsafe { lb.launch(elem_cfg(gpu, (rows * dim) as u32)) }.expect("strided rows");
}

/// Masked softmax cross-entropy (the hierarchical decode loss). `mask[r] == false`
/// marks a padding row: zero loss, zero grad. Normalized by the number of valid
/// rows. Returns `(mean_loss, dlogits)`.
pub fn masked_softmax_cross_entropy(
    gpu: &Gpu,
    logits: &GTensor<f32>,
    targets: &[usize],
    mask: &[bool],
) -> (f32, GTensor<f32>) {
    let num_valid = mask.iter().filter(|&&m| m).count().max(1) as f32;
    masked_softmax_cross_entropy_scaled(gpu, logits, targets, mask, 1.0 / num_valid)
}

/// Masked CE with an explicit `1/N` normalizer. When one window is split into
/// several rectangles (the length-grouped word batches), every group must be
/// scaled by the window's TOTAL valid-row count — not its own — so the summed
/// losses and gradients equal the single-rectangle result.
pub fn masked_softmax_cross_entropy_scaled(
    gpu: &Gpu,
    logits: &GTensor<f32>,
    targets: &[usize],
    mask: &[bool],
    inv: f32,
) -> (f32, GTensor<f32>) {
    let r = logits.rows();
    assert_eq!(targets.len(), r, "masked CE — targets len != rows");
    assert_eq!(mask.len(), r, "masked CE — mask len != rows");
    let dtargets = upload_ids(gpu, targets);
    let mask_u: Vec<u32> = mask.iter().map(|&m| m as u32).collect();
    let dmask = upload_ids_u32(gpu, &mask_u);
    masked_softmax_cross_entropy_u32(gpu, logits, &dtargets.slice(..), &dmask.slice(..), inv)
}

/// [`masked_softmax_cross_entropy_scaled`] against target/mask buffers already
/// resident on the device — the hot path, where the caller builds both as `u32`
/// once per group instead of narrowing a `usize`/`bool` pair per call.
pub fn masked_softmax_cross_entropy_u32(
    gpu: &Gpu,
    logits: &GTensor<f32>,
    dtargets: &CudaView<'_, u32>,
    dmask: &CudaView<'_, u32>,
    inv: f32,
) -> (f32, GTensor<f32>) {
    let (loss, dlogits) = masked_softmax_ce_u32_into(gpu, logits, dtargets, dmask, inv, None);
    (loss, dlogits)
}

/// [`masked_softmax_cross_entropy_u32`], optionally accumulating the group's loss into
/// a device scalar instead of reading it back.
///
/// Reading the row losses to the host is a **blocking** `memcpy_dtov`: it drains the
/// stream and page-locks a staging buffer, once per decoder group. That is a full
/// CPU/GPU sync in the middle of the decode loop, and the value it fetches is only ever
/// summed for logging. Passing `acc` keeps the reduction on the device and lets the
/// caller read the total once per step; the returned `f32` is then `0.0`.
pub fn masked_softmax_ce_u32_into(
    gpu: &Gpu,
    logits: &GTensor<f32>,
    dtargets: &CudaView<'_, u32>,
    dmask: &CudaView<'_, u32>,
    inv: f32,
    acc: Option<&mut CudaSlice<f32>>,
) -> (f32, GTensor<f32>) {
    let (r, c) = (logits.rows(), logits.cols());
    let mut dlogits = GTensor::uninit(gpu, &[r, c]);
    let mut row_loss = gpu.stream.alloc_zeros::<f32>(r).expect("alloc row_loss");
    let (c_i, r_i) = (c as i32, r as i32);
    let block = !no_block_scan();
    let f = gpu.kernels.get(if block {
        "masked_softmax_ce_block"
    } else {
        "masked_softmax_ce"
    });
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&logits.buf)
        .arg(dtargets)
        .arg(dmask)
        .arg(&mut dlogits.buf)
        .arg(&mut row_loss)
        .arg(&c_i)
        .arg(&inv)
        .arg(&r_i);
    let cfg = if block {
        // One block per row; the row is C wide, so size the block to it.
        LaunchConfig {
            grid_dim: (r as u32, 1, 1),
            block_dim: (scan_block_dim(c), 1, 1),
            shared_mem_bytes: 0,
        }
    } else {
        elem_cfg(gpu, r as u32)
    };
    unsafe { lb.launch(cfg) }.expect("masked_softmax_ce");
    match acc {
        Some(dst) => {
            sum_into(gpu, &row_loss, r, inv, dst);
            (0.0, dlogits)
        }
        None => {
            let losses = gpu.stream.clone_dtoh(&row_loss).expect("download row_loss");
            (losses.iter().sum::<f32>() * inv, dlogits)
        }
    }
}

/// Launch config for a kernel running one block per row, lanes walking the row.
/// `row_len` sets the block width so short rows do not launch idle warps; the
/// shared-memory reduction those kernels use assumes at most 32 warps.
fn row_block_cfg(rows: usize, row_len: usize) -> LaunchConfig {
    let block = row_len.next_power_of_two().clamp(32, 256);
    LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (block as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// `*dst += inv · Σ src[..n]`, entirely on the device.
fn sum_into(gpu: &Gpu, src: &CudaSlice<f32>, n: usize, inv: f32, dst: &mut CudaSlice<f32>) {
    let n_i = n as i32;
    let f = gpu.kernels.get("sum_accum");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(src).arg(dst).arg(&n_i).arg(&inv);
    // One block: the reduction is over at most a group's rows and the result is a
    // single scalar, so a second pass would cost more than the block costs.
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4,
    };
    unsafe { lb.launch(cfg) }.expect("sum_accum");
}

/// Grid for the o-gate pair: `x` over the `d` columns, `y` one block row per row of
/// the `[rows, d]` shape. The row index is then `blockIdx.y` and the kernels never
/// divide by the runtime `d`.
fn ogate_cfg(rows: usize, d: usize) -> LaunchConfig {
    const T: u32 = 256;
    LaunchConfig {
        grid_dim: ((d as u32).div_ceil(T), rows as u32, 1),
        block_dim: (T, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// o-gate backward into `d_yhat`, with `do_pre` written into `dxh`'s o columns.
///
/// `xh` is the fused `q‖k‖v‖o` forward output and `dxh` its gradient, so `do_pre`
/// lands exactly where `mlstm_fused_bw` puts dq/dk/dv — the two together fill the
/// buffer the merged projection's backward reads whole.
///
/// `d_yhat` is the caller's `[rows, d]` buffer, written in full.
pub fn ogate_bwd(
    gpu: &Gpu,
    d_hconcat: &GTensor<f32>,
    xh: &SlabBuf,
    yhat: &GTensor<f32>,
    dxh: &mut GTensor<f32>,
    d_yhat: &mut GTensor<f32>,
    o_off: usize,
) {
    let (rows, d, stride) = (d_hconcat.rows(), d_hconcat.cols(), dxh.cols());
    let (d_i, stride_i, off_i) = (d as i32, stride as i32, o_off as i32);
    debug_assert_eq!(d_yhat.dims(), d_hconcat.dims(), "ogate_bwd — d_yhat shape");
    let f = gpu.kernels.get("ogate_bwd");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&d_hconcat.buf);
    push_slab_ref!(lb, *xh);
    lb.arg(&yhat.buf)
        .arg(&mut dxh.buf)
        .arg(&mut d_yhat.buf)
        .arg(&d_i)
        .arg(&stride_i)
        .arg(&off_i);
    unsafe { lb.launch(ogate_cfg(rows, d)) }.expect("ogate_bwd");
}

/// Reduce `P` `[BH,T,T]` into `(dfc, dig)` `[BH,T]` (see kernel).
pub fn mlstm_dfc_dig(gpu: &Gpu, p: &GTensor<f32>) -> (GTensor<f32>, GTensor<f32>) {
    let (bh, t) = (p.shape[0], p.shape[1]);
    let mut dfc = GTensor::uninit(gpu, &[bh, t]);
    let mut dig = GTensor::uninit(gpu, &[bh, t]);
    let (ti, bht) = (t as i32, (bh * t) as i32);
    let f = gpu.kernels.get("mlstm_dfc_dig");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&p.buf)
        .arg(&mut dfc.buf)
        .arg(&mut dig.buf)
        .arg(&ti)
        .arg(&bht);
    unsafe { lb.launch(row_block_cfg(bh * t, t)) }.expect("mlstm_dfc_dig");
    (dfc, dig)
}

/// Reverse-cumsum + logσ' backward of `fc`: `df[BH,T]` from `dfc` and saved `f`.
pub fn revcumsum_dlogsig(gpu: &Gpu, dfc: &GTensor<f32>, f: &GTensor<f32>) -> GTensor<f32> {
    let (bh, t) = (dfc.rows(), dfc.cols());
    let mut df = GTensor::uninit(gpu, &[bh, t]);
    let (ti, bhi) = (t as i32, bh as i32);
    if !no_block_scan() {
        let func = gpu.kernels.get("revcumsum_dlogsig_block");
        let mut lb = gpu.stream.launch_builder(&func);
        lb.arg(&dfc.buf)
            .arg(&f.buf)
            .arg(&mut df.buf)
            .arg(&ti)
            .arg(&bhi);
        let cfg = LaunchConfig {
            grid_dim: (bh as u32, 1, 1),
            block_dim: (scan_block_dim(t), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { lb.launch(cfg) }.expect("revcumsum_dlogsig_block");
        return df;
    }
    let func = gpu.kernels.get("revcumsum_dlogsig");
    let mut lb = gpu.stream.launch_builder(&func);
    lb.arg(&dfc.buf)
        .arg(&f.buf)
        .arg(&mut df.buf)
        .arg(&ti)
        .arg(&bhi);
    unsafe { lb.launch(elem_cfg(gpu, bh as u32)) }.expect("revcumsum_dlogsig");
    df
}

/// Elementwise product `out = a ⊙ b` (fresh allocation).
pub fn mul(gpu: &Gpu, a: &GTensor<f32>, b: &GTensor<f32>) -> GTensor<f32> {
    let mut out = GTensor::uninit(gpu, a.dims());
    mul_into(gpu, a, b, &mut out);
    out
}

/// Elementwise `out = a * b` into a caller-owned buffer — the no-allocation form
/// of [`mul`]. `out` may alias either operand.
/// o-gate forward: `hconcat = σ(o) ⊙ yhat`, reading `o` out of the fused
/// `q‖k‖v‖o` projection output at `o_off`. The pre-activation is left as the GEMM
/// wrote it — [`ogate_bwd`] recomputes σ rather than reading it back.
pub fn ogate_fwd(
    gpu: &Gpu,
    xh: &SlabBuf,
    yhat: &GTensor<f32>,
    hconcat: &mut GTensor<f32>,
    o_off: usize,
) {
    assert_eq!(
        yhat.len(),
        hconcat.len(),
        "ogate_fwd: output length mismatch"
    );
    let (rows, d, stride) = (yhat.rows(), yhat.cols(), xh.dims()[1]);
    let (d_i, stride_i, off_i) = (d as i32, stride as i32, o_off as i32);
    let f = gpu.kernels.get("ogate_fwd");
    let mut lb = gpu.stream.launch_builder(&f);
    push_slab_ref!(lb, *xh);
    lb.arg(&yhat.buf)
        .arg(&mut hconcat.buf)
        .arg(&d_i)
        .arg(&stride_i)
        .arg(&off_i);
    unsafe { lb.launch(ogate_cfg(rows, d)) }.expect("ogate_fwd");
}

pub fn mul_into(gpu: &Gpu, a: &GTensor<f32>, b: &GTensor<f32>, out: &mut GTensor<f32>) {
    let n = a.len();
    assert_eq!(n, b.len(), "mul: length mismatch");
    assert_eq!(n, out.len(), "mul: output length mismatch");
    let n_i = n as i32;
    let f = gpu.kernels.get("mul");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut out.buf).arg(&a.buf).arg(&b.buf).arg(&n_i);
    unsafe { lb.launch(elem_cfg(gpu, n as u32)) }.expect("mul");
}

// Fused chunkwise mLSTM (TFLA). See the kernel block in `gpu/kernels.rs`.
//
// Four launches each way for a whole sequence, in mirror-image shapes: a per-chunk
// state product, an elementwise scan over the chunk recurrence, and a per-chunk
// block holding all the intra-chunk FLOPs. The [L, L] decay matrix stays in shared
// memory and never reaches HBM.

/// Output tiles one `mlstm_fw_dC` block covers: the `[dhv, dqk]` state, in mma
/// (16 x 8) units.
fn dc_tiles(dqk: usize, dhv: usize) -> usize {
    if let Some(w) = warps_pin("MLSTM_WARPS_DC") {
        return w;
    }
    (mma_pad(dhv, 16) / 16) * (mma_pad(dqk, 8) / 8)
}

/// Output tiles one FORWARD parallel block covers — the larger of the `[L, L]`
/// decay block and the `[L, dhv-slice]` output, since one width serves both loops.
fn fw_parallel_warps(l: usize, dhv: usize) -> usize {
    if let Some(w) = warps_pin("MLSTM_WARPS_FW") {
        return w;
    }
    let lp = mma_pad(l, 16);
    let vt = mma_pad(mlstm_vt().min(mma_pad(dhv, 8)), 8);
    (lp / 16) * (lp / 8).max(vt / 8)
}

/// Warps one BACKWARD parallel block wants.
///
/// Not its tile count, which is what sizes the forward. This kernel is
/// latency-bound, not tile-bound: it walks five slice loops that re-stage `[L, KT]`,
/// `[L, VT]` and `[VT, KT]` tiles between short bursts of mma, and measurement puts
/// the block width on the STAGING extent rather than on the dots. At the backbone
/// (`L = 32`, `dqk = dhv = 96`, KT = VT = 32) the tile rule asks for 8 warps and 16
/// is 1.36x faster — 49.7 us against 67.6 — even though half of them sit out every
/// mma pass. At the encoder and decoder (`dqk = dhv = 16`, so a quarter of the
/// staging) 16 warps is 2x SLOWER than 8, because there `bh` alone is thousands of
/// blocks and wide ones only cut how many stay resident.
///
/// One thread per element of the widest staged tile predicts both, where the tile
/// count predicts neither.
fn bw_parallel_warps(l: usize, dqk: usize, dhv: usize) -> usize {
    if let Some(w) = warps_pin("MLSTM_WARPS_BW") {
        return w;
    }
    let lp = mma_pad(l, 16);
    let kt = mma_pad(mlstm_kt().min(mma_pad(dqk, 16)), 16);
    let vt = mma_pad(mlstm_vt().min(mma_pad(dhv, 16)), 16);
    (lp * kt.max(vt)).div_ceil(32)
}

/// Threads per block for a parallel kernel, from the warps one block can keep busy.
///
/// These kernels walk their output tiles one WARP at a time, so for the forward —
/// which is tile-bound — warps past the tile count have nothing to do but hold
/// slots. The two cells in this model are two orders of magnitude apart here: the
/// backbone (`dqk = dhv = 96`, `L = 32`) has 72 tiles and wants every warp the cap
/// allows, while the encoder and decoder (`dqk = dhv = 16`, a word's length for `L`)
/// have TWO — and a 512-thread block ran 14 of its 16 warps empty, at a `bh` of
/// several thousand blocks. See [`bw_parallel_warps`] for why the backward counts
/// something else.
///
/// Floored at four warps so the staging loops still have threads to spread the tile
/// loads across, and capped at [`FUSED_THREADS_PAR`].
fn parallel_threads(warps: usize) -> u32 {
    const MIN_WARPS: usize = 4;
    let max_warps = fused_threads_par() as usize / 32;
    (warps.clamp(MIN_WARPS, max_warps) * 32) as u32
}

/// Warp-count pin for one parallel kernel, read from `var` once. The three warp
/// rules below are fitted per shape, so a new head width is swept through these
/// rather than guessed — see [`bw_parallel_warps`] for why the rules differ.
fn warps_pin(var: &'static str) -> Option<usize> {
    static PINS: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<&'static str, Option<usize>>>,
    > = std::sync::OnceLock::new();
    let map = PINS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut map = map.lock().unwrap();
    *map.entry(var).or_insert_with(|| {
        std::env::var(var)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n >= 1)
    })
}

/// [`FUSED_THREADS_PAR`], with `MLSTM_THREADS_PAR=<n>` honoured so the ceiling
/// itself can be swept. Rounded down to a power of two — `bw_parallel`'s `dg`
/// reduction halves the block width.
fn fused_threads_par() -> u32 {
    static PIN: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *PIN.get_or_init(|| {
        std::env::var("MLSTM_THREADS_PAR")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|&n| (32..=1024).contains(&n))
            .map(|n| 1 << n.ilog2())
            .unwrap_or(FUSED_THREADS_PAR)
    })
}

/// Widest block a parallel kernel launches with.
///
/// 512, not 256, because shared memory — not registers — is what caps these: the
/// footprint per block is fixed by the blocking, so a wider block doubles the
/// resident warps *for free* wherever there are tiles to give them. What decides
/// the actual width is [`parallel_threads`]; this is only the ceiling.
/// `gpu_occupancy` is how you check the register side is still clear after a change.
///
/// Must stay a power of two — the `dg` reduction in `bw_parallel` halves it.
pub const FUSED_THREADS_PAR: u32 = 512;

/// Threads per block for `mlstm_fw_gates`. A block owns one `(b, h)` and gives one
/// WARP to each chunk, so this caps how many chunks it works on at a time; the
/// serial scan that follows them runs on a single thread either way.
pub const GATES_THREADS: u32 = 256;

/// Threads per block for `mlstm_state_scan`. Pure streaming over state elements —
/// the block width only sets how the grid is cut.
pub const SCAN_THREADS: u32 = 256;

/// Longest chunk the fused kernels support.
///
/// A WARP is what pins this, not shared memory. `mlstm_bw_parallel`'s tail is two
/// reductions over the chunk's timesteps — Σ_j da·a and the reverse cumsum of dfc —
/// and both run on one warp with shuffles, which is only a whole chunk while
/// `L <= 32`. Shared memory is no longer what binds — the backward is 43.8 KB at
/// L=32, against the 99 KB a block can opt into — but it would not let L=64 through
/// either (102.2 KB), and the ceiling would land in the wrong place anyway: the
/// intra-chunk work is O(L) per position while the state traffic it saves is O(1/L),
/// and past 32 the causal mask throws away more than the traffic is worth. Raising
/// it means tiling the `L` axis, as the reference does (`siz_b_LQ = 32` at chunk
/// 64), not just widening the buffers.
///
/// Chunk length is a pure blocking choice (`mlstm_chunking_matches_single_chunk`
/// pins every L to the same numbers), so capping it costs accuracy nothing.
pub const FUSED_MAX_L: usize = 32;

/// Head-dim slice width the two parallel kernels walk the `dqk` axis in.
///
/// Must match `MLSTM_KT` in `mlstm_fused.cu` — the host sizes the dynamic shared
/// memory the kernel then indexes, so a disagreement is an out-of-bounds walk rather
/// than a compile error. A fixed cap, not a divisor of any head dim: widening `dqk`
/// raises the kernel's trip count and leaves the footprint alone. A multiple of 16,
/// the bf16 mma's contraction depth.
///
/// One width for every head dim. A wider slice at `dqk >= 128` used to measure
/// better, but that was fitted before the backward moved to bf16 mma and became
/// latency-bound; with the current kernels 32 wins at every width measured
/// (B=1 T=1024 dqk=128 0.895 vs 0.999 ms, T=4096 3.55 vs 3.89, T=512 dqk=192
/// 0.981 vs 1.036, dqk=256 1.554 vs 1.691).
pub const MLSTM_KT: usize = 32;

/// [`MLSTM_VT`], with `MLSTM_VT=<n>` honoured as a pin for a sweep. Rounded to the
/// mma contraction pad so the value the host sizes for is the value the kernel tiles
/// to: `dhv` is an output index in the forward but a CONTRACTION in the backward's
/// phases 4 and 5, so the slice has to be a multiple of the bf16 mma's K = 16.
pub fn mlstm_vt() -> usize {
    static PIN: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let pinned = *PIN.get_or_init(|| {
        std::env::var("MLSTM_VT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n >= 16)
    });
    // 32 everywhere by default. Swept against `dqk` at the shape the backbone runs
    // (B=1, T=BACKBONE_CHUNK): at dhv=128 the four widths 16/32/64/96 measure
    // 0.575/0.550/0.564/0.599 ms, so the ladder the dqk axis wanted has no twin here.
    mma_pad(pinned.unwrap_or(32), 16)
}

/// `dhv` slice width, the `dhv` twin of [`MLSTM_KT`]. Slices OUTPUT columns rather
/// than a contraction, so no slice needs accumulating across iterations.
pub const MLSTM_VT: usize = 64;

/// Widest head dim the fused kernels are efficient at.
///
/// Not a capability limit — both head dims are tiled, so they FIT at any width (flat
/// 43.8 KB at dqk <= 96, 64 KB above it). It is where they stop being the fast choice: a block walks `NKT·NVT`
/// slice passes, so its work grows quadratically in the head dim. Measured at B=1
/// T=4096 H=8 against the op-at-a-time path that used to serve as the fallback
/// (median ms/iter, fused vs that path): dqk=128 5.94/6.24, dqk=192 13.7/10.0,
/// dqk=256 17.6/13.7, dqk=512 71/40.
///
/// **Nothing enforces this any more.** The op-at-a-time path was removed once every
/// shape the model builds (dqk 96/24/16) sat below this line, so a wider head runs
/// the fused kernels regardless and pays the ratio above. Every current config is
/// well inside it; a future `WORD_HIDDEN >= 1536` (dqk 192 at 8 heads) is where it
/// starts to matter, and that is the point to revisit the L-axis blocking rather
/// than reinstate a second code path.
pub const FUSED_MAX_HEAD: usize = 128;

pub fn mlstm_kt() -> usize {
    static PIN: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let pinned = *PIN.get_or_init(|| {
        std::env::var("MLSTM_KT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n >= 8)
    });
    // The forward's parallel kernel contracts 16 at a time (the bf16 mma's K), so
    // every slice width is a multiple of 16; the backward's contracts 8 and is happy
    // with any multiple of 16 too.
    mma_pad(pinned.unwrap_or(MLSTM_KT), 16)
}

/// What forward hands to backward. Everything here is `[BH, …]` head-major.
///
/// Deliberately absent: `D̄` and `D̄⊙S`. The op-at-a-time path saved both — two
/// `[BH, L, L]` slabs per chunk — and backward re-read them from HBM; the fused
/// backward recomputes them in shared memory from `(Q, K, fc, ig, m)` instead.
pub struct MlstmFused {
    pub l: usize,
    pub nc: usize,
    pub ytil: SlabBuf,  // [BH, T, dhv]  bf16 storage (reference: matHout at DTYPE)
    cst: GTensor<f32>,  // [BH, NC+1, dhv, dqk]  state entering each chunk
    nst: GTensor<f32>,  // [BH, NC+1, dqk]
    mst: GTensor<f32>,  // [BH, NC+1]
    fcb: GTensor<f32>,  // [BH, NC, L]  chunk-local cumulative log-forget
    gvec: GTensor<f32>, // [BH, NC]     per-chunk state decay, for both scan directions
    msv: GTensor<f32>,  // [BH, T]      per-row stabilizer
    psiv: GTensor<f32>, // [BH, T]
    qnv: GTensor<f32>,  // [BH, T]
}

impl MlstmFused {
    /// The state leaving the last chunk — index `NC` of each array, which the kernel
    /// publishes as the final state. Feed to the next chunk's `carry_in`.
    pub fn final_state(&self, gpu: &Gpu, bh: usize, dhv: usize, dqk: usize) -> MlstmState {
        let slots = self.nc + 1;
        // Slot NC of each bh — one launch, not a `bh`-long loop of tiny copies.
        let grab = |src: &GTensor<f32>, stride: usize| {
            let mut out = GTensor::uninit(gpu, &[bh * stride]);
            state_slot_copy(gpu, src, &mut out, bh, slots, self.nc, stride, true);
            out
        };
        MlstmState {
            c: grab(&self.cst, dhv * dqk),
            n: grab(&self.nst, dqk),
            m: grab(&self.mst, 1),
        }
    }

    /// Device bytes this fused cache holds. Diagnostic.
    ///
    /// `cst` dominates: `[BH, NC+1, dhv, dqk]` is the per-chunk state, quadratic in
    /// the head dims where everything else is linear in `T`.
    pub fn retained_bytes(&self) -> usize {
        let f32s: usize = [
            &self.cst, &self.nst, &self.mst, &self.fcb, &self.gvec, &self.msv, &self.psiv,
            &self.qnv,
        ]
        .iter()
        .map(|t| t.capacity() * 4)
        .sum();
        f32s + self.ytil.retained_bytes()
    }
}

/// Shared-memory floats each kernel needs, given the blocking. Kept next to the
/// kernels' `extern __shared__` layouts — the two must agree exactly.
///
/// The `+ 1`s are the bank-conflict pad: every 2D shared array is stored with its
/// row stride one float longer than its row, so that warps walking `row` at fixed
/// `col` spread across banks instead of piling onto one. Tiles that feed an mma are
/// bf16 and pad by `bf16_ld` instead, which spreads the 32-bit PAIR loads a fragment
/// makes.
fn fused_smem(kind: &str, l: usize, dqk: usize, dhv: usize) -> usize {
    match kind {
        // The two ΔC kernels stage the chunk's two operands, both padded up to the
        // mma tile, and nothing else: no `[L, L]` extent and no state tile, since the
        // output goes straight from the accumulators to HBM. The forward folds `a`
        // into V and pairs it with K, the backward folds `b` into d_num and pairs it
        // with Q; the backward's two per-row scalars are the only difference.
        "fw_dC" | "bw_dC" => {
            let (lp, vp, kp) = (mma_pad(l, 16), mma_pad(dhv, 16), mma_pad(dqk, 8));
            let rows = if kind == "fw_dC" { l } else { 2 * l };
            // The two tiles are bf16, so they count half a float per element.
            rows + (lp * bf16_ld(vp) + lp * bf16_ld(kp)).div_ceil(2)
        }
        // The parallel kernels pad every dim up to the tensor-core tile (rows to the
        // mma M=16, the contraction to K=8, the columns to N=8) and zero-fill the
        // pad, which is what lets one code path serve a short last chunk and an odd
        // dqk/dhv alike. Must match the `LP`/`KP`/`VP` the kernel recomputes.
        // Both head dims are sliced (`MLSTM_KT` / `MLSTM_VT`), so every buffer
        // spanning one is sized to the slice and the footprint is flat in dqk/dhv.
        // `sAcc` stages the output tile between the two dots that write it, which
        // no longer share a warp once the dqk contraction has its own slice loop.
        "fw_parallel" => {
            let (lp, kp, vp) = (mma_pad(l, 16), mma_pad(dqk, 16), mma_pad(dhv, 8));
            let kt = mma_pad(mlstm_kt().min(kp), 16);
            let vt = mma_pad(mlstm_vt().min(vp), 8);
            let (ls, la) = (lp + 1, vt + 1);
            let (lq, lv, ld) = (bf16_ld(kt), bf16_ld(vt), bf16_ld(lp));
            // fp32: the decay matrix, the per-row scalars and the output tile.
            let f32s = lp * ls + kt + 6 * lp + lp * la;
            // bf16: every mma operand, at half a float per element.
            let bf16s = 2 * lp * lq + lp * lv + vt * lq + lp * ld;
            f32s + bf16s.div_ceil(2)
        }
        // The dqk axis is sliced `MLSTM_KT` wide and the dhv axis `MLSTM_VT` wide
        // (see `mlstm_bw_parallel`), so every buffer spanning one is sized to the
        // slice, not the head dim, and the footprint is flat in dqk/dhv. Unlike the
        // forward, `dhv` is a CONTRACTION here (phases 4 and 5), so it pads to the
        // mma's K = 16 rather than its N = 8.
        "bw_parallel" => {
            let (lp, kp, vp) = (mma_pad(l, 16), mma_pad(dqk, 16), mma_pad(dhv, 16));
            let kt = mma_pad(mlstm_kt().min(kp), 16);
            let vt = mma_pad(mlstm_vt().min(vp), 16);
            let (ls, lv, lk) = (lp + 1, vt + 1, kt + 1);
            let (lqb, lvb, ldb) = (bf16_ld(kt), bf16_ld(vt), bf16_ld(lp));
            // fp32: the two [L, L] extents (DS/dS and phase 4's dhv contraction), the
            // four cross-slice accumulators, and the per-row scalars.
            let f32s = 2 * lp * ls + lp * lv + 3 * lp * lk + 2 * kt + 10 * lp;
            // bf16: every mma operand, at half a float per element.
            let bf16s = 2 * lp * lqb + 2 * lp * lvb + 2 * vt * lqb + lp * ldb;
            f32s + bf16s.div_ceil(2)
        }
        _ => unreachable!(),
    }
}

/// Round `n` up to a multiple of `to` (a power of two): the mma tile padding.
fn mma_pad(n: usize, to: usize) -> usize {
    n.div_ceil(to) * to
}

/// Row stride of a bf16 shared tile, in elements. Must match `BF16_LD` in
/// `mlstm_fused.cu` — the host sizes the dynamic shared memory the kernel indexes.
fn bf16_ld(w: usize) -> usize {
    mma_pad(w, 16) + 8
}

/// Fetch a fused kernel and opt it into the shared memory it needs.
///
/// A block gets 48 KB of shared memory by default; past that the driver requires
/// an explicit per-function opt-in. Nothing needs one at the shapes this model
/// builds — the backward's parallel kernel is 43.8 KB at the backbone's width, so
/// two of its blocks would fit on an SM — but the opt-in stays for the wider heads where
/// the trade the design makes still holds: one big block that keeps the decay
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

/// [`fused_kernel`], but preferring a build with the shape baked in as compile-time
/// constants — the reference's `tl.constexpr` shape parameters.
///
/// All four heavy kernels go through here. The parallel pair folds its padded tile
/// dims; the recurrent pair additionally folds `tv` and the block width, which are
/// the bound and the stride of every loop they run. A shape that fails to compile
/// falls back to the generic kernel, which is always present and always correct.
fn fused_kernel_spec(
    gpu: &Gpu,
    name: &'static str,
    smem_floats: usize,
    spec: super::kernels::MlstmSpec,
) -> (cudarc::driver::CudaFunction, u32) {
    let bytes = (smem_floats * std::mem::size_of::<f32>()) as u32;
    match gpu.kernels.mlstm_specialized(&gpu.context, name, spec) {
        Some(f) => {
            if bytes > 48 * 1024 {
                f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    bytes as i32,
                )
                .unwrap_or_else(|e| {
                    panic!("{name}: cannot opt into {bytes} B of shared memory: {e:?}")
                });
            }
            (f, bytes)
        }
        None => fused_kernel(gpu, name, smem_floats),
    }
}

fn fused_cfg(grid: (u32, u32, u32), threads: u32, smem: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: grid,
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem,
    }
}

/// Threads a fused kernel launches with. The four heavy ones take their width from
/// the mma tiles a block covers (see [`parallel_threads`]); the two housekeeping
/// kernels have a fixed one.
pub fn fused_threads(name: &str, l: usize, dqk: usize, dhv: usize) -> u32 {
    match name {
        "mlstm_fw_gates" => GATES_THREADS,
        "mlstm_state_scan" => SCAN_THREADS,
        "mlstm_fw_dC" | "mlstm_bw_dC" => parallel_threads(dc_tiles(dqk, dhv)),
        "mlstm_bw_parallel" => parallel_threads(bw_parallel_warps(l, dqk, dhv)),
        _ => parallel_threads(fw_parallel_warps(l, dhv)),
    }
}

/// Whether the fused kernels can run this shape. Diagnostic — there is no longer a
/// second path to fall back to, so a shape outside this is a bug, not a dispatch.
pub fn mlstm_fused_supported(l: usize, dqk: usize, dhv: usize) -> bool {
    l >= 1 && l <= FUSED_MAX_L && dqk >= 1 && dhv >= 1
}

/// Peak per-block shared memory (bytes) any fused kernel needs at this blocking.
///
/// The kernels keep their whole `[L, L]` decay matrix — and, in the backward, two
/// `[dhv, dqk]` state-staging tiles — resident in shared memory, so at large head
/// dims this used to cross what a 100 KB-shared card can opt a single block into.
/// Both head dims are tiled and every mma operand is bf16, so it is flat in
/// `dqk`/`dhv` (21.6 KB forward, 43.8 KB backward at the backbone's width) and
/// `MLstm::fused_chunk` only debug-asserts it.
pub fn mlstm_fused_smem_bytes(l: usize, dqk: usize, dhv: usize) -> usize {
    ["fw_dC", "fw_parallel", "bw_dC", "bw_parallel"]
        .iter()
        .map(|k| fused_smem(k, l, dqk, dhv) * std::mem::size_of::<f32>())
        .max()
        .unwrap()
}

/// The forward and backward parallel kernels' shared memory, separately. Both are
/// flat in the head dims once tiled, so a width that regresses points at exactly one
/// of them. Diagnostic; the gate uses [`mlstm_fused_smem_bytes`].
pub fn mlstm_fused_smem_parts(l: usize, dqk: usize, dhv: usize) -> (usize, usize) {
    let f = |k: &str| fused_smem(k, l, dqk, dhv) * std::mem::size_of::<f32>();
    (f("fw_parallel"), f("bw_parallel"))
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
    let bh = bh as u32;
    // Every one of them is a block per (chunk, bh) now — the backward's ΔC kernel
    // included, which is the whole point of the decomposition.
    ["fw_dC", "fw_parallel", "bw_dC", "bw_parallel"]
        .into_iter()
        .map(|kind| {
            let name: &'static str = match kind {
                "fw_dC" => "mlstm_fw_dC",
                "fw_parallel" => "mlstm_fw_parallel",
                "bw_dC" => "mlstm_bw_dC",
                _ => "mlstm_bw_parallel",
            };
            let (f, smem) = fused_kernel(gpu, name, fused_smem(kind, l, dqk, dhv));
            (name, f, smem, (nc, bh))
        })
        .collect()
}

/// Chunkwise forward: `mlstm_fw_gates` → `mlstm_fw_dC` → `mlstm_state_scan` →
/// `mlstm_fw_parallel`.
#[allow(clippy::too_many_arguments)]
/// The recurrent state entering a chunked mLSTM call: `C`, `n`, `m` for every
/// `(batch, head)`, laid out exactly as one slice of `MlstmFused`'s `cst`/`nst`/`mst`.
///
/// Produced by [`MlstmFused::final_state`] at the end of one chunk and handed to the
/// next call's `carry_in`. Zero-initialised state is represented by `None`, not by a
/// zero-filled `MlstmState` — that keeps the common (unchunked) path free of the copy.
pub struct MlstmState {
    pub c: GTensor<f32>, // [BH, dhv, dqk]
    pub n: GTensor<f32>, // [BH, dqk]
    pub m: GTensor<f32>, // [BH]
}

/// The BPTT state crossing a chunk border in the backward direction: the gradient wrt
/// the recurrent state entering the chunk to the right.
///
/// Backward unwinds chunks right to left, so chunk `c` produces this for chunk `c-1`.
pub struct MlstmDState {
    pub dc: GTensor<f32>, // [BH, dhv, dqk]
    pub dn: GTensor<f32>, // [BH, dqk]
}

/// Walk the chunk recurrence over a `[BH, NC+1, ...]` state pair, in place.
///
/// `rev` picks the direction: forward folds each chunk's ΔC into the state to its
/// left (`C_{k+1} = g_k·C_k + ΔC_k`), backward folds each chunk's ΔdC into the
/// gradient to its RIGHT (`dC_k = g_k·dC_{k+1} + ΔdC_k`). Same recurrence, same
/// decay, so one kernel serves both — see `mlstm_state_scan`.
#[allow(clippy::too_many_arguments)]
fn state_scan(
    gpu: &Gpu,
    cst: &mut GTensor<f32>,
    nst: &mut GTensor<f32>,
    gvec: &GTensor<f32>,
    bh: usize,
    nc: usize,
    dqk: usize,
    dhv: usize,
    rev: bool,
) {
    let elems = dhv * dqk + dqk;
    let (nc_i, dqk_i, dhv_i, rev_i) = (nc as i32, dqk as i32, dhv as i32, i32::from(rev));
    let f = gpu.kernels.get("mlstm_state_scan");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut cst.buf)
        .arg(&mut nst.buf)
        .arg(&gvec.buf)
        .arg(&nc_i)
        .arg(&dqk_i)
        .arg(&dhv_i)
        .arg(&rev_i);
    unsafe {
        lb.launch(fused_cfg(
            (elems.div_ceil(SCAN_THREADS as usize) as u32, bh as u32, 1),
            SCAN_THREADS,
            0,
        ))
    }
    .expect("mlstm_state_scan");
}

/// One launch for the whole `[BH, stride]` <-> `[BH, slots, stride]` slot copy.
///
/// `gather` reads slot `idx` into a flat tensor; otherwise it writes the flat tensor
/// into slot `idx`. See `state_slot_copy` in `common.cu` for why this is a kernel and
/// not a `bh`-long loop of `memcpy_dtod`.
fn state_slot_copy(
    gpu: &Gpu,
    src: &GTensor<f32>,
    dst: &mut GTensor<f32>,
    bh: usize,
    slots: usize,
    idx: usize,
    stride: usize,
    gather: bool,
) {
    let n = bh * stride;
    let f = gpu.kernels.get("state_slot_copy");
    let (bh_i, slots_i, idx_i, stride_i, g_i) = (
        bh as i32,
        slots as i32,
        idx as i32,
        stride as i32,
        i32::from(gather),
    );
    let mut b = gpu.stream.launch_builder(&f);
    b.arg(&src.buf)
        .arg(&mut dst.buf)
        .arg(&bh_i)
        .arg(&slots_i)
        .arg(&idx_i)
        .arg(&stride_i)
        .arg(&g_i);
    unsafe { b.launch(elem_cfg(gpu, n as u32)) }.expect("state_slot_copy");
}

/// Seed slot `idx` of a `[BH, slots, ...]` array from a `[BH, ...]` state.
fn seed_state_slot_n(
    gpu: &Gpu,
    src: &GTensor<f32>,
    dst: &mut GTensor<f32>,
    bh: usize,
    slots: usize,
    idx: usize,
    stride: usize,
) {
    debug_assert_eq!(src.len(), bh * stride, "carry state: wrong source size");
    state_slot_copy(gpu, src, dst, bh, slots, idx, stride, false);
}

/// Read slot `idx` of a `[BH, slots, ...]` array into a fresh `[BH, ...]` tensor.
fn read_state_slot_n(
    gpu: &Gpu,
    src: &GTensor<f32>,
    bh: usize,
    slots: usize,
    idx: usize,
    stride: usize,
    into: Option<GTensor<f32>>,
) -> GTensor<f32> {
    // `into` is the previous chunk's buffer handed back: this runs once per chunk per
    // layer, so allocating a fresh pair each time is one allocation per block per chunk
    // on the hot path for a value whose shape never changes within a sweep.
    let mut out = match into {
        Some(t) if t.len() == bh * stride => t,
        _ => GTensor::uninit(gpu, &[bh * stride]),
    };
    state_slot_copy(gpu, src, &mut out, bh, slots, idx, stride, true);
    out
}

/// Seed slot 0 of a `[BH, NC+1, ...]` chunk-state array from a `[BH, ...]` state.
///
/// The destination holds `slots` states per `bh`, each `stride` elements; the source
/// holds one per `bh`. So `bh`'s slot 0 lives at `bh * slots * stride`, and this walks
/// `bh` copying `stride` elements into each.
fn seed_state_slot0(
    gpu: &Gpu,
    src: &GTensor<f32>,
    dst: &mut GTensor<f32>,
    bh: usize,
    slots: usize,
    stride: usize,
) {
    debug_assert_eq!(src.len(), bh * stride, "carry state: wrong source size");
    state_slot_copy(gpu, src, dst, bh, slots, 0, stride, false);
}

/// The logical shape of one fused mLSTM call.
///
/// It is all the kernels need to address their inputs: q/k/v and the gate logits are
/// position-major `[B*T, H*W]` — the layout the projections write — so element
/// `(b, h, t, c)` sits at `((b*T + t)*H + h)*W + c`, with `W` the group's width. The
/// reference (`mlstm_kernels`) passes explicit strides because PyTorch hands it
/// whatever the caller had; here one producer feeds one consumer, so the kernels
/// derive them (see `MLSTM_STRIDES` in `mlstm_fused.cu`).
#[derive(Clone, Copy)]
pub struct MlstmShape {
    pub batch: usize,
    pub heads: usize,
    pub t: usize,
    pub dqk: usize,
    pub dhv: usize,
}

impl MlstmShape {
    /// `B*H` — the outer dimension every fused kernel's grid is keyed on.
    pub fn bh(&self) -> usize {
        self.batch * self.heads
    }
}

pub fn mlstm_fused_fw(
    gpu: &Gpu,
    // q ‖ k ‖ v, `[B*T, H*(2*dqk + dhv)]`, bf16 storage (reference: matQ/matK/matV at
    // DTYPE, concatenated as its fused `qkv_opreact`). k carries the 1/√dqk already.
    xh: &SlabBuf,
    // ĩ ‖ f̃, `[B*T, 2*H]`, fp32: gate logits (the reference pins vecI/vecB to fp32
    // too) from one `ifgate_preact` projection.
    gates: &GTensor<f32>,
    l: usize,
    // State this call continues from, or `None` to start the recurrence at zero.
    carry_in: Option<&MlstmState>,
    st: MlstmShape,
) -> MlstmFused {
    let (bh, t, dqk, dhv) = (st.batch * st.heads, st.t, st.dqk, st.dhv);
    let l = l.min(t);
    assert!(
        mlstm_fused_supported(l, dqk, dhv),
        "fused mLSTM: unsupported shape"
    );
    let nc = t.div_ceil(l);
    let h_i = st.heads as i32;
    let (t_i, l_i, nc_i, dqk_i, dhv_i) = (t as i32, l as i32, nc as i32, dqk as i32, dhv as i32);

    let mut fcb = GTensor::uninit(gpu, &[bh, nc, l]);
    let mut cst = GTensor::uninit(gpu, &[bh, nc + 1, dhv, dqk]);
    let mut nst = GTensor::uninit(gpu, &[bh, nc + 1, dqk]);
    let mut mst = GTensor::uninit(gpu, &[bh, nc + 1]);
    // Carrying: stage the incoming state into slot 0 of each array, which is where
    // `mlstm_state_scan` (and `mlstm_fw_gates`, for `m`) picks it up under `CARRY`.
    // Copying rather than aliasing keeps the "state entering chunk k lives at index k"
    // invariant intact for backward.
    if let Some(st) = carry_in {
        seed_state_slot0(gpu, &st.c, &mut cst, bh, nc + 1, dhv * dqk);
        seed_state_slot0(gpu, &st.n, &mut nst, bh, nc + 1, dqk);
        seed_state_slot0(gpu, &st.m, &mut mst, bh, nc + 1, 1);
    }
    let carry_i = carry_in.is_some() as i32;
    // ytil is the kernel's output h (reference stores matHout at DTYPE); the
    // stabilizer/normalizer triple stays fp32, as does the chunk state. Written
    // position-major like q/k/v, so it reaches the head norm without a scatter.
    let mut ytil = SlabBuf::new(gpu, &[st.batch * t, st.heads * dhv]);
    let mut msv = GTensor::uninit(gpu, &[bh, t]);
    let mut psiv = GTensor::uninit(gpu, &[bh, t]);
    let mut qnv = GTensor::uninit(gpu, &[bh, t]);

    // Gates: `fcb` plus the per-chunk `a`/`g` and the stabilizer scan. One block per
    // (b, h) — the scan over chunks is serial, and it is the only thing left in the
    // forward that is. `avec` is consumed by the next launch and by nothing else;
    // `gvec` is the decay both scan directions need, so it joins the saved cache
    // rather than being recomputed from `fcb`/`mst`/`msv` in backward.
    let mut avec = GTensor::<f32>::uninit(gpu, &[bh, nc, l]);
    let mut gvec = GTensor::uninit(gpu, &[bh, nc]);
    let f = gpu.kernels.get("mlstm_fw_gates");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&gates.buf)
        .arg(&mut fcb.buf)
        .arg(&mut avec.buf)
        .arg(&mut gvec.buf)
        .arg(&mut mst.buf)
        .arg(&t_i)
        .arg(&l_i)
        .arg(&nc_i)
        .arg(&carry_i)
        .arg(&h_i);
    // One warp per chunk, so a block wider than `nc` warps is idle warps — and at the
    // encoder/decoder's word-length sequences `nc` is 1.
    let gates_threads = (nc.clamp(1, GATES_THREADS as usize / 32) * 32) as u32;
    unsafe {
        lb.launch(fused_cfg(
            (bh as u32, 1, 1),
            gates_threads,
            (2 * nc * std::mem::size_of::<f32>()) as u32,
        ))
    }
    .expect("mlstm_fw_gates");

    // Every chunk's own contribution to the state, all at once.
    let dc_threads = parallel_threads(dc_tiles(dqk, dhv));
    let (f, smem) = fused_kernel_spec(
        gpu,
        "mlstm_fw_dC",
        fused_smem("fw_dC", l, dqk, dhv),
        super::kernels::MlstmSpec {
            l,
            dqk,
            dhv,
            h: st.heads,
            threads: dc_threads,
            kt: 0,
            vt: 0,
        },
    );
    let mut lb = gpu.stream.launch_builder(&f);
    push_slab_ref!(lb, *xh);
    lb.arg(&avec.buf)
        .arg(&mut cst.buf)
        .arg(&mut nst.buf)
        .arg(&t_i)
        .arg(&l_i)
        .arg(&nc_i)
        .arg(&dqk_i)
        .arg(&dhv_i)
        .arg(&carry_i)
        .arg(&h_i);
    unsafe { lb.launch(fused_cfg((nc as u32, bh as u32, 1), dc_threads, smem)) }
        .expect("mlstm_fw_dC");

    // ... folded into the running state, one thread per state element. A single
    // chunk with nothing carried in has nothing to fold — `mlstm_fw_dC` already wrote
    // both slots — and the encoder and decoder are always in that case, so the launch
    // is skipped rather than run as an expensive copy.
    if nc > 1 || carry_in.is_some() {
        state_scan(gpu, &mut cst, &mut nst, &gvec, bh, nc, dqk, dhv, false);
    }

    let par_threads = parallel_threads(fw_parallel_warps(l, dhv));
    let (f, smem) = fused_kernel_spec(
        gpu,
        "mlstm_fw_parallel",
        fused_smem("fw_parallel", l, dqk, dhv),
        super::kernels::MlstmSpec {
            l,
            dqk,
            dhv,
            h: st.heads,
            threads: par_threads,
            kt: mlstm_kt(),
            vt: mlstm_vt(),
        },
    );
    let mut lb = gpu.stream.launch_builder(&f);
    push_slab_ref!(lb, *xh);
    lb.arg(&gates.buf)
        .arg(&fcb.buf)
        .arg(&cst.buf)
        .arg(&nst.buf)
        .arg(&mst.buf);
    push_slab!(lb, ytil);
    lb.arg(&mut msv.buf)
        .arg(&mut psiv.buf)
        .arg(&mut qnv.buf)
        .arg(&t_i)
        .arg(&l_i)
        .arg(&nc_i)
        .arg(&dqk_i)
        .arg(&dhv_i)
        .arg(&carry_i)
        .arg(&h_i);
    unsafe { lb.launch(fused_cfg((nc as u32, bh as u32, 1), par_threads, smem)) }
        .expect("mlstm_fw_parallel");

    MlstmFused {
        l,
        nc,
        ytil,
        cst,
        nst,
        mst,
        fcb,
        gvec,
        msv,
        psiv,
        qnv,
    }
}

/// Chunkwise backward: `mlstm_bw_dqn` → `mlstm_bw_dC` → `mlstm_state_scan` (REV) →
/// `mlstm_bw_parallel` — the mirror of [`mlstm_fused_fw`], launch for launch.
///
/// Writes dq/dk/dv into `dxh`'s own column blocks — `ogate_bwd` fills the fourth —
/// and `dgates` (the gate grads, laid out like the gate logits), and returns the BPTT
/// state to hand to the chunk on the left (see [`MlstmDState`]).
///
/// `dxh` and `dgates` are both the caller's. `dxh` has to be, because the o-gate half
/// of it is produced before this runs; between the two, every column is written, so it
/// may come in uninitialised.
#[allow(clippy::too_many_arguments)]
pub fn mlstm_fused_bw(
    gpu: &Gpu,
    sv: &MlstmFused,
    xh: &SlabBuf,
    gates: &GTensor<f32>,
    dxh: &mut GTensor<f32>,
    // The incoming gradient is fp32: it is a transient, not a saved tensor, so
    // narrowing it would buy no memory. (The reference does cast matDeltaH to
    // DTYPE, but only to feed its tensor cores, which is where the kernels narrow
    // it too — on the way into shared memory.)
    d_ytil: &GTensor<f32>, // [BH, T, dhv]
    // BPTT state flowing in from the chunk to the RIGHT, or `None` for the rightmost
    // chunk (and for an unchunked call), where it is zero.
    //
    // Taken by value, and handed back as the return: it is seeded into `dcst`/`dnst`
    // below and dead from that point, so its two buffers are rewritten with this
    // chunk's outgoing state instead of a fresh pair being allocated. That pair is
    // otherwise one allocation per block per chunk, on the hot path, for a shape that
    // never changes within a sweep.
    carry_in: Option<MlstmDState>,
    // `[N, 2·heads]`, written in full by `mlstm_bw_parallel`.
    dgates: &mut GTensor<f32>,
    // Scratch this call needs and the caller never sees — the per-timestep dψ.
    // Borrowed rather than allocated, so it is bounded by the slot count like every
    // other temporary; see [`super::temp`].
    cache: &super::temp::TempCache,
    st: MlstmShape,
) -> MlstmDState {
    let (bh, t, dqk, dhv) = (st.batch * st.heads, st.t, st.dqk, st.dhv);
    let (l, nc) = (sv.l, sv.nc);
    let h_i = st.heads as i32;
    let (t_i, l_i, nc_i, dqk_i, dhv_i) = (t as i32, l as i32, nc as i32, dqk as i32, dhv as i32);

    // `uninit`, not zeroed: `mlstm_bw_dC` writes slots 0..NC outright — its own ΔdC
    // into slot k, and zeros into slot NC (the gradient flowing in from the right)
    // when nothing is carried in. At the backbone's shape that is a 38 MB memset
    // saved, and it is what lets `mlstm_bw_parallel` read slot k+1 unconditionally
    // instead of branching on being the rightmost chunk.
    // Their own array, not stage-sized slots: one state per chunk *boundary*, so these
    // grow as the chunk length SHRINKS, where every other temporary is bounded by
    // `rows x width`. See `temp::widest_chunk`.
    let mut dcst = cache.get_chunk::<f32>(gpu, &[bh, nc + 1, dhv, dqk]);
    let mut dnst = cache.get_chunk::<f32>(gpu, &[bh, nc + 1, dqk]);
    // Seed slot NC with the gradient from the chunk to the right; `mlstm_bw_dC` reads
    // CARRY and leaves it alone instead of zeroing it.
    let carry_i = carry_in.is_some() as i32;
    if let Some(dst) = &carry_in {
        seed_state_slot_n(gpu, &dst.dc, &mut dcst, bh, nc + 1, nc, dhv * dqk);
        seed_state_slot_n(gpu, &dst.dn, &mut dnst, bh, nc + 1, nc, dqk);
    }

    // dψ per timestep. A reduction over the whole `dhv` row, and both kernels below
    // want it, so it is computed once rather than twice.
    let mut dqnv = cache.get_small::<f32>(gpu, &[bh, t]);
    {
        let n = bh * t;
        let (n_i, dhv_only) = (n as i32, dhv as i32);
        let f = gpu.kernels.get("mlstm_bw_dqn");
        let mut lb = gpu.stream.launch_builder(&f);
        lb.arg(&d_ytil.buf);
        push_slab_ref!(lb, sv.ytil);
        lb.arg(&sv.psiv.buf)
            .arg(&sv.qnv.buf)
            .arg(&sv.msv.buf)
            .arg(&mut dqnv.buf)
            .arg(&n_i)
            .arg(&t_i)
            .arg(&dhv_only)
            .arg(&h_i);
        // One warp per timestep.
        let warps = SCAN_THREADS as usize / 32;
        unsafe { lb.launch(fused_cfg((n.div_ceil(warps) as u32, 1, 1), SCAN_THREADS, 0)) }
            .expect("mlstm_bw_dqn");
    }

    // Every chunk's own contribution to the BPTT state, all at once.
    let dc_threads = parallel_threads(dc_tiles(dqk, dhv));
    let (f, smem) = fused_kernel_spec(
        gpu,
        "mlstm_bw_dC",
        fused_smem("bw_dC", l, dqk, dhv),
        super::kernels::MlstmSpec {
            l,
            dqk,
            dhv,
            h: st.heads,
            threads: dc_threads,
            kt: 0,
            vt: 0,
        },
    );
    let mut lb = gpu.stream.launch_builder(&f);
    push_slab_ref!(lb, *xh);
    lb.arg(&d_ytil.buf)
        .arg(&sv.psiv.buf)
        .arg(&dqnv.buf)
        .arg(&sv.fcb.buf)
        .arg(&sv.mst.buf)
        .arg(&sv.msv.buf)
        .arg(&mut dcst.buf)
        .arg(&mut dnst.buf)
        .arg(&t_i)
        .arg(&l_i)
        .arg(&nc_i)
        .arg(&dqk_i)
        .arg(&dhv_i)
        .arg(&carry_i)
        .arg(&h_i);
    unsafe { lb.launch(fused_cfg((nc as u32, bh as u32, 1), dc_threads, smem)) }
        .expect("mlstm_bw_dC");

    // ... folded right to left. As in the forward, a single chunk with nothing
    // carried in has nothing to fold: slot 0 already holds the answer.
    if nc > 1 || carry_in.is_some() {
        state_scan(gpu, &mut dcst, &mut dnst, &sv.gvec, bh, nc, dqk, dhv, true);
    }

    // Shaped as the kernel writes them — position-major and concatenated, exactly as
    // the two projections laid out their outputs — so the caller needs neither a
    // scatter pass nor a per-part backward.
    let n = st.batch * t;
    debug_assert_eq!(
        dxh.dims(),
        [n, st.heads * (2 * dqk + 2 * dhv)],
        "mlstm_fused_bw — dq‖dk‖dv‖do shape"
    );
    debug_assert_eq!(
        dgates.dims(),
        [n, 2 * st.heads],
        "mlstm_fused_bw — dgates shape"
    );

    let par_threads = parallel_threads(bw_parallel_warps(l, dqk, dhv));
    let (f, smem) = fused_kernel_spec(
        gpu,
        "mlstm_bw_parallel",
        fused_smem("bw_parallel", l, dqk, dhv),
        super::kernels::MlstmSpec {
            l,
            dqk,
            dhv,
            h: st.heads,
            threads: par_threads,
            kt: mlstm_kt(),
            vt: mlstm_vt(),
        },
    );
    let mut lb = gpu.stream.launch_builder(&f);
    push_slab_ref!(lb, *xh);
    lb.arg(&gates.buf)
        .arg(&sv.fcb.buf)
        .arg(&sv.cst.buf)
        .arg(&sv.nst.buf)
        .arg(&sv.mst.buf)
        .arg(&dcst.buf)
        .arg(&dnst.buf)
        .arg(&d_ytil.buf)
        .arg(&sv.psiv.buf)
        .arg(&dqnv.buf)
        .arg(&sv.msv.buf)
        .arg(&mut dxh.buf)
        .arg(&mut dgates.buf)
        .arg(&t_i)
        .arg(&l_i)
        .arg(&nc_i)
        .arg(&dqk_i)
        .arg(&dhv_i)
        .arg(&carry_i)
        .arg(&h_i);
    unsafe { lb.launch(fused_cfg((nc as u32, bh as u32, 1), par_threads, smem)) }
        .expect("mlstm_bw_parallel");

    // Slot 0 is the gradient wrt the state entering chunk 0 — i.e. what flows out of
    // this chunk into the one on its left.
    // `carry_in` is dead now — seeded into the arrays above — so its buffers become
    // this chunk's output.
    let (prev_dc, prev_dn) = match carry_in {
        Some(d) => (Some(d.dc), Some(d.dn)),
        None => (None, None),
    };
    let dstate = MlstmDState {
        dc: read_state_slot_n(gpu, &dcst, bh, nc + 1, 0, dhv * dqk, prev_dc),
        dn: read_state_slot_n(gpu, &dnst, bh, nc + 1, 0, dqk, prev_dn),
    };
    dstate
}

#[cfg(test)]
mod tests {

    /// One temp cache per test, sized past every shape this module presents.
    fn test_cache(gpu: &Gpu) -> super::super::arena::TrainingCache {
        super::super::arena::TrainingCache::new(gpu, 1 << 20, 1 << 16, 1 << 20)
    }
    use super::*;
    use crate::tensor::{Tensor, gemm};

    fn assert_close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "length mismatch");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < 1e-3, "index {i}: gpu {g} vs cpu {w}");
        }
    }

    /// The block-per-row CE and chunk-ab backward must match their thread-per-row
    /// twins. Masked rows are included: a masked row must zero its gradient row and
    /// contribute no loss, which the block form does on a separate early-out path.
    #[test]
    fn block_ce_and_chunk_ab_bwd_match_thread_per_row() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        for &(r, c) in &[(5usize, 9usize), (7, 32), (4, 260), (3, 700)] {
            let logits = {
                let d: Vec<f32> = (0..r * c)
                    .map(|i| ((i as f32 * 0.31).cos()) * 4.0)
                    .collect();
                GTensor::from_host(&gpu, &Tensor::new(&[r, c], d))
            };
            let targets: Vec<u32> = (0..r).map(|i| (i * 7 % c) as u32).collect();
            // Mask out every third row, so both branches are exercised.
            let mask: Vec<u32> = (0..r).map(|i| u32::from(i % 3 != 0)).collect();
            let dt = gpu.stream.clone_htod(&targets).expect("targets");
            let dm = gpu.stream.clone_htod(&mask).expect("mask");
            let inv = 0.25;

            let (tv, mv) = (dt.slice(..), dm.slice(..));
            let (_, fast) = masked_softmax_ce_u32_into(&gpu, &logits, &tv, &mv, inv, None);

            let (c_i, r_i) = (c as i32, r as i32);
            let mut slow = GTensor::uninit(&gpu, &[r, c]);
            let mut rl = gpu.stream.alloc_zeros::<f32>(r).expect("row_loss");
            let f = gpu.kernels.get("masked_softmax_ce");
            let mut lb = gpu.stream.launch_builder(&f);
            lb.arg(&logits.buf)
                .arg(&tv)
                .arg(&mv)
                .arg(&mut slow.buf)
                .arg(&mut rl)
                .arg(&c_i)
                .arg(&inv)
                .arg(&r_i);
            unsafe { lb.launch(elem_cfg(&gpu, r as u32)) }.expect("slow ce");
            assert_close(&fast.to_host(&gpu).data, &slow.to_host(&gpu).data);
        }

        // chunk_ab backward: dfc/dig are accumulated onto, so both paths start from
        // the same non-zero state to catch a kernel that overwrites instead of adds.
        for &(bh, l) in &[(3usize, 8usize), (6, 64), (4, 256)] {
            let mk = |s: f32| {
                let d: Vec<f32> = (0..bh * l).map(|i| (i as f32 * 0.23 + s).sin()).collect();
                GTensor::from_host(&gpu, &Tensor::new(&[bh, l], d))
            };
            let (db, da, b, a) = (mk(0.0), mk(1.0), mk(2.0), mk(3.0));
            let (mut dfc_f, mut dig_f) = (mk(4.0), mk(5.0));
            let (mut dfc_s, mut dig_s) = (mk(4.0), mk(5.0));

            mlstm_chunk_ab_bwd(&gpu, &db, &da, &b, &a, &mut dfc_f, &mut dig_f);

            let (li, bhl) = (l as i32, (bh * l) as i32);
            let f = gpu.kernels.get("mlstm_chunk_ab_bwd");
            let mut lb = gpu.stream.launch_builder(&f);
            lb.arg(&db.buf)
                .arg(&da.buf)
                .arg(&b.buf)
                .arg(&a.buf)
                .arg(&mut dfc_s.buf)
                .arg(&mut dig_s.buf)
                .arg(&li)
                .arg(&bhl);
            unsafe { lb.launch(elem_cfg(&gpu, (bh * l) as u32)) }.expect("slow ab");

            assert_close(&dfc_f.to_host(&gpu).data, &dfc_s.to_host(&gpu).data);
            assert_close(&dig_f.to_host(&gpu).data, &dig_s.to_host(&gpu).data);
        }
    }

    /// The block-per-row gate scans must match the thread-per-row kernels exactly.
    ///
    /// Row lengths straddle the block width on purpose: `T` below the warp size, at a
    /// warp boundary, and above one block's worth (so the kernel's tile loop and its
    /// `running` carry between tiles are both exercised). A scan that lost the carry
    /// would still be right for `T <= blockDim` and wrong only past it.
    #[test]
    fn block_scans_match_thread_per_row() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        for &(bh, t) in &[(3usize, 7usize), (5, 32), (4, 100), (9, 256), (2, 1100)] {
            // Values spanning both signs, so log_sigmoid/sigmoid are exercised away
            // from their saturated tails where everything agrees trivially.
            let mk = |seed: f32| {
                let data: Vec<f32> = (0..bh * t)
                    .map(|i| ((i as f32 * 0.37 + seed).sin()) * 3.0)
                    .collect();
                GTensor::from_host(&gpu, &Tensor::new(&[bh, t], data))
            };
            let f = mk(0.0);
            let dfc = mk(1.7);

            let fast_fc = cumsum_logsig(&gpu, &f).to_host(&gpu);
            let fast_df = revcumsum_dlogsig(&gpu, &dfc, &f).to_host(&gpu);

            // Same inputs through the thread-per-row kernels.
            let (ti, bhi) = (t as i32, bh as i32);
            let mut slow_fc = GTensor::uninit(&gpu, &[bh, t]);
            let func = gpu.kernels.get("cumsum_logsig");
            let mut lb = gpu.stream.launch_builder(&func);
            lb.arg(&f.buf).arg(&mut slow_fc.buf).arg(&ti).arg(&bhi);
            unsafe { lb.launch(elem_cfg(&gpu, bh as u32)) }.expect("slow cumsum");

            let mut slow_df = GTensor::uninit(&gpu, &[bh, t]);
            let func = gpu.kernels.get("revcumsum_dlogsig");
            let mut lb = gpu.stream.launch_builder(&func);
            lb.arg(&dfc.buf)
                .arg(&f.buf)
                .arg(&mut slow_df.buf)
                .arg(&ti)
                .arg(&bhi);
            unsafe { lb.launch(elem_cfg(&gpu, bh as u32)) }.expect("slow revcumsum");

            assert_close(&fast_fc.data, &slow_fc.to_host(&gpu).data);
            assert_close(&fast_df.data, &slow_df.to_host(&gpu).data);
        }
    }

    /// The batched slice/unslice must agree element-for-element with the per-tensor
    /// kernels, including at differing row widths and a ragged final chunk.
    ///
    /// Widths differ on purpose: q/k/v share `dqk` but the backward pass batches
    /// `[dqk, dqk, dhv, 1, 1]`-wide tensors together, so a kernel that assumed one
    /// common `W` would pass a uniform test and corrupt the real sweep.
    #[test]
    fn slice_batch_matches_per_tensor_slice() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let (bh, t) = (6, 20);
        let widths = [4usize, 4, 7, 1, 3];
        let srcs: Vec<GTensor<f32>> = widths
            .iter()
            .enumerate()
            .map(|(k, &w)| {
                let data: Vec<f32> = (0..bh * t * w)
                    .map(|i| (i as f32) * 0.25 + k as f32 * 1000.0)
                    .collect();
                GTensor::from_host(&gpu, &Tensor::new(&[bh, t, w], data))
            })
            .collect();
        let refs: Vec<(&GTensor<f32>, usize)> = srcs
            .iter()
            .zip(widths.iter())
            .map(|(x, &w)| (x, w))
            .collect();

        // A chunk that does not divide T, so the last one is ragged.
        for (c0, len) in [(0usize, 7usize), (7, 7), (14, 6)] {
            let batched = slice_t_batch(&gpu, &refs, bh, t, c0, len);
            for (i, x) in srcs.iter().enumerate() {
                let want = slice_t(&gpu, x, c0, len).to_host(&gpu);
                assert_close(&batched[i].to_host(&gpu).data, &want.data);
            }

            // Round-trip: unslice the chunks back into fresh tensors and compare
            // against the per-tensor path doing the same.
            let mut got: Vec<GTensor<f32>> = widths
                .iter()
                .map(|&w| GTensor::zeros(&gpu, &[bh, t, w]))
                .collect();
            let mut want: Vec<GTensor<f32>> = widths
                .iter()
                .map(|&w| GTensor::zeros(&gpu, &[bh, t, w]))
                .collect();
            {
                let mut pairs: Vec<(&mut GTensor<f32>, &GTensor<f32>, usize)> = got
                    .iter_mut()
                    .zip(batched.iter())
                    .zip(widths.iter())
                    .map(|((d, s), &w)| (d, s, w))
                    .collect();
                unslice_t_batch(&gpu, &mut pairs, bh, t, c0, len);
            }
            for (w, b) in want.iter_mut().zip(batched.iter()) {
                unslice_t(&gpu, w, b, c0);
            }
            for (g, w) in got.iter().zip(want.iter()) {
                assert_close(&g.to_host(&gpu).data, &w.to_host(&gpu).data);
            }
        }
    }

    /// The bf16 GEMM must agree with the fp32 one to bf16's precision, in all three
    /// transpose forms — the operand swap and leading dimensions are hand-derived
    /// per form, so a wrong `ld` would silently transpose or stride the result.
    ///
    /// The bound is a relative one against the result's own scale: each product
    /// carries ~2^-8 relative error, and summing K of them lets that grow like √K
    /// in the worst realistic case, so the tolerance is scaled by √K rather than
    /// being a fixed constant.
    #[test]
    fn gemm_bf16_matches_fp32_all_forms() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        if !gemm_bf16_enabled(&gpu) {
            eprintln!("skipping: bf16 unavailable");
            return;
        }
        let (m, k, n) = (37usize, 53usize, 29usize);
        for form in [MmForm::Nn, MmForm::Nt, MmForm::Tn] {
            let (ad, bd) = match form {
                MmForm::Nn => ([m, k], [k, n]),
                MmForm::Nt => ([m, k], [n, k]),
                MmForm::Tn => ([k, m], [k, n]),
            };
            let a = Tensor::random(&ad, 1.0);
            let b = Tensor::random(&bd, 1.0);
            let (da, db) = (GTensor::from_host(&gpu, &a), GTensor::from_host(&gpu, &b));

            let mut want = GTensor::uninit(&gpu, &[m, n]);
            match form {
                MmForm::Nn => matmul_nn_into(&gpu, &da, &db, &mut want, 0.0),
                MmForm::Nt => matmul_nt_into(&gpu, &da, &db, &mut want, 0.0),
                MmForm::Tn => matmul_tn_into(&gpu, &da, &db, &mut want, 0.0),
            }
            let mut got = GTensor::uninit(&gpu, &[m, n]);
            GemmBf16::new().run(&gpu, form, &da, &db, &mut got, 0.0);

            let (w, g) = (want.to_host(&gpu).data, got.to_host(&gpu).data);
            let scale = w.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
            let bound = 4e-3 * scale * (k as f32).sqrt();
            for (i, (a, b)) in g.iter().zip(&w).enumerate() {
                assert!(
                    (a - b).abs() < bound,
                    "form {} elem {i}: bf16 {a} vs fp32 {b} (bound {bound:.2e})",
                    match form {
                        MmForm::Nn => "NN",
                        MmForm::Nt => "NT",
                        MmForm::Tn => "TN",
                    }
                );
            }
        }
    }

    /// The staging buffers must be REUSED across calls, not reallocated.
    ///
    /// This runs on every projection of every block, every step, so a staging buffer
    /// that reallocated per call would put the allocator back on the hot path — and
    /// because cudarc frees through a retaining async pool, the freed copies would
    /// also inflate the resident set. Pinned by device address: a steady shape must
    /// keep the same allocation, and a smaller one must reuse it rather than
    /// allocate a second.
    #[test]
    fn gemm_bf16_staging_reuses_its_buffers() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        if !gemm_bf16_enabled(&gpu) {
            return;
        }
        use cudarc::driver::DevicePtr;
        let mut g = GemmBf16::new();
        let addr = |g: &GemmBf16| {
            let l = g
                .lhs
                .as_ref()
                .expect("staged")
                .buf
                .device_ptr(&gpu.stream)
                .0;
            let r = g
                .rhs
                .as_ref()
                .expect("staged")
                .buf
                .device_ptr(&gpu.stream)
                .0;
            (l, r)
        };

        let a = GTensor::from_host(&gpu, &Tensor::random(&[64, 32], 1.0));
        let b = GTensor::from_host(&gpu, &Tensor::random(&[32, 16], 1.0));
        let mut c = GTensor::uninit(&gpu, &[64, 16]);
        g.run(&gpu, MmForm::Nn, &a, &b, &mut c, 0.0);
        let first = addr(&g);
        for _ in 0..5 {
            g.run(&gpu, MmForm::Nn, &a, &b, &mut c, 0.0);
            assert_eq!(
                first,
                addr(&g),
                "steady shape must reuse the staging buffers"
            );
        }

        // A smaller call fits the existing allocation, so it must not reallocate —
        // this is what keeps a varying window size from thrashing.
        let a2 = GTensor::from_host(&gpu, &Tensor::random(&[32, 32], 1.0));
        let mut c2 = GTensor::uninit(&gpu, &[32, 16]);
        g.run(&gpu, MmForm::Nn, &a2, &b, &mut c2, 0.0);
        assert_eq!(
            first,
            addr(&g),
            "a smaller shape must fit the existing buffers"
        );
    }

    /// `beta = 1` must accumulate into C rather than overwrite it — the weight
    /// gradient form relies on this.
    #[test]
    fn gemm_bf16_accumulates_with_beta_one() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        if !gemm_bf16_enabled(&gpu) {
            return;
        }
        let (m, k, n) = (8usize, 16usize, 4usize);
        let a = Tensor::random(&[m, k], 1.0);
        let b = Tensor::random(&[k, n], 1.0);
        let (da, db) = (GTensor::from_host(&gpu, &a), GTensor::from_host(&gpu, &b));

        let mut once = GTensor::zeros(&gpu, &[m, n]);
        GemmBf16::new().run(&gpu, MmForm::Nn, &da, &db, &mut once, 0.0);
        let single = once.to_host(&gpu).data;

        let mut twice = GTensor::zeros(&gpu, &[m, n]);
        let mut g = GemmBf16::new();
        g.run(&gpu, MmForm::Nn, &da, &db, &mut twice, 0.0);
        g.run(&gpu, MmForm::Nn, &da, &db, &mut twice, 1.0);
        let doubled = twice.to_host(&gpu).data;

        for (i, (d, s)) in doubled.iter().zip(&single).enumerate() {
            assert!(
                (d - 2.0 * s).abs() < 1e-4 * s.abs().max(1.0),
                "beta=1 did not accumulate at {i}: {d} vs 2*{s}"
            );
        }
    }

    /// Tolerance is one bf16 rounding (~8 mantissa bits), not the fp32 path's.
    #[test]
    fn gemm_bf16_bias_matches_broadcast_plus_beta() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        if !gemm_bf16_enabled(&gpu) {
            return;
        }
        let (m, k, n) = (16usize, 32usize, 8usize);
        let x = Tensor::random(&[m, k], 1.0);
        let w = Tensor::random(&[k, n], 1.0);
        let bias = Tensor::random(&[n], 1.0);
        let (dx, dw) = (GTensor::from_host(&gpu, &x), GTensor::from_host(&gpu, &w));
        let dbias = GTensor::from_host(&gpu, &bias);

        // Reference: seed the bias, accumulate the GEMM onto it in fp32.
        let mut want = GTensor::uninit(&gpu, &[m, n]);
        broadcast_row(&gpu, &mut want, &dbias);
        GemmBf16::new().run(&gpu, MmForm::Nn, &dx, &dw, &mut want, 1.0);
        let want = want.to_host(&gpu).data;

        // Lt: same operands, bias in the epilogue, bf16 output.
        let mut xb = super::super::GTensor::uninit(&gpu, &[m, k]);
        xb.store(&gpu, &dx);
        let mut wb = super::super::GTensor::uninit(&gpu, &[k, n]);
        wb.store(&gpu, &dw);
        let mut bb = super::super::GTensor::uninit(&gpu, &[n]);
        bb.store(&gpu, &dbias);
        let mut got_b = super::super::GTensor::uninit(&gpu, &[m, n]);
        matmul_bf16_bias_into(&gpu, &xb, &wb, &bb, &mut got_b);
        let mut wide = GTensor::uninit(&gpu, &[m, n]);
        got_b.load(&gpu, &mut wide);
        let got = wide.to_host(&gpu).data;

        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 2e-2 * w.abs().max(1.0),
                "Lt bias epilogue differs at {i}: {g} vs {w}"
            );
        }
    }

    #[test]
    fn gemm_nn_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let (m, k, n) = (17, 23, 11);
        let a = Tensor::random(&[m, k], 1.0);
        let b = Tensor::random(&[k, n], 1.0);
        let want = gemm::matmul(&a, &b);
        let got = matmul(
            &gpu,
            &GTensor::from_host(&gpu, &a),
            &GTensor::from_host(&gpu, &b),
        );
        assert_close(&got.to_host(&gpu).data, &want.data);
    }

    #[test]
    fn gemm_nt_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let (m, k, n) = (17, 23, 11);
        let a = Tensor::random(&[m, k], 1.0);
        let b = Tensor::random(&[n, k], 1.0);
        let want = gemm::matmul_nt(&a, &b);
        let got = matmul_nt(
            &gpu,
            &GTensor::from_host(&gpu, &a),
            &GTensor::from_host(&gpu, &b),
        );
        assert_close(&got.to_host(&gpu).data, &want.data);
    }

    #[test]
    fn gemm_tn_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let (m, k, n) = (17, 23, 11);
        let a = Tensor::random(&[k, m], 1.0);
        let b = Tensor::random(&[k, n], 1.0);
        let want = gemm::matmul_tn(&a, &b);
        let got = matmul_tn(
            &gpu,
            &GTensor::from_host(&gpu, &a),
            &GTensor::from_host(&gpu, &b),
        );
        assert_close(&got.to_host(&gpu).data, &want.data);
    }

    /// Per-batch CPU reference for the three strided-batched GEMM forms, checked
    /// against `tensor::gemm` looped over the batch axis.
    #[test]
    fn matmul_batched_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let (batch, m, k, n) = (5, 7, 6, 4);

        // nn: [batch,m,k]·[batch,k,n]
        let a = Tensor::random(&[batch, m, k], 1.0);
        let b = Tensor::random(&[batch, k, n], 1.0);
        let mut want = Tensor::zeros(&[batch, m, n]);
        for g in 0..batch {
            let (ao, bo, co) = (g * m * k, g * k * n, g * m * n);
            gemm::gemm_nn(
                m,
                k,
                n,
                &a.data[ao..ao + m * k],
                &b.data[bo..bo + k * n],
                &mut want.data[co..co + m * n],
                0.0,
            );
        }
        let got = matmul_batched_nn(
            &gpu,
            &GTensor::from_host(&gpu, &a),
            &GTensor::from_host(&gpu, &b),
        );
        assert_close(&got.to_host(&gpu).data, &want.data);

        // nt: [batch,m,k]·[batch,n,k]ᵀ
        let bt = Tensor::random(&[batch, n, k], 1.0);
        let mut want_nt = Tensor::zeros(&[batch, m, n]);
        for g in 0..batch {
            let (ao, bo, co) = (g * m * k, g * n * k, g * m * n);
            gemm::gemm_nt(
                m,
                k,
                n,
                &a.data[ao..ao + m * k],
                &bt.data[bo..bo + n * k],
                &mut want_nt.data[co..co + m * n],
                0.0,
            );
        }
        let got_nt = matmul_batched_nt(
            &gpu,
            &GTensor::from_host(&gpu, &a),
            &GTensor::from_host(&gpu, &bt),
        );
        assert_close(&got_nt.to_host(&gpu).data, &want_nt.data);

        // tn: [batch,k,m]ᵀ·[batch,k,n]
        let at = Tensor::random(&[batch, k, m], 1.0);
        let bn = Tensor::random(&[batch, k, n], 1.0);
        let mut want_tn = Tensor::zeros(&[batch, m, n]);
        for g in 0..batch {
            let (ao, bo, co) = (g * k * m, g * k * n, g * m * n);
            gemm::gemm_tn(
                m,
                k,
                n,
                &at.data[ao..ao + k * m],
                &bn.data[bo..bo + k * n],
                &mut want_tn.data[co..co + m * n],
                0.0,
            );
        }
        let got_tn = matmul_batched_tn(
            &gpu,
            &GTensor::from_host(&gpu, &at),
            &GTensor::from_host(&gpu, &bn),
        );
        assert_close(&got_tn.to_host(&gpu).data, &want_tn.data);
    }

    #[test]
    fn scale_and_sigmoid_match_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let x = Tensor::random(&[6, 5], 2.0);
        // scale
        let s = 0.37;
        let mut want_scale = x.clone();
        for v in want_scale.data.iter_mut() {
            *v *= s;
        }
        let mut dx = GTensor::from_host(&gpu, &x);
        scale_(&gpu, &mut dx, s);
        assert_close(&dx.to_host(&gpu).data, &want_scale.data);
        // sigmoid (matches the cell's stable_sigmoid)
        let sig = |v: f32| {
            if v >= 0.0 {
                1.0 / (1.0 + (-v).exp())
            } else {
                let e = v.exp();
                e / (1.0 + e)
            }
        };
        let want_sig: Vec<f32> = x.data.iter().map(|&v| sig(v)).collect();
        let mut dx2 = GTensor::from_host(&gpu, &x);
        sigmoid_(&gpu, &mut dx2);
        assert_close(&dx2.to_host(&gpu).data, &want_sig);
    }

    #[test]
    fn softcap_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        use crate::nn2::ops as cpu;
        let cap = 30.0;
        let x = Tensor::random(&[9, 13], 50.0);
        let dy = Tensor::random(&[9, 13], 1.0);
        let y_cpu = cpu::softcap_forward(&x, cap);
        let dx_cpu = cpu::softcap_backward(&dy, &y_cpu, cap);
        let dx = GTensor::from_host(&gpu, &x);
        let y_gpu = softcap_forward(&gpu, &dx, cap);
        let dx_gpu = softcap_backward(&gpu, &GTensor::from_host(&gpu, &dy), &y_gpu, cap);
        assert_close(&y_gpu.to_host(&gpu).data, &y_cpu.data);
        assert_close(&dx_gpu.to_host(&gpu).data, &dx_cpu.data);
    }

    #[test]
    fn linear_bias_helpers_match_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        use crate::nn2::ops as cpu;
        let (b, n) = (7, 5);
        let bias = Tensor::random(&[n], 1.0);
        let dy = Tensor::random(&[b, n], 1.0);
        // broadcast_row
        let mut out_cpu = Tensor::zeros(&[b, n]);
        cpu::broadcast_row(&mut out_cpu, &bias);
        let mut out_gpu = GTensor::zeros(&gpu, &[b, n]);
        broadcast_row(&gpu, &mut out_gpu, &GTensor::from_host(&gpu, &bias));
        assert_close(&out_gpu.to_host(&gpu).data, &out_cpu.data);
        // add_col_sum (start from a nonzero db to check accumulation)
        let mut db_cpu = Tensor::random(&[n], 1.0);
        let mut db_gpu = GTensor::from_host(&gpu, &db_cpu);
        cpu::add_col_sum(&mut db_cpu, &dy);
        add_col_sum(&gpu, &mut db_gpu, &GTensor::from_host(&gpu, &dy), &tc.temps);
        assert_close(&db_gpu.to_host(&gpu).data, &db_cpu.data);
    }

    /// `add_col_sum` bands its row axis across `blockIdx.y` above a row count, so the
    /// shapes that matter are the wide, many-row ones a real layer sees — where the
    /// band split is active, including a row count the bands do not divide evenly.
    #[test]
    fn add_col_sum_matches_cpu_at_layer_shapes() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        use crate::nn2::ops as cpu;
        for (rows, n) in [(512, 768), (2048, 768), (1, 768), (333, 257)] {
            let dy = Tensor::random(&[rows, n], 1.0);
            let mut db_cpu = Tensor::random(&[n], 1.0);
            let mut db_gpu = GTensor::from_host(&gpu, &db_cpu);
            cpu::add_col_sum(&mut db_cpu, &dy);
            add_col_sum(&gpu, &mut db_gpu, &GTensor::from_host(&gpu, &dy), &tc.temps);
            let got = db_gpu.to_host(&gpu).data;
            // Summation order differs from the CPU's row-major walk, so this is a
            // float-reassociation tolerance, not an exactness check.
            for (i, (g, w)) in got.iter().zip(&db_cpu.data).enumerate() {
                assert!(
                    (g - w).abs() < 1e-3,
                    "[{rows}x{n}] col {i}: gpu {g} vs cpu {w}"
                );
            }
        }
    }

    #[test]
    fn embedding_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        use crate::nn2::ops as cpu;
        let (vocab, dim) = (11, 6);
        let table = Tensor::random(&[vocab, dim], 1.0);
        let ids = [3usize, 0, 3, 7, 3]; // repeats -> exercises scatter atomics
        let gathered_cpu = cpu::embedding_gather(&table, &ids, dim);
        let gathered_gpu = embedding_gather(&gpu, &GTensor::from_host(&gpu, &table), &ids, dim);
        assert_close(&gathered_gpu.to_host(&gpu).data, &gathered_cpu.data);
        // scatter_add from the gathered grads
        let mut dt_cpu = Tensor::zeros(&[vocab, dim]);
        cpu::embedding_scatter_add(&mut dt_cpu, &ids, &gathered_cpu, dim);
        let mut dt_gpu = GTensor::zeros(&gpu, &[vocab, dim]);
        embedding_scatter_add(&gpu, &mut dt_gpu, &ids, &gathered_gpu, dim);
        assert_close(&dt_gpu.to_host(&gpu).data, &dt_cpu.data);
    }

    /// The block-per-group RMSNorm kernels must match the CPU reference at a WIDE
    /// group — the regime they exist for, and the one `rms_norm_matches_cpu` does
    /// NOT cover (it uses group=4, which routes to the thread-per-group kernel).
    ///
    /// Both directions, and `dgamma` too: the backward accumulates it with atomics
    /// from every thread of every block, so a reduction bug shows up there first.
    /// Non-multiple-of-blockDim widths are included because the strided loops must
    /// handle a ragged tail.
    #[test]
    fn rms_norm_block_kernel_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        use crate::nn2::ops as cpu;
        let eps = 1e-5;
        // (rows, features, group): ungrouped wide (the block norms), a multi-group
        // wide case, and a width that is not a multiple of the 256 block threads.
        for &(b, f, group) in &[(7usize, 1024usize, 1024usize), (3, 512, 256), (5, 300, 300)] {
            assert!(group >= 128, "this test must exercise the block kernel");
            let x = Tensor::random(&[b, f], 1.0);
            let gamma = Tensor::random(&[f], 1.0);
            let dy = Tensor::random(&[b, f], 1.0);

            let fwd_cpu = cpu::rms_norm_forward(&x, &gamma, group, eps);
            let mut dg_cpu = Tensor::zeros(&[f]);
            let dx_cpu = cpu::rms_norm_backward(
                &dy,
                &fwd_cpu.x_hat,
                &fwd_cpu.inv_rms,
                &gamma,
                &mut dg_cpu,
                group,
            );

            let gamma_d = GTensor::from_host(&gpu, &gamma);
            let (out_gpu, fwd_gpu) =
                rms_norm_forward(&gpu, &GTensor::from_host(&gpu, &x), &gamma_d, group, eps);
            let mut dg_gpu = GTensor::zeros(&gpu, &[f]);
            let dx_gpu = rms_norm_backward(
                &gpu,
                &GTensor::from_host(&gpu, &dy),
                &fwd_gpu,
                &out_gpu,
                &gamma_d,
                &mut dg_gpu,
                group,
                &tc.temps,
            );

            let what = format!("b={b} f={f} group={group}");
            for (name, got, want) in [
                ("out", out_gpu.to_host(&gpu).data, fwd_cpu.out.data.clone()),
                ("dx", dx_gpu.to_host(&gpu).data, dx_cpu.data.clone()),
                ("dgamma", dg_gpu.to_host(&gpu).data, dg_cpu.data.clone()),
            ] {
                assert_eq!(got.len(), want.len(), "{what}: {name} length");
                let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    // A tree reduction sums in a different order than the CPU's
                    // sequential one, so this is fp32 reassociation, not an error.
                    assert!(
                        (g - w).abs() < 1e-4 * scale.max(1.0),
                        "{what}: {name}[{i}] gpu {g} vs cpu {w}"
                    );
                }
            }
        }
    }

    #[test]
    fn rms_norm_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        use crate::nn2::ops as cpu;
        let (b, f, group, eps) = (5, 12, 4, 1e-5); // head-wise case: 3 groups/row
        let x = Tensor::random(&[b, f], 1.0);
        let gamma = Tensor::random(&[f], 1.0);
        let dy = Tensor::random(&[b, f], 1.0);
        let fwd_cpu = cpu::rms_norm_forward(&x, &gamma, group, eps);
        let mut dg_cpu = Tensor::zeros(&[f]);
        let dx_cpu = cpu::rms_norm_backward(
            &dy,
            &fwd_cpu.x_hat,
            &fwd_cpu.inv_rms,
            &gamma,
            &mut dg_cpu,
            group,
        );

        let dgamma_t = GTensor::from_host(&gpu, &gamma);
        let (out_gpu, fwd_gpu) =
            rms_norm_forward(&gpu, &GTensor::from_host(&gpu, &x), &dgamma_t, group, eps);
        let mut dg_gpu = GTensor::zeros(&gpu, &[f]);
        let dx_gpu = rms_norm_backward(
                &gpu,
            &GTensor::from_host(&gpu, &dy),
            &fwd_gpu,
            &out_gpu,
            &dgamma_t,
            &mut dg_gpu,
            group,
            &tc.temps,
        );
        assert_close(&out_gpu.to_host(&gpu).data, &fwd_cpu.out.data);
        assert_close(&dx_gpu.to_host(&gpu).data, &dx_cpu.data);
        assert_close(&dg_gpu.to_host(&gpu).data, &dg_cpu.data);
    }

    #[test]
    fn softmax_ce_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        use crate::nn2::loss;
        let (b, c) = (6, 9);
        let logits = Tensor::random(&[b, c], 2.0);
        let targets = [0usize, 8, 3, 5, 1, 7];
        let (loss_cpu, d_cpu) = loss::softmax_cross_entropy(&logits, &targets);
        let (loss_gpu, d_gpu) =
            softmax_cross_entropy(&gpu, &GTensor::from_host(&gpu, &logits), &targets);
        assert!(
            (loss_cpu - loss_gpu).abs() < 1e-4,
            "loss {loss_cpu} vs {loss_gpu}"
        );
        assert_close(&d_gpu.to_host(&gpu).data, &d_cpu.data);
    }

    #[test]
    fn adamw_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        use crate::nn2::optim::{AdamCfg, AdamState};
        let n = 20;
        let param = Tensor::random(&[n], 1.0);
        let grad = Tensor::random(&[n], 1.0);
        let mut cfg = AdamCfg::new(1e-3, 0.01);
        cfg.t = 1;

        let mut param_cpu = param.clone();
        let mut st = AdamState::new();
        st.step(&mut param_cpu.data, &grad.data, &cfg, true);

        let mut p_gpu = GTensor::from_host(&gpu, &param);
        let mut m = GTensor::zeros(&gpu, &[n]);
        let mut v = GTensor::zeros(&gpu, &[n]);
        adamw(
            &gpu,
            &mut p_gpu,
            &GTensor::from_host(&gpu, &grad),
            &mut m,
            &mut v,
            &cfg,
            true,
        );
        assert_close(&p_gpu.to_host(&gpu).data, &param_cpu.data);
    }
}
