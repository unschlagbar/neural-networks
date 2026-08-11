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

use cudarc::cublas::sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};
use cudarc::cublas::{Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use super::{DTensor, Gpu};
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

/// A reusable staging buffer that uploads several id lists in **one** transfer.
///
/// The per-window index bookkeeping (which row is a `[W]` step, which slot is a char,
/// the CE targets and mask) is a handful of small `u32` lists. Sending each with its
/// own [`upload_ids_u32`] costs one device allocation and one *blocking* H2D apiece —
/// `memcpy_htod` from a pageable host slice is synchronous, so the compute stream
/// stalls on every one. The decoder issues six per length group, the encoder two, and
/// both run again in backward: ~30 stalls per window, for ~100 KB of data.
///
/// This packs the lists back-to-back into one pinned host buffer, does a single async
/// copy into one device buffer, and hands back views. Pinned + async means the copy
/// also overlaps whatever compute is already queued instead of blocking on it.
///
/// The buffers are reused across calls, so a steady training loop neither allocates
/// nor page-locks after the first window.
///
/// # Reuse hazard
///
/// The H2D is **async**, so the host and device buffers are still being read after
/// `upload` returns. A single set of buffers would therefore be overwritten by the
/// next group while the previous group's copy — and the kernels reading its device
/// views — were still outstanding: the ids silently become the *next* group's, which
/// showed up as training diverging (loss 2.6 → 69) rather than as an error.
///
/// Rotating slots alone does **not** fix this, which is the trap: it only buys
/// `ID_SLOTS - 1` iterations of slack, and the CPU runs arbitrarily far ahead of the
/// device inside a loop over groups. What makes it safe is the per-slot event — the
/// CPU blocks on the previous copy out of that slot before refilling it. The extra
/// slots exist only so that wait is almost never reached.
///
/// `get` borrows `&self` so the views cannot outlive the next `upload`, which the
/// borrow checker enforces.
///
/// Staging slots. The per-slot event is what makes reuse safe; the slot count only
/// affects how often the CPU reaches that wait. Measured at 2, 4 and 64 slots: no
/// difference in step time, so the wait is essentially never hit in practice and
/// two slots is enough.
const ID_SLOTS: usize = 2;

pub struct IdBatch {
    host: [Option<cudarc::driver::PinnedHostSlice<u32>>; ID_SLOTS],
    dev: [Option<CudaSlice<u32>>; ID_SLOTS],
    /// Which slot the last `upload` wrote, and which `get` therefore reads.
    cur: usize,
    /// Completion of each half's H2D copy.
    ///
    /// The copy is async and reads the *pinned host* buffer, but the refill at the
    /// top of `upload` is a plain CPU write. Rotating slots alone only buys
    /// `ID_SLOTS - 1` iterations of slack, and the CPU runs arbitrarily far ahead of
    /// the device inside a loop over groups, so it laps the rotation and overwrites
    /// bytes a queued copy has not read yet. That is silent, nondeterministic
    /// corruption of the ids, not a crash.
    copied: [Option<cudarc::driver::CudaEvent>; ID_SLOTS],
    /// `(offset, len)` of each list in this batch, in element units.
    spans: Vec<(usize, usize)>,
}

impl Default for IdBatch {
    fn default() -> Self {
        Self::new()
    }
}

impl IdBatch {
    pub const fn new() -> Self {
        Self {
            host: [const { None }; ID_SLOTS],
            dev: [const { None }; ID_SLOTS],
            copied: [const { None }; ID_SLOTS],
            cur: 0,
            spans: Vec::new(),
        }
    }

    /// Pack `lists` into one upload. Returns immediately; the copy is queued on
    /// `gpu.stream`, so the views are safe for any kernel launched after this.
    ///
    /// Views are indexed by position in `lists` — see [`get`](Self::get).
    pub fn upload(&mut self, gpu: &Gpu, lists: &[&[u32]]) {
        self.spans.clear();
        let mut total = 0;
        for l in lists {
            self.spans.push((total, l.len()));
            total += l.len();
        }
        if total == 0 {
            return;
        }

        // Grow the staging buffers only when this batch does not fit. Element counts
        // vary window to window, so an exact fit would reallocate constantly — and
        // page-locking is far too expensive to do per window.
        // Alternate halves so this upload never overwrites buffers the previous
        // one's copy (or the kernels reading its views) may still be using.
        self.cur = (self.cur + 1) % ID_SLOTS;
        let half = self.cur;

        // Block until this half's previous copy has actually read the pinned buffer.
        // A device-side `stream.wait` would not do: the racing write below is the
        // CPU's, and the CPU is what has to be held back.
        if let Some(ev) = &self.copied[half] {
            ev.synchronize().expect("IdBatch: wait for prior H2D");
        }

        let need = total;
        if self.host[half].as_ref().is_none_or(|h| h.len() < need) {
            // SAFETY: uninitialised pinned memory, fully written below before the copy
            // reads it (only `spans` worth is ever copied out).
            self.host[half] = Some(
                unsafe { gpu.context.alloc_pinned::<u32>(need) }.expect("IdBatch: pinned alloc"),
            );
        }
        if self.dev[half].as_ref().is_none_or(|d| d.len() < need) {
            // SAFETY: every element read downstream lies in a span written by the copy.
            self.dev[half] =
                Some(unsafe { gpu.stream.alloc::<u32>(need) }.expect("IdBatch: alloc"));
        }

        let host = self.host[half].as_mut().expect("just filled");
        let dst = host.as_mut_slice().expect("pinned host slice");
        for (l, (off, len)) in lists.iter().zip(&self.spans) {
            dst[*off..*off + *len].copy_from_slice(l);
        }
        // Copy only the used prefix: the pinned buffer is sized to the largest batch
        // seen, and the tail holds a previous window's ids.
        let dev = self.dev[half].as_mut().expect("just filled");
        gpu.stream
            .memcpy_htod(&dst[..need], &mut dev.slice_mut(..need))
            .expect("IdBatch: H2D");
        self.copied[half] = Some(
            gpu.stream
                .record_event(None)
                .expect("IdBatch: record H2D completion"),
        );
    }

    /// The `i`-th uploaded list, as a device view.
    pub fn get(&self, i: usize) -> cudarc::driver::CudaView<'_, u32> {
        let (off, len) = self.spans[i];
        self.dev[self.cur]
            .as_ref()
            .expect("IdBatch::get before upload")
            .slice(off..off + len)
    }
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
/// This is what the reference does at its autograd boundary: `@custom_fwd(
/// cast_inputs=autocast_kernel_dtype)` with `autocast_kernel_dtype="bfloat16"`
/// casts the projection inputs to bf16, and the matmuls then run on the tensor
/// cores with `CUBLAS_COMPUTE_32F` — operands narrow, accumulation wide. cudarc
/// ships exactly this shape for `half::f16` (`Gemm<f16>` calls `result::gemm_ex`
/// with `CUDA_R_16F` + `CUBLAS_COMPUTE_32F`); this is the `CUDA_R_16BF` twin, which
/// the crate does not provide.
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
    a: &super::BTensor,
    b: &super::BTensor,
    c: &mut DTensor,
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
    lhs: Option<super::BTensor>,
    rhs: Option<super::BTensor>,
    /// Narrowed copy of the layer's weight, reused until [`invalidate_w`](Self::invalidate_w).
    ///
    /// The weight is the one operand that does not change between GEMMs: a layer
    /// narrows it for its forward (`Y = X·W`) and again for `dX = dY·Wᵀ`, and only an
    /// optimizer step writes it in between. Re-narrowing it each time made
    /// `cast_f32_to_bf16` the single largest kernel on the profile (17% of GPU time).
    w: Option<super::BTensor>,
    /// Whether [`w`](Self::w) currently matches the fp32 weight.
    w_valid: bool,
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
        }
    }

    /// Mark the cached bf16 weight stale. **Must** be called by anything that writes
    /// the fp32 weight — an optimizer step, a checkpoint load, a manual overwrite.
    /// Missing a call means the GEMMs keep reading a stale weight, which shows up as
    /// training silently not learning rather than as an error.
    pub fn invalidate_w(&mut self) {
        self.w_valid = false;
    }

    /// Narrow `w` once and reuse it until invalidated. See [`w`](Self::w).
    fn stage_w(&mut self, gpu: &Gpu, src: &DTensor) -> &super::BTensor {
        // A hit reuses the buffer *at the dims it was narrowed with*, so it is only
        // sound while the weight's shape is fixed — which it is: a layer's `w` is
        // allocated once at `[in, out]` and only ever stepped in place. A shape change
        // therefore means a different tensor, and is treated as a miss.
        let same_shape = self.w.as_ref().is_some_and(|t| t.dims() == src.dims());
        if !(self.w_valid && same_shape) {
            let n = src.len();
            match &mut self.w {
                Some(t) if t.capacity() >= n => t.shrink_to(src.dims()),
                _ => self.w = Some(super::BTensor::uninit(gpu, src.dims())),
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
        self.lhs = None;
        self.rhs = None;
        self.w = None;
        self.w_valid = false;
    }

    /// Narrow `a` and `b` into the owned staging buffers and run the GEMM.
    pub fn run(
        &mut self,
        gpu: &Gpu,
        form: MmForm,
        a: &DTensor,
        b: &DTensor,
        c: &mut DTensor,
        beta: f32,
    ) {
        fn stage<'s>(
            gpu: &Gpu,
            slot: &'s mut Option<super::BTensor>,
            src: &DTensor,
        ) -> &'s super::BTensor {
            let n = src.len();
            match slot {
                Some(t) if t.capacity() >= n => t.shrink_to(src.dims()),
                _ => *slot = Some(super::BTensor::uninit(gpu, src.dims())),
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
        a: &DTensor,
        w: &DTensor,
        c: &mut DTensor,
        beta: f32,
    ) {
        let n = a.len();
        match &mut self.lhs {
            Some(t) if t.capacity() >= n => t.shrink_to(a.dims()),
            _ => self.lhs = Some(super::BTensor::uninit(gpu, a.dims())),
        }
        self.lhs.as_mut().expect("just filled").store(gpu, a);
        if !wcache_enabled() {
            // Narrow through the shared `rhs` slot, so the dedicated cache buffer is
            // never allocated — the point of the switch is to give the memory back,
            // and keeping the buffer while skipping only the cast would not.
            let n2 = w.len();
            match &mut self.rhs {
                Some(t) if t.capacity() >= n2 => t.shrink_to(w.dims()),
                _ => self.rhs = Some(super::BTensor::uninit(gpu, w.dims())),
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
    b: &DTensor,
    a: &DTensor,
    c: &mut DTensor,
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
pub fn matmul_batched_nn(gpu: &Gpu, a: &DTensor, b: &DTensor) -> DTensor {
    let (batch, m, ka) = (a.shape[0], a.shape[1], a.shape[2]);
    let (kb, n) = (b.shape[1], b.shape[2]);
    assert_eq!(ka, kb, "matmul_batched_nn: inner dims {ka} != {kb}");
    assert_eq!(batch, b.shape[0], "matmul_batched_nn: batch mismatch");
    let mut c = DTensor::uninit(gpu, &[batch, m, n]);
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
pub fn matmul_batched_nt(gpu: &Gpu, a: &DTensor, b: &DTensor) -> DTensor {
    let (batch, m, ka) = (a.shape[0], a.shape[1], a.shape[2]);
    let (n, kb) = (b.shape[1], b.shape[2]);
    assert_eq!(ka, kb, "matmul_batched_nt: inner dims {ka} != {kb}");
    assert_eq!(batch, b.shape[0], "matmul_batched_nt: batch mismatch");
    let mut c = DTensor::uninit(gpu, &[batch, m, n]);
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
pub fn matmul_batched_tn(gpu: &Gpu, a: &DTensor, b: &DTensor) -> DTensor {
    let (batch, ka, m) = (a.shape[0], a.shape[1], a.shape[2]);
    let (kb, n) = (b.shape[1], b.shape[2]);
    assert_eq!(ka, kb, "matmul_batched_tn: outer dims {ka} != {kb}");
    assert_eq!(batch, b.shape[0], "matmul_batched_tn: batch mismatch");
    let mut c = DTensor::uninit(gpu, &[batch, m, n]);
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
    lb.arg(&dy.buf)
        .arg(&y.buf)
        .arg(&mut dx.buf)
        .arg(&cap)
        .arg(&n_i);
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
    // Split the row axis over `blockIdx.y` until the launch has enough blocks to fill
    // the device: `n` alone is a handful of warps for a typical width. Capped so the
    // atomic traffic per column stays small, and at `rows` so no slice is empty.
    const BLOCK: usize = 256;
    const TARGET_BLOCKS: usize = 256;
    let x_blocks = n.div_ceil(BLOCK).max(1);
    let y = rows.min(TARGET_BLOCKS.div_ceil(x_blocks)).max(1);
    let cfg = LaunchConfig {
        grid_dim: (x_blocks as u32, y as u32, 1),
        block_dim: (BLOCK as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { lb.launch(cfg) }.expect("add_col_sum");
}

/// Gather rows of `table` (`[vocab, dim]`) by `ids` into a `[ids.len(), dim]`
/// tensor.
pub fn embedding_gather(gpu: &Gpu, table: &DTensor, ids: &[usize], dim: usize) -> DTensor {
    embedding_gather_u32(gpu, table, &upload_ids(gpu, ids).slice(..), ids.len(), dim)
}

/// [`embedding_gather`] against ids already resident on the device. `rows` is
/// the id count, which the caller knows and a `CudaSlice` may over-allocate.
pub fn embedding_gather_u32(
    gpu: &Gpu,
    table: &DTensor,
    dids: &cudarc::driver::CudaView<'_, u32>,
    rows: usize,
    dim: usize,
) -> DTensor {
    let (dim_i, rows_i) = (dim as i32, rows as i32);
    let mut out = DTensor::uninit(gpu, &[rows, dim]);
    let f = gpu.kernels.get("embedding_gather");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&table.buf)
        .arg(dids)
        .arg(&mut out.buf)
        .arg(&dim_i)
        .arg(&rows_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems((rows * dim) as u32)) }
        .expect("embedding_gather");
    out
}

/// Scatter-add: `dtable[ids[r]] += dy[r]` (atomic; ids may repeat).
pub fn embedding_scatter_add(
    gpu: &Gpu,
    dtable: &mut DTensor,
    ids: &[usize],
    dy: &DTensor,
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
    dtable: &mut DTensor,
    dids: &cudarc::driver::CudaView<'_, u32>,
    rows: usize,
    dy: &DTensor,
    dim: usize,
) {
    let (dim_i, rows_i) = (dim as i32, rows as i32);
    let f = gpu.kernels.get("embedding_scatter_add");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut dtable.buf)
        .arg(dids)
        .arg(&dy.buf)
        .arg(&dim_i)
        .arg(&rows_i);
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
pub fn rms_norm_forward(
    gpu: &Gpu,
    x: &DTensor,
    gamma: &DTensor,
    group: usize,
    eps: f32,
) -> (DTensor, GpuRmsForward) {
    let (b, f) = (x.rows(), x.cols());
    let total_groups = b * (f / group);
    let mut out = DTensor::uninit(gpu, &[b, f]);
    let mut saved = GpuRmsForward {
        x_hat: DTensor::uninit(gpu, &[b, f]),
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
fn rms_norm_cfg(total_groups: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (total_groups as u32, 1, 1),
        block_dim: (RMSN_THREADS, 1, 1),
        // One float per warp for the cross-warp fold.
        shared_mem_bytes: (RMSN_THREADS / 32) * std::mem::size_of::<f32>() as u32,
    }
}

pub fn rms_norm_forward_into(
    gpu: &Gpu,
    x: &DTensor,
    gamma: &DTensor,
    group: usize,
    eps: f32,
    out: &mut DTensor,
    saved: &mut GpuRmsForward,
) {
    // Position-wise over the last axis, so a `[B, T, H]` caller is served as-is —
    // see `DTensor::as_2d`.
    let (b, f) = x.as_2d();
    let groups_per_row = f / group;
    let total_groups = b * groups_per_row;
    assert_eq!(out.as_2d(), (b, f), "rms_norm_forward: out shape");
    assert_eq!(saved.x_hat.as_2d(), (b, f), "rms_norm_forward: x_hat shape");
    assert_eq!(
        saved.inv_rms.len(),
        total_groups,
        "rms_norm_forward: inv_rms length"
    );
    let (gpr_i, group_i, tg_i) = (groups_per_row as i32, group as i32, total_groups as i32);
    let cfg = rms_norm_cfg(total_groups);
    let func = gpu.kernels.get("rms_norm_forward");
    let mut lb = gpu.stream.launch_builder(&func);
    lb.arg(&x.buf)
        .arg(&gamma.buf)
        .arg(&mut out.buf)
        .arg(&mut saved.x_hat.buf)
        .arg(&mut saved.inv_rms)
        .arg(&gpr_i)
        .arg(&group_i)
        .arg(&eps)
        .arg(&tg_i);
    unsafe { lb.launch(cfg) }.expect("rms_norm_forward");
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
    let mut dx = DTensor::uninit(gpu, &[dy.rows(), dy.cols()]);
    rms_norm_backward_into(gpu, dy, fwd, gamma, dgamma, group, &mut dx);
    dx
}

/// Grouped RMSNorm backward into a caller-owned `dx` — the no-allocation form of
/// [`rms_norm_backward`]. `dgamma` is accumulated into (not overwritten); `dx` is
/// written in full.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_backward_into(
    gpu: &Gpu,
    dy: &DTensor,
    fwd: &GpuRmsForward,
    gamma: &DTensor,
    dgamma: &mut DTensor,
    group: usize,
    dx: &mut DTensor,
) {
    let (b, f) = dy.as_2d();
    let groups_per_row = f / group;
    let total_groups = b * groups_per_row;
    assert_eq!(dx.as_2d(), (b, f), "rms_norm_backward: dx shape");
    let (gpr_i, group_i, tg_i) = (groups_per_row as i32, group as i32, total_groups as i32);
    let cfg = rms_norm_cfg(total_groups);
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
    unsafe { lb.launch(cfg) }.expect("rms_norm_backward");
}

/// Fused softmax + cross-entropy. Returns `(mean_loss, dlogits)` with
/// `dlogits = (softmax − onehot) / B`, matching `nn2::loss`.
pub fn softmax_cross_entropy(gpu: &Gpu, logits: &DTensor, targets: &[usize]) -> (f32, DTensor) {
    let (b, c) = (logits.rows(), logits.cols());
    assert_eq!(
        targets.len(),
        b,
        "softmax_cross_entropy — targets len != batch"
    );
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
    let losses = gpu.stream.clone_dtoh(&row_loss).expect("download row_loss");
    let loss = losses.iter().sum::<f32>() * inv_b;
    (loss, dlogits)
}

/// One AdamW step of `param` from `grad`, updating moments `m`/`v` in place.
/// `decay` toggles the decoupled weight-decay term. Mirrors `nn2::optim`.
/// Most parameter tensors one batched AdamW launch can carry. Must match
/// `ADAMW_BATCH_MAX` in `common.cu`.
pub const ADAMW_BATCH_MAX: usize = 24;

/// Kernel-argument twin of `AdamwBatch` in `common.cu`. Passed by value, so field
/// order and types must match the `.cu` definition exactly.
#[repr(C)]
#[derive(Clone, Copy)]
struct AdamwBatchArg {
    param: [u64; ADAMW_BATCH_MAX],
    grad: [u64; ADAMW_BATCH_MAX],
    m: [u64; ADAMW_BATCH_MAX],
    v: [u64; ADAMW_BATCH_MAX],
    wd: [f32; ADAMW_BATCH_MAX],
    off: [i32; ADAMW_BATCH_MAX + 1],
    n: i32,
}

// SAFETY: plain-old-data — pointers and scalars — passed by value as a kernel
// argument exactly as `cudarc` does for the scalar types.
unsafe impl cudarc::driver::DeviceRepr for AdamwBatchArg {}

/// `GPU_NO_ADAMW_BATCH=1` steps each parameter tensor with its own `adamw` launch.
pub fn no_adamw_batch() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("GPU_NO_ADAMW_BATCH").is_ok_and(|v| v != "0"))
}

/// Collects AdamW updates and issues them a batch at a time.
///
/// One `adamw` launch per parameter tensor is ~11 us of mostly launch overhead for
/// tensors far too small to fill the GPU. Layers push their tensors here instead and
/// the model flushes once, turning hundreds of launches into a handful.
///
/// This holds raw device pointers rather than `&mut DTensor` because the tensors
/// live in different layers all over the model, so no single `&mut` can span them.
/// The contract the caller must keep: every queued tensor stays alive and is not
/// reallocated until [`flush`](Self::flush), and no tensor is queued twice in a step
/// (a duplicate would race with itself).
///
/// The pointer guards `device_ptr` returns are dropped immediately rather than held.
/// Under `disable_event_tracking` (see `Gpu::new`) they are `SyncOnDrop::Record(None)`
/// — a no-op — and the buffers are owned by the model for the whole step, so there is
/// nothing for them to order.
#[derive(Default)]
pub struct AdamwQueue {
    param: Vec<u64>,
    grad: Vec<u64>,
    m: Vec<u64>,
    v: Vec<u64>,
    wd: Vec<f32>,
    len: Vec<usize>,
}

impl AdamwQueue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue one tensor's update. `decay` follows the project convention: interior
    /// projections decay, embeddings and logit heads do not.
    pub fn push(
        &mut self,
        gpu: &Gpu,
        param: &mut DTensor,
        grad: &DTensor,
        mm: &mut DTensor,
        vv: &mut DTensor,
        cfg: &AdamCfg,
        decay: bool,
    ) {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let n = param.len();
        debug_assert_eq!(grad.len(), n, "AdamwQueue: grad length != param length");
        let (pp, _g0) = param.buf.device_ptr_mut(&gpu.stream);
        let (pg, _g1) = grad.buf.device_ptr(&gpu.stream);
        let (pm, _g2) = mm.buf.device_ptr_mut(&gpu.stream);
        let (pv, _g3) = vv.buf.device_ptr_mut(&gpu.stream);
        self.param.push(pp);
        self.grad.push(pg);
        self.m.push(pm);
        self.v.push(pv);
        self.wd.push(if decay { cfg.weight_decay } else { 0.0 });
        self.len.push(n);
    }

    /// Issue every queued update, `ADAMW_BATCH_MAX` tensors per launch, then zero
    /// every queued gradient.
    ///
    /// Zeroing belongs here, not at the push site: a queued update has not read the
    /// gradient yet, so a layer that cleared its own grads at queue time would feed
    /// zeros to the kernel. Doing it here makes that ordering impossible to get wrong
    /// — and the memsets batch into the same pass.
    pub fn flush(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        let bc1 = 1.0 - cfg.beta1.powi(cfg.t as i32);
        let bc2 = 1.0 - cfg.beta2.powi(cfg.t as i32);
        let (lr, b1, b2, eps) = (cfg.lr, cfg.beta1, cfg.beta2, cfg.eps);
        let f = gpu.kernels.get("adamw_batch");

        for chunk in (0..self.param.len())
            .collect::<Vec<_>>()
            .chunks(ADAMW_BATCH_MAX)
        {
            let mut a = AdamwBatchArg {
                param: [0; ADAMW_BATCH_MAX],
                grad: [0; ADAMW_BATCH_MAX],
                m: [0; ADAMW_BATCH_MAX],
                v: [0; ADAMW_BATCH_MAX],
                wd: [0.0; ADAMW_BATCH_MAX],
                off: [0; ADAMW_BATCH_MAX + 1],
                n: chunk.len() as i32,
            };
            let mut total = 0usize;
            for (slot, &i) in chunk.iter().enumerate() {
                a.param[slot] = self.param[i];
                a.grad[slot] = self.grad[i];
                a.m[slot] = self.m[i];
                a.v[slot] = self.v[i];
                a.wd[slot] = self.wd[i];
                a.off[slot] = total as i32;
                total += self.len[i];
            }
            a.off[chunk.len()] = total as i32;
            let total_i = total as i32;
            let mut lb = gpu.stream.launch_builder(&f);
            lb.arg(&a)
                .arg(&lr)
                .arg(&b1)
                .arg(&b2)
                .arg(&eps)
                .arg(&bc1)
                .arg(&bc2)
                .arg(&total_i);
            unsafe { lb.launch(LaunchConfig::for_num_elems(total as u32)) }.expect("adamw_batch");
        }

        // Clear the gradients now that every queued update has consumed them.
        for (&g, &n) in self.grad.iter().zip(self.len.iter()) {
            // SAFETY: `g` is a live device allocation of at least `n` floats (it was
            // read from a `DTensor` of that length above and the caller guarantees it
            // is still alive), and the memset is queued on the same stream as the
            // launches that just read it, so it cannot run early.
            unsafe {
                let r = cudarc::driver::sys::cuMemsetD8Async(
                    g,
                    0,
                    n * std::mem::size_of::<f32>(),
                    gpu.stream.cu_stream(),
                );
                assert_eq!(
                    r,
                    cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                    "AdamwQueue: zeroing gradient failed: {r:?}"
                );
            }
        }
        self.clear();
    }

    fn clear(&mut self) {
        self.param.clear();
        self.grad.clear();
        self.m.clear();
        self.v.clear();
        self.wd.clear();
        self.len.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.param.is_empty()
    }
}

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

// sLSTM cell kernels (recurrent core, see gpu/slstm.rs). Each is the device
// counterpart of an inner step of `nn2::SLstm`; all state stays resident in
// `DTensor`s across the T-loop — the only host transfers are the layer's input
// and output.

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
    lb.arg(&mut xh.buf)
        .arg(&x.buf)
        .arg(&h_state.buf)
        .arg(&t_i)
        .arg(&bigt_i)
        .arg(&inp_i)
        .arg(&h_i)
        .arg(&br_i);
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
    lb.arg(&dxh.buf)
        .arg(&mut dx.buf)
        .arg(&mut dh_bptt.buf)
        .arg(&t_i)
        .arg(&bigt_i)
        .arg(&inp_i)
        .arg(&h_i)
        .arg(&br_i);
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
    unsafe { lb.launch(LaunchConfig::for_num_elems(bh as u32)) }.expect("slstm_cell_step_bwd");
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
    lb.arg(&w[0].buf)
        .arg(&w[1].buf)
        .arg(&w[2].buf)
        .arg(&w[3].buf)
        .arg(&mut wx.buf)
        .arg(&mut wh.buf)
        .arg(&inp_i)
        .arg(&h_i)
        .arg(&rows_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems((rows * 4 * h) as u32)) }.expect("slstm_pack_w");

    let f = gpu.kernels.get("slstm_pack_b");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&bias[0].buf)
        .arg(&bias[1].buf)
        .arg(&bias[2].buf)
        .arg(&bias[3].buf)
        .arg(&mut bcat.buf)
        .arg(&h_i);
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
    lb.arg(&dwx.buf)
        .arg(&dwh.buf)
        .arg(&mut dw0.buf)
        .arg(&mut dw1.buf)
        .arg(&mut dw2.buf)
        .arg(&mut dw3.buf)
        .arg(&inp_i)
        .arg(&h_i)
        .arg(&rows_i);
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
        .arg(&mut db0.buf)
        .arg(&mut db1.buf)
        .arg(&mut db2.buf)
        .arg(&mut db3.buf)
        .arg(&h_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems((4 * h) as u32)) }.expect("slstm_unpack_db");
}

/// Saved forward tensors of a fused-gate sLSTM, each a `[B, T, H]` slab (indexed
/// `(b·T + t)·H + j`) rather than one small tensor per timestep.
pub struct SlstmSlabs {
    // Stabilizer-carrying: fp32, always. `c`/`n` and their `_prev` counterparts
    // hold the exp(-m)-scaled cell and normalizer, and `i_prime`/`f_prime` ARE
    // exp(·-m). Narrowing any of them injects a multiplicative error into the
    // quantity that keeps the recurrence bounded — see `gpu::bf16`.
    pub c_prev: DTensor,
    pub n_prev: DTensor,
    pub i_prime: DTensor,
    pub f_prime: DTensor,
    pub c: DTensor,
    pub n: DTensor,
    // Plain activations: bf16 when the kernels were built for it. These are bounded
    // by construction (`zt` is a tanh, `ot` a sigmoid, `h_prev` their product over a
    // normalized ratio), written once and read once, and enter no reduction in their
    // stored form — so storage precision is free of the recurrence's error growth.
    //
    // Held as `SlabBuf` rather than `DTensor`: the width must match what the kernels
    // were compiled against (`Kernels::slab_bf16`), which is checked on construction.
    pub zt: SlabBuf,
    pub ot: SlabBuf,
    pub h_prev: SlabBuf,
}

impl SlstmSlabs {
    /// Device bytes this saved set holds. Diagnostic — the fp32 stabilizer group and
    /// the (possibly bf16) plain slabs are counted at their real widths.
    pub fn retained_bytes(&self) -> usize {
        let f32s: usize = [
            &self.c_prev,
            &self.n_prev,
            &self.i_prime,
            &self.f_prime,
            &self.c,
            &self.n,
        ]
        .iter()
        .map(|t| t.capacity() * 4)
        .sum();
        f32s + self.zt.retained_bytes() + self.ot.retained_bytes() + self.h_prev.retained_bytes()
    }
}

/// A saved slab whose element width follows the compiled kernels: bf16 when they
/// were built with `-DSLAB_BF16`, fp32 otherwise.
///
/// This exists rather than a bare `DTensor` because the kernels index these buffers
/// at a **compile-time** width. Handing a kernel built for `__nv_bfloat16` an fp32
/// buffer is not a type error anywhere — it is a silent stride mismatch that reads
/// half the tensor and writes past the end of it. Routing every allocation through
/// [`SlabBuf::new`], which reads `Kernels::slab_bf16`, makes the two agree by
/// construction.
pub enum SlabBuf {
    F32(DTensor),
    Bf16(super::BTensor),
}

impl SlabBuf {
    /// An uninitialized slab at the width the kernels expect.
    pub fn new(gpu: &Gpu, dims: &[usize]) -> Self {
        if gpu.kernels.slab_bf16 {
            SlabBuf::Bf16(super::BTensor::uninit(gpu, dims))
        } else {
            SlabBuf::F32(DTensor::uninit(gpu, dims))
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
    pub fn from_f32(gpu: &Gpu, t: DTensor) -> Self {
        if !gpu.kernels.slab_bf16 {
            return SlabBuf::F32(t);
        }
        let mut b = super::BTensor::uninit(gpu, t.dims());
        b.store(gpu, &t);
        SlabBuf::Bf16(b)
    }

    /// An fp32 view of this slab, for a consumer that cannot read bf16 — chiefly
    /// cuBLAS, which has no bf16 operand on these GEMMs.
    ///
    /// `scratch` receives the widened copy on the bf16 path and is left untouched on
    /// the fp32 path, where the slab is already what the caller wants. (A `Cow` would
    /// be the natural shape, but `DTensor` is deliberately not `Clone`.)
    pub fn as_f32<'a>(&'a self, gpu: &Gpu, scratch: &'a mut DTensor) -> &'a DTensor {
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
    lb.arg(&mut g.buf).arg(&gh.buf).arg(&bcat.buf);
    push_slab!(lb, slabs.h_prev);
    lb.arg(&mut c_state.buf)
        .arg(&mut n_state.buf)
        .arg(&mut m_state.buf)
        .arg(&mut h_state.buf)
        .arg(&mut slabs.c_prev.buf)
        .arg(&mut slabs.n_prev.buf);
    push_slab!(lb, slabs.zt);
    push_slab!(lb, slabs.ot);
    lb.arg(&mut slabs.i_prime.buf)
        .arg(&mut slabs.f_prime.buf)
        .arg(&mut slabs.c.buf)
        .arg(&mut slabs.n.buf)
        .arg(&mut out.buf)
        .arg(&t_i)
        .arg(&bigt_i)
        .arg(&h_i)
        .arg(&bh_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(bh as u32)) }.expect("slstm_step_fused");
}

/// Whether the fused kernels stage their `Wh` slice (and the `h` they multiply) in
/// **bf16** inside shared memory, accumulating in fp32.
///
/// This is FlashRNN's arrangement — `R_shared` is `FLASHRNN_DTYPE_R` (bf16) while
/// `ACC_DTYPE` stays `float`. Unlike the earlier standalone `SLSTM_BF16` path, the
/// conversion happens where the data already lives (shared memory / registers), so
/// it costs a couple of ALU ops rather than a separate kernel and a global
/// round-trip per timestep — which is exactly why that version lost at H=512.
///
/// Halving the weights' shared footprint also lets a block own more units, which
/// is what decides whether the fused path exists at all above H=640.
///
/// **On by default**; `SLSTM_NO_BF16=1` forces fp32 staging, which is the A/B
/// baseline.
///
/// At H=512 this was neutral (5.66ms against 5.60ms) and the path was off: the grid
/// there is pinned by the SM count, not by shared memory, so `geometry` picks one
/// block per SM under either dtype and the freed shared memory buys nothing a
/// cooperative grid can use.
///
/// At H=768 — the backbone's width — the grid becomes shared-memory-bound instead,
/// which is the case that paragraph predicted. A block's fp32 slice needs
/// `H*4*units*4` bytes, so `max_units` falls to 8 and the grid would need 96 blocks
/// of an 84-SM card: the cooperative launch cannot be co-resident and the fused path
/// **declines entirely**, dropping the cell to the per-step loop. bf16 staging halves
/// the slice, 10 units fit per block, 77 blocks suffice, and the path exists again.
/// Measured on the sLSTM cell at B=1, T=1024: 20.50 -> 9.56ms; on a whole 4096-word
/// training step, 1017 -> 715ms.
///
/// The cost is an 8-bit mantissa on the staged operand: parity against the per-step
/// loop moves from ~1e-6 to ~1e-3 relative (see `examples/fused_bf16_parity.rs`),
/// which is the storage dtype's noise floor, not a defect. The accumulator stays
/// fp32 throughout.
pub fn fused_bf16_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SLSTM_NO_BF16").is_err())
}

/// Launch geometry for [`slstm_fused_time`]: `(blocks, threads, cols_per_block,
/// shared_bytes)`, or `None` when the shape does not fit the kernel's contract.
///
/// The constraint that drives everything is shared memory: each block stages an
/// `[H, cols_per_block]` fp32 slice of `Wh`, so `H * cols_per_block * 4` must fit
/// the device's opt-in limit. We pick the *smallest* block count that satisfies
/// that (fewest blocks => cheapest `grid.sync()`), then check the whole grid can
/// actually be co-resident — a cooperative launch deadlocks if it cannot.
pub fn slstm_fused_time_geometry(
    gpu: &Gpu,
    h: usize,
    b: usize,
) -> Option<(usize, usize, usize, usize)> {
    let h4 = 4 * h;
    let threads = 256usize;
    // Leave a little headroom under the opt-in cap for the driver's own use.
    let smem_cap = gpu.max_shared_optin.saturating_sub(1024);
    // `cols_per_block` counts HIDDEN UNITS; each carries four Wh columns, so a block
    // stages `[H, 4*units]` weights plus a `[B, 4*units]` fp32 gate scratch. Under
    // bf16 the weights are 2 bytes instead of 4 and a `[B, H]` bf16 mirror of h is
    // added; the gate scratch stays fp32 either way (it feeds the recurrence).
    let wbytes = if fused_bf16_enabled() { 2 } else { 4 };
    let h_mirror = if fused_bf16_enabled() { b * h * 2 } else { 0 };
    let bytes_per_unit = |units: usize| h * 4 * units * wbytes + b * 4 * units * 4 + h_mirror;
    let max_units = {
        let per_unit = h * 4 * wbytes + b * 4 * 4;
        smem_cap.saturating_sub(h_mirror) / per_unit
    };
    if max_units == 0 {
        return None; // one unit's slice does not fit: H is too large for this path
    }
    // Spread over as many SMs as the grid may use, not as few blocks as possible.
    // The fewest-blocks choice minimises `grid.sync()` cost but leaves most of the
    // device idle — at H=512 it picks 42 blocks of an 84-SM card, and the gate
    // reduction (not the sync) is what dominates, so halving the parallelism there
    // costs far more than the extra sync saves.
    let min_blocks = h.div_ceil(max_units);
    let blocks = gpu.sm_count.max(min_blocks).min(h);
    let cols_per_block = h.div_ceil(blocks);
    // Recompute: rounding the slice up may need fewer blocks than requested.
    let blocks = h.div_ceil(cols_per_block);
    let shared_bytes = bytes_per_unit(cols_per_block);
    if shared_bytes > smem_cap {
        return None;
    }
    // A cooperative launch requires the WHOLE grid to be co-resident: `grid.sync()`
    // blocks until every block arrives, so a block that has not been scheduled yet
    // deadlocks the kernel. With one block per SM at this shared-memory footprint
    // (98 KB of a ~100 KB budget leaves room for exactly one), the grid must fit in
    // the SM count. Declining here is what keeps large H on the per-step path
    // instead of hanging the device.
    if blocks > gpu.sm_count {
        return None;
    }
    // The pointwise phase is grid-strided over B*H, so any grid covers it; but the
    // gate phase needs every column owned by some block, which `blocks` guarantees.
    debug_assert!(blocks * cols_per_block >= h);
    let _ = h4;
    let _ = b;
    Some((blocks, threads, cols_per_block, shared_bytes))
}

/// Launch geometry for [`slstm_fused_time_bwd`], mirroring
/// [`slstm_fused_time_geometry`] but sized for the backward's staging: a block
/// owning `units` output units holds `units` whole **rows** of `Wh` (`4H` each),
/// where the forward held `4*units` columns of `H`. Same total per unit, different
/// axis — see the kernel for why the ownership transposes.
pub fn slstm_fused_time_bwd_geometry(
    gpu: &Gpu,
    h: usize,
    b: usize,
) -> Option<(usize, usize, usize, usize)> {
    let h4 = 4 * h;
    let threads = 256usize;
    let smem_cap = gpu.max_shared_optin.saturating_sub(1024);
    // One staged Wh row, 2 bytes per entry under bf16 staging.
    let per_unit = h4 * if fused_bf16_enabled() { 2 } else { 4 };
    // Plus one fp32 [B, 4H] copy of the step's gate deltas, shared by every warp in
    // the block (see the kernel's `dgsh`). Fixed cost, independent of the unit count.
    let dg_bytes = b * h4 * 4;
    let max_units = smem_cap.saturating_sub(dg_bytes) / per_unit;
    if max_units == 0 {
        return None;
    }
    let min_blocks = h.div_ceil(max_units);
    let blocks = gpu.sm_count.max(min_blocks).min(h);
    let units_per_block = h.div_ceil(blocks);
    let blocks = h.div_ceil(units_per_block);
    let shared_bytes = units_per_block * per_unit + dg_bytes;
    if shared_bytes > smem_cap || blocks > gpu.sm_count {
        return None;
    }
    Some((blocks, threads, units_per_block, shared_bytes))
}

/// The whole backward T-loop as **one cooperative launch**: see
/// `slstm_fused_time_bwd` in `kernels.rs`. Writes the gate deltas into `g` exactly
/// as the per-step path does, so the post-loop dWx/dWh/db GEMMs are unaffected.
///
/// Returns `false` (having launched nothing) when unavailable, so the caller falls
/// back to the per-step loop.
#[allow(clippy::too_many_arguments)]
pub fn slstm_fused_time_bwd(
    gpu: &Gpu,
    wh: &DTensor,
    dy: &DTensor,
    g: &mut DTensor,
    dgates_all: &mut DTensor,
    dh_recur: &mut DTensor,
    dc_recur: &mut DTensor,
    dn_recur: &mut DTensor,
    slabs: &SlstmSlabs,
    t: usize,
) -> bool {
    if !gpu.kernels.has_coop {
        return false;
    }
    let (b, h) = (dc_recur.rows(), dc_recur.cols());
    let Some((blocks, threads, units_per_block, shared_bytes)) =
        slstm_fused_time_bwd_geometry(gpu, h, b)
    else {
        return false;
    };
    let (t_i, h_i, b_i, upb_i) = (t as i32, h as i32, b as i32, units_per_block as i32);
    let f = gpu
        .kernels
        .specialized(&gpu.context, "slstm_fused_time_bwd", h, b, fused_bf16_enabled());
    // The unspecialized module is built without -DFUSED_BF16, so it would read the
    // staged rows at the wrong stride. Under bf16 there is no fallback to take.
    let f = match f {
        Some(f) => f,
        None if !fused_bf16_enabled() => gpu.kernels.get("slstm_fused_time_bwd"),
        None => return false,
    };
    if let Err(e) = f.set_attribute(
        cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared_bytes as i32,
    ) {
        eprintln!("slstm_fused_time_bwd: shared-memory opt-in failed: {e:?}");
        return false;
    }
    let cfg = LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: shared_bytes as u32,
    };
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&wh.buf)
        .arg(&dy.buf)
        .arg(&mut g.buf)
        .arg(&mut dgates_all.buf)
        .arg(&mut dh_recur.buf)
        .arg(&mut dc_recur.buf)
        .arg(&mut dn_recur.buf);
    push_slab_ref!(lb, slabs.ot);
    lb.arg(&slabs.c.buf)
        .arg(&slabs.n.buf)
        .arg(&slabs.c_prev.buf)
        .arg(&slabs.n_prev.buf);
    push_slab_ref!(lb, slabs.zt);
    lb.arg(&slabs.i_prime.buf)
        .arg(&slabs.f_prime.buf)
        .arg(&t_i)
        .arg(&h_i)
        .arg(&b_i)
        .arg(&upb_i);
    // SAFETY: geometry guarantees a co-resident grid and a fitting shared slice.
    match unsafe { lb.launch_cooperative(cfg) } {
        Ok(_) => true,
        Err(e) => {
            eprintln!("slstm_fused_time_bwd: cooperative launch failed: {e:?}");
            false
        }
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
    wh: &DTensor,
    g: &mut DTensor,
    bcat: &DTensor,
    c_state: &mut DTensor,
    n_state: &mut DTensor,
    m_state: &mut DTensor,
    h_state: &mut DTensor,
    slabs: &mut SlstmSlabs,
    out: &mut DTensor,
    t: usize,
) -> bool {
    if !gpu.kernels.has_coop {
        return false;
    }
    let (b, h) = (c_state.rows(), c_state.cols());
    let Some((blocks, threads, cols_per_block, shared_bytes)) =
        slstm_fused_time_geometry(gpu, h, b)
    else {
        return false;
    };
    let (t_i, h_i, b_i, cpb_i) = (t as i32, h as i32, b as i32, cols_per_block as i32);
    // Prefer the (H, B)-specialized build. The generic module is only a valid
    // fallback when bf16 staging is OFF: it is compiled without -DFUSED_BF16, so it
    // stages fp32 and would read past a shared allocation that `geometry` sized for
    // bf16. With bf16 on and no specialized build available, decline instead and let
    // the caller take the per-step path.
    let f =
        match gpu
            .kernels
            .specialized(&gpu.context, "slstm_fused_time", h, b, fused_bf16_enabled())
        {
            Some(f) => f,
            None if !fused_bf16_enabled() => gpu.kernels.get("slstm_fused_time"),
            None => return false,
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
    let cfg = LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: shared_bytes as u32,
    };
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&wh.buf).arg(&mut g.buf).arg(&bcat.buf);
    push_slab!(lb, slabs.h_prev);
    lb.arg(&mut c_state.buf)
        .arg(&mut n_state.buf)
        .arg(&mut m_state.buf)
        .arg(&mut h_state.buf)
        .arg(&mut slabs.c_prev.buf)
        .arg(&mut slabs.n_prev.buf);
    push_slab!(lb, slabs.zt);
    push_slab!(lb, slabs.ot);
    lb.arg(&mut slabs.i_prime.buf)
        .arg(&mut slabs.f_prime.buf)
        .arg(&mut slabs.c.buf)
        .arg(&mut slabs.n.buf)
        .arg(&mut out.buf)
        .arg(&t_i)
        .arg(&h_i)
        .arg(&b_i)
        .arg(&cpb_i);
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
/// * `d_gates_flat` — `[B, 4H]` this step's deltas, contiguous, for the dh GEMM.
/// * `d_h_recur` — `[B, H]` grad arriving from step `t+1` through `h`.
/// * `slabs` — the forward's saved activations (`o`, `c`, `n`, `z`, `i'`, `f'`).
/// * `d_c_recur` / `d_n_recur` — `[B, H]` cell and normalizer grads, carried
///   back a step in place.
/// * `t` — the timestep being differentiated.
#[allow(clippy::too_many_arguments)]
pub fn slstm_step_fused_bwd(
    gpu: &Gpu,
    d_out: &DTensor,
    gates: &mut DTensor,
    d_gates_flat: &mut DTensor,
    d_h_recur: &DTensor,
    slabs: &SlstmSlabs,
    d_c_recur: &mut DTensor,
    d_n_recur: &mut DTensor,
    t: usize,
) {
    let (b, h) = (d_c_recur.rows(), d_c_recur.cols());
    let bh = b * h;
    let big_t = d_out.shape[1];
    let (t_i, bigt_i, h_i, bh_i) = (t as i32, big_t as i32, h as i32, bh as i32);
    let f = gpu.kernels.get("slstm_step_fused_bwd");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&d_out.buf)
        .arg(&mut gates.buf)
        .arg(&mut d_gates_flat.buf)
        .arg(&d_h_recur.buf);
    push_slab_ref!(lb, slabs.ot);
    lb.arg(&slabs.c.buf)
        .arg(&slabs.n.buf)
        .arg(&slabs.c_prev.buf)
        .arg(&slabs.n_prev.buf);
    push_slab_ref!(lb, slabs.zt);
    lb.arg(&slabs.i_prime.buf)
        .arg(&slabs.f_prime.buf)
        .arg(&mut d_c_recur.buf)
        .arg(&mut d_n_recur.buf)
        .arg(&t_i)
        .arg(&bigt_i)
        .arg(&h_i)
        .arg(&bh_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(bh as u32)) }.expect("slstm_step_fused_bwd");
}

// Residual block / SwiGLU kernels (see gpu/block.rs).

/// Elementwise `out = a + b` (fresh allocation). Used for residual adds and the
/// grad accumulations that are plain sums.
pub fn add(gpu: &Gpu, a: &DTensor, b: &DTensor) -> DTensor {
    let mut out = DTensor::uninit(gpu, a.dims());
    add_into(gpu, a, b, &mut out);
    out
}

/// In-place `acc += b`.
///
/// The running sums in mLSTM's backward (`dkc`, `dqc`, `dxf`, …) were written as
/// `x = add(&x, &term)`, which allocates a buffer per term and drops the previous
/// one. This accumulates into `acc` instead, so a sum of k terms costs no
/// allocations at all.
pub fn add_assign(gpu: &Gpu, acc: &mut DTensor, b: &DTensor) {
    let n = acc.len();
    assert_eq!(n, b.len(), "add_assign: length mismatch");
    let n_i = n as i32;
    let f = gpu.kernels.get("add_assign");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut acc.buf).arg(&b.buf).arg(&n_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(n as u32)) }.expect("add_assign");
}

/// Elementwise `out = a + b` into a caller-owned buffer. The allocating [`add`]
/// is a thin wrapper over this; layers that own their buffers call this directly
/// so a residual add costs no allocation.
///
/// `out` may alias `a` or `b` (the kernel reads element `i` before writing it).
pub fn add_into(gpu: &Gpu, a: &DTensor, b: &DTensor, out: &mut DTensor) {
    let n = a.len();
    assert_eq!(n, b.len(), "add: length mismatch");
    assert_eq!(n, out.len(), "add: output length mismatch");
    let n_i = n as i32;
    let f = gpu.kernels.get("add");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut out.buf).arg(&a.buf).arg(&b.buf).arg(&n_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(n as u32)) }.expect("add");
}

/// SwiGLU forward: returns `(gate_act = SiLU(gate_pre), mixed = gate_act ⊙ value)`.
pub fn swiglu_forward(gpu: &Gpu, gate_pre: &DTensor, value: &DTensor) -> (DTensor, DTensor) {
    let mut gate_act = DTensor::uninit(gpu, gate_pre.dims());
    let mut mixed = DTensor::uninit(gpu, gate_pre.dims());
    swiglu_forward_into(gpu, gate_pre, value, &mut gate_act, &mut mixed);
    (gate_act, mixed)
}

/// SwiGLU forward into caller-owned buffers — the no-allocation form of
/// [`swiglu_forward`]. Both outputs are written in full.
pub fn swiglu_forward_into(
    gpu: &Gpu,
    gate_pre: &DTensor,
    value: &DTensor,
    gate_act: &mut DTensor,
    mixed: &mut DTensor,
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
    unsafe { lb.launch(LaunchConfig::for_num_elems(n as u32)) }.expect("swiglu_forward");
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
    let mut d_gate = DTensor::uninit(gpu, d_mixed.dims());
    let mut d_value = DTensor::uninit(gpu, d_mixed.dims());
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
    d_mixed: &DTensor,
    gate_act: &DTensor,
    value: &DTensor,
    gate_pre: &DTensor,
    d_gate: &mut DTensor,
    d_value: &mut DTensor,
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
    unsafe { lb.launch(LaunchConfig::for_num_elems(n as u32)) }.expect("swiglu_backward");
}

// mLSTM parallel/chunkwise core (see gpu/mlstm.rs).

/// Position-major `[N, H*W]` (N=B·T) → head-major `[B*H, T, W]`.
pub fn head_gather(gpu: &Gpu, x: &DTensor, b: usize, h: usize, t: usize, w: usize) -> DTensor {
    let mut out = DTensor::uninit(gpu, &[b * h, t, w]);
    let (bi, hi, ti, wi) = (b as i32, h as i32, t as i32, w as i32);
    let f = gpu.kernels.get("head_gather");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&x.buf)
        .arg(&mut out.buf)
        .arg(&bi)
        .arg(&hi)
        .arg(&ti)
        .arg(&wi);
    unsafe { lb.launch(LaunchConfig::for_num_elems((b * h * t * w) as u32)) }.expect("head_gather");
    out
}

/// Head-major `[B*H, T, W]` → position-major `[N, H*W]` (inverse of `head_gather`).
pub fn head_scatter(gpu: &Gpu, x: &DTensor, b: usize, h: usize, t: usize, w: usize) -> DTensor {
    let mut out = DTensor::uninit(gpu, &[b * t, h * w]);
    head_scatter_into(gpu, x, b, h, t, w, &mut out);
    out
}

/// Head-major `[B*H, T, W]` → position-major `[N, H*W]` into a caller-owned
/// buffer — the no-allocation form of [`head_scatter`].
pub fn head_scatter_into(
    gpu: &Gpu,
    x: &DTensor,
    b: usize,
    h: usize,
    t: usize,
    w: usize,
    out: &mut DTensor,
) {
    assert_eq!(out.len(), b * t * h * w, "head_scatter: output size");
    let (bi, hi, ti, wi) = (b as i32, h as i32, t as i32, w as i32);
    let f = gpu.kernels.get("head_scatter");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&x.buf)
        .arg(&mut out.buf)
        .arg(&bi)
        .arg(&hi)
        .arg(&ti)
        .arg(&wi);
    unsafe { lb.launch(LaunchConfig::for_num_elems((b * h * t * w) as u32)) }
        .expect("head_scatter");
}

/// [`head_gather`] writing straight into a [`SlabBuf`], narrowing as it goes.
///
/// The conversion has to happen *here* rather than on the result of a normal
/// `head_gather`. Materializing the fp32 tensor first and then narrowing it pays
/// for the wide allocation regardless — and cudarc frees through an async pool that
/// retains blocks, so the fp32 buffer stays resident afterwards and the pair costs
/// **more** than fp32 alone. That was measured: the naive ordering made a training
/// step 32 MB larger rather than smaller. Writing narrow from the start is what
/// removes the wide buffer from the working set.
pub fn head_gather_slab(gpu: &Gpu, x: &DTensor, b: usize, h: usize, t: usize, w: usize) -> SlabBuf {
    let mut out = SlabBuf::new(gpu, &[b * h, t, w]);
    let (bi, hi, ti, wi) = (b as i32, h as i32, t as i32, w as i32);
    let name = match out {
        SlabBuf::F32(_) => "head_gather",
        SlabBuf::Bf16(_) => "head_gather_slab",
    };
    let f = gpu.kernels.get(name);
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&x.buf);
    push_slab!(lb, out);
    lb.arg(&bi).arg(&hi).arg(&ti).arg(&wi);
    unsafe { lb.launch(LaunchConfig::for_num_elems((b * h * t * w) as u32)) }
        .expect("head_gather_slab");
    out
}

/// [`head_scatter_into`] reading a [`SlabBuf`] source, converting on the fly.
///
/// The mLSTM's `ytil` is stored bf16; widening it into a temporary before the
/// scatter would allocate the very fp32 buffer the narrow storage exists to avoid,
/// so the kernel reads it narrow instead. Output is fp32 either way.
pub fn head_scatter_slab_into(
    gpu: &Gpu,
    x: &SlabBuf,
    b: usize,
    h: usize,
    t: usize,
    w: usize,
    out: &mut DTensor,
) {
    assert_eq!(out.len(), b * t * h * w, "head_scatter_slab: output size");
    let (bi, hi, ti, wi) = (b as i32, h as i32, t as i32, w as i32);
    // The fp32 variant is the same kernel without the convert; dispatch on the
    // slab's own width so this works on both paths.
    let name = match x {
        SlabBuf::F32(_) => "head_scatter",
        SlabBuf::Bf16(_) => "head_scatter_slab",
    };
    let f = gpu.kernels.get(name);
    let mut lb = gpu.stream.launch_builder(&f);
    push_slab_ref!(lb, *x);
    lb.arg(&mut out.buf).arg(&bi).arg(&hi).arg(&ti).arg(&wi);
    unsafe { lb.launch(LaunchConfig::for_num_elems((b * h * t * w) as u32)) }
        .expect("head_scatter_slab");
}

/// Inclusive cumsum of logσ along T, per row of `f` `[BH, T]` → `fc` `[BH, T]`.
pub fn cumsum_logsig(gpu: &Gpu, f: &DTensor) -> DTensor {
    let (bh, t) = (f.rows(), f.cols());
    let mut fc = DTensor::uninit(gpu, &[bh, t]);
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
    unsafe { lb.launch(LaunchConfig::for_num_elems(bh as u32)) }.expect("cumsum_logsig");
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

/// Per-row stabilizer `m` `[BH, T]` = `max(max_{j≤t}(fc_t−fc_j+ig_j), fc_t+m_prev)`.
pub fn mlstm_rowmax_m(gpu: &Gpu, fc: &DTensor, ig: &DTensor, m_prev: &DTensor) -> DTensor {
    let (bh, t) = (fc.rows(), fc.cols());
    let mut m = DTensor::uninit(gpu, &[bh, t]);
    let (ti, bht) = (t as i32, (bh * t) as i32);
    let f = gpu.kernels.get("mlstm_rowmax_m");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&fc.buf)
        .arg(&ig.buf)
        .arg(&m_prev.buf)
        .arg(&mut m.buf)
        .arg(&ti)
        .arg(&bht);
    unsafe { lb.launch(row_block_cfg(bh * t, t)) }.expect("mlstm_rowmax_m");
    m
}

/// `DS = D̄ ⊙ S` plus the decay weights `D̄`, row normalizer `qn` and `ψ`. `s` is
/// `[BH, T, T]`; returns `(Dbar, DS, qn, psi)`. `D̄` is retained for the backward.
pub fn mlstm_ds(
    gpu: &Gpu,
    s: &DTensor,
    fc: &DTensor,
    ig: &DTensor,
    m: &DTensor,
) -> (DTensor, DTensor, DTensor, DTensor) {
    let (bh, t) = (fc.rows(), fc.cols());
    let mut dbar = DTensor::uninit(gpu, &[bh, t, t]);
    let mut ds = DTensor::uninit(gpu, &[bh, t, t]);
    let mut qn = DTensor::uninit(gpu, &[bh, t]);
    let mut psi = DTensor::uninit(gpu, &[bh, t]);
    let (ti, bht) = (t as i32, (bh * t) as i32);
    let f = gpu.kernels.get("mlstm_ds");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&s.buf)
        .arg(&fc.buf)
        .arg(&ig.buf)
        .arg(&m.buf)
        .arg(&mut dbar.buf)
        .arg(&mut ds.buf)
        .arg(&mut qn.buf)
        .arg(&mut psi.buf)
        .arg(&ti)
        .arg(&bht);
    unsafe { lb.launch(row_block_cfg(bh * t, t)) }.expect("mlstm_ds");
    (dbar, ds, qn, psi)
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
    x: &DTensor,
    bh: usize,
    t: usize,
    w: usize,
    c0: usize,
    len: usize,
) -> DTensor {
    assert_eq!(
        bh * t * w,
        x.len(),
        "slice_t_as: dims do not cover the tensor"
    );
    slice_t_inner(gpu, x, bh, t, w, c0, len)
}

/// Extract the T-range `[c0, c0+len)` of a head-major `[BH, T, W]` tensor.
pub fn slice_t(gpu: &Gpu, x: &DTensor, c0: usize, len: usize) -> DTensor {
    let (bh, t, w) = (x.shape[0], x.shape[1], x.shape[2]);
    slice_t_inner(gpu, x, bh, t, w, c0, len)
}

fn slice_t_inner(
    gpu: &Gpu,
    x: &DTensor,
    bh: usize,
    t: usize,
    w: usize,
    c0: usize,
    len: usize,
) -> DTensor {
    assert!(
        c0 + len <= t,
        "slice_t: chunk [{c0}, {}) out of range T={t}",
        c0 + len
    );
    let mut out = DTensor::uninit(gpu, &[bh, len, w]);
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
    unsafe { lb.launch(LaunchConfig::for_num_elems(total as u32)) }.expect("slice_t");
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
    srcs: &[(&DTensor, usize)],
    bh: usize,
    t: usize,
    c0: usize,
    len: usize,
) -> Vec<DTensor> {
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

    let mut outs: Vec<DTensor> = srcs
        .iter()
        .map(|&(_, w)| DTensor::uninit(gpu, &[bh, len, w]))
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
        unsafe { lb.launch(LaunchConfig::for_num_elems(total as u32)) }.expect("slice_t_batch");
    }
    outs
}

/// Write several chunks back into their full tensors in one launch — the
/// [`slice_t_batch`] inverse. Each entry is `(dst, src, W)`, with `W` explicit for
/// the same reason as in `slice_t_batch`: a source may be rank-2.
pub fn unslice_t_batch(
    gpu: &Gpu,
    pairs: &mut [(&mut DTensor, &DTensor, usize)],
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
    unsafe { lb.launch(LaunchConfig::for_num_elems(total as u32)) }.expect("unslice_t_batch");
}

/// Write `src` `[BH, len, W]` back into the T-range `[c0, c0+len)` of `dst`
/// `[BH, T, W]`. Chunks partition T, so this is a store, not an accumulate.
/// [`unslice_t`] with the source's `[BH, len, W]` interpretation given explicitly,
/// so a rank-2 source needs no reshape (and hence no `dup`). See [`slice_t_as`].
pub fn unslice_t_as(gpu: &Gpu, dst: &mut DTensor, src: &DTensor, len: usize, c0: usize) {
    let (bh, t, w) = (dst.shape[0], dst.shape[1], dst.shape[2]);
    unslice_t_inner(gpu, dst, src, bh, t, w, len, c0);
}

pub fn unslice_t(gpu: &Gpu, dst: &mut DTensor, src: &DTensor, c0: usize) {
    let (bh, t, w) = (dst.shape[0], dst.shape[1], dst.shape[2]);
    let len = src.shape[1];
    unslice_t_inner(gpu, dst, src, bh, t, w, len, c0);
}

fn unslice_t_inner(
    gpu: &Gpu,
    dst: &mut DTensor,
    src: &DTensor,
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
    unsafe { lb.launch(LaunchConfig::for_num_elems(total as u32)) }.expect("unslice_t");
}

/// The chunk's two inter-chunk decay weight vectors, both `[BH, L]`:
/// `b_t = exp(fc_t + m_prev − m_t)` (carried state → row t) and
/// `a_j = exp(fc_last − fc_j + ig_j − m_last)` (row j → outgoing state).
pub fn mlstm_chunk_ab(
    gpu: &Gpu,
    fc: &DTensor,
    ig: &DTensor,
    m: &DTensor,
    m_prev: &DTensor,
) -> (DTensor, DTensor) {
    let (bh, l) = (fc.rows(), fc.cols());
    let mut b = DTensor::uninit(gpu, &[bh, l]);
    let mut a = DTensor::uninit(gpu, &[bh, l]);
    let (li, bhl) = (l as i32, (bh * l) as i32);
    let f = gpu.kernels.get("mlstm_chunk_ab");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&fc.buf)
        .arg(&ig.buf)
        .arg(&m.buf)
        .arg(&m_prev.buf)
        .arg(&mut b.buf)
        .arg(&mut a.buf)
        .arg(&li)
        .arg(&bhl);
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
    lb.arg(&mut out.buf)
        .arg(&x.buf)
        .arg(&s.buf)
        .arg(&wi)
        .arg(&total_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(total as u32)) }.expect("mul_rows");
    out
}

/// `out += s ⊙ x` broadcast over the trailing width `w`. Doubles as the per-head
/// state decay (`C += g[head]·C_prev`, with `w` = the head's element count).
pub fn mul_rows_add(gpu: &Gpu, out: &mut DTensor, x: &DTensor, s: &DTensor, w: usize) {
    let total = x.len();
    assert_eq!(total, out.len(), "mul_rows_add: out/x length mismatch");
    assert_eq!(
        total,
        s.len() * w,
        "mul_rows_add: {total} != {} · {w}",
        s.len()
    );
    let (wi, total_i) = (w as i32, total as i32);
    let f = gpu.kernels.get("mul_rows_add");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut out.buf)
        .arg(&x.buf)
        .arg(&s.buf)
        .arg(&wi)
        .arg(&total_i);
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
    lb.arg(&mut out.buf)
        .arg(&x.buf)
        .arg(&y.buf)
        .arg(&wi)
        .arg(&ri);
    unsafe { lb.launch(row_block_cfg(r, w)) }.expect("row_dot_add");
}

/// `out[g] += Σ_e x[g,e]·y[g,e]` for `[G, E]` operands (`out` is `[G]`) — the
/// per-head reduction behind `dg`.
pub fn group_dot_add(gpu: &Gpu, out: &mut DTensor, x: &DTensor, y: &DTensor) {
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
    db: &DTensor,
    da: &DTensor,
    b: &DTensor,
    a: &DTensor,
    dfc: &mut DTensor,
    dig: &mut DTensor,
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
    unsafe { lb.launch(LaunchConfig::for_num_elems((bh * l) as u32)) }.expect("mlstm_chunk_ab_bwd");
}

/// Copy rows of `src` into arbitrary rows of `dst`: `dst[row_ids[i]] = src[i]`.
/// The inverse (pulling those rows back out) is [`embedding_gather`] with the same
/// row ids, treating the matrix as the "table".
pub fn scatter_rows(gpu: &Gpu, dst: &mut DTensor, src: &DTensor, row_ids: &[usize]) {
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
    dst: &mut DTensor,
    src: &DTensor,
    ids: &cudarc::driver::CudaView<'_, u32>,
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
    unsafe { lb.launch(LaunchConfig::for_num_elems((rows * dim) as u32)) }.expect("scatter_rows");
}

/// Masked softmax cross-entropy (the hierarchical decode loss). `mask[r] == false`
/// marks a padding row: zero loss, zero grad. Normalized by the number of valid
/// rows. Returns `(mean_loss, dlogits)`.
pub fn masked_softmax_cross_entropy(
    gpu: &Gpu,
    logits: &DTensor,
    targets: &[usize],
    mask: &[bool],
) -> (f32, DTensor) {
    let num_valid = mask.iter().filter(|&&m| m).count().max(1) as f32;
    masked_softmax_cross_entropy_scaled(gpu, logits, targets, mask, 1.0 / num_valid)
}

/// Masked CE with an explicit `1/N` normalizer. When one window is split into
/// several rectangles (the length-grouped word batches), every group must be
/// scaled by the window's TOTAL valid-row count — not its own — so the summed
/// losses and gradients equal the single-rectangle result.
pub fn masked_softmax_cross_entropy_scaled(
    gpu: &Gpu,
    logits: &DTensor,
    targets: &[usize],
    mask: &[bool],
    inv: f32,
) -> (f32, DTensor) {
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
    logits: &DTensor,
    dtargets: &cudarc::driver::CudaView<'_, u32>,
    dmask: &cudarc::driver::CudaView<'_, u32>,
    inv: f32,
) -> (f32, DTensor) {
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
    logits: &DTensor,
    dtargets: &cudarc::driver::CudaView<'_, u32>,
    dmask: &cudarc::driver::CudaView<'_, u32>,
    inv: f32,
    acc: Option<&mut CudaSlice<f32>>,
) -> (f32, DTensor) {
    let (r, c) = (logits.rows(), logits.cols());
    let mut dlogits = DTensor::uninit(gpu, &[r, c]);
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
        LaunchConfig::for_num_elems(r as u32)
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

/// o-gate backward → `(do_pre, d_yhat)` from `d_hconcat` and saved `o`/`yhat`.
pub fn ogate_bwd(
    gpu: &Gpu,
    d_hconcat: &DTensor,
    o: &DTensor,
    yhat: &DTensor,
) -> (DTensor, DTensor) {
    let n = d_hconcat.len();
    let n_i = n as i32;
    let mut do_pre = DTensor::uninit(gpu, d_hconcat.dims());
    let mut d_yhat = DTensor::uninit(gpu, d_hconcat.dims());
    let f = gpu.kernels.get("ogate_bwd");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&d_hconcat.buf)
        .arg(&o.buf)
        .arg(&yhat.buf)
        .arg(&mut do_pre.buf)
        .arg(&mut d_yhat.buf)
        .arg(&n_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(n as u32)) }.expect("ogate_bwd");
    (do_pre, d_yhat)
}

/// Backward of `ytil = num/ψ` → `(d_num, d_qn)`. `num` `[BH,T,dhv]`, `psi`/`qn` `[BH,T]`.
pub fn div_rows_bwd(
    gpu: &Gpu,
    d_ytil: &DTensor,
    num: &DTensor,
    psi: &DTensor,
    qn: &DTensor,
    m: &DTensor,
    dhv: usize,
) -> (DTensor, DTensor) {
    let (bh, t) = (psi.rows(), psi.cols());
    let mut d_num = DTensor::uninit(gpu, num.dims());
    let mut d_qn = DTensor::uninit(gpu, &[bh, t]);
    let (dhv_i, bht) = (dhv as i32, (bh * t) as i32);
    let f = gpu.kernels.get("div_rows_bwd");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&d_ytil.buf)
        .arg(&num.buf)
        .arg(&psi.buf)
        .arg(&qn.buf)
        .arg(&m.buf)
        .arg(&mut d_num.buf)
        .arg(&mut d_qn.buf)
        .arg(&dhv_i)
        .arg(&bht);
    unsafe { lb.launch(row_block_cfg(bh * t, dhv)) }.expect("div_rows_bwd");
    (d_num, d_qn)
}

/// Backward of `DS = D̄⊙S` + qn-sum → `(dS, P)`. `dds_num = d_num·Vᵀ` (num path),
/// `d_qn` the qn path; `dbar`/`ds` the saved forward weights.
pub fn mlstm_ds_bwd(
    gpu: &Gpu,
    dds_num: &DTensor,
    d_qn: &DTensor,
    dbar: &DTensor,
    ds: &DTensor,
) -> (DTensor, DTensor) {
    let (bh, t, _) = (dds_num.shape[0], dds_num.shape[1], dds_num.shape[2]);
    let total = bh * t * t;
    let mut d_s = DTensor::uninit(gpu, &[bh, t, t]);
    let mut p = DTensor::uninit(gpu, &[bh, t, t]);
    let (ti, total_i) = (t as i32, total as i32);
    let f = gpu.kernels.get("mlstm_ds_bwd");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&dds_num.buf)
        .arg(&d_qn.buf)
        .arg(&dbar.buf)
        .arg(&ds.buf)
        .arg(&mut d_s.buf)
        .arg(&mut p.buf)
        .arg(&ti)
        .arg(&total_i);
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
    lb.arg(&p.buf)
        .arg(&mut dfc.buf)
        .arg(&mut dig.buf)
        .arg(&ti)
        .arg(&bht);
    unsafe { lb.launch(row_block_cfg(bh * t, t)) }.expect("mlstm_dfc_dig");
    (dfc, dig)
}

/// Reverse-cumsum + logσ' backward of `fc`: `df[BH,T]` from `dfc` and saved `f`.
pub fn revcumsum_dlogsig(gpu: &Gpu, dfc: &DTensor, f: &DTensor) -> DTensor {
    let (bh, t) = (dfc.rows(), dfc.cols());
    let mut df = DTensor::uninit(gpu, &[bh, t]);
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
    lb.arg(&num.buf)
        .arg(&psi.buf)
        .arg(&mut ytil.buf)
        .arg(&dhv_i)
        .arg(&total_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(total as u32)) }.expect("div_rows");
    ytil
}

/// Elementwise product `out = a ⊙ b` (fresh allocation).
pub fn mul(gpu: &Gpu, a: &DTensor, b: &DTensor) -> DTensor {
    let mut out = DTensor::uninit(gpu, a.dims());
    mul_into(gpu, a, b, &mut out);
    out
}

/// Elementwise `out = a * b` into a caller-owned buffer — the no-allocation form
/// of [`mul`]. `out` may alias either operand.
pub fn mul_into(gpu: &Gpu, a: &DTensor, b: &DTensor, out: &mut DTensor) {
    let n = a.len();
    assert_eq!(n, b.len(), "mul: length mismatch");
    assert_eq!(n, out.len(), "mul: output length mismatch");
    let n_i = n as i32;
    let f = gpu.kernels.get("mul");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&mut out.buf).arg(&a.buf).arg(&b.buf).arg(&n_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems(n as u32)) }.expect("mul");
}

// Fused chunkwise mLSTM (TFLA). See the kernel block in `gpu/kernels.rs`.
//
// Three launches forward, two backward, for a whole sequence — against ~25 per
// chunk on the op-at-a-time path. The [L, L] decay matrix stays in shared memory,
// so none of the per-chunk intermediates the old path allocated exist at all.

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
    pub ytil: SlabBuf, // [BH, T, dhv]  bf16 storage (reference: matHout at DTYPE)
    cst: DTensor,      // [BH, NC+1, dhv, dqk]  state entering each chunk
    nst: DTensor,      // [BH, NC+1, dqk]
    mst: DTensor,      // [BH, NC+1]
    fcb: DTensor,      // [BH, NC, L]  chunk-local cumulative log-forget
    msv: DTensor,      // [BH, T]      per-row stabilizer
    psiv: DTensor,     // [BH, T]
    qnv: DTensor,      // [BH, T]
}

impl MlstmFused {
    /// The state leaving the last chunk — index `NC` of each array, which the kernel
    /// publishes as the final state. Feed to the next chunk's `carry_in`.
    pub fn final_state(&self, gpu: &Gpu, bh: usize, dhv: usize, dqk: usize) -> MlstmState {
        let slots = self.nc + 1;
        // Slot NC of each bh — one launch, not a `bh`-long loop of tiny copies.
        let grab = |src: &DTensor, stride: usize| {
            let mut out = DTensor::uninit(gpu, &[bh * stride]);
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
            &self.cst, &self.nst, &self.mst, &self.fcb, &self.msv, &self.psiv, &self.qnv,
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

/// Public read of [`mma_enabled`] for the fused-vs-legacy dispatch, which must know
/// which path's shared-memory footprint to compare against the device limit.
pub fn mma_enabled_pub() -> bool {
    mma_enabled()
}

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
    LaunchConfig {
        grid_dim: grid,
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem,
    }
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

/// Peak per-block shared memory (bytes) any fused kernel needs at this blocking,
/// on the given path (`mma` = the tensor-core kernels are selected).
///
/// The kernels keep their whole `[L, L]` decay matrix — and, in the backward, two
/// `[dhv, dqk]` state-staging tiles — resident in shared memory, so at large head
/// dims (`dqk = dhv = 96` on the backbone) this crosses what a 100 KB-shared card
/// can opt a single block into. `MLstm::fused_chunk` compares it against
/// `Gpu::max_shared_optin` and drops to the op-at-a-time path when it does not fit;
/// the scalar and mma paths carry the same footprint, so `MLSTM_NO_MMA` is not an
/// escape from it.
pub fn mlstm_fused_smem_bytes(l: usize, dqk: usize, dhv: usize, mma: bool) -> usize {
    let kinds: &[&str] = if mma {
        &["fw_C", "fw_parallel_mma", "bw_dC", "bw_parallel_mma"]
    } else {
        &["fw_C", "fw_parallel", "bw_dC", "bw_parallel"]
    };
    kinds
        .iter()
        .map(|k| fused_smem(k, l, dqk, dhv) * std::mem::size_of::<f32>())
        .max()
        .unwrap()
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
/// The recurrent state entering a chunked mLSTM call: `C`, `n`, `m` for every
/// `(batch, head)`, laid out exactly as one slice of `MlstmFused`'s `cst`/`nst`/`mst`.
///
/// Produced by [`MlstmFused::final_state`] at the end of one chunk and handed to the
/// next call's `carry_in`. Zero-initialised state is represented by `None`, not by a
/// zero-filled `MlstmState` — that keeps the common (unchunked) path free of the copy.
pub struct MlstmState {
    pub c: DTensor, // [BH, dhv, dqk]
    pub n: DTensor, // [BH, dqk]
    pub m: DTensor, // [BH]
}

/// The BPTT state crossing a chunk border in the backward direction: the gradient wrt
/// the recurrent state entering the chunk to the right.
///
/// Backward unwinds chunks right to left, so chunk `c` produces this for chunk `c-1`.
pub struct MlstmDState {
    pub dc: DTensor, // [BH, dhv, dqk]
    pub dn: DTensor, // [BH, dqk]
}

/// One launch for the whole `[BH, stride]` <-> `[BH, slots, stride]` slot copy.
///
/// `gather` reads slot `idx` into a flat tensor; otherwise it writes the flat tensor
/// into slot `idx`. See `state_slot_copy` in `common.cu` for why this is a kernel and
/// not a `bh`-long loop of `memcpy_dtod`.
fn state_slot_copy(
    gpu: &Gpu,
    src: &DTensor,
    dst: &mut DTensor,
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
    unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }.expect("state_slot_copy");
}

/// Seed slot `idx` of a `[BH, slots, ...]` array from a `[BH, ...]` state.
fn seed_state_slot_n(
    gpu: &Gpu,
    src: &DTensor,
    dst: &mut DTensor,
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
    src: &DTensor,
    bh: usize,
    slots: usize,
    idx: usize,
    stride: usize,
) -> DTensor {
    let mut out = DTensor::uninit(gpu, &[bh * stride]);
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
    src: &DTensor,
    dst: &mut DTensor,
    bh: usize,
    slots: usize,
    stride: usize,
) {
    debug_assert_eq!(src.len(), bh * stride, "carry state: wrong source size");
    state_slot_copy(gpu, src, dst, bh, slots, 0, stride, false);
}

pub fn mlstm_fused_fw(
    gpu: &Gpu,
    qh: &SlabBuf,  // [BH, T, dqk]   bf16 storage (reference: matQ at DTYPE)
    kh: &SlabBuf,  // [BH, T, dqk]   bf16, already scaled by 1/√dqk
    vh: &SlabBuf,  // [BH, T, dhv]   bf16 (matV)
    igh: &DTensor, // [BH, T]        fp32: gate logit (reference pins vecI)
    fgh: &DTensor, // [BH, T]        fp32: gate logit (vecB is fp32 too)
    l: usize,
    // State this call continues from, or `None` to start the recurrence at zero.
    carry_in: Option<&MlstmState>,
) -> MlstmFused {
    let (bh, t, dqk) = (qh.dims()[0], qh.dims()[1], qh.dims()[2]);
    let dhv = vh.dims()[2];
    let l = l.min(t);
    assert!(
        mlstm_fused_supported(l, dqk, dhv),
        "fused mLSTM: unsupported shape"
    );
    let nc = t.div_ceil(l);
    let (t_i, l_i, nc_i, dqk_i, dhv_i, bh_i) = (
        t as i32, l as i32, nc as i32, dqk as i32, dhv as i32, bh as i32,
    );

    let mut fcb = DTensor::uninit(gpu, &[bh, nc, l]);
    let mut cst = DTensor::uninit(gpu, &[bh, nc + 1, dhv, dqk]);
    let mut nst = DTensor::uninit(gpu, &[bh, nc + 1, dqk]);
    let mut mst = DTensor::uninit(gpu, &[bh, nc + 1]);
    // Carrying: stage the incoming state into slot 0 of each array, which is exactly
    // where `mlstm_fw_C` reads it from under `CARRY` (and where it would otherwise
    // publish the zero initial state). Copying rather than aliasing keeps the kernel's
    // "state entering chunk k lives at index k" invariant intact for backward.
    if let Some(st) = carry_in {
        seed_state_slot0(gpu, &st.c, &mut cst, bh, nc + 1, dhv * dqk);
        seed_state_slot0(gpu, &st.n, &mut nst, bh, nc + 1, dqk);
        seed_state_slot0(gpu, &st.m, &mut mst, bh, nc + 1, 1);
    }
    let carry_i = carry_in.is_some() as i32;
    // ytil is the kernel's output h (reference stores matHout at DTYPE); the
    // stabilizer/normalizer triple stays fp32, as does the chunk state.
    let mut ytil = SlabBuf::new(gpu, &[bh, t, dhv]);
    let mut msv = DTensor::uninit(gpu, &[bh, t]);
    let mut psiv = DTensor::uninit(gpu, &[bh, t]);
    let mut qnv = DTensor::uninit(gpu, &[bh, t]);

    let f = gpu.kernels.get("mlstm_fw_gates");
    let mut lb = gpu.stream.launch_builder(&f);
    lb.arg(&fgh.buf)
        .arg(&mut fcb.buf)
        .arg(&t_i)
        .arg(&l_i)
        .arg(&nc_i)
        .arg(&bh_i);
    unsafe { lb.launch(LaunchConfig::for_num_elems((bh * nc) as u32)) }.expect("mlstm_fw_gates");

    let tv = fused_tv(dhv);
    let tv_i = tv as i32;
    let ntv = dhv.div_ceil(tv) as u32;

    let (f, smem) = fused_kernel(gpu, "mlstm_fw_C", fused_smem("fw_C", l, dqk, dhv));
    let mut lb = gpu.stream.launch_builder(&f);
    push_slab_ref!(lb, *kh);
    push_slab_ref!(lb, *vh);
    lb.arg(&igh.buf)
        .arg(&fcb.buf)
        .arg(&mut cst.buf)
        .arg(&mut nst.buf)
        .arg(&mut mst.buf)
        .arg(&t_i)
        .arg(&l_i)
        .arg(&nc_i)
        .arg(&dqk_i)
        .arg(&dhv_i)
        .arg(&tv_i)
        .arg(&carry_i);
    unsafe { lb.launch(fused_cfg((ntv, bh as u32, 1), FUSED_THREADS_REC, smem)) }
        .expect("mlstm_fw_C");

    let (name, kind) = fw_parallel_kernel(gpu);
    let (f, smem) = fused_kernel(gpu, name, fused_smem(kind, l, dqk, dhv));
    let mut lb = gpu.stream.launch_builder(&f);
    push_slab_ref!(lb, *qh);
    push_slab_ref!(lb, *kh);
    push_slab_ref!(lb, *vh);
    lb.arg(&igh.buf)
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
        .arg(&dhv_i);
    unsafe {
        lb.launch(fused_cfg(
            (nc as u32, bh as u32, 1),
            FUSED_THREADS_PAR,
            smem,
        ))
    }
    .expect("mlstm_fw_parallel");

    MlstmFused {
        l,
        nc,
        ytil,
        cst,
        nst,
        mst,
        fcb,
        msv,
        psiv,
        qnv,
    }
}

/// Chunkwise backward: `mlstm_bw_dC` → `mlstm_bw_parallel`.
/// Returns `(dq, dk, dv, dig, dfg, dstate)` — the grads head-major like the inputs,
/// plus the BPTT state to hand to the chunk on the left (see [`MlstmDState`]).
#[allow(clippy::too_many_arguments)]
pub fn mlstm_fused_bw(
    gpu: &Gpu,
    sv: &MlstmFused,
    qh: &SlabBuf,
    kh: &SlabBuf,
    vh: &SlabBuf,
    igh: &DTensor,
    fgh: &DTensor,
    // The incoming gradient is fp32: it is a transient, not a saved tensor, so
    // narrowing it would buy no memory. (The reference does cast matDeltaH to
    // DTYPE, but only to feed its tensor cores.)
    d_ytil: &DTensor, // [BH, T, dhv]
    // BPTT state flowing in from the chunk to the RIGHT, or `None` for the rightmost
    // chunk (and for an unchunked call), where it is zero.
    carry_in: Option<&MlstmDState>,
) -> (DTensor, DTensor, DTensor, DTensor, DTensor, MlstmDState) {
    let (bh, t, dqk) = (qh.dims()[0], qh.dims()[1], qh.dims()[2]);
    let dhv = vh.dims()[2];
    let (l, nc) = (sv.l, sv.nc);
    let (t_i, l_i, nc_i, dqk_i, dhv_i) = (t as i32, l as i32, nc as i32, dqk as i32, dhv as i32);

    // Zeroed, not `uninit`: under CARRY `mlstm_bw_parallel` reads slot NC for every
    // `dhv` tile, while the seeding below and `mlstm_bw_dC`'s tiled writes only cover
    // the slices each tile owns. Leaving the rest uninitialised made the chunked
    // backward read garbage — which showed up as a run-to-run varying gradient, not a
    // crash (the unchunked path is bit-exact reproducible, the chunked one was not).
    let mut dcst = DTensor::zeros(gpu, &[bh, nc + 1, dhv, dqk]);
    let mut dnst = DTensor::zeros(gpu, &[bh, nc + 1, dqk]);
    // Seed slot NC with the gradient from the chunk to the right; the kernel reads it
    // there under CARRY instead of starting at zero.
    if let Some(st) = carry_in {
        seed_state_slot_n(gpu, &st.dc, &mut dcst, bh, nc + 1, nc, dhv * dqk);
        seed_state_slot_n(gpu, &st.dn, &mut dnst, bh, nc + 1, nc, dqk);
    }
    let carry_i = carry_in.is_some() as i32;

    let tv = fused_tv(dhv);
    let tv_i = tv as i32;
    let ntv = dhv.div_ceil(tv) as u32;

    let (f, smem) = fused_kernel(gpu, "mlstm_bw_dC", fused_smem("bw_dC", l, dqk, dhv));
    let mut lb = gpu.stream.launch_builder(&f);
    push_slab_ref!(lb, *qh);
    lb.arg(&d_ytil.buf);
    push_slab_ref!(lb, sv.ytil);
    lb.arg(&sv.psiv.buf)
        .arg(&sv.qnv.buf)
        .arg(&sv.msv.buf)
        .arg(&sv.fcb.buf)
        .arg(&sv.mst.buf)
        .arg(&mut dcst.buf)
        .arg(&mut dnst.buf)
        .arg(&t_i)
        .arg(&l_i)
        .arg(&nc_i)
        .arg(&dqk_i)
        .arg(&dhv_i)
        .arg(&tv_i)
        .arg(&carry_i);
    unsafe { lb.launch(fused_cfg((ntv, bh as u32, 1), FUSED_THREADS_REC, smem)) }
        .expect("mlstm_bw_dC");

    let mut dq = DTensor::uninit(gpu, &[bh, t, dqk]);
    let mut dk = DTensor::uninit(gpu, &[bh, t, dqk]);
    let mut dv = DTensor::uninit(gpu, &[bh, t, dhv]);
    let mut dig = DTensor::uninit(gpu, &[bh, t]);
    let mut dfg = DTensor::uninit(gpu, &[bh, t]);

    let (name, kind) = bw_parallel_kernel(gpu);
    let (f, smem) = fused_kernel(gpu, name, fused_smem(kind, l, dqk, dhv));
    let mut lb = gpu.stream.launch_builder(&f);
    push_slab_ref!(lb, *qh);
    push_slab_ref!(lb, *kh);
    push_slab_ref!(lb, *vh);
    lb.arg(&igh.buf)
        .arg(&fgh.buf)
        .arg(&sv.fcb.buf)
        .arg(&sv.cst.buf)
        .arg(&sv.nst.buf)
        .arg(&sv.mst.buf)
        .arg(&dcst.buf)
        .arg(&dnst.buf);
    push_slab_ref!(lb, sv.ytil);
    lb.arg(&d_ytil.buf)
        .arg(&sv.psiv.buf)
        .arg(&sv.qnv.buf)
        .arg(&sv.msv.buf)
        .arg(&mut dq.buf)
        .arg(&mut dk.buf)
        .arg(&mut dv.buf)
        .arg(&mut dig.buf)
        .arg(&mut dfg.buf)
        .arg(&t_i)
        .arg(&l_i)
        .arg(&nc_i)
        .arg(&dqk_i)
        .arg(&dhv_i)
        .arg(&carry_i);
    unsafe {
        lb.launch(fused_cfg(
            (nc as u32, bh as u32, 1),
            FUSED_THREADS_PAR,
            smem,
        ))
    }
    .expect("mlstm_bw_parallel");

    // Slot 0 is the gradient wrt the state entering chunk 0 — i.e. what flows out of
    // this chunk into the one on its left.
    let dstate = MlstmDState {
        dc: read_state_slot_n(gpu, &dcst, bh, nc + 1, 0, dhv * dqk),
        dn: read_state_slot_n(gpu, &dnst, bh, nc + 1, 0, dqk),
    };
    (dq, dk, dv, dig, dfg, dstate)
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

    /// A queued+flushed AdamW must leave params, moments and grads exactly where the
    /// per-tensor `adamw` leaves them.
    ///
    /// Tensor sizes are deliberately uneven and the decay flags mixed, so the
    /// kernel's per-slot offset search and per-slot `wd` are both exercised — a
    /// kernel that used slot 0's `wd` for every slot would pass a uniform test.
    #[test]
    fn adamw_batch_matches_per_tensor_adamw() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let sizes = [1usize, 17, 256, 1000, 4096];
        let decays = [true, false, true, true, false];
        let mk = |seed: f32, n: usize| {
            let d: Vec<f32> = (0..n).map(|i| (i as f32 * 0.19 + seed).sin()).collect();
            DTensor::from_host(&gpu, &Tensor::new(&[n], d))
        };
        // The second moment is a running mean of squares, so it is non-negative by
        // construction; seeding it from `sin` would put `sqrt` on negative input.
        let mk_v = |seed: f32, n: usize| {
            let d: Vec<f32> = (0..n)
                .map(|i| ((i as f32 * 0.19 + seed).sin()).abs())
                .collect();
            DTensor::from_host(&gpu, &Tensor::new(&[n], d))
        };
        // lr·wd must be large enough for a wrong per-slot `wd` to exceed the 1e-3
        // comparison tolerance — at the production 1e-3/0.05 the decay term is 5e-5
        // and a kernel using slot 0's `wd` everywhere would pass unnoticed.
        let cfg = AdamCfg {
            t: 3,
            ..AdamCfg::new(0.5, 0.5)
        };

        let mut fast: Vec<_> = sizes
            .iter()
            .enumerate()
            .map(|(k, &n)| {
                (
                    mk(k as f32, n),
                    mk(k as f32 + 10.0, n),
                    mk(k as f32 + 20.0, n),
                    mk_v(k as f32 + 30.0, n),
                )
            })
            .collect();
        let mut slow: Vec<_> = sizes
            .iter()
            .enumerate()
            .map(|(k, &n)| {
                (
                    mk(k as f32, n),
                    mk(k as f32 + 10.0, n),
                    mk(k as f32 + 20.0, n),
                    mk_v(k as f32 + 30.0, n),
                )
            })
            .collect();

        let mut q = AdamwQueue::new();
        for (i, (p, g, m, v)) in fast.iter_mut().enumerate() {
            q.push(&gpu, p, g, m, v, &cfg, decays[i]);
        }
        q.flush(&gpu, &cfg);

        for (i, (p, g, m, v)) in slow.iter_mut().enumerate() {
            adamw(&gpu, p, g, m, v, &cfg, decays[i]);
            g.zero_(&gpu);
        }

        for (i, ((pf, gf, mf, vf), (ps, gs, ms, vs))) in fast.iter().zip(slow.iter()).enumerate() {
            assert_close(&pf.to_host(&gpu).data, &ps.to_host(&gpu).data);
            assert_close(&mf.to_host(&gpu).data, &ms.to_host(&gpu).data);
            assert_close(&vf.to_host(&gpu).data, &vs.to_host(&gpu).data);
            // Both paths must leave the gradient zeroed.
            let gd = gf.to_host(&gpu).data;
            assert!(
                gd.iter().all(|x| *x == 0.0),
                "tensor {i}: queued grad not zeroed"
            );
            assert_close(&gd, &gs.to_host(&gpu).data);
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
                DTensor::from_host(&gpu, &Tensor::new(&[r, c], d))
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
            let mut slow = DTensor::uninit(&gpu, &[r, c]);
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
            unsafe { lb.launch(LaunchConfig::for_num_elems(r as u32)) }.expect("slow ce");
            assert_close(&fast.to_host(&gpu).data, &slow.to_host(&gpu).data);
        }

        // chunk_ab backward: dfc/dig are accumulated onto, so both paths start from
        // the same non-zero state to catch a kernel that overwrites instead of adds.
        for &(bh, l) in &[(3usize, 8usize), (6, 64), (4, 256)] {
            let mk = |s: f32| {
                let d: Vec<f32> = (0..bh * l).map(|i| (i as f32 * 0.23 + s).sin()).collect();
                DTensor::from_host(&gpu, &Tensor::new(&[bh, l], d))
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
            unsafe { lb.launch(LaunchConfig::for_num_elems((bh * l) as u32)) }.expect("slow ab");

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
                DTensor::from_host(&gpu, &Tensor::new(&[bh, t], data))
            };
            let f = mk(0.0);
            let dfc = mk(1.7);

            let fast_fc = cumsum_logsig(&gpu, &f).to_host(&gpu);
            let fast_df = revcumsum_dlogsig(&gpu, &dfc, &f).to_host(&gpu);

            // Same inputs through the thread-per-row kernels.
            let (ti, bhi) = (t as i32, bh as i32);
            let mut slow_fc = DTensor::uninit(&gpu, &[bh, t]);
            let func = gpu.kernels.get("cumsum_logsig");
            let mut lb = gpu.stream.launch_builder(&func);
            lb.arg(&f.buf).arg(&mut slow_fc.buf).arg(&ti).arg(&bhi);
            unsafe { lb.launch(LaunchConfig::for_num_elems(bh as u32)) }.expect("slow cumsum");

            let mut slow_df = DTensor::uninit(&gpu, &[bh, t]);
            let func = gpu.kernels.get("revcumsum_dlogsig");
            let mut lb = gpu.stream.launch_builder(&func);
            lb.arg(&dfc.buf)
                .arg(&f.buf)
                .arg(&mut slow_df.buf)
                .arg(&ti)
                .arg(&bhi);
            unsafe { lb.launch(LaunchConfig::for_num_elems(bh as u32)) }.expect("slow revcumsum");

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
        let srcs: Vec<DTensor> = widths
            .iter()
            .enumerate()
            .map(|(k, &w)| {
                let data: Vec<f32> = (0..bh * t * w)
                    .map(|i| (i as f32) * 0.25 + k as f32 * 1000.0)
                    .collect();
                DTensor::from_host(&gpu, &Tensor::new(&[bh, t, w], data))
            })
            .collect();
        let refs: Vec<(&DTensor, usize)> = srcs
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
            let mut got: Vec<DTensor> = widths
                .iter()
                .map(|&w| DTensor::zeros(&gpu, &[bh, t, w]))
                .collect();
            let mut want: Vec<DTensor> = widths
                .iter()
                .map(|&w| DTensor::zeros(&gpu, &[bh, t, w]))
                .collect();
            {
                let mut pairs: Vec<(&mut DTensor, &DTensor, usize)> = got
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

    /// Consecutive `IdBatch` uploads must not corrupt each other's ids.
    ///
    /// The uploads are async, so a batch whose buffers are reused too early has its
    /// ids replaced by a later batch's — no error, just wrong gathers.
    ///
    /// **This test does not reproduce that failure.** Single-buffering `IdBatch`
    /// diverges real training deterministically (char loss 2.6 -> 69 within ~10 steps)
    /// while this test still passes, at every group count and contention level tried.
    /// The reuse window evidently needs the real workload's queue depth. It is kept as
    /// a guard on the packing and offset arithmetic, which it does cover; the
    /// double-buffering it cannot see is pinned only by that training run, so treat
    /// `IdBatch` as a place where a green suite is not sufficient evidence.
    #[test]
    fn id_batch_uploads_do_not_clobber_each_other() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        // A table whose row `i` is all `i`, so a gathered row names the id that
        // fetched it and a stale id is obvious.
        let rows = 64;
        let table = DTensor::from_host(
            &gpu,
            &Tensor::new(
                &[rows, 4],
                (0..rows).flat_map(|i| [i as f32; 4]).collect::<Vec<_>>(),
            ),
        );

        let mut batch = IdBatch::new();
        let mut got = Vec::new();
        // Distinct id lists of *differing* lengths, so the staging buffers resize
        // mid-sequence — the case where a naive implementation reallocates under an
        // in-flight copy.
        let groups: Vec<Vec<u32>> = (0..8)
            .map(|g| {
                (0..(3 + g * 5))
                    .map(|i| ((g * 7 + i) % rows) as u32)
                    .collect()
            })
            .collect();

        // Something slow between the upload and the gather, so the copy for group
        // `g+1` is issued while group `g`'s gather is still queued — the interleaving
        // that real training produces and that a bare upload/gather pair does not.
        let busy = DTensor::zeros(&gpu, &[512, 512]);
        let mut sink = DTensor::uninit(&gpu, &[512, 512]);

        for ids in &groups {
            batch.upload(&gpu, &[ids]);
            matmul_nn_into(&gpu, &busy, &busy, &mut sink, 0.0);
            // Gather, keeping the result on the device — no sync here, so an unsound
            // reuse has every chance to show.
            let out = embedding_gather_u32(&gpu, &table, &batch.get(0), ids.len(), 4);
            got.push(out);
        }

        for (ids, out) in groups.iter().zip(&got) {
            let host = out.to_host(&gpu).data;
            for (row, &id) in ids.iter().enumerate() {
                assert_eq!(
                    host[row * 4],
                    id as f32,
                    "row {row} gathered id {} instead of {id} — a later upload clobbered it",
                    host[row * 4]
                );
            }
        }
    }

    /// Several lists in one batch must come back at their own offsets.
    #[test]
    fn id_batch_packs_multiple_lists() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let mut batch = IdBatch::new();
        let a: Vec<u32> = vec![5, 1, 9];
        let b: Vec<u32> = vec![7, 7];
        let c: Vec<u32> = vec![0, 3, 2, 8];
        batch.upload(&gpu, &[&a, &b, &c]);

        for (i, want) in [&a, &b, &c].iter().enumerate() {
            let view = batch.get(i);
            let host = gpu.stream.clone_dtoh(&view).expect("dtoh");
            assert_eq!(&host, *want, "list {i} came back wrong");
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
            let (da, db) = (DTensor::from_host(&gpu, &a), DTensor::from_host(&gpu, &b));

            let mut want = DTensor::uninit(&gpu, &[m, n]);
            match form {
                MmForm::Nn => matmul_nn_into(&gpu, &da, &db, &mut want, 0.0),
                MmForm::Nt => matmul_nt_into(&gpu, &da, &db, &mut want, 0.0),
                MmForm::Tn => matmul_tn_into(&gpu, &da, &db, &mut want, 0.0),
            }
            let mut got = DTensor::uninit(&gpu, &[m, n]);
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

        let a = DTensor::from_host(&gpu, &Tensor::random(&[64, 32], 1.0));
        let b = DTensor::from_host(&gpu, &Tensor::random(&[32, 16], 1.0));
        let mut c = DTensor::uninit(&gpu, &[64, 16]);
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
        let a2 = DTensor::from_host(&gpu, &Tensor::random(&[32, 32], 1.0));
        let mut c2 = DTensor::uninit(&gpu, &[32, 16]);
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
        let (da, db) = (DTensor::from_host(&gpu, &a), DTensor::from_host(&gpu, &b));

        let mut once = DTensor::zeros(&gpu, &[m, n]);
        GemmBf16::new().run(&gpu, MmForm::Nn, &da, &db, &mut once, 0.0);
        let single = once.to_host(&gpu).data;

        let mut twice = DTensor::zeros(&gpu, &[m, n]);
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
            &DTensor::from_host(&gpu, &a),
            &DTensor::from_host(&gpu, &b),
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
            &DTensor::from_host(&gpu, &a),
            &DTensor::from_host(&gpu, &b),
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
            &DTensor::from_host(&gpu, &a),
            &DTensor::from_host(&gpu, &b),
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
            &DTensor::from_host(&gpu, &a),
            &DTensor::from_host(&gpu, &b),
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
            &DTensor::from_host(&gpu, &a),
            &DTensor::from_host(&gpu, &bt),
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
            &DTensor::from_host(&gpu, &at),
            &DTensor::from_host(&gpu, &bn),
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
        let mut dx = DTensor::from_host(&gpu, &x);
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
        let mut dx2 = DTensor::from_host(&gpu, &x);
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
        let dx = DTensor::from_host(&gpu, &x);
        let y_gpu = softcap_forward(&gpu, &dx, cap);
        let dx_gpu = softcap_backward(&gpu, &DTensor::from_host(&gpu, &dy), &y_gpu, cap);
        assert_close(&y_gpu.to_host(&gpu).data, &y_cpu.data);
        assert_close(&dx_gpu.to_host(&gpu).data, &dx_cpu.data);
    }

    #[test]
    fn linear_bias_helpers_match_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
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

    /// `add_col_sum` splits its row axis across `blockIdx.y` and combines the slices
    /// with `atomicAdd`, so the shapes that matter are the wide, many-row ones a real
    /// layer sees — where the split is active and every column takes several atomics.
    #[test]
    fn add_col_sum_matches_cpu_at_layer_shapes() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        use crate::nn2::ops as cpu;
        for (rows, n) in [(512, 768), (2048, 768), (1, 768), (333, 257)] {
            let dy = Tensor::random(&[rows, n], 1.0);
            let mut db_cpu = Tensor::random(&[n], 1.0);
            let mut db_gpu = DTensor::from_host(&gpu, &db_cpu);
            cpu::add_col_sum(&mut db_cpu, &dy);
            add_col_sum(&gpu, &mut db_gpu, &DTensor::from_host(&gpu, &dy));
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
        let gathered_gpu = embedding_gather(&gpu, &DTensor::from_host(&gpu, &table), &ids, dim);
        assert_close(&gathered_gpu.to_host(&gpu).data, &gathered_cpu.data);
        // scatter_add from the gathered grads
        let mut dt_cpu = Tensor::zeros(&[vocab, dim]);
        cpu::embedding_scatter_add(&mut dt_cpu, &ids, &gathered_cpu, dim);
        let mut dt_gpu = DTensor::zeros(&gpu, &[vocab, dim]);
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

            let gamma_d = DTensor::from_host(&gpu, &gamma);
            let (out_gpu, fwd_gpu) =
                rms_norm_forward(&gpu, &DTensor::from_host(&gpu, &x), &gamma_d, group, eps);
            let mut dg_gpu = DTensor::zeros(&gpu, &[f]);
            let dx_gpu = rms_norm_backward(
                &gpu,
                &DTensor::from_host(&gpu, &dy),
                &fwd_gpu,
                &gamma_d,
                &mut dg_gpu,
                group,
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

        let dgamma_t = DTensor::from_host(&gpu, &gamma);
        let (out_gpu, fwd_gpu) =
            rms_norm_forward(&gpu, &DTensor::from_host(&gpu, &x), &dgamma_t, group, eps);
        let mut dg_gpu = DTensor::zeros(&gpu, &[f]);
        let dx_gpu = rms_norm_backward(
            &gpu,
            &DTensor::from_host(&gpu, &dy),
            &fwd_gpu,
            &dgamma_t,
            &mut dg_gpu,
            group,
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
            softmax_cross_entropy(&gpu, &DTensor::from_host(&gpu, &logits), &targets);
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

        let mut p_gpu = DTensor::from_host(&gpu, &param);
        let mut m = DTensor::zeros(&gpu, &[n]);
        let mut v = DTensor::zeros(&gpu, &[n]);
        adamw(
            &gpu,
            &mut p_gpu,
            &DTensor::from_host(&gpu, &grad),
            &mut m,
            &mut v,
            &cfg,
            true,
        );
        assert_close(&p_gpu.to_host(&gpu).data, &param_cpu.data);
    }
}
