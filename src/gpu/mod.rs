//! CUDA backend (feature `cuda`).
//!
//! This module is the device side of the Backend/ops seam described in
//! `PLAN-gpu.md`. Bring-up order:
//!   1. Prove the toolchain: a lazy `CudaContext`, an NVRTC-compiled kernel, and
//!      a round-trip smoke test (this file). <-- we are here.
//!   2. Give `Tensor` an optional resident device buffer + up/download.
//!   3. Port the ops-seam kernels one at a time, each checked against the CPU
//!      implementation / finite-difference tests.
//!
//! Nothing here is wired into the layers yet; enabling the feature only adds the
//! context + smoke test so we can confirm `cudarc`, NVRTC and the driver all work
//! on the laptop before porting real kernels.

use std::sync::Arc;

use cudarc::cublas::CudaBlas;
use cudarc::cublas::sys::{cublasMath_t, cublasSetMathMode};
use cudarc::driver::{CudaContext, CudaStream};

pub mod bf16;
pub mod block;
pub mod buf;
pub mod dtensor;
pub mod flat;
pub mod hierarchical;
pub mod kernels;
pub mod linear;
pub mod lm;
pub mod mlstm;
pub mod offload;
pub mod ops;
pub mod rms_norm;
pub mod slstm;
pub mod train;

pub use bf16::{BTensor, Slab};
pub use buf::{Buf, Pool};
pub use dtensor::DTensor;
pub use offload::OffloadRing;
use kernels::Kernels;

use iron_oxide::collections::Matrix;

/// Download a 2-D device tensor into an `iron_oxide` `Matrix` (row-major), used
/// when exporting device weights into the CPU `nn` layer format for `HIER`.
pub(crate) fn dt_matrix(gpu: &Gpu, t: &DTensor) -> Matrix {
    let h = t.to_host(gpu);
    let (rows, cols) = (h.dims()[0], h.dims()[1]);
    Matrix::from_vec(h.data, rows, cols)
}

/// Download a 1-D device tensor into a boxed slice.
pub(crate) fn dt_vec(gpu: &Gpu, t: &DTensor) -> Box<[f32]> {
    t.to_host(gpu).data.into_boxed_slice()
}

/// Host `Matrix` → `Tensor` (2-D, row-major), for uploading `nn` weights back to
/// the device when importing a `HIER` checkpoint.
pub(crate) fn tensor_from_matrix(m: &Matrix) -> crate::tensor::Tensor {
    crate::tensor::Tensor::new(&[m.rows(), m.cols()], m.as_slice().to_vec())
}

/// Host slice → 1-D `Tensor`.
pub(crate) fn tensor_from_slice(s: &[f32]) -> crate::tensor::Tensor {
    crate::tensor::Tensor::new(&[s.len()], s.to_vec())
}

/// Whether host waits park the thread instead of spinning. On unless
/// `GPU_SPIN_SYNC` is set, which restores the driver's default busy-wait.
///
/// Costs nothing measurable: 679 vs 675 ms/step over three `step_time` runs each,
/// inside run-to-run noise. It also recovers less than one might hope — 100% -> 95%
/// of a core — because the submitting thread is not mostly *waiting*: at 4096 words
/// a step's launches put 40% of its CPU in `libcuda` and 28% in libc marshalling
/// launch arguments. Only the residual sync time is spin, and that is the 5%.
///
/// Read through a `OnceLock` so the context flag and the event-creation flags,
/// which are set at different times, cannot disagree.
pub fn blocking_sync() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("GPU_SPIN_SYNC").is_err())
}

/// Flags for an event the host will wait on with `synchronize`. The context's
/// scheduling policy does not reach `cuEventSynchronize` — an event spins unless it
/// was *created* with `CU_EVENT_BLOCKING_SYNC` — so host-waited events need this.
/// Events only ordered device-side (`stream.wait`) never block a thread and can
/// keep the cheaper default.
pub fn host_wait_event_flags() -> Option<cudarc::driver::sys::CUevent_flags> {
    blocking_sync().then_some(cudarc::driver::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC)
}

/// A live CUDA device: the context, its default stream, a cuBLAS handle, and the
/// NVRTC-compiled kernel set. Cheap to clone (every field is an `Arc`). Created
/// once via [`Gpu::new`].
#[derive(Clone)]
pub struct Gpu {
    pub context: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub blas: Arc<CudaBlas>,
    pub kernels: Arc<Kernels>,
    /// Most shared memory a single block may opt into on this device, in bytes
    /// (`CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`). The fused mLSTM
    /// kernels keep their whole decay matrix in shared memory, so at large head
    /// dims they can exceed this — see `MLstm::fused_chunk`, which falls back to
    /// the op-at-a-time path rather than failing the opt-in.
    pub max_shared_optin: usize,
    /// Number of SMs. A cooperative launch's grid must be co-resident, so this is
    /// the ceiling on its block count — see `ops::slstm_fused_time_geometry`.
    pub sm_count: usize,
}

impl Gpu {
    /// Initialise CUDA device 0. Returns an error string (rather than panicking)
    /// so callers can fall back to the CPU path when no GPU is present.
    ///
    /// The work goes on an explicitly created stream, not `default_stream()`:
    /// cudarc's default stream is the *legacy* NULL stream, which implicitly
    /// synchronizes against every other stream on the context. Everything is
    /// submitted to this one stream in issue order, so ordering semantics are
    /// those of program order.
    pub fn new() -> Result<Self, String> {
        let context = CudaContext::new(0).map_err(|e| format!("CUDA init failed: {e:?}"))?;
        // Leaving the default stream also puts cudarc into "multi stream mode", where
        // every `CudaSlice` records a CUDA event on each use so that buffers shared
        // across streams stay ordered. We submit everything to the one stream below,
        // in issue order, so that bookkeeping buys nothing and costs host time on the
        // per-launch path we are here to shorten.
        //
        // NOTE: this is what a second stream would have to contend with — the flag
        // removes the automatic cross-stream ordering, so any future multi-stream
        // work must place its own events explicitly.
        //
        // SAFETY: the contract is "the caller manages stream synchronization". There
        // is a single stream, so program order *is* the synchronization. Must happen
        // before any allocation — the flag only affects slices created after it.
        unsafe { context.disable_event_tracking() };
        // `cuDevicePrimaryCtxRetain` leaves the scheduling policy at CU_CTX_SCHED_AUTO,
        // which with one active context on a many-core host resolves to a busy-wait: a
        // host core sits at 100% spinning inside every `synchronize()`. BLOCKING_SYNC
        // parks the thread on an interrupt instead, trading a few microseconds of wake
        // latency per sync for an idle core.
        if blocking_sync() {
            context
                .set_blocking_synchronize()
                .map_err(|e| format!("setting blocking sync failed: {e:?}"))?;
        }
        let stream = context
            .new_stream()
            .map_err(|e| format!("stream creation failed: {e:?}"))?;
        let blas =
            CudaBlas::new(stream.clone()).map_err(|e| format!("cuBLAS init failed: {e:?}"))?;
        set_tf32(&blas)?;
        let kernels = Kernels::load(&context)?;
        let max_shared_optin = context
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )
            .map_err(|e| format!("querying max shared memory failed: {e:?}"))? as usize;
        let sm_count = context
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(|e| format!("querying SM count failed: {e:?}"))? as usize;
        Ok(Self {
            context,
            stream,
            blas: Arc::new(blas),
            kernels: Arc::new(kernels),
            max_shared_optin,
            sm_count,
        })
    }
}

/// Let cuBLAS run our f32 GEMMs on the tensor cores in TF32.
///
/// A plain `sgemm` in the default math mode uses the FP32 CUDA cores and nothing
/// else — on Ampere and later that leaves the tensor cores, i.e. most of the
/// machine's matmul throughput, idle. `CUBLAS_TF32_TENSOR_OP_MATH` keeps the
/// interface fp32 (our buffers, the accumulator and the result stay fp32) and only
/// rounds the *multiplicand mantissas* to TF32's 10 bits before they enter the
/// tensor core. Dynamic range is fp32's — TF32 keeps all 8 exponent bits — so no
/// value here can overflow or flush that would not have in fp32; the products just
/// carry ~10 mantissa bits instead of 24, which is why this is a knob and not a
/// silent default.
///
/// That is a real precision cut, and the reason it is safe for us is that the
/// error lands where we already tolerate it: gradients and activations, summed in
/// fp32 over long K, whose per-step noise is orders of magnitude above 2^-11
/// relative. What must *not* go through it is the stabilizer's own arithmetic
/// (`m`, `exp(-m)`) — but that lives in the fused kernels, not in cuBLAS, so it is
/// untouched by this.
///
/// **Opt-in** (`GPU_TF32=1`), unlike the tensor cores in the mLSTM kernels, which
/// are on by default. The reason is what each one buys: the fused kernels' dots are
/// where the model's own arithmetic lives, and the reference (`mlstm_kernels`) puts
/// them on the tensor cores too. cuBLAS's math mode, by contrast, reaches *every*
/// GEMM in the backend — including the ones the parity tests use as an exact-fp32
/// oracle — so it is a switch rather than a default.
///
/// What it buys has grown as the rest of the step got faster: GEMMs are now 43% of
/// kernel time, and enabling this cuts GEMM time 191.1 -> 161.9 ms and whole-step
/// kernel time 447.9 -> 416.0 ms (~7%), against the ~4% measured when the step was
/// still dominated by the backbone's sLSTM.
///
/// The cost is that six parity tests fail with it on — the three `gemm_*_matches_cpu`
/// oracles plus `slstm_fused_time_matches_per_step` and the two hierarchical grouping
/// tests. The error is ~1.3e-3 relative (TF32's ~10-bit mantissa) against their 1e-3
/// tolerance, i.e. the kernels are right and the tolerance is simply tighter than TF32
/// can meet. Turning this on by default therefore means retuning those tolerances and
/// giving up the exact-fp32 oracle, which is a numerics decision, not a perf one.
fn set_tf32(blas: &CudaBlas) -> Result<(), String> {
    let mode = if std::env::var("GPU_TF32").as_deref() == Ok("1") {
        cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH
    } else {
        cublasMath_t::CUBLAS_DEFAULT_MATH
    };
    // SAFETY: `blas` owns a live cuBLAS handle; setting its math mode is a
    // handle-local property and touches no device memory.
    let status = unsafe { cublasSetMathMode(*blas.handle(), mode) };
    match status {
        cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
        e => Err(format!("cublasSetMathMode({mode:?}) failed: {e:?}")),
    }
}

#[cfg(test)]
/// A `Gpu` plus the lock that serializes GPU tests; derefs to `Gpu`, so call sites
/// use it exactly like one. See [`test_gpu`] for why the lock exists.
pub struct TestGpu {
    gpu: Gpu,
    _lock: std::sync::MutexGuard<'static, ()>,
}

#[cfg(test)]
impl std::ops::Deref for TestGpu {
    type Target = Gpu;
    fn deref(&self) -> &Gpu {
        &self.gpu
    }
}

#[cfg(test)]
/// Shared test helper: a `Gpu` if the machine has one, else `None` (so GPU tests
/// self-skip on the dev box with no Nvidia card).
///
/// Holds a process-wide lock for the caller's lifetime, so **only one GPU test runs
/// at a time**. `cargo test` runs tests on parallel threads, which would otherwise
/// have them contend for the same context: the cooperative launches need the whole
/// grid co-resident, and the larger shapes want most of the card's memory, so
/// concurrent tests both slow each other down and can fail to launch. Serializing
/// here keeps `cargo test --features cuda` honest without a `--test-threads=1`
/// incantation that a future runner would forget.
fn test_gpu() -> Option<TestGpu> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    // A panicking test poisons the lock; the guarded data is `()`, so there is no
    // broken invariant to protect and the next test may proceed.
    let lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    match Gpu::new() {
        Ok(gpu) => Some(TestGpu { gpu, _lock: lock }),
        Err(e) => {
            eprintln!("skipping GPU test: {e}");
            None
        }
    }
}

/// Device memory currently **allocated** from the async pool, in MB.
///
/// Distinct from `mem_get_info`, which reports everything the driver has handed the
/// process — including blocks the allocator is merely caching for reuse. That cached
/// portion is reclaimable and is *not* a reason to OOM, so a memory report built on
/// `mem_get_info` alone cannot tell "the model grew" from "the allocator is holding
/// cache". This is the number that decides whether the next allocation succeeds.
///
/// `None` if the pool attribute cannot be read.
pub fn pool_used_mb() -> Option<f64> {
    use cudarc::driver::sys;
    let pool = unsafe {
        let dev = cudarc::driver::result::device::get(0).ok()?;
        cudarc::driver::result::device::get_default_mem_pool(dev).ok()?
    };
    let mut v: u64 = 0;
    unsafe {
        cudarc::driver::result::mem_pool::get_attribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            &mut v as *mut u64 as *mut std::ffi::c_void,
        )
        .ok()?;
    }
    Some(v as f64 / (1024.0 * 1024.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::compile_ptx;

    /// End-to-end toolchain check: compile a trivial vector-add kernel with
    /// NVRTC, upload two host vectors, launch, download, and verify the result.
    /// If this passes, `cudarc` + NVRTC + the driver are all working and we can
    /// start porting the real ops kernels.
    #[test]
    fn vector_add_roundtrip() {
        let Some(gpu) = test_gpu() else { return };

        const SRC: &str = r#"
extern "C" __global__ void vadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"#;
        let ptx = compile_ptx(SRC).expect("NVRTC compile");
        let module = gpu.context.load_module(ptx).expect("load module");
        let vadd = module.load_function("vadd").expect("load function");

        let n = 1024usize;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (2 * i) as f32).collect();

        let da = gpu.stream.clone_htod(&a).unwrap();
        let db = gpu.stream.clone_htod(&b).unwrap();
        let mut dc = gpu.stream.alloc_zeros::<f32>(n).unwrap();

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i32 = n as i32;
        let mut launch = gpu.stream.launch_builder(&vadd);
        launch.arg(&da).arg(&db).arg(&mut dc).arg(&n_i32);
        unsafe { launch.launch(cfg) }.unwrap();

        let c = gpu.stream.clone_dtoh(&dc).unwrap();
        for i in 0..n {
            assert_eq!(c[i], a[i] + b[i], "mismatch at {i}");
        }
    }
}
