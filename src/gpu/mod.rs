//! CUDA backend (feature `cuda`).

use std::sync::Arc;

use cudarc::cublas::CudaBlas;
use cudarc::cublas::sys::{cublasMath_t, cublasSetMathMode};
use cudarc::cublaslt::CudaBlasLT;
use cudarc::driver::{CudaContext, CudaStream};

pub mod arena;
pub mod audio;
pub mod audio_sample;
pub mod audio_train;
pub mod bf16;
pub mod block;
pub mod buf;
pub mod flat;
pub mod gtensor;
pub mod hierarchical;
pub mod kernels;
pub mod linear;
pub mod lm;
pub mod mlstm;
pub mod offload;
pub mod ops;
pub mod profile;
pub mod rms_norm;
pub mod slstm;
pub mod temp;
pub mod train;
pub mod word_groups;

pub use arena::{ParamArena, ParamKind, ParamSlot};
pub use bf16::Slab;
pub use buf::{Buf, SlabSlot};
pub use gtensor::GTensor;
use kernels::Kernels;
pub use offload::{Frame, FrameStack};
pub use temp::{Temp, TempCache};

/// Whether host waits park the thread instead of spinning. On unless
/// `GPU_SPIN_SYNC` is set, which restores the driver's default busy-wait.
pub fn blocking_sync() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("GPU_SPIN_SYNC").is_err())
}

/// Flags for an event the host will wait on with `synchronize`. The context's
/// scheduling policy does not reach `cuEventSynchronize` — an event spins unless it
/// was *created* with `CU_EVENT_BLOCKING_SYNC` — so host-waited events need this.
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
    /// cuBLASLt handle, for GEMMs that fold the bias into the epilogue instead of
    /// seeding it with `broadcast_row` (see `Linear::forward_raw`). Shares `stream`
    /// with `blas`, so the two stay in issue order.
    pub blas_lt: Arc<CudaBlasLT>,
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
    /// 32-bit registers one SM's file holds. Shared memory is not the only thing
    /// that can keep a block off an SM entirely, and a cooperative launch that
    /// cannot place one block deadlocks rather than running slowly — see
    /// `ops::fused_fwd_threads_for`.
    pub regs_per_sm: usize,
    /// One counter per column tile for the banded `col_sum_part`, whose last block
    /// per tile folds the bands. All-zero between launches; shared by clones, which
    /// is sound because they share `stream` too.
    pub col_sum_tickets: Arc<cudarc::driver::CudaSlice<u32>>,
}

impl Gpu {
    /// Initialise CUDA device 0. Returns an error string (rather than panicking)
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
        let blas_lt =
            CudaBlasLT::new(stream.clone()).map_err(|e| format!("cuBLASLt init failed: {e:?}"))?;
        let kernels = Kernels::load(&context)?;
        let max_shared_optin = context
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )
            .map_err(|e| format!("querying max shared memory failed: {e:?}"))? as usize;
        let sm_count = context
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )
            .map_err(|e| format!("querying SM count failed: {e:?}"))?
            as usize;
        let regs_per_sm = context
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR)
            .map_err(|e| format!("querying register file size failed: {e:?}"))? as usize;
        let col_sum_tickets = stream
            .alloc_zeros::<u32>(ops::COL_SUM_TICKETS)
            .map_err(|e| format!("allocating col_sum tickets failed: {e:?}"))?;
        Ok(Self {
            context,
            stream,
            blas: Arc::new(blas),
            blas_lt: Arc::new(blas_lt),
            kernels: Arc::new(kernels),
            max_shared_optin,
            sm_count,
            regs_per_sm,
            col_sum_tickets: Arc::new(col_sum_tickets),
        })
    }
}

/// Let cuBLAS run our f32 GEMMs on the tensor cores in TF32.
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
fn test_gpu() -> Option<TestGpu> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
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

/// Hand every cached-but-unused block in the async pool back to the driver.
pub fn trim_pool(gpu: &Gpu) {
    gpu.stream.synchronize().expect("sync before trim");
    unsafe {
        let Ok(dev) = cudarc::driver::result::device::get(0) else {
            return;
        };
        let Ok(pool) = cudarc::driver::result::device::get_default_mem_pool(dev) else {
            return;
        };
        let _ = cudarc::driver::result::mem_pool::trim_to(pool, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::compile_ptx;

    /// End-to-end toolchain check: compile a trivial vector-add kernel with
    /// NVRTC, upload two host vectors, launch, download, and verify the result.
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
