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

pub mod block;
pub mod dtensor;
pub mod flat;
pub mod hierarchical;
pub mod kernels;
pub mod linear;
pub mod lm;
pub mod mlstm;
pub mod ops;
pub mod rms_norm;
pub mod slstm;
pub mod train;

pub use dtensor::DTensor;
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

/// A live CUDA device: the context, its default stream, a cuBLAS handle, and the
/// NVRTC-compiled kernel set. Cheap to clone (every field is an `Arc`). Created
/// once via [`Gpu::new`].
#[derive(Clone)]
pub struct Gpu {
    pub context: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub blas: Arc<CudaBlas>,
    pub kernels: Arc<Kernels>,
}

impl Gpu {
    /// Initialise CUDA device 0. Returns an error string (rather than panicking)
    /// so callers can fall back to the CPU path when no GPU is present.
    ///
    /// The work goes on an explicitly created stream, not `default_stream()`:
    /// cudarc's default stream is the *legacy* NULL stream, and CUDA refuses to
    /// capture that one (`cuStreamBeginCapture` errors on it). The sLSTM's
    /// per-timestep loop is captured as a graph (see `gpu::slstm`), so the whole
    /// backend has to live on a capturable stream. Everything is submitted to this
    /// one stream in issue order, so ordering semantics are unchanged.
    pub fn new() -> Result<Self, String> {
        let context = CudaContext::new(0).map_err(|e| format!("CUDA init failed: {e:?}"))?;
        // Leaving the default stream also puts cudarc into "multi stream mode", where
        // every `CudaSlice` records a CUDA event on each use so that buffers shared
        // across streams stay ordered. We submit everything to the one stream below,
        // in issue order, so that bookkeeping buys nothing and costs host time on the
        // per-launch path we are here to shorten. It is also a capture hazard: a
        // captured launch may not wait on an event recorded outside the capture.
        //
        // SAFETY: the contract is "the caller manages stream synchronization". There
        // is a single stream, so program order *is* the synchronization. Must happen
        // before any allocation — the flag only affects slices created after it.
        unsafe { context.disable_event_tracking() };
        let stream = context
            .new_stream()
            .map_err(|e| format!("stream creation failed: {e:?}"))?;
        let blas = CudaBlas::new(stream.clone()).map_err(|e| format!("cuBLAS init failed: {e:?}"))?;
        set_tf32(&blas)?;
        let kernels = Kernels::load(&context)?;
        Ok(Self { context, stream, blas: Arc::new(blas), kernels: Arc::new(kernels) })
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
/// oracle — and measures at ~1.35x on the GEMMs but only ~4% on a training step,
/// because the step is dominated by the backbone's sLSTM, not by matmul. That is a
/// poor trade to take silently, so it is a switch rather than a default.
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
/// at a time**. `cargo test` runs tests on parallel threads, and CUDA stream capture
/// (`gpu::slstm`) does not tolerate other threads allocating on the same context
/// while a capture is open — it intermittently produced a corrupt graph and a
/// `CUBLAS_STATUS_EXECUTION_FAILED` in whichever test got unlucky. Serializing here
/// keeps `cargo test --features cuda` honest without a `--test-threads=1` incantation
/// that a future runner would forget.
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

        let da = gpu.stream.memcpy_stod(&a).unwrap();
        let db = gpu.stream.memcpy_stod(&b).unwrap();
        let mut dc = gpu.stream.alloc_zeros::<f32>(n).unwrap();

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i32 = n as i32;
        let mut launch = gpu.stream.launch_builder(&vadd);
        launch.arg(&da).arg(&db).arg(&mut dc).arg(&n_i32);
        unsafe { launch.launch(cfg) }.unwrap();

        let c = gpu.stream.memcpy_dtov(&dc).unwrap();
        for i in 0..n {
            assert_eq!(c[i], a[i] + b[i], "mismatch at {i}");
        }
    }
}
