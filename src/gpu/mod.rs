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
    pub fn new() -> Result<Self, String> {
        let context = CudaContext::new(0).map_err(|e| format!("CUDA init failed: {e:?}"))?;
        let stream = context.default_stream();
        let blas = CudaBlas::new(stream.clone()).map_err(|e| format!("cuBLAS init failed: {e:?}"))?;
        let kernels = Kernels::load(&context)?;
        Ok(Self { context, stream, blas: Arc::new(blas), kernels: Arc::new(kernels) })
    }
}

#[cfg(test)]
/// Shared test helper: a `Gpu` if the machine has one, else `None` (so GPU tests
/// self-skip on the dev box with no Nvidia card).
fn test_gpu() -> Option<Gpu> {
    match Gpu::new() {
        Ok(g) => Some(g),
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
