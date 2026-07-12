//! A device-resident tensor: the shape metadata of a host [`Tensor`] plus a
//! `CudaSlice<f32>` living in GPU memory.
//!
//! This is the "resident device buffer" of the bring-up plan. Rather than bolt
//! a `CudaSlice` onto the host `Tensor` (which derives `Clone`/`PartialEq` and
//! is used everywhere, incl. serialization), `DTensor` is a parallel type: data
//! stays on the GPU across a chain of ops and only crosses the PCIe bus at the
//! explicit `from_host` / `to_host` boundaries. Layers will hold `DTensor`s on
//! the GPU path exactly where they hold `Tensor`s on the CPU path.

use cudarc::driver::CudaSlice;

use super::Gpu;
use crate::tensor::{MAX_RANK, Tensor};

/// A dense, row-major, contiguous `f32` tensor resident in device memory.
/// Shape layout mirrors [`Tensor`]: only `shape[..rank]` are meaningful.
pub struct DTensor {
    pub shape: [usize; MAX_RANK],
    pub rank: usize,
    pub buf: CudaSlice<f32>,
}

impl DTensor {
    /// Upload a host tensor to the device (blocking H2D copy on the stream).
    pub fn from_host(gpu: &Gpu, t: &Tensor) -> Self {
        let buf = gpu.stream.memcpy_stod(&t.data).expect("H2D upload");
        Self { shape: t.shape, rank: t.rank, buf }
    }

    /// Download back to a host tensor (blocking D2H copy on the stream).
    pub fn to_host(&self, gpu: &Gpu) -> Tensor {
        let data = gpu.stream.memcpy_dtov(&self.buf).expect("D2H download");
        Tensor { shape: self.shape, rank: self.rank, data }
    }

    /// Allocate a zeroed device tensor of the given shape.
    pub fn zeros(gpu: &Gpu, dims: &[usize]) -> Self {
        let mut t = Self::uninit(gpu, dims);
        gpu.stream.memset_zeros(&mut t.buf).expect("memset");
        t
    }

    /// Allocate an **uninitialized** device tensor. Cheaper than [`zeros`] (skips
    /// the memset kernel); the caller must fully overwrite it before any read —
    /// used for op outputs that a kernel/GEMM writes in their entirety.
    pub fn uninit(gpu: &Gpu, dims: &[usize]) -> Self {
        assert!(dims.len() <= MAX_RANK, "rank {} exceeds MAX_RANK", dims.len());
        let n: usize = dims.iter().product();
        // SAFETY: contract above — every element is written before it is read.
        let buf = unsafe { gpu.stream.alloc::<f32>(n) }.expect("device alloc");
        let mut shape = [0usize; MAX_RANK];
        shape[..dims.len()].copy_from_slice(dims);
        Self { shape, rank: dims.len(), buf }
    }

    /// Zero this tensor's buffer in place (memset), reusing the existing device
    /// allocation — used to clear gradient accumulators between optimizer steps
    /// without freeing/reallocating.
    pub fn zero_(&mut self, gpu: &Gpu) {
        gpu.stream.memset_zeros(&mut self.buf).expect("memset");
    }

    /// Device-to-device copy: a fresh device tensor with the same shape and
    /// contents. Used to save a layer's forward input for its backward without a
    /// round-trip through host memory.
    pub fn dup(&self, gpu: &Gpu) -> Self {
        let buf = gpu.stream.clone_dtod(&self.buf).expect("device clone");
        Self { shape: self.shape, rank: self.rank, buf }
    }

    /// Reinterpret the shape without touching the buffer (metadata-only, zero
    /// copy) — the data is contiguous row-major, so `[B, T, H]` ↔ `[B·T, H]` is
    /// just a rank/shape rewrite. Consumes `self` to make the move-not-alias
    /// explicit (the old shape is gone). Panics if the element count changes.
    pub fn reshaped(mut self, dims: &[usize]) -> Self {
        assert!(dims.len() <= MAX_RANK, "reshape rank {} exceeds MAX_RANK", dims.len());
        let n: usize = dims.iter().product();
        assert_eq!(n, self.buf.len(), "reshape changes element count");
        self.shape = [0usize; MAX_RANK];
        self.shape[..dims.len()].copy_from_slice(dims);
        self.rank = dims.len();
        self
    }

    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.shape[..self.rank]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buf.len() == 0
    }

    /// Rows of a rank-2 device tensor.
    #[inline]
    pub fn rows(&self) -> usize {
        assert_eq!(self.rank, 2, "rows(): tensor is rank {}", self.rank);
        self.shape[0]
    }

    /// Columns of a rank-2 device tensor.
    #[inline]
    pub fn cols(&self) -> usize {
        assert_eq!(self.rank, 2, "cols(): tensor is rank {}", self.rank);
        self.shape[1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_device_roundtrip() {
        let Some(gpu) = super::super::test_gpu() else { return };
        let t = Tensor::random(&[7, 13], 1.0);
        let d = DTensor::from_host(&gpu, &t);
        let back = d.to_host(&gpu);
        assert_eq!(t, back, "round-trip through device memory changed the data");
    }
}
