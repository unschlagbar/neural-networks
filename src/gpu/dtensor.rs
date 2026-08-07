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
        let buf = gpu.stream.clone_htod(&t.data).expect("H2D upload");
        Self {
            shape: t.shape,
            rank: t.rank,
            buf,
        }
    }

    /// Download back to a host tensor (blocking D2H copy on the stream).
    ///
    /// Copies the elements the SHAPE covers, which for a pooled buffer may be
    /// fewer than are allocated — the host tensor must match `dims()`, not the
    /// capacity behind it.
    pub fn to_host(&self, gpu: &Gpu) -> Tensor {
        let n = self.len();
        let data = gpu
            .stream
            .clone_dtoh(&self.buf.slice(..n))
            .expect("D2H download");
        Tensor {
            shape: self.shape,
            rank: self.rank,
            data,
        }
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
        assert!(
            dims.len() <= MAX_RANK,
            "rank {} exceeds MAX_RANK",
            dims.len()
        );
        let n: usize = dims.iter().product();
        // SAFETY: contract above — every element is written before it is read.
        let buf = unsafe { gpu.stream.alloc::<f32>(n) }.expect("device alloc");
        let mut shape = [0usize; MAX_RANK];
        shape[..dims.len()].copy_from_slice(dims);
        Self {
            shape,
            rank: dims.len(),
            buf,
        }
    }

    /// Zero this tensor's buffer in place (memset), reusing the existing device
    /// allocation — used to clear gradient accumulators between optimizer steps
    /// without freeing/reallocating.
    /// Zeroes the elements the SHAPE covers; a pooled buffer's slack beyond that
    /// is left alone, since nothing may read it anyway.
    pub fn zero_(&mut self, gpu: &Gpu) {
        let n = self.len();
        gpu.stream
            .memset_zeros(&mut self.buf.slice_mut(..n))
            .expect("memset");
    }

    /// Device-to-device copy: a fresh device tensor with the same shape and
    /// contents. Used to save a layer's forward input for its backward without a
    /// round-trip through host memory.
    pub fn dup(&self, gpu: &Gpu) -> Self {
        // Clone only what the shape covers: duplicating a pooled buffer's slack
        // would copy — and retain — memory the result never uses.
        let buf = gpu
            .stream
            .clone_dtod(&self.buf.slice(..self.len()))
            .expect("device clone");
        Self {
            shape: self.shape,
            rank: self.rank,
            buf,
        }
    }

    /// Device-to-device copy of `src`'s elements into this tensor.
    ///
    /// Both sides are sliced to `src`'s element count first. A pooled buffer can
    /// be larger than the shape it is presented at (see [`shrink_to`]), and
    /// `memcpy_dtod` requires both lengths to agree — copying the raw `buf`s would
    /// fail, or move slack that means nothing, the moment either side is pooled.
    ///
    /// [`shrink_to`]: Self::shrink_to
    pub fn copy_from(&mut self, gpu: &Gpu, src: &DTensor) {
        let n = src.len();
        assert!(
            n <= self.capacity(),
            "copy_from: {n} elements into a {} buffer",
            self.capacity()
        );
        gpu.stream
            .memcpy_dtod(&src.buf.slice(..n), &mut self.buf.slice_mut(..n))
            .expect("copy_from");
    }

    /// Reinterpret the shape without touching the buffer (metadata-only, zero
    /// copy) — the data is contiguous row-major, so `[B, T, H]` ↔ `[B·T, H]` is
    /// just a rank/shape rewrite. Consumes `self` to make the move-not-alias
    /// explicit (the old shape is gone). Panics if the element count changes.
    pub fn reshaped(mut self, dims: &[usize]) -> Self {
        assert!(
            dims.len() <= MAX_RANK,
            "reshape rank {} exceeds MAX_RANK",
            dims.len()
        );
        let n: usize = dims.iter().product();
        assert_eq!(n, self.len(), "reshape changes element count");
        self.shape = [0usize; MAX_RANK];
        self.shape[..dims.len()].copy_from_slice(dims);
        self.rank = dims.len();
        self
    }

    /// Reinterpret this tensor's shape **in place**, keeping the same device
    /// buffer (metadata-only, no copy).
    ///
    /// [`reshaped`](Self::reshaped) consumes `self`, which a layer-owned buffer
    /// borrowed as `&mut DTensor` cannot give up. This is the in-place form, used
    /// where an owned buffer must be seen as `[B,T,H]` by one op and `[N,H]` by
    /// the next. Panics if the element count would change.
    pub fn reshape_to(&mut self, dims: &[usize]) {
        assert!(
            dims.len() <= MAX_RANK,
            "reshape rank {} exceeds MAX_RANK",
            dims.len()
        );
        let n: usize = dims.iter().product();
        assert_eq!(n, self.len(), "reshape changes element count");
        self.shape = [0usize; MAX_RANK];
        self.shape[..dims.len()].copy_from_slice(dims);
        self.rank = dims.len();
    }

    /// Present this tensor at `dims`, which must fit **within** its allocation.
    ///
    /// [`reshape_to`](Self::reshape_to) insists the element count match exactly.
    /// This is the looser form for a pooled buffer that may be larger than the
    /// request: the shape metadata describes the used prefix, while the underlying
    /// `CudaSlice` keeps its full capacity, so returning it to the pool restores
    /// the whole buffer. The trailing elements are untouched and must not be read.
    pub fn shrink_to(&mut self, dims: &[usize]) {
        assert!(
            dims.len() <= MAX_RANK,
            "shrink rank {} exceeds MAX_RANK",
            dims.len()
        );
        let n: usize = dims.iter().product();
        assert!(
            n <= self.buf.len(),
            "shrink_to {n} elements exceeds the {} allocated",
            self.buf.len()
        );
        self.shape = [0usize; MAX_RANK];
        self.shape[..dims.len()].copy_from_slice(dims);
        self.rank = dims.len();
    }

    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.shape[..self.rank]
    }

    /// Number of elements the current **shape** describes.
    ///
    /// Not the allocation size: a buffer handed out by a `Pool` may be larger than
    /// the shape it was asked for (see [`shrink_to`](Self::shrink_to)). Every op
    /// sizes its launch from this, so it must track the shape — reporting capacity
    /// here would run kernels over the untouched tail.
    #[inline]
    pub fn len(&self) -> usize {
        self.shape[..self.rank].iter().product()
    }

    /// Elements actually allocated, which may exceed [`len`](Self::len) for a
    /// pooled buffer. Used by `Pool` to file a returned buffer under its true
    /// capacity rather than the shape it happened to be used at.
    #[inline]
    pub fn capacity(&self) -> usize {
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
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let t = Tensor::random(&[7, 13], 1.0);
        let d = DTensor::from_host(&gpu, &t);
        let back = d.to_host(&gpu);
        assert_eq!(t, back, "round-trip through device memory changed the data");
    }
}
