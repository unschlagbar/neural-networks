//! A device-resident tensor: the shape metadata of a host [`Tensor`] plus a [`GBuf`]
//! — an allocation in GPU memory, or a window into one.
//!
//! [`GTensor`] is generic in the element width. `GTensor<f32>` is what every
//! kernel and GEMM computes in; `GTensor<u16>` holds bf16, whose bits Rust has no
//! native type for (see [`bf16`](super::bf16)).
//!
//! This is the "resident device buffer" of the bring-up plan. Rather than bolt
//! a `CudaSlice` onto the host `Tensor` (which derives `Clone`/`PartialEq` and
//! is used everywhere, incl. serialization), `GTensor` is a parallel type: data
//! stays on the GPU across a chain of ops and only crosses the PCIe bus at the
//! explicit `from_host` / `to_host` boundaries. Layers hold `GTensor<f32>` on the
//! GPU path exactly where they hold `Tensor` on the CPU path.

use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use cudarc::driver::{
    CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, DeviceSlice, LaunchArgs,
    PushKernelArg, SyncOnDrop, ValidAsZeroBits, sys,
};

use super::Gpu;
use crate::tensor::{MAX_RANK, Tensor};

/// Records the device address of every `uninit` allocation, so a run can be checked
/// for pointer stability across steps — i.e. how much of the allocator traffic the
/// per-layer buffer reuse is actually catching.
///
/// Diagnostic only, off unless `GPU_PTR_PROBE=1`; `dump` prints per-step overlap.
pub mod ptr_probe {
    use super::{CudaSlice, Gpu};
    use std::sync::Mutex;
    use std::sync::OnceLock;

    static SEQ: OnceLock<Mutex<Vec<u64>>> = OnceLock::new();

    pub fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var("GPU_PTR_PROBE").is_ok_and(|v| v != "0"))
    }

    pub fn record<T>(gpu: &Gpu, buf: &CudaSlice<T>) {
        if !on() {
            return;
        }
        use cudarc::driver::DevicePtr;
        let (p, _g) = buf.device_ptr(&gpu.stream);
        SEQ.get_or_init(|| Mutex::new(Vec::new())).lock().unwrap().push(p);
    }

    /// Split the recorded addresses into `steps` equal parts and report, for each
    /// consecutive pair, how often the same allocation index yields the same address.
    pub fn dump(steps: usize) {
        if !on() {
            return;
        }
        let v = SEQ.get_or_init(|| Mutex::new(Vec::new())).lock().unwrap();
        let n = v.len() / steps;
        println!("ptr_probe: {} allocs total, {} per step", v.len(), n);
        for s in 1..steps {
            let (a, b) = (&v[(s - 1) * n..s * n], &v[s * n..(s + 1) * n]);
            let same = a.iter().zip(b).filter(|(x, y)| x == y).count();
            println!(
                "  step {} vs {}: {}/{} identical addresses ({:.1}%)",
                s - 1,
                s,
                same,
                n,
                100.0 * same as f64 / n as f64
            );
        }
    }
}

/// A device allocation, or a non-owning window into a larger one.
///
/// Every parameter lives in one contiguous [`ParamArena`](super::arena::ParamArena)
/// and each layer holds a window into it, so a tensor's storage is not always its
/// own. A window is a real `CudaSlice` at an offset inside the arena — freeing it
/// would hand the driver an interior pointer, so `owned` decides what drop does.
///
/// Derefs to `CudaSlice<T>` and carries the cudarc traits, so a window is passed
/// to kernels, cuBLAS and the copy helpers exactly like an allocation.
pub struct GBuf<T> {
    slice: ManuallyDrop<CudaSlice<T>>,
    owned: bool,
}

impl<T> GBuf<T> {
    fn owned(slice: CudaSlice<T>) -> Self {
        Self {
            slice: ManuallyDrop::new(slice),
            owned: true,
        }
    }

    /// A window of `len` elements at element offset `off` in `base`.
    ///
    /// The caller must keep `base` alive for as long as the window: the arena owns
    /// the storage and outlives every layer that views it.
    ///
    /// Windows of one allocation are independent handles, so they carry no ordering
    /// between them. That is exactly the contract `Gpu::new` already takes on with
    /// `disable_event_tracking` — one stream, program order — and re-enabling cudarc's
    /// event tracking would not extend to them.
    fn window(gpu: &Gpu, base: &CudaSlice<T>, off: usize, len: usize) -> Self {
        assert!(
            off + len <= base.len(),
            "window {off}..{} exceeds the {} element arena",
            off + len,
            base.len()
        );
        let (p, _g) = base.device_ptr(&gpu.stream);
        let at = p + (off * std::mem::size_of::<T>()) as u64;
        // SAFETY: `at` is inside a live allocation with `len` elements after it, and
        // the arena keeps it mapped; `Drop` below never frees a window.
        let slice = unsafe { gpu.stream.upgrade_device_ptr::<T>(at, len) };
        Self {
            slice: ManuallyDrop::new(slice),
            owned: false,
        }
    }

    /// Whether this buffer owns its allocation (false for an arena window).
    #[inline]
    pub fn is_owned(&self) -> bool {
        self.owned
    }
}

impl<T> Drop for GBuf<T> {
    fn drop(&mut self) {
        // SAFETY: the only take, and the field is never read again.
        let slice = unsafe { ManuallyDrop::take(&mut self.slice) };
        if !self.owned {
            // Gives up the handle without freeing what the arena owns.
            let _ = slice.leak();
        }
    }
}

impl<T> Deref for GBuf<T> {
    type Target = CudaSlice<T>;
    #[inline]
    fn deref(&self) -> &CudaSlice<T> {
        &self.slice
    }
}

impl<T> DerefMut for GBuf<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.slice
    }
}

impl<T> DeviceSlice<T> for GBuf<T> {
    fn len(&self) -> usize {
        self.slice.len()
    }
    fn stream(&self) -> &Arc<CudaStream> {
        DeviceSlice::stream(&*self.slice)
    }
}

impl<T> DevicePtr<T> for GBuf<T> {
    fn device_ptr<'a>(&'a self, stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        self.slice.device_ptr(stream)
    }
}

impl<T> DevicePtrMut<T> for GBuf<T> {
    fn device_ptr_mut<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        self.slice.device_ptr_mut(stream)
    }
}

unsafe impl<'a, 'b: 'a, T> PushKernelArg<&'b GBuf<T>> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b GBuf<T>) -> &mut Self {
        PushKernelArg::arg(self, &*arg.slice)
    }
}

unsafe impl<'a, 'b: 'a, T> PushKernelArg<&'b mut GBuf<T>> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b mut GBuf<T>) -> &mut Self {
        PushKernelArg::arg(self, &mut *arg.slice)
    }
}

/// A dense, row-major, contiguous tensor of `T` resident in device memory.
/// Shape layout mirrors [`Tensor`]: only `shape[..rank]` are meaningful.
///
/// The element type is a parameter because the two widths the model stores differ
/// only in it: `f32` is what everything computes in, `u16` is the bf16 storage slab.
/// Shape handling, pooling, arena windows and reshaping are identical for both, so
/// they are written once here; what genuinely differs — host round-trips, the cast
/// kernels — lives in the per-width impls.
pub struct GTensor<T> {
    pub shape: [usize; MAX_RANK],
    pub rank: usize,
    pub buf: GBuf<T>,
}

impl GTensor<f32> {
    /// Upload a host tensor to the device (blocking H2D copy on the stream).
    pub fn from_host(gpu: &Gpu, t: &Tensor) -> Self {
        let buf = gpu.stream.clone_htod(&t.data).expect("H2D upload");
        Self {
            shape: t.shape,
            rank: t.rank,
            buf: GBuf::owned(buf),
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
}

/// The bound is what any stored width has to satisfy: allocatable by cudarc and
/// zeroable by `memset`. `f32` and `u16` both are.
impl<T: DeviceRepr + ValidAsZeroBits> GTensor<T> {
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
        let buf = unsafe { gpu.stream.alloc::<T>(n) }.expect("device alloc");
        if ptr_probe::on() {
            ptr_probe::record(gpu, &buf);
        }
        let mut shape = [0usize; MAX_RANK];
        shape[..dims.len()].copy_from_slice(dims);
        Self {
            shape,
            rank: dims.len(),
            buf: GBuf::owned(buf),
        }
    }

    /// A tensor of shape `dims` viewing `base` from element offset `off`.
    ///
    /// Owns nothing: the storage stays the arena's, and this tensor's device address
    /// is fixed for as long as the arena lives. See [`super::arena`].
    pub fn view(gpu: &Gpu, base: &CudaSlice<T>, off: usize, dims: &[usize]) -> Self {
        assert!(
            dims.len() <= MAX_RANK,
            "rank {} exceeds MAX_RANK",
            dims.len()
        );
        let n: usize = dims.iter().product();
        let mut shape = [0usize; MAX_RANK];
        shape[..dims.len()].copy_from_slice(dims);
        Self {
            shape,
            rank: dims.len(),
            buf: GBuf::window(gpu, base, off, n),
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
            buf: GBuf::owned(buf),
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
    pub fn copy_from(&mut self, gpu: &Gpu, src: &Self) {
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
    /// borrowed as `&mut GTensor<f32>` cannot give up. This is the in-place form, used
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

    /// This tensor as `[N, F]` — the last axis kept, every leading axis folded into
    /// `N`. For position-wise ops (RMSNorm), which act on the last axis and do not
    /// care whether the caller is holding `[B, T, H]` or the flat `[N, H]`.
    ///
    /// Metadata only, and borrowing: unlike [`reshaped`](Self::reshaped) it does not
    /// consume the tensor, so a `&GTensor<f32>` argument can be viewed this way.
    #[inline]
    pub fn as_2d(&self) -> (usize, usize) {
        assert!(self.rank >= 1, "as_2d(): rank 0");
        let f = self.shape[self.rank - 1];
        (self.len() / f, f)
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
        let d = GTensor::from_host(&gpu, &t);
        let back = d.to_host(&gpu);
        assert_eq!(t, back, "round-trip through device memory changed the data");
    }
}
