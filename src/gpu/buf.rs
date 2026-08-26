//! Layer-owned device buffers: allocate once, reuse forever.
//!
//! Every GPU layer used to return its output as a freshly allocated `GTensor<f32>`,
//! so a training step allocated and freed hundreds of device buffers — one per
//! op, per layer, per window. cudarc routes those through `cuMemAllocAsync` on a
//! memory pool, so each one is cheap in isolation, but they are not free: they
//! cost host time on the critical path (the backbone is launch-bound, see
//! `gpu::slstm::fwd_loop`), they make device memory-in-use drift upward across
//! windows as the pool retains freed blocks, and — the part that has actually
//! caused bugs — they mean a buffer's device address changes from call to call.
//!
//! A [`Buf`] is a slot a layer owns for the whole of its life. [`Buf::get`] hands
//! back the existing allocation whenever the requested shape matches, and only
//! reallocates when it does not. In steady state (the same shapes recurring, as
//! in a training loop) it never allocates at all.
//!
//! # Why reuse matters
//!
//! A step issues thousands of ops, and a layer that returns a fresh tensor per call
//! puts an allocate/free pair on the hot path for each of them. `gpu::slstm` grew a
//! private `take_uninit` for its own buffers first; `Buf` is that mechanism,
//! generalized so every layer gets it.
//!
//! # Contract
//!
//! [`Buf::get`] returns **uninitialised** memory when it reallocates and **stale**
//! memory (the previous call's contents) when it reuses. Callers must fully
//! overwrite it before reading — which is what an op that writes its whole output
//! does anyway. Use [`Buf::get_zeroed`] when the buffer is accumulated into rather
//! than overwritten.

use super::{GTensor, Gpu, ops};

/// How much larger than the request a retained allocation may be before it is
/// dropped and replaced with a right-sized one.
///
/// Reuse-if-it-fits alone makes every slot ratchet to the largest window ever seen:
/// real windows vary (`MIN_WORDS_PER_SEQ`..`WORDS_PER_SEQ`, and one encoder/decoder
/// rectangle per word-length bucket), so a single unusually large window permanently
/// inflates every buffer behind it. Measured on the real `hg` path at
/// `WORDS_PER_SEQ = 2048`: device memory sat flat for many windows, then one larger
/// window walked it from 14.2 GB to 16.5 GB and aborted.
///
/// 4x is deliberately loose. The point is to cap the ratchet, not to chase an exact
/// fit — reallocating whenever a window is merely a bit smaller would put the
/// allocator back on the hot path, which is what `Buf` exists to avoid. At 4x a slot
/// still absorbs the ordinary window-to-window spread without a single free.
const RETAIN_SLACK: usize = 4;

/// Whether an existing `capacity` should be kept for a request of `want` elements.
#[inline]
fn fits(capacity: usize, want: usize) -> bool {
    capacity >= want && capacity <= want.saturating_mul(RETAIN_SLACK)
}

/// Round an allocation up to a **size class**, so near-identical shapes share one
/// buffer instead of each pinning its own.
///
/// `RETAIN_SLACK` alone bounds how oversized any single reuse may be, but it does not
/// bound *how many distinct sizes* the free list accumulates — and that is the leak
/// that actually aborts a run. Real windows vary continuously (word count, and one
/// encoder/decoder rectangle per length bucket: 14-16 buckets per window measured on
/// the real `hg` path), so essentially every window asks for a handful of sizes no
/// previous window used. Each is individually within slack of something, so `trim`
/// keeps it, and the free list grows without bound: measured over 40 varying windows
/// it went 60 distinct sizes / 182 buffers -> 238 / 542, and device memory climbed
/// 7571 MB -> 10307 MB in lockstep. Real training showed the same shape, 12.9 -> 16.4
/// GB, and then OOMed.
///
/// Quantizing collapses that. `1/16`-granular classes (mantissa rounded up within
/// each power of two) waste at most 6.25% per buffer and reduce the reachable size
/// count to ~16 per octave — a constant, independent of how many distinct shapes the
/// corpus produces.
#[inline]
fn size_class(n: usize) -> usize {
    // Small allocations are left exact: the waste would be proportionally large and
    // there are few enough distinct small sizes for them not to be the problem.
    const MIN_CLASS: usize = 1024;
    if n <= MIN_CLASS {
        return n;
    }
    let bits = usize::BITS - n.leading_zeros(); // position of the top set bit
    let shift = bits.saturating_sub(5); // keep 4 mantissa bits below the top
    let step = 1usize << shift;
    n.div_ceil(step) * step
}

/// A device buffer owned by a layer across calls, resized only when the shape
/// it is asked for changes.
///
/// Starts empty; the first [`get`](Self::get) allocates. Holding this rather than
/// returning fresh tensors keeps the allocator off the hot path and, as a side
/// effect, a layer's output address stable across calls.
#[derive(Default)]
pub struct Buf {
    slot: Option<GTensor<f32>>,
}

impl Buf {
    /// An empty slot. The first [`get`](Self::get) does the allocation.
    pub const fn new() -> Self {
        Self { slot: None }
    }

    /// Device bytes this slot is holding. Diagnostic — see
    /// [`Hierarchical::retained_report`](crate::gpu::hierarchical::Hierarchical::retained_report).
    ///
    /// Counts **capacity**, not the shape the buffer was last used at: reuse here is
    /// by capacity, so the allocation is what occupies memory.
    pub fn retained_bytes(&self) -> usize {
        self.slot.as_ref().map_or(0, |t| t.capacity() * 4)
    }

    /// The buffer, shaped `dims` — reusing the existing allocation when it
    /// already has that shape, else allocating a fresh (uninitialised) one.
    ///
    /// The contents are **not** cleared: on reuse they are the previous call's
    /// data, on reallocation they are uninitialised. The caller must write every
    /// element it later reads. For an accumulator, use
    /// [`get_zeroed`](Self::get_zeroed).
    /// Reuse is by **element count**, not by exact shape: a buffer already holding
    /// the right number of elements is reshaped in place (metadata only) rather
    /// than reallocated. That is what lets one owned buffer serve both the
    /// `[B, T, H]` and `[N, H]` views of the same activations without a copy.
    pub fn get(&mut self, gpu: &Gpu, dims: &[usize]) -> &mut GTensor<f32> {
        let n: usize = dims.iter().product();
        match &mut self.slot {
            // Reuse whenever the allocation is big enough, presenting it at the
            // asked-for shape. Requiring an exact match would reallocate on every
            // shape change, which for a varying window size is every call.
            //
            // But only while it is not *wildly* too big — see `RETAIN_SLACK`. A slot
            // that keeps every allocation it has ever been big enough for ratchets to
            // the largest window in the corpus and stays there.
            Some(t) if fits(t.capacity(), n) => t.shrink_to(dims),
            // Allocate at the size class so the next window's slightly-different shape
            // reuses this allocation instead of replacing it. See `size_class`.
            _ => {
                let mut t = GTensor::uninit(gpu, &[size_class(n)]);
                t.shrink_to(dims);
                self.slot = Some(t);
            }
        }
        self.slot.as_mut().expect("just filled")
    }

    /// Like [`get`](Self::get) but zeroed, reusing the allocation when the shape
    /// matches (an in-place memset, no realloc). For buffers that are accumulated
    /// into rather than fully overwritten.
    pub fn get_zeroed(&mut self, gpu: &Gpu, dims: &[usize]) -> &mut GTensor<f32> {
        let n: usize = dims.iter().product();
        match &mut self.slot {
            Some(t) if fits(t.capacity(), n) => {
                t.shrink_to(dims);
                t.zero_(gpu);
            }
            _ => {
                let mut t = GTensor::zeros(gpu, &[size_class(n)]);
                t.shrink_to(dims);
                self.slot = Some(t);
            }
        }
        self.slot.as_mut().expect("just filled")
    }

    /// Copy `src` into this slot and return it. The shape follows `src`, so this
    /// is the owned-buffer replacement for `src.dup(gpu)` — same result, but
    /// reusing this layer's allocation instead of making a new one.
    pub fn copy_of(&mut self, gpu: &Gpu, src: &GTensor<f32>) -> &mut GTensor<f32> {
        let n = src.len();
        let dst = self.get(gpu, src.dims());
        // Slice both sides to the shape: either may be a pooled buffer with slack
        // beyond it, and `memcpy_dtod` requires the two lengths to agree.
        gpu.stream
            .memcpy_dtod(&src.buf.slice(..n), &mut dst.buf.slice_mut(..n))
            .expect("Buf::copy_of");
        dst
    }

    /// The current contents, if this slot has ever been filled. Used by a
    /// backward that needs what the forward saved here.
    pub fn as_ref(&self) -> Option<&GTensor<f32>> {
        self.slot.as_ref()
    }

    /// The saved tensor, panicking with `what` if the forward never ran.
    pub fn expect(&self, what: &str) -> &GTensor<f32> {
        self.slot.as_ref().expect(what)
    }

    /// Move the tensor out of this slot, leaving it empty.
    ///
    /// For a caller that needs to *own* the buffer for a while — typically because it
    /// must be borrowed alongside other fields of the same struct, which a borrow of
    /// the whole slot would prevent. Pair with [`put`](Self::put) to give it back and
    /// keep the allocation for the next call.
    pub fn take(&mut self) -> Option<GTensor<f32>> {
        self.slot.take()
    }

    /// Put a tensor (back) into this slot, replacing any current one.
    pub fn put(&mut self, t: GTensor<f32>) {
        self.slot = Some(t);
    }

    /// Release the allocation. Only for a layer being torn down or deliberately
    /// shrunk — the point of a `Buf` is to keep its memory across calls.
    pub fn clear(&mut self) {
        self.slot = None;
    }
}

/// A [`Buf`] whose tensor is stored at the slab width — bf16 where the kernels were
/// built for it, fp32 otherwise.
///
/// Same contract as `Buf` (persist across calls, reuse by capacity, allocate at a size
/// class), for a value whose every reader takes it narrow. `zn` in a `Block` is the
/// case that motivated it: its norm writes it, two GEMMs and that norm's own backward
/// read it, and all four are happy with bf16 — so materializing it fp32 only to narrow
/// it again was a full extra pass over the tensor per reader.
#[derive(Default)]
pub struct SlabSlot {
    slot: Option<ops::SlabBuf>,
    /// Whether this slot narrows. Fixed for the slot's life, because a forward and its
    /// backward have to agree on the layout of what was written.
    bf16: bool,
}

impl SlabSlot {
    /// `bf16` is the caller's decision, not the kernels' — see
    /// [`ops::SlabBuf::new_width`].
    pub const fn new(bf16: bool) -> Self {
        Self { slot: None, bf16 }
    }

    /// Device bytes held, at this slab's actual element width. Diagnostic.
    pub fn retained_bytes(&self) -> usize {
        self.slot.as_ref().map_or(0, |t| t.retained_bytes())
    }

    /// The buffer at `dims`, reusing the allocation while it fits. See [`Buf::get`].
    pub fn get(&mut self, gpu: &Gpu, dims: &[usize]) -> &mut ops::SlabBuf {
        let n: usize = dims.iter().product();
        match &mut self.slot {
            Some(t) if fits(t.capacity(), n) => t.shrink_to(dims),
            _ => {
                let mut t = ops::SlabBuf::new_width(gpu, &[size_class(n)], self.bf16);
                t.shrink_to(dims);
                self.slot = Some(t);
            }
        }
        self.slot.as_mut().expect("just filled")
    }

    /// The saved slab, panicking with `what` if the forward never ran.
    pub fn expect(&self, what: &str) -> &ops::SlabBuf {
        self.slot.as_ref().expect(what)
    }

    pub fn take(&mut self) -> Option<ops::SlabBuf> {
        self.slot.take()
    }

    pub fn put(&mut self, t: ops::SlabBuf) {
        self.slot = Some(t);
    }

    pub fn clear(&mut self) {
        self.slot = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point: asking for the same shape twice must not reallocate, and
    /// the device pointer must not move.
    #[test]
    fn same_shape_reuses_the_allocation() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        use cudarc::driver::DevicePtr;
        let mut buf = Buf::new();
        let addr = |b: &mut Buf, dims: &[usize]| {
            let t = b.get(&gpu, dims);
            let (p, _sync) = t.buf.device_ptr(&gpu.stream);
            p
        };

        let p1 = addr(&mut buf, &[4, 8]);
        let p2 = addr(&mut buf, &[4, 8]);
        assert_eq!(p1, p2, "same shape must reuse the same device allocation");

        // A different shape must reallocate (the old buffer is the wrong size)...
        addr(&mut buf, &[4, 9]);
        // ...and coming back to the original shape is a fresh allocation too: the
        // slot only ever holds one buffer. What matters is that a STEADY shape is
        // stable, which is the case a training loop actually hits.
        let p4 = addr(&mut buf, &[4, 8]);
        let p5 = addr(&mut buf, &[4, 8]);
        assert_eq!(p4, p5, "shape is steady again, so the address must be too");
    }







    /// `get` preserves contents on reuse (callers overwrite); `get_zeroed` clears.
    #[test]
    fn get_keeps_contents_and_get_zeroed_clears() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let mut buf = Buf::new();

        let t = buf.get_zeroed(&gpu, &[2, 3]);
        let src = GTensor::from_host(&gpu, &crate::tensor::Tensor::new(&[2, 3], vec![1.0; 6]));
        gpu.stream.memcpy_dtod(&src.buf, &mut t.buf).unwrap();
        assert_eq!(buf.expect("filled").to_host(&gpu).data, vec![1.0; 6]);

        // Same shape via `get`: the allocation is reused, so the data survives.
        assert_eq!(buf.get(&gpu, &[2, 3]).to_host(&gpu).data, vec![1.0; 6]);
        // `get_zeroed` memsets it in place.
        assert_eq!(
            buf.get_zeroed(&gpu, &[2, 3]).to_host(&gpu).data,
            vec![0.0; 6]
        );
    }
}
