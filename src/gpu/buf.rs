//! Layer-owned device buffers: allocate once, reuse forever.
//!
//! Every GPU layer used to return its output as a freshly allocated `DTensor`,
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
//! # Why the address stability matters
//!
//! A captured CUDA graph bakes the raw device pointer of every buffer its nodes
//! touch into the graph itself, so replaying it is only correct while those
//! buffers are still at the addresses they had at capture time. `gpu::slstm`
//! discovered this the hard way — see the use-after-free documented in
//! `SLstm::forward` — and grew a private `take_uninit` to pin its buffers down.
//! `Buf` is that mechanism, generalized so every layer gets it.
//!
//! # Contract
//!
//! [`Buf::get`] returns **uninitialised** memory when it reallocates and **stale**
//! memory (the previous call's contents) when it reuses. Callers must fully
//! overwrite it before reading — which is what an op that writes its whole output
//! does anyway. Use [`Buf::get_zeroed`] when the buffer is accumulated into rather
//! than overwritten.

use super::{DTensor, Gpu};

/// A device buffer owned by a layer across calls, resized only when the shape
/// it is asked for changes.
///
/// Starts empty; the first [`get`](Self::get) allocates. Holding this rather than
/// returning fresh tensors is what keeps a layer's output address stable, which
/// is what makes graph replay legal and keeps the allocator off the hot path.
#[derive(Default)]
pub struct Buf {
    slot: Option<DTensor>,
}

impl Buf {
    /// An empty slot. The first [`get`](Self::get) does the allocation.
    pub const fn new() -> Self {
        Self { slot: None }
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
    pub fn get(&mut self, gpu: &Gpu, dims: &[usize]) -> &mut DTensor {
        let n: usize = dims.iter().product();
        match &mut self.slot {
            Some(t) if t.len() == n => t.reshape_to(dims),
            _ => self.slot = Some(DTensor::uninit(gpu, dims)),
        }
        self.slot.as_mut().expect("just filled")
    }

    /// Like [`get`](Self::get) but zeroed, reusing the allocation when the shape
    /// matches (an in-place memset, no realloc). For buffers that are accumulated
    /// into rather than fully overwritten.
    pub fn get_zeroed(&mut self, gpu: &Gpu, dims: &[usize]) -> &mut DTensor {
        let n: usize = dims.iter().product();
        match &mut self.slot {
            Some(t) if t.len() == n => {
                t.reshape_to(dims);
                t.zero_(gpu);
            }
            _ => self.slot = Some(DTensor::zeros(gpu, dims)),
        }
        self.slot.as_mut().expect("just filled")
    }

    /// Copy `src` into this slot and return it. The shape follows `src`, so this
    /// is the owned-buffer replacement for `src.dup(gpu)` — same result, but
    /// reusing this layer's allocation instead of making a new one.
    pub fn copy_of(&mut self, gpu: &Gpu, src: &DTensor) -> &mut DTensor {
        let dst = self.get(gpu, src.dims());
        gpu.stream
            .memcpy_dtod(&src.buf, &mut dst.buf)
            .expect("Buf::copy_of");
        dst
    }

    /// The current contents, if this slot has ever been filled. Used by a
    /// backward that needs what the forward saved here.
    pub fn as_ref(&self) -> Option<&DTensor> {
        self.slot.as_ref()
    }

    /// The saved tensor, panicking with `what` if the forward never ran.
    pub fn expect(&self, what: &str) -> &DTensor {
        self.slot.as_ref().expect(what)
    }

    /// Release the allocation. Only for a layer being torn down or deliberately
    /// shrunk — the point of a `Buf` is to keep its memory across calls.
    pub fn clear(&mut self) {
        self.slot = None;
    }
}

/// A pool of scratch buffers, recycled by size within a layer.
///
/// A [`Buf`] is the right tool for a value that must *persist* — a forward output
/// its backward will read. Most of what a layer allocates is not that: it is a
/// temporary consumed by the next op or two and then dead. Giving each of those
/// its own permanent `Buf` would pin every temporary's peak simultaneously, which
/// is how a naive owned-buffer conversion ends up using *more* memory than the
/// allocator it replaced.
///
/// A `Pool` instead hands a temporary out with [`take`](Self::take) and gets it
/// back with [`put`](Self::put). A returned buffer goes on a free list keyed by
/// element count, so the next request for that size reuses it instead of
/// allocating. The pool therefore converges on the *high-water mark of buffers
/// live at once*, not the sum of every temporary the layer ever creates.
///
/// Buffers are never freed while the pool lives, so a steady shape allocates
/// nothing after the first pass — while a shape that grows reallocates only the
/// difference.
///
/// # Discipline
///
/// [`put`] must be called once the value is dead, and the buffer must not be read
/// afterwards — it may be handed to an unrelated request on the very next `take`.
/// [`scope`](Self::scope) is the safe form where a temporary's life fits a block.
#[derive(Default)]
pub struct Pool {
    /// Free buffers, grouped by element count. Small map: a layer touches a
    /// handful of distinct sizes, so a linear scan beats hashing.
    free: Vec<(usize, Vec<DTensor>)>,
    /// Buffers currently out on loan — see [`outstanding`](Self::outstanding).
    lent: usize,
}

impl Pool {
    pub const fn new() -> Self {
        Self {
            free: Vec::new(),
            lent: 0,
        }
    }

    /// A scratch buffer shaped `dims`, recycled from the free list when one of
    /// that element count is available, else freshly allocated.
    ///
    /// Contents are **undefined** — a recycled buffer still holds whatever the
    /// previous user left. Write it in full before reading, or use
    /// [`take_zeroed`](Self::take_zeroed).
    pub fn take(&mut self, gpu: &Gpu, dims: &[usize]) -> DTensor {
        self.lent += 1;
        let n: usize = dims.iter().product();
        for (size, bufs) in self.free.iter_mut() {
            if *size == n {
                if let Some(mut t) = bufs.pop() {
                    t.reshape_to(dims);
                    return t;
                }
                break;
            }
        }
        DTensor::uninit(gpu, dims)
    }

    /// Like [`take`](Self::take) but zeroed, for a buffer that is accumulated
    /// into rather than fully overwritten.
    pub fn take_zeroed(&mut self, gpu: &Gpu, dims: &[usize]) -> DTensor {
        let mut t = self.take(gpu, dims);
        t.zero_(gpu);
        t
    }

    /// Return a buffer to the pool. It must be dead: the next [`take`](Self::take)
    /// of the same size may hand it to an unrelated caller.
    ///
    /// Only give back what [`take`](Self::take) handed out. Donating buffers that
    /// were allocated elsewhere makes the free list grow by however many arrive
    /// each pass — the pool stops being a fixed working set and becomes a leak.
    /// [`outstanding`](Self::outstanding) is the invariant that catches this: it
    /// must return to zero, not climb, between passes.
    pub fn put(&mut self, t: DTensor) {
        debug_assert!(
            self.lent > 0,
            "Pool::put of a buffer this pool never lent — see the note on `put`"
        );
        self.lent = self.lent.saturating_sub(1);
        let n = t.len();
        for (size, bufs) in self.free.iter_mut() {
            if *size == n {
                bufs.push(t);
                return;
            }
        }
        self.free.push((n, vec![t]));
    }

    /// How many buffers are currently out on loan. Zero between passes if every
    /// `take` is matched by a `put`; a value that climbs pass over pass means
    /// buffers are being dropped instead of returned (memory the pool can never
    /// reuse), and a `put` without a matching `take` trips the assert above.
    pub fn outstanding(&self) -> usize {
        self.lent
    }

    /// Return several buffers at once.
    pub fn put_all<I: IntoIterator<Item = DTensor>>(&mut self, ts: I) {
        for t in ts {
            self.put(t);
        }
    }

    /// Run `f` with a scratch buffer, returning it to the pool afterwards.
    ///
    /// The safe form of take/put: the buffer cannot outlive the closure, so it
    /// cannot be read after being recycled.
    pub fn scope<R>(
        &mut self,
        gpu: &Gpu,
        dims: &[usize],
        f: impl FnOnce(&mut Self, &mut DTensor) -> R,
    ) -> R {
        let mut t = self.take(gpu, dims);
        let r = f(self, &mut t);
        self.put(t);
        r
    }

    /// Total elements currently held on the free list — the pool's retained
    /// memory, in f32s. For tests and memory reporting.
    pub fn pooled_elems(&self) -> usize {
        self.free
            .iter()
            .map(|(size, bufs)| size * bufs.len())
            .sum()
    }

    /// Drop every pooled buffer, releasing the memory.
    pub fn clear(&mut self) {
        self.free.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point: asking for the same shape twice must not reallocate, and
    /// the device pointer must not move. Pointer stability is what graph capture
    /// depends on, so this is the property to pin.
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

    /// The pool's reason for existing: a temporary returned after use must be
    /// handed back out rather than reallocated, so N sequential temporaries of one
    /// size cost ONE buffer — not N.
    #[test]
    fn pool_recycles_a_returned_buffer() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        use cudarc::driver::DevicePtr;
        let addr = |t: &DTensor| t.buf.device_ptr(&gpu.stream).0;
        let mut pool = Pool::new();

        let a = pool.take(&gpu, &[8, 16]);
        let pa = addr(&a);
        pool.put(a);

        // Same size again: must be the very same allocation.
        let b = pool.take(&gpu, &[8, 16]);
        assert_eq!(pa, addr(&b), "a returned buffer must be recycled");
        // ...and while it is out on loan, a second request cannot alias it.
        let c = pool.take(&gpu, &[8, 16]);
        assert_ne!(addr(&b), addr(&c), "two live buffers must be distinct");
        pool.put_all([b, c]);

        // Two were live at once, so the pool retains exactly two.
        assert_eq!(pool.pooled_elems(), 2 * 128);
    }

    /// Reuse is by element count, so a `[4, 32]` temporary can serve a later
    /// `[128]` one — the buffers are contiguous and shape is only metadata.
    #[test]
    fn pool_reuses_across_shapes_of_equal_size() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        use cudarc::driver::DevicePtr;
        let addr = |t: &DTensor| t.buf.device_ptr(&gpu.stream).0;
        let mut pool = Pool::new();

        let a = pool.take(&gpu, &[4, 32]);
        let pa = addr(&a);
        pool.put(a);

        let b = pool.take(&gpu, &[128]);
        assert_eq!(pa, addr(&b), "same element count must recycle");
        assert_eq!(b.dims(), [128], "and must come back at the requested shape");
    }

    /// `scope` returns the buffer automatically, so a loop of scoped temporaries
    /// never grows the pool past one buffer.
    #[test]
    fn pool_scope_returns_the_buffer() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let mut pool = Pool::new();
        for _ in 0..5 {
            pool.scope(&gpu, &[64], |_, t| {
                assert_eq!(t.len(), 64);
            });
        }
        assert_eq!(pool.pooled_elems(), 64, "scoped temporaries must not stack up");
    }

    /// `get` preserves contents on reuse (callers overwrite); `get_zeroed` clears.
    #[test]
    fn get_keeps_contents_and_get_zeroed_clears() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let mut buf = Buf::new();

        let t = buf.get_zeroed(&gpu, &[2, 3]);
        let src = DTensor::from_host(&gpu, &crate::tensor::Tensor::new(&[2, 3], vec![1.0; 6]));
        gpu.stream.memcpy_dtod(&src.buf, &mut t.buf).unwrap();
        assert_eq!(buf.expect("filled").to_host(&gpu).data, vec![1.0; 6]);

        // Same shape via `get`: the allocation is reused, so the data survives.
        assert_eq!(buf.get(&gpu, &[2, 3]).to_host(&gpu).data, vec![1.0; 6]);
        // `get_zeroed` memsets it in place.
        assert_eq!(buf.get_zeroed(&gpu, &[2, 3]).to_host(&gpu).data, vec![0.0; 6]);
    }
}
