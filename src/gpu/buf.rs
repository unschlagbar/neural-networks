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

use super::{GTensor, Gpu};

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
    free: Vec<(usize, Vec<GTensor<f32>>)>,
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

    /// Device bytes held on the free list. Diagnostic — see
    /// [`Hierarchical::retained_report`](crate::gpu::hierarchical::Hierarchical::retained_report).
    ///
    /// Buffers currently on loan are not counted: they belong to whoever borrowed
    /// them, and after a completed step there should be none (`lent == 0`).
    pub fn retained_bytes(&self) -> usize {
        self.free
            .iter()
            .map(|(_, bufs)| bufs.iter().map(|t| t.capacity() * 4).sum::<usize>())
            .sum()
    }

    /// A scratch buffer holding at least `dims`, recycled from the free list when
    /// a large enough one is available, else freshly allocated.
    ///
    /// Reuse is by **capacity, not exact size**: the smallest free buffer that
    /// fits is taken and presented at the requested shape. Demanding an exact
    /// match instead makes the pool useless the moment shapes vary — real training
    /// windows run from `MIN_WORDS_PER_SEQ` up to `WORDS_PER_SEQ`, and the
    /// encoder/decoder run one rectangle per word length — so every distinct size
    /// would pin its own buffer forever and the pool would grow without bound.
    /// That was an out-of-memory abort in real runs while a fixed-size benchmark
    /// looked perfectly stable.
    ///
    /// Contents are **undefined** — a recycled buffer still holds whatever the
    /// previous user left, and may be larger than asked for. Write the requested
    /// region in full before reading, or use [`take_zeroed`](Self::take_zeroed).
    pub fn take(&mut self, gpu: &Gpu, dims: &[usize]) -> GTensor<f32> {
        self.lent += 1;
        let n: usize = dims.iter().product();
        // Best fit: the smallest buffer that still holds `n`. Picking the smallest
        // keeps the big ones available for the requests that actually need them.
        // Only buffers within `RETAIN_SLACK` of the request are candidates. A pooled
        // buffer sized for the corpus's largest window would otherwise be handed to
        // every small window that follows, keeping it alive forever — the same ratchet
        // `Buf` has, one level down.
        let mut best: Option<usize> = None;
        for (i, (size, bufs)) in self.free.iter().enumerate() {
            if fits(*size, n) && !bufs.is_empty() && best.is_none_or(|b| *size < self.free[b].0) {
                best = Some(i);
            }
        }
        if let Some(i) = best {
            let mut t = self.free[i].1.pop().expect("checked non-empty");
            // The buffer may be larger than requested; `reshape_to` insists on an
            // exact element count, so trim the view to the asked-for shape.
            t.shrink_to(dims);
            return t;
        }
        // Allocate at the SIZE CLASS, not the exact request, then present it at the
        // asked-for shape. That is what makes the free list converge: the next window's
        // slightly-different shape rounds to the same class and reuses this buffer
        // instead of adding another entry. See `size_class`.
        let mut t = GTensor::uninit(gpu, &[size_class(n)]);
        t.shrink_to(dims);
        t
    }

    /// Like [`take`](Self::take) but zeroed, for a buffer that is accumulated
    /// into rather than fully overwritten.
    pub fn take_zeroed(&mut self, gpu: &Gpu, dims: &[usize]) -> GTensor<f32> {
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
    ///
    /// Oversized buffers are filed as usual; [`trim`](Self::trim) is what evicts them,
    /// because only the caller knows when a pass boundary has been reached and the
    /// free list can safely be pruned.
    pub fn put(&mut self, t: GTensor<f32>) {
        debug_assert!(
            self.lent > 0,
            "Pool::put of a buffer this pool never lent — see the note on `put`"
        );
        // A parameter window would hand out the model's own weights as scratch.
        debug_assert!(t.buf.is_owned(), "Pool::put of an arena window");
        self.lent = self.lent.saturating_sub(1);
        // File under the ALLOCATION size, not the shape it was last used at — a
        // buffer handed out shrunk still owns its full capacity.
        let n = t.capacity();
        for (size, bufs) in self.free.iter_mut() {
            if *size == n {
                bufs.push(t);
                return;
            }
        }
        self.free.push((n, vec![t]));
    }

    /// Drop free buffers more than `RETAIN_SLACK`x larger than `want` elements.
    ///
    /// Call at a pass boundary with the size the *next* pass will ask for. `take`
    /// already refuses to hand out anything that oversized, so those buffers are dead
    /// weight from the moment a big window ends — this is what actually releases them.
    ///
    /// Without it, one unusually large window leaves the pool permanently inflated:
    /// measured on the real `hg` path at `WORDS_PER_SEQ = 2048`, device memory sat
    /// flat for many windows and then a single larger one walked it from 14.2 GB to
    /// 16.5 GB and aborted.
    pub fn trim(&mut self, want: usize) {
        let cap = want.saturating_mul(RETAIN_SLACK);
        self.free
            .retain(|(size, bufs)| *size <= cap && !bufs.is_empty());
        // At most one spare per size class. A class holding several buffers means
        // several were live at once *within* a pass — but the next pass re-takes them
        // one at a time, so the extras are dead weight until then, and at the
        // encoder/decoder's one-rectangle-per-length-bucket rate they accumulate fast:
        // measured at 686 MB (encoder) + 615 MB (decoder) of pure spares.
        //
        // Keeping one means the common case (a class the next pass touches once) still
        // hits the free list; the rest reallocate, which at a window boundary is off
        // the hot path.
        for (_, bufs) in self.free.iter_mut() {
            bufs.truncate(1);
        }
        self.cap_free_list();
    }

    /// Hard cap on how many distinct size classes the free list may hold.
    ///
    /// `size_class` bounds the classes *reachable per octave*, and the size bound in
    /// `trim` bounds how oversized any one buffer may be — but neither bounds the
    /// count when the workload spans many octaves at once. Real training does exactly
    /// that: measured on `hg`, a window carries 12-16 word-length buckets and the word
    /// count ranges 100..2048, so the encoder and decoder alone ask for dozens of
    /// unrelated rectangles per window. The free list reached 320 classes / 820
    /// buffers and the run OOMed, while a synthetic sweep converged at 218.
    ///
    /// Capping by count is what makes the footprint bounded *regardless* of how much
    /// shape diversity the corpus has. The evicted entries are the largest ones beyond
    /// the cap: they are the most expensive to keep and the least likely to be reused
    /// (a big rectangle comes from a big window, which is rare), while the small
    /// classes that every window touches stay resident.
    /// Sized against what ONE pool needs, not the model-wide total. Each block owns its
    /// own pool, so a 16-block stack reporting "320 classes" is ~20 per pool — a cap of
    /// 48 there never fires. A single block's forward+backward touches a handful of
    /// distinct shapes per window (the SwiGLU widths, the cell's reorgs), so 8 classes
    /// covers the working set with room for the window-to-window drift, and anything
    /// beyond that is a record of shapes that are no longer being asked for.
    const MAX_FREE_CLASSES: usize = 8;

    /// Drop the classes holding the most bytes, down to [`MAX_FREE_CLASSES`].
    ///
    /// Only the *duplicates* within a class are dropped first: a class holding four
    /// spare buffers of one size is four times the cost of one holding a single
    /// buffer, and the spares are what accumulate when window sizes drift. Dropping a
    /// whole class outright is the last resort, because the next window of that shape
    /// then reallocates it — thrash rather than a saving.
    fn cap_free_list(&mut self) {
        if self.free.len() <= Self::MAX_FREE_CLASSES {
            return;
        }
        // First pass: keep at most one spare per class. That alone usually brings the
        // total down without losing the ability to serve any shape.
        for (_, bufs) in self.free.iter_mut() {
            bufs.truncate(1);
        }
        if self.free.len() <= Self::MAX_FREE_CLASSES {
            return;
        }
        // Still too many distinct shapes: drop the largest classes, which cost the
        // most and recur the least (a big rectangle needs a big window).
        self.free.sort_unstable_by_key(|(size, _)| *size);
        self.free.truncate(Self::MAX_FREE_CLASSES);
    }

    /// Distinct sizes on the free list, and total buffers. Diagnostic: a pool that is
    /// working holds a handful of sizes, one per shape the pass actually uses. A count
    /// that climbs window over window is the free list accumulating one entry per
    /// distinct shape ever seen — each individually within `RETAIN_SLACK`, so `trim`
    /// keeps all of them, and together unbounded.
    pub fn free_list_shape(&self) -> (usize, usize) {
        (
            self.free.len(),
            self.free.iter().map(|(_, b)| b.len()).sum(),
        )
    }

    /// How many buffers are currently out on loan. Zero between passes if every
    /// `take` is matched by a `put`; a value that climbs pass over pass means
    /// buffers are being dropped instead of returned (memory the pool can never
    /// reuse), and a `put` without a matching `take` trips the assert above.
    pub fn outstanding(&self) -> usize {
        self.lent
    }

    /// Assert that nothing is still on loan — call at a point where every scratch
    /// value should be dead, i.e. the top of a forward or backward.
    ///
    /// A buffer taken and then *moved* somewhere permanent (a cache the next pass
    /// reads) never comes back, so the pool reallocates a replacement every pass
    /// while the free list stays the same size: a slow leak that neither
    /// [`pooled_elems`](Self::pooled_elems) nor a memory graph makes obvious. This
    /// turns that into a panic at the point of the mistake. Debug builds only.
    pub fn assert_drained(&self, what: &str) {
        debug_assert_eq!(
            self.lent, 0,
            "{what}: {} pooled buffer(s) never returned — a `take` whose value was \
             moved somewhere permanent instead of `put` back",
            self.lent
        );
    }

    /// Take a borrowed buffer permanently out of the pool's accounting.
    ///
    /// For a value that was pooled scratch but is now being moved somewhere that
    /// outlives the pass — a forward cache backward reads. Without this the loan
    /// counter never comes back down and [`assert_drained`](Self::assert_drained)
    /// fires on what is actually a deliberate hand-off. The buffer is simply
    /// returned to the caller, who now owns it.
    pub fn detach(&mut self, t: GTensor<f32>) -> GTensor<f32> {
        debug_assert!(
            self.lent > 0,
            "Pool::detach of a buffer this pool never lent"
        );
        self.lent = self.lent.saturating_sub(1);
        t
    }

    /// Return several buffers at once.
    pub fn put_all<I: IntoIterator<Item = GTensor<f32>>>(&mut self, ts: I) {
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
        f: impl FnOnce(&mut Self, &mut GTensor<f32>) -> R,
    ) -> R {
        let mut t = self.take(gpu, dims);
        let r = f(self, &mut t);
        self.put(t);
        r
    }

    /// Total elements currently held on the free list — the pool's retained
    /// memory, in f32s. For tests and memory reporting.
    pub fn pooled_elems(&self) -> usize {
        self.free.iter().map(|(size, bufs)| size * bufs.len()).sum()
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

    /// The pool's reason for existing: a temporary returned after use must be
    /// handed back out rather than reallocated, so N sequential temporaries of one
    /// size cost ONE buffer — not N.
    #[test]
    fn pool_recycles_a_returned_buffer() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        use cudarc::driver::DevicePtr;
        let addr = |t: &GTensor<f32>| t.buf.device_ptr(&gpu.stream).0;
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

    /// The bug this pool shipped with: requiring an EXACT size match meant every
    /// distinct shape pinned its own buffer forever. Real training windows vary
    /// (trailing windows shrink to `MIN_WORDS_PER_SEQ`, and the encoder runs one
    /// rectangle per word length), so the free list grew without bound and the run
    /// died of out-of-memory — while a fixed-shape benchmark looked stable.
    ///
    /// Reuse is now by capacity, so a descending sequence of sizes must recycle
    /// ONE buffer rather than accumulate one per size.
    #[test]
    fn pool_reuses_one_buffer_across_shrinking_shapes() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let mut pool = Pool::new();
        // All within RETAIN_SLACK (4x) of 1024, so one buffer serves them all.
        for len in [1024, 768, 512, 300] {
            let t = pool.take(&gpu, &[len]);
            assert_eq!(t.len(), len, "presented at the requested shape");
            pool.put(t);
        }
        assert_eq!(
            pool.pooled_elems(),
            1024,
            "one allocation must serve every request within the slack bound"
        );
    }

    /// A request far below the retained capacity gets its own buffer rather than the
    /// oversized one — the bound that stops a pool from ratcheting to the largest
    /// window it ever saw. See [`RETAIN_SLACK`].
    #[test]
    fn pool_does_not_hand_out_wildly_oversized_buffers() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let mut pool = Pool::new();
        let big = pool.take(&gpu, &[4096]);
        pool.put(big);

        // 8 is 512x smaller: reusing the 4096 buffer would pin it forever.
        let small = pool.take(&gpu, &[8]);
        assert_eq!(
            small.capacity(),
            8,
            "a tiny request took the oversized buffer"
        );
        pool.put(small);

        // `trim` at a pass boundary then releases the buffer nothing can use.
        pool.trim(8);
        assert_eq!(
            pool.pooled_elems(),
            8,
            "trim must drop free buffers beyond the slack bound"
        );
    }

    /// Growing past the retained capacity allocates once, and that bigger buffer
    /// then serves everything below it.
    #[test]
    fn pool_grows_once_then_reuses() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let mut pool = Pool::new();
        for len in [64, 512, 4096] {
            let t = pool.take(&gpu, &[len]);
            pool.put(t);
        }
        // Every size is on the list; nothing within the slack bound reallocates.
        let before = pool.pooled_elems();
        for len in [4096, 2048, 1500, 4096] {
            let t = pool.take(&gpu, &[len]);
            pool.put(t);
        }
        assert_eq!(
            pool.pooled_elems(),
            before,
            "a request within RETAIN_SLACK of a pooled buffer may not allocate"
        );
    }

    /// Reuse is by element count, so a `[4, 32]` temporary can serve a later
    /// `[128]` one — the buffers are contiguous and shape is only metadata.
    #[test]
    fn pool_reuses_across_shapes_of_equal_size() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        use cudarc::driver::DevicePtr;
        let addr = |t: &GTensor<f32>| t.buf.device_ptr(&gpu.stream).0;
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
        assert_eq!(
            pool.pooled_elems(),
            64,
            "scoped temporaries must not stack up"
        );
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
