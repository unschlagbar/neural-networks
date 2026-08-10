//! Host-backed storage for activations that forward saves and backward reads.
//!
//! Training is bounded by activation VRAM, and that bound is linear in the sequence
//! length: step 0's saved tensors must survive until backward unwinds to `t = 0`, so
//! no reordering of the forward loop frees them. Only moving them off the device — or
//! recomputing them — changes the scaling.
//!
//! An [`OffloadRing`] is the third storage kind in this module, alongside the two in
//! [`buf`](super::buf):
//!
//! | | lives | sized by |
//! |---|---|---|
//! | [`Pool`](super::Pool) | within one phase | high-water mark of live temporaries |
//! | [`Buf`](super::Buf) | across calls, on device | one activation |
//! | `OffloadRing` | across calls, **on the host** | K timesteps of device staging |
//!
//! The device cost of a ring is `2·K` timesteps of staging regardless of `T`; the
//! full `T` timeline sits in pinned host memory. That is the whole point — device
//! memory stops growing with `T`.
//!
//! # How the overlap works
//!
//! Forward writes chunk `i` into one half of the double buffer while the previous
//! chunk's device→host copy drains from the other half, on a second stream. Backward
//! runs it in reverse, prefetching chunk `i-1` while compute consumes chunk `i`.
//! Measured on this machine (`examples/offload_probe.rs`): a 55 MB chunk round-trips
//! in ~2.6 ms against ~16.5 ms of compute, and ~95% of the transfer hides.
//!
//! # Ordering is by hand
//!
//! `Gpu::new` calls `disable_event_tracking` (see [`super::Gpu::new`]), so cudarc
//! places **no** automatic cross-stream ordering. Every handoff between the compute
//! stream and this module's transfer stream is an explicit [`CudaEvent`]. A missing
//! event is silent corruption rather than an error, so the two directions are named
//! and documented individually below, and `ring_roundtrip_survives_contention` pins
//! them by value under deliberate contention.
//!
//! # Pinned host memory
//!
//! Host staging is page-locked ([`CudaContext::alloc_pinned`]), which is what makes
//! the copies async and fast: measured 37.9 GB/s D2H against 1.5 GB/s pageable. Note
//! that cudarc allocates it **write-combined** — fast for the device to write and
//! read over PCIe, but slow to read from the CPU. Nothing here reads it on the host,
//! and nothing should start: the host is only a DMA endpoint.
//!
//! # What a "chunk" is
//!
//! The ring indexes chunks; it does not care what one contains. The backbone sweeps
//! **block by block** over the whole sequence (`Hierarchical::forward_backward`
//! runs block `i` to completion before block `i+1`, and unwinds in reverse), so its
//! natural chunk is one block's saved activations — written once in forward, read
//! once in backward, ~15 blocks of compute apart. A per-timestep consumer would
//! index the same ring by time instead. Both get the same events.

use std::sync::Arc;

use cudarc::driver::{CudaEvent, CudaStream, PinnedHostSlice};

use super::{BTensor, DTensor, Gpu};

/// Where one chunk of a tensor's timeline lives while it is off the device.
///
/// One `PinnedHostSlice` per chunk rather than a single slab: the chunks are copied
/// independently and each copy needs its own completion event, which
/// `PinnedHostSlice` already carries internally.
struct HostChunk {
    mem: PinnedHostSlice<f32>,
    /// Elements actually used. The last chunk of a ragged `T` is shorter than the
    /// allocation, and copying the full slice would move — and later restore —
    /// garbage past the end of the timeline.
    len: usize,
}

/// A set of device tensors with a leading time axis, staged through host memory.
///
/// Construct with [`new`](Self::new), then in forward: [`write`](Self::write) each
/// chunk in order. In backward: [`read`](Self::read) each chunk in reverse. Both
/// place their own events; the caller never touches the transfer stream.
///
/// **Currently exercised only by its own tests.** The offloaded consumer that exists
/// today — the backbone's FFN — sweeps block by block rather than timestep by
/// timestep, so it uses [`HostPark`] instead. This is the mechanism for a per-timestep
/// consumer (the sLSTM's `[·, T, ·]` slabs, the mLSTM's saved fields), which is where
/// the remaining per-T activation memory lives. Kept rather than deleted because the
/// hard parts — the double buffer, the four hand-placed events, the ragged last chunk
/// — are done and tested; see `ring_roundtrip_survives_contention`.
pub struct OffloadRing {
    /// The full `T` timeline, one entry per chunk, in host memory.
    host: Vec<HostChunk>,
    /// Double buffer: `K` timesteps of device staging, alternating so a copy can
    /// drain from one half while compute fills the other.
    dev: [DTensor; 2],
    /// Copy stream, distinct from `gpu.stream` so transfers overlap compute.
    xfer: Arc<CudaStream>,
    /// Per-half "the copy out of this half has finished", so the next writer of that
    /// half waits rather than overwriting a buffer still being read.
    drained: [Option<CudaEvent>; 2],
    /// Per-half "the copy into this half has finished", so a reader waits rather than
    /// consuming a buffer the transfer has not filled yet.
    filled: [Option<CudaEvent>; 2],
    /// Timesteps per chunk (the last chunk may be shorter).
    k: usize,
    /// Elements per timestep — the product of the trailing (non-time) dims.
    per_step: usize,
    /// Total timesteps this ring was sized for.
    t: usize,
}

impl OffloadRing {
    /// A ring for a `[T, per_step]` timeline cut into chunks of `k` timesteps.
    ///
    /// Allocates the whole host timeline up front (`T · per_step` floats, pinned) plus
    /// `2 · k · per_step` floats of device staging. Both are one-time: reuse the ring
    /// across steps rather than rebuilding it, or the pinned allocation — which is
    /// expensive, it page-locks — lands on the hot path.
    pub fn new(gpu: &Gpu, t: usize, per_step: usize, k: usize) -> Result<Self, String> {
        assert!(k > 0, "OffloadRing: chunk length must be positive");
        assert!(per_step > 0, "OffloadRing: per-step width must be positive");
        let xfer = gpu
            .context
            .new_stream()
            .map_err(|e| format!("offload: transfer stream creation failed: {e:?}"))?;

        let mut host = Vec::with_capacity(t.div_ceil(k));
        for c0 in (0..t).step_by(k) {
            let len = k.min(t - c0) * per_step;
            // SAFETY: freshly allocated pinned memory is uninitialised, and the
            // contract is that a chunk is written (by `write`) before it is read (by
            // `read`) — the ring never hands out a chunk it has not staged.
            let mem = unsafe { gpu.context.alloc_pinned::<f32>(len) }
                .map_err(|e| format!("offload: pinned host alloc of {len} floats failed: {e:?}"))?;
            host.push(HostChunk { mem, len });
        }

        Ok(Self {
            host,
            dev: [
                DTensor::uninit(gpu, &[k, per_step]),
                DTensor::uninit(gpu, &[k, per_step]),
            ],
            xfer,
            drained: [None, None],
            filled: [None, None],
            k,
            per_step,
            t,
        })
    }

    /// Number of chunks the timeline is cut into.
    #[inline]
    pub fn chunks(&self) -> usize {
        self.host.len()
    }

    /// Timesteps in chunk `i` — `k`, except for a ragged last chunk.
    #[inline]
    pub fn chunk_steps(&self, i: usize) -> usize {
        self.host[i].len / self.per_step
    }

    /// Total device bytes this ring holds (the staging, not the timeline).
    #[inline]
    pub fn device_bytes(&self) -> usize {
        2 * self.k * self.per_step * 4
    }

    /// Total pinned host bytes (the timeline).
    #[inline]
    pub fn host_bytes(&self) -> usize {
        self.t * self.per_step * 4
    }

    /// Device staging for chunk `i`, ready to be written by compute.
    ///
    /// Waits — on the compute stream, so the wait costs nothing on the host — until
    /// the previous copy **out of** this half has finished. Without that wait, a
    /// forward two chunks ahead would overwrite data the transfer stream is still
    /// draining.
    pub fn stage(&mut self, gpu: &Gpu, i: usize) -> &mut DTensor {
        let half = i % 2;
        if let Some(ev) = &self.drained[half] {
            gpu.stream.wait(ev).expect("offload: wait for drain");
        }
        let steps = self.chunk_steps(i);
        self.dev[half].shrink_to(&[steps, self.per_step]);
        &mut self.dev[half]
    }

    /// Send chunk `i` to the host, having filled [`stage`](Self::stage) with it.
    ///
    /// Two events, one per direction of the handoff:
    ///   * `produced` — the transfer may not start until compute has finished writing.
    ///   * `drained` — a later [`stage`](Self::stage) of this half must wait for this
    ///     copy, recorded for that half.
    pub fn write(&mut self, gpu: &Gpu, i: usize) {
        let half = i % 2;
        let produced = gpu
            .stream
            .record_event(None)
            .expect("offload: record produced");
        self.xfer.wait(&produced).expect("offload: xfer waits");

        let chunk = &mut self.host[i];
        let src = self.dev[half].buf.slice(..chunk.len);
        self.xfer
            .memcpy_dtoh(&src, &mut chunk.mem)
            .expect("offload: D2H");

        self.drained[half] = Some(self.xfer.record_event(None).expect("offload: record drain"));
    }

    /// Start chunk `i`'s host→device copy without waiting for it.
    ///
    /// Call this one chunk *ahead* of the consumer so the transfer overlaps compute;
    /// [`read`](Self::read) then returns it without stalling. Prefetching a chunk
    /// whose half still holds an unconsumed one is the caller's error to avoid — with
    /// a double buffer that means never prefetching more than one chunk ahead.
    pub fn prefetch(&mut self, gpu: &Gpu, i: usize) {
        let half = i % 2;
        // The staging half may still be feeding a consumer on the compute stream, so
        // the fill waits for compute before overwriting it.
        let consumed = gpu
            .stream
            .record_event(None)
            .expect("offload: record consumed");
        self.xfer.wait(&consumed).expect("offload: xfer waits");

        let chunk = &self.host[i];
        let mut dst = self.dev[half].buf.slice_mut(..chunk.len);
        self.xfer
            .memcpy_htod(&chunk.mem, &mut dst)
            .expect("offload: H2D");

        self.filled[half] = Some(self.xfer.record_event(None).expect("offload: record fill"));
    }

    /// Chunk `i` back on the device, ready for compute to read.
    ///
    /// Issues the copy first if [`prefetch`](Self::prefetch) was not called for `i`
    /// (correct, just not overlapped). Waits on the compute stream until the fill has
    /// landed — the event that makes the returned tensor safe to read.
    pub fn read(&mut self, gpu: &Gpu, i: usize) -> &DTensor {
        let half = i % 2;
        if self.filled[half].is_none() {
            self.prefetch(gpu, i);
        }
        let ev = self.filled[half].take().expect("just prefetched");
        gpu.stream.wait(&ev).expect("offload: wait for fill");
        let steps = self.chunk_steps(i);
        self.dev[half].shrink_to(&[steps, self.per_step]);
        &self.dev[half]
    }

    /// Block until every queued transfer has completed.
    ///
    /// Only needed at a teardown or measurement boundary — the per-chunk events
    /// already order the streams against each other, so the training loop does not
    /// call this.
    pub fn sync(&self) {
        self.xfer.synchronize().expect("offload: sync");
    }
}

/// A buffer a [`HostPark`] can move: an fp32 tensor or a bf16 slab.
///
/// The two differ only in element width, and the park's job is to move bytes — so it
/// carries the kind through the round trip and hands back exactly what it was given.
/// Widening a bf16 slab on the way out would double its transfer *and* its restored
/// footprint, and narrowing an fp32 one would silently break the stabilizer arithmetic
/// that `gpu::bf16` deliberately keeps wide.
pub enum Parked {
    F32(DTensor),
    Bf16(BTensor),
}

impl Parked {
    fn dims(&self) -> &[usize] {
        match self {
            Parked::F32(t) => t.dims(),
            Parked::Bf16(t) => t.dims(),
        }
    }

    /// Size in `u16` units — the pinned slots' element type, so fp32 counts double.
    fn u16_len(&self) -> usize {
        match self {
            Parked::F32(t) => t.len() * 2,
            Parked::Bf16(t) => t.len(),
        }
    }

    pub fn bytes(&self) -> usize {
        match self {
            Parked::F32(t) => t.len() * 4,
            Parked::Bf16(t) => t.len() * 2,
        }
    }

    fn is_bf16(&self) -> bool {
        matches!(self, Parked::Bf16(_))
    }

    /// Copy this buffer's bytes into a pinned `u16` host slot, on `xfer`.
    ///
    /// The fp32 case views its `CudaSlice<f32>` as `u16` so both widths take the same
    /// path. That is a pure reinterpretation of the same bytes — the host slot is only
    /// ever written here and read back by `fill_from_host` below, which applies the
    /// inverse view, so no value is ever interpreted at the wrong width.
    fn copy_to_host(&self, xfer: &Arc<CudaStream>, dst: &mut PinnedHostSlice<u16>) {
        // The slot is reused by capacity and may be longer than this buffer, so both
        // sides are cut to `u16_len()` — the copy must not be sized by the slot.
        let n16 = self.u16_len();
        let dst = &mut dst.as_mut_slice().expect("offload: pinned slot view")[..n16];
        match self {
            Parked::Bf16(t) => {
                let n = t.len();
                xfer.memcpy_dtoh(&t.buf.slice(..n), dst)
            }
            Parked::F32(t) => {
                let n = t.len();
                let src = t.buf.slice(..n);
                // SAFETY: `f32` and `u16` are both plain data with no invalid bit
                // patterns, and n f32 cover exactly 2n u16 with identical alignment
                // requirements met (f32 is 4-aligned, hence 2-aligned).
                let view = unsafe { src.transmute::<u16>(n * 2) }
                    .expect("offload: f32->u16 view");
                xfer.memcpy_dtoh(&view, dst)
            }
        }
        .expect("offload: D2H");
    }

    /// Fill this buffer from a pinned `u16` host slot, on `xfer`. Inverse of
    /// [`copy_to_host`](Self::copy_to_host).
    fn fill_from_host(&mut self, xfer: &Arc<CudaStream>, src: &PinnedHostSlice<u16>) {
        // Cut to this buffer's length, not the slot's — see `copy_to_host`.
        let n16 = self.u16_len();
        let src = &src.as_slice().expect("offload: pinned slot view")[..n16];
        match self {
            Parked::Bf16(t) => {
                let n = t.len();
                xfer.memcpy_htod(src, &mut t.buf.slice_mut(..n))
            }
            Parked::F32(t) => {
                let n = t.len();
                let mut dst = t.buf.slice_mut(..n);
                // SAFETY: as in `copy_to_host` — same bytes, same element count.
                let mut view = unsafe { dst.transmute_mut::<u16>(n * 2) }
                    .expect("offload: f32->u16 view");
                xfer.memcpy_htod(src, &mut view)
            }
        }
        .expect("offload: H2D");
    }

    /// Download to the host as fp32, widening a bf16 slab.
    ///
    /// For tests and debugging only — it allocates and synchronizes. Nothing on the
    /// training path reads a parked buffer from the host; see the module note on
    /// write-combined memory.
    pub fn to_host(&self, gpu: &Gpu) -> crate::tensor::Tensor {
        match self {
            Parked::F32(t) => t.to_host(gpu),
            Parked::Bf16(t) => {
                let mut wide = DTensor::uninit(gpu, t.dims());
                t.load(gpu, &mut wide);
                wide.to_host(gpu)
            }
        }
    }

    /// Unwrap an fp32 tensor, panicking if this is a bf16 slab.
    pub fn f32(self) -> DTensor {
        match self {
            Parked::F32(t) => t,
            Parked::Bf16(_) => panic!("offload: expected an fp32 tensor, got a bf16 slab"),
        }
    }

    /// Unwrap a bf16 slab, panicking if this is an fp32 tensor.
    pub fn bf16(self) -> BTensor {
        match self {
            Parked::Bf16(t) => t,
            Parked::F32(_) => panic!("offload: expected a bf16 slab, got an fp32 tensor"),
        }
    }
}

impl From<DTensor> for Parked {
    fn from(t: DTensor) -> Self {
        Parked::F32(t)
    }
}

/// A slab is the same fp32-or-bf16 pair, chosen by `Kernels::slab_bf16` rather than
/// per value — so it parks directly, at whatever width it was built with.
impl From<super::ops::SlabBuf> for Parked {
    fn from(s: super::ops::SlabBuf) -> Self {
        match s {
            super::ops::SlabBuf::F32(t) => Parked::F32(t),
            super::ops::SlabBuf::Bf16(t) => Parked::Bf16(t),
        }
    }
}

impl From<Parked> for super::ops::SlabBuf {
    fn from(p: Parked) -> Self {
        match p {
            Parked::F32(t) => super::ops::SlabBuf::F32(t),
            Parked::Bf16(t) => super::ops::SlabBuf::Bf16(t),
        }
    }
}

impl From<BTensor> for Parked {
    fn from(t: BTensor) -> Self {
        Parked::Bf16(t)
    }
}

/// Shape and kind of one parked buffer, enough to rebuild it on restore.
struct ParkedShape {
    dims: Vec<usize>,
    bf16: bool,
}

/// Host parking space for one layer's saved activations.
///
/// [`OffloadRing`] suits a consumer that walks a time axis with a fixed per-step
/// width. A [`Block`](super::block::Block) is the other shape: it saves a handful of
/// differently-shaped buffers, all at once, and does not read any of them again until
/// backward reaches it — 15 blocks of compute later, in the backbone's block-major
/// sweep. There is nothing to double-buffer against, because the block's own compute
/// is long since finished; the transfer just has to not race it.
///
/// So this is the simpler mechanism: [`evict`](Self::evict) sends a set of device
/// buffers to pinned host memory and lets the device ones go, and
/// [`restore`](Self::restore) brings them back. Same transfer stream, same hand-placed
/// events, no ring.
pub struct HostPark {
    /// One generation per parked chunk, in eviction order; within a generation, one
    /// pinned slot per parked buffer.
    ///
    /// A chunked backbone sweep evicts once per `(block, chunk)` and unwinds chunks
    /// right to left, so a park owes as many restores as the sweep made evictions.
    /// Holding a single generation would let chunk c+1's eviction overwrite the slots
    /// chunk c's backward still has to read — a wrong gradient, not a crash. The
    /// unchunked path is exactly this with `gens.len() == 1`.
    ///
    /// Slots are typed `u16` and sized in *elements of u16* so one park can hold both
    /// fp32 tensors and bf16 slabs — see [`Parked`]. A bf16 slab must come back as
    /// bf16: this module moves bytes and never changes a value's width, because the
    /// precision split is decided at each value's production point (`gpu::bf16`), not
    /// on the way to host memory.
    gens: Vec<ParkedGen>,
    /// How many of `gens` currently hold live data, i.e. how many restores are owed.
    ///
    /// Distinct from `gens.len()`: a restore pops the data but leaves the pinned slots
    /// allocated so the next step's eviction of the same shape reuses them instead of
    /// page-locking again. `gens[..live]` is live; `gens[live..]` is spare capacity.
    live: usize,
    xfer: Arc<CudaStream>,
    /// Device tensors handed over by [`evict`](Self::evict), held until their D2H copy
    /// completes.
    ///
    /// This is what makes eviction asynchronous. Freeing them at `evict` would mean
    /// either blocking on the copy first — which serializes the transfer against
    /// compute and costs the whole point of offloading (measured: a 24% step
    /// regression, the full un-overlapped transfer time) — or freeing memory under a
    /// live DMA. Holding them lets the copy proceed while compute runs on.
    ///
    /// The slot is **shared between every park in a model** (see
    /// [`InFlight`](InFlight)): a park that held its own buffers until its *own* next
    /// eviction would keep them for a whole step, and with one park per block that is
    /// 16 blocks' worth alive at once — measured **1728 MB worse** than not offloading
    /// at all. Sharing bounds it at one block's worth, released one block later.
    in_flight: SharedInFlight,
    /// Uploads started by [`prefetch`](Self::prefetch) but not yet consumed, with the
    /// event that says they have landed. Belongs to the generation that
    /// [`restore`](Self::restore) will take next — the last one evicted.
    prefetched: Option<(Vec<Parked>, CudaEvent)>,
    /// Pinned slots not currently held by a generation, reusable by capacity.
    ///
    /// Page-locking costs ~640 us per slot, so a park that allocated on every eviction
    /// would spend most of a step in `cuMemHostAlloc`. Slots come back here when a
    /// generation is displaced and are handed out again by [`evict`](Self::evict).
    ///
    /// Bounded by [`peak_slots`](Self::peak_slots) plus [`SPARE_SLACK`] — an unbounded
    /// pool would pin host memory in proportion to the largest shape ever evicted and
    /// never give it back.
    spare: Vec<PinnedHostSlice<u16>>,
    /// Most slots this park has ever owned at once, across every generation and the
    /// spare pool. The retention bound follows this, so a park keeps exactly the slots
    /// its own sweep shape turns over and no more.
    peak_slots: usize,
    /// Page-locking calls this park has made, for the reuse test to observe.
    #[cfg(test)]
    allocs: usize,
}


/// Headroom above a park's observed peak demand, in slots.
///
/// The retention bound cannot be a fixed constant: a chunked sweep keeps one generation
/// per chunk, so peak demand scales with sequence length (at 4069 words a backbone park
/// peaks at 40 slots — a fixed 32 dropped 256 slots per step and page-locked them again,
/// ~225 ms of `cuMemHostAlloc` on the critical path). The slack absorbs the ragged last
/// chunk and the alternating FFN/cell buffer counts without a second round of misses.
const SPARE_SLACK: usize = 8;

/// One eviction's pinned slots and the shapes needed to rebuild its tensors.
struct ParkedGen {
    slots: Vec<PinnedHostSlice<u16>>,
    shapes: Vec<ParkedShape>,
}

/// The one block's worth of evicted buffers that may be awaiting a copy at any time.
///
/// Every [`HostPark`] in a model holds a clone of the same slot, so each eviction
/// releases the *previous* block's buffers — whose copy has by then had a full block
/// of compute to finish in — and leaves its own in their place.
///
/// Shared rather than threaded through the call chain because eviction happens deep
/// inside `Block::forward`, and `BlockLike::forward` is a trait method whose signature
/// every caller and both cell kinds would otherwise have to grow a parameter for.
pub type SharedInFlight = std::rc::Rc<std::cell::RefCell<InFlight>>;

/// One block's worth of evicted buffers, waiting for their copy to land.
#[derive(Default)]
pub struct InFlight {
    /// Evictions awaiting their copy, oldest first.
    ///
    /// Two deep, not one. Releasing a buffer makes the compute stream wait on its D2H,
    /// so with a single generation block `i+1` waits on block `i`'s copy immediately
    /// after issuing it — the transfer is exposed (measured: +20 ms of forward glue
    /// against ~19 ms of D2H, i.e. no overlap). Holding two generations puts a whole
    /// block of compute between a copy and the wait on it, at the cost of one extra
    /// block's activations resident.
    pending: Vec<(Vec<Parked>, CudaEvent)>,
}

/// How many evictions may be awaiting their copy at once. See [`InFlight::pending`].
///
/// A block evicts twice — once for its FFN, once for its cell — so at depth 2 the two
/// halves of one block are in flight together and each wait falls on a copy issued a
/// whole block of compute ago. That is the guarantee that matters, and it is what
/// keeps the transfers hidden.
///
/// Deeper is not better: raising this to 4 (a full two blocks) measured **384 MB worse
/// at the same step time** — an extra generation of activations held resident for a
/// wait that was never going to block.
const IN_FLIGHT_DEPTH: usize = 2;

impl InFlight {
    /// A fresh shared slot. One per model, cloned into each block's park.
    pub fn shared() -> SharedInFlight {
        Default::default()
    }

    /// Free the oldest eviction's buffers if the queue is full, ordering the *compute
    /// stream* against its copy rather than blocking the host.
    ///
    /// `gpu.stream.wait(ev)` is the whole point: the buffers are freed with
    /// `cuMemFreeAsync` on the compute stream, so making that stream wait for the copy
    /// is sufficient — and it costs no host time, which a `synchronize()` would.
    /// Blocking the host instead measured a 26% step regression, because it serializes
    /// every eviction against the compute it is supposed to hide behind.
    pub fn release(&mut self, gpu: &Gpu) {
        while self.pending.len() >= IN_FLIGHT_DEPTH {
            let (bufs, ev) = self.pending.remove(0);
            gpu.stream.wait(&ev).expect("offload: order free after park");
            drop(bufs);
        }
    }

    /// Block the host until every queued copy has landed, then free. Teardown/tests.
    pub fn release_blocking(&mut self) {
        for (_, ev) in &self.pending {
            ev.synchronize().expect("offload: await park");
        }
        self.pending.clear();
    }

    /// Queue an eviction's buffers behind the event that says its copy has landed.
    fn push(&mut self, bufs: Vec<Parked>, done: CudaEvent) {
        self.pending.push((bufs, done));
    }

    /// Device bytes currently held awaiting a copy.
    pub fn bytes(&self) -> usize {
        self.pending
            .iter()
            .flat_map(|(bufs, _)| bufs)
            .map(Parked::bytes)
            .sum()
    }
}

impl HostPark {
    /// An empty park, sharing `in_flight` with every other park in the model.
    pub fn new(gpu: &Gpu, in_flight: SharedInFlight) -> Result<Self, String> {
        Ok(Self {
            gens: Vec::new(),
            live: 0,
            xfer: gpu
                .context
                .new_stream()
                .map_err(|e| format!("offload: transfer stream creation failed: {e:?}"))?,
            in_flight,
            prefetched: None,
            spare: Vec::new(),
            peak_slots: 0,
            #[cfg(test)]
            allocs: 0,
        })
    }

    /// Bytes currently parked on the host, over every generation.
    pub fn host_bytes(&self) -> usize {
        self.gens[..self.live]
            .iter()
            .flat_map(|g| g.slots.iter())
            .map(|s| s.num_bytes())
            .sum()
    }

    /// Whether anything is parked (i.e. a [`restore`](Self::restore) is owed).
    pub fn is_parked(&self) -> bool {
        self.gens[..self.live].iter().any(|g| !g.slots.is_empty())
    }

    /// Copy `bufs` to pinned host memory.
    ///
    /// The copy is issued on the transfer stream after an event says compute has
    /// finished producing them, and this returns **without waiting for it** — that is
    /// what lets the DMA run underneath the next block's compute. The caller must
    /// therefore keep the device buffers alive until
    /// [`take_evicted`](Self::take_evicted) says the copy has landed; releasing them
    /// earlier frees memory out from under an in-flight DMA.
    ///
    /// Reuses its pinned slots across steps when the shapes repeat, so a steady
    /// training loop page-locks nothing after the first window.
    pub fn evict(&mut self, gpu: &Gpu, bufs: Vec<Parked>) {
        // Make room for this eviction. `Block::forward` and `MLstm::forward_alloc`
        // release before allocating, which is what keeps the wait off the hot path, but
        // a block releases twice and then evicts twice — so the second eviction can
        // still arrive with the queue full. Draining here bounds the queue at the
        // depth regardless of how the callers interleave.
        let mut in_flight = self.in_flight.borrow_mut();
        in_flight.release(gpu);

        let produced = gpu
            .stream
            .record_event(None)
            .expect("offload: record produced");
        self.xfer.wait(&produced).expect("offload: xfer waits");

        // Append a generation rather than overwriting the last: a chunked sweep evicts
        // once per chunk and every one of them is owed a restore.
        let depth = self.live;
        // Slots are matched by capacity rather than generation shape-for-shape. Two
        // things defeat an exact per-depth match: one park serves both the cell and the
        // FFN, whose evictions differ in buffer count and land at the same depth
        // alternately, and a balanced `chunk_spans` makes the last chunk one row shorter
        // than the rest. Either alone means a depth's shapes never repeat, and
        // page-locking is expensive enough (~640 us per slot) that missing puts it
        // squarely on the hot path.
        //
        // A slot is reusable when it is at least as large as the buffer: the copy writes
        // `u16_len()` elements and `restore` rebuilds the tensor from `ParkedShape`, so
        // the slot's own length is never read back. Reusing a larger slot wastes only
        // the tail.

        // Peak slots this park circulates: everything it currently owns — live
        // generations, restored-but-not-yet-displaced ones, and the spares between them
        // — plus what this eviction takes. The spares must be counted: with capacity
        // matching a smaller slot can sit unused while a larger request misses, and a
        // bound that ignored them would discard exactly the slots the next sweep needs.
        let owned: usize =
            self.gens.iter().map(|g| g.slots.len()).sum::<usize>() + self.spare.len();
        self.peak_slots = self.peak_slots.max(owned + bufs.len());
        let spare_max = self.peak_slots + SPARE_SLACK;

        // Reclaim the generations this eviction displaces *first*, so their slots are
        // available to it. Reclaiming afterwards would make every re-eviction at a
        // depth allocate before the slot it is about to replace comes back.
        if self.gens.len() > depth {
            for g in self.gens.drain(depth..) {
                for s in g.slots {
                    if self.spare.len() < spare_max {
                        self.spare.push(s);
                    }
                }
            }
        }

        let mut slots = Vec::with_capacity(bufs.len());
        let mut shapes = Vec::with_capacity(bufs.len());
        for b in &bufs {
            let need = b.u16_len();
            // Best fit, not first fit. The pool holds several capacities at once (the
            // chunked sweep's ragged last chunk, the alternating FFN/cell shapes), and
            // taking the first slot that merely fits lets a small request consume a
            // large slot — the large requests then find nothing and page-lock, once per
            // sweep, forever. Picking the tightest slot keeps each capacity class for
            // the requests that need it.
            let hit = self
                .spare
                .iter()
                .enumerate()
                .filter(|(_, s)| s.len() >= need)
                .min_by_key(|(_, s)| s.len())
                .map(|(i, _)| i)
                .map(|i| self.spare.swap_remove(i));
            let mem = match hit {
                Some(m) => m,
                None => {
                    #[cfg(test)]
                    {
                        self.allocs += 1;
                    }
                    // SAFETY: uninitialised pinned memory, fully written by the copy
                    // below before any read (`restore` is the only reader).
                    unsafe { gpu.context.alloc_pinned::<u16>(need) }
                        .expect("offload: pinned host alloc")
                }
            };
            slots.push(mem);
            shapes.push(ParkedShape {
                dims: b.dims().to_vec(),
                bf16: b.is_bf16(),
            });
        }
        self.gens.push(ParkedGen { slots, shapes });

        self.live = depth + 1;
        let cur = &mut self.gens[depth];
        for (slot, b) in cur.slots.iter_mut().zip(&bufs) {
            b.copy_to_host(&self.xfer, slot);
        }
        // Hand the sources, and the event that says their copy has landed, to the
        // caller's slot. The next block's evict waits on it and frees them.
        //
        // `restore` needs no event of its own: it runs a whole forward pass after this
        // eviction, and `InFlight::release` has synchronized on the copy long before
        // then — the host data is complete by construction.
        let done = self.xfer.record_event(None).expect("offload: record park");
        in_flight.push(bufs, done);
    }

    /// Free the previous eviction's buffers, ordered on the compute stream.
    ///
    /// Call at the *start* of the next block's forward, before it allocates: freeing
    /// returns memory to the allocator, and the allocator must not hand it back while
    /// a copy is still reading it.
    pub fn release_previous(&self, gpu: &Gpu) {
        self.in_flight.borrow_mut().release(gpu);
    }

    /// Block the host until any in-flight eviction has landed. Teardown and tests.
    pub fn sync_parked(&self) {
        self.in_flight.borrow_mut().release_blocking();
    }

    /// Start this park's uploads without waiting for them.
    ///
    /// Call one block *ahead* of the consumer: the H2D then runs underneath that
    /// block's backward compute instead of stalling in front of its own. Without it,
    /// restore issues a copy and immediately waits, and the transfer is fully exposed
    /// — measured as +37 ms of "block glue" against 32 ms of raw transfer, i.e. no
    /// overlap at all.
    ///
    /// Idempotent: a second call before [`take_prefetched`](Self::take_prefetched) is
    /// a no-op.
    pub fn prefetch(&mut self, gpu: &Gpu) {
        if self.prefetched.is_some() || self.live == 0 {
            return;
        }
        self.prefetched = Some(self.issue_uploads(gpu));
    }

    /// The prefetched tensors, waiting for their uploads if they have not landed.
    ///
    /// Issues the uploads first if [`prefetch`](Self::prefetch) was not called
    /// (correct, just not overlapped).
    pub fn take_prefetched(&mut self, gpu: &Gpu) -> Vec<Parked> {
        let (out, filled) = match self.prefetched.take() {
            Some(p) => p,
            None => self.issue_uploads(gpu),
        };
        gpu.stream.wait(&filled).expect("offload: wait for fill");
        out
    }

    /// Allocate the destinations and queue their H2D copies, returning them with the
    /// event that says the copies have landed. Shared by prefetch and restore.
    /// Reads the newest live generation — backward unwinds chunks right to left, so the
    /// last chunk evicted is the first one restored. Popping is `restore`'s job: a
    /// prefetch runs a block ahead of the take and must leave the generation in place.
    fn issue_uploads(&mut self, gpu: &Gpu) -> (Vec<Parked>, CudaEvent) {
        let cur = &self.gens[self.live - 1];
        let mut out: Vec<Parked> = cur
            .shapes
            .iter()
            .map(|s| {
                if s.bf16 {
                    Parked::Bf16(BTensor::uninit(gpu, &s.dims))
                } else {
                    Parked::F32(DTensor::uninit(gpu, &s.dims))
                }
            })
            .collect();
        // The allocations above are *stream-ordered* on the compute stream
        // (`cuMemAllocAsync`), and with `disable_event_tracking` the transfer stream
        // knows nothing about that ordering. Without this event the H2D below can
        // write into memory whose allocation has not yet been reached on the compute
        // stream — an illegal access that only shows up asynchronously.
        let allocated = gpu
            .stream
            .record_event(None)
            .expect("offload: record allocated");
        self.xfer.wait(&allocated).expect("offload: xfer waits");

        for (slot, t) in self.gens[self.live - 1].slots.iter().zip(&mut out) {
            t.fill_from_host(&self.xfer, slot);
        }
        let filled = self.xfer.record_event(None).expect("offload: record fill");
        (out, filled)
    }

    /// Bring the parked buffers back, in the order they were evicted.
    ///
    /// Waits — on the compute stream — until the uploads have landed, so the returned
    /// tensors are safe for compute to read. Consumes a [`prefetch`](Self::prefetch)
    /// if one is outstanding, which is how the transfer gets hidden; without one it
    /// issues and waits, correct but fully exposed.
    /// Pops the generation it consumed, so a chunked sweep's next restore reads the
    /// chunk to its left. The pinned slots stay allocated for the next step to reuse.
    pub fn restore(&mut self, gpu: &Gpu) -> Vec<Parked> {
        let out = self.take_prefetched(gpu);
        self.live -= 1;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    /// A full write-then-read round trip must return every chunk unchanged, including
    /// a **ragged last chunk** (`t` deliberately not a multiple of `k`).
    ///
    /// This is the property every offload consumer rests on: what backward reads is
    /// exactly what forward wrote.
    #[test]
    fn ring_roundtrip_preserves_values() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let (t, per_step, k) = (14usize, 8usize, 4usize); // 14 = 3·4 + 2 (ragged)
        let mut ring = OffloadRing::new(&gpu, t, per_step, k).expect("ring");
        assert_eq!(ring.chunks(), 4);
        assert_eq!(ring.chunk_steps(3), 2, "last chunk should be ragged");

        // Distinct value per element, so a swapped or truncated chunk cannot pass.
        let full = Tensor::new(
            &[t, per_step],
            (0..t * per_step).map(|i| i as f32 * 0.5).collect(),
        );
        let src = DTensor::from_host(&gpu, &full);

        for i in 0..ring.chunks() {
            let steps = ring.chunk_steps(i);
            let off = i * k * per_step;
            let stage = ring.stage(&gpu, i);
            gpu.stream
                .memcpy_dtod(
                    &src.buf.slice(off..off + steps * per_step),
                    &mut stage.buf.slice_mut(..steps * per_step),
                )
                .expect("fill stage");
            ring.write(&gpu, i);
        }

        // Read back in reverse, the order backward uses.
        for i in (0..ring.chunks()).rev() {
            let steps = ring.chunk_steps(i);
            let got = ring.read(&gpu, i).to_host(&gpu);
            let off = i * k * per_step;
            assert_eq!(
                &got.data[..steps * per_step],
                &full.data[off..off + steps * per_step],
                "chunk {i} came back changed"
            );
        }
    }

    /// The hand-placed events must hold up when the compute stream is genuinely busy.
    ///
    /// With `disable_event_tracking` on, nothing but those events orders the transfer
    /// stream against compute — so this writes each chunk from a kernel, keeps the
    /// compute stream loaded, and checks the values survive. A missing event shows up
    /// here as a stale or torn chunk rather than as an error.
    #[test]
    fn ring_roundtrip_survives_contention() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let (t, per_step, k) = (32usize, 1024usize, 8usize);
        let mut ring = OffloadRing::new(&gpu, t, per_step, k).expect("ring");

        // Something slow enough on the compute stream that the transfers must really
        // overlap it rather than trivially finishing first.
        let busy = DTensor::zeros(&gpu, &[512, 512]);
        let mut sink = DTensor::uninit(&gpu, &[512, 512]);

        for i in 0..ring.chunks() {
            let steps = ring.chunk_steps(i);
            {
                let stage = ring.stage(&gpu, i);
                // Tag every element of the chunk with its chunk index.
                let tag = Tensor::new(&[steps, per_step], vec![i as f32; steps * per_step]);
                let host = DTensor::from_host(&gpu, &tag);
                stage.copy_from(&gpu, &host);
            }
            ring.write(&gpu, i);
            // Load the compute stream so the copy above has something to hide behind.
            super::super::ops::matmul_nn_into(&gpu, &busy, &busy, &mut sink, 0.0);
        }

        for i in (0..ring.chunks()).rev() {
            let steps = ring.chunk_steps(i);
            super::super::ops::matmul_nn_into(&gpu, &busy, &busy, &mut sink, 0.0);
            let got = ring.read(&gpu, i).to_host(&gpu);
            assert!(
                got.data[..steps * per_step].iter().all(|&v| v == i as f32),
                "chunk {i} did not come back uniformly tagged — cross-stream ordering is wrong"
            );
        }
        ring.sync();
    }

    /// Evict-then-restore must return every buffer unchanged, at its own shape.
    ///
    /// The block-major consumer parks a set of differently-shaped activations at once,
    /// so the ordering and the per-buffer shapes both have to survive the round trip.
    #[test]
    fn park_roundtrip_preserves_values_and_shapes() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let mut park = HostPark::new(&gpu, InFlight::shared()).expect("park");
        assert!(!park.is_parked());

        // Deliberately mismatched shapes and widths, as a block's Act really is.
        let hosts = [
            Tensor::new(&[7, 5], (0..35).map(|i| i as f32 * 0.25).collect()),
            Tensor::new(&[3, 11], (0..33).map(|i| -(i as f32)).collect()),
            Tensor::new(&[64], (0..64).map(|i| i as f32 * 1e-3).collect()),
        ];
        let devs: Vec<DTensor> = hosts.iter().map(|t| DTensor::from_host(&gpu, t)).collect();

        // `evict` takes ownership: the park holds the device tensors until its copy
        // lands, then frees them. `sync_parked` forces that here.
        park.evict(&gpu, devs.into_iter().map(Parked::from).collect());
        park.sync_parked();
        assert!(park.is_parked());
        assert_eq!(park.host_bytes(), (35 + 33 + 64) * 4);

        let back = park.restore(&gpu);
        assert_eq!(back.len(), hosts.len());
        for (got, want) in back.iter().zip(&hosts) {
            assert_eq!(got.dims(), want.dims(), "parked shape changed");
            assert_eq!(&got.to_host(&gpu).data, &want.data, "parked data changed");
        }
    }

    /// A bf16 slab must come back as bf16, bit-for-bit, and must cost half an fp32
    /// tensor of the same shape on the host.
    ///
    /// The park moves bytes; it must never change a value's width. Widening on the way
    /// out would double both the transfer and the restored footprint, and the
    /// precision split is decided at each value's production point (`gpu::bf16`), not
    /// here — see the module docs.
    #[test]
    fn park_preserves_bf16_width() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        // Values exactly representable in bf16, so the comparison is exact and the
        // test says something about the transfer rather than about rounding.
        let src = Tensor::new(&[8, 16], (0..128).map(|i| (i as f32 - 64.0) * 0.5).collect());
        let mut slab = BTensor::uninit(&gpu, &[8, 16]);
        slab.store(&gpu, &DTensor::from_host(&gpu, &src));
        let slab = Parked::Bf16(slab);
        let before = slab.to_host(&gpu).data;

        let mut park = HostPark::new(&gpu, InFlight::shared()).expect("park");
        park.evict(&gpu, vec![slab]);
        park.sync_parked();

        // Half the bytes an fp32 [8,16] would take — the width really was preserved.
        assert_eq!(park.host_bytes(), 8 * 16 * 2);

        let back = park.restore(&gpu);
        assert!(
            matches!(back[0], Parked::Bf16(_)),
            "a bf16 slab came back as something else"
        );
        assert_eq!(back[0].to_host(&gpu).data, before, "bf16 round trip changed bits");
    }

    /// A prefetched restore must return exactly what an un-prefetched one does.
    ///
    /// Prefetch is what hides the upload behind a block of compute, so it runs on every
    /// backbone block in training — but it changes *when* the copy is issued, and a
    /// mistake there yields stale data rather than an error.
    #[test]
    fn prefetched_restore_matches_direct_restore() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let want = [
            Tensor::random(&[9, 7], 1.0),
            Tensor::random(&[4, 16], 1.0),
        ];

        let mut direct = HostPark::new(&gpu, InFlight::shared()).expect("park");
        let mut early = HostPark::new(&gpu, InFlight::shared()).expect("park");
        for park in [&mut direct, &mut early] {
            park.evict(
                &gpu,
                want.iter()
                    .map(|t| Parked::from(DTensor::from_host(&gpu, t)))
                    .collect(),
            );
            park.sync_parked();
        }

        // The prefetching park starts its uploads well before it consumes them, with
        // unrelated compute in between — the training-path shape.
        early.prefetch(&gpu);
        let busy = DTensor::zeros(&gpu, &[256, 256]);
        let mut sink = DTensor::uninit(&gpu, &[256, 256]);
        super::super::ops::matmul_nn_into(&gpu, &busy, &busy, &mut sink, 0.0);
        // A second prefetch before consuming must be a harmless no-op.
        early.prefetch(&gpu);

        let prefetched = early.restore(&gpu);
        let plain = direct.restore(&gpu);
        assert_eq!(prefetched.len(), want.len());
        for ((got, base), w) in prefetched.iter().zip(&plain).zip(&want) {
            assert_eq!(got.dims(), w.dims(), "prefetched restore changed the shape");
            assert_eq!(
                got.to_host(&gpu).data,
                w.data,
                "prefetched restore differs from the evicted source"
            );
            assert_eq!(
                base.to_host(&gpu).data,
                w.data,
                "direct restore differs from the evicted source"
            );
        }
    }

    /// Re-evicting the same shapes must not re-allocate pinned memory: page-locking is
    /// expensive, and a training loop parks the same shapes every window.
    #[test]
    fn park_reuses_pinned_slots_across_steps() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let mut park = HostPark::new(&gpu, InFlight::shared()).expect("park");
        let src = Tensor::random(&[16, 32], 1.0);

        park.evict(&gpu, vec![DTensor::from_host(&gpu, &src).into()]);
        park.sync_parked();

        // Each step evicts and restores once, as a real sweep does — the restore is
        // what returns the generation to the spare pool for the next step to reuse.
        let settled = park.allocs;
        for _ in 0..4 {
            let _ = park.restore(&gpu);
            park.evict(&gpu, vec![DTensor::from_host(&gpu, &src).into()]);
            park.sync_parked();
        }
        assert_eq!(
            park.allocs, settled,
            "re-evicting the same shape re-allocated pinned memory"
        );

        // The shapes a real sweep alternates between at one depth: one park serves both
        // the cell and the FFN (different buffer counts), and a balanced `chunk_spans`
        // makes the last chunk one row short. A smaller shape must reuse the slot it
        // already has rather than page-locking a fresh one.
        let short = Tensor::random(&[15, 32], 1.0);
        let pair = Tensor::random(&[16, 32], 1.0);
        let settled = park.allocs;
        for _ in 0..4 {
            let _ = park.restore(&gpu);
            park.evict(&gpu, vec![DTensor::from_host(&gpu, &short).into()]);
            park.sync_parked();
            let _ = park.restore(&gpu);
            park.evict(
                &gpu,
                vec![
                    DTensor::from_host(&gpu, &pair).into(),
                    DTensor::from_host(&gpu, &short).into(),
                ],
            );
            park.sync_parked();
        }
        // The two-buffer eviction needs one slot more than the pool held, so one
        // allocation is legitimate; what must not happen is one per eviction.
        let grew = park.allocs - settled;
        assert!(
            grew <= 1,
            "alternating shapes page-locked {grew} times, expected at most 1"
        );

        // A larger shape legitimately reallocates: no spare is big enough.
        let _ = park.restore(&gpu);
        let b = Tensor::random(&[16, 64], 1.0);
        park.evict(&gpu, vec![DTensor::from_host(&gpu, &b).into()]);
        park.sync_parked();
        assert_eq!(park.host_bytes(), 16 * 64 * 4);
    }

    /// A chunked sweep holds one generation per chunk, so a park's peak slot demand
    /// scales with the number of chunks — not with `IN_FLIGHT_DEPTH`. Retention must
    /// follow that demand, and the restored data must survive the reuse.
    ///
    /// With a fixed cap below the peak, every step displaced more slots than it could
    /// retain and page-locked them again (measured: 256 allocations per step, ~225 ms
    /// of `cuMemHostAlloc` on the critical path).
    #[test]
    fn park_reuses_slots_across_a_chunked_sweep() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let mut park = HostPark::new(&gpu, InFlight::shared()).expect("park");
        // Enough chunks that the live generations alone exceed any small fixed cap.
        let chunks = 12;
        let bufs_per_evict = 6;
        let src: Vec<Tensor> = (0..chunks)
            .map(|c| Tensor::random(&[8 + c % 3, 32], 1.0))
            .collect();

        let mut settled = 0;
        for step in 0..4 {
            // Forward: evict every chunk, all staying live at increasing depth.
            for s in &src {
                let bufs = (0..bufs_per_evict)
                    .map(|_| DTensor::from_host(&gpu, s).into())
                    .collect();
                park.evict(&gpu, bufs);
            }
            // Backward: unwind right to left, checking each chunk comes back intact.
            for s in src.iter().rev() {
                for got in park.restore(&gpu) {
                    assert_eq!(
                        got.to_host(&gpu).data,
                        s.data,
                        "a reused pinned slot returned the wrong data"
                    );
                }
            }
            park.sync_parked();
            // The first sweep legitimately page-locks its working set; later sweeps
            // turn over the same shapes and must allocate nothing.
            if step == 0 {
                settled = park.allocs;
            } else {
                assert_eq!(
                    park.allocs, settled,
                    "sweep {step} re-page-locked slots the pool should have retained"
                );
            }
        }
    }

    /// Device staging must stay at `2·k` timesteps no matter how long the timeline is
    /// — the property the whole plan exists for.
    #[test]
    fn device_footprint_is_independent_of_sequence_length() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let (per_step, k) = (256usize, 8usize);
        let short = OffloadRing::new(&gpu, 64, per_step, k).expect("ring");
        let long = OffloadRing::new(&gpu, 4096, per_step, k).expect("ring");
        assert_eq!(short.device_bytes(), long.device_bytes());
        assert_eq!(long.host_bytes(), 64 * short.host_bytes());
    }
}
