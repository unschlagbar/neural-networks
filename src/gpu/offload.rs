//! Storage for the activations a forward saves and its backward reads.
//!
//! Everything one block saves — its own FFN activations, its cell's cache, its norms'
//! `inv_rms` — lives in one [`Frame`]: a byte range the layers carve typed views out of,
//! in a fixed order that forward and backward both replay. So a block's saved state is
//! one contiguous range, and moving it is one copy.
//!
//! A [`FrameStack`] hands the frames out, in one of two ways:
//!
//!   * **resident** ([`ResidentFrames`]) — one device buffer per stack depth, reused
//!     across windows. The encoder, the decoder, eval and `GPU_NO_OFFLOAD`.
//!   * **offloaded** ([`Offload`]) — the backbone. Every frame is copied to a pinned host
//!     stack as its forward ends and comes back before its backward, through a fixed
//!     ring of [`RING_SLOTS`] device slots. Nothing is allocated or freed per frame.
//!
//! Training is bounded by activation VRAM, and that bound is linear in the sequence
//! length: step 0's saved tensors must survive until backward unwinds to `t = 0`, so
//! no reordering of the forward loop frees them. Only moving them off the device — or
//! recomputing them — changes the scaling.

use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, DevicePtr, PinnedHostSlice, result};
use std::{cell::RefCell, rc::Rc, sync::Arc};

use super::buf::{fits, size_class};
use super::ops::SlabBuf;
use super::{GTensor, Gpu};

/// Alignment of every region carved from a frame, in bytes — what `cuMemAlloc` itself
/// guarantees, so a view is as aligned as an allocation would be.
const ALIGN: usize = 256;

pub fn align_up(n: usize) -> usize {
    n.div_ceil(ALIGN) * ALIGN
}

/// A byte range that one block's saved activations are carved out of.
///
/// Carving is sequential: each call takes the next aligned region. A layer carves in a
/// single function that its forward and its backward both call, so the two see the same
/// regions without either storing an offset. The views own nothing; the storage stays the
/// [`FrameStack`]'s.
pub struct Frame {
    base: u64,
    cap: usize,
    used: usize,
}

impl Frame {
    /// A frame with no memory behind it, for sizing a layout: carve it, then read
    /// [`bytes`](Self::bytes). Its views must never reach a kernel.
    pub fn measure() -> Self {
        Self {
            base: 0,
            cap: usize::MAX,
            used: 0,
        }
    }

    fn at(base: u64, cap: usize) -> Self {
        Self { base, cap, used: 0 }
    }

    /// A frame over the whole of `buf`, which must outlive every view carved from it.
    pub fn over(gpu: &Gpu, buf: &CudaSlice<u8>) -> Self {
        let (p, _g) = buf.device_ptr(&gpu.stream);
        Self::at(p, buf.len())
    }

    /// Bytes carved so far; after a full carve, the frame's size.
    pub fn bytes(&self) -> usize {
        self.used
    }

    fn region(&mut self, bytes: usize) -> u64 {
        let off = align_up(self.used);
        assert!(
            off + bytes <= self.cap,
            "frame: carving {} B past a {} B frame",
            off + bytes,
            self.cap
        );
        self.used = off + bytes;
        self.base + off as u64
    }

    pub fn f32(&mut self, gpu: &Gpu, dims: &[usize]) -> GTensor<f32> {
        let n: usize = dims.iter().product();
        GTensor::view_at(gpu, self.region(n * 4), dims)
    }

    pub fn bf16(&mut self, gpu: &Gpu, dims: &[usize]) -> GTensor<u16> {
        let n: usize = dims.iter().product();
        GTensor::view_at(gpu, self.region(n * 2), dims)
    }

    /// A slab at an explicit width — see [`SlabBuf::new_width`].
    pub fn slab(&mut self, gpu: &Gpu, dims: &[usize], bf16: bool) -> SlabBuf {
        if bf16 {
            SlabBuf::Bf16(self.bf16(gpu, dims))
        } else {
            SlabBuf::F32(self.f32(gpu, dims))
        }
    }
}

/// Where a block's frames live. See the module docs.
pub enum FrameStack {
    Resident(ResidentFrames),
    Offload(SharedOffload),
}

impl Default for FrameStack {
    fn default() -> Self {
        FrameStack::Resident(ResidentFrames::default())
    }
}

impl FrameStack {
    /// A frame of `bytes` for the forward about to run. `stacked` keeps the frames
    /// already pushed (a chunked sweep, whose earlier chunks are still owed a backward);
    /// otherwise a resident stack reuses its one frame.
    pub fn push(&mut self, gpu: &Gpu, bytes: usize, stacked: bool) -> Frame {
        match self {
            FrameStack::Resident(r) => {
                if !stacked {
                    r.reset();
                }
                r.push(gpu, bytes)
            }
            FrameStack::Offload(o) => o.borrow_mut().push(gpu, bytes),
        }
    }

    /// The forward that filled the last pushed frame has been issued.
    pub fn pushed(&mut self, gpu: &Gpu) {
        if let FrameStack::Offload(o) = self {
            o.borrow_mut().pushed(gpu);
        }
    }

    /// The last pushed frame, back on the device for its backward.
    pub fn pop(&mut self, gpu: &Gpu) -> Frame {
        match self {
            FrameStack::Resident(r) => r.pop(gpu),
            FrameStack::Offload(o) => o.borrow_mut().pop(gpu),
        }
    }

    /// The backward that read the last popped frame has been issued.
    pub fn popped(&mut self, gpu: &Gpu) {
        if let FrameStack::Offload(o) = self {
            o.borrow_mut().popped(gpu);
        }
    }

    /// Forget every frame, keeping the memory. An offloaded stack is shared, so its
    /// owner resets it once per sweep instead — see [`Offload::reset`].
    pub fn reset(&mut self) {
        if let FrameStack::Resident(r) = self {
            r.reset();
        }
    }

    /// Forget every frame and free a resident stack's memory.
    pub fn release(&mut self) {
        if let FrameStack::Resident(r) = self {
            r.release();
        }
    }

    /// Device bytes a resident stack holds. An offloaded one holds its ring, which is
    /// the model's, not the block's.
    pub fn device_bytes(&self) -> usize {
        match self {
            FrameStack::Resident(r) => r.device_bytes(),
            FrameStack::Offload(_) => 0,
        }
    }

    /// The most recently used frame of a resident stack, for diagnostics that read a
    /// cache after the fact. `None` when offloaded.
    pub fn recent(&self, gpu: &Gpu) -> Option<Frame> {
        match self {
            FrameStack::Resident(r) => r.recent(gpu),
            FrameStack::Offload(_) => None,
        }
    }
}

/// Frames kept on the device: one buffer per stack depth, reused across windows.
///
/// A stack because a chunked sweep forwards every chunk before unwinding any, so chunk
/// c's frame must survive chunk c+1's forward. Each depth reuses its buffer while it fits
/// (see [`fits`]), so a steady window shape allocates nothing.
#[derive(Default)]
pub struct ResidentFrames {
    bufs: Vec<CudaSlice<u8>>,
    depth: usize,
}

impl ResidentFrames {
    pub fn push(&mut self, gpu: &Gpu, bytes: usize) -> Frame {
        let d = self.depth;
        if d == self.bufs.len() || !fits(self.bufs[d].len(), bytes) {
            // SAFETY: a frame's regions are written by the forward before the backward
            // reads them.
            let buf = unsafe { gpu.stream.alloc::<u8>(size_class(bytes).max(1)) }
                .expect("frame alloc");
            if d == self.bufs.len() {
                self.bufs.push(buf);
            } else {
                self.bufs[d] = buf;
            }
        }
        self.depth += 1;
        Frame::over(gpu, &self.bufs[d])
    }

    pub fn pop(&mut self, gpu: &Gpu) -> Frame {
        assert!(self.depth > 0, "frame stack: pop with nothing pushed");
        self.depth -= 1;
        // The address the matching push handed out: nothing reallocates this depth
        // until it is pushed again.
        Frame::over(gpu, &self.bufs[self.depth])
    }

    pub fn reset(&mut self) {
        self.depth = 0;
    }

    pub fn release(&mut self) {
        self.bufs.clear();
        self.depth = 0;
    }

    pub fn device_bytes(&self) -> usize {
        self.bufs.iter().map(|b| b.len()).sum()
    }

    pub fn recent(&self, gpu: &Gpu) -> Option<Frame> {
        let buf = self.bufs.get(self.depth.saturating_sub(1))?;
        Some(Frame::over(gpu, buf))
    }
}

/// Device slots in the offload ring.
///
/// Three because each direction wants a block of compute between a copy and the next
/// write to its slot: forward writes frame j+3 into frame j's slot, so frame j's D2H has
/// had frames j+1 and j+2 to hide behind; backward reads frame j while j-1 and j-2 are
/// already on their way up.
pub const RING_SLOTS: usize = 3;

/// One device slot of the ring.
struct Slot {
    /// Held for ownership; every access goes through `addr`.
    #[allow(dead_code)]
    mem: CudaSlice<u8>,
    addr: u64,
    /// Index into the host stack of the frame this slot holds a valid copy of.
    holds: Option<usize>,
    /// The last transfer that touched this slot. Compute waits on it before writing the
    /// slot (a D2H may still be reading it) or reading it (an H2D may still be filling
    /// it).
    done: Option<CudaEvent>,
}

/// Pinned host memory the offloaded frames live in, as one LIFO.
struct HostStack {
    mem: Option<PinnedHostSlice<u8>>,
    ptr: *mut u8,
    cap: usize,
    /// `(offset, bytes)` of every frame pushed and not yet popped, bottom first.
    frames: Vec<(usize, usize)>,
}

impl HostStack {
    fn top(&self) -> usize {
        self.frames.last().map_or(0, |&(o, b)| o + b)
    }

    fn slice(&mut self, i: usize) -> &mut [u8] {
        let (off, bytes) = self.frames[i];
        // SAFETY: `off + bytes <= cap` was checked on push, and the pinned allocation is
        // alive for as long as `mem` is.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(off), bytes) }
    }
}

/// The frames of a whole stack, staged through pinned host memory.
///
/// Shared by every block of the stack, because its frames form one LIFO: the backbone
/// forwards chunk-major (chunk c through every block, then chunk c+1) and unwinds in
/// exactly the reverse order, so the frame a backward wants is always the last one
/// pushed. That is what lets one host stack hold them all, and lets [`pop`](Self::pop)
/// prefetch the frames below it without being told which block runs next.
///
/// The ring is a write-through cache of the host stack: every frame is copied up when its
/// forward ends, and a slot that still holds a frame when its backward comes serves it
/// without a copy back — which is the case for the last frames of a sweep.
pub struct Offload {
    xfer: Arc<CudaStream>,
    slots: [Slot; RING_SLOTS],
    slot_bytes: usize,
    host: HostStack,
    /// The slot of the frame between `push` and `pushed`, or `pop` and `popped`.
    open: Option<usize>,
    /// H2D copies issued, for the tests to observe which pops were served in place.
    #[cfg(test)]
    loads: usize,
}

pub type SharedOffload = Rc<RefCell<Offload>>;

impl Offload {
    /// A ring of [`RING_SLOTS`] slots of `slot_bytes` each, and an empty host stack.
    pub fn new(gpu: &Gpu, slot_bytes: usize) -> Self {
        let xfer = gpu
            .context
            .new_stream()
            .expect("offload: transfer stream creation failed");
        let slots = Self::alloc_slots(gpu, slot_bytes);
        Self {
            xfer,
            slots,
            slot_bytes,
            host: HostStack {
                mem: None,
                ptr: std::ptr::null_mut(),
                cap: 0,
                frames: Vec::new(),
            },
            open: None,
            #[cfg(test)]
            loads: 0,
        }
    }

    pub fn shared(gpu: &Gpu, slot_bytes: usize) -> SharedOffload {
        Rc::new(RefCell::new(Self::new(gpu, slot_bytes)))
    }

    fn alloc_slots(gpu: &Gpu, bytes: usize) -> [Slot; RING_SLOTS] {
        let slots = std::array::from_fn(|_| {
            // SAFETY: a slot is written (by a forward or an H2D) before it is read.
            let mem = unsafe { gpu.stream.alloc::<u8>(bytes.max(1)) }.expect("offload slot");
            let addr = mem.device_ptr(&gpu.stream).0;
            Slot {
                mem,
                addr,
                holds: None,
                done: None,
            }
        });
        // The allocation is stream-ordered on the compute stream, which the transfer
        // stream knows nothing about; it must have happened before a copy touches it.
        gpu.stream.synchronize().expect("offload: slot alloc");
        slots
    }

    /// Make room for a sweep of `total` frame bytes, none larger than `largest`. Call
    /// with the stack empty, before the sweep's first push, so neither the ring nor the
    /// host stack grows inside it.
    pub fn reserve(&mut self, gpu: &Gpu, total: usize, largest: usize) {
        assert!(
            self.host.frames.is_empty(),
            "offload: reserve with frames still pushed"
        );
        if largest > self.slot_bytes {
            self.grow_slots(gpu, largest);
        }
        if total > self.host.cap {
            // Headroom: windows vary by well under a percent, and page-locking ~10 GB
            // again for every one that sets a new maximum costs seconds.
            self.grow_host(gpu, total + total / 16);
        }
    }

    /// Forget every frame. For the start of a sweep, and for one abandoned before its
    /// backward consumed the frames.
    pub fn reset(&mut self) {
        assert!(self.open.is_none(), "offload: reset inside a frame");
        self.host.frames.clear();
        for s in &mut self.slots {
            s.holds = None;
        }
    }

    pub fn host_bytes(&self) -> usize {
        self.host.cap
    }

    pub fn device_bytes(&self) -> usize {
        RING_SLOTS * self.slot_bytes
    }

    /// Frames pushed and not yet popped.
    pub fn depth(&self) -> usize {
        self.host.frames.len()
    }

    /// Block until every queued copy has landed. Teardown and tests.
    pub fn sync(&self) {
        self.xfer.synchronize().expect("offload: sync");
    }

    fn grow_slots(&mut self, gpu: &Gpu, bytes: usize) {
        gpu.stream.synchronize().expect("offload: sync");
        self.sync();
        self.slots = Self::alloc_slots(gpu, bytes);
        self.slot_bytes = bytes;
    }

    fn grow_host(&mut self, gpu: &Gpu, bytes: usize) {
        // Copies still in flight read or write the old allocation.
        self.sync();
        let top = self.host.top();
        if top == 0 {
            // Nothing to carry over, so the old allocation goes first: holding both
            // pins twice the stack at once, ~20 GB at the backbone's size.
            self.host.mem = None;
            self.host.ptr = std::ptr::null_mut();
            self.host.cap = 0;
        }
        // SAFETY: the bytes below `top` are copied over; everything above is written by
        // a D2H before an H2D reads it.
        let mut mem = unsafe { gpu.context.alloc_pinned::<u8>(bytes) }
            .expect("offload: pinned host alloc");
        let ptr = mem.as_mut_ptr().expect("offload: pinned ptr");
        if top > 0 {
            // SAFETY: both ranges are live pinned allocations of at least `top` bytes.
            unsafe { std::ptr::copy_nonoverlapping(self.host.ptr, ptr, top) };
        }
        self.host.mem = Some(mem);
        self.host.ptr = ptr;
        self.host.cap = bytes;
    }

    /// The slot to overwrite: one that holds nothing, else the one whose frame is
    /// furthest from being needed. `keep` holds the slots that must not be touched.
    fn victim(&self, keep: impl Fn(usize, &Slot) -> bool) -> Option<usize> {
        let free = (0..RING_SLOTS).filter(|&i| !keep(i, &self.slots[i]));
        free.min_by_key(|&i| self.slots[i].holds.map_or((0, 0), |h| (1, h)))
    }

    fn push(&mut self, gpu: &Gpu, bytes: usize) -> Frame {
        assert!(self.open.is_none(), "offload: push inside a frame");
        if bytes > self.slot_bytes {
            self.grow_slots(gpu, bytes);
        }
        let j = self.host.frames.len();
        let off = align_up(self.host.top());
        if off + bytes > self.host.cap {
            self.grow_host(gpu, (off + bytes).max(2 * self.host.cap));
        }
        self.host.frames.push((off, bytes));
        // Whatever a slot held at or above `j` belonged to frames already popped.
        for s in &mut self.slots {
            if s.holds.is_some_and(|h| h >= j) {
                s.holds = None;
            }
        }
        let s = self.victim(|_, _| false).expect("offload: no slot");
        let slot = &mut self.slots[s];
        if let Some(ev) = &slot.done {
            gpu.stream.wait(ev).expect("offload: wait for slot");
        }
        slot.holds = None;
        self.open = Some(s);
        Frame::at(slot.addr, self.slot_bytes)
    }

    fn pushed(&mut self, gpu: &Gpu) {
        let s = self.open.take().expect("offload: pushed without push");
        let j = self.host.frames.len() - 1;
        let produced = gpu.stream.record_event(None).expect("offload: record");
        self.xfer.wait(&produced).expect("offload: xfer waits");
        let addr = self.slots[s].addr;
        let dst = self.host.slice(j);
        self.xfer.context().bind_to_thread().expect("offload: bind");
        // SAFETY: `dst` is pinned and outlives the copy (the host stack only reallocates
        // after syncing this stream); the slot holds `dst.len()` bytes of frame `j`.
        unsafe { result::memcpy_dtoh_async(dst, addr, self.xfer.cu_stream()) }
            .expect("offload: D2H");
        let slot = &mut self.slots[s];
        slot.done = Some(self.xfer.record_event(None).expect("offload: record"));
        slot.holds = Some(j);
    }

    /// Issue frame `i`'s H2D into slot `s`, once compute is done with what it held.
    fn load(&mut self, gpu: &Gpu, i: usize, s: usize) {
        let consumed = gpu.stream.record_event(None).expect("offload: record");
        self.xfer.wait(&consumed).expect("offload: xfer waits");
        let addr = self.slots[s].addr;
        let src = self.host.slice(i);
        self.xfer.context().bind_to_thread().expect("offload: bind");
        // SAFETY: as in `pushed`, and the D2H that filled `src` is earlier on this stream.
        unsafe { result::memcpy_htod_async(addr, src, self.xfer.cu_stream()) }
            .expect("offload: H2D");
        let slot = &mut self.slots[s];
        slot.done = Some(self.xfer.record_event(None).expect("offload: record"));
        slot.holds = Some(i);
        #[cfg(test)]
        {
            self.loads += 1;
        }
    }

    fn pop(&mut self, gpu: &Gpu) -> Frame {
        assert!(self.open.is_none(), "offload: pop inside a frame");
        let j = self
            .host
            .frames
            .len()
            .checked_sub(1)
            .expect("offload: pop with nothing pushed");
        let held = (0..RING_SLOTS).find(|&i| self.slots[i].holds == Some(j));
        let s = match held {
            Some(s) => s,
            None => {
                let s = self.victim(|_, _| false).expect("offload: no slot");
                self.load(gpu, j, s);
                s
            }
        };
        if let Some(ev) = &self.slots[s].done {
            gpu.stream.wait(ev).expect("offload: wait for fill");
        }
        self.open = Some(s);
        // The frames below are the next backwards to run; start them on their way up
        // while this one is read.
        for i in (j.saturating_sub(RING_SLOTS - 1)..j).rev() {
            if self.slots.iter().any(|sl| sl.holds == Some(i)) {
                continue;
            }
            let keep = |k: usize, sl: &Slot| k == s || sl.holds.is_some_and(|h| h >= i && h <= j);
            match self.victim(keep) {
                Some(v) => self.load(gpu, i, v),
                None => break,
            }
        }
        let slot = &self.slots[s];
        Frame::at(slot.addr, self.slot_bytes)
    }

    fn popped(&mut self, _gpu: &Gpu) {
        let s = self.open.take().expect("offload: popped without pop");
        // A backward may write into its frame (the sLSTM reuses `g` for its deltas), so
        // what the slot holds is no longer the frame.
        self.slots[s].holds = None;
        self.host.frames.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    /// Push `n` frames through `stack`, frame `i` filled with the value `i` at `lens[i]`
    /// floats.
    fn push_tagged(gpu: &Gpu, stack: &mut FrameStack, lens: &[usize]) {
        for (i, &len) in lens.iter().enumerate() {
            let mut f = stack.push(gpu, len * 4, true);
            let mut v = f.f32(gpu, &[len]);
            let tag = GTensor::from_host(gpu, &Tensor::new(&[len], vec![i as f32; len]));
            v.copy_from(gpu, &tag);
            stack.pushed(gpu);
        }
    }

    /// Pop every frame, checking each comes back as it was pushed.
    fn pop_checked(gpu: &Gpu, stack: &mut FrameStack, lens: &[usize]) {
        for (i, &len) in lens.iter().enumerate().rev() {
            let mut f = stack.pop(gpu);
            let got = f.f32(gpu, &[len]).to_host(gpu);
            assert!(
                got.data.iter().all(|&v| v == i as f32),
                "frame {i} came back changed"
            );
            stack.popped(gpu);
        }
    }

    /// Regions are aligned and sequential, and a measuring frame sizes a layout exactly
    /// as a real carve lays it out.
    #[test]
    fn carve_is_aligned_and_measurable() {
        let Some(gpu) = crate::gpu::test_gpu() else {
            return;
        };
        let carve = |f: &mut Frame| {
            let a = f.f32(&gpu, &[3]);
            let b = f.bf16(&gpu, &[5, 7]);
            let c = f.slab(&gpu, &[2, 2], false);
            (a, b, c)
        };
        let mut m = Frame::measure();
        carve(&mut m);
        assert_eq!(m.bytes(), 2 * ALIGN + 16);

        let mut r = ResidentFrames::default();
        let mut f = r.push(&gpu, m.bytes());
        let (a, b, _) = carve(&mut f);
        let addr = |p: u64| p % ALIGN as u64;
        assert_eq!(addr(a.buf.device_ptr(&gpu.stream).0), 0);
        assert_eq!(addr(b.buf.device_ptr(&gpu.stream).0), 0);
        assert_eq!(f.bytes(), m.bytes());
    }

    /// A resident stack returns every frame intact, and a steady shape reuses the same
    /// device addresses window after window.
    #[test]
    fn resident_frames_roundtrip_and_reuse() {
        let Some(gpu) = crate::gpu::test_gpu() else {
            return;
        };
        let lens = [64, 65, 64, 30];
        let mut stack = FrameStack::default();
        let mut first = Vec::new();
        for window in 0..3 {
            push_tagged(&gpu, &mut stack, &lens);
            let FrameStack::Resident(r) = &stack else {
                unreachable!()
            };
            let addrs: Vec<u64> = r.bufs.iter().map(|b| b.device_ptr(&gpu.stream).0).collect();
            if window == 0 {
                first = addrs;
            } else {
                assert_eq!(addrs, first, "window {window} reallocated a frame");
            }
            pop_checked(&gpu, &mut stack, &lens);
        }
    }

    /// Frames pushed through the ring come back exactly, ragged sizes included, with
    /// the compute stream kept busy so the hand-placed events are what orders the copies.
    #[test]
    fn offload_roundtrip_survives_contention() {
        let Some(gpu) = crate::gpu::test_gpu() else {
            return;
        };
        let lens: Vec<usize> = (0..11).map(|i| 4096 + 37 * i).collect();
        let off = Offload::shared(&gpu, lens.iter().max().unwrap() * 4);
        let mut stack = FrameStack::Offload(off.clone());
        let busy = GTensor::zeros(&gpu, &[512, 512]);
        let mut sink = GTensor::uninit(&gpu, &[512, 512]);
        for _ in 0..2 {
            for (i, &len) in lens.iter().enumerate() {
                let mut f = stack.push(&gpu, len * 4, true);
                let mut v = f.f32(&gpu, &[len]);
                super::super::ops::matmul_nn_into(&gpu, &busy, &busy, &mut sink, 0.0);
                let tag = GTensor::from_host(&gpu, &Tensor::new(&[len], vec![i as f32; len]));
                v.copy_from(&gpu, &tag);
                stack.pushed(&gpu);
            }
            for (i, &len) in lens.iter().enumerate().rev() {
                super::super::ops::matmul_nn_into(&gpu, &busy, &busy, &mut sink, 0.0);
                let mut f = stack.pop(&gpu);
                let got = f.f32(&gpu, &[len]).to_host(&gpu);
                assert!(
                    got.data.iter().all(|&v| v == i as f32),
                    "frame {i} came back changed — cross-stream ordering is wrong"
                );
                stack.popped(&gpu);
            }
            assert_eq!(off.borrow().depth(), 0);
        }
        off.borrow().sync();
    }

    /// The last frames of a sweep are still in the ring when backward starts, so they
    /// come back without a copy; every earlier one takes exactly one H2D.
    #[test]
    fn turnaround_frames_skip_the_upload() {
        let Some(gpu) = crate::gpu::test_gpu() else {
            return;
        };
        let lens = [256; 8];
        let off = Offload::shared(&gpu, 256 * 4);
        let mut stack = FrameStack::Offload(off.clone());
        push_tagged(&gpu, &mut stack, &lens);
        pop_checked(&gpu, &mut stack, &lens);
        assert_eq!(off.borrow().loads, lens.len() - RING_SLOTS);
    }

    /// `reserve` sizes both sides once; a sweep inside the reservation grows neither.
    #[test]
    fn reserved_sweep_does_not_grow() {
        let Some(gpu) = crate::gpu::test_gpu() else {
            return;
        };
        let lens = [100, 300, 200, 300];
        let total: usize = lens.iter().map(|&l| align_up(l * 4)).sum();
        let off = Offload::shared(&gpu, 16);
        off.borrow_mut().reserve(&gpu, total, 300 * 4);
        let (host, dev) = (off.borrow().host_bytes(), off.borrow().device_bytes());
        let mut stack = FrameStack::Offload(off.clone());
        for _ in 0..3 {
            off.borrow_mut().reset();
            push_tagged(&gpu, &mut stack, &lens);
            pop_checked(&gpu, &mut stack, &lens);
        }
        assert_eq!(off.borrow().host_bytes(), host, "host stack grew");
        assert_eq!(off.borrow().device_bytes(), dev, "ring grew");
    }
}
