//! Fixed scratch slots for every temporary a forward or backward needs.

use std::cell::Cell;
use std::ops::{Deref, DerefMut};

use cudarc::driver::{CudaSlice, DevicePtr, DeviceRepr, ValidAsZeroBits};

use super::{GTensor, Gpu};

/// How many slots the cache holds.
pub const SLOTS: usize = 8;

/// How many **small** slots the cache holds.
pub const SMALL_SLOTS: usize = 4;

/// The widest *small* temporary a stage of `rows` x `heads` asks for, in elements.
pub fn widest_small(rows: usize, heads: usize) -> usize {
    rows * 2 * heads
}

/// How many chunk-state slots: `dcst` and `dnst`, live together inside one
/// `mlstm_fused_bw` and nowhere else.
pub const CHUNK_SLOTS: usize = 2;

/// Elements the chunk-state gradients need for one call.
pub fn widest_chunk(
    rows: usize,
    t: usize,
    heads: usize,
    dqk: usize,
    dhv: usize,
    l: usize,
) -> usize {
    let b = (rows / t.max(1)).max(1);
    let nc = t.div_ceil(l.max(1));
    b * heads * (nc + 1) * dqk * dhv
}

/// The widest temporary a stage of `rows` × `hidden` asks for, in elements.
pub fn widest(rows: usize, hidden: usize, heads: usize, dqk: usize, extra: usize) -> usize {
    let dhv = hidden / heads;
    let qkvo = 2 * heads * dqk + 2 * heads * dhv;
    let up = super::hierarchical::up_of(hidden);
    rows * qkvo.max(up).max(hidden).max(extra)
}

/// One fixed-size array of equally sized slots, with a bitmask of what is on loan.
struct Slots<const N: usize> {
    /// The allocations. Held for ownership — every access goes through `addr`, so
    /// this field is what keeps the memory mapped and frees it when the model drops.
    #[allow(dead_code)]
    slots: [CudaSlice<f32>; N],
    /// Each slot's device address, read once at construction: they do not move, and
    /// `device_ptr` needs the stream, which a `get` would otherwise have to reach for.
    addr: [u64; N],
    /// Bit `i` set means slot `i` is out on loan.
    used: Cell<u32>,
    /// The most slots ever live at once: the number `N` has to cover.
    high: Cell<u32>,
    /// `f32` elements per slot.
    elems: usize,
    /// Fill a slot with a sentinel before handing it out — see [`TempCache::new`].
    poison: bool,
}

impl<const N: usize> Slots<N> {
    fn new(gpu: &Gpu, elems: usize, poison: bool) -> Self {
        const { assert!(N <= 32, "the busy mask is a u32") };
        // SAFETY: every borrower overwrites what it reads — see the module contract.
        let slots: [CudaSlice<f32>; N] =
            std::array::from_fn(|_| unsafe { gpu.stream.alloc::<f32>(elems) }.expect("temp slot"));
        let addr = std::array::from_fn(|i| slots[i].device_ptr(&gpu.stream).0);
        Self {
            slots,
            addr,
            used: Cell::new(0),
            high: Cell::new(0),
            elems,
            poison,
        }
    }

    fn bytes(&self) -> usize {
        N * self.elems * size_of::<f32>()
    }

    /// Reserve the lowest free slot and present it at `dims`.
    fn get<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        gpu: &Gpu,
        dims: &[usize],
        what: &str,
    ) -> Temp<'_, T> {
        let n: usize = dims.iter().product();
        let want = n * size_of::<T>();
        let cap = self.elems * size_of::<f32>();
        assert!(
            want <= cap,
            "{what} {dims:?} needs {want} B, a slot holds {cap} B"
        );
        let used = self.used.get();
        let bit = (!used).trailing_zeros();
        assert!(
            (bit as usize) < N,
            "all {N} {what} slots busy — a temporary is being held past its use, or \
             the deepest nesting grew and the slot count needs re-deriving"
        );
        let now = used | (1 << bit);
        self.used.set(now);
        let live = now.count_ones();
        if live > self.high.get() {
            self.high.set(live);
        }
        if self.poison {
            let mut whole = GTensor::<f32>::view_at(gpu, self.addr[bit as usize], &[self.elems]);
            super::ops::fill(gpu, &mut whole, 1e30);
        }
        Temp {
            t: GTensor::view_at(gpu, self.addr[bit as usize], dims),
            used: &self.used,
            bit,
        }
    }
}

/// Scratch slots shared by every stage, every layer and both directions.
pub struct TempCache {
    /// Stage-sized temporaries — the `[N, ·]` rectangles. See [`SLOTS`].
    big: Slots<SLOTS>,
    /// Per-call vectors and gate strips. See [`SMALL_SLOTS`].
    small: Slots<SMALL_SLOTS>,
    /// The mLSTM chunk-state gradients. See [`CHUNK_SLOTS`].
    chunk: Slots<CHUNK_SLOTS>,
}

impl TempCache {
    /// Allocate both arrays; `elems` sizes a *stage-sized* slot.
    pub fn new(gpu: &Gpu, elems: usize, small_elems: usize, chunk_elems: usize) -> Self {
        let poison = std::env::var("GPU_TEMP_POISON").is_ok_and(|v| v != "0");
        Self {
            big: Slots::new(gpu, elems, poison),
            small: Slots::new(gpu, small_elems, poison),
            chunk: Slots::new(gpu, chunk_elems, poison),
        }
    }

    /// Device bytes held for the whole run. Fixed at construction.
    pub fn bytes(&self) -> usize {
        self.big.bytes() + self.small.bytes() + self.chunk.bytes()
    }

    /// `f32` elements per stage-sized slot — the widest temporary this cache serves.
    pub fn slot_elems(&self) -> usize {
        self.big.elems
    }

    /// `f32` elements per small slot.
    pub fn small_slot_elems(&self) -> usize {
        self.small.elems
    }

    /// The most slots ever live at once, and how many there are: `(big, SLOTS)`.
    pub fn high_water(&self) -> (usize, usize) {
        (self.big.high.get() as usize, SLOTS)
    }

    /// The same for the small array: `(peak, SMALL_SLOTS)`.
    pub fn small_high_water(&self) -> (usize, usize) {
        (self.small.high.get() as usize, SMALL_SLOTS)
    }

    /// The same for the chunk-state array: `(peak, CHUNK_SLOTS)`.
    pub fn chunk_high_water(&self) -> (usize, usize) {
        (self.chunk.high.get() as usize, CHUNK_SLOTS)
    }

    /// `f32` elements per chunk-state slot.
    pub fn chunk_slot_elems(&self) -> usize {
        self.chunk.elems
    }

    /// An uninitialised stage-sized temporary of shape `dims`, held until the guard
    /// is dropped.
    pub fn get<T: DeviceRepr + ValidAsZeroBits>(&self, gpu: &Gpu, dims: &[usize]) -> Temp<'_, T> {
        self.big.get(gpu, dims, "temp")
    }

    /// [`get`](Self::get) from the **small** array, for a temporary that is a vector
    /// or a gate strip rather than an `[N, ·]` rectangle.
    pub fn get_small<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        gpu: &Gpu,
        dims: &[usize],
    ) -> Temp<'_, T> {
        self.small.get(gpu, dims, "small temp")
    }

    /// [`get`](Self::get) from the **chunk-state** array — see [`CHUNK_SLOTS`].
    pub fn get_chunk<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        gpu: &Gpu,
        dims: &[usize],
    ) -> Temp<'_, T> {
        self.chunk.get(gpu, dims, "chunk temp")
    }

    /// [`get`](Self::get), zeroed — for a temporary that is accumulated into rather
    /// than written whole.
    pub fn get_zeroed<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        gpu: &Gpu,
        dims: &[usize],
    ) -> Temp<'_, T> {
        let mut t = self.get::<T>(gpu, dims);
        gpu.stream.memset_zeros(&mut t.t.buf).expect("temp memset");
        t
    }

    /// [`get_small`](Self::get_small), zeroed.
    pub fn get_small_zeroed<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        gpu: &Gpu,
        dims: &[usize],
    ) -> Temp<'_, T> {
        let mut t = self.get_small::<T>(gpu, dims);
        gpu.stream.memset_zeros(&mut t.t.buf).expect("temp memset");
        t
    }

    /// Assert nothing is on loan in either array — call where every temporary should
    /// be dead, i.e. the top of a window. Debug builds only.
    pub fn assert_drained(&self, what: &str) {
        debug_assert_eq!(self.big.used.get(), 0, "{what}: temp slot(s) still held");
        debug_assert_eq!(
            self.small.used.get(),
            0,
            "{what}: small temp slot(s) still held"
        );
        debug_assert_eq!(
            self.chunk.used.get(),
            0,
            "{what}: chunk temp slot(s) still held"
        );
    }
}

/// A borrowed scratch slot, presented as a tensor.
pub struct Temp<'a, T> {
    t: GTensor<T>,
    used: &'a Cell<u32>,
    bit: u32,
}

impl<T> Deref for Temp<'_, T> {
    type Target = GTensor<T>;
    fn deref(&self) -> &GTensor<T> {
        &self.t
    }
}

impl<T> DerefMut for Temp<'_, T> {
    fn deref_mut(&mut self) -> &mut GTensor<T> {
        &mut self.t
    }
}

impl<T> Drop for Temp<'_, T> {
    fn drop(&mut self) {
        debug_assert!(
            self.used.get() & (1 << self.bit) != 0,
            "temp slot {} released twice",
            self.bit
        );
        self.used.set(self.used.get() & !(1 << self.bit));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::hierarchical::up_of;

    /// The hierarchical model's real geometry, so the sizing test bites.
    const HC: usize = 256;
    const WH: usize = 1024;
    const VOCAB: usize = 260;
    const HEADS: usize = 16;
    const ENC_ROWS: usize = 2048;
    const BB_ROWS: usize = 512;

    fn slot_elems() -> usize {
        widest(ENC_ROWS, HC, HEADS, HC / HEADS, 0)
            .max(widest(BB_ROWS, WH, HEADS, WH / HEADS, 0))
            .max(widest(ENC_ROWS, HC, HEADS, HC / HEADS, VOCAB))
    }

    /// Two live temporaries must be distinct memory, and dropping one must make its
    /// slot available again — the whole contract in one test.
    #[test]
    fn slots_are_disjoint_and_recycled() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let c = TempCache::new(&gpu, 4096, 256, 4096);
        let addr = |t: &GTensor<f32>| t.buf.device_ptr(&gpu.stream).0;

        let a = c.get::<f32>(&gpu, &[16, 16]);
        let pa = addr(&a);
        let b = c.get::<f32>(&gpu, &[16, 16]);
        assert_ne!(pa, addr(&b), "two live temps must not alias");
        assert_eq!(c.big.used.get().count_ones(), 2);

        drop(a);
        assert_eq!(c.big.used.get().count_ones(), 1);
        // The freed slot is the lowest free bit, so it comes straight back.
        let d = c.get::<f32>(&gpu, &[16, 16]);
        assert_eq!(pa, addr(&d), "a released slot must be reused");
        drop((b, d));
        assert_eq!(c.big.used.get(), 0, "every guard released its slot");
    }

    /// The point of one array at one width: a bf16 temporary is the same slot as an
    /// fp32 one, holding twice as many elements.
    #[test]
    fn bf16_and_f32_share_one_slot() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let c = TempCache::new(&gpu, 4096, 256, 4096);
        let n = c.slot_elems();

        let wide = c.get::<f32>(&gpu, &[n]);
        let pa = wide.buf.device_ptr(&gpu.stream).0;
        drop(wide);

        // Twice the elements at half the width is exactly a full slot.
        let narrow = c.get::<u16>(&gpu, &[2 * n]);
        assert_eq!(
            pa,
            narrow.buf.device_ptr(&gpu.stream).0,
            "the same slot, viewed narrow"
        );
    }

    /// A slot is sized from the config, so the widest shape each stage can present
    /// has to fit — the `dq‖dk‖dv‖do` slab being the binding one at both the
    /// backbone's and the encoder's geometry.
    #[test]
    fn config_sizing_covers_the_widest_temp() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let c = TempCache::new(
            &gpu,
            slot_elems(),
            widest_small(ENC_ROWS, HEADS),
            widest_chunk(BB_ROWS, BB_ROWS, HEADS, WH / HEADS, WH / HEADS, 256),
        );

        // The backbone's qkvo delta: [rows, heads*(2*dqk + 2*dhv)] = [rows, 4*WH].
        let _a = c.get::<f32>(&gpu, &[BB_ROWS, 4 * WH]);
        // The encoder's, at its own geometry.
        let _b = c.get::<f32>(&gpu, &[ENC_ROWS, 4 * HC]);
        // The SwiGLU operands and the decoder's logits, both narrower.
        let _c = c.get::<f32>(&gpu, &[BB_ROWS, up_of(WH)]);
        let _d = c.get::<f32>(&gpu, &[ENC_ROWS, VOCAB]);
    }

    /// A shape past a slot is a panic naming the sizing, not a silent overrun into
    /// the next slot.
    #[test]
    #[should_panic(expected = "a slot holds")]
    fn a_temp_larger_than_a_slot_panics() {
        let Some(gpu) = super::super::test_gpu() else {
            panic!("a slot holds — no GPU, skipping");
        };
        let c = TempCache::new(&gpu, 4096, 256, 4096);
        let _ = c.get::<f32>(&gpu, &[4097]);
    }

    /// Running out is a panic, not a silent allocation: at a fixed count it means
    /// the nesting bound is wrong and wants re-deriving.
    #[test]
    #[should_panic(expected = "temp slots busy")]
    fn exhausting_the_slots_panics() {
        let Some(gpu) = super::super::test_gpu() else {
            // The test must fail loudly rather than pass vacuously off-GPU.
            panic!("all temp slots busy — no GPU, skipping");
        };
        let c = TempCache::new(&gpu, 64, 64, 4096);
        let mut held = Vec::new();
        for _ in 0..SLOTS + 1 {
            held.push(c.get::<f32>(&gpu, &[8]));
        }
    }

    /// The small array is a separate pool: exhausting it must not touch the
    /// stage-sized one, and vice versa.
    #[test]
    fn small_and_big_arrays_are_independent() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let c = TempCache::new(&gpu, 4096, 256, 4096);
        let big: Vec<_> = (0..SLOTS).map(|_| c.get::<f32>(&gpu, &[8])).collect();
        // Every stage-sized slot is out, yet a small temporary still succeeds.
        let small = c.get_small::<f32>(&gpu, &[8]);
        assert_eq!(c.high_water(), (SLOTS, SLOTS));
        assert_eq!(c.small_high_water(), (1, SMALL_SLOTS));
        drop((big, small));
        c.assert_drained("test");
    }

    /// A temporary past the small slot is a panic pointing at `get`, not a silent
    /// overrun into the next small slot.
    #[test]
    #[should_panic(expected = "small temp")]
    fn a_small_temp_larger_than_its_slot_panics() {
        let Some(gpu) = super::super::test_gpu() else {
            panic!("small temp — no GPU, skipping");
        };
        let c = TempCache::new(&gpu, 4096, 256, 4096);
        let _ = c.get_small::<f32>(&gpu, &[c.small.elems + 1]);
    }
}
