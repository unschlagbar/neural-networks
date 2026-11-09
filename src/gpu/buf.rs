//! Layer-owned device buffers: allocate once, reuse forever.

use super::{GTensor, Gpu, ops};

/// How much larger than the request a retained allocation may be before it is
/// dropped and replaced with a right-sized one.
const RETAIN_SLACK: usize = 4;

/// Whether an existing `capacity` should be kept for a request of `want` elements.
#[inline]
pub(super) fn fits(capacity: usize, want: usize) -> bool {
    capacity >= want && capacity <= want.saturating_mul(RETAIN_SLACK)
}

/// Round an allocation up to a **size class**, so near-identical shapes share one
/// buffer instead of each pinning its own.
#[inline]
pub(super) fn size_class(n: usize) -> usize {
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
    pub fn retained_bytes(&self) -> usize {
        self.slot.as_ref().map_or(0, |t| t.capacity() * 4)
    }

    /// The buffer, shaped `dims` — reusing the existing allocation when it
    /// already has that shape, else allocating a fresh (uninitialised) one.
    pub fn get(&mut self, gpu: &Gpu, dims: &[usize]) -> &mut GTensor<f32> {
        let n: usize = dims.iter().product();
        match &mut self.slot {
            Some(t) if fits(t.capacity(), n) => t.shrink_to(dims),
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
