//! One contiguous device allocation holding every parameter of a model.
//!
//! A layer built on its own allocates a buffer per tensor: weight, gradient and the
//! two AdamW moments, times ~900 tensors across the hierarchical model. That costs
//! twice over — the step becomes hundreds of tiny launches and memsets, and every
//! parameter sits at whatever address the allocator handed out, which is not stable
//! across runs and rules out capturing the step in a CUDA graph.
//!
//! [`ParamArena`] takes the tensors a model hands it ([`ParamSlot`]) and re-points
//! each one at a window of big allocations — params, grads and the two AdamW moments
//! — laid out identically. The whole step is then one AdamW launch over one range
//! plus one memset over the gradients, and every parameter has a fixed address for
//! the life of the model.
//!
//! Layout: slots are packed by [`ParamKind`], decayed first, so the decay term and
//! the extent of the update are both bound checks on the element index rather than a
//! per-tensor lookup.
//!
//! The moments are held in 8 bits ([`Moments`]) — a byte per parameter plus one fp32
//! scale per [`QBLOCK`], against fp32's four bytes. On the hierarchical model that is
//! the largest single block of device memory in the process after the activations. The
//! reference the quantized step is measured against is [`step_slots`], which runs the
//! per-tensor fp32 AdamW.

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use super::temp::TempCache;
use super::{GTensor, Gpu, ops};
use crate::nn2::optim::AdamCfg;

/// How the optimizer treats a parameter.
///
/// The project convention: interior projection matrices decay, embeddings, logit
/// heads, biases and norm scales do not.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum ParamKind {
    Decay,
    NoDecay,
    /// Never stepped, though its gradient is still accumulated and cleared: the
    /// decoder's logit head keeps a bias at its zero init so it stays equivalent to
    /// the `LinearNoBias` it exports as.
    Frozen,
}

/// One parameter with the three tensors AdamW keeps alongside it.
pub struct ParamSlot<'a> {
    pub param: &'a mut GTensor<f32>,
    pub grad: &'a mut GTensor<f32>,
    pub m: &'a mut GTensor<f32>,
    pub v: &'a mut GTensor<f32>,
    pub kind: ParamKind,
}

impl<'a> ParamSlot<'a> {
    pub fn new(
        param: &'a mut GTensor<f32>,
        grad: &'a mut GTensor<f32>,
        m: &'a mut GTensor<f32>,
        v: &'a mut GTensor<f32>,
        kind: ParamKind,
    ) -> Self {
        Self {
            param,
            grad,
            m,
            v,
            kind,
        }
    }

    /// The slot's tensors by arena buffer: 0 parameter, 1 gradient, 2 `m`, 3 `v`.
    fn role_mut(&mut self, role: usize) -> &mut GTensor<f32> {
        match role {
            0 => self.param,
            1 => self.grad,
            2 => self.m,
            _ => self.v,
        }
    }
}

/// A tensor whose storage the arena provides: a layer's AdamW moments before it is
/// bound, and its gradient once `bind` has released it.
///
/// A layer built to be bound never allocates the fp32 form of a moment at all. At 584M
/// parameters `m` and `v` are 4.5 GB that used to be alive across the whole of
/// [`ParamArena::bind`], on top of the buffers it is building — which is where the
/// process peaks. A layer stepped on its own rather than through an arena materializes
/// them on its first step ([`step_slots`]).
pub fn unbacked(gpu: &Gpu) -> GTensor<f32> {
    GTensor::zeros(gpu, &[0])
}

/// Step each slot with its own launch, clearing the gradients.
///
/// The fallback for a layer used on its own — the parity tests and the small
/// bring-up stacks. A whole model steps its [`ParamArena`] instead.
pub fn step_slots(gpu: &Gpu, slots: &mut [ParamSlot<'_>], cfg: &AdamCfg) {
    for s in slots {
        if s.kind != ParamKind::Frozen {
            if s.m.is_empty() {
                let dims = s.param.dims().to_vec();
                *s.m = GTensor::zeros(gpu, &dims);
                *s.v = GTensor::zeros(gpu, &dims);
            }
            ops::adamw(
                gpu,
                s.param,
                s.grad,
                s.m,
                s.v,
                cfg,
                s.kind == ParamKind::Decay,
            );
        }
        s.grad.zero_(gpu);
    }
}

/// Elements sharing one quantization scale. Must match `ADAM_QBLOCK` in `common.cu`.
pub const QBLOCK: usize = 2048;
/// Threads per quantization range. Must match `ADAM_QTHREADS` in `common.cu`.
const QTHREADS: u32 = 256;

/// The 256 magnitudes an 8-bit moment code stands for, as fractions of its range's
/// absmax.
///
/// Non-uniform, after Dettmers et al. (arXiv 2110.02861): seven decades, each split
/// into twice as many levels as the decade below it. The elements that dominate the
/// update sit in the top decade and resolve there to a fraction of a percent, while a
/// value a millionth of the range absmax is still not zero — which is what `v` needs,
/// since it enters the update as `1/sqrt(v)`.
///
/// `signed` spends one bit on the sign, for `m`. `v` is non-negative by construction
/// and spends the whole byte on magnitude. Both tables are sorted ascending (the
/// kernel binary-searches them), hold exactly 0.0 and exactly 1.0, and repeat 1.0 in
/// the last slot: the construction leaves one code over and a repeat keeps the table
/// sorted without inventing a magnitude.
pub fn dynamic_map(signed: bool) -> Vec<f32> {
    const DECADES: i32 = 7;
    let mut mags = Vec::new();
    for i in 0..DECADES {
        // Midpoints of an even split of [0.1, 1] — the decade's own dynamic range —
        // scaled into decade `i`, doubling in count as the decade rises.
        let levels = 1usize << if signed { i } else { i + 1 };
        let decade = 10f32.powi(i - (DECADES - 1));
        for j in 0..levels {
            let lo = 0.1 + 0.9 * j as f32 / levels as f32;
            let hi = 0.1 + 0.9 * (j + 1) as f32 / levels as f32;
            mags.push(decade * 0.5 * (lo + hi));
        }
    }
    // The top level is the absmax itself: an element at the range maximum is then
    // carried exactly rather than 0.3% low.
    *mags.last_mut().unwrap() = 1.0;

    let mut map = Vec::with_capacity(256);
    if signed {
        map.extend(mags.iter().rev().map(|m| -m));
    }
    map.push(0.0);
    map.extend_from_slice(&mags);
    while map.len() < 256 {
        map.push(1.0);
    }
    assert_eq!(map.len(), 256, "the map must fill a byte exactly");
    map
}

/// The two AdamW moments: one byte per parameter each, plus an fp32 absmax per
/// [`QBLOCK`] elements, plus the two shared 256-entry [`dynamic_map`]s. Sized to
/// `step_end` — a frozen parameter is never stepped, so it carries no moment.
struct Moments {
    m: CudaSlice<u8>,
    v: CudaSlice<u8>,
    m_scale: CudaSlice<f32>,
    v_scale: CudaSlice<f32>,
    signed_map: CudaSlice<f32>,
    unsigned_map: CudaSlice<f32>,
}

/// Every parameter of a model, packed into parallel allocations.
pub struct ParamArena {
    param: CudaSlice<f32>,
    grad: CudaSlice<f32>,
    moments: Moments,
    /// Storage for the moment tensors, which are bytes and so have no fp32 form to be
    /// a window into. One element, shared by every slot: after `bind` a layer's
    /// `m`/`v` are shaped `[1]`, so a stray use fails a shape check instead of
    /// writing somewhere real. Nothing reads them — `params_mut` and `grads` do not,
    /// and a bound model steps the arena rather than its slots.
    #[allow(dead_code, reason = "held to keep the allocation the slots view alive")]
    dummy: CudaSlice<f32>,
    /// Elements before this are weight-decayed.
    decay_end: usize,
    /// Elements before this are stepped; the rest are frozen.
    step_end: usize,
}

impl ParamArena {
    /// Move every slot into the arena, leaving each layer holding windows into it.
    ///
    /// **Parameters** are carried across, so this may run on a model whose weights are
    /// already loaded. **Gradients and moments are not**: both come out of `bind` at
    /// their zero init, the state they are in after a step. `bind` runs from a
    /// constructor, before any backward, and not reading them is what keeps the peak
    /// down — see the packing order below.
    pub fn bind(gpu: &Gpu, mut slots: Vec<ParamSlot<'_>>) -> Self {
        for s in slots.iter() {
            let n = s.param.len();
            for t in [&s.grad, &s.m, &s.v] {
                // Unbacked tensors ([`unbacked`]) are the ones the arena is about to
                // give storage to; a materialized one must match its parameter.
                assert!(
                    t.len() == n || t.is_empty(),
                    "slot tensor length != parameter length"
                );
            }
        }
        // Stable sort: the layout is a function of the traversal order alone.
        slots.sort_by_key(|s| s.kind);
        let count = |k: ParamKind| -> usize {
            slots
                .iter()
                .filter(|s| s.kind == k)
                .map(|s| s.param.len())
                .sum()
        };
        let decay_end = count(ParamKind::Decay);
        let step_end = decay_end + count(ParamKind::NoDecay);
        let total = step_end + count(ParamKind::Frozen);
        assert!(total > 0, "ParamArena::bind on a model with no parameters");

        // Release the layers' own gradient accumulators before anything is allocated.
        // A weight and its gradient are built next to each other, so while both are
        // live neither one's freed blocks leave a pool page whole enough for the
        // driver to reuse — packing the parameters with the gradients still held cost
        // a second model's worth of reserved memory (measured: 2.2 GB of 2.2 GB
        // unreclaimable). Nothing is lost: `grad` below is allocated zeroed, which is
        // where a gradient stands at construction and after every step.
        for s in slots.iter_mut() {
            *s.grad = unbacked(gpu);
        }
        // One buffer at a time, and within it one tensor at a time: each old
        // allocation goes back to the driver as soon as it is copied, so the transient
        // peak is the model plus one buffer rather than twice the model.
        let param = pack(gpu, &mut slots, total, 0);
        super::trim_pool(gpu);
        let grad = pack(gpu, &mut slots, total, 1);
        // SAFETY: written before any read — every slot is repointed into it below.
        let dummy = unsafe { gpu.stream.alloc::<f32>(1) }.expect("arena dummy alloc");
        let signed_map = upload_map(gpu, true);
        let unsigned_map = upload_map(gpu, false);
        let (m, m_scale) = pack_q8(gpu, &mut slots, step_end, 2, &signed_map, &dummy);
        let (v, v_scale) = pack_q8(gpu, &mut slots, step_end, 3, &unsigned_map, &dummy);
        let moments = Moments {
            m,
            v,
            m_scale,
            v_scale,
            signed_map,
            unsigned_map,
        };
        Self {
            param,
            grad,
            moments,
            dummy,
            decay_end,
            step_end,
        }
    }

    /// One AdamW step over every parameter, then clear every gradient.
    pub fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        let bc1 = 1.0 - cfg.beta1.powi(cfg.t as i32);
        let bc2 = 1.0 - cfg.beta2.powi(cfg.t as i32);
        let (lr, b1, b2, eps, wd) = (cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay);
        let clip = cfg.clip;
        let (decay_end, n) = (self.decay_end as i32, self.step_end as i32);
        let Moments {
            m,
            v,
            m_scale,
            v_scale,
            signed_map,
            unsigned_map,
        } = &mut self.moments;
        let f = gpu.kernels.get("adamw_arena_q8");
        let mut lb = gpu.stream.launch_builder(&f);
        lb.arg(&mut self.param)
            .arg(&self.grad)
            .arg(m)
            .arg(v)
            .arg(m_scale)
            .arg(v_scale)
            .arg(signed_map)
            .arg(unsigned_map)
            .arg(&lr)
            .arg(&b1)
            .arg(&b2)
            .arg(&eps)
            .arg(&wd)
            .arg(&bc1)
            .arg(&bc2)
            .arg(&clip)
            .arg(&decay_end)
            .arg(&n);
        // One block per quantization range — the block reduces the range's new
        // absmax, so the launch shape is not a free choice.
        unsafe { lb.launch(qblock_cfg(self.step_end)) }.expect("adamw_arena_q8");
        self.zero_grad(gpu);
    }

    /// Device bytes the whole arena holds.
    pub fn bytes(&self) -> usize {
        (self.param.len() + self.grad.len()) * 4 + self.moment_bytes()
    }

    /// Device bytes the moments hold, for the memory report.
    pub fn moment_bytes(&self) -> usize {
        let Moments {
            m,
            v,
            m_scale,
            v_scale,
            ..
        } = &self.moments;
        m.len() + v.len() + (m_scale.len() + v_scale.len()) * 4
    }

    /// Clear every gradient, frozen parameters included.
    pub fn zero_grad(&mut self, gpu: &Gpu) {
        gpu.stream
            .memset_zeros(&mut self.grad)
            .expect("zero param arena grads");
    }
}

/// Allocate one arena buffer and move every slot's tensor for `role` into it, in
/// slot order. Each tensor is copied and then re-pointed at its window, which drops
/// the allocation it came from.
fn pack(gpu: &Gpu, slots: &mut [ParamSlot<'_>], total: usize, role: usize) -> CudaSlice<f32> {
    // Unbacked slots have nothing to copy in, so any buffer that can hold one starts
    // zeroed. Only the parameters are written in full.
    let mut base = if role == 0 {
        // SAFETY: every element is written by the copies below before any read.
        unsafe { gpu.stream.alloc::<f32>(total) }.expect("param arena alloc")
    } else {
        gpu.stream
            .alloc_zeros::<f32>(total)
            .expect("param arena alloc")
    };
    let mut off = 0;
    for s in slots.iter_mut() {
        let n = s.param.len();
        let dims = s.param.dims().to_vec();
        let t = s.role_mut(role);
        if !t.is_empty() {
            gpu.stream
                .memcpy_dtod(&t.buf.slice(..n), &mut base.slice_mut(off..off + n))
                .expect("arena copy");
        }
        *t = GTensor::view(gpu, &base, off, &dims);
        off += n;
    }
    base
}

/// Launch shape of the quantized kernels: one block per [`QBLOCK`] elements.
fn qblock_cfg(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(QBLOCK) as u32, 1, 1),
        block_dim: (QTHREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn upload_map(gpu: &Gpu, signed: bool) -> CudaSlice<f32> {
    gpu.stream
        .clone_htod(&dynamic_map(signed))
        .expect("quantization map upload")
}

/// Quantize every slot's tensor for `role` into one byte-per-element arena, in slot
/// order, and repoint the tensor at the shared dummy.
///
/// The fp32 form never exists all at once: slots are copied into a bounded staging
/// buffer, quantized a window at a time, and each source allocation goes back to the
/// driver as it is consumed. `upto` is `step_end` — the frozen tail is stepped by
/// nothing and gets no moment at all.
fn pack_q8(
    gpu: &Gpu,
    slots: &mut [ParamSlot<'_>],
    upto: usize,
    role: usize,
    map: &CudaSlice<f32>,
    dummy: &CudaSlice<f32>,
) -> (CudaSlice<u8>, CudaSlice<f32>) {
    /// Staging window, 8 MB of f32. A whole number of quantization ranges, so a
    /// window boundary is never a range boundary.
    const STAGE: usize = 1 << 21;

    // SAFETY: every element below `upto` is written by the quantize launches; the
    // partial tail of the last range is never read (the kernel bounds on `n`).
    let mut q = unsafe { gpu.stream.alloc::<u8>(upto.max(1)) }.expect("moment arena alloc");
    let mut scale = gpu
        .stream
        .alloc_zeros::<f32>(upto.div_ceil(QBLOCK).max(1))
        .expect("moment scale alloc");
    let stage_len = upto.clamp(1, STAGE);
    // SAFETY: only the `filled` prefix each launch reads is ever copied into.
    let mut stage = unsafe { gpu.stream.alloc::<f32>(stage_len) }.expect("moment staging alloc");

    let mut base = 0; // arena element offset of stage[0]
    let mut filled = 0;
    let flush = |gpu: &Gpu,
                 stage: &CudaSlice<f32>,
                 q: &mut CudaSlice<u8>,
                 scale: &mut CudaSlice<f32>,
                 base: usize,
                 filled: usize| {
        let f = gpu.kernels.get("quantize_blockwise");
        let n = filled as i32;
        let mut qw = q.slice_mut(base..base + filled);
        let mut sw = scale.slice_mut(base / QBLOCK..(base + filled).div_ceil(QBLOCK));
        let mut lb = gpu.stream.launch_builder(&f);
        lb.arg(stage).arg(&mut qw).arg(&mut sw).arg(map).arg(&n);
        unsafe { lb.launch(qblock_cfg(filled)) }.expect("quantize_blockwise");
    };

    for s in slots.iter_mut() {
        let n = s.param.len();
        let t = s.role_mut(role);
        if base + filled < upto {
            let mut done = 0;
            while done < n {
                let take = (stage_len - filled).min(n - done);
                let mut dst = stage.slice_mut(filled..filled + take);
                if t.is_empty() {
                    // Unmaterialized: the moment is at its zero init.
                    gpu.stream.memset_zeros(&mut dst)
                } else {
                    gpu.stream
                        .memcpy_dtod(&t.buf.slice(done..done + take), &mut dst)
                }
                .expect("moment staging copy");
                filled += take;
                done += take;
                if filled == stage_len {
                    flush(gpu, &stage, &mut q, &mut scale, base, filled);
                    base += filled;
                    filled = 0;
                }
            }
        }
        // Drops the tensor's own allocation, which the staged copy has consumed.
        *t = GTensor::view(gpu, dummy, 0, &[1]);
    }
    if filled > 0 {
        flush(gpu, &stage, &mut q, &mut scale, base, filled);
        base += filled;
    }
    assert_eq!(base, upto, "packed {base} moment elements, expected {upto}");
    (q, scale)
}

/// Everything a window's forward and backward borrow rather than allocate.
///
/// Threaded through every layer's `forward`/`backward` by shared reference: a caller
/// holds several temporaries while calling further down the stack, so a `&mut` here
/// would forbid the nesting that makes the slots worth having. See [`super::temp`].
pub struct TrainingCache {
    /// Scratch slots for every temporary, of every stage, in both directions.
    pub temps: TempCache,
}

impl TrainingCache {
    /// Allocate the window scratch for a model whose widest temporary is `elems`
    /// `f32`s — see [`temp::widest`](super::temp::widest).
    pub fn new(gpu: &Gpu, elems: usize, small_elems: usize, chunk_elems: usize) -> Self {
        Self {
            temps: TempCache::new(gpu, elems, small_elems, chunk_elems),
        }
    }

    /// A cache for a single stack of `rows` × `hidden`, for a one-shot call site or
    /// a test that runs a layer on its own rather than as part of a model.
    ///
    /// Sized for the widest head split (`dqk = hidden`, `heads = hidden`), so it
    /// covers any geometry that stack could have.
    pub fn for_shape(gpu: &Gpu, rows: usize, hidden: usize) -> Self {
        Self::new(
            gpu,
            super::temp::widest(rows, hidden, 1, hidden, 0),
            super::temp::widest_small(rows, hidden),
            super::temp::widest_chunk(rows, rows, 1, hidden, hidden, 1),
        )
    }

    /// Device bytes held for the whole run.
    pub fn bytes(&self) -> usize {
        self.temps.bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    /// One arena-wide AdamW must leave parameters, moments and gradients exactly
    /// where the per-tensor `adamw` leaves them, and must lay the parameters out
    /// contiguously in kind order.
    ///
    /// Sizes are deliberately uneven and the kinds mixed, so both bounds the kernel
    /// reads — the decay boundary and the frozen tail — are exercised: a kernel that
    /// decayed everything, or stepped the frozen tensor, would pass a uniform test.
    #[test]
    fn arena_step_matches_per_tensor_step() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let sizes = [1usize, 17, 256, 1000, 4096];
        let kinds = [
            ParamKind::Decay,
            ParamKind::NoDecay,
            ParamKind::Decay,
            ParamKind::Frozen,
            ParamKind::NoDecay,
        ];
        let mk = |seed: f32, n: usize| {
            let d: Vec<f32> = (0..n).map(|i| (i as f32 * 0.19 + seed).sin()).collect();
            GTensor::from_host(&gpu, &Tensor::new(&[n], d))
        };
        // The second moment is a running mean of squares, so it is non-negative by
        // construction; seeding it from `sin` would put `sqrt` on negative input.
        let mk_v = |seed: f32, n: usize| {
            let d: Vec<f32> = (0..n)
                .map(|i| ((i as f32 * 0.19 + seed).sin()).abs())
                .collect();
            GTensor::from_host(&gpu, &Tensor::new(&[n], d))
        };
        // lr·wd must be large enough for a wrong decay to exceed the comparison
        // tolerance — at the production 1e-3/0.05 the decay term is 5e-5 and an
        // arena that decayed every parameter would pass unnoticed.
        let cfg = AdamCfg {
            t: 3,
            ..AdamCfg::new(0.5, 0.5)
        };

        // Two identical sets of (param, grad, m, v).
        let build = || -> Vec<[GTensor<f32>; 4]> {
            sizes
                .iter()
                .enumerate()
                .map(|(k, &n)| {
                    let k = k as f32;
                    [
                        mk(k, n),
                        mk(k + 10.0, n),
                        mk(k + 20.0, n),
                        mk_v(k + 30.0, n),
                    ]
                })
                .collect()
        };
        let mut eager = build();
        let mut packed = build();

        fn slots<'a>(ts: &'a mut [[GTensor<f32>; 4]], kinds: &[ParamKind]) -> Vec<ParamSlot<'a>> {
            ts.iter_mut()
                .zip(kinds)
                .map(|(t, &kind)| {
                    let [p, g, m, v] = t;
                    ParamSlot::new(p, g, m, v, kind)
                })
                .collect()
        }

        // `bind` clears the gradients, so the arena is seeded through its own buffer
        // — which is how a model reaches a step anyway.
        let seed: Vec<Vec<f32>> = eager.iter().map(|t| t[1].to_host(&gpu).data).collect();
        let start: Vec<Vec<f32>> = packed.iter().map(|t| t[0].to_host(&gpu).data).collect();
        step_slots(&gpu, &mut slots(&mut eager, &kinds), &cfg);
        let mut arena = ParamArena::bind(&gpu, slots(&mut packed, &kinds));
        for (t, g) in packed.iter_mut().zip(&seed) {
            let n = g.len();
            t[1].copy_from(
                &gpu,
                &GTensor::from_host(&gpu, &Tensor::new(&[n], g.clone())),
            );
        }
        arena.step(&gpu, &cfg);

        // The moments are bytes, so only the parameters can be compared against the
        // fp32 reference, and the tolerance is the map's per-element error on one
        // step rather than an exactness bound (the same 5% of an update of `lr` that
        // `q8_bind_carries_the_moments` measures). What the test pins stays sharp
        // against it: a misplaced decay boundary moves a parameter by `lr·wd` = 0.25,
        // and the frozen tail is checked below for having not moved at all. What the moments themselves cost is measured over a run in
        // `q8_moments_track_the_fp32_reference`.
        let close = |a: &[f32], b: &[f32], what: &str| {
            assert_eq!(a.len(), b.len(), "{what}: length mismatch");
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                assert!((x - y).abs() < 0.05, "{what}[{i}]: eager {x} vs arena {y}");
            }
        };
        for (i, (e, a)) in eager.iter().zip(&packed).enumerate() {
            for (j, what) in ["param", "grad"].iter().enumerate() {
                close(
                    &e[j].to_host(&gpu).data,
                    &a[j].to_host(&gpu).data,
                    &format!("tensor {i} {what}"),
                );
            }
            let g = a[1].to_host(&gpu).data;
            assert!(g.iter().all(|&x| x == 0.0), "tensor {i}: grad not cleared");
            if kinds[i] == ParamKind::Frozen {
                // Exactly, not within a tolerance: a frozen parameter is not stepped
                // at all, so nothing about the moments enters it.
                assert_eq!(a[0].to_host(&gpu).data, start[i], "frozen tensor {i} moved");
            }
        }

        // Contiguous and in kind order, which is what makes the step one launch.
        let addr = |t: &GTensor<f32>| {
            use cudarc::driver::DevicePtr;
            t.buf.device_ptr(&gpu.stream).0
        };
        let mut order: Vec<usize> = (0..packed.len()).collect();
        order.sort_by_key(|&i| kinds[i]);
        let mut want = addr(&packed[order[0]][0]);
        for &i in &order {
            assert_eq!(addr(&packed[i][0]), want, "parameter {i} is out of place");
            want += (sizes[i] * std::mem::size_of::<f32>()) as u64;
        }
    }
    /// The quantization map has to fill a byte, stay sorted for the kernel's binary
    /// search, and carry the two values a moment arena needs exactly: 0 (a fresh or
    /// dead range) and 1 (the range absmax itself).
    ///
    /// The bound checked last is what the step tolerances downstream rest on: the
    /// coarsest gap in the table is the one at the top, so a quantized moment is never
    /// further than that half-gap from the truth in absolute terms, whatever decade it
    /// sits in.
    /// A slot that never materialized its moments must step exactly like one that
    /// allocated them at their zero init. That equivalence is the whole licence for
    /// [`unbacked`]: at production width the fp32 `m` and `v` a layer would build
    /// are 4.5 GB alive across the whole of `bind`.
    #[test]
    fn unbacked_moments_step_like_zeroed_ones() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let sizes = [1usize, 17, 4096, 5000];
        let kinds = [
            ParamKind::Decay,
            ParamKind::NoDecay,
            ParamKind::Decay,
            ParamKind::NoDecay,
        ];
        let cfg = AdamCfg {
            t: 1,
            ..AdamCfg::new(0.5, 0.5)
        };
        let build = |lazy: bool| -> Vec<[GTensor<f32>; 4]> {
            sizes
                .iter()
                .enumerate()
                .map(|(k, &n)| {
                    let d = |seed: f32| {
                        let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.19 + seed).sin()).collect();
                        GTensor::from_host(&gpu, &Tensor::new(&[n], v))
                    };
                    let moment = || {
                        if lazy {
                            unbacked(&gpu)
                        } else {
                            GTensor::zeros(&gpu, &[n])
                        }
                    };
                    [d(k as f32), d(k as f32 + 10.0), moment(), moment()]
                })
                .collect()
        };
        fn slots<'a>(ts: &'a mut [[GTensor<f32>; 4]], kinds: &[ParamKind]) -> Vec<ParamSlot<'a>> {
            ts.iter_mut()
                .zip(kinds)
                .map(|(t, &kind)| {
                    let [p, g, m, v] = t;
                    ParamSlot::new(p, g, m, v, kind)
                })
                .collect()
        }

        let (mut zeroed, mut lazy) = (build(false), build(true));
        let mut a = ParamArena::bind(&gpu, slots(&mut zeroed, &kinds));
        let mut b = ParamArena::bind(&gpu, slots(&mut lazy, &kinds));
        // `bind` clears the gradients, so both arenas are seeded through their own
        // buffers, with the same values.
        for (z, l) in zeroed.iter_mut().zip(&mut lazy) {
            let n = z[0].len();
            let g: Vec<f32> = (0..n).map(|i| (i as f32 * 0.23).sin()).collect();
            let src = GTensor::from_host(&gpu, &Tensor::new(&[n], g));
            z[1].copy_from(&gpu, &src);
            l[1].copy_from(&gpu, &src);
        }
        // Twice: the second step is the one that reads back what the first wrote, so a
        // moment left as garbage rather than zero shows up here.
        for t in 1..=2 {
            let cfg = AdamCfg { t, ..cfg };
            a.step(&gpu, &cfg);
            b.step(&gpu, &cfg);
        }
        for (i, (z, l)) in zeroed.iter().zip(&lazy).enumerate() {
            let (z, l) = (z[0].to_host(&gpu).data, l[0].to_host(&gpu).data);
            for (j, (x, y)) in z.iter().zip(&l).enumerate() {
                assert_eq!(x, y, "tensor {i}[{j}]");
            }
        }
    }

    #[test]
    fn dynamic_map_fills_the_byte() {
        for signed in [true, false] {
            let map = dynamic_map(signed);
            assert_eq!(map.len(), 256);
            for w in map.windows(2) {
                assert!(w[0] <= w[1], "map is not sorted: {} then {}", w[0], w[1]);
            }
            assert!(map.contains(&0.0), "no code for zero");
            assert_eq!(*map.last().unwrap(), 1.0, "the top code is not the absmax");
            assert_eq!(map[0], if signed { -1.0 } else { 0.0 });
            if signed {
                // Symmetric: `m` changes sign every time a gradient does, and a map
                // that resolved one direction better than the other would bias the
                // update.
                for (a, b) in map.iter().zip(map[..255].iter().rev()) {
                    assert_eq!(*a, -b, "map is not symmetric about zero");
                }
            }
            let gap = map
                .windows(2)
                .map(|w| w[1] - w[0])
                .fold(0.0, |a: f32, b| a.max(b));
            assert!(gap / 2.0 < 0.011, "worst absolute error {}", gap / 2.0);
            // Seven decades: an element a millionth of the range absmax must still not
            // quantize to zero, or `1/sqrt(v)` would be unbounded for it.
            let smallest = map.iter().cloned().find(|&x| x > 0.0).unwrap();
            assert!(smallest < 1e-6, "smallest positive code is {smallest}");
        }
    }

    /// Binding an arena must carry the moments it was handed, across the staging
    /// window the fp32 form is copied through (`pack_q8`) and across the scale-range
    /// borders inside it.
    ///
    /// With a zero gradient the step is a pure function of the state that survived
    /// bind: `Δp = -lr·(β₁m/bc₁)/(√(β₂v/bc₂)+ε)`. An arena that packed zeros, or
    /// misaligned a scale against its range, does not move the parameters where the
    /// fp32 reference moves them — and the moments here are deliberately in the top
    /// decade of their ranges, where the map's own error is bounded well below the
    /// tolerance.
    #[test]
    fn q8_bind_carries_the_moments() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        // Crosses `pack_q8`'s 2^21-element staging window inside the first tensor, so
        // the second window starts mid-slot.
        let sizes = [2_100_000usize, 1000];
        let m_at = |i: usize| (i as f32 * 0.017).sin();
        let v_at = |i: usize| 1.0 + 0.5 * (i as f32 * 0.011).cos();
        let build = || -> Vec<[GTensor<f32>; 4]> {
            sizes
                .iter()
                .map(|&n| {
                    let mk = |f: &dyn Fn(usize) -> f32| {
                        GTensor::from_host(&gpu, &Tensor::new(&[n], (0..n).map(f).collect()))
                    };
                    [mk(&|_| 0.0), mk(&|_| 0.0), mk(&m_at), mk(&v_at)]
                })
                .collect()
        };
        fn slots(ts: &mut [[GTensor<f32>; 4]]) -> Vec<ParamSlot<'_>> {
            ts.iter_mut()
                .map(|t| {
                    let [p, g, m, v] = t;
                    ParamSlot::new(p, g, m, v, ParamKind::Decay)
                })
                .collect()
        }
        // `t` large enough that both bias corrections are effectively 1, so the
        // reference below is the plain moment ratio.
        let cfg = AdamCfg {
            t: 10_000,
            eps: 1e-8,
            ..AdamCfg::new(1.0, 0.0)
        };

        let mut fp32 = build();
        let mut q8 = build();
        // The reference is the per-tensor fp32 step, which keeps its moments as they
        // were handed over.
        step_slots(&gpu, &mut slots(&mut fp32), &cfg);
        let mut b = ParamArena::bind(&gpu, slots(&mut q8));
        b.step(&gpu, &cfg);

        let mut worst = 0.0f32;
        let mut moved = 0.0f32;
        for (x, y) in fp32.iter().zip(&q8) {
            let (want, got) = (x[0].to_host(&gpu).data, y[0].to_host(&gpu).data);
            for (w, g) in want.iter().zip(&got) {
                worst = worst.max((w - g).abs());
                moved = moved.max(w.abs());
            }
        }
        assert!(moved > 0.5, "the reference step barely moved ({moved})");
        assert!(worst < 0.05, "quantized step is {worst} off the fp32 step");
    }

    /// What the 8-bit moments actually cost, over a run rather than a step: the two
    /// arenas see the same gradients and must stay together.
    ///
    /// Quantization touches only the state carried between steps — the update itself
    /// is computed in fp32 — so the error to bound is the drift it accumulates. The
    /// tolerance is stated against the displacement the run produced, not against an
    /// absolute number, because that is the quantity a training run cares about.
    #[test]
    fn q8_moments_track_the_fp32_reference() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let sizes = [4097usize, 2048, 1, 5000, 33];
        let kinds = [
            ParamKind::Decay,
            ParamKind::NoDecay,
            ParamKind::Decay,
            ParamKind::NoDecay,
            ParamKind::Frozen,
        ];
        let build = || -> Vec<[GTensor<f32>; 4]> {
            sizes
                .iter()
                .enumerate()
                .map(|(k, &n)| {
                    let p: Vec<f32> = (0..n).map(|i| ((i + k) as f32 * 0.03).sin()).collect();
                    let z = || GTensor::from_host(&gpu, &Tensor::new(&[n], vec![0.0; n]));
                    [
                        GTensor::from_host(&gpu, &Tensor::new(&[n], p)),
                        z(),
                        z(),
                        z(),
                    ]
                })
                .collect()
        };
        fn slots<'a>(ts: &'a mut [[GTensor<f32>; 4]], kinds: &[ParamKind]) -> Vec<ParamSlot<'a>> {
            ts.iter_mut()
                .zip(kinds)
                .map(|(t, &kind)| {
                    let [p, g, m, v] = t;
                    ParamSlot::new(p, g, m, v, kind)
                })
                .collect()
        }
        // A gradient that changes scale and sign per step and per tensor: a moment
        // that is only ever re-quantized against a settled absmax is the easy case.
        let grad = |step: usize, k: usize, i: usize| -> f32 {
            let phase = (i as f32 * 0.07 + k as f32 + step as f32 * 0.9).sin();
            phase * 10f32.powi((step % 4) as i32 - 2)
        };

        let start = build();
        let mut fp32 = build();
        let mut q8 = build();
        // The reference is the per-tensor fp32 step: same update, moments in full
        // width, no arena in the way.
        let mut b = ParamArena::bind(&gpu, slots(&mut q8, &kinds));
        let mut cfg = AdamCfg::new(1e-3, 0.01);
        let steps = 20;
        for step in 0..steps {
            for (k, &n) in sizes.iter().enumerate() {
                let g: Vec<f32> = (0..n).map(|i| grad(step, k, i)).collect();
                let src = GTensor::from_host(&gpu, &Tensor::new(&[n], g));
                fp32[k][1].copy_from(&gpu, &src);
                q8[k][1].copy_from(&gpu, &src);
            }
            cfg.t = step as u64 + 1;
            step_slots(&gpu, &mut slots(&mut fp32, &kinds), &cfg);
            b.step(&gpu, &cfg);
        }

        // Aggregate, not per element. An element whose moments sit far below their
        // range's absmax carries most of the map's error, and Adam's per-element
        // scale invariance turns that into a differently-directed step of the same
        // size — the known cost of blockwise quantization, and the reason the claim
        // to check is about the run, not about the worst parameter in it.
        let (mut drift, mut travel, mut n) = (0.0f64, 0.0f64, 0usize);
        for k in 0..sizes.len() {
            let s = start[k][0].to_host(&gpu).data;
            let x = fp32[k][0].to_host(&gpu).data;
            let y = q8[k][0].to_host(&gpu).data;
            for i in 0..s.len() {
                drift += ((x[i] - y[i]) as f64).powi(2);
                travel += ((x[i] - s[i]) as f64).powi(2);
                n += 1;
            }
        }
        let (drift, travel) = ((drift / n as f64).sqrt(), (travel / n as f64).sqrt());
        // Adam moves each parameter by about `lr` per step and the gradient here
        // changes sign, so the run's displacement is a walk of about `sqrt(steps)·lr`.
        assert!(
            travel > 0.1 * steps as f64 * cfg.lr as f64,
            "the reference run barely moved ({travel})"
        );
        // Measured 0.9%. Moments that did not survive quantization at all would put
        // the two runs on unrelated walks of the same length — a ratio above 1.
        assert!(
            drift < 0.05 * travel,
            "8-bit moments drifted {drift} rms against a displacement of {travel}"
        );
    }

    /// The gradient clip must bite on the device exactly where the CPU reference
    /// clips, and `clip = INFINITY` must leave the update untouched. Gradients here
    /// straddle the bound in both directions — a kernel that dropped the clip agrees
    /// with the reference only on the interior elements.
    #[test]
    fn arena_step_clips_gradients_like_the_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        use crate::nn2::optim::AdamState;

        let n = 512;
        let param: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).sin()).collect();
        // Spread well past ±clip on both sides.
        let grad: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).sin() * 40.0).collect();
        assert!(grad.iter().any(|g| *g > 5.0) && grad.iter().any(|g| *g < -5.0));

        for clip in [5.0, f32::INFINITY] {
            let cfg = AdamCfg {
                clip,
                t: 1,
                ..AdamCfg::new(0.5, 0.0)
            };
            let mut want = param.clone();
            AdamState::new().step(&mut want, &grad, &cfg, false);

            let mk = |d: &[f32]| GTensor::from_host(&gpu, &Tensor::new(&[n], d.to_vec()));
            let mut t = [mk(&param), mk(&grad), mk(&vec![0.0; n]), mk(&vec![0.0; n])];
            // The arena owns the allocation the tensors now view into, so it has to
            // outlive the download.
            let mut arena = {
                let [p, g, m, v] = &mut t;
                ParamArena::bind(&gpu, vec![ParamSlot::new(p, g, m, v, ParamKind::NoDecay)])
            };
            // `bind` clears the gradients: seed through the arena's own buffer.
            t[1].copy_from(&gpu, &mk(&grad));
            arena.step(&gpu, &cfg);
            let got = t[0].to_host(&gpu).data;
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                // The moments are quantized, so this is the map's one-step error, not
                // an exactness bound. A dropped clip moves an element by `lr` — two
                // orders more.
                assert!(
                    (a - b).abs() < 0.02,
                    "clip {clip}, element {i} (grad {}): gpu {a} vs cpu {b}",
                    grad[i]
                );
            }
        }
    }
}
