//! One contiguous device allocation holding every parameter of a model.
//!
//! A layer built on its own allocates a buffer per tensor: weight, gradient and the
//! two AdamW moments, times ~900 tensors across the hierarchical model. That costs
//! twice over — the step becomes hundreds of tiny launches and memsets, and every
//! parameter sits at whatever address the allocator handed out, which is not stable
//! across runs and rules out capturing the step in a CUDA graph.
//!
//! [`ParamArena`] takes the tensors a model hands it ([`ParamSlot`]) and re-points
//! each one at a window of four big allocations — params, grads, moments `m` and `v`
//! — laid out identically. The whole step is then one AdamW launch over one range
//! plus one memset over the gradients, and every parameter has a fixed address for
//! the life of the model.
//!
//! Layout: slots are packed by [`ParamKind`], decayed first, so the decay term and
//! the extent of the update are both bound checks on the element index rather than a
//! per-tensor lookup.

use cudarc::driver::{CudaSlice, PushKernelArg};

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

/// Step each slot with its own launch, clearing the gradients.
///
/// The fallback for a layer used on its own — the parity tests and the small
/// bring-up stacks. A whole model steps its [`ParamArena`] instead.
pub fn step_slots(gpu: &Gpu, slots: &mut [ParamSlot<'_>], cfg: &AdamCfg) {
    for s in slots {
        if s.kind != ParamKind::Frozen {
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

/// Every parameter of a model, packed into four parallel allocations.
pub struct ParamArena {
    param: CudaSlice<f32>,
    grad: CudaSlice<f32>,
    m: CudaSlice<f32>,
    v: CudaSlice<f32>,
    /// Elements before this are weight-decayed.
    decay_end: usize,
    /// Elements before this are stepped; the rest are frozen.
    step_end: usize,
}

impl ParamArena {
    /// Move every slot's four tensors into the arena, leaving each layer holding
    /// windows into it. Contents are preserved, so this may run on a model whose
    /// weights are already loaded.
    pub fn bind(gpu: &Gpu, mut slots: Vec<ParamSlot<'_>>) -> Self {
        for s in slots.iter() {
            let n = s.param.len();
            assert_eq!(n, s.grad.len(), "gradient length != parameter length");
            assert_eq!(n, s.m.len(), "moment length != parameter length");
            assert_eq!(n, s.v.len(), "moment length != parameter length");
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

        // One buffer at a time, and within it one tensor at a time: each old
        // allocation goes back to the driver as soon as it is copied, so the transient
        // peak is the model plus a quarter of the arena rather than twice the model.
        let [param, grad, m, v] = std::array::from_fn(|role| pack(gpu, &mut slots, total, role));
        Self {
            param,
            grad,
            m,
            v,
            decay_end,
            step_end,
        }
    }

    /// One AdamW step over every parameter, then clear every gradient.
    pub fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        let bc1 = 1.0 - cfg.beta1.powi(cfg.t as i32);
        let bc2 = 1.0 - cfg.beta2.powi(cfg.t as i32);
        let (lr, b1, b2, eps, wd) = (cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay);
        let (decay_end, n) = (self.decay_end as i32, self.step_end as i32);
        let f = gpu.kernels.get("adamw_arena");
        let mut lb = gpu.stream.launch_builder(&f);
        lb.arg(&mut self.param)
            .arg(&self.grad)
            .arg(&mut self.m)
            .arg(&mut self.v)
            .arg(&lr)
            .arg(&b1)
            .arg(&b2)
            .arg(&eps)
            .arg(&wd)
            .arg(&bc1)
            .arg(&bc2)
            .arg(&decay_end)
            .arg(&n);
        unsafe { lb.launch(super::ops::elem_cfg(gpu, self.step_end as u32)) }
            .expect("adamw_arena");
        self.zero_grad(gpu);
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
    // SAFETY: every element is written by the copies below before any read.
    let mut base = unsafe { gpu.stream.alloc::<f32>(total) }.expect("param arena alloc");
    let mut off = 0;
    for s in slots.iter_mut() {
        let t = s.role_mut(role);
        let n = t.len();
        gpu.stream
            .memcpy_dtod(&t.buf.slice(..n), &mut base.slice_mut(off..off + n))
            .expect("arena copy");
        let view = GTensor::view(gpu, &base, off, t.dims());
        *t = view;
        off += n;
    }
    base
}

#[derive(Default)]
pub struct TrainingCache {}

impl TrainingCache {
    pub fn new() -> Self {
        Self {}
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

        step_slots(&gpu, &mut slots(&mut eager, &kinds), &cfg);
        let mut arena = ParamArena::bind(&gpu, slots(&mut packed, &kinds));
        arena.step(&gpu, &cfg);

        let close = |a: &[f32], b: &[f32], what: &str| {
            assert_eq!(a.len(), b.len(), "{what}: length mismatch");
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                assert!((x - y).abs() < 1e-3, "{what}[{i}]: eager {x} vs arena {y}");
            }
        };
        for (i, (e, a)) in eager.iter().zip(&packed).enumerate() {
            for (j, what) in ["param", "grad", "m", "v"].iter().enumerate() {
                close(
                    &e[j].to_host(&gpu).data,
                    &a[j].to_host(&gpu).data,
                    &format!("tensor {i} {what}"),
                );
            }
            let g = a[1].to_host(&gpu).data;
            assert!(g.iter().all(|&x| x == 0.0), "tensor {i}: grad not cleared");
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
}
