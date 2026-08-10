//! Device-resident RMSNorm, the GPU counterpart of
//! [`nn2::rms_norm::RmsNorm`](crate::nn2::rms_norm::RmsNorm).
//!
//! Pure normalization (`y = γ ⊙ x / rms(x)`, row-wise) with a learned scale that
//! trains on undecayed AdamW (norm scales are never weight-decayed). Wraps the
//! grouped `rms_norm_forward`/`backward` ops with the single-group (plain) config
//! `group == size`. Scale, grad, moments and the saved `x̂`/`inv_rms` all live on
//! the device.

use super::ops::{self, GpuRmsForward};
use super::{DTensor, Gpu};
use crate::nn2::optim::AdamCfg;
use crate::tensor::Tensor;

/// Matches `nn2::rms_norm::EPS` so the two systems normalize identically.
const EPS: f32 = 1e-6;

pub struct RmsNorm {
    pub gamma: DTensor,  // [F]
    pub dgamma: DTensor, // [F]
    m: DTensor,
    v: DTensor,
    size: usize,
    /// Normalization group width: `== size` for plain RMSNorm, `== dhv` for the
    /// head-wise variant (`F/group` independent groups per row, one γ slice each).
    group: usize,
    /// Saved `x̂` / `inv_rms`, reused across calls: both are shape-determined, so
    /// a steady batch size means the forward reallocates nothing.
    fwd: Option<GpuRmsForward>,
    /// Earlier chunks' `x̂`/`inv_rms`, oldest first. The single `fwd` slot holds one
    /// chunk's, so without this chunk c+1's forward overwrites what chunk c's backward
    /// reads. Empty unless [`set_carry`](Self::set_carry) is on.
    chunk_saved: Vec<GpuRmsForward>,
    /// Whether this norm is inside a chunked sweep. See [`set_carry`](Self::set_carry).
    carry: bool,
}

impl RmsNorm {
    /// Build from a host scale `[F]` (uploaded). Plain (single-group) norm.
    pub fn from_parts(gpu: &Gpu, gamma: &Tensor) -> Self {
        let size = gamma.len();
        Self::from_parts_grouped(gpu, gamma, size)
    }

    /// Head-wise variant: `group` is the per-head width (`dhv`); `F` must be a
    /// multiple of it. Matches `nn2` head-wise RMSNorm (γ is `[F]`, group `grp`
    /// uses `γ[grp*group ..]`).
    pub fn from_parts_grouped(gpu: &Gpu, gamma: &Tensor, group: usize) -> Self {
        let size = gamma.len();
        assert!(
            size.is_multiple_of(group),
            "RmsNorm: size {size} not divisible by group {group}"
        );
        Self {
            gamma: DTensor::from_host(gpu, gamma),
            dgamma: DTensor::zeros(gpu, &[size]),
            m: DTensor::zeros(gpu, &[size]),
            v: DTensor::zeros(gpu, &[size]),
            size,
            group,
            fwd: None,
            chunk_saved: Vec::new(),
            carry: false,
        }
    }

    /// Fresh RMSNorm with `γ = 1` (matches `nn2::RmsNorm::new`).
    pub fn new(gpu: &Gpu, size: usize) -> Self {
        Self::from_parts(gpu, &Tensor::new(&[size], vec![1.0; size]))
    }

    /// `y = γ ⊙ (x / rms(x))`, row-wise, into the caller's `out` `[B, F]`. Saves
    /// `x̂`/`inv_rms` for backward.
    ///
    /// `out` may alias `x` (the kernel reads each row before writing it), which
    /// is what lets a caller normalize a buffer in place.
    pub fn forward(&mut self, gpu: &Gpu, x: &DTensor, out: &mut DTensor) {
        // Position-wise: any rank is accepted and folded to `[N, F]` over the last
        // axis, so a caller holding `[B, T, H]` need not reshape.
        let (b, f) = x.as_2d();
        assert_eq!(f, self.size, "RmsNorm::forward — width mismatch");
        let total_groups = b * (f / self.group);
        // Chunked sweep: the previous chunk's `x̂`/`inv_rms` are still owed a backward,
        // so set them aside rather than letting the refit below reuse their buffers.
        if self.carry {
            if let Some(prev) = self.fwd.take() {
                self.chunk_saved.push(prev);
            }
        }
        // Refit the saved intermediates, reusing them whenever the shape holds.
        match &self.fwd {
            Some(s) if s.x_hat.len() == b * f && s.inv_rms.len() == total_groups => {}
            _ => {
                self.fwd = Some(GpuRmsForward {
                    x_hat: DTensor::uninit(gpu, &[b, f]),
                    inv_rms: gpu
                        .stream
                        .alloc_zeros::<f32>(total_groups)
                        .expect("alloc inv_rms"),
                })
            }
        }
        let saved = self.fwd.as_mut().expect("just filled");
        saved.x_hat.reshape_to(&[b, f]);
        ops::rms_norm_forward_into(gpu, x, &self.gamma, self.group, EPS, out, saved);
    }

    /// Forward into a freshly allocated `[B, F]` — the by-value companion to
    /// [`forward`](Self::forward), for call sites that still compose by value.
    pub fn forward_alloc(&mut self, gpu: &Gpu, x: &DTensor) -> DTensor {
        let mut out = DTensor::uninit(gpu, &[x.rows(), x.cols()]);
        self.forward(gpu, x, &mut out);
        out
    }

    /// Backward into a freshly allocated `dX` `[B, F]`.
    pub fn backward_alloc(&mut self, gpu: &Gpu, dy: &DTensor) -> DTensor {
        let mut dx = DTensor::uninit(gpu, &[dy.rows(), dy.cols()]);
        self.backward(gpu, dy, &mut dx);
        dx
    }

    /// Given `dY` `[B, F]`, accumulate `dγ` and write `dX` `[B, F]` into `dx`.
    pub fn backward(&mut self, gpu: &Gpu, dy: &DTensor, dx: &mut DTensor) {
        let (_, f) = dy.as_2d();
        assert_eq!(f, self.size, "RmsNorm::backward — width mismatch");
        let fwd = self.fwd.as_ref().expect("RmsNorm::backward before forward");
        ops::rms_norm_backward_into(gpu, dy, fwd, &self.gamma, &mut self.dgamma, self.group, dx);
        // Chunks unwind right to left, so hand the slot to the chunk on the left.
        if let Some(prev) = self.chunk_saved.pop() {
            self.fwd = Some(prev);
        }
    }

    /// Every learnable tensor, in a fixed order (used by checkpoint save/load).
    pub fn params_mut(&mut self) -> Vec<&mut DTensor> {
        vec![&mut self.gamma]
    }

    pub fn zero_grad(&mut self, gpu: &Gpu) {
        self.dgamma.zero_(gpu);
    }

    /// Device bytes held, split `(params, activations)`. Diagnostic — see
    /// [`Hierarchical::retained_report`](super::hierarchical::Hierarchical::retained_report).
    ///
    /// The params are four `[F]` vectors — negligible. The activations are the saved
    /// `x̂` `[B, F]` and `inv_rms`, which scale with the batch and are held across
    /// calls, so a norm inside a per-word stage retains a full window's worth.
    pub fn retained_bytes(&self) -> (usize, usize) {
        let params = [&self.gamma, &self.dgamma, &self.m, &self.v]
            .iter()
            .map(|t| t.capacity() * 4)
            .sum();
        let act = self
            .fwd
            .as_ref()
            .map_or(0, |s| (s.x_hat.capacity() + s.inv_rms.len()) * 4);
        (params, act)
    }

    /// Keep one saved `x̂`/`inv_rms` per chunk, for a sweep whose chunks all forward
    /// before any of them unwinds. Off means the single slot is reused per call, which
    /// is what every unchunked caller wants.
    pub fn set_carry(&mut self, carry: bool) {
        self.carry = carry;
        if !carry {
            self.chunk_saved.clear();
        }
    }

    /// Release the saved `x̂` / `inv_rms`. The next forward reallocates them.
    pub fn drop_saved_act(&mut self) {
        self.fwd = None;
        self.chunk_saved.clear();
    }

    /// AdamW step (norm scale is never decayed). Clears the grad.
    pub fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        ops::adamw(
            gpu,
            &mut self.gamma,
            &self.dgamma,
            &mut self.m,
            &mut self.v,
            cfg,
            false,
        );
        self.zero_grad(gpu);
    }
}
