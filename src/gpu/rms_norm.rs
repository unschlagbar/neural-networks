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
        assert_eq!(x.cols(), self.size, "RmsNorm::forward — width mismatch");
        let (b, f) = (x.rows(), x.cols());
        let total_groups = b * (f / self.group);
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
        assert_eq!(dy.cols(), self.size, "RmsNorm::backward — width mismatch");
        let fwd = self.fwd.as_ref().expect("RmsNorm::backward before forward");
        ops::rms_norm_backward_into(gpu, dy, fwd, &self.gamma, &mut self.dgamma, self.group, dx);
    }

    /// Every learnable tensor, in a fixed order (used by checkpoint save/load).
    pub fn params_mut(&mut self) -> Vec<&mut DTensor> {
        vec![&mut self.gamma]
    }

    pub fn zero_grad(&mut self, gpu: &Gpu) {
        self.dgamma.zero_(gpu);
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
