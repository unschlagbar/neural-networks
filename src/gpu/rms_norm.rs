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
        assert!(size.is_multiple_of(group), "RmsNorm: size {size} not divisible by group {group}");
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

    /// `y = γ ⊙ (x / rms(x))`, row-wise. `x` is `[B, F]`, result `[B, F]`. Saves
    /// `x̂`/`inv_rms` for backward.
    pub fn forward(&mut self, gpu: &Gpu, x: &DTensor) -> DTensor {
        assert_eq!(x.cols(), self.size, "RmsNorm::forward — width mismatch");
        let (out, saved) = ops::rms_norm_forward(gpu, x, &self.gamma, self.group, EPS);
        self.fwd = Some(saved);
        out
    }

    /// Given `dY` `[B, F]`, accumulate `dγ` and return `dX` `[B, F]`.
    pub fn backward(&mut self, gpu: &Gpu, dy: &DTensor) -> DTensor {
        assert_eq!(dy.cols(), self.size, "RmsNorm::backward — width mismatch");
        let fwd = self.fwd.as_ref().expect("RmsNorm::backward before forward");
        ops::rms_norm_backward(gpu, dy, fwd, &self.gamma, &mut self.dgamma, self.group)
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
        ops::adamw(gpu, &mut self.gamma, &self.dgamma, &mut self.m, &mut self.v, cfg, false);
        self.zero_grad(gpu);
    }
}
