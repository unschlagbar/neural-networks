//! Device-resident LayerNorm — mean-subtracting, weight-only.
//!
//! The only place the model uses a mean-subtracting norm is the Gated Recurrent
//! Transformer's gate path (`super::grt`), which is specified in terms of `LN`
//! (Ba et al.) and not RMSNorm. It is kept as its own layer rather than folded into
//! [`RmsNorm`](super::rms_norm::RmsNorm) because the two differ in the forward
//! reduction, the backward Jacobian and the saved tensor, i.e. in everything.
//!
//! Weight-only: the reference implementation runs its `LayerNorm(..., bias=False)`,
//! so there is no `beta` here either.
//!
//! **Stateless.** Unlike `RmsNorm`, the forward saves nothing inside the layer: the
//! caller owns the `[rows]` `inv_std` buffer. That is what lets the GRT backward
//! *re-run* the whole gate forward from `(h, h_pre)` instead of keeping R
//! recurrences' worth of gate intermediates alive across a chunked sweep — the layer
//! has no per-call slot that a second forward would clobber.

use super::arena::{self, ParamKind, ParamSlot, TrainingCache};
use super::{GTensor, Gpu, ops};
use crate::nn2::optim::AdamCfg;
use crate::tensor::Tensor;

/// PyTorch's `nn.LayerNorm` default, so the two normalize identically.
pub const EPS: f32 = 1e-5;

pub struct LayerNorm {
    pub gamma: GTensor<f32>,  // [F]
    pub dgamma: GTensor<f32>, // [F]
    m: GTensor<f32>,
    v: GTensor<f32>,
    size: usize,
}

impl LayerNorm {
    /// Build from a host scale `[F]` (uploaded).
    pub fn from_parts(gpu: &Gpu, gamma: &Tensor) -> Self {
        let size = gamma.len();
        Self {
            gamma: GTensor::from_host(gpu, gamma),
            dgamma: GTensor::zeros(gpu, &[size]),
            m: GTensor::zeros(gpu, &[size]),
            v: GTensor::zeros(gpu, &[size]),
            size,
        }
    }

    /// Fresh norm with `γ = 1`.
    pub fn new(gpu: &Gpu, size: usize) -> Self {
        Self::from_parts(gpu, &Tensor::new(&[size], vec![1.0; size]))
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// `y = γ ⊙ (x − μ) / σ`, row-wise. `inv_std` takes one value per row and is
    /// the only thing backward needs beyond `y` itself.
    ///
    /// `out` may alias `x`.
    pub fn forward(
        &self,
        gpu: &Gpu,
        x: &GTensor<f32>,
        out: &mut GTensor<f32>,
        inv_std: &mut GTensor<f32>,
    ) {
        let (_, f) = x.as_2d();
        assert_eq!(f, self.size, "LayerNorm::forward — width mismatch");
        ops::layer_norm_forward(gpu, x, &self.gamma, EPS, out, inv_std);
    }

    /// Given `dY`, accumulate `dγ` and write `dX`.
    ///
    /// `y` is this norm's own forward output and `inv_std` the buffer that forward
    /// filled; `x̂` is rebuilt as `y/γ`, so no `[rows, F]` intermediate is kept.
    pub fn backward(
        &mut self,
        gpu: &Gpu,
        dy: &GTensor<f32>,
        y: &GTensor<f32>,
        inv_std: &GTensor<f32>,
        dx: &mut GTensor<f32>,
        cache: &TrainingCache,
    ) {
        let (_, f) = dy.as_2d();
        assert_eq!(f, self.size, "LayerNorm::backward — width mismatch");
        ops::layer_norm_backward(
            gpu,
            dy,
            y,
            inv_std,
            &self.gamma,
            &mut self.dgamma,
            dx,
            &cache.temps,
        );
    }

    /// The norm scale with its gradient and AdamW moments. Never decayed — the
    /// reference puts every `dim < 2` parameter in the no-decay group.
    pub fn param_slots(&mut self) -> Vec<ParamSlot<'_>> {
        vec![ParamSlot::new(
            &mut self.gamma,
            &mut self.dgamma,
            &mut self.m,
            &mut self.v,
            ParamKind::NoDecay,
        )]
    }

    pub fn params_mut(&mut self) -> Vec<&mut GTensor<f32>> {
        vec![&mut self.gamma]
    }

    pub fn zero_grad(&mut self, gpu: &Gpu) {
        self.dgamma.zero_(gpu);
    }

    pub fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        arena::step_slots(gpu, &mut self.param_slots(), cfg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Forward and backward against a straightforward host reference.
    ///
    /// The backward is checked as a whole (dx AND dγ) rather than by finite
    /// differences: LayerNorm's Jacobian has two subtracted mean terms, and a
    /// kernel that dropped either one still produces a plausible-looking `dx`.
    #[test]
    fn matches_host_reference() {
        let Some(gpu) = crate::gpu::test_gpu() else {
            return;
        };
        let tc = TrainingCache::new(&gpu, 1 << 20, 1 << 16, 1 << 20);
        // A width below one warp, one that is not a multiple of the block width, and
        // a realistic backbone width.
        for (rows, f) in [(7usize, 17usize), (64, 300), (129, 512)] {
            let g_host = Tensor::random(&[f], 0.5);
            let x_host = Tensor::random(&[rows, f], 1.3);
            let dy_host = Tensor::random(&[rows, f], 0.8);

            let mut ln = LayerNorm::from_parts(&gpu, &g_host);
            let x = GTensor::from_host(&gpu, &x_host);
            let dy = GTensor::from_host(&gpu, &dy_host);
            let mut y = GTensor::uninit(&gpu, &[rows, f]);
            let mut inv = GTensor::uninit(&gpu, &[rows]);
            ln.forward(&gpu, &x, &mut y, &mut inv);
            let mut dx = GTensor::uninit(&gpu, &[rows, f]);
            ln.backward(&gpu, &dy, &y, &inv, &mut dx, &tc);

            let y_got = y.to_host(&gpu).data;
            let dx_got = dx.to_host(&gpu).data;
            let dg_got = ln.dgamma.to_host(&gpu).data;

            let mut dg_ref = vec![0.0; f];
            for r in 0..rows {
                let xr = &x_host.data[r * f..(r + 1) * f];
                let dyr = &dy_host.data[r * f..(r + 1) * f];
                let mean: f32 = xr.iter().sum::<f32>() / f as f32;
                let var: f32 = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / f as f32;
                let inv_s = 1.0 / (var + EPS).sqrt();
                let xhat: Vec<f32> = xr.iter().map(|v| (v - mean) * inv_s).collect();
                let u: Vec<f32> = (0..f).map(|i| g_host.data[i] * dyr[i]).collect();
                let mu = u.iter().sum::<f32>() / f as f32;
                let mux = (0..f).map(|i| u[i] * xhat[i]).sum::<f32>() / f as f32;
                for i in 0..f {
                    let yr = g_host.data[i] * xhat[i];
                    assert!(
                        (y_got[r * f + i] - yr).abs() <= 1e-4 * yr.abs().max(1.0),
                        "y[{r},{i}] at ({rows},{f}): {} vs {yr}",
                        y_got[r * f + i]
                    );
                    let dxr = inv_s * (u[i] - mu - xhat[i] * mux);
                    assert!(
                        (dx_got[r * f + i] - dxr).abs() <= 1e-4 * dxr.abs().max(1.0),
                        "dx[{r},{i}] at ({rows},{f}): {} vs {dxr}",
                        dx_got[r * f + i]
                    );
                    dg_ref[i] += dyr[i] * xhat[i];
                }
            }
            for i in 0..f {
                assert!(
                    (dg_got[i] - dg_ref[i]).abs() <= 1e-3 * dg_ref[i].abs().max(1.0),
                    "dgamma[{i}] at ({rows},{f}): {} vs {}",
                    dg_got[i],
                    dg_ref[i]
                );
            }
        }
    }
}
