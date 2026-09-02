// layer_norm.rs — mean-subtracting LayerNorm, weight-only (no beta).
//
//   mu     = mean(x)
//   sigma  = sqrt(var(x) + eps)
//   x_hat  = (x - mu) / sigma
//   out[i] = gamma[i] * x_hat[i]
//
// The model normalises with RMSNorm everywhere except the Gated Recurrent
// Transformer's gate path (arXiv 2608.15062), which is specified in terms of LN and
// where the mean subtraction is load-bearing: the gate is a sigmoid with its bias
// pinned at +4, and h^(r) is a convex blend whose mean can drift as depth
// accumulates. RMSNorm would pass that drift through as a systematic gate bias.
//
// beta is absent because the reference builds every one of these `bias=False`.
//
// Save format (tag 18):
//   gamma [f32 x norm_size]   (length-prefixed via write_f32_slice)

use std::{any::Any, io};

use crate::{
    nn_layer::{DynCache, NnLayer},
    optimizers::{GradVec, GradVecOps, add_grad_vec},
    saving::write_f32_slice,
};

/// PyTorch's `nn.LayerNorm` default, and what `gpu::layer_norm` uses.
const EPS: f32 = 1e-5;

pub struct LayerNormCache {
    /// x_hat = (x - mu) / sigma — reused in the backward pass.
    pub x_hat: Box<[f32]>,
    /// 1 / sigma.
    pub inv_std: f32,
    pub output: Box<[f32]>,
    pub dx: Box<[f32]>,
}

impl DynCache for LayerNormCache {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn output(&self) -> &[f32] {
        &self.output
    }
    fn input_grad(&self) -> &[f32] {
        &self.dx
    }
}

pub struct LayerNorm {
    pub gamma: Box<[f32]>,
    pub grads_gamma: GradVec,
    pub norm_size: usize,
}

impl LayerNorm {
    pub fn new(size: usize) -> Self {
        Self {
            gamma: vec![1.0; size].into(),
            grads_gamma: GradVec::zeros(size),
            norm_size: size,
        }
    }

    pub fn from_loaded(size: usize, gamma: Box<[f32]>) -> Self {
        Self {
            gamma,
            grads_gamma: GradVec::zeros(size),
            norm_size: size,
        }
    }

    fn alloc(size: usize) -> LayerNormCache {
        LayerNormCache {
            x_hat: vec![0.0; size].into(),
            inv_std: 0.0,
            output: vec![0.0; size].into(),
            dx: vec![0.0; size].into(),
        }
    }

    pub fn add_grads(&mut self, other: &mut Self) {
        add_grad_vec(&mut self.grads_gamma, &mut other.grads_gamma);
    }

    pub fn copy_weights(&mut self, other: &Self) {
        self.gamma.copy_from_slice(&other.gamma);
    }
}

impl NnLayer for LayerNorm {
    fn forward(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        let c = cache.as_any_mut().downcast_mut::<LayerNormCache>().unwrap();
        let n = self.norm_size;
        let mean = input.iter().sum::<f32>() / n as f32;
        let var = input.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        c.inv_std = 1.0 / (var + EPS).sqrt();
        for i in 0..n {
            c.x_hat[i] = (input[i] - mean) * c.inv_std;
            c.output[i] = self.gamma[i] * c.x_hat[i];
        }
    }

    /// `delta` = dL/d(output); leaves dL/d(input) in `delta` (and in `cache.dx`).
    ///
    ///   u   = gamma * delta
    ///   dx  = inv_std * (u - mean(u) - x_hat * mean(u * x_hat))
    fn backward(&mut self, delta: &mut [f32], cache: &mut dyn DynCache) {
        let c = cache.as_any_mut().downcast_mut::<LayerNormCache>().unwrap();
        let n = self.norm_size;
        let (mut su, mut sux) = (0.0, 0.0);
        for i in 0..n {
            let u = self.gamma[i] * delta[i];
            self.grads_gamma.vec()[i] += delta[i] * c.x_hat[i];
            su += u;
            sux += u * c.x_hat[i];
        }
        let (mean_u, mean_ux) = (su / n as f32, sux / n as f32);
        for i in 0..n {
            let u = self.gamma[i] * delta[i];
            delta[i] = c.inv_std * (u - mean_u - c.x_hat[i] * mean_ux);
            c.dx[i] = delta[i];
        }
    }

    fn layer_tag(&self) -> u8 {
        18
    }

    fn save(&self, w: &mut dyn io::Write) -> io::Result<()> {
        write_f32_slice(w, &self.gamma)
    }

    fn make_cache(&self) -> Box<dyn DynCache> {
        Box::new(Self::alloc(self.norm_size))
    }

    fn input_size(&self) -> usize {
        self.norm_size
    }
    fn output_size(&self) -> usize {
        self.norm_size
    }

    fn apply_grads(&mut self, lr: f32, _weight_decay: f32) {
        // Norm scale: never weight-decayed.
        self.grads_gamma.apply_to(&mut self.gamma, lr);
    }
    fn clear_grads(&mut self) {
        self.grads_gamma.clear();
    }

    fn add_grads_from(&mut self, other: &mut dyn NnLayer) {
        let o = other
            .as_any_mut()
            .downcast_mut::<Self>()
            .expect("LayerNorm::add_grads_from — replica layer type mismatch");
        self.add_grads(o);
    }

    fn copy_weights_from(&mut self, other: &dyn NnLayer) {
        let o = other
            .as_any()
            .downcast_ref::<Self>()
            .expect("LayerNorm::copy_weights_from — replica layer type mismatch");
        self.copy_weights(o);
    }
}
