// norm.rs — RMSNorm-Wrapper with pre-norm residual connection
//
// Architecture (per timestep):
//
//   x ──┬── RMSNorm ──► gamma·x̂ ──► inner layer ──► y ──┬──► output
//       └──────────────────────────────────────────────┘  (residual: y + x)
//
// Why RMSNorm instead of LayerNorm:
//   • No mean subtraction  → one fewer pass over the vector
//   • No beta parameter    → fewer parameters, slightly less overfitting risk
//   • Empirically matches LayerNorm quality on LM tasks (Zhang & Sennrich 2019)
//
// Residual constraint: inner.input_size() == inner.output_size() (asserted in new()).
// The residual stabilises deep recurrent stacks and provides a free gradient highway.
//
// Save format (tag 8):
//   gamma        [f32 × norm_size]   (length-prefixed via write_f32_slice)
//   inner_tag    u8
//   inner_input  u32
//   inner_output u32
//   inner data   <layer-specific>
//
// NOTE: beta is gone — old checkpoints using LayerNorm (with beta) are incompatible.

use std::{any::Any, io};

use crate::{
    nn_layer::{DynCache, NnLayer},
    optimizers::{GradVec, GradVecOps, add_grad_vec},
    saving::write_f32_slice,
};

// Larger than f32::EPSILON to avoid denormal blow-up on near-zero vectors at init.
const EPS: f32 = 1e-6;

pub struct RMSNormCache {
    /// Saved raw input x (residual add + backward).
    pub input: Box<[f32]>,
    /// x̂ = x / rms — reused in the backward pass.
    pub x_hat: Box<[f32]>,
    /// 1 / rms (scalar) — stored to avoid re-sqrt in backward.
    pub inv_rms: f32,
    /// Final output = inner(normed) + x — seen by the next layer.
    pub output: Box<[f32]>,
    /// dL/d(input) of the whole wrapper, written by backward.
    pub dx: Box<[f32]>,
}

impl DynCache for RMSNormCache {
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

pub struct RMSNorm {
    /// Learnable per-element scale (init 1 → wrapper starts as near-identity).
    pub gamma: Box<[f32]>,
    pub grads_gamma: GradVec,
    pub norm_size: usize,
}

impl RMSNorm {
    pub fn new(size: usize) -> Self {
        let n = size;

        Self {
            gamma: vec![1.0; n].into(),
            grads_gamma: GradVec::zeros(n),
            norm_size: n,
        }
    }

    //   rms    = sqrt( mean(x²) + ε )
    //   x̂[i]  = x[i] / rms
    //   out[i] = gamma[i] · x̂[i]

    #[inline]
    fn rms_norm(
        input: &[f32],
        gamma: &[f32],
        x_hat: &mut [f32],
        normed: &mut [f32],
        inv_rms: &mut f32,
        n: usize,
    ) {
        let ss: f32 = input.iter().map(|&v| v * v).sum();
        let rms = (ss / n as f32 + EPS).sqrt();
        *inv_rms = 1.0 / rms;
        for i in 0..n {
            x_hat[i] = input[i] * *inv_rms;
            normed[i] = gamma[i] * x_hat[i];
        }
    }

    pub fn from_loaded(size: usize, gamma: Box<[f32]>) -> Self {
        Self {
            gamma,
            grads_gamma: GradVec::zeros(size),
            norm_size: size,
        }
    }

    /// Allokiert einen passenden Cache ohne Boxing.
    pub fn alloc_cache(&self) -> RMSNormCache {
        let n = self.norm_size;
        RMSNormCache {
            input: vec![0.0; n].into(),
            x_hat: vec![0.0; n].into(),
            inv_rms: 0.0,
            output: vec![0.0; n].into(),
            dx: vec![0.0; n].into(),
        }
    }

    /// Forward mit konkretem Cache — kein dyn-Overhead.
    ///
    ///   rms    = sqrt( mean(x²) + ε )
    ///   x̂[i]  = x[i] / rms
    ///   out[i] = gamma[i] · x̂[i]   → cache.output
    pub fn forward_into(&mut self, input: &[f32], cache: &mut RMSNormCache) {
        cache.input.copy_from_slice(input);
        Self::rms_norm(
            input,
            &self.gamma,
            &mut cache.x_hat,
            &mut cache.output,
            &mut cache.inv_rms,
            self.norm_size,
        );
    }

    /// Backward mit konkretem Cache.
    ///
    /// `delta` = dL/d(output). Schreibt dL/d(input) nach `cache.dx`.
    /// Akkumuliert `grads_gamma` wie gewohnt.
    ///
    /// Jacobian (RMSNorm ohne Bias):
    ///   S          = Σⱼ gamma[j] · delta[j] · x̂[j]
    ///   dx[i]      = inv_rms · ( gamma[i]·delta[i] − x̂[i]·S/n )
    pub fn backward_into(&mut self, delta: &[f32], cache: &mut RMSNormCache) {
        let n = self.norm_size;
        let irms = cache.inv_rms;
        let mut s = 0.0;

        let gamma_grads = self.grads_gamma.vec();
        for i in 0..n {
            gamma_grads[i] += delta[i] * cache.x_hat[i];
            s += self.gamma[i] * delta[i] * cache.x_hat[i];
        }
        let s_n = s / n as f32;
        for i in 0..n {
            cache.dx[i] = irms * (self.gamma[i] * delta[i] - cache.x_hat[i] * s_n);
        }
    }

    /// Fold a replica's gamma grads into this layer (data-parallel reduction).
    pub fn add_grads(&mut self, other: &mut Self) {
        add_grad_vec(&mut self.grads_gamma, &mut other.grads_gamma);
    }

    /// Overwrite gamma with `other`'s (in-place replica refresh).
    pub fn copy_weights(&mut self, other: &Self) {
        self.gamma.copy_from_slice(&other.gamma);
    }
}

impl NnLayer for RMSNorm {
    //type Cache = RMSNormCache;
    fn forward(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        let c = cache.as_any_mut().downcast_mut::<RMSNormCache>().unwrap();

        c.input.copy_from_slice(input);

        Self::rms_norm(
            input,
            &self.gamma,
            &mut c.x_hat,
            &mut c.output,
            &mut c.inv_rms,
            self.norm_size,
        );
    }

    fn forward_sample(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        self.forward(input, cache);
    }

    fn backward(&mut self, delta: &mut [f32], cache: &mut dyn DynCache) {
        let c = cache.as_any_mut().downcast_mut::<RMSNormCache>().unwrap();

        c.dx.copy_from_slice(delta);
        let d_normed = &mut c.dx;

        let n = self.norm_size;
        let inv_rms = c.inv_rms;

        // Steps 3+4: accumulate S and gamma gradient in one pass.
        let mut s = 0.0;
        for i in 0..n {
            self.grads_gamma.vec()[i] += d_normed[i] * c.x_hat[i];
            s += self.gamma[i] * d_normed[i] * c.x_hat[i];
        }
        let s_over_n = s / n as f32;

        for i in 0..n {
            d_normed[i] = inv_rms * (self.gamma[i] * d_normed[i] - c.x_hat[i] * s_over_n);
        }
    }

    fn layer_tag(&self) -> u8 {
        9
    }

    fn save(&self, w: &mut dyn io::Write) -> io::Result<()> {
        write_f32_slice(w, &self.gamma)
    }

    fn make_cache(&self) -> Box<dyn DynCache> {
        let n = self.norm_size;
        Box::new(RMSNormCache {
            input: vec![0.0; n].into(),
            x_hat: vec![0.0; n].into(),
            inv_rms: 0.0,
            output: vec![0.0; n].into(),
            dx: vec![0.0; n].into(),
        })
    }

    fn input_size(&self) -> usize {
        self.norm_size
    }
    fn output_size(&self) -> usize {
        self.norm_size
    }

    fn apply_grads(&mut self, lr: f32, _weight_decay: f32) {
        // Norm scale (gamma): never weight-decayed.
        self.grads_gamma.apply_to(&mut self.gamma, lr);
    }
    fn clear_grads(&mut self) {
        self.grads_gamma.clear();
    }

    fn add_grads_from(&mut self, other: &mut dyn NnLayer) {
        let o = other
            .as_any_mut()
            .downcast_mut::<Self>()
            .expect("RMSNorm::add_grads_from — replica layer type mismatch");
        self.add_grads(o);
    }

    fn copy_weights_from(&mut self, other: &dyn NnLayer) {
        let o = other
            .as_any()
            .downcast_ref::<Self>()
            .expect("RMSNorm::copy_weights_from — replica layer type mismatch");
        self.copy_weights(o);
    }
}
