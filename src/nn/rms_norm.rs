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
    opimizers::{GradVec, GradVecOps},
    saving::{write_f32_slice, write_u8, write_u32},
};

// Larger than f32::EPSILON to avoid denormal blow-up on near-zero vectors at init.
const EPS: f32 = 1e-6;

pub struct RMSNormResidualCache {
    pub inner_cache: Box<dyn DynCache>,
    /// Saved raw input x (residual add + backward).
    pub input: Box<[f32]>,
    /// x̂ = x / rms — reused in the backward pass.
    pub x_hat: Box<[f32]>,
    /// 1 / rms (scalar) — stored to avoid re-sqrt in backward.
    pub inv_rms: f32,
    /// gamma · x̂ — fed to the inner layer.
    pub normed: Box<[f32]>,
    /// Final output = inner(normed) + x — seen by the next layer.
    pub output: Box<[f32]>,
    /// dL/d(input) of the whole wrapper, written by backward.
    pub dx: Box<[f32]>,
}

impl DynCache for RMSNormResidualCache {
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

pub struct RMSNormResidual {
    pub inner: Box<dyn NnLayer>,
    /// Learnable per-element scale (init 1 → wrapper starts as near-identity).
    pub gamma: Box<[f32]>,
    pub grads_gamma: GradVec,
    pub norm_size: usize,
}

impl RMSNormResidual {
    pub fn new(inner: Box<dyn NnLayer>) -> Self {
        assert_eq!(
            inner.input_size(),
            inner.output_size(),
            "RMSNormWrapper: inner layer must have equal input/output size \
             for the residual x + inner(norm(x)) to be well-defined \
             (got input={} output={})",
            inner.input_size(),
            inner.output_size()
        );
        let n = inner.input_size();

        Self {
            inner,
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
}

impl NnLayer for RMSNormResidual {
    //type Cache = RMSNormResidualCache;
    fn forward(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<RMSNormResidualCache>()
            .unwrap();

        c.input.copy_from_slice(input);

        Self::rms_norm(
            input,
            &self.gamma,
            &mut c.x_hat,
            &mut c.normed,
            &mut c.inv_rms,
            self.norm_size,
        );

        self.inner.forward(&c.normed, &mut *c.inner_cache);
        let inner_out = c.inner_cache.output();

        for i in 0..self.norm_size {
            c.output[i] = inner_out[i] + input[i];
        }
    }

    fn forward_sample(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        self.forward(input, cache);
    }

    // Notation:
    //   x        = saved input
    //   rms      = sqrt( mean(x²) + ε )
    //   x̂[i]    = x[i] / rms
    //   ŷ[i]    = γ[i] · x̂[i]          (→ inner layer input)
    //   output  = inner(ŷ) + x          (residual)
    //   δ       = dL/d(output)           (incoming gradient)
    //
    // Step 1 — residual branch gives identity gradient:
    //   dL/dx[i] += δ[i]
    //
    // Step 2 — inner backward:
    //   inner.backward(δ) → d_normed = dL/dŷ  (via inner_cache.input_grad)
    //
    // Step 3 — gamma gradient:
    //   dL/dγ[i] += d_normed[i] · x̂[i]
    //
    // Step 4 — RMSNorm Jacobian–vector product:
    //   S          = Σⱼ γ[j] · d_normed[j] · x̂[j]   (scalar)
    //   dx_rms[i]  = inv_rms · ( γ[i]·d_normed[i] − x̂[i]·S/n )
    //
    // Step 5 — combine both paths:
    //   c.dx[i] = dx_rms[i] + δ[i]

    fn backward(&mut self, delta: &mut [f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<RMSNormResidualCache>()
            .unwrap();

        // Save δ into dx now (residual branch, step 1+5).
        // inner.backward() may clobber `delta` in-place, so we preserve δ here.
        c.dx.copy_from_slice(delta);
        // Step 2: inner backward; d_normed = dL/dŷ lands in inner_cache.input_grad().
        self.inner.backward(delta, &mut *c.inner_cache);
        let d_normed = c.inner_cache.input_grad();

        let n = self.norm_size;
        let inv_rms = c.inv_rms;

        // Steps 3+4: accumulate S and gamma gradient in one pass.
        let mut s = 0.0;
        for i in 0..n {
            self.grads_gamma.vec()[i] += d_normed[i] * c.x_hat[i];
            s += self.gamma[i] * d_normed[i] * c.x_hat[i];
        }
        let s_over_n = s / n as f32;

        // Step 5: dx[i] (already = δ[i]) += dx_rms[i]
        for i in 0..n {
            let dx_rms = inv_rms * (self.gamma[i] * d_normed[i] - c.x_hat[i] * s_over_n);
            c.dx[i] += dx_rms;
        }
    }

    fn layer_tag(&self) -> u8 {
        8
    }

    fn save(&self, w: &mut dyn io::Write) -> io::Result<()> {
        write_f32_slice(w, &self.gamma)?; // no beta — RMSNorm only
        write_u8(w, self.inner.layer_tag())?;
        write_u32(w, self.inner.input_size() as u32)?;
        write_u32(w, self.inner.output_size() as u32)?;
        self.inner.save(w)
    }

    fn make_cache(&self) -> Box<dyn DynCache> {
        let n = self.norm_size;
        Box::new(RMSNormResidualCache {
            inner_cache: self.inner.make_cache(),
            input: vec![0.0; n].into(),
            x_hat: vec![0.0; n].into(),
            inv_rms: 0.0,
            normed: vec![0.0; n].into(),
            output: vec![0.0; n].into(),
            dx: vec![0.0; n].into(),
        })
    }

    fn input_size(&self) -> usize {
        self.inner.input_size()
    }
    fn output_size(&self) -> usize {
        self.inner.output_size()
    }

    fn apply_grads(&mut self, lr: f32) {
        self.grads_gamma.apply_to(&mut self.gamma, lr);
        self.inner.apply_grads(lr);
    }
    fn clear_grads(&mut self) {
        self.grads_gamma.clear();
        self.inner.clear_grads();
    }

    fn reset_state(&mut self) {
        self.inner.reset_state();
    }
    fn zero_bptt_state(&mut self) {
        self.inner.zero_bptt_state();
    }
    fn accumulate_init_grad(&mut self) {
        self.inner.accumulate_init_grad()
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
        for i in 0..n {
            self.grads_gamma.vec()[i] += delta[i] * cache.x_hat[i];
            s += self.gamma[i] * delta[i] * cache.x_hat[i];
        }
        let s_n = s / n as f32;
        for i in 0..n {
            cache.dx[i] = irms * (self.gamma[i] * delta[i] - cache.x_hat[i] * s_n);
        }
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

    fn apply_grads(&mut self, lr: f32) {
        self.grads_gamma.apply_to(&mut self.gamma, lr);
    }
    fn clear_grads(&mut self) {
        self.grads_gamma.clear();
    }
}
