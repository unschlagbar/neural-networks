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
    lstm::{CLIP, sub_vec_in_place},
    nn_layer::{DynCache, NnLayer},
    saving::{write_f32_slice, write_u8, write_u32},
};

// Larger than f32::EPSILON to avoid denormal blow-up on near-zero vectors at init.
const EPS: f32 = 1e-6;

// ── Cache ─────────────────────────────────────────────────────────────────────

pub struct LayerNormWrapperCache {
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

impl DynCache for LayerNormWrapperCache {
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

// ── Wrapper ───────────────────────────────────────────────────────────────────

pub struct LayerNormWrapper {
    pub inner: Box<dyn NnLayer>,
    /// Learnable per-element scale (init 1 → wrapper starts as near-identity).
    pub gamma: Box<[f32]>,
    pub grads_gamma: Box<[f32]>,
    pub norm_size: usize,
}

impl LayerNormWrapper {
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
            gamma: vec![1.0; n].into_boxed_slice(),
            grads_gamma: vec![0.0; n].into_boxed_slice(),
            norm_size: n,
        }
    }

    // ── RMSNorm forward kernel ────────────────────────────────────────────────
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

// ── NnLayer impl ─────────────────────────────────────────────────────────────

impl NnLayer for LayerNormWrapper {
    // ── forward ───────────────────────────────────────────────────────────────

    fn forward(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<LayerNormWrapperCache>()
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

        // Residual: output = inner_out + x
        let inner_out = c.inner_cache.output();
        for i in 0..self.norm_size {
            c.output[i] = inner_out[i] + input[i];
        }
    }

    fn forward_sample(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        self.forward(input, cache);
    }

    // ── backward ──────────────────────────────────────────────────────────────
    //
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
            .downcast_mut::<LayerNormWrapperCache>()
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
        let mut s = 0.0_f32;
        for i in 0..n {
            self.grads_gamma[i] += d_normed[i] * c.x_hat[i];
            s += self.gamma[i] * d_normed[i] * c.x_hat[i];
        }
        let s_over_n = s / n as f32;

        // Step 5: dx[i] (already = δ[i]) += dx_rms[i]
        for i in 0..n {
            let dx_rms = inv_rms * (self.gamma[i] * d_normed[i] - c.x_hat[i] * s_over_n);
            c.dx[i] += dx_rms;
        }
    }

    // ── bookkeeping ───────────────────────────────────────────────────────────

    fn layer_tag(&self) -> u8 {
        8
    } // TAG_NORM_WRAPPER (unchanged)

    fn save(&self, w: &mut dyn io::Write) -> io::Result<()> {
        write_f32_slice(w, &self.gamma)?; // no beta — RMSNorm only
        write_u8(w, self.inner.layer_tag())?;
        write_u32(w, self.inner.input_size() as u32)?;
        write_u32(w, self.inner.output_size() as u32)?;
        self.inner.save(w)
    }

    fn make_cache(&self) -> Box<dyn DynCache> {
        let n = self.norm_size;
        Box::new(LayerNormWrapperCache {
            inner_cache: self.inner.make_cache(),
            input: vec![0.0; n].into_boxed_slice(),
            x_hat: vec![0.0; n].into_boxed_slice(),
            inv_rms: 0.0,
            normed: vec![0.0; n].into_boxed_slice(),
            output: vec![0.0; n].into_boxed_slice(),
            dx: vec![0.0; n].into_boxed_slice(),
        })
    }

    fn input_size(&self) -> usize {
        self.inner.input_size()
    }
    fn output_size(&self) -> usize {
        self.inner.output_size()
    }

    fn apply_grads(&mut self, lr: f32) {
        sub_vec_in_place(&mut self.gamma, &self.grads_gamma, lr);
        self.inner.apply_grads(lr);
    }
    fn clear_grads(&mut self) {
        self.grads_gamma.fill(0.0);
        self.inner.clear_grads();
    }
    fn scale_grads(&mut self, scale: f32) {
        self.grads_gamma.iter_mut().for_each(|x| *x *= scale);
        self.inner.scale_grads(scale);
    }
    fn clip_grads(&mut self) {
        self.grads_gamma
            .iter_mut()
            .for_each(|x| *x = x.clamp(-CLIP, CLIP));
        self.inner.clip_grads();
    }

    fn reset_state(&mut self) {
        self.inner.reset_state();
    }
    fn bptt_hidden_grad(&mut self) -> Option<&[f32]> {
        self.inner.bptt_hidden_grad()
    }
    fn accumulate_init_grad(&mut self) {
        self.inner.accumulate_init_grad()
    }
}
