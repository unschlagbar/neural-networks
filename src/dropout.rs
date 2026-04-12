// ── DropoutLayer ──────────────────────────────────────────────────────────────
//
// Inverted dropout: during training each unit is kept with probability (1 − rate)
// and scaled by 1/(1 − rate), so the expected value stays the same.
// At inference time (`training == false`) the layer is a pure pass-through.
//
// Usage via SequentialBuilder:
//
//   SequentialBuilder::new(input_size)
//       .lstm(256)
//       .dropout(0.3)          // 30 % dropped during training
//       .dense(vocab, Linear)
//       .softmax()
//       .build()
//
// Call `set_training(false)` before sampling / evaluation.

use std::{any::Any, io};

use rand::{RngExt, rng};

use crate::{
    nn_layer::{DynCache, NnLayer},
    saving::write_f32,
};

// ── DropoutCache ──────────────────────────────────────────────────────────────

pub struct DropoutCache {
    /// Boolean mask stored as f32 (0.0 or scale) — avoids a second Vec.
    /// `mask[i] = 1/(1-rate)` if the unit is kept, `0.0` if dropped.
    pub mask: Vec<f32>,
    /// Post-mask output fed to the next layer.
    pub output: Vec<f32>,
    /// dL/d(input) = delta ⊙ mask — written during backward.
    pub dx: Vec<f32>,
}

impl DropoutCache {
    pub fn new(size: usize) -> Self {
        Self {
            mask: vec![0.0; size],
            output: vec![0.0; size],
            dx: vec![0.0; size],
        }
    }
}

impl DynCache for DropoutCache {
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

// ── DropoutLayer ──────────────────────────────────────────────────────────────

pub struct DropoutLayer {
    size: usize,
    /// Fraction of units to zero out (0.0 = no dropout, 0.5 = half dropped).
    rate: f32,
}

impl DropoutLayer {
    pub fn new(size: usize, rate: f32) -> Self {
        assert!((0.0..1.0).contains(&rate), "Dropout rate must be in [0, 1)");
        Self { size, rate }
    }

    fn forward_impl(&self, input: &[f32], cache: &mut DropoutCache) {
        if self.rate == 0.0 {
            // Inference path: identity.
            cache.output.copy_from_slice(input);
            cache.mask.fill(1.0);
            return;
        }

        let scale = 1.0 / (1.0 - self.rate);
        let mut rng = rng();

        for i in 0..self.size {
            let keep = rng.random::<f32>() >= self.rate;
            cache.mask[i] = if keep { scale } else { 0.0 };
            cache.output[i] = input[i] * cache.mask[i];
        }
    }

    fn backward_impl(&self, delta: &[f32], cache: &mut DropoutCache) {
        // dL/dx_i = delta_i * mask_i  (mask is already 0 or scale)
        for i in 0..self.size {
            cache.dx[i] = delta[i] * cache.mask[i];
        }
    }
}

// ── impl NnLayer for DropoutLayer ─────────────────────────────────────────────

impl NnLayer for DropoutLayer {
    fn forward(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<DropoutCache>()
            .expect("DropoutLayer::forward — expected DropoutCache");
        self.forward_impl(input, c);
    }

    fn forward_sample(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<DropoutCache>()
            .expect("DropoutLayer::forward — expected DropoutCache");
        c.output.copy_from_slice(input);
        c.mask.fill(1.0);
    }

    fn backward(&mut self, delta: &mut [f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<DropoutCache>()
            .expect("DropoutLayer::backward — expected DropoutCache");
        self.backward_impl(delta, c);
    }

    fn layer_tag(&self) -> u8 {
        6
    } // TAG_DROPOUT

    fn save(&self, w: &mut dyn io::Write) -> io::Result<()> {
        write_f32(w, self.rate)?;
        Ok(())
    }

    fn make_cache(&self) -> Box<dyn DynCache> {
        Box::new(DropoutCache::new(self.size))
    }

    fn input_size(&self) -> usize {
        self.size
    }
    fn output_size(&self) -> usize {
        self.size
    }

    // No learnable parameters — all gradient ops are no-ops.
    fn apply_grads(&mut self, _lr: f32) {}
    fn clear_grads(&mut self) {}
    fn scale_grads(&mut self, _scale: f32) {}
    fn clip_grads(&mut self) {}
}
