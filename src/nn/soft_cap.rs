// Logit soft-capping (xLSTM-7B / Gemma-2 style):
//
//   y = cap · tanh(x / cap)
//
// A weightless squashing layer placed after a logit head. Logits are bounded
// to (−cap, cap), so the cross-entropy incentive to grow the head weights
// without limit dies off as logits approach the cap (the tanh derivative
// vanishes there). Backward is the exact derivative — no gradient clipping
// or signal suppression involved.

use std::{any::Any, io};

use crate::{
    nn_layer::{DynCache, NnLayer},
    saving::write_f32,
};

pub struct SoftCapCache {
    /// Capped output fed to the next layer / the softmax.
    pub output: Vec<f32>,
    /// dL/d(input) = delta · (1 − (output/cap)²) — written during backward.
    pub dx: Vec<f32>,
}

impl SoftCapCache {
    pub fn new(size: usize) -> Self {
        Self {
            output: vec![0.0; size],
            dx: vec![0.0; size],
        }
    }
}

impl DynCache for SoftCapCache {
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

pub struct SoftCapLayer {
    pub size: usize,
    pub cap: f32,
}

impl SoftCapLayer {
    pub fn new(size: usize, cap: f32) -> Self {
        assert!(cap > 0.0, "SoftCap cap must be positive");
        Self { size, cap }
    }
}

impl NnLayer for SoftCapLayer {
    fn forward(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<SoftCapCache>()
            .expect("SoftCapLayer::forward — expected SoftCapCache");
        for i in 0..self.size {
            c.output[i] = self.cap * (input[i] / self.cap).tanh();
        }
    }

    fn backward(&mut self, delta: &mut [f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<SoftCapCache>()
            .expect("SoftCapLayer::backward — expected SoftCapCache");
        // d/dx (cap · tanh(x/cap)) = 1 − tanh²(x/cap) = 1 − (y/cap)²
        for i in 0..self.size {
            let t = c.output[i] / self.cap;
            c.dx[i] = delta[i] * (1.0 - t * t);
        }
    }

    fn layer_tag(&self) -> u8 {
        17
    }

    fn save(&self, w: &mut dyn io::Write) -> io::Result<()> {
        write_f32(w, self.cap)?;
        Ok(())
    }

    fn make_cache(&self) -> Box<dyn DynCache> {
        Box::new(SoftCapCache::new(self.size))
    }

    fn input_size(&self) -> usize {
        self.size
    }
    fn output_size(&self) -> usize {
        self.size
    }

    // No learnable parameters — all gradient ops are no-ops.
    fn apply_grads(&mut self, _lr: f32, _weight_decay: f32) {}
    fn clear_grads(&mut self) {}
    fn add_grads_from(&mut self, _other: &mut dyn NnLayer) {}
    fn copy_weights_from(&mut self, _other: &dyn NnLayer) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backward_matches_finite_differences() {
        let cap = 30.0;
        let mut layer = SoftCapLayer::new(4, cap);
        let input = [-45.0, -3.0, 0.5, 60.0];
        let upstream = [1.0, -2.0, 0.7, 3.0];

        let mut cache = SoftCapCache::new(4);
        layer.forward(&input, &mut cache);
        for (i, &x) in input.iter().enumerate() {
            assert!(cache.output[i].abs() < cap);
            assert!((cache.output[i] - cap * (x / cap).tanh()).abs() < 1e-5);
        }

        let mut delta = upstream;
        layer.backward(&mut delta, &mut cache);

        let eps = 1e-2;
        for i in 0..4 {
            let mut plus = SoftCapCache::new(4);
            let mut minus = SoftCapCache::new(4);
            let mut x = input;
            x[i] += eps;
            layer.forward(&x, &mut plus);
            x[i] -= 2.0 * eps;
            layer.forward(&x, &mut minus);
            let numeric: f32 = (0..4)
                .map(|j| upstream[j] * (plus.output[j] - minus.output[j]) / (2.0 * eps))
                .sum();
            assert!(
                (cache.dx[i] - numeric).abs() < 1e-3,
                "dx[{i}] = {} vs numeric {numeric}",
                cache.dx[i],
            );
        }
    }
}
