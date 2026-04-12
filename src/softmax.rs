// ── SoftmaxLayer ──────────────────────────────────────────────────────────────
//
// Why a separate layer instead of `DenseLayer(Softmax)`?
//
//   PyTorch / Keras do the same: the output Dense has no activation, and a
//   separate Softmax (or fused CrossEntropyLoss) sits on top. This removes the
//   `is_softmax()` flag hack entirely — backward here receives the fused
//   cross-entropy gradient ŷ−y directly from `Sequential`, which is exactly
//   the Jacobian of Softmax·CE collapsed. No further transformation needed,
//   so dx = delta and there are no learnable parameters.

use std::{any::Any, io};

use crate::{
    nn_layer::{DynCache, NnLayer},
    saving::write_u32,
};

// ── SoftmaxCache ──────────────────────────────────────────────────────────────

pub struct SoftmaxCache {
    pub output: Vec<f32>,
    pub dx: Vec<f32>,
}

impl SoftmaxCache {
    pub fn new(size: usize) -> Self {
        Self {
            output: vec![0.0; size],
            dx: vec![0.0; size],
        }
    }
}

impl DynCache for SoftmaxCache {
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

// ── helpers ───────────────────────────────────────────────────────────────────

pub fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().fold(f32::NEG_INFINITY, |x, &y| x.max(y));
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    x.iter_mut().for_each(|v| *v /= sum);
}

/// Returns a heap-allocated softmax result — used for sampling temperature scaling.
pub fn softmax(vec: &[f32]) -> Box<[f32]> {
    let max = vec.iter().fold(f32::NEG_INFINITY, |x, &y| x.max(y));
    let mut sum = 0.0;
    let mut out: Box<[f32]> = vec
        .iter()
        .map(|&x| {
            let e = (x - max).exp();
            sum += e;
            e
        })
        .collect();
    out.iter_mut().for_each(|x| *x /= sum);
    out
}

pub struct SoftmaxLayer {
    size: usize,
}

impl SoftmaxLayer {
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl NnLayer for SoftmaxLayer {
    fn forward(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        let c = cache.as_any_mut().downcast_mut::<SoftmaxCache>().unwrap();
        c.output.copy_from_slice(input);
        softmax_inplace(&mut c.output);
    }

    fn forward_sample(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        let c = cache.as_any_mut().downcast_mut::<SoftmaxCache>().unwrap();
        c.output.copy_from_slice(input);
        //softmax_inplace(&mut c.output);
    }

    fn backward(&mut self, delta: &mut [f32], cache: &mut dyn DynCache) {
        let c = cache.as_any_mut().downcast_mut::<SoftmaxCache>().unwrap();
        // delta ist bereits ŷ−y — einfach durchleiten
        c.dx.copy_from_slice(delta);
    }

    fn layer_tag(&self) -> u8 {
        4
    } // TAG_SOFTMAX

    /// Keine Gewichte — size steht bereits im Architektur-Header.
    fn save(&self, w: &mut dyn io::Write) -> io::Result<()> {
        write_u32(w, self.size as u32)
    }

    fn make_cache(&self) -> Box<dyn DynCache> {
        Box::new(SoftmaxCache::new(self.size))
    }
    fn input_size(&self) -> usize {
        self.size
    }
    fn output_size(&self) -> usize {
        self.size
    }

    // No parameters → all grad ops are no-ops.
    fn apply_grads(&mut self, _lr: f32) {}
    fn clear_grads(&mut self) {}
    fn scale_grads(&mut self, _scale: f32) {}
    fn clip_grads(&mut self) {}
}
