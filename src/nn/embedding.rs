use std::any::Any;

use iron_oxide::collections::Matrix;

use crate::{
    nn_layer::{DynCache, NnLayer},
    optimizers::{GradMatrixNoDecay, GradMatrixOps},
};

pub struct EmbeddingCache {
    /// Token index of the one-hot input — used in backward to update one row.
    pub token_idx: usize,
    /// Post-activation output; activation derivative is applied in-place during backward.
    pub output: Vec<f32>,
    /// dL/d(input), kept for interface compatibility (embedding is always the first layer).
    pub dx: Vec<f32>,
}

impl EmbeddingCache {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            token_idx: 0,
            output: vec![0.0; output_size],
            dx: vec![0.0; input_size],
        }
    }
}

impl DynCache for EmbeddingCache {
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

pub struct EmbeddingGrads {
    pub weights: GradMatrixNoDecay,
}

impl EmbeddingGrads {
    pub fn zeros(input_size: usize, output_size: usize) -> Self {
        Self {
            weights: GradMatrixNoDecay::zeros(input_size, output_size),
        }
    }
}

pub struct EmbeddingLayer {
    pub weights: Matrix,
    /// Gradient accumulators — cleared per batch, applied in `sgd_step`.
    pub grads: EmbeddingGrads,
}

impl EmbeddingLayer {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let scale = (6.0 / (input_size as f32 + hidden_size as f32)).sqrt();
        let weights = Matrix::random(input_size, hidden_size, scale);
        Self {
            weights,
            grads: EmbeddingGrads::zeros(input_size, hidden_size),
        }
    }

    pub fn from_loaded(input_size: usize, output_size: usize, weights: Matrix) -> Self {
        debug_assert_eq!(weights.rows(), input_size);
        debug_assert_eq!(weights.cols(), output_size);

        Self {
            weights,
            grads: EmbeddingGrads::zeros(input_size, output_size),
        }
    }

    /// Forward: one-hot input → row lookup, O(hidden) instead of O(vocab × hidden).
    pub fn forward(&self, input: &[f32], cache: &mut EmbeddingCache) {
        let tok = input.iter().position(|&x| x != 0.0).unwrap_or(0);
        cache.token_idx = tok;
        cache.output.copy_from_slice(&self.weights[tok]);
    }

    /// Backward: only the one active row of the grad matrix needs updating.
    pub fn backward(&mut self, delta: &mut [f32], cache: &mut EmbeddingCache) {
        let row = &mut self.grads.weights.matrix()[cache.token_idx];
        for (g, &d) in row.iter_mut().zip(delta.iter()) {
            *g += d;
        }
        cache.dx.fill(0.0);
    }

    pub fn input_size(&self) -> usize {
        self.weights.rows()
    }
    pub fn output_size(&self) -> usize {
        self.weights.cols()
    }
}

impl NnLayer for EmbeddingLayer {
    fn forward(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<EmbeddingCache>()
            .expect("DenseLayer::forward — expected DenseCache");
        EmbeddingLayer::forward(self, input, c);
    }

    fn backward(&mut self, delta: &mut [f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<EmbeddingCache>()
            .expect("DenseLayer::backward — expected DenseCache");
        EmbeddingLayer::backward(self, delta, c);
    }

    fn layer_tag(&self) -> u8 {
        7
    }

    fn save(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        crate::saving::write_matrix(w, &self.weights)
    }

    fn make_cache(&self) -> Box<dyn DynCache> {
        Box::new(EmbeddingCache::new(self.input_size(), self.output_size()))
    }

    fn input_size(&self) -> usize {
        self.weights.rows()
    }
    fn output_size(&self) -> usize {
        self.weights.cols()
    }

    fn apply_grads(&mut self, lr: f32) {
        self.grads.weights.apply_to(&mut self.weights, lr);
    }

    fn clear_grads(&mut self) {
        self.grads.weights.clear();
    }
}
