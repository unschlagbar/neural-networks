#![allow(unused)]

use std::any::Any;

use iron_oxide::collections::Matrix;
use rand::{Rng, rng};

use crate::{
    activations::Activate,
    layer::{DenseCache, DenseGrads},
    lstm::{CLIP, add_vec_in_place, sub_in_place, sub_vec_in_place},
    nn_layer::{DynCache, NnLayer},
};

pub struct ProjectionGrads {
    pub weights: Matrix,
}

impl ProjectionGrads {
    pub fn zeros(input_size: usize, output_size: usize) -> Self {
        Self {
            weights: Matrix::zeros(input_size, output_size),
        }
    }
}

// ── Projection ────────────────────────────────────────────────────────────────

pub struct Projection<A: Activate> {
    pub weights: Matrix,
    pub activation: A,
    /// Gradient accumulators — cleared per batch, applied in `sgd_step`.
    pub grads: ProjectionGrads,
}

impl<A: Activate> Projection<A> {
    pub fn new(input_size: usize, output_size: usize, activation: A) -> Self {
        let weights = Matrix::random(input_size, output_size, 1.0);

        Self {
            weights,
            activation,
            grads: ProjectionGrads::zeros(input_size, output_size),
        }
    }

    pub fn from_loaded(
        input_size: usize,
        output_size: usize,
        activation: A,
        weights: Matrix,
    ) -> Self {
        debug_assert_eq!(weights.rows(), input_size);
        debug_assert_eq!(weights.cols(), output_size);

        Self {
            weights,
            activation,
            grads: ProjectionGrads::zeros(input_size, output_size),
        }
    }

    /// matmul into a pre-allocated output buffer (no bias).
    #[inline]
    fn matmul(&self, input: &[f32], out: &mut [f32]) {
        out.fill(0.0);
        for (i, &x) in input.iter().enumerate() {
            for (j, &w) in self.weights[i].iter().enumerate() {
                out[j] += x * w;
            }
        }
    }

    /// Standard forward: z = Wx → activation(z).
    pub fn forward(&self, input: &[f32], cache: &mut DenseCache) {
        cache.input.copy_from_slice(input);
        self.matmul(input, &mut cache.output);
        self.activation.activate(&mut cache.output);
    }

    /// Backward.
    pub fn backward(&mut self, delta: &mut [f32], cache: &mut DenseCache) {
        self.activation.derivative_active(&mut cache.output);
        delta
            .iter_mut()
            .zip(&cache.output)
            .for_each(|(d, o)| *d *= o);

        self.grads.weights.add_outer(&cache.input, delta);

        cache.dx.fill(0.0);
        for (i, dx) in cache.dx.iter_mut().enumerate() {
            for (&dy, &w) in delta.iter().zip(&self.weights[i]) {
                *dx += dy * w;
            }
        }
    }

    pub fn input_size(&self) -> usize {
        self.weights.rows()
    }
    pub fn output_size(&self) -> usize {
        self.weights.cols()
    }
}

// ── impl NnLayer for Projection ───────────────────────────────────────────────

impl<A: Activate> NnLayer for Projection<A> {
    fn forward(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<DenseCache>()
            .expect("Projection::forward — expected DenseCache");
        Projection::forward(self, input, c);
    }

    fn backward(&mut self, delta: &mut [f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<DenseCache>()
            .expect("Projection::backward — expected DenseCache");
        Projection::backward(self, delta, c);
    }

    fn layer_tag(&self) -> u8 {
        3
    }

    /// Schreibt NUR die Gewichtsmatrix.
    fn save(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        crate::saving::write_u8(w, self.activation.activation_id())?;
        crate::saving::write_matrix(w, &self.weights)?;
        Ok(())
    }

    fn activation_id(&self) -> Option<u8> {
        Some(self.activation.activation_id())
    }

    fn make_cache(&self) -> Box<dyn DynCache> {
        Box::new(DenseCache::new(self.input_size(), self.output_size()))
    }

    fn input_size(&self) -> usize {
        self.weights.rows()
    }
    fn output_size(&self) -> usize {
        self.weights.cols()
    }

    fn apply_grads(&mut self, lr: f32) {
        sub_in_place(&mut self.weights, &self.grads.weights, lr);
    }

    fn clear_grads(&mut self) {
        self.grads.weights.clear();
    }

    fn scale_grads(&mut self, scale: f32) {
        self.grads.weights.scale(scale);
    }

    fn clip_grads(&mut self) {
        self.grads.weights.clip(-CLIP, CLIP);
    }
}
