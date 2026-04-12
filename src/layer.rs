#![allow(unused)]

use std::{any::Any, io};

use iron_oxide::collections::Matrix;
use rand::{Rng, RngExt, rng};

use crate::{
    activations::Activate,
    lstm::{CLIP, add_vec_in_place, sub_in_place, sub_vec_in_place},
    nn_layer::{DynCache, NnLayer},
};

// ── DenseCache ────────────────────────────────────────────────────────────────

pub struct DenseCache {
    /// Saved input (for ∂L/∂W = x · δᵀ).
    pub input: Vec<f32>,
    /// Post-activation output; activation derivative is applied in-place during backward.
    pub output: Vec<f32>,
    /// dL/d(input), populated in backward.
    pub dx: Vec<f32>,
}

impl DenseCache {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            input: vec![0.0; input_size],
            output: vec![0.0; output_size],
            dx: vec![0.0; input_size],
        }
    }
}

impl DynCache for DenseCache {
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

// ── DenseGrads (stored inside the layer) ─────────────────────────────────────

pub struct DenseGrads {
    pub weights: Matrix,
    pub biases: Vec<f32>,
}

impl DenseGrads {
    pub fn zeros(input_size: usize, output_size: usize) -> Self {
        Self {
            weights: Matrix::zeros(input_size, output_size),
            biases: vec![0.0; output_size],
        }
    }
}

// ── DenseLayer ────────────────────────────────────────────────────────────────

pub struct DenseLayer<A: Activate> {
    pub weights: Matrix,
    pub biases: Box<[f32]>,
    pub activation: A,
    /// Gradient accumulators — cleared per batch, applied in `sgd_step`.
    pub grads: DenseGrads,
}

impl<A: Activate> DenseLayer<A> {
    pub fn new(input_size: usize, hidden_size: usize, activation: A) -> Self {
        let mut rng = rng();
        let weights = Matrix::random(input_size, hidden_size, 1.0);
        let biases = (0..hidden_size)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        Self {
            weights,
            biases,
            activation,
            grads: DenseGrads::zeros(input_size, hidden_size),
        }
    }

    pub fn from_loaded(
        input_size: usize,
        output_size: usize,
        activation: A,
        weights: Matrix,
        biases: Box<[f32]>,
    ) -> Self {
        debug_assert_eq!(weights.rows(), input_size);
        debug_assert_eq!(weights.cols(), output_size);
        debug_assert_eq!(biases.len(), output_size);

        Self {
            weights,
            biases,
            activation,
            grads: DenseGrads::zeros(input_size, output_size),
        }
    }

    /// Convenience: pass a concrete activation directly — it is boxed internally.
    pub fn with_activation(input_size: usize, output_size: usize, activation: A) -> Self {
        Self::new(input_size, output_size, activation)
    }

    /// matmul + bias into a pre-allocated output buffer.
    #[inline]
    fn matmul_add_bias(&self, input: &[f32], out: &mut [f32]) {
        out.copy_from_slice(&self.biases);
        for (i, &x) in input.iter().enumerate() {
            for (j, &w) in self.weights[i].iter().enumerate() {
                out[j] += x * w;
            }
        }
    }

    /// Standard forward: z = Wx+b → activation(z).
    pub fn forward(&self, input: &[f32], cache: &mut DenseCache) {
        cache.input.copy_from_slice(input);
        self.matmul_add_bias(input, &mut cache.output);
        self.activation.activate(&mut cache.output);
    }

    /// Backward.
    ///
    /// `delta` (dL/d output) is modified in-place:
    ///   - Non-softmax: element-wise multiplied by the activation derivative.
    ///   - Softmax output layer: caller passes the fused cross-entropy gradient ŷ−y.
    ///
    /// `cache.dx` ← dL/d(input) = Wᵀ · delta.
    pub fn backward(&mut self, delta: &mut [f32], cache: &mut DenseCache) {
        self.activation.derivative_active(&mut cache.output);
        delta
            .iter_mut()
            .zip(&cache.output)
            .for_each(|(d, o)| *d *= o);

        self.grads.weights.add_outer(&cache.input, delta);
        add_vec_in_place(&mut self.grads.biases, delta);

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

// ── impl NnLayer for DenseLayer ───────────────────────────────────────────────

impl<A: Activate> NnLayer for DenseLayer<A> {
    fn forward(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<DenseCache>()
            .expect("DenseLayer::forward — expected DenseCache");
        DenseLayer::forward(self, input, c);
    }

    fn backward(&mut self, delta: &mut [f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<DenseCache>()
            .expect("DenseLayer::backward — expected DenseCache");
        DenseLayer::backward(self, delta, c);
    }

    fn layer_tag(&self) -> u8 {
        1
    } // TAG_DENSE

    fn save(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        crate::saving::write_u32(w, self.input_size() as u32)?;
        crate::saving::write_u32(w, self.output_size() as u32)?;
        crate::saving::write_u8(w, self.activation.activation_id())?;
        crate::saving::write_matrix(w, &self.weights)?;
        crate::saving::write_f32_slice(w, &self.biases)?;
        Ok(())
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
        sub_vec_in_place(&mut self.biases, &self.grads.biases, lr);
    }

    fn clear_grads(&mut self) {
        self.grads.weights.clear();
        self.grads.biases.fill(0.0);
    }

    fn scale_grads(&mut self, scale: f32) {
        self.grads.weights.scale(scale);
        self.grads.biases.iter_mut().for_each(|x| *x *= scale);
    }

    fn clip_grads(&mut self) {
        self.grads.weights.clip(-CLIP, CLIP);
        self.grads
            .biases
            .iter_mut()
            .for_each(|x| *x = x.clamp(-CLIP, CLIP));
    }
}
