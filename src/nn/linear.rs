use std::any::Any;

use iron_oxide::collections::Matrix;
use rand::random_range;

use crate::{
    nn::{GRAD_BLOCK, add_vec_in_place, dot, matvec_acc, outer_acc_block},
    nn_layer::{DynCache, NnLayer},
    optimizers::{GradMatrix, GradMatrixOps, GradVec, GradVecOps, add_grad_matrix, add_grad_vec},
};

pub struct LinearCache {
    /// Saved input (for ∂L/∂W = x · δᵀ).
    pub input: Box<[f32]>,
    /// Post-activation output; activation derivative is applied in-place during backward.
    pub output: Box<[f32]>,
    /// dL/d(input), populated in backward.
    pub dx: Box<[f32]>,
}

impl LinearCache {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            input: vec![0.0; input_size].into(),
            output: vec![0.0; output_size].into(),
            dx: vec![0.0; input_size].into(),
        }
    }
}

impl DynCache for LinearCache {
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

pub struct LinearGrads {
    pub weights: GradMatrix,
    pub biases: GradVec,
}

impl LinearGrads {
    pub fn zeros(input_size: usize, output_size: usize) -> Self {
        Self {
            weights: GradMatrix::zeros(input_size, output_size),
            biases: GradVec::zeros(output_size),
        }
    }
}

pub struct LinearLayer {
    pub weights: Matrix,
    pub biases: Box<[f32]>,
    /// Gradient accumulators — cleared per batch, applied in `sgd_step`.
    pub grads: LinearGrads,

    // Deferred weight-gradient accumulation (see slstm.rs): backward stashes
    // (input, delta) per step; blocks are folded into `grads.weights` every
    // GRAD_BLOCK steps and before any read of the grads.
    pend_len: usize,
    pend_x: Box<[f32]>, // GRAD_BLOCK × input
    pend_d: Box<[f32]>, // GRAD_BLOCK × output
}

impl LinearLayer {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let scale = (6.0 / (input_size as f32 + hidden_size as f32)).sqrt();
        let weights = Matrix::random(input_size, hidden_size, scale);

        let scale = (1.0 / input_size as f32).sqrt();
        let biases = (0..hidden_size)
            .map(|_| random_range(-scale..scale))
            .collect();
        Self::from_parts(weights, biases)
    }

    pub fn zeroed(input_size: usize, hidden_size: usize) -> Self {
        Self::from_parts(
            Matrix::zeros(input_size, hidden_size),
            vec![0.0; hidden_size].into(),
        )
    }

    pub fn from_loaded(
        input_size: usize,
        output_size: usize,
        weights: Matrix,
        biases: Box<[f32]>,
    ) -> Self {
        debug_assert_eq!(weights.rows(), input_size);
        debug_assert_eq!(weights.cols(), output_size);
        debug_assert_eq!(biases.len(), output_size);
        Self::from_parts(weights, biases)
    }

    fn from_parts(weights: Matrix, biases: Box<[f32]>) -> Self {
        let (input_size, output_size) = (weights.rows(), weights.cols());
        Self {
            weights,
            biases,
            grads: LinearGrads::zeros(input_size, output_size),
            pend_len: 0,
            pend_x: vec![0.0; GRAD_BLOCK * input_size].into(),
            pend_d: vec![0.0; GRAD_BLOCK * output_size].into(),
        }
    }

    /// Standard forward: z = Wx+b → activation(z).
    pub fn forward(&self, input: &[f32], cache: &mut LinearCache) {
        cache.input.copy_from_slice(input);
        cache.output.copy_from_slice(&self.biases);
        matvec_acc(&self.weights, input, &mut cache.output);
    }

    /// Backward.
    ///
    /// `delta` (dL/d output) is modified in-place:
    ///   - Non-softmax: element-wise multiplied by the activation derivative.
    ///   - Softmax output layer: caller passes the fused cross-entropy gradient ŷ−y.
    ///
    /// `cache.dx` ← dL/d(input) = Wᵀ · delta.
    pub fn backward(&mut self, delta: &mut [f32], cache: &mut LinearCache) {
        add_vec_in_place(&mut self.grads.biases.vec(), delta);

        // Defer the weight-grad outer product: stash (input, delta) and fold
        // blocks in flush_grads, so gW is streamed once per block instead of
        // read-modified-written every step.
        let (r, c) = (self.weights.rows(), self.weights.cols());
        let slot = self.pend_len;
        self.pend_x[slot * r..(slot + 1) * r].copy_from_slice(&cache.input);
        self.pend_d[slot * c..(slot + 1) * c].copy_from_slice(delta);
        self.pend_len += 1;

        // Lane-accumulated dot per weight row — a serial `*dx +=` chain
        // would keep this loop scalar (f32 adds don't reassociate).
        for (i, dx) in cache.dx.iter_mut().enumerate() {
            *dx = dot(delta, &self.weights[i]);
        }

        if self.pend_len == GRAD_BLOCK {
            self.flush_grads();
        }
    }

    /// Fold the pending (input, delta) outer products into the weight grads.
    /// No-op when nothing is pending.
    pub fn flush_grads(&mut self) {
        let n = self.pend_len;
        if n == 0 {
            return;
        }
        self.pend_len = 0;
        let (r, c) = (self.weights.rows(), self.weights.cols());
        let gw = self.grads.weights.matrix().as_slice_mut();
        outer_acc_block(gw, &self.pend_x, &self.pend_d, n, r, c);
    }

    pub fn input_size(&self) -> usize {
        self.weights.rows()
    }
    pub fn output_size(&self) -> usize {
        self.weights.cols()
    }

    /// Fold a replica's weight/bias grads into this layer (data-parallel reduction).
    pub fn add_grads(&mut self, other: &mut Self) {
        self.flush_grads();
        other.flush_grads();
        add_grad_matrix(&mut self.grads.weights, &mut other.grads.weights);
        add_grad_vec(&mut self.grads.biases, &mut other.grads.biases);
    }

    /// Overwrite weights and biases with `other`'s (in-place replica refresh).
    pub fn copy_weights(&mut self, other: &Self) {
        self.weights.copy_from(&other.weights);
        self.biases.copy_from_slice(&other.biases);
    }
}

impl NnLayer for LinearLayer {
    //type Cache = LinearCache;
    fn forward(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<LinearCache>()
            .expect("DenseLayer::forward — expected DenseCache");
        LinearLayer::forward(self, input, c);
    }

    fn backward(&mut self, delta: &mut [f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<LinearCache>()
            .expect("DenseLayer::backward — expected DenseCache");
        LinearLayer::backward(self, delta, c);
    }

    fn layer_tag(&self) -> u8 {
        12
    }

    fn save(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        crate::saving::write_u32(w, self.input_size() as u32)?;
        crate::saving::write_u32(w, self.output_size() as u32)?;
        crate::saving::write_matrix(w, &self.weights)?;
        crate::saving::write_f32_slice(w, &self.biases)?;
        Ok(())
    }

    fn make_cache(&self) -> Box<dyn DynCache> {
        Box::new(LinearCache::new(self.input_size(), self.output_size()))
    }

    fn input_size(&self) -> usize {
        self.weights.rows()
    }
    fn output_size(&self) -> usize {
        self.weights.cols()
    }

    fn apply_grads(&mut self, lr: f32) {
        self.flush_grads();
        self.grads.weights.apply_to(&mut self.weights, lr);
        self.grads.biases.apply_to(&mut self.biases, lr);
    }

    // Window seams — fold any partial pending block so callers (drivers,
    // tests, diagnostics) that read `grads` after a backward window see the
    // materialized values.
    fn accumulate_init_grad(&mut self) {
        self.flush_grads();
    }

    fn reset_bptt_state(&mut self) {
        self.flush_grads();
    }

    fn clear_grads(&mut self) {
        // Pending outer products belong to the grads being discarded.
        self.pend_len = 0;
        self.grads.weights.clear();
        self.grads.biases.clear();
    }

    fn add_grads_from(&mut self, other: &mut dyn NnLayer) {
        let o = other
            .as_any_mut()
            .downcast_mut::<Self>()
            .expect("LinearLayer::add_grads_from — replica layer type mismatch");
        self.add_grads(o);
    }

    fn copy_weights_from(&mut self, other: &dyn NnLayer) {
        let o = other
            .as_any()
            .downcast_ref::<Self>()
            .expect("LinearLayer::copy_weights_from — replica layer type mismatch");
        self.copy_weights(o);
    }
}
