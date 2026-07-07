use std::any::Any;

use iron_oxide::collections::Matrix;

use crate::{
    nn::{GRAD_BLOCK, dot, matvec_into, outer_acc_block},
    nn_layer::{DynCache, NnLayer},
    optimizers::{GradMatrixNoDecay, GradMatrixOps, add_grad_matrix},
};

pub struct LinearNBCache {
    /// Saved input (for ∂L/∂W = x · δᵀ).
    pub input: Box<[f32]>,
    /// Post-activation output; activation derivative is applied in-place during backward.
    pub output: Box<[f32]>,
    /// dL/d(input), populated in backward.
    pub dx: Box<[f32]>,
}

impl LinearNBCache {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            input: vec![0.0; input_size].into(),
            output: vec![0.0; output_size].into(),
            dx: vec![0.0; input_size].into(),
        }
    }
}

impl DynCache for LinearNBCache {
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

pub struct LinearNBGrads {
    pub weights: GradMatrixNoDecay,
}

impl LinearNBGrads {
    pub fn zeros(input_size: usize, output_size: usize) -> Self {
        Self {
            weights: GradMatrixNoDecay::zeros(input_size, output_size),
        }
    }
}

pub struct LinearNBLayer {
    pub weights: Matrix,
    /// Gradient accumulators — cleared per batch, applied in `sgd_step`.
    pub grads: LinearNBGrads,

    // Deferred weight-gradient accumulation (see slstm.rs): backward stashes
    // (input, delta) per step; blocks are folded into `grads.weights` every
    // GRAD_BLOCK steps and before any read of the grads.
    pend_len: usize,
    pend_x: Box<[f32]>, // GRAD_BLOCK × input
    pend_d: Box<[f32]>, // GRAD_BLOCK × output
}

impl LinearNBLayer {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let scale = (6.0 / (input_size as f32 + hidden_size as f32)).sqrt();
        Self::from_parts(Matrix::random(input_size, hidden_size, scale))
    }

    pub fn zeroed(input_size: usize, hidden_size: usize) -> Self {
        Self::from_parts(Matrix::zeros(input_size, hidden_size))
    }

    pub fn from_loaded(input_size: usize, output_size: usize, weights: Matrix) -> Self {
        debug_assert_eq!(weights.rows(), input_size);
        debug_assert_eq!(weights.cols(), output_size);
        Self::from_parts(weights)
    }

    fn from_parts(weights: Matrix) -> Self {
        let (input_size, output_size) = (weights.rows(), weights.cols());
        Self {
            weights,
            grads: LinearNBGrads::zeros(input_size, output_size),
            pend_len: 0,
            pend_x: vec![0.0; GRAD_BLOCK * input_size].into(),
            pend_d: vec![0.0; GRAD_BLOCK * output_size].into(),
        }
    }

    /// Standard forward: z = Wx → activation(z).
    pub fn forward(&self, input: &[f32], cache: &mut LinearNBCache) {
        cache.input.copy_from_slice(input);
        matvec_into(&self.weights, input, &mut cache.output);
    }

    /// Backward.
    ///
    /// `delta` (dL/d output) is modified in-place:
    ///   - Non-softmax: element-wise multiplied by the activation derivative.
    ///   - Softmax output layer: caller passes the fused cross-entropy gradient ŷ−y.
    ///
    /// `cache.dx` ← dL/d(input) = Wᵀ · delta.
    pub fn backward(&mut self, delta: &mut [f32], cache: &mut LinearNBCache) {
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
}

impl NnLayer for LinearNBLayer {
    //type Cache = LinearNBCache;
    fn forward(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<LinearNBCache>()
            .expect("DenseLayer::forward — expected DenseCache");
        LinearNBLayer::forward(self, input, c);
    }

    fn backward(&mut self, delta: &mut [f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<LinearNBCache>()
            .expect("DenseLayer::backward — expected DenseCache");
        LinearNBLayer::backward(self, delta, c);
    }

    fn layer_tag(&self) -> u8 {
        3
    }

    fn save(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        crate::saving::write_matrix(w, &self.weights)
    }

    fn make_cache(&self) -> Box<dyn DynCache> {
        Box::new(LinearNBCache::new(self.input_size(), self.output_size()))
    }

    fn input_size(&self) -> usize {
        self.weights.rows()
    }
    fn output_size(&self) -> usize {
        self.weights.cols()
    }

    fn apply_grads(&mut self, lr: f32) {
        self.flush_grads();
        self.grads.weights.clip();
        self.grads.weights.apply_to(&mut self.weights, lr);
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
    }

    fn add_grads_from(&mut self, other: &mut dyn NnLayer) {
        let o = other
            .as_any_mut()
            .downcast_mut::<Self>()
            .expect("LinearNBLayer::add_grads_from — replica layer type mismatch");
        self.flush_grads();
        o.flush_grads();
        add_grad_matrix(&mut self.grads.weights, &mut o.grads.weights);
    }

    fn copy_weights_from(&mut self, other: &dyn NnLayer) {
        let o = other
            .as_any()
            .downcast_ref::<Self>()
            .expect("LinearNBLayer::copy_weights_from — replica layer type mismatch");
        self.weights.copy_from(&o.weights);
    }
}
