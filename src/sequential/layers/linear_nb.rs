use std::any::Any;

use iron_oxide::collections::Matrix;

use crate::{
    sequential::layers::{GRAD_BLOCK, dot, matvec_into, outer_acc_block},
    sequential::layer::{DynCache, NnLayer},
    sequential::optim::GradMatrix,
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
    pub weights: GradMatrix,
}

impl LinearNBGrads {
    pub fn zeros(input_size: usize, output_size: usize) -> Self {
        Self {
            weights: GradMatrix::zeros(input_size, output_size),
        }
    }
}

pub struct LinearNBLayer {
    pub weights: Matrix,
    /// Gradient accumulators — cleared per batch, applied in `sgd_step`.
    pub grads: LinearNBGrads,

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
    pub fn backward(&mut self, delta: &mut [f32], cache: &mut LinearNBCache) {
        let (r, c) = (self.weights.rows(), self.weights.cols());
        let slot = self.pend_len;
        self.pend_x[slot * r..(slot + 1) * r].copy_from_slice(&cache.input);
        self.pend_d[slot * c..(slot + 1) * c].copy_from_slice(delta);
        self.pend_len += 1;

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
        crate::bytes::write_matrix(w, &self.weights)
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

    fn apply_grads(&mut self, lr: f32, _weight_decay: f32) {
        self.flush_grads();
        self.grads.weights.clip();
        // No-bias logit / embedding-like head: never weight-decayed (pass 0.0).
        self.grads.weights.apply_to(&mut self.weights, lr, 0.0);
    }

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

}
