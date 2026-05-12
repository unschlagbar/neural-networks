use std::any::Any;

use iron_oxide::collections::Matrix;
use rand::random_range;

use crate::{
    nn::add_vec_in_place,
    nn_layer::{DynCache, NnLayer},
    optimizers::{GradMatrix, GradMatrixOps, GradVec, GradVecOps},
};

pub struct SiluDenseCache {
    /// Saved input (für ∂L/∂W = x · δᵀ).
    pub input: Box<[f32]>,
    /// Pre-activation z = Wx + b — im Backward für silu'(z) benötigt.
    pub pre_activation: Box<[f32]>,
    /// Post-activation output y = silu(z) = z · σ(z).
    pub output: Box<[f32]>,
    /// dL/d(input).
    pub dx: Box<[f32]>,
}

impl SiluDenseCache {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            input: vec![0.0; input_size].into(),
            pre_activation: vec![0.0; output_size].into(),
            output: vec![0.0; output_size].into(),
            dx: vec![0.0; input_size].into(),
        }
    }
}

impl DynCache for SiluDenseCache {
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

pub struct SiluDenseGrads {
    pub weights: GradMatrix,
    pub biases: GradVec,
}

impl SiluDenseGrads {
    pub fn zeros(input_size: usize, output_size: usize) -> Self {
        Self {
            weights: GradMatrix::zeros(input_size, output_size),
            biases: GradVec::zeros(output_size),
        }
    }
}

pub struct SiluDenseLayer {
    pub weights: Matrix,
    pub biases: Box<[f32]>,
    /// Gradient-Akkumulatoren — pro Batch geleert, in `apply_grads` angewendet.
    pub grads: SiluDenseGrads,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    // numerisch stabile Variante
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

impl SiluDenseLayer {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        // Glorot-artige Initialisierung (identisch zu DenseLayer::new)
        let scale = (6.0 / (input_size as f32 + hidden_size as f32)).sqrt();
        let weights = Matrix::random(input_size, hidden_size, scale);
        let biases = (0..hidden_size).map(|_| random_range(-0.5..0.5)).collect();
        Self {
            weights,
            biases,
            grads: SiluDenseGrads::zeros(input_size, hidden_size),
        }
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

        Self {
            weights,
            biases,
            grads: SiluDenseGrads::zeros(input_size, output_size),
        }
    }

    /// matmul + bias in einen vorallokierten Puffer.
    #[inline]
    fn matmul_add_bias(&self, input: &[f32], out: &mut [f32]) {
        out.copy_from_slice(&self.biases);
        for (i, &x) in input.iter().enumerate() {
            for (j, &w) in self.weights[i].iter().enumerate() {
                out[j] += x * w;
            }
        }
    }

    /// Forward: z = Wx + b → y = silu(z).
    pub fn forward(&self, input: &[f32], cache: &mut SiluDenseCache) {
        cache.input.copy_from_slice(input);
        self.matmul_add_bias(input, &mut cache.pre_activation);

        // y = z · σ(z)
        for (y, &z) in cache.output.iter_mut().zip(&cache.pre_activation) {
            *y = z * sigmoid(z);
        }
    }

    /// Backward.
    ///
    /// `delta` (dL/d output) wird in-place mit silu'(z) multipliziert:
    ///     silu'(z) = σ(z) · (1 + z · (1 − σ(z)))
    ///
    /// Anschließend:
    ///     grads.W += x · deltaᵀ
    ///     grads.b += delta
    ///     cache.dx = Wᵀ · delta
    pub fn backward(&mut self, delta: &mut [f32], cache: &mut SiluDenseCache) {
        for (d, &z) in delta.iter_mut().zip(cache.pre_activation.iter()) {
            let s = sigmoid(z);
            *d *= s * (1.0 + z * (1.0 - s));
        }

        self.grads.weights.matrix().add_outer(&cache.input, delta);
        add_vec_in_place(&mut self.grads.biases.vec(), delta);

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

impl NnLayer for SiluDenseLayer {
    //type Cache = SiluDenseCache;
    fn forward(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<SiluDenseCache>()
            .expect("SiluDenseLayer::forward — expected SiluDenseCache");
        SiluDenseLayer::forward(self, input, c);
    }

    fn backward(&mut self, delta: &mut [f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<SiluDenseCache>()
            .expect("SiluDenseLayer::backward — expected SiluDenseCache");
        SiluDenseLayer::backward(self, delta, c);
    }

    fn layer_tag(&self) -> u8 {
        10
    }

    fn save(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        crate::saving::write_matrix(w, &self.weights)?;
        crate::saving::write_f32_slice(w, &self.biases)?;
        Ok(())
    }

    fn make_cache(&self) -> Box<dyn DynCache> {
        Box::new(SiluDenseCache::new(self.input_size(), self.output_size()))
    }

    fn input_size(&self) -> usize {
        self.weights.rows()
    }
    fn output_size(&self) -> usize {
        self.weights.cols()
    }

    fn apply_grads(&mut self, lr: f32) {
        self.grads.weights.apply_to(&mut self.weights, lr);
        self.grads.biases.apply_to(&mut self.biases, lr);
    }

    fn clear_grads(&mut self) {
        self.grads.weights.clear();
        self.grads.biases.clear();
    }
}
