use std::any::Any;
use std::io;

use super::Sequential;
use super::layers::{
    causal_conv1d::CausalConv1dLayer, dropout::DropoutLayer, embedding::EmbeddingLayer,
    linear::LinearLayer, linear_nb::LinearNBLayer, lstm::LSTMLayer, rms_norm::RMSNorm,
    silu_dense::SiluDenseLayer, slstm::SLSTMLayer, soft_cap::SoftCapLayer,
};

/// Type-erased per-timestep forward cache.
/// Concrete types downcast via `as_any_mut()` inside each layer's own backward impl.
/// Storing `Vec<Box<dyn DynCache>>` lets `Sequential` iterate without any match.
pub trait DynCache: Send + Sync {
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn as_any(&self) -> &dyn Any;
    /// Post-activation output — fed as input to the next layer.
    fn output(&self) -> &[f32];
    /// dL/d(input), written by the backward pass. Used to form `delta` for the layer below.
    fn input_grad(&self) -> &[f32];
}

/// Unified trait for all layer types.
pub trait Dyn: 'static {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}
impl<T: 'static> Dyn for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub trait NnLayer: Dyn + Send + Sync {
    fn forward(&mut self, input: &[f32], cache: &mut dyn DynCache);

    fn forward_sample(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        self.forward(input, cache);
    }

    /// Backward for one timestep.
    fn backward(&mut self, delta: &mut [f32], cache: &mut dyn DynCache);

    fn layer_tag(&self) -> u8;

    /// Writes ONLY the weights.
    /// Shapes, act_id and dropout_rate are already in the architecture header
    /// (Sequential::save) and are not repeated here.
    fn save(&self, w: &mut dyn io::Write) -> io::Result<()>;

    fn make_cache(&self) -> Box<dyn DynCache>;

    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;

    /// Apply accumulated gradients. `weight_decay` is the per-step decoupled
    /// decay coefficient (λ) applied to this layer's *decay-eligible* weight
    /// matrices (interior projections); embedding tables, no-bias logit heads,
    /// biases and norm scales are never decayed regardless of λ. Pass `0.0` for
    /// plain Adam, a positive λ for AdamW.
    fn apply_grads(&mut self, lr: f32, weight_decay: f32);
    fn clear_grads(&mut self);

    /// Clear h, c
    fn reset_state(&mut self) {}

    /// Zero BPTT gradient accumulators (dh_bptt, dc_bptt, …) without touching
    /// the forward hidden state (h, c).  Called at HM-RNN FLUSH boundaries so
    /// that gradients do not leak across a char-model reset.  No-op for
    /// stateless layers.
    fn reset_bptt_state(&mut self) {}

    fn accumulate_init_grad(&mut self) {}

}

pub struct SequentialBuilder {
    input_size: usize,
    output_size: usize,
    layers: Vec<Box<dyn NnLayer>>,
}
impl SequentialBuilder {
    pub fn new(input_size: usize) -> Self {
        Self {
            input_size,
            output_size: input_size,
            layers: Vec::new(),
        }
    }

    pub fn embedding(mut self, hidden: usize) -> Self {
        let layer = EmbeddingLayer::new(self.output_size, hidden);
        self.layer(Box::new(layer), hidden);
        self
    }

    pub fn silu_dense(mut self, hidden: usize) -> Self {
        let layer = SiluDenseLayer::new(self.output_size, hidden);
        self.layer(Box::new(layer), hidden);
        self
    }

    pub fn linear(mut self, hidden: usize) -> Self {
        let layer = LinearLayer::new(self.output_size, hidden);
        self.layer(Box::new(layer), hidden);
        self
    }

    pub fn linear_zeroed(mut self, hidden: usize) -> Self {
        let layer = LinearLayer::zeroed(self.output_size, hidden);
        self.layer(Box::new(layer), hidden);
        self
    }

    pub fn linear_no_bias(mut self, hidden: usize) -> Self {
        let layer = LinearNBLayer::new(self.output_size, hidden);
        self.layer(Box::new(layer), hidden);
        self
    }

    pub fn lstm(mut self, hidden: usize) -> Self {
        let layer = LSTMLayer::new(self.output_size, hidden);
        self.layer(Box::new(layer), hidden);
        self
    }

    pub fn slstm(mut self, hidden: usize) -> Self {
        let layer = SLSTMLayer::new(self.output_size, hidden);
        self.layer(Box::new(layer), hidden);
        self
    }

    pub fn causal_conv1d(mut self, kernel_size: usize) -> Self {
        let channels = self.output_size;
        let layer = CausalConv1dLayer::new(channels, kernel_size);
        self.layer(Box::new(layer), channels);
        self
    }

    /// Soft-caps the previous layer's output to (−cap, cap) via
    /// `y = cap · tanh(x / cap)`. Meant to follow a logit head.
    pub fn soft_cap(mut self, cap: f32) -> Self {
        let layer = SoftCapLayer::new(self.output_size, cap);
        self.layer(Box::new(layer), self.output_size);
        self
    }

    pub fn dropout(mut self, rate: f32) -> Self {
        let layer = DropoutLayer::new(self.output_size, rate);
        self.layer(Box::new(layer), self.output_size);
        self
    }

    pub fn rms_norm(mut self) -> Self {
        let wrapper = RMSNorm::new(self.output_size);
        self.layers.push(Box::new(wrapper));
        self
    }

    fn layer(&mut self, layer: Box<dyn NnLayer>, hidden: usize) {
        self.output_size = hidden;
        self.layers.push(layer);
    }

    pub fn build(self) -> Sequential {
        let max_size = self
            .layers
            .iter()
            .map(|l| l.output_size())
            .max()
            .unwrap_or(0);

        Sequential {
            input_size: self.input_size,
            output_size: self.output_size,
            layers: self.layers,
            cache: Vec::new(),
            delta_buf: vec![0.0; max_size].into(),
            input_buf: vec![0.0; self.input_size].into(),
        }
    }
}
