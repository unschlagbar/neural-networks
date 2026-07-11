use std::any::Any;
use std::io;

use crate::nn::causal_conv1d::CausalConv1dLayer;
use crate::nn::dropout::DropoutLayer;
use crate::nn::embedding::EmbeddingLayer;
use crate::nn::linear::LinearLayer;
use crate::nn::linear_nb::LinearNBLayer;
use crate::nn::lstm::LSTMLayer;
use crate::nn::mlstm::MLSTMLayer;
use crate::nn::mlstm_block::MLSTMBlock;
use crate::nn::rms_norm::RMSNorm;
use crate::nn::silu_dense::SiluDenseLayer;
use crate::nn::slstm::SLSTMLayer;
use crate::nn::slstm_block::SLSTMBlock;
use crate::nn::soft_cap::SoftCapLayer;
use crate::sequential::Sequential;

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
///
/// Caches are type-erased (`dyn DynCache`) but each layer impl downcasts to its
/// own concrete type, so there is no runtime overhead beyond one virtual dispatch.
///
/// Gradient accumulators live *inside* the layer (not in `Sequential`), which:
///   • halves the number of parallel `Vec`s in `Sequential`,
///   • keeps related data (weights + their grads) physically adjacent in memory.
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
    ///
    /// `delta` (dL/d output) is modified in-place and the caller should discard it
    /// afterwards — use `cache.input_grad()` for the gradient flowing to the layer below.
    ///
    /// For LSTM layers `bptt_hidden_grad()` is updated as a side-effect.
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
    /// plain Adam, a positive λ for AdamW. Callers set λ per stack, so the
    /// hierarchical encoder/decoder and backbone can be decayed independently.
    fn apply_grads(&mut self, lr: f32, weight_decay: f32);
    fn clear_grads(&mut self);

    /// Add the gradient accumulators of `other` — a same-type replica of this
    /// layer — into this layer's accumulators (raw grads only; optimizer
    /// moments are untouched). Reduces per-thread gradients after a
    /// data-parallel backward phase; layers that appear in a parallel-trained
    /// stack must override this.
    fn add_grads_from(&mut self, _other: &mut dyn NnLayer) {
        unimplemented!("add_grads_from: this layer type does not support data-parallel training");
    }

    /// Overwrite this layer's weights with those of `other` — a same-type,
    /// same-shape layer. Refreshes an existing replica in place after an
    /// optimizer step (plain memcpy: no serialization, no allocation); layers
    /// that appear in a parallel-trained stack must override this.
    fn copy_weights_from(&mut self, _other: &dyn NnLayer) {
        unimplemented!(
            "copy_weights_from: this layer type does not support data-parallel training"
        );
    }

    /// Clear h, c
    fn reset_state(&mut self) {}

    /// Zero BPTT gradient accumulators (dh_bptt, dc_bptt, …) without touching
    /// the forward hidden state (h, c).  Called at HM-RNN FLUSH boundaries so
    /// that gradients do not leak across a char-model reset.  No-op for
    /// stateless layers.
    fn reset_bptt_state(&mut self) {}

    fn accumulate_init_grad(&mut self) {}

    /// Total h+c state size (0 for stateless layers, 2*hidden for sLSTM layers).
    fn state_size(&self) -> usize {
        0
    }

    /// Overwrite the layer's h and c from `buf[offset..]`; returns the new offset.
    fn inject_state(&mut self, _buf: &[f32], offset: usize) -> usize {
        offset
    }

    /// Copy dh_bptt then dc_bptt into `buf[offset..]`; returns the new offset.
    fn collect_bptt_grad(&mut self, _buf: &mut [f32], offset: usize) -> usize {
        offset
    }
}

pub struct SequentialBuilder {
    input_size: usize,
    output_size: usize,
    layers: Vec<Box<dyn NnLayer>>,
    in_parallel: bool,
    branch1: Option<Box<dyn NnLayer>>,
    branch2: Option<Box<dyn NnLayer>>,
}
impl SequentialBuilder {
    pub fn new(input_size: usize) -> Self {
        Self {
            input_size,
            output_size: input_size,
            layers: Vec::new(),
            in_parallel: false,
            branch1: None,
            branch2: None,
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

    pub fn mlstm(mut self, num_heads: usize, dqk: usize) -> Self {
        let hidden = self.output_size;
        let layer = MLSTMLayer::new(hidden, hidden, num_heads, dqk);
        self.layer(Box::new(layer), hidden);
        self
    }

    /// Multi-Head-mLSTM-Block im Stil von xLSTM-7B
    /// (zwei separate Residuals + RMSNorm + SwiGLU).
    /// up_size = 8·hidden / 3 (paper default).
    pub fn mlstm_block(mut self, num_heads: usize, dqk: usize) -> Self {
        let hidden = self.output_size;
        let up = (hidden * 8) / 3;
        let layer = MLSTMBlock::new(hidden, num_heads, dqk, up);
        self.layer(Box::new(layer), hidden);
        self
    }

    /// xLSTM-style sLSTM-Block:
    ///   RMSNorm1 → sLSTM cell → RMSNorm2 → SwiGLU-MLP → Residual.
    ///
    /// Input and output are both `hidden` (that is the purpose of the residual —
    /// the block can be stacked freely). The internal SwiGLU width (`up_size`)
    /// is chosen as `8·hidden / 3` (MLP params ≈ 8·H², analogous to GPT-NeoX
    /// / LLaMA-style blocks).
    pub fn slstm_block(mut self, hidden: usize) -> Self {
        assert_eq!(
            self.output_size, hidden,
            "slstm_block erwartet input_size == hidden ({} != {})",
            self.output_size, hidden,
        );
        let up = (hidden * 8) / 3;
        let layer = SLSTMBlock::new(hidden, up);
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

    // In nn_layer.rs → impl SequentialBuilder
    pub fn rms_norm(mut self) -> Self {
        let wrapper = RMSNorm::new(self.output_size);
        self.layers.push(Box::new(wrapper));
        self
    }

    fn layer(&mut self, layer: Box<dyn NnLayer>, hidden: usize) {
        if self.in_parallel {
            if self.branch1.is_none() {
                self.branch1 = Some(layer);
            } else if self.branch2.is_none() {
                self.branch2 = Some(layer);
            } else {
                unreachable!()
            }
        } else {
            self.output_size = hidden;
            self.layers.push(layer);
        }
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
