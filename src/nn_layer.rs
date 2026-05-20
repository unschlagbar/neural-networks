use std::any::Any;
use std::io;

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
use crate::sequential::Sequential;

/// Type-erased per-timestep forward cache.
/// Concrete types downcast via `as_any_mut()` inside each layer's own backward impl.
/// Storing `Vec<Box<dyn DynCache>>` lets `Sequential` iterate without any match.
pub trait DynCache: Send {
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
}
impl<T: 'static> Dyn for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub trait NnLayer: Dyn {
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

    /// Schreibt NUR die Gewichte.
    /// Shapes, act_id und dropout_rate stehen bereits im Architektur-Header
    /// (Sequential::save), hier kommen sie nicht nochmal vor.
    fn save(&self, w: &mut dyn io::Write) -> io::Result<()>;

    fn make_cache(&self) -> Box<dyn DynCache>;

    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;

    fn apply_grads(&mut self, lr: f32);
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
    ///   RMSNorm1 → sLSTM-Zelle → RMSNorm2 → SwiGLU-MLP → Residual.
    ///
    /// Ein-/Ausgang ist jeweils `hidden` (das ist der Sinn des Residuals — man
    /// kann den Block einfach stapeln). Die interne SwiGLU-Weite (`up_size`)
    /// wird als `8·hidden / 3` gewählt (MLP-Parameter ≈ 8·H², analog zu GPT-NeoX
    /// / LLaMA-Style-Blöcken).
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
