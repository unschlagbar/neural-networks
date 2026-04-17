use std::any::Any;
use std::io;

use crate::activations::Activate;
use crate::dense::DenseLayer;
use crate::dropout::DropoutLayer;
use crate::indrnn::IndRNNLayer;
use crate::lstm::LSTMLayer;
use crate::parallel::ParallelLayer;
use crate::projection::Projection;
use crate::sequential::Sequential;
use crate::softmax::SoftmaxLayer;

// ── DynCache trait ────────────────────────────────────────────────────────────

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

// ── NnLayer trait ─────────────────────────────────────────────────────────────

/// Unified trait for all layer types.
///
/// Caches are type-erased (`dyn DynCache`) but each layer impl downcasts to its
/// own concrete type, so there is no runtime overhead beyond one virtual dispatch.
///
/// Gradient accumulators live *inside* the layer (not in `Sequential`), which:
///   • halves the number of parallel `Vec`s in `Sequential`,
///   • keeps related data (weights + their grads) physically adjacent in memory.
pub trait NnLayer {
    // ── forward ───────────────────────────────────────────────────────────────

    fn forward(&mut self, input: &[f32], cache: &mut dyn DynCache);

    fn forward_sample(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        self.forward(input, cache);
    }

    // ── backward ─────────────────────────────────────────────────────────────

    /// Backward for one timestep.
    ///
    /// `delta` (dL/d output) is modified in-place and the caller should discard it
    /// afterwards — use `cache.input_grad()` for the gradient flowing to the layer below.
    ///
    /// For LSTM layers `bptt_hidden_grad()` is updated as a side-effect.
    fn backward(&mut self, delta: &mut [f32], cache: &mut dyn DynCache);

    fn layer_tag(&self) -> u8;

    // ── serialisation ─────────────────────────────────────────────────────────

    /// Schreibt NUR die Gewichte.
    /// Shapes, act_id und dropout_rate stehen bereits im Architektur-Header
    /// (Sequential::save), hier kommen sie nicht nochmal vor.
    fn save(&self, w: &mut dyn io::Write) -> io::Result<()>;

    // ── header extras (für Sequential::save) ─────────────────────────────────

    /// Activation-ID für den Architektur-Header.
    /// Nur Dense (tag 1), IndRNN (tag 2) und Projection (tag 3) überschreiben das.
    fn activation_id(&self) -> Option<u8> {
        None
    }

    /// Dropout-Rate für den Architektur-Header.
    /// Nur DropoutLayer (tag 6) überschreibt das.
    fn dropout_rate(&self) -> Option<f32> {
        None
    }

    // ── cache ─────────────────────────────────────────────────────────────────

    fn make_cache(&self) -> Box<dyn DynCache>;

    // ── shapes ────────────────────────────────────────────────────────────────

    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;

    // ── gradient management ───────────────────────────────────────────────────

    fn apply_grads(&mut self, lr: f32);
    fn clear_grads(&mut self);
    fn scale_grads(&mut self, scale: f32);
    fn clip_grads(&mut self);

    // ── recurrent state ───────────────────────────────────────────────────────

    /// Clear h, c, dh_bptt, dc_bptt between sequences (no-op for Dense).
    fn reset_state(&mut self) {}

    /// dL/dh flowing from future timestep t+1 → t, or `None` for non-recurrent layers.
    ///
    /// `Sequential::backwards_sequence` reads this to combine with the dx coming
    /// from the layer above before passing `delta` down.
    fn bptt_hidden_grad(&mut self) -> Option<&[f32]> {
        None
    }

    // ← NEU: Wird von Sequential/Hierarchical nach jeder Sequenz aufgerufen
    fn accumulate_init_grad(&mut self) {}
}

// ── LayerBuilder ──────────────────────────────────────────────────────────────

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

    pub fn dense<A: Activate + 'static>(mut self, hidden: usize, act: A) -> Self {
        let layer = DenseLayer::new(self.output_size, hidden, act);
        self.layer(Box::new(layer), hidden);
        self
    }

    pub fn project<A: Activate + 'static>(mut self, hidden: usize, act: A) -> Self {
        let layer = Projection::new(self.output_size, hidden, act);
        self.layer(Box::new(layer), hidden);
        self
    }

    pub fn softmax(mut self) -> Self {
        let layer = SoftmaxLayer::new(self.output_size);
        self.layer(Box::new(layer), self.output_size);
        self
    }

    pub fn lstm(mut self, hidden: usize) -> Self {
        let layer = LSTMLayer::new(self.output_size, hidden);
        self.layer(Box::new(layer), hidden);
        self
    }

    pub fn indrnn<A: Activate + 'static>(mut self, hidden: usize, act: A) -> Self {
        let layer = IndRNNLayer::new(self.output_size, hidden, act);
        self.layer(Box::new(layer), hidden);
        self
    }

    pub fn dropout(mut self, rate: f32) -> Self {
        let layer = DropoutLayer::new(self.output_size, rate);
        self.layer(Box::new(layer), self.output_size);
        self
    }

    pub fn parallel<F: FnMut(Self) -> Self>(mut self, mut inside_layer: F) -> Self {
        self.in_parallel = true;
        let mut this = (inside_layer)(self);
        this.in_parallel = false;
        let layer = if let Some(branch1) = this.branch1.take()
            && let Some(branch2) = this.branch2.take()
        {
            ParallelLayer::new(branch1, branch2, 1.0, 1.0)
        } else {
            unreachable!()
        };
        this.output_size = layer.output_size();
        this.layers.push(Box::new(layer));
        this
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
            delta_buf: vec![0.0; max_size],
        }
    }
}
