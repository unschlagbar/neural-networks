//! A flat layer stack, run one timestep at a time: `SequentialBuilder` assembles
//! it from the layers in `layers`, every layer owns its weights and gradients,
//! and `io` is its file format.

use iron_oxide::collections::Matrix;
use rand::random_range;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

pub mod io;
pub mod layer;
pub mod layers;
pub mod optim;

pub use layer::{DynCache, NnLayer, SequentialBuilder};
pub use layers::lstm::sigmoid;

use layers::softmax::{sample_top_p, softmax, softmax_inplace};

pub struct Sequential {
    pub input_size: usize,
    pub output_size: usize,
    pub layers: Vec<Box<dyn NnLayer>>,
    /// cache[t][l] — pre-allocated, never reallocated during training.
    pub cache: Vec<Vec<Box<dyn DynCache>>>,
    /// Reusable scratch buffer for the backward delta — no heap alloc per step.
    pub delta_buf: Box<[f32]>,
    /// Reusable one-hot input buffer — avoids per-timestep heap allocation.
    pub input_buf: Box<[f32]>,
}

impl Sequential {
    /// Assemble a stack from already-built layers (input/output/scratch derived
    /// from them, no forward cache). Mirrors what `load_stack` produces, so the
    /// result serializes identically via `write_stack`.
    pub fn from_layers(layers: Vec<Box<dyn NnLayer>>) -> Self {
        let input_size = layers.first().map(|l| l.input_size()).unwrap_or(0);
        let output_size = layers.last().map(|l| l.output_size()).unwrap_or(0);
        let max_size = layers.iter().map(|l| l.output_size()).max().unwrap_or(0);
        Sequential {
            input_size,
            output_size,
            layers,
            cache: Vec::new(),
            delta_buf: vec![0.0; max_size].into(),
            input_buf: vec![0.0; input_size].into(),
        }
    }

    /// Pre-allocate caches for a fixed sequence length.
    /// Call once before the training loop (or when seq length changes).
    pub fn make_cache(&mut self, seq_len: usize) {
        self.cache = (0..seq_len)
            .map(|_| self.layers.iter().map(|l| l.make_cache()).collect())
            .collect();
    }

    /// Largest input/output dimension across the stack — sizes the BPTT delta
    /// scratch a backward pass needs.
    pub fn max_layer_dim(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.input_size().max(l.output_size()))
            .max()
            .unwrap_or(0)
    }

    pub fn forward_over(&mut self, input: &[u16]) {
        let Sequential {
            layers,
            cache,
            input_buf,
            ..
        } = self;
        for t in 0..input.len() {
            let tok = input[t] as usize;
            input_buf[tok] = 1.0;
            Self::forward_step(layers, &mut cache[t], input_buf);
            input_buf[tok] = 0.0;
        }
    }

    pub fn forward_step(
        layers: &mut [Box<dyn NnLayer>],
        caches_t: &mut [Box<dyn DynCache>],
        input: &[f32],
    ) {
        for l in 0..layers.len() {
            let (left, right) = caches_t.split_at_mut(l);
            let inp: &[f32] = if l == 0 { input } else { left[l - 1].output() };
            layers[l].forward(inp, right[0].as_mut());
        }
    }

    fn forward_sample_step(
        layers: &mut [Box<dyn NnLayer>],
        caches_t: &mut [Box<dyn DynCache>],
        input: &[f32],
    ) {
        for l in 0..layers.len() {
            let (left, right) = caches_t.split_at_mut(l);
            let inp: &[f32] = if l == 0 { input } else { left[l - 1].output() };
            layers[l].forward_sample(inp, right[0].as_mut());
        }
    }

    /// Forward returning raw logits (softmax layer is skipped so temperature scaling works).
    pub fn forward_sample(&mut self, input: &[u16]) -> Matrix {
        let out_layer = self.layers.len() - 1;
        let mut logits = Matrix::uninit(input.len(), self.output_size);
        for t in 0..input.len() {
            let tok = input[t] as usize;
            self.input_buf[tok] = 1.0;
            let Sequential {
                layers,
                cache,
                input_buf,
                ..
            } = self;
            Self::forward_sample_step(layers, &mut cache[0], input_buf);
            logits[t].copy_from_slice(self.cache[0][out_layer].output());
            self.input_buf[tok] = 0.0;
        }
        logits
    }

    pub fn forward(&mut self, input: u16) -> &[f32] {
        let out_layer = self.layers.len() - 1;
        let tok = input as usize;
        self.input_buf[tok] = 1.0;
        let Sequential {
            layers,
            cache,
            input_buf,
            ..
        } = self;
        Self::forward_sample_step(layers, &mut cache[0], input_buf);
        self.input_buf[tok] = 0.0;
        self.cache[0][out_layer].output()
    }

    /// Layer-major BPTT driver shared by every backward variant.
    fn backwards_driver(
        &mut self,
        n_steps: usize,
        mut top_delta: impl FnMut(usize, &[f32], &mut [f32]) -> usize,
    ) {
        let n = self.layers.len();
        let last = n - 1;

        let Sequential {
            layers,
            cache,
            delta_buf,
            ..
        } = self;

        for l in (0..n).rev() {
            for t in (0..n_steps).rev() {
                let delta_len = if l == last {
                    // Top layer: loss-specific gradient w.r.t. the logits.
                    top_delta(t, cache[t][last].output(), delta_buf)
                } else {
                    // Spatial gradient handed down from the layer above at t.
                    let dx = cache[t][l + 1].input_grad();
                    let len = dx.len();
                    delta_buf[..len].copy_from_slice(dx);
                    len
                };

                // backward writes dL/d(input) into cache[t][l].input_grad().
                layers[l].backward(&mut delta_buf[..delta_len], cache[t][l].as_mut());
            }
        }
    }

    pub fn backwards_sequence(&mut self, targets: &[u16]) {
        self.backwards_driver(targets.len(), |t, logits, delta_buf| {
            // Gradient of cross-entropy under softmax: ŷ − y.
            let len = logits.len();
            delta_buf[..len].copy_from_slice(logits);
            softmax_inplace(&mut delta_buf[..len]);
            delta_buf[targets[t] as usize] -= 1.0;
            len
        });
    }

    pub fn sample(
        &mut self,
        prefix: &[u16],
        max_len: usize,
        temperature: f32,
        top_p: f32,
        mut callback: impl FnMut(u16) -> bool,
    ) -> Vec<u16> {
        for layer in &mut self.layers {
            layer.reset_state();
        }

        let mut last_token = if prefix.is_empty() {
            random_range(0..self.input_size) as u16
        } else {
            let _ = self.forward_sample(&prefix[..prefix.len() - 1]);
            prefix[prefix.len() - 1]
        };

        let mut out = Vec::with_capacity(max_len);

        for _ in 0..max_len {
            let logits = self.forward(last_token);
            let next = sample_top_p(logits, temperature, top_p) as u16;

            out.push(next);
            if !callback(next) {
                break;
            }
            last_token = next;
        }
        out
    }

    /// Single inference step with a raw f32 input (no embedding lookup).
    /// Requires `make_cache(1)` to have been called first.
    pub fn forward_raw(&mut self, input: &[f32]) -> &[f32] {
        let out_layer = self.layers.len() - 1;
        let Sequential { layers, cache, .. } = self;
        Self::forward_sample_step(layers, &mut cache[0], input);
        self.cache[0][out_layer].output()
    }

    /// Forward pass over a sequence of raw f32 frames (training mode, dropout active).
    /// Returns the scalar output of the last layer at the last timestep.
    /// The cache is grown if it is too small; reset state before calling.
    pub fn forward_raw_seq<F: AsRef<[f32]>>(&mut self, features: &[F]) {
        let n = features.len();
        if self.cache.len() < n {
            self.make_cache(n);
        }
        let Sequential { layers, cache, .. } = self;
        for (t, frame) in features.iter().enumerate() {
            Self::forward_step(layers, &mut cache[t], frame.as_ref());
        }
    }

    /// Backward for per-step weighted BCE on a single-logit head:
    /// `delta[t] = weights[t] · (σ(z_t) − labels[t])`.
    pub fn backwards_bce(&mut self, labels: &[f32], weights: &[f32]) {
        assert_eq!(labels.len(), weights.len(), "backwards_bce: one weight per label");
        self.backwards_driver(labels.len(), |t, logits, delta_buf| {
            delta_buf[0] = weights[t] * (sigmoid(logits[0]) - labels[t]);
            1
        });
    }

    /// Optimizer step for every layer. `weight_decay` is the decoupled decay
    /// (λ); `0.0` is plain Adam.
    pub fn step(&mut self, lr: f32, weight_decay: f32) {
        self.layers.par_iter_mut().for_each(|layer| {
            layer.apply_grads(lr, weight_decay);
            layer.clear_grads();
        });
    }

    pub fn clear_grads(&mut self) {
        for layer in &mut self.layers {
            layer.clear_grads();
        }
    }

    pub fn seq_loss(&self, targets: &[u16]) -> f32 {
        let last = self.layers.len() - 1;
        let mut l = 0.0;
        for (t, target) in targets.iter().enumerate() {
            let probs = softmax(self.cache[t][last].output());
            let p = probs[*target as usize] + 1e-12;
            l -= p.ln();
        }
        l / targets.len() as f32
    }

    pub fn output(&self, t: usize) -> &[f32] {
        let last_layer = self.layers.len() - 1;
        self.cache[t][last_layer].output()
    }
}
