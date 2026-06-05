use std::time::Instant;

use iron_oxide::collections::Matrix;
use rand::random_range;

use crate::{
    config::SEQ_LOC,
    nn::{
        lstm::sigmoid,
        softmax::{softmax, softmax_inplace},
    },
    nn_layer::{DynCache, NnLayer},
    training::TrainingState,
    wake_word::training::Sequence,
};

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
    /// Pre-allocate caches for a fixed sequence length.
    /// Call once before the training loop (or when seq length changes).
    pub fn make_cache(&mut self, seq_len: usize) {
        self.cache = (0..seq_len)
            .map(|_| self.layers.iter().map(|l| l.make_cache()).collect())
            .collect();
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

    pub fn backwards_sequence(&mut self, targets: &[u16]) {
        let n = self.layers.len();

        // Destructure so we can mutably borrow delta_buf independently of
        // layers and cache. This is the key trick that avoids to_vec():
        // all three fields are disjoint, the borrow checker accepts it.
        let Sequential {
            layers,
            cache,
            delta_buf,
            ..
        } = self;

        for t in (0..targets.len()).rev() {
            let out = cache[t].last().unwrap().output();
            let mut delta_len = out.len();
            delta_buf[..delta_len].copy_from_slice(out);
            softmax_inplace(&mut delta_buf[..delta_len]);
            delta_buf[targets[t] as usize] -= 1.0;

            for l in (0..n).rev() {
                // backward writes dL/d(input) into cache[t][l].input_grad().
                layers[l].backward(&mut delta_buf[..delta_len], cache[t][l].as_mut());

                if l == 0 {
                    break;
                }

                let dx = cache[t][l].input_grad();
                let new_len = dx.len();

                delta_buf[..new_len].copy_from_slice(dx);

                delta_len = new_len;
            }
        }
    }

    pub fn train<'a, I: Iterator<Item = (&'a [u16], &'a [u16])>>(
        &mut self,
        data: I,
        state: &mut TrainingState,
    ) {
        let mut tokens = 0;
        let mut time = Instant::now();

        for (inputs, targets) in data {
            for layer in &mut self.layers {
                layer.reset_state();
            }
            self.forward_over(inputs);
            let loss = self.seq_loss(targets);
            tokens += inputs.len();

            self.backwards_sequence(targets);
            for layer in &mut self.layers {
                layer.accumulate_init_grad();
                layer.reset_bptt_state();
            }
            state.log_tokens(inputs.len());

            if let Some(lr) = state.step(loss) {
                self.sgd_step(lr);
            }

            if state.save() {
                match self.save(SEQ_LOC) {
                    Ok(()) => println!("saved"),
                    Err(e) => eprintln!("save failed: {e}"),
                }
            }

            if state.print() {
                let loss = state.get_loss();
                let elapsed = time.elapsed();
                println!(
                    "{} | loss {:.4} | ppl {:.4} | {} tok | {:.1?}",
                    state.step,
                    loss,
                    loss.exp(),
                    tokens,
                    elapsed,
                );
                tokens = 0;
                time = Instant::now();
            }
        }
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

            let scaled: Vec<f32> = logits.iter().map(|&v| v / temperature.max(1e-8)).collect();
            let q = softmax(&scaled);

            let mut idx: Vec<usize> = (0..q.len()).collect();
            idx.sort_unstable_by(|&a, &b| q[b].partial_cmp(&q[a]).unwrap());

            let mut cum = 0.0;
            let candidates: Vec<usize> = idx
                .iter()
                .copied()
                .take_while(|&i| {
                    if cum >= top_p {
                        return false;
                    }
                    cum += q[i];
                    true
                })
                .collect();

            let total: f32 = candidates.iter().map(|&i| q[i]).sum();
            let r = random_range(0.0..total);
            let mut cum = 0.0;
            let mut next = candidates[0] as u16;
            for &i in &candidates {
                cum += q[i];
                if cum >= r {
                    next = i as u16;
                    break;
                }
            }

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

    /// Backward for binary BCE classification with dense per-frame loss.
    /// Intermediate frames use label=0; only the final frame uses the sequence label.
    /// All frames contribute gradients (BPTT through the full sequence).
    pub fn backwards_wake_bce(&mut self, seq: &Sequence, weight_pos: f32, weight_neg: f32) {
        let n_frames = seq.frames.len();
        let n_layers = self.layers.len();
        let last = n_layers - 1;

        let Sequential {
            layers,
            cache,
            delta_buf,
            ..
        } = self;
        for t in (0..n_frames).rev() {
            let label = seq.frames[t].label;
            let weight = if label > 0.5 { weight_pos } else { weight_neg };

            let logit = cache[t][last].output()[0];
            delta_buf[0] = weight * (sigmoid(logit) - label);

            let mut delta_len = 1;
            for l in (0..n_layers).rev() {
                layers[l].backward(&mut delta_buf[..delta_len], cache[t][l].as_mut());
                if l == 0 {
                    break;
                }
                let dx = cache[t][l].input_grad();
                let new_len = dx.len();
                delta_buf[..new_len].copy_from_slice(dx);
                delta_len = new_len;
            }
        }
    }

    pub fn sgd_step(&mut self, lr: f32) {
        for layer in &mut self.layers {
            layer.apply_grads(lr);
        }
        self.clear_grads();
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
