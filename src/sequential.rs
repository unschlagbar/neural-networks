use iron_oxide::collections::Matrix;
use rand::random_range;

use crate::{
    lstm::{add_vec_in_place, one_hot},
    nn_layer::{DynCache, NnLayer},
    softmax::softmax,
};

// ── Sequential ────────────────────────────────────────────────────────────────

pub struct Sequential {
    pub input_size: usize,
    pub output_size: usize,
    pub layers: Vec<Box<dyn NnLayer>>,
    /// cache[t][l] — pre-allocated, never reallocated during training.
    pub cache: Vec<Vec<Box<dyn DynCache>>>,
    /// Reusable scratch buffer for the backward delta — no heap alloc per step.
    pub delta_buf: Vec<f32>,
}

impl Sequential {
    /// Pre-allocate caches for a fixed sequence length.
    /// Call once before the training loop (or when seq length changes).
    pub fn make_cache(&mut self, seq_len: usize) {
        self.cache = (0..seq_len)
            .map(|_| self.layers.iter().map(|l| l.make_cache()).collect())
            .collect();
    }

    // ── forward ───────────────────────────────────────────────────────────────

    pub fn forward_over(&mut self, input: &[u16]) {
        let Sequential {
            output_size,
            layers,
            cache,
            ..
        } = self;
        for t in 0..input.len() {
            let input = one_hot(input[t] as usize, *output_size);
            Self::forward_step(layers, &mut cache[t], &input);
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
        let n = self.layers.len();
        let out_layer = n - 1;

        let mut logits = Matrix::uninit(input.len(), self.output_size);

        let Sequential {
            output_size,
            layers,
            cache,
            ..
        } = self;

        for t in 0..input.len() {
            let input_vec = one_hot(input[t] as usize, *output_size);
            Self::forward_sample_step(&mut layers[..], &mut cache[0], &input_vec);
            logits[t].copy_from_slice(cache[0][out_layer].output());
        }
        logits
    }

    // ── backward ──────────────────────────────────────────────────────────────

    pub fn backwards_sequence(&mut self, _seq: &[u16], targets: &[u16]) {
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
            delta_buf[..out.len()].copy_from_slice(out);
            delta_buf[targets[t] as usize] -= 1.0;

            let mut delta_len = out.len();

            // ── layer loop top → bottom ─────────────────────────────────────
            for l in (0..n).rev() {
                // backward writes dL/d(input) into cache[t][l].input_grad().
                layers[l].backward(&mut delta_buf[..delta_len], cache[t][l].as_mut());

                if l == 0 {
                    break;
                }

                // Form delta for layer l-1:
                //   delta = input_grad(l) + bptt_hidden_grad(l-1)   [if recurrent]
                //         = input_grad(l)                            [otherwise]
                let dx = cache[t][l].input_grad();
                let new_len = dx.len();

                match layers[l - 1].bptt_hidden_grad() {
                    Some(bptt) => {
                        // bptt borrows from layers[l-1]; dx borrows from cache[t][l].
                        // delta_buf is a separate field — no conflict.
                        delta_buf[..new_len].copy_from_slice(bptt);
                        add_vec_in_place(&mut delta_buf[..new_len], dx);
                    }
                    None => {
                        delta_buf[..new_len].copy_from_slice(dx);
                    }
                }

                delta_len = new_len;
            }
        }

        for layer in &mut self.layers {
            // oder self.char_model.layers
            layer.accumulate_init_grad();
        }
    }

    pub fn backwards_sequence_with_layer_deltas(
        &mut self,
        _seq: &[u16],
        targets: &[u16],
        layer_idx: usize,
        extra_deltas: &[Option<Vec<f32>>],
    ) {
        let n = self.layers.len();
        assert_eq!(extra_deltas.len(), targets.len());

        let Sequential {
            layers,
            cache,
            delta_buf,
            ..
        } = self;

        for t in (0..targets.len()).rev() {
            let out = cache[t].last().unwrap().output();
            delta_buf[..out.len()].copy_from_slice(out);
            delta_buf[targets[t] as usize] -= 1.0;

            let mut delta_len = out.len();

            // ── layer loop top → bottom ─────────────────────────────────────
            for l in (0..n).rev() {
                if l == layer_idx {
                    if let Some(extra) = &extra_deltas[t] {
                        debug_assert_eq!(extra.len(), delta_len);
                        add_vec_in_place(&mut delta_buf[..delta_len], extra);
                    }
                }

                layers[l].backward(&mut delta_buf[..delta_len], cache[t][l].as_mut());

                if l == 0 {
                    break;
                }

                let dx = cache[t][l].input_grad();
                let new_len = dx.len();

                match layers[l - 1].bptt_hidden_grad() {
                    Some(bptt) => {
                        delta_buf[..new_len].copy_from_slice(bptt);
                        add_vec_in_place(&mut delta_buf[..new_len], dx);
                    }
                    None => {
                        delta_buf[..new_len].copy_from_slice(dx);
                    }
                }

                delta_len = new_len;
            }
        }
        for layer in &mut self.layers {
            // oder self.char_model.layers
            layer.accumulate_init_grad();
        }
    }

    pub fn backward_from_layer(
        layers: &mut [Box<dyn NnLayer>],
        layer_idx: usize,
        mut delta: Vec<f32>,
        cache_t: &mut [Box<dyn DynCache>],
    ) {
        let n = layers.len();
        assert!(layer_idx < n, "backward_from_layer: invalid layer index");

        let mut delta_len = delta.len();

        for l in (0..=layer_idx).rev() {
            layers[l].backward(&mut delta[..delta_len], cache_t[l].as_mut());

            if l == 0 {
                break;
            }

            let dx = cache_t[l].input_grad();
            let new_len = dx.len();

            match layers[l - 1].bptt_hidden_grad() {
                Some(bptt) => {
                    delta.resize(new_len, 0.0);
                    delta.copy_from_slice(bptt);
                    add_vec_in_place(&mut delta[..new_len], dx);
                }
                None => {
                    delta.resize(new_len, 0.0);
                    delta.copy_from_slice(dx);
                }
            }
            delta_len = new_len;
        }
    }

    // ── training ──────────────────────────────────────────────────────────────

    pub fn train<'a, I: Iterator<Item = (&'a [u16], &'a [u16])>>(
        &mut self,
        data: I,
        lr: f32,
        iteration: &mut usize,
        j: &mut usize,
        batch_size: usize,
    ) {
        let mut total_loss = 0.0;
        let mut steps = 0;

        for (inputs, targets) in data {
            for layer in &mut self.layers {
                layer.reset_state();
            }
            self.forward_over(inputs);
            total_loss += self.seq_loss(targets);
            steps += 1;

            self.backwards_sequence(inputs, targets);

            *iteration += 1;
            if *iteration % batch_size == 0 {
                self.sgd_step(lr / batch_size as f32);
                *iteration = 0;
                *j += 1;
            }
        }

        println!("{j} Average loss = {:.4}", total_loss / steps.max(1) as f32);
    }

    // ── sampling ──────────────────────────────────────────────────────────────

    pub fn sample(
        &mut self,
        prefix: &[u16],
        max_len: usize,
        temperature: f32,
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
            let logits = self.forward_sample(&[last_token]);
            let p = &logits[0];
            let cols = logits.cols();

            // Temperature scaling + softmax for sampling — this path runs outside
            // the training hot loop, so the two small Vec allocs here are fine.
            let scaled: Vec<f32> = p.iter().map(|&v| v / temperature.max(1e-8)).collect();
            let q = softmax(&scaled);

            let mut idx: Vec<usize> = (0..cols).collect();
            idx.sort_unstable_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap());

            let r = random_range(0.0..1.0);
            let mut cum = 0.0;
            let mut next = 0;
            for i in idx {
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

    // ── gradient helpers ──────────────────────────────────────────────────────

    fn sgd_step(&mut self, lr: f32) {
        for layer in &mut self.layers {
            layer.clip_grads();
            layer.apply_grads(lr);
        }
        self.scale_grads(0.0);
    }

    pub fn clear_grads(&mut self) {
        for layer in &mut self.layers {
            layer.clear_grads();
        }
    }

    pub fn scale_grads(&mut self, scale: f32) {
        for layer in &mut self.layers {
            layer.scale_grads(scale);
        }
    }

    pub fn seq_loss(&self, targets: &[u16]) -> f32 {
        let last = self.layers.len() - 1;
        let mut l = 0.0;
        for t in 0..targets.len() {
            let p = self.cache[t][last].output()[targets[t] as usize] + 1e-12;
            l -= p.ln();
        }
        l / targets.len() as f32
    }
}
