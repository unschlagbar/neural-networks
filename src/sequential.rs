use std::time::Instant;

use iron_oxide::collections::Matrix;
use rand::random_range;

use crate::{
    config::{SAVE_EVERY, SEQ_LOC},
    nn::{one_hot, softmax::softmax},
    nn_layer::{DynCache, NnLayer},
};

// ── Sequential ────────────────────────────────────────────────────────────────

pub struct Sequential {
    pub input_size: usize,
    pub output_size: usize,
    pub layers: Vec<Box<dyn NnLayer>>,
    /// cache[t][l] — pre-allocated, never reallocated during training.
    pub cache: Vec<Vec<Box<dyn DynCache>>>,
    /// Reusable scratch buffer for the backward delta — no heap alloc per step.
    pub delta_buf: Box<[f32]>,
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

    pub fn forward(&mut self, input: u16) -> &[f32] {
        let n = self.layers.len();
        let out_layer = n - 1;

        let Sequential {
            output_size,
            layers,
            cache,
            ..
        } = self;

        let input_vec = one_hot(input as usize, *output_size);
        Self::forward_sample_step(&mut layers[..], &mut cache[0], &input_vec);
        cache[0][out_layer].output()
    }

    // ── backward ──────────────────────────────────────────────────────────────

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

                let dx = cache[t][l].input_grad();
                let new_len = dx.len();

                delta_buf[..new_len].copy_from_slice(dx);

                delta_len = new_len;
            }
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
        print_every: usize,
        step: &mut usize,
    ) {
        let mut window_loss = 0.0;
        let mut window_steps = 0;
        let mut window_tokens = 0;
        let mut window_start = Instant::now();

        for (inputs, targets) in data {
            for layer in &mut self.layers {
                layer.reset_state();
            }
            self.forward_over(inputs);
            let loss = self.seq_loss(targets);
            window_loss += loss;
            window_steps += 1;
            window_tokens += inputs.len();
            *step += 1;

            self.backwards_sequence(targets);
            assert_eq!(inputs[1], targets[0]);
            for layer in &mut self.layers {
                layer.accumulate_init_grad();
                layer.zero_bptt_state();
            }

            *iteration += 1;
            if *iteration % batch_size == 0 {
                self.sgd_step(lr / batch_size as f32);
                *iteration = 0;
                *j += 1;
            }

            if *step % SAVE_EVERY == 0 {
                match self.save(SEQ_LOC) {
                    Ok(()) => println!("saved"),
                    Err(e) => eprintln!("save failed: {e}"),
                }
            }

            if print_every > 0 && window_steps >= print_every {
                let avg = window_loss / window_steps as f32;
                let elapsed = window_start.elapsed();
                println!(
                    "{} | loss {:.4} | ppl {:.4} | {} tok | {:.1?}",
                    *step,
                    avg,
                    avg.exp(),
                    window_tokens,
                    elapsed,
                );
                window_loss = 0.0;
                window_steps = 0;
                window_tokens = 0;
                window_start = Instant::now();
            }
        }
    }

    // ── sampling ──────────────────────────────────────────────────────────────

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

    // ── gradient helpers ──────────────────────────────────────────────────────

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
