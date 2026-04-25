// hierarchical_sequential.rs
//
// Fixes vs original:
//   1. sample() — forward_step_char used `forward` (applies softmax), then called
//      softmax(x/T) again → double-softmax.  Now uses `forward_sample_step_char`
//      which calls `layer.forward_sample()` so SoftmaxLayer passes raw logits.
//
//   2. backwards_sequence() — BPTT between word boundaries was dropped.
//      `dh_bptt` from the high model's first LSTM was never fed back into the
//      delta for the preceding word.  Now captured and added each iteration.
//
//   3. forward_over() / sample() — one_hot() + full_input Vec were re-allocated
//      every timestep.  Replaced with a pre-allocated scratch buffer.
//
//   4. save() / load() — new: model can now be persisted and reloaded.

use std::{
    fs::File,
    io::{self, Cursor, Read, Write},
    u32,
};

use crate::{
    loading::read_u16,
    loading::read_u32,
    nn_layer::NnLayer,
    saving::{HIER_MAGIC, write_u16, write_u32},
    sequential::Sequential,
    softmax::softmax,
};

pub struct HierarchicalSequential {
    pub char_model: Sequential,
    pub char2_model: Sequential,
    pub high_model: Sequential,

    pub vocab_size: usize,
    pub context_size: usize,

    pub boundary_token_ids: Vec<u16>,

    word_context: Box<[f32]>,
    boundary_timesteps: Vec<usize>,

    char_input_buf: Vec<f32>,

    pub last_high_grad_signal: f32,
}

impl HierarchicalSequential {
    /// `boundary_token_ids` — from `tokenizer.boundary_token_ids()`.
    pub fn new(
        char_model: Sequential,
        char2_model: Sequential,
        high_model: Sequential,
        vocab_size: usize,
        boundary_token_ids: Vec<u16>,
    ) -> Self {
        let context_size = high_model.output_size;
        assert_eq!(
            char_model.input_size, vocab_size,
            "char_model.input_size must equal vocab_size"
        );

        assert_eq!(
            char2_model.output_size, vocab_size,
            "char2_model.output_size must equal vocab_size"
        );

        assert_eq!(char_model.output_size, high_model.input_size);

        assert_eq!(
            char2_model.input_size,
            high_model.output_size + char_model.output_size
        );

        let char_input_buf = vec![0.0; vocab_size];
        let word_context = vec![0.0; char_model.output_size + context_size];

        Self {
            char_model,
            char2_model,
            high_model,
            vocab_size,
            context_size,
            boundary_token_ids,
            word_context: word_context.into(),
            boundary_timesteps: vec![],
            char_input_buf,
            last_high_grad_signal: 0.0,
        }
    }

    // ── Cache allocation ──────────────────────────────────────────────────────

    pub fn make_cache(&mut self, seq_len: usize) {
        self.char_model.make_cache(seq_len);
        self.char2_model.make_cache(seq_len);
        self.high_model.make_cache(seq_len);
    }

    // ── State reset ───────────────────────────────────────────────────────────

    pub fn reset(&mut self) {
        for layer in &mut self.char_model.layers {
            layer.reset_state();
        }
        for layer in &mut self.char2_model.layers {
            layer.reset_state();
        }
        for layer in &mut self.high_model.layers {
            layer.reset_state();
        }
        self.boundary_timesteps.clear();
    }

    // ── Private forward helpers ───────────────────────────────────────────────

    fn forward_step(
        layers: &mut [Box<dyn NnLayer>],
        caches_t: &mut [Box<dyn crate::nn_layer::DynCache>],
        input: &[f32],
    ) {
        for l in 0..layers.len() {
            let (left, right) = caches_t.split_at_mut(l);
            let inp = if l == 0 { input } else { left[l - 1].output() };
            layers[l].forward(inp, right[0].as_mut());
        }
    }

    /// Like `forward_step_char` but calls `forward_sample` so that the
    /// SoftmaxLayer passes through raw logits instead of applying softmax.
    fn forward_sample_step(
        layers: &mut [Box<dyn NnLayer>],
        caches_t: &mut [Box<dyn crate::nn_layer::DynCache>],
        input: &[f32],
    ) {
        for l in 0..layers.len() {
            let (left, right) = caches_t.split_at_mut(l);
            let inp = if l == 0 { input } else { left[l - 1].output() };
            layers[l].forward_sample(inp, right[0].as_mut());
        }
    }

    // ── Build the combined [one_hot | context] input in the scratch buffer ────

    #[inline]
    fn fill_input_buf(&mut self, token: u16) {
        self.char_input_buf[..self.vocab_size].fill(0.0);
        self.char_input_buf[token as usize] = 1.0;
    }

    fn assert_no_nan(&self, slice: &[f32], name: &str, t: usize) {
        if let Some(pos) = slice.iter().position(|x| !x.is_finite()) {
            panic!("NaN/Inf detected in {name} at t={t}, idx={pos}");
        }
    }

    // ── Forward (training) ───────────────────────────────────────────────────

    pub fn forward_over(&mut self, input: &[u16]) {
        self.word_context.fill(0.0);
        let char_output = self.char_model.output_size;
        for t in 0..input.len() {
            let token = input[t];
            let next_token = input.get(t + 1);

            self.fill_input_buf(token);

            Self::forward_step(
                &mut self.char_model.layers,
                &mut self.char_model.cache[t],
                &self.char_input_buf,
            );

            let char1_output = self.char_model.cache[t].last().unwrap().output();
            self.word_context[..char_output].copy_from_slice(char1_output);

            Self::forward_step(
                &mut self.char2_model.layers,
                &mut self.char2_model.cache[t],
                &self.word_context,
            );

            self.assert_no_nan(
                self.char2_model.cache[t].last().unwrap().output(),
                "char_output",
                t,
            );

            let is_boundary = next_token.map_or(false, |v| self.boundary_token_ids.contains(v));

            if is_boundary {
                self.boundary_timesteps.push(t);
                let word_idx = self.boundary_timesteps.len() - 1;

                let word_rep = self.char_model.cache[t].last().unwrap().output();

                Self::forward_step(
                    &mut self.high_model.layers,
                    &mut self.high_model.cache[word_idx],
                    word_rep,
                );

                for layer in &mut self.char_model.layers {
                    layer.reset_state();
                }
                for layer in &mut self.char2_model.layers {
                    layer.reset_state();
                }

                let high_out = self.high_model.cache[word_idx].last().unwrap().output();
                self.assert_no_nan(high_out, "high_output", word_idx);
                self.word_context[char_output..].copy_from_slice(high_out);
            }
        }
    }

    // ── Backward pass (BPTT) ─────────────────────────────────────────────────
    //
    // Forward graph at timestep t:
    //
    //   char1[t]   = char_model( one_hot(inputs[t]) )
    //   char2[t]   = char2_model( [char1[t] | high_ctx_PREV] )   ← PREV = last boundary's output
    //   loss[t]    = CE( char2[t], targets[t] )
    //
    //   if inputs[t+1] is boundary:          ← t is the LAST char of the current word
    //     high[w]  = high_model( char1[t] )  ← produces high_ctx_NEW
    //     char_model.reset(), char2_model.reset()
    //     high_ctx = high_ctx_NEW            ← takes effect from t+1 onwards
    //
    // CRITICAL invariant:
    //   char2[t_w] uses high_ctx_PREV  (the context from BEFORE the boundary)
    //   high_model[w] produces high_ctx_NEW (used by chars AFTER t_w)
    //
    // Therefore dL/d(high_ctx_NEW)  = sum of dx2[char_output..] for t in (t_w, t_{w+1}]
    //                                  i.e. all chars STRICTLY AFTER the boundary t_w
    //           dL/d(high_ctx_PREV) = sum of dx2[char_output..] for t in (t_{w-1}, t_w]
    //                                  i.e. chars in the current word INCLUDING t_w itself
    //
    // Backward order at boundary t_w:
    //   A. Backprop high_model[w] with d_high_ctx_NEW (accumulated so far, chars after t_w)
    //   B. Reset d_high_ctx → start accumulating dL/d(high_ctx_PREV)
    //   C. Zero char BPTT — reset in forward means no gradient crosses this boundary
    //   D. Backprop char2[t_w] — its dx2[char_output..] feeds into d_high_ctx_PREV
    //   E. d_char1[t_w] = dx2[..char_output] + high_model.input_grad
    //   F. Backprop char_model[t_w] with d_char1[t_w]

    pub fn backwards_sequence(&mut self, _inputs: &[u16], targets: &[u16]) {
        let t_len = targets.len();
        let char_output = self.char_model.output_size;
        let context_size = self.context_size;

        // Zero all BPTT state before starting — same as Sequential::train does.
        // This prevents stale gradients from a previous sequence leaking in.
        for layer in &mut self.char_model.layers {
            layer.zero_bptt_state();
        }
        for layer in &mut self.char2_model.layers {
            layer.zero_bptt_state();
        }
        for layer in &mut self.high_model.layers {
            layer.zero_bptt_state();
        }

        let max_delta = self
            .char2_model
            .output_size // CE start delta
            .max(char_output + context_size) // char2 input grad
            .max(self.high_model.output_size) // high start delta
            .max(char_output); // char_model input grad

        let mut delta_buf = vec![0.0f32; max_delta];

        // d_high_ctx accumulates dL/d(high_ctx) for the word whose chars we are
        // currently processing backward.  Reset each time we fire the high model.
        let mut d_high_ctx = vec![0.0f32; context_size];

        let n_boundaries = self.boundary_timesteps.len();
        let mut boundary_ptr: isize = n_boundaries as isize - 1;

        let mut high_grad_sq_sum = 0.0f32;
        let mut high_grad_count = 0usize;

        for t in (0..t_len).rev() {
            let is_boundary =
                boundary_ptr >= 0 && self.boundary_timesteps[boundary_ptr as usize] == t;

            if is_boundary {
                let word_idx = boundary_ptr as usize;
                boundary_ptr -= 1;

                // ── A: backprop high_model[word_idx] ─────────────────────────
                // d_high_ctx = dL/d(high_ctx_NEW) accumulated from chars AFTER t.
                // high_model[word_idx] is exactly the function that produced high_ctx_NEW.
                high_grad_sq_sum += d_high_ctx.iter().map(|x| x * x).sum::<f32>();
                high_grad_count += 1;

                let high_out_len = self.high_model.output_size;
                delta_buf[..high_out_len].copy_from_slice(&d_high_ctx);
                backward_through_layers(
                    &mut self.high_model.layers,
                    &mut self.high_model.cache[word_idx],
                    &mut delta_buf,
                    high_out_len,
                );
                // Save high model's gradient on its input (= char1[t]).
                let dx_high = self.high_model.cache[word_idx][0].input_grad().to_vec();

                // ── B: reset d_high_ctx for the PREVIOUS word ─────────────────
                d_high_ctx.fill(0.0);

                // ── C: zero char BPTT across this boundary ────────────────────
                // The forward reset means no recurrent gradient may cross t_w.
                // We zero BEFORE backpropping char2/char_model at t_w so that the
                // bptt from the next-word's backward doesn't contaminate word w's backward.
                for layer in &mut self.char_model.layers {
                    layer.zero_bptt_state();
                }
                for layer in &mut self.char2_model.layers {
                    layer.zero_bptt_state();
                }

                // ── D: backprop char2_model[t] ────────────────────────────────
                // char2[t_w] ran with high_ctx_PREV, so its dx2[char_output..]
                // belongs to d_high_ctx_PREV — the one we just reset.
                let out2 = self.char2_model.cache[t].last().unwrap().output();
                let out2_len = out2.len();
                delta_buf[..out2_len].copy_from_slice(out2);
                delta_buf[targets[t] as usize] -= 1.0;
                backward_through_layers(
                    &mut self.char2_model.layers,
                    &mut self.char2_model.cache[t],
                    &mut delta_buf,
                    out2_len,
                );
                let dx2 = self.char2_model.cache[t][0].input_grad().to_vec();

                // ── E: split dx2 and build d_char1[t] ────────────────────────
                let mut d_char1 = dx2[..char_output].to_vec();
                for i in 0..context_size {
                    d_high_ctx[i] += dx2[char_output + i];
                }
                // high_model also received char1[t] as its input.
                for i in 0..dx_high.len().min(d_char1.len()) {
                    d_char1[i] += dx_high[i];
                }

                // ── F: backprop char_model[t] ─────────────────────────────────
                let d_len = d_char1.len();
                delta_buf[..d_len].copy_from_slice(&d_char1);
                backward_through_layers(
                    &mut self.char_model.layers,
                    &mut self.char_model.cache[t],
                    &mut delta_buf,
                    d_len,
                );
            } else {
                // ── Non-boundary: standard char step ─────────────────────────

                // Backprop char2_model[t]
                let out2 = self.char2_model.cache[t].last().unwrap().output();
                let out2_len = out2.len();
                delta_buf[..out2_len].copy_from_slice(out2);
                delta_buf[targets[t] as usize] -= 1.0;
                backward_through_layers(
                    &mut self.char2_model.layers,
                    &mut self.char2_model.cache[t],
                    &mut delta_buf,
                    out2_len,
                );
                let dx2 = self.char2_model.cache[t][0].input_grad().to_vec();

                // Split: char1 gradient and high_ctx gradient
                let d_char1 = &dx2[..char_output];
                for i in 0..context_size {
                    d_high_ctx[i] += dx2[char_output + i];
                }

                // Backprop char_model[t]
                let d_len = d_char1.len();
                delta_buf[..d_len].copy_from_slice(d_char1);
                backward_through_layers(
                    &mut self.char_model.layers,
                    &mut self.char_model.cache[t],
                    &mut delta_buf,
                    d_len,
                );
            }
        }

        for layer in &mut self.char_model.layers {
            layer.accumulate_init_grad();
        }
        for layer in &mut self.char2_model.layers {
            layer.accumulate_init_grad();
        }
        for layer in &mut self.high_model.layers {
            layer.accumulate_init_grad();
        }

        self.last_high_grad_signal = if high_grad_count > 0 {
            (high_grad_sq_sum / high_grad_count as f32).sqrt()
        } else {
            0.0
        };
    }
    // ── Training ─────────────────────────────────────────────────────────────

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
            self.reset();
            self.forward_over(inputs);
            total_loss += self.char2_model.seq_loss(targets);
            steps += 1;

            self.backwards_sequence(inputs, targets);

            *iteration += 1;
            if *iteration % batch_size == 0 {
                let char_lr = self.boundary_timesteps.len() as f32 / inputs.len() as f32 * lr;
                //let char_lr = lr;

                for layer in &mut self.char_model.layers {
                    layer.apply_grads(char_lr);
                }
                self.char_model.clear_grads();

                for layer in &mut self.char2_model.layers {
                    layer.apply_grads(char_lr);
                }
                self.char2_model.clear_grads();

                for layer in &mut self.high_model.layers {
                    layer.apply_grads(lr);
                }
                self.high_model.clear_grads();

                *iteration = 0;
                *j += 1;
            }
        }

        println!(
            "{j} | char loss = {:.4} | high ∇ = {:.4} | ppl = {:.4}",
            total_loss / steps.max(1) as f32,
            self.last_high_grad_signal,
            (total_loss / steps.max(1) as f32).exp()
        );
    }

    pub fn sample(
        &mut self,
        prefix: &[u16],
        max_len: usize,
        temperature: f32,
        mut callback: impl FnMut(u16) -> bool,
    ) -> Vec<u16> {
        self.reset();

        // Feed the prefix through the model to warm up all states.
        // The last token of the prefix becomes the first "last_token".
        let mut last_token = if prefix.is_empty() {
            rand::random_range(0..self.vocab_size) as u16
        } else {
            self.forward_sample_prefix(&prefix[..prefix.len() - 1]);
            prefix[prefix.len() - 1]
        };

        let mut out = Vec::with_capacity(max_len);

        for _ in 0..max_len {
            // ── one char step (uses cache[0]) ────────────────────────────────
            self.fill_input_buf(last_token);

            // char_model step → raw embedding
            Self::forward_sample_step(
                &mut self.char_model.layers,
                &mut self.char_model.cache[0],
                &self.char_input_buf,
            );

            // copy char1 output into word_context[..char_output]
            let char_output = self.char_model.output_size;
            let char1_out = self.char_model.cache[0].last().unwrap().output();
            self.word_context[..char_output].copy_from_slice(&char1_out);

            // char2_model step → logits over vocab
            Self::forward_sample_step(
                &mut self.char2_model.layers,
                &mut self.char2_model.cache[0],
                &self.word_context,
            );

            let logits = self.char2_model.cache[0].last().unwrap().output();

            // Temperature-scaled softmax sampling (identical to Sequential::sample)
            let scaled: Vec<f32> = logits.iter().map(|&v| v / temperature.max(1e-8)).collect();
            let probs = softmax(&scaled);
            let next = self.sample_from_probs(&probs);

            // ── word-boundary handling ────────────────────────────────────────
            // If the *just-emitted* token is a boundary, clock the high model.
            // (The paper feeds the boundary token to the word-level RNN.)
            if self.boundary_token_ids.contains(&next) {
                Self::forward_sample_step(
                    &mut self.high_model.layers,
                    &mut self.high_model.cache[0],
                    &char1_out,
                );

                // Reset char-level states for the next word
                for layer in &mut self.char_model.layers {
                    layer.reset_state();
                }
                for layer in &mut self.char2_model.layers {
                    layer.reset_state();
                }

                // Update context fed into char2 for the next word
                let high_out = self.high_model.cache[0].last().unwrap().output();
                self.word_context[char_output..].copy_from_slice(&high_out);
            }

            out.push(next);
            if !callback(next) {
                break;
            }
            last_token = next;
        }

        out
    }

    /// Feed a prefix into the model *without* saving training caches —
    /// updates all layer states (h, c) so subsequent `sample` steps are
    /// conditioned on the prefix.  Uses cache[0] as a single-step scratch.
    fn forward_sample_prefix(&mut self, prefix: &[u16]) {
        let char_output = self.char_model.output_size;
        self.word_context.fill(0.0);

        for t in 0..prefix.len() {
            let token = prefix[t];
            let next_token = prefix.get(t + 1);

            self.fill_input_buf(token);

            Self::forward_sample_step(
                &mut self.char_model.layers,
                &mut self.char_model.cache[0],
                &self.char_input_buf,
            );

            let char1_out = self.char_model.cache[0].last().unwrap().output().to_vec();
            self.word_context[..char_output].copy_from_slice(&char1_out);

            Self::forward_sample_step(
                &mut self.char2_model.layers,
                &mut self.char2_model.cache[0],
                &self.word_context,
            );

            let is_boundary = next_token.map_or(false, |v| self.boundary_token_ids.contains(v));

            if is_boundary {
                Self::forward_sample_step(
                    &mut self.high_model.layers,
                    &mut self.high_model.cache[0],
                    &char1_out,
                );

                for layer in &mut self.char_model.layers {
                    layer.reset_state();
                }
                for layer in &mut self.char2_model.layers {
                    layer.reset_state();
                }

                let high_out = self.high_model.cache[0].last().unwrap().output().to_vec();
                self.word_context[char_output..].copy_from_slice(&high_out);
            }
        }
    }

    fn sample_from_probs(&self, probs: &[f32]) -> u16 {
        let r = rand::random_range(0.0..1.0);
        let mut cum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= r {
                return i as u16;
            }
        }
        (probs.len() - 1) as u16
    }

    // ── Persistence ───────────────────────────────────────────────────────────
    //
    // Format:
    //   HIER_MAGIC    u32
    //   vocab_size    u32
    //   context_size  u32
    //   n_boundaries  u32
    //   boundary_ids  [u16 × n_boundaries]
    //   char_model    <Sequential blob>
    //   high_model    <Sequential blob>

    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut buf = Cursor::new(Vec::<u8>::new());
        let w = &mut buf as &mut dyn Write;

        // Header
        crate::saving::write_u32(w, HIER_MAGIC)?;
        write_u32(w, self.vocab_size as u32)?;
        write_u32(w, self.context_size as u32)?;
        write_u32(w, self.boundary_token_ids.len() as u32)?;
        for &id in &self.boundary_token_ids {
            write_u16(w, id)?;
        }

        // Sub-models
        self.char_model.write_to(w)?;
        self.char2_model.write_to(w)?;
        self.high_model.write_to(w)?;

        File::create(path)?.write_all(&buf.into_inner())
    }

    pub fn load(path: &str) -> io::Result<Self> {
        let r = &mut File::open(path)? as &mut dyn Read;

        let magic = read_u32(r)?;
        if magic != HIER_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Expected HIER magic 0x{HIER_MAGIC:08X}, got 0x{magic:08X}"),
            ));
        }

        let vocab_size = read_u32(r)? as usize;
        let context_size = read_u32(r)? as usize;
        let n_boundaries = read_u32(r)? as usize;
        let mut boundary_token_ids = Vec::with_capacity(n_boundaries);
        for _ in 0..n_boundaries {
            boundary_token_ids.push(read_u16(r)?);
        }

        let char_model = Sequential::load_from(r)?;
        let char2_model = Sequential::load_from(r)?;
        let high_model = Sequential::load_from(r)?;

        let char_input_buf = vec![0.0; vocab_size];
        let word_context = vec![0.0; char_model.output_size + context_size];

        Ok(Self {
            vocab_size,
            context_size,
            boundary_token_ids,
            char_model,
            char2_model,
            high_model,
            word_context: word_context.into_boxed_slice(),
            boundary_timesteps: vec![],
            char_input_buf,
            last_high_grad_signal: 0.0,
        })
    }
}

// ── Freie Helper ─────────────────────────────────────────────────────────────

/// Ein Layer-Stack rückwärts durchlaufen — exakt dieselbe Mechanik wie in
/// `Sequential::backwards_sequence`, nur ohne den Cross-Entropy-Start-Delta.
/// Der Aufrufer legt das Start-Delta in `delta_buf[..delta_len]` ab; die
/// Funktion schreibt dann `input_grad()` in jeden Cache und — für rekurrente
/// Layer — `dh_bptt` in den Layer selbst.
fn backward_through_layers(
    layers: &mut [Box<dyn NnLayer>],
    caches_t: &mut [Box<dyn crate::nn_layer::DynCache>],
    delta_buf: &mut [f32],
    mut delta_len: usize,
) {
    let n = layers.len();

    for l in (0..n).rev() {
        layers[l].backward(&mut delta_buf[..delta_len], caches_t[l].as_mut());

        if l == 0 {
            break;
        }

        let dx = caches_t[l].input_grad();
        let new_len = dx.len();
        delta_buf[..new_len].copy_from_slice(dx);

        delta_len = new_len;
    }
}
