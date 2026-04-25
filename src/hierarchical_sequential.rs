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
    /// Scratch buffer for the high model's input: [char1_out | one_hot(boundary_token)].
    /// Paper (Hwang & Sung 2016): "the current word boundary token information
    /// is given to the word-level module."
    high_input_buf: Vec<f32>,

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

        // Paper (Hwang & Sung 2016): the boundary token itself is fed to the
        // word-level (high) module together with the character embedding.
        // Therefore high_model.input_size = char_model.output_size + vocab_size.
        assert_eq!(
            char_model.output_size + vocab_size,
            high_model.input_size,
            "high_model.input_size must equal char_model.output_size + vocab_size (boundary char is concatenated to the word embedding)"
        );

        assert_eq!(
            char2_model.input_size,
            high_model.output_size + char_model.output_size
        );

        let char_input_buf = vec![0.0; vocab_size];
        let high_input_buf = vec![0.0; char_model.output_size + vocab_size];
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
            high_input_buf,
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

                // Build high_model input = [char1_out | one_hot(boundary_token)].
                // The boundary token is input[t+1] (the char that triggered the clock).
                let boundary_token = *next_token.unwrap();
                let char1_out = self.char_model.cache[t].last().unwrap().output();
                self.high_input_buf[..char_output].copy_from_slice(char1_out);
                self.high_input_buf[char_output..].fill(0.0);
                self.high_input_buf[char_output + boundary_token as usize] = 1.0;

                Self::forward_step(
                    &mut self.high_model.layers,
                    &mut self.high_model.cache[word_idx],
                    &self.high_input_buf,
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
        let seq_len = targets.len();
        let char_output = self.char_model.output_size;
        let context_size = self.context_size;

        // ── Scratch buffers (no heap alloc inside the loop) ──────────────────
        // delta_buf must hold the largest intermediate gradient vector:
        //   • CE gradient      : vocab_size     (char2 output)
        //   • char2 input grad : char_output + context_size
        //   • high input grad  : char_output + vocab_size
        let buf_size = self
            .vocab_size
            .max(self.char2_model.input_size) // char_output + context_size
            .max(self.char_model.output_size + self.vocab_size); // high input
        let mut delta_buf = vec![0.0; buf_size];

        // Gradient flowing back into the high-model's output (= context vector).
        // Accumulates contributions from all chars that USED a given high context.
        let mut d_high_ctx: Vec<f32> = vec![0.0; context_size];

        // Combined d(char1[t]) = contribution from char2 + contribution from high.
        let mut d_char1_buf = vec![0.0; char_output];

        // ── Zero BPTT state ──────────────────────────────────────────────────
        for layer in &mut self.char_model.layers {
            layer.zero_bptt_state();
        }
        for layer in &mut self.char2_model.layers {
            layer.zero_bptt_state();
        }
        for layer in &mut self.high_model.layers {
            layer.zero_bptt_state();
        }

        // Walk boundary_timesteps backwards: boundary_ptr points one past the
        // current unprocessed boundary.
        let mut boundary_ptr = self.boundary_timesteps.len();
        let mut high_grad_accum = 0.0;

        for t in (0..seq_len).rev() {
            let at_boundary = boundary_ptr > 0 && self.boundary_timesteps[boundary_ptr - 1] == t;

            if at_boundary {
                boundary_ptr -= 1;
                let word_idx = boundary_ptr; // 0-indexed word whose high step ran at t

                // A. Capture init-grad for the word segment that starts AFTER this
                //    boundary (chars from t+1 up to the next boundary).  Their BPTT
                //    signal has fully propagated into dh_bptt / dc_bptt by now.
                // B. Zero char BPTT — the char model was reset in the forward pass,
                //    so no gradient should cross this word boundary.
                for layer in &mut self.char_model.layers {
                    layer.accumulate_init_grad();
                    layer.zero_bptt_state();
                }
                for layer in &mut self.char2_model.layers {
                    layer.accumulate_init_grad();
                    layer.zero_bptt_state();
                }

                // C. Backprop high_model[word_idx] with d_high_ctx.
                //    d_high_ctx = Σ dx2[char_output..] for all t STRICTLY AFTER this
                //    boundary — exactly dL/d(high_ctx_NEW).
                high_grad_accum +=
                    d_high_ctx.iter().map(|x| x * x).sum::<f32>() / context_size as f32;

                delta_buf[..context_size].copy_from_slice(&d_high_ctx);
                backward_through_layers(
                    &mut self.high_model.layers,
                    &mut self.high_model.cache[word_idx],
                    &mut delta_buf,
                    context_size,
                );

                // D. Extract d(char1[t]) from the high model's input gradient.
                //    high input was [char1[t] | one_hot(boundary_token)], so the
                //    first char_output elements are dL/d(char1[t]) via the high path.
                {
                    let d_hi = self.high_model.cache[word_idx][0].input_grad();
                    d_char1_buf.copy_from_slice(&d_hi[..char_output]);
                }

                // E. Reset d_high_ctx — now accumulate dL/d(high_ctx_PREV),
                //    i.e. the gradient toward high_model[word_idx - 1].
                d_high_ctx.fill(0.0);

                // F. CE loss gradient for char2[t], then backprop through char2.
                let out_len = {
                    let out = self.char2_model.cache[t].last().unwrap().output();
                    let len = out.len();
                    delta_buf[..len].copy_from_slice(out);
                    len
                };
                delta_buf[targets[t] as usize] -= 1.0;
                backward_through_layers(
                    &mut self.char2_model.layers,
                    &mut self.char2_model.cache[t],
                    &mut delta_buf,
                    out_len,
                );

                // G. Read dx2 = dL/d([char1[t] | high_ctx_PREV]).
                //    • dx2[char_output..] → first contribution to d_high_ctx_PREV
                //      (char2[t] itself used high_ctx_PREV).
                //    • dx2[..char_output] → combined with d_char1_buf from high.
                {
                    let dx2 = self.char2_model.cache[t][0].input_grad();
                    for i in 0..context_size {
                        d_high_ctx[i] += dx2[char_output + i];
                    }
                    for i in 0..char_output {
                        d_char1_buf[i] += dx2[i]; // add char2 contribution
                    }
                }

                // H. Backprop char_model[t] with the combined d(char1[t]).
                delta_buf[..char_output].copy_from_slice(&d_char1_buf);
                backward_through_layers(
                    &mut self.char_model.layers,
                    &mut self.char_model.cache[t],
                    &mut delta_buf,
                    char_output,
                );
            } else {
                // ── Non-boundary timestep: standard BPTT ─────────────────────

                // CE loss gradient, backprop char2.
                let out_len = {
                    let out = self.char2_model.cache[t].last().unwrap().output();
                    let len = out.len();
                    delta_buf[..len].copy_from_slice(out);
                    len
                };
                delta_buf[targets[t] as usize] -= 1.0;
                backward_through_layers(
                    &mut self.char2_model.layers,
                    &mut self.char2_model.cache[t],
                    &mut delta_buf,
                    out_len,
                );

                // Accumulate dL/d(high_ctx) from dx2[char_output..].
                // Backprop char_model[t] with dx2[..char_output].
                {
                    let dx2 = self.char2_model.cache[t][0].input_grad();
                    for i in 0..context_size {
                        d_high_ctx[i] += dx2[char_output + i];
                    }
                    delta_buf[..char_output].copy_from_slice(&dx2[..char_output]);
                }
                backward_through_layers(
                    &mut self.char_model.layers,
                    &mut self.char_model.cache[t],
                    &mut delta_buf,
                    char_output,
                );
            }
        }

        // ── Final init-grad accumulation ─────────────────────────────────────
        // After t=0, dh_bptt/dc_bptt hold dL/d(h_init) for word 0's char segment
        // and for the entire high-model chain.
        for layer in &mut self.char_model.layers {
            layer.accumulate_init_grad();
        }
        for layer in &mut self.char2_model.layers {
            layer.accumulate_init_grad();
        }
        for layer in &mut self.high_model.layers {
            layer.accumulate_init_grad();
        }

        self.last_high_grad_signal = if self.boundary_timesteps.len() > 0 {
            (high_grad_accum / self.boundary_timesteps.len() as f32).sqrt()
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
                let char_lr = lr;
                let high_lr = if self.boundary_timesteps.is_empty() {
                    lr
                } else {
                    //lr * (inputs.len() as f32 / self.boundary_timesteps.len() as f32)
                    lr
                };

                for layer in &mut self.char_model.layers {
                    layer.apply_grads(char_lr);
                }
                self.char_model.clear_grads();

                for layer in &mut self.char2_model.layers {
                    layer.apply_grads(char_lr);
                }
                self.char2_model.clear_grads();

                for layer in &mut self.high_model.layers {
                    layer.apply_grads(high_lr);
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
        // Ensure at least one scratch cache slot exists.
        if self.char_model.cache.is_empty() {
            self.make_cache(1);
        }

        self.reset();
        self.word_context.fill(0.0);

        let char_output = self.char_model.output_size;

        // Warm up the model state from the prefix (all tokens but the last),
        // then seed the generation loop with the last prefix token.
        let mut last_token = if prefix.is_empty() {
            rand::random_range(0..self.vocab_size) as u16
        } else {
            if prefix.len() > 1 {
                self.forward_sample_prefix(&prefix[..prefix.len() - 1]);
            }
            prefix[prefix.len() - 1]
        };

        let mut out = Vec::with_capacity(max_len);

        for _ in 0..max_len {
            // ── 1. Forward char_model ─────────────────────────────────────────
            self.fill_input_buf(last_token);
            Self::forward_sample_step(
                &mut self.char_model.layers,
                &mut self.char_model.cache[0],
                &self.char_input_buf,
            );

            // ── 2. Update word_context[..char_output] with char1 output ───────
            {
                let char1_out = self.char_model.cache[0].last().unwrap().output();
                self.word_context[..char_output].copy_from_slice(char1_out);
            }

            // ── 3. Forward char2_model → logits ───────────────────────────────
            Self::forward_sample_step(
                &mut self.char2_model.layers,
                &mut self.char2_model.cache[0],
                &self.word_context,
            );

            // ── 4. Temperature-scaled sampling ────────────────────────────────
            let next_token = {
                let logits = self.char2_model.cache[0].last().unwrap().output();
                let scaled: Vec<f32> = logits.iter().map(|&v| v / temperature.max(1e-8)).collect();
                let probs = softmax(&scaled);
                self.sample_from_probs(&probs)
            };

            // ── 5. Boundary: run high_model, reset char/char2 ─────────────────
            if self.boundary_token_ids.contains(&next_token) {
                // high_input = [char1_out | one_hot(next_token)]
                {
                    let char1_out = self.char_model.cache[0].last().unwrap().output();
                    self.high_input_buf[..char_output].copy_from_slice(char1_out);
                    self.high_input_buf[char_output..].fill(0.0);
                    self.high_input_buf[char_output + next_token as usize] = 1.0;
                }
                Self::forward_sample_step(
                    &mut self.high_model.layers,
                    &mut self.high_model.cache[0],
                    &self.high_input_buf,
                );
                {
                    let high_out = self.high_model.cache[0].last().unwrap().output();
                    self.word_context[char_output..].copy_from_slice(high_out);
                }
                for layer in &mut self.char_model.layers {
                    layer.reset_state();
                }
                for layer in &mut self.char2_model.layers {
                    layer.reset_state();
                }
            }

            out.push(next_token);
            if !callback(next_token) {
                break;
            }
            last_token = next_token;
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
            let next_token = prefix.get(t + 1).copied();

            // Forward char_model
            self.fill_input_buf(token);
            Self::forward_sample_step(
                &mut self.char_model.layers,
                &mut self.char_model.cache[0],
                &self.char_input_buf,
            );

            // Copy char1 output into word_context
            {
                let char1_out = self.char_model.cache[0].last().unwrap().output();
                self.word_context[..char_output].copy_from_slice(char1_out);
            }

            // Forward char2_model (we only need the state update, output discarded)
            Self::forward_sample_step(
                &mut self.char2_model.layers,
                &mut self.char2_model.cache[0],
                &self.word_context,
            );

            // If the NEXT token is a boundary, run the high model exactly as in
            // the training forward pass, then reset char/char2 states.
            let is_boundary = next_token.map_or(false, |v| self.boundary_token_ids.contains(&v));
            if is_boundary {
                let boundary_token = next_token.unwrap();

                // Fill high_input_buf = [char1_out | one_hot(boundary_token)]
                {
                    let char1_out = self.char_model.cache[0].last().unwrap().output();
                    self.high_input_buf[..char_output].copy_from_slice(char1_out);
                    self.high_input_buf[char_output..].fill(0.0);
                    self.high_input_buf[char_output + boundary_token as usize] = 1.0;
                }
                Self::forward_sample_step(
                    &mut self.high_model.layers,
                    &mut self.high_model.cache[0],
                    &self.high_input_buf,
                );

                // Update high context in word_context
                {
                    let high_out = self.high_model.cache[0].last().unwrap().output();
                    self.word_context[char_output..].copy_from_slice(high_out);
                }

                // Reset char/char2 — next char starts a fresh word
                for layer in &mut self.char_model.layers {
                    layer.reset_state();
                }
                for layer in &mut self.char2_model.layers {
                    layer.reset_state();
                }
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
        let high_input_buf = vec![0.0; char_model.output_size + vocab_size];
        let word_context = vec![0.0; char_model.output_size + context_size];

        Ok(Self {
            vocab_size,
            context_size,
            boundary_token_ids,
            char_model,
            char2_model,
            high_model,
            word_context: word_context.into(),
            boundary_timesteps: vec![],
            char_input_buf,
            high_input_buf,
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
