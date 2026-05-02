use std::{
    fs::File,
    io::{self, Cursor, Read, Write},
    time::Instant,
    u32,
};

use crate::{
    config::{MODEL_LOC, SAVE_EVERY},
    loading::{read_u16, read_u32},
    nn_layer::{DynCache, NnLayer},
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

    delta_buf: Box<[f32]>,

    char_input_buf: Box<[f32]>,
    high_input_buf: Box<[f32]>,

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

        let char_input_buf = vec![0.0; vocab_size].into();
        let high_input_buf = vec![0.0; char_model.output_size + vocab_size].into();
        let word_context = vec![0.0; char_model.output_size + context_size].into();

        let buf_size = vocab_size
            .max(char2_model.input_size) // char_output + context_size
            .max(char_model.output_size + vocab_size); // high i

        let delta_buf = vec![0.0; buf_size].into();

        Self {
            char_model,
            char2_model,
            high_model,
            vocab_size,
            context_size,
            boundary_token_ids,
            word_context,
            delta_buf,
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
        caches_t: &mut [Box<dyn DynCache>],
        input: &[f32],
    ) {
        for l in 0..layers.len() {
            let (left, right) = caches_t.split_at_mut(l);
            let inp = if l == 0 { input } else { left[l - 1].output() };
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
            let inp = if l == 0 { input } else { left[l - 1].output() };
            layers[l].forward_sample(inp, right[0].as_mut());
        }
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

            if self.boundary_token_ids.contains(&token) {
                self.boundary_timesteps.push(t);
                let word_idx = self.boundary_timesteps.len() - 1;

                self.high_input_buf[..char_output]
                    .copy_from_slice(&self.word_context[..char_output]);
                self.high_input_buf[char_output..].fill(0.0);
                self.high_input_buf[char_output + token as usize] = 1.0;

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

            self.char_input_buf.fill(0.0);
            self.char_input_buf[token as usize] = 1.0;

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
        }
    }

    #[allow(unused)]
    pub fn backwards_sequence(&mut self, targets: &[u16], lr: f32) {
        if self.boundary_timesteps.is_empty() {
            return;
        }

        let char_output = self.char_model.output_size;
        let context_size = self.context_size;

        let mut d_high_ctx = vec![0.0; context_size];
        let mut d_char1_buf = vec![0.0; char_output];

        let mut boundary_i = self.boundary_timesteps.len() - 1;
        let mut high_grad_accum = 0.0;

        for t in (0..targets.len()).rev() {
            let is_boundary = boundary_i != usize::MAX && self.boundary_timesteps[boundary_i] == t;

            let out_len = {
                let out = self.char2_model.cache[t].last().unwrap().output();
                let len = out.len();
                self.delta_buf[..len].copy_from_slice(out);
                len
            };
            self.delta_buf[targets[t] as usize] -= 1.0;
            backward_through_layers(
                &mut self.char2_model.layers,
                &mut self.char2_model.cache[t],
                &mut self.delta_buf,
                out_len,
            );

            let dx2 = self.char2_model.cache[t][0].input_grad();
            for i in 0..context_size {
                d_high_ctx[i] += dx2[char_output + i];
            }

            for i in 0..char_output {
                self.delta_buf[i] = dx2[i] + d_char1_buf[i];
            }
            d_char1_buf.fill(0.0);

            backward_through_layers(
                &mut self.char_model.layers,
                &mut self.char_model.cache[t],
                &mut self.delta_buf,
                char_output,
            );
            if is_boundary {
                high_grad_accum +=
                    d_high_ctx.iter().map(|x| x.abs()).sum::<f32>() / context_size as f32;

                backward_through_layers(
                    &mut self.high_model.layers,
                    &mut self.high_model.cache[boundary_i],
                    &mut d_high_ctx,
                    context_size,
                );

                let d_hi = self.high_model.cache[boundary_i][0].input_grad();
                d_char1_buf.copy_from_slice(&d_hi[..char_output]);
                d_high_ctx.fill(0.0);

                for layer in &mut self.char_model.layers {
                    layer.accumulate_init_grad();
                    layer.zero_bptt_state();
                }
                for layer in &mut self.char2_model.layers {
                    layer.accumulate_init_grad();
                    layer.zero_bptt_state();
                }

                //self.char_model.sgd_step(lr * 0.5);
                //self.char2_model.sgd_step(lr * 0.5);

                boundary_i = boundary_i.wrapping_sub(1);
            }
        }

        for layer in &mut self.char_model.layers {
            layer.accumulate_init_grad();
            layer.zero_bptt_state();
        }
        for layer in &mut self.char2_model.layers {
            layer.accumulate_init_grad();
            layer.zero_bptt_state();
        }
        for layer in &mut self.high_model.layers {
            layer.accumulate_init_grad();
            layer.zero_bptt_state();
        }

        self.last_high_grad_signal = high_grad_accum / self.boundary_timesteps.len() as f32;
    }

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
            self.reset();
            self.forward_over(inputs);

            let loss = self.char2_model.seq_loss(targets);
            window_loss += loss;
            window_steps += 1;
            window_tokens += inputs.len();

            self.backwards_sequence(targets, lr / batch_size as f32);

            *step += 1;
            *iteration += 1;
            if *iteration % batch_size == 0 {
                let lr = lr / batch_size as f32;

                self.char_model.sgd_step(lr / 5.0);
                self.char2_model.sgd_step(lr / 5.0);
                self.high_model.sgd_step(lr);

                *iteration = 0;
                *j += 1;
            }

            if print_every > 0 && window_steps >= print_every {
                let avg = window_loss / window_steps as f32;
                let elapsed = window_start.elapsed();
                println!(
                    "{} | char loss {:.4} | ppl {:.4} | high ∇ {:.4} | {} tok | {:.1?}",
                    *step,
                    avg,
                    avg.exp(),
                    self.last_high_grad_signal,
                    window_tokens,
                    elapsed,
                );
                window_loss = 0.0;
                window_steps = 0;
                window_tokens = 0;
                window_start = std::time::Instant::now();
            }
            if *step % SAVE_EVERY == 0 {
                match self.save(MODEL_LOC) {
                    Ok(()) => println!("saved"),
                    Err(e) => eprintln!("save failed: {e}"),
                }
            }
        }

        if window_steps > 0 {
            let avg = window_loss / window_steps as f32;
            println!(
                "  step {:>7} | upd {:>5} | char loss {:.4} | ppl {:.4} | high ∇ {:.4}  (flush)",
                *step,
                *j,
                avg,
                avg.exp(),
                self.last_high_grad_signal,
            );
        }
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
            self.char_input_buf.fill(0.0);
            self.char_input_buf[last_token as usize] = 1.0;
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

    fn forward_sample_prefix(&mut self, prefix: &[u16]) {
        let char_output = self.char_model.output_size;
        self.word_context.fill(0.0);

        for t in 0..prefix.len() {
            let token = prefix[t];

            let is_boundary = self.boundary_token_ids.contains(&token);
            if is_boundary {
                // Fill high_input_buf = [char1_out | one_hot(boundary_token)]
                {
                    let char1_out = self.char_model.cache[0].last().unwrap().output();
                    self.high_input_buf[..char_output].copy_from_slice(char1_out);
                    self.high_input_buf[char_output..].fill(0.0);
                    self.high_input_buf[char_output + token as usize] = 1.0;
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

            // Forward char_model
            self.char_input_buf.fill(0.0);
            self.char_input_buf[token as usize] = 1.0;

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

        let char_input_buf = vec![0.0; vocab_size].into();
        let high_input_buf = vec![0.0; char_model.output_size + vocab_size].into();
        let word_context = vec![0.0; char_model.output_size + context_size].into();

        let buf_size = vocab_size
            .max(char2_model.input_size) // char_output + context_size
            .max(char_model.output_size + vocab_size); // high i

        let delta_buf = vec![0.0; buf_size].into();

        Ok(Self {
            vocab_size,
            context_size,
            boundary_token_ids,
            char_model,
            char2_model,
            high_model,
            word_context,
            delta_buf,
            boundary_timesteps: vec![],
            char_input_buf,
            high_input_buf,
            last_high_grad_signal: 0.0,
        })
    }
}

fn backward_through_layers(
    layers: &mut [Box<dyn NnLayer>],
    caches_t: &mut [Box<dyn DynCache>],
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
