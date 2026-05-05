use std::{
    fs::File,
    io::{self, Cursor, Read, Write},
    time::Instant,
    u32,
};

use crate::{
    config::MODEL_LOC,
    loading::{read_u16, read_u32},
    nn::softmax::softmax,
    nn_layer::{DynCache, NnLayer},
    saving::{HIER_MAGIC, write_u16, write_u32},
    sequential::Sequential,
    training::TrainingState,
};

pub struct HierarchicalSequential {
    pub char_model: Sequential,
    pub char2_model: Sequential,
    pub word_model: Sequential,

    pub vocab_size: usize,
    pub context_size: usize,

    pub boundary_token_ids: Vec<u16>,
    boundary_timesteps: Vec<usize>,

    delta_buf: Box<[f32]>,

    char_input: Box<[f32]>,
    char2_input: Box<[f32]>,

    d_high_ctx: Box<[f32]>,

    pub last_high_grad_signal: f32,
}

impl HierarchicalSequential {
    /// `boundary_token_ids` — from `tokenizer.boundary_token_ids()`.
    pub fn new(
        char_model: Sequential,
        char2_model: Sequential,
        word_model: Sequential,
        vocab_size: usize,
        boundary_token_ids: Vec<u16>,
    ) -> Self {
        let context_size = word_model.output_size;

        assert_eq!(
            char_model.input_size, vocab_size,
            "char_model.input_size must equal vocab_size"
        );
        assert_eq!(
            char2_model.output_size, vocab_size,
            "char2_model.output_size must equal vocab_size"
        );
        // word_model now only receives char1_out — no boundary token one-hot
        assert_eq!(
            char_model.output_size, word_model.input_size,
            "word_model.input_size must equal char_model.output_size"
        );
        assert_eq!(
            char2_model.input_size,
            word_model.output_size + char_model.output_size,
            "char2_model.input_size must equal word_model.output_size + char_model.output_size"
        );

        let char_input = vec![0.0; vocab_size].into();
        let char2_input = vec![0.0; char_model.output_size + context_size].into();

        let buf_size = vocab_size
            .max(char2_model.input_size)
            .max(char_model.output_size);

        let delta_buf = vec![0.0; buf_size].into();
        let d_high_ctx = vec![0.0; context_size].into();

        Self {
            char_model,
            char2_model,
            word_model,
            vocab_size,
            context_size,
            boundary_token_ids,
            char2_input,
            delta_buf,
            boundary_timesteps: vec![],
            char_input,
            d_high_ctx,
            last_high_grad_signal: 0.0,
        }
    }

    // ── Cache allocation ──────────────────────────────────────────────────────

    pub fn make_cache(&mut self, seq_len: usize) {
        self.char_model.make_cache(seq_len);
        self.char2_model.make_cache(seq_len);
        self.word_model.make_cache((seq_len / 3).max(1));
    }

    // ── State reset ───────────────────────────────────────────────────────────

    pub fn reset(&mut self) {
        for layer in &mut self.char_model.layers {
            layer.reset_state();
        }
        for layer in &mut self.char2_model.layers {
            layer.reset_state();
        }
        for layer in &mut self.word_model.layers {
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

    pub fn forward_over(&mut self, input: &[u16]) {
        self.char2_input.fill(0.0);
        let char_output = self.char_model.output_size;

        for t in 0..input.len() {
            let token = input[t];

            // ── 1. char1 forward ──────────────────────────────────────────────
            self.char_input.fill(0.0);
            self.char_input[token as usize] = 1.0;
            Self::forward_step(
                &mut self.char_model.layers,
                &mut self.char_model.cache[t],
                &self.char_input,
            );
            {
                let char1_out = self.char_model.cache[t].last().unwrap().output();
                self.char2_input[..char_output].copy_from_slice(char1_out);
            }

            if self.boundary_token_ids.contains(&token) {
                self.boundary_timesteps.push(t);
                let word_idx = self.boundary_timesteps.len() - 1;

                Self::forward_step(
                    &mut self.word_model.layers,
                    &mut self.word_model.cache[word_idx],
                    &self.char2_input[..char_output],
                );

                for layer in &mut self.char_model.layers {
                    layer.reset_state();
                }
                for layer in &mut self.char2_model.layers {
                    layer.reset_state();
                }

                let high_out = self.word_model.cache[word_idx].last().unwrap().output();
                self.assert_no_nan(high_out, "high_output", word_idx);
                self.char2_input[char_output..].copy_from_slice(high_out);
            }

            // ── 2. char2 forward ──────────────────────────────────────────────
            Self::forward_step(
                &mut self.char2_model.layers,
                &mut self.char2_model.cache[t],
                &self.char2_input,
            );
            self.assert_no_nan(
                self.char2_model.cache[t].last().unwrap().output(),
                "char2_output",
                t,
            );
        }
    }

    pub fn backwards_sequence(&mut self, targets: &[u16]) {
        if self.boundary_timesteps.is_empty() {
            return;
        }

        let char_output = self.char_model.output_size;
        let context_size = self.context_size;

        let mut high_grad_accum = 0.0;
        self.d_high_ctx.fill(0.0);

        let mut boundary_i: Option<usize> = self.boundary_timesteps.len().checked_sub(1);

        for t in (0..targets.len()).rev() {
            let boundary = boundary_i.filter(|&bi| self.boundary_timesteps[bi] == t);

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

            if boundary.is_some() {
                for layer in &mut self.char2_model.layers {
                    layer.accumulate_init_grad();
                    layer.zero_bptt_state();
                }
                for layer in &mut self.char_model.layers {
                    layer.accumulate_init_grad();
                    layer.zero_bptt_state();
                }
            }

            {
                let dx2 = self.char2_model.cache[t][0].input_grad();
                for i in 0..context_size {
                    self.d_high_ctx[i] += dx2[char_output + i];
                }
                self.delta_buf[..char_output].copy_from_slice(&dx2[..char_output]);
            }

            if let Some(bi) = boundary {
                high_grad_accum +=
                    self.d_high_ctx.iter().map(|x| x.abs()).sum::<f32>() / context_size as f32;

                backward_through_layers(
                    &mut self.word_model.layers,
                    &mut self.word_model.cache[bi],
                    &mut self.d_high_ctx,
                    context_size,
                );

                let d_hi = self.word_model.cache[bi][0].input_grad();
                for i in 0..char_output {
                    self.delta_buf[i] += d_hi[i];
                }
                self.d_high_ctx.fill(0.0);
            }

            backward_through_layers(
                &mut self.char_model.layers,
                &mut self.char_model.cache[t],
                &mut self.delta_buf,
                char_output,
            );

            if let Some(_) = boundary {
                boundary_i = boundary_i.and_then(|bi| bi.checked_sub(1));
            }
        }

        for layer in &mut self.char_model.layers {
            //layer.accumulate_init_grad();
            layer.zero_bptt_state();
        }
        for layer in &mut self.char2_model.layers {
            //layer.accumulate_init_grad();
            layer.zero_bptt_state();
        }
        for layer in &mut self.word_model.layers {
            layer.accumulate_init_grad();
            layer.zero_bptt_state();
        }

        self.last_high_grad_signal = high_grad_accum / self.boundary_timesteps.len() as f32;
    }

    pub fn train<'a, I: Iterator<Item = (&'a [u16], &'a [u16])>>(
        &mut self,
        data: I,
        state: &mut TrainingState,
    ) {
        let mut tokens = 0;
        let mut time = Instant::now();

        for (inputs, targets) in data {
            self.reset();
            self.forward_over(inputs);

            let loss = self.char2_model.seq_loss(targets);
            tokens += inputs.len();

            self.backwards_sequence(targets);

            if let Some(lr) = state.step(loss) {
                self.char_model.sgd_step(lr);
                self.char2_model.sgd_step(lr);
                self.word_model.sgd_step(lr);
            }

            if state.print() {
                let loss = state.get_loss();
                let elapsed = time.elapsed();
                println!(
                    "{} | char loss {:.4} | ppl {:.4} | high ∇ {:.4} | {} tok | {:.1?}",
                    state.step,
                    loss,
                    loss.exp(),
                    self.last_high_grad_signal,
                    tokens,
                    elapsed,
                );
                tokens = 0;
                time = Instant::now();
            }
            if state.save() {
                match self.save(MODEL_LOC) {
                    Ok(()) => println!("saved"),
                    Err(e) => eprintln!("save failed: {e}"),
                }
            }
        }
    }

    pub fn sample(
        &mut self,
        prefix: &[u16],
        max_len: usize,
        temperature: f32,
        mut callback: impl FnMut(u16) -> bool,
    ) -> Vec<u16> {
        if self.char_model.cache.is_empty() {
            self.make_cache(1);
        }

        self.reset();
        self.char2_input.fill(0.0);

        let char_output = self.char_model.output_size;

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
            // ── 1. char1 forward ──────────────────────────────────────────────
            self.char_input.fill(0.0);
            self.char_input[last_token as usize] = 1.0;
            Self::forward_sample_step(
                &mut self.char_model.layers,
                &mut self.char_model.cache[0],
                &self.char_input,
            );
            {
                let char1_out = self.char_model.cache[0].last().unwrap().output();
                self.char2_input[..char_output].copy_from_slice(char1_out);
            }

            // ── 2. boundary: word_model with char1_out only, then reset ───────
            if self.boundary_token_ids.contains(&last_token) {
                {
                    let char1_out = self.char_model.cache[0].last().unwrap().output();
                    Self::forward_sample_step(
                        &mut self.word_model.layers,
                        &mut self.word_model.cache[0],
                        char1_out,
                    );
                }
                {
                    let high_out = self.word_model.cache[0].last().unwrap().output();
                    self.char2_input[char_output..].copy_from_slice(high_out);
                }
                for layer in &mut self.char_model.layers {
                    layer.reset_state();
                }
                for layer in &mut self.char2_model.layers {
                    layer.reset_state();
                }
            }

            // ── 3. char2 → logits ─────────────────────────────────────────────
            Self::forward_sample_step(
                &mut self.char2_model.layers,
                &mut self.char2_model.cache[0],
                &self.char2_input,
            );

            // ── 4. temperature-scaled sampling ────────────────────────────────
            let next_token = {
                let logits = self.char2_model.cache[0].last().unwrap().output();
                let scaled: Vec<f32> = logits.iter().map(|&v| v / temperature.max(1e-8)).collect();
                let probs = softmax(&scaled);
                self.sample_from_probs(&probs)
            };

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
        self.char2_input.fill(0.0);

        for t in 0..prefix.len() {
            let token = prefix[t];

            // ── 1. char1 forward ──────────────────────────────────────────────
            self.char_input.fill(0.0);
            self.char_input[token as usize] = 1.0;
            Self::forward_sample_step(
                &mut self.char_model.layers,
                &mut self.char_model.cache[0],
                &self.char_input,
            );
            {
                let char1_out = self.char_model.cache[0].last().unwrap().output();
                self.char2_input[..char_output].copy_from_slice(char1_out);
            }

            // ── 2. boundary: word_model with char1_out only, then reset ───────
            if self.boundary_token_ids.contains(&token) {
                {
                    let char1_out = self.char_model.cache[0].last().unwrap().output();
                    Self::forward_sample_step(
                        &mut self.word_model.layers,
                        &mut self.word_model.cache[0],
                        char1_out,
                    );
                }
                {
                    let high_out = self.word_model.cache[0].last().unwrap().output();
                    self.char2_input[char_output..].copy_from_slice(high_out);
                }
                for layer in &mut self.char_model.layers {
                    layer.reset_state();
                }
                for layer in &mut self.char2_model.layers {
                    layer.reset_state();
                }
            }

            // ── 3. char2 state update (output discarded) ──────────────────────
            Self::forward_sample_step(
                &mut self.char2_model.layers,
                &mut self.char2_model.cache[0],
                &self.char2_input,
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

    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut buf = Cursor::new(Vec::<u8>::new());
        let w = &mut buf as &mut dyn Write;

        crate::saving::write_u32(w, HIER_MAGIC)?;
        write_u32(w, self.vocab_size as u32)?;
        write_u32(w, self.context_size as u32)?;
        write_u32(w, self.boundary_token_ids.len() as u32)?;
        for &id in &self.boundary_token_ids {
            write_u16(w, id)?;
        }

        self.char_model.write_to(w)?;
        self.char2_model.write_to(w)?;
        self.word_model.write_to(w)?;

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
        let char2_input = vec![0.0; char_model.output_size + context_size].into();

        let buf_size = vocab_size
            .max(char2_model.input_size)
            .max(char_model.output_size);

        let delta_buf = vec![0.0; buf_size].into();
        let d_high_ctx = vec![0.0; context_size].into();

        Ok(Self {
            vocab_size,
            context_size,
            boundary_token_ids,
            char_model,
            char2_model,
            word_model: high_model,
            char2_input,
            delta_buf,
            boundary_timesteps: vec![],
            char_input: char_input_buf,
            d_high_ctx,
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
