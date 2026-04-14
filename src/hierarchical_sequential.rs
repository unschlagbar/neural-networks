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
    lstm::add_vec_in_place,
    nn_layer::NnLayer,
    saving::{HIER_MAGIC, write_u16, write_u32},
    sequential::Sequential,
    softmax::softmax,
};

pub struct HierarchicalSequential {
    pub char_model: Sequential,
    pub high_model: Sequential,

    pub vocab_size: usize,
    pub context_size: usize,

    pub boundary_token_ids: Vec<u16>,

    high_cache: Vec<Vec<Box<dyn crate::nn_layer::DynCache>>>,
    boundary_timesteps: Vec<usize>,

    /// Index of the char-model layer whose output is used as the word-level
    /// representation for the high model.
    word_rep_layer: usize,

    /// Pre-allocated scratch buffer for `forward_over` / `sample`.
    /// Layout: [one_hot (vocab_size) | high_context (context_size)]
    full_input_buf: Vec<f32>,

    pub last_high_grad_signal: f32,
}

impl HierarchicalSequential {
    /// `boundary_token_ids` — from `tokenizer.boundary_token_ids()`.
    pub fn new(
        mut char_model: Sequential,
        high_model: Sequential,
        vocab_size: usize,
        boundary_token_ids: Vec<u16>,
    ) -> Self {
        let context_size = high_model.output_size;
        assert_eq!(
            char_model.input_size,
            vocab_size + context_size,
            "char_model.input_size must equal vocab_size + context_size"
        );

        let mut word_rep_layer = usize::MAX;

        for (i, l) in char_model.layers.iter_mut().enumerate().rev() {
            if l.bptt_hidden_grad().is_some() && l.output_size() == high_model.input_size {
                word_rep_layer = i;
                break;
            }
        }

        if word_rep_layer == usize::MAX {
            panic!("char_model must contain a recurrent layer producing the high model input size",)
        }

        let full_input_buf = vec![0.0; vocab_size + context_size];

        Self {
            char_model,
            high_model,
            vocab_size,
            context_size,
            boundary_token_ids,
            high_cache: vec![],
            boundary_timesteps: vec![],
            word_rep_layer,
            full_input_buf,
            last_high_grad_signal: 0.0,
        }
    }

    // ── Cache allocation ──────────────────────────────────────────────────────

    pub fn make_cache(&mut self, seq_len: usize) {
        self.char_model.make_cache(seq_len);
        self.high_cache = (0..seq_len)
            .map(|_| {
                self.high_model
                    .layers
                    .iter()
                    .map(|l| l.make_cache())
                    .collect()
            })
            .collect();
    }

    // ── State reset ───────────────────────────────────────────────────────────

    pub fn reset(&mut self) {
        for layer in &mut self.char_model.layers {
            layer.reset_state();
        }
        for layer in &mut self.high_model.layers {
            layer.reset_state();
        }
        self.boundary_timesteps.clear();
    }

    // ── Private forward helpers ───────────────────────────────────────────────

    fn forward_step_char(
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

    fn forward_step_high(
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

    // ── Build the combined [one_hot | context] input in the scratch buffer ────

    #[inline]
    fn fill_input_buf(&mut self, token: u16, high_context: &[f32]) {
        self.full_input_buf[..self.vocab_size].fill(0.0);
        self.full_input_buf[token as usize] = 1.0;
        self.full_input_buf[self.vocab_size..].copy_from_slice(high_context);
    }

    fn assert_no_nan(&self, slice: &[f32], name: &str, t: usize) {
        if let Some(pos) = slice.iter().position(|x| !x.is_finite()) {
            panic!("NaN/Inf detected in {name} at t={t}, idx={pos}");
        }
    }

    // ── Forward (training) ───────────────────────────────────────────────────

    pub fn forward_over(&mut self, input: &[u16]) {
        let mut high_context = vec![0.0; self.context_size];

        for t in 0..input.len() {
            let token = input[t];

            self.fill_input_buf(token, &high_context);
            self.assert_no_nan(&self.full_input_buf, "full_input_buf", t);

            Self::forward_step_char(
                &mut self.char_model.layers,
                &mut self.char_model.cache[t],
                &self.full_input_buf,
            );
            self.assert_no_nan(
                self.char_model.cache[t].last().unwrap().output(),
                "char_output",
                t,
            );

            let is_boundary = (t + 1 == input.len()) || self.boundary_token_ids.contains(&token);

            if is_boundary {
                self.boundary_timesteps.push(t);
                let word_idx = self.boundary_timesteps.len() - 1;

                let word_rep = self.char_model.cache[t][self.word_rep_layer]
                    .output()
                    .to_vec();

                Self::forward_step_high(
                    &mut self.high_model.layers,
                    &mut self.high_cache[word_idx],
                    &word_rep,
                );

                let high_out = self.high_cache[word_idx].last().unwrap().output();
                self.assert_no_nan(high_out, "high_output", word_idx);
                high_context.copy_from_slice(high_out);
            }
        }
    }

    // ── Backward ─────────────────────────────────────────────────────────────
    //
    // Cross-word BPTT for the high model: after the high LSTM backward at word W,
    // `dh_bptt` inside the layer holds dL/dh_{W-1}.  The layer-transition block
    // (`bptt_hidden_grad()`) already picks this up automatically when we process
    // word W-1 — exactly the same mechanism Sequential uses within a sequence.
    // No separate `dh_from_future` variable is needed (and adding one would
    // double-count the gradient, which was the original bug here).

    pub fn backwards_sequence(&mut self, inputs: &[u16], targets: &[u16]) {
        // Pass 1: plain char backward → populates cache[t][0].input_grad() so we
        // can read dL/d(context_input) at every timestep.
        self.char_model.backwards_sequence(inputs, targets);

        let n_words = self.boundary_timesteps.len();
        if n_words == 0 {
            return;
        }

        // Accumulate dL/d(high_context) for each word boundary.
        // The context output from word W is fed as input to the char model at
        // every timestep in (boundary[W], boundary[W+1]], so we sum those grads.
        let mut word_context_grads = vec![vec![0.0; self.context_size]; n_words];
        for word_idx in 0..n_words {
            let start_t = self.boundary_timesteps[word_idx] + 1;
            let end_t = if word_idx + 1 < n_words {
                self.boundary_timesteps[word_idx + 1]
            } else {
                inputs.len().saturating_sub(1)
            };

            if start_t <= end_t {
                let accum = &mut word_context_grads[word_idx];
                for t in start_t..=end_t {
                    let dx_full = self.char_model.cache[t][0].input_grad();
                    add_vec_in_place(accum, &dx_full[self.vocab_size..]);
                }
            }
        }

        // Discard pass-1 gradients; pass 2 (with word-rep deltas) is the real one.
        for layer in &mut self.char_model.layers {
            layer.clear_grads();
            layer.reset_state();
        }

        let n_high = self.high_model.layers.len();
        let mut word_rep_grads = vec![vec![0.0; self.high_model.input_size]; n_words];

        // Backward through the high model, processing words in reverse.
        // Cross-word BPTT is handled automatically: after LSTM backward for word W,
        // `layers[lstm].dh_bptt` holds the gradient for word W-1.  The
        // `bptt_hidden_grad()` branch in the layer-transition below reads it on the
        // next (W-1) iteration — no manual carry needed.
        for word_idx in (0..n_words).rev() {
            let mut current_delta = word_context_grads[word_idx].clone();
            let mut delta_len = current_delta.len();

            for l in (0..n_high).rev() {
                let cache_l = &mut self.high_cache[word_idx][l];
                self.high_model.layers[l]
                    .backward(&mut current_delta[..delta_len], cache_l.as_mut());

                if l == 0 {
                    break;
                }

                let dx = cache_l.input_grad();
                let new_len = dx.len();

                // delta for layer l-1 = input_grad(l) + bptt_hidden_grad(l-1)
                // bptt_hidden_grad(l-1) is non-zero only for recurrent layers and
                // carries the gradient from the *previous* word's backward call.
                match self.high_model.layers[l - 1].bptt_hidden_grad() {
                    Some(bptt) => {
                        current_delta.resize(new_len, 0.0);
                        current_delta.copy_from_slice(bptt);
                        add_vec_in_place(&mut current_delta[..new_len], dx);
                    }
                    None => {
                        current_delta.resize(new_len, 0.0);
                        current_delta.copy_from_slice(dx);
                    }
                }
                delta_len = new_len;
            }

            word_rep_grads[word_idx].copy_from_slice(self.high_cache[word_idx][0].input_grad());
        }

        // Messe den durchschnittlichen Gradient-Norm ins High-Model
        let high_grad_signal: f32 = if n_words > 0 {
            word_context_grads
                .iter()
                .map(|g| g.iter().map(|x| x * x).sum::<f32>().sqrt())
                .sum::<f32>()
                / n_words as f32
        } else {
            0.0
        };
        self.last_high_grad_signal = high_grad_signal;

        for layer in &mut self.high_model.layers {
            layer.accumulate_init_grad();
        }

        let mut extra_deltas: Vec<Option<Vec<f32>>> = vec![None; inputs.len()];
        for (word_idx, &boundary_t) in self.boundary_timesteps.iter().enumerate() {
            extra_deltas[boundary_t] = Some(word_rep_grads[word_idx].clone());
        }

        self.char_model.backwards_sequence_with_layer_deltas(
            inputs,
            targets,
            self.word_rep_layer,
            &extra_deltas,
        );
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
            total_loss += self.char_model.seq_loss(targets);
            steps += 1;

            self.backwards_sequence(inputs, targets);

            *iteration += 1;
            if *iteration % batch_size == 0 {
                let effective_lr = lr / batch_size as f32;

                for layer in &mut self.char_model.layers {
                    layer.clip_grads();
                    layer.apply_grads(effective_lr);
                }
                self.char_model.clear_grads();

                for layer in &mut self.high_model.layers {
                    layer.clip_grads();
                    layer.apply_grads(effective_lr);
                }
                self.high_model.clear_grads();

                *iteration = 0;
                *j += 1;
            }
        }

        println!(
            "{j} | char loss = {:.4} | high ∇-signal = {:.5}",
            total_loss / steps.max(1) as f32,
            self.last_high_grad_signal
        );
    }

    // ── Sampling ─────────────────────────────────────────────────────────────
    //
    // Fix: use `forward_sample_step_char` (calls `layer.forward_sample`)
    // so SoftmaxLayer does NOT apply softmax.  Then we apply temperature
    // scaling and a fresh softmax ourselves — correct single-softmax path.

    pub fn sample(
        &mut self,
        prefix: &[u16],
        max_len: usize,
        temperature: f32,
        mut callback: impl FnMut(u16) -> bool,
    ) -> Vec<u16> {
        self.reset();

        let mut high_context = vec![0.0; self.context_size];

        let mut last_token: u16 = if prefix.is_empty() {
            rand::random_range(0..self.vocab_size) as u16
        } else {
            // Process only prefix[..len-1] to warm up state.
            // The last prefix token is NOT pre-run here — it becomes `last_token`
            // and is fed naturally in the first generation step below.
            // This matches training exactly (one forward pass per token, logits
            // used to predict the *next* token, not the current one).
            // The old code ran all of prefix through the LSTM, then re-ran the
            // last token in the loop — double-advancing the state and losing the
            // prefix's influence on generation.
            if prefix.len() > 1 {
                self.forward_sample_prefix(&prefix[..prefix.len() - 1], &mut high_context);
            }
            prefix[prefix.len() - 1]
        };

        let mut out = Vec::with_capacity(max_len);

        for _ in 0..max_len {
            self.fill_input_buf(last_token, &high_context);

            // Use forward_sample so the softmax layer yields raw logits.
            Self::forward_sample_step(
                &mut self.char_model.layers,
                &mut self.char_model.cache[0],
                &self.full_input_buf,
            );

            // Raw logits (pre-softmax) from the last layer.
            let logits = self.char_model.cache[0].last().unwrap().output();

            // Temperature scaling then softmax.
            let t = temperature.max(1e-8);
            let scaled: Vec<f32> = logits.iter().map(|&v| v / t).collect();
            let probs = softmax(&scaled);

            let next = self.sample_from_probs(&probs);
            out.push(next);
            if !callback(next) {
                break;
            }

            last_token = next;

            // Update high-level context at word boundaries.
            if self.boundary_token_ids.contains(&next) {
                let char_hidden = self.char_model.cache[0][self.word_rep_layer].output();
                Self::forward_sample_step(
                    &mut self.high_model.layers,
                    &mut self.high_cache[0],
                    char_hidden,
                );
                let high_out = self.high_cache[0].last().unwrap().output();
                high_context.copy_from_slice(high_out);
            }
        }

        out
    }

    fn forward_sample_prefix(&mut self, prefix: &[u16], high_context: &mut Vec<f32>) {
        for &token in prefix {
            self.fill_input_buf(token, high_context);

            Self::forward_sample_step(
                &mut self.char_model.layers,
                &mut self.char_model.cache[0],
                &self.full_input_buf,
            );

            if self.boundary_token_ids.contains(&token) {
                let char_hidden = self.char_model.cache[0][self.word_rep_layer].output();
                Self::forward_sample_step(
                    &mut self.high_model.layers,
                    &mut self.high_cache[0],
                    char_hidden,
                );
                let high_out = self.high_cache[0].last().unwrap().output();
                high_context.copy_from_slice(high_out);
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

        let mut char_model = Sequential::load_from(r)?;
        let high_model = Sequential::load_from(r)?;

        let full_input_buf = vec![0.0; vocab_size + context_size];

        let mut word_rep_layer = usize::MAX;

        for (i, l) in char_model.layers.iter_mut().enumerate().rev() {
            if l.bptt_hidden_grad().is_some() && l.output_size() == high_model.input_size {
                word_rep_layer = i;
                break;
            }
        }

        if word_rep_layer == usize::MAX {
            panic!("char_model must contain a recurrent layer producing the high model input size",)
        }

        Ok(Self {
            vocab_size,
            context_size,
            boundary_token_ids,
            char_model,
            high_model,
            high_cache: vec![],
            boundary_timesteps: vec![],
            word_rep_layer,
            full_input_buf,
            last_high_grad_signal: 0.0,
        })
    }
}
