use std::{
    fs::File,
    io::{self, Cursor, Read, Write},
    range::Range,
    rc::Rc,
    time::Instant,
};

use crate::{
    batches::WordBatch,
    bi_encoder::BiEncoder,
    config::MAX_SEQ_LEN,
    loading::{read_f32_vec, read_matrix, read_u16, read_u32, read_u64},
    nn::{
        linear::LinearLayer,
        mlstm_block::{MLSTMBlock, MLSTMBlockCache},
        rms_norm::RMSNorm,
        softmax::{softmax, softmax_inplace},
    },
    nn_layer::{DynCache, NnLayer},
    saving::{HIER_MAGIC, write_f32_slice, write_matrix, write_u16, write_u32, write_u64},
    sequential::Sequential,
    tokenizer::Tokenizer,
    training::TrainingState,
};

/// Inference-time ablation of the cross-word context path, used by `probe`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackboneMode {
    /// Normal: the backbone carries recurrent state across all words.
    Normal,
    /// Reset the backbone's recurrent state before every word, so the context
    /// `o` reflects only the immediately preceding word — no longer history.
    ResetEachWord,
    /// Zero the word context handed to the decoder: it predicts each word's
    /// chars with no information about any previous word (within-word floor).
    ZeroContext,
}

pub struct Hierarchical {
    /// Bidirectional per-word encoder: reads each word in both directions and
    /// merges the readouts into a word embedding `e_w`. (See `bi_encoder.rs`.)
    pub encoder: BiEncoder,
    pub char2_model: Sequential,
    pub word_model: Sequential,

    pub vocab_size: usize,
    pub context_size: usize,

    pub tokenizer: Rc<Tokenizer>,
    /// The `[W]` marker id (encoder prefix / decoder BOS+EOS).
    w_token: u16,

    pub boundary_token_ids: Vec<u16>,
    /// `[start, end)` (half-open) token ranges over the window, one per word.
    /// Filled by `forward_over`, reused by `backwards_sequence`.
    word_segments: Vec<Range<usize>>,

    /// Per decode-word: decoder cache slot range (length `word_len + 1`, the
    /// trailing `[W]` EOS step included).
    dec_ranges: Vec<Range<usize>>,
    /// Target token id for each decoder cache slot (parallel to `char2_model.cache`).
    dec_targets: Vec<u16>,

    /// Decoder input: `[char one-hot ‖ word context]` (size `vocab + context`).
    char2_input: Box<[f32]>,

    /// Gradient w.r.t. the word embedding `e_w` (size `char_out`).
    d_ew_buf: Box<[f32]>,

    /// Backward scratch: top-level delta as it flows down a layer stack.
    delta_buf: Box<[f32]>,
    /// Accumulated gradient w.r.t. the decoder's word context, per word.
    d_o_buf: Box<[f32]>,

    pub last_high_grad_signal: f32,
    pub last_char1_grad_signal: f32,
    pub last_word_loss: f32,
    pub step: usize,

    /// Cross-word context ablation applied during `forward_over` (probing only).
    pub backbone_mode: BackboneMode,
    /// When true, `forward_over`/`backwards_sequence` print encoder & decoder
    /// inputs and targets (one line per word). For debugging only.
    pub trace_io: bool,
}

impl Hierarchical {
    /// `boundary_token_ids` — from `tokenizer.boundary_tokens()`.
    pub fn new(
        char_fwd: Sequential,
        char_bwd: Sequential,
        combine: LinearLayer,
        combine_norm: RMSNorm,
        char2_model: Sequential,
        word_model: Sequential,
        vocab_size: usize,
        boundary_token_ids: Vec<u16>,
        tokenizer: Rc<Tokenizer>,
    ) -> Self {
        let context_size = word_model.output_size;
        let char_out = char_fwd.output_size;

        assert_eq!(
            char2_model.output_size, vocab_size,
            "decoder.output_size must equal vocab_size"
        );
        assert_eq!(
            combine.output_size(),
            word_model.input_size,
            "combine.output must equal backbone.input_size"
        );
        assert_eq!(
            char2_model.input_size,
            context_size + vocab_size,
            "decoder.input_size must equal backbone.output_size + vocab_size"
        );

        let w_token = tokenizer.w_token();
        // The encoder validates its own dimension constraints (fwd/bwd in/out,
        // combine in/out) against vocab_size and char_out.
        let encoder = BiEncoder::new(
            char_fwd,
            char_bwd,
            combine,
            combine_norm,
            vocab_size,
            w_token,
        );

        let char2_input = vec![0.0; vocab_size + context_size].into();

        // delta_buf holds a layer's input_grad as the backward delta flows down a
        // stack, so it must be sized to the LARGEST layer dimension across the
        // backbone and decoder (plus char_out / context for the seam buffers).
        let max_layer_dim = |m: &Sequential| {
            m.layers
                .iter()
                .map(|l| l.output_size().max(l.input_size()))
                .max()
                .unwrap_or(0)
        };
        let buf_size = max_layer_dim(&word_model)
            .max(max_layer_dim(&char2_model))
            .max(char_out)
            .max(context_size);
        let delta_buf = vec![0.0; buf_size].into();

        Self {
            encoder,
            char2_model,
            word_model,
            vocab_size,
            context_size,
            tokenizer,
            w_token,
            boundary_token_ids,
            word_segments: Vec::new(),
            dec_ranges: Vec::new(),
            dec_targets: Vec::new(),
            char2_input,
            d_ew_buf: vec![0.0; char_out].into(),
            delta_buf,
            d_o_buf: vec![0.0; context_size].into(),
            last_high_grad_signal: 0.0,
            last_char1_grad_signal: 0.0,
            last_word_loss: 0.0,
            step: 0,
            backbone_mode: BackboneMode::Normal,
            trace_io: false,
        }
    }

    /// `word_steps` = max WORDS per window (backbone unroll length);
    /// `max_tokens` = max TOKEN span of any window. Encoder/decoder caches are
    /// indexed by a running step cursor, so they need one slot per token plus
    /// one extra `[W]` step per word → size by token capacity, not word count.
    pub fn make_cache(&mut self, word_steps: usize, max_tokens: usize) {
        let slots = max_tokens + word_steps + 8;
        self.encoder.make_cache(slots, word_steps);
        self.char2_model.make_cache(slots);
        self.word_model.make_cache(word_steps.max(1));
        self.dec_targets = vec![0; slots];
    }

    pub fn reset(&mut self) {
        self.encoder.reset();
        for layer in &mut self.char2_model.layers {
            layer.reset_state();
        }
        for layer in &mut self.word_model.layers {
            layer.reset_state();
        }
        self.word_segments.clear();
        self.dec_ranges.clear();
    }

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

    pub fn forward_over(&mut self, tokens: &[u16], words: &[Range<usize>]) {
        self.word_segments.clear();
        self.word_segments.extend_from_slice(words);
        self.encoder.reset();
        self.dec_ranges.clear();
        let n = self.word_segments.len();
        let vocab = self.vocab_size;
        let w_tok = self.w_token as usize;
        let char_out = self.encoder.output_size;

        let mut dec_cursor = 0;

        for w in 0..n.saturating_sub(1) {
            let word = self.word_segments[w]; // encoder word
            let next_word = self.word_segments[w + 1]; // decoder word

            // --- ENCODER: bidirectional encode of word w → e_w ---
            self.encoder.encode_word(w, tokens, word);

            // --- BACKBONE: consume e_w → o_{w+1} (context for the next word) ---
            // Probe: drop cross-word recurrent state so o reflects only this word.
            if self.backbone_mode == BackboneMode::ResetEachWord {
                for layer in &mut self.word_model.layers {
                    layer.reset_state();
                }
            }
            self.delta_buf[..char_out].copy_from_slice(self.encoder.e_w(w));
            Self::forward_step(
                &mut self.word_model.layers,
                &mut self.word_model.cache[w],
                &self.delta_buf[..char_out],
            );

            // --- DECODER: chars of word w+1, conditioned on o_{w+1} ---
            // inputs  [W], c1, …, cn   →   targets c1, …, cn, [W]
            {
                let o = self.word_model.cache[w].last().unwrap().output();
                self.char2_input[vocab..].copy_from_slice(o);
            }
            // Probe: cut the context path entirely (within-word floor).
            if self.backbone_mode == BackboneMode::ZeroContext {
                self.char2_input[vocab..].fill(0.0);
            }
            for layer in &mut self.char2_model.layers {
                layer.reset_state();
            }

            let dec_start = dec_cursor;
            let dword_len = next_word.end - next_word.start;
            let dec_end = dec_start + dword_len + 1; // +1 for the [W] EOS step
            dec_cursor = dec_end;
            self.dec_ranges.push(Range {
                start: dec_start,
                end: dec_end,
            });

            for k in 0..=dword_len {
                let in_tok = if k == 0 {
                    w_tok
                } else {
                    tokens[next_word.start + k - 1] as usize
                };
                let target = if k < dword_len {
                    tokens[next_word.start + k]
                } else {
                    self.w_token
                };
                let slot = dec_start + k;
                self.char2_input[in_tok] = 1.0;
                Self::forward_step(
                    &mut self.char2_model.layers,
                    &mut self.char2_model.cache[slot],
                    &self.char2_input,
                );
                self.char2_input[in_tok] = 0.0;
                self.dec_targets[slot] = target;
            }

            if self.trace_io {
                self.trace_word(w, tokens, word, next_word);
            }
        }
    }

    /// Debug: print the encoder inputs (both directions) and the decoder
    /// input/target strings for word `w`. Active only when `trace_io` is set.
    fn trace_word(&self, w: usize, tokens: &[u16], word: Range<usize>, next_word: Range<usize>) {
        let wm = self.tokenizer.display(self.w_token);
        let enc_fwd = format!(
            "{wm}{}",
            self.tokenizer.display_tokens(&tokens[word.start..word.end])
        );
        let mut rev: Vec<u16> = tokens[word.start..word.end].to_vec();
        rev.reverse();
        let enc_bwd = format!("{}{wm}", self.tokenizer.display_tokens(&rev));
        let dec_in = format!(
            "{wm}{}",
            self.tokenizer
                .display_tokens(&tokens[next_word.start..next_word.end - 1])
        );
        let dec_tgt = format!(
            "{}{wm}",
            self.tokenizer
                .display_tokens(&tokens[next_word.start..next_word.end])
        );
        println!(
            "fwd[{w:>3}] enc.fwd {enc_fwd:?} | enc.bwd {enc_bwd:?} | dec in {dec_in:?} → tgt {dec_tgt:?}"
        );
    }

    pub fn backwards_sequence(&mut self) {
        let vocab = self.vocab_size;
        let context = self.context_size;
        let char_out = self.encoder.output_size;
        let words = self.encoder.num_words(); // = number of decode steps (n - 1)

        let mut high_grad_accum = 0.0;
        let mut char1_grad_accum = 0.0;

        // Walk words in reverse so the backbone's cross-word BPTT state carries.
        for w in (0..words).rev() {
            let dec_range = self.dec_ranges[w];

            if self.trace_io {
                let tgt: Vec<u16> = dec_range
                    .into_iter()
                    .map(|slot| self.dec_targets[slot])
                    .collect();
                println!(
                    "bwd[{w:>3}] dec tgt {:?}",
                    self.tokenizer.display_tokens(&tgt)
                );
            }

            // --- DECODER backward; accumulate grad w.r.t. the word context o ---
            self.d_o_buf.fill(0.0);
            for slot in dec_range.into_iter().rev() {
                let out_len = {
                    let out = self.char2_model.cache[slot].last().unwrap().output();
                    let len = out.len();
                    self.delta_buf[..len].copy_from_slice(out);
                    len
                };
                softmax_inplace(&mut self.delta_buf[..out_len]);
                self.delta_buf[self.dec_targets[slot] as usize] -= 1.0;

                backward_through_layers(
                    &mut self.char2_model.layers,
                    &mut self.char2_model.cache[slot],
                    &mut self.delta_buf,
                    out_len,
                );

                // Gradient w.r.t. the concatenated word context.
                let dx = self.char2_model.cache[slot][0].input_grad();
                for i in 0..context {
                    self.d_o_buf[i] += dx[vocab + i];
                }
            }
            for layer in &mut self.char2_model.layers {
                layer.accumulate_init_grad();
                layer.reset_bptt_state();
            }
            high_grad_accum += self.d_o_buf.iter().map(|x| x.abs()).sum::<f32>() / context as f32;

            // --- BACKBONE backward → grad w.r.t. the word embedding e_w ---
            self.delta_buf[..context].copy_from_slice(&self.d_o_buf);
            backward_through_layers(
                &mut self.word_model.layers,
                &mut self.word_model.cache[w],
                &mut self.delta_buf,
                context,
            );
            {
                let de = self.word_model.cache[w][0].input_grad();
                self.d_ew_buf[..char_out].copy_from_slice(&de[..char_out]);
            }

            // --- ENCODER backward (combine split + both directions' BPTT) ---
            char1_grad_accum += self.encoder.backward_word(w, &self.d_ew_buf[..char_out]);
        }

        for layer in &mut self.word_model.layers {
            layer.accumulate_init_grad();
            layer.reset_bptt_state();
        }

        let denom = words.max(1) as f32;
        self.last_high_grad_signal = high_grad_accum / denom;
        self.last_char1_grad_signal = char1_grad_accum / denom;
    }

    /// Average NLL per decoded word (chars + the trailing `[W]` summed, then
    /// averaged over words).
    fn compute_word_loss(&self) -> f32 {
        let last = self.char2_model.layers.len() - 1;
        let mut total = 0.0;
        for &range in &self.dec_ranges {
            let mut word_nll = 0.0;
            for slot in range {
                let probs = softmax(self.char2_model.cache[slot][last].output());
                word_nll -= (probs[self.dec_targets[slot] as usize] + 1e-12).ln();
            }
            total += word_nll;
        }
        total / self.dec_ranges.len().max(1) as f32
    }

    /// Mean per-step decode cross-entropy across every recorded decoder step.
    fn decode_loss(&self) -> f32 {
        let last = self.char2_model.layers.len() - 1;
        let mut total = 0.0;
        let mut count = 0;
        for &range in &self.dec_ranges {
            for slot in range {
                let probs = softmax(self.char2_model.cache[slot][last].output());
                total -= (probs[self.dec_targets[slot] as usize] + 1e-12).ln();
                count += 1;
            }
        }
        total / count.max(1) as f32
    }

    /// Summed decoder NLL (nats) and step count over the currently cached window.
    fn decode_loss_sum(&self) -> (f32, usize) {
        let last = self.char2_model.layers.len() - 1;
        let mut total = 0.0;
        let mut count = 0;
        for &range in &self.dec_ranges {
            for slot in range {
                let probs = softmax(self.char2_model.cache[slot][last].output());
                total -= (probs[self.dec_targets[slot] as usize] + 1e-12).ln();
                count += 1;
            }
        }
        (total, count)
    }

    /// Forward-only evaluation: runs `forward_over` (under the current
    /// `backbone_mode`) over every window and returns mean per-token decode
    /// cross-entropy in nats. No backward, no weight updates.
    pub fn eval_decode_loss<'a, I: Iterator<Item = WordBatch<'a>>>(&mut self, data: I) -> f32 {
        let mut total = 0.0;
        let mut count = 0;
        for batch in data {
            let WordBatch { tokens, words } = batch;
            self.reset();
            self.forward_over(tokens, &words);
            if self.word_segments.len() < 2 {
                continue;
            }
            let (t, c) = self.decode_loss_sum();
            total += t;
            count += c;
        }
        total / count.max(1) as f32
    }

    /// Per-backbone-block, per-head mean effective forget multiplier (`f_prime`)
    /// over the words of the currently cached window. `f_prime ∈ (0,1]` is the
    /// fraction of the cell state carried from the previous word, so it directly
    /// bounds how far memory can persist. Outer index = mLSTM block, inner = head.
    /// Run `forward_over` first; reads the live forward cache.
    pub fn backbone_forget_per_head(&self) -> Vec<Vec<f32>> {
        let steps = self.encoder.num_words(); // backbone steps = decoded words
        let block_layers: Vec<usize> = self
            .word_model
            .layers
            .iter()
            .enumerate()
            .filter(|(_, l)| l.as_any().is::<MLSTMBlock>())
            .map(|(l, _)| l)
            .collect();

        let mut out: Vec<Vec<f32>> = Vec::new();
        for &l in &block_layers {
            let mut head_sums: Vec<f64> = Vec::new();
            for w in 0..steps {
                let cache = self.word_model.cache[w][l]
                    .as_any()
                    .downcast_ref::<MLSTMBlockCache>()
                    .expect("backbone layer flagged as MLSTMBlock but cache is not");
                let fp = &cache.cell.f_prime;
                if head_sums.is_empty() {
                    head_sums = vec![0.0; fp.len()];
                }
                for (h, &v) in fp.iter().enumerate() {
                    head_sums[h] += v as f64;
                }
            }
            let denom = steps.max(1) as f64;
            out.push(head_sums.iter().map(|s| (s / denom) as f32).collect());
        }
        out
    }

    pub fn train<'a, I: Iterator<Item = WordBatch<'a>>>(
        &mut self,
        data: I,
        state: &mut TrainingState,
    ) {
        let mut tokens = 0;
        let mut time = Instant::now();

        for batch in data {
            let WordBatch {
                tokens: window,
                words,
            } = batch;
            self.reset();
            self.forward_over(window, &words);

            // Word 0 is the given prefix (encode-only); skip degenerate windows
            // that contain no decoded word.
            if self.word_segments.len() < 2 {
                self.encoder.reset_bptt();
                for layer in &mut self.char2_model.layers {
                    layer.reset_bptt_state();
                }
                for layer in &mut self.word_model.layers {
                    layer.reset_bptt_state();
                }
                continue;
            }

            let loss = self.decode_loss();
            let word_loss = self.compute_word_loss();
            self.last_word_loss = word_loss;
            tokens += window.len();

            self.backwards_sequence();
            state.log_tokens(window.len());
            state.log_metric("delta_word", self.last_high_grad_signal);
            state.log_metric("delta_char1", self.last_char1_grad_signal);
            state.log_metric("word_loss", word_loss);

            if let Some(lr) = state.step(loss) {
                self.encoder.sgd_step(lr);
                self.char2_model.sgd_step(lr);
                self.word_model.sgd_step(lr);
            }
            self.step = state.step;

            if state.print() {
                let loss = state.get_loss();
                let elapsed = time.elapsed();
                println!(
                    "{} | char loss {:.4} | ppl {:.4} | word loss {:.4} | word ppl {:.1} | high ∇ {:.4} | char1 ∇ {:.4} | {} tok | {:.1?}",
                    state.step,
                    loss,
                    loss.exp(),
                    self.last_word_loss,
                    self.last_word_loss.exp(),
                    self.last_high_grad_signal,
                    self.last_char1_grad_signal,
                    tokens,
                    elapsed,
                );
                tokens = 0;
                time = Instant::now();
            }
            if state.save() {
                match self.save(state.save_path()) {
                    Ok(()) => println!("saved"),
                    Err(e) => eprintln!("save failed: {e}"),
                }
            }
        }
    }

    /// Bidirectionally encode one completed word (`[W]` prepended) and step the
    /// backbone, leaving the new word context in `char2_input[vocab..]`.
    fn encode_word_advance(&mut self, word: &[u16]) {
        let vocab = self.vocab_size;
        let char_out = self.encoder.output_size;

        // Encode the word → e_w, then step the backbone once on it.
        let e_w = self.encoder.encode_word_sample(word);
        self.delta_buf[..char_out].copy_from_slice(e_w);
        Self::forward_sample_step(
            &mut self.word_model.layers,
            &mut self.word_model.cache[0],
            &self.delta_buf[..char_out],
        );

        let o = self.word_model.cache[0].last().unwrap().output();
        self.char2_input[vocab..].copy_from_slice(o);
    }

    pub fn sample(
        &mut self,
        prefix: &[u16],
        max_len: usize,
        temperature: f32,
        mut callback: impl FnMut(u16) -> bool,
    ) -> Vec<u16> {
        if !self.encoder.cache_ready() {
            self.make_cache(1, MAX_SEQ_LEN);
        }

        self.reset();
        let vocab = self.vocab_size;
        let w_tok = self.w_token as usize;

        // --- Bootstrap the backbone from the prefix --------------------------
        // Encode each COMPLETE word of the prefix (boundary-terminated) to advance
        // the backbone, exactly like the encode words in training. The decoder only
        // ever decodes the word currently being generated.
        let mut enc_buf: Vec<u16> = Vec::new();
        let mut context_ready = false;
        for &tok in prefix {
            enc_buf.push(tok);
            if self.boundary_token_ids.contains(&tok) {
                self.encode_word_advance(&enc_buf);
                enc_buf.clear();
                context_ready = true;
            }
        }

        // No complete word in the prefix → treat the whole prefix (or a random seed
        // when empty) as the first encode word, so a word context exists.
        if !context_ready {
            if enc_buf.is_empty() {
                enc_buf.push(rand::random_range(0..vocab) as u16);
            }
            self.encode_word_advance(&enc_buf);
            enc_buf.clear();
        }

        // Whatever trailed the last boundary is the teacher-forced start of the
        // word we are about to generate (empty unless the prefix ended mid-word).
        let mut forced = enc_buf.into_iter();

        // --- Generation ------------------------------------------------------
        let mut out = Vec::with_capacity(max_len);
        let mut word_chars: Vec<u16> = Vec::new();
        // Each word starts with the [W] BOS input; the model emits the word's chars
        // then a [W] which marks end-of-word.
        for layer in &mut self.char2_model.layers {
            layer.reset_state();
        }
        let mut next_input = w_tok;

        loop {
            self.char2_input[next_input] = 1.0;
            Self::forward_sample_step(
                &mut self.char2_model.layers,
                &mut self.char2_model.cache[0],
                &self.char2_input,
            );
            self.char2_input[next_input] = 0.0;

            // Teacher-force the known start of the word, then sample the rest.
            let (tok, forced_tok) = match forced.next() {
                Some(t) => (t, true),
                None => {
                    let logits = self.char2_model.cache[0].last().unwrap().output();
                    let scaled: Vec<f32> =
                        logits.iter().map(|&v| v / temperature.max(1e-8)).collect();
                    let probs = softmax(&scaled);
                    (self.sample_from_probs(&probs), false)
                }
            };

            if tok as usize == w_tok {
                // End of word: encode it to advance the backbone, then start the next
                // word from the [W] BOS. (A forced prefix never contains [W].)
                if !word_chars.is_empty() {
                    self.encode_word_advance(&word_chars);
                    word_chars.clear();
                }
                for layer in &mut self.char2_model.layers {
                    layer.reset_state();
                }
                next_input = w_tok;
                continue;
            }

            if !forced_tok {
                out.push(tok);
                if out.len() >= max_len || !callback(tok) {
                    break;
                }
            }
            word_chars.push(tok);
            next_input = tok as usize;
        }

        out
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

    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut buf = Cursor::new(Vec::<u8>::new());
        let w = &mut buf as &mut dyn Write;

        write_u32(w, HIER_MAGIC)?;
        write_u32(w, self.vocab_size as u32)?;
        write_u32(w, self.context_size as u32)?;
        write_u32(w, self.boundary_token_ids.len() as u32)?;
        for &id in &self.boundary_token_ids {
            write_u16(w, id)?;
        }

        self.encoder.char_fwd.write_to(w)?;
        self.encoder.char_bwd.write_to(w)?;
        self.word_model.write_to(w)?;
        self.char2_model.write_to(w)?;
        write_matrix(w, &self.encoder.combine.weights)?;
        write_f32_slice(w, &self.encoder.combine.biases)?;
        write_f32_slice(w, &self.encoder.combine_norm.gamma)?;
        write_u64(w, self.step as u64)?;

        File::create(path)?.write_all(&buf.into_inner())
    }

    pub fn load(path: &str, tokenizer: Rc<Tokenizer>) -> io::Result<Self> {
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

        let char_fwd = Sequential::load_from(r)?;
        let char_bwd = Sequential::load_from(r)?;
        let word_model = Sequential::load_from(r)?;
        let char2_model = Sequential::load_from(r)?;
        let combine_weights = read_matrix(r)?;
        let combine_biases: Box<[f32]> = read_f32_vec(r)?;
        let combine_norm_gamma: Box<[f32]> = read_f32_vec(r)?;
        let step = read_u64(r).unwrap_or(0) as usize;

        let (ci, co) = (combine_weights.rows(), combine_weights.cols());
        let combine = LinearLayer::from_loaded(ci, co, combine_weights, combine_biases);
        let combine_norm = RMSNorm::from_loaded(co, combine_norm_gamma);

        let mut model = Self::new(
            char_fwd,
            char_bwd,
            combine,
            combine_norm,
            char2_model,
            word_model,
            vocab_size,
            boundary_token_ids,
            tokenizer,
        );
        debug_assert_eq!(model.context_size, context_size);
        model.step = step;
        Ok(model)
    }
}

#[cfg(test)]
mod orchestration_tests {
    use super::*;
    use crate::model::build_hierarchical_model;

    /// Build a 4-word input, run the full hierarchical forward, and check that the
    /// word context `o` produced for the LAST word changes when only the FIRST
    /// word's characters change. If it does not, the backbone's recurrent state is
    /// not carried across words by the orchestration (the bug the user suspects).
    #[test]
    fn word_context_depends_on_earlier_words() {
        let tokenizer = Rc::new(Tokenizer::new(crate::config::CHARSET, false));
        let vocab = tokenizer.vocab_size();
        let boundaries = tokenizer.boundary_tokens();
        assert!(!boundaries.is_empty(), "need at least one boundary token");

        let mut model = build_hierarchical_model(vocab, boundaries.clone(), tokenizer.clone());
        model.make_cache(64, MAX_SEQ_LEN);

        let b = boundaries[0];
        // Distinct non-boundary content tokens.
        let content: Vec<u16> = (0..vocab as u16)
            .filter(|t| !boundaries.contains(t))
            .take(8)
            .collect();
        assert!(content.len() >= 5, "need a few non-boundary tokens");

        // Four single-char words, each closed by the boundary `b`.
        // Only word 0's character differs between the two runs.
        let words: Vec<Range<usize>> = vec![
            Range { start: 0, end: 2 },
            Range { start: 2, end: 4 },
            Range { start: 4, end: 6 },
            Range { start: 6, end: 8 },
        ];
        let make = |w0: u16| vec![w0, b, content[1], b, content[2], b, content[3], b];

        let run = |model: &mut Hierarchical, toks: &[u16]| -> Vec<f32> {
            model.reset();
            model.forward_over(toks, &words);
            // n = 4 → last backbone step is cache[n-2] = cache[2], the context that
            // conditions the decode of word 3.
            model.word_model.cache[2].last().unwrap().output().to_vec()
        };

        let o_a = run(&mut model, &make(content[0]));
        let o_b = run(&mut model, &make(content[4]));

        let mean_diff: f32 = o_a
            .iter()
            .zip(&o_b)
            .map(|(a, c)| (a - c).abs())
            .sum::<f32>()
            / o_a.len() as f32;
        println!("mean |Δo| at word 3 from changing word 0 only = {mean_diff:.6e}");

        assert!(
            mean_diff > 1e-6,
            "context of the last word is independent of the first word → \
             backbone recurrence is NOT carried across words in forward_over"
        );
    }

    /// Reproduce the hierarchical decode pattern in isolation: a constant raw
    /// input fed at EVERY step through sLSTM blocks, backward'd per-step with the
    /// same `backward_through_layers` + accumulate_init_grad/reset_bptt epilogue
    /// the decoder uses. Grad-check layer 0, whose input is active at every step.
    ///
    /// Regression guard for the LSTM m-stabilizer BPTT bug: the sLSTM backward
    /// used to treat the stabilizer `m` as constant, which is only exact while
    /// psi=max(|n|,1) is set by |n|>1. Early in a segment |n| is small, psi clamps
    /// to 1, and the dropped m-recurrence gradient corrupted input/recurrent grads
    /// (worst at the first timestep). Fixed by carrying `dm_bptt` in the sLSTM.
    #[test]
    fn const_input_through_slstm_grad() {
        use crate::nn::softmax::softmax;
        use crate::nn_layer::SequentialBuilder;
        use crate::optimizers::GradMatrixOps;

        let d = 16usize;
        let h = 24usize;
        let mut m = SequentialBuilder::new(d)
            .linear(h)
            .slstm_block(h)
            .slstm_block(h)
            .rms_norm()
            .linear(d)
            .build();
        let steps = 4usize;
        m.make_cache(steps);
        let x: Vec<f32> = (0..d).map(|k| ((k * 13 % 7) as f32 - 3.0) * 0.2).collect();
        let targets: Vec<u16> = (0..steps as u16).map(|t| (t * 3 + 1) % d as u16).collect();

        // Forward: same constant x at every step (like the word context).
        let forward = |m: &mut Sequential| {
            for l in &mut m.layers {
                l.reset_state();
            }
            for t in 0..steps {
                let Sequential { layers, cache, .. } = m;
                Sequential::forward_step(layers, &mut cache[t], &x);
            }
        };

        let summed_loss = |m: &mut Sequential| -> f32 {
            forward(m);
            let last = m.layers.len() - 1;
            let mut loss = 0.0;
            for t in 0..steps {
                let probs = softmax(m.cache[t][last].output());
                loss -= (probs[targets[t] as usize] + 1e-12).ln();
            }
            loss
        };

        // Analytic via the SAME per-step backward the decoder uses.
        forward(&mut m);
        let mut delta = vec![0.0f32; d.max(h)];
        for t in (0..steps).rev() {
            let last = m.layers.len() - 1;
            let out = m.cache[t][last].output();
            let len = out.len();
            delta[..len].copy_from_slice(out);
            softmax_inplace(&mut delta[..len]);
            delta[targets[t] as usize] -= 1.0;
            backward_through_layers(&mut m.layers, &mut m.cache[t], &mut delta, len);
        }
        for l in &mut m.layers {
            l.accumulate_init_grad();
            l.reset_bptt_state();
        }

        let (i, j) = (3usize, 7usize);
        let g_analytic = m.layers[0]
            .as_any_mut()
            .downcast_mut::<LinearLayer>()
            .unwrap()
            .grads
            .weights
            .matrix()[i][j];
        let eps = 1e-3;
        let bump = |m: &mut Sequential, dd: f32| {
            m.layers[0]
                .as_any_mut()
                .downcast_mut::<LinearLayer>()
                .unwrap()
                .weights[i][j] += dd;
        };
        bump(&mut m, eps);
        let lp = summed_loss(&mut m);
        bump(&mut m, -2.0 * eps);
        let lm = summed_loss(&mut m);
        bump(&mut m, eps);
        let g_numeric = (lp - lm) / (2.0 * eps);
        let rel = (g_analytic - g_numeric).abs() / g_analytic.abs().max(g_numeric.abs()).max(1e-6);
        println!(
            "CONST-input sLSTM layer0 w[{i}][{j}]: analytic {g_analytic:.6e}  numeric {g_numeric:.6e}  rel {rel:.3e}"
        );
        assert!(
            rel < 2e-2,
            "constant multi-step input through sLSTM grad mismatch (rel {rel:.3e})"
        );
    }

    /// Control for `const_input_through_slstm_grad`: identical harness, but with
    /// plain LSTM cells in place of the sLSTM blocks. The vanilla LSTM has a
    /// straightforward, well-tested backward (no m-stabilizer), so if this passes
    /// while the sLSTM version fails, the bug is isolated to the sLSTM cell — not the
    /// per-step `backward_through_layers` orchestration nor the grad-check harness.
    #[test]
    fn const_input_through_lstm_grad() {
        use crate::nn::softmax::softmax;
        use crate::nn_layer::SequentialBuilder;
        use crate::optimizers::GradMatrixOps;

        let d = 16usize;
        let h = 24usize;
        let mut m = SequentialBuilder::new(d)
            .linear(h)
            .lstm(h)
            .lstm(h)
            .rms_norm()
            .linear(d)
            .build();
        let steps = 4usize;
        m.make_cache(steps);
        let x: Vec<f32> = (0..d).map(|k| ((k * 13 % 7) as f32 - 3.0) * 0.2).collect();
        let targets: Vec<u16> = (0..steps as u16).map(|t| (t * 3 + 1) % d as u16).collect();

        let forward = |m: &mut Sequential| {
            for l in &mut m.layers {
                l.reset_state();
            }
            for t in 0..steps {
                let Sequential { layers, cache, .. } = m;
                Sequential::forward_step(layers, &mut cache[t], &x);
            }
        };

        let summed_loss = |m: &mut Sequential| -> f32 {
            forward(m);
            let last = m.layers.len() - 1;
            let mut loss = 0.0;
            for t in 0..steps {
                let probs = softmax(m.cache[t][last].output());
                loss -= (probs[targets[t] as usize] + 1e-12).ln();
            }
            loss
        };

        forward(&mut m);
        let mut delta = vec![0.0f32; d.max(h)];
        for t in (0..steps).rev() {
            let last = m.layers.len() - 1;
            let out = m.cache[t][last].output();
            let len = out.len();
            delta[..len].copy_from_slice(out);
            softmax_inplace(&mut delta[..len]);
            delta[targets[t] as usize] -= 1.0;
            backward_through_layers(&mut m.layers, &mut m.cache[t], &mut delta, len);
        }
        for l in &mut m.layers {
            l.accumulate_init_grad();
            l.reset_bptt_state();
        }

        let (i, j) = (3usize, 7usize);
        let g_analytic = m.layers[0]
            .as_any_mut()
            .downcast_mut::<LinearLayer>()
            .unwrap()
            .grads
            .weights
            .matrix()[i][j];
        let eps = 1e-3;
        let bump = |m: &mut Sequential, dd: f32| {
            m.layers[0]
                .as_any_mut()
                .downcast_mut::<LinearLayer>()
                .unwrap()
                .weights[i][j] += dd;
        };
        bump(&mut m, eps);
        let lp = summed_loss(&mut m);
        bump(&mut m, -2.0 * eps);
        let lm = summed_loss(&mut m);
        bump(&mut m, eps);
        let g_numeric = (lp - lm) / (2.0 * eps);
        let rel = (g_analytic - g_numeric).abs() / g_analytic.abs().max(g_numeric.abs()).max(1e-6);
        println!(
            "CONST-input LSTM layer0 w[{i}][{j}]: analytic {g_analytic:.6e}  numeric {g_numeric:.6e}  rel {rel:.3e}"
        );
        assert!(
            rel < 2e-2,
            "constant multi-step input through LSTM grad mismatch (rel {rel:.3e})"
        );
    }

    /// Same as `const_input_through_slstm_grad` but through mLSTM blocks. The
    /// mLSTM's normalizer stays >1 in practice, so its m-as-constant backward is
    /// already accurate — this is a regression guard.
    #[test]
    fn const_input_through_mlstm_grad() {
        use crate::nn::softmax::softmax;
        use crate::nn_layer::SequentialBuilder;
        use crate::optimizers::GradMatrixOps;

        let d = 32usize;
        let heads = 4usize;
        let dqk = 4usize;
        let mut m = SequentialBuilder::new(d)
            .linear(d)
            .mlstm_block(heads, dqk)
            .mlstm_block(heads, dqk)
            .rms_norm()
            .linear(d)
            .build();
        let steps = 6usize;
        m.make_cache(steps);
        let x: Vec<f32> = (0..d).map(|k| ((k * 11 % 9) as f32 - 4.0) * 0.25).collect();
        let targets: Vec<u16> = (0..steps as u16).map(|t| (t * 5 + 1) % d as u16).collect();

        let forward = |m: &mut Sequential| {
            for l in &mut m.layers {
                l.reset_state();
            }
            for t in 0..steps {
                let Sequential { layers, cache, .. } = m;
                Sequential::forward_step(layers, &mut cache[t], &x);
            }
        };
        let summed_loss = |m: &mut Sequential| -> f32 {
            forward(m);
            let last = m.layers.len() - 1;
            let mut loss = 0.0;
            for t in 0..steps {
                let probs = softmax(m.cache[t][last].output());
                loss -= (probs[targets[t] as usize] + 1e-12).ln();
            }
            loss
        };

        forward(&mut m);
        let mut delta = vec![0.0f32; d * 2];
        for t in (0..steps).rev() {
            let last = m.layers.len() - 1;
            let out = m.cache[t][last].output();
            let len = out.len();
            delta[..len].copy_from_slice(out);
            softmax_inplace(&mut delta[..len]);
            delta[targets[t] as usize] -= 1.0;
            backward_through_layers(&mut m.layers, &mut m.cache[t], &mut delta, len);
        }
        for l in &mut m.layers {
            l.accumulate_init_grad();
            l.reset_bptt_state();
        }

        let (i, j) = (3usize, 7usize);
        let g_analytic = m.layers[0]
            .as_any_mut()
            .downcast_mut::<LinearLayer>()
            .unwrap()
            .grads
            .weights
            .matrix()[i][j];
        let bump = |m: &mut Sequential, dd: f32| {
            m.layers[0]
                .as_any_mut()
                .downcast_mut::<LinearLayer>()
                .unwrap()
                .weights[i][j] += dd;
        };
        // eps=1e-2 minimises finite-difference roundoff at this gradient magnitude.
        let eps = 1e-2;
        bump(&mut m, eps);
        let lp = summed_loss(&mut m);
        bump(&mut m, -2.0 * eps);
        let lm = summed_loss(&mut m);
        bump(&mut m, eps);
        let g_numeric = (lp - lm) / (2.0 * eps);
        let rel = (g_analytic - g_numeric).abs() / g_analytic.abs().max(g_numeric.abs()).max(1e-6);
        println!(
            "CONST-input mLSTM layer0 w[{i}][{j}]: analytic {g_analytic:.6e}  numeric {g_numeric:.6e}  rel {rel:.3e}"
        );
        assert!(
            rel < 1e-2,
            "mLSTM const multi-step input grad mismatch (rel {rel:.3e})"
        );
    }

    /// Grad-check a weight BELOW the recurrent blocks in a standalone Sequential
    /// (the model's own backward), to see whether the sLSTM input_grad propagation
    /// is correct in isolation — i.e. whether the bug is in the layers or only in
    /// the hierarchical orchestration.
    #[test]
    fn flat_slstm_below_recurrence_grad() {
        use crate::nn::softmax::softmax;
        use crate::nn_layer::SequentialBuilder;
        use crate::optimizers::GradMatrixOps;

        let vocab = 24usize;
        let h = 32usize;
        let mut model = SequentialBuilder::new(vocab)
            .linear(h)
            .slstm_block(h)
            .slstm_block(h)
            .rms_norm()
            .linear(vocab)
            .build();
        let seq_len = 5;
        model.make_cache(seq_len);
        let tokens: Vec<u16> = (0..seq_len as u16)
            .map(|t| (t * 7 + 2) % vocab as u16)
            .collect();

        let summed_loss = |model: &mut Sequential| -> f32 {
            for l in &mut model.layers {
                l.reset_state();
            }
            model.forward_over(&tokens);
            let last = model.layers.len() - 1;
            let mut loss = 0.0;
            for t in 0..tokens.len() {
                let probs = softmax(model.cache[t][last].output());
                loss -= (probs[tokens[t] as usize] + 1e-12).ln();
            }
            loss
        };

        for l in &mut model.layers {
            l.reset_state();
        }
        model.forward_over(&tokens);
        model.backwards_sequence(&tokens);

        // layer 0 = the input Linear; its gradient flows back through both sLSTMs.
        // Row i must be a token that is actually fed (one-hot input), else grad is 0.
        // TIDX selects WHICH timestep's input_grad we sample (tokens are distinct).
        let tidx: usize = std::env::var("TIDX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let (i, j) = (tokens[tidx] as usize, 7usize);
        let g_analytic = model.layers[0]
            .as_any_mut()
            .downcast_mut::<LinearLayer>()
            .unwrap()
            .grads
            .weights
            .matrix()[i][j];
        let eps = 1e-3;
        let bump = |model: &mut Sequential, d: f32| {
            model.layers[0]
                .as_any_mut()
                .downcast_mut::<LinearLayer>()
                .unwrap()
                .weights[i][j] += d;
        };
        bump(&mut model, eps);
        let lp = summed_loss(&mut model);
        bump(&mut model, -2.0 * eps);
        let lm = summed_loss(&mut model);
        bump(&mut model, eps);
        let g_numeric = (lp - lm) / (2.0 * eps);
        let rel = (g_analytic - g_numeric).abs() / g_analytic.abs().max(g_numeric.abs()).max(1e-6);
        println!(
            "FLAT below-sLSTM layer0 w[{i}][{j}]: analytic {g_analytic:.6e}  numeric {g_numeric:.6e}  rel {rel:.3e}"
        );
        assert!(
            rel < 2e-2,
            "sLSTM input_grad propagation is itself wrong (rel {rel:.3e})"
        );
    }

    /// Harness self-check: grad-check the *flat* model (known-good Sequential
    /// forward/backward) so we trust the finite-difference setup before trusting
    /// its verdict on the hierarchical model.
    #[test]
    fn flat_model_numeric_grad_sanity() {
        use crate::model::build_normal_model;
        use crate::nn::softmax::softmax;
        use crate::optimizers::GradMatrixOps;

        let vocab = 32usize;
        let mut model = build_normal_model(vocab);
        let seq_len = 6;
        model.make_cache(seq_len);
        let tokens: Vec<u16> = (0..seq_len as u16)
            .map(|t| (t * 5 + 1) % vocab as u16)
            .collect();

        // backward differentiates the SUM of per-position cross-entropy.
        let summed_loss = |model: &mut Sequential| -> f32 {
            for l in &mut model.layers {
                l.reset_state();
            }
            model.forward_over(&tokens);
            let last = model.layers.len() - 1;
            let mut loss = 0.0;
            for t in 0..tokens.len() {
                let probs = softmax(model.cache[t][last].output());
                loss -= (probs[tokens[t] as usize] + 1e-12).ln();
            }
            loss
        };

        for l in &mut model.layers {
            l.reset_state();
        }
        model.forward_over(&tokens);
        model.backwards_sequence(&tokens);

        let last = model.layers.len() - 1;
        let (i, j) = (3usize, 7usize);
        let g_analytic = {
            let layer = model.layers[last]
                .as_any_mut()
                .downcast_mut::<LinearLayer>()
                .expect("last layer is Linear");
            layer.grads.weights.matrix()[i][j]
        };
        let eps = 1e-3;
        {
            let layer = model.layers[last]
                .as_any_mut()
                .downcast_mut::<LinearLayer>()
                .unwrap();
            layer.weights[i][j] += eps;
        }
        let lp = summed_loss(&mut model);
        {
            let layer = model.layers[last]
                .as_any_mut()
                .downcast_mut::<LinearLayer>()
                .unwrap();
            layer.weights[i][j] -= 2.0 * eps;
        }
        let lm = summed_loss(&mut model);
        {
            let layer = model.layers[last]
                .as_any_mut()
                .downcast_mut::<LinearLayer>()
                .unwrap();
            layer.weights[i][j] += eps;
        }
        let g_numeric = (lp - lm) / (2.0 * eps);
        let rel = (g_analytic - g_numeric).abs() / g_analytic.abs().max(g_numeric.abs()).max(1e-6);
        println!(
            "FLAT last-linear w[{i}][{j}]: analytic {g_analytic:.6e}  numeric {g_numeric:.6e}  rel {rel:.3e}"
        );
        assert!(
            rel < 2e-2,
            "harness/flat-model grad mismatch (rel {rel:.3e})"
        );
    }

    /// Numeric gradient check across the whole encoder→backbone→decoder chain.
    ///
    /// Builds a hierarchical model from cells with EXACT backward (LSTM encoder +
    /// decoder, mLSTM backbone) so the finite-difference verdict is not polluted by
    /// the sLSTM cell's known-inexact stabilizer backward. Grad-checks one weight in
    /// each orchestration path — the cross-word backbone input projection, the
    /// combine head (encoder readout merge), and both encoder directions — against a
    /// central finite difference of the summed decode loss. A match proves the new
    /// bidirectional encoder readout split, the `[W]` decode steps, and the
    /// cross-word backbone BPTT are all consistent.
    #[test]
    fn backward_matches_numeric_grad_across_words() {
        use crate::nn::softmax::softmax;
        use crate::nn_layer::SequentialBuilder;
        use crate::optimizers::GradMatrixOps;
        use iron_oxide::collections::Matrix;

        // Small dims, all exact-backward cells.
        let tokenizer = Rc::new(Tokenizer::new(crate::config::CHARSET, false));
        let vocab = tokenizer.vocab_size();
        let boundaries = tokenizer.boundary_tokens();
        let h = 16usize; // encoder hidden + combine output + backbone input
        let oh = 16usize; // backbone output (context)
        let wh = 32usize; // backbone inner width
        let heads = 4usize;
        let dqk = wh / heads;

        let enc = || {
            SequentialBuilder::new(vocab)
                .embedding(h)
                .lstm(h)
                .lstm(h)
                .build()
        };
        let char_fwd = enc();
        let char_bwd = enc();
        let combine = LinearLayer::new(2 * h, h);
        let combine_norm = RMSNorm::new(h);
        let word_model = SequentialBuilder::new(h)
            .linear(wh)
            .mlstm_block(heads, dqk)
            .linear(oh)
            .rms_norm()
            .build();
        let char2_model = SequentialBuilder::new(oh + vocab)
            .linear(oh)
            .lstm(oh)
            .lstm(oh)
            .linear(oh)
            .rms_norm()
            .linear_no_bias(vocab)
            .build();
        let mut model = Hierarchical::new(
            char_fwd,
            char_bwd,
            combine,
            combine_norm,
            char2_model,
            word_model,
            vocab,
            boundaries.clone(),
            tokenizer.clone(),
        );
        model.make_cache(64, MAX_SEQ_LEN);

        // 4 two-token words → three decoded words, exercising cross-word backbone BPTT.
        let b = boundaries[0];
        let content: Vec<u16> = (0..vocab as u16)
            .filter(|t| !boundaries.contains(t))
            .take(8)
            .collect();
        let mut tokens = Vec::new();
        let mut words: Vec<Range<usize>> = Vec::new();
        for k in 0..4 {
            let start = tokens.len();
            tokens.push(content[k % content.len()]);
            tokens.push(b);
            words.push(Range {
                start,
                end: start + 2,
            });
        }

        // Loss accumulated in f64: the central finite difference subtracts two large,
        // nearly-equal sums, so an f32 accumulator loses the signal for the small
        // gradients deep in the net (combine / encoder). The forward is still f32, so
        // we grad-check each path at its LARGEST-magnitude weight (best SNR).
        let summed_loss = |model: &mut Hierarchical| -> f64 {
            model.reset();
            model.forward_over(&tokens, &words);
            let last = model.char2_model.layers.len() - 1;
            let mut loss = 0.0f64;
            for range in model.dec_ranges.clone() {
                for slot in range.start..range.end {
                    let probs = softmax(model.char2_model.cache[slot][last].output());
                    loss -= ((probs[model.dec_targets[slot] as usize] as f64) + 1e-12).ln();
                }
            }
            loss
        };

        // Analytic gradients via the real backward.
        model.reset();
        model.forward_over(&tokens, &words);
        model.backwards_sequence();

        let eps = 1e-2;
        let argmax = |m: &Matrix| -> (usize, usize) {
            let mut best = (0, 0);
            let mut bv = -1.0;
            for r in 0..m.rows() {
                for c in 0..m.cols() {
                    let v = m[r][c].abs();
                    if v > bv {
                        bv = v;
                        best = (r, c);
                    }
                }
            }
            best
        };

        // Each tuple names an orchestration path and a `pick` to its weight + grad.
        // word/0 : backbone input proj (shared every word → cross-word backbone BPTT)
        // word/2 : backbone output proj (top of the backbone, above the recurrence)
        // combine: bidirectional readout merge (fwd@[W] ‖ bwd@[W] → e_w)
        // char2/3: decoder linear above the recurrence (the [W] BOS/EOS decode steps)
        // char_fwd / char_bwd embedding: encoder internal backward + readout slot
        use crate::nn::embedding::EmbeddingLayer;
        type Pick = fn(&mut Hierarchical) -> (&mut Matrix, &mut Matrix); // (weights, grad)
        fn lin(l: &mut Box<dyn NnLayer>) -> (&mut Matrix, &mut Matrix) {
            let l = l.as_any_mut().downcast_mut::<LinearLayer>().unwrap();
            (&mut l.weights, l.grads.weights.matrix())
        }
        let cases: [(&str, Pick); 6] = [
            ("word/0", |m| lin(&mut m.word_model.layers[0])),
            ("word/2", |m| lin(&mut m.word_model.layers[2])),
            ("combine", |m| {
                (
                    &mut m.encoder.combine.weights,
                    m.encoder.combine.grads.weights.matrix(),
                )
            }),
            ("char2/3", |m| lin(&mut m.char2_model.layers[3])),
            ("char_fwd/emb", |m| {
                let e = m.encoder.char_fwd.layers[0]
                    .as_any_mut()
                    .downcast_mut::<EmbeddingLayer>()
                    .unwrap();
                (&mut e.weights, e.grads.weights.matrix())
            }),
            ("char_bwd/emb", |m| {
                let e = m.encoder.char_bwd.layers[0]
                    .as_any_mut()
                    .downcast_mut::<EmbeddingLayer>()
                    .unwrap();
                (&mut e.weights, e.grads.weights.matrix())
            }),
        ];

        for (label, pick) in cases {
            let (i, j) = {
                let (_, g) = pick(&mut model);
                argmax(g)
            };
            let g_analytic = {
                let (_, g) = pick(&mut model);
                g[i][j]
            };
            let bump = |model: &mut Hierarchical, d: f32| {
                let (w, _) = pick(model);
                w[i][j] += d;
            };
            bump(&mut model, eps);
            let l_plus = summed_loss(&mut model);
            bump(&mut model, -2.0 * eps);
            let l_minus = summed_loss(&mut model);
            bump(&mut model, eps); // restore
            let g_numeric = ((l_plus - l_minus) / (2.0 * eps as f64)) as f32;
            let rel =
                (g_analytic - g_numeric).abs() / g_analytic.abs().max(g_numeric.abs()).max(1e-6);
            println!(
                "{label} w[{i}][{j}]: analytic {g_analytic:.6e}  numeric {g_numeric:.6e}  rel {rel:.3e}"
            );
            assert!(
                rel < 5e-2,
                "{label}: analytic vs numeric gradient disagree (rel {rel:.3e})"
            );
        }
    }

    /// Build a small hierarchical model (LSTM encoder/decoder, mLSTM backbone) for
    /// fast end-to-end smoke tests.
    fn build_small(tokenizer: Rc<Tokenizer>) -> Hierarchical {
        use crate::nn_layer::SequentialBuilder;
        let vocab = tokenizer.vocab_size();
        let (h, oh, wh, heads) = (16usize, 16usize, 32usize, 4usize);
        let enc = || {
            SequentialBuilder::new(vocab)
                .embedding(h)
                .lstm(h)
                .lstm(h)
                .build()
        };
        let word_model = SequentialBuilder::new(h)
            .linear(wh)
            .mlstm_block(heads, wh / heads)
            .linear(oh)
            .rms_norm()
            .build();
        let char2_model = SequentialBuilder::new(oh + vocab)
            .linear(oh)
            .lstm(oh)
            .lstm(oh)
            .linear(oh)
            .rms_norm()
            .linear_no_bias(vocab)
            .build();
        Hierarchical::new(
            enc(),
            enc(),
            LinearLayer::new(2 * h, h),
            RMSNorm::new(h),
            char2_model,
            word_model,
            vocab,
            tokenizer.boundary_tokens(),
            tokenizer.clone(),
        )
    }

    /// End-to-end: a few SGD steps on a fixed short corpus must reduce the decode
    /// loss; saving and reloading the new HIER format must preserve the model
    /// (identical loss); sampling must run, terminate words on the internal `[W]`
    /// and never emit it.
    fn segment(toks: &[u16], boundaries: &[u16]) -> Vec<Range<usize>> {
        let mut words = Vec::new();
        let mut start = 0;
        for (t, tok) in toks.iter().enumerate() {
            if boundaries.contains(tok) {
                words.push(Range { start, end: t + 1 });
                start = t + 1;
            }
        }
        if start < toks.len() {
            words.push(Range {
                start,
                end: toks.len(),
            });
        }
        words
    }

    #[test]
    fn train_save_load_sample_roundtrip() {
        let tokenizer = Rc::new(Tokenizer::new(crate::config::CHARSET, false));
        let boundaries = tokenizer.boundary_tokens();
        let mut model = build_small(tokenizer.clone());
        model.make_cache(64, MAX_SEQ_LEN);

        let tokens = tokenizer.to_tokens("the cat sat on the mat. the cat ran. ");
        let words = segment(&tokens, &boundaries);
        assert!(words.len() >= 4);

        let step = |m: &mut Hierarchical| {
            m.reset();
            m.forward_over(&tokens, &words);
            let loss = m.decode_loss();
            m.backwards_sequence();
            let lr = 0.05;
            m.encoder.sgd_step(lr);
            m.char2_model.sgd_step(lr);
            m.word_model.sgd_step(lr);
            loss
        };

        let loss0 = step(&mut model);
        for _ in 0..40 {
            step(&mut model);
        }
        model.reset();
        model.forward_over(&tokens, &words);
        let loss_final = model.decode_loss();
        println!("roundtrip: loss {loss0:.4} → {loss_final:.4}");
        assert!(loss_final < loss0 * 0.9, "training did not reduce loss");

        // Save → load → identical decode loss.
        let path = std::env::temp_dir().join("hier_roundtrip_test.bin");
        let path = path.to_str().unwrap();
        model.save(path).unwrap();
        let mut reloaded = Hierarchical::load(path, tokenizer.clone()).unwrap();
        reloaded.make_cache(64, MAX_SEQ_LEN);
        reloaded.reset();
        reloaded.forward_over(&tokens, &words);
        let loss_reloaded = reloaded.decode_loss();
        assert!(
            (loss_reloaded - loss_final).abs() < 1e-3,
            "save/load changed the model: {loss_final} vs {loss_reloaded}"
        );

        // Sampling runs, terminates, and never emits the internal [W] marker.
        let prefix = tokenizer.to_tokens("the ");
        let out = reloaded.sample(&prefix, 80, 0.7, |_| true);
        assert!(!out.is_empty(), "sampler produced nothing");
        assert!(
            !out.contains(&tokenizer.w_token()),
            "sampler emitted the internal [W] marker"
        );
        std::fs::remove_file(path).ok();
    }
}

pub(crate) fn backward_through_layers(
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
