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
        linear_nb::LinearNBLayer,
        mlstm_block::{MLSTMBlock, MLSTMBlockCache},
        rms_norm::RMSNorm,
        softmax::{nll_from_logits, softmax, softmax_inplace},
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

    /// Backward scratch: top-level delta as it flows down a layer stack.
    delta_buf: Box<[f32]>,
    /// Scratch accumulator for one word's gradient w.r.t. the decoder context.
    d_o_buf: Box<[f32]>,

    /// Phase seam (decoder→backbone): grad w.r.t. each word's context `o`
    /// (size `context`), one entry per decoded word. Filled by the decoder
    /// backward phase, consumed by the backbone backward phase.
    d_o_words: Vec<Box<[f32]>>,
    /// Phase seam (backbone→encoder): grad w.r.t. each word embedding `e_w`
    /// (size `char_out`), one entry per decoded word. Filled by the backbone
    /// backward phase, consumed by the encoder backward phase.
    d_ew_words: Vec<Box<[f32]>>,

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
        combine: LinearNBLayer,
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
            delta_buf,
            d_o_buf: vec![0.0; context_size].into(),
            d_o_words: Vec::new(),
            d_ew_words: Vec::new(),
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

        // Per-word phase-seam buffers (one slot per decodable word).
        let words = word_steps.max(1);
        let char_out = self.encoder.output_size;
        self.d_o_words = (0..words)
            .map(|_| vec![0.0; self.context_size].into())
            .collect();
        self.d_ew_words = (0..words).map(|_| vec![0.0; char_out].into()).collect();

        // Pre-size the per-window index Vecs so `forward_over` never reallocates
        // them in the training loop (it only clear()s and re-fills them).
        self.word_segments = Vec::with_capacity(words + 1);
        self.dec_ranges = Vec::with_capacity(words);
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

    /// Run the full hierarchical forward over a window, **phase by phase**: all
    /// encoder steps first, then all backbone steps, then all decoder steps. This
    /// is numerically identical to interleaving per word (the three stages have no
    /// within-phase cross-dependencies that order would change), but it groups the
    /// work the way a GPU wants it: encoder words are mutually independent, the
    /// backbone is a single recurrent sweep, and decoder words are independent too.
    pub fn forward_over(&mut self, tokens: &[u16], words: &[Range<usize>]) {
        self.word_segments.clear();
        self.word_segments.extend_from_slice(words);
        self.encoder.reset();
        self.dec_ranges.clear();
        let n = self.word_segments.len();
        let vocab = self.vocab_size;
        let w_tok = self.w_token as usize;
        let char_out = self.encoder.output_size;

        // Word 0 is the encode-only prefix; words 1..n are decoded, so word w is
        // encoded and the backbone turns it into the context for word w+1.
        let decode_words = n.saturating_sub(1);

        // --- PHASE 1: ENCODER — bidirectionally encode every word w → e_w. ---
        // Each word resets its own state, so the words are fully independent.
        for w in 0..decode_words {
            self.encoder.encode_word(w, tokens, self.word_segments[w]);
        }

        // --- PHASE 2: BACKBONE — autoregress e_0 … e_{n-2}, carrying recurrent
        // state across words. Step w consumes e_w and emits o_{w+1}. ---
        for w in 0..decode_words {
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
        }

        // --- PHASE 3: DECODER — predict the chars of word w+1, conditioned on
        // o_{w+1}. Each word's decode is reset and independent of the others. ---
        // inputs  [W], c1, …, cn   →   targets c1, …, cn, [W]
        let mut dec_cursor = 0;
        for w in 0..decode_words {
            let next_word = self.word_segments[w + 1];

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
                self.trace_word(w, tokens, self.word_segments[w], next_word);
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

    /// Backward of the whole window, mirroring `forward_over`'s phase split:
    /// decoder backward for every word, then a single reverse backbone sweep,
    /// then encoder backward for every word. The two phase seams (`d_o_words`,
    /// `d_ew_words`) hold the per-word gradients passed between stages. Only the
    /// backbone is order-sensitive (reverse, so its cross-word BPTT state carries);
    /// the decoder and encoder phases are per-word independent and batchable.
    pub fn backwards_sequence(&mut self) {
        let vocab = self.vocab_size;
        let context = self.context_size;
        let char_out = self.encoder.output_size;
        let words = self.encoder.num_words(); // = number of decode steps (n - 1)

        let mut high_grad_accum = 0.0;
        let mut char1_grad_accum = 0.0;

        // --- PHASE 3': DECODER backward → grad w.r.t. each word's context o. ---
        // Words decode independently, so order is free; stash each d_o per word.
        for w in 0..words {
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
            self.d_o_words[w][..context].copy_from_slice(&self.d_o_buf[..context]);
        }

        // --- PHASE 2': BACKBONE backward → grad w.r.t. each word embedding e_w. ---
        // Reverse word order so the backbone's cross-word BPTT state carries.
        for w in (0..words).rev() {
            self.delta_buf[..context].copy_from_slice(&self.d_o_words[w][..context]);
            backward_through_layers(
                &mut self.word_model.layers,
                &mut self.word_model.cache[w],
                &mut self.delta_buf,
                context,
            );
            let de = self.word_model.cache[w][0].input_grad();
            self.d_ew_words[w][..char_out].copy_from_slice(&de[..char_out]);
        }
        for layer in &mut self.word_model.layers {
            layer.accumulate_init_grad();
            layer.reset_bptt_state();
        }

        // --- PHASE 1': ENCODER backward (combine split + both directions' BPTT). ---
        // Per word, independent.
        for w in 0..words {
            char1_grad_accum += self
                .encoder
                .backward_word(w, &self.d_ew_words[w][..char_out]);
        }

        let denom = words.max(1) as f32;
        self.last_high_grad_signal = high_grad_accum / denom;
        self.last_char1_grad_signal = char1_grad_accum / denom;
    }

    /// Average NLL per decoded word (chars + the trailing `[W]` summed, then
    /// averaged over words).
    fn word_loss(&self) -> f32 {
        let last = self.char2_model.layers.len() - 1;
        let mut total = 0.0;
        for &range in &self.dec_ranges {
            let mut word_nll = 0.0;
            for slot in range {
                let logits = self.char2_model.cache[slot][last].output();
                word_nll += nll_from_logits(logits, self.dec_targets[slot] as usize);
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
                let logits = self.char2_model.cache[slot][last].output();
                total += nll_from_logits(logits, self.dec_targets[slot] as usize);
                count += 1;
            }
        }
        total / count.max(1) as f32
    }

    /// Forward-only evaluation: runs `forward_over` (under the current
    /// `backbone_mode`) over every window and returns mean per-token decode
    /// cross-entropy in nats. No backward, no weight updates.
    pub fn eval_decode_loss<'a, I: Iterator<Item = WordBatch<'a>>>(
        &mut self,
        data: I,
    ) -> (f32, f32) {
        let mut c_total = 0.0;
        let mut w_total = 0.0;
        let mut count = 0;
        for batch in data {
            let WordBatch { tokens, words } = batch;
            self.reset();
            self.forward_over(tokens, &words);
            if self.word_segments.len() < 2 {
                continue;
            }
            let c_loss = self.decode_loss();
            let w_loss = self.word_loss();
            c_total += c_loss;
            w_total += w_loss;

            count += 1;
        }

        let c_loss = c_total / count.max(1) as f32;
        let w_loss = w_total / count.max(1) as f32;
        (c_loss, w_loss)
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
            let word_loss = self.word_loss();
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
        write_f32_slice(w, &vec![0.0; self.encoder.combine.weights.rows()])?;
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
        let _combine_biases: Box<[f32]> = read_f32_vec(r)?;
        let combine_norm_gamma: Box<[f32]> = read_f32_vec(r)?;
        let step = read_u64(r).unwrap_or(0) as usize;

        let (ci, co) = (combine_weights.rows(), combine_weights.cols());
        let combine = LinearNBLayer::from_loaded(ci, co, combine_weights);
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
