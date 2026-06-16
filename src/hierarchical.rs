use std::{
    fs::File,
    io::{self, Cursor, Read, Write},
    range::Range,
    rc::Rc,
    time::Instant,
};

use crate::{
    batches::WordBatch,
    config::{INJECT_C, INJECT_H, MODEL_LOC, STOP_WORD_DIRECT_FEED},
    loading::{read_f32_vec, read_matrix, read_u16, read_u32, read_u64},
    nn::{
        linear::{LinearCache, LinearLayer},
        softmax::{softmax, softmax_inplace},
    },
    nn_layer::{DynCache, NnLayer},
    saving::{HIER_MAGIC, write_f32_slice, write_matrix, write_u16, write_u32, write_u64},
    sequential::Sequential,
    tokenizer::Tokenizer,
    training::TrainingState,
};

pub struct Hierarchical {
    pub char_model: Sequential,
    pub char2_model: Sequential,
    pub word_model: Sequential,

    /// Projects the word context `o_w` → decoder sLSTM h+c initial states, used
    /// when `INJECT_H`/`INJECT_C` are on. One cache per word.
    state_head: LinearLayer,
    state_head_caches: Vec<LinearCache>,
    char2_state_size: usize,
    /// Scratch holding predicted / collected h+c states for `state_head`.
    state_grad_buf: Box<[f32]>,

    pub vocab_size: usize,
    pub context_size: usize,

    pub tokenizer: Rc<Tokenizer>,

    pub boundary_token_ids: Vec<u16>,
    /// `[start, end)` (half-open) ranges over target positions, one per word;
    /// the boundary token is the last element, at index `end - 1`.
    /// Filled by `forward_over`, reused by `backwards_sequence` / `compute_word_loss`.
    word_segments: Vec<Range<usize>>,

    /// Reusable one-hot input buffer for the encoder (size `vocab`).
    char_input: Box<[f32]>,
    /// Decoder input: `[char one-hot ‖ word context]` (size `vocab + context`).
    char2_input: Box<[f32]>,

    /// Backward scratch: top-level delta as it flows down a layer stack.
    delta_buf: Box<[f32]>,
    /// Accumulated gradient w.r.t. the decoder's word context, per word.
    d_o_buf: Box<[f32]>,
    /// Gradient w.r.t. an encoder embedding `e_w`.
    d_e_buf: Box<[f32]>,

    pub last_high_grad_signal: f32,
    pub last_char1_grad_signal: f32,
    pub last_word_loss: f32,
    pub step: usize,

    /// Debug-only: when set, the word context `o` is zeroed before it reaches the
    /// decoder, so the decoder must predict from the teacher-forced characters
    /// alone. Used to measure how much the decoder actually relies on the backbone.
    pub ablate_context: bool,

    /// Debug-only: when set, the backbone (word_model) is reset before every word,
    /// so its output `o` depends only on the immediately-preceding word (Markov-1).
    /// Comparing loss against the unreset backbone shows whether long-range history
    /// actually improves prediction.
    pub markov_backbone: bool,
}

impl Hierarchical {
    /// `boundary_token_ids` — from `tokenizer.boundary_tokens()`.
    pub fn new(
        char_model: Sequential,
        char2_model: Sequential,
        word_model: Sequential,
        vocab_size: usize,
        boundary_token_ids: Vec<u16>,
        tokenizer: Rc<Tokenizer>,
    ) -> Self {
        let context_size = word_model.output_size;

        assert_eq!(
            char_model.input_size, vocab_size,
            "encoder.input_size must equal vocab_size"
        );
        assert_eq!(
            char2_model.output_size, vocab_size,
            "decoder.output_size must equal vocab_size"
        );
        assert_eq!(
            char_model.output_size, word_model.input_size,
            "backbone.input_size must equal encoder.output_size"
        );
        assert_eq!(
            char2_model.input_size,
            context_size + vocab_size,
            "decoder.input_size must equal backbone.output_size + vocab_size"
        );

        let char_input = vec![0.0; vocab_size].into();
        let char2_input = vec![0.0; vocab_size + context_size].into();

        let buf_size = vocab_size
            .max(char2_model.input_size)
            .max(char_model.output_size)
            .max(context_size);
        let delta_buf = vec![0.0; buf_size].into();
        let d_e_buf = vec![0.0; char_model.output_size].into();

        let char2_state_size: usize = char2_model.layers.iter().map(|l| l.state_size()).sum();
        let state_head = LinearLayer::zeroed(context_size, char2_state_size);
        let state_grad_buf = vec![0.0; char2_state_size].into();

        Self {
            char_model,
            char2_model,
            word_model,
            state_head,
            state_head_caches: Vec::new(),
            char2_state_size,
            state_grad_buf,
            vocab_size,
            context_size,
            tokenizer,
            boundary_token_ids,
            word_segments: Vec::new(),
            char_input,
            char2_input,
            delta_buf,
            d_o_buf: vec![0.0; context_size].into(),
            d_e_buf,
            last_high_grad_signal: 0.0,
            last_char1_grad_signal: 0.0,
            last_word_loss: 0.0,
            step: 0,
            ablate_context: false,
            markov_backbone: false,
        }
    }

    pub fn make_cache(&mut self, seq_len: usize) {
        self.char_model.make_cache(seq_len * 6);
        self.char2_model.make_cache(seq_len * 6);
        self.word_model.make_cache(seq_len);
        self.state_head_caches = (0..seq_len.max(1))
            .map(|_| LinearCache::new(self.context_size, self.char2_state_size))
            .collect();
    }

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
        self.word_segments.clear();
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

    fn condition_decoder(&mut self, cache_idx: usize) {
        let vocab = self.vocab_size;
        if INJECT_C || INJECT_H {
            {
                let c = &mut self.state_head_caches[cache_idx];
                LinearLayer::forward(&self.state_head, &self.char2_input[vocab..], c);
            }
            if STOP_WORD_DIRECT_FEED {
                self.char2_input[vocab..].fill(0.0);
            }
            self.state_grad_buf
                .copy_from_slice(self.state_head_caches[cache_idx].output());
            let mut offset = 0;
            for layer in &mut self.char2_model.layers {
                offset = layer.inject_state(&self.state_grad_buf, offset);
            }
        } else {
            if STOP_WORD_DIRECT_FEED {
                self.char2_input[vocab..].fill(0.0);
            }
            for layer in &mut self.char2_model.layers {
                layer.reset_state();
            }
        }
    }

    pub fn forward_over(&mut self, tokens: &[u16], words: &[Range<usize>]) {
        self.word_segments.clear();
        self.word_segments.extend_from_slice(words);
        let n = self.word_segments.len();
        let vocab = self.vocab_size;

        for w in 0..n.saturating_sub(1) {
            let word = self.word_segments[w]; // encoder word
            let next_word = self.word_segments[w + 1]; // decoder word

            // --- ENCODER: read word w's chars → embedding e_w at its last char ---
            for layer in &mut self.char_model.layers {
                layer.reset_state();
            }

            //print!("word: \"");

            for u in word {
                let tok = tokens[u] as usize;
                //print!("{}", self.tokenizer.display(tok as u16));
                self.char_input[tok] = 1.0;
                Self::forward_step(
                    &mut self.char_model.layers,
                    &mut self.char_model.cache[u],
                    &self.char_input,
                );
                self.char_input[tok] = 0.0;
            }

            //println!("\"");

            // --- BACKBONE: consume e_w → o_{w+1} (context for the next word) ---
            if self.markov_backbone {
                for layer in &mut self.word_model.layers {
                    layer.reset_state();
                }
            }
            let e_len = self.char_model.output_size;
            self.delta_buf[..e_len]
                .copy_from_slice(self.char_model.cache[word.end - 1].last().unwrap().output());
            Self::forward_step(
                &mut self.word_model.layers,
                &mut self.word_model.cache[w],
                &self.delta_buf[..e_len],
            );

            // --- DECODER: generate word w+1's chars, conditioned on o_{w+1} ---
            {
                let o = self.word_model.cache[w].last().unwrap().output();
                self.char2_input[vocab..].copy_from_slice(o);
            }
            if self.ablate_context {
                self.char2_input[vocab..].fill(0.0);
            }
            self.condition_decoder(w + 1);

            for t in next_word {
                let tok = tokens[t - 1] as usize;
                self.char2_input[tok] = 1.0;
                Self::forward_step(
                    &mut self.char2_model.layers,
                    &mut self.char2_model.cache[t],
                    &self.char2_input,
                );
                self.char2_input[tok] = 0.0;
            }
        }
    }

    pub fn backwards_sequence(&mut self, targets: &[u16]) {
        let n = self.word_segments.len();

        let vocab = self.vocab_size;
        let context = self.context_size;
        let char_out = self.char_model.output_size;

        let mut high_grad_accum = 0.0;
        let mut char1_grad_accum = 0.0;
        let mut encoded_words = 0;

        //println!("start\n");
        for w in (1..n).rev() {
            let word = self.word_segments[w];

            // --- DECODER backward; accumulate grad w.r.t. the word context o_w ---
            self.d_o_buf.fill(0.0);

            // print!("word: \"");
            //word.iter()
            //    .for_each(|c| print!("{}", self.tokenizer.display(targets[c])));
            //println!("\"");

            for t in word.iter().rev() {
                let out_len = {
                    let out = self.char2_model.cache[t].last().unwrap().output();
                    let len = out.len();
                    self.delta_buf[..len].copy_from_slice(out);
                    len
                };
                softmax_inplace(&mut self.delta_buf[..out_len]);
                self.delta_buf[targets[t] as usize] -= 1.0;

                backward_through_layers(
                    &mut self.char2_model.layers,
                    &mut self.char2_model.cache[t],
                    &mut self.delta_buf,
                    out_len,
                );

                // Gradient via the concatenated context (only when it was fed).
                if !STOP_WORD_DIRECT_FEED {
                    let dx = self.char2_model.cache[t][0].input_grad();
                    for i in 0..context {
                        self.d_o_buf[i] += dx[vocab + i];
                    }
                }
            }

            // Gradient via the injected initial state (if injection was used).
            if INJECT_C || INJECT_H {
                self.state_grad_buf.fill(0.0);
                let mut offset = 0;
                for layer in &mut self.char2_model.layers {
                    offset = layer.collect_bptt_grad(&mut self.state_grad_buf, offset);
                }
                {
                    let size = self.char2_state_size;
                    let c = &mut self.state_head_caches[w];
                    LinearLayer::backward(
                        &mut self.state_head,
                        &mut self.state_grad_buf[..size],
                        c,
                    );
                }
                let dx = self.state_head_caches[w].input_grad();
                for i in 0..context {
                    self.d_o_buf[i] += dx[i];
                }
            } else {
                for layer in &mut self.char2_model.layers {
                    layer.accumulate_init_grad();
                }
            }
            for layer in &mut self.char2_model.layers {
                layer.reset_bptt_state();
            }
            high_grad_accum += self.d_o_buf.iter().map(|x| x.abs()).sum::<f32>() / context as f32;

            self.delta_buf[..context].copy_from_slice(&self.d_o_buf);
            backward_through_layers(
                &mut self.word_model.layers,
                &mut self.word_model.cache[w - 1],
                &mut self.delta_buf,
                context,
            );
            {
                let de = self.word_model.cache[w - 1][0].input_grad();
                self.d_e_buf[..char_out].copy_from_slice(&de[..char_out]);
            }
            char1_grad_accum += self.d_e_buf[..char_out]
                .iter()
                .map(|x| x.abs())
                .sum::<f32>()
                / char_out as f32;
            encoded_words += 1;

            let word = self.word_segments[w - 1];
            for u in word.iter().rev() {
                if u == word.end - 1 {
                    self.delta_buf[..char_out].copy_from_slice(&self.d_e_buf[..char_out]);
                } else {
                    self.delta_buf[..char_out].fill(0.0);
                }
                backward_through_layers(
                    &mut self.char_model.layers,
                    &mut self.char_model.cache[u],
                    &mut self.delta_buf,
                    char_out,
                );
            }
            for layer in &mut self.char_model.layers {
                layer.accumulate_init_grad();
                layer.reset_bptt_state();
            }
        }

        for layer in &mut self.word_model.layers {
            layer.accumulate_init_grad();
            layer.reset_bptt_state();
        }

        self.last_high_grad_signal = high_grad_accum / (n - 1) as f32;
        self.last_char1_grad_signal = if encoded_words > 0 {
            char1_grad_accum / encoded_words as f32
        } else {
            0.0
        };
    }

    /// Average NLL per word (chars of each word summed, then averaged over words).
    fn compute_word_loss(&self, targets: &[u16]) -> f32 {
        let last = self.char2_model.layers.len() - 1;
        let mut total = 0.0;
        // Word 0 is the encode-only prefix; only decoded words (1..) have logits.
        for &word in self.word_segments.iter().skip(1) {
            let mut word_nll = 0.0;
            for t in word {
                let probs = softmax(self.char2_model.cache[t][last].output());
                word_nll -= (probs[targets[t] as usize] + 1e-12).ln();
            }
            total += word_nll;
        }
        let word_count = self.word_segments.len().saturating_sub(1).max(1);
        total / word_count as f32
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
                for layer in &mut self.char_model.layers {
                    layer.reset_bptt_state();
                }
                for layer in &mut self.char2_model.layers {
                    layer.reset_bptt_state();
                }
                for layer in &mut self.word_model.layers {
                    layer.reset_bptt_state();
                }
                continue;
            }

            let decode_start = self.word_segments[1].start;
            let loss = self.char2_model.seq_loss_from(window, decode_start);
            let word_loss = self.compute_word_loss(window);
            self.last_word_loss = word_loss;
            tokens += window.len();

            self.backwards_sequence(window);
            state.log_tokens(window.len());
            state.log_metric("delta_word", self.last_high_grad_signal);
            state.log_metric("delta_char1", self.last_char1_grad_signal);
            state.log_metric("word_loss", word_loss);

            if let Some(lr) = state.step(loss) {
                self.char_model.sgd_step(lr);
                self.char2_model.sgd_step(lr);
                self.word_model.sgd_step(lr);
                if INJECT_C || INJECT_H {
                    self.state_head.apply_grads(lr);
                    self.state_head.clear_grads();
                }
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
                match self.save(MODEL_LOC) {
                    Ok(()) => println!("saved"),
                    Err(e) => eprintln!("save failed: {e}"),
                }
            }
        }
    }

    fn encode_word_advance(&mut self, word: &[u16]) {
        let vocab = self.vocab_size;

        for layer in &mut self.char_model.layers {
            layer.reset_state();
        }
        for &tok in word {
            self.char_input[tok as usize] = 1.0;
            Self::forward_sample_step(
                &mut self.char_model.layers,
                &mut self.char_model.cache[0],
                &self.char_input,
            );
            self.char_input[tok as usize] = 0.0;
        }

        let e_len = self.char_model.output_size;
        self.delta_buf[..e_len].copy_from_slice(self.char_model.cache[0].last().unwrap().output());
        Self::forward_sample_step(
            &mut self.word_model.layers,
            &mut self.word_model.cache[0],
            &self.delta_buf[..e_len],
        );

        {
            let o = self.word_model.cache[0].last().unwrap().output();
            self.char2_input[vocab..].copy_from_slice(o);
        }
        self.condition_decoder(0);
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
        let vocab = self.vocab_size;

        // Start with the learnable initial context.
        self.char2_input.fill(0.0);
        self.condition_decoder(0);

        // `last_token` is the decoder input (the previously produced char). The
        // first one is the prefix's head, or a random seed.
        let mut last_token = if prefix.is_empty() {
            rand::random_range(0..vocab) as u16
        } else {
            prefix[0]
        };
        let mut prefix_pos = 1;

        let mut word_chars: Vec<u16> = Vec::new();
        let mut out = Vec::with_capacity(max_len);

        loop {
            // The previous char closes the current word when it is a boundary:
            // encode that word and advance the backbone before decoding the next.
            word_chars.push(last_token);
            if self.boundary_token_ids.contains(&last_token) {
                self.encode_word_advance(&word_chars);
                word_chars.clear();
            }

            // Decode one step conditioned on the current context.
            self.char2_input[last_token as usize] = 1.0;
            Self::forward_sample_step(
                &mut self.char2_model.layers,
                &mut self.char2_model.cache[0],
                &self.char2_input,
            );
            self.char2_input[last_token as usize] = 0.0;

            let forced = prefix_pos < prefix.len();
            let next = if forced {
                let t = prefix[prefix_pos];
                prefix_pos += 1;
                t
            } else {
                let logits = self.char2_model.cache[0].last().unwrap().output();
                let scaled: Vec<f32> = logits.iter().map(|&v| v / temperature.max(1e-8)).collect();
                let probs = softmax(&scaled);
                self.sample_from_probs(&probs)
            };

            if !forced {
                out.push(next);
                if out.len() >= max_len || !callback(next) {
                    break;
                }
            }
            last_token = next;
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

        self.char_model.write_to(w)?;
        self.char2_model.write_to(w)?;
        self.word_model.write_to(w)?;
        write_matrix(w, &self.state_head.weights)?;
        write_f32_slice(w, &self.state_head.biases)?;
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

        let char_model = Sequential::load_from(r)?;
        let char2_model = Sequential::load_from(r)?;
        let word_model = Sequential::load_from(r)?;
        let state_head_weights = read_matrix(r)?;
        let state_head_biases: Box<[f32]> = read_f32_vec(r)?;
        let step = read_u64(r).unwrap_or(0) as usize;

        let mut model = Self::new(
            char_model,
            char2_model,
            word_model,
            vocab_size,
            boundary_token_ids,
            tokenizer,
        );
        debug_assert_eq!(model.context_size, context_size);
        model.state_head = LinearLayer::from_loaded(
            model.context_size,
            model.char2_state_size,
            state_head_weights,
            state_head_biases,
        );
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
        model.make_cache(64);

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

    /// Probe a TRAINED model: how strongly does the context for the last word
    /// depend on each earlier word? Perturb one word at a time and measure the
    /// change in the backbone's last-step output. If the effect collapses to ~0
    /// for words further back than the immediately-preceding one, the trained
    /// backbone has become Markov-1 (ignores history) — exactly the user's claim.
    #[test]
    fn trained_word_context_memory_profile() {
        let tokenizer = Rc::new(Tokenizer::new(crate::config::CHARSET, false));
        let path = std::env::var("MODELP").unwrap_or_else(|_| "models/garbage".into());
        let mut model = match Hierarchical::load(&path, tokenizer.clone()) {
            Ok(m) => m,
            Err(e) => {
                println!("skip trained probe — could not load {path}: {e}");
                return;
            }
        };
        model.make_cache(256);

        let vocab = tokenizer.vocab_size();
        let boundaries = tokenizer.boundary_tokens();
        let text = "the king gave the order to his loyal army today and";
        let base: Vec<u16> = tokenizer.to_tokens(text);

        // Segment into [start,end) words, each closed by a boundary token.
        let mut words: Vec<Range<usize>> = Vec::new();
        let mut start = 0;
        for t in 0..base.len() {
            if boundaries.contains(&base[t]) {
                words.push(Range { start, end: t + 1 });
                start = t + 1;
            }
        }
        let n = words.len();
        assert!(n >= 4, "need several words; got {n}");

        let last_ctx = |model: &mut Hierarchical, toks: &[u16]| -> Vec<f32> {
            model.reset();
            model.forward_over(toks, &words);
            model.word_model.cache[n - 2]
                .last()
                .unwrap()
                .output()
                .to_vec()
        };

        let o0 = last_ctx(&mut model, &base);
        println!("\nmemory profile of last-word context (model {path}, {n} words):");
        println!("  dist = how many words back from the last encoded word was changed");
        for k in 0..n - 1 {
            let mut toks = base.clone();
            let pos = words[k].start;
            let alt = (0..vocab as u16)
                .find(|t| !boundaries.contains(t) && *t != toks[pos])
                .unwrap();
            toks[pos] = alt;
            let ok = last_ctx(&mut model, &toks);
            let d: f32 =
                o0.iter().zip(&ok).map(|(a, c)| (a - c).abs()).sum::<f32>() / o0.len() as f32;
            let dist = (n - 2) - k;
            println!("  word {k:2} (dist {dist:2}): mean|Δctx| = {d:.6e}");
        }
    }

    /// Probe a TRAINED model: does the DECODER actually use the word context?
    /// Compute the real decode loss, then zero the context (`ablate_context`) and
    /// recompute. If the loss barely rises, the decoder predicts from teacher-forced
    /// characters alone and the backbone is dead weight — which would explain why a
    /// bigger word model does not help. Runs on a real chunk of the training file.
    #[test]
    fn trained_decoder_uses_context() {
        let tokenizer = Rc::new(Tokenizer::new(crate::config::CHARSET, false));
        let path = std::env::var("MODELP").unwrap_or_else(|_| "models/opus_fix".into());
        let mut model = match Hierarchical::load(&path, tokenizer.clone()) {
            Ok(m) => m,
            Err(e) => {
                println!("skip — could not load {path}: {e}");
                return;
            }
        };

        // Read a chunk of real training text.
        let raw = std::fs::read_to_string(crate::config::DATA_FILE).unwrap_or_else(|_| {
            "the king gave the order to his loyal army today and the people \
                 cheered loudly while the sun set behind the distant hills."
                .into()
        });
        let chunk: String = raw.chars().filter(|c| *c != '\u{0}').take(1200).collect();
        let base: Vec<u16> = tokenizer.to_tokens(&chunk);

        let boundaries = tokenizer.boundary_tokens();
        let mut words: Vec<Range<usize>> = Vec::new();
        let mut start = 0;
        for t in 0..base.len() {
            if boundaries.contains(&base[t]) {
                words.push(Range { start, end: t + 1 });
                start = t + 1;
            }
        }
        let n = words.len();
        assert!(n >= 8, "need several words; got {n}");
        model.make_cache(base.len());

        model.ablate_context = false;
        model.reset();
        model.forward_over(&base, &words);
        let loss_real = model.compute_word_loss(&base);

        model.ablate_context = true;
        model.reset();
        model.forward_over(&base, &words);
        let loss_ablated = model.compute_word_loss(&base);
        model.ablate_context = false;

        // Backbone reset every word → o depends only on the last word (Markov-1).
        model.markov_backbone = true;
        model.reset();
        model.forward_over(&base, &words);
        let loss_markov = model.compute_word_loss(&base);
        model.markov_backbone = false;

        let rel = (loss_ablated - loss_real) / loss_real.max(1e-6);
        let rel_m = (loss_markov - loss_real) / loss_real.max(1e-6);
        println!("\ndecoder context reliance (model {path}, {n} words):");
        println!("  per-word NLL  real            = {loss_real:.4}");
        println!(
            "  per-word NLL  context zeroed   = {loss_ablated:.4}  ({:+.1}%)",
            rel * 100.0
        );
        println!(
            "  per-word NLL  backbone Markov-1 = {loss_markov:.4}  ({:+.1}%)",
            rel_m * 100.0
        );
        println!("  → if Markov-1 ≈ real, long-range history does not help prediction.");
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
    /// Perturbs one weight of the backbone's input projection (shared across every
    /// word step, so its gradient sums the direct + recurrent BPTT paths) and
    /// compares finite differences of the summed decode loss against the analytic
    /// gradient produced by `backwards_sequence`. A match proves forward and
    /// backward are consistent, including the cross-word recurrence. Env knobs for
    /// manual probing: `WMODEL=word|char2`, `NWORDS=n`, `LIDX=layer`.
    #[test]
    fn backward_matches_numeric_grad_across_words() {
        use crate::nn::softmax::softmax;
        use crate::optimizers::GradMatrixOps;

        let tokenizer = Rc::new(Tokenizer::new(crate::config::CHARSET, false));
        let vocab = tokenizer.vocab_size();
        let boundaries = tokenizer.boundary_tokens();
        let mut model = build_hierarchical_model(vocab, boundaries.clone(), tokenizer.clone());
        model.make_cache(64);

        let b = boundaries[0];
        let content: Vec<u16> = (0..vocab as u16)
            .filter(|t| !boundaries.contains(t))
            .take(8)
            .collect();
        // Toggle here to localize: 2 words = single backbone step (no recurrence),
        // 4 words = three backbone steps (exercises cross-word BPTT).
        let n_words = std::env::var("NWORDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4usize);
        let mut tokens = Vec::new();
        let mut words: Vec<Range<usize>> = Vec::new();
        for k in 0..n_words {
            let start = tokens.len();
            tokens.push(content[k % content.len()]);
            tokens.push(b);
            words.push(Range {
                start,
                end: start + 2,
            });
        }

        // Summed cross-entropy over decoded positions (words 1..n) — exactly the
        // scalar whose gradient `backwards_sequence` accumulates (delta = p - onehot,
        // unscaled).
        let summed_loss = |model: &mut Hierarchical| -> f32 {
            model.reset();
            model.forward_over(&tokens, &words);
            let last = model.char2_model.layers.len() - 1;
            let mut loss = 0.0;
            for w in 1..words.len() {
                let word = words[w];
                for t in word {
                    let probs = softmax(model.char2_model.cache[t][last].output());
                    loss -= (probs[tokens[t] as usize] + 1e-12).ln();
                }
            }
            loss
        };

        // Analytic gradient via the real backward.
        model.reset();
        model.forward_over(&tokens, &words);
        model.backwards_sequence(&tokens);

        let which = std::env::var("WMODEL").unwrap_or_else(|_| "word".into());
        let lidx: usize = std::env::var("LIDX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        // For char2 layer 0, rows 0..vocab are sparse one-hot token features; pick a
        // row in the always-active context region instead. For other layers row 3 is fine.
        let i = if which == "char2" && lidx == 0 {
            vocab + 3
        } else {
            3usize
        };
        let j = 7usize;
        fn pick<'a>(model: &'a mut Hierarchical, which: &str, lidx: usize) -> &'a mut LinearLayer {
            let m = match which {
                "char2" => &mut model.char2_model,
                _ => &mut model.word_model,
            };
            m.layers[lidx]
                .as_any_mut()
                .downcast_mut::<LinearLayer>()
                .expect("selected layer must be a LinearLayer")
        }
        let g_analytic = pick(&mut model, &which, lidx).grads.weights.matrix()[i][j];

        // Central finite difference at the same weight, swept over eps.
        let perturb = |model: &mut Hierarchical, d: f32| {
            pick(model, &which, lidx).weights[i][j] += d;
        };

        let eps = 5e-3;
        perturb(&mut model, eps);
        let l_plus = summed_loss(&mut model);
        perturb(&mut model, -2.0 * eps);
        let l_minus = summed_loss(&mut model);
        perturb(&mut model, eps); // restore
        let g_numeric = (l_plus - l_minus) / (2.0 * eps);
        let rel = (g_analytic - g_numeric).abs() / g_analytic.abs().max(g_numeric.abs()).max(1e-6);
        println!(
            "checking {which} model, n_words = {n_words}: analytic {g_analytic:.6e}  numeric {g_numeric:.6e}  rel {rel:.3e}"
        );
        assert!(
            rel < 3e-2,
            "analytic and numeric gradient disagree (rel {rel:.3e}) → forward/backward inconsistent"
        );
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
