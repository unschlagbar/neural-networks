// Per-word encoder for the hierarchical model.
//
// Encodes ONE word into a fixed-size embedding `e_w` with a normal (forward-only)
// `Sequential` stack: the word's characters are fed in order, followed by one
// closing `[W]` end-of-word step, and `e_w` is the stack's output at that `[W]`
// step — the recurrent state has seen the whole word AND knows it is complete
// (no suffix like an "s" can still change it). The `[W]` is fed virtually (the
// token slices never contain it) and is gated by `ENC_W_EOS` in `config.rs`;
// disabled, the readout is at the last char (for checkpoints trained without
// it). State is reset per word, so words encode
// independently — and that independence is exploited: the words of a window are
// split across a replica pool (see `parallel.rs`) and encoded data-parallel,
// each worker writing into its own slice of the shared forward cache.
//
// The forward cache layout per word is one slot per token plus the `[W]` slot,
// addressed by a running cursor (`enc_ranges`).
//
// Pulled out of `hierarchical.rs` so the orchestration there stays readable; the
// encoder owns its own replica pool and gradient bookkeeping.

use std::range::Range;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    config::ENC_W_EOS,
    hierarchical::backward_through_layers,
    nn::embedding::EmbeddingLayer,
    nn_layer::{DynCache, NnLayer},
    parallel::{ReplicaPool, WorkerChunk, chunk_words},
    sequential::Sequential,
};

pub struct WordEncoder {
    /// The character stack (`c1 … cn [W]`, readout at `[W]`).
    pub chars: Sequential,
    /// Per-thread worker copies of `chars` for the parallel word phases.
    pool: ReplicaPool,

    /// Reusable one-hot input buffer (size `vocab`) — sampling path only.
    char_input: Box<[f32]>,
    /// Largest layer dimension in the stack — sizes each worker's BPTT scratch.
    max_layer_dim: usize,

    /// Per encode-word: cache slot range `[start, end)` (length `word_len + 1`,
    /// the closing `[W]` step included).
    enc_ranges: Vec<Range<usize>>,

    /// The `[W]` marker id, fed as every word's closing encoder step.
    w_token: u16,

    /// Encoder output dimension `H`.
    pub output_size: usize,
}

impl WordEncoder {
    pub fn new(chars: Sequential, vocab_size: usize, w_token: u16) -> Self {
        let char_out = chars.output_size;
        assert_eq!(
            chars.input_size, vocab_size,
            "encoder.input_size must equal vocab_size"
        );

        let max_layer_dim = chars.max_layer_dim().max(char_out);

        Self {
            chars,
            pool: ReplicaPool::new(),
            char_input: vec![0.0; vocab_size].into(),
            max_layer_dim,
            enc_ranges: Vec::new(),
            w_token,
            output_size: char_out,
        }
    }

    /// `slots` = max TOKEN span of a window (cache depth).
    pub fn make_cache(&mut self, slots: usize) {
        self.chars.make_cache(slots);
    }

    /// Reset recurrent state and the per-window ranges (call once per window).
    pub fn reset(&mut self) {
        for layer in &mut self.chars.layers {
            layer.reset_state();
        }
        self.enc_ranges.clear();
    }

    pub fn num_words(&self) -> usize {
        self.enc_ranges.len()
    }

    /// Clear BPTT gradient-carry state (degenerate windows).
    pub fn reset_bptt(&mut self) {
        for layer in &mut self.chars.layers {
            layer.reset_bptt_state();
        }
    }

    pub fn enc_range(&self, w: usize) -> Range<usize> {
        self.enc_ranges[w]
    }

    /// Word embedding `e_w` of the `w`-th encoded word (valid after `encode_words`).
    pub fn e_w(&self, w: usize) -> &[f32] {
        let readout = self.enc_ranges[w].end - 1;
        self.chars.cache[readout].last().unwrap().output()
    }

    /// The char embedding table (first layer). The decoder shares it (tied
    /// embedding): `Hierarchical` injects its rows as the decoder's char-step
    /// inputs and adds the decoder-side gradients back into it.
    pub fn char_embedding(&self) -> &EmbeddingLayer {
        self.chars.layers[0]
            .as_any()
            .downcast_ref::<EmbeddingLayer>()
            .expect("encoder stack must start with an EmbeddingLayer")
    }

    pub fn char_embedding_mut(&mut self) -> &mut EmbeddingLayer {
        self.chars.layers[0]
            .as_any_mut()
            .downcast_mut::<EmbeddingLayer>()
            .expect("encoder stack must start with an EmbeddingLayer")
    }

    /// Encode every word of the window into the shared cache, data-parallel
    /// across the replica pool (words are mutually independent — each resets
    /// its own state). The results are read via [`e_w`](Self::e_w).
    pub fn encode_words(&mut self, tokens: &[u16], words: &[Range<usize>]) {
        self.enc_ranges.clear();
        let mut cursor = 0;
        for word in words {
            // One slot per char plus the closing [W] step (if enabled).
            let end = cursor + (word.end - word.start) + ENC_W_EOS as usize;
            self.enc_ranges.push(Range { start: cursor, end });
            cursor = end;
        }
        if words.is_empty() {
            return;
        }

        self.pool.sync(&self.chars);
        let vocab = self.chars.input_size;
        let w_token = self.w_token;
        let enc_ranges = &self.enc_ranges;
        let chunks = chunk_words(
            &mut self.pool.replicas,
            enc_ranges,
            words.len(),
            &mut self.chars.cache,
        );

        chunks.into_par_iter().for_each(|chunk| {
            let WorkerChunk {
                replica,
                words: wr,
                cache,
                slot_base,
            } = chunk;
            let mut char_input = vec![0.0; vocab];
            for w in wr.into_iter() {
                // c1, …, cn, [W] ; readout at the [W] step (or the last char
                // when ENC_W_EOS is off).
                for layer in &mut replica.layers {
                    layer.reset_state();
                }
                let base = enc_ranges[w].start - slot_base;
                let toks = words[w]
                    .into_iter()
                    .map(|u| tokens[u])
                    .chain(ENC_W_EOS.then_some(w_token));
                for (k, tok) in toks.enumerate() {
                    let tok = tok as usize;
                    char_input[tok] = 1.0;
                    Sequential::forward_step(&mut replica.layers, &mut cache[base + k], &char_input);
                    char_input[tok] = 0.0;
                }
            }
        });
    }

    /// Backprop every word of the window, data-parallel like `encode_words`:
    /// takes the per-word gradients w.r.t. `e_w`, seeds each at its readout
    /// step, runs BPTT down the stack, and reduces the per-replica weight
    /// gradients into `chars`. Returns the summed mean-abs seed gradient (a
    /// diagnostic signal).
    pub fn backward_words(&mut self, d_ew_words: &[Box<[f32]>]) -> f32 {
        let n = self.enc_ranges.len();
        if n == 0 {
            return 0.0;
        }

        let char_out = self.output_size;
        let max_dim = self.max_layer_dim;
        let enc_ranges = &self.enc_ranges;
        let chunks = chunk_words(&mut self.pool.replicas, enc_ranges, n, &mut self.chars.cache);

        let signal: f32 = chunks
            .into_par_iter()
            .map(|chunk| {
                let WorkerChunk {
                    replica,
                    words: wr,
                    cache,
                    slot_base,
                } = chunk;
                let mut delta_buf = vec![0.0; max_dim];
                let mut sig = 0.0;
                for w in wr.into_iter() {
                    let d_ew = &d_ew_words[w][..char_out];
                    sig += d_ew.iter().map(|x| x.abs()).sum::<f32>() / char_out as f32;

                    // The readout (and thus the seed gradient) is at the [W]
                    // step, enc_end-1.
                    let enc_range = enc_ranges[w];
                    for slot in enc_range.into_iter().rev() {
                        if slot == enc_range.end - 1 {
                            delta_buf[..char_out].copy_from_slice(d_ew);
                        } else {
                            delta_buf[..char_out].fill(0.0);
                        }
                        backward_through_layers(
                            &mut replica.layers,
                            &mut cache[slot - slot_base],
                            &mut delta_buf,
                            char_out,
                        );
                    }
                    for layer in &mut replica.layers {
                        layer.accumulate_init_grad();
                        layer.reset_bptt_state();
                    }
                }
                sig
            })
            .sum();

        self.pool.reduce_into(&mut self.chars);
        signal
    }

    pub fn sgd_step(&mut self, lr: f32, weight_decay: f32) {
        self.chars.sgd_step(lr, weight_decay);
        self.pool.mark_dirty();
    }

    /// Sampling-time encode of a single completed word (uses `forward_sample` and
    /// the single-word cache slots `0..=n`). Returns `e_w`.
    pub fn encode_word_sample(&mut self, word: &[u16]) -> &[f32] {
        let n = word.len();
        debug_assert!(n > 0, "cannot encode an empty word");

        // c1, …, cn, [W] — readout at the [W] step (slot n), or at the last
        // char (slot n-1) when ENC_W_EOS is off.
        for layer in &mut self.chars.layers {
            layer.reset_state();
        }
        let toks = word
            .iter()
            .copied()
            .chain(ENC_W_EOS.then_some(self.w_token));
        for (k, tok) in toks.enumerate() {
            self.char_input[tok as usize] = 1.0;
            forward_sample_step(
                &mut self.chars.layers,
                &mut self.chars.cache[k],
                &self.char_input,
            );
            self.char_input[tok as usize] = 0.0;
        }

        self.chars.cache[n - 1 + ENC_W_EOS as usize]
            .last()
            .unwrap()
            .output()
    }

    /// True once `make_cache` has allocated the forward cache (sampling guard).
    pub fn cache_ready(&self) -> bool {
        !self.chars.cache.is_empty()
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
