// Per-word encoder for the hierarchical model — bidirectional.
//
// Encodes ONE word into a fixed-size embedding `e_w` with TWO forward-only
// `Sequential` stacks (a BiLSTM):
//
//   - `fwd` reads the word's characters in order `c1 … cn` plus a closing `[W]`
//     end-of-word step, and reads out at that `[W]` step — the state has seen
//     the whole word AND knows it is complete.
//   - `bwd` reads the characters back-to-front, `cn … c1`, and then the SAME
//     closing `[W]`, reading out there too.
//
// Both directions therefore end on `[W]` (`hallo` → fwd `h a l l o [W]`, bwd
// `o l l a h [W]`), so each readout is taken at a step carrying the
// end-of-word signal. Reversing the whole sequence instead (`[W] o l l a h`)
// would read the backward direction out at an ordinary char step, an asymmetry
// the two directions do not recover from.
//
// The two readouts are concatenated `[fwd ; bwd]` (width `2·char_out`) and a
// `combine` Linear projects them back to `char_out`, which is `e_w`. Keeping
// `e_w` at `char_out` leaves every downstream contract unchanged: the backbone
// input width and the tied char-embedding width (CHAR_HIDDEN == OUT_HIDDEN).
//
// The `[W]` is fed virtually to each stack (the token slices never contain it)
// and is gated by `ENC_W_EOS` in `config.rs`; disabled, each direction reads out
// at its own last character (`cn` for `fwd`, `c1` for `bwd`). State is reset per word, so words
// encode independently — and that independence is exploited: the words of a
// window are split across replica pools (see `parallel.rs`) and encoded
// data-parallel, each worker writing into its own slice of the shared caches.
//
// The forward-cache layout per word is one slot per token plus the `[W]` slot,
// addressed by a running cursor (`enc_ranges`); `fwd` and `bwd` each own a
// cache with that same slot layout.
//
// Pulled out of `hierarchical.rs` so the orchestration there stays readable; the
// encoder owns its own replica pools and gradient bookkeeping.

use std::range::Range;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    config::ENC_W_EOS,
    hierarchical::backward_through_layers,
    nn::{embedding::EmbeddingLayer, linear::LinearLayer},
    nn_layer::{DynCache, NnLayer},
    parallel::{ReplicaPool, WorkerChunk, chunk_words},
    sequential::Sequential,
};

pub struct WordEncoder {
    /// Forward character stack (`c1 … cn [W]`, readout at `[W]`).
    pub fwd: Sequential,
    /// Backward character stack (`cn … c1 [W]`, readout at `[W]`).
    pub bwd: Sequential,
    /// Projection of the concatenated readouts `[fwd ; bwd]` → `e_w`.
    pub combine: LinearLayer,

    /// Per-thread worker copies of `fwd` / `bwd` for the parallel word phases.
    fwd_pool: ReplicaPool,
    bwd_pool: ReplicaPool,

    /// Reusable one-hot input buffer (size `vocab`) — sampling path only.
    char_input: Box<[f32]>,
    /// Largest layer dimension across both stacks — sizes each worker's BPTT scratch.
    max_layer_dim: usize,

    /// Per encode-word: cache slot range `[start, end)` (length `word_len + 1`,
    /// the closing `[W]` step included). Shared by both stacks.
    enc_ranges: Vec<Range<usize>>,

    /// The combined word embedding `e_w` per word (owned; sized `char_out`).
    e_w: Vec<Box<[f32]>>,
    /// The `combine` forward cache per word (holds the concatenated input and
    /// the readout output, so the parallel backward can run `combine` per word).
    combine_cache: Vec<Box<dyn DynCache>>,

    /// The `[W]` marker id, fed as every word's `[W]` encoder step.
    w_token: u16,

    /// Encoder output dimension `H` (= `char_out` = combine output).
    pub output_size: usize,
}

impl WordEncoder {
    pub fn new(fwd: Sequential, bwd: Sequential, vocab_size: usize, w_token: u16) -> Self {
        let char_out = fwd.output_size;
        assert_eq!(
            fwd.input_size, vocab_size,
            "encoder.fwd.input_size must equal vocab_size"
        );
        assert_eq!(
            bwd.input_size, vocab_size,
            "encoder.bwd.input_size must equal vocab_size"
        );
        assert_eq!(
            bwd.output_size, char_out,
            "encoder.bwd.output_size must equal encoder.fwd.output_size"
        );

        // combine: [fwd ; bwd] (2·char_out) → e_w (char_out).
        let combine = LinearLayer::new(2 * char_out, char_out);

        let max_layer_dim = fwd
            .max_layer_dim()
            .max(bwd.max_layer_dim())
            .max(2 * char_out);

        Self {
            fwd,
            bwd,
            combine,
            fwd_pool: ReplicaPool::new(),
            bwd_pool: ReplicaPool::new(),
            char_input: vec![0.0; vocab_size].into(),
            max_layer_dim,
            enc_ranges: Vec::new(),
            e_w: Vec::new(),
            combine_cache: Vec::new(),
            w_token,
            output_size: char_out,
        }
    }

    /// `slots` = max TOKEN span of a window (cache depth).
    pub fn make_cache(&mut self, slots: usize) {
        self.fwd.make_cache(slots);
        self.bwd.make_cache(slots);
    }

    /// Reset recurrent state and the per-window ranges (call once per window).
    pub fn reset(&mut self) {
        for layer in &mut self.fwd.layers {
            layer.reset_state();
        }
        for layer in &mut self.bwd.layers {
            layer.reset_state();
        }
        self.enc_ranges.clear();
    }

    pub fn num_words(&self) -> usize {
        self.enc_ranges.len()
    }

    /// Clear BPTT gradient-carry state (degenerate windows).
    pub fn reset_bptt(&mut self) {
        for layer in &mut self.fwd.layers {
            layer.reset_bptt_state();
        }
        for layer in &mut self.bwd.layers {
            layer.reset_bptt_state();
        }
    }

    pub fn enc_range(&self, w: usize) -> Range<usize> {
        self.enc_ranges[w]
    }

    /// Word embedding `e_w` of the `w`-th encoded word (valid after `encode_words`).
    pub fn e_w(&self, w: usize) -> &[f32] {
        &self.e_w[w]
    }

    /// The char embedding table (first layer of `fwd`). The decoder shares it
    /// (tied embedding): `Hierarchical` injects its rows as the decoder's
    /// char-step inputs and adds the decoder-side gradients back into it. Only
    /// `fwd`'s table is tied; `bwd` keeps its own.
    pub fn char_embedding(&self) -> &EmbeddingLayer {
        self.fwd.layers[0]
            .as_any()
            .downcast_ref::<EmbeddingLayer>()
            .expect("encoder fwd stack must start with an EmbeddingLayer")
    }

    pub fn char_embedding_mut(&mut self) -> &mut EmbeddingLayer {
        self.fwd.layers[0]
            .as_any_mut()
            .downcast_mut::<EmbeddingLayer>()
            .expect("encoder fwd stack must start with an EmbeddingLayer")
    }

    /// Fill `enc_ranges` from the window's words (cursor-ordered slot runs) and
    /// size the per-word owned buffers. Shared by encode/backward.
    fn set_ranges(&mut self, words: &[Range<usize>]) {
        self.enc_ranges.clear();
        let mut cursor = 0;
        for word in words {
            // One slot per char plus the closing [W] step (if enabled).
            let end = cursor + (word.end - word.start) + ENC_W_EOS as usize;
            self.enc_ranges.push(Range { start: cursor, end });
            cursor = end;
        }
        let char_out = self.output_size;
        if self.e_w.len() < words.len() {
            self.e_w
                .resize_with(words.len(), || vec![0.0; char_out].into());
            self.combine_cache
                .resize_with(words.len(), || self.combine.make_cache());
        }
    }

    /// Encode every word of the window into `e_w`, data-parallel across the
    /// replica pools (words are mutually independent — each resets its own
    /// state). Runs `fwd` and `bwd` over the shared cache slices, then combines
    /// the two readouts per word. Results are read via [`e_w`](Self::e_w).
    pub fn encode_words(&mut self, tokens: &[u16], words: &[Range<usize>]) {
        self.set_ranges(words);
        if words.is_empty() {
            return;
        }

        let w_token = self.w_token;
        let vocab = self.fwd.input_size;
        let enc_ranges = &self.enc_ranges;

        // fwd: c1 … cn [W]  |  bwd: [W] cn … c1
        self.fwd_pool.sync(&self.fwd);
        let fwd_chunks = chunk_words(
            &mut self.fwd_pool.replicas,
            enc_ranges,
            words.len(),
            &mut self.fwd.cache,
        );
        fwd_chunks.into_par_iter().for_each(|chunk| {
            encode_dir(chunk, tokens, words, enc_ranges, w_token, vocab, false);
        });

        self.bwd_pool.sync(&self.bwd);
        let bwd_chunks = chunk_words(
            &mut self.bwd_pool.replicas,
            enc_ranges,
            words.len(),
            &mut self.bwd.cache,
        );
        bwd_chunks.into_par_iter().for_each(|chunk| {
            encode_dir(chunk, tokens, words, enc_ranges, w_token, vocab, true);
        });

        // combine: e_w = W·[fwd_readout ; bwd_readout] + b  (serial, cheap)
        // LinearLayer::forward copies its input into the cache, so build the
        // concatenated readout in a scratch buffer and hand it in. Destructure
        // so `combine` and `combine_cache` borrow disjointly.
        let char_out = self.output_size;
        let Self {
            combine,
            combine_cache,
            e_w,
            fwd,
            bwd,
            ..
        } = self;
        let mut input = vec![0.0; 2 * char_out];
        for w in 0..words.len() {
            let range = enc_ranges[w];
            let fwd_ro = fwd.cache[range.end - 1].last().unwrap().output();
            // bwd reads out at its LAST step too (which corresponds to c1).
            let bwd_ro = bwd.cache[range.end - 1].last().unwrap().output();
            input[..char_out].copy_from_slice(fwd_ro);
            input[char_out..].copy_from_slice(bwd_ro);

            let cache = combine_cache[w]
                .as_any_mut()
                .downcast_mut::<crate::nn::linear::LinearCache>()
                .expect("combine cache type mismatch");
            combine.forward(&input, cache);
            e_w[w].copy_from_slice(&cache.output);
        }
    }

    /// Backprop every word of the window, data-parallel like `encode_words`:
    /// takes the per-word gradients w.r.t. `e_w`, runs `combine` backward to
    /// split them into `fwd`/`bwd` readout grads, seeds each stack at its
    /// readout step, runs BPTT down each stack, and reduces the per-replica
    /// weight gradients into `fwd`/`bwd`. Returns the summed mean-abs seed
    /// gradient (a diagnostic signal).
    pub fn backward_words(&mut self, d_ew_words: &[Box<[f32]>]) -> f32 {
        let n = self.enc_ranges.len();
        if n == 0 {
            return 0.0;
        }

        let char_out = self.output_size;

        // combine backward: d_ew → [d_fwd_ro ; d_bwd_ro] (serial, cheap).
        // Stash each direction's readout gradient per word.
        let mut d_fwd: Vec<Box<[f32]>> = (0..n).map(|_| vec![0.0; char_out].into()).collect();
        let mut d_bwd: Vec<Box<[f32]>> = (0..n).map(|_| vec![0.0; char_out].into()).collect();
        let mut signal = 0.0;
        let mut delta = vec![0.0; char_out];
        let combine = &mut self.combine;
        let combine_cache = &mut self.combine_cache;
        for w in 0..n {
            let d_ew = &d_ew_words[w][..char_out];
            signal += d_ew.iter().map(|x| x.abs()).sum::<f32>() / char_out as f32;

            delta.copy_from_slice(d_ew);
            let cache = combine_cache[w]
                .as_any_mut()
                .downcast_mut::<crate::nn::linear::LinearCache>()
                .expect("combine cache type mismatch");
            combine.backward(&mut delta, cache);
            let dx = &cache.dx;
            d_fwd[w].copy_from_slice(&dx[..char_out]);
            d_bwd[w].copy_from_slice(&dx[char_out..]);
        }
        self.combine.flush_grads();

        // fwd / bwd BPTT, data-parallel per word.
        self.backward_dir(false, &d_fwd);
        self.backward_dir(true, &d_bwd);

        signal
    }

    /// Run one direction's per-word BPTT: seed each word at its readout step
    /// (`enc_end-1`) with the direction's readout grad, run BPTT down the stack,
    /// and reduce replica grads into the master. `reversed` selects `bwd`.
    fn backward_dir(&mut self, reversed: bool, d_readout: &[Box<[f32]>]) {
        let n = self.enc_ranges.len();
        let char_out = self.output_size;
        let max_dim = self.max_layer_dim;
        let enc_ranges = &self.enc_ranges;
        let (stack, pool) = if reversed {
            (&mut self.bwd, &mut self.bwd_pool)
        } else {
            (&mut self.fwd, &mut self.fwd_pool)
        };

        pool.sync(stack);
        let chunks = chunk_words(&mut pool.replicas, enc_ranges, n, &mut stack.cache);

        chunks.into_par_iter().for_each(|chunk| {
            let WorkerChunk {
                replica,
                words: wr,
                cache,
                slot_base,
            } = chunk;
            let mut delta_buf = vec![0.0; max_dim];
            for w in wr.into_iter() {
                let enc_range = enc_ranges[w];
                // Readout (and thus the seed gradient) is at the last slot for
                // BOTH directions: fwd's [W] step and bwd's c1 step both land in
                // enc_end-1 of their own cache.
                for slot in enc_range.into_iter().rev() {
                    if slot == enc_range.end - 1 {
                        delta_buf[..char_out].copy_from_slice(&d_readout[w]);
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
        });

        pool.reduce_into(stack);
    }

    pub fn sgd_step(&mut self, lr: f32, weight_decay: f32) {
        self.fwd.sgd_step(lr, weight_decay);
        self.bwd.sgd_step(lr, weight_decay);
        // combine is an interior projection — decay it like the stacks' linears.
        self.combine.apply_grads(lr, weight_decay);
        self.combine.clear_grads();
        self.fwd_pool.mark_dirty();
        self.bwd_pool.mark_dirty();
    }

    /// Sampling-time encode of a single completed word (uses `forward_sample`
    /// and the single-word cache slots `0..=n`). Returns `e_w`.
    pub fn encode_word_sample(&mut self, word: &[u16]) -> &[f32] {
        let n = word.len();
        debug_assert!(n > 0, "cannot encode an empty word");
        let char_out = self.output_size;
        let readout = n - 1 + ENC_W_EOS as usize;

        // fwd: c1 … cn [W]   (readout at slot n-1+[W])
        for layer in &mut self.fwd.layers {
            layer.reset_state();
        }
        let fwd_toks = dir_stream(word, self.w_token, false);
        for (k, tok) in fwd_toks.into_iter().enumerate() {
            self.char_input[tok as usize] = 1.0;
            forward_sample_step(
                &mut self.fwd.layers,
                &mut self.fwd.cache[k],
                &self.char_input,
            );
            self.char_input[tok as usize] = 0.0;
        }

        // bwd: cn … c1 [W]   (readout at the same slot index, on the [W] step)
        for layer in &mut self.bwd.layers {
            layer.reset_state();
        }
        let bwd_toks = dir_stream(word, self.w_token, true);
        for (k, tok) in bwd_toks.into_iter().enumerate() {
            self.char_input[tok as usize] = 1.0;
            forward_sample_step(
                &mut self.bwd.layers,
                &mut self.bwd.cache[k],
                &self.char_input,
            );
            self.char_input[tok as usize] = 0.0;
        }

        // combine into e_w[0] (sampling uses one word at a time).
        if self.e_w.is_empty() {
            self.e_w.push(vec![0.0; char_out].into());
            self.combine_cache.push(self.combine.make_cache());
        }
        let fwd_ro = self.fwd.cache[readout].last().unwrap().output();
        let bwd_ro = self.bwd.cache[readout].last().unwrap().output();
        let mut input = vec![0.0; 2 * char_out];
        input[..char_out].copy_from_slice(fwd_ro);
        input[char_out..].copy_from_slice(bwd_ro);
        let combine = &self.combine;
        let cache = self.combine_cache[0]
            .as_any_mut()
            .downcast_mut::<crate::nn::linear::LinearCache>()
            .expect("combine cache type mismatch");
        combine.forward(&input, cache);
        self.e_w[0].copy_from_slice(&cache.output);
        &self.e_w[0][..]
    }

    /// True once `make_cache` has allocated the forward caches (sampling guard).
    pub fn cache_ready(&self) -> bool {
        !self.fwd.cache.is_empty()
    }
}

/// One direction's parallel encode over a worker's word chunk. `reversed` feeds
/// the word's characters back-to-front but keeps the closing `[W]` LAST, so
/// both directions end on the end-of-word marker and read out there:
///
///   fwd:  h a l l o [W]
///   bwd:  o l l a h [W]
///
/// The `[W]` step is what tells the cell the word is complete, so both readouts
/// are taken at a step that carries that signal (the alternative — reversing the
/// whole sequence to `[W] o l l a h` — reads the backward direction out at an
/// ordinary char step with no completion marker). Either way the readout lands
/// in the chunk's last slot for the word.
fn encode_dir(
    chunk: WorkerChunk,
    tokens: &[u16],
    words: &[Range<usize>],
    enc_ranges: &[Range<usize>],
    w_token: u16,
    vocab: usize,
    reversed: bool,
) {
    let WorkerChunk {
        replica,
        words: wr,
        cache,
        slot_base,
    } = chunk;
    let mut char_input = vec![0.0; vocab];
    for w in wr.into_iter() {
        for layer in &mut replica.layers {
            layer.reset_state();
        }
        let base = enc_ranges[w].start - slot_base;
        let word = words[w];
        let stream = dir_stream(&tokens[word.start..word.end], w_token, reversed);
        for (k, tok) in stream.into_iter().enumerate() {
            let tok = tok as usize;
            char_input[tok] = 1.0;
            Sequential::forward_step(&mut replica.layers, &mut cache[base + k], &char_input);
            char_input[tok] = 0.0;
        }
    }
}

/// The token stream one direction consumes for a word: its characters (reversed
/// for `bwd`) followed by the closing `[W]`. Both directions end on `[W]`, so
/// the readout — always the last step — carries the end-of-word signal.
///
///   `hallo` → fwd `h a l l o [W]` · bwd `o l l a h [W]`
pub(crate) fn dir_stream(chars: &[u16], w_token: u16, reversed: bool) -> Vec<u16> {
    let mut out: Vec<u16> = if reversed {
        chars.iter().rev().copied().collect()
    } else {
        chars.to_vec()
    };
    if ENC_W_EOS {
        out.push(w_token);
    }
    out
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer_utf8::Utf8Tokenizer;

    /// Both encoder directions must end on the `[W]` marker, so each reads out
    /// at a step that knows the word is complete:
    ///   `hallo` → fwd `h a l l o [W]` · bwd `o l l a h [W]`
    /// Reversing the WHOLE sequence instead (`[W] o l l a h`) would read the
    /// backward direction out at a plain char step — that asymmetry is the bug
    /// this pins against.
    #[test]
    fn both_directions_end_on_the_w_marker() {
        let tok = Utf8Tokenizer::new();
        let w = tok.w_token();
        let chars = tok.to_tokens("hallo");

        let fwd = dir_stream(&chars, w, false);
        let bwd = dir_stream(&chars, w, true);

        assert_eq!(tok.to_text(&fwd[..fwd.len() - 1]), "hallo");
        assert_eq!(tok.to_text(&bwd[..bwd.len() - 1]), "ollah");
        // The readout step (the last one) is the [W] marker in BOTH directions.
        assert_eq!(*fwd.last().unwrap(), w);
        assert_eq!(*bwd.last().unwrap(), w);
        // Same length, so both share the cache slot range and readout index.
        assert_eq!(fwd.len(), bwd.len());
    }

    /// A one-char word is its own reverse — the two directions differ only by
    /// their weights, and the readout is still the [W] step.
    #[test]
    fn single_char_word_is_symmetric() {
        let tok = Utf8Tokenizer::new();
        let w = tok.w_token();
        let chars = tok.to_tokens("x");
        assert_eq!(dir_stream(&chars, w, false), dir_stream(&chars, w, true));
        assert_eq!(*dir_stream(&chars, w, true).last().unwrap(), w);
    }
}
