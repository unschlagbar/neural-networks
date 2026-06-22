// Bidirectional per-word encoder for the hierarchical model.
//
// Encodes ONE word into a fixed-size embedding `e_w`. Two independent sLSTM/LSTM
// stacks read the word in opposite directions, each prefixed/suffixed with the
// `[W]` marker:
//
//   forward :  [W] c1 c2 … cn      → read the state at the LAST char (cn): the
//                                     forward pass has seen the whole word there.
//   backward:  cn … c2 c1 [W]      → read the state at the trailing [W]: the
//                                     backward pass has likewise seen the whole word.
//
// `combine` merges `concat(fwd, bwd) ∈ ℝ^{2H}` into `e_w ∈ ℝ^H`, then `combine_norm`
// (RMSNorm) normalises it. The forward cache layout per word is one slot per token
// plus one extra `[W]` step, addressed by a running cursor (`enc_ranges`).
//
// Pulled out of `hierarchical.rs` so the orchestration there stays readable; the
// encoder owns its own scratch buffers and gradient bookkeeping.

use std::range::Range;

use crate::{
    hierarchical::backward_through_layers,
    nn::{
        linear_nb::{LinearNBCache, LinearNBLayer},
        rms_norm::{RMSNorm, RMSNormCache},
    },
    nn_layer::{DynCache, NnLayer},
    sequential::Sequential,
};

pub struct BiEncoder {
    /// Forward direction stack (`[W] c1 … cn`).
    pub char_fwd: Sequential,
    /// Backward direction stack (`cn … c1 [W]`).
    pub char_bwd: Sequential,
    /// Merges `concat(fwd, bwd) ∈ ℝ^{2H}` → word embedding `e_w ∈ ℝ^H`.
    pub combine: LinearNBLayer,
    pub combine_norm: RMSNorm,

    combine_caches: Vec<LinearNBCache>,
    combine_norm_caches: Vec<RMSNormCache>,

    /// Reusable one-hot input buffer (size `vocab`).
    char_input: Box<[f32]>,
    /// `concat(fwd, bwd)` scratch (size `2H`).
    e_pre_buf: Box<[f32]>,
    /// Gradient w.r.t. `e_pre` (size `2H`).
    d_e_pre_buf: Box<[f32]>,
    /// BPTT delta scratch as it flows down one direction's stack.
    delta_buf: Box<[f32]>,

    /// Per encode-word: cache slot range `[start, end)` (length `word_len + 1`,
    /// including the `[W]` step). Indexes both `char_fwd.cache` and `char_bwd.cache`.
    enc_ranges: Vec<Range<usize>>,
    /// Running cache-slot cursor across the words of the current window.
    enc_cursor: usize,

    w_token: u16,
    /// Encoder output dimension `H` (= each direction's output size).
    pub output_size: usize,
}

impl BiEncoder {
    pub fn new(
        char_fwd: Sequential,
        char_bwd: Sequential,
        combine: LinearNBLayer,
        combine_norm: RMSNorm,
        vocab_size: usize,
        w_token: u16,
    ) -> Self {
        let char_out = char_fwd.output_size;
        assert_eq!(
            char_fwd.input_size, vocab_size,
            "forward encoder.input_size must equal vocab_size"
        );
        assert_eq!(
            char_bwd.input_size, vocab_size,
            "backward encoder.input_size must equal vocab_size"
        );
        assert_eq!(
            char_bwd.output_size, char_out,
            "both encoder directions must share the same output size"
        );
        assert_eq!(
            combine.input_size(),
            2 * char_out,
            "combine.input must equal 2 * encoder output"
        );
        assert_eq!(
            combine.output_size(),
            char_out,
            "combine.output must equal the encoder output size"
        );

        // BPTT delta scratch must hold the largest layer dim across both directions.
        let max_layer_dim = char_fwd
            .layers
            .iter()
            .chain(char_bwd.layers.iter())
            .map(|l| l.input_size().max(l.output_size()))
            .max()
            .unwrap_or(char_out)
            .max(char_out);

        Self {
            char_fwd,
            char_bwd,
            combine,
            combine_norm,
            combine_caches: Vec::new(),
            combine_norm_caches: Vec::new(),
            char_input: vec![0.0; vocab_size].into(),
            e_pre_buf: vec![0.0; 2 * char_out].into(),
            d_e_pre_buf: vec![0.0; 2 * char_out].into(),
            delta_buf: vec![0.0; max_layer_dim].into(),
            enc_ranges: Vec::new(),
            enc_cursor: 0,
            w_token,
            output_size: char_out,
        }
    }

    /// `slots` = max TOKEN span of a window (cache depth); `word_steps` = max
    /// WORDS per window (one combine cache per word).
    pub fn make_cache(&mut self, slots: usize, word_steps: usize) {
        let char_out = self.output_size;
        self.char_fwd.make_cache(slots);
        self.char_bwd.make_cache(slots);
        self.combine_caches = (0..word_steps.max(1))
            .map(|_| LinearNBCache::new(2 * char_out, char_out))
            .collect();
        self.combine_norm_caches = (0..word_steps.max(1))
            .map(|_| self.combine_norm.alloc_cache())
            .collect();
    }

    /// Reset recurrent state and the per-window cursor (call once per window).
    pub fn reset(&mut self) {
        for layer in &mut self.char_fwd.layers {
            layer.reset_state();
        }
        for layer in &mut self.char_bwd.layers {
            layer.reset_state();
        }
        self.enc_ranges.clear();
        self.enc_cursor = 0;
    }

    pub fn num_words(&self) -> usize {
        self.enc_ranges.len()
    }

    /// Clear BPTT gradient-carry state in both directions (degenerate windows).
    pub fn reset_bptt(&mut self) {
        for layer in &mut self.char_fwd.layers {
            layer.reset_bptt_state();
        }
        for layer in &mut self.char_bwd.layers {
            layer.reset_bptt_state();
        }
    }

    pub fn enc_range(&self, w: usize) -> Range<usize> {
        self.enc_ranges[w]
    }

    /// Word embedding `e_w` of the `w`-th encoded word (valid after `encode_word`).
    pub fn e_w(&self, w: usize) -> &[f32] {
        &self.combine_norm_caches[w].output
    }

    /// Encode word `w` (token range `word` into `tokens`) into `e_w`, advancing the
    /// cache cursor. The result is read via [`e_w`](Self::e_w).
    pub fn encode_word(&mut self, w: usize, tokens: &[u16], word: Range<usize>) {
        let char_out = self.output_size;
        let w_tok = self.w_token as usize;
        let word_len = word.end - word.start;

        let enc_start = self.enc_cursor;
        let enc_end = enc_start + word_len + 1; // +1 for the [W] step
        self.enc_cursor = enc_end;
        self.enc_ranges.push(Range {
            start: enc_start,
            end: enc_end,
        });

        // Forward: [W], c1, …, cn ; readout at the last char (enc_end-1).
        for layer in &mut self.char_fwd.layers {
            layer.reset_state();
        }
        self.char_input[w_tok] = 1.0;
        Sequential::forward_step(
            &mut self.char_fwd.layers,
            &mut self.char_fwd.cache[enc_start],
            &self.char_input,
        );
        self.char_input[w_tok] = 0.0;
        for (k, u) in word.into_iter().enumerate() {
            let tok = tokens[u] as usize;
            self.char_input[tok] = 1.0;
            Sequential::forward_step(
                &mut self.char_fwd.layers,
                &mut self.char_fwd.cache[enc_start + 1 + k],
                &self.char_input,
            );
            self.char_input[tok] = 0.0;
        }

        // Backward: cn, …, c1, [W] ; readout at the trailing [W] (enc_end-1).
        for layer in &mut self.char_bwd.layers {
            layer.reset_state();
        }
        for k in 0..word_len {
            let tok = tokens[word.end - 1 - k] as usize;
            self.char_input[tok] = 1.0;
            Sequential::forward_step(
                &mut self.char_bwd.layers,
                &mut self.char_bwd.cache[enc_start + k],
                &self.char_input,
            );
            self.char_input[tok] = 0.0;
        }
        self.char_input[w_tok] = 1.0;
        Sequential::forward_step(
            &mut self.char_bwd.layers,
            &mut self.char_bwd.cache[enc_end - 1],
            &self.char_input,
        );
        self.char_input[w_tok] = 0.0;

        // Combine concat(fwd@last_char, bwd@[W]) → e_w → RMSNorm.
        self.e_pre_buf[..char_out]
            .copy_from_slice(self.char_fwd.cache[enc_end - 1].last().unwrap().output());
        self.e_pre_buf[char_out..2 * char_out]
            .copy_from_slice(self.char_bwd.cache[enc_end - 1].last().unwrap().output());
        self.combine
            .forward(&self.e_pre_buf[..2 * char_out], &mut self.combine_caches[w]);
        self.combine_norm.forward_into(
            &self.combine_caches[w].output,
            &mut self.combine_norm_caches[w],
        );
    }

    /// Backprop one word: takes the gradient w.r.t. `e_w` (`d_ew`, length `H`),
    /// runs combine + both encoder directions, accumulates weight gradients, and
    /// returns the mean-abs gradient flowing into `e_pre` (a diagnostic signal).
    pub fn backward_word(&mut self, w: usize, d_ew: &[f32]) -> f32 {
        let char_out = self.output_size;

        // Combine backward → split into d_fwd ‖ d_bwd.
        self.combine_norm
            .backward_into(&d_ew[..char_out], &mut self.combine_norm_caches[w]);
        self.delta_buf[..char_out].copy_from_slice(&self.combine_norm_caches[w].dx);
        self.combine
            .backward(&mut self.delta_buf[..char_out], &mut self.combine_caches[w]);
        self.d_e_pre_buf[..2 * char_out].copy_from_slice(&self.combine_caches[w].dx);
        let signal = self.d_e_pre_buf.iter().map(|x| x.abs()).sum::<f32>() / (2 * char_out) as f32;

        let enc_range = self.enc_ranges[w];

        // Forward direction: readout (and thus the seed gradient) is at enc_end-1.
        for slot in enc_range.into_iter().rev() {
            if slot == enc_range.end - 1 {
                self.delta_buf[..char_out].copy_from_slice(&self.d_e_pre_buf[..char_out]);
            } else {
                self.delta_buf[..char_out].fill(0.0);
            }
            backward_through_layers(
                &mut self.char_fwd.layers,
                &mut self.char_fwd.cache[slot],
                &mut self.delta_buf,
                char_out,
            );
        }
        for layer in &mut self.char_fwd.layers {
            layer.accumulate_init_grad();
            layer.reset_bptt_state();
        }

        // Backward direction: readout is the trailing [W] at enc_end-1.
        for slot in enc_range.into_iter().rev() {
            if slot == enc_range.end - 1 {
                self.delta_buf[..char_out]
                    .copy_from_slice(&self.d_e_pre_buf[char_out..2 * char_out]);
            } else {
                self.delta_buf[..char_out].fill(0.0);
            }
            backward_through_layers(
                &mut self.char_bwd.layers,
                &mut self.char_bwd.cache[slot],
                &mut self.delta_buf,
                char_out,
            );
        }
        for layer in &mut self.char_bwd.layers {
            layer.accumulate_init_grad();
            layer.reset_bptt_state();
        }

        signal
    }

    pub fn sgd_step(&mut self, lr: f32) {
        self.char_fwd.sgd_step(lr);
        self.char_bwd.sgd_step(lr);
        self.combine.apply_grads(lr);
        self.combine.clear_grads();
        self.combine_norm.apply_grads(lr);
        self.combine_norm.clear_grads();
    }

    /// Sampling-time encode of a single completed word (uses `forward_sample` and
    /// the single-word cache slots `0..=n`). Returns `e_w`.
    pub fn encode_word_sample(&mut self, word: &[u16]) -> &[f32] {
        let char_out = self.output_size;
        let w_tok = self.w_token as usize;
        let n = word.len();

        // Forward: [W], c1, …, cn — readout at the last char (slot n).
        for layer in &mut self.char_fwd.layers {
            layer.reset_state();
        }
        self.char_input[w_tok] = 1.0;
        forward_sample_step(
            &mut self.char_fwd.layers,
            &mut self.char_fwd.cache[0],
            &self.char_input,
        );
        self.char_input[w_tok] = 0.0;
        for (k, &tok) in word.iter().enumerate() {
            self.char_input[tok as usize] = 1.0;
            forward_sample_step(
                &mut self.char_fwd.layers,
                &mut self.char_fwd.cache[1 + k],
                &self.char_input,
            );
            self.char_input[tok as usize] = 0.0;
        }
        self.e_pre_buf[..char_out].copy_from_slice(self.char_fwd.cache[n].last().unwrap().output());

        // Backward: cn, …, c1, [W] — readout at the trailing [W] (slot n).
        for layer in &mut self.char_bwd.layers {
            layer.reset_state();
        }
        for k in 0..n {
            let tok = word[n - 1 - k] as usize;
            self.char_input[tok] = 1.0;
            forward_sample_step(
                &mut self.char_bwd.layers,
                &mut self.char_bwd.cache[k],
                &self.char_input,
            );
            self.char_input[tok] = 0.0;
        }
        self.char_input[w_tok] = 1.0;
        forward_sample_step(
            &mut self.char_bwd.layers,
            &mut self.char_bwd.cache[n],
            &self.char_input,
        );
        self.char_input[w_tok] = 0.0;
        self.e_pre_buf[char_out..2 * char_out]
            .copy_from_slice(self.char_bwd.cache[n].last().unwrap().output());

        self.combine
            .forward(&self.e_pre_buf[..2 * char_out], &mut self.combine_caches[0]);
        self.combine_norm.forward_into(
            &self.combine_caches[0].output,
            &mut self.combine_norm_caches[0],
        );
        &self.combine_norm_caches[0].output
    }

    /// True once `make_cache` has allocated the forward cache (sampling guard).
    pub fn cache_ready(&self) -> bool {
        !self.char_fwd.cache.is_empty()
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
