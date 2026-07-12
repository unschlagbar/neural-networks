//! Batched hierarchical (HAT-style) model — nn2 port of `crate::hierarchical`.
//!
//! Three coupled stages, run phase-by-phase over a window of words. The **word**
//! is the batch axis for the encoder and decoder (the old system ran words
//! data-parallel across a replica pool; here they are the leading batch dim),
//! while the backbone is a single sequence over the words carrying recurrent
//! state across them.
//!
//!   1. encoder   — per word, `Embedding → sLSTM block×N`, read out `e_w` at the
//!                  closing `[W]` step.
//!   2. backbone  — `Linear → (sLSTM/mLSTM block)×N → Linear`, autoregress over
//!                  `e_w`; output `o_w` = context for word w+1.
//!   3. decoder   — per word, first step is the injected backbone context `o`
//!                  (BOS slot), later steps feed the previous char via the
//!                  **tied** encoder char table; `sLSTM block×N → RMSNorm →
//!                  Linear head → SoftCap`, predicting the word's chars + `[W]`.
//!
//! Norm placement matches `model.rs::build_hierarchical_model`: the decoder's
//! pre-head RMSNorm is the **only** stage-level norm (the blocks keep their own
//! internal pre/post norms). Encoder and backbone have none.
//!
//! Words are end-padded to the batch's max length; the recurrence is causal and
//! readout/loss are masked, so padding never corrupts a valid position.

use crate::config::LOGIT_SOFTCAP;
use crate::nn2::block::{MLstmBlock, SLstmBlock};
use crate::nn2::loss::softmax_rows;
use crate::nn2::optim::AdamCfg;
use crate::nn2::word_encoder::WordEncoder;
use crate::nn2::{Embedding, Linear, RmsNorm, SoftCap};
use crate::tensor::Tensor;

/// Backbone block: alternating sLSTM / mLSTM (they have different types, so a
/// small enum dispatches forward/backward/step).
pub enum AnyBlock {
    S(SLstmBlock),
    M(MLstmBlock),
}

impl AnyBlock {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        match self {
            AnyBlock::S(b) => b.forward(x),
            AnyBlock::M(b) => b.forward(x),
        }
    }
    fn backward(&mut self, dy: &Tensor) -> Tensor {
        match self {
            AnyBlock::S(b) => b.backward(dy),
            AnyBlock::M(b) => b.backward(dy),
        }
    }
    fn step(&mut self, cfg: &AdamCfg) {
        match self {
            AnyBlock::S(b) => b.step(cfg),
            AnyBlock::M(b) => b.step(cfg),
        }
    }
}

pub struct Hierarchical {
    pub char_embed: Embedding, // [vocab, HC], tied between encoder and decoder
    pub encoder: WordEncoder,

    pub bb_front: Linear,         // HC → WH
    pub bb_blocks: Vec<AnyBlock>, // WH
    pub bb_back: Linear,          // WH → HC (context)

    pub dec_blocks: Vec<SLstmBlock>, // HC
    pub dec_norm: RmsNorm,           // HC
    pub dec_head: Linear,            // HC → vocab
    pub softcap: SoftCap,

    vocab: usize,
    hc: usize,
    w_token: usize,

    // Saved between forward and backward.
    dec_ids: Vec<Vec<usize>>, // per decoded word: its char ids (for tied-embedding grad scatter)
    dec_tmax: usize,
    o_dims: (usize,), // decode_words
}

/// Config for the hierarchical stack.
pub struct HierCfg {
    pub vocab: usize,
    pub hc: usize, // char/context hidden (tied embedding + decoder input width)
    pub wh: usize, // backbone width
    pub up: usize, // SwiGLU up-projection width for all blocks
    pub enc_blocks: usize,
    pub bb_blocks: usize, // alternates sLSTM (even) / mLSTM (odd)
    pub dec_blocks: usize,
    pub heads: usize, // mLSTM heads
    pub dqk: usize,
    pub w_token: usize,
}

impl Hierarchical {
    pub fn new(c: &HierCfg) -> Self {
        let bb_blocks = (0..c.bb_blocks)
            .map(|i| {
                if i % 2 == 0 {
                    AnyBlock::S(SLstmBlock::new_slstm(c.wh, c.up))
                } else {
                    AnyBlock::M(MLstmBlock::new_mlstm(c.wh, c.up, c.heads, c.dqk))
                }
            })
            .collect();
        Self {
            char_embed: Embedding::new(c.vocab, c.hc),
            encoder: WordEncoder::new(c.hc, c.up, c.enc_blocks),
            bb_front: Linear::new(c.hc, c.wh),
            bb_blocks,
            bb_back: Linear::new(c.wh, c.hc),
            dec_blocks: (0..c.dec_blocks).map(|_| SLstmBlock::new_slstm(c.hc, c.up)).collect(),
            dec_norm: RmsNorm::new(c.hc),
            dec_head: Linear::new(c.hc, c.vocab),
            softcap: SoftCap::new(LOGIT_SOFTCAP),
            vocab: c.vocab,
            hc: c.hc,
            w_token: c.w_token,
            dec_ids: Vec::new(),
            dec_tmax: 0,
            o_dims: (0,),
        }
    }

    /// Forward + backward over one window; accumulates all grads and returns the
    /// mean decode cross-entropy. `tokens` are char ids; `words` are `(start,
    /// end)` char ranges. Word 0 is encode-only; words 1..n are decoded.
    pub fn forward_backward(&mut self, tokens: &[usize], words: &[(usize, usize)]) -> f32 {
        let n = words.len();
        if n < 2 {
            return 0.0;
        }
        let dw = n - 1; // decoded words (and encoded words)
        let hc = self.hc;
        let vocab = self.vocab;
        self.o_dims = (dw,);

        // ---- PHASE 1: ENCODER ----------------------------------------------
        // Encode words 0..dw. Sequence = chars + [W]; readout at the [W] step.
        let enc_lens: Vec<usize> = (0..dw).map(|w| words[w].1 - words[w].0).collect();
        let enc_tmax = enc_lens.iter().map(|&l| l + 1).max().unwrap();
        let mut enc_ids = vec![0usize; dw * enc_tmax];
        let mut readout = vec![0usize; dw];
        for w in 0..dw {
            let (s, _) = words[w];
            for k in 0..enc_lens[w] {
                enc_ids[w * enc_tmax + k] = tokens[s + k];
            }
            enc_ids[w * enc_tmax + enc_lens[w]] = self.w_token; // [W]
            readout[w] = enc_lens[w];
        }
        let embedded = self.char_embed.forward(&enc_ids); // [dw*enc_tmax, HC]
        let e_w = self.encoder.forward(&embedded.reshape(&[dw, enc_tmax, hc]), &readout); // [dw, HC]

        // ---- PHASE 2: BACKBONE ---------------------------------------------
        // Autoregress e_w as one sequence; o[w] is the context for word w+1.
        let bb_in = self.bb_front.forward(&e_w); // [dw, WH]
        let wh = bb_in.cols();
        let mut h = bb_in.reshape(&[1, dw, wh]);
        for blk in &mut self.bb_blocks {
            h = blk.forward(&h);
        }
        let o = self.bb_back.forward(&h.reshape(&[dw, wh])); // [dw, HC]  o[w] decodes word w+1

        // ---- PHASE 3: DECODER ----------------------------------------------
        // Word w's decode target is word w+1. Slot 0 = o[w]; slot k = embed(prev char).
        let dec_lens: Vec<usize> = (0..dw).map(|w| words[w + 1].1 - words[w + 1].0).collect();
        let dec_tmax = dec_lens.iter().map(|&l| l + 1).max().unwrap();
        self.dec_tmax = dec_tmax;
        self.dec_ids = (0..dw)
            .map(|w| {
                let (s, _) = words[w + 1];
                (0..dec_lens[w]).map(|k| tokens[s + k]).collect()
            })
            .collect();

        let mut dec_in = Tensor::zeros(&[dw, dec_tmax, hc]);
        let mut targets = vec![0usize; dw * dec_tmax];
        let mut mask = vec![false; dw * dec_tmax];
        for w in 0..dw {
            let m = dec_lens[w];
            // slot 0: injected context.
            let base = (w * dec_tmax) * hc;
            dec_in.data[base..base + hc].copy_from_slice(&o.data[w * hc..(w + 1) * hc]);
            // slots 1..=m: embed(prev char); targets/mask over 0..=m.
            for k in 1..=m {
                let cid = self.dec_ids[w][k - 1];
                let src = cid * hc;
                let dst = (w * dec_tmax + k) * hc;
                dec_in.data[dst..dst + hc].copy_from_slice(&self.char_embed.table.data[src..src + hc]);
            }
            for k in 0..m {
                targets[w * dec_tmax + k] = self.dec_ids[w][k];
                mask[w * dec_tmax + k] = true;
            }
            targets[w * dec_tmax + m] = self.w_token;
            mask[w * dec_tmax + m] = true;
        }

        let mut hd = dec_in.clone();
        for blk in &mut self.dec_blocks {
            hd = blk.forward(&hd);
        }
        let hdn = self.dec_norm.forward(&hd.reshape(&[dw * dec_tmax, hc]));
        let logits = self.dec_head.forward(&hdn);
        let capped = self.softcap.forward(&logits); // [dw*dec_tmax, vocab]

        // Masked cross-entropy.
        let (loss, d_capped) = masked_ce(&capped, &targets, &mask, vocab);

        // ---- BACKWARD ------------------------------------------------------
        let d_logits = self.softcap.backward(&d_capped);
        let d_hdn = self.dec_head.backward(&d_logits);
        let d_hd_flat = self.dec_norm.backward(&d_hdn);
        let mut d_hd = d_hd_flat.reshape(&[dw, dec_tmax, hc]);
        for blk in self.dec_blocks.iter_mut().rev() {
            d_hd = blk.backward(&d_hd);
        }
        // Split decoder input grad: slot 0 → d_o; slots 1..=m → tied char table.
        let mut d_o = Tensor::zeros(&[dw, hc]);
        for w in 0..dw {
            let base = (w * dec_tmax) * hc;
            d_o.data[w * hc..(w + 1) * hc].copy_from_slice(&d_hd.data[base..base + hc]);
            let m = dec_lens[w];
            for k in 1..=m {
                let cid = self.dec_ids[w][k - 1];
                let src = (w * dec_tmax + k) * hc;
                let dst = cid * hc;
                for i in 0..hc {
                    self.char_embed.dtable.data[dst + i] += d_hd.data[src + i];
                }
            }
        }

        // Backbone backward.
        let d_hn = self.bb_back.backward(&d_o);
        let mut d_h = d_hn.reshape(&[1, dw, wh]);
        for blk in self.bb_blocks.iter_mut().rev() {
            d_h = blk.backward(&d_h);
        }
        let d_e_w = self.bb_front.backward(&d_h.reshape(&[dw, wh])); // [dw, HC]

        // Encoder backward → grad w.r.t. embedded input → tied char table.
        let d_embedded = self.encoder.backward(&d_e_w); // [dw, enc_tmax, HC]
        self.char_embed.backward(&d_embedded.reshape(&[dw * enc_tmax, hc]));

        loss
    }

    pub fn step(&mut self, cfg: &AdamCfg) {
        self.char_embed.step(cfg);
        self.encoder.step(cfg);
        self.bb_front.step(cfg);
        for blk in &mut self.bb_blocks {
            blk.step(cfg);
        }
        self.bb_back.step(cfg);
        for blk in &mut self.dec_blocks {
            blk.step(cfg);
        }
        self.dec_norm.step(cfg);
        self.dec_head.step_wd(cfg, false); // logit head: no weight decay
    }
}

/// Row-wise softmax cross-entropy over only the masked (valid) rows. Returns the
/// mean loss and `d_capped` (`(softmax − onehot)/num_valid` on valid rows, zero
/// elsewhere).
fn masked_ce(capped: &Tensor, targets: &[usize], mask: &[bool], vocab: usize) -> (f32, Tensor) {
    let mut probs = softmax_rows(capped);
    let num_valid = mask.iter().filter(|&&m| m).count().max(1) as f32;
    let inv = 1.0 / num_valid;
    let mut loss = 0.0;
    for (r, (&t, &m)) in targets.iter().zip(mask).enumerate() {
        let row = &mut probs.data[r * vocab..(r + 1) * vocab];
        if !m {
            row.iter_mut().for_each(|v| *v = 0.0);
            continue;
        }
        loss -= row[t].max(1e-30).ln();
        row[t] -= 1.0;
        row.iter_mut().for_each(|v| *v *= inv);
    }
    (loss * inv, probs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end: the three-stage model must memorize a tiny fixed window,
    /// driving the decode loss down. Exercises the full wiring — tied char
    /// embedding, backbone context injection at the decoder BOS slot, masked
    /// per-word CE, and AdamW across every stage.
    #[test]
    fn hierarchical_memorizes_window() {
        // vocab 0..7 chars, 8 = [W]. Four short words in one window.
        let cfg = HierCfg {
            vocab: 9, hc: 16, wh: 24, up: 24,
            enc_blocks: 1, bb_blocks: 2, dec_blocks: 1,
            heads: 2, dqk: 8, w_token: 8,
        };
        let mut model = Hierarchical::new(&cfg);

        // tokens + word ranges (variable lengths, to exercise padding/masking).
        let tokens = vec![1, 2, 3, /*|*/ 4, 5, /*|*/ 6, 7, 1, /*|*/ 2, 3];
        let words = vec![(0, 3), (3, 5), (5, 8), (8, 10)];

        let mut opt = AdamCfg::new(5e-3, 0.0);
        let first = model.forward_backward(&tokens, &words);
        // grads from the loss-measuring pass above are discarded via the step's
        // zero; run a clean training loop.
        for _ in 0..250 {
            let _ = model.forward_backward(&tokens, &words);
            opt.t += 1;
            model.step(&opt);
        }
        let last = model.forward_backward(&tokens, &words);
        assert!(last < first * 0.4, "decode loss did not fall enough: {first} -> {last}");
    }
}
