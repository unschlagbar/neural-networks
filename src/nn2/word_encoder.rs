//! Batched per-word encoder for the hierarchical model (nn2 port).
//!
//! Encodes a batch of words into fixed-size embeddings `e_w`. The batch axis is
//! the **word** (the old system ran words data-parallel across a replica pool;
//! here they are simply the leading batch dimension). Each word is a padded
//! char-plus-`[W]` sequence `[W_count, T, Hc]` that has already been embedded by
//! the caller (the char table is tied with the decoder and owned by
//! `Hierarchical`). The stack is `sLSTM block × N`; `e_w` for word `w` is read out
//! at that word's closing `[W]` step (`readout[w]`), so the state has seen the
//! whole word and knows it is complete.
//!
//! No trailing norm: the real model (`model.rs::build_hierarchical_model`) ends
//! the encoder at the last `slstm_block`. The only RMSNorm in the whole
//! hierarchical stack is the decoder's, right before the logit head.
//!
//! Words encode independently — the recurrent cells reset state per sequence
//! (i.e. per word) at the start of each block forward.

use crate::nn2::SLstmBlock;
use crate::nn2::optim::AdamCfg;
use crate::tensor::Tensor;

pub struct WordEncoder {
    pub blocks: Vec<SLstmBlock>,
    hc: usize,
    // Saved for backward.
    dims: (usize, usize), // (W_count, T)
    readout: Vec<usize>,
}

impl WordEncoder {
    pub fn new(hc: usize, up: usize, n_blocks: usize) -> Self {
        Self {
            blocks: (0..n_blocks)
                .map(|_| SLstmBlock::new_slstm(hc, up))
                .collect(),
            hc,
            dims: (0, 0),
            readout: Vec::new(),
        }
    }

    /// `embedded` is `[W, T, Hc]` (words × padded-length × char-hidden), already
    /// gathered from the tied char table. `readout[w]` is the step index of
    /// word `w`'s closing `[W]` token. Returns `e_w` `[W, Hc]`.
    pub fn forward(&mut self, embedded: &Tensor, readout: &[usize]) -> Tensor {
        assert_eq!(
            embedded.rank(),
            3,
            "WordEncoder::forward expects [W, T, Hc]"
        );
        let (w, t, hc) = (embedded.shape[0], embedded.shape[1], embedded.shape[2]);
        assert_eq!(hc, self.hc, "WordEncoder — width mismatch");
        assert_eq!(readout.len(), w, "WordEncoder — readout len != words");
        self.dims = (w, t);
        self.readout = readout.to_vec();

        let mut h = embedded.clone();
        for blk in &mut self.blocks {
            h = blk.forward(&h);
        }

        // Gather e_w at each word's [W] step.
        let mut e_w = Tensor::zeros(&[w, hc]);
        for wi in 0..w {
            let src = (wi * t + readout[wi]) * hc;
            e_w.data[wi * hc..(wi + 1) * hc].copy_from_slice(&h.data[src..src + hc]);
        }
        e_w
    }

    /// Given `d_e_w` `[W, Hc]`, returns the gradient w.r.t. the embedded input
    /// `[W, T, Hc]` (for the caller to scatter into the tied char table).
    pub fn backward(&mut self, d_e_w: &Tensor) -> Tensor {
        let (w, t) = self.dims;
        let hc = self.hc;
        assert_eq!(
            d_e_w.rows(),
            w,
            "WordEncoder::backward — word count mismatch"
        );

        // Scatter d_e_w back to the [W]-step rows; all other rows have zero grad.
        let mut d_h = Tensor::zeros(&[w, t, hc]);
        for wi in 0..w {
            let dst = (wi * t + self.readout[wi]) * hc;
            d_h.data[dst..dst + hc].copy_from_slice(&d_e_w.data[wi * hc..(wi + 1) * hc]);
        }

        for blk in self.blocks.iter_mut().rev() {
            d_h = blk.backward(&d_h);
        }
        d_h
    }

    pub fn zero_grad(&mut self) {
        for blk in &mut self.blocks {
            blk.zero_grad();
        }
    }

    pub fn step(&mut self, cfg: &AdamCfg) {
        for blk in &mut self.blocks {
            blk.step(cfg);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Directional FD through the whole encoder stack + the [W]-step readout
    /// (contains recurrent cells → loose tol like the cell tests).
    #[test]
    fn readout_grads_match_fd() {
        let (w, t, hc, up) = (3, 5, 8, 12);
        let mut enc = WordEncoder::new(hc, up, 2);
        let readout = [4usize, 3, 4]; // each word's [W] step index (< T)
        let embedded = Tensor::random(&[w, t, hc], 0.5);
        let g = Tensor::random(&[w, hc], 1.0);

        let _e = enc.forward(&embedded, &readout);
        let d_emb = enc.backward(&g);

        let loss = |enc: &mut WordEncoder, x: &Tensor| -> f32 {
            let e = enc.forward(x, &readout);
            e.data.iter().zip(&g.data).map(|(a, b)| a * b).sum()
        };

        // Grad w.r.t. the embedded input.
        let grad = &d_emb.data;
        let norm: f32 = grad.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm > 1e-6);
        let u: Vec<f32> = grad.iter().map(|v| v / norm).collect();
        let eps = 2e-4;
        let mut xp = embedded.clone();
        for (v, &ui) in xp.data.iter_mut().zip(&u) {
            *v += eps * ui;
        }
        let plus = loss(&mut enc, &xp);
        for (v, &ui) in xp.data.iter_mut().zip(&u) {
            *v -= 2.0 * eps * ui;
        }
        let minus = loss(&mut enc, &xp);
        let fd = (plus - minus) / (2.0 * eps);
        assert!(
            (fd - norm).abs() <= 0.3 * norm + 1e-3,
            "d_embedded: ‖G‖ {norm} vs fd {fd}"
        );
    }
}
