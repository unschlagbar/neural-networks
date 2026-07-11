//! Batched embedding table: token id -> row of a `[vocab, dim]` matrix.
//!
//! Forward gathers one row per id into a `[B, dim]` tensor; backward
//! scatter-adds the upstream grad rows back into the table gradient. There is
//! no input gradient (the input is a discrete id).
//!
//! The tied encoder/decoder char table in the hierarchical model is handled at
//! the model level by sharing one `Embedding` (or by reducing the decoder-side
//! table grad into the encoder's), exactly as the old system does.

use crate::nn2::optim::{AdamCfg, AdamState};
use crate::nn2::ops;
use crate::tensor::Tensor;

pub struct Embedding {
    pub table: Tensor,  // [vocab, dim]
    pub dtable: Tensor, // [vocab, dim]

    /// Token ids from the last forward, for the scatter-add backward.
    ids: Vec<usize>,
    dim: usize,
    opt: AdamState,
}

impl Embedding {
    pub fn new(vocab: usize, dim: usize) -> Self {
        // Small uniform init, like a typical embedding table.
        let scale = (1.0 / dim as f32).sqrt();
        Self {
            table: Tensor::random(&[vocab, dim], scale),
            dtable: Tensor::zeros(&[vocab, dim]),
            ids: Vec::new(),
            dim,
            opt: AdamState::new(),
        }
    }

    /// AdamW step (embedding table is never decayed). Clears the grad.
    pub fn step(&mut self, cfg: &AdamCfg) {
        self.opt.step(&mut self.table.data, &self.dtable.data, cfg, false);
        self.zero_grad();
    }

    #[inline]
    pub fn vocab(&self) -> usize {
        self.table.rows()
    }
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Gather: `out[b] = table[ids[b]]`. Result is `[B, dim]`.
    pub fn forward(&mut self, ids: &[usize]) -> Tensor {
        let out = ops::embedding_gather(&self.table, ids, self.dim);
        self.ids = ids.to_vec();
        out
    }

    /// Scatter-add: `dtable[ids[b]] += dy[b]`. No input gradient.
    pub fn backward(&mut self, dy: &Tensor) {
        let d = self.dim;
        assert_eq!(dy.rows(), self.ids.len(), "Embedding::backward — batch mismatch");
        assert_eq!(dy.cols(), d, "Embedding::backward — dim mismatch");
        ops::embedding_scatter_add(&mut self.dtable, &self.ids, dy, d);
    }

    pub fn zero_grad(&mut self) {
        self.dtable.zero_();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_and_scatter_add() {
        let (vocab, dim) = (5, 3);
        let mut emb = Embedding::new(vocab, dim);
        let ids = [1usize, 3, 1]; // id 1 appears twice -> its grad should sum

        let out = emb.forward(&ids);
        // Rows must equal the gathered table rows.
        for (r, &id) in ids.iter().enumerate() {
            assert_eq!(&out.data[r * dim..(r + 1) * dim], &emb.table.data[id * dim..(id + 1) * dim]);
        }

        // Upstream grad of all-ones: id 1 gets +2 per element, id 3 gets +1.
        let dy = Tensor::new(&[3, dim], vec![1.0; 3 * dim]);
        emb.backward(&dy);
        for k in 0..dim {
            assert_eq!(emb.dtable.data[1 * dim + k], 2.0);
            assert_eq!(emb.dtable.data[3 * dim + k], 1.0);
            assert_eq!(emb.dtable.data[0 * dim + k], 0.0);
        }
    }

    /// FD check: gather grad through a scalar loss L = Σ (out ⊙ G).
    #[test]
    fn grad_matches_finite_difference() {
        let (vocab, dim) = (6, 4);
        let mut emb = Embedding::new(vocab, dim);
        let ids = [2usize, 5, 2, 0];
        let g = Tensor::random(&[ids.len(), dim], 1.0);

        let out = emb.forward(&ids);
        let _ = out;
        emb.backward(&g);
        let dtable = emb.dtable.clone();

        let loss = |emb: &mut Embedding| -> f32 {
            let o = emb.forward(&ids);
            o.data.iter().zip(&g.data).map(|(a, b)| a * b).sum()
        };
        let eps = 1e-3;
        for i in 0..emb.table.data.len() {
            let orig = emb.table.data[i];
            emb.table.data[i] = orig + eps;
            let lp = loss(&mut emb);
            emb.table.data[i] = orig - eps;
            let lm = loss(&mut emb);
            emb.table.data[i] = orig;
            let num = (lp - lm) / (2.0 * eps);
            assert!((num - dtable.data[i]).abs() < 1e-3, "dtable[{i}] {num} vs {}", dtable.data[i]);
        }
    }
}
