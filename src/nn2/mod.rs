//! Clean-room batched NN layers built on `crate::tensor`.
//!
//! Parallel, independent reimplementation of the modelling stack — kept next to
//! the old `crate::nn` (not replacing it) so the two can be benchmarked against
//! each other. Everything here operates on `[B, F]` / `[B, T, F]` tensors: the
//! leading batch axis turns the old per-timestep matrix-vector products into
//! matrix-matrix products.
//!
//! No autograd. Each layer keeps a forward cache and hand-writes its backward,
//! same as the old system, just batched.

pub mod block;
pub mod embedding;
pub mod headwise_rms_norm;
pub mod hierarchical;
pub mod linear;
pub mod loss;
pub mod mlstm;
pub mod ops;
pub mod optim;
pub mod rms_norm;
pub mod slstm;
pub mod soft_cap;
pub mod word_encoder;

pub use block::{Block, MLstmBlock, SLstmBlock};
pub use embedding::Embedding;
pub use headwise_rms_norm::HeadwiseRmsNorm;
pub use linear::Linear;
pub use mlstm::MLstm;
pub use optim::{AdamCfg, AdamState};
pub use rms_norm::RmsNorm;
pub use slstm::SLstm;
pub use soft_cap::SoftCap;

#[cfg(test)]
mod train_smoke {
    use super::Linear;
    use super::loss::softmax_cross_entropy;
    use crate::tensor::Tensor;

    /// End-to-end batched training smoke test: a single Linear + softmax must
    /// learn a linearly separable 3-class mapping, driving the loss down.
    /// Exercises forward, fused CE backward, grad accumulation and the SGD step
    /// all together on `[B, F]` tensors.
    #[test]
    fn linear_softmax_learns_toy_task() {
        let (batch, features, classes) = (30, 2, 3);
        let mut lin = Linear::new(features, classes);

        // Fixed batch: class from the value of a+b — a cheap, deterministic,
        // linearly separable 3-band target.
        let mut xs = Vec::with_capacity(batch * features);
        let mut ys = Vec::with_capacity(batch);
        for i in 0..batch {
            let a = (i as f32 / batch as f32) * 6.0 - 3.0;
            let b = ((i * 7) % batch) as f32 / batch as f32 * 6.0 - 3.0;
            xs.push(a);
            xs.push(b);
            // Three bands by the value of a+b.
            let s = a + b;
            ys.push(if s < -1.0 {
                0
            } else if s < 1.0 {
                1
            } else {
                2
            });
        }
        let x = Tensor::new(&[batch, features], xs);

        let first = softmax_cross_entropy(&lin.forward(&x), &ys).0;
        for _ in 0..400 {
            let logits = lin.forward(&x);
            let (_loss, dlogits) = softmax_cross_entropy(&logits, &ys);
            lin.backward(&dlogits);
            lin.sgd_step(0.5);
        }
        let last = softmax_cross_entropy(&lin.forward(&x), &ys).0;

        assert!(
            last < first * 0.5,
            "loss did not fall enough: {first} -> {last}"
        );
        assert!(last < 0.7, "final loss too high: {last}");
    }
}

#[cfg(test)]
mod train_e2e {
    use super::loss::softmax_cross_entropy;
    use super::{AdamCfg, Embedding, Linear, SLstmBlock};
    use crate::tensor::Tensor;

    /// End-to-end: a real flat model — Embedding → sLSTM residual block →
    /// Linear head — trained with AdamW on a next-token task, must drive the
    /// loss down. Exercises the whole nn2 stack (embedding gather, the block's
    /// two residuals + recurrent cell + SwiGLU, fused CE, AdamW step) wired
    /// together, with gradients flowing through every layer.
    #[test]
    fn flat_model_trains_with_adamw() {
        let (vocab, hidden, up) = (12usize, 32usize, 48usize);
        let (b, t) = (8usize, 16usize);
        let n = b * t;

        let mut emb = Embedding::new(vocab, hidden);
        let mut block = SLstmBlock::new_slstm(hidden, up);
        let mut head = Linear::new(hidden, vocab);

        // Learnable next-token task: token[i+1] = (token[i] + 1) % vocab.
        // Each of the B sequences starts at a different value.
        let mut ids = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);
        for seq in 0..b {
            let mut cur = seq % vocab;
            for _ in 0..t {
                ids.push(cur);
                let next = (cur + 1) % vocab;
                targets.push(next);
                cur = next;
            }
        }

        let forward = |emb: &mut Embedding, block: &mut SLstmBlock, head: &mut Linear| -> Tensor {
            let e = emb.forward(&ids); // [N, H]
            let h = block.forward(&e.reshape(&[b, t, hidden])); // [B, T, H]
            head.forward(&h.reshape(&[n, hidden])) // [N, vocab]
        };

        let mut cfg = AdamCfg::new(5e-3, 0.0);
        let first = softmax_cross_entropy(&forward(&mut emb, &mut block, &mut head), &targets).0;

        for _ in 0..300 {
            let logits = forward(&mut emb, &mut block, &mut head);
            let (_loss, dlogits) = softmax_cross_entropy(&logits, &targets);
            let dh = head.backward(&dlogits); // [N, H]
            let de = block.backward(&dh.reshape(&[b, t, hidden])); // [B, T, H]
            emb.backward(&de.reshape(&[n, hidden]));

            cfg.t += 1;
            emb.step(&cfg);
            block.step(&cfg);
            head.step(&cfg);
        }

        let last = softmax_cross_entropy(&forward(&mut emb, &mut block, &mut head), &targets).0;
        assert!(
            last < first * 0.3,
            "loss did not fall enough: {first} -> {last}"
        );
        assert!(last < 0.3, "final loss too high: {last}");
    }
}
