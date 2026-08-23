//! The real GPU language model (Phase D) — fully device-resident:
//!
//!   Embedding → Block×N (alternating sLSTM / mLSTM) → RMSNorm → Linear head
//!   → SoftCap → fused softmax/CE
//!
//! This is the stack the flat toy (`gpu::flat`) stood in for: it has an actual
//! recurrent core. Everything — the embedding table, every block (norms, SwiGLU
//! projections, recurrent cell), the final norm and the logit head, plus all
//! gradients and AdamW moments — lives in `GTensor<f32>`s, so a whole
//! forward → loss → backward → step cycle crosses PCIe only for the token ids in
//! and the scalar loss out.
//!
//! Optimizer convention (same as the rest of the project): the embedding table,
//! the norm scales and the logit head train on **undecayed** Adam; the interior
//! projections inside the blocks decay (handled by `gpu::Linear`).
//!
//! The blocks are held as `Vec<Box<dyn BlockLike>>` because `Block<SLstm>` and
//! `Block<MLstm>` are different concrete types.

use super::{GTensor, Gpu, linear::Linear, ops, rms_norm::RmsNorm};
use crate::gpu::arena::TrainingCache;
use crate::gpu::block::BlockLike;
use crate::nn2::optim::AdamCfg;
use crate::tensor::Tensor;

pub struct Lm {
    // Embedding table + grad + Adam moments.
    table: GTensor<f32>,
    dtable: GTensor<f32>,
    m_tbl: GTensor<f32>,
    v_tbl: GTensor<f32>,

    blocks: Vec<Box<dyn BlockLike>>,
    norm: RmsNorm,
    head: Linear,
    cap: f32,

    hidden: usize,
    // Per-forward cache for backward.
    ids: Vec<usize>,
    logits: Option<GTensor<f32>>, // capped logits, for the SoftCap backward
    seq: (usize, usize),     // (B, T)
    /// Forward activations, owned so the buffers survive across calls.
    cache: TrainingCache,
}

impl Lm {
    /// Build from explicit host parameters so an identical CPU reference stack can
    /// start from the same weights. `table` is `[vocab, hidden]`, `gamma` the final
    /// norm scale `[hidden]`, `head_*` maps `hidden → vocab`. `blocks` are uploaded
    /// by the caller (e.g. `Block::<SLstm>::from_cpu`).
    pub fn from_parts(
        gpu: &Gpu,
        table: &Tensor,
        blocks: Vec<Box<dyn BlockLike>>,
        gamma: &Tensor,
        head_w: &Tensor,
        head_b: &Tensor,
        cap: f32,
    ) -> Self {
        let (vocab, hidden) = (table.rows(), table.cols());
        Self {
            table: GTensor::from_host(gpu, table),
            dtable: GTensor::zeros(gpu, &[vocab, hidden]),
            m_tbl: GTensor::zeros(gpu, &[vocab, hidden]),
            v_tbl: GTensor::zeros(gpu, &[vocab, hidden]),
            blocks,
            norm: RmsNorm::from_parts(gpu, gamma),
            head: {
                // The logit head stays fp32. Its output goes straight into the
                // softmax/cross-entropy, where a bf16-sized perturbation of a logit
                // is exponentiated — and the vocab-sized reduction that follows is
                // exactly the place the reference keeps wide. It is also one GEMM
                // per step against the backbone's many, so there is little to win.
                let mut h = Linear::from_parts(gpu, head_w, head_b);
                h.set_fp32();
                h
            },
            cap,
            hidden,
            ids: Vec::new(),
            logits: None,
            seq: (0, 0),
            cache: TrainingCache::default(),
        }
    }

    /// Forward `B·T` token ids (row-major `[B, T]`) to capped logits `[B·T, vocab]`.
    pub fn forward(&mut self, gpu: &Gpu, ids: &[usize], b: usize, t: usize) -> GTensor<f32> {
        let (n, h) = (b * t, self.hidden);
        assert_eq!(ids.len(), n, "Lm::forward — ids len != B·T");

        let e = ops::embedding_gather(gpu, &self.table, ids, h); // [N, H]
        let mut seq = e.reshaped(&[b, t, h]);
        // Blocks are H-in == H-out, so one spare buffer ping-pongs the whole stack.
        let mut next = GTensor::uninit(gpu, &[b, t, h]);
        for blk in self.blocks.iter_mut() {
            blk.forward(gpu, &seq, &mut next, &mut self.cache);
            std::mem::swap(&mut seq, &mut next);
        }
        let flat = seq.reshaped(&[n, h]);
        let normed = self.norm.forward_alloc(gpu, &flat);
        let pre = self.head.forward_alloc(gpu, &normed);
        let logits = ops::softcap_forward(gpu, &pre, self.cap);

        self.ids = ids.to_vec();
        self.seq = (b, t);
        self.logits = Some(logits.dup(gpu));
        logits
    }

    /// Backprop `dlogits` (from the fused CE) through the whole stack.
    pub fn backward(&mut self, gpu: &Gpu, dlogits: &GTensor<f32>) {
        let (b, t) = self.seq;
        let (n, h) = (b * t, self.hidden);
        let logits = self.logits.as_ref().expect("Lm::backward before forward");

        let d_pre = ops::softcap_backward(gpu, dlogits, logits, self.cap);
        let d_normed = self.head.backward_alloc(gpu, &d_pre);
        let d_flat = self.norm.backward_alloc(gpu, &d_normed); // [N, H]

        let mut d_seq = d_flat.reshaped(&[b, t, h]);
        let mut d_next = GTensor::uninit(gpu, &[b, t, h]);
        for blk in self.blocks.iter_mut().rev() {
            blk.backward(gpu, &d_seq, &mut d_next);
            std::mem::swap(&mut d_seq, &mut d_next);
        }
        let d_e = d_seq.reshaped(&[n, h]);
        ops::embedding_scatter_add(gpu, &mut self.dtable, &self.ids, &d_e, h);
    }

    /// AdamW step for every parameter (table / norm / head undecayed; the blocks'
    /// interior projections decay), then clear the accumulators.
    pub fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        ops::adamw(
            gpu,
            &mut self.table,
            &self.dtable,
            &mut self.m_tbl,
            &mut self.v_tbl,
            cfg,
            false,
        );
        self.dtable.zero_(gpu);
        for blk in self.blocks.iter_mut() {
            blk.step(gpu, cfg);
        }
        self.norm.step(gpu, cfg);
        self.head.step(gpu, cfg);
    }

    /// One full training step; returns the mean CE loss.
    pub fn train_step(
        &mut self,
        gpu: &Gpu,
        ids: &[usize],
        targets: &[usize],
        b: usize,
        t: usize,
        cfg: &AdamCfg,
    ) -> f32 {
        let logits = self.forward(gpu, ids, b, t);
        let (loss, dlogits) = ops::softmax_cross_entropy(gpu, &logits, targets);
        self.backward(gpu, &dlogits);
        self.step(gpu, cfg);
        loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::block::Block;
    use crate::gpu::{mlstm::MLstm, slstm::SLstm};
    use crate::nn2::loss;
    use crate::nn2::{
        Embedding, Linear as CpuLinear, MLstmBlock, RmsNorm as CpuRms, SLstmBlock, SoftCap,
    };

    fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length mismatch");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < tol, "{what}[{i}]: gpu {g} vs cpu {w}");
        }
    }

    /// The whole GPU LM — embedding, an sLSTM block **and** an mLSTM block, final
    /// norm, capped head, fused CE — must match the identical CPU `nn2` stack for
    /// loss → backward → one AdamW step. This is the Phase-D end-to-end check: it
    /// exercises every ported kernel composed in the real architecture.
    #[test]
    fn lm_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let (vocab, hidden, up, heads, dqk) = (13, 8, 12, 2, 4);
        let (b, t) = (2, 5);
        let n = b * t;
        let cap = 30.0;

        // shared initial parameters
        let table = Tensor::random(&[vocab, hidden], 0.1);
        let gamma = Tensor::random(&[hidden], 1.0);
        let head_w = Tensor::xavier(hidden, vocab);
        let head_b = Tensor::random(&[vocab], 0.1);

        let mut c_emb = Embedding::new(vocab, hidden);
        c_emb.table = table.clone();
        let mut c_blk1 = SLstmBlock::new_slstm(hidden, up);
        let mut c_blk2 = MLstmBlock::new_mlstm(hidden, up, heads, dqk);
        // Non-trivial mLSTM gate weights (nn2::MLstm::new zero-inits wi/wf).
        c_blk2.cell.wi = Tensor::random(&[hidden, heads], 0.3);
        c_blk2.cell.wf = Tensor::random(&[hidden, heads], 0.3);
        let mut c_norm = CpuRms::new(hidden);
        c_norm.gamma = gamma.clone();
        let mut c_head = CpuLinear::from_parts(head_w.clone(), head_b.clone());
        let mut c_cap = SoftCap::new(cap);

        let ids: Vec<usize> = (0..n).map(|i| (i * 5 + 1) % vocab).collect();
        let targets: Vec<usize> = (0..n).map(|i| (i * 3 + 2) % vocab).collect();

        // Upload the GPU model NOW, from the pristine CPU weights — the CPU step
        // below mutates them in place.
        let blocks: Vec<Box<dyn BlockLike>> = vec![
            Box::new(Block::<SLstm>::from_cpu(&gpu, &c_blk1)),
            Box::new(Block::<MLstm>::from_cpu(&gpu, &c_blk2)),
        ];
        let mut dev = Lm::from_parts(&gpu, &table, blocks, &gamma, &head_w, &head_b, cap);

        // CPU reference: forward → loss → backward → step
        let e = c_emb.forward(&ids); // [N, H]
        let h1 = c_blk1.forward(&e.reshape(&[b, t, hidden]));
        let h2 = c_blk2.forward(&h1);
        let nrm = c_norm.forward(&h2.reshape(&[n, hidden]));
        let pre = c_head.forward(&nrm);
        let clogits = c_cap.forward(&pre);
        let (cpu_loss, dlog) = loss::softmax_cross_entropy(&clogits, &targets);

        let d_pre = c_cap.backward(&dlog);
        let d_nrm = c_head.backward(&d_pre);
        let d_flat = c_norm.backward(&d_nrm);
        let d_h1 = c_blk2.backward(&d_flat.reshape(&[b, t, hidden]));
        let d_e = c_blk1.backward(&d_h1);
        c_emb.backward(&d_e.reshape(&[n, hidden]));

        let mut cfg = AdamCfg::new(1e-3, 0.01);
        cfg.t = 1;
        c_emb.step(&cfg);
        c_blk1.step(&cfg);
        c_blk2.step(&cfg);
        c_norm.step(&cfg);
        c_head.step_wd(&cfg, false);

        // GPU: same init, one train step
        let gpu_loss = dev.train_step(&gpu, &ids, &targets, b, t, &cfg);

        // The interior projections run bf16 operands / fp32 accumulate when enabled
        // (the head itself is pinned fp32 — see `Lm::from_parts`), so both the loss
        // and the stepped parameters land within bf16's precision of the CPU's.
        //
        // The parameter bound is the looser of the two because Adam's update is
        // scale-invariant in the gradient: `lr·ĝ/(√v̂+ε)` divides a near-zero
        // gradient by an equally small √v̂, turning a ~1e-7 difference into ~1e-4 on
        // the weight. Same amplification the arch flag hits in `kernels.rs`.
        let bf = ops::gemm_bf16_enabled(&gpu);
        let ltol = if bf { 5e-3 } else { 1e-3 };
        let ptol = if bf { 2e-2 } else { 3e-3 };
        assert!(
            (cpu_loss - gpu_loss).abs() < ltol,
            "loss: cpu {cpu_loss} vs gpu {gpu_loss} (tolerance {ltol:.0e})"
        );
        assert_close(
            &dev.table.to_host(&gpu).data,
            &c_emb.table.data,
            ptol,
            "table",
        );
        assert_close(
            &dev.norm.gamma.to_host(&gpu).data,
            &c_norm.gamma.data,
            ptol,
            "gamma",
        );
        assert_close(
            &dev.head.w.to_host(&gpu).data,
            &c_head.w.data,
            ptol,
            "head.w",
        );
    }
}
