//! A small fully-GPU-resident training stack, composing every ported kernel:
//!
//!   Embedding → Linear → RMSNorm → Linear(head) → SoftCap → softmax/CE
//!
//! This is *not* the real language model yet — it has no recurrent core (the
//! sLSTM/mLSTM block is the next port). Its purpose is to prove the ported ops
//! and the `Linear` layer **compose** into a correct forward → loss → backward →
//! optimizer-step cycle end to end on the device: buffer threading, transpose
//! forms, gradient-accumulation order and the multi-layer optimizer convention
//! all exercised at once. The parity test checks loss and every updated
//! parameter against the identical stack built from `nn2` CPU layers.
//!
//! Optimizer convention (same as the rest of the project): the interior `Linear`
//! decays its weight; the embedding table, the RMSNorm scale and the logit head
//! train on undecayed Adam.

use super::ops::{self, GpuRmsForward};
use super::{DTensor, Gpu, linear::Linear};
use crate::nn2::optim::AdamCfg;
use crate::tensor::Tensor;

/// RMSNorm epsilon — matches `nn2::rms_norm::EPS` so the two systems normalize
/// identically.
const EPS: f32 = 1e-6;

pub struct Flat {
    // Embedding table + grad + Adam moments.
    table: DTensor,
    dtable: DTensor,
    m_tbl: DTensor,
    v_tbl: DTensor,
    dim: usize,

    lin1: Linear,

    // RMSNorm scale + grad + moments.
    gamma: DTensor,
    dgamma: DTensor,
    m_g: DTensor,
    v_g: DTensor,
    hidden: usize,

    head: Linear,
    cap: f32,

    // Per-forward caches for backward.
    ids: Vec<usize>,
    rms: Option<GpuRmsForward>,
    logits: Option<DTensor>, // capped logits, for SoftCap backward
}

impl Flat {
    /// Build from explicit host weights so a CPU reference stack can start from
    /// the exact same parameters. `table` is `[vocab, dim]`, `lin1_*` map
    /// `dim → hidden`, `gamma` is `[hidden]`, `head_*` map `hidden → vocab`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        gpu: &Gpu,
        table: &Tensor,
        lin1_w: &Tensor,
        lin1_b: &Tensor,
        gamma: &Tensor,
        head_w: &Tensor,
        head_b: &Tensor,
        cap: f32,
    ) -> Self {
        let (vocab, dim) = (table.rows(), table.cols());
        let hidden = gamma.len();
        Self {
            table: DTensor::from_host(gpu, table),
            dtable: DTensor::zeros(gpu, &[vocab, dim]),
            m_tbl: DTensor::zeros(gpu, &[vocab, dim]),
            v_tbl: DTensor::zeros(gpu, &[vocab, dim]),
            dim,
            lin1: Linear::from_parts(gpu, lin1_w, lin1_b),
            gamma: DTensor::from_host(gpu, gamma),
            dgamma: DTensor::zeros(gpu, &[hidden]),
            m_g: DTensor::zeros(gpu, &[hidden]),
            v_g: DTensor::zeros(gpu, &[hidden]),
            hidden,
            head: Linear::from_parts(gpu, head_w, head_b),
            cap,
            ids: Vec::new(),
            rms: None,
            logits: None,
        }
    }

    /// Forward to capped logits `[B, vocab]`. Caches everything backward needs.
    pub fn forward(&mut self, gpu: &Gpu, ids: &[usize]) -> DTensor {
        let e = ops::embedding_gather(gpu, &self.table, ids, self.dim);
        let h = self.lin1.forward(gpu, &e);
        let (rms_out, rms) = ops::rms_norm_forward(gpu, &h, &self.gamma, self.hidden, EPS);
        let pre = self.head.forward(gpu, &rms_out);
        let logits = ops::softcap_forward(gpu, &pre, self.cap);
        self.ids = ids.to_vec();
        self.rms = Some(rms);
        self.logits = Some(logits.dup(gpu));
        logits
    }

    /// Given `dlogits` from the fused CE, backprop through the whole stack,
    /// accumulating every parameter gradient.
    pub fn backward(&mut self, gpu: &Gpu, dlogits: &DTensor) {
        let logits = self.logits.as_ref().expect("forward before backward");
        let rms = self.rms.as_ref().expect("forward before backward");
        let d_pre = ops::softcap_backward(gpu, dlogits, logits, self.cap);
        let d_rmsout = self.head.backward(gpu, &d_pre);
        let d_h = ops::rms_norm_backward(
            gpu,
            &d_rmsout,
            rms,
            &self.gamma,
            &mut self.dgamma,
            self.hidden,
        );
        let d_e = self.lin1.backward(gpu, &d_h);
        ops::embedding_scatter_add(gpu, &mut self.dtable, &self.ids, &d_e, self.dim);
    }

    /// AdamW step for every parameter (embedding/scale/head undecayed, interior
    /// Linear decayed), then clear the accumulators.
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
        self.lin1.step(gpu, cfg);
        ops::adamw(
            gpu,
            &mut self.gamma,
            &self.dgamma,
            &mut self.m_g,
            &mut self.v_g,
            cfg,
            false,
        );
        self.dgamma.zero_(gpu);
        self.head.step_wd(gpu, cfg, false);
    }

    /// Convenience: one full train step, returning the mean CE loss.
    pub fn train_step(
        &mut self,
        gpu: &Gpu,
        ids: &[usize],
        targets: &[usize],
        cfg: &AdamCfg,
    ) -> f32 {
        let logits = self.forward(gpu, ids);
        let (loss, dlogits) = ops::softmax_cross_entropy(gpu, &logits, targets);
        self.backward(gpu, &dlogits);
        self.step(gpu, cfg);
        loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn2::embedding::Embedding as CpuEmb;
    use crate::nn2::linear::Linear as CpuLinear;
    use crate::nn2::loss;
    use crate::nn2::rms_norm::RmsNorm as CpuRms;
    use crate::nn2::soft_cap::SoftCap as CpuSoftCap;

    fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length mismatch");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < tol, "{what}[{i}]: gpu {g} vs cpu {w}");
        }
    }

    /// The whole GPU stack must match an identical CPU `nn2` stack for
    /// loss → backward → one AdamW step, from the same initial weights.
    #[test]
    fn flat_stack_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let (vocab, dim, hidden, _batch) = (17, 8, 12, 6);
        let cap = 30.0;

        // Shared initial parameters.
        let table = Tensor::random(&[vocab, dim], 0.1);
        let lin1_w = Tensor::xavier(dim, hidden);
        let lin1_b = Tensor::random(&[hidden], 0.1);
        let gamma = Tensor::random(&[hidden], 1.0);
        let head_w = Tensor::xavier(hidden, vocab);
        let head_b = Tensor::random(&[vocab], 0.1);

        let ids: Vec<usize> = vec![3, 1, 4, 1, 5, 9]
            .into_iter()
            .map(|v| v % vocab)
            .collect();
        let targets: Vec<usize> = vec![2, 7, 1, 8, 2, 8]
            .into_iter()
            .map(|v| v % vocab)
            .collect();

        // --- CPU reference stack (nn2 layers), same init ---
        let mut c_emb = CpuEmb::new(vocab, dim);
        c_emb.table = table.clone();
        let mut c_lin1 = CpuLinear::from_parts(lin1_w.clone(), lin1_b.clone());
        let mut c_rms = CpuRms::new(hidden);
        c_rms.gamma = gamma.clone();
        let mut c_head = CpuLinear::from_parts(head_w.clone(), head_b.clone());
        let mut c_sc = CpuSoftCap::new(cap);

        let e = c_emb.forward(&ids);
        let h = c_lin1.forward(&e);
        let r = c_rms.forward(&h);
        let pre = c_head.forward(&r);
        let clogits = c_sc.forward(&pre);
        let (cpu_loss, dlog) = loss::softmax_cross_entropy(&clogits, &targets);
        let d_pre = c_sc.backward(&dlog);
        let d_r = c_head.backward(&d_pre);
        let d_h = c_rms.backward(&d_r);
        let d_e = c_lin1.backward(&d_h);
        c_emb.backward(&d_e);

        let mut cfg = AdamCfg::new(1e-3, 0.01);
        cfg.t = 1;
        c_emb.step(&cfg);
        c_lin1.step(&cfg);
        c_rms.step(&cfg);
        c_head.step_wd(&cfg, false);

        // --- GPU stack, same init, one train step ---
        let mut dev = Flat::from_parts(
            &gpu, &table, &lin1_w, &lin1_b, &gamma, &head_w, &head_b, cap,
        );
        let gpu_loss = dev.train_step(&gpu, &ids, &targets, &cfg);

        assert!(
            (cpu_loss - gpu_loss).abs() < 1e-4,
            "loss: cpu {cpu_loss} vs gpu {gpu_loss}"
        );
        // Updated parameters must match after the step.
        assert_close(
            &dev.table.to_host(&gpu).data,
            &c_emb.table.data,
            1e-5,
            "table",
        );
        assert_close(
            &dev.lin1.w.to_host(&gpu).data,
            &c_lin1.w.data,
            1e-5,
            "lin1.w",
        );
        assert_close(
            &dev.gamma.to_host(&gpu).data,
            &c_rms.gamma.data,
            1e-5,
            "gamma",
        );
        assert_close(
            &dev.head.w.to_host(&gpu).data,
            &c_head.w.data,
            1e-5,
            "head.w",
        );
    }
}
