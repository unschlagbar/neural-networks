//! Device-resident multi-head mLSTM cell, GPU counterpart of
//! [`nn2::mlstm::MLstm`](crate::nn2::mlstm::MLstm) — in the **parallel /
//! chunkwise** formulation, not the scalar per-head recurrence (the CPU's sub-1×
//! path; see PLAN-gpu.md Phase C).
//!
//! The equivalence (derived in the plan): the CPU stores the *stabilized* state
//! `C_t = C_t^true·exp(−m_t)`, and the running stabilizer unrolls to a row-max
//! over the log-decay matrix `logD_{tj} = fc_t − fc_j + ĩ_j` (`fc` = cumulative
//! log-forget). For the whole sequence as a single chunk (`C_prev=n_prev=0`,
//! `m_prev=0`):
//! ```text
//!   S = Q·Kᵀ ;  m_t = max(max_{j≤t} logD_{tj}, fc_t)
//!   D̄_{tj} = exp(logD_{tj} − m_t)  (j≤t else 0)
//!   ỹ_t = ((D̄⊙S)·V)_t / ψ_t ,  ψ_t = max(|Σ_j (D̄⊙S)_{tj}|, 1)
//! ```
//! then head-norm(ỹ) → ŷ, `y = o⊙ŷ`, `h = y·W_out + b_out`. Backward
//! differentiates this graph with `m` held constant (the reference stabilizer
//! approximation, same as the CPU / the sLSTM cell).
//!
//! The six projections and `W_out` are `gpu::Linear`; only the attention core is
//! bespoke kernels + strided-batched GEMM, on the head-major `[B*H, T, ·]` layout.
//!
//! **Single-chunk only so far (O(T²)); forward+backward parity-tested vs the CPU
//! scalar cell. Chunking (O(T)) is the next Phase-C step.**

use super::block::Cell;
use super::{DTensor, Gpu, linear::Linear, ops, rms_norm::RmsNorm};
use crate::nn2::optim::AdamCfg;
use crate::tensor::Tensor;

/// Forward intermediates retained for the backward pass.
struct Saved {
    b: usize,
    t: usize,
    qh: DTensor,   // [BH, T, dqk]
    kh: DTensor,   // [BH, T, dqk]  (already ×1/√dqk)
    vh: DTensor,   // [BH, T, dhv]
    fgh: DTensor,  // [BH, T]  forget-gate logit (head-major)
    dbar: DTensor, // [BH, T, T]
    ds: DTensor,   // [BH, T, T]
    num: DTensor,  // [BH, T, dhv]
    qn: DTensor,   // [BH, T]
    psi: DTensor,  // [BH, T]
    o: DTensor,    // [N, d]  (post-sigmoid)
    yhat: DTensor, // [N, d]
}

pub struct MLstm {
    pub input_size: usize,
    pub d: usize,
    pub heads: usize,
    pub dqk: usize,
    pub dhv: usize,
    inv_sqrt_dqk: f32,

    // Projections (in → ·) and the output projection (d → d). Bias, weight decay
    // and AdamW all handled by `Linear`, matching the CPU cell's conventions.
    lin_q: Linear,
    lin_k: Linear,
    lin_v: Linear,
    lin_o: Linear,
    lin_i: Linear,
    lin_f: Linear,
    lin_out: Linear,
    headnorm: RmsNorm, // head-wise (group == dhv)

    saved: Option<Saved>,
}

impl MLstm {
    /// Build from a CPU cell's host weights (all 15 parameter tensors uploaded).
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        gpu: &Gpu,
        input_size: usize, d: usize, heads: usize, dqk: usize,
        wq: &Tensor, wk: &Tensor, wv: &Tensor, wo: &Tensor, wi: &Tensor, wf: &Tensor,
        bq: &Tensor, bk: &Tensor, bv: &Tensor, bo: &Tensor, bi: &Tensor, bf: &Tensor,
        w_out: &Tensor, b_out: &Tensor, gamma: &Tensor,
    ) -> Self {
        let dhv = d / heads;
        Self {
            input_size, d, heads, dqk, dhv,
            inv_sqrt_dqk: 1.0 / (dqk as f32).sqrt(),
            lin_q: Linear::from_parts(gpu, wq, bq),
            lin_k: Linear::from_parts(gpu, wk, bk),
            lin_v: Linear::from_parts(gpu, wv, bv),
            lin_o: Linear::from_parts(gpu, wo, bo),
            lin_i: Linear::from_parts(gpu, wi, bi),
            lin_f: Linear::from_parts(gpu, wf, bf),
            lin_out: Linear::from_parts(gpu, w_out, b_out),
            headnorm: RmsNorm::from_parts_grouped(gpu, gamma, dhv),
            saved: None,
        }
    }

    /// Freshly-initialised cell, matching `nn2::MLstm::new`'s init exactly.
    pub fn new_rand(gpu: &Gpu, input_size: usize, d: usize, heads: usize, dqk: usize) -> Self {
        Self::from_cpu(gpu, &crate::nn2::MLstm::new(input_size, d, heads, dqk))
    }

    /// Export this cell into the CPU `nn::MLSTMLayer` format. Used to write a
    /// `HIER` checkpoint from a GPU model.
    pub fn to_nn_cell(&self, gpu: &Gpu) -> crate::nn::mlstm::MLSTMLayer {
        use super::{dt_matrix, dt_vec};
        let w_out = crate::nn::linear::LinearLayer::from_loaded(
            self.d,
            self.d,
            dt_matrix(gpu, &self.lin_out.w),
            dt_vec(gpu, &self.lin_out.b),
        );
        crate::nn::mlstm::MLSTMLayer::from_loaded(
            self.input_size,
            self.d,
            self.heads,
            self.dqk,
            dt_matrix(gpu, &self.lin_q.w),
            dt_matrix(gpu, &self.lin_k.w),
            dt_matrix(gpu, &self.lin_v.w),
            dt_matrix(gpu, &self.lin_o.w),
            dt_matrix(gpu, &self.lin_i.w),
            dt_matrix(gpu, &self.lin_f.w),
            dt_vec(gpu, &self.lin_q.b),
            dt_vec(gpu, &self.lin_k.b),
            dt_vec(gpu, &self.lin_v.b),
            dt_vec(gpu, &self.lin_o.b),
            dt_vec(gpu, &self.lin_i.b),
            dt_vec(gpu, &self.lin_f.b),
            w_out,
            dt_vec(gpu, &self.headnorm.gamma),
        )
    }

    /// Rebuild a GPU cell from a CPU `nn::MLSTMLayer` (inverse of `to_nn_cell`).
    pub fn from_nn_cell(gpu: &Gpu, c: &crate::nn::mlstm::MLSTMLayer) -> Self {
        use super::{tensor_from_matrix as m, tensor_from_slice as v};
        Self::from_parts(
            gpu,
            c.input_size,
            c.hidden_size,
            c.num_heads,
            c.dqk,
            &m(&c.wq), &m(&c.wk), &m(&c.wv), &m(&c.wo), &m(&c.wi), &m(&c.wf),
            &v(&c.bq), &v(&c.bk), &v(&c.bv), &v(&c.bo), &v(&c.bi), &v(&c.bf),
            &m(&c.w_out.weights),
            &v(&c.w_out.biases),
            &v(&c.head_norm.gamma),
        )
    }

    /// Upload a CPU cell (weights are copied; grads/moments start at zero).
    pub fn from_cpu(gpu: &Gpu, c: &crate::nn2::MLstm) -> Self {
        Self::from_parts(
            gpu, c.input_size, c.d, c.heads, c.dqk,
            &c.wq, &c.wk, &c.wv, &c.wo, &c.wi, &c.wf,
            &c.bq, &c.bk, &c.bv, &c.bo, &c.bi, &c.bf,
            &c.w_out, &c.b_out, &c.gamma,
        )
    }

    /// Forward over `[B, T, in]` → `[B, T, d]`. Single-chunk parallel form.
    pub fn forward(&mut self, gpu: &Gpu, x: &DTensor) -> DTensor {
        assert_eq!(x.rank, 3, "MLstm::forward expects [B, T, in]");
        let (b, t, inp) = (x.shape[0], x.shape[1], x.shape[2]);
        assert_eq!(inp, self.input_size, "MLstm::forward — input width mismatch");
        let (d, h, dqk, dhv) = (self.d, self.heads, self.dqk, self.dhv);
        let n = b * t;

        // Projections on the flat [N, in] view.
        let xf = x.dup(gpu).reshaped(&[n, inp]);
        let q = self.lin_q.forward(gpu, &xf);
        let mut k = self.lin_k.forward(gpu, &xf);
        ops::scale_(gpu, &mut k, self.inv_sqrt_dqk);
        let v = self.lin_v.forward(gpu, &xf);
        let mut o = self.lin_o.forward(gpu, &xf);
        ops::sigmoid_(gpu, &mut o);
        let ig = self.lin_i.forward(gpu, &xf); // [N, H]
        let fg = self.lin_f.forward(gpu, &xf); // [N, H]

        // Head-major reorg for the per-(b,h) batched matmuls.
        let qh = ops::head_gather(gpu, &q, b, h, t, dqk); // [BH, T, dqk]
        let kh = ops::head_gather(gpu, &k, b, h, t, dqk);
        let vh = ops::head_gather(gpu, &v, b, h, t, dhv); // [BH, T, dhv]
        let igh = ops::head_gather(gpu, &ig, b, h, t, 1).reshaped(&[b * h, t]); // [BH, T]
        let fgh = ops::head_gather(gpu, &fg, b, h, t, 1).reshaped(&[b * h, t]);

        // Decay/stabilizer machinery.
        let fc = ops::cumsum_logsig(gpu, &fgh); // [BH, T]
        let s = ops::matmul_batched_nt(gpu, &qh, &kh); // S = Q·Kᵀ  [BH, T, T]
        let m_prev = DTensor::zeros(gpu, &[b * h]); // single chunk: m_0 = 0
        let m = ops::mlstm_rowmax_m(gpu, &fc, &igh, &m_prev);
        let (dbar, ds, qn, psi) = ops::mlstm_ds(gpu, &s, &fc, &igh, &m);
        let num = ops::matmul_batched_nn(gpu, &ds, &vh); // (D̄⊙S)·V  [BH, T, dhv]
        let ytil = ops::div_rows(gpu, &num, &psi, dhv); // ỹ  [BH, T, dhv]

        // Back to position-major, head-norm, o-gate, output projection.
        let h_tilde = ops::head_scatter(gpu, &ytil, b, h, t, dhv); // [N, d]
        let yhat = self.headnorm.forward(gpu, &h_tilde);
        let hconcat = ops::mul(gpu, &o, &yhat); // o ⊙ ŷ  [N, d]
        let out = self.lin_out.forward(gpu, &hconcat); // [N, d]

        // `o`/`yhat` are unused after `mul`, so move (not dup) them into the cache.
        self.saved = Some(Saved { b, t, qh, kh, vh, fgh, dbar, ds, num, qn, psi, o, yhat });
        out.reshaped(&[b, t, d])
    }

    /// Backward over `[B, T, d]` → `dx` `[B, T, in]`. Accumulates all grads.
    pub fn backward(&mut self, gpu: &Gpu, dy: &DTensor) -> DTensor {
        let (d, h, dqk, dhv, inp) = (self.d, self.heads, self.dqk, self.dhv, self.input_size);
        // `take`, not `as_ref`: the cache holds the two [BH, T, T] decay matrices,
        // by far the largest tensors in the model. Dropping them at the end of this
        // call (rather than when the next forward overwrites the field) keeps a
        // window's activations from staying resident across the optimizer step.
        let sv = self.saved.take().expect("MLstm::backward before forward");
        let sv = &sv;
        let (b, t) = (sv.b, sv.t);
        let n = b * t;

        let dy_flat = dy.dup(gpu).reshaped(&[n, d]);

        // Output projection + o-gate.
        let d_hconcat = self.lin_out.backward(gpu, &dy_flat); // [N, d]
        let (do_pre, d_yhat) = ops::ogate_bwd(gpu, &d_hconcat, &sv.o, &sv.yhat);

        // Head-norm backward → d_h_tilde, then head-gather to head-major d_ytil.
        let d_h_tilde = self.headnorm.backward(gpu, &d_yhat); // [N, d]
        let d_ytil = ops::head_gather(gpu, &d_h_tilde, b, h, t, dhv); // [BH, T, dhv]

        // ỹ = num/ψ  → d_num, d_qn.
        let (d_num, d_qn) = ops::div_rows_bwd(gpu, &d_ytil, &sv.num, &sv.psi, &sv.qn, dhv);

        // num = DS·V:  dV = DSᵀ·d_num ;  dDS(num path) = d_num·Vᵀ.
        let dvh = ops::matmul_batched_tn(gpu, &sv.ds, &d_num); // [BH, T, dhv]
        let dds_num = ops::matmul_batched_nt(gpu, &d_num, &sv.vh); // [BH, T, T]

        // DS = D̄⊙S + qn-sum:  dS and P (= dD̄⊙D̄, feeds fc/ig grads).
        let (d_s, p) = ops::mlstm_ds_bwd(gpu, &dds_num, &d_qn, &sv.dbar, &sv.ds);

        // S = Q·Kᵀ:  dQ = dS·K ;  dK = dSᵀ·Q.
        let dqh = ops::matmul_batched_nn(gpu, &d_s, &sv.kh); // [BH, T, dqk]
        let dkh = ops::matmul_batched_tn(gpu, &d_s, &sv.qh); // [BH, T, dqk]

        // Decay grads: P → (dfc, dig); dfc → d(f-logit) via reverse-cumsum·logσ'.
        let (dfc, dig) = ops::mlstm_dfc_dig(gpu, &p);
        let d_fgh = ops::revcumsum_dlogsig(gpu, &dfc, &sv.fgh); // [BH, T]

        // Scatter head-major grads back to position-major [N, ·].
        let dq = ops::head_scatter(gpu, &dqh, b, h, t, dqk); // [N, d_qk]
        let mut dk = ops::head_scatter(gpu, &dkh, b, h, t, dqk);
        ops::scale_(gpu, &mut dk, self.inv_sqrt_dqk); // k = (·)·1/√dqk
        let dv = ops::head_scatter(gpu, &dvh, b, h, t, dhv); // [N, d]
        let d_ig = ops::head_scatter(gpu, &dig.reshaped(&[b * h, t, 1]), b, h, t, 1); // [N, H]
        let d_fg = ops::head_scatter(gpu, &d_fgh.reshaped(&[b * h, t, 1]), b, h, t, 1);

        // Projection backward; sum the input grads (all share the saved xf).
        let mut dxf = self.lin_q.backward(gpu, &dq);
        dxf = ops::add(gpu, &dxf, &self.lin_k.backward(gpu, &dk));
        dxf = ops::add(gpu, &dxf, &self.lin_v.backward(gpu, &dv));
        dxf = ops::add(gpu, &dxf, &self.lin_o.backward(gpu, &do_pre));
        dxf = ops::add(gpu, &dxf, &self.lin_i.backward(gpu, &d_ig));
        dxf = ops::add(gpu, &dxf, &self.lin_f.backward(gpu, &d_fg));
        dxf.reshaped(&[b, t, inp])
    }

    /// Every learnable tensor, in a fixed order (used by checkpoint save/load).
    pub fn params_mut(&mut self) -> Vec<&mut DTensor> {
        let mut v = Vec::new();
        for l in [&mut self.lin_q, &mut self.lin_k, &mut self.lin_v, &mut self.lin_o,
                  &mut self.lin_i, &mut self.lin_f, &mut self.lin_out] {
            v.extend(l.params_mut());
        }
        v.extend(self.headnorm.params_mut());
        v
    }

    pub fn zero_grad(&mut self, gpu: &Gpu) {
        for l in [&mut self.lin_q, &mut self.lin_k, &mut self.lin_v, &mut self.lin_o,
                  &mut self.lin_i, &mut self.lin_f, &mut self.lin_out] {
            l.zero_grad(gpu);
        }
        self.headnorm.zero_grad(gpu);
    }

    /// AdamW step: projection + output matrices decay; biases and head-norm γ
    /// don't (all handled by the sub-layers). Clears the grads.
    pub fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        for l in [&mut self.lin_q, &mut self.lin_k, &mut self.lin_v, &mut self.lin_o,
                  &mut self.lin_i, &mut self.lin_f, &mut self.lin_out] {
            l.step(gpu, cfg);
        }
        self.headnorm.step(gpu, cfg);
    }
}

impl Cell for MLstm {
    fn forward(&mut self, gpu: &Gpu, x: &DTensor) -> DTensor { MLstm::forward(self, gpu, x) }
    fn backward(&mut self, gpu: &Gpu, dy: &DTensor) -> DTensor { MLstm::backward(self, gpu, dy) }
    fn zero_grad(&mut self, gpu: &Gpu) { MLstm::zero_grad(self, gpu) }
    fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) { MLstm::step(self, gpu, cfg) }
    fn params_mut(&mut self) -> Vec<&mut DTensor> { MLstm::params_mut(self) }
    fn wants_post_cell_norm(&self) -> bool { false }
    fn to_nn_block(
        &self,
        gpu: &Gpu,
        hidden: usize,
        up: usize,
        pre_norm1: crate::nn::rms_norm::RMSNorm,
        _post_cell_norm: Option<crate::nn::rms_norm::RMSNorm>,
        pre_norm2: crate::nn::rms_norm::RMSNorm,
        lin_gate: crate::nn::linear::LinearLayer,
        lin_value: crate::nn::linear::LinearLayer,
        lin_down: crate::nn::linear::LinearLayer,
    ) -> Box<dyn crate::nn_layer::NnLayer> {
        // mLSTM blocks have no post-cell norm (the cell's head norm normalizes).
        Box::new(crate::nn::mlstm_block::MLSTMBlock::from_loaded(
            hidden,
            up,
            pre_norm1,
            pre_norm2,
            self.to_nn_cell(gpu),
            lin_gate,
            lin_value,
            lin_down,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn2::mlstm::MLstm as CpuMLstm;
    use crate::nn2::optim::AdamCfg;

    fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length mismatch");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < tol, "{what}[{i}]: gpu {g} vs cpu {w}");
        }
    }

    /// Single-chunk parallel forward+backward+step must match the CPU scalar
    /// recurrence (`nn2::MLstm`) from identical weights. The CPU backward is
    /// itself FD-verified, so a GPU-vs-CPU grad match is the (tighter) check.
    #[test]
    fn mlstm_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else { return };
        let (b, t, inp, d, heads, dqk) = (2, 6, 5, 8, 2, 4); // dhv = 4

        let mut cpu = CpuMLstm::new(inp, d, heads, dqk);
        // Non-trivial gate weights so the decay/stabilizer path is exercised.
        cpu.wi = Tensor::random(&[inp, heads], 0.3);
        cpu.wf = Tensor::random(&[inp, heads], 0.3);
        let mut dev = MLstm::from_cpu(&gpu, &cpu);

        let x = Tensor::random(&[b, t, inp], 0.5);
        let g = Tensor::random(&[b, t, d], 1.0);

        // Forward
        let y_cpu = cpu.forward(&x);
        let y_dev = dev.forward(&gpu, &DTensor::from_host(&gpu, &x));
        assert_close(&y_dev.to_host(&gpu).data, &y_cpu.data, 3e-3, "y");

        // Backward
        let dx_cpu = cpu.backward(&g);
        let dx_dev = dev.backward(&gpu, &DTensor::from_host(&gpu, &g));
        assert_close(&dx_dev.to_host(&gpu).data, &dx_cpu.data, 3e-3, "dx");

        // One AdamW step; compare representative updated parameters.
        let mut cfg = AdamCfg::new(1e-3, 0.01);
        cfg.t = 1;
        cpu.step(&cfg);
        dev.step(&gpu, &cfg);
        // (weights live in the Linear sub-layers; check q, v, out projections + γ)
        assert_close(&dev.lin_q.w.to_host(&gpu).data, &cpu.wq.data, 3e-3, "wq");
        assert_close(&dev.lin_v.w.to_host(&gpu).data, &cpu.wv.data, 3e-3, "wv");
        assert_close(&dev.lin_out.w.to_host(&gpu).data, &cpu.w_out.data, 3e-3, "w_out");
        assert_close(&dev.headnorm.gamma.to_host(&gpu).data, &cpu.gamma.data, 3e-3, "gamma");
    }
}
