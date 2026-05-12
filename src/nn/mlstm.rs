// mlstm.rs - Multi-head mLSTM based on xLSTM (Beck et al. 2024)
//
// Hyperparameters:
//   d   = hidden_size      (input == output)
//   H   = num_heads
//   dqk = query/key dim per head
//   dhv = d / H            (value/output dim per head)
//   d_qk = H * dqk         (full query/key dimension)
//
// Each head maintains its own state:
//   C_h ∈ ℝ^{dhv × dqk}    flattened row-major in self.c
//   n_h ∈ ℝ^{dqk}          normalizer vector
//   m_h ∈ ℝ                scalar stabilizer
//
// Forward pass per timestep:
//   q   = W_q x + b_q                     ∈ ℝ^{d_qk}
//   k   = (W_k x + b_k) / √dqk            ∈ ℝ^{d_qk}
//   v   = W_v x + b_v                     ∈ ℝ^{d}
//   o   = σ(W_o x + b_o)                  ∈ ℝ^{d}
//   ĩ   = W_i x + b_i                     ∈ ℝ^{H}
//   f̃   = W_f x + b_f                     ∈ ℝ^{H}
//
// Per head h, with q_h = q[h*dqk..(h+1)*dqk]:
//   log_f_h = log σ(f̃_h)
//   m_h     = max(log_f_h + m_prev_h, ĩ_h)
//   i'_h    = exp(ĩ_h - m_h)
//   f'_h    = exp(log_f_h + m_prev_h - m_h)
//   C_h     = f'_h * C_prev_h + i'_h * v_h ⊗ k_h
//   n_h     = f'_h * n_prev_h + i'_h * k_h
//   ψ_h     = max(|n_hᵀ q_h|, 1)
//   ỹ_h     = C_h q_h / ψ_h               ∈ ℝ^{dhv}   (raw, before norm)
//   ŷ_h     = HeadwiseRMSNorm(ỹ_h)        ∈ ℝ^{dhv}   (per xLSTM 7B)
//   y_h     = o_h ⊙ ŷ_h                  ∈ ℝ^{dhv}
//
// Concatenate y_h into y_concat ∈ ℝ^{d}, then output:
//   h = W_out · y_concat + b_out           ∈ ℝ^{d}
//
// State layout is flat row-major for C_h and n_h.
// BPTT holds dc_bptt (H·dhv·dqk) and dn_bptt (H·dqk). No dh_bptt is needed.

use iron_oxide::collections::Matrix;

use crate::{
    nn::{
        headwise_rms_norm::{HeadwiseRMSNorm, HeadwiseRMSNormCache},
        linear::{LinearCache, LinearLayer},
    },
    nn_layer::{DynCache, NnLayer},
    optimizers::{GradMatrix, GradMatrixOps, GradVec, GradVecOps},
    saving,
};
use std::{any::Any, io};

#[inline]
fn stable_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
fn log_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        -((-x).exp().ln_1p())
    } else {
        x - x.exp().ln_1p()
    }
}

pub struct MLSTMCache {
    /// Saved input x_t for weight gradient computation.
    pub x: Box<[f32]>,

    // Previous state values needed for backward pass.
    pub c_prev: Box<[f32]>, // H·dhv·dqk
    pub n_prev: Box<[f32]>, // H·dqk
    pub m_prev: Box<[f32]>, // H

    // Forward activations.
    pub q: Box<[f32]>,       // after bias              (H·dqk)
    pub k: Box<[f32]>,       // after bias and scaling  (H·dqk)
    pub v: Box<[f32]>,       // after bias              (d)
    pub o: Box<[f32]>,       // after sigmoid           (d)
    pub i_pre: Box<[f32]>,   // pre-activation          (H)
    pub f_pre: Box<[f32]>,   // pre-activation          (H)
    pub log_f: Box<[f32]>,   // log σ(f̃)               (H)
    pub i_prime: Box<[f32]>, // exp(ĩ - m)             (H)
    pub f_prime: Box<[f32]>, // exp(log_f + m_prev - m) (H)

    // Current state values at timestep t.
    // TODO make this `Box<[Matrix]>`
    pub c: Box<[f32]>, // H·dhv·dqk
    // TODO make this `Matrix`
    pub n: Box<[f32]>, // H·dqk
    pub m: Box<[f32]>, // H

    // Output-Zwischenstufen
    pub cq: Box<[f32]>,                  // (d)  Concat aller C_h·q_h
    pub nq: Box<[f32]>,                  // (H)
    pub psi: Box<[f32]>,                 // (H)
    pub h_tilde: Box<[f32]>,             // (d)  cq / psi  (raw, Eingang in headwise RMS norm)
    pub head_norm: HeadwiseRMSNormCache, // headwise RMS norm: output = ŷ = gamma ⊙ h_tilde / rms

    /// Linear cache for the output projection W_out.
    /// `w_out.input` stores h_concat = o ⊙ head_norm.output.
    /// `w_out.output` stores the final cell output h.
    pub w_out: LinearCache,

    /// dL/d(input).
    pub dx: Box<[f32]>,
}

impl DynCache for MLSTMCache {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn output(&self) -> &[f32] {
        // Cell-Output = W_out · h_concat
        &self.w_out.output
    }
    fn input_grad(&self) -> &[f32] {
        &self.dx
    }
}

pub struct MLSTMLayerGrads {
    pub wq: GradMatrix, // d × H·dqk
    pub wk: GradMatrix, // d × H·dqk
    pub wv: GradMatrix, // d × d
    pub wo: GradMatrix, // d × d
    pub wi: GradMatrix, // d × H
    pub wf: GradMatrix, // d × H

    pub bq: GradVec, // H·dqk
    pub bk: GradVec, // H·dqk
    pub bv: GradVec, // d
    pub bo: GradVec, // d
    pub bi: GradVec, // H
    pub bf: GradVec, // H
}

impl MLSTMLayerGrads {
    pub fn zeros(input_size: usize, hidden_size: usize, num_heads: usize, dqk: usize) -> Self {
        let d_qk = num_heads * dqk;
        Self {
            wq: GradMatrix::zeros(input_size, d_qk),
            wk: GradMatrix::zeros(input_size, d_qk),
            wv: GradMatrix::zeros(input_size, hidden_size),
            wo: GradMatrix::zeros(input_size, hidden_size),
            wi: GradMatrix::zeros(input_size, num_heads),
            wf: GradMatrix::zeros(input_size, num_heads),
            bq: GradVec::zeros(d_qk),
            bk: GradVec::zeros(d_qk),
            bv: GradVec::zeros(hidden_size),
            bo: GradVec::zeros(hidden_size),
            bi: GradVec::zeros(num_heads),
            bf: GradVec::zeros(num_heads),
        }
    }
}

pub struct MLSTMLayer {
    pub input_size: usize,
    pub hidden_size: usize, // d
    pub num_heads: usize,   // H
    pub dqk: usize,
    pub dhv: usize, // = d / H
    pub inv_sqrt_dqk: f32,

    pub wq: Matrix,
    pub wk: Matrix,
    pub wv: Matrix,
    pub wo: Matrix,
    pub wi: Matrix,
    pub wf: Matrix,
    pub bq: Box<[f32]>,
    pub bk: Box<[f32]>,
    pub bv: Box<[f32]>,
    pub bo: Box<[f32]>,
    pub bi: Box<[f32]>,
    pub bf: Box<[f32]>,

    /// Output projection W_out · y_concat + b_out.
    pub w_out: LinearLayer,

    /// Headwise RMS norm applied to ỹ_h = C_h q_h / ψ_h before the output gate.
    pub head_norm: HeadwiseRMSNorm,

    pub c: Box<[f32]>, // H·dhv·dqk
    pub n: Box<[f32]>, // H·dqk
    pub m: Box<[f32]>, // H

    // BPTT-Grads (t+1 → t)
    pub dc_bptt: Box<[f32]>, // H·dhv·dqk
    pub dn_bptt: Box<[f32]>, // H·dqk

    pub grads: MLSTMLayerGrads,

    // Backward-Scratch
    pub dq: Box<[f32]>,        // H·dqk
    pub dk: Box<[f32]>,        // H·dqk  (before /√dqk reverse scaling)
    pub dk_pre: Box<[f32]>,    // H·dqk  (after /√dqk reverse scaling; flows into Wᵀ·dk_pre)
    pub dv: Box<[f32]>,        // d
    pub do_pre: Box<[f32]>,    // d
    pub di_pre: Box<[f32]>,    // H
    pub df_pre: Box<[f32]>,    // H
    pub d_h_tilde: Box<[f32]>, // d  (dL/d(ŷ) → after norm backward: dL/d(ỹ))
    pub dc_total: Box<[f32]>,  // H·dhv·dqk
    pub dn_total: Box<[f32]>,  // H·dqk
}

impl MLSTMLayer {
    pub fn new(input_size: usize, hidden_size: usize, num_heads: usize, dqk: usize) -> Self {
        assert!(num_heads > 0, "num_heads muss > 0 sein");
        assert_eq!(
            hidden_size % num_heads,
            0,
            "hidden_size ({}) muss durch num_heads ({}) teilbar sein",
            hidden_size,
            num_heads,
        );
        let d = hidden_size;
        let heads = num_heads;
        let dhv = d / heads;
        let d_qk = heads * dqk;

        // Glorot-Skalen
        let scale_q = (6.0 / (input_size as f32 + d_qk as f32)).sqrt();
        let scale_v = (6.0 / (input_size as f32 + d as f32)).sqrt();
        let scale_g = (6.0 / (input_size as f32 + heads as f32)).sqrt();

        // Forget-Gate-Bias positiv
        let bf: Box<[f32]> = vec![1.0; heads].into();
        let bi: Box<[f32]> = vec![-10.0; heads].into();

        Self {
            input_size,
            hidden_size: d,
            num_heads: heads,
            dqk,
            dhv,
            inv_sqrt_dqk: 1.0 / (dqk as f32).sqrt(),

            wq: Matrix::random(input_size, d_qk, scale_q),
            wk: Matrix::random(input_size, d_qk, scale_q),
            wv: Matrix::random(input_size, d, scale_v),
            wo: Matrix::random(input_size, d, scale_v),
            wi: Matrix::random(input_size, heads, scale_g),
            wf: Matrix::random(input_size, heads, scale_g),
            bq: vec![0.0; d_qk].into(),
            bk: vec![0.0; d_qk].into(),
            bv: vec![0.0; d].into(),
            bo: vec![0.0; d].into(),
            bi,
            bf,

            // W_out: d → d
            w_out: LinearLayer::from_loaded(
                d,
                d,
                Matrix::random(d, d, (6.0 / (2.0 * d as f32)).sqrt()),
                vec![0.0; d].into(),
            ),

            head_norm: HeadwiseRMSNorm::new(d, heads),

            c: vec![0.0; heads * dhv * dqk].into(),
            n: vec![0.0; heads * dqk].into(),
            m: vec![0.0; heads].into(),

            dc_bptt: vec![0.0; heads * dhv * dqk].into(),
            dn_bptt: vec![0.0; heads * dqk].into(),

            grads: MLSTMLayerGrads::zeros(input_size, d, heads, dqk),

            dq: vec![0.0; d_qk].into(),
            dk: vec![0.0; d_qk].into(),
            dk_pre: vec![0.0; d_qk].into(),
            dv: vec![0.0; d].into(),
            do_pre: vec![0.0; d].into(),
            di_pre: vec![0.0; heads].into(),
            df_pre: vec![0.0; heads].into(),
            d_h_tilde: vec![0.0; d].into(),
            dc_total: vec![0.0; heads * dhv * dqk].into(),
            dn_total: vec![0.0; heads * dqk].into(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_loaded(
        input_size: usize,
        hidden_size: usize,
        num_heads: usize,
        dqk: usize,
        wq: Matrix,
        wk: Matrix,
        wv: Matrix,
        wo: Matrix,
        wi: Matrix,
        wf: Matrix,
        bq: Box<[f32]>,
        bk: Box<[f32]>,
        bv: Box<[f32]>,
        bo: Box<[f32]>,
        bi: Box<[f32]>,
        bf: Box<[f32]>,
        w_out: LinearLayer,
        head_norm_gamma: Box<[f32]>,
    ) -> Self {
        let d = hidden_size;
        let h = num_heads;
        let dhv = d / h;
        let d_qk = h * dqk;
        Self {
            input_size,
            hidden_size: d,
            num_heads: h,
            dqk,
            dhv,
            inv_sqrt_dqk: 1.0 / (dqk as f32).sqrt(),
            wq,
            wk,
            wv,
            wo,
            wi,
            wf,
            bq,
            bk,
            bv,
            bo,
            bi,
            bf,
            w_out,
            head_norm: HeadwiseRMSNorm::from_loaded(d, h, head_norm_gamma),
            c: vec![0.0; h * dhv * dqk].into(),
            n: vec![0.0; h * dqk].into(),
            m: vec![0.0; h].into(),
            dc_bptt: vec![0.0; h * dhv * dqk].into(),
            dn_bptt: vec![0.0; h * dqk].into(),
            grads: MLSTMLayerGrads::zeros(input_size, d, h, dqk),
            dq: vec![0.0; d_qk].into(),
            dk: vec![0.0; d_qk].into(),
            dk_pre: vec![0.0; d_qk].into(),
            dv: vec![0.0; d].into(),
            do_pre: vec![0.0; d].into(),
            di_pre: vec![0.0; h].into(),
            df_pre: vec![0.0; h].into(),
            d_h_tilde: vec![0.0; d].into(),
            dc_total: vec![0.0; h * dhv * dqk].into(),
            dn_total: vec![0.0; h * dqk].into(),
        }
    }

    pub fn forward(&mut self, input: &[f32], cache: &mut MLSTMCache) {
        let d = self.hidden_size;
        let h = self.num_heads;
        let dqk = self.dqk;
        let dhv = self.dhv;
        let d_qk = h * dqk;
        debug_assert_eq!(input.len(), self.input_size);

        // Save previous state for backward pass.
        cache.c_prev.copy_from_slice(&self.c);
        cache.n_prev.copy_from_slice(&self.n);
        cache.m_prev.copy_from_slice(&self.m);
        cache.x.copy_from_slice(input);

        // Vector pre-activations: one matrix-vector multiply per gate.
        self.wq.row_mul(input, &mut cache.q);
        self.wk.row_mul(input, &mut cache.k); // writes raw k values; bias and scaling applied next
        self.wv.row_mul(input, &mut cache.v);
        self.wo.row_mul(input, &mut cache.o); // writes raw o values; sigmoid applied next
        for j in 0..d_qk {
            cache.q[j] += self.bq[j];
            cache.k[j] = (cache.k[j] + self.bk[j]) * self.inv_sqrt_dqk;
        }
        for j in 0..d {
            cache.v[j] += self.bv[j];
            cache.o[j] = stable_sigmoid(cache.o[j] + self.bo[j]);
        }

        // Scalar pre-activations per head.
        self.wi.row_mul(input, &mut cache.i_pre);
        self.wf.row_mul(input, &mut cache.f_pre);
        for hd in 0..h {
            cache.i_pre[hd] += self.bi[hd];
            cache.f_pre[hd] += self.bf[hd];
        }

        // Per-Head Update: C, n, cq, nq, psi, h_tilde (raw)
        for hd in 0..h {
            let qk_off = hd * dqk;
            let v_off = hd * dhv;
            let c_off = hd * dhv * dqk;

            // Stabilizer
            cache.log_f[hd] = log_sigmoid(cache.f_pre[hd]);
            let mh = (cache.log_f[hd] + cache.m_prev[hd]).max(cache.i_pre[hd]);
            cache.m[hd] = mh;
            let i_prime = (cache.i_pre[hd] - mh).exp();
            let f_prime = (cache.log_f[hd] + cache.m_prev[hd] - mh).exp();
            cache.i_prime[hd] = i_prime;
            cache.f_prime[hd] = f_prime;

            // C_h-Update:  C_h[i,j] = f' · C_prev_h[i,j] + i' · v_i · k_j
            for i in 0..dhv {
                let v_i = cache.v[v_off + i];
                let row_off = c_off + i * dqk;
                for j in 0..dqk {
                    cache.c[row_off + j] =
                        f_prime * cache.c_prev[row_off + j] + i_prime * v_i * cache.k[qk_off + j];
                }
            }
            // n_h-Update
            for j in 0..dqk {
                cache.n[qk_off + j] =
                    f_prime * cache.n_prev[qk_off + j] + i_prime * cache.k[qk_off + j];
            }

            // Cq_h[i] = Σ_j C_h[i,j] · q_h[j]
            for i in 0..dhv {
                let row_off = c_off + i * dqk;
                let mut s = 0.0;
                for j in 0..dqk {
                    s += cache.c[row_off + j] * cache.q[qk_off + j];
                }
                cache.cq[v_off + i] = s;
            }
            // n_h⊤ q_h, ψ_h
            let mut nq = 0.0;
            for j in 0..dqk {
                nq += cache.n[qk_off + j] * cache.q[qk_off + j];
            }
            cache.nq[hd] = nq;
            cache.psi[hd] = nq.abs().max(1.0);

            // Raw ỹ_h = C_h q_h / ψ_h (stored in h_tilde before norm is applied)
            let psi = cache.psi[hd];
            for i in 0..dhv {
                cache.h_tilde[v_off + i] = cache.cq[v_off + i] / psi;
            }
        }

        // Headwise RMS norm: ŷ = head_norm(ỹ)  →  head_norm.output
        self.head_norm
            .forward_into(&cache.h_tilde, &mut cache.head_norm);

        // h_concat = o ⊙ ŷ, then output projection h = W_out · h_concat + b_out.
        for i in 0..d {
            cache.w_out.input[i] = cache.o[i] * cache.head_norm.output[i];
        }
        cache.w_out.output.copy_from_slice(&self.w_out.biases);
        for (i, &xi) in cache.w_out.input.iter().enumerate() {
            for (j, &w) in self.w_out.weights[i].iter().enumerate() {
                cache.w_out.output[j] += xi * w;
            }
        }

        // Persistenten State propagieren
        self.c.copy_from_slice(&cache.c);
        self.n.copy_from_slice(&cache.n);
        self.m.copy_from_slice(&cache.m);
    }

    // Incoming `delta` = dL/d(cell-output), size d.
    pub fn backward(&mut self, delta: &mut [f32], cache: &mut MLSTMCache) {
        let d = self.hidden_size;
        let h = self.num_heads;
        let dqk = self.dqk;
        let dhv = self.dhv;
        let d_qk = h * dqk;
        let inv_sqrt_dqk = self.inv_sqrt_dqk;

        self.w_out.backward(delta, &mut cache.w_out);

        // Pass 1: output gate backward → dL/d(ŷ) into d_h_tilde.
        // h_concat = o ⊙ ŷ  →  do_pre = dL/dh · ŷ · σ'(o_pre)
        //                        dL/dŷ  = dL/dh · o
        for hd in 0..h {
            let v_off = hd * dhv;
            for i in 0..dhv {
                let o_i = cache.o[v_off + i];
                let y_hat_i = cache.head_norm.output[v_off + i];
                let dc_h_i = cache.w_out.dx[v_off + i];
                self.do_pre[v_off + i] = dc_h_i * y_hat_i * o_i * (1.0 - o_i);
                self.d_h_tilde[v_off + i] = dc_h_i * o_i;
            }
        }

        // Headwise RMS norm backward: dL/d(ŷ) → dL/d(ỹ)  into head_norm.dx.
        // NLL borrow checker allows &self.d_h_tilde and &mut self.head_norm simultaneously.
        self.head_norm
            .backward_into(&self.d_h_tilde, &mut cache.head_norm);
        self.d_h_tilde.copy_from_slice(&cache.head_norm.dx);

        // Pass 2: per-head backward through ψ, C, n, q, k, v, i, f gates.
        for hd in 0..h {
            let qk_off = hd * dqk;
            let v_off = hd * dhv;
            let c_off = hd * dhv * dqk;

            // dψ_h and dnq_h for ψ_h = max(|nq|, 1)
            let psi = cache.psi[hd];
            let mut dpsi = 0.0;
            for i in 0..dhv {
                dpsi += self.d_h_tilde[v_off + i] * (-cache.cq[v_off + i] / (psi * psi));
            }
            let dnq_h = if cache.nq[hd].abs() > 1.0 {
                cache.nq[hd].signum() * dpsi
            } else {
                0.0
            };

            // dC_total_h[i,j] = (d_h_tilde[i] / ψ) * q_h[j] + dC_bptt_h[i,j]
            // dn_total_h[j] = dnq_h * q_h[j] + dn_bptt_h[j]
            for i in 0..dhv {
                let dh_over_psi = self.d_h_tilde[v_off + i] / psi;
                let row_off = c_off + i * dqk;
                for j in 0..dqk {
                    self.dc_total[row_off + j] =
                        dh_over_psi * cache.q[qk_off + j] + self.dc_bptt[row_off + j];
                }
            }
            for j in 0..dqk {
                self.dn_total[qk_off + j] = dnq_h * cache.q[qk_off + j] + self.dn_bptt[qk_off + j];
            }

            // dq_h[j] = Σ_i (d_h_tilde[i] / ψ) · C_h[i,j] + dnq_h · n_h[j]
            let dq_slice = &mut self.dq[qk_off..qk_off + dqk];
            for j in 0..dqk {
                dq_slice[j] = dnq_h * cache.n[qk_off + j];
            }
            for i in 0..dhv {
                let dh_over_psi = self.d_h_tilde[v_off + i] / psi;
                let row_off = c_off + i * dqk;
                let c_row = &cache.c[row_off..row_off + dqk];
                for j in 0..dqk {
                    dq_slice[j] += dh_over_psi * c_row[j];
                }
            }

            // C_h = f' * C_prev_h + i' * v_h ⊗ k_h
            // n_h = f' * n_prev_h + i' * k_h
            let i_prime = cache.i_prime[hd];
            let f_prime = cache.f_prime[hd];
            let mut df_prime = 0.0;
            let mut di_prime = 0.0;

            for i in 0..dhv {
                let v_i = cache.v[v_off + i];
                let row_off = c_off + i * dqk;
                let mut row_dv = 0.0;
                for j in 0..dqk {
                    let dct = self.dc_total[row_off + j];
                    df_prime += dct * cache.c_prev[row_off + j];
                    di_prime += dct * v_i * cache.k[qk_off + j];
                    row_dv += dct * cache.k[qk_off + j];
                }
                self.dv[v_off + i] = i_prime * row_dv;
            }

            let dk_slice = &mut self.dk[qk_off..qk_off + dqk];
            for j in 0..dqk {
                dk_slice[j] = i_prime * self.dn_total[qk_off + j];
            }
            for i in 0..dhv {
                let v_i = cache.v[v_off + i];
                let row_off = c_off + i * dqk;
                let dc_row = &self.dc_total[row_off..row_off + dqk];
                for j in 0..dqk {
                    dk_slice[j] += i_prime * dc_row[j] * v_i;
                }
            }

            for j in 0..dqk {
                df_prime += self.dn_total[qk_off + j] * cache.n_prev[qk_off + j];
                di_prime += self.dn_total[qk_off + j] * cache.k[qk_off + j];
            }

            for i in 0..dhv {
                let row_off = c_off + i * dqk;
                for j in 0..dqk {
                    self.dc_bptt[row_off + j] = f_prime * self.dc_total[row_off + j];
                }
            }
            for j in 0..dqk {
                self.dn_bptt[qk_off + j] = f_prime * self.dn_total[qk_off + j];
            }

            self.di_pre[hd] = di_prime * i_prime;
            let sigm_f = cache.log_f[hd].exp();
            self.df_pre[hd] = df_prime * f_prime * (1.0 - sigm_f);
        }

        for j in 0..d_qk {
            self.dk_pre[j] = self.dk[j] * inv_sqrt_dqk;
        }

        let g = &mut self.grads;
        g.wq.matrix().add_outer(&cache.x, &self.dq);
        g.wk.matrix().add_outer(&cache.x, &self.dk_pre);
        g.wv.matrix().add_outer(&cache.x, &self.dv);
        g.wo.matrix().add_outer(&cache.x, &self.do_pre);
        g.wi.matrix().add_outer(&cache.x, &self.di_pre);
        g.wf.matrix().add_outer(&cache.x, &self.df_pre);

        for j in 0..d_qk {
            g.bq.vec()[j] += self.dq[j];
            g.bk.vec()[j] += self.dk_pre[j];
        }
        for j in 0..d {
            g.bv.vec()[j] += self.dv[j];
            g.bo.vec()[j] += self.do_pre[j];
        }
        for hd in 0..h {
            g.bi.vec()[hd] += self.di_pre[hd];
            g.bf.vec()[hd] += self.df_pre[hd];
        }

        for (idx, dxi) in cache.dx.iter_mut().enumerate() {
            let mut s = 0.0;

            for j in 0..d_qk {
                s += self.wq[idx][j] * self.dq[j] + self.wk[idx][j] * self.dk_pre[j];
            }
            for j in 0..d {
                s += self.wv[idx][j] * self.dv[j] + self.wo[idx][j] * self.do_pre[j];
            }
            for hd in 0..h {
                s += self.wi[idx][hd] * self.di_pre[hd] + self.wf[idx][hd] * self.df_pre[hd];
            }
            *dxi = s;
        }
    }

    pub fn alloc_cache(&self) -> MLSTMCache {
        let d = self.hidden_size;
        let h = self.num_heads;
        let dqk = self.dqk;
        let dhv = self.dhv;
        let d_qk = h * dqk;
        MLSTMCache {
            x: vec![0.0; self.input_size].into(),
            c_prev: vec![0.0; h * dhv * dqk].into(),
            n_prev: vec![0.0; h * dqk].into(),
            m_prev: vec![0.0; h].into(),
            q: vec![0.0; d_qk].into(),
            k: vec![0.0; d_qk].into(),
            v: vec![0.0; d].into(),
            o: vec![0.0; d].into(),
            i_pre: vec![0.0; h].into(),
            f_pre: vec![0.0; h].into(),
            log_f: vec![0.0; h].into(),
            i_prime: vec![0.0; h].into(),
            f_prime: vec![0.0; h].into(),
            c: vec![0.0; h * dhv * dqk].into(),
            n: vec![0.0; h * dqk].into(),
            m: vec![0.0; h].into(),
            cq: vec![0.0; d].into(),
            nq: vec![0.0; h].into(),
            psi: vec![1.0; h].into(),
            h_tilde: vec![0.0; d].into(),
            head_norm: self.head_norm.alloc_cache(),
            w_out: LinearCache::new(d, d),
            dx: vec![0.0; self.input_size].into(),
        }
    }
}

impl NnLayer for MLSTMLayer {
    //type Cache = MLSTMCache;
    fn forward(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<MLSTMCache>()
            .expect("MLSTMLayer::forward — expected MLSTMCache");
        MLSTMLayer::forward(self, input, c);
    }

    fn backward(&mut self, delta: &mut [f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<MLSTMCache>()
            .expect("MLSTMLayer::backward — expected MLSTMCache");
        MLSTMLayer::backward(self, delta, c);
    }

    fn layer_tag(&self) -> u8 {
        13
    }

    fn save(&self, w: &mut dyn io::Write) -> io::Result<()> {
        saving::write_u32(w, self.num_heads as u32)?;
        saving::write_u32(w, self.dqk as u32)?;
        saving::write_matrix(w, &self.wq)?;
        saving::write_matrix(w, &self.wk)?;
        saving::write_matrix(w, &self.wv)?;
        saving::write_matrix(w, &self.wo)?;
        saving::write_matrix(w, &self.wi)?;
        saving::write_matrix(w, &self.wf)?;
        saving::write_f32_slice(w, &self.bq)?;
        saving::write_f32_slice(w, &self.bk)?;
        saving::write_f32_slice(w, &self.bv)?;
        saving::write_f32_slice(w, &self.bo)?;
        saving::write_f32_slice(w, &self.bi)?;
        saving::write_f32_slice(w, &self.bf)?;
        saving::write_matrix(w, &self.w_out.weights)?;
        saving::write_f32_slice(w, &self.w_out.biases)?;
        saving::write_f32_slice(w, &self.head_norm.gamma)
    }

    fn make_cache(&self) -> Box<dyn DynCache> {
        Box::new(MLSTMLayer::alloc_cache(self))
    }

    fn input_size(&self) -> usize {
        self.input_size
    }
    fn output_size(&self) -> usize {
        self.hidden_size
    }

    fn apply_grads(&mut self, lr: f32) {
        self.grads.wq.apply_to(&mut self.wq, lr);
        self.grads.wk.apply_to(&mut self.wk, lr);
        self.grads.wv.apply_to(&mut self.wv, lr);
        self.grads.wo.apply_to(&mut self.wo, lr);
        self.grads.wi.apply_to(&mut self.wi, lr);
        self.grads.wf.apply_to(&mut self.wf, lr);

        self.grads.bq.apply_to(&mut self.bq, lr);
        self.grads.bk.apply_to(&mut self.bk, lr);
        self.grads.bv.apply_to(&mut self.bv, lr);
        self.grads.bo.apply_to(&mut self.bo, lr);
        self.grads.bi.apply_to(&mut self.bi, lr);
        self.grads.bf.apply_to(&mut self.bf, lr);

        self.w_out.apply_grads(lr);
        self.head_norm.apply_grads(lr);
    }

    fn clear_grads(&mut self) {
        self.grads.wq.clear();
        self.grads.wk.clear();
        self.grads.wv.clear();
        self.grads.wo.clear();
        self.grads.wi.clear();
        self.grads.wf.clear();
        self.grads.bq.clear();
        self.grads.bk.clear();
        self.grads.bv.clear();
        self.grads.bo.clear();
        self.grads.bi.clear();
        self.grads.bf.clear();
        self.w_out.clear_grads();
        self.head_norm.clear_grads();
    }

    fn reset_state(&mut self) {
        self.c.fill(0.0);
        self.n.fill(0.0);
        self.m.fill(0.0);
    }

    fn zero_bptt_state(&mut self) {
        self.dc_bptt.fill(0.0);
        self.dn_bptt.fill(0.0);
    }
}
