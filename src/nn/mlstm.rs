// mlstm.rs - Multi-head mLSTM based on xLSTM (Beck et al. 2024)
//
// Hyperparameters:
//   d   = hidden_size      (input == output)
//   H   = num_heads
//   dqk = query/key dim per head
//   dhv = d / H            (value/output dim per head)
//
// Each head maintains its own state:
//   C_h ∈ ℝ^{dhv × dqk}    stored as Box<[Matrix]> (one Matrix per head)
//   n_h ∈ ℝ^{dqk}          normalizer — Matrix H × dqk
//   m_h ∈ ℝ                scalar stabilizer
//
// Forward pass per timestep:
//   q   = W_q x + b_q                     ∈ ℝ^{H × dqk}
//   k   = (W_k x + b_k) / √dqk            ∈ ℝ^{H × dqk}
//   v   = W_v x + b_v                     ∈ ℝ^{d}
//   o   = σ(W_o x + b_o)                  ∈ ℝ^{d}
//   ĩ   = W_i x + b_i                     ∈ ℝ^{H}
//   f̃   = W_f x + b_f                     ∈ ℝ^{H}
//
// Per head h:
//   log_f_h = log σ(f̃_h)
//   m_h     = max(log_f_h + m_prev_h, ĩ_h)
//   i'_h    = exp(ĩ_h - m_h)
//   f'_h    = exp(log_f_h + m_prev_h - m_h)
//   C_h     = f'_h * C_prev_h + i'_h * v_h ⊗ k_h
//   n_h     = f'_h * n_prev_h + i'_h * k_h
//   ψ_h     = max(|n_hᵀ q_h|, 1)
//   ỹ_h     = C_h q_h / ψ_h               ∈ ℝ^{dhv}
//   ŷ_h     = HeadwiseRMSNorm(ỹ_h)
//   y_h     = o_h ⊙ ŷ_h
//
// Concatenate y_h → y_concat ∈ ℝ^{d}, then:
//   h = W_out · y_concat + b_out
//
// All per-head work is done inside one head loop in both forward and backward.
// BPTT holds dc_bptt (H matrices dhv×dqk) and dn_bptt (H×dqk matrix).

use iron_oxide::collections::Matrix;
use rand::random_range;

use crate::{
    nn::{
        activations::{log_sigmoid, stable_sigmoid},
        headwise_rms_norm::{HeadwiseRMSNorm, HeadwiseRMSNormCache},
        linear::{LinearCache, LinearLayer},
    },
    nn_layer::{DynCache, NnLayer},
    optimizers::{GradMatrix, GradMatrixOps, GradVec, GradVecOps, add_grad_matrix, add_grad_vec},
    saving,
};
use std::{any::Any, io};

pub struct MLSTMCache {
    pub x: Box<[f32]>,

    // State before this timestep — needed by backward for df_prime/dn.
    pub c_prev: Box<[Matrix]>, // H matrices of dhv × dqk
    pub n_prev: Matrix,        // H × dqk

    // Forward activations.
    pub q: Box<[f32]>,       // H·dqk  flat, row-major by head  (after bias)
    pub k: Box<[f32]>,       // H·dqk  flat, row-major by head  (after bias and scaling)
    pub v: Box<[f32]>,       // d  (after bias)
    pub o: Box<[f32]>,       // d  (after sigmoid)
    pub i_pre: Box<[f32]>,   // H  (pre-activation)
    pub f_pre: Box<[f32]>,   // H  (pre-activation)
    pub log_f: Box<[f32]>,   // H  log σ(f̃)
    pub i_prime: Box<[f32]>, // H  exp(ĩ - m)
    pub f_prime: Box<[f32]>, // H  exp(log_f + m_prev - m)

    // Output intermediates.
    pub cq: Box<[f32]>,      // d
    pub nq: Box<[f32]>,      // H
    pub psi: Box<[f32]>,     // H
    pub h_tilde: Box<[f32]>, // d
    pub head_norm: HeadwiseRMSNormCache,

    /// `w_out.input` = o ⊙ ŷ,  `w_out.output` = final cell output.
    pub w_out: LinearCache,

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
        &self.w_out.output
    }
    fn input_grad(&self) -> &[f32] {
        &self.dx
    }
}

pub struct MLSTMLayerGrads {
    pub wq: GradMatrix, // input_size × H·dqk
    pub wk: GradMatrix,
    pub wv: GradMatrix, // input_size × d
    pub wo: GradMatrix,
    pub wi: GradMatrix, // input_size × H
    pub wf: GradMatrix,

    pub bq: GradVec, // H·dqk
    pub bk: GradVec,
    pub bv: GradVec, // d
    pub bo: GradVec,
    pub bi: GradVec, // H
    pub bf: GradVec,
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
    pub dhv: usize, // d / H
    pub inv_sqrt_dqk: f32,

    pub wq: Matrix,
    pub wk: Matrix,
    pub wv: Matrix,
    pub wo: Matrix,
    pub wi: Matrix,
    pub wf: Matrix,
    pub bq: Box<[f32]>, // H·dqk  (flat, row-major by head)
    pub bk: Box<[f32]>,
    pub bv: Box<[f32]>, // d
    pub bo: Box<[f32]>,
    pub bi: Box<[f32]>, // H
    pub bf: Box<[f32]>,

    pub w_out: LinearLayer,
    pub head_norm: HeadwiseRMSNorm,

    pub c: Box<[Matrix]>, // H matrices of dhv × dqk
    pub n: Matrix,        // H × dqk
    pub m: Box<[f32]>,    // H

    // BPTT-Grads (t+1 → t) — doubled as dc_total/dn_total scratch during backward.
    pub dc_bptt: Box<[Matrix]>, // H matrices of dhv × dqk
    pub dn_bptt: Matrix,        // H × dqk

    pub grads: MLSTMLayerGrads,

    // Backward scratch
    pub dq: Box<[f32]>,        // H·dqk  flat, row-major by head
    pub dk_pre: Box<[f32]>,    // H·dqk  (dk scaled by 1/√dqk, written flat)
    pub dv: Box<[f32]>,        // d
    pub do_pre: Box<[f32]>,    // d
    pub di_pre: Box<[f32]>,    // H
    pub df_pre: Box<[f32]>,    // H
    pub d_h_tilde: Box<[f32]>, // d
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

        let scale_q = (6.0 / (input_size as f32 + d_qk as f32)).sqrt();
        let scale_v = (6.0 / (input_size as f32 + d as f32)).sqrt();

        let bf: Box<[f32]> = (0..heads).map(|_| random_range(3.0..6.0)).collect();
        let bi: Box<[f32]> = (0..heads).map(|_| random_range(-6.0..-3.0)).collect();

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
            wi: Matrix::zeros(input_size, heads),
            wf: Matrix::zeros(input_size, heads),
            bq: vec![0.0; d_qk].into(),
            bk: vec![0.0; d_qk].into(),
            bv: vec![0.0; d].into(),
            bo: vec![0.0; d].into(),
            bi,
            bf,

            w_out: LinearLayer::from_loaded(
                d,
                d,
                Matrix::random(d, d, (6.0 / (2.0 * d as f32)).sqrt()),
                vec![0.0; d].into(),
            ),
            head_norm: HeadwiseRMSNorm::new(d, heads),

            c: (0..heads).map(|_| Matrix::zeros(dhv, dqk)).collect(),
            n: Matrix::zeros(heads, dqk),
            m: vec![0.0; heads].into(),

            dc_bptt: (0..heads).map(|_| Matrix::zeros(dhv, dqk)).collect(),
            dn_bptt: Matrix::zeros(heads, dqk),

            grads: MLSTMLayerGrads::zeros(input_size, d, heads, dqk),

            dq: vec![0.0; d_qk].into(),
            dk_pre: vec![0.0; d_qk].into(),
            dv: vec![0.0; d].into(),
            do_pre: vec![0.0; d].into(),
            di_pre: vec![0.0; heads].into(),
            df_pre: vec![0.0; heads].into(),
            d_h_tilde: vec![0.0; d].into(),
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
            c: (0..h).map(|_| Matrix::zeros(dhv, dqk)).collect(),
            n: Matrix::zeros(h, dqk),
            m: vec![0.0; h].into(),
            dc_bptt: (0..h).map(|_| Matrix::zeros(dhv, dqk)).collect(),
            dn_bptt: Matrix::zeros(h, dqk),
            grads: MLSTMLayerGrads::zeros(input_size, d, h, dqk),
            dq: vec![0.0; d_qk].into(),
            dk_pre: vec![0.0; d_qk].into(),
            dv: vec![0.0; d].into(),
            do_pre: vec![0.0; d].into(),
            di_pre: vec![0.0; h].into(),
            df_pre: vec![0.0; h].into(),
            d_h_tilde: vec![0.0; d].into(),
        }
    }

    pub fn forward(&mut self, input: &[f32], cache: &mut MLSTMCache) {
        let d = self.hidden_size;
        let h = self.num_heads;
        let dqk = self.dqk;
        let dhv = self.dhv;
        debug_assert_eq!(input.len(), self.input_size);

        cache.x.copy_from_slice(input);

        // Matrix-vector multiplies for all gates.
        // (Forking these across rayon was tried and measured SLOWER at the
        // current dims — one matvec is only ~65K MACs, below fork/steal
        // overhead. Revisit if hidden sizes grow well past 256.)
        self.wq.row_mul(input, &mut cache.q);
        self.wk.row_mul(input, &mut cache.k);
        self.wv.row_mul(input, &mut cache.v);
        self.wo.row_mul(input, &mut cache.o);
        self.wi.row_mul(input, &mut cache.i_pre);
        self.wf.row_mul(input, &mut cache.f_pre);

        // Bias + scaling — contiguous flat passes, SIMD-friendly.
        for j in 0..h * dqk {
            cache.q[j] += self.bq[j];
            cache.k[j] = (cache.k[j] + self.bk[j]) * self.inv_sqrt_dqk;
        }
        for i in 0..d {
            cache.v[i] += self.bv[i];
            cache.o[i] = stable_sigmoid(cache.o[i] + self.bo[i]);
        }
        for hd in 0..h {
            cache.i_pre[hd] += self.bi[hd];
            cache.f_pre[hd] += self.bf[hd];
        }

        // O(1) swap — cache.n_prev becomes n_{t-1}, self.n becomes the write buffer for n_t.
        // Mirrors the swap at the end of backward; no per-element copy needed.
        std::mem::swap(&mut self.n, &mut cache.n_prev);

        for hd in 0..h {
            let qk_off = hd * dqk;
            let v_off = hd * dhv;

            // Stabilizer.
            let m_prev_h = self.m[hd];
            cache.log_f[hd] = log_sigmoid(cache.f_pre[hd]);
            let mh = (cache.log_f[hd] + m_prev_h).max(cache.i_pre[hd]);
            self.m[hd] = mh;
            let i_prime = (cache.i_pre[hd] - mh).exp();
            let f_prime = (cache.log_f[hd] + m_prev_h - mh).exp();
            cache.i_prime[hd] = i_prime;
            cache.f_prime[hd] = f_prime;

            // Fused n update + nq. Reads n_{t-1} from cache.n_prev (after the swap above),
            // writes n_t into self.n — one pass, no extra n copy.
            let mut nq = 0.0;
            for j in 0..dqk {
                let n_prev_j = cache.n_prev[hd][j];
                let n_j = f_prime * n_prev_j + i_prime * cache.k[qk_off + j];
                self.n[hd][j] = n_j;
                nq += n_j * cache.q[qk_off + j];
            }
            cache.nq[hd] = nq;
            let psi = nq.abs().max(1.0);
            cache.psi[hd] = psi;
            let inv_psi = 1.0 / psi;

            // O(1) swap — cache.c_prev[hd] becomes C_{t-1}, self.c[hd] becomes the write
            // buffer. Mirrors the swap at the end of backward; no per-element copy.
            std::mem::swap(&mut self.c[hd], &mut cache.c_prev[hd]);
            for i in 0..dhv {
                let v_i = cache.v[v_off + i];
                let iprime_vi = i_prime * v_i;
                let mut s = 0.0;
                for j in 0..dqk {
                    let c_ij = f_prime * cache.c_prev[hd][i][j] + iprime_vi * cache.k[qk_off + j];
                    self.c[hd][i][j] = c_ij;
                    s += c_ij * cache.q[qk_off + j];
                }
                cache.cq[v_off + i] = s;
                cache.h_tilde[v_off + i] = s * inv_psi;
            }
        }

        // Headwise RMS norm: ŷ = head_norm(h_tilde)
        self.head_norm
            .forward_into(&cache.h_tilde, &mut cache.head_norm);

        // h_concat = o ⊙ ŷ,  output h = W_out · h_concat + b_out
        for i in 0..d {
            cache.w_out.input[i] = cache.o[i] * cache.head_norm.output[i];
        }
        cache.w_out.output.copy_from_slice(&self.w_out.biases);
        for (i, &xi) in cache.w_out.input.iter().enumerate() {
            for (j, &w) in self.w_out.weights[i].iter().enumerate() {
                cache.w_out.output[j] += xi * w;
            }
        }
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

        for i in 0..d {
            let o_i = cache.o[i];
            let dc_h_i = cache.w_out.dx[i];
            self.do_pre[i] = dc_h_i * cache.head_norm.output[i] * o_i * (1.0 - o_i);
            self.d_h_tilde[i] = dc_h_i * o_i;
        }

        self.head_norm
            .backward_into(&self.d_h_tilde, &mut cache.head_norm);
        self.d_h_tilde.copy_from_slice(&cache.head_norm.dx);

        for hd in 0..h {
            let qk_off = hd * dqk;
            let v_off = hd * dhv;

            let inv_psi = 1.0 / cache.psi[hd];
            let psi_sq = inv_psi * inv_psi;
            let mut dpsi = 0.0;
            for i in 0..dhv {
                dpsi += self.d_h_tilde[v_off + i] * (-cache.cq[v_off + i] * psi_sq);
            }
            let dnq_h = if cache.nq[hd].abs() > 1.0 {
                cache.nq[hd].signum() * dpsi
            } else {
                0.0
            };

            let i_prime = cache.i_prime[hd];
            let f_prime = cache.f_prime[hd];
            let mut df_prime = 0.0;
            let mut di_prime = 0.0;

            // Init dq, zero dk_pre, fold dnq into dn_bptt — single j pass.
            for j in 0..dqk {
                self.dq[qk_off + j] = dnq_h * self.n[hd][j];
                self.dk_pre[qk_off + j] = 0.0;
                self.dn_bptt[hd][j] += dnq_h * cache.q[qk_off + j];
            }

            // Fused (i,j): update dc_bptt → dc_total, dq from C, df/dv/dk — one pass over dc_bptt.
            for i in 0..dhv {
                let dh_over_psi = self.d_h_tilde[v_off + i] * inv_psi;
                let v_i = cache.v[v_off + i];
                let mut row_dv = 0.0;
                let mut row_df = 0.0;
                for j in 0..dqk {
                    let dc = self.dc_bptt[hd][i][j] + dh_over_psi * cache.q[qk_off + j];
                    self.dc_bptt[hd][i][j] = dc * f_prime; // scale for BPTT in-place
                    self.dq[qk_off + j] += dh_over_psi * self.c[hd][i][j];
                    row_df += dc * cache.c_prev[hd][i][j];
                    row_dv += dc * cache.k[qk_off + j];
                    self.dk_pre[qk_off + j] += dc * v_i;
                }
                df_prime += row_df;
                di_prime += v_i * row_dv; // saves dhv·dqk multiplications vs inner accumulation
                self.dv[v_off + i] = i_prime * row_dv;
            }

            // dn contribution to df/di/dk, finalize dk_pre with i', scale dn_bptt.
            for j in 0..dqk {
                let dn = self.dn_bptt[hd][j];
                df_prime += dn * cache.n_prev[hd][j];
                di_prime += dn * cache.k[qk_off + j];
                self.dk_pre[qk_off + j] = i_prime * (self.dk_pre[qk_off + j] + dn);
                self.dn_bptt[hd][j] = dn * f_prime;
            }

            self.di_pre[hd] = di_prime * i_prime;
            let sigm_f = cache.log_f[hd].exp();
            self.df_pre[hd] = df_prime * f_prime * (1.0 - sigm_f);
        }

        // Scale dk_pre by 1/√dqk.
        for j in 0..d_qk {
            self.dk_pre[j] *= inv_sqrt_dqk;
        }

        // Bias grads.
        {
            let g = &mut self.grads;
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
        }

        // Fused weight-grad accumulation + dx — one pass per input row instead of 7.
        // Each weight row and its grad row are touched together while both fit in L1,
        // avoiding the double L2/L3 reload that separate add_outer + dx caused.
        // (Chunking this pass across rayon was tried and measured slower at the
        // current dims — per-row work is too small; keep it serial.)
        {
            let wq = self.wq.as_slice();
            let wk = self.wk.as_slice();
            let wv = self.wv.as_slice();
            let wo = self.wo.as_slice();
            let wi = self.wi.as_slice();
            let wf = self.wf.as_slice();
            // Different fields from wq..wf → Rust field borrow-splitting allows simultaneous borrows.
            let gwq = self.grads.wq.matrix().as_slice_mut();
            let gwk = self.grads.wk.matrix().as_slice_mut();
            let gwv = self.grads.wv.matrix().as_slice_mut();
            let gwo = self.grads.wo.matrix().as_slice_mut();
            let gwi = self.grads.wi.matrix().as_slice_mut();
            let gwf = self.grads.wf.matrix().as_slice_mut();
            let dq = &*self.dq;
            let dk = &*self.dk_pre;
            let dv = &*self.dv;
            let doo = &*self.do_pre;
            let di = &*self.di_pre;
            let df = &*self.df_pre;
            let x = &*cache.x;
            let dx = &mut *cache.dx;

            for idx in 0..self.input_size {
                let xi = x[idx];
                let mut s = 0.0;
                let qk_off = idx * d_qk;
                let d_off = idx * d;
                let h_off = idx * h;
                for j in 0..d_qk {
                    let dq_j = dq[j];
                    let dk_j = dk[j];
                    gwq[qk_off + j] += xi * dq_j;
                    gwk[qk_off + j] += xi * dk_j;
                    s += wq[qk_off + j] * dq_j + wk[qk_off + j] * dk_j;
                }
                for j in 0..d {
                    let dv_j = dv[j];
                    let do_j = doo[j];
                    gwv[d_off + j] += xi * dv_j;
                    gwo[d_off + j] += xi * do_j;
                    s += wv[d_off + j] * dv_j + wo[d_off + j] * do_j;
                }
                for hd2 in 0..h {
                    let di_h = di[hd2];
                    let df_h = df[hd2];
                    gwi[h_off + hd2] += xi * di_h;
                    gwf[h_off + hd2] += xi * df_h;
                    s += wi[h_off + hd2] * di_h + wf[h_off + hd2] * df_h;
                }
                dx[idx] = s;
            }
        }

        // Roll self.c/n backward via O(1) pointer swap — self.c becomes C_{t-1} for next step.
        for hd in 0..h {
            std::mem::swap(&mut self.c[hd], &mut cache.c_prev[hd]);
        }
        std::mem::swap(&mut self.n, &mut cache.n_prev);
    }

    /// Fold a replica's grads into this layer (data-parallel reduction).
    pub fn add_grads(&mut self, other: &mut Self) {
        let g = &mut self.grads;
        let o = &mut other.grads;
        add_grad_matrix(&mut g.wq, &mut o.wq);
        add_grad_matrix(&mut g.wk, &mut o.wk);
        add_grad_matrix(&mut g.wv, &mut o.wv);
        add_grad_matrix(&mut g.wo, &mut o.wo);
        add_grad_matrix(&mut g.wi, &mut o.wi);
        add_grad_matrix(&mut g.wf, &mut o.wf);
        add_grad_vec(&mut g.bq, &mut o.bq);
        add_grad_vec(&mut g.bk, &mut o.bk);
        add_grad_vec(&mut g.bv, &mut o.bv);
        add_grad_vec(&mut g.bo, &mut o.bo);
        add_grad_vec(&mut g.bi, &mut o.bi);
        add_grad_vec(&mut g.bf, &mut o.bf);
        self.w_out.add_grads(&mut other.w_out);
        self.head_norm.add_grads(&mut other.head_norm);
    }

    /// Overwrite all weights with `other`'s (in-place replica refresh).
    pub fn copy_weights(&mut self, other: &Self) {
        self.wq.copy_from(&other.wq);
        self.wk.copy_from(&other.wk);
        self.wv.copy_from(&other.wv);
        self.wo.copy_from(&other.wo);
        self.wi.copy_from(&other.wi);
        self.wf.copy_from(&other.wf);
        self.bq.copy_from_slice(&other.bq);
        self.bk.copy_from_slice(&other.bk);
        self.bv.copy_from_slice(&other.bv);
        self.bo.copy_from_slice(&other.bo);
        self.bi.copy_from_slice(&other.bi);
        self.bf.copy_from_slice(&other.bf);
        self.w_out.copy_weights(&other.w_out);
        self.head_norm.copy_weights(&other.head_norm);
    }

    pub fn alloc_cache(&self) -> MLSTMCache {
        let d = self.hidden_size;
        let h = self.num_heads;
        let dqk = self.dqk;
        let dhv = self.dhv;
        MLSTMCache {
            x: vec![0.0; self.input_size].into(),
            c_prev: (0..h).map(|_| Matrix::zeros(dhv, dqk)).collect(),
            n_prev: Matrix::zeros(h, dqk),
            q: vec![0.0; h * dqk].into(),
            k: vec![0.0; h * dqk].into(),
            v: vec![0.0; d].into(),
            o: vec![0.0; d].into(),
            i_pre: vec![0.0; h].into(),
            f_pre: vec![0.0; h].into(),
            log_f: vec![0.0; h].into(),
            i_prime: vec![0.0; h].into(),
            f_prime: vec![0.0; h].into(),
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
        self.grads.wo.clip();
        self.grads.wi.clip();

        self.grads.bo.clip();
        self.grads.bi.clip();

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

    fn add_grads_from(&mut self, other: &mut dyn NnLayer) {
        let o = other
            .as_any_mut()
            .downcast_mut::<Self>()
            .expect("MLSTMLayer::add_grads_from — replica layer type mismatch");
        self.add_grads(o);
    }

    fn copy_weights_from(&mut self, other: &dyn NnLayer) {
        let o = other
            .as_any()
            .downcast_ref::<Self>()
            .expect("MLSTMLayer::copy_weights_from — replica layer type mismatch");
        self.copy_weights(o);
    }

    fn reset_state(&mut self) {
        for mat in self.c.iter_mut() {
            mat.clear();
        }
        self.n.clear();
        self.m.fill(0.0);
    }

    fn reset_bptt_state(&mut self) {
        for mat in self.dc_bptt.iter_mut() {
            mat.clear();
        }
        self.dn_bptt.clear();
    }

    fn state_size(&self) -> usize {
        self.num_heads * self.dhv * self.dqk + self.num_heads * self.dqk
    }

    fn inject_state(&mut self, buf: &[f32], offset: usize) -> usize {
        let c_size = self.num_heads * self.dhv * self.dqk;
        let n_size = self.num_heads * self.dqk;
        let head_size = self.dhv * self.dqk;
        let mut off = offset;
        for hd in 0..self.num_heads {
            self.c[hd]
                .as_slice_mut()
                .copy_from_slice(&buf[off..off + head_size]);
            off += head_size;
        }
        self.n
            .as_slice_mut()
            .copy_from_slice(&buf[offset + c_size..offset + c_size + n_size]);
        self.m.fill(0.0);
        offset + c_size + n_size
    }

    fn collect_bptt_grad(&mut self, buf: &mut [f32], offset: usize) -> usize {
        let c_size = self.num_heads * self.dhv * self.dqk;
        let n_size = self.num_heads * self.dqk;
        let head_size = self.dhv * self.dqk;
        let mut off = offset;
        for hd in 0..self.num_heads {
            buf[off..off + head_size].copy_from_slice(self.dc_bptt[hd].as_slice());
            off += head_size;
        }
        buf[offset + c_size..offset + c_size + n_size].copy_from_slice(self.dn_bptt.as_slice());
        offset + c_size + n_size
    }
}
