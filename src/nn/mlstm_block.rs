// mlstm_block.rs ── Optimized xLSTM 7B mLSTM Block (Appendix, arXiv:2411.xxxxx)
//
// Block structure (Equations 12a / 12b):
//
//   z = x + mLSTM_layer( Norm(x) )          sequence-mix sub-layer
//   y = z + MLP( Norm(z) )                   channel-mix  sub-layer (SwiGLU)
//
// Compared to the earlier SLSTMBlock:
//   • TWO separate pre-norm + residual connections (Transformer style).
//   • mLSTM operates in the model embedding dimension d (no up-projection).
//   • MULTI-HEAD: H heads, each with asymmetric d_qk ≠ d_hv (d_qk = d_hv/2).
//   • Dense linear projections for Q, K, V (no block-diagonal, no conv).
//   • Scalar input/forget gate computed per head independently from x.
//   • Output projection W_out: H·d_hv → d brings the concat back to d.
//   • No channel-wise convolution, no learnable skip connection.
//
// Dimensions:
//   d_model = H * d_hv          (embedding dim = heads × value-head-dim)
//   d_qk    = d_hv / 2          (query/key head dim, recommended by paper)
//
// Per-head recurrence:
//   C_h ∈ ℝ^{d_hv × d_qk}      matrix cell state  (row = value, col = key)
//   n_h ∈ ℝ^{d_qk}              normaliser vector
//   m_h ∈ ℝ                     stabiliser scalar
//
// TAG_MLSTM_BLOCK = 13

use std::{any::Any, io};

use iron_oxide::collections::Matrix;

use crate::{
    nn::{
        linear::{LinearCache, LinearLayer},
        rms_norm::{RMSNorm, RMSNormCache},
        sub_in_place, sub_vec_in_place,
    },
    nn_layer::{DynCache, NnLayer},
    saving::{write_f32_slice, write_matrix, write_u32},
};

// ── SiLU helpers ──────────────────────────────────────────────────────────────

#[inline]
fn sigmoid(x: f32) -> f32 {
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
        -((-x).exp() + 1.0).ln()
    } else {
        x - (x.exp() + 1.0).ln()
    }
}
#[inline]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
#[inline]
fn silu_prime(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
}
#[inline]
fn dsigmoid(s: f32) -> f32 {
    s * (1.0 - s)
}

const CLIP: f32 = 15.0;

// ═══════════════════════════════════════════════════════════════════════════════
// Part 1 — Multi-head mLSTM sequence-mixing layer
// ═══════════════════════════════════════════════════════════════════════════════

// ── MLSTMSeqCache ─────────────────────────────────────────────────────────────

/// Per-timestep activations for the multi-head mLSTM layer.
/// All per-head tensors are stored concatenated (no nested vecs, no heap in hot path).
pub struct MLSTMSeqCache {
    pub d_model: usize,
    pub num_heads: usize,
    pub d_qk: usize,
    pub d_hv: usize,

    /// Saved input x (d_model) for weight gradients.
    pub x: Box<[f32]>,

    // ── Previous per-head states ─────────────────────────────────────────────
    /// C_{t−1}, row-major per head: [h][i * d_qk + j]  total: H * d_hv * d_qk
    pub c_prev: Box<[f32]>,
    /// n_{t−1}: total H * d_qk
    pub n_prev: Box<[f32]>,
    /// m_{t−1}: total H
    pub m_prev: Box<[f32]>,

    // ── Projections (pre-activations, concatenated across heads) ─────────────
    pub q_all: Box<[f32]>,         // H * d_qk  (W_q x + b_q, no extra act)
    pub k_all: Box<[f32]>,         // H * d_qk  (W_k x + b_k, already scaled /√d_qk)
    pub v_all: Box<[f32]>,         // H * d_hv
    pub o_pre_all: Box<[f32]>,     // H * d_hv  (W_o x + b_o, before sigmoid)
    pub o_all: Box<[f32]>,         // H * d_hv  (σ(o_pre))
    pub i_pre_all: Box<[f32]>,     // H  scalar per head
    pub f_pre_all: Box<[f32]>,     // H  scalar per head
    pub sigmoid_f_all: Box<[f32]>, // H  σ(f̃) per head

    // ── Stabilised gates ─────────────────────────────────────────────────────
    pub m_all: Box<[f32]>,       // H
    pub i_prime_all: Box<[f32]>, // H
    pub f_prime_all: Box<[f32]>, // H

    // ── States at time t ─────────────────────────────────────────────────────
    /// C_t, same layout as c_prev.  H * d_hv * d_qk
    pub c_all: Box<[f32]>,
    /// n_t: H * d_qk
    pub n_all: Box<[f32]>,

    /// r_h = C_h q_h per head, concatenated. H * d_hv
    pub r_all: Box<[f32]>,
    /// ψ_h = max(|n_h^T q_h|, 1) per head. H
    pub psi_all: Box<[f32]>,
    /// h_h = o_h ⊙ (r_h / ψ_h), concatenated. H * d_hv (= d_model)
    pub h_concat: Box<[f32]>,

    // ── Output projection cache ───────────────────────────────────────────────
    pub out_proj: LinearCache, // (H*d_hv → d_model)

    /// dL/dx_t written by backward.
    pub dx: Box<[f32]>, // d_model
}

impl DynCache for MLSTMSeqCache {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn output(&self) -> &[f32] {
        &self.out_proj.output
    }
    fn input_grad(&self) -> &[f32] {
        &self.dx
    }
}

// ── MLSTMSeqGrads ─────────────────────────────────────────────────────────────

pub struct MLSTMSeqGrads {
    pub wq: Matrix, // d_model × (H * d_qk)
    pub wk: Matrix,
    pub wv: Matrix, // d_model × (H * d_hv)
    pub wo: Matrix,
    pub wi: Matrix, // d_model × H  (scalar gate per head)
    pub wf: Matrix,
    pub bq: Box<[f32]>, // H * d_qk
    pub bk: Box<[f32]>,
    pub bv: Box<[f32]>, // H * d_hv
    pub bo: Box<[f32]>,
    pub bi: Box<[f32]>, // H
    pub bf: Box<[f32]>,
}

impl MLSTMSeqGrads {
    pub fn zeros(d_model: usize, num_heads: usize, d_qk: usize, d_hv: usize) -> Self {
        let hq = num_heads * d_qk;
        let hv = num_heads * d_hv;
        Self {
            wq: Matrix::zeros(d_model, hq),
            wk: Matrix::zeros(d_model, hq),
            wv: Matrix::zeros(d_model, hv),
            wo: Matrix::zeros(d_model, hv),
            wi: Matrix::zeros(d_model, num_heads),
            wf: Matrix::zeros(d_model, num_heads),
            bq: vec![0.0; hq].into(),
            bk: vec![0.0; hq].into(),
            bv: vec![0.0; hv].into(),
            bo: vec![0.0; hv].into(),
            bi: vec![0.0; num_heads].into(),
            bf: vec![0.0; num_heads].into(),
        }
    }
}

// ── MLSTMSeqLayer ─────────────────────────────────────────────────────────────

/// Multi-head mLSTM sequence-mixing layer (no norms, no residual, no MLP).
/// Projects x → Q, K, V, O_gate, i, f; runs H mLSTM cells; projects back.
pub struct MLSTMSeqLayer {
    pub d_model: usize,
    pub num_heads: usize,
    pub d_qk: usize, // query/key head dim
    pub d_hv: usize, // value/hidden head dim;  d_model = num_heads * d_hv

    // ── Input projections (d_model × total_dim) ───────────────────────────────
    pub wq: Matrix, // d_model × (H * d_qk)
    pub wk: Matrix,
    pub wv: Matrix, // d_model × (H * d_hv)
    pub wo: Matrix,
    pub wi: Matrix, // d_model × H  (scalar input gate per head)
    pub wf: Matrix, // d_model × H

    // ── Biases ────────────────────────────────────────────────────────────────
    pub bq: Box<[f32]>, // H * d_qk
    pub bk: Box<[f32]>,
    pub bv: Box<[f32]>, // H * d_hv
    pub bo: Box<[f32]>,
    pub bi: Box<[f32]>, // H  (forget bias init > 0)
    pub bf: Box<[f32]>, // H

    // ── Output projection (H*d_hv → d_model) ─────────────────────────────────
    pub w_out: LinearLayer,

    // ── Recurrent state (all heads bundled) ───────────────────────────────────
    /// C_t: H * d_hv * d_qk  (row = value dim, col = key dim per head)
    pub c: Box<[f32]>,
    /// n_t: H * d_qk
    pub n: Box<[f32]>,
    /// m_t: H (one scalar stabiliser per head)
    pub m: Box<[f32]>,

    // ── BPTT gradients ────────────────────────────────────────────────────────
    pub dc_bptt: Box<[f32]>, // H * d_hv * d_qk
    pub dn_bptt: Box<[f32]>, // H * d_qk

    pub grads: MLSTMSeqGrads,

    // ── Backward scratch (pre-allocated) ─────────────────────────────────────
    /// Concat of per-head dq gradients  (H * d_qk)
    pub dq_all: Box<[f32]>,
    /// Concat of per-head dk gradients  (H * d_qk)
    pub dk_all: Box<[f32]>,
    /// Concat of per-head dv gradients  (H * d_hv)
    pub dv_all: Box<[f32]>,
    /// Concat of per-head do_pre grad   (H * d_hv)
    pub do_pre_all: Box<[f32]>,
    /// Scalar gate gradients per head   (H)
    pub di_pre_all: Box<[f32]>,
    pub df_pre_all: Box<[f32]>,
    /// Gradient of h_concat before output projection  (H * d_hv)
    pub dh_concat: Box<[f32]>,
}

impl MLSTMSeqLayer {
    pub fn new(d_model: usize, num_heads: usize, d_qk: usize, d_hv: usize) -> Self {
        assert_eq!(
            d_model,
            num_heads * d_hv,
            "d_model must equal num_heads * d_hv"
        );
        assert_eq!(d_qk, d_hv / 2, "d_qk must equal d_hv / 2 (paper §3.1)");

        let hq = num_heads * d_qk;
        let hv = num_heads * d_hv; // = d_model
        let c_size = num_heads * d_hv * d_qk;

        let scale_proj = (6.0 / (d_model + d_qk.max(d_hv)) as f32).sqrt();
        let scale_gate = (6.0 / (d_model + 1) as f32).sqrt();

        // forget-gate biases start positive (Jozefowicz et al. 2015).
        let bf: Box<[f32]> = vec![3.0; num_heads].into();
        let bi: Box<[f32]> = vec![0.0; num_heads].into();

        Self {
            d_model,
            num_heads,
            d_qk,
            d_hv,
            wq: Matrix::random(d_model, hq, scale_proj),
            wk: Matrix::random(d_model, hq, scale_proj),
            wv: Matrix::random(d_model, hv, scale_proj),
            wo: Matrix::random(d_model, hv, scale_proj),
            wi: Matrix::random(d_model, num_heads, scale_gate),
            wf: Matrix::random(d_model, num_heads, scale_gate),
            bq: vec![0.0; hq].into(),
            bk: vec![0.0; hq].into(),
            bv: vec![0.0; hv].into(),
            bo: vec![0.0; hv].into(),
            bi,
            bf,
            w_out: LinearLayer::new(hv, d_model),
            c: vec![0.0; c_size].into(),
            n: vec![0.0; hq].into(),
            m: vec![0.0; num_heads].into(),
            dc_bptt: vec![0.0; c_size].into(),
            dn_bptt: vec![0.0; hq].into(),
            grads: MLSTMSeqGrads::zeros(d_model, num_heads, d_qk, d_hv),
            dq_all: vec![0.0; hq].into(),
            dk_all: vec![0.0; hq].into(),
            dv_all: vec![0.0; hv].into(),
            do_pre_all: vec![0.0; hv].into(),
            di_pre_all: vec![0.0; num_heads].into(),
            df_pre_all: vec![0.0; num_heads].into(),
            dh_concat: vec![0.0; hv].into(),
        }
    }

    pub fn alloc_cache(&self) -> MLSTMSeqCache {
        let d = self.d_model;
        let hq = self.num_heads * self.d_qk;
        let hv = self.num_heads * self.d_hv;
        let h = self.num_heads;
        let cs = self.num_heads * self.d_hv * self.d_qk;

        MLSTMSeqCache {
            d_model: d,
            num_heads: h,
            d_qk: self.d_qk,
            d_hv: self.d_hv,
            x: vec![0.0; d].into(),
            c_prev: vec![0.0; cs].into(),
            n_prev: vec![0.0; hq].into(),
            m_prev: vec![0.0; h].into(),
            q_all: vec![0.0; hq].into(),
            k_all: vec![0.0; hq].into(),
            v_all: vec![0.0; hv].into(),
            o_pre_all: vec![0.0; hv].into(),
            o_all: vec![0.0; hv].into(),
            i_pre_all: vec![0.0; h].into(),
            f_pre_all: vec![0.0; h].into(),
            sigmoid_f_all: vec![0.5; h].into(),
            m_all: vec![0.0; h].into(),
            i_prime_all: vec![0.0; h].into(),
            f_prime_all: vec![0.0; h].into(),
            c_all: vec![0.0; cs].into(),
            n_all: vec![0.0; hq].into(),
            r_all: vec![0.0; hv].into(),
            psi_all: vec![1.0; h].into(),
            h_concat: vec![0.0; hv].into(),
            out_proj: LinearCache::new(hv, d),
            dx: vec![0.0; d].into(),
        }
    }

    // ── Forward ───────────────────────────────────────────────────────────────

    pub fn forward(&mut self, x: &[f32], cache: &mut MLSTMSeqCache) {
        let h = self.num_heads;
        let dqk = self.d_qk;
        let dhv = self.d_hv;
        let k_scale = 1.0 / (dqk as f32).sqrt();
        let c_head = dhv * dqk; // elements per head in c

        cache.x.copy_from_slice(x);
        cache.c_prev.copy_from_slice(&self.c);
        cache.n_prev.copy_from_slice(&self.n);
        cache.m_prev.copy_from_slice(&self.m);

        // ── Project x ────────────────────────────────────────────────────────
        self.wq.row_mul(x, &mut cache.q_all);
        self.wk.row_mul(x, &mut cache.k_all);
        self.wv.row_mul(x, &mut cache.v_all);
        self.wo.row_mul(x, &mut cache.o_pre_all);
        self.wi.row_mul(x, &mut cache.i_pre_all);
        self.wf.row_mul(x, &mut cache.f_pre_all);

        // Add biases and apply activations
        for j in 0..h * dqk {
            cache.q_all[j] += self.bq[j];
            cache.k_all[j] = (cache.k_all[j] + self.bk[j]) * k_scale;
        }
        for j in 0..h * dhv {
            cache.v_all[j] += self.bv[j];
            cache.o_pre_all[j] += self.bo[j];
            cache.o_all[j] = sigmoid(cache.o_pre_all[j]);
        }
        for hd in 0..h {
            let i_pre = (cache.i_pre_all[hd] + self.bi[hd]).clamp(-CLIP, CLIP);
            let f_pre = (cache.f_pre_all[hd] + self.bf[hd]).clamp(-CLIP, CLIP);
            cache.i_pre_all[hd] = i_pre;
            cache.f_pre_all[hd] = f_pre;
            cache.sigmoid_f_all[hd] = sigmoid(f_pre);

            let log_f = log_sigmoid(f_pre);
            let m_new = (log_f + self.m[hd]).max(i_pre);
            let i_prime = (i_pre - m_new).exp();
            let f_prime = (log_f + self.m[hd] - m_new).exp();

            cache.m_all[hd] = m_new;
            cache.i_prime_all[hd] = i_prime;
            cache.f_prime_all[hd] = f_prime;

            // ── Per-head cell + normaliser update ─────────────────────────
            // k_h, q_h offsets: hd * dqk
            // v_h, o_h offsets: hd * dhv
            // C_h offset in flat array: hd * c_head  (dhv rows × dqk cols, row-major)
            let co = hd * c_head; // c offset for this head
            let no = hd * dqk; // n offset
            let vo = hd * dhv; // v/o offset
            let qo = hd * dqk; // q/k offset

            // C_t = f′ C_{t-1} + i′ (v ⊗ k)
            for i in 0..dhv {
                for j in 0..dqk {
                    self.c[co + i * dqk + j] = f_prime * cache.c_prev[co + i * dqk + j]
                        + i_prime * cache.v_all[vo + i] * cache.k_all[qo + j];
                }
            }

            // n_t = f′ n_{t-1} + i′ k_t
            for j in 0..dqk {
                self.n[no + j] = f_prime * cache.n_prev[no + j] + i_prime * cache.k_all[qo + j];
            }
        }

        // Save updated states to cache
        cache.c_all.copy_from_slice(&self.c);
        cache.n_all.copy_from_slice(&self.n);
        self.m.copy_from_slice(&cache.m_all);

        // ── r_h = C_h q_h, ψ_h, h_h = o_h ⊙ (r_h / ψ_h) ─────────────────
        for hd in 0..h {
            let co = hd * c_head;
            let qo = hd * dqk;
            let vo = hd * dhv;

            // r_h[i] = Σ_j C_h[i, j] * q_h[j]
            for i in 0..dhv {
                let mut s = 0.0_f32;
                for j in 0..dqk {
                    s += self.c[co + i * dqk + j] * cache.q_all[qo + j];
                }
                cache.r_all[vo + i] = s;
            }

            // ψ_h = max(|n_h^T q_h|, 1)
            let ntq: f32 = (0..dqk).map(|j| self.n[qo + j] * cache.q_all[qo + j]).sum();
            let psi = ntq.abs().max(1.0);
            cache.psi_all[hd] = psi;

            // h_h = o_h ⊙ (r_h / ψ_h)
            for i in 0..dhv {
                cache.h_concat[vo + i] = cache.o_all[vo + i] * cache.r_all[vo + i] / psi;
            }
        }

        // ── Output projection: out = W_out h_concat + b_out ──────────────────
        self.w_out.forward(&cache.h_concat, &mut cache.out_proj);
    }

    // ── Backward ──────────────────────────────────────────────────────────────
    //
    // `delta` = dL/d(layer output) = dL/d(W_out h_concat + b_out)  (d_model).
    // Writes `cache.dx` = dL/dx  and updates dc_bptt, dn_bptt for t−1.

    pub fn backward(&mut self, delta: &mut [f32], cache: &mut MLSTMSeqCache) {
        let h = self.num_heads;
        let dqk = self.d_qk;
        let dhv = self.d_hv;
        let c_head = dhv * dqk;
        let k_scale = 1.0 / (dqk as f32).sqrt();

        // ── 1. Backward through output projection ─────────────────────────────
        // delta = dL/dy (d_model).
        // After backward: w_out.cache.dx = dL/d(h_concat)  (H * dhv)
        self.w_out.backward(delta, &mut cache.out_proj);
        self.dh_concat.copy_from_slice(&cache.out_proj.dx);

        // ── 2. Per-head backward ──────────────────────────────────────────────
        self.di_pre_all.fill(0.0);
        self.df_pre_all.fill(0.0);

        for hd in 0..h {
            let co = hd * c_head;
            let qo = hd * dqk;
            let vo = hd * dhv;

            let i_prime = cache.i_prime_all[hd];
            let f_prime = cache.f_prime_all[hd];
            let psi = cache.psi_all[hd];

            // (A) Gradient of o_h and r_h from h_h = o_h ⊙ (r_h / ψ)
            for i in 0..dhv {
                let dh_i = self.dh_concat[vo + i];
                self.do_pre_all[vo + i] =
                    dh_i * (cache.r_all[vo + i] / psi) * dsigmoid(cache.o_all[vo + i]);
                // dr[i] temporarily stored in dv_all (re-used as scratch):
                self.dv_all[vo + i] = dh_i * cache.o_all[vo + i] / psi; // dr
            }

            // (B) Accumulate dC from r = C q  into dc_bptt[co..co+c_head]:
            //     dC_total[i,j] = dr[i] * q[j]   (+= because dc_bptt already has t+1 grad)
            for i in 0..dhv {
                let dr_i = self.dv_all[vo + i]; // dr stored here
                for j in 0..dqk {
                    self.dc_bptt[co + i * dqk + j] += dr_i * cache.q_all[qo + j];
                }
            }
            // dc_bptt[co..] now holds total dC_t for head hd.

            // (B) dq from r = C q:  dq[j] = Σ_i C_h[i,j] * dr[i]  (C^T dr)
            //     Use dk_all as temporary for dq (will be filled properly below).
            for j in 0..dqk {
                let mut s = 0.0_f32;
                for i in 0..dhv {
                    s += cache.c_all[co + i * dqk + j] * self.dv_all[vo + i]; // dv_all=dr
                }
                self.dk_all[qo + j] = s; // real dq → dk_all temporarily
            }
            // Copy real dq into dq_all, then we'll compute real dk below.
            self.dq_all[qo..qo + dqk].copy_from_slice(&self.dk_all[qo..qo + dqk]);

            // (C) Gradients from C_t and n_t updates:
            //     dc_bptt[co..] = dC_total; dn_bptt[no..] = dn_total
            let mut d_f_prime = 0.0_f32;
            let mut d_i_prime = 0.0_f32;
            // Reset dv and dk for real accumulation.
            self.dv_all[vo..vo + dhv].fill(0.0);
            self.dk_all[qo..qo + dqk].fill(0.0);

            for i in 0..dhv {
                let v_i = cache.v_all[vo + i];
                let mut dv_i = 0.0_f32;
                for j in 0..dqk {
                    let dc = self.dc_bptt[co + i * dqk + j];
                    let k_j = cache.k_all[qo + j];
                    d_f_prime += dc * cache.c_prev[co + i * dqk + j];
                    d_i_prime += dc * v_i * k_j;
                    dv_i += i_prime * dc * k_j;
                    self.dk_all[qo + j] += i_prime * dc * v_i;
                }
                self.dv_all[vo + i] = dv_i;
            }
            // Contribution from normaliser n_t:
            for j in 0..dqk {
                let dn = self.dn_bptt[qo + j];
                d_f_prime += dn * cache.n_prev[qo + j];
                d_i_prime += dn * cache.k_all[qo + j];
                self.dk_all[qo + j] += i_prime * dn;
            }

            // (D) Propagate BPTT state to t−1 (in-place):
            for ij in co..co + c_head {
                self.dc_bptt[ij] *= f_prime;
            }
            for j in 0..dqk {
                self.dn_bptt[qo + j] *= f_prime;
            }

            // (E) Stabilised-gate pre-activation gradients:
            let d_it_pre = d_i_prime * i_prime;
            let d_ft_pre = d_f_prime * f_prime * (1.0 - cache.sigmoid_f_all[hd]);
            self.di_pre_all[hd] = d_it_pre;
            self.df_pre_all[hd] = d_ft_pre;

            // (F) Key scaling: dk → d(kt_pre) = dk * (1/√d_qk)
            for j in 0..dqk {
                self.dk_all[qo + j] *= k_scale;
            }
        }
        // All per-head gradients are now in:
        //   dq_all (H*d_qk), dk_all (H*d_qk), dv_all (H*d_hv),
        //   do_pre_all (H*d_hv), di_pre_all (H), df_pre_all (H).

        // ── 3. Weight gradients via add_outer(x, d_pre-act) ───────────────────
        let x = &cache.x;
        self.grads.wq.add_outer(x, &self.dq_all);
        self.grads.wk.add_outer(x, &self.dk_all);
        self.grads.wv.add_outer(x, &self.dv_all);
        self.grads.wo.add_outer(x, &self.do_pre_all);
        self.grads.wi.add_outer(x, &self.di_pre_all);
        self.grads.wf.add_outer(x, &self.df_pre_all);

        for j in 0..h * dqk {
            self.grads.bq[j] += self.dq_all[j];
            self.grads.bk[j] += self.dk_all[j];
        }
        for j in 0..h * dhv {
            self.grads.bv[j] += self.dv_all[j];
            self.grads.bo[j] += self.do_pre_all[j];
        }
        for hd in 0..h {
            self.grads.bi[hd] += self.di_pre_all[hd];
            self.grads.bf[hd] += self.df_pre_all[hd];
        }

        // ── 4. Input gradient dx ──────────────────────────────────────────────
        // dx[i] = Σ_j (wq[i,j]*dq[j] + wk[i,j]*dk[j] + wv[i,j]*dv[j]
        //              + wo[i,j]*do_pre[j] + wi[i,j]*di[j] + wf[i,j]*df[j])
        let d = self.d_model;
        let hq = h * dqk;
        let hv = h * dhv;
        for i in 0..d {
            let mut s = 0.0_f32;
            for j in 0..hq {
                s += self.wq[i][j] * self.dq_all[j] + self.wk[i][j] * self.dk_all[j];
            }
            for j in 0..hv {
                s += self.wv[i][j] * self.dv_all[j] + self.wo[i][j] * self.do_pre_all[j];
            }
            for hd in 0..h {
                s += self.wi[i][hd] * self.di_pre_all[hd] + self.wf[i][hd] * self.df_pre_all[hd];
            }
            cache.dx[i] = s;
        }
    }

    // ── Weight management ─────────────────────────────────────────────────────

    pub fn apply_grads(&mut self, lr: f32) {
        self.grads.wi.clip(-CLIP, CLIP);
        self.grads.wf.clip(-CLIP, CLIP);
        sub_in_place(&mut self.wq, &self.grads.wq, lr);
        sub_in_place(&mut self.wk, &self.grads.wk, lr);
        sub_in_place(&mut self.wv, &self.grads.wv, lr);
        sub_in_place(&mut self.wo, &self.grads.wo, lr);
        sub_in_place(&mut self.wi, &self.grads.wi, lr);
        sub_in_place(&mut self.wf, &self.grads.wf, lr);
        sub_vec_in_place(&mut self.bq, &self.grads.bq, lr);
        sub_vec_in_place(&mut self.bk, &self.grads.bk, lr);
        sub_vec_in_place(&mut self.bv, &self.grads.bv, lr);
        sub_vec_in_place(&mut self.bo, &self.grads.bo, lr);
        sub_vec_in_place(&mut self.bi, &self.grads.bi, lr);
        sub_vec_in_place(&mut self.bf, &self.grads.bf, lr);
        self.w_out.apply_grads(lr);
    }

    pub fn clear_grads(&mut self) {
        self.grads.wq.clear();
        self.grads.wk.clear();
        self.grads.wv.clear();
        self.grads.wo.clear();
        self.grads.wi.clear();
        self.grads.wf.clear();
        self.grads.bq.fill(0.0);
        self.grads.bk.fill(0.0);
        self.grads.bv.fill(0.0);
        self.grads.bo.fill(0.0);
        self.grads.bi.fill(0.0);
        self.grads.bf.fill(0.0);
        self.w_out.clear_grads();
    }

    pub fn scale_grads(&mut self, scale: f32) {
        self.grads.wq.scale(scale);
        self.grads.wk.scale(scale);
        self.grads.wv.scale(scale);
        self.grads.wo.scale(scale);
        self.grads.wi.scale(scale);
        self.grads.wf.scale(scale);
        self.grads.bq.iter_mut().for_each(|g| *g *= scale);
        self.grads.bk.iter_mut().for_each(|g| *g *= scale);
        self.grads.bv.iter_mut().for_each(|g| *g *= scale);
        self.grads.bo.iter_mut().for_each(|g| *g *= scale);
        self.grads.bi.iter_mut().for_each(|g| *g *= scale);
        self.grads.bf.iter_mut().for_each(|g| *g *= scale);
        self.w_out.scale_grads(scale);
    }

    pub fn reset_state(&mut self) {
        self.c.fill(0.0);
        self.n.fill(0.0);
        self.m.fill(0.0);
    }

    pub fn zero_bptt_state(&mut self) {
        self.dc_bptt.fill(0.0);
        self.dn_bptt.fill(0.0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Part 2 — Full mLSTM Block  (Eq. 12a + 12b)
// ═══════════════════════════════════════════════════════════════════════════════

// ── MLSTMBlockCache ───────────────────────────────────────────────────────────

pub struct MLSTMBlockCache {
    // ── Sub-layer 1: Pre-norm₁ + mLSTM + residual ────────────────────────────
    pub norm1: RMSNormCache,  // input → x_norm
    pub mlstm: MLSTMSeqCache, // x_norm → mlstm_out  (d_model)

    // ── Intermediate: z = x + mlstm_out ──────────────────────────────────────
    pub z: Box<[f32]>, // d_model

    // ── Sub-layer 2: Pre-norm₂ + SwiGLU + residual ───────────────────────────
    pub norm2: RMSNormCache,    // z → z_norm
    pub lin_gate: LinearCache,  // z_norm → gate_pre   (up_size)
    pub gate_act: Box<[f32]>,   // SiLU(gate_pre)       (up_size)
    pub lin_value: LinearCache, // z_norm → value       (up_size)
    pub mixed: Box<[f32]>,      // gate_act ⊙ value      (up_size)
    pub lin_down: LinearCache,  // mixed   → mlp_out    (d_model)

    pub output: Box<[f32]>, // y = z + mlp_out       (d_model)
    pub dx: Box<[f32]>,     // dL/d(input)           (d_model)

    // Scratch buffers for backward (pre-allocated)
    pub sc1: Box<[f32]>, // d_model
    pub sc2: Box<[f32]>, // up_size
    pub sc3: Box<[f32]>, // up_size
}

impl DynCache for MLSTMBlockCache {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn output(&self) -> &[f32] {
        &self.output
    }
    fn input_grad(&self) -> &[f32] {
        &self.dx
    }
}

// ── MLSTMBlock ────────────────────────────────────────────────────────────────

pub struct MLSTMBlock {
    pub d_model: usize,
    pub num_heads: usize,
    pub d_qk: usize,
    pub d_hv: usize,
    pub up_size: usize,

    // Sub-layer 1
    pub norm1: RMSNorm,
    pub mlstm: MLSTMSeqLayer,

    // Sub-layer 2
    pub norm2: RMSNorm,
    pub lin_gate: LinearLayer,  // d_model → up_size
    pub lin_value: LinearLayer, // d_model → up_size
    pub lin_down: LinearLayer,  // up_size → d_model

    // Backward scratch
    pub sc_d: Box<[f32]>,  // d_model
    pub sc_u1: Box<[f32]>, // up_size
    pub sc_u2: Box<[f32]>, // up_size
}

impl MLSTMBlock {
    /// Construct a block.
    /// `up_size`: SwiGLU hidden width (typically ≈ 8/3 * d_model, rounded to multiple of 256).
    pub fn new(d_model: usize, num_heads: usize, d_qk: usize, d_hv: usize, up_size: usize) -> Self {
        let d = d_model;
        let u = up_size;
        Self {
            d_model: d,
            num_heads,
            d_qk,
            d_hv,
            up_size: u,
            norm1: RMSNorm::new(d),
            mlstm: MLSTMSeqLayer::new(d, num_heads, d_qk, d_hv),
            norm2: RMSNorm::new(d),
            lin_gate: LinearLayer::new(d, u),
            lin_value: LinearLayer::new(d, u),
            lin_down: LinearLayer::new(u, d),
            sc_d: vec![0.0; d].into(),
            sc_u1: vec![0.0; u].into(),
            sc_u2: vec![0.0; u].into(),
        }
    }

    pub fn alloc_cache(&self) -> MLSTMBlockCache {
        let d = self.d_model;
        let u = self.up_size;
        MLSTMBlockCache {
            norm1: self.norm1.alloc_cache(),
            mlstm: self.mlstm.alloc_cache(),
            z: vec![0.0; d].into(),
            norm2: self.norm2.alloc_cache(),
            lin_gate: LinearCache::new(d, u),
            gate_act: vec![0.0; u].into(),
            lin_value: LinearCache::new(d, u),
            mixed: vec![0.0; u].into(),
            lin_down: LinearCache::new(u, d),
            output: vec![0.0; d].into(),
            dx: vec![0.0; d].into(),
            sc1: vec![0.0; d].into(),
            sc2: vec![0.0; u].into(),
            sc3: vec![0.0; u].into(),
        }
    }

    // ── Forward ───────────────────────────────────────────────────────────────
    //
    // z = x + mLSTM(Norm₁(x))             Eq. 12a
    // y = z + SwiGLU(Norm₂(z))            Eq. 12b

    pub fn forward(&mut self, x: &[f32], cache: &mut MLSTMBlockCache) {
        let d = self.d_model;
        let u = self.up_size;

        // ── Sub-layer 1: sequence mix ─────────────────────────────────────────
        self.norm1.forward_into(x, &mut cache.norm1);
        self.mlstm.forward(&cache.norm1.output, &mut cache.mlstm);

        // z = x + mlstm_out
        for i in 0..d {
            cache.z[i] = x[i] + cache.mlstm.out_proj.output[i];
        }

        // ── Sub-layer 2: channel mix (SwiGLU) ─────────────────────────────────
        self.norm2.forward_into(&cache.z, &mut cache.norm2);

        self.lin_gate
            .forward(&cache.norm2.output, &mut cache.lin_gate);
        self.lin_value
            .forward(&cache.norm2.output, &mut cache.lin_value);

        for j in 0..u {
            cache.gate_act[j] = silu(cache.lin_gate.output[j]);
            cache.mixed[j] = cache.gate_act[j] * cache.lin_value.output[j];
        }
        self.lin_down.forward(&cache.mixed, &mut cache.lin_down);

        // y = z + mlp_out
        for i in 0..d {
            cache.output[i] = cache.z[i] + cache.lin_down.output[i];
        }
    }

    // ── Backward ──────────────────────────────────────────────────────────────
    //
    // `delta` = dL/dy.
    // Writes cache.dx = dL/dx.
    //
    // Backward through y = z + MLP(Norm₂(z)):
    //   dL/dz   = delta  (identity branch)
    //   dMLP/dz = backward through MLP and norm₂
    //   total dL/dz = delta + dMLP/dz
    //
    // Backward through z = x + mLSTM(Norm₁(x)):
    //   dL/dx   = dL/dz  (identity branch)
    //   dmLSTM/dx = backward through mLSTM and norm₁
    //   total dL/dx = dL/dz + dmLSTM/dx

    pub fn backward(&mut self, delta: &mut [f32], cache: &mut MLSTMBlockCache) {
        let d = self.d_model;
        let u = self.up_size;

        // (a) Residual save: cache.dx = delta  (identity branch of y = z + …)
        cache.dx.copy_from_slice(delta);

        // (b) SwiGLU backward
        //     delta → lin_down → mixed → gate/value → norm2 → dz_from_mlp

        // lin_down backward (delta is dL/d(mlp_out) = delta)
        self.lin_down.backward(delta, &mut cache.lin_down);
        // cache.lin_down.dx = dL/d(mixed)

        // Element-wise split: mixed = gate_act ⊙ value
        for j in 0..u {
            self.sc_u1[j] = cache.lin_down.dx[j] * cache.lin_value.output[j]; // d(gate_act)
            self.sc_u2[j] = cache.lin_down.dx[j] * cache.gate_act[j]; // d(value)
        }

        // lin_value backward
        self.lin_value
            .backward(&mut self.sc_u2, &mut cache.lin_value);
        // cache.lin_value.dx = dL/d(z_norm) from value branch

        // SiLU backward: d(gate_pre) = d(gate_act) · SiLU'(gate_pre)
        for j in 0..u {
            self.sc_u1[j] *= silu_prime(cache.lin_gate.output[j]);
        }
        // lin_gate backward
        self.lin_gate.backward(&mut self.sc_u1, &mut cache.lin_gate);
        // cache.lin_gate.dx = dL/d(z_norm) from gate branch

        // d(z_norm) = sum of both branches
        for i in 0..d {
            self.sc_d[i] = cache.lin_value.dx[i] + cache.lin_gate.dx[i];
        }

        // norm2 backward: d(z_norm) → d(z)
        self.norm2.backward_into(&self.sc_d, &mut cache.norm2);
        // cache.norm2.dx = dL/d(z) from MLP branch

        // (c) Total dL/dz = delta (from residual save) + d(z from MLP)
        //     We use cache.dx as d(z) accumulator (it already holds delta).
        for i in 0..d {
            cache.dx[i] += cache.norm2.dx[i];
        }
        // cache.dx now holds total dL/dz.

        // (d) mLSTM backward:
        //     d(mlstm_out) = total dL/dz  (identity branch is already stored)
        //     Use sc_d as the mutable delta for the mLSTM backward.
        self.sc_d.copy_from_slice(&cache.dx);
        self.mlstm.backward(&mut self.sc_d, &mut cache.mlstm);
        // cache.mlstm.dx = dL/d(x_norm)

        // norm1 backward: d(x_norm) → d(x)
        self.norm1.backward_into(&cache.mlstm.dx, &mut cache.norm1);
        // cache.norm1.dx = dL/dx from mLSTM branch

        // (e) Total dL/dx = dL/dz (identity branch of z = x + …) + d(x from mLSTM)
        for i in 0..d {
            cache.dx[i] += cache.norm1.dx[i];
        }
        // cache.dx now holds the final dL/dx.
    }
}

// ── impl NnLayer for MLSTMBlock ───────────────────────────────────────────────

impl NnLayer for MLSTMBlock {
    fn forward(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<MLSTMBlockCache>()
            .expect("MLSTMBlock::forward — expected MLSTMBlockCache");
        MLSTMBlock::forward(self, input, c);
    }

    fn backward(&mut self, delta: &mut [f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<MLSTMBlockCache>()
            .expect("MLSTMBlock::backward — expected MLSTMBlockCache");
        MLSTMBlock::backward(self, delta, c);
    }

    fn layer_tag(&self) -> u8 {
        14 // TAG_MLSTM_BLOCK
    }

    /// Binary layout:
    ///   num_heads: u32,  d_qk: u32,  d_hv: u32,  up_size: u32
    ///   norm1.gamma, norm2.gamma : f32_slice(d_model)
    ///   mlstm: wq, wk, wv, wo, wi, wf (Matrix each)
    ///          bq, bk, bv, bo, bi, bf (f32_slice each)
    ///          w_out.weights, w_out.biases
    ///   lin_gate:  weights, biases
    ///   lin_value: weights, biases
    ///   lin_down:  weights, biases
    fn save(&self, w: &mut dyn io::Write) -> io::Result<()> {
        let ml = &self.mlstm;
        write_u32(w, self.num_heads as u32)?;
        write_u32(w, self.d_qk as u32)?;
        write_u32(w, self.d_hv as u32)?;
        write_u32(w, self.up_size as u32)?;
        write_f32_slice(w, &self.norm1.gamma)?;
        write_f32_slice(w, &self.norm2.gamma)?;
        write_matrix(w, &ml.wq)?;
        write_matrix(w, &ml.wk)?;
        write_matrix(w, &ml.wv)?;
        write_matrix(w, &ml.wo)?;
        write_matrix(w, &ml.wi)?;
        write_matrix(w, &ml.wf)?;
        write_f32_slice(w, &ml.bq)?;
        write_f32_slice(w, &ml.bk)?;
        write_f32_slice(w, &ml.bv)?;
        write_f32_slice(w, &ml.bo)?;
        write_f32_slice(w, &ml.bi)?;
        write_f32_slice(w, &ml.bf)?;
        write_matrix(w, &ml.w_out.weights)?;
        write_f32_slice(w, &ml.w_out.biases)?;
        write_matrix(w, &self.lin_gate.weights)?;
        write_f32_slice(w, &self.lin_gate.biases)?;
        write_matrix(w, &self.lin_value.weights)?;
        write_f32_slice(w, &self.lin_value.biases)?;
        write_matrix(w, &self.lin_down.weights)?;
        write_f32_slice(w, &self.lin_down.biases)
    }

    fn make_cache(&self) -> Box<dyn DynCache> {
        Box::new(self.alloc_cache())
    }

    fn input_size(&self) -> usize {
        self.d_model
    }
    fn output_size(&self) -> usize {
        self.d_model
    }

    fn apply_grads(&mut self, lr: f32) {
        sub_vec_in_place(&mut self.norm1.gamma, &self.norm1.grads_gamma, lr);
        sub_vec_in_place(&mut self.norm2.gamma, &self.norm2.grads_gamma, lr);
        self.mlstm.apply_grads(lr);
        self.lin_gate.apply_grads(lr);
        self.lin_value.apply_grads(lr);
        self.lin_down.apply_grads(lr);
    }

    fn clear_grads(&mut self) {
        self.norm1.grads_gamma.fill(0.0);
        self.norm2.grads_gamma.fill(0.0);
        self.mlstm.clear_grads();
        self.lin_gate.clear_grads();
        self.lin_value.clear_grads();
        self.lin_down.clear_grads();
    }

    fn scale_grads(&mut self, scale: f32) {
        self.norm1.grads_gamma.iter_mut().for_each(|g| *g *= scale);
        self.norm2.grads_gamma.iter_mut().for_each(|g| *g *= scale);
        self.mlstm.scale_grads(scale);
        self.lin_gate.scale_grads(scale);
        self.lin_value.scale_grads(scale);
        self.lin_down.scale_grads(scale);
    }

    fn reset_state(&mut self) {
        self.mlstm.reset_state();
    }

    fn zero_bptt_state(&mut self) {
        self.mlstm.zero_bptt_state();
    }
}
