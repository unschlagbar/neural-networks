// mlstm.rs ── mLSTM from xLSTM (Beck et al. 2024, arXiv:2405.04517)
//
// Single-head matrix-memory mLSTM.  Key differences from sLSTM:
//
//   sLSTM: scalar cell c_t,  h_{t-1} recurrence in all gate pre-activations.
//   mLSTM: matrix cell C_t ∈ ℝ^{d×d}, NO h recurrence — memory is fully in C_t.
//
// Forward equations (paper §3.2, single head, d = head_dim):
//
//   q_t  = W_q x_t + b_q                           ∈ ℝ^d
//   k_t  = (W_k x_t + b_k) / √d                   ∈ ℝ^d   (key scaling)
//   v_t  = W_v x_t + b_v                           ∈ ℝ^d
//   ĩ_t  = w_i⊤ x_t + b_i                         ∈ ℝ     (scalar pre-act)
//   f̃_t  = w_f⊤ x_t + b_f                         ∈ ℝ     (scalar pre-act)
//   õ_t  = W_o x_t + b_o                           ∈ ℝ^d
//
//   log f_t = log σ(f̃_t)
//   m_t     = max(log f_t + m_{t−1}, ĩ_t)          stabiliser (scalar)
//   i′_t    = exp(ĩ_t − m_t)                       stabilised input gate
//   f′_t    = exp(log f_t + m_{t−1} − m_t)         stabilised forget gate
//
//   C_t  = f′_t · C_{t−1} + i′_t · (v_t ⊗ k_t)  matrix cell update  ∈ ℝ^{d×d}
//   n_t  = f′_t · n_{t−1} + i′_t · k_t           normaliser vector   ∈ ℝ^d
//
//   o_t  = σ(õ_t)
//   ψ_t  = max(|n_t⊤ q_t|, 1)                    denominator
//   h_t  = o_t ⊙ (C_t q_t) / ψ_t                 output              ∈ ℝ^d
//
// Backward notes:
//   • m_t and ψ_t are treated as constants in backprop (same approximation as
//     softmax attention and sLSTM).
//   • BPTT state: dc_bptt (d²) and dn_bptt (d) flow backwards through time.
//   • No dh_bptt — mLSTM carries no hidden-state recurrence.
//   • C_init and n_init are fixed zeros (no learnable init: C is d²-dimensional).
//
// Memory: O(d²) for C and dc_bptt.  For d = 64 → 4096 f32 ≈ 16 KB.
//
// Code style mirrors slstm.rs: one Matrix per projection, gradient accumulators
// inside the layer struct, per-timestep cache for BPTT, zero heap alloc in hot path.

use iron_oxide::collections::Matrix;

use crate::{
    nn::{sub_in_place, sub_vec_in_place},
    nn_layer::{DynCache, NnLayer},
    saving,
};
use std::{any::Any, io};

const CLIP: f32 = 15.0;

// ── helpers ───────────────────────────────────────────────────────────────────

/// Numerically stable log σ(x) = −softplus(−x).
#[inline]
fn log_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        -((-x).exp() + 1.0).ln()
    } else {
        x - (x.exp() + 1.0).ln()
    }
}

/// Numerically stable σ(x).
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Sigmoid derivative from post-sigmoid value s = σ(x): s·(1−s).
#[inline]
fn dsigmoid(s: f32) -> f32 {
    s * (1.0 - s)
}

// ── MLSTMCache ────────────────────────────────────────────────────────────────

/// All per-timestep activations needed for the backward pass.
/// Pre-allocated at the start of training; zero dynamic allocation in the hot path.
pub struct MLSTMCache {
    pub input_size: usize,
    pub head_dim: usize,

    /// Saved input x_t (needed for weight gradient computation).
    pub x: Box<[f32]>,

    // ── Previous states saved at time t (used in backward) ───────────────────
    pub c_prev: Box<[f32]>, // C_{t−1}, flattened row-major,  head_dim²
    pub n_prev: Box<[f32]>, // n_{t−1},                       head_dim
    pub m_prev: f32,        // m_{t−1}

    // ── Gate pre-activations ─────────────────────────────────────────────────
    pub it_pre: f32,        // ĩ_t  (scalar, after clip)
    pub ft_pre: f32,        // f̃_t  (scalar, after clip)
    pub sigmoid_ft: f32,    // σ(f̃_t) — cached for d(f̃_t) = d(log f_t) · (1 − σ(f̃_t))
    pub ot_pre: Box<[f32]>, // W_o x_t + b_o,  head_dim

    // ── Post-activations ─────────────────────────────────────────────────────
    pub q: Box<[f32]>, // q_t = W_q x_t + b_q              (no extra activation)
    pub k: Box<[f32]>, // k_t = (W_k x_t + b_k) / √d
    pub v: Box<[f32]>, // v_t = W_v x_t + b_v
    pub o: Box<[f32]>, // o_t = σ(ot_pre),                 head_dim

    // ── Stabilised gates (scalars) ────────────────────────────────────────────
    pub i_prime: f32, // exp(ĩ_t − m_t)
    pub f_prime: f32, // exp(log σ(f̃_t) + m_{t−1} − m_t)

    // ── States at time t ─────────────────────────────────────────────────────
    pub m: f32,
    pub c: Box<[f32]>, // C_t,  head_dim²
    pub n: Box<[f32]>, // n_t,  head_dim

    /// r_t = C_t q_t  (numerator before output gate and normaliser division)
    pub r: Box<[f32]>, // head_dim

    /// ψ_t = max(|n_t⊤ q_t|, 1)
    pub psi: f32,

    /// h_t = o_t ⊙ (r_t / ψ_t)
    pub h: Box<[f32]>, // head_dim

    /// dL/d(x_t) written by backward.
    pub dx: Box<[f32]>, // input_size
}

impl DynCache for MLSTMCache {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn output(&self) -> &[f32] {
        &self.h
    }
    fn input_grad(&self) -> &[f32] {
        &self.dx
    }
}

// ── MLSTMLayerGrads ───────────────────────────────────────────────────────────

pub struct MLSTMLayerGrads {
    pub wq: Matrix, // input_size × head_dim
    pub wk: Matrix,
    pub wv: Matrix,
    pub wo: Matrix,
    pub wi: Box<[f32]>, // input_size  (scalar input-gate weight vector)
    pub wf: Box<[f32]>, // input_size  (scalar forget-gate weight vector)
    pub bq: Box<[f32]>, // head_dim
    pub bk: Box<[f32]>,
    pub bv: Box<[f32]>,
    pub bo: Box<[f32]>,
    pub bi: f32,
    pub bf: f32,
}

impl MLSTMLayerGrads {
    pub fn zeros(input_size: usize, head_dim: usize) -> Self {
        Self {
            wq: Matrix::zeros(input_size, head_dim),
            wk: Matrix::zeros(input_size, head_dim),
            wv: Matrix::zeros(input_size, head_dim),
            wo: Matrix::zeros(input_size, head_dim),
            wi: vec![0.0; input_size].into(),
            wf: vec![0.0; input_size].into(),
            bq: vec![0.0; head_dim].into(),
            bk: vec![0.0; head_dim].into(),
            bv: vec![0.0; head_dim].into(),
            bo: vec![0.0; head_dim].into(),
            bi: 0.0,
            bf: 0.0,
        }
    }
}

// ── MLSTMLayer ────────────────────────────────────────────────────────────────

pub struct MLSTMLayer {
    pub input_size: usize,
    pub head_dim: usize, // d — also the output size

    // ── Projection weights (input_size × head_dim, row-major via Matrix) ─────
    pub wq: Matrix,
    pub wk: Matrix,
    pub wv: Matrix,
    pub wo: Matrix, // output gate projection

    // ── Scalar gate weight vectors (input_size each) ──────────────────────────
    pub wi: Box<[f32]>,
    pub wf: Box<[f32]>,

    // ── Biases ────────────────────────────────────────────────────────────────
    pub bq: Box<[f32]>, // head_dim
    pub bk: Box<[f32]>,
    pub bv: Box<[f32]>,
    pub bo: Box<[f32]>,
    pub bi: f32, // scalar input-gate bias
    pub bf: f32, // scalar forget-gate bias (init > 0 to avoid tiny f′ early on)

    // ── Recurrent forward state (advanced in forward, reset in reset_state) ───
    pub c: Box<[f32]>, // C_t, flattened row-major, head_dim²
    pub n: Box<[f32]>, // n_t, head_dim
    pub m: f32,        // m_t (stabiliser scalar)

    // ── BPTT gradients flowing t → t−1 ───────────────────────────────────────
    pub dc_bptt: Box<[f32]>, // dL/dC_{t−1}, head_dim²
    pub dn_bptt: Box<[f32]>, // dL/dn_{t−1}, head_dim

    pub grads: MLSTMLayerGrads,

    // ── Pre-allocated backward scratch (zero heap in hot path) ────────────────
    pub dq: Box<[f32]>,     // head_dim — holds dr temporarily, then real dq
    pub dk: Box<[f32]>,     // head_dim — also used as temp buffer for dq
    pub dv: Box<[f32]>,     // head_dim
    pub do_pre: Box<[f32]>, // head_dim — gradient wrt ot_pre (before sigmoid)
}

impl MLSTMLayer {
    pub fn new(input_size: usize, head_dim: usize) -> Self {
        let scale = (6.0 / (input_size + head_dim) as f32).sqrt(); // Glorot
        let gate_scale = (6.0 / (input_size + 1) as f32).sqrt(); // scalar gates

        let d2 = head_dim * head_dim;

        // We need a random Box<[f32]>; borrow a temporary 1-column Matrix.
        let wi: Box<[f32]> = Matrix::random(input_size, 1, gate_scale)
            .as_slice()
            .to_vec()
            .into_boxed_slice();
        let wf: Box<[f32]> = Matrix::random(input_size, 1, gate_scale)
            .as_slice()
            .to_vec()
            .into_boxed_slice();

        Self {
            input_size,
            head_dim,
            wq: Matrix::random(input_size, head_dim, scale),
            wk: Matrix::random(input_size, head_dim, scale),
            wv: Matrix::random(input_size, head_dim, scale),
            wo: Matrix::random(input_size, head_dim, scale),
            wi,
            wf,
            bq: vec![0.0; head_dim].into(),
            bk: vec![0.0; head_dim].into(),
            bv: vec![0.0; head_dim].into(),
            bo: vec![0.0; head_dim].into(),
            bi: 0.0,
            // Positive forget-gate bias: avoids very small f′ at the start of training
            // (same convention as sLSTM, see Jozefowicz et al. 2015).
            bf: 3.0,
            c: vec![0.0; d2].into(),
            n: vec![0.0; head_dim].into(),
            m: 0.0,
            dc_bptt: vec![0.0; d2].into(),
            dn_bptt: vec![0.0; head_dim].into(),
            grads: MLSTMLayerGrads::zeros(input_size, head_dim),
            dq: vec![0.0; head_dim].into(),
            dk: vec![0.0; head_dim].into(),
            dv: vec![0.0; head_dim].into(),
            do_pre: vec![0.0; head_dim].into(),
        }
    }

    /// Construct from pre-loaded weights (used by the loading module).
    #[allow(clippy::too_many_arguments)]
    pub fn from_loaded(
        input_size: usize,
        head_dim: usize,
        wq: Matrix,
        wk: Matrix,
        wv: Matrix,
        wo: Matrix,
        wi: Box<[f32]>,
        wf: Box<[f32]>,
        bq: Box<[f32]>,
        bk: Box<[f32]>,
        bv: Box<[f32]>,
        bo: Box<[f32]>,
        bi: f32,
        bf: f32,
    ) -> Self {
        let d2 = head_dim * head_dim;
        Self {
            input_size,
            head_dim,
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
            c: vec![0.0; d2].into(),
            n: vec![0.0; head_dim].into(),
            m: 0.0,
            dc_bptt: vec![0.0; d2].into(),
            dn_bptt: vec![0.0; head_dim].into(),
            grads: MLSTMLayerGrads::zeros(input_size, head_dim),
            dq: vec![0.0; head_dim].into(),
            dk: vec![0.0; head_dim].into(),
            dv: vec![0.0; head_dim].into(),
            do_pre: vec![0.0; head_dim].into(),
        }
    }

    pub fn alloc_cache(&self) -> MLSTMCache {
        let inp = self.input_size;
        let d = self.head_dim;
        let d2 = d * d;
        MLSTMCache {
            input_size: inp,
            head_dim: d,
            x: vec![0.0; inp].into(),
            c_prev: vec![0.0; d2].into(),
            n_prev: vec![0.0; d].into(),
            m_prev: 0.0,
            it_pre: 0.0,
            ft_pre: 0.0,
            sigmoid_ft: 0.5,
            ot_pre: vec![0.0; d].into(),
            q: vec![0.0; d].into(),
            k: vec![0.0; d].into(),
            v: vec![0.0; d].into(),
            o: vec![0.0; d].into(),
            i_prime: 0.0,
            f_prime: 0.0,
            m: 0.0,
            c: vec![0.0; d2].into(),
            n: vec![0.0; d].into(),
            r: vec![0.0; d].into(),
            psi: 1.0,
            h: vec![0.0; d].into(),
            dx: vec![0.0; inp].into(),
        }
    }

    // ── Forward ───────────────────────────────────────────────────────────────

    pub fn forward(&mut self, x: &[f32], cache: &mut MLSTMCache) {
        let d = self.head_dim;
        let k_scale = 1.0 / (d as f32).sqrt();

        // ── Save input and previous state for backward ────────────────────────
        cache.x.copy_from_slice(x);
        cache.c_prev.copy_from_slice(&self.c);
        cache.n_prev.copy_from_slice(&self.n);
        cache.m_prev = self.m;

        // ── Project x into q, k, v, output gate ──────────────────────────────
        // Weights are stored as (input_size × head_dim); row_mul computes
        // y[j] = Σ_i W[i][j] · x[i]  (= W⊤x).
        self.wq.row_mul(x, &mut cache.q);
        self.wk.row_mul(x, &mut cache.k);
        self.wv.row_mul(x, &mut cache.v);
        self.wo.row_mul(x, &mut cache.ot_pre);

        for j in 0..d {
            cache.q[j] += self.bq[j];
            cache.k[j] = (cache.k[j] + self.bk[j]) * k_scale; // key scaling 1/√d
            cache.v[j] += self.bv[j];
            cache.ot_pre[j] += self.bo[j];
            cache.o[j] = sigmoid(cache.ot_pre[j]);
        }

        // ── Scalar gate pre-activations ───────────────────────────────────────
        let mut it_pre = self.bi;
        let mut ft_pre = self.bf;
        for i in 0..self.input_size {
            it_pre += self.wi[i] * x[i];
            ft_pre += self.wf[i] * x[i];
        }
        it_pre = it_pre.clamp(-CLIP, CLIP);
        ft_pre = ft_pre.clamp(-CLIP, CLIP);
        cache.it_pre = it_pre;
        cache.ft_pre = ft_pre;
        cache.sigmoid_ft = sigmoid(ft_pre); // saved for backward

        let log_f = log_sigmoid(ft_pre);

        // ── Stabiliser m_t = max(log f_t + m_{t−1}, ĩ_t) ────────────────────
        let m_new = (log_f + self.m).max(it_pre);
        cache.m = m_new;

        // Both stabilised gates ∈ (0, 1] by construction (non-positive exponent).
        let i_prime = (it_pre - m_new).exp();
        let f_prime = (log_f + self.m - m_new).exp();
        cache.i_prime = i_prime;
        cache.f_prime = f_prime;

        // ── Matrix cell: C_t = f′ · C_{t−1} + i′ · (v ⊗ k)  (row-major) ────
        for i in 0..d {
            for j in 0..d {
                self.c[i * d + j] =
                    f_prime * cache.c_prev[i * d + j] + i_prime * cache.v[i] * cache.k[j];
            }
        }
        cache.c.copy_from_slice(&self.c);

        // ── Normaliser: n_t = f′ · n_{t−1} + i′ · k_t ───────────────────────
        for j in 0..d {
            self.n[j] = f_prime * cache.n_prev[j] + i_prime * cache.k[j];
        }
        cache.n.copy_from_slice(&self.n);

        // Advance stabiliser state.
        self.m = m_new;

        // ── r_t = C_t q_t ─────────────────────────────────────────────────────
        for i in 0..d {
            let mut s = 0.0_f32;
            for j in 0..d {
                s += self.c[i * d + j] * cache.q[j];
            }
            cache.r[i] = s;
        }

        // ── ψ_t = max(|n_t⊤ q_t|, 1) ─────────────────────────────────────────
        let ntq: f32 = (0..d).map(|j| self.n[j] * cache.q[j]).sum();
        cache.psi = ntq.abs().max(1.0);

        // ── h_t = o_t ⊙ (r_t / ψ_t) ──────────────────────────────────────────
        for j in 0..d {
            cache.h[j] = cache.o[j] * cache.r[j] / cache.psi;
        }
    }

    // ── Backward ──────────────────────────────────────────────────────────────
    //
    // `delta` comes in as dL/dh_t (length = head_dim).
    // On return, `cache.dx` holds dL/dx_t (length = input_size), and
    // self.dc_bptt / self.dn_bptt are updated to dL/dC_{t−1} and dL/dn_{t−1}.
    //
    // Derivation (ψ and m treated as constants throughout):
    //
    //   (A) h = o ⊙ r/ψ
    //       do[j] = δ[j] · r[j]/ψ → do_pre via dsigmoid
    //       dr[j] = δ[j] · o[j]/ψ
    //
    //   (B) r = C q
    //       dC_from_h[i,j] = dr[i] · q[j]   (outer product)
    //       dq[j]          = Σ_i C[i,j] · dr[i]   (C⊤ dr)
    //
    //   (C) Add BPTT from t+1:  dC_total = dC_from_h + dc_bptt_incoming
    //                            dn_total = dn_bptt_incoming
    //
    //   (D) From C_t = f′ C_prev + i′ (v ⊗ k):
    //       d(f′)   = <dC_total, C_prev>_F + dn_total · n_prev
    //       d(i′)   = <dC_total, v ⊗ k>_F  + dn_total · k
    //       dv[i]   = i′ · (dC_total_row_i · k)
    //       dk[j]   = i′ · (dC_total_col_j · v)  +  i′ · dn_total[j]
    //       dC_prev = f′ · dC_total       (stored back in dc_bptt for t−1)
    //       dn_prev = f′ · dn_total       (stored back in dn_bptt for t−1)
    //
    //   (E) Stabilised gates (m constant):
    //       dĩ_t   = d(i′) · i′       [chain rule through exp(ĩ − m)]
    //       d(log f) = d(f′) · f′    [chain rule through exp(log f + m_prev − m)]
    //       df̃_t   = d(log f) · (1 − σ(f̃))  [d/dx log σ(x) = 1 − σ(x)]
    //
    //   (F) Key scaling: k = kt_pre / √d → d(kt_pre) = dk · (1/√d)
    //
    //   (G) Weight grads via add_outer; input grad via W · d_pre-act.

    pub fn backward(&mut self, delta: &mut [f32], cache: &mut MLSTMCache) {
        let d = self.head_dim;
        let k_scale = 1.0 / (d as f32).sqrt();

        // (A) Output gate and r gradients.
        // dq temporarily stores dr = dL/dr.
        for j in 0..d {
            let dh = delta[j];
            self.do_pre[j] = dh * (cache.r[j] / cache.psi) * dsigmoid(cache.o[j]);
            self.dq[j] = dh * cache.o[j] / cache.psi; // dr stored in dq
        }

        // (B+C) Accumulate dC_from_h into dc_bptt (which already holds the t+1 BPTT
        //       gradient), giving dc_bptt = total dC_t.
        //   dC_total[i,j] += dr[i] · q[j]
        for i in 0..d {
            let dr_i = self.dq[i]; // dr still lives in self.dq
            for j in 0..d {
                self.dc_bptt[i * d + j] += dr_i * cache.q[j];
            }
        }
        // dc_bptt now holds the total dC_t.

        // (B) Compute real dq = C⊤ dr.
        // Use dk as temporary to avoid clobbering dq while computing from it.
        for j in 0..d {
            let mut s = 0.0_f32;
            for i in 0..d {
                s += cache.c[i * d + j] * self.dq[i]; // self.dq is still dr here
            }
            self.dk[j] = s; // real dq lands in dk temporarily
        }
        self.dq.copy_from_slice(&self.dk); // dq now holds the true query gradient

        // (D) Gradients through C_t and n_t.
        // We compute: d(f′), d(i′), dv, dk from dc_bptt = total dC_t.
        let mut d_f_prime = 0.0_f32;
        let mut d_i_prime = 0.0_f32;
        self.dv.fill(0.0);
        self.dk.fill(0.0);

        for i in 0..d {
            let v_i = cache.v[i];
            let mut dv_i = 0.0_f32;
            for j in 0..d {
                let dc = self.dc_bptt[i * d + j];
                let k_j = cache.k[j];
                let c_prev_ij = cache.c_prev[i * d + j];
                d_f_prime += dc * c_prev_ij;
                d_i_prime += dc * v_i * k_j;
                dv_i += cache.i_prime * dc * k_j;
                self.dk[j] += cache.i_prime * dc * v_i;
            }
            self.dv[i] = dv_i;
        }

        // Contribution from the normaliser.
        for j in 0..d {
            let dn = self.dn_bptt[j];
            d_f_prime += dn * cache.n_prev[j];
            d_i_prime += dn * cache.k[j];
            self.dk[j] += cache.i_prime * dn;
        }

        // Propagate BPTT gradients to t−1 (in-place; dc_bptt / dn_bptt are now fresh).
        for ij in 0..(d * d) {
            self.dc_bptt[ij] *= cache.f_prime; // dC_{t−1} = f′ · dC_total
        }
        for j in 0..d {
            self.dn_bptt[j] *= cache.f_prime; // dn_{t−1} = f′ · dn_total
        }

        // (E) Map f′/i′ gradients back to pre-activations.
        let d_it_pre = d_i_prime * cache.i_prime; // dĩ_t
        let d_log_f = d_f_prime * cache.f_prime; // d(log f_t)
        let d_ft_pre = d_log_f * (1.0 - cache.sigmoid_ft); // df̃_t

        // (F) Key scaling: d(kt_pre) = dk · (1/√d).
        for j in 0..d {
            self.dk[j] *= k_scale;
        }

        // (G) Weight gradients: dW += outer(x, d_pre-act)  via add_outer.
        let x = &cache.x;
        self.grads.wq.add_outer(x, &self.dq);
        self.grads.wk.add_outer(x, &self.dk);
        self.grads.wv.add_outer(x, &self.dv);
        self.grads.wo.add_outer(x, &self.do_pre);

        for i in 0..self.input_size {
            self.grads.wi[i] += d_it_pre * x[i];
            self.grads.wf[i] += d_ft_pre * x[i];
        }
        for j in 0..d {
            self.grads.bq[j] += self.dq[j];
            self.grads.bk[j] += self.dk[j];
            self.grads.bv[j] += self.dv[j];
            self.grads.bo[j] += self.do_pre[j];
        }
        self.grads.bi += d_it_pre;
        self.grads.bf += d_ft_pre;

        // (G) Input gradient:
        //   dx[i] = Σ_j ( wq[i,j]·dq[j] + wk[i,j]·dk[j]
        //                + wv[i,j]·dv[j] + wo[i,j]·do_pre[j] )
        //           + wi[i]·dĩ_t + wf[i]·df̃_t
        let inp = self.input_size;
        for i in 0..inp {
            let mut s = 0.0_f32;
            for j in 0..d {
                s += self.wq[i][j] * self.dq[j]
                    + self.wk[i][j] * self.dk[j]
                    + self.wv[i][j] * self.dv[j]
                    + self.wo[i][j] * self.do_pre[j];
            }
            s += self.wi[i] * d_it_pre + self.wf[i] * d_ft_pre;
            delta[i] = s;
        }
        cache.dx.copy_from_slice(delta);
    }
}

// ── impl NnLayer for MLSTMLayer ───────────────────────────────────────────────

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
        12
    }

    /// Binary layout (must match `load_mlstm` in loading.rs):
    ///   wq, wk, wv, wo  — Matrix (input_size × head_dim), each in turn
    ///   wi, wf          — f32_slice (input_size)
    ///   bq, bk, bv, bo  — f32_slice (head_dim)
    ///   bi, bf          — f32 (scalar biases, 4 bytes each)
    fn save(&self, w: &mut dyn io::Write) -> io::Result<()> {
        saving::write_matrix(w, &self.wq)?;
        saving::write_matrix(w, &self.wk)?;
        saving::write_matrix(w, &self.wv)?;
        saving::write_matrix(w, &self.wo)?;
        saving::write_f32_slice(w, &self.wi)?;
        saving::write_f32_slice(w, &self.wf)?;
        saving::write_f32_slice(w, &self.bq)?;
        saving::write_f32_slice(w, &self.bk)?;
        saving::write_f32_slice(w, &self.bv)?;
        saving::write_f32_slice(w, &self.bo)?;
        saving::write_f32(w, self.bi)?;
        saving::write_f32(w, self.bf)
    }

    fn make_cache(&self) -> Box<dyn DynCache> {
        Box::new(MLSTMLayer::alloc_cache(self))
    }

    fn input_size(&self) -> usize {
        self.input_size
    }

    fn output_size(&self) -> usize {
        self.head_dim
    }

    fn apply_grads(&mut self, lr: f32) {
        // Clip scalar gate gradients (can spike due to the scalar bottleneck).
        self.grads
            .wi
            .iter_mut()
            .for_each(|g| *g = g.clamp(-CLIP, CLIP));
        self.grads
            .wf
            .iter_mut()
            .for_each(|g| *g = g.clamp(-CLIP, CLIP));

        sub_in_place(&mut self.wq, &self.grads.wq, lr);
        sub_in_place(&mut self.wk, &self.grads.wk, lr);
        sub_in_place(&mut self.wv, &self.grads.wv, lr);
        sub_in_place(&mut self.wo, &self.grads.wo, lr);
        sub_vec_in_place(&mut self.wi, &self.grads.wi, lr);
        sub_vec_in_place(&mut self.wf, &self.grads.wf, lr);
        sub_vec_in_place(&mut self.bq, &self.grads.bq, lr);
        sub_vec_in_place(&mut self.bk, &self.grads.bk, lr);
        sub_vec_in_place(&mut self.bv, &self.grads.bv, lr);
        sub_vec_in_place(&mut self.bo, &self.grads.bo, lr);
        self.bi -= lr * self.grads.bi;
        self.bf -= lr * self.grads.bf;
    }

    fn clear_grads(&mut self) {
        self.grads.wq.clear();
        self.grads.wk.clear();
        self.grads.wv.clear();
        self.grads.wo.clear();
        self.grads.wi.fill(0.0);
        self.grads.wf.fill(0.0);
        self.grads.bq.fill(0.0);
        self.grads.bk.fill(0.0);
        self.grads.bv.fill(0.0);
        self.grads.bo.fill(0.0);
        self.grads.bi = 0.0;
        self.grads.bf = 0.0;
    }

    fn scale_grads(&mut self, scale: f32) {
        self.grads.wq.scale(scale);
        self.grads.wk.scale(scale);
        self.grads.wv.scale(scale);
        self.grads.wo.scale(scale);
        self.grads.wi.iter_mut().for_each(|g| *g *= scale);
        self.grads.wf.iter_mut().for_each(|g| *g *= scale);
        self.grads.bq.iter_mut().for_each(|g| *g *= scale);
        self.grads.bk.iter_mut().for_each(|g| *g *= scale);
        self.grads.bv.iter_mut().for_each(|g| *g *= scale);
        self.grads.bo.iter_mut().for_each(|g| *g *= scale);
        self.grads.bi *= scale;
        self.grads.bf *= scale;
    }

    fn reset_state(&mut self) {
        // C, n, m all reset to zero (fixed init — no learnable init in mLSTM).
        self.c.fill(0.0);
        self.n.fill(0.0);
        self.m = 0.0;
    }

    fn zero_bptt_state(&mut self) {
        self.dc_bptt.fill(0.0);
        self.dn_bptt.fill(0.0);
    }

    // accumulate_init_grad: no-op — mLSTM has no learnable initial states.
}
