// mlstm.rs — mLSTM-Zelle aus xLSTM (Beck et al., "xLSTM: Extended Long
// Short-Term Memory", 2024, arXiv:2405.04517).
//
// Single-Head Implementation:
//
//   q_t = W_q · x_t + b_q
//   k_t = (W_k · x_t + b_k) / √d     (wie Attention)
//   v_t = W_v · x_t + b_v
//   o_t = σ(W_o · x_t + b_o)
//
//   i_log = w_iᵀ x_t + b_i           (skalares Input-Gate, Log-Space)
//   f_log = w_fᵀ x_t + b_f           (skalares Forget-Gate, Log-Space)
//
//   m_t  = max(f_log + m_{t-1}, i_log)            ← Stabilisierer
//   i'_t = exp(i_log - m_t)
//   f'_t = exp(f_log + m_{t-1} - m_t)
//
//   C_t = f'_t · C_{t-1} + i'_t · v_t · k_tᵀ     (Matrix-Memory)
//   n_t = f'_t · n_{t-1} + i'_t · k_t            (Normalizer)
//
//   h̃_t = (C_t · q_t) / max(|n_tᵀ q_t|, 1)
//   h_t = o_t ⊙ h̃_t
//
// Speicher pro Timestep: O(d²) wegen C_prev. Für d=128, seq=512 → 32 MB/Layer.
//
// Layer-Tag: 9  (nicht von den existierenden 0–8 belegt)

use iron_oxide::collections::Matrix;
use rand::random_range;

use crate::{
    activations::sigmoid,
    nn::{add_vec_in_place, sub_in_place, sub_vec_in_place},
    nn_layer::{DynCache, NnLayer},
    saving,
};
use std::{any::Any, io};

// ── MLSTMCache ────────────────────────────────────────────────────────────────

/// Alles, was ein Timestep für Forward+Backward braucht — eine einzige
/// Allokation pro Timestep in `make_cache`, keine im Hot-Path.
pub struct MLSTMCache {
    pub input_size: usize,
    pub hidden_size: usize,

    pub x: Box<[f32]>, // Input x_t
    pub q: Box<[f32]>, // Query q_t
    pub k: Box<[f32]>, // Key k_t (bereits mit 1/√d skaliert)
    pub v: Box<[f32]>, // Value v_t
    pub o: Box<[f32]>, // Output-Gate σ(·), aktiviert

    pub i_log: f32,  // i_log (vor Stabilisierung)
    pub f_log: f32,  // f_log
    pub i_t: f32,    // stabilisiertes i'_t
    pub f_t: f32,    // stabilisiertes f'_t
    pub m: f32,      // m_t
    pub m_prev: f32, // m_{t-1}
    /// True = f_log + m_prev ≥ i_log (Case A des max(·)). Für Backward.
    pub case_a: bool,

    pub c_prev: Box<[f32]>, // C_{t-1} (d·d flattened) — für d_f und d_q
    pub n_prev: Box<[f32]>, // n_{t-1}
    pub n: Box<[f32]>,      // n_t

    pub cq: Box<[f32]>,      // C_t · q_t
    pub h_tilde: Box<[f32]>, // h̃_t
    pub output: Box<[f32]>,  // h_t = o ⊙ h̃
    pub nq_dot: f32,         // n_tᵀ · q_t
    pub denom: f32,          // max(|n.q|, 1)
    pub denom_is_nq: bool,   // true, wenn |n.q| ≥ 1
    pub nq_sign: f32,        // sign(n.q)

    pub dx: Box<[f32]>, // dL/dx_t
}

impl DynCache for MLSTMCache {
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

// ── MLSTMLayerGrads ───────────────────────────────────────────────────────────

pub struct MLSTMLayerGrads {
    pub w_q: Matrix,
    pub w_k: Matrix,
    pub w_v: Matrix,
    pub w_o: Matrix,
    pub b_q: Box<[f32]>,
    pub b_k: Box<[f32]>,
    pub b_v: Box<[f32]>,
    pub b_o: Box<[f32]>,
    pub w_i: Box<[f32]>,
    pub w_f: Box<[f32]>,
    pub b_i: f32,
    pub b_f: f32,
    pub c_init: Box<[f32]>, // d·d
    pub n_init: Box<[f32]>,
    pub m_init: f32,
}

impl MLSTMLayerGrads {
    pub fn zeros(input_size: usize, d: usize) -> Self {
        Self {
            w_q: Matrix::zeros(input_size, d),
            w_k: Matrix::zeros(input_size, d),
            w_v: Matrix::zeros(input_size, d),
            w_o: Matrix::zeros(input_size, d),
            b_q: vec![0.0; d].into_boxed_slice(),
            b_k: vec![0.0; d].into_boxed_slice(),
            b_v: vec![0.0; d].into_boxed_slice(),
            b_o: vec![0.0; d].into_boxed_slice(),
            w_i: vec![0.0; input_size].into_boxed_slice(),
            w_f: vec![0.0; input_size].into_boxed_slice(),
            b_i: 0.0,
            b_f: 0.0,
            c_init: vec![0.0; d * d].into_boxed_slice(),
            n_init: vec![0.0; d].into_boxed_slice(),
            m_init: 0.0,
        }
    }
}

// ── MLSTMLayer ────────────────────────────────────────────────────────────────

pub struct MLSTMLayer {
    pub input_size: usize,
    pub hidden_size: usize, // = d (Head-Dimension)

    // Projektionen
    pub w_q: Matrix,
    pub w_k: Matrix,
    pub w_v: Matrix,
    pub w_o: Matrix,
    pub b_q: Box<[f32]>,
    pub b_k: Box<[f32]>,
    pub b_v: Box<[f32]>,
    pub b_o: Box<[f32]>,

    // Skalare Gate-Projektionen
    pub w_i: Box<[f32]>,
    pub w_f: Box<[f32]>,
    pub b_i: f32,
    pub b_f: f32,

    // Lernbare Initialzustände (analog h_init/c_init beim LSTM)
    pub c_init: Box<[f32]>,
    pub n_init: Box<[f32]>,
    pub m_init: f32,

    // Forward-State (wird durch die Zeit getragen)
    pub c: Box<[f32]>,
    pub n: Box<[f32]>,
    pub m: f32,

    // BPTT-Gradienten (dL/dC_{t-1}, dL/dn_{t-1}, dL/dm_{t-1})
    pub dc_bptt: Box<[f32]>,
    pub dn_bptt: Box<[f32]>,
    pub dm_bptt: f32,

    pub grads: MLSTMLayerGrads,
}

impl MLSTMLayer {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let d = hidden_size;
        let scale = (6.0 / (input_size as f32 + d as f32)).sqrt();

        // Kleinere Skala für i/f-Gate-Gewichte → initial fast neutrale Gates.
        let gate_scale = 0.1;

        let mk_vec =
            |n: usize, s: f32| -> Box<[f32]> { (0..n).map(|_| random_range(-s..s)).collect() };

        Self {
            input_size,
            hidden_size: d,

            w_q: Matrix::random(input_size, d, scale),
            w_k: Matrix::random(input_size, d, scale),
            w_v: Matrix::random(input_size, d, scale),
            w_o: Matrix::random(input_size, d, scale),
            b_q: vec![0.0; d].into_boxed_slice(),
            b_k: vec![0.0; d].into_boxed_slice(),
            b_v: vec![0.0; d].into_boxed_slice(),
            b_o: vec![0.0; d].into_boxed_slice(),

            w_i: mk_vec(input_size, gate_scale),
            w_f: mk_vec(input_size, gate_scale),
            b_i: -10.0,
            b_f: 3.0,

            c_init: vec![0.0; d * d].into_boxed_slice(),
            n_init: vec![0.0; d].into_boxed_slice(),
            m_init: 0.0,

            c: vec![0.0; d * d].into_boxed_slice(),
            n: vec![0.0; d].into_boxed_slice(),
            m: 0.0,

            dc_bptt: vec![0.0; d * d].into_boxed_slice(),
            dn_bptt: vec![0.0; d].into_boxed_slice(),
            dm_bptt: 0.0,

            grads: MLSTMLayerGrads::zeros(input_size, d),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_loaded(
        input_size: usize,
        hidden_size: usize,
        w_q: Matrix,
        w_k: Matrix,
        w_v: Matrix,
        w_o: Matrix,
        b_q: Box<[f32]>,
        b_k: Box<[f32]>,
        b_v: Box<[f32]>,
        b_o: Box<[f32]>,
        w_i: Box<[f32]>,
        w_f: Box<[f32]>,
        b_i: f32,
        b_f: f32,
        c_init: Box<[f32]>,
        n_init: Box<[f32]>,
        m_init: f32,
    ) -> Self {
        let d = hidden_size;
        debug_assert_eq!(w_q.rows(), input_size);
        debug_assert_eq!(w_q.cols(), d);
        debug_assert_eq!(c_init.len(), d * d);
        debug_assert_eq!(n_init.len(), d);

        let c = c_init.clone();
        let n = n_init.clone();
        Self {
            input_size,
            hidden_size: d,
            w_q,
            w_k,
            w_v,
            w_o,
            b_q,
            b_k,
            b_v,
            b_o,
            w_i,
            w_f,
            b_i,
            b_f,
            c_init,
            n_init,
            m_init,
            c,
            n,
            m: m_init,
            dc_bptt: vec![0.0; d * d].into_boxed_slice(),
            dn_bptt: vec![0.0; d].into_boxed_slice(),
            dm_bptt: 0.0,
            grads: MLSTMLayerGrads::zeros(input_size, d),
        }
    }

    /// State vor jeder neuen Sequenz zurücksetzen.
    pub fn reset(&mut self) {
        self.c.copy_from_slice(&self.c_init);
        self.n.copy_from_slice(&self.n_init);
        self.m = self.m_init;
        self.dc_bptt.fill(0.0);
        self.dn_bptt.fill(0.0);
        self.dm_bptt = 0.0;
    }

    pub fn alloc_cache(&self) -> MLSTMCache {
        let d = self.hidden_size;
        let i = self.input_size;
        MLSTMCache {
            input_size: i,
            hidden_size: d,
            x: vec![0.0; i].into_boxed_slice(),
            q: vec![0.0; d].into_boxed_slice(),
            k: vec![0.0; d].into_boxed_slice(),
            v: vec![0.0; d].into_boxed_slice(),
            o: vec![0.0; d].into_boxed_slice(),
            i_log: 0.0,
            f_log: 0.0,
            i_t: 0.0,
            f_t: 0.0,
            m: 0.0,
            m_prev: 0.0,
            case_a: false,
            c_prev: vec![0.0; d * d].into_boxed_slice(),
            n_prev: vec![0.0; d].into_boxed_slice(),
            n: vec![0.0; d].into_boxed_slice(),
            cq: vec![0.0; d].into_boxed_slice(),
            h_tilde: vec![0.0; d].into_boxed_slice(),
            output: vec![0.0; d].into_boxed_slice(),
            nq_dot: 0.0,
            denom: 1.0,
            denom_is_nq: false,
            nq_sign: 0.0,
            dx: vec![0.0; i].into_boxed_slice(),
        }
    }

    // ── Forward ───────────────────────────────────────────────────────────────

    pub fn forward(&mut self, input: &[f32], cache: &mut MLSTMCache) {
        let d = self.hidden_size;
        let in_size = self.input_size;

        // Snapshot des Vorzustands
        cache.x.copy_from_slice(input);
        cache.c_prev.copy_from_slice(&self.c);
        cache.n_prev.copy_from_slice(&self.n);
        cache.m_prev = self.m;

        // QKV + Output-Gate-Logit
        self.w_q.row_mul(input, &mut cache.q);
        self.w_k.row_mul(input, &mut cache.k);
        self.w_v.row_mul(input, &mut cache.v);
        self.w_o.row_mul(input, &mut cache.o);
        for j in 0..d {
            cache.q[j] += self.b_q[j];
            cache.k[j] += self.b_k[j];
            cache.v[j] += self.b_v[j];
            cache.o[j] = sigmoid(cache.o[j] + self.b_o[j]);
        }

        // Key-Skalierung 1/√d (wie bei Attention)
        let k_scale = 1.0 / (d as f32).sqrt();
        for j in 0..d {
            cache.k[j] *= k_scale;
        }

        // Skalare Gate-Logits
        let mut i_log = self.b_i;
        let mut f_log = self.b_f;
        for i in 0..in_size {
            i_log += self.w_i[i] * input[i];
            f_log += self.w_f[i] * input[i];
        }
        cache.i_log = i_log;
        cache.f_log = f_log;

        // Stabilisierung
        let alpha = f_log + self.m; // f_log + m_{t-1}
        let beta = i_log;
        let case_a = alpha >= beta;
        cache.case_a = case_a;

        let m_new = if case_a { alpha } else { beta };
        let i_t = (beta - m_new).exp(); // in Case A: exp(β - α), in Case B: 1
        let f_t = (alpha - m_new).exp(); // in Case A: 1, in Case B: exp(α - β)
        cache.m = m_new;
        cache.i_t = i_t;
        cache.f_t = f_t;

        // Memory-Update: C_t = f_t · C_{t-1} + i_t · v · kᵀ
        for row in 0..d {
            let v_row = cache.v[row];
            let base = row * d;
            for col in 0..d {
                self.c[base + col] = f_t * cache.c_prev[base + col] + i_t * v_row * cache.k[col];
            }
        }

        // Normalizer: n_t = f_t · n_{t-1} + i_t · k
        for j in 0..d {
            self.n[j] = f_t * cache.n_prev[j] + i_t * cache.k[j];
        }
        cache.n.copy_from_slice(&self.n);

        self.m = m_new;

        // Read: Cq = C_t · q
        for row in 0..d {
            let base = row * d;
            let mut s = 0.0;
            for col in 0..d {
                s += self.c[base + col] * cache.q[col];
            }
            cache.cq[row] = s;
        }

        // Denominator: max(|n·q|, 1)
        let mut nq = 0.0;
        for j in 0..d {
            nq += cache.n[j] * cache.q[j];
        }
        cache.nq_dot = nq;
        let abs_nq = nq.abs();
        if abs_nq >= 1.0 {
            cache.denom = abs_nq;
            cache.denom_is_nq = true;
        } else {
            cache.denom = 1.0;
            cache.denom_is_nq = false;
        }
        cache.nq_sign = if nq >= 0.0 { 1.0 } else { -1.0 };

        // h̃ = Cq / denom  und  h = o ⊙ h̃
        let inv_denom = 1.0 / cache.denom;
        for j in 0..d {
            cache.h_tilde[j] = cache.cq[j] * inv_denom;
            cache.output[j] = cache.o[j] * cache.h_tilde[j];
        }
    }

    // ── Backward ──────────────────────────────────────────────────────────────
    //
    // Gegeben  δ = dL/dh_t   (Länge d).
    //
    // Laufende Akkumulatoren (vorm Aufruf):
    //   self.dc_bptt = dL/dC_t   (aus Zukunft, wird hier komplettiert)
    //   self.dn_bptt = dL/dn_t   (dto.)
    //   self.dm_bptt = dL/dm_t   (dto.)
    //
    // Nach dem Aufruf:
    //   self.dc_bptt = dL/dC_{t-1}  (also f_t · altes dc_bptt)
    //   self.dn_bptt = dL/dn_{t-1}
    //   self.dm_bptt = dL/dm_{t-1}
    //   cache.dx     = dL/dx_t
    //   self.grads.* aktualisiert

    pub fn backward(&mut self, delta: &mut [f32], cache: &mut MLSTMCache) {
        let d = self.hidden_size;
        let in_size = self.input_size;

        // ── 1) δ → d_o (am Logit) und d_h̃ ────────────────────────────────────
        //
        //   o_logit hinzu: d_o_logit = δ · h̃ · o · (1-o)
        //   d_h̃         = δ · o
        //
        let mut d_o_logit = vec![0.0; d];
        let mut d_h_tilde = vec![0.0; d];
        for j in 0..d {
            let d_o = delta[j] * cache.h_tilde[j];
            d_o_logit[j] = d_o * cache.o[j] * (1.0 - cache.o[j]);
            d_h_tilde[j] = delta[j] * cache.o[j];
        }

        // ── 2) h̃ = Cq / denom  →  d_Cq, d_denom ──────────────────────────────
        let inv_denom = 1.0 / cache.denom;
        let mut d_cq = vec![0.0; d];
        let mut d_denom: f32 = 0.0;
        for j in 0..d {
            d_cq[j] = d_h_tilde[j] * inv_denom;
            // ∂(Cq_j/denom)/∂denom = -Cq_j/denom² = -h̃_j/denom
            d_denom -= d_h_tilde[j] * cache.h_tilde[j] * inv_denom;
        }

        // ── 3) denom = max(|n·q|, 1) ──────────────────────────────────────────
        let d_nq_dot = if cache.denom_is_nq {
            d_denom * cache.nq_sign
        } else {
            0.0
        };

        // ── 4) Lokale Beiträge in dc_bptt / dn_bptt einspeisen ────────────────
        //
        //   n·q       →  dn += d_nq · q,  dq += d_nq · n
        //   Cq = C·q  →  dC[i,j] += d_cq_i · q_j,  dq += Cᵀ d_cq
        //
        // Beachte: dc_bptt / dn_bptt enthalten schon die Zukunftsbeiträge.

        let mut d_q = vec![0.0; d];
        for j in 0..d {
            self.dn_bptt[j] += d_nq_dot * cache.q[j];
            d_q[j] = d_nq_dot * cache.n[j];
        }
        for row in 0..d {
            let base = row * d;
            for col in 0..d {
                self.dc_bptt[base + col] += d_cq[row] * cache.q[col];
            }
        }

        // d_q += Cᵀ d_cq      mit     C_t[i,j] = f_t · C_prev[i,j] + i_t · v_i · k_j
        //   ⇒ (Cᵀ d_cq)[j] = f_t · (C_prevᵀ d_cq)[j] + i_t · k_j · (v · d_cq)
        let mut v_dot_dcq = 0.0;
        for i in 0..d {
            v_dot_dcq += cache.v[i] * d_cq[i];
        }
        for j in 0..d {
            let mut s = 0.0;
            for i in 0..d {
                s += cache.c_prev[i * d + j] * d_cq[i];
            }
            d_q[j] += cache.f_t * s + cache.i_t * cache.k[j] * v_dot_dcq;
        }

        // ── 5) Aus vollen dC_t, dn_t die Gradienten für (f_t, i_t, v, k) ──────
        //
        //   C_t = f_t · C_prev + i_t · v · kᵀ
        //   n_t = f_t · n_prev + i_t · k
        //
        //   d_f  = ⟨dC_t, C_prev⟩_F + ⟨dn_t, n_prev⟩
        //   d_i  = Σ dC_t[i,j] · v_i · k_j  +  Σ dn_t[i] · k_i
        //   d_v  = i_t · (dC_t · k)
        //   d_k  = i_t · (dC_tᵀ · v) + i_t · dn_t
        //
        let mut d_f_t: f32 = 0.0;
        let mut d_i_t: f32 = 0.0;
        let mut d_v = vec![0.0; d];
        let mut d_k = vec![0.0; d];

        for row in 0..d {
            let base = row * d;
            for col in 0..d {
                let dc_ij = self.dc_bptt[base + col];
                d_f_t += dc_ij * cache.c_prev[base + col];
                d_i_t += dc_ij * cache.v[row] * cache.k[col];
                d_v[row] += dc_ij * cache.k[col];
                d_k[col] += dc_ij * cache.v[row];
            }
            d_f_t += self.dn_bptt[row] * cache.n_prev[row];
            d_i_t += self.dn_bptt[row] * cache.k[row];
        }
        for j in 0..d {
            d_v[j] *= cache.i_t;
            d_k[j] = cache.i_t * (d_k[j] + self.dn_bptt[j]);
        }

        // ── 6) dC_bptt / dn_bptt für t-1 weiterleiten (Faktor f_t) ────────────
        for e in self.dc_bptt.iter_mut() {
            *e *= cache.f_t;
        }
        for e in self.dn_bptt.iter_mut() {
            *e *= cache.f_t;
        }

        // ── 7) k-Gradient ist in skalierter Form: unskalieren für W_k ─────────
        let k_scale = 1.0 / (d as f32).sqrt();
        for e in d_k.iter_mut() {
            *e *= k_scale;
        }

        // ── 8) Stabilisator-Backward (Case A / Case B) ────────────────────────
        //
        //   α = f_log + m_prev,   β = i_log,   m_t = max(α, β)
        //
        //   Case A  (α ≥ β):      m_t = α,  i_t = exp(β-α),  f_t = 1
        //     d_i_log = d_i_t · i_t
        //     d_f_log = -d_i_t · i_t + dm_t
        //     d_m_prev = -d_i_t · i_t + dm_t
        //
        //   Case B  (α < β):      m_t = β,  i_t = 1,  f_t = exp(α-β)
        //     d_i_log = -d_f_t · f_t + dm_t
        //     d_f_log = d_f_t · f_t
        //     d_m_prev = d_f_t · f_t
        //
        let dm_t = self.dm_bptt;
        let (d_i_log, d_f_log, d_m_prev) = if cache.case_a {
            let di = d_i_t * cache.i_t;
            (di, -di + dm_t, -di + dm_t)
        } else {
            let df = d_f_t * cache.f_t;
            (-df + dm_t, df, df)
        };
        self.dm_bptt = d_m_prev;

        // ── 9) Parameter-Gradienten akkumulieren + dx berechnen ───────────────
        let g = &mut self.grads;
        let x = &cache.x[..];

        g.w_q.add_outer(x, &d_q);
        g.w_k.add_outer(x, &d_k);
        g.w_v.add_outer(x, &d_v);
        g.w_o.add_outer(x, &d_o_logit);

        for j in 0..d {
            g.b_q[j] += d_q[j];
            g.b_k[j] += d_k[j];
            g.b_v[j] += d_v[j];
            g.b_o[j] += d_o_logit[j];
        }

        // dx durch alle vier Projektionen + skalare Gates
        for i in 0..in_size {
            let mut sx = 0.0;
            for j in 0..d {
                sx += self.w_q[i][j] * d_q[j]
                    + self.w_k[i][j] * d_k[j]
                    + self.w_v[i][j] * d_v[j]
                    + self.w_o[i][j] * d_o_logit[j];
            }
            // Skalare Gates
            g.w_i[i] += d_i_log * x[i];
            g.w_f[i] += d_f_log * x[i];
            sx += self.w_i[i] * d_i_log + self.w_f[i] * d_f_log;
            cache.dx[i] = sx;
        }
        g.b_i += d_i_log;
        g.b_f += d_f_log;
    }
}

// ── impl NnLayer ──────────────────────────────────────────────────────────────

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
        13 // TAG_MLSTM — neu, nicht mit 0–8 kollidierend
    }

    fn save(&self, w: &mut dyn io::Write) -> io::Result<()> {
        saving::write_matrix(w, &self.w_q)?;
        saving::write_matrix(w, &self.w_k)?;
        saving::write_matrix(w, &self.w_v)?;
        saving::write_matrix(w, &self.w_o)?;
        saving::write_f32_slice(w, &self.b_q)?;
        saving::write_f32_slice(w, &self.b_k)?;
        saving::write_f32_slice(w, &self.b_v)?;
        saving::write_f32_slice(w, &self.b_o)?;
        saving::write_f32_slice(w, &self.w_i)?;
        saving::write_f32_slice(w, &self.w_f)?;
        saving::write_f32(w, self.b_i)?;
        saving::write_f32(w, self.b_f)?;
        saving::write_f32_slice(w, &self.c_init)?;
        saving::write_f32_slice(w, &self.n_init)?;
        saving::write_f32(w, self.m_init)
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
        sub_in_place(&mut self.w_q, &self.grads.w_q, lr);
        sub_in_place(&mut self.w_k, &self.grads.w_k, lr);
        sub_in_place(&mut self.w_v, &self.grads.w_v, lr);
        sub_in_place(&mut self.w_o, &self.grads.w_o, lr);
        sub_vec_in_place(&mut self.b_q, &self.grads.b_q, lr);
        sub_vec_in_place(&mut self.b_k, &self.grads.b_k, lr);
        sub_vec_in_place(&mut self.b_v, &self.grads.b_v, lr);
        sub_vec_in_place(&mut self.b_o, &self.grads.b_o, lr);
        sub_vec_in_place(&mut self.w_i, &self.grads.w_i, lr);
        sub_vec_in_place(&mut self.w_f, &self.grads.w_f, lr);
        self.b_i -= lr * self.grads.b_i;
        self.b_f -= lr * self.grads.b_f;
        sub_vec_in_place(&mut self.c_init, &self.grads.c_init, lr);
        sub_vec_in_place(&mut self.n_init, &self.grads.n_init, lr);
        self.m_init -= lr * self.grads.m_init;
    }

    fn clear_grads(&mut self) {
        self.grads.w_q.clear();
        self.grads.w_k.clear();
        self.grads.w_v.clear();
        self.grads.w_o.clear();
        self.grads.b_q.fill(0.0);
        self.grads.b_k.fill(0.0);
        self.grads.b_v.fill(0.0);
        self.grads.b_o.fill(0.0);
        self.grads.w_i.fill(0.0);
        self.grads.w_f.fill(0.0);
        self.grads.b_i = 0.0;
        self.grads.b_f = 0.0;
        self.grads.c_init.fill(0.0);
        self.grads.n_init.fill(0.0);
        self.grads.m_init = 0.0;
    }

    fn scale_grads(&mut self, scale: f32) {
        self.grads.w_q.scale(scale);
        self.grads.w_k.scale(scale);
        self.grads.w_v.scale(scale);
        self.grads.w_o.scale(scale);
        for v in self.grads.b_q.iter_mut() {
            *v *= scale;
        }
        for v in self.grads.b_k.iter_mut() {
            *v *= scale;
        }
        for v in self.grads.b_v.iter_mut() {
            *v *= scale;
        }
        for v in self.grads.b_o.iter_mut() {
            *v *= scale;
        }
        for v in self.grads.w_i.iter_mut() {
            *v *= scale;
        }
        for v in self.grads.w_f.iter_mut() {
            *v *= scale;
        }
        self.grads.b_i *= scale;
        self.grads.b_f *= scale;
        for v in self.grads.c_init.iter_mut() {
            *v *= scale;
        }
        for v in self.grads.n_init.iter_mut() {
            *v *= scale;
        }
        self.grads.m_init *= scale;
    }

    fn reset_state(&mut self) {
        self.reset();
    }

    fn accumulate_init_grad(&mut self) {
        // Nach der letzten Sequenz sind dc_bptt / dn_bptt / dm_bptt die
        // Gradienten bezüglich der Initialzustände.
        add_vec_in_place(&mut self.grads.c_init, &self.dc_bptt);
        add_vec_in_place(&mut self.grads.n_init, &self.dn_bptt);
        self.grads.m_init += self.dm_bptt;
    }
}
