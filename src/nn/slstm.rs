// slstm.rs ── sLSTM from xLSTM (Beck et al. 2024, arXiv:2405.04517)
//
// Single-head scalar-memory sLSTM with:
//   • exponential input gate      i_t = exp(ĩ_t)
//   • sigmoid forget gate         f_t = σ(f̃_t)           (log-space internally)
//   • normalizer state            n_t = f'_t·n_{t-1} + i'_t
//   • stabilizer state            m_t = max(log f_t + m_{t-1}, ĩ_t)
//   • stabilized gates            i'_t = exp(ĩ_t − m_t),  f'_t = exp(log f_t + m_{t-1} − m_t)
//   • scalar cell                 c_t = f'_t·c_{t-1} + i'_t·z_t
//   • normalized hidden           h_t = o_t · c_t / max(|n_t|, 1)
//
// Gradient treatment of the stabilizer follows the paper / reference impl:
// m_t is treated as a constant w.r.t. backprop (like the max in softmax).
// For |n_t| > 1 this is exact because the exp(−m_t) factor cancels in c_t/n_t.
// For |n_t| < 1 it's a standard approximation that works well in practice.
//
// Code style mirrors lstm.rs: concat-trick xh=[x, h_{t-1}], one matrix per gate,
// gradient accumulators inside the layer, scratch buffers pre-allocated.

use iron_oxide::collections::Matrix;

use crate::{
    nn::{add_vec_in_place, sub_in_place, sub_vec_in_place},
    nn_layer::{DynCache, NnLayer},
    saving,
};
use std::{any::Any, io};

// Gate row indices inside the bias matrix `b` of shape (4, hidden_size).
const Z: usize = 0; // cell input (z̃)
const I: usize = 1; // input gate (ĩ)
const G_F: usize = 2; // forget gate (f̃)
const O: usize = 3; // output gate (õ)

const CLIP: f32 = 10.0;

// ── SLSTMCache ────────────────────────────────────────────────────────────────

/// All per-timestep activations + scratch needed for backward.
/// Pre-allocated at the start of training; zero dynamic allocation in the hot path.
pub struct SLSTMCache {
    pub input_size: usize,

    /// concat(x_t, h_{t-1})
    pub xh: Box<[f32]>,

    /// Previous states (saved for backward).
    pub c_prev: Box<[f32]>,
    pub n_prev: Box<[f32]>,
    pub m_prev: Box<[f32]>,

    // Pre-activations (linear outputs before any non-linearity).
    pub zt_pre: Box<[f32]>, // z̃
    pub it_pre: Box<[f32]>, // ĩ       (log-space input gate)
    pub ft_pre: Box<[f32]>, // f̃
    pub ot_pre: Box<[f32]>, // õ

    // Post-activations.
    pub zt: Box<[f32]>,    // tanh(z̃)
    pub ot: Box<[f32]>,    // σ(õ)
    pub log_f: Box<[f32]>, // log σ(f̃)

    // Stabilized exponential gates.
    pub i_prime: Box<[f32]>, // exp(ĩ − m)
    pub f_prime: Box<[f32]>, // exp(log_f + m_prev − m)

    // States at time t.
    m: Box<[f32]>,
    c: Box<[f32]>,
    n: Box<[f32]>,
    psi: Box<[f32]>, // denominator max(|n|, 1)
    pub h: Box<[f32]>,

    /// dL/d(concat(x_t, h_{t-1})).  First `input_size` entries = dx.
    pub dconcat: Box<[f32]>,
}

impl DynCache for SLSTMCache {
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
        &self.dconcat[..self.input_size]
    }
}

// ── SLSTMLayerGrads ───────────────────────────────────────────────────────────

pub struct SLSTMLayerGrads {
    pub wz: Matrix,
    pub wi: Matrix,
    pub wf: Matrix,
    pub wo: Matrix,
    pub b: Matrix, // (4, h)

    pub h_init_grad: Box<[f32]>,
    pub c_init_grad: Box<[f32]>,
}

impl SLSTMLayerGrads {
    pub fn zeros(rows: usize, h: usize) -> Self {
        Self {
            wz: Matrix::zeros(rows, h),
            wi: Matrix::zeros(rows, h),
            wf: Matrix::zeros(rows, h),
            wo: Matrix::zeros(rows, h),
            b: Matrix::zeros(4, h),
            h_init_grad: vec![0.0; h].into(),
            c_init_grad: vec![0.0; h].into(),
        }
    }
}

// ── SLSTMLayer ────────────────────────────────────────────────────────────────

pub struct SLSTMLayer {
    pub input_size: usize,
    pub hidden_size: usize,

    pub wz: Matrix,
    pub wi: Matrix,
    pub wf: Matrix,
    pub wo: Matrix,
    pub b: Matrix,

    // ── Forward state carried across timesteps ────────────────────────────────
    pub h: Box<[f32]>,
    pub c: Box<[f32]>,
    pub n: Box<[f32]>,
    pub m: Box<[f32]>,

    // ── BPTT gradients flowing t+1 → t ───────────────────────────────────────
    dh_bptt: Box<[f32]>,
    dc_bptt: Box<[f32]>,
    dn_bptt: Box<[f32]>,

    // ── Learnable initial states (h, c only — n, m start at 0 per paper) ─────
    pub h_init: Box<[f32]>,
    pub c_init: Box<[f32]>,

    pub grads: SLSTMLayerGrads,

    // ── backward scratch (no allocation during backward) ──────────────────────
    pub dz: Box<[f32]>,
    pub di_pre: Box<[f32]>,
    pub df_pre: Box<[f32]>,
    pub do_pre: Box<[f32]>,
    pub dc_scratch: Box<[f32]>,
    pub dn_scratch: Box<[f32]>,
}

impl SLSTMLayer {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let rows = input_size + hidden_size;
        let scale = (6.0 / rows as f32).sqrt();

        // Forget-gate bias init to a positive value (Jozefowicz et al. 2015;
        // xLSTM paper keeps this convention to avoid very small f_t at start).
        let mut b = Matrix::zeros(4, hidden_size);
        b[G_F].fill(1.0);
        //b[I].fill(-10.0);

        let h_init: Box<[f32]> = vec![0.0; hidden_size].into();
        let c_init: Box<[f32]> = vec![0.0; hidden_size].into();

        Self {
            input_size,
            hidden_size,
            wz: Matrix::random(rows, hidden_size, scale),
            wi: Matrix::random(rows, hidden_size, scale),
            wf: Matrix::random(rows, hidden_size, scale),
            wo: Matrix::random(rows, hidden_size, scale),
            b,
            h: h_init.clone(),
            c: c_init.clone(),
            n: vec![0.0; hidden_size].into(),
            m: vec![0.0; hidden_size].into(),
            dh_bptt: vec![0.0; hidden_size].into(),
            dc_bptt: vec![0.0; hidden_size].into(),
            dn_bptt: vec![0.0; hidden_size].into(),
            h_init,
            c_init,
            grads: SLSTMLayerGrads::zeros(rows, hidden_size),
            dz: vec![0.0; hidden_size].into(),
            di_pre: vec![0.0; hidden_size].into(),
            df_pre: vec![0.0; hidden_size].into(),
            do_pre: vec![0.0; hidden_size].into(),
            dc_scratch: vec![0.0; hidden_size].into(),
            dn_scratch: vec![0.0; hidden_size].into(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_loaded(
        input_size: usize,
        hidden_size: usize,
        wz: Matrix,
        wi: Matrix,
        wf: Matrix,
        wo: Matrix,
        b: Matrix,
        h_init: Box<[f32]>,
        c_init: Box<[f32]>,
    ) -> Self {
        let rows = input_size + hidden_size;
        Self {
            input_size,
            hidden_size,
            wz,
            wi,
            wf,
            wo,
            b,
            h: h_init.clone(),
            c: c_init.clone(),
            n: vec![0.0; hidden_size].into(),
            m: vec![0.0; hidden_size].into(),
            dh_bptt: vec![0.0; hidden_size].into(),
            dc_bptt: vec![0.0; hidden_size].into(),
            dn_bptt: vec![0.0; hidden_size].into(),
            h_init,
            c_init,
            grads: SLSTMLayerGrads::zeros(rows, hidden_size),
            dz: vec![0.0; hidden_size].into(),
            di_pre: vec![0.0; hidden_size].into(),
            df_pre: vec![0.0; hidden_size].into(),
            do_pre: vec![0.0; hidden_size].into(),
            dc_scratch: vec![0.0; hidden_size].into(),
            dn_scratch: vec![0.0; hidden_size].into(),
        }
    }

    // ── forward ───────────────────────────────────────────────────────────────

    pub fn forward(&mut self, input: &[f32], cache: &mut SLSTMCache) {
        let h = self.hidden_size;

        // Save prev states for backward
        cache.c_prev.copy_from_slice(&self.c);
        cache.n_prev.copy_from_slice(&self.n);
        cache.m_prev.copy_from_slice(&self.m);

        // Build xh = concat(x, h_{t-1})
        cache.xh[..input.len()].copy_from_slice(input);
        cache.xh[input.len()..].copy_from_slice(&self.h);

        // Pre-activations (one matmul per gate)
        self.wz.row_mul(&cache.xh, &mut cache.zt_pre);
        self.wi.row_mul(&cache.xh, &mut cache.it_pre);
        self.wf.row_mul(&cache.xh, &mut cache.ft_pre);
        self.wo.row_mul(&cache.xh, &mut cache.ot_pre);

        for j in 0..h {
            cache.zt_pre[j] += self.b[Z][j];
            cache.it_pre[j] += self.b[I][j];
            cache.ft_pre[j] += self.b[G_F][j];
            cache.ot_pre[j] += self.b[O][j];
        }

        // Activations
        for j in 0..h {
            cache.zt[j] = cache.zt_pre[j].tanh();
            cache.ot[j] = stable_sigmoid(cache.ot_pre[j]);
            cache.log_f[j] = log_sigmoid(cache.ft_pre[j]);
        }

        // Stabilizer  m_t = max(log f_t + m_{t-1},  ĩ_t)
        for j in 0..h {
            let a = cache.log_f[j] + cache.m_prev[j];
            let b = cache.it_pre[j];
            cache.m[j] = a.max(b);
        }

        // Stabilized exponential gates
        for j in 0..h {
            cache.i_prime[j] = (cache.it_pre[j] - cache.m[j]).exp();
            cache.f_prime[j] = (cache.log_f[j] + cache.m_prev[j] - cache.m[j]).exp();
        }

        // Cell, normalizer, hidden
        for j in 0..h {
            cache.c[j] = cache.f_prime[j] * cache.c_prev[j] + cache.i_prime[j] * cache.zt[j];
            cache.n[j] = cache.f_prime[j] * cache.n_prev[j] + cache.i_prime[j];
            cache.psi[j] = cache.n[j].abs().max(1.0);
            cache.h[j] = cache.ot[j] * cache.c[j] / cache.psi[j];
        }

        // Propagate persistent state
        self.h.copy_from_slice(&cache.h);
        self.c.copy_from_slice(&cache.c);
        self.n.copy_from_slice(&cache.n);
        self.m.copy_from_slice(&cache.m);
    }

    // ── backward ──────────────────────────────────────────────────────────────
    //
    // Incoming `delta` = dL/dh_t (already combined with dh_bptt from t+1 by Sequential).
    //
    // Writes:
    //   weight/bias gradients into self.grads
    //   dL/dx into cache.dconcat[..input_size]   (exposed via input_grad())
    //   dL/dh_{t-1} into self.dh_bptt            (read by bptt_hidden_grad())
    //   dc_bptt, dn_bptt updated for next iter   (layer-internal BPTT channels)

    pub fn backward(&mut self, delta: &mut [f32], cache: &mut SLSTMCache) {
        let h = self.hidden_size;
        let r = self.input_size + h;

        for j in 0..h {
            delta[j] += self.dh_bptt[j];
        }

        // ── 1. Output gate ────────────────────────────────────────────────────
        //   h = o · (c / ψ)   →   do/dõ = δh · c/ψ · o·(1-o)
        for j in 0..h {
            self.do_pre[j] =
                delta[j] * (cache.c[j] / cache.psi[j]) * cache.ot[j] * (1.0 - cache.ot[j]);
        }

        // ── 2. Split δh into δc and δn paths, add BPTT from future ────────────
        for j in 0..h {
            let psi = cache.psi[j];
            let dc_from_h = delta[j] * cache.ot[j] / psi;

            // dψ through max(|n|, 1):  only active when |n| > 1
            let dn_from_h = if cache.n[j].abs() > 1.0 {
                let dpsi = delta[j] * cache.ot[j] * (-cache.c[j]) / (psi * psi);
                dpsi * cache.n[j].signum()
            } else {
                0.0
            };

            self.dc_scratch[j] = dc_from_h + self.dc_bptt[j];
            self.dn_scratch[j] = dn_from_h + self.dn_bptt[j];
        }

        // ── 3. Gradients w.r.t. stabilized gates + z, and BPTT for c, n ───────
        //   c = f'·c_prev + i'·z
        //   n = f'·n_prev + i'
        for j in 0..h {
            let dc = self.dc_scratch[j];
            let dn = self.dn_scratch[j];

            let df_prime = dc * cache.c_prev[j] + dn * cache.n_prev[j];
            let di_prime = dc * cache.zt[j] + dn;
            let dz_post = dc * cache.i_prime[j]; // dL/dz (post-tanh)

            // tanh derivative for z
            self.dz[j] = dz_post * (1.0 - cache.zt[j] * cache.zt[j]);

            // i' = exp(ĩ − m)   →   dĩ = di' · i'     (m treated as constant — see header)
            self.di_pre[j] = di_prime * cache.i_prime[j];

            // f' = exp(log_f + m_prev − m)
            //   d(log_f)/df̃ = 1 − σ(f̃)
            //   → df̃ = df' · f' · (1 − σ(f̃))
            let sigm_f = stable_sigmoid(cache.ft_pre[j]);
            self.df_pre[j] = df_prime * cache.f_prime[j] * (1.0 - sigm_f);

            // BPTT channels for t-1
            self.dc_bptt[j] = dc * cache.f_prime[j];
            self.dn_bptt[j] = dn * cache.f_prime[j];
        }

        // ── 4. Weight and bias gradients ──────────────────────────────────────
        let g = &mut self.grads;
        g.wz.add_outer(&cache.xh, &self.dz);
        g.wi.add_outer(&cache.xh, &self.di_pre);
        g.wf.add_outer(&cache.xh, &self.df_pre);
        g.wo.add_outer(&cache.xh, &self.do_pre);

        for j in 0..h {
            g.b[Z][j] += self.dz[j];
            g.b[I][j] += self.di_pre[j];
            g.b[G_F][j] += self.df_pre[j];
            g.b[O][j] += self.do_pre[j];
        }

        // ── 5. dL/d(xh) for layers below + BPTT to h_{t-1} ────────────────────
        for i in 0..r {
            let mut s = 0.0;
            for j in 0..h {
                s += self.wz[i][j] * self.dz[j]
                    + self.wi[i][j] * self.di_pre[j]
                    + self.wf[i][j] * self.df_pre[j]
                    + self.wo[i][j] * self.do_pre[j];
            }
            cache.dconcat[i] = s;
        }
        self.dh_bptt
            .copy_from_slice(&cache.dconcat[self.input_size..]);
    }

    pub fn alloc_cache(&self) -> SLSTMCache {
        let h = self.hidden_size;
        let r = self.input_size + h;
        SLSTMCache {
            input_size: self.input_size,
            xh: vec![0.0; r].into(),
            c_prev: vec![0.0; h].into(),
            n_prev: vec![0.0; h].into(),
            m_prev: vec![0.0; h].into(),
            zt_pre: vec![0.0; h].into(),
            it_pre: vec![0.0; h].into(),
            ft_pre: vec![0.0; h].into(),
            ot_pre: vec![0.0; h].into(),
            zt: vec![0.0; h].into(),
            ot: vec![0.0; h].into(),
            log_f: vec![0.0; h].into(),
            i_prime: vec![0.0; h].into(),
            f_prime: vec![0.0; h].into(),
            m: vec![0.0; h].into(),
            c: vec![0.0; h].into(),
            n: vec![0.0; h].into(),
            psi: vec![0.0; h].into(),
            h: vec![0.0; h].into(),
            dconcat: vec![0.0; r].into(),
        }
    }
}

// ── impl NnLayer for SLSTMLayer ───────────────────────────────────────────────

impl NnLayer for SLSTMLayer {
    fn forward(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<SLSTMCache>()
            .expect("SLSTMLayer::forward — expected SLSTMCache");
        SLSTMLayer::forward(self, input, c);
    }

    fn backward(&mut self, delta: &mut [f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<SLSTMCache>()
            .expect("SLSTMLayer::backward — expected SLSTMCache");
        SLSTMLayer::backward(self, delta, c);
    }

    fn layer_tag(&self) -> u8 {
        5
    }

    fn save(&self, w: &mut dyn io::Write) -> io::Result<()> {
        saving::write_matrix(w, &self.wz)?;
        saving::write_matrix(w, &self.wi)?;
        saving::write_matrix(w, &self.wf)?;
        saving::write_matrix(w, &self.wo)?;
        saving::write_matrix(w, &self.b)?;
        saving::write_f32_slice(w, &self.h_init)?;
        saving::write_f32_slice(w, &self.c_init)
    }

    fn make_cache(&self) -> Box<dyn DynCache> {
        Box::new(SLSTMLayer::alloc_cache(self))
    }

    fn input_size(&self) -> usize {
        self.input_size
    }
    fn output_size(&self) -> usize {
        self.hidden_size
    }

    fn apply_grads(&mut self, lr: f32) {
        self.grads.wi.clip(-CLIP, CLIP);
        self.grads.wf.clip(-CLIP, CLIP);

        self.grads.wz.clip(-CLIP, CLIP);
        self.grads.wo.clip(-CLIP, CLIP);

        sub_in_place(&mut self.wz, &self.grads.wz, lr);
        sub_in_place(&mut self.wi, &self.grads.wi, lr);
        sub_in_place(&mut self.wf, &self.grads.wf, lr);
        sub_in_place(&mut self.wo, &self.grads.wo, lr);
        sub_vec_in_place(self.b.as_slice_mut(), self.grads.b.as_slice(), lr);
        sub_vec_in_place(&mut self.h_init, &self.grads.h_init_grad, lr);
        sub_vec_in_place(&mut self.c_init, &self.grads.c_init_grad, lr);
    }

    fn clear_grads(&mut self) {
        self.grads.wz.clear();
        self.grads.wi.clear();
        self.grads.wf.clear();
        self.grads.wo.clear();
        self.grads.b.clear();
        self.grads.h_init_grad.fill(0.0);
        self.grads.c_init_grad.fill(0.0);
    }

    fn scale_grads(&mut self, scale: f32) {
        self.grads.wz.scale(scale);
        self.grads.wi.scale(scale);
        self.grads.wf.scale(scale);
        self.grads.wo.scale(scale);
        self.grads.b.scale(scale);
        self.grads.h_init_grad.iter_mut().for_each(|x| *x *= scale);
        self.grads.c_init_grad.iter_mut().for_each(|x| *x *= scale);
    }

    fn reset_state(&mut self) {
        self.h.copy_from_slice(&self.h_init);
        self.c.copy_from_slice(&self.c_init);
        self.n.fill(0.0);
        self.m.fill(0.0);
    }
    fn zero_bptt_state(&mut self) {
        self.dh_bptt.fill(0.0);
        self.dc_bptt.fill(0.0);
        self.dn_bptt.fill(0.0);
    }

    fn accumulate_init_grad(&mut self) {
        add_vec_in_place(&mut self.grads.h_init_grad, &self.dh_bptt);
        add_vec_in_place(&mut self.grads.c_init_grad, &self.dc_bptt);
    }
}

// ── numerically stable helpers ────────────────────────────────────────────────

#[inline]
fn stable_sigmoid(x: f32) -> f32 {
    // Avoids overflow of exp(-x) for large negative x.
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
fn log_sigmoid(x: f32) -> f32 {
    // log σ(x) = −softplus(−x), computed stably.
    //   x ≥ 0 :  −ln(1 + e^{-x})
    //   x < 0 :   x − ln(1 + e^{x})
    if x >= 0.0 {
        -((-x).exp() + 1.0).ln()
    } else {
        x - (x.exp() + 1.0).ln()
    }
}
