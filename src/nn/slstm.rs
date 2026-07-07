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
use rand::random_range;

use crate::{
    nn::{
        activations::{log_sigmoid, stable_sigmoid},
        dot, matvec_acc, outer_acc_block, GRAD_BLOCK,
    },
    nn_layer::{DynCache, NnLayer},
    optimizers::{GradMatrix, GradMatrixOps, GradVec, GradVecOps, add_grad_matrix, add_grad_vec},
    saving,
};
use std::{any::Any, io};

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

pub struct SLSTMLayerGrads {
    pub wz: GradMatrix,
    pub wi: GradMatrix,
    pub wf: GradMatrix,
    pub wo: GradMatrix,
    pub bz: GradVec,
    pub bi: GradVec,
    pub bf: GradVec,
    pub bo: GradVec,
    //pub h_init_grad: GradVec,
    //pub c_init_grad: GradVec,
}

impl SLSTMLayerGrads {
    pub fn zeros(rows: usize, h: usize) -> Self {
        Self {
            wz: GradMatrix::zeros(rows, h),
            wi: GradMatrix::zeros(rows, h),
            wf: GradMatrix::zeros(rows, h),
            wo: GradMatrix::zeros(rows, h),
            bz: GradVec::zeros(h),
            bi: GradVec::zeros(h),
            bf: GradVec::zeros(h),
            bo: GradVec::zeros(h),
            // h_init_grad: GradVec::zeros(h),
            // c_init_grad: GradVec::zeros(h),
        }
    }
}

pub struct SLSTMLayer {
    pub input_size: usize,
    pub hidden_size: usize,

    pub wz: Matrix,
    pub wi: Matrix,
    pub wf: Matrix,
    pub wo: Matrix,

    pub bz: Box<[f32]>,
    pub bi: Box<[f32]>,
    pub bf: Box<[f32]>,
    pub bo: Box<[f32]>,

    // Forward state carried across timesteps
    pub h: Box<[f32]>,
    pub c: Box<[f32]>,
    pub n: Box<[f32]>,
    pub m: Box<[f32]>,

    // BPTT gradients flowing t+1 → t
    dh_bptt: Box<[f32]>,
    dc_bptt: Box<[f32]>,
    dn_bptt: Box<[f32]>,

    // ── Learnable initial states (h, c only — n, m start at 0 per paper) ─────
    pub h_init: Box<[f32]>,
    pub c_init: Box<[f32]>,

    pub grads: SLSTMLayerGrads,

    // ── Deferred weight-gradient accumulation ─────────────────────────────
    // backward() stashes (xh, per-gate deltas) per step and folds them into
    // the grad matrices in blocks of GRAD_BLOCK, so each grad matrix is
    // streamed once per block instead of read-modified-written every step.
    // The delta rows double as the backward scratch for the current step.
    pend_len: usize,
    pend_xh: Box<[f32]>, // GRAD_BLOCK × (input+hidden)
    pend_dz: Box<[f32]>, // GRAD_BLOCK × hidden
    pend_di: Box<[f32]>,
    pend_df: Box<[f32]>,
    pend_do: Box<[f32]>,
}

impl SLSTMLayer {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let rows = input_size + hidden_size;
        let scale = (6.0 / rows as f32).sqrt();
        let h_init: Box<[f32]> = vec![0.0; hidden_size].into();
        let c_init: Box<[f32]> = vec![0.0; hidden_size].into();

        Self {
            input_size,
            hidden_size,

            wz: Matrix::random(rows, hidden_size, scale),
            wi: Matrix::random(rows, hidden_size, scale),
            wf: Matrix::random(rows, hidden_size, scale),
            wo: Matrix::random(rows, hidden_size, scale),

            bz: vec![0.0; hidden_size].into(),
            bi: vec![0.0; hidden_size].into(),
            // Forget-gate bias init to a positive value (Jozefowicz et al. 2015;
            // xLSTM paper keeps this convention to avoid very small f_t at start).
            bf: (0..hidden_size).map(|_| random_range(3.0..6.0)).collect(),
            bo: vec![0.0; hidden_size].into(),

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
            pend_len: 0,
            pend_xh: vec![0.0; GRAD_BLOCK * rows].into(),
            pend_dz: vec![0.0; GRAD_BLOCK * hidden_size].into(),
            pend_di: vec![0.0; GRAD_BLOCK * hidden_size].into(),
            pend_df: vec![0.0; GRAD_BLOCK * hidden_size].into(),
            pend_do: vec![0.0; GRAD_BLOCK * hidden_size].into(),
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
        bz: Box<[f32]>,
        bi: Box<[f32]>,
        bf: Box<[f32]>,
        bo: Box<[f32]>,
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
            bz,
            bi,
            bf,
            bo,
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
            pend_len: 0,
            pend_xh: vec![0.0; GRAD_BLOCK * rows].into(),
            pend_dz: vec![0.0; GRAD_BLOCK * hidden_size].into(),
            pend_di: vec![0.0; GRAD_BLOCK * hidden_size].into(),
            pend_df: vec![0.0; GRAD_BLOCK * hidden_size].into(),
            pend_do: vec![0.0; GRAD_BLOCK * hidden_size].into(),
        }
    }

    pub fn forward(&mut self, input: &[f32], cache: &mut SLSTMCache) {
        let h = self.hidden_size;

        // Save prev states for backward
        cache.c_prev.copy_from_slice(&self.c);
        cache.n_prev.copy_from_slice(&self.n);
        cache.m_prev.copy_from_slice(&self.m);

        // Build xh = concat(x, h_{t-1})
        cache.xh[..input.len()].copy_from_slice(input);
        cache.xh[input.len()..].copy_from_slice(&self.h);

        // Pre-activations (one matmul per gate); the bias is the accumulator init.
        cache.zt_pre.copy_from_slice(&self.bz);
        cache.it_pre.copy_from_slice(&self.bi);
        cache.ft_pre.copy_from_slice(&self.bf);
        cache.ot_pre.copy_from_slice(&self.bo);
        matvec_acc(&self.wz, &cache.xh, &mut cache.zt_pre);
        matvec_acc(&self.wi, &cache.xh, &mut cache.it_pre);
        matvec_acc(&self.wf, &cache.xh, &mut cache.ft_pre);
        matvec_acc(&self.wo, &cache.xh, &mut cache.ot_pre);

        // Activations, stabilizer, gates, cell, normalizer, hidden — one pass;
        // the transcendentals keep this loop scalar, so extra passes over the
        // h-sized buffers would only add load/store traffic.
        for j in 0..h {
            let zt = cache.zt_pre[j].tanh();
            let ot = stable_sigmoid(cache.ot_pre[j]);
            let log_f = log_sigmoid(cache.ft_pre[j]);

            // Stabilizer  m_t = max(log f_t + m_{t-1},  ĩ_t)
            let fm = log_f + cache.m_prev[j];
            let m = fm.max(cache.it_pre[j]);

            // Stabilized exponential gates
            let i_prime = (cache.it_pre[j] - m).exp();
            let f_prime = (fm - m).exp();

            let c = f_prime * cache.c_prev[j] + i_prime * zt;
            let n = f_prime * cache.n_prev[j] + i_prime;
            let psi = n.abs().max(1.0);

            cache.zt[j] = zt;
            cache.ot[j] = ot;
            cache.log_f[j] = log_f;
            cache.m[j] = m;
            cache.i_prime[j] = i_prime;
            cache.f_prime[j] = f_prime;
            cache.c[j] = c;
            cache.n[j] = n;
            cache.psi[j] = psi;
            cache.h[j] = ot * c / psi;
        }

        // Propagate persistent state
        self.h.copy_from_slice(&cache.h);
        self.c.copy_from_slice(&cache.c);
        self.n.copy_from_slice(&cache.n);
        self.m.copy_from_slice(&cache.m);
    }

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

        // This step's gate deltas are written straight into the pending block
        // (they double as the scratch the dconcat sweep reads below).
        let slot = self.pend_len;
        self.pend_xh[slot * r..(slot + 1) * r].copy_from_slice(&cache.xh);
        let dz = &mut self.pend_dz[slot * h..(slot + 1) * h];
        let di = &mut self.pend_di[slot * h..(slot + 1) * h];
        let df = &mut self.pend_df[slot * h..(slot + 1) * h];
        let dob = &mut self.pend_do[slot * h..(slot + 1) * h];

        // ── 1–3. Elementwise chain in one pass: output gate, δh split into
        // δc/δn paths (+BPTT from future), gate pre-activation grads, BPTT
        // channels for t-1. Scalar anyway (branch + transcendental), so one
        // pass over the h-sized buffers beats three.
        for j in 0..h {
            let d = delta[j] + self.dh_bptt[j];
            delta[j] = d;
            let psi = cache.psi[j];
            let ot = cache.ot[j];

            //   h = o · (c / ψ)   →   do/dõ = δh · c/ψ · o·(1-o)
            dob[j] = d * (cache.c[j] / psi) * ot * (1.0 - ot);

            // dψ through max(|n|, 1):  only active when |n| > 1
            let dn_from_h = if cache.n[j].abs() > 1.0 {
                let dpsi = d * ot * (-cache.c[j]) / (psi * psi);
                dpsi * cache.n[j].signum()
            } else {
                0.0
            };

            //   c = f'·c_prev + i'·z
            //   n = f'·n_prev + i'
            let dc = d * ot / psi + self.dc_bptt[j];
            let dn = dn_from_h + self.dn_bptt[j];

            let df_prime = dc * cache.c_prev[j] + dn * cache.n_prev[j];
            let di_prime = dc * cache.zt[j] + dn;
            let dz_post = dc * cache.i_prime[j]; // dL/dz (post-tanh)

            // tanh derivative for z
            dz[j] = dz_post * (1.0 - cache.zt[j] * cache.zt[j]);

            // i' = exp(ĩ − m)   →   dĩ = di' · i'     (m treated as constant — see header)
            di[j] = di_prime * cache.i_prime[j];

            // f' = exp(log_f + m_prev − m)
            //   d(log_f)/df̃ = 1 − σ(f̃)
            //   → df̃ = df' · f' · (1 − σ(f̃))
            let sigm_f = stable_sigmoid(cache.ft_pre[j]);
            df[j] = df_prime * cache.f_prime[j] * (1.0 - sigm_f);

            // BPTT channels for t-1
            self.dc_bptt[j] = dc * cache.f_prime[j];
            self.dn_bptt[j] = dn * cache.f_prime[j];
        }

        // ── 4. Bias gradients (per step — h-sized, cheap)
        {
            let g = &mut self.grads;
            for j in 0..h {
                g.bz.vec()[j] += dz[j];
                g.bi.vec()[j] += di[j];
                g.bf.vec()[j] += df[j];
                g.bo.vec()[j] += dob[j];
            }
        }

        // ── 5. dL/d(xh): four lane-accumulated dots per weight row, one sweep
        // over W (read-only — the weight-grad outer products are deferred to
        // flush_grads, so gW is not read-modified-written here every step).
        {
            let wz = self.wz.as_slice();
            let wi = self.wi.as_slice();
            let wf = self.wf.as_slice();
            let wo = self.wo.as_slice();
            let dz = &self.pend_dz[slot * h..(slot + 1) * h];
            let di = &self.pend_di[slot * h..(slot + 1) * h];
            let df = &self.pend_df[slot * h..(slot + 1) * h];
            let dob = &self.pend_do[slot * h..(slot + 1) * h];

            for i in 0..r {
                let off = i * h;
                // Exact-length row slices — elides the bounds checks a flat
                // `[off + j]` index would keep.
                cache.dconcat[i] = dot(&wz[off..off + h], dz)
                    + dot(&wi[off..off + h], di)
                    + dot(&wf[off..off + h], df)
                    + dot(&wo[off..off + h], dob);
            }
        }
        self.dh_bptt
            .copy_from_slice(&cache.dconcat[self.input_size..]);

        self.pend_len += 1;
        if self.pend_len == GRAD_BLOCK {
            self.flush_grads();
        }
    }

    /// Fold the pending (xh, gate-delta) outer products into the weight-grad
    /// matrices. No-op when nothing is pending.
    fn flush_grads(&mut self) {
        let n = self.pend_len;
        if n == 0 {
            return;
        }
        self.pend_len = 0;
        let h = self.hidden_size;
        let r = self.input_size + h;
        let xh = &self.pend_xh;
        let g = &mut self.grads;
        outer_acc_block(g.wz.matrix().as_slice_mut(), xh, &self.pend_dz, n, r, h);
        outer_acc_block(g.wi.matrix().as_slice_mut(), xh, &self.pend_di, n, r, h);
        outer_acc_block(g.wf.matrix().as_slice_mut(), xh, &self.pend_df, n, r, h);
        outer_acc_block(g.wo.matrix().as_slice_mut(), xh, &self.pend_do, n, r, h);
    }

    /// Fold a replica's grads into this layer (data-parallel reduction).
    pub fn add_grads(&mut self, other: &mut Self) {
        self.flush_grads();
        other.flush_grads();
        let g = &mut self.grads;
        let o = &mut other.grads;
        add_grad_matrix(&mut g.wz, &mut o.wz);
        add_grad_matrix(&mut g.wi, &mut o.wi);
        add_grad_matrix(&mut g.wf, &mut o.wf);
        add_grad_matrix(&mut g.wo, &mut o.wo);
        add_grad_vec(&mut g.bz, &mut o.bz);
        add_grad_vec(&mut g.bi, &mut o.bi);
        add_grad_vec(&mut g.bf, &mut o.bf);
        add_grad_vec(&mut g.bo, &mut o.bo);
        // add_grad_vec(&mut g.h_init_grad, &mut o.h_init_grad);
        //add_grad_vec(&mut g.c_init_grad, &mut o.c_init_grad);
    }

    /// Overwrite all weights (incl. the learnable initial states) with
    /// `other`'s (in-place replica refresh).
    pub fn copy_weights(&mut self, other: &Self) {
        self.wz.copy_from(&other.wz);
        self.wi.copy_from(&other.wi);
        self.wf.copy_from(&other.wf);
        self.wo.copy_from(&other.wo);
        self.bz.copy_from_slice(&other.bz);
        self.bi.copy_from_slice(&other.bi);
        self.bf.copy_from_slice(&other.bf);
        self.bo.copy_from_slice(&other.bo);
        self.h_init.copy_from_slice(&other.h_init);
        self.c_init.copy_from_slice(&other.c_init);
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

impl NnLayer for SLSTMLayer {
    //type Cache = SLSTMCache;
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

        saving::write_f32_slice(w, &self.bz)?;
        saving::write_f32_slice(w, &self.bi)?;
        saving::write_f32_slice(w, &self.bf)?;
        saving::write_f32_slice(w, &self.bo)?;

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
        self.flush_grads();
        self.grads.wi.clip();
        self.grads.wo.clip();

        self.grads.wz.apply_to(&mut self.wz, lr);
        self.grads.wi.apply_to(&mut self.wi, lr);
        self.grads.wf.apply_to(&mut self.wf, lr);
        self.grads.wo.apply_to(&mut self.wo, lr);
        self.grads.bz.apply_to(&mut self.bz, lr);
        self.grads.bi.apply_to(&mut self.bi, lr);
        self.grads.bf.apply_to(&mut self.bf, lr);
        self.grads.bo.apply_to(&mut self.bo, lr);

        // self.grads.c_init_grad.apply_to(&mut self.c_init, lr);
        // self.grads.h_init_grad.apply_to(&mut self.h_init, lr);
    }

    fn clear_grads(&mut self) {
        // Pending outer products belong to the grads being discarded.
        self.pend_len = 0;
        self.grads.wz.clear();
        self.grads.wi.clear();
        self.grads.wf.clear();
        self.grads.wo.clear();
        self.grads.bz.clear();
        self.grads.bi.clear();
        self.grads.bf.clear();
        self.grads.bo.clear();
        // self.grads.h_init_grad.clear();
        //self.grads.c_init_grad.clear();
    }

    fn reset_state(&mut self) {
        self.h.fill(0.0);
        self.c.fill(0.0);
        //self.h.copy_from_slice(&self.h_init);
        //self.c.copy_from_slice(&self.c_init);
        self.n.fill(0.0);
        self.m.fill(0.0);
    }
    fn reset_bptt_state(&mut self) {
        // Window seam — fold any partial pending block into the grad matrices.
        self.flush_grads();
        self.dh_bptt.fill(0.0);
        self.dc_bptt.fill(0.0);
        self.dn_bptt.fill(0.0);
    }

    fn accumulate_init_grad(&mut self) {
        // Window seam — fold any partial pending block into the grad matrices.
        self.flush_grads();
        // add_vec_in_place(&mut self.grads.h_init_grad.vec(), &self.dh_bptt);
        // add_vec_in_place(&mut self.grads.c_init_grad.vec(), &self.dc_bptt);
    }

    fn add_grads_from(&mut self, other: &mut dyn NnLayer) {
        let o = other
            .as_any_mut()
            .downcast_mut::<Self>()
            .expect("SLSTMLayer::add_grads_from — replica layer type mismatch");
        self.add_grads(o);
    }

    fn copy_weights_from(&mut self, other: &dyn NnLayer) {
        let o = other
            .as_any()
            .downcast_ref::<Self>()
            .expect("SLSTMLayer::copy_weights_from — replica layer type mismatch");
        self.copy_weights(o);
    }
}
