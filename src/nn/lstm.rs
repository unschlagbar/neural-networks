use iron_oxide::collections::Matrix;

use crate::{
    nn::{add_vec_in_place, sub_in_place, sub_vec_in_place},
    nn_layer::{DynCache, NnLayer},
    saving,
};
use std::{any::Any, io};

const F: usize = 0;
const I: usize = 1;
const C: usize = 2;
const O: usize = 3;

// ── LSTMCache ─────────────────────────────────────────────────────────────────

/// All activations and scratch buffers needed for one LSTM timestep.
/// Pre-allocated at the start of training; zero dynamic allocation in the hot path.
pub struct LSTMCache {
    /// Needed for `input_grad()` to split `dconcat` correctly.
    pub input_size: usize,
    /// concat(x_t, h_{t-1})
    pub xh: Box<[f32]>,
    /// c_{t-1} (saved for gate grad computation)
    pub c_prev: Box<[f32]>,
    pub ft: Box<[f32]>,
    pub it: Box<[f32]>,
    /// Cell candidate c̃_t (tanh output)
    pub ct: Box<[f32]>,
    pub ot: Box<[f32]>,
    /// h_t output (carries to next layer or timestep)
    pub h: Box<[f32]>,
    /// c_t (tanh is applied)
    pub c: Box<[f32]>,
    /// dconcat layout: [dL/d(x_t) | dh_{t-1}]
    pub dconcat: Box<[f32]>,
}

impl DynCache for LSTMCache {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn output(&self) -> &[f32] {
        &self.h
    }
    /// dL/d(x_t) — the input portion of dconcat.
    fn input_grad(&self) -> &[f32] {
        &self.dconcat[..self.input_size]
    }
}

// ── LSTMLayerGrads ────────────────────────────────────────────────────────────

pub struct LSTMLayerGrads {
    pub wf: Matrix,
    pub wi: Matrix,
    pub wc: Matrix,
    pub wo: Matrix,
    pub b: Matrix,

    pub h_init_grad: Box<[f32]>,
    pub c_init_grad: Box<[f32]>,
}

impl LSTMLayerGrads {
    pub fn zeros(rows: usize, h: usize) -> Self {
        Self {
            wf: Matrix::zeros(rows, h),
            wi: Matrix::zeros(rows, h),
            wc: Matrix::zeros(rows, h),
            wo: Matrix::zeros(rows, h),
            b: Matrix::zeros(4, h),
            h_init_grad: vec![0.0; h].into(),
            c_init_grad: vec![0.0; h].into(),
        }
    }
}

// ── LSTMLayer ─────────────────────────────────────────────────────────────────

pub struct LSTMLayer {
    pub input_size: usize,
    pub hidden_size: usize,

    pub wf: Matrix,
    pub wi: Matrix,
    pub wc: Matrix,
    pub wo: Matrix,
    pub b: Matrix,

    /// Forward hidden state h_t — carried to the next timestep.
    pub h: Box<[f32]>,
    /// Forward cell state c_t — carried to the next timestep.
    pub c: Box<[f32]>,

    /// BPTT gradient dL/dh flowing *back* from t+1 → t.
    pub dh_bptt: Box<[f32]>,
    /// BPTT gradient dL/dc flowing back from t+1 → t (= dc · f_t).
    pub dc_bptt: Box<[f32]>,

    // Learnable initial hidden/cell state (updated by gradients).
    pub h_init: Box<[f32]>,
    pub c_init: Box<[f32]>,

    /// Gradient accumulators, cleared per batch, applied in `sgd_step`.
    pub grads: LSTMLayerGrads,

    // ── backward scratch (no allocation during backward) ──────────────────────
    pub do_: Box<[f32]>,
    pub df: Box<[f32]>,
    pub di: Box<[f32]>,
    pub dct: Box<[f32]>,
}

impl LSTMLayer {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let rows = input_size + hidden_size;

        let scale = (6.0 / rows as f32).sqrt();

        let h_init: Box<[f32]> = vec![0.0; hidden_size].into();
        let c_init: Box<[f32]> = vec![0.0; hidden_size].into();

        let mut b = Matrix::zeros(4, hidden_size);

        b[F].fill(1.0);

        Self {
            input_size,
            hidden_size,
            wf: Matrix::random(rows, hidden_size, scale),
            wi: Matrix::random(rows, hidden_size, scale),
            wc: Matrix::random(rows, hidden_size, scale),
            wo: Matrix::random(rows, hidden_size, scale),
            b,
            h: h_init.clone(),
            c: c_init.clone(),
            dh_bptt: vec![0.0; hidden_size].into(),
            dc_bptt: vec![0.0; hidden_size].into(),
            h_init,
            c_init,
            grads: LSTMLayerGrads::zeros(rows, hidden_size),

            do_: vec![0.0; hidden_size].into(),
            df: vec![0.0; hidden_size].into(),
            di: vec![0.0; hidden_size].into(),
            dct: vec![0.0; hidden_size].into(),
        }
    }

    pub fn from_loaded(
        input_size: usize,
        hidden_size: usize,
        wf: Matrix,
        wi: Matrix,
        wc: Matrix,
        wo: Matrix,
        b: Matrix,
        h_init: Box<[f32]>,
        c_init: Box<[f32]>,
    ) -> Self {
        Self {
            input_size,
            hidden_size,
            wf,
            wi,
            wc,
            wo,
            b,
            h: h_init.clone(),
            c: c_init.clone(),
            h_init,
            c_init,
            dh_bptt: vec![0.0; hidden_size].into(),
            dc_bptt: vec![0.0; hidden_size].into(),
            grads: LSTMLayerGrads::zeros(input_size + hidden_size, hidden_size),

            do_: vec![0.0; hidden_size].into(),
            df: vec![0.0; hidden_size].into(),
            di: vec![0.0; hidden_size].into(),
            dct: vec![0.0; hidden_size].into(),
        }
    }

    pub fn forward(&mut self, input: &[f32], cache: &mut LSTMCache) {
        let h = self.hidden_size;

        cache.c_prev.copy_from_slice(&self.c);

        cache.xh[..input.len()].copy_from_slice(input);
        cache.xh[input.len()..].copy_from_slice(&self.h);

        self.wf.row_mul(&cache.xh, &mut cache.ft);
        self.wi.row_mul(&cache.xh, &mut cache.it);
        self.wc.row_mul(&cache.xh, &mut cache.ct);
        self.wo.row_mul(&cache.xh, &mut cache.ot);

        for j in 0..h {
            cache.ft[j] = sigmoid(cache.ft[j] + self.b[F][j]);
            cache.it[j] = sigmoid(cache.it[j] + self.b[I][j]);
            cache.ct[j] = (cache.ct[j] + self.b[C][j]).tanh();
            cache.ot[j] = sigmoid(cache.ot[j] + self.b[O][j]);
        }

        for j in 0..h {
            self.c[j] = cache.ft[j] * cache.c_prev[j] + cache.it[j] * cache.ct[j];
        }
        for j in 0..h {
            let tanh_c = self.c[j].tanh();
            self.h[j] = cache.ot[j] * tanh_c;
            cache.c[j] = tanh_c;
        }

        cache.h.copy_from_slice(&self.h);
    }

    pub fn backward(&mut self, delta: &mut [f32], cache: &mut LSTMCache) {
        let h = self.hidden_size;
        let r = self.input_size + h;

        for j in 0..h {
            delta[j] += self.dh_bptt[j];
        }

        for i in 0..h {
            self.do_[i] = delta[i] * cache.c[i] * dsigmoid(cache.ot[i]);
        }

        for i in 0..h {
            delta[i] = delta[i] * cache.ot[i] * dtanh(cache.c[i]) + self.dc_bptt[i];
        }

        for i in 0..h {
            let dc = delta[i];
            self.df[i] = dc * cache.c_prev[i] * dsigmoid(cache.ft[i]);
            self.di[i] = dc * cache.ct[i] * dsigmoid(cache.it[i]);
            self.dct[i] = dc * cache.it[i] * dtanh(cache.ct[i]);
        }

        let g = &mut self.grads;
        g.wf.add_outer(&cache.xh, &self.df); // wf += xhᵀ · df
        g.wi.add_outer(&cache.xh, &self.di);
        g.wc.add_outer(&cache.xh, &self.dct);
        g.wo.add_outer(&cache.xh, &self.do_);

        for j in 0..h {
            g.b[F][j] += self.df[j];
            g.b[I][j] += self.di[j];
            g.b[C][j] += self.dct[j];
            g.b[O][j] += self.do_[j];
        }

        for i in 0..r {
            let mut s = 0.0;
            for j in 0..h {
                s += self.wf[i][j] * self.df[j]
                    + self.wi[i][j] * self.di[j]
                    + self.wc[i][j] * self.dct[j]
                    + self.wo[i][j] * self.do_[j];
            }
            cache.dconcat[i] = s;
        }

        self.dh_bptt
            .copy_from_slice(&cache.dconcat[self.input_size..]);
        for i in 0..h {
            self.dc_bptt[i] = delta[i] * cache.ft[i];
        }
    }

    pub fn alloc_cache(&self) -> LSTMCache {
        let h = self.hidden_size;
        let r = self.input_size + h;
        LSTMCache {
            input_size: self.input_size,
            xh: vec![0.0; r].into(),
            c_prev: vec![0.0; h].into(),
            ft: vec![0.0; h].into(),
            it: vec![0.0; h].into(),
            ct: vec![0.0; h].into(),
            ot: vec![0.0; h].into(),
            h: vec![0.0; h].into(),
            c: vec![0.0; h].into(),
            dconcat: vec![0.0; r].into(),
        }
    }
}

// ── impl NnLayer for LSTMLayer ────────────────────────────────────────────────

impl NnLayer for LSTMLayer {
    fn forward(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<LSTMCache>()
            .expect("LSTMLayer::forward — expected LSTMCache");
        LSTMLayer::forward(self, input, c);
    }

    fn backward(&mut self, delta: &mut [f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<LSTMCache>()
            .expect("LSTMLayer::backward — expected LSTMCache");
        LSTMLayer::backward(self, delta, c);
    }

    fn layer_tag(&self) -> u8 {
        0
    }

    fn save(&self, w: &mut dyn io::Write) -> io::Result<()> {
        saving::write_matrix(w, &self.wf)?;
        saving::write_matrix(w, &self.wi)?;
        saving::write_matrix(w, &self.wc)?;
        saving::write_matrix(w, &self.wo)?;
        saving::write_matrix(w, &self.b)?;
        saving::write_f32_slice(w, &self.h_init)?;
        saving::write_f32_slice(w, &self.c_init)
    }

    fn make_cache(&self) -> Box<dyn DynCache> {
        Box::new(LSTMLayer::alloc_cache(self))
    }

    fn input_size(&self) -> usize {
        self.input_size
    }
    fn output_size(&self) -> usize {
        self.hidden_size
    }

    fn apply_grads(&mut self, lr: f32) {
        println!("{:?}", self.b);
        sub_in_place(&mut self.wf, &self.grads.wf, lr);
        sub_in_place(&mut self.wi, &self.grads.wi, lr);
        sub_in_place(&mut self.wc, &self.grads.wc, lr);
        sub_in_place(&mut self.wo, &self.grads.wo, lr);
        sub_vec_in_place(self.b.as_slice_mut(), self.grads.b.as_slice(), lr);
        sub_vec_in_place(&mut self.h_init, &self.grads.h_init_grad, lr);
        sub_vec_in_place(&mut self.c_init, &self.grads.c_init_grad, lr);
    }

    fn clear_grads(&mut self) {
        self.grads.wf.clear();
        self.grads.wi.clear();
        self.grads.wc.clear();
        self.grads.wo.clear();
        self.grads.b.clear();
        self.grads.h_init_grad.fill(0.0);
        self.grads.c_init_grad.fill(0.0);
    }

    fn scale_grads(&mut self, scale: f32) {
        self.grads.wf.scale(scale);
        self.grads.wi.scale(scale);
        self.grads.wc.scale(scale);
        self.grads.wo.scale(scale);
        self.grads.b.scale(scale);
        self.grads.h_init_grad.iter_mut().for_each(|x| *x *= scale);
        self.grads.c_init_grad.iter_mut().for_each(|x| *x *= scale);
    }

    fn reset_state(&mut self) {
        self.c.copy_from_slice(&self.c_init);
        self.h.copy_from_slice(&self.h_init);
    }

    fn zero_bptt_state(&mut self) {
        self.dh_bptt.fill(0.0);
        self.dc_bptt.fill(0.0);
    }

    fn accumulate_init_grad(&mut self) {
        add_vec_in_place(&mut self.grads.h_init_grad, &self.dh_bptt);
        add_vec_in_place(&mut self.grads.c_init_grad, &self.dc_bptt);
    }
}

#[inline]
fn dsigmoid(y: f32) -> f32 {
    y * (1.0 - y)
}
#[inline]
fn dtanh(y: f32) -> f32 {
    1.0 - y * y
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
