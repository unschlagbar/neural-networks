use iron_oxide::collections::Matrix;

use crate::layer::{Activation, sigmoid};

pub const CLIP: f32 = 1.0;
const F: usize = 0;
const I: usize = 1;
const C: usize = 2;
const O: usize = 3;

pub const DH: usize = 0;
const DC: usize = 1;

pub struct LSTMLayer {
    pub input_size: usize,
    pub hidden_size: usize,

    pub wf: Matrix,
    pub wi: Matrix,
    pub wc: Matrix,
    pub wo: Matrix,

    pub b: Matrix,
    pub d: Matrix,
}

impl LSTMLayer {
    pub fn random(input_size: usize, hidden_size: usize) -> Self {
        let rows = input_size + hidden_size;
        Self {
            input_size,
            hidden_size,

            wf: Matrix::random(rows, hidden_size, 0.08),
            wi: Matrix::random(rows, hidden_size, 0.08),
            wc: Matrix::random(rows, hidden_size, 0.08),
            wo: Matrix::random(rows, hidden_size, 0.08),

            b: Matrix::zeros(4, hidden_size),
            d: Matrix::zeros(2, hidden_size),
        }
    }

    pub fn validate_cache(&mut self, cache: &LSTMCache) {
        let input_size = self.input_size;
        let hidden_size = self.hidden_size;
        let rows = input_size + hidden_size;

        assert_eq!(cache.states.cols(), hidden_size);
        assert_eq!(cache.states_prev.cols(), hidden_size);
        assert_eq!(cache.xh.len(), rows);
        assert_eq!(cache.ct.len(), hidden_size);
        assert_eq!(cache.ft.len(), hidden_size);
        assert_eq!(cache.it.len(), hidden_size);
        assert_eq!(cache.ot.len(), hidden_size);
    }

    pub fn forward(&mut self, input: &[f32], cache: &mut LSTMCache) {
        #[cfg(debug_assertions)]
        self.validate_cache(cache);

        cache.states_prev.copy_from(&self.d);
        let states_prev = &cache.states_prev;

        // Combine input + previous hidden state
        let xh = &mut cache.xh;
        xh[..input.len()].copy_from_slice(input);
        xh[input.len()..].copy_from_slice(&states_prev[DH]);

        // Compute gates
        self.wf.row_mul(&xh, &mut cache.ft);
        self.wi.row_mul(&xh, &mut cache.it);
        self.wc.row_mul(&xh, &mut cache.ct);
        self.wo.row_mul(&xh, &mut cache.ot);

        let ft = &mut cache.ft;
        let it = &mut cache.it;
        let ct = &mut cache.ct;
        let ot = &mut cache.ot;

        let h = self.hidden_size;

        for j in 0..h {
            ft[j] = sigmoid(ft[j] + self.b[F][j]);
            it[j] = sigmoid(it[j] + self.b[I][j]);
            ct[j] = (ct[j] + self.b[C][j]).tanh();
            ot[j] = sigmoid(ot[j] + self.b[O][j]);
        }

        // Update cell state: c_t = f_t * c_{t-1} + i_t * c~_t
        let prev_c = &states_prev[DC];
        for j in 0..h {
            self.d[DC][j] = ft[j] * prev_c[j] + it[j] * ct[j];
        }

        // Compute hidden state: h_t = o_t * tanh(c_t)
        debug_assert_eq!(ot.len(), self.d[DC].len());
        for (j, &ot) in ot.iter().enumerate() {
            self.d[DH][j] = ot * self.d[DC][j].tanh();
        }

        cache.states.copy_from(&self.d);
    }

    pub fn backwards(
        &mut self,
        cache: &mut LSTMCache,
        delta_h: &[f32],
        dc_next: &mut [f32],
        grads: &mut LSTMLayerGrads,
    ) -> Vec<f32> {
        let h = self.hidden_size;
        let rows = self.input_size + self.hidden_size;

        // 1) prepare: compute tanh(c_t) once (same as original code)
        // Note: caller already used Activation::Tanh.activate in original; keep same order:
        Activation::Tanh.activate(&mut cache.states[DC]);

        let mut ds = vec![0.0; h * 4];
        let (do_, split) = ds.split_at_mut(h);
        let (df, split) = split.split_at_mut(h);
        let (di, dct) = split.split_at_mut(h);

        // 2) dc = dh ⊙ o * (1 - tanh(c)^2) + dc_next
        let dc = dc_next;
        for i in 0..h {
            dc[i] += delta_h[i] * cache.ot[i] * dtanh_from_y(cache.states[DC][i]);
        }

        // 3) do = dh ⊙ tanh(c) * σ'(o)
        for i in 0..h {
            do_[i] = delta_h[i] * cache.states[DC][i] * dsigmoid_from_y(cache.ot[i]);
        }

        // 4) gate grads: df, di, dct
        for i in 0..h {
            df[i] = dc[i] * cache.states_prev[DC][i] * dsigmoid_from_y(cache.ft[i]);
            di[i] = dc[i] * cache.ct[i] * dsigmoid_from_y(cache.it[i]);
            dct[i] = dc[i] * cache.it[i] * dtanh_from_y(cache.ct[i]);
        }

        // 5) accumulate grads WITHOUT allocating
        // grads.w*.rows = rows, cols = h
        // cache.xh.len() == rows
        // Update weight matrices: grads.wf[i][j] += xh[i] * df[j]
        for i in 0..rows {
            let xi = cache.xh[i];
            let wf_row = &mut grads.wf[i];
            let wi_row = &mut grads.wi[i];
            let wc_row = &mut grads.wc[i];
            let wo_row = &mut grads.wo[i];
            for j in 0..h {
                // inline multiply-add beats forming an outer-product Matrix then add_inplace
                wf_row[j] += xi * df[j];
                wi_row[j] += xi * di[j];
                wc_row[j] += xi * dct[j];
                wo_row[j] += xi * do_[j];
            }
        }

        // biases
        for j in 0..h {
            grads.b[0][j] += df[j];
            grads.b[1][j] += di[j];
            grads.b[2][j] += dct[j];
            grads.b[3][j] += do_[j];
        }

        // 6) backprop to inputs+prev-hidden: dconcat = Wf^T * df + Wi^T * di + ...
        // compute row-wise accumulation to keep contiguous memory access on W rows
        let mut dconcat = vec![0.0; rows];
        for (i, dconcat) in dconcat.iter_mut().enumerate() {
            let wf_row = &self.wf[i];
            let wi_row = &self.wi[i];
            let wc_row = &self.wc[i];
            let wo_row = &self.wo[i];
            let mut s = 0.0;
            // sum over columns (hidden units)
            for j in 0..h {
                s += wf_row[j] * df[j];
                s += wi_row[j] * di[j];
                s += wc_row[j] * dct[j];
                s += wo_row[j] * do_[j];
            }
            *dconcat = s;
        }

        // split dx and dh_prev
        //let dh_prev = dconcat.split_off(self.input_size);
        //let dx = dconcat;

        // dc_prev (for previous time-step) = dc ⊙ f_t
        dc.iter_mut().zip(&cache.ft).for_each(|(a, b)| *a *= b);

        dconcat
    }

    // This is the fn for my new dynamic system
    pub fn backward(
        &mut self,
        mut cache: LSTMCache,
        delta_h: &[f32],
        dc_next: &[f32],
        grads: &mut LSTMLayerGrads,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let h = self.hidden_size;
        let rows = self.input_size + self.hidden_size;

        // 1) prepare: compute tanh(c_t) once (same as original code)
        // Note: caller already used Activation::Tanh.activate in original; keep same order:
        Activation::Tanh.activate(&mut cache.states[DC]);

        let mut ds = vec![0.0; h * 4];
        let (do_, split) = ds.split_at_mut(h);
        let (df, split) = split.split_at_mut(h);
        let (di, dct) = split.split_at_mut(h);

        // 2) do = dh ⊙ tanh(c) * σ'(o)
        for i in 0..h {
            do_[i] = delta_h[i] * cache.states[DC][i] * dsigmoid_from_y(cache.ot[i]);
        }

        // 3) dc = dh ⊙ o * (1 - tanh(c)^2) + dc_next
        let mut dc = dc_next.to_vec();
        for i in 0..h {
            dc[i] += delta_h[i] * cache.ot[i] * dtanh_from_y(cache.states[DC][i]);
        }

        // 4) gate grads: df, di, dct
        for i in 0..h {
            df[i] = dc[i] * cache.states_prev[DC][i] * dsigmoid_from_y(cache.ft[i]);
            di[i] = dc[i] * cache.ct[i] * dsigmoid_from_y(cache.it[i]);
            dct[i] = dc[i] * cache.it[i] * dtanh_from_y(cache.ct[i]);
        }

        // 5) accumulate grads WITHOUT allocating outer products:
        // grads.w*.rows = rows, cols = h
        // cache.xh.len() == rows
        // Update weight matrices: grads.wf[i][j] += xh[i] * df[j]
        for i in 0..rows {
            let xi = cache.xh[i];
            let wf_row = &mut grads.wf[i];
            let wi_row = &mut grads.wi[i];
            let wc_row = &mut grads.wc[i];
            let wo_row = &mut grads.wo[i];
            for j in 0..h {
                // inline multiply-add beats forming an outer-product Matrix then add_inplace
                wf_row[j] += xi * df[j];
                wi_row[j] += xi * di[j];
                wc_row[j] += xi * dct[j];
                wo_row[j] += xi * do_[j];
            }
        }

        // biases
        for j in 0..h {
            grads.b[0][j] += df[j];
            grads.b[1][j] += di[j];
            grads.b[2][j] += dct[j];
            grads.b[3][j] += do_[j];
        }

        // 6) backprop to inputs+prev-hidden: dconcat = Wf^T * df + Wi^T * di + ...
        // compute row-wise accumulation to keep contiguous memory access on W rows
        let mut dconcat = vec![0.0; rows];
        for (i, dconcat) in dconcat.iter_mut().enumerate() {
            let wf_row = &self.wf[i];
            let wi_row = &self.wi[i];
            let wc_row = &self.wc[i];
            let wo_row = &self.wo[i];
            let mut s = 0.0;
            // sum over columns (hidden units)
            for j in 0..h {
                s += wf_row[j] * df[j];
                s += wi_row[j] * di[j];
                s += wc_row[j] * dct[j];
                s += wo_row[j] * do_[j];
            }
            *dconcat = s;
        }

        // split dx and dh_prev
        let dh_prev = dconcat.split_off(self.input_size);
        let dx = dconcat;

        // dc_prev (for previous time-step) = dc ⊙ f_t
        let dc_prev = dc.iter().zip(&cache.ft).map(|(a, b)| a * b).collect();

        (dh_prev, dc_prev, dx)
    }

    pub fn make_cache(&self) -> LSTMCache {
        let hidden_size = self.hidden_size;
        let input_size = self.input_size;
        let rows = input_size + hidden_size;

        LSTMCache {
            xh: vec![0.0; rows],
            states_prev: Matrix::zeros(2, hidden_size),
            ft: vec![0.0; hidden_size],
            it: vec![0.0; hidden_size],
            ct: vec![0.0; hidden_size],
            ot: vec![0.0; hidden_size],
            states: Matrix::zeros(2, hidden_size),
        }
    }
}

#[derive(Debug)]
pub struct LSTMCache {
    pub xh: Vec<f32>,

    pub states_prev: Matrix,

    pub ft: Vec<f32>,
    pub it: Vec<f32>,
    pub ct: Vec<f32>,
    pub ot: Vec<f32>,

    pub states: Matrix,
}

pub struct LSTMGrads {
    pub layers: Vec<LSTMLayerGrads>,
    pub wy: Matrix,
    pub by: Box<[f32]>,
}

pub struct LSTMLayerGrads {
    pub wf: Matrix,
    pub wi: Matrix,
    pub wc: Matrix,
    pub wo: Matrix,

    pub b: Matrix,
}

pub fn one_hot(index: usize, size: usize) -> Vec<f32> {
    let mut out = vec![0.0; size];
    out[index] = 1.0;
    out
}

pub fn add_vec_in_place(x: &mut [f32], y: &[f32]) {
    debug_assert_eq!(x.len(), y.len());
    x.iter_mut().zip(y).for_each(|(x, y)| *x += y);
}

pub fn outer(a: &[f32], b: &[f32]) -> Matrix {
    let mut mtx = Matrix::uninit(a.len(), b.len());
    for (x, &a) in a.iter().enumerate() {
        for (y, &b) in b.iter().enumerate() {
            mtx[x][y] = a * b;
        }
    }

    mtx
}

fn dsigmoid_from_y(y: f32) -> f32 {
    y * (1.0 - y)
}

fn dtanh_from_y(y: f32) -> f32 {
    1.0 - y * y
}

pub fn sub_in_place(a: &mut Matrix, b: &Matrix, lr: f32) {
    debug_assert_eq!(
        a.rows(),
        b.rows(),
        "rows do not match, {} to {}",
        a.rows(),
        b.rows()
    );
    debug_assert_eq!(
        a.cols(),
        b.cols(),
        "cols do not match, {} to {}",
        a.cols(),
        b.cols()
    );

    a.as_slice_mut()
        .iter_mut()
        .zip(b.as_slice())
        .for_each(|(a, b)| *a -= lr * b);
}

pub fn sub_vec_in_place(a: &mut [f32], b: &[f32], lr: f32) {
    a.iter_mut().zip(b).for_each(|(a, b)| *a -= lr * b);
}
