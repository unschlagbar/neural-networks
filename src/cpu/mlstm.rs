//! Batched multi-head mLSTM cell (xLSTM, Beck et al. 2024).

use rand::random_range;
use rayon::prelude::*;

use crate::cpu::ops;
use crate::cpu::optim::{AdamCfg, AdamState, ensure_states};
use crate::tensor::{Tensor, gemm};

const EPS: f32 = 1e-6;

/// Steps between saved `C` states; backward recomputes the ones in between.
const SEGMENT: usize = 128;

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
        -(1.0 + (-x).exp()).ln()
    } else {
        x - (1.0 + x.exp()).ln()
    }
}

/// Lane-accumulated dot product. `f32` addition doesn't reassociate, so a plain
/// `iter().sum()` compiles to a serial dependency chain; the 8 independent lanes
/// let the vectorizer emit FMA.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut acc = [0.0f32; 8];
    let chunks = n / 8;
    for c in 0..chunks {
        let o = c * 8;
        for l in 0..8 {
            acc[l] = a[o + l].mul_add(b[o + l], acc[l]);
        }
    }
    let mut s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for i in chunks * 8..n {
        s += a[i] * b[i];
    }
    s
}

/// `x·W + bias` over `rows` rows, times `scale`, optionally through σ.
fn project(x: &[f32], rows: usize, inp: usize, w: &Tensor, bias: &Tensor, scale: f32, sigmoid: bool) -> Tensor {
    let out_w = w.cols();
    let mut out = Tensor::zeros(&[rows, out_w]);
    for row in out.data.chunks_mut(out_w) {
        row.copy_from_slice(&bias.data);
    }
    gemm::gemm_nn(rows, inp, out_w, x, &w.data, &mut out.data, 1.0);
    if scale != 1.0 || sigmoid {
        for o in out.data.iter_mut() {
            *o *= scale;
            if sigmoid {
                *o = stable_sigmoid(*o);
            }
        }
    }
    out
}

/// What one `(batch row, head)` recurrence saves for backward, time-major.
#[derive(Default)]
struct HeadCache {
    log_f: Vec<f32>,
    i_prime: Vec<f32>,
    f_prime: Vec<f32>,
    nq: Vec<f32>,
    psi: Vec<f32>,
    /// The lower clamp of ψ, `exp(−m)`: backward needs to know which side of the
    /// `max` won, and `m` is overwritten by the next step.
    psi_floor: Vec<f32>,
    /// `n_t`, `[T, dqk]`
    n: Vec<f32>,
    /// `C_t·q_t`, `[T, dhv]`
    cq: Vec<f32>,
    /// `C` before each segment's first step, `[segments, dhv·dqk]`
    ckpt: Vec<f32>,
}

/// Shapes and shared inputs of one recurrence sweep.
struct Sweep<'a> {
    t: usize,
    heads: usize,
    dqk: usize,
    dhv: usize,
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    i_pre: &'a [f32],
    f_pre: &'a [f32],
    reset: &'a [bool],
}

impl Sweep<'_> {
    fn d_qk(&self) -> usize {
        self.heads * self.dqk
    }
    fn d(&self) -> usize {
        self.heads * self.dhv
    }

    /// `C_t = f'·C_prev + i'·v⊗k` into `c`, the forward's exact arithmetic.
    fn advance_c(&self, c: &mut [f32], prev: &[f32], ip: f32, fp: f32, v_h: &[f32], k_h: &[f32]) {
        let dqk = self.dqk;
        for i in 0..self.dhv {
            let ivi = ip * v_h[i];
            let row = i * dqk;
            for j in 0..dqk {
                c[row + j] = fp.mul_add(prev[row + j], ivi * k_h[j]);
            }
        }
    }

    fn forward(&self, bh: usize, hc: &mut HeadCache) {
        let (t, heads, dqk, dhv) = (self.t, self.heads, self.dqk, self.dhv);
        let (b, hd) = (bh / heads, bh % heads);
        let (d_qk, d) = (self.d_qk(), self.d());
        let cd = dhv * dqk;
        for buf in [&mut hc.log_f, &mut hc.i_prime, &mut hc.f_prime, &mut hc.nq, &mut hc.psi, &mut hc.psi_floor] {
            buf.resize(t, 0.0);
        }
        hc.n.resize(t * dqk, 0.0);
        hc.cq.resize(t * dhv, 0.0);
        hc.ckpt.resize(t.div_ceil(SEGMENT) * cd, 0.0);

        let mut c = vec![0.0; cd];
        let mut prev = vec![0.0; cd];
        let mut n = vec![0.0; dqk];
        let mut m = 0.0;
        for s in 0..t {
            if s % SEGMENT == 0 {
                let seg = s / SEGMENT;
                hc.ckpt[seg * cd..(seg + 1) * cd].copy_from_slice(&c);
            }
            if self.reset[s] {
                c.fill(0.0);
                n.fill(0.0);
                m = 0.0;
            }
            let r = b * t + s;
            let i_pre = self.i_pre[r * heads + hd];
            let lf = log_sigmoid(self.f_pre[r * heads + hd]);
            let mh = (lf + m).max(i_pre);
            let ip = (i_pre - mh).exp();
            let fp = (lf + m - mh).exp();
            hc.log_f[s] = lf;
            hc.i_prime[s] = ip;
            hc.f_prime[s] = fp;
            m = mh;

            let q_h = &self.q[r * d_qk + hd * dqk..r * d_qk + (hd + 1) * dqk];
            let k_h = &self.k[r * d_qk + hd * dqk..r * d_qk + (hd + 1) * dqk];
            let v_h = &self.v[r * d + hd * dhv..r * d + (hd + 1) * dhv];

            for j in 0..dqk {
                n[j] = fp.mul_add(n[j], ip * k_h[j]);
            }
            hc.n[s * dqk..(s + 1) * dqk].copy_from_slice(&n);
            let nqv = dot(&n, q_h);
            hc.nq[s] = nqv;
            let floor = (-mh).exp();
            hc.psi_floor[s] = floor;
            hc.psi[s] = nqv.abs().max(floor);

            std::mem::swap(&mut c, &mut prev);
            self.advance_c(&mut c, &prev, ip, fp, v_h, k_h);
            for i in 0..dhv {
                hc.cq[s * dhv + i] = dot(&c[i * dqk..(i + 1) * dqk], q_h);
            }
        }
    }

    /// Backward of one recurrence: `d_ht` is the gradient w.r.t. ỹ (`[B·T, d]`),
    /// the outputs are this head's `dq`, `dk` (before the 1/√dqk), `dv`, `dĩ`, `df̃`.
    fn backward(&self, bh: usize, hc: &HeadCache, d_ht: &[f32], g: &mut HeadGrads) {
        let (t, heads, dqk, dhv) = (self.t, self.heads, self.dqk, self.dhv);
        let (b, hd) = (bh / heads, bh % heads);
        let (d_qk, d) = (self.d_qk(), self.d());
        let cd = dhv * dqk;
        g.dq.resize(t * dqk, 0.0);
        g.dk.resize(t * dqk, 0.0);
        g.dv.resize(t * dhv, 0.0);
        g.di.resize(t, 0.0);
        g.df.resize(t, 0.0);

        let zeros = vec![0.0; cd.max(dqk)];
        let mut dc = vec![0.0; cd];
        let mut dn = vec![0.0; dqk];
        let mut tmp = vec![0.0; dqk];
        let mut seg_c = vec![0.0; SEGMENT.min(t) * cd];

        for seg in (0..t.div_ceil(SEGMENT)).rev() {
            let (s0, s1) = (seg * SEGMENT, ((seg + 1) * SEGMENT).min(t));
            let ckpt = &hc.ckpt[seg * cd..(seg + 1) * cd];
            for s in s0..s1 {
                let r = b * t + s;
                let k_h = &self.k[r * d_qk + hd * dqk..r * d_qk + (hd + 1) * dqk];
                let v_h = &self.v[r * d + hd * dhv..r * d + (hd + 1) * dhv];
                let (done, rest) = seg_c.split_at_mut((s - s0) * cd);
                let prev: &[f32] = if self.reset[s] {
                    &zeros[..cd]
                } else if s == s0 {
                    ckpt
                } else {
                    &done[(s - s0 - 1) * cd..]
                };
                self.advance_c(&mut rest[..cd], prev, hc.i_prime[s], hc.f_prime[s], v_h, k_h);
            }

            for s in (s0..s1).rev() {
                let r = b * t + s;
                let q_h = &self.q[r * d_qk + hd * dqk..r * d_qk + (hd + 1) * dqk];
                let k_h = &self.k[r * d_qk + hd * dqk..r * d_qk + (hd + 1) * dqk];
                let v_h = &self.v[r * d + hd * dhv..r * d + (hd + 1) * dhv];
                let cq_h = &hc.cq[s * dhv..(s + 1) * dhv];
                let dht_h = &d_ht[r * d + hd * dhv..r * d + (hd + 1) * dhv];
                let c_t = &seg_c[(s - s0) * cd..(s - s0 + 1) * cd];
                let first = self.reset[s] || s == 0;
                let c_prev: &[f32] = if first {
                    &zeros[..cd]
                } else if s == s0 {
                    ckpt
                } else {
                    &seg_c[(s - s0 - 1) * cd..(s - s0) * cd]
                };
                let n_h = &hc.n[s * dqk..(s + 1) * dqk];
                let n_prev: &[f32] = if first { &zeros[..dqk] } else { &hc.n[(s - 1) * dqk..s * dqk] };

                let inv_psi = 1.0 / hc.psi[s];
                let dpsi = -(inv_psi * inv_psi) * dot(dht_h, cq_h);
                let dnq = if hc.nq[s].abs() > hc.psi_floor[s] {
                    hc.nq[s].signum() * dpsi
                } else {
                    0.0
                };

                let ip = hc.i_prime[s];
                let fp = hc.f_prime[s];
                let mut df_prime = 0.0;
                let mut di_prime = 0.0;

                let dq_h = &mut g.dq[s * dqk..(s + 1) * dqk];
                let dk_h = &mut g.dk[s * dqk..(s + 1) * dqk];
                for j in 0..dqk {
                    dq_h[j] = dnq * n_h[j];
                    dk_h[j] = 0.0;
                    dn[j] += dnq * q_h[j];
                }

                for i in 0..dhv {
                    let dh_over_psi = dht_h[i] * inv_psi;
                    let v_i = v_h[i];
                    let row = i * dqk;
                    let dcb = &mut dc[row..row + dqk];
                    for j in 0..dqk {
                        tmp[j] = dh_over_psi.mul_add(q_h[j], dcb[j]);
                    }
                    let row_df = dot(&tmp, &c_prev[row..row + dqk]);
                    let row_dv = dot(&tmp, k_h);
                    let c_row = &c_t[row..row + dqk];
                    for j in 0..dqk {
                        dcb[j] = tmp[j] * fp;
                        dq_h[j] = dh_over_psi.mul_add(c_row[j], dq_h[j]);
                        dk_h[j] = tmp[j].mul_add(v_i, dk_h[j]);
                    }
                    df_prime += row_df;
                    di_prime += v_i * row_dv;
                    g.dv[s * dhv + i] = ip * row_dv;
                }

                df_prime += dot(&dn, n_prev);
                di_prime += dot(&dn, k_h);
                for j in 0..dqk {
                    dk_h[j] = ip * (dk_h[j] + dn[j]);
                    dn[j] *= fp;
                }

                g.di[s] = di_prime * ip;
                let sigm_f = hc.log_f[s].exp();
                g.df[s] = df_prime * fp * (1.0 - sigm_f);

                if self.reset[s] {
                    dc.fill(0.0);
                    dn.fill(0.0);
                }
            }
        }
    }
}

/// One recurrence's gradients, time-major.
#[derive(Default)]
struct HeadGrads {
    dq: Vec<f32>,
    dk: Vec<f32>,
    dv: Vec<f32>,
    di: Vec<f32>,
    df: Vec<f32>,
}

/// Everything the forward saves for backward.
#[derive(Default)]
struct Saved {
    b: usize,
    t: usize,
    /// input `[N, in]`
    x: Vec<f32>,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    /// σ(o) `[N, d]`
    o: Tensor,
    heads: Vec<HeadCache>,
    /// head-norm `x̂` `[N, d]` and its `1/rms` per (row, head)
    x_hat: Tensor,
    inv_rms: Vec<f32>,
    /// `o ⊙ ŷ` `[N, d]`, the output projection's input
    hconcat: Tensor,
}

pub struct MLstm {
    pub input_size: usize,
    pub d: usize,
    pub heads: usize,
    pub dqk: usize,
    pub dhv: usize,
    inv_sqrt_dqk: f32,

    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,
    pub wi: Tensor,
    pub wf: Tensor,
    pub bq: Tensor,
    pub bk: Tensor,
    pub bv: Tensor,
    pub bo: Tensor,
    pub bi: Tensor,
    pub bf: Tensor,
    pub w_out: Tensor,
    pub b_out: Tensor,
    pub gamma: Tensor,

    pub dwq: Tensor,
    pub dwk: Tensor,
    pub dwv: Tensor,
    pub dwo: Tensor,
    pub dwi: Tensor,
    pub dwf: Tensor,
    pub dbq: Tensor,
    pub dbk: Tensor,
    pub dbv: Tensor,
    pub dbo: Tensor,
    pub dbi: Tensor,
    pub dbf: Tensor,
    pub dw_out: Tensor,
    pub db_out: Tensor,
    pub dgamma: Tensor,

    resets: Vec<usize>,
    saved: Saved,
    opt: Vec<AdamState>,
}

impl MLstm {
    pub fn new(input_size: usize, d: usize, heads: usize, dqk: usize) -> Self {
        assert!(
            heads > 0 && d.is_multiple_of(heads),
            "d must be divisible by heads"
        );
        let dhv = d / heads;
        let d_qk = heads * dqk;
        let sq = |fi: usize, fo: usize| (6.0 / (fi as f32 + fo as f32)).sqrt();
        let bf: Vec<f32> = (0..heads).map(|_| random_range(4.0..6.0)).collect();
        let bi: Vec<f32> = (0..heads).map(|_| -10.0).collect();

        Self {
            input_size,
            d,
            heads,
            dqk,
            dhv,
            inv_sqrt_dqk: 1.0 / (dqk as f32).sqrt(),
            wq: Tensor::random(&[input_size, d_qk], sq(input_size, d_qk)),
            wk: Tensor::random(&[input_size, d_qk], sq(input_size, d_qk)),
            wv: Tensor::random(&[input_size, d], sq(input_size, d)),
            wo: Tensor::random(&[input_size, d], sq(input_size, d)),
            wi: Tensor::zeros(&[input_size, heads]),
            wf: Tensor::zeros(&[input_size, heads]),
            bq: Tensor::zeros(&[d_qk]),
            bk: Tensor::zeros(&[d_qk]),
            bv: Tensor::zeros(&[d]),
            bo: Tensor::zeros(&[d]),
            bi: Tensor::new(&[heads], bi),
            bf: Tensor::new(&[heads], bf),
            w_out: Tensor::random(&[d, d], sq(d, d)),
            b_out: Tensor::zeros(&[d]),
            gamma: Tensor::new(&[d], vec![1.0; d]),
            dwq: Tensor::zeros(&[input_size, d_qk]),
            dwk: Tensor::zeros(&[input_size, d_qk]),
            dwv: Tensor::zeros(&[input_size, d]),
            dwo: Tensor::zeros(&[input_size, d]),
            dwi: Tensor::zeros(&[input_size, heads]),
            dwf: Tensor::zeros(&[input_size, heads]),
            dbq: Tensor::zeros(&[d_qk]),
            dbk: Tensor::zeros(&[d_qk]),
            dbv: Tensor::zeros(&[d]),
            dbo: Tensor::zeros(&[d]),
            dbi: Tensor::zeros(&[heads]),
            dbf: Tensor::zeros(&[heads]),
            dw_out: Tensor::zeros(&[d, d]),
            db_out: Tensor::zeros(&[d]),
            dgamma: Tensor::zeros(&[d]),
            resets: Vec::new(),
            saved: Saved::default(),
            opt: Vec::new(),
        }
    }

    #[inline]
    fn d_qk(&self) -> usize {
        self.heads * self.dqk
    }

    /// Time steps of the next forward (and its backward) at which every sequence
    /// restarts from zero state. Step 0 always does.
    pub fn set_resets(&mut self, steps: &[usize]) {
        self.resets.clear();
        self.resets.extend_from_slice(steps);
    }

    fn reset_mask(&self, t: usize) -> Vec<bool> {
        let mut mask = vec![false; t];
        for &s in &self.resets {
            if s < t {
                mask[s] = true;
            }
        }
        mask
    }

    /// Forward over `[B, T, in]`; returns `[B, T, d]`.
    pub fn forward(&mut self, x: &Tensor) -> Tensor {
        assert_eq!(x.rank(), 3, "MLstm::forward expects [B, T, in]");
        let (b, t, inp) = (x.shape[0], x.shape[1], x.shape[2]);
        assert_eq!(inp, self.input_size, "MLstm::forward — input width mismatch");
        let (d, h, dhv) = (self.d, self.heads, self.dhv);
        let n = b * t;
        let reset = self.reset_mask(t);

        let mut sv = std::mem::take(&mut self.saved);
        sv.b = b;
        sv.t = t;
        sv.x.clear();
        sv.x.extend_from_slice(&x.data);
        sv.q = project(&x.data, n, inp, &self.wq, &self.bq, 1.0, false);
        sv.k = project(&x.data, n, inp, &self.wk, &self.bk, self.inv_sqrt_dqk, false);
        sv.v = project(&x.data, n, inp, &self.wv, &self.bv, 1.0, false);
        sv.o = project(&x.data, n, inp, &self.wo, &self.bo, 1.0, true);
        let i_pre = project(&x.data, n, inp, &self.wi, &self.bi, 1.0, false);
        let f_pre = project(&x.data, n, inp, &self.wf, &self.bf, 1.0, false);

        let sweep = Sweep {
            t,
            heads: h,
            dqk: self.dqk,
            dhv,
            q: &sv.q.data,
            k: &sv.k.data,
            v: &sv.v.data,
            i_pre: &i_pre.data,
            f_pre: &f_pre.data,
            reset: &reset,
        };
        sv.heads.resize_with(b * h, HeadCache::default);
        sv.heads.truncate(b * h);
        sv.heads
            .par_iter_mut()
            .enumerate()
            .for_each(|(bh, hc)| sweep.forward(bh, hc));

        let mut h_tilde = Tensor::zeros(&[n, d]);
        for (bh, hc) in sv.heads.iter().enumerate() {
            let (bi_, hd) = (bh / h, bh % h);
            for s in 0..t {
                let inv_psi = 1.0 / hc.psi[s];
                let dst = (bi_ * t + s) * d + hd * dhv;
                for i in 0..dhv {
                    h_tilde.data[dst + i] = hc.cq[s * dhv + i] * inv_psi;
                }
            }
        }
        let norm = ops::rms_norm_forward(&h_tilde, &self.gamma, dhv, EPS);
        let mut hconcat = norm.out;
        for (y, &o) in hconcat.data.iter_mut().zip(&sv.o.data) {
            *y *= o;
        }
        let mut out = Tensor::zeros(&[n, d]);
        ops::broadcast_row(&mut out, &self.b_out);
        gemm::gemm_nn(n, d, d, &hconcat.data, &self.w_out.data, &mut out.data, 1.0);

        sv.x_hat = norm.x_hat;
        sv.inv_rms = norm.inv_rms;
        sv.hconcat = hconcat;
        self.saved = sv;
        out.reshape(&[b, t, d])
    }

    /// Backward over `[B, T, d]`; returns `dx` `[B, T, in]`. Accumulates all grads.
    pub fn backward(&mut self, dy: &Tensor) -> Tensor {
        assert_eq!(dy.rank(), 3, "MLstm::backward expects [B, T, d]");
        let (b, t, d) = (dy.shape[0], dy.shape[1], dy.shape[2]);
        let sv = std::mem::take(&mut self.saved);
        assert_eq!((b, t), (sv.b, sv.t), "MLstm::backward — shape differs from the forward");
        assert_eq!(d, self.d, "MLstm::backward — hidden mismatch");
        let (h, dqk, dhv, inp) = (self.heads, self.dqk, self.dhv, self.input_size);
        let d_qk = self.d_qk();
        let n = b * t;
        let reset = self.reset_mask(t);

        // Output projection.
        gemm::gemm_tn(d, n, d, &sv.hconcat.data, &dy.data, &mut self.dw_out.data, 1.0);
        ops::add_col_sum(&mut self.db_out, &dy.reshape(&[n, d]));
        let mut w_out_t = Tensor::zeros(&[d, d]);
        gemm::transpose(d, d, &self.w_out.data, &mut w_out_t.data);
        let mut d_hconcat = Tensor::zeros(&[n, d]);
        gemm::gemm_nn(n, d, d, &dy.data, &w_out_t.data, &mut d_hconcat.data, 0.0);

        // y = o ⊙ γ·x̂
        let mut do_pre = Tensor::zeros(&[n, d]);
        let mut d_yhat = Tensor::zeros(&[n, d]);
        for idx in 0..n * d {
            let dch = d_hconcat.data[idx];
            let oi = sv.o.data[idx];
            let yhat = self.gamma.data[idx % d] * sv.x_hat.data[idx];
            do_pre.data[idx] = dch * yhat * oi * (1.0 - oi);
            d_yhat.data[idx] = dch * oi;
        }
        let d_ht = ops::rms_norm_backward(&d_yhat, &sv.x_hat, &sv.inv_rms, &self.gamma, &mut self.dgamma, dhv);

        let zero = Tensor::zeros(&[0]);
        let sweep = Sweep {
            t,
            heads: h,
            dqk,
            dhv,
            q: &sv.q.data,
            k: &sv.k.data,
            v: &sv.v.data,
            i_pre: &zero.data,
            f_pre: &zero.data,
            reset: &reset,
        };
        let mut grads: Vec<HeadGrads> = (0..b * h).map(|_| HeadGrads::default()).collect();
        grads
            .par_iter_mut()
            .enumerate()
            .for_each(|(bh, g)| sweep.backward(bh, &sv.heads[bh], &d_ht.data, g));
        drop(sv.heads);

        let mut dq = Tensor::zeros(&[n, d_qk]);
        let mut dk = Tensor::zeros(&[n, d_qk]);
        let mut dv = Tensor::zeros(&[n, d]);
        let mut di = Tensor::zeros(&[n, h]);
        let mut df = Tensor::zeros(&[n, h]);
        let s = self.inv_sqrt_dqk;
        for (bh, g) in grads.iter().enumerate() {
            let (bi_, hd) = (bh / h, bh % h);
            for st in 0..t {
                let r = bi_ * t + st;
                let qk = r * d_qk + hd * dqk;
                dq.data[qk..qk + dqk].copy_from_slice(&g.dq[st * dqk..(st + 1) * dqk]);
                for j in 0..dqk {
                    dk.data[qk + j] = g.dk[st * dqk + j] * s;
                }
                let vo = r * d + hd * dhv;
                dv.data[vo..vo + dhv].copy_from_slice(&g.dv[st * dhv..(st + 1) * dhv]);
                di.data[r * h + hd] = g.di[st];
                df.data[r * h + hd] = g.df[st];
            }
        }
        drop(grads);

        let mut dx = Tensor::zeros(&[n, inp]);
        let mut wt = Tensor::zeros(&[0]);
        for (k, (dg, w, dw, db)) in [
            (&dq, &self.wq, &mut self.dwq, &mut self.dbq),
            (&dk, &self.wk, &mut self.dwk, &mut self.dbk),
            (&dv, &self.wv, &mut self.dwv, &mut self.dbv),
            (&do_pre, &self.wo, &mut self.dwo, &mut self.dbo),
            (&di, &self.wi, &mut self.dwi, &mut self.dbi),
            (&df, &self.wf, &mut self.dwf, &mut self.dbf),
        ]
        .into_iter()
        .enumerate()
        {
            let out_w = w.cols();
            ops::add_col_sum(db, dg);
            gemm::gemm_tn(inp, n, out_w, &sv.x, &dg.data, &mut dw.data, 1.0);
            wt.fit(&[out_w, inp]);
            gemm::transpose(inp, out_w, &w.data, &mut wt.data);
            let beta = if k == 0 { 0.0 } else { 1.0 };
            gemm::gemm_nn(n, out_w, inp, &dg.data, &wt.data, &mut dx.data, beta);
        }
        dx.reshape(&[b, t, inp])
    }

    pub fn grads_mut(&mut self) -> Vec<&mut Tensor> {
        vec![
            &mut self.dwq,
            &mut self.dwk,
            &mut self.dwv,
            &mut self.dwo,
            &mut self.dwi,
            &mut self.dwf,
            &mut self.dbq,
            &mut self.dbk,
            &mut self.dbv,
            &mut self.dbo,
            &mut self.dbi,
            &mut self.dbf,
            &mut self.dw_out,
            &mut self.db_out,
            &mut self.dgamma,
        ]
    }

    pub fn zero_grad(&mut self) {
        for g in self.grads_mut() {
            g.zero_();
        }
    }

    /// AdamW step: projection + output matrices decay; biases and head-norm
    /// gamma don't. Clears the grads.
    pub fn step(&mut self, cfg: &AdamCfg) {
        ensure_states(&mut self.opt, 15);
        self.opt[0].step(&mut self.wq.data, &self.dwq.data, cfg, true);
        self.opt[1].step(&mut self.wk.data, &self.dwk.data, cfg, true);
        self.opt[2].step(&mut self.wv.data, &self.dwv.data, cfg, true);
        self.opt[3].step(&mut self.wo.data, &self.dwo.data, cfg, true);
        self.opt[4].step(&mut self.wi.data, &self.dwi.data, cfg, true);
        self.opt[5].step(&mut self.wf.data, &self.dwf.data, cfg, true);
        self.opt[6].step(&mut self.w_out.data, &self.dw_out.data, cfg, true);
        self.opt[7].step(&mut self.bq.data, &self.dbq.data, cfg, false);
        self.opt[8].step(&mut self.bk.data, &self.dbk.data, cfg, false);
        self.opt[9].step(&mut self.bv.data, &self.dbv.data, cfg, false);
        self.opt[10].step(&mut self.bo.data, &self.dbo.data, cfg, false);
        self.opt[11].step(&mut self.bi.data, &self.dbi.data, cfg, false);
        self.opt[12].step(&mut self.bf.data, &self.dbf.data, cfg, false);
        self.opt[13].step(&mut self.b_out.data, &self.db_out.data, cfg, false);
        self.opt[14].step(&mut self.gamma.data, &self.dgamma.data, cfg, false);
        self.zero_grad();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Ctx {
        cell: MLstm,
        x: Tensor,
        g: Tensor,
    }

    fn loss(c: &mut Ctx) -> f32 {
        let y = c.cell.forward(&c.x);
        y.data.iter().zip(&c.g.data).map(|(a, b)| a * b).sum()
    }

    fn check(
        state: &mut Ctx,
        grad: &[f32],
        name: &str,
        eps: f32,
        tol: f32,
        mut perturb: impl FnMut(&mut Ctx, f32, &[f32]),
    ) {
        let norm: f32 = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
        assert!(norm > 1e-6, "{name}: analytic grad ~zero");
        let u: Vec<f32> = grad.iter().map(|g| g / norm).collect();
        perturb(state, eps, &u);
        let plus = loss(state);
        perturb(state, -2.0 * eps, &u);
        let minus = loss(state);
        perturb(state, eps, &u);
        let fd = (plus - minus) / (2.0 * eps);
        assert!(
            (fd - norm).abs() <= tol * norm + 1e-3,
            "{name}: ‖G‖ {norm} vs fd {fd}"
        );
    }

    #[test]
    fn backward_matches_finite_difference() {
        let (b, t, inp, d, heads, dqk) = (2, 3, 5, 8, 2, 4); // dhv = 4
        let mut ctx = Ctx {
            cell: MLstm::new(inp, d, heads, dqk),
            x: Tensor::random(&[b, t, inp], 0.5),
            g: Tensor::random(&[b, t, d], 1.0),
        };
        let _y = ctx.cell.forward(&ctx.x);
        let g = ctx.g.clone();
        let dx = ctx.cell.backward(&g);
        let dwq = ctx.cell.dwq.clone();
        let dwv = ctx.cell.dwv.clone();
        let dwo_out = ctx.cell.dw_out.clone();

        let eps = 2e-4;
        let tol = 0.3;

        check(&mut ctx, &dwq.data, "dwq", eps, tol, |c, s, u| {
            for (w, &ui) in c.cell.wq.data.iter_mut().zip(u) {
                *w += s * ui;
            }
        });
        check(&mut ctx, &dwv.data, "dwv", eps, tol, |c, s, u| {
            for (w, &ui) in c.cell.wv.data.iter_mut().zip(u) {
                *w += s * ui;
            }
        });
        check(&mut ctx, &dwo_out.data, "dw_out", eps, tol, |c, s, u| {
            for (w, &ui) in c.cell.w_out.data.iter_mut().zip(u) {
                *w += s * ui;
            }
        });
        check(&mut ctx, &dx.data, "dx", eps, tol, |c, s, u| {
            for (v, &ui) in c.x.data.iter_mut().zip(u) {
                *v += s * ui;
            }
        });
    }

    /// A reset step makes the sequence after it exactly a sequence of its own, in
    /// the output and in the gradient its input receives, and nothing flows back
    /// across it.
    #[test]
    fn reset_splits_the_sequence() {
        let (t, cut, inp, d, heads, dqk) = (SEGMENT + 9, SEGMENT - 3, 6, 8, 2, 4);
        let mut cell = MLstm::new(inp, d, heads, dqk);
        cell.wi = Tensor::random(&[inp, heads], 0.5);
        cell.bi = Tensor::random(&[heads], 1.0);
        let x = Tensor::random(&[1, t, inp], 0.7);
        let g = Tensor::random(&[1, t, d], 1.0);

        cell.set_resets(&[cut]);
        let y = cell.forward(&x);
        let dx = cell.backward(&g);

        let tail = |a: &Tensor, w: usize| Tensor::new(&[1, t - cut, w], a.data[cut * w..].to_vec());
        cell.set_resets(&[]);
        let y_tail = cell.forward(&tail(&x, inp));
        let dx_tail = cell.backward(&tail(&g, d));
        let close = |a: &[f32], b: &[f32], what: &str| {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                assert!((x - y).abs() <= 1e-5 * y.abs().max(1.0), "{what}[{i}]: {x} vs {y}");
            }
        };
        close(&y.data[cut * d..], &y_tail.data, "y");
        close(&dx.data[cut * inp..], &dx_tail.data, "dx");

        let mut g_tail_only = g.clone();
        g_tail_only.data[..cut * d].fill(0.0);
        cell.set_resets(&[cut]);
        cell.forward(&x);
        let dx = cell.backward(&g_tail_only);
        assert!(dx.data[..cut * inp].iter().all(|&v| v == 0.0));
    }
}
