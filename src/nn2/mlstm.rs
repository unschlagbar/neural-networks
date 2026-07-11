//! Batched multi-head mLSTM cell (xLSTM, Beck et al. 2024), sequential over time.
//!
//! Direct port of `nn::mlstm::MLSTMLayer`, batched over the leading axis. Per
//! head `h` the state is a matrix `C_h ∈ ℝ^{dhv×dqk}`, a normalizer `n_h ∈
//! ℝ^{dqk}` and a scalar stabilizer `m_h`; here each is carried as `[B, …]`.
//!
//! Forward per step (batched):
//!   q = Xt·Wq+bq, k = (Xt·Wk+bk)/√dqk, v = Xt·Wv+bv,
//!   o = σ(Xt·Wo+bo), ĩ = Xt·Wi+bi, f̃ = Xt·Wf+bf
//!   per head:  m = max(logσ(f̃)+m_prev, ĩ),  i'=exp(ĩ-m),  f'=exp(logσ(f̃)+m_prev-m)
//!              C = f'·C_prev + i'·(v⊗k),  n = f'·n_prev + i'·k
//!              ψ = max(|nᵀq|, 1),  ỹ = C·q / ψ,  ŷ = headnorm(ỹ),  y = o⊙ŷ
//!   h = y·W_out + b_out
//!
//! The projections and output projection are batched GEMMs; the per-head
//! recurrence (outer product, C·q) is small-matrix work looped over B·H.
//! Head-wise RMSNorm is folded in (own gamma). All working buffers are pooled
//! and reused across calls (see `Tensor::fit`) — steady-state training allocates
//! only the returned output/dx. Stabilizer `m` is treated as constant in
//! backward (reference approximation), so grads use the directional whole-tensor
//! finite-difference check.

use crate::nn2::optim::{AdamCfg, AdamState, ensure_states};
use crate::tensor::{Tensor, gemm};

const EPS: f32 = 1e-6;

#[inline]
fn stable_sigmoid(x: f32) -> f32 {
    if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e = x.exp(); e / (1.0 + e) }
}

#[inline]
fn log_sigmoid(x: f32) -> f32 {
    if x >= 0.0 { -(1.0 + (-x).exp()).ln() } else { x - (1.0 + x.exp()).ln() }
}

/// Lane-accumulated dot product. `f32` addition doesn't reassociate, so a plain
/// `iter().sum()` compiles to a serial dependency chain (scalar); the 8
/// independent lanes let the vectorizer emit AVX FMA. Matches the old `nn::dot`.
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

/// `out = Xt·W + bias`, optionally scaled and/or sigmoid'd. Writes into `out`.
fn project(b: usize, inp: usize, out_w: usize, xt: &[f32], w: &[f32], bias: &[f32],
           scale: f32, sigmoid: bool, out: &mut [f32]) {
    gemm::gemm_nn(b, inp, out_w, xt, w, out, 0.0);
    for row in out.chunks_mut(out_w) {
        for (o, &bv) in row.iter_mut().zip(bias) {
            *o = (*o + bv) * scale;
            if sigmoid {
                *o = stable_sigmoid(*o);
            }
        }
    }
}

/// Per-timestep forward cache; pooled and reused across calls.
struct Step {
    xt: Tensor, q: Tensor, k: Tensor, v: Tensor, o: Tensor,
    log_f: Tensor, i_prime: Tensor, f_prime: Tensor, nq: Tensor, psi: Tensor,
    cq: Tensor, c: Tensor, c_prev: Tensor, n: Tensor, n_prev: Tensor,
    yhat: Tensor, x_hat: Tensor, inv_rms: Tensor, hconcat: Tensor,
}

impl Step {
    fn new() -> Self {
        let z = || Tensor::zeros(&[0, 0]);
        Self {
            xt: z(), q: z(), k: z(), v: z(), o: z(), log_f: z(), i_prime: z(),
            f_prime: z(), nq: z(), psi: z(), cq: z(), c: z(), c_prev: z(),
            n: z(), n_prev: z(), yhat: z(), x_hat: z(), inv_rms: z(), hconcat: z(),
        }
    }

    fn fit(&mut self, b: usize, inp: usize, d: usize, d_qk: usize, h: usize, dhv: usize, dqk: usize) {
        let cdim = h * dhv * dqk;
        let ndim = h * dqk;
        self.xt.fit(&[b, inp]);
        self.q.fit(&[b, d_qk]);
        self.k.fit(&[b, d_qk]);
        self.v.fit(&[b, d]);
        self.o.fit(&[b, d]);
        self.log_f.fit(&[b, h]);
        self.i_prime.fit(&[b, h]);
        self.f_prime.fit(&[b, h]);
        self.nq.fit(&[b, h]);
        self.psi.fit(&[b, h]);
        self.cq.fit(&[b, d]);
        self.c.fit(&[b, cdim]);
        self.c_prev.fit(&[b, cdim]);
        self.n.fit(&[b, ndim]);
        self.n_prev.fit(&[b, ndim]);
        self.yhat.fit(&[b, d]);
        self.x_hat.fit(&[b, d]);
        self.inv_rms.fit(&[b, h]);
        self.hconcat.fit(&[b, d]);
    }
}

pub struct MLstm {
    pub input_size: usize,
    pub d: usize,
    pub heads: usize,
    pub dqk: usize,
    pub dhv: usize,
    inv_sqrt_dqk: f32,

    pub wq: Tensor, pub wk: Tensor, pub wv: Tensor, pub wo: Tensor, pub wi: Tensor, pub wf: Tensor,
    pub bq: Tensor, pub bk: Tensor, pub bv: Tensor, pub bo: Tensor, pub bi: Tensor, pub bf: Tensor,
    pub w_out: Tensor, pub b_out: Tensor, pub gamma: Tensor,

    pub dwq: Tensor, pub dwk: Tensor, pub dwv: Tensor, pub dwo: Tensor, pub dwi: Tensor, pub dwf: Tensor,
    pub dbq: Tensor, pub dbk: Tensor, pub dbv: Tensor, pub dbo: Tensor, pub dbi: Tensor, pub dbf: Tensor,
    pub dw_out: Tensor, pub db_out: Tensor, pub dgamma: Tensor,

    // --- pooled buffers ----------------------------------------------------
    steps: Vec<Step>,
    c_state: Tensor,   // [b, h*dhv*dqk]
    n_state: Tensor,   // [b, h*dqk]
    m_state: Tensor,   // [b, h]
    // forward transient scratch
    i_pre: Tensor, f_pre: Tensor, h_tilde: Tensor, out_t: Tensor,
    // backward transient scratch
    delta: Tensor, d_hconcat: Tensor, do_pre: Tensor, d_yhat: Tensor, d_ht: Tensor,
    dq: Tensor, dk: Tensor, dv: Tensor, di_pre: Tensor, df_pre: Tensor, dxt: Tensor,
    dc_bptt: Tensor, dn_bptt: Tensor, tmp: Vec<f32>,
    // transposed weights for the fast dx path
    wqt: Tensor, wkt: Tensor, wvt: Tensor, wot: Tensor, wit: Tensor, wft: Tensor, w_out_t: Tensor,
    opt: Vec<AdamState>,
    batch: usize,
}

impl MLstm {
    pub fn new(input_size: usize, d: usize, heads: usize, dqk: usize) -> Self {
        assert!(heads > 0 && d % heads == 0, "d must be divisible by heads");
        let dhv = d / heads;
        let d_qk = heads * dqk;
        let sq = |fi: usize, fo: usize| (6.0 / (fi as f32 + fo as f32)).sqrt();
        let bf: Vec<f32> = (0..heads).map(|_| 4.5).collect();
        let bi: Vec<f32> = (0..heads).map(|_| -4.5).collect();
        let z = || Tensor::zeros(&[0, 0]);

        Self {
            input_size, d, heads, dqk, dhv,
            inv_sqrt_dqk: 1.0 / (dqk as f32).sqrt(),
            wq: Tensor::random(&[input_size, d_qk], sq(input_size, d_qk)),
            wk: Tensor::random(&[input_size, d_qk], sq(input_size, d_qk)),
            wv: Tensor::random(&[input_size, d], sq(input_size, d)),
            wo: Tensor::random(&[input_size, d], sq(input_size, d)),
            wi: Tensor::zeros(&[input_size, heads]),
            wf: Tensor::zeros(&[input_size, heads]),
            bq: Tensor::zeros(&[d_qk]), bk: Tensor::zeros(&[d_qk]),
            bv: Tensor::zeros(&[d]), bo: Tensor::zeros(&[d]),
            bi: Tensor::new(&[heads], bi), bf: Tensor::new(&[heads], bf),
            w_out: Tensor::random(&[d, d], sq(d, d)), b_out: Tensor::zeros(&[d]),
            gamma: Tensor::new(&[d], vec![1.0; d]),
            dwq: Tensor::zeros(&[input_size, d_qk]), dwk: Tensor::zeros(&[input_size, d_qk]),
            dwv: Tensor::zeros(&[input_size, d]), dwo: Tensor::zeros(&[input_size, d]),
            dwi: Tensor::zeros(&[input_size, heads]), dwf: Tensor::zeros(&[input_size, heads]),
            dbq: Tensor::zeros(&[d_qk]), dbk: Tensor::zeros(&[d_qk]),
            dbv: Tensor::zeros(&[d]), dbo: Tensor::zeros(&[d]),
            dbi: Tensor::zeros(&[heads]), dbf: Tensor::zeros(&[heads]),
            dw_out: Tensor::zeros(&[d, d]), db_out: Tensor::zeros(&[d]),
            dgamma: Tensor::zeros(&[d]),
            steps: Vec::new(),
            c_state: z(), n_state: z(), m_state: z(),
            i_pre: z(), f_pre: z(), h_tilde: z(), out_t: z(),
            delta: z(), d_hconcat: z(), do_pre: z(), d_yhat: z(), d_ht: z(),
            dq: z(), dk: z(), dv: z(), di_pre: z(), df_pre: z(), dxt: z(),
            dc_bptt: z(), dn_bptt: z(), tmp: Vec::new(),
            wqt: z(), wkt: z(), wvt: z(), wot: z(), wit: z(), wft: z(), w_out_t: z(),
            opt: Vec::new(),
            batch: 0,
        }
    }

    #[inline]
    fn d_qk(&self) -> usize {
        self.heads * self.dqk
    }

    /// Forward over `[B, T, in]`; returns `[B, T, d]`. State resets to zero.
    pub fn forward(&mut self, x: &Tensor) -> Tensor {
        assert_eq!(x.rank(), 3, "MLstm::forward expects [B, T, in]");
        let (b, t, inp) = (x.shape[0], x.shape[1], x.shape[2]);
        assert_eq!(inp, self.input_size, "MLstm::forward — input width mismatch");
        let (d, h, dqk, dhv) = (self.d, self.heads, self.dqk, self.dhv);
        let d_qk = self.d_qk();
        let inv_sqrt = self.inv_sqrt_dqk;
        self.batch = b;

        self.c_state.fit(&[b, h * dhv * dqk]);
        self.n_state.fit(&[b, h * dqk]);
        self.m_state.fit(&[b, h]);
        self.c_state.zero_();
        self.n_state.zero_();
        self.m_state.zero_();
        self.i_pre.fit(&[b, h]);
        self.f_pre.fit(&[b, h]);
        self.h_tilde.fit(&[b, d]);
        self.out_t.fit(&[b, d]);
        while self.steps.len() < t {
            self.steps.push(Step::new());
        }
        for s in 0..t {
            self.steps[s].fit(b, inp, d, d_qk, h, dhv, dqk);
        }

        let mut out = Tensor::zeros(&[b, t, d]);

        let Self {
            steps, wq, wk, wv, wo, wi, wf, bq, bk, bv, bo, bi, bf, w_out, b_out, gamma,
            c_state, n_state, m_state, i_pre, f_pre, h_tilde, out_t, ..
        } = self;

        for step in 0..t {
            let st = &mut steps[step];

            for bi_ in 0..b {
                let src = &x.data[(bi_ * t + step) * inp..(bi_ * t + step) * inp + inp];
                st.xt.data[bi_ * inp..(bi_ + 1) * inp].copy_from_slice(src);
            }

            project(b, inp, d_qk, &st.xt.data, &wq.data, &bq.data, 1.0, false, &mut st.q.data);
            project(b, inp, d_qk, &st.xt.data, &wk.data, &bk.data, inv_sqrt, false, &mut st.k.data);
            project(b, inp, d, &st.xt.data, &wv.data, &bv.data, 1.0, false, &mut st.v.data);
            project(b, inp, d, &st.xt.data, &wo.data, &bo.data, 1.0, true, &mut st.o.data);
            project(b, inp, h, &st.xt.data, &wi.data, &bi.data, 1.0, false, &mut i_pre.data);
            project(b, inp, h, &st.xt.data, &wf.data, &bf.data, 1.0, false, &mut f_pre.data);

            st.c_prev.data.copy_from_slice(&c_state.data);
            st.n_prev.data.copy_from_slice(&n_state.data);

            for bi_ in 0..b {
                for hd in 0..h {
                    let bh = bi_ * h + hd;
                    let lf = log_sigmoid(f_pre.data[bh]);
                    let m_prev = m_state.data[bh];
                    let mh = (lf + m_prev).max(i_pre.data[bh]);
                    let ip = (i_pre.data[bh] - mh).exp();
                    let fp = (lf + m_prev - mh).exp();
                    st.log_f.data[bh] = lf;
                    st.i_prime.data[bh] = ip;
                    st.f_prime.data[bh] = fp;
                    m_state.data[bh] = mh;

                    let qk_off = bi_ * d_qk + hd * dqk;
                    let v_off = bi_ * d + hd * dhv;
                    let q_h = &st.q.data[qk_off..qk_off + dqk];
                    let k_h = &st.k.data[qk_off..qk_off + dqk];
                    let v_h = &st.v.data[v_off..v_off + dhv];

                    let n_off = bh * dqk;
                    {
                        let n_row = &mut n_state.data[n_off..n_off + dqk];
                        let np_row = &st.n_prev.data[n_off..n_off + dqk];
                        for j in 0..dqk {
                            n_row[j] = fp.mul_add(np_row[j], ip * k_h[j]);
                        }
                    }
                    let nqv = dot(&n_state.data[n_off..n_off + dqk], q_h);
                    st.nq.data[bh] = nqv;
                    let p = nqv.abs().max(1.0);
                    st.psi.data[bh] = p;
                    let inv_psi = 1.0 / p;

                    let cbase = bh * dhv * dqk;
                    for i in 0..dhv {
                        let ivi = ip * v_h[i];
                        let row = cbase + i * dqk;
                        let c_row = &mut c_state.data[row..row + dqk];
                        let cp_row = &st.c_prev.data[row..row + dqk];
                        for j in 0..dqk {
                            c_row[j] = fp.mul_add(cp_row[j], ivi * k_h[j]);
                        }
                        let s = dot(c_row, q_h);
                        st.cq.data[v_off + i] = s;
                        h_tilde.data[v_off + i] = s * inv_psi;
                    }
                }
            }

            // Head-wise RMSNorm of h_tilde -> yhat.
            for bi_ in 0..b {
                for hd in 0..h {
                    let off = bi_ * d + hd * dhv;
                    let ss: f32 = h_tilde.data[off..off + dhv].iter().map(|&z| z * z).sum();
                    let inv = 1.0 / (ss / dhv as f32 + EPS).sqrt();
                    st.inv_rms.data[bi_ * h + hd] = inv;
                    for i in 0..dhv {
                        let xh = h_tilde.data[off + i] * inv;
                        st.x_hat.data[off + i] = xh;
                        st.yhat.data[off + i] = gamma.data[hd * dhv + i] * xh;
                    }
                }
            }

            for idx in 0..b * d {
                st.hconcat.data[idx] = st.o.data[idx] * st.yhat.data[idx];
            }
            for row in out_t.data.chunks_mut(d) {
                row.copy_from_slice(&b_out.data);
            }
            gemm::gemm_nn(b, d, d, &st.hconcat.data, &w_out.data, &mut out_t.data, 1.0);
            for bi_ in 0..b {
                out.data[(bi_ * t + step) * d..(bi_ * t + step) * d + d]
                    .copy_from_slice(&out_t.data[bi_ * d..(bi_ + 1) * d]);
            }

            // Advance the recurrent state (c_state/n_state now hold C_t/n_t).
            st.c.data.copy_from_slice(&c_state.data);
            st.n.data.copy_from_slice(&n_state.data);
        }
        out
    }

    /// Backward over `[B, T, d]`; returns `dx` `[B, T, in]`. Accumulates all grads.
    pub fn backward(&mut self, dy: &Tensor) -> Tensor {
        assert_eq!(dy.rank(), 3, "MLstm::backward expects [B, T, d]");
        let (b, t, d) = (dy.shape[0], dy.shape[1], dy.shape[2]);
        assert_eq!(b, self.batch, "MLstm::backward — batch mismatch");
        assert_eq!(d, self.d, "MLstm::backward — hidden mismatch");
        let (h, dqk, dhv, inp) = (self.heads, self.dqk, self.dhv, self.input_size);
        let d_qk = self.d_qk();
        let inv_sqrt = self.inv_sqrt_dqk;

        let mut dx = Tensor::zeros(&[b, t, inp]);

        self.dc_bptt.fit(&[b, h * dhv * dqk]);
        self.dn_bptt.fit(&[b, h * dqk]);
        self.dc_bptt.zero_();
        self.dn_bptt.zero_();
        for buf in [
            &mut self.delta, &mut self.d_hconcat, &mut self.do_pre, &mut self.d_yhat, &mut self.d_ht,
        ] {
            buf.fit(&[b, d]);
        }
        self.dq.fit(&[b, d_qk]);
        self.dk.fit(&[b, d_qk]);
        self.dv.fit(&[b, d]);
        self.di_pre.fit(&[b, h]);
        self.df_pre.fit(&[b, h]);
        self.dxt.fit(&[b, inp]);
        self.tmp.resize(dqk, 0.0);

        // Transpose weights once (reused across all T steps).
        self.wqt.fit(&[d_qk, inp]); gemm::transpose(inp, d_qk, &self.wq.data, &mut self.wqt.data);
        self.wkt.fit(&[d_qk, inp]); gemm::transpose(inp, d_qk, &self.wk.data, &mut self.wkt.data);
        self.wvt.fit(&[d, inp]); gemm::transpose(inp, d, &self.wv.data, &mut self.wvt.data);
        self.wot.fit(&[d, inp]); gemm::transpose(inp, d, &self.wo.data, &mut self.wot.data);
        self.wit.fit(&[h, inp]); gemm::transpose(inp, h, &self.wi.data, &mut self.wit.data);
        self.wft.fit(&[h, inp]); gemm::transpose(inp, h, &self.wf.data, &mut self.wft.data);
        self.w_out_t.fit(&[d, d]); gemm::transpose(d, d, &self.w_out.data, &mut self.w_out_t.data);

        let Self {
            steps, gamma, dwq, dwk, dwv, dwo, dwi, dwf, dbq, dbk, dbv, dbo, dbi, dbf,
            dw_out, db_out, dgamma, delta, d_hconcat, do_pre, d_yhat, d_ht,
            dq, dk, dv, di_pre, df_pre, dxt, dc_bptt, dn_bptt, tmp,
            wqt, wkt, wvt, wot, wit, wft, w_out_t, ..
        } = self;

        for step in (0..t).rev() {
            let st = &steps[step];

            for bi_ in 0..b {
                delta.data[bi_ * d..(bi_ + 1) * d]
                    .copy_from_slice(&dy.data[(bi_ * t + step) * d..(bi_ * t + step) * d + d]);
            }

            // W_out backward.
            gemm::gemm_tn(d, b, d, &st.hconcat.data, &delta.data, &mut dw_out.data, 1.0);
            for row in delta.data.chunks(d) {
                for (o, v) in db_out.data.iter_mut().zip(row) {
                    *o += *v;
                }
            }
            gemm::gemm_nn(b, d, d, &delta.data, &w_out_t.data, &mut d_hconcat.data, 0.0);

            for idx in 0..b * d {
                let dch = d_hconcat.data[idx];
                let oi = st.o.data[idx];
                do_pre.data[idx] = dch * st.yhat.data[idx] * oi * (1.0 - oi);
                d_yhat.data[idx] = dch * oi;
            }

            // Head-norm backward: d_yhat -> d_ht (grad wrt ỹ), accumulate dgamma.
            for bi_ in 0..b {
                for hd in 0..h {
                    let off = bi_ * d + hd * dhv;
                    let g_off = hd * dhv;
                    let inv = st.inv_rms.data[bi_ * h + hd];
                    let mut s = 0.0;
                    for i in 0..dhv {
                        let dyxh = d_yhat.data[off + i] * st.x_hat.data[off + i];
                        dgamma.data[g_off + i] += dyxh;
                        s += gamma.data[g_off + i] * dyxh;
                    }
                    let s_n = s / dhv as f32;
                    for i in 0..dhv {
                        d_ht.data[off + i] = inv
                            * (gamma.data[g_off + i] * d_yhat.data[off + i]
                                - st.x_hat.data[off + i] * s_n);
                    }
                }
            }

            for bi_ in 0..b {
                for hd in 0..h {
                    let bh = bi_ * h + hd;
                    let qk_off = bi_ * d_qk + hd * dqk;
                    let v_off = bi_ * d + hd * dhv;
                    let q_h = &st.q.data[qk_off..qk_off + dqk];
                    let k_h = &st.k.data[qk_off..qk_off + dqk];
                    let v_h = &st.v.data[v_off..v_off + dhv];
                    let cq_h = &st.cq.data[v_off..v_off + dhv];
                    let dht_h = &d_ht.data[v_off..v_off + dhv];

                    let inv_psi = 1.0 / st.psi.data[bh];
                    let dpsi = -(inv_psi * inv_psi) * dot(dht_h, cq_h);
                    let dnq = if st.nq.data[bh].abs() > 1.0 { st.nq.data[bh].signum() * dpsi } else { 0.0 };

                    let ip = st.i_prime.data[bh];
                    let fp = st.f_prime.data[bh];
                    let mut df_prime = 0.0;
                    let mut di_prime = 0.0;

                    let n_off = bh * dqk;
                    let dq_h = &mut dq.data[qk_off..qk_off + dqk];
                    let dk_h = &mut dk.data[qk_off..qk_off + dqk];
                    {
                        let dn_h = &mut dn_bptt.data[n_off..n_off + dqk];
                        let n_h = &st.n.data[n_off..n_off + dqk];
                        for j in 0..dqk {
                            dq_h[j] = dnq * n_h[j];
                            dk_h[j] = 0.0;
                            dn_h[j] += dnq * q_h[j];
                        }
                    }

                    let cbase = bh * dhv * dqk;
                    for i in 0..dhv {
                        let dh_over_psi = dht_h[i] * inv_psi;
                        let v_i = v_h[i];
                        let row = cbase + i * dqk;
                        let dcb = &mut dc_bptt.data[row..row + dqk];
                        for j in 0..dqk {
                            tmp[j] = dh_over_psi.mul_add(q_h[j], dcb[j]);
                        }
                        let row_df = dot(tmp, &st.c_prev.data[row..row + dqk]);
                        let row_dv = dot(tmp, k_h);
                        let c_row = &st.c.data[row..row + dqk];
                        for j in 0..dqk {
                            dcb[j] = tmp[j] * fp;
                            dq_h[j] = dh_over_psi.mul_add(c_row[j], dq_h[j]);
                            dk_h[j] = tmp[j].mul_add(v_i, dk_h[j]);
                        }
                        df_prime += row_df;
                        di_prime += v_i * row_dv;
                        dv.data[v_off + i] = ip * row_dv;
                    }

                    {
                        let dn_h = &mut dn_bptt.data[n_off..n_off + dqk];
                        let n_prev_h = &st.n_prev.data[n_off..n_off + dqk];
                        df_prime += dot(dn_h, n_prev_h);
                        di_prime += dot(dn_h, k_h);
                        for j in 0..dqk {
                            dk_h[j] = ip * (dk_h[j] + dn_h[j]);
                            dn_h[j] *= fp;
                        }
                    }

                    di_pre.data[bh] = di_prime * ip;
                    let sigm_f = st.log_f.data[bh].exp();
                    df_pre.data[bh] = df_prime * fp * (1.0 - sigm_f);
                }
            }

            for v in dk.data.iter_mut() {
                *v *= inv_sqrt;
            }

            accum(&mut dbq.data, &dq.data, b, d_qk);
            accum(&mut dbk.data, &dk.data, b, d_qk);
            accum(&mut dbv.data, &dv.data, b, d);
            accum(&mut dbo.data, &do_pre.data, b, d);
            accum(&mut dbi.data, &di_pre.data, b, h);
            accum(&mut dbf.data, &df_pre.data, b, h);

            gemm::gemm_tn(inp, b, d_qk, &st.xt.data, &dq.data, &mut dwq.data, 1.0);
            gemm::gemm_tn(inp, b, d_qk, &st.xt.data, &dk.data, &mut dwk.data, 1.0);
            gemm::gemm_tn(inp, b, d, &st.xt.data, &dv.data, &mut dwv.data, 1.0);
            gemm::gemm_tn(inp, b, d, &st.xt.data, &do_pre.data, &mut dwo.data, 1.0);
            gemm::gemm_tn(inp, b, h, &st.xt.data, &di_pre.data, &mut dwi.data, 1.0);
            gemm::gemm_tn(inp, b, h, &st.xt.data, &df_pre.data, &mut dwf.data, 1.0);

            gemm::gemm_nn(b, d_qk, inp, &dq.data, &wqt.data, &mut dxt.data, 0.0);
            gemm::gemm_nn(b, d_qk, inp, &dk.data, &wkt.data, &mut dxt.data, 1.0);
            gemm::gemm_nn(b, d, inp, &dv.data, &wvt.data, &mut dxt.data, 1.0);
            gemm::gemm_nn(b, d, inp, &do_pre.data, &wot.data, &mut dxt.data, 1.0);
            gemm::gemm_nn(b, h, inp, &di_pre.data, &wit.data, &mut dxt.data, 1.0);
            gemm::gemm_nn(b, h, inp, &df_pre.data, &wft.data, &mut dxt.data, 1.0);
            for bi_ in 0..b {
                dx.data[(bi_ * t + step) * inp..(bi_ * t + step) * inp + inp]
                    .copy_from_slice(&dxt.data[bi_ * inp..(bi_ + 1) * inp]);
            }
        }
        dx
    }

    pub fn zero_grad(&mut self) {
        for g in [
            &mut self.dwq, &mut self.dwk, &mut self.dwv, &mut self.dwo, &mut self.dwi, &mut self.dwf,
            &mut self.dbq, &mut self.dbk, &mut self.dbv, &mut self.dbo, &mut self.dbi, &mut self.dbf,
            &mut self.dw_out, &mut self.db_out, &mut self.dgamma,
        ] {
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

/// Accumulate `Σ_batch grad` into a `[width]` bias-grad slice.
fn accum(dbias: &mut [f32], grad: &[f32], b: usize, width: usize) {
    for bi in 0..b {
        for (a, &g) in dbias.iter_mut().zip(&grad[bi * width..(bi + 1) * width]) {
            *a += g;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Ctx { cell: MLstm, x: Tensor, g: Tensor }

    fn loss(c: &mut Ctx) -> f32 {
        let y = c.cell.forward(&c.x);
        y.data.iter().zip(&c.g.data).map(|(a, b)| a * b).sum()
    }

    fn check(state: &mut Ctx, grad: &[f32], name: &str, eps: f32, tol: f32,
             mut perturb: impl FnMut(&mut Ctx, f32, &[f32])) {
        let norm: f32 = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
        assert!(norm > 1e-6, "{name}: analytic grad ~zero");
        let u: Vec<f32> = grad.iter().map(|g| g / norm).collect();
        perturb(state, eps, &u);
        let plus = loss(state);
        perturb(state, -2.0 * eps, &u);
        let minus = loss(state);
        perturb(state, eps, &u);
        let fd = (plus - minus) / (2.0 * eps);
        assert!((fd - norm).abs() <= tol * norm + 1e-3, "{name}: ‖G‖ {norm} vs fd {fd}");
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

        check(&mut ctx, &dwq.data, "dwq", eps, tol,
            |c, s, u| for (w, &ui) in c.cell.wq.data.iter_mut().zip(u) { *w += s * ui; });
        check(&mut ctx, &dwv.data, "dwv", eps, tol,
            |c, s, u| for (w, &ui) in c.cell.wv.data.iter_mut().zip(u) { *w += s * ui; });
        check(&mut ctx, &dwo_out.data, "dw_out", eps, tol,
            |c, s, u| for (w, &ui) in c.cell.w_out.data.iter_mut().zip(u) { *w += s * ui; });
        check(&mut ctx, &dx.data, "dx", eps, tol,
            |c, s, u| for (v, &ui) in c.x.data.iter_mut().zip(u) { *v += s * ui; });
    }
}
