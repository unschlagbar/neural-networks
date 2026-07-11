//! Batched sLSTM cell (xLSTM, Beck et al. 2024), sequential over time.
//!
//! Direct port of `nn::slstm::SLSTMLayer`'s equations, but batched: the state
//! `(h, c, n, m)` is `[B, H]` and every gate pre-activation is a `[B, rows] ·
//! [rows, H]` GEMM instead of `B` separate matrix-vector products. Time is a
//! serial loop (irreducible for sLSTM); the batch is the parallel axis.
//!
//! One `forward` consumes a whole `[B, T, in]` sequence and returns `[B, T, H]`,
//! resetting the recurrent state to zero at the start (independent sequences of
//! equal length T). Cross-sequence state carrying — needed by the backbone — is
//! the caller's concern, layered on top later.
//!
//! Stabilizer `m_t` is treated as constant in backward (same approximation as
//! the reference and the old code): exact for |n_t| > 1, standard elsewhere.

use crate::nn2::optim::{AdamCfg, AdamState, ensure_states};
use crate::tensor::{Tensor, gemm};

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

/// Everything saved at one timestep for the backward pass. Each field is
/// `[B, H]` except `xh` which is `[B, rows]`. Pooled and reused across forward
/// calls (see `Step::fit`) so steady-state training allocates nothing per step.
struct Step {
    xh: Tensor,     // concat(x_t, h_{t-1})           [B, rows]
    c_prev: Tensor, // [B, H]
    n_prev: Tensor,
    ft_pre: Tensor, // f̃ pre-activation (for σ(f̃) in backward)
    zt: Tensor,     // tanh(z̃)
    ot: Tensor,     // σ(õ)
    i_prime: Tensor,
    f_prime: Tensor,
    c: Tensor,
    n: Tensor,
    psi: Tensor, // max(|n|, 1)
}

impl Step {
    /// Empty placeholder; `fit` sizes it on first use.
    fn new() -> Self {
        let z = || Tensor::zeros(&[0, 0]);
        Self {
            xh: z(), c_prev: z(), n_prev: z(), ft_pre: z(), zt: z(), ot: z(),
            i_prime: z(), f_prime: z(), c: z(), n: z(), psi: z(),
        }
    }

    /// Reuse all buffers at the given batch/hidden/rows (no alloc when unchanged).
    fn fit(&mut self, b: usize, h: usize, rows: usize) {
        self.xh.fit(&[b, rows]);
        for t in [
            &mut self.c_prev, &mut self.n_prev, &mut self.ft_pre, &mut self.zt,
            &mut self.ot, &mut self.i_prime, &mut self.f_prime, &mut self.c,
            &mut self.n, &mut self.psi,
        ] {
            t.fit(&[b, h]);
        }
    }
}

pub struct SLstm {
    pub input_size: usize,
    pub hidden_size: usize,

    // Gate weights, each [rows, H] with rows = input+hidden. Same layout as the
    // old code (concat-trick, one matrix per gate).
    pub wz: Tensor,
    pub wi: Tensor,
    pub wf: Tensor,
    pub wo: Tensor,
    pub bz: Tensor,
    pub bi: Tensor,
    pub bf: Tensor,
    pub bo: Tensor,

    // Gradient accumulators (same shapes).
    pub dwz: Tensor,
    pub dwi: Tensor,
    pub dwf: Tensor,
    pub dwo: Tensor,
    pub dbz: Tensor,
    pub dbi: Tensor,
    pub dbf: Tensor,
    pub dbo: Tensor,

    // --- pooled buffers, reused across forward/backward calls ---------------
    steps: Vec<Step>,
    // Recurrent state carried across timesteps within one call, [B, H].
    h_state: Tensor,
    c_state: Tensor,
    n_state: Tensor,
    m_state: Tensor,
    // Transient forward gate pre-activations (not needed in backward), [B, H].
    zt_pre: Tensor,
    it_pre: Tensor,
    ot_pre: Tensor,
    // Backward scratch: per-gate deltas [B, H], dxh [B, rows], BPTT channels [B, H].
    dz: Tensor,
    di: Tensor,
    df: Tensor,
    dob: Tensor,
    dxh: Tensor,
    dh_bptt: Tensor,
    dc_bptt: Tensor,
    dn_bptt: Tensor,
    // Transposed gate weights [H, rows], rebuilt once per backward so dxh uses
    // the fast gemm_nn instead of the reduction-form gemm_nt.
    wtz: Tensor,
    wti: Tensor,
    wtf: Tensor,
    wto: Tensor,
    opt: Vec<AdamState>,
    batch: usize,
}

impl SLstm {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let rows = input_size + hidden_size;
        let scale = (6.0 / rows as f32).sqrt();
        // Positive forget-gate bias (Jozefowicz 2015 / xLSTM convention).
        let bf: Vec<f32> = (0..hidden_size).map(|_| 4.5).collect();
        let z = || Tensor::zeros(&[0, 0]);
        Self {
            input_size,
            hidden_size,
            wz: Tensor::random(&[rows, hidden_size], scale),
            wi: Tensor::random(&[rows, hidden_size], scale),
            wf: Tensor::random(&[rows, hidden_size], scale),
            wo: Tensor::random(&[rows, hidden_size], scale),
            bz: Tensor::zeros(&[hidden_size]),
            bi: Tensor::zeros(&[hidden_size]),
            bf: Tensor::new(&[hidden_size], bf),
            bo: Tensor::zeros(&[hidden_size]),
            dwz: Tensor::zeros(&[rows, hidden_size]),
            dwi: Tensor::zeros(&[rows, hidden_size]),
            dwf: Tensor::zeros(&[rows, hidden_size]),
            dwo: Tensor::zeros(&[rows, hidden_size]),
            dbz: Tensor::zeros(&[hidden_size]),
            dbi: Tensor::zeros(&[hidden_size]),
            dbf: Tensor::zeros(&[hidden_size]),
            dbo: Tensor::zeros(&[hidden_size]),
            steps: Vec::new(),
            h_state: z(), c_state: z(), n_state: z(), m_state: z(),
            zt_pre: z(), it_pre: z(), ot_pre: z(),
            dz: z(), di: z(), df: z(), dob: z(), dxh: z(),
            dh_bptt: z(), dc_bptt: z(), dn_bptt: z(),
            wtz: z(), wti: z(), wtf: z(), wto: z(),
            opt: Vec::new(),
            batch: 0,
        }
    }

    #[inline]
    fn rows(&self) -> usize {
        self.input_size + self.hidden_size
    }

    /// Forward over a whole sequence. `x` is `[B, T, in]`; returns `[B, T, H]`.
    /// State resets to zero at t=0. All working buffers are pooled and reused
    /// across calls — steady state allocates only the returned output.
    pub fn forward(&mut self, x: &Tensor) -> Tensor {
        assert_eq!(x.rank(), 3, "SLstm::forward expects [B, T, in]");
        let (b, t, inp) = (x.shape[0], x.shape[1], x.shape[2]);
        assert_eq!(inp, self.input_size, "SLstm::forward — input width mismatch");
        let h = self.hidden_size;
        let rows = self.rows();
        self.batch = b;

        // Size (reuse) the pooled buffers; reset the recurrent state to zero.
        self.h_state.fit(&[b, h]);
        self.c_state.fit(&[b, h]);
        self.n_state.fit(&[b, h]);
        self.m_state.fit(&[b, h]);
        self.h_state.zero_();
        self.c_state.zero_();
        self.n_state.zero_();
        self.m_state.zero_();
        self.zt_pre.fit(&[b, h]);
        self.it_pre.fit(&[b, h]);
        self.ot_pre.fit(&[b, h]);
        while self.steps.len() < t {
            self.steps.push(Step::new());
        }
        for s in 0..t {
            self.steps[s].fit(b, h, rows);
        }

        let mut out = Tensor::zeros(&[b, t, h]);

        // Disjoint field borrows so the per-step loop can hold several buffers
        // mutably at once without going through `self` methods.
        let Self {
            steps, wz, wi, wf, wo, bz, bi, bf, bo, zt_pre, it_pre, ot_pre,
            h_state, c_state, n_state, m_state, ..
        } = self;

        for step in 0..t {
            let st = &mut steps[step];

            // Build xh = concat(x_t, h_{t-1}) as [B, rows].
            for bi_ in 0..b {
                let xsrc = &x.data[(bi_ * t + step) * inp..(bi_ * t + step) * inp + inp];
                let row = &mut st.xh.data[bi_ * rows..(bi_ + 1) * rows];
                row[..inp].copy_from_slice(xsrc);
                row[inp..].copy_from_slice(&h_state.data[bi_ * h..(bi_ + 1) * h]);
            }

            // Gate pre-activations: [B, rows] · [rows, H] -> [B, H], + bias.
            // z/i/o go to transient scratch; f is stored (needed in backward).
            gemm::gemm_nn(b, rows, h, &st.xh.data, &wz.data, &mut zt_pre.data, 0.0);
            gemm::gemm_nn(b, rows, h, &st.xh.data, &wi.data, &mut it_pre.data, 0.0);
            gemm::gemm_nn(b, rows, h, &st.xh.data, &wf.data, &mut st.ft_pre.data, 0.0);
            gemm::gemm_nn(b, rows, h, &st.xh.data, &wo.data, &mut ot_pre.data, 0.0);
            add_bias(&mut zt_pre.data, &bz.data);
            add_bias(&mut it_pre.data, &bi.data);
            add_bias(&mut st.ft_pre.data, &bf.data);
            add_bias(&mut ot_pre.data, &bo.data);

            // Save previous states into the step (for backward).
            st.c_prev.data.copy_from_slice(&c_state.data);
            st.n_prev.data.copy_from_slice(&n_state.data);

            // Elementwise cell over B*H.
            for k in 0..b * h {
                let z = zt_pre.data[k].tanh();
                let o = stable_sigmoid(ot_pre.data[k]);
                let log_f = log_sigmoid(st.ft_pre.data[k]);
                let fm = log_f + m_state.data[k];
                let m = fm.max(it_pre.data[k]);
                let ip = (it_pre.data[k] - m).exp();
                let fp = (fm - m).exp();
                let c = fp * st.c_prev.data[k] + ip * z;
                let n = fp * st.n_prev.data[k] + ip;
                let p = n.abs().max(1.0);

                st.zt.data[k] = z;
                st.ot.data[k] = o;
                st.i_prime.data[k] = ip;
                st.f_prime.data[k] = fp;
                st.c.data[k] = c;
                st.n.data[k] = n;
                st.psi.data[k] = p;

                // Advance the recurrent state for the next step.
                c_state.data[k] = c;
                n_state.data[k] = n;
                m_state.data[k] = m;
                let hh = o * c / p;
                h_state.data[k] = hh;

                let bi_ = k / h;
                let j = k % h;
                out.data[(bi_ * t + step) * h + j] = hh;
            }
        }

        out
    }

    /// Backward over the whole sequence. `dy` is `[B, T, H]`; returns
    /// `dx` `[B, T, in]`. Accumulates weight/bias grads.
    pub fn backward(&mut self, dy: &Tensor) -> Tensor {
        assert_eq!(dy.rank(), 3, "SLstm::backward expects [B, T, H]");
        let (b, t, h) = (dy.shape[0], dy.shape[1], dy.shape[2]);
        assert_eq!(b, self.batch, "SLstm::backward — batch mismatch");
        assert_eq!(h, self.hidden_size, "SLstm::backward — hidden mismatch");
        let inp = self.input_size;
        let rows = self.rows();

        let mut dx = Tensor::zeros(&[b, t, inp]);

        // Size (reuse) backward scratch; zero the BPTT channels (start at 0).
        for buf in [
            &mut self.dz, &mut self.di, &mut self.df, &mut self.dob,
            &mut self.dh_bptt, &mut self.dc_bptt, &mut self.dn_bptt,
        ] {
            buf.fit(&[b, h]);
        }
        self.dxh.fit(&[b, rows]);
        self.dh_bptt.zero_();
        self.dc_bptt.zero_();
        self.dn_bptt.zero_();

        // Transpose the gate weights [rows,H] -> [H,rows] once; reused across all
        // T steps so dxh can use the fast gemm_nn instead of gemm_nt.
        self.wtz.fit(&[h, rows]);
        self.wti.fit(&[h, rows]);
        self.wtf.fit(&[h, rows]);
        self.wto.fit(&[h, rows]);
        gemm::transpose(rows, h, &self.wz.data, &mut self.wtz.data);
        gemm::transpose(rows, h, &self.wi.data, &mut self.wti.data);
        gemm::transpose(rows, h, &self.wf.data, &mut self.wtf.data);
        gemm::transpose(rows, h, &self.wo.data, &mut self.wto.data);

        let Self {
            steps, dwz, dwi, dwf, dwo, dbz, dbi, dbf, dbo,
            dz, di, df, dob, dxh, dh_bptt, dc_bptt, dn_bptt,
            wtz, wti, wtf, wto, ..
        } = self;

        for step in (0..t).rev() {
            let st = &steps[step];

            for k in 0..b * h {
                let bi = k / h;
                let j = k % h;
                let d = dy.data[(bi * t + step) * h + j] + dh_bptt.data[k];
                let psi = st.psi.data[k];
                let o = st.ot.data[k];
                let c = st.c.data[k];
                let n = st.n.data[k];

                dob.data[k] = d * (c / psi) * o * (1.0 - o);

                let dn_from_h = if n.abs() > 1.0 {
                    let dpsi = d * o * (-c) / (psi * psi);
                    dpsi * n.signum()
                } else {
                    0.0
                };

                let dc = d * o / psi + dc_bptt.data[k];
                let dn = dn_from_h + dn_bptt.data[k];

                let df_prime = dc * st.c_prev.data[k] + dn * st.n_prev.data[k];
                let di_prime = dc * st.zt.data[k] + dn;
                let dz_post = dc * st.i_prime.data[k];

                dz.data[k] = dz_post * (1.0 - st.zt.data[k] * st.zt.data[k]);
                di.data[k] = di_prime * st.i_prime.data[k];
                let sig_f = stable_sigmoid(st.ft_pre.data[k]);
                df.data[k] = df_prime * st.f_prime.data[k] * (1.0 - sig_f);

                dc_bptt.data[k] = dc * st.f_prime.data[k];
                dn_bptt.data[k] = dn * st.f_prime.data[k];
            }

            // Bias grads: sum gate deltas over the batch.
            accum_bias(&mut dbz.data, &dz.data, b, h);
            accum_bias(&mut dbi.data, &di.data, b, h);
            accum_bias(&mut dbf.data, &df.data, b, h);
            accum_bias(&mut dbo.data, &dob.data, b, h);

            // Weight grads: dW += xhᵀ · dGate   ([rows,B]·[B,H] -> [rows,H]).
            gemm::gemm_tn(rows, b, h, &st.xh.data, &dz.data, &mut dwz.data, 1.0);
            gemm::gemm_tn(rows, b, h, &st.xh.data, &di.data, &mut dwi.data, 1.0);
            gemm::gemm_tn(rows, b, h, &st.xh.data, &df.data, &mut dwf.data, 1.0);
            gemm::gemm_tn(rows, b, h, &st.xh.data, &dob.data, &mut dwo.data, 1.0);

            // dxh = Σ_gate dGate · Wᵀ = Σ_gate dGate · Wt   ([B,H]·[H,rows] -> [B,rows]).
            gemm::gemm_nn(b, h, rows, &dz.data, &wtz.data, &mut dxh.data, 0.0);
            gemm::gemm_nn(b, h, rows, &di.data, &wti.data, &mut dxh.data, 1.0);
            gemm::gemm_nn(b, h, rows, &df.data, &wtf.data, &mut dxh.data, 1.0);
            gemm::gemm_nn(b, h, rows, &dob.data, &wto.data, &mut dxh.data, 1.0);

            // Split dxh into dx (first `inp`) and dh_bptt (remaining `h`).
            for bi in 0..b {
                let src = &dxh.data[bi * rows..(bi + 1) * rows];
                dx.data[(bi * t + step) * inp..(bi * t + step) * inp + inp]
                    .copy_from_slice(&src[..inp]);
                dh_bptt.data[bi * h..(bi + 1) * h].copy_from_slice(&src[inp..]);
            }
        }

        dx
    }

    pub fn zero_grad(&mut self) {
        for g in [
            &mut self.dwz, &mut self.dwi, &mut self.dwf, &mut self.dwo,
            &mut self.dbz, &mut self.dbi, &mut self.dbf, &mut self.dbo,
        ] {
            g.zero_();
        }
    }

    /// AdamW step: gate matrices decay, biases don't. Clears the grads.
    pub fn step(&mut self, cfg: &AdamCfg) {
        ensure_states(&mut self.opt, 8);
        self.opt[0].step(&mut self.wz.data, &self.dwz.data, cfg, true);
        self.opt[1].step(&mut self.wi.data, &self.dwi.data, cfg, true);
        self.opt[2].step(&mut self.wf.data, &self.dwf.data, cfg, true);
        self.opt[3].step(&mut self.wo.data, &self.dwo.data, cfg, true);
        self.opt[4].step(&mut self.bz.data, &self.dbz.data, cfg, false);
        self.opt[5].step(&mut self.bi.data, &self.dbi.data, cfg, false);
        self.opt[6].step(&mut self.bf.data, &self.dbf.data, cfg, false);
        self.opt[7].step(&mut self.bo.data, &self.dbo.data, cfg, false);
        self.zero_grad();
    }
}

/// Broadcast-add a `[H]` bias over the batch rows of a `[B, H]` buffer.
fn add_bias(data: &mut [f32], bias: &[f32]) {
    let h = bias.len();
    for row in data.chunks_mut(h) {
        for (v, &bv) in row.iter_mut().zip(bias) {
            *v += bv;
        }
    }
}

/// Accumulate `Σ_batch grad` into a `[H]` bias-grad accumulator.
fn accum_bias(dbias: &mut [f32], grad: &[f32], b: usize, h: usize) {
    for bi in 0..b {
        let row = &grad[bi * h..(bi + 1) * h];
        for (a, &g) in dbias.iter_mut().zip(row) {
            *a += g;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Directional finite-difference check (whole-tensor projection along the
    /// normalized analytic gradient): (L(p+εu) − L(p−εu)) / 2ε with u = G/‖G‖
    /// must equal ‖G‖. This averages the per-element max-stabilizer kink noise
    /// that makes element-wise FD undershoot by up to ~20% on stabilized sLSTM
    /// cells (see the hierarchical FD tests). The loose tolerance still catches
    /// routing/plumbing bugs, which miss by far more. `perturb` moves the chosen
    /// parameter (or input) by a step along `u`; `loss` re-runs the forward.
    /// Bundles everything the perturb/loss closures need, so both take one
    /// `&mut Ctx` and never alias-borrow the cell.
    struct Ctx {
        cell: SLstm,
        x: Tensor, // perturbed for the dx check, fixed otherwise
        g: Tensor,
    }

    fn check_direction<S>(
        state: &mut S,
        grad: &[f32],
        name: &str,
        eps: f32,
        tol: f32,
        mut perturb: impl FnMut(&mut S, f32, &[f32]),
        mut loss: impl FnMut(&mut S) -> f32,
    ) {
        let norm: f32 = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
        assert!(norm > 1e-6, "{name}: analytic gradient ~zero, check is vacuous");
        let u: Vec<f32> = grad.iter().map(|g| g / norm).collect();

        perturb(state, eps, &u);
        let plus = loss(state);
        perturb(state, -2.0 * eps, &u);
        let minus = loss(state);
        perturb(state, eps, &u); // restore

        let fd = (plus - minus) / (2.0 * eps);
        assert!(
            (fd - norm).abs() <= tol * norm + 1e-3,
            "{name}: analytic ‖G‖ {norm} vs finite difference {fd}"
        );
    }

    // Loss L = Σ (Y ⊙ G); dL/dY = G, so backward(&g) yields the analytic grads.
    fn loss(c: &mut Ctx) -> f32 {
        let y = c.cell.forward(&c.x);
        y.data.iter().zip(&c.g.data).map(|(a, b)| a * b).sum()
    }

    #[test]
    fn backward_matches_finite_difference() {
        let (b, t, inp, h) = (2, 4, 4, 5);
        let mut ctx = Ctx {
            cell: SLstm::new(inp, h),
            x: Tensor::random(&[b, t, inp], 0.5),
            g: Tensor::random(&[b, t, h], 1.0),
        };

        let _y = ctx.cell.forward(&ctx.x);
        let g = ctx.g.clone();
        let dx = ctx.cell.backward(&g);
        let dwz = ctx.cell.dwz.clone();
        let dwf = ctx.cell.dwf.clone();

        let eps = 2e-4;
        let tol = 0.3; // stabilizer kinks — same tolerance as the hierarchical sLSTM FD tests

        check_direction(
            &mut ctx, &dwz.data, "dwz", eps, tol,
            |c, step, u| for (w, &ui) in c.cell.wz.data.iter_mut().zip(u) { *w += step * ui; },
            loss,
        );

        // Forget-gate weights, exercising the log-sigmoid / stabilizer path.
        check_direction(
            &mut ctx, &dwf.data, "dwf", eps, tol,
            |c, step, u| for (w, &ui) in c.cell.wf.data.iter_mut().zip(u) { *w += step * ui; },
            loss,
        );

        // dx: perturb the input along its gradient direction.
        check_direction(
            &mut ctx, &dx.data, "dx", eps, tol,
            |c, step, u| for (v, &ui) in c.x.data.iter_mut().zip(u) { *v += step * ui; },
            loss,
        );
    }
}
