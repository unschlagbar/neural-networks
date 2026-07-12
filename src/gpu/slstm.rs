//! Device-resident batched sLSTM cell, the GPU counterpart of
//! [`nn2::slstm::SLstm`](crate::nn2::slstm::SLstm).
//!
//! Same equations, same weight layout (4 gates `[rows, H]`, `rows = in + H`,
//! concat-trick), and the same AdamW convention (gate matrices decay, biases do
//! not), so a GPU cell built from a CPU cell's weights matches it for
//! forward → backward → step — which the parity test checks against `nn2::SLstm`.
//!
//! Time is a serial loop; the batch is the parallel axis. **The whole recurrent
//! state `(h,c,n,m)` stays resident in `DTensor`s across the entire T-loop** — no
//! per-step host transfer.
//!
//! The four gates run **fused**: a timestep is one cuBLAS GEMM plus one kernel.
//! That matters because the backbone runs this cell at batch 1 over ~2000 words,
//! where every launch is pure latency — the per-timestep GEMM there is a
//! matrix-vector product that takes far less time than the launch itself, so the
//! step cost is simply the number of launches.
//!
//! Concretely, per timestep `t`:
//!   * `x·Wx` for **all** timesteps is one GEMM hoisted out of the loop (it has no
//!     recurrent dependency), landing in a `[B, T, 4H]` gate buffer `g`;
//!   * the loop only adds the recurrent half, `g[:, t, :] += h_{t-1}·Wh`;
//!   * `slstm_step_fused` adds the biases and runs the elementwise recurrence.
//!
//! Backward mirrors it: the per-step kernel writes the four gate deltas back into
//! `g` (its forward contents are dead by then), the loop carries only the BPTT
//! channels — `dh = dg[:, t, :]·Whᵀ` — and `dx`, `dWx`, `dWh` and the bias grads
//! all fall out of three whole-sequence GEMMs plus one reduction *after* the loop.
//!
//! The gate weights of record stay the four `[rows, H]` matrices `nn2::SLstm` and
//! the checkpoints use; `slstm_pack` derives the fused `Wx`/`Wh`/`bias` operands
//! from them each forward, and `slstm_unpack_dw` folds the fused gradients back.

use super::ops::{self, SlstmSlabs};
use super::{DTensor, Gpu};
use crate::nn2::optim::AdamCfg;
use crate::tensor::Tensor;

pub struct SLstm {
    input: usize,
    hidden: usize,

    // Gate weights/biases and their grads/moments, indexed z=0, i=1, f=2, o=3.
    // These are the parameters of record: the optimizer steps them and the
    // checkpoint stores them. The fused operands below are derived from them.
    pub w: [DTensor; 4],    // [rows, H]
    pub bias: [DTensor; 4], // [H]
    dw: [DTensor; 4],
    db: [DTensor; 4],
    mw: [DTensor; 4],
    vw: [DTensor; 4],
    mb: [DTensor; 4],
    vb: [DTensor; 4],

    // Fused operands, repacked from `w`/`bias` at the top of every forward.
    wx: DTensor,   // [in, 4H]
    whr: DTensor,  // [H, 4H]  (recurrent half)
    bcat: DTensor, // [4H]

    // Recurrent state carried across timesteps within one call, [B, H].
    h_state: DTensor,
    c_state: DTensor,
    n_state: DTensor,
    m_state: DTensor,
    /// Contiguous `[B, 4H]` scratch for the current timestep's recurrent gate half
    /// (`h_{t-1}·Wh` forward, the gate deltas backward). It exists so both of those
    /// GEMMs stay dense at any batch size — see `slstm_step_fused` in `kernels.rs`.
    gh: DTensor,
    /// `[1, N]` of ones and a `[1, 4H]` landing pad: the bias gradient is the column
    /// sum of the gate deltas, which cuBLAS reduces as a `ones · dgates` GEMM.
    ones: DTensor,
    dbcat: DTensor,
    // BPTT channels, [B, H].
    dh_bptt: DTensor,
    dc_bptt: DTensor,
    dn_bptt: DTensor,

    // Handed from forward to backward: the gate buffer [B, T, 4H], the saved
    // [B, T, H] slabs, and the flattened input [B·T, in] (needed for dWx).
    g: Option<DTensor>,
    slabs: Option<SlstmSlabs>,
    x_saved: Option<DTensor>,
    batch: usize,
}

/// Reuse `t`'s device buffer when the shape matches, zeroing it in place; else
/// (re)allocate a zeroed buffer. For state / BPTT channels that must start at 0.
fn fit_zeros(gpu: &Gpu, t: &mut DTensor, dims: &[usize]) {
    if t.dims() == dims {
        t.zero_(gpu);
    } else {
        *t = DTensor::zeros(gpu, dims);
    }
}

/// Reuse `t`'s device buffer when the shape matches (leaving its contents); else
/// (re)allocate uninitialised. For outputs a kernel/GEMM overwrites in full.
fn fit_uninit(gpu: &Gpu, t: &mut DTensor, dims: &[usize]) {
    if t.dims() != dims {
        *t = DTensor::uninit(gpu, dims);
    }
}

impl SLstm {
    /// Build from a CPU cell's host weights (gate order z, i, f, o). The `w{*}`
    /// are `[rows, H]` and the `b{*}` are `[H]`; they are uploaded to the device.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        gpu: &Gpu,
        input: usize,
        hidden: usize,
        wz: &Tensor, wi: &Tensor, wf: &Tensor, wo: &Tensor,
        bz: &Tensor, bi: &Tensor, bf: &Tensor, bo: &Tensor,
    ) -> Self {
        let rows = input + hidden;
        let up = |t: &Tensor| DTensor::from_host(gpu, t);
        let zw = || DTensor::zeros(gpu, &[rows, hidden]);
        let zb = || DTensor::zeros(gpu, &[hidden]);
        Self {
            input,
            hidden,
            w: [up(wz), up(wi), up(wf), up(wo)],
            bias: [up(bz), up(bi), up(bf), up(bo)],
            dw: [zw(), zw(), zw(), zw()],
            db: [zb(), zb(), zb(), zb()],
            mw: [zw(), zw(), zw(), zw()],
            vw: [zw(), zw(), zw(), zw()],
            mb: [zb(), zb(), zb(), zb()],
            vb: [zb(), zb(), zb(), zb()],
            wx: DTensor::zeros(gpu, &[0, 0]),
            whr: DTensor::zeros(gpu, &[0, 0]),
            bcat: DTensor::zeros(gpu, &[0]),
            h_state: DTensor::zeros(gpu, &[0, 0]),
            c_state: DTensor::zeros(gpu, &[0, 0]),
            n_state: DTensor::zeros(gpu, &[0, 0]),
            m_state: DTensor::zeros(gpu, &[0, 0]),
            gh: DTensor::zeros(gpu, &[0, 0]),
            ones: DTensor::zeros(gpu, &[0, 0]),
            dbcat: DTensor::zeros(gpu, &[0, 0]),
            dh_bptt: DTensor::zeros(gpu, &[0, 0]),
            dc_bptt: DTensor::zeros(gpu, &[0, 0]),
            dn_bptt: DTensor::zeros(gpu, &[0, 0]),
            g: None,
            slabs: None,
            x_saved: None,
            batch: 0,
        }
    }

    /// Freshly-initialised cell, matching `nn2::SLstm::new`'s init exactly
    /// (including the +4.5 forget-gate bias).
    pub fn new_rand(gpu: &Gpu, input: usize, hidden: usize) -> Self {
        Self::from_cpu(gpu, &crate::nn2::SLstm::new(input, hidden))
    }

    /// Upload a CPU cell (weights are copied; grads/moments start at zero).
    pub fn from_cpu(gpu: &Gpu, cpu: &crate::nn2::SLstm) -> Self {
        Self::from_parts(
            gpu, cpu.input_size, cpu.hidden_size,
            &cpu.wz, &cpu.wi, &cpu.wf, &cpu.wo,
            &cpu.bz, &cpu.bi, &cpu.bf, &cpu.bo,
        )
    }

    #[inline]
    pub fn input_size(&self) -> usize {
        self.input
    }
    #[inline]
    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    /// Forward over a whole `[B, T, in]` sequence → `[B, T, H]`. State resets to
    /// zero at t=0 and stays device-resident across the T-loop.
    pub fn forward(&mut self, gpu: &Gpu, x: &DTensor) -> DTensor {
        assert_eq!(x.rank, 3, "SLstm::forward expects [B, T, in]");
        let (b, t, inp) = (x.shape[0], x.shape[1], x.shape[2]);
        assert_eq!(inp, self.input, "SLstm::forward — input width mismatch");
        let h = self.hidden;
        let h4 = 4 * h;
        let n = b * t;
        self.batch = b;

        // Rebuild the fused operands from the gate weights the optimizer owns.
        fit_uninit(gpu, &mut self.wx, &[inp, h4]);
        fit_uninit(gpu, &mut self.whr, &[h, h4]);
        fit_uninit(gpu, &mut self.bcat, &[h4]);
        ops::slstm_pack(
            gpu, &self.w, &self.bias, &mut self.wx, &mut self.whr, &mut self.bcat, inp, h,
        );

        // Recurrent state starts at zero.
        for s in [&mut self.h_state, &mut self.c_state, &mut self.n_state, &mut self.m_state] {
            fit_zeros(gpu, s, &[b, h]);
        }

        // The input half of every gate pre-activation, for all timesteps at once —
        // it has no recurrent dependency, so it is one GEMM outside the loop.
        let x_flat = x.dup(gpu).reshaped(&[n, inp]);
        let mut g = DTensor::uninit(gpu, &[n, h4]);
        ops::matmul_nn_into(gpu, &x_flat, &self.wx, &mut g, 0.0);
        let mut g = g.reshaped(&[b, t, h4]);

        let slab = || DTensor::uninit(gpu, &[b, t, h]);
        let mut slabs = SlstmSlabs {
            c_prev: slab(), n_prev: slab(), zt: slab(), ot: slab(), i_prime: slab(),
            f_prime: slab(), c: slab(), n: slab(), psi: slab(), h_prev: slab(),
        };
        let mut out = DTensor::uninit(gpu, &[b, t, h]);
        fit_uninit(gpu, &mut self.gh, &[b, h4]);

        for step in 0..t {
            // Recurrent half of the gates (one dense GEMM into the contiguous
            // scratch), then the elementwise recurrence: two launches per timestep.
            ops::matmul_nn_into(gpu, &self.h_state, &self.whr, &mut self.gh, 0.0);
            ops::slstm_step_fused(
                gpu, &mut g, &self.gh, &self.bcat,
                &mut self.c_state, &mut self.n_state, &mut self.m_state, &mut self.h_state,
                &mut slabs, &mut out, step,
            );
        }

        self.g = Some(g);
        self.slabs = Some(slabs);
        self.x_saved = Some(x_flat);
        out
    }

    /// Backward over the whole sequence. `dy` is `[B, T, H]`; returns
    /// `dx` `[B, T, in]`. Accumulates weight/bias grads.
    pub fn backward(&mut self, gpu: &Gpu, dy: &DTensor) -> DTensor {
        assert_eq!(dy.rank, 3, "SLstm::backward expects [B, T, H]");
        let (b, t, h) = (dy.shape[0], dy.shape[1], dy.shape[2]);
        assert_eq!(b, self.batch, "SLstm::backward — batch mismatch");
        assert_eq!(h, self.hidden, "SLstm::backward — hidden mismatch");
        let inp = self.input;
        let h4 = 4 * h;
        let n = b * t;

        // Taken, not borrowed: these are rebuilt by every forward, so releasing them
        // here frees the device memory across the optimizer step.
        let mut g = self.g.take().expect("forward before backward");
        let slabs = self.slabs.take().expect("forward before backward");
        let x_flat = self.x_saved.take().expect("forward before backward");

        for buf in [&mut self.dh_bptt, &mut self.dc_bptt, &mut self.dn_bptt] {
            fit_zeros(gpu, buf, &[b, h]);
        }
        fit_uninit(gpu, &mut self.gh, &[b, h4]);

        // The only thing the loop must carry is BPTT: the gate deltas go straight
        // back into `g`, and everything derived from them waits until the loop ends.
        for step in (0..t).rev() {
            ops::slstm_step_fused_bwd(
                gpu, dy, &mut g, &mut self.gh, &self.dh_bptt, &slabs,
                &mut self.dc_bptt, &mut self.dn_bptt, step,
            );
            // dh_{t-1} = dgates_t · Whᵀ — the one gradient BPTT cannot defer.
            ops::matmul_nt_into(gpu, &self.gh, &self.whr, &mut self.dh_bptt, 0.0);
        }

        // `g` now holds the gate deltas for the whole sequence: dx, dWx, dWh and the
        // bias grads are three GEMMs and one reduction over it.
        let dg = g.reshaped(&[n, h4]);
        let dx = ops::matmul_nt(gpu, &dg, &self.wx); // [N, in]

        let mut dwx = DTensor::uninit(gpu, &[inp, h4]);
        ops::matmul_tn_into(gpu, &x_flat, &dg, &mut dwx, 0.0);
        let h_prev = slabs.h_prev.reshaped(&[n, h]);
        let mut dwh = DTensor::uninit(gpu, &[h, h4]);
        ops::matmul_tn_into(gpu, &h_prev, &dg, &mut dwh, 0.0);

        ops::slstm_unpack_dw(gpu, &dwx, &dwh, &mut self.dw, inp, h);

        fit_uninit(gpu, &mut self.ones, &[1, n]);
        ops::fill(gpu, &mut self.ones, 1.0);
        fit_uninit(gpu, &mut self.dbcat, &[1, h4]);
        ops::slstm_db_from_dg(gpu, &dg, &self.ones, &mut self.dbcat, &mut self.db, h);

        dx.reshaped(&[b, t, inp])
    }

    /// Every learnable tensor, in a fixed order (used by checkpoint save/load).
    pub fn params_mut(&mut self) -> Vec<&mut DTensor> {
        self.w.iter_mut().chain(self.bias.iter_mut()).collect()
    }

    pub fn zero_grad(&mut self, gpu: &Gpu) {
        for g in self.dw.iter_mut().chain(self.db.iter_mut()) {
            g.zero_(gpu);
        }
    }

    /// AdamW step: gate matrices decay, biases don't. Clears the grads.
    pub fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        for i in 0..4 {
            ops::adamw(gpu, &mut self.w[i], &self.dw[i], &mut self.mw[i], &mut self.vw[i], cfg, true);
        }
        for i in 0..4 {
            ops::adamw(gpu, &mut self.bias[i], &self.db[i], &mut self.mb[i], &mut self.vb[i], cfg, false);
        }
        self.zero_grad(gpu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn2::optim::AdamCfg;
    use crate::nn2::slstm::SLstm as CpuSLstm;

    fn assert_close(got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len(), "length mismatch");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < tol, "index {i}: gpu {g} vs cpu {w}");
        }
    }

    fn from_cpu(gpu: &Gpu, cpu: &CpuSLstm) -> SLstm {
        SLstm::from_parts(
            gpu, cpu.input_size, cpu.hidden_size,
            &cpu.wz, &cpu.wi, &cpu.wf, &cpu.wo,
            &cpu.bz, &cpu.bi, &cpu.bf, &cpu.bo,
        )
    }

    /// GPU sLSTM must match `nn2::SLstm` (cell alone) for a full
    /// forward → backward → AdamW-step cycle, from identical weights. Tolerance
    /// is loose-ish because the two paths differ in float reduction order (cuBLAS
    /// vs the CPU gemm), but the recurrence math is identical.
    #[test]
    fn slstm_matches_cpu_layer() {
        let Some(gpu) = super::super::test_gpu() else { return };
        let (b, t, inp, h) = (2, 5, 4, 6);

        let mut cpu = CpuSLstm::new(inp, h);
        let mut dev = from_cpu(&gpu, &cpu);

        let x = Tensor::random(&[b, t, inp], 0.5);
        let g = Tensor::random(&[b, t, h], 1.0);

        // Forward
        let y_cpu = cpu.forward(&x);
        let y_dev = dev.forward(&gpu, &DTensor::from_host(&gpu, &x));
        assert_close(&y_dev.to_host(&gpu).data, &y_cpu.data, 2e-3);

        // Backward
        let dx_cpu = cpu.backward(&g);
        let dx_dev = dev.backward(&gpu, &DTensor::from_host(&gpu, &g));
        assert_close(&dx_dev.to_host(&gpu).data, &dx_cpu.data, 2e-3);
        assert_close(&dev.dw[0].to_host(&gpu).data, &cpu.dwz.data, 2e-3);
        assert_close(&dev.dw[2].to_host(&gpu).data, &cpu.dwf.data, 2e-3);
        assert_close(&dev.db[2].to_host(&gpu).data, &cpu.dbf.data, 2e-3);

        // One AdamW step, then compare the updated gate weights.
        let mut cfg = AdamCfg::new(1e-3, 0.01);
        cfg.t = 1;
        cpu.step(&cfg);
        dev.step(&gpu, &cfg);
        // Looser than Linear's 1e-5: the AdamW update is ~lr in magnitude, and a
        // near-zero grad element can sign-flip between the cuBLAS and CPU gemm
        // reduction orders, swinging its update by ~2·lr. A plumbing bug misses by
        // O(weight), far more than this.
        assert_close(&dev.w[0].to_host(&gpu).data, &cpu.wz.data, 2e-3);
        assert_close(&dev.w[2].to_host(&gpu).data, &cpu.wf.data, 2e-3);
        assert_close(&dev.bias[2].to_host(&gpu).data, &cpu.bf.data, 2e-3);
    }
}
