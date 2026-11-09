//! Device-resident batched sLSTM cell, the GPU counterpart of
//! [`cpu::slstm::SLstm`](crate::cpu::slstm::SLstm).
//!
//! Weights are stored fused, gate order z, i, f, o: `wx [in, 4H]`, `whr [H, 4H]`,
//! `bcat [4H]`. The cell owns its post-cell RMSNorm.

use super::arena::{self, ParamKind, ParamSlot};
use super::block::{Resets, phase};
use super::offload::{Frame, ResidentFrames};
use super::ops::{self, GpuRmsForward, SlabBuf, SlstmSlabs};
use super::rms_norm::RmsNorm;
use super::{Buf, GTensor, Gpu};
use crate::gpu::arena::TrainingCache;
use crate::cpu::optim::AdamCfg;
use crate::tensor::Tensor;

/// Below this sequence length the T-loop runs step by step instead of as one
/// time-fused cooperative launch. The fused kernel amortises its `Wh` staging over
/// the whole sequence, so a handful of timesteps never earns it back — and the
/// encoder/decoder call this cell with T = a word length (<= MAX_WORD_BYTES + 1),
/// which is exactly that case. The backbone, where T is a whole chunk of words,
/// fuses.
const FUSED_MIN_T: usize = 32;

/// Above this batch the time-fused path stops paying and the per-step path wins.
const FUSED_MAX_B: usize = 32;

/// Whether the sLSTM's own weight gradients are held at bf16 precision
/// (`SLSTM_BF16_GRAD=1`), matching FlashRNN's `dR`/`db`.
fn grad_bf16(gpu: &Gpu) -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SLSTM_BF16_GRAD").as_deref() == Ok("1"))
        && gpu.kernels.has_bf16
}

/// The time-fused T-loop (`slstm_fused_time` / `_bwd`): the whole forward or
/// backward sequence as ONE cooperative launch instead of two launches per
/// timestep. **On by default**; `SLSTM_NO_FUSED_TIME=1` forces the per-step path,
/// which is the A/B baseline and the fallback if a driver ever mis-schedules a
/// cooperative grid.
fn fused_time_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SLSTM_NO_FUSED_TIME").is_err())
}

/// Narrowest batch the batch-parallel fused path takes. The mma M axis is 16 rows
/// wide, so below a couple of tiles the contraction is mostly padding and the
/// per-step cuBLAS call is doing the same thing with less setup.
const SB_MIN_B: usize = 32;

/// The batch-parallel fused T-loop (`slstm_batched_fwd`). **On by default**;
/// `SLSTM_NO_BATCHED=1` forces the per-step path, which is the A/B baseline.
fn batched_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SLSTM_NO_BATCHED").is_err())
}

pub struct SLstm {
    input: usize,
    hidden: usize,

    /// Post-cell RMSNorm, applied to the cell's output before it leaves `forward`.
    post_norm: RmsNorm,

    pub wx: GTensor<f32>,   // [in, 4H]   input half
    pub whr: GTensor<f32>,  // [H, 4H]    recurrent half
    pub bcat: GTensor<f32>, // [4H]
    dwx: GTensor<f32>,
    dwhr: GTensor<f32>,
    dbcat: GTensor<f32>,
    mwx: GTensor<f32>,
    vwx: GTensor<f32>,
    mwhr: GTensor<f32>,
    vwhr: GTensor<f32>,
    mbcat: GTensor<f32>,
    vbcat: GTensor<f32>,

    /// Per-instance override of the time-fused path. `None` follows the global
    /// default (see [`fused_time_enabled`]); `Some(false)` pins this cell to the
    /// per-step loop and `Some(true)` to the fused one. The env flag is read through
    /// a `OnceLock`, so a test that needs BOTH paths in one process sets this — and
    /// it must set it on both cells, since the default alone no longer distinguishes
    /// them.
    pub force_fused_time: Option<bool>,

    /// Per-instance override of the batch-parallel fused path (`slstm_batched_fwd`),
    /// the mma-based twin of the time-fused one for wide batches. `None` follows the
    /// global default (see [`batched_enabled`]).
    pub force_batched: Option<bool>,

    /// Continue the previous call's recurrence instead of starting from zero.
    carry: bool,

    // Recurrent state carried across timesteps within one call, [B, H].
    h_state: GTensor<f32>,
    c_state: GTensor<f32>,
    n_state: GTensor<f32>,
    m_state: GTensor<f32>,
    /// Contiguous `[B, 4H]` scratch for the current timestep's recurrent gate half
    /// (`h_{t-1}·Wh`). It exists so that GEMM stays dense at any batch size — see
    /// `slstm_step_fused` in `kernels.rs`.
    gh: GTensor<f32>,
    /// The backward twin of [`gh`](Self::gh): this timestep's gate deltas, `[B, 4H]`,
    /// at the slab width. `slstm_step_fused_bwd` writes it, `dh = dg·Whᵀ` reads it —
    /// nothing else — so it is narrow for the same reason [`h_narrow`](Self::h_narrow)
    dgh: ops::SlabBuf,
    /// The left operand of the per-timestep recurrent GEMM: `h_{t-1}` at the slab
    /// width, `[B, H]`.
    h_narrow: ops::SlabBuf,
    // BPTT channels, [B, H].
    dh_bptt: GTensor<f32>,
    dc_bptt: GTensor<f32>,
    dn_bptt: GTensor<f32>,

    /// The loop's output before the post-cell norm, and where that norm's backward
    /// lands — scratch this cell reuses call to call.
    out_buf: Buf,
    dy_buf: Buf,
    /// Frames of a cell used on its own. Inside a [`Block`](super::block::Block) the
    /// block's frame holds this cell's cache and this stays empty.
    frames: ResidentFrames,
    /// `(B, T)` of the last forward, for [`state_extremes`](Self::state_extremes).
    last_shape: (usize, usize),
    /// bf16 staging for the **whole-sequence** GEMMs (`x·Wx` forward; `dg·Wxᵀ`,
    /// `xᵀ·dg` and `h_prevᵀ·dg` backward). Those run once per call over `[N, ·]`,
    /// exactly like a `Linear`'s, and were the last fp32 SIMT matmuls left on the
    /// profile. The recurrent `Wh` GEMM is deliberately NOT included: it runs per
    /// timestep inside the loop, where the staging would cost a cast per step, and the
    /// fused kernel owns that path anyway.
    gemm_x: ops::GemmBf16,
    gemm_dx: ops::GemmBf16,
    /// Weight cache for the **per-timestep** recurrent GEMMs `h_{t-1}·Whr` and
    /// `dg_t·Whrᵀ`. Separate from the two above only because `GemmBf16` holds one
    /// cached weight and this one is `whr`, not `wx`; both GEMMs read `whr` as their
    /// right operand, so one narrowing per optimizer step serves the whole sweep. The
    /// left operand ([`h_narrow`](Self::h_narrow) forward, [`dgh`](Self::dgh)
    gemm_h: ops::GemmBf16,
    /// Whether the whole-sequence GEMMs take the bf16 path. Pinned at construction so
    /// forward and backward cannot disagree.
    bf16: bool,
    /// Rows of the next call where a document starts: the step there runs exactly as a
    /// sequence's first. See [`set_resets`](Self::set_resets).
    resets: Resets,
}

/// What a forward saves for its backward, carved from a frame — see [`SLstm::carve`].
pub struct SlstmSaved {
    /// The gate buffer `[B, T, 4H]`: pre-activations in forward, and the gate deltas
    /// backward writes over them.
    g: GTensor<f32>,
    slabs: SlstmSlabs,
    post: GpuRmsForward,
}

/// Reuse `t`'s device buffer when the shape matches, zeroing it in place; else
/// (re)allocate a zeroed buffer. For state / BPTT channels that must start at 0.
fn fit_zeros(gpu: &Gpu, t: &mut GTensor<f32>, dims: &[usize]) {
    if t.dims() == dims {
        t.zero_(gpu);
    } else {
        *t = GTensor::zeros(gpu, dims);
    }
}

/// Reuse `t`'s device buffer when the shape matches (leaving its contents); else
/// (re)allocate uninitialised. For outputs a kernel/GEMM overwrites in full.
fn fit_uninit(gpu: &Gpu, t: &mut GTensor<f32>, dims: &[usize]) {
    if t.dims() != dims {
        *t = GTensor::uninit(gpu, dims);
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
        wz: &Tensor,
        wi: &Tensor,
        wf: &Tensor,
        wo: &Tensor,
        bz: &Tensor,
        bi: &Tensor,
        bf: &Tensor,
        bo: &Tensor,
        post_gamma: Option<&Tensor>,
    ) -> Self {
        let (h, h4) = (hidden, 4 * hidden);
        let rows = input + h;
        let w = [wz, wi, wf, wo];
        let b = [bz, bi, bf, bo];
        for (g, t) in w.iter().enumerate() {
            assert_eq!(t.dims(), [rows, h], "SLstm::from_parts — gate {g} weight");
            assert_eq!(b[g].dims(), [h], "SLstm::from_parts — gate {g} bias");
        }

        let mut wx = vec![0.0; input * h4];
        let mut whr = vec![0.0; h * h4];
        let mut bcat = vec![0.0; h4];
        for (g, gw) in w.iter().enumerate() {
            let src = gw.data.as_slice();
            for r in 0..rows {
                let (dst, dr) = if r < input {
                    (&mut wx, r)
                } else {
                    (&mut whr, r - input)
                };
                let (from, to) = (r * h, dr * h4 + g * h);
                dst[to..to + h].copy_from_slice(&src[from..from + h]);
            }
            bcat[g * h..(g + 1) * h].copy_from_slice(&b[g].data);
        }

        let up = |d: Vec<f32>, dims: &[usize]| GTensor::from_host(gpu, &Tensor::new(dims, d));
        Self {
            input,
            hidden,
            post_norm: match post_gamma {
                Some(g) => {
                    assert_eq!(g.dims(), [h], "SLstm::from_parts — post_norm gamma");
                    RmsNorm::from_parts(gpu, g)
                }
                None => RmsNorm::new(gpu, h),
            },
            wx: up(wx, &[input, h4]),
            whr: up(whr, &[h, h4]),
            bcat: up(bcat, &[h4]),
            dwx: GTensor::zeros(gpu, &[input, h4]),
            dwhr: GTensor::zeros(gpu, &[h, h4]),
            dbcat: GTensor::zeros(gpu, &[h4]),
            mwx: arena::unbacked(gpu),
            vwx: arena::unbacked(gpu),
            mwhr: arena::unbacked(gpu),
            vwhr: arena::unbacked(gpu),
            mbcat: arena::unbacked(gpu),
            vbcat: arena::unbacked(gpu),
            force_fused_time: None,
            force_batched: None,
            carry: false,
            h_state: GTensor::zeros(gpu, &[0, 0]),
            c_state: GTensor::zeros(gpu, &[0, 0]),
            n_state: GTensor::zeros(gpu, &[0, 0]),
            m_state: GTensor::zeros(gpu, &[0, 0]),
            gh: GTensor::zeros(gpu, &[0, 0]),
            dgh: ops::SlabBuf::new(gpu, &[0, 0]),
            h_narrow: ops::SlabBuf::new(gpu, &[0, 0]),
            dh_bptt: GTensor::zeros(gpu, &[0, 0]),
            dc_bptt: GTensor::zeros(gpu, &[0, 0]),
            dn_bptt: GTensor::zeros(gpu, &[0, 0]),
            out_buf: Buf::new(),
            dy_buf: Buf::new(),
            frames: ResidentFrames::default(),
            last_shape: (0, 0),
            gemm_x: ops::GemmBf16::new(),
            gemm_dx: ops::GemmBf16::new(),
            gemm_h: ops::GemmBf16::new(),
            bf16: ops::gemm_bf16_enabled(gpu),
            resets: Resets::default(),
        }
    }

    /// Freshly-initialised cell, matching `cpu::SLstm::new`'s init exactly
    /// (including the +4.5 forget-gate bias). Post-norm starts at γ=1.
    pub fn new_rand(gpu: &Gpu, input: usize, hidden: usize) -> Self {
        Self::from_cpu(gpu, &crate::cpu::SLstm::new(input, hidden), None)
    }

    /// Upload a CPU cell (weights are copied; grads/moments start at zero).
    pub fn from_cpu(gpu: &Gpu, cpu: &crate::cpu::SLstm, post_gamma: Option<&Tensor>) -> Self {
        Self::from_parts(
            gpu,
            cpu.input_size,
            cpu.hidden_size,
            &cpu.wz,
            &cpu.wi,
            &cpu.wf,
            &cpu.wo,
            &cpu.bz,
            &cpu.bi,
            &cpu.bf,
            &cpu.bo,
            post_gamma,
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

    /// Gate `g`'s weights as the `[rows, H]` matrix the CPU cell holds, downloaded
    /// and unpacked. For parity tests, which compare per gate.
    #[cfg(test)]
    pub(crate) fn gate_w(&self, gpu: &Gpu, g: usize) -> Vec<f32> {
        let m = &self.gate_matrices(gpu)[g];
        m.as_slice().to_vec()
    }

    /// Gate `g`'s bias `[H]`, sliced out of the fused `bcat`. Test companion to
    /// [`gate_w`](Self::gate_w).
    #[cfg(test)]
    fn gate_b(&self, gpu: &Gpu, g: usize) -> Vec<f32> {
        let h = self.hidden;
        self.bcat.to_host(gpu).data[g * h..(g + 1) * h].to_vec()
    }

    /// Gate `g`'s weight gradient, unpacked from the fused `dwx`/`dwhr` into the
    /// `[rows, H]` layout the CPU cell's `dw*` use. Test-only.
    #[cfg(test)]
    fn gate_dw(&self, gpu: &Gpu, g: usize) -> Vec<f32> {
        let (h, h4, inp) = (self.hidden, 4 * self.hidden, self.input);
        let rows = inp + h;
        let dwx = self.dwx.to_host(gpu).data;
        let dwhr = self.dwhr.to_host(gpu).data;
        let mut out = vec![0.0; rows * h];
        for r in 0..rows {
            let (src, sr) = if r < inp { (&dwx, r) } else { (&dwhr, r - inp) };
            let from = sr * h4 + g * h;
            out[r * h..(r + 1) * h].copy_from_slice(&src[from..from + h]);
        }
        out
    }

    /// Gate `g`'s bias gradient `[H]`. Test companion to [`gate_dw`](Self::gate_dw).
    #[cfg(test)]
    fn gate_db(&self, gpu: &Gpu, g: usize) -> Vec<f32> {
        let h = self.hidden;
        self.dbcat.to_host(gpu).data[g * h..(g + 1) * h].to_vec()
    }

    /// Unpack the fused `wx`/`whr` back into the four `[rows, H]` gate matrices the
    /// CPU cell uses — the inverse of the pack in [`from_parts`](Self::from_parts).
    #[cfg(test)]
    fn gate_matrices(&self, gpu: &Gpu) -> [iron_oxide::collections::Matrix; 4] {
        let (h, h4, inp) = (self.hidden, 4 * self.hidden, self.input);
        let rows = inp + h;
        let wx = self.wx.to_host(gpu).data;
        let whr = self.whr.to_host(gpu).data;
        std::array::from_fn(|g| {
            let mut m = vec![0.0; rows * h];
            for r in 0..rows {
                let (src, sr) = if r < inp { (&wx, r) } else { (&whr, r - inp) };
                let from = sr * h4 + g * h;
                m[r * h..(r + 1) * h].copy_from_slice(&src[from..from + h]);
            }
            iron_oxide::collections::Matrix::from_vec(m, rows, h)
        })
    }

    /// The post-cell norm's scale, for exporting the enclosing block to a CPU
    /// `SLSTMBlock` (which still keeps the norm at block level).
    pub fn post_norm_gamma(&self) -> &GTensor<f32> {
        &self.post_norm.gamma
    }

    /// This cell's saved tensors for a `[B, T, ·]` call, carved from `f` in the one order
    /// its forward and backward share. fp32 for the stabilizer-carrying slabs,
    /// kernel-matched width for the plain activations — see `SlstmSlabs` and `gpu::bf16`.
    pub fn carve(&self, gpu: &Gpu, f: &mut Frame, b: usize, t: usize) -> SlstmSaved {
        let (h, n) = (self.hidden, b * t);
        let (st, sl) = (gpu.kernels.state_bf16, gpu.kernels.slab_bf16);
        SlstmSaved {
            g: f.f32(gpu, &[b, t, 4 * h]),
            slabs: SlstmSlabs {
                // `c_entry`/`n_entry` hold one timestep, not the sweep.
                c_entry: f.slab(gpu, &[b, h], st),
                n_entry: f.slab(gpu, &[b, h], st),
                i_prime: f.slab(gpu, &[b, t, h], st),
                f_prime: f.slab(gpu, &[b, t, h], st),
                c: f.slab(gpu, &[b, t, h], st),
                n: f.slab(gpu, &[b, t, h], st),
                zt: f.slab(gpu, &[b, t, h], sl),
                ot: f.slab(gpu, &[b, t, h], sl),
                h_prev: f.slab(gpu, &[b, t, h], sl),
            },
            post: self.post_norm.carve(gpu, f, n),
        }
    }

    /// Forward over a whole `[B, T, in]` sequence into `y` `[B, T, H]`, saving into a
    /// frame of this cell's own. See [`forward_saved`](Self::forward_saved).
    pub fn forward(
        &mut self,
        gpu: &Gpu,
        x: &GTensor<f32>,
        y: &mut GTensor<f32>,
        cache: &TrainingCache,
    ) {
        let (b, t) = (x.shape[0], x.shape[1]);
        let mut m = Frame::measure();
        self.carve(gpu, &mut m, b, t);
        self.carve_x(gpu, &mut m, b, t);
        if !self.carry {
            self.frames.reset();
        }
        let mut f = self.frames.push(gpu, m.bytes());
        let mut sv = self.carve(gpu, &mut f, b, t);
        let mut xs = self.carve_x(gpu, &mut f, b, t);
        xs.store(gpu, x);
        self.forward_saved(gpu, &xs, y, &mut sv, cache);
    }

    /// The input a standalone [`forward`](Self::forward) keeps for its backward, carved
    /// after the cell's own tensors, at the width the GEMMs read it. In a
    /// [`Block`](super::block::Block) the block's `norm1_out` plays this part.
    fn carve_x(&self, gpu: &Gpu, f: &mut Frame, b: usize, t: usize) -> SlabBuf {
        f.slab(gpu, &[b, t, self.input], self.bf16)
    }

    /// Forward over a whole `[B, T, in]` sequence into `y` `[B, T, H]`, saving into `sv`.
    pub fn forward_saved(
        &mut self,
        gpu: &Gpu,
        x: &SlabBuf,
        y: &mut GTensor<f32>,
        sv: &mut SlstmSaved,
        cache: &TrainingCache,
    ) {
        let xd = x.dims();
        assert_eq!(xd.len(), 3, "SLstm::forward expects [B, T, in]");
        let (b, t, inp) = (xd[0], xd[1], xd[2]);
        assert_eq!(inp, self.input, "SLstm::forward — input width mismatch");
        assert_eq!(
            y.dims(),
            [b, t, self.hidden],
            "SLstm::forward — output shape"
        );
        let h = self.hidden;
        let h4 = 4 * h;
        let n = b * t;
        self.last_shape = (b, t);


        let carry = self.carry && self.h_state.dims() == [b, h];
        for s in [
            &mut self.h_state,
            &mut self.c_state,
            &mut self.n_state,
            &mut self.m_state,
        ] {
            fit_uninit(gpu, s, &[b, h]);
        }

        let SlstmSaved { g, slabs, post } = sv;

        g.reshape_to(&[n, h4]);
        let (gemm_x, wx_w) = (&mut self.gemm_x, &self.wx);
        let xv = x.view(gpu, &[n, inp]);
        let xs = xv.at_width(gpu, self.bf16, &cache.temps);
        phase::timed(gpu, phase::Bucket::SlstmGemmFwd, || match &xs.x {
            SlabBuf::Bf16(xb) => gemm_x.run_staged_lhs(gpu, ops::MmForm::Nn, xb, wx_w, g, 0.0),
            SlabBuf::F32(xf) => ops::matmul_nn_into(gpu, xf, wx_w, g, 0.0),
        });
        drop(xs);
        g.reshape_to(&[b, t, h4]);

        let mut out_buf = std::mem::take(&mut self.out_buf);
        let out = out_buf.get(gpu, &[b, t, h]);
        phase::timed(gpu, phase::Bucket::SlstmLoopFwd, || {
            self.fwd_loop(gpu, g, slabs, out, t, carry, cache);
        });

        phase::timed(gpu, phase::Bucket::SlstmCopyFwd, || {
            self.post_norm.forward_saved(gpu, out, y, post);
        });
        self.out_buf = out_buf;
    }

    /// The forward time loop: one cooperative launch when T is long enough, else
    /// step by step.
    fn fwd_loop(
        &mut self,
        gpu: &Gpu,
        g: &mut GTensor<f32>,
        slabs: &mut SlstmSlabs,
        out: &mut GTensor<f32>,
        t: usize,
        carry: bool,
        cache: &TrainingCache,
    ) {
        let b = self.h_state.rows();
        if self.fuses_at(b, t)
            && ops::slstm_fused_time(
                gpu,
                &self.whr,
                g,
                &self.bcat,
                &mut self.c_state,
                &mut self.n_state,
                &mut self.m_state,
                &mut self.h_state,
                slabs,
                out,
                t,
                carry,
                self.resets.mask,
                &cache.temps,
            )
        {
            return;
        }
        if self.resets.is_empty()
            && self.batches_at(gpu, b)
            && ops::slstm_batched_fwd(
                gpu,
                &self.whr,
                g,
                &self.bcat,
                &mut self.c_state,
                &mut self.n_state,
                &mut self.m_state,
                &mut self.h_state,
                slabs,
                out,
                t,
                carry,
                &cache.temps,
            )
        {
            return;
        }
        self.fwd_steps(gpu, g, slabs, out, t, carry);
    }

    /// The loop body, one timestep at a time.
    fn fwd_steps(
        &mut self,
        gpu: &Gpu,
        g: &mut GTensor<f32>,
        slabs: &mut SlstmSlabs,
        out: &mut GTensor<f32>,
        t: usize,
        carry: bool,
    ) {
        let (b, h) = (self.h_state.rows(), self.h_state.cols());
        fit_uninit(gpu, &mut self.gh, &[b, 4 * h]);
        self.h_narrow.fit(gpu, &[b, h]);
        if carry {
            let (h_narrow, h_state) = (&mut self.h_narrow, &self.h_state);
            h_narrow.store(gpu, h_state);
        }
        let mut resets = self.resets.rows.iter().copied().peekable();
        for step in 0..t {
            let reset = resets.next_if_eq(&step).is_some();
            let first = (step == 0 && !carry) || reset;
            if !first {
                let Self {
                    gemm_h,
                    h_narrow,
                    whr,
                    gh,
                    ..
                } = self;
                match h_narrow {
                    ops::SlabBuf::Bf16(h) => {
                        gemm_h.run_staged_lhs(gpu, ops::MmForm::Nn, h, whr, gh, 0.0)
                    }
                    ops::SlabBuf::F32(h) => ops::matmul_nn_into(gpu, h, whr, gh, 0.0),
                }
            }
            ops::slstm_step_fused(
                gpu,
                g,
                &self.gh,
                &self.bcat,
                &mut self.c_state,
                &mut self.n_state,
                &mut self.m_state,
                &mut self.h_state,
                &mut self.h_narrow,
                slabs,
                out,
                step,
                first,
            );
        }
    }

    /// Forward into a freshly allocated `[B, T, H]` — the by-value companion to
    /// [`forward`](Self::forward), used by tests and one-shot call sites.
    pub fn forward_alloc(
        &mut self,
        gpu: &Gpu,
        x: &GTensor<f32>,
        cache: &TrainingCache,
    ) -> GTensor<f32> {
        let mut y: GTensor<f32> = GTensor::uninit(gpu, &[x.shape[0], x.shape[1], self.hidden]);
        self.forward(gpu, x, &mut y, cache);
        y
    }

    /// Backward into a freshly allocated `dx` `[B, T, in]`. `y` is the forward output,
    /// as for [`backward`](Self::backward).
    pub fn backward_alloc(
        &mut self,
        gpu: &Gpu,
        y: &GTensor<f32>,
        dy: &GTensor<f32>,
        cache: &TrainingCache,
    ) -> GTensor<f32> {
        let mut dx = GTensor::uninit(gpu, &[dy.shape[0], dy.shape[1], self.input]);
        self.backward(gpu, y, dy, &mut dx, cache);
        dx
    }

    /// Backward over the whole sequence, reading the frame the matching
    /// [`forward`](Self::forward) pushed. See [`backward_saved`](Self::backward_saved).
    pub fn backward(
        &mut self,
        gpu: &Gpu,
        y: &GTensor<f32>,
        dy: &GTensor<f32>,
        dx: &mut GTensor<f32>,
        cache: &TrainingCache,
    ) {
        let mut f = self.frames.pop(gpu);
        let (b, t) = (dy.shape[0], dy.shape[1]);
        let mut sv = self.carve(gpu, &mut f, b, t);
        let xs = self.carve_x(gpu, &mut f, b, t);
        self.backward_saved(gpu, &xs, y, dy, dx, &mut sv, cache);
    }

    /// Backward over the whole sequence. `dy` is `[B, T, H]`, `dx` is the caller's
    /// `[B, T, in]` output. Accumulates weight/bias grads.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_saved(
        &mut self,
        gpu: &Gpu,
        x: &SlabBuf,
        y: &GTensor<f32>,
        dy: &GTensor<f32>,
        dx: &mut GTensor<f32>,
        sv: &mut SlstmSaved,
        cache: &TrainingCache,
    ) {
        assert_eq!(dy.rank, 3, "SLstm::backward expects [B, T, H]");
        let (b, t, h) = (dy.shape[0], dy.shape[1], dy.shape[2]);
        assert_eq!(h, self.hidden, "SLstm::backward — hidden mismatch");
        assert_eq!(dx.dims(), [b, t, self.input], "SLstm::backward — dx shape");
        assert_eq!(x.dims(), [b, t, self.input], "SLstm::backward — x shape");
        let inp = self.input;
        let h4 = 4 * h;
        let n = b * t;
        let SlstmSaved { g, slabs, post } = sv;

        let carry = self.carry && self.dh_bptt.dims() == [b, h];
        for buf in [&mut self.dh_bptt, &mut self.dc_bptt, &mut self.dn_bptt] {
            if !carry {
                fit_zeros(gpu, buf, &[b, h]);
            }
        }

        let mut dy_buf = std::mem::take(&mut self.dy_buf);
        let d_out = dy_buf.get(gpu, &[b, t, h]);
        phase::timed(gpu, phase::Bucket::SlstmCopyBwd, || {
            self.post_norm
                .backward_saved(gpu, dy, y, post, None, d_out, cache);
        });

        phase::timed(gpu, phase::Bucket::SlstmLoopBwd, || {
            self.bwd_loop(gpu, d_out, g, slabs, t, cache);
        });
        self.dy_buf = dy_buf;

        g.reshape_to(&[n, h4]);
        let dg = &*g;
        dx.reshape_to(&[n, inp]);
        slabs.h_prev.shrink_to(&[n, h]);
        let wx = &self.wx;
        let dwx = &mut self.dwx;
        let dwhr = &mut self.dwhr;
        let gemm_dx = &mut self.gemm_dx;
        let xv = x.view(gpu, &[n, inp]);
        let xs = xv.at_width(gpu, self.bf16, &cache.temps);
        phase::timed(gpu, phase::Bucket::SlstmGemmBwd, || {
            match (&xs.x, &slabs.h_prev) {
                (SlabBuf::Bf16(xb), SlabBuf::Bf16(hb)) => {
                    gemm_dx.run_slstm_backward(gpu, xb, hb, dg, wx, dwx, dwhr, dx)
                }
                (x, hp) => {
                    match x {
                        SlabBuf::Bf16(xb) => {
                            gemm_dx.run_backward_staged_x(gpu, xb, dg, wx, dwx, dx, 0.0)
                        }
                        SlabBuf::F32(xf) => {
                            ops::matmul_nt_into(gpu, dg, wx, dx, 0.0);
                            ops::matmul_tn_into(gpu, xf, dg, dwx, 1.0);
                        }
                    }
                    let mut scratch = cache.temps.get::<f32>(gpu, &[n, h]);
                    let hf = hp.as_f32(gpu, &mut scratch);
                    ops::matmul_tn_into(gpu, hf, dg, dwhr, 1.0);
                }
            }
        });

        let dbcat = &mut self.dbcat;
        phase::timed(gpu, phase::Bucket::SlstmGemmBwd, || {
            ops::add_col_sum(gpu, dbcat, dg, &cache.temps);
        });

        if grad_bf16(gpu) {
            let Self {
                dwx, dwhr, dbcat, ..
            } = self;
            for t in [dwx, dwhr, dbcat] {
                ops::quantize_bf16_(gpu, t);
            }
        }
        dx.reshape_to(&[b, t, inp]);
    }

    /// The backward time loop — time-fused on the same terms as [`Self::fwd_loop`].
    fn bwd_loop(
        &mut self,
        gpu: &Gpu,
        dy: &GTensor<f32>,
        g: &mut GTensor<f32>,
        slabs: &SlstmSlabs,
        t: usize,
        cache: &TrainingCache,
    ) {
        let b = dy.shape[0];
        if self.fuses_at(b, t)
            && ops::slstm_fused_time_bwd(
                gpu,
                &self.whr,
                dy,
                g,
                &mut self.dh_bptt,
                &mut self.dc_bptt,
                &mut self.dn_bptt,
                slabs,
                t,
                self.resets.mask,
            )
        {
            return;
        }
        if self.resets.is_empty()
            && self.batches_at(gpu, b)
            && ops::slstm_batched_bwd(
                gpu,
                &self.whr,
                dy,
                g,
                &mut self.dh_bptt,
                &mut self.dc_bptt,
                &mut self.dn_bptt,
                slabs,
                t,
                &cache.temps,
            )
        {
            return;
        }
        self.bwd_steps(gpu, dy, g, slabs, t);
    }

    fn bwd_steps(
        &mut self,
        gpu: &Gpu,
        dy: &GTensor<f32>,
        g: &mut GTensor<f32>,
        slabs: &SlstmSlabs,
        t: usize,
    ) {
        let (b, h) = (self.dc_bptt.rows(), self.dc_bptt.cols());
        self.dgh.fit(gpu, &[b, 4 * h]);
        let mut resets = self.resets.rows.iter().rev().copied().peekable();
        for step in (0..t).rev() {
            let reset = resets.next_if_eq(&step).is_some();
            ops::slstm_step_fused_bwd(
                gpu,
                dy,
                g,
                &mut self.dgh,
                &self.dh_bptt,
                slabs,
                &mut self.dc_bptt,
                &mut self.dn_bptt,
                step,
                reset,
            );
            // A reset step read no `h_{t-1}`, so nothing reaches it through `Wh`.
            if reset {
                self.dh_bptt.zero_(gpu);
                continue;
            }
            let Self {
                gemm_h,
                dgh,
                whr,
                dh_bptt,
                ..
            } = self;
            match dgh {
                ops::SlabBuf::Bf16(d) => {
                    gemm_h.run_staged_lhs(gpu, ops::MmForm::Nt, d, whr, dh_bptt, 0.0)
                }
                ops::SlabBuf::F32(d) => ops::matmul_nt_into(gpu, d, whr, dh_bptt, 0.0),
            }
        }
    }

    /// Every parameter with its gradient and AdamW moments. The gate matrices decay,
    /// the fused bias does not; the post-cell norm's γ comes last.
    pub fn param_slots(&mut self) -> Vec<ParamSlot<'_>> {
        self.gemm_x.invalidate_w();
        self.gemm_dx.invalidate_w();
        self.gemm_h.invalidate_w();
        let mut v = vec![
            ParamSlot::new(
                &mut self.wx,
                &mut self.dwx,
                &mut self.mwx,
                &mut self.vwx,
                ParamKind::Decay,
            ),
            ParamSlot::new(
                &mut self.whr,
                &mut self.dwhr,
                &mut self.mwhr,
                &mut self.vwhr,
                ParamKind::Decay,
            ),
            ParamSlot::new(
                &mut self.bcat,
                &mut self.dbcat,
                &mut self.mbcat,
                &mut self.vbcat,
                ParamKind::NoDecay,
            ),
        ];
        v.extend(self.post_norm.param_slots());
        v
    }

    /// Every learnable tensor, in a fixed order (used by checkpoint save/load).
    pub fn params_mut(&mut self) -> Vec<&mut GTensor<f32>> {
        self.param_slots().into_iter().map(|s| s.param).collect()
    }

    /// Gradient accumulators, in the same order as `params_mut`. Diagnostic.
    pub fn grads(&mut self) -> Vec<&GTensor<f32>> {
        self.param_slots().into_iter().map(|s| &*s.grad).collect()
    }

    /// Forward-cache extremes of the last standalone sweep: `(min |n|, max |c|,
    /// max |c/n|)`. See [`extremes_of`](Self::extremes_of).
    pub fn state_extremes(&self, gpu: &Gpu) -> Option<(f32, f32, f32)> {
        let mut f = self.frames.recent(gpu)?;
        let (b, t) = self.last_shape;
        Some(Self::extremes_of(gpu, &self.carve(gpu, &mut f, b, t)))
    }

    /// `(min |n|, max |c|, max |c/n|)` over a saved sweep.
    pub fn extremes_of(gpu: &Gpu, sv: &SlstmSaved) -> (f32, f32, f32) {
        let slabs = &sv.slabs;
        let mut scratch = GTensor::uninit(gpu, slabs.n.dims());
        let n = slabs.n.as_f32(gpu, &mut scratch).to_host(gpu).data.to_vec();
        let mut scratch = GTensor::uninit(gpu, slabs.c.dims());
        let c = slabs.c.as_f32(gpu, &mut scratch).to_host(gpu).data.to_vec();
        let min_n = n.iter().map(|v| v.abs()).fold(f32::INFINITY, f32::min);
        let max_c = c.iter().map(|v| v.abs()).fold(0.0, f32::max);
        let max_ratio = c
            .iter()
            .zip(&n)
            .map(|(a, b)| (a / b).abs())
            .fold(0.0, f32::max);
        (min_n, max_c, max_ratio)
    }

    /// Continue the previous call's recurrence rather than starting from zero.
    pub fn fuses_at(&self, b: usize, t: usize) -> bool {
        t >= FUSED_MIN_T
            && self
                .force_fused_time
                .unwrap_or_else(|| fused_time_enabled() && b <= FUSED_MAX_B)
    }

    /// Whether a `[B, T, H]` sweep takes the **batch-parallel** fused path.
    pub fn batches_at(&self, gpu: &Gpu, b: usize) -> bool {
        self.force_batched.unwrap_or_else(|| {
            batched_enabled() && b >= SB_MIN_B && ops::slstm_batched_pays(gpu, self.hidden, b)
        })
    }

    pub fn set_carry(&mut self, carry: bool) {
        self.carry = carry;
    }

    /// Zero the carried **forward** state, so the next `forward` starts the recurrence
    /// from scratch whatever `carry` says.
    pub fn reset_state(&mut self, gpu: &Gpu) {
        self.zero_state(gpu);
        self.reset_caches(gpu);
    }

    /// The carried `h`, `c`, `n`, `m`, moved out; the cell is left with none,
    /// as before its first forward.
    pub fn take_carried(&mut self, gpu: &Gpu) -> [GTensor<f32>; 4] {
        [
            &mut self.h_state,
            &mut self.c_state,
            &mut self.n_state,
            &mut self.m_state,
        ]
        .map(|s| std::mem::replace(s, GTensor::zeros(gpu, &[0, 0])))
    }

    pub fn put_carried(&mut self, [h, c, n, m]: [GTensor<f32>; 4]) {
        (self.h_state, self.c_state, self.n_state, self.m_state) = (h, c, n, m);
    }

    /// Start the recurrence over at the next forward, keeping everything already
    /// cached.
    pub fn zero_state(&mut self, gpu: &Gpu) {
        for s in [
            &mut self.h_state,
            &mut self.c_state,
            &mut self.n_state,
            &mut self.m_state,
        ] {
            if !s.is_empty() {
                s.zero_(gpu);
            }
        }
    }

    /// Drop the caches of an already-unwound sweep, keeping `h`/`c`/`n`/`m` — so the
    /// next forward continues this recurrence. See [`Cell::reset_caches`].
    pub fn reset_caches(&mut self, _gpu: &Gpu) {
        self.frames.reset();
    }

    /// Zero the carried **BPTT** channels, so the next `backward` starts with no
    /// incoming gradient from the right. Call before the rightmost chunk's backward.
    pub fn reset_bptt(&mut self, gpu: &Gpu) {
        for s in [&mut self.dh_bptt, &mut self.dc_bptt, &mut self.dn_bptt] {
            if !s.is_empty() {
                s.zero_(gpu);
            }
        }
    }

    /// Where documents start inside the next call. See [`Cell::set_resets`](super::block::Cell::set_resets).
    pub fn set_resets(&mut self, r: &Resets) {
        self.resets.clone_from(r);
    }

    /// Retained activation bytes split `(saved_cache, other)`.
    pub fn act_split(&self) -> (usize, usize) {
        let saved = self.frames.device_bytes();
        let (_, all) = self.retained_bytes();
        (saved, all - saved)
    }

    /// Release this cell's frames and its post-norm's slot.
    pub fn drop_all_act(&mut self) {
        self.frames.release();
        self.post_norm.drop_saved_act();
    }

    /// Device bytes held, split `(params, activations)`. Diagnostic — see
    /// [`Hierarchical::retained_report`](super::hierarchical::Hierarchical::retained_report).
    pub fn retained_bytes(&self) -> (usize, usize) {
        let params: usize = [
            &self.wx,
            &self.whr,
            &self.bcat,
            &self.dwx,
            &self.dwhr,
            &self.dbcat,
            &self.mwx,
            &self.vwx,
            &self.mwhr,
            &self.vwhr,
            &self.mbcat,
            &self.vbcat,
        ]
        .iter()
        .map(|t| t.capacity() * 4)
        .sum();
        let opt = self.out_buf.retained_bytes()
            + self.dy_buf.retained_bytes()
            + self.frames.device_bytes();
        let live: usize = [
            &self.h_state,
            &self.c_state,
            &self.n_state,
            &self.m_state,
            &self.gh,
            &self.dh_bptt,
            &self.dc_bptt,
            &self.dn_bptt,
        ]
        .iter()
        .map(|t| t.capacity() * 4)
        .sum::<usize>()
            + self.h_narrow.retained_bytes()
            + self.dgh.retained_bytes();
        let staging = self.gemm_x.retained_bytes()
            + self.gemm_dx.retained_bytes()
            + self.gemm_h.retained_bytes();
        let (pn_p, _) = self.post_norm.retained_bytes();
        (params + pn_p, opt + live + staging)
    }

    pub fn zero_grad(&mut self, gpu: &Gpu) {
        for g in [&mut self.dwx, &mut self.dwhr, &mut self.dbcat] {
            g.zero_(gpu);
        }
        self.post_norm.zero_grad(gpu);
    }

    /// AdamW over this cell's own parameters, then clear the grads. A model steps
    /// its whole `ParamArena` in one launch instead.
    pub fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        arena::step_slots(gpu, &mut self.param_slots(), cfg);
    }
}
#[cfg(test)]
mod tests {

    /// One temp cache per test, sized past every shape this module presents.
    fn test_cache(gpu: &Gpu) -> TrainingCache {
        TrainingCache::new(gpu, 1 << 20, 1 << 16, 1 << 20)
    }
    use super::*;
    use crate::cpu::optim::AdamCfg;
    use crate::cpu::slstm::SLstm as CpuSLstm;

    /// GPU-vs-CPU tolerance, scaled for the slab storage dtype in use.
    fn tol(gpu: &Gpu, base: f32) -> f32 {
        if gpu.kernels.slab_bf16 {
            base * 128.0
        } else {
            base
        }
    }

    /// Tolerance for a parameter compared AFTER an Adam step. Wider than [`tol`]:
    /// `lr·ĝ/(√v̂+ε)` is scale-invariant in the gradient, so a near-zero-gradient
    /// weight turns a ~1e-7 difference into ~1e-4.
    fn step_tol(gpu: &Gpu, base: f32) -> f32 {
        if gpu.kernels.slab_bf16 {
            base * 16.0
        } else {
            base
        }
    }

    fn assert_close(got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len(), "length mismatch");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < tol, "index {i}: gpu {g} vs cpu {w}");
        }
    }

    fn from_cpu(gpu: &Gpu, cpu: &CpuSLstm) -> SLstm {
        SLstm::from_parts(
            gpu,
            cpu.input_size,
            cpu.hidden_size,
            &cpu.wz,
            &cpu.wi,
            &cpu.wf,
            &cpu.wo,
            &cpu.bz,
            &cpu.bi,
            &cpu.bf,
            &cpu.bo,
            None,
        )
    }

    /// The CPU reference for a GPU cell: `cpu::SLstm` is the bare recurrence, while
    /// the GPU cell now also owns the post-cell norm, so the comparison composes the
    /// two by hand. γ starts at 1 on both sides (`from_cpu` above passes `None`).
    struct CpuRef {
        cell: CpuSLstm,
        norm: crate::cpu::rms_norm::RmsNorm,
    }

    impl CpuRef {
        fn new(inp: usize, h: usize) -> Self {
            Self {
                cell: CpuSLstm::new(inp, h),
                norm: crate::cpu::rms_norm::RmsNorm::new(h),
            }
        }
        /// `cpu::RmsNorm` is strictly rank 2 (its GPU twin folds leading axes
        /// itself), so the `[B, T, H]` sequences are flattened around it here.
        fn flat(t: &Tensor) -> Tensor {
            let f = t.shape[t.rank - 1];
            t.reshape(&[t.len() / f, f])
        }
        fn forward(&mut self, x: &Tensor) -> Tensor {
            let y = self.cell.forward(x);
            let dims = y.dims().to_vec();
            self.norm.forward(&Self::flat(&y)).reshape(&dims)
        }
        fn backward(&mut self, dy: &Tensor) -> Tensor {
            let dims = dy.dims().to_vec();
            let d = self.norm.backward(&Self::flat(dy)).reshape(&dims);
            self.cell.backward(&d)
        }
        fn step(&mut self, cfg: &AdamCfg) {
            self.cell.step(cfg);
            self.norm.step(cfg);
        }
    }

    /// The whole-sequence GEMMs cache a bf16 copy of `Wx`, so every write to `Wx` must
    /// invalidate it. A stale cache does not error and does not go non-finite — the
    /// forward simply keeps reading the weight the optimizer has already moved past, so
    /// the observable is that the output stops tracking `Wx`.
    #[test]
    fn wx_cache_follows_optimizer_steps() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        let (b, t, inp, h) = (2, 4, 5, 6);
        let mut cell = SLstm::new_rand(&gpu, inp, h);
        let x = GTensor::from_host(&gpu, &Tensor::random(&[b, t, inp], 0.5));

        for round in 1..=3 {
            let before = cell.forward_alloc(&gpu, &x, &tc).to_host(&gpu).data;

            let mut hw = cell.wx.to_host(&gpu);
            for (i, v) in hw.data.iter_mut().enumerate() {
                *v += 0.25 * ((i % 7) as f32 - 3.0) * round as f32;
            }
            cell.wx = GTensor::from_host(&gpu, &hw);
            // The write above is exactly what `params_mut` exists to guard.
            cell.params_mut();

            let after = cell.forward_alloc(&gpu, &x, &tc).to_host(&gpu).data;
            let changed = before
                .iter()
                .zip(&after)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                changed > 1e-4,
                "round {round}: Wx was overwritten but the output did not change — the \
                 bf16 weight cache was not invalidated"
            );
        }
    }

    /// GPU sLSTM must match `cpu::SLstm` (cell alone) for a full
    /// forward → backward → AdamW-step cycle, from identical weights. Tolerance
    /// is loose-ish because the two paths differ in float reduction order (cuBLAS
    /// vs the CPU gemm), but the recurrence math is identical.
    #[test]
    fn slstm_matches_cpu_layer() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        let (b, t, inp, h) = (2, 5, 4, 6);

        let mut cpu = CpuRef::new(inp, h);
        let mut dev = from_cpu(&gpu, &cpu.cell);

        let x = Tensor::random(&[b, t, inp], 0.5);
        let g = Tensor::random(&[b, t, h], 1.0);

        // Forward
        let y_cpu = cpu.forward(&x);
        let y_dev = dev.forward_alloc(&gpu, &GTensor::from_host(&gpu, &x), &tc);
        assert_close(&y_dev.to_host(&gpu).data, &y_cpu.data, tol(&gpu, 2e-3));

        // Backward
        let dx_cpu = cpu.backward(&g);
        let dx_dev = dev.backward_alloc(&gpu, &y_dev, &GTensor::from_host(&gpu, &g), &tc);
        assert_close(&dx_dev.to_host(&gpu).data, &dx_cpu.data, tol(&gpu, 2e-3));
        assert_close(&dev.gate_dw(&gpu, 0), &cpu.cell.dwz.data, tol(&gpu, 2e-3));
        assert_close(&dev.gate_dw(&gpu, 2), &cpu.cell.dwf.data, tol(&gpu, 2e-3));
        assert_close(&dev.gate_db(&gpu, 2), &cpu.cell.dbf.data, tol(&gpu, 2e-3));

        // One AdamW step, then compare the updated gate weights.
        let mut cfg = AdamCfg::new(1e-3, 0.01);
        cfg.t = 1;
        cpu.step(&cfg);
        dev.step(&gpu, &cfg);
        assert_close(
            &dev.gate_w(&gpu, 0),
            &cpu.cell.wz.data,
            step_tol(&gpu, 2e-3),
        );
        assert_close(
            &dev.gate_w(&gpu, 2),
            &cpu.cell.wf.data,
            step_tol(&gpu, 2e-3),
        );
        assert_close(
            &dev.gate_b(&gpu, 2),
            &cpu.cell.bf.data,
            step_tol(&gpu, 2e-3),
        );
    }

    /// The same parity check at `T > FUSED_MIN_T`, so both time loops run as **one
    /// cooperative launch** — the path `slstm_matches_cpu_layer` (T=5) never reaches.
    #[test]
    fn slstm_long_t_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        assert!(
            64 > FUSED_MIN_T,
            "this test must exercise the time-fused path"
        );
        let (b, t, inp, h) = (2, 64, 8, 12);

        let mut cpu = CpuRef::new(inp, h);
        let mut dev = from_cpu(&gpu, &cpu.cell);
        let mut cfg = AdamCfg::new(1e-3, 0.01);

        for pass in 0..2 {
            let x = Tensor::random(&[b, t, inp], 0.5);
            let g = Tensor::random(&[b, t, h], 1.0);

            let y_cpu = cpu.forward(&x);
            let y_dev = dev.forward_alloc(&gpu, &GTensor::from_host(&gpu, &x), &tc);
            assert_close(&y_dev.to_host(&gpu).data, &y_cpu.data, tol(&gpu, 2e-3));

            let dx_cpu = cpu.backward(&g);
            let dx_dev = dev.backward_alloc(&gpu, &y_dev, &GTensor::from_host(&gpu, &g), &tc);
            assert_close(&dx_dev.to_host(&gpu).data, &dx_cpu.data, tol(&gpu, 2e-3));
            assert_close(&dev.gate_dw(&gpu, 0), &cpu.cell.dwz.data, tol(&gpu, 2e-3));
            assert_close(&dev.gate_dw(&gpu, 2), &cpu.cell.dwf.data, tol(&gpu, 2e-3));

            cfg.t = pass + 1;
            cpu.step(&cfg);
            dev.step(&gpu, &cfg);
            assert_close(
                &dev.gate_w(&gpu, 0),
                &cpu.cell.wz.data,
                step_tol(&gpu, 2e-3),
            );
        }
    }

    /// The cell's per-call buffers are *reused* whenever the shape matches and
    /// reallocated when it does not, so a run of differently shaped windows walks
    /// them through allocate/reuse/free repeatedly. This is the real training
    /// pattern — windows never cross a document border, so every short document
    /// yields a short window — and anything that survived a call it should not
    /// (a stale slab, a scratch sized for the previous shape) shows up here.
    #[test]
    fn slstm_survives_shape_changes() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        let (b, inp, h) = (1, 64, 64);
        let mut cpu = CpuRef::new(inp, h);
        let mut dev = from_cpu(&gpu, &cpu.cell);

        let poison = |floats: usize| {
            let mut d = GTensor::uninit(&gpu, &[floats]);
            ops::fill(&gpu, &mut d, 1e30);
        };

        for &t in &[256usize, 8, 256, 192, 5, 256] {
            let x = Tensor::random(&[b, t, inp], 0.5);
            let g = Tensor::random(&[b, t, h], 1.0);

            let y_cpu = cpu.forward(&x);
            let y_dev = dev.forward_alloc(&gpu, &GTensor::from_host(&gpu, &x), &tc);
            assert_close(&y_dev.to_host(&gpu).data, &y_cpu.data, tol(&gpu, 2e-3));
            let dx_cpu = cpu.backward(&g);
            let dx_dev = dev.backward_alloc(&gpu, &y_dev, &GTensor::from_host(&gpu, &g), &tc);
            assert_close(&dx_dev.to_host(&gpu).data, &dx_cpu.data, tol(&gpu, 2e-3));

            poison(256 * 4 * h * b);
        }
    }

    /// A shape long enough to *ask* for the time-fused path but too wide for it must
    /// still run, and run correctly.
    #[test]
    fn wide_batch_falls_back_to_the_per_step_loop() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        let (b, t, h) = (1088usize, 32usize, 32usize);
        let s = 1.0 / (h as f32).sqrt();
        let w: Vec<Tensor> = (0..4)
            .map(|g| Tensor::random(&[2 * h, h], s * (1.0 + g as f32 * 0.05)))
            .collect();
        let bi: Vec<Tensor> = (0..4).map(|_g| Tensor::random(&[h], 0.2)).collect();
        let x = GTensor::from_host(&gpu, &Tensor::random(&[b, t, h], 0.5));

        let build = || {
            SLstm::from_parts(
                &gpu, h, h, &w[0], &w[1], &w[2], &w[3], &bi[0], &bi[1], &bi[2], &bi[3], None,
            )
        };
        // Default selection: asks for the fused path, gets told no, falls back.
        let got = build().forward_alloc(&gpu, &x, &tc).to_host(&gpu);
        // The path it lands on, pinned directly.
        let mut forced = build();
        forced.force_fused_time = Some(false);
        let want = forced.forward_alloc(&gpu, &x, &tc).to_host(&gpu);
        assert_eq!(
            got.data, want.data,
            "fallback must be the per-step loop exactly"
        );
        assert!(got.data.iter().all(|v| v.is_finite()), "non-finite output");
    }

    /// The time-fused forward (one cooperative launch for the whole T-loop) must
    /// agree with the per-step path it replaces.
    #[test]
    fn slstm_fused_time_matches_per_step() {
        for b in [1usize, 64, 256] {
            fused_time_matches_per_step_at(b, 64);
        }
        fused_time_matches_per_step_at(1, 1024);
    }

    fn fused_time_matches_per_step_at(b: usize, h: usize) {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        if !gpu.kernels.has_coop {
            return; // cooperative kernels unavailable (no CUDA headers)
        }
        let t = 64usize;
        if ops::slstm_fused_time_geometry(&gpu, h, b).is_none()
            || ops::slstm_fused_time_bwd_geometry(&gpu, h, b).is_none()
        {
            return; // shape does not fit the fused path on this device
        }
        let s = 1.0 / (h as f32).sqrt();
        let w: Vec<Tensor> = (0..4)
            .map(|g| Tensor::random(&[2 * h, h], s * (1.0 + g as f32 * 0.05)))
            .collect();
        let bi: Vec<Tensor> = (0..4)
            .map(|g| Tensor::random(&[h], 0.2 + g as f32 * 0.01))
            .collect();
        let x = Tensor::random(&[b, t, h], 0.5);
        let dx = GTensor::from_host(&gpu, &x);

        let mut per_step = SLstm::from_parts(
            &gpu, h, h, &w[0], &w[1], &w[2], &w[3], &bi[0], &bi[1], &bi[2], &bi[3], None,
        );
        per_step.force_fused_time = Some(false);
        let y_per = per_step.forward_alloc(&gpu, &dx, &tc);
        let want = y_per.to_host(&gpu);

        let mut fused = SLstm::from_parts(
            &gpu, h, h, &w[0], &w[1], &w[2], &w[3], &bi[0], &bi[1], &bi[2], &bi[3], None,
        );
        fused.force_fused_time = Some(true);
        let y_fused = fused.forward_alloc(&gpu, &dx, &tc);
        let got = y_fused.to_host(&gpu);

        let rel_tol = 1e-2;
        assert_eq!(want.data.len(), got.data.len());
        for (i, (a, c)) in want.data.iter().zip(got.data.iter()).enumerate() {
            assert!(
                (a - c).abs() <= rel_tol * a.abs().max(1.0),
                "fused vs per-step forward diverged at {i} (B={b}, H={h}): {a} vs {c}"
            );
        }

        let gy = GTensor::from_host(&gpu, &Tensor::random(&[b, t, h], 0.7));
        let close_rel = |dead: f32, want: &[f32], got: &[f32], what: &str| {
            let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            if scale < dead {
                return;
            }
            for (i, (a, c)) in want.iter().zip(got).enumerate() {
                assert!(
                    (a - c).abs() <= rel_tol * scale,
                    "{what} diverged at {i} (B={b}, H={h}): {a} vs {c} (scale {scale})"
                );
            }
        };
        let want_dx = per_step
            .backward_alloc(&gpu, &y_per, &gy, &tc)
            .to_host(&gpu);
        let got_dx = fused.backward_alloc(&gpu, &y_fused, &gy, &tc).to_host(&gpu);
        close_rel(0.0, &want_dx.data, &got_dx.data, "dx");
        let db_scale = (0..4)
            .flat_map(|gi| per_step.gate_db(&gpu, gi))
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        let dead = db_scale * 1e-3;
        for gi in 0..4 {
            // `dW` runs orders above the bias floor, so it is checked unconditionally.
            close_rel(
                0.0,
                &per_step.gate_dw(&gpu, gi),
                &fused.gate_dw(&gpu, gi),
                &format!("dW[{gi}]"),
            );
            close_rel(
                dead,
                &per_step.gate_db(&gpu, gi),
                &fused.gate_db(&gpu, gi),
                &format!("db[{gi}]"),
            );
        }
    }

    /// The batch-parallel fused forward (`slstm_batched_fwd`) must agree with the
    /// per-step path it replaces.
    #[test]
    fn slstm_batched_matches_per_step() {
        for (b, t) in [(120usize, 16usize), (341, 5), (1024, 1), (682, 2), (64, 33)] {
            batched_matches_per_step_at(b, t, 256);
        }
    }

    fn batched_matches_per_step_at(b: usize, t: usize, h: usize) {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        if !gpu.kernels.has_coop || ops::slstm_batched_geometry(&gpu, h, b).is_none() {
            return;
        }
        let s = 1.0 / (h as f32).sqrt();
        let w: Vec<Tensor> = (0..4)
            .map(|g| Tensor::random_seeded(&[2 * h, h], s * (1.0 + g as f32 * 0.05), 11 + g as u64))
            .collect();
        let bi: Vec<Tensor> = (0..4)
            .map(|g| Tensor::random_seeded(&[h], 0.2 + g as f32 * 0.01, 31 + g as u64))
            .collect();
        let x = GTensor::from_host(&gpu, &Tensor::random_seeded(&[b, t, h], 0.5, 7));
        let gy = GTensor::from_host(&gpu, &Tensor::random_seeded(&[b, t, h], 0.7, 9));

        let build = |batched: bool| {
            let mut c = SLstm::from_parts(
                &gpu, h, h, &w[0], &w[1], &w[2], &w[3], &bi[0], &bi[1], &bi[2], &bi[3], None,
            );
            c.force_fused_time = Some(false);
            c.force_batched = Some(batched);
            c
        };
        let mut per_step = build(false);
        let y_per = per_step.forward_alloc(&gpu, &x, &tc);
        let want = y_per.to_host(&gpu);
        let mut batched = build(true);
        let y_bat = batched.forward_alloc(&gpu, &x, &tc);
        let got = y_bat.to_host(&gpu);

        let rel_tol = 1e-2;
        for (i, (a, c)) in want.data.iter().zip(got.data.iter()).enumerate() {
            assert!(
                (a - c).abs() <= rel_tol * a.abs().max(1.0),
                "batched vs per-step forward diverged at {i} (B={b}, T={t}): {a} vs {c}"
            );
        }

        let close_rel = |want: &[f32], got: &[f32], what: &str| {
            let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            for (i, (a, c)) in want.iter().zip(got).enumerate() {
                assert!(
                    (a - c).abs() <= rel_tol * scale,
                    "{what} diverged at {i} (B={b}, T={t}): {a} vs {c} (scale {scale})"
                );
            }
        };
        let want_dx = per_step
            .backward_alloc(&gpu, &y_per, &gy, &tc)
            .to_host(&gpu);
        let got_dx = batched.backward_alloc(&gpu, &y_bat, &gy, &tc).to_host(&gpu);
        close_rel(&want_dx.data, &got_dx.data, "dx");
        for gi in 0..4 {
            close_rel(
                &per_step.gate_dw(&gpu, gi),
                &batched.gate_dw(&gpu, gi),
                &format!("dW[{gi}]"),
            );
        }
    }

    /// The batched path must honour `carry` — the state a chunk hands the next one.
    #[test]
    fn slstm_batched_carry_matches_per_step() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        let (b, h, chunks) = (64usize, 256usize, [6usize, 10]);
        if !gpu.kernels.has_coop || ops::slstm_batched_geometry(&gpu, h, b).is_none() {
            return;
        }
        let t: usize = chunks.iter().sum();
        let sc = 1.0 / (h as f32).sqrt();
        let w: Vec<Tensor> = (0..4)
            .map(|g| {
                Tensor::random_seeded(&[2 * h, h], sc * (1.0 + g as f32 * 0.05), 41 + g as u64)
            })
            .collect();
        let bi: Vec<Tensor> = (0..4)
            .map(|g| Tensor::random_seeded(&[h], 0.2, 61 + g as u64))
            .collect();
        let x = Tensor::random_seeded(&[b, t, h], 0.5, 13);
        let cut = |off: usize, len: usize| {
            let mut d = Vec::with_capacity(b * len * h);
            for r in 0..b {
                let s = (r * t + off) * h;
                d.extend_from_slice(&x.data[s..s + len * h]);
            }
            Tensor::new(&[b, len, h], d)
        };

        let sweep = |batched: bool| -> Vec<f32> {
            let mut cell = SLstm::from_parts(
                &gpu, h, h, &w[0], &w[1], &w[2], &w[3], &bi[0], &bi[1], &bi[2], &bi[3], None,
            );
            cell.force_fused_time = Some(false);
            cell.force_batched = Some(batched);
            cell.set_carry(true);
            cell.reset_state(&gpu);
            let mut y = Vec::with_capacity(b * t * h);
            let mut off = 0;
            for &len in &chunks {
                let xc = GTensor::from_host(&gpu, &cut(off, len));
                y.extend(cell.forward_alloc(&gpu, &xc, &tc).to_host(&gpu).data);
                off += len;
            }
            y
        };
        let (want, got) = (sweep(false), sweep(true));
        let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        for (i, (a, c)) in want.iter().zip(got.iter()).enumerate() {
            assert!(
                (a - c).abs() <= 1e-2 * scale,
                "carried forward diverged at {i}: {a} vs {c} (scale {scale})"
            );
        }
    }

    /// A chunked sweep with `carry` must reproduce the unchunked one.
    #[test]
    fn chunked_carry_matches_unchunked() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        let (b, inp, h) = (1, 8, 12);
        let chunks = [48usize, 48, 32]; // ragged: the last chunk is a different length
        let t: usize = chunks.iter().sum();

        let cpu = CpuSLstm::new(inp, h);
        let x = Tensor::random(&[b, t, inp], 0.5);
        let dy = Tensor::random(&[b, t, h], 1.0);

        // Reference: one call over the whole sequence.
        let mut whole = from_cpu(&gpu, &cpu);
        let y_whole_dev = whole.forward_alloc(&gpu, &GTensor::from_host(&gpu, &x), &tc);
        let y_whole = y_whole_dev.to_host(&gpu);
        let dx_whole = whole
            .backward_alloc(&gpu, &y_whole_dev, &GTensor::from_host(&gpu, &dy), &tc)
            .to_host(&gpu);

        // Chunked: same weights, same input, one call per chunk.
        let mut part = from_cpu(&gpu, &cpu);
        part.set_carry(true);
        part.reset_state(&gpu);

        let cut = |src: &Tensor, f: usize, off: usize, len: usize| {
            Tensor::new(&[b, len, f], src.data[off * f..(off + len) * f].to_vec())
        };

        let mut y_parts = Vec::with_capacity(t * h);
        let mut off = 0;
        for &c in &chunks {
            let xc = GTensor::from_host(&gpu, &cut(&x, inp, off, c));
            y_parts.extend(part.forward_alloc(&gpu, &xc, &tc).to_host(&gpu).data);
            off += c;
        }
        assert_close(&y_parts, &y_whole.data, 1e-4);

        let mut ends: Vec<usize> = Vec::with_capacity(chunks.len());
        let mut acc = 0;
        for &c in &chunks {
            ends.push(acc);
            acc += c;
        }
        part.zero_grad(&gpu);

        let mut dx_parts: Vec<Vec<f32>> = Vec::with_capacity(chunks.len());
        for (i, &c) in chunks.iter().enumerate().rev() {
            part.set_carry(true);
            part.reset_state(&gpu);
            let mut o = 0;
            for &pc in &chunks[..i] {
                let xp = GTensor::from_host(&gpu, &cut(&x, inp, o, pc));
                part.forward_alloc(&gpu, &xp, &tc);
                o += pc;
            }
            let xc = GTensor::from_host(&gpu, &cut(&x, inp, ends[i], c));
            let yc = part.forward_alloc(&gpu, &xc, &tc);

            if i + 1 == chunks.len() {
                part.reset_bptt(&gpu);
            }
            let dyc = GTensor::from_host(&gpu, &cut(&dy, h, ends[i], c));
            dx_parts.push(part.backward_alloc(&gpu, &yc, &dyc, &tc).to_host(&gpu).data);
        }
        dx_parts.reverse();
        let dx_flat: Vec<f32> = dx_parts.concat();
        assert_close(&dx_flat, &dx_whole.data, 1e-4);

        for gi in 0..4 {
            assert_close(&part.gate_dw(&gpu, gi), &whole.gate_dw(&gpu, gi), 1e-3);
            assert_close(&part.gate_db(&gpu, gi), &whole.gate_db(&gpu, gi), 1e-3);
        }
    }

    /// A reset row inside one call must split it into two independent sequences: the
    /// outputs, the input gradient and the summed weight gradients of the whole call
    /// equal those of the two halves run on their own, each from zero state — on the
    /// time-fused loops and on the per-step one.
    #[test]
    fn document_reset_matches_separate_sequences() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        for fused in [true, false] {
            reset_matches_split(&gpu, fused);
        }
    }

    fn reset_matches_split(gpu: &Gpu, fused: bool) {
        use cudarc::driver::DevicePtr;
        let tc = test_cache(gpu);
        let (b, inp, h, t, r) = (1, 8, 12, 80, 37);
        let cpu = CpuSLstm::new(inp, h);
        let x = Tensor::random(&[b, t, inp], 0.5);
        let dy = Tensor::random(&[b, t, h], 1.0);
        let cut = |src: &Tensor, f: usize, off: usize, len: usize| {
            GTensor::from_host(
                gpu,
                &Tensor::new(&[b, len, f], src.data[off * f..(off + len) * f].to_vec()),
            )
        };
        let cell = || {
            let mut c = from_cpu(gpu, &cpu);
            c.force_fused_time = Some(fused);
            c
        };

        let mask: Vec<i32> = (0..t).map(|i| i32::from(i == r)).collect();
        let mask = gpu.stream.clone_htod(&mask).expect("mask");
        let mut whole = cell();
        whole.set_resets(&Resets {
            rows: vec![r],
            mask: mask.device_ptr(&gpu.stream).0,
        });
        let yw = whole.forward_alloc(gpu, &GTensor::from_host(gpu, &x), &tc);
        let dxw = whole
            .backward_alloc(gpu, &yw, &GTensor::from_host(gpu, &dy), &tc)
            .to_host(gpu);

        let halves = [(0, r), (r, t - r)].map(|(off, len)| {
            let mut c = cell();
            let y = c.forward_alloc(gpu, &cut(&x, inp, off, len), &tc);
            let dx = c.backward_alloc(gpu, &y, &cut(&dy, h, off, len), &tc);
            (
                c,
                y.to_host(gpu).data.to_vec(),
                dx.to_host(gpu).data.to_vec(),
            )
        });
        let [(c0, y0, dx0), (c1, y1, dx1)] = &halves;
        assert_close(&yw.to_host(gpu).data, &[y0.as_slice(), y1].concat(), 1e-5);
        assert_close(&dxw.data, &[dx0.as_slice(), dx1].concat(), 1e-4);
        let add = |p: Vec<f32>, q: Vec<f32>| -> Vec<f32> {
            p.iter().zip(&q).map(|(a, b)| a + b).collect()
        };
        for gi in 0..4 {
            let dw = add(c0.gate_dw(gpu, gi), c1.gate_dw(gpu, gi));
            let db = add(c0.gate_db(gpu, gi), c1.gate_db(gpu, gi));
            assert_close(&whole.gate_dw(gpu, gi), &dw, 1e-3);
            assert_close(&whole.gate_db(gpu, gi), &db, 1e-3);
        }
    }

    /// The dispatch predicate is two-dimensional, and production depends on it landing
    /// the right way for shapes the model actually runs. `T` alone is not enough: the
    /// fused kernel's recurrent product is scalar and therefore linear in `B`, while the
    /// per-step path's cuBLAS call is flat until its tensor-core tiles fill, so a wide
    /// batch must NOT fuse (measured crossover B ~= 32).
    #[test]
    fn dispatch_picks_the_right_path_for_production_shapes() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let cell = SLstm::new_rand(&gpu, 6, 6);

        // Backbone: one sequence, a whole chunk of words. Fuses.
        assert!(
            cell.fuses_at(1, crate::config::BACKBONE_CHUNK),
            "backbone must fuse"
        );

        for t in 1..=(crate::config::MAX_WORD_BYTES + 1) {
            for b in [120, 227, 512, 1024, crate::config::GROUP_MAX_ROWS] {
                assert!(
                    !cell.fuses_at(b, t),
                    "encoder shape B={b} T={t} must not fuse"
                );
            }
        }

        // The batch axis alone decides it at a length that clears FUSED_MIN_T.
        assert!(cell.fuses_at(FUSED_MAX_B, 512), "B at the cap still fuses");
        assert!(
            !cell.fuses_at(FUSED_MAX_B + 1, 512),
            "one past the cap must not"
        );

        let enc = SLstm::new_rand(&gpu, crate::config::CHAR_HIDDEN, crate::config::CHAR_HIDDEN);
        for b in [120, 128, 186] {
            assert!(
                enc.batches_at(&gpu, b),
                "encoder group B={b} must take the batched path"
            );
        }
        for b in [292, 512, 1024, crate::config::GROUP_MAX_ROWS] {
            assert!(
                !enc.batches_at(&gpu, b),
                "B={b} re-reads too much to be worth it"
            );
        }
        assert!(
            !enc.batches_at(&gpu, SB_MIN_B - 1),
            "a batch that cannot fill an mma tile"
        );
    }
}
