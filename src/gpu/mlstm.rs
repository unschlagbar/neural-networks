//! Device-resident multi-head mLSTM cell, GPU counterpart of
//! [`cpu::mlstm::MLstm`](crate::cpu::mlstm::MLstm) — in the **parallel /
//! chunkwise** formulation, not the scalar per-head recurrence (the CPU's sub-1×
//! path; see PLAN-gpu.md Phase C).

use std::sync::OnceLock;

use super::arena::{self, ParamSlot};
use super::block::{Cell, Resets};
use super::offload::{Frame, ResidentFrames};
use super::ops::{GpuRmsForward, SlabBuf};
use super::{GTensor, Gpu, linear::Linear, ops, rms_norm::RmsNorm};
use crate::gpu::arena::TrainingCache;
use crate::cpu::optim::AdamCfg;
use crate::tensor::Tensor;

/// Lay `[rows, ·]` weight (or `[·]` bias) tensors side by side into the one matrix
/// a fused projection holds.
fn concat_cols(parts: &[&Tensor]) -> Tensor {
    let rows = if parts[0].dims().len() == 1 {
        1
    } else {
        parts[0].dims()[0]
    };
    let cols: Vec<usize> = parts.iter().map(|p| p.data.len() / rows).collect();
    let total: usize = cols.iter().sum();
    let mut data = Vec::with_capacity(rows * total);
    for r in 0..rows {
        for (p, c) in parts.iter().zip(&cols) {
            data.extend_from_slice(&p.data[r * c..(r + 1) * c]);
        }
    }
    let dims: &[usize] = if parts[0].dims().len() == 1 {
        &[total]
    } else {
        &[rows, total]
    };
    Tensor::new(dims, data)
}

/// Chunk length: `config::MLSTM_CHUNK`, overridable with `MLSTM_CHUNK=<L>` for A/B
/// runs. Resolved once — the env read must not sit in forward.
fn chunk_len() -> usize {
    static L: OnceLock<usize> = OnceLock::new();
    *L.get_or_init(|| {
        std::env::var("MLSTM_CHUNK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(crate::config::MLSTM_CHUNK)
    })
}

/// What a forward saves for its backward, carved from a frame — see [`MLstm::carve`].
pub struct MlstmSaved {
    xh: ops::SlabBuf,
    gates: GTensor<f32>,
    fused: ops::MlstmFused,
    /// The head norm's output: read by `ogate_bwd`, what the norm's backward rebuilds
    /// `x̂` from, and — with `o` — what `lin_out`'s input `σ(o) ⊙ yhat` is rebuilt from.
    yhat: GTensor<f32>,
    headnorm: GpuRmsForward,
}

pub struct MLstm {
    pub input_size: usize,
    pub d: usize,
    pub heads: usize,
    pub dqk: usize,
    pub dhv: usize,
    /// Chunk length (0 = single-chunk). Defaults to [`chunk_len`].
    chunk: usize,

    pub(super) lin_qkvo: Linear,
    pub(super) lin_gates: Linear,
    pub(super) lin_out: Linear,
    pub(super) headnorm: RmsNorm, // head-wise (group == dhv)
    /// Frames of a cell used on its own. Inside a [`Block`](super::block::Block) the
    /// block's frame holds this cell's cache and this stays empty.
    frames: ResidentFrames,
    /// Continue the previous call's recurrence instead of starting at zero — set for a
    /// chunked sweep. See [`MLstm::set_carry`].
    carry: bool,
    /// The state the previous chunk ended with, seeded into the next chunk's kernel.
    /// `None` before the first chunk of a sweep (and whenever `carry` is off).
    carry_state: Option<ops::MlstmState>,
    /// BPTT state from the chunk to the right, for a chunked backward.
    carry_dstate: Option<ops::MlstmDState>,
    /// Rows of the next call where a document starts. See [`set_resets`](Self::set_resets).
    resets: Vec<usize>,
}

/// The forget-gate logit forced onto a document's first row, which makes the state
/// arriving from the row before decay by `σ(-100) ≈ e^-100` — a restart, done inside
/// the chunkwise kernels instead of by cutting the sequence there (`reset_fgate_value`
/// in NX-AI's xlstm-jax, `blocks/mlstm/cell.py`).
pub const DOC_RESET_FGATE: f32 = -100.0;

impl MLstm {
    /// Build from a CPU cell's host weights (all 15 parameter tensors uploaded).
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        gpu: &Gpu,
        input_size: usize,
        d: usize,
        heads: usize,
        dqk: usize,
        wq: &Tensor,
        wk: &Tensor,
        wv: &Tensor,
        wo: &Tensor,
        wi: &Tensor,
        wf: &Tensor,
        bq: &Tensor,
        bk: &Tensor,
        bv: &Tensor,
        bo: &Tensor,
        bi: &Tensor,
        bf: &Tensor,
        w_out: &Tensor,
        b_out: &Tensor,
        gamma: &Tensor,
    ) -> Self {
        let dhv = d / heads;
        let inv_sqrt_dqk = 1.0 / (dqk as f32).sqrt();
        let scaled =
            |t: &Tensor| Tensor::new(t.dims(), t.data.iter().map(|v| v * inv_sqrt_dqk).collect());
        let (wk, bk) = (&scaled(wk), &scaled(bk));
        Self {
            input_size,
            d,
            heads,
            dqk,
            dhv,
            chunk: chunk_len(),
            lin_qkvo: Linear::from_parts(
                gpu,
                &concat_cols(&[wq, wk, wv, wo]),
                &concat_cols(&[bq, bk, bv, bo]),
            ),
            lin_gates: Linear::from_parts(gpu, &concat_cols(&[wi, wf]), &concat_cols(&[bi, bf])),
            lin_out: Linear::from_parts(gpu, w_out, b_out),
            headnorm: RmsNorm::from_parts_grouped(gpu, gamma, dhv),
            frames: ResidentFrames::default(),
            carry: false,
            carry_state: None,
            carry_dstate: None,
            resets: Vec::new(),
        }
    }

    /// Freshly-initialised cell, matching `cpu::MLstm::new`'s init exactly.
    pub fn new_rand(gpu: &Gpu, input_size: usize, d: usize, heads: usize, dqk: usize) -> Self {
        Self::from_cpu(gpu, &crate::cpu::MLstm::new(input_size, d, heads, dqk))
    }

    /// Column where `o` starts in the fused `q ‖ k ‖ v ‖ o` projection.
    fn o_off(&self) -> usize {
        2 * self.heads * self.dqk + self.d
    }

    /// Upload a CPU cell (weights are copied; grads/moments start at zero).
    pub fn from_cpu(gpu: &Gpu, c: &crate::cpu::MLstm) -> Self {
        Self::from_parts(
            gpu,
            c.input_size,
            c.d,
            c.heads,
            c.dqk,
            &c.wq,
            &c.wk,
            &c.wv,
            &c.wo,
            &c.wi,
            &c.wf,
            &c.bq,
            &c.bk,
            &c.bv,
            &c.bo,
            &c.bi,
            &c.bf,
            &c.w_out,
            &c.b_out,
            &c.gamma,
        )
    }

    /// Override the chunk length. Lets a caller — or a test — pick a length per cell
    /// instead of taking the `config`/env default; `fused_chunk` clamps it into what
    /// the kernels support.
    pub fn set_chunk(&mut self, chunk: usize) {
        self.chunk = chunk;
    }

    /// The chunk length the fused kernels run this sequence at.
    fn fused_chunk(&self, gpu: &Gpu, t: usize) -> usize {
        let l = self.chunk.min(ops::FUSED_MAX_L).min(t).max(1);
        debug_assert!(
            ops::mlstm_fused_smem_bytes(l, self.dqk, self.dhv) <= gpu.max_shared_optin,
            "fused mLSTM shared memory exceeds this device's opt-in limit",
        );
        l
    }

    /// Whether `ogate_fwd` writes `lin_out`'s input narrow: only when that GEMM reads
    /// bf16 and the kernels can store it.
    fn hconcat_narrow(&self, gpu: &Gpu) -> bool {
        gpu.kernels.slab_bf16 && self.lin_out.is_bf16()
    }

    fn shape(&self, batch: usize, t: usize) -> ops::MlstmShape {
        ops::MlstmShape {
            batch,
            heads: self.heads,
            t,
            dqk: self.dqk,
            dhv: self.dhv,
        }
    }

    /// This cell's saved tensors for a `[B, T, ·]` call, carved from `f` in the one order
    /// its forward and backward share.
    pub fn carve(&self, gpu: &Gpu, f: &mut Frame, b: usize, t: usize) -> MlstmSaved {
        let rows = b * t;
        MlstmSaved {
            xh: f.slab(
                gpu,
                &[rows, self.lin_qkvo.output_size()],
                gpu.kernels.slab_bf16,
            ),
            gates: f.f32(gpu, &[rows, self.lin_gates.output_size()]),
            fused: ops::MlstmFused::carve(gpu, f, self.shape(b, t), self.fused_chunk(gpu, t)),
            yhat: f.f32(gpu, &[rows, self.d]),
            headnorm: self.headnorm.carve(gpu, f, rows),
        }
    }

    /// Forward over `[B, T, in]` → `[B, T, d]`, saving into a frame of this cell's own.
    /// See [`forward_saved`](Self::forward_saved).
    pub fn forward(
        &mut self,
        gpu: &Gpu,
        x: &GTensor<f32>,
        out: &mut GTensor<f32>,
        cache: &TrainingCache,
    ) {
        let (b, t) = (x.shape[0], x.shape[1]);
        let mut m = Frame::measure();
        self.carve(gpu, &mut m, b, t);
        if !self.carry {
            self.frames.reset();
        }
        let mut f = self.frames.push(gpu, m.bytes());
        let mut sv = self.carve(gpu, &mut f, b, t);
        let xs = SlabBuf::F32(GTensor::view(gpu, &x.buf, 0, x.dims()));
        self.forward_saved(gpu, &xs, out, &mut sv, cache);
    }

    /// Forward over `[B, T, in]` → `[B, T, d]`, saving into `sv`.
    pub fn forward_saved(
        &mut self,
        gpu: &Gpu,
        x: &SlabBuf,
        out: &mut GTensor<f32>,
        sv: &mut MlstmSaved,
        cache: &TrainingCache,
    ) {
        let xd = x.dims();
        assert_eq!(xd.len(), 3, "MLstm::forward expects [B, T, in]");
        let (batch, t, inp) = (xd[0], xd[1], xd[2]);
        assert_eq!(
            inp, self.input_size,
            "MLstm::forward — input width mismatch"
        );
        let dim = self.d;
        assert_eq!(out.dims(), [batch, t, dim], "MLstm::forward — output shape");
        let rows = batch * t;
        let shape = self.shape(batch, t);

        let xv = x.view(gpu, &[rows, inp]);

        let xf = xv.at_width(gpu, self.lin_qkvo.is_bf16(), &cache.temps);
        self.lin_qkvo.forward_slab_slab(gpu, &xf.x, &mut sv.xh);
        self.lin_gates.forward_slab_lhs(gpu, &xf.x, &mut sv.gates);
        drop(xf);
        // The frame keeps the forced logits, so backward's recompute sees the same gates.
        self.force_forget(gpu, &mut sv.gates, batch);

        let mut avec = cache
            .temps
            .get_small::<f32>(gpu, &[shape.bh(), sv.fused.nc, sv.fused.l]);
        ops::mlstm_fused_fw_into(
            gpu,
            &sv.xh,
            &sv.gates,
            if self.carry {
                self.carry_state.as_ref()
            } else {
                None
            },
            shape,
            &mut sv.fused,
            &mut avec,
        );
        drop(avec);
        if self.carry {
            sv.fused
                .final_state_into(gpu, shape.bh(), self.dhv, self.dqk, &mut self.carry_state);
        }
        // `ytil` comes out `[rows, d]` already; the head norm only needs it widened.
        let mut h_tilde = cache.temps.get::<f32>(gpu, &[rows, dim]);
        self.headnorm.forward_saved(
            gpu,
            sv.fused.ytil.as_f32(gpu, &mut h_tilde),
            &mut sv.yhat,
            &mut sv.headnorm,
        );
        drop(h_tilde);
        // Narrow when `lin_out`'s GEMM reads bf16: it is the only reader.
        let mut hconcat = ops::SlabAt::temp(gpu, &cache.temps, &[rows, dim], self.hconcat_narrow(gpu));
        ops::ogate_fwd(gpu, &sv.xh, &sv.yhat, &mut hconcat.x, self.o_off());
        out.reshape_to(&[rows, dim]);
        self.lin_out.forward_slab_lhs(gpu, &hconcat.x, out);
        out.reshape_to(&[batch, t, dim]);
    }

    /// Device bytes held, split `(params, activations)`. Diagnostic — see
    /// [`Hierarchical::retained_report`](super::hierarchical::Hierarchical::retained_report).
    pub fn retained_bytes(&self) -> (usize, usize) {
        let (mut params, mut act) = (0, 0);
        for l in [&self.lin_qkvo, &self.lin_gates, &self.lin_out] {
            let (p, a) = l.retained_bytes();
            params += p;
            act += a;
        }
        let (hn_p, hn_a) = self.headnorm.retained_bytes();
        params += hn_p;
        act += hn_a;
        act += self.frames.device_bytes();
        (params, act)
    }

    /// Continue the previous call's recurrence rather than starting from zero.
    pub fn set_carry(&mut self, carry: bool) {
        self.carry = carry;
        if !carry {
            self.carry_state = None;
            self.carry_dstate = None;
        }
    }

    /// Drop the carried state, so the next forward starts the recurrence at zero
    /// whatever `carry` says.
    pub fn reset_state(&mut self, gpu: &Gpu) {
        self.zero_state(gpu);
        self.reset_caches(gpu);
    }

    /// Start the recurrence over at the next forward, keeping everything already
    /// cached.
    pub fn zero_state(&mut self, _gpu: &Gpu) {
        self.carry_state = None;
    }

    /// Drop the caches of an already-unwound sweep, keeping `C`/`n`/`m` — so the next
    /// forward continues this recurrence. See [`Cell::reset_caches`].
    pub fn reset_caches(&mut self, _gpu: &Gpu) {
        self.frames.reset();
    }

    /// Drop the carried BPTT state, so the next backward starts with no incoming
    /// gradient from the right. Call before the rightmost chunk's backward.
    pub fn reset_bptt(&mut self, _gpu: &Gpu) {
        self.carry_dstate = None;
    }

    /// Where documents start inside the next call. See [`Cell::set_resets`].
    pub fn set_resets(&mut self, r: &Resets) {
        self.resets.clone_from(&r.rows);
    }

    /// Overwrite the forget-gate logit of every reset row with [`DOC_RESET_FGATE`].
    fn force_forget(&self, gpu: &Gpu, gates: &mut GTensor<f32>, batch: usize) {
        if self.resets.is_empty() {
            return;
        }
        assert_eq!(batch, 1, "MLstm: document resets on a batched call");
        let h = self.heads;
        for &r in &self.resets {
            ops::fill_f32(gpu, gates, r * 2 * h + h, h, DOC_RESET_FGATE);
        }
    }

    /// Drop the forward caches, leaving the pool and the weight staging alone —
    /// for a caller that forwards repeatedly and never backwards.
    pub fn drop_saved(&mut self) {
        self.frames.reset();
    }

    /// Retained activation bytes split `(saved_cache, other)`.
    pub fn act_split(&self) -> (usize, usize) {
        let saved = self.frames.device_bytes();
        let (_, all) = self.retained_bytes();
        (saved, all - saved)
    }

    /// Release everything a forward left behind that no backward will read: the
    /// frames, and each projection's saved input and bf16 staging.
    pub fn drop_all_act(&mut self, gpu: &Gpu) {
        self.frames.release();
        for l in [&mut self.lin_qkvo, &mut self.lin_gates, &mut self.lin_out] {
            l.drop_saved_act(gpu);
        }
        self.headnorm.drop_saved_act();
    }

    /// Backward over `[B, T, d]` → `dx` `[B, T, in]`, reading the frame the matching
    /// [`forward`](Self::forward) pushed. Accumulates all grads.
    pub fn backward(
        &mut self,
        gpu: &Gpu,
        x: &GTensor<f32>,
        dy: &GTensor<f32>,
        dx: &mut GTensor<f32>,
        cache: &TrainingCache,
    ) {
        let mut f = self.frames.pop(gpu);
        let mut sv = self.carve(gpu, &mut f, dy.shape[0], dy.shape[1]);
        let xs = SlabBuf::F32(GTensor::view(gpu, &x.buf, 0, x.dims()));
        self.backward_saved(gpu, &xs, dy, dx, &mut sv, cache)
    }

    /// Backward into a freshly allocated `dx` `[B, T, in]` — the by-value
    /// companion to [`backward`](Self::backward).
    pub fn backward_alloc(
        &mut self,
        gpu: &Gpu,
        x: &GTensor<f32>,
        dy: &GTensor<f32>,
        cache: &TrainingCache,
    ) -> GTensor<f32> {
        let mut dx = GTensor::uninit(gpu, &[dy.shape[0], dy.shape[1], self.input_size]);
        self.backward(gpu, x, dy, &mut dx, cache);
        dx
    }

    /// Backward over `[B, T, d]` → `dx` `[B, T, in]`, reading what
    /// [`forward_saved`](Self::forward_saved) wrote into `sv`. Accumulates all grads.
    pub fn backward_saved(
        &mut self,
        gpu: &Gpu,
        x: &SlabBuf,
        dy: &GTensor<f32>,
        dx: &mut GTensor<f32>,
        sv: &mut MlstmSaved,
        cache: &TrainingCache,
    ) {
        let (d, h, dqk, dhv, inp) = (self.d, self.heads, self.dqk, self.dhv, self.input_size);
        let (b, t) = (dy.shape[0], dy.shape[1]);
        let n = b * t;
        assert!(
            n * inp <= dx.capacity(),
            "MLstm::backward — dx holds {} elements, needs {}",
            dx.capacity(),
            n * inp
        );

        let dy_flat = GTensor::view(gpu, &dy.buf, 0, &[n, d]);
        let mut hconcat = ops::SlabAt::temp(gpu, &cache.temps, &[n, d], self.hconcat_narrow(gpu));
        ops::ogate_fwd(gpu, &sv.xh, &sv.yhat, &mut hconcat.x, self.o_off());
        let mut d_hconcat = cache.temps.get::<f32>(gpu, &[n, d]);
        self.lin_out
            .backward_slab_x(gpu, &hconcat.x, &dy_flat, &mut d_hconcat, 0.0, cache);
        drop(hconcat);
        let mut dxh = cache
            .temps
            .get::<f32>(gpu, &[n, self.lin_qkvo.output_size()]);
        let mut d_yhat = cache.temps.get::<f32>(gpu, &[n, d]);
        ops::ogate_bwd(
            gpu,
            &d_hconcat,
            &sv.xh,
            &sv.yhat,
            &mut dxh,
            &mut d_yhat,
            self.o_off(),
        );
        drop(d_hconcat);
        let mut d_ytil = cache.temps.get::<f32>(gpu, &[n, d]);
        self.headnorm
            .backward_saved(gpu, &d_yhat, &sv.yhat, &sv.headnorm, None, &mut d_ytil, cache);
        drop(d_yhat);

        let mut dgates = cache.temps.get_small::<f32>(gpu, &[n, 2 * h]);
        let dstate = ops::mlstm_fused_bw(
            gpu,
            &sv.fused,
            &sv.xh,
            &sv.gates,
            &mut dxh,
            &d_ytil,
            if self.carry {
                self.carry_dstate.take()
            } else {
                None
            },
            &mut dgates,
            &cache.temps,
            ops::MlstmShape {
                batch: b,
                heads: h,
                t,
                dqk,
                dhv,
            },
        );
        drop(d_ytil);
        if self.carry {
            self.carry_dstate = Some(dstate);
        }
        for &r in &self.resets {
            ops::fill_f32(gpu, &mut dgates, r * 2 * h + h, h, 0.0);
        }

        let mut acc = GTensor::view(gpu, &dx.buf, 0, &[n, inp]);
        let xv = x.view(gpu, &[n, inp]);
        let xf = xv.at_width(gpu, self.lin_qkvo.is_bf16(), &cache.temps);
        self.lin_qkvo
            .backward_slab_x(gpu, &xf.x, &dxh, &mut acc, 0.0, cache);
        self.lin_gates
            .backward_slab_x(gpu, &xf.x, &dgates, &mut acc, 1.0, cache);
        drop((acc, xf, dxh, dgates));
        dx.reshape_to(&[b, t, inp]);
    }

    /// Every parameter with its gradient and AdamW moments. The projection and
    /// output matrices decay, biases and the head-norm γ do not — all decided by
    /// the sub-layers.
    pub fn param_slots(&mut self) -> Vec<ParamSlot<'_>> {
        let mut v = Vec::new();
        for l in [&mut self.lin_qkvo, &mut self.lin_gates, &mut self.lin_out] {
            v.extend(l.param_slots());
        }
        v.extend(self.headnorm.param_slots());
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

    pub fn zero_grad(&mut self, gpu: &Gpu) {
        for l in [&mut self.lin_qkvo, &mut self.lin_gates, &mut self.lin_out] {
            l.zero_grad(gpu);
        }
        self.headnorm.zero_grad(gpu);
    }

    /// AdamW over this cell's own parameters, then clear the grads. A model steps
    /// its whole `ParamArena` in one launch instead.
    pub fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        arena::step_slots(gpu, &mut self.param_slots(), cfg);
    }
}

impl Cell for MLstm {
    type Saved = MlstmSaved;
    /// The head norm keeps its own output (`yhat`, for the o-gate), so this cell has
    /// no use for the block's copy.
    const NEEDS_Y: bool = false;
    fn carve(&self, gpu: &Gpu, f: &mut Frame, b: usize, t: usize) -> MlstmSaved {
        MLstm::carve(self, gpu, f, b, t)
    }
    fn forward_saved(
        &mut self,
        gpu: &Gpu,
        x: &SlabBuf,
        out: &mut GTensor<f32>,
        sv: &mut MlstmSaved,
        cache: &TrainingCache,
    ) {
        MLstm::forward_saved(self, gpu, x, out, sv, cache)
    }
    fn backward_saved(
        &mut self,
        gpu: &Gpu,
        x: &SlabBuf,
        _y: Option<&GTensor<f32>>,
        dy: &GTensor<f32>,
        dx: &mut GTensor<f32>,
        sv: &mut MlstmSaved,
        cache: &TrainingCache,
    ) {
        MLstm::backward_saved(self, gpu, x, dy, dx, sv, cache)
    }
    fn zero_grad(&mut self, gpu: &Gpu) {
        MLstm::zero_grad(self, gpu)
    }
    fn param_slots(&mut self) -> Vec<ParamSlot<'_>> {
        MLstm::param_slots(self)
    }
    fn phase_buckets(&self) -> (super::block::phase::Bucket, super::block::phase::Bucket) {
        use super::block::phase::Bucket;
        (Bucket::MlstmCellFwd, Bucket::MlstmCellBwd)
    }
    fn trim_to(&mut self, _rows: usize) {}
    fn retained_bytes(&self) -> (usize, usize) {
        MLstm::retained_bytes(self)
    }
    fn drop_all_act(&mut self, gpu: &Gpu) {
        MLstm::drop_all_act(self, gpu)
    }
    fn act_split(&self) -> (usize, usize) {
        MLstm::act_split(self)
    }
    fn set_carry(&mut self, carry: bool) {
        MLstm::set_carry(self, carry)
    }
    fn reset_state(&mut self, gpu: &Gpu) {
        MLstm::reset_state(self, gpu)
    }
    fn reset_caches(&mut self, gpu: &Gpu) {
        MLstm::reset_caches(self, gpu)
    }
    fn reset_bptt(&mut self, gpu: &Gpu) {
        MLstm::reset_bptt(self, gpu)
    }
    fn take_carried(&mut self, _gpu: &Gpu) -> super::block::CarriedState {
        super::block::CarriedState::Mlstm(self.carry_state.take())
    }
    fn put_carried(&mut self, state: super::block::CarriedState) {
        if let super::block::CarriedState::Mlstm(s) = state {
            self.carry_state = s;
        }
    }
    fn set_resets(&mut self, r: &Resets) {
        MLstm::set_resets(self, r)
    }
}

#[cfg(test)]
mod tests {

    /// One temp cache per test, sized past every shape this module presents — the
    /// widest is `[b·t, heads·(2·dqk + 2·dhv)]` at `b·t = 400`, `dqk = dhv = 96`.
    fn test_cache(gpu: &Gpu) -> TrainingCache {
        TrainingCache::new(gpu, 1 << 20, 1 << 16, 1 << 24)
    }
    use super::*;
    use crate::cpu::mlstm::MLstm as CpuMLstm;
    use crate::cpu::optim::AdamCfg;

    fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length mismatch");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < tol, "{what}[{i}]: gpu {g} vs cpu {w}");
        }
    }

    /// Worst absolute difference measured against the tensor's own magnitude.
    fn cols_of(m: &[f32], w: usize, at: usize, cols: usize) -> Vec<f32> {
        m.chunks(w)
            .flat_map(|r| r[at..at + cols].iter().copied())
            .collect()
    }

    /// Lay separate `[B, T, ·]` q/k/v out the way the fused projection writes them —
    /// concatenated into `[B*T, H*(2*dqk + 2*dhv)]`, with `H = 1` for these tests.
    fn pack_qkv(
        gpu: &Gpu,
        q: &GTensor<f32>,
        k: &GTensor<f32>,
        v: &GTensor<f32>,
        bt: usize,
        dqk: usize,
        dhv: usize,
    ) -> ops::SlabBuf {
        let (qh, kh, vh) = (q.to_host(gpu), k.to_host(gpu), v.to_host(gpu));
        let w = 2 * dqk + 2 * dhv;
        let mut out = Vec::with_capacity(bt * w);
        for r in 0..bt {
            out.extend_from_slice(&qh.data[r * dqk..(r + 1) * dqk]);
            out.extend_from_slice(&kh.data[r * dqk..(r + 1) * dqk]);
            out.extend_from_slice(&vh.data[r * dhv..(r + 1) * dhv]);
            out.resize(out.len() + dhv, 0.0);
        }
        let t = GTensor::from_host(gpu, &Tensor::new(&[bt, w], out));
        ops::SlabBuf::from_f32(gpu, t)
    }

    /// The gate twin of [`pack_qkv`]: `ĩ ‖ f̃` into `[B*T, 2*H]`, `H = 1`.
    fn pack_gates(gpu: &Gpu, ig: &GTensor<f32>, fg: &GTensor<f32>, bt: usize) -> GTensor<f32> {
        let (ih, fh) = (ig.to_host(gpu), fg.to_host(gpu));
        let mut out = Vec::with_capacity(bt * 2);
        for r in 0..bt {
            out.push(ih.data[r]);
            out.push(fh.data[r]);
        }
        GTensor::from_host(gpu, &Tensor::new(&[bt, 2], out))
    }

    /// The right check when two float orderings are compared: a per-element
    /// relative test explodes on elements that happen to sit near zero, while an
    /// absolute one silently becomes a very tight relative demand on small tensors.
    fn assert_close_rel(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length mismatch");
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let (worst, at) =
            got.iter()
                .zip(want)
                .enumerate()
                .fold((0.0f32, 0usize), |(m, at), (i, (&a, &b))| {
                    let d = (a - b).abs();
                    if d > m { (d, i) } else { (m, at) }
                });
        assert!(
            worst / scale.max(f32::MIN_POSITIVE) < tol,
            "{what}: worst |a-b| {worst:.3e} at [{at}] on scale {scale:.3e} \
             -> {:.2e} relative, exceeds {tol:.0e}",
            worst / scale.max(f32::MIN_POSITIVE)
        );
    }

    /// Single-chunk parallel forward+backward+step must match the CPU scalar
    /// recurrence (`cpu::MLstm`) from identical weights. The CPU backward is
    /// itself FD-verified, so a GPU-vs-CPU grad match is the (tighter) check.
    #[test]
    fn mlstm_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        let (b, t, inp, d, heads, dqk) = (2, 6, 5, 8, 2, 4); // dhv = 4

        let mut cpu = CpuMLstm::new(inp, d, heads, dqk);
        // Non-trivial gate weights so the decay/stabilizer path is exercised.
        cpu.wi = Tensor::random(&[inp, heads], 0.3);
        cpu.wf = Tensor::random(&[inp, heads], 0.3);
        let mut dev = MLstm::from_cpu(&gpu, &cpu);

        let x = Tensor::random(&[b, t, inp], 0.5);
        let g = Tensor::random(&[b, t, d], 1.0);

        // Forward
        let y_cpu = cpu.forward(&x);
        let xd = GTensor::from_host(&gpu, &x);
        let mut y_dev = GTensor::uninit(&gpu, &[b, t, d]);
        dev.forward(&gpu, &xd, &mut y_dev, &tc);
        assert_close(&y_dev.to_host(&gpu).data, &y_cpu.data, 3e-3, "y");

        // Backward
        let dx_cpu = cpu.backward(&g);
        let dx_dev = dev.backward_alloc(&gpu, &xd, &GTensor::from_host(&gpu, &g), &tc);
        assert_close(&dx_dev.to_host(&gpu).data, &dx_cpu.data, 3e-3, "dx");

        // One AdamW step; compare representative updated parameters.
        let mut cfg = AdamCfg::new(1e-3, 0.01);
        cfg.t = 1;
        cpu.step(&cfg);
        dev.step(&gpu, &cfg);
        // (weights live in the Linear sub-layers; check q, v, out projections + γ)
        let xw = dev.lin_qkvo.w.to_host(&gpu).data;
        let xwid = dev.lin_qkvo.output_size();
        assert_close(
            &cols_of(&xw, xwid, 0, heads * dqk),
            &cpu.wq.data,
            3e-3,
            "wq",
        );
        assert_close(
            &cols_of(&xw, xwid, 2 * heads * dqk, d),
            &cpu.wv.data,
            3e-3,
            "wv",
        );
        assert_close(
            &dev.lin_out.w.to_host(&gpu).data,
            &cpu.w_out.data,
            3e-3,
            "w_out",
        );
        assert_close(
            &dev.headnorm.gamma.to_host(&gpu).data,
            &cpu.gamma.data,
            3e-3,
            "gamma",
        );
    }

    /// Chunking is an exact refactoring of the single-chunk form, so every chunk
    /// length must give the same forward, the same dx and the same weight update —
    /// including a length that leaves a SHORT final chunk (T=20, L=8 → 8+8+4), and
    /// L=1 (the fully recurrent extreme, where every intra-chunk matrix is 1×1 and
    /// all the work goes through the carried state).
    #[test]
    fn mlstm_chunking_matches_single_chunk() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        let (b, t, inp, d, heads, dqk) = (2, 20, 5, 8, 2, 4);

        let mut proto = CpuMLstm::new(inp, d, heads, dqk);
        proto.wi = Tensor::random_seeded(&[inp, heads], 0.3, 0xB1);
        proto.wf = Tensor::random_seeded(&[inp, heads], 0.3, 0xB2);

        let x = Tensor::random_seeded(&[b, t, inp], 0.5, 0xB3);
        let g = Tensor::random_seeded(&[b, t, d], 1.0, 0xB4);
        let dx = GTensor::from_host(&gpu, &x);
        let dg = GTensor::from_host(&gpu, &g);

        // Reference: one chunk over the whole sequence.
        let run = |chunk: usize| {
            let mut dev = MLstm::from_cpu(&gpu, &proto);
            dev.set_chunk(chunk);
            let mut yt = GTensor::uninit(&gpu, &[b, t, d]);
            dev.forward(&gpu, &dx, &mut yt, &tc);
            let y = yt.to_host(&gpu).data;
            let dxo = dev.backward_alloc(&gpu, &dx, &dg, &tc).to_host(&gpu).data;
            let mut cfg = AdamCfg::new(1e-3, 0.01);
            cfg.t = 1;
            dev.step(&gpu, &cfg);
            // wf/wi ride the decay path; w_out rides the value path.
            let wf = cols_of(&dev.lin_gates.w.to_host(&gpu).data, 2 * heads, heads, heads);
            let wout = dev.lin_out.w.to_host(&gpu).data;
            (y, dxo, wf, wout)
        };

        let slab = if gpu.kernels.slab_bf16 { 8.0 } else { 1.0 };
        let (y0, dx0, wf0, wo0) = run(0); // single chunk
        for l in [1, 3, 8, 16, 32] {
            let (y, dxo, wf, wo) = run(l);
            assert_close(&y, &y0, 2e-4 * slab, &format!("y (chunk {l})"));
            assert_close(&dxo, &dx0, 2e-4 * slab, &format!("dx (chunk {l})"));
            assert_close_rel(&wf, &wf0, 1e-3 * slab, &format!("wf (chunk {l})"));
            assert_close_rel(&wo, &wo0, 1e-3 * slab, &format!("w_out (chunk {l})"));
        }
    }

    /// Chunked (L = 32) vs step-by-step (L = 1), at a shape wide enough to matter.
    #[test]
    fn mlstm_fused_chunking_matches_unit_chunk() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        let (b, t, inp, d, heads, dqk) = (2, 200, 64, 512, 8, 64); // dhv = 64
        assert!(
            t % super::ops::FUSED_MAX_L != 0,
            "the last chunk must be short"
        );

        let mut proto = CpuMLstm::new(inp, d, heads, dqk);
        proto.wi = Tensor::random_seeded(&[inp, heads], 0.3, 0xA1);
        proto.wf = Tensor::random_seeded(&[inp, heads], 0.3, 0xA2);

        let x = GTensor::from_host(&gpu, &Tensor::random_seeded(&[b, t, inp], 0.5, 0xA3));
        let g = GTensor::from_host(&gpu, &Tensor::random_seeded(&[b, t, d], 1.0, 0xA4));

        let run = |chunk: usize| {
            let mut dev = MLstm::from_cpu(&gpu, &proto);
            dev.set_chunk(chunk);
            let mut yt = GTensor::uninit(&gpu, &[b, t, d]);
            dev.forward(&gpu, &x, &mut yt, &tc);
            let y = yt.to_host(&gpu).data;
            let dx = dev.backward_alloc(&gpu, &x, &g, &tc).to_host(&gpu).data;
            let mut cfg = AdamCfg::new(1e-3, 0.01);
            cfg.t = 1;
            dev.step(&gpu, &cfg);
            // rides the decay path
            let wf = cols_of(&dev.lin_gates.w.to_host(&gpu).data, 2 * heads, heads, heads);
            let wout = dev.lin_out.w.to_host(&gpu).data; // rides the value path
            (y, dx, wf, wout)
        };

        let (y0, dx0, wf0, wo0) = run(1);
        let (y1, dx1, wf1, wo1) = run(32);

        let slab = if gpu.kernels.slab_bf16 { 8.0 } else { 1.0 };
        assert_close(&y1, &y0, 2e-3 * slab, "y");
        assert_close(&dx1, &dx0, 2e-3 * slab, "dx");
        assert_close_rel(&wf1, &wf0, 5e-3 * slab, "wf");
        assert_close_rel(&wo1, &wo0, 5e-3 * slab, "w_out");
    }

    /// The fused forward and backward against the CPU scalar recurrence at a shape
    /// with a SHORT final chunk (T = 200 = 6·32 + 8).
    #[test]
    fn mlstm_fused_ragged_chunk_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        let (b, t, inp, d, heads, dqk) = (2, 200, 64, 512, 8, 64); // dhv = 64
        assert!(
            t % super::ops::FUSED_MAX_L != 0,
            "the last chunk must be short"
        );

        let mut cpu = CpuMLstm::new(inp, d, heads, dqk);
        cpu.wi = Tensor::random_seeded(&[inp, heads], 0.3, 0xE1);
        cpu.wf = Tensor::random_seeded(&[inp, heads], 0.3, 0xE2);
        let mut dev = MLstm::from_cpu(&gpu, &cpu);
        dev.set_chunk(32);

        let x = Tensor::random_seeded(&[b, t, inp], 0.5, 0xE3);
        let g = Tensor::random_seeded(&[b, t, d], 1.0, 0xE4);

        let y_cpu = cpu.forward(&x);
        let xd = GTensor::from_host(&gpu, &x);
        let mut yt = GTensor::uninit(&gpu, &[b, t, d]);
        dev.forward(&gpu, &xd, &mut yt, &tc);
        assert_close_rel(&yt.to_host(&gpu).data, &y_cpu.data, 1e-2, "y");

        let dx_cpu = cpu.backward(&g);
        let dx = dev.backward_alloc(&gpu, &xd, &GTensor::from_host(&gpu, &g), &tc);
        assert_close_rel(&dx.to_host(&gpu).data, &dx_cpu.data, 1e-2, "dx");
    }

    /// The fused kernels at a head width wide enough to TILE, against the CPU scalar
    /// recurrence.
    #[test]
    fn mlstm_fused_tiled_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        let (b, t, inp, d, heads, dqk) = (2, 70, 12, 768, 8, 96);

        let mut cpu = CpuMLstm::new(inp, d, heads, dqk);
        cpu.wi = Tensor::random_seeded(&[inp, heads], 0.3, 0xD1);
        cpu.wf = Tensor::random_seeded(&[inp, heads], 0.3, 0xD2);
        let mut dev = MLstm::from_cpu(&gpu, &cpu);

        let x = Tensor::random_seeded(&[b, t, inp], 0.5, 0xD3);
        let g = Tensor::random_seeded(&[b, t, d], 1.0, 0xD4);

        let y_cpu = cpu.forward(&x);
        let xd = GTensor::from_host(&gpu, &x);
        let mut y_dev = GTensor::uninit(&gpu, &[b, t, d]);
        dev.forward(&gpu, &xd, &mut y_dev, &tc);
        assert_close_rel(&y_dev.to_host(&gpu).data, &y_cpu.data, 1e-2, "y");

        let dx_cpu = cpu.backward(&g);
        let dx_dev = dev.backward_alloc(&gpu, &xd, &GTensor::from_host(&gpu, &g), &tc);
        assert_close_rel(&dx_dev.to_host(&gpu).data, &dx_cpu.data, 1e-2, "dx");
    }

    /// The chunked path vs the CPU scalar recurrence — the same check as
    /// `mlstm_matches_cpu`, but at a T long enough to span several chunks, so the
    /// state carry is validated against the recurrence it is supposed to reproduce
    /// (not just against the GPU's own single-chunk form).
    #[test]
    fn mlstm_chunked_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        let (b, t, inp, d, heads, dqk) = (2, 20, 5, 8, 2, 4);

        let mut cpu = CpuMLstm::new(inp, d, heads, dqk);
        cpu.wi = Tensor::random(&[inp, heads], 0.3);
        cpu.wf = Tensor::random(&[inp, heads], 0.3);
        let mut dev = MLstm::from_cpu(&gpu, &cpu);
        dev.set_chunk(6); // 6 + 6 + 6 + 2

        let x = Tensor::random(&[b, t, inp], 0.5);
        let g = Tensor::random(&[b, t, d], 1.0);

        let y_cpu = cpu.forward(&x);
        let xd = GTensor::from_host(&gpu, &x);
        let mut y_dev = GTensor::uninit(&gpu, &[b, t, d]);
        dev.forward(&gpu, &xd, &mut y_dev, &tc);
        assert_close(&y_dev.to_host(&gpu).data, &y_cpu.data, 3e-3, "y");

        let dx_cpu = cpu.backward(&g);
        let dx_dev = dev.backward_alloc(&gpu, &xd, &GTensor::from_host(&gpu, &g), &tc);
        assert_close(&dx_dev.to_host(&gpu).data, &dx_cpu.data, 3e-3, "dx");

        let mut cfg = AdamCfg::new(1e-3, 0.01);
        cfg.t = 1;
        cpu.step(&cfg);
        dev.step(&gpu, &cfg);
        let xw = dev.lin_qkvo.w.to_host(&gpu).data;
        assert_close(
            &cols_of(&xw, dev.lin_qkvo.output_size(), 0, heads * dqk),
            &cpu.wq.data,
            3e-3,
            "wq",
        );
        let gw = dev.lin_gates.w.to_host(&gpu).data;
        assert_close(
            &cols_of(&gw, 2 * heads, heads, heads),
            &cpu.wf.data,
            3e-3,
            "wf",
        );
        assert_close(
            &dev.lin_out.w.to_host(&gpu).data,
            &cpu.w_out.data,
            3e-3,
            "w_out",
        );
    }

    /// The fused forward, run in chunks with the state carried, must reproduce the
    /// single whole-sequence call.
    #[test]
    fn mlstm_chunked_carry_matches_whole() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let (bh, dqk, dhv, l) = (4, 64, 64, 32);
        let parts = [64usize, 64, 32]; // ragged: the last chunk is short
        let t: usize = parts.iter().sum();

        let mk = |dims: &[usize], seed: u64| {
            GTensor::from_host(&gpu, &Tensor::random_seeded(dims, 0.5, seed))
        };
        let q = mk(&[bh, t, dqk], 0xC1);
        let k = mk(&[bh, t, dqk], 0xC2);
        let v = mk(&[bh, t, dhv], 0xC3);
        let ig = mk(&[bh, t], 0xC4);
        let fg = mk(&[bh, t], 0xC5);
        // `ytil` is slab-typed; widen it to fp32 to compare.
        let ytil_host = |f: &ops::MlstmFused, n: usize| {
            let mut scratch = GTensor::uninit(&gpu, &[n]);
            f.ytil.as_f32(&gpu, &mut scratch).to_host(&gpu).data
        };

        // Reference: one call over the whole sequence.
        let whole = ops::mlstm_fused_fw(
            &gpu,
            &pack_qkv(&gpu, &q, &k, &v, bh * t, dqk, dhv),
            &pack_gates(&gpu, &ig, &fg, bh * t),
            l,
            None,
            ops::MlstmShape {
                batch: bh,
                heads: 1,
                t,
                dqk,
                dhv,
            },
        );
        let want = ytil_host(&whole, bh * t * dhv);

        // Chunked: slice the time axis, carrying the state across the borders.
        let mut got: Vec<f32> = Vec::with_capacity(bh * t * dhv);
        let mut state: Option<ops::MlstmState> = None;
        let mut off = 0;
        let mut per_chunk: Vec<Vec<f32>> = Vec::new();
        for &c in &parts {
            let cut = |src: &GTensor<f32>, w: usize| {
                let h = src.to_host(&gpu);
                let mut out = Vec::with_capacity(bh * c * w);
                for b in 0..bh {
                    let base = b * t * w + off * w;
                    out.extend_from_slice(&h.data[base..base + c * w]);
                }
                GTensor::from_host(&gpu, &Tensor::new(&[bh, c, w], out))
            };
            let (qc, kc, vc) = (cut(&q, dqk), cut(&k, dqk), cut(&v, dhv));
            let (igc, fgc) = (cut(&ig, 1), cut(&fg, 1));
            let f = ops::mlstm_fused_fw(
                &gpu,
                &pack_qkv(&gpu, &qc, &kc, &vc, bh * c, dqk, dhv),
                &pack_gates(&gpu, &igc, &fgc, bh * c),
                l,
                state.as_ref(),
                ops::MlstmShape {
                    batch: bh,
                    heads: 1,
                    t: c,
                    dqk,
                    dhv,
                },
            );
            per_chunk.push(ytil_host(&f, bh * c * dhv));
            state = Some(f.final_state(&gpu, bh, dhv, dqk));
            off += c;
        }
        // Re-interleave: each chunk holds `[bh, c, dhv]`, the reference `[bh, t, dhv]`.
        for b in 0..bh {
            let mut o = 0;
            for (ci, &c) in parts.iter().enumerate() {
                let base = b * c * dhv;
                got.extend_from_slice(&per_chunk[ci][base..base + c * dhv]);
                o += c;
            }
            let _ = o;
        }

        assert_eq!(got.len(), want.len(), "chunked output length");
        assert_close(&got, &want, 3e-2, "chunked ytil vs whole");
    }

    /// The chunked **backward** must reproduce the whole-sequence one.
    #[test]
    fn mlstm_chunked_backward_matches_whole() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        let (inp, d, heads, dqk, b, t) = (8usize, 16usize, 2usize, 8usize, 1usize, 12usize);
        let mut proto = CpuMLstm::new(inp, d, heads, dqk);
        proto.wi = Tensor::random_seeded(&[inp, heads], 0.3, 0xE1);
        proto.wf = Tensor::random_seeded(&[inp, heads], 0.3, 0xE2);
        proto.wq = Tensor::random_seeded(&[inp, d], 0.3, 0xE5);
        proto.wk = Tensor::random_seeded(&[inp, d], 0.3, 0xE6);
        proto.wv = Tensor::random_seeded(&[inp, d], 0.3, 0xE7);
        proto.wo = Tensor::random_seeded(&[inp, d], 0.3, 0xE8);
        proto.w_out = Tensor::random_seeded(&[d, d], 0.3, 0xE9);
        let x = Tensor::random_seeded(&[b, t, inp], 0.5, 0xE3);
        let g = Tensor::random_seeded(&[b, t, d], 1.0, 0xE4);
        let cut = |src: &Tensor, c0: usize, len: usize, w: usize| {
            let mut o = Vec::new();
            for bb in 0..b {
                let base = bb * t * w + c0 * w;
                o.extend_from_slice(&src.data[base..base + len * w]);
            }
            GTensor::from_host(&gpu, &Tensor::new(&[b, len, w], o))
        };

        let mut whole = MLstm::from_cpu(&gpu, &proto);
        let xd = GTensor::from_host(&gpu, &x);
        let mut y_whole = GTensor::uninit(&gpu, &[b, t, d]);
        whole.forward(&gpu, &xd, &mut y_whole, &tc);
        let want = whole
            .backward_alloc(&gpu, &xd, &GTensor::from_host(&gpu, &g), &tc)
            .to_host(&gpu)
            .data
            .to_vec();

        let err = |got: &[f32]| -> f32 {
            let scale = want
                .iter()
                .chain(got.iter())
                .fold(0.0f32, |m, v| m.max(v.abs()))
                .max(1e-12);
            want.iter()
                .zip(got)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
                / scale
        };

        // One border, then two. The bug this pins made the error grow with the count.
        for parts in [vec![(0usize, 6usize), (6, 6)], vec![(0, 4), (4, 4), (8, 4)]] {
            let mut part = MLstm::from_cpu(&gpu, &proto);
            part.set_carry(true);
            part.reset_state(&gpu);
            let xs: Vec<GTensor<f32>> = parts
                .iter()
                .map(|&(c0, len)| cut(&x, c0, len, inp))
                .collect();
            for (i, &(_, len)) in parts.iter().enumerate() {
                let mut y_part = GTensor::uninit(&gpu, &[b, len, d]);
                part.forward(&gpu, &xs[i], &mut y_part, &tc);
            }
            // Backward unwinds right to left, starting with no gradient from the right.
            part.reset_bptt(&gpu);
            let mut pieces: Vec<Vec<f32>> = vec![Vec::new(); parts.len()];
            for (i, &(c0, len)) in parts.iter().enumerate().rev() {
                pieces[i] = part
                    .backward_alloc(&gpu, &xs[i], &cut(&g, c0, len, d), &tc)
                    .to_host(&gpu)
                    .data
                    .to_vec();
            }
            let got: Vec<f32> = pieces.concat();
            let e = err(&got);
            assert!(
                e < 1e-2,
                "chunked backward dx differs by {e} over {} chunks",
                parts.len()
            );
        }
    }

    /// A reset row inside one call must split it into two independent sequences: the
    /// outputs, the input gradient and the summed weight gradients equal those of the
    /// two halves run on their own. The reset lands mid-chunk, with chunks on both
    /// sides of it.
    #[test]
    fn document_reset_matches_separate_sequences() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        reset_matches_split(&gpu, 1.0);
    }

    /// The same where the document before the reset drives the stabilizer far above
    /// the next row's input gate — static heads (`f̃` large, so `m` only ratchets up)
    #[test]
    fn document_reset_survives_a_saturated_stabilizer() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        reset_matches_split(&gpu, 150.0);
    }

    fn reset_matches_split(gpu: &Gpu, prefix_gain: f32) {
        use cudarc::driver::DevicePtr;
        let tc = test_cache(gpu);
        let (inp, d, heads, dqk, t, r) = (8usize, 16usize, 2usize, 8usize, 70usize, 29usize);
        // Every tensor seeded, so the three cells below are one instance.
        let make = || {
            let mut c = CpuMLstm::new(inp, d, heads, dqk);
            c.wi = Tensor::random_seeded(&[inp, heads], 0.3, 0xD1);
            c.wf = Tensor::random_seeded(&[inp, heads], 0.3, 0xD2);
            c.wq = Tensor::random_seeded(&[inp, d], 0.3, 0xD5);
            c.wk = Tensor::random_seeded(&[inp, d], 0.3, 0xD6);
            c.wv = Tensor::random_seeded(&[inp, d], 0.3, 0xD7);
            c.wo = Tensor::random_seeded(&[inp, d], 0.3, 0xD8);
            c.w_out = Tensor::random_seeded(&[d, d], 0.3, 0xD9);
            c.bi = Tensor::random_seeded(&[heads], 0.1, 0xDA);
            c.bf = Tensor::new(
                &[heads],
                vec![if prefix_gain > 1.0 { 8.0 } else { 3.0 }; heads],
            );
            c
        };
        let mut x = Tensor::random_seeded(&[1, t, inp], 0.5, 0xD3);
        for v in &mut x.data[..r * inp] {
            *v *= prefix_gain;
        }
        let dy = Tensor::random_seeded(&[1, t, d], 1.0, 0xD4);
        let cut = |src: &Tensor, off: usize, len: usize, w: usize| {
            Tensor::new(&[1, len, w], src.data[off * w..(off + len) * w].to_vec())
        };
        let gpu_run = |x: &Tensor, dy: &Tensor, resets: Option<Resets>| -> (Vec<f32>, Vec<f32>) {
            let mut c = MLstm::from_cpu(gpu, &make());
            c.set_chunk(8);
            if let Some(r) = resets {
                c.set_resets(&r);
            }
            let len = x.dims()[1];
            let xd = GTensor::from_host(gpu, x);
            let mut y = GTensor::uninit(gpu, &[1, len, d]);
            c.forward(gpu, &xd, &mut y, &tc);
            let dx = c.backward_alloc(gpu, &xd, &GTensor::from_host(gpu, dy), &tc);
            (y.to_host(gpu).data.to_vec(), dx.to_host(gpu).data.to_vec())
        };

        let mask: Vec<i32> = (0..t).map(|i| i32::from(i == r)).collect();
        let mask = gpu.stream.clone_htod(&mask).expect("mask");
        let resets = Resets {
            rows: vec![r],
            mask: mask.device_ptr(&gpu.stream).0,
        };
        let (y_whole, dx_whole) = gpu_run(&x, &dy, Some(resets));
        let mut halves = Vec::new();
        let mut cpu = Vec::new();
        for (off, len) in [(0, r), (r, t - r)] {
            let (xs, dys) = (cut(&x, off, len, inp), cut(&dy, off, len, d));
            halves.push(gpu_run(&xs, &dys, None));
            let mut c = make();
            let y = c.forward(&xs);
            let dx = c.backward(&dys);
            cpu.push((y.data.to_vec(), dx.data.to_vec()));
        }

        // Scale-aware worst error of rows `rows` of two `[t, w]` host tensors.
        let err = |a: &[f32], b: &[f32], w: usize, rows: std::ops::Range<usize>| -> f32 {
            let (a, b) = (
                &a[rows.start * w..rows.end * w],
                &b[rows.start * w..rows.end * w],
            );
            let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-12);
            a.iter()
                .zip(b)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0f32, f32::max)
                / scale
        };
        for (what, w, whole, half, reference) in [
            (
                "y",
                d,
                &y_whole,
                [&halves[0].0, &halves[1].0],
                [&cpu[0].0, &cpu[1].0],
            ),
            (
                "dx",
                inp,
                &dx_whole,
                [&halves[0].1, &halves[1].1],
                [&cpu[0].1, &cpu[1].1],
            ),
        ] {
            let e = err(whole, half[0], w, 0..r);
            assert!(
                e < 1e-5,
                "{what}: prefix rows moved by {e} with the next document attached"
            );
            let suffix = |v: &Vec<f32>| v[r * w..].to_vec();
            let (e_whole, e_half) = (
                err(&suffix(whole), reference[1], w, 0..t - r),
                err(half[1], reference[1], w, 0..t - r),
            );
            assert!(
                e_whole < (3.0 * e_half).max(5e-3),
                "{what}: the document after the border is {e_whole} from fp32, \
                 against {e_half} for the same document run on its own"
            );
        }
    }
}
