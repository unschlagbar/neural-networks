//! Device-resident xLSTM-style residual block, the GPU counterpart of
//! [`nn2::block::Block`](crate::nn2::block::Block).
//!
//!   z = x + cell(pre_norm1(x))
//!   y = z + lin_down( SiLU(lin_gate·pre_norm2(z)) ⊙ (lin_value·pre_norm2(z)) )
//!
//! The cell owns its own post-norm, so its output enters the residual normalized.
//! Everything but the cell is position-wise and runs on an `[N, H]` view of the
//! `[B, T, H]` sequence (`N = B·T`), which is metadata-only — the storage is shared.

use super::{
    GTensor, Gpu,
    arena::{self, ParamSlot},
    linear::Linear,
    mlstm::MLstm,
    offload::{Frame, FrameStack, SharedOffload},
    ops,
    rms_norm::RmsNorm,
    slstm::{SLstm, SlstmSaved},
};
use crate::{
    gpu::arena::TrainingCache,
    nn::{linear::LinearLayer, rms_norm::RMSNorm, slstm_block::SLSTMBlock},
    nn_layer::NnLayer,
    nn2::{self, optim::AdamCfg},
    tensor::Tensor,
};

/// Per-phase timing, off unless `GPU_PHASE=1`. It synchronizes around each phase, so
/// its numbers are for attribution only — never read a step time off a run with it on.
pub mod phase {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    /// Nanoseconds accumulated per bucket. Index with [`Bucket`].
    static NS: [AtomicU64; Bucket::COUNT] = [const { AtomicU64::new(0) }; Bucket::COUNT];

    #[derive(Clone, Copy)]
    pub enum Bucket {
        SlstmCellFwd = 0,
        SlstmCellBwd = 1,
        MlstmCellFwd = 2,
        MlstmCellBwd = 3,
        FfnFwd = 4,
        FfnBwd = 5,
        /// sLSTM sub-phases: `Copy` stages `x`/`dy`/`out` into the cell's buffers,
        /// `Gemm` is every whole-sequence matmul, `Loop` the serial T-loop.
        SlstmCopyFwd = 6,
        SlstmGemmFwd = 7,
        SlstmLoopFwd = 8,
        SlstmCopyBwd = 9,
        SlstmGemmBwd = 10,
        SlstmLoopBwd = 11,
    }

    impl Bucket {
        pub const COUNT: usize = 12;
        pub const ALL: [(Bucket, &'static str); Self::COUNT] = [
            (Bucket::SlstmCellFwd, "sLSTM cell"),
            (Bucket::SlstmCellBwd, "sLSTM cell"),
            (Bucket::MlstmCellFwd, "mLSTM cell"),
            (Bucket::MlstmCellBwd, "mLSTM cell"),
            (Bucket::FfnFwd, "SwiGLU FFN"),
            (Bucket::FfnBwd, "SwiGLU FFN"),
            (Bucket::SlstmCopyFwd, "sLSTM copies"),
            (Bucket::SlstmGemmFwd, "sLSTM gemm"),
            (Bucket::SlstmLoopFwd, "sLSTM T-loop"),
            (Bucket::SlstmCopyBwd, "sLSTM copies"),
            (Bucket::SlstmGemmBwd, "sLSTM gemm"),
            (Bucket::SlstmLoopBwd, "sLSTM T-loop"),
        ];
    }

    pub fn enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("GPU_PHASE").as_deref() == Ok("1"))
    }

    /// Whether accumulation is currently live — lets a caller skip warmup iters.
    static RECORDING: AtomicBool = AtomicBool::new(false);

    pub fn set_recording(on: bool) {
        RECORDING.store(on, Ordering::Relaxed);
    }

    pub fn reset() {
        for a in &NS {
            a.store(0, Ordering::Relaxed);
        }
    }

    pub fn add(b: Bucket, ns: u64) {
        if RECORDING.load(Ordering::Relaxed) {
            NS[b as usize].fetch_add(ns, Ordering::Relaxed);
        }
    }

    pub fn get(b: Bucket) -> u64 {
        NS[b as usize].load(Ordering::Relaxed)
    }

    /// Time `f`, attributing it to `b`. Synchronizes on both sides, so the measured
    /// span is real device time rather than submission time.
    pub fn timed<R>(gpu: &super::Gpu, b: Bucket, f: impl FnOnce() -> R) -> R {
        if !enabled() {
            return f();
        }
        gpu.stream.synchronize().expect("sync");
        let t0 = std::time::Instant::now();
        let r = f();
        gpu.stream.synchronize().expect("sync");
        add(b, t0.elapsed().as_nanos() as u64);
        r
    }
}

/// Rows of one `[1, T, ·]` call at which a packed document begins.
///
/// A cell restarts its recurrence at each of them: the row sees nothing of the rows
/// before it, and in backward no gradient crosses it towards them. Set for the call
/// about to run, forward and backward alike (see [`Cell::set_resets`]).
#[derive(Clone, Default)]
pub struct Resets {
    /// Ascending, local to the call. Never 0 of a sweep's first call: a sequence that
    /// starts fresh is the cell's own zero state.
    pub rows: Vec<usize>,
    /// Device address of an `i32[T]` that is 1 at `rows` and 0 elsewhere, for kernels
    /// that test each row inside their loop. 0 when `rows` is empty.
    pub mask: u64,
}

impl Resets {
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// A recurrent cell operating on `[B, T, H]` device sequences (H in == H out).
///
/// The cell saves nothing of its own: what its backward needs is carved from the
/// surrounding block's frame ([`carve`](Self::carve)) and handed to both directions.
pub trait Cell {
    /// What this cell's forward saves for its backward.
    type Saved;
    /// Whether the backward reads the cell's own output `y`. A cell that does not lets
    /// the block keep that output in scratch instead of in its frame.
    const NEEDS_Y: bool;
    /// This cell's saved tensors for a `[B, T, ·]` call, carved from `f`.
    fn carve(&self, gpu: &Gpu, f: &mut Frame, b: usize, t: usize) -> Self::Saved;
    /// `x` is the block's `norm1_out`, a slab that lives in the block's frame until
    /// the matching backward, which hands it back.
    fn forward_saved(
        &mut self,
        gpu: &Gpu,
        x: &ops::SlabBuf,
        out: &mut GTensor<f32>,
        sv: &mut Self::Saved,
        cache: &TrainingCache,
    );
    /// `x` is the forward input and `y` the cell's own output (`Some` exactly when
    /// [`NEEDS_Y`](Self::NEEDS_Y)), handed back so a cell that needs them for `dW` or
    /// `x̂` keeps no copy.
    #[allow(clippy::too_many_arguments)]
    fn backward_saved(
        &mut self,
        gpu: &Gpu,
        x: &ops::SlabBuf,
        y: Option<&GTensor<f32>>,
        dy: &GTensor<f32>,
        dx: &mut GTensor<f32>,
        sv: &mut Self::Saved,
        cache: &TrainingCache,
    );
    fn zero_grad(&mut self, gpu: &Gpu);
    /// Every parameter with its gradient and AdamW moments, in a fixed order.
    /// A model binds these into its [`ParamArena`](super::arena::ParamArena).
    fn param_slots(&mut self) -> Vec<ParamSlot<'_>>;
    /// AdamW over this cell alone, for a standalone cell and the parity tests.
    fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        arena::step_slots(gpu, &mut self.param_slots(), cfg);
    }
    /// Learnable tensors in a fixed order (checkpoint save/load).
    fn params_mut(&mut self) -> Vec<&mut GTensor<f32>> {
        self.param_slots().into_iter().map(|s| s.param).collect()
    }
    /// Gradient accumulators, matching `params_mut`'s order. Diagnostic.
    fn grads(&mut self) -> Vec<&GTensor<f32>> {
        self.param_slots().into_iter().map(|s| &*s.grad).collect()
    }
    /// Extremes of a saved forward cache, for cells that carry a stabilized
    /// normalizer. `None` when the cell has nothing of the sort. Diagnostic.
    fn state_extremes(&self, _gpu: &Gpu, _sv: &Self::Saved) -> Option<(f32, f32, f32)> {
        None
    }
    /// Which phase buckets this cell's forward/backward count toward, so a mixed
    /// stack can be attributed per cell kind. See [`phase`].
    fn phase_buckets(&self) -> (phase::Bucket, phase::Bucket);
    /// Drop pooled scratch far larger than a `rows`-row window needs.
    /// See [`Block::trim_to`].
    fn trim_to(&mut self, _rows: usize) {}
    /// Continue the previous call's recurrence instead of starting from zero, for a
    /// sweep chunked across calls. Default: ignore — no cross-call state to carry.
    fn set_carry(&mut self, _carry: bool) {}
    /// Zero the carried forward state (before the first chunk of a sweep).
    fn reset_state(&mut self, _gpu: &Gpu) {}
    /// Drop the caches an unwound sweep left behind while keeping the carried forward
    /// state — [`reset_state`](Self::reset_state) minus the zeroing, for a sweep that
    /// continues the previous one. Default: nothing cached, nothing carried.
    fn reset_caches(&mut self, _gpu: &Gpu) {}
    /// Zero the carried BPTT state (before the last chunk's backward — backward
    /// unwinds chunks right to left).
    fn reset_bptt(&mut self, _gpu: &Gpu) {}
    /// Where documents start inside the next call, forward or backward. Holds until
    /// replaced; a caller that never sets it runs every call as one sequence.
    fn set_resets(&mut self, r: &Resets);
    /// Device bytes this cell holds, split `(params, activations)`. Diagnostic.
    fn retained_bytes(&self) -> (usize, usize);
    /// Retained activation bytes split `(saved_cache, other)`. Diagnostic.
    fn act_split(&self) -> (usize, usize);
    /// Release every activation this cell holds (its projections and norms keep their
    /// own staging).
    fn drop_all_act(&mut self, gpu: &Gpu);
    /// Build the matching CPU `nn` block (`SLSTMBlock` / `MLSTMBlock`) from this
    /// cell plus the already-exported surrounding norms and projections.
    #[allow(clippy::too_many_arguments)]
    fn to_nn_block(
        &self,
        gpu: &Gpu,
        hidden: usize,
        up: usize,
        pre_norm1: RMSNorm,
        pre_norm2: RMSNorm,
        lin_gate: LinearLayer,
        lin_value: LinearLayer,
        lin_down: LinearLayer,
    ) -> Box<dyn NnLayer>;
}

impl Cell for SLstm {
    type Saved = SlstmSaved;
    /// The post-cell norm's backward rebuilds `x̂` from it.
    const NEEDS_Y: bool = true;
    fn carve(&self, gpu: &Gpu, f: &mut Frame, b: usize, t: usize) -> SlstmSaved {
        SLstm::carve(self, gpu, f, b, t)
    }
    fn forward_saved(
        &mut self,
        gpu: &Gpu,
        x: &ops::SlabBuf,
        out: &mut GTensor<f32>,
        sv: &mut SlstmSaved,
        cache: &TrainingCache,
    ) {
        SLstm::forward_saved(self, gpu, x, out, sv, cache)
    }
    fn backward_saved(
        &mut self,
        gpu: &Gpu,
        x: &ops::SlabBuf,
        y: Option<&GTensor<f32>>,
        dy: &GTensor<f32>,
        dx: &mut GTensor<f32>,
        sv: &mut SlstmSaved,
        cache: &TrainingCache,
    ) {
        let y = y.expect("sLSTM backward reads its output");
        SLstm::backward_saved(self, gpu, x, y, dy, dx, sv, cache)
    }
    fn zero_grad(&mut self, gpu: &Gpu) {
        SLstm::zero_grad(self, gpu)
    }
    fn param_slots(&mut self) -> Vec<ParamSlot<'_>> {
        SLstm::param_slots(self)
    }
    fn state_extremes(&self, gpu: &Gpu, sv: &SlstmSaved) -> Option<(f32, f32, f32)> {
        Some(SLstm::extremes_of(gpu, sv))
    }
    fn phase_buckets(&self) -> (phase::Bucket, phase::Bucket) {
        (phase::Bucket::SlstmCellFwd, phase::Bucket::SlstmCellBwd)
    }
    fn retained_bytes(&self) -> (usize, usize) {
        SLstm::retained_bytes(self)
    }
    fn drop_all_act(&mut self, _gpu: &Gpu) {
        SLstm::drop_all_act(self)
    }
    fn act_split(&self) -> (usize, usize) {
        SLstm::act_split(self)
    }
    fn set_carry(&mut self, carry: bool) {
        SLstm::set_carry(self, carry)
    }
    fn reset_state(&mut self, gpu: &Gpu) {
        SLstm::reset_state(self, gpu)
    }
    fn reset_caches(&mut self, gpu: &Gpu) {
        SLstm::reset_caches(self, gpu)
    }
    fn reset_bptt(&mut self, gpu: &Gpu) {
        SLstm::reset_bptt(self, gpu)
    }
    fn set_resets(&mut self, r: &Resets) {
        SLstm::set_resets(self, r)
    }
    fn to_nn_block(
        &self,
        gpu: &Gpu,
        hidden: usize,
        up: usize,
        pre_norm1: RMSNorm,
        pre_norm2: RMSNorm,
        lin_gate: LinearLayer,
        lin_value: LinearLayer,
        lin_down: LinearLayer,
    ) -> Box<dyn NnLayer> {
        // The CPU `SLSTMBlock` still keeps the post-cell norm at block level (it is
        // the checkpoint layout), so the cell hands its own γ back out here.
        let post = RMSNorm::from_loaded(hidden, super::dt_vec(gpu, self.post_norm_gamma()));
        Box::new(SLSTMBlock::from_loaded(
            hidden,
            up,
            pre_norm1,
            post,
            pre_norm2,
            self.to_nn_cell(gpu),
            lin_gate,
            lin_value,
            lin_down,
        ))
    }
}

/// Type-erased `Block`: `Block<C>` is generic over its cell, so an alternating
/// sLSTM/mLSTM stack needs this to be one `Vec`.
pub trait BlockLike {
    fn forward(
        &mut self,
        gpu: &Gpu,
        x: &GTensor<f32>,
        out: &mut GTensor<f32>,
        cache: &TrainingCache,
    );
    fn backward(
        &mut self,
        gpu: &Gpu,
        dy: &GTensor<f32>,
        dx: &mut GTensor<f32>,
        cache: &TrainingCache,
    );
    /// Forward into a fresh `[B, T, H]` (H in == H out). For one-shot call sites; a
    /// training loop passes its own buffer to [`forward`](Self::forward).
    fn forward_alloc(
        &mut self,
        gpu: &Gpu,
        x: &GTensor<f32>,
        cache: &TrainingCache,
    ) -> GTensor<f32> {
        let mut y = GTensor::uninit(gpu, x.dims());
        self.forward(gpu, x, &mut y, cache);
        y
    }
    /// Backward into a freshly allocated `dx`, shaped like `dy`.
    fn backward_alloc(
        &mut self,
        gpu: &Gpu,
        dy: &GTensor<f32>,
        cache: &TrainingCache,
    ) -> GTensor<f32> {
        let mut dx = GTensor::uninit(gpu, dy.dims());
        self.backward(gpu, dy, &mut dx, cache);
        dx
    }
    fn zero_grad(&mut self, gpu: &Gpu);
    /// Every parameter with its gradient and AdamW moments, in a fixed order.
    /// A model binds these into its [`ParamArena`](super::arena::ParamArena).
    fn param_slots(&mut self) -> Vec<ParamSlot<'_>>;
    /// AdamW over this block alone, for a standalone block and the parity tests.
    fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        arena::step_slots(gpu, &mut self.param_slots(), cfg);
    }
    /// Learnable tensors in a fixed order (checkpoint save/load).
    fn params_mut(&mut self) -> Vec<&mut GTensor<f32>> {
        self.param_slots().into_iter().map(|s| s.param).collect()
    }
    /// Gradient accumulators, matching `params_mut`'s order. Diagnostic.
    fn grads(&mut self) -> Vec<&GTensor<f32>> {
        self.param_slots().into_iter().map(|s| &*s.grad).collect()
    }
    /// The cell's forward-cache extremes of the last forward. See
    /// [`Cell::state_extremes`].
    fn state_extremes(&self, gpu: &Gpu) -> Option<(f32, f32, f32)>;
    /// Keep this block's frames in `off` rather than on the device. See
    /// [`Block::set_offload`].
    fn set_offload(&mut self, off: Option<SharedOffload>);
    /// Bytes of one frame for a `[B, T, H]` forward. See [`Block::frame_bytes`].
    fn frame_bytes(&self, gpu: &Gpu, b: usize, t: usize) -> usize;
    /// Release the saved forward activations without reading them, for a stack that
    /// re-forwards rather than unwinding. See [`Block::drop_saved_act`].
    fn drop_saved_act(&mut self);
    /// Drop pooled scratch far larger than a `rows`-row window needs, at a window
    /// boundary. See [`Block::trim_to`].
    fn trim_to(&mut self, rows: usize);
    /// Device bytes held, split `(params, activations)`. See
    /// [`Block::retained_bytes`].
    fn retained_bytes(&self) -> (usize, usize);
    /// Release every activation, including what `drop_saved_act` keeps. See
    /// [`Block::drop_all_act`].
    fn drop_all_act(&mut self, gpu: &Gpu);
    /// Retained activation bytes by owner. See [`Block::act_breakdown`].
    fn act_breakdown(&self) -> [usize; 4];

    /// Carry the cell's recurrence across calls, for a chunked sweep.
    fn set_carry(&mut self, carry: bool);
    /// Zero the carried forward state (before a sweep's first chunk).
    fn reset_state(&mut self, gpu: &Gpu);
    /// Drop the block's caches but keep the carried forward state, for a sweep that
    /// continues the previous one. See [`Cell::reset_caches`].
    fn reset_caches(&mut self, gpu: &Gpu);
    /// Zero the carried BPTT state (before a sweep's last chunk backward).
    fn reset_bptt(&mut self, gpu: &Gpu);
    /// Where documents start inside the next call. See [`Cell::set_resets`].
    fn set_resets(&mut self, r: &Resets);
    /// The cell's `(saved_cache, other)` activation split. See [`Cell::act_split`].
    fn cell_act_split(&self) -> (usize, usize);
    /// Export the block into the matching CPU `nn` block (`SLSTMBlock` /
    /// `MLSTMBlock`) for a `HIER` checkpoint.
    fn to_nn_layer(&mut self, gpu: &Gpu) -> Box<dyn NnLayer>;
}

impl<C: Cell> BlockLike for Block<C> {
    fn forward(
        &mut self,
        gpu: &Gpu,
        x: &GTensor<f32>,
        out: &mut GTensor<f32>,
        cache: &TrainingCache,
    ) {
        Block::forward(self, gpu, x, out, cache)
    }
    fn backward(
        &mut self,
        gpu: &Gpu,
        dy: &GTensor<f32>,
        dx: &mut GTensor<f32>,
        cache: &TrainingCache,
    ) {
        Block::backward(self, gpu, dy, dx, cache)
    }
    fn zero_grad(&mut self, gpu: &Gpu) {
        Block::zero_grad(self, gpu)
    }
    fn param_slots(&mut self) -> Vec<ParamSlot<'_>> {
        Block::param_slots(self)
    }
    fn state_extremes(&self, gpu: &Gpu) -> Option<(f32, f32, f32)> {
        Block::state_extremes(self, gpu)
    }
    fn set_offload(&mut self, off: Option<SharedOffload>) {
        Block::set_offload(self, off)
    }
    fn frame_bytes(&self, gpu: &Gpu, b: usize, t: usize) -> usize {
        Block::frame_bytes(self, gpu, b, t)
    }
    fn drop_saved_act(&mut self) {
        Block::drop_saved_act(self)
    }
    fn trim_to(&mut self, rows: usize) {
        Block::trim_to(self, rows)
    }
    fn retained_bytes(&self) -> (usize, usize) {
        Block::retained_bytes(self)
    }
    fn drop_all_act(&mut self, gpu: &Gpu) {
        Block::drop_all_act(self, gpu)
    }
    fn act_breakdown(&self) -> [usize; 4] {
        Block::act_breakdown(self)
    }

    fn set_carry(&mut self, carry: bool) {
        self.carry = carry;
        self.cell.set_carry(carry)
    }
    fn reset_state(&mut self, gpu: &Gpu) {
        // A sweep that forwarded chunks and never unwound them would otherwise leave
        // its frames to accumulate across steps.
        self.frames.reset();
        self.cell.reset_state(gpu)
    }
    fn reset_caches(&mut self, gpu: &Gpu) {
        self.frames.reset();
        self.cell.reset_caches(gpu)
    }
    fn reset_bptt(&mut self, gpu: &Gpu) {
        self.cell.reset_bptt(gpu)
    }
    fn set_resets(&mut self, r: &Resets) {
        self.cell.set_resets(r)
    }
    fn cell_act_split(&self) -> (usize, usize) {
        self.cell.act_split()
    }
    fn to_nn_layer(&mut self, gpu: &Gpu) -> Box<dyn NnLayer> {
        Block::to_nn_layer(self, gpu)
    }
}

pub struct Block<C: Cell> {
    pub hidden: usize,
    pub up: usize,
    pub pre_norm1: RmsNorm,
    pub cell: C,
    pub pre_norm2: RmsNorm,
    pub lin_gate: Linear,
    pub lin_value: Linear,
    pub lin_down: Linear,

    /// Where each forward's frame lives until its backward — see [`BlockSaved`].
    frames: FrameStack,
    /// Whether `norm2_out` and `mixed` are stored bf16. Both switches must allow it:
    /// either one off means these values have an fp32 reader (the norm kernels, or
    /// the GEMMs). Fixed for the block's life, because a forward and its backward have
    /// to agree on the layout of what was written.
    narrow: bool,
    /// Whether this block is part of a chunked sweep, i.e. whether its frame must
    /// survive the next chunk's forward. Set alongside the cell's own carry.
    carry: bool,
    /// `(B, T)` of the last forward, for [`state_extremes`](Self::state_extremes).
    last_shape: (usize, usize),
}

/// What a block's forward saves for its backward, carved from one frame.
///
/// Named after the forward, which computes them in this order:
///
/// ```text
///   norm1_out = pre_norm1(x)               the cell's input
///   cell_out  = cell(norm1_out)            already post-normed; kept if the cell asks
///   z         = x + cell_out               residual 1  (not kept)
///   norm2_out = pre_norm2(z)               both FFN projections' input
///   gate_pre  = lin_gate(norm2_out)        value = lin_value(norm2_out)
///   mixed     = SiLU(gate_pre) ⊙ value     lin_down's input
///   out       = z + lin_down(mixed)        residual 2
/// ```
///
/// Everything else (`z`, every `d_*`) comes from [`temp`](super::temp) instead, and
/// `SiLU(gate_pre)` is recomputed by the backward rather than saved.
struct BlockSaved<S> {
    norm1_out: ops::SlabBuf,        // [B, T, H], at the width the cell's GEMMs read
    cell_out: Option<GTensor<f32>>, // [B, T, H], only when `C::NEEDS_Y`
    gate_pre: GTensor<f32>,         // [N, U]
    value: GTensor<f32>,            // [N, U]
    norm2_out: ops::SlabBuf,        // [N, H], kept once for both projections
    mixed: ops::SlabBuf,            // [N, U]
    norm1: ops::GpuRmsForward,
    norm2: ops::GpuRmsForward,
    cell: S,
}

impl<C: Cell> Block<C> {
    fn assemble(
        gpu: &Gpu,
        hidden: usize,
        up: usize,
        pre_norm1: RmsNorm,
        cell: C,
        pre_norm2: RmsNorm,
        [lin_gate, lin_value, lin_down]: [Linear; 3],
    ) -> Self {
        Self {
            hidden,
            up,
            pre_norm1,
            cell,
            pre_norm2,
            lin_gate,
            lin_value,
            lin_down,
            frames: FrameStack::default(),
            narrow: gpu.kernels.slab_bf16 && ops::gemm_bf16_enabled(gpu),
            carry: false,
            last_shape: (0, 0),
        }
    }

    /// Keep this block's frames in the shared `off` instead of on the device, or back
    /// on the device with `None`.
    ///
    /// **Only legal where the whole stack forwards before any of it unwinds**, and
    /// only for a stack whose blocks all push to the same `off` in sweep order — the
    /// frames form one LIFO across the stack. See [`Offload`](super::offload::Offload).
    pub fn set_offload(&mut self, off: Option<SharedOffload>) {
        self.frames = match off {
            Some(o) => FrameStack::Offload(o),
            None => FrameStack::default(),
        };
    }

    /// Every tensor a `[B, T, H]` forward saves, carved from `f` — the block's own, its
    /// norms' and its cell's, in the one order forward and backward share.
    fn carve(&self, gpu: &Gpu, f: &mut Frame, b: usize, t: usize) -> BlockSaved<C::Saved> {
        let (h, u, n) = (self.hidden, self.up, b * t);
        BlockSaved {
            norm1_out: f.slab(gpu, &[b, t, h], self.narrow),
            cell_out: C::NEEDS_Y.then(|| f.f32(gpu, &[b, t, h])),
            gate_pre: f.f32(gpu, &[n, u]),
            value: f.f32(gpu, &[n, u]),
            norm2_out: f.slab(gpu, &[n, h], self.narrow),
            mixed: f.slab(gpu, &[n, u], self.narrow),
            norm1: self.pre_norm1.carve(gpu, f, n),
            norm2: self.pre_norm2.carve(gpu, f, n),
            cell: self.cell.carve(gpu, f, b, t),
        }
    }

    /// Bytes of one frame for a `[B, T, H]` forward.
    pub fn frame_bytes(&self, gpu: &Gpu, b: usize, t: usize) -> usize {
        let mut m = Frame::measure();
        self.carve(gpu, &mut m, b, t);
        m.bytes()
    }

    /// Drop pooled scratch far larger than a `rows`-row window needs. Call at a window
    /// boundary: [`Buf`](super::Buf) reuses by capacity, so otherwise every buffer
    /// ratchets to the largest window ever seen (10.4, 13.4, 15.9, 16.6 GB, then abort).
    pub fn trim_to(&mut self, rows: usize) {
        self.cell.trim_to(rows);
    }

    /// Release the saved activations unread, for a stack that re-forwards instead of
    /// unwinding: the encoder rebuilds each group's cache in backward.
    pub fn drop_saved_act(&mut self) {
        self.frames.release();
    }

    /// Device bytes this block holds, split `(params, activations)`. Diagnostic — see
    /// [`Hierarchical::retained_report`](super::hierarchical::Hierarchical::retained_report).
    pub fn retained_bytes(&self) -> (usize, usize) {
        let (mut params, mut act) = self.cell.retained_bytes();
        for n in [&self.pre_norm1, &self.pre_norm2] {
            let (p, a) = n.retained_bytes();
            params += p;
            act += a;
        }
        for l in [&self.lin_gate, &self.lin_value, &self.lin_down] {
            let (p, a) = l.retained_bytes();
            params += p;
            act += a;
        }
        act += self.frames.device_bytes();
        (params, act)
    }

    /// Retained activation bytes by owner: `(frames, norms, projections, cell)`.
    /// Only the first is reachable from `drop_saved_act` + `trim_to`.
    pub fn act_breakdown(&self) -> [usize; 4] {
        let norms: usize = [&self.pre_norm1, &self.pre_norm2]
            .iter()
            .map(|n| n.retained_bytes().1)
            .sum();
        let proj: usize = [&self.lin_gate, &self.lin_value, &self.lin_down]
            .iter()
            .map(|l| l.retained_bytes().1)
            .sum();
        [
            self.frames.device_bytes(),
            norms,
            proj,
            self.cell.retained_bytes().1,
        ]
    }

    /// Release every activation
    pub fn drop_all_act(&mut self, gpu: &Gpu) {
        self.drop_saved_act();
        // The pool is NOT emptied (see `MLstm::drop_all_act`): dropping scratch per
        // group puts the allocator on the hot path. `trim_to` sizes it per window.
        self.pre_norm1.drop_saved_act();
        self.pre_norm2.drop_saved_act();
        for l in [&mut self.lin_gate, &mut self.lin_value, &mut self.lin_down] {
            l.drop_saved_act(gpu);
        }
        self.cell.drop_all_act(gpu);
    }

    /// The cell's forward-cache extremes of the last forward, read back out of its
    /// frame. `None` for an offloaded block, whose frames are not on the device.
    pub fn state_extremes(&self, gpu: &Gpu) -> Option<(f32, f32, f32)> {
        let mut f = self.frames.recent(gpu)?;
        let (b, t) = self.last_shape;
        let sv = self.carve(gpu, &mut f, b, t);
        self.cell.state_extremes(gpu, &sv.cell)
    }
}

impl<C: Cell> Block<C> {
    /// Assemble a block around a cell, with fresh norms (γ=1) and Xavier `Linear`
    /// weights. `hidden` is the model width, `up` the SwiGLU inner width.
    pub fn from_cell(gpu: &Gpu, hidden: usize, up: usize, cell: C) -> Self {
        Self::assemble(
            gpu,
            hidden,
            up,
            RmsNorm::new(gpu, hidden),
            cell,
            RmsNorm::new(gpu, hidden),
            [
                Linear::from_parts(gpu, &Tensor::xavier(hidden, up), &Tensor::zeros(&[up])),
                Linear::from_parts(gpu, &Tensor::xavier(hidden, up), &Tensor::zeros(&[up])),
                Linear::from_parts(gpu, &Tensor::xavier(up, hidden), &Tensor::zeros(&[hidden])),
            ],
        )
    }

    /// Assemble around a cell, with the surrounding norms/projections from a CPU block
    /// (the caller uploads the cell). Shared by the `from_cpu` constructors.
    fn from_cpu_parts<D: nn2::block::Cell>(gpu: &Gpu, cpu: &nn2::Block<D>, cell: C) -> Self {
        Self::assemble(
            gpu,
            cpu.hidden,
            cpu.up,
            RmsNorm::from_parts(gpu, &cpu.pre_norm1.gamma),
            cell,
            RmsNorm::from_parts(gpu, &cpu.pre_norm2.gamma),
            [
                Linear::from_parts(gpu, &cpu.lin_gate.w, &cpu.lin_gate.b),
                Linear::from_parts(gpu, &cpu.lin_value.w, &cpu.lin_value.b),
                Linear::from_parts(gpu, &cpu.lin_down.w, &cpu.lin_down.b),
            ],
        )
    }

    /// Forward over `[B, T, H]` → `out` `[B, T, H]`. They must not alias: `input` is
    /// read from the caller's buffer, so writing `out` over it corrupts the residual.
    pub fn forward(
        &mut self,
        gpu: &Gpu,
        input: &GTensor<f32>,
        out: &mut GTensor<f32>,
        cache: &TrainingCache,
    ) {
        assert_eq!(input.rank, 3, "Block::forward expects [B, T, H]");
        let (b, t, h) = (input.shape[0], input.shape[1], input.shape[2]);
        assert_eq!(h, self.hidden, "Block::forward — hidden mismatch");
        assert_eq!(out.dims(), input.dims(), "Block::forward — output shape");

        let n = b * t;
        self.last_shape = (b, t);
        let bytes = self.frame_bytes(gpu, b, t);
        let mut f = self.frames.push(gpu, bytes, self.carry);
        let mut sv = self.carve(gpu, &mut f, b, t);

        // The input as [N, H]. A view, not a pooled copy: the storage stays the
        // caller's, so it must not outlive `input` nor go back to the pool.
        let x_flat = GTensor::view(gpu, &input.buf, 0, &[n, h]);

        // Residual 1: z = input + cell(pre_norm1(input)). Both intermediates are saved:
        // each is a norm's output, which is all that norm's backward needs.
        self.pre_norm1
            .forward_slab_saved(gpu, input, None, &mut sv.norm1_out, &mut sv.norm1);
        let mut cell_tmp = (!C::NEEDS_Y).then(|| cache.temps.get::<f32>(gpu, &[b, t, h]));
        let cell_out: &mut GTensor<f32> = match (&mut sv.cell_out, &mut cell_tmp) {
            (Some(saved), _) => saved,
            (None, Some(tmp)) => tmp,
            (None, None) => unreachable!("carve keeps cell_out exactly when NEEDS_Y"),
        };
        let (cf, _cb) = self.cell.phase_buckets();
        phase::timed(gpu, cf, || {
            self.cell
                .forward_saved(gpu, &sv.norm1_out, cell_out, &mut sv.cell, cache)
        });

        // Downstream is position-wise [N, H]. The residual sum `z` comes out of
        // pre_norm2's own pass over it.
        let cell_flat = GTensor::view(gpu, &cell_out.buf, 0, &[n, h]);
        let mut z = cache.temps.get::<f32>(gpu, &[n, h]);

        // Residual 2: out = z + SwiGLU(pre_norm2(z)). `norm2_out` is kept once here
        // because `lin_gate` and `lin_value` share it.
        self.pre_norm2.forward_slab_saved(
            gpu,
            &x_flat,
            Some((&cell_flat, &mut z)),
            &mut sv.norm2_out,
            &mut sv.norm2,
        );
        drop((cell_flat, cell_tmp));
        phase::timed(gpu, phase::Bucket::FfnFwd, || {
            self.lin_gate
                .forward_slab_lhs(gpu, &sv.norm2_out, &mut sv.gate_pre);
            self.lin_value
                .forward_slab_lhs(gpu, &sv.norm2_out, &mut sv.value);
            ops::swiglu_forward_slab(gpu, &sv.gate_pre, &sv.value, &mut sv.mixed);
            // `out = z + down(mixed)`: the residual rides in `lin_down`'s bias seed, so
            // there is no separate add and no buffer for `down`'s output.
            out.reshape_to(&[n, h]);
            self.lin_down
                .forward_slab_lhs_resid(gpu, &sv.mixed, &z, out);
        });
        out.reshape_to(&[b, t, h]);
        drop((z, sv));
        self.frames.pushed(gpu);
    }

    /// Backward over `[B, T, H]` → `dx` `[B, T, H]`.
    /// `dy` and `dx` must not alias, for the reason given on [`forward`](Self::forward).
    pub fn backward(
        &mut self,
        gpu: &Gpu,
        dy: &GTensor<f32>,
        dx: &mut GTensor<f32>,
        cache: &TrainingCache,
    ) {
        let (b, t) = (dy.shape[0], dy.shape[1]);
        let (h, u) = (self.hidden, self.up);
        let n = b * t;
        assert_eq!(dx.dims(), [b, t, h], "Block::backward — dx shape");
        let mut f = self.frames.pop(gpu);
        let mut sv = self.carve(gpu, &mut f, b, t);

        // The incoming delta as [N, H], the counterpart of `x_flat`: read by
        // lin_down's backward and again by the d_z residual, never written.
        let dy_flat = GTensor::view(gpu, &dy.buf, 0, &[n, h]);

        // Residual 2.
        let mut d_mixed = cache.temps.get::<f32>(gpu, &[n, u]);
        let ffn_t0 = phase::enabled().then(|| {
            gpu.stream.synchronize().expect("sync");
            std::time::Instant::now()
        });
        self.lin_down
            .backward_slab_x(gpu, &sv.mixed, &dy_flat, &mut d_mixed, 0.0, cache);
        // The deltas are written at the width `norm2_out` has: their only readers are
        // the two projections' GEMMs and bias sums, which take them narrow.
        let mut d_gate = ops::SlabAt::temp(gpu, &cache.temps, &[n, u], self.narrow);
        let mut d_value = ops::SlabAt::temp(gpu, &cache.temps, &[n, u], self.narrow);
        ops::swiglu_backward_into_slab(
            gpu,
            &d_mixed,
            &sv.value,
            &sv.gate_pre,
            &mut d_gate.x,
            &mut d_value.x,
        );
        drop(d_mixed);
        // Forward wrote `norm2_out` at the width both GEMMs want, so neither narrows here.
        // Both projections read `norm2_out`, so the value one accumulates its `dx` onto
        // the gate one's.
        let mut d_norm2_out = cache.temps.get::<f32>(gpu, &[n, h]);
        self.lin_gate
            .backward_slab_xy(gpu, &sv.norm2_out, &d_gate.x, &mut d_norm2_out, 0.0, cache);
        self.lin_value
            .backward_slab_xy(gpu, &sv.norm2_out, &d_value.x, &mut d_norm2_out, 1.0, cache);
        if let Some(t0) = ffn_t0 {
            gpu.stream.synchronize().expect("sync");
            phase::add(phase::Bucket::FfnBwd, t0.elapsed().as_nanos() as u64);
        }
        drop((d_gate, d_value));

        // z feeds pre_norm2 and the y = z + down residual, so the norm's dx and the
        // incoming dy sum into d_z, inside the norm's backward.
        // `norm2_out` is pre_norm2's own output — see `RmsNorm::backward`.
        let mut d_z = cache.temps.get::<f32>(gpu, &[n, h]);
        self.pre_norm2.backward_slab_saved(
            gpu,
            &d_norm2_out,
            &sv.norm2_out,
            &sv.norm2,
            Some(&dy_flat),
            &mut d_z,
            cache,
        );
        drop(d_norm2_out);

        // Residual 1. Neither cell writes its `dy`, so d_z goes in as a `[B, T, H]`
        // view — the dx residual below still needs it as `[N, H]`.
        let d_cell_out = GTensor::view(gpu, &d_z.buf, 0, &[b, t, h]);
        let mut d_cell_in = cache.temps.get::<f32>(gpu, &[b, t, h]);
        let (_cf, cb) = self.cell.phase_buckets();
        phase::timed(gpu, cb, || {
            self.cell.backward_saved(
                gpu,
                &sv.norm1_out,
                sv.cell_out.as_ref(),
                &d_cell_out,
                &mut d_cell_in,
                &mut sv.cell,
                cache,
            )
        });
        drop(d_cell_out);
        d_cell_in.reshape_to(&[n, h]);
        // x feeds pre_norm1 (cell path) and the z = x + cn residual.
        dx.reshape_to(&[n, h]);
        self.pre_norm1.backward_slab_saved(
            gpu,
            &d_cell_in,
            &sv.norm1_out,
            &sv.norm1,
            Some(&d_z),
            dx,
            cache,
        );
        dx.reshape_to(&[b, t, h]);
        drop((d_cell_in, d_z, sv));
        self.frames.popped(gpu);
    }

    /// Every parameter with its gradient and AdamW moments, in a fixed order.
    pub fn param_slots(&mut self) -> Vec<ParamSlot<'_>> {
        let mut v = Vec::new();
        v.extend(self.pre_norm1.param_slots());
        v.extend(self.cell.param_slots());
        v.extend(self.pre_norm2.param_slots());
        v.extend(self.lin_gate.param_slots());
        v.extend(self.lin_value.param_slots());
        v.extend(self.lin_down.param_slots());
        v
    }

    /// Export into the matching CPU `nn` block for a `HIER` checkpoint: downloads the
    /// norms/projections, then lets the cell assemble the concrete block.
    pub fn to_nn_layer(&mut self, gpu: &Gpu) -> Box<dyn crate::nn_layer::NnLayer> {
        use super::{dt_matrix, dt_vec};
        let (h, u) = (self.hidden, self.up);
        let pre1 = RMSNorm::from_loaded(h, dt_vec(gpu, &self.pre_norm1.gamma));
        let pre2 = RMSNorm::from_loaded(h, dt_vec(gpu, &self.pre_norm2.gamma));
        let gate = LinearLayer::from_loaded(
            h,
            u,
            dt_matrix(gpu, &self.lin_gate.w),
            dt_vec(gpu, &self.lin_gate.b),
        );
        let value = LinearLayer::from_loaded(
            h,
            u,
            dt_matrix(gpu, &self.lin_value.w),
            dt_vec(gpu, &self.lin_value.b),
        );
        let down = LinearLayer::from_loaded(
            u,
            h,
            dt_matrix(gpu, &self.lin_down.w),
            dt_vec(gpu, &self.lin_down.b),
        );
        self.cell
            .to_nn_block(gpu, h, u, pre1, pre2, gate, value, down)
    }

    pub fn zero_grad(&mut self, gpu: &Gpu) {
        self.pre_norm1.zero_grad(gpu);
        self.cell.zero_grad(gpu);
        self.pre_norm2.zero_grad(gpu);
        self.lin_gate.zero_grad(gpu);
        self.lin_value.zero_grad(gpu);
        self.lin_down.zero_grad(gpu);
    }
}

/// Upload an `nn::LinearLayer` to the device.
fn lin_from_nn(gpu: &Gpu, l: &LinearLayer) -> Linear {
    use super::{tensor_from_matrix as m, tensor_from_slice as v};
    Linear::from_parts(gpu, &m(&l.weights), &v(&l.biases))
}

impl Block<SLstm> {
    /// Upload a whole CPU sLSTM block (norms, SwiGLU projections and the cell).
    pub fn from_cpu(gpu: &Gpu, cpu: &crate::nn2::SLstmBlock) -> Self {
        // `nn2` still keeps the post-cell norm on the block; the GPU cell owns it.
        let post = cpu.post_cell_norm.as_ref().map(|n| &n.gamma);
        Self::from_cpu_parts(gpu, cpu, SLstm::from_cpu(gpu, &cpu.cell, post))
    }

    /// Import an `nn::SLSTMBlock` (from a `HIER` checkpoint) onto the device.
    pub fn from_nn_block(gpu: &Gpu, cpu: &crate::nn::slstm_block::SLSTMBlock) -> Self {
        use super::tensor_from_slice as v;
        Self::assemble(
            gpu,
            cpu.hidden_size,
            cpu.up_size,
            RmsNorm::from_parts(gpu, &v(&cpu.pre_norm1.gamma)),
            // The checkpoint keeps the post-cell norm on the block; the GPU cell owns
            // it, so its γ is handed down here.
            SLstm::from_nn_cell(gpu, &cpu.cell, Some(&v(&cpu.post_cell_norm.gamma))),
            RmsNorm::from_parts(gpu, &v(&cpu.pre_norm2.gamma)),
            [
                lin_from_nn(gpu, &cpu.lin_gate),
                lin_from_nn(gpu, &cpu.lin_value),
                lin_from_nn(gpu, &cpu.lin_down),
            ],
        )
    }
}

impl Block<MLstm> {
    /// Upload a whole CPU mLSTM block (norms, SwiGLU projections and the cell).
    pub fn from_cpu(gpu: &Gpu, cpu: &crate::nn2::MLstmBlock) -> Self {
        Self::from_cpu_parts(gpu, cpu, MLstm::from_cpu(gpu, &cpu.cell))
    }

    /// Import an `nn::MLSTMBlock` (from a `HIER` checkpoint) onto the device.
    pub fn from_nn_block(gpu: &Gpu, cpu: &crate::nn::mlstm_block::MLSTMBlock) -> Self {
        use super::tensor_from_slice as v;
        Self::assemble(
            gpu,
            cpu.hidden_size,
            cpu.up_size,
            RmsNorm::from_parts(gpu, &v(&cpu.pre_norm1.gamma)),
            MLstm::from_nn_cell(gpu, &cpu.cell),
            RmsNorm::from_parts(gpu, &v(&cpu.pre_norm2.gamma)),
            [
                lin_from_nn(gpu, &cpu.lin_gate),
                lin_from_nn(gpu, &cpu.lin_value),
                lin_from_nn(gpu, &cpu.lin_down),
            ],
        )
    }
}

#[cfg(test)]
mod tests {

    /// One temp cache per test, sized past every shape this module presents.
    fn test_cache(gpu: &Gpu) -> TrainingCache {
        TrainingCache::new(gpu, 1 << 20, 1 << 16, 1 << 20)
    }
    use super::*;
    use crate::gpu::offload::Offload;
    use crate::nn2::block::{MLstmBlock as CpuMLstmBlock, SLstmBlock as CpuSLstmBlock};
    use crate::nn2::optim::AdamCfg;
    use crate::tensor::Tensor;

    /// Absolute floor plus `rel * max|want|` — scaled by the whole tensor, **not** per
    /// element: `dx[i]` sums many bf16-perturbed terms, so a small element (one made
    /// small by cancellation) carries the same absolute error as a large one.
    fn assert_close_rel(got: &[f32], want: &[f32], abs: f32, rel: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length mismatch");
        let scale = want.iter().fold(0.0, |m: f32, w| m.max(w.abs()));
        let bound = abs + rel * scale;
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() < bound,
                "{what}[{i}]: gpu {g} vs cpu {w} (tolerance {bound:.2e}, scale {scale:.2e})"
            );
        }
    }

    #[allow(dead_code)]
    fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_close_rel(got, want, tol, 0.0, what);
    }

    /// Tolerance for a parameter compared AFTER an Adam step, wider than [`rel`]: the
    /// update divides by √v̂, so on a near-zero gradient a 1e-7 wobble lands as 1e-4.
    fn step_rel(gpu: &Gpu) -> f32 {
        if ops::gemm_bf16_enabled(gpu) || gpu.kernels.slab_bf16 {
            5e-2
        } else {
            0.0
        }
    }

    /// Tolerance against the all-fp32 CPU reference, 0 on the fp32 path. Three bf16
    /// sources chain here, so it is a small multiple of one quantization: the worst
    /// element over ten runs was 0.0110. Growth with T is pinned in `gpu::slstm`.
    fn rel(gpu: &Gpu) -> f32 {
        if ops::gemm_bf16_enabled(gpu) || gpu.kernels.slab_bf16 {
            2e-2
        } else {
            0.0
        }
    }

    /// GPU `Block<SLstm>` must match `nn2::SLstmBlock` for forward → backward →
    /// AdamW-step from identical parameters.
    #[test]
    fn slstm_block_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        let (b, t, h, u) = (2, 4, 8, 12);

        let mut cpu = CpuSLstmBlock::new_slstm(h, u);
        let mut dev = Block::<SLstm>::from_cpu(&gpu, &cpu);

        let x = Tensor::random(&[b, t, h], 0.5);
        let g = Tensor::random(&[b, t, h], 1.0);

        // Forward
        let y_cpu = cpu.forward(&x);
        let y_dev = dev.forward_alloc(&gpu, &GTensor::from_host(&gpu, &x), &tc);
        assert_close_rel(&y_dev.to_host(&gpu).data, &y_cpu.data, 3e-3, rel(&gpu), "y");

        // Backward
        let dx_cpu = cpu.backward(&g);
        let dx_dev = dev.backward_alloc(&gpu, &GTensor::from_host(&gpu, &g), &tc);
        assert_close_rel(
            &dx_dev.to_host(&gpu).data,
            &dx_cpu.data,
            3e-3,
            rel(&gpu),
            "dx",
        );

        // One AdamW step; compare a representative parameter from each path.
        let mut cfg = AdamCfg::new(1e-3, 0.01);
        cfg.t = 1;
        cpu.step(&cfg);
        dev.step(&gpu, &cfg);
        assert_close_rel(
            &dev.lin_down.w.to_host(&gpu).data,
            &cpu.lin_down.w.data,
            3e-3,
            step_rel(&gpu),
            "lin_down.w",
        );
        assert_close_rel(
            &dev.pre_norm1.gamma.to_host(&gpu).data,
            &cpu.pre_norm1.gamma.data,
            3e-3,
            step_rel(&gpu),
            "pre_norm1.gamma",
        );
        assert_close_rel(
            &dev.cell.gate_w(&gpu, 0),
            &cpu.cell.wz.data,
            3e-3,
            step_rel(&gpu),
            "cell.wz",
        );
    }

    /// GPU `Block<MLstm>` (parallel-form cell) must match `nn2::MLstmBlock` (scalar
    /// recurrence) for forward → backward → AdamW step from identical parameters.
    #[test]
    fn mlstm_block_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        let (b, t, h, u, heads, dqk) = (2, 5, 8, 12, 2, 4); // dhv = 4

        let mut cpu = CpuMLstmBlock::new_mlstm(h, u, heads, dqk);
        // Non-trivial gate weights so the decay/stabilizer path is exercised
        // (nn2::MLstm::new zero-inits wi/wf).
        cpu.cell.wi = Tensor::random(&[h, heads], 0.3);
        cpu.cell.wf = Tensor::random(&[h, heads], 0.3);
        let mut dev = Block::<MLstm>::from_cpu(&gpu, &cpu);

        let x = Tensor::random(&[b, t, h], 0.5);
        let g = Tensor::random(&[b, t, h], 1.0);

        let y_cpu = cpu.forward(&x);
        let y_dev = dev.forward_alloc(&gpu, &GTensor::from_host(&gpu, &x), &tc);
        assert_close_rel(&y_dev.to_host(&gpu).data, &y_cpu.data, 3e-3, rel(&gpu), "y");

        let dx_cpu = cpu.backward(&g);
        let dx_dev = dev.backward_alloc(&gpu, &GTensor::from_host(&gpu, &g), &tc);
        assert_close_rel(
            &dx_dev.to_host(&gpu).data,
            &dx_cpu.data,
            3e-3,
            rel(&gpu),
            "dx",
        );

        let mut cfg = AdamCfg::new(1e-3, 0.01);
        cfg.t = 1;
        cpu.step(&cfg);
        dev.step(&gpu, &cfg);
        assert_close_rel(
            &dev.lin_down.w.to_host(&gpu).data,
            &cpu.lin_down.w.data,
            3e-3,
            step_rel(&gpu),
            "lin_down.w",
        );
        assert_close_rel(
            &dev.pre_norm1.gamma.to_host(&gpu).data,
            &cpu.pre_norm1.gamma.data,
            3e-3,
            step_rel(&gpu),
            "pre_norm1.gamma",
        );
    }

    /// Offload moves bytes and reorders no arithmetic, so this is **exact** — a
    /// tolerance would hide the bugs it exists for (stale buffer, chunk out of order,
    /// missing event).
    #[test]
    fn offload_matches_resident_exactly() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        let (b, t, h, u) = (2, 5, 8, 12);

        // `run` builds an identical block from identical CPU weights, so the only
        // difference between the two calls is where the activations lived.
        fn run<C: Cell, F: Fn() -> Block<C>>(
            gpu: &Gpu,
            tc: &TrainingCache,
            build: F,
            x: &Tensor,
            g: &Tensor,
            offload: bool,
        ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
            let mut dev = build();
            if offload {
                dev.set_offload(Some(Offload::shared(gpu, 0)));
            }
            let y = dev.forward_alloc(gpu, &GTensor::from_host(gpu, x), tc);
            let dx = dev.backward_alloc(gpu, &GTensor::from_host(gpu, g), tc);
            // Gradients of the three FFN projections are the ones the offloaded frame
            // feeds, so they are the sharpest probe.
            let dw_down = dev.lin_down.dw.to_host(gpu).data;
            let dw_gate = dev.lin_gate.dw.to_host(gpu).data;
            (y.to_host(gpu).data, dx.to_host(gpu).data, dw_down, dw_gate)
        }

        let x = Tensor::random(&[b, t, h], 0.5);
        let g = Tensor::random(&[b, t, h], 1.0);

        // sLSTM cell.
        let cpu_s = CpuSLstmBlock::new_slstm(h, u);
        let build_s = || Block::<SLstm>::from_cpu(&gpu, &cpu_s);
        let resident = run(&gpu, &tc, build_s, &x, &g, false);
        let parked = run(&gpu, &tc, build_s, &x, &g, true);
        assert_eq!(parked.0, resident.0, "sLSTM: y differs under offload");
        assert_eq!(parked.1, resident.1, "sLSTM: dx differs under offload");
        assert_eq!(
            parked.2, resident.2,
            "sLSTM: lin_down.dw differs under offload"
        );
        assert_eq!(
            parked.3, resident.3,
            "sLSTM: lin_gate.dw differs under offload"
        );

        // mLSTM cell.
        let mut cpu_m = CpuMLstmBlock::new_mlstm(h, u, 2, 4);
        cpu_m.cell.wi = Tensor::random(&[h, 2], 0.3);
        cpu_m.cell.wf = Tensor::random(&[h, 2], 0.3);
        let build_m = || Block::<MLstm>::from_cpu(&gpu, &cpu_m);
        let resident = run(&gpu, &tc, build_m, &x, &g, false);
        let parked = run(&gpu, &tc, build_m, &x, &g, true);
        assert_eq!(parked.0, resident.0, "mLSTM: y differs under offload");
        assert_eq!(parked.1, resident.1, "mLSTM: dx differs under offload");
        assert_eq!(
            parked.2, resident.2,
            "mLSTM: lin_down.dw differs under offload"
        );
        assert_eq!(
            parked.3, resident.3,
            "mLSTM: lin_gate.dw differs under offload"
        );
    }

    /// Several cycles at changing shapes: one cycle misses a frame that leaks across
    /// steps — a slot reused at the wrong shape, or a copy racing the compute.
    #[test]
    fn offload_is_stable_across_steps_and_shapes() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        let (h, u) = (8, 12);
        let cpu = CpuSLstmBlock::new_slstm(h, u);
        let mut resident = Block::<SLstm>::from_cpu(&gpu, &cpu);
        let mut parked = Block::<SLstm>::from_cpu(&gpu, &cpu);
        parked.set_offload(Some(Offload::shared(&gpu, 0)));

        let mut cfg = AdamCfg::new(1e-3, 0.01);
        for step in 0..4 {
            let (b, t) = (1 + step % 2, 3 + step); // shapes vary per step
            let x = Tensor::random(&[b, t, h], 0.5);
            let g = Tensor::random(&[b, t, h], 1.0);
            let dx = GTensor::from_host(&gpu, &x);
            let dg = GTensor::from_host(&gpu, &g);

            let y_r = resident.forward_alloc(&gpu, &dx, &tc).to_host(&gpu).data;
            let y_p = parked.forward_alloc(&gpu, &dx, &tc).to_host(&gpu).data;
            assert_eq!(y_p, y_r, "step {step}: y diverged");

            let dxr = resident.backward_alloc(&gpu, &dg, &tc).to_host(&gpu).data;
            let dxp = parked.backward_alloc(&gpu, &dg, &tc).to_host(&gpu).data;
            assert_eq!(dxp, dxr, "step {step}: dx diverged");

            cfg.t += 1;
            resident.step(&gpu, &cfg);
            parked.step(&gpu, &cfg);
            assert_eq!(
                parked.lin_down.w.to_host(&gpu).data,
                resident.lin_down.w.to_host(&gpu).data,
                "step {step}: weights diverged after the optimizer step"
            );
        }
    }

    /// A **chunked** sLSTM sweep, bit-exact under offload: several frames on the stack
    /// at once, unwound in reverse.
    #[test]
    fn slstm_chunked_offload_matches_resident_exactly() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        let (h, u, b, t) = (8, 12, 1, 6);
        let cpu = CpuSLstmBlock::new_slstm(h, u);
        let mut resident = Block::<SLstm>::from_cpu(&gpu, &cpu);
        let mut parked = Block::<SLstm>::from_cpu(&gpu, &cpu);
        parked.set_offload(Some(Offload::shared(&gpu, 0)));

        // Three chunks of one sequence: carry on, all forwards, then all backwards.
        let xs: Vec<Tensor> = (0..3).map(|_| Tensor::random(&[b, t, h], 0.5)).collect();
        let gs: Vec<Tensor> = (0..3).map(|_| Tensor::random(&[b, t, h], 1.0)).collect();

        let run = |dev: &mut Block<SLstm>| {
            dev.set_carry(false);
            dev.reset_state(&gpu);
            dev.set_carry(true);
            let mut ys = Vec::new();
            for x in &xs {
                let dx = GTensor::from_host(&gpu, x);
                ys.push(dev.forward_alloc(&gpu, &dx, &tc).to_host(&gpu).data);
            }
            // Unwind right to left; the rightmost chunk starts with no incoming grad.
            dev.reset_bptt(&gpu);
            let mut dxs = Vec::new();
            for g in gs.iter().rev() {
                let dg = GTensor::from_host(&gpu, g);
                dxs.push(dev.backward_alloc(&gpu, &dg, &tc).to_host(&gpu).data);
            }
            (ys, dxs, dev.lin_down.dw.to_host(&gpu).data)
        };

        let r = run(&mut resident);
        let p = run(&mut parked);
        assert_eq!(p.0, r.0, "chunked sLSTM: y differs under offload");
        assert_eq!(p.1, r.1, "chunked sLSTM: dx differs under offload");
        assert_eq!(p.2, r.2, "chunked sLSTM: lin_down.dw differs under offload");
    }
}
