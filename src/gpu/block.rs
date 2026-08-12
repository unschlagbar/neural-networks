//! Device-resident xLSTM-style residual block, the GPU counterpart of
//! [`nn2::block::Block`](crate::nn2::block::Block).
//!
//!   z = x + cell(pre_norm1(x))
//!   y = z + lin_down( SiLU(lin_gate·pre_norm2(z)) ⊙ (lin_value·pre_norm2(z)) )
//!
//! The cell's output goes into the residual already normalized: each cell owns the
//! post-norm that suits it — sLSTM a plain row-wise RMSNorm, mLSTM its head-wise
//! `headnorm` — and the block neither holds one nor knows which shape applies.
//!
//! The norms and the SwiGLU MLP are position-wise and run on the flattened
//! `[N, H]` view (`N = B·T`); only the recurrent `cell` sees the `[B, T, H]`
//! sequence. Since a `DTensor` is contiguous row-major, the `[B,T,H] ↔ [N,H]`
//! reshapes are metadata-only (`DTensor::reshaped`), no copy. The block composes
//! the already-parity-tested `gpu::Linear` / `gpu::RmsNorm` sub-layers plus three
//! small elementwise kernels (`add`, `swiglu_forward`, `swiglu_backward`) around
//! a generic GPU `Cell`.

use super::{
    Buf, DTensor, Gpu, Pool, linear::Linear, mlstm::MLstm, offload, ops, rms_norm::RmsNorm,
    slstm::SLstm,
};
use crate::{
    nn::{linear::LinearLayer, rms_norm::RMSNorm, slstm_block::SLSTMBlock},
    nn_layer::NnLayer,
    nn2::optim::AdamCfg,
    tensor::Tensor,
};

/// Per-phase timing, off unless `GPU_PHASE=1`.
///
/// A block is a cell (the recurrence) wrapped in norms, residuals and a SwiGLU MLP,
/// and the whole point of a breakdown is to say which of those the time is in. The
/// GPU is asynchronous, so a phase can only be timed by synchronizing around it —
/// which perturbs the schedule and is exactly why this is opt-in rather than always
/// compiled in. Numbers taken with it on are for *attribution*, not for the headline
/// step time; take that from a run with it off.
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
        /// Norms, residual adds and the block-level buffer copies — everything in a
        /// block that is neither the cell nor the MLP.
        GlueFwd = 6,
        GlueBwd = 7,
        /// sLSTM sub-phases, so the 70%-of-a-step cell can be broken down further.
        /// `Copy` is the device-to-device staging of `x`/`dy`/`out` into the buffers
        /// the cell owns; `Gemm` is the whole-sequence input projection plus
        /// backward's dx/dW/db GEMMs; `Loop` is the serial T-loop itself.
        SlstmCopyFwd = 8,
        SlstmGemmFwd = 9,
        SlstmLoopFwd = 10,
        SlstmCopyBwd = 11,
        SlstmGemmBwd = 12,
        SlstmLoopBwd = 13,
    }

    impl Bucket {
        pub const COUNT: usize = 14;
        pub const ALL: [(Bucket, &'static str); Self::COUNT] = [
            (Bucket::SlstmCellFwd, "sLSTM cell"),
            (Bucket::SlstmCellBwd, "sLSTM cell"),
            (Bucket::MlstmCellFwd, "mLSTM cell"),
            (Bucket::MlstmCellBwd, "mLSTM cell"),
            (Bucket::FfnFwd, "SwiGLU FFN"),
            (Bucket::FfnBwd, "SwiGLU FFN"),
            (Bucket::GlueFwd, "norms/residual/copies"),
            (Bucket::GlueBwd, "norms/residual/copies"),
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

/// A recurrent cell operating on `[B, T, H]` device sequences (H in == H out).
pub trait Cell {
    fn forward(&mut self, gpu: &Gpu, x: &DTensor, out: &mut DTensor);
    fn backward(&mut self, gpu: &Gpu, dy: &DTensor, dx: &mut DTensor);
    fn zero_grad(&mut self, gpu: &Gpu);
    fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg);
    /// [`step`](Self::step), optionally queueing the AdamW updates into one batched
    /// launch instead of one per tensor. Default: ignore the queue and step eagerly,
    /// so a cell that has not been converted still works.
    fn step_q(&mut self, gpu: &Gpu, cfg: &AdamCfg, _q: Option<&mut ops::AdamwQueue>) {
        self.step(gpu, cfg);
    }
    /// Learnable tensors in a fixed order (checkpoint save/load).
    fn params_mut(&mut self) -> Vec<&mut DTensor>;
    /// Gradient accumulators, matching `params_mut`'s order. Diagnostic.
    fn grads(&self) -> Vec<&DTensor>;
    /// Forward-cache extremes, for cells that carry a stabilized normalizer.
    /// `None` when the cell has nothing of the sort. Diagnostic.
    fn state_extremes(&self, _gpu: &Gpu) -> Option<(f32, f32, f32)> {
        None
    }
    /// Which phase buckets this cell's forward/backward count toward, so a mixed
    /// stack can be attributed per cell kind. See [`phase`].
    fn phase_buckets(&self) -> (phase::Bucket, phase::Bucket);
    /// Park this cell's saved activations on the host between forward and backward.
    ///
    /// Default: do nothing, for a cell with no large per-`N` cache to move. The
    /// surrounding [`Block`] calls this from its own `enable_offload`, so a cell opts
    /// in wherever the block does — subject to the same whole-forward-then-backward
    /// constraint.
    fn enable_offload(&mut self, _gpu: &Gpu, _in_flight: offload::SharedInFlight) {}
    /// Start this cell's parked activations back to the device without waiting.
    /// Called one block ahead of its backward, so the upload overlaps compute.
    fn prefetch_act(&mut self, _gpu: &Gpu) {}
    /// Release this cell's forward cache without reading it, for a stack that
    /// re-forwards rather than unwinding. See [`Block::drop_saved_act`].
    fn drop_saved_act(&mut self) {}
    /// Drop pooled scratch far larger than a `rows`-row window needs.
    /// See [`Block::trim_to`].
    fn trim_to(&mut self, _rows: usize) {}
    /// Continue the previous call's recurrence instead of starting from zero.
    ///
    /// For a chunked sweep, where one sequence is split across several calls and the
    /// state has to cross the chunk borders. Default: ignore it — a cell with no
    /// cross-call state has nothing to carry.
    fn set_carry(&mut self, _carry: bool) {}
    /// Zero the carried forward state (before the first chunk of a sweep).
    fn reset_state(&mut self, _gpu: &Gpu) {}
    /// Zero the carried BPTT state (before the last chunk's backward — backward
    /// unwinds chunks right to left).
    fn reset_bptt(&mut self, _gpu: &Gpu) {}
    /// Device bytes this cell holds, split `(params, activations)`. Diagnostic.
    fn retained_bytes(&self) -> (usize, usize);
    /// Retained activation bytes split `(saved_cache, other)` — "other" being the
    /// cell's own stable buffers plus whatever its internal projections and norms
    /// hold, i.e. the part `drop_saved_act` does **not** reach. Diagnostic.
    fn act_split(&self) -> (usize, usize);
    /// Release every activation this cell holds, including the ones its
    /// `drop_saved_act` leaves behind (a cell's projections and norms keep their own).
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
    fn forward(&mut self, gpu: &Gpu, x: &DTensor, out: &mut DTensor) {
        SLstm::forward(self, gpu, x, out)
    }
    fn backward(&mut self, gpu: &Gpu, dy: &DTensor, dx: &mut DTensor) {
        SLstm::backward(self, gpu, dy, dx)
    }
    fn zero_grad(&mut self, gpu: &Gpu) {
        SLstm::zero_grad(self, gpu)
    }
    fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        SLstm::step(self, gpu, cfg)
    }
    fn step_q(&mut self, gpu: &Gpu, cfg: &AdamCfg, q: Option<&mut ops::AdamwQueue>) {
        SLstm::step_q(self, gpu, cfg, q)
    }
    fn params_mut(&mut self) -> Vec<&mut DTensor> {
        SLstm::params_mut(self)
    }
    fn grads(&self) -> Vec<&DTensor> {
        SLstm::grads(self)
    }
    fn state_extremes(&self, gpu: &Gpu) -> Option<(f32, f32, f32)> {
        SLstm::state_extremes(self, gpu)
    }
    fn phase_buckets(&self) -> (phase::Bucket, phase::Bucket) {
        (phase::Bucket::SlstmCellFwd, phase::Bucket::SlstmCellBwd)
    }
    fn drop_saved_act(&mut self) {
        SLstm::drop_saved_act(self)
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
    fn reset_bptt(&mut self, gpu: &Gpu) {
        SLstm::reset_bptt(self, gpu)
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

/// Type-erased `Block`, so a model can hold a heterogeneous stack (alternating
/// sLSTM / mLSTM blocks) as `Vec<Box<dyn BlockLike>>`. `Block<C>` is generic over
/// its cell, so the concrete types differ; this is the common interface.
pub trait BlockLike {
    fn forward(&mut self, gpu: &Gpu, x: &DTensor, out: &mut DTensor);
    fn backward(&mut self, gpu: &Gpu, dy: &DTensor, dx: &mut DTensor);
    /// Forward into a freshly allocated `[B, T, H]`. Blocks are H-in == H-out, so
    /// the shape follows the input. For benchmarks and one-shot call sites; a
    /// training loop passes its own buffer to [`forward`](Self::forward).
    fn forward_alloc(&mut self, gpu: &Gpu, x: &DTensor) -> DTensor {
        let mut y = DTensor::uninit(gpu, x.dims());
        self.forward(gpu, x, &mut y);
        y
    }
    /// Backward into a freshly allocated `dx`, shaped like `dy`.
    fn backward_alloc(&mut self, gpu: &Gpu, dy: &DTensor) -> DTensor {
        let mut dx = DTensor::uninit(gpu, dy.dims());
        self.backward(gpu, dy, &mut dx);
        dx
    }
    fn zero_grad(&mut self, gpu: &Gpu);
    fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg);
    /// [`step`](Self::step), optionally queueing the AdamW updates into one batched
    /// launch. Default: step eagerly, ignoring the queue.
    fn step_q(&mut self, gpu: &Gpu, cfg: &AdamCfg, _q: Option<&mut ops::AdamwQueue>) {
        self.step(gpu, cfg);
    }
    /// Learnable tensors in a fixed order (checkpoint save/load).
    fn params_mut(&mut self) -> Vec<&mut DTensor>;
    /// Gradient accumulators, matching `params_mut`'s order. Diagnostic.
    fn grads(&self) -> Vec<&DTensor>;
    /// The cell's forward-cache extremes. See [`Cell::state_extremes`].
    fn state_extremes(&self, gpu: &Gpu) -> Option<(f32, f32, f32)>;
    /// Park this block's FFN activations on the host between forward and backward.
    /// See [`Block::enable_offload`] — only valid for a whole-forward-then-backward
    /// stack, i.e. the backbone.
    fn enable_offload(&mut self, gpu: &Gpu, in_flight: offload::SharedInFlight);
    /// Start this block's parked activations on their way back to the device, without
    /// waiting. Call one block ahead of its backward so the upload overlaps compute;
    /// no-op when this block is not offloaded. See [`Block::prefetch_act`].
    fn prefetch_act(&mut self, gpu: &Gpu);
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
    fn act_breakdown(&self) -> [usize; 5];
    /// Pool free-list shape `(distinct sizes, buffers)`. See [`Block::pool_shape`].
    fn pool_shape(&self) -> (usize, usize);
    /// Carry the cell's recurrence across calls, for a chunked sweep.
    fn set_carry(&mut self, carry: bool);
    /// Zero the carried forward state (before a sweep's first chunk).
    fn reset_state(&mut self, gpu: &Gpu);
    /// Zero the carried BPTT state (before a sweep's last chunk backward).
    fn reset_bptt(&mut self, gpu: &Gpu);
    /// The cell's `(saved_cache, other)` activation split. See [`Cell::act_split`].
    fn cell_act_split(&self) -> (usize, usize);
    /// Export the block into the matching CPU `nn` block (`SLSTMBlock` /
    /// `MLSTMBlock`) for a `HIER` checkpoint.
    fn to_nn_layer(&mut self, gpu: &Gpu) -> Box<dyn NnLayer>;
}

impl<C: Cell> BlockLike for Block<C> {
    fn forward(&mut self, gpu: &Gpu, x: &DTensor, out: &mut DTensor) {
        Block::forward(self, gpu, x, out)
    }
    fn backward(&mut self, gpu: &Gpu, dy: &DTensor, dx: &mut DTensor) {
        Block::backward(self, gpu, dy, dx)
    }
    fn zero_grad(&mut self, gpu: &Gpu) {
        Block::zero_grad(self, gpu)
    }
    fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        Block::step(self, gpu, cfg)
    }
    fn step_q(&mut self, gpu: &Gpu, cfg: &AdamCfg, q: Option<&mut ops::AdamwQueue>) {
        Block::step_q(self, gpu, cfg, q)
    }
    fn params_mut(&mut self) -> Vec<&mut DTensor> {
        Block::params_mut(self)
    }
    fn grads(&self) -> Vec<&DTensor> {
        Block::grads(self)
    }
    fn state_extremes(&self, gpu: &Gpu) -> Option<(f32, f32, f32)> {
        self.cell.state_extremes(gpu)
    }
    fn enable_offload(&mut self, gpu: &Gpu, in_flight: offload::SharedInFlight) {
        Block::enable_offload(self, gpu, in_flight)
    }
    fn prefetch_act(&mut self, gpu: &Gpu) {
        Block::prefetch_act(self, gpu)
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
    fn act_breakdown(&self) -> [usize; 5] {
        Block::act_breakdown(self)
    }
    fn pool_shape(&self) -> (usize, usize) {
        Block::pool_shape(self)
    }
    fn set_carry(&mut self, carry: bool) {
        self.carry = carry;
        // The two pre-norms save an `x̂` per forward, exactly like the FFN and the cell.
        self.pre_norm1.set_carry(carry);
        self.pre_norm2.set_carry(carry);
        self.cell.set_carry(carry)
    }
    fn reset_state(&mut self, gpu: &Gpu) {
        // A sweep that forwarded chunks and never unwound them would otherwise leave
        // its FFN caches to accumulate across steps.
        self.act.chunk_saved.clear();
        self.seq.clear();
        self.fwd_chunks = 0;
        self.cell.reset_state(gpu)
    }
    fn reset_bptt(&mut self, gpu: &Gpu) {
        self.cell.reset_bptt(gpu)
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

    /// This block's activations, owned across calls.
    act: Act,
    /// `(B, T)` of each forward still owed a backward, oldest first. A chunked sweep's
    /// last chunk is shorter than the rest, so backward cannot assume one shape — it
    /// pops the shape belonging to the chunk it is unwinding.
    seq: Vec<(usize, usize)>,
    /// Whether this block is part of a chunked sweep, i.e. whether its forward caches
    /// must survive the next chunk's forward. Set alongside the cell's own carry.
    carry: bool,
    /// Chunks forwarded in the current sweep and not yet unwound. Drives the stash:
    /// the first chunk has nothing to preserve, every later one does. Counted rather
    /// than inferred from the `Buf` slots, which `FfnSaved::put_back` refills.
    fwd_chunks: usize,
}

/// A block's activations.
///
/// Only three values have to survive the forward: the SwiGLU operands its
/// backward differentiates. Those get a permanent [`Buf`] each. Everything else
/// — the residual chain, the norm outputs, every `d_*` — is a temporary consumed
/// within the same call, and lives in a [`Pool`] instead.
///
/// That split is what keeps the memory honest. Giving all 23 intermediates a
/// permanent buffer pins every one of them at once: measured at the backbone's
/// shape (16 blocks, N=2048, H=1024) that is 4–6 GB of retained activations, and
/// it pushed an 11 GB step to an out-of-memory abort. Pooling the temporaries
/// keeps only as many buffers as are simultaneously live — a handful — while
/// still allocating nothing once the shapes are steady.
#[derive(Default)]
struct Act {
    /// Scratch for every temporary, recycled by size within the call.
    pool: Pool,
    // The SwiGLU operands, read by backward, so they outlive the forward.
    gate_pre: Buf, // [N, U] pre-activation for SiLU'
    gate_act: Buf, // [N, U] SiLU(gate_pre)
    value: Buf,    // [N, U]
    // The FFN projections' saved inputs, held here rather than inside the three
    // `Linear`s (which would keep `zn` twice — see `forward`). Backward hands these
    // back through `Linear::backward_with_x`.
    zn: Buf,    // [N, H] pre_norm2(z) — input to lin_gate AND lin_value
    mixed: Buf, // [N, U] SwiGLU output — input to lin_down
    /// Host parking for the five buffers above, when offload is enabled.
    ///
    /// The backbone sweeps block by block, so a block's activations sit unread from
    /// its own forward until backward unwinds back to it — 15 blocks of compute at
    /// the backbone's depth. Parking them on the host over that gap trades ~1.2 ms of
    /// (overlapped) PCIe for ~46 MB of device memory per block.
    park: Option<super::offload::HostPark>,
    /// The parked tensors between `restore` and their consumption in backward. Only
    /// non-empty inside `backward`.
    restored: Vec<offload::Parked>,
    /// Earlier chunks' FFN activations, oldest first, when the sweep is chunked.
    ///
    /// The five `Buf` slots above hold one chunk's worth, so without this chunk c+1's
    /// forward overwrites what chunk c's backward reads. Each chunk's set moves here
    /// as the next one's forward starts, and backward pops them right to left. Empty
    /// on the unchunked path, and on the offload path (where the park holds a
    /// generation per chunk instead).
    chunk_saved: Vec<FfnSaved>,
}

impl Act {
    /// A block's activation set, with no offload.
    ///
    /// One place decides, so every way of building a `Block` — fresh, from a CPU
    /// layer, from a checkpoint — gets the same behaviour.
    /// A block's activation set with no offload — the default.
    ///
    /// Offload is opt-in per block via [`Block::enable_offload`], not a property of
    /// construction: only the backbone qualifies. See that method for why.
    fn new(_gpu: &Gpu) -> Self {
        Default::default()
    }
}

/// The five FFN activations backward reads, moved out of the block for the duration
/// of the call.
///
/// They come from one of two places — the owned [`Buf`] slots, or the tensors
/// [`HostPark`](offload::HostPark) just restored — and backward should not care which.
/// Moving them out (rather than borrowing) is what lets the `Linear`s and the pool be
/// borrowed mutably at the same time.
struct FfnSaved {
    gate_pre: DTensor,
    gate_act: DTensor,
    value: DTensor,
    zn: DTensor,
    mixed: DTensor,
    /// Whether these came from the owned `Buf`s and must go back into them.
    owned: bool,
}

impl<C: Cell> Block<C> {
    /// Park this block's FFN activations on the host between forward and backward,
    /// sharing `in_flight` with the other blocks of the same sweep.
    ///
    /// **Only legal for a stack whose forward completes before its backward begins** —
    /// the backbone, which runs all 16 blocks forward and only then unwinds. Two
    /// properties depend on that gap:
    ///
    ///   * the D2H copy has a whole block of compute to finish in, so releasing the
    ///     source buffers at the *next* block's eviction does not stall; and
    ///   * a block's own restore happens long after its eviction landed.
    ///
    /// The decoder violates both: `Hierarchical::forward_backward` runs it forward and
    /// straight back again per length group, so with only two blocks a shared slot
    /// would release buffers still being read — which showed up as
    /// `CUDA_ERROR_ILLEGAL_ADDRESS`, not as a wrong number. The encoder likewise
    /// re-forwards per group rather than saving. Hence opt-in, per stack, rather than
    /// a property of every `Block`.
    pub fn enable_offload(&mut self, gpu: &Gpu, in_flight: offload::SharedInFlight) {
        assert!(
            self.act.restored.is_empty(),
            "enable_offload between forward and backward"
        );
        // The cell gets its own park — its saved set is separate from the FFN's, and
        // in the mLSTM's case comparable in size — but shares the in-flight slot, so
        // the whole block still has only one eviction outstanding at a time.
        self.cell.enable_offload(gpu, in_flight.clone());
        self.act.park = Some(offload::HostPark::new(gpu, in_flight).expect("offload: host park"));
    }

    /// Release the saved FFN activations without reading them.
    ///
    /// For a stack that **re-forwards instead of unwinding** — the encoder, which runs
    /// its forward once per length group and then, in backward, re-runs each group's
    /// forward to rebuild that group's cache (activation checkpointing; see
    /// `Hierarchical::forward_backward`). Every group but the last therefore leaves
    /// buffers nothing will ever read, and because [`Buf`] reuses by capacity they
    /// settle at the largest group's size and stay resident for the whole step.
    ///
    /// Release pooled scratch far larger than a `rows`-row window needs.
    ///
    /// Call at a window boundary. Window sizes vary across a corpus and both [`Buf`]
    /// and [`Pool`] reuse by capacity, so without this every buffer ratchets to the
    /// largest window ever seen: measured on the real `hg` path at
    /// `WORDS_PER_SEQ = 2048`, device memory climbed monotonically window over window
    /// — 10.4 GB, 13.4, 15.9, 16.6 — until it aborted. Nothing about the *steady*
    /// footprint was the problem.
    pub fn trim_to(&mut self, rows: usize) {
        // The widest thing this block pools is `[rows, up]`; sizing the bound from
        // that keeps the ordinary spread of window sizes reusable.
        self.act.pool.trim(rows * self.up);
        self.cell.trim_to(rows);
    }

    /// Modest in absolute terms — the encoder runs at `CHAR_HIDDEN`, an order of
    /// magnitude narrower than the backbone — but these activations are garbage by
    /// construction, and nothing should hold garbage across a step.
    pub fn drop_saved_act(&mut self) {
        let a = &mut self.act;
        for b in [
            &mut a.gate_pre,
            &mut a.gate_act,
            &mut a.value,
            &mut a.zn,
            &mut a.mixed,
        ] {
            b.clear();
        }
        a.restored.clear();
        // These caches are being abandoned, not consumed, so the bookkeeping that
        // tracks what is owed a backward goes with them — otherwise a stack that
        // re-forwards per group (the encoder) accumulates shapes it will never pop.
        a.chunk_saved.clear();
        self.seq.clear();
        self.fwd_chunks = 0;
        self.cell.drop_saved_act();
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
        let a = &self.act;
        act += a.pool.retained_bytes()
            + a.gate_pre.retained_bytes()
            + a.gate_act.retained_bytes()
            + a.value.retained_bytes()
            + a.zn.retained_bytes()
            + a.mixed.retained_bytes();
        (params, act)
    }

    /// Retained activation bytes broken out by owner, for the memory audit:
    /// `(ffn_bufs, pool, norms, projections, cell)`.
    ///
    /// The split matters because only the first two are reachable from
    /// [`drop_saved_act`](Self::drop_saved_act) + [`trim_to`](Self::trim_to); the last
    /// three are held inside the sub-layers and survive both.
    pub fn act_breakdown(&self) -> [usize; 5] {
        let a = &self.act;
        let ffn = a.gate_pre.retained_bytes()
            + a.gate_act.retained_bytes()
            + a.value.retained_bytes()
            + a.zn.retained_bytes()
            + a.mixed.retained_bytes();
        let norms: usize = [&self.pre_norm1, &self.pre_norm2]
            .iter()
            .map(|n| n.retained_bytes().1)
            .sum();
        let proj: usize = [&self.lin_gate, &self.lin_value, &self.lin_down]
            .iter()
            .map(|l| l.retained_bytes().1)
            .sum();
        [ffn, a.pool.retained_bytes(), norms, proj, self.cell.retained_bytes().1]
    }

    /// This block's pool free-list shape `(distinct sizes, buffers)`. Diagnostic.
    pub fn pool_shape(&self) -> (usize, usize) {
        self.act.pool.free_list_shape()
    }

    /// Release every activation this block holds, everywhere — the FFN buffers and
    /// pool, the cell's caches, and the saved inputs and bf16 staging inside the
    /// norms and projections.
    ///
    /// [`drop_saved_act`](Self::drop_saved_act) deliberately keeps the last group; this
    /// does not. For a window boundary, not the hot path.
    pub fn drop_all_act(&mut self, gpu: &Gpu) {
        self.drop_saved_act();
        // The pool is NOT emptied here — see `MLstm::drop_all_act`. It is this block's
        // scratch working set, re-taken in full on the next call, and dropping it at a
        // group boundary puts the allocator back on the hot path. `trim_to` at the
        // window boundary is what sizes it.
        self.pre_norm1.drop_saved_act();
        self.pre_norm2.drop_saved_act();
        for l in [&mut self.lin_gate, &mut self.lin_value, &mut self.lin_down] {
            l.drop_saved_act(gpu);
        }
        self.cell.drop_all_act(gpu);
    }

    /// Turn offload back off. For the parity test, which runs both paths in one
    /// process (the `GPU_NO_OFFLOAD` env gate resolves once and cannot be flipped).
    #[cfg(test)]
    fn disable_offload(&mut self) {
        assert!(
            self.act.restored.is_empty(),
            "disable_offload between forward and backward"
        );
        self.act.park = None;
    }
}

impl FfnSaved {
    /// Move the saved activations out of wherever forward left them.
    fn take(act: &mut Act) -> Self {
        if act.restored.is_empty() {
            let take = |b: &mut Buf, what: &str| b.take().expect(what);
            Self {
                gate_pre: take(&mut act.gate_pre, "forward before backward: gate_pre"),
                gate_act: take(&mut act.gate_act, "forward before backward: gate_act"),
                value: take(&mut act.value, "forward before backward: value"),
                zn: take(&mut act.zn, "forward before backward: zn"),
                mixed: take(&mut act.mixed, "forward before backward: mixed"),
                owned: true,
            }
        } else {
            assert_eq!(
                act.restored.len(),
                5,
                "Block::backward — restored buffer count"
            );
            // Every FFN activation is fp32, so each comes back as `Parked::F32`;
            // `f32()` panics if the park ever hands one back at the wrong width.
            let mut it = act.restored.drain(..);
            let mut next = |what: &str| it.next().expect(what).f32();
            Self {
                gate_pre: next("restored gate_pre"),
                gate_act: next("restored gate_act"),
                value: next("restored value"),
                zn: next("restored zn"),
                mixed: next("restored mixed"),
                owned: false,
            }
        }
    }

    /// Return the buffers to their owned slots, so the next forward reuses the same
    /// allocations. On the offload path there is nothing to return — the tensors were
    /// allocated by `restore` and are dropped here, which is what frees the device
    /// memory again.
    fn put_back(self, act: &mut Act) {
        if !self.owned {
            return;
        }
        act.gate_pre.put(self.gate_pre);
        act.gate_act.put(self.gate_act);
        act.value.put(self.value);
        act.zn.put(self.zn);
        act.mixed.put(self.mixed);
    }
}

impl<C: Cell> Block<C> {
    /// Assemble a block around a cell, with fresh norms (γ=1) and Xavier `Linear`
    /// weights. `hidden` is the model width, `up` the SwiGLU inner width.
    pub fn from_cell(gpu: &Gpu, hidden: usize, up: usize, cell: C) -> Self {
        Self {
            hidden,
            up,
            pre_norm1: RmsNorm::new(gpu, hidden),
            cell,
            pre_norm2: RmsNorm::new(gpu, hidden),
            lin_gate: Linear::from_parts(gpu, &Tensor::xavier(hidden, up), &Tensor::zeros(&[up])),
            lin_value: Linear::from_parts(gpu, &Tensor::xavier(hidden, up), &Tensor::zeros(&[up])),
            lin_down: Linear::from_parts(
                gpu,
                &Tensor::xavier(up, hidden),
                &Tensor::zeros(&[hidden]),
            ),
            act: Act::new(gpu),
            seq: Vec::new(),
            carry: false,
            fwd_chunks: 0,
        }
    }

    /// Assemble around a cell, taking the surrounding norms/projections from a
    /// CPU block (the cell is uploaded by the caller). Shared by the `from_cpu`
    /// constructors below.
    fn from_cpu_parts<D>(gpu: &Gpu, cpu: &crate::nn2::Block<D>, cell: C) -> Self
    where
        D: crate::nn2::block::Cell,
    {
        Self {
            hidden: cpu.hidden,
            up: cpu.up,
            pre_norm1: RmsNorm::from_parts(gpu, &cpu.pre_norm1.gamma),
            cell,
            pre_norm2: RmsNorm::from_parts(gpu, &cpu.pre_norm2.gamma),
            lin_gate: Linear::from_parts(gpu, &cpu.lin_gate.w, &cpu.lin_gate.b),
            lin_value: Linear::from_parts(gpu, &cpu.lin_value.w, &cpu.lin_value.b),
            lin_down: Linear::from_parts(gpu, &cpu.lin_down.w, &cpu.lin_down.b),
            act: Act::new(gpu),
            seq: Vec::new(),
            carry: false,
            fwd_chunks: 0,
        }
    }

    /// Forward over `[B, T, H]` → `y` `[B, T, H]`.
    pub fn forward(&mut self, gpu: &Gpu, x: &DTensor, y: &mut DTensor) {
        assert_eq!(x.rank, 3, "Block::forward expects [B, T, H]");
        let (b, t, h) = (x.shape[0], x.shape[1], x.shape[2]);
        assert_eq!(h, self.hidden, "Block::forward — hidden mismatch");
        assert_eq!(y.dims(), x.dims(), "Block::forward — output shape");
        let (n, u) = (b * t, self.up);
        self.seq.push((b, t));
        // Whole-block span; `glue` is this minus the cell and FFN spans, i.e. the
        // norms, residual adds and buffer copies that are neither.
        let blk_t0 = phase::enabled().then(|| {
            gpu.stream.synchronize().expect("sync");
            (
                std::time::Instant::now(),
                phase::get(self.cell.phase_buckets().0),
                phase::get(phase::Bucket::FfnFwd),
            )
        });
        // Release the previous block's evicted buffers BEFORE allocating this block's.
        //
        // The order matters and is not obvious: eviction leaves the source tensors
        // alive until their D2H lands, and freeing them returns that memory to the
        // CUDA allocator. If this block allocated first, the allocator could hand back
        // memory a live DMA was still reading — an illegal access that only appears
        // asynchronously (it vanishes under CUDA_LAUNCH_BLOCKING=1, which is how it
        // was diagnosed). Releasing first means any memory the allocator reuses here
        // is already drained.
        //
        // The release is an event on the compute stream, not a host wait: the free is
        // itself stream-ordered, so ordering the stream suffices and the transfer
        // still overlaps this block's compute.
        if let Some(park) = &self.act.park {
            park.release_previous(gpu);
        }
        let a = &mut self.act;
        a.pool.assert_drained("Block::forward");
        // Chunked sweep without offload: the previous chunk's FFN activations are still
        // owed a backward, so move them aside before the `Buf` slots below overwrite
        // them. With offload on, the park already holds a generation per chunk and the
        // slots are empty here.
        if self.carry && a.park.is_none() && self.fwd_chunks > 0 {
            let prev = FfnSaved::take(a);
            a.chunk_saved.push(prev);
        }
        if self.carry {
            self.fwd_chunks += 1;
        }

        // Owned [N, H] copy of the input: it feeds both the norm path and the
        // residual, and the caller's `x` is only lent to us.
        let mut x_flat = a.pool.take(gpu, &[n, h]);
        x_flat.copy_from(gpu, x);

        // Residual 1: z = x + cell(pre_norm1(x)). The cell's output is already
        // normalized — each kind does it its own way, inside itself.
        let mut xn1 = a.pool.take(gpu, &[n, h]);
        self.pre_norm1.forward(gpu, &x_flat, &mut xn1);
        xn1.reshape_to(&[b, t, h]);
        let mut cell_out = a.pool.take(gpu, &[b, t, h]);
        let (cf, _cb) = self.cell.phase_buckets();
        phase::timed(gpu, cf, || self.cell.forward(gpu, &xn1, &mut cell_out));
        a.pool.put(xn1);

        // Downstream is position-wise [N, H].
        cell_out.reshape_to(&[n, h]);
        let mut z = a.pool.take(gpu, &[n, h]);
        ops::add_into(gpu, &x_flat, &cell_out, &mut z);
        a.pool.put_all([x_flat, cell_out]);

        // Residual 2: y = z + SwiGLU(pre_norm2(z)). The three SwiGLU operands are
        // the only values backward needs, so they alone go to permanent buffers.
        //
        // `zn` and `mixed` are owned here rather than pooled, because backward reads
        // them as the saved inputs of the three projections. Keeping them once in the
        // block beats `Linear::forward` saving its own copy: `lin_gate` and
        // `lin_value` share `zn`, so that path would hold it twice (4 MB per block at
        // the backbone's shape, 64 MB across 16 blocks) for one tensor.
        self.pre_norm2.forward(gpu, &z, a.zn.get(gpu, &[n, h]));
        // The saved buffers are disjoint `Buf` slots, but each `get`/`expect` borrows
        // `a` as a whole — so take the handles apart once, up front.
        let Act {
            zn,
            gate_pre,
            gate_act,
            value,
            mixed,
            ..
        } = a;
        let zn = zn.expect("normalized");
        phase::timed(gpu, phase::Bucket::FfnFwd, || {
            self.lin_gate
                .forward_shared(gpu, zn, gate_pre.get(gpu, &[n, u]));
            self.lin_value
                .forward_shared(gpu, zn, value.get(gpu, &[n, u]));
            ops::swiglu_forward_into(
                gpu,
                gate_pre.expect("projected"),
                value.expect("projected"),
                gate_act.get(gpu, &[n, u]),
                mixed.get(gpu, &[n, u]),
            );
            // `y = z + down(mixed)`: the residual rides in `lin_down`'s bias seed, so
            // there is no separate add and no `down` buffer to hold its output.
            y.reshape_to(&[n, h]);
            self.lin_down
                .forward_shared_resid(gpu, mixed.expect("mixed"), &z, y);
        });
        y.reshape_to(&[b, t, h]);
        a.pool.put(z);
        // With offload on, this block's saved activations go to the host now and the
        // device buffers are released. Backward restores them (see `restore_act`).
        self.evict_act(gpu);
        if let Some((t0, cell0, ffn0)) = blk_t0 {
            gpu.stream.synchronize().expect("sync");
            let total = t0.elapsed().as_nanos() as u64;
            let inner = (phase::get(self.cell.phase_buckets().0) - cell0)
                + (phase::get(phase::Bucket::FfnFwd) - ffn0);
            phase::add(phase::Bucket::GlueFwd, total.saturating_sub(inner));
        }
    }

    /// Send this block's saved FFN activations to host memory. No-op unless offload is
    /// enabled.
    ///
    /// Called at the end of forward. The five buffers are dead to the device until
    /// backward reaches this block, which in the backbone's block-major sweep is a
    /// whole forward-and-partial-backward away.
    ///
    /// The device tensors are not freed here — they are handed to the shared in-flight
    /// slot, and the *next* block's eviction releases them once the copy has landed.
    /// See [`InFlight`](super::offload::InFlight) for why the reclaim has to be one
    /// block behind rather than immediate.
    fn evict_act(&mut self, gpu: &Gpu) {
        let Act {
            park,
            gate_pre,
            gate_act,
            value,
            zn,
            mixed,
            ..
        } = &mut self.act;
        let Some(park) = park else { return };
        // Hand the device tensors to the park rather than copying and freeing them
        // here. It holds them until its D2H has landed, so the copy overlaps the next
        // block's compute instead of blocking on this one — waiting here costs the
        // entire transfer time (measured: +24% on a step, the un-overlapped total).
        let take = |b: &mut Buf, what: &str| offload::Parked::from(b.take().expect(what));
        park.evict(
            gpu,
            vec![
                take(gate_pre, "forward filled gate_pre"),
                take(gate_act, "forward filled gate_act"),
                take(value, "forward filled value"),
                take(zn, "forward filled zn"),
                take(mixed, "forward filled mixed"),
            ],
        );
    }

    /// Start this block's parked activations on their way back to the device.
    ///
    /// The caller runs this one block ahead of the block whose backward it belongs to,
    /// so the upload overlaps that block's compute. Without it, `restore_act` issues
    /// the copy and immediately waits — the transfer is fully exposed, which measured
    /// as +37 ms of "block glue" against ~32 ms of raw transfer, i.e. no overlap.
    pub fn prefetch_act(&mut self, gpu: &Gpu) {
        // Backward reads the FFN's activations first and the cell's after, so they are
        // issued in that order — the transfer stream serves them FIFO.
        if let Some(park) = &mut self.act.park {
            park.prefetch(gpu);
        }
        self.cell.prefetch_act(gpu);
    }

    /// Bring the parked activations back, in the order `evict_act` sent them.
    ///
    /// Returns them rather than refilling the `Buf`s: they are consumed once, within
    /// this backward, and putting them back in the owned slots would keep the device
    /// memory alive until the next forward overwrote it — exactly what parking exists
    /// to avoid.
    fn restore_act(&mut self, gpu: &Gpu) {
        let Act { park, restored, .. } = &mut self.act;
        let Some(park) = park else { return };
        *restored = park.restore(gpu);
    }

    /// Backward over `[B, T, H]` → `dx` `[B, T, H]`.
    pub fn backward(&mut self, gpu: &Gpu, dy: &DTensor, dx: &mut DTensor) {
        let (b, t) = self.seq.pop().expect("Block::backward before forward");
        let (h, u) = (self.hidden, self.up);
        let n = b * t;
        assert_eq!(dx.dims(), [b, t, h], "Block::backward — dx shape");
        let blk_t0 = phase::enabled().then(|| {
            gpu.stream.synchronize().expect("sync");
            (
                std::time::Instant::now(),
                phase::get(self.cell.phase_buckets().1),
                phase::get(phase::Bucket::FfnBwd),
            )
        });
        // With offload on, this block's activations come back from the host now, into
        // `self.restored`. The H2D copies are issued on the transfer stream and waited
        // for on the compute stream, so they overlap whatever the previous block's
        // backward is still finishing.
        self.restore_act(gpu);
        // Take the saved set out of `self` entirely for the duration of this call, so
        // the `Linear`s and the pool below can be borrowed mutably alongside it. The
        // buffers are returned to their slots at the end (`FfnSaved::put_back`), which
        // for the offload path means simply dropping them.
        let saved = FfnSaved::take(&mut self.act);
        self.fwd_chunks = self.fwd_chunks.saturating_sub(1);
        let a = &mut self.act;
        a.pool.assert_drained("Block::backward");

        // Owned [N, H]: read by lin_down.backward and again by the d_z residual.
        let mut dy_flat = a.pool.take(gpu, &[n, h]);
        dy_flat.copy_from(gpu, dy);

        // Residual 2.
        let mut d_mixed = a.pool.take(gpu, &[n, u]);
        let ffn_t0 = phase::enabled().then(|| {
            gpu.stream.synchronize().expect("sync");
            std::time::Instant::now()
        });
        self.lin_down
            .backward_with_x(gpu, &saved.mixed, &dy_flat, &mut d_mixed);
        let mut d_gate = a.pool.take(gpu, &[n, u]);
        let mut d_value = a.pool.take(gpu, &[n, u]);
        ops::swiglu_backward_into(
            gpu,
            &d_mixed,
            &saved.gate_act,
            &saved.value,
            &saved.gate_pre,
            &mut d_gate,
            &mut d_value,
        );
        a.pool.put(d_mixed);
        // Both projections read the one saved `zn` (see `forward`).
        let mut d_zn_g = a.pool.take(gpu, &[n, h]);
        self.lin_gate
            .backward_with_x(gpu, &saved.zn, &d_gate, &mut d_zn_g);
        let mut d_zn_v = a.pool.take(gpu, &[n, h]);
        self.lin_value
            .backward_with_x(gpu, &saved.zn, &d_value, &mut d_zn_v);
        let mut d_zn = a.pool.take(gpu, &[n, h]);
        ops::add_into(gpu, &d_zn_g, &d_zn_v, &mut d_zn);
        if let Some(t0) = ffn_t0 {
            gpu.stream.synchronize().expect("sync");
            phase::add(phase::Bucket::FfnBwd, t0.elapsed().as_nanos() as u64);
        }
        a.pool.put_all([d_gate, d_value, d_zn_g, d_zn_v]);

        // z feeds pre_norm2 (the MLP path) and the y = z + down residual, so the
        // norm's dx and the incoming dy sum into d_z. `d_z_mlp` is separate because
        // `add_into`'s destination must not be one of its operands.
        let mut d_z_mlp = a.pool.take(gpu, &[n, h]);
        self.pre_norm2.backward(gpu, &d_zn, &mut d_z_mlp);
        let mut d_z = a.pool.take(gpu, &[n, h]);
        ops::add_into(gpu, &d_z_mlp, &dy_flat, &mut d_z);
        a.pool.put_all([d_zn, d_z_mlp, dy_flat]);

        // Residual 1. The cell receives a copy of d_z rather than d_z itself: its
        // backward must not clobber d_z, which the dx residual still needs.
        let mut d_cell_out = a.pool.take(gpu, &[n, h]);
        d_cell_out.copy_from(gpu, &d_z);
        d_cell_out.reshape_to(&[b, t, h]);
        let mut d_cell_in = a.pool.take(gpu, &[b, t, h]);
        let (_cf, cb) = self.cell.phase_buckets();
        phase::timed(gpu, cb, || {
            self.cell.backward(gpu, &d_cell_out, &mut d_cell_in)
        });
        a.pool.put(d_cell_out);
        d_cell_in.reshape_to(&[n, h]);
        let mut d_xn1 = a.pool.take(gpu, &[n, h]);
        self.pre_norm1.backward(gpu, &d_cell_in, &mut d_xn1);
        // x feeds pre_norm1 (cell path) and the z = x + cn residual.
        dx.reshape_to(&[n, h]);
        ops::add_into(gpu, &d_xn1, &d_z, dx);
        dx.reshape_to(&[b, t, h]);
        a.pool.put_all([d_cell_in, d_xn1, d_z]);
        // Give the saved buffers back to their owned slots so the next forward reuses
        // the allocations. On the offload path this drops them instead, which is what
        // releases the restored device memory again.
        saved.put_back(a);
        // Chunked sweep: the slots now hold the chunk just unwound, which nothing will
        // read again. Replace them with the chunk to its left — the next to unwind —
        // so `FfnSaved::take` finds that chunk's own activations. The allocations the
        // line above returned are dropped here, releasing them a chunk earlier than
        // the next forward would.
        if let Some(prev) = a.chunk_saved.pop() {
            prev.put_back(a);
        }
        if let Some((t0, cell0, ffn0)) = blk_t0 {
            gpu.stream.synchronize().expect("sync");
            let total = t0.elapsed().as_nanos() as u64;
            let inner = (phase::get(self.cell.phase_buckets().1) - cell0)
                + (phase::get(phase::Bucket::FfnBwd) - ffn0);
            phase::add(phase::Bucket::GlueBwd, total.saturating_sub(inner));
        }
    }

    /// Learnable tensors in a fixed order (checkpoint save/load).
    pub fn params_mut(&mut self) -> Vec<&mut DTensor> {
        let mut v = Vec::new();
        v.extend(self.pre_norm1.params_mut());
        v.extend(self.cell.params_mut());
        v.extend(self.pre_norm2.params_mut());
        v.extend(self.lin_gate.params_mut());
        v.extend(self.lin_value.params_mut());
        v.extend(self.lin_down.params_mut());
        v
    }

    /// Gradient accumulators, matching `params_mut`'s order. Diagnostic.
    pub fn grads(&self) -> Vec<&DTensor> {
        let mut v = vec![&self.pre_norm1.dgamma];
        v.extend(self.cell.grads());
        v.push(&self.pre_norm2.dgamma);
        for l in [&self.lin_gate, &self.lin_value, &self.lin_down] {
            v.push(&l.dw);
            v.push(&l.db);
        }
        v
    }

    /// Export the block into the matching CPU `nn` block for a `HIER` checkpoint.
    /// Downloads every surrounding norm/projection, then lets the cell assemble
    /// the concrete `SLSTMBlock` / `MLSTMBlock`.
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

    /// AdamW step across every sub-layer.
    pub fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        self.step_q(gpu, cfg, None);
    }

    /// [`step`](Self::step), optionally queueing instead of launching.
    pub fn step_q(&mut self, gpu: &Gpu, cfg: &AdamCfg, mut q: Option<&mut ops::AdamwQueue>) {
        self.pre_norm1.step_q(gpu, cfg, q.as_deref_mut());
        self.cell.step_q(gpu, cfg, q.as_deref_mut());
        self.pre_norm2.step_q(gpu, cfg, q.as_deref_mut());
        self.lin_gate.step_wd_q(gpu, cfg, true, q.as_deref_mut());
        self.lin_value.step_wd_q(gpu, cfg, true, q.as_deref_mut());
        self.lin_down.step_wd_q(gpu, cfg, true, q.as_deref_mut());
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
        Self {
            hidden: cpu.hidden_size,
            up: cpu.up_size,
            pre_norm1: RmsNorm::from_parts(gpu, &v(&cpu.pre_norm1.gamma)),
            // The checkpoint keeps the post-cell norm on the block; the GPU cell owns
            // it, so its γ is handed down here.
            cell: SLstm::from_nn_cell(gpu, &cpu.cell, Some(&v(&cpu.post_cell_norm.gamma))),
            pre_norm2: RmsNorm::from_parts(gpu, &v(&cpu.pre_norm2.gamma)),
            lin_gate: lin_from_nn(gpu, &cpu.lin_gate),
            lin_value: lin_from_nn(gpu, &cpu.lin_value),
            lin_down: lin_from_nn(gpu, &cpu.lin_down),
            act: Act::new(gpu),
            seq: Vec::new(),
            carry: false,
            fwd_chunks: 0,
        }
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
        Self {
            hidden: cpu.hidden_size,
            up: cpu.up_size,
            pre_norm1: RmsNorm::from_parts(gpu, &v(&cpu.pre_norm1.gamma)),
            cell: MLstm::from_nn_cell(gpu, &cpu.cell),
            pre_norm2: RmsNorm::from_parts(gpu, &v(&cpu.pre_norm2.gamma)),
            lin_gate: lin_from_nn(gpu, &cpu.lin_gate),
            lin_value: lin_from_nn(gpu, &cpu.lin_value),
            lin_down: lin_from_nn(gpu, &cpu.lin_down),
            act: Act::new(gpu),
            seq: Vec::new(),
            carry: false,
            fwd_chunks: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn2::block::{MLstmBlock as CpuMLstmBlock, SLstmBlock as CpuSLstmBlock};
    use crate::nn2::optim::AdamCfg;
    use crate::tensor::Tensor;

    /// Compare with an absolute floor plus a term scaled by the tensor's magnitude.
    ///
    /// The scale term is `rel * max|want|`, **not** `rel * |want[i]|`. bf16 slab
    /// storage perturbs each saved `zt`/`ot` by ~2^-8 of *its own* magnitude, but
    /// `dx[i]` is a sum over many such terms, so what lands on any one element is an
    /// absolute error set by the size of the whole tensor. An individual `dx[i]` that
    /// comes out small does so by cancellation between larger terms — its error does
    /// not shrink with it, and measured runs show ~0.006-0.027 absolute spread
    /// roughly evenly whether the element is 0.09 or 4.6.
    ///
    /// A per-element relative bound therefore fails on exactly the elements where the
    /// physics says it should: the near-zero ones. On the fp32 path `rel` is 0 and
    /// this reduces to the original absolute check.
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

    /// Relative tolerance term for GPU-vs-CPU comparisons: bf16's half-ulp bound
    /// when the sLSTM's saved slabs are stored narrow, zero on the fp32 path.
    ///
    /// The CPU reference is all-fp32, so with bf16 slabs the gap is one quantization
    /// of `zt`/`ot` propagated through the block — bounded in relative terms, which
    /// is why it belongs here rather than in the absolute tolerance. The dangerous
    /// failure mode (error *growing* with sequence length) is pinned separately by
    /// `gpu::slstm::tests::bf16_slab_error_does_not_compound_with_t`.
    /// Relative term for a parameter compared AFTER an Adam step.
    ///
    /// Larger than [`rel`] on purpose. Adam's `lr·ĝ/(√v̂+ε)` is scale-invariant in
    /// the gradient, so a weight whose gradient is near zero has its bf16-sized
    /// difference divided by an equally small √v̂ — a ~1e-7 wobble lands as ~1e-4 on
    /// the weight. Reusing the activation tolerance here leaves a bound that passes
    /// in isolation and fails on roughly one full-suite run in five.
    fn step_rel(gpu: &Gpu) -> f32 {
        if ops::gemm_bf16_enabled(gpu) || gpu.kernels.slab_bf16 {
            5e-2
        } else {
            0.0
        }
    }

    fn rel(gpu: &Gpu) -> f32 {
        // Three independent bf16 sources, not one: the saved slabs, the projections'
        // GEMM operands, and the sLSTM's own whole-sequence GEMMs (`x·Wx`, `dg·Wxᵀ`,
        // `xᵀ·dg`, also `ops::GemmBf16`). A block chains several matmuls, each
        // contributing ~2^-8 relative on its own operands, so the budget against an
        // all-fp32 CPU reference is a small multiple of the single-quantization bound
        // rather than exactly it.
        //
        // 1e-2 was calibrated before the sLSTM GEMMs joined and sat right on the
        // measured maximum, so it failed roughly one run in three. Ten runs of the
        // worst element: `y` peaks at 0.0094, `dx` at 0.0110 — the error is bounded and
        // does not grow, the bound simply had no margin. 2e-2 is ~1.8x the observed max.
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
        let (b, t, h, u) = (2, 4, 8, 12);

        let mut cpu = CpuSLstmBlock::new_slstm(h, u);
        let mut dev = Block::<SLstm>::from_cpu(&gpu, &cpu);

        let x = Tensor::random(&[b, t, h], 0.5);
        let g = Tensor::random(&[b, t, h], 1.0);

        // Forward
        let y_cpu = cpu.forward(&x);
        let y_dev = dev.forward_alloc(&gpu, &DTensor::from_host(&gpu, &x));
        assert_close_rel(&y_dev.to_host(&gpu).data, &y_cpu.data, 3e-3, rel(&gpu), "y");

        // Backward
        let dx_cpu = cpu.backward(&g);
        let dx_dev = dev.backward_alloc(&gpu, &DTensor::from_host(&gpu, &g));
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
    /// recurrence) for forward → backward → AdamW-step from identical parameters.
    /// This closes Phase C: the block wiring is shared, so it also re-checks that
    /// the mLSTM cell composes correctly inside the two residuals.
    #[test]
    fn mlstm_block_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
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
        let y_dev = dev.forward_alloc(&gpu, &DTensor::from_host(&gpu, &x));
        assert_close_rel(&y_dev.to_host(&gpu).data, &y_cpu.data, 3e-3, rel(&gpu), "y");

        let dx_cpu = cpu.backward(&g);
        let dx_dev = dev.backward_alloc(&gpu, &DTensor::from_host(&gpu, &g));
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

    /// Parking the FFN activations on the host must not change a single bit.
    ///
    /// Offload only moves bytes — it reorders no arithmetic and renormalizes nothing —
    /// so this is **exact** equality on the output, every gradient, and the weights
    /// after a step. A tolerance here would hide precisely the bugs that matter: a
    /// stale buffer, a chunk restored out of order, a missing cross-stream event.
    ///
    /// Both cell kinds run, because the block's saved set is the same either way but
    /// the surrounding cell's memory traffic is not.
    #[test]
    fn offload_matches_resident_exactly() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let (b, t, h, u) = (2, 5, 8, 12);

        // `run` builds an identical block from identical CPU weights, so the only
        // difference between the two calls is where the activations lived.
        fn run<C: Cell, F: Fn() -> Block<C>>(
            gpu: &Gpu,
            build: F,
            x: &Tensor,
            g: &Tensor,
            offload: bool,
        ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
            let mut dev = build();
            if offload {
                dev.enable_offload(gpu, offload::InFlight::shared());
            }
            let y = dev.forward_alloc(gpu, &DTensor::from_host(gpu, x));
            let dx = dev.backward_alloc(gpu, &DTensor::from_host(gpu, g));
            // Gradients of the three FFN projections are the ones the parked buffers
            // feed, so they are the sharpest probe.
            let dw_down = dev.lin_down.dw.to_host(gpu).data;
            let dw_gate = dev.lin_gate.dw.to_host(gpu).data;
            (y.to_host(gpu).data, dx.to_host(gpu).data, dw_down, dw_gate)
        }

        let x = Tensor::random(&[b, t, h], 0.5);
        let g = Tensor::random(&[b, t, h], 1.0);

        // sLSTM cell.
        let cpu_s = CpuSLstmBlock::new_slstm(h, u);
        let build_s = || Block::<SLstm>::from_cpu(&gpu, &cpu_s);
        let resident = run(&gpu, build_s, &x, &g, false);
        let parked = run(&gpu, build_s, &x, &g, true);
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
        let resident = run(&gpu, build_m, &x, &g, false);
        let parked = run(&gpu, build_m, &x, &g, true);
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

    /// Several forward/backward cycles on one offloaded block must stay exact.
    ///
    /// A single cycle would not catch a park that leaks state across steps — a pinned
    /// slot reused at the wrong shape, or an event left un-awaited so step `n+1`'s
    /// eviction races step `n`'s restore. The shapes deliberately change between
    /// cycles, which is what the real dataset does.
    #[test]
    fn offload_is_stable_across_steps_and_shapes() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let (h, u) = (8, 12);
        let cpu = CpuSLstmBlock::new_slstm(h, u);
        let mut resident = Block::<SLstm>::from_cpu(&gpu, &cpu);
        let mut parked = Block::<SLstm>::from_cpu(&gpu, &cpu);
        resident.disable_offload();
        parked.enable_offload(&gpu, offload::InFlight::shared());

        let mut cfg = AdamCfg::new(1e-3, 0.01);
        for step in 0..4 {
            let (b, t) = (1 + step % 2, 3 + step); // shapes vary per step
            let x = Tensor::random(&[b, t, h], 0.5);
            let g = Tensor::random(&[b, t, h], 1.0);
            let dx = DTensor::from_host(&gpu, &x);
            let dg = DTensor::from_host(&gpu, &g);

            let y_r = resident.forward_alloc(&gpu, &dx).to_host(&gpu).data;
            let y_p = parked.forward_alloc(&gpu, &dx).to_host(&gpu).data;
            assert_eq!(y_p, y_r, "step {step}: y diverged");

            let dxr = resident.backward_alloc(&gpu, &dg).to_host(&gpu).data;
            let dxp = parked.backward_alloc(&gpu, &dg).to_host(&gpu).data;
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
}
