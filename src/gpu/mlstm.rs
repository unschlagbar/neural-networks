//! Device-resident multi-head mLSTM cell, GPU counterpart of
//! [`nn2::mlstm::MLstm`](crate::nn2::mlstm::MLstm) — in the **parallel /
//! chunkwise** formulation, not the scalar per-head recurrence (the CPU's sub-1×
//! path; see PLAN-gpu.md Phase C).
//!
//! The equivalence (derived in the plan): the CPU stores the *stabilized* state
//! `C_t = C_t^true·exp(−m_t)`, and the running stabilizer unrolls to a row-max
//! over the log-decay matrix `logD_{tj} = fc_t − fc_j + ĩ_j` (`fc` = cumulative
//! log-forget). For the whole sequence as a single chunk (`C_prev=n_prev=0`,
//! `m_prev=0`):
//! ```text
//!   S = Q·Kᵀ ;  m_t = max(max_{j≤t} logD_{tj}, fc_t)
//!   D̄_{tj} = exp(logD_{tj} − m_t)  (j≤t else 0)
//!   ỹ_t = ((D̄⊙S)·V)_t / ψ_t ,  ψ_t = max(|Σ_j (D̄⊙S)_{tj}|, exp(−m_t))
//! ```
//! then head-norm(ỹ) → ŷ, `y = o⊙ŷ`, `h = y·W_out + b_out`. Backward
//! differentiates this graph with `m` held constant (the reference stabilizer
//! approximation, same as the CPU / the sLSTM cell).
//!
//! The input projections and `W_out` are `gpu::Linear`; only the attention core is
//! bespoke kernels. Those read q/k/v where the projections left them —
//! position-major and concatenated — so the cell has no reorg pass at all, and every
//! contraction in them runs on the tensor cores with bf16 operands, which is what
//! the projections already produced.
//!
//! # Chunking (`config::MLSTM_CHUNK`)
//!
//! Taking the whole sequence as one chunk costs O(T²) — the `[BH, T, T]` matrices.
//! Instead the sequence is cut into chunks of length `L`, each evaluated by the
//! parallel form above over its own `[BH, L, L]` matrices, with the stabilized
//! recurrent state carried across boundaries (`C_prev`, `n_prev`, `m_prev`):
//! ```text
//!   num_t += b_t·(C_prev·q_t) ,  qn_t += b_t·(q_t·n_prev)      b_t = exp(fc_t+m_prev−m_t)
//!   C ← g·C_prev + (a⊙V)ᵀ·K  ,  n ← g·n_prev + Σ_j a_j k_j     a_j = D̄_{last,j}, g = b_last
//! ```
//! with `fc` the chunk-LOCAL cumulative log-forget. This is O(T·L) — linear in T.
//!
//! It is an exact refactoring, not an approximation: the chunk-local row-max
//! `m_t = max(max_{j∈chunk, j≤t} logD_tj, fc_t + m_prev)` telescopes to the global
//! row-max, so chunked and single-chunk agree to fp tolerance in both forward and
//! backward (`mlstm_chunking_matches_single_chunk`). Backward sweeps chunks in
//! reverse, carrying `dC`/`dn` — BPTT over chunks, parallel form within each.
//!
//! Each direction is four launches, in mirror-image shapes, and only one of them is
//! serial in anything:
//!
//! | | forward | backward |
//! |---|---|---|
//! | per-timestep scalars | `mlstm_fw_gates` (gate scan, one block per `(b, h)`) | `mlstm_bw_dqn` (dψ, one warp per step) |
//! | per-chunk state product | `mlstm_fw_dC` (every chunk's `ΔC`, all at once) | `mlstm_bw_dC` (every chunk's `ΔdC`) |
//! | the chunk recurrence | `mlstm_state_scan` | the same kernel, reversed |
//! | intra-chunk attention | `mlstm_fw_parallel` | `mlstm_bw_parallel` |
//!
//! The scan is the one place the chunk recurrence is actually walked, and it is
//! elementwise, so it parallelises over the whole state either way. Nothing loops
//! chunks inside a kernel: a chunk's own contribution to the state depends on
//! nothing outside it, because `a` and `b` already carry the stabilizer.
//!
//! A sequence already shorter than `L` (the encoder/decoder, where T is a word
//! length) takes the single-chunk path with no inter-chunk work at all — both scans
//! are skipped outright.

use std::sync::OnceLock;

use super::arena::{self, ParamSlot};
use super::block::Cell;
use super::nn_convert::concat_cols;
use super::{GTensor, Gpu, linear::Linear, ops, rms_norm::RmsNorm};
use crate::gpu::arena::TrainingCache;
use crate::nn2::optim::AdamCfg;
use crate::tensor::Tensor;

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

/// Forward intermediates of the **fused** path. Far smaller than `Saved`: the
/// per-chunk `[BH, L, L]` decay matrices do not exist — backward rebuilds them in
/// shared memory (see `ops::MlstmFused`).
struct SavedFused {
    b: usize,
    t: usize,
    /// The three large per-`N` tensors, `None` exactly while they are parked on the
    /// host.
    ///
    /// `Option` rather than a `parked: bool` flag: backward reads these through
    /// `expect`, so a missing `restore_saved` is a named panic at the use site instead
    /// of a silent read of a stale buffer. They are `Some` for the whole of a
    /// non-offloaded run.
    // `q ‖ k ‖ v ‖ o` in bf16 storage, mirroring the reference's DTYPE tensors
    // (matQ/matK/matV plus the o-gate pre-activation) concatenated as its fused
    // `qkv_opreact` output. Holds the RAW `o`: `ogate_bwd` recomputes the sigmoid.
    xh: Option<ops::SlabBuf>,
    // fp32: `ĩ ‖ f̃`. The reference loads vecI/vecB `.to(tl.float32)` — they are
    // exponents feeding the stabilizer, where an absolute error becomes a
    // multiplicative one. See `gpu::bf16`.
    // Left resident when parking: 128 KB against the 4 MB tensors above, so moving it
    // would cost bookkeeping and PCIe for nothing.
    gates: GTensor<f32>,
    fused: ops::MlstmFused,
    yhat: Option<GTensor<f32>>,
    /// `lin_out`'s input, kept here rather than as the projection's private copy so a
    /// chunked sweep's later chunk cannot overwrite it. Fed back via `backward_with_x`.
    hconcat: Option<GTensor<f32>>,
}

impl SavedFused {
    /// The parkable tensors, in the one order `evict`/`restore` both use.
    ///
    /// Panics if they are parked — every reader runs after `restore_saved`.
    fn yhat(&self) -> &GTensor<f32> {
        self.yhat
            .as_ref()
            .expect("mLSTM: yhat is parked on the host")
    }
    fn hconcat(&self) -> &GTensor<f32> {
        self.hconcat
            .as_ref()
            .expect("mLSTM: hconcat is parked on the host")
    }
    fn xh(&self) -> &ops::SlabBuf {
        self.xh
            .as_ref()
            .expect("mLSTM: q‖k‖v‖o is parked on the host")
    }

    /// Device bytes held. Parked tensors count as zero — they are on the host, which
    /// is the whole point of parking them.
    fn retained_bytes(&self) -> usize {
        let opt_f32: usize = [&self.yhat, &self.hconcat]
            .iter()
            .filter_map(|s| s.as_ref())
            .map(|t| t.capacity() * 4)
            .sum();
        let slabs: usize = self.xh.as_ref().map_or(0, |s| s.retained_bytes());
        opt_f32 + slabs + self.gates.capacity() * 4 + self.fused.retained_bytes()
    }
}

pub struct MLstm {
    pub input_size: usize,
    pub d: usize,
    pub heads: usize,
    pub dqk: usize,
    pub dhv: usize,
    pub(super) inv_sqrt_dqk: f32,
    /// Chunk length (0 = single-chunk). Defaults to [`chunk_len`].
    chunk: usize,

    // Projections (in → ·) and the output projection (d → d). Bias, weight decay
    // and AdamW all handled by `Linear`, matching the CPU cell's conventions.
    //
    // Two projections, not six: `q ‖ k ‖ v ‖ o` of width `H*(2*dqk + 2*dhv)` and
    // the two gate logits of width `2*H` — the reference's fused weight mode
    // (nx-ai/xlstm, `xlstm_large/model.py`: `qkv_opreact` + `ifgate_preact`). The
    // kernels address each part where the GEMM left it, so nothing is split apart
    // afterwards.
    //
    // The gate logits keep a projection of their own because they must stay fp32
    // (see `forward`), while `q ‖ k ‖ v ‖ o` lands in a bf16 slab.
    pub(super) lin_qkvo: Linear,
    pub(super) lin_gates: Linear,
    pub(super) lin_out: Linear,
    pub(super) headnorm: RmsNorm, // head-wise (group == dhv)
    /// Forward caches awaiting a backward, one per chunk in eviction order.
    ///
    /// A chunked sweep forwards every chunk before unwinding any, so chunk c's cache
    /// must survive chunk c+1's forward; backward pops from the end (right to left).
    /// The unchunked path is this with a single element.
    saved: Vec<SavedFused>,
    /// Host parking for the fused cache's large per-N tensors, when the surrounding
    /// stack opted in (`Block::enable_offload` → `MLstm::enable_offload`).
    ///
    /// Same mechanism as the FFN's: written once in forward, read once in backward,
    /// and in the backbone's block-major sweep those are a whole pass apart. Only the
    /// big ones ride — see `evict_saved` for which and why.
    park: Option<super::offload::HostPark>,
    /// Continue the previous call's recurrence instead of starting at zero — set for a
    /// chunked sweep. See [`MLstm::set_carry`].
    carry: bool,
    /// The state the previous chunk ended with, seeded into the next chunk's kernel.
    /// `None` before the first chunk of a sweep (and whenever `carry` is off).
    carry_state: Option<ops::MlstmState>,
    /// BPTT state from the chunk to the right, for a chunked backward.
    carry_dstate: Option<ops::MlstmDState>,
}

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
        // q·kᵀ wants k scaled by 1/√dqk. The factor is constant, so it is folded into
        // wk/bk here and unfolded in `to_nn_cell`: the checkpoint keeps the unscaled
        // weights, while the runtime never spends a pass rescaling k or dk.
        let inv_sqrt_dqk = 1.0 / (dqk as f32).sqrt();
        let scaled =
            |t: &Tensor| Tensor::new(&t.dims(), t.data.iter().map(|v| v * inv_sqrt_dqk).collect());
        let (wk, bk) = (&scaled(wk), &scaled(bk));
        Self {
            input_size,
            d,
            heads,
            dqk,
            dhv,
            inv_sqrt_dqk,
            chunk: chunk_len(),
            lin_qkvo: Linear::from_parts(
                gpu,
                &concat_cols(&[wq, wk, wv, wo]),
                &concat_cols(&[bq, bk, bv, bo]),
            ),
            lin_gates: Linear::from_parts(gpu, &concat_cols(&[wi, wf]), &concat_cols(&[bi, bf])),
            lin_out: Linear::from_parts(gpu, w_out, b_out),
            headnorm: RmsNorm::from_parts_grouped(gpu, gamma, dhv),
            saved: Vec::new(),
            park: None,
            carry: false,
            carry_state: None,
            carry_dstate: None,
        }
    }

    /// Freshly-initialised cell, matching `nn2::MLstm::new`'s init exactly.
    pub fn new_rand(gpu: &Gpu, input_size: usize, d: usize, heads: usize, dqk: usize) -> Self {
        Self::from_cpu(gpu, &crate::nn2::MLstm::new(input_size, d, heads, dqk))
    }

    /// Column where `o` starts in the fused `q ‖ k ‖ v ‖ o` projection.
    ///
    /// The kernels derive the same offset from `(H, dqk, dhv)` themselves
    /// (`MLSTM_OFF_O`); the o-gate pair takes it as an argument because those two are
    /// plain elementwise kernels with no shape specialization to fold it into.
    fn o_off(&self) -> usize {
        2 * self.heads * self.dqk + self.d
    }

    /// Upload a CPU cell (weights are copied; grads/moments start at zero).
    pub fn from_cpu(gpu: &Gpu, c: &crate::nn2::MLstm) -> Self {
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
    ///
    /// Chunk length is a blocking choice with no effect on the result
    /// (`mlstm_chunking_matches_single_chunk` pins that), so a configured chunk is
    /// clamped into what the kernels support rather than dispatched on: longer than
    /// `FUSED_MAX_L` clamps down (the decay matrix must fit in shared memory), and
    /// `chunk == 0` clamps UP to 1, a step-by-step recurrence.
    fn fused_chunk(&self, gpu: &Gpu, t: usize) -> usize {
        let l = self.chunk.min(ops::FUSED_MAX_L).min(t).max(1);
        debug_assert!(
            ops::mlstm_fused_smem_bytes(l, self.dqk, self.dhv) <= gpu.max_shared_optin,
            "fused mLSTM shared memory exceeds this device's opt-in limit",
        );
        l
    }

    /// Forward over `[B, T, in]` → `[B, T, d]`.
    ///
    /// Chunkwise: the sequence is cut into chunks of `chunk_len()`, each handled by
    /// the parallel (attention) form over its own `[BH, L, L]` decay matrix, with
    /// the recurrent state `(C, n, m)` carried across chunk boundaries. One chunk
    /// covering the whole sequence reduces to the single-chunk form.
    ///
    /// `out` is the caller's `[B, T, d]` buffer, written in place.
    ///
    /// `x` must stay alive and unchanged until this call's [`backward`](Self::backward),
    /// which takes it back: the three projections read it for their `dW`, and this cell
    /// keeps no copy of it. In a [`Block`](super::block::Block) that is the block's own
    /// `xn1` — the pre-norm output it already holds for the norm's backward.
    pub fn forward(
        &mut self,
        gpu: &Gpu,
        x: &GTensor<f32>,
        out: &mut GTensor<f32>,
        cache: &TrainingCache,
    ) {
        // Release the previous eviction before allocating anything here: freeing
        // returns memory to the CUDA allocator, which must not hand it back while a
        // copy is still reading it. See `InFlight::release`.
        if let Some(park) = &self.park {
            park.release_previous();
        }
        assert_eq!(x.rank, 3, "MLstm::forward expects [B, T, in]");
        let (batch, t, inp) = (x.shape[0], x.shape[1], x.shape[2]);
        assert_eq!(
            inp, self.input_size,
            "MLstm::forward — input width mismatch"
        );
        let dim = self.d;
        assert_eq!(out.dims(), [batch, t, dim], "MLstm::forward — output shape");
        let rows = batch * t;
        let shape = ops::MlstmShape {
            batch,
            heads: self.heads,
            t,
            dqk: self.dqk,
            dhv: self.dhv,
        };

        // The input as `[N, in]`, which is the shape both projections want. A view:
        // every projection's `dW` reads it again in backward, but the caller hands it
        // back there (`backward`'s `x`), so this cell keeps no copy of its own.
        let xf = GTensor::view(gpu, &x.buf, 0, &[rows, inp]);
        let l = self.fused_chunk(gpu, t);

        // Two projections, both off the same narrowed `xf`. `q ‖ k ‖ v ‖ o` lands in
        // one slab at the kernels' own width and stays exactly where the GEMM wrote
        // it — the position-major, concatenated `[rows, H*(2*dqk+2*dhv)]` is what the
        // fused kernels and `ogate_fwd` address. The gate logits keep a projection of
        // their own so they can stay fp32: they are exponents feeding the stabilizer,
        // where an absolute error becomes a multiplicative one, whereas `o` feeds a
        // sigmoid, which is bounded and Lipschitz. At `[rows, 2*H]` they are a factor
        // of `dqk` smaller than the slab, so narrowing them would buy nothing anyway.
        //
        // Both outlive the call — backward reads them — so neither is pooled.
        let mut xh = ops::SlabBuf::new(gpu, &[rows, self.lin_qkvo.output_size()]);
        let mut gates = GTensor::uninit(gpu, &[rows, self.lin_gates.output_size()]);
        // Both projections read the same narrowed `xf`, so it is staged once here
        // rather than by each of them. A borrowed slot, at bf16's own width: the same
        // array the fp32 temporaries come from, viewed narrow.
        let mut xf_b = cache.temps.get::<u16>(gpu, xf.dims());
        xf_b.store(gpu, &xf);
        self.lin_qkvo.forward_staged_slab(gpu, &xf, &xf_b, &mut xh);
        self.lin_gates.forward_staged(gpu, &xf, &xf_b, &mut gates);
        drop(xf_b);
        // k needs no rescale: 1/√dqk is folded into its columns of `lin_qkvo`. `o` is
        // squashed later, fused into the product that consumes it.

        // Carry the recurrent state in from the previous chunk when the surrounding
        // sweep is chunked; `None` (the unchunked case) starts it at zero inside
        // the kernel. The outgoing state is taken below, after the cache is built.
        let fused = ops::mlstm_fused_fw(
            gpu,
            &xh,
            &gates,
            l,
            if self.carry {
                self.carry_state.as_ref()
            } else {
                None
            },
            shape,
        );
        if self.carry {
            self.carry_state = Some(fused.final_state(gpu, shape.bh(), self.dhv, self.dqk));
        }
        // `ytil` comes out `[rows, d]` already; the head norm only needs it widened.
        let mut h_tilde = cache.temps.get::<f32>(gpu, &[rows, dim]);
        let mut yhat = GTensor::uninit(gpu, &[rows, dim]);
        self.headnorm
            .forward(gpu, fused.ytil.as_f32(gpu, &mut h_tilde), &mut yhat);
        drop(h_tilde);
        // `hconcat` is kept in the cache rather than pooled, and `lin_out` takes it
        // back through `backward_with_x`: a chunked sweep would otherwise have the
        // next chunk's forward overwrite the private copy `forward` saves.
        let mut hconcat = GTensor::uninit(gpu, &[rows, dim]);
        ops::ogate_fwd(gpu, &xh, &yhat, &mut hconcat, self.o_off());
        out.reshape_to(&[rows, dim]);
        self.lin_out.forward_shared(gpu, &hconcat, out);
        out.reshape_to(&[batch, t, dim]);
        self.saved.push(SavedFused {
            b: batch,
            t,
            xh: Some(xh),
            gates,
            fused,
            yhat: Some(yhat),
            hconcat: Some(hconcat),
        });
        self.evict_saved(gpu);
    }

    /// Park this cell's saved activations on the host, if offload is enabled.
    ///
    /// Called at the end of a fused forward. Only the three large per-`N` tensors ride:
    /// `o`/`yhat` (4 MB each at the backbone's shape) and the `q‖k‖v` slab (6 MB
    /// on the bf16 path). The rest (`gates` at 128 KB,
    /// and everything inside `fused`) is left resident: each is two orders of
    /// magnitude smaller, so moving it would add PCIe traffic and bookkeeping for
    /// nothing.
    ///
    /// The slab parks at **its own width** — a bf16 slab comes back bf16, since the
    /// precision split belongs at each value's production point, not here.
    fn evict_saved(&mut self, gpu: &Gpu) {
        use super::offload::Parked;
        let Some(park) = &mut self.park else { return };
        // The chunk just forwarded — the one this call is closing out.
        let Some(sv) = self.saved.last_mut() else {
            return;
        };
        // Fixed order, mirrored exactly by `restore_saved`.
        park.evict(
            gpu,
            vec![
                Parked::from(sv.yhat.take().expect("evict before restore: yhat")),
                Parked::from(sv.xh.take().expect("evict before restore: q‖k‖v‖o")),
            ],
        );
    }

    /// Start the parked activations on their way back, without waiting. Called one
    /// block ahead of this cell's backward so the upload overlaps compute.
    pub fn prefetch_saved(&mut self, gpu: &Gpu) {
        if let Some(park) = &mut self.park {
            park.prefetch(gpu);
        }
    }

    /// Put the parked activations back into the cache, in `evict_saved`'s order.
    fn restore_saved(&mut self, gpu: &Gpu) {
        let Some(park) = &mut self.park else { return };
        // Backward pops from the end, so the cache being refilled is the last one —
        // matching the park's own LIFO restore order.
        let Some(sv) = self.saved.last_mut() else {
            return;
        };
        let mut it = park.restore(gpu).into_iter();
        let mut next = |what: &str| it.next().expect(what);
        sv.yhat = Some(next("parked yhat").f32());
        sv.xh = Some(next("parked q‖k‖v‖o").into());
    }

    /// Device bytes held, split `(params, activations)`. Diagnostic — see
    /// [`Hierarchical::retained_report`](super::hierarchical::Hierarchical::retained_report).
    ///
    /// Note the multiplier: this cell owns **three** [`Linear`]s, each with its own
    /// saved input and bf16 GEMM staging, plus a head-wise [`RmsNorm`] holding an
    /// `x̂` the width of the cell. `drop_saved_act` clears only `saved` — the
    /// per-`Linear` retention survives it.
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
        act += self.saved_bytes();
        (params, act)
    }

    /// Continue the previous call's recurrence rather than starting from zero.
    ///
    /// For a chunked sweep: the state `C`/`n`/`m` crosses the chunk borders, which is
    /// what makes the split reproduce the unchunked recurrence (see
    /// `mlstm_chunked_carry_matches_whole`). Clear it — or call
    /// [`reset_state`](Self::reset_state) — before the first chunk of a sweep.
    pub fn set_carry(&mut self, carry: bool) {
        self.carry = carry;
        self.headnorm.set_carry(carry);
        if !carry {
            self.carry_state = None;
            self.carry_dstate = None;
        }
    }

    /// Drop the carried state, so the next forward starts the recurrence at zero
    /// whatever `carry` says.
    pub fn reset_state(&mut self, _gpu: &Gpu) {
        self.carry_state = None;
    }

    /// Drop the carried BPTT state, so the next backward starts with no incoming
    /// gradient from the right. Call before the rightmost chunk's backward.
    pub fn reset_bptt(&mut self, _gpu: &Gpu) {
        self.carry_dstate = None;
    }

    /// Drop the forward caches, leaving the pool and the weight staging alone —
    /// for a caller that forwards repeatedly and never backwards.
    pub fn drop_saved(&mut self) {
        self.saved.clear();
    }

    /// Retained activation bytes split `(saved_cache, other)`.
    ///
    /// `saved_cache` is what `drop_saved_act` releases. `other` is the pooled scratch
    /// plus what the **three** projections and the head norm hold internally — their
    /// saved inputs and bf16 GEMM staging — which no `drop_saved_act` reaches.
    pub fn act_split(&self) -> (usize, usize) {
        let saved = self.saved_bytes();
        let (_, all) = self.retained_bytes();
        (saved, all - saved)
    }

    /// Device bytes held by the forward caches, summed over every chunk still awaiting
    /// its backward.
    fn saved_bytes(&self) -> usize {
        self.saved.iter().map(SavedFused::retained_bytes).sum()
    }

    /// Release everything a forward left behind that no backward will read: the
    /// saved cache, and each projection's saved input and bf16 staging. The broader companion to the `Cell::drop_saved_act`, which only
    /// clears `saved`.
    pub fn drop_all_act(&mut self, gpu: &Gpu) {
        self.saved.clear();
        for l in [&mut self.lin_qkvo, &mut self.lin_gates, &mut self.lin_out] {
            l.drop_saved_act(gpu);
        }
        self.headnorm.drop_saved_act();
        // The bf16 staging scratch is process-wide and is NOT released here: this runs
        // per block, mid-sweep, so freeing it would pull the buffer out from under the
        // blocks still to come. `Hierarchical::drop_all_act` clears it once, at the
        // point where the whole stack is done.
    }

    /// Park this cell's saved activations on the host between forward and backward.
    ///
    /// Opted into by the surrounding [`Block`](super::block::Block), and subject to the
    /// same constraint: only for a stack whose whole forward precedes its backward.
    /// See `Block::enable_offload`.
    pub fn enable_offload(&mut self, gpu: &Gpu, in_flight: super::offload::SharedInFlight) {
        self.park =
            Some(super::offload::HostPark::new(gpu, in_flight).expect("offload: host park"));
    }

    /// Backward over `[B, T, d]` → `dx` `[B, T, in]`. Accumulates all grads.
    ///
    /// Chunks are swept in reverse, carrying `dC`/`dn` (the grad wrt the state a
    /// chunk hands to its successor) backwards the way forward carried `C`/`n`
    /// forwards — BPTT over chunks, with the parallel form inside each.
    pub fn backward(
        &mut self,
        gpu: &Gpu,
        x: &GTensor<f32>,
        dy: &GTensor<f32>,
        dx: &mut GTensor<f32>,
        cache: &TrainingCache,
    ) {
        // Bring the parked activations back into the cache first, so everything below
        // reads them as if they had never left. No-op unless offload is enabled.
        self.restore_saved(gpu);
        // `take`, not `as_ref`: the cache holds a window's activations, and dropping
        // them at the end of this call (rather than when the next forward overwrites
        // the field) keeps them from staying resident across the optimizer step.
        let sv = self.saved.pop().expect("MLstm::backward before forward");
        self.backward_fused(gpu, x, dy, sv, dx, cache)
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

    /// Backward of the fused path: two kernels for the whole chunkwise core, with
    /// the same projection/head-norm/o-gate shell around it.
    fn backward_fused(
        &mut self,
        gpu: &Gpu,
        x: &GTensor<f32>,
        dy: &GTensor<f32>,
        sv: SavedFused,
        dx: &mut GTensor<f32>,
        cache: &TrainingCache,
    ) {
        let (d, h, dqk, dhv, inp) = (self.d, self.heads, self.dqk, self.dhv, self.input_size);
        let (b, t) = (sv.b, sv.t);
        let n = b * t;
        assert!(
            n * inp <= dx.capacity(),
            "MLstm::backward — dx holds {} elements, needs {}",
            dx.capacity(),
            n * inp
        );

        // The whole shell is temporaries: each value below dies as soon as the next
        // op has read it, so all of them come from the pool and go back.
        // `dy` is `[B, T, d]` and `lin_out` wants `[N, d]` — the same elements, so a
        // view rather than a copy into scratch of its own.
        let dy_flat = GTensor::view(gpu, &dy.buf, 0, &[n, d]);
        let mut d_hconcat = cache.temps.get::<f32>(gpu, &[n, d]);
        self.lin_out
            .backward_with_x(gpu, sv.hconcat(), &dy_flat, &mut d_hconcat, cache);
        // `dq‖dk‖dv‖do` is one buffer: `ogate_bwd` fills the `o` block here and
        // `mlstm_fused_bw` the other three below, between them writing every column,
        // so it comes in uninitialised and feeds `lin_qkvo`'s backward whole.
        let mut dxh = cache.temps.get::<f32>(gpu, &[n, self.lin_qkvo.output_size()]);
        let mut d_yhat = cache.temps.get::<f32>(gpu, &[n, d]);
        ops::ogate_bwd(
            gpu,
            &d_hconcat,
            sv.xh(),
            sv.yhat(),
            &mut dxh,
            &mut d_yhat,
            self.o_off(),
        );
        drop(d_hconcat);
        // No gather: the kernels read `d_ytil` through the same position-major strides
        // they wrote `ytil` with, and the head norm's output is already in that layout.
        let mut d_ytil = cache.temps.get::<f32>(gpu, &[n, d]);
        // `yhat` is the head norm's own output, kept for `ogate_bwd` — which is also
        // what its backward rebuilds `x̂` from, so the norm stores no `[N, d]` of its own.
        self.headnorm.backward(gpu, &d_yhat, sv.yhat(), &mut d_ytil, cache);
        drop(d_yhat);

        // `[N, 2·heads]` — a gate strip, not a rectangle, so it comes from the small
        // array rather than spending a stage-sized slot on 64 KB.
        let mut dgates = cache.temps.get_small::<f32>(gpu, &[n, 2 * h]);
        // Backward unwinds chunks right to left, so the carried BPTT state comes from
        // the chunk to the RIGHT — `None` on the rightmost (and on any unchunked call).
        let dstate = ops::mlstm_fused_bw(
            gpu,
            &sv.fused,
            sv.xh(),
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

        // No scatter and no split: the kernel wrote `dq`/`dk`/`dv` into the one
        // `[N, H*(2*dqk+dhv)]` buffer through the same strides the projection wrote
        // q/k/v with, so it feeds `lin_qkvo`'s backward whole. `dk` needs no 1/√dqk
        // either — the factor lives in `lin_qkvo`'s k columns, so it is already in
        // their scale.
        //
        // dx is the sum of the three projection backwards, accumulated into one
        // buffer with one pooled scratch — not a fresh [N, in] per term. The
        // accumulator is the caller's `dx` seen as `[N, in]`, so the sum lands where
        // it is wanted instead of in scratch that then has to be copied out.
        //
        // All three read the one `x` the caller kept from the forward, so they take
        // `backward_with_x` rather than each consulting a private saved copy. It is
        // narrowed once for all three, as in forward.
        let mut acc = GTensor::view(gpu, &dx.buf, 0, &[n, inp]);
        let mut part = cache.temps.get::<f32>(gpu, &[n, inp]);
        let xf = GTensor::view(gpu, &x.buf, 0, &[n, inp]);
        let mut xf_b = cache.temps.get::<u16>(gpu, xf.dims());
        xf_b.store(gpu, &xf);
        self.lin_qkvo
            .backward_staged_x(gpu, &xf, &xf_b, &dxh, &mut acc, cache);
        self.lin_gates
            .backward_staged_x(gpu, &xf, &xf_b, &dgates, &mut part, cache);
        ops::add_assign(gpu, &mut acc, &part);
        drop((acc, xf_b, part, dxh, dgates));
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
    fn forward(
        &mut self,
        gpu: &Gpu,
        x: &GTensor<f32>,
        out: &mut GTensor<f32>,
        cache: &TrainingCache,
    ) {
        MLstm::forward(self, gpu, x, out, cache)
    }
    /// The head norm keeps its own output (`yhat`, for the o-gate), so this cell has
    /// no use for the block's copy.
    fn backward(
        &mut self,
        gpu: &Gpu,
        x: &GTensor<f32>,
        _y: &GTensor<f32>,
        dy: &GTensor<f32>,
        dx: &mut GTensor<f32>,
        cache: &TrainingCache,
    ) {
        MLstm::backward(self, gpu, x, dy, dx, cache)
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
    fn enable_offload(&mut self, gpu: &Gpu, in_flight: super::offload::SharedInFlight) {
        MLstm::enable_offload(self, gpu, in_flight)
    }
    fn prefetch_act(&mut self, gpu: &Gpu) {
        MLstm::prefetch_saved(self, gpu)
    }
    fn trim_to(&mut self, _rows: usize) {}
    fn drop_saved_act(&mut self) {
        // The whole fused cache — the q‖k‖v slab, o, yhat and the `MlstmFused`
        // internals. Safe to drop wholesale because the only caller re-forwards to
        // rebuild it (see `Block::drop_saved_act`).
        self.saved.clear();
    }
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
    fn reset_bptt(&mut self, gpu: &Gpu) {
        MLstm::reset_bptt(self, gpu)
    }
    fn to_nn_block(
        &self,
        gpu: &Gpu,
        hidden: usize,
        up: usize,
        pre_norm1: crate::nn::rms_norm::RMSNorm,
        pre_norm2: crate::nn::rms_norm::RMSNorm,
        lin_gate: crate::nn::linear::LinearLayer,
        lin_value: crate::nn::linear::LinearLayer,
        lin_down: crate::nn::linear::LinearLayer,
    ) -> Box<dyn crate::nn_layer::NnLayer> {
        // mLSTM blocks have no post-cell norm (the cell's head norm normalizes).
        Box::new(crate::nn::mlstm_block::MLSTMBlock::from_loaded(
            hidden,
            up,
            pre_norm1,
            pre_norm2,
            self.to_nn_cell(gpu),
            lin_gate,
            lin_value,
            lin_down,
        ))
    }
}

#[cfg(test)]
mod tests {

    /// One temp cache per test, sized past every shape this module presents — the
    /// widest is `[b·t, heads·(2·dqk + 2·dhv)]` at `b·t = 400`, `dqk = dhv = 96`.
    /// The chunk array is deliberately generous here: these tests drive `set_chunk`
    /// down to 1, where the per-chunk state arrays hold `nc == t` states instead of the
    /// `t/32` a real run sees. Production sizes it from the config — see
    /// `Hierarchical::temp_chunk_elems`.
    fn test_cache(gpu: &Gpu) -> TrainingCache {
        TrainingCache::new(gpu, 1 << 20, 1 << 16, 1 << 24)
    }
    use super::*;
    use crate::nn2::mlstm::MLstm as CpuMLstm;
    use crate::nn2::optim::AdamCfg;

    fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length mismatch");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < tol, "{what}[{i}]: gpu {g} vs cpu {w}");
        }
    }

    /// Worst absolute difference measured against the tensor's own magnitude.
    ///
    /// Column block `[at, at + cols)` of a `[·, w]` host matrix.
    ///
    /// The cell holds q/k/v in one projection and ĩ/f̃ in another; the CPU cell it is
    /// compared against keeps all six apart, so the parity checks cut a part back out.
    fn cols_of(m: &[f32], w: usize, at: usize, cols: usize) -> Vec<f32> {
        m.chunks(w)
            .flat_map(|r| r[at..at + cols].iter().copied())
            .collect()
    }

    /// Lay separate `[B, T, ·]` q/k/v out the way the fused projection writes them —
    /// concatenated into `[B*T, H*(2*dqk + 2*dhv)]`, with `H = 1` for these tests.
    ///
    /// The trailing `o` block is zero-filled: nothing in the fused kernels reads it,
    /// but it sets the row stride they address q/k/v through, so it has to be there.
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

    /// The checkpoint round-trip must put every weight back byte for byte.
    ///
    /// `q ‖ k ‖ v` and `ĩ ‖ f̃` live as one matrix each at runtime while the
    /// checkpoint format keeps all six apart, so `to_nn_cell` cuts them up and
    /// `from_nn_cell` stitches them back. A wrong offset there corrupts a saved model
    /// silently — the shapes still fit, the loss still falls, and only the parts sit
    /// in each other's columns. Exact equality is the right bar: the split, the
    /// concatenation and the 1/√dqk fold/unfold are all lossless.
    ///
    /// Head dims are deliberately unequal (`dqk != dhv`) so a check that only ever
    /// sees square parts cannot pass by accident.
    #[test]
    fn nn_cell_round_trip_preserves_the_fused_projections() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let (inp, d, heads, dqk) = (12usize, 24usize, 4usize, 5usize);
        let before = MLstm::new_rand(&gpu, inp, d, heads, dqk);
        let after = MLstm::from_nn_cell(&gpu, &before.to_nn_cell(&gpu));
        for (name, a, b) in [
            ("qkvo w", &before.lin_qkvo.w, &after.lin_qkvo.w),
            ("qkvo b", &before.lin_qkvo.b, &after.lin_qkvo.b),
            ("gates w", &before.lin_gates.w, &after.lin_gates.w),
            ("gates b", &before.lin_gates.b, &after.lin_gates.b),
            ("out w", &before.lin_out.w, &after.lin_out.w),
        ] {
            let (a, b) = (a.to_host(&gpu), b.to_host(&gpu));
            assert_eq!(a.dims(), b.dims(), "{name}: shape changed");
            assert_eq!(a.data, b.data, "{name}: round trip is not the identity");
        }
    }

    /// Single-chunk parallel forward+backward+step must match the CPU scalar
    /// recurrence (`nn2::MLstm`) from identical weights. The CPU backward is
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
    ///
    /// This is the tighter of the two mLSTM tests: it pins the inter-chunk state
    /// carry and its BPTT, which `mlstm_matches_cpu` (T < L, single chunk) never
    /// reaches.
    #[test]
    fn mlstm_chunking_matches_single_chunk() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        let (b, t, inp, d, heads, dqk) = (2, 20, 5, 8, 2, 4);

        let mut proto = CpuMLstm::new(inp, d, heads, dqk);
        // Non-trivial gate weights so the decay/stabilizer path is exercised — with
        // zero gates every chunk would decay identically and the test would pass
        // vacuously.
        // Seeded for the same reason as `mlstm_fused_chunking_matches_unit_chunk`: this compares
        // two chunk blockings whose sums reassociate, so it is a tolerance test and
        // must not redraw its data every run.
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

        // Both sides store q/k/v and ytil bf16, but a different chunking quantizes
        // at different points, so the two do not cancel — the floor is bf16's, and
        // Adam then amplifies it on near-zero-gradient weights (`lr·ĝ/(√v̂+ε)` is
        // scale-invariant in the gradient). `GPU_NO_BF16=1` restores the fp32
        // tolerances. Measured drift without this was ~3.5e-3 relative on `w_out`,
        // intermittently over the old 1e-3 bound.
        let slab = if gpu.kernels.slab_bf16 { 8.0 } else { 1.0 };
        let (y0, dx0, wf0, wo0) = run(0); // single chunk
        for l in [1, 3, 8, 16, 32] {
            let (y, dxo, wf, wo) = run(l);
            assert_close(&y, &y0, 2e-4 * slab, &format!("y (chunk {l})"));
            assert_close(&dxo, &dx0, 2e-4 * slab, &format!("dx (chunk {l})"));
            // Scale-relative: post-Adam weights are ~1e-2, so an absolute 2e-6 is a
            // 1e-4 relative demand that chunk-boundary reassociation cannot meet.
            assert_close_rel(&wf, &wf0, 1e-3 * slab, &format!("wf (chunk {l})"));
            assert_close_rel(&wo, &wo0, 1e-3 * slab, &format!("w_out (chunk {l})"));
        }
    }

    /// Chunked (L = 32) vs step-by-step (L = 1), at a shape wide enough to matter.
    ///
    /// The chunk length is a blocking choice with no effect on the result, so the two
    /// must agree — what is on trial is the inter-chunk state carry and the ragged
    /// final chunk. The other tests run at toy dims (dqk = dhv = 4, T = 20), where a
    /// fused block uses a few hundred bytes of shared memory and most of its threads
    /// idle. This one runs dqk = dhv = 64 over a T that spans many chunks and ends on
    /// a SHORT one (T = 200 = 6·32 + 8), which is what exercises the shared-memory
    /// staging, the block-wide max/sum reductions, and the `len` masking. A bug in
    /// any of those is invisible at the toy dims.
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

        // Seeded: this is a tolerance comparison between two float orderings, so
        // unseeded data made it pass or fail by luck (~3 of 5 runs failed on a
        // different element each time). See `Tensor::random_seeded`.
        let mut proto = CpuMLstm::new(inp, d, heads, dqk);
        proto.wi = Tensor::random_seeded(&[inp, heads], 0.3, 0xA1);
        proto.wf = Tensor::random_seeded(&[inp, heads], 0.3, 0xA2);

        let x = GTensor::from_host(&gpu, &Tensor::random_seeded(&[b, t, inp], 0.5, 0xA3));
        let g = GTensor::from_host(&gpu, &Tensor::random_seeded(&[b, t, d], 1.0, 0xA4));

        // `chunk` here selects the path, not just the blocking: 0 is the only value
        // the fused kernels decline, so it is the reference. See `fused_chunk`.
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

        // L = 1 degenerates the chunkwise form to a pure step-by-step recurrence (no
        // intra-chunk parallel term at all); L = 32 is the real blocking. The state
        // carry is the only thing that can make them differ.
        let (y0, dx0, wf0, wo0) = run(1);
        let (y1, dx1, wf1, wo1) = run(32);

        // Both sides keep their saved q/k/v and ytil in bf16 (the reference's DTYPE
        // tensors), so the floor is bf16's half-ulp (2^-8) rather than fp32
        // reassociation, and both sides' dots are on the tensor cores. `GPU_NO_BF16=1`
        // puts the slabs back on fp32 and the original tolerances apply.
        let slab = if gpu.kernels.slab_bf16 { 8.0 } else { 1.0 };
        assert_close(&y1, &y0, 2e-3 * slab, "y");
        assert_close(&dx1, &dx0, 2e-3 * slab, "dx");
        // The weights come out of an Adam step, so they sit at ~1e-2 while the two
        // paths' GEMMs reassociate at ~1e-3 relative. An ABSOLUTE 2e-5 on values of
        // that size is really a 4e-4 relative demand, tighter than summation order
        // guarantees — which is what made this test intermittent. Compare against
        // the tensor's own scale.
        // Adam divides by √v̂, which amplifies a small gradient difference on a
        // near-zero-gradient weight, so the weight check needs more headroom than
        // the activations above.
        assert_close_rel(&wf1, &wf0, 5e-3 * slab, "wf");
        assert_close_rel(&wo1, &wo0, 5e-3 * slab, "w_out");
    }

    /// The fused forward and backward against the CPU scalar recurrence at a shape
    /// with a SHORT final chunk (T = 200 = 6·32 + 8).
    ///
    /// What is on trial is the ragged tail: the mma fragment loaders read whole
    /// 16x8x16 blocks, so every buffer is zero-filled past `len` and the pad rows must
    /// contribute nothing. A wrong fragment index or a missed zero-fill garbles the
    /// result outright rather than by 1e-3, so the tolerance is loose on purpose —
    /// the operands are bf16, and the dots accumulate over 200 timesteps.
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
    ///
    /// The other CPU checks run `dqk = 4`, where `MLSTM_KT`/`MLSTM_VT` exceed the head
    /// dim and both slice loops are a single pass — so they exercise none of the
    /// head-dim tiling. Here `dqk = dhv = 96` gives NKT = NVT = 3, and the CPU
    /// recurrence is an independent implementation rather than a second GPU path.
    #[test]
    fn mlstm_fused_tiled_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        // T spans several fused chunks (FUSED_MAX_L = 32) so the state carry is
        // covered too; dqk = dhv = 96 is the backbone's width.
        let (b, t, inp, d, heads, dqk) = (2, 70, 12, 768, 8, 96);

        let mut cpu = CpuMLstm::new(inp, d, heads, dqk);
        cpu.wi = Tensor::random_seeded(&[inp, heads], 0.3, 0xD1);
        cpu.wf = Tensor::random_seeded(&[inp, heads], 0.3, 0xD2);
        let mut dev = MLstm::from_cpu(&gpu, &cpu);

        let x = Tensor::random_seeded(&[b, t, inp], 0.5, 0xD3);
        let g = Tensor::random_seeded(&[b, t, d], 1.0, 0xD4);

        // Relative, not absolute: at this width `y` lands around 2e-4, so an absolute
        // 5e-3 would be ~20x the signal and pass on nearly any output.
        //
        // The bound is bf16's half-ulp (2^-8 ~ 3.9e-3) with room for the accumulated
        // rounding on top: the fused path stores q/k/v and ytil narrow, so a tighter
        // relative bound is measuring the storage format, not the kernel. Sitting at
        // 5e-3 made this fail ~2 runs in 3 at 5.07e-3.
        let y_cpu = cpu.forward(&x);
        let xd = GTensor::from_host(&gpu, &x);
        let mut y_dev = GTensor::uninit(&gpu, &[b, t, d]);
        dev.forward(&gpu, &xd, &mut y_dev, &tc);
        assert_close_rel(&y_dev.to_host(&gpu).data, &y_cpu.data, 1e-2, "y");

        // dx is looser than y: it carries the bf16 q/k/v narrowing and the TF32 mma
        // rounding of every backward contraction, against an fp32 CPU recurrence.
        // The injected-bug check that justifies this test lands at 2.2e-1, two orders
        // above either bound.
        // dx is looser than y: it carries the bf16 q/k/v narrowing and the TF32 mma
        // rounding of every backward contraction, against an fp32 CPU recurrence.
        // The injected-bug check that justifies this test lands at 2.2e-1, an order
        // above either bound.
        //
        // Weights after an Adam step are deliberately NOT compared: the update is
        // `lr·g/(√(g²)+ε)`, which normalizes away the gradient's magnitude and turns
        // a last-bit difference into a full-size one — 3.1e-2 here. `dx` is the
        // gradient check; a weight check on top of it only measures the optimizer.
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
    ///
    /// This is the load-bearing property of the chunked sweep, tested at the kernel
    /// level so a failure points at the `CARRY` seeding in `mlstm_fw_gates` /
    /// `mlstm_state_scan` rather than at anything layered above it. `m` is the stabilizer: seeding it wrong does not
    /// crash and produces no NaN, it silently rescales every value in the chunk — so
    /// comparing `ytil` (the kernel's output) across the split is the only thing that
    /// actually catches it.
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
        // bf16 `ytil` storage, so this is a storage-precision comparison, not an exact
        // one. What would signal a broken carry is a whole chunk being off — a
        // stabilizer error is multiplicative and hits every element after the border.
        assert_close(&got, &want, 3e-2, "chunked ytil vs whole");
    }

    /// The chunked **backward** must reproduce the whole-sequence one.
    ///
    /// `mlstm_chunked_carry_matches_whole` only compares the forward, and that gap hid
    /// a real bug: `mlstm_bw_parallel` forced the last chunk's incoming state gradient
    /// to zero (`is_last`), which is right for a whole sequence but wrong under CARRY,
    /// where that chunk feeds the chunk to its right. The gradient stayed plausible and
    /// the loss stayed right — only `dW` was off, and the error grew with the number of
    /// chunk borders (measured 1.2% across one, 2.6% across two).
    ///
    /// A tolerance test: the two blockings sum the same terms in a different order.
    /// `dx` is the observable, not `dW` — the projection gradients average the error
    /// down to a factor of two, while `dx` separates the two states by ~700x.
    #[test]
    fn mlstm_chunked_backward_matches_whole() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let tc = test_cache(&gpu);
        let (inp, d, heads, dqk, b, t) = (8usize, 16usize, 2usize, 8usize, 1usize, 12usize);
        // Seeded: this compares two blockings whose sums reassociate, so it must not
        // redraw its instance every run.
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

        // Reference: one call over the whole sequence.
        // Internal chunk length 2, so each call has NC > 1. At the default (256) a
        // 6-step call is a single internal chunk, `is_last` is true for it either way,
        // and the CARRY path under test is never reached.
        let mut whole = MLstm::from_cpu(&gpu, &proto);
        let xd = GTensor::from_host(&gpu, &x);
        let mut y_whole = GTensor::uninit(&gpu, &[b, t, d]);
        whole.forward(&gpu, &xd, &mut y_whole, &tc);
        let want = whole
            .backward_alloc(&gpu, &xd, &GTensor::from_host(&gpu, &g), &tc)
            .to_host(&gpu)
            .data
            .to_vec();

        // Scale-aware: `dW` has near-zero entries where a pointwise relative error
        // says nothing.
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
            // Each chunk's input outlives its forward: backward reads it again for
            // the projections' `dW`.
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
            // Measured on this instance: 1e-3 with the carry honoured, 0.69 without —
            // the threshold sits between, far from both.
            assert!(
                e < 1e-2,
                "chunked backward dx differs by {e} over {} chunks",
                parts.len()
            );
        }
    }
}
