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
//! The six projections and `W_out` are `gpu::Linear`; only the attention core is
//! bespoke kernels + strided-batched GEMM, on the head-major `[B*H, T, ·]` layout.
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
//! A sequence already shorter than `L` (the encoder/decoder, where T is a word
//! length) takes the single-chunk path with no inter-chunk work at all.

use std::sync::OnceLock;

use super::block::Cell;
use super::{DTensor, Gpu, linear::Linear, ops, rms_norm::RmsNorm, Pool};
use crate::nn2::optim::AdamCfg;
use crate::tensor::Tensor;

/// Chunk length: `config::MLSTM_CHUNK`, overridable with `MLSTM_CHUNK=<L>` for A/B
/// runs (0 = single-chunk). Resolved once — the env read must not sit in forward.
fn chunk_len() -> usize {
    static L: OnceLock<usize> = OnceLock::new();
    *L.get_or_init(|| {
        std::env::var("MLSTM_CHUNK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(crate::config::MLSTM_CHUNK)
    })
}

/// Per-chunk forward intermediates.
///
/// The two [BH, L, L] decay matrices (D̄ and DS) are the largest tensors here, and
/// they are simply kept. Before chunking they were [BH, T, T] — 134 MB *each, per
/// block* at 2048 words — which is why they used to be recomputed in backward
/// instead; chunking makes them small enough (2·BH·L² floats per chunk) that
/// caching costs ~64 MB across the whole backbone and saves the rebuild GEMM.
///
/// Everything else is at most [BH, L, dhv].
struct Chunk {
    c0: usize,     // chunk start in T
    len: usize,    // chunk length (the last chunk may be short)
    bvec: DTensor, // [BH, L]  b_t: carried state → row t
    avec: DTensor, // [BH, L]  a_j: row j → outgoing state
    qn: DTensor,   // [BH, L]  (intra + inter)
    psi: DTensor,  // [BH, L]
    /// Per-row stabilizer. Backward needs it because ψ = max(|qn|, exp(−m)), so the
    /// branch "did qn win the max?" is against exp(−m), not against 1.
    m: DTensor, // [BH, L]
    num: DTensor,  // [BH, L, dhv]  (intra + inter)
    dbar: DTensor, // [BH, L, L]  D̄
    ds: DTensor,   // [BH, L, L]  D̄⊙S
    /// The carried state entering this chunk, and the inter-chunk products it
    /// produced. `None` on the first chunk, where the state is zero and the whole
    /// inter path is skipped — that makes a single-chunk sequence take exactly the
    /// pre-chunking code path, with no extra GEMMs.
    inter: Option<Inter>,
}

/// The inter-chunk half of a chunk: incoming state plus the two products read out
/// of it (both needed to form `db` in backward).
///
/// `m_prev` is deliberately absent: the stabilizer is held constant in backward
/// (the reference approximation, as on the CPU and in the sLSTM), so no gradient
/// flows through it — it is only ever a forward input.
struct Inter {
    c_prev: DTensor,    // [BH, dhv, dqk]
    n_prev: DTensor,    // [BH, 1, dqk]
    inter_num: DTensor, // [BH, L, dhv]   Q·C_prevᵀ   (pre-b)
    inter_qn: DTensor,  // [BH, L, 1]     Q·n_prevᵀ   (pre-b)
}

/// Forward intermediates retained for the backward pass.
struct Saved {
    b: usize,
    t: usize,
    /// The flat `[N, in]` input, shared by all six projections (see `forward_alloc`).
    /// Held here rather than six times inside the `Linear`s.
    xf: DTensor,
    qh: DTensor, // [BH, T, dqk]
    kh: DTensor, // [BH, T, dqk]  (already ×1/√dqk)
    vh: DTensor, // [BH, T, dhv]
    // The forget-gate logit is still needed in backward (`revcumsum_dlogsig` chains
    // dfc through logσ'); the input-gate logit is not — it only ever fed the D̄
    // build, which now happens once, in forward.
    fgh: DTensor, // [BH, T]  forget-gate logit (head-major)
    chunks: Vec<Chunk>,
    o: DTensor,    // [N, d]  (post-sigmoid)
    yhat: DTensor, // [N, d]
    /// `lin_out`'s input, kept here rather than as the projection's private copy so a
    /// chunked sweep's later chunk cannot overwrite it. Fed back via `backward_with_x`.
    hconcat: DTensor, // [N, d]
}

/// Forward intermediates of the **fused** path. Far smaller than `Saved`: the
/// per-chunk `[BH, L, L]` decay matrices do not exist — backward rebuilds them in
/// shared memory (see `ops::MlstmFused`).
struct SavedFused {
    b: usize,
    t: usize,
    /// The six large per-`N` tensors, `None` exactly while they are parked on the host.
    ///
    /// `Option` rather than a `parked: bool` flag: backward reads these through
    /// `expect`, so a missing `restore_saved` is a named panic at the use site instead
    /// of a silent read of a stale buffer. They are `Some` for the whole of a
    /// non-offloaded run.
    ///
    /// The flat `[N, in]` input `xf` is shared by all six projections (see
    /// `forward_alloc`) — held here rather than six times inside the `Linear`s.
    xf: Option<DTensor>,
    // bf16 storage, mirroring the reference's DTYPE tensors (matQ/matK/matV).
    qh: Option<ops::SlabBuf>,
    kh: Option<ops::SlabBuf>,
    vh: Option<ops::SlabBuf>,
    // fp32: gate logits. The reference loads vecI/vecB `.to(tl.float32)` — they are
    // exponents feeding the stabilizer, where an absolute error becomes a
    // multiplicative one. See `gpu::bf16`.
    // Left resident when parking: 64 KB each against the 4 MB tensors above, so
    // moving them would cost bookkeeping and PCIe for nothing.
    igh: DTensor,
    fgh: DTensor,
    fused: ops::MlstmFused,
    o: Option<DTensor>,
    yhat: Option<DTensor>,
    /// `lin_out`'s input, kept here rather than as the projection's private copy so a
    /// chunked sweep's later chunk cannot overwrite it. Fed back via `backward_with_x`.
    hconcat: Option<DTensor>,
}

impl SavedFused {
    /// The six parkable tensors, in the one order `evict`/`restore` both use.
    ///
    /// Panics if they are parked — every reader runs after `restore_saved`.
    fn xf(&self) -> &DTensor {
        self.xf.as_ref().expect("mLSTM: xf is parked on the host")
    }
    fn o(&self) -> &DTensor {
        self.o.as_ref().expect("mLSTM: o is parked on the host")
    }
    fn yhat(&self) -> &DTensor {
        self.yhat.as_ref().expect("mLSTM: yhat is parked on the host")
    }
    fn hconcat(&self) -> &DTensor {
        self.hconcat
            .as_ref()
            .expect("mLSTM: hconcat is parked on the host")
    }
    fn qh(&self) -> &ops::SlabBuf {
        self.qh.as_ref().expect("mLSTM: qh is parked on the host")
    }
    fn kh(&self) -> &ops::SlabBuf {
        self.kh.as_ref().expect("mLSTM: kh is parked on the host")
    }
    fn vh(&self) -> &ops::SlabBuf {
        self.vh.as_ref().expect("mLSTM: vh is parked on the host")
    }

    /// Device bytes held. Parked tensors count as zero — they are on the host, which
    /// is the whole point of parking them.
    fn retained_bytes(&self) -> usize {
        let opt_f32: usize = [&self.xf, &self.o, &self.yhat, &self.hconcat]
            .iter()
            .filter_map(|s| s.as_ref())
            .map(|t| t.capacity() * 4)
            .sum();
        let slabs: usize = [&self.qh, &self.kh, &self.vh]
            .iter()
            .filter_map(|s| s.as_ref())
            .map(|s| s.retained_bytes())
            .sum();
        opt_f32
            + slabs
            + (self.igh.capacity() + self.fgh.capacity()) * 4
            + self.fused.retained_bytes()
    }
}

impl Saved {
    /// Device bytes held by the legacy (op-at-a-time) cache. Only reachable under
    /// `MLSTM_LEGACY=1`; the per-chunk `[BH, L, L]` matrices dominate it, which is
    /// why the fused path exists.
    fn retained_bytes(&self) -> usize {
        let flat: usize = [
            &self.xf,
            &self.qh,
            &self.kh,
            &self.vh,
            &self.fgh,
            &self.o,
            &self.yhat,
            &self.hconcat,
        ]
            .iter()
            .map(|t| t.capacity() * 4)
            .sum();
        let chunks: usize = self
            .chunks
            .iter()
            .map(|c| {
                [
                    &c.bvec, &c.avec, &c.qn, &c.psi, &c.m, &c.num, &c.dbar, &c.ds,
                ]
                .iter()
                .map(|t| t.capacity() * 4)
                .sum::<usize>()
            })
            .sum();
        flat + chunks
    }
}

/// Which forward ran, and hence which backward must.
enum Cache {
    Fused(SavedFused),
    Legacy(Saved),
}

/// `MLSTM_LEGACY=1` forces the op-at-a-time chunk loop — the A/B baseline for
/// `mlstm_fused_bench`, and the escape hatch if a fused kernel ever misbehaves.
fn legacy_forced() -> bool {
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| std::env::var("MLSTM_LEGACY").is_ok())
}

pub struct MLstm {
    pub input_size: usize,
    pub d: usize,
    pub heads: usize,
    pub dqk: usize,
    pub dhv: usize,
    inv_sqrt_dqk: f32,
    /// Chunk length (0 = single-chunk). Defaults to [`chunk_len`].
    chunk: usize,

    // Projections (in → ·) and the output projection (d → d). Bias, weight decay
    // and AdamW all handled by `Linear`, matching the CPU cell's conventions.
    lin_q: Linear,
    lin_k: Linear,
    lin_v: Linear,
    lin_o: Linear,
    lin_i: Linear,
    lin_f: Linear,
    lin_out: Linear,
    headnorm: RmsNorm, // head-wise (group == dhv)

    /// Forward caches awaiting a backward, one per chunk in eviction order.
    ///
    /// A chunked sweep forwards every chunk before unwinding any, so chunk c's cache
    /// must survive chunk c+1's forward; backward pops from the end (right to left).
    /// The unchunked path is this with a single element.
    saved: Vec<Cache>,
    /// Scratch buffers for this cell's temporaries, recycled by size. The
    /// projections and head-major reorgs produce a dozen values that die within
    /// the same call; pooling them means the cell converges on the peak number
    /// live at once instead of reallocating each one every window.
    pool: Pool,
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
        Self {
            input_size,
            d,
            heads,
            dqk,
            dhv,
            inv_sqrt_dqk: 1.0 / (dqk as f32).sqrt(),
            chunk: chunk_len(),
            lin_q: Linear::from_parts(gpu, wq, bq),
            lin_k: Linear::from_parts(gpu, wk, bk),
            lin_v: Linear::from_parts(gpu, wv, bv),
            lin_o: Linear::from_parts(gpu, wo, bo),
            lin_i: Linear::from_parts(gpu, wi, bi),
            lin_f: Linear::from_parts(gpu, wf, bf),
            lin_out: Linear::from_parts(gpu, w_out, b_out),
            headnorm: RmsNorm::from_parts_grouped(gpu, gamma, dhv),
            saved: Vec::new(),
            pool: Pool::default(),
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

    /// Export this cell into the CPU `nn::MLSTMLayer` format. Used to write a
    /// `HIER` checkpoint from a GPU model.
    pub fn to_nn_cell(&self, gpu: &Gpu) -> crate::nn::mlstm::MLSTMLayer {
        use super::{dt_matrix, dt_vec};
        let w_out = crate::nn::linear::LinearLayer::from_loaded(
            self.d,
            self.d,
            dt_matrix(gpu, &self.lin_out.w),
            dt_vec(gpu, &self.lin_out.b),
        );
        crate::nn::mlstm::MLSTMLayer::from_loaded(
            self.input_size,
            self.d,
            self.heads,
            self.dqk,
            dt_matrix(gpu, &self.lin_q.w),
            dt_matrix(gpu, &self.lin_k.w),
            dt_matrix(gpu, &self.lin_v.w),
            dt_matrix(gpu, &self.lin_o.w),
            dt_matrix(gpu, &self.lin_i.w),
            dt_matrix(gpu, &self.lin_f.w),
            dt_vec(gpu, &self.lin_q.b),
            dt_vec(gpu, &self.lin_k.b),
            dt_vec(gpu, &self.lin_v.b),
            dt_vec(gpu, &self.lin_o.b),
            dt_vec(gpu, &self.lin_i.b),
            dt_vec(gpu, &self.lin_f.b),
            w_out,
            dt_vec(gpu, &self.headnorm.gamma),
        )
    }

    /// Rebuild a GPU cell from a CPU `nn::MLSTMLayer` (inverse of `to_nn_cell`).
    pub fn from_nn_cell(gpu: &Gpu, c: &crate::nn::mlstm::MLSTMLayer) -> Self {
        use super::{tensor_from_matrix as m, tensor_from_slice as v};
        Self::from_parts(
            gpu,
            c.input_size,
            c.hidden_size,
            c.num_heads,
            c.dqk,
            &m(&c.wq),
            &m(&c.wk),
            &m(&c.wv),
            &m(&c.wo),
            &m(&c.wi),
            &m(&c.wf),
            &v(&c.bq),
            &v(&c.bk),
            &v(&c.bv),
            &v(&c.bo),
            &v(&c.bi),
            &v(&c.bf),
            &m(&c.w_out.weights),
            &v(&c.w_out.biases),
            &v(&c.head_norm.gamma),
        )
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

    /// Override the chunk length (0 = single-chunk). Lets a caller — or a test —
    /// pick a length per cell instead of taking the `config`/env default.
    pub fn set_chunk(&mut self, chunk: usize) {
        self.chunk = chunk;
    }

    /// The chunk length the **fused** kernels would run this sequence at, or `None`
    /// if it must take the op-at-a-time path.
    ///
    /// `chunk == 0` means "one chunk over the whole sequence", which the fused
    /// kernels cannot do beyond `FUSED_MAX_L` (the decay matrix would not fit in
    /// shared memory) — and it is only ever set to A/B the O(T²) single-chunk form,
    /// so it keeps the old path rather than being silently reblocked. A configured
    /// chunk longer than `FUSED_MAX_L` *is* silently clamped: chunk length is a
    /// blocking choice with no effect on the result, which
    /// `mlstm_chunking_matches_single_chunk` pins.
    fn fused_chunk(&self, gpu: &Gpu, t: usize) -> Option<usize> {
        if legacy_forced() || self.chunk == 0 {
            return None;
        }
        let l = self.chunk.min(ops::FUSED_MAX_L).min(t).max(1);
        if !ops::mlstm_fused_supported(l, self.dqk, self.dhv) {
            return None;
        }
        // The fused kernels hold their decay matrix (and backward's state tiles) in
        // shared memory; at the backbone's head dims that can exceed what this device
        // lets one block opt into. Fall back to the op-at-a-time path rather than
        // failing the opt-in — reducing `l` barely helps, the `[dhv, dqk]` tiles
        // dominate, and the scalar path has the same footprint.
        let mma = gpu.kernels.has_mma && ops::mma_enabled_pub();
        if ops::mlstm_fused_smem_bytes(l, self.dqk, self.dhv, mma) > gpu.max_shared_optin {
            return None;
        }
        Some(l)
    }

    /// Chunk boundaries for a sequence of length `t`: `[(c0, len), …]`. A `t` that
    /// already fits in one chunk yields a single full-length chunk, i.e. exactly
    /// the pre-chunking path.
    fn chunk_spans(&self, t: usize) -> Vec<(usize, usize)> {
        let l = match self.chunk {
            0 => t,
            l => l.min(t),
        };
        (0..t).step_by(l).map(|c0| (c0, l.min(t - c0))).collect()
    }

    /// Forward over `[B, T, in]` → `[B, T, d]`.
    ///
    /// Chunkwise: the sequence is cut into chunks of `chunk_len()`, each handled by
    /// the parallel (attention) form over its own `[BH, L, L]` decay matrix, with
    /// the recurrent state `(C, n, m)` carried across chunk boundaries. One chunk
    /// covering the whole sequence reduces to the single-chunk form.
    pub fn forward(&mut self, gpu: &Gpu, x: &DTensor, y: &mut DTensor) {
        let out = self.forward_alloc(gpu, x);
        y.copy_from(gpu, &out);
    }

    /// The chunkwise core, returning its own output buffer. `forward` copies that
    /// into the caller's; the internals still allocate their per-chunk temporaries.
    pub fn forward_alloc(&mut self, gpu: &Gpu, x: &DTensor) -> DTensor {
        // Release the previous eviction before allocating anything here: freeing
        // returns memory to the CUDA allocator, which must not hand it back while a
        // copy is still reading it. Ordered on the compute stream, so it costs no host
        // time. See `Block::forward` for the failure this prevents.
        if let Some(park) = &self.park {
            park.release_previous(gpu);
        }
        assert_eq!(x.rank, 3, "MLstm::forward expects [B, T, in]");
        let (b, t, inp) = (x.shape[0], x.shape[1], x.shape[2]);
        assert_eq!(
            inp, self.input_size,
            "MLstm::forward — input width mismatch"
        );
        let (d, h, dqk, dhv) = (self.d, self.heads, self.dqk, self.dhv);
        let (n, bh) = (b * t, b * h);

        // Projections on the flat [N, in] view. Every one of these is consumed by
        // the head-major reorg below and then dead, so they come from the pool and
        // go straight back — the cell holds one set of them, not one per call.
        // Widths differ per projection — q/k are `heads·dqk`, v/o are `heads·dhv`
        // (== d), i/f are one logit per head — so each buffer is sized from its own
        // layer rather than assuming `d`.
        //
        // All six projections read the SAME `xf`, so it is saved once here and handed
        // to `forward_shared`, which copies nothing. `Linear::forward` would instead
        // deep-copy it into each layer's own `self.x` — five identical [N, in] copies,
        // 21 MB per cell at H=1024 and 252 MB across the backbone, for one tensor.
        // Backward pairs this with `backward_with_x(&sv.xf, …)`.
        //
        // `xf` therefore outlives the call and is NOT pooled: it must survive to
        // backward, and the pool has to get back everything it lends
        // (`assert_drained` at the top of `backward_alloc`).
        let mut xf = DTensor::uninit(gpu, &[n, inp]);
        xf.copy_from(gpu, x);
        let mut q = self.pool.take(gpu, &[n, self.lin_q.output_size()]);
        self.lin_q.forward_shared(gpu, &xf, &mut q);
        let mut k = self.pool.take(gpu, &[n, self.lin_k.output_size()]);
        self.lin_k.forward_shared(gpu, &xf, &mut k);
        ops::scale_(gpu, &mut k, self.inv_sqrt_dqk);
        let mut v = self.pool.take(gpu, &[n, self.lin_v.output_size()]);
        self.lin_v.forward_shared(gpu, &xf, &mut v);
        // `o` is kept in the cache for backward, so it is NOT pooled: the pool must
        // get back everything it lends, and this one never comes back.
        let mut o = DTensor::uninit(gpu, &[n, self.lin_o.output_size()]);
        self.lin_o.forward_shared(gpu, &xf, &mut o);
        ops::sigmoid_(gpu, &mut o);
        let mut ig = self.pool.take(gpu, &[n, self.lin_i.output_size()]); // [N, H]
        self.lin_i.forward_shared(gpu, &xf, &mut ig);
        let mut fg = self.pool.take(gpu, &[n, self.lin_f.output_size()]); // [N, H]
        self.lin_f.forward_shared(gpu, &xf, &mut fg);

        // The gate logits go head-major as fp32 on either path: the reference pins
        // vecI/vecB to fp32, and they are [BH, T] — a factor of `dqk` smaller than
        // q/k/v, so there is nothing to win by narrowing them anyway.
        let igh = ops::head_gather(gpu, &ig, b, h, t, 1).reshaped(&[bh, t]); // [BH, T]
        let fgh = ops::head_gather(gpu, &fg, b, h, t, 1).reshaped(&[bh, t]);

        // The fused kernels do the whole chunkwise core — states and all chunks — in
        // three launches. Everything before and after (projections, head norm, the
        // o-gate, the output projection) is shared with the path below.
        //
        // The path is chosen BEFORE the q/k/v reorg, because the two want different
        // destinations: the fused path gathers straight into bf16 slabs, while the
        // op-at-a-time path below slices q/k/v for cuBLAS and needs them fp32.
        // Gathering fp32 first and narrowing afterwards would allocate the wide
        // buffer regardless — see `ops::head_gather_slab` for why that costs rather
        // than saves.
        if let Some(l) = self.fused_chunk(gpu, t) {
            // Head-major reorg straight into slab storage. These outlive the call
            // (backward reads them), so they are NOT pooled.
            let qh = ops::head_gather_slab(gpu, &q, b, h, t, dqk); // [BH, T, dqk]
            let kh = ops::head_gather_slab(gpu, &k, b, h, t, dqk);
            let vh = ops::head_gather_slab(gpu, &v, b, h, t, dhv); // [BH, T, dhv]
            self.pool.put_all([q, k, v, ig, fg]);
            // Carry the recurrent state in from the previous chunk when the surrounding
            // sweep is chunked; `None` (the unchunked case) starts it at zero inside
            // the kernel. The outgoing state is taken below, after the cache is built.
            let fused = ops::mlstm_fused_fw(
                gpu,
                &qh,
                &kh,
                &vh,
                &igh,
                &fgh,
                l,
                if self.carry { self.carry_state.as_ref() } else { None },
            );
            if self.carry {
                let bh = b * h;
                self.carry_state = Some(fused.final_state(gpu, bh, dhv, dqk));
            }
            // `h_tilde` and `hconcat` die inside this block; `yhat` is cached for
            // backward, so only the first two are pooled.
            let mut h_tilde = self.pool.take(gpu, &[n, d]); // [N, d]
            ops::head_scatter_slab_into(gpu, &fused.ytil, b, h, t, dhv, &mut h_tilde);
            let mut yhat = DTensor::uninit(gpu, &[n, d]);
            self.headnorm.forward(gpu, &h_tilde, &mut yhat);
            self.pool.put(h_tilde);
            // `hconcat` is kept in the cache rather than pooled, and `lin_out` takes it
            // back through `backward_with_x`: a chunked sweep would otherwise have the
            // next chunk's forward overwrite the private copy `forward_alloc` saves.
            let mut hconcat = DTensor::uninit(gpu, &[n, d]);
            ops::mul_into(gpu, &o, &yhat, &mut hconcat);
            let mut out = DTensor::uninit(gpu, &[n, d]);
            self.lin_out.forward_shared(gpu, &hconcat, &mut out);
            self.saved.push(Cache::Fused(SavedFused {
                b,
                t,
                xf: Some(xf),
                qh: Some(qh),
                kh: Some(kh),
                vh: Some(vh),
                igh,
                fgh,
                fused,
                o: Some(o),
                yhat: Some(yhat),
                hconcat: Some(hconcat),
            }));
            self.evict_saved(gpu);
            return out.reshaped(&[b, t, d]);
        }

        // Op-at-a-time path: q/k/v go head-major as fp32, because this path slices
        // them and hands the slices to cuBLAS, which has no bf16 operand here.
        let qh = ops::head_gather(gpu, &q, b, h, t, dqk); // [BH, T, dqk]
        let kh = ops::head_gather(gpu, &k, b, h, t, dqk);
        let vh = ops::head_gather(gpu, &v, b, h, t, dhv); // [BH, T, dhv]
        self.pool.put_all([q, k, v, ig, fg]);

        // Recurrent state carried across chunks (stabilized, as on the CPU).
        let mut c_state = DTensor::zeros(gpu, &[bh, dhv, dqk]);
        let mut n_state = DTensor::zeros(gpu, &[bh, 1, dqk]);
        let mut m_state = DTensor::zeros(gpu, &[bh]);

        let spans = self.chunk_spans(t);
        let last_span = spans.len() - 1;
        let mut ytil = DTensor::uninit(gpu, &[bh, t, dhv]);
        let mut chunks = Vec::with_capacity(spans.len());

        for (ci, &(c0, len)) in spans.iter().enumerate() {
            // One launch for all five: q/k/v are [BH, T, dqk|dhv] and the two gate
            // tensors are [BH, T, 1], so they share BH and T and differ only in width.
            let mut sl = ops::slice_t_batch(
                gpu,
                &[(&qh, dqk), (&kh, dqk), (&vh, dhv), (&igh, 1), (&fgh, 1)],
                bh,
                t,
                c0,
                len,
            )
            .into_iter();
            let qc = sl.next().expect("qc"); // [BH, L, dqk]
            let kc = sl.next().expect("kc");
            let vc = sl.next().expect("vc"); // [BH, L, dhv]
            let igc = sl.next().expect("igc").reshaped(&[bh, len]);
            let fgc = sl.next().expect("fgc").reshaped(&[bh, len]);

            // Decay/stabilizer machinery, on the chunk-LOCAL cumulative log-forget.
            // `m_state` enters via the `fc_t + m_prev` branch of the row-max, which
            // is what makes the local stabilizer equal the global one.
            let fc = ops::cumsum_logsig(gpu, &fgc); // [BH, L]
            let m = ops::mlstm_rowmax_m(gpu, &fc, &igc, &m_state);
            let (bvec, avec) = ops::mlstm_chunk_ab(gpu, &fc, &igc, &m, &m_state);

            // Intra-chunk (the parallel form). S, D̄ and DS are the [BH, L, L]
            // tensors; S never outlives this scope, D̄/DS are kept for backward.
            let (mut num, mut qn, psi_intra, dbar, ds) = {
                let s = ops::matmul_batched_nt(gpu, &qc, &kc); // S = Q·Kᵀ  [BH, L, L]
                let (dbar, ds, qn, psi) = ops::mlstm_ds(gpu, &s, &fc, &igc, &m);
                let num = ops::matmul_batched_nn(gpu, &ds, &vc); // (D̄⊙S)·V  [BH, L, dhv]
                (num, qn, psi, dbar, ds)
            };

            // Inter-chunk: read the carried state out through Q, scaled by b_t.
            // Skipped on the first chunk, where the state is still zero.
            let (inter, psi) = if ci == 0 {
                (None, psi_intra)
            } else {
                let inter_num = ops::matmul_batched_nt(gpu, &qc, &c_state); // [BH, L, dhv]
                let inter_qn = ops::matmul_batched_nt(gpu, &qc, &n_state); // [BH, L, 1]
                ops::mul_rows_add(gpu, &mut num, &inter_num, &bvec, dhv);
                ops::mul_rows_add(gpu, &mut qn, &inter_qn, &bvec, 1);
                let psi = ops::psi_from_qn(gpu, &qn, &m); // ψ follows the COMBINED qn
                let inter = Inter {
                    c_prev: c_state.dup(gpu),
                    n_prev: n_state.dup(gpu),
                    inter_num,
                    inter_qn,
                };
                (Some(inter), psi)
            };

            let yc = ops::div_rows(gpu, &num, &psi, dhv); // ỹ  [BH, L, dhv]
            ops::unslice_t(gpu, &mut ytil, &yc, c0);

            // End-of-chunk state update (skipped after the last chunk — nothing reads
            // it). a_j is the last row of D̄ and g = b_last, both already in hand:
            //   C ← g·C + (a⊙V)ᵀ·K ,  n ← g·n + Σ_j a_j k_j
            if ci != last_span {
                let g = ops::slice_t_as(gpu, &bvec, bh, len, 1, len - 1, 1).reshaped(&[bh]); // [BH]
                let va = ops::mul_rows(gpu, &vc, &avec, dhv); // [BH, L, dhv]
                let mut c_new = ops::matmul_batched_tn(gpu, &va, &kc); // [BH, dhv, dqk]
                let a3 = avec.dup(gpu).reshaped(&[bh, len, 1]);
                let mut n_new = ops::matmul_batched_tn(gpu, &a3, &kc); // [BH, 1, dqk]
                ops::mul_rows_add(gpu, &mut c_new, &c_state, &g, dhv * dqk);
                ops::mul_rows_add(gpu, &mut n_new, &n_state, &g, dqk);
                c_state = c_new;
                n_state = n_new;
                // m_new = the chunk's last-row stabilizer.
                m_state = ops::slice_t_as(gpu, &m, bh, len, 1, len - 1, 1).reshaped(&[bh]);
            }

            chunks.push(Chunk {
                c0,
                len,
                bvec,
                avec,
                qn,
                psi,
                m,
                num,
                dbar,
                ds,
                inter,
            });
        }

        // Back to position-major, head-norm, o-gate, output projection. `h_tilde`
        // and `hconcat` die here; `yhat` is cached for backward.
        let mut h_tilde = self.pool.take(gpu, &[n, d]); // [N, d]
        ops::head_scatter_into(gpu, &ytil, b, h, t, dhv, &mut h_tilde);
        let mut yhat = DTensor::uninit(gpu, &[n, d]);
        self.headnorm.forward(gpu, &h_tilde, &mut yhat);
        self.pool.put(h_tilde);
        // `hconcat` is cached rather than pooled, and `lin_out` takes it back through
        // `backward_with_x`: under a chunked sweep the private copy `forward_alloc`
        // saves would be overwritten by the next chunk's forward.
        let mut hconcat = DTensor::uninit(gpu, &[n, d]);
        ops::mul_into(gpu, &o, &yhat, &mut hconcat); // o ⊙ ŷ  [N, d]
        let mut out = DTensor::uninit(gpu, &[n, d]);
        self.lin_out.forward_shared(gpu, &hconcat, &mut out); // [N, d]

        // `o`/`yhat` are unused after `mul`, so move (not dup) them into the cache.
        self.saved.push(Cache::Legacy(Saved {
            b,
            t,
            xf,
            qh,
            kh,
            vh,
            fgh,
            chunks,
            o,
            yhat,
            hconcat,
        }));
        out.reshaped(&[b, t, d])
    }

    /// Park this cell's saved activations on the host, if offload is enabled.
    ///
    /// Called at the end of a fused forward. Only the six large per-`N` tensors ride:
    /// `xf`/`o`/`yhat` (4 MB each at the backbone's shape) and the `qh`/`kh`/`vh`
    /// slabs (2 MB each on the bf16 path) — 18 MB of the cell's 21.5 MB. The rest
    /// (`igh`/`fgh` at 64 KB, and everything inside `fused`) is left resident: each is
    /// two orders of magnitude smaller, so moving it would add PCIe traffic and
    /// bookkeeping for nothing.
    ///
    /// The slabs park at **their own width** — a bf16 slab comes back bf16, since the
    /// precision split belongs at each value's production point, not here.
    fn evict_saved(&mut self, gpu: &Gpu) {
        use super::offload::Parked;
        let Some(park) = &mut self.park else { return };
        // The chunk just forwarded — the one this call is closing out.
        let Some(Cache::Fused(sv)) = self.saved.last_mut() else {
            return;
        };
        // Fixed order, mirrored exactly by `restore_saved`.
        park.evict(
            gpu,
            vec![
                Parked::from(sv.xf.take().expect("evict before restore: xf")),
                Parked::from(sv.o.take().expect("evict before restore: o")),
                Parked::from(sv.yhat.take().expect("evict before restore: yhat")),
                Parked::from(sv.qh.take().expect("evict before restore: qh")),
                Parked::from(sv.kh.take().expect("evict before restore: kh")),
                Parked::from(sv.vh.take().expect("evict before restore: vh")),
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
        let Some(Cache::Fused(sv)) = self.saved.last_mut() else {
            return;
        };
        let mut it = park.restore(gpu).into_iter();
        let mut next = |what: &str| it.next().expect(what);
        sv.xf = Some(next("parked xf").f32());
        sv.o = Some(next("parked o").f32());
        sv.yhat = Some(next("parked yhat").f32());
        sv.qh = Some(next("parked qh").into());
        sv.kh = Some(next("parked kh").into());
        sv.vh = Some(next("parked vh").into());
    }

    /// Device bytes held, split `(params, activations)`. Diagnostic — see
    /// [`Hierarchical::retained_report`](super::hierarchical::Hierarchical::retained_report).
    ///
    /// Note the multiplier: this cell owns **seven** [`Linear`]s, each with its own
    /// saved input and bf16 GEMM staging, plus a head-wise [`RmsNorm`] holding an
    /// `x̂` the width of the cell. `drop_saved_act` clears only `saved` — the
    /// per-`Linear` retention survives it.
    pub fn retained_bytes(&self) -> (usize, usize) {
        let (mut params, mut act) = (0, 0);
        for l in [
            &self.lin_q,
            &self.lin_k,
            &self.lin_v,
            &self.lin_o,
            &self.lin_i,
            &self.lin_f,
            &self.lin_out,
        ] {
            let (p, a) = l.retained_bytes();
            params += p;
            act += a;
        }
        let (hn_p, hn_a) = self.headnorm.retained_bytes();
        params += hn_p;
        act += hn_a + self.pool.retained_bytes();
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

    /// Retained activation bytes split `(saved_cache, other)`.
    ///
    /// `saved_cache` is what `drop_saved_act` releases. `other` is the pooled scratch
    /// plus what the **seven** projections and the head norm hold internally — their
    /// saved inputs and bf16 GEMM staging — which no `drop_saved_act` reaches.
    pub fn act_split(&self) -> (usize, usize) {
        let saved = self.saved_bytes();
        let (_, all) = self.retained_bytes();
        (saved, all - saved)
    }

    /// Device bytes held by the forward caches, summed over every chunk still awaiting
    /// its backward.
    fn saved_bytes(&self) -> usize {
        self.saved
            .iter()
            .map(|c| match c {
                Cache::Fused(s) => s.retained_bytes(),
                Cache::Legacy(s) => s.retained_bytes(),
            })
            .sum()
    }

    /// Release everything a forward left behind that no backward will read: the
    /// saved cache, the pooled scratch, and each projection's saved input and bf16
    /// staging. The broader companion to the `Cell::drop_saved_act`, which only
    /// clears `saved`.
    pub fn drop_all_act(&mut self, gpu: &Gpu) {
        self.saved.clear();
        // NOT `pool.trim(0)`: the pool is the cell's per-call scratch working set, and
        // it is re-taken in full on the very next call. Emptying it at a group boundary
        // means every group reallocates every temporary — the allocator back on the hot
        // path, which is exactly what `Pool` exists to avoid. The window boundary
        // (`Hierarchical::trim_pools`) is where it gets sized down, against the largest
        // group that actually ran.
        for l in [
            &mut self.lin_q,
            &mut self.lin_k,
            &mut self.lin_v,
            &mut self.lin_o,
            &mut self.lin_i,
            &mut self.lin_f,
            &mut self.lin_out,
        ] {
            l.drop_saved_act(gpu);
        }
        self.headnorm.drop_saved_act();
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

    /// Backward into a freshly allocated `dx` `[B, T, in]` — the by-value
    /// companion to [`backward`](Self::backward).
    pub fn backward_alloc_dx(&mut self, gpu: &Gpu, dy: &DTensor) -> DTensor {
        let mut dx =
            DTensor::uninit(gpu, &[dy.shape[0], dy.shape[1], self.input_size]);
        self.backward(gpu, dy, &mut dx);
        dx
    }

    /// Backward over `[B, T, d]` → `dx` `[B, T, in]`. Accumulates all grads.
    ///
    /// Chunks are swept in reverse, carrying `dC`/`dn` (the grad wrt the state a
    /// chunk hands to its successor) backwards the way forward carried `C`/`n`
    /// forwards — BPTT over chunks, with the parallel form inside each.
    pub fn backward(&mut self, gpu: &Gpu, dy: &DTensor, dx: &mut DTensor) {
        let out = self.backward_alloc(gpu, dy);
        dx.copy_from(gpu, &out);
    }

    pub fn backward_alloc(&mut self, gpu: &Gpu, dy: &DTensor) -> DTensor {
        self.pool.assert_drained("MLstm::backward");
        // Bring the parked activations back into the cache first, so everything below
        // reads them as if they had never left. No-op unless offload is enabled.
        self.restore_saved(gpu);
        // `take`, not `as_ref`: the cache holds a window's activations, and dropping
        // them at the end of this call (rather than when the next forward overwrites
        // the field) keeps them from staying resident across the optimizer step.
        match self
            .saved
            .pop()
            .expect("MLstm::backward before forward")
        {
            Cache::Fused(sv) => self.backward_fused(gpu, dy, sv),
            Cache::Legacy(sv) => self.backward_legacy(gpu, dy, sv),
        }
    }

    /// Backward of the fused path: two kernels for the whole chunkwise core, with
    /// the same projection/head-norm/o-gate shell as the op-at-a-time path.
    fn backward_fused(&mut self, gpu: &Gpu, dy: &DTensor, sv: SavedFused) -> DTensor {
        let (d, h, dqk, dhv, inp) = (self.d, self.heads, self.dqk, self.dhv, self.input_size);
        let (b, t) = (sv.b, sv.t);
        let (n, bh) = (b * t, b * h);

        // The whole shell is temporaries: each value below dies as soon as the next
        // op has read it, so all of them come from the pool and go back.
        let mut dy_flat = self.pool.take(gpu, &[n, d]);
        dy_flat.copy_from(gpu, dy);
        let mut d_hconcat = self.pool.take(gpu, &[n, d]);
        self.lin_out
            .backward_with_x(gpu, sv.hconcat(), &dy_flat, &mut d_hconcat);
        self.pool.put(dy_flat);
        let (do_pre, d_yhat) = ops::ogate_bwd(gpu, &d_hconcat, sv.o(), sv.yhat());
        self.pool.put(d_hconcat);
        let mut d_h_tilde = self.pool.take(gpu, &[n, d]);
        self.headnorm.backward(gpu, &d_yhat, &mut d_h_tilde);
        let d_ytil = ops::head_gather(gpu, &d_h_tilde, b, h, t, dhv); // [BH, T, dhv]
        // `d_h_tilde` is the pool's; `d_yhat` came from `ogate_bwd`, so it is only
        // dropped — see the note at the end of this function.
        self.pool.put(d_h_tilde);
        drop(d_yhat);

        // Backward unwinds chunks right to left, so the carried BPTT state comes from
        // the chunk to the RIGHT — `None` on the rightmost (and on any unchunked call).
        let (dqh, dkh, dvh, digh, dfgh, dstate) = ops::mlstm_fused_bw(
            gpu,
            &sv.fused,
            sv.qh(),
            sv.kh(),
            sv.vh(),
            &sv.igh,
            &sv.fgh,
            &d_ytil,
            if self.carry { self.carry_dstate.as_ref() } else { None },
        );
        if self.carry {
            self.carry_dstate = Some(dstate);
        }

        let dq = ops::head_scatter(gpu, &dqh, b, h, t, dqk); // [N, dqk·H]
        let mut dk = ops::head_scatter(gpu, &dkh, b, h, t, dqk);
        ops::scale_(gpu, &mut dk, self.inv_sqrt_dqk); // k = (·)·1/√dqk
        let dv = ops::head_scatter(gpu, &dvh, b, h, t, dhv);
        let d_ig = ops::head_scatter(gpu, &digh.reshaped(&[bh, t, 1]), b, h, t, 1);
        let d_fg = ops::head_scatter(gpu, &dfgh.reshaped(&[bh, t, 1]), b, h, t, 1);

        // dx is the sum of the six projection backwards, accumulated into one
        // buffer with one pooled scratch — not a fresh [N, in] per term.
        //
        // All six read the one shared `sv.xf` (see `forward_alloc`), so they take
        // `backward_with_x` rather than each consulting a private saved copy.
        let mut acc = DTensor::uninit(gpu, &[n, inp]);
        self.lin_q.backward_with_x(gpu, sv.xf(), &dq, &mut acc);
        let mut part = self.pool.take(gpu, &[n, inp]);
        for (lin, grad) in [
            (&mut self.lin_k, &dk),
            (&mut self.lin_v, &dv),
            (&mut self.lin_o, &do_pre),
            (&mut self.lin_i, &d_ig),
            (&mut self.lin_f, &d_fg),
        ] {
            lin.backward_with_x(gpu, sv.xf(), grad, &mut part);
            ops::add_assign(gpu, &mut acc, &part);
        }
        // Only `part` came from the pool; `dq`..`d_fg` were allocated by
        // `head_scatter` and `ogate_bwd`, so they are dropped, not donated. Handing
        // the pool buffers it never lent would grow the free list every window —
        // a leak, not a cache.
        self.pool.put(part);
        drop((dq, dk, dv, do_pre, d_ig, d_fg));
        acc.reshaped(&[b, t, inp])
    }

    fn backward_legacy(&mut self, gpu: &Gpu, dy: &DTensor, sv: Saved) -> DTensor {
        let (d, h, dqk, dhv, inp) = (self.d, self.heads, self.dqk, self.dhv, self.input_size);
        let sv = &sv;
        let (b, t) = (sv.b, sv.t);
        let (n, bh) = (b * t, b * h);

        // The shell before the chunk loop is all temporaries; they come from the
        // pool and go back as soon as their consumer has read them.
        let mut dy_flat = self.pool.take(gpu, &[n, d]);
        dy_flat.copy_from(gpu, dy);

        // Output projection + o-gate.
        let mut d_hconcat = self.pool.take(gpu, &[n, d]); // [N, d]
        self.lin_out
            .backward_with_x(gpu, &sv.hconcat, &dy_flat, &mut d_hconcat);
        self.pool.put(dy_flat);
        let (do_pre, d_yhat) = ops::ogate_bwd(gpu, &d_hconcat, &sv.o, &sv.yhat);
        self.pool.put(d_hconcat);

        // Head-norm backward → d_h_tilde, then head-gather to head-major d_ytil.
        let mut d_h_tilde = self.pool.take(gpu, &[n, d]); // [N, d]
        self.headnorm.backward(gpu, &d_yhat, &mut d_h_tilde);
        let d_ytil = ops::head_gather(gpu, &d_h_tilde, b, h, t, dhv); // [BH, T, dhv]
        // `d_h_tilde` is the pool's; `d_yhat` came from `ogate_bwd`, so it is only
        // dropped — the pool must get back exactly what it lent.
        self.pool.put(d_h_tilde);
        drop(d_yhat);

        // Full-sequence grad buffers; each chunk writes its own disjoint T-range.
        let mut dqh = DTensor::uninit(gpu, &[bh, t, dqk]);
        let mut dkh = DTensor::uninit(gpu, &[bh, t, dqk]);
        let mut dvh = DTensor::uninit(gpu, &[bh, t, dhv]);
        let mut digh = DTensor::uninit(gpu, &[bh, t, 1]);
        let mut d_fgh3 = DTensor::uninit(gpu, &[bh, t, 1]);

        // Grad wrt the state leaving the chunk under consideration. Zero for the
        // last chunk (nothing downstream reads its outgoing state).
        let mut dc_carry = DTensor::zeros(gpu, &[bh, dhv, dqk]);
        let mut dn_carry = DTensor::zeros(gpu, &[bh, 1, dqk]);

        for (ci, ch) in sv.chunks.iter().enumerate().rev() {
            let (c0, len) = (ch.c0, ch.len);
            let is_last = ci + 1 == sv.chunks.len();

            let mut sl = ops::slice_t_batch(
                gpu,
                &[
                    (&sv.qh, dqk),
                    (&sv.kh, dqk),
                    (&sv.vh, dhv),
                    (&sv.fgh, 1),
                    (&d_ytil, dhv),
                ],
                bh,
                t,
                c0,
                len,
            )
            .into_iter();
            let qc = sl.next().expect("qc"); // [BH, L, dqk]
            let kc = sl.next().expect("kc");
            let vc = sl.next().expect("vc"); // [BH, L, dhv]
            let fgc = sl.next().expect("fgc").reshaped(&[bh, len]);
            let d_ytil_c = sl.next().expect("d_ytil_c"); // [BH, L, dhv]

            // ỹ = num/ψ  → d_num, d_qn  (num/ψ/qn all include the inter term).
            let (d_num, d_qn) =
                ops::div_rows_bwd(gpu, &d_ytil_c, &ch.num, &ch.psi, &ch.qn, &ch.m, dhv);

            // The [BH, L, L] tensors, from forward's cache. Everything derived from
            // them below is at most [BH, L, dqk].
            let (mut dvc, d_s, p) = {
                // num = DS·V:  dV = DSᵀ·d_num ;  dDS(num path) = d_num·Vᵀ.
                let dvc = ops::matmul_batched_tn(gpu, &ch.ds, &d_num); // [BH, L, dhv]
                let dds_num = ops::matmul_batched_nt(gpu, &d_num, &vc); // [BH, L, L]

                // DS = D̄⊙S + qn-sum:  dS and P (= dD̄⊙D̄, feeds fc/ig grads).
                let (d_s, p) = ops::mlstm_ds_bwd(gpu, &dds_num, &d_qn, &ch.dbar, &ch.ds);
                (dvc, d_s, p)
            };

            // S = Q·Kᵀ:  dQ = dS·K ;  dK = dSᵀ·Q.
            let mut dqc = ops::matmul_batched_nn(gpu, &d_s, &kc); // [BH, L, dqk]
            let mut dkc = ops::matmul_batched_tn(gpu, &d_s, &qc); // [BH, L, dqk]
            drop(d_s);

            // Decay grads from the intra-chunk D̄. `mlstm_dfc_dig` WRITES these; the
            // a/b contributions below accumulate on top.
            let (mut dfc, mut dig) = ops::mlstm_dfc_dig(gpu, &p); // [BH, L] each
            drop(p);

            let mut db = DTensor::zeros(gpu, &[bh, len]); // grad wrt b_t
            let mut da = DTensor::zeros(gpu, &[bh, len]); // grad wrt a_j

            // state-update path: how this chunk fed the NEXT chunk's state
            //   C_out = g·C_in + (a⊙V)ᵀ·K ,  n_out = g·n_in + Σ_j a_j k_j
            // Skipped for the last chunk (dc_carry / dn_carry are zero there).
            let (mut dc_in, mut dn_in) = (None, None);
            if !is_last {
                let a3 = ch.avec.dup(gpu).reshaped(&[bh, len, 1]);
                let va = ops::mul_rows(gpu, &vc, &ch.avec, dhv); // a⊙V  [BH, L, dhv]

                // C_out = Vaᵀ·K:  dVa = K·dC_outᵀ ;  dK += Va·dC_out.
                let dva = ops::matmul_batched_nt(gpu, &kc, &dc_carry); // [BH, L, dhv]
                ops::add_assign(gpu, &mut dkc, &ops::matmul_batched_nn(gpu, &va, &dc_carry));
                // Va = a⊙V:  dV += a⊙dVa ;  da += Σ_p dVa·V.
                ops::mul_rows_add(gpu, &mut dvc, &dva, &ch.avec, dhv);
                ops::row_dot_add(gpu, &mut da, &dva, &vc, dhv);

                // n_out = Σ_j a_j k_j:  dK += a ⊗ dn_out ;  da += K·dn_outᵀ.
                ops::add_assign(gpu, &mut dkc, &ops::matmul_batched_nn(gpu, &a3, &dn_carry));
                let da_n = ops::matmul_batched_nt(gpu, &kc, &dn_carry); // [BH, L, 1]
                ops::add_assign(gpu, &mut da, &da_n.reshaped(&[bh, len]));

                // g·state: dg = Σ(dC_out ⊙ C_in) + Σ(dn_out ⊙ n_in); dstate_in += g·dC_out.
                // Only reachable when this chunk HAS an incoming state (ci > 0) — for
                // chunk 0 the state is zero, so g contributes nothing and there is no
                // predecessor to hand dC_in to.
                if let Some(it) = &ch.inter {
                    let g = ops::slice_t_as(gpu, &ch.bvec, bh, len, 1, len - 1, 1).reshaped(&[bh]);
                    let mut dg = DTensor::zeros(gpu, &[bh]);
                    ops::group_dot_add(gpu, &mut dg, &dc_carry, &it.c_prev);
                    ops::group_dot_add(gpu, &mut dg, &dn_carry, &it.n_prev);

                    let mut dc = DTensor::zeros(gpu, &[bh, dhv, dqk]);
                    let mut dn = DTensor::zeros(gpu, &[bh, 1, dqk]);
                    ops::mul_rows_add(gpu, &mut dc, &dc_carry, &g, dhv * dqk);
                    ops::mul_rows_add(gpu, &mut dn, &dn_carry, &g, dqk);
                    dc_in = Some(dc);
                    dn_in = Some(dn);

                    // g IS b_last, so dg lands on the last column of db.
                    let mut dg_pad = DTensor::zeros(gpu, &[bh, len, 1]);
                    ops::unslice_t_as(gpu, &mut dg_pad, &dg, 1, len - 1);
                    ops::add_assign(gpu, &mut db, &dg_pad.reshaped(&[bh, len]));
                }
            }

            // inter path: how this chunk READ its incoming state
            //   num += b⊙(Q·C_inᵀ) ,  qn += b⊙(Q·n_inᵀ)
            if let Some(it) = &ch.inter {
                // db from both products (they are saved pre-b, which is what db needs).
                ops::row_dot_add(gpu, &mut db, &d_num, &it.inter_num, dhv);
                ops::row_dot_add(gpu, &mut db, &d_qn, &it.inter_qn, 1);

                let d_inter_num = ops::mul_rows(gpu, &d_num, &ch.bvec, dhv); // [BH, L, dhv]
                let d_inter_qn = ops::mul_rows(gpu, &d_qn, &ch.bvec, 1).reshaped(&[bh, len, 1]);

                // dQ from both readouts.
                dqc = ops::add(
                    gpu,
                    &dqc,
                    &ops::matmul_batched_nn(gpu, &d_inter_num, &it.c_prev),
                );
                dqc = ops::add(
                    gpu,
                    &dqc,
                    &ops::matmul_batched_nn(gpu, &d_inter_qn, &it.n_prev),
                );

                // dC_in / dn_in from both readouts (adding to the g·state term above).
                let dc_r = ops::matmul_batched_tn(gpu, &d_inter_num, &qc); // [BH, dhv, dqk]
                let dn_r = ops::matmul_batched_tn(gpu, &d_inter_qn, &qc); // [BH, 1, dqk]
                dc_in = Some(match dc_in {
                    Some(dc) => ops::add(gpu, &dc, &dc_r),
                    None => dc_r,
                });
                dn_in = Some(match dn_in {
                    Some(dn) => ops::add(gpu, &dn, &dn_r),
                    None => dn_r,
                });
            }

            // a/b → (dfc, dig), accumulated onto the intra-chunk D̄ contribution.
            ops::mlstm_chunk_ab_bwd(gpu, &db, &da, &ch.bvec, &ch.avec, &mut dfc, &mut dig);

            // dfc → d(f-logit) via reverse-cumsum·logσ' — within the chunk, since fc
            // is the chunk-local cumsum.
            let d_fgc = ops::revcumsum_dlogsig(gpu, &dfc, &fgc); // [BH, L]

            // One launch for all five; the widths are explicit, so the rank-2 gate
            // gradients need no reshape (and hence no temporary).
            ops::unslice_t_batch(
                gpu,
                &mut [
                    (&mut dqh, &dqc, dqk),
                    (&mut dkh, &dkc, dqk),
                    (&mut dvh, &dvc, dhv),
                    (&mut digh, &dig, 1),
                    (&mut d_fgh3, &d_fgc, 1),
                ],
                bh,
                t,
                c0,
                len,
            );

            // Hand the incoming-state grads to the predecessor chunk.
            dc_carry = dc_in.unwrap_or_else(|| DTensor::zeros(gpu, &[bh, dhv, dqk]));
            dn_carry = dn_in.unwrap_or_else(|| DTensor::zeros(gpu, &[bh, 1, dqk]));
        }

        // Scatter head-major grads back to position-major [N, ·].
        let dq = ops::head_scatter(gpu, &dqh, b, h, t, dqk); // [N, d_qk]
        let mut dk = ops::head_scatter(gpu, &dkh, b, h, t, dqk);
        ops::scale_(gpu, &mut dk, self.inv_sqrt_dqk); // k = (·)·1/√dqk
        let dv = ops::head_scatter(gpu, &dvh, b, h, t, dhv); // [N, d]
        let d_ig = ops::head_scatter(gpu, &digh, b, h, t, 1); // [N, H]
        let d_fg = ops::head_scatter(gpu, &d_fgh3, b, h, t, 1);

        // Projection backward; sum the input grads (all share the saved xf, held
        // once in the cache rather than copied into each of the six `Linear`s).
        // dx is the sum of the six projection backwards, accumulated into one
        // buffer with one pooled scratch — not a fresh [N, in] per term.
        let mut dxf = DTensor::uninit(gpu, &[n, inp]);
        self.lin_q.backward_with_x(gpu, &sv.xf, &dq, &mut dxf);
        let mut part = self.pool.take(gpu, &[n, inp]);
        for (lin, grad) in [
            (&mut self.lin_k, &dk),
            (&mut self.lin_v, &dv),
            (&mut self.lin_o, &do_pre),
            (&mut self.lin_i, &d_ig),
            (&mut self.lin_f, &d_fg),
        ] {
            lin.backward_with_x(gpu, &sv.xf, grad, &mut part);
            ops::add_assign(gpu, &mut dxf, &part);
        }
        self.pool.put(part);
        dxf.reshaped(&[b, t, inp])
    }

    /// Every learnable tensor, in a fixed order (used by checkpoint save/load).
    pub fn params_mut(&mut self) -> Vec<&mut DTensor> {
        let mut v = Vec::new();
        for l in [
            &mut self.lin_q,
            &mut self.lin_k,
            &mut self.lin_v,
            &mut self.lin_o,
            &mut self.lin_i,
            &mut self.lin_f,
            &mut self.lin_out,
        ] {
            v.extend(l.params_mut());
        }
        v.extend(self.headnorm.params_mut());
        v
    }

    pub fn zero_grad(&mut self, gpu: &Gpu) {
        for l in [
            &mut self.lin_q,
            &mut self.lin_k,
            &mut self.lin_v,
            &mut self.lin_o,
            &mut self.lin_i,
            &mut self.lin_f,
            &mut self.lin_out,
        ] {
            l.zero_grad(gpu);
        }
        self.headnorm.zero_grad(gpu);
    }

    /// AdamW step: projection + output matrices decay; biases and head-norm γ
    /// don't (all handled by the sub-layers). Clears the grads.
    pub fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        self.step_q(gpu, cfg, None);
    }

    /// [`step`](Self::step), optionally queueing instead of launching.
    pub fn step_q(&mut self, gpu: &Gpu, cfg: &AdamCfg, mut q: Option<&mut ops::AdamwQueue>) {
        for l in [
            &mut self.lin_q,
            &mut self.lin_k,
            &mut self.lin_v,
            &mut self.lin_o,
            &mut self.lin_i,
            &mut self.lin_f,
            &mut self.lin_out,
        ] {
            l.step_wd_q(gpu, cfg, true, q.as_deref_mut());
        }
        self.headnorm.step_q(gpu, cfg, q.as_deref_mut());
    }
}

impl Cell for MLstm {
    fn forward(&mut self, gpu: &Gpu, x: &DTensor, out: &mut DTensor) {
        MLstm::forward(self, gpu, x, out)
    }
    fn backward(&mut self, gpu: &Gpu, dy: &DTensor, dx: &mut DTensor) {
        MLstm::backward(self, gpu, dy, dx)
    }
    fn zero_grad(&mut self, gpu: &Gpu) {
        MLstm::zero_grad(self, gpu)
    }
    fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        MLstm::step(self, gpu, cfg)
    }
    fn step_q(&mut self, gpu: &Gpu, cfg: &AdamCfg, q: Option<&mut ops::AdamwQueue>) {
        MLstm::step_q(self, gpu, cfg, q)
    }
    fn params_mut(&mut self) -> Vec<&mut DTensor> {
        MLstm::params_mut(self)
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
    fn trim_to(&mut self, rows: usize) {
        // The widest thing this cell pools is `[rows, d]` (the projections and the
        // head-major reorgs); size the bound from that.
        self.pool.trim(rows * self.d);
    }
    fn drop_saved_act(&mut self) {
        // The whole fused cache — qh/kh/vh slabs, o, yhat, xf and the `MlstmFused`
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
    /// The right check when two float orderings are compared: a per-element
    /// relative test explodes on elements that happen to sit near zero, while an
    /// absolute one silently becomes a very tight relative demand on small tensors.
    fn assert_close_rel(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length mismatch");
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let (worst, at) = got.iter().zip(want).enumerate().fold(
            (0.0f32, 0usize),
            |(m, at), (i, (&a, &b))| {
                let d = (a - b).abs();
                if d > m { (d, i) } else { (m, at) }
            },
        );
        assert!(
            worst / scale.max(f32::MIN_POSITIVE) < tol,
            "{what}: worst |a-b| {worst:.3e} at [{at}] on scale {scale:.3e} \
             -> {:.2e} relative, exceeds {tol:.0e}",
            worst / scale.max(f32::MIN_POSITIVE)
        );
    }

    /// Single-chunk parallel forward+backward+step must match the CPU scalar
    /// recurrence (`nn2::MLstm`) from identical weights. The CPU backward is
    /// itself FD-verified, so a GPU-vs-CPU grad match is the (tighter) check.
    #[test]
    fn mlstm_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
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
        let y_dev = dev.forward_alloc(&gpu, &DTensor::from_host(&gpu, &x));
        assert_close(&y_dev.to_host(&gpu).data, &y_cpu.data, 3e-3, "y");

        // Backward
        let dx_cpu = cpu.backward(&g);
        let dx_dev = dev.backward_alloc(&gpu, &DTensor::from_host(&gpu, &g));
        assert_close(&dx_dev.to_host(&gpu).data, &dx_cpu.data, 3e-3, "dx");

        // One AdamW step; compare representative updated parameters.
        let mut cfg = AdamCfg::new(1e-3, 0.01);
        cfg.t = 1;
        cpu.step(&cfg);
        dev.step(&gpu, &cfg);
        // (weights live in the Linear sub-layers; check q, v, out projections + γ)
        assert_close(&dev.lin_q.w.to_host(&gpu).data, &cpu.wq.data, 3e-3, "wq");
        assert_close(&dev.lin_v.w.to_host(&gpu).data, &cpu.wv.data, 3e-3, "wv");
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
        let (b, t, inp, d, heads, dqk) = (2, 20, 5, 8, 2, 4);

        let mut proto = CpuMLstm::new(inp, d, heads, dqk);
        // Non-trivial gate weights so the decay/stabilizer path is exercised — with
        // zero gates every chunk would decay identically and the test would pass
        // vacuously.
        // Seeded for the same reason as `mlstm_fused_matches_legacy`: this compares
        // two chunk blockings whose sums reassociate, so it is a tolerance test and
        // must not redraw its data every run.
        proto.wi = Tensor::random_seeded(&[inp, heads], 0.3, 0xB1);
        proto.wf = Tensor::random_seeded(&[inp, heads], 0.3, 0xB2);

        let x = Tensor::random_seeded(&[b, t, inp], 0.5, 0xB3);
        let g = Tensor::random_seeded(&[b, t, d], 1.0, 0xB4);
        let dx = DTensor::from_host(&gpu, &x);
        let dg = DTensor::from_host(&gpu, &g);

        // Reference: one chunk over the whole sequence.
        let run = |chunk: usize| {
            let mut dev = MLstm::from_cpu(&gpu, &proto);
            dev.set_chunk(chunk);
            let y = dev.forward_alloc(&gpu, &dx).to_host(&gpu).data;
            let dxo = dev.backward_alloc(&gpu, &dg).to_host(&gpu).data;
            let mut cfg = AdamCfg::new(1e-3, 0.01);
            cfg.t = 1;
            dev.step(&gpu, &cfg);
            // wf/wi ride the decay path; w_out rides the value path.
            let wf = dev.lin_f.w.to_host(&gpu).data;
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

    /// The fused kernels vs the op-at-a-time path at the **backbone's real shape**.
    ///
    /// The other tests run at toy dims (dqk = dhv = 4, T = 20), where a fused block
    /// uses a few hundred bytes of shared memory and most of its 256 threads idle.
    /// This one runs dqk = dhv = 64 over a T that spans many chunks and ends on a
    /// SHORT one (T = 200 = 6·32 + 8), which is what actually exercises the shared-
    /// memory staging, the block-wide max/sum reductions, and the `len` masking on
    /// the ragged final chunk. A bug in any of those is invisible at the toy dims.
    #[test]
    fn mlstm_fused_matches_legacy() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
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

        let x = DTensor::from_host(&gpu, &Tensor::random_seeded(&[b, t, inp], 0.5, 0xA3));
        let g = DTensor::from_host(&gpu, &Tensor::random_seeded(&[b, t, d], 1.0, 0xA4));

        // `chunk` here selects the path, not just the blocking: 0 is the only value
        // the fused kernels decline, so it is the reference. See `fused_chunk`.
        let run = |chunk: usize| {
            let mut dev = MLstm::from_cpu(&gpu, &proto);
            dev.set_chunk(chunk);
            let y = dev.forward_alloc(&gpu, &x).to_host(&gpu).data;
            let dx = dev.backward_alloc(&gpu, &g).to_host(&gpu).data;
            let mut cfg = AdamCfg::new(1e-3, 0.01);
            cfg.t = 1;
            dev.step(&gpu, &cfg);
            let wf = dev.lin_f.w.to_host(&gpu).data; // rides the decay path
            let wout = dev.lin_out.w.to_host(&gpu).data; // rides the value path
            (y, dx, wf, wout)
        };

        // The fusion algebra is what is on trial here, so this runs the SCALAR fused
        // kernels: their dots are fp32 and the comparison against the op-at-a-time
        // path stays exact to fp32 tolerance. The tensor-core dots are a separate,
        // deliberately looser question — see `mlstm_fused_mma_matches_scalar`.
        let _mma = with_mma(false);

        let (y0, dx0, wf0, wo0) = run(0); // op-at-a-time, single chunk
        let (y1, dx1, wf1, wo1) = run(32); // fused

        // The two sides no longer share a storage dtype. The fused path keeps its
        // saved q/k/v and ytil in bf16 (the reference's DTYPE tensors), while the
        // op-at-a-time path is fp32 throughout — so this compares the fusion algebra
        // ACROSS a precision boundary, and the floor is bf16's half-ulp (2^-8), not
        // fp32 reassociation. `GPU_NO_BF16=1` puts both sides back on fp32 and the
        // original tolerances apply.
        let slab = if gpu.kernels.slab_bf16 { 8.0 } else { 1.0 };
        assert_close(&y1, &y0, 2e-3 * slab, "y");
        assert_close(&dx1, &dx0, 2e-3 * slab, "dx");
        // The weights come out of an Adam step, so they sit at ~1e-2 while the two
        // paths' GEMMs reassociate at ~1e-3 relative. An ABSOLUTE 2e-5 on values of
        // that size is really a 4e-4 relative demand, tighter than fp32 summation
        // order guarantees — which is what made this test intermittent. Compare
        // against the tensor's own scale, as `mlstm_fused_mma_matches_scalar` does.
        // Adam divides by √v̂, which amplifies a small gradient difference on a
        // near-zero-gradient weight, so the weight check needs more headroom than
        // the activations above.
        assert_close_rel(&wf1, &wf0, 5e-3 * slab, "wf");
        assert_close_rel(&wo1, &wo0, 5e-3 * slab, "w_out");
    }

    /// Serializes the tests that select a kernel path through the process-global
    /// mma flag. Cargo runs a suite's tests on parallel threads in ONE process, so
    /// the flag is shared: without this lock, `mlstm_fused_matches_legacy` (which
    /// wants the scalar dots) and `mlstm_fused_mma_matches_scalar` (which wants the
    /// tensor-core dots) race, and whichever loses runs against the other's kernel
    /// and misses its tolerance. That was an intermittent failure in roughly half
    /// of full-suite runs while passing whenever either test ran alone.
    static MMA_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Holds the flag for the caller's scope and restores it on exit, panic or not,
    /// so one test's A/B neither races nor leaks into the next.
    struct MmaGuard(Option<std::sync::MutexGuard<'static, ()>>);
    impl Drop for MmaGuard {
        fn drop(&mut self) {
            super::ops::set_mma_enabled(true);
            // Release the lock only after the flag is back to its default.
            self.0.take();
        }
    }
    fn with_mma(on: bool) -> MmaGuard {
        // A poisoned lock just means some earlier test panicked; the flag is reset
        // by that test's own guard, so the state is still sound to take over.
        let g = MMA_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        super::ops::set_mma_enabled(on);
        MmaGuard(Some(g))
    }

    /// The tensor-core (`mma.sync`, TF32) forward against the scalar fp32 one, at
    /// the backbone's real shape and with a short final chunk.
    ///
    /// The two kernels implement the *same* algorithm on the same shared-memory
    /// plan; the only difference is that the three contractions (`Q·Kᵀ`, `(D̄⊙S)·V`,
    /// `Q·C_prevᵀ`) run on the tensor cores, which round their inputs to TF32 — 10
    /// mantissa bits instead of 24, fp32 exponent and fp32 accumulate. So this is
    /// not an exactness check and must not be tightened into one: it asserts that
    /// the mma fragment layouts, the zero-padding of the ragged chunk, and the
    /// fused decay/mask epilogue are all *right*, to a tolerance that a wrong
    /// fragment index (which garbles the result outright, not by 1e-3) cannot pass.
    ///
    /// It also covers what the scalar path cannot reach on its own: a `dhv`/`dqk`
    /// that is not a multiple of the mma tile, and a `len` shorter than one tile.
    #[test]
    fn mlstm_fused_mma_matches_scalar() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        if !gpu.kernels.has_mma {
            eprintln!("skipping: device has no tensor cores (needs sm_80+)");
            return;
        }
        let (b, t, inp, d, heads, dqk) = (2, 200, 64, 512, 8, 64); // dhv = 64
        assert!(
            t % super::ops::FUSED_MAX_L != 0,
            "the last chunk must be short"
        );

        let mut proto = CpuMLstm::new(inp, d, heads, dqk);
        proto.wi = Tensor::random(&[inp, heads], 0.3);
        proto.wf = Tensor::random(&[inp, heads], 0.3);

        let x = DTensor::from_host(&gpu, &Tensor::random(&[b, t, inp], 0.5));
        let g = DTensor::from_host(&gpu, &Tensor::random(&[b, t, d], 1.0));

        let run = |mma: bool| {
            let _guard = with_mma(mma);
            let mut dev = MLstm::from_cpu(&gpu, &proto);
            dev.set_chunk(32);
            let y = dev.forward_alloc(&gpu, &x).to_host(&gpu).data;
            let dx = dev.backward_alloc(&gpu, &g).to_host(&gpu).data;
            (y, dx)
        };

        let (y0, dx0) = run(false); // scalar fp32 dots — the oracle
        let (y1, dx1) = run(true); // tensor-core TF32 dots

        // TF32 error is relative to the size of the DOT, not of the element, so the
        // scale to divide by is the tensor's own magnitude — a per-element relative
        // check would explode on the elements that happen to sit near zero.
        let close = |got: &[f32], want: &[f32], tol: f32, what: &str| {
            let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let worst = got
                .iter()
                .zip(want)
                .fold(0.0f32, |m, (&a, &b)| m.max((a - b).abs()));
            println!(
                "{what}: worst |mma - scalar| {worst:.3e} on a scale of {scale:.3e} -> {:.2e} relative",
                worst / scale
            );
            assert!(
                worst / scale < tol,
                "{what}: worst |mma - scalar| {worst:.3e} vs scale {scale:.3e} exceeds {tol:.0e}"
            );
        };
        close(&y1, &y0, 5e-3, "y");
        close(&dx1, &dx0, 5e-3, "dx");
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
        let (b, t, inp, d, heads, dqk) = (2, 20, 5, 8, 2, 4);

        let mut cpu = CpuMLstm::new(inp, d, heads, dqk);
        cpu.wi = Tensor::random(&[inp, heads], 0.3);
        cpu.wf = Tensor::random(&[inp, heads], 0.3);
        let mut dev = MLstm::from_cpu(&gpu, &cpu);
        dev.set_chunk(6); // 6 + 6 + 6 + 2

        let x = Tensor::random(&[b, t, inp], 0.5);
        let g = Tensor::random(&[b, t, d], 1.0);

        let y_cpu = cpu.forward(&x);
        let y_dev = dev.forward_alloc(&gpu, &DTensor::from_host(&gpu, &x));
        assert_close(&y_dev.to_host(&gpu).data, &y_cpu.data, 3e-3, "y");

        let dx_cpu = cpu.backward(&g);
        let dx_dev = dev.backward_alloc(&gpu, &DTensor::from_host(&gpu, &g));
        assert_close(&dx_dev.to_host(&gpu).data, &dx_cpu.data, 3e-3, "dx");

        let mut cfg = AdamCfg::new(1e-3, 0.01);
        cfg.t = 1;
        cpu.step(&cfg);
        dev.step(&gpu, &cfg);
        assert_close(&dev.lin_q.w.to_host(&gpu).data, &cpu.wq.data, 3e-3, "wq");
        assert_close(&dev.lin_f.w.to_host(&gpu).data, &cpu.wf.data, 3e-3, "wf");
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
    /// level so a failure points at `mlstm_fw_C`'s `CARRY` seeding rather than at
    /// anything layered above it. `m` is the stabilizer: seeding it wrong does not
    /// crash and produces no NaN, it silently rescales every value in the chunk — so
    /// comparing `ytil` (the kernel's output) across the split is the only thing that
    /// actually catches it.
    #[test]
    fn mlstm_chunked_carry_matches_whole() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let _mma = with_mma(false); // scalar dots, so the split stays an fp32 question
        let (bh, dqk, dhv, l) = (4, 64, 64, 32);
        let parts = [64usize, 64, 32]; // ragged: the last chunk is short
        let t: usize = parts.iter().sum();

        let mk = |dims: &[usize], seed: u64| {
            DTensor::from_host(&gpu, &Tensor::random_seeded(dims, 0.5, seed))
        };
        let q = mk(&[bh, t, dqk], 0xC1);
        let k = mk(&[bh, t, dqk], 0xC2);
        let v = mk(&[bh, t, dhv], 0xC3);
        let ig = mk(&[bh, t], 0xC4);
        let fg = mk(&[bh, t], 0xC5);
        let slab = |src: &DTensor, _dims: &[usize]| ops::SlabBuf::from_f32(&gpu, src.dup(&gpu));
        // `ytil` is slab-typed; widen it to fp32 to compare.
        let ytil_host = |f: &ops::MlstmFused, n: usize| {
            let mut scratch = DTensor::uninit(&gpu, &[n]);
            f.ytil.as_f32(&gpu, &mut scratch).to_host(&gpu).data
        };

        // Reference: one call over the whole sequence.
        let whole = ops::mlstm_fused_fw(
            &gpu,
            &slab(&q, &[bh, t, dqk]),
            &slab(&k, &[bh, t, dqk]),
            &slab(&v, &[bh, t, dhv]),
            &ig,
            &fg,
            l,
            None,
        );
        let want = ytil_host(&whole, bh * t * dhv);

        // Chunked: slice the time axis, carrying the state across the borders.
        let mut got: Vec<f32> = Vec::with_capacity(bh * t * dhv);
        let mut state: Option<ops::MlstmState> = None;
        let mut off = 0;
        let mut per_chunk: Vec<Vec<f32>> = Vec::new();
        for &c in &parts {
            let cut = |src: &DTensor, w: usize| {
                let h = src.to_host(&gpu);
                let mut out = Vec::with_capacity(bh * c * w);
                for b in 0..bh {
                    let base = b * t * w + off * w;
                    out.extend_from_slice(&h.data[base..base + c * w]);
                }
                DTensor::from_host(&gpu, &Tensor::new(&[bh, c, w], out))
            };
            let (qc, kc, vc) = (cut(&q, dqk), cut(&k, dqk), cut(&v, dhv));
            let (igc, fgc) = (cut(&ig, 1), cut(&fg, 1));
            let f = ops::mlstm_fused_fw(
                &gpu,
                &slab(&qc, &[bh, c, dqk]),
                &slab(&kc, &[bh, c, dqk]),
                &slab(&vc, &[bh, c, dhv]),
                &igc.reshaped(&[bh, c]),
                &fgc.reshaped(&[bh, c]),
                l,
                state.as_ref(),
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
            DTensor::from_host(&gpu, &Tensor::new(&[b, len, w], o))
        };

        // Reference: one call over the whole sequence.
        // Internal chunk length 2, so each call has NC > 1. At the default (256) a
        // 6-step call is a single internal chunk, `is_last` is true for it either way,
        // and the CARRY path under test is never reached.
        let mut whole = MLstm::from_cpu(&gpu, &proto);
        let _ = whole.forward_alloc(&gpu, &DTensor::from_host(&gpu, &x));
        let want = whole
            .backward_alloc(&gpu, &DTensor::from_host(&gpu, &g))
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
            for &(c0, len) in &parts {
                let _ = part.forward_alloc(&gpu, &cut(&x, c0, len, inp));
            }
            // Backward unwinds right to left, starting with no gradient from the right.
            part.reset_bptt(&gpu);
            let mut pieces: Vec<Vec<f32>> = vec![Vec::new(); parts.len()];
            for (i, &(c0, len)) in parts.iter().enumerate().rev() {
                pieces[i] = part
                    .backward_alloc(&gpu, &cut(&g, c0, len, d))
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
