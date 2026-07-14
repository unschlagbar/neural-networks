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

use super::block::Cell;
use super::{DTensor, Gpu, linear::Linear, ops, rms_norm::RmsNorm};
use crate::nn2::optim::AdamCfg;
use crate::tensor::Tensor;

/// Chunk length: `config::MLSTM_CHUNK`, overridable with `MLSTM_CHUNK=<L>` for A/B
/// runs (0 = single-chunk). Resolved once — the env read must not sit in forward.
fn chunk_len() -> usize {
    static L: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
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
    c0: usize,  // chunk start in T
    len: usize, // chunk length (the last chunk may be short)
    bvec: DTensor, // [BH, L]  b_t: carried state → row t
    avec: DTensor, // [BH, L]  a_j: row j → outgoing state
    qn: DTensor,   // [BH, L]  (intra + inter)
    psi: DTensor,  // [BH, L]
    /// Per-row stabilizer. Backward needs it because ψ = max(|qn|, exp(−m)), so the
    /// branch "did qn win the max?" is against exp(−m), not against 1.
    m: DTensor,    // [BH, L]
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
    qh: DTensor,  // [BH, T, dqk]
    kh: DTensor,  // [BH, T, dqk]  (already ×1/√dqk)
    vh: DTensor,  // [BH, T, dhv]
    // The forget-gate logit is still needed in backward (`revcumsum_dlogsig` chains
    // dfc through logσ'); the input-gate logit is not — it only ever fed the D̄
    // build, which now happens once, in forward.
    fgh: DTensor, // [BH, T]  forget-gate logit (head-major)
    chunks: Vec<Chunk>,
    o: DTensor,   // [N, d]  (post-sigmoid)
    yhat: DTensor, // [N, d]
}

/// Forward intermediates of the **fused** path. Far smaller than `Saved`: the
/// per-chunk `[BH, L, L]` decay matrices do not exist — backward rebuilds them in
/// shared memory (see `ops::MlstmFused`).
struct SavedFused {
    b: usize,
    t: usize,
    qh: DTensor,
    kh: DTensor,
    vh: DTensor,
    igh: DTensor,
    fgh: DTensor,
    fused: ops::MlstmFused,
    o: DTensor,
    yhat: DTensor,
}

/// Which forward ran, and hence which backward must.
enum Cache {
    Fused(SavedFused),
    Legacy(Saved),
}

/// `MLSTM_LEGACY=1` forces the op-at-a-time chunk loop — the A/B baseline for
/// `mlstm_fused_bench`, and the escape hatch if a fused kernel ever misbehaves.
fn legacy_forced() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
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

    saved: Option<Cache>,
}

impl MLstm {
    /// Build from a CPU cell's host weights (all 15 parameter tensors uploaded).
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        gpu: &Gpu,
        input_size: usize, d: usize, heads: usize, dqk: usize,
        wq: &Tensor, wk: &Tensor, wv: &Tensor, wo: &Tensor, wi: &Tensor, wf: &Tensor,
        bq: &Tensor, bk: &Tensor, bv: &Tensor, bo: &Tensor, bi: &Tensor, bf: &Tensor,
        w_out: &Tensor, b_out: &Tensor, gamma: &Tensor,
    ) -> Self {
        let dhv = d / heads;
        Self {
            input_size, d, heads, dqk, dhv,
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
            saved: None,
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
            &m(&c.wq), &m(&c.wk), &m(&c.wv), &m(&c.wo), &m(&c.wi), &m(&c.wf),
            &v(&c.bq), &v(&c.bk), &v(&c.bv), &v(&c.bo), &v(&c.bi), &v(&c.bf),
            &m(&c.w_out.weights),
            &v(&c.w_out.biases),
            &v(&c.head_norm.gamma),
        )
    }

    /// Upload a CPU cell (weights are copied; grads/moments start at zero).
    pub fn from_cpu(gpu: &Gpu, c: &crate::nn2::MLstm) -> Self {
        Self::from_parts(
            gpu, c.input_size, c.d, c.heads, c.dqk,
            &c.wq, &c.wk, &c.wv, &c.wo, &c.wi, &c.wf,
            &c.bq, &c.bk, &c.bv, &c.bo, &c.bi, &c.bf,
            &c.w_out, &c.b_out, &c.gamma,
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
    fn fused_chunk(&self, t: usize) -> Option<usize> {
        if legacy_forced() || self.chunk == 0 {
            return None;
        }
        let l = self.chunk.min(ops::FUSED_MAX_L).min(t).max(1);
        ops::mlstm_fused_supported(l, self.dqk, self.dhv).then_some(l)
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
    pub fn forward(&mut self, gpu: &Gpu, x: &DTensor) -> DTensor {
        assert_eq!(x.rank, 3, "MLstm::forward expects [B, T, in]");
        let (b, t, inp) = (x.shape[0], x.shape[1], x.shape[2]);
        assert_eq!(inp, self.input_size, "MLstm::forward — input width mismatch");
        let (d, h, dqk, dhv) = (self.d, self.heads, self.dqk, self.dhv);
        let (n, bh) = (b * t, b * h);

        // Projections on the flat [N, in] view.
        let xf = x.dup(gpu).reshaped(&[n, inp]);
        let q = self.lin_q.forward(gpu, &xf);
        let mut k = self.lin_k.forward(gpu, &xf);
        ops::scale_(gpu, &mut k, self.inv_sqrt_dqk);
        let v = self.lin_v.forward(gpu, &xf);
        let mut o = self.lin_o.forward(gpu, &xf);
        ops::sigmoid_(gpu, &mut o);
        let ig = self.lin_i.forward(gpu, &xf); // [N, H]
        let fg = self.lin_f.forward(gpu, &xf); // [N, H]

        // Head-major reorg for the per-(b,h) batched matmuls.
        let qh = ops::head_gather(gpu, &q, b, h, t, dqk); // [BH, T, dqk]
        let kh = ops::head_gather(gpu, &k, b, h, t, dqk);
        let vh = ops::head_gather(gpu, &v, b, h, t, dhv); // [BH, T, dhv]
        let igh = ops::head_gather(gpu, &ig, b, h, t, 1).reshaped(&[bh, t]); // [BH, T]
        let fgh = ops::head_gather(gpu, &fg, b, h, t, 1).reshaped(&[bh, t]);

        // The fused kernels do the whole chunkwise core — states and all chunks — in
        // three launches. Everything before and after (projections, head norm, the
        // o-gate, the output projection) is shared with the path below.
        if let Some(l) = self.fused_chunk(t) {
            let fused = ops::mlstm_fused_fw(gpu, &qh, &kh, &vh, &igh, &fgh, l);
            let h_tilde = ops::head_scatter(gpu, &fused.ytil, b, h, t, dhv); // [N, d]
            let yhat = self.headnorm.forward(gpu, &h_tilde);
            let hconcat = ops::mul(gpu, &o, &yhat);
            let out = self.lin_out.forward(gpu, &hconcat);
            self.saved = Some(Cache::Fused(SavedFused {
                b, t, qh, kh, vh, igh, fgh, fused, o, yhat,
            }));
            return out.reshaped(&[b, t, d]);
        }

        // Recurrent state carried across chunks (stabilized, as on the CPU).
        let mut c_state = DTensor::zeros(gpu, &[bh, dhv, dqk]);
        let mut n_state = DTensor::zeros(gpu, &[bh, 1, dqk]);
        let mut m_state = DTensor::zeros(gpu, &[bh]);

        let spans = self.chunk_spans(t);
        let last_span = spans.len() - 1;
        let mut ytil = DTensor::uninit(gpu, &[bh, t, dhv]);
        let mut chunks = Vec::with_capacity(spans.len());

        for (ci, &(c0, len)) in spans.iter().enumerate() {
            let qc = ops::slice_t(gpu, &qh, c0, len); // [BH, L, dqk]
            let kc = ops::slice_t(gpu, &kh, c0, len);
            let vc = ops::slice_t(gpu, &vh, c0, len); // [BH, L, dhv]
            let igc = ops::slice_t(gpu, &igh.dup(gpu).reshaped(&[bh, t, 1]), c0, len)
                .reshaped(&[bh, len]);
            let fgc = ops::slice_t(gpu, &fgh.dup(gpu).reshaped(&[bh, t, 1]), c0, len)
                .reshaped(&[bh, len]);

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
                let g = ops::slice_t(gpu, &bvec.dup(gpu).reshaped(&[bh, len, 1]), len - 1, 1)
                    .reshaped(&[bh]); // [BH]
                let va = ops::mul_rows(gpu, &vc, &avec, dhv); // [BH, L, dhv]
                let mut c_new = ops::matmul_batched_tn(gpu, &va, &kc); // [BH, dhv, dqk]
                let a3 = avec.dup(gpu).reshaped(&[bh, len, 1]);
                let mut n_new = ops::matmul_batched_tn(gpu, &a3, &kc); // [BH, 1, dqk]
                ops::mul_rows_add(gpu, &mut c_new, &c_state, &g, dhv * dqk);
                ops::mul_rows_add(gpu, &mut n_new, &n_state, &g, dqk);
                c_state = c_new;
                n_state = n_new;
                // m_new = the chunk's last-row stabilizer.
                m_state = ops::slice_t(gpu, &m.dup(gpu).reshaped(&[bh, len, 1]), len - 1, 1)
                    .reshaped(&[bh]);
            }

            chunks.push(Chunk { c0, len, bvec, avec, qn, psi, m, num, dbar, ds, inter });
        }

        // Back to position-major, head-norm, o-gate, output projection.
        let h_tilde = ops::head_scatter(gpu, &ytil, b, h, t, dhv); // [N, d]
        let yhat = self.headnorm.forward(gpu, &h_tilde);
        let hconcat = ops::mul(gpu, &o, &yhat); // o ⊙ ŷ  [N, d]
        let out = self.lin_out.forward(gpu, &hconcat); // [N, d]

        // `o`/`yhat` are unused after `mul`, so move (not dup) them into the cache.
        self.saved = Some(Cache::Legacy(Saved { b, t, qh, kh, vh, fgh, chunks, o, yhat }));
        out.reshaped(&[b, t, d])
    }

    /// Backward over `[B, T, d]` → `dx` `[B, T, in]`. Accumulates all grads.
    ///
    /// Chunks are swept in reverse, carrying `dC`/`dn` (the grad wrt the state a
    /// chunk hands to its successor) backwards the way forward carried `C`/`n`
    /// forwards — BPTT over chunks, with the parallel form inside each.
    pub fn backward(&mut self, gpu: &Gpu, dy: &DTensor) -> DTensor {
        // `take`, not `as_ref`: the cache holds a window's activations, and dropping
        // them at the end of this call (rather than when the next forward overwrites
        // the field) keeps them from staying resident across the optimizer step.
        match self.saved.take().expect("MLstm::backward before forward") {
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

        let dy_flat = dy.dup(gpu).reshaped(&[n, d]);
        let d_hconcat = self.lin_out.backward(gpu, &dy_flat);
        let (do_pre, d_yhat) = ops::ogate_bwd(gpu, &d_hconcat, &sv.o, &sv.yhat);
        let d_h_tilde = self.headnorm.backward(gpu, &d_yhat);
        let d_ytil = ops::head_gather(gpu, &d_h_tilde, b, h, t, dhv); // [BH, T, dhv]

        let (dqh, dkh, dvh, digh, dfgh) = ops::mlstm_fused_bw(
            gpu, &sv.fused, &sv.qh, &sv.kh, &sv.vh, &sv.igh, &sv.fgh, &d_ytil,
        );

        let dq = ops::head_scatter(gpu, &dqh, b, h, t, dqk); // [N, dqk·H]
        let mut dk = ops::head_scatter(gpu, &dkh, b, h, t, dqk);
        ops::scale_(gpu, &mut dk, self.inv_sqrt_dqk); // k = (·)·1/√dqk
        let dv = ops::head_scatter(gpu, &dvh, b, h, t, dhv);
        let d_ig = ops::head_scatter(gpu, &digh.reshaped(&[bh, t, 1]), b, h, t, 1);
        let d_fg = ops::head_scatter(gpu, &dfgh.reshaped(&[bh, t, 1]), b, h, t, 1);

        let mut dxf = self.lin_q.backward(gpu, &dq);
        dxf = ops::add(gpu, &dxf, &self.lin_k.backward(gpu, &dk));
        dxf = ops::add(gpu, &dxf, &self.lin_v.backward(gpu, &dv));
        dxf = ops::add(gpu, &dxf, &self.lin_o.backward(gpu, &do_pre));
        dxf = ops::add(gpu, &dxf, &self.lin_i.backward(gpu, &d_ig));
        dxf = ops::add(gpu, &dxf, &self.lin_f.backward(gpu, &d_fg));
        dxf.reshaped(&[b, t, inp])
    }

    fn backward_legacy(&mut self, gpu: &Gpu, dy: &DTensor, sv: Saved) -> DTensor {
        let (d, h, dqk, dhv, inp) = (self.d, self.heads, self.dqk, self.dhv, self.input_size);
        let sv = &sv;
        let (b, t) = (sv.b, sv.t);
        let (n, bh) = (b * t, b * h);

        let dy_flat = dy.dup(gpu).reshaped(&[n, d]);

        // Output projection + o-gate.
        let d_hconcat = self.lin_out.backward(gpu, &dy_flat); // [N, d]
        let (do_pre, d_yhat) = ops::ogate_bwd(gpu, &d_hconcat, &sv.o, &sv.yhat);

        // Head-norm backward → d_h_tilde, then head-gather to head-major d_ytil.
        let d_h_tilde = self.headnorm.backward(gpu, &d_yhat); // [N, d]
        let d_ytil = ops::head_gather(gpu, &d_h_tilde, b, h, t, dhv); // [BH, T, dhv]

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

            let qc = ops::slice_t(gpu, &sv.qh, c0, len); // [BH, L, dqk]
            let kc = ops::slice_t(gpu, &sv.kh, c0, len);
            let vc = ops::slice_t(gpu, &sv.vh, c0, len); // [BH, L, dhv]
            let fgc = ops::slice_t(gpu, &sv.fgh.dup(gpu).reshaped(&[bh, t, 1]), c0, len)
                .reshaped(&[bh, len]);
            let d_ytil_c = ops::slice_t(gpu, &d_ytil, c0, len); // [BH, L, dhv]

            // ỹ = num/ψ  → d_num, d_qn  (num/ψ/qn all include the inter term).
            let (d_num, d_qn) = ops::div_rows_bwd(gpu, &d_ytil_c, &ch.num, &ch.psi, &ch.qn, &ch.m, dhv);

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

            // --- state-update path: how this chunk fed the NEXT chunk's state -----
            //   C_out = g·C_in + (a⊙V)ᵀ·K ,  n_out = g·n_in + Σ_j a_j k_j
            // Skipped for the last chunk (dc_carry / dn_carry are zero there).
            let (mut dc_in, mut dn_in) = (None, None);
            if !is_last {
                let a3 = ch.avec.dup(gpu).reshaped(&[bh, len, 1]);
                let va = ops::mul_rows(gpu, &vc, &ch.avec, dhv); // a⊙V  [BH, L, dhv]

                // C_out = Vaᵀ·K:  dVa = K·dC_outᵀ ;  dK += Va·dC_out.
                let dva = ops::matmul_batched_nt(gpu, &kc, &dc_carry); // [BH, L, dhv]
                dkc = ops::add(gpu, &dkc, &ops::matmul_batched_nn(gpu, &va, &dc_carry));
                // Va = a⊙V:  dV += a⊙dVa ;  da += Σ_p dVa·V.
                ops::mul_rows_add(gpu, &mut dvc, &dva, &ch.avec, dhv);
                ops::row_dot_add(gpu, &mut da, &dva, &vc, dhv);

                // n_out = Σ_j a_j k_j:  dK += a ⊗ dn_out ;  da += K·dn_outᵀ.
                dkc = ops::add(gpu, &dkc, &ops::matmul_batched_nn(gpu, &a3, &dn_carry));
                let da_n = ops::matmul_batched_nt(gpu, &kc, &dn_carry); // [BH, L, 1]
                da = ops::add(gpu, &da, &da_n.reshaped(&[bh, len]));

                // g·state: dg = Σ(dC_out ⊙ C_in) + Σ(dn_out ⊙ n_in); dstate_in += g·dC_out.
                // Only reachable when this chunk HAS an incoming state (ci > 0) — for
                // chunk 0 the state is zero, so g contributes nothing and there is no
                // predecessor to hand dC_in to.
                if let Some(it) = &ch.inter {
                    let g = ops::slice_t(
                        gpu, &ch.bvec.dup(gpu).reshaped(&[bh, len, 1]), len - 1, 1,
                    ).reshaped(&[bh]);
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
                    ops::unslice_t(gpu, &mut dg_pad, &dg.reshaped(&[bh, 1, 1]), len - 1);
                    db = ops::add(gpu, &db, &dg_pad.reshaped(&[bh, len]));
                }
            }

            // --- inter path: how this chunk READ its incoming state ---------------
            //   num += b⊙(Q·C_inᵀ) ,  qn += b⊙(Q·n_inᵀ)
            if let Some(it) = &ch.inter {
                // db from both products (they are saved pre-b, which is what db needs).
                ops::row_dot_add(gpu, &mut db, &d_num, &it.inter_num, dhv);
                ops::row_dot_add(gpu, &mut db, &d_qn, &it.inter_qn, 1);

                let d_inter_num = ops::mul_rows(gpu, &d_num, &ch.bvec, dhv); // [BH, L, dhv]
                let d_inter_qn = ops::mul_rows(gpu, &d_qn, &ch.bvec, 1).reshaped(&[bh, len, 1]);

                // dQ from both readouts.
                dqc = ops::add(gpu, &dqc, &ops::matmul_batched_nn(gpu, &d_inter_num, &it.c_prev));
                dqc = ops::add(gpu, &dqc, &ops::matmul_batched_nn(gpu, &d_inter_qn, &it.n_prev));

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

            ops::unslice_t(gpu, &mut dqh, &dqc, c0);
            ops::unslice_t(gpu, &mut dkh, &dkc, c0);
            ops::unslice_t(gpu, &mut dvh, &dvc, c0);
            ops::unslice_t(gpu, &mut digh, &dig.reshaped(&[bh, len, 1]), c0);
            ops::unslice_t(gpu, &mut d_fgh3, &d_fgc.reshaped(&[bh, len, 1]), c0);

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

        // Projection backward; sum the input grads (all share the saved xf).
        let mut dxf = self.lin_q.backward(gpu, &dq);
        dxf = ops::add(gpu, &dxf, &self.lin_k.backward(gpu, &dk));
        dxf = ops::add(gpu, &dxf, &self.lin_v.backward(gpu, &dv));
        dxf = ops::add(gpu, &dxf, &self.lin_o.backward(gpu, &do_pre));
        dxf = ops::add(gpu, &dxf, &self.lin_i.backward(gpu, &d_ig));
        dxf = ops::add(gpu, &dxf, &self.lin_f.backward(gpu, &d_fg));
        dxf.reshaped(&[b, t, inp])
    }

    /// Every learnable tensor, in a fixed order (used by checkpoint save/load).
    pub fn params_mut(&mut self) -> Vec<&mut DTensor> {
        let mut v = Vec::new();
        for l in [&mut self.lin_q, &mut self.lin_k, &mut self.lin_v, &mut self.lin_o,
                  &mut self.lin_i, &mut self.lin_f, &mut self.lin_out] {
            v.extend(l.params_mut());
        }
        v.extend(self.headnorm.params_mut());
        v
    }

    pub fn zero_grad(&mut self, gpu: &Gpu) {
        for l in [&mut self.lin_q, &mut self.lin_k, &mut self.lin_v, &mut self.lin_o,
                  &mut self.lin_i, &mut self.lin_f, &mut self.lin_out] {
            l.zero_grad(gpu);
        }
        self.headnorm.zero_grad(gpu);
    }

    /// AdamW step: projection + output matrices decay; biases and head-norm γ
    /// don't (all handled by the sub-layers). Clears the grads.
    pub fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        for l in [&mut self.lin_q, &mut self.lin_k, &mut self.lin_v, &mut self.lin_o,
                  &mut self.lin_i, &mut self.lin_f, &mut self.lin_out] {
            l.step(gpu, cfg);
        }
        self.headnorm.step(gpu, cfg);
    }
}

impl Cell for MLstm {
    fn forward(&mut self, gpu: &Gpu, x: &DTensor) -> DTensor { MLstm::forward(self, gpu, x) }
    fn backward(&mut self, gpu: &Gpu, dy: &DTensor) -> DTensor { MLstm::backward(self, gpu, dy) }
    fn zero_grad(&mut self, gpu: &Gpu) { MLstm::zero_grad(self, gpu) }
    fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) { MLstm::step(self, gpu, cfg) }
    fn params_mut(&mut self) -> Vec<&mut DTensor> { MLstm::params_mut(self) }
    fn wants_post_cell_norm(&self) -> bool { false }
    fn to_nn_block(
        &self,
        gpu: &Gpu,
        hidden: usize,
        up: usize,
        pre_norm1: crate::nn::rms_norm::RMSNorm,
        _post_cell_norm: Option<crate::nn::rms_norm::RMSNorm>,
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

    /// Single-chunk parallel forward+backward+step must match the CPU scalar
    /// recurrence (`nn2::MLstm`) from identical weights. The CPU backward is
    /// itself FD-verified, so a GPU-vs-CPU grad match is the (tighter) check.
    #[test]
    fn mlstm_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else { return };
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
        let y_dev = dev.forward(&gpu, &DTensor::from_host(&gpu, &x));
        assert_close(&y_dev.to_host(&gpu).data, &y_cpu.data, 3e-3, "y");

        // Backward
        let dx_cpu = cpu.backward(&g);
        let dx_dev = dev.backward(&gpu, &DTensor::from_host(&gpu, &g));
        assert_close(&dx_dev.to_host(&gpu).data, &dx_cpu.data, 3e-3, "dx");

        // One AdamW step; compare representative updated parameters.
        let mut cfg = AdamCfg::new(1e-3, 0.01);
        cfg.t = 1;
        cpu.step(&cfg);
        dev.step(&gpu, &cfg);
        // (weights live in the Linear sub-layers; check q, v, out projections + γ)
        assert_close(&dev.lin_q.w.to_host(&gpu).data, &cpu.wq.data, 3e-3, "wq");
        assert_close(&dev.lin_v.w.to_host(&gpu).data, &cpu.wv.data, 3e-3, "wv");
        assert_close(&dev.lin_out.w.to_host(&gpu).data, &cpu.w_out.data, 3e-3, "w_out");
        assert_close(&dev.headnorm.gamma.to_host(&gpu).data, &cpu.gamma.data, 3e-3, "gamma");
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
        let Some(gpu) = super::super::test_gpu() else { return };
        let (b, t, inp, d, heads, dqk) = (2, 20, 5, 8, 2, 4);

        let mut proto = CpuMLstm::new(inp, d, heads, dqk);
        // Non-trivial gate weights so the decay/stabilizer path is exercised — with
        // zero gates every chunk would decay identically and the test would pass
        // vacuously.
        proto.wi = Tensor::random(&[inp, heads], 0.3);
        proto.wf = Tensor::random(&[inp, heads], 0.3);

        let x = Tensor::random(&[b, t, inp], 0.5);
        let g = Tensor::random(&[b, t, d], 1.0);
        let dx = DTensor::from_host(&gpu, &x);
        let dg = DTensor::from_host(&gpu, &g);

        // Reference: one chunk over the whole sequence.
        let run = |chunk: usize| {
            let mut dev = MLstm::from_cpu(&gpu, &proto);
            dev.set_chunk(chunk);
            let y = dev.forward(&gpu, &dx).to_host(&gpu).data;
            let dxo = dev.backward(&gpu, &dg).to_host(&gpu).data;
            let mut cfg = AdamCfg::new(1e-3, 0.01);
            cfg.t = 1;
            dev.step(&gpu, &cfg);
            // wf/wi ride the decay path; w_out rides the value path.
            let wf = dev.lin_f.w.to_host(&gpu).data;
            let wout = dev.lin_out.w.to_host(&gpu).data;
            (y, dxo, wf, wout)
        };

        let (y0, dx0, wf0, wo0) = run(0); // single chunk
        for l in [1, 3, 8, 16, 32] {
            let (y, dxo, wf, wo) = run(l);
            assert_close(&y, &y0, 2e-4, &format!("y (chunk {l})"));
            assert_close(&dxo, &dx0, 2e-4, &format!("dx (chunk {l})"));
            assert_close(&wf, &wf0, 2e-6, &format!("wf (chunk {l})"));
            assert_close(&wo, &wo0, 2e-6, &format!("w_out (chunk {l})"));
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
        let Some(gpu) = super::super::test_gpu() else { return };
        let (b, t, inp, d, heads, dqk) = (2, 200, 64, 512, 8, 64); // dhv = 64
        assert!(t % super::ops::FUSED_MAX_L != 0, "the last chunk must be short");

        let mut proto = CpuMLstm::new(inp, d, heads, dqk);
        proto.wi = Tensor::random(&[inp, heads], 0.3);
        proto.wf = Tensor::random(&[inp, heads], 0.3);

        let x = DTensor::from_host(&gpu, &Tensor::random(&[b, t, inp], 0.5));
        let g = DTensor::from_host(&gpu, &Tensor::random(&[b, t, d], 1.0));

        // `chunk` here selects the path, not just the blocking: 0 is the only value
        // the fused kernels decline, so it is the reference. See `fused_chunk`.
        let run = |chunk: usize| {
            let mut dev = MLstm::from_cpu(&gpu, &proto);
            dev.set_chunk(chunk);
            let y = dev.forward(&gpu, &x).to_host(&gpu).data;
            let dx = dev.backward(&gpu, &g).to_host(&gpu).data;
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
        assert_close(&y1, &y0, 2e-3, "y");
        assert_close(&dx1, &dx0, 2e-3, "dx");
        assert_close(&wf1, &wf0, 2e-5, "wf");
        assert_close(&wo1, &wo0, 2e-5, "w_out");
    }

    /// Restores the tensor-core path on scope exit, panic or not, so one test's A/B
    /// cannot leak its setting into whichever test the harness runs next.
    struct MmaGuard;
    impl Drop for MmaGuard {
        fn drop(&mut self) {
            super::ops::set_mma_enabled(true);
        }
    }
    fn with_mma(on: bool) -> MmaGuard {
        super::ops::set_mma_enabled(on);
        MmaGuard
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
        let Some(gpu) = super::super::test_gpu() else { return };
        if !gpu.kernels.has_mma {
            eprintln!("skipping: device has no tensor cores (needs sm_80+)");
            return;
        }
        let (b, t, inp, d, heads, dqk) = (2, 200, 64, 512, 8, 64); // dhv = 64
        assert!(t % super::ops::FUSED_MAX_L != 0, "the last chunk must be short");

        let mut proto = CpuMLstm::new(inp, d, heads, dqk);
        proto.wi = Tensor::random(&[inp, heads], 0.3);
        proto.wf = Tensor::random(&[inp, heads], 0.3);

        let x = DTensor::from_host(&gpu, &Tensor::random(&[b, t, inp], 0.5));
        let g = DTensor::from_host(&gpu, &Tensor::random(&[b, t, d], 1.0));

        let run = |mma: bool| {
            let _guard = with_mma(mma);
            let mut dev = MLstm::from_cpu(&gpu, &proto);
            dev.set_chunk(32);
            let y = dev.forward(&gpu, &x).to_host(&gpu).data;
            let dx = dev.backward(&gpu, &g).to_host(&gpu).data;
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
            println!("{what}: worst |mma - scalar| {worst:.3e} on a scale of {scale:.3e} -> {:.2e} relative", worst / scale);
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
        let Some(gpu) = super::super::test_gpu() else { return };
        let (b, t, inp, d, heads, dqk) = (2, 20, 5, 8, 2, 4);

        let mut cpu = CpuMLstm::new(inp, d, heads, dqk);
        cpu.wi = Tensor::random(&[inp, heads], 0.3);
        cpu.wf = Tensor::random(&[inp, heads], 0.3);
        let mut dev = MLstm::from_cpu(&gpu, &cpu);
        dev.set_chunk(6); // 6 + 6 + 6 + 2

        let x = Tensor::random(&[b, t, inp], 0.5);
        let g = Tensor::random(&[b, t, d], 1.0);

        let y_cpu = cpu.forward(&x);
        let y_dev = dev.forward(&gpu, &DTensor::from_host(&gpu, &x));
        assert_close(&y_dev.to_host(&gpu).data, &y_cpu.data, 3e-3, "y");

        let dx_cpu = cpu.backward(&g);
        let dx_dev = dev.backward(&gpu, &DTensor::from_host(&gpu, &g));
        assert_close(&dx_dev.to_host(&gpu).data, &dx_cpu.data, 3e-3, "dx");

        let mut cfg = AdamCfg::new(1e-3, 0.01);
        cfg.t = 1;
        cpu.step(&cfg);
        dev.step(&gpu, &cfg);
        assert_close(&dev.lin_q.w.to_host(&gpu).data, &cpu.wq.data, 3e-3, "wq");
        assert_close(&dev.lin_f.w.to_host(&gpu).data, &cpu.wf.data, 3e-3, "wf");
        assert_close(&dev.lin_out.w.to_host(&gpu).data, &cpu.w_out.data, 3e-3, "w_out");
    }
}
