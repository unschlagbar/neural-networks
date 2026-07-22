//! Device-resident hierarchical (HAT-style) model — GPU counterpart of
//! [`nn2::hierarchical::Hierarchical`](crate::nn2::hierarchical::Hierarchical).
//!
//! Three coupled stages, run phase-by-phase over a window of words:
//!
//!   1. encoder  — per word, `Embedding → sLSTM block → mLSTM block×(N-1)` (16
//!                 heads), read out `e_w` at the closing `[W]` step. Words are
//!                 the batch axis.
//!   2. backbone — `Linear → (sLSTM/mLSTM block)×N → Linear`, autoregressing over
//!                 the word embeddings as one sequence (batch 1, length = words).
//!   3. decoder  — per word, slot 0 is the injected backbone context, later slots
//!                 feed the previous char through the **tied** char table;
//!                 `sLSTM block → mLSTM block×(N-1)` (16 heads) `→ RMSNorm →
//!                 head → SoftCap`.
//!
//! The decoder's pre-head RMSNorm is the **only** stage-level norm, matching
//! `model.rs::build_hierarchical_model` (the blocks keep their internal norms).
//!
//! Everything — the tied char table, every block, the projections, the norm and
//! the head, plus all gradients and AdamW moments — lives in `DTensor`s. Index
//! bookkeeping (which row is a `[W]` step, which slot is a char) is computed on
//! the host and uploaded as id lists; only tensor *data* stays on the device.
//!
//! Checkpoints: `save`/`load` use the unified `NNM1` container (`src/format.rs`,
//! kind = Hierarchical) — the same named-section layout the CPU model produces,
//! so a GPU-trained model opens directly in the CPU sampler/probe (`hs` / `hp`).
//! Weights only; the AdamW moments are not persisted, so a resumed run restarts
//! them.

use std::collections::BTreeMap;
use std::io;

use super::block::{Block, BlockLike};
use super::{DTensor, Gpu, linear::Linear, mlstm::MLstm, ops, rms_norm::RmsNorm, slstm::SLstm};
use crate::format::{Meta, ModelKind, Writer};
use crate::gpu::{dt_matrix, dt_vec, tensor_from_matrix, tensor_from_slice};
use crate::nn::embedding::EmbeddingLayer;
use crate::nn::linear::LinearLayer;
use crate::nn::linear_nb::LinearNBLayer;
use crate::nn::mlstm_block::MLSTMBlock;
use crate::nn::rms_norm::RMSNorm;
use crate::nn::slstm_block::SLSTMBlock;
use crate::nn::soft_cap::SoftCapLayer;
use crate::nn_layer::NnLayer;
use crate::nn2::optim::AdamCfg;
use crate::sequential::Sequential;
use crate::tensor::Tensor;

/// Config for the hierarchical stack (mirrors `nn2::HierCfg`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HierCfg {
    pub vocab: usize,
    pub hc: usize, // char/context hidden (tied embedding + decoder width)
    pub wh: usize, // backbone width
    pub enc_blocks: usize,
    pub bb_blocks: usize, // alternates sLSTM (even) / mLSTM (odd)
    pub dec_blocks: usize,
    pub heads: usize, // mLSTM heads
    pub dqk: usize,
    pub w_token: usize,
    pub cap: f32,
}

/// SwiGLU inner width, derived per block from its own hidden width — `8·h/3`,
/// the paper default, exactly as `SequentialBuilder::{slstm,mlstm}_block` does.
/// It therefore differs between stages (e.g. 128 → 341, 384 → 1024).
#[inline]
pub fn up_of(hidden: usize) -> usize {
    hidden * 8 / 3
}

/// Per-word encoder: first block sLSTM, remaining blocks mLSTM (16 heads),
/// `e_w` read out at the `[W]` step.
pub struct WordEncoder {
    pub blocks: Vec<Box<dyn BlockLike>>,
}

impl WordEncoder {
    fn new(gpu: &Gpu, hc: usize, n: usize) -> Self {
        let dqk = hc / 16;
        Self {
            blocks: (0..n)
                .map(|i| {
                    if i == 0 {
                        Box::new(Block::from_cell(
                            gpu,
                            hc,
                            up_of(hc),
                            SLstm::new_rand(gpu, hc, hc),
                        )) as Box<dyn BlockLike>
                    } else {
                        Box::new(Block::from_cell(
                            gpu,
                            hc,
                            up_of(hc),
                            MLstm::new_rand(gpu, hc, hc, 16, dqk),
                        )) as Box<dyn BlockLike>
                    }
                })
                .collect(),
        }
    }
}

/// Partition word indices into length groups, so each group can run as a dense
/// `[words, tmax]` rectangle instead of every word being padded to the longest
/// word in the whole window.
///
/// Words are bucketed by `len.next_power_of_two()`: within a bucket the padding
/// is at most 2x (usually far less, since `tmax` is the bucket's ACTUAL longest
/// word, not the bucket's upper bound), and a 1..=16-byte word range collapses to
/// ~5 buckets. Exact-length buckets would remove padding entirely but would fire
/// ~17 rectangles of a few hundred rows each, and this backend is bound by cuBLAS
/// parallelism rather than launch count — small matrices would lose more than the
/// padding costs.
/// `no_group` (from `GPU_NO_GROUP=1`) puts every word in one group, which
/// reproduces the old single-rectangle behavior exactly — the A/B baseline for
/// benchmarking, and what `grouping_matches_single_rectangle` checks the grouped
/// path against.
///
/// Writes into `out`, reusing the inner `Vec`s across windows: a window fires
/// this twice (encoder + decoder) and a training run fires it forever, so the
/// bucket allocations are worth keeping.
fn group_by_len(lens: &[usize], no_group: bool, out: &mut Vec<Vec<usize>>) {
    for g in out.iter_mut() {
        g.clear();
    }
    if no_group {
        out.resize_with(1, Vec::new);
        out[0].extend(0..lens.len());
        return;
    }
    // Bucket key -> slot in `out`. The keys are collected first so the groups keep
    // the BTreeMap's ascending-length order: group order decides the order the
    // per-group losses are summed and the grads are scattered, and this stays
    // bit-identical to the version that returned `buckets.into_values()`.
    let mut slot: BTreeMap<usize, usize> = Default::default();
    for &l in lens {
        slot.insert(l.max(1).next_power_of_two(), 0);
    }
    let used = slot.len();
    for (i, v) in slot.values_mut().enumerate() {
        *v = i;
    }
    while out.len() < used {
        out.push(Vec::new());
    }
    for (w, &l) in lens.iter().enumerate() {
        out[slot[&l.max(1).next_power_of_two()]].push(w);
    }
    out.truncate(used);
}

/// Per-window host-side scratch, owned by the model so a training run allocates
/// it once instead of once per window. Everything here is index bookkeeping fed
/// to the gather/scatter/CE kernels; the ids are built as `u32` (what the
/// kernels take) so uploading them does not need a narrowing copy.
#[derive(Default)]
struct Scratch {
    enc_lens: Vec<usize>,
    dec_lens: Vec<usize>,
    enc_groups: Vec<Vec<usize>>,
    dec_groups: Vec<Vec<usize>>,
    /// `enc_layout[g]` is group `g`'s id rectangle + readout rows + `tmax`,
    /// built once in the encoder forward and reused by the encoder backward's
    /// re-forward (the two passes want byte-identical rectangles).
    enc_layout: Vec<EncGroup>,
    /// Decoder per-group buffers, cleared and refilled for each group.
    grp_ids: Vec<u32>,
    o_rows: Vec<u32>,
    char_rows: Vec<u32>,
    char_ids: Vec<u32>,
    targets: Vec<u32>,
    mask: Vec<u32>,
}

/// One encoder group's `[words, tmax]` id rectangle and `[W]`-step readout rows.
#[derive(Default)]
struct EncGroup {
    ids: Vec<u32>,
    readout: Vec<u32>,
    tmax: usize,
}

/// Fill `out` with the `[words, tmax]` id rectangle for one encoder group, plus
/// each word's `[W]`-step row (where `e_w` is read out) and the group's `tmax`.
///
/// Padding slots stay at id 0 — they are masked out of the readout, and the
/// buffers are cleared and refilled rather than reallocated per window.
fn enc_group_rows(
    tokens: &[usize],
    words: &[(usize, usize)],
    grp: &[usize],
    enc_lens: &[usize],
    w_token: usize,
    out: &mut EncGroup,
) {
    let tmax = grp.iter().map(|&w| enc_lens[w] + 1).max().unwrap();
    out.tmax = tmax;
    out.ids.clear();
    out.ids.resize(grp.len() * tmax, 0);
    out.readout.clear();
    for (i, &w) in grp.iter().enumerate() {
        let (s, _) = words[w];
        let len = enc_lens[w];
        for k in 0..len {
            out.ids[i * tmax + k] = tokens[s + k] as u32;
        }
        out.ids[i * tmax + len] = w_token as u32;
        out.readout.push((i * tmax + len) as u32);
    }
}

pub struct Hierarchical {
    pub cfg: HierCfg,

    // Tied char table (encoder input + decoder char slots) + grad/moments.
    pub table: DTensor,
    dtable: DTensor,
    m_tbl: DTensor,
    v_tbl: DTensor,

    pub encoder: WordEncoder,

    pub bb_front: Linear,                   // HC → WH
    pub bb_blocks: Vec<Box<dyn BlockLike>>, // WH
    pub bb_back: Linear,                    // WH → HC (context)

    pub dec_blocks: Vec<Box<dyn BlockLike>>, // HC
    pub dec_norm: RmsNorm,                   // HC — the only stage-level norm
    pub dec_head: Linear,                    // HC → vocab

    /// Optimizer step count, persisted with the checkpoint so training resumes.
    pub step_count: usize,

    /// Debug/benchmark switches, read from the environment once at construction
    /// rather than per window — `std::env::var` takes a process-wide lock and
    /// allocates, and a window would otherwise pay for four of them.
    flags: Flags,

    /// Reused host-side index buffers (see [`Scratch`]).
    scratch: Scratch,
}

/// Environment switches, resolved once when the model is built.
#[derive(Clone, Copy, Default)]
struct Flags {
    /// `GPU_PROF=1` — per-phase timings (each mark syncs the stream).
    prof: bool,
    /// `GPU_MEM=1` — device memory in use after each phase.
    mem: bool,
    /// `GPU_NO_GROUP=1` — one rectangle per window instead of length groups.
    no_group: bool,
}

impl Flags {
    fn from_env() -> Self {
        Self {
            prof: std::env::var("GPU_PROF").is_ok(),
            mem: std::env::var("GPU_MEM").is_ok(),
            no_group: std::env::var("GPU_NO_GROUP").is_ok(),
        }
    }
}

impl Hierarchical {
    pub fn new(gpu: &Gpu, cfg: &HierCfg) -> Self {
        let bb_blocks: Vec<Box<dyn BlockLike>> = (0..cfg.bb_blocks)
            .map(|i| {
                if i % 4 == 0 {
                    Box::new(Block::from_cell(
                        gpu,
                        cfg.wh,
                        up_of(cfg.wh),
                        SLstm::new_rand(gpu, cfg.wh, cfg.wh),
                    )) as Box<dyn BlockLike>
                } else {
                    Box::new(Block::from_cell(
                        gpu,
                        cfg.wh,
                        up_of(cfg.wh),
                        MLstm::new_rand(gpu, cfg.wh, cfg.wh, cfg.heads, cfg.dqk),
                    )) as Box<dyn BlockLike>
                }
            })
            .collect();
        let dec_blocks: Vec<Box<dyn BlockLike>> = (0..cfg.dec_blocks)
            .map(|i| {
                if i != 1 {
                    Box::new(Block::from_cell(
                        gpu,
                        cfg.hc,
                        up_of(cfg.hc),
                        SLstm::new_rand(gpu, cfg.hc, cfg.hc),
                    )) as Box<dyn BlockLike>
                } else {
                    Box::new(Block::from_cell(
                        gpu,
                        cfg.hc,
                        up_of(cfg.hc),
                        MLstm::new_rand(gpu, cfg.hc, cfg.hc, 16, cfg.hc / 16),
                    )) as Box<dyn BlockLike>
                }
            })
            .collect();
        Self {
            cfg: *cfg,
            table: DTensor::from_host(gpu, &Tensor::random(&[cfg.vocab, cfg.hc], 0.02)),
            dtable: DTensor::zeros(gpu, &[cfg.vocab, cfg.hc]),
            m_tbl: DTensor::zeros(gpu, &[cfg.vocab, cfg.hc]),
            v_tbl: DTensor::zeros(gpu, &[cfg.vocab, cfg.hc]),
            encoder: WordEncoder::new(gpu, cfg.hc, cfg.enc_blocks),
            bb_front: Linear::new_rand(gpu, cfg.hc, cfg.wh),
            bb_blocks,
            bb_back: Linear::new_rand(gpu, cfg.wh, cfg.hc),
            dec_blocks,
            dec_norm: RmsNorm::new(gpu, cfg.hc),
            dec_head: Linear::new_rand(gpu, cfg.hc, cfg.vocab),
            step_count: 0,
            flags: Flags::from_env(),
            scratch: Scratch::default(),
        }
    }

    /// Forward + backward over one window; accumulates all grads and returns the
    /// mean decode cross-entropy. `tokens` are char ids; `words` are `(start,
    /// end)` char ranges. Word 0 is encode-only; words 1..n are decoded.
    pub fn forward_backward(
        &mut self,
        gpu: &Gpu,
        tokens: &[usize],
        words: &[(usize, usize)],
    ) -> f32 {
        let loss = self.forward_backward_window(gpu, tokens, words);
        // The window's temporaries have dropped by now, so their `cuMemFreeAsync`
        // frees are queued on the stream. CUDA's stream-ordered pool only hands
        // that memory back at a synchronization point — without one it just keeps
        // reserving fresh blocks for every new window shape and grows without
        // bound. One sync per window is noise next to the window's own kernels.
        gpu.stream.synchronize().expect("stream sync");
        loss
    }

    fn forward_backward_window(
        &mut self,
        gpu: &Gpu,
        tokens: &[usize],
        words: &[(usize, usize)],
    ) -> f32 {
        // Phase timing, off unless GPU_PROF is set (each mark syncs the stream).
        // GPU_MEM additionally reports device memory in use after each phase —
        // the window's padded rectangles and the backbone's [heads, T, T]
        // temporaries are what decide whether a config fits.
        let (prof, memp, no_group) = (self.flags.prof, self.flags.mem, self.flags.no_group);
        let mut t0 = std::time::Instant::now();
        let mut mark = |name: &str| {
            if prof || memp {
                gpu.stream.synchronize().expect("sync");
                let mut line = format!("  {name:<22} {:>8.1?}", t0.elapsed());
                if memp {
                    let (free, total) =
                        cudarc::driver::result::mem_get_info().expect("mem_get_info");
                    line.push_str(&format!(
                        "  |  in use {:>6.0} MB",
                        (total - free) as f64 / 1e6
                    ));
                }
                println!("{line}");
                t0 = std::time::Instant::now();
            }
        };
        let n = words.len();
        if n < 2 {
            return 0.0;
        }
        let dw = n - 1;
        let (hc, wh) = (self.cfg.hc, self.cfg.wh);
        let w_token = self.cfg.w_token;

        // The scratch buffers are borrowed for the whole window while `self`'s
        // layers are borrowed mutably, so move them out and hand them back at the
        // end. `Scratch` is all `Vec`s, so the take/restore is a few pointer moves
        // and the capacities survive into the next window.
        let mut sc = std::mem::take(&mut self.scratch);

        // PHASE 1: ENCODER
        // Words are batched as [words, tmax] rectangles, and `tmax` is set by the
        // LONGEST word — so one 16-byte word would pad every 2-byte word out to 17
        // steps (~4.5x wasted rows on Rust source, in both FLOPs and VRAM). Instead
        // group the words by length and run one dense rectangle per group: the
        // padding collapses to within-group slack, and each group is still a clean
        // rectangle, which the mLSTM's per-word [T, T] attention requires.
        sc.enc_lens.clear();
        sc.enc_lens.extend((0..dw).map(|w| words[w].1 - words[w].0));
        group_by_len(&sc.enc_lens, no_group, &mut sc.enc_groups);

        // Build every group's id rectangle once. The encoder BACKWARD re-forwards
        // each group (activation checkpointing) and needs the exact same ids, so
        // computing them here and keeping them costs one pass instead of two.
        let n_groups = sc.enc_groups.len();
        if sc.enc_layout.len() < n_groups {
            sc.enc_layout.resize_with(n_groups, EncGroup::default);
        }
        for g in 0..n_groups {
            enc_group_rows(
                tokens,
                words,
                &sc.enc_groups[g],
                &sc.enc_lens,
                w_token,
                &mut sc.enc_layout[g],
            );
        }

        let mut e_w = DTensor::zeros(gpu, &[dw, hc]);
        for g in 0..n_groups {
            let grp = &sc.enc_groups[g];
            let lay = &sc.enc_layout[g];
            let tmax = lay.tmax;
            let d_ids = ops::upload_ids_u32(gpu, &lay.ids);
            let embedded = ops::embedding_gather_u32(gpu, &self.table, &d_ids, lay.ids.len(), hc);
            let mut h = embedded.reshaped(&[grp.len(), tmax, hc]);
            for blk in self.encoder.blocks.iter_mut() {
                h = blk.forward(gpu, &h);
            }
            let h_flat = h.reshaped(&[grp.len() * tmax, hc]);
            // e_w = each word's [W]-step row, scattered back to its window slot.
            let d_readout = ops::upload_ids_u32(gpu, &lay.readout);
            let e_w_grp =
                ops::embedding_gather_u32(gpu, &h_flat, &d_readout, lay.readout.len(), hc); // [n_g, HC]
            ops::scatter_rows(gpu, &mut e_w, &e_w_grp, grp);
        }
        mark("encoder fwd");

        // PHASE 2: BACKBONE
        let bb_in = self.bb_front.forward(gpu, &e_w); // [dw, WH]
        let mut hb = bb_in.reshaped(&[1, dw, wh]);
        for blk in self.bb_blocks.iter_mut() {
            hb = blk.forward(gpu, &hb);
        }
        let o = self.bb_back.forward(gpu, &hb.reshaped(&[dw, wh])); // [dw, HC]
        mark("backbone fwd");

        // PHASE 3: DECODER (forward + backward, per length group)
        // Word w's decode target is word w+1, so groups are keyed on the length of
        // the DECODED word. Each group runs forward and straight back again: the
        // decoder's backward needs nothing from the backbone's, so a group's
        // activations die before the next group allocates — only one group's worth
        // of decoder rows (and of the [rows, vocab] logits) is ever resident.
        sc.dec_lens.clear();
        sc.dec_lens
            .extend((0..dw).map(|w| words[w + 1].1 - words[w + 1].0));
        group_by_len(&sc.dec_lens, no_group, &mut sc.dec_groups);
        // Every group scales by the WINDOW's valid-row count, so the summed loss and
        // grads match what one big rectangle would have produced.
        let valid_rows: usize = sc.dec_lens.iter().map(|&m| m + 1).sum();
        let inv = 1.0 / (valid_rows.max(1) as f32);

        let mut loss = 0.0;
        let mut d_o = DTensor::zeros(gpu, &[dw, hc]);
        for g in 0..sc.dec_groups.len() {
            let grp = &sc.dec_groups[g];
            let n_g = grp.len();
            let tmax = grp.iter().map(|&w| sc.dec_lens[w] + 1).max().unwrap();
            let rows = n_g * tmax;

            // Refill the per-group index buffers in place (see `Scratch`).
            sc.o_rows.clear(); // dest row of each word's slot 0
            sc.char_rows.clear(); // dest rows of the char slots
            sc.char_ids.clear(); // the char id feeding each of those slots
            sc.targets.clear();
            sc.targets.resize(rows, 0);
            sc.mask.clear();
            sc.mask.resize(rows, 0);
            for (i, &w) in grp.iter().enumerate() {
                let m = sc.dec_lens[w];
                let (s, _) = words[w + 1];
                sc.o_rows.push((i * tmax) as u32);
                for k in 1..=m {
                    sc.char_rows.push((i * tmax + k) as u32);
                    sc.char_ids.push(tokens[s + k - 1] as u32);
                }
                for k in 0..m {
                    sc.targets[i * tmax + k] = tokens[s + k] as u32;
                    sc.mask[i * tmax + k] = 1;
                }
                sc.targets[i * tmax + m] = w_token as u32;
                sc.mask[i * tmax + m] = 1;
            }

            // The group's word indices, narrowed once for the two gathers/scatters
            // that address `o` / `d_o` by window slot.
            sc.grp_ids.clear();
            sc.grp_ids.extend(grp.iter().map(|&w| w as u32));

            let d_grp = ops::upload_ids_u32(gpu, &sc.grp_ids);
            let d_o_rows = ops::upload_ids_u32(gpu, &sc.o_rows);
            let d_char_rows = ops::upload_ids_u32(gpu, &sc.char_rows);
            let d_char_ids = ops::upload_ids_u32(gpu, &sc.char_ids);
            let d_targets = ops::upload_ids_u32(gpu, &sc.targets);
            let d_mask = ops::upload_ids_u32(gpu, &sc.mask);

            // Build the decoder input: zeros, then scatter the context and char rows.
            let o_grp = ops::embedding_gather_u32(gpu, &o, &d_grp, n_g, hc); // this group's contexts
            let mut dec_in = DTensor::zeros(gpu, &[rows, hc]);
            ops::scatter_rows_u32(gpu, &mut dec_in, &o_grp, &d_o_rows);
            let char_vecs =
                ops::embedding_gather_u32(gpu, &self.table, &d_char_ids, sc.char_ids.len(), hc);
            ops::scatter_rows_u32(gpu, &mut dec_in, &char_vecs, &d_char_rows);

            let mut hd = dec_in.reshaped(&[n_g, tmax, hc]);
            for blk in self.dec_blocks.iter_mut() {
                hd = blk.forward(gpu, &hd);
            }
            let hdn = self.dec_norm.forward(gpu, &hd.reshaped(&[rows, hc]));
            let logits = self.dec_head.forward(gpu, &hdn);
            let capped = ops::softcap_forward(gpu, &logits, self.cfg.cap);

            let (l, d_capped) =
                ops::masked_softmax_cross_entropy_u32(gpu, &capped, &d_targets, &d_mask, inv);
            loss += l;

            let d_logits = ops::softcap_backward(gpu, &d_capped, &capped, self.cfg.cap);
            let d_hdn = self.dec_head.backward(gpu, &d_logits);
            let d_hd_flat = self.dec_norm.backward(gpu, &d_hdn);
            let mut d_hd = d_hd_flat.reshaped(&[n_g, tmax, hc]);
            for blk in self.dec_blocks.iter_mut().rev() {
                d_hd = blk.backward(gpu, &d_hd);
            }
            let d_dec_in = d_hd.reshaped(&[rows, hc]);
            // Slot 0 rows → d_o; char-slot rows → tied table (gather then scatter-add).
            let d_o_grp = ops::embedding_gather_u32(gpu, &d_dec_in, &d_o_rows, n_g, hc); // [n_g, HC]
            ops::scatter_rows_u32(gpu, &mut d_o, &d_o_grp, &d_grp);
            let n_chars = sc.char_rows.len();
            let d_char = ops::embedding_gather_u32(gpu, &d_dec_in, &d_char_rows, n_chars, hc);
            ops::embedding_scatter_add_u32(
                gpu,
                &mut self.dtable,
                &d_char_ids,
                n_chars,
                &d_char,
                hc,
            );
        }
        mark("decoder fwd + bwd");

        // Backbone backward.
        let d_bb_out = self.bb_back.backward(gpu, &d_o); // [dw, WH]
        let mut d_hb = d_bb_out.reshaped(&[1, dw, wh]);
        for blk in self.bb_blocks.iter_mut().rev() {
            d_hb = blk.backward(gpu, &d_hb);
        }
        let d_e_w = self.bb_front.backward(gpu, &d_hb.reshaped(&[dw, wh])); // [dw, HC]
        mark("backbone bwd");

        // ENCODER BACKWARD (per group, re-forwarded)
        // Each encoder group's forward cache was overwritten by the group after it
        // (and by the OTHER direction), so re-run that group's fwd+bwd stacks and
        // the combine forward to refill their caches, then backward immediately.
        // Forward is deterministic, so this reproduces the exact activations — it
        // is activation checkpointing, and it keeps just one group resident. The
        // cost is one extra encoder forward, over the SMALL grouped rectangles.
        for g in 0..n_groups {
            let grp = &sc.enc_groups[g];
            let lay = &sc.enc_layout[g];
            let tmax = lay.tmax;
            let d_ids = ops::upload_ids_u32(gpu, &lay.ids);
            let embedded = ops::embedding_gather_u32(gpu, &self.table, &d_ids, lay.ids.len(), hc);
            let mut h = embedded.reshaped(&[grp.len(), tmax, hc]);
            for blk in self.encoder.blocks.iter_mut() {
                h = blk.forward(gpu, &h);
            }
            drop(h);

            // Scatter this group's d_e_w onto its [W]-step rows, rest zero.
            sc.grp_ids.clear();
            sc.grp_ids.extend(grp.iter().map(|&w| w as u32));
            let d_grp = ops::upload_ids_u32(gpu, &sc.grp_ids);
            let d_e_w_grp = ops::embedding_gather_u32(gpu, &d_e_w, &d_grp, grp.len(), hc); // [n_g, HC]
            let mut d_h = DTensor::zeros(gpu, &[grp.len() * tmax, hc]);
            let d_readout = ops::upload_ids_u32(gpu, &lay.readout);
            ops::scatter_rows_u32(gpu, &mut d_h, &d_e_w_grp, &d_readout);
            let mut d_h = d_h.reshaped(&[grp.len(), tmax, hc]);
            for blk in self.encoder.blocks.iter_mut().rev() {
                d_h = blk.backward(gpu, &d_h);
            }
            let d_embedded = d_h.reshaped(&[grp.len() * tmax, hc]);
            ops::embedding_scatter_add_u32(
                gpu,
                &mut self.dtable,
                &d_ids,
                lay.ids.len(),
                &d_embedded,
                hc,
            );
        }
        mark("encoder bwd");

        self.scratch = sc;
        loss
    }

    /// AdamW across every stage. Tied table and the logit head are undecayed;
    /// interior projections decay (matching the project's optimizer convention).
    pub fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        ops::adamw(
            gpu,
            &mut self.table,
            &self.dtable,
            &mut self.m_tbl,
            &mut self.v_tbl,
            cfg,
            false,
        );
        self.dtable.zero_(gpu);
        for b in self.encoder.blocks.iter_mut() {
            b.step(gpu, cfg);
        }
        self.bb_front.step(gpu, cfg);
        for b in self.bb_blocks.iter_mut() {
            b.step(gpu, cfg);
        }
        self.bb_back.step(gpu, cfg);
        for b in self.dec_blocks.iter_mut() {
            b.step(gpu, cfg);
        }
        self.dec_norm.step(gpu, cfg);
        // Logit head: no weight decay and no bias (bias stays at its zero init) so
        // it matches `nn::linear_no_bias` and exports faithfully to the HIER head.
        self.dec_head.step_w_only(gpu, cfg, false);
        self.step_count += 1;
    }

    // checkpointing

    /// Export the three stages into CPU `nn` `Sequential`s laid out exactly like
    /// `model::build_hierarchical_model`, so the result serializes to the same
    /// `NNM1` container a CPU-trained model would (readable by `hp` / `hs`).
    fn to_sequentials(&mut self, gpu: &Gpu) -> (Sequential, Sequential, Sequential) {
        let (vocab, hc, wh) = (self.cfg.vocab, self.cfg.hc, self.cfg.wh);

        // Encoder: Embedding(tied table) → blocks.
        let mut enc: Vec<Box<dyn NnLayer>> = Vec::new();
        enc.push(Box::new(EmbeddingLayer::from_loaded(
            vocab,
            hc,
            dt_matrix(gpu, &self.table),
        )));
        for b in self.encoder.blocks.iter_mut() {
            enc.push(b.to_nn_layer(gpu));
        }

        // Backbone: Linear(HC→WH) → blocks → Linear(WH→HC).
        let mut wm: Vec<Box<dyn NnLayer>> = Vec::new();
        wm.push(Box::new(LinearLayer::from_loaded(
            hc,
            wh,
            dt_matrix(gpu, &self.bb_front.w),
            dt_vec(gpu, &self.bb_front.b),
        )));
        for b in self.bb_blocks.iter_mut() {
            wm.push(b.to_nn_layer(gpu));
        }
        wm.push(Box::new(LinearLayer::from_loaded(
            wh,
            hc,
            dt_matrix(gpu, &self.bb_back.w),
            dt_vec(gpu, &self.bb_back.b),
        )));

        // Decoder: sLSTM blocks → RMSNorm → LinearNoBias(head) → SoftCap.
        let mut dec: Vec<Box<dyn NnLayer>> = Vec::new();
        for b in self.dec_blocks.iter_mut() {
            dec.push(b.to_nn_layer(gpu));
        }
        dec.push(Box::new(RMSNorm::from_loaded(
            hc,
            dt_vec(gpu, &self.dec_norm.gamma),
        )));
        dec.push(Box::new(LinearNBLayer::from_loaded(
            hc,
            vocab,
            dt_matrix(gpu, &self.dec_head.w),
        )));
        dec.push(Box::new(SoftCapLayer::new(vocab, self.cfg.cap)));

        (
            Sequential::from_layers(enc),
            Sequential::from_layers(wm),
            Sequential::from_layers(dec),
        )
    }

    /// Write an `NNM1` hierarchical checkpoint — the same container the CPU
    /// hierarchical model uses, so `hp` / `hs` can open a GPU-trained model
    /// directly. Weights only (Adam moments are not persisted, so a resumed run
    /// restarts them).
    pub fn save(&mut self, gpu: &Gpu, path: &str, _boundary_token_ids: &[u16]) -> io::Result<()> {
        let (encoder, word_model, char2_model) = self.to_sequentials(gpu);

        // context_size == the backbone's output width == HC (the decoder's input),
        // exactly what `Hierarchical::new` recomputes and debug-asserts on load.
        let context_size = word_model.output_size;

        Writer::new(
            ModelKind::Hierarchical,
            Meta {
                vocab_size: self.cfg.vocab as u32,
                context_size: context_size as u32,
                step: self.step_count as u64,
            },
        )
        .section("encoder", &encoder.layers)
        .section("word_model", &word_model.layers)
        .section("char2_model", &char2_model.layers)
        .save(path)
    }

    /// Load a `HIER` checkpoint (written by this model or a CPU run), rebuilding
    /// the device model. `w_token` is supplied by the caller (from the tokenizer)
    /// because the HIER format does not store it.
    pub fn load(gpu: &Gpu, path: &str, w_token: usize) -> io::Result<Self> {
        let stacks = crate::hierarchical::Hierarchical::load_stacks(path)?;

        let err = |m: String| io::Error::new(io::ErrorKind::InvalidData, m);
        let to_block =
            |gpu: &Gpu, l: &Box<dyn crate::nn_layer::NnLayer>| -> io::Result<Box<dyn BlockLike>> {
                if let Some(s) = l.as_any().downcast_ref::<SLSTMBlock>() {
                    Ok(Box::new(Block::<SLstm>::from_nn_block(gpu, s)))
                } else if let Some(m) = l.as_any().downcast_ref::<MLSTMBlock>() {
                    Ok(Box::new(Block::<MLstm>::from_nn_block(gpu, m)))
                } else {
                    Err(err("expected an sLSTM/mLSTM block in the checkpoint".into()))
                }
            };

        // Encoder: Embedding + blocks
        let enc = &stacks.encoder_chars.layers;
        let emb = enc[0]
            .as_any()
            .downcast_ref::<EmbeddingLayer>()
            .ok_or_else(|| err("encoder must start with an Embedding".into()))?;
        let vocab = emb.input_size();
        let hc = emb.output_size();
        let table = DTensor::from_host(gpu, &tensor_from_matrix(&emb.weights));
        let enc_blocks: Vec<Box<dyn BlockLike>> = enc[1..]
            .iter()
            .map(|l| to_block(gpu, l))
            .collect::<io::Result<_>>()?;

        // Backbone: Linear + blocks + Linear
        let wm = &stacks.word_model.layers;
        let front = wm[0]
            .as_any()
            .downcast_ref::<LinearLayer>()
            .ok_or_else(|| err("backbone must start with a Linear".into()))?;
        let wh = front.output_size();
        let bb_front = linear_layer_to_gpu(gpu, front);
        let back = wm[wm.len() - 1]
            .as_any()
            .downcast_ref::<LinearLayer>()
            .ok_or_else(|| err("backbone must end with a Linear".into()))?;
        let bb_back = linear_layer_to_gpu(gpu, back);
        let bb_blocks: Vec<Box<dyn BlockLike>> = wm[1..wm.len() - 1]
            .iter()
            .map(|l| to_block(gpu, l))
            .collect::<io::Result<_>>()?;

        // heads/dqk read off the first mLSTM block (all mLSTM blocks share them).
        let (heads, dqk) = wm[1..wm.len() - 1]
            .iter()
            .find_map(|l| l.as_any().downcast_ref::<MLSTMBlock>())
            .map(|m| (m.cell.num_heads, m.cell.dqk))
            .unwrap_or((8, wh / 8));

        // Decoder: sLSTM blocks + RMSNorm + LinearNoBias + SoftCap
        let dl = &stacks.char2_model.layers;
        let norm_idx = dl
            .iter()
            .position(|l| l.as_any().downcast_ref::<RMSNorm>().is_some())
            .ok_or_else(|| err("decoder is missing its RMSNorm".into()))?;
        let dec_blocks: Vec<Box<dyn BlockLike>> = dl[..norm_idx]
            .iter()
            .map(|l| to_block(gpu, l))
            .collect::<io::Result<_>>()?;
        let rms = dl[norm_idx].as_any().downcast_ref::<RMSNorm>().unwrap();
        let dec_norm = super::rms_norm::RmsNorm::from_parts(gpu, &tensor_from_slice(&rms.gamma));
        let head = dl
            .iter()
            .find_map(|l| l.as_any().downcast_ref::<LinearNBLayer>())
            .ok_or_else(|| err("decoder is missing its LinearNoBias head".into()))?;
        let dec_head = super::linear::Linear::from_parts(
            gpu,
            &tensor_from_matrix(&head.weights),
            &crate::tensor::Tensor::zeros(&[vocab]),
        );
        let cap = dl
            .iter()
            .find_map(|l| l.as_any().downcast_ref::<SoftCapLayer>())
            .map(|s| s.cap)
            .unwrap_or(crate::config::LOGIT_SOFTCAP);

        let cfg = HierCfg {
            vocab,
            hc,
            wh,
            enc_blocks: enc_blocks.len(),
            bb_blocks: bb_blocks.len(),
            dec_blocks: dec_blocks.len(),
            heads,
            dqk,
            w_token,
            cap,
        };

        // Build a fresh model (for the zeroed grads/moments), then swap in the
        // loaded weight-bearing parts.
        let mut model = Hierarchical::new(gpu, &cfg);
        model.step_count = stacks.step;
        model.table = table;
        model.encoder.blocks = enc_blocks;
        model.bb_front = bb_front;
        model.bb_blocks = bb_blocks;
        model.bb_back = bb_back;
        model.dec_blocks = dec_blocks;
        model.dec_norm = dec_norm;
        model.dec_head = dec_head;
        Ok(model)
    }
}

/// Upload an `nn::LinearLayer` (weights + bias) to a device `Linear`.
fn linear_layer_to_gpu(gpu: &Gpu, l: &LinearLayer) -> Linear {
    Linear::from_parts(
        gpu,
        &super::tensor_from_matrix(&l.weights),
        &super::tensor_from_slice(&l.biases),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The GPU hierarchical stack must actually learn: memorize one tiny window,
    /// driving the decode loss down. Exercises the full wiring — tied char table
    /// (encoder input + decoder char slots), backbone context injection at the
    /// decoder's slot 0, the [W]-step readout, masked CE and AdamW across stages.
    /// Then round-trip a checkpoint and confirm the loss is unchanged.
    #[test]
    fn hierarchical_memorizes_and_checkpoints() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let cfg = HierCfg {
            vocab: 9,
            hc: 16,
            wh: 24,
            enc_blocks: 1,
            bb_blocks: 2,
            dec_blocks: 1,
            heads: 2,
            dqk: 8,
            w_token: 8,
            cap: 30.0,
        };
        let mut model = Hierarchical::new(&gpu, &cfg);

        let tokens = vec![1usize, 2, 3, 4, 5, 6, 7, 1, 2, 3];
        let words = vec![(0, 3), (3, 5), (5, 8), (8, 10)];

        let mut opt = AdamCfg::new(5e-3, 0.0);
        let first = model.forward_backward(&gpu, &tokens, &words);
        for _ in 0..250 {
            let _ = model.forward_backward(&gpu, &tokens, &words);
            opt.t += 1;
            model.step(&gpu, &opt);
        }
        let last = model.forward_backward(&gpu, &tokens, &words);
        assert!(
            last < first * 0.4,
            "decode loss did not fall: {first} -> {last}"
        );

        // Checkpoint round-trip: reloading must reproduce the exact same loss.
        // Saves in the CPU HIER format; `w_token` is supplied on load.
        let path = std::env::temp_dir().join("gpu_hier_test.hier");
        let path = path.to_str().unwrap();
        model.save(&gpu, path, &[cfg.w_token as u16]).expect("save");
        let mut back = Hierarchical::load(&gpu, path, cfg.w_token).expect("load");
        assert_eq!(back.cfg, cfg, "config did not survive the round-trip");
        assert_eq!(back.step_count, model.step_count, "step count lost");
        let reloaded = back.forward_backward(&gpu, &tokens, &words);
        assert!(
            (reloaded - last).abs() < 1e-4,
            "reloaded model gives a different loss: {last} -> {reloaded}"
        );
        let _ = std::fs::remove_file(path);
    }

    /// Splitting a window into length groups is a pure batching change: it must
    /// give the same loss AND the same gradients as one padded rectangle. Words of
    /// four different lengths here, so the grouped path really does fire several
    /// rectangles (1, 2, 4 and 8-step groups) instead of one.
    ///
    /// The two runs are compared through a full optimizer step: identical weights
    /// afterwards means the reduced grads agreed, not just the loss.
    #[test]
    fn grouping_matches_single_rectangle() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let cfg = HierCfg {
            vocab: 12,
            hc: 16,
            wh: 24,
            enc_blocks: 2,
            bb_blocks: 2,
            dec_blocks: 2,
            heads: 2,
            dqk: 8,
            w_token: 11,
            cap: 30.0,
        };
        // Word lengths 1, 3, 2, 6, 4 — four distinct power-of-two buckets.
        let tokens: Vec<usize> = (0..16).map(|i| 1 + i % 9).collect();
        let words = vec![(0, 1), (1, 4), (4, 6), (6, 12), (12, 16)];

        let run = |grouped: bool| -> (f32, Vec<f32>) {
            // SAFETY-adjacent: tests in this binary run in threads. The flag is
            // read once when the model below is constructed, so it must be set
            // BEFORE `new`/`load` — not merely before `forward_backward`.
            if grouped {
                unsafe { std::env::remove_var("GPU_NO_GROUP") };
            } else {
                unsafe { std::env::set_var("GPU_NO_GROUP", "1") };
            }
            let mut model = Hierarchical::new(&gpu, &cfg);
            // Same starting weights for both runs.
            let seed = std::env::temp_dir().join("gpu_group_seed.hier");
            let seed = seed.to_str().unwrap();
            if grouped {
                model.save(&gpu, seed, &[]).expect("save seed");
            } else {
                model = Hierarchical::load(&gpu, seed, cfg.w_token).expect("load seed");
            }
            let loss = model.forward_backward(&gpu, &tokens, &words);
            let mut opt = AdamCfg::new(1e-2, 0.0);
            opt.t += 1;
            model.step(&gpu, &opt); // folds every stage's grads into the weights
            let w: Vec<f32> = model.table.to_host(&gpu).data.to_vec();
            (loss, w)
        };

        let (loss_grouped, w_grouped) = run(true);
        let (loss_single, w_single) = run(false);
        unsafe { std::env::remove_var("GPU_NO_GROUP") };

        assert!(
            (loss_grouped - loss_single).abs() < 1e-5,
            "grouped loss {loss_grouped} != single-rectangle loss {loss_single}"
        );
        for (i, (a, b)) in w_grouped.iter().zip(&w_single).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "post-step weight {i} diverged: grouped {a} vs single {b}"
            );
        }
    }
}
