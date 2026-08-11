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
use std::range::Range;

use super::block::{Block, BlockLike};
use super::{DTensor, Gpu, linear::Linear, mlstm::MLstm, ops, rms_norm::RmsNorm, slstm::SLstm};
use crate::format::{Meta, ModelKind, Seen, Writer};
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
    pub bb_blocks: usize, // sLSTM every 4th block, mLSTM otherwise (see `model.rs`)
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
fn group_by_len(lens: &[usize], no_group: bool, cap: usize, out: &mut Vec<Vec<usize>>) {
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
    let used = split_oversized_groups(lens, out, used, cap);
    out.truncate(used);
}

/// Cap each group at [`config::GROUP_MAX_ROWS`] rows, splitting oversized ones into
/// same-shape pieces. Returns the new group count.
///
/// A bucket holds every word of its length in the window, so it grows with the window
/// and the whole stage's buffers size to the largest one. The groups already run one
/// after another with independent activations and accumulating grads, so cutting one in
/// two is a pure batching choice — no arithmetic changes, which is what
/// `grouping_matches_single_rectangle` and `group_cap_matches_uncapped` pin.
///
/// Rows are `words × tmax` and `tmax` is fixed by the bucket, so the cap converts to a
/// word count per piece. `GROUP_MAX_ROWS == 0` disables the split.
fn split_oversized_groups(
    lens: &[usize],
    out: &mut Vec<Vec<usize>>,
    used: usize,
    cap: usize,
) -> usize {
    if cap == 0 {
        return used;
    }
    let mut n = used;
    for g in 0..used {
        // `tmax` is the bucket's actual longest word, matching `enc_group_rows`.
        let tmax = out[g].iter().map(|&w| lens[w]).max().unwrap_or(0).max(1);
        // +1 for the encoder's virtual `[W]` step, so the bound matches the rectangle
        // the stage actually runs.
        let per_piece = (cap / (tmax + 1)).max(1);
        if out[g].len() <= per_piece {
            continue;
        }
        // Move the tail into fresh groups, keeping the first piece in place so the
        // ascending-length group order (which fixes grad scatter order) is preserved.
        let tail: Vec<usize> = out[g].split_off(per_piece);
        for piece in tail.chunks(per_piece) {
            if n == out.len() {
                out.push(Vec::new());
            }
            out[n].clear();
            out[n].extend_from_slice(piece);
            n += 1;
        }
    }
    n
}

/// Whether the backbone sweep is chunked over the word axis.
///
/// The backbone holds one row per word in every block from that block's forward to its
/// backward, so an unchunked sweep is O(words) resident per block and device memory
/// scales with the window. Chunk-major makes that O(BACKBONE_CHUNK): each chunk passes
/// through all blocks carrying the cells' recurrent state, and backward unwinds them
/// right to left carrying the BPTT state the other way.
///
/// Two things had to be true before this could be on, and both are pinned by tests:
///
///   * **Per-chunk activation storage.** Every cache that a later chunk's forward would
///     overwrite is now one-per-chunk: the FFN's five buffers, both pre-norms, the
///     cell's own cache and its head norm, and one `HostPark` generation per chunk
///     under offload. `backbone_chunked_matches_unchunked` pins the gradients.
///   * **The mLSTM's backward carry.** `mlstm_bw_parallel` forced the last chunk's
///     incoming state gradient to zero, which is right for a whole sequence but wrong
///     under CARRY, where that chunk feeds the one to its right — a wrong gradient with
///     a right-looking loss. `mlstm_chunked_backward_matches_whole` pins it.
const BACKBONE_CHUNKED_BACKWARD: bool = true;

/// Backbone chunk length for a `words`-word window, or `words` (one chunk, the
/// unchunked path) when chunking is off or the window already fits in one.
///
/// `GPU_BACKBONE_CHUNK` overrides [`config::BACKBONE_CHUNK`] for A/B measurement.
fn backbone_chunk(words: usize) -> usize {
    // Read per call rather than cached: this runs once per window, not per chunk, and
    // a process-wide cache would make an A/B (or `backbone_chunked_matches_unchunked`)
    // silently compare the first value against itself.
    let c = std::env::var("GPU_BACKBONE_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(crate::config::BACKBONE_CHUNK);
    if c == 0 { words } else { c.min(words).max(1) }
}

/// `[(start, len), ...]` covering `n` rows in near-equal pieces of at most `chunk`.
///
/// A single full-length span when `chunk >= n`, which is exactly the pre-chunking
/// path — so the chunked and unchunked code paths are the same code, not two.
///
/// The pieces are **balanced** rather than "full chunks plus a short tail": every
/// buffer in the stage is reused by size class, so a ragged last chunk adds a second
/// set of shapes for the pool to hold. Balancing keeps the spread inside one class.
fn chunk_spans(n: usize, chunk: usize) -> Vec<(usize, usize)> {
    if chunk >= n {
        return vec![(0, n)];
    }
    let pieces = n.div_ceil(chunk);
    let base = n / pieces;
    let extra = n % pieces; // the first `extra` pieces take one more row
    let mut out = Vec::with_capacity(pieces);
    let mut c0 = 0;
    for i in 0..pieces {
        let len = base + usize::from(i < extra);
        out.push((c0, len));
        c0 += len;
    }
    debug_assert_eq!(c0, n, "chunk_spans: pieces do not tile the rows");
    out
}

/// Rows `[c0, c0+len)` of a `[N, width]` tensor, as a fresh tensor.
///
/// A copy, not a view: the chunk is handed to blocks that write through their own
/// buffers, and a view into the whole-window tensor would alias what the next chunk
/// still needs.
fn slice_rows(gpu: &Gpu, src: &DTensor, c0: usize, len: usize, width: usize) -> DTensor {
    let mut out = DTensor::uninit(gpu, &[len, width]);
    gpu.stream
        .memcpy_dtod(
            &src.buf.slice(c0 * width..(c0 + len) * width),
            &mut out.buf.slice_mut(..len * width),
        )
        .expect("slice_rows");
    out
}

/// Concatenate row-blocks back into one `[rows, width]` tensor, in order.
fn concat_rows(gpu: &Gpu, parts: &[DTensor], rows: usize, width: usize) -> DTensor {
    let mut out = DTensor::uninit(gpu, &[rows, width]);
    let mut off = 0;
    for p in parts {
        let n = p.len();
        gpu.stream
            .memcpy_dtod(&p.buf.slice(..n), &mut out.buf.slice_mut(off..off + n))
            .expect("concat_rows");
        off += n;
    }
    debug_assert_eq!(off, rows * width, "concat_rows: parts do not tile the output");
    out
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
    words: &[Range<usize>],
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
        let s = words[w].start;
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

    /// Backbone chunk length for this model, overriding [`config::BACKBONE_CHUNK`].
    ///
    /// Per-instance rather than an env var: the test suite runs threads in one process,
    /// so a process-global override races with whatever else is mid-window.
    bb_chunk: Option<usize>,
    /// Encoder/decoder group row cap for this model, overriding
    /// [`config::GROUP_MAX_ROWS`]. `Some(0)` disables the cap. Per-instance for the
    /// same reason as `bb_chunk`: the suite runs threads in one process.
    group_cap: Option<usize>,
    pub bb_front: Linear,                   // HC → WH
    pub bb_blocks: Vec<Box<dyn BlockLike>>, // WH
    pub bb_back: Linear,                    // WH → HC (context)

    pub dec_blocks: Vec<Box<dyn BlockLike>>, // HC
    pub dec_norm: RmsNorm,                   // HC — the only stage-level norm
    pub dec_head: Linear,                    // HC → vocab

    /// Optimizer step count, persisted with the checkpoint so training resumes.
    pub step_count: usize,
    /// Cumulative chars/words this model has trained on (see `format::Seen`).
    /// Advanced by the training loops, persisted in the checkpoint.
    pub seen: Seen,

    /// Debug/benchmark switches, read from the environment once at construction
    /// rather than per window — `std::env::var` takes a process-wide lock and
    /// allocates, and a window would otherwise pay for four of them.
    flags: Flags,

    /// Reused host-side index buffers (see [`Scratch`]).
    scratch: Scratch,
    /// Staging for the per-group index lists, so each group's ids go up in one
    /// pinned async transfer rather than one blocking transfer per list.
    ids: ops::IdBatch,
    /// Device accumulator for the decode loss, summed across the window's groups and
    /// read back once at the end of the step.
    ///
    /// Reading each group's loss where it is produced means a blocking `clone_dtoh`
    /// inside the decode loop — it drains the stream and page-locks a staging buffer,
    /// per group, for a number only used in logging.
    loss_acc: Option<cudarc::driver::CudaSlice<f32>>,
    /// Mean NLL per decoded word of the last window — the same quantity the CPU
    /// path logs as `word_loss`. Derived from the window's row loss on the host,
    /// so it costs no extra device work.
    last_word_loss: f32,
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
    /// Park the backbone's saved activations in host memory between forward and
    /// backward, trading (overlapped) PCIe for device memory. **On by default**;
    /// `GPU_NO_OFFLOAD=1` forces the all-resident path.
    ///
    /// Default-on because it is bit-exact against the resident path (it moves bytes
    /// and reorders no arithmetic — see `Block::offload_matches_resident_exactly`),
    /// costs ~2% of a step, and frees ~420 MB at the backbone's config. A knob whose
    /// off-state is strictly worse is just a way to forget to turn it on.
    /// See `Hierarchical::enable_backbone_offload`.
    offload: bool,
}

impl Flags {
    fn from_env() -> Self {
        Self {
            prof: std::env::var("GPU_PROF").is_ok(),
            mem: std::env::var("GPU_MEM").is_ok(),
            no_group: std::env::var("GPU_NO_GROUP").is_ok(),
            offload: std::env::var("GPU_NO_OFFLOAD").is_err(),
        }
    }
}

impl Hierarchical {
    pub fn new(gpu: &Gpu, cfg: &HierCfg) -> Self {
        let bb_blocks: Vec<Box<dyn BlockLike>> = (0..cfg.bb_blocks)
            .map(|i| {
                if i.is_multiple_of(4) {
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
        let mut model = Self {
            cfg: *cfg,
            table: DTensor::from_host(gpu, &Tensor::random(&[cfg.vocab, cfg.hc], 0.02)),
            dtable: DTensor::zeros(gpu, &[cfg.vocab, cfg.hc]),
            m_tbl: DTensor::zeros(gpu, &[cfg.vocab, cfg.hc]),
            v_tbl: DTensor::zeros(gpu, &[cfg.vocab, cfg.hc]),
            encoder: WordEncoder::new(gpu, cfg.hc, cfg.enc_blocks),
            bb_chunk: None,
            group_cap: None,
            bb_front: Linear::new_rand(gpu, cfg.hc, cfg.wh),
            bb_blocks,
            bb_back: Linear::new_rand(gpu, cfg.wh, cfg.hc),
            dec_blocks,
            dec_norm: RmsNorm::new(gpu, cfg.hc),
            dec_head: {
                // fp32: the head's output is exponentiated by the softmax/CE, and a
                // bf16 wobble on a logit turns into a multiplicative error on a
                // probability. One GEMM per decoded word against the backbone's
                // many, so keeping it wide costs almost nothing.
                let mut h = Linear::new_rand(gpu, cfg.hc, cfg.vocab);
                h.set_fp32();
                h
            },
            step_count: 0,
            seen: Seen::default(),
            flags: Flags::from_env(),
            scratch: Scratch::default(),
            ids: ops::IdBatch::new(),
            loss_acc: None,
            last_word_loss: 0.0,
        };
        model.enable_backbone_offload(gpu);
        model
    }

    /// Park the backbone blocks' saved activations on the host (unless
    /// `GPU_NO_OFFLOAD=1`).
    ///
    /// **Backbone only.** It is the one stack that runs its whole forward before any
    /// of its backward, which is what gives each block's device→host copy a full block
    /// of compute to hide behind and what makes releasing the source buffers one block
    /// later safe. The encoder re-forwards per group instead of saving, and the
    /// decoder runs forward-then-backward within each length group — only two blocks
    /// apart, so parking there frees buffers that are still being read (observed as
    /// `CUDA_ERROR_ILLEGAL_ADDRESS`). See `Block::enable_offload`.
    fn enable_backbone_offload(&mut self, gpu: &Gpu) {
        if !self.flags.offload {
            return;
        }
        // One shared slot across the backbone: block i+1's eviction releases block i's
        // buffers, bounding in-flight device memory at a single block's worth.
        let in_flight = crate::gpu::offload::InFlight::shared();
        for blk in self.bb_blocks.iter_mut() {
            blk.enable_offload(gpu, in_flight.clone());
        }
        if self.flags.mem {
            println!(
                "  offload: parking activations for {} backbone blocks",
                self.bb_blocks.len()
            );
        }
    }

    /// Forward + backward over one window; accumulates all grads and returns the
    /// mean decode cross-entropy. `tokens` are char ids; `words` are `(start,
    /// end)` char ranges. Word 0 is encode-only; words 1..n are decoded.
    /// Mean NLL per decoded word of the last window (`word_loss` in the logs).
    pub fn last_word_loss(&self) -> f32 {
        self.last_word_loss
    }

    pub fn forward_backward(&mut self, gpu: &Gpu, tokens: &[usize], words: &[Range<usize>]) -> f32 {
        let loss = self.forward_backward_window(gpu, tokens, words, None);
        gpu.stream.synchronize().expect("stream sync");
        loss
    }

    /// Like [`forward_backward`](Self::forward_backward) but for SFT: `word_loss`
    /// has one flag per DECODED word (word `w` decodes `words[w+1]`, so
    /// `word_loss.len() == words.len() - 1`). Only words flagged `true`
    /// contribute to the loss and gradient — every word is still encoded and
    /// decoded (the backbone must read the prompt), but a masked word's decoder
    /// slots get CE mask 0, so no gradient flows from the prompt tokens. The
    /// returned loss is the mean CE over the *unmasked* response tokens.
    pub fn forward_backward_masked(
        &mut self,
        gpu: &Gpu,
        tokens: &[usize],
        words: &[Range<usize>],
        word_loss: &[bool],
    ) -> f32 {
        let loss = self.forward_backward_window(gpu, tokens, words, Some(word_loss));
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
        words: &[Range<usize>],
        word_loss: Option<&[bool]>,
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
                    // Both numbers, because they answer different questions and the
                    // difference has repeatedly been mistaken for a leak. `driver` is
                    // what `mem_get_info` reports — every block the CUDA async
                    // allocator holds, including what it is merely CACHING for reuse.
                    // `live` is what is actually allocated, and it is what decides
                    // whether the next allocation OOMs. A climbing `driver` with a flat
                    // `live` is the allocator holding cache, not the model growing.
                    line.push_str(&format!(
                        "  |  driver {:>6.0} MB  live {:>6.0} MB",
                        (total - free) as f64 / 1e6,
                        super::pool_used_mb().unwrap_or(f64::NAN)
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

        // The window's actual shape, which is what every buffer below is sized from.
        // A synthetic probe picks its own word-length spread and can easily miss the
        // one real text produces — and since the encoder/decoder run one rectangle per
        // length BUCKET, the histogram matters as much as the totals.
        if memp {
            let mut hist = [0usize; crate::config::MAX_WORD_BYTES + 2];
            for w in words.iter() {
                hist[(w.end - w.start).min(hist.len() - 1)] += 1;
            }
            let widest = hist.iter().rposition(|&c| c > 0).unwrap_or(0);
            let buckets = hist.iter().filter(|&&c| c > 0).count();
            // Pool free-list shape alongside the shape of the window that produced it:
            // `mem_get_info` cannot separate live memory from allocator cache, and a
            // climbing size count is the signal that the free list is accumulating one
            // entry per distinct shape rather than converging. See `buf::size_class`.
            let ps = self.pool_shapes();
            let sizes: usize = ps.iter().map(|(_, s, _)| s).sum();
            let bufs: usize = ps.iter().map(|(_, _, b)| b).sum();
            println!(
                "  window: {n} words, {} tokens, longest {widest}, {buckets} buckets  \
                 | pool {sizes} sizes / {bufs} bufs",
                tokens.len()
            );
        }

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
        sc.enc_lens
            .extend((0..dw).map(|w| words[w].end - words[w].start));
        let group_cap = self.group_cap.unwrap_or(crate::config::GROUP_MAX_ROWS);
        group_by_len(&sc.enc_lens, no_group, group_cap, &mut sc.enc_groups);

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

        // The row count every encoder group will be run at.
        //
        // Groups run one after another and a group's activations die before the next
        // starts, so only one group is ever resident — but each group has its own
        // `[n_g, tmax]` shape, and every `Buf` and `Linear::x` inside the stack resizes
        // to whatever it is handed. Feeding the stack a different shape per group
        // therefore reallocates the whole encoder's activation set once per group, and
        // (because reuse is by capacity, within `RETAIN_SLACK`) leaves the discarded
        // ones resident: measured at 1259 MB for a stage whose largest single rectangle
        // is ~25 MB.
        //
        // A block takes `[B, T, H]` and runs its recurrence along `T`, so `T` (= tmax)
        // is fixed by the group's bucket and only `B` (the word count) can be padded.
        // Padding every group to the same `B` therefore does NOT give every group the
        // same row count — `B * tmax` still differs across buckets — so this is a
        // bound on the largest group, used to size the trim, not a uniform shape.
        let enc_rows = (0..n_groups)
            .map(|g| sc.enc_groups[g].len() * sc.enc_layout[g].tmax)
            .max()
            .unwrap_or(0);

        let mut e_w = DTensor::zeros(gpu, &[dw, hc]);
        for g in 0..n_groups {
            let grp = &sc.enc_groups[g];
            let lay = &sc.enc_layout[g];
            let tmax = lay.tmax;
            // Both id lists in one pinned async upload instead of two blocking ones.
            self.ids.upload(gpu, &[&lay.ids, &lay.readout]);
            let embedded =
                ops::embedding_gather_u32(gpu, &self.table, &self.ids.get(0), lay.ids.len(), hc);
            let mut h = embedded.reshaped(&[grp.len(), tmax, hc]);
            // Blocks are H-in == H-out, so one spare buffer ping-pongs the stack.
            let mut next = DTensor::uninit(gpu, &[grp.len(), tmax, hc]);
            for blk in self.encoder.blocks.iter_mut() {
                blk.forward(gpu, &h, &mut next);
                std::mem::swap(&mut h, &mut next);
            }
            let h_flat = h.reshaped(&[grp.len() * tmax, hc]);
            // e_w = each word's [W]-step row, scattered back to its window slot.
            let e_w_grp =
                ops::embedding_gather_u32(gpu, &h_flat, &self.ids.get(1), lay.readout.len(), hc); // [n_g, HC]
            ops::scatter_rows(gpu, &mut e_w, &e_w_grp, grp);
            // This group's forward cache is dead: the encoder backward re-forwards each
            // group to rebuild it (activation checkpointing, see below), so nothing
            // will ever read what was just saved.
            //
            // `drop_all_act`, not `drop_saved_act`: the latter clears only the FFN
            // buffers and the cell cache, leaving each projection's saved input, its
            // bf16 GEMM staging and the norms' `x̂` resident. Those are sized to THIS
            // group's rectangle, the next group's rectangle is a different shape, and
            // reuse is by capacity — so without this every group leaves a full set
            // behind. Measured: the encoder held 1259 MB for a stage whose largest
            // single rectangle is ~25 MB.
            for blk in self.encoder.blocks.iter_mut() {
                blk.drop_all_act(gpu);
            }
        }
        mark("encoder fwd");

        // PHASE 2: BACKBONE
        //
        // Chunked over the word axis when `BACKBONE_CHUNK` is set. The backbone holds
        // one row per word in every block, from that block's forward until its
        // backward, so an unchunked sweep is O(words) resident per block and device
        // memory scales with the window — the reason long windows OOM.
        //
        // Chunk-major instead: chunk c passes through all blocks with each cell's
        // recurrent state carried from chunk c-1, so the arithmetic is identical to the
        // unchunked sweep (pinned by `chunked_carry_matches_unchunked` and
        // `mlstm_chunked_carry_matches_whole`) while only the chunks in flight are
        // resident. Backward then unwinds chunks right to left, carrying the BPTT
        // state the other way.
        let bb_in = self.bb_front.forward_alloc(gpu, &e_w); // [dw, WH]
        // A block holds ONE forward cache, so a chunk-major forward would have chunk
        // `chunk_spans(dw, dw)` is exactly the pre-chunking path, so the two modes are
        // the same code rather than two.
        let chunk = match self.bb_chunk {
            Some(c) => c.min(dw).max(1),
            None if BACKBONE_CHUNKED_BACKWARD => backbone_chunk(dw),
            None => dw,
        };
        let spans = chunk_spans(dw, chunk);
        // The largest chunk's row count — what the backbone's pool actually needs to
        // hold, as opposed to the whole window's `dw`.
        let bb_rows_max = spans.iter().map(|&(_, len)| len).max().unwrap_or(0);

        let chunked = spans.len() > 1;
        if chunked {
            for blk in self.bb_blocks.iter_mut() {
                blk.set_carry(true);
                blk.reset_state(gpu);
            }
        }
        // `bb_back`'s input per chunk. `Linear::forward` saves its input for `dW`, and a
        // later chunk's forward would overwrite it — so each chunk's is kept here and
        // handed back through `backward_with_x`. Without this the backbone's output
        // projection would accumulate `dW` from the LAST chunk only: a silent wrong
        // gradient, not a crash.
        let mut back_in: Vec<DTensor> = Vec::with_capacity(spans.len());
        let mut o_parts: Vec<DTensor> = Vec::with_capacity(spans.len());
        for &(c0, len) in &spans {
            // This chunk's slice of the front projection's output, as its own tensor:
            // the blocks write through their own buffers and a chunk's activations must
            // not alias the whole-window one.
            let mut hb = slice_rows(gpu, &bb_in, c0, len, wh).reshaped(&[1, len, wh]);
            let mut hb_next = DTensor::uninit(gpu, &[1, len, wh]);
            for blk in self.bb_blocks.iter_mut() {
                blk.forward(gpu, &hb, &mut hb_next);
                std::mem::swap(&mut hb, &mut hb_next);
            }
            let flat = hb.reshaped(&[len, wh]);
            let mut y = DTensor::uninit(gpu, &[len, hc]);
            self.bb_back.forward_shared(gpu, &flat, &mut y);
            back_in.push(flat);
            o_parts.push(y);
        }
        let o = if o_parts.len() == 1 {
            o_parts.pop().expect("one chunk")
        } else {
            concat_rows(gpu, &o_parts, dw, hc)
        };
        mark("backbone fwd");

        // PHASE 3: DECODER (forward + backward, per length group)
        // Word w's decode target is word w+1, so groups are keyed on the length of
        // the DECODED word. Each group runs forward and straight back again: the
        // decoder's backward needs nothing from the backbone's, so a group's
        // activations die before the next group allocates — only one group's worth
        // of decoder rows (and of the [rows, vocab] logits) is ever resident.
        sc.dec_lens.clear();
        sc.dec_lens
            .extend((0..dw).map(|w| words[w + 1].end - words[w + 1].start));
        group_by_len(&sc.dec_lens, no_group, group_cap, &mut sc.dec_groups);
        // Every group scales by the WINDOW's valid-row count, so the summed loss and
        // grads match what one big rectangle would have produced. Under SFT masking
        // only the response words' rows count, so the normalizer (and the reported
        // loss) is the number of *unmasked* rows — the mean CE over response tokens.
        let word_on = |w: usize| word_loss.is_none_or(|m| m[w]);
        let valid_rows: usize = (0..dw)
            .filter(|&w| word_on(w))
            .map(|w| sc.dec_lens[w] + 1)
            .sum();
        let inv = 1.0 / (valid_rows.max(1) as f32);

        // Zero the device loss accumulator for this window. Each group adds its own
        // scaled row sum; the total comes back in one read after the decode loop.
        let mut acc = match self.loss_acc.take() {
            Some(a) => a,
            None => gpu.stream.alloc_zeros::<f32>(1).expect("alloc loss_acc"),
        };
        gpu.stream
            .memset_zeros(&mut acc)
            .expect("zero loss_acc");
        let mut d_o = DTensor::zeros(gpu, &[dw, hc]);
        // Largest single decoder group, for the pool trim at the end of the window —
        // the groups are sequential and only one is resident at a time.
        let mut dec_rows_max = 0usize;
        for g in 0..sc.dec_groups.len() {
            let grp = &sc.dec_groups[g];
            let n_g = grp.len();
            let tmax = grp.iter().map(|&w| sc.dec_lens[w] + 1).max().unwrap();
            let rows = n_g * tmax;
            dec_rows_max = dec_rows_max.max(rows);

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
                let s = words[w + 1].start;
                // A masked (prompt) word is still fed forward so its state and
                // the tied-table char grads are produced, but its slots get CE
                // mask 0 — no loss, no logit gradient from a prompt token.
                let on = word_on(w) as u32;
                sc.o_rows.push((i * tmax) as u32);
                for k in 1..=m {
                    sc.char_rows.push((i * tmax + k) as u32);
                    sc.char_ids.push(tokens[s + k - 1] as u32);
                }
                for k in 0..m {
                    sc.targets[i * tmax + k] = tokens[s + k] as u32;
                    sc.mask[i * tmax + k] = on;
                }
                sc.targets[i * tmax + m] = w_token as u32;
                sc.mask[i * tmax + m] = on;
            }

            // The group's word indices, narrowed once for the two gathers/scatters
            // that address `o` / `d_o` by window slot.
            sc.grp_ids.clear();
            sc.grp_ids.extend(grp.iter().map(|&w| w as u32));

            // All six id lists in one pinned async upload. Six separate
            // `upload_ids_u32` calls would be six device allocations and six
            // *blocking* H2Ds — `memcpy_htod` from a pageable slice is synchronous, so
            // each one stalls the compute stream, and this runs once per length group.
            self.ids.upload(
                gpu,
                &[
                    &sc.grp_ids,
                    &sc.o_rows,
                    &sc.char_rows,
                    &sc.char_ids,
                    &sc.targets,
                    &sc.mask,
                ],
            );
            let (d_grp, d_o_rows) = (self.ids.get(0), self.ids.get(1));
            let (d_char_rows, d_char_ids) = (self.ids.get(2), self.ids.get(3));
            let (d_targets, d_mask) = (self.ids.get(4), self.ids.get(5));

            // Build the decoder input: zeros, then scatter the context and char rows.
            let o_grp = ops::embedding_gather_u32(gpu, &o, &d_grp, n_g, hc); // this group's contexts
            let mut dec_in = DTensor::zeros(gpu, &[rows, hc]);
            ops::scatter_rows_u32(gpu, &mut dec_in, &o_grp, &d_o_rows);
            let char_vecs =
                ops::embedding_gather_u32(gpu, &self.table, &d_char_ids, sc.char_ids.len(), hc);
            ops::scatter_rows_u32(gpu, &mut dec_in, &char_vecs, &d_char_rows);

            let mut hd = dec_in.reshaped(&[n_g, tmax, hc]);
            let mut hd_next = DTensor::uninit(gpu, &[n_g, tmax, hc]);
            for blk in self.dec_blocks.iter_mut() {
                blk.forward(gpu, &hd, &mut hd_next);
                std::mem::swap(&mut hd, &mut hd_next);
            }
            let hdn = self.dec_norm.forward_alloc(gpu, &hd.reshaped(&[rows, hc]));
            let logits = self.dec_head.forward_alloc(gpu, &hdn);
            let capped = ops::softcap_forward(gpu, &logits, self.cfg.cap);

            let (_, d_capped) = ops::masked_softmax_ce_u32_into(
                gpu,
                &capped,
                &d_targets,
                &d_mask,
                inv,
                Some(&mut acc),
            );

            let d_logits = ops::softcap_backward(gpu, &d_capped, &capped, self.cfg.cap);
            let d_hdn = self.dec_head.backward_alloc(gpu, &d_logits);
            let d_hd_flat = self.dec_norm.backward_alloc(gpu, &d_hdn);
            let mut d_hd = d_hd_flat.reshaped(&[n_g, tmax, hc]);
            let mut d_hd_next = DTensor::uninit(gpu, &[n_g, tmax, hc]);
            for blk in self.dec_blocks.iter_mut().rev() {
                blk.backward(gpu, &d_hd, &mut d_hd_next);
                std::mem::swap(&mut d_hd, &mut d_hd_next);
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
            // This group is completely finished — forward AND backward — so nothing
            // here will be read again. Release the whole activation set rather than
            // letting the next group's differently-shaped rectangle reallocate around
            // it: same reasoning as the encoder above, and the decoder is the same
            // size (1225 MB measured for a ~25 MB working set).
            //
            // Safe here in a way it would not be mid-loop: the decoder's backward has
            // already consumed everything its forward produced.
            for blk in self.dec_blocks.iter_mut() {
                blk.drop_all_act(gpu);
            }
        }
        // One readback for the whole window, outside the group loop. Blocking here is
        // fine — the decode phase is over — where per-group it stalled the pipeline.
        let loss = gpu
            .stream
            .clone_dtoh(&acc)
            .expect("download loss_acc")[0];
        self.loss_acc = Some(acc);
        // Per-word NLL from the same total: `loss` is the row sum over `valid_rows`,
        // so scaling by rows/words re-averages it over words instead. Both counts are
        // already on the host, so the second metric costs no device work.
        let scored_words = (0..dw).filter(|&w| word_on(w)).count();
        self.last_word_loss = loss * (valid_rows as f32) / (scored_words.max(1) as f32);
        mark("decoder fwd + bwd");

        // Backbone backward.
        //
        // The last block's activations are wanted first, and nothing inside the loop
        // precedes them — so their upload is issued here, ahead of `bb_back`'s
        // backward, and overlaps it.
        if let Some(last) = self.bb_blocks.last_mut() {
            last.prefetch_act(gpu);
        }
        // `bb_back` ran `forward_shared` per chunk, so it saved nothing — its input
        // came back through `back_in`. Feed the chunks' inputs back in the same order
        // so `dW = XᵀdY` accumulates over all of them (beta = 1), which is what makes
        // the chunked gradient equal the whole-window one.
        let d_bb_out = if back_in.len() == 1 {
            self.bb_back
                .backward_alloc_with_x(gpu, &back_in[0], &d_o)
        } else {
            let mut parts: Vec<DTensor> = Vec::with_capacity(back_in.len());
            for (i, &(c0, len)) in spans.iter().enumerate() {
                let d_o_c = slice_rows(gpu, &d_o, c0, len, hc);
                parts.push(
                    self.bb_back
                        .backward_alloc_with_x(gpu, &back_in[i], &d_o_c),
                );
            }
            concat_rows(gpu, &parts, dw, wh)
        };
        // Chunks unwind right to left, the mirror of the forward's left-to-right, so the
        // BPTT state each cell carries flows backwards across the same borders the
        // forward state crossed forwards. Every chunk passes through all blocks before
        // the next one starts, which is what lets a chunk's activations be released as
        // soon as it is unwound.
        let mut d_parts: Vec<Option<DTensor>> = (0..spans.len()).map(|_| None).collect();
        let mut d_bb_out = Some(d_bb_out);
        for (ci, &(_c0, len)) in spans.iter().enumerate().rev() {
            // The last block is wanted first in every chunk, not just the first one:
            // the prefetch above covers only the leftmost pass, so without this each
            // later chunk opens with an unhidden upload. Issued before the slicing and
            // the BPTT reset below, which is the compute it hides behind.
            if chunked {
                if let Some(last) = self.bb_blocks.last_mut() {
                    last.prefetch_act(gpu);
                }
            }
            // The rightmost chunk starts with no gradient coming from its right.
            if chunked && ci + 1 == spans.len() {
                for blk in self.bb_blocks.iter_mut() {
                    blk.reset_bptt(gpu);
                }
            }
            let mut d_hb = if chunked {
                let src = d_bb_out.as_ref().expect("d_bb_out");
                slice_rows(gpu, src, _c0, len, wh).reshaped(&[1, len, wh])
            } else {
                // Single chunk: hand the whole tensor over rather than copying it.
                d_bb_out.take().expect("d_bb_out").reshaped(&[1, len, wh])
            };
            let mut d_hb_next = DTensor::uninit(gpu, &[1, len, wh]);
            // With offload on, each block's saved activations have to come back from the
            // host first — so block i-1's upload is started *before* block i's backward
            // runs, giving it a whole block of compute to hide behind. Issuing the copy
            // and waiting for it in the same breath exposes the whole transfer
            // (measured: +37 ms).
            for i in (0..self.bb_blocks.len()).rev() {
                if i > 0 {
                    let (head, tail) = self.bb_blocks.split_at_mut(i);
                    head[i - 1].prefetch_act(gpu);
                    tail[0].backward(gpu, &d_hb, &mut d_hb_next);
                } else {
                    self.bb_blocks[0].backward(gpu, &d_hb, &mut d_hb_next);
                }
                std::mem::swap(&mut d_hb, &mut d_hb_next);
            }
            d_parts[ci] = Some(d_hb.reshaped(&[len, wh]));
        }
        let mut d_parts: Vec<DTensor> = d_parts
            .into_iter()
            .map(|p| p.expect("every chunk unwound"))
            .collect();
        let d_hb_all = if d_parts.len() == 1 {
            d_parts.pop().expect("one chunk")
        } else {
            concat_rows(gpu, &d_parts, dw, wh)
        };
        let d_e_w = self.bb_front.backward_alloc(gpu, &d_hb_all); // [dw, HC]
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
            // This group's three id lists in one pinned async upload.
            sc.grp_ids.clear();
            sc.grp_ids.extend(grp.iter().map(|&w| w as u32));
            self.ids.upload(gpu, &[&lay.ids, &lay.readout, &sc.grp_ids]);
            let embedded =
                ops::embedding_gather_u32(gpu, &self.table, &self.ids.get(0), lay.ids.len(), hc);
            let mut h = embedded.reshaped(&[grp.len(), tmax, hc]);
            let mut next = DTensor::uninit(gpu, &[grp.len(), tmax, hc]);
            for blk in self.encoder.blocks.iter_mut() {
                blk.forward(gpu, &h, &mut next);
                std::mem::swap(&mut h, &mut next);
            }
            drop((h, next));

            // Scatter this group's d_e_w onto its [W]-step rows, rest zero.
            let d_e_w_grp = ops::embedding_gather_u32(gpu, &d_e_w, &self.ids.get(2), grp.len(), hc); // [n_g, HC]
            let mut d_h = DTensor::zeros(gpu, &[grp.len() * tmax, hc]);
            ops::scatter_rows_u32(gpu, &mut d_h, &d_e_w_grp, &self.ids.get(1));
            let mut d_h = d_h.reshaped(&[grp.len(), tmax, hc]);
            let mut d_h_next = DTensor::uninit(gpu, &[grp.len(), tmax, hc]);
            for blk in self.encoder.blocks.iter_mut().rev() {
                blk.backward(gpu, &d_h, &mut d_h_next);
                std::mem::swap(&mut d_h, &mut d_h_next);
            }
            let d_embedded = d_h.reshaped(&[grp.len() * tmax, hc]);
            ops::embedding_scatter_add_u32(
                gpu,
                &mut self.dtable,
                &self.ids.get(0),
                lay.ids.len(),
                &d_embedded,
                hc,
            );
            for blk in self.encoder.blocks.iter_mut() {
                blk.drop_all_act(gpu);
            }
        }
        mark("encoder bwd");

        self.scratch = sc;
        // Release pooled scratch far larger than this window needed.
        //
        // Window sizes vary across a corpus and both `Buf` and `Pool` reuse by
        // capacity, so without this every buffer ratchets to the largest window ever
        // seen — device memory climbed monotonically window over window on the real
        // `hg` path at WORDS_PER_SEQ=2048 (10.4 GB, 13.4, 15.9, 16.6) until it
        // aborted, while the per-window footprint was never the problem.
        //
        // The bound is generous (see `buf::RETAIN_SLACK`), so an ordinary run of
        // similar windows frees nothing and the allocator stays off the hot path.
        // One chunk's rows, not the window's — only a chunk is ever resident.
        self.trim_pools(bb_rows_max, enc_rows, dec_rows_max);
        loss
    }

    /// Drop every layer-owned activation buffer and pooled scratch, everywhere.
    ///
    /// Diagnostic (`examples/vram_audit.rs`): after a completed step nothing but
    /// weights and optimizer moments should be live, so whatever this frees was being
    /// retained across steps. Not for the training path — the next window reallocates
    /// all of it.
    /// Override the backbone chunk length for this model. `None` uses
    /// [`config::BACKBONE_CHUNK`]; a value >= the window's word count is the unchunked
    /// path. Per-instance rather than an env var so callers do not race each other.
    pub fn set_bb_chunk(&mut self, chunk: Option<usize>) {
        self.bb_chunk = chunk;
    }

    /// Override the encoder/decoder group row cap. `None` uses
    /// [`config::GROUP_MAX_ROWS`]; `Some(0)` disables the cap.
    pub fn set_group_cap(&mut self, cap: Option<usize>) {
        self.group_cap = cap;
    }

    /// L2 norm of each block's accumulated gradient, for one stage ("encoder",
    /// "backbone", "decoder"). Diagnostic — one host round-trip per tensor, so this
    /// belongs in a probe, not a training loop.
    pub fn grad_norms_by_block(&self, gpu: &Gpu, stage: &str) -> Vec<f32> {
        let blocks = match stage {
            "encoder" => &self.encoder.blocks,
            "backbone" => &self.bb_blocks,
            "decoder" => &self.dec_blocks,
            other => panic!("grad_norms_by_block: unknown stage {other}"),
        };
        blocks
            .iter()
            .map(|b| {
                let sq: f32 = b
                    .grads()
                    .iter()
                    .flat_map(|g| g.to_host(gpu).data.to_vec())
                    .map(|v| v * v)
                    .sum();
                sq.sqrt()
            })
            .collect()
    }

    /// Per-block `(min |n|, max |c|, max |c/n|)` for one stage, `None` for blocks
    /// whose cell carries no stabilized normalizer. Diagnostic.
    pub fn state_extremes_by_block(
        &self,
        gpu: &Gpu,
        stage: &str,
    ) -> Vec<Option<(f32, f32, f32)>> {
        let blocks = match stage {
            "encoder" => &self.encoder.blocks,
            "backbone" => &self.bb_blocks,
            "decoder" => &self.dec_blocks,
            other => panic!("state_extremes_by_block: unknown stage {other}"),
        };
        blocks.iter().map(|b| b.state_extremes(gpu)).collect()
    }

    pub fn release_activation_buffers(&mut self) {
        for blk in self
            .bb_blocks
            .iter_mut()
            .chain(self.encoder.blocks.iter_mut())
            .chain(self.dec_blocks.iter_mut())
        {
            blk.drop_saved_act();
            blk.trim_to(0);
        }
    }

    /// Per-stage device bytes, split into parameters and retained activations.
    ///
    /// Diagnostic (`examples/vram_audit.rs`). Every shape-based estimate of this
    /// model's memory has been off by roughly an order of magnitude, so this walks the
    /// actual buffers and reports their **capacity** — reuse throughout is by
    /// capacity, so that is what occupies the device.
    ///
    /// Returns `(stage, params, activations)` rows plus a total.
    pub fn retained_report(&self) -> Vec<(String, usize, usize)> {
        let mut rows = Vec::new();
        let mut sum = |label: &str, blocks: &[Box<dyn BlockLike>]| {
            let (mut p, mut a) = (0, 0);
            for b in blocks {
                let (bp, ba) = b.retained_bytes();
                p += bp;
                a += ba;
            }
            rows.push((label.to_string(), p, a));
        };
        sum("encoder blocks", &self.encoder.blocks);
        sum("backbone blocks", &self.bb_blocks);
        sum("decoder blocks", &self.dec_blocks);

        let table = [&self.table, &self.dtable, &self.m_tbl, &self.v_tbl]
            .iter()
            .map(|t| t.capacity() * 4)
            .sum();
        rows.push(("tied table".into(), table, 0));

        for (label, l) in [
            ("bb_front", &self.bb_front),
            ("bb_back", &self.bb_back),
            ("dec_head", &self.dec_head),
        ] {
            let (p, a) = l.retained_bytes();
            rows.push((label.into(), p, a));
        }
        let (p, a) = self.dec_norm.retained_bytes();
        rows.push(("dec_norm".into(), p, a));
        rows
    }

    /// Per-stage retained activations broken out by owner, for the memory audit.
    ///
    /// Columns: `ffn_bufs, pool, norms, projections, cell_saved, cell_other`. The
    /// first two are what `drop_saved_act` + `trim_to` reach; the rest live inside the
    /// sub-layers and survive both, which is where the unaccounted memory was.
    pub fn act_breakdown(&self) -> Vec<(String, [usize; 6])> {
        let stages: [(&str, &Vec<Box<dyn BlockLike>>); 3] = [
            ("encoder", &self.encoder.blocks),
            ("backbone", &self.bb_blocks),
            ("decoder", &self.dec_blocks),
        ];
        stages
            .iter()
            .map(|(label, blocks)| {
                let mut cols = [0usize; 6];
                for b in blocks.iter() {
                    let [ffn, pool, norms, proj, _] = b.act_breakdown();
                    let (cs, co) = b.cell_act_split();
                    for (c, v) in cols.iter_mut().zip([ffn, pool, norms, proj, cs, co]) {
                        *c += v;
                    }
                }
                (label.to_string(), cols)
            })
            .collect()
    }

    /// Per-stage pool free-list shape: `(distinct sizes, buffers)` summed over the
    /// stage's blocks. Diagnostic — a climbing size count is the free list keeping one
    /// entry per distinct shape ever seen.
    pub fn pool_shapes(&self) -> Vec<(String, usize, usize)> {
        let stages: [(&str, &Vec<Box<dyn BlockLike>>); 3] = [
            ("encoder", &self.encoder.blocks),
            ("backbone", &self.bb_blocks),
            ("decoder", &self.dec_blocks),
        ];
        stages
            .iter()
            .map(|(label, blocks)| {
                let (mut sizes, mut bufs) = (0, 0);
                for b in blocks.iter() {
                    let (s, n) = b.pool_shape();
                    sizes += s;
                    bufs += n;
                }
                (label.to_string(), sizes, bufs)
            })
            .collect()
    }

    /// Release every activation every layer holds, everywhere.
    ///
    /// Unlike [`release_activation_buffers`](Self::release_activation_buffers), which
    /// only reaches the blocks' own `Buf`s and pools, this also clears what lives
    /// inside each block's norms and projections — the saved forward inputs, the bf16
    /// GEMM staging and the norms' `x̂` — plus the stage-level `Linear`s and norm.
    /// Diagnostic; the next window reallocates all of it.
    pub fn drop_all_act(&mut self, gpu: &Gpu) {
        for blk in self
            .bb_blocks
            .iter_mut()
            .chain(self.encoder.blocks.iter_mut())
            .chain(self.dec_blocks.iter_mut())
        {
            blk.drop_all_act(gpu);
        }
        for l in [&mut self.bb_front, &mut self.bb_back, &mut self.dec_head] {
            l.drop_saved_act(gpu);
        }
        self.dec_norm.drop_saved_act();
    }

    /// Release the retained activations of one stage only, for the memory audit.
    ///
    /// `stage` is "encoder", "backbone" or "decoder". Lets the audit attribute retained
    /// memory to a stage by releasing them one at a time. Diagnostic only.
    pub fn release_stage(&mut self, stage: &str) {
        let blocks = match stage {
            "encoder" => &mut self.encoder.blocks,
            "backbone" => &mut self.bb_blocks,
            "decoder" => &mut self.dec_blocks,
            other => panic!("release_stage: unknown stage {other}"),
        };
        for blk in blocks.iter_mut() {
            blk.drop_saved_act();
            blk.trim_to(0);
        }
    }

    /// Drop pooled scratch beyond what a `words`-word window needs, in every stage.
    ///
    /// Each stage is sized by what its rectangles actually are: the backbone runs one
    /// row per word, while the encoder and decoder run one row per *character* and are
    /// bounded by `MAX_WORD_BYTES`.
    /// `enc_rows`/`dec_rows` are the LARGEST single group's row count in each stage,
    /// not the window's total. The groups run strictly one after another and a group's
    /// activations are dead before the next one starts, so the resident set is one
    /// group — sizing the bound off `words * (MAX_WORD_BYTES + 1)` (the whole window's
    /// characters) is ~5x too generous and lets every group's oversized buffers
    /// survive instead of being replaced by the one shape that serves all of them.
    ///
    /// `bb_rows` is likewise one chunk, not the window: a chunked sweep only ever has
    /// a chunk's rows live, so bounding the backbone by the whole window leaves its
    /// pool sized for memory it never holds (measured 176 -> 346 MB going 2048 -> 4096
    /// words, while every other backbone column stayed flat).
    fn trim_pools(&mut self, bb_rows: usize, enc_rows: usize, dec_rows: usize) {
        for blk in self.bb_blocks.iter_mut() {
            blk.trim_to(bb_rows);
        }
        for blk in self.encoder.blocks.iter_mut() {
            blk.trim_to(enc_rows);
        }
        for blk in self.dec_blocks.iter_mut() {
            blk.trim_to(dec_rows);
        }
    }

    /// AdamW across every stage. Tied table and the logit head are undecayed;
    /// interior projections decay (matching the project's optimizer convention).
    pub fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        // Every parameter tensor in the model queues its update and one flush issues
        // them a batch at a time. Stepping eagerly is ~883 launches, most of them a
        // single block (biases, norm scales), i.e. almost pure launch overhead.
        // `GPU_NO_ADAMW_BATCH=1` restores the per-tensor path.
        let mut queue = (!ops::no_adamw_batch()).then(ops::AdamwQueue::new);
        let mut q = queue.as_mut();

        match q.as_deref_mut() {
            Some(q) => q.push(
                gpu,
                &mut self.table,
                &self.dtable,
                &mut self.m_tbl,
                &mut self.v_tbl,
                cfg,
                false,
            ),
            None => {
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
            }
        }
        for b in self.encoder.blocks.iter_mut() {
            b.step_q(gpu, cfg, q.as_deref_mut());
        }
        self.bb_front.step_wd_q(gpu, cfg, true, q.as_deref_mut());
        for b in self.bb_blocks.iter_mut() {
            b.step_q(gpu, cfg, q.as_deref_mut());
        }
        self.bb_back.step_wd_q(gpu, cfg, true, q.as_deref_mut());
        for b in self.dec_blocks.iter_mut() {
            b.step_q(gpu, cfg, q.as_deref_mut());
        }
        self.dec_norm.step_q(gpu, cfg, q.as_deref_mut());
        // Logit head: no weight decay and no bias (bias stays at its zero init) so
        // it matches `nn::linear_no_bias` and exports faithfully to the HIER head.
        self.dec_head.step_w_only_q(gpu, cfg, false, q.as_deref_mut());
        if let Some(q) = queue.as_mut() {
            q.flush(gpu, cfg);
        }
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
                seen: self.seen,
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
        let dec_head = {
            // fp32, matching the freshly-built model above — a checkpoint must not
            // load into a different numeric path than it was created with.
            let mut h = super::linear::Linear::from_parts(
                gpu,
                &tensor_from_matrix(&head.weights),
                &crate::tensor::Tensor::zeros(&[vocab]),
            );
            h.set_fp32();
            h
        };
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
        model.seen = stacks.seen;
        model.table = table;
        model.encoder.blocks = enc_blocks;
        model.bb_front = bb_front;
        model.bb_blocks = bb_blocks;
        model.bb_back = bb_back;
        model.dec_blocks = dec_blocks;
        model.dec_norm = dec_norm;
        model.dec_head = dec_head;
        // The loaded blocks replaced the ones `new` set up, so re-apply the offload
        // opt-in to the stack that actually ends up in the model.
        model.enable_backbone_offload(gpu);
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
        let words = vec![
            Range { start: 0, end: 3 },
            Range { start: 3, end: 5 },
            Range { start: 5, end: 8 },
            Range { start: 8, end: 10 },
        ];

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
        model.seen.add_pretrain(tokens.len(), words.len());
        model.seen.add_sft(10, 4, 6, 2);
        model.save(&gpu, path, &[cfg.w_token as u16]).expect("save");
        let mut back = Hierarchical::load(&gpu, path, cfg.w_token).expect("load");
        assert_eq!(back.cfg, cfg, "config did not survive the round-trip");
        assert_eq!(back.step_count, model.step_count, "step count lost");
        assert_eq!(back.seen, model.seen, "data counters lost");
        let reloaded = back.forward_backward(&gpu, &tokens, &words);
        assert!(
            (reloaded - last).abs() < 1e-4,
            "reloaded model gives a different loss: {last} -> {reloaded}"
        );
        let _ = std::fs::remove_file(path);
    }

    /// The same window, run twice, must give bit-identical losses.
    ///
    /// The grouped encoder/decoder fires one `IdBatch::upload` per group, and the
    /// H2D out of its pinned staging buffer is async. Rotating slots alone only
    /// buys `ID_SLOTS - 1` iterations of slack, so with enough groups the CPU laps
    /// the device and refills a buffer whose copy has not been read yet — the ids
    /// silently become another group's. It is invisible in a small window:
    /// `grouping_matches_single_rectangle` uses 5 words and never reproduced it,
    /// while ~160 words reproduces it every time. Hence the word count here.
    ///
    /// Deviations ranged from 5e-3 to 6e1 — the large end is a diverged run.
    ///
    /// The widths below are the real model's, and they have to be: the race needs
    /// enough queued device work per group for the CPU to lap it, and at this
    /// suite's usual toy sizes (hc=16, wh=24) it does not reproduce at any word
    /// count. Verified to fail with the event wait in `IdBatch::upload` removed.
    #[test]
    fn repeated_forward_backward_is_deterministic() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let cfg = HierCfg {
            vocab: 12,
            hc: 256,
            wh: 768,
            enc_blocks: 4,
            bb_blocks: 8,
            dec_blocks: 4,
            heads: 8,
            dqk: 96,
            w_token: 11,
            cap: 30.0,
        };
        // Enough words, with a spread of lengths, that the encoder and decoder each
        // fire many groups per window.
        let mut tokens: Vec<usize> = Vec::new();
        let mut words: Vec<Range<usize>> = Vec::new();
        for w in 0..1024 {
            let s = tokens.len();
            let len = 1 + (w * 7 + w / 3) % 16;
            for k in 0..len {
                tokens.push(1 + (k + w) % 9);
            }
            words.push(Range { start: s, end: tokens.len() });
        }

        // One model, never stepped: identical weights every rep, so any spread is
        // the forward/backward itself.
        let mut model = Hierarchical::new(&gpu, &cfg);
        let first = model.forward_backward(&gpu, &tokens, &words);
        for rep in 1..4 {
            let got = model.forward_backward(&gpu, &tokens, &words);
            assert_eq!(
                got.to_bits(),
                first.to_bits(),
                "rep {rep}: loss {got} != {first} — nondeterministic forward/backward"
            );
        }
    }

    /// `word_loss` is the window's summed NLL re-averaged over words instead of
    /// rows, so it must equal the char loss scaled by the true rows-per-word ratio.
    /// The ratio here is recomputed from the word lengths, independently of the
    /// counters the model itself used — including the `+1` for each `[W]` step.
    #[test]
    fn word_loss_is_char_loss_rescaled_by_rows_per_word() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let cfg = HierCfg {
            vocab: 12,
            hc: 32,
            wh: 48,
            enc_blocks: 2,
            bb_blocks: 2,
            dec_blocks: 2,
            heads: 4,
            dqk: 8,
            w_token: 11,
            cap: 30.0,
        };
        // Mixed lengths, so rows-per-word is not a whole number and a mistaken
        // `+1` (or a missing one) cannot coincidentally match.
        let mut tokens: Vec<usize> = Vec::new();
        let mut words: Vec<Range<usize>> = Vec::new();
        for w in 0..32 {
            let s = tokens.len();
            let len = 1 + (w * 5) % 7;
            for k in 0..len {
                tokens.push(1 + (k + w) % 9);
            }
            words.push(Range { start: s, end: tokens.len() });
        }

        let mut model = Hierarchical::new(&gpu, &cfg);
        let char_loss = model.forward_backward(&gpu, &tokens, &words);
        let word_loss = model.last_word_loss();

        // Word 0 is encode-only; every decoded word contributes its chars plus [W].
        let decoded = &words[1..];
        let rows: usize = decoded.iter().map(|r| r.end - r.start + 1).sum();
        let expect = char_loss * rows as f32 / decoded.len() as f32;
        assert!(
            (word_loss - expect).abs() <= 1e-4 * expect.abs().max(1.0),
            "word_loss {word_loss} != {expect} (char {char_loss}, {rows} rows, {} words)",
            decoded.len()
        );
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
        let words = vec![
            Range { start: 0, end: 1 },
            Range { start: 1, end: 4 },
            Range { start: 4, end: 6 },
            Range { start: 6, end: 12 },
            Range { start: 12, end: 16 },
        ];

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

    /// Chunking the backbone's word axis is a pure memory change: the recurrent state
    /// crosses the chunk borders forwards and the BPTT state crosses them backwards, so
    /// the arithmetic is the unchunked sweep's, refactored.
    ///
    /// This is the test that catches the failure mode chunking actually has. Every
    /// block keeps one forward cache per chunk still owed a backward; if any of them
    /// were overwritten by a later chunk's forward, backward would read the wrong
    /// activations and produce a *plausible but wrong* gradient rather than crashing.
    /// So the comparison runs through a full optimizer step and checks the tied table
    /// (fed by all three stages) and the backbone's own projections.
    #[test]
    fn backbone_chunked_matches_unchunked() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let cfg = HierCfg {
            vocab: 12,
            hc: 16,
            wh: 24,
            enc_blocks: 1,
            bb_blocks: 3,
            dec_blocks: 1,
            heads: 2,
            dqk: 8,
            w_token: 11,
            cap: 30.0,
        };
        // 12 words, so a chunk of 4 gives three chunks — two interior borders, which
        // is what a single-border test would miss.
        let tokens: Vec<usize> = (0..24).map(|i| 1 + i % 9).collect();
        let words: Vec<Range<usize>> = (0..12).map(|w| Range { start: w * 2, end: w * 2 + 2 }).collect();

        let run = |chunk: usize| -> (f32, Vec<f32>, Vec<f32>) {
            let mut model = Hierarchical::new(&gpu, &cfg);
            // Same starting weights for both legs.
            let seed = std::env::temp_dir().join("gpu_chunk_seed.hier");
            let seed = seed.to_str().unwrap();
            if chunk != usize::MAX {
                model.save(&gpu, seed, &[]).expect("save seed");
            } else {
                model = Hierarchical::load(&gpu, seed, cfg.w_token).expect("load seed");
            }
            // Per-instance, so this test does not race the rest of the suite. A chunk
            // >= the word count is exactly the single-span (unchunked) path.
            model.bb_chunk = Some(chunk);
            let loss = model.forward_backward(&gpu, &tokens, &words);
            // The raw gradients, before Adam: its update is scale-invariant in the
            // gradient, so a near-zero gradient turns a last-digit difference into a
            // full-size weight difference and post-step weights cannot distinguish
            // that from a real error.
            let table: Vec<f32> = model.dtable.to_host(&gpu).data.to_vec();
            let front: Vec<f32> = model.bb_front.dw.to_host(&gpu).data.to_vec();
            (loss, table, front)
        };

        let (loss_c, table_c, front_c) = run(4);
        let (loss_u, table_u, front_u) = run(usize::MAX);

        // A tolerance test, not an equality one, for the reason
        // `mlstm_chunking_matches_single_chunk` documents: the two blockings sum the
        // same terms in a different order, and with bf16 q/k/v slabs they also quantize
        // at different points. What it must catch is a backward reading another chunk's
        // activations, which is a structural error orders of magnitude larger than this
        // floor — not a last-digit difference.
        let slab = if gpu.kernels.slab_bf16 { 8.0 } else { 1.0 };
        assert!(
            (loss_c - loss_u).abs() < 2e-4 * slab,
            "chunked loss {loss_c} != unchunked loss {loss_u}"
        );
        // Post-Adam weights are ~1e-2 and the update is scale-invariant in the
        // gradient, so these are relative bounds — an absolute one would demand
        // precision chunk-boundary reassociation cannot give.
        // Scale-aware, not pointwise-relative: these gradient vectors have near-zero
        // entries where a pointwise relative error is meaningless (a 1e-9 against a
        // 2e-9 is "100% off" and says nothing). The meaningful question is the error
        // against the vector's own magnitude.
        let check = |c: &Vec<f32>, u: &Vec<f32>, what: &str| {
            let scale = c
                .iter()
                .chain(u.iter())
                .fold(0.0f32, |m, v| m.max(v.abs()))
                .max(1e-12);
            let worst = c
                .iter()
                .zip(u)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
                / scale;
            assert!(
                worst < 2e-2 * slab,
                "{what} gradient diverged: worst {worst} relative to magnitude {scale}"
            );
        };
        check(&table_c, &table_u, "tied table");
        check(&front_c, &front_u, "bb_front");
    }

    /// A backbone sweep of three or more chunks must give the same gradients as one.
    ///
    /// Distinct from `backbone_chunked_matches_unchunked`, which runs chunks shorter
    /// than `GRAPH_MIN_T` and so never exercises the captured-graph path. Here every
    /// chunk is long enough to be captured *and* there are three of them, which is what
    /// it takes to replay a graph after `chunk_saved` has swapped the buffers it was
    /// captured against — a replay against freed allocations, i.e. NaN gradients from
    /// an out-of-bounds read rather than a wrong-but-finite number. Two chunks cannot
    /// catch it: the first replay still sees its own buffers.
    #[test]
    fn backbone_three_chunks_match_unchunked() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        // Wide enough that a chunk's buffers are a real allocation: at a toy width the
        // pool hands every chunk the same address back and the stale-pointer replay is
        // accidentally harmless, so the bug does not reproduce.
        let cfg = HierCfg {
            vocab: 64,
            hc: 256,
            wh: 768,
            enc_blocks: 1,
            // Block 0 is sLSTM under the `i % 4` rule — the cell whose loop is captured.
            bb_blocks: 2,
            dec_blocks: 1,
            heads: 8,
            dqk: 96,
            w_token: 63,
            cap: 30.0,
        };
        // 384 decoded words in chunks of 128: three chunks, each well over GRAPH_MIN_T.
        let words_n = 385;
        let tokens: Vec<usize> = (0..words_n * 2).map(|i| 1 + i % 9).collect();
        let words: Vec<Range<usize>> = (0..words_n)
            .map(|w| Range { start: w * 2, end: w * 2 + 2 })
            .collect();

        let run = |chunk: usize| -> (f32, Vec<f32>, Vec<f32>) {
            let mut model = Hierarchical::new(&gpu, &cfg);
            let seed = std::env::temp_dir().join("gpu_three_chunk_seed.hier");
            let seed = seed.to_str().unwrap();
            if chunk != usize::MAX {
                model.save(&gpu, seed, &[]).expect("save seed");
            } else {
                model = Hierarchical::load(&gpu, seed, cfg.w_token).expect("load seed");
            }
            model.bb_chunk = Some(chunk);
            let loss = model.forward_backward(&gpu, &tokens, &words);
            let table: Vec<f32> = model.dtable.to_host(&gpu).data.to_vec();
            let front: Vec<f32> = model.bb_front.dw.to_host(&gpu).data.to_vec();
            (loss, table, front)
        };

        let (loss_c, table_c, front_c) = run(128);
        let (loss_u, table_u, front_u) = run(usize::MAX);

        assert!(
            loss_c.is_finite() && table_c.iter().all(|v| v.is_finite()),
            "three-chunk sweep produced a non-finite loss/gradient (loss {loss_c})"
        );
        let slab = if gpu.kernels.slab_bf16 { 8.0 } else { 1.0 };
        // Relative, and looser than the toy-width test's absolute bound: at this width
        // the unchunked leg is one 384-step sweep against three 128-step ones, so the
        // reassociation spread is genuinely larger. What must be caught is a replay
        // against another chunk's buffers, which is a NaN or an order-of-magnitude
        // miss, not a third decimal place.
        assert!(
            (loss_c - loss_u).abs() < 1e-3 * slab * loss_u.abs().max(1.0),
            "three-chunk loss {loss_c} != unchunked loss {loss_u}"
        );
        // Every gradient finite, and the two legs agreeing in aggregate. A pointwise
        // bound cannot be tight here — 384 recurrent steps reassociated three ways
        // spread individual entries by tens of percent — but a replay against another
        // chunk's buffers moves the whole vector's norm, which this does catch.
        let check = |c: &Vec<f32>, u: &Vec<f32>, what: &str| {
            assert!(
                c.iter().all(|v| v.is_finite()),
                "{what} gradient has a non-finite entry"
            );
            let norm = |v: &Vec<f32>| v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let (nc, nu) = (norm(c), norm(u));
            let rel = (nc - nu).abs() / nu.max(1e-12);
            assert!(
                rel < 5e-2 * slab,
                "{what} gradient norm diverged: {nc} vs {nu} ({rel} relative)"
            );
        };
        check(&table_c, &table_u, "tied table");
        check(&front_c, &front_u, "bb_front");
    }

    /// Capping a group's rows is a pure batching change: splitting one length bucket
    /// into several same-shape rectangles must give the same loss AND the same
    /// gradients as running it whole.
    ///
    /// The cap exists because a bucket holds every word of its length in the window, so
    /// it grows with the window and the whole stage's buffers size to it. This is what
    /// says the split is free — the groups were already independent.
    #[test]
    fn group_cap_matches_uncapped() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let cfg = HierCfg {
            vocab: 12,
            hc: 16,
            wh: 24,
            enc_blocks: 1,
            bb_blocks: 2,
            dec_blocks: 1,
            heads: 2,
            dqk: 8,
            w_token: 11,
            cap: 30.0,
        };
        // 24 words of one length, so they land in ONE bucket and the cap really splits
        // it rather than the lengths doing the work.
        let tokens: Vec<usize> = (0..48).map(|i| 1 + i % 9).collect();
        let words: Vec<Range<usize>> = (0..24)
            .map(|w| Range {
                start: w * 2,
                end: w * 2 + 2,
            })
            .collect();

        let run = |cap: usize| -> (f32, Vec<f32>) {
            let mut model = Hierarchical::new(&gpu, &cfg);
            let seed = std::env::temp_dir().join("gpu_groupcap_seed.hier");
            let seed = seed.to_str().unwrap();
            if cap == 0 {
                model.save(&gpu, seed, &[]).expect("save seed");
            } else {
                model = Hierarchical::load(&gpu, seed, cfg.w_token).expect("load seed");
            }
            model.group_cap = Some(cap);
            let loss = model.forward_backward(&gpu, &tokens, &words);
            // Raw gradients, not post-step weights: Adam's update is scale-invariant,
            // so a near-zero gradient turns a last-digit difference into a full-size
            // weight difference.
            let g: Vec<f32> = model.dtable.to_host(&gpu).data.to_vec();
            (loss, g)
        };

        let (loss_whole, g_whole) = run(0); // uncapped: one group
        let (loss_split, g_split) = run(9); // forces several pieces per bucket

        assert!(
            (loss_whole - loss_split).abs() < 1e-5,
            "capped loss {loss_split} != uncapped {loss_whole}"
        );
        // Same terms, same order within a piece, so this is near-exact — only the
        // per-piece grad accumulation reassociates.
        let scale = g_whole
            .iter()
            .chain(g_split.iter())
            .fold(0.0f32, |m, v| m.max(v.abs()))
            .max(1e-12);
        let worst = g_whole
            .iter()
            .zip(&g_split)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
            / scale;
        assert!(worst < 1e-4, "capped gradient diverged by {worst}");
    }
}
