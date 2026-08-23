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
//! the head, plus all gradients and AdamW moments — lives in `GTensor<f32>`s. Index
//! bookkeeping (which row is a `[W]` step, which slot is a char) is computed on
//! the host and uploaded as id lists; only tensor *data* stays on the device.
//!
//! Checkpoints: `save`/`load` use the unified `NNM1` container (`src/format.rs`,
//! kind = Hierarchical) — the same named-section layout the CPU model produces,
//! so a GPU-trained model opens directly in the CPU sampler/probe (`hs` / `hp`).
//! Weights only; the AdamW moments are not persisted, so a resumed run restarts
//! them.

use std::collections::BTreeMap;
use std::range::Range;
use std::time::Instant;
use std::{io, mem};

use cudarc::driver::CudaSlice;

use super::arena::{ParamArena, ParamKind, ParamSlot};
use super::block::Block;
use super::{GTensor, Gpu, linear::Linear, mlstm::MLstm, ops, rms_norm::RmsNorm, slstm::SLstm};
use crate::format::{Meta, ModelKind, Seen, Writer};
use crate::gpu::arena::TrainingCache;
use crate::gpu::block::BlockLike;
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
pub struct ModelCfg {
    pub vocab: usize,
    /// char/context hidden (tied embedding + decoder width)
    pub hc: usize,
    /// backbone width
    pub wh: usize,
    pub enc_blocks: usize,
    pub bb_blocks: usize,
    pub dec_blocks: usize,
    /// mLSTM heads
    pub heads: usize,
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
                    if i.is_multiple_of(2) {
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
/// `no_group` (from `GPU_NO_GROUP=1`) puts every word in one group — one rectangle
/// per window, the A/B baseline for benchmarking and what
/// `grouping_matches_single_rectangle` checks the grouped path against.
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
/// Two things this rests on, both pinned by tests:
///
///   * **Per-chunk activation storage.** Every cache a later chunk's forward would
///     overwrite is one-per-chunk: the FFN's five buffers, both pre-norms, the cell's
///     own cache and its head norm, and one `HostPark` generation per chunk under
///     offload. `backbone_chunked_matches_unchunked` pins the gradients.
///   * **The mLSTM's backward carry.** Under CARRY the last chunk's incoming state
///     gradient is not zero — that chunk feeds the one to its right — and zeroing it
///     gives a wrong gradient with a right-looking loss.
///     `mlstm_chunked_backward_matches_whole` pins it.
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
/// A single full-length span when `chunk >= n`, so the unchunked sweep is the chunked
/// code with one chunk rather than a second path.
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
fn slice_rows(gpu: &Gpu, src: &GTensor<f32>, c0: usize, len: usize, width: usize) -> GTensor<f32> {
    let mut out = GTensor::uninit(gpu, &[len, width]);
    gpu.stream
        .memcpy_dtod(
            &src.buf.slice(c0 * width..(c0 + len) * width),
            &mut out.buf.slice_mut(..len * width),
        )
        .expect("slice_rows");
    out
}

/// Concatenate row-blocks back into one `[rows, width]` tensor, in order.
fn concat_rows(gpu: &Gpu, parts: &[GTensor<f32>], rows: usize, width: usize) -> GTensor<f32> {
    let mut out = GTensor::uninit(gpu, &[rows, width]);
    let mut off = 0;
    for p in parts {
        let n = p.len();
        gpu.stream
            .memcpy_dtod(&p.buf.slice(..n), &mut out.buf.slice_mut(off..off + n))
            .expect("concat_rows");
        off += n;
    }
    debug_assert_eq!(
        off,
        rows * width,
        "concat_rows: parts do not tile the output"
    );
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

/// Refill `sc`'s per-group decoder index buffers for group `g` (see [`Scratch`]).
///
/// Six lists, all addressing the group's `[n_g, tmax]` rectangle: word slot per group
/// row (`grp_ids`), the row holding each word's injected context (`o_rows`), the rows
/// fed by a previous char and the char feeding them (`char_rows` / `char_ids`), and the
/// CE `targets` / `mask`. A masked (prompt) word is still fed forward so its state and
/// the tied-table char grads are produced, but its slots get CE mask 0 — no loss, no
/// logit gradient from a prompt token.
fn fill_decoder_group_ids(
    tokens: &[usize],
    words: &[Range<usize>],
    g: usize,
    tmax: usize,
    rows: usize,
    w_token: usize,
    word_on: &impl Fn(usize) -> bool,
    sc: &mut Scratch,
) {
    sc.o_rows.clear();
    sc.char_rows.clear();
    sc.char_ids.clear();
    sc.targets.clear();
    sc.targets.resize(rows, 0);
    sc.mask.clear();
    sc.mask.resize(rows, 0);
    sc.grp_ids.clear();
    for (i, &w) in sc.dec_groups[g].iter().enumerate() {
        let m = sc.dec_lens[w];
        let s = words[w + 1].start;
        let on = word_on(w) as u32;
        sc.grp_ids.push(w as u32);
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
}

/// One encoder group's `[words, tmax]` id rectangle and `[W]`-step readout rows.
#[derive(Default)]
struct EncGroup {
    ids: Vec<u32>,
    readout: Vec<u32>,
    tmax: usize,
}

/// Per-phase stopwatch for a window, silent unless `GPU_PROF` or `GPU_MEM` is set.
///
/// Each [`mark`](PhaseTimer::mark) syncs the stream, so the phases are measured
/// rather than merely enqueued — which also means it must stay off in a real run.
struct PhaseTimer {
    prof: bool,
    mem: bool,
    t0: Instant,
}

impl PhaseTimer {
    fn new(flags: &Flags) -> Self {
        Self {
            prof: flags.prof,
            mem: flags.mem,
            t0: Instant::now(),
        }
    }

    /// Restart the clock at the top of a window.
    fn reset(&mut self) {
        self.t0 = Instant::now();
    }

    fn mark(&mut self, gpu: &Gpu, name: &str) {
        if !(self.prof || self.mem) {
            return;
        }
        gpu.stream.synchronize().expect("sync");
        let mut line = format!("  {name:<22} {:>8.1?}", self.t0.elapsed());
        if self.mem {
            let (free, total) = cudarc::driver::result::mem_get_info().expect("mem_get_info");
            // `driver` is what `mem_get_info` reports — every block the CUDA async
            // allocator holds, including what it merely CACHES for reuse. `live` is what
            // is actually allocated and what decides whether the next allocation OOMs. A
            // climbing `driver` with a flat `live` is allocator cache, not model growth.
            line.push_str(&format!(
                "  |  driver {:>6.0} MB  live {:>6.0} MB",
                (total - free) as f64 / 1e6,
                super::pool_used_mb().unwrap_or(f64::NAN)
            ));
        }
        println!("{line}");
        self.t0 = Instant::now();
    }
}

/// The backbone forward's outputs, handed to the decoder and then to the backward.
struct BackboneFwd {
    /// `[dw, HC]` context, one row per word — the decoder's slot-0 injection.
    o: GTensor<f32>,
    /// Chunk spans over the word axis; a single full-length span when unchunked.
    spans: Vec<(usize, usize)>,
    /// Each chunk's input to `bb_back`, kept because `forward_shared` saves nothing
    /// and the backward needs every chunk's `X` for `dW = XᵀdY`.
    back_in: Vec<GTensor<f32>>,
    /// Rows of the largest chunk — what the backbone's pool has to hold.
    rows_max: usize,
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
    let tmax = grp.iter().map(|&w| enc_lens[w]).max().unwrap() + 1;
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
    pub cfg: ModelCfg,

    // Tied char table (encoder input + decoder char slots) + grad/moments.
    pub table: GTensor<f32>,
    dtable: GTensor<f32>,
    m_tbl: GTensor<f32>,
    v_tbl: GTensor<f32>,

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

    /// Every parameter, gradient and AdamW moment in four contiguous allocations,
    /// with each layer above holding windows into them, so every parameter has a
    /// fixed device address. `None` only while a constructor is still assembling the
    /// stack — see [`bind_params`](Self::bind_params) and [`super::arena`].
    arena: Option<ParamArena>,

    /// Optimizer step count, persisted with the checkpoint so training resumes.
    pub step_count: usize,
    /// Cumulative chars/words this model has trained on (see `format::Seen`).
    /// Advanced by the training loops, persisted in the checkpoint.
    pub seen: Seen,

    /// Debug/benchmark switches, read from the environment once at construction
    /// rather than per window — `std::env::var` takes a process-wide lock and
    /// allocates, and a window would otherwise pay for four of them.
    flags: Flags,

    /// Per-phase wall-clock timer, reset at the top of every window.
    timer: PhaseTimer,
    /// Reused host-side index buffers (see [`Scratch`]).
    scratch: Scratch,
    /// Forward activations of the current window, owned by the model so its buffers
    /// survive across windows instead of being rebuilt every step (see [`Scratch`]).
    cache: TrainingCache,
    /// Staging for the per-group index lists, so each group's ids go up in one
    /// pinned async transfer rather than one blocking transfer per list.
    ids: ops::IdBatch,
    /// Device accumulator for the decode loss, summed across the window's groups and
    /// read back once at the end of the step.
    ///
    /// Reading each group's loss where it is produced means a blocking `clone_dtoh`
    /// inside the decode loop — it drains the stream and page-locks a staging buffer,
    /// per group, for a number only used in logging.
    loss_acc: Option<CudaSlice<f32>>,
    /// Mean NLL per decoded word of the last window — the same quantity the CPU
    /// path logs as `word_loss`. Derived from the window's row loss on the host,
    /// so it costs no extra device work.
    last_word_loss: f32,
    /// Decoder rows the last window's loss was averaged over (chars plus one `[W]`
    /// per scored word). The divisor behind `last_word_loss`, exposed so a caller
    /// can recover the window's total NLL under SFT masking too.
    last_rows: usize,
}

/// Bit-exact checksum of a device tensor, printed when `GPU_HASH` is set.
///
/// Hashes the raw bits, not the values: a one-ULP difference between two runs of
/// the same window has to show up, which is the whole point when localizing
/// nondeterminism to a phase. Blocking D2H — debug only.
fn hash_dbg(gpu: &Gpu, name: &str, t: &GTensor<f32>) {
    if std::env::var("GPU_HASH").is_err() {
        return;
    }
    let h = t.to_host(gpu);
    let mut acc = 0xcbf29ce484222325u64;
    for v in h.data.iter() {
        acc = (acc ^ v.to_bits() as u64).wrapping_mul(0x100000001b3);
    }
    let sum: f64 = h.data.iter().map(|&v| v as f64).sum();
    println!(
        "  #{name:<10} {acc:#018x}  n={:<8} sum={sum:+.9e}",
        h.data.len()
    );
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
    pub fn new(gpu: &Gpu, cfg: ModelCfg) -> Self {
        let mut model = Self::unbound(gpu, cfg);
        model.bind_params(gpu);
        model
    }

    /// The model with its parameters still in per-tensor allocations.
    ///
    /// Only [`load`](Self::load) uses this directly: it swaps whole layers in after
    /// construction, so it must pack the arena once, afterwards.
    fn unbound(gpu: &Gpu, cfg: ModelCfg) -> Self {
        let bb_blocks: Vec<Box<dyn BlockLike>> = (0..cfg.bb_blocks)
            .map(|i| {
                if i.is_multiple_of(8) {
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
                if i.is_multiple_of(2) {
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
        let flags = Flags::from_env();
        let mut model = Self {
            cfg,
            table: GTensor::from_host(gpu, &Tensor::random(&[cfg.vocab, cfg.hc], 0.02)),
            dtable: GTensor::zeros(gpu, &[cfg.vocab, cfg.hc]),
            m_tbl: GTensor::zeros(gpu, &[cfg.vocab, cfg.hc]),
            v_tbl: GTensor::zeros(gpu, &[cfg.vocab, cfg.hc]),
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
                head_optimizer_convention(&mut h);
                h
            },
            arena: None,
            step_count: 0,
            seen: Seen::default(),
            timer: PhaseTimer::new(&flags),
            flags,
            scratch: Scratch::default(),
            cache: TrainingCache::default(),
            ids: ops::IdBatch::new(),
            loss_acc: None,
            last_word_loss: 0.0,
            last_rows: 0,
        };
        model.enable_backbone_offload(gpu);
        model
    }

    /// Pack every parameter into one [`ParamArena`], leaving the layers holding
    /// windows into it.
    ///
    /// Called from the constructors, before a forward has allocated anything: packing
    /// holds one arena buffer alongside the tensors it is replacing, and doing it here
    /// keeps that transient off the peak instead of stacking it on a window's
    /// activations.
    fn bind_params(&mut self, gpu: &Gpu) {
        let arena = ParamArena::bind(gpu, self.param_slots());
        self.arena = Some(arena);
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

    /// Decoder rows the last window's returned loss was averaged over. Multiplying
    /// the two recovers the window's total NLL in nats.
    pub fn last_rows(&self) -> usize {
        self.last_rows
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

    /// The five phases of a window, in order: encode the words, sweep the backbone,
    /// decode (forward *and* backward) per length group, unwind the backbone, unwind
    /// the encoder. Each phase is a method below; this is the order and the plumbing.
    fn forward_backward_window(
        &mut self,
        gpu: &Gpu,
        tokens: &[usize],
        words: &[Range<usize>],
        word_loss: Option<&[bool]>,
    ) -> f32 {
        if words.len() < 2 {
            self.last_word_loss = 0.0;
            self.last_rows = 0;
            return 0.0;
        }

        let dw = words.len() - 1; // decoded words: word 0 is encode-only
        self.timer.reset();
        if self.flags.mem {
            self.log_window_shape(tokens, words);
        }

        let mut sc = mem::take(&mut self.scratch);
        let mut cache = mem::take(&mut self.cache);

        let (e_w, enc_rows) = self.encoder_forward(gpu, tokens, words, &mut sc, &mut cache);
        let bb = self.backbone_forward(gpu, &e_w, dw, &mut cache);

        let bb_rows_max = bb.rows_max;
        let (loss, d_o, dec_rows_max) =
            self.decode_groups(gpu, tokens, words, word_loss, &bb.o, &mut sc, &mut cache);

        let d_e_w = self.backbone_backward(gpu, bb, &d_o, dw);
        self.encoder_backward(gpu, &d_e_w, &mut sc, &mut cache);

        self.scratch = sc;
        self.cache = cache;

        // Release pooled scratch far larger than this window needed.
        self.trim_pools(bb_rows_max, enc_rows, dec_rows_max);
        loss
    }

    /// Print the window's actual shape under `GPU_MEM`.
    ///
    /// A synthetic probe picks its own word-length spread and can easily miss the one
    /// real text produces — and since the encoder/decoder run one rectangle per length
    /// BUCKET, the histogram matters as much as the totals.
    fn log_window_shape(&self, tokens: &[usize], words: &[Range<usize>]) {
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
            "  window: {} words, {} tokens, longest {widest}, {buckets} buckets  \
             | pool {sizes} sizes / {bufs} bufs",
            words.len(),
            tokens.len()
        );
    }

    /// Group the words by length and fill `sc.enc_layout` with each group's id
    /// rectangle, returning the group count.
    ///
    /// Words are batched as `[words, tmax]` rectangles, and `tmax` is set by the
    /// LONGEST word — so one 16-byte word would pad every 2-byte word out to 17 steps
    /// (~4.5x wasted rows on Rust source, in both FLOPs and VRAM). One dense rectangle
    /// per length group instead: the padding collapses to within-group slack, and each
    /// group is still a clean rectangle, which the mLSTM's per-word `[T, T]` attention
    /// requires.
    ///
    /// The rectangles are built once here, not once per direction: the encoder BACKWARD
    /// re-forwards each group (activation checkpointing) and needs byte-identical ids.
    fn plan_encoder_groups(
        &self,
        tokens: &[usize],
        words: &[Range<usize>],
        sc: &mut Scratch,
    ) -> usize {
        let dw = words.len() - 1;
        sc.enc_lens.clear();
        sc.enc_lens
            .extend((0..dw).map(|w| words[w].end - words[w].start));
        group_by_len(
            &sc.enc_lens,
            self.flags.no_group,
            self.group_cap(),
            &mut sc.enc_groups,
        );

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
                self.cfg.w_token,
                &mut sc.enc_layout[g],
            );
        }
        n_groups
    }

    fn group_cap(&self) -> usize {
        static ENV: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
        let env = *ENV.get_or_init(|| {
            std::env::var("GPU_GROUP_CAP")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
        });
        self.group_cap
            .or(env)
            .unwrap_or(crate::config::GROUP_MAX_ROWS)
    }

    /// Run one encoder group's rectangle through the block stack, leaving the result
    /// in the returned `[n_g * tmax, HC]` tensor. The ids must already be uploaded:
    /// slot 0 of `self.ids` is the group's id rectangle.
    fn encoder_stack_forward(
        &mut self,
        gpu: &Gpu,
        n_g: usize,
        tmax: usize,
        ids: usize,
        cache: &mut TrainingCache,
    ) -> GTensor<f32> {
        let hc = self.cfg.hc;
        let embedded = ops::embedding_gather_u32(gpu, &self.table, &self.ids.get(0), ids, hc);
        let mut h = embedded.reshaped(&[n_g, tmax, hc]);
        // Blocks are H-in == H-out, so one spare buffer ping-pongs the stack.
        let mut next = GTensor::uninit(gpu, &[n_g, tmax, hc]);
        for blk in self.encoder.blocks.iter_mut() {
            blk.forward(gpu, &h, &mut next, cache);
            mem::swap(&mut h, &mut next);
        }
        h.reshaped(&[n_g * tmax, hc])
    }

    /// PHASE 1 — encode every word to its `e_w`, one dense rectangle per length group.
    ///
    /// Returns `e_w` (`[dw, HC]`, one row per word) and the largest group's row count.
    /// The latter bounds the encoder's pool: groups run one after another and a group's
    /// activations die before the next starts, so only one group is ever resident — but
    /// each group has its own `[n_g, tmax]` shape and every `Buf`/`Linear::x` inside the
    /// stack resizes to whatever it is handed, so reuse is by capacity and the largest
    /// shape is what the pool has to hold. A block runs its recurrence along `T`, so
    /// `tmax` is fixed by the bucket and only `n_g` could be padded — which is why this
    /// is a bound for the trim, not a uniform shape.
    fn encoder_forward(
        &mut self,
        gpu: &Gpu,
        tokens: &[usize],
        words: &[Range<usize>],
        sc: &mut Scratch,
        cache: &mut TrainingCache,
    ) -> (GTensor<f32>, usize) {
        let hc = self.cfg.hc;
        let dw = words.len() - 1;
        let n_groups = self.plan_encoder_groups(tokens, words, sc);
        let enc_rows = (0..n_groups)
            .map(|g| sc.enc_groups[g].len() * sc.enc_layout[g].tmax)
            .max()
            .unwrap_or(0);

        let mut e_w = GTensor::zeros(gpu, &[dw, hc]);
        for g in 0..n_groups {
            let (n_g, tmax, ids, readout) = {
                let lay = &sc.enc_layout[g];
                // Both id lists in one pinned async upload instead of two blocking ones.
                self.ids.upload(gpu, &[&lay.ids, &lay.readout]);
                (
                    sc.enc_groups[g].len(),
                    lay.tmax,
                    lay.ids.len(),
                    lay.readout.len(),
                )
            };
            let h_flat = self.encoder_stack_forward(gpu, n_g, tmax, ids, cache);
            // e_w = each word's [W]-step row, scattered back to its window slot.
            let e_w_grp = ops::embedding_gather_u32(gpu, &h_flat, &self.ids.get(1), readout, hc);
            ops::scatter_rows(gpu, &mut e_w, &e_w_grp, &sc.enc_groups[g]);
            // This group's forward cache is dead: the encoder backward re-forwards each
            // group to rebuild it (activation checkpointing), so nothing will ever read
            // what was just saved.
            //
            // `drop_all_act` rather than `drop_saved_act`: the latter leaves each
            // projection's saved input, its bf16 GEMM staging and the norms' `x̂`
            // resident. Those are sized to this group's rectangle, the next group's is a
            // different shape and reuse is by capacity, so every group would leave a
            // full set behind — 1259 MB for a stage whose largest rectangle is ~25 MB.
            for blk in self.encoder.blocks.iter_mut() {
                blk.drop_all_act(gpu);
            }
        }
        self.timer.mark(gpu, "encoder fwd");
        hash_dbg(gpu, "e_w", &e_w);
        (e_w, enc_rows)
    }

    /// The word axis split into the spans the backbone sweeps, and whether that is
    /// more than one. `chunk_spans(dw, dw)` is the unchunked sweep, so both modes run
    /// the same code.
    fn backbone_spans(&self, dw: usize) -> Vec<(usize, usize)> {
        let chunk = match self.bb_chunk {
            Some(c) => c.min(dw).max(1),
            None if BACKBONE_CHUNKED_BACKWARD => backbone_chunk(dw),
            None => dw,
        };
        chunk_spans(dw, chunk)
    }

    /// PHASE 2 — autoregress the backbone over the word embeddings.
    ///
    /// Chunked over the word axis when `BACKBONE_CHUNK` is set. The backbone holds one
    /// row per word in every block, from that block's forward until its backward, so an
    /// unchunked sweep is O(words) resident per block and device memory scales with the
    /// window — the reason long windows OOM.
    ///
    /// Chunk-major instead: chunk c passes through all blocks with each cell's recurrent
    /// state carried from chunk c-1, so the arithmetic is identical to the unchunked
    /// sweep (pinned by `chunked_carry_matches_unchunked` and
    /// `mlstm_chunked_carry_matches_whole`) while only the chunks in flight are
    /// resident.
    fn backbone_forward(
        &mut self,
        gpu: &Gpu,
        e_w: &GTensor<f32>,
        dw: usize,
        cache: &mut TrainingCache,
    ) -> BackboneFwd {
        let (hc, wh) = (self.cfg.hc, self.cfg.wh);
        let bb_in = self.bb_front.forward_alloc(gpu, e_w); // [dw, WH]
        let spans = self.backbone_spans(dw);
        let rows_max = spans.iter().map(|&(_, len)| len).max().unwrap_or(0);

        // Runs for every window, single-chunk ones included: a window that fits in one
        // chunk is an unchunked sweep, and without the reset it would resume the
        // previous window's state and BPTT gradient across a document border.
        let carry = spans.len() > 1;
        for blk in self.bb_blocks.iter_mut() {
            blk.set_carry(carry);
            blk.reset_state(gpu);
        }
        // `bb_back`'s input per chunk. `Linear::forward` saves its input for `dW` and a
        // later chunk's forward overwrites it, so each chunk's is kept here and handed
        // back through `backward_with_x` — otherwise `dW` accumulates from the last
        // chunk only, a silent wrong gradient rather than a crash.
        let mut back_in: Vec<GTensor<f32>> = Vec::with_capacity(spans.len());
        let mut o_parts: Vec<GTensor<f32>> = Vec::with_capacity(spans.len());
        for &(c0, len) in &spans {
            // This chunk's slice of the front projection's output, as its own tensor:
            // the blocks write through their own buffers and a chunk's activations must
            // not alias the whole-window one.
            let mut hb = slice_rows(gpu, &bb_in, c0, len, wh).reshaped(&[1, len, wh]);
            let mut hb_next = GTensor::uninit(gpu, &[1, len, wh]);
            for blk in self.bb_blocks.iter_mut() {
                blk.forward(gpu, &hb, &mut hb_next, cache);
                mem::swap(&mut hb, &mut hb_next);
            }
            let flat = hb.reshaped(&[len, wh]);
            let mut y = GTensor::uninit(gpu, &[len, hc]);
            self.bb_back.forward_shared(gpu, &flat, &mut y);
            back_in.push(flat);
            o_parts.push(y);
        }
        let o = if o_parts.len() == 1 {
            o_parts.pop().expect("one chunk")
        } else {
            concat_rows(gpu, &o_parts, dw, hc)
        };
        self.timer.mark(gpu, "backbone fwd");
        hash_dbg(gpu, "bb_in", &bb_in);
        hash_dbg(gpu, "o", &o);
        BackboneFwd {
            o,
            spans,
            back_in,
            rows_max,
        }
    }

    /// PHASE 3 — decode every word, forward and straight back again, per length group.
    ///
    /// Word w's decode target is word w+1, so groups are keyed on the length of the
    /// DECODED word. The decoder's backward needs nothing from the backbone's, so a
    /// group's activations die before the next group allocates — only one group's worth
    /// of decoder rows (and of the `[rows, vocab]` logits) is ever resident.
    ///
    /// Returns the window's mean CE, `d_o` (`[dw, HC]`, the gradient of the injected
    /// context) and the largest group's row count for the pool trim.
    fn decode_groups(
        &mut self,
        gpu: &Gpu,
        tokens: &[usize],
        words: &[Range<usize>],
        word_loss: Option<&[bool]>,
        o: &GTensor<f32>,
        sc: &mut Scratch,
        cache: &mut TrainingCache,
    ) -> (f32, GTensor<f32>, usize) {
        let hc = self.cfg.hc;
        let dw = words.len() - 1;
        sc.dec_lens.clear();
        sc.dec_lens
            .extend((0..dw).map(|w| words[w + 1].end - words[w + 1].start));
        group_by_len(
            &sc.dec_lens,
            self.flags.no_group,
            self.group_cap(),
            &mut sc.dec_groups,
        );

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
        gpu.stream.memset_zeros(&mut acc).expect("zero loss_acc");

        let mut d_o = GTensor::zeros(gpu, &[dw, hc]);
        let mut dec_rows_max = 0usize;
        for g in 0..sc.dec_groups.len() {
            let n_g = sc.dec_groups[g].len();
            let tmax = sc.dec_groups[g]
                .iter()
                .map(|&w| sc.dec_lens[w] + 1)
                .max()
                .unwrap();
            let rows = n_g * tmax;
            dec_rows_max = dec_rows_max.max(rows);
            fill_decoder_group_ids(tokens, words, g, tmax, rows, self.cfg.w_token, &word_on, sc);

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
            let n_chars = sc.char_rows.len();
            let dec_in = self.build_decoder_input(gpu, o, n_g, rows, n_chars);
            let capped = self.decoder_stack_forward(gpu, dec_in, n_g, tmax, rows, cache);

            let (_, d_capped) = ops::masked_softmax_ce_u32_into(
                gpu,
                &capped,
                &self.ids.get(4),
                &self.ids.get(5),
                inv,
                Some(&mut acc),
            );

            let d_dec_in = self.decoder_stack_backward(gpu, &d_capped, &capped, n_g, tmax, rows);
            self.scatter_decoder_grads(gpu, &d_dec_in, &mut d_o, n_g, n_chars);

            // This group is completely finished — forward AND backward — so nothing
            // here will be read again. Release the whole activation set rather than
            // letting the next group's differently-shaped rectangle reallocate around
            // it: same reasoning as the encoder, and the decoder is the same size
            // (1225 MB measured for a ~25 MB working set).
            //
            // Safe here in a way it would not be mid-loop: the decoder's backward has
            // already consumed everything its forward produced.
            for blk in self.dec_blocks.iter_mut() {
                blk.drop_all_act(gpu);
            }
        }
        // One readback for the whole window, outside the group loop. Blocking here is
        // fine — the decode phase is over — where per-group it stalled the pipeline.
        let loss = gpu.stream.clone_dtoh(&acc).expect("download loss_acc")[0];
        self.loss_acc = Some(acc);
        // Per-word NLL from the same total: `loss` is the row sum over `valid_rows`,
        // so scaling by rows/words re-averages it over words instead. Both counts are
        // already on the host, so the second metric costs no device work.
        let scored_words = (0..dw).filter(|&w| word_on(w)).count();
        self.last_word_loss = loss * (valid_rows as f32) / (scored_words.max(1) as f32);
        self.last_rows = valid_rows;
        self.timer.mark(gpu, "decoder fwd + bwd");
        (loss, d_o, dec_rows_max)
    }

    /// The decoder's `[rows, HC]` input: zeros, then the backbone context scattered
    /// into each word's slot 0 and the previous char into every later slot.
    fn build_decoder_input(
        &mut self,
        gpu: &Gpu,
        o: &GTensor<f32>,
        n_g: usize,
        rows: usize,
        n_chars: usize,
    ) -> GTensor<f32> {
        let hc = self.cfg.hc;
        let o_grp = ops::embedding_gather_u32(gpu, o, &self.ids.get(0), n_g, hc);
        let mut dec_in = GTensor::zeros(gpu, &[rows, hc]);
        ops::scatter_rows_u32(gpu, &mut dec_in, &o_grp, &self.ids.get(1));
        let char_vecs = ops::embedding_gather_u32(gpu, &self.table, &self.ids.get(3), n_chars, hc);
        ops::scatter_rows_u32(gpu, &mut dec_in, &char_vecs, &self.ids.get(2));
        dec_in
    }

    /// Decoder blocks → norm → head → softcap, returning the capped `[rows, vocab]`
    /// logits (kept: the softcap backward needs its own output).
    fn decoder_stack_forward(
        &mut self,
        gpu: &Gpu,
        dec_in: GTensor<f32>,
        n_g: usize,
        tmax: usize,
        rows: usize,
        cache: &mut TrainingCache,
    ) -> GTensor<f32> {
        let hc = self.cfg.hc;
        let mut hd = dec_in.reshaped(&[n_g, tmax, hc]);
        let mut hd_next = GTensor::uninit(gpu, &[n_g, tmax, hc]);
        for blk in self.dec_blocks.iter_mut() {
            blk.forward(gpu, &hd, &mut hd_next, cache);
            mem::swap(&mut hd, &mut hd_next);
        }
        let hdn = self.dec_norm.forward_alloc(gpu, &hd.reshaped(&[rows, hc]));
        let logits = self.dec_head.forward_alloc(gpu, &hdn);
        ops::softcap_forward(gpu, &logits, self.cfg.cap)
    }

    /// The mirror of [`decoder_stack_forward`](Self::decoder_stack_forward), returning
    /// the gradient of the decoder's `[rows, HC]` input.
    fn decoder_stack_backward(
        &mut self,
        gpu: &Gpu,
        d_capped: &GTensor<f32>,
        capped: &GTensor<f32>,
        n_g: usize,
        tmax: usize,
        rows: usize,
    ) -> GTensor<f32> {
        let hc = self.cfg.hc;
        let d_logits = ops::softcap_backward(gpu, d_capped, capped, self.cfg.cap);
        let d_hdn = self.dec_head.backward_alloc(gpu, &d_logits);
        let d_hd_flat = self.dec_norm.backward_alloc(gpu, &d_hdn);
        let mut d_hd = d_hd_flat.reshaped(&[n_g, tmax, hc]);
        let mut d_hd_next = GTensor::uninit(gpu, &[n_g, tmax, hc]);
        for blk in self.dec_blocks.iter_mut().rev() {
            blk.backward(gpu, &d_hd, &mut d_hd_next);
            mem::swap(&mut d_hd, &mut d_hd_next);
        }
        d_hd.reshaped(&[rows, hc])
    }

    /// Route the decoder input's gradient to its two sources: slot-0 rows to `d_o`
    /// (the backbone context), char-slot rows to the tied table.
    fn scatter_decoder_grads(
        &mut self,
        gpu: &Gpu,
        d_dec_in: &GTensor<f32>,
        d_o: &mut GTensor<f32>,
        n_g: usize,
        n_chars: usize,
    ) {
        let hc = self.cfg.hc;
        let d_o_grp = ops::embedding_gather_u32(gpu, d_dec_in, &self.ids.get(1), n_g, hc);
        ops::scatter_rows_u32(gpu, d_o, &d_o_grp, &self.ids.get(0));
        let d_char = ops::embedding_gather_u32(gpu, d_dec_in, &self.ids.get(2), n_chars, hc);
        ops::embedding_scatter_add_u32(
            gpu,
            &mut self.dtable,
            &self.ids.get(3),
            n_chars,
            &d_char,
            hc,
        );
    }

    /// PHASE 4 — unwind the backbone, returning `d_e_w` for the encoder backward.
    ///
    /// Chunks unwind right to left, the mirror of the forward's left-to-right, so the
    /// BPTT state each cell carries flows backwards across the same borders the forward
    /// state crossed forwards. Every chunk passes through all blocks before the next one
    /// starts, which is what lets a chunk's activations be released as soon as it is
    /// unwound.
    fn backbone_backward(
        &mut self,
        gpu: &Gpu,
        bb: BackboneFwd,
        d_o: &GTensor<f32>,
        dw: usize,
    ) -> GTensor<f32> {
        let wh = self.cfg.wh;
        let BackboneFwd { spans, back_in, .. } = bb;
        let chunked = spans.len() > 1;

        // The last block's activations are wanted first, and nothing inside the loop
        // precedes them — so their upload is issued here, ahead of `bb_back`'s
        // backward, and overlaps it.
        if let Some(last) = self.bb_blocks.last_mut() {
            last.prefetch_act(gpu);
        }
        let d_bb_out = self.bb_back_backward(gpu, &back_in, &spans, d_o, dw);

        let mut d_parts: Vec<Option<GTensor<f32>>> = (0..spans.len()).map(|_| None).collect();
        let mut d_bb_out = Some(d_bb_out);
        for (ci, &(c0, len)) in spans.iter().enumerate().rev() {
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
            let d_hb = if chunked {
                let src = d_bb_out.as_ref().expect("d_bb_out");
                slice_rows(gpu, src, c0, len, wh).reshaped(&[1, len, wh])
            } else {
                // Single chunk: hand the whole tensor over rather than copying it.
                d_bb_out.take().expect("d_bb_out").reshaped(&[1, len, wh])
            };
            d_parts[ci] = Some(self.bb_blocks_backward(gpu, d_hb, len));
        }
        let mut d_parts: Vec<GTensor<f32>> = d_parts
            .into_iter()
            .map(|p| p.expect("every chunk unwound"))
            .collect();
        let d_hb_all = if d_parts.len() == 1 {
            d_parts.pop().expect("one chunk")
        } else {
            concat_rows(gpu, &d_parts, dw, wh)
        };
        let d_e_w = self.bb_front.backward_alloc(gpu, &d_hb_all); // [dw, HC]
        self.timer.mark(gpu, "backbone bwd");
        d_e_w
    }

    /// `bb_back`'s backward over all chunks, returning the `[dw, WH]` gradient at the
    /// last block's output.
    ///
    /// `bb_back` ran `forward_shared` per chunk, so it saved nothing — its input comes
    /// back through `back_in`. Feeding the chunks' inputs in the same order makes
    /// `dW = XᵀdY` accumulate over all of them (beta = 1), which is what makes the
    /// chunked gradient equal the whole-window one.
    fn bb_back_backward(
        &mut self,
        gpu: &Gpu,
        back_in: &[GTensor<f32>],
        spans: &[(usize, usize)],
        d_o: &GTensor<f32>,
        dw: usize,
    ) -> GTensor<f32> {
        if back_in.len() == 1 {
            return self.bb_back.backward_alloc_with_x(gpu, &back_in[0], d_o);
        }
        let (hc, wh) = (self.cfg.hc, self.cfg.wh);
        let mut parts: Vec<GTensor<f32>> = Vec::with_capacity(back_in.len());
        for (i, &(c0, len)) in spans.iter().enumerate() {
            let d_o_c = slice_rows(gpu, d_o, c0, len, hc);
            parts.push(self.bb_back.backward_alloc_with_x(gpu, &back_in[i], &d_o_c));
        }
        concat_rows(gpu, &parts, dw, wh)
    }

    /// One chunk's `[1, len, WH]` gradient down through the backbone blocks.
    ///
    /// With offload on, each block's saved activations have to come back from the host
    /// first — so block i-1's upload is started *before* block i's backward runs, giving
    /// it a whole block of compute to hide behind. Issuing the copy and waiting for it
    /// in the same breath exposes the whole transfer (measured: +37 ms).
    fn bb_blocks_backward(&mut self, gpu: &Gpu, d_hb: GTensor<f32>, len: usize) -> GTensor<f32> {
        let wh = self.cfg.wh;
        let mut d_hb = d_hb;
        let mut d_hb_next = GTensor::uninit(gpu, &[1, len, wh]);
        for i in (0..self.bb_blocks.len()).rev() {
            if i > 0 {
                let (head, tail) = self.bb_blocks.split_at_mut(i);
                head[i - 1].prefetch_act(gpu);
                tail[0].backward(gpu, &d_hb, &mut d_hb_next);
            } else {
                self.bb_blocks[0].backward(gpu, &d_hb, &mut d_hb_next);
            }
            mem::swap(&mut d_hb, &mut d_hb_next);
        }
        d_hb.reshaped(&[len, wh])
    }

    /// PHASE 5 — unwind the encoder, one group at a time, re-forwarding each.
    ///
    /// Each encoder group's forward cache was overwritten by the group after it (and by
    /// the other direction), so re-run that group's forward to refill it, then backward
    /// immediately. Forward is deterministic, so this reproduces the exact activations —
    /// it is activation checkpointing, and it keeps just one group resident. The cost is
    /// one extra encoder forward, over the SMALL grouped rectangles.
    fn encoder_backward(
        &mut self,
        gpu: &Gpu,
        d_e_w: &GTensor<f32>,
        sc: &mut Scratch,
        cache: &mut TrainingCache,
    ) {
        let hc = self.cfg.hc;
        for g in 0..sc.enc_groups.len() {
            let (n_g, tmax, ids) = {
                let lay = &sc.enc_layout[g];
                // This group's three id lists in one pinned async upload.
                sc.grp_ids.clear();
                sc.grp_ids
                    .extend(sc.enc_groups[g].iter().map(|&w| w as u32));
                self.ids.upload(gpu, &[&lay.ids, &lay.readout, &sc.grp_ids]);
                (sc.enc_groups[g].len(), lay.tmax, lay.ids.len())
            };
            drop(self.encoder_stack_forward(gpu, n_g, tmax, ids, cache));

            // Scatter this group's d_e_w onto its [W]-step rows, rest zero.
            let d_e_w_grp = ops::embedding_gather_u32(gpu, d_e_w, &self.ids.get(2), n_g, hc);
            let mut d_h = GTensor::zeros(gpu, &[n_g * tmax, hc]);
            ops::scatter_rows_u32(gpu, &mut d_h, &d_e_w_grp, &self.ids.get(1));
            let mut d_h = d_h.reshaped(&[n_g, tmax, hc]);
            let mut d_h_next = GTensor::uninit(gpu, &[n_g, tmax, hc]);
            for blk in self.encoder.blocks.iter_mut().rev() {
                blk.backward(gpu, &d_h, &mut d_h_next);
                mem::swap(&mut d_h, &mut d_h_next);
            }
            let d_embedded = d_h.reshaped(&[n_g * tmax, hc]);
            ops::embedding_scatter_add_u32(
                gpu,
                &mut self.dtable,
                &self.ids.get(0),
                ids,
                &d_embedded,
                hc,
            );
            for blk in self.encoder.blocks.iter_mut() {
                blk.drop_all_act(gpu);
            }
        }
        self.timer.mark(gpu, "encoder bwd");
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

    /// A bit-hash of every accumulated gradient, named by owner.
    ///
    /// Diagnostic for reproducibility: two runs of the same window against the same
    /// weights must produce the same list, and when they do not, this says which
    /// tensor moved. One host round-trip per tensor, so it belongs in a probe.
    pub fn grad_signature(&mut self, gpu: &Gpu) -> Vec<(String, u64)> {
        fn hash(v: &[f32]) -> u64 {
            v.iter().fold(0xcbf29ce484222325u64, |h, x| {
                (h ^ x.to_bits() as u64).wrapping_mul(0x100000001b3)
            })
        }
        self.grad_values(gpu)
            .into_iter()
            .map(|(name, v)| (name, hash(&v)))
            .collect()
    }

    /// Every accumulated gradient as host values, named by owner, in the same order
    /// as [`grad_signature`](Self::grad_signature).
    ///
    /// Diagnostic: unlike the hashes this can be summed and compared numerically,
    /// which is what a gradient-coverage check needs. One host round-trip per
    /// tensor, so it belongs in a probe.
    pub fn grad_values(&mut self, gpu: &Gpu) -> Vec<(String, Vec<f32>)> {
        let mut out: Vec<(String, Vec<f32>)> = Vec::new();
        let mut push =
            |name: String, g: &GTensor<f32>| out.push((name, g.to_host(gpu).data.to_vec()));
        push("table".into(), &self.dtable);
        for (stage, blocks) in [
            ("enc", &mut self.encoder.blocks),
            ("bb", &mut self.bb_blocks),
            ("dec", &mut self.dec_blocks),
        ] {
            for (i, b) in blocks.iter_mut().enumerate() {
                for (j, g) in b.grads().iter().enumerate() {
                    push(format!("{stage}{i}.g{j}"), g);
                }
            }
        }
        for (name, l) in [
            ("bb_front", &self.bb_front),
            ("bb_back", &self.bb_back),
            ("dec_head", &self.dec_head),
        ] {
            push(format!("{name}.dw"), &l.dw);
            push(format!("{name}.db"), &l.db);
        }
        push("dec_norm.dgamma".into(), &self.dec_norm.dgamma);
        out
    }

    /// L2 norm of each block's accumulated gradient, for one stage ("encoder",
    /// "backbone", "decoder"). Diagnostic — one host round-trip per tensor, so this
    /// belongs in a probe, not a training loop.
    pub fn grad_norms_by_block(&mut self, gpu: &Gpu, stage: &str) -> Vec<f32> {
        let blocks = match stage {
            "encoder" => &mut self.encoder.blocks,
            "backbone" => &mut self.bb_blocks,
            "decoder" => &mut self.dec_blocks,
            other => panic!("grad_norms_by_block: unknown stage {other}"),
        };
        blocks
            .iter_mut()
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
    pub fn state_extremes_by_block(&self, gpu: &Gpu, stage: &str) -> Vec<Option<(f32, f32, f32)>> {
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

    /// Learnable parameter counts as `(encoder, backbone, decoder, other)`.
    ///
    /// Counts the same tensors a checkpoint stores, so the total times 4 bytes is
    /// directly comparable to a saved file's size. `other` is the tied char table plus
    /// the backbone's front/back projections, the decoder norm and the logit head.
    pub fn param_counts(&mut self) -> (usize, usize, usize, usize) {
        fn blocks(bs: &mut [Box<dyn BlockLike>]) -> usize {
            bs.iter_mut()
                .flat_map(|b| b.params_mut())
                .map(|t| t.len())
                .sum()
        }
        let enc = blocks(&mut self.encoder.blocks);
        let bb = blocks(&mut self.bb_blocks);
        let dec = blocks(&mut self.dec_blocks);
        let other = self.table.len()
            + self
                .bb_front
                .params_mut()
                .into_iter()
                .chain(self.bb_back.params_mut())
                .chain(self.dec_head.params_mut())
                .chain(self.dec_norm.params_mut())
                .map(|t| t.len())
                .sum::<usize>();
        (enc, bb, dec, other)
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
        // Safe here and not per block: the whole stack is done, so no cell is mid-sweep
        // holding a reference to the shared staging scratch.
        ops::clear_shared_lhs();
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

    /// Every parameter with its gradient and AdamW moments, in stage order.
    ///
    /// The single enumeration of this model's parameters: the arena binds it, and
    /// the checkpoint's `params_mut` / the diagnostics' `grads` read it back out.
    pub fn param_slots(&mut self) -> Vec<ParamSlot<'_>> {
        // The tied char table feeds the encoder's embedding and the decoder's char
        // slots; like every embedding-like table it trains undecayed.
        let mut v = vec![ParamSlot::new(
            &mut self.table,
            &mut self.dtable,
            &mut self.m_tbl,
            &mut self.v_tbl,
            ParamKind::NoDecay,
        )];
        for b in self.encoder.blocks.iter_mut() {
            v.extend(b.param_slots());
        }
        v.extend(self.bb_front.param_slots());
        for b in self.bb_blocks.iter_mut() {
            v.extend(b.param_slots());
        }
        v.extend(self.bb_back.param_slots());
        for b in self.dec_blocks.iter_mut() {
            v.extend(b.param_slots());
        }
        v.extend(self.dec_norm.param_slots());
        v.extend(self.dec_head.param_slots());
        v
    }

    /// AdamW across every stage: one launch over the parameter arena, one memset
    /// over its gradients.
    pub fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        // Walked for its side effect: handing out `&mut w` drops each layer's cached
        // bf16 weight. Skipping it leaves every later forward reading the pre-step
        // weight — training silently stops learning.
        drop(self.param_slots());
        self.arena
            .as_mut()
            .expect("parameters were never bound — see Hierarchical::bind_params")
            .step(gpu, cfg);
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
        let table = GTensor::from_host(gpu, &tensor_from_matrix(&emb.weights));
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
            head_optimizer_convention(&mut h);
            h
        };
        let cap = dl
            .iter()
            .find_map(|l| l.as_any().downcast_ref::<SoftCapLayer>())
            .map(|s| s.cap)
            .unwrap_or(crate::config::LOGIT_SOFTCAP);

        let cfg = ModelCfg {
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
        // loaded weight-bearing parts. Unbound: the parameters that end up in the
        // arena are the loaded ones, packed once below.
        let mut model = Hierarchical::unbound(gpu, cfg);
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
        // The loaded blocks replaced the ones `unbound` set up, so the offload opt-in
        // and the parameter arena both apply to the stack that ends up in the model.
        model.enable_backbone_offload(gpu);
        model.bind_params(gpu);
        Ok(model)
    }
}

/// Put a `Linear` on the logit head's optimizer footing: undecayed (it is a logit
/// head, not an interior projection) and with the bias frozen at its zero init, so
/// the layer stays equivalent to `nn::LinearNoBias` and exports to the `HIER` head
/// faithfully.
fn head_optimizer_convention(head: &mut Linear) {
    head.set_no_decay();
    head.freeze_bias();
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
        let cfg = ModelCfg {
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
        let mut model = Hierarchical::new(&gpu, cfg);

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
        let cfg = ModelCfg {
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
            words.push(Range {
                start: s,
                end: tokens.len(),
            });
        }

        // One model, never stepped: identical weights every rep, so any spread is
        // the forward/backward itself.
        let mut model = Hierarchical::new(&gpu, cfg);
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
        let cfg = ModelCfg {
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
            words.push(Range {
                start: s,
                end: tokens.len(),
            });
        }

        let mut model = Hierarchical::new(&gpu, cfg);
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
        let cfg = ModelCfg {
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
            let mut model = Hierarchical::new(&gpu, cfg);
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

        // Splitting a bucket changes only which rows share a rectangle, so the two legs
        // agree to reassociation. The sLSTM's fused path stages `Wh` in bf16, and the
        // two groupings then reduce the recurrence in a different order at 8-bit
        // mantissas — hence 1e-3 rather than fp32's 1e-5. A real batching bug (a row in
        // the wrong rectangle, a grad dropped on a split) moves these by orders of
        // magnitude, not by 1e-5.
        let tol = 1e-3;
        assert!(
            (loss_grouped - loss_single).abs() < tol,
            "grouped loss {loss_grouped} != single-rectangle loss {loss_single}"
        );
        for (i, (a, b)) in w_grouped.iter().zip(&w_single).enumerate() {
            assert!(
                (a - b).abs() < tol,
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
    /// Backbone chunks of four words: shorter than the mLSTM's own chunk length, so
    /// each cell call runs a single internal chunk.
    #[test]
    fn backbone_chunked_matches_unchunked() {
        chunked_vs_unchunked(12, 4);
    }

    /// The same comparison with each backbone chunk *longer* than the mLSTM's `L`, so
    /// a cell call runs several internal chunks and the state crossing the call border
    /// is a scanned one rather than a single chunk's own. Cross-call carry on top of a
    /// multi-chunk scan is a third path, reachable from neither alone, and it is the
    /// one the real config takes: 512-word chunks against `L` = 32 is 16 internal
    /// chunks, where the four-word case above is one.
    #[test]
    fn backbone_chunked_matches_unchunked_multi_mlstm_chunk() {
        let chunk = crate::gpu::ops::FUSED_MAX_L * 3 / 2;
        chunked_vs_unchunked(chunk * 3, chunk);
    }

    fn chunked_vs_unchunked(nwords: usize, chunk: usize) {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let cfg = ModelCfg {
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
        // Callers pass three chunks' worth, so there are two interior borders — a
        // single-border test misses a state that only goes wrong from the second on.
        let tokens: Vec<usize> = (0..nwords * 2).map(|i| 1 + i % 9).collect();
        let words: Vec<Range<usize>> = (0..nwords)
            .map(|w| Range {
                start: w * 2,
                end: w * 2 + 2,
            })
            .collect();

        let run = |chunk: usize| -> (f32, Vec<f32>, Vec<f32>) {
            let mut model = Hierarchical::new(&gpu, cfg);
            // Same starting weights for both legs. Named per word count so the two
            // callers cannot hand each other the wrong seed.
            let seed = std::env::temp_dir().join(format!("gpu_chunk_seed_{nwords}.hier"));
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
            // The backbone's own gradients, not just what leaks out through
            // `bb_front`: a state that crosses a chunk border wrong lands on the cells
            // that carry it, and is diluted by the time it reaches the stages on
            // either side.
            let mut bb: Vec<f32> = Vec::new();
            for blk in model.bb_blocks.iter_mut() {
                for g in blk.grads() {
                    bb.extend_from_slice(&g.to_host(&gpu).data);
                }
            }
            (loss, table, bb)
        };

        let (loss_c, table_c, front_c) = run(chunk);
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
        check(&front_c, &front_u, "backbone blocks");
    }

    /// A backbone sweep of three or more chunks must give the same gradients as one.
    ///
    /// Distinct from `backbone_chunked_matches_unchunked`, which runs chunks shorter
    /// than `FUSED_MIN_T` and so never exercises the time-fused loops. Here every chunk
    /// takes them *and* there are three, which is what it takes to run a chunk after
    /// `chunk_saved` has swapped the live buffers out from under the cell — a stale
    /// slab there gives NaN gradients from an out-of-bounds read rather than a
    /// wrong-but-finite number. Two chunks cannot catch it: the first still sees its
    /// own buffers.
    #[test]
    fn backbone_three_chunks_match_unchunked() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        // Wide enough that a chunk's buffers are a real allocation: at a toy width the
        // pool hands every chunk the same address back and a stale reference is
        // accidentally harmless, so the bug does not reproduce.
        let cfg = ModelCfg {
            vocab: 64,
            hc: 256,
            wh: 768,
            enc_blocks: 1,
            // Block 0 is sLSTM under the `i % 4` rule — the cell with the fused loop.
            bb_blocks: 2,
            dec_blocks: 1,
            heads: 8,
            dqk: 96,
            w_token: 63,
            cap: 30.0,
        };
        // 384 decoded words in chunks of 128: three chunks, each well over FUSED_MIN_T.
        let words_n = 385;
        let tokens: Vec<usize> = (0..words_n * 2).map(|i| 1 + i % 9).collect();
        let words: Vec<Range<usize>> = (0..words_n)
            .map(|w| Range {
                start: w * 2,
                end: w * 2 + 2,
            })
            .collect();

        let run = |chunk: usize| -> (f32, Vec<f32>, Vec<f32>) {
            let mut model = Hierarchical::new(&gpu, cfg);
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
        let cfg = ModelCfg {
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
            let mut model = Hierarchical::new(&gpu, cfg);
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
