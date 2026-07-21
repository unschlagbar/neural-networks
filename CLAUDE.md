# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Build (debug)
cargo build

# Build (optimized — fat LTO, single codegen unit)
cargo build --release

# Run
cargo run [--release]

# Bench
cargo bench --bench lstm_training
cargo bench --bench hierarchical_training   # set RAYON_NUM_THREADS=1 for a serial reference

# Check without producing a binary
cargo check
```

The binary reads one line from stdin to select its mode:

| Input | Action |
|-------|--------|
| *(empty)* | `train_normal` — trains the flat mLSTM model |
| `h` | `train_hierarchical` — trains the three-part hierarchical model (CPU) |
| `hg` | `train_hierarchical_gpu` — same model, trained on the GPU (needs `--features cuda`); checkpoints to `models/hier_gpu` as a `GHIR` blob |
| `s` | `sample_normal` — interactive sampling from the flat model |
| `hs` | `sample_hierarchical` — interactive sampling from the hierarchical model |
| `i` | `inspect_model` — prompts for a model name, looks it up in `models/` and prints all layers with their settings |

## Style 

do not use `0.0_f32`, always prever the simple mumber: `0.0`
All comments need to be in englisch not in german

## Architecture

### Layer abstraction (`src/nn_layer.rs`)

`NnLayer` is the core trait every layer implements. It owns its weights **and** its gradient accumulators (keeping weight/grad data adjacent). `DynCache` is a type-erased per-timestep forward cache; layers downcast it to their concrete type in `backward`. `SequentialBuilder` provides a fluent builder that assembles a `Sequential` from typed layer methods.

### Flat model (`src/sequential.rs`, `src/model.rs`)

`Sequential` holds `Vec<Box<dyn NnLayer>>` and a pre-allocated `cache[t][l]` matrix. The canonical model is `Embedding → mLSTMBlock → LinearNoBias → Softmax`. `make_cache(seq_len)` must be called once before training or sampling.

### Hierarchical model (`src/hierarchical.rs`, `src/word_encoder.rs`, `src/model.rs`)

`Hierarchical` (HAT-style, arXiv 2501.10322) couples three stages:

- **encoder** (`WordEncoder`) — a **bidirectional** BiLSTM over the characters of one word plus a closing `[W]` end-of-word step (fed virtually — token slices never contain it). Two forward-only `Sequential` stacks (`Embedding → sLSTMBlock × 2` each): `fwd` reads `c1 … cn [W]`, `bwd` reads `cn … c1 [W]` — the characters are reversed but the closing `[W]` stays LAST in both, so **both directions read out at a `[W]` step** (`hallo` → fwd `h a l l o [W]`, bwd `o l l a h [W]`). Reversing the whole sequence instead (`[W] o l l a h`) would read `bwd` out at an ordinary char step with no end-of-word signal; `word_encoder::dir_stream` and its tests pin the correct alignment. The two readouts are concatenated `[fwd ; bwd]` (width `2·CHAR_HIDDEN`) and a `combine` Linear projects back to `CHAR_HIDDEN` → `e_w`, keeping the backbone input width and the tied char-table width unchanged. Only `fwd`'s embedding (layer 0) is the tied table; `bwd` keeps its own. State is reset per word. Each direction has its own replica pool; encode/backward run data-parallel over words on both.
- **word_model** (backbone) — `Linear → alternating sLSTM/mLSTM blocks × WORD_BLOCKS → Linear` — autoregresses over word embeddings, carrying recurrent state across words; its output is the context for decoding the *next* word.
- **char2_model** (decoder) — `sLSTMBlock × 2 → RMSNorm → LinearNoBias → SoftCap`, input width OUT_HIDDEN — the decoder has no front layer; `Hierarchical` builds its inputs itself (paper eq. 3–4): a word's **first sequence step is the injected backbone context** (it takes the BOS slot), every later step feeds the previous char through the **encoder's char embedding** (tied table — requires CHAR_HIDDEN == OUT_HIDDEN; decoder-side embedding grads are reduced back into the encoder's table in `backwards_sequence`). Predicts the word's chars plus a trailing `[W]` (EOS). Reset per word.

Optimizer assignment convention: interior projections use `linear`/`Linear` (Muon, weight-decayed); `linear_no_bias` and the embedding layers train on plain Adam without decay and are reserved for embedding-like tables and logit heads — putting hidden projections on the Adam path causes unbounded weight growth over long runs. The decoder's logit head is additionally followed by `SoftCap` (`src/nn/soft_cap.rs`, tag 17, `LOGIT_SOFTCAP = 30` like xLSTM-7B): `logits = cap·tanh(z/cap)` bounds the logits so the undecayed Adam head has no incentive to grow without limit.

Words come from `src/segment.rs` (see Tokenizer below), not from a boundary-token set. Forward/backward run phase by phase over a whole window (all encodes, then the backbone sweep, then all decodes).

The encoder and decoder phases run **data-parallel over words** (rayon; see `src/parallel.rs`): each worker thread gets a full replica of the stack (`ReplicaPool`, copied weights via the NNFW round-trip, own recurrent state and grad accumulators) plus a disjoint slice of the shared forward cache, and after a parallel backward phase the replica grads are reduced into the master (`NnLayer::add_grads_from`). Replicas are rebuilt lazily after each optimizer step. Only the backbone sweep is serial (it carries cross-word state). Layers that appear in a parallel-trained stack must implement `add_grads_from`; `tests/parallel_parity.rs` checks the parallel path against single-threaded execution.

### The stabilizer must stay invisible (both cells)

Both recurrent cells carry a running stabilizer `m` and store **stabilized** state:
`c = c_true·exp(−m)`, `n = n_true·exp(−m)` (and `C` likewise in the mLSTM). `m` is a
pure numerical reparametrization — it must not change what the model computes. Any
clamp applied to a *stabilized* quantity breaks that, because the clamp threshold
then sits at `exp(m)` in unstabilized terms. Both cells used to get this wrong (a
`max(…, 1)` on the stabilized normalizer); both are now fixed, and the rule is:

- **sLSTM**: `h = o·c/n`, with **no clamp** — the `exp(−m)` cancels in the ratio, which
  is what makes it invariant. On the first step of a sequence `n == 0`, so `m` is set
  to `ĩ` (not `max(logσ(f̃)+m_prev, ĩ)`), making `i'` exactly 1 and keeping `c/n` off
  `0/0`. State resets per word here, so that case is hit constantly.
- **mLSTM**: `ψ = max(|nᵀq|, exp(−m))` — *not* `max(|nᵀq|, 1)`. This is xLSTM's
  `max(|n_trueᵀq|, 1)` with the `exp(m)` cancelled against the numerator.

Both match `NX-AI/xlstm` (`slstm/src/vanilla/slstm.py`) and `NX-AI/mlstm_kernels`.
**Finite-difference tests cannot catch this class of bug** — the backward correctly
differentiates the wrong forward, so FD stays green either way. Only comparing
against the reference finds it. Treat any new `max(…, 1)` on a stabilized value as a
bug.

### Optimizer (`src/optimizers/mod.rs`)

`pub type Optimizer = Muon` selects the active optimizer (Muon for 2D hidden weights, aux-Adam for embeddings and 1D params). All layers use the type aliases `GradMatrix` / `GradVec` from this module, so swapping optimizers only requires changing that one type alias. `BATCH_SIZE` in `config.rs` accumulates gradients over that many windows before each optimizer step.

### Tokenizer (`src/tokenizer_utf8.rs`) and word segmentation (`src/segment.rs`)

`Utf8Tokenizer` is byte-level: ids `0..256` are raw UTF-8 bytes, ids `256..` are the specials in `SPECIAL_TOKENS` (`<W>` = end-of-word marker, `<END>`), so `vocab_size() == 256 + SPECIAL_TOKENS.len()` (258). Any UTF-8 text round-trips losslessly and no input byte can collide with a special. There is no charset file. Sampling decodes with `Utf8Printer` (`src/sampling.rs`), which holds bytes back until they form a whole character.

`segment::word_ends` decides what a "word" is — the unit the backbone autoregresses over — with a lexer-shaped split tuned for Rust: a whitespace run is one unit and attaches as a *suffix* to the word before it (`"use "`, `";\n    "`), so a word carries the separator that closes it and the decoder emits that separator right before `[W]`; identifiers/keywords and numbers stay whole (`foo`, `1_000u32`), multi-byte operators are one word (`::`, `->`, `..=`, `//`), lifetimes (`'a`) differ from char literals (`'a'`), and non-ASCII bytes group into their character. Words tile the sequence contiguously and are capped at `MAX_WORD_BYTES` (config), which bounds the decoder unroll. Roughly 3–4 bytes per word on Rust source. `cargo run --example seg_demo -- src/foo.rs` prints the split for a file.

### Dataset (`src/batches.rs`)

Both training modes stream `DATA_FILE` (single file with `<|endoftext|>` document separators) through `ChunkedWordDataSet`: it reads `CHUNK_BYTES` of raw text at a time, cuts at the last complete document (the partial tail is carried into the next chunk), and tokenizes + word-windows each chunk into a `WordChunk`. Windows never cross document borders, so a streamed epoch yields exactly the windows a whole-file load would — but peak memory is bounded by `CHUNK_BYTES`, not corpus size (>1 GB corpora stream fine). Training loops call `rewind()` per epoch and grow the model cache on demand when a chunk's `max_window_tokens()` exceeds the current size; `count_windows()` does a cheap counting pass for resume arithmetic. `PreparedDataSet` (token-window loading from `DATA_DIR`) still exists but is currently unused by training.

### GPU backend (`src/gpu/`, feature `cuda`, default-off)

A device-resident port of the `nn2` stack (see `PLAN-gpu.md`): `DTensor` (resident
device buffer), cuBLAS GEMM + ~21 NVRTC kernels, and layers `Linear`, `RmsNorm`,
`SLstm`, `MLstm`, `Block<Cell>`, `Lm`, `Hierarchical`. Every layer is parity-tested
against its `nn2` twin (`cargo test --features cuda gpu::`). The mLSTM uses the
**chunkwise parallel/attention** formulation, not the scalar recurrence (~128x
faster than the CPU cell): the sequence is cut into chunks of `config::MLSTM_CHUNK`
(256), each an attention-style `[heads, L, L]` block, with the stabilized recurrent
state `(C, n, m)` carried across boundaries — O(T·L), linear in T, where the
single-chunk form was O(T²). It is an exact refactoring, not an approximation (the
chunk-local row-max telescopes to the global one), so both paths agree to fp
tolerance; `MLSTM_CHUNK=0` restores the single-chunk path for A/B benchmarking and
`mlstm_chunking_matches_single_chunk` pins the two together. A sequence shorter
than one chunk (the encoder/decoder, where T is a word length) takes the
single-chunk path unchanged. `gpu::Hierarchical` checkpoints to the `GHIR` format
(config header + every param in `params_mut` order); weights only, so a resumed run
restarts the Adam moments.

#### Fused mLSTM kernels (TFLA)

The chunkwise math above runs as **five fused kernels** (after
[nx-ai/mlstm_kernels](https://github.com/NX-AI/mlstm_kernels)): three launches for a
whole forward, two for a backward. It replaces an op-at-a-time chunk loop that
issued ~25 launches *per chunk* — ~600 for one fwd+bwd at the backbone's shape,
which was ~1 ms of arithmetic stretched over 14 ms of driver latency. That is why a
5070 and a 4050m used to score the same: neither was computing, both were waiting on
launches.

The decomposition splits the one sequential axis (the chunk state `C, n, m`, which is
small) from the FLOPs (every chunk's intra-chunk attention, which is independent):

| kernel | grid | does |
|---|---|---|
| `mlstm_fw_gates` | one thread per (bh, chunk) | `fc`, the chunk-local cumulative log-forget |
| `mlstm_fw_C` | (dhv tiles, BH) | all chunk states, looping chunks **inside** the kernel |
| `mlstm_fw_parallel` | (chunks, BH) | every chunk independently: intra attention + state read-out |
| `mlstm_bw_dC` | (dhv tiles, BH) | the state gradients, chunks in reverse, inside the kernel |
| `mlstm_bw_parallel` | (chunks, BH) | dQ/dK/dV and the gate grads, every chunk independently |

The `[L, L]` decay matrix never reaches HBM — it lives in shared memory for the life
of a block, and backward *recomputes* it from `(Q, K, fc, ig, m)` rather than reading
back what forward saved. Three things dominated the kernels' own speed, in order:
**bank conflicts** (every 2D shared array is stored with its row stride padded by one
float, or a warp walking rows at fixed column hits one bank 32 ways — worth ~2x);
**tiling the two recurrent kernels over `dhv`** (their grid is otherwise BH alone, 8
blocks on a 48-SM card — worth ~1.6x); and **parallelizing the `da`/`db` reductions**
in backward over the full `len·dhv` grid instead of `len` threads.

`cargo run --release --features cuda --example gpu_occupancy` asks the **driver** what
these kernels cost — registers/thread, shared memory, register spills, blocks resident
per SM, and `waves` (grid ÷ machine capacity). Run it after touching a kernel. What it
currently says: no spills anywhere; shared memory (not registers) is what caps the two
parallel kernels, which is why they launch 512 threads and the recurrent ones 256
(`FUSED_THREADS_PAR` / `_REC`); and `fw_C`/`bw_dC` sit at `waves` ≈ 0.15, i.e. their
grid is too small to fill the GPU at B=1 — the standing argument for batching the
backbone over windows.

`FUSED_MAX_L` (32, in `gpu/ops.rs`) caps the chunk length the kernels accept — it is
what bounds their shared memory. A configured `MLSTM_CHUNK` above it is silently
clamped, which is free: chunk length is a blocking choice with no effect on the
result. `MLSTM_CHUNK=0` (single chunk) and `MLSTM_LEGACY=1` both fall back to the
op-at-a-time path, which is retained as the A/B baseline and as the oracle for
`mlstm_fused_matches_legacy` — the test that runs both at the real backbone shape
(dqk=dhv=64, T ending on a short chunk) and requires identical forward, `dx` and
weight updates.

#### Tensor cores (`mma.sync`, TF32)

Every contraction in the two *parallel* kernels runs on the **tensor cores**, which
is what the reference gets from Triton (`tl.dot` on fp32 inputs is TF32 by default).
`mlstm_fw_parallel_mma` / `mlstm_bw_parallel_mma` are drop-in twins of the scalar
kernels — same algorithm, same shared-memory plan — with the dots issued as
`mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` (SASS: `HMMA.1688.F32.TF32`).
The forward's three dots are `Q·Kᵀ`, `(D̄⊙S)·V` and `Q·C_prevᵀ`; the backward's seven
are grouped so that dots writing the *same* output tile share one warp pass and one
epilogue (dV/`st`/`pre` all land on `[L, dhv]`; dQ/dK all land on `[L, dqk]`).

Three things this buys and costs:

- **Speed.** The fused core goes 1.89 → 1.43 ms fwd+bwd at the backbone shape (1.32x;
  backward alone 1.37 → 1.00 ms). End-to-end that is only ~1% of a training window —
  the backbone is dominated by its **sLSTM** blocks (~124 ms of a ~260 ms window,
  against ~50 ms for all the mLSTMs). The mLSTM core is no longer where the time is.
- **Precision.** TF32 keeps fp32's exponent and accumulator but rounds the
  multiplicands to 10 mantissa bits, so the dots carry ~4e-4 relative error (measured,
  and exactly what `mlstm_kernels` carries). `mlstm_fused_mma_matches_scalar` pins the
  tensor-core path against the scalar one at that tolerance and must not be tightened
  into an exactness check. `MLSTM_NO_MMA=1` selects the scalar kernels — the A/B
  baseline and the fp32 oracle.
- **Padding, not special cases.** Shapes are padded up to the mma tile (rows to 16,
  contraction to 8, columns to 8) and the pad is zero-filled, so a short final chunk
  and an odd `dqk`/`dhv` need no separate code path: a zero row contributes nothing to
  a dot. `fused_smem("*_mma", ..)` must track the kernel's `LP`/`KP`/`VP` exactly.

`kernels.rs` compiles `SRC` **twice**, on purpose. `mma.sync` does not exist at
NVRTC's default target arch, so the tensor-core kernels need `--gpu-architecture`
pointed at the real device — but that flag also changes how ptxas contracts FMAs in
every *other* kernel, shifting their last bits. So the base module is compiled exactly
as before (default arch, `MMA_TF32` undefined, the `#if` blocks preprocessed away) and
every pre-existing kernel is taken from it bit-for-bit; a second module, compiled for
the device arch with `MMA_TF32`, supplies only the two mma kernels. Devices below
sm_80 simply skip the second compile and run the scalar path.

> **Known-flaky test.** `mlstm_fused_matches_legacy` fails ~50% of runs *on `main`*,
> and has nothing to do with the tensor cores. It compares weights after **one** Adam
> step, where the update is `lr·g/(√(g²)+ε)` — for a weight whose gradient is near
> zero that ratio is hypersensitive, so a 1e-7 difference in gradient becomes ~1e-4 in
> the weight, past the 2e-5 tolerance. The test data is randomly redrawn each run,
> hence the coin flip. Comparing the *gradients* instead of the post-step weights
> would fix it.

**VRAM.** Word batching is what dominates a window, and it has a knob:
- *Word batching.* The encoder/decoder run words as `[words, tmax]` rectangles and
  `tmax` is the longest word in the group, so a single 16-byte word would pad every
  2-byte word (~4.5x waste on Rust source). `group_by_len` splits a window's words
  into power-of-two length groups, each its own dense rectangle — the mLSTM's
  per-word `[T, T]` attention still needs a rectangle, which is why groups exist
  rather than a fully packed row layout. The decoder runs forward+backward per
  group; the encoder re-forwards each group in its backward phase (activation
  checkpointing), so only one group is ever resident. `gpu::hierarchical`'s
  `grouping_matches_single_rectangle` pins this to be numerically identical to the
  unsplit path, and `GPU_NO_GROUP=1` restores that path for A/B benchmarking.
The backbone's decay matrices used to be the other one — `[heads, T, T]`, 134 MB
each per mLSTM block at 2048 words — and were recomputed in backward behind an
`MLSTM_RECOMPUTE` flag to claw that back. Chunking shrank them to `[heads, L, L]`
per chunk (~64 MB across the whole backbone), so the flag was **removed**: forward
just keeps them and backward skips the rebuild GEMM.

`cargo run --release --features cuda --example mlstm_chunk_bench` sweeps the cell in
T and reports ms/iter + VRAM (honors `MLSTM_CHUNK`); it is how the 256 default was
picked. `cargo run --release --features cuda --example mlstm_stage_prof` breaks one
cell down into cuBLAS throughput / projections / fused core / shell, which is how the
fused kernels were tuned — run it with `MLSTM_LEGACY=1` for the before picture.

`GPU_PROF=1` prints per-phase timings; `GPU_MEM=1` adds device memory in use after
each phase. `cargo run --release --features cuda --example gpu_fit -- <corpus> [words]`
runs a corpus's worst-shaped window and reports peak VRAM and s/window.

`GPU_TF32=1` (**off by default**) puts cuBLAS's math mode on the tensor cores, so
every GEMM in the backend runs in TF32 instead of fp32 SGEMM. It is worth ~1.35x on
the GEMMs (18 → 25 TFLOP/s at the backbone's shapes) but only ~4% on a training step,
because a step is dominated by the backbone's sLSTM rather than by matmul — and it
reaches *every* GEMM, including the ones the parity tests use as an exact-fp32 oracle
(8 of them fail with it on). Hence a switch, not a default. This is independent of the
mLSTM kernels' own tensor-core dots, which are on by default.

### Persistence (`src/format.rs`, `src/saving.rs`, `src/loading.rs`)

One unified on-disk format, **`NNM1`** (magic `0x4E4E_4D31`), owned by `src/format.rs`. Every model — flat and hierarchical, CPU and GPU — writes this container; there are no other file magics (the old `NNFW`/`HIE4`/`HIE5`/`HMRN` are gone, no backward compatibility).

Layout: `MAGIC u32`, `VERSION u8`, `KIND u8` (`0` = Flat, `1` = Hierarchical), a typed metadata head (Hierarchical carries `vocab u32`, `context u32`, `step u64`; Flat carries none), then `N_SECTIONS u32` followed by that many **named sections**, each a length-prefixed name string plus one *stack blob*.
- **Flat** models write a single `"model"` section.
- **Hierarchical** models write `"encoder_fwd"`, `"encoder_bwd"`, `"combine"`, `"word_model"`, `"char2_model"`. `combine` is a one-layer stack (the single projection Linear); every section is uniformly a stack blob.

A **stack blob** (`saving::write_layers` / `loading::load_layers`, magic `STACK_MAGIC` = `0x4E4E_4657`) is one layer stack's architecture header (per-layer tag + in/out sizes) then its weights. It is the reusable building block: the container frames it as a section, and it is also used in-memory for the replica round-trip (`Sequential::replicas`) and for building GPU models. It is not a standalone file format.

`format::Writer` builds a container fluently (`.section(name, &layers)` / `.section_layer(name, &layer)`); `format::Reader` reads one back and hands out sections by name (`take` / `take_stack`), with `peek_kind` for `inspect` to label a file. The GPU encoder is still unidirectional, so `gpu::Hierarchical::save` writes `encoder_bwd` as a copy of `encoder_fwd` and a fresh `combine` so the CPU bidirectional model can load and fine-tune it.

Models are saved to `models/` (created automatically). Paths and all hyperparameters (LR, dims, sequence length, save/print intervals) live in `src/config.rs`.

### Key dependency

`iron_oxide` is a local crate at `../iron_oxide` providing the `Matrix` type used throughout for weight storage.
