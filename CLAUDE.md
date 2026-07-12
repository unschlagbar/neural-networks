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

- **encoder** (`WordEncoder`) — a normal forward-only `Sequential` (`Embedding → sLSTMBlock × 2`) over the characters of one word plus a closing `[W]` end-of-word step (fed virtually — token slices never contain it); the word embedding `e_w` is the output at the `[W]` step. State is reset per word.
- **word_model** (backbone) — `Linear → alternating sLSTM/mLSTM blocks × WORD_BLOCKS → Linear` — autoregresses over word embeddings, carrying recurrent state across words; its output is the context for decoding the *next* word.
- **char2_model** (decoder) — `sLSTMBlock × 2 → RMSNorm → LinearNoBias → SoftCap`, input width OUT_HIDDEN — the decoder has no front layer; `Hierarchical` builds its inputs itself (paper eq. 3–4): a word's **first sequence step is the injected backbone context** (it takes the BOS slot), every later step feeds the previous char through the **encoder's char embedding** (tied table — requires CHAR_HIDDEN == OUT_HIDDEN; decoder-side embedding grads are reduced back into the encoder's table in `backwards_sequence`). Predicts the word's chars plus a trailing `[W]` (EOS). Reset per word.

Optimizer assignment convention: interior projections use `linear`/`Linear` (Muon, weight-decayed); `linear_no_bias` and the embedding layers train on plain Adam without decay and are reserved for embedding-like tables and logit heads — putting hidden projections on the Adam path causes unbounded weight growth over long runs. The decoder's logit head is additionally followed by `SoftCap` (`src/nn/soft_cap.rs`, tag 17, `LOGIT_SOFTCAP = 30` like xLSTM-7B): `logits = cap·tanh(z/cap)` bounds the logits so the undecayed Adam head has no incentive to grow without limit.

Boundary tokens come from `Tokenizer::boundary_tokens()`; a word is the token span up to and including its boundary token. Forward/backward run phase by phase over a whole window (all encodes, then the backbone sweep, then all decodes).

The encoder and decoder phases run **data-parallel over words** (rayon; see `src/parallel.rs`): each worker thread gets a full replica of the stack (`ReplicaPool`, copied weights via the NNFW round-trip, own recurrent state and grad accumulators) plus a disjoint slice of the shared forward cache, and after a parallel backward phase the replica grads are reduced into the master (`NnLayer::add_grads_from`). Replicas are rebuilt lazily after each optimizer step. Only the backbone sweep is serial (it carries cross-word state). Layers that appear in a parallel-trained stack must implement `add_grads_from`; `tests/parallel_parity.rs` checks the parallel path against single-threaded execution.

### Optimizer (`src/optimizers/mod.rs`)

`pub type Optimizer = Muon` selects the active optimizer (Muon for 2D hidden weights, aux-Adam for embeddings and 1D params). All layers use the type aliases `GradMatrix` / `GradVec` from this module, so swapping optimizers only requires changing that one type alias. `BATCH_SIZE` in `config.rs` accumulates gradients over that many windows before each optimizer step.

### Tokenizer (`src/tokenizer.rs`)

Character-level tokenizer built from `charset.txt`. Special tokens `<SPACE2>`, `<SPACE4>`, `<END>`, `<QSTART>`, `<QEND>` extend the base vocab. `vocab_size()` includes these specials.

### Dataset (`src/batches.rs`)

Both training modes stream `DATA_FILE` (single file with `<|endoftext|>` document separators) through `ChunkedWordDataSet`: it reads `CHUNK_BYTES` of raw text at a time, cuts at the last complete document (the partial tail is carried into the next chunk), and tokenizes + word-windows each chunk into a `WordChunk`. Windows never cross document borders, so a streamed epoch yields exactly the windows a whole-file load would — but peak memory is bounded by `CHUNK_BYTES`, not corpus size (>1 GB corpora stream fine). Training loops call `rewind()` per epoch and grow the model cache on demand when a chunk's `max_window_tokens()` exceeds the current size; `count_windows()` does a cheap counting pass for resume arithmetic. `PreparedDataSet` (token-window loading from `DATA_DIR`) still exists but is currently unused by training.

### GPU backend (`src/gpu/`, feature `cuda`, default-off)

A device-resident port of the `nn2` stack (see `PLAN-gpu.md`): `DTensor` (resident
device buffer), cuBLAS GEMM + ~21 NVRTC kernels, and layers `Linear`, `RmsNorm`,
`SLstm`, `MLstm`, `Block<Cell>`, `Lm`, `Hierarchical`. Every layer is parity-tested
against its `nn2` twin (`cargo test --features cuda gpu::`). The mLSTM uses the
**parallel/attention** formulation, not the scalar recurrence — O(T²) in sequence
length but ~128x faster than the CPU cell. `gpu::Hierarchical` checkpoints to the
`GHIR` format (config header + every param in `params_mut` order); weights only,
so a resumed run restarts the Adam moments.

### Persistence (`src/saving.rs`, `src/loading.rs`)

Two binary formats:
- `NNFW` (magic `0x4E4E_4657`) — `Sequential` flat format: architecture header then weights, all little-endian.
- `HIER` (magic `0x4849_4552`) — wraps three `NNFW` blobs with vocab/boundary metadata.

Models are saved to `models/` (created automatically). Paths and all hyperparameters (LR, dims, sequence length, save/print intervals) live in `src/config.rs`.

### Key dependency

`iron_oxide` is a local crate at `../iron_oxide` providing the `Matrix` type used throughout for weight storage.
