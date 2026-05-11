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

# Check without producing a binary
cargo check
```

The binary reads one line from stdin to select its mode:

| Input | Action |
|-------|--------|
| *(empty)* | `train_normal` — trains the flat mLSTM model |
| `h` | `train_hierarchical` — trains the three-part hierarchical model |
| `s` | `sample_normal` — interactive sampling from the flat model |
| `hs` | `sample_hierarchical` — interactive sampling from the hierarchical model |

## Architecture

### Layer abstraction (`src/nn_layer.rs`)

`NnLayer` is the core trait every layer implements. It owns its weights **and** its gradient accumulators (keeping weight/grad data adjacent). `DynCache` is a type-erased per-timestep forward cache; layers downcast it to their concrete type in `backward`. `SequentialBuilder` provides a fluent builder that assembles a `Sequential` from typed layer methods.

### Flat model (`src/sequential.rs`, `src/model.rs`)

`Sequential` holds `Vec<Box<dyn NnLayer>>` and a pre-allocated `cache[t][l]` matrix. The canonical model is `Embedding → mLSTMBlock → LinearNoBias → Softmax`. `make_cache(seq_len)` must be called once before training or sampling.

### Hierarchical model (`src/hierarchical.rs`, `src/model.rs`)

`HierarchicalSequential` couples three `Sequential` sub-models:

- **char_model** — `Embedding → sLSTMBlock` — encodes the current character into a `CHAR_HIDDEN`-dim vector.
- **word_model** — `Linear → mLSTMBlock × 4 → sLSTMBlock` — receives `char_model` output **only at boundary tokens** (spaces, punctuation), producing a `WORD_HIDDEN`-dim context vector. It is reset between boundaries.
- **char2_model** — `Linear → sLSTMBlock × 2 → RMSNorm → Linear → Softmax` — takes `[char1_out ‖ high_ctx]` and predicts the next token every step.

Boundary tokens are determined by `Tokenizer::boundary_tokens()` and passed in at construction. At each boundary the char and char2 models are reset; the word_model accumulates across boundaries.

### Optimizer (`src/opimizers/mod.rs`)

`pub type Optimizer = AdamW` selects the active optimizer. All layers use the type aliases `GradMatrix` / `GradVec` from this module, so swapping optimizers only requires changing that one type alias.

### Tokenizer (`src/tokenizer.rs`)

Character-level tokenizer built from `charset.txt`. Special tokens `<SPACE2>`, `<SPACE4>`, `<END>`, `<QSTART>`, `<QEND>` extend the base vocab. `vocab_size()` includes these specials.

### Dataset (`src/batches.rs`)

`PreparedDataSet` tokenizes files once at startup and pre-computes sliding windows (each window extends its right edge to the nearest boundary token). `shuffle()` reorders only the window index list — no re-tokenization. Normal training reads files from `DATA_DIR` (directory); hierarchical training reads `DATA_FILE` (single file with `---FILE---` chunk separators).

### Persistence (`src/saving.rs`, `src/loading.rs`)

Two binary formats:
- `NNFW` (magic `0x4E4E_4657`) — `Sequential` flat format: architecture header then weights, all little-endian.
- `HIER` (magic `0x4849_4552`) — wraps three `NNFW` blobs with vocab/boundary metadata.

Models are saved to `models/` (created automatically). Paths and all hyperparameters (LR, dims, sequence length, save/print intervals) live in `src/config.rs`.

### Key dependency

`iron_oxide` is a local crate at `../iron_oxide` providing the `Matrix` type used throughout for weight storage.
