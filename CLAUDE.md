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
| `av` | `grow_vocab` — **only for old checkpoints**: adds the SFT chat markers to a checkpoint pretrained before they existed (grows the tied embedding + logit head to the current `vocab_size()`). New models are built with the full vocab and never need this |
| `hq` | `train_sft` — CPU Q-A / instruction fine-tuning on `SFT_DATA`, loss masked to the response. The CPU twin of `hqg` |
| `hqg` | `train_sft_gpu` — GPU Q-A / instruction fine-tuning on `SFT_DATA`, loss masked to the response (needs `--features cuda`). Works directly on any current checkpoint; only an old (smaller-vocab) one must be run through `av` first |
| `s` | `sample_normal` — interactive sampling from the flat model |
| `hs` | `sample_hierarchical` — interactive sampling from the hierarchical model |
| `hqs` | `sample_chat` — interactive Q-A sampling from an SFT model (prompts for an instruction + optional context, generates the response until `<END>`) |
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

- **encoder** (`WordEncoder`) — a normal forward-only `Sequential` (`Embedding → sLSTMBlock × N`) over the characters of one word plus a closing `[W]` end-of-word step (fed virtually — token slices never contain it); the word embedding `e_w` is the output at the `[W]` step, where the state has seen the whole word and knows it is complete. State is reset per word; the words of a window encode data-parallel across a replica pool.
- **word_model** (backbone) — `Linear → alternating sLSTM/mLSTM blocks × WORD_BLOCKS → Linear` — autoregresses over word embeddings, carrying recurrent state across words; its output is the context for decoding the *next* word.
- **char2_model** (decoder) — `sLSTMBlock × 2 → RMSNorm → LinearNoBias → SoftCap`, input width OUT_HIDDEN — the decoder has no front layer; `Hierarchical` builds its inputs itself (paper eq. 3–4): a word's **first sequence step is the injected backbone context** (it takes the BOS slot), every later step feeds the previous char through the **encoder's char embedding** (tied table — requires CHAR_HIDDEN == OUT_HIDDEN; decoder-side embedding grads are reduced back into the encoder's table in `backwards_sequence`). Predicts the word's chars plus a trailing `[W]` (EOS). Reset per word.

Optimizer assignment convention: interior projections use `linear`/`Linear` (Muon, weight-decayed); `linear_no_bias` and the embedding layers train on plain Adam without decay and are reserved for embedding-like tables and logit heads — putting hidden projections on the Adam path causes unbounded weight growth over long runs. The decoder's logit head is additionally followed by `SoftCap` (`src/nn/soft_cap.rs`, tag 17, `LOGIT_SOFTCAP = 30` like xLSTM-7B): `logits = cap·tanh(z/cap)` bounds the logits so the undecayed Adam head has no incentive to grow without limit.

Words come from `src/segment.rs` (see Tokenizer below), not from a boundary-token set. Forward/backward run phase by phase over a whole window (all encodes, then the backbone sweep, then all decodes).

The encoder and decoder phases run **data-parallel over words** (rayon; see `src/parallel.rs`): each worker thread gets a full replica of the stack (`ReplicaPool`, copied weights via the NNFW round-trip, own recurrent state and grad accumulators) plus a disjoint slice of the shared forward cache, and after a parallel backward phase the replica grads are reduced into the master (`NnLayer::add_grads_from`). Replicas are rebuilt lazily after each optimizer step. Only the backbone sweep is serial (it carries cross-word state). Layers that appear in a parallel-trained stack must implement `add_grads_from`; `tests/parallel_parity.rs` checks the parallel path against single-threaded execution.

### Optimizer (`src/optimizers/mod.rs`)

`pub type Optimizer = Muon` selects the active optimizer (Muon for 2D hidden weights, aux-Adam for embeddings and 1D params). All layers use the type aliases `GradMatrix` / `GradVec` from this module, so swapping optimizers only requires changing that one type alias. `BATCH_SIZE` in `config.rs` accumulates gradients over that many windows before each optimizer step.

### Tokenizer (`src/tokenizer_utf8.rs`) and word segmentation (`src/segment.rs`)

`Utf8Tokenizer` is byte-level: ids `0..256` are raw UTF-8 bytes, ids `256..` are the specials in `SPECIAL_TOKENS` (`<W>` = end-of-word marker, `<END>`), so `vocab_size() == 256 + SPECIAL_TOKENS.len()` (258). Any UTF-8 text round-trips losslessly and no input byte can collide with a special. There is no charset file. Sampling decodes with `Utf8Printer` (`src/sampling.rs`), which holds bytes back until they form a whole character.

`segment::word_ends` decides what a "word" is — the unit the backbone autoregresses over — with a lexer-shaped split tuned for Rust: a whitespace run is one unit and attaches as a *suffix* to the word before it (`"use "`, `";\n    "`), so a word carries the separator that closes it and the decoder emits that separator right before `[W]`; identifiers/keywords and numbers stay whole (`foo`, `1_000u32`), multi-byte operators are one word (`::`, `->`, `..=`, `//`), lifetimes (`'a`) differ from char literals (`'a'`), and non-ASCII bytes group into their character. Words tile the sequence contiguously and are capped at `MAX_WORD_BYTES` (config), which bounds the decoder unroll. Roughly 3–4 bytes per word on Rust source. `cargo run --example seg_demo -- src/foo.rs` prints the split for a file.

### Post-training / SFT (`src/sft.rs`, `src/grow_vocab.rs`, `src/gpu/train.rs`)

Q-A instruction tuning of a pretrained hierarchical model, GPU only. Three pieces:

- **Chat special tokens.** `SPECIAL_TOKENS` gains `<CONTEXT>` (258) and `<SEP>` (259) *after* the pretraining `<W>`/`<END>`, so no existing byte/`<W>`/`<END>` id ever shifts. `vocab_size()` is therefore 260, while a pretrained checkpoint has 258 (`PRETRAIN_SPECIALS`). The pretraining `<END>` (257) doubles as the SFT end-of-response target/stop.
- **Growing the vocab (`grow_vocab.rs`, mode `av`) — only for old checkpoints.** New models are built at `tokenizer.vocab_size()`, so they already carry the SFT marker rows and go straight into `hqg`. `av` exists solely to upgrade a checkpoint pretrained *before* the markers were added: its two vocab-sized tables — the encoder's tied char **embedding** (`[vocab, HC]`, rows) and the decoder's logit **head** (`LinearNoBias`, `[HC, vocab]`, cols; its trailing `SoftCap` is resized too) — have no rows/cols for the new ids. `grow_checkpoint` appends them: new embedding rows get a small random init, new head columns start at **zero** (neutral initial logit), every existing weight preserved byte-for-byte, backbone untouched (it never sees token ids). `hqg` detects a stale-vocab checkpoint and points at `av`; a current model never triggers it.
- **SFT dataset (`sft.rs`).** Reads a dolly-style JSONL (`{instruction, context, response, category}`, dependency-free string parser — no serde) and formats each record as `{instruction}[<CONTEXT>{context}]<SEP>{response}<END>`, one window per example. `segment::word_ends` already makes each special token its own one-token word, so the per-word **loss mask** is a clean flag: words at/after the response carry loss (`true`), the prompt (through `<SEP>`) does not. The whole sequence is still encoded/decoded — the backbone must read the prompt — but gradient flows only from the response. Examples longer than `SFT_MAX_TOKENS` (2048) are dropped at load: the per-example cache is sized to one slot per token, so dolly's ~27k-token tail would otherwise force a ~16 GB cache (~1.2 GB at 2048) and OOM before the first step.
- **Training — two paths, same masking.** Both load the whole (small) SFT set into memory, shuffle per epoch, reject a stale-vocab checkpoint with a pointer to `av`, and fine-tune at `SFT_LR` (1e-5, ×10 below pretraining) over `SFT_EPOCHS`.
  - **GPU (`train_sft_gpu`, mode `hqg`)** calls `gpu::Hierarchical::forward_backward_masked`, which mirrors `forward_backward` but takes a per-decoded-word mask (`word_loss[w]` gates the CE mask of every decoder slot of word `w+1`) and normalizes the loss by the **unmasked** row count.
  - **CPU (`train_sft`, mode `hq`)** calls `Hierarchical::train_sft`, which sets a per-window `dec_word_loss` mask consumed by `backwards_sequence`: a masked word's decoder slots seed a **zero** delta (BPTT still unwinds, but no gradient flows to logits, context or the tied table) and are dropped from `decode_loss`. `hierarchical.rs` pins this with two tests — a fully-masked window gives exactly zero head gradient, and the per-word masks sum to the full-window gradient (masking is additive and isolating).
- **Inference (`sample_chat`, mode `hqs`).** Prompts for an instruction + optional context, formats them with `sft::format_prompt` (trailing `<SEP>`), encodes the *whole* prompt into the backbone (nothing teacher-forced, unlike `sample`), then generates the response word by word until `<END>`.

### Dataset (`src/batches.rs`)

Both training modes stream `TRAIN_DATA` through `ChunkedWordDataSet`, which reads roughly `CHUNK_BYTES` of raw text at a time and tokenizes + word-windows it into a `WordChunk`. The raw documents come from a `TextSource`, picked by file extension:

- **text** (anything but `.parquet`) — a single file with `<|endoftext|>` document separators; each read is cut at the last complete document and the partial tail is carried into the next chunk.
- **`.parquet`** — one row per document, read from the `text` column (override with the `PARQUET_TEXT_COLUMN` env var). Row groups are already document-aligned, so there is no separator scanning and no carry; a chunk is however many row groups it takes to reach `CHUNK_BYTES`.

`config::ALLOWED_LANGUAGES` filters a parquet corpus by its `language` column (`PARQUET_LANGUAGE_COLUMN`): rows whose ISO code is not listed are dropped *before* tokenizing, which is the expensive part. The language column decodes alongside the text in the same row group, so the two stay row-aligned. An empty list disables filtering; a corpus lacking the column is read unfiltered with a printed note rather than failing, and plain-text corpora are never filtered (no per-document language exists). Filtering out an entire row group is not end-of-corpus — the loader keeps advancing until the row groups actually run out.

Both hand the loader the same thing — a batch of *complete* documents — so windows never cross a document border either way.

`src/parquet.rs` is a dependency-free reader covering just the slice of the format these corpora use: flat schema, one BYTE_ARRAY column, REQUIRED or OPTIONAL (nulls skipped), UNCOMPRESSED/SNAPPY, data pages v1 and v2, PLAIN and dictionary encodings, RLE definition levels. Anything outside that returns an `Err` naming what it hit rather than silently decoding garbage. `cargo run --release --example parquet_demo -- <file.parquet> [column] [--all]` dumps row/group counts, the first documents, and decode throughput (`--all` streams the whole file — the end-to-end check for a new corpus). Windows never cross document borders, so a streamed epoch yields exactly the windows a whole-file load would — but peak memory is bounded by `CHUNK_BYTES`, not corpus size (>1 GB corpora stream fine). Training loops call `rewind()` per epoch and grow the model cache on demand when a chunk's `max_window_tokens()` exceeds the current size; `count_windows()` does a cheap counting pass for resume arithmetic. `PreparedDataSet` (token-window loading from `DATA_DIR`) still exists but is currently unused by training.
