# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
cargo build [--release]            # release: fat LTO, single codegen unit
cargo run [--release] [--features cuda]
cargo check --features cuda --examples --tests
cargo test --release --features cuda --lib -- <module>::   # narrow filter, never the whole suite
```

The binary reads one line from stdin to select its mode:

| Input | Action |
|-------|--------|
| `h` / `hg` | pretrain the hierarchical model on CPU (`cpu::hierarchical`) / GPU (`--features cuda`) |
| `hga` | GPU audio stage after text pretraining: an audio window (LibriSpeech, voice or TTS takes heard through `<model>.ae2`) every `AUDIO_PRETRAIN_EVERY` text windows, backbone at `AUDIO_STAGE_LR`; trains `<model>_audio` |
| `hq` / `hqg` | SFT on `SFT_DATA`, loss masked to the response, CPU / GPU |
| `hv` / `hvg` | forward-only validation on `VAL_DATA` (mean char and word loss), CPU / GPU |
| `hs` | interactive sampling (`src/infer.rs`) |
| `hqs` | interactive multi-turn chat with an SFT model |
| `i` | print a model's structure: v2 widths, blocks, parameter and token counts, or a flat `Sequential`'s layers |

## Style 

do not use `0.0_f32`, always prever the simple mumber: `0.0`
All comments need to be in englisch not in german

Comments describe the code, not the edit. Never write a comment about what you
changed, added, moved, or why you changed it — no "was X, now Y", no "kept for
compatibility", no "moved here from foo.rs", no "NEW:". That belongs in the commit
message or the chat, not in the file: the next reader has no idea what the previous
version looked like and does not care.

Keep comments short. Only comment what is not immediately understandable from the
code — a non-obvious invariant, a unit, a reason the naive approach fails, a
reference to a paper or reference implementation. Do not restate what the line
already says, and do not write a paragraph where a line does.

No banner or separator lines in comments: no `// =====`, no `// -----`, no boxed
headers. They add lines without adding information and make the file look more
verbose than it is. A section heading is one plain comment line.

## Architecture

### Checkpoints (`src/checkpoint.rs`)

Format v2: a header (`ModelCfg` widths, a `Layout` with every block's `BlockKind` as a one-byte tag, step, `Seen`), then every parameter tensor in `param_slots` order as rank + dims + raw f32, in the GPU layouts (fused projections, the mLSTM's 1/√dqk folded into `wk`/`bk`). A new cell type is a new tag. The GPU model reads/writes it from the device; `infer` and `cpu::hierarchical` convert on load and save. `examples/gpu_cpu_parity.rs` checks CPU against GPU loss on a real checkpoint. Every hierarchical mode reads `models/<name>.v2`.

### Hierarchical model (`src/cpu/`, `src/gpu/`, `src/infer.rs`)

HAT-style (arXiv 2501.10322), three stages run phase by phase over a window of words (all encodes, backbone sweep, all decodes):

- **encoder**: the tied char table and sLSTM/mLSTM blocks over one word's chars plus a virtual closing `[W]` step; `e_w` is the output at `[W]`. State resets per word.
- **backbone**: `Linear → blocks × WORD_BLOCKS → Linear` over the word embeddings, state carried across words (s4/s5 have sLSTM blocks too; the `Layout` records each). Its output is the context for decoding the next word.
- **decoder**: `blocks → RMSNorm → head → SoftCap`. Step 0 is the injected backbone context; later steps feed the previous char through the tied char table (CHAR_HIDDEN == OUT_HIDDEN). Predicts the chars plus a trailing `[W]`.

`gpu::Hierarchical` trains. `cpu::Hierarchical` (batched `[B, T, F]` layers on `crate::tensor`, hand-written backward) is the CPU twin, the GPU's parity reference and init source; it does not fit the full config and is for small models, SFT and checks. `infer::Hierarchical` is forward-only, one position at a time, and is what `hs`, `hqs` and `harness` run.

Interior projections are weight-decayed; embeddings and the logit head are not, and the head is followed by `SoftCap` (`LOGIT_SOFTCAP = 30`) so it cannot grow without bound.

### Sequential (`src/sequential/`)

A flat per-timestep layer stack used by `smart-light`'s wake-word detector. Layers own weights and gradients and step on per-tensor AdamW (`optim.rs`). The `NNM1` file format (`io.rs`) must stay byte-compatible.

### Optimizer (`src/gpu/arena.rs`)

CPU: AdamW per layer (`cpu::optim`). `BATCH_SIZE` windows accumulate per step. GPU: every parameter, gradient and moment lives in a **parameter arena**, one per stage (encoder incl. char table, backbone, decoder) so each stage gets its own `AdamCfg` (`ENCODER_LR_SCALE`, `DECODER_LR_SCALE`, per-stack weight decay). Layers hold non-owning windows into it; a step is one `adamw_arena_q8` launch per arena, and every weight has a fixed device address. Layers enumerate their tensors once in `param_slots` with a `ParamKind` (`Decay`/`NoDecay`/`Frozen`). `bind` carries parameters across and gives back zeroed grads and moments; nothing allocates what the arena will provide. Moments are **8-bit** (arXiv 2110.02861): a byte per element into a 256-entry non-uniform map, one fp32 scale per 2048-element block, dequantized into registers per step; `v` rounds up. The fp32 reference is `arena::step_slots`.

### Saved activations and offload (`src/gpu/offload.rs`)

Everything a GPU block's backward reads lives in one **`Frame`**, a byte range carved by a `carve(frame, b, t)` that forward and backward both call. What is cheaper to recompute is not saved (`SiLU(gate_pre)`, the mLSTM's `σ(o)⊙yhat`, and in offloaded blocks the FFN, rebuilt from `norm2_out`). `FrameStack` is either resident (one buffer per stack depth) or `Offload` (backbone only): its frames form one LIFO, held in a pinned host stack behind a ring of `RING_SLOTS = 3` device slots, written through on forward and prefetched two ahead on backward. The forward is PCIe-bound, so frame bytes are step time.

### Tokenizer and words (`wordseg/`)

Own crate, because the trainer and `datamix` must agree byte for byte. `Utf8Tokenizer`: ids `0..256` are bytes, `256..` the `SPECIAL_TOKENS` (`<W>`, `<END>`, then the SFT markers), `vocab_size() == 266`; a pretrained checkpoint has 258. `segment::word_ends` is a lexer-shaped split tuned for Rust: whitespace attaches as a suffix to the word before it, identifiers/numbers/multi-byte operators stay whole, each special token is its own word, words capped at `MAX_WORD_BYTES`. `cargo run --example seg_demo -- <file>` prints a split.

### SFT (`src/sft.rs`, `src/training.rs`, `src/gpu/train.rs`)

Chat markers come after `<W>`/`<END>` so no pretrained id shifts: `<CONTEXT>`, `<SEP>`, then markup `<tool>`/`</tool>`, `<result>`/`</result>`, `<think>`/`</think>`, which `to_tokens_markup` also folds out of literal text (pretraining too). `<END>` ends a response. `load_jsonl` reads dolly records (`{instruction}[<CONTEXT>{context}]<SEP>{response}<END>`) or `messages` conversations: assistant turns carry loss, user and `tool` turns (rendered `<result>…</result><SEP>`) do not, `system` takes the `<CONTEXT>` slot, `assistant_context` turns and `no_loss` byte ranges are written but never trained (how a wrong call followed by its recovery is taught). The mask is per word; the whole sequence is still encoded. Examples over `SFT_MAX_TOKENS` are dropped. Both trainers mask the same way (`forward_backward_masked`, loss normalized by the unmasked rows). `hqs` keeps the conversation, prompts for a tool result whenever a reply contains the `<tool>` token, up to `MAX_TOOL_ROUNDS`.

The tool surface is a convention of the data (`mixes/synth/*.syn`) and must match the runtime in `../harness/src/tools/`: `lamp.set(on=…)`, `lamp.state()`, `app.launch(name=…)`, `app.list()`, `browser.open(url=…)`, `media.control(action=…)`, `math.eval(expr=…)` (one infix expression string, `+ - * / ^`, parentheses, `sqrt/abs/min/max/round/floor/ceil`, no units). `apps.syn` is generated by `scripts/gen_apps_syn.py` from `datamix/pools/installed_apps.json`.

### Dataset mixing (`datamix/`, `mixes/`)

```bash
cargo build -p datamix --release
./target/release/datamix check|build mixes/<mix>.toml
./target/release/datamix sample mixes/<mix>.toml -n 8
./target/release/datamix verify data/mix/assistant.jsonl
./target/release/datamix synth mixes/synth/apps.syn -n 10
./target/release/datamix edit data/sources/handmade.jsonl   # browser editor, port 7878
./target/release/datamix ping [mix.toml]
```

A mixture (`mixes/*.toml`, hand-parsed TOML subset, schema in `datamix/mixture.schema.json`) names sources, weights and filters. A build streams each source through its filters into a staging shard, then draws each source's share of the budget (`weight_by` tokens or records; `epochs` caps repeats, and the binding source is reported). Outputs: `sft` JSONL, `text` (`<|endoftext|>`-separated), `none`. Inputs: `file`, `dir`, `parquet`, `chat`, `dolly`, `jsonl`, `synth`, `llm`. Filters: size/token caps, quality ratios, per-role substring blocklists (`^` anchors at turn start), `dedup`, an LLM `judge` (temperature 0, verdict first), a `transform` rewrite (same turns and roles back or refused), `verify = "rust"` with one repair pass. LLM calls go to a local OpenAI-compatible server (`[llm]`, hand-rolled HTTP client, `workers` in flight, `completion_template` to prefill an empty `<think>`) and are cached under `data/llm-cache`. Raw corpora sit in `~/training_data` (`scripts/fetch_sft_data.sh`). Main mixes: `assistant_full.toml` (full SFT corpus), `tools.toml` (tool-calling corpus into `data/sources/tools.jsonl`), `rustified.toml` (Python→Rust rewrite into `data/sources/rustified.jsonl`).

### Dataset (`src/batches.rs`, `src/parquet.rs`)

`ChunkedWordDataSet` streams `TRAIN_DATA` in `CHUNK_BYTES` chunks from a text file (`<|endoftext|>` separators) or parquet (`text` column, `ALLOWED_LANGUAGES` filtered on `language`; dependency-free reader for flat BYTE_ARRAY columns, PLAIN/dictionary, UNCOMPRESSED/SNAPPY). A chunk's documents are packed into one stream, each closed by `<END>`, and cut into `WORDS_PER_SEQ`-word windows overlapping by one word. With `CARRY_WINDOW_STATE` the GPU pretrainer carries the backbone state across windows (truncated BPTT). `WordBatch::doc_starts` → `set_doc_starts`: every backbone cell restarts at a document border in place (sLSTM exactly; mLSTM by overwriting the border row's forget logit with `DOC_RESET_FGATE = -100`), and the document's first word and the closing `<END>` carry no loss. SFT stays unpacked and stateless. Packing makes window boundaries depend on `CHUNK_BYTES`; the `.pretrainprog` resume sidecar checks the window count.
