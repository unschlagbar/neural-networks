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
| `i` | `inspect_model` — prompts for a model name, looks it up in `models/` and prints all layers with their settings, plus how many chars/words it has trained on (pretraining and SFT counted separately) |

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

The GPU path runs AdamW out of a **parameter arena** (`src/gpu/arena.rs`): every parameter, gradient and AdamW moment lives in one of four contiguous allocations laid out identically, and each layer holds windows (non-owning `DTensor`s) into them. A step is therefore one `adamw_arena` launch plus one memset over the gradients instead of ~900 of each, and every weight keeps a fixed device address — the prerequisite for capturing a step in a CUDA graph. Each layer enumerates its tensors exactly once, in `param_slots`, tagging each with a `ParamKind` (`Decay` / `NoDecay` / `Frozen`); the arena packs them in that order so the decay term and the extent of the update are bound checks on the element index. `params_mut` (checkpointing) and `grads` (diagnostics) are derived from the same enumeration, and a standalone layer or a parity test steps its own slots eagerly through `arena::step_slots`.

### Tokenizer and word segmentation (`wordseg/`)

The byte tokenizer and the word split live in their own dependency-free workspace crate, because both sides of the pipeline must agree on them byte for byte: the model trains on words, and `datamix` caps a corpus in the same words the trainer will see. `neural-networks` re-exports both modules (`pub use wordseg::{segment, tokenizer_utf8}`), so `crate::segment::…` and `crate::tokenizer_utf8::…` still resolve everywhere inside it, and `MAX_WORD_BYTES` now lives with the splitter it defines (re-exported from `config`).


`Utf8Tokenizer` (`wordseg/src/tokenizer_utf8.rs`) is byte-level: ids `0..256` are raw UTF-8 bytes, ids `256..` are the specials in `SPECIAL_TOKENS` (`<W>` = end-of-word marker, `<END>`, plus the SFT markers below), so `vocab_size() == 256 + SPECIAL_TOKENS.len()` (266; a pretraining-only model uses just the first two). Any UTF-8 text round-trips losslessly and no input byte can collide with a special. There is no charset file. Sampling decodes with `Utf8Printer` (`src/sampling.rs`), which holds bytes back until they form a whole character.

`segment::word_ends` (`wordseg/src/segment.rs`) decides what a "word" is — the unit the backbone autoregresses over — with a lexer-shaped split tuned for Rust: a whitespace run is one unit and attaches as a *suffix* to the word before it (`"use "`, `";\n    "`), so a word carries the separator that closes it and the decoder emits that separator right before `[W]`; identifiers/keywords and numbers stay whole (`foo`, `1_000u32`), multi-byte operators are one word (`::`, `->`, `..=`, `//`), lifetimes (`'a`) differ from char literals (`'a'`), and non-ASCII bytes group into their character. Words tile the sequence contiguously and are capped at `MAX_WORD_BYTES` (config), which bounds the decoder unroll. Roughly 3–4 bytes per word on Rust source. `cargo run --example seg_demo -- src/foo.rs` prints the split for a file.

### Post-training / SFT (`src/sft.rs`, `src/grow_vocab.rs`, `src/gpu/train.rs`)

Q-A instruction tuning of a pretrained hierarchical model, GPU only. Three pieces:

- **Chat special tokens.** `SPECIAL_TOKENS` gains, *after* the pretraining `<W>`/`<END>` so no existing byte/`<W>`/`<END>` id ever shifts: the structural markers `<CONTEXT>` (258) and `<SEP>` (259), then the **markup** markers `<tool>`/`</tool>` (260/261), `<result>`/`</result>` (262/263) and `<think>`/`</think>` (264/265). `vocab_size()` is therefore 266, while a pretrained checkpoint has 258 (`PRETRAIN_SPECIALS`). Markup differs from the structural markers in that it also occurs verbatim in the data: `to_tokens_markup` folds each literal occurrence into its token, `to_text_markup` writes it back out, and `MARKUP_START` is where the markup range begins. `ChunkedWordDataSet` tokenizes pretraining documents through it too, so a literal `<tool>` in a web page or a source file is the same one token there as in an SFT window — measured on the shipped corpus that is ~270 occurrences in 34.4G tokens, so the point is one representation everywhere, not volume. A call boundary is therefore one token the model either emits or does not, instead of a spelling it can get subtly wrong, and `sample_chat` detects a call by the `<tool>` **token** rather than by a substring. What sits *between* the markers is still plain text, so the call syntax itself stays a convention of the data alone (`mixes/synth/*.syn`) and a new protocol costs no vocab entry. A **tool result** still needs no role marker: `Role::Tool` renders as `<result>…</result>` followed by `<SEP>`, so it sits on the prompt side exactly like a user turn and the reply after it is what carries loss. The pretraining `<END>` (257) doubles as the SFT end-of-response target/stop.
- **Growing the vocab (`grow_vocab.rs`, mode `av`) — only for old checkpoints.** New models are built at `tokenizer.vocab_size()`, so they already carry the SFT marker rows and go straight into `hqg`. `av` exists solely to upgrade a checkpoint pretrained *before* the markers were added: its two vocab-sized tables — the encoder's tied char **embedding** (`[vocab, HC]`, rows) and the decoder's logit **head** (`LinearNoBias`, `[HC, vocab]`, cols; its trailing `SoftCap` is resized too) — have no rows/cols for the new ids. `grow_checkpoint` appends them: new embedding rows get a small random init, new head columns start at **zero** (neutral initial logit), every existing weight preserved byte-for-byte, backbone untouched (it never sees token ids). `hqg` detects a stale-vocab checkpoint and points at `av`; a current model never triggers it.
- **SFT dataset (`sft.rs`).** Reads a JSONL in either of two shapes (dependency-free string parser — no serde), one window per record. A dolly record (`{instruction, context, response, category}`) becomes `{instruction}[<CONTEXT>{context}]<SEP>{response}<END>`. A record carrying a `messages` array (`{"role", "content"}`, the OpenAI/LM-Studio shape) becomes a **conversation** — `user₁<SEP>asst₁<END>user₂<SEP>asst₂<END>…` — where every assistant turn carries loss and every user turn does not; a `system` message takes the `<CONTEXT>` slot of the first turn, so multi-turn needs no new vocab and no checkpoint surgery. A `tool` message is what a call returned: prompt-side like a user turn, so the model is trained to *read* a result and answer from it (`<tool>lamp.set(on=true)</tool><END><result>already_on</result><SEP>It's already on.<END>`) rather than to invent one. An `assistant_context` message is an assistant turn laid out exactly like a normal one, `<END>` included, that records **no loss span** — so a conversation can contain a wrong tool call the model then recovers from, and only the recovery is trained. That is the whole mechanism for error handling: the mistake is context the backbone reads, never an output any gradient pushes it towards. `parse_messages`/`build_example_turns` handle the second shape, `segment_with_spans` builds the multi-span mask, and `load_jsonl` picks the shape per line. Both mask paths downstream (CPU `dec_word_loss`, GPU `forward_backward_masked`) are already per-word, so several loss spans need nothing new. `segment::word_ends` already makes each special token its own one-token word, so the per-word **loss mask** is a clean flag: words at/after the response carry loss (`true`), the prompt (through `<SEP>`) does not. The whole sequence is still encoded/decoded — the backbone must read the prompt — but gradient flows only from the response. Examples longer than `SFT_MAX_TOKENS` (4096) are dropped at load: the per-example cache is sized to one slot per token, so dolly's ~27k-token tail would otherwise force a ~16 GB cache (~2.4 GB at 4096) and OOM before the first step. `datamix`'s `trim_turns` keeps a long conversation's leading exchanges instead of dropping it — at 2048 that was the difference between 35% and 82% of SmolTalk.
- **Training — two paths, same masking.** Both load the whole (small) SFT set into memory, shuffle per epoch, reject a stale-vocab checkpoint with a pointer to `av`, and fine-tune at `SFT_LR` (1e-5, ×10 below pretraining) over `SFT_EPOCHS`.
  - **GPU (`train_sft_gpu`, mode `hqg`)** calls `gpu::Hierarchical::forward_backward_masked`, which mirrors `forward_backward` but takes a per-decoded-word mask (`word_loss[w]` gates the CE mask of every decoder slot of word `w+1`) and normalizes the loss by the **unmasked** row count.
  - **CPU (`train_sft`, mode `hq`)** calls `Hierarchical::train_sft`, which sets a per-window `dec_word_loss` mask consumed by `backwards_sequence`: a masked word's decoder slots seed a **zero** delta (BPTT still unwinds, but no gradient flows to logits, context or the tied table) and are dropped from `decode_loss`. `hierarchical.rs` pins this with two tests — a fully-masked window gives exactly zero head gradient, and the per-word masks sum to the full-window gradient (masking is additive and isolating).
- **Inference (`sample_chat`, mode `hqs`).** A multi-turn REPL: it keeps the conversation as `Vec<Turn>`, formats it with `sft::format_chat_prompt` (every completed exchange, then the pending user turn and its `<SEP>`), encodes the *whole* prompt into the backbone (nothing teacher-forced, unlike `sample`), generates until `<END>`, and appends the reply as an assistant turn. A reply containing `<tool>` is not an answer but a request for a result: the REPL prompts for one, appends it as a `Role::Tool` turn and generates again, up to `MAX_TOOL_ROUNDS` — which is what lets `app.launch` → `not_found` → `app.list` → `app.launch` play out. Context is asked once per conversation and lands in the `<CONTEXT>` slot; `new` clears the history. `sft::format_prompt` remains the single-turn form.

### Dataset mixing (`datamix/`, `mixes/`)

`datamix` is a workspace member (CPU-only, `default-features = false` on the parent crate) that builds the corpora the trainers read. A **mixture file** (`mixes/*.toml`) names the sources, their weights and the quality gates; nothing is compiled in. The format is TOML, parsed by `datamix/src/toml.rs` — a hand-written subset (tables incl. dotted/quoted names, basic and literal strings with their multi-line forms, integers, floats, booleans, arrays; no inline tables, arrays of tables or dates), because the repo carries no serde. The point of TOML here is tooling: `datamix/mixture.schema.json` (named by a `#:schema` line in each mixture, and wired up in `.vscode/settings.json`) gives key completion and unknown-key warnings in the editor, and `config.rs` re-checks the same things — unknown key, unknown table, wrong type — with `file:line` at build time. `config::tests::the_shipped_mixtures_parse` pins the shipped files, including that their wrapped prompts join into flowing text.

```bash
cargo build -p datamix --release
./target/release/datamix check  mixes/pretrain.toml        # stats only
./target/release/datamix sample mixes/assistant_sft.toml -n 8
./target/release/datamix build  mixes/assistant_sft.toml
./target/release/datamix verify data/mix/assistant.jsonl # loads it via sft::load_jsonl
./target/release/datamix synth  mixes/synth/apps.syn -n 10
```

A build runs in two passes: every source streams through its filters into a staging shard under `target/datamix-stage` (so memory is bounded by one record, not by the corpus), then the mixer draws each source's shuffled records until it has covered its share of the budget. `weight` is a share of the output, counted in the unit `[output] weight_by` names — `tokens` (budget `tokens`) or `records` (budget `records`), one example counting as one whatever its length. On an SFT mix the unit decides the corpus: a tool call is a handful of tokens and a chat conversation is thousands, so a source at 12% of the tokens is most of the questions (`mixes/assistant_qa.toml` is the record-weighted SmolTalk mix, ~1.7% tool calls). `epochs` caps how often a source may repeat. With no explicit `tokens` budget the corpus grows until the first source would exceed its `epochs` — that binding source is named on stdout, because it sets the size of everything. With an explicit budget, any source that cannot fill its share is reported the same way (stdout plus a `capped` column in the report): what it asked for, what its `epochs` allowed, and how far short of the budget that left the corpus.

Output kinds are `sft` (JSONL: dolly-layout records, or `{"messages": [...]}` conversations for multi-turn — both what `hqg`/`src/sft.rs` reads), `text` (`<|endoftext|>`-separated, what `ChunkedWordDataSet` reads — the separator is consumed by its `split`, so no end-of-document token is ever trained) and `none` (report only). Input kinds are `file`, `dir`, `parquet` (via `src/parquet.rs`, with the same `language` column filtering), `chat`, `dolly`, `jsonl`, `synth` and `llm`. `chat` reads a parquet whose messages column is a `list<struct<role, content>>` — SmolTalk and every other HF chat corpus — one `Record::Chat` per row; `select_column`/`select_values` keep only the rows whose flat subset column matches, which is what lets one corpus be split into several weighted sources (SmolTalk's subsets differ by 10x in answer length, so taken whole its longest subset sets the model's register). A `dir` path — or a `parquet` path naming a directory — reads every matching file in sorted order, with `skip_files`/`max_files` cutting the listing down (that is how a sharded corpus already half-trained-on is resumed).

Filters are inherited from `[filter]` and overridable per source: size and token caps, `min_alpha_ratio`, `max_line_bytes`, `max_dup_line_ratio`, substring allow/blocklists, `dedup = off|exact|near` (near = MinHash over word 5-grams; on short formulaic SFT records prefer `exact` — `near` reads two commands differing only in the room as one), and `judge` (see below). Every rejection is counted by reason and printed in the report — `data/rust-lib` is 97.7% byte-identical duplicates (1.2k unique files of 62.8k), which only the report makes visible. An unknown key is an error naming the line.

A **local OpenAI-compatible server** (LM Studio's `http://localhost:1234/v1`, configured in `[llm]`) does two optional jobs, both opt-in — a mixture using neither never opens a socket. `kind = llm` sources generate examples: the reply is read as JSONL (one `{instruction, context, response}` per line, or one `{"messages": [...]}` conversation, `<think>` preludes and code fences stripped), and `seed_file` renders `seeds` examples into `{seed}` per call **through the same `mix::render` the writer uses**, so the model sees the exact output format. That sharing is load-bearing: seeded with bare instructions, a local 20B model invented `turn_lights_off(room='dining room')` instead of the corpus's `<tool>lamp.set(...)</tool>`. A `judge` prompt (in `[filter]` or one source) vets each surviving record — the reply must start with `judge_expect` (default `yes`) — and shows up in the report like any other filter. The judge runs at `judge_temperature` (default **0**, not the generator's temperature) and optionally `judge_model`: measured on 10 generated tool calls, the same judge kept a malformed invented-tool example at temperature 1.0 and rejected it at 0. A judge prompt that names the exact tools, arguments and allowed values catches far more than a general one. Every completion is cached under `target/datamix-llm-cache` keyed on model/temperature/prompt/call index, so a rebuild makes no calls. `datamix ping [mix.toml]` checks the server. The HTTP client is hand-rolled over `TcpStream` (`datamix/src/llm.rs`): one POST with `Connection: close`, http only, every resolved address tried (localhost resolves to ::1 first).

`synth` sources expand template files (`mixes/synth/*.syn`) into instruction data for skills no public corpus covers (the lamp, app launching, the assistant's own identity): a template with `user =` / `assistant =` lines produces a multi-turn conversation whose lists bind once per example, so a later turn can refer back to what the first one named, a `tool =` line is what the call returned — the reply after it is what the model is trained to write — and an `assistant_context =` line is a turn the model reads but is never trained to produce, which is how the error-recovery templates put a wrong call in front of the correction without teaching the wrong call (`mixes/synth/lamp.syn`, `apps.syn`, `tools_in_context.syn`, `persona.syn`); `list` entries carry `;`-separated fields so the spoken form (`{room.0}`) and the tool identifier (`{room.1}`) stay bound to the same entry, and `{a|b|c}` is an inline paraphrase drawn per example (never trimmed, so `{|s}` writes an optional word). The tool-call syntax in the responses is a convention of the data alone — whatever parses it downstream must agree with those files. See `datamix/README.md`.

### Dataset (`src/batches.rs`)

Both training modes stream `TRAIN_DATA` through `ChunkedWordDataSet`, which reads roughly `CHUNK_BYTES` of raw text at a time and tokenizes + word-windows it into a `WordChunk`. The raw documents come from a `TextSource`, picked by file extension:

- **text** (anything but `.parquet`) — a single file with `<|endoftext|>` document separators; each read is cut at the last complete document and the partial tail is carried into the next chunk.
- **`.parquet`** — one row per document, read from the `text` column (override with the `PARQUET_TEXT_COLUMN` env var). Row groups are already document-aligned, so there is no separator scanning and no carry; a chunk is however many row groups it takes to reach `CHUNK_BYTES`.

`config::ALLOWED_LANGUAGES` filters a parquet corpus by its `language` column (`PARQUET_LANGUAGE_COLUMN`): rows whose ISO code is not listed are dropped *before* tokenizing, which is the expensive part. The language column decodes alongside the text in the same row group, so the two stay row-aligned. An empty list disables filtering; a corpus lacking the column is read unfiltered with a printed note rather than failing, and plain-text corpora are never filtered (no per-document language exists). Filtering out an entire row group is not end-of-corpus — the loader keeps advancing until the row groups actually run out.

Both hand the loader the same thing — a batch of *complete* documents — so windows never cross a document border either way.

`src/parquet.rs` is a dependency-free reader covering just the slice of the format these corpora use: flat schema, one BYTE_ARRAY column, REQUIRED or OPTIONAL (nulls skipped), UNCOMPRESSED/SNAPPY, data pages v1 and v2, PLAIN and dictionary encodings, RLE definition levels. Anything outside that returns an `Err` naming what it hit rather than silently decoding garbage. `cargo run --release --example parquet_demo -- <file.parquet> [column] [--all]` dumps row/group counts, the first documents, and decode throughput (`--all` streams the whole file — the end-to-end check for a new corpus). Windows never cross document borders, so a streamed epoch yields exactly the windows a whole-file load would — but peak memory is bounded by `CHUNK_BYTES`, not corpus size (>1 GB corpora stream fine). Training loops call `rewind()` per epoch and grow the model cache on demand when a chunk's `max_window_tokens()` exceeds the current size; `count_windows()` does a cheap counting pass for resume arithmetic.
