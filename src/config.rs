// Sequenz-Len

pub const SEQ_LEN: usize = 512 * 1;
pub const MAX_SEQ_LEN: usize = SEQ_LEN + 128;

// Word-grouped training (both the flat and the hierarchical model train on
// these K-word windows, so WORDS_PER_SEQ is the one binding knob).
pub const WORDS_PER_SEQ: usize = 1024 * 4; // K — words per window / backbone unroll length
pub const MIN_WORDS_PER_SEQ: usize = 8; // keep a trailing window only if >= this

/// Cap on the bytes of a single word (see `crate::segment`). Words longer than
/// this — a giant identifier, a base64 blob, a huge indent block — are chopped
/// into pieces, which bounds the decoder's per-word unroll.
pub const MAX_WORD_BYTES: usize = 16;

/// Safety cap on tokens per word-window. Deliberately generous: WORDS_PER_SEQ
/// is meant to bind first — this only guards against a pathological run with no
/// boundary token. Caches are sized to the *actual* longest window
/// (`WordChunk::max_window_tokens`), never to this cap, so raising it is free.
pub const MAX_WINDOW_TOKENS: usize = WORDS_PER_SEQ * 7;

// Training-Schedule

pub const LR: f32 = 1e-4;
pub const MIN_LR: f32 = 1e-5;
// Warmup/decay horizons are counted in *windows* (data seen), not optimizer
// steps, so changing BATCH_SIZE reshapes how often the LR is updated but not
// the curve it follows over the corpus.
//
pub const WARMUP_WINDOWS: usize = 1_200;
pub const DECAY_WINDOWS: usize = 1_500_000;
// Windows whose gradients are accumulated before one optimizer step. Muon
// (matrices) is scale-invariant via the Frobenius normalization and aux-Adam
// (vectors) via its second moment, so summed grads need no manual rescaling.
pub const BATCH_SIZE: usize = 2;
pub const EPOCHS: usize = 1;

pub const SAVE_EVERY: usize = 1000;
pub const LOG_EVERY: usize = 100;

// Per-stack decoupled weight decay (λ), passed at optimizer-step time. `0.0`
// makes a stack plain Adam; a positive λ makes it AdamW (decoupled decay on the
// interior projection matrices only — embeddings, logit heads, biases and norm
// scales are never decayed). This lets the hierarchical character stacks
// (encoder/decoder) train as Adam while the backbone stays AdamW.
pub const ENCODER_WEIGHT_DECAY: f32 = 0.01;
pub const BACKBONE_WEIGHT_DECAY: f32 = crate::optimizers::WEIGHT_DECAY;
pub const DECODER_WEIGHT_DECAY: f32 = 0.01;

/// Weight decay for the flat (non-hierarchical) model's optimizer step. Kept at
/// `0.0` to preserve the current plain-Adam behavior; raise it to run the flat
/// model as AdamW.
pub const FLAT_WEIGHT_DECAY: f32 = 0.0;

// Sampling

pub const MAX_LEN: usize = 2000;
pub const TEMPERATURE: f32 = 0.5;
pub const TOP_P: f32 = 0.9;

// Modell-Dimensions

pub const CHAR_HIDDEN: usize = 256;
pub const OUT_HIDDEN: usize = 256;
pub const WORD_HIDDEN: usize = 768;

/// Output-logit soft cap (xLSTM-7B uses 30): logits = cap · tanh(z / cap).
/// Bounds the logits and removes the cross-entropy incentive for unbounded
/// head-weight growth on the no-decay Adam path.
pub const LOGIT_SOFTCAP: f32 = 30.0;

/// Number of mLSTM backbone blocks in the hierarchical word model.
pub const WORD_BLOCKS: usize = 32;

/// Backbone sweep chunk length, in words. `0` disables chunking (one whole-sequence
/// sweep, the pre-chunking behaviour).
///
/// The backbone runs one row per word across every block, and its activations must
/// survive from a block's forward to its backward — so an unchunked sweep holds
/// `O(words)` per block and device memory scales with the window. Chunking the row
/// axis makes that `O(BACKBONE_CHUNK)` instead: the sweep runs chunk by chunk, each
/// chunk passing through all blocks with the recurrent state carried across the chunk
/// borders, so only the chunks in flight are resident.
///
/// Sized above the sLSTM's `GRAPH_MIN_T` (32) so every chunk still takes the captured-
/// graph path — the backbone is launch-bound at batch 1, and dropping to the per-step
/// loop would cost far more than the memory is worth.
pub const BACKBONE_CHUNK: usize = 512;

/// Largest encoder/decoder group, in rows (`words_in_group × tmax`). `0` disables the
/// cap (one group per length bucket, the pre-cap behaviour).
///
/// The encoder and decoder run length-bucketed groups strictly one after another, and a
/// group's activations die before the next one starts — so the resident set is one
/// group, not the window. But a bucket holds *every* word of that length in the window,
/// so the largest group grows with the window and every buffer in the stage sizes to it
/// (`trim_pools`). Measured 66 -> 264 MB (encoder) and 80 -> 322 MB (decoder) going from
/// 512 to 2048 words.
///
/// Splitting an oversized bucket into several same-shape sub-groups bounds that at the
/// cap instead. It changes no arithmetic: the groups were already independent — each is
/// its own rectangle, and the per-group grads accumulate — so a split is a pure batching
/// choice, exactly like `BACKBONE_CHUNK` on the backbone.
///
/// Sized so the rectangle still fills the device: too small and the stage becomes
/// launch-bound at batch 1 per word, which costs far more than the memory is worth.
pub const GROUP_MAX_ROWS: usize = 2048;

/// GPU mLSTM: chunk length for the chunkwise formulation, or `0` for the
/// single-chunk (whole-sequence) form.
///
/// The single-chunk parallel form materializes `[heads, T, T]` matrices, so it is
/// O(T²) in both time and memory — fine for the encoder/decoder (T = word length,
/// ≤ MAX_WORD_BYTES) but quadratic exactly where the backbone lives (T = the
/// window's word count). Chunking splits the sequence into `T/L` chunks of length
/// `L`, carrying the recurrent state `(C, n, m)` across them: the attention core
/// is then `[heads, L, L]` per chunk, i.e. O(T·L) — linear in T.
///
/// The math is an exact refactoring, not an approximation: the chunk-local
/// stabilizer `max(max_{j in chunk} logD_tj, fc_t + m_prev)` telescopes to the
/// single-chunk global row-max, so both paths agree to floating-point tolerance
/// (`mlstm_chunking_matches_single_chunk` pins this). A cell whose T is already
/// ≤ L takes the single-chunk path unchanged.
///
/// Override at runtime with `MLSTM_CHUNK=<L>` (0 = single-chunk) for A/B runs.
///
/// 256 measured fastest on the RTX 4050 at the backbone shape (B=1, d=512, 16
/// heads): at T=2048 it is 18.9 ms/iter against 63.5 for single-chunk, and the
/// scaling is linear (T=1024 → 9.3 ms) instead of quadratic. Below ~64 the chunk
/// loop becomes launch-bound; above ~512 the [L, L] matrices start to dominate
/// again. `cargo run --release --features cuda --example mlstm_chunk_bench`.
pub const MLSTM_CHUNK: usize = 256;

/// Append a closing `[W]` end-of-word step to every encoder word and read the
/// word embedding `e_w` out at that step (the state then knows the word is
/// complete). Set to `false` to evaluate checkpoints trained without it
/// (readout at the last char, the old behavior).
pub const ENC_W_EOS: bool = true;

// Dataset

/// Bytes of raw corpus text loaded per streaming chunk. Each chunk covers only
/// complete documents and is tokenized + windowed independently, so peak
/// dataset memory scales with this constant — not with the corpus size.
pub const CHUNK_BYTES: usize = 32 * 1024 * 1024;

/// Languages kept from a parquet corpus, as the ISO codes its `language` column
/// uses (FineWeb-style dumps: `"en"`, `"de"`, ...). A document whose language is
/// not listed is dropped before tokenizing.
///
/// An empty list disables filtering and keeps every row — which is also what
/// happens for corpora with no `language` column, and for plain-text corpora,
/// where there is no per-document language to filter on.
pub const ALLOWED_LANGUAGES: &[&str] = &["en", "de"];

/// Column holding the per-document language code. Only consulted when
/// `ALLOWED_LANGUAGES` is non-empty; a corpus without it is read unfiltered.
pub const PARQUET_LANGUAGE_COLUMN: &str = "language";

/// Corpus path. A `.parquet` extension selects the parquet reader (one row per
/// document, column `text` — override with `PARQUET_TEXT_COLUMN`); anything else
/// is read as plain text with `<|endoftext|>` document separators.
pub const TRAIN_DATA: &str = "../../training_data/000_00002.parquet";
pub const VAL_DATA: &str = "../../training_data/TinyStoriesV2-GPT4-valid.txt";

// Post-training (SFT / instruction tuning)

/// Instruction dataset (JSONL: `{instruction, context, response, category}`,
/// databricks-dolly-15k layout). Formatted into masked chat windows by
/// `crate::sft`; the loss counts only the response tokens.
pub const SFT_DATA: &str = "/home/unschlagbar/training_data/databricks-dolly-15k.jsonl";
/// Passes over the SFT set. A few epochs is typical for instruction tuning on a
/// small set; too many overfits the ~15k examples.
pub const SFT_EPOCHS: usize = 1;
/// SFT learning rate — an order of magnitude below pretraining `LR`, so
/// fine-tuning nudges the pretrained weights instead of overwriting them.
pub const SFT_LR: f32 = 5e-6;

/// Cap on the token length of one SFT example. The per-example cache (encoder +
/// decoder, one slot per token) is sized to the LONGEST example, so a single
/// giant record (dolly's tail runs to ~27k tokens) would blow up memory for the
/// whole run. Examples longer than this are dropped when the set is loaded —
/// 2048 keeps ~93% of dolly while bounding the cache. Raise it if you have the
/// RAM/VRAM; lower it if you still OOM.
pub const SFT_MAX_TOKENS: usize = 2048;

// Wake Word

pub const WAKE_HIDDEN: usize = 128;
pub const WAKE_SR: usize = 16_000;
pub const WAKE_FRAME_LEN: usize = 320;
pub const WAKE_FRAME_SHIFT: usize = 320;
pub const WAKE_N_FFT: usize = 512;
pub const WAKE_N_MELS: usize = 80;
pub const WAKE_INPUT_DIM: usize = WAKE_N_MELS;
pub const WAKE_THRESHOLD: f32 = 0.6;
pub const WAKE_POS_WEIGHT: f32 = 1.0;
pub const WAKE_LR: f32 = 1e-3;
pub const WAKE_EPOCHS: usize = 35;
pub const WAKE_MODEL_LOC: &str = "models/wake_word3";
pub const WAKE_DATA_POS: &str = "data/wake_word/positive";
pub const WAKE_DATA_NEG: &str = "data/wake_word/negative";
