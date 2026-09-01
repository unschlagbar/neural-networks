// Sequenz-Len

pub const SEQ_LEN: usize = 512 * 1;
pub const MAX_SEQ_LEN: usize = SEQ_LEN + 128;

// Word-grouped training (both the flat and the hierarchical model train on
// these K-word windows, so WORDS_PER_SEQ is the one binding knob).
pub const WORDS_PER_SEQ: usize = 1024 * 4; // K — words per window / backbone unroll length
pub const MIN_WORDS_PER_SEQ: usize = 8; // keep a trailing window only if >= this

// The word split's own constant: it defines what a word is, so it lives with
// the splitter and is re-exported here for the call sites that read it as a
// model dimension.
pub use wordseg::MAX_WORD_BYTES;

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
pub const BATCH_SIZE: usize = 8;
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

pub const CHAR_HIDDEN: usize = 192;
pub const OUT_HIDDEN: usize = 192;
pub const WORD_HIDDEN: usize = 512;

/// SwiGLU inner width for a block of width `hidden`: the `8·hidden/3` paper
/// default rounded up to a multiple of 64, so every up/down projection GEMM has
/// a tile-aligned inner dimension (128 -> 384, 1024 -> 2752).
#[inline]
pub fn up_of(hidden: usize) -> usize {
    let up = hidden * 8 / 3;
    up.div_ceil(64).max(1) * 64
}

/// Output-logit soft cap (xLSTM-7B uses 30): logits = cap · tanh(z / cap).
/// Bounds the logits and removes the cross-entropy incentive for unbounded
/// head-weight growth on the no-decay Adam path.
pub const LOGIT_SOFTCAP: f32 = 30.0;

/// Number of mLSTM backbone blocks in the hierarchical word model.
pub const WORD_BLOCKS: usize = 12;

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
/// Sized above the sLSTM's `FUSED_MIN_T` (32) so every chunk still runs its T-loop as
/// one time-fused launch — the backbone is launch-bound at batch 1, and dropping to
/// the per-step loop would cost far more than the memory is worth.
///
/// It also sets how many times each backbone cell is invoked per step
/// (`words / BACKBONE_CHUNK` per layer per window), and a shorter chunk runs those
/// cells at a shorter `T`, where the fused mLSTM kernels are further from their
/// tuned shape. Measured at 4096 words, 3 interleaved repeats (ms/step, peak MiB of
/// 16303): 512 -> 566 / 9469, 1024 -> 536 / 11549, 2048 -> 531 / 14939, 4096 OOMs.
///
/// `GPU_BACKBONE_CHUNK` overrides it for an A/B.
pub const BACKBONE_CHUNK: usize = 512;

/// Largest encoder/decoder group, in rows (`words_in_group × tmax`). `0` disables the
/// cap (one group per word length, the pre-cap behaviour).
///
/// The encoder and decoder run length-grouped rectangles strictly one after another, and
/// a group's activations die before the next one starts — so the resident set is one
/// group, not the window. But a group holds *every* word of that length in the window,
/// so the largest group grows with the window and every buffer in the stage sizes to it
/// (`trim_pools`), and pooled buffers never shrink: one unusual window raises the floor
/// for the whole run.
///
/// Splitting an oversized group into several equal sub-groups bounds that at the cap
/// instead. It changes no arithmetic: the groups were already independent — each is its
/// own rectangle, and the per-group grads accumulate — so a split is a pure batching
/// choice, exactly like `BACKBONE_CHUNK` on the backbone.
///
/// Exact-length grouping already keeps the typical peak near this value (measured over
/// 4096-word windows of Rust source and Markdown: largest group 2500-3310 rows, so the
/// cap fires on 2-4 of ~16 lengths). What it still bounds is the worst case — a corpus
/// of uniform word length puts all `WORDS_PER_SEQ` words in one group, 69632 rows at the
/// current settings, several GB the run would then carry forever.
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
/// Override at runtime with `MLSTM_CHUNK=<L>` for A/B runs. The fused kernels cap
/// the effective L at `ops::FUSED_MAX_L`, so values above that are clamped, and 0 —
/// once "one chunk over the whole sequence" — now means L = 1.
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
// Default corpus offered when a run has no progress sidecar to continue from.
// A directory is walked shard by shard (see `pretrain_progress::Corpus`); a
// single file still works as a one-file corpus.
pub const TRAIN_DATA: &str = "data/mix/pretrain_v2";
pub const VAL_DATA: &str = "../../training_data/TinyStoriesV2-GPT4-valid.txt";

// Post-training (SFT / instruction tuning)

/// Instruction dataset (JSONL: `{instruction, context, response, category}`,
/// databricks-dolly-15k layout). Formatted into masked chat windows by
/// `crate::sft`; the loss counts only the response tokens.
pub const SFT_DATA: &str = "data/mix/assistant_qa.jsonl";
/// Passes over the SFT set. A few epochs is typical for instruction tuning on a
/// small set; too many overfits the ~15k examples.
pub const SFT_EPOCHS: usize = 1;
/// SFT learning rate — an order of magnitude below pretraining `LR`, so
/// fine-tuning nudges the pretrained weights instead of overwriting them.
pub const SFT_LR: f32 = 5e-6;

/// Cap on the **words** of one SFT example — the unit the model actually
/// unrolls in: one backbone step and one decoder rollout per word. Matching
/// `WORDS_PER_SEQ` makes an SFT window the same shape as a pretraining window,
/// so a run that fits `hg` fits `hqg`.
pub const SFT_MAX_WORDS: usize = WORDS_PER_SEQ;

/// Safety cap on the tokens of one SFT example. `SFT_MAX_WORDS` is meant to
/// bind first; this only guards against a pathological record whose words are
/// all `MAX_WORD_BYTES` long. Sized like `MAX_WINDOW_TOKENS` for the same
/// reason: the per-example cache holds one slot per token.
pub const SFT_MAX_TOKENS: usize = MAX_WINDOW_TOKENS;

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
