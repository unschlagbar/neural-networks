// Sequenz-Len

pub const SEQ_LEN: usize = 512;
pub const MAX_SEQ_LEN: usize = SEQ_LEN + 128;

// Word-grouped training: both models train on these K-word windows, so
// WORDS_PER_SEQ is the one binding knob.
pub const WORDS_PER_SEQ: usize = 1024 * 4; // K — words per window / backbone unroll length
pub const MIN_WORDS_PER_SEQ: usize = 8; // keep a trailing window only if >= this

// Documents are packed, so a window is K words whatever the documents in it are long
// — which makes a window a fixed amount of data and every word equally weighted in the
// loss. On `pretrain_v2` that is 4094 words per window against 1046 before, so a
// window (and a `BATCH_SIZE` step, and everything the schedule below counts in
// windows) covers 3.9x the corpus it used to.

// Defines what a word is, so it lives with the splitter.
pub use wordseg::MAX_WORD_BYTES;

/// Safety cap on tokens per word-window; WORDS_PER_SEQ is meant to bind first.
/// Caches size to the actual longest window, so raising this is free.
pub const MAX_WINDOW_TOKENS: usize = WORDS_PER_SEQ * 7;

// Training-Schedule

pub const LR: f32 = 1e-4;
pub const MIN_LR: f32 = 1e-5;
// Warmup/decay horizons count *windows* (data seen), not optimizer steps, so
// BATCH_SIZE does not reshape the curve over the corpus.
pub const WARMUP_WINDOWS: usize = 1_200;
pub const DECAY_WINDOWS: usize = 1_500_000;
// Windows accumulated per optimizer step. Muon (Frobenius normalization) and
// aux-Adam (second moment) are scale-invariant, so summed grads need no rescale.
pub const BATCH_SIZE: usize = 8;
pub const EPOCHS: usize = 1;

pub const SAVE_EVERY: usize = 1000;
pub const LOG_EVERY: usize = 100;

// Per-stack decoupled weight decay (λ): `0.0` is plain Adam, positive is AdamW
// on the interior projections only (never embeddings, heads, biases, norms).
pub const ENCODER_WEIGHT_DECAY: f32 = 0.01;
pub const BACKBONE_WEIGHT_DECAY: f32 = crate::optimizers::WEIGHT_DECAY;
pub const DECODER_WEIGHT_DECAY: f32 = 0.01;

// Per-stack learning rate, as a multiple of `LR`. Under Adam the update is about
// `lr` per parameter whatever the gradient is, so the change a step makes to a
// layer's output grows with its fan-in: the rate that suits one width is too
// small for a narrower stack. μP's prescription for a hidden weight is
// `lr ∝ 1/fan_in`, which puts the `CHAR_HIDDEN`-wide encoder and decoder at
// `WORD_HIDDEN / CHAR_HIDDEN` times the backbone's rate. `1.0` is one rate for
// the whole model, which is what every checkpoint before these existed trained
// under.
pub const ENCODER_LR_SCALE: f32 = 2.0;
pub const DECODER_LR_SCALE: f32 = 2.0;

/// Weight decay for the flat model. `0.0` keeps it plain Adam.
pub const FLAT_WEIGHT_DECAY: f32 = 0.0;

// Sampling

pub const MAX_LEN: usize = 2000;
pub const TEMPERATURE: f32 = 0.4;
pub const TOP_P: f32 = 0.98;

// Modell-Dimensions

pub const CHAR_HIDDEN: usize = 256;
pub const OUT_HIDDEN: usize = 256;
pub const WORD_HIDDEN: usize = 1024;

/// SwiGLU inner width: the `8·hidden/3` paper default rounded up to a multiple
/// of 64, keeping every up/down projection GEMM tile-aligned.
#[inline]
pub fn up_of(hidden: usize) -> usize {
    let up = hidden * 8 / 3;
    up.div_ceil(64).max(1) * 64
}

/// Output-logit soft cap (xLSTM-7B uses 30): logits = cap · tanh(z / cap).
/// Removes the CE incentive for unbounded head growth on the no-decay Adam path.
pub const LOGIT_SOFTCAP: f32 = 30.0;

/// Number of mLSTM backbone blocks in the hierarchical word model.
pub const WORD_BLOCKS: usize = 28;

/// Carry the backbone's recurrent state from one window into the next. Windows are
/// cut from one packed stream, so every window but a chunk's first continues the one
/// before it and the whole chunk is autoregressed as a single sequence, with the
/// state zeroed only where a document begins. The state crosses a window border, the
/// gradient does not — truncated BPTT, exactly what already happens at a
/// `BACKBONE_CHUNK` border inside a window.
pub const CARRY_WINDOW_STATE: bool = true;

/// Backbone sweep chunk length in words (`0` = one whole-sequence sweep), which
/// bounds resident activations at O(chunk) instead of O(words). Kept above the
/// sLSTM's `FUSED_MIN_T` (32) so a chunk still runs time-fused.
pub const BACKBONE_CHUNK: usize = 512 * 2;

/// Largest encoder/decoder group in rows (`words × tmax`), `0` = uncapped.
/// A group holds every word of one length and pooled buffers never shrink, so
/// one unusual window would raise the run's memory floor forever. Splitting is
/// pure batching — the groups were already independent rectangles. Sized so the
/// rectangle still fills the device.
pub const GROUP_MAX_ROWS: usize = 2048;

/// GPU mLSTM chunk length `L` for the chunkwise formulation (`0` = L = 1).
/// The single-chunk form materializes `[heads, T, T]`, quadratic exactly where
/// the backbone lives; chunking carries `(C, n, m)` across chunks for O(T·L).
/// It is an exact refactoring, not an approximation — the chunk-local stabilizer
/// telescopes to the global row-max (`mlstm_chunking_matches_single_chunk`).
/// 256 measured fastest at the backbone shape (T=2048: 18.9 vs 63.5 ms/iter, and
/// linear in T); below ~64 launch-bound, above ~512 the [L, L] matrices dominate.
/// `MLSTM_CHUNK` overrides; the fused kernels clamp L at `ops::FUSED_MAX_L`.
pub const MLSTM_CHUNK: usize = 256;

/// Read the word embedding `e_w` at an appended `[W]` step, where the state
/// knows the word is complete. `false` evaluates checkpoints trained without it.
pub const ENC_W_EOS: bool = true;

// Dataset

/// Bytes of raw text per streaming chunk. Peak dataset memory scales with this,
/// not with the corpus size.
pub const CHUNK_BYTES: usize = 32 * 1024 * 1024;

/// ISO codes kept from a parquet corpus; other rows are dropped before
/// tokenizing. Empty disables filtering, as does a corpus without the column.
pub const ALLOWED_LANGUAGES: &[&str] = &["en", "de"];

/// Column holding the per-document language code.
pub const PARQUET_LANGUAGE_COLUMN: &str = "language";

/// Corpus path: `.parquet` selects the parquet reader (column `text`, override
/// with `PARQUET_TEXT_COLUMN`), anything else is text with `<|endoftext|>`
/// separators. A directory is walked shard by shard.
pub const TRAIN_DATA: &str = "data/mix/pretrain_v2";
pub const VAL_DATA: &str = "../../training_data/TinyStoriesV2-GPT4-valid.txt";

// Post-training (SFT / instruction tuning)

/// Instruction dataset (JSONL), formatted into masked chat windows by
/// `crate::sft`; the loss counts only the response tokens.
pub const SFT_DATA: &str = "data/mix/assistant_qa.jsonl";
/// Passes over the SFT set. Too many overfits a small set.
pub const SFT_EPOCHS: usize = 1;
/// An order of magnitude below `LR`, so fine-tuning nudges the pretrained
/// weights instead of overwriting them.
pub const SFT_LR: f32 = 5e-6;
/// Windows over which the SFT LR ramps to `SFT_LR`, counted from the start of
/// the fine-tune. Short: it only covers Adam's moments filling, not a random
/// init. The rate is held afterwards — an SFT run is too short to decay.
pub const SFT_WARMUP_WINDOWS: usize = 1000;
/// Windows per optimizer step during SFT. Separate from `BATCH_SIZE` because an
/// SFT window is one example, and examples vary in length far more.
pub const SFT_BATCH_SIZE: usize = 4;

/// Cap on the words of one SFT example — the unit the model unrolls in. Matching
/// `WORDS_PER_SEQ` means a run that fits `hg` fits `hqg`.
pub const SFT_MAX_WORDS: usize = WORDS_PER_SEQ;

/// Safety cap on the tokens of one SFT example; `SFT_MAX_WORDS` binds first.
/// The per-example cache holds one slot per token.
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
