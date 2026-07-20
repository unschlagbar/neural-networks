// Sequenz-Len

pub const SEQ_LEN: usize = 512 * 1;
pub const MAX_SEQ_LEN: usize = SEQ_LEN + 128;

// Word-grouped training (both the flat and the hierarchical model train on
// these K-word windows, so WORDS_PER_SEQ is the one binding knob).
pub const WORDS_PER_SEQ: usize = 1024 * 1; // K — words per window / backbone unroll length
pub const MIN_WORDS_PER_SEQ: usize = 8; // keep a trailing window only if >= this

/// Cap on the bytes of a single word (see `crate::segment`). Words longer than
/// this — a giant identifier, a base64 blob, a huge indent block — are chopped
/// into pieces, which bounds the decoder's per-word unroll.
pub const MAX_WORD_BYTES: usize = 16;

/// Safety cap on tokens per word-window. Deliberately generous: WORDS_PER_SEQ
/// is meant to bind first — this only guards against a pathological run with no
/// boundary token. Caches are sized to the *actual* longest window
/// (`WordChunk::max_window_tokens`), never to this cap, so raising it is free.
pub const MAX_WINDOW_TOKENS: usize = WORDS_PER_SEQ * 6;

// Training-Schedule

pub const LR: f32 = 2e-4;
pub const MIN_LR: f32 = 2e-5;
pub const WARMUP_STEPS: usize = 150;
pub const DECAY_STEPS: usize = 150_000;
// Windows whose gradients are accumulated before one optimizer step. Muon
// (matrices) is scale-invariant via the Frobenius normalization and aux-Adam
// (vectors) via its second moment, so summed grads need no manual rescaling.
pub const BATCH_SIZE: usize = 8;
pub const EPOCHS: usize = 1;

pub const SAVE_EVERY: usize = 100;
pub const LOG_EVERY: usize = 10;

// Per-stack decoupled weight decay (λ), passed at optimizer-step time. `0.0`
// makes a stack plain Adam; a positive λ makes it AdamW (decoupled decay on the
// interior projection matrices only — embeddings, logit heads, biases and norm
// scales are never decayed). This lets the hierarchical character stacks
// (encoder/decoder) train as Adam while the backbone stays AdamW.
pub const ENCODER_WEIGHT_DECAY: f32 = 0.0;
pub const BACKBONE_WEIGHT_DECAY: f32 = crate::optimizers::WEIGHT_DECAY;
pub const DECODER_WEIGHT_DECAY: f32 = 0.0;

/// Weight decay for the flat (non-hierarchical) model's optimizer step. Kept at
/// `0.0` to preserve the current plain-Adam behavior; raise it to run the flat
/// model as AdamW.
pub const FLAT_WEIGHT_DECAY: f32 = 0.0;

// Sampling

pub const MAX_LEN: usize = 2000;
pub const TEMPERATURE: f32 = 0.4;
pub const TOP_P: f32 = 0.9;

// Modell-Dimensions

pub const CHAR_HIDDEN: usize = 192;
pub const OUT_HIDDEN: usize = 192;
pub const WORD_HIDDEN: usize = 512;

/// Output-logit soft cap (xLSTM-7B uses 30): logits = cap · tanh(z / cap).
/// Bounds the logits and removes the cross-entropy incentive for unbounded
/// head-weight growth on the no-decay Adam path.
pub const LOGIT_SOFTCAP: f32 = 30.0;

/// Number of mLSTM backbone blocks in the hierarchical word model.
pub const WORD_BLOCKS: usize = 12;

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

pub const TRAIN_DATA: &str = "../../training_data/TinyStoriesV2-GPT4-train.txt";
pub const VAL_DATA: &str = "../../training_data/TinyStoriesV2-GPT4-valid.txt";

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
