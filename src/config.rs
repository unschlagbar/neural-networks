// Sequence length

pub const SEQ_LEN: usize = 512;
pub const MAX_SEQ_LEN: usize = SEQ_LEN + 128;

// Word-grouped training: both models train on these K-word windows, so
// WORDS_PER_SEQ is the one binding knob.
pub const WORDS_PER_SEQ: usize = 1024 * 4; // K — words per window / backbone unroll length
pub const MIN_WORDS_PER_SEQ: usize = 8; // keep a trailing window only if >= this

// Documents are packed, so a window is K words whatever the documents in it are long:
// a fixed amount of data, every word equally weighted in the loss.

// Defines what a word is, so it lives with the splitter.
pub use wordseg::MAX_WORD_BYTES;

/// Safety cap on tokens per word-window; WORDS_PER_SEQ is meant to bind first.
/// Caches size to the actual longest window, so raising this is free.
pub const MAX_WINDOW_TOKENS: usize = WORDS_PER_SEQ * 7;

// Training schedule

pub const LR: f32 = 1e-4;
pub const MIN_LR: f32 = 1e-5;
// Warmup/decay horizons count *windows* (data seen), not optimizer steps, so
// BATCH_SIZE does not reshape the curve over the corpus.
pub const WARMUP_WINDOWS: usize = 1_200;
pub const DECAY_WINDOWS: usize = 1_500_000;
// Windows accumulated per optimizer step. AdamW is scale-invariant (up to eps),
// so summed grads need no rescale.
pub const BATCH_SIZE: usize = 12;
// Windows over which a resumed GPU pretraining run ramps its rate up again from 0.
// Checkpoints carry no AdamW moments; see `LrSchedule::rewarm_windows`.
pub const RESUME_WARMUP_WINDOWS: usize = 960;
// Global L2 norm the GPU pretrainer clips each step's mean (per-window) gradient to.
// `f32::INFINITY` only measures it; the `grad_norm` log column is the unclipped norm.
// The typical norm is ~1-1.5, so this only catches outliers.
pub const GRAD_CLIP_NORM: f32 = 5.0;
pub const EPOCHS: usize = 1;

pub const SAVE_EVERY: usize = 1000;
pub const LOG_EVERY: usize = 100;

// Per-stack decoupled weight decay (λ): `0.0` is plain Adam, positive is AdamW
// on the interior projections only (never embeddings, heads, biases, norms).
pub const ENCODER_WEIGHT_DECAY: f32 = 0.01;
pub const BACKBONE_WEIGHT_DECAY: f32 = 0.03;
pub const DECODER_WEIGHT_DECAY: f32 = 0.01;

// Per-stack learning rate, as a multiple of `LR`. Under Adam the update is about
// `lr` per parameter whatever the gradient is, so the change a step makes to a
// layer's output grows with its fan-in: the rate that suits one width is too
// small for a narrower stack. μP's prescription for a hidden weight is
// `lr ∝ 1/fan_in`, which puts the `CHAR_HIDDEN`-wide encoder and decoder at
// `WORD_HIDDEN / CHAR_HIDDEN` times the backbone's rate. `1.0` is one rate for
// the whole model.
pub const ENCODER_LR_SCALE: f32 = 2.0;
pub const DECODER_LR_SCALE: f32 = 2.0;

// Sampling

pub const MAX_LEN: usize = 2000;
pub const TEMPERATURE: f32 = 0.4;
pub const TOP_P: f32 = 0.98;
/// Temperature between a sampled `<tool>` and its `</tool>`: a call is a syntax
/// to get exactly right, not prose to vary.
pub const TOOL_TEMPERATURE: f32 = 0.1;

// Model dimensions

pub const CHAR_HIDDEN: usize = 256;
pub const OUT_HIDDEN: usize = 256;
pub const WORD_HIDDEN: usize = 1280;

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
pub const WORD_BLOCKS: usize = 48;

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
pub const GROUP_MAX_ROWS: usize = 2048;

/// GPU mLSTM chunk length `L` for the chunkwise formulation (`0` = L = 1).
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
/// `crate::sft`; the loss counts only the response tokens. A user turn naming
/// an `audio` take of `SFT_SPEECH_DIRS` is heard instead of read.
pub const SFT_DATA: &str = "data/mix/assistant_audio.jsonl";
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

// Audio encoder

pub const AUDIO_SR: usize = 16_000;
/// Analysis window and hop in samples: 25 ms every 10 ms, so features run at
/// 100 Hz. Without the overlap the Hann taper hides short sounds at window borders.
pub const AUDIO_WIN: usize = 400;
pub const AUDIO_HOP: usize = 160;
/// FFT length; the window is zero-padded up to it.
pub const AUDIO_N_FFT: usize = 512;
pub const AUDIO_N_MELS: usize = 128;
/// Per-frame features: the mel bins plus the frame's absolute log energy, which
/// carries loudness past any normalization.
pub const AUDIO_FEATS: usize = AUDIO_N_MELS + 1;

/// The stem reads `AUDIO_STACK` consecutive feature frames per output frame and
/// advances `AUDIO_STRIDE` of them, so the encoder runs at 25 Hz (40 ms). 12.5 Hz
/// would put two short function words into one frame in fast speech.
pub const AUDIO_STACK: usize = 4;
pub const AUDIO_STRIDE: usize = 4;

pub const AUDIO_HIDDEN: usize = 512;
pub const AUDIO_BLOCKS: usize = 12;
pub const AUDIO_HEADS: usize = 8;
pub const AUDIO_DQK: usize = AUDIO_HIDDEN / AUDIO_HEADS;
/// Blocks of the per-word segment encoder, bottom to top: `s` an sLSTM block,
/// `m` an mLSTM block. Every word runs from a zero state.
pub const AUDIO_SEG_STACK: &str = "ss";
/// Frames of a word the segment encoder reads, counted back from its end
/// (64 = 2.56 s). A segment runs from the previous word's end, pause
/// included: 2.5% of LibriSpeech's are longer than 24 frames, 0.035% than
/// 64. The frame stack before it has heard the rest.
pub const AUDIO_SEG_MAX_FRAMES: usize = 64;
/// Rows (words × steps, every word padded to the batch's longest) the segment
/// encoder runs at most; a batch of many words with one long one reads fewer
/// frames per word instead. 20 000 is 776 words at 24 frames.
pub const AUDIO_SEG_MAX_ROWS: usize = 20_000;

/// The boundary head decides about a word end up to this many encoder frames
/// late (5 = 200 ms). The word's embedding never reads them.
pub const BOUNDARY_DELAY: usize = 5;
/// Encoder frames on either side of a labelled word end that still count as it:
/// forced alignment is good to about ±20 ms, a frame is 40 ms.
pub const BOUNDARY_TOLERANCE: usize = 1;

// Audio encoder training (`ae2`: the encoder against a frozen text model)

/// Root holding `LibriSpeech/<split>/` and `alignments/<split>/` (see
/// `audio::librispeech`).
pub const AUDIO_DATA: &str = "../../training_data/librispeech";
pub const AUDIO_SPLITS: &[&str] = &[
    "dev-clean",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
];
/// Speakers of the first split (the last ones by id) never trained on, scored
/// every `AUDIO_VAL_EVERY` steps.
pub const AUDIO_HELD_OUT_SPEAKERS: usize = 4;
/// Length of an `ae2` run in optimizer steps; the LR schedule spans it, and a
/// resumed run continues to it.
pub const AUDIO_E2_STEPS: usize = 100_000;
/// The encoder an `ae2` run starts from when `<model>.ae2` does not exist, at
/// step 0 of a fresh schedule; empty starts a new one.
pub const AUDIO_E2_FROM: &str = "models/s4_S.ae2.final";
/// Utterances per optimizer step.
pub const AUDIO_BATCH: usize = 8;
/// Words of the chapter's transcript the text model reads before an
/// utterance's audio (training only, with chance `AUDIO_PREFIX_P`) and after it
/// (validation always reads `AUDIO_CONTEXT_WORDS`): the audio has to leave the
/// backbone in a state the text can continue from.
pub const AUDIO_CONTEXT_WORDS: usize = 24;
pub const AUDIO_PREFIX_P: f32 = 0.0;
/// Words after the audio a training window scores, for takes read out of a
/// document (voice, TTS); a LibriSpeech utterance is followed by its book and
/// scores none. The loss is a mean over every scored word, so too many of them
/// outweigh the utterance's own words.
pub const AUDIO_TRAIN_AFTER_WORDS: usize = 16;
/// Weight of the text model's own distribution (reading the transcript) in the
/// decoder's target for the audio window; 0 trains on the transcript alone.
pub const DISTILL_ALPHA: f32 = 0.5;
/// The same for windows with no document text after the audio (LibriSpeech,
/// SLURP, VoxPopuli): 1 trains towards the reading alone, so audio that
/// beats it (a book's genre, a parliament's phrasing) is not rewarded.
pub const DISTILL_ALPHA_PLAIN: f32 = 1.0;
/// Weight `β` of the backbone-context distillation in `ae2`: after the audio
/// the backbone's per-word context should be the one reading the transcript
/// gives, `β/2 · mean ‖o − o_read‖² / mean ‖o_read‖²`. The teacher is the same
/// backbone reading, so it stays meaningful once the backbone trains too.
pub const STATE_DISTILL_BETA: f32 = 1.0;
/// Weight `γ` of `γ/2 · Σ‖a − t‖² / Σ‖t‖²` in `ae2`: each heard word's audio
/// embedding `a` pulled onto the text encoder's embedding `t` of the same word
/// as written. Independent of the context, so it applies to every take.
pub const EMBED_ALIGN_WEIGHT: f32 = 0.0;
/// Weight of the contrastive term in `ae2`: each heard word's audio embedding
/// should be nearer (by cosine, at temperature `EMBED_CONTRAST_TEMP`) to the
/// text encoder's embedding of its own word than to those of the other words
/// heard in the batch. It asks the audio word to point at its text word, not to
/// become it.
pub const EMBED_CONTRAST_WEIGHT: f32 = 0.1;
pub const EMBED_CONTRAST_TEMP: f32 = 0.1;
/// Training utterances heard through a random room, noise and microphone
/// (`audio::augment`); validation stays clean.
pub const AUDIO_AUGMENT: bool = true;
/// Background recordings mixed in under augmentation (ESC-50); colored noise
/// stands in where there are none.
pub const AUDIO_NOISE_DIR: &str = "../../training_data/noise";
/// The LibriSpeech split whose first speakers are mixed under training speech
/// as other people talking.
pub const AUDIO_BABBLE_SPLIT: &str = "train-other-500";
/// Takes recorded on the harness VOICE page (`audio::voice`), aligned by mode
/// `ae2v`.
pub const AUDIO_VOICE_DIR: &str = "../../training_data/voice";
/// Sessions of it held out as their own validation line: a new day and setup.
pub const AUDIO_VOICE_VAL_SESSIONS: &[&str] = &["2026-10-17_1534"];
/// Of the other takes only the newest this many are trained on (all of
/// them once each take has been listened to and the cut ones removed).
pub const AUDIO_VOICE_LAST: usize = usize::MAX;
/// How often each training voice take is heard over a run of
/// `AUDIO_E2_STEPS`: a batch carries one take, in place of a LibriSpeech
/// utterance, with the chance that spreads them evenly. Too many and the
/// encoder learns the takes' sentences instead of the voice.
pub const AUDIO_VOICE_EPOCHS: f32 = 8.0;
/// The documents the voice prompts were read out of: the text around a take.
pub const AUDIO_VOICE_DOCS: &str = "data/voice/prompts_docs.jsonl";
/// Further sets of takes in the voice recorder's layout, with their documents
/// (if any) in `<dir>/docs.jsonl`: assistant questions with their answers and
/// sentences of pretraining documents read by a multi-speaker TTS
/// (`scripts/tts_questions.py`), such sentences in the user's voice cloned from
/// their takes (`scripts/tts_clone.py`), SLURP's people talking to a home
/// assistant (`scripts/slurp_takes.py`), VoxPopuli's European Parliament
/// speeches, many by non-native speakers (`scripts/fetch_voxpopuli.py`),
/// Common Voice's crowd-read sentences (`scripts/fetch_common_voice.py`) and
/// HeySQuAD's crowd-read questions (`scripts/heysquad_takes.py`).
pub const AUDIO_TAKE_DIRS: &[&str] = &[
    "../../training_data/tts_questions",
    "../../training_data/tts_text",
    "../../training_data/tts_clone",
    "../../training_data/slurp",
    "../../training_data/voxpopuli",
    "../../training_data/common_voice",
    "../../training_data/heysquad_human",
];
/// Peak learning rate: 3e-4 from scratch; continuing a finished encoder at
/// 3e-4 first loses CER to its parent.
pub const AUDIO_LR: f32 = 1e-4;
pub const AUDIO_WARMUP_STEPS: usize = 100;
pub const AUDIO_VAL_EVERY: usize = 200;
/// The audio stage (mode `hga`), after text pretraining: the pretraining loop
/// with one audio window (`gpu::audio_train::AudioPretrain`) after every this
/// many text windows. Mode `hg` never runs one.
pub const AUDIO_PRETRAIN_EVERY: usize = 10;
/// The backbone's learning rate in the audio stage, constant: a finished
/// pretraining schedule ends near zero, where nothing would learn to read audio.
pub const AUDIO_STAGE_LR: f32 = 2e-5;
/// Words of a LibriSpeech chapter run per audio window, and how many of its
/// utterances are heard instead of read.
pub const AUDIO_PRETRAIN_WORDS: usize = 1024;
pub const AUDIO_PRETRAIN_HEARD: usize = 3;
/// Share of the audio windows that are a voice document instead: one of the
/// pretraining-holdout documents the voice prompts were read out of, the
/// takes recorded from it heard where their sentences stand.
pub const AUDIO_PRETRAIN_VOICE_P: f32 = 0.25;
/// Share that are a TTS take (`AUDIO_TAKE_DIRS`) heard, then its document read on.
pub const AUDIO_PRETRAIN_TTS_P: f32 = 0.25;
/// Takes longer than this are left out: the training cache is sized for the
/// longest item, and VoxPopuli holds speeches of several minutes.
pub const AUDIO_MAX_TAKE_SECONDS: f32 = 25.0;
/// Utterances with more words are left out: the segment encoder's scratch is
/// sized for a batch of the longest (LibriSpeech has at most 92).
pub const AUDIO_MAX_WORDS: usize = 100;
/// While `ae2` trains, a take's words are forced-aligned on the character
/// head's output of the same step, before they are pooled into embeddings,
/// when that output read the take at no more than this CER; otherwise its
/// last such alignment, or the one it was loaded with, stands.
pub const AUDIO_REALIGN_MAX_CER: f32 = 0.35;
/// Whether `ae2` aligns take words online at all; off keeps the alignment the
/// takes were loaded with (e.g. `scripts/mms_align.py`) for the whole run.
pub const AUDIO_ALIGN_ONLINE: bool = false;
/// Chance that a training utterance's words are pooled as the encoder's own
/// boundary head cuts them (given the word count, as the evaluation's
/// "boundary head" line does) rather than at the reference ends: live, the
/// head decides, and embeddings trained on reference cuts only degrade under it.
pub const AUDIO_HEAD_CUT_P: f32 = 0.5;
/// Share of an `ae2` batch drawn from `AUDIO_TAKE_DIRS`.
pub const AUDIO_TAKE_SHARE: f32 = 0.625;
/// Each take slot picks a set with probability ∝ size^power: 0 is every set
/// alike (a 7k-take set heard 14 times over a run, a 160k one not once), 1 is
/// in proportion to size.
pub const AUDIO_TAKE_SET_POWER: f32 = 0.5;
/// Weight `α` of the read window's predictions in the heard window's target,
/// `(1 − α)·y + α·p_read`: after a heard sentence the backbone should predict
/// what it predicts after reading it. `0` runs the read window for the report only.
pub const AUDIO_PRETRAIN_DISTILL: f32 = 0.5;
/// Audio windows between two runs over the held-out windows (and one before
/// the first).
pub const AUDIO_PRETRAIN_EVAL_EVERY: usize = 300;
/// The audio encoder's learning rate there, constant.
pub const AUDIO_PRETRAIN_LR: f32 = 1e-4;
/// The encoder a run starts from when `<model>.ae2` does not exist yet.
pub const AUDIO_PRETRAIN_START: &str = "models/s4.ae2.H-8ep";
/// Take sets of audio SFT (mode `hqg`): the user turns of SFT conversations
/// read by a multi-speaker TTS (`scripts/sft_speech.py`). A user message whose
/// `audio` names one of their takes is heard; one whose take is not here is read.
pub const SFT_SPEECH_DIRS: &[&str] = &["../../training_data/sft_speech"];
/// The encoder audio SFT starts from when `<model>.ae2` does not exist yet.
pub const SFT_AUDIO_START: &str = "models/s4_T.ae2.final";
/// The audio encoder's learning rate in SFT, constant.
pub const SFT_AUDIO_LR: f32 = 2e-5;
/// Weight `α` of the reply's predictions after reading a heard turn's words in
/// the target after hearing them, as `AUDIO_PRETRAIN_DISTILL`.
pub const SFT_AUDIO_DISTILL: f32 = 0.5;
/// Weight on the boundary head's positive term: about one frame in seven ends a word.
pub const BOUNDARY_POS_WEIGHT: f32 = 2.0;
/// Weight of the boundary head's word-count loss (`CountLoss`): the squared
/// relative error of its probability mass, against a per-element BCE.
pub const BOUNDARY_COUNT_WEIGHT: f32 = 0.0;
/// CTC steps per encoder frame for the character head: 2 is 50 Hz, above the
/// ~15 characters per second of fast read speech.
pub const CTC_PER_FRAME: usize = 2;
/// Share of the CTC gradient the frame encoder receives; 0 makes the character
/// head a probe of what the frames already know.
pub const CTC_WEIGHT: f32 = 1.0;
