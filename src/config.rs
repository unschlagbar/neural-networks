// Model-path

pub const MODEL_LOC: &str = "models/main_igh_lr";
pub const SEQ_LOC: &str = "models/seq";

// Sequenz-Len

pub const SEQ_LEN: usize = 512 * 1;
pub const MAX_SEQ_LEN: usize = SEQ_LEN + 128;

// Hierarchical word-grouped training
pub const WORDS_PER_SEQ: usize = 512; // K — backbone unroll length per sample
pub const MIN_WORDS_PER_SEQ: usize = 8; // keep a trailing window only if >= this

// Training-Schedule

pub const LR: f32 = 2e-4;
pub const MIN_LR: f32 = 2e-5;
pub const WARMUP_STEPS: usize = 500;
pub const DECAY_STEPS: usize = 50_000;
pub const BATCH_SIZE: usize = 1;
pub const EPOCHS: usize = 2;

pub const SAVE_EVERY: usize = 30;
pub const LOG_EVERY: usize = 10;

// Sampling

pub const MAX_LEN: usize = 2000;
pub const TEMPERATURE: f32 = 0.3;
pub const TOP_P: f32 = 1.;

// Modell-Dimensions

pub const CHAR_HIDDEN: usize = 128;
pub const OUT_HIDDEN: usize = 128;
pub const WORD_HIDDEN: usize = 512;

/// Number of mLSTM backbone blocks in the hierarchical word model.
pub const WORD_BLOCKS: usize = 16;

// Experiments

pub const STOP_WORD_DIRECT_FEED: bool = false;
pub const INJECT_H: bool = false;
pub const INJECT_C: bool = false;

// Dataset

pub const DATA_DIR: &str = "data/rust-lib/";
pub const DATA_FILE: &str = "../../training_data/political_speeches.txt";
pub const CHARSET: &str = "charset.txt";

// Wake Word

pub const WAKE_HIDDEN: usize = 96;
pub const WAKE_SR: usize = 16_000;
pub const WAKE_FRAME_LEN: usize = 400;
pub const WAKE_FRAME_SHIFT: usize = 320;
pub const WAKE_N_FFT: usize = 512;
pub const WAKE_N_MELS: usize = 80;
pub const WAKE_INPUT_DIM: usize = WAKE_N_MELS;
pub const WAKE_THRESHOLD: f32 = 0.8;
pub const WAKE_POS_WEIGHT: f32 = 1.0;
pub const WAKE_LR: f32 = 6e-4;
pub const WAKE_EPOCHS: usize = 35;
pub const WAKE_MODEL_LOC: &str = "models/wake_word";
pub const WAKE_DATA_POS: &str = "data/wake_word/positive";
pub const WAKE_DATA_NEG: &str = "data/wake_word/negative";
