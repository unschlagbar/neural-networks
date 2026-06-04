// Model-path

pub const MODEL_LOC: &str = "models/no_feed_inject_c_long";
pub const SEQ_LOC: &str = "models/seq";

// Sequenz-Len

pub const SEQ_LEN: usize = 512 * 4;
pub const MAX_SEQ_LEN: usize = SEQ_LEN + 128;

// Training-Schedule

pub const LR: f32 = 2e-4;
pub const MIN_LR: f32 = 2e-5;
pub const WARMUP_STEPS: usize = 200;
pub const DECAY_STEPS: usize = 10_000;
pub const BATCH_SIZE: usize = 1;
pub const EPOCHS: usize = 1;

pub const SAVE_EVERY: usize = 30;
pub const LOG_EVERY: usize = 10;

// Sampling

pub const MAX_LEN: usize = 2000;
pub const TEMPERATURE: f32 = 0.4;
pub const TOP_P: f32 = 1.;

// Modell-Dimensions

pub const CHAR_HIDDEN: usize = 512;
pub const OUT_HIDDEN: usize = 512;
pub const WORD_HIDDEN: usize = 512;

// Experiments

pub const STOP_WORD_DIRECT_FEED: bool = true;
pub const INJECT_H: bool = false;
pub const INJECT_C: bool = true;

// Dataset

pub const DATA_DIR: &str = "data/rust-lib/";
pub const DATA_FILE: &str = "../../training_data/political_speeches.txt";
pub const CHARSET: &str = "charset.txt";

// Wake Word

pub const WAKE_HIDDEN: usize = 96;
pub const WAKE_SR: usize = 16_000;
pub const WAKE_FRAME_LEN: usize = 80;
pub const WAKE_FRAME_SHIFT: usize = 20;
pub const WAKE_N_FFT: usize = 512;
pub const WAKE_N_MELS: usize = 80;
pub const WAKE_INPUT_DIM: usize = WAKE_N_MELS;
pub const WAKE_FRAMES: usize = 1 + (WAKE_SR - WAKE_FRAME_LEN) / WAKE_FRAME_SHIFT;
pub const WAKE_THRESHOLD: f32 = 0.5;
pub const WAKE_POS_WEIGHT: f32 = 1.0;
pub const WAKE_LR: f32 = 6e-4;
pub const WAKE_EPOCHS: usize = 500;
pub const WAKE_MODEL_LOC: &str = "models/wake_word2";
pub const WAKE_DATA_POS: &str = "data/wake_word/positive";
pub const WAKE_DATA_NEG: &str = "data/wake_word/negative";
