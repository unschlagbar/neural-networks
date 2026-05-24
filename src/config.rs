// Model-path

pub const MODEL_LOC: &str = "models/normal_very_big";
pub const SEQ_LOC: &str = "models/seq";

// Sequenz-Len

pub const SEQ_LEN: usize = 512 * 1;
pub const MAX_SEQ_LEN: usize = SEQ_LEN + 128;

// Training-Schedule

pub const LR: f32 = 2e-4;
pub const MIN_LR: f32 = 3e-5;
pub const WARMUP_STEPS: usize = 200;
pub const DECAY_STEPS: usize = 10_000;
pub const BATCH_SIZE: usize = 1;
pub const EPOCHS: usize = 1;

pub const SAVE_EVERY: usize = 30;
pub const PRINT_EVERY: usize = 10;

// Sampling

pub const MAX_LEN: usize = 2000;
pub const TEMPERATURE: f32 = 0.4;
pub const TOP_P: f32 = 1.;

// Modell-Dimensions

pub const CHAR_HIDDEN: usize = 768;
pub const OUT_HIDDEN: usize = 768;
pub const WORD_HIDDEN: usize = 768;

// Flat-mLSTM-Architektur (abgeleitet aus WORD_HIDDEN)

pub const NUM_HEADS: usize = 16;
pub const DQK: usize = WORD_HIDDEN / NUM_HEADS / 2;
pub const DHV: usize = WORD_HIDDEN / NUM_HEADS;
pub const C_SIZE: usize = NUM_HEADS * DHV * DQK;
pub const N_SIZE: usize = NUM_HEADS * DQK;

// Experiments

pub const STOP_CHAR1_AT_BOUNDARY: bool = true;
pub const STOP_WORD_DIRECT_FEED: bool = false;
pub const USE_STATE_INJECT: bool = true;

// Dataset

pub const DATA_DIR: &str = "data/rust-lib/";
pub const DATA_FILE: &str = "../../training_data/political_speeches.txt";
pub const CHARSET: &str = "charset.txt";
