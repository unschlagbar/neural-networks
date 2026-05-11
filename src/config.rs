// Model-path

pub const MODEL_LOC: &str = "models/hierarchical";
pub const SEQ_LOC: &str = "models/seq";

// Sequenz-Len
pub const SEQ_LEN: usize = 1024 * 1;
pub const MAX_SEQ_LEN: usize = SEQ_LEN + 1024 * 4;

pub const LR: f32 = 10e-4;
pub const WARMUP_STEPS: usize = 100;
pub const BATCH_SIZE: usize = 1;

// ── Training-Schedule

pub const EPOCHS: usize = 1;

pub const SAVE_EVERY: usize = 30;

pub const PRINT_EVERY: usize = 10;

// Sampling

pub const MAX_LEN: usize = 2000;
pub const TEMPERATURE: f32 = 0.3;
pub const TOP_P: f32 = 1.;

// Modell-Dimensions

pub const CHAR_HIDDEN: usize = 128;
pub const OUT_HIDDEN: usize = 128;
pub const WORD_HIDDEN: usize = 256;

// Dataset

pub const DATA_DIR: &str = "data/rust-lib/";
pub const DATA_FILE: &str = "data/train.txt";
pub const CHARSET: &str = "charset.txt";
