pub mod batches;
pub mod config;
pub mod format;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod grow_vocab;
pub mod hierarchical;
pub mod inspect;
pub mod loading;
pub mod model;
pub mod nn;
pub mod nn2;
pub mod nn_layer;
pub mod optimizers;
pub mod parallel;
pub mod parquet;
pub mod prepare_set;
pub mod pretrain_progress;
pub mod sampling;
pub mod saving;
pub mod sequential;
// The corpus definitions live in their own crate so `datamix` counts exactly
// the words this model trains on; re-exported so `crate::segment::…` and
// `crate::tokenizer_utf8::…` keep resolving.
pub use wordseg::{segment, tokenizer_utf8};
pub mod sft;
pub mod sft_progress;
pub mod tensor;
pub mod training;
pub mod wake_word;
pub mod word_encoder;

use std::{
    fs,
    io::{BufRead, Write, stdin, stdout},
    path::Path,
};

pub fn run() {
    if !Path::new("models/").exists() {
        fs::create_dir("models/").unwrap();
    }

    let mut line = String::new();
    stdin().lock().read_line(&mut line).unwrap();
    let cmd = line.trim();

    match cmd {
        "" => training::train_normal(&read_model_path()),
        "h" => training::train_hierarchical(&read_model_path()),
        #[cfg(feature = "cuda")]
        "hg" => gpu::train::train_hierarchical_gpu(&read_model_path()),
        #[cfg(feature = "cuda")]
        "hqg" => gpu::train::train_sft_gpu(&read_model_path()),
        "hq" => training::train_sft(&read_model_path()),
        "av" => grow_vocab::grow_model_interactive(),
        "hp" => training::probe_hierarchical(&read_model_path()),
        "hv" => training::validate_hierarchical(&read_model_path()),
        #[cfg(feature = "cuda")]
        "hvg" => gpu::train::validate_hierarchical_gpu(&read_model_path()),
        "ht" => training::trace_hierarchical(&read_model_path()),
        "s" => sampling::sample_normal(&read_model_path()),
        "hs" => sampling::sample_hierarchical(&read_model_path()),
        "hqs" => sampling::sample_chat(&read_model_path()),
        "i" => inspect::inspect_model(),
        "wr" => wake_word::record::record_samples(),
        "wt" => wake_word::training::train_wake(),
        "w" => wake_word::detector::run_detector(),
        other => {
            eprintln!(
                "Unknown mode {other:?}. Modes: '' train_normal | 'h' train_hierarchical | \
                 'hv' validate_hierarchical | 'hvg' validate_hierarchical on GPU | \
                 'hg' train_hierarchical on GPU | \
                 'hq' SFT (Q-A) fine-tune on CPU | 'hqg' SFT fine-tune on GPU | \
                 'av' add SFT vocab to a model | \
                 's' sample_normal | 'hs' sample_hierarchical | 'hqs' chat/Q-A sampling | \
                 'i' inspect model | \
                 'wr' record wake-word samples | 'wt' train wake-word | 'w' run detector",
            );
            std::process::exit(2);
        }
    }
}

/// Prompts for a model name at runtime. Empty input keeps `default`. A bare name
/// (no `/`) is resolved under `models/`, so typing `seq` selects `models/seq`.
fn read_model_path() -> String {
    print!("Model name: ");
    stdout().flush().ok();
    let mut line = String::new();
    stdin().lock().read_line(&mut line).unwrap();
    let name = line.trim();
    if name.is_empty() {
        String::new()
    } else if name.contains('/') {
        name.to_string()
    } else {
        format!("models/{name}")
    }
}
