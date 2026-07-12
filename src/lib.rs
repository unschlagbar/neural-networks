pub mod batches;
pub mod config;
pub mod hierarchical;
pub mod inspect;
pub mod loading;
pub mod model;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod nn;
pub mod nn2;
pub mod nn_layer;
pub mod optimizers;
pub mod parallel;
pub mod prepare_set;
pub mod sampling;
pub mod saving;
pub mod segment;
pub mod sequential;
pub mod tensor;
pub mod tokenizer_utf8;
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
        "" => training::train_normal(&read_model_path("models/seq")),
        "h" => training::train_hierarchical(&read_model_path("models/fix_bi")),
        #[cfg(feature = "cuda")]
        "hg" => gpu::train::train_hierarchical_gpu(&read_model_path("models/hier_gpu")),
        "hp" => training::probe_hierarchical(&read_model_path("models/fix_bi")),
        "hv" => training::validate_hierarchical(&read_model_path("models/fix_bi")),
        "ht" => training::trace_hierarchical(&read_model_path("models/fix_bi")),
        "s" => sampling::sample_normal(&read_model_path("models/seq")),
        "hs" => sampling::sample_hierarchical(&read_model_path("models/fix_bi")),
        "i" => inspect::inspect_model(),
        "wr" => wake_word::record::record_samples(),
        "wt" => wake_word::training::train_wake(),
        "w" => wake_word::detector::run_detector(),
        other => {
            eprintln!(
                "Unknown mode {other:?}. Modes: '' train_normal | 'h' train_hierarchical | \
                 'hv' validate_hierarchical | \
                 'hg' train_hierarchical on GPU | \
                 's' sample_normal | 'hs' sample_hierarchical | 'i' inspect model | \
                 'wr' record wake-word samples | 'wt' train wake-word | 'w' run detector",
            );
            std::process::exit(2);
        }
    }
}

/// Prompts for a model name at runtime. Empty input keeps `default`. A bare name
/// (no `/`) is resolved under `models/`, so typing `seq` selects `models/seq`.
fn read_model_path(default: &str) -> String {
    print!("Model name [{default}]: ");
    stdout().flush().ok();
    let mut line = String::new();
    stdin().lock().read_line(&mut line).unwrap();
    let name = line.trim();
    if name.is_empty() {
        default.to_string()
    } else if name.contains('/') {
        name.to_string()
    } else {
        format!("models/{name}")
    }
}
