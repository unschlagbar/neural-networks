pub mod audio;
pub mod batches;
pub mod checkpoint;
pub mod config;
pub mod format;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod hierarchical;
pub mod infer;
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
        "h" => training::train_hierarchical(&read_model_path()),
        #[cfg(feature = "cuda")]
        "hg" => gpu::train::train_hierarchical_gpu(&read_model_path(), false),
        "hga" => gpu::train::train_hierarchical_gpu(&read_model_path(), true),
        #[cfg(feature = "cuda")]
        "hqg" => gpu::train::train_sft_gpu(&read_model_path()),
        "hq" => training::train_sft(&read_model_path()),
        "hv" => training::validate_hierarchical(&read_model_path()),
        #[cfg(feature = "cuda")]
        "hvg" => gpu::train::validate_hierarchical_gpu(&read_model_path()),
        #[cfg(feature = "cuda")]
        "convert" => gpu::train::convert_checkpoint(&read_model_path()),
        "hs" => sampling::sample_hierarchical(&read_model_path()),
        "hqs" => sampling::sample_chat(&read_model_path()),
        #[cfg(feature = "cuda")]
        "ae2" => gpu::audio_train::train_audio_e2(&read_model_path()),
        #[cfg(feature = "cuda")]
        "ae2a" => gpu::audio_train::evaluate_alignment(&read_model_path()),
        "ae2v" => gpu::audio_train::align_voice(&read_model_path()),
        "ae2e" => gpu::audio_train::eval_audio_e2(&read_model_path()),
        #[cfg(feature = "cuda")]
        "ae2s" => gpu::audio_sample::sample_audio(&read_model_path()),
        "i" => inspect::inspect_model(),
        other => {
            eprintln!(
                "Unknown mode {other:?}. Modes: 'h' train_hierarchical | \
                 'hv' validate_hierarchical | 'hvg' validate_hierarchical on GPU | \
                 'hg' train_hierarchical on GPU | \
                 'hq' SFT (Q-A) fine-tune on CPU | 'hqg' SFT fine-tune on GPU | \
                 'hs' sample_hierarchical | 'hqs' chat/Q-A sampling | \
                 'i' inspect model | 'convert' NNM1 checkpoint to v2 | 'ae2' audio encoder against a frozen text model | 'ae2a' its count-constrained alignment vs MFA | 'ae2s' speak to the text model",
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
