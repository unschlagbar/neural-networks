pub mod batches;
pub mod config;
pub mod hierarchical;
pub mod loading;
pub mod model;
pub mod nn;
pub mod nn_layer;
pub mod optimizers;
pub mod prepare_set;
pub mod sampling;
pub mod saving;
pub mod sequential;
pub mod tokenizer;
pub mod training;
pub mod wake_word;

use std::{
    fs,
    io::{BufRead, stdin},
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
        "" => training::train_normal(),
        "h" => training::train_hierarchical(),
        "s" => sampling::sample_normal(),
        "hs" => sampling::sample_hierarchical(),
        "wr" => wake_word::record::record_samples(),
        "wt" => wake_word::training::train_wake(),
        "w" => wake_word::detector::run_detector(),
        other => {
            eprintln!(
                "Unknown mode {other:?}. Modes: '' train_normal | 'h' train_hierarchical | \
                 's' sample_normal | 'hs' sample_hierarchical | \
                 'wr' record wake-word samples | 'wt' train wake-word | 'w' run detector",
            );
            std::process::exit(2);
        }
    }
}
