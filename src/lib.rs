pub mod batches;
pub mod config;
pub mod hierarchical;
pub mod loading;
pub mod model;
pub mod nn;
pub mod nn_layer;
pub mod opimizers;
pub mod prepare_set;
pub mod sampling;
pub mod saving;
pub mod sequential;
pub mod tokenizer;
pub mod training;

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
        other => {
            eprintln!(
                "Unknown mode {other:?}. Erlaubt: '' (train_normal), 'h' (train_hierarchical), \
                 's' (sample_normal), 'hs' (sample_hierarchical).",
            );
            std::process::exit(2);
        }
    }
}
