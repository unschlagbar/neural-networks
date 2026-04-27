// ── main.rs ──────────────────────────────────────────────────────────────────
//
// Dünner Dispatcher. Die eigentliche Logik liegt in config/model/training/
// sampling. Argumente:
//
//   (nichts / leere Zeile)  → train_normal     (sLSTM-Block-Stapel)
//   "h"                     → train_hierarchical
//   "s"                     → sample_normal
//   "hs"                    → sample_hierarchical
//
// Die alte Variante „leere Zeile = train, sonst = sample_old" war zu
// implizit, man konnte den hierarchischen Pfad gar nicht direkt ansteuern.

use std::{
    fs,
    io::{BufRead, stdin},
    path::Path,
};

pub mod activations;
pub mod batches;
pub mod config;
pub mod data_set_loading;
pub mod dense;
pub mod dropout;
pub mod embedding;
pub mod hierarchical_sequential;
pub mod linear;
pub mod loading;
pub mod lstm;
pub mod model;
pub mod nn_layer;
pub mod projection;
pub mod rms_norm;
pub mod sampling;
pub mod saving;
pub mod sequential;
pub mod silu_dense;
pub mod slstm;
pub mod slstm_block;
pub mod softmax;
pub mod tokenizer;
pub mod training;

fn main() {
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
