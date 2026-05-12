pub mod batches;
pub mod config;
pub mod hierarchical;
pub mod loading;
pub mod model;
pub mod nn;
pub mod nn_layer;
pub mod npu_inference;
pub mod onnx_export;
pub mod optimizers;
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
        "npu" => npu_inference::sample_npu(),
        "export" => {
            let model = match crate::sequential::Sequential::load(config::SEQ_LOC) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("load failed: {e}");
                    std::process::exit(1);
                }
            };
            match onnx_export::export_flat_model(&model, "model.onnx") {
                Ok(()) => println!("ONNX model written to model.onnx"),
                Err(e) => eprintln!("export failed: {e}"),
            }
        }
        other => {
            eprintln!(
                "Unknown mode {other:?}. Erlaubt: '' (train_normal), 'h' (train_hierarchical), \
                 's' (sample_normal), 'hs' (sample_hierarchical), \
                 'export' (ONNX-Export), 'npu' (NPU-Inferenz via OpenVINO).",
            );
            std::process::exit(2);
        }
    }
}
