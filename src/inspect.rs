use std::{
    io::{BufRead, stdin},
    path::{Path, PathBuf},
};

use crate::{
    format::{ModelKind, Reader},
    hierarchical::Hierarchical,
    nn::{
        causal_conv1d::CausalConv1dLayer, dropout::DropoutLayer, embedding::EmbeddingLayer,
        linear::LinearLayer, linear_nb::LinearNBLayer, lstm::LSTMLayer, mlstm::MLSTMLayer,
        mlstm_block::MLSTMBlock, rms_norm::RMSNorm, silu_dense::SiluDenseLayer, slstm::SLSTMLayer,
        slstm_block::SLSTMBlock, soft_cap::SoftCapLayer,
    },
    nn_layer::NnLayer,
    sequential::Sequential,
};

/// Interactive mode: reads a model name from stdin, looks it up (as given,
/// then under `models/`), detects the file format by magic number and prints
/// every layer with its settings.
pub fn inspect_model() {
    println!("Model name:");
    let mut line = String::new();
    stdin().lock().read_line(&mut line).unwrap();
    let name = line.trim();

    let Some(path) = resolve(name) else {
        eprintln!("No model named {name:?} found (tried {name:?} and \"models/{name}\")");
        std::process::exit(2);
    };
    let path = path.to_string_lossy();

    let kind = match Reader::peek_kind(&path) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("{path}: not an NNM1 model file ({e})");
            std::process::exit(2);
        }
    };
    match kind {
        ModelKind::Flat => {
            let model = Sequential::load(&path).unwrap();
            println!("\n{path}  (NNM1, flat)");
            print_sequential(&model);
        }
        ModelKind::Hierarchical => {
            // Raw stacks, not `Hierarchical::load` — old layouts stay
            // inspectable even though they fail the current-architecture asserts.
            let stacks = Hierarchical::load_stacks(&path).unwrap();
            println!("\n{path}  (NNM1, hierarchical)");
            println!(
                "vocab_size = {}, context_size = {}, step = {}",
                stacks.vocab_size, stacks.context_size, stacks.step,
            );
            println!("\nencoder fwd (char model):");
            print_sequential(&stacks.encoder_fwd);
            println!("\nencoder bwd (char model):");
            print_sequential(&stacks.encoder_bwd);
            println!("\ncombine: [fwd ; bwd] → e_w (Linear)");
            println!("\nword_model:");
            print_sequential(&stacks.word_model);
            println!("\nchar2_model:");
            print_sequential(&stacks.char2_model);
        }
    }
}

fn resolve(name: &str) -> Option<PathBuf> {
    let direct = Path::new(name);
    if direct.is_file() {
        return Some(direct.to_path_buf());
    }
    let in_models = Path::new("models").join(name);
    in_models.is_file().then_some(in_models)
}

fn print_sequential(model: &Sequential) {
    println!(
        "  {} -> {}, {} layers",
        model.input_size,
        model.output_size,
        model.layers.len(),
    );
    for (i, layer) in model.layers.iter().enumerate() {
        let (name, settings) = describe(layer.as_ref());
        println!(
            "  [{i:>2}] {name:<14} {:>5} -> {:<5} {settings}",
            layer.input_size(),
            layer.output_size(),
        );
    }
}

/// Returns the layer's display name and its extra settings (beyond in/out size).
fn describe(layer: &dyn NnLayer) -> (&'static str, String) {
    let any = layer.as_any();
    if any.is::<EmbeddingLayer>() {
        ("Embedding", String::new())
    } else if any.is::<LinearLayer>() {
        ("Linear", String::new())
    } else if any.is::<LinearNBLayer>() {
        ("LinearNoBias", String::new())
    } else if any.is::<SiluDenseLayer>() {
        ("SiluDense", String::new())
    } else if any.is::<LSTMLayer>() {
        ("LSTM", String::new())
    } else if any.is::<SLSTMLayer>() {
        ("sLSTM", String::new())
    } else if let Some(l) = any.downcast_ref::<MLSTMLayer>() {
        (
            "mLSTM",
            format!("heads={} dqk={} dhv={}", l.num_heads, l.dqk, l.dhv),
        )
    } else if let Some(l) = any.downcast_ref::<MLSTMBlock>() {
        (
            "MLSTMBlock",
            format!(
                "heads={} dqk={} dhv={} up={}",
                l.cell.num_heads, l.cell.dqk, l.cell.dhv, l.up_size,
            ),
        )
    } else if let Some(l) = any.downcast_ref::<SLSTMBlock>() {
        ("SLSTMBlock", format!("up={}", l.up_size))
    } else if any.is::<RMSNorm>() {
        ("RMSNorm", String::new())
    } else if let Some(l) = any.downcast_ref::<SoftCapLayer>() {
        ("SoftCap", format!("cap={}", l.cap))
    } else if let Some(l) = any.downcast_ref::<DropoutLayer>() {
        ("Dropout", format!("rate={}", l.rate))
    } else if let Some(l) = any.downcast_ref::<CausalConv1dLayer>() {
        ("CausalConv1d", format!("kernel={}", l.kernel_size))
    } else {
        ("Unknown", format!("tag={}", layer.layer_tag()))
    }
}
