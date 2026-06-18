use std::{
    fs::File,
    io::{BufRead, stdin},
    path::{Path, PathBuf},
    rc::Rc,
};

use crate::{
    hierarchical::Hierarchical,
    loading::read_u32,
    nn::{
        causal_conv1d::CausalConv1dLayer, dropout::DropoutLayer, embedding::EmbeddingLayer,
        linear::LinearLayer, linear_nb::LinearNBLayer, lstm::LSTMLayer, mlstm::MLSTMLayer,
        mlstm_block::MLSTMBlock, rms_norm::RMSNorm, silu_dense::SiluDenseLayer, slstm::SLSTMLayer,
        slstm_block::SLSTMBlock,
    },
    nn_layer::NnLayer,
    saving::{HIER_MAGIC, MAGIC},
    sequential::Sequential,
    tokenizer::Tokenizer,
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

    let magic = read_u32(&mut File::open(path.as_ref()).unwrap()).unwrap_or(0);
    match magic {
        MAGIC => {
            let model = Sequential::load(&path).unwrap();
            println!("\n{path}  (NNFW, flat)");
            print_sequential(&model);
        }
        HIER_MAGIC => {
            let model = Hierarchical::load(&path, Rc::new(Tokenizer::default())).unwrap();
            println!("\n{path}  (HIER, hierarchical)");
            println!(
                "vocab_size = {}, context_size = {}, boundary tokens = {:?}",
                model.vocab_size, model.context_size, model.boundary_token_ids,
            );
            println!("\nchar_fwd (encoder, forward):");
            print_sequential(&model.encoder.char_fwd);
            println!("\nchar_bwd (encoder, backward):");
            print_sequential(&model.encoder.char_bwd);
            println!("\nword_model:");
            print_sequential(&model.word_model);
            println!("\nchar2_model:");
            print_sequential(&model.char2_model);
        }
        other => {
            eprintln!("{path}: unknown magic 0x{other:08X} — not a NNFW or HIER model file");
            std::process::exit(2);
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
    } else if let Some(l) = any.downcast_ref::<DropoutLayer>() {
        ("Dropout", format!("rate={}", l.rate))
    } else if let Some(l) = any.downcast_ref::<CausalConv1dLayer>() {
        ("CausalConv1d", format!("kernel={}", l.kernel_size))
    } else {
        ("Unknown", format!("tag={}", layer.layer_tag()))
    }
}
