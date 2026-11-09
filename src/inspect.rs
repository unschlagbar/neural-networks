use std::{
    io::{BufRead, stdin},
    path::{Path, PathBuf},
};

use crate::{
    checkpoint::{self, BlockKind},
    sequential::{
        NnLayer, Sequential,
        layers::{
            causal_conv1d::CausalConv1dLayer, dropout::DropoutLayer, embedding::EmbeddingLayer,
            linear::LinearLayer, linear_nb::LinearNBLayer, lstm::LSTMLayer, rms_norm::RMSNorm,
            silu_dense::SiluDenseLayer, slstm::SLSTMLayer, soft_cap::SoftCapLayer,
        },
    },
};

/// Interactive mode: reads a model name from stdin, looks it up (as given, then
/// under `models/`) and prints its structure: a v2 hierarchical checkpoint, or a
/// flat `Sequential` file.
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

    let result = match checkpoint::is_v2(&*path) {
        Ok(true) => print_v2(&path),
        Ok(false) => Sequential::load(&path).map(|model| {
            println!("\n{path}  (NNM1, flat)");
            print_sequential(&model);
        }),
        Err(e) => Err(e),
    };
    if let Err(e) = result {
        eprintln!("{path}: {e}");
        std::process::exit(2);
    }
}

fn print_v2(path: &str) -> std::io::Result<()> {
    let mut r = checkpoint::Reader::open(path)?;
    let checkpoint::Header {
        cfg,
        layout,
        step,
        seen,
    } = r.header.clone();
    let tensors = r.remaining();
    let mut params = 0usize;
    while r.remaining() > 0 {
        params += r.skip()?.iter().product::<usize>();
    }
    r.finish()?;

    println!("\n{path}  (v2, hierarchical)");
    println!(
        "vocab = {}, char hidden = {}, word hidden = {}, [W] = {}, logit cap = {}, step = {step}",
        cfg.vocab, cfg.hc, cfg.wh, cfg.w_token, cfg.cap,
    );
    println!("{tensors} tensors, {params} parameters");
    println!("\ntrained on:");
    print!("{}", seen.report());
    for (name, blocks) in [
        ("encoder", &layout.encoder),
        ("backbone", &layout.backbone),
        ("decoder", &layout.decoder),
    ] {
        println!("\n{name}: {} blocks", blocks.len());
        for (i, b) in blocks.iter().enumerate() {
            match b {
                BlockKind::SLstm => println!("  [{i:>2}] sLSTM"),
                BlockKind::MLstm { heads, dqk } => {
                    println!("  [{i:>2}] mLSTM  heads={heads} dqk={dqk}")
                }
            }
        }
    }
    Ok(())
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

/// The layer's display name and its settings beyond in/out size.
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
