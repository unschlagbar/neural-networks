//! Prints per-tensor weight statistics of saved models so norm growth can be
//! compared across checkpoints: `cargo run --example weight_stats -- batch4 fable6`

use std::path::Path;

use neural_networks::{
    hierarchical::Hierarchical,
    nn::{
        embedding::EmbeddingLayer, linear::LinearLayer, linear_nb::LinearNBLayer,
        mlstm::MLSTMLayer, mlstm_block::MLSTMBlock, rms_norm::RMSNorm, slstm::SLSTMLayer,
        slstm_block::SLSTMBlock,
    },
    nn_layer::NnLayer,
    sequential::Sequential,
};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: weight_stats <model-name>...");
        std::process::exit(2);
    }
    for name in &args {
        let path = resolve(name);
        // Raw stacks so checkpoints of older decoder layouts stay readable.
        let stacks = Hierarchical::load_stacks(&path).unwrap();
        println!("\n================ {name} ================");
        println!("--- encoder ---");
        print_stack(&stacks.encoder_chars);
        println!("--- word_model ---");
        print_stack(&stacks.word_model);
        println!("--- char2_model ---");
        print_stack(&stacks.char2_model);
    }
}

fn resolve(name: &str) -> String {
    if Path::new(name).is_file() {
        name.to_string()
    } else {
        format!("models/{name}")
    }
}

fn print_stack(model: &Sequential) {
    for (i, layer) in model.layers.iter().enumerate() {
        print_layer(i, layer.as_ref());
    }
}

fn print_layer(i: usize, layer: &dyn NnLayer) {
    let any = layer.as_any();
    if let Some(l) = any.downcast_ref::<EmbeddingLayer>() {
        row(i, "Embedding", "weights", l.weights.as_slice());
    } else if let Some(l) = any.downcast_ref::<LinearNBLayer>() {
        row(i, "LinearNoBias", "weights", l.weights.as_slice());
    } else if let Some(l) = any.downcast_ref::<LinearLayer>() {
        row(i, "Linear", "weights", l.weights.as_slice());
        row(i, "Linear", "biases", &l.biases);
    } else if let Some(l) = any.downcast_ref::<RMSNorm>() {
        row(i, "RMSNorm", "gamma", &l.gamma);
    } else if let Some(l) = any.downcast_ref::<SLSTMBlock>() {
        row(i, "SLSTMBlock", "pre_norm1.gamma", &l.pre_norm1.gamma);
        slstm_rows(i, &l.cell);
        row(
            i,
            "SLSTMBlock",
            "post_cell_norm.gamma",
            &l.post_cell_norm.gamma,
        );
        row(i, "SLSTMBlock", "pre_norm2.gamma", &l.pre_norm2.gamma);
        row(i, "SLSTMBlock", "lin_gate.W", l.lin_gate.weights.as_slice());
        row(
            i,
            "SLSTMBlock",
            "lin_value.W",
            l.lin_value.weights.as_slice(),
        );
        row(i, "SLSTMBlock", "lin_down.W", l.lin_down.weights.as_slice());
    } else if let Some(l) = any.downcast_ref::<MLSTMBlock>() {
        row(i, "MLSTMBlock", "pre_norm1.gamma", &l.pre_norm1.gamma);
        mlstm_rows(i, &l.cell);
        row(i, "MLSTMBlock", "pre_norm2.gamma", &l.pre_norm2.gamma);
        row(i, "MLSTMBlock", "lin_gate.W", l.lin_gate.weights.as_slice());
        row(
            i,
            "MLSTMBlock",
            "lin_value.W",
            l.lin_value.weights.as_slice(),
        );
        row(i, "MLSTMBlock", "lin_down.W", l.lin_down.weights.as_slice());
    } else if let Some(l) = any.downcast_ref::<SLSTMLayer>() {
        slstm_rows(i, l);
    } else if let Some(l) = any.downcast_ref::<MLSTMLayer>() {
        mlstm_rows(i, l);
    } else {
        println!("  [{i:>2}] (tag {}) — no stats", layer.layer_tag());
    }
}

fn slstm_rows(i: usize, cell: &SLSTMLayer) {
    row(i, "sLSTM", "wz", cell.wz.as_slice());
    row(i, "sLSTM", "wi", cell.wi.as_slice());
    row(i, "sLSTM", "wf", cell.wf.as_slice());
    row(i, "sLSTM", "wo", cell.wo.as_slice());
    row(i, "sLSTM", "bz", &cell.bz);
    row(i, "sLSTM", "bi", &cell.bi);
    row(i, "sLSTM", "bf", &cell.bf);
    row(i, "sLSTM", "bo", &cell.bo);
    row(i, "sLSTM", "h_init", &cell.h_init);
    row(i, "sLSTM", "c_init", &cell.c_init);
}

fn mlstm_rows(i: usize, cell: &MLSTMLayer) {
    row(i, "mLSTM", "wq", cell.wq.as_slice());
    row(i, "mLSTM", "wk", cell.wk.as_slice());
    row(i, "mLSTM", "wv", cell.wv.as_slice());
    row(i, "mLSTM", "wo", cell.wo.as_slice());
    row(i, "mLSTM", "wi", cell.wi.as_slice());
    row(i, "mLSTM", "wf", cell.wf.as_slice());
    row(i, "mLSTM", "bq", &cell.bq);
    row(i, "mLSTM", "bk", &cell.bk);
    row(i, "mLSTM", "bv", &cell.bv);
    row(i, "mLSTM", "bo", &cell.bo);
    row(i, "mLSTM", "bi", &cell.bi);
    row(i, "mLSTM", "bf", &cell.bf);
    row(i, "mLSTM", "w_out.W", cell.w_out.weights.as_slice());
    row(i, "mLSTM", "w_out.b", &cell.w_out.biases);
    row(i, "mLSTM", "head_norm.gamma", &cell.head_norm.gamma);
}

/// Prints mean, RMS and max-abs of one tensor.
fn row(i: usize, layer: &str, tensor: &str, data: &[f32]) {
    let n = data.len().max(1) as f32;
    let mean: f32 = data.iter().sum::<f32>() / n;
    let rms = (data.iter().map(|x| x * x).sum::<f32>() / n).sqrt();
    let max = data.iter().fold(0.0, |a: f32, &x| a.max(x.abs()));
    println!(
        "  [{i:>2}] {layer:<12} {tensor:<22} mean {mean:>9.4}  rms {rms:>9.4}  max|x| {max:>9.4}"
    );
}
