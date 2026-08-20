//! Per-tensor relative change between two checkpoints, so a training blowup can be
//! localised to the tensors that actually moved:
//!
//!   cargo run --release --example weight_delta -- <before> <after>
//!
//! Prints `||b-a|| / ||a||` and the mean shift for every tensor, worst first. RMS
//! alone is nearly blind here: Muon's update is orthogonalised, so it rotates weights
//! without changing their norm.

use std::path::Path;

use neural_networks::{
    hierarchical::Hierarchical,
    nn::{
        embedding::EmbeddingLayer, linear::LinearLayer, linear_nb::LinearNBLayer,
        mlstm::MLSTMLayer, mlstm_block::MLSTMBlock, rms_norm::RMSNorm, slstm::SLSTMLayer,
        slstm_block::SLSTMBlock,
    },
    sequential::Sequential,
};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("usage: weight_delta <before> <after>");
        std::process::exit(2);
    }
    let a = Hierarchical::load_stacks(&resolve(&args[0])).unwrap();
    let b = Hierarchical::load_stacks(&resolve(&args[1])).unwrap();

    let mut rows = Vec::new();
    for (stage, sa, sb) in [
        ("encoder", &a.encoder_chars, &b.encoder_chars),
        ("word_model", &a.word_model, &b.word_model),
        ("char2_model", &a.char2_model, &b.char2_model),
    ] {
        collect(stage, sa, sb, &mut rows);
    }
    rows.sort_by(|x: &Row, y: &Row| y.rel.total_cmp(&x.rel));
    println!("{:<44} {:>10} {:>12} {:>12}", "tensor", "rel |d|", "mean a", "mean b");
    for r in rows.iter().take(40) {
        println!("{:<44} {:>10.4} {:>12.5} {:>12.5}", r.name, r.rel, r.mean_a, r.mean_b);
    }
}

struct Row {
    name: String,
    rel: f32,
    mean_a: f32,
    mean_b: f32,
}

fn resolve(name: &str) -> String {
    if Path::new(name).is_file() { name.to_string() } else { format!("models/{name}") }
}

fn collect(stage: &str, a: &Sequential, b: &Sequential, out: &mut Vec<Row>) {
    for (i, (la, lb)) in a.layers.iter().zip(&b.layers).enumerate() {
        let mut push = |what: &str, x: &[f32], y: &[f32]| {
            let na: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
            let d: f32 = x.iter().zip(y).map(|(p, q)| (q - p) * (q - p)).sum::<f32>().sqrt();
            out.push(Row {
                name: format!("{stage}[{i}] {what}"),
                rel: if na > 0.0 { d / na } else { d },
                mean_a: x.iter().sum::<f32>() / x.len().max(1) as f32,
                mean_b: y.iter().sum::<f32>() / y.len().max(1) as f32,
            });
        };
        let (aa, ab) = (la.as_any(), lb.as_any());
        if let (Some(p), Some(q)) =
            (aa.downcast_ref::<EmbeddingLayer>(), ab.downcast_ref::<EmbeddingLayer>())
        {
            push("Embedding.W", p.weights.as_slice(), q.weights.as_slice());
        } else if let (Some(p), Some(q)) =
            (aa.downcast_ref::<LinearNBLayer>(), ab.downcast_ref::<LinearNBLayer>())
        {
            push("LinearNoBias.W", p.weights.as_slice(), q.weights.as_slice());
        } else if let (Some(p), Some(q)) =
            (aa.downcast_ref::<LinearLayer>(), ab.downcast_ref::<LinearLayer>())
        {
            push("Linear.W", p.weights.as_slice(), q.weights.as_slice());
            push("Linear.b", &p.biases, &q.biases);
        } else if let (Some(p), Some(q)) =
            (aa.downcast_ref::<RMSNorm>(), ab.downcast_ref::<RMSNorm>())
        {
            push("RMSNorm.gamma", &p.gamma, &q.gamma);
        } else if let (Some(p), Some(q)) =
            (aa.downcast_ref::<SLSTMBlock>(), ab.downcast_ref::<SLSTMBlock>())
        {
            push("pre_norm1.gamma", &p.pre_norm1.gamma, &q.pre_norm1.gamma);
            slstm(&mut push, &p.cell, &q.cell);
            push("post_cell_norm.gamma", &p.post_cell_norm.gamma, &q.post_cell_norm.gamma);
            push("pre_norm2.gamma", &p.pre_norm2.gamma, &q.pre_norm2.gamma);
            push("lin_gate.W", p.lin_gate.weights.as_slice(), q.lin_gate.weights.as_slice());
            push("lin_value.W", p.lin_value.weights.as_slice(), q.lin_value.weights.as_slice());
            push("lin_down.W", p.lin_down.weights.as_slice(), q.lin_down.weights.as_slice());
        } else if let (Some(p), Some(q)) =
            (aa.downcast_ref::<MLSTMBlock>(), ab.downcast_ref::<MLSTMBlock>())
        {
            push("pre_norm1.gamma", &p.pre_norm1.gamma, &q.pre_norm1.gamma);
            mlstm(&mut push, &p.cell, &q.cell);
            push("pre_norm2.gamma", &p.pre_norm2.gamma, &q.pre_norm2.gamma);
            push("lin_gate.W", p.lin_gate.weights.as_slice(), q.lin_gate.weights.as_slice());
            push("lin_value.W", p.lin_value.weights.as_slice(), q.lin_value.weights.as_slice());
            push("lin_down.W", p.lin_down.weights.as_slice(), q.lin_down.weights.as_slice());
        } else if let (Some(p), Some(q)) =
            (aa.downcast_ref::<SLSTMLayer>(), ab.downcast_ref::<SLSTMLayer>())
        {
            slstm(&mut push, p, q);
        } else if let (Some(p), Some(q)) =
            (aa.downcast_ref::<MLSTMLayer>(), ab.downcast_ref::<MLSTMLayer>())
        {
            mlstm(&mut push, p, q);
        }
    }
}

fn slstm(push: &mut impl FnMut(&str, &[f32], &[f32]), a: &SLSTMLayer, b: &SLSTMLayer) {
    push("sLSTM.wz", a.wz.as_slice(), b.wz.as_slice());
    push("sLSTM.wi", a.wi.as_slice(), b.wi.as_slice());
    push("sLSTM.wf", a.wf.as_slice(), b.wf.as_slice());
    push("sLSTM.wo", a.wo.as_slice(), b.wo.as_slice());
    push("sLSTM.bz", &a.bz, &b.bz);
    push("sLSTM.bi", &a.bi, &b.bi);
    push("sLSTM.bf", &a.bf, &b.bf);
    push("sLSTM.bo", &a.bo, &b.bo);
}

fn mlstm(push: &mut impl FnMut(&str, &[f32], &[f32]), a: &MLSTMLayer, b: &MLSTMLayer) {
    push("mLSTM.wq", a.wq.as_slice(), b.wq.as_slice());
    push("mLSTM.wk", a.wk.as_slice(), b.wk.as_slice());
    push("mLSTM.wv", a.wv.as_slice(), b.wv.as_slice());
    push("mLSTM.wo", a.wo.as_slice(), b.wo.as_slice());
    push("mLSTM.wi", a.wi.as_slice(), b.wi.as_slice());
    push("mLSTM.wf", a.wf.as_slice(), b.wf.as_slice());
    push("mLSTM.bq", &a.bq, &b.bq);
    push("mLSTM.bk", &a.bk, &b.bk);
    push("mLSTM.bv", &a.bv, &b.bv);
    push("mLSTM.bo", &a.bo, &b.bo);
    push("mLSTM.bi", &a.bi, &b.bi);
    push("mLSTM.bf", &a.bf, &b.bf);
    push("mLSTM.head_norm.gamma", &a.head_norm.gamma, &b.head_norm.gamma);
}
