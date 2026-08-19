//! Parameter count of the hierarchical model, per stack.
//!
//! Counts `params_mut` — the learnable tensors a checkpoint stores — so the total
//! is directly comparable to the size of a saved `models/` file.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use neural_networks::config::{
        CHAR_HIDDEN, LOGIT_SOFTCAP, OUT_HIDDEN, WORD_BLOCKS, WORD_HIDDEN,
    };
    use neural_networks::gpu::Gpu;
    use neural_networks::gpu::hierarchical::{Hierarchical, ModelCfg};
    use neural_networks::tokenizer_utf8::Utf8Tokenizer;

    let gpu = Gpu::new().expect("gpu");
    let tok = Utf8Tokenizer::new();
    let heads = 8;
    let cfg = ModelCfg {
        vocab: tok.vocab_size(),
        hc: CHAR_HIDDEN,
        wh: WORD_HIDDEN,
        enc_blocks: 5,
        bb_blocks: WORD_BLOCKS,
        dec_blocks: 5,
        heads,
        dqk: WORD_HIDDEN / heads,
        w_token: neural_networks::tokenizer_utf8::W_TOKEN as usize,
        cap: LOGIT_SOFTCAP,
    };
    assert_eq!(CHAR_HIDDEN, OUT_HIDDEN);
    let mut model = Hierarchical::new(&gpu, cfg);

    let (enc, bb, dec, other) = model.param_counts();
    let tot = enc + bb + dec + other;
    println!("config: wh {WORD_HIDDEN}, blocks {WORD_BLOCKS}, hc {CHAR_HIDDEN}");
    println!("encoder    {:>10.2} M", enc as f64 / 1e6);
    println!("backbone   {:>10.2} M", bb as f64 / 1e6);
    println!("decoder    {:>10.2} M", dec as f64 / 1e6);
    println!("other      {:>10.2} M", other as f64 / 1e6);
    println!(
        "TOTAL      {:>10.2} M  ({:.0} MB fp32)",
        tot as f64 / 1e6,
        tot as f64 * 4.0 / 1e6
    );
}
