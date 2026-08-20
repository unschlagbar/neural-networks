//! Write the seed checkpoint `grad_dump` replays. Separate from the probe itself so
//! the probe only ever calls `load`, and therefore compiles against older trees whose
//! model-construction API differed.
//!
//!   cargo run --release --features cuda --example grad_seed -- <path> [enc] [bb] [dec]

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use neural_networks::config::{CHAR_HIDDEN, OUT_HIDDEN, WORD_HIDDEN};
    use neural_networks::gpu::Gpu;
    use neural_networks::gpu::hierarchical::{Hierarchical, ModelCfg};
    use neural_networks::tokenizer_utf8::Utf8Tokenizer;

    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: grad_seed <path> [enc] [bb] [dec]");
    let mut next = |d: usize| args.next().and_then(|s| s.parse().ok()).unwrap_or(d);

    assert_eq!(CHAR_HIDDEN, OUT_HIDDEN, "decoder ties the encoder char table");
    let gpu = Gpu::new().expect("gpu");
    let tok = Utf8Tokenizer::new();
    let cfg = ModelCfg {
        vocab: tok.vocab_size(),
        hc: CHAR_HIDDEN,
        wh: WORD_HIDDEN,
        enc_blocks: next(4),
        bb_blocks: next(9),
        dec_blocks: next(4),
        heads: 8,
        dqk: WORD_HIDDEN / 8,
        w_token: neural_networks::tokenizer_utf8::W_TOKEN as usize,
        cap: 30.0,
    };
    Hierarchical::new(&gpu, cfg).save(&gpu, &path, &[]).expect("save seed");
    println!("wrote {path}");
}
