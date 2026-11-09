//! Peak activation VRAM of one hierarchical training step, at the real config.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use neural_networks::config::{
        CHAR_HIDDEN, MAX_WORD_BYTES, OUT_HIDDEN, WORD_BLOCKS, WORD_HIDDEN, WORDS_PER_SEQ,
    };
    use neural_networks::gpu::Gpu;
    use neural_networks::gpu::hierarchical::{Hierarchical, ModelCfg};
    use neural_networks::cpu::optim::AdamCfg;
    use neural_networks::tokenizer_utf8::Utf8Tokenizer;

    let words: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(WORDS_PER_SEQ);

    let gpu = Gpu::new().expect("gpu");
    let free_mb = || cudarc::driver::result::mem_get_info().unwrap().0 as f64 / (1024.0 * 1024.0);
    let total_mb = cudarc::driver::result::mem_get_info().unwrap().1 as f64 / (1024.0 * 1024.0);

    println!(
        "device: {total_mb:.0} MB total, {:.0} MB free at start",
        free_mb()
    );
    println!("config: WORD_HIDDEN {WORD_HIDDEN}, WORD_BLOCKS {WORD_BLOCKS}, words/window {words}");

    let tok = Utf8Tokenizer::new();
    let cfg = ModelCfg {
        vocab: tok.vocab_size(),
        hc: CHAR_HIDDEN,
        wh: WORD_HIDDEN,
        enc_blocks: 2,
        bb_blocks: WORD_BLOCKS,
        dec_blocks: 2,
        heads: 8,
        dqk: 64,
        w_token: neural_networks::tokenizer_utf8::W_TOKEN as usize,
        cap: 30.0,
    };
    assert_eq!(
        CHAR_HIDDEN, OUT_HIDDEN,
        "decoder ties the encoder char table"
    );

    let mut model = Hierarchical::new(&gpu, cfg);
    let mut opt = AdamCfg::new(3e-4, 0.01);
    let after_init = free_mb();
    println!("after model init: {after_init:.0} MB free");

    let mut tokens = Vec::with_capacity(words * 4);
    let mut spans: Vec<std::range::Range<usize>> = Vec::with_capacity(words);
    for w in 0..words {
        let s = tokens.len();
        // Cycle the whole legal range so every length bucket gets built.
        let wlen = 1 + (w % MAX_WORD_BYTES);
        for k in 0..wlen {
            tokens.push(1 + (k % 90));
        }
        spans.push((s..tokens.len()).into());
    }

    for _ in 0..2 {
        model.forward_backward(&gpu, &tokens, &spans);
        opt.t += 1;
        model.step(&gpu, &opt);
    }
    gpu.stream.synchronize().unwrap();

    let loss = model.forward_backward(&gpu, &tokens, &spans);
    gpu.stream.synchronize().unwrap();
    model.step(&gpu, &opt);
    gpu.stream.synchronize().unwrap();
    let hwm = free_mb();

    let held = after_init - hwm;
    println!("\nloss {loss:.3}");
    println!("free at high-water mark  {hwm:8.0} MB");
    println!("activations + scratch    {held:8.0} MB   ({words} words)");
    println!(
        "  per word               {:8.1} KB",
        held * 1024.0 / words as f64
    );

    const REPS: usize = 5;
    let t0 = std::time::Instant::now();
    for _ in 0..REPS {
        model.forward_backward(&gpu, &tokens, &spans);
        opt.t += 1;
        model.step(&gpu, &opt);
    }
    gpu.stream.synchronize().unwrap();
    println!(
        "\nstep time                {:8.1} ms  (mean of {REPS})",
        t0.elapsed().as_secs_f64() * 1e3 / REPS as f64
    );
}
