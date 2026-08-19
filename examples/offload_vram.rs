//! Peak activation VRAM of one hierarchical training step, at the real config.
//!
//! The number that matters for the offload plan: device memory held *between*
//! forward and backward, which is what scales linearly in T and caps sequence
//! length. Measured as (free before forward) − (free after forward, before
//! backward), so it excludes weights, optimizer moments and the pooled scratch that
//! backward returns.
//!
//! Re-run after each offload step to check the predicted saving actually landed.
//!
//! Run: cargo run --release --features cuda --example offload_vram [words]

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
    use neural_networks::nn2::optim::AdamCfg;
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

    // A window of `words` words with **mixed** lengths, 1..=MAX_WORD_BYTES.
    //
    // Deliberately not a fixed 4 bytes: `segment::word_ends` yields a spread, and the
    // encoder/decoder bucket words by length into `[words, tmax]` rectangles whose
    // sizes then set the high-water mark of every `Buf` and `Pool` behind them. A
    // uniform window exercises exactly one rectangle shape and badly under-reports
    // what a real corpus makes resident.
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

    // Warm up: first step allocates the caches and grows the pools, so its delta is
    // not the steady-state figure.
    for _ in 0..2 {
        model.forward_backward(&gpu, &tokens, &spans);
        opt.t += 1;
        model.step(&gpu, &opt);
    }
    gpu.stream.synchronize().unwrap();

    // NOTE: the CUDA allocator caches freed blocks, so `mem_get_info` reports the
    // process high-water mark, not live usage — a forward/backward delta reads as
    // zero once the pools are warm. The meaningful figure is therefore the
    // high-water mark itself, compared across builds or across `words`.
    let loss = model.forward_backward(&gpu, &tokens, &spans);
    gpu.stream.synchronize().unwrap();
    model.step(&gpu, &opt);
    gpu.stream.synchronize().unwrap();
    let hwm = free_mb();

    let held = after_init - hwm;
    println!("\nloss {loss:.4}");
    println!("free at high-water mark  {hwm:8.0} MB");
    println!("activations + scratch    {held:8.0} MB   ({words} words)");
    println!(
        "  per word               {:8.1} KB",
        held * 1024.0 / words as f64
    );

    // Step time, so a memory win that costs throughput shows up here rather than in
    // a training run days later. Offload's transfers are meant to hide behind
    // compute; if they do not, this is where it surfaces.
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
