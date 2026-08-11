//! Uninstrumented wall time of one hierarchical training step.
//!
//! `phase_table` synchronizes around every span, which inflates its total and makes
//! it useless for an A/B of a whole change. This runs the same model and shape with
//! no instrumentation at all and reports ms/step, so two builds (or two env
//! settings) can be compared directly.
//!
//! `WORDS` sets the window length (default `WORDS_PER_SEQ`), `ITERS` the timed
//! iteration count.
//!
//!   SLSTM_BF16=1 cargo run --release --features cuda --example step_time

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use neural_networks::config::{WORD_BLOCKS, WORD_HIDDEN, WORDS_PER_SEQ};
    use neural_networks::gpu::Gpu;
    use neural_networks::gpu::hierarchical::{HierCfg, Hierarchical};
    use neural_networks::nn2::optim::AdamCfg;

    let words: usize = std::env::var("WORDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(WORDS_PER_SEQ);
    let iters: u32 = std::env::var("ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);

    let gpu = Gpu::new().expect("gpu");
    let cfg = HierCfg {
        vocab: 260,
        hc: 256,
        wh: WORD_HIDDEN,
        enc_blocks: 2,
        bb_blocks: WORD_BLOCKS,
        dec_blocks: 2,
        heads: 16,
        dqk: WORD_HIDDEN / 16,
        w_token: 256,
        cap: 30.0,
    };
    let mut model = Hierarchical::new(&gpu, &cfg);
    let mut opt = AdamCfg::new(3e-4, 0.01);

    let mut tokens = Vec::new();
    let mut ws: Vec<std::range::Range<usize>> = Vec::new();
    for w in 0..words {
        let start = tokens.len();
        for c in 0..4 {
            tokens.push((w * 7 + c * 13) % 256);
        }
        ws.push((start..tokens.len()).into());
    }

    for it in 0..3 {
        model.forward_backward(&gpu, &tokens, &ws);
        opt.t = it + 1;
        model.step(&gpu, &opt);
    }
    gpu.stream.synchronize().unwrap();

    let t0 = std::time::Instant::now();
    for it in 0..iters as usize {
        model.forward_backward(&gpu, &tokens, &ws);
        opt.t = 4 + it as u64;
        model.step(&gpu, &opt);
    }
    gpu.stream.synchronize().unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

    println!(
        "WH={WORD_HIDDEN} blocks={WORD_BLOCKS} words={words} iters={iters}  =>  {ms:.1} ms/step"
    );
}
