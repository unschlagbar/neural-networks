//! Step time against backbone chunk length, at the real model shape.
//!
//! Chunking the backbone bounds its activation memory but adds per-chunk work — the
//! state carries, the slice/concat at the chunk borders, and (under offload) one park
//! generation per chunk. This is the A/B that says what that costs.
//!
//! `cargo run --release --features cuda --example chunk_ab -- [words]`

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use neural_networks::config::{BACKBONE_CHUNK, GROUP_MAX_ROWS, WORD_BLOCKS, WORD_HIDDEN};
    use neural_networks::gpu::Gpu;
    use neural_networks::gpu::hierarchical::{HierCfg, Hierarchical};
    use neural_networks::nn2::optim::AdamCfg;

    let n_words: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);

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

    let mut tokens = Vec::new();
    let mut words: Vec<std::range::Range<usize>> = Vec::new();
    for w in 0..n_words {
        let start = tokens.len();
        for c in 0..4 {
            tokens.push((w * 7 + c * 13) % 256);
        }
        words.push((start..tokens.len()).into());
    }

    println!("{n_words} words, {} tokens, WH={WORD_HIDDEN}, {WORD_BLOCKS} blocks", tokens.len());
    println!("{:>16}  {:>10}  {:>10}", "config", "ms/step", "vs old");

    let mut base = 0.0f64;
    // Group cap sweep at the configured backbone chunk: `0` is uncapped (one group per
    // length bucket), then progressively tighter caps.
    // Backbone chunk sweep with the group cap at its configured value: `usize::MAX`
    // is the unchunked backbone (one span), the pre-chunking behaviour.
    // (label, backbone chunk, group cap) — "old" is the pre-chunking behaviour:
    // unchunked backbone, uncapped groups.
    for (label, chunk, cap) in [
        ("old (both off)", usize::MAX, 0usize),
        ("cap only", usize::MAX, GROUP_MAX_ROWS),
        ("chunk only", BACKBONE_CHUNK, 0),
        ("both (current)", BACKBONE_CHUNK, GROUP_MAX_ROWS),
    ] {
        let mut model = Hierarchical::new(&gpu, &cfg);
        model.set_bb_chunk(Some(chunk));
        model.set_group_cap(Some(cap));
        let mut opt = AdamCfg::new(3e-4, 0.01);

        for it in 0..3 {
            model.forward_backward(&gpu, &tokens, &words);
            opt.t = it + 1;
            model.step(&gpu, &opt);
        }

        let iters = 5u64;
        gpu.stream.synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for it in 0..iters {
            model.forward_backward(&gpu, &tokens, &words);
            opt.t = 4 + it;
            model.step(&gpu, &opt);
        }
        gpu.stream.synchronize().unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        if label == "old (both off)" {
            base = ms;
            println!("{label:>16}  {ms:>10.1}  {:>10}", "-");
        } else {
            println!("{label:>16}  {ms:>10.1}  {:>9.2}x", ms / base);
        }
    }
}
