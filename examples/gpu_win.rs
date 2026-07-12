//! One hierarchical GPU training window at config.rs shapes — the number the
//! training loop actually pays per step.
//!
//!   cargo run --release --features cuda --example gpu_win [words]

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda`");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;

    use neural_networks::config::{CHAR_HIDDEN, WORD_BLOCKS, WORD_HIDDEN};
    use neural_networks::gpu::Gpu;
    use neural_networks::gpu::hierarchical::{HierCfg, Hierarchical};

    let words_n: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(2048);

    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU: {e}");
            return;
        }
    };

    let cfg = HierCfg {
        vocab: 100,
        hc: CHAR_HIDDEN,
        wh: WORD_HIDDEN,
        enc_blocks: 3,
        bb_blocks: WORD_BLOCKS,
        dec_blocks: 3,
        heads: 8,
        dqk: WORD_HIDDEN / 8,
        w_token: 99,
        cap: 30.0,
    };
    let mut model = Hierarchical::new(&gpu, &cfg);

    // Synthetic window: `words_n` words of 3..8 chars, like the real corpus.
    let mut tokens: Vec<usize> = Vec::new();
    let mut words: Vec<(usize, usize)> = Vec::new();
    for w in 0..words_n {
        let start = tokens.len();
        let len = 3 + (w % 5);
        for k in 0..len {
            tokens.push(1 + (w + k) % 90);
        }
        words.push((start, tokens.len()));
    }

    let loss = model.forward_backward(&gpu, &tokens, &words); // warm up
    let iters = 3;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = model.forward_backward(&gpu, &tokens, &words);
    }
    let per = t0.elapsed() / iters;
    println!(
        "{words_n} words / {} tokens: {per:.1?} per forward_backward (loss {loss:.3})",
        tokens.len()
    );
}
