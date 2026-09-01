//! Forward-only evaluation against a full training window, at config.rs shapes:
//! what `hvg` (`Hierarchical::eval_loss`) saves over a training step.
//!
//!   cargo run --release --features cuda --example eval_win [words] [iters]
//!
//! The two are interleaved rather than run in blocks — the SM clock drifts far
//! enough over a run to swamp the effect if they are not.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda`");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::time::{Duration, Instant};

    use neural_networks::config::{CHAR_HIDDEN, WORD_BLOCKS, WORD_HIDDEN};
    use neural_networks::gpu::Gpu;
    use neural_networks::gpu::hierarchical::{Hierarchical, ModelCfg};

    let arg = |i: usize, d: usize| {
        std::env::args()
            .nth(i)
            .and_then(|a| a.parse().ok())
            .unwrap_or(d)
    };
    let words_n = arg(1, 2048);
    let iters = arg(2, 3);

    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU: {e}");
            return;
        }
    };

    let cfg = ModelCfg {
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
    let mut model = Hierarchical::new(&gpu, cfg);

    // Synthetic window: `words_n` words of 3..8 chars, like the real corpus.
    let mut tokens: Vec<usize> = Vec::new();
    let mut words: Vec<std::range::Range<usize>> = Vec::new();
    for w in 0..words_n {
        let start = tokens.len();
        let len = 3 + (w % 5);
        for k in 0..len {
            tokens.push(1 + (w + k) % 90);
        }
        words.push((start..tokens.len()).into());
    }

    let train_loss = model.forward_backward(&gpu, &tokens, &words); // warm up
    let eval_loss = model.eval_loss(&gpu, &tokens, &words);

    let mut train = Duration::ZERO;
    let mut eval = Duration::ZERO;
    for _ in 0..iters {
        let t = Instant::now();
        let _ = model.forward_backward(&gpu, &tokens, &words);
        train += t.elapsed();
        let t = Instant::now();
        let _ = model.eval_loss(&gpu, &tokens, &words);
        eval += t.elapsed();
    }
    let (train, eval) = (train / iters as u32, eval / iters as u32);
    println!(
        "{words_n} words / {} tokens\n  forward_backward {train:.1?}\n  eval_loss        {eval:.1?}  ({:.2}x)\n  loss {train_loss:.4} / {eval_loss:.4}",
        tokens.len(),
        train.as_secs_f64() / eval.as_secs_f64(),
    );
}
