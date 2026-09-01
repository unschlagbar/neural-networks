//! Repeated `eval_loss` at varying window shapes, reporting host RSS and the
//! device pool after each — the shape a leak in the forward-only path shows up as.
//!
//!   cargo run --release --features cuda --example eval_leak [iters] [--train]

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda`");
}

#[cfg(feature = "cuda")]
fn main() {
    use neural_networks::config::{CHAR_HIDDEN, WORD_BLOCKS, WORD_HIDDEN};
    use neural_networks::gpu::Gpu;
    use neural_networks::gpu::hierarchical::{Hierarchical, ModelCfg};

    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(20);
    let train = std::env::args().any(|a| a == "--train");

    let rss_mb = || {
        let s = std::fs::read_to_string("/proc/self/statm").unwrap();
        let pages: f64 = s.split_whitespace().nth(1).unwrap().parse().unwrap();
        pages * 4096.0 / (1024.0 * 1024.0)
    };
    let smaps = |key: &str| -> f64 {
        let s = std::fs::read_to_string("/proc/self/smaps_rollup").unwrap_or_default();
        s.lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0)
            / 1024.0
    };
    let driver_mb = || {
        let (free, total) = cudarc::driver::result::mem_get_info().expect("mem_get_info");
        (total - free) as f64 / (1024.0 * 1024.0)
    };

    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU: {e}");
            return;
        }
    };
    let cfg = ModelCfg {
        vocab: 266,
        hc: CHAR_HIDDEN,
        wh: WORD_HIDDEN,
        enc_blocks: 4,
        bb_blocks: WORD_BLOCKS,
        dec_blocks: 4,
        heads: 8,
        dqk: WORD_HIDDEN / 8,
        w_token: 256,
        cap: 30.0,
    };
    let mut model = Hierarchical::new(&gpu, cfg);
    if !train {
        model.set_offload(&gpu, false);
    }

    for i in 0..iters {
        // Real windows vary in word count and length spread; a fixed shape hides
        // anything that is kept per shape.
        let words_n = 1024 + (i % 7) * 400;
        let mut tokens: Vec<usize> = Vec::new();
        let mut words: Vec<std::range::Range<usize>> = Vec::new();
        for w in 0..words_n {
            let start = tokens.len();
            let len = 2 + (w + i) % 9;
            for k in 0..len {
                tokens.push(1 + (w + k) % 200);
            }
            words.push((start..tokens.len()).into());
        }
        if train {
            let _ = model.forward_backward(&gpu, &tokens, &words);
            if std::env::args().any(|a| a == "--drop") {
                model.drop_all_act(&gpu);
            }
        } else {
            let _ = model.eval_loss(&gpu, &tokens, &words);
        }
        println!(
            "{i:3}  words {words_n:5}  rss {:8.1} MB  anon {:8.1} MB  file {:8.1} MB  device {:8.1} MB  pool {:8.1} MB",
            rss_mb(),
            smaps("Anonymous:"),
            smaps("Rss:") - smaps("Anonymous:"),
            driver_mb(),
            neural_networks::gpu::pool_used_mb().unwrap_or(0.0),
        );
    }
}
