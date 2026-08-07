//! Many hierarchical windows at VARYING word counts — the shape pattern real
//! training produces (trailing windows shrink to MIN_WORDS_PER_SEQ, and words are
//! grouped by length), which a fixed-size benchmark never exercises.
//!
//!   cargo run --release --features cuda --example vary_win
//!
//! Prints device memory after each window. A pool that reuses buffers only on an
//! exact size match grows here without bound; one that reuses by capacity levels
//! off at the high-water mark.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda`");
}

#[cfg(feature = "cuda")]
fn main() {
    use neural_networks::config::{CHAR_HIDDEN, WORD_BLOCKS, WORD_HIDDEN};
    use neural_networks::gpu::Gpu;
    use neural_networks::gpu::hierarchical::{HierCfg, Hierarchical};

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

    let build = |words_n: usize| {
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
        (tokens, words)
    };

    // Two passes over a spread of sizes: the second must not allocate anything the
    // first did not, since every shape is at or below the high-water mark by then.
    let sizes = [2048usize, 1500, 900, 300, 64, 16, 1200, 2048, 700, 2048];
    for pass in 0..2 {
        for &w in &sizes {
            let (tokens, words) = build(w);
            let loss = model.forward_backward(&gpu, &tokens, &words);
            let (free, total) = cudarc::driver::result::mem_get_info().expect("mem_get_info");
            println!(
                "pass {pass} words {w:5}  loss {loss:6.3}  in use {:>6.0} MB",
                (total - free) as f64 / 1e6
            );
        }
    }
}
