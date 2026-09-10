//! Window shape of a corpus after packing: how full the windows are, how many
//! documents share one, and how much of the corpus the border masks drop.
//!
//! `cargo run --release --example win_carry_stats -- <shard> [chunks]`

use neural_networks::{
    batches::ChunkedWordDataSet,
    config::{CHUNK_BYTES, MAX_WINDOW_TOKENS, MIN_WORDS_PER_SEQ, WORDS_PER_SEQ},
    tokenizer_utf8::Utf8Tokenizer,
};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: win_carry_stats <file> [chunks]");
    let chunks: usize = args.next().map_or(2, |s| s.parse().unwrap());

    let mut data = ChunkedWordDataSet::open(
        Utf8Tokenizer::new(),
        &path,
        WORDS_PER_SEQ,
        MIN_WORDS_PER_SEQ,
        MAX_WINDOW_TOKENS,
        CHUNK_BYTES,
    );

    let (mut n, mut cont, mut words, mut toks, mut borders) = (0usize, 0usize, 0usize, 0usize, 0);
    let mut short = 0usize;
    // Tokens per window: the shape the caches are sized to, and the only axis packing
    // does *not* make constant — a window is K words, and a word is 1 to
    // `MAX_WORD_BYTES` tokens.
    let mut tok_per_win: Vec<usize> = Vec::new();
    // Document pieces per window — a window inside one long document holds one.
    let mut docs_hist = [0usize; 10];
    for _ in 0..chunks {
        let Some(chunk) = data.next_chunk() else {
            break;
        };
        for b in chunk.iter() {
            let k = b.words.len();
            n += 1;
            words += k;
            toks += b.tokens.len();
            tok_per_win.push(b.tokens.len());
            short += usize::from(k < WORDS_PER_SEQ);
            // Word 0 is not a border inside the window — it is where the state starts —
            // and the trailing offset is the document after this window.
            let cuts = b.doc_starts.iter().filter(|&&d| d > 0 && d < k).count();
            borders += cuts;
            docs_hist[(cuts + 1).min(9)] += 1;
            cont += usize::from(b.continues);
        }
    }
    println!("\n{n} windows, {words} words, {toks} tokens");
    println!(
        "  mean {:.0} words/window, {:.0}% of the {WORDS_PER_SEQ}-word cap ({short} short)",
        words as f32 / n.max(1) as f32,
        100.0 * words as f32 / (n.max(1) * WORDS_PER_SEQ) as f32
    );
    println!(
        "  continuations: {cont} ({:.1}% of windows)",
        100.0 * cont as f32 / n.max(1) as f32
    );
    tok_per_win.sort_unstable();
    if let (Some(&lo), Some(&hi)) = (tok_per_win.first(), tok_per_win.last()) {
        let mean = toks as f64 / n.max(1) as f64;
        let pick = |q: f64| tok_per_win[((n - 1) as f64 * q) as usize];
        println!(
            "  tokens/window: mean {mean:.0} ({:.2} per word), min {lo}, \
             p5 {}, p50 {}, p95 {}, max {hi} ({MAX_WINDOW_TOKENS} cap)",
            toks as f64 / words.max(1) as f64,
            pick(0.05),
            pick(0.50),
            pick(0.95),
        );
    }
    println!(
        "  document borders inside a window: {borders} ({:.2} per window), \
         costing {:.3}% of all predictions",
        borders as f32 / n.max(1) as f32,
        // Two masked words per border: the `<END>` and the word after it.
        200.0 * borders as f32 / words.max(1) as f32
    );
    println!("  docs/window   windows     share");
    for (d, &c) in docs_hist.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let label = if d == 9 {
            "9+".to_string()
        } else {
            d.to_string()
        };
        println!(
            "  {label:>11}   {c:>7}  {:>8.1}%",
            100.0 * c as f32 / n as f32
        );
    }
}
