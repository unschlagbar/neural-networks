//! Distribution of backbone chunk counts over the corpus windows.
//!
//!   cargo run --release --example window_spans [corpus] [max_chunks_to_scan]
//!
//! A window whose word count fits in one `BACKBONE_CHUNK` produces a single span, and
//! the backbone's per-window carry setup is guarded on `spans.len() > 1` — so what
//! this counts is how often that guard is skipped.

use neural_networks::batches::ChunkedWordDataSet;
use neural_networks::config::{
    BACKBONE_CHUNK, CHUNK_BYTES, MAX_WINDOW_TOKENS, MIN_WORDS_PER_SEQ, TRAIN_DATA, WORDS_PER_SEQ,
};
use neural_networks::tokenizer_utf8::Utf8Tokenizer;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| TRAIN_DATA.to_string());
    let max_chunks: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3);

    let mut ds = ChunkedWordDataSet::open(
        Utf8Tokenizer::new(),
        &path,
        WORDS_PER_SEQ,
        MIN_WORDS_PER_SEQ,
        MAX_WINDOW_TOKENS,
        CHUNK_BYTES,
    );

    let mut hist: Vec<usize> = Vec::new();
    let (mut total, mut single) = (0usize, 0usize);
    for _ in 0..max_chunks {
        let Some(chunk) = ds.next_chunk() else { break };
        for w in chunk.iter() {
            let words = w.words.len();
            let spans = words.div_ceil(BACKBONE_CHUNK).max(1);
            if spans >= hist.len() {
                hist.resize(spans + 1, 0);
            }
            hist[spans] += 1;
            total += 1;
            if spans == 1 {
                single += 1;
            }
        }
    }

    println!("corpus {path:?}, WORDS_PER_SEQ={WORDS_PER_SEQ}, BACKBONE_CHUNK={BACKBONE_CHUNK}");
    println!("{total} windows scanned");
    for (spans, n) in hist.iter().enumerate().skip(1) {
        if *n > 0 {
            println!("  {spans:>2} span(s): {n:>7}  ({:.1}%)", 100.0 * *n as f64 / total as f64);
        }
    }
    println!(
        "\nsingle-span windows: {single} / {total} = {:.1}%",
        100.0 * single as f64 / total as f64
    );
}
