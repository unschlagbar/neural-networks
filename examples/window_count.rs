//! Count the corpus windows, to size DECAY_WINDOWS for a run.
//!
//! Does a full streaming pass (memory stays chunk-bounded). `--sample` instead
//! extrapolates from a few chunks — but on parquet that scales decompressed
//! bytes read against a *compressed* file size, so it underestimates badly
//! (measured 297k against an exact 447k). Only trust it on plain text.

use neural_networks::batches::ChunkedWordDataSet;
use neural_networks::config::{
    CHUNK_BYTES, MAX_WINDOW_TOKENS, MIN_WORDS_PER_SEQ, TRAIN_DATA, WORDS_PER_SEQ,
};
use neural_networks::tokenizer_utf8::Utf8Tokenizer;

fn main() {
    let sample = std::env::args().any(|a| a == "--sample");
    let path = std::env::args()
        .nth(1)
        .filter(|a| !a.starts_with("--"))
        .unwrap_or_else(|| TRAIN_DATA.to_string());

    let file_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    println!(
        "corpus {path:?} ({:.2} GB), WORDS_PER_SEQ={WORDS_PER_SEQ}",
        file_bytes as f64 / 1e9
    );

    let mut ds = ChunkedWordDataSet::open(
        Utf8Tokenizer::new(),
        &path,
        WORDS_PER_SEQ,
        MIN_WORDS_PER_SEQ,
        MAX_WINDOW_TOKENS,
        CHUNK_BYTES,
    );

    if !sample {
        let n = ds.count_windows();
        println!("\nexact: {n} windows");
        println!("  -> DECAY_WINDOWS ~= {n} for a single epoch");
        return;
    }

    // Sample a few chunks and scale by the fraction of the file they cover.
    const SAMPLE_CHUNKS: usize = 4;
    let mut windows = 0usize;
    let mut tokens = 0usize;
    let mut chunks = 0usize;
    while chunks < SAMPLE_CHUNKS {
        let Some(chunk) = ds.next_chunk() else { break };
        windows += chunk.len();
        tokens += chunk.total_tokens();
        chunks += 1;
    }

    if chunks == 0 {
        println!("no windows produced");
        return;
    }

    // Each chunk consumes ~CHUNK_BYTES of raw corpus.
    let sampled_bytes = (chunks * CHUNK_BYTES) as f64;
    let scale = (file_bytes as f64 / sampled_bytes).max(1.0);
    let est = (windows as f64 * scale) as usize;

    println!(
        "\nsampled {chunks} chunks: {windows} windows, {tokens} tokens \
         ({:.2} tokens/window)",
        tokens as f64 / windows as f64
    );
    println!("estimated total: ~{est} windows for one epoch");
    println!("  -> DECAY_WINDOWS ~= {est}");
}
