// Sanity check against the real corpus: stream it in small chunks and verify
// the pass completes with consistent counts. Ignored by default — run with
// `cargo test --release --test chunked_real_corpus -- --ignored --nocapture`.
use std::rc::Rc;

use neural_networks::{batches::ChunkedWordDataSet, config, tokenizer::Tokenizer};

#[test]
#[ignore]
fn stream_real_corpus() {
    let tokenizer = Rc::new(Tokenizer::new("charset.txt", false));
    let boundaries = tokenizer.boundary_tokens();
    let start = std::time::Instant::now();
    let mut loader = ChunkedWordDataSet::open(
        tokenizer,
        config::TRAIN_DATA,
        config::WORDS_PER_SEQ,
        config::MIN_WORDS_PER_SEQ,
        config::MAX_WINDOW_TOKENS,
        &boundaries,
        4 * 1024 * 1024,
    );
    let (mut chunks, mut windows, mut tokens, mut max_span) = (0, 0, 0usize, 0);
    while let Some(chunk) = loader.next_chunk() {
        chunks += 1;
        windows += chunk.len();
        tokens += chunk.total_tokens();
        max_span = max_span.max(chunk.max_window_tokens());
    }
    println!(
        "{chunks} chunks, {windows} windows, {tokens} tokens, max span {max_span}, full pass {:.2?}",
        start.elapsed()
    );
    assert!(windows > 0);
    assert_eq!(loader.count_windows(), windows);
}
