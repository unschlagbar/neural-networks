use criterion::{Criterion, criterion_group, criterion_main};
use std::{rc::Rc, time::Duration};

use neural_networks::{
    batches::ChunkedWordDataSet, model::build_hierarchical_model, tokenizer_utf8::Utf8Tokenizer,
    training::TrainingState,
};

// Bench-local window shape: small enough for a quick iteration, big enough that
// the backbone unroll (words) and the decoder work dominate like in real training.
const WORDS_PER_SEQ: usize = 64;
const MIN_WORDS: usize = 8;
const MAX_TOKENS: usize = WORDS_PER_SEQ * 8;
/// Windows consumed per bench iteration (fixed workload).
const WINDOWS: usize = 4;

fn benchmark_hierarchical(c: &mut Criterion) {
    let mut group = c.benchmark_group("Hierarchical");
    group.sample_size(10);
    group.measurement_time(Duration::from_nanos(1));

    let tokenizer = Rc::new(Utf8Tokenizer::new());
    let vocab = tokenizer.vocab_size();

    let mut loader = ChunkedWordDataSet::open(
        tokenizer.clone(),
        "alice.txt",
        WORDS_PER_SEQ,
        MIN_WORDS,
        MAX_TOKENS,
        64 * 1024 * 1024, // alice.txt fits in one chunk
    );
    let data = loader
        .next_chunk()
        .expect("alice.txt yields no windows for the bench");
    assert!(
        data.len() >= WINDOWS,
        "alice.txt yields too few windows for the bench"
    );

    let mut model = build_hierarchical_model(vocab, tokenizer.clone());
    model.make_cache(WORDS_PER_SEQ, data.max_window_tokens());

    let mut state = TrainingState::new();
    state.print_interval = usize::MAX;
    state.save_interval = usize::MAX;

    group.bench_function("train_window", |b| {
        b.iter(|| model.train(data.iter().take(WINDOWS), &mut state));
    });

    // Same work minus the optimizer: a huge batch_size keeps `state.step` from
    // ever returning an lr, so `sgd_step` never runs. The difference to
    // train_window is the pure Muon/optimizer cost.
    let mut no_opt_state = TrainingState::new();
    no_opt_state.print_interval = usize::MAX;
    no_opt_state.save_interval = usize::MAX;
    no_opt_state.batch_size = usize::MAX;

    group.bench_function("forward_backward", |b| {
        b.iter(|| model.train(data.iter().take(WINDOWS), &mut no_opt_state));
    });

    group.bench_function("forward_eval", |b| {
        b.iter(|| model.eval_decode_loss(data.iter().take(WINDOWS)));
    });

    group.finish();
}

criterion_group!(benches, benchmark_hierarchical);
criterion_main!(benches);
