// The data-parallel word phases must not change what the hierarchical model
// computes: a replica runs each word with the master's weights and the same
// per-word op order, so the forward pass has to be bit-identical no matter how
// the words are partitioned. Training only differs in the order the per-word
// gradients are summed, so trajectories may drift by float rounding, not more.

use std::rc::Rc;

use neural_networks::{
    batches::ChunkedWordDataSet, hierarchical::Hierarchical, model::build_hierarchical_model,
    tokenizer::Tokenizer, training::TrainingState,
};

const WORDS_PER_SEQ: usize = 32;
const MIN_WORDS: usize = 4;
const MAX_TOKENS: usize = WORDS_PER_SEQ * 8;
const EVAL_WINDOWS: usize = 4;
const TRAIN_WINDOWS: usize = 8;

/// Load the saved model, eval, train a few windows, eval again. Builds all its
/// (non-Send) state locally so it can run inside any rayon pool; the pool it
/// runs under decides how many replicas the parallel word phases use.
fn eval_train_eval(model_path: &str) -> (f32, f32, f32) {
    let tokenizer = Rc::new(Tokenizer::new("charset.txt", false));
    let boundaries = tokenizer.boundary_tokens();
    let mut loader = ChunkedWordDataSet::open(
        tokenizer.clone(),
        "alice.txt",
        WORDS_PER_SEQ,
        MIN_WORDS,
        MAX_TOKENS,
        &boundaries,
        64 * 1024 * 1024, // alice.txt fits in one chunk
    );
    let data = loader.next_chunk().expect("alice.txt yields no windows");
    assert!(data.len() >= TRAIN_WINDOWS, "alice.txt yields too few windows");

    let mut model = Hierarchical::load(model_path, tokenizer).unwrap();
    model.make_cache(WORDS_PER_SEQ, data.max_window_tokens());

    let (c_before, w_before) = model.eval_decode_loss(data.iter().take(EVAL_WINDOWS));

    let mut state = TrainingState::new();
    state.print_interval = usize::MAX;
    state.save_interval = usize::MAX;
    model.train(data.iter().take(TRAIN_WINDOWS), &mut state);

    let (c_after, _) = model.eval_decode_loss(data.iter().take(EVAL_WINDOWS));
    (c_before, w_before, c_after)
}

#[test]
fn parallel_matches_single_thread() {
    // One set of random weights, shared by both runs via disk round-trip.
    let tokenizer = Rc::new(Tokenizer::new("charset.txt", false));
    let boundaries = tokenizer.boundary_tokens();
    let model = build_hierarchical_model(tokenizer.vocab_size(), boundaries, tokenizer);
    let path = std::env::temp_dir().join("hier_parallel_parity.model");
    let path = path.to_str().unwrap();
    model.save(path).unwrap();

    let single = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let (c1, w1, t1) = single.install(|| eval_train_eval(path));
    let (cn, wn, tn) = eval_train_eval(path); // global pool, all cores

    // Forward parity: bit-exact regardless of the word partition.
    assert_eq!(c1, cn, "parallel forward diverged from single-thread");
    assert_eq!(w1, wn, "parallel forward diverged from single-thread");

    // Training parity: same windows, same optimizer steps; only the gradient
    // reduction order differs, so allow float-rounding drift.
    assert!(
        (t1 - tn).abs() < 1e-3 * t1.abs().max(1.0),
        "post-training eval loss diverged: single-thread {t1} vs parallel {tn}"
    );
    assert!(
        t1 < c1,
        "training did not reduce the eval loss ({c1} -> {t1})"
    );
}
