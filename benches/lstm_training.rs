use criterion::{Criterion, criterion_group, criterion_main};
use neural_networks::batches::PreparedDataSet;
use neural_networks::training::TrainingState;
use std::{rc::Rc, time::Duration};

use neural_networks::nn_layer::SequentialBuilder;
use neural_networks::{self, tokenizer::Tokenizer};

const HIDDEN_SIZE: usize = 128;
const SEQ_LEN: usize = 25;
const LR: f32 = 0.005;
const BATCH_SIZE: usize = 1;

pub const VOCAB: &[char] = &[
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
    't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'ä', 'ö', 'ü', 'Ä', 'Ö',
    'Ü', 'ß', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '.', ',', ':', ';', '(', ')',
    '{', '}', '[', ']', '+', '-', '*', '/', '=', '<', '>', '!', '?', '&', '|', '^', '%', '#', '@',
    '~', '"', '\'', '`', '\\', '_', '$', '“', '„', '🦀', '·', '\n',
];

pub fn train(tokenizer: Rc<Tokenizer>, data: &PreparedDataSet) {
    let vocab = tokenizer.vocab_size();

    let mut model = SequentialBuilder::new(vocab)
        .embedding(HIDDEN_SIZE)
        .slstm(HIDDEN_SIZE)
        .slstm(HIDDEN_SIZE)
        .linear(vocab)
        .softmax()
        .build();

    model.make_cache(SEQ_LEN);

    let mut state = TrainingState::new();
    state.lr = LR;
    state.batch_size = BATCH_SIZE;
    state.print_interval = 1;

    model.train(data.iter().take(2), &mut state);
}

fn benchmark_lstm_behavior(c: &mut Criterion) {
    let mut group = c.benchmark_group("LSTM");
    group.sample_size(20);
    group.measurement_time(Duration::from_nanos(1));

    let tokenizer = Rc::new(Tokenizer::new_vocab(VOCAB, false));
    let boundaries = tokenizer.boundary_tokens();
    let data =
        PreparedDataSet::from_dir(&tokenizer, "data/political_speeches/", SEQ_LEN, &boundaries);

    group.bench_function("lstm_training", |b| {
        b.iter(|| train(tokenizer.clone(), &data));
    });

    group.finish();
}

criterion_group!(benches, benchmark_lstm_behavior);
criterion_main!(benches);
