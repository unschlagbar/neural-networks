use criterion::{Criterion, criterion_group, criterion_main};
use neural_networks::activations::{Linear, Relu};
use std::{rc::Rc, time::Duration};

use neural_networks::nn_layer::SequentialBuilder;
use neural_networks::{
    self, batches::RandomBatches, data_set_loading::DataSet, tokenizer::Tokenizer,
};

const HIDDEN_SIZE: usize = 128;
const SEQ_LEN: usize = 25;
const LR: f32 = 0.01;
const BATCH_SIZE: usize = 2;

pub const VOCAB: &[char] = &[
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
    't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'ä', 'ö', 'ü', 'Ä', 'Ö',
    'Ü', 'ß', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '.', ',', ':', ';', '(', ')',
    '{', '}', '[', ']', '+', '-', '*', '/', '=', '<', '>', '!', '?', '&', '|', '^', '%', '#', '@',
    '~', '"', '\'', '`', '\\', '_', '$', '“', '„', '🦀', '·', '\n',
];

pub fn train(tokenizer: Rc<Tokenizer>, raw_data: &Vec<Vec<u16>>) {
    let vocab = tokenizer.vocab_size();

    let mut model = SequentialBuilder::new(vocab)
        .dense(HIDDEN_SIZE, Relu)
        .lstm(HIDDEN_SIZE)
        .lstm(HIDDEN_SIZE)
        .dense(vocab, Linear)
        .softmax()
        .build();

    model.make_cache(SEQ_LEN);

    let mut iteration = 0;
    let mut j = 0;

    for data in RandomBatches::new(SEQ_LEN, BATCH_SIZE, raw_data).take(2) {
        model.train(data.into_iter(), LR, &mut iteration, &mut j, BATCH_SIZE);
    }
}

fn benchmark_lstm_behavior(c: &mut Criterion) {
    let mut group = c.benchmark_group("LSTM");
    group.sample_size(20);
    group.measurement_time(Duration::from_nanos(1));

    let tokenizer = Rc::new(Tokenizer::new_vocab(VOCAB, false));
    let raw_data = DataSet::load_from_dir(tokenizer.clone(), "political_speeches/").to_raw_data();

    group.bench_function("lstm_training", |b| {
        b.iter(|| train(tokenizer.clone(), &raw_data));
    });

    group.finish();
}

criterion_group!(benches, benchmark_lstm_behavior);
criterion_main!(benches);
