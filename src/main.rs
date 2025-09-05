use crate::data_set_loading::DataSet;
use crate::layer::{Activation, DenseLayer};
use crate::lstm::{LSTM, LSTMLayer};
use crate::sequential::Layer;
use crate::tokenizer::Tokenizer;
use crate::{batches::Batches, sequential::Sequential};

pub mod batches;
pub mod data_set_loading;
pub mod jarvis;
pub mod layer;
pub mod lstm;
pub mod mlp;
pub mod saving;
pub mod sequential;
pub mod tokenizer;

use std::{
    io::{Write, stdin, stdout},
    time::{Duration, Instant},
};

pub const MODEL_LOC: &str = "rust_rnn";
pub const MODEL2_LOC: &str = "rust_rnn2";

fn main() {
    //test_response_new();
    run_new();
}

pub fn test_response_old(model: &mut LSTM, tokenizer: &Tokenizer) {
    loop {
        let mut input = String::new();
        stdin().read_line(&mut input).unwrap();

        let prefix = if !input.trim().is_empty() {
            tokenizer.to_tokens(input.trim())
        } else {
            Vec::new()
        };

        print!("response: ");
        let _ = model.sample(&prefix, 2000, 0.3, |token| {
            if tokenizer.get_char(token) == "<END>" {
                false
            } else {
                print!("{}", tokenizer.get_char(token));
                stdout().flush().unwrap();
                true
            }
        });
    }
}

pub fn test_response_new() {
    let tokenizer = Tokenizer::new("charset.txt");
    let mut model = Sequential::load(MODEL2_LOC).unwrap();

    loop {
        let mut input = String::new();
        stdin().read_line(&mut input).unwrap();

        let prefix = if !input.trim().is_empty() {
            tokenizer.to_tokens(input.trim())
        } else {
            Vec::new()
        };

        print!("response: ");
        let _ = model.sample(&prefix, 2000, 0.4, |token| {
            if tokenizer.get_char(token) == "<END>" {
                false
            } else {
                print!("{}", tokenizer.get_char(token));
                stdout().flush().unwrap();
                true
            }
        });
    }
}

pub fn run_old() {
    let tokenizer = Tokenizer::new("charset.txt");

    let mut model = if let Ok(model) = LSTM::load(MODEL_LOC) {
        model
    } else {
        let model = LSTM::new(&[tokenizer.vocab_size(), 384, 384], tokenizer.vocab_size());
        //model.save(MODEL_LOC).unwrap();
        model
    };

    //test(&mut model, &tokenizer);

    let mut iteration = 1;
    let mut j = 1;

    let mut total_time = Duration::from_secs(0);
    let mut divider = 0;

    for _ in 0..400 {
        for (i, data) in DataSet::load_file(&tokenizer, "alice.txt")
            .into_iter()
            .enumerate()
        {
            let start_time = Instant::now();
            model.train(
                Batches::new(&data, &[], 100..100),
                0.001,
                &mut iteration,
                &mut j,
                1,
            );
            divider += 1;
            total_time += start_time.elapsed();
            println!("completed data {i}, in: {:?}", total_time / divider)
        }
    }
}

pub fn run_new() {
    let tokenizer = Tokenizer::new("charset.txt");

    let mut model = if let Ok(model) = Sequential::load(MODEL2_LOC) {
        println!("loading modell");
        model
    } else {
        let vocab = tokenizer.vocab_size();
        let activation = Activation::Relu;

        let model = Sequential::new(vec![
            Layer::Dense(DenseLayer::random(vocab, 256, activation)),
            Layer::Lstm(LSTMLayer::random(256, 256)),
            Layer::Dense(DenseLayer::random(256, 256, activation)),
            Layer::Lstm(LSTMLayer::random(256, 256)),
            Layer::Dense(DenseLayer::random(256, 256, activation)),
            Layer::Lstm(LSTMLayer::random(256, 256)),
            Layer::Dense(DenseLayer::random(256, vocab, Activation::Softmax)),
        ]);
        model.save(MODEL2_LOC).unwrap();
        model
    };

    //test(&mut model, &tokenizer);

    let mut iteration = 1;
    let mut j = 1;

    let mut total_time = Duration::from_secs(0);
    let mut divider = 0;

    for (i, data) in DataSet::load_rust_files(&tokenizer).into_iter().enumerate() {
        let start_time = Instant::now();
        model.train(
            Batches::new(&data, &[], 100..150),
            0.001,
            &mut iteration,
            &mut j,
            1,
        );
        model.save(MODEL2_LOC).unwrap();
        divider += 1;
        total_time += start_time.elapsed();
        println!("completed data {i}, in: {:?}", total_time / divider)
    }
}
