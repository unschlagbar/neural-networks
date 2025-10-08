use std::ops::Range;
use std::rc::Rc;
use std::{
    io::{Write, stdin, stdout},
    time::{Duration, Instant},
};

use crate::data_set_loading::DataSet;
use crate::layer::Activation;
use crate::sequential::LayerBuilder::*;
use crate::tokenizer::Tokenizer;
use crate::{batches::Batches, sequential::Sequential};

pub mod batches;
pub mod data_set_loading;
pub mod jarvis;
pub mod layer;
pub mod lstm;
pub mod old;
pub mod saving;
pub mod sequential;
pub mod tokenizer;

const MODEL_LOC: &str = "rust_rnn";
const SEQ_LEN: Range<usize> = 300..300;
const LR: f32 = 0.00005;
const BATCH_SIZE: usize = 1;
const EPOCHS: usize = 25;

const MAX_LEN: usize = 2000;
const TEMPERATURE: f32 = 0.4;

const TRAIN: bool = true;

fn main() {
    if TRAIN {
        train();
    } else {
        sample();
    }
}

pub fn train() {
    let tokenizer = Rc::new(Tokenizer::new("charset.txt"));

    let mut model = if let Ok(model) = Sequential::load(MODEL_LOC) {
        println!("loaded model");
        model
    } else {
        let vocab = tokenizer.vocab_size();
        let activation = Activation::Relu;

        let layout = vec![
            Dense(256, activation),
            LSTM(256),
            LSTM(256),
            Dense(vocab, Activation::Softmax),
        ];

        let model = Sequential::new(layout, vocab);
        model.save(MODEL_LOC).unwrap();
        model
    };

    model.make_cache(SEQ_LEN.end);

    let mut iteration = 1;
    let mut j = 1;
    let mut i = 1;

    let mut total_time = Duration::from_secs(0);

    for _ in 0..EPOCHS {
        for data in DataSet::load_pumpkin_files(tokenizer.clone(), "../Pumpkin") {
            let start_time = Instant::now();
            model.train(
                Batches::new(&data, &[], SEQ_LEN),
                LR,
                &mut iteration,
                &mut j,
                BATCH_SIZE,
            );
            model.save(MODEL_LOC).unwrap();
            total_time += start_time.elapsed();

            println!("completed data {i}, in: {:?}", total_time / i);
            i += 1;
        }
    }
}

pub fn sample() {
    let tokenizer = Tokenizer::new("charset.txt");
    let mut model = Sequential::load(MODEL_LOC).unwrap();
    model.make_cache(1);

    loop {
        let mut input = String::new();
        stdin().read_line(&mut input).unwrap();

        let prefix = if !input.trim().is_empty() {
            tokenizer.to_tokens(input.trim())
        } else {
            Vec::new()
        };

        print!("response: ");
        let _ = model.sample(&prefix, MAX_LEN, TEMPERATURE, |token| {
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
