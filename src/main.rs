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

pub mod aud_text;
pub mod batches;
pub mod data_set_loading;
pub mod jarvis;
pub mod layer;
pub mod lstm;
pub mod mlp;
pub mod old;
pub mod saving;
pub mod sequential;
pub mod tokenizer;

const MODEL_LOC: &str = "rust_rnn";
const SEQ_LEN: Range<usize> = 300..300;
const LR: f32 = 0.00001;
const BATCH_SIZE: usize = 1;
const EPOCHS: usize = 2000;

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
            Dense(256, activation),
            LSTM(256),
            Dense(256, activation),
            LSTM(256),
            Dense(256, activation),
            Dense(vocab, Activation::Softmax),
        ];

        let model = Sequential::new(layout, vocab);
        model.save(MODEL_LOC).unwrap();
        model
    };

    let mut iteration = 1;
    let mut j = 1;
    let mut i = 1;

    let mut total_time = Duration::from_secs(0);
    let mut divider = 0;

    for _ in 0..EPOCHS {
        for data in DataSet::load_pumpkin_files(tokenizer.clone(), r"C:\Users\e7438\Desktop\Pumpkin".into()) {
            let start_time = Instant::now();
            model.train(
                Batches::new(&data, &[], SEQ_LEN),
                LR,
                &mut iteration,
                &mut j,
                BATCH_SIZE,
            );
            model.save(MODEL_LOC).unwrap();
            divider += 1;
            total_time += start_time.elapsed();
            println!("completed data {i}, in: {:?}", total_time / divider);
            i += 1;
        }
    }

}

pub fn sample() {
    let tokenizer = Tokenizer::new("charset.txt");
    let mut model = Sequential::load(MODEL_LOC).unwrap();

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