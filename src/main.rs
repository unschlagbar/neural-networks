use std::rc::Rc;
use std::{
    io::{Write, stdin, stdout},
    time::{Duration, Instant},
};

use crate::batches::Batches;
use crate::data_set_loading::DataSet;
use crate::layer::Activation;
use crate::sequential::LayerBuilder::*;
use crate::sequential::Sequential;
use crate::tokenizer::Tokenizer;

pub mod batches;
pub mod data_set_loading;
pub mod jarvis;
pub mod layer;
pub mod lstm;
pub mod prepare_set;
pub mod saving;
pub mod sequential;
pub mod tokenizer;

const MODEL_LOC: &str = "test";
const SEQ_LEN: usize = 100;
const LR: f32 = 0.00005;
const BATCH_SIZE: usize = 2;
const EPOCHS: usize = 1;

const MAX_LEN: usize = 2000;
const TEMPERATURE: f32 = 0.1;

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
            Dense(384, activation),
            LSTM(384),
            LSTM(384),
            Dense(vocab, Activation::Softmax),
        ];

        println!("init model");

        Sequential::new(layout, vocab)
    };

    model.make_cache(SEQ_LEN);

    let mut iteration = 0;
    let mut j = 0;
    let mut i = 1;

    let mut total_time = Duration::from_secs(0);

    let data_set = DataSet::load_from_dir(tokenizer.clone(), "political_speeches/").into_iter();

    for _ in 0..EPOCHS {
        for data in data_set.clone() {
            let data = Batches::new(&data, SEQ_LEN);
            let start_time = Instant::now();
            model.train(data.into_iter(), LR, &mut iteration, &mut j, BATCH_SIZE);
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
        println!("Sample mode. Type the seq start!");
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
