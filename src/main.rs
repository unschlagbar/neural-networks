use crate::batches::Batches;
use crate::data_set_loading::DataSet;
use crate::lstm::LSTM;
use crate::tokenizer::Tokenizer;

pub mod batches;
pub mod data_set_loading;
pub mod jarvis;
pub mod layer;
pub mod lstm;
pub mod mlp;
pub mod tokenizer;

use std::io::{Write, stdin, stdout};

pub const MODEL_LOC: &str = "rust_rnn";

fn main() {
    let tokenizer = Tokenizer::new("charset.txt");

    // 5. Model initialisieren
    let mut model = if let Ok(model) = LSTM::load(MODEL_LOC) {
        model
    } else {
        let model = LSTM::new(&[tokenizer.vocab_size(), 512, 512], tokenizer.vocab_size());
        model.save(MODEL_LOC).unwrap();
        model
    };

    //test(&mut model, &tokenizer);

    let mut iteration = 1;
    let mut j = 1;

    for _ in 0..1000 {
        for (i, data) in DataSet::load_rust_files(&tokenizer).into_iter().enumerate() {
            model.train(
                Batches::new(&data, &[], 100..200),
                0.0001,
                &mut iteration,
                &mut j,
                5,
            );
    
            println!("completed data {i}")
        }
    }
}

pub fn test(model: &mut LSTM, tokenizer: &Tokenizer) {
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