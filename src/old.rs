use std::{
    io::{stdin, stdout, Write}, rc::Rc, time::{Duration, Instant}
};

use crate::{batches::Batches, data_set_loading::DataSet, lstm::LSTM, tokenizer::Tokenizer};

pub const MODEL_LOC: &str = "rust_rnn_old";

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

pub fn run_old() {
    let tokenizer = Rc::new(Tokenizer::new("charset.txt"));

    let mut model = if let Ok(model) = LSTM::load(MODEL_LOC) {
        model
    } else {
        let model = LSTM::new(&[tokenizer.vocab_size(), 384, 384], tokenizer.vocab_size());
        model.save(MODEL_LOC).unwrap();
        model
    };

    //test(&mut model, &tokenizer);

    let mut iteration = 1;
    let mut j = 1;

    let mut total_time = Duration::from_secs(0);
    let mut divider = 0;

    for _ in 0..400 {
        for (i, data) in DataSet::load_file(tokenizer.clone(), "alice.txt")
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
