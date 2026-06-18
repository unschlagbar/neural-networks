use std::{
    io::{Write, stdin, stdout},
    rc::Rc,
};

use crate::{
    config::{CHARSET, MAX_LEN, MAX_SEQ_LEN, TEMPERATURE, TOP_P},
    hierarchical::Hierarchical,
    sequential::Sequential,
    tokenizer::Tokenizer,
};

pub fn sample_normal(model_path: &str) {
    let tokenizer = Tokenizer::new(CHARSET, false);

    let mut model = match Sequential::load(model_path) {
        Ok(m) => {
            println!("Loaded sequential model from '{model_path}'.");
            m
        }
        Err(e) => {
            eprintln!("Failed to load '{model_path}': {e}");
            std::process::exit(1);
        }
    };

    // For single-step sampling, cache[0] is sufficient.
    model.make_cache(1);

    loop {
        println!("\nSample mode — type a prefix (empty = random start, Ctrl+D = quit):");
        let mut input = String::new();
        if stdin().read_line(&mut input).unwrap() == 0 {
            println!();
            return;
        }

        let prefix: Vec<u16> = if !input.trim().is_empty() {
            tokenizer.to_tokens(input.trim())
        } else {
            Vec::new()
        };

        print!(">>> ");
        stdout().flush().unwrap();

        model.sample(&prefix, MAX_LEN, TEMPERATURE, TOP_P, |token| {
            let s = tokenizer.get_char(token);
            if s == "<END>" {
                false
            } else {
                print!("{s}");
                stdout().flush().unwrap();
                true
            }
        });

        println!();
    }
}

pub fn sample_hierarchical(model_path: &str) {
    let tokenizer = Tokenizer::new(CHARSET, false);

    let mut model = Hierarchical::load(model_path, Rc::new(tokenizer.clone())).unwrap();

    model.make_cache(1, MAX_SEQ_LEN);

    loop {
        println!("\nSample mode — type a prefix (empty = random start, Ctrl+D = quit):");
        let mut input = String::new();
        if stdin().read_line(&mut input).unwrap() == 0 {
            println!();
            return;
        }

        let prefix: Vec<u16> = if !input.trim().is_empty() {
            tokenizer.to_tokens(input.trim())
        } else {
            Vec::new()
        };

        print!(">>> ");
        stdout().flush().unwrap();

        model.sample(&prefix, MAX_LEN, TEMPERATURE, |token| {
            let s = tokenizer.get_char(token);
            if s == "<END>" {
                false
            } else {
                print!("{s}");
                stdout().flush().unwrap();
                true
            }
        });

        println!();
    }
}
