use std::io::{Write, stdin, stdout};

use crate::{
    config::{CHARSET, MAX_LEN, MODEL_LOC, SEQ_LOC, TEMPERATURE, TOP_P},
    hierarchical::HierarchicalSequential,
    sequential::Sequential,
    tokenizer::Tokenizer,
};

pub fn sample_normal() {
    let tokenizer = Tokenizer::new(CHARSET, false);

    let mut model = match Sequential::load(SEQ_LOC) {
        Ok(m) => {
            println!("Loaded sequential model from '{SEQ_LOC}'.");
            m
        }
        Err(e) => {
            eprintln!("Failed to load '{SEQ_LOC}': {e}");
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

pub fn sample_hierarchical() {
    let tokenizer = Tokenizer::new(CHARSET, false);

    let mut model = HierarchicalSequential::load(MODEL_LOC).unwrap();

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
