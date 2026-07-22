use std::io::{Write, stdin, stdout};

use crate::{
    config::{MAX_LEN, MAX_SEQ_LEN, TEMPERATURE, TOP_P},
    hierarchical::Hierarchical,
    sequential::Sequential,
    tokenizer_utf8::{BYTE_TOKENS, Utf8Tokenizer},
};

/// Streams sampled byte tokens to stdout. A UTF-8 character spans several byte
/// tokens, so bytes are held back until they form a complete character.
#[derive(Default)]
struct Utf8Printer {
    pending: Vec<u8>,
}

impl Utf8Printer {
    /// Print `token` once its character is complete. Returns false on `<END>`.
    fn print(&mut self, token: u16, tokenizer: &Utf8Tokenizer) -> bool {
        if token == tokenizer.end_token() {
            return false;
        }
        if (token as usize) >= BYTE_TOKENS {
            return true; // model-internal marker — never displayed
        }

        self.pending.push(token as u8);
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                print!("{s}");
                stdout().flush().unwrap();
                self.pending.clear();
            }
            // Incomplete character: keep collecting. Anything else is a byte the
            // model made up that can never complete — drop it.
            Err(e) if e.error_len().is_none() => {}
            Err(_) => self.pending.clear(),
        }
        true
    }
}

pub fn sample_normal(model_path: &str) {
    let tokenizer = Utf8Tokenizer::new();

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

        let mut printer = Utf8Printer::default();
        model.sample(&prefix, MAX_LEN, TEMPERATURE, TOP_P, |token| {
            printer.print(token, &tokenizer)
        });

        println!();
    }
}

pub fn sample_hierarchical(model_path: &str) {
    let tokenizer = Utf8Tokenizer::new();

    let mut model = Hierarchical::load(model_path, tokenizer).unwrap();

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

        let mut printer = Utf8Printer::default();
        model.sample(&prefix, MAX_LEN, TEMPERATURE, TOP_P, |token| {
            printer.print(token, &tokenizer)
        });

        println!();
    }
}
