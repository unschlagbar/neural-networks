// ── sampling.rs ──────────────────────────────────────────────────────────────
//
// Beide Sampling-Modi: normaler Sequential sowie Hierarchical. Die Logik ist
// in beiden Fällen identisch — Modell laden, Cache für 1 Token allokieren,
// dann in einer REPL-artigen Schleife Prefix einlesen und streamend Tokens
// ausgeben.
//
// Ich habe hier NICHT versucht mit einer Closure/Trait zu abstrahieren, weil
// `Sequential::sample` und `HierarchicalSequential::sample` zwar syntaktisch
// identisch aussehen, aber keine gemeinsame Trait-Schnittstelle haben. Eine
// Generic-Abstraktion bringt hier mehr Lifetime-Komplexität als sie an
// Zeilen spart.

use std::io::{Write, stdin, stdout};

use crate::{
    config::{CHARSET, MAX_LEN, MODEL_LOC, SEQ_LOC, TEMPERATURE},
    hierarchical_sequential::HierarchicalSequential,
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

    // Für Einzelschritt-Sampling reicht cache[0].
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

pub fn sample_hierarchical() {
    let tokenizer = Tokenizer::new(CHARSET, false);

    let mut model = match HierarchicalSequential::load(MODEL_LOC) {
        Ok(m) => {
            println!("Loaded hierarchical model from '{MODEL_LOC}'.");
            m
        }
        Err(e) => {
            eprintln!("Failed to load '{MODEL_LOC}': {e}");
            std::process::exit(1);
        }
    };

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
