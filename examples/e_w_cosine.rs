//! Probes the encoder's word-embedding geometry: encodes a set of words to
//! their `e_w` vectors and prints the pairwise cosine-similarity matrix.
//! Tests whether orthographic neighbors (pumpkin/puzzle/puddle/puppy/purple)
//! collapse together independent of `OUT_HIDDEN` width.
//!
//!   cargo run --release --example e_w_cosine -- fable15 fable15_small

use std::path::Path;
use std::rc::Rc;

use neural_networks::{
    config::{CHARSET, MAX_SEQ_LEN},
    hierarchical::Hierarchical,
    tokenizer::Tokenizer,
};

// `pu-` cluster first, then unrelated controls.
const WORDS: &[&str] = &[
    "pumpkin", "puzzle", "puddle", "puppy", "purple", // orthographic neighbors
    "banana", "table", "friend", "apple", // controls
];

fn resolve(name: &str) -> String {
    if Path::new(name).is_file() {
        name.to_string()
    } else {
        format!("models/{name}")
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-12)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: e_w_cosine <model-name>...");
        std::process::exit(2);
    }

    let tokenizer = Rc::new(Tokenizer::new(CHARSET, false));

    for name in &args {
        let path = resolve(name);
        let mut model = Hierarchical::load(&path, tokenizer.clone()).unwrap();
        model.make_cache(1, MAX_SEQ_LEN);

        // Encode each word to its e_w (clone out of the borrowed cache).
        let embeddings: Vec<Vec<f32>> = WORDS
            .iter()
            .map(|w| {
                let toks = tokenizer.to_tokens(w);
                model.encoder.encode_word_sample(&toks).to_vec()
            })
            .collect();

        println!("\n================ {name} (dim {}) ================", embeddings[0].len());
        // Header row.
        print!("{:>9}", "");
        for w in WORDS {
            print!("{:>9}", w);
        }
        println!();
        // Cosine matrix.
        for (i, wi) in WORDS.iter().enumerate() {
            print!("{:>9}", wi);
            for j in 0..WORDS.len() {
                print!("{:>9.3}", cosine(&embeddings[i], &embeddings[j]));
            }
            println!();
        }
    }
}
