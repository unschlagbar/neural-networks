use std::fs;
use std::path::Path;
use std::rc::Rc;
use std::{
    io::{Write, stdin, stdout},
    time::{Duration, Instant},
};

pub mod activations;
pub mod batches;
pub mod data_set_loading;
pub mod dense;
pub mod dropout;
pub mod hierarchical_sequential;
pub mod indrnn;
pub mod loading;
pub mod lstm;
pub mod nn_layer;
pub mod parallel;
pub mod projection;
pub mod saving;
pub mod sequential;
pub mod softmax;
pub mod tokenizer;

#[allow(unused)]
use crate::activations::{Linear, Relu, Tanh};
use crate::batches::{BatchDebugger, WordBoundaryBatches};
use crate::data_set_loading::DataSet;
use crate::hierarchical_sequential::HierarchicalSequential;
use crate::nn_layer::SequentialBuilder;
use crate::sequential::Sequential;
use crate::tokenizer::Tokenizer;

// ── Config ────────────────────────────────────────────────────────────────────

const MODEL_LOC: &str = "models/hric5";
const SEQ_LOC: &str = "models/seq";
const SEQ_LEN: usize = 500;
const MAX_SEQ_LEN: usize = 2500;
const LR: f32 = 0.008;
const BATCH_SIZE: usize = 1;
const EPOCHS: usize = 1000;
/// Save after every N completed files (0 = never save mid-epoch, only at epoch end).
const SAVE_EVERY: usize = 2;

const MAX_LEN: usize = 1000;
const TEMPERATURE: f32 = 0.4;

const CHAR_HIDDEN: usize = 512;
const CONTEXT_DIM: usize = 512;

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    let mut input = String::new();
    stdin().read_line(&mut input).unwrap();
    if input.trim().is_empty() {
        if !Path::new("models/").exists() {
            fs::create_dir("models/").unwrap();
        }
        //thread::spawn(|| train());
        train_new();
    } else {
        sample();
    }
}

// ── Training ──────────────────────────────────────────────────────────────────

fn build_new_model(vocab: usize, boundary_ids: Vec<u16>) -> HierarchicalSequential {
    let char_model = SequentialBuilder::new(vocab + CONTEXT_DIM)
        .lstm(CHAR_HIDDEN)
        .dropout(0.3)
        .parallel(|b| {
            b.lstm(CHAR_HIDDEN / 4 * 3)
                .indrnn(CHAR_HIDDEN / 4 * 1, Relu)
        })
        .dropout(0.3)
        .dense(vocab, Linear)
        .softmax()
        .build();

    let high_model = SequentialBuilder::new(CHAR_HIDDEN)
        .parallel(|b| {
            b.lstm(CONTEXT_DIM / 4 * 3)
                .indrnn(CONTEXT_DIM / 4 * 1, Relu)
        })
        .dropout(0.3)
        .parallel(|b| {
            b.lstm(CONTEXT_DIM / 4 * 3)
                .indrnn(CONTEXT_DIM / 4 * 1, Relu)
        })
        .dropout(0.3)
        .parallel(|b| {
            b.lstm(CONTEXT_DIM / 4 * 3)
                .indrnn(CONTEXT_DIM / 4 * 1, Relu)
        })
        .dropout(0.3)
        .build();

    HierarchicalSequential::new(char_model, high_model, vocab, boundary_ids)
}

fn build_new_normal_model(vocab: usize) -> Sequential {
    SequentialBuilder::new(vocab)
        .parallel(|b| b.lstm(CONTEXT_DIM / 2).indrnn(CONTEXT_DIM / 2, Relu))
        .dropout(0.3)
        .parallel(|b| b.lstm(CONTEXT_DIM / 2).indrnn(CONTEXT_DIM / 2, Relu))
        .dropout(0.3)
        .dense(vocab, Linear)
        .softmax()
        .build()
}

pub fn train_new() {
    let tokenizer = Rc::new(Tokenizer::new("charset.txt", true));
    let vocab = tokenizer.vocab_size();
    let sentence_boundary_ids = tokenizer.sentence_token_ids();
    let boundary_ids = tokenizer.boundary_token_ids();

    // ── Load existing model or build a fresh one ──────────────────────────────
    let mut model = match HierarchicalSequential::load(MODEL_LOC) {
        Ok(m) => {
            println!("Loaded model from '{MODEL_LOC}'.");
            m
        }
        Err(e) => {
            println!("Could not load '{MODEL_LOC}' ({e}) — creating a new model.");
            build_new_model(vocab, boundary_ids.clone())
        }
    };

    model.make_cache(MAX_SEQ_LEN);

    let mut iteration = 0;
    let mut j = 0; // gradient updates
    let mut file_count = 0; // files processed this run
    let mut total_time = Duration::ZERO;

    let data_set = DataSet::load_from_dir(tokenizer.clone(), "political_speeches/");
    println!("Dataset: {} files, {EPOCHS} epochs", data_set.len());

    for epoch in 1..=EPOCHS {
        println!("── Epoch {epoch} ──────────────────────────────────────────");

        for data in &data_set {
            let batches = WordBoundaryBatches::new(&data, &sentence_boundary_ids, SEQ_LEN);
            //let batches = Batches::new(&data, SEQ_LEN);

            // Wrap with BatchDebugger every 100 files for a quick sanity check.
            let start = Instant::now();
            if file_count % 100 == 0 {
                let mut dbg = BatchDebugger::new(batches, format!("file {file_count}"));
                model.train(&mut dbg, LR, &mut iteration, &mut j, BATCH_SIZE);
                dbg.print_summary();
            } else {
                model.train(batches, LR, &mut iteration, &mut j, BATCH_SIZE);
            }
            total_time += start.elapsed();
            file_count += 1;

            if file_count % 10 == 0 {
                println!(
                    "  [{file_count} files | {j} updates | avg {:.0?}/file]  \r",
                    total_time / file_count as u32
                );
                stdout().flush().unwrap();
            }

            // Periodic mid-epoch save.
            if SAVE_EVERY > 0 && file_count % SAVE_EVERY == 0 {
                save_model(&model);
            }
        }

        // End-of-epoch save.
        println!();
        save_model(&model);
    }
}

pub fn train() {
    let tokenizer = Rc::new(Tokenizer::new("charset.txt", true));
    let vocab = tokenizer.vocab_size();
    let boundary_ids = tokenizer.sentence_token_ids();

    // ── Load existing model or build a fresh one ──────────────────────────────
    let mut model = match Sequential::load(SEQ_LOC) {
        Ok(m) => {
            println!("Loaded model from '{SEQ_LOC}'.");
            m
        }
        Err(e) => {
            println!("Could not load '{SEQ_LOC}' ({e}) — creating a new model.");
            build_new_normal_model(vocab)
        }
    };

    model.make_cache(MAX_SEQ_LEN);

    let mut iteration = 0;
    let mut j = 0; // gradient updates
    let mut file_count = 0; // files processed this run
    let mut total_time = Duration::ZERO;

    let data_set = DataSet::load_from_dir(tokenizer.clone(), "political_speeches/");
    println!("Dataset: {} files, {EPOCHS} epochs", data_set.len());

    for epoch in 1..=EPOCHS {
        println!("── Epoch {epoch} ──────────────────────────────────────────");

        for data in &data_set {
            let batches = WordBoundaryBatches::new(&data, &boundary_ids, SEQ_LEN);
            //let batches = Batches::new(&data, SEQ_LEN);

            // Wrap with BatchDebugger every 100 files for a quick sanity check.
            let start = Instant::now();
            if file_count % 100 == 0 {
                let mut dbg = BatchDebugger::new(batches, format!("file {file_count}"));
                model.train(&mut dbg, LR, &mut iteration, &mut j, BATCH_SIZE);
                dbg.print_summary();
            } else {
                model.train(batches, LR, &mut iteration, &mut j, BATCH_SIZE);
            }
            total_time += start.elapsed();
            file_count += 1;

            if file_count % 10 == 0 {
                println!(
                    "  [{file_count} files | {j} updates | avg {:.0?}/file]  \r",
                    total_time / file_count as u32
                );
                stdout().flush().unwrap();
            }

            // Periodic mid-epoch save.
            if SAVE_EVERY > 0 && file_count % SAVE_EVERY == 0 {
                model.save(SEQ_LOC).unwrap();
            }
        }

        // End-of-epoch save.
        println!();
        model.save(SEQ_LOC).unwrap();
    }
}

fn save_model(model: &HierarchicalSequential) {
    match model.save(MODEL_LOC) {
        Ok(()) => println!("  ✓ saved to '{MODEL_LOC}'"),
        Err(e) => eprintln!("  ✗ save failed: {e}"),
    }
}

// ── Sampling ──────────────────────────────────────────────────────────────────

pub fn sample() {
    let tokenizer = Tokenizer::new("charset.txt", false);

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

    // cache[0] is enough for one-step sampling.
    model.make_cache(1);

    loop {
        println!("\nSample mode — type a prefix (empty = random start):");
        let mut input = String::new();
        stdin().read_line(&mut input).unwrap();

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

pub fn sample_old() {
    let tokenizer = Tokenizer::new("charset.txt", false);

    let mut model = match Sequential::load(SEQ_LOC) {
        Ok(m) => {
            println!("Loaded lstm model from '{MODEL_LOC}'.");
            m
        }
        Err(e) => {
            eprintln!("Failed to load '{MODEL_LOC}': {e}");
            std::process::exit(1);
        }
    };

    // cache[0] is enough for one-step sampling.
    model.make_cache(1);

    loop {
        println!("\nSample mode — type a prefix (empty = random start):");
        let mut input = String::new();
        stdin().read_line(&mut input).unwrap();

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
