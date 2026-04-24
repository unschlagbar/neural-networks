// ── training.rs ──────────────────────────────────────────────────────────────
//
// Training-Loops für beide Modellvarianten. Die Loops sind fast identisch
// aufgebaut — wer Duplikation reduzieren will, kann später eine gemeinsame
// Trainer-Abstraktion einführen, das ist aber nur lohnend wenn die beiden
// wirklich ähnlich bleiben. Im Moment unterscheiden sie sich nur in den
// boundary_ids (sentence_token_ids vs. boundary_token_ids) und dem Save-Pfad.

use std::{
    io::{Write, stdout},
    rc::Rc,
    time::{Duration, Instant},
};

use crate::{
    batches::{BatchDebugger, WordBoundaryBatches},
    config::{
        BATCH_SIZE, DATA_DIR, EPOCHS, LR, MAX_SEQ_LEN, MODEL_LOC, SAVE_EVERY, SEQ_LEN, SEQ_LOC,
    },
    data_set_loading::DataSet,
    hierarchical_sequential::HierarchicalSequential,
    model::{build_hierarchical_model, build_normal_model},
    sequential::Sequential,
    tokenizer::Tokenizer,
};

// ── normales Sequential ──────────────────────────────────────────────────────

pub fn train_normal() {
    let tokenizer = Rc::new(Tokenizer::new(crate::config::CHARSET, true));
    let vocab = tokenizer.vocab_size();
    let boundary_ids = tokenizer.sentence_token_ids();

    // Existierendes Modell laden, sonst neu bauen.
    let mut model = match Sequential::load(SEQ_LOC) {
        Ok(m) => {
            println!("Loaded model from '{SEQ_LOC}'.");
            m
        }
        Err(e) => {
            println!("Could not load '{SEQ_LOC}' ({e}) — creating a new model.");
            build_normal_model(vocab)
        }
    };

    model.make_cache(MAX_SEQ_LEN);

    let data_set = DataSet::load_from_dir(tokenizer.clone(), DATA_DIR);
    println!(
        "Dataset: {} files, {EPOCHS} epochs, LR={LR}, batch={BATCH_SIZE}",
        data_set.len()
    );

    let mut iteration = 0;
    let mut j = 0; // Anzahl Gradient-Updates (= iteration / BATCH_SIZE)
    let mut file_count = 0;
    let mut total_time = Duration::ZERO;

    for epoch in 1..=EPOCHS {
        println!("── Epoch {epoch} ──────────────────────────────────────────");

        for data in &data_set {
            let batches = WordBoundaryBatches::new(&data, &boundary_ids, SEQ_LEN);

            let start = Instant::now();
            // Alle 100 Files ein Sanity-Check mit BatchDebugger.
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
                    "  [{file_count} files | {j} updates | avg {:.0?}/file]",
                    total_time / file_count as u32
                );
                stdout().flush().unwrap();
            }

            if SAVE_EVERY > 0 && file_count % SAVE_EVERY == 0 {
                if let Err(e) = model.save(SEQ_LOC) {
                    eprintln!("  ✗ save failed: {e}");
                } else {
                    println!("  ✓ saved to '{SEQ_LOC}'");
                }
            }
        }

        println!();
        if let Err(e) = model.save(SEQ_LOC) {
            eprintln!("  ✗ end-of-epoch save failed: {e}");
        } else {
            println!("  ✓ end-of-epoch save to '{SEQ_LOC}'");
        }
    }
}

// ── HM-RNN (Chung, Ahn, Bengio 2016) ─────────────────────────────────────────

pub fn train_hierarchical() {
    let tokenizer = Rc::new(Tokenizer::new(crate::config::CHARSET, true));
    let vocab = tokenizer.vocab_size();
    let sentence_boundary_ids = tokenizer.sentence_token_ids();
    let word_boundary_ids = tokenizer.word_token_ids();

    let mut model = match HierarchicalSequential::load(MODEL_LOC) {
        Ok(m) => {
            println!("Loaded HM-RNN from '{MODEL_LOC}'.");
            m
        }
        Err(e) => {
            println!("Could not load '{MODEL_LOC}' ({e}) — creating new HM-RNN.");
            build_hierarchical_model(vocab, word_boundary_ids)
        }
    };

    model.make_cache(MAX_SEQ_LEN);

    let data_set = DataSet::load_from_dir(tokenizer.clone(), DATA_DIR);
    println!(
        "Dataset: {} files, {EPOCHS} epochs, LR={LR}, batch={BATCH_SIZE}",
        data_set.len()
    );

    let mut iteration = 0;
    let mut j = 0;
    let mut file_count = 0;
    let mut total_time = Duration::ZERO;

    for epoch in 1..=EPOCHS {
        println!("── Epoch {epoch} ──────────────────────────────────────────");

        for data in &data_set {
            let batches = WordBoundaryBatches::new(&data, &sentence_boundary_ids, SEQ_LEN);

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
                    "  [{file_count} files | {j} updates | avg {:.0?}/file]",
                    total_time / file_count as u32
                );
                stdout().flush().unwrap();
            }

            if SAVE_EVERY > 0 && file_count % SAVE_EVERY == 0 {
                match model.save(MODEL_LOC) {
                    Ok(()) => println!("  ✓ saved to '{MODEL_LOC}'"),
                    Err(e) => eprintln!("  ✗ save failed: {e}"),
                }
            }
        }

        println!();
        match model.save(MODEL_LOC) {
            Ok(()) => println!("  ✓ end-of-epoch save to '{MODEL_LOC}'"),
            Err(e) => eprintln!("  ✗ end-of-epoch save failed: {e}"),
        }
    }
}
