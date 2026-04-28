// ── training.rs ──────────────────────────────────────────────────────────────
//
// Training-Loops für beide Modellvarianten. Der Datensatz wird *einmal* am
// Anfang vollständig geladen, tokenisiert und in Word-Boundary-Windows zerlegt
// (siehe PreparedDataSet). Pro Epoche shuffeln wir nur die Window-Liste neu —
// kein Re-Read, keine Re-Tokenisierung. Loss wird alle PRINT_EVERY Steps
// gemeldet, nicht erst am Ende einer Datei.

use std::{
    rc::Rc,
    time::{Duration, Instant},
};

use crate::{
    batches::PreparedDataSet,
    config::{
        BATCH_SIZE, DATA_DIR, EPOCHS, LR, MAX_SEQ_LEN, MODEL_LOC, PRINT_EVERY, SEQ_LEN, SEQ_LOC,
    },
    hierarchical_sequential::HierarchicalSequential,
    model::{build_hierarchical_model, build_normal_model},
    sequential::Sequential,
    tokenizer::Tokenizer,
};

// ── normales Sequential ──────────────────────────────────────────────────────

pub fn train_normal() {
    let tokenizer = Rc::new(Tokenizer::new(crate::config::CHARSET, false));
    let vocab = tokenizer.vocab_size();
    let word_boundary_ids = tokenizer.word_token_ids();

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

    // ── Dataset einmal vorbereiten ────────────────────────────────────────────
    println!("Preparing dataset from '{DATA_DIR}' ...");
    let prep_start = Instant::now();
    let mut data = PreparedDataSet::from_dir(&tokenizer, DATA_DIR, SEQ_LEN, &word_boundary_ids);
    println!(
        "  {} files → {} windows, {} tokens (prep took {:.1?})",
        data.num_sequences(),
        data.len(),
        data.total_tokens(),
        prep_start.elapsed(),
    );
    println!(
        "Training: {EPOCHS} epochs, LR={LR}, batch={BATCH_SIZE}, log every {PRINT_EVERY} steps"
    );

    let mut iteration = 0;
    let mut j = 0;
    let mut step = 0;
    let mut total_time = Duration::ZERO;

    for epoch in 1..=EPOCHS {
        println!("── Epoch {epoch} ───────────────────────────────────────");

        // Shuffeln verändert nur die Reihenfolge der Windows — die tokenisierten
        // Sequenzen selbst bleiben unverändert im Speicher.
        data.shuffle();

        let start = Instant::now();
        model.train(
            data.iter(),
            LR,
            &mut iteration,
            &mut j,
            BATCH_SIZE,
            PRINT_EVERY,
            &mut step,
        );
        let epoch_time = start.elapsed();
        total_time += epoch_time;

        match model.save(SEQ_LOC) {
            Ok(()) => println!(
                "  ✓ end-of-epoch save to '{SEQ_LOC}'  (epoch {epoch_time:.0?}, total {total_time:.0?})"
            ),
            Err(e) => eprintln!("  ✗ end-of-epoch save failed: {e}"),
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

    println!("Preparing dataset from '{DATA_DIR}' ...");
    let prep_start = Instant::now();
    let mut data = PreparedDataSet::from_dir(&tokenizer, DATA_DIR, SEQ_LEN, &sentence_boundary_ids);
    println!(
        "  {} files → {} windows, {} tokens (prep took {:.1?})",
        data.num_sequences(),
        data.len(),
        data.total_tokens(),
        prep_start.elapsed(),
    );
    println!(
        "Training: {EPOCHS} epochs, LR={LR}, batch={BATCH_SIZE}, log every {PRINT_EVERY} steps"
    );

    let mut iteration = 0;
    let mut j = 0;
    let mut step = 0;
    let mut total_time = Duration::ZERO;

    for epoch in 1..=EPOCHS {
        println!("── Epoch {epoch} ───────────────────────────────────────");

        data.shuffle();

        let start = Instant::now();
        model.train(
            data.iter(),
            LR,
            &mut iteration,
            &mut j,
            BATCH_SIZE,
            PRINT_EVERY,
            &mut step,
        );
        let epoch_time = start.elapsed();
        total_time += epoch_time;

        match model.save(MODEL_LOC) {
            Ok(()) => println!(
                "  ✓ end-of-epoch save to '{MODEL_LOC}'  (epoch {epoch_time:.0?}, total {total_time:.0?})"
            ),
            Err(e) => eprintln!("  ✗ end-of-epoch save failed: {e}"),
        }
    }
}
