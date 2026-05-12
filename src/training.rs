use std::{
    rc::Rc,
    time::{Duration, Instant},
};

use crate::{
    batches::PreparedDataSet,
    config::{
        self, BATCH_SIZE, DATA_DIR, DATA_FILE, EPOCHS, LR, MAX_SEQ_LEN, MODEL_LOC, PRINT_EVERY,
        SAVE_EVERY, SEQ_LEN, SEQ_LOC, WARMUP_STEPS,
    },
    hierarchical::HierarchicalSequential,
    model::{build_hierarchical_model, build_normal_model},
    optimizers::Optimizer,
    sequential::Sequential,
    tokenizer::Tokenizer,
};

pub fn train_normal() {
    let tokenizer = Rc::new(Tokenizer::new(crate::config::CHARSET, false));
    let vocab = tokenizer.vocab_size();
    let word_boundary_ids = tokenizer.boundary_tokens();

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

    println!("Preparing dataset from '{DATA_DIR}' ...");
    let prep_start = Instant::now();
    let mut data =
        PreparedDataSet::from_single_file(&tokenizer, DATA_FILE, SEQ_LEN, &word_boundary_ids);
    println!(
        "  {} files → {} windows, {} tokens (prep took {:.1?})",
        data.num_sequences(),
        data.len(),
        data.total_tokens(),
        prep_start.elapsed(),
    );
    println!(
        "Training: {EPOCHS} epochs, LR={LR}, batch={BATCH_SIZE}, optimizer={:?}, log every {PRINT_EVERY} steps",
        Optimizer {}
    );

    let mut training_state = TrainingState::new();
    let mut total_time = Duration::ZERO;

    for epoch in 1..=EPOCHS {
        println!("── Epoch {epoch} ───────────────────────────────────────");

        // Shuffeln verändert nur die Reihenfolge der Windows — die tokenisierten
        // Sequenzen selbst bleiben unverändert im Speicher.
        data.shuffle();

        let start = Instant::now();
        model.train(data.iter(), &mut training_state);

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

pub fn train_hierarchical() {
    let tokenizer = Rc::new(Tokenizer::new(config::CHARSET, false));
    let vocab = tokenizer.vocab_size();
    let word_boundary_ids = tokenizer.boundary_tokens();

    let mut model = match HierarchicalSequential::load(MODEL_LOC) {
        Ok(m) => {
            println!("Loaded HM-RNN from '{MODEL_LOC}'.");
            m
        }
        Err(e) => {
            println!("Could not load '{MODEL_LOC}' ({e}) — creating new HM-RNN.");
            build_hierarchical_model(vocab, word_boundary_ids.clone())
        }
    };
    model.make_cache(MAX_SEQ_LEN);

    println!("Preparing dataset from '{DATA_DIR}' ...");
    let prep_start = Instant::now();
    let mut data =
        PreparedDataSet::from_single_file(&tokenizer, DATA_FILE, SEQ_LEN, &word_boundary_ids);
    println!(
        "  {} files → {} windows, {} tokens (prep took {:.1?})",
        data.num_sequences(),
        data.len(),
        data.total_tokens(),
        prep_start.elapsed(),
    );
    println!(
        "Training: {EPOCHS} epochs, LR={LR}, batch={BATCH_SIZE}, optimizer={:?}, log every {PRINT_EVERY} steps",
        Optimizer {}
    );

    let mut training_state = TrainingState::from_step(model.step);
    let mut total_time = Duration::ZERO;

    for epoch in 1..=EPOCHS {
        println!("── Epoch {epoch} ───────────────────────────────────────");

        data.shuffle();

        let start = Instant::now();
        model.train(data.iter(), &mut training_state);

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

pub struct TrainingState {
    pub step: usize,
    pub batch_size: usize,
    pub lr: f32,
    loss: f32,
    loss_steps: usize,
    pub print_interval: usize,
    pub save_interval: usize,
}

impl TrainingState {
    pub fn new() -> Self {
        Self::from_step(0)
    }

    pub fn from_step(step: usize) -> Self {
        Self {
            step,
            lr: LR,
            loss: 0.0,
            loss_steps: 0,
            batch_size: BATCH_SIZE,
            print_interval: PRINT_EVERY,
            save_interval: SAVE_EVERY,
        }
    }

    pub fn step(&mut self, loss: f32) -> Option<f32> {
        self.step += 1;
        self.loss += loss;
        self.loss_steps += 1;
        if self.step.is_multiple_of(self.batch_size) {
            let batch_num = self.step / self.batch_size;
            let warmup_scale = (batch_num as f32 / WARMUP_STEPS as f32).min(1.0);
            Some(self.lr * warmup_scale / self.batch_size as f32)
        } else {
            None
        }
    }

    pub fn print(&mut self) -> bool {
        self.step.is_multiple_of(self.print_interval)
    }

    pub fn get_loss(&mut self) -> f32 {
        let loss = self.loss / self.loss_steps as f32;
        self.loss = 0.0;
        self.loss_steps = 0;
        loss
    }

    pub fn save(&self) -> bool {
        self.step.is_multiple_of(self.save_interval)
    }
}
