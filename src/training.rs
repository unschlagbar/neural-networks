use std::{
    f32::consts::PI,
    fs::File,
    io::{BufWriter, Write},
    rc::Rc,
    time::{Duration, Instant},
};

use crate::{
    batches::{PreparedDataSet, WordDataSet},
    config::{
        self, BATCH_SIZE, DATA_DIR, DATA_FILE, DECAY_STEPS, EPOCHS, LOG_EVERY, LR, MAX_SEQ_LEN,
        MIN_LR, MIN_WORDS_PER_SEQ, MODEL_LOC, SAVE_EVERY, SEQ_LEN, SEQ_LOC, WARMUP_STEPS,
        WORDS_PER_SEQ,
    },
    hierarchical::Hierarchical,
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
        "Training: {EPOCHS} epochs, LR={LR}, batch={BATCH_SIZE}, optimizer={:?}, log every {LOG_EVERY} steps",
        Optimizer {}
    );

    let mut training_state = TrainingState::new();
    training_state.init_log(SEQ_LOC, &[]);
    let mut total_time = Duration::ZERO;

    for epoch in 1..=EPOCHS {
        println!("── Epoch {epoch} ───────────────────────────────────────");

        // Shuffling only changes the order of windows — the tokenised
        // sequences themselves remain unchanged in memory.
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

    let mut model = match Hierarchical::load(MODEL_LOC, tokenizer.clone()) {
        Ok(m) => {
            println!("Loaded HM-RNN from '{MODEL_LOC}'.");
            m
        }
        Err(e) => {
            println!("Could not load '{MODEL_LOC}' ({e}) — creating new HM-RNN.");
            build_hierarchical_model(vocab, word_boundary_ids.clone(), tokenizer.clone())
        }
    };
    model.make_cache(WORDS_PER_SEQ);

    println!("Preparing dataset from '{DATA_FILE}' ...");
    let prep_start = Instant::now();
    let data = WordDataSet::from_single_file(
        &tokenizer,
        DATA_FILE,
        WORDS_PER_SEQ,
        MIN_WORDS_PER_SEQ,
        MAX_SEQ_LEN,
        &word_boundary_ids,
    );
    println!("  prep took {:.1?}", prep_start.elapsed());
    println!(
        "Training: {EPOCHS} epochs, LR={LR}, batch={BATCH_SIZE}, optimizer={:?}, log every {LOG_EVERY} steps",
        Optimizer {}
    );

    let mut training_state = TrainingState::from_step(model.step);
    training_state.init_log(MODEL_LOC, &["delta_word", "delta_char1", "word_loss"]);
    let mut total_time = Duration::ZERO;

    for epoch in 1..=EPOCHS {
        println!("── Epoch {epoch} ───────────────────────────────────────");

        //data.shuffle();

        let resume_at = if epoch == 1 {
            model.step % data.len()
        } else {
            0
        };
        if resume_at > 0 {
            println!(
                "  Resuming from window {resume_at} / {} (step {})",
                data.len(),
                model.step
            );
        }

        let start = Instant::now();
        model.train(data.iter().skip(resume_at), &mut training_state);

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
    current_lr: f32,
    warmup_lr: f32,
    decay_lr: f32,
    loss: f32,
    loss_steps: usize,
    pub print_interval: usize,
    pub save_interval: usize,
    log_writer: Option<BufWriter<File>>,
    last_log: Instant,
    steps_since_log: usize,
    tokens_since_log: usize,
    extra_cols: Vec<String>,
    extra_vals: Vec<(f32, usize)>,
}

impl TrainingState {
    pub fn new() -> Self {
        Self::from_step(0)
    }

    pub fn from_step(step: usize) -> Self {
        Self {
            step,
            lr: LR,
            current_lr: LR,
            warmup_lr: LR,
            decay_lr: LR,
            loss: 0.0,
            loss_steps: 0,
            batch_size: BATCH_SIZE,
            print_interval: LOG_EVERY,
            save_interval: SAVE_EVERY,
            log_writer: None,
            last_log: Instant::now(),
            steps_since_log: 0,
            tokens_since_log: 0,
            extra_cols: Vec::new(),
            extra_vals: Vec::new(),
        }
    }

    pub fn init_log(&mut self, model_path: &str, extra_cols: &[&str]) {
        std::fs::create_dir_all("logs").ok();
        let name = std::path::Path::new(model_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("model");
        let path = format!("logs/{name}.csv");
        self.extra_cols = extra_cols.iter().map(|s| s.to_string()).collect();
        self.extra_vals = vec![(0.0, 0); extra_cols.len()];
        let is_new = !std::path::Path::new(&path).exists()
            || std::fs::metadata(&path)
                .map(|m| m.len() == 0)
                .unwrap_or(true);
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            Ok(f) => {
                let mut writer = BufWriter::new(f);
                if is_new {
                    let extra_header: String = extra_cols.iter().map(|c| format!(",{c}")).collect();
                    let _ = writeln!(
                        writer,
                        "step,batch,lr,loss,perplexity,ms_per_step,s_per_norm_seq{extra_header}"
                    );
                }
                self.log_writer = Some(writer);
                println!("Logging to '{path}'");
            }
            Err(e) => eprintln!("Could not open log file '{path}': {e}"),
        }
    }

    pub fn log_tokens(&mut self, n: usize) {
        self.tokens_since_log += n;
    }

    pub fn log_metric(&mut self, name: &str, val: f32) {
        if let Some(i) = self.extra_cols.iter().position(|c| c == name) {
            self.extra_vals[i].0 += val;
            self.extra_vals[i].1 += 1;
        }
    }

    pub fn step(&mut self, loss: f32) -> Option<f32> {
        self.step += 1;
        self.loss += loss;
        self.loss_steps += 1;
        self.steps_since_log += 1;
        if self.step.is_multiple_of(self.batch_size) {
            let batch_num = self.step / self.batch_size;
            self.warmup_lr = self.lr * (batch_num as f32 / WARMUP_STEPS as f32).min(1.0);
            let t = (batch_num as f32 / DECAY_STEPS as f32).min(1.0);
            self.decay_lr = MIN_LR + 0.5 * (self.lr - MIN_LR) * (1.0 + (PI * t).cos());
            self.current_lr = self.warmup_lr.min(self.decay_lr);
            Some(self.current_lr)
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
        if let Some(writer) = &mut self.log_writer {
            let batch_num = self.step / self.batch_size;
            let elapsed_ms = self.last_log.elapsed().as_secs_f32() * 1000.0;
            let ms_per_step = elapsed_ms / self.steps_since_log.max(1) as f32;
            let s_per_norm = if self.tokens_since_log > 0 {
                (elapsed_ms / 1000.0 / self.tokens_since_log as f32) * SEQ_LEN as f32
            } else {
                0.0
            };
            let extra: String = self
                .extra_vals
                .iter()
                .map(|(sum, count)| {
                    format!(",{:.6}", if *count > 0 { sum / *count as f32 } else { 0.0 })
                })
                .collect();
            let _ = writeln!(
                writer,
                "{},{},{:e},{:.6},{:.4},{:.2},{:.4}{extra}",
                self.step,
                batch_num,
                self.current_lr,
                loss,
                loss.exp(),
                ms_per_step,
                s_per_norm,
            );
            let _ = writer.flush();
            for v in &mut self.extra_vals {
                *v = (0.0, 0);
            }
        }
        self.last_log = Instant::now();
        self.steps_since_log = 0;
        self.tokens_since_log = 0;
        loss
    }

    pub fn save(&self) -> bool {
        self.step.is_multiple_of(self.save_interval)
    }
}
