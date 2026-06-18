use std::{
    f32::consts::PI,
    fs::File,
    io::{BufWriter, Write},
    rc::Rc,
    time::{Duration, Instant},
};

use crate::{
    batches::WordDataSet,
    config::{
        self, BATCH_SIZE, DATA_FILE, DECAY_STEPS, EPOCHS, LOG_EVERY, LR, MAX_WINDOW_TOKENS, MIN_LR,
        MIN_WORDS_PER_SEQ, SAVE_EVERY, SEQ_LEN, WARMUP_STEPS, WORDS_PER_SEQ,
    },
    hierarchical::{BackboneMode, Hierarchical},
    model::{build_hierarchical_model, build_normal_model},
    optimizers::Optimizer,
    sequential::Sequential,
    tokenizer::Tokenizer,
};

pub fn train_normal(model_path: &str) {
    let tokenizer = Rc::new(Tokenizer::new(crate::config::CHARSET, false));
    let vocab = tokenizer.vocab_size();
    let word_boundary_ids = tokenizer.boundary_tokens();

    // Existierendes Modell laden, sonst neu bauen.
    let mut model = match Sequential::load(model_path) {
        Ok(m) => {
            println!("Loaded model from '{model_path}'.");
            m
        }
        Err(e) => {
            println!("Could not load '{model_path}' ({e}) — creating a new model.");
            build_normal_model(vocab)
        }
    };
    // Same X-word windows the hierarchical model trains on (identical params),
    // so the two systems can be compared on exactly the same samples.
    println!("Preparing dataset from '{DATA_FILE}' ...");
    let prep_start = Instant::now();
    let data = WordDataSet::from_single_file(
        &tokenizer,
        DATA_FILE,
        WORDS_PER_SEQ,
        MIN_WORDS_PER_SEQ,
        MAX_WINDOW_TOKENS,
        &word_boundary_ids,
    );
    println!(
        "  {} windows (prep took {:.1?})",
        data.len(),
        prep_start.elapsed(),
    );
    // Size the cache to the longest window that actually occurs.
    model.make_cache(data.max_window_tokens());
    println!(
        "Training: {EPOCHS} epochs, LR={LR}, batch={BATCH_SIZE}, optimizer={:?}, log every {LOG_EVERY} steps",
        Optimizer {}
    );

    let mut training_state = TrainingState::new();
    training_state.init_log(model_path, &["word_loss"]);
    let mut total_time = Duration::ZERO;

    for epoch in 1..=EPOCHS {
        println!("── Epoch {epoch} ───────────────────────────────────────");

        // Hierarchical training keeps corpus order (no shuffle) so both systems
        // see the same window order — leave it unshuffled here too for parity.

        let start = Instant::now();
        model.train_words(data.iter(), &mut training_state);

        let epoch_time = start.elapsed();
        total_time += epoch_time;

        match model.save(model_path) {
            Ok(()) => println!(
                "  ✓ end-of-epoch save to '{model_path}'  (epoch {epoch_time:.0?}, total {total_time:.0?})"
            ),
            Err(e) => eprintln!("  ✗ end-of-epoch save failed: {e}"),
        }
    }
}

pub fn train_hierarchical(model_path: &str) {
    let tokenizer = Rc::new(Tokenizer::new(config::CHARSET, false));
    let vocab = tokenizer.vocab_size();
    let word_boundary_ids = tokenizer.boundary_tokens();

    let mut model = match Hierarchical::load(model_path, tokenizer.clone()) {
        Ok(m) => {
            println!("Loaded HM-RNN from '{model_path}'.");
            m
        }
        Err(e) => {
            println!("Could not load '{model_path}' ({e}) — creating new HM-RNN.");
            build_hierarchical_model(vocab, word_boundary_ids.clone(), tokenizer.clone())
        }
    };
    println!("Preparing dataset from '{DATA_FILE}' ...");
    let prep_start = Instant::now();
    let data = WordDataSet::from_single_file(
        &tokenizer,
        DATA_FILE,
        WORDS_PER_SEQ,
        MIN_WORDS_PER_SEQ,
        MAX_WINDOW_TOKENS,
        &word_boundary_ids,
    );
    println!("  prep took {:.1?}", prep_start.elapsed());
    model.make_cache(WORDS_PER_SEQ, data.max_window_tokens());
    println!(
        "Training: {EPOCHS} epochs, LR={LR}, batch={BATCH_SIZE}, optimizer={:?}, log every {LOG_EVERY} steps",
        Optimizer {}
    );

    let mut training_state = TrainingState::from_step(model.step);
    training_state.init_log(model_path, &["delta_word", "delta_char1", "word_loss"]);
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

        match model.save(model_path) {
            Ok(()) => println!(
                "  ✓ end-of-epoch save to '{model_path}'  (epoch {epoch_time:.0?}, total {total_time:.0?})"
            ),
            Err(e) => eprintln!("  ✗ end-of-epoch save failed: {e}"),
        }
    }
}

/// Debug: load the hierarchical model and print encoder/decoder inputs and
/// targets (forward) plus decoder targets (backward) for the first few words of
/// one window. No training. Set via the `ht` mode.
pub fn trace_hierarchical(model_path: &str) {
    const TRACE_WORDS: usize = 12;

    let tokenizer = Rc::new(Tokenizer::new(config::CHARSET, false));
    let word_boundary_ids = tokenizer.boundary_tokens();

    let mut model = match Hierarchical::load(model_path, tokenizer.clone()) {
        Ok(m) => m,
        Err(_) => build_hierarchical_model(
            tokenizer.vocab_size(),
            word_boundary_ids.clone(),
            tokenizer.clone(),
        ),
    };
    let data = WordDataSet::from_single_file(
        &tokenizer,
        DATA_FILE,
        WORDS_PER_SEQ,
        MIN_WORDS_PER_SEQ,
        MAX_WINDOW_TOKENS,
        &word_boundary_ids,
    );
    model.make_cache(WORDS_PER_SEQ, data.max_window_tokens());
    model.trace_io = true;

    let Some(batch) = data.iter().next() else {
        eprintln!("no windows in dataset");
        return;
    };
    // Trim to the first few words so the trace stays short.
    let words: Vec<_> = batch.words.iter().take(TRACE_WORDS + 1).cloned().collect();
    let end = words.last().map(|r| r.end).unwrap_or(0);

    println!("── forward ─────────────────────────────────────────────");
    model.reset();
    model.forward_over(&batch.tokens[..end], &words);
    println!("\n── backward ────────────────────────────────────────────");
    model.backwards_sequence();
}

/// Inference-time probe: how much does cross-word context actually help the
/// trained hierarchical model? Evaluates decode cross-entropy under three
/// ablations of the backbone context path. No training, no weight changes.
pub fn probe_hierarchical(model_path: &str) {
    // How many windows to evaluate (each is up to WORDS_PER_SEQ words). Keep it
    // modest — this is forward-only but still O(tokens) per mode.
    const PROBE_WINDOWS: usize = 10;

    let tokenizer = Rc::new(Tokenizer::new(config::CHARSET, false));
    let word_boundary_ids = tokenizer.boundary_tokens();

    let mut model = match Hierarchical::load(model_path, tokenizer.clone()) {
        Ok(m) => {
            println!("Loaded '{model_path}' (step {}).", m.step);
            m
        }
        Err(e) => {
            eprintln!("Could not load '{model_path}': {e}");
            std::process::exit(2);
        }
    };
    println!("Preparing dataset from '{DATA_FILE}' ...");
    let data = WordDataSet::from_single_file(
        &tokenizer,
        DATA_FILE,
        WORDS_PER_SEQ,
        MIN_WORDS_PER_SEQ,
        MAX_WINDOW_TOKENS,
        &word_boundary_ids,
    );
    model.make_cache(WORDS_PER_SEQ, data.max_window_tokens());
    let n = PROBE_WINDOWS.min(data.len());
    println!("Probing on {n} windows.\n");

    let modes = [
        (BackboneMode::Normal, "Normal (full history)"),
        (
            BackboneMode::ResetEachWord,
            "ResetEachWord (prev word only)",
        ),
    ];

    let mut results = Vec::new();
    for (mode, label) in modes {
        model.backbone_mode = mode;
        let ce = model.eval_decode_loss(data.iter().take(n));
        println!("  {label:<34} CE = {ce:.4} nats   ppl = {:.3}", ce.exp());
        results.push((label, ce));
    }
    model.backbone_mode = BackboneMode::Normal;

    let normal = results[0].1;
    let reset = results[1].1;
    println!("\n  ── context usage ──");
    println!(
        "  long-range history (Reset → Normal): {:.4} nats  <- what the deep backbone actually buys",
        reset - normal
    );

    // ── Forget-gate / memory-horizon probe ──────────────────────────────
    // The decisive "can it remember?" measurement. f_prime ∈ (0,1] is the
    // fraction of cell state each backbone mLSTM head carries from word to
    // word. survival(k) = f̄^k, so the 1/e memory horizon is -1/ln(f̄) words.
    // Low horizon → the state decays (capacity it can't use); high horizon →
    // it CAN remember and simply doesn't, i.e. nothing to fix in the cell.
    let mut window = data.iter();
    if let Some(crate::batches::WordBatch { tokens, words }) = window.next() {
        model.reset();
        model.forward_over(tokens, &words);
        let per_head = model.backbone_forget_per_head();

        println!(
            "\n  ── backbone forget gates (over {} words, {} blocks) ──",
            words.len().saturating_sub(1),
            per_head.len(),
        );
        let mut all: Vec<f32> = Vec::new();
        for (b, heads) in per_head.iter().enumerate() {
            let mean = heads.iter().sum::<f32>() / heads.len().max(1) as f32;
            let max = heads.iter().fold(0.0_f32, |x, y| x.max(*y));
            println!(
                "  block {b:>2}: mean f̄ = {mean:.4} (~{:.1} words)   best head f̄ = {max:.4} (~{:.1} words)",
                horizon(mean),
                horizon(max),
            );
            all.extend_from_slice(heads);
        }
        all.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if !all.is_empty() {
            let median = all[all.len() / 2];
            let max = *all.last().unwrap();
            let long = all.iter().filter(|&&f| horizon(f) >= 10.0).count();
            println!(
                "\n  median head horizon: {:.1} words | longest head: {:.1} words | heads with ≥10-word memory: {}/{}",
                horizon(median),
                horizon(max),
                long,
                all.len(),
            );
            println!(
                "  → {}",
                if horizon(max) < 3.0 {
                    "state decays within ~2 words even at best — backbone CANNOT carry long-range (worth fixing)"
                } else if long == 0 {
                    "no head holds memory ≥10 words — effectively short-range despite the depth"
                } else {
                    "some heads CAN hold long memory — capability exists, it just isn't useful on this data"
                }
            );
        }
    }
}

/// 1/e memory horizon in words for a mean per-step retention `f̄ ∈ (0,1]`.
fn horizon(f: f32) -> f32 {
    if f >= 1.0 {
        f32::INFINITY
    } else if f <= 0.0 {
        0.0
    } else {
        -1.0 / f.ln()
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
    /// Where the model is saved during training. Set by `init_log`.
    save_path: String,
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
            save_path: String::new(),
            log_writer: None,
            last_log: Instant::now(),
            steps_since_log: 0,
            tokens_since_log: 0,
            extra_cols: Vec::new(),
            extra_vals: Vec::new(),
        }
    }

    pub fn init_log(&mut self, model_path: &str, extra_cols: &[&str]) {
        self.save_path = model_path.to_string();
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

    /// Path the model is saved to during training (set by `init_log`).
    pub fn save_path(&self) -> &str {
        &self.save_path
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
