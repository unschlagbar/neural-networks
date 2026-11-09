use std::{
    f32::consts::PI,
    fs::File,
    io::{BufWriter, Write},
    time::Instant,
};

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::{
    batches::ChunkedWordDataSet,
    checkpoint::ModelCfg,
    config::{
        BATCH_SIZE, CHUNK_BYTES, DECAY_WINDOWS, EPOCHS, GRAD_CLIP_NORM, LOG_EVERY, LR,
        MAX_WINDOW_TOKENS, MIN_LR, MIN_WORDS_PER_SEQ, RESUME_WARMUP_WINDOWS, SAVE_EVERY, SEQ_LEN,
        SFT_BATCH_SIZE, SFT_DATA, SFT_EPOCHS, SFT_LR, SFT_MAX_TOKENS, SFT_WARMUP_WINDOWS,
        TRAIN_DATA, VAL_DATA, WARMUP_WINDOWS, WORDS_PER_SEQ,
    },
    cpu::{hierarchical::Hierarchical, optim::AdamCfg},
    pretrain_progress, sft, sft_progress,
    tokenizer_utf8::Utf8Tokenizer,
};

/// Pretraining on the CPU (mode `h`): the loop of `gpu::train::train_hierarchical_gpu`
/// without the audio stage, and without carrying the backbone state from one window
/// into the next.
pub fn train_hierarchical(model_path: &str) {
    let tokenizer = Utf8Tokenizer::new();
    let cfg = ModelCfg::configured(tokenizer.vocab_size(), tokenizer.w_token() as usize);

    let (mut model, loaded) = match Hierarchical::load(model_path) {
        Ok(m) => {
            println!("Loaded '{model_path}' (step {}).", m.step_count);
            println!("Trained on so far:");
            print!("{}", m.seen.report());
            (m, true)
        }
        Err(e) => {
            println!("Could not load '{model_path}' ({e}) — creating a new model.");
            (Hierarchical::new(cfg), false)
        }
    };
    if model.cfg != cfg {
        eprintln!(
            "WARNING: checkpoint architecture {:?} differs from config.rs {:?} — \
             continuing with the checkpoint's.",
            model.cfg, cfg
        );
    }

    // A resumed run already knows which corpus it walks; only a new one asks.
    let corpus_root = match pretrain_progress::load(model_path) {
        Some(p) => {
            println!("Found a progress file — continuing on '{}'.", p.corpus);
            p.corpus
        }
        None => pretrain_progress::ask_corpus(TRAIN_DATA),
    };
    let corpus = match pretrain_progress::Corpus::open(&corpus_root) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Cannot read corpus '{corpus_root}': {e}");
            std::process::exit(1);
        }
    };
    println!(
        "Corpus '{}': {} file(s), streamed in {CHUNK_BYTES}-byte chunks",
        corpus.root,
        corpus.len()
    );
    println!("Training on CPU: {EPOCHS} epochs, LR={LR}, batch={BATCH_SIZE} windows, log every {LOG_EVERY} steps");

    let mut state = TrainingState::from_step(model.step_count);
    if loaded {
        state.schedule = LrSchedule::pretrain_resumed(model.step_count);
    }
    state.init_log(model_path, &["word_loss", "grad_norm"]);
    // The log on disk never gets ahead of the checkpoint: rows flush on save.
    state.set_defer_log_flush(true);
    let mut opt = AdamCfg::new(LR, crate::config::BACKBONE_WEIGHT_DECAY);

    let start = pretrain_progress::resume_or_fresh(model_path, &corpus, model.step_count, EPOCHS);
    let mut progress = start.progress;
    let start_epoch = progress.epoch;
    let start_file = start.file;
    let start_done = progress.done;
    if start_epoch > EPOCHS {
        return;
    }

    for epoch in start_epoch..=EPOCHS {
        println!("── Epoch {epoch} ───────────────────────────────────────");
        let epoch_start = Instant::now();
        let first_file = if epoch == start_epoch { start_file } else { 0 };

        for file_index in first_file..corpus.len() {
            let path = corpus.path_of(file_index).display().to_string();
            println!("── File {}/{}: {path} ─────────────────", file_index + 1, corpus.len());
            let mut data = ChunkedWordDataSet::open(
                tokenizer,
                &path,
                WORDS_PER_SEQ,
                MIN_WORDS_PER_SEQ,
                MAX_WINDOW_TOKENS,
                CHUNK_BYTES,
            );
            let mut skip = if epoch == start_epoch && file_index == start_file {
                start_done
            } else {
                0
            };
            // The resume offset is only meaningful against the window count it was
            // measured with; `windows == 0` is a count that was never taken.
            let t0 = Instant::now();
            let total = data.count_windows();
            println!("  {total} windows (counting pass took {:.1?})", t0.elapsed());
            if skip > 0 && progress.windows != 0 && total != progress.windows {
                println!(
                    "  file has {total} windows but the progress file recorded {} — \
                     starting this file from the beginning.",
                    progress.windows
                );
                skip = 0;
            }
            if skip > 0 {
                println!("  Resuming from window {skip} (step {})", model.step_count);
            }
            progress.epoch = epoch;
            progress.files_done = file_index;
            progress.file = corpus.name_of(file_index);
            progress.windows = total;
            progress.done = skip;

            let mut tokens_since_print = 0;
            let mut time = Instant::now();
            data.rewind();
            while let Some(chunk) = data.next_chunk() {
                if skip >= chunk.len() {
                    skip -= chunk.len();
                    continue;
                }
                for batch in chunk.iter().skip(skip) {
                    // Counts every window the iterator yields, so `done` stays
                    // aligned with where `chunk.iter().skip(done)` resumes.
                    progress.done += 1;
                    let tokens: Vec<usize> = batch.tokens.iter().map(|&t| t as usize).collect();
                    let words = &batch.words;
                    if words.len() < 2 {
                        continue;
                    }
                    model.set_doc_starts(&batch.doc_starts);
                    let loss = model.forward_backward(&tokens, words);
                    model.seen.add_pretrain(tokens.len(), words.len());
                    tokens_since_print += tokens.len();
                    state.log_tokens(tokens.len());
                    state.log_metric("word_loss", model.last_word_loss());
                    state.log_bpb(
                        loss,
                        model.last_rows(),
                        model.last_scored_tokens(),
                        model.last_scored_words(),
                    );

                    if let Some(lr) = state.step(loss) {
                        opt.lr = lr;
                        opt.t += 1;
                        // The gradient is a sum over BATCH_SIZE windows; the clip and
                        // the log are on the mean.
                        let batch = BATCH_SIZE as f32;
                        let norm = model.clip_grad_norm(GRAD_CLIP_NORM * batch);
                        state.log_metric("grad_norm", norm / batch);
                        model.step(&opt);
                    }
                    model.step_count = state.step;

                    if state.print() {
                        let word_loss = state.metric_mean("word_loss");
                        let loss = state.get_loss();
                        println!(
                            "{} | char loss {:.3} | ppl {:.3} | word loss {:.3} | lr {:.1e} | {} bytes | {:.1?}",
                            state.step,
                            loss,
                            loss.exp(),
                            word_loss,
                            opt.lr,
                            tokens_since_print,
                            time.elapsed(),
                        );
                        tokens_since_print = 0;
                        time = Instant::now();
                    }
                    if state.save() {
                        save_pretrain(&model, &mut state, &mut progress);
                    }
                }
                skip = 0;
            }
            // The recorded position is the START of the next file.
            progress.files_done = file_index + 1;
            progress.file = corpus.name_of(file_index + 1);
            progress.done = 0;
            progress.windows = 0;
            progress.step = state.step;
            if let Err(e) = pretrain_progress::save(state.save_path(), &progress) {
                eprintln!("progress save failed: {e}");
            }
        }

        println!("Epoch {epoch} took {:.1?}", epoch_start.elapsed());
        progress.epoch = epoch + 1;
        progress.files_done = 0;
        progress.file = corpus.name_of(0);
        progress.done = 0;
        progress.step = state.step;
        if let Err(e) = pretrain_progress::save(state.save_path(), &progress) {
            eprintln!("progress save failed: {e}");
        }
    }

    save_pretrain(&model, &mut state, &mut progress);
    // The sidecar stays, saying the corpus is done: it records which shards these
    // weights have seen, and deleting it is the user's call.
    println!(
        "Run complete — '{}' says the corpus is finished; delete it or point the \
         next run at another corpus.",
        pretrain_progress::progress_path(state.save_path()).display()
    );
}

/// Save the weights, then the log rows and the progress sidecar, so neither runs
/// ahead of the checkpoint.
fn save_pretrain(
    model: &Hierarchical,
    state: &mut TrainingState,
    progress: &mut pretrain_progress::PretrainProgress,
) {
    match model.save(state.save_path()) {
        Ok(()) => {
            state.flush_log();
            println!("saved -> {}", state.save_path());
            if let Some((bpb, word)) = state.take_bpb() {
                println!("  bpb {bpb:.3} | word loss {word:.3}");
            }
            println!("  trained on: {}", model.seen.save_line());
            progress.step = state.step;
            if let Err(e) = pretrain_progress::save(state.save_path(), progress) {
                eprintln!("progress save failed: {e}");
            }
        }
        Err(e) => eprintln!("save failed: {e}"),
    }
}

/// Q-A / instruction fine-tuning on the CPU (mode `hq`), the twin of
/// `gpu::train::train_sft_gpu` without heard turns: the whole (small) SFT set in
/// memory, the loss counted only on the response words.
pub fn train_sft(model_path: &str) {
    let tokenizer = Utf8Tokenizer::new();
    let vocab = tokenizer.vocab_size();

    let mut model = match Hierarchical::load(model_path) {
        Ok(m) => {
            println!("Loaded '{model_path}' (step {}).", m.step_count);
            println!("Trained on so far:");
            print!("{}", m.seen.report());
            m
        }
        Err(e) => {
            eprintln!(
                "Could not load '{model_path}' ({e}). SFT fine-tunes an existing \
                 pretrained model — train one with 'h' first."
            );
            return;
        }
    };
    if model.cfg.vocab != vocab {
        eprintln!(
            "This is an older checkpoint: its vocab is {} but the tokenizer now has \
             {vocab}: it predates the SFT markers and cannot be fine-tuned.",
            model.cfg.vocab
        );
        return;
    }

    let data = std::env::var("SFT_DATA").unwrap_or_else(|_| SFT_DATA.to_string());
    let mut examples = match sft::load_jsonl(&tokenizer, &data, SFT_MAX_TOKENS) {
        Ok(e) if !e.is_empty() => e,
        Ok(_) => {
            eprintln!("SFT set '{data}' produced no trainable examples.");
            return;
        }
        Err(e) => {
            eprintln!("Could not read SFT set '{data}': {e}");
            return;
        }
    };
    let max_words = examples.iter().map(|e| e.words.len()).max().unwrap();
    let max_tokens = examples.iter().map(|e| e.tokens.len()).max().unwrap();
    println!(
        "SFT (CPU): {} examples, longest {max_words} words / {max_tokens} tokens. \
         {SFT_EPOCHS} epochs, LR={SFT_LR} over {SFT_WARMUP_WINDOWS} warmup windows, \
         batch={SFT_BATCH_SIZE} windows.",
        examples.len()
    );

    let mut state = TrainingState::from_step(model.step_count);
    state.lr = SFT_LR;
    state.batch_size = SFT_BATCH_SIZE;
    state.init_log(&format!("{model_path}_sft"), &["resp_ppl"]);
    state.set_defer_log_flush(true);
    let mut opt = AdamCfg::new(SFT_LR, crate::config::BACKBONE_WEIGHT_DECAY);

    // The shuffle is seeded from the sidecar, so a resume replays the same
    // permutation and skipping `done` examples skips the ones already trained on.
    let mut progress =
        sft_progress::resume_or_fresh(model_path, examples.len(), model.step_count, SFT_EPOCHS);
    // Warmup counts from where this fine-tune began, not from the checkpoint's step.
    state.schedule = LrSchedule::sft(progress.origin);
    let start_epoch = progress.epoch;
    let start_done = progress.done;

    for epoch in start_epoch..=SFT_EPOCHS {
        println!("── SFT epoch {epoch}/{SFT_EPOCHS} ──");
        let mut rng = StdRng::seed_from_u64(
            progress.seed ^ (epoch as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        );
        examples.shuffle(&mut rng);
        let skip = if epoch == start_epoch { start_done } else { 0 };
        if skip > 0 {
            println!("  skipping {skip} examples already trained this epoch");
        }
        progress.epoch = epoch;
        progress.done = skip;

        let epoch_start = Instant::now();
        let mut tokens_since_print = 0;
        let mut time = Instant::now();
        for ex in examples.iter().skip(skip) {
            progress.done += 1;
            let tokens: Vec<usize> = ex.tokens.iter().map(|&t| t as usize).collect();
            let words = &ex.words;
            if words.len() < 2 {
                continue;
            }
            // One flag per DECODED word: word w decodes words[w + 1].
            let loss = model.forward_backward_masked(&tokens, words, &ex.loss[1..]);
            let (resp_chars, resp_words) = ex.response_extent();
            model.seen.add_sft(tokens.len(), words.len(), resp_chars, resp_words);
            tokens_since_print += tokens.len();
            state.log_tokens(tokens.len());
            state.log_metric("resp_ppl", loss.exp());

            if let Some(lr) = state.step(loss) {
                opt.lr = lr;
                opt.t += 1;
                model.step(&opt);
            }
            model.step_count = state.step;

            if state.print() {
                let loss = state.get_loss();
                println!(
                    "{} | resp loss {:.3} | ppl {:.3} | lr {:.2e} | {} tok | {:.1?}",
                    state.step,
                    loss,
                    loss.exp(),
                    opt.lr,
                    tokens_since_print,
                    time.elapsed(),
                );
                tokens_since_print = 0;
                time = Instant::now();
            }
            if state.save() {
                save_sft(&model, model_path, &mut state, &mut progress);
            }
        }
        println!("SFT epoch {epoch} took {:.1?}", epoch_start.elapsed());
        // The recorded position is the START of the next epoch.
        progress.epoch = epoch + 1;
        progress.done = 0;
        save_sft(&model, model_path, &mut state, &mut progress);
    }
    // Every epoch done: drop the sidecar so the next run starts fresh.
    sft_progress::clear(model_path);
}

fn save_sft(
    model: &Hierarchical,
    model_path: &str,
    state: &mut TrainingState,
    progress: &mut sft_progress::SftProgress,
) {
    match model.save(model_path) {
        Ok(()) => {
            state.flush_log();
            progress.step = state.step;
            if let Err(e) = sft_progress::save(model_path, progress) {
                eprintln!("progress save failed: {e}");
            }
            println!("saved -> {model_path}");
            println!("  trained on: {}", model.seen.save_line());
        }
        Err(e) => eprintln!("save failed: {e}"),
    }
}

/// Forward-only validation (mode `hv`): the whole of `VAL_DATA` through the model,
/// reporting the mean decode char loss and word loss. No backward, no update.
pub fn validate_hierarchical(model_path: &str) {
    let tokenizer = Utf8Tokenizer::new();
    let mut model = match Hierarchical::load(model_path) {
        Ok(m) => {
            println!("Loaded '{model_path}' (step {}).", m.step_count);
            m
        }
        Err(e) => {
            eprintln!("Could not load '{model_path}': {e}");
            std::process::exit(2);
        }
    };
    println!("Streaming validation set from '{VAL_DATA}' in {CHUNK_BYTES}-byte chunks ...");
    let mut data = ChunkedWordDataSet::open(
        tokenizer,
        VAL_DATA,
        WORDS_PER_SEQ,
        MIN_WORDS_PER_SEQ,
        MAX_WINDOW_TOKENS,
        CHUNK_BYTES,
    );

    let start = Instant::now();
    let (mut c_total, mut w_total, mut windows) = (0.0, 0.0, 0usize);
    while let Some(chunk) = data.next_chunk() {
        for batch in chunk.iter() {
            let tokens: Vec<usize> = batch.tokens.iter().map(|&t| t as usize).collect();
            if batch.words.len() < 2 {
                continue;
            }
            model.set_doc_starts(&batch.doc_starts);
            c_total += model.eval_loss(&tokens, &batch.words);
            w_total += model.last_word_loss();
            windows += 1;
        }
        println!(
            "  {windows} windows  char {:.3}  word {:.3}  ({:.0?})",
            c_total / windows.max(1) as f32,
            w_total / windows.max(1) as f32,
            start.elapsed(),
        );
    }

    if windows == 0 {
        eprintln!("no windows in validation set");
        std::process::exit(2);
    }
    let c_loss = c_total / windows as f32;
    let w_loss = w_total / windows as f32;
    println!("\n── validation ({windows} windows, {:.0?}) ──", start.elapsed());
    println!("  Char loss = {c_loss:.3} nats   ppl = {:.3}", c_loss.exp());
    println!("  Word loss = {w_loss:.3} nats   ppl = {:.3}", w_loss.exp());
}

/// The LR curve a run follows. Positions are counted in windows since
/// `origin`, so a fine-tune that resumes a checkpoint at step 400k still warms
/// up from the start of its own run. `warmup_windows == 0` disables the ramp
/// and `decay_windows == 0` the cosine, each leaving the rate at `lr`.
#[derive(Clone, Copy, Debug)]
pub struct LrSchedule {
    pub origin: usize,
    pub warmup_windows: usize,
    pub decay_windows: usize,
    pub min_lr: f32,
    /// A second linear ramp from `rewarm_origin`, multiplied onto the curve above;
    /// `rewarm_windows == 0` disables it. A resumed run needs it because its AdamW
    /// moments restart at zero: the first steps are then sign-like, every parameter
    /// moving by ~lr, and on `s5` one such step at the full rate took the held-out
    /// loss from 1.24 to 2.13 (at 0.1x the rate it improved it).
    pub rewarm_origin: usize,
    pub rewarm_windows: usize,
}

impl LrSchedule {
    /// Pretraining: long warmup, cosine decay to `MIN_LR`, counted from step 0.
    pub fn pretrain() -> Self {
        Self {
            origin: 0,
            warmup_windows: WARMUP_WINDOWS,
            decay_windows: DECAY_WINDOWS,
            min_lr: MIN_LR,
            rewarm_origin: 0,
            rewarm_windows: 0,
        }
    }

    /// Pretraining resumed at window `from`: the same curve, ramped up again from there.
    pub fn pretrain_resumed(from: usize) -> Self {
        Self {
            rewarm_origin: from,
            rewarm_windows: RESUME_WARMUP_WINDOWS,
            ..Self::pretrain()
        }
    }

    /// Fine-tuning: short warmup from `origin`, then hold.
    pub fn sft(origin: usize) -> Self {
        Self {
            origin,
            warmup_windows: crate::config::SFT_WARMUP_WINDOWS,
            decay_windows: 0,
            min_lr: 0.0,
            rewarm_origin: 0,
            rewarm_windows: 0,
        }
    }
}

pub struct TrainingState {
    pub step: usize,
    pub batch_size: usize,
    pub lr: f32,
    pub schedule: LrSchedule,
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
    /// When true, `get_loss` writes CSV rows but does not flush them; the caller
    /// flushes explicitly via `flush_log` (aligned with model saves) so the log on
    /// disk never gets ahead of the last saved checkpoint.
    defer_log_flush: bool,
    last_log: Instant,
    steps_since_log: usize,
    tokens_since_log: usize,
    extra_cols: Vec<String>,
    extra_vals: Vec<(f32, usize)>,
    /// Total decode NLL (nats) and the byte and word counts it was measured over,
    /// since the last save. Pooled rather than a mean of per-window means, so windows
    /// count by how much text they actually carry.
    bpb_nats: f64,
    bpb_bytes: usize,
    bpb_words: usize,
}

impl TrainingState {
    pub fn new() -> Self {
        Self::from_step(0)
    }

    pub fn from_step(step: usize) -> Self {
        Self {
            step,
            lr: LR,
            schedule: LrSchedule::pretrain(),
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
            defer_log_flush: false,
            last_log: Instant::now(),
            steps_since_log: 0,
            tokens_since_log: 0,
            extra_cols: Vec::new(),
            extra_vals: Vec::new(),
            bpb_nats: 0.0,
            bpb_bytes: 0,
            bpb_words: 0,
        }
    }

    /// Accumulate one window's decode loss for the save-time metrics. `loss` is
    /// the mean NLL per decoder ROW, `rows` the count it was averaged over (chars
    /// plus one `[W]` per word), `bytes` the raw UTF-8 bytes those rows cover and
    /// `words` the scored words they belong to.
    /// The `[W]` rows are part of the model's cost but not of the text, so the
    /// total nats are charged against bytes only — that is what makes bpb
    /// comparable to a byte-level model that never emits a word marker.
    pub fn log_bpb(&mut self, loss: f32, rows: usize, bytes: usize, words: usize) {
        if bytes == 0 || !loss.is_finite() {
            return;
        }
        self.bpb_nats += loss as f64 * rows as f64;
        self.bpb_bytes += bytes;
        self.bpb_words += words;
    }

    /// Since the last take: bits per byte, and nats per word with each word's `[W]`
    /// row in the numerator — both pooled over every scored word, unlike the logged
    /// `word_loss`, which averages per-window means. `None` if nothing accumulated.
    pub fn take_bpb(&mut self) -> Option<(f32, f32)> {
        if self.bpb_bytes == 0 {
            return None;
        }
        let bpb = self.bpb_nats / self.bpb_bytes as f64 / std::f64::consts::LN_2;
        let word = self.bpb_nats / self.bpb_words.max(1) as f64;
        self.bpb_nats = 0.0;
        self.bpb_bytes = 0;
        self.bpb_words = 0;
        Some((bpb as f32, word as f32))
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
        let extra_header: String = extra_cols.iter().map(|c| format!(",{c}")).collect();
        let header = format!("step,batch,lr,loss,perplexity,ms_per_step,s_per_norm_seq{extra_header}");
        if !is_new {
            // A resumed log written before a column existed: replace its header so the
            // new rows line up. Older rows simply end before the new columns.
            if let Ok(text) = std::fs::read_to_string(&path)
                && let Some((first, rest)) = text.split_once('\n')
                && first != header
            {
                let _ = std::fs::write(&path, format!("{header}\n{rest}"));
            }
        }
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            Ok(f) => {
                let mut writer = BufWriter::new(f);
                if is_new {
                    let _ = writeln!(writer, "{header}");
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
            // Schedule position is the number of windows seen since the run
            // started, so the curve over the corpus is the same at any batch size.
            let sched = self.schedule;
            let windows = self.step.saturating_sub(sched.origin) as f32;
            self.warmup_lr = if sched.warmup_windows == 0 {
                self.lr
            } else {
                self.lr * (windows / sched.warmup_windows as f32).min(1.0)
            };
            self.decay_lr = if sched.decay_windows == 0 {
                self.lr
            } else {
                let t = (windows / sched.decay_windows as f32).min(1.0);
                sched.min_lr + 0.5 * (self.lr - sched.min_lr) * (1.0 + (PI * t).cos())
            };
            self.current_lr = self.warmup_lr.min(self.decay_lr);
            if sched.rewarm_windows > 0 {
                let since = self.step.saturating_sub(sched.rewarm_origin) as f32;
                self.current_lr *= (since / sched.rewarm_windows as f32).min(1.0);
            }
            Some(self.current_lr)
        } else {
            None
        }
    }

    pub fn print(&mut self) -> bool {
        self.step.is_multiple_of(self.print_interval)
    }

    /// Running mean of an extra column since the last log, without consuming it.
    /// `get_loss` resets the accumulators, so read this before calling it.
    pub fn metric_mean(&self, name: &str) -> f32 {
        match self.extra_cols.iter().position(|c| c == name) {
            Some(i) if self.extra_vals[i].1 > 0 => {
                self.extra_vals[i].0 / self.extra_vals[i].1 as f32
            }
            _ => 0.0,
        }
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
                "{},{},{:e},{:.6},{:.3},{:.2},{:.3}{extra}",
                self.step,
                batch_num,
                self.current_lr,
                loss,
                loss.exp(),
                ms_per_step,
                s_per_norm,
            );
            if !self.defer_log_flush {
                let _ = writer.flush();
            }
            self.extra_vals.fill((0.0, 0));
        }
        self.last_log = Instant::now();
        self.steps_since_log = 0;
        self.tokens_since_log = 0;
        loss
    }

    pub fn save(&self) -> bool {
        self.step.is_multiple_of(self.save_interval)
    }

    /// Defer CSV flushing to explicit `flush_log` calls (see the field docs).
    pub fn set_defer_log_flush(&mut self, defer: bool) {
        self.defer_log_flush = defer;
    }

    /// Flush any buffered CSV rows to disk. Call right after a successful model
    /// save so the log never reflects a state newer than the checkpoint.
    pub fn flush_log(&mut self) {
        if let Some(writer) = &mut self.log_writer {
            let _ = writer.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A uniform distribution over 256 byte values costs exactly 8 bits per row.
    /// With one `[W]` row per word those rows are extra cost spread over the same
    /// bytes, so bpb must come out ABOVE 8 — the row mean alone would say 8.
    #[test]
    fn bpb_charges_marker_rows_against_bytes() {
        let mut s = TrainingState::new();
        let ln256 = 256f32.ln();
        // 4 words x 4 bytes = 16 bytes, 20 rows (4 markers).
        s.log_bpb(ln256, 20, 16, 4);
        let (bpb, word) = s.take_bpb().expect("accumulated");
        assert!((bpb - 8.0 * 20.0 / 16.0).abs() < 1e-4, "got {bpb}");
        // 20 rows of ln 256 nats over 4 words, the markers included.
        assert!((word - ln256 * 5.0).abs() < 1e-3, "got {word}");
        assert!(s.take_bpb().is_none(), "take must reset the accumulator");
    }

    /// Windows are weighted by their byte count, not averaged as per-window means.
    #[test]
    fn bpb_is_byte_weighted_across_windows() {
        let ln2 = std::f32::consts::LN_2;
        let mut s = TrainingState::new();
        s.log_bpb(ln2, 10, 10, 2); // 1 bpb over 10 bytes
        s.log_bpb(3.0 * ln2, 90, 90, 3); // 3 bpb over 90 bytes
        let (bpb, word) = s.take_bpb().expect("accumulated");
        // Pooled over the 5 words: (10 + 270) ln 2 / 5, not a mean of window means.
        assert!((word - 56.0 * ln2).abs() < 1e-4, "got {word}");
        // Byte-weighted: (10*1 + 90*3)/100 = 2.8, not the naive mean of 2.
        assert!((bpb - 2.8).abs() < 1e-4, "got {bpb}");
    }

    /// A window with no decoded bytes must not poison the average.
    #[test]
    fn bpb_ignores_empty_and_nonfinite_windows() {
        let mut s = TrainingState::new();
        s.log_bpb(1.0, 0, 0, 0);
        assert!(s.take_bpb().is_none());
        s.log_bpb(f32::NAN, 5, 5, 1);
        assert!(s.take_bpb().is_none());
    }
}
