use std::{
    f32::consts::PI,
    fs::File,
    io::{BufWriter, Write},
    time::{Duration, Instant},
};

use crate::{
    batches::ChunkedWordDataSet,
    config::{
        BATCH_SIZE, CHUNK_BYTES, DECAY_WINDOWS, EPOCHS, LOG_EVERY, LR, MAX_WINDOW_TOKENS, MIN_LR,
        MIN_WORDS_PER_SEQ, SAVE_EVERY, SEQ_LEN, TRAIN_DATA, VAL_DATA, WARMUP_WINDOWS,
        WORDS_PER_SEQ,
    },
    hierarchical::{BackboneMode, Hierarchical},
    model::{build_hierarchical_model, build_normal_model},
    optimizers::Optimizer,
    sequential::Sequential,
    tokenizer_utf8::Utf8Tokenizer,
};

pub fn train_normal(model_path: &str) {
    let tokenizer = Utf8Tokenizer::new();
    let vocab = tokenizer.vocab_size();

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
    println!("Streaming dataset from '{TRAIN_DATA}' in {CHUNK_BYTES}-byte chunks ...");
    let mut data = ChunkedWordDataSet::open(
        tokenizer,
        TRAIN_DATA,
        WORDS_PER_SEQ,
        MIN_WORDS_PER_SEQ,
        MAX_WINDOW_TOKENS,
        CHUNK_BYTES,
    );
    println!(
        "Training: {EPOCHS} epochs, LR={LR}, batch={BATCH_SIZE}, optimizer={:?}, log every {LOG_EVERY} steps",
        Optimizer {}
    );

    let mut training_state = TrainingState::new();
    training_state.init_log(model_path, &["word_loss"]);
    let mut total_time = Duration::ZERO;
    // Cache is sized to the longest window seen so far and grown on demand —
    // with streaming chunks the global maximum is not known up front.
    let mut cache_tokens = 0;

    for epoch in 1..=EPOCHS {
        println!("── Epoch {epoch} ───────────────────────────────────────");

        // Hierarchical training keeps corpus order (no shuffle) so both systems
        // see the same window order — leave it unshuffled here too for parity.

        let start = Instant::now();
        data.rewind();
        while let Some(chunk) = data.next_chunk() {
            if chunk.max_window_tokens() > cache_tokens {
                cache_tokens = chunk.max_window_tokens();
                model.make_cache(cache_tokens);
            }
            model.train_words(chunk.iter(), &mut training_state);
        }

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
    let tokenizer = Utf8Tokenizer::new();
    let vocab = tokenizer.vocab_size();

    let mut model = match Hierarchical::load(model_path, tokenizer) {
        Ok(m) => {
            println!("Loaded HM-RNN from '{model_path}'.");
            println!("Trained on so far:");
            print!("{}", m.seen.report());
            m
        }
        Err(e) => {
            println!("Could not load '{model_path}' ({e}) — creating new HM-RNN.");
            build_hierarchical_model(vocab, tokenizer)
        }
    };
    println!("Streaming dataset from '{TRAIN_DATA}' in {CHUNK_BYTES}-byte chunks ...");
    let mut data = ChunkedWordDataSet::open(
        tokenizer,
        TRAIN_DATA,
        WORDS_PER_SEQ,
        MIN_WORDS_PER_SEQ,
        MAX_WINDOW_TOKENS,
        CHUNK_BYTES,
    );
    println!(
        "Training: {EPOCHS} epochs, LR={LR}, batch={BATCH_SIZE}, optimizer={:?}, log every {LOG_EVERY} steps",
        Optimizer {}
    );

    let mut training_state = TrainingState::from_step(model.step);
    let extra_cols = ["delta_word", "delta_char1", "word_loss"];
    training_state.init_log(model_path, &extra_cols);
    let mut total_time = Duration::ZERO;
    // Cache is sized to the longest window seen so far and grown on demand.
    let mut cache_tokens = 0;

    // Where the last run stopped, from the sidecar next to the checkpoint. The
    // step count cannot answer this: it spans every corpus the weights have seen.
    let mut progress =
        crate::pretrain_progress::resume_or_fresh(model_path, TRAIN_DATA, model.step, EPOCHS);
    // Every run must stamp its window count into the sidecar, otherwise the
    // resume it writes cannot be validated later. `windows == 0` marks a count
    // that was never taken — unmeasured, not mismatched.
    let prep_start = Instant::now();
    let total = data.count_windows();
    println!(
        "  {total} windows total (counting pass took {:.1?})",
        prep_start.elapsed()
    );
    if !progress.is_fresh() && progress.windows != 0 && total != progress.windows {
        // A resume offset is only meaningful against the window count it was
        // measured with, so verify it before skipping anything.
        println!(
            "  corpus has {total} windows but the progress file recorded {} — \
             starting a fresh pass.",
            progress.windows
        );
        progress = crate::pretrain_progress::PretrainProgress::fresh(TRAIN_DATA, model.step);
    }
    progress.windows = total;
    let start_epoch = progress.epoch;
    let start_done = progress.done;

    for epoch in start_epoch..=EPOCHS {
        println!("── Epoch {epoch} ───────────────────────────────────────");

        // Only the resumed epoch skips; later epochs run whole.
        let mut skip = if epoch == start_epoch { start_done } else { 0 };
        if skip > 0 {
            println!("  Resuming from window {skip} (step {})", model.step);
        }
        progress.epoch = epoch;
        progress.done = skip;

        let start = Instant::now();
        data.rewind();
        while let Some(chunk) = data.next_chunk() {
            // Fast-forward over already-trained windows when resuming.
            if skip >= chunk.len() {
                skip -= chunk.len();
                continue;
            }
            if chunk.max_window_tokens() > cache_tokens {
                cache_tokens = chunk.max_window_tokens();
                model.make_cache(WORDS_PER_SEQ, cache_tokens);
            }
            let trained = chunk.len() - skip;
            model.train(chunk.iter().skip(skip), &mut training_state);
            // Chunk granularity: `train` consumes the iterator whole, so a stop
            // mid-chunk resumes from the chunk boundary before it.
            progress.done += trained;
            skip = 0;
        }

        let epoch_time = start.elapsed();
        total_time += epoch_time;

        match model.save(model_path) {
            Ok(()) => {
                println!(
                    "  ✓ end-of-epoch save to '{model_path}'  (epoch {epoch_time:.0?}, total {total_time:.0?})"
                );
                println!("    trained on: {}", model.seen.save_line());
                // The recorded position is the START of the next epoch, so a
                // stop here resumes without redoing the epoch just finished.
                progress.epoch = epoch + 1;
                progress.done = 0;
                progress.step = model.step;
                if let Err(e) = crate::pretrain_progress::save(model_path, &progress) {
                    eprintln!("  ✗ progress save failed: {e}");
                }
            }
            Err(e) => eprintln!("  ✗ end-of-epoch save failed: {e}"),
        }
    }

    // The run consumed all its epochs: drop the sidecar so the next invocation
    // starts at the beginning of whatever corpus it is pointed at.
    crate::pretrain_progress::clear(model_path);
    println!("Run complete — progress file cleared; the next run starts a fresh pass.");
}

/// CPU supervised fine-tuning (Q-A / instruction tuning) of a hierarchical
/// model. The CPU twin of `gpu::train::train_sft_gpu`: loads the (small) SFT set
/// fully into memory as masked chat windows (`crate::sft`) and fine-tunes with
/// loss counted only on the response tokens (`Hierarchical::train_sft`).
///
/// A model built by this crate already carries the SFT vocabulary; only an older
/// checkpoint (pretrained before the markers existed) must be run through the
/// `av` mode first — that mismatch is caught with a hint.
pub fn train_sft(model_path: &str) {
    use crate::config::{SFT_DATA, SFT_EPOCHS, SFT_LR, SFT_MAX_TOKENS};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;

    let tokenizer = Utf8Tokenizer::new();
    let vocab = tokenizer.vocab_size();

    let mut model = match Hierarchical::load(model_path, tokenizer) {
        Ok(m) => {
            println!(
                "Loaded hierarchical model from '{model_path}' (step {}).",
                m.step
            );
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

    if model.vocab_size != vocab {
        eprintln!(
            "This is an older checkpoint: its vocab is {} but the tokenizer now has \
             {vocab} tokens. Upgrade it once with the 'av' mode, then fine-tune the \
             grown model. Newly trained models already have the full vocab.",
            model.vocab_size
        );
        return;
    }

    let mut examples = match crate::sft::load_jsonl(&tokenizer, SFT_DATA, SFT_MAX_TOKENS) {
        Ok(e) if !e.is_empty() => e,
        Ok(_) => {
            eprintln!("SFT set '{SFT_DATA}' produced no trainable examples.");
            return;
        }
        Err(e) => {
            eprintln!("Could not read SFT set '{SFT_DATA}': {e}");
            return;
        }
    };

    // Size the caches to the longest example (word count / token span). Both
    // caps are known up front since the whole set is in memory.
    let max_words = examples.iter().map(|e| e.words.len()).max().unwrap();
    let max_tokens = examples.iter().map(|e| e.tokens.len()).max().unwrap();
    model.make_cache(max_words, max_tokens);
    println!(
        "SFT (CPU): {} examples, longest {max_words} words / {max_tokens} tokens. \
         {SFT_EPOCHS} epochs, LR={SFT_LR}, batch={BATCH_SIZE} windows.",
        examples.len()
    );

    let mut state = TrainingState::from_step(model.step);
    state.lr = SFT_LR;
    state.init_log(&format!("{model_path}_sft"), &["resp_ppl"]);

    // Where a previous, interrupted run stopped (or a fresh start). The shuffle
    // is seeded from this so resuming replays the same permutation and skipping
    // `done` examples skips exactly the ones already trained on.
    let mut progress =
        crate::sft_progress::resume_or_fresh(model_path, examples.len(), model.step, SFT_EPOCHS);
    let start_epoch = progress.epoch;
    let start_done = progress.done;

    let mut total_time = Duration::ZERO;
    for epoch in start_epoch..=SFT_EPOCHS {
        println!("── SFT epoch {epoch}/{SFT_EPOCHS} ──");
        // Per-epoch permutation derived from (seed, epoch), so it depends on the
        // run identity rather than on how many times the process restarted.
        let mut rng = StdRng::seed_from_u64(
            progress.seed ^ (epoch as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        );
        examples.shuffle(&mut rng);

        // Only the resumed epoch skips; later epochs run whole.
        let skip = if epoch == start_epoch { start_done } else { 0 };
        if skip > 0 {
            println!("  skipping {skip} examples already trained this epoch");
        }
        progress.epoch = epoch;
        progress.done = skip;

        let start = Instant::now();
        model.train_sft(examples.iter().skip(skip), &mut state, &mut progress);
        let epoch_time = start.elapsed();
        total_time += epoch_time;

        match model.save(model_path) {
            Ok(()) => {
                // The recorded position is the START of the next epoch, so a
                // stop here resumes without redoing the epoch just finished.
                progress.epoch = epoch + 1;
                progress.done = 0;
                progress.step = state.step;
                if let Err(e) = crate::sft_progress::save(model_path, &progress) {
                    eprintln!("  ✗ progress save failed: {e}");
                }
                println!(
                    "  ✓ end-of-epoch save to '{model_path}'  (epoch {epoch_time:.0?}, total {total_time:.0?})"
                );
                println!("    trained on: {}", model.seen.save_line());
            }
            Err(e) => eprintln!("  ✗ save failed: {e}"),
        }
    }

    // Every epoch done — drop the sidecar so the next invocation starts a new
    // run rather than resuming a completed one.
    crate::sft_progress::clear(model_path);
}

/// Debug: load the hierarchical model and print encoder/decoder inputs and
/// targets (forward) plus decoder targets (backward) for the first few words of
/// one window. No training. Set via the `ht` mode.
pub fn trace_hierarchical(model_path: &str) {
    const TRACE_WORDS: usize = 12;

    let tokenizer = Utf8Tokenizer::new();

    let mut model = match Hierarchical::load(model_path, tokenizer) {
        Ok(m) => m,
        Err(_) => build_hierarchical_model(tokenizer.vocab_size(), tokenizer),
    };
    let mut data = ChunkedWordDataSet::open(
        tokenizer,
        TRAIN_DATA,
        WORDS_PER_SEQ,
        MIN_WORDS_PER_SEQ,
        MAX_WINDOW_TOKENS,
        CHUNK_BYTES,
    );
    // The trace only needs the first window — one chunk is plenty.
    let Some(chunk) = data.next_chunk() else {
        eprintln!("no windows in dataset");
        return;
    };
    model.make_cache(WORDS_PER_SEQ, chunk.max_window_tokens());
    model.trace_io = true;

    let Some(batch) = chunk.iter().next() else {
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

    let tokenizer = Utf8Tokenizer::new();

    let mut model = match Hierarchical::load(model_path, tokenizer) {
        Ok(m) => {
            println!("Loaded '{model_path}' (step {}).", m.step);
            m
        }
        Err(e) => {
            eprintln!("Could not load '{model_path}': {e}");
            std::process::exit(2);
        }
    };
    println!("Preparing dataset from '{TRAIN_DATA}' ...");
    let mut data = ChunkedWordDataSet::open(
        tokenizer,
        TRAIN_DATA,
        WORDS_PER_SEQ,
        MIN_WORDS_PER_SEQ,
        MAX_WINDOW_TOKENS,
        CHUNK_BYTES,
    );
    // The probe only evaluates a handful of windows — the first chunk suffices.
    let Some(chunk) = data.next_chunk() else {
        eprintln!("no windows in dataset");
        std::process::exit(2);
    };
    model.make_cache(WORDS_PER_SEQ, chunk.max_window_tokens());
    let n = PROBE_WINDOWS.min(chunk.len());
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
        let (ce, we) = model.eval_decode_loss(chunk.iter().take(n));
        println!(
            "  {label:<34} Char loss = {ce:.4} nats   ppl = {:.3}",
            ce.exp()
        );
        println!(
            "  {label:<34} Word loss = {we:.4} nats   ppl = {:.3}",
            we.exp()
        );
        results.push((label, ce, we));
    }
    model.backbone_mode = BackboneMode::Normal;

    let c_normal = results[0].1;
    let w_normal = results[0].2;
    let c_reset = results[1].1;
    let w_reset = results[1].2;
    println!("\n  ── context usage ──");
    println!(
        "  long-range history (Reset → Normal): Char {:.4} nats, Word {:.4} <- what the deep backbone actually buys",
        c_reset - c_normal,
        w_reset - w_normal,
    );

    // ── Forget-gate / memory-horizon probe ──────────────────────────────
    // The decisive "can it remember?" measurement. f_prime ∈ (0,1] is the
    // fraction of cell state each backbone mLSTM head carries from word to
    // word. survival(k) = f̄^k, so the 1/e memory horizon is -1/ln(f̄) words.
    // Low horizon → the state decays (capacity it can't use); high horizon →
    // it CAN remember and simply doesn't, i.e. nothing to fix in the cell.
    //
    // Per head we also report the resting gate σ(bf) (retention at zero input
    // drive) and the observed min/max: a head whose swing is ~0 never
    // modulates — its horizon is just the learned bias, not data-dependent
    // behavior. Note bf is initialized in [3, 6] (σ ∈ [0.95, 0.998]), so
    // every head STARTS with a ≥20-word bias-only horizon; heads below that
    // were actively pulled down by training or by input drive.
    let bias = model.backbone_forget_bias();
    let mut samples: Vec<Vec<Vec<f32>>> = Vec::new();
    for crate::batches::WordBatch { tokens, words } in chunk.iter().take(n) {
        model.reset();
        model.forward_over(tokens, &words);
        let per_window = model.backbone_forget_samples();
        if samples.is_empty() {
            samples = per_window;
        } else {
            for (b, heads) in per_window.into_iter().enumerate() {
                for (h, s) in heads.into_iter().enumerate() {
                    samples[b][h].extend(s);
                }
            }
        }
    }

    let words_probed = samples.first().and_then(|b| b.first()).map_or(0, Vec::len);
    println!(
        "\n  ── backbone forget gates (over {} windows, {} words, {} blocks) ──",
        n,
        words_probed,
        samples.len(),
    );
    let mut all: Vec<f32> = Vec::new();
    let mut static_heads = 0;
    // A gate that never leaves this band around its resting point is static:
    // its horizon is set by the bias alone, not by what the model reads.
    const STATIC_SWING: f32 = 0.02;
    for (b, heads) in samples.iter().enumerate() {
        let means: Vec<f32> = heads
            .iter()
            .map(|s| (s.iter().map(|&v| v as f64).sum::<f64>() / s.len().max(1) as f64) as f32)
            .collect();
        let block_mean = means.iter().sum::<f32>() / means.len().max(1) as f32;
        println!(
            "  block {b:>2}: mean f̄ = {block_mean:.4} (~{:.1} words)",
            horizon(block_mean),
        );
        println!("      head   f̄ mean    horizon      min      max   rest σ(bf)    swing");
        for (h, s) in heads.iter().enumerate() {
            let mean = means[h];
            let min = s.iter().fold(1.0, |a: f32, &v| a.min(v));
            let max = s.iter().fold(0.0, |a: f32, &v| a.max(v));
            let rest = bias[b][h];
            let swing = max - min;
            if swing < STATIC_SWING {
                static_heads += 1;
            }
            println!(
                "      {h:>4}   {mean:.4}   {:>7.1} w   {min:.4}   {max:.4}       {rest:.4}   {swing:.4}{}",
                horizon(mean),
                if swing < STATIC_SWING {
                    "  (static)"
                } else {
                    ""
                },
            );
        }
        all.extend(means);
    }
    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if !all.is_empty() {
        let median = all[all.len() / 2];
        let max = *all.last().unwrap();
        let long = all.iter().filter(|&&f| horizon(f) >= 10.0).count();
        let buckets = [2.0, 10.0, 100.0];
        let mut counts = [0usize; 4];
        for &f in &all {
            let h = horizon(f);
            let idx = buckets.iter().position(|&b| h < b).unwrap_or(3);
            counts[idx] += 1;
        }
        println!(
            "\n  horizon buckets: <2 w: {} | 2–10 w: {} | 10–100 w: {} | ≥100 w: {}",
            counts[0], counts[1], counts[2], counts[3],
        );
        println!(
            "  median head horizon: {:.1} words | longest head: {:.1} words | heads with ≥10-word memory: {}/{} | static gates: {}/{}",
            horizon(median),
            horizon(max),
            long,
            all.len(),
            static_heads,
            all.len(),
        );
        println!(
            "  → {}",
            if horizon(max) < 3.0 {
                "state decays within ~2 words even at best"
            } else if long == 0 {
                "no head holds memory ≥10 words"
            } else if static_heads == all.len() {
                "all gates are static — horizons are pure bias, the input projection contributes nothing"
            } else {
                "some heads CAN hold long memory"
            }
        );
    }
}

/// Forward-only validation: streams the *entire* validation set through the
/// hierarchical model and reports mean decode char loss and word loss. No
/// backward, no weight updates. Set via the `hv` mode.
pub fn validate_hierarchical(model_path: &str) {
    let tokenizer = Utf8Tokenizer::new();

    let mut model = match Hierarchical::load(model_path, tokenizer) {
        Ok(m) => {
            println!("Loaded '{model_path}' (step {}).", m.step);
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
    let mut cache_tokens = 0;
    let mut c_total = 0.0;
    let mut w_total = 0.0;
    let mut windows = 0;
    while let Some(chunk) = data.next_chunk() {
        if chunk.max_window_tokens() > cache_tokens {
            cache_tokens = chunk.max_window_tokens();
            model.make_cache(WORDS_PER_SEQ, cache_tokens);
        }
        let (c, w, n) = model.eval_decode_loss_sums(chunk.iter());
        c_total += c;
        w_total += w;
        windows += n;
        println!(
            "  {windows} windows  char {:.4}  word {:.4}  ({:.0?})",
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
    println!(
        "\n── validation ({windows} windows, {:.0?}) ──",
        start.elapsed()
    );
    println!("  Char loss = {c_loss:.4} nats   ppl = {:.3}", c_loss.exp());
    println!("  Word loss = {w_loss:.4} nats   ppl = {:.3}", w_loss.exp());
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
    /// When true, `get_loss` writes CSV rows but does not flush them; the caller
    /// flushes explicitly via `flush_log` (aligned with model saves) so the log on
    /// disk never gets ahead of the last saved checkpoint.
    defer_log_flush: bool,
    last_log: Instant,
    steps_since_log: usize,
    tokens_since_log: usize,
    extra_cols: Vec<String>,
    extra_vals: Vec<(f32, usize)>,
    /// Total decode NLL (nats) and the byte count it was measured over, since the
    /// last save. Byte-weighted rather than a mean of per-window means, so windows
    /// count by how much text they actually carry.
    bpb_nats: f64,
    bpb_bytes: usize,
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
            defer_log_flush: false,
            last_log: Instant::now(),
            steps_since_log: 0,
            tokens_since_log: 0,
            extra_cols: Vec::new(),
            extra_vals: Vec::new(),
            bpb_nats: 0.0,
            bpb_bytes: 0,
        }
    }

    /// Accumulate one window's decode loss for the bits-per-byte metric. `loss` is
    /// the mean NLL per decoder ROW, `rows` the count it was averaged over (chars
    /// plus one `[W]` per word) and `bytes` the raw UTF-8 bytes those rows cover.
    /// The `[W]` rows are part of the model's cost but not of the text, so the
    /// total nats are charged against bytes only — that is what makes the number
    /// comparable to a byte-level model that never emits a word marker.
    pub fn log_bpb(&mut self, loss: f32, rows: usize, bytes: usize) {
        if bytes == 0 || !loss.is_finite() {
            return;
        }
        self.bpb_nats += loss as f64 * rows as f64;
        self.bpb_bytes += bytes;
    }

    /// Mean bits per byte since the last `take_bpb`, or `None` if nothing was
    /// accumulated.
    pub fn take_bpb(&mut self) -> Option<f32> {
        if self.bpb_bytes == 0 {
            return None;
        }
        let bpb = self.bpb_nats / self.bpb_bytes as f64 / std::f64::consts::LN_2;
        self.bpb_nats = 0.0;
        self.bpb_bytes = 0;
        Some(bpb as f32)
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
            // Schedule position is the number of windows seen, so the curve over
            // the corpus is the same at any BATCH_SIZE.
            let windows = self.step as f32;
            self.warmup_lr = self.lr * (windows / WARMUP_WINDOWS as f32).min(1.0);
            let t = (windows / DECAY_WINDOWS as f32).min(1.0);
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
                "{},{},{:e},{:.6},{:.4},{:.2},{:.4}{extra}",
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
        s.log_bpb(ln256, 20, 16);
        let bpb = s.take_bpb().expect("accumulated");
        assert!((bpb - 8.0 * 20.0 / 16.0).abs() < 1e-4, "got {bpb}");
        assert!(s.take_bpb().is_none(), "take must reset the accumulator");
    }

    /// Windows are weighted by their byte count, not averaged as per-window means.
    #[test]
    fn bpb_is_byte_weighted_across_windows() {
        let ln2 = std::f32::consts::LN_2;
        let mut s = TrainingState::new();
        s.log_bpb(ln2, 10, 10); // 1 bpb over 10 bytes
        s.log_bpb(3.0 * ln2, 90, 90); // 3 bpb over 90 bytes
        let bpb = s.take_bpb().expect("accumulated");
        // Byte-weighted: (10*1 + 90*3)/100 = 2.8, not the naive mean of 2.
        assert!((bpb - 2.8).abs() < 1e-4, "got {bpb}");
    }

    /// A window with no decoded bytes must not poison the average.
    #[test]
    fn bpb_ignores_empty_and_nonfinite_windows() {
        let mut s = TrainingState::new();
        s.log_bpb(1.0, 0, 0);
        assert!(s.take_bpb().is_none());
        s.log_bpb(f32::NAN, 5, 5);
        assert!(s.take_bpb().is_none());
    }
}
