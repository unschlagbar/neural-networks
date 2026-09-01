//! GPU training loop for the hierarchical model.
//!
//! Mirrors `training::train_hierarchical` — same streaming dataset, same
//! `TrainingState` (LR warmup/cosine-decay schedule, CSV logging, print/save
//! intervals), same gradient accumulation over `BATCH_SIZE` windows — but the
//! whole model lives on the GPU (`gpu::Hierarchical`).
//!
//! Checkpoints use the CPU `HIER` format (see `gpu::hierarchical`), written to
//! `<model_path>` every `SAVE_EVERY` steps and reloaded on startup, so a run can
//! be stopped and resumed — and the same file opens in `hp` / `hs`.

use std::time::Instant;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::batches::ChunkedWordDataSet;
use crate::config::{
    BATCH_SIZE, CHAR_HIDDEN, CHUNK_BYTES, EPOCHS, LOG_EVERY, LOGIT_SOFTCAP, LR, MAX_WINDOW_TOKENS,
    MIN_WORDS_PER_SEQ, SFT_DATA, SFT_EPOCHS, SFT_LR, SFT_MAX_TOKENS, TRAIN_DATA, VAL_DATA,
    WORD_BLOCKS, WORD_HIDDEN, WORDS_PER_SEQ,
};
use crate::gpu::Gpu;
use crate::gpu::hierarchical::{Hierarchical, ModelCfg};
use crate::nn2::optim::AdamCfg;
use crate::pretrain_progress;
use crate::sft;
use crate::sft_progress;
use crate::tokenizer_utf8::Utf8Tokenizer;
use crate::training::TrainingState;

/// Architecture, taken from `config.rs` so the GPU model matches the CPU one.
/// `heads`/`dqk` mirror `model.rs::build_hierarchical_model`.
fn cfg_from_config(vocab: usize, w_token: usize) -> ModelCfg {
    let heads = 8;
    ModelCfg {
        vocab,
        hc: CHAR_HIDDEN,
        wh: WORD_HIDDEN,
        enc_blocks: 4,
        bb_blocks: WORD_BLOCKS,
        dec_blocks: 4,
        heads,
        dqk: WORD_HIDDEN / heads,
        w_token,
        cap: LOGIT_SOFTCAP,
    }
}

pub fn train_hierarchical_gpu(model_path: &str) {
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No CUDA GPU available ({e}). Use 'h' for the CPU trainer.");
            return;
        }
    };

    let tokenizer = Utf8Tokenizer::new();
    let vocab = tokenizer.vocab_size();
    let w_token = tokenizer.w_token() as usize;
    let cfg = cfg_from_config(vocab, w_token);

    let mut model = match Hierarchical::load(&gpu, model_path, w_token) {
        Ok(m) => {
            println!(
                "Loaded GPU hierarchical model from '{model_path}' (step {}).",
                m.step_count
            );
            println!("Trained on so far:");
            print!("{}", m.seen.report());
            m
        }
        Err(e) => {
            println!("Could not load '{model_path}' ({e}) — creating new GPU model.");
            Hierarchical::new(&gpu, cfg)
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
    // Pointing a resume at a different corpus by accident is exactly the
    // mistake the sidecar exists to prevent.
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
    println!(
        "Training on GPU: {EPOCHS} epochs, LR={LR}, batch={BATCH_SIZE} windows, \
         log every {LOG_EVERY} steps"
    );

    let mut state = TrainingState::from_step(model.step_count);
    state.init_log(model_path, &["word_loss"]);
    // Buffer CSV rows and only flush them when the model is saved (every
    // SAVE_EVERY steps), so the on-disk log never gets ahead of the checkpoint.
    state.set_defer_log_flush(true);

    let mut opt = AdamCfg::new(LR, crate::optimizers::WEIGHT_DECAY);

    // Where the last run stopped, from the sidecar next to the checkpoint. The
    // step count cannot answer this: it spans every corpus the weights have seen.
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
        // Only the resumed epoch resumes mid-corpus; later epochs run whole.
        let first_file = if epoch == start_epoch { start_file } else { 0 };

        for file_index in first_file..corpus.len() {
        let path = corpus.path_of(file_index).display().to_string();
        println!(
            "── File {}/{}: {path} ─────────────────",
            file_index + 1,
            corpus.len()
        );
        let mut data = ChunkedWordDataSet::open(
            tokenizer,
            &path,
            WORDS_PER_SEQ,
            MIN_WORDS_PER_SEQ,
            MAX_WINDOW_TOKENS,
            CHUNK_BYTES,
        );
        // Only the file the run stopped inside skips; the rest run whole.
        let mut skip = if epoch == start_epoch && file_index == start_file {
            start_done
        } else {
            0
        };
        // Every run must stamp its window count into the sidecar, otherwise the
        // resume it writes cannot be validated later. `windows == 0` marks a
        // count that was never taken — unmeasured, not mismatched.
        let t0 = Instant::now();
        let total = data.count_windows();
        println!(
            "  {total} windows (counting pass took {:.1?})",
            t0.elapsed()
        );
        if skip > 0 && progress.windows != 0 && total != progress.windows {
            // A resume offset is only meaningful against the window count it
            // was measured with, so verify it before skipping anything.
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
                // Counts every window the iterator yields, including the ones
                // skipped below — `done` must stay aligned with the position
                // `chunk.iter().skip(done)` resumes at.
                progress.done += 1;

                // The dataset speaks u16 / Range; the model takes usize / (start, end).
                let tokens: Vec<usize> = batch.tokens.iter().map(|&t| t as usize).collect();
                let words = &batch.words;
                if words.len() < 2 {
                    continue; // no decoded word in this window
                }

                let loss = model.forward_backward(&gpu, &tokens, words);
                model.seen.add_pretrain(tokens.len(), words.len());
                tokens_since_print += tokens.len();
                state.log_tokens(tokens.len());
                state.log_metric("word_loss", model.last_word_loss());
                // Bits per byte counts the decoded words' raw bytes only: the `[W]`
                // rows are part of the model's cost but not of the text.
                let dec_bytes: usize = words[1..].iter().map(|w| w.end - w.start).sum();
                state.log_bpb(loss, model.last_rows(), dec_bytes);

                // `state.step` returns Some(lr) only on a batch boundary, so grads
                // accumulate over BATCH_SIZE windows before each optimizer step.
                if let Some(lr) = state.step(loss) {
                    opt.lr = lr;
                    opt.t += 1;
                    model.step(&gpu, &opt);
                }
                model.step_count = state.step;

                if state.print() {
                    let word_loss = state.metric_mean("word_loss");
                    let loss = state.get_loss();
                    println!(
                        "{} | char loss {:.4} | ppl {:.4} | word loss {:.4} | lr {:.2e} | {} tok | {:.1?}",
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
                    match model.save(&gpu, state.save_path(), &[]) {
                        Ok(()) => {
                            // Flush the log only now, so it never reflects a step
                            // past the checkpoint just written.
                            state.flush_log();
                            println!("saved -> {}", state.save_path());
                            if let Some(bpb) = state.take_bpb() {
                                println!("  bpb {bpb:.4} (mean since last save)");
                            }
                            println!("  trained on: {}", model.seen.save_line());
                            // Written after the weights, so the recorded position
                            // never runs ahead of the checkpoint it describes.
                            progress.step = state.step;
                            if let Err(e) = pretrain_progress::save(state.save_path(), &progress) {
                                eprintln!("progress save failed: {e}");
                            }
                        }
                        Err(e) => eprintln!("save failed: {e}"),
                    }
                }
            }
            skip = 0;
        }
        // The file is finished: the recorded position is the START of the next
        // one, so a stop here resumes without re-reading it.
        progress.files_done = file_index + 1;
        progress.file = corpus.name_of(file_index + 1);
        progress.done = 0;
        progress.windows = 0;
        progress.step = state.step;
        if let Err(e) = pretrain_progress::save(state.save_path(), &progress) {
            eprintln!("progress save failed: {e}");
        }
        } // files

        println!("Epoch {epoch} took {:.1?}", epoch_start.elapsed());
        // The recorded position is the START of the next epoch, so a stop here
        // resumes without redoing the epoch just finished.
        progress.epoch = epoch + 1;
        progress.files_done = 0;
        progress.file = corpus.name_of(0);
        progress.done = 0;
        progress.step = state.step;
        if let Err(e) = pretrain_progress::save(state.save_path(), &progress) {
            eprintln!("progress save failed: {e}");
        }
    }

    match model.save(&gpu, state.save_path(), &[]) {
        Ok(()) => {
            state.flush_log();
            println!("final save -> {}", state.save_path());
            if let Some(bpb) = state.take_bpb() {
                println!("  bpb {bpb:.4} (mean since last save)");
            }
            println!("  trained on: {}", model.seen.save_line());
        }
        Err(e) => eprintln!("final save failed: {e}"),
    }
    // The sidecar is left behind saying the corpus is done: it is the record of
    // which shards these weights have seen, and deleting it is the user's call.
    println!(
        "Run complete — '{}' says the corpus is finished; delete it or point the \
         next run at another corpus.",
        pretrain_progress::progress_path(state.save_path()).display()
    );
}

/// Forward validation of a GPU checkpoint: streams the whole validation set
/// (`config::VAL_DATA`) through the model and reports mean decode char loss and
/// word loss. The GPU twin of `training::validate_hierarchical`, mode `hvg`.
///
/// Runs `Hierarchical::eval_loss`, which stops at the decode loss: no backward
/// phase runs, no gradient is produced and no optimizer step is taken, so the
/// checkpoint is untouched.
pub fn validate_hierarchical_gpu(model_path: &str) {
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No CUDA GPU available ({e}). Use 'hv' for the CPU validator.");
            return;
        }
    };

    let tokenizer = Utf8Tokenizer::new();
    let w_token = tokenizer.w_token() as usize;

    let mut model = match Hierarchical::load(&gpu, model_path, w_token) {
        Ok(m) => {
            println!("Loaded '{model_path}' (step {}).", m.step_count);
            m
        }
        Err(e) => {
            eprintln!("Could not load '{model_path}': {e}");
            std::process::exit(2);
        }
    };

    // Nothing will unwind this pass, so parking the backbone's activations on the
    // host would only pay for a D2H no backward ever reads back.
    model.set_offload(&gpu, false);

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
    let mut c_total = 0.0;
    let mut w_total = 0.0;
    let mut windows = 0usize;
    while let Some(chunk) = data.next_chunk() {
        for batch in chunk.iter() {
            let tokens: Vec<usize> = batch.tokens.iter().map(|&t| t as usize).collect();
            let words = &batch.words;
            if words.len() < 2 {
                continue; // no decoded word in this window
            }
            c_total += model.eval_loss(&gpu, &tokens, words);
            w_total += model.last_word_loss();
            windows += 1;
        }
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

/// GPU supervised fine-tuning (Q-A instruction tuning) of a pretrained
/// hierarchical model.
///
/// Loads the SFT set (`config::SFT_DATA`) fully into memory as masked chat
/// windows (see `crate::sft`) — the corpus is small — and fine-tunes the model
/// with loss counted only on the response tokens (`forward_backward_masked`).
///
/// A model built by this crate already has the full SFT vocabulary (new models
/// are created at `tokenizer.vocab_size()`), so it just works. Only an *older*
/// checkpoint — pretrained back when the vocab was smaller — lacks the SFT
/// marker rows; such a checkpoint is rejected with a hint to run `av`
/// (`grow_vocab`) once, rather than silently indexing past the tied table.
pub fn train_sft_gpu(model_path: &str) {
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No CUDA GPU available ({e}).");
            return;
        }
    };

    let tokenizer = Utf8Tokenizer::new();
    let vocab = tokenizer.vocab_size();
    let w_token = tokenizer.w_token() as usize;

    let mut model = match Hierarchical::load(&gpu, model_path, w_token) {
        Ok(m) => {
            println!(
                "Loaded GPU hierarchical model from '{model_path}' (step {}).",
                m.step_count
            );
            println!("Trained on so far:");
            print!("{}", m.seen.report());
            m
        }
        Err(e) => {
            eprintln!(
                "Could not load '{model_path}' ({e}). SFT fine-tunes an existing \
                 pretrained model — train one with 'hg' first."
            );
            return;
        }
    };

    if model.cfg.vocab != vocab {
        eprintln!(
            "This is an older checkpoint: its vocab is {} but the tokenizer now has \
             {vocab} tokens (the SFT markers were added later). Upgrade it once with \
             the 'av' mode, then fine-tune the grown model. Newly trained models \
             already have the full vocab and skip this step.",
            model.cfg.vocab
        );
        return;
    }

    let mut examples = match sft::load_jsonl(&tokenizer, SFT_DATA, SFT_MAX_TOKENS) {
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

    // Size the caches to the longest example (word count + token span).
    let max_words = examples.iter().map(|e| e.words.len()).max().unwrap();
    let max_tokens = examples.iter().map(|e| e.tokens.len()).max().unwrap();
    // The GPU model builds its own per-window rectangles, so no persistent cache
    // allocation is needed here (unlike the CPU path); max_* only informs logs.
    println!(
        "SFT: {} examples, longest {max_words} words / {max_tokens} tokens. \
         {SFT_EPOCHS} epochs, LR={SFT_LR}, batch={BATCH_SIZE} windows.",
        examples.len()
    );

    let mut state = TrainingState::from_step(model.step_count);
    state.lr = SFT_LR;
    state.init_log(&format!("{model_path}_sft"), &["resp_ppl"]);
    state.set_defer_log_flush(true);

    let mut opt = AdamCfg::new(SFT_LR, crate::optimizers::WEIGHT_DECAY);

    // Where a previous, interrupted run stopped (or a fresh start). The shuffle
    // is seeded from this so resuming replays the exact same permutation and
    // skipping `done` examples really skips the ones already trained on.
    let mut progress =
        sft_progress::resume_or_fresh(model_path, examples.len(), model.step_count, SFT_EPOCHS);
    let start_epoch = progress.epoch;
    let start_done = progress.done;

    for epoch in start_epoch..=SFT_EPOCHS {
        println!("── SFT epoch {epoch}/{SFT_EPOCHS} ──");
        // Per-epoch permutation, derived from (seed, epoch) so it is a function
        // of the run identity, not of how many times the process restarted.
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

        let epoch_start = Instant::now();
        let mut tokens_since_print = 0usize;
        let mut time = Instant::now();

        for ex in examples.iter().skip(skip) {
            // Counted for every example the loop consumes — including the ones
            // skipped below — so `done` always means "examples of this epoch's
            // permutation already passed", which is exactly what resume skips.
            progress.done += 1;

            let tokens: Vec<usize> = ex.tokens.iter().map(|&t| t as usize).collect();
            let words = &ex.words;
            if words.len() < 2 {
                continue;
            }
            // `word_loss` is per DECODED word: word w decodes words[w+1], so its
            // flag is ex.loss[w+1] (ex.loss[0] belongs to the encode-only prefix).
            let word_loss: Vec<bool> = ex.loss[1..].to_vec();

            let loss = model.forward_backward_masked(&gpu, &tokens, words, &word_loss);
            let (resp_chars, resp_words) = ex.response_extent();
            model
                .seen
                .add_sft(tokens.len(), words.len(), resp_chars, resp_words);
            tokens_since_print += tokens.len();
            state.log_tokens(tokens.len());
            state.log_metric("resp_ppl", loss.exp());

            if let Some(lr) = state.step(loss) {
                opt.lr = lr;
                opt.t += 1;
                model.step(&gpu, &opt);
            }
            model.step_count = state.step;

            if state.print() {
                let loss = state.get_loss();
                println!(
                    "{} | resp loss {:.4} | ppl {:.4} | lr {:.2e} | {} tok | {:.1?}",
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
                match model.save(&gpu, model_path, &[]) {
                    Ok(()) => {
                        state.flush_log();
                        // Written right after the weights, so the recorded
                        // position never runs ahead of the checkpoint.
                        progress.step = state.step;
                        if let Err(e) = sft_progress::save(model_path, &progress) {
                            eprintln!("progress save failed: {e}");
                        }
                        println!("saved -> {model_path}");
                        println!("  trained on: {}", model.seen.save_line());
                    }
                    Err(e) => eprintln!("save failed: {e}"),
                }
            }
        }
        println!("SFT epoch {epoch} took {:.1?}", epoch_start.elapsed());
        // Save at every epoch boundary too. The recorded position is the START
        // of the next epoch, so a stop here resumes without redoing this one.
        if let Err(e) = model.save(&gpu, model_path, &[]) {
            eprintln!("epoch save failed: {e}");
        } else {
            state.flush_log();
            progress.epoch = epoch + 1;
            progress.done = 0;
            progress.step = state.step;
            if let Err(e) = sft_progress::save(model_path, &progress) {
                eprintln!("progress save failed: {e}");
            }
            println!("  trained on: {}", model.seen.save_line());
        }
    }

    match model.save(&gpu, model_path, &[]) {
        Ok(()) => {
            state.flush_log();
            // The run finished every epoch — drop the sidecar so the next
            // invocation starts a new run instead of resuming a completed one.
            sft_progress::clear(model_path);
            println!("final SFT save -> {model_path}");
            println!("  trained on: {}", model.seen.save_line());
        }
        Err(e) => eprintln!("final save failed: {e}"),
    }
}
