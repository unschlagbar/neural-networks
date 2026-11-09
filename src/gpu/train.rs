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
use crate::checkpoint;
use crate::config::{
    AUDIO_STAGE_LR, BATCH_SIZE, CARRY_WINDOW_STATE, CHAR_HIDDEN, CHUNK_BYTES, EPOCHS, GRAD_CLIP_NORM, LOG_EVERY,
    LOGIT_SOFTCAP, LR, MAX_WINDOW_TOKENS, MIN_WORDS_PER_SEQ, SFT_BATCH_SIZE, SFT_DATA, SFT_EPOCHS,
    SFT_LR, SFT_MAX_TOKENS, SFT_WARMUP_WINDOWS, TRAIN_DATA, VAL_DATA, WORD_BLOCKS, WORD_HIDDEN,
    WORDS_PER_SEQ,
};
use crate::gpu::Gpu;
use crate::gpu::audio_train::{AudioPretrain, SftAudio};
use crate::gpu::hierarchical::{Hierarchical, ModelCfg};
use crate::nn2::optim::AdamCfg;
use crate::pretrain_progress;
use crate::sft;
use crate::sft_progress;
use crate::tokenizer_utf8::Utf8Tokenizer;
use crate::training::TrainingState;

/// `audio` is the stage after text pretraining: the same loop, with audio
/// windows among the text ones (`AudioPretrain`) and the backbone at the
/// constant `AUDIO_STAGE_LR`, on a copy `<model>_audio` of the model.
pub fn train_hierarchical_gpu(model_path: &str, audio: bool) {
    // The audio stage never writes the text model it starts from: it trains
    // `<model>_audio`, copied from `<model>` on the first run.
    let stage_path;
    let model_path = if audio {
        stage_path = format!("{model_path}_audio");
        if !std::path::Path::new(&stage_path).exists() {
            println!("Copying '{model_path}' to '{stage_path}' for the audio stage.");
            if let Err(e) = std::fs::copy(model_path, &stage_path) {
                eprintln!("Cannot copy '{model_path}' to '{stage_path}': {e}");
                return;
            }
        }
        stage_path.as_str()
    } else {
        model_path
    };
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

    let heads = 8;
    let cfg = ModelCfg {
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
    };

    let (mut model, loaded) = match Hierarchical::load(&gpu, model_path, w_token) {
        Ok(m) => {
            println!(
                "Loaded GPU hierarchical model from '{model_path}' (step {}).",
                m.step_count
            );
            println!("Trained on so far:");
            print!("{}", m.seen.report());
            (m, true)
        }
        Err(e) => {
            println!("Could not load '{model_path}' ({e}) — creating new GPU model.");
            (Hierarchical::new(&gpu, cfg), false)
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
    if loaded {
        state.schedule = crate::training::LrSchedule::pretrain_resumed(model.step_count);
    }
    state.init_log(model_path, &["word_loss", "grad_norm"]);
    // Buffer CSV rows and only flush them when the model is saved (every
    // SAVE_EVERY steps), so the on-disk log never gets ahead of the checkpoint.
    state.set_defer_log_flush(true);

    let mut opt = AdamCfg::new(LR, crate::optimizers::WEIGHT_DECAY);
    let mut capture = crate::gpu::profile::Capture::from_env();
    let mut audio = if audio {
        let a = AudioPretrain::open(&gpu, model_path, model.cfg.hc);
        if a.is_none() {
            eprintln!("The audio stage needs an audio encoder; nothing to train.");
            return;
        }
        a
    } else {
        None
    };
    let mut text_windows = 0usize;

    // The loop below walks a file's windows in corpus order, which is what makes a
    // window's `continues` flag mean anything.
    model.set_stateful(CARRY_WINDOW_STATE);

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
                // Whether the window just trained is the one a `continues` flag refers
                // to. False at the start of a chunk and after any window the loop
                // skipped, so a resume never inherits state from a document it did not
                // run — and false with `CARRY_WINDOW_STATE` off, where nothing carries.
                let mut ran_prev = false;
                for batch in chunk.iter().skip(skip) {
                    // Counts every window the iterator yields, including the ones
                    // skipped below — `done` must stay aligned with the position
                    // `chunk.iter().skip(done)` resumes at.
                    progress.done += 1;

                    // The dataset speaks u16 / Range; the model takes usize / (start, end).
                    let tokens: Vec<usize> = batch.tokens.iter().map(|&t| t as usize).collect();
                    let words = &batch.words;
                    if words.len() < 2 {
                        ran_prev = false;
                        continue; // no decoded word in this window
                    }
                    model.set_continues(batch.continues && ran_prev);
                    model.set_doc_starts(&batch.doc_starts);
                    ran_prev = true;

                    if let Some(c) = capture.as_mut() {
                        c.before_window(&gpu);
                    }
                    let loss = model.forward_backward(&gpu, &tokens, words);
                    text_windows += 1;
                    if let Some(a) = audio.as_mut().filter(|_| AudioPretrain::due(text_windows)) {
                        a.window(&gpu, &mut model);
                    }
                    model.seen.add_pretrain(tokens.len(), words.len());
                    tokens_since_print += tokens.len();
                    state.log_tokens(tokens.len());
                    state.log_metric("word_loss", model.last_word_loss());
                    // Bits per byte counts the scored words' raw bytes only: the `[W]`
                    // rows are part of the model's cost but not of the text, and a
                    // border word the loss masks costs no nats, so its bytes are out too.
                    state.log_bpb(
                        loss,
                        model.last_rows(),
                        model.last_scored_tokens(),
                        model.last_scored_words(),
                    );

                    // `state.step` returns Some(lr) only on a batch boundary, so grads
                    // accumulate over BATCH_SIZE windows before each optimizer step.
                    if let Some(lr) = state.step(loss) {
                        opt.lr = if audio.is_some() { AUDIO_STAGE_LR } else { lr };
                        opt.t += 1;
                        // The accumulated gradient is a sum over BATCH_SIZE windows; the
                        // clip and the log are on the mean, so neither depends on it.
                        let batch = BATCH_SIZE as f32;
                        model.clip_grad_norm(&gpu, GRAD_CLIP_NORM * batch);
                        state.log_metric("grad_norm", model.grad_norm(&gpu) / batch);
                        model.step(&gpu, &opt);
                    }
                    model.step_count = state.step;
                    if let Some(c) = capture.as_mut() {
                        c.after_window(&gpu);
                    }

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
                        if let Some(line) = audio.as_mut().and_then(AudioPretrain::report) {
                            println!("{line}");
                        }
                        tokens_since_print = 0;
                        time = Instant::now();
                    }
                    if state.save() {
                        if let Some(a) = audio.as_mut() {
                            a.save(&gpu);
                        }
                        match model.save(&gpu, state.save_path()) {
                            Ok(()) => {
                                // Flush the log only now, so it never reflects a step
                                // past the checkpoint just written.
                                state.flush_log();
                                println!("saved -> {}", state.save_path());
                                if let Some((bpb, word)) = state.take_bpb() {
                                    println!("  bpb {bpb:.3} | word loss {word:.3}");
                                }
                                println!("  trained on: {}", model.seen.save_line());
                                // Written after the weights, so the recorded position
                                // never runs ahead of the checkpoint it describes.
                                progress.step = state.step;
                                if let Err(e) =
                                    pretrain_progress::save(state.save_path(), &progress)
                                {
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

    if let Some(a) = audio.as_mut() {
        a.save(&gpu);
    }
    match model.save(&gpu, state.save_path()) {
        Ok(()) => {
            state.flush_log();
            println!("final save -> {}", state.save_path());
            if let Some((bpb, word)) = state.take_bpb() {
                println!("  bpb {bpb:.3} | word loss {word:.3}");
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
            model.set_doc_starts(&batch.doc_starts);
            c_total += model.eval_loss(&gpu, &tokens, words);
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
    println!(
        "\n── validation ({windows} windows, {:.0?}) ──",
        start.elapsed()
    );
    println!("  Char loss = {c_loss:.3} nats   ppl = {:.3}", c_loss.exp());
    println!("  Word loss = {w_loss:.3} nats   ppl = {:.3}", w_loss.exp());
}

/// Rewrite an `NNM1` checkpoint as v2 at `<model>.v2`, leaving the original in place.
///
/// The written file is read back tensor by tensor and compared bitwise against the
/// device copy it came from, then loaded as a model.
pub fn convert_checkpoint(model_path: &str) {
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No CUDA GPU available ({e}).");
            return;
        }
    };
    match checkpoint::is_v2(model_path) {
        Ok(true) => {
            println!("'{model_path}' is already a v2 checkpoint.");
            return;
        }
        Ok(false) => {}
        Err(e) => {
            eprintln!("Could not open '{model_path}': {e}");
            std::process::exit(2);
        }
    }
    let w_token = Utf8Tokenizer::new().w_token() as usize;
    let mut model = match Hierarchical::load(&gpu, model_path, w_token) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Could not load '{model_path}': {e}");
            std::process::exit(2);
        }
    };
    let out = format!("{model_path}.v2");
    if let Err(e) = model.save(&gpu, &out) {
        eprintln!("Could not write '{out}': {e}");
        std::process::exit(2);
    }

    let mut verify = || -> std::io::Result<usize> {
        let mut r = checkpoint::Reader::open(&out)?;
        let bad = |m: String| std::io::Error::new(std::io::ErrorKind::InvalidData, m);
        if r.header.cfg != model.cfg
            || r.header.step != model.step_count as u64
            || r.header.seen != model.seen
        {
            return Err(bad("header differs from the loaded model".into()));
        }
        let mut buf = Vec::new();
        let mut params = 0;
        for (i, slot) in model.param_slots().into_iter().enumerate() {
            let want = slot.param.to_host(&gpu);
            r.next_into(want.dims(), &mut buf)?;
            if buf.iter().zip(&want.data).any(|(a, b)| a.to_bits() != b.to_bits()) {
                return Err(bad(format!("tensor {i} differs")));
            }
            params += buf.len();
        }
        r.finish()?;
        Ok(params)
    };
    let params = match verify() {
        Ok(params) => params,
        Err(e) => {
            eprintln!("'{out}' does not match the loaded model: {e}");
            let _ = std::fs::remove_file(&out);
            std::process::exit(2);
        }
    };
    let (cfg, step) = (model.cfg, model.step_count);
    drop(model);

    // The v2 loader rebuilds the layers from the config alone, so a checkpoint whose
    // layer pattern the config does not describe is written fine and cannot be read.
    if let Err(e) = Hierarchical::load(&gpu, &out, w_token) {
        eprintln!("'{out}' does not load back ({e}); removed it.");
        let _ = std::fs::remove_file(&out);
        std::process::exit(2);
    }
    println!("Wrote '{out}': {cfg:?}, step {step}, {params} parameters, verified bit-exact.");
}

/// GPU supervised fine-tuning (Q-A instruction tuning) of a pretrained
/// hierarchical model.
///
/// Loads the SFT set (`config::SFT_DATA`) fully into memory as masked chat
/// windows (see `crate::sft`) — the corpus is small — and fine-tunes the model
/// with loss counted only on the response tokens (`forward_backward_masked`).
///
/// A checkpoint with a smaller vocab than the tokenizer is rejected: its tied
/// table has no rows for the SFT markers.
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
             {vocab}: it predates the SFT markers and cannot be fine-tuned.",
            model.cfg.vocab
        );
        return;
    }

    let mut audio = SftAudio::open(&gpu, model_path, model.cfg.hc);
    let heard = |id: &str| audio.as_ref().and_then(|a| a.heard_tokens(id));
    // `SFT_DATA` in the environment overrides the configured set.
    let data = std::env::var("SFT_DATA").unwrap_or_else(|_| SFT_DATA.to_string());
    let mut examples = match sft::load_jsonl_heard(&tokenizer, &data, SFT_MAX_TOKENS, &heard) {
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

    let spoken = examples.iter().filter(|e| !e.heard.is_empty()).count();
    if let Some(a) = audio.as_mut() {
        println!(
            "SFT: {spoken} of {} examples have a heard user turn ({:.1}%)",
            examples.len(),
            100.0 * spoken as f64 / examples.len() as f64
        );
        a.reserve(&gpu, examples.iter().map(|e| e.heard.len()).max().unwrap_or(0));
    }

    // Size the caches to the longest example (word count + token span).
    let max_words = examples.iter().map(|e| e.words.len()).max().unwrap();
    let max_tokens = examples.iter().map(|e| e.tokens.len()).max().unwrap();
    // The GPU model builds its own per-window rectangles, so no persistent cache
    // allocation is needed here (unlike the CPU path); max_* only informs logs.
    println!(
        "SFT: {} examples, longest {max_words} words / {max_tokens} tokens. \
         {SFT_EPOCHS} epochs, LR={SFT_LR} over {SFT_WARMUP_WINDOWS} warmup windows, \
         batch={SFT_BATCH_SIZE} windows.",
        examples.len()
    );

    let mut state = TrainingState::from_step(model.step_count);
    state.lr = SFT_LR;
    state.batch_size = SFT_BATCH_SIZE;
    state.init_log(&format!("{model_path}_sft"), &["resp_ppl"]);
    state.set_defer_log_flush(true);

    let mut opt = AdamCfg::new(SFT_LR, crate::optimizers::WEIGHT_DECAY);

    // Where a previous, interrupted run stopped (or a fresh start). The shuffle
    // is seeded from this so resuming replays the exact same permutation and
    // skipping `done` examples really skips the ones already trained on.
    let mut progress =
        sft_progress::resume_or_fresh(model_path, examples.len(), model.step_count, SFT_EPOCHS);
    // Warmup counts from where this fine-tune began, not from the pretraining
    // step the checkpoint carries; a resume continues the same ramp.
    state.schedule = crate::training::LrSchedule::sft(progress.origin);
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

            let loss = match audio.as_mut().filter(|_| !ex.heard.is_empty()) {
                Some(a) => a.example(&gpu, &mut model, &tokens, words, &word_loss, &ex.heard),
                None => model.forward_backward_masked(&gpu, &tokens, words, &word_loss),
            };
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
                    "{} | resp loss {:.3} | ppl {:.3} | lr {:.2e} | {} tok | {:.1?}",
                    state.step,
                    loss,
                    loss.exp(),
                    opt.lr,
                    tokens_since_print,
                    time.elapsed(),
                );
                if let Some(line) = audio.as_mut().and_then(|a| a.report()) {
                    println!("{line}");
                }
                tokens_since_print = 0;
                time = Instant::now();
            }
            if state.save() {
                if let Some(a) = audio.as_mut() {
                    a.save(&gpu);
                }
                match model.save(&gpu, model_path) {
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
        if let Some(a) = audio.as_mut() {
            a.save(&gpu);
        }
        if let Err(e) = model.save(&gpu, model_path) {
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

    if let Some(a) = audio.as_mut() {
        a.save(&gpu);
    }
    match model.save(&gpu, model_path) {
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
