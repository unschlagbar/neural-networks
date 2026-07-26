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

use rand::seq::SliceRandom;

use crate::batches::ChunkedWordDataSet;
use crate::config::{
    BATCH_SIZE, CHAR_HIDDEN, CHUNK_BYTES, EPOCHS, LOG_EVERY, LOGIT_SOFTCAP, LR, MAX_WINDOW_TOKENS,
    MIN_WORDS_PER_SEQ, SFT_DATA, SFT_EPOCHS, SFT_LR, TRAIN_DATA, WORD_BLOCKS, WORD_HIDDEN,
    WORDS_PER_SEQ,
};
use crate::gpu::Gpu;
use crate::gpu::hierarchical::{HierCfg, Hierarchical};
use crate::nn2::optim::AdamCfg;
use crate::sft;
use crate::tokenizer_utf8::Utf8Tokenizer;
use crate::training::TrainingState;

/// Architecture, taken from `config.rs` so the GPU model matches the CPU one.
/// `heads`/`dqk` mirror `model.rs::build_hierarchical_model`.
fn cfg_from_config(vocab: usize, w_token: usize) -> HierCfg {
    let heads = 8;
    HierCfg {
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
            m
        }
        Err(e) => {
            println!("Could not load '{model_path}' ({e}) — creating new GPU model.");
            Hierarchical::new(&gpu, &cfg)
        }
    };
    if model.cfg != cfg {
        eprintln!(
            "WARNING: checkpoint architecture {:?} differs from config.rs {:?} — \
             continuing with the checkpoint's.",
            model.cfg, cfg
        );
    }

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
        "Training on GPU: {EPOCHS} epochs, LR={LR}, batch={BATCH_SIZE} windows, \
         log every {LOG_EVERY} steps"
    );

    let mut state = TrainingState::from_step(model.step_count);
    state.init_log(model_path, &[]);
    // Buffer CSV rows and only flush them when the model is saved (every
    // SAVE_EVERY steps), so the on-disk log never gets ahead of the checkpoint.
    state.set_defer_log_flush(true);

    let mut opt = AdamCfg::new(LR, crate::optimizers::WEIGHT_DECAY);

    // Resume needs the total window count for the modulo — one cheap counting pass.
    let resume_windows = if model.step_count > 0 {
        let t0 = Instant::now();
        let total = data.count_windows();
        println!(
            "  {total} windows total (counting pass took {:.1?})",
            t0.elapsed()
        );
        model.step_count % total.max(1)
    } else {
        0
    };

    for epoch in 1..=EPOCHS {
        println!("── Epoch {epoch} ───────────────────────────────────────");
        let mut skip = if epoch == 1 { resume_windows } else { 0 };
        if skip > 0 {
            println!("  Resuming from window {skip} (step {})", model.step_count);
        }

        let epoch_start = Instant::now();
        let mut tokens_since_print = 0usize;
        let mut time = Instant::now();
        data.rewind();

        while let Some(chunk) = data.next_chunk() {
            if skip >= chunk.len() {
                skip -= chunk.len();
                continue;
            }
            for batch in chunk.iter().skip(skip) {
                // The dataset speaks u16 / Range; the model takes usize / (start, end).
                let tokens: Vec<usize> = batch.tokens.iter().map(|&t| t as usize).collect();
                let words: Vec<(usize, usize)> =
                    batch.words.iter().map(|r| (r.start, r.end)).collect();
                if words.len() < 2 {
                    continue; // no decoded word in this window
                }

                let loss = model.forward_backward(&gpu, &tokens, &words);
                tokens_since_print += tokens.len();
                state.log_tokens(tokens.len());

                // `state.step` returns Some(lr) only on a batch boundary, so grads
                // accumulate over BATCH_SIZE windows before each optimizer step.
                if let Some(lr) = state.step(loss) {
                    opt.lr = lr;
                    opt.t += 1;
                    model.step(&gpu, &opt);
                }
                model.step_count = state.step;

                if state.print() {
                    let loss = state.get_loss();
                    println!(
                        "{} | char loss {:.4} | ppl {:.4} | lr {:.2e} | {} tok | {:.1?}",
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
                    match model.save(&gpu, state.save_path(), &[]) {
                        Ok(()) => {
                            // Flush the log only now, so it never reflects a step
                            // past the checkpoint just written.
                            state.flush_log();
                            println!("saved -> {}", state.save_path());
                        }
                        Err(e) => eprintln!("save failed: {e}"),
                    }
                }
            }
            skip = 0;
        }
        println!("Epoch {epoch} took {:.1?}", epoch_start.elapsed());
    }

    match model.save(&gpu, state.save_path(), &[]) {
        Ok(()) => {
            state.flush_log();
            println!("final save -> {}", state.save_path());
        }
        Err(e) => eprintln!("final save failed: {e}"),
    }
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

    let mut examples = match sft::load_jsonl(&tokenizer, SFT_DATA) {
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

    let mut rng = rand::rng();
    for epoch in 1..=SFT_EPOCHS {
        println!("── SFT epoch {epoch}/{SFT_EPOCHS} ──");
        examples.shuffle(&mut rng);

        let epoch_start = Instant::now();
        let mut tokens_since_print = 0usize;
        let mut time = Instant::now();

        for ex in examples.iter() {
            let tokens: Vec<usize> = ex.tokens.iter().map(|&t| t as usize).collect();
            let words: Vec<(usize, usize)> = ex.words.iter().map(|r| (r.start, r.end)).collect();
            if words.len() < 2 {
                continue;
            }
            // `word_loss` is per DECODED word: word w decodes words[w+1], so its
            // flag is ex.loss[w+1] (ex.loss[0] belongs to the encode-only prefix).
            let word_loss: Vec<bool> = ex.loss[1..].to_vec();

            let loss = model.forward_backward_masked(&gpu, &tokens, &words, &word_loss);
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
                        println!("saved -> {model_path}");
                    }
                    Err(e) => eprintln!("save failed: {e}"),
                }
            }
        }
        println!("SFT epoch {epoch} took {:.1?}", epoch_start.elapsed());
        // Save at every epoch boundary too.
        if let Err(e) = model.save(&gpu, model_path, &[]) {
            eprintln!("epoch save failed: {e}");
        } else {
            state.flush_log();
        }
    }

    match model.save(&gpu, model_path, &[]) {
        Ok(()) => {
            state.flush_log();
            println!("final SFT save -> {model_path}");
        }
        Err(e) => eprintln!("final save failed: {e}"),
    }
}
