//! Streams real windows from the corpus through the GPU hierarchical trainer,
//! exactly as `train_hierarchical_gpu` does — same dataset, same forward_backward,
//! same optimizer cadence — but on a fresh model and writing no checkpoint, so it
//! can be run against a live training setup without touching `models/`.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda` to run this benchmark");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;

    use neural_networks::batches::ChunkedWordDataSet;
    use neural_networks::config::*;
    use neural_networks::gpu::Gpu;
    use neural_networks::gpu::hierarchical::{Hierarchical, ModelCfg};
    use neural_networks::cpu::optim::AdamCfg;
    use neural_networks::tokenizer_utf8::Utf8Tokenizer;

    let path = std::env::args()
        .nth(1)
        .expect("usage: gpu_soak <corpus> [windows]");
    let limit: usize = std::env::args()
        .nth(3 - 1)
        .map_or(400, |s| s.parse().unwrap_or(400));

    let gpu = Gpu::new().expect("no GPU");
    let tok = Utf8Tokenizer::new();
    let vocab = tok.vocab_size();
    let w_token = tok.w_token() as usize;
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
    let mut model = match std::env::var("SOAK_MODEL") {
        Ok(p) => Hierarchical::load(&gpu, &p, w_token).expect("load SOAK_MODEL"),
        Err(_) => Hierarchical::new(&gpu, cfg),
    };
    let mut skip: usize = std::env::var("SOAK_SKIP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let mut opt = AdamCfg::new(LR, neural_networks::config::BACKBONE_WEIGHT_DECAY);

    let mut data = ChunkedWordDataSet::open(
        tok,
        &path,
        WORDS_PER_SEQ,
        MIN_WORDS_PER_SEQ,
        MAX_WINDOW_TOKENS,
        CHUNK_BYTES,
    );

    println!("soaking {limit} windows from '{path}' ...");
    let t0 = Instant::now();
    let mut seen = 0usize;
    let (mut tiny, mut short, mut full) = (0usize, 0usize, 0usize);
    let mut loss_sum = 0.0;
    let warmup = BATCH_SIZE;
    let (mut hot, mut hot_tokens) = (None, 0usize);

    let stateful = CARRY_WINDOW_STATE && std::env::var("SOAK_STATELESS").is_err();
    model.set_stateful(stateful);

    'outer: while let Some(chunk) = data.next_chunk() {
        if skip >= chunk.len() {
            skip -= chunk.len();
            continue;
        }
        let mut ran_prev = false;
        for batch in chunk.iter().skip(std::mem::take(&mut skip)) {
            let tokens: Vec<usize> = batch.tokens.iter().map(|&t| t as usize).collect();
            let words = &batch.words;
            if words.len() < 2 {
                ran_prev = false;
                continue;
            }
            let dw = words.len() - 1;
            if dw < 32 {
                tiny += 1;
            } else if dw < WORDS_PER_SEQ - 1 {
                short += 1;
            } else {
                full += 1;
            }

            model.set_continues(batch.continues && ran_prev);
            model.set_doc_starts(&batch.doc_starts);
            ran_prev = true;
            loss_sum += model.forward_backward(&gpu, &tokens, words);
            seen += 1;
            if seen > warmup {
                hot.get_or_insert_with(Instant::now);
                hot_tokens += tokens.len();
            }
            if seen % BATCH_SIZE == 0 {
                opt.t += 1;
                model.step(&gpu, &opt);
            }
            if seen % 25 == 0 {
                println!(
                    "  {seen:>5} windows | dw<32: {tiny}, 32..full: {short}, full: {full} \
                     | mean loss {:.3} | {:.1?}",
                    loss_sum / seen as f32,
                    t0.elapsed(),
                );
            }
            if seen >= limit {
                break 'outer;
            }
        }
    }

    let hot_secs = hot.map_or(0.0, |t| t.elapsed().as_secs_f64());
    let hot_windows = seen.saturating_sub(warmup);
    println!(
        "\nOK — {seen} windows, no device fault.\n\
         shapes exercised: dw<32 (eager path) {tiny}, 32..full {short}, full {full}\n\
         {:.3} s/window over all {seen}\n\
         after warmup: {:.3} s/window, {:.0} tokens/s ({hot_windows} windows, \
         {hot_tokens} tokens)",
        t0.elapsed().as_secs_f64() / seen.max(1) as f64,
        hot_secs / hot_windows.max(1) as f64,
        hot_tokens as f64 / hot_secs.max(1e-9),
    );
}
