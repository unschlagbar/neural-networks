//! Streams real windows from the corpus through the GPU hierarchical trainer,
//! exactly as `train_hierarchical_gpu` does — same dataset, same forward_backward,
//! same optimizer cadence — but on a fresh model and writing no checkpoint, so it
//! can be run against a live training setup without touching `models/`.
//!
//!   cargo run --release --features cuda --example gpu_soak -- <corpus> [windows]
//!
//! It exists to reproduce a device-side fault that only real data triggers: the
//! window's word count `dw` varies (windows never cross document borders, so every
//! short document yields a short window), and the sLSTM's captured CUDA graphs are
//! bound to the buffers they were captured against. A window shape that reallocates
//! those buffers while a graph still refers to them is a use-after-free on the
//! device, which surfaces — asynchronously, and possibly much later — as a sticky
//! CUBLAS_STATUS_EXECUTION_FAILED.
//!
//! Prints the distribution of window shapes it actually exercised, so a clean run
//! is evidence about the shapes it saw rather than a bare "it didn't crash".

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda` to run this benchmark");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::rc::Rc;
    use std::time::Instant;

    use neural_networks::batches::ChunkedWordDataSet;
    use neural_networks::config::*;
    use neural_networks::gpu::Gpu;
    use neural_networks::gpu::hierarchical::{HierCfg, Hierarchical};
    use neural_networks::nn2::optim::AdamCfg;
    use neural_networks::tokenizer_utf8::Utf8Tokenizer;

    let path = std::env::args().nth(1).expect("usage: gpu_soak <corpus> [windows]");
    let limit: usize = std::env::args().nth(3 - 1).map_or(400, |s| s.parse().unwrap_or(400));

    let gpu = Gpu::new().expect("no GPU");
    let tok = Rc::new(Utf8Tokenizer::new());
    let vocab = tok.vocab_size();
    let w_token = tok.w_token() as usize;
    let heads = 8;
    let cfg = HierCfg {
        vocab,
        hc: CHAR_HIDDEN,
        wh: WORD_HIDDEN,
        enc_blocks: 2,
        bb_blocks: WORD_BLOCKS,
        dec_blocks: 2,
        heads,
        dqk: WORD_HIDDEN / heads,
        w_token,
        cap: LOGIT_SOFTCAP,
    };
    let mut model = Hierarchical::new(&gpu, &cfg);
    let mut opt = AdamCfg::new(LR, neural_networks::optimizers::WEIGHT_DECAY);

    let mut data = ChunkedWordDataSet::open(
        tok.clone(), &path, WORDS_PER_SEQ, MIN_WORDS_PER_SEQ, MAX_WINDOW_TOKENS, CHUNK_BYTES,
    );

    println!("soaking {limit} windows from '{path}' ...");
    let t0 = Instant::now();
    let mut seen = 0usize;
    // How many windows fell in each backbone-unroll band. The short ones are the
    // interesting ones: below 32 the sLSTM runs eagerly and refits its buffers
    // without ever consulting the graph cache.
    let (mut tiny, mut short, mut full) = (0usize, 0usize, 0usize);
    let mut loss_sum = 0.0;

    'outer: while let Some(chunk) = data.next_chunk() {
        for batch in chunk.iter() {
            let tokens: Vec<usize> = batch.tokens.iter().map(|&t| t as usize).collect();
            let words: Vec<(usize, usize)> =
                batch.words.iter().map(|r| (r.start, r.end)).collect();
            if words.len() < 2 {
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

            loss_sum += model.forward_backward(&gpu, &tokens, &words);
            seen += 1;
            if seen % BATCH_SIZE == 0 {
                opt.t += 1;
                model.step(&gpu, &opt);
            }
            if seen % 25 == 0 {
                println!(
                    "  {seen:>5} windows | dw<32: {tiny}, 32..full: {short}, full: {full} \
                     | mean loss {:.4} | {:.1?}",
                    loss_sum / seen as f32,
                    t0.elapsed(),
                );
            }
            if seen >= limit {
                break 'outer;
            }
        }
    }

    println!(
        "\nOK — {seen} windows, no device fault.\n\
         shapes exercised: dw<32 (eager path) {tiny}, 32..full {short}, full {full}\n\
         {:.3} s/window",
        t0.elapsed().as_secs_f64() / seen.max(1) as f64,
    );
}
