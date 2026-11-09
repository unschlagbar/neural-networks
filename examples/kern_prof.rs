//! One model, one window shape, N steps — a clean target for `nsys` and `ncu`.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use neural_networks::config::{
        CHAR_HIDDEN, MAX_WINDOW_TOKENS, OUT_HIDDEN, WORD_BLOCKS, WORD_HIDDEN, WORDS_PER_SEQ,
    };
    use neural_networks::gpu::Gpu;
    use neural_networks::gpu::hierarchical::{Hierarchical, ModelCfg};
    use neural_networks::cpu::optim::AdamCfg;
    use neural_networks::tokenizer_utf8::Utf8Tokenizer;

    let mut args = std::env::args().skip(1);
    let words: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(WORDS_PER_SEQ);
    let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);
    let warmup: usize = std::env::var("PROF_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    let gpu = Gpu::new().expect("gpu");
    let tok = Utf8Tokenizer::new();
    let cfg = ModelCfg {
        vocab: tok.vocab_size(),
        hc: CHAR_HIDDEN,
        wh: WORD_HIDDEN,
        enc_blocks: 4,
        bb_blocks: WORD_BLOCKS,
        dec_blocks: 4,
        heads: 8,
        dqk: WORD_HIDDEN / 8,
        w_token: neural_networks::tokenizer_utf8::W_TOKEN as usize,
        cap: 30.0,
    };
    assert_eq!(
        CHAR_HIDDEN, OUT_HIDDEN,
        "decoder ties the encoder char table"
    );

    let mut tokens = Vec::with_capacity(words * 6);
    let mut spans: Vec<std::range::Range<usize>> = Vec::with_capacity(words);
    for w in 0..words {
        let s = tokens.len();
        let r = (w * 2654435761) % 100;
        let want = match r {
            0..=24 => 1 + (w % 3),
            25..=59 => 3 + (w % 4),
            60..=84 => 6 + (w % 5),
            _ => 10 + (w % 7),
        };
        if s + want > MAX_WINDOW_TOKENS {
            break;
        }
        for k in 0..want {
            tokens.push(1 + (k % 90));
        }
        spans.push((s..tokens.len()).into());
    }
    println!("window: {} words, {} tokens", spans.len(), tokens.len());

    let mut model = Hierarchical::new(&gpu, cfg);
    let mut opt = AdamCfg::new(3e-4, 0.01);

    for _ in 0..warmup {
        model.forward_backward(&gpu, &tokens, &spans);
        opt.t += 1;
        model.step(&gpu, &opt);
    }
    gpu.stream.synchronize().unwrap();

    let t0 = std::time::Instant::now();
    for _ in 0..steps {
        model.forward_backward(&gpu, &tokens, &spans);
        opt.t += 1;
        model.step(&gpu, &opt);
    }
    gpu.stream.synchronize().unwrap();
    let el = t0.elapsed().as_secs_f64();
    println!(
        "{steps} steps in {:.3} s -> {:.1} ms/step",
        el,
        el * 1000.0 / steps as f64
    );
    neural_networks::gpu::gtensor::ptr_probe::dump(warmup + steps);
}
