//! Two identical training runs, step for step — the loss trace must be bit-identical
//! and must go DOWN.
//!
//!   cargo run --release --features cuda --example train_determinism [words] [steps]
//!
//! `determinism.rs` repeats one forward/backward against frozen weights, so it only
//! sees the forward: a nondeterministic gradient leaves the loss it prints alone.
//! Here the model is stepped, so step k's loss depends on every gradient before it —
//! which is what turns a backward race into a visible divergence.
//!
//! Both runs start from the same weights via a saved checkpoint (`Hierarchical::new`
//! draws fresh random weights, so two constructions are not comparable).

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use neural_networks::config::{
        CHAR_HIDDEN, MAX_WINDOW_TOKENS, OUT_HIDDEN, WORD_BLOCKS, WORD_HIDDEN,
    };
    use neural_networks::gpu::Gpu;
    use neural_networks::gpu::hierarchical::{Hierarchical, ModelCfg};
    use neural_networks::nn2::optim::AdamCfg;
    use neural_networks::tokenizer_utf8::Utf8Tokenizer;

    let mut args = std::env::args().skip(1);
    let words: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);
    const GRAD_REPS: usize = 20;

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
    assert_eq!(CHAR_HIDDEN, OUT_HIDDEN, "decoder ties the encoder char table");

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
    println!("window: {} words, {} tokens, {steps} steps", spans.len(), tokens.len());

    let seed = std::env::temp_dir().join("train_determinism_seed.hier");
    let seed = seed.to_str().unwrap();
    Hierarchical::new(&gpu, cfg).save(&gpu, seed, &[]).expect("save seed");

    let run = || -> Vec<f32> {
        let mut model = Hierarchical::load(&gpu, seed, cfg.w_token).expect("load seed");
        let mut out = Vec::with_capacity(steps);
        for t in 1..=steps {
            let loss = model.forward_backward(&gpu, &tokens, &spans);
            let mut c = AdamCfg::new(3e-4, 0.01);
            c.t = t as u64;
            model.step(&gpu, &c);
            out.push(loss);
        }
        gpu.stream.synchronize().unwrap();
        out
    };

    // Which gradient tensors are unstable, before any of it is folded into a step.
    {
        // One model at a time: at the real window size two of them do not fit.
        let sig = || {
            let mut model = Hierarchical::load(&gpu, seed, cfg.w_token).expect("load seed");
            model.forward_backward(&gpu, &tokens, &spans);
            model.grad_signature(&gpu)
        };
        let s0 = sig();
        let mut bad: std::collections::BTreeSet<String> = Default::default();
        for _ in 0..GRAD_REPS {
            for ((n, h0), (_, h1)) in s0.iter().zip(sig()) {
                if *h0 != h1 {
                    bad.insert(n.clone());
                }
            }
        }
        if bad.is_empty() {
            println!("gradients: all {} tensors bit-identical over {GRAD_REPS} reps", s0.len());
        } else {
            let show: Vec<&str> = bad.iter().take(12).map(|s| s.as_str()).collect();
            println!("gradients: {}/{} unstable, e.g. {:?}", bad.len(), s0.len(), show);
        }
    }

    let a = run();
    let b = run();
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        let tag = if x.to_bits() == y.to_bits() { "" } else { "  <-- DIFFERS" };
        println!("  step {i}: {x:.9} / {y:.9}{tag}");
    }
    let same = a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits());
    let fell = a.last().is_some_and(|&l| l < a[0]);
    println!(
        "\n{}   loss {} ({:.6} -> {:.6})",
        if same { "DETERMINISTIC" } else { "NONDETERMINISTIC" },
        if fell { "fell" } else { "ROSE" },
        a[0],
        a[a.len() - 1],
    );
    if !same || !fell {
        std::process::exit(1);
    }
}
