//! Gradient reproducibility across *different* window shapes, per stage.
//!
//!   cargo run --release --features cuda --example window_determinism [reps] [full]
//!
//! `train_determinism.rs` repeats one window. A window's shape decides how the
//! encoder and decoder bucket their words (one group per length, split at the row
//! cap) and how many chunks the backbone sweep runs, so a race that only fires at
//! a particular group geometry is invisible there. Here several windows with
//! deliberately different shapes are each hashed over `reps` fresh
//! forward/backwards, the unstable tensors reported per stage, and then the whole
//! sequence is trained twice and the loss traces compared.
//!
//! The geometry that matters is reached with `set_bb_chunk` / `set_group_cap`
//! rather than with word count: a 96-word window under a 16-row cap splits its
//! buckets exactly like a 4096-word one under the real cap, in a fraction of the
//! time. Pass `full` for the production widths and window sizes instead.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use neural_networks::config::{
        BACKBONE_CHUNK, CHAR_HIDDEN, MAX_WINDOW_TOKENS, MAX_WORD_BYTES, OUT_HIDDEN, WORD_BLOCKS,
        WORD_HIDDEN,
    };
    use neural_networks::gpu::Gpu;
    use neural_networks::gpu::hierarchical::{Hierarchical, ModelCfg};
    use neural_networks::nn2::optim::AdamCfg;
    use neural_networks::tokenizer_utf8::Utf8Tokenizer;
    use std::collections::BTreeSet;
    use std::range::Range;

    let mut args = std::env::args().skip(1);
    let reps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(20);
    let full = args.next().is_some_and(|s| s == "full");

    let gpu = Gpu::new().expect("gpu");
    let tok = Utf8Tokenizer::new();
    assert_eq!(
        CHAR_HIDDEN, OUT_HIDDEN,
        "decoder ties the encoder char table"
    );
    // Encoder/decoder width and dqk are kept at the production values — those pick
    // which mLSTM path a stage takes. The backbone is narrowed and shortened, so
    // its sLSTM geometry (and with it the fused-time decision) is NOT production's.
    let cfg = ModelCfg {
        vocab: tok.vocab_size(),
        hc: CHAR_HIDDEN,
        wh: if full { WORD_HIDDEN } else { 256 },
        enc_blocks: if full { 4 } else { 2 },
        bb_blocks: if full { WORD_BLOCKS } else { 4 }, // >= 4: sLSTM every 4th
        dec_blocks: if full { 4 } else { 2 },
        heads: if full { 8 } else { 2 },
        dqk: 128,
        w_token: neural_networks::tokenizer_utf8::W_TOKEN as usize,
        cap: 30.0,
    };

    // (name, words, token count of word w, backbone chunk, group row cap)
    type Len = fn(usize) -> usize;
    type Shape = (&'static str, usize, Len, Option<usize>, Option<usize>);
    let ragged: Len = |w| 1 + (w * 2654435761 >> 13) % MAX_WORD_BYTES;
    let shapes: Vec<Shape> = if full {
        vec![
            ("uniform-short", BACKBONE_CHUNK / 2, |_| 3, None, None),
            (
                "uniform-max",
                BACKBONE_CHUNK / 2,
                |_| MAX_WORD_BYTES,
                None,
                None,
            ),
            (
                "fanned",
                BACKBONE_CHUNK,
                |w| 1 + w % MAX_WORD_BYTES,
                None,
                None,
            ),
            ("ragged-odd", BACKBONE_CHUNK * 2 + 37, ragged, None, None),
            (
                "bimodal-wide",
                4400,
                |w| if w % 2 == 0 { 2 } else { 11 },
                None,
                None,
            ),
        ]
    } else {
        vec![
            // one bucket, one chunk: the plain case
            ("uniform-short", 48, |_| 3, None, None),
            // one bucket, deepest decoder unroll
            ("uniform-max", 48, |_| MAX_WORD_BYTES, None, None),
            // one bucket per length: many groups, few rows each
            ("fanned", 64, |w| 1 + w % MAX_WORD_BYTES, None, None),
            // ragged, and a word count that is not a multiple of the chunk
            ("ragged-chunked", 133, ragged, Some(32), None),
            // two heavy buckets, each split into sub-groups by the row cap
            (
                "bimodal-split",
                96,
                |w| if w % 2 == 0 { 2 } else { 11 },
                Some(64),
                Some(32),
            ),
        ]
    };

    let build = |words: usize, len: Len| -> (Vec<usize>, Vec<Range<usize>>) {
        let mut tokens = Vec::new();
        let mut spans = Vec::new();
        for w in 0..words {
            let s = tokens.len();
            let n = len(w).clamp(1, MAX_WORD_BYTES);
            if s + n > MAX_WINDOW_TOKENS {
                break;
            }
            for k in 0..n {
                tokens.push(1 + (w + k) % 90);
            }
            spans.push((s..tokens.len()).into());
        }
        (tokens, spans)
    };

    let windows: Vec<(
        &str,
        Vec<usize>,
        Vec<Range<usize>>,
        Option<usize>,
        Option<usize>,
    )> = shapes
        .iter()
        .map(|(name, words, len, chunk, cap)| {
            let (t, s) = build(*words, *len);
            (*name, t, s, *chunk, *cap)
        })
        .collect();

    let seed = std::env::temp_dir().join("window_determinism_seed.hier");
    let seed = seed.to_str().unwrap();
    Hierarchical::new(&gpu, cfg)
        .save(&gpu, seed, &[])
        .expect("save seed");
    let fresh = |chunk, cap| {
        let mut m = Hierarchical::load(&gpu, seed, cfg.w_token).expect("load seed");
        m.set_bb_chunk(chunk);
        m.set_group_cap(cap);
        m
    };

    // One model for every rep. Grads accumulate, so a rep has to start from zero;
    // a step at lr 0 zeroes them and leaves the weights alone, which reloading the
    // checkpoint (~110 ms) only achieved the slow way. Checked, not assumed:
    let mut model = fresh(None, None);
    let clear = {
        let mut c = AdamCfg::new(0.0, 0.0);
        c.t = 1;
        c
    };
    {
        let (_, tokens, spans, chunk, cap) = &windows[0];
        model.set_bb_chunk(*chunk);
        model.set_group_cap(*cap);
        model.forward_backward(&gpu, tokens, spans);
        model.step(&gpu, &clear);
        model.forward_backward(&gpu, tokens, spans);
        let reused = model.grad_signature(&gpu);
        model.step(&gpu, &clear);
        let mut m = fresh(*chunk, *cap);
        m.forward_backward(&gpu, tokens, spans);
        if reused != m.grad_signature(&gpu) {
            println!("WARNING: an lr-0 step is not a clean reset - everything below is suspect\n");
        }
    }

    println!(
        "reps {reps}, {} model\n",
        if full { "production" } else { "small" }
    );
    let mut any_bad = false;

    for (name, tokens, spans, chunk, cap) in &windows {
        model.set_bb_chunk(*chunk);
        model.set_group_cap(*cap);
        let mut sig = || {
            model.forward_backward(&gpu, tokens, spans);
            let s = model.grad_signature(&gpu);
            model.step(&gpu, &clear); // lr 0: zeroes the grads, leaves the weights
            s
        };
        let s0 = sig();
        let mut bad: BTreeSet<String> = Default::default();
        for _ in 1..reps {
            for ((n, h0), (_, h1)) in s0.iter().zip(sig()) {
                if *h0 != h1 {
                    bad.insert(n.clone());
                }
            }
        }
        let count = |p: &str| bad.iter().filter(|n| n.starts_with(p)).count();
        let of = |p: &str| s0.iter().filter(|(n, _)| n.starts_with(p)).count();
        println!(
            "{name:<15}{:>5} words {:>6} tokens  unstable {:>3}/{:<4} enc {}/{} bb {}/{} dec {}/{}",
            spans.len(),
            tokens.len(),
            bad.len(),
            s0.len(),
            count("enc"),
            of("enc"),
            count("bb"),
            of("bb"),
            count("dec"),
            of("dec"),
        );
        if !bad.is_empty() {
            let show: Vec<&str> = bad.iter().take(8).map(|s| s.as_str()).collect();
            println!("{:<15}  e.g. {show:?}", "");
            any_bad = true;
        }
    }

    // The windows in sequence, stepping between them: the training loop's own path,
    // where a window also inherits the backbone state of the one before.
    let run = || -> Vec<f32> {
        let mut m = fresh(None, None);
        let mut out = Vec::new();
        for (t, (_, tokens, spans, chunk, cap)) in windows.iter().enumerate() {
            m.set_bb_chunk(*chunk);
            m.set_group_cap(*cap);
            let loss = m.forward_backward(&gpu, tokens, spans);
            let mut c = AdamCfg::new(3e-4, 0.01);
            c.t = t as u64 + 1;
            m.step(&gpu, &c);
            out.push(loss);
        }
        gpu.stream.synchronize().unwrap();
        out
    };
    let a = run();
    let b = run();
    println!();
    for (i, ((x, y), w)) in a.iter().zip(&b).zip(&windows).enumerate() {
        let tag = if x.to_bits() == y.to_bits() {
            ""
        } else {
            "  <-- DIFFERS"
        };
        println!("  step {i} {:<15}{x:.9} / {y:.9}{tag}", w.0);
    }
    let same = a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits());
    println!(
        "\n{}",
        if same && !any_bad {
            "DETERMINISTIC"
        } else if same {
            "NONDETERMINISTIC (gradients move; the traces happened to agree)"
        } else {
            "NONDETERMINISTIC"
        }
    );
    if !same || any_bad {
        std::process::exit(1);
    }
}
