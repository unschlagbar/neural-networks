//! Same window, same weights, N times — the loss must be bit-identical.
//!
//! Catches shared-memory and warp-shuffle races that unit tests miss: a stale
//! `__shared__` slot read by a partial final warp is usually zero in a small
//! isolated launch and only turns non-zero once many blocks have cycled through
//! an SM. That makes it a whole-model, many-block property, not a kernel one.
//!
//! Run: cargo run --release --features cuda --example determinism [words] [reps]

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
    use neural_networks::gpu::hierarchical::{HierCfg, Hierarchical};
    use neural_networks::tokenizer_utf8::Utf8Tokenizer;

    let mut args = std::env::args().skip(1);
    let words: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let reps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);

    let gpu = Gpu::new().expect("gpu");
    let tok = Utf8Tokenizer::new();
    let cfg = HierCfg {
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
    println!("window: {} words, {} tokens", spans.len(), tokens.len());

    // One model, never stepped: every repetition sees identical weights, so any
    // spread in the loss is the forward/backward itself being nondeterministic.
    let mut model = Hierarchical::new(&gpu, &cfg);

    let mut losses = Vec::with_capacity(reps);
    for i in 0..reps {
        let loss = model.forward_backward(&gpu, &tokens, &spans);
        gpu.stream.synchronize().unwrap();
        println!("  rep {i}: loss {loss:.9}  bits {:#018x}", loss.to_bits());
        losses.push(loss);
    }

    let first = losses[0];
    let worst = losses.iter().fold(0.0f32, |a, &l| a.max((l - first).abs()));
    if losses.iter().all(|&l| l.to_bits() == first.to_bits()) {
        println!("\nDETERMINISTIC: all {reps} reps bit-identical");
    } else {
        println!("\nNONDETERMINISTIC: max deviation {worst:.3e} over {reps} reps");
        std::process::exit(1);
    }
}
