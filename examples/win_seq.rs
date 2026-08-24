//! Replay a sequence of differently-sized windows through one model, the way a real
//! corpus does.
//!
//!   cargo run --release --features cuda --example win_seq -- <checkpoint> w1 w2 w3 ...
//!
//! `grad_dump` runs one window per process, so anything a window leaves behind for the
//! next one — recurrent state, a carry flag, an unpopped activation stack — is invisible
//! to it. Document-bounded windows vary in length, and a window short enough for a
//! single backbone chunk takes a different path through the sweep than a long one.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use neural_networks::config::MAX_WINDOW_TOKENS;
    use neural_networks::gpu::Gpu;
    use neural_networks::gpu::hierarchical::Hierarchical;
    use neural_networks::tokenizer_utf8::W_TOKEN;

    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: win_seq <checkpoint> w1 w2 ...");
    let sizes: Vec<usize> = args.filter_map(|s| s.parse().ok()).collect();
    assert!(!sizes.is_empty(), "give at least one window size");

    let uniform: Option<usize> = std::env::var("UNIFORM").ok().and_then(|v| v.parse().ok());
    let gpu = Gpu::new().expect("gpu");
    let mut model = Hierarchical::load(&gpu, &path, W_TOKEN as usize).expect("load");

    for (i, &words) in sizes.iter().enumerate() {
        let mut tokens = Vec::with_capacity(words * 6);
        let mut spans: Vec<std::range::Range<usize>> = Vec::with_capacity(words);
        for w in 0..words {
            let s = tokens.len();
            // `UNIFORM=n` gives every word the same length, so the encoder/decoder length
            // buckets hold one shape and no group rectangle needs padding rows. Mixed
            // lengths (the default, and what a real corpus gives) do pad.
            let want = match uniform {
                Some(n) => n,
                None => match (w * 2654435761) % 100 {
                    0..=24 => 1 + (w % 3),
                    25..=59 => 3 + (w % 4),
                    60..=84 => 6 + (w % 5),
                    _ => 10 + (w % 7),
                },
            };
            if s + want > MAX_WINDOW_TOKENS {
                break;
            }
            for k in 0..want {
                tokens.push(1 + (w * 7 + k * 13) % 90);
            }
            spans.push((s..tokens.len()).into());
        }
        let loss = model.forward_backward(&gpu, &tokens, &spans);
        println!(
            "[{i}] {:>6} words, {:>7} tokens -> loss {loss:.6}",
            spans.len(),
            tokens.len()
        );
    }
    // Accumulated gradient norms after the whole sequence. Grads are never stepped, so
    // two runs over the same sizes are comparing the same quantity.
    for stage in ["encoder", "backbone", "decoder"] {
        for (i, n) in model.grad_norms_by_block(&gpu, stage).iter().enumerate() {
            println!("g {stage}[{i}] {n:.7e}");
        }
    }
    println!("OK");
}
