//! One forward/backward of a saved checkpoint on a fixed window, dumping the loss and
//! every block's gradient norm. Run it on two source trees and diff the output to see
//! whether a change moved the gradients.
//!
//!   cargo run --release --features cuda --example grad_dump -- <checkpoint> [words]
//!
//! Only `load` / `forward_backward` / `grad_norms_by_block` are touched, so this also
//! builds against older trees — which is the point: the baseline for "did a kernel
//! change move the gradients" is the tree that last trained without trouble, not the
//! previous commit. `grad_seed` writes the checkpoint.

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
    let path = args.next().expect("usage: grad_dump <checkpoint> [words]");
    let words: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4096);

    let gpu = Gpu::new().expect("gpu");

    // Word lengths spread over the buckets the segmenter actually produces, so the
    // encoder and decoder see a realistic mix of group widths rather than one length.
    let mut tokens = Vec::with_capacity(words * 6);
    let mut spans: Vec<std::range::Range<usize>> = Vec::with_capacity(words);
    for w in 0..words {
        let s = tokens.len();
        let want = match (w * 2654435761) % 100 {
            0..=24 => 1 + (w % 3),
            25..=59 => 3 + (w % 4),
            60..=84 => 6 + (w % 5),
            _ => 10 + (w % 7),
        };
        if s + want > MAX_WINDOW_TOKENS {
            break;
        }
        for k in 0..want {
            tokens.push(1 + (w * 7 + k * 13) % 90);
        }
        spans.push((s..tokens.len()).into());
    }
    println!("{} words, {} tokens", spans.len(), tokens.len());

    let mut model = Hierarchical::load(&gpu, &path, W_TOKEN as usize).expect("load");
    let loss = model.forward_backward(&gpu, &tokens, &spans);
    println!("loss {loss:.9}");
    for stage in ["encoder", "backbone", "decoder"] {
        for (i, n) in model.grad_norms_by_block(&gpu, stage).iter().enumerate() {
            println!("{stage}[{i}] {n:.7e}");
        }
    }
}
