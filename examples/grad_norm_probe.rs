//! Global gradient norm of a checkpoint on real windows, and the gradient noise scale.
//!
//!   cargo run --release --features cuda --example grad_norm_probe -- <model> <corpus> [windows]
//!
//! Accumulates gradients over `windows` windows without ever stepping (the checkpoint
//! is read, never written) and measures the norm of the running sum `S_k` after each
//! one. `||S_k / k||` is the mean-gradient norm at batch size k — the number
//! `GRAD_CLIP_NORM` is compared against, at k = BATCH_SIZE.
//!
//! With per-window gradients of mean G and covariance trace tr(Σ),
//! E||S_k||² / k = k·|G|² + tr(Σ), so a line through `||S_k||² / k` against k gives both,
//! and B_simple = tr(Σ) / |G|² is the simple noise scale of McCandlish et al.
//! (arXiv 1812.06162): roughly the batch size, in windows, past which a larger batch
//! stops buying fewer steps. The points are cumulative and so correlated — treat the
//! result as an order of magnitude.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda` to run this probe");
}

#[cfg(feature = "cuda")]
fn main() {
    use neural_networks::batches::ChunkedWordDataSet;
    use neural_networks::config::*;
    use neural_networks::gpu::Gpu;
    use neural_networks::gpu::hierarchical::Hierarchical;
    use neural_networks::tokenizer_utf8::Utf8Tokenizer;

    let mut args = std::env::args().skip(1);
    let usage = "usage: grad_norm_probe <model> <corpus> [windows]";
    let model_path = args.next().expect(usage);
    let corpus = args.next().expect(usage);
    let limit: usize = args.next().map_or(BATCH_SIZE, |s| s.parse().expect(usage));

    let gpu = Gpu::new().expect("no GPU");
    let tok = Utf8Tokenizer::new();
    let w_token = tok.w_token() as usize;
    let mut model = Hierarchical::load(&gpu, &model_path, w_token).expect("load model");
    model.set_stateful(CARRY_WINDOW_STATE);
    let mut data = ChunkedWordDataSet::open(
        tok,
        &corpus,
        WORDS_PER_SEQ,
        MIN_WORDS_PER_SEQ,
        MAX_WINDOW_TOKENS,
        CHUNK_BYTES,
    );

    println!("    k |   loss | ||S_k||/k | ||S_k||^2/k");
    // (k, ||S_k||² / k)
    let mut pts: Vec<(f64, f64)> = Vec::new();
    let mut ran_prev = false;
    'outer: while let Some(chunk) = data.next_chunk() {
        for batch in chunk.iter() {
            let tokens: Vec<usize> = batch.tokens.iter().map(|&t| t as usize).collect();
            if batch.words.len() < 2 {
                ran_prev = false;
                continue;
            }
            model.set_continues(batch.continues && ran_prev);
            model.set_doc_starts(&batch.doc_starts);
            ran_prev = true;
            let loss = model.forward_backward(&gpu, &tokens, &batch.words);
            model.clip_grad_norm(&gpu, f32::INFINITY);
            let s = model.grad_norm(&gpu) as f64;
            let k = pts.len() as f64 + 1.0;
            pts.push((k, s * s / k));
            println!("{k:>5} | {loss:>6.3} | {:>9.4} | {:>11.4}", s / k, s * s / k);
            if pts.len() >= limit {
                break 'outer;
            }
        }
    }

    let n = pts.len() as f64;
    let (mx, my) = pts
        .iter()
        .fold((0.0, 0.0), |(a, b), &(x, y)| (a + x / n, b + y / n));
    let sxy: f64 = pts.iter().map(|&(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = pts.iter().map(|&(x, _)| (x - mx) * (x - mx)).sum();
    let g2 = sxy / sxx;
    let tr = my - g2 * mx;
    println!("\n|G|^2 ~ {g2:.4e}, tr(Sigma) ~ {tr:.4e} (per window)");
    println!("|G| ~ {:.4}, single-window noise ~ {:.4}", g2.max(0.0).sqrt(), tr.max(0.0).sqrt());
    if g2 > 0.0 {
        println!("B_simple ~ {:.0} windows (BATCH_SIZE = {BATCH_SIZE})", tr / g2);
    } else {
        println!("|G|^2 fit non-positive: noise dominates at this sample size");
    }
}
