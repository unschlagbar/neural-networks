//! Composition parity: GPU training path vs CPU sampling path.
//!
//! The layers are each checked against CPU, but the hierarchical composition —
//! encode -> backbone -> decode, the [W] slot, the injected context, the tied
//! table — is only ever checked GPU against GPU. A model is trained by the GPU
//! path and sampled by the CPU one, so a disagreement here is invisible to
//! every existing test and shows up only as bad generation.
//!
//!   cargo run --release --features cuda --example gpu_cpu_parity [model] [file] [bytes]

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda`");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::range::Range;

    use neural_networks::batches::WordBatch;
    use neural_networks::gpu::Gpu;
    use neural_networks::hierarchical::Hierarchical;
    use neural_networks::{segment, tokenizer_utf8::Utf8Tokenizer};

    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| "models/s3".into());
    let file = args.next().unwrap_or_else(|| "src/sequential.rs".into());
    let budget: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(1500);

    let mut text = std::fs::read_to_string(&file).unwrap();
    let cut = (0..=budget.min(text.len()))
        .rev()
        .find(|&i| text.is_char_boundary(i))
        .unwrap();
    text.truncate(cut);

    let tokenizer = Utf8Tokenizer::new();
    let tokens = tokenizer.to_tokens(&text);
    let mut words: Vec<Range<usize>> = Vec::new();
    let mut start = 0;
    for e in segment::word_ends(&tokens) {
        words.push(Range { start, end: e as usize });
        start = e as usize;
    }
    println!("{} tokens, {} words", tokens.len(), words.len());

    // CPU: the path `hs` samples with.
    let mut cpu = Hierarchical::load(&path, tokenizer.clone()).unwrap();
    cpu.make_cache(words.len() + 2, tokens.len() + words.len() + 8);
    let (cpu_char, cpu_word) = cpu.eval_decode_loss(std::iter::once(WordBatch {
        tokens: &tokens,
        words: words.clone(),
    }));

    // GPU: the path the model was trained with.
    let gpu = Gpu::new().expect("cuda init");
    let mut g = neural_networks::gpu::hierarchical::Hierarchical::load(
        &gpu,
        &path,
        tokenizer.w_token() as usize,
    )
    .unwrap();
    let ids: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();
    let gpu_char = g.eval_loss(&gpu, &ids, &words);

    println!("\nCPU  mean per-token decode CE : {cpu_char:.6}  (ppl {:.4})", cpu_char.exp());
    println!("GPU  mean per-token decode CE : {gpu_char:.6}  (ppl {:.4})", gpu_char.exp());
    let rel = (cpu_char - gpu_char).abs() / cpu_char.abs().max(1e-6);
    println!("relative difference           : {:.4}%", 100.0 * rel);
    println!("CPU per-word loss             : {cpu_word:.6}");
}
