//! Composition parity: the GPU model vs the CPU one (`cpu::hierarchical`).

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda`");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::range::Range;

    use neural_networks::gpu::Gpu;
    use neural_networks::cpu::hierarchical::Hierarchical;
    use neural_networks::{segment, tokenizer_utf8::Utf8Tokenizer};

    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| "models/s4.v2".into());
    let file = args.next().unwrap_or_else(|| "src/checkpoint.rs".into());
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
        words.push(Range {
            start,
            end: e as usize,
        });
        start = e as usize;
    }
    println!("{} tokens, {} words", tokens.len(), words.len());

    let ids: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();
    let border = [words.len() / 2];

    let t0 = std::time::Instant::now();
    let mut cpu = Hierarchical::load(&path).unwrap();
    println!("CPU model loaded in {:.1?}", t0.elapsed());
    let t0 = std::time::Instant::now();
    let cpu_char = cpu.eval_loss(&ids, &words);
    let cpu_word = cpu.last_word_loss();
    cpu.set_doc_starts(&border);
    let cpu_border = cpu.eval_loss(&ids, &words);
    println!("CPU: two windows in {:.1?}", t0.elapsed());
    drop(cpu);

    let gpu = Gpu::new().expect("cuda init");
    let mut g = neural_networks::gpu::hierarchical::Hierarchical::load(
        &gpu,
        &path,
        tokenizer.w_token() as usize,
    )
    .unwrap();
    let gpu_char = g.eval_loss(&gpu, &ids, &words);
    let gpu_word = g.last_word_loss();
    g.set_doc_starts(&border);
    let gpu_border = g.eval_loss(&gpu, &ids, &words);

    let rel = |a: f32, b: f32| 100.0 * (a - b).abs() / a.abs().max(1e-6);
    println!("\n                 CPU        GPU        rel. diff");
    println!("char loss    {cpu_char:>9.6}  {gpu_char:>9.6}  {:>7.3}%", rel(cpu_char, gpu_char));
    println!("word loss    {cpu_word:>9.6}  {gpu_word:>9.6}  {:>7.3}%", rel(cpu_word, gpu_word));
    println!(
        "with border  {cpu_border:>9.6}  {gpu_border:>9.6}  {:>7.3}%",
        rel(cpu_border, gpu_border)
    );
}
