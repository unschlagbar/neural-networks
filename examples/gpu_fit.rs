// Runs the worst-shaped window of a corpus through the GPU trainer at the
// current config and reports peak VRAM (and step time).
use std::rc::Rc;
use std::time::Instant;
use neural_networks::{
    batches::ChunkedWordDataSet, config::*,
    gpu::{Gpu, hierarchical::{HierCfg, Hierarchical}},
    nn2::optim::AdamCfg, tokenizer_utf8::Utf8Tokenizer,
};

fn used_mb() -> f64 {
    let o = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"]).output().unwrap();
    String::from_utf8_lossy(&o.stdout).trim().parse().unwrap_or(0.0)
}

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let wps: usize = std::env::args().nth(2).map_or(WORDS_PER_SEQ, |s| s.parse().unwrap());
    let tok = Rc::new(Utf8Tokenizer::new());
    let mut data = ChunkedWordDataSet::open(tok.clone(), &path, wps, MIN_WORDS_PER_SEQ, MAX_WINDOW_TOKENS, CHUNK_BYTES);
    let chunk = data.next_chunk().unwrap();
    let worst = chunk.iter().max_by_key(|b| {
        let tmax = b.words.iter().map(|r| r.end - r.start).max().unwrap() + 1;
        (b.words.len() - 1) * tmax
    }).unwrap();
    let dw = worst.words.len() - 1;
    let tmax = worst.words.iter().map(|r| r.end - r.start).max().unwrap() + 1;

    let gpu = Gpu::new().expect("no GPU");
    let heads = 8;
    let cfg = HierCfg {
        vocab: tok.vocab_size(), hc: CHAR_HIDDEN, wh: WORD_HIDDEN,
        enc_blocks: 2, bb_blocks: WORD_BLOCKS, dec_blocks: 2,
        heads, dqk: WORD_HIDDEN / heads, w_token: tok.w_token() as usize, cap: LOGIT_SOFTCAP,
    };
    let base0 = used_mb();
    let mut model = Hierarchical::new(&gpu, &cfg);
    println!("ctx {base0:.0} MB -> after model init {:.0} MB", used_mb());
    let mut opt = AdamCfg::new(LR, neural_networks::optimizers::WEIGHT_DECAY);
    let tokens: Vec<usize> = worst.tokens.iter().map(|&t| t as usize).collect();
    let words: Vec<(usize, usize)> = worst.words.iter().map(|r| (r.start, r.end)).collect();

    let mut peak: f64 = 0.0;
    let after_weights = used_mb();
    let mut secs = 0.0;
    for i in 0..4 {
        let t0 = Instant::now();
        let l = model.forward_backward(&gpu, &tokens, &words);
        opt.t += 1;
        model.step(&gpu, &opt);
        if i > 0 { secs += t0.elapsed().as_secs_f64(); }
        peak = peak.max(used_mb());
        std::hint::black_box(l);
    }
    println!("  (weights+optimizer resident: {:.0} MB)", after_weights);
    println!("wh={WORD_HIDDEN} blocks={WORD_BLOCKS} words={wps} maxword={MAX_WORD_BYTES}: dw={dw} tmax={tmax} -> peak {peak:.0} MB, {:.2}s/window", secs / 3.0);
}
