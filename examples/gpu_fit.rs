// Runs the worst-shaped window of a corpus through the GPU trainer at the
// current config and reports peak VRAM (and step time).
//
//   gpu_fit <corpus> [words_per_seq] [chunks to scan]
//
// Packing fixes the word count, so what varies — and what sizes the encoder and
// decoder rectangles — is the token span. One chunk is not a sample of that: a chunk
// of source code and a chunk of prose differ by more than 2x in tokens per word, so
// scan several before believing the peak.
use neural_networks::{
    batches::ChunkedWordDataSet,
    config::*,
    gpu::{
        Gpu,
        hierarchical::{Hierarchical, ModelCfg},
    },
    nn2::optim::AdamCfg,
    tokenizer_utf8::Utf8Tokenizer,
};
use std::{range::Range, time::Instant};

fn used_mb() -> f64 {
    let o = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .unwrap();
    String::from_utf8_lossy(&o.stdout)
        .trim()
        .parse()
        .unwrap_or(0.0)
}

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let wps: usize = std::env::args()
        .nth(2)
        .map_or(WORDS_PER_SEQ, |s| s.parse().unwrap());
    let scan: usize = std::env::args().nth(3).map_or(1, |s| s.parse().unwrap());
    let tok = Utf8Tokenizer::new();
    let mut data = ChunkedWordDataSet::open(
        tok,
        &path,
        wps,
        MIN_WORDS_PER_SEQ,
        MAX_WINDOW_TOKENS,
        CHUNK_BYTES,
    );
    // The window's own tokens and words, copied out: `worst` borrows the chunk it came
    // from, and only one chunk can be alive at a time while scanning.
    let mut worst: Option<(Vec<u16>, Vec<Range<usize>>, Vec<usize>)> = None;
    let cost = |words: &[Range<usize>]| {
        let tmax = words.iter().map(|r| r.end - r.start).max().unwrap_or(0) + 1;
        (words.len() - 1) * tmax
    };
    for _ in 0..scan {
        let Some(chunk) = data.next_chunk() else {
            break;
        };
        for b in chunk.iter() {
            if b.words.len() < 2 {
                continue;
            }
            if worst
                .as_ref()
                .is_none_or(|(_, w, _)| cost(w) < cost(&b.words))
            {
                worst = Some((b.tokens.to_vec(), b.words.clone(), b.doc_starts.clone()));
            }
        }
    }
    let (worst_tokens, worst_words, worst_doc_starts) = worst.expect("corpus yields no window");
    let dw = worst_words.len() - 1;
    let tmax = worst_words.iter().map(|r| r.end - r.start).max().unwrap() + 1;

    let gpu = Gpu::new().expect("no GPU");
    let heads = 8;
    let cfg = ModelCfg {
        vocab: tok.vocab_size(),
        hc: CHAR_HIDDEN,
        wh: WORD_HIDDEN,
        enc_blocks: 4,
        bb_blocks: WORD_BLOCKS,
        dec_blocks: 4,
        heads,
        dqk: WORD_HIDDEN / heads,
        w_token: tok.w_token() as usize,
        cap: LOGIT_SOFTCAP,
    };
    let base0 = used_mb();
    let mut model = Hierarchical::new(&gpu, cfg);
    println!("ctx {base0:.0} MB -> after model init {:.0} MB", used_mb());
    let mut opt = AdamCfg::new(LR, neural_networks::optimizers::WEIGHT_DECAY);
    let tokens: Vec<usize> = worst_tokens.iter().map(|&t| t as usize).collect();
    let words = &worst_words;

    let mut peak: f64 = 0.0;
    let after_weights = used_mb();
    let mut secs = 0.0;
    for i in 0..4 {
        let t0 = Instant::now();
        model.set_doc_starts(&worst_doc_starts);
        let l = model.forward_backward(&gpu, &tokens, words);
        opt.t += 1;
        model.step(&gpu, &opt);
        if i > 0 {
            secs += t0.elapsed().as_secs_f64();
        }
        peak = peak.max(used_mb());
        std::hint::black_box(l);
    }
    println!("  (weights+optimizer resident: {:.0} MB)", after_weights);
    println!(
        "wh={WORD_HIDDEN} blocks={WORD_BLOCKS} words={wps} maxword={MAX_WORD_BYTES}: dw={dw} tmax={tmax} -> peak {peak:.0} MB, {:.2}s/window",
        secs / 3.0
    );
}
