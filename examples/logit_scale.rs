//! Logit scale and sampling confidence.
//!
//! The decoder head is undecayed Adam behind SoftCap(30). If its logits sit
//! near the cap, dividing by TEMPERATURE turns top-p sampling into argmax and
//! any repetition the model prefers becomes a fixed point. Reports the logit
//! spread and the top-1 probability after temperature over real text.

use std::range::Range;

use neural_networks::{
    config::{TEMPERATURE, TOP_P},
    hierarchical::Hierarchical,
    nn::softmax::softmax,
    segment,
    tokenizer_utf8::Utf8Tokenizer,
};

fn ranges(tokens: &[u16]) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut start = 0;
    for e in segment::word_ends(tokens) {
        out.push(Range { start, end: e as usize });
        start = e as usize;
    }
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| "models/s3".into());
    let file = args.next().unwrap_or_else(|| "src/sequential.rs".into());
    let budget: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(1500);

    let mut text = std::fs::read_to_string(&file).unwrap();
    let cut = (0..=budget.min(text.len())).rev().find(|&i| text.is_char_boundary(i)).unwrap();
    text.truncate(cut);

    let tokenizer = Utf8Tokenizer::new();
    let mut model = Hierarchical::load(&path, tokenizer.clone()).unwrap();
    let tokens = tokenizer.to_tokens(&text);
    let words = ranges(&tokens);
    model.make_cache(words.len() + 2, tokens.len() + words.len() + 8);
    model.forward_over(&tokens, &words);

    let mut cursor = 0;
    let mut max_abs: f32 = 0.0;
    let mut sum_top1 = 0.0;
    let mut sum_top1_t = 0.0;
    let mut nucleus_1 = 0; // slots where top-p keeps a single candidate
    let mut n = 0;
    for w in 0..words.len() - 1 {
        let len = words[w + 1].end - words[w + 1].start;
        for k in 0..=len {
            let z = model.char2_model.cache[cursor + k].last().unwrap().output();
            max_abs = max_abs.max(z.iter().fold(0.0, |a: f32, &v| a.max(v.abs())));
            let p = softmax(z);
            let scaled: Vec<f32> = z.iter().map(|&v| v / TEMPERATURE).collect();
            let q = softmax(&scaled);
            sum_top1 += p.iter().fold(0.0, |a: f32, &v| a.max(v));
            let mut sorted: Vec<f32> = q.to_vec();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            sum_top1_t += sorted[0];
            if sorted[0] >= TOP_P {
                nucleus_1 += 1;
            }
            n += 1;
        }
        cursor += len + 1;
    }
    let n = n as f32;
    println!("slots: {n}");
    println!("max |logit| seen            : {max_abs:.3}  (SoftCap = 30)");
    println!("mean top-1 prob  (T = 1.0)  : {:.4}", sum_top1 / n);
    println!("mean top-1 prob  (T = {TEMPERATURE}) : {:.4}", sum_top1_t / n);
    println!(
        "slots where top-p={TOP_P} keeps ONE candidate: {nucleus_1}/{n} = {:.1}%",
        100.0 * nucleus_1 as f32 / n
    );
}
