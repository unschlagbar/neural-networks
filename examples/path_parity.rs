//! Numerical parity between the two forward paths.
//!
//! `forward_over` (training) and `sample` (inference) must agree token for
//! token on the same history. For every word boundary this replays the prefix
//! through the sampling path greedily and compares its first freely generated
//! token against the training path's top-1 prediction for that same position.
//! Any mismatch is a divergence between training and inference, not a model
//! preference: both are argmax over the same history.

use std::range::Range;

use neural_networks::{
    hierarchical::Hierarchical, segment, tokenizer_utf8::Utf8Tokenizer,
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

fn show(tok: &Utf8Tokenizer, ids: &[u16]) -> String {
    let mut s = String::new();
    for &t in ids {
        if t as usize >= 256 {
            s.push_str(&tok.display(t));
        } else {
            s.push_str(&String::from_utf8_lossy(&[t as u8]).escape_debug().to_string());
        }
    }
    s
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| "models/s3".into());
    let file = args.next().unwrap_or_else(|| "src/sequential.rs".into());
    let budget: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(400);

    let mut text = std::fs::read_to_string(&file).unwrap();
    let cut = (0..=budget.min(text.len())).rev().find(|&i| text.is_char_boundary(i)).unwrap();
    text.truncate(cut);

    let tokenizer = Utf8Tokenizer::new();
    let mut model = Hierarchical::load(&path, tokenizer.clone()).unwrap();
    let tokens = tokenizer.to_tokens(&text);
    let words = ranges(&tokens);
    let w_tok = tokenizer.w_token();
    println!("{} tokens, {} words", tokens.len(), words.len());

    // Training path: top-1 at every decoder slot, indexed by word.
    model.make_cache(words.len() + 2, tokens.len() + words.len() + 8);
    model.forward_over(&tokens, &words);
    let mut trained: Vec<Vec<u16>> = Vec::new();
    let mut cursor = 0;
    for w in 0..words.len() - 1 {
        let len = words[w + 1].end - words[w + 1].start;
        let mut pred = Vec::new();
        for k in 0..=len {
            let p = model.char2_model.cache[cursor + k].last().unwrap().output();
            pred.push(p.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u16);
        }
        cursor += len + 1;
        trained.push(pred);
    }

    // Inference path: replay each prefix, greedy, take the first free token.
    model.make_cache(1, tokens.len() + words.len() + 8);
    let mut mism = 0;
    let mut checked = 0;
    for w in 0..words.len() - 1 {
        // Comparable only when the training path closes word w with [W]: the
        // sampler's first free token is then word w+1's first char in both.
        if w == 0 || *trained[w - 1].last().unwrap() != w_tok {
            continue;
        }
        let prefix = &tokens[..words[w].end];
        let mut first = None;
        model.sample(prefix, 8, 1e-6, 1.0, |t| {
            if first.is_none() {
                first = Some(t);
            }
            false
        });
        let Some(got) = first else { continue };
        let want = trained[w][0];
        checked += 1;
        if got != want {
            mism += 1;
            if mism <= 25 {
                println!(
                    "w{w:3} after {:<14} training-path top-1 {:<8} sampling-path {}",
                    show(&tokenizer, &tokens[words[w].start..words[w].end]),
                    show(&tokenizer, &[want]),
                    show(&tokenizer, &[got]),
                );
            }
        }
    }
    println!("\nmismatches: {mism}/{checked}");
}
