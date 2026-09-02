//! How much does the decoder actually use the backbone?
//!
//! Runs the training-shaped forward three times over the same text and reports
//! teacher-forced top-1 accuracy per backbone mode:
//!   Normal        — full cross-word recurrent state
//!   ResetEachWord — state dropped before every word: context is the one word before
//!   ZeroContext   — no context at all: the within-word floor
//! If Normal ≈ ResetEachWord the backbone carries no history, whatever its state
//! nominally holds.

use std::range::Range;

use neural_networks::{
    hierarchical::{BackboneMode, Hierarchical},
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

/// Teacher-forced top-1 accuracy and mean CE over every decoder slot.
fn score(model: &mut Hierarchical, tokens: &[u16], words: &[Range<usize>], w_tok: u16) -> (f32, f32) {
    model.forward_over(tokens, words);
    let mut cursor = 0;
    let (mut ok, mut tot, mut ce) = (0usize, 0usize, 0.0);
    for w in 0..words.len().saturating_sub(1) {
        let next = words[w + 1];
        let len = next.end - next.start;
        for k in 0..=len {
            let p = model.char2_model.cache[cursor + k].last().unwrap().output();
            let target = if k == len { w_tok } else { tokens[next.start + k] } as usize;
            let top = p.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            // The decoder stack ends in Softmax, so `p` is already normalized.
            ce -= p[target].max(1e-30).ln();
            tot += 1;
            ok += (top == target) as usize;
        }
        cursor += len + 1;
    }
    (100.0 * ok as f32 / tot as f32, ce / tot as f32)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| "models/s3".into());
    let mut text = match args.next() {
        Some(f) => std::fs::read_to_string(&f).unwrap_or(f),
        None => include_str!("../src/sequential.rs").to_string(),
    };
    // The cache is one slot per token, so a whole file is gigabytes.
    let budget: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(3000);
    if text.len() > budget {
        let cut = (0..=budget).rev().find(|&i| text.is_char_boundary(i)).unwrap();
        text.truncate(cut);
    }

    let tokenizer = Utf8Tokenizer::new();
    let mut model = Hierarchical::load(&path, tokenizer.clone()).unwrap();
    let tokens = tokenizer.to_tokens(&text);
    let words = ranges(&tokens);
    println!("{} tokens, {} words", tokens.len(), words.len());
    model.make_cache(words.len() + 2, tokens.len() + words.len() + 8);

    let w_tok = tokenizer.w_token();
    for mode in [
        BackboneMode::Normal,
        BackboneMode::ResetEachWord,
        BackboneMode::ZeroContext,
    ] {
        model.backbone_mode = mode;
        let (acc, ce) = score(&mut model, &tokens, &words, w_tok);
        println!("{mode:?}");
        println!("   top-1 {acc:.2}%   CE {ce:.4}   ppl {:.3}", ce.exp());
    }
}
