// Prints the word segmentation of a Rust file: words separated by │, plus stats.
use neural_networks::{segment, tokenizer_utf8::Utf8Tokenizer};

fn main() {
    let path = std::env::args().nth(1).expect("usage: seg_demo <file.rs>");
    let text = std::fs::read_to_string(&path).unwrap();
    let tok = Utf8Tokenizer::new();
    let seq = tok.to_tokens(&text);
    let ends = segment::word_ends(&seq);

    let mut start = 0;
    let mut lens = Vec::new();
    let mut shown = String::new();
    for e in &ends {
        let w = tok.to_text(&seq[start..*e as usize]);
        lens.push(w.len());
        if start < 400 {
            shown.push_str(&w.replace('\n', "⏎"));
            shown.push('│');
        }
        start = *e as usize;
    }
    println!("{shown}\n");
    let n = lens.len();
    let total: usize = lens.iter().sum();
    let mut sorted = lens.clone();
    sorted.sort_unstable();
    println!("{path}: {total} bytes → {n} words");
    println!("  bytes/word: mean {:.2}, median {}, max {}", total as f32 / n as f32, sorted[n / 2], sorted[n - 1]);
}
