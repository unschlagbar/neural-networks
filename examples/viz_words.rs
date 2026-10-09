// Dumps the word segmentation of a text as JSON for the video scenes:
// `cargo run --example viz_words -- input.rs > video/src/data/words.json`
use neural_networks::{segment, tokenizer_utf8::Utf8Tokenizer};

fn json_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: viz_words <file>");
    let text = std::fs::read_to_string(&path).unwrap();
    let tok = Utf8Tokenizer::new();
    let seq = tok.to_tokens(&text);
    let ends = segment::word_ends(&seq);

    let mut start = 0;
    let mut words = Vec::new();
    for &e in &ends {
        let ids = &seq[start..e as usize];
        let bytes: Vec<String> = ids.iter().map(|b| b.to_string()).collect();
        words.push(format!(
            "{{\"text\":{},\"bytes\":[{}]}}",
            json_str(&tok.to_text(ids)),
            bytes.join(",")
        ));
        start = e as usize;
    }
    println!("{{\"words\":[\n  {}\n]}}", words.join(",\n  "));
}
