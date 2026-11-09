//! Greedy tokens of `infer::Hierarchical` (v2) against the old CPU model (NNM1).
//! `cargo run --release --example infer_parity -- models/s4 models/s4.v2`

use std::time::Instant;

use neural_networks::tokenizer_utf8::Utf8Tokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tok = Utf8Tokenizer::new();
    if args.len() == 2 {
        let mut new = neural_networks::infer::Hierarchical::load(&args[1]).unwrap();
        let prefix = tok.to_tokens("fn main() {\n    let ");
        for _ in 0..3 {
            let t = Instant::now();
            let words = 200;
            new.feed(&tok.to_tokens(&"let x = 1; ".repeat(words / 6)));
            let tf = t.elapsed();
            let t = Instant::now();
            let b = new.sample(&prefix, 300, 0.0, 1.0, |_| true);
            println!("feed ~{words} words {tf:.2?}, sample {} tokens {:.2?}", b.len(), t.elapsed());
        }
        return;
    }
    let (old_path, new_path) = (&args[1], &args[2]);
    let mut old = neural_networks::hierarchical::Hierarchical::load(old_path, tok).unwrap();
    old.make_cache(1, neural_networks::config::MAX_SEQ_LEN);
    let mut new = neural_networks::infer::Hierarchical::load(new_path).unwrap();
    let prompts = [
        "fn main() {\n    let ",
        "The quick brown fox",
        "use std::collections::HashMap;\n\n",
        "",
    ];
    let n = 300;
    let mut all_same = true;
    for p in prompts {
        let prefix = tok.to_tokens(p);
        let t = Instant::now();
        let a = old.sample(&prefix, n, 0.0, 1.0, |_| true);
        let ta = t.elapsed();
        let t = Instant::now();
        let b = new.sample(&prefix, n, 0.0, 1.0, |_| true);
        let tb = t.elapsed();
        let same = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
        // An empty prefix seeds a random byte, so the two runs start differently.
        if !p.is_empty() {
            all_same &= a == b;
        }
        println!(
            "{p:?}: {} / {} tokens agree (old {ta:.1?}, new {tb:.1?})\n  old: {:?}\n  new: {:?}",
            same,
            a.len().max(b.len()),
            tok.to_text_markup(&a),
            tok.to_text_markup(&b)
        );
    }

    let chat = neural_networks::sft::format_chat_prompt(
        &tok,
        &[neural_networks::sft::Turn::new(neural_networks::sft::Role::User, "What is the capital of France?")],
    );
    let a = old.sample_chat(&chat, n, 0.0, 0.0, 1.0, |_| true);
    let b = new.sample_chat(&chat, n, 0.0, 0.0, 1.0, |_| true);
    all_same &= a == b;
    println!("chat: equal {}\n  old: {:?}\n  new: {:?}", a == b, tok.to_text_markup(&a), tok.to_text_markup(&b));
    println!("{}", if all_same { "IDENTICAL" } else { "DIFFERENT" });
}
