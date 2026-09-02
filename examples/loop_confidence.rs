//! How sure is the model inside a repetition loop?
//!
//! A loop the model "believes in" would show near-1.0 top-1 probabilities. A
//! loop that is only an artifact of argmax decoding shows modest ones: the top
//! token wins by a nose, every other branch stays available, and nothing but
//! the absence of sampling noise keeps the state on the cycle.

use std::range::Range;

use neural_networks::{
    config::TEMPERATURE, hierarchical::Hierarchical, nn::softmax::softmax, segment,
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

fn disp(t: &Utf8Tokenizer, id: u16) -> String {
    if id as usize >= 256 {
        t.display(id)
    } else {
        String::from_utf8_lossy(&[id as u8]).escape_debug().to_string()
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| "models/s3".into());
    let text = args.next().unwrap_or_default();
    let tail: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(30);

    let tokenizer = Utf8Tokenizer::new();
    let mut model = Hierarchical::load(&path, tokenizer.clone()).unwrap();
    let tokens = tokenizer.to_tokens(&text);
    let words = ranges(&tokens);
    model.make_cache(words.len() + 2, tokens.len() + words.len() + 8);
    model.forward_over(&tokens, &words);

    // Flatten every decoder slot with its target, then report the tail.
    let mut slots: Vec<(usize, u16)> = Vec::new();
    let mut cursor = 0;
    for w in 0..words.len() - 1 {
        let next = words[w + 1];
        let len = next.end - next.start;
        for k in 0..=len {
            let target = if k == len { tokenizer.w_token() } else { tokens[next.start + k] };
            slots.push((cursor + k, target));
        }
        cursor += len + 1;
    }

    println!("{:>5}  {:<6} {:<6}  {:>8} {:>8}  {:>7}", "slot", "target", "top-1", "p(T=1)", &format!("p(T={TEMPERATURE})"), "entropy");
    let mut sum_p1 = 0.0;
    let mut n = 0.0;
    for &(slot, target) in slots.iter().skip(slots.len().saturating_sub(tail)) {
        let z = model.char2_model.cache[slot].last().unwrap().output();
        let p = softmax(z);
        let scaled: Vec<f32> = z.iter().map(|&v| v / TEMPERATURE).collect();
        let q = softmax(&scaled);
        let top = p.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let ent: f32 = -p.iter().filter(|&&v| v > 0.0).map(|&v| v * v.ln()).sum::<f32>();
        sum_p1 += p[top];
        n += 1.0;
        println!(
            "{slot:>5}  {:<6} {:<6}  {:>8.4} {:>8.4}  {ent:>7.4}",
            disp(&tokenizer, target),
            disp(&tokenizer, top as u16),
            p[top],
            q[top],
        );
    }
    println!("\nmean top-1 probability over this tail: {:.4}", sum_p1 / n);
}
