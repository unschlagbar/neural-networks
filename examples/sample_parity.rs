//! Compares the training-shaped forward (`forward_over`, teacher-forced) with
//! the free-running sampling loop on the same prefix. If the teacher-forced
//! top-1 stream is sane where sampling collapses, the fault is in the sampling
//! path; if both collapse, the fault is the model.

use std::range::Range;

use neural_networks::{
    config::MAX_SEQ_LEN,
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
    let path = args.next().unwrap_or_else(|| "s3".into());
    let text = args.next().unwrap_or_else(|| {
        "fn main() {\n    let v = vec![1, 2, 3];\n    for x in v {\n        println!(\"{x}\");\n    }\n}\n".into()
    });

    let tokenizer = Utf8Tokenizer::new();
    let mut model = Hierarchical::load(&path, tokenizer.clone()).unwrap();
    let tokens = tokenizer.to_tokens(&text);
    let words = ranges(&tokens);
    println!("{} tokens, {} words\n", tokens.len(), words.len());

    model.make_cache(words.len() + 2, MAX_SEQ_LEN);
    model.forward_over(&tokens, &words);

    // Teacher-forced top-1 per decoder slot. Slot ranges mirror `forward_over`:
    // word w decodes word w+1, one slot per char plus the trailing [W].
    let w_tok = tokenizer.w_token();
    let mut cursor = 0;
    let mut forced_ok = 0;
    let mut forced_tot = 0;
    let mut prev_first: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
    for w in 0..words.len().saturating_sub(1) {
        let next = words[w + 1];
        let len = next.end - next.start;
        let mut pred = Vec::new();
        for k in 0..=len {
            let logits = model.char2_model.cache[cursor + k].last().unwrap().output();
            let top = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0 as u16;
            let target = if k == len { w_tok } else { tokens[next.start + k] };
            forced_tot += 1;
            if top == target {
                forced_ok += 1;
            }
            pred.push(top);
        }
        // Fixed-point check: the backbone is recurrent, so if its state has
        // converged the decoder's first-slot logits repeat bit for bit.
        let first: Vec<f32> = model.char2_model.cache[cursor].last().unwrap().output().to_vec();
        if let Some(prev) = prev_first.get(&show(&tokenizer, &tokens[words[w].start..words[w].end])) {
            let d: f32 = first
                .iter()
                .zip(prev.iter())
                .map(|(a, b): (&f32, &f32)| (a - b).abs())
                .fold(0.0, f32::max);
            if w > 30 {
                println!("      ^ max|Δlogit| vs previous same-context word: {d:.3e}");
            }
        }
        prev_first.insert(show(&tokenizer, &tokens[words[w].start..words[w].end]), first);
        cursor += len + 1;
        let mut target: Vec<u16> = tokens[next.start..next.end].to_vec();
        target.push(w_tok);
        println!(
            "w{w:3} ctx {:<12} want {:<20} got {}",
            show(&tokenizer, &tokens[words[w].start..words[w].end]),
            show(&tokenizer, &target),
            show(&tokenizer, &pred),
        );
    }
    println!(
        "\nteacher-forced top-1 accuracy: {}/{} = {:.1}%\n",
        forced_ok,
        forced_tot,
        100.0 * forced_ok as f32 / forced_tot as f32
    );

    // Free-running: greedy (temperature ~0) so it is comparable to the top-1
    // stream above, from the same prefix.
    println!("free-run (greedy) from the same prefix:");
    let mut buf = Vec::new();
    model.make_cache(1, MAX_SEQ_LEN);
    model.sample(&tokens, 300, 1e-6, 1.0, |t| {
        buf.push(t);
        true
    });
    println!("{}", show(&tokenizer, &buf));
}
