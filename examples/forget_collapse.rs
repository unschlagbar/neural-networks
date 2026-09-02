//! Backbone forget gates during a repetition collapse.
//!
//! `f_prime` is the fraction of an mLSTM head's cell state carried into the
//! next word. A head sitting at 1.0 never forgets, so its state stops moving
//! and the context it feeds the decoder freezes — which is what a repetition
//! loop looks like from the inside. Runs the training-shaped forward over one
//! text and reports, per backbone block, the mean f_prime and how many heads
//! are pinned above 0.999.

use std::range::Range;

use neural_networks::{hierarchical::Hierarchical, segment, tokenizer_utf8::Utf8Tokenizer};

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
    let text = match args.next() {
        Some(a) => std::fs::read_to_string(&a).unwrap_or(a),
        None => "fn main() {\n    let v = 1;\n}\n".into(),
    };

    let tokenizer = Utf8Tokenizer::new();
    let mut model = Hierarchical::load(&path, tokenizer.clone()).unwrap();
    let tokens = tokenizer.to_tokens(&text);
    let words = ranges(&tokens);
    model.make_cache(words.len() + 2, tokens.len() + words.len() + 8);
    model.forward_over(&tokens, &words);

    let f = model.backbone_forget_samples();
    let steps = f[0][0].len();
    println!("{} words, {} backbone mLSTM blocks, {} heads\n", steps, f.len(), f[0].len());

    println!("block  mean f'   heads>0.999   mean f' (first 8 words)   mean f' (last 8 words)");
    for (b, heads) in f.iter().enumerate() {
        let n = (heads.len() * steps) as f32;
        let mean: f32 = heads.iter().flatten().sum::<f32>() / n;
        let pinned = heads
            .iter()
            .filter(|h| h.iter().all(|&v| v > 0.999))
            .count();
        let head_mean = |r: std::ops::Range<usize>| -> f32 {
            let c = (heads.len() * r.len()) as f32;
            heads.iter().map(|h| h[r.clone()].iter().sum::<f32>()).sum::<f32>() / c
        };
        let first = head_mean(0..8.min(steps));
        let last = head_mean(steps.saturating_sub(8)..steps);
        println!(
            "{b:5}  {mean:.5}   {pinned:>3}/{:<3}      {first:.5}                  {last:.5}",
            heads.len()
        );
    }

    // Per-word trajectory of the retention averaged over every block and head:
    // if it climbs toward 1 while the text repeats, the state is freezing.
    println!("\nper-word mean f' over all blocks/heads:");
    for w in 0..steps {
        let mut s = 0.0;
        let mut c = 0.0;
        for heads in &f {
            for h in heads {
                s += h[w];
                c += 1.0;
            }
        }
        let word = &tokens[words[w].start..words[w].end];
        let mut disp = String::new();
        for &t in word {
            if t as usize >= 256 {
                disp.push_str(&tokenizer.display(t));
            } else {
                disp.push_str(&String::from_utf8_lossy(&[t as u8]).escape_debug().to_string());
            }
        }
        println!("  w{w:<4} {:<14} f' {:.5}", disp, s / c);
    }
}
