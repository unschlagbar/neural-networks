//! Does the repetition survive a non-greedy sampler?
//!
//! `hs` samples at config's TEMPERATURE/TOP_P. Measures, per setting, the
//! longest run of a repeating token cycle in the generated text — the thing
//! that shows up as `};};};`.

use neural_networks::{hierarchical::Hierarchical, tokenizer_utf8::Utf8Tokenizer};

/// Length of the longest tail that is a repetition of some cycle of period <= 8.
fn cycle_tail(out: &[u16]) -> (usize, usize) {
    let mut best = (0, 0);
    for period in 1..=8 {
        if out.len() < 2 * period {
            continue;
        }
        let mut n = 0;
        let mut i = out.len();
        while i >= 2 * period && out[i - period..i] == out[i - 2 * period..i - period] {
            n += period;
            i -= period;
        }
        if n > best.0 {
            best = (n, period);
        }
    }
    best
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| "models/s3".into());
    let prefix = args
        .next()
        .unwrap_or_else(|| "fn main() {\n    let v = vec![1, 2, 3];\n".into());

    let tokenizer = Utf8Tokenizer::new();
    let mut model = Hierarchical::load(&path, tokenizer.clone()).unwrap();
    model.make_cache(1, 4096);
    let tokens = tokenizer.to_tokens(&prefix);

    println!("{:>6} {:>7}  {:>10}  {}", "T", "top_p", "cycle tail", "tail of output");
    for &(t, p) in &[
        (0.5, 0.9),   // config
        (0.7, 0.9),
        (0.8, 0.95),
        (1.0, 0.95),
        (1.0, 1.0),
    ] {
        let mut out = Vec::new();
        model.sample(&tokens, 600, t, p, |tok| {
            out.push(tok);
            true
        });
        let (n, period) = cycle_tail(&out);
        let tail: String = String::from_utf8_lossy(
            &out[out.len().saturating_sub(48)..]
                .iter()
                .filter(|&&x| (x as usize) < 256)
                .map(|&x| x as u8)
                .collect::<Vec<u8>>(),
        )
        .escape_debug()
        .to_string();
        println!("{t:>6} {p:>7}  {n:>4} (p={period})  {tail}");
    }
}
