// Count encoder/decoder length groups for a real window.
fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "src/gpu/ops.rs".into());
    let text = std::fs::read_to_string(&path).unwrap();
    let tok = neural_networks::tokenizer_utf8::Utf8Tokenizer::new();
    let ids = tok.to_tokens(&text);
    let ends = neural_networks::segment::word_ends(&ids);
    let mut words: Vec<std::ops::Range<usize>> = Vec::new();
    let mut s = 0usize;
    for e in ends {
        words.push(s..e as usize);
        s = e as usize;
    }
    let n = words.len().min(neural_networks::config::WORDS_PER_SEQ);
    let words = &words[..n];
    let dw = n - 1;
    let lens: Vec<usize> = (0..dw).map(|w| words[w].end - words[w].start).collect();
    let mut hist: std::collections::BTreeMap<usize, usize> = Default::default();
    for &l in &lens {
        *hist.entry(l.max(1).next_power_of_two()).or_insert(0) += 1;
    }
    println!("tokens={} words={} dw={}", ids.len(), n, dw);
    let cap = neural_networks::config::GROUP_MAX_ROWS;
    let mut groups = 0;
    for (k, c) in &hist {
        let tmax = *k;
        let per = (cap / (tmax + 1)).max(1);
        let pieces = c.div_ceil(per);
        println!("  bucket len<={k:3}  words={c:5}  per_piece={per:5}  pieces={pieces}");
        groups += pieces;
    }
    println!("total encoder groups ~= {groups}");
}
