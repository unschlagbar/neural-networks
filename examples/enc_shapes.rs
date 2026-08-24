//! Distinct encoder rectangles a real corpus produces, the number a CUDA graph cache
//! would have to hold. Usage: `cargo run --release --example enc_shapes -- <file> [windows]`

use std::collections::{HashMap, HashSet};
use std::range::Range;

use neural_networks::config::{GROUP_MAX_ROWS, WORDS_PER_SEQ};
use neural_networks::gpu::word_groups::EncoderGroups;
use neural_networks::segment;
use neural_networks::tokenizer_utf8::Utf8Tokenizer;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: enc_shapes <file> [windows]");
    let max_windows: usize = args.next().map_or(usize::MAX, |a| a.parse().unwrap());

    let text = std::fs::read_to_string(&path).expect("read");
    let tok = Utf8Tokenizer::new();
    let raw = tok.to_tokens(&text);
    let ends = segment::word_ends(&raw);
    let tokens: Vec<usize> = raw.iter().map(|&t| t as usize).collect();
    let words: Vec<Range<usize>> = {
        let mut v = Vec::with_capacity(ends.len());
        let mut start = 0u32;
        for e in ends {
            v.push(Range::from(start as usize..e as usize));
            start = e;
        }
        v
    };

    let mut enc = EncoderGroups::new();
    // (tmax, n_g) -> how many groups of that exact rectangle were launched.
    let mut shapes: HashMap<(usize, usize), usize> = HashMap::new();
    let mut per_window: Vec<usize> = Vec::new();
    let mut groups_total = 0usize;

    let mut w0 = 0;
    let mut windows = 0;
    while w0 + 2 < words.len() && windows < max_windows {
        let w1 = (w0 + WORDS_PER_SEQ).min(words.len());
        let win = &words[w0..w1];
        enc.build(&tokens, win, tok.w_token() as usize, GROUP_MAX_ROWS);
        let mut here = HashSet::new();
        for g in 0..enc.len() {
            let grp = enc.group(g);
            let key = (grp.tmax, grp.n_words());
            *shapes.entry(key).or_default() += 1;
            here.insert(key);
            groups_total += 1;
        }
        per_window.push(here.len());
        windows += 1;
        w0 = w1;
    }

    let mut keys: Vec<_> = shapes.keys().copied().collect();
    keys.sort();
    let lens: HashSet<usize> = keys.iter().map(|k| k.0).collect();
    println!("{windows} windows, {groups_total} encoder groups launched");
    println!(
        "distinct lengths (tmax):        {}  <- graphs if n_g were pinned",
        lens.len()
    );
    println!(
        "distinct (tmax, n_g) shapes:    {}  <- graphs as it stands",
        keys.len()
    );
    println!(
        "groups per window: {:.1} avg, {} max distinct in one window",
        groups_total as f64 / windows.max(1) as f64,
        per_window.iter().copied().max().unwrap_or(0)
    );

    println!("\ntmax  distinct n_g  n_g range        groups");
    for t in 1..=17 {
        let ks: Vec<_> = keys.iter().filter(|k| k.0 == t).collect();
        if ks.is_empty() {
            continue;
        }
        let lo = ks.iter().map(|k| k.1).min().unwrap();
        let hi = ks.iter().map(|k| k.1).max().unwrap();
        let n: usize = ks.iter().map(|k| shapes[k]).sum();
        println!("{t:>4}  {:>12}  {lo:>6}..{hi:<8} {n:>6}", ks.len());
    }

    // Chunk the batch axis instead: run each group as ceil(n_g / c) launches of a
    // fixed [c, tmax] rectangle, padding only the tail. Words in a group are
    // independent (the recurrence runs along tmax), so this is a free split, and the
    // shape count collapses to one per length.
    println!("\nchunked batch axis — one graph per (tmax, chunk):");
    println!("  chunk  graphs  launches  padded rows");
    for c in [32usize, 64, 128, 256, 512, 1024] {
        let mut launches = 0usize;
        let mut real = 0usize;
        let mut padded = 0usize;
        for (&(t, n), &count) in shapes.iter() {
            let full = n / c;
            let tail = n % c;
            let l = full + usize::from(tail > 0);
            launches += l * count;
            real += n * t * count;
            padded += l * c * t * count;
        }
        // A tail chunk is its own shape unless it is padded up to `c`; padding it keeps
        // the cache at one graph per length.
        println!(
            "  {c:>5}  {:>6}  {launches:>8}  {:>+9.1}%",
            lens.len(),
            100.0 * (padded as f64 - real as f64) / real as f64
        );
    }
}
