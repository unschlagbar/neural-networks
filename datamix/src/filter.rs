// Quality gates and deduplication.
//
// Every rejection is counted by reason, so the report can tell you *why* a
// source shrank — a filter that silently eats 90% of a corpus is the most
// expensive kind of configuration bug.

use std::collections::HashSet;

use neural_networks::sft::Role;

use crate::config::{Dedup, Filters};
use crate::record::Record;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Reject {
    TooShort,
    TooLong,
    TooFewWords,
    TooManyWords,
    LowAlpha,
    RepeatedLines,
    LongLine,
    Missing,
    Blocked,
    Language,
    Duplicate,
    Judged,
}

impl Reject {
    pub fn label(self) -> &'static str {
        match self {
            Reject::TooShort => "too short",
            Reject::TooLong => "too long",
            Reject::TooFewWords => "too few words",
            Reject::TooManyWords => "too many words",
            Reject::LowAlpha => "low letter ratio",
            Reject::RepeatedLines => "repeated lines",
            Reject::LongLine => "over-long line",
            Reject::Missing => "missing required text",
            Reject::Blocked => "blocklist hit",
            Reject::Language => "language",
            Reject::Duplicate => "duplicate",
            Reject::Judged => "rejected by the judge",
        }
    }
}

pub struct Filter {
    cfg: Filters,
    seen: HashSet<u64>,
    /// MinHash bands for near-duplicate detection: `(band index, band hash)`.
    bands: HashSet<(u8, u64)>,
}

const HASHES: usize = 8;
const BAND: usize = 2;
const SHINGLE: usize = 5;

impl Filter {
    pub fn new(cfg: Filters) -> Self {
        Self {
            cfg,
            seen: HashSet::new(),
            bands: HashSet::new(),
        }
    }

    pub fn check(&mut self, rec: &Record) -> Option<Reject> {
        let text = rec.train_text();
        match rec {
            Record::Sft {
                instruction,
                response,
                ..
            } if instruction.trim().is_empty() || response.trim().is_empty() => {
                return Some(Reject::Missing);
            }
            // A conversation needs a prompt and an answer: `sft.rs` masks the
            // assistant turns into the loss and drops any turn with nothing in
            // front of it, so one without an exchange trains on nothing.
            Record::Chat { turns, .. }
                if !turns.iter().any(|t| t.role == Role::Assistant)
                    || !turns.iter().any(|t| t.role == Role::User) =>
            {
                return Some(Reject::Missing);
            }
            _ => {}
        }

        let bytes = text.len();
        if bytes < self.cfg.min_bytes {
            return Some(Reject::TooShort);
        }
        if bytes > self.cfg.max_bytes {
            return Some(Reject::TooLong);
        }
        if rec.tokens() > self.cfg.max_tokens {
            return Some(Reject::TooLong);
        }
        if self.cfg.min_words > 0 && text.split_whitespace().count() < self.cfg.min_words {
            return Some(Reject::TooFewWords);
        }
        if self.cfg.min_alpha_ratio > 0.0 && alpha_ratio(&text) < self.cfg.min_alpha_ratio {
            return Some(Reject::LowAlpha);
        }
        if self.cfg.max_line_bytes != usize::MAX
            && text.lines().any(|l| l.len() > self.cfg.max_line_bytes)
        {
            return Some(Reject::LongLine);
        }
        if self.cfg.max_dup_line_ratio < 1.0 && dup_line_ratio(&text) > self.cfg.max_dup_line_ratio
        {
            return Some(Reject::RepeatedLines);
        }
        if !self.cfg.must_contain.is_empty()
            && !self.cfg.must_contain.iter().any(|n| contains_ci(&text, n))
        {
            return Some(Reject::Missing);
        }
        if self
            .cfg
            .must_not_contain
            .iter()
            .any(|n| contains_ci(&text, n))
        {
            return Some(Reject::Blocked);
        }
        // Last of the local gates: counting words means tokenizing and
        // segmenting the record, which costs more than every check above it.
        if self.cfg.max_words != usize::MAX && rec.words() > self.cfg.max_words {
            return Some(Reject::TooManyWords);
        }
        if self.is_duplicate(&text) {
            return Some(Reject::Duplicate);
        }
        None
    }

    fn is_duplicate(&mut self, text: &str) -> bool {
        match self.cfg.dedup {
            Dedup::Off => false,
            Dedup::Exact => !self.seen.insert(hash_normalized(text)),
            Dedup::Near => {
                let sig = minhash(text);
                // Two documents collide when a whole band of signatures matches,
                // the standard LSH trade: more bands catches more, at more
                // false positives.
                let mut hit = false;
                for (b, chunk) in sig.chunks(BAND).enumerate() {
                    let h = chunk.iter().fold(0xcbf29ce484222325u64, |a, &x| {
                        (a ^ x).wrapping_mul(0x100000001b3)
                    });
                    if !self.bands.insert((b as u8, h)) {
                        hit = true;
                    }
                }
                hit
            }
        }
    }
}

fn contains_ci(hay: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    hay.to_lowercase().contains(&needle.to_lowercase())
}

fn alpha_ratio(text: &str) -> f32 {
    let mut alpha = 0usize;
    let mut total = 0usize;
    for c in text.chars() {
        if c.is_whitespace() {
            continue;
        }
        total += 1;
        if c.is_alphanumeric() {
            alpha += 1;
        }
    }
    if total == 0 {
        return 0.0;
    }
    alpha as f32 / total as f32
}

/// Share of non-empty lines that repeat an earlier line — the cheap half of the
/// Gopher repetition rules, and the one that catches generated file lists,
/// license headers repeated per block, and log dumps.
fn dup_line_ratio(text: &str) -> f32 {
    let mut seen = HashSet::new();
    let mut total = 0usize;
    let mut dups = 0usize;
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() {
            continue;
        }
        total += 1;
        if !seen.insert(hash_bytes(l.as_bytes())) {
            dups += 1;
        }
    }
    if total == 0 {
        return 0.0;
    }
    dups as f32 / total as f32
}

fn hash_bytes(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf29ce484222325u64, |a, &x| {
        (a ^ x as u64).wrapping_mul(0x100000001b3)
    })
}

/// Hash of the text with runs of whitespace collapsed, so reindented or
/// re-wrapped copies of the same document hash alike.
fn hash_normalized(text: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let mut in_ws = true;
    for c in text.chars() {
        let c = if c.is_whitespace() {
            if in_ws {
                continue;
            }
            in_ws = true;
            ' '
        } else {
            in_ws = false;
            c.to_ascii_lowercase()
        };
        let mut buf = [0u8; 4];
        for &b in c.encode_utf8(&mut buf).as_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// MinHash signature over word 5-grams: `HASHES` independent hash families,
/// each keeping the minimum shingle hash. Jaccard similarity of two documents
/// is approximated by how many of the signatures agree.
fn minhash(text: &str) -> [u64; HASHES] {
    let words: Vec<u64> = text
        .split_whitespace()
        .map(|w| hash_bytes(w.to_lowercase().as_bytes()))
        .collect();
    let mut sig = [u64::MAX; HASHES];
    if words.len() < SHINGLE {
        // Too short to shingle: fall back to the whole-text hash so short
        // records still deduplicate exactly.
        let h = hash_normalized(text);
        for (i, s) in sig.iter_mut().enumerate() {
            *s = h.wrapping_add(i as u64);
        }
        return sig;
    }
    for w in words.windows(SHINGLE) {
        let base = w
            .iter()
            .fold(0xcbf29ce484222325u64, |a, &x| (a ^ x).wrapping_mul(0x100000001b3));
        for (i, s) in sig.iter_mut().enumerate() {
            // Cheap independent families: mix the shingle hash with a distinct
            // odd constant per family.
            let h = (base ^ (i as u64).wrapping_mul(0x9e3779b97f4a7c15))
                .wrapping_mul(0xff51afd7ed558ccd);
            *s = (*s).min(h ^ (h >> 33));
        }
    }
    sig
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Filters;

    fn doc(text: &str) -> Record {
        Record::Doc {
            text: text.to_string(),
        }
    }

    #[test]
    fn exact_dedup_ignores_whitespace_and_case() {
        let mut f = Filter::new(Filters {
            dedup: Dedup::Exact,
            ..Filters::default()
        });
        assert_eq!(f.check(&doc("fn main() { }")), None);
        assert_eq!(
            f.check(&doc("FN   main()\n{ }")),
            Some(Reject::Duplicate),
            "reindented copy must be caught"
        );
        assert_eq!(f.check(&doc("fn other() { }")), None);
    }

    #[test]
    fn near_dedup_catches_a_small_edit_but_not_a_different_text() {
        let a = "the quick brown fox jumps over the lazy dog again and again and then stops";
        let b = "the quick brown fox jumps over the lazy dog again and again and then halts";
        let c = "completely unrelated prose about parquet row groups and column chunks here";
        let mut f = Filter::new(Filters {
            dedup: Dedup::Near,
            ..Filters::default()
        });
        assert_eq!(f.check(&doc(a)), None);
        assert_eq!(f.check(&doc(b)), Some(Reject::Duplicate));
        assert_eq!(f.check(&doc(c)), None);
    }

    #[test]
    fn quality_gates_name_their_reason() {
        let mut f = Filter::new(Filters {
            min_bytes: 10,
            max_bytes: 40,
            min_alpha_ratio: 0.5,
            max_dup_line_ratio: 0.5,
            max_line_bytes: 30,
            ..Filters::default()
        });
        assert_eq!(f.check(&doc("short")), Some(Reject::TooShort));
        assert_eq!(f.check(&doc(&"x".repeat(50))), Some(Reject::TooLong));
        assert_eq!(f.check(&doc("!!!! ?? ;;;; ++++ ==== &&&&")), Some(Reject::LowAlpha));
        assert_eq!(f.check(&doc("aaa\naaa\naaa\naaa")), Some(Reject::RepeatedLines));
        assert_eq!(f.check(&doc("hello there world")), None);

        let mut f = Filter::new(Filters {
            max_line_bytes: 30,
            ..Filters::default()
        });
        assert_eq!(
            f.check(&doc(&format!("{}\nsecond line", "y".repeat(31)))),
            Some(Reject::LongLine)
        );
    }
}
