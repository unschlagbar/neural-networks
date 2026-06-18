// PreparedDataSet — load + tokenize + window the entire corpus *once* up front
// and reuse it across every epoch. Shuffling reorders only the cheap window
// index list; the actual token sequences stay put, so we never reread or
// retokenize a file during training.
//
// One window = one (input, target) pair sliced from one tokenized file with
// the same word-boundary logic the old `WordBoundaryBatches` used:
//   - start at `index`, take `seq_len` tokens
//   - extend the right edge until the last token is a boundary token
//   - input  = data[start..end-1]
//   - target = data[start+1..end]
//
// Why store indices instead of `&[u16]`s? A `Vec<&[u16]>` pointing into a
// sibling `Vec<Vec<u16>>` field would make the struct self-referential, which
// Rust's borrow checker forbids. Twelve bytes (3 × u32) per window is cheap —
// even a million windows is only ~12 MB.

use std::{fs, path::PathBuf, range::Range};

use rand::{rng, seq::SliceRandom};

use crate::tokenizer::Tokenizer;

#[derive(Clone, Copy, Debug)]
struct Window {
    seq: u32,
    start: u32,
    end: u32,
}

pub struct PreparedDataSet {
    /// One Vec<u16> per source file, in original file order. Never moved
    /// after construction — windows hold stable indices into this array.
    sequences: Vec<Vec<u16>>,
    /// Every (input, target) window across the whole corpus, flattened.
    /// `shuffle()` reorders this list; iteration walks it in order.
    windows: Vec<Window>,
}

impl PreparedDataSet {
    pub fn from_single_file(
        tokenizer: &Tokenizer,
        path: &str,
        seq_len: usize,
        boundary_ids: &[u16],
    ) -> Self {
        let content =
            fs::read_to_string(path).unwrap_or_else(|e| panic!("konnte {path:?} nicht lesen: {e}"));

        let mut sequences: Vec<Vec<u16>> = Vec::new();
        let mut skipped = 0;

        for chunk in content.split("---FILE---") {
            let chunk = chunk.trim();
            if chunk.is_empty() {
                continue;
            }
            let toks = tokenizer.to_tokens(chunk);
            if toks.len() >= 2 {
                sequences.push(toks);
            } else {
                skipped += 1;
            }
        }

        if skipped > 0 {
            eprintln!("PreparedDataSet: {skipped} leere Chunks übersprungen");
        }

        println!(
            "  {} Chunks geladen, {} Tokens gesamt",
            sequences.len(),
            sequences.iter().map(|s| s.len()).sum::<usize>(),
        );

        let windows = build_windows(&sequences, seq_len, boundary_ids);
        Self { sequences, windows }
    }
    /// Read every file directly inside `dir` (non-recursive), tokenize it, and
    /// pre-compute all windows. Files that fail to read or are too short are
    /// skipped with a warning.
    pub fn from_dir(
        tokenizer: &Tokenizer,
        dir: &str,
        seq_len: usize,
        boundary_ids: &[u16],
    ) -> Self {
        let paths: Vec<PathBuf> = fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("could not read dir {dir:?}: {e}"))
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_file())
            .collect();
        Self::from_paths(tokenizer, &paths, seq_len, boundary_ids)
    }

    /// Build a `PreparedDataSet` from an explicit file list. Use this when you
    /// want recursive collection (gather paths yourself, then pass them in).
    pub fn from_paths(
        tokenizer: &Tokenizer,
        paths: &[PathBuf],
        seq_len: usize,
        boundary_ids: &[u16],
    ) -> Self {
        assert!(seq_len >= 2, "seq_len must be >= 2");

        let mut sequences: Vec<Vec<u16>> = Vec::with_capacity(paths.len());
        let mut skipped = 0;
        for path in paths {
            match fs::read_to_string(path) {
                Ok(content) => {
                    let toks = tokenizer.to_tokens(&content);
                    if toks.len() >= 2 {
                        sequences.push(toks);
                    } else {
                        skipped += 1;
                    }
                }
                Err(_) => skipped += 1,
            }
        }
        if skipped > 0 {
            eprintln!("PreparedDataSet: skipped {skipped} unreadable/empty file(s)");
        }

        let windows = build_windows(&sequences, seq_len, boundary_ids);
        let this = Self { sequences, windows };

        //this.iter().skip(2).take(2).for_each(|t| {
        //    println!(
        //        "input: \"{}\"\n output: \"{}\"",
        //        tokenizer.display_tokens(t.0),
        //        tokenizer.display_tokens(t.1)
        //    )
        //});

        this
    }

    /// Reorder the window list in place. Sequences themselves are untouched —
    /// only the iteration order over their windows changes. Call this before
    /// each epoch for fresh shuffling without rebuilding anything.
    pub fn shuffle(&mut self) {
        self.windows.shuffle(&mut rng());
    }

    pub fn len(&self) -> usize {
        self.windows.len()
    }
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }
    pub fn num_sequences(&self) -> usize {
        self.sequences.len()
    }
    pub fn total_tokens(&self) -> usize {
        self.sequences.iter().map(|s| s.len()).sum()
    }

    /// Iterate (input, target) pairs in current shuffle order. Borrows `self`,
    /// so the dataset has to outlive the iterator — which is exactly what we
    /// want during a training pass.
    pub fn iter(&self) -> PreparedIter<'_> {
        PreparedIter { ds: self, idx: 0 }
    }
}

pub struct PreparedIter<'a> {
    ds: &'a PreparedDataSet,
    idx: usize,
}

impl<'a> Iterator for PreparedIter<'a> {
    type Item = (&'a [u16], &'a [u16]);

    fn next(&mut self) -> Option<Self::Item> {
        let w = *self.ds.windows.get(self.idx)?;
        self.idx += 1;
        let seq = &self.ds.sequences[w.seq as usize];
        let s = w.start as usize;
        let e = w.end as usize;
        Some((&seq[s..e - 1], &seq[s + 1..e]))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.ds.windows.len().saturating_sub(self.idx);
        (rem, Some(rem))
    }
}

impl ExactSizeIterator for PreparedIter<'_> {}

// Same semantics as the old WordBoundaryBatches: take `seq_len` tokens, then
// extend the right edge until the last token is a boundary token. The cut
// position becomes the start of the next window — no overlap.

fn build_windows(sequences: &[Vec<u16>], seq_len: usize, boundary_ids: &[u16]) -> Vec<Window> {
    if boundary_ids.is_empty() {
        let mut out = Vec::new();
        for (s_idx, seq) in sequences.iter().enumerate() {
            let mut idx = 0;
            loop {
                let remaining = seq.len().saturating_sub(idx);
                if remaining < 2 {
                    break;
                }

                let end = (idx + seq_len).min(seq.len());

                out.push(Window {
                    seq: s_idx as u32,
                    start: idx as u32,
                    end: end as u32,
                });
                idx = end;
            }
        }
        out
    } else {
        let mut out = Vec::new();
        for (s_idx, seq) in sequences.iter().enumerate() {
            let seq_windows_start = out.len();
            let mut window_start = 0;
            'seq: loop {
                if seq.len().saturating_sub(window_start) < 2 {
                    break;
                }

                let mut window_end = (window_start + seq_len).min(seq.len());
                let boundary_search_start = window_end;
                while window_end < seq.len() && !boundary_ids.contains(&seq[window_end - 1]) {
                    window_end += 1;
                    if window_end - boundary_search_start > 127 {
                        // no boundary found — discard all windows from this sequence
                        out.truncate(seq_windows_start);
                        break 'seq;
                    }
                }

                out.push(Window {
                    seq: s_idx as u32,
                    start: window_start as u32,
                    end: window_end as u32,
                });
                window_start = window_end;
            }
        }
        out
    }
}

// ── Word-grouped dataset (hierarchical training) ────────────────────────────
//
// Instead of fixed-token windows, group the corpus into words once up front and
// emit fixed-size *K-word* sequences. Every sample then unrolls the backbone for
// the same number of word steps. Token counts still vary per window, so a window
// is closed early if it would exceed `max_tokens` (the token-cache cap).

/// Split `seq` into `[start, end)` word ranges. A word ends at the first boundary
/// token (the boundary is its last element); any trailing non-boundary chars form
/// a final word. Same rule the old `Hierarchical::segment_words` used.
fn segment_words(seq: &[u16], boundary_ids: &[u16]) -> Vec<Range<usize>> {
    let mut words = Vec::new();
    let mut start = 0;
    for (t, tok) in seq.iter().enumerate() {
        if boundary_ids.contains(tok) {
            words.push(Range { start, end: t + 1 });
            start = t + 1;
        }
    }
    if start < seq.len() {
        words.push(Range {
            start,
            end: seq.len(),
        });
    }
    words
}

#[derive(Clone, Copy, Debug)]
struct WordWindow {
    seq: u32,
    word_start: u32,
    word_count: u32,
}

pub struct WordDataSet {
    /// One Vec<u16> per source chunk, in original order. Never moved after
    /// construction — windows hold stable indices into this array.
    sequences: Vec<Vec<u16>>,
    /// Per-sequence word ranges (absolute token positions into `sequences[seq]`).
    segments: Vec<Vec<Range<usize>>>,
    /// Every K-word window across the corpus. `shuffle()` reorders this list.
    windows: Vec<WordWindow>,
    /// Token span of the longest window. Callers size their caches to exactly
    /// this — no guessing, no waste.
    max_window_tokens: usize,
}

impl WordDataSet {
    /// Load a single file (split on `---FILE---`), tokenize and word-segment each
    /// chunk, then build contiguous windows of up to `words_per_seq` words. A
    /// window is closed early if adding the next word would push its token span
    /// past `max_tokens`. A finished window is kept only if it has at least
    /// `min_words` words (so the trailing remnant of a chunk is dropped when too
    /// short, and there is always at least one decoded word).
    pub fn from_single_file(
        tokenizer: &Tokenizer,
        path: &str,
        words_per_seq: usize,
        min_words: usize,
        max_tokens: usize,
        boundary_ids: &[u16],
    ) -> Self {
        assert!(words_per_seq >= 2, "words_per_seq must be >= 2");

        let content =
            fs::read_to_string(path).unwrap_or_else(|e| panic!("konnte {path:?} nicht lesen: {e}"));

        let mut sequences: Vec<Vec<u16>> = Vec::new();
        let mut skipped = 0;
        for chunk in content.split("---FILE---") {
            let chunk = chunk.trim();
            if chunk.is_empty() {
                continue;
            }
            let toks = tokenizer.to_tokens(chunk);
            if toks.len() >= 2 {
                sequences.push(toks);
            } else {
                skipped += 1;
            }
        }
        if skipped > 0 {
            eprintln!("WordDataSet: {skipped} leere Chunks übersprungen");
        }

        let segments: Vec<Vec<Range<usize>>> = sequences
            .iter()
            .map(|seq| segment_words(seq, boundary_ids))
            .collect();

        let (windows, max_window_tokens) =
            build_word_windows(&segments, words_per_seq, min_words, max_tokens);

        let total_words: usize = windows.iter().map(|w| w.word_count as usize).sum();
        let avg_words = total_words as f32 / windows.len().max(1) as f32;
        println!(
            "  {} Chunks, {} Tokens, {} Wörter, {} Fenster (Ziel {} Wörter, Ø {:.0} Wörter / max {} Tokens je Fenster)",
            sequences.len(),
            sequences.iter().map(|s| s.len()).sum::<usize>(),
            segments.iter().map(|s| s.len()).sum::<usize>(),
            windows.len(),
            words_per_seq,
            avg_words,
            max_window_tokens,
        );

        Self {
            sequences,
            segments,
            windows,
            max_window_tokens,
        }
    }

    /// Token span of the longest window — size training caches to this.
    pub fn max_window_tokens(&self) -> usize {
        self.max_window_tokens
    }

    /// Reorder the window list in place. Sequences themselves are untouched.
    pub fn shuffle(&mut self) {
        self.windows.shuffle(&mut rng());
    }

    pub fn len(&self) -> usize {
        self.windows.len()
    }
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    pub fn iter(&self) -> WordIter<'_> {
        WordIter { ds: self, idx: 0 }
    }
}

/// One training sample: the window's contiguous tokens plus its word ranges
/// (relative to `tokens`). `words` is a fresh small Vec (K ranges) per item.
pub struct WordBatch<'a> {
    pub tokens: &'a [u16],
    pub words: Vec<Range<usize>>,
}

pub struct WordIter<'a> {
    ds: &'a WordDataSet,
    idx: usize,
}

impl<'a> Iterator for WordIter<'a> {
    type Item = WordBatch<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let w = *self.ds.windows.get(self.idx)?;
        self.idx += 1;
        let segs = &self.ds.segments[w.seq as usize];
        let first = w.word_start as usize;
        let count = w.word_count as usize;
        let abs_start = segs[first].start;
        let abs_end = segs[first + count - 1].end;
        let tokens = &self.ds.sequences[w.seq as usize][abs_start..abs_end];
        let words: Vec<Range<usize>> = segs[first..first + count]
            .iter()
            .map(|r| Range {
                start: r.start - abs_start,
                end: r.end - abs_start,
            })
            .collect();
        Some(WordBatch { tokens, words })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.ds.windows.len().saturating_sub(self.idx);
        (rem, Some(rem))
    }
}

impl ExactSizeIterator for WordIter<'_> {}

/// Walk each sequence's word ranges contiguously, packing up to `words_per_seq`
/// words per window but never letting the token span exceed `max_tokens` (the
/// first word of a window is always included even if it alone is longer). Keep a
/// window only if it gathered at least `min_words` words.
fn build_word_windows(
    segments: &[Vec<Range<usize>>],
    words_per_seq: usize,
    min_words: usize,
    max_tokens: usize,
) -> (Vec<WordWindow>, usize) {
    let mut out = Vec::new();
    let mut max_span = 0;
    for (s_idx, segs) in segments.iter().enumerate() {
        let n = segs.len();
        let mut wi = 0;
        while wi < n {
            let start_tok = segs[wi].start;
            let mut count = 0;
            while wi + count < n && count < words_per_seq {
                let span = segs[wi + count].end - start_tok;
                if span > max_tokens && count > 0 {
                    break;
                }
                count += 1;
            }
            if count >= min_words {
                let span = segs[wi + count - 1].end - start_tok;
                max_span = max_span.max(span);
                out.push(WordWindow {
                    seq: s_idx as u32,
                    word_start: wi as u32,
                    word_count: count as u32,
                });
            }
            wi += count;
        }
    }
    (out, max_span)
}
