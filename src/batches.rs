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

use std::{fs, path::PathBuf};

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
            let mut idx = 0;
            loop {
                let remaining = seq.len().saturating_sub(idx);
                if remaining < 2 {
                    break;
                }

                let mut end = (idx + seq_len).min(seq.len());
                while end < seq.len() && !boundary_ids.contains(&seq[end - 1]) {
                    end += 1;
                }

                out.push(Window {
                    seq: s_idx as u32,
                    start: idx as u32,
                    end: end as u32,
                });
                idx = end;
            }
        }
        out
    }
}
