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

use std::{
    fs::{self, File},
    io::{BufReader, Read, Seek, SeekFrom},
    mem,
    path::PathBuf,
    range::Range,
    rc::Rc,
};

use rand::{rng, seq::SliceRandom};

use crate::{segment, tokenizer_utf8::Utf8Tokenizer};

const SPLIT: &str = "<|endoftext|>";
//const SPLIT: &str = "---FILE---";

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
        tokenizer: &Utf8Tokenizer,
        path: &str,
        seq_len: usize,
        boundary_ids: &[u16],
    ) -> Self {
        let content =
            fs::read_to_string(path).unwrap_or_else(|e| panic!("konnte {path:?} nicht lesen: {e}"));

        let mut sequences: Vec<Vec<u16>> = Vec::new();
        let mut skipped = 0;

        for chunk in content.split(SPLIT) {
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
        tokenizer: &Utf8Tokenizer,
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
        tokenizer: &Utf8Tokenizer,
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

// ── Word-grouped dataset (hierarchical + flat word training) ────────────────
//
// Instead of fixed-token windows, group the corpus into words and emit
// fixed-size *K-word* sequences. Every sample then unrolls the backbone for
// the same number of word steps. Token counts still vary per window, so a window
// is closed early if it would exceed `max_tokens` (the token-cache cap).
//
// The corpus is streamed in chunks (`ChunkedWordDataSet`) instead of being
// loaded and tokenized whole: each chunk covers only complete documents (split
// on `SPLIT`), the trailing partial document is carried into the next chunk.
// Windows never cross document borders, so a streamed epoch yields exactly the
// same windows a whole-file load would — but peak memory is bounded by the
// chunk size, not the corpus size (> 1 GB corpora stream fine).

#[derive(Clone, Copy, Debug)]
struct WordWindow {
    seq: u32,
    word_start: u32,
    word_count: u32,
}

/// One streamed chunk of the corpus, fully tokenized and windowed. Everything
/// that used to live on the whole-file `WordDataSet` lives here per chunk.
pub struct WordChunk {
    /// One Vec<u16> per document, in file order. Never moved after
    /// construction — windows hold stable indices into this array.
    sequences: Vec<Vec<u16>>,
    /// Per-sequence word segmentation, ends-only (see `segment_word_ends`).
    segments: Vec<Vec<u32>>,
    /// Every K-word window in this chunk. `shuffle()` reorders this list.
    windows: Vec<WordWindow>,
    /// Token span of the longest window. Callers size their caches to exactly
    /// this — no guessing, no waste.
    max_window_tokens: usize,
}

impl WordChunk {
    fn build(
        sequences: Vec<Vec<u16>>,
        words_per_seq: usize,
        min_words: usize,
        max_tokens: usize,
    ) -> Self {
        let segments: Vec<Vec<u32>> = sequences
            .iter()
            .map(|seq| segment::word_ends(seq))
            .collect();

        let (windows, max_window_tokens) =
            build_word_windows(&segments, words_per_seq, min_words, max_tokens);

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

    pub fn total_tokens(&self) -> usize {
        self.sequences.iter().map(|s| s.len()).sum()
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

/// Streaming loader: reads `chunk_bytes` of raw text at a time, cuts at the
/// last complete document, and hands out ready-to-train `WordChunk`s.
pub struct ChunkedWordDataSet {
    tokenizer: Rc<Utf8Tokenizer>,
    words_per_seq: usize,
    min_words: usize,
    max_tokens: usize,
    chunk_bytes: usize,
    reader: BufReader<File>,
    /// Bytes after the last complete document of the previous read — they
    /// become the prefix of the next chunk.
    carry: Vec<u8>,
    eof: bool,
    /// Suppress the per-chunk summary line (used by counting passes).
    quiet: bool,
}

impl ChunkedWordDataSet {
    pub fn open(
        tokenizer: Rc<Utf8Tokenizer>,
        path: &str,
        words_per_seq: usize,
        min_words: usize,
        max_tokens: usize,
        chunk_bytes: usize,
    ) -> Self {
        assert!(words_per_seq >= 2, "words_per_seq must be >= 2");
        assert!(chunk_bytes > SPLIT.len(), "chunk_bytes is too small");

        let file = File::open(path).unwrap_or_else(|e| panic!("could not open {path:?}: {e}"));

        Self {
            tokenizer,
            words_per_seq,
            min_words,
            max_tokens,
            chunk_bytes,
            reader: BufReader::new(file),
            carry: Vec::new(),
            eof: false,
            quiet: false,
        }
    }

    /// Seek back to the start of the corpus. Call before every epoch.
    pub fn rewind(&mut self) {
        self.reader
            .seek(SeekFrom::Start(0))
            .expect("could not seek corpus file");
        self.carry.clear();
        self.eof = false;
    }

    /// Total window count of the whole corpus, via one streaming pass (memory
    /// stays chunk-bounded). Rewinds before and after. Only needed for resume
    /// arithmetic — a plain epoch never has to know the total in advance.
    pub fn count_windows(&mut self) -> usize {
        self.rewind();
        self.quiet = true;
        let mut n = 0;
        while let Some(chunk) = self.next_chunk() {
            n += chunk.len();
        }
        self.quiet = false;
        self.rewind();
        n
    }

    /// Load, tokenize and window the next chunk. Returns `None` once the file
    /// is exhausted; chunks that yield no windows are skipped transparently.
    pub fn next_chunk(&mut self) -> Option<WordChunk> {
        loop {
            if self.eof && self.carry.is_empty() {
                return None;
            }

            let mut buf = mem::take(&mut self.carry);

            // Fill: normally one chunk-sized read. Keep growing only when a
            // single document is larger than the chunk (no separator yet).
            let mut last_split = None;
            while !self.eof {
                let want = if buf.len() < self.chunk_bytes {
                    self.chunk_bytes - buf.len()
                } else {
                    last_split = find_last(&buf, SPLIT.as_bytes());
                    if last_split.is_some() {
                        break;
                    }
                    self.chunk_bytes
                };
                let got = (&mut self.reader)
                    .take(want as u64)
                    .read_to_end(&mut buf)
                    .unwrap_or_else(|e| panic!("could not read corpus file: {e}"));
                if got < want {
                    self.eof = true;
                }
            }

            // Cut right after the last separator; the tail is carried into the
            // next chunk. The separator is ASCII, so the cut always lands on a
            // UTF-8 boundary. At EOF the whole rest is the final chunk.
            let cut = if self.eof {
                buf.len()
            } else {
                last_split.expect("fill loop guarantees a separator") + SPLIT.len()
            };
            self.carry.extend_from_slice(&buf[cut..]);
            buf.truncate(cut);
            let text = String::from_utf8(buf).expect("corpus is not valid UTF-8");

            let mut sequences: Vec<Vec<u16>> = Vec::new();
            for doc in text.split(SPLIT) {
                let doc = doc.trim();
                if doc.is_empty() {
                    continue;
                }
                let toks = self.tokenizer.to_tokens(doc);
                if toks.len() >= 2 {
                    sequences.push(toks);
                }
            }

            let chunk = WordChunk::build(
                sequences,
                self.words_per_seq,
                self.min_words,
                self.max_tokens,
            );
            if chunk.is_empty() {
                continue;
            }
            if !self.quiet {
                println!(
                    "  chunk: {} docs, {} tokens, {} windows (max span {})",
                    chunk.sequences.len(),
                    chunk.total_tokens(),
                    chunk.len(),
                    chunk.max_window_tokens(),
                );
            }
            return Some(chunk);
        }
    }
}

/// Byte offset of the last occurrence of `needle` in `haystack`.
fn find_last(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).rposition(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect_all(loader: &mut ChunkedWordDataSet) -> Vec<(Vec<u16>, Vec<Range<usize>>)> {
        let mut out = Vec::new();
        while let Some(chunk) = loader.next_chunk() {
            for b in chunk.iter() {
                out.push((b.tokens.to_vec(), b.words.clone()));
            }
        }
        out
    }

    /// Streaming in tiny chunks must yield exactly the windows a whole-file
    /// load produces, and rewinding must reproduce them deterministically.
    #[test]
    fn tiny_chunks_match_whole_file() {
        let tokenizer = Rc::new(Utf8Tokenizer::new());

        // A few hundred small documents so many chunk cuts land mid-file.
        let mut text = String::new();
        for i in 0..300 {
            text.push_str(&format!(
                "story number {i} begins. someone walks, talks and stops! the end?\n"
            ));
            text.push_str(SPLIT);
        }
        let path = std::env::temp_dir().join("chunked_word_dataset_test.txt");
        fs::write(&path, &text).unwrap();
        let path = path.to_str().unwrap();

        let open = |chunk_bytes: usize| {
            let mut l = ChunkedWordDataSet::open(tokenizer.clone(), path, 6, 2, 64, chunk_bytes);
            l.quiet = true;
            l
        };

        let whole = collect_all(&mut open(1 << 30));
        assert!(whole.len() > 100, "test corpus yields too few windows");

        let mut small_loader = open(256);
        let small = collect_all(&mut small_loader);
        assert_eq!(whole, small);

        small_loader.rewind();
        assert_eq!(whole, collect_all(&mut small_loader));

        assert_eq!(small_loader.count_windows(), whole.len());
    }
}

/// One training sample: the window's contiguous tokens plus its word ranges
/// (relative to `tokens`). `words` is a fresh small Vec (K ranges) per item.
pub struct WordBatch<'a> {
    pub tokens: &'a [u16],
    pub words: Vec<Range<usize>>,
}

pub struct WordIter<'a> {
    ds: &'a WordChunk,
    idx: usize,
}

impl<'a> Iterator for WordIter<'a> {
    type Item = WordBatch<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let w = *self.ds.windows.get(self.idx)?;
        self.idx += 1;
        let ends = &self.ds.segments[w.seq as usize];
        let first = w.word_start as usize;
        let count = w.word_count as usize;
        let abs_start = if first == 0 {
            0
        } else {
            ends[first - 1] as usize
        };
        let abs_end = ends[first + count - 1] as usize;
        let tokens = &self.ds.sequences[w.seq as usize][abs_start..abs_end];
        let mut words = Vec::with_capacity(count);
        let mut start = 0;
        for &e in &ends[first..first + count] {
            let end = e as usize - abs_start;
            words.push(Range { start, end });
            start = end;
        }
        Some(WordBatch { tokens, words })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.ds.windows.len().saturating_sub(self.idx);
        (rem, Some(rem))
    }
}

impl ExactSizeIterator for WordIter<'_> {}

/// Walk each sequence's words contiguously, packing up to `words_per_seq`
/// words per window but never letting the token span exceed `max_tokens` (the
/// first word of a window is always included even if it alone is longer). Keep a
/// window only if it gathered at least `min_words` words.
fn build_word_windows(
    segments: &[Vec<u32>],
    words_per_seq: usize,
    min_words: usize,
    max_tokens: usize,
) -> (Vec<WordWindow>, usize) {
    let mut out = Vec::new();
    let mut max_span = 0;
    for (s_idx, ends) in segments.iter().enumerate() {
        let n = ends.len();
        let mut wi = 0;
        while wi < n {
            let start_tok = if wi == 0 { 0 } else { ends[wi - 1] as usize };
            let mut count = 0;
            while wi + count < n && count < words_per_seq {
                let span = ends[wi + count] as usize - start_tok;
                if span > max_tokens && count > 0 {
                    break;
                }
                count += 1;
            }
            if count >= min_words {
                let span = ends[wi + count - 1] as usize - start_tok;
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
