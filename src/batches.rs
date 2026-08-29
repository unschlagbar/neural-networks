// Word-grouped corpus loading for the hierarchical model: stream the corpus in
// bounded chunks, tokenize each chunk and cut it into fixed-size K-word windows.

use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    mem,
    range::Range,
};

use rand::{rng, seq::SliceRandom};

use crate::{
    config::{ALLOWED_LANGUAGES, PARQUET_LANGUAGE_COLUMN},
    parquet::ParquetColumnReader,
    segment,
    tokenizer_utf8::Utf8Tokenizer,
};

const SPLIT: &str = "<|endoftext|>";

/// Column read from a parquet corpus. FineWeb-style dumps put the document body
/// in `text`; override with the `PARQUET_TEXT_COLUMN` env var.
const PARQUET_TEXT_COLUMN: &str = "text";
//const SPLIT: &str = "---FILE---";

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

/// Where raw documents come from. Both variants hand the loader the same thing
/// — a batch of *complete* documents — so everything downstream (tokenizing,
/// windowing, the guarantee that windows never cross a document border) is
/// shared. A `.parquet` path selects `Parquet`, anything else `Text`.
enum TextSource {
    /// Plain text with `<|endoftext|>` separators, read `chunk_bytes` at a time
    /// and cut at the last complete document.
    Text {
        reader: BufReader<File>,
        /// Bytes after the last complete document of the previous read — they
        /// become the prefix of the next chunk.
        carry: Vec<u8>,
        eof: bool,
    },
    /// One BYTE_ARRAY column, one row per document. Row groups are already
    /// document-aligned, so no carry and no separator scanning is needed; a
    /// chunk is however many row groups it takes to reach `chunk_bytes`.
    Parquet {
        reader: ParquetColumnReader,
        /// Whether a second (language) column is being read alongside the text.
        /// False when filtering is off or the corpus has no language column.
        filter_language: bool,
    },
}

/// Streaming loader: gathers roughly `chunk_bytes` of raw text at a time from
/// the source and hands out ready-to-train `WordChunk`s.
pub struct ChunkedWordDataSet {
    tokenizer: Utf8Tokenizer,
    words_per_seq: usize,
    min_words: usize,
    max_tokens: usize,
    chunk_bytes: usize,
    source: TextSource,
    /// Suppress the per-chunk summary line (used by counting passes).
    quiet: bool,
}

impl ChunkedWordDataSet {
    pub fn open(
        tokenizer: Utf8Tokenizer,
        path: &str,
        words_per_seq: usize,
        min_words: usize,
        max_tokens: usize,
        chunk_bytes: usize,
    ) -> Self {
        assert!(words_per_seq >= 2, "words_per_seq must be >= 2");
        assert!(chunk_bytes > SPLIT.len(), "chunk_bytes is too small");

        let source = if path.rsplit('.').next() == Some("parquet") {
            let column = std::env::var("PARQUET_TEXT_COLUMN")
                .unwrap_or_else(|_| PARQUET_TEXT_COLUMN.to_string());

            // Read the language column alongside the text when filtering is on.
            // A corpus without that column is not an error — it just means there
            // is nothing to filter on, so fall back to reading the text alone.
            let (reader, filter_language) = if ALLOWED_LANGUAGES.is_empty() {
                (None, false)
            } else {
                match ParquetColumnReader::open_columns(path, &[&column, PARQUET_LANGUAGE_COLUMN]) {
                    Ok(r) => (Some(r), true),
                    Err(e) => {
                        println!(
                            "  parquet: no usable {PARQUET_LANGUAGE_COLUMN:?} column ({e}) \
                             — keeping every document"
                        );
                        (None, false)
                    }
                }
            };
            let reader = reader.unwrap_or_else(|| {
                ParquetColumnReader::open(path, &column)
                    .unwrap_or_else(|e| panic!("could not open parquet corpus: {e}"))
            });

            if filter_language {
                println!(
                    "  parquet: {} rows in {} row groups, reading column {column:?}, \
                     keeping languages {ALLOWED_LANGUAGES:?}",
                    reader.num_rows(),
                    reader.num_row_groups(),
                );
            } else {
                println!(
                    "  parquet: {} rows in {} row groups, reading column {column:?}",
                    reader.num_rows(),
                    reader.num_row_groups(),
                );
            }
            TextSource::Parquet {
                reader,
                filter_language,
            }
        } else {
            let file = File::open(path).unwrap_or_else(|e| panic!("could not open {path:?}: {e}"));
            TextSource::Text {
                reader: BufReader::new(file),
                carry: Vec::new(),
                eof: false,
            }
        };

        Self {
            tokenizer,
            words_per_seq,
            min_words,
            max_tokens,
            chunk_bytes,
            source,
            quiet: false,
        }
    }

    /// Seek back to the start of the corpus. Call before every epoch.
    pub fn rewind(&mut self) {
        match &mut self.source {
            TextSource::Text { reader, carry, eof } => {
                reader
                    .seek(SeekFrom::Start(0))
                    .expect("could not seek corpus file");
                carry.clear();
                *eof = false;
            }
            TextSource::Parquet { reader, .. } => reader.rewind(),
        }
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
            let sequences = match &mut self.source {
                TextSource::Text { .. } => self.next_text_sequences()?,
                TextSource::Parquet { .. } => self.next_parquet_sequences()?,
            };

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

    /// Text source: read up to `chunk_bytes`, cut at the last complete document,
    /// carry the tail, and tokenize each document. `None` at end of file.
    fn next_text_sequences(&mut self) -> Option<Vec<Vec<u16>>> {
        let chunk_bytes = self.chunk_bytes;
        let TextSource::Text { reader, carry, eof } = &mut self.source else {
            unreachable!("next_text_sequences on a non-text source");
        };

        if *eof && carry.is_empty() {
            return None;
        }

        let mut buf = mem::take(carry);

        // Fill: normally one chunk-sized read. Keep growing only when a
        // single document is larger than the chunk (no separator yet).
        let mut last_split = None;
        while !*eof {
            let want = if buf.len() < chunk_bytes {
                chunk_bytes - buf.len()
            } else {
                last_split = find_last(&buf, SPLIT.as_bytes());
                if last_split.is_some() {
                    break;
                }
                chunk_bytes
            };
            let got = reader
                .take(want as u64)
                .read_to_end(&mut buf)
                .unwrap_or_else(|e| panic!("could not read corpus file: {e}"));
            if got < want {
                *eof = true;
            }
        }

        // Cut right after the last separator; the tail is carried into the
        // next chunk. The separator is ASCII, so the cut always lands on a
        // UTF-8 boundary. At EOF the whole rest is the final chunk.
        let cut = if *eof {
            buf.len()
        } else {
            last_split.expect("fill loop guarantees a separator") + SPLIT.len()
        };
        carry.extend_from_slice(&buf[cut..]);
        buf.truncate(cut);
        let text = String::from_utf8(buf).expect("corpus is not valid UTF-8");

        let mut sequences: Vec<Vec<u16>> = Vec::new();
        for doc in text.split(SPLIT) {
            let doc = doc.trim();
            if doc.is_empty() {
                continue;
            }
            let toks = self.tokenizer.to_tokens_markup(doc);
            if toks.len() >= 2 {
                sequences.push(toks);
            }
        }
        Some(sequences)
    }

    /// Parquet source: pull row groups until roughly `chunk_bytes` of *kept*
    /// text has accumulated. Each row is already one complete document, so
    /// there is no separator to find and nothing to carry across chunks. `None`
    /// once the last row group has been consumed.
    ///
    /// When language filtering is on, the language column decodes alongside the
    /// text and rows whose code is not in `ALLOWED_LANGUAGES` are dropped here —
    /// before tokenizing, which is the expensive part.
    fn next_parquet_sequences(&mut self) -> Option<Vec<Vec<u16>>> {
        let chunk_bytes = self.chunk_bytes;
        let mut sequences: Vec<Vec<u16>> = Vec::new();
        let mut bytes = 0usize;
        let mut exhausted = false;

        while bytes < chunk_bytes {
            let TextSource::Parquet {
                reader,
                filter_language,
            } = &mut self.source
            else {
                unreachable!("next_parquet_sequences on a non-parquet source");
            };
            let filter_language = *filter_language;

            let group = reader
                .next_row_group_columns()
                .unwrap_or_else(|e| panic!("could not read parquet corpus: {e}"));
            let Some(mut group) = group else {
                exhausted = true;
                break;
            };

            // Columns come back in the order requested: text first, language
            // second when we asked for it.
            let languages = if filter_language { group.pop() } else { None };
            let texts = group.swap_remove(0);

            for (i, raw) in texts.into_iter().enumerate() {
                if let Some(langs) = &languages {
                    // A row whose language is missing or not allowed is skipped.
                    // Missing means the columns fell out of alignment (a null),
                    // in which case dropping is the safe choice.
                    let keep = langs
                        .get(i)
                        .map(|l| ALLOWED_LANGUAGES.iter().any(|a| a.as_bytes() == &l[..]))
                        .unwrap_or(false);
                    if !keep {
                        continue;
                    }
                }

                bytes += raw.len();
                // A corpus row that is not valid UTF-8 is a broken document,
                // not a broken file — drop it and keep streaming.
                let Ok(doc) = String::from_utf8(raw) else {
                    continue;
                };
                let doc = doc.trim();
                if doc.is_empty() {
                    continue;
                }
                let toks = self.tokenizer.to_tokens_markup(doc);
                if toks.len() >= 2 {
                    sequences.push(toks);
                }
            }
        }

        if sequences.is_empty() && exhausted {
            // Exhausted: no more row groups left to read.
            return None;
        }
        Some(sequences)
    }
}

/// Byte offset of the last occurrence of `needle` in `haystack`.
fn find_last(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).rposition(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

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
        let tokenizer = Utf8Tokenizer::new();

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
            let mut l = ChunkedWordDataSet::open(tokenizer, path, 6, 2, 64, chunk_bytes);
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
