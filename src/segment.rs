// Word segmentation for the hierarchical model.
//
// A "word" is the unit the backbone autoregresses over: the encoder compresses
// its bytes into one embedding, the decoder spells it back out. So the split
// decides what the backbone gets to reason about. The old rule — a word ends at
// a boundary *byte* (space, `.`, `(`, …) — falls apart on code: `foo::bar(x)`
// becomes `foo:`, `:`, `bar(`, `x)`, and a 4-space indent becomes four words.
//
// This is a lexer-shaped split instead, close to how rustc tokenizes:
//
//   - a run of whitespace is one unit, and attaches as a *suffix* to the word
//     it follows (`"use "` is one word, and the decoder emits its trailing
//     space before `[W]`) — so a word carries the separator that closes it,
//     and the backbone step count stays near one per real token instead of
//     doubling it. A whitespace run that is long on its own (or that opens the
//     input) stands alone.
//   - identifiers and keywords are one word: `[A-Za-z_][A-Za-z0-9_]*`. Non-ASCII
//     bytes count as identifier bytes, so a UTF-8 character in a string or
//     comment stays inside one word instead of splitting into stray bytes.
//   - numbers are one word, including suffixes and decimal points: `1_000u32`,
//     `3.14`, `0xFF` — but `1..5` still splits at the range operator.
//   - multi-byte operators are one word: `::`, `->`, `=>`, `==`, `..=`, `<<=`,
//     `//`, `/*`, … Splitting `::` into two `:` words would make the backbone
//     predict a path separator in two unrelated steps.
//   - a lifetime (`'a`) is one word, while a char literal (`'a'`) keeps its
//     quotes as separate words.
//   - anything else (a single delimiter, a stray byte) is its own word.
//
// Words tile the token sequence contiguously — `word_ends` returns exclusive
// ends only, starts are implied — and no word is longer than `MAX_WORD_BYTES`,
// which bounds the decoder unroll.

use crate::config::MAX_WORD_BYTES;
use crate::tokenizer_utf8::BYTE_TOKENS;

/// Whitespace runs longer than this are not glued onto the preceding word —
/// a blank-line gap or a big indent block is its own unit. Sized to cover a
/// newline plus three levels of Rust indentation (`\n` + 12 spaces), so a
/// line break and the next line's indent close the word before them.
const MAX_WS_SUFFIX: usize = 16;

/// Multi-byte operators, longest first (the scanner takes the first match).
const OPERATORS: &[&[u8]] = &[
    b"..=", b"...", b"<<=", b">>=", b"///", b"//!", b"#![", b"::", b"->", b"=>", b"==", b"!=",
    b"<=", b">=", b"&&", b"||", b"+=", b"-=", b"*=", b"/=", b"%=", b"^=", b"&=", b"|=", b"<<",
    b">>", b"..", b"//", b"/*", b"*/", b"#[",
];

#[inline]
fn byte_of(tok: u16) -> Option<u8> {
    ((tok as usize) < BYTE_TOKENS).then_some(tok as u8)
}

#[inline]
fn is_ws(tok: u16) -> bool {
    matches!(byte_of(tok), Some(b' ' | b'\t' | b'\n' | b'\r'))
}

/// Identifier byte: ASCII alphanumeric, `_`, or any non-ASCII byte (so the
/// bytes of one UTF-8 character never split across words).
#[inline]
fn is_ident(tok: u16) -> bool {
    match byte_of(tok) {
        Some(b) => b.is_ascii_alphanumeric() || b == b'_' || b >= 0x80,
        None => false,
    }
}

#[inline]
fn is_ident_start(tok: u16) -> bool {
    is_ident(tok) && !matches!(byte_of(tok), Some(b'0'..=b'9'))
}

#[inline]
fn is_digit(tok: u16) -> bool {
    matches!(byte_of(tok), Some(b'0'..=b'9'))
}

/// Length of the operator matching at `i`, if any.
fn operator_len(seq: &[u16], i: usize) -> Option<usize> {
    OPERATORS.iter().find_map(|op| {
        let end = i + op.len();
        (end <= seq.len()
            && seq[i..end]
                .iter()
                .zip(*op)
                .all(|(&t, &b)| byte_of(t) == Some(b)))
        .then_some(op.len())
    })
}

/// Scan one core word (no leading whitespace) starting at `i`. Always advances.
fn core_end(seq: &[u16], i: usize) -> usize {
    let tok = seq[i];

    if is_ident_start(tok) {
        let mut e = i + 1;
        while e < seq.len() && is_ident(seq[e]) {
            e += 1;
        }
        return e;
    }

    if is_digit(tok) {
        // Digits, suffixes and separators (`0xFF`, `1_000u32`), plus a `.` only
        // when a digit follows — so `1..5` and `x.0` keep their operators.
        let mut e = i + 1;
        while e < seq.len() {
            if is_ident(seq[e]) {
                e += 1;
            } else if byte_of(seq[e]) == Some(b'.') && e + 1 < seq.len() && is_digit(seq[e + 1]) {
                e += 2;
            } else {
                break;
            }
        }
        return e;
    }

    // Lifetime `'a`: a quote, an identifier, and no closing quote (that would
    // be a char literal, whose quotes we keep separate).
    if byte_of(tok) == Some(b'\'') && i + 1 < seq.len() && is_ident_start(seq[i + 1]) {
        let mut e = i + 1;
        while e < seq.len() && is_ident(seq[e]) {
            e += 1;
        }
        if e >= seq.len() || byte_of(seq[e]) != Some(b'\'') {
            return e;
        }
    }

    if let Some(len) = operator_len(seq, i) {
        return i + len;
    }

    i + 1
}

/// Segment `seq` into words, returning exclusive ends: word `i` spans
/// `ends[i-1]..ends[i]` (`0..ends[0]` for the first). See the module docs.
pub fn word_ends(seq: &[u16]) -> Vec<u32> {
    let n = seq.len();
    let mut ends: Vec<u32> = Vec::new();
    let mut i = 0;

    while i < n {
        let start = i;

        // A whitespace run with no word before it (start of input, or the tail
        // of an over-long run) is a word of its own.
        if is_ws(seq[i]) {
            let mut ws = i;
            while ws < n && is_ws(seq[ws]) {
                ws += 1;
            }
            push_capped(&mut ends, i, ws);
            i = ws;
            continue;
        }

        let core = core_end(seq, i);

        // Absorb the whitespace that follows, so the word ends with its own
        // separator. An over-long run is left to stand alone.
        let mut end = core;
        while end < n && is_ws(seq[end]) && end - core < MAX_WS_SUFFIX {
            end += 1;
        }
        if end < n && is_ws(seq[end]) {
            end = core; // run is longer than the cap — none of it belongs here
        }

        push_capped(&mut ends, start, end);
        i = end;
    }

    ends
}

/// Push word `start..end`, chopping it into `MAX_WORD_BYTES` pieces if it is
/// longer (a huge identifier, a base64 blob, an indent block).
fn push_capped(ends: &mut Vec<u32>, start: usize, end: usize) {
    let mut s = start;
    while end - s > MAX_WORD_BYTES {
        s += MAX_WORD_BYTES;
        ends.push(s as u32);
    }
    ends.push(end as u32);
}

/// Split a sampling prefix into the words that are certainly complete and the
/// trailing partial word. The last word is always treated as unfinished — the
/// user may be mid-identifier (`fn ma`) — so it is teacher-forced into the
/// decoder rather than encoded into the backbone.
pub fn split_prefix(prefix: &[u16]) -> (Vec<&[u16]>, &[u16]) {
    let ends = word_ends(prefix);
    let Some((&last, complete)) = ends.split_last() else {
        return (Vec::new(), &[]);
    };
    debug_assert_eq!(last as usize, prefix.len());

    let mut words = Vec::with_capacity(complete.len());
    let mut start = 0;
    for &e in complete {
        words.push(&prefix[start..e as usize]);
        start = e as usize;
    }
    (words, &prefix[start..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer_utf8::Utf8Tokenizer;

    /// Segment `text` and render the words back as strings.
    fn words(text: &str) -> Vec<String> {
        let tok = Utf8Tokenizer::new();
        let seq = tok.to_tokens(text);
        let mut out = Vec::new();
        let mut start = 0;
        for e in word_ends(&seq) {
            out.push(tok.to_text(&seq[start..e as usize]));
            start = e as usize;
        }
        out
    }

    #[test]
    fn words_tile_the_input_exactly() {
        let src = "fn main() {\n    let x: Vec<u8> = vec![1, 2];\n}\n";
        assert_eq!(words(src).concat(), src);
    }

    #[test]
    fn identifiers_paths_and_operators_stay_whole() {
        assert_eq!(
            words("foo::bar(x) -> u32"),
            ["foo", "::", "bar", "(", "x", ") ", "-> ", "u32"]
        );
    }

    #[test]
    fn whitespace_closes_the_word_it_follows() {
        assert_eq!(
            words("fn f() {\n    let y = 1;\n}"),
            [
                "fn ", "f", "(", ") ", "{\n    ", "let ", "y ", "= ", "1", ";\n", "}"
            ]
        );
    }

    #[test]
    fn numbers_keep_suffixes_but_not_ranges() {
        assert_eq!(words("1_000u32"), ["1_000u32"]);
        assert_eq!(words("3.14"), ["3.14"]);
        assert_eq!(words("0..=9"), ["0", "..=", "9"]);
    }

    #[test]
    fn lifetimes_differ_from_char_literals() {
        assert_eq!(words("&'a str"), ["&", "'a ", "str"]);
        assert_eq!(words("'a'"), ["'", "a", "'"]);
    }

    #[test]
    fn multibyte_chars_stay_inside_one_word() {
        assert_eq!(words("\"Größe\""), ["\"", "Größe", "\""]);
    }

    #[test]
    fn no_word_exceeds_the_cap() {
        let long = "x".repeat(MAX_WORD_BYTES * 3 + 5);
        let text = format!("let {long} = 1;");
        assert!(words(&text).iter().all(|w| w.len() <= MAX_WORD_BYTES));
        assert_eq!(words(&text).concat(), text);
    }

    #[test]
    fn prefix_splits_off_the_partial_word() {
        let tok = Utf8Tokenizer::new();
        let seq = tok.to_tokens("fn ma");
        let (complete, tail) = split_prefix(&seq);
        assert_eq!(complete.len(), 1);
        assert_eq!(tok.to_text(complete[0]), "fn ");
        assert_eq!(tok.to_text(tail), "ma");
    }
}
