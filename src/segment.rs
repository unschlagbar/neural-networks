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
//   - identifiers and keywords are one word: `[A-Za-z_][A-Za-z0-9_]*`. A
//     non-ASCII character joins the word when it is alphanumeric (`Größe`,
//     `日本語`) and stands alone when it is punctuation (`“`, `—`, `…`), which
//     is what its ASCII twin does. The decision is per *character*, so a UTF-8
//     character never splits into stray bytes either way.
//   - a hyphen or an underscore between identifier bytes closes the word it
//     follows, like a trailing space: `cross-entropy` → `cross-` + `entropy`,
//     `build_hierarchical_model` → `build_` + `hierarchical_` + `model`, so a
//     snake_case name reaches the backbone as its parts. A separator with no
//     identifier before it (`a - b`, `-5`, `_private`) is untouched.
//   - digits come out in PAIRS — `128` → `12` + `8` — since a number is where
//     byte-level spelling carries the meaning, and two digits per backbone step
//     keeps magnitude legible without a step per digit. The digit/letter cut is
//     one-directional: a letter AFTER digits closes the word, so a hex prefix
//     or type suffix stays whole (`0xFF` → `0` + `xFF`, `1u32` → `1` + `u32`),
//     while digits after letters stay in their name (`Vec2`, `u32`).
//   - multi-byte operators are one word: `::`, `->`, `=>`, `==`, `..=`, `<<=`,
//     `//`, `/*`, … Splitting `::` into two `:` words would make the backbone
//     predict a path separator in two unrelated steps.
//   - a run of three or more of the SAME punctuation byte is one word — a
//     divider rule (`--------`, `====`, `****`) is not a stack of operators,
//     and would otherwise cost one backbone step per byte. Two of a kind stays
//     an operator (`--`, `==`, `//`, `::`).
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

/// ASCII identifier byte. Non-ASCII is handled by `non_ascii`, which needs the
/// surrounding bytes to decode a character and so cannot go through here.
#[inline]
fn is_ident(tok: u16) -> bool {
    match byte_of(tok) {
        Some(b) => b.is_ascii_alphanumeric() || b == b'_',
        None => false,
    }
}

#[inline]
fn is_ident_start(tok: u16) -> bool {
    is_ident(tok) && !matches!(byte_of(tok), Some(b'0'..=b'9'))
}

/// Decode the non-ASCII UTF-8 character starting at `i`, returning its byte
/// length and whether it is alphanumeric (a letter or digit in some script).
///
/// A character's bytes must never split across words, but the old rule — every
/// byte `>= 0x80` is an identifier byte — kept them together by making *all*
/// non-ASCII text identifier-like, so typographic punctuation (`“ ” — …`) glued
/// itself onto the neighbouring word. Deciding per character instead keeps
/// `Größe` whole while letting `“` stand alone like `"` does.
fn non_ascii(seq: &[u16], i: usize) -> Option<(usize, bool)> {
    let b0 = byte_of(seq[i])?;
    let len = match b0 {
        0x80..=0xBF => return None, // stray continuation byte
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF7 => 4,
        _ => return None, // ASCII, or invalid lead
    };
    if i + len > seq.len() {
        return None;
    }
    let mut buf = [0u8; 4];
    for (slot, &t) in buf[..len].iter_mut().zip(&seq[i..i + len]) {
        *slot = byte_of(t)?;
    }
    let ch = std::str::from_utf8(&buf[..len]).ok()?.chars().next()?;
    Some((len, ch.is_alphanumeric()))
}

/// Length of the identifier run at `i` (ASCII bytes plus whole non-ASCII
/// alphanumeric characters), or 0 if none starts here.
fn ident_run(seq: &[u16], mut e: usize) -> usize {
    let start = e;
    while e < seq.len() {
        if is_ident(seq[e]) {
            e += 1;
        } else if let Some((len, true)) = non_ascii(seq, e) {
            e += len;
        } else {
            break;
        }
    }
    e - start
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

    // An identifier starts at an ASCII ident-start byte or at a non-ASCII
    // alphanumeric character (`Größe`, `日本語`); non-ASCII punctuation does not.
    if is_ident_start(tok) || matches!(non_ascii(seq, i), Some((_, true))) {
        let mut e = i + ident_run(seq, i).max(1);
        // An underscore BETWEEN identifier bytes closes the word, exactly like
        // the hyphen rule below: `build_hierarchical_model` becomes `build_` +
        // `hierarchical_` + `model`, so the backbone sees a snake_case name as
        // its parts instead of one opaque blob. A leading `_` (`_private`) has
        // no identifier before it and so never enters here; a trailing `_` has
        // no identifier after it and stays glued to its word.
        if let Some(u) = seq[i..e]
            .iter()
            .position(|&t| byte_of(t) == Some(b'_'))
            .filter(|&p| p > 0)
        {
            let mut u = i + u;
            // A run of underscores (`foo__bar`) closes as one separator.
            while u < e && byte_of(seq[u]) == Some(b'_') {
                u += 1;
            }
            if u < e {
                return u;
            }
        }
        // Digits inside the identifier pair off like a standalone number, and a
        // letter after them closes the word (`0xFF` → `0` + `xFF`, `1u32` → `1`
        // + `u32`). Digits FOLLOWING letters do not open a new word — `Vec2` and
        // `u32` are one name — so the leading letters ride along with the run's
        // first pair (`sha256` → `sha25` + `6`).
        if let Some(d) = seq[i..e].iter().position(|&t| is_digit(t)) {
            let d = i + d;
            let run = seq[d..e].iter().take_while(|&&t| is_digit(t)).count();
            if run > 2 || d + run < e {
                return d + run.min(2);
            }
        }
        // Absorb an English contraction / possessive suffix: an apostrophe that
        // FOLLOWS identifier bytes and is itself followed by more identifier
        // bytes stays in the word (`can't`, `they've`, `o'clock`, `it's`).
        // A lifetime (`'a`) is unaffected — it starts WITH the apostrophe, so it
        // never enters this branch. `rock'n'roll` chains through both apostrophes.
        while e + 1 < seq.len() && byte_of(seq[e]) == Some(b'\'') && is_ident(seq[e + 1]) {
            e += 1; // over the apostrophe
            e += ident_run(seq, e);
        }
        // A hyphen BETWEEN identifier bytes closes the word instead of standing
        // alone: `cross-entropy` → `cross-` + `entropy`. Unlike the apostrophe
        // above it does not pull the following run in — a compound's parts are
        // real words the backbone should get separately, and the leading part
        // carries its separator the same way a trailing space does. `a - b`,
        // `-5` and `-> u32` are untouched: none has an identifier byte directly
        // before the `-`.
        if e + 1 < seq.len() && byte_of(seq[e]) == Some(b'-') && is_ident(seq[e + 1]) {
            e += 1;
        }
        return e;
    }

    if is_digit(tok) {
        // A digit run is emitted in PAIRS: `128` → `12` + `8`, `1234` → `12` +
        // `34`. A number is the one place where byte-level spelling carries the
        // meaning, and one backbone step per two digits keeps magnitude legible
        // (the pairs line up with the decimal groups a reader sees) without
        // spending a step per digit. Pairing runs from the left, so an odd
        // count leaves the single digit last.
        //
        // Only the digits pair. A letter after them closes the word, so a hex
        // prefix and a type suffix come out whole on the next entry: `0xFF` →
        // `0` + `xFF`, `1u32` → `1` + `u32`, `3.14` → `3` + `.` + `14`. The cut
        // is one-directional — digits *following* letters stay in their name
        // (`Vec2`, `sha256`), handled in the identifier branch above.
        let run = seq[i..].iter().take_while(|&&t| is_digit(t)).count();
        return i + run.min(2);
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

    // A run of the same punctuation byte is a divider (`--------`, `=====`,
    // `****`, `###`), not a stack of operators. Without this it costs one
    // backbone step per byte — an 80-column rule in a comment would be 80
    // words. Checked BEFORE `operator_len` so the run wins over the `--`/`==`/
    // `///` prefixes it would otherwise be chewed into. Three is the threshold:
    // two is still plausibly an operator (`--`, `==`, `//`), and `..`/`::` must
    // keep their existing meaning.
    if let Some(b) = byte_of(tok).filter(u8::is_ascii_punctuation) {
        let run = seq[i..]
            .iter()
            .take_while(|&&t| byte_of(t) == Some(b))
            .count();
        if run >= 3 {
            return i + run;
        }
    }

    if let Some(len) = operator_len(seq, i) {
        return i + len;
    }

    // A non-ASCII punctuation character (`“`, `—`, `…`) is its own word, but
    // still one word per character — never one per byte.
    if let Some((len, false)) = non_ascii(seq, i) {
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

    /// Digits come out in pairs — a number is where byte-level spelling carries
    /// the meaning, and two digits per backbone step keeps magnitude legible
    /// without spending a step per digit. Pairing runs from the left, so an odd
    /// count leaves the lone digit last.
    #[test]
    fn digits_segment_into_pairs() {
        assert_eq!(words("128"), ["12", "8"]);
        assert_eq!(words("1234"), ["12", "34"]);
        assert_eq!(words("7"), ["7"]);
        assert_eq!(words("1000000"), ["10", "00", "00", "0"]);
        assert_eq!(words("x = 42;"), ["x ", "= ", "42", ";"]);
    }

    /// The digit/letter cut is one-directional: a letter AFTER digits closes the
    /// word, so a hex prefix or a type suffix comes out whole; digits after
    /// letters stay inside the name they belong to.
    #[test]
    fn letters_after_digits_split_but_not_the_reverse() {
        assert_eq!(words("0xFF"), ["0", "xFF"]);
        assert_eq!(words("1u32"), ["1", "u32"]);
        // The `_` rides along with the run's first pair, like the letters in
        // `sha256` do — a lone `_` word would be a wasted backbone step.
        assert_eq!(words("1_000u32"), ["1", "_00", "0", "u32"]);
        assert_eq!(words("3.14"), ["3", ".", "14"]);
        assert_eq!(words("0..=9"), ["0", "..=", "9"]);
        // Digits following letters do not open a new word.
        assert_eq!(words("Vec2"), ["Vec2"]);
        assert_eq!(words("u32"), ["u32"]);
        assert_eq!(words("sha256"), ["sha25", "6"]);
    }

    /// An underscore between identifier bytes closes the word it follows, the
    /// way the hyphen and a trailing space do — a snake_case name reaches the
    /// backbone as its parts rather than as one opaque blob.
    #[test]
    fn underscore_cuts_the_word() {
        assert_eq!(
            words("build_hierarchical_model"),
            ["build_", "hierarchical_", "model"]
        );
        assert_eq!(words("MAX_LEN"), ["MAX_", "LEN"]);
        assert_eq!(words("foo__bar"), ["foo__", "bar"]);
        // A leading underscore has no word before it to close; a trailing one
        // has nothing after it, so both stay glued.
        assert_eq!(words("_private"), ["_private"]);
        assert_eq!(words("x_ = 1"), ["x_ ", "= ", "1"]);
        assert_eq!(words("let x_y = 1;"), ["let ", "x_", "y ", "= ", "1", ";"]);
    }

    #[test]
    fn lifetimes_differ_from_char_literals() {
        assert_eq!(words("&'a str"), ["&", "'a ", "str"]);
        assert_eq!(words("'a'"), ["'", "a", "'"]);
    }

    /// English contractions and possessives stay one word — an apostrophe that
    /// FOLLOWS letters is absorbed, so prose isn't shredded into `can` + `'t`.
    /// Lifetimes (apostrophe FIRST) are unaffected, checked above.
    #[test]
    fn contractions_stay_whole() {
        assert_eq!(words("can't"), ["can't"]);
        assert_eq!(words("they've"), ["they've"]);
        assert_eq!(words("o'clock"), ["o'clock"]);
        assert_eq!(words("rock'n'roll"), ["rock'n'roll"]);
        // A contraction mid-sentence keeps its trailing space suffix.
        assert_eq!(words("it's fine"), ["it's ", "fine"]);
    }

    #[test]
    fn multibyte_chars_stay_inside_one_word() {
        assert_eq!(words("\"Größe\""), ["\"", "Größe", "\""]);
        assert_eq!(words("日本語"), ["日本語"]);
    }

    /// A divider rule is one word, not one word per byte — an 80-column `---`
    /// line used to cost 80 backbone steps. Real operators are unaffected: the
    /// run rule needs three or more of the SAME byte.
    #[test]
    fn punctuation_runs_are_one_word() {
        assert_eq!(words("--------"), ["--------"]);
        assert_eq!(words("===="), ["===="]);
        assert_eq!(words("****"), ["****"]);
        assert_eq!(words("###"), ["###"]);
        assert_eq!(words("// ------- x"), ["// ", "------- ", "x"]);
        // Two of a kind is still an operator, and mixed runs stay split.
        assert_eq!(words("a..=b"), ["a", "..=", "b"]);
        assert_eq!(words("x::y"), ["x", "::", "y"]);
        assert_eq!(words("/// doc"), ["/// ", "doc"]);
        assert_eq!(words("a // c"), ["a ", "// ", "c"]);
        assert_eq!(words("0..9"), ["0", "..", "9"]);
        assert_eq!(words("x <<= 2"), ["x ", "<<= ", "2"]);
        assert_eq!(words("#![allow]"), ["#![", "allow", "]"]);
        // A rule longer than the cap still splits, and still tiles exactly.
        let rule = "-".repeat(80);
        assert!(words(&rule).iter().all(|w| w.len() <= MAX_WORD_BYTES));
        assert_eq!(words(&rule).concat(), rule);
    }

    /// A hyphen between identifier bytes closes the word it follows, the way a
    /// trailing space does — so a compound splits into its parts instead of
    /// spending a whole backbone step on a lone `-`. Arithmetic and operator
    /// minus are unaffected.
    #[test]
    fn hyphen_sticks_to_the_word_before_it() {
        assert_eq!(words("cross-entropy"), ["cross-", "entropy"]);
        assert_eq!(words("well-known word"), ["well-", "known ", "word"]);
        assert_eq!(words("a-b-c"), ["a-", "b-", "c"]);
        // No identifier byte before the `-` → it stays its own word.
        assert_eq!(words("a - b"), ["a ", "- ", "b"]);
        assert_eq!(words("-5"), ["-", "5"]);
        assert_eq!(words("x -= 1"), ["x ", "-= ", "1"]);
        assert_eq!(words("-> u32"), ["-> ", "u32"]);
        // A trailing hyphen has no identifier after it, so it stands alone.
        assert_eq!(words("wait- x"), ["wait", "- ", "x"]);
    }

    /// Non-ASCII *punctuation* segments like its ASCII twin instead of gluing
    /// onto the neighbouring word — but stays one word per character, not one
    /// per UTF-8 byte.
    #[test]
    fn typographic_punctuation_splits_like_ascii() {
        assert_eq!(words("“hi”"), ["“", "hi", "”"]);
        assert_eq!(words("\"hi\""), ["\"", "hi", "\""]);
        assert_eq!(words("a — b"), ["a ", "— ", "b"]);
        assert_eq!(words("wait…"), ["wait", "…"]);
        // Accented letters are alphanumeric, so they still bind into the word.
        assert_eq!(words("café “x”"), ["café ", "“", "x", "”"]);
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





