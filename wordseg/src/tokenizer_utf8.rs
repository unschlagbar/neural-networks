// Byte-level tokenizer: token ids 0..256 are raw UTF-8 bytes, ids 256.. are
// special tokens. Vocab is exactly 256 + SPECIAL_TOKENS.len().
//
// No charset file and no HashMap — encoding is `str::as_bytes` and decoding is
// `String::from_utf8_lossy`, so text in any language round-trips losslessly.
// Special tokens live *above* the byte range instead of stealing byte values,
// so no valid input can ever collide with them.

/// Special tokens, in id order. Their ids are `256 + index`.
///
/// The first two (`<W>`, `<END>`) are the pretraining specials and their ids
/// are load-bearing — `<W>` (256) is the encoder/decoder end-of-word marker and
/// `<END>` (257) the end-of-text stop. The rest are post-training (SFT) chat
/// markers, appended *after* them so an existing byte/`<W>`/`<END>` id never
/// shifts. A pretrained checkpoint has a smaller vocab than this list implies;
/// `crate::grow_vocab` widens its tables to make room for the new rows.
///
/// From `MARKUP_START` on the specials are *markup*: unlike the structural
/// markers above them they also occur verbatim in the training data, so
/// `to_tokens_markup` folds each literal occurrence into its token and
/// `to_text_markup` writes it back out unchanged.
pub const SPECIAL_TOKENS: &[&str] = &[
    "<W>",
    "<END>",
    "<CONTEXT>",
    "<SEP>",
    "<tool>",
    "</tool>",
    "<result>",
    "</result>",
    "<think>",
    "</think>",
];

/// Index into `SPECIAL_TOKENS` of the first markup special.
pub const MARKUP_START: usize = 4;

/// Number of specials present in a model trained *before* SFT (only `<W>` and
/// `<END>`). A pretrained checkpoint therefore has `256 + PRETRAIN_SPECIALS`
/// vocab rows; grow-vocab lifts it to the full `vocab_size()`.
pub const PRETRAIN_SPECIALS: usize = 2;

/// Number of byte tokens — ids `0..256` are exactly the UTF-8 byte values.
pub const BYTE_TOKENS: usize = 256;

/// `[W]` word-boundary marker (HAT): appended as the encoder's end-of-word step
/// and as the decoder's end-of-word target. Model-internal — never in the data.
pub const W_TOKEN: u16 = BYTE_TOKENS as u16;
/// End-of-text marker. Not emitted by `to_tokens`; used by samplers as a stop
/// and, in SFT, as the end-of-response target.
pub const END_TOKEN: u16 = BYTE_TOKENS as u16 + 1;
/// `<CONTEXT>` — SFT marker opening the optional context block of a prompt.
pub const CONTEXT_TOKEN: u16 = BYTE_TOKENS as u16 + 2;
/// `<SEP>` — SFT marker separating the prompt from the assistant response.
pub const SEP_TOKEN: u16 = BYTE_TOKENS as u16 + 3;
/// `<tool>` … `</tool>` — an assistant-side tool call. Written by the model.
pub const TOOL_OPEN_TOKEN: u16 = BYTE_TOKENS as u16 + 4;
pub const TOOL_CLOSE_TOKEN: u16 = BYTE_TOKENS as u16 + 5;
/// `<result>` … `</result>` — what a call returned. Prompt-side: read, never
/// produced, so it is followed by a `<SEP>` like any other prompt turn.
pub const RESULT_OPEN_TOKEN: u16 = BYTE_TOKENS as u16 + 6;
pub const RESULT_CLOSE_TOKEN: u16 = BYTE_TOKENS as u16 + 7;
/// `<think>` … `</think>` — an assistant reasoning block preceding the answer.
pub const THINK_OPEN_TOKEN: u16 = BYTE_TOKENS as u16 + 8;
pub const THINK_CLOSE_TOKEN: u16 = BYTE_TOKENS as u16 + 9;

#[derive(Clone, Copy, Default)]
pub struct Utf8Tokenizer;

impl Utf8Tokenizer {
    pub fn new() -> Self {
        Utf8Tokenizer
    }

    /// Encode `text` into a token sequence (one token per UTF-8 byte).
    pub fn to_tokens(&self, text: &str) -> Vec<u16> {
        text.bytes().map(u16::from).collect()
    }

    /// Encode `text`, folding every literal markup marker (`<tool>`,
    /// `</result>`, …) into its own token instead of its bytes. Identical to
    /// [`Self::to_tokens`] for text that contains none.
    pub fn to_tokens_markup(&self, text: &str) -> Vec<u16> {
        let bytes = text.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            let marker = (bytes[i] == b'<')
                .then(|| {
                    SPECIAL_TOKENS[MARKUP_START..]
                        .iter()
                        .position(|name| bytes[i..].starts_with(name.as_bytes()))
                })
                .flatten();
            match marker {
                Some(k) => {
                    out.push((BYTE_TOKENS + MARKUP_START + k) as u16);
                    i += SPECIAL_TOKENS[MARKUP_START + k].len();
                }
                None => {
                    out.push(u16::from(bytes[i]));
                    i += 1;
                }
            }
        }
        out
    }

    /// Decode a token sequence, spelling markup markers back out as text and
    /// skipping the structural ones (`<W>`, `<END>`, `<SEP>`, `<CONTEXT>`). The
    /// inverse of [`Self::to_tokens_markup`].
    pub fn to_text_markup(&self, tokens: &[u16]) -> String {
        let mut out = String::new();
        let mut run: Vec<u16> = Vec::new();
        for &t in tokens {
            match (t as usize).checked_sub(BYTE_TOKENS) {
                None => run.push(t),
                Some(s) if s >= MARKUP_START => {
                    out.push_str(&self.to_text(&run));
                    run.clear();
                    out.push_str(SPECIAL_TOKENS[s]);
                }
                Some(_) => {}
            }
        }
        out.push_str(&self.to_text(&run));
        out
    }

    /// Whether `token` is a markup marker — a special that is part of the text
    /// the model writes, not a structural boundary.
    pub fn is_markup(&self, token: u16) -> bool {
        (token as usize) >= BYTE_TOKENS + MARKUP_START
            && (token as usize) < self.vocab_size()
    }

    /// Decode a token sequence back into text. Special tokens are skipped;
    /// invalid UTF-8 (a multi-byte char cut off at a window edge) becomes U+FFFD.
    pub fn to_text(&self, tokens: &[u16]) -> String {
        let bytes: Vec<u8> = tokens
            .iter()
            .filter(|&&t| (t as usize) < BYTE_TOKENS)
            .map(|&t| t as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Display string for a single token id: printable ASCII as itself, special
    /// tokens by name, every other byte as an escape (a lone byte of a
    /// multi-byte char is not valid text on its own).
    pub fn display(&self, token: u16) -> String {
        let id = token as usize;
        if let Some(name) = SPECIAL_TOKENS.get(id.wrapping_sub(BYTE_TOKENS)) {
            return (*name).to_string();
        }
        assert!(id < BYTE_TOKENS, "Token {token} not in vocabulary");
        match token as u8 {
            b'\n' => "\\n".to_string(),
            b'\t' => "\\t".to_string(),
            b @ b' '..=b'~' => (b as char).to_string(),
            b => format!("\\x{b:02X}"),
        }
    }

    /// Human-readable rendering of a token sequence: decodes the byte tokens as
    /// UTF-8 and spells out any special token inline.
    pub fn display_tokens(&self, tokens: &[u16]) -> String {
        let mut out = String::new();
        let mut run: Vec<u16> = Vec::new();
        for &t in tokens {
            if (t as usize) < BYTE_TOKENS {
                run.push(t);
            } else {
                out.push_str(&self.to_text(&run));
                run.clear();
                out.push_str(&self.display(t));
            }
        }
        out.push_str(&self.to_text(&run));
        out
    }

    /// Token id of a single-byte (ASCII) character.
    pub fn get_token(&self, c: char) -> u16 {
        assert!(
            c.is_ascii(),
            "Char {c:?} is multi-byte; use to_tokens for non-ASCII"
        );
        c as u16
    }

    /// 256 byte tokens plus the specials.
    pub const fn vocab_size(&self) -> usize {
        BYTE_TOKENS + SPECIAL_TOKENS.len()
    }

    /// The `[W]` end-of-word marker id (encoder EOS step / decoder EOS target).
    pub const fn w_token(&self) -> u16 {
        W_TOKEN
    }

    /// The `<END>` end-of-text marker id.
    pub const fn end_token(&self) -> u16 {
        END_TOKEN
    }

    /// The `<CONTEXT>` SFT marker id.
    pub const fn context_token(&self) -> u16 {
        CONTEXT_TOKEN
    }

    /// The `<SEP>` SFT marker id.
    pub const fn sep_token(&self) -> u16 {
        SEP_TOKEN
    }

    /// Bytes a fixed-size token window may be cut at. The word split the models
    /// actually train on lives in `crate::segment`.
    pub fn boundary_tokens(&self) -> Vec<u16> {
        [
            b' ', b'.', b'!', b'?', b',', b';', b':', b'\n', b'{', b'}', b'(', b')',
        ]
        .iter()
        .map(|&b| u16::from(b))
        .collect()
    }

    /// Round-trip a string through encode → decode and check it matches.
    /// Byte-level encoding is lossless, so this holds for any valid UTF-8 input.
    pub fn roundtrip_check(&self, text: &str) -> bool {
        self.to_text(&self.to_tokens(text)) == text
    }

    /// Print a brief summary: vocab size and a sample encoding.
    pub fn debug_summary(&self, sample: &str) {
        println!("=== Utf8Tokenizer ===");
        println!("  vocab_size : {}", self.vocab_size());
        println!("  specials   : {SPECIAL_TOKENS:?} (ids {BYTE_TOKENS}..)");
        let tokens = self.to_tokens(sample);
        println!("  encode({sample:?}) → {tokens:?}");
        println!("  decode     → {:?}", self.to_text(&tokens));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_ascii_and_multibyte() {
        let tok = Utf8Tokenizer::new();
        assert!(tok.roundtrip_check("fn main() { println!(\"hi\"); }\n"));
        assert!(tok.roundtrip_check("Größe — äöü ß 日本語 🦀"));
    }

    #[test]
    fn specials_sit_above_the_byte_range() {
        let tok = Utf8Tokenizer::new();
        assert_eq!(tok.vocab_size(), 256 + SPECIAL_TOKENS.len());
        assert_eq!(tok.w_token(), 256);
        assert_eq!(tok.end_token(), 257);
        // No encoded text can collide with a special token.
        assert!(tok.to_tokens("Größe 🦀 日本語").iter().all(|&t| t < 256));
        assert_eq!(tok.display(W_TOKEN), "<W>");
    }

    #[test]
    fn markup_markers_are_single_tokens() {
        let tok = Utf8Tokenizer::new();
        let text = "<think>the lamp</think><tool>lamp.set(on=true)</tool>";
        let tokens = tok.to_tokens_markup(text);
        assert_eq!(tokens.iter().filter(|&&t| t == THINK_OPEN_TOKEN).count(), 1);
        assert_eq!(tokens.iter().filter(|&&t| t == TOOL_CLOSE_TOKEN).count(), 1);
        assert!(!tok.to_text(&tokens).contains("<tool>"));
        assert_eq!(tok.to_text_markup(&tokens), text);
    }

    #[test]
    fn markup_encoding_leaves_plain_text_alone() {
        let tok = Utf8Tokenizer::new();
        for s in ["a < b && c > d", "Vec<String>", "</notatool>", "日本語 🦀"] {
            assert_eq!(tok.to_tokens_markup(s), tok.to_tokens(s));
        }
    }

    #[test]
    fn multibyte_chars_span_several_tokens() {
        let tok = Utf8Tokenizer::new();
        let tokens = tok.to_tokens("ä");
        assert_eq!(tokens, vec![0xC3, 0xA4]);
        assert_eq!(tok.to_text(&tokens), "ä");
    }
}
