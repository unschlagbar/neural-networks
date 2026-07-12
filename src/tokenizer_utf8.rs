// Byte-level tokenizer: token ids 0..256 are raw UTF-8 bytes, ids 256.. are
// special tokens. Vocab is exactly 256 + SPECIAL_TOKENS.len().
//
// No charset file and no HashMap — encoding is `str::as_bytes` and decoding is
// `String::from_utf8_lossy`, so text in any language round-trips losslessly.
// Special tokens live *above* the byte range instead of stealing byte values,
// so no valid input can ever collide with them.

/// Special tokens, in id order. Their ids are `256 + index`.
pub const SPECIAL_TOKENS: &[&str] = &["<W>", "<END>"];

/// Number of byte tokens — ids `0..256` are exactly the UTF-8 byte values.
pub const BYTE_TOKENS: usize = 256;

/// `[W]` word-boundary marker (HAT): appended as the encoder's end-of-word step
/// and as the decoder's end-of-word target. Model-internal — never in the data.
pub const W_TOKEN: u16 = BYTE_TOKENS as u16;
/// End-of-text marker. Not emitted by `to_tokens`; used by samplers as a stop.
pub const END_TOKEN: u16 = BYTE_TOKENS as u16 + 1;

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

    /// Bytes a fixed-size token window may be cut at (used by the flat model's
    /// window builder — the hierarchical word split lives in `crate::segment`).
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
    fn multibyte_chars_span_several_tokens() {
        let tok = Utf8Tokenizer::new();
        let tokens = tok.to_tokens("ä");
        assert_eq!(tokens, vec![0xC3, 0xA4]);
        assert_eq!(tok.to_text(&tokens), "ä");
    }
}
