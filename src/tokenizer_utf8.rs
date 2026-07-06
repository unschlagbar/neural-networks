// Byte-level tokenizer: every token is one raw UTF-8 byte, vocab is exactly 256.
//
// No charset file, no HashMap, no special tokens appended on top — encoding is
// `str::as_bytes` and decoding is `String::from_utf8_lossy`. Any text in any
// language round-trips losslessly.
//
// The hierarchical model still needs a word-boundary marker `[W]`. Bytes
// 0xC0, 0xC1 and 0xF5..=0xFF can never occur in valid UTF-8, so 0xFF is used
// as `[W]`. It is model-internal (fed virtually, never present in token
// slices), so decoding never sees it either.

#[derive(Clone, Default)]
pub struct Utf8Tokenizer;

/// `[W]` word-boundary marker (HAT): 0xFF never occurs in valid UTF-8.
pub const W_TOKEN: u16 = 0xFF;

impl Utf8Tokenizer {
    pub fn new() -> Self {
        Utf8Tokenizer
    }

    /// Encode `text` into a token sequence (one token per UTF-8 byte).
    pub fn to_tokens(&self, text: &str) -> Vec<u16> {
        text.bytes().map(u16::from).collect()
    }

    /// Decode a token sequence back into text. Invalid UTF-8 sequences
    /// (e.g. a multi-byte character cut off mid-window) decode to U+FFFD.
    pub fn to_text(&self, tokens: &[u16]) -> String {
        let bytes: Vec<u8> = tokens.iter().map(|&t| t as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Display string for a token id: printable ASCII as-is, everything else
    /// as an escape like `\xC3` (a lone byte of a multi-byte char is not
    /// valid text on its own).
    pub fn display(&self, token: u16) -> String {
        assert!(token < 256, "Token {} not in vocabulary", token);
        if token == W_TOKEN {
            return "<W>".to_string();
        }
        match token as u8 {
            b' '..=b'~' => (token as u8 as char).to_string(),
            b'\n' => "\\n".to_string(),
            b'\t' => "\\t".to_string(),
            b => format!("\\x{:02X}", b),
        }
    }

    /// Human-readable rendering of a token sequence: decodes valid UTF-8
    /// runs and escapes stray bytes.
    pub fn display_tokens(&self, tokens: &[u16]) -> String {
        self.to_text(tokens)
    }

    /// Token id of a single-byte (ASCII) character.
    pub fn get_token(&self, c: char) -> u16 {
        let mapped = if c == '—' { '-' } else { c };
        assert!(
            mapped.is_ascii(),
            "Char {:?} is multi-byte; use to_tokens for non-ASCII",
            c
        );
        mapped as u16
    }

    /// Always 256 — one token per byte value.
    pub const fn vocab_size(&self) -> usize {
        256
    }

    /// The `[W]` word-boundary marker id (HAT encoder prefix / decoder EOS target).
    pub const fn w_token(&self) -> u16 {
        W_TOKEN
    }

    const BOUNDARIES: &[u8] = &[
        b' ', b'.', b'!', b'?', b',', b';', b':', b'\n', b'{', b'}', b'(', b')', b'[', b']', b'<',
        b'>', b'|', b'"', b'\'',
    ];

    /// Token ids that count as word/segment boundaries for the hierarchical model.
    pub fn boundary_tokens(&self) -> Vec<u16> {
        Self::BOUNDARIES.iter().map(|&b| u16::from(b)).collect()
    }

    /// Token ids that count as sentence boundaries.
    pub fn sentence_tokens(&self) -> Vec<u16> {
        vec![b'.' as u16, b'!' as u16, b'?' as u16, b';' as u16]
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
        let tokens = self.to_tokens(sample);
        println!("  encode({:?}) → {:?}", sample, tokens);
        println!("  decode     → {:?}", self.to_text(&tokens));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_ascii_and_multibyte() {
        let tok = Utf8Tokenizer::new();
        assert!(tok.roundtrip_check("Hello, world!\n"));
        assert!(tok.roundtrip_check("Größe — äöü ß 日本語 🦀"));
    }

    #[test]
    fn vocab_is_256_and_w_token_is_reserved() {
        let tok = Utf8Tokenizer::new();
        assert_eq!(tok.vocab_size(), 256);
        assert_eq!(tok.w_token(), 0xFF);
        // 0xFF never appears when encoding valid UTF-8.
        assert!(!tok.to_tokens("Größe 🦀 日本語").contains(&0xFF));
    }

    #[test]
    fn multibyte_chars_span_several_tokens() {
        let tok = Utf8Tokenizer::new();
        let tokens = tok.to_tokens("ä");
        assert_eq!(tokens, vec![0xC3, 0xA4]);
        assert_eq!(tok.to_text(&tokens), "ä");
    }
}
