use std::collections::HashMap;
use std::fs;

// ── Tokenizer ─────────────────────────────────────────────────────────────────
//
// Character-level tokenizer with:
//   - O(1) ASCII lookup via a 128-entry array (no HashMap for common chars)
//   - Correct space encoding for any count (old code lost spaces when count ≥ 5)
//   - Clean separation of vocab_size (real chars) vs full itos length (incl. specials)

#[derive(Clone)]
pub struct Tokenizer {
    pub lowercase: bool,
    /// char/special-token string → token id
    pub stoi: HashMap<String, u16>,
    /// token id → display string
    pub itos: Vec<String>,
    /// Fast O(1) lookup for ASCII bytes 0..128.
    /// Entry is u16::MAX when the byte has no token.
    ascii_map: Box<[u16; 128]>,
    // Cached ids for the hot encode path
    id_space: u16,
    id_space2: u16,
    id_space4: u16,
}

// ── Special token names ───────────────────────────────────────────────────────
pub const SPECIAL_TOKENS: &[&str] = &["<SPACE2>", "<SPACE4>", "<END>", "<QSTART>", "<QEND>"];

impl Tokenizer {
    // ── Constructors ──────────────────────────────────────────────────────────

    pub fn new(charset_path: &str, lowercase: bool) -> Self {
        let chars = fs::read_to_string(charset_path).expect("Charset could not be read");
        let mut vocab: Vec<char> = chars.chars().collect();
        vocab.sort_unstable();
        Self::new_vocab(&vocab, lowercase)
    }

    pub fn new_vocab(vocab: &[char], lowercase: bool) -> Self {
        let mut stoi = HashMap::new();
        let mut itos = Vec::new();

        for (i, c) in vocab.iter().enumerate() {
            let s = c.to_string();
            stoi.insert(s.clone(), i as u16);
            itos.push(s);
        }

        for (i, &token) in SPECIAL_TOKENS.iter().enumerate() {
            let index = vocab.len() + i;
            stoi.insert(token.to_string(), index as u16);
            itos.push(token.to_string());
        }

        println!("Vocabulary size: {}", itos.len());

        // ── ASCII fast-lookup table ───────────────────────────────────────────
        let mut ascii_map = Box::new([u16::MAX; 128]);
        for (s, &id) in &stoi {
            if s.len() == 1 {
                let b = s.as_bytes()[0];
                if (b as usize) < 128 {
                    ascii_map[b as usize] = id;
                }
            }
        }
        // If lowercase, remap uppercase bytes to point at their lowercase token id.
        if lowercase {
            for b in b'A'..=b'Z' {
                let lower = b.to_ascii_lowercase();
                if ascii_map[lower as usize] != u16::MAX {
                    ascii_map[b as usize] = ascii_map[lower as usize];
                }
            }
        }

        let id_space = stoi[" "];
        let id_space2 = stoi["<SPACE2>"];
        let id_space4 = stoi["<SPACE4>"];

        Tokenizer {
            lowercase,
            stoi,
            itos,
            ascii_map,
            id_space,
            id_space2,
            id_space4,
        }
    }

    // ── Encoding ──────────────────────────────────────────────────────────────

    /// Emit the correct token sequence for `count` consecutive spaces.
    /// Correctly handles any count (old code lost the remainder for count ≥ 5).
    #[inline]
    fn emit_spaces(&self, count: usize, tokens: &mut Vec<u16>) {
        let mut rem = count;
        // Groups of 4 → SPACE4
        for _ in 0..rem / 4 {
            tokens.push(self.id_space4);
        }
        rem %= 4;
        // Remainder
        match rem {
            3 => {
                tokens.push(self.id_space);
                tokens.push(self.id_space2);
            }
            2 => tokens.push(self.id_space2),
            1 => tokens.push(self.id_space),
            _ => {}
        }
    }

    /// Encode `text` into a token sequence.
    pub fn to_tokens(&self, text: &str) -> Vec<u16> {
        let mut tokens = Vec::with_capacity(text.len());
        let mut spaces = 0usize;

        for c in text.chars() {
            if c == ' ' {
                spaces += 1;
                continue;
            }

            // Flush pending spaces before any non-space character.
            if spaces > 0 {
                self.emit_spaces(spaces, &mut tokens);
                spaces = 0;
            }

            // ── em-dash → hyphen ─────────────────────────────────────────────
            let c = if c == '—' { '-' } else { c };

            // ── ASCII fast path (avoids HashMap) ─────────────────────────────
            let byte = c as u32;
            if byte < 128 {
                let effective = if self.lowercase && (c as u8).is_ascii_uppercase() {
                    c.to_ascii_lowercase() as usize
                } else {
                    byte as usize
                };
                let id = self.ascii_map[effective];
                if id != u16::MAX {
                    tokens.push(id);
                }
                // Unknown ASCII chars are silently skipped (same as original).
                continue;
            }

            // ── Non-ASCII slow path ───────────────────────────────────────────
            let s = c.to_string();
            if let Some(&id) = self.stoi.get(&s) {
                tokens.push(id);
            }
        }

        // Flush trailing spaces.
        if spaces > 0 {
            self.emit_spaces(spaces, &mut tokens);
        }

        tokens
    }

    // ── Decoding ──────────────────────────────────────────────────────────────

    pub fn to_text(&self, tokens: &[u16]) -> String {
        let mut result = String::with_capacity(tokens.len());
        for &token in tokens {
            result.push_str(self.display(token));
        }
        result
    }

    /// Display string for a token id (panics on unknown id).
    pub fn display(&self, token: u16) -> &str {
        match self.itos.get(token as usize).map(|s| s.as_str()) {
            Some("<SPACE2>") => "  ",
            Some("<SPACE4>") => "    ",
            Some(s) => s,
            None => panic!("Token {} not in vocabulary", token),
        }
    }

    /// Alias kept for backwards compatibility.
    #[inline]
    pub fn get_char(&self, token: u16) -> &str {
        self.display(token)
    }

    // ── Lookup ────────────────────────────────────────────────────────────────

    pub fn get_token(&self, s: &str) -> u16 {
        if let Some(&id) = self.stoi.get(s) {
            return id;
        }
        // Try single-char em-dash conversion
        if let Some(c) = s.chars().next() {
            let mapped = if c == '—' { '-' } else { c };
            if let Some(&id) = self.stoi.get(&mapped.to_string()) {
                return id;
            }
        }
        panic!("Char {:?} not in vocabulary", s);
    }

    // ── Sizes ────────────────────────────────────────────────────────────────

    /// Number of tokens including specials (= length of `itos`).
    pub const fn vocab_size(&self) -> usize {
        self.itos.len()
    }

    // ── Boundary ids ─────────────────────────────────────────────────────────

    /// Token ids that count as word/segment boundaries for the hierarchical model.
    pub fn boundary_token_ids(&self) -> Vec<u16> {
        let mut ids = vec![self.id_space, self.id_space2, self.id_space4];
        for c in [".", "!", "?", ",", ";", ":", "\n"] {
            if let Some(&id) = self.stoi.get(c) {
                ids.push(id);
            }
        }
        ids
    }

    // ── Boundary ids ─────────────────────────────────────────────────────────

    /// Token ids that count as word/segment boundaries for the hierarchical model.
    pub fn sentence_token_ids(&self) -> Vec<u16> {
        let mut ids = Vec::with_capacity(4);
        for c in ['.', '!', '?', ';'] {
            if let Some(&id) = self.stoi.get(&c.to_string()) {
                ids.push(id);
            }
        }
        ids
    }
}

// ── Debug / test helpers ──────────────────────────────────────────────────────

impl Tokenizer {
    /// Round-trip a string through encode → decode and check it matches.
    /// Useful in unit tests or sanity checks during data loading.
    pub fn roundtrip_check(&self, text: &str) -> bool {
        let tokens = self.to_tokens(text);
        let decoded = self.to_text(&tokens);
        // Normalise the original the same way the encoder does before comparing.
        let normalised: String = text
            .chars()
            .map(|c| if c == '—' { '-' } else { c })
            .collect();
        decoded == normalised
    }

    /// Print a brief summary: vocab size, first N tokens, sample encoding.
    pub fn debug_summary(&self, sample: &str) {
        println!("=== Tokenizer ===");
        println!("  vocab_size : {}", self.vocab_size());
        println!("  lowercase  : {}", self.lowercase);
        println!("  first 10   : {:?}", &self.itos[..self.itos.len().min(10)]);
        let tokens = self.to_tokens(sample);
        println!("  encode({:?}) → {:?}", sample, tokens);
        println!("  decode     → {:?}", self.to_text(&tokens));
    }
}
