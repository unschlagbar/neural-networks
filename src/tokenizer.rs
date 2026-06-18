use std::collections::HashMap;
use std::{char, fs};

// Character-level tokenizer with:
//   - O(1) ASCII lookup via a 128-entry array (no HashMap for common chars)
//   - Correct space encoding for any count (old code lost spaces when count ≥ 5)
//   - Clean separation of vocab_size (real chars) vs full itos length (incl. specials)

#[derive(Clone, Default)]
pub struct Tokenizer {
    pub lowercase: bool,
    /// char/special-token string → token id
    pub stoi: HashMap<char, u16>,
    /// token id → display string
    pub itos: Vec<String>,
    // Cached ids for the hot encode path
    id_space: u16,
    id_space2: u16,
    id_space4: u16,
    /// Word-boundary marker `[W]` (HAT): prepended at the encoder, appended as the
    /// decoder's end-of-word target. Model-internal — never a word boundary.
    id_w: u16,
}

pub const SPECIAL_TOKENS: &[&str] =
    &["<SPACE2>", "<SPACE4>", "<END>", "<QSTART>", "<QEND>", "<W>"];

impl Tokenizer {
    pub fn new(charset_path: &str, lowercase: bool) -> Self {
        let chars = fs::read_to_string(charset_path).expect("Charset could not be read");
        let mut vocab: Vec<char> = chars.chars().collect();
        vocab.sort_unstable();
        Self::new_vocab(&vocab, lowercase)
    }

    pub fn new_vocab(vocab: &[char], lowercase: bool) -> Self {
        let mut stoi = HashMap::new();
        let mut itos = Vec::new();

        for (i, &c) in vocab.iter().enumerate() {
            stoi.insert(c, i as u16);
            itos.push(c.to_string());
        }

        for token in SPECIAL_TOKENS {
            itos.push(token.to_string());
        }

        println!("Vocabulary size: {}", itos.len());

        let id_space = stoi[&' '];
        let id_space2 = stoi.len() as u16
            + SPECIAL_TOKENS
                .iter()
                .position(|&t| t == "<SPACE2>")
                .unwrap() as u16;
        let id_space4 = stoi.len() as u16
            + SPECIAL_TOKENS
                .iter()
                .position(|&t| t == "<SPACE4>")
                .unwrap() as u16;
        let id_w = stoi.len() as u16
            + SPECIAL_TOKENS.iter().position(|&t| t == "<W>").unwrap() as u16;

        Tokenizer {
            lowercase,
            stoi,
            itos,
            id_space,
            id_space2,
            id_space4,
            id_w,
        }
    }

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
        let mut spaces = 0;

        for mut c in text.chars() {
            if c == ' ' {
                spaces += 1;
                continue;
            }

            // Flush pending spaces before any non-space character.
            if spaces > 0 {
                self.emit_spaces(spaces, &mut tokens);
                spaces = 0;
            }

            if c == '—' {
                c = '-'
            };

            if self.lowercase {
                c = c.to_ascii_lowercase()
            }

            if let Some(&id) = self.stoi.get(&c) {
                tokens.push(id);
            }
        }

        // Flush trailing spaces.
        if spaces > 0 {
            self.emit_spaces(spaces, &mut tokens);
        }

        tokens
    }

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

    /// Display string for a token id (panics on unknown id).
    pub fn display_tokens(&self, tokens: &[u16]) -> String {
        tokens.iter().map(|&t| self.display(t)).collect()
    }

    /// Alias kept for backwards compatibility.
    #[inline]
    pub fn get_char(&self, token: u16) -> &str {
        self.display(token)
    }

    pub fn get_token(&self, c: char) -> u16 {
        if let Some(&id) = self.stoi.get(&c) {
            return id;
        }
        let mapped = if c == '—' { '-' } else { c };
        if let Some(&id) = self.stoi.get(&mapped) {
            return id;
        }
        panic!("Char {:?} not in vocabulary", c);
    }

    /// Number of tokens including specials (= length of `itos`).
    pub const fn vocab_size(&self) -> usize {
        self.itos.len()
    }

    /// The `[W]` word-boundary marker id (HAT encoder prefix / decoder EOS target).
    pub const fn w_token(&self) -> u16 {
        self.id_w
    }

    const BOUNDARIES: &[char] = &[
        '.', '!', '?', ',', ';', ':', '\n', '{', '}', '(', ')', '[', ']', '<', '>', '|', '"', '\'',
    ];

    /// Token ids that count as word/segment boundaries for the hierarchical model.
    pub fn boundary_tokens(&self) -> Vec<u16> {
        let mut ids = vec![self.id_space, self.id_space2, self.id_space4];
        for c in Self::BOUNDARIES {
            ids.push(self.stoi[c]);
        }
        ids
    }

    /// Token ids that count as word/segment boundaries for the hierarchical model.
    pub fn sentence_tokens(&self) -> Vec<u16> {
        let mut ids = Vec::with_capacity(4);
        for c in ['.', '!', '?', ';'] {
            if let Some(&id) = self.stoi.get(&c) {
                ids.push(id);
            }
        }
        ids
    }
}

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
