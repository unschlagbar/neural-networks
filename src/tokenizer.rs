use std::collections::HashMap;
use std::fs;

#[derive(Clone)]
pub struct Tokenizer {
    pub stoi: HashMap<String, u16>,
    pub itos: Vec<String>,
    pub non_token_indexes: usize,
}

impl Tokenizer {
    pub fn new(charset_path: &str) -> Self {
        let chars = fs::read_to_string(charset_path).expect("Charset konnte nicht gelesen werden");
        let mut vocab: Vec<char> = chars.chars().collect();

        vocab.sort();
        vocab.dedup();

        let mut stoi = HashMap::new();
        let mut itos = Vec::new();

        for (i, c) in vocab.iter().enumerate() {
            let s = c.to_string();
            stoi.insert(s.clone(), i as u16);
            itos.push(s);
        }

        // Spezielle Tokens für Mehrfach-Leerzeichen
        let specials = vec!["<SPACE2>", "<SPACE4>", "<END>", "<QSTART>", "<QEND>"];
        for (i, token) in specials.iter().enumerate() {
            let index = vocab.len() + i;
            stoi.insert(token.to_string(), index as u16);
            itos.push(token.to_string());
        }

        println!("Vokabulargröße: {}", itos.len());

        Tokenizer {
            stoi,
            itos,
            non_token_indexes: 1,
        }
    }

    pub fn to_tokens(&self, text: &str) -> Vec<u16> {
        let mut tokens = Vec::new();
        let mut space_counter = 0;

        for c in text.chars() {
            if c == ' ' {
                space_counter += 1;
                continue;
            }

            // Wenn vorher Leerzeichen waren, verarbeite sie
            match space_counter {
                1 => {
                    tokens.push(self.stoi[" "]);
                }
                2 => {
                    tokens.push(self.stoi["<SPACE2>"]);
                }
                3 => {
                    tokens.push(self.stoi[" "]);
                    tokens.push(self.stoi["<SPACE2>"]);
                }
                4 => {
                    tokens.push(self.stoi["<SPACE4>"]);
                }
                _ => {
                    for _ in 0..space_counter / 4 {
                        tokens.push(self.stoi["<SPACE4>"]);
                    }
                }
            }

            space_counter = 0;

            let s = c.to_string();
            if let Some(&token) = self.stoi.get(&s) {
                tokens.push(token);
            } else {
                println!("Unbekanntes Zeichen: {:04}", c);
            }
        }

        // Falls am Ende noch Leerzeichen übrig sind
        if space_counter > 0 {
            match space_counter {
                1 => {
                    tokens.push(self.stoi[" "]);
                }
                2 => {
                    tokens.push(self.stoi["<SPACE2>"]);
                }
                3 => {
                    tokens.push(self.stoi[" "]);
                    tokens.push(self.stoi["<SPACE2>"]);
                }
                4 => {
                    tokens.push(self.stoi["<SPACE4>"]);
                }
                _ => {
                    for _ in 0..space_counter / 4 {
                        tokens.push(self.stoi["<SPACE4>"]);
                    }
                }
            }
        }

        tokens
    }

    pub fn to_text(&self, tokens: &[u16]) -> String {
        let mut result = String::new();

        for &token in tokens {
            if let Some(symbol) = self.itos.get(token as usize) {
                match symbol.as_str() {
                    "<SPACE2>" => result.push_str("  "),
                    "<SPACE4>" => result.push_str("    "),
                    "<END>" => result.push_str("<END>"),
                    "<QSTART>" => result.push_str("<QSTART>"),
                    "<QEND>" => result.push_str("<QEND>"),
                    _ => result.push_str(symbol),
                }
            } else {
                panic!("Token {} nicht im Vokabular gefunden", token);
            }
        }

        result
    }

    pub fn get_char(&self, token: u16) -> &str {
        if let Some(symbol) = self.itos.get(token as usize) {
            match symbol.as_str() {
                "<SPACE2>" => "  ",
                "<SPACE4>" => "    ",
                "<END>" => "<END>",
                "<QSTART>" => "<QSTART>",
                "<QEND>" => "<QEND>",
                _ => symbol,
            }
        } else {
            panic!("Token {} nicht im Vokabular gefunden", token);
        }
    }

    pub fn get_token(&self, char: &str) -> u16 {
        if let Some(symbol) = self.stoi.get(char) {
            *symbol
        } else {
            panic!("Char {} nicht im Vokabular gefunden", char);
        }
    }

    pub fn get_token_space<'a>(&self, output: &'a [f32]) -> &'a [f32] {
        assert_eq!(output.len(), self.vocab_size() + self.non_token_indexes);
        &output[0..self.vocab_size()]
    }

    pub const fn vocab_size(&self) -> usize {
        self.itos.len()
    }

    pub const fn size(&self) -> usize {
        self.itos.len() + self.non_token_indexes
    }

    pub fn build_input(&self, index: usize, non_token_data: &[f32]) -> Vec<f32> {
        assert!(index < self.vocab_size());
        assert_eq!(non_token_data.len(), self.non_token_indexes);
        let mut input = vec![0.0; self.size()];
        input[index] = 1.0;
        input[self.vocab_size()..].copy_from_slice(non_token_data);
        input
    }
}
