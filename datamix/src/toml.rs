// The slice of TOML the mixture format uses, parsed by hand (the repo carries
// no serde).
//
// Supported: table headers including dotted and quoted names, basic and literal
// strings with their multi-line forms, integers, floats, booleans, and arrays
// of those (arrays may span lines). Not supported, because the format has no
// use for them: inline tables, arrays of tables, dates, dotted keys inside a
// table. Each is refused by name rather than mis-parsed.

use std::fmt;

pub type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Array(Vec<Value>),
}

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Str(_) => "string",
            Value::Int(_) => "integer",
            Value::Float(_) => "float",
            Value::Bool(_) => "boolean",
            Value::Array(_) => "array",
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Str(s) => write!(f, "{s:?}"),
            Value::Int(i) => write!(f, "{i}"),
            Value::Float(x) => write!(f, "{x}"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Array(_) => write!(f, "an array"),
        }
    }
}

/// One key/value pair, with the table it belongs to and the line it was written
/// on. A flat list is all the mixture format needs — grouping into sections is
/// `config.rs`'s job.
pub struct Entry {
    pub table: Vec<String>,
    pub key: String,
    pub value: Value,
    pub line: usize,
}

pub fn parse(text: &str, path: &str) -> Result<Vec<Entry>> {
    Parser {
        src: text.as_bytes(),
        text,
        pos: 0,
        line: 1,
        path,
    }
    .document()
}

struct Parser<'a> {
    src: &'a [u8],
    text: &'a str,
    pos: usize,
    line: usize,
    path: &'a str,
}

impl<'a> Parser<'a> {
    fn err<T>(&self, msg: impl fmt::Display) -> Result<T> {
        Err(format!("{}:{}: {msg}", self.path, self.line))
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        if b == b'\n' {
            self.line += 1;
        }
        Some(b)
    }

    fn eat(&mut self, b: u8) -> bool {
        if self.peek() == Some(b) {
            self.bump();
            true
        } else {
            false
        }
    }

    /// Whitespace, newlines and comments. A `#` inside a string is not a
    /// comment, which is why comments are stripped here and not line by line.
    fn skip_trivia(&mut self) {
        loop {
            match self.peek() {
                Some(b) if b.is_ascii_whitespace() => {
                    self.bump();
                }
                Some(b'#') => {
                    while let Some(b) = self.peek() {
                        if b == b'\n' {
                            break;
                        }
                        self.bump();
                    }
                }
                _ => return,
            }
        }
    }

    /// Spaces and tabs only — used to check that a value is followed by nothing
    /// but a comment or a newline.
    fn skip_inline(&mut self) {
        while matches!(self.peek(), Some(b' ') | Some(b'\t') | Some(b'\r')) {
            self.bump();
        }
    }

    fn document(&mut self) -> Result<Vec<Entry>> {
        let mut out = Vec::new();
        let mut table: Vec<String> = Vec::new();
        loop {
            self.skip_trivia();
            let Some(b) = self.peek() else { return Ok(out) };
            if b == b'[' {
                self.bump();
                if self.peek() == Some(b'[') {
                    return self.err("arrays of tables ([[x]]) are not supported here");
                }
                table = self.table_header()?;
                continue;
            }
            let line = self.line;
            let key = self.key()?;
            self.skip_inline();
            if !self.eat(b'=') {
                return self.err(format!("expected '=' after key '{key}'"));
            }
            self.skip_inline();
            let value = self.value()?;
            self.skip_inline();
            match self.peek() {
                None | Some(b'\n') | Some(b'#') => {}
                Some(b) => {
                    return self.err(format!(
                        "unexpected {:?} after the value of '{key}'",
                        b as char
                    ));
                }
            }
            if table.is_empty() {
                return self.err(format!("key '{key}' is outside any table"));
            }
            out.push(Entry {
                table: table.clone(),
                key,
                value,
                line,
            });
        }
    }

    fn table_header(&mut self) -> Result<Vec<String>> {
        let mut parts = Vec::new();
        loop {
            self.skip_inline();
            parts.push(self.key()?);
            self.skip_inline();
            if self.eat(b'.') {
                continue;
            }
            if self.eat(b']') {
                return Ok(parts);
            }
            return self.err("expected '.' or ']' in a table header");
        }
    }

    /// A bare key (`A-Za-z0-9_-`) or a quoted one.
    fn key(&mut self) -> Result<String> {
        if matches!(self.peek(), Some(b'"') | Some(b'\'')) {
            return match self.string()? {
                Value::Str(s) => Ok(s),
                _ => unreachable!(),
            };
        }
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b.is_ascii_alphanumeric() || b == b'_' || b == b'-' {
                self.bump();
            } else {
                break;
            }
        }
        if self.pos == start {
            return self.err("expected a key");
        }
        Ok(self.text[start..self.pos].to_string())
    }

    fn value(&mut self) -> Result<Value> {
        match self.peek() {
            Some(b'"') | Some(b'\'') => self.string(),
            Some(b'[') => self.array(),
            Some(b'{') => self.err("inline tables are not supported here"),
            Some(_) => self.scalar(),
            None => self.err("expected a value"),
        }
    }

    fn array(&mut self) -> Result<Value> {
        self.bump(); // '['
        let mut items = Vec::new();
        loop {
            self.skip_trivia();
            if self.eat(b']') {
                return Ok(Value::Array(items));
            }
            if self.peek().is_none() {
                return self.err("unterminated array");
            }
            items.push(self.value()?);
            self.skip_trivia();
            if self.eat(b',') {
                continue;
            }
            self.skip_trivia();
            if self.eat(b']') {
                return Ok(Value::Array(items));
            }
            return self.err("expected ',' or ']' in an array");
        }
    }

    fn scalar(&mut self) -> Result<Value> {
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b == b'\n' || b == b',' || b == b']' || b == b'#' {
                break;
            }
            self.bump();
        }
        let raw = self.text[start..self.pos].trim();
        match raw {
            "true" => return Ok(Value::Bool(true)),
            "false" => return Ok(Value::Bool(false)),
            "" => return self.err("expected a value"),
            _ => {}
        }
        let clean = raw.replace('_', "");
        if let Ok(i) = clean.parse::<i64>() {
            return Ok(Value::Int(i));
        }
        if let Ok(x) = clean.parse::<f64>() {
            return Ok(Value::Float(x));
        }
        self.err(format!(
            "{raw:?} is not a value — strings must be quoted in TOML"
        ))
    }

    fn string(&mut self) -> Result<Value> {
        let quote = self.bump().unwrap();
        let multi = self.peek() == Some(quote) && self.src.get(self.pos + 1) == Some(&quote);
        if multi {
            self.bump();
            self.bump();
            // A newline immediately after the opening delimiter is dropped.
            if self.peek() == Some(b'\r') {
                self.bump();
            }
            if self.peek() == Some(b'\n') {
                self.bump();
            }
        }
        let literal = quote == b'\'';
        let mut out = String::new();
        loop {
            let Some(b) = self.peek() else {
                return self.err("unterminated string");
            };
            if b == quote {
                if !multi {
                    self.bump();
                    return Ok(Value::Str(out));
                }
                if self.src.get(self.pos + 1) == Some(&quote)
                    && self.src.get(self.pos + 2) == Some(&quote)
                {
                    self.bump();
                    self.bump();
                    self.bump();
                    return Ok(Value::Str(out));
                }
            }
            if b == b'\n' && !multi {
                return self.err("a single-line string cannot span lines");
            }
            if b == b'\\' && !literal {
                self.bump();
                let Some(e) = self.bump() else {
                    return self.err("unterminated escape");
                };
                match e {
                    b'n' => out.push('\n'),
                    b't' => out.push('\t'),
                    b'r' => out.push('\r'),
                    b'"' => out.push('"'),
                    b'\'' => out.push('\''),
                    b'\\' => out.push('\\'),
                    b'0' => out.push('\0'),
                    b'u' | b'U' => {
                        let n = if e == b'u' { 4 } else { 8 };
                        let mut hex = String::new();
                        for _ in 0..n {
                            match self.bump() {
                                Some(c) => hex.push(c as char),
                                None => return self.err("unterminated \\u escape"),
                            }
                        }
                        match u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                            Some(c) => out.push(c),
                            None => return self.err(format!("bad \\u escape: {hex:?}")),
                        }
                    }
                    // A backslash at end of line eats the newline and the
                    // indentation after it, so a long prompt can be wrapped.
                    b'\n' | b'\r' => {
                        while matches!(self.peek(), Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r'))
                        {
                            self.bump();
                        }
                    }
                    other => return self.err(format!("unknown escape \\{}", other as char)),
                }
                continue;
            }
            // Consume a whole character: `bump` moves one byte, so a multi-byte
            // character has to be stepped over in full or the next slice lands
            // inside it.
            let ch = self.text[self.pos..]
                .chars()
                .next()
                .ok_or_else(|| format!("{}:{}: unterminated string", self.path, self.line))?;
            for _ in 0..ch.len_utf8() {
                self.bump();
            }
            out.push(ch);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tables_keys_and_types() {
        let doc = "\
# a comment\n\
[output]\n\
kind = \"sft\"\n\
seed = 1234\n\
holdout = 0.02\n\
shuffle = true\n\
\n\
[source.rust-std]\n\
ext = [\"rs\", \"md\"]\n\
weight = 2\n";
        let e = parse(doc, "t.toml").unwrap();
        assert_eq!(e.len(), 6);
        assert_eq!(e[0].table, vec!["output"]);
        assert_eq!(e[0].value, Value::Str("sft".into()));
        assert_eq!(e[1].value, Value::Int(1234));
        assert_eq!(e[2].value, Value::Float(0.02));
        assert_eq!(e[3].value, Value::Bool(true));
        assert_eq!(e[4].table, vec!["source", "rust-std"]);
        assert_eq!(
            e[4].value,
            Value::Array(vec![Value::Str("rs".into()), Value::Str("md".into())])
        );
        assert_eq!(e[4].line, 9, "the entry's own line, not its table's");
    }

    #[test]
    fn a_hash_inside_a_string_is_not_a_comment() {
        let e = parse("[filter]\nmust_contain = \"#![no_std]\" # real comment\n", "t").unwrap();
        assert_eq!(e[0].value, Value::Str("#![no_std]".into()));
    }

    #[test]
    fn multi_line_strings_keep_their_newlines() {
        let doc = "[source.g]\nprompt = \"\"\"\nline one\nline two\n\"\"\"\n";
        let e = parse(doc, "t").unwrap();
        assert_eq!(e[0].value, Value::Str("line one\nline two\n".into()));
    }

    #[test]
    fn literal_strings_do_not_unescape() {
        let e = parse("[source.g]\nsep = '\\n\\n'\n", "t").unwrap();
        assert_eq!(e[0].value, Value::Str("\\n\\n".into()));
        let e = parse("[source.g]\nsep = \"\\n\\n\"\n", "t").unwrap();
        assert_eq!(e[0].value, Value::Str("\n\n".into()));
    }

    /// The mixture files wrap long prompts with a trailing backslash; a broken
    /// continuation would put stray newlines and indentation into the prompt.
    #[test]
    fn a_trailing_backslash_swallows_the_line_break() {
        let doc = "[source.g]\nprompt = \"\"\"\nask for {n} \\\n    examples\n\"\"\"\n";
        let e = parse(doc, "t").unwrap();
        assert_eq!(e[0].value, Value::Str("ask for {n} examples\n".into()));
    }

    /// A multi-byte character inside a string must survive: the parser steps
    /// over bytes, and a naive step lands inside an em dash or an umlaut.
    #[test]
    fn multibyte_characters_survive_in_strings() {
        let e = parse("[source.g]\nprompt = \"do not answer — write data für mich ✅\"\n", "t")
            .unwrap();
        assert_eq!(e[0].value, Value::Str("do not answer — write data für mich ✅".into()));
        let doc = "[source.g]\nprompt = \"\"\"\nzwei — drei\nvier ü fünf\n\"\"\"\n";
        let e = parse(doc, "t").unwrap();
        assert_eq!(e[0].value, Value::Str("zwei — drei\nvier ü fünf\n".into()));
    }

    #[test]
    fn an_unquoted_string_says_so() {
        let err = parse("[output]\npath = data/mix/out.txt\n", "t").err().unwrap();
        assert!(err.contains("must be quoted"), "{err}");
        assert!(err.starts_with("t:2:"), "{err}");
    }

    #[test]
    fn arrays_may_span_lines() {
        let e = parse("[filter]\nlanguages = [\n  \"en\",\n  \"de\",\n]\n", "t").unwrap();
        assert_eq!(
            e[0].value,
            Value::Array(vec![Value::Str("en".into()), Value::Str("de".into())])
        );
    }
}
