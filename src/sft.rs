// Supervised fine-tuning (SFT) dataset for instruction / Q-A post-training.
//
// Reads a JSONL file of `{instruction, context, response, category}` records
// (the databricks-dolly-15k layout) and turns each into ONE training window for
// the hierarchical model: a token sequence with chat markers, segmented into
// words, plus a per-word **loss mask** so gradient flows only from the
// assistant response.
//
// Prompt layout (context section omitted when the record has no context):
//
//   {instruction} <CONTEXT> {context} <SEP> {response} <END>
//   └──────────── prompt (mask 0) ───────────┘└── response (mask 1) ──┘
//
// The whole sequence is encoded and decoded as usual — the backbone must read
// the prompt to condition on it — but only the words at or after `<SEP>` (the
// response and its `<END>`) are counted in the loss. That is standard SFT
// masking: the model learns to *produce* the response, not to reproduce the
// prompt it was given.
//
// Words come from `crate::segment`, which already treats a special token (id
// ≥ 256) as its own one-token word, so `<SEP>`, `<CONTEXT>` and `<END>` each
// land on a clean word boundary and the mask is a per-word flag.

use std::{
    fs::File,
    io::{BufRead, BufReader},
    range::Range,
};

use crate::{
    segment,
    tokenizer_utf8::{CONTEXT_TOKEN, END_TOKEN, SEP_TOKEN, Utf8Tokenizer},
};

/// One formatted SFT example: the token sequence, its word ranges, and a
/// per-word mask (`true` = counted in the loss).
pub struct SftExample {
    pub tokens: Vec<u16>,
    pub words: Vec<Range<usize>>,
    /// One flag per word, parallel to `words`: whether that word's decode is
    /// counted in the loss. `words[0]` is never decoded (it is the encode-only
    /// prefix), so its flag is unused; masking begins at the response.
    pub loss: Vec<bool>,
}

/// Assemble the token sequence for one record and mark the first response token.
/// Returns `(tokens, response_start)` where `response_start` is the token index
/// of the first response token (right after `<SEP>`).
fn build_tokens(
    tok: &Utf8Tokenizer,
    instruction: &str,
    context: &str,
    response: &str,
) -> (Vec<u16>, usize) {
    let mut tokens = tok.to_tokens(instruction.trim());
    if !context.trim().is_empty() {
        tokens.push(CONTEXT_TOKEN);
        tokens.extend(tok.to_tokens(context.trim()));
    }
    tokens.push(SEP_TOKEN);
    let response_start = tokens.len();
    tokens.extend(tok.to_tokens(response.trim()));
    tokens.push(END_TOKEN);
    (tokens, response_start)
}

/// Format an inference prompt: `{instruction}[<CONTEXT>{context}]<SEP>`, with
/// the trailing `<SEP>` that cues the model to start the response. This is the
/// prompt side of [`build_tokens`] with no response — feed it to
/// [`Hierarchical::sample_chat`](crate::hierarchical::Hierarchical::sample_chat).
pub fn format_prompt(tok: &Utf8Tokenizer, instruction: &str, context: &str) -> Vec<u16> {
    let mut tokens = tok.to_tokens(instruction.trim());
    if !context.trim().is_empty() {
        tokens.push(CONTEXT_TOKEN);
        tokens.extend(tok.to_tokens(context.trim()));
    }
    tokens.push(SEP_TOKEN);
    tokens
}

/// Segment `tokens` into words and build the loss mask: a word is a response
/// word iff it starts at or after `response_start`. `<SEP>` itself is a prompt
/// word (mask 0) — the model predicts the response FROM the separator, it does
/// not predict the separator as a response token.
fn segment_with_mask(tokens: &[u16], response_start: usize) -> (Vec<Range<usize>>, Vec<bool>) {
    let ends = segment::word_ends(tokens);
    let mut words = Vec::with_capacity(ends.len());
    let mut loss = Vec::with_capacity(ends.len());
    let mut start = 0usize;
    for &e in &ends {
        let end = e as usize;
        words.push(Range { start, end });
        loss.push(start >= response_start);
        start = end;
    }
    (words, loss)
}

/// Build one SFT example from a record, or `None` if it is too short to train on
/// (needs at least one prompt word and one response word).
pub fn build_example(
    tok: &Utf8Tokenizer,
    instruction: &str,
    context: &str,
    response: &str,
) -> Option<SftExample> {
    if instruction.trim().is_empty() || response.trim().is_empty() {
        return None;
    }
    let (tokens, response_start) = build_tokens(tok, instruction, context, response);
    let (words, loss) = segment_with_mask(&tokens, response_start);

    // A trainable window needs word 0 (encode-only prefix) plus at least one
    // response word carrying loss.
    if words.len() < 2 || !loss.iter().any(|&b| b) {
        return None;
    }
    Some(SftExample {
        tokens,
        words,
        loss,
    })
}

/// Load and format every record of a dolly-style JSONL file into SFT examples.
/// Records that fail to parse or are too short are skipped; records longer than
/// `max_tokens` are dropped (their per-example cache would dominate memory — see
/// `config::SFT_MAX_TOKENS`). Both are counted separately.
pub fn load_jsonl(
    tok: &Utf8Tokenizer,
    path: &str,
    max_tokens: usize,
) -> std::io::Result<Vec<SftExample>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut examples = Vec::new();
    let mut skipped = 0usize;
    let mut too_long = 0usize;
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let Some((instruction, context, response)) = parse_record(&line) else {
            skipped += 1;
            continue;
        };
        match build_example(tok, &instruction, &context, &response) {
            Some(ex) if ex.tokens.len() <= max_tokens => examples.push(ex),
            Some(_) => too_long += 1,
            None => skipped += 1,
        }
    }

    println!(
        "SFT: loaded {} examples from '{path}' ({skipped} skipped, \
         {too_long} over {max_tokens} tokens dropped)",
        examples.len(),
    );
    Ok(examples)
}

/// Pull the `instruction`, `context` and `response` string fields out of one
/// JSONL line. Missing `context` decodes as empty (dolly always has the key,
/// often with an empty value). Returns `None` if `instruction` or `response` is
/// absent or the line is not a JSON object.
fn parse_record(line: &str) -> Option<(String, String, String)> {
    let instruction = json_string_field(line, "instruction")?;
    let response = json_string_field(line, "response")?;
    let context = json_string_field(line, "context").unwrap_or_default();
    Some((instruction, context, response))
}

/// Find `"key"` in `line` and decode the JSON string value that follows its
/// colon. Dependency-free (no serde): the corpus is flat objects with string
/// values, so a scan for the key plus a JSON-string decode is enough. Returns
/// `None` if the key is missing or its value is not a string.
fn json_string_field(line: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    // Scan for the key as an object key: `"key"` followed (after whitespace) by
    // a colon. A plain `find` could match the same text inside a value, so the
    // colon check anchors it to a key position.
    let bytes = line.as_bytes();
    let mut from = 0;
    loop {
        let rel = line[from..].find(&needle)?;
        let mut i = from + rel + needle.len();
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b':' {
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i < bytes.len() && bytes[i] == b'"' {
                return decode_json_string(bytes, i);
            }
            return None; // value is not a string
        }
        from = from + rel + needle.len();
    }
}

/// Decode a JSON string starting at the opening quote at `start`. Handles the
/// standard escapes (`\" \\ \/ \n \r \t \b \f` and `\uXXXX`, including
/// surrogate pairs). Returns the decoded string, or `None` if unterminated.
fn decode_json_string(bytes: &[u8], start: usize) -> Option<String> {
    debug_assert_eq!(bytes[start], b'"');
    let mut out = String::new();
    let mut i = start + 1;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => return Some(out),
            b'\\' => {
                i += 1;
                let c = *bytes.get(i)?;
                match c {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000C}'),
                    b'u' => {
                        let cp = decode_hex4(bytes, i + 1)?;
                        i += 4;
                        // Surrogate pair: a high surrogate must be followed by
                        // `\uXXXX` low surrogate to form one code point.
                        if (0xD800..=0xDBFF).contains(&cp) {
                            if bytes.get(i + 1) == Some(&b'\\') && bytes.get(i + 2) == Some(&b'u') {
                                let lo = decode_hex4(bytes, i + 3)?;
                                i += 6;
                                let c = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                out.push(char::from_u32(c)?);
                            } else {
                                out.push('\u{FFFD}');
                            }
                        } else {
                            out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                        }
                    }
                    _ => return None,
                }
                i += 1;
            }
            _ => {
                // A raw UTF-8 byte run: copy until the next quote or backslash.
                let s = i;
                while i < bytes.len() && bytes[i] != b'"' && bytes[i] != b'\\' {
                    i += 1;
                }
                out.push_str(std::str::from_utf8(&bytes[s..i]).ok()?);
            }
        }
    }
    None // unterminated string
}

/// Decode four hex digits at `at` into a code-point value.
fn decode_hex4(bytes: &[u8], at: usize) -> Option<u32> {
    let mut v = 0u32;
    for k in 0..4 {
        let d = (*bytes.get(at + k)? as char).to_digit(16)?;
        v = v * 16 + d;
    }
    Some(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_escapes_and_unicode() {
        let line = r#"{"instruction": "a’b", "context": "", "response": "line1\nline2", "category": "x"}"#;
        let (instr, ctx, resp) = parse_record(line).unwrap();
        assert_eq!(instr, "a\u{2019}b");
        assert_eq!(ctx, "");
        assert_eq!(resp, "line1\nline2");
    }

    #[test]
    fn parses_surrogate_pair() {
        // U+1F600 encodes as the surrogate pair D83D DE00.
        let line = r#"{"instruction": "hi 😀", "response": "ok"}"#;
        let (instr, _, resp) = parse_record(line).unwrap();
        assert_eq!(instr, "hi \u{1F600}");
        assert_eq!(resp, "ok");
    }

    /// The mask is 0 across the prompt and 1 across the response, and the first
    /// response word is exactly the one that starts right after `<SEP>`.
    #[test]
    fn mask_covers_only_the_response() {
        let tok = Utf8Tokenizer::new();
        let ex = build_example(&tok, "what is 2+2?", "", "four").unwrap();

        // Reconstruct where the response begins.
        let sep_pos = ex
            .tokens
            .iter()
            .position(|&t| t == SEP_TOKEN)
            .expect("SEP present");
        for (w, &on) in ex.words.iter().zip(&ex.loss) {
            let expect = w.start > sep_pos; // words after the SEP word
            assert_eq!(on, expect, "word {w:?} mask mismatch");
        }
        // Response side has at least one loss word, prompt side has none-after.
        assert!(ex.loss.iter().any(|&b| b));
        // The very last word (the <END> word) carries loss.
        assert!(*ex.loss.last().unwrap());
    }

    /// A record with context inserts the `<CONTEXT>` marker and still masks only
    /// the response.
    #[test]
    fn context_section_is_prompt_side() {
        let tok = Utf8Tokenizer::new();
        let ex = build_example(&tok, "summarize", "a long passage here", "short").unwrap();
        assert!(ex.tokens.contains(&CONTEXT_TOKEN));
        assert!(ex.tokens.contains(&SEP_TOKEN));
        // Everything up to and including <SEP> is prompt (no loss).
        let sep_pos = ex.tokens.iter().position(|&t| t == SEP_TOKEN).unwrap();
        for (w, &on) in ex.words.iter().zip(&ex.loss) {
            if w.end <= sep_pos + 1 {
                assert!(!on, "prompt word {w:?} must not carry loss");
            }
        }
    }

    #[test]
    fn empty_response_is_skipped() {
        let tok = Utf8Tokenizer::new();
        assert!(build_example(&tok, "hello", "", "   ").is_none());
        assert!(build_example(&tok, "", "", "hi").is_none());
    }
}
