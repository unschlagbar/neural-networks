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
// A record may instead carry a `messages` array (`{"role", "content"}`, the
// OpenAI/LM-Studio shape), which is a conversation of any length:
//
//   {user1} <SEP> {asst1} <END> {user2} <SEP> {asst2} <END>
//   └ mask 0 ────┘└ mask 1 ────┘└ mask 0 ────┘└ mask 1 ────┘
//
// Every assistant turn carries loss, every user turn does not — the same rule
// as the single-turn case, applied once per turn. A `system` message takes the
// <CONTEXT> slot of the first turn, so multi-turn data needs no new tokens and
// no checkpoint surgery.
//
// A `tool` message is what a tool call returned. It sits on the prompt side
// like a user turn — the model did not write it, it reads it — and the reply
// after it carries loss:
//
//   {user} <SEP> <tool>app.launch(...)</tool> <END> <result>ok</result> <SEP> {asst} <END>
//   └ mask 0 ───┘└ mask 1 ────────────────────────┘└ mask 0 ─────────┘└ mask 1 ┘
//
// `<result>` is plain text, not a token: the tool syntax is a convention of the
// data, so a new tool protocol never costs a vocab entry.
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

/// Wrapper a tool result is rendered in. Plain text on purpose — see the
/// module header.
pub const RESULT_OPEN: &str = "<result>";
pub const RESULT_CLOSE: &str = "</result>";

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

impl SftExample {
    /// `(chars, words)` of the response — the part the loss is computed on.
    /// Word 0 is the encode-only prefix and never decodes, so it is excluded
    /// whatever its flag says.
    pub fn response_extent(&self) -> (usize, usize) {
        self.words
            .iter()
            .zip(&self.loss)
            .skip(1)
            .filter(|&(_, &keep)| keep)
            .fold((0, 0), |(chars, words), (r, _)| {
                (chars + (r.end - r.start), words + 1)
            })
    }
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

/// One message of a conversation.
#[derive(Clone, Debug, PartialEq)]
pub struct Turn {
    pub role: Role,
    pub content: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    /// What a tool call returned. Prompt-side: the model reads it, the reply
    /// after it is what carries loss.
    Tool,
    /// An assistant turn the model must READ but never learn to produce. It is
    /// laid out exactly like an assistant turn, `<END>` included, and carries no
    /// loss span — which is what lets a conversation contain a mistake the model
    /// then recovers from, without also teaching it to make that mistake.
    AssistantContext,
}

impl Role {
    /// Map a corpus's role string onto a role. `None` for anything the chat
    /// template has no place for (a tool call, say).
    pub fn parse(s: &str) -> Option<Role> {
        match s.trim().to_ascii_lowercase().as_str() {
            "system" | "developer" => Some(Role::System),
            "user" | "human" => Some(Role::User),
            "assistant" | "gpt" | "bot" => Some(Role::Assistant),
            "tool" | "function" | "observation" => Some(Role::Tool),
            "assistant_context" | "assistant_noloss" => Some(Role::AssistantContext),
            _ => None,
        }
    }
}

/// Pull a `messages` array out of one JSONL line. Returns `None` when the line
/// has no such key — that is how [`load_jsonl`] tells a conversation from a
/// dolly record without a second pass.
pub fn parse_messages(line: &str) -> Option<Vec<Turn>> {
    let at = key_position(line, "messages")?;
    let rest = &line[at..];
    let open = rest.find('[')?;
    let mut turns = Vec::new();
    for obj in json_objects(&rest[open + 1..]) {
        let (Some(role), Some(content)) = (
            json_string_field(obj, "role").and_then(|r| Role::parse(&r)),
            json_string_field(obj, "content"),
        ) else {
            continue;
        };
        turns.push(Turn { role, content });
    }
    (!turns.is_empty()).then_some(turns)
}

/// The `{...}` objects of an array body, as slices. Brace depth is tracked
/// outside strings only, so a brace inside a message body cannot end an object.
fn json_objects(body: &str) -> Vec<&str> {
    let bytes = body.as_bytes();
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut in_str = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_str {
            match b {
                _ if escaped => escaped = false,
                b'\\' => escaped = true,
                b'"' => in_str = false,
                _ => {}
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => {
                if depth == 0 {
                    start = i;
                }
                depth += 1;
            }
            b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    out.push(&body[start..=i]);
                }
            }
            b']' if depth == 0 => break,
            _ => {}
        }
    }
    out
}

/// Assemble the token sequence for a conversation, plus the token spans that
/// carry loss (one per assistant turn, its trailing `<END>` included).
///
/// A leading `system` message is folded into the first user turn's <CONTEXT>
/// slot. Anything before the first user turn, and any assistant turn with no
/// user turn before it, is dropped: the model is trained to answer, so every
/// loss span must have a prompt in front of it.
fn build_conversation(
    tok: &Utf8Tokenizer,
    turns: &[Turn],
    keep_open: bool,
) -> (Vec<u16>, Vec<(usize, usize)>) {
    let mut tokens: Vec<u16> = Vec::new();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut system = String::new();
    let mut open_prompt = false;

    for turn in turns {
        let content = turn.content.trim();
        match turn.role {
            Role::System => {
                if !content.is_empty() {
                    system = content.to_string();
                }
            }
            Role::User => {
                if content.is_empty() {
                    continue;
                }
                tokens.extend(tok.to_tokens(content));
                if !system.is_empty() {
                    tokens.push(CONTEXT_TOKEN);
                    tokens.extend(tok.to_tokens(&system));
                    system.clear();
                }
                tokens.push(SEP_TOKEN);
                open_prompt = true;
            }
            Role::Tool => {
                // A result with no call in front of it is nothing the model can
                // read; and it re-opens the prompt so the reply after it scores.
                if content.is_empty() || open_prompt {
                    continue;
                }
                tokens.extend(tok.to_tokens(&format!("{RESULT_OPEN}{content}{RESULT_CLOSE}")));
                tokens.push(SEP_TOKEN);
                open_prompt = true;
            }
            Role::Assistant | Role::AssistantContext => {
                if content.is_empty() || !open_prompt {
                    continue;
                }
                let start = tokens.len();
                tokens.extend(tok.to_tokens(content));
                tokens.push(END_TOKEN);
                // The only difference: a context turn records no loss span, so
                // it is read by the backbone and never predicted.
                if turn.role == Role::Assistant {
                    spans.push((start, tokens.len()));
                }
                open_prompt = false;
            }
        }
    }

    // A prompt with no answer after it teaches nothing and would be decoded
    // with no loss; drop the dangling tail. At inference that dangling prompt
    // is the whole point, so `keep_open` keeps it.
    if open_prompt && !keep_open {
        let cut = spans.last().map(|&(_, e)| e).unwrap_or(0);
        tokens.truncate(cut);
    }
    (tokens, spans)
}

/// Build one example from a conversation of any length, or `None` if no
/// assistant turn survives.
pub fn build_example_turns(tok: &Utf8Tokenizer, turns: &[Turn]) -> Option<SftExample> {
    let (tokens, spans) = build_conversation(tok, turns, false);
    if spans.is_empty() {
        return None;
    }
    let (words, loss) = segment_with_spans(&tokens, &spans);
    if words.len() < 2 || !loss.iter().any(|&b| b) {
        return None;
    }
    Some(SftExample {
        tokens,
        words,
        loss,
    })
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

/// Format a conversation for inference: every completed exchange followed by
/// the pending user turn and its `<SEP>`, which is what cues the model to
/// answer. The multi-turn twin of [`format_prompt`] — feed it to
/// [`Hierarchical::sample_chat`](crate::hierarchical::Hierarchical::sample_chat)
/// and append the reply as an `Assistant` turn to continue.
pub fn format_chat_prompt(tok: &Utf8Tokenizer, turns: &[Turn]) -> Vec<u16> {
    build_conversation(tok, turns, true).0
}

/// Segment `tokens` into words and build the loss mask: a word is a response
/// word iff it starts at or after `response_start`. `<SEP>` itself is a prompt
/// word (mask 0) — the model predicts the response FROM the separator, it does
/// not predict the separator as a response token.
fn segment_with_mask(tokens: &[u16], response_start: usize) -> (Vec<Range<usize>>, Vec<bool>) {
    segment_with_spans(tokens, &[(response_start, tokens.len())])
}

/// Segment `tokens` into words and mark every word that begins inside one of
/// the loss `spans`. Words tile the sequence and a special token is always its
/// own word, so a span border is always a word border.
fn segment_with_spans(tokens: &[u16], spans: &[(usize, usize)]) -> (Vec<Range<usize>>, Vec<bool>) {
    let ends = segment::word_ends(tokens);
    let mut words = Vec::with_capacity(ends.len());
    let mut loss = Vec::with_capacity(ends.len());
    let mut start = 0usize;
    for &e in &ends {
        let end = e as usize;
        words.push(Range { start, end });
        loss.push(spans.iter().any(|&(s, t)| start >= s && start < t));
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

/// Load and format every record of an SFT JSONL file into examples. Records
/// that fail to parse or are too short are skipped; records over
/// `config::SFT_MAX_WORDS` words or `max_tokens` tokens are dropped — the
/// per-example cache holds one slot per token and one rollout per word, so a
/// single outsized record sets the memory cost of the whole run. Both are
/// counted separately.
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
    let mut multi = 0usize;
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        // A `messages` array is a conversation; anything else is a dolly
        // record. Both end up as one window with the same per-word mask.
        let built = match parse_messages(&line) {
            Some(turns) => {
                multi += usize::from(turns.iter().filter(|t| t.role == Role::Assistant).count() > 1);
                build_example_turns(tok, &turns)
            }
            None => match parse_record(&line) {
                Some((instruction, context, response)) => {
                    build_example(tok, &instruction, &context, &response)
                }
                None => {
                    skipped += 1;
                    continue;
                }
            },
        };
        match built {
            Some(ex)
                if ex.tokens.len() <= max_tokens
                    && ex.words.len() <= crate::config::SFT_MAX_WORDS =>
            {
                examples.push(ex)
            }
            Some(_) => too_long += 1,
            None => skipped += 1,
        }
    }

    println!(
        "SFT: loaded {} examples from '{path}' ({multi} multi-turn, {skipped} skipped, \
         {too_long} over {max_words} words / {max_tokens} tokens dropped)",
        examples.len(),
        max_words = crate::config::SFT_MAX_WORDS,
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
/// Byte index just past `"key"` when it appears as an object *key* (followed
/// by a colon), else `None`.
fn key_position(line: &str, key: &str) -> Option<usize> {
    let bytes = line.as_bytes();
    let needle = format!("\"{key}\"");
    let mut from = 0usize;
    loop {
        let rel = line[from..].find(&needle)?;
        let mut i = from + rel + needle.len();
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b':' {
            return Some(i + 1);
        }
        from = from + rel + needle.len();
    }
}

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
        let line =
            r#"{"instruction": "a’b", "context": "", "response": "line1\nline2", "category": "x"}"#;
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

    /// `response_extent` counts exactly the loss-carrying words and their
    /// tokens — the prompt side and the never-decoded word 0 stay out.
    #[test]
    fn response_extent_counts_only_loss_words() {
        let tok = Utf8Tokenizer::new();
        let ex = build_example(&tok, "what is 2+2?", "", "four").unwrap();
        let (chars, words) = ex.response_extent();

        let expect_words = ex.loss.iter().skip(1).filter(|&&b| b).count();
        let expect_chars: usize = ex
            .words
            .iter()
            .zip(&ex.loss)
            .skip(1)
            .filter(|&(_, &on)| on)
            .map(|(w, _)| w.end - w.start)
            .sum();
        assert_eq!((chars, words), (expect_chars, expect_words));
        assert!(words > 0 && chars > 0, "response must be counted");
        assert!(
            chars < ex.tokens.len(),
            "the prompt must not be counted as response"
        );
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

    #[test]
    fn parses_a_messages_array() {
        let line = r#"{"messages": [{"role": "user", "content": "turn on the {kitchen} light"},
            {"role": "assistant", "content": "<tool>lamp.set(room=\"kitchen\")</tool>"}],
            "category": "smart_home"}"#;
        let turns = parse_messages(line).unwrap();
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].role, Role::User);
        // Braces and escaped quotes inside a message must not end the object.
        assert_eq!(turns[0].content, "turn on the {kitchen} light");
        assert_eq!(turns[1].content, "<tool>lamp.set(room=\"kitchen\")</tool>");
        assert!(parse_messages(r#"{"instruction": "x", "response": "y"}"#).is_none());
    }

    /// Every assistant turn carries loss, every user turn does not — the whole
    /// point of multi-turn masking.
    #[test]
    fn each_assistant_turn_is_masked_in() {
        let tok = Utf8Tokenizer::new();
        let turns = vec![
            Turn { role: Role::User, content: "first question".into() },
            Turn { role: Role::Assistant, content: "first answer".into() },
            Turn { role: Role::User, content: "second question".into() },
            Turn { role: Role::Assistant, content: "second answer".into() },
        ];
        let ex = build_example_turns(&tok, &turns).unwrap();

        // Two <END>s, two <SEP>s: two complete exchanges in one window.
        assert_eq!(ex.tokens.iter().filter(|&&t| t == END_TOKEN).count(), 2);
        assert_eq!(ex.tokens.iter().filter(|&&t| t == SEP_TOKEN).count(), 2);

        let text_of = |r: &Range<usize>| tok.to_text(&ex.tokens[r.start..r.end]);
        for (w, &keep) in ex.words.iter().zip(&ex.loss) {
            let t = text_of(w);
            if t.contains("question") {
                assert!(!keep, "prompt word {t:?} must not carry loss");
            }
            if t.contains("answer") {
                assert!(keep, "response word {t:?} must carry loss");
            }
        }
        // Both answers are counted, not just the last one.
        let (_, loss_words) = ex.response_extent();
        assert_eq!(loss_words, 6, "2 words per answer plus one <END> each");
    }

    /// The whole point of a tool result: the model is trained to write the
    /// reply that follows it, and never to write the result itself.
    #[test]
    fn a_tool_result_is_prompt_side() {
        let tok = Utf8Tokenizer::new();
        let turns = vec![
            Turn { role: Role::User, content: "turn on the light".into() },
            Turn { role: Role::Assistant, content: "<tool>lamp.set(on=true)</tool>".into() },
            Turn { role: Role::Tool, content: "already_on".into() },
            Turn { role: Role::Assistant, content: "It's already on.".into() },
        ];
        let ex = build_example_turns(&tok, &turns).unwrap();
        let text = tok.to_text(&ex.tokens);
        assert!(text.contains("<result>already_on</result>"), "{text:?}");

        // Two prompts (the user turn and the result), two scored replies.
        assert_eq!(ex.tokens.iter().filter(|&&t| t == SEP_TOKEN).count(), 2);
        assert_eq!(ex.tokens.iter().filter(|&&t| t == END_TOKEN).count(), 2);

        for (w, &keep) in ex.words.iter().zip(&ex.loss) {
            let t = tok.to_text(&ex.tokens[w.start..w.end]);
            assert!(!keep || !t.contains("already_on"), "the result must not be scored");
        }
        // The call and the reply after the result both carry loss.
        let scored: String = ex
            .words
            .iter()
            .zip(&ex.loss)
            .filter(|&(_, &k)| k)
            .map(|(w, _)| tok.to_text(&ex.tokens[w.start..w.end]))
            .collect();
        assert!(scored.contains("lamp.set(on=true)"), "{scored:?}");
        assert!(scored.contains("It's already on."), "{scored:?}");
    }

    /// The point of a masked assistant turn: the mistake is in the context the
    /// backbone reads, and in none of the gradient.
    #[test]
    fn a_context_assistant_turn_carries_no_loss() {
        let tok = Utf8Tokenizer::new();
        let turns = vec![
            Turn { role: Role::User, content: "turn the light on".into() },
            Turn {
                role: Role::AssistantContext,
                content: "<tool>lamp.set(on=false)</tool>".into(),
            },
            Turn { role: Role::Tool, content: "already_off".into() },
            Turn {
                role: Role::Assistant,
                content: "<tool>lamp.set(on=true)</tool>".into(),
            },
        ];
        let ex = build_example_turns(&tok, &turns).unwrap();
        let text = tok.to_text(&ex.tokens);

        // The mistake is present, and laid out like any assistant turn.
        assert!(text.contains("lamp.set(on=false)"), "{text:?}");
        assert_eq!(ex.tokens.iter().filter(|&&t| t == END_TOKEN).count(), 2);

        let scored: String = ex
            .words
            .iter()
            .zip(&ex.loss)
            .filter(|&(_, &k)| k)
            .map(|(w, _)| tok.to_text(&ex.tokens[w.start..w.end]))
            .collect();
        assert!(
            !scored.contains("on=false"),
            "the mistake must not be trained on: {scored:?}"
        );
        assert!(scored.contains("lamp.set(on=true)"), "{scored:?}");
    }

    /// A conversation whose only assistant turns are masked has nothing to
    /// learn from, however long it is.
    #[test]
    fn a_conversation_of_only_context_turns_is_rejected() {
        let tok = Utf8Tokenizer::new();
        let turns = vec![
            Turn { role: Role::User, content: "hello".into() },
            Turn { role: Role::AssistantContext, content: "wrong".into() },
        ];
        assert!(build_example_turns(&tok, &turns).is_none());
    }

    /// A result with no call in front of it is not something the model can read.
    #[test]
    fn a_tool_result_without_a_call_is_dropped() {
        let tok = Utf8Tokenizer::new();
        let turns = vec![
            Turn { role: Role::User, content: "hello".into() },
            Turn { role: Role::Tool, content: "ok".into() },
            Turn { role: Role::Assistant, content: "hi".into() },
        ];
        let ex = build_example_turns(&tok, &turns).unwrap();
        assert!(!tok.to_text(&ex.tokens).contains("<result>"));
    }

    #[test]
    fn a_system_message_takes_the_context_slot_and_carries_no_loss() {
        let tok = Utf8Tokenizer::new();
        let turns = vec![
            Turn { role: Role::System, content: "you are terse".into() },
            Turn { role: Role::User, content: "hi".into() },
            Turn { role: Role::Assistant, content: "hello".into() },
        ];
        let ex = build_example_turns(&tok, &turns).unwrap();
        assert_eq!(ex.tokens.iter().filter(|&&t| t == CONTEXT_TOKEN).count(), 1);
        let text = tok.to_text(&ex.tokens);
        assert!(text.contains("you are terse"), "{text}");
        for (w, &keep) in ex.words.iter().zip(&ex.loss) {
            if tok.to_text(&ex.tokens[w.start..w.end]).contains("terse") {
                assert!(!keep, "the system prompt must not carry loss");
            }
        }
    }

    #[test]
    fn a_dangling_user_turn_is_dropped() {
        let tok = Utf8Tokenizer::new();
        let turns = vec![
            Turn { role: Role::User, content: "answered".into() },
            Turn { role: Role::Assistant, content: "yes".into() },
            Turn { role: Role::User, content: "unanswered".into() },
        ];
        let ex = build_example_turns(&tok, &turns).unwrap();
        let text = tok.to_text(&ex.tokens);
        assert!(!text.contains("unanswered"), "{text}");
        assert!(text.ends_with(&tok.to_text(&[END_TOKEN])) || ex.tokens.last() == Some(&END_TOKEN));
    }

    #[test]
    fn an_assistant_only_conversation_is_rejected() {
        let tok = Utf8Tokenizer::new();
        let turns = vec![Turn { role: Role::Assistant, content: "unprompted".into() }];
        assert!(build_example_turns(&tok, &turns).is_none());
    }
}
