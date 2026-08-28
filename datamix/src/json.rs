// The slice of JSON this crate needs: pull a string field out of a flat-ish
// object, and escape a string into one. Dependency-free, like `src/sft.rs`
// (which parses the same dolly records the same way).

/// Decode the JSON string value of `key`. The scan anchors on a *key* position
/// (`"key"` followed by a colon), so text that merely contains the key name
/// inside a value cannot match. Nested objects are searched too — that is what
/// finds `choices[0].message.content` in a chat completion without a real
/// parser.
pub fn field(text: &str, key: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let needle = format!("\"{key}\"");
    let mut from = 0usize;
    loop {
        let at = text[from..].find(&needle)? + from;
        let mut i = at + needle.len();
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b':' {
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i < bytes.len() && bytes[i] == b'"' {
                return decode(&text[i + 1..]);
            }
            return None;
        }
        from = at + needle.len();
    }
}

/// Decode a JSON string body, starting just after its opening quote.
pub fn decode(rest: &str) -> Option<String> {
    let mut out = String::new();
    let mut it = rest.chars();
    while let Some(c) = it.next() {
        match c {
            '"' => return Some(out),
            '\\' => match it.next()? {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                'r' => out.push('\r'),
                'b' => out.push('\u{8}'),
                'f' => out.push('\u{c}'),
                'u' => {
                    let hex: String = (0..4).filter_map(|_| it.next()).collect();
                    let cp = u32::from_str_radix(&hex, 16).ok()?;
                    // A surrogate half on its own is not a char; the pair form
                    // does not appear in the responses we read.
                    out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                }
                other => out.push(other),
            },
            _ => out.push(c),
        }
    }
    None
}

pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_a_nested_field() {
        let body = r#"{"id":"x","choices":[{"index":0,"message":{"role":"assistant",
            "content":"turn on the {lamp}\nline two"}}],"usage":{"total_tokens":9}}"#;
        assert_eq!(
            field(body, "content").unwrap(),
            "turn on the {lamp}\nline two"
        );
        assert_eq!(field(body, "role").unwrap(), "assistant");
        assert_eq!(field(body, "missing"), None);
    }

    #[test]
    fn a_key_name_inside_a_value_does_not_match() {
        let body = r#"{"content":"the word content: is in here","role":"user"}"#;
        assert_eq!(field(body, "content").unwrap(), "the word content: is in here");
        assert_eq!(field(body, "role").unwrap(), "user");
    }

    #[test]
    fn escape_round_trips() {
        let s = "he said \"hi\"\n\tand \\ left";
        assert_eq!(decode(&format!("{}\"", escape(s))).unwrap(), s);
    }
}
