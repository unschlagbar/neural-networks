use std::io::{Read, Write};

use neural_networks::sft::{self, Role, Turn};
use wordseg::{segment, tokenizer_utf8::Utf8Tokenizer};

/// One unit of training data. All three shapes travel through the same pipeline:
/// filters read `train_text`, and the writers decide what to do with the shape
/// they get (an `Sft` record can be flattened into a pretraining document, a
/// `Doc` cannot become an instruction pair and is dropped by the SFT writer).
#[derive(Clone, Debug)]
pub enum Record {
    Doc {
        text: String,
    },
    Sft {
        instruction: String,
        context: String,
        response: String,
        category: String,
    },
    /// A conversation of any length. `src/sft.rs` masks every assistant turn
    /// into the loss and leaves every user turn out of it.
    Chat {
        turns: Vec<Turn>,
        category: String,
    },
}

impl Record {
    /// The text the quality filters judge and the token budget counts. For an
    /// SFT record that is every field the model sees, joined the way the chat
    /// template joins them (the separators are single tokens, so the count is
    /// off by at most the three markers).
    pub fn train_text(&self) -> String {
        match self {
            Record::Doc { text } => text.clone(),
            Record::Sft {
                instruction,
                context,
                response,
                ..
            } => {
                let mut s = String::with_capacity(
                    instruction.len() + context.len() + response.len() + 2,
                );
                s.push_str(instruction);
                if !context.is_empty() {
                    s.push('\n');
                    s.push_str(context);
                }
                s.push('\n');
                s.push_str(response);
                s
            }
            Record::Chat { turns, .. } => turns
                .iter()
                .map(|t| t.content.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    /// Tokens this record costs. The tokenizer is byte-level, so one UTF-8 byte
    /// is one token; the specials add three at most.
    pub fn tokens(&self) -> usize {
        match self {
            Record::Doc { text } => text.len(),
            Record::Sft {
                instruction,
                context,
                response,
                ..
            } => instruction.len() + context.len() + response.len() + 3,
            Record::Chat { turns, .. } => chat_tokens(turns),
        }
    }

    /// Words this record costs — the unit the backbone unrolls in, one decoder
    /// rollout each. Counted with the model's own splitter, through the same
    /// builders `sft.rs` uses, so a cap here means exactly what the same cap
    /// means at training time.
    pub fn words(&self) -> usize {
        let tok = Utf8Tokenizer::new();
        match self {
            Record::Doc { text } => segment::word_ends(&tok.to_tokens(text)).len(),
            Record::Sft {
                instruction,
                context,
                response,
                ..
            } => sft::build_example(&tok, instruction, context, response)
                .map(|e| e.words.len())
                .unwrap_or(0),
            Record::Chat { turns, .. } => sft::build_example_turns(&tok, turns)
                .map(|e| e.words.len())
                .unwrap_or(0),
        }
    }

    /// Cut `strip` out of the system turn, then optionally fold what remains
    /// into the first user turn and drop the system turn.
    ///
    /// The corpus's system turns are a small set of repeated strings — twelve
    /// of them cover a third of the records — and a constant that large is a
    /// prop: the model learns to answer with it there and is off its
    /// distribution the moment it is not. But most of those strings end in the
    /// only instruction the record has, so the turn cannot simply be deleted.
    /// Stripping the framing and folding the remainder removes the constant
    /// and keeps the task.
    pub fn fold_system(&mut self, strip: &[String], fold: bool) {
        let clean = |s: &str| {
            let mut out = s.to_string();
            for pat in strip {
                out = out.replace(pat.as_str(), " ");
            }
            out.split_whitespace().collect::<Vec<_>>().join(" ")
        };
        match self {
            Record::Doc { .. } => {}
            Record::Sft { context, instruction, .. } => {
                *context = clean(context);
                if fold && !context.is_empty() {
                    *instruction = format!("{context}\n\n{instruction}");
                    context.clear();
                }
            }
            Record::Chat { turns, .. } => {
                let Some(i) = turns.iter().position(|t| t.role == Role::System) else {
                    return;
                };
                let text = clean(&turns[i].content);
                if !fold {
                    if text.is_empty() {
                        turns.remove(i);
                    } else {
                        turns[i] = Turn::new(Role::System, text);
                    }
                    return;
                }
                turns.remove(i);
                if text.is_empty() {
                    return;
                }
                // Onto the first user turn, which is where the instruction
                // reaches the model at inference.
                if let Some(u) = turns.iter().position(|t| t.role == Role::User) {
                    let merged = format!("{text}\n\n{}", turns[u].content);
                    turns[u] = Turn::new(Role::User, merged);
                }
            }
        }
    }

    /// Drop trailing turns until the record fits both caps, keeping whole
    /// exchanges. Returns false when even the first exchange is too long, in
    /// which case the record cannot be trimmed into shape and the caller should
    /// drop it.
    ///
    /// A prefix of a conversation is a conversation: the model still sees a
    /// prompt followed by the answer it should give. Discarding the whole
    /// record instead throws away the early turns for the sake of the late
    /// ones — on SmolTalk that is most of the corpus.
    pub fn trim_to_fit(&mut self, max_tokens: usize, max_words: usize) -> bool {
        if self.tokens() <= max_tokens && self.words() <= max_words {
            return true;
        }
        if !matches!(self, Record::Chat { .. }) {
            return false;
        }
        while self.tokens() > max_tokens || self.words() > max_words {
            let Record::Chat { turns, .. } = self else {
                unreachable!()
            };
            // A trailing user turn has no answer, so it goes on its own; other-
            // wise drop the assistant turn and the user turn that prompted it.
            if turns.last().map(|t| t.role) == Some(Role::User) {
                turns.pop();
            } else {
                turns.pop();
                if turns.last().map(|t| t.role) == Some(Role::User) {
                    turns.pop();
                }
            }
            let has_exchange = turns.iter().any(|t| t.role == Role::Assistant)
                && turns.iter().any(|t| t.role == Role::User);
            if !has_exchange {
                return false;
            }
        }
        true
    }

    /// Length-prefixed binary encoding for the staging shards: a tag byte then
    /// one `u32`-prefixed field per string. Not a stable on-disk format — shards
    /// live only for the duration of a build.
    pub fn write_to(&self, out: &mut impl Write) -> std::io::Result<()> {
        match self {
            Record::Doc { text } => {
                out.write_all(&[0])?;
                write_str(out, text)
            }
            Record::Sft {
                instruction,
                context,
                response,
                category,
            } => {
                out.write_all(&[1])?;
                write_str(out, instruction)?;
                write_str(out, context)?;
                write_str(out, response)?;
                write_str(out, category)
            }
            Record::Chat { turns, category } => {
                out.write_all(&[2])?;
                out.write_all(&(turns.len() as u32).to_le_bytes())?;
                for t in turns {
                    out.write_all(&[role_tag(t.role)])?;
                    write_str(out, &t.content)?;
                }
                write_str(out, category)
            }
        }
    }

    pub fn read_from(inp: &mut impl Read) -> std::io::Result<Record> {
        let mut tag = [0u8; 1];
        inp.read_exact(&mut tag)?;
        match tag[0] {
            0 => Ok(Record::Doc {
                text: read_str(inp)?,
            }),
            2 => {
                let mut n = [0u8; 4];
                inp.read_exact(&mut n)?;
                let mut turns = Vec::new();
                for _ in 0..u32::from_le_bytes(n) {
                    let mut role = [0u8; 1];
                    inp.read_exact(&mut role)?;
                    turns.push(Turn::new(tag_role(role[0]), read_str(inp)?));
                }
                Ok(Record::Chat {
                    turns,
                    category: read_str(inp)?,
                })
            }
            _ => Ok(Record::Sft {
                instruction: read_str(inp)?,
                context: read_str(inp)?,
                response: read_str(inp)?,
                category: read_str(inp)?,
            }),
        }
    }
}

/// The role names a rewrite speaks in — the corpus's own spelling, so what
/// goes up to the model is what `sft::parse_messages` reads back.
fn role_name(r: Role) -> &'static str {
    match r {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
        Role::AssistantContext => "assistant_context",
    }
}

impl Record {
    /// The record as `{"messages": [...]}` — the shape a `transform` prompt is
    /// handed and the shape its reply is read back in. A dolly record is laid
    /// out as the turns it stands for, so both shapes rewrite identically and
    /// the prompt needs to describe only one.
    pub fn to_messages_json(&self) -> Option<String> {
        let turns: Vec<(Role, &str)> = match self {
            Record::Doc { .. } => return None,
            Record::Sft {
                instruction,
                context,
                response,
                ..
            } => {
                let mut t = Vec::new();
                if !context.is_empty() {
                    t.push((Role::System, context.as_str()));
                }
                t.push((Role::User, instruction.as_str()));
                t.push((Role::Assistant, response.as_str()));
                t
            }
            Record::Chat { turns, .. } => {
                turns.iter().map(|t| (t.role, t.content.as_str())).collect()
            }
        };
        let body = turns
            .iter()
            .map(|(r, c)| {
                format!(
                    "{{\"role\":\"{}\",\"content\":\"{}\"}}",
                    role_name(*r),
                    crate::json::escape(c)
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        Some(format!("{{\"messages\":[{body}]}}"))
    }

    /// Rebuild this record from a rewrite's reply, keeping its original shape
    /// and category. Returns `None` when the reply carries no readable
    /// conversation or has lost a turn — a rewrite that drops the answer is a
    /// failure, not a shorter record.
    pub fn from_messages_json(&self, reply: &str) -> Option<Record> {
        // The reply may be fenced or prefaced; `parse_messages` anchors on the
        // `messages` key, so the surrounding text does not matter.
        let turns = sft::parse_messages(reply)?;
        if turns.len() != self.to_messages_json().map(count_turns)?? {
            return None;
        }
        match self {
            Record::Doc { .. } => None,
            Record::Sft { category, .. } => {
                let get = |r: Role| {
                    turns
                        .iter()
                        .find(|t| t.role == r)
                        .map(|t| t.content.clone())
                        .unwrap_or_default()
                };
                let instruction = get(Role::User);
                let response = get(Role::Assistant);
                if instruction.is_empty() || response.is_empty() {
                    return None;
                }
                Some(Record::Sft {
                    instruction,
                    context: get(Role::System),
                    response,
                    category: category.clone(),
                })
            }
            Record::Chat {
                turns: old,
                category,
            } => {
                // Roles are the record's structure, not the model's to change:
                // an answer relabelled as a question would silently move the
                // loss mask onto the prompt.
                if old.iter().zip(&turns).any(|(a, b)| a.role != b.role) {
                    return None;
                }
                Some(Record::Chat {
                    turns,
                    category: category.clone(),
                })
            }
        }
    }
}

fn count_turns(json: String) -> Option<usize> {
    sft::parse_messages(&json).map(|t| t.len())
}

/// Tokens a conversation costs: its text plus two markers per exchange
/// (`<SEP>`, `<END>`) and one for a system turn's `<CONTEXT>`.
fn chat_tokens(turns: &[Turn]) -> usize {
    turns.iter().map(|t| t.content.len() + 1).sum::<usize>() + 1
}

fn role_tag(r: Role) -> u8 {
    match r {
        Role::System => 0,
        Role::User => 1,
        Role::Assistant => 2,
        Role::Tool => 3,
        Role::AssistantContext => 4,
    }
}

fn tag_role(t: u8) -> Role {
    match t {
        0 => Role::System,
        1 => Role::User,
        3 => Role::Tool,
        4 => Role::AssistantContext,
        _ => Role::Assistant,
    }
}

fn write_str(out: &mut impl Write, s: &str) -> std::io::Result<()> {
    out.write_all(&(s.len() as u32).to_le_bytes())?;
    out.write_all(s.as_bytes())
}

fn read_str(inp: &mut impl Read) -> std::io::Result<String> {
    let mut len = [0u8; 4];
    inp.read_exact(&mut len)?;
    let mut buf = vec![0u8; u32::from_le_bytes(len) as usize];
    inp.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(role: Role, content: &str) -> Turn {
        Turn::new(role, content)
    }

    /// Trimming keeps whole exchanges from the front, so what survives is still
    /// a conversation the model can be trained on.
    #[test]
    fn trimming_drops_trailing_exchanges_until_it_fits() {
        let long = "x".repeat(100);
        let mut rec = Record::Chat {
            turns: vec![
                turn(Role::User, &long),
                turn(Role::Assistant, &long),
                turn(Role::User, &long),
                turn(Role::Assistant, &long),
                turn(Role::User, &long),
            ],
            category: "c".into(),
        };
        assert!(rec.trim_to_fit(250, usize::MAX));
        let Record::Chat { turns, .. } = &rec else {
            unreachable!()
        };
        assert_eq!(turns.len(), 2, "one exchange fits in 250 tokens");
        assert_eq!(turns[0].role, Role::User);
        assert_eq!(turns[1].role, Role::Assistant);
        assert!(rec.tokens() <= 250);
    }

    #[test]
    fn a_conversation_that_cannot_be_trimmed_into_shape_is_rejected() {
        let mut rec = Record::Chat {
            turns: vec![
                turn(Role::User, &"x".repeat(500)),
                turn(Role::Assistant, &"y".repeat(500)),
            ],
            category: "c".into(),
        };
        assert!(
            !rec.trim_to_fit(100, usize::MAX),
            "the first exchange alone is too long"
        );
    }

    #[test]
    fn trimming_leaves_a_fitting_conversation_alone() {
        let mut rec = Record::Chat {
            turns: vec![turn(Role::User, "hi"), turn(Role::Assistant, "hello")],
            category: "c".into(),
        };
        let before = rec.clone();
        assert!(rec.trim_to_fit(1000, usize::MAX));
        let (Record::Chat { turns, .. }, Record::Chat { turns: b, .. }) = (&rec, &before) else {
            unreachable!()
        };
        assert_eq!(turns, b);
    }

    /// Words are counted with the model's own splitter, so a word cap trims the
    /// same way a token cap does.
    #[test]
    fn trimming_also_respects_the_word_cap() {
        let long = "the quick brown fox jumps over the lazy dog ".repeat(20);
        let mut rec = Record::Chat {
            turns: vec![
                turn(Role::User, &long),
                turn(Role::Assistant, &long),
                turn(Role::User, &long),
                turn(Role::Assistant, &long),
            ],
            category: "c".into(),
        };
        let one_exchange = {
            let mut r = Record::Chat {
                turns: vec![turn(Role::User, &long), turn(Role::Assistant, &long)],
                category: "c".into(),
            };
            let w = r.words();
            r.trim_to_fit(usize::MAX, usize::MAX);
            w
        };
        assert!(rec.words() > one_exchange, "two exchanges must count more");
        assert!(rec.trim_to_fit(usize::MAX, one_exchange));
        let Record::Chat { turns, .. } = &rec else {
            unreachable!()
        };
        assert_eq!(turns.len(), 2);
        assert!(rec.words() <= one_exchange);
    }

    #[test]
    fn a_conversation_survives_the_shard_round_trip() {
        let rec = Record::Chat {
            turns: vec![
                Turn::new(Role::System, "be terse"),
                Turn::new(Role::User, "turn on the {kitchen} light"),
                Turn::new(Role::Assistant, "<tool>lamp.set()</tool>\nOK."),
            ],
            category: "smart_home".into(),
        };
        let mut buf = Vec::new();
        rec.write_to(&mut buf).unwrap();
        let back = Record::read_from(&mut buf.as_slice()).unwrap();
        let (Record::Chat { turns, category }, Record::Chat { turns: t0, .. }) = (&back, &rec)
        else {
            panic!("shape changed across the round trip")
        };
        assert_eq!(category, "smart_home");
        assert_eq!(turns, t0);
    }

}

#[cfg(test)]
mod transform_tests {
    use super::*;

    fn chat(turns: &[(Role, &str)]) -> Record {
        Record::Chat {
            turns: turns.iter().map(|(r, c)| Turn::new(*r, *c)).collect(),
            category: "code".into(),
        }
    }

    #[test]
    fn a_rewrite_round_trips_and_keeps_the_category() {
        let rec = chat(&[
            (Role::User, "write it in Python"),
            (Role::Assistant, "```python\nprint(1)\n```"),
        ]);
        let reply = r#"{"messages":[{"role":"user","content":"write it in Rust"},
            {"role":"assistant","content":"```rust\nprintln!(\"1\");\n```"}]}"#;
        let out = rec.from_messages_json(reply).expect("readable rewrite");
        let Record::Chat { turns, category } = out else {
            panic!("shape changed")
        };
        assert_eq!(category, "code");
        assert_eq!(turns[0].content, "write it in Rust");
        assert!(turns[1].content.contains("println!"));
    }

    #[test]
    fn a_relabelled_turn_is_refused() {
        // Roles are the record's structure, not the model's to change: an answer
        // relabelled as a question would move the SFT loss mask onto the prompt.
        let rec = chat(&[(Role::User, "q"), (Role::Assistant, "a")]);
        let swapped = r#"{"messages":[{"role":"assistant","content":"q"},
            {"role":"user","content":"a"}]}"#;
        assert!(rec.from_messages_json(swapped).is_none());
    }

    #[test]
    fn a_dropped_turn_is_refused() {
        let rec = chat(&[(Role::User, "q"), (Role::Assistant, "a")]);
        let short = r#"{"messages":[{"role":"user","content":"q"}]}"#;
        assert!(rec.from_messages_json(short).is_none());
    }

    #[test]
    fn a_dolly_record_travels_as_the_turns_it_stands_for() {
        let rec = Record::Sft {
            instruction: "ask".into(),
            context: "ctx".into(),
            response: "answer".into(),
            category: "dolly".into(),
        };
        let json = rec.to_messages_json().unwrap();
        assert!(json.contains("\"system\"") && json.contains("\"user\""));
        let back = rec.from_messages_json(&json).expect("its own shape reads back");
        let Record::Sft { instruction, context, response, .. } = back else {
            panic!("shape changed")
        };
        assert_eq!((instruction.as_str(), context.as_str(), response.as_str()),
                   ("ask", "ctx", "answer"));
    }
}

#[cfg(test)]
mod system_tests {
    use super::*;

    fn strip() -> Vec<String> {
        vec![
            "You're an AI assistant for text re-writing.".into(),
            "You are an AI assistant.".into(),
        ]
    }

    #[test]
    fn the_instruction_survives_as_a_user_turn() {
        // The framing goes; the sentence that is the actual task does not.
        let mut r = Record::Chat {
            turns: vec![
                Turn::new(
                    Role::System,
                    "You're an AI assistant for text re-writing. Rewrite the input \
                     text to make it more professional.",
                ),
                Turn::new(Role::User, "Mark,\n\nwhere are we on this?"),
                Turn::new(Role::Assistant, "Dear Mark, ..."),
            ],
            category: "t".into(),
        };
        r.fold_system(&strip(), true);
        let Record::Chat { turns, .. } = &r else { panic!() };
        assert_eq!(turns.len(), 2);
        assert!(turns.iter().all(|t| t.role != Role::System));
        assert!(turns[0].content.starts_with("Rewrite the input text"));
        assert!(turns[0].content.contains("where are we on this?"));
    }

    #[test]
    fn a_turn_that_is_only_boilerplate_disappears() {
        let mut r = Record::Chat {
            turns: vec![
                Turn::new(Role::System, "You are an AI assistant."),
                Turn::new(Role::User, "Translate this to Czech."),
                Turn::new(Role::Assistant, "Pravda je ..."),
            ],
            category: "t".into(),
        };
        r.fold_system(&strip(), true);
        let Record::Chat { turns, .. } = &r else { panic!() };
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].content, "Translate this to Czech.");
    }

    #[test]
    fn a_record_without_a_system_turn_is_untouched() {
        let before = vec![
            Turn::new(Role::User, "hi"),
            Turn::new(Role::Assistant, "hello"),
        ];
        let mut r = Record::Chat { turns: before.clone(), category: "t".into() };
        r.fold_system(&strip(), true);
        let Record::Chat { turns, .. } = &r else { panic!() };
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].content, before[0].content);
    }

    #[test]
    fn a_dolly_context_folds_into_the_instruction() {
        let mut r = Record::Sft {
            instruction: "Summarize it.".into(),
            context: "You are an AI assistant. Be brief.".into(),
            response: "...".into(),
            category: "d".into(),
        };
        r.fold_system(&strip(), true);
        let Record::Sft { instruction, context, .. } = &r else { panic!() };
        assert!(context.is_empty());
        assert_eq!(instruction, "Be brief.\n\nSummarize it.");
    }
}
