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
                    turns.push(Turn {
                        role: tag_role(role[0]),
                        content: read_str(inp)?,
                    });
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
        Turn {
            role,
            content: content.to_string(),
        }
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
                Turn { role: Role::System, content: "be terse".into() },
                Turn { role: Role::User, content: "turn on the {kitchen} light".into() },
                Turn { role: Role::Assistant, content: "<tool>lamp.set()</tool>\nOK.".into() },
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
