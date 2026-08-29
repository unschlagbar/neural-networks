use std::io::{Write, stdin, stdout};

use crate::{
    config::{MAX_LEN, MAX_SEQ_LEN, TEMPERATURE, TOP_P},
    hierarchical::Hierarchical,
    sequential::Sequential,
    sft::{Role, Turn},
    tokenizer_utf8::{BYTE_TOKENS, TOOL_OPEN_TOKEN, Utf8Tokenizer},
};

/// How many times one user turn may go round the call-result loop before the
/// REPL gives up on it concluding.
const MAX_TOOL_ROUNDS: usize = 6;

/// Streams sampled byte tokens to stdout. A UTF-8 character spans several byte
/// tokens, so bytes are held back until they form a complete character.
#[derive(Default)]
struct Utf8Printer {
    pending: Vec<u8>,
}

impl Utf8Printer {
    /// Print `token` once its character is complete. Returns false on `<END>`.
    fn print(&mut self, token: u16, tokenizer: &Utf8Tokenizer) -> bool {
        if token == tokenizer.end_token() {
            return false;
        }
        if (token as usize) >= BYTE_TOKENS {
            // Markup (`<tool>`, `<think>`, …) is part of the reply and is
            // spelled out; the structural markers are model-internal.
            if tokenizer.is_markup(token) {
                print!("{}", tokenizer.display(token));
                stdout().flush().unwrap();
            }
            return true;
        }

        self.pending.push(token as u8);
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                print!("{s}");
                stdout().flush().unwrap();
                self.pending.clear();
            }
            // Incomplete character: keep collecting. Anything else is a byte the
            // model made up that can never complete — drop it.
            Err(e) if e.error_len().is_none() => {}
            Err(_) => self.pending.clear(),
        }
        true
    }
}

pub fn sample_normal(model_path: &str) {
    let tokenizer = Utf8Tokenizer::new();

    let mut model = match Sequential::load(model_path) {
        Ok(m) => {
            println!("Loaded sequential model from '{model_path}'.");
            m
        }
        Err(e) => {
            eprintln!("Failed to load '{model_path}': {e}");
            std::process::exit(1);
        }
    };

    // For single-step sampling, cache[0] is sufficient.
    model.make_cache(1);

    loop {
        println!("\nSample mode — type a prefix (empty = random start, Ctrl+D = quit):");
        let mut input = String::new();
        if stdin().read_line(&mut input).unwrap() == 0 {
            println!();
            return;
        }

        let prefix: Vec<u16> = if !input.trim().is_empty() {
            tokenizer.to_tokens(input.trim())
        } else {
            Vec::new()
        };

        print!(">>> ");
        stdout().flush().unwrap();

        let mut printer = Utf8Printer::default();
        model.sample(&prefix, MAX_LEN, TEMPERATURE, TOP_P, |token| {
            printer.print(token, &tokenizer)
        });

        println!();
    }
}

/// Interactive Q-A sampling for an SFT (instruction-tuned) model. Each line the
/// user types is an instruction; it is wrapped as `{instruction}<SEP>`, encoded
/// as context, and the model generates the response until `<END>`. An empty
/// line lets the user add a context block via a second prompt.
pub fn sample_chat(model_path: &str) {
    let tokenizer = Utf8Tokenizer::new();

    let mut model = match Hierarchical::load(model_path, tokenizer) {
        Ok(m) => {
            println!("Loaded SFT model from '{model_path}' (step {}).", m.step);
            m
        }
        Err(e) => {
            eprintln!("Failed to load '{model_path}': {e}");
            std::process::exit(1);
        }
    };
    model.make_cache(1, MAX_SEQ_LEN);

    // The conversation so far. Each generated reply is appended as an
    // `Assistant` turn, so the next prompt carries the whole history — the same
    // layout the multi-turn training data has.
    let mut turns: Vec<Turn> = Vec::new();

    loop {
        println!("\nYou (empty line = quit, 'new' = fresh conversation):");
        let mut input = String::new();
        if stdin().read_line(&mut input).unwrap() == 0 || input.trim().is_empty() {
            println!();
            return;
        }
        let input = input.trim();
        if input == "new" {
            turns.clear();
            println!("(conversation cleared)");
            continue;
        }

        // Context is asked for once, at the start of a conversation: it takes
        // the <CONTEXT> slot of the first turn and conditions every reply after.
        if turns.is_empty() {
            print!("Context (optional, blank for none): ");
            stdout().flush().unwrap();
            let mut context = String::new();
            stdin().read_line(&mut context).unwrap();
            if !context.trim().is_empty() {
                turns.push(Turn {
                    role: Role::System,
                    content: context.trim().to_string(),
                });
            }
        }

        turns.push(Turn {
            role: Role::User,
            content: input.to_string(),
        });

        // One user turn can take several rounds: a reply that is a tool call is
        // not the answer, it is a request for a result. `app.launch` failing into
        // `app.list` and back is three rounds, so the cap is what stops a model
        // that has learned to call and never conclude.
        for _ in 0..MAX_TOOL_ROUNDS {
            let prompt = crate::sft::format_chat_prompt(&tokenizer, &turns);
            print!(">>> ");
            stdout().flush().unwrap();
            let mut printer = Utf8Printer::default();
            let reply = model.sample_chat(&prompt, MAX_LEN, TEMPERATURE, TOP_P, |token| {
                printer.print(token, &tokenizer)
            });
            println!();
            let called_tool = reply.contains(&TOOL_OPEN_TOKEN);
            // Markup is kept in the turn text so re-encoding the history
            // reproduces the exact tokens the model just wrote.
            turns.push(Turn {
                role: Role::Assistant,
                content: tokenizer.to_text_markup(&reply),
            });
            if !called_tool {
                break;
            }
            print!("result (blank = ok): ");
            stdout().flush().unwrap();
            let mut result = String::new();
            if stdin().read_line(&mut result).unwrap() == 0 {
                break;
            }
            let result = match result.trim() {
                "" => "ok",
                other => other,
            };
            turns.push(Turn {
                role: Role::Tool,
                content: result.to_string(),
            });
        }
    }
}

pub fn sample_hierarchical(model_path: &str) {
    let tokenizer = Utf8Tokenizer::new();

    let mut model = Hierarchical::load(model_path, tokenizer).unwrap();

    model.make_cache(1, MAX_SEQ_LEN);

    loop {
        println!("\nSample mode — type a prefix (empty = random start, Ctrl+D = quit):");
        let mut input = String::new();
        if stdin().read_line(&mut input).unwrap() == 0 {
            println!();
            return;
        }

        let prefix: Vec<u16> = if !input.trim().is_empty() {
            tokenizer.to_tokens(input.trim())
        } else {
            Vec::new()
        };

        print!(">>> ");
        stdout().flush().unwrap();

        let mut printer = Utf8Printer::default();
        model.sample(&prefix, MAX_LEN, TEMPERATURE, TOP_P, |token| {
            printer.print(token, &tokenizer)
        });

        println!();
    }
}
