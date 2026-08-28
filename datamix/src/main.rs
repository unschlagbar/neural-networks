// datamix — dataset mixing and filtering for the hierarchical model.
//
//   datamix build  <mix.toml>          stage, mix, write the corpus + report
//   datamix check  <mix.toml>          same, but write nothing (filter/share stats)
//   datamix sample <mix.toml> [-n 10]  print records drawn from the mixture
//   datamix synth  <file.syn> [-n 10] preview a template file's expansion
//   datamix verify <out.jsonl>       load it exactly as `hqg` does and report
//   datamix ping   [mix.toml]          check the local LM server and list its models

mod config;
mod filter;
mod json;
mod llm;
mod mix;
mod record;
mod report;
mod rng;
mod shard;
mod source;
mod toml;
mod synth;

use mix::{Options, human};
use rng::Rng;

fn main() {
    if let Err(e) = run() {
        eprintln!("datamix: {e}");
        std::process::exit(1);
    }
}

fn run() -> config::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("");
    let path = args.get(1).cloned().unwrap_or_default();
    let n = flag_n(&args).unwrap_or(10);

    match cmd {
        "build" | "check" | "sample" if !path.is_empty() => {
            let m = config::load(&path)?;
            let opts = Options {
                write: cmd == "build",
                preview: if cmd == "sample" { n } else { 0 },
            };
            let st = mix::build(&m, &opts)?;
            let text = report::render(&m, &st);
            println!("\n{text}");
            if cmd == "build" {
                println!(
                    "wrote {} records / {} tokens",
                    st.written,
                    human(st.written_tokens)
                );
                if !m.output.report.is_empty() {
                    std::fs::write(&m.output.report, &text)
                        .map_err(|e| format!("{}: {e}", m.output.report))?;
                    println!("report: {}", m.output.report);
                }
            }
            Ok(())
        }
        // The point of verifying here is that it runs the *training* loader,
        // not a second parser that might disagree with it.
        "verify" if !path.is_empty() => {
            let tok = neural_networks::tokenizer_utf8::Utf8Tokenizer::new();
            let max = neural_networks::config::SFT_MAX_TOKENS;
            let examples =
                neural_networks::sft::load_jsonl(&tok, &path, max).map_err(|e| e.to_string())?;
            if examples.is_empty() {
                return Err(format!("{path}: no usable examples"));
            }
            let tokens: usize = examples.iter().map(|e| e.tokens.len()).sum();
            let words: usize = examples.iter().map(|e| e.words.len()).sum();
            let resp: usize = examples.iter().map(|e| e.response_extent().1).sum();
            let longest = examples.iter().map(|e| e.tokens.len()).max().unwrap_or(0);
            // Every <SEP> is one exchange, so the count says how much of the
            // corpus is actually multi-turn.
            let sep = neural_networks::tokenizer_utf8::SEP_TOKEN;
            let exchanges: usize = examples
                .iter()
                .map(|e| e.tokens.iter().filter(|&&t| t == sep).count())
                .sum();
            println!(
                "{} examples · {} tokens · {} words · {} response words (loss-carrying)",
                examples.len(),
                human(tokens),
                human(words),
                human(resp)
            );
            println!(
                "mean {:.0} tokens/example, longest {longest} (cap {max}), \
                 {exchanges} exchanges ({:.2} turns/example)",
                tokens as f64 / examples.len() as f64,
                exchanges as f64 / examples.len() as f64
            );
            Ok(())
        }
        "ping" => {
            let cfg = if path.is_empty() {
                config::Llm::default()
            } else {
                config::load(&path)?.llm
            };
            println!("endpoint {}", cfg.endpoint);
            let mut client = llm::Client::new(&cfg)?;
            let models = client.models()?;
            if models.is_empty() {
                println!("server reachable, but it reports no loaded model");
            }
            for m in &models {
                println!("  model: {m}");
            }
            let reply = client.chat("Reply with exactly: ok", "ping", 0)?;
            println!("completion: {reply:?}");
            Ok(())
        }
        "synth" if !path.is_empty() => {
            let syn = synth::load(&path)?;
            let mut rng = Rng::new(1234);
            println!("{} combinations available", human(syn.combinations()));
            for rec in syn.expand(n, &mut rng, "synth") {
                match rec {
                    record::Record::Sft {
                        instruction,
                        context,
                        response,
                        category,
                    } => {
                        println!("\n[{category}] {instruction}");
                        if !context.is_empty() {
                            println!("  context: {context}");
                        }
                        for line in response.lines() {
                            println!("  > {line}");
                        }
                    }
                    record::Record::Chat { turns, category } => {
                        println!("\n[{category}] conversation, {} turns", turns.len());
                        for t in turns {
                            let tag = match t.role {
                                neural_networks::sft::Role::System => "sys",
                                neural_networks::sft::Role::User => "you",
                                neural_networks::sft::Role::Assistant => ">>>",
                                neural_networks::sft::Role::Tool => "<--",
                                neural_networks::sft::Role::AssistantContext => "(x)",
                            };
                            for line in t.content.lines() {
                                println!("  {tag} {line}");
                            }
                        }
                    }
                    record::Record::Doc { .. } => {}
                }
            }
            Ok(())
        }
        _ => {
            eprintln!(
                "usage:\n  \
                 datamix build  <mix.toml>            stage, mix, write corpus + report\n  \
                 datamix check  <mix.toml>            compute the mixture, write nothing\n  \
                 datamix sample <mix.toml> [-n 10]    print records drawn from the mixture\n  \
                 datamix synth  <file.syn> [-n 10]  preview a template file\n  \
                 datamix verify <out.jsonl>         load an SFT corpus the way training does\n  \
                 datamix ping   [mix.toml]            check the local LM server, list its models"
            );
            std::process::exit(2);
        }
    }
}

fn flag_n(args: &[String]) -> Option<usize> {
    let i = args.iter().position(|a| a == "-n" || a == "--count")?;
    args.get(i + 1)?.parse().ok()
}
