// The build itself: stage every source, size the shares, draw the mixture,
// write the corpus.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};

use crate::config::{Filters, Mix, OutKind, Result, Source, WeightUnit};
use crate::filter::{Filter, Reject};
use crate::llm::Client;
use std::cell::RefCell;
use crate::record::Record;
use crate::rng::Rng;
use crate::shard::Shard;
use crate::source;

pub struct SourceStats {
    pub name: String,
    pub read: usize,
    pub kept: usize,
    pub kept_tokens: usize,
    pub rejects: HashMap<Reject, usize>,
    pub emitted: usize,
    pub emitted_tokens: usize,
    pub share: f32,
    pub epochs_used: f32,
    /// What this source's weight asked for, before its `epochs` cap, in the
    /// mixture's weight unit. When it exceeds what was emitted, this source ran
    /// out of data and the corpus is short by the difference.
    pub wanted: usize,
    pub unit: WeightUnit,
}

impl SourceStats {
    /// What this source held / contributed, in the weight unit.
    pub fn kept_units(&self) -> usize {
        match self.unit {
            WeightUnit::Tokens => self.kept_tokens,
            WeightUnit::Records => self.kept,
        }
    }

    pub fn emitted_units(&self) -> usize {
        match self.unit {
            WeightUnit::Tokens => self.emitted_tokens,
            WeightUnit::Records => self.emitted,
        }
    }

    /// Did this source fail to fill the share its weight asked for? The 1%
    /// slack keeps rounding on the last record from reading as a shortfall.
    pub fn is_short(&self) -> bool {
        let got = self.emitted_units();
        self.wanted > got + got / 100
    }
}

pub struct BuildStats {
    pub sources: Vec<SourceStats>,
    pub budget: usize,
    pub unit: WeightUnit,
    pub written: usize,
    pub written_tokens: usize,
    pub held_out: usize,
    pub dropped_shape: usize,
    pub out_paths: Vec<String>,
}

/// Name of the weight unit, for the messages that report a budget.
pub fn unit_name(unit: WeightUnit) -> &'static str {
    match unit {
        WeightUnit::Tokens => "tokens",
        WeightUnit::Records => "records",
    }
}

/// Where the staging shards live. Deleted when the build finishes.
const STAGE_DIR: &str = "target/datamix-stage";

/// What a run does beyond computing the mixture: `write` puts the corpus on
/// disk, `preview` prints that many drawn records instead.
pub struct Options {
    pub write: bool,
    pub preview: usize,
}

pub fn build(mix: &Mix, opts: &Options) -> Result<BuildStats> {
    let unit = mix.output.weight_by;
    let mut rng = Rng::new(mix.output.seed);
    let mut readers = Vec::new();
    let mut stats = Vec::new();
    // Constructed even when nothing uses it: it opens no socket until the first
    // completion, so a mixture with no `llm` source and no `judge` never talks
    // to the server.
    // A `RefCell` because both halves of staging reach for it: the generator
    // (making records) and the judge (vetting them) run inside one another.
    let llm = RefCell::new(Client::new(&mix.llm)?);

    for (si, src) in mix.sources.iter().enumerate() {
        println!("[{}/{}] staging '{}'", si + 1, mix.sources.len(), src.name);
        let (reader, st) = stage(src, mix.output.seed ^ (si as u64 + 1), &llm)?;
        println!(
            "      {} read, {} kept ({} tokens)",
            st.read,
            st.kept,
            human(st.kept_tokens)
        );
        readers.push(reader);
        stats.push(SourceStats { unit, ..st });
    }

    // Shares are relative weights; a source with nothing left after filtering
    // drops out and its weight is redistributed over the rest.
    let total_w: f32 = mix
        .sources
        .iter()
        .zip(&stats)
        .filter(|(_, s)| s.kept > 0)
        .map(|(s, _)| s.weight.max(0.0))
        .sum();
    if total_w <= 0.0 {
        return Err("every source is empty after filtering".into());
    }
    for (src, st) in mix.sources.iter().zip(&mut stats) {
        st.share = if st.kept > 0 {
            src.weight.max(0.0) / total_w
        } else {
            0.0
        };
    }

    // With no explicit budget, take the largest corpus in which every source
    // stays inside its own `epochs` — i.e. the binding source is the one whose
    // available data runs out first at its share.
    let asked = match unit {
        WeightUnit::Tokens => mix.output.tokens,
        WeightUnit::Records => mix.output.records,
    };
    let budget = if asked > 0 {
        asked
    } else {
        mix.sources
            .iter()
            .zip(&stats)
            .filter(|(_, s)| s.share > 0.0)
            .map(|(src, s)| {
                (s.kept_units() as f64 * src.epochs.max(0.0) as f64 / s.share as f64) as usize
            })
            .min()
            .unwrap_or(0)
    };

    // The binding source — the one that runs out first at its share — sets the
    // size of the whole corpus, so name it: a 500 MB mixture collapsing to a
    // few thousand tokens is otherwise a silent surprise.
    if asked == 0
        && let Some((src, st)) = mix
            .sources
            .iter()
            .zip(&stats)
            .filter(|(_, s)| s.share > 0.0)
            .min_by_key(|(src, s)| {
                (s.kept_units() as f64 * src.epochs.max(0.0) as f64 / s.share as f64) as usize
            })
    {
        println!(
            "budget {} {unit_name}, set by '{}' ({} {unit_name} x {} epochs at {:.1}% share). \
             Raise its `epochs`, lower its `weight`, or set `{unit_name}` in [output] to override.",
            human(budget),
            src.name,
            human(st.kept_units()),
            src.epochs,
            100.0 * st.share,
            unit_name = unit_name(unit),
        );
    }

    // Draw each source's records until it has covered its share of the budget.
    let mut plan: Vec<(usize, usize)> = Vec::new();
    for (si, (src, st)) in mix.sources.iter().zip(&mut stats).enumerate() {
        if st.share <= 0.0 {
            continue;
        }
        let target = ((budget as f64) * st.share as f64) as usize;
        let cap = (st.kept_units() as f64 * src.epochs.max(0.0) as f64) as usize;
        st.wanted = target;
        let target = target.min(cap);
        let mut order: Vec<usize> = (0..readers[si].len()).collect();
        rng.shuffle(&mut order);
        let mut drawn = 0usize;
        let mut i = 0usize;
        while drawn < target && !order.is_empty() {
            if i == order.len() {
                // Next epoch over this source: reshuffle so the repeat is not
                // the same sequence again.
                rng.shuffle(&mut order);
                i = 0;
            }
            let idx = order[i];
            i += 1;
            let rec = readers[si].get(idx)?;
            drawn += match unit {
                WeightUnit::Tokens => rec.tokens(),
                WeightUnit::Records => 1,
            };
            plan.push((si, idx));
            st.emitted += 1;
            st.emitted_tokens += rec.tokens();
        }
        st.epochs_used = if st.kept_units() > 0 {
            st.emitted_units() as f32 / st.kept_units() as f32
        } else {
            0.0
        };
    }

    // Say which sources ran out. With an explicit `tokens` budget nothing else
    // reports it: the corpus is simply smaller than asked for, and the reason is
    // one source's `epochs` cap.
    let short: Vec<&SourceStats> = stats.iter().filter(|s| s.is_short()).collect();
    if !short.is_empty() {
        let missing: usize = short.iter().map(|s| s.wanted - s.emitted_units()).sum();
        println!(
            "capped: {} {} short of the budget, {} source(s) ran out of data:",
            human(missing),
            unit_name(unit),
            short.len()
        );
        for s in &short {
            let src = mix.sources.iter().find(|x| x.name == s.name).unwrap();
            println!(
                "  '{}' asked for {} at {:.1}% share but has {} x {} epochs = {}",
                s.name,
                human(s.wanted),
                100.0 * s.share,
                human(s.kept_units()),
                src.epochs,
                human(s.emitted_units()),
            );
        }
        println!("  raise their `epochs`, lower their `weight`, or add data.");
    }

    if mix.output.shuffle {
        rng.shuffle(&mut plan);
    }

    let mut out = BuildStats {
        sources: stats,
        budget,
        unit,
        written: 0,
        written_tokens: 0,
        held_out: 0,
        dropped_shape: 0,
        out_paths: Vec::new(),
    };

    if opts.preview > 0 {
        for (n, (si, idx)) in plan.iter().take(opts.preview).enumerate() {
            let rec = readers[*si].get(*idx)?;
            let from = &mix.sources[*si].name;
            println!("\n--- {} of {} from '{from}' ---", n + 1, plan.len());
            print!("{}", render(&rec, mix.output.kind).unwrap_or_default());
        }
        println!();
    }

    if !opts.write || mix.output.kind == OutKind::None {
        for (si, idx) in &plan {
            let rec = readers[*si].get(*idx)?;
            out.written += 1;
            out.written_tokens += rec.tokens();
        }
    } else {
        write_corpus(mix, &plan, &mut readers, &mut out)?;
    }

    for r in readers {
        r.remove();
    }
    let _ = std::fs::remove_dir(STAGE_DIR);
    Ok(out)
}

fn stage(
    src: &Source,
    seed: u64,
    llm: &RefCell<Client>,
) -> Result<(crate::shard::ShardReader, SourceStats)> {
    let mut shard = Shard::create(STAGE_DIR, &src.name)?;
    let mut filter = Filter::new(src.filters.clone());
    let mut st = SourceStats {
        name: src.name.clone(),
        read: 0,
        kept: 0,
        kept_tokens: 0,
        rejects: HashMap::new(),
        emitted: 0,
        emitted_tokens: 0,
        share: 0.0,
        epochs_used: 0.0,
        wanted: 0,
        unit: WeightUnit::Tokens,
    };
    let mut err: Option<String> = None;
    let judge = src.filters.judge.clone();
    let trim = src.filters.trim_turns;
    let max_tokens = src.filters.max_tokens;
    let max_words = src.filters.max_words;
    {
        let st = &mut st;
        let shard = &mut shard;
        let err = &mut err;
        let skipped = source::read(src, seed, llm, &mut |mut rec| {
            st.read += 1;
            // Trimming runs before the gates: it decides whether the record is
            // over the token cap at all.
            if trim && !rec.trim_to_fit(max_tokens, max_words) {
                *st.rejects.entry(Reject::TooLong).or_insert(0) += 1;
                return true;
            }
            if let Some(reason) = filter.check(&rec) {
                *st.rejects.entry(reason).or_insert(0) += 1;
                return true;
            }
            // The judge runs last, on what survived the cheap gates: it costs a
            // round trip per record.
            if !judge.is_empty() {
                match ask_judge(llm, &src.filters, &rec) {
                    Ok(true) => {}
                    Ok(false) => {
                        *st.rejects.entry(Reject::Judged).or_insert(0) += 1;
                        return true;
                    }
                    Err(e) => {
                        *err = Some(e);
                        return false;
                    }
                }
            }
            st.kept += 1;
            st.kept_tokens += rec.tokens();
            if let Err(e) = shard.push(&rec) {
                *err = Some(e);
                return false;
            }
            true
        })?;
        if skipped > 0 {
            st.read += skipped;
            *st.rejects.entry(Reject::Language).or_insert(0) += skipped;
        }
    }
    if let Some(e) = err {
        return Err(e);
    }
    st.kept = shard.len();
    Ok((shard.into_reader()?, st))
}

/// Ask the local model whether one record belongs in the corpus. The reply is
/// read on its first word, so an answer of "yes, because ..." still counts —
/// anything that does not begin with `judge_expect` drops the record.
fn ask_judge(llm: &RefCell<Client>, cfg: &Filters, rec: &Record) -> Result<bool> {
    let body = match rec {
        Record::Doc { text } => text.clone(),
        Record::Sft {
            instruction,
            context,
            response,
            ..
        } => format!(
            "INSTRUCTION:\n{instruction}\n\nCONTEXT:\n{context}\n\nRESPONSE:\n{response}"
        ),
        Record::Chat { turns, .. } => turns
            .iter()
            .map(|t| format!("{}:\n{}", role_name(t.role).to_uppercase(), t.content))
            .collect::<Vec<_>>()
            .join("\n\n"),
    };
    // Long documents are judged on their head: a local model's context is the
    // binding constraint, and the first few KB decide quality in practice.
    let head: String = body.chars().take(6000).collect();
    let system = format!(
        "{}\n\nAnswer with a single word: yes or no. No explanation.",
        cfg.judge
    );
    let reply = llm.borrow_mut().chat_as(
        &system,
        &head,
        0,
        &cfg.judge_model,
        cfg.judge_temperature,
    )?;
    Ok(reply
        .trim()
        .to_lowercase()
        .trim_start_matches(|c: char| !c.is_alphanumeric())
        .starts_with(&cfg.judge_expect.to_lowercase()))
}

fn write_corpus(
    mix: &Mix,
    plan: &[(usize, usize)],
    readers: &mut [crate::shard::ShardReader],
    out: &mut BuildStats,
) -> Result<()> {
    if let Some(parent) = std::path::Path::new(&mix.output.path).parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let holdout = (plan.len() as f32 * mix.output.holdout.clamp(0.0, 0.9)) as usize;

    let eval_path = format!("{}.eval", mix.output.path);
    let mut eval = if holdout > 0 {
        out.out_paths.push(eval_path.clone());
        Some(BufWriter::new(
            File::create(&eval_path).map_err(|e| format!("{eval_path}: {e}"))?,
        ))
    } else {
        None
    };

    let mut writer = Writer::new(mix)?;
    for (n, (si, idx)) in plan.iter().enumerate() {
        let rec = readers[*si].get(*idx)?;
        let Some(text) = render(&rec, mix.output.kind) else {
            out.dropped_shape += 1;
            continue;
        };
        if n < holdout {
            let w = eval.as_mut().unwrap();
            w.write_all(text.as_bytes()).map_err(|e| e.to_string())?;
            out.held_out += 1;
            continue;
        }
        writer.write(&text)?;
        out.written += 1;
        out.written_tokens += rec.tokens();
    }
    writer.finish(out)?;
    if let Some(mut w) = eval {
        w.flush().map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// One record as it appears in the output file, including its trailing
/// separator. Also what a generator shows the model as an example of what to
/// write, which is why it is shared rather than reimplemented there: a seed
/// that is not in the exact output format teaches the wrong format. `None` means the record's shape does not fit the output kind (a
/// plain document has no instruction/response to write into an SFT file).
pub fn render(rec: &Record, kind: OutKind) -> Option<String> {
    match (kind, rec) {
        (OutKind::Sft, Record::Doc { .. }) => None,
        (
            OutKind::Sft,
            Record::Sft {
                instruction,
                context,
                response,
                category,
            },
        ) => Some(format!(
            "{{\"instruction\": \"{}\", \"context\": \"{}\", \"response\": \"{}\", \
             \"category\": \"{}\"}}\n",
            escape(instruction),
            escape(context),
            escape(response),
            escape(category),
        )),
        (
            OutKind::Sft,
            Record::Chat {
                turns,
                category,
            },
        ) => {
            let messages = turns
                .iter()
                .map(|t| {
                    format!(
                        "{{\"role\": \"{}\", \"content\": \"{}\"}}",
                        role_name(t.role),
                        crate::json::escape(&t.content)
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            Some(format!(
                "{{\"messages\": [{messages}], \"category\": \"{}\"}}\n",
                crate::json::escape(category)
            ))
        }
        (OutKind::Text, Record::Doc { text }) => Some(format!("{text}<|endoftext|>")),
        // An instruction pair still makes a usable pretraining document; the
        // chat markers are dropped because they are token ids, not text.
        (OutKind::Text, rec) => Some(format!("{}<|endoftext|>", rec.train_text())),
        (OutKind::None, _) => None,
    }
}

/// Output file(s). A text corpus may be sharded so a single file stays under a
/// size the rest of the toolchain is comfortable with.
struct Writer {
    base: String,
    shard_bytes: usize,
    out: BufWriter<File>,
    written: usize,
    index: usize,
    paths: Vec<String>,
}

impl Writer {
    fn new(mix: &Mix) -> Result<Self> {
        let base = mix.output.path.clone();
        let shard_bytes = mix.output.shard_bytes;
        let path = if shard_bytes > 0 {
            shard_name(&base, 0)
        } else {
            base.clone()
        };
        let out = BufWriter::new(File::create(&path).map_err(|e| format!("{path}: {e}"))?);
        Ok(Self {
            base,
            shard_bytes,
            out,
            written: 0,
            index: 0,
            paths: vec![path],
        })
    }

    fn write(&mut self, text: &str) -> Result<()> {
        if self.shard_bytes > 0 && self.written + text.len() > self.shard_bytes && self.written > 0
        {
            self.out.flush().map_err(|e| e.to_string())?;
            self.index += 1;
            let path = shard_name(&self.base, self.index);
            self.out =
                BufWriter::new(File::create(&path).map_err(|e| format!("{path}: {e}"))?);
            self.paths.push(path);
            self.written = 0;
        }
        self.out.write_all(text.as_bytes()).map_err(|e| e.to_string())?;
        self.written += text.len();
        Ok(())
    }

    fn finish(mut self, out: &mut BuildStats) -> Result<()> {
        self.out.flush().map_err(|e| e.to_string())?;
        out.out_paths.extend(self.paths);
        Ok(())
    }
}

fn role_name(r: neural_networks::sft::Role) -> &'static str {
    match r {
        neural_networks::sft::Role::System => "system",
        neural_networks::sft::Role::User => "user",
        neural_networks::sft::Role::Assistant => "assistant",
        neural_networks::sft::Role::Tool => "tool",
        neural_networks::sft::Role::AssistantContext => "assistant_context",
    }
}

fn shard_name(base: &str, i: usize) -> String {
    match base.rsplit_once('.') {
        Some((stem, ext)) => format!("{stem}.{i:03}.{ext}"),
        None => format!("{base}.{i:03}"),
    }
}

fn escape(s: &str) -> String {
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

pub fn human(n: usize) -> String {
    match n {
        n if n >= 1_000_000_000 => format!("{:.2}G", n as f64 / 1e9),
        n if n >= 1_000_000 => format!("{:.2}M", n as f64 / 1e6),
        n if n >= 1_000 => format!("{:.1}k", n as f64 / 1e3),
        n => n.to_string(),
    }
}
