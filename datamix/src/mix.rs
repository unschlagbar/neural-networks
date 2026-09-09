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

/// Where this build's staging shards live. Deleted when the build finishes.
///
/// One directory per process, because two builds can legitimately be running at
/// once: a long rewrite filling the cache, and a `cache_only` build assembling
/// a corpus out of what it has finished so far. A shared path would have them
/// writing each other's shards, and the shards are named by source — which the
/// two mixtures share.
fn stage_dir() -> String {
    format!("target/datamix-stage-{}", std::process::id())
}

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
    let _ = std::fs::remove_dir(stage_dir());
    Ok(out)
}

/// Rewrite a batch of records through the model, `[llm] workers` in flight.
/// Order is preserved and a record whose rewrite could not be read back comes
/// home as `None` — the caller drops it rather than keeping an original the
/// mixture has asked not to contain.
///
/// Each worker gets its own `Client` because a request needs `&mut self` for
/// its counters; they share only the cache directory, and cache files are
/// named by content hash, so two workers writing the same answer write the
/// same bytes to the same path.
/// Does this record fall under the source's rewrite?
fn wants_transform(f: &Filters, rec: &Record) -> bool {
    if f.transform.is_empty() {
        return false;
    }
    if f.transform_when.is_empty() {
        return true;
    }
    let text = rec.train_text();
    f.transform_when.iter().any(|n| text.contains(n.as_str()))
}

/// Everything after the rewrite: the cheap gates, then the judge, then the
/// shard. `false` stops the read.
#[allow(clippy::too_many_arguments)]
fn accept(
    rec: Record,
    src: &Source,
    llm: &RefCell<Client>,
    st: &mut SourceStats,
    filter: &mut Filter,
    shard: &mut Shard,
    judging: &mut Vec<Record>,
    err: &mut Option<String>,
) -> bool {
    if let Some(reason) = filter.check(&rec) {
        *st.rejects.entry(reason).or_insert(0) += 1;
        return true;
    }
    // The judge runs last, on what survived the cheap gates. It is one round
    // trip per record, so records queue here and go up a batch at a time.
    if !src.filters.judge.is_empty() {
        judging.push(rec);
        if judging.len() >= judge_batch_size(llm) {
            return flush_judged(judging, src, llm, st, shard, err);
        }
        return true;
    }
    keep(rec, st, shard, err)
}

/// Shard one record that has passed everything.
fn keep(rec: Record, st: &mut SourceStats, shard: &mut Shard, err: &mut Option<String>) -> bool {
    st.kept += 1;
    st.kept_tokens += rec.tokens();
    if let Err(e) = shard.push(&rec) {
        *err = Some(e);
        return false;
    }
    true
}

fn judge_batch_size(llm: &RefCell<Client>) -> usize {
    // Four deep per worker: enough that no slot waits on the slowest record,
    // small enough that a source's progress line still moves.
    llm.borrow().cfg.workers.max(1) * 4
}

/// Judge the queue and shard what survives, in order.
fn flush_judged(
    judging: &mut Vec<Record>,
    src: &Source,
    llm: &RefCell<Client>,
    st: &mut SourceStats,
    shard: &mut Shard,
    err: &mut Option<String>,
) -> bool {
    if judging.is_empty() {
        return true;
    }
    // Under `cache_only` nothing opens a socket, so a judge error is never a
    // transport failure — it is a record whose verdict has not been paid for.
    // That is a drop, the same as any other unfinished work, not a dead build.
    let unpaid = llm.borrow().cfg.cache_only;
    let verdicts = judge_batch(&mut llm.borrow_mut(), &src.filters, judging);
    let recs = std::mem::take(judging);
    for (rec, verdict) in recs.into_iter().zip(verdicts) {
        match verdict {
            Ok(true) => {
                if !keep(rec, st, shard, err) {
                    return false;
                }
            }
            Ok(false) => *st.rejects.entry(Reject::Judged).or_insert(0) += 1,
            Err(_) if unpaid => *st.rejects.entry(Reject::Unjudged).or_insert(0) += 1,
            Err(e) => {
                *err = Some(e);
                return false;
            }
        }
    }
    true
}

/// Rewrite the buffered records and pass the survivors on. A record whose
/// rewrite could not be read back is dropped and counted: the mixture asked
/// for a corpus without the original, so keeping it would be the one outcome
/// nobody wanted.
#[allow(clippy::too_many_arguments)]
fn flush(
    pending: &mut Vec<Record>,
    src: &Source,
    llm: &RefCell<Client>,
    st: &mut SourceStats,
    filter: &mut Filter,
    shard: &mut Shard,
    judging: &mut Vec<Record>,
    err: &mut Option<String>,
) -> bool {
    if pending.is_empty() {
        return true;
    }
    // Scoped: `accept` below judges through the same client.
    let done = transform_batch(&mut llm.borrow_mut(), &src.filters, pending);
    pending.clear();
    for slot in done {
        match slot {
            Ok(rec) => {
                if !accept(rec, src, llm, st, filter, shard, judging, err) {
                    return false;
                }
            }
            Err(reason) => *st.rejects.entry(reason).or_insert(0) += 1,
        }
    }
    true
}

/// Compile the record's code, and if it does not build, give the model one
/// chance to fix it with `rustc`'s own errors in hand. `None` when it still
/// does not build — a record that teaches code which cannot compile is worse
/// than one record fewer.
fn verified(
    client: &mut Client,
    f: &Filters,
    original: &Record,
    rewritten: Record,
    scratch: &std::path::Path,
) -> std::result::Result<Record, Reject> {
    if f.verify != "rust" {
        return Ok(rewritten);
    }
    let Record::Chat { turns, .. } = &rewritten else {
        // A dolly record carries its code the same way; verify it through the
        // turns it stands for.
        let json = rewritten.to_messages_json().ok_or(Reject::Rewritten)?;
        let turns = neural_networks::sft::parse_messages(&json).ok_or(Reject::Rewritten)?;
        let src = crate::verify::rust_blocks(&turns);
        return match crate::verify::rust_compiles(&src, scratch) {
            Ok(()) => Ok(rewritten),
            Err(_) => Err(Reject::Uncompilable),
        };
    };
    let src = crate::verify::rust_blocks(turns);
    let Err(errors) = crate::verify::rust_compiles(&src, scratch) else {
        return Ok(rewritten);
    };
    if f.verify_repair.is_empty() {
        return Err(Reject::Uncompilable);
    }
    // The nonce keeps the repair from colliding with the rewrite's own cache
    // entry for the same record.
    let payload = format!(
        "{}\n\nrustc says:\n{errors}",
        rewritten.to_messages_json().ok_or(Reject::Rewritten)?
    );
    let reply = client
        .chat_as(&f.verify_repair, &payload, 1, &f.transform_model, f.transform_temperature)
        .map_err(|_| Reject::Uncompilable)?;
    let fixed = original
        .from_messages_json(&reply)
        .ok_or(Reject::Uncompilable)?;
    let Record::Chat { turns, .. } = &fixed else {
        return Err(Reject::Uncompilable);
    };
    let src = crate::verify::rust_blocks(turns);
    crate::verify::rust_compiles(&src, scratch)
        .map(|()| fixed)
        .map_err(|_| Reject::Uncompilable)
}

fn transform_batch(
    client: &mut Client,
    f: &Filters,
    recs: &[Record],
) -> Vec<std::result::Result<Record, Reject>> {
    let workers = client.cfg.workers.max(1);
    // A queue, not a static split. Records cost wildly different amounts — one
    // whose code compiles first time is a single call, one that needs a repair
    // is two calls and four rustc runs — so handing each worker a fixed slice
    // leaves most of them waiting on the slowest, and the server idle with
    // them. Measured mid-run that way: GPU utilization dipping to 22%.
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: std::sync::Mutex<Vec<std::result::Result<Record, Reject>>> =
        std::sync::Mutex::new(vec![Err(Reject::Rewritten); recs.len()]);
    let counts: Vec<(usize, usize)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|w| {
                let mut c = client.clone();
                let next = &next;
                let out = &out;
                // Each worker compiles in its own directory: rustc writes fixed
                // names there and two workers sharing one would race.
                let scratch = std::env::temp_dir()
                    .join(format!("datamix-verify-{}-{w}", std::process::id()));
                scope.spawn(move || {
                    let _ = std::fs::create_dir_all(&scratch);
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(rec) = recs.get(i) else { break };
                        let Some(payload) = rec.to_messages_json() else {
                            continue;
                        };
                        // nonce 0: the same record must always land on the same
                        // cache entry. That identity is the whole persistence
                        // story — a rebuild re-reads the corpus, hashes the same
                        // text and never reaches the server.
                        let Ok(reply) = c.chat_as(
                            &f.transform,
                            &payload,
                            0,
                            &f.transform_model,
                            f.transform_temperature,
                        ) else {
                            continue;
                        };
                        let Some(rewritten) = rec.from_messages_json(&reply) else {
                            continue;
                        };
                        let slot = verified(&mut c, f, rec, rewritten, &scratch);
                        out.lock().expect("results")[i] = slot;
                    }
                    (c.calls, c.cached)
                })
            })
            .collect();
        handles.into_iter().filter_map(|h| h.join().ok()).collect()
    });
    let out = out.into_inner().expect("results");
    for (calls, cached) in counts {
        client.calls += calls;
        client.cached += cached;
    }
    out
}

fn stage(
    src: &Source,
    seed: u64,
    llm: &RefCell<Client>,
) -> Result<(crate::shard::ShardReader, SourceStats)> {
    let mut shard = Shard::create(&stage_dir(), &src.name)?;
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
    let trim = src.filters.trim_turns;
    let max_tokens = src.filters.max_tokens;
    let max_words = src.filters.max_words;
    let system_strip = src.filters.system_strip.clone();
    let fold_system = src.filters.system_fold;
    {
        let st = &mut st;
        let shard = &mut shard;
        let err = &mut err;
        // A rewrite is worth batching: the server answers `workers` requests at
        // once, and a record at a time would leave all but one slot idle.
        // Eight deep per worker: the queue only helps if there is work in it
        // when a worker finishes early.
        let batch = if src.filters.transform.is_empty() {
            0
        } else {
            llm.borrow().cfg.workers.max(1) * 8
        };
        let mut pending: Vec<Record> = Vec::new();
        let mut judging: Vec<Record> = Vec::new();
        let skipped = source::read(src, seed, llm, &mut |mut rec| {
            st.read += 1;
            // Before anything else: the record's own text decides its rewrite
            // cache key, so the system turn has to be settled first or the
            // same record hashes two ways.
            if fold_system || !system_strip.is_empty() {
                rec.fold_system(&system_strip, fold_system);
            }
            // Trimming runs before the gates: it decides whether the record is
            // over the token cap at all — and before the rewrite, so no turn
            // is translated only to be dropped.
            if trim && !rec.trim_to_fit(max_tokens, max_words) {
                *st.rejects.entry(Reject::TooLong).or_insert(0) += 1;
                return true;
            }
            // The gates judge the record the corpus will actually hold, so the
            // rewrite comes first — but only for the records it applies to.
            if batch == 0 || !wants_transform(&src.filters, &rec) {
                // `transform_only` makes the rewrite the corpus: a record the
                // gate never sent is not a record this mixture is collecting.
                if src.filters.transform_only && !src.filters.transform.is_empty() {
                    *st.rejects.entry(Reject::NotTransformed).or_insert(0) += 1;
                    return true;
                }
                return accept(rec, src, llm, st, &mut filter, shard, &mut judging, err);
            }
            pending.push(rec);
            if pending.len() < batch {
                return true;
            }
            flush(&mut pending, src, llm, st, &mut filter, shard, &mut judging, err)
        })?;
        flush(&mut pending, src, llm, st, &mut filter, shard, &mut judging, err);
        // Whatever is still queued for the judge, judged before the shard closes.
        flush_judged(&mut judging, src, llm, st, shard, err);
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
fn judge_payload(cfg: &Filters, rec: &Record) -> (String, String) {
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
        "{}\n\nAnswer with the verdict FIRST — yes or no — then, after a dash, \
         the one clause that decided it.",
        cfg.judge
    );
    (system, head)
}

/// Did the reply keep the record? Read on its first word, so "yes, because ..."
/// counts and anything not beginning with `judge_expect` drops it.
fn judged_keep(cfg: &Filters, reply: &str) -> bool {
    reply
        .trim()
        .to_lowercase()
        .trim_start_matches(|c: char| !c.is_alphanumeric())
        .starts_with(&cfg.judge_expect.to_lowercase())
}

/// Judge a batch, `[llm] workers` in flight. Judging is one short round trip per
/// record and it was the only part of a build still made one at a time:
/// measured at 36% of the wall clock on `mixes/tools.toml`, with seven of the
/// server's eight slots idle throughout.
fn judge_batch(client: &mut Client, cfg: &Filters, recs: &[Record]) -> Vec<Result<bool>> {
    let workers = client.cfg.workers.max(1);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: std::sync::Mutex<Vec<Result<bool>>> =
        std::sync::Mutex::new((0..recs.len()).map(|_| Ok(true)).collect());
    let counts: Vec<(usize, usize)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers.min(recs.len().max(1)))
            .map(|_| {
                let mut c = client.clone();
                c.calls = 0;
                c.cached = 0;
                // A verdict plus one clause. Measured over 5,407 cached judge
                // replies: p90 is 25 tokens, but p99 is 434 and one reached the
                // inherited 4096-token cap — a judge that starts rambling holds
                // a server slot for minutes and decides nothing after its first
                // word. The cap costs nothing and removes that whole tail.
                c.cfg.max_tokens = c.cfg.max_tokens.min(96);
                let (next, out) = (&next, &out);
                scope.spawn(move || {
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(rec) = recs.get(i) else { break };
                        let (system, head) = judge_payload(cfg, rec);
                        let slot = c
                            .chat_as(&system, &head, 0, &cfg.judge_model, cfg.judge_temperature)
                            .map(|reply| judged_keep(cfg, &reply));
                        out.lock().expect("judged")[i] = slot;
                    }
                    (c.calls, c.cached)
                })
            })
            .collect();
        handles.into_iter().filter_map(|h| h.join().ok()).collect()
    });
    for (calls, cached) in counts {
        client.calls += calls;
        client.cached += cached;
    }
    out.into_inner().expect("judged")
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
                    // The mask is written only when there is one, so an
                    // unmasked corpus reads exactly as it did before.
                    let mask = if t.no_loss.is_empty() {
                        String::new()
                    } else {
                        let pairs = t
                            .no_loss
                            .iter()
                            .map(|(s, e)| format!("[{s}, {e}]"))
                            .collect::<Vec<_>>()
                            .join(", ");
                        format!(", \"no_loss\": [{pairs}]")
                    };
                    format!(
                        "{{\"role\": \"{}\", \"content\": \"{}\"{mask}}}",
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
