// Input readers. Every kind streams: nothing here holds a whole corpus in
// memory, so a 500 MB directory and a 4 GB parquet cost the same.

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use neural_networks::parquet::ParquetColumnReader;
use neural_networks::sft::{Role, Turn};

use crate::config::{OutKind, Result, Source, SourceKind};
use crate::json;
use crate::llm::Client;
use std::cell::RefCell;
use crate::record::Record;
use crate::rng::Rng;
use crate::synth;

/// Documents are handed to `sink` one at a time; returning `false` stops the
/// read (a source that already filled its quota). Returns how many rows the
/// language filter dropped — it runs inside the parquet reader, before the
/// text is even decoded, so it cannot be counted with the other filters.
pub fn read(
    src: &Source,
    seed: u64,
    llm: &RefCell<Client>,
    sink: &mut impl FnMut(Record) -> bool,
) -> Result<usize> {
    match src.kind {
        SourceKind::File => read_file(&src.path, &src.separator, sink).map(|_| 0),
        SourceKind::Dir => read_dir(src, sink).map(|_| 0),
        SourceKind::Parquet => read_parquet(src, sink),
        SourceKind::Chat => read_chat_parquet(src, sink).map(|_| 0),
        SourceKind::Dolly => read_jsonl(src, true, sink).map(|_| 0),
        SourceKind::Jsonl => read_jsonl(src, false, sink).map(|_| 0),
        SourceKind::Synth => {
            let syn = synth::load(&src.path)?;
            let mut rng = Rng::new(seed);
            for rec in syn.expand(src.count, &mut rng, &src.category) {
                if !sink(rec) {
                    break;
                }
            }
            Ok(0)
        }
        SourceKind::Llm => generate(src, seed, llm, sink).map(|_| 0),
    }
}

/// Ask the local model for examples until `count` of them are in hand.
///
/// The reply is read as JSONL — one `{instruction, context, response}` object
/// per line — because that is the shape a small local model holds onto over a
/// long answer, and a line that fails to parse costs one example instead of the
/// whole call. Optional `seed_file` renders a template instruction into
/// `{seed}` so successive calls are not all the same request.
fn generate(
    src: &Source,
    seed: u64,
    llm: &RefCell<Client>,
    sink: &mut impl FnMut(Record) -> bool,
) -> Result<()> {
    let want = if src.count == 0 { 50 } else { src.count };
    let seeds = if src.seed_file.is_empty() {
        None
    } else {
        Some(synth::load(&src.seed_file)?)
    };
    let mut rng = Rng::new(seed);
    let mut made = 0usize;
    let mut call = 0u64;
    // Stop chasing a model that has stopped producing parsable lines rather
    // than looping on it.
    let mut barren = 0usize;

    while made < want && barren < 5 {
        // Seeds are rendered in the exact output format the model is being
        // asked to produce — the instruction alone would show it what to write
        // about but not how, and it will happily invent its own tool syntax.
        let seed_text = match &seeds {
            Some(syn) => syn
                .expand(src.seeds, &mut rng, &src.category)
                .iter()
                .filter_map(|r| crate::mix::render(r, OutKind::Sft))
                .collect::<Vec<_>>()
                .concat(),
            None => String::new(),
        };
        let batch = src.batch.min(want - made);
        let user = src
            .prompt
            .replace("{seed}", &seed_text)
            .replace("{n}", &batch.to_string());
        // The borrow is scoped to the call: `sink` below may judge with the
        // same client.
        let reply = llm.borrow_mut().chat_as(
            &src.system,
            &user,
            call,
            &src.model,
            src.temperature,
        )?;
        call += 1;

        let before = made;
        for line in reply.lines() {
            let line = line.trim().trim_start_matches("```json").trim_matches('`');
            // A `messages` array is a conversation, anything else a single
            // exchange — the same two shapes the loader reads.
            let rec = if let Some(turns) = neural_networks::sft::parse_messages(line) {
                Record::Chat {
                    turns,
                    category: json::field(line, "category")
                        .unwrap_or_else(|| src.category.clone()),
                }
            } else {
                let Some(instruction) = json::field(line, "instruction") else {
                    continue;
                };
                let Some(response) = json::field(line, "response") else {
                    continue;
                };
                Record::Sft {
                    instruction,
                    response,
                    context: json::field(line, "context").unwrap_or_default(),
                    category: json::field(line, "category")
                        .unwrap_or_else(|| src.category.clone()),
                }
            };
            made += 1;
            if !sink(rec) {
                return Ok(());
            }
            if made >= want {
                break;
            }
        }
        if made == before {
            barren += 1;
        } else {
            barren = 0;
        }
        // `call` counts generation calls only: the client's own counter also
        // holds the judge's, which runs once per record as it is produced.
        let cached = llm.borrow().cached;
        print!("\r      {made}/{want} generated ({call} calls, {cached} cached)");
        use std::io::Write;
        std::io::stdout().flush().ok();
    }
    println!();
    if made == 0 {
        return Err(format!(
            "[source {}] produced no parsable examples — the model must answer \
             with one JSON object per line, keys instruction/context/response",
            src.name
        ));
    }
    Ok(())
}

/// Chunk size for the streaming split. Large enough that a document never
/// spans more than two chunks in practice, small enough to bound memory.
const CHUNK: usize = 8 << 20;

fn read_file(path: &str, sep: &str, sink: &mut impl FnMut(Record) -> bool) -> Result<()> {
    let file = File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut reader = BufReader::new(file);
    if sep.is_empty() {
        let mut text = String::new();
        reader
            .read_to_string(&mut text)
            .map_err(|e| format!("{path}: {e}"))?;
        sink(Record::Doc { text });
        return Ok(());
    }

    // Split on the separator over raw bytes and carry the tail past the last
    // complete one into the next chunk: a document is never cut in half by a
    // read boundary, and neither is a multi-byte character.
    let sep = sep.as_bytes();
    let mut carry: Vec<u8> = Vec::new();
    let mut buf = vec![0u8; CHUNK];
    loop {
        let n = reader.read(&mut buf).map_err(|e| format!("{path}: {e}"))?;
        if n == 0 {
            break;
        }
        carry.extend_from_slice(&buf[..n]);
        let mut from = 0usize;
        while let Some(i) = find(&carry[from..], sep) {
            let doc = String::from_utf8_lossy(&carry[from..from + i]).into_owned();
            from += i + sep.len();
            if !sink(Record::Doc { text: doc }) {
                return Ok(());
            }
        }
        carry.drain(..from);
    }
    if !carry.is_empty() {
        let text = String::from_utf8_lossy(&carry).into_owned();
        if !text.trim().is_empty() {
            sink(Record::Doc { text });
        }
    }
    Ok(())
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

fn read_dir(src: &Source, sink: &mut impl FnMut(Record) -> bool) -> Result<()> {
    for f in list_files(src, &src.ext)? {
        let Ok(text) = std::fs::read(&f) else { continue };
        let text = String::from_utf8_lossy(&text).into_owned();
        if !sink(Record::Doc { text }) {
            break;
        }
    }
    Ok(())
}

/// The files of a directory source, sorted and cut down by `skip_files` /
/// `max_files`. Directory order is filesystem order, which differs between
/// machines — sorting is what makes a build reproducible, and what makes
/// "start at shard 6" mean the same thing twice.
fn list_files(src: &Source, ext: &[String]) -> Result<Vec<std::path::PathBuf>> {
    let mut files = Vec::new();
    walk(Path::new(&src.path), ext, &mut files)?;
    files.sort();
    if src.skip_files >= files.len() && !files.is_empty() {
        return Err(format!(
            "[source {}] skip_files = {} but the directory holds only {} matching files",
            src.name,
            src.skip_files,
            files.len()
        ));
    }
    let mut files = files.split_off(src.skip_files.min(files.len()));
    if src.max_files > 0 {
        files.truncate(src.max_files);
    }
    Ok(files)
}

fn walk(dir: &Path, ext: &[String], out: &mut Vec<std::path::PathBuf>) -> Result<()> {
    if dir.is_file() {
        out.push(dir.to_path_buf());
        return Ok(());
    }
    let entries = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, ext, out)?;
        } else if ext.is_empty()
            || p.extension()
                .and_then(|s| s.to_str())
                .is_some_and(|s| ext.iter().any(|e| e == s))
        {
            out.push(p);
        }
    }
    Ok(())
}

fn read_parquet(src: &Source, sink: &mut impl FnMut(Record) -> bool) -> Result<usize> {
    // A directory is a sharded corpus: every .parquet in it, in sorted order.
    if Path::new(&src.path).is_dir() {
        let files = list_files(src, &["parquet".to_string()])?;
        if files.is_empty() {
            return Err(format!("[source {}] no .parquet files in {}", src.name, src.path));
        }
        let mut skipped = 0usize;
        for (i, f) in files.iter().enumerate() {
            println!("      shard {}/{}: {}", i + 1, files.len(), f.display());
            let mut one = src.clone();
            one.path = f.display().to_string();
            skipped += read_parquet_file(&one, sink)?;
        }
        return Ok(skipped);
    }
    read_parquet_file(src, sink)
}

fn read_parquet_file(src: &Source, sink: &mut impl FnMut(Record) -> bool) -> Result<usize> {
    let langs = &src.filters.languages;
    let want_lang = !langs.is_empty();
    let cols: Vec<&str> = if want_lang {
        vec![&src.text_column, &src.language_column]
    } else {
        vec![&src.text_column]
    };
    let mut has_lang = want_lang;
    let mut reader = match ParquetColumnReader::open_columns(&src.path, &cols) {
        Ok(r) => r,
        // A corpus without the language column is read unfiltered rather than
        // failing the whole build — same rule as `batches.rs`.
        Err(e) if want_lang => {
            println!("  note: {e}; reading '{}' without language filtering", src.name);
            has_lang = false;
            ParquetColumnReader::open_columns(&src.path, &[&src.text_column])?
        }
        Err(e) => return Err(e),
    };

    let mut skipped = 0usize;
    while let Some(groups) = reader.next_row_group_columns()? {
        let texts = &groups[0];
        let langs_col = if has_lang && groups.len() > 1 {
            Some(&groups[1])
        } else {
            None
        };
        for (i, t) in texts.iter().enumerate() {
            if let Some(lc) = langs_col {
                let code = lc
                    .get(i)
                    .map(|b| String::from_utf8_lossy(b).to_lowercase())
                    .unwrap_or_default();
                if !langs.iter().any(|l| l.eq_ignore_ascii_case(&code)) {
                    skipped += 1;
                    continue;
                }
            }
            let text = String::from_utf8_lossy(t).into_owned();
            if !sink(Record::Doc { text }) {
                return Ok(skipped);
            }
        }
    }
    Ok(skipped)
}

/// A parquet chat corpus: one `list<struct<role, content>>` column per row, one
/// conversation per row. Turns whose role is none of system/user/assistant (a
/// tool call, say) are dropped — the model has no token for them — and so is a
/// conversation left without an exchange.
fn read_chat_parquet(src: &Source, sink: &mut impl FnMut(Record) -> bool) -> Result<()> {
    let files = if Path::new(&src.path).is_dir() {
        list_files(src, &["parquet".to_string()])?
    } else {
        vec![std::path::PathBuf::from(&src.path)]
    };
    if files.is_empty() {
        return Err(format!("[source {}] no .parquet files in {}", src.name, src.path));
    }
    for (i, f) in files.iter().enumerate() {
        if files.len() > 1 {
            println!("      shard {}/{}: {}", i + 1, files.len(), f.display());
        }
        let path = f.display().to_string();
        let mut reader = ParquetColumnReader::open_columns(
            &path,
            &[&src.role_column, &src.content_column],
        )?;
        // The subset column is flat, not inside the repeated messages group, so
        // it needs its own reader. Row groups are the same on both sides of the
        // file, so the two stay row-aligned group by group.
        let mut subset = match src.select_column.is_empty() {
            true => None,
            false => Some(ParquetColumnReader::open(&path, &src.select_column)?),
        };
        while let Some(rows) = reader.next_row_group_lists()? {
            let names = match subset.as_mut() {
                Some(r) => r.next_row_group()?.unwrap_or_default(),
                None => Vec::new(),
            };
            for (i, row) in rows.into_iter().enumerate() {
                if subset.is_some() {
                    // A missing name would silently pass the filter and let the
                    // whole corpus through — the one failure that must be loud.
                    let Some(name) = names.get(i) else {
                        return Err(format!(
                            "[source {}] {path}: column '{}' has {} rows but the \
                             messages column has more",
                            src.name,
                            src.select_column,
                            names.len()
                        ));
                    };
                    let name = String::from_utf8_lossy(name);
                    if !src.select_values.iter().any(|v| v == name.as_ref()) {
                        continue;
                    }
                }
                let mut turns = Vec::with_capacity(row.len());
                for element in row {
                    let role = String::from_utf8_lossy(&element[0]);
                    let Some(role) = Role::parse(&role) else { continue };
                    turns.push(Turn {
                        role,
                        content: String::from_utf8_lossy(&element[1]).into_owned(),
                    });
                }
                if turns.is_empty() {
                    continue;
                }
                if !sink(Record::Chat {
                    turns,
                    category: src.category.clone(),
                }) {
                    return Ok(());
                }
            }
        }
    }
    Ok(())
}

fn read_jsonl(src: &Source, dolly: bool, sink: &mut impl FnMut(Record) -> bool) -> Result<()> {
    let file = File::open(&src.path).map_err(|e| format!("{}: {e}", src.path))?;
    for line in BufReader::new(file).lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        // `messages` wins over the flat keys whichever kind was configured: a
        // conversation cannot be squeezed into instruction/response.
        if let Some(turns) = neural_networks::sft::parse_messages(&line) {
            let rec = Record::Chat {
                turns,
                category: json::field(&line, "category").unwrap_or_else(|| src.category.clone()),
            };
            if !sink(rec) {
                break;
            }
            continue;
        }
        let rec = if dolly {
            let Some(instruction) = json::field(&line, "instruction") else {
                continue;
            };
            let Some(response) = json::field(&line, "response") else {
                continue;
            };
            Record::Sft {
                instruction,
                response,
                context: json::field(&line, "context").unwrap_or_default(),
                category: json::field(&line, "category").unwrap_or_else(|| src.category.clone()),
            }
        } else {
            match json::field(&line, &src.text_column) {
                Some(text) => Record::Doc { text },
                None => continue,
            }
        };
        if !sink(rec) {
            break;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Filters, SourceKind};

    fn src(path: &str, skip: usize, max: usize) -> Source {
        Source {
            name: "t".into(),
            kind: SourceKind::Dir,
            path: path.into(),
            select_column: String::new(),
            select_values: Vec::new(),
            weight: 1.0,
            epochs: 1.0,
            ext: Vec::new(),
            separator: String::new(),
            skip_files: skip,
            max_files: max,
            text_column: "text".into(),
            language_column: "language".into(),
            role_column: "role".into(),
            content_column: "content".into(),
            count: 0,
            category: "t".into(),
            system: String::new(),
            prompt: String::new(),
            batch: 1,
            seed_file: String::new(),
            seeds: 1,
            model: String::new(),
            temperature: -1.0,
            filters: Filters::default(),
        }
    }

    /// A sharded corpus is resumed by skipping files of the *sorted* listing —
    /// `skip_files = 2` must start at the third shard, not the third the
    /// filesystem happens to return.
    #[test]
    fn skip_and_max_files_cut_the_sorted_listing() {
        let dir = std::env::temp_dir().join("datamix-list-files-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Written out of order on purpose.
        for i in [3usize, 0, 4, 1, 2] {
            std::fs::write(dir.join(format!("shard_{i:03}.parquet")), b"x").unwrap();
        }
        let d = dir.display().to_string();
        let names = |s: &Source| -> Vec<String> {
            list_files(s, &["parquet".to_string()])
                .unwrap()
                .iter()
                .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
                .collect()
        };
        assert_eq!(names(&src(&d, 0, 0)).len(), 5);
        assert_eq!(
            names(&src(&d, 2, 0)),
            vec!["shard_002.parquet", "shard_003.parquet", "shard_004.parquet"]
        );
        assert_eq!(names(&src(&d, 2, 2)), vec!["shard_002.parquet", "shard_003.parquet"]);
        // Skipping everything is a configuration mistake, not an empty corpus.
        assert!(list_files(&src(&d, 5, 0), &["parquet".to_string()]).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
