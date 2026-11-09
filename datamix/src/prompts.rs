// Sentences for the voice recorder: what the user reads aloud.
//
// Every staged record is cut into sentences, each sentence passes the gates
// below, and what survives is written as `{id, source, text, spoken}`. `text`
// stays as written, for the text model; `spoken` is what is actually said —
// lower case, no punctuation, digits and symbols written out — because that is
// what CTC aligns against. A sentence the normalizer cannot read one way only
// is dropped rather than guessed: a wrong transcript is worse than a missing
// prompt.
//
// The ids are a hash of the normalized text, so a rebuild keeps every id a
// recording already points at.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use neural_networks::sft::Role;

use crate::filter::Reject;
use crate::record::Record;

/// Tokens that never end a sentence although they end in a period.
const ABBREVIATIONS: [&str; 14] = [
    "mr", "mrs", "ms", "dr", "st", "vs", "e.g", "i.e", "etc", "approx", "no", "jr", "sr", "prof",
];

/// Shorthand as it is typed, and what is said for it.
const SHORTHAND: [(&str, &str); 26] = [
    ("u", "you"),
    ("ur", "your"),
    ("r", "are"),
    ("pls", "please"),
    ("plz", "please"),
    ("thx", "thanks"),
    ("ty", "thank you"),
    ("idk", "i don't know"),
    ("imo", "in my opinion"),
    ("btw", "by the way"),
    ("tbh", "to be honest"),
    ("rn", "right now"),
    ("b4", "before"),
    ("w/", "with"),
    ("e.g.", "for example"),
    ("i.e.", "that is"),
    ("etc.", "et cetera"),
    ("etc", "et cetera"),
    ("vs", "versus"),
    ("vs.", "versus"),
    ("mr.", "mister"),
    ("mrs.", "missus"),
    ("dr.", "doctor"),
    ("ok", "okay"),
    ("&", "and"),
    ("=", "equals"),
];

/// Acronyms that are said letter by letter. Any other capitalized short token
/// could be a word ("NASA") or letters ("NSA"), so it drops the sentence.
const SPELLED: [&str; 16] = [
    "TV", "PC", "USB", "PDF", "AI", "CPU", "GPU", "DVD", "CD", "URL", "FAQ", "USA", "UK", "BBC",
    "DJ", "ID",
];

/// Everything that marks a line as code, markup or an address rather than
/// something a person would say.
const MARKUP: [&str; 21] = [
    "http", "www.", "://", "@", "<", ">", "{", "}", "[", "]", "`", "|", "\\", "_", "#", "~", "^",
    "::", "->", ";", "()",
];

const ONES: [&str; 20] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen",
];
const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

pub struct Gate {
    pub min_words: usize,
    pub max_words: usize,
    /// Past this many words a sentence is kept only at `long_keep`: long
    /// prompts exist so the encoder sees every length, not to be the bulk.
    pub long_words: usize,
    pub long_keep: f32,
    pub origins: RefCell<Origins>,
}

/// Where each prompt came from: the record it was cut from and its index among
/// that record's [`sentences`], so a recording can be put back between its
/// neighbours — a text window with a spoken span in it. The records are kept
/// beside the pool as `<pool>_docs.jsonl`.
pub struct Origins {
    first: HashMap<u64, (u64, usize)>,
    logged: HashSet<u64>,
    staged: PathBuf,
    docs: BufWriter<File>,
}

impl Origins {
    pub fn create(staged: PathBuf) -> Result<Self, String> {
        if let Some(dir) = staged.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        let docs = File::create(&staged).map_err(|e| format!("{}: {e}", staged.display()))?;
        Ok(Self {
            first: HashMap::new(),
            logged: HashSet::new(),
            staged,
            docs: BufWriter::new(docs),
        })
    }

    /// Records the first place `sentence` was seen, and stages its record once.
    pub fn note(&mut self, rec: &Record, source: &str, index: usize, sentence: &str) -> Result<(), String> {
        let doc = key(&rec.train_text());
        self.first.entry(key(sentence)).or_insert((doc, index));
        if !self.logged.insert(doc) {
            return Ok(());
        }
        let body = match rec {
            Record::Doc { text } => format!("\"text\": \"{}\"", crate::json::escape(text)),
            other => {
                let rendered = crate::mix::render(other, crate::config::OutKind::Sft).unwrap_or_default();
                format!("\"record\": {}", rendered.trim_end())
            }
        };
        writeln!(
            self.docs,
            "{{\"doc\": \"{doc:016x}\", \"source\": \"{}\", {body}}}",
            crate::json::escape(source)
        )
        .map_err(|e| e.to_string())
    }

    pub fn get(&self, sentence: &str) -> Option<(u64, usize)> {
        self.first.get(&key(sentence)).copied()
    }

    /// Writes the staged records that `used` names to `path`.
    pub fn write_docs(&mut self, path: &str, used: &HashSet<u64>) -> Result<usize, String> {
        self.docs.flush().map_err(|e| e.to_string())?;
        let staged = File::open(&self.staged).map_err(|e| format!("{}: {e}", self.staged.display()))?;
        let mut out = BufWriter::new(File::create(path).map_err(|e| format!("{path}: {e}"))?);
        let mut written = 0;
        for line in BufReader::new(staged).lines() {
            let line = line.map_err(|e| e.to_string())?;
            let doc = crate::json::field(&line, "doc").and_then(|d| u64::from_str_radix(&d, 16).ok());
            if doc.is_some_and(|d| used.contains(&d)) {
                writeln!(out, "{line}").map_err(|e| e.to_string())?;
                written += 1;
            }
        }
        out.flush().map_err(|e| e.to_string())?;
        let _ = std::fs::remove_file(&self.staged);
        Ok(written)
    }
}

impl Gate {
    /// The spoken form of one sentence, or why it is not a prompt.
    pub fn check(&self, sentence: &str) -> Result<String, Reject> {
        let words = sentence.split_whitespace().count();
        if words < self.min_words || words > self.max_words {
            return Err(Reject::PromptLength);
        }
        // Drawn from the sentence's own hash, so a rebuild keeps the same ones.
        let draw = (key(sentence) >> 11) as f64 / (1u64 << 53) as f64;
        if words > self.long_words && draw >= self.long_keep as f64 {
            return Err(Reject::PromptLong);
        }
        if !sentence.is_ascii() {
            return Err(Reject::PromptCharset);
        }
        let lower = sentence.to_lowercase();
        if MARKUP.iter().any(|m| lower.contains(m)) {
            return Err(Reject::PromptMarkup);
        }
        let digits = sentence.bytes().filter(u8::is_ascii_digit).count();
        let numbers = sentence
            .split_whitespace()
            .filter(|w| w.bytes().any(|b| b.is_ascii_digit()))
            .count();
        let visible = sentence.bytes().filter(|b| !b.is_ascii_whitespace()).count();
        if numbers > 3 || digits * 10 > visible * 3 {
            return Err(Reject::PromptNumbers);
        }
        spoken(sentence).ok_or(Reject::Unspeakable)
    }
}

/// The sentences a record holds that a person could say to the assistant: a
/// document's text, an instruction, and a conversation's user turns — never
/// the assistant's side, which Jarvis says and nobody says to it.
pub fn sentences(rec: &Record) -> Vec<String> {
    match rec {
        Record::Doc { text } => split(text),
        Record::Sft { instruction, .. } => split(instruction),
        Record::Chat { turns, .. } => turns
            .iter()
            .filter(|t| t.role == Role::User)
            .flat_map(|t| split(&t.content))
            .collect(),
    }
}

/// Cuts text at sentence ends: `.`, `!` or `?` followed by a space, unless the
/// token before is an abbreviation or a lone letter, or the text uses capitals
/// and the next word does not start with one. All-lowercase text (chat) has no
/// capitals to go by, so there every end mark counts. Line breaks always end a
/// sentence.
pub fn split(text: &str) -> Vec<String> {
    // Typographic quotes and dashes have plain forms, and the harness font
    // draws nothing else.
    let text = text
        .replace(['\u{2019}', '\u{2018}'], "'")
        .replace(['\u{201c}', '\u{201d}'], "\"")
        .replace(['\u{2014}', '\u{2013}'], " - ");
    let cased = text.chars().any(|c| c.is_uppercase());
    let mut out = Vec::new();
    for line in text.lines() {
        let chars: Vec<char> = line.chars().collect();
        let mut start = 0;
        let mut i = 0;
        while i < chars.len() {
            if !matches!(chars[i], '.' | '!' | '?') {
                i += 1;
                continue;
            }
            let mut end = i + 1;
            while end < chars.len() && matches!(chars[end], '.' | '!' | '?' | '"' | '\'' | ')') {
                end += 1;
            }
            if end < chars.len() && !chars[end].is_whitespace() {
                i = end;
                continue;
            }
            let token: String = chars[start..i]
                .iter()
                .rev()
                .take_while(|c| !c.is_whitespace())
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<String>()
                .to_lowercase();
            let token = token.trim_start_matches(['"', '(', '\'']);
            let abbreviation = chars[i] == '.'
                && (ABBREVIATIONS.contains(&token)
                    || (token.len() == 1 && token.chars().all(|c| c.is_alphabetic())));
            let next = chars[end..].iter().find(|c| !c.is_whitespace());
            let lower_next = cased && next.is_some_and(|c| c.is_lowercase());
            if abbreviation || lower_next {
                i = end;
                continue;
            }
            push(&mut out, &chars[start..end]);
            start = end;
            i = end;
        }
        push(&mut out, &chars[start..]);
    }
    out
}

fn push(out: &mut Vec<String>, chars: &[char]) {
    let s: String = chars.iter().collect();
    let s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if !s.is_empty() {
        out.push(s);
    }
}

/// What is said for `sentence`, or `None` when there is more than one way to
/// say it.
pub fn spoken(sentence: &str) -> Option<String> {
    let shouting = !sentence.chars().any(|c| c.is_lowercase());
    let tokens: Vec<&str> = sentence.split_whitespace().collect();
    let mut out: Vec<String> = Vec::new();
    let mut skip = false;
    for (i, raw) in tokens.iter().enumerate() {
        if std::mem::take(&mut skip) {
            continue;
        }
        let lower = raw.to_lowercase();
        let is_number = |t: Option<&&str>| {
            t.is_some_and(|t| t.trim_matches(|c: char| !c.is_ascii_alphanumeric()).parse::<f64>().is_ok())
        };
        // Operators between two numbers.
        if is_number(i.checked_sub(1).and_then(|p| tokens.get(p))) && is_number(tokens.get(i + 1)) {
            let op = match *raw {
                "+" => Some("plus"),
                "-" => Some("minus"),
                "*" | "x" | "X" => Some("times"),
                "/" => Some("divided by"),
                _ => None,
            };
            if let Some(op) = op {
                out.push(op.to_string());
                continue;
            }
        }
        if matches!(*raw, "-" | "--") {
            continue;
        }
        if let Some((_, said)) = SHORTHAND.iter().find(|(typed, _)| *typed == lower) {
            out.push(said.to_string());
            continue;
        }
        let core = raw.trim_start_matches(['"', '\'', '(']);
        let core = core.trim_end_matches(['.', ',', '!', '?', ';', ':', '"', '\'', ')']);
        if core.is_empty() {
            continue;
        }
        let core_lower = core.to_lowercase();
        if let Some((_, said)) = SHORTHAND.iter().find(|(typed, _)| *typed == core_lower) {
            out.push(said.to_string());
            continue;
        }
        // "$100 billion" is said "one hundred billion dollars".
        let scale = tokens.get(i + 1).map(|t| {
            t.trim_end_matches(['.', ',', '!', '?', ';', ':', ')']).to_lowercase()
        });
        if let (Some(amount), Some(scale)) = (core.strip_prefix('$'), &scale)
            && matches!(scale.as_str(), "thousand" | "million" | "billion" | "trillion")
        {
            out.push(format!("{} {scale} dollars", decimal(amount)?));
            skip = true;
            continue;
        }
        if core.bytes().any(|b| b.is_ascii_digit()) {
            out.push(number_token(core)?);
            continue;
        }
        // An acronym, possibly plural or possessive: "TV", "TVs", "TV's".
        let bare = core
            .strip_suffix("'s")
            .or_else(|| core.strip_suffix('s'))
            .filter(|b| b.chars().filter(|c| c.is_alphabetic()).count() >= 2)
            .unwrap_or(core);
        let letters = bare.chars().filter(|c| c.is_alphabetic()).count();
        if !shouting && letters >= 2 && bare.chars().all(|c| !c.is_lowercase()) {
            if !SPELLED.contains(&bare) {
                return None;
            }
            let mut spelled: Vec<String> = bare.chars().map(|c| c.to_lowercase().to_string()).collect();
            spelled.last_mut().unwrap().push_str(&core[bare.len()..]);
            out.extend(spelled);
            continue;
        }
        if let Some(domain) = domain(&core_lower) {
            out.push(domain);
            continue;
        }
        // A lone letter other than a word ("a", "i") is a unit or a label:
        // "4 m" is metres or minutes.
        if core_lower.len() == 1 && !matches!(core_lower.as_str(), "a" | "i") {
            return None;
        }
        let word = core_lower.replace('-', " ");
        if !word.chars().all(|c| c.is_ascii_lowercase() || c == '\'' || c == ' ') {
            return None;
        }
        out.push(word);
    }
    let said = out.join(" ").split_whitespace().collect::<Vec<_>>().join(" ");
    said.chars().any(|c| c.is_ascii_lowercase()).then_some(said)
}

/// `youtube.com` -> `youtube dot com`.
fn domain(token: &str) -> Option<String> {
    let parts: Vec<&str> = token.split('.').collect();
    let label = |p: &str| !p.is_empty() && p.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
    if parts.len() < 2 || !parts.iter().all(|p| label(p)) {
        return None;
    }
    let tld = parts.last().unwrap();
    if tld.len() < 2 || !tld.chars().all(|c| c.is_ascii_lowercase()) {
        return None;
    }
    Some(parts.join(" dot ").replace('-', " dash "))
}

/// One token holding digits: a count, an amount, a percentage, an ordinal, a
/// decimal or a clock hour. A four-digit number without a separator is a year
/// as often as a count and is said differently as each, so it is refused.
fn number_token(token: &str) -> Option<String> {
    if let Some(amount) = token.strip_prefix('$') {
        let (whole, cents) = match amount.split_once('.') {
            Some((w, c)) if c.len() == 2 && c.bytes().all(|b| b.is_ascii_digit()) => (w, Some(c)),
            Some(_) => return None,
            None => (amount, None),
        };
        let n = integer(whole)?;
        let unit = if n == 1 { "dollar" } else { "dollars" };
        let mut said = format!("{} {unit}", words(n));
        if let Some(c) = cents {
            let c: u64 = c.parse().ok()?;
            if c > 0 {
                said.push(' ');
                said.push_str(&words(c));
            }
        }
        return Some(said);
    }
    if let Some(n) = token.strip_suffix('%') {
        return Some(format!("{} percent", decimal(n)?));
    }
    for (suffix, said) in [("am", "a m"), ("pm", "p m")] {
        if let Some(h) = token.to_lowercase().strip_suffix(suffix) {
            let h = integer(h)?;
            return (1..=12).contains(&h).then(|| format!("{} {said}", words(h)));
        }
    }
    for suffix in ["st", "nd", "rd", "th"] {
        if let Some(n) = token.to_lowercase().strip_suffix(suffix) {
            return Some(ordinal(&words(integer(n)?)));
        }
    }
    decimal(token)
}

fn decimal(token: &str) -> Option<String> {
    match token.split_once('.') {
        Some((whole, frac)) if !frac.is_empty() && frac.bytes().all(|b| b.is_ascii_digit()) => {
            let digits: Vec<&str> = frac.bytes().map(|b| ONES[(b - b'0') as usize]).collect();
            Some(format!("{} point {}", words(integer(whole)?), digits.join(" ")))
        }
        Some(_) => None,
        None => Some(words(integer(token)?)),
    }
}

/// A whole number, with or without thousands commas (which must then sit every
/// three digits).
fn integer(s: &str) -> Option<u64> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit() || b == b',') {
        return None;
    }
    if s.contains(',') {
        let groups: Vec<&str> = s.split(',').collect();
        let ok = (1..=3).contains(&groups[0].len()) && groups[1..].iter().all(|g| g.len() == 3);
        return if ok { s.replace(',', "").parse().ok() } else { None };
    }
    if s.len() == 4 || (s.len() > 1 && s.starts_with('0')) {
        return None;
    }
    s.parse().ok()
}

pub fn words(n: u64) -> String {
    if n < 20 {
        return ONES[n as usize].to_string();
    }
    if n < 100 {
        let t = TENS[(n / 10) as usize];
        return match n % 10 {
            0 => t.to_string(),
            r => format!("{t} {}", ONES[r as usize]),
        };
    }
    if n < 1000 {
        return match n % 100 {
            0 => format!("{} hundred", ONES[(n / 100) as usize]),
            r => format!("{} hundred {}", ONES[(n / 100) as usize], words(r)),
        };
    }
    for (scale, name) in [(1_000_000_000_000, "trillion"), (1_000_000_000, "billion"), (1_000_000, "million"), (1000, "thousand")] {
        if n >= scale {
            return match n % scale {
                0 => format!("{} {name}", words(n / scale)),
                r => format!("{} {name} {}", words(n / scale), words(r)),
            };
        }
    }
    unreachable!()
}

fn ordinal(cardinal: &str) -> String {
    let (head, last) = match cardinal.rsplit_once(' ') {
        Some((h, l)) => (format!("{h} "), l),
        None => (String::new(), cardinal),
    };
    let last = match last {
        "one" => "first".to_string(),
        "two" => "second".to_string(),
        "three" => "third".to_string(),
        "five" => "fifth".to_string(),
        "eight" => "eighth".to_string(),
        "nine" => "ninth".to_string(),
        "twelve" => "twelfth".to_string(),
        l if l.ends_with('y') => format!("{}ieth", &l[..l.len() - 1]),
        l => format!("{l}th"),
    };
    format!("{head}{last}")
}

/// Identity of a sentence for dedup and exclusion: its letters and digits,
/// lower-cased, one space between words. "Turn on the lamp." and "turn on the
/// lamp" are the same prompt.
pub fn key(sentence: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let mut space = false;
    let mut any = false;
    for c in sentence.chars().flat_map(char::to_lowercase) {
        if c.is_alphanumeric() {
            if space && any {
                h = (h ^ b' ' as u64).wrapping_mul(0x100000001b3);
            }
            let mut buf = [0u8; 4];
            for &b in c.encode_utf8(&mut buf).as_bytes() {
                h = (h ^ b as u64).wrapping_mul(0x100000001b3);
            }
            space = false;
            any = true;
        } else {
            space = true;
        }
    }
    h
}

/// The hex id written with a prompt.
pub fn id(sentence: &str) -> String {
    format!("{:016x}", key(sentence))
}

/// Which of `candidates` occur as a sentence anywhere in `paths`. Only the
/// candidates are held, so a 33 GB pretraining corpus costs a hash set of the
/// prompts, not of itself. JSONL is read as every string value on the line;
/// anything else as text. A directory is every file under it, and one `*` in
/// a file name matches any run of characters (`shard.*.txt`).
pub fn find_in(paths: &[String], candidates: &HashSet<u64>) -> Result<HashSet<u64>, String> {
    let mut files = Vec::new();
    for p in paths {
        collect_files(Path::new(p), &mut files).map_err(|e| format!("{p}: {e}"))?;
    }
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(files.len().max(1));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: Vec<Result<HashSet<u64>, String>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut hits = HashSet::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(file) = files.get(i) else { return Ok(hits) };
                        scan_file(file, candidates, &mut hits)
                            .map_err(|e| format!("{}: {e}", file.display()))?;
                    }
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut hits = HashSet::new();
    for r in results {
        hits.extend(r?);
    }
    Ok(hits)
}

fn collect_files(path: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if let Some((head, tail)) = name.split_once('*') {
        let dir = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
        let mut matched: Vec<PathBuf> = std::fs::read_dir(dir)?
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                let n = p.file_name().unwrap().to_string_lossy();
                n.len() >= head.len() + tail.len() && n.starts_with(head) && n.ends_with(tail)
            })
            .collect();
        if matched.is_empty() {
            return Err(std::io::Error::new(std::io::ErrorKind::NotFound, "pattern matches nothing"));
        }
        matched.sort();
        out.extend(matched);
        return Ok(());
    }
    if path.is_dir() {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(path)?.flatten().map(|e| e.path()).collect();
        entries.sort();
        for e in entries {
            collect_files(&e, out)?;
        }
    } else {
        out.push(path.to_path_buf());
    }
    Ok(())
}

fn scan_file(path: &Path, candidates: &HashSet<u64>, hits: &mut HashSet<u64>) -> std::io::Result<()> {
    let jsonl = path.extension().is_some_and(|e| e == "jsonl" || e == "eval")
        || path.to_string_lossy().ends_with(".jsonl.eval");
    let mut look = |text: &str| {
        for s in split(text) {
            let k = key(&s);
            if candidates.contains(&k) {
                hits.insert(k);
            }
        }
    };
    let mut reader = BufReader::with_capacity(1 << 20, std::fs::File::open(path)?);
    let mut line = Vec::new();
    loop {
        line.clear();
        if reader.read_until(b'\n', &mut line)? == 0 {
            return Ok(());
        }
        let line = String::from_utf8_lossy(&line);
        if jsonl {
            for value in json_strings(&line) {
                look(&value);
            }
        } else {
            for piece in line.split("<|endoftext|>") {
                look(piece);
            }
        }
    }
}

/// Every string value on one JSON line, decoded. Keys come out too; they are
/// single words and never match a prompt.
fn json_strings(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = line;
    while let Some(at) = rest.find('"') {
        let body = &rest[at + 1..];
        let Some(value) = crate::json::decode(body) else { break };
        // Step past the closing quote: the decoded length differs from the raw
        // one, so find it by scanning the raw text for an unescaped quote.
        let mut escaped = false;
        let mut close = body.len();
        for (i, b) in body.bytes().enumerate() {
            match b {
                b'\\' if !escaped => escaped = true,
                b'"' if !escaped => {
                    close = i;
                    break;
                }
                _ => escaped = false,
            }
        }
        out.push(value);
        rest = &body[(close + 1).min(body.len())..];
    }
    out
}

/// One prompt as written to the output. `origin` is its record and its index
/// among that record's sentences.
pub fn render(text: &str, source: &str, origin: Option<(u64, usize)>) -> Option<String> {
    let said = spoken(text)?;
    let origin = match origin {
        Some((doc, index)) => format!(", \"doc\": \"{doc:016x}\", \"index\": {index}"),
        None => String::new(),
    };
    Some(format!(
        "{{\"id\": \"{}\", \"source\": \"{}\", \"text\": \"{}\", \"spoken\": \"{}\"{origin}}}\n",
        id(text),
        crate::json::escape(source),
        crate::json::escape(text),
        crate::json::escape(&said),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentences_split_at_ends_but_not_abbreviations() {
        assert_eq!(
            split("Turn on the lamp. Then open Firefox! Dr. Smith said e.g. this.\nnew line"),
            ["Turn on the lamp.", "Then open Firefox!", "Dr. Smith said e.g. this.", "new line"]
        );
        assert_eq!(split("ok so. then what? idk"), ["ok so.", "then what?", "idk"]);
        assert_eq!(split("Version 3.5 is out."), ["Version 3.5 is out."]);
        assert_eq!(split("a bill for $87.40 \u{2014} how much?"), ["a bill for $87.40 - how much?"]);
    }

    #[test]
    fn spoken_writes_out_numbers_and_shorthand() {
        assert_eq!(spoken("What's 6 times 14?").unwrap(), "what's six times fourteen");
        assert_eq!(
            spoken("We split a bill for $87.40 three ways").unwrap(),
            "we split a bill for eighty seven dollars forty three ways"
        );
        assert_eq!(spoken("can u turn it off pls").unwrap(), "can you turn it off please");
        assert_eq!(spoken("Open youtube.com").unwrap(), "open youtube dot com");
        assert_eq!(spoken("Plug in the USB stick.").unwrap(), "plug in the u s b stick");
        assert_eq!(spoken("It's 20% off, 1,500 left").unwrap(), "it's twenty percent off one thousand five hundred left");
        assert_eq!(spoken("the 21st at 5pm").unwrap(), "the twenty first at five p m");
        assert_eq!(spoken("12 / 4 - 1").unwrap(), "twelve divided by four minus one");
        assert_eq!(spoken("a well-known fact").unwrap(), "a well known fact");
        assert_eq!(
            spoken("spending upwards of $100 billion each year").unwrap(),
            "spending upwards of one hundred billion dollars each year"
        );
        assert_eq!(spoken("two TVs and a PC's fan").unwrap(), "two t vs and a p c's fan");
    }

    #[test]
    fn ambiguous_readings_are_refused() {
        assert_eq!(spoken("back in 1999 it was fine"), None);
        assert_eq!(spoken("meet at 3:30"), None);
        assert_eq!(spoken("NASA launched it"), None);
        assert_eq!(spoken("solar flares and CMEs"), None);
        assert_eq!(spoken("it weighs 5kg"), None);
        assert_eq!(spoken("the room is 6.5 by 4 m"), None);
    }

    #[test]
    fn the_gate_reports_why() {
        let origins = || RefCell::new(Origins::create(std::env::temp_dir().join(format!("prompts-test-{}", std::process::id()))).unwrap());
        let gate = Gate { min_words: 4, max_words: 100, long_words: 25, long_keep: 0.0, origins: origins() };
        assert_eq!(gate.check("lights off"), Err(Reject::PromptLength));
        assert_eq!(gate.check("see https://x.org for more"), Err(Reject::PromptMarkup));
        assert_eq!(gate.check("das ist schön, oder nicht"), Err(Reject::PromptCharset));
        assert_eq!(gate.check("add 1 2 3 4 and 5"), Err(Reject::PromptNumbers));
        assert_eq!(gate.check("turn the lamp on").unwrap(), "turn the lamp on");
        let long = "word ".repeat(30);
        assert_eq!(gate.check(&long), Err(Reject::PromptLong));
        let keep_all = Gate { long_keep: 1.0, origins: origins(), ..gate };
        assert!(keep_all.check(&long).is_ok());
    }

    #[test]
    fn a_prompt_points_back_into_its_record() {
        let dir = std::env::temp_dir().join(format!("origins-{}", std::process::id()));
        let mut origins = Origins::create(dir.join("staged.jsonl")).unwrap();
        let rec = Record::Doc { text: "First one here. Second one there.".into() };
        let sentences = sentences(&rec);
        for (i, s) in sentences.iter().enumerate() {
            origins.note(&rec, "web", i, s).unwrap();
        }
        let (doc, index) = origins.get("Second one there.").unwrap();
        assert_eq!(index, 1);
        assert_eq!(sentences[index], "Second one there.");
        let line = render("Second one there.", "web", Some((doc, index))).unwrap();
        assert!(line.contains(&format!("\"doc\": \"{doc:016x}\", \"index\": 1")));

        let out = dir.join("docs.jsonl");
        let used = HashSet::from([doc]);
        assert_eq!(origins.write_docs(out.to_str().unwrap(), &used).unwrap(), 1);
        let docs = std::fs::read_to_string(&out).unwrap();
        assert!(docs.contains("\"text\": \"First one here. Second one there.\""));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn keys_ignore_case_and_punctuation() {
        assert_eq!(key("Turn on the lamp."), key("turn on  the lamp"));
        assert_ne!(key("turn on the lamp"), key("turn on the lamps"));
    }

    #[test]
    fn json_strings_reads_every_value() {
        let line = r#"{"messages": [{"role": "user", "content": "say \"hi\" now"}], "category": "x"}"#;
        let v = json_strings(line);
        assert!(v.contains(&"say \"hi\" now".to_string()));
        assert!(v.contains(&"x".to_string()));
    }
}
