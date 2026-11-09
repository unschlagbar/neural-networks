//! Takes recorded on the harness VOICE page: `<dir>/<speaker>/metadata.jsonl`,
//! one line per kept take with the written prompt (`text`), what was said
//! (`spoken`, lower case, numbers written out) and the session it belongs to;
//! the audio is `<dir>/<speaker>/<file_name>`.

use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use super::librispeech::{AlignedWord, Utterance};
use crate::sft::json_string_field;

pub const ALIGNMENT_FILE: &str = "alignment.jsonl";
/// Verdicts from the harness VOICE page's review mode,
/// `{"id", "verdict": "good"|"bad"}`, appended; the last line for an id wins.
pub const REVIEW_FILE: &str = "review.jsonl";

#[derive(Clone, Debug)]
pub struct Take {
    /// `<session>/<prompt hash>`, unique per speaker.
    pub id: String,
    pub speaker: String,
    pub session: String,
    pub audio: PathBuf,
    pub text: String,
    pub spoken: String,
    /// The document the prompt was cut from, when it was (`data/voice/prompts_docs.jsonl`).
    pub doc: Option<String>,
    /// End of each spoken word, seconds.
    pub ends: Option<Vec<f32>>,
}

impl Take {
    pub fn spoken_words(&self) -> Vec<&str> {
        self.spoken.split_whitespace().collect()
    }

    /// As an utterance once aligned. Its speaker is `<speaker>/<session>`, so
    /// holding out speakers holds out whole sessions.
    pub fn utterance(&self) -> Option<Utterance> {
        let ends = self.ends.as_ref()?;
        let spoken = self.spoken_words();
        if ends.len() != spoken.len() {
            return None;
        }
        let mut start = 0.0;
        let words = spoken
            .iter()
            .zip(ends)
            .map(|(w, &end)| {
                let word = AlignedWord {
                    text: w.to_string(),
                    start,
                    end,
                };
                start = end;
                word
            })
            .collect();
        Some(Utterance {
            id: format!("{}/{}", self.speaker, self.id),
            speaker: format!("{}/{}", self.speaker, self.session),
            audio: self.audio.clone(),
            words,
            written: Some(self.text.clone()),
        })
    }
}

/// Every take under `dir`, speakers in name order, with its alignment when
/// there is one. Takes reviewed as bad are left out.
pub fn load(dir: impl AsRef<Path>) -> io::Result<Vec<Take>> {
    let mut speakers: Vec<PathBuf> = fs::read_dir(dir.as_ref())?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("metadata.jsonl").is_file())
        .collect();
    speakers.sort();
    let mut out = Vec::new();
    for root in speakers {
        let speaker = root.file_name().unwrap().to_string_lossy().into_owned();
        let ends = alignments(&root);
        let verdicts = reviews(&root);
        for line in fs::read_to_string(root.join("metadata.jsonl"))?.lines() {
            let field = |k| json_string_field(line, k);
            let (Some(id), Some(file), Some(text), Some(spoken)) =
                (field("id"), field("file_name"), field("text"), field("spoken"))
            else {
                continue;
            };
            if verdicts.get(&id) == Some(&false) {
                continue;
            }
            out.push(Take {
                session: field("session").unwrap_or_default(),
                speaker: speaker.clone(),
                audio: root.join(file),
                ends: ends.get(&id).cloned(),
                doc: field("doc").filter(|d| !d.is_empty()),
                id,
                text,
                spoken,
            });
        }
    }
    Ok(out)
}

fn alignments(root: &Path) -> HashMap<String, Vec<f32>> {
    let Ok(text) = fs::read_to_string(root.join(ALIGNMENT_FILE)) else {
        return HashMap::new();
    };
    text.lines()
        .filter_map(|line| {
            let ends = json_string_field(line, "ends")?
                .split_whitespace()
                .map(str::parse::<f32>)
                .collect::<Result<Vec<f32>, _>>()
                .ok()?;
            Some((json_string_field(line, "id")?, ends))
        })
        .collect()
}

/// `id -> good` from `<speaker dir>/review.jsonl`.
pub fn reviews(root: &Path) -> HashMap<String, bool> {
    let Ok(text) = fs::read_to_string(root.join(REVIEW_FILE)) else {
        return HashMap::new();
    };
    text.lines()
        .filter_map(|line| {
            let good = json_string_field(line, "verdict")? == "good";
            Some((json_string_field(line, "id")?, good))
        })
        .collect()
}

/// Rewrites `<dir>/<speaker>/alignment.jsonl` for every speaker in `takes`.
pub fn save_alignments(dir: impl AsRef<Path>, takes: &[Take]) -> io::Result<()> {
    let mut by_speaker: HashMap<&str, Vec<&Take>> = HashMap::new();
    for t in takes.iter().filter(|t| t.ends.is_some()) {
        by_speaker.entry(&t.speaker).or_default().push(t);
    }
    for (speaker, takes) in by_speaker {
        let path = dir.as_ref().join(speaker).join(ALIGNMENT_FILE);
        let tmp = path.with_extension("jsonl.tmp");
        let mut f = io::BufWriter::new(fs::File::create(&tmp)?);
        for t in takes {
            let ends: Vec<String> = t.ends.as_ref().unwrap().iter().map(|e| format!("{e:.2}")).collect();
            writeln!(f, "{{\"id\": \"{}\", \"ends\": \"{}\"}}", t.id, ends.join(" "))?;
        }
        f.flush()?;
        drop(f);
        fs::rename(tmp, path)?;
    }
    Ok(())
}
