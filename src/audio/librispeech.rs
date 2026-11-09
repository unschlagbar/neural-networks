//! LibriSpeech utterances with word-level forced alignments.
//!
//! Audio: `<root>/LibriSpeech/<split>/<speaker>/<chapter>/<id>.flac` (openslr 12).
//! Alignments: `<root>/alignments/<split>/<speaker>/<chapter>/<id>.TextGrid`, the
//! Montreal Forced Aligner output of zenodo record 2619474, whose `words` tier
//! holds every word with its start and end time and pauses as empty intervals.
//! Written text: `<root>/pc/<split>.json`, LibriSpeech-PC (openslr 145), the
//! transcripts with punctuation and capitalization restored, where it exists.
//! All of it for a split is kept in one file, `<root>/index/<split>.tsv`, made
//! on the first load: reading one file is a second where hundreds of thousands
//! of small ones on a hard disk are minutes. Delete it after changing the
//! alignments or the written transcripts.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// One spoken word and where it sits in the utterance, in seconds.
#[derive(Clone, Debug, PartialEq)]
pub struct AlignedWord {
    pub text: String,
    pub start: f32,
    pub end: f32,
}

#[derive(Clone, Debug)]
pub struct Utterance {
    pub id: String,
    pub speaker: String,
    pub audio: PathBuf,
    /// Spoken words in order, pauses dropped.
    pub words: Vec<AlignedWord>,
    /// The same words as written text, with punctuation and capitals.
    pub written: Option<String>,
}

impl Utterance {
    pub fn times(&self) -> Vec<(f32, f32)> {
        self.words.iter().map(|w| (w.start, w.end)).collect()
    }

    /// The transcript as running text: LibriSpeech is all caps and alignments
    /// all lower case, so it is lower-cased with the first word and the pronoun
    /// "I" capitalized — the closest the transcript gets to written text.
    pub fn text(&self) -> String {
        let mut out = String::new();
        for (i, w) in self.words.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            let pronoun = w.text == "i" || w.text.starts_with("i'");
            let mut chars = w.text.chars();
            match chars.next() {
                Some(c) if i == 0 || pronoun => {
                    out.extend(c.to_uppercase());
                    out.push_str(chars.as_str());
                }
                _ => out.push_str(&w.text),
            }
        }
        out
    }
}

pub struct LibriSpeech {
    pub utterances: Vec<Utterance>,
}

impl LibriSpeech {
    /// Every utterance of `split` under `root` that has both audio and an
    /// alignment, sorted by id; from the split's index when there is one.
    pub fn load(root: impl AsRef<Path>, split: &str) -> io::Result<Self> {
        let root = root.as_ref();
        let index = root.join("index").join(format!("{split}.tsv"));
        if let Some(c) = Self::read_index(&index) {
            return Ok(c);
        }
        let c = Self::scan(root, split)?;
        if let Err(e) = c.write_index(&index) {
            eprintln!("could not write {}: {e}", index.display());
        }
        Ok(c)
    }

    const INDEX_HEADER: &str = "#librispeech-index 1";

    /// One line per utterance: id, audio path, written text (empty if none),
    /// and the words as `word start end` triples, tab-separated.
    fn write_index(&self, path: &Path) -> io::Result<()> {
        let mut out = String::from(Self::INDEX_HEADER);
        out.push('\n');
        for u in &self.utterances {
            let written = u.written.as_deref().unwrap_or("").replace(['\t', '\n'], " ");
            let words: Vec<String> = u.words.iter().map(|w| format!("{} {} {}", w.text, w.start, w.end)).collect();
            out.push_str(&format!("{}\t{}\t{written}\t{}\n", u.id, u.audio.display(), words.join(" ")));
        }
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        let tmp = path.with_extension("tsv.tmp");
        fs::write(&tmp, out)?;
        fs::rename(tmp, path)
    }

    fn read_index(path: &Path) -> Option<Self> {
        let text = fs::read_to_string(path).ok()?;
        let mut lines = text.lines();
        (lines.next()? == Self::INDEX_HEADER).then_some(())?;
        let utterances = lines
            .map(|line| {
                let mut f = line.split('\t');
                let (id, audio, written, words) = (f.next()?, f.next()?, f.next()?, f.next()?);
                let parts: Vec<&str> = words.split(' ').collect();
                let words = parts
                    .chunks(3)
                    .map(|w| {
                        Some(AlignedWord {
                            text: w.first()?.to_string(),
                            start: w.get(1)?.parse().ok()?,
                            end: w.get(2)?.parse().ok()?,
                        })
                    })
                    .collect::<Option<Vec<_>>>()?;
                Some(Utterance {
                    speaker: id.split('-').next().unwrap_or("").to_string(),
                    id: id.to_string(),
                    audio: PathBuf::from(audio),
                    words,
                    written: (!written.is_empty()).then(|| written.to_string()),
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Self { utterances })
    }

    /// Every utterance of `split`, read from the audio, alignment and
    /// transcript files themselves.
    fn scan(root: &Path, split: &str) -> io::Result<Self> {
        let audio_root = root.join("LibriSpeech").join(split);
        let align_root = root.join("alignments").join(split);
        let written = Self::written(&root.join("pc").join(format!("{split}.json")));
        let mut utterances = Vec::new();
        for speaker in sorted_dirs(&audio_root)? {
            for chapter in sorted_dirs(&speaker)? {
                let mut flacs: Vec<PathBuf> = fs::read_dir(&chapter)?
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.extension().is_some_and(|e| e == "flac"))
                    .collect();
                flacs.sort();
                for audio in flacs {
                    let id = audio.file_stem().unwrap().to_string_lossy().into_owned();
                    let rel = chapter.strip_prefix(&audio_root).unwrap();
                    let grid = align_root.join(rel).join(format!("{id}.TextGrid"));
                    let Ok(text) = fs::read_to_string(&grid) else {
                        continue;
                    };
                    let words = TextGrid::words(&text)
                        .map_err(|e| io::Error::other(format!("{}: {e}", grid.display())))?;
                    utterances.push(Utterance {
                        speaker: id.split('-').next().unwrap_or("").to_string(),
                        written: written.get(&id).cloned(),
                        id,
                        audio,
                        words,
                    });
                }
            }
        }
        Ok(Self { utterances })
    }

    /// LibriSpeech-PC's written transcripts by utterance id; none when the
    /// manifest is missing.
    fn written(manifest: &Path) -> HashMap<String, String> {
        let Ok(text) = fs::read_to_string(manifest) else {
            return HashMap::new();
        };
        text.lines()
            .filter_map(|line| {
                let path = crate::sft::json_string_field(line, "audio_filepath")?;
                let id = Path::new(&path).file_stem()?.to_string_lossy().into_owned();
                Some((id, crate::sft::json_string_field(line, "text")?))
            })
            .collect()
    }

    /// Speakers in id order.
    pub fn speakers(&self) -> Vec<&str> {
        let mut s: Vec<&str> = self.utterances.iter().map(|u| u.speaker.as_str()).collect();
        s.dedup();
        s
    }
}

fn sorted_dirs(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut out: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    out.sort();
    Ok(out)
}

/// The slice of Praat's long TextGrid format that MFA writes.
pub struct TextGrid;

impl TextGrid {
    /// The non-empty intervals of the `words` tier.
    pub fn words(text: &str) -> Result<Vec<AlignedWord>, String> {
        let mut in_words = false;
        let (mut xmin, mut xmax) = (None, None);
        let mut out = Vec::new();
        for line in text.lines().map(str::trim) {
            if let Some(name) = line.strip_prefix("name = ") {
                in_words = name.trim_matches('"') == "words";
            } else if !in_words {
                continue;
            } else if let Some(v) = line.strip_prefix("xmin = ") {
                xmin = v.parse::<f32>().ok();
            } else if let Some(v) = line.strip_prefix("xmax = ") {
                xmax = v.parse::<f32>().ok();
            } else if let Some(v) = line.strip_prefix("text = ") {
                let word = v.trim_matches('"');
                let (Some(start), Some(end)) = (xmin.take(), xmax.take()) else {
                    return Err(format!("interval {word:?} without xmin/xmax"));
                };
                if !word.is_empty() {
                    out.push(AlignedWord {
                        text: word.to_string(),
                        start,
                        end,
                    });
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRID: &str = r#"File type = "ooTextFile"
item []:
	item [1]:
		class = "IntervalTier"
		name = "words"
		xmin = 0.0
		xmax = 1.2
		intervals: size = 3
			intervals [1]:
				xmin = 0.0
				xmax = 0.370
				text = ""
			intervals [2]:
				xmin = 0.370
				xmax = 0.720
				text = "i'm"
			intervals [3]:
				xmin = 0.720
				xmax = 1.2
				text = "well"
	item [2]:
		class = "IntervalTier"
		name = "phones"
		xmin = 0.0
		xmax = 1.2
		intervals: size = 1
			intervals [1]:
				xmin = 0.0
				xmax = 1.2
				text = "AY1"
"#;

    #[test]
    fn the_index_holds_what_was_scanned() {
        let dir = std::env::temp_dir().join(format!("nn-libri-index-{}", std::process::id()));
        let words = TextGrid::words(GRID).unwrap();
        let corpus = LibriSpeech {
            utterances: vec![
                Utterance {
                    id: "1-2-3".into(),
                    speaker: "1".into(),
                    audio: PathBuf::from("/a/1-2-3.flac"),
                    words: words.clone(),
                    written: Some("I'm well.".into()),
                },
                Utterance {
                    id: "4-5-6".into(),
                    speaker: "4".into(),
                    audio: PathBuf::from("/a/4-5-6.flac"),
                    words,
                    written: None,
                },
            ],
        };
        let path = dir.join("index").join("x.tsv");
        corpus.write_index(&path).unwrap();
        let back = LibriSpeech::read_index(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(back.utterances.len(), 2);
        for (a, b) in back.utterances.iter().zip(&corpus.utterances) {
            assert_eq!((&a.id, &a.speaker, &a.audio, &a.words, &a.written), (&b.id, &b.speaker, &b.audio, &b.words, &b.written));
        }
    }

    #[test]
    fn reads_the_words_tier_only() {
        let words = TextGrid::words(GRID).unwrap();
        assert_eq!(
            words,
            vec![
                AlignedWord { text: "i'm".into(), start: 0.37, end: 0.72 },
                AlignedWord { text: "well".into(), start: 0.72, end: 1.2 },
            ]
        );
        let u = Utterance {
            id: "1-2-3".into(),
            speaker: "1".into(),
            audio: PathBuf::new(),
            words,
            written: None,
        };
        assert_eq!(u.text(), "I'm well");
    }
}
