// Resume state for pretraining runs.
//
// A checkpoint stores weights and a step count, but the step count alone cannot
// say where in the corpus a run stopped: it counts optimizer steps across every
// corpus the weights have ever seen. Deriving a window offset from it
// (`step % windows`) silently breaks in two ways — a finished epoch resumes into
// the middle of the corpus instead of starting over, and pointing the trainer at
// a different corpus reuses an offset that means nothing there.
//
// This module stores the missing piece: WHICH corpus the run was walking and how
// far it got, as a small sidecar file next to the checkpoint.
//
// A corpus is a *directory* of shards (what `datamix` writes) or a single file.
// A run walks the sorted listing one file at a time, and the sidecar records
// both which file it is on and where inside it, so a stop anywhere resumes
// exactly there and a finished file is never re-read.
//
// Sidecar layout (`<model_path>.pretrainprog`, plain text, one `key value` per line):
//
//   version 2
//   corpus <string>     corpus root the run was pointed at (dir or file)
//   files_done <usize>  files of the sorted listing already finished
//   file <string>       the file in progress (empty between files)
//   windows <usize>     window count of THAT file when it started; 0 means the
//                       counting pass never ran, so the offset is unvalidated
//   epoch <usize>       1-based epoch (one epoch = one pass over every file)
//   done <usize>        windows completed WITHIN the file in progress
//   step <usize>        trainer step count, cross-checked against the checkpoint
//
// The file is written together with every checkpoint save, so the pair stays
// consistent: the sidecar never describes a position past the saved weights.
//
// It is never deleted. A finished run leaves the sidecar behind saying so —
// that is what lets a run be stopped and restarted freely, and what makes
// "which shards have I trained on?" answerable after the fact. To start over,
// delete it by hand or point the run at a different corpus.

use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

/// Where an interrupted pretraining run left off.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PretrainProgress {
    /// Corpus root the offset refers to; a different one invalidates the resume.
    pub corpus: String,
    /// Files of the sorted listing already finished.
    pub files_done: usize,
    /// The file in progress, by name. Empty between files.
    pub file: String,
    /// Window count of `file`, to detect a changed or re-generated shard.
    /// 0 means uncounted: the offset is still usable, just unvalidated.
    pub windows: usize,
    /// 1-based epoch: one epoch is one pass over every file.
    pub epoch: usize,
    /// Windows already trained on within `file`.
    pub done: usize,
    /// Trainer step count at the time of writing.
    pub step: usize,
}

impl PretrainProgress {
    /// A fresh run over `corpus`: epoch 1, first file, nothing done.
    pub fn fresh(corpus: &str, step: usize) -> Self {
        Self {
            corpus: corpus.to_string(),
            files_done: 0,
            file: String::new(),
            // Filled in by the trainer's counting pass before the first save.
            windows: 0,
            epoch: 1,
            done: 0,
            step,
        }
    }

    /// Whether this progress asks the trainer to skip anything at all.
    pub fn is_fresh(&self) -> bool {
        self.epoch == 1 && self.done == 0 && self.files_done == 0
    }
}

/// The files a corpus is made of: the sorted listing of a directory, or the one
/// file it names. Sorted, because "the third shard" has to mean the same thing
/// on every machine and after every restart.
pub struct Corpus {
    pub root: String,
    pub files: Vec<PathBuf>,
}

impl Corpus {
    pub fn open(root: &str) -> io::Result<Corpus> {
        let path = Path::new(root);
        if path.is_file() {
            return Ok(Corpus {
                root: root.to_string(),
                files: vec![path.to_path_buf()],
            });
        }
        if !path.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("corpus '{root}' is neither a file nor a directory"),
            ));
        }
        let mut files: Vec<PathBuf> = fs::read_dir(path)?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file())
            // A corpus folder also holds the build's report and its eval
            // holdout; neither is training data.
            .filter(|p| {
                let name = p.file_name().unwrap_or_default().to_string_lossy();
                !name.ends_with(".md") && !name.ends_with(".eval")
            })
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("corpus directory '{root}' holds no data files"),
            ));
        }
        Ok(Corpus {
            root: root.to_string(),
            files,
        })
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// The name recorded in the sidecar for a file, and looked up on resume.
    pub fn name_of(&self, index: usize) -> String {
        self.files
            .get(index)
            .map(|p| p.display().to_string())
            .unwrap_or_default()
    }

    pub fn path_of(&self, index: usize) -> &Path {
        &self.files[index]
    }
}

/// Sidecar path for a checkpoint path.
pub fn progress_path(model_path: &str) -> PathBuf {
    PathBuf::from(format!("{model_path}.pretrainprog"))
}

/// Write the sidecar. Called right after a checkpoint save so the two agree.
pub fn save(model_path: &str, p: &PretrainProgress) -> io::Result<()> {
    let path = progress_path(model_path);
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).ok();
    }
    let mut f = fs::File::create(&path)?;
    writeln!(f, "version 2")?;
    writeln!(f, "corpus {}", p.corpus)?;
    writeln!(f, "files_done {}", p.files_done)?;
    writeln!(f, "file {}", p.file)?;
    writeln!(f, "windows {}", p.windows)?;
    writeln!(f, "epoch {}", p.epoch)?;
    writeln!(f, "done {}", p.done)?;
    writeln!(f, "step {}", p.step)?;
    Ok(())
}

/// Read the sidecar, or `None` if it is absent or malformed. A version-1 file
/// (single-corpus, no file list) reads as a run on the first and only file.
pub fn load(model_path: &str) -> Option<PretrainProgress> {
    let text = fs::read_to_string(progress_path(model_path)).ok()?;
    let mut corpus = None;
    let mut files_done = 0usize;
    let mut file = String::new();
    let mut windows = None;
    let mut epoch = None;
    let mut done = None;
    let mut step = None;
    let mut version = None;
    for line in text.lines() {
        let mut it = line.splitn(2, char::is_whitespace);
        let (Some(key), Some(val)) = (it.next(), it.next()) else {
            continue;
        };
        let val = val.trim();
        match key {
            "version" => version = val.parse::<u32>().ok(),
            // A corpus path may contain spaces, so it is taken verbatim.
            "corpus" => corpus = Some(val.to_string()),
            "files_done" => files_done = val.parse::<usize>().unwrap_or(0),
            "file" => file = val.to_string(),
            "windows" => windows = val.parse::<usize>().ok(),
            "epoch" => epoch = val.parse::<usize>().ok(),
            "done" => done = val.parse::<usize>().ok(),
            "step" => step = val.parse::<usize>().ok(),
            _ => {}
        }
    }
    let version = version?;
    if version != 1 && version != 2 {
        return None;
    }
    let corpus = corpus?;
    // Version 1 named the single corpus file; it is both the root and the file.
    if version == 1 {
        file = corpus.clone();
    }
    Some(PretrainProgress {
        corpus,
        files_done,
        file,
        windows: windows?,
        epoch: epoch?,
        done: done?,
        step: step?,
    })
}

/// Where a run should start. `file` indexes into [`Corpus::files`];
/// `file == corpus.len()` means every file of this epoch is already done.
pub struct Start {
    pub progress: PretrainProgress,
    pub file: usize,
}

/// Decide the starting point for a run: either a validated resume from the
/// sidecar, or a fresh start. `corpus` is what is about to be trained on and
/// `step` the checkpoint's step count; both are cross-checked, and any mismatch
/// falls back to a fresh run with a printed reason. Resuming into a corpus the
/// offset was not measured against would skip the wrong windows.
pub fn resume_or_fresh(
    model_path: &str,
    corpus: &Corpus,
    step: usize,
    epochs: usize,
) -> Start {
    let fresh = |reason: Option<String>| {
        if let Some(r) = reason {
            println!("Pretrain resume: {r}");
        }
        Start {
            progress: PretrainProgress::fresh(&corpus.root, step),
            file: 0,
        }
    };

    let Some(p) = load(model_path) else {
        // No sidecar: either a genuinely new run, or a checkpoint written before
        // progress files existed. Both start at the beginning of this corpus —
        // the alternative (guessing an offset from the step count) is what this
        // module exists to remove.
        return fresh((step > 0).then(|| {
            format!(
                "no progress file next to '{model_path}' — starting at the beginning of \
                 '{}' (weights and step {step} are kept).",
                corpus.root
            )
        }));
    };
    if p.corpus != corpus.root {
        return fresh(Some(format!(
            "progress file is for corpus '{}' but training on '{}' — starting a fresh pass.",
            p.corpus, corpus.root
        )));
    }
    if p.step != step {
        return fresh(Some(format!(
            "checkpoint is at step {step} but progress file says {} — starting a fresh pass.",
            p.step
        )));
    }
    if p.epoch > epochs {
        println!(
            "Pretrain resume: corpus '{}' is complete ({} files, {} epochs). Delete '{}' \
             or point the run at another corpus to start again.",
            corpus.root,
            corpus.len(),
            epochs,
            progress_path(model_path).display()
        );
        return Start {
            file: corpus.len(),
            progress: p,
        };
    }
    if p.files_done > corpus.len() {
        return fresh(Some(format!(
            "progress file says {} files done but '{}' holds {} — starting a fresh pass.",
            p.files_done,
            corpus.root,
            corpus.len()
        )));
    }
    // The listing may have grown (new shards dropped into the folder) or been
    // rebuilt. The recorded name is what says whether `files_done` still points
    // at the same place.
    let at = p.files_done;
    if at < corpus.len() && !p.file.is_empty() && p.file != corpus.name_of(at) {
        return fresh(Some(format!(
            "progress file was in '{}' but file {at} of '{}' is now '{}' — starting a \
             fresh pass.",
            p.file,
            corpus.root,
            corpus.name_of(at)
        )));
    }
    if p.is_fresh() {
        return Start {
            progress: p,
            file: 0,
        };
    }
    if at >= corpus.len() {
        println!(
            "Pretrain resume: epoch {} of '{}' is complete ({} files).",
            p.epoch,
            corpus.root,
            corpus.len()
        );
    } else {
        println!(
            "Pretrain resume: epoch {}, file {}/{} ('{}') at window {} (step {step}).",
            p.epoch,
            at + 1,
            corpus.len(),
            corpus.name_of(at),
            p.done
        );
    }
    Start {
        file: at,
        progress: p,
    }
}

/// Ask for the corpus to train on. Called only when there is no sidecar — a
/// resumed run already knows which corpus it is walking, and asking again would
/// invite pointing it at a different one by accident.
pub fn ask_corpus(default: &str) -> String {
    print!("Training data (file or directory) [{default}]: ");
    io::stdout().flush().ok();
    let mut line = String::new();
    if io::stdin().read_line(&mut line).is_err() {
        return default.to_string();
    }
    let name = line.trim();
    if name.is_empty() {
        default.to_string()
    } else {
        name.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> String {
        let dir = std::env::temp_dir().join(name);
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir.join("model").to_str().unwrap().to_string()
    }

    /// A corpus directory of shards, plus the report and eval file a build
    /// leaves beside them.
    fn temp_corpus(name: &str, shards: usize) -> String {
        let dir = std::env::temp_dir().join(name);
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        for i in 0..shards {
            fs::write(dir.join(format!("shard.{i:03}.txt")), b"hello<|endoftext|>").unwrap();
        }
        fs::write(dir.join("report.md"), b"# report").unwrap();
        fs::write(dir.join("shard.txt.eval"), b"held out").unwrap();
        dir.to_str().unwrap().to_string()
    }

    #[test]
    fn round_trips() {
        let path = temp_path("pretrainprog_round_trip");
        let p = PretrainProgress {
            corpus: "data/mix/pretrain_en".to_string(),
            files_done: 3,
            file: "data/mix/pretrain_en/shard.003.txt".to_string(),
            windows: 120_000,
            epoch: 1,
            done: 45_678,
            step: 45_678,
        };
        save(&path, &p).unwrap();
        assert_eq!(load(&path), Some(p));
    }

    /// A corpus is its sorted data files — the report and the eval holdout a
    /// build leaves in the folder are not training data.
    #[test]
    fn corpus_lists_only_data_files_in_order() {
        let dir = temp_corpus("pretrainprog_corpus_list", 3);
        let c = Corpus::open(&dir).unwrap();
        assert_eq!(c.len(), 3);
        let names: Vec<String> = c
            .files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            vec!["shard.000.txt", "shard.001.txt", "shard.002.txt"]
        );
        // A single file is a one-file corpus, so both shapes walk the same way.
        let one = Corpus::open(c.files[0].to_str().unwrap()).unwrap();
        assert_eq!(one.len(), 1);
    }

    /// The point of the walk: a run that stopped inside shard 2 resumes inside
    /// shard 2, not at the start of the corpus.
    #[test]
    fn resumes_at_the_file_it_stopped_in() {
        let path = temp_path("pretrainprog_resume_file");
        let dir = temp_corpus("pretrainprog_resume_corpus", 4);
        let c = Corpus::open(&dir).unwrap();
        let p = PretrainProgress {
            corpus: dir.clone(),
            files_done: 2,
            file: c.name_of(2),
            windows: 500,
            epoch: 1,
            done: 137,
            step: 900,
        };
        save(&path, &p).unwrap();

        let start = resume_or_fresh(&path, &c, 900, 1);
        assert_eq!(start.file, 2);
        assert_eq!(start.progress.done, 137);

        // A different checkpoint step means the pair is inconsistent.
        let start = resume_or_fresh(&path, &c, 12, 1);
        assert_eq!(start.file, 0);
        assert!(start.progress.is_fresh());
    }

    /// Shards that were rebuilt (so file 2 is now a different file) must not be
    /// resumed into at the old offset.
    #[test]
    fn a_renamed_shard_forces_a_fresh_pass() {
        let path = temp_path("pretrainprog_renamed");
        let dir = temp_corpus("pretrainprog_renamed_corpus", 3);
        let c = Corpus::open(&dir).unwrap();
        let p = PretrainProgress {
            corpus: dir.clone(),
            files_done: 1,
            file: format!("{dir}/shard.999.txt"),
            windows: 10,
            epoch: 1,
            done: 5,
            step: 5,
        };
        save(&path, &p).unwrap();
        let start = resume_or_fresh(&path, &c, 5, 1);
        assert!(start.progress.is_fresh());
        assert_eq!(start.file, 0);
    }

    /// Pointing the trainer at a different corpus must not reuse the offset.
    #[test]
    fn corpus_change_forces_fresh() {
        let path = temp_path("pretrainprog_corpus_change");
        let dir = temp_corpus("pretrainprog_change_a", 2);
        let other = temp_corpus("pretrainprog_change_b", 2);
        let p = PretrainProgress {
            corpus: dir,
            files_done: 1,
            file: String::new(),
            windows: 100,
            epoch: 1,
            done: 50,
            step: 50,
        };
        save(&path, &p).unwrap();
        let start = resume_or_fresh(&path, &Corpus::open(&other).unwrap(), 50, 1);
        assert!(start.progress.is_fresh());
    }

    /// A finished run keeps its sidecar and reports the corpus as done, instead
    /// of silently starting over.
    #[test]
    fn a_finished_corpus_stays_finished() {
        let path = temp_path("pretrainprog_finished");
        let dir = temp_corpus("pretrainprog_finished_corpus", 2);
        let c = Corpus::open(&dir).unwrap();
        let p = PretrainProgress {
            corpus: dir,
            files_done: 2,
            file: String::new(),
            windows: 0,
            epoch: 2, // past EPOCHS = 1
            done: 0,
            step: 700,
        };
        save(&path, &p).unwrap();
        let start = resume_or_fresh(&path, &c, 700, 1);
        assert_eq!(start.file, c.len(), "nothing left to train on");
        assert!(load(&path).is_some(), "the sidecar is never deleted");
    }

    /// A version-1 sidecar (single corpus file, no walk) still resumes.
    #[test]
    fn version_1_sidecar_still_loads() {
        let path = temp_path("pretrainprog_v1");
        let file = progress_path(&path);
        fs::write(
            &file,
            "version 1\ncorpus data/c.txt\nwindows 500\nepoch 1\ndone 300\nstep 300\n",
        )
        .unwrap();
        let p = load(&path).unwrap();
        assert_eq!(p.corpus, "data/c.txt");
        assert_eq!(p.file, "data/c.txt");
        assert_eq!(p.files_done, 0);
        assert_eq!(p.done, 300);
    }

    /// A corpus path containing spaces survives the round trip.
    #[test]
    fn corpus_with_spaces() {
        let path = temp_path("pretrainprog_spaces");
        let p = PretrainProgress {
            corpus: "/data/my corpus".to_string(),
            files_done: 0,
            file: "/data/my corpus/000 1.txt".to_string(),
            windows: 10,
            epoch: 2,
            done: 3,
            step: 13,
        };
        save(&path, &p).unwrap();
        let got = load(&path).unwrap();
        assert_eq!(got.corpus, p.corpus);
        assert_eq!(got.file, p.file);
    }
}
