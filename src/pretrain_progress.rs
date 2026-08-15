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
// Sidecar layout (`<model_path>.pretrainprog`, plain text, one `key value` per line):
//
//   version 1
//   corpus <string>   TRAIN_DATA path the offset refers to
//   windows <usize>   window count of that corpus when the run started; 0 means
//                     the counting pass never ran, so the offset is unvalidated
//   epoch <usize>     1-based epoch the run was in
//   done <usize>      windows completed WITHIN that epoch
//   step <usize>      trainer step count, cross-checked against the checkpoint
//
// The file is written together with every checkpoint save, so the pair stays
// consistent: the sidecar never describes a position past the saved weights. It
// is deleted once the last epoch completes, so a finished run starts over rather
// than resuming at the end of the corpus.

use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

/// Where an interrupted pretraining run left off.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PretrainProgress {
    /// Corpus the offset refers to; a different one invalidates the resume.
    pub corpus: String,
    /// Window count of `corpus`, to detect a changed or re-generated file.
    /// 0 means uncounted: the offset is still usable, just unvalidated.
    pub windows: usize,
    /// 1-based epoch the run was in.
    pub epoch: usize,
    /// Windows already trained on within `epoch`.
    pub done: usize,
    /// Trainer step count at the time of writing.
    pub step: usize,
}

impl PretrainProgress {
    /// A fresh run over `corpus`: epoch 1, nothing done.
    pub fn fresh(corpus: &str, step: usize) -> Self {
        Self {
            corpus: corpus.to_string(),
            // Filled in by the trainer's counting pass before the first save.
            windows: 0,
            epoch: 1,
            done: 0,
            step,
        }
    }

    /// Whether this progress asks the trainer to skip anything at all.
    pub fn is_fresh(&self) -> bool {
        self.epoch == 1 && self.done == 0
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
    writeln!(f, "version 1")?;
    writeln!(f, "corpus {}", p.corpus)?;
    writeln!(f, "windows {}", p.windows)?;
    writeln!(f, "epoch {}", p.epoch)?;
    writeln!(f, "done {}", p.done)?;
    writeln!(f, "step {}", p.step)?;
    Ok(())
}

/// Read the sidecar, or `None` if it is absent or malformed.
pub fn load(model_path: &str) -> Option<PretrainProgress> {
    let text = fs::read_to_string(progress_path(model_path)).ok()?;
    let mut corpus = None;
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
            "windows" => windows = val.parse::<usize>().ok(),
            "epoch" => epoch = val.parse::<usize>().ok(),
            "done" => done = val.parse::<usize>().ok(),
            "step" => step = val.parse::<usize>().ok(),
            _ => {}
        }
    }
    if version? != 1 {
        return None;
    }
    Some(PretrainProgress {
        corpus: corpus?,
        windows: windows?,
        epoch: epoch?,
        done: done?,
        step: step?,
    })
}

/// Delete the sidecar — done when a run finishes all its epochs, so the next
/// invocation starts a genuinely new run rather than resuming a finished one.
pub fn clear(model_path: &str) {
    let path = progress_path(model_path);
    if Path::new(&path).exists() {
        let _ = fs::remove_file(path);
    }
}

/// Decide the starting point for a run: either a validated resume from the
/// sidecar, or a fresh start. `corpus` is the corpus about to be trained on and
/// `step` the checkpoint's step count; both are cross-checked, and any mismatch
/// falls back to a fresh run with a printed reason. Resuming into a corpus the
/// offset was not measured against would skip the wrong windows.
pub fn resume_or_fresh(
    model_path: &str,
    corpus: &str,
    step: usize,
    epochs: usize,
) -> PretrainProgress {
    let Some(p) = load(model_path) else {
        // No sidecar: either a genuinely new run, or a checkpoint written before
        // progress files existed. Both start at the beginning of this corpus —
        // the alternative (guessing an offset from the step count) is what this
        // module exists to remove.
        if step > 0 {
            println!(
                "Pretrain resume: no progress file next to '{model_path}' — \
                 starting at the beginning of '{corpus}' (weights and step {step} are kept)."
            );
        }
        return PretrainProgress::fresh(corpus, step);
    };
    if p.corpus != corpus {
        println!(
            "Pretrain resume: progress file is for corpus '{}' but training on '{corpus}' — \
             starting a fresh pass.",
            p.corpus
        );
        return PretrainProgress::fresh(corpus, step);
    }
    if p.step != step {
        println!(
            "Pretrain resume: checkpoint is at step {step} but progress file says {} — \
             starting a fresh pass.",
            p.step
        );
        return PretrainProgress::fresh(corpus, step);
    }
    if p.epoch > epochs {
        println!(
            "Pretrain resume: recorded epoch {} is past EPOCHS={epochs} — starting a fresh pass.",
            p.epoch
        );
        return PretrainProgress::fresh(corpus, step);
    }
    if p.is_fresh() {
        return p;
    }
    println!(
        "Pretrain resume: continuing epoch {} at window {} of '{corpus}' (step {step}).",
        p.epoch, p.done
    );
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> String {
        let dir = std::env::temp_dir().join(name);
        fs::create_dir_all(&dir).unwrap();
        dir.join("model").to_str().unwrap().to_string()
    }

    /// A written sidecar reads back identically.
    #[test]
    fn round_trips() {
        let path = temp_path("pretrainprog_round_trip");
        let p = PretrainProgress {
            corpus: "../../training_data/000_00001.parquet".to_string(),
            windows: 120_000,
            epoch: 1,
            done: 45_678,
            step: 45_678,
        };
        save(&path, &p).unwrap();
        assert_eq!(load(&path), Some(p));
        clear(&path);
        assert_eq!(load(&path), None);
    }

    /// The bug this module fixes: swapping in a different corpus must not reuse
    /// the previous corpus's window offset.
    #[test]
    fn corpus_change_forces_fresh() {
        let path = temp_path("pretrainprog_corpus_change");
        let p = PretrainProgress {
            corpus: "data/000_00001.parquet".to_string(),
            windows: 100_000,
            epoch: 1,
            done: 100_000,
            step: 100_000,
        };
        save(&path, &p).unwrap();

        // Same corpus and step resumes exactly.
        assert_eq!(resume_or_fresh(&path, "data/000_00001.parquet", 100_000, 1), p);
        // The next parquet starts at window 0, not at 100k.
        let fresh = resume_or_fresh(&path, "data/000_00002.parquet", 100_000, 1);
        assert!(fresh.is_fresh());
        assert_eq!(fresh.corpus, "data/000_00002.parquet");
        // A step mismatch starts over.
        let fresh = resume_or_fresh(&path, "data/000_00001.parquet", 999, 1);
        assert!(fresh.is_fresh());
        clear(&path);
    }

    /// A finished run deletes its sidecar, so the next invocation re-reads the
    /// corpus from the start instead of resuming past its end.
    #[test]
    fn finished_run_starts_over() {
        let path = temp_path("pretrainprog_finished");
        let p = PretrainProgress {
            corpus: "data/c.parquet".to_string(),
            windows: 500,
            epoch: 1,
            done: 500,
            step: 500,
        };
        save(&path, &p).unwrap();
        clear(&path);
        let next = resume_or_fresh(&path, "data/c.parquet", 500, 1);
        assert!(next.is_fresh());
    }

    /// A sidecar written before any counting pass has `windows == 0`. That is an
    /// unmeasured count, not a mismatched one, and must not discard the offset.
    #[test]
    fn uncounted_windows_still_resumes() {
        let path = temp_path("pretrainprog_uncounted");
        let p = PretrainProgress {
            corpus: "data/c.parquet".to_string(),
            windows: 0,
            epoch: 1,
            done: 59_216,
            step: 1_004_000,
        };
        save(&path, &p).unwrap();
        let got = resume_or_fresh(&path, "data/c.parquet", 1_004_000, 1);
        assert!(!got.is_fresh());
        assert_eq!(got.done, 59_216);
        clear(&path);
    }

    /// A corpus path containing spaces survives the round trip.
    #[test]
    fn corpus_with_spaces() {
        let path = temp_path("pretrainprog_spaces");
        let p = PretrainProgress {
            corpus: "/data/my corpus/000 1.parquet".to_string(),
            windows: 10,
            epoch: 2,
            done: 3,
            step: 13,
        };
        save(&path, &p).unwrap();
        assert_eq!(load(&path).unwrap().corpus, p.corpus);
        clear(&path);
    }
}
