// Resume state for SFT fine-tuning runs.
//
// A checkpoint stores weights and a step count, so stopping a run never loses
// what was learned — but on restart the trainer would begin again at epoch 1,
// example 0, re-training examples it has already seen and restarting the epoch
// budget. This module stores the missing piece: WHERE in the SFT set the run
// was, written as a small sidecar file next to the checkpoint.
//
// Sidecar layout (`<model_path>.sftprog`, plain text, one `key value` per line):
//
//   version 1
//   seed <u64>        shuffle seed — the per-epoch order is seeded, so resuming
//                     reproduces the exact permutation the run was walking
//   epoch <usize>     1-based epoch the run was in
//   done <usize>      examples completed WITHIN that epoch
//   step <usize>      trainer step count, cross-checked against the checkpoint
//   examples <usize>  size of the SFT set when the run started
//
// The file is written together with every checkpoint save, so the pair stays
// consistent: the sidecar never describes a position past the saved weights.
// A missing or unreadable sidecar simply means "start from the beginning".

use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

/// Where an interrupted SFT run left off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SftProgress {
    /// Seed for the per-epoch shuffle, so resume walks the same permutation.
    pub seed: u64,
    /// 1-based epoch the run was in.
    pub epoch: usize,
    /// Examples already trained on within `epoch`.
    pub done: usize,
    /// Trainer step count at the time of writing.
    pub step: usize,
    /// Number of examples the run loaded, to detect a changed dataset.
    pub examples: usize,
}

impl SftProgress {
    /// A fresh run: epoch 1, nothing done, with a newly drawn shuffle seed.
    pub fn fresh(examples: usize, step: usize) -> Self {
        Self {
            seed: rand::random::<u64>(),
            epoch: 1,
            done: 0,
            step,
            examples,
        }
    }
}

/// Sidecar path for a checkpoint path.
pub fn progress_path(model_path: &str) -> PathBuf {
    PathBuf::from(format!("{model_path}.sftprog"))
}

/// Write the sidecar. Called right after a checkpoint save so the two agree.
pub fn save(model_path: &str, p: &SftProgress) -> io::Result<()> {
    let path = progress_path(model_path);
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).ok();
    }
    let mut f = fs::File::create(&path)?;
    writeln!(f, "version 1")?;
    writeln!(f, "seed {}", p.seed)?;
    writeln!(f, "epoch {}", p.epoch)?;
    writeln!(f, "done {}", p.done)?;
    writeln!(f, "step {}", p.step)?;
    writeln!(f, "examples {}", p.examples)?;
    Ok(())
}

/// Read the sidecar, or `None` if it is absent or malformed.
pub fn load(model_path: &str) -> Option<SftProgress> {
    let text = fs::read_to_string(progress_path(model_path)).ok()?;
    let mut seed = None;
    let mut epoch = None;
    let mut done = None;
    let mut step = None;
    let mut examples = None;
    let mut version = None;
    for line in text.lines() {
        let mut it = line.split_whitespace();
        let (Some(key), Some(val)) = (it.next(), it.next()) else {
            continue;
        };
        match key {
            "version" => version = val.parse::<u32>().ok(),
            "seed" => seed = val.parse::<u64>().ok(),
            "epoch" => epoch = val.parse::<usize>().ok(),
            "done" => done = val.parse::<usize>().ok(),
            "step" => step = val.parse::<usize>().ok(),
            "examples" => examples = val.parse::<usize>().ok(),
            _ => {}
        }
    }
    if version? != 1 {
        return None;
    }
    Some(SftProgress {
        seed: seed?,
        epoch: epoch?,
        done: done?,
        step: step?,
        examples: examples?,
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
/// sidecar, or a fresh start. `examples` is the size of the freshly loaded SFT
/// set and `step` the checkpoint's step count; both are cross-checked, and any
/// mismatch falls back to a fresh run with a printed reason (silently resuming
/// into a changed dataset would skip the wrong examples).
pub fn resume_or_fresh(
    model_path: &str,
    examples: usize,
    step: usize,
    epochs: usize,
) -> SftProgress {
    let Some(p) = load(model_path) else {
        return SftProgress::fresh(examples, step);
    };
    if p.examples != examples {
        println!(
            "SFT resume: dataset changed ({} examples recorded, {examples} loaded) \
             — starting a fresh run.",
            p.examples
        );
        return SftProgress::fresh(examples, step);
    }
    if p.step != step {
        println!(
            "SFT resume: checkpoint is at step {step} but progress file says {} \
             — starting a fresh run.",
            p.step
        );
        return SftProgress::fresh(examples, step);
    }
    if p.epoch > epochs {
        println!(
            "SFT resume: recorded epoch {} is past SFT_EPOCHS={epochs} — starting a fresh run.",
            p.epoch
        );
        return SftProgress::fresh(examples, step);
    }
    println!(
        "SFT resume: continuing epoch {} at example {}/{examples} (step {step}).",
        p.epoch, p.done
    );
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A written sidecar reads back identically.
    #[test]
    fn round_trips() {
        let dir = std::env::temp_dir().join("sftprog_round_trip");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model").to_str().unwrap().to_string();
        let p = SftProgress {
            seed: 0xDEAD_BEEF_1234_5678,
            epoch: 3,
            done: 421,
            step: 90_210,
            examples: 15_011,
        };
        save(&path, &p).unwrap();
        assert_eq!(load(&path), Some(p));
        clear(&path);
        assert_eq!(load(&path), None);
    }

    /// A changed dataset size must not resume into the middle of a different
    /// permutation — it falls back to a fresh run.
    #[test]
    fn dataset_change_forces_fresh() {
        let dir = std::env::temp_dir().join("sftprog_dataset_change");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model").to_str().unwrap().to_string();
        let p = SftProgress {
            seed: 7,
            epoch: 2,
            done: 10,
            step: 100,
            examples: 500,
        };
        save(&path, &p).unwrap();

        // Same size + step resumes exactly.
        assert_eq!(resume_or_fresh(&path, 500, 100, 3), p);
        // A different example count starts over.
        let fresh = resume_or_fresh(&path, 400, 100, 3);
        assert_eq!((fresh.epoch, fresh.done), (1, 0));
        // A step mismatch starts over.
        let fresh = resume_or_fresh(&path, 500, 999, 3);
        assert_eq!((fresh.epoch, fresh.done), (1, 0));
        clear(&path);
    }
}
