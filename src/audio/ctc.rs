//! Connectionist temporal classification (Graves et al., 2006) over characters,
//! for the audio encoder's character head: the loss and its gradient, greedy
//! decoding and the character error rate. Host-side; per utterance.

/// Lower-case letters, apostrophe and space, after the blank at id 0 — the
/// whole of what LibriSpeech transcripts contain.
pub struct CtcAlphabet;

impl CtcAlphabet {
    pub const BLANK: u32 = 0;
    pub const SPACE: u32 = 1;
    pub const SIZE: usize = 29;

    /// Ids of `text`, lower-cased; `None` if it holds anything else.
    pub fn encode(text: &str) -> Option<Vec<u32>> {
        text.chars()
            .map(|c| match c.to_ascii_lowercase() {
                ' ' => Some(1),
                '\'' => Some(2),
                c @ 'a'..='z' => Some(3 + (c as u32 - 'a' as u32)),
                _ => None,
            })
            .collect()
    }

    /// Position in `ids` of every word's last character.
    pub fn word_ends(ids: &[u32]) -> Vec<usize> {
        (0..ids.len())
            .filter(|&i| ids[i] != Self::SPACE && ids.get(i + 1).is_none_or(|&n| n == Self::SPACE))
            .collect()
    }

    pub fn decode(ids: &[u32]) -> String {
        ids.iter()
            .map(|&i| match i {
                1 => ' ',
                2 => '\'',
                i @ 3..=28 => (b'a' + (i - 3) as u8) as char,
                _ => '·',
            })
            .collect()
    }
}

/// CTC over `[t, v]` row-major logits.
#[derive(Clone, Copy, Debug)]
pub struct Ctc {
    pub vocab: usize,
}

impl Ctc {
    /// `−log p(target | logits)` and its gradient with respect to the logits;
    /// `None` when the target cannot fit in `t` steps (each repeated label needs
    /// a blank between).
    pub fn loss_and_grad(self, logits: &[f32], target: &[u32]) -> Option<(f32, Vec<f32>)> {
        let v = self.vocab;
        let t = logits.len() / v;
        let repeats = target.windows(2).filter(|w| w[0] == w[1]).count();
        if target.is_empty() || t < target.len() + repeats {
            return None;
        }
        let logp = log_softmax(logits, v);
        // The extended label sequence: blank, l1, blank, l2, …, blank.
        let s = 2 * target.len() + 1;
        let label = |i: usize| {
            if i % 2 == 0 {
                Ctc::BLANK
            } else {
                target[i / 2]
            }
        };
        // A step may skip the blank between two different labels.
        let skip = |i: usize| i >= 2 && i % 2 == 1 && label(i) != label(i - 2);

        let mut alpha = vec![f32::NEG_INFINITY; t * s];
        alpha[0] = logp[Ctc::BLANK as usize];
        alpha[1] = logp[label(1) as usize];
        for ti in 1..t {
            for i in 0..s {
                let prev = &alpha[(ti - 1) * s..ti * s];
                let mut a = prev[i];
                if i >= 1 {
                    a = log_add(a, prev[i - 1]);
                }
                if skip(i) {
                    a = log_add(a, prev[i - 2]);
                }
                alpha[ti * s + i] = a + logp[ti * v + label(i) as usize];
            }
        }
        let mut beta = vec![f32::NEG_INFINITY; t * s];
        beta[(t - 1) * s + s - 1] = logp[(t - 1) * v + Ctc::BLANK as usize];
        beta[(t - 1) * s + s - 2] = logp[(t - 1) * v + label(s - 2) as usize];
        for ti in (0..t - 1).rev() {
            for i in 0..s {
                let next = &beta[(ti + 1) * s..(ti + 2) * s];
                let mut b = next[i];
                if i + 1 < s {
                    b = log_add(b, next[i + 1]);
                }
                if i + 2 < s && skip(i + 2) {
                    b = log_add(b, next[i + 2]);
                }
                beta[ti * s + i] = b + logp[ti * v + label(i) as usize];
            }
        }
        let log_p = log_add(alpha[(t - 1) * s + s - 1], alpha[(t - 1) * s + s - 2]);
        if !log_p.is_finite() {
            return None;
        }

        // dL/dz = softmax − (Σ over the label's positions of αβ / p(label)) / p.
        // αβ counts the step's own emission twice, hence the − logp.
        let mut grad: Vec<f32> = logp.iter().map(|l| l.exp()).collect();
        let mut occ = vec![f32::NEG_INFINITY; v];
        for ti in 0..t {
            occ.fill(f32::NEG_INFINITY);
            for i in 0..s {
                let k = label(i) as usize;
                occ[k] = log_add(occ[k], alpha[ti * s + i] + beta[ti * s + i]);
            }
            for k in 0..v {
                if occ[k].is_finite() {
                    grad[ti * v + k] -= (occ[k] - logp[ti * v + k] - log_p).exp();
                }
            }
        }
        Some((-log_p, grad))
    }

    /// Forced alignment: the single most likely path that spells `target`
    /// (Viterbi over the same lattice as the loss). Per label, the last step it
    /// is emitted on; `None` when the target cannot fit.
    pub fn align(self, logits: &[f32], target: &[u32]) -> Option<Vec<usize>> {
        let v = self.vocab;
        let t = logits.len() / v;
        let repeats = target.windows(2).filter(|w| w[0] == w[1]).count();
        if target.is_empty() || t < target.len() + repeats {
            return None;
        }
        let logp = log_softmax(logits, v);
        let s = 2 * target.len() + 1;
        let label = |i: usize| {
            if i % 2 == 0 {
                Ctc::BLANK
            } else {
                target[i / 2]
            }
        };
        let skip = |i: usize| i >= 2 && i % 2 == 1 && label(i) != label(i - 2);

        let mut score = vec![f32::NEG_INFINITY; s];
        let mut next = vec![f32::NEG_INFINITY; s];
        // back[ti·s + i]: how far the path moved to reach state i at step ti.
        let mut back = vec![0u8; t * s];
        score[0] = logp[Ctc::BLANK as usize];
        score[1] = logp[label(1) as usize];
        for ti in 1..t {
            for i in 0..s {
                let mut best = (score[i], 0u8);
                if i >= 1 && score[i - 1] > best.0 {
                    best = (score[i - 1], 1);
                }
                if skip(i) && score[i - 2] > best.0 {
                    best = (score[i - 2], 2);
                }
                next[i] = best.0 + logp[ti * v + label(i) as usize];
                back[ti * s + i] = best.1;
            }
            std::mem::swap(&mut score, &mut next);
        }
        let mut i = if score[s - 1] >= score[s - 2] {
            s - 1
        } else {
            s - 2
        };
        if !score[i].is_finite() {
            return None;
        }
        let mut last = vec![usize::MAX; target.len()];
        for ti in (0..t).rev() {
            if i % 2 == 1 && last[i / 2] == usize::MAX {
                last[i / 2] = ti;
            }
            i -= back[ti * s + i] as usize;
        }
        Some(last)
    }

    /// Best path: the most likely label per step, repeats merged, blanks dropped.
    pub fn greedy(self, logits: &[f32]) -> Vec<u32> {
        let mut out = Vec::new();
        let mut prev = Ctc::BLANK;
        for row in logits.chunks_exact(self.vocab) {
            let k = (0..self.vocab)
                .max_by(|&a, &b| row[a].total_cmp(&row[b]))
                .unwrap() as u32;
            if k != prev && k != Ctc::BLANK {
                out.push(k);
            }
            prev = k;
        }
        out
    }

    const BLANK: u32 = CtcAlphabet::BLANK;
}

/// Levenshtein distance: the edits a character error rate counts.
pub fn edit_distance(a: &[u32], b: &[u32]) -> usize {
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, &x) in a.iter().enumerate() {
        let mut diag = row[0];
        row[0] = i + 1;
        for (j, &y) in b.iter().enumerate() {
            let up = row[j + 1];
            row[j + 1] = (up + 1).min(row[j] + 1).min(diag + usize::from(x != y));
            diag = up;
        }
    }
    row[b.len()]
}

fn log_softmax(logits: &[f32], v: usize) -> Vec<f32> {
    let mut out = vec![0.0; logits.len()];
    for (row, o) in logits.chunks_exact(v).zip(out.chunks_exact_mut(v)) {
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse = m + row.iter().map(|&z| (z - m).exp()).sum::<f32>().ln();
        for (o, &z) in o.iter_mut().zip(row) {
            *o = z - lse;
        }
    }
    out
}

fn log_add(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let (hi, lo) = if a > b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alphabet_round_trips() {
        let ids = CtcAlphabet::encode("I'm well").unwrap();
        assert_eq!(CtcAlphabet::decode(&ids), "i'm well");
        assert_eq!(CtcAlphabet::word_ends(&ids), vec![2, 7]);
        assert_eq!(CtcAlphabet::encode("$5"), None);
    }

    #[test]
    fn a_forced_path_costs_its_emissions() {
        // t == target length with no repeats: one path, the labels themselves.
        let ctc = Ctc { vocab: 4 };
        let logits = [0.5, 2.0, -1.0, 0.0, 1.0, -0.5, 0.3, 2.5];
        let (loss, _) = ctc.loss_and_grad(&logits, &[1, 3]).unwrap();
        let lsm = log_softmax(&logits, 4);
        assert!((loss + lsm[1] + lsm[4 + 3]).abs() < 1e-5, "{loss}");
        // A repeat needs a blank between, which two steps do not have.
        assert!(ctc.loss_and_grad(&logits, &[2, 2]).is_none());
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let ctc = Ctc { vocab: 5 };
        let logits: Vec<f32> = (0..7 * 5)
            .map(|i| ((i * 37 % 11) as f32 - 5.0) * 0.3)
            .collect();
        let target = [1, 3, 3, 2];
        let (_, g) = ctc.loss_and_grad(&logits, &target).unwrap();
        for i in 0..logits.len() {
            let (mut p, mut m) = (logits.clone(), logits.clone());
            p[i] += 1e-2;
            m[i] -= 1e-2;
            let fd = (ctc.loss_and_grad(&p, &target).unwrap().0
                - ctc.loss_and_grad(&m, &target).unwrap().0)
                / 2e-2;
            assert!((fd - g[i]).abs() < 2e-3, "logit {i}: fd {fd} vs {}", g[i]);
        }
    }

    #[test]
    fn forced_alignment_follows_the_peaks() {
        let ctc = Ctc { vocab: 3 };
        // Label 1 peaks at steps 1-2, label 2 at step 4, a blank between.
        let mut logits = vec![0.0; 6 * 3];
        for (t, k) in [(0, 0), (1, 1), (2, 1), (3, 0), (4, 2), (5, 0)] {
            logits[t * 3 + k] = 5.0;
        }
        assert_eq!(ctc.align(&logits, &[1, 2]), Some(vec![2, 4]));
        // A repeated label must close on a blank before it starts again.
        assert_eq!(ctc.align(&logits, &[1, 1]).map(|l| l[0] < l[1]), Some(true));
        assert_eq!(ctc.align(&logits[..2 * 3], &[1, 1, 2]), None);
    }

    #[test]
    fn greedy_merges_repeats_and_drops_blanks() {
        let ctc = Ctc { vocab: 3 };
        // argmax per step: 1 1 0 1 2 2 0
        let rows = [
            [0., 1., 0.],
            [0., 1., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 0., 1.],
            [1., 0., 0.],
        ];
        let logits: Vec<f32> = rows.concat();
        assert_eq!(ctc.greedy(&logits), vec![1, 1, 2]);
    }

    #[test]
    fn edit_distance_counts_edits() {
        assert_eq!(edit_distance(&[1, 2, 3], &[1, 3]), 1);
        assert_eq!(edit_distance(&[], &[4, 4]), 2);
        assert_eq!(edit_distance(&[1, 2], &[2, 1]), 2);
    }
}
