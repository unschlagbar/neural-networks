//! Host-side layout of the audio encoder: which feature rows the stem stacks,
//! which encoder frames form a word segment, and the boundary head's targets.
//!
//! Units: a *feature frame* is one [`LogMel`](super::frontend::LogMel) row
//! (10 ms); an *encoder frame* is one stem output (`stride` feature frames,
//! 40 ms). Segments and boundaries are in encoder frames.

use crate::config::{AUDIO_HOP, AUDIO_SR, AUDIO_STRIDE};

/// Seconds per encoder frame.
pub const ENC_FRAME_S: f32 = (AUDIO_HOP * AUDIO_STRIDE) as f32 / AUDIO_SR as f32;

/// How the stem turns feature frames into encoder frames: `stack` frames read
/// per output, `stride` frames advanced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StemLayout {
    pub stack: usize,
    pub stride: usize,
}

impl StemLayout {
    /// Encoder frames for `t_in` feature frames. A trailing partial stride is
    /// dropped rather than padded with the future.
    pub fn frames(self, t_in: usize) -> usize {
        t_in / self.stride
    }

    /// Feature-row ids to gather, `stack` per encoder frame, oldest first:
    /// encoder frame `j` of utterance `b` reads the rows ending at
    /// `j·stride + stride − 1`. Utterance `b`'s rows start at `b · t_in`; a row
    /// before its start reads `pad`, which the caller keeps zero.
    pub fn ids(self, batch: usize, t_in: usize, pad: u32) -> Vec<u32> {
        let t_out = self.frames(t_in);
        let mut ids = Vec::with_capacity(batch * t_out * self.stack);
        for b in 0..batch {
            for j in 0..t_out {
                let last = j * self.stride + self.stride - 1;
                for k in 0..self.stack {
                    let i = last as isize - (self.stack - 1 - k) as isize;
                    ids.push(if i < 0 { pad } else { (b * t_in) as u32 + i as u32 });
                }
            }
        }
        ids
    }
}

/// One spoken word's encoder frames, `[start, end)`. `end − 1` is the frame its
/// boundary is labelled on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment {
    pub start: usize,
    pub end: usize,
}

impl Segment {
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }

    pub fn last(&self) -> usize {
        self.end - 1
    }
}

/// The words of one utterance, in order, disjoint.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WordSegments(pub Vec<Segment>);

impl WordSegments {
    /// Segments for words given as `(start, end)` seconds in order, inside an
    /// utterance of `n_frames` encoder frames. A word owns the frames its
    /// interval touches; when fast speech puts two word ends into one frame, the
    /// later word moves on to the next frame, so segments never overlap and each
    /// is at least a frame long. Words starting past the end are dropped.
    pub fn from_word_times(words: &[(f32, f32)], n_frames: usize) -> Self {
        // A time exactly on a frame border must not round into the neighbour
        // through the division's last bit.
        const EPS: f32 = 1e-3;
        let mut out: Vec<Segment> = Vec::with_capacity(words.len());
        for &(s, e) in words {
            let prev_end = out.last().map_or(0, |p| p.end);
            let start = ((s / ENC_FRAME_S + EPS).floor().max(0.0) as usize).max(prev_end);
            if start >= n_frames {
                break;
            }
            let last = ((e / ENC_FRAME_S - EPS).ceil().max(0.0) as usize)
                .saturating_sub(1)
                .clamp(start, n_frames - 1);
            out.push(Segment {
                start,
                end: last + 1,
            });
        }
        Self(out)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Per segment `[ln(1 + frames), ln(1 + frames of silence before it)]`: the
    /// timing that pooling a word into one vector would otherwise lose —
    /// speaking rate, a drawn-out word, a hesitation.
    pub fn timing(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.0.len() * 2);
        let mut prev_end = 0;
        for s in &self.0 {
            out.push((1.0 + s.len() as f32).ln());
            out.push((1.0 + s.start.saturating_sub(prev_end) as f32).ln());
            prev_end = s.end;
        }
        out
    }
}

/// The encoder-frame rows the segment encoder runs over, for a batch whose
/// utterance `b` holds encoder frames `b · t_out ..`.
///
/// Each segment reads its own frames plus up to `delay` after its end: the
/// boundary head waits that long before it commits, so the segment encoder may
/// look as far ahead for free. Segments are laid end to end in one sequence and
/// the cells restart at each one.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SegmentRows {
    /// Encoder-frame row for every row of the segment sequence.
    pub ids: Vec<u32>,
    /// Rows of the sequence where a segment other than the first begins.
    pub resets: Vec<usize>,
    /// Row of each segment's last step, where its embedding is read.
    pub last: Vec<u32>,
}

impl SegmentRows {
    pub fn new(utterances: &[WordSegments], t_out: usize, delay: usize) -> Self {
        let mut rows = Self::default();
        for (b, segs) in utterances.iter().enumerate() {
            for s in &segs.0 {
                if !rows.ids.is_empty() {
                    rows.resets.push(rows.ids.len());
                }
                let stop = (s.last() + delay).min(t_out - 1);
                rows.ids
                    .extend((s.start..=stop).map(|f| (b * t_out + f) as u32));
                rows.last.push(rows.ids.len() as u32 - 1);
            }
        }
        rows
    }

    pub fn segments(&self) -> usize {
        self.last.len()
    }
}

/// Targets for the boundary head, `[n_frames, delay + 1]` row-major.
///
/// Column `d` at frame `t` asks "did a word end at frame `t − d`?", so the head
/// can commit early when the evidence is clear (a pause) and wait up to `delay`
/// frames when it is not. A frame within `tolerance` of a labelled end gets a
/// soft target that falls off linearly. `weights` is 0 where `t − d` precedes
/// the utterance.
pub struct BoundaryTargets {
    pub targets: Vec<f32>,
    pub weights: Vec<f32>,
}

impl BoundaryTargets {
    pub fn new(segs: &WordSegments, n_frames: usize, delay: usize, tolerance: usize) -> Self {
        let mut score = vec![0.0f32; n_frames];
        for s in &segs.0 {
            for off in 0..=tolerance {
                let v = 1.0 - off as f32 / (tolerance + 1) as f32;
                for f in [s.last().checked_sub(off), Some(s.last() + off)]
                    .into_iter()
                    .flatten()
                    .filter(|&f| f < n_frames)
                {
                    score[f] = score[f].max(v);
                }
            }
        }
        let cols = delay + 1;
        let mut targets = vec![0.0; n_frames * cols];
        let mut weights = vec![0.0; n_frames * cols];
        for t in 0..n_frames {
            for d in 0..=t.min(delay) {
                targets[t * cols + d] = score[t - d];
                weights[t * cols + d] = 1.0;
            }
        }
        Self { targets, weights }
    }

    /// The targets of several utterances, one after another.
    pub fn concat(parts: impl IntoIterator<Item = Self>) -> Self {
        let (mut targets, mut weights) = (Vec::new(), Vec::new());
        for p in parts {
            targets.extend(p.targets);
            weights.extend(p.weights);
        }
        Self { targets, weights }
    }
}

/// Weighted sigmoid cross-entropy, `pos_weight` on the positive term,
/// normalized by the total weight.
#[derive(Clone, Copy, Debug)]
pub struct Bce {
    pub pos_weight: f32,
}

impl Bce {
    /// The loss and `dL/dlogit`.
    pub fn loss_and_grad(self, logits: &[f32], targets: &BoundaryTargets) -> (f32, Vec<f32>) {
        let total: f32 = targets.weights.iter().sum::<f32>().max(1e-6);
        let pw = self.pos_weight;
        let mut loss = 0.0;
        let mut grad = vec![0.0; logits.len()];
        for (i, ((&z, &y), &w)) in logits
            .iter()
            .zip(&targets.targets)
            .zip(&targets.weights)
            .enumerate()
        {
            if w == 0.0 {
                continue;
            }
            // log σ(z) = −softplus(−z), log(1 − σ(z)) = −softplus(z).
            loss += w * (pw * y * softplus(-z) + (1.0 - y) * softplus(z));
            grad[i] = w * (sigmoid(z) * (pw * y + 1.0 - y) - pw * y) / total;
        }
        (loss / total, grad)
    }
}

/// Turns boundary probabilities `[n_frames, delay + 1]` into word ends the way
/// the streaming head commits to them.
#[derive(Clone, Copy, Debug)]
pub struct BoundaryDecoder {
    pub delay: usize,
    pub threshold: f32,
}

impl BoundaryDecoder {
    /// Frame `e` is an end as soon as some column `d` at frame `e + d` passes the
    /// threshold. Returns `(end frame, frame it was decided at)`. A decided end
    /// suppresses the frame after it, so one end is not split in two.
    pub fn decide(self, probs: &[f32], n_frames: usize) -> Vec<(usize, usize)> {
        let cols = self.delay + 1;
        let mut out: Vec<(usize, usize)> = Vec::new();
        for e in 0..n_frames {
            if out.last().is_some_and(|&(prev, _)| prev + 1 == e) {
                continue;
            }
            if let Some(d) = (0..=self.delay)
                .take_while(|d| e + d < n_frames)
                .find(|&d| probs[(e + d) * cols + d] > self.threshold)
            {
                out.push((e, e + d));
            }
        }
        out
    }
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: usize, end: usize) -> Segment {
        Segment { start, end }
    }

    #[test]
    fn stem_reads_the_frames_ending_at_each_stride() {
        // stack 3, stride 2, 6 input frames → 3 outputs reading rows ending at 1, 3, 5.
        let ids = StemLayout { stack: 3, stride: 2 }.ids(2, 6, 99);
        assert_eq!(ids.len(), 2 * 3 * 3);
        assert_eq!(&ids[..9], &[99, 0, 1, 1, 2, 3, 3, 4, 5]);
        assert_eq!(&ids[9..12], &[99, 6, 7]);
    }

    #[test]
    fn words_become_disjoint_segments() {
        let f = ENC_FRAME_S;
        let words = [
            (0.0, 2.5 * f),       // frames 0..=2
            (2.6 * f, 2.9 * f),   // also ends in frame 2: pushed to frame 3
            (6.0 * f, 8.0 * f),   // frames 6..=7
            (50.0 * f, 60.0 * f), // past the end
        ];
        let segs = WordSegments::from_word_times(&words, 10);
        assert_eq!(segs.0, vec![seg(0, 3), seg(3, 4), seg(6, 8)]);
        let timing = segs.timing();
        assert!((timing[5] - 2f32.ln_1p()).abs() < 1e-6, "pause before the third word");
    }

    #[test]
    fn segment_rows_look_ahead_and_restart() {
        let utts = vec![
            WordSegments(vec![seg(0, 2), seg(2, 5)]),
            WordSegments(vec![seg(1, 3)]),
        ];
        let rows = SegmentRows::new(&utts, 6, 2);
        // utt 0: seg 0 reads 0..=3, seg 1 reads 2..=5 (clamped at t_out − 1);
        // utt 1: seg reads 1..=4, offset by 6.
        assert_eq!(rows.ids, vec![0, 1, 2, 3, 2, 3, 4, 5, 7, 8, 9, 10]);
        assert_eq!(rows.resets, vec![4, 8]);
        assert_eq!(rows.last, vec![3, 7, 11]);
    }

    #[test]
    fn boundary_targets_are_delayed_and_soft() {
        let bt = BoundaryTargets::new(&WordSegments(vec![seg(0, 4)]), 8, 2, 1); // ends at 3
        let at = |t: usize, d: usize| bt.targets[t * 3 + d];
        assert_eq!(at(3, 0), 1.0);
        assert_eq!(at(5, 2), 1.0);
        assert_eq!(at(4, 0), 0.5);
        assert_eq!(at(6, 2), 0.5);
        assert_eq!(at(7, 2), 0.0);
        assert_eq!(bt.weights[3 + 2], 0.0, "frame 1 − 2 precedes the utterance");
    }

    #[test]
    fn bce_gradient_matches_finite_differences() {
        let z = [0.3, -1.2, 2.0, 0.0];
        let targets = BoundaryTargets {
            targets: vec![1.0, 0.0, 0.5, 1.0],
            weights: vec![1.0, 1.0, 1.0, 0.0],
        };
        let bce = Bce { pos_weight: 3.0 };
        let (_, g) = bce.loss_and_grad(&z, &targets);
        for i in 0..3 {
            let (mut zp, mut zm) = (z, z);
            zp[i] += 1e-3;
            zm[i] -= 1e-3;
            let fd = (bce.loss_and_grad(&zp, &targets).0 - bce.loss_and_grad(&zm, &targets).0) / 2e-3;
            assert!((fd - g[i]).abs() < 1e-3, "logit {i}: fd {fd} vs {}", g[i]);
        }
        assert_eq!(g[3], 0.0);
    }

    #[test]
    fn boundaries_are_decided_as_early_as_the_evidence_allows() {
        // delay 2: frame 2 is clear at once, frame 6 only two frames later; frame 3
        // repeats frame 2's end and is suppressed.
        let mut p = vec![0.0; 10 * 3];
        p[2 * 3] = 0.9;
        p[3 * 3] = 0.9;
        p[8 * 3 + 2] = 0.9;
        let dec = BoundaryDecoder {
            delay: 2,
            threshold: 0.5,
        };
        assert_eq!(dec.decide(&p, 10), vec![(2, 2), (6, 8)]);
    }
}
