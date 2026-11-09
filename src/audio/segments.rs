//! Host-side layout of the audio encoder: which feature rows the stem stacks,
//! which encoder frames form a word segment, and the boundary head's targets.

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
                    ids.push(if i < 0 {
                        pad
                    } else {
                        (b * t_in) as u32 + i as u32
                    });
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
    /// utterance of `n_frames` encoder frames. A word ends on the last frame that
    /// lies entirely before its end time: a frame reaching past it would carry
    /// the onset of the next word into this word's embedding, which is the very
    /// thing the text model is asked to predict. A word shorter than that still
    /// gets its first frame. Segments never overlap — a word starting inside the
    /// previous one's last frame moves on to the next — and words starting past
    /// the end are dropped.
    pub fn from_word_times(words: &[(f32, f32)], n_frames: usize) -> Self {
        const EPS: f32 = 1e-3;
        let mut out: Vec<Segment> = Vec::with_capacity(words.len());
        for &(s, e) in words {
            let prev_end = out.last().map_or(0, |p| p.end);
            let start = ((s / ENC_FRAME_S + EPS).floor().max(0.0) as usize).max(prev_end);
            if start >= n_frames {
                break;
            }
            let last = ((e / ENC_FRAME_S + EPS).floor().max(0.0) as usize)
                .saturating_sub(1)
                .clamp(start, n_frames - 1);
            out.push(Segment {
                start,
                end: last + 1,
            });
        }
        Self(out)
    }

    /// Segments for words ending on frames `ends` (increasing), each starting
    /// right after the previous one: with only ends known, a pause is part of
    /// the word after it.
    pub fn from_ends(ends: &[usize]) -> Self {
        let mut start = 0;
        Self(
            ends.iter()
                .map(|&e| {
                    let s = Segment { start, end: e + 1 };
                    start = e + 1;
                    s
                })
                .collect(),
        )
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
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SegmentRows {
    /// Frame row of every step, `[segments, steps]` row-major; `[W]` and the
    /// padding read row `pad`, one past the last encoder frame.
    pub ids: Vec<u32>,
    pub steps: usize,
    /// Step of each segment where its embedding is read, as a flat row index.
    pub last: Vec<u32>,
}

impl SegmentRows {
    pub fn new(utterances: &[WordSegments], t_out: usize, max_frames: usize, word: bool) -> Self {
        let pad = (utterances.len() * t_out) as u32;
        let kept = |s: &Segment| s.len().min(max_frames);
        let steps = utterances
            .iter()
            .flat_map(|u| &u.0)
            .map(kept)
            .max()
            .map_or(0, |k| k + word as usize);
        let mut rows = Self {
            steps,
            ..Self::default()
        };
        for (b, segs) in utterances.iter().enumerate() {
            for s in &segs.0 {
                let row = rows.ids.len();
                let k = kept(s);
                assert!(k > 0, "SegmentRows: an empty segment");
                rows.ids
                    .extend((s.end - k..s.end).map(|f| (b * t_out + f) as u32));
                rows.ids.resize(row + steps, pad);
                rows.last.push((row + k - 1 + word as usize) as u32);
            }
        }
        rows
    }

    /// Rows and steps of the largest batch of `words` segments over `t_out`
    /// frames.
    pub fn bound(words: usize, t_out: usize, max_frames: usize, word: bool) -> (usize, usize) {
        let steps = t_out.min(max_frames) + word as usize;
        (words * steps, steps)
    }

    pub fn segments(&self) -> usize {
        self.last.len()
    }
}

/// Targets for the boundary head, `[n_frames, delay + 1]` row-major.
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

/// The boundary probabilities of an utterance must add up to what its word
/// count asks for: CIF's quantity loss (Dong & Xu, arXiv 1905.11235), which
/// needs the transcript but no timing. On column 0, the only one that covers an
/// utterance's last frames.
#[derive(Clone, Copy, Debug)]
pub struct CountLoss {
    /// As in `BoundaryTargets`: an end lights its frame and this many soft
    /// neighbours on each side.
    pub tolerance: usize,
    pub weight: f32,
}

impl CountLoss {
    /// Target probability mass per word.
    pub fn per_word(self) -> f32 {
        let t = self.tolerance;
        1.0 + 2.0 * (1..=t).map(|o| 1.0 - o as f32 / (t + 1) as f32).sum::<f32>()
    }

    /// The relative count error over the first `n_frames` rows of `logits`
    /// (`[.., cols]`) against `words`; adds `scale` times the gradient of
    /// `weight · error²` into `grad`.
    pub fn add(
        self,
        logits: &[f32],
        cols: usize,
        n_frames: usize,
        words: usize,
        scale: f32,
        grad: &mut [f32],
    ) -> f32 {
        let target = words as f32 * self.per_word();
        let mass: f32 = (0..n_frames).map(|t| sigmoid(logits[t * cols])).sum();
        let rel = (mass - target) / target;
        let g = self.weight * scale * 2.0 * rel / target;
        for t in 0..n_frames {
            let p = sigmoid(logits[t * cols]);
            grad[t * cols] += g * p * (1.0 - p);
        }
        rel
    }
}

/// Streaming word ends from boundary logits `[n_frames, delay + 1]`, one per
/// peak: the head's targets are soft, so a single end lights up its neighbours
/// too. Frame `e` is decided at `e + delay`, once every column has seen it.
#[derive(Clone, Copy, Debug)]
pub struct PeakDecoder {
    pub delay: usize,
    pub threshold: f32,
    /// Frames on each side a peak must top; at most `delay`, which is all the
    /// lookahead there is.
    pub radius: usize,
}

impl PeakDecoder {
    pub fn decide(self, logits: &[f32], n_frames: usize) -> Vec<usize> {
        let scores = CountAligner {
            delay: self.delay,
            from_col: 0,
            off_peak: 0.0,
        }
        .end_scores(logits, n_frames);
        let threshold = (self.threshold / (1.0 - self.threshold)).ln();
        let mut out: Vec<usize> = Vec::new();
        for e in 0..n_frames {
            let lo = e.saturating_sub(self.radius);
            let hi = (e + self.radius).min(n_frames - 1);
            // Ties go to the earliest frame.
            let peak = (lo..e).all(|i| scores[i] < scores[e])
                && (e + 1..=hi).all(|i| scores[i] <= scores[e]);
            if scores[e] > threshold && peak && out.last().is_none_or(|&p| e > p + self.radius) {
                out.push(e);
            }
        }
        out
    }
}

/// Log-normal prior on how many frames a word of a given length takes.
#[derive(Clone, Copy, Debug)]
pub struct DurationPrior {
    /// Spread of `ln(frames)` around the expectation.
    pub sigma: f32,
    /// Scale against the boundary log-odds.
    pub weight: f32,
    /// Longest word, pause before it included, in frames.
    pub max_frames: usize,
}

impl DurationPrior {
    fn log_prob(self, frames: usize, expected_ln: f32) -> f32 {
        if frames == 0 {
            return f32::NEG_INFINITY;
        }
        let z = ((frames as f32).ln() - expected_ln) / self.sigma;
        -0.5 * self.weight * z * z
    }
}

/// Places a known number of word ends on the boundary head's output: the
/// transcript says how many words were spoken, so alignment is choosing that
/// many frames, not deciding whether each frame is one. Offline only — it reads
/// the whole utterance.
#[derive(Clone, Copy, Debug)]
pub struct CountAligner {
    pub delay: usize,
    /// Columns `from_col..=delay` are averaged into a frame's end score: a column
    /// is the head's opinion after `d` frames of lookahead.
    pub from_col: usize,
    /// Frames a score that is not a local maximum loses: the head's targets are
    /// soft, so one end lights up its neighbours too, and without this the
    /// alignment spends two words on one strong end instead of one on a weak one.
    pub off_peak: f32,
}

impl CountAligner {
    /// Log-odds that a word ends at each of `n_frames` frames, from boundary
    /// logits `[n_frames, delay + 1]`. Near the end, where the later columns
    /// fall past the utterance, the latest one available is used.
    pub fn end_scores(self, logits: &[f32], n_frames: usize) -> Vec<f32> {
        let cols = self.delay + 1;
        (0..n_frames)
            .map(|e| {
                let last = self.delay.min(n_frames - 1 - e);
                let first = self.from_col.min(last);
                let sum: f32 = (first..=last).map(|d| logits[(e + d) * cols + d]).sum();
                sum / (last - first + 1) as f32
            })
            .collect()
    }

    /// `scores` with every frame below a neighbour lowered by `off_peak`.
    pub fn peaks(self, scores: &[f32]) -> Vec<f32> {
        let n = scores.len();
        (0..n)
            .map(|i| {
                let left = i > 0 && scores[i - 1] > scores[i];
                let right = i + 1 < n && scores[i + 1] > scores[i];
                if left || right {
                    scores[i] - self.off_peak
                } else {
                    scores[i]
                }
            })
            .collect()
    }

    /// Word ends for words of `lens` units (bytes) from boundary logits
    /// `[n_frames, delay + 1]`: peak-picked scores plus the duration prior,
    /// whose speaking rate is first guessed from the whole utterance and then
    /// re-measured on that first alignment.
    pub fn align_words(
        self,
        logits: &[f32],
        n_frames: usize,
        lens: &[usize],
        prior: DurationPrior,
    ) -> Option<Vec<usize>> {
        let scores = self.peaks(&self.end_scores(logits, n_frames));
        let units: usize = lens.iter().sum();
        let mut rate = n_frames as f32 / units.max(1) as f32;
        let mut ends = None;
        for _ in 0..2 {
            ends = Some(self.align_durations(&scores, lens, prior, rate)?);
            let e = ends.as_ref().unwrap();
            if e.len() >= 2 {
                rate = (e[e.len() - 1] - e[0]) as f32 / units.saturating_sub(lens[0]).max(1) as f32;
            }
        }
        ends
    }

    /// [`align`](Self::align) with `prior` scoring each word's frames since the
    /// previous end against `rate · len` frames. The first word has no previous
    /// end and no prior. `O(frames · words · max_frames)`.
    pub fn align_durations(
        self,
        scores: &[f32],
        lens: &[usize],
        prior: DurationPrior,
        rate: f32,
    ) -> Option<Vec<usize>> {
        let (t, n) = (scores.len(), lens.len());
        if n > t {
            return None;
        }
        if n == 0 {
            return Some(Vec::new());
        }
        let mut back = vec![0u32; n * t];
        let mut dp: Vec<f32> = scores.to_vec();
        let mut next = vec![f32::NEG_INFINITY; t];
        for k in 1..n {
            let expected = (rate * lens[k] as f32).max(1.0).ln();
            let cost: Vec<f32> = (0..=prior.max_frames)
                .map(|d| prior.log_prob(d, expected))
                .collect();
            next.fill(f32::NEG_INFINITY);
            for i in k..t {
                let (mut best, mut arg) = (f32::NEG_INFINITY, i - 1);
                for d in 1..=prior.max_frames.min(i) {
                    let v = dp[i - d] + cost[d];
                    if v > best {
                        best = v;
                        arg = i - d;
                    }
                }
                next[i] = best + scores[i];
                back[k * t + i] = arg as u32;
            }
            std::mem::swap(&mut dp, &mut next);
        }
        let mut end = (n - 1..t)
            .filter(|&i| dp[i].is_finite())
            .max_by(|&a, &b| dp[a].total_cmp(&dp[b]))?;
        let mut ends = vec![0; n];
        for k in (0..n).rev() {
            ends[k] = end;
            end = back[k * t + end] as usize;
        }
        Some(ends)
    }

    /// The `n_words` strictly increasing end frames with the highest total
    /// score, or `None` when there are more words than frames. Every frame not
    /// chosen contributes `log(1 − p)` either way, so maximizing the sum of the
    /// chosen log-odds maximizes the likelihood of the whole labelling.
    pub fn align(self, scores: &[f32], n_words: usize) -> Option<Vec<usize>> {
        let t = scores.len();
        if n_words > t {
            return None;
        }
        if n_words == 0 {
            return Some(Vec::new());
        }
        // back[k·t + i]: end of word k − 1 on the best path with word k ending at i.
        let mut back = vec![0u32; n_words * t];
        let mut dp: Vec<f32> = scores.to_vec();
        let mut next = vec![f32::NEG_INFINITY; t];
        for k in 1..n_words {
            let (mut best, mut arg) = (f32::NEG_INFINITY, 0);
            next.fill(f32::NEG_INFINITY);
            for i in k..t {
                if dp[i - 1] > best {
                    best = dp[i - 1];
                    arg = i - 1;
                }
                next[i] = best + scores[i];
                back[k * t + i] = arg as u32;
            }
            std::mem::swap(&mut dp, &mut next);
        }
        let mut end = (n_words - 1..t).max_by(|&a, &b| dp[a].total_cmp(&dp[b]))?;
        let mut ends = vec![0; n_words];
        for k in (0..n_words).rev() {
            ends[k] = end;
            end = back[k * t + end] as usize;
        }
        Some(ends)
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
        let ids = StemLayout {
            stack: 3,
            stride: 2,
        }
        .ids(2, 6, 99);
        assert_eq!(ids.len(), 2 * 3 * 3);
        assert_eq!(&ids[..9], &[99, 0, 1, 1, 2, 3, 3, 4, 5]);
        assert_eq!(&ids[9..12], &[99, 6, 7]);
    }

    #[test]
    fn words_become_disjoint_segments() {
        let f = ENC_FRAME_S;
        let words = [
            (0.0, 2.5 * f),       // frame 2 reaches past the end: frames 0..=1
            (2.6 * f, 2.9 * f),   // shorter than a frame: just its first, frame 2
            (6.0 * f, 8.0 * f),   // ends exactly on a border: frames 6..=7
            (50.0 * f, 60.0 * f), // past the end
        ];
        let segs = WordSegments::from_word_times(&words, 10);
        assert_eq!(segs.0, vec![seg(0, 2), seg(2, 3), seg(6, 8)]);
        let timing = segs.timing();
        assert!(
            (timing[5] - 3f32.ln_1p()).abs() < 1e-6,
            "pause before the third word"
        );
    }

    #[test]
    fn segment_rows_stop_at_the_word_end_and_pad() {
        let utts = vec![
            WordSegments(vec![seg(0, 2), seg(2, 5)]),
            WordSegments(vec![seg(1, 3)]),
        ];
        // utt 1's frames start at 6; row 12 is the pad.
        let rows = SegmentRows::new(&utts, 6, 8, true);
        assert_eq!(rows.steps, 4);
        assert_eq!(rows.ids, vec![0, 1, 12, 12, 2, 3, 4, 12, 7, 8, 12, 12]);
        assert_eq!(rows.last, vec![2, 7, 10]);

        let rows = SegmentRows::new(&utts, 6, 2, false);
        assert_eq!(rows.steps, 2);
        assert_eq!(rows.ids, vec![0, 1, 3, 4, 7, 8], "the last two frames of a long word");
        assert_eq!(rows.last, vec![1, 3, 5]);
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
            let fd =
                (bce.loss_and_grad(&zp, &targets).0 - bce.loss_and_grad(&zm, &targets).0) / 2e-3;
            assert!((fd - g[i]).abs() < 1e-3, "logit {i}: fd {fd} vs {}", g[i]);
        }
        assert_eq!(g[3], 0.0);
    }

    #[test]
    fn count_aligner_places_exactly_the_given_number_of_ends() {
        let aligner = CountAligner {
            delay: 0,
            from_col: 0,
            off_peak: 0.0,
        };
        let scores = [-5.0, 3.0, -1.0, -4.0, 2.0, 1.5, -6.0];
        assert_eq!(aligner.align(&scores, 2), Some(vec![1, 4]));
        // A third word takes the best remaining frame after the second.
        assert_eq!(aligner.align(&scores, 3), Some(vec![1, 4, 5]));
        // Order is kept even where a later frame scores higher.
        assert_eq!(
            aligner.align(&[1.0, -9.0, -9.0, 5.0], 4),
            Some(vec![0, 1, 2, 3])
        );
        assert_eq!(aligner.align(&scores, 8), None);
    }

    #[test]
    fn peak_decoder_takes_one_end_per_cluster() {
        let dec = PeakDecoder {
            delay: 0,
            threshold: 0.5,
            radius: 1,
        };
        // A soft cluster around frame 2, a lone end at 6, a sub-threshold peak at 8.
        let logits = [-5.0, 1.0, 2.0, 1.0, -5.0, -5.0, 3.0, -5.0, -0.5, -5.0];
        assert_eq!(dec.decide(&logits, 10), vec![2, 6]);
    }

    #[test]
    fn duration_prior_spaces_ends_by_word_length() {
        let aligner = CountAligner {
            delay: 0,
            from_col: 0,
            off_peak: 0.0,
        };
        let prior = DurationPrior {
            sigma: 0.5,
            weight: 1.0,
            max_frames: 20,
        };
        let ends = aligner
            .align_durations(&[0.0; 12], &[2, 2, 6], prior, 1.0)
            .unwrap();
        assert_eq!([ends[1] - ends[0], ends[2] - ends[1]], [2, 6]);
    }

    #[test]
    fn end_scores_average_the_delayed_columns() {
        // delay 1: frame e is read at (e, 0) and (e + 1, 1).
        let logits = [1.0, 0.0, 3.0, 5.0, 7.0, 9.0];
        let all = CountAligner {
            delay: 1,
            from_col: 0,
            off_peak: 0.0,
        }
        .end_scores(&logits, 3);
        assert_eq!(all, vec![3.0, 6.0, 7.0]);
        let latest = CountAligner {
            delay: 1,
            from_col: 1,
            off_peak: 0.0,
        }
        .end_scores(&logits, 3);
        assert_eq!(latest, vec![5.0, 9.0, 7.0]);
    }

    #[test]
    fn count_loss_matches_targets_and_its_gradient() {
        let segs = WordSegments::from_ends(&[5, 14, 30]);
        let (n, delay, tol) = (40, 2, 1);
        let bt = BoundaryTargets::new(&segs, n, delay, tol);
        let count = CountLoss { tolerance: tol, weight: 1.0 };
        let cols = delay + 1;
        let mass: f32 = (0..n).map(|t| bt.targets[t * cols]).sum();
        assert!((mass - 3.0 * count.per_word()).abs() < 1e-5, "targets carry {mass}");

        let logits: Vec<f32> = (0..n * cols).map(|i| ((i * 37) % 11) as f32 * 0.3 - 2.0).collect();
        let mut grad = vec![0.0; logits.len()];
        let rel = count.add(&logits, cols, n, 3, 1.0, &mut grad);
        let loss = |l: &[f32]| count.add(l, cols, n, 3, 0.0, &mut vec![0.0; l.len()]).powi(2);
        let (i, h) = (7 * cols, 1e-3);
        let mut up = logits.clone();
        up[i] += h;
        let mut down = logits.clone();
        down[i] -= h;
        let fd = (loss(&up) - loss(&down)) / (2.0 * h);
        assert!((fd - grad[i]).abs() < 1e-2 * fd.abs().max(1e-3), "fd {fd} vs {}", grad[i]);
        assert!(rel != 0.0);
        assert_eq!(grad[7 * cols + 1], 0.0, "only column 0");
    }
}
