//! The audio encoder's features at 100 Hz: 128 Slaney mel bins (as Whisper and
//! librosa build them) plus the frame's absolute log energy.
//!
//! Framing is causal — frame `i` ends at sample `(i + 1) · AUDIO_HOP` — so a
//! streaming caller and a whole-utterance call produce the same frames. Nothing
//! is normalized per utterance or per frame: loudness and dynamics are part of
//! what the encoder is meant to hear. The only normalization is
//! [`FeatureStats`], fixed over a corpus.

use super::mel::{MelFilterBank, log_total_energy};
use super::spectrum::{Spectrogram, Window};
use crate::config::{AUDIO_FEATS, AUDIO_HOP, AUDIO_N_FFT, AUDIO_N_MELS, AUDIO_SR, AUDIO_WIN};

pub struct LogMel {
    spectrum: Spectrogram,
    bank: MelFilterBank,
}

impl Default for LogMel {
    fn default() -> Self {
        Self::new()
    }
}

impl LogMel {
    pub fn new() -> Self {
        Self {
            spectrum: Spectrogram::new(AUDIO_WIN, AUDIO_N_FFT, Window::Hann),
            bank: MelFilterBank::slaney(AUDIO_N_MELS, AUDIO_SR, AUDIO_N_FFT, 0.0, AUDIO_SR as f32 / 2.0),
        }
    }

    /// Features of one `AUDIO_WIN`-sample window into `out` (`AUDIO_FEATS` long):
    /// the log mel energies, then the log energy of the whole spectrum.
    pub fn frame(&mut self, samples: &[f32], out: &mut [f32]) {
        assert_eq!(out.len(), AUDIO_FEATS, "LogMel::frame writes AUDIO_FEATS");
        let power = self.spectrum.frame(samples);
        self.bank.log_energies(power, &mut out[..AUDIO_N_MELS]);
        out[AUDIO_N_MELS] = log_total_energy(power);
    }

    /// Every complete frame of `audio`, `[audio.len() / AUDIO_HOP, AUDIO_FEATS]`
    /// row-major. Samples before the start read as silence.
    pub fn utterance(&mut self, audio: &[f32]) -> Vec<f32> {
        let n = audio.len() / AUDIO_HOP;
        let mut out = vec![0.0; n * AUDIO_FEATS];
        let mut buf = vec![0.0; AUDIO_WIN];
        for (i, row) in out.chunks_exact_mut(AUDIO_FEATS).enumerate() {
            let end = (i + 1) * AUDIO_HOP;
            let skip = AUDIO_WIN.saturating_sub(end);
            buf[..skip].fill(0.0);
            buf[skip..].copy_from_slice(&audio[end + skip - AUDIO_WIN..end]);
            self.frame(&buf, row);
        }
        out
    }
}

/// Feeds a stream through [`LogMel`] hop by hop.
pub struct StreamingLogMel {
    mel: LogMel,
    /// The last `AUDIO_WIN` samples, oldest first.
    history: Vec<f32>,
    /// Samples received since the last emitted frame.
    pending: Vec<f32>,
}

impl Default for StreamingLogMel {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingLogMel {
    pub fn new() -> Self {
        Self {
            mel: LogMel::new(),
            history: vec![0.0; AUDIO_WIN],
            pending: Vec::with_capacity(AUDIO_HOP),
        }
    }

    /// Takes any number of samples and appends one feature row to `out` for
    /// every hop they complete.
    pub fn push(&mut self, samples: &[f32], out: &mut Vec<f32>) {
        let mut row = [0.0; AUDIO_FEATS];
        for &s in samples {
            self.pending.push(s);
            if self.pending.len() == AUDIO_HOP {
                self.history.copy_within(AUDIO_HOP.., 0);
                self.history[AUDIO_WIN - AUDIO_HOP..].copy_from_slice(&self.pending);
                self.pending.clear();
                self.mel.frame(&self.history, &mut row);
                out.extend_from_slice(&row);
            }
        }
    }
}

/// Per-feature mean and standard deviation over a corpus, applied to every
/// frame alike. Fixed once measured, so a louder utterance stays louder.
#[derive(Clone, Debug)]
pub struct FeatureStats {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl FeatureStats {
    /// No-op statistics, for tests and for features consumed raw.
    pub fn identity() -> Self {
        Self {
            mean: vec![0.0; AUDIO_FEATS],
            std: vec![1.0; AUDIO_FEATS],
        }
    }

    /// Normalizes `[n, AUDIO_FEATS]` row-major features in place.
    pub fn apply(&self, feats: &mut [f32]) {
        for row in feats.chunks_exact_mut(AUDIO_FEATS) {
            for ((x, m), s) in row.iter_mut().zip(&self.mean).zip(&self.std) {
                *x = (*x - m) / s;
            }
        }
    }
}

/// Running sums for [`FeatureStats`], in f64 because a corpus is billions of
/// frames.
pub struct StatsAccumulator {
    n: u64,
    sum: Vec<f64>,
    sum_sq: Vec<f64>,
}

impl Default for StatsAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl StatsAccumulator {
    pub fn new() -> Self {
        Self {
            n: 0,
            sum: vec![0.0; AUDIO_FEATS],
            sum_sq: vec![0.0; AUDIO_FEATS],
        }
    }

    /// Adds `[n, AUDIO_FEATS]` row-major features.
    pub fn add(&mut self, feats: &[f32]) {
        for row in feats.chunks_exact(AUDIO_FEATS) {
            for ((s, q), &x) in self.sum.iter_mut().zip(&mut self.sum_sq).zip(row) {
                *s += x as f64;
                *q += (x as f64) * (x as f64);
            }
            self.n += 1;
        }
    }

    pub fn finish(&self) -> FeatureStats {
        let n = self.n.max(1) as f64;
        let mean: Vec<f64> = self.sum.iter().map(|s| s / n).collect();
        FeatureStats {
            mean: mean.iter().map(|&m| m as f32).collect(),
            std: self
                .sum_sq
                .iter()
                .zip(&mean)
                .map(|(q, m)| ((q / n - m * m).max(0.0).sqrt() as f32).max(1e-3))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use super::*;
    use crate::audio::mel::MelScale;

    fn sine(hz: f32, amp: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| amp * (2.0 * PI * hz * i as f32 / AUDIO_SR as f32).sin())
            .collect()
    }

    #[test]
    fn a_tone_peaks_in_its_own_mel_bin() {
        let mut mel = LogMel::new();
        let scale = MelScale::Slaney;
        for hz in [200.0, 1000.0, 4000.0] {
            let f = mel.utterance(&sine(hz, 0.5, AUDIO_SR / 4));
            let row = &f[10 * AUDIO_FEATS..11 * AUDIO_FEATS];
            let peak = (0..AUDIO_N_MELS)
                .max_by(|&a, &b| row[a].total_cmp(&row[b]))
                .unwrap();
            // Bin m is centred on edge m + 1 of the n_mels + 2 evenly spaced in mel.
            let want = (scale.to_mel(hz) / scale.to_mel(8000.0) * (AUDIO_N_MELS + 1) as f32).round()
                as usize
                - 1;
            assert!(peak.abs_diff(want) <= 1, "{hz} Hz peaked in bin {peak}, expected ~{want}");
        }
    }

    #[test]
    fn loudness_survives_the_features() {
        // Twice the amplitude is four times the power: +ln 4 in every log energy.
        let mut mel = LogMel::new();
        let quiet = mel.utterance(&sine(440.0, 0.1, AUDIO_SR / 4));
        let loud = mel.utterance(&sine(440.0, 0.2, AUDIO_SR / 4));
        let at = 10 * AUDIO_FEATS + AUDIO_N_MELS;
        assert!((loud[at] - quiet[at] - 4f32.ln()).abs() < 1e-3);
    }

    #[test]
    fn streaming_matches_the_whole_utterance() {
        let audio = sine(523.0, 0.3, 3 * AUDIO_HOP * 7 + 55);
        let whole = LogMel::new().utterance(&audio);
        let mut stream = StreamingLogMel::new();
        let mut out = Vec::new();
        for chunk in audio.chunks(97) {
            stream.push(chunk, &mut out);
        }
        assert_eq!(out.len(), whole.len());
        for (a, b) in out.iter().zip(&whole) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn stats_normalize_to_zero_mean_unit_std() {
        let feats: Vec<f32> = (0..50 * AUDIO_FEATS).map(|i| (i % 7) as f32 * 0.5 + 3.0).collect();
        let mut acc = StatsAccumulator::new();
        acc.add(&feats);
        let stats = acc.finish();
        let mut norm = feats.clone();
        stats.apply(&mut norm);
        let mut check = StatsAccumulator::new();
        check.add(&norm);
        let s = check.finish();
        for (m, d) in s.mean.iter().zip(&s.std) {
            assert!(m.abs() < 1e-4 && (d - 1.0).abs() < 1e-3);
        }
    }
}
