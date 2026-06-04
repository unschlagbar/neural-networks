use std::f32::consts::PI;

use crate::config::{WAKE_FRAME_LEN, WAKE_N_FFT, WAKE_N_MELS, WAKE_SR};

pub struct MelExtractor {
    window: Vec<f32>,
    mel_filters: Vec<Vec<f32>>, // [N_MELS × (N_FFT/2+1)]
}

impl MelExtractor {
    pub fn new() -> Self {
        Self {
            window: hann_window(WAKE_FRAME_LEN),
            mel_filters: build_mel_filters(WAKE_N_MELS, WAKE_SR, WAKE_N_FFT, 80.0, 8000.0),
        }
    }

    /// Extract log-mel features from a single frame of exactly `WAKE_FRAME_LEN` samples.
    /// Returns `WAKE_N_MELS` values. Suitable for streaming inference.
    pub fn extract_frame(&self, samples: &[f32]) -> Vec<f32> {
        debug_assert_eq!(samples.len(), WAKE_FRAME_LEN);
        let n_bins = WAKE_N_FFT / 2 + 1;
        let mut re = vec![0.0; WAKE_N_FFT];
        let mut im = vec![0.0; WAKE_N_FFT];

        for j in 0..WAKE_FRAME_LEN {
            re[j] = samples[j] * self.window[j];
        }
        fft_inplace(&mut re, &mut im);

        let power: Vec<f32> = (0..n_bins).map(|k| re[k] * re[k] + im[k] * im[k]).collect();

        let mut log_mel: Vec<f32> = self
            .mel_filters
            .iter()
            .map(|filt| {
                let e: f32 = filt.iter().zip(&power).map(|(w, p)| w * p).sum();
                (e + 1e-10).ln()
            })
            .collect();

        // Per-frame mean subtraction: removes overall volume offset so
        // the model sees spectral shape rather than absolute energy level.
        let mean = log_mel.iter().sum::<f32>() / log_mel.len() as f32;
        log_mel.iter_mut().for_each(|v| *v -= mean);
        log_mel
    }
}

// ── Hann window ──────────────────────────────────────────────────────────────

fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1) as f32).cos()))
        .collect()
}

// ── Mel filterbank ───────────────────────────────────────────────────────────

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

fn build_mel_filters(
    n_mels: usize,
    sr: usize,
    n_fft: usize,
    low_hz: f32,
    high_hz: f32,
) -> Vec<Vec<f32>> {
    let n_bins = n_fft / 2 + 1;
    let low_mel = hz_to_mel(low_hz);
    let high_mel = hz_to_mel(high_hz);

    // n_mels + 2 evenly spaced mel points
    let mel_pts: Vec<f32> = (0..n_mels + 2)
        .map(|i| low_mel + (high_mel - low_mel) * i as f32 / (n_mels + 1) as f32)
        .collect();

    // Convert to nearest FFT bin
    let bins: Vec<usize> = mel_pts
        .iter()
        .map(|&m| ((mel_to_hz(m) * (n_fft + 1) as f32 / sr as f32) as usize).min(n_bins - 1))
        .collect();

    (0..n_mels)
        .map(|m| {
            let mut filt = vec![0.0f32; n_bins];
            let (l, c, r) = (bins[m], bins[m + 1], bins[m + 2]);
            if c > l {
                for k in l..=c {
                    filt[k] = (k - l) as f32 / (c - l) as f32;
                }
            }
            if r > c {
                for k in c..=r {
                    filt[k] = (r - k) as f32 / (r - c) as f32;
                }
            }
            filt
        })
        .collect()
}

// ── Radix-2 Cooley-Tukey FFT (in-place, DIT) ─────────────────────────────────

fn fft_inplace(re: &mut Vec<f32>, im: &mut Vec<f32>) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());
    let log2n = n.trailing_zeros() as usize;

    for i in 0..n {
        let j = bit_reverse(i, log2n);
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    let mut half = 1;
    while half < n {
        let step = half * 2;
        let angle = -PI / half as f32;
        let (wre, wim) = (angle.cos(), angle.sin());
        let mut k = 0;
        while k < n {
            let (mut ur, mut ui) = (1.0, 0.0);
            for j in 0..half {
                let a = k + j;
                let b = k + j + half;
                let vr = re[b] * ur - im[b] * ui;
                let vi = re[b] * ui + im[b] * ur;
                re[b] = re[a] - vr;
                im[b] = im[a] - vi;
                re[a] += vr;
                im[a] += vi;
                let new_ur = ur * wre - ui * wim;
                ui = ur * wim + ui * wre;
                ur = new_ur;
            }
            k += step;
        }
        half = step;
    }
}

fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut r = 0;
    for _ in 0..bits {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    r
}

// ── Linear resampler ─────────────────────────────────────────────────────────

/// Resample `samples` from `from_sr` to `to_sr` using linear interpolation.
pub fn resample(samples: &[f32], from_sr: usize, to_sr: usize) -> Vec<f32> {
    debug_assert_ne!(from_sr, to_sr, "cannot resample to same sample rate");

    let ratio = from_sr as f32 / to_sr as f32;
    let out_len = (samples.len() as f32 / ratio) as usize;
    (0..out_len)
        .map(|i| {
            let pos = i as f32 * ratio;
            let idx = pos as usize;
            let frac = pos - idx as f32;
            let a = samples[idx];
            let b = samples.get(idx + 1).copied().unwrap_or(a);
            a + frac * (b - a)
        })
        .collect()
}

/// Trim or zero-pad to exactly `target` samples.
pub fn normalize_length(mut samples: Vec<f32>, target: usize) -> Vec<f32> {
    if samples.len() >= target {
        samples.truncate(target);
    } else {
        samples.resize(target, 0.0);
    }
    samples
}

pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
