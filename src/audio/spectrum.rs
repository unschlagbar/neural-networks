//! Windowed power spectra of single frames.

use std::f32::consts::PI;

use super::fft::Fft;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Window {
    /// Periodic Hann, the form spectral analysis uses (torch/librosa default).
    Hann,
    /// Symmetric Hann, zero at both ends — what the wake-word models were trained with.
    SymmetricHann,
}

impl Window {
    pub fn coefficients(self, n: usize) -> Vec<f32> {
        let period = match self {
            Self::Hann => n,
            Self::SymmetricHann => n - 1,
        } as f32;
        (0..n)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / period).cos()))
            .collect()
    }
}

/// Power spectrum of one `win_len`-sample frame, zero-padded to `n_fft`.
pub struct Spectrogram {
    window: Vec<f32>,
    fft: Fft,
    windowed: Vec<f32>,
    power: Vec<f32>,
}

impl Spectrogram {
    pub fn new(win_len: usize, n_fft: usize, window: Window) -> Self {
        assert!(win_len <= n_fft, "Spectrogram: window longer than the FFT");
        Self {
            window: window.coefficients(win_len),
            fft: Fft::new(n_fft),
            windowed: vec![0.0; win_len],
            power: vec![0.0; n_fft / 2 + 1],
        }
    }

    pub fn win_len(&self) -> usize {
        self.window.len()
    }

    /// `|X_k|²`, `n_fft / 2 + 1` bins, valid until the next call.
    pub fn frame(&mut self, samples: &[f32]) -> &[f32] {
        assert_eq!(samples.len(), self.window.len(), "Spectrogram::frame takes one window");
        for (o, (&s, &w)) in self.windowed.iter_mut().zip(samples.iter().zip(&self.window)) {
            *o = s * w;
        }
        self.fft.power(&self.windowed, &mut self.power);
        &self.power
    }
}
