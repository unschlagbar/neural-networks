//! Mel filterbanks over a power spectrum.

/// Floor under every energy before the log, about -100 dB.
const ENERGY_FLOOR: f32 = 1e-10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MelScale {
    /// `2595 · log10(1 + f/700)`.
    Htk,
    /// Linear below 1 kHz, logarithmic above (librosa's and Whisper's default).
    Slaney,
}

const SLANEY_F_SP: f32 = 200.0 / 3.0;
const SLANEY_MIN_LOG_HZ: f32 = 1000.0;
const SLANEY_MIN_LOG_MEL: f32 = SLANEY_MIN_LOG_HZ / SLANEY_F_SP;

impl MelScale {
    pub fn to_mel(self, hz: f32) -> f32 {
        match self {
            Self::Htk => 2595.0 * (1.0 + hz / 700.0).log10(),
            Self::Slaney if hz < SLANEY_MIN_LOG_HZ => hz / SLANEY_F_SP,
            Self::Slaney => SLANEY_MIN_LOG_MEL + (hz / SLANEY_MIN_LOG_HZ).ln() / Self::slaney_log_step(),
        }
    }

    pub fn to_hz(self, mel: f32) -> f32 {
        match self {
            Self::Htk => 700.0 * (10f32.powf(mel / 2595.0) - 1.0),
            Self::Slaney if mel < SLANEY_MIN_LOG_MEL => mel * SLANEY_F_SP,
            Self::Slaney => SLANEY_MIN_LOG_HZ * ((mel - SLANEY_MIN_LOG_MEL) * Self::slaney_log_step()).exp(),
        }
    }

    fn slaney_log_step() -> f32 {
        6.4f32.ln() / 27.0
    }

    /// `n + 2` frequencies evenly spaced on this scale from `f_min` to `f_max`:
    /// the lower edge, centre and upper edge of `n` overlapping triangles.
    fn edges(self, n: usize, f_min: f32, f_max: f32) -> Vec<f32> {
        let (lo, hi) = (self.to_mel(f_min), self.to_mel(f_max));
        (0..n + 2)
            .map(|i| self.to_hz(lo + (hi - lo) * i as f32 / (n + 1) as f32))
            .collect()
    }
}

pub struct MelFilterBank {
    /// Per filter: the first FFT bin it covers and its weights from there.
    filters: Vec<(usize, Vec<f32>)>,
    /// Power-spectrum length it expects, `n_fft / 2 + 1`.
    bins: usize,
}

impl MelFilterBank {
    /// Slaney-scale triangles evaluated at each bin's frequency, area-normalized,
    /// as `librosa.filters.mel` builds them. Evaluating the triangle rather than
    /// snapping its corners to bins keeps the narrow low filters from coming out
    /// empty.
    pub fn slaney(n_mels: usize, sample_rate: usize, n_fft: usize, f_min: f32, f_max: f32) -> Self {
        let bins = n_fft / 2 + 1;
        let edges = MelScale::Slaney.edges(n_mels, f_min, f_max);
        let filters = (0..n_mels)
            .map(|m| {
                let (lo, mid, hi) = (edges[m], edges[m + 1], edges[m + 2]);
                let norm = 2.0 / (hi - lo);
                Self::compact((0..bins).filter_map(|k| {
                    let f = k as f32 * sample_rate as f32 / n_fft as f32;
                    let w = ((f - lo) / (mid - lo)).min((hi - f) / (hi - mid));
                    (w > 0.0).then_some((k, w * norm))
                }))
            })
            .collect();
        Self { filters, bins }
    }

    /// HTK-scale triangles with their corners snapped to the nearest lower bin
    /// and peak weight 1 — the bank the wake-word models were trained on.
    pub fn htk_snapped(n_mels: usize, sample_rate: usize, n_fft: usize, f_min: f32, f_max: f32) -> Self {
        let bins = n_fft / 2 + 1;
        let corners: Vec<usize> = MelScale::Htk
            .edges(n_mels, f_min, f_max)
            .iter()
            .map(|&hz| ((hz * (n_fft + 1) as f32 / sample_rate as f32) as usize).min(bins - 1))
            .collect();
        let filters = (0..n_mels)
            .map(|m| {
                let (l, c, r) = (corners[m], corners[m + 1], corners[m + 2]);
                let mut w = vec![0.0; bins];
                if c > l {
                    for (k, v) in w.iter_mut().enumerate().take(c + 1).skip(l) {
                        *v = (k - l) as f32 / (c - l) as f32;
                    }
                }
                if r > c {
                    for (k, v) in w.iter_mut().enumerate().take(r + 1).skip(c) {
                        *v = (r - k) as f32 / (r - c) as f32;
                    }
                }
                Self::compact(w.into_iter().enumerate().filter(|&(_, v)| v > 0.0))
            })
            .collect();
        Self { filters, bins }
    }

    /// `(bin, weight)` pairs in ascending bin order as a first bin plus a dense run.
    fn compact(weights: impl Iterator<Item = (usize, f32)>) -> (usize, Vec<f32>) {
        let weights: Vec<(usize, f32)> = weights.collect();
        let first = weights.first().map_or(0, |&(k, _)| k);
        let mut dense = vec![0.0; weights.last().map_or(0, |&(k, _)| k + 1 - first)];
        for (k, w) in weights {
            dense[k - first] = w;
        }
        (first, dense)
    }

    pub fn len(&self) -> usize {
        self.filters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    /// Weight sum of each filter. Diagnostic.
    pub fn filter_mass(&self) -> Vec<f32> {
        self.filters.iter().map(|(_, w)| w.iter().sum()).collect()
    }

    /// `ln(energy)` of every filter over `power` (`n_fft / 2 + 1` bins) into `out`.
    pub fn log_energies(&self, power: &[f32], out: &mut [f32]) {
        assert_eq!(power.len(), self.bins, "MelFilterBank: power spectrum length");
        assert_eq!(out.len(), self.filters.len(), "MelFilterBank: one output per filter");
        for (o, (first, weights)) in out.iter_mut().zip(&self.filters) {
            let e: f32 = weights.iter().zip(&power[*first..]).map(|(w, p)| w * p).sum();
            *o = (e + ENERGY_FLOOR).ln();
        }
    }
}

/// `ln` of a spectrum's total energy, floored like the filter energies.
pub fn log_total_energy(power: &[f32]) -> f32 {
    (power.iter().sum::<f32>() + ENERGY_FLOOR).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scales_round_trip() {
        for scale in [MelScale::Htk, MelScale::Slaney] {
            for hz in [0.0, 300.0, 999.0, 1000.0, 4321.0, 8000.0] {
                let back = scale.to_hz(scale.to_mel(hz));
                assert!((back - hz).abs() < 0.05, "{scale:?} {hz} -> {back}");
            }
        }
    }

    #[test]
    fn no_slaney_filter_is_empty() {
        let bank = MelFilterBank::slaney(128, 16_000, 512, 0.0, 8000.0);
        for (m, mass) in bank.filter_mass().iter().enumerate() {
            assert!(*mass > 0.0, "mel filter {m} is empty");
        }
    }
}
