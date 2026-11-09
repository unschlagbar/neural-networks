//! Making clean recordings sound like a room: reverberation, background noise,
//! microphone coloration and distance. Rooms and coloration follow the wake
//! word's augmentation in `smart-light`; convolution runs through the FFT, since
//! a speech utterance is seconds long.
//!
//! Nothing here normalizes loudness: the level a word arrives at is part of how
//! it was said.

use std::f32::consts::{LN_10, PI};
use std::path::Path;

use rand::random_range;

use super::fft::Fft;
use super::io::Clip;

/// How hard to augment, and how often.
#[derive(Clone, Debug)]
pub struct AugmentCfg {
    pub sample_rate: usize,
    /// Chance of a room, background noise and microphone coloration, each
    /// drawn on its own.
    pub p_room: f32,
    pub p_noise: f32,
    pub p_mic: f32,
    /// Chance of another speaker under the speech (a television, someone else
    /// in the room), and its level below the speech, dB.
    pub p_babble: f32,
    pub babble_snr_db: (f32, f32),
    /// RT60 range of a synthetic room, seconds.
    pub rt60: (f32, f32),
    /// Direct-to-reverberant ratio, dB: high is next to the microphone.
    pub drr_db: (f32, f32),
    /// Speech-to-noise ratio, dB.
    pub snr_db: (f32, f32),
    /// Overall gain, dB: distance to the microphone and how loudly one
    /// speaks (the user's takes sit near −25 dBFS; −18 dB brings them to −43).
    pub gain_db: (f32, f32),
    /// The microphone's own noise, dBFS, white: added after the gain, so
    /// quieter speech lies closer to it. Never within `floor_below_db` of the
    /// speech.
    pub floor_dbfs: (f32, f32),
    pub floor_below_db: f32,
    /// Speed factor range: tempo and pitch together, as resampling moves them
    /// (1.1 = 10% faster and higher). A caller that changes the length must
    /// stretch its timings with it ([`Augment::speed`]).
    pub speed: (f32, f32),
}

impl AugmentCfg {
    /// A person talking to a microphone in a living room.
    pub fn speech(sample_rate: usize) -> Self {
        Self {
            sample_rate,
            p_room: 0.5,
            p_noise: 0.6,
            p_mic: 0.5,
            p_babble: 0.2,
            babble_snr_db: (5.0, 20.0),
            rt60: (0.15, 0.8),
            drr_db: (-3.0, 12.0),
            snr_db: (0.0, 25.0),
            gain_db: (-18.0, 6.0),
            floor_dbfs: (-80.0, -60.0),
            floor_below_db: 15.0,
            speed: (0.9, 1.1),
        }
    }
}

/// A path from speaker to microphone, as an impulse response with the direct
/// sound at sample 0, energy-normalized so loudness is roughly preserved.
#[derive(Clone, Debug)]
pub struct Room {
    pub ir: Vec<f32>,
}

impl Room {
    /// The direct path, a few early reflections within 40 ms, and a diffuse
    /// tail whose highs (above 1.5 kHz) decay 0.4-0.8x as fast as its lows.
    /// `drr_db` sets the direct path's energy against everything after it.
    pub fn synthetic(sample_rate: usize, rt60: f32, drr_db: f32) -> Self {
        let sr = sample_rate as f32;
        let early = (0.04 * sr) as usize;
        let len = ((rt60 * sr) as usize).max(early + 1);
        // Amplitude decays by 60 dB (factor 1000) over rt60.
        let tau_low = rt60 / (3.0 * LN_10);
        let tau_high = tau_low * random_range(0.4..0.8);
        let split = one_pole(1500.0, sample_rate);

        let mut ir = vec![0.0; len];
        let mut low = 0.0;
        for (i, h) in ir.iter_mut().enumerate().skip(1) {
            let n = random_range(-1.0..1.0);
            low += split * (n - low);
            let t = i as f32 / sr;
            *h = low * (-t / tau_low).exp() + (n - low) * (-t / tau_high).exp();
        }
        for _ in 0..random_range(3..=8) {
            let i = random_range(1..early);
            let sign = if random_range(0..2) == 0 { -1.0 } else { 1.0 };
            ir[i] += sign * random_range(0.2..0.7) * (-(i as f32 / sr) / tau_low).exp();
        }

        let reverb: f32 = ir.iter().map(|h| h * h).sum();
        if reverb > 0.0 {
            let gain = (10f32.powf(-drr_db / 10.0) / reverb).sqrt();
            ir.iter_mut().for_each(|h| *h *= gain);
        }
        ir[0] = 1.0;
        Self::normalized(ir)
    }

    /// A measured response, aligned on its direct path and cut to `max_len`.
    pub fn measured(ir: &[f32], max_len: usize) -> Self {
        let peak = ir
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .map_or(0, |(i, _)| i);
        let mut out: Vec<f32> = ir[peak..].iter().take(max_len).copied().collect();
        if out.is_empty() {
            out.push(1.0);
        }
        Self::normalized(out)
    }

    fn normalized(mut ir: Vec<f32>) -> Self {
        let energy = ir.iter().map(|h| h * h).sum::<f32>().sqrt();
        if energy > 0.0 {
            ir.iter_mut().for_each(|h| *h /= energy);
        }
        Self { ir }
    }

    /// `audio` heard through this room, cut to its own length: the tail past
    /// the end belongs to no word.
    pub fn apply(&self, audio: &[f32]) -> Vec<f32> {
        let n = (audio.len() + self.ir.len() - 1).next_power_of_two();
        let mut out = Fft::new(n).convolve(audio, &self.ir);
        out.truncate(audio.len());
        out
    }
}

/// Microphone and placement coloration: a high-pass at the mic's low cutoff
/// (40-200 Hz), then a tilt of up to ±6 dB between lows and highs around 1 kHz.
pub struct MicColor {
    highpass: f32,
    split: f32,
    /// Gain on the highs, and its inverse on the lows.
    half_tilt: f32,
}

impl MicColor {
    pub fn random(sample_rate: usize) -> Self {
        Self {
            highpass: one_pole(random_range(40.0..200.0), sample_rate),
            split: one_pole(1000.0, sample_rate),
            half_tilt: 10f32.powf(random_range(-6.0..6.0) / 40.0),
        }
    }

    pub fn apply(&self, audio: &mut [f32]) {
        let (mut rumble, mut low) = (0.0, 0.0);
        for s in audio.iter_mut() {
            rumble += self.highpass * (*s - rumble);
            let x = *s - rumble;
            low += self.split * (x - low);
            *s = low / self.half_tilt + (x - low) * self.half_tilt;
        }
    }
}

/// Background noise: recordings when there are any, otherwise or besides
/// them white, pink or brown noise.
pub struct NoiseBank {
    clips: Vec<Vec<f32>>,
}

impl NoiseBank {
    /// Every WAV/FLAC under `dir` (recursively) at `sample_rate`; none if the
    /// directory does not exist.
    pub fn load(dir: impl AsRef<Path>, sample_rate: usize) -> Self {
        let mut paths = Vec::new();
        let mut stack = vec![dir.as_ref().to_path_buf()];
        while let Some(d) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&d) else {
                continue;
            };
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|x| {
                    x.eq_ignore_ascii_case("wav") || x.eq_ignore_ascii_case("flac")
                }) {
                    paths.push(p);
                }
            }
        }
        paths.sort();
        let clips = paths
            .iter()
            .filter_map(|p| Clip::read_at(p, sample_rate).ok())
            .map(|c| c.samples)
            .filter(|s| s.iter().any(|&x| x != 0.0))
            .collect();
        Self { clips }
    }

    /// The recordings at `paths` at `sample_rate`; unreadable ones are skipped.
    pub fn from_files(paths: &[std::path::PathBuf], sample_rate: usize) -> Self {
        let clips = paths
            .iter()
            .filter_map(|p| Clip::read_at(p, sample_rate).ok())
            .map(|c| c.samples)
            .filter(|s| s.iter().any(|&x| x != 0.0))
            .collect();
        Self { clips }
    }

    pub fn len(&self) -> usize {
        self.clips.len()
    }

    pub fn is_empty(&self) -> bool {
        self.clips.is_empty()
    }

    /// `len` samples of noise: a random recording from a random offset, looped,
    /// or (a quarter of the time, and always without recordings) colored noise.
    pub fn draw(&self, len: usize) -> Vec<f32> {
        if self.clips.is_empty() || random_range(0..4) == 0 {
            return colored(len, random_range(0..3));
        }
        let clip = &self.clips[random_range(0..self.clips.len())];
        let start = random_range(0..clip.len());
        clip.iter().cycle().skip(start).take(len).copied().collect()
    }
}

/// White (0), pink (1) or brown (2) noise at unit RMS.
fn colored(len: usize, kind: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    // Pink: Paul Kellet's economy filter. Brown: a leaky integrator.
    let (mut b0, mut b1, mut b2, mut brown) = (0.0, 0.0, 0.0, 0.0);
    for _ in 0..len {
        let w: f32 = random_range(-1.0..1.0);
        out.push(match kind {
            0 => w,
            1 => {
                b0 = 0.99765 * b0 + w * 0.099_046;
                b1 = 0.96300 * b1 + w * 0.296_516_4;
                b2 = 0.57000 * b2 + w * 1.052_691_3;
                b0 + b1 + b2 + w * 0.1848
            }
            _ => {
                brown = 0.998 * brown + w * 0.05;
                brown
            }
        });
    }
    let rms = rms(&out);
    if rms > 0.0 {
        out.iter_mut().for_each(|s| *s /= rms);
    }
    out
}

/// Draws and applies a random condition to each clip.
pub struct Augment {
    pub cfg: AugmentCfg,
    pub noise: NoiseBank,
    /// Other people talking, for `cfg.p_babble`; none leaves it out.
    pub babble: NoiseBank,
}

impl Augment {
    /// A random speed factor from `cfg.speed`.
    pub fn draw_speed(&self) -> f32 {
        random_range(self.cfg.speed.0..=self.cfg.speed.1)
    }

    /// `audio` played `factor` times as fast: `1 / factor` as long, every
    /// frequency `factor` times as high.
    pub fn speed(audio: &[f32], sample_rate: usize, factor: f32) -> Vec<f32> {
        if factor == 1.0 {
            return audio.to_vec();
        }
        let played = (sample_rate as f32 * factor).round() as usize;
        Clip::new(audio.to_vec(), played).resampled(sample_rate).samples
    }

    /// `audio` as if heard in a random room: reverb, then background noise and
    /// another speaker at random SNRs against the speech, then the microphone,
    /// then the distance, then the microphone's own noise floor.
    pub fn apply(&self, audio: &mut Vec<f32>) {
        let c = &self.cfg;
        let speech = rms(audio);
        if speech <= 0.0 {
            return;
        }
        if random_range(0.0..1.0) < c.p_room {
            let room = Room::synthetic(
                c.sample_rate,
                random_range(c.rt60.0..c.rt60.1),
                random_range(c.drr_db.0..c.drr_db.1),
            );
            *audio = room.apply(audio);
        }
        if random_range(0.0..1.0) < c.p_noise {
            let noise = self.noise.draw(audio.len());
            let snr_db = random_range(c.snr_db.0..c.snr_db.1);
            let gain = speech / (rms(&noise).max(1e-9) * 10f32.powf(snr_db / 20.0));
            for (s, n) in audio.iter_mut().zip(&noise) {
                *s += n * gain;
            }
        }
        if !self.babble.is_empty() && random_range(0.0..1.0) < c.p_babble {
            let clip = &self.babble.clips[random_range(0..self.babble.clips.len())];
            let start = random_range(0..clip.len());
            let other: Vec<f32> = clip.iter().cycle().skip(start).take(audio.len()).copied().collect();
            let snr_db = random_range(c.babble_snr_db.0..c.babble_snr_db.1);
            let gain = speech / (rms(&other).max(1e-9) * 10f32.powf(snr_db / 20.0));
            for (s, n) in audio.iter_mut().zip(&other) {
                *s += n * gain;
            }
        }
        if random_range(0.0..1.0) < c.p_mic {
            MicColor::random(c.sample_rate).apply(audio);
        }
        let gain = 10f32.powf(random_range(c.gain_db.0..c.gain_db.1) / 20.0);
        for s in audio.iter_mut() {
            *s = (*s * gain).clamp(-1.0, 1.0);
        }
        let below = 20.0 * rms(audio).max(1e-9).log10() - c.floor_below_db;
        let floor_db = random_range(c.floor_dbfs.0..c.floor_dbfs.1).min(below);
        // Uniform on ±a has an RMS of a/√3.
        let a = 10f32.powf(floor_db / 20.0) * 3f32.sqrt();
        for s in audio.iter_mut() {
            *s = (*s + random_range(-a..a)).clamp(-1.0, 1.0);
        }
    }
}

pub fn rms(audio: &[f32]) -> f32 {
    if audio.is_empty() {
        return 0.0;
    }
    (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt()
}

/// Smoothing coefficient of a one-pole low-pass at `hz`.
fn one_pole(hz: f32, sample_rate: usize) -> f32 {
    1.0 - (-2.0 * PI * hz / sample_rate as f32).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_room_keeps_the_direct_sound_first_and_the_length() {
        let room = Room::synthetic(16_000, 0.4, 6.0);
        let peak = room.ir.iter().map(|h| h.abs()).fold(0.0, f32::max);
        assert_eq!(room.ir[0], peak, "the direct path is the strongest");
        let click: Vec<f32> = (0..4000)
            .map(|i| if i == 100 { 1.0 } else { 0.0 })
            .collect();
        let heard = room.apply(&click);
        assert_eq!(heard.len(), click.len());
        assert!(
            heard[..100].iter().all(|s| s.abs() < 1e-4),
            "nothing before the click"
        );
        assert!((heard[100] - room.ir[0]).abs() < 1e-4);
    }

    #[test]
    fn noise_lands_at_the_drawn_snr() {
        let cfg = AugmentCfg {
            p_room: 0.0,
            p_noise: 1.0,
            p_mic: 0.0,
            snr_db: (10.0, 10.0 + 1e-6),
            gain_db: (0.0, 1e-6),
            floor_dbfs: (-140.0, -139.9),
            ..AugmentCfg::speech(16_000)
        };
        let aug = Augment {
            cfg,
            noise: NoiseBank { clips: Vec::new() },
            babble: NoiseBank { clips: Vec::new() },
        };
        let speech: Vec<f32> = (0..16_000).map(|i| 0.1 * (i as f32 * 0.05).sin()).collect();
        let mut noisy = speech.clone();
        aug.apply(&mut noisy);
        let noise: Vec<f32> = noisy.iter().zip(&speech).map(|(a, b)| a - b).collect();
        let snr = 20.0 * (rms(&speech) / rms(&noise)).log10();
        assert!((snr - 10.0).abs() < 0.1, "SNR {snr}");
    }

    #[test]
    fn the_noise_floor_stays_below_quiet_speech() {
        let cfg = AugmentCfg {
            p_room: 0.0,
            p_noise: 0.0,
            p_mic: 0.0,
            p_babble: 0.0,
            gain_db: (0.0, 1e-6),
            floor_dbfs: (-40.0, -39.9),
            ..AugmentCfg::speech(16_000)
        };
        let aug = Augment {
            cfg,
            noise: NoiseBank { clips: Vec::new() },
            babble: NoiseBank { clips: Vec::new() },
        };
        // −50 dBFS: a floor drawn at −40 would bury it.
        let speech: Vec<f32> = (0..16_000).map(|i| 0.0045 * (i as f32 * 0.05).sin()).collect();
        let mut heard = speech.clone();
        aug.apply(&mut heard);
        let floor: Vec<f32> = heard.iter().zip(&speech).map(|(a, b)| a - b).collect();
        let below = 20.0 * (rms(&speech) / rms(&floor)).log10();
        assert!((below - 15.0).abs() < 0.3, "floor {below} dB below the speech");
    }
}
