//! Audio in and out of files and devices: WAV clips, channel mixdown and
//! sample-rate conversion.

use std::io;
use std::path::Path;

/// A mono clip.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Clip {
    pub samples: Vec<f32>,
    pub sample_rate: usize,
}

impl Clip {
    pub fn new(samples: Vec<f32>, sample_rate: usize) -> Self {
        Self {
            samples,
            sample_rate,
        }
    }

    /// Reads a 16/24/32-bit integer or 32-bit float WAV, mixing it down to mono.
    pub fn read(path: impl AsRef<Path>) -> io::Result<Self> {
        let reader = hound::WavReader::open(path).map_err(io::Error::other)?;
        let spec = reader.spec();
        let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
            (hound::SampleFormat::Float, 32) => reader
                .into_samples::<f32>()
                .collect::<Result<_, _>>()
                .map_err(io::Error::other)?,
            (hound::SampleFormat::Int, 16) => reader
                .into_samples::<i16>()
                .map(|s| s.map(|s| s as f32 / 32768.0))
                .collect::<Result<_, _>>()
                .map_err(io::Error::other)?,
            (hound::SampleFormat::Int, 24 | 32) => {
                let scale = (1u64 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .into_samples::<i32>()
                    .map(|s| s.map(|s| s as f32 / scale))
                    .collect::<Result<_, _>>()
                    .map_err(io::Error::other)?
            }
            (format, bits) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unsupported WAV format {format:?} {bits}-bit"),
                ));
            }
        };
        let mut mono = Vec::with_capacity(samples.len() / spec.channels.max(1) as usize);
        Channels(spec.channels as usize).mix_f32(&samples, &mut mono);
        Ok(Self::new(mono, spec.sample_rate as usize))
    }

    /// Reads a WAV and converts it to `sample_rate`.
    pub fn read_at(path: impl AsRef<Path>, sample_rate: usize) -> io::Result<Self> {
        Ok(Self::read(path)?.resampled(sample_rate))
    }

    /// Writes 16-bit PCM, clipping at full scale.
    pub fn write(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate as u32,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).map_err(io::Error::other)?;
        for &s in &self.samples {
            writer
                .write_sample((s * 32767.0).clamp(-32768.0, 32767.0) as i16)
                .map_err(io::Error::other)?;
        }
        writer.finalize().map_err(io::Error::other)
    }

    pub fn resampled(self, sample_rate: usize) -> Self {
        if sample_rate == self.sample_rate {
            return self;
        }
        let samples = Resampler::new(self.sample_rate, sample_rate).process(&self.samples);
        Self::new(samples, sample_rate)
    }

    pub fn seconds(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32
    }
}

/// Interleaved channel count of a device or file stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Channels(pub usize);

impl Channels {
    /// Appends the mean of each interleaved frame of `data` to `out`.
    pub fn mix_f32(self, data: &[f32], out: &mut Vec<f32>) {
        if self.0 <= 1 {
            out.extend_from_slice(data);
            return;
        }
        let inv = 1.0 / self.0 as f32;
        out.extend(data.chunks(self.0).map(|c| c.iter().sum::<f32>() * inv));
    }

    /// [`mix_f32`](Self::mix_f32) for 16-bit integer samples.
    pub fn mix_i16(self, data: &[i16], out: &mut Vec<f32>) {
        let inv = 1.0 / (32768.0 * self.0.max(1) as f32);
        out.extend(
            data.chunks(self.0.max(1))
                .map(|c| c.iter().map(|&s| s as f32).sum::<f32>() * inv),
        );
    }
}

/// Linear-interpolation sample-rate conversion. No anti-alias filter: fine for
/// speed perturbation and for devices that already deliver near the target
/// rate, not for downsampling far.
#[derive(Clone, Copy, Debug)]
pub struct Resampler {
    /// Input samples per output sample.
    ratio: f32,
}

impl Resampler {
    pub fn new(from: usize, to: usize) -> Self {
        Self {
            ratio: from as f32 / to as f32,
        }
    }

    pub fn process(&self, input: &[f32]) -> Vec<f32> {
        let mut out = Vec::new();
        self.process_into(input, &mut out);
        out
    }

    /// Replaces `out`'s contents, reusing its capacity.
    pub fn process_into(&self, input: &[f32], out: &mut Vec<f32>) {
        let n = (input.len() as f32 / self.ratio) as usize;
        out.clear();
        out.extend((0..n).map(|i| {
            let pos = i as f32 * self.ratio;
            let idx = pos as usize;
            let a = input[idx];
            let b = input.get(idx + 1).copied().unwrap_or(a);
            a + (pos - idx as f32) * (b - a)
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixdown_averages_channels() {
        let mut out = Vec::new();
        Channels(2).mix_f32(&[1.0, 0.0, 0.5, 0.5], &mut out);
        assert_eq!(out, vec![0.5, 0.5]);
        out.clear();
        Channels(2).mix_i16(&[16384, 16384], &mut out);
        assert_eq!(out, vec![0.5]);
    }

    #[test]
    fn resampling_keeps_duration_and_interpolates() {
        let up = Resampler::new(8_000, 16_000).process(&[0.0, 1.0, 0.0, -1.0]);
        assert_eq!(up.len(), 8);
        assert_eq!(&up[..3], &[0.0, 0.5, 1.0]);
    }

    #[test]
    fn wav_round_trips_at_16_bit() {
        let dir = std::env::temp_dir().join(format!("nn-wav-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("clip.wav");
        let clip = Clip::new(vec![0.0, 0.25, -0.5, 0.999], 16_000);
        clip.write(&path).unwrap();
        let back = Clip::read(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(back.sample_rate, 16_000);
        for (a, b) in back.samples.iter().zip(&clip.samples) {
            assert!((a - b).abs() < 1e-4);
        }
    }
}
