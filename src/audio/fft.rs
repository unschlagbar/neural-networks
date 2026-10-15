//! Radix-2 FFT with twiddles and bit-reversal order computed once per length.

use std::f32::consts::PI;

pub struct Fft {
    n: usize,
    /// `exp(−2πik/n)` for `k < n/2`, as `(cos, sin)`.
    twiddles: Vec<(f32, f32)>,
    /// Bit-reversed index of every position.
    rev: Vec<u32>,
    re: Vec<f32>,
    im: Vec<f32>,
}

impl Fft {
    pub fn new(n: usize) -> Self {
        assert!(n.is_power_of_two(), "Fft: length {n} is not a power of two");
        let bits = n.trailing_zeros();
        Self {
            n,
            twiddles: (0..n / 2)
                .map(|k| {
                    let a = -2.0 * PI * k as f32 / n as f32;
                    (a.cos(), a.sin())
                })
                .collect(),
            rev: (0..n as u32)
                .map(|i| i.reverse_bits().checked_shr(32 - bits).unwrap_or(0))
                .collect(),
            re: vec![0.0; n],
            im: vec![0.0; n],
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// `|X_k|²` for `k = 0..=n/2` of `input` zero-padded to the transform length.
    pub fn power(&mut self, input: &[f32], out: &mut [f32]) {
        assert!(input.len() <= self.n, "Fft::power: input longer than the transform");
        assert_eq!(out.len(), self.n / 2 + 1, "Fft::power: output is n/2 + 1 bins");
        self.re[..input.len()].copy_from_slice(input);
        self.re[input.len()..].fill(0.0);
        self.im.fill(0.0);
        self.transform();
        for (k, p) in out.iter_mut().enumerate() {
            *p = self.re[k] * self.re[k] + self.im[k] * self.im[k];
        }
    }

    /// In-place decimation-in-time transform of `re`/`im`.
    fn transform(&mut self) {
        let n = self.n;
        for i in 0..n {
            let j = self.rev[i] as usize;
            if i < j {
                self.re.swap(i, j);
                self.im.swap(i, j);
            }
        }
        let mut half = 1;
        while half < n {
            let stride = n / (2 * half);
            for start in (0..n).step_by(2 * half) {
                for j in 0..half {
                    let (wr, wi) = self.twiddles[j * stride];
                    let (a, b) = (start + j, start + j + half);
                    let vr = self.re[b] * wr - self.im[b] * wi;
                    let vi = self.re[b] * wi + self.im[b] * wr;
                    self.re[b] = self.re[a] - vr;
                    self.im[b] = self.im[a] - vi;
                    self.re[a] += vr;
                    self.im[a] += vi;
                }
            }
            half *= 2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_direct_dft() {
        let n = 16;
        let x: Vec<f32> = (0..12).map(|i| ((i * 7) % 5) as f32 - 2.0).collect();
        let mut out = vec![0.0; n / 2 + 1];
        Fft::new(n).power(&x, &mut out);
        for (k, &got) in out.iter().enumerate() {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (t, &v) in x.iter().enumerate() {
                let a = -2.0 * std::f64::consts::PI * (k * t) as f64 / n as f64;
                re += v as f64 * a.cos();
                im += v as f64 * a.sin();
            }
            let want = (re * re + im * im) as f32;
            assert!((got - want).abs() < 1e-3 * want.max(1.0), "bin {k}: {got} vs {want}");
        }
    }
}
