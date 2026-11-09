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
        assert!(
            input.len() <= self.n,
            "Fft::power: input longer than the transform"
        );
        assert_eq!(
            out.len(),
            self.n / 2 + 1,
            "Fft::power: output is n/2 + 1 bins"
        );
        self.re[..input.len()].copy_from_slice(input);
        self.re[input.len()..].fill(0.0);
        self.im.fill(0.0);
        self.transform();
        for (k, p) in out.iter_mut().enumerate() {
            *p = self.re[k] * self.re[k] + self.im[k] * self.im[k];
        }
    }

    /// The linear convolution of `a` and `b`, `a.len() + b.len() − 1` long;
    /// the transform must be at least that long. Both real inputs ride in one
    /// complex transform (`a + ib`), whose spectrum separates them by symmetry.
    pub fn convolve(&mut self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let n = self.n;
        let len = a.len() + b.len() - 1;
        assert!(
            len <= n,
            "Fft::convolve: {len} samples need a longer transform than {n}"
        );
        self.re[..a.len()].copy_from_slice(a);
        self.re[a.len()..].fill(0.0);
        self.im[..b.len()].copy_from_slice(b);
        self.im[b.len()..].fill(0.0);
        self.transform();
        // A_k = (Z_k + conj Z_{n−k}) / 2, B_k = (Z_k − conj Z_{n−k}) / 2i, and the
        // product A_k·B_k goes back into the buffers.
        let mut prod = vec![(0.0f32, 0.0f32); n];
        for (k, p) in prod.iter_mut().enumerate() {
            let m = (n - k) % n;
            let (zr, zi, cr, ci) = (self.re[k], self.im[k], self.re[m], -self.im[m]);
            let (ar, ai) = (0.5 * (zr + cr), 0.5 * (zi + ci));
            let (br, bi) = (0.5 * (zi - ci), -0.5 * (zr - cr));
            *p = (ar * br - ai * bi, ar * bi + ai * br);
        }
        // The inverse as a forward transform of the conjugate.
        for (k, &(r, i)) in prod.iter().enumerate() {
            self.re[k] = r;
            self.im[k] = -i;
        }
        self.transform();
        self.re[..len].iter().map(|&r| r / n as f32).collect()
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
    fn convolution_matches_the_direct_sum() {
        let a: Vec<f32> = (0..37).map(|i| ((i * 11) % 7) as f32 - 3.0).collect();
        let b: Vec<f32> = (0..9).map(|i| ((i * 5) % 4) as f32 * 0.5 - 0.7).collect();
        let got = Fft::new(64).convolve(&a, &b);
        assert_eq!(got.len(), a.len() + b.len() - 1);
        for (k, &g) in got.iter().enumerate() {
            let want: f32 = (0..b.len())
                .filter(|&j| j <= k && k - j < a.len())
                .map(|j| a[k - j] * b[j])
                .sum();
            assert!((g - want).abs() < 1e-3, "sample {k}: {g} vs {want}");
        }
    }

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
            assert!(
                (got - want).abs() < 1e-3 * want.max(1.0),
                "bin {k}: {got} vs {want}"
            );
        }
    }
}
