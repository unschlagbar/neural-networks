// headwise_rms_norm.rs — RMS normalization applied independently per head slice.
//
// For x ∈ ℝ^{d} split into H heads of size dhv = d/H:
//   rms_h  = sqrt( mean(x_h²) + ε )
//   out_h  = gamma_h ⊙ x_h / rms_h
//
// gamma is a learnable per-element scale (d elements, init 1).
// Each head is normalized independently (grouped norm, group size = dhv).

use crate::optimizers::{GradVec, GradVecOps};

const EPS: f32 = 1e-6;

pub struct HeadwiseRMSNormCache {
    pub x_hat: Box<[f32]>,   // (d)  x / rms, reused in backward
    pub inv_rms: Box<[f32]>, // (H)  per-head 1/rms
    pub output: Box<[f32]>,  // (d)  normed output
    pub dx: Box<[f32]>,      // (d)  dL/d(input)
}

impl HeadwiseRMSNormCache {
    pub fn new(hidden_size: usize, num_heads: usize) -> Self {
        Self {
            x_hat: vec![0.0; hidden_size].into(),
            inv_rms: vec![0.0; num_heads].into(),
            output: vec![0.0; hidden_size].into(),
            dx: vec![0.0; hidden_size].into(),
        }
    }
}

pub struct HeadwiseRMSNorm {
    pub gamma: Box<[f32]>,
    pub grads_gamma: GradVec,
    pub num_heads: usize,
    pub dhv: usize,
}

impl HeadwiseRMSNorm {
    pub fn new(hidden_size: usize, num_heads: usize) -> Self {
        assert_eq!(hidden_size % num_heads, 0);
        Self {
            gamma: vec![1.0; hidden_size].into(),
            grads_gamma: GradVec::zeros(hidden_size),
            num_heads,
            dhv: hidden_size / num_heads,
        }
    }

    pub fn from_loaded(hidden_size: usize, num_heads: usize, gamma: Box<[f32]>) -> Self {
        assert_eq!(hidden_size % num_heads, 0);
        Self {
            grads_gamma: GradVec::zeros(hidden_size),
            gamma,
            num_heads,
            dhv: hidden_size / num_heads,
        }
    }

    pub fn alloc_cache(&self) -> HeadwiseRMSNormCache {
        HeadwiseRMSNormCache::new(self.num_heads * self.dhv, self.num_heads)
    }

    /// Forward: `out_h = gamma_h ⊙ x_h / rms_h` for each head h.
    pub fn forward_into(&self, input: &[f32], cache: &mut HeadwiseRMSNormCache) {
        let dhv = self.dhv;
        for hd in 0..self.num_heads {
            let off = hd * dhv;
            let sq_sum: f32 = input[off..off + dhv].iter().map(|&x| x * x).sum();
            let inv_rms = 1.0 / (sq_sum / dhv as f32 + EPS).sqrt();
            cache.inv_rms[hd] = inv_rms;
            for i in 0..dhv {
                cache.x_hat[off + i] = input[off + i] * inv_rms;
                cache.output[off + i] = self.gamma[off + i] * cache.x_hat[off + i];
            }
        }
    }

    /// Backward: given `delta` = dL/d(output), writes dL/d(input) into `cache.dx`.
    ///
    ///   S[h]    = Σⱼ gamma_h[j] · delta_h[j] · x̂_h[j]
    ///   dx_h[i] = inv_rms_h · ( gamma_h[i]·delta_h[i] − x̂_h[i]·S[h]/dhv )
    pub fn backward_into(&mut self, delta: &[f32], cache: &mut HeadwiseRMSNormCache) {
        let dhv = self.dhv;
        for hd in 0..self.num_heads {
            let off = hd * dhv;
            let inv_rms = cache.inv_rms[hd];
            let grads_gamma = self.grads_gamma.vec();
            let mut s = 0.0;
            for i in 0..dhv {
                let dyxh = delta[off + i] * cache.x_hat[off + i];
                grads_gamma[off + i] += dyxh;
                s += self.gamma[off + i] * dyxh;
            }
            let s_n = s / dhv as f32;
            for i in 0..dhv {
                cache.dx[off + i] =
                    inv_rms * (self.gamma[off + i] * delta[off + i] - cache.x_hat[off + i] * s_n);
            }
        }
    }

    pub fn apply_grads(&mut self, lr: f32) {
        self.grads_gamma.apply_to(&mut self.gamma, lr);
    }

    pub fn clear_grads(&mut self) {
        self.grads_gamma.clear();
    }
}
