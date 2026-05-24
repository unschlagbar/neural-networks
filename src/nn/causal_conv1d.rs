// Causal depthwise conv1d with Swish activation, as used in the xLSTM paper.
//
// Applied per-timestep. Each channel has its own kernel (depthwise — no cross-channel mixing).
// A causal ring buffer keeps the last `kernel_size` inputs; only past and present tokens contribute.
//
// Forward: z[c] = bias[c] + Σ_{k=0}^{K-1} w[c,k] · x[t-k,c]
//          y[c] = z[c] · σ(z[c])           (Swish)
//
// Backward (BPTT for the delayed gradients):
//   d_pre[c] = δ[c] · σ(z)·(1 + z·(1−σ(z)))
//   dx[t, c] = d_pre[c]·w[c,0] + bptt[0,c]
//   bptt[k,c] ← bptt[k+1,c]_old + d_pre[c]·w[c,k+1]   for k = 0..K-2
// (forward-k order is safe in-place: we always read slot k+1 before we overwrite slot k)

use std::{any::Any, io};

use iron_oxide::collections::Matrix;

use crate::{
    nn_layer::{DynCache, NnLayer},
    optimizers::{GradMatrix, GradMatrixOps, GradVec, GradVecOps},
    saving,
};

#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

pub struct CausalConv1dCache {
    /// Snapshot of the K inputs used this timestep: slot 0 = x[t], slot k = x[t-k].
    /// Layout: inputs[k * channels + c].
    pub inputs: Box<[f32]>,    // kernel_size * channels
    pub pre_swish: Box<[f32]>, // channels
    pub output: Box<[f32]>,    // channels
    pub dx: Box<[f32]>,        // channels
}

impl DynCache for CausalConv1dCache {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn output(&self) -> &[f32] {
        &self.output
    }
    fn input_grad(&self) -> &[f32] {
        &self.dx
    }
}

pub struct CausalConv1dLayer {
    pub channels: usize,
    pub kernel_size: usize,

    /// weights[c, k]: channel c, kernel offset k (0 = current step, 1 = one step back, …).
    /// Matrix layout: rows = channels, cols = kernel_size  →  flat index = c * kernel_size + k.
    pub weights: Matrix,
    pub bias: Box<[f32]>, // channels

    /// Causal ring buffer.  ring[pos * channels + c] = input[c] at ring slot pos.
    ring: Box<[f32]>,
    ring_pos: usize,

    /// BPTT gradient buffer.  dx_bptt[k * channels + c] = gradient contribution
    /// for the input that is (k+1) timesteps back, accumulated from future backward steps.
    /// Size: (kernel_size - 1) * channels.
    dx_bptt: Box<[f32]>,

    pub dw: GradMatrix,
    pub db: GradVec,
    d_pre: Box<[f32]>, // scratch — pre-swish gradient for the current backward step
}

impl CausalConv1dLayer {
    pub fn new(channels: usize, kernel_size: usize) -> Self {
        assert!(kernel_size >= 1, "kernel_size must be >= 1");
        let scale = (1.0 / kernel_size as f32).sqrt();
        let bptt_slots = kernel_size.saturating_sub(1);
        Self {
            channels,
            kernel_size,
            weights: Matrix::random(channels, kernel_size, scale),
            bias: vec![0.0; channels].into(),
            ring: vec![0.0; kernel_size * channels].into(),
            ring_pos: 0,
            dx_bptt: vec![0.0; bptt_slots * channels].into(),
            dw: GradMatrix::zeros(channels, kernel_size),
            db: GradVec::zeros(channels),
            d_pre: vec![0.0; channels].into(),
        }
    }

    pub fn from_loaded(
        channels: usize,
        kernel_size: usize,
        weights: Matrix,
        bias: Box<[f32]>,
    ) -> Self {
        let bptt_slots = kernel_size.saturating_sub(1);
        Self {
            channels,
            kernel_size,
            weights,
            bias,
            ring: vec![0.0; kernel_size * channels].into(),
            ring_pos: 0,
            dx_bptt: vec![0.0; bptt_slots * channels].into(),
            dw: GradMatrix::zeros(channels, kernel_size),
            db: GradVec::zeros(channels),
            d_pre: vec![0.0; channels].into(),
        }
    }

    pub fn forward(&mut self, input: &[f32], cache: &mut CausalConv1dCache) {
        let c = self.channels;
        let k = self.kernel_size;
        let pos = self.ring_pos;

        // Write current input into ring buffer.
        self.ring[pos * c..pos * c + c].copy_from_slice(input);

        // Snapshot K most-recent inputs into the cache (k=0 = most recent, k=K-1 = oldest).
        for ki in 0..k {
            let ring_idx = (pos + k - ki) % k;
            cache.inputs[ki * c..ki * c + c]
                .copy_from_slice(&self.ring[ring_idx * c..ring_idx * c + c]);
        }

        // Depthwise conv followed by Swish.
        let w = self.weights.as_slice();
        for ch in 0..c {
            let mut sum = self.bias[ch];
            for ki in 0..k {
                sum += w[ch * k + ki] * cache.inputs[ki * c + ch];
            }
            cache.pre_swish[ch] = sum;
            let s = sigmoid(sum);
            cache.output[ch] = sum * s;
        }

        self.ring_pos = (pos + 1) % k;
    }

    pub fn backward(&mut self, delta: &mut [f32], cache: &mut CausalConv1dCache) {
        let c = self.channels;
        let k = self.kernel_size;
        let w = self.weights.as_slice();

        // Swish gradient: d/dz [z·σ(z)] = σ(z)·(1 + z·(1 − σ(z)))
        for ch in 0..c {
            let z = cache.pre_swish[ch];
            let s = sigmoid(z);
            self.d_pre[ch] = delta[ch] * s * (1.0 + z * (1.0 - s));
        }

        // Weight and bias gradients.
        {
            let dw = self.dw.matrix().as_slice_mut();
            let db = self.db.vec();
            for ch in 0..c {
                let dp = self.d_pre[ch];
                db[ch] += dp;
                for ki in 0..k {
                    dw[ch * k + ki] += dp * cache.inputs[ki * c + ch];
                }
            }
        }

        // dx for the current input: direct contribution (k=0) plus BPTT accumulation.
        for ch in 0..c {
            let bptt = if k > 1 { self.dx_bptt[ch] } else { 0.0 };
            cache.dx[ch] = self.d_pre[ch] * w[ch * k] + bptt;
        }

        // Shift BPTT buffer one step earlier.
        // dx_bptt[ki, c] ← dx_bptt[ki+1, c]_old + d_pre[c] · w[c, ki+1]
        // Forward ki order is safe: slot ki+1 is read before slot ki is overwritten.
        if k > 1 {
            for ki in 0..k - 1 {
                for ch in 0..c {
                    let old_next = if ki + 1 < k - 1 {
                        self.dx_bptt[(ki + 1) * c + ch]
                    } else {
                        0.0
                    };
                    self.dx_bptt[ki * c + ch] =
                        old_next + self.d_pre[ch] * w[ch * k + ki + 1];
                }
            }
        }
    }
}

impl NnLayer for CausalConv1dLayer {
    fn forward(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<CausalConv1dCache>()
            .expect("CausalConv1dLayer::forward — expected CausalConv1dCache");
        CausalConv1dLayer::forward(self, input, c);
    }

    fn backward(&mut self, delta: &mut [f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<CausalConv1dCache>()
            .expect("CausalConv1dLayer::backward — expected CausalConv1dCache");
        CausalConv1dLayer::backward(self, delta, c);
    }

    fn layer_tag(&self) -> u8 {
        15
    }

    fn save(&self, w: &mut dyn io::Write) -> io::Result<()> {
        saving::write_u32(w, self.kernel_size as u32)?;
        saving::write_matrix(w, &self.weights)?;
        saving::write_f32_slice(w, &self.bias)
    }

    fn make_cache(&self) -> Box<dyn DynCache> {
        Box::new(CausalConv1dCache {
            inputs: vec![0.0; self.kernel_size * self.channels].into(),
            pre_swish: vec![0.0; self.channels].into(),
            output: vec![0.0; self.channels].into(),
            dx: vec![0.0; self.channels].into(),
        })
    }

    fn input_size(&self) -> usize {
        self.channels
    }

    fn output_size(&self) -> usize {
        self.channels
    }

    fn apply_grads(&mut self, lr: f32) {
        self.dw.clip();
        self.db.clip();
        self.dw.apply_to(&mut self.weights, lr);
        self.db.apply_to(&mut self.bias, lr);
    }

    fn clear_grads(&mut self) {
        self.dw.clear();
        self.db.clear();
    }

    fn reset_state(&mut self) {
        self.ring.fill(0.0);
        self.ring_pos = 0;
    }

    fn reset_bptt_state(&mut self) {
        self.dx_bptt.fill(0.0);
    }
}
