use iron_oxide::collections::Matrix;

use crate::{
    activations::Activate,
    lstm::{CLIP, sub_in_place, sub_vec_in_place},
    nn_layer::{DynCache, NnLayer},
    saving,
};
use rand::{RngExt, rng};
use std::{any::Any, io};

// ── IndRNNCache ───────────────────────────────────────────────────────────────

/// Per-timestep cache (genau wie bei LSTM / Dense – keine Allokation im Hot-Path).
pub struct IndRNNCache {
    pub input_size: usize,
    /// x_t (für W-Gradient und dx)
    pub x: Box<[f32]>,
    /// h_{t-1} (für u-Gradient und BPTT)
    pub h_prev: Box<[f32]>,
    /// h_t = activation(z_t)  → wird im Backward in-place zur Ableitung überschrieben
    pub h: Box<[f32]>,
    /// dL/d(x_t) – wird an die vorherige Schicht weitergereicht
    pub dx: Box<[f32]>,
}

impl DynCache for IndRNNCache {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn output(&self) -> &[f32] {
        &self.h
    }
    fn input_grad(&self) -> &[f32] {
        &self.dx
    }
}

// ── IndRNNLayerGrads ──────────────────────────────────────────────────────────

pub struct IndRNNLayerGrads {
    pub w: Matrix,     // Input-Gewichte
    pub u: Box<[f32]>, // Diagonale Rekurrenz-Gewichte (IndRNN-spezifisch)
    pub b: Box<[f32]>,
}

impl IndRNNLayerGrads {
    pub fn zeros(input_size: usize, hidden_size: usize) -> Self {
        Self {
            w: Matrix::zeros(input_size, hidden_size),
            u: vec![0.0; hidden_size].into(),
            b: vec![0.0; hidden_size].into(),
        }
    }
}

// ── IndRNNLayer (generisch wie DenseLayer<A>) ────────────────────────────────

pub struct IndRNNLayer<A: Activate> {
    pub input_size: usize,
    pub hidden_size: usize,

    pub w: Matrix,
    pub u: Box<[f32]>,
    pub b: Box<[f32]>,
    pub activation: A,

    /// Forward hidden state (wird zum nächsten Timestep getragen)
    pub h: Box<[f32]>,
    /// BPTT-Gradient dL/dh_{t-1}
    pub dh_bptt: Box<[f32]>,

    /// Gradient-Akkumulatoren
    pub grads: IndRNNLayerGrads,
}

impl<A: Activate> IndRNNLayer<A> {
    /// Neue Konstruktor-Signatur – genau wie `DenseLayer::new`
    pub fn new(input_size: usize, hidden_size: usize, activation: A) -> Self {
        let mut rng = rng();
        let h = hidden_size;

        // Gewichte exakt wie bei LSTM (kleine Initialisierung für Rekurrenz)
        let w = Matrix::random(input_size, h, 0.08);
        let u = (0..h).map(|_| rng.random_range(-1.0..1.0)).collect(); // ✓

        Self {
            input_size,
            hidden_size: h,
            w,
            u,
            b: vec![0.0; h].into(),
            activation,
            h: vec![0.0; h].into(),
            dh_bptt: vec![0.0; h].into(),
            grads: IndRNNLayerGrads::zeros(input_size, h),
        }
    }

    pub fn from_loaded(
        input_size: usize,
        hidden_size: usize,
        activation: A,
        w: Matrix,
        u: Box<[f32]>,
        b: Box<[f32]>,
    ) -> Self {
        debug_assert_eq!(w.rows(), input_size);
        debug_assert_eq!(w.cols(), hidden_size);
        debug_assert_eq!(u.len(), hidden_size);
        debug_assert_eq!(b.len(), hidden_size);

        Self {
            input_size,
            hidden_size,
            w,
            u,
            b,
            activation,
            h: vec![0.0; hidden_size].into_boxed_slice(),
            dh_bptt: vec![0.0; hidden_size].into_boxed_slice(),
            grads: IndRNNLayerGrads::zeros(input_size, hidden_size),
        }
    }

    pub fn reset(&mut self) {
        self.h.fill(0.0);
        self.dh_bptt.fill(0.0);
    }

    pub fn forward(&mut self, input: &[f32], cache: &mut IndRNNCache) {
        let h = self.hidden_size;

        cache.h_prev.copy_from_slice(&self.h);
        cache.x.copy_from_slice(input);

        // z = W·x + b + u ⊙ h_{t-1}
        self.w.row_mul(&cache.x, &mut cache.h);
        for j in 0..h {
            cache.h[j] += self.b[j] + self.u[j] * cache.h_prev[j];
        }

        // Aktivierung (generisch!)
        self.activation.activate(&mut cache.h);

        self.h.copy_from_slice(&cache.h);
    }

    pub fn backward(&mut self, delta: &mut [f32], cache: &mut IndRNNCache) {
        let h = self.hidden_size;
        let input_size = self.input_size;

        // Jetzt ist `delta` = volle dL/dh_t (lokaler + zukünftiger Anteil)

        // delta = dL/dh_t  →  nach Derivative wird es dL/dz
        self.activation.derivative_active(&mut cache.h);
        delta
            .iter_mut()
            .zip(&cache.h)
            .for_each(|(d, deriv)| *d *= *deriv);
        // jetzt ist `delta` = dL/dz (voll)

        let g = &mut self.grads;

        // W-Gradient
        for i in 0..input_size {
            let xi = cache.x[i];
            for j in 0..h {
                g.w[i][j] += xi * delta[j];
            }
        }

        // u- und b-Gradient (diagonal)
        for j in 0..h {
            g.u[j] += cache.h_prev[j] * delta[j];
            g.b[j] += delta[j];
        }

        // dx für vorherige Schicht
        for i in 0..input_size {
            let mut s = 0.0;
            for j in 0..h {
                s += self.w[i][j] * delta[j];
            }
            cache.dx[i] = s;
        }

        // BPTT: dh_{t-1} = u ⊙ (dL/dz)  ← jetzt mit vollem Gradienten
        for j in 0..h {
            self.dh_bptt[j] = self.u[j] * delta[j];
        }
    }

    pub fn alloc_cache(&self) -> IndRNNCache {
        let h = self.hidden_size;
        let i = self.input_size;
        IndRNNCache {
            input_size: i,
            x: vec![0.0; i].into(),
            h_prev: vec![0.0; h].into(),
            h: vec![0.0; h].into(),
            dx: vec![0.0; i].into(),
        }
    }
}

// ── impl NnLayer für generische IndRNNLayer ───────────────────────────────────

impl<A: Activate> NnLayer for IndRNNLayer<A> {
    fn forward(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<IndRNNCache>()
            .expect("IndRNNLayer::forward — expected IndRNNCache");
        IndRNNLayer::forward(self, input, c);
    }

    fn backward(&mut self, delta: &mut [f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<IndRNNCache>()
            .expect("IndRNNLayer::backward — expected IndRNNCache");
        IndRNNLayer::backward(self, delta, c);
    }

    fn layer_tag(&self) -> u8 {
        2
    } // TAG_INDRNN

    /// Schreibt NUR die Gewichtsmatrizen w, u, b.
    fn save(&self, w: &mut dyn io::Write) -> io::Result<()> {
        saving::write_u8(w, self.activation.activation_id())?;
        saving::write_matrix(w, &self.w)?;
        saving::write_f32_slice(w, &self.u)?;
        saving::write_f32_slice(w, &self.b)
    }

    fn make_cache(&self) -> Box<dyn DynCache> {
        Box::new(IndRNNLayer::alloc_cache(self))
    }

    fn input_size(&self) -> usize {
        self.input_size
    }
    fn output_size(&self) -> usize {
        self.hidden_size
    }

    fn apply_grads(&mut self, lr: f32) {
        sub_in_place(&mut self.w, &self.grads.w, lr);
        sub_vec_in_place(&mut self.u, &self.grads.u, lr);
        self.u.iter_mut().for_each(|x| *x = x.clamp(-1.0, 1.0));
        sub_vec_in_place(&mut self.b, &self.grads.b, lr);
    }

    fn clear_grads(&mut self) {
        self.grads.w.clear();
        self.grads.u.fill(0.0);
        self.grads.b.fill(0.0);
    }

    fn scale_grads(&mut self, scale: f32) {
        self.grads.w.scale(scale);
        self.grads.u.iter_mut().for_each(|x| *x *= scale);
        self.grads.b.iter_mut().for_each(|x| *x *= scale);
    }

    fn clip_grads(&mut self) {
        self.grads.w.clip(-CLIP, CLIP);
        self.grads
            .u
            .iter_mut()
            .for_each(|x| *x = x.clamp(-CLIP, CLIP));
        self.grads
            .b
            .iter_mut()
            .for_each(|x| *x = x.clamp(-CLIP, CLIP));
    }

    fn reset_state(&mut self) {
        self.reset();
    }

    fn bptt_hidden_grad(&mut self) -> Option<&[f32]> {
        Some(&self.dh_bptt)
    }
}
