//! Adam with decoupled weight decay, per tensor, for the `Sequential` layers.
//! Biases and norm scales are never decayed; matrices take λ per call.

use iron_oxide::collections::Matrix;

const CLIP: f32 = 5.0;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.95;
const EPS: f32 = 1e-8;

/// Gradient and moments are allocated on first use, so a layer that is only built
/// to be serialized (checkpoint save/load) holds its weights and nothing else.
pub struct GradMatrix {
    rows: usize,
    cols: usize,
    grads: Option<Matrix>,
    moments: Option<(Matrix, Matrix)>,
    step: usize,
}

pub struct GradVec {
    grads: Box<[f32]>,
    m: Box<[f32]>,
    v: Box<[f32]>,
    step: usize,
}

impl GradMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            grads: None,
            moments: None,
            step: 0,
        }
    }

    /// One Adam step on `weights`; `weight_decay` is the decoupled decay (λ),
    /// `0.0` is plain Adam.
    pub fn apply_to(&mut self, weights: &mut Matrix, lr: f32, weight_decay: f32) {
        self.step += 1;
        debug_assert_eq!(weights.rows(), self.rows);
        debug_assert_eq!(weights.cols(), self.cols);

        self.clip();
        let (rows, cols) = (self.rows, self.cols);
        let g = self.grads.get_or_insert_with(|| Matrix::zeros(rows, cols));
        let (m, v) = self
            .moments
            .get_or_insert_with(|| (Matrix::zeros(rows, cols), Matrix::zeros(rows, cols)));

        let t = self.step as f32;
        let beta1_t = 1.0 - BETA1.powf(t);
        let beta2_t = 1.0 - BETA2.powf(t);

        let decay = 1.0 - lr * weight_decay;

        for ((w, g), (m_val, v_val)) in weights
            .iter_mut()
            .zip(g.as_slice())
            .zip(m.iter_mut().zip(v.iter_mut()))
        {
            *m_val = BETA1 * *m_val + (1.0 - BETA1) * *g;
            *v_val = BETA2 * *v_val + (1.0 - BETA2) * g * g;
            let m_hat = *m_val / beta1_t;
            let v_hat = *v_val / beta2_t;
            *w = *w * decay - lr * m_hat / (v_hat.sqrt() + EPS);
        }
    }

    pub fn clear(&mut self) {
        if let Some(g) = &mut self.grads {
            g.clear();
        }
    }

    pub fn clip(&mut self) {
        if let Some(g) = &mut self.grads {
            g.clip(-CLIP, CLIP);
        }
    }

    pub fn matrix(&mut self) -> &mut Matrix {
        let (rows, cols) = (self.rows, self.cols);
        self.grads.get_or_insert_with(|| Matrix::zeros(rows, cols))
    }
}

impl GradVec {
    pub fn zeros(len: usize) -> Self {
        Self {
            grads: vec![0.0; len].into(),
            m: vec![0.0; len].into(),
            v: vec![0.0; len].into(),
            step: 0,
        }
    }

    pub fn apply_to(&mut self, weights: &mut [f32], lr: f32) {
        self.step += 1;
        debug_assert_eq!(weights.len(), self.grads.len());
        debug_assert_eq!(self.m.len(), self.grads.len());
        debug_assert_eq!(self.v.len(), self.grads.len());

        self.clip();

        let t = self.step as f32;
        let beta1_t = 1.0 - BETA1.powf(t);
        let beta2_t = 1.0 - BETA2.powf(t);

        for (((w, g), m_val), v_val) in weights
            .iter_mut()
            .zip(&self.grads)
            .zip(self.m.iter_mut())
            .zip(self.v.iter_mut())
        {
            *m_val = BETA1 * *m_val + (1.0 - BETA1) * g;
            *v_val = BETA2 * *v_val + (1.0 - BETA2) * g * g;
            let m_hat = *m_val / beta1_t;
            let v_hat = *v_val / beta2_t;
            *w -= lr * m_hat / (v_hat.sqrt() + EPS);
        }
    }

    pub fn clear(&mut self) {
        self.grads.fill(0.0);
    }

    pub fn clip(&mut self) {
        self.grads
            .iter_mut()
            .for_each(|v| *v = v.clamp(-CLIP, CLIP));
    }
    pub fn vec(&mut self) -> &mut [f32] {
        &mut self.grads
    }
}
