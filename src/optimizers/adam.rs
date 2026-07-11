use iron_oxide::collections::Matrix;

use crate::optimizers::{GradMatrixOps, GradVecOps, OptimizerGradTypes};

const CLIP: f32 = 5.0;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.95;
const EPS: f32 = 1e-8;

#[derive(Debug)]
pub struct Adam;

// Adam: 3 Matrizen + step
pub struct AdamGradMatrix {
    grads: Matrix,
    m: Matrix,
    v: Matrix,
    step: usize,
}

pub struct AdamGradVec {
    grads: Box<[f32]>,
    m: Box<[f32]>,
    v: Box<[f32]>,
    step: usize,
}

impl GradMatrixOps for AdamGradMatrix {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            grads: Matrix::zeros(rows, cols),
            m: Matrix::zeros(rows, cols),
            v: Matrix::zeros(rows, cols),
            step: 0,
        }
    }

    fn apply_to(&mut self, weights: &mut Matrix, lr: f32, weight_decay: f32) {
        self.step += 1;
        debug_assert_eq!(weights.rows(), self.grads.rows());
        debug_assert_eq!(weights.cols(), self.grads.cols());
        debug_assert_eq!(self.m.rows(), self.grads.rows());
        debug_assert_eq!(self.m.cols(), self.grads.cols());
        debug_assert_eq!(self.v.rows(), self.grads.rows());
        debug_assert_eq!(self.v.cols(), self.grads.cols());

        self.clip();

        let t = self.step as f32;
        let beta1_t = 1.0 - BETA1.powf(t);
        let beta2_t = 1.0 - BETA2.powf(t);

        // Decoupled weight decay (AdamW): applied directly to the weights, not
        // through the gradient. `weight_decay == 0.0` recovers plain Adam, so a
        // stack can be Adam or AdamW purely by what it passes here.
        let decay = 1.0 - lr * weight_decay;

        for ((w, g), (m_val, v_val)) in weights
            .iter_mut()
            .zip(self.grads.as_slice())
            .zip(self.m.iter_mut().zip(self.v.iter_mut()))
        {
            *m_val = BETA1 * *m_val + (1.0 - BETA1) * *g;
            *v_val = BETA2 * *v_val + (1.0 - BETA2) * g * g;
            let m_hat = *m_val / beta1_t;
            let v_hat = *v_val / beta2_t;
            *w = *w * decay - lr * m_hat / (v_hat.sqrt() + EPS);
        }
    }

    fn clear(&mut self) {
        self.grads.clear();
    }

    fn clip(&mut self) {
        self.grads.clip(-CLIP, CLIP);
    }

    fn matrix(&mut self) -> &mut Matrix {
        &mut self.grads
    }
}

impl GradVecOps for AdamGradVec {
    fn zeros(len: usize) -> Self {
        Self {
            grads: vec![0.0; len].into(),
            m: vec![0.0; len].into(),
            v: vec![0.0; len].into(),
            step: 0,
        }
    }

    fn apply_to(&mut self, weights: &mut [f32], lr: f32) {
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

    fn clear(&mut self) {
        self.grads.fill(0.0);
    }

    fn clip(&mut self) {
        self.grads
            .iter_mut()
            .for_each(|v| *v = v.clamp(-CLIP, CLIP));
    }
    fn vec(&mut self) -> &mut [f32] {
        &mut self.grads
    }
}

impl OptimizerGradTypes for Adam {
    type GradMatrix = AdamGradMatrix;
    type GradMatrixNoDecay = AdamGradMatrix;
    type GradVec = AdamGradVec;
}
