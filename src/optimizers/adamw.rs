use iron_oxide::collections::Matrix;

use crate::optimizers::{
    GradMatrixOps, OptimizerGradTypes,
    adam::{AdamGradMatrix, AdamGradVec},
};

const CLIP: f32 = 5.0;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.95;
const EPS: f32 = 1e-8;
/// Decoupled weight decay (λ). Applied directly to weights, NOT through gradients.
/// Typical range for LLM pretraining: 0.01–0.1.
/// Weight decay is intentionally NOT applied to bias vectors (see AdamWGradVec).

#[derive(Debug)]
pub struct AdamW;

pub struct AdamWGradMatrix {
    grads: Matrix,
    m: Matrix,
    v: Matrix,
    step: usize,
}

// Kein Weight Decay auf Biases – das ist Best Practice (GPT-2, LLaMA etc.).

impl GradMatrixOps for AdamWGradMatrix {
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

        // AdamW: Decoupled weight decay.
        //
        // Instead of L2 regularisation through the gradient (classic Adam+L2):
        //   g_t ← g_t + λ·w          (wrong: decay passes through momentum)
        //
        // weight decay is applied directly to the weights:
        //   w ← w · (1 - lr·λ) − lr · m̂ / (√v̂ + ε)
        //
        // Referenz: Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2019)
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

impl OptimizerGradTypes for AdamW {
    type GradMatrix = AdamWGradMatrix;
    type GradMatrixNoDecay = AdamGradMatrix;
    type GradVec = AdamGradVec;
}
