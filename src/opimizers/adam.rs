use iron_oxide::collections::Matrix;

use crate::{
    nn::{sub_in_place, sub_vec_in_place},
    opimizers::{GradMatrixOps, GradVecOps, OptimizerGradTypes},
};

const CLIP: f32 = 15.0;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPS: f32 = 1e-8;

pub struct Adam;

impl Adam {
    pub fn clip_matrix(grads: &mut Matrix) {
        grads
            .as_slice_mut()
            .iter_mut()
            .for_each(|v| *v = v.clamp(-CLIP, CLIP));
    }

    pub fn clip_vec(grads: &mut [f32]) {
        grads.iter_mut().for_each(|v| *v = v.clamp(-CLIP, CLIP));
    }

    pub fn apply_matrix(weights: &mut Matrix, grads: &Matrix, lr: f32) {
        sub_in_place(weights, grads, lr);
    }

    pub fn apply_vec(weights: &mut [f32], grads: &[f32], lr: f32) {
        sub_vec_in_place(weights, grads, lr);
    }

    pub fn apply_matrix_state(
        weights: &mut Matrix,
        grads: &Matrix,
        m: &mut Matrix,
        v: &mut Matrix,
        step: usize,
        lr: f32,
    ) {
        debug_assert_eq!(weights.rows(), grads.rows());
        debug_assert_eq!(weights.cols(), grads.cols());
        debug_assert_eq!(m.rows(), grads.rows());
        debug_assert_eq!(m.cols(), grads.cols());
        debug_assert_eq!(v.rows(), grads.rows());
        debug_assert_eq!(v.cols(), grads.cols());

        let t = step as f32;
        let beta1_t = 1.0 - BETA1.powf(t);
        let beta2_t = 1.0 - BETA2.powf(t);

        for ((w, g), (m_val, v_val)) in weights
            .as_slice_mut()
            .iter_mut()
            .zip(grads.as_slice())
            .zip(m.as_slice_mut().iter_mut().zip(v.as_slice_mut()))
        {
            *m_val = BETA1 * *m_val + (1.0 - BETA1) * *g;
            *v_val = BETA2 * *v_val + (1.0 - BETA2) * g * g;
            let m_hat = *m_val / beta1_t;
            let v_hat = *v_val / beta2_t;
            *w -= lr * m_hat / (v_hat.sqrt() + EPS);
        }
    }

    pub fn apply_vec_state(
        weights: &mut [f32],
        grads: &[f32],
        m: &mut [f32],
        v: &mut [f32],
        step: usize,
        lr: f32,
    ) {
        debug_assert_eq!(weights.len(), grads.len());
        debug_assert_eq!(m.len(), grads.len());
        debug_assert_eq!(v.len(), grads.len());

        let t = step as f32;
        let beta1_t = 1.0 - BETA1.powf(t);
        let beta2_t = 1.0 - BETA2.powf(t);

        for (((w, g), m_val), v_val) in weights
            .iter_mut()
            .zip(grads)
            .zip(m.iter_mut())
            .zip(v.iter_mut())
        {
            *m_val = BETA1 * *m_val + (1.0 - BETA1) * *g;
            *v_val = BETA2 * *v_val + (1.0 - BETA2) * g * g;
            let m_hat = *m_val / beta1_t;
            let v_hat = *v_val / beta2_t;
            *w -= lr * m_hat / (v_hat.sqrt() + EPS);
        }
    }
}

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

    fn apply_to(&mut self, weights: &mut Matrix, lr: f32) {
        self.step += 1;
        Adam::apply_matrix_state(
            weights,
            &self.grads,
            &mut self.m,
            &mut self.v,
            self.step,
            lr,
        );
    }

    fn clear(&mut self) {
        self.grads.clear();
    }

    fn clip(&mut self) {
        Adam::clip_matrix(&mut self.grads);
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
        Adam::apply_vec_state(
            weights,
            &self.grads,
            &mut self.m,
            &mut self.v,
            self.step,
            lr,
        );
    }

    fn clear(&mut self) {
        self.grads.fill(0.0);
    }

    fn clip(&mut self) {
        Adam::clip_vec(&mut self.grads);
    }
    fn vec(&mut self) -> &mut [f32] {
        &mut self.grads
    }
}

impl OptimizerGradTypes for Adam {
    type GradMatrix = AdamGradMatrix;
    type GradVec = AdamGradVec;
}
