use iron_oxide::collections::Matrix;

use crate::{
    nn::{sub_in_place, sub_vec_in_place},
    opimizers::{GradMatrixOps, GradVecOps, OptimizerGradTypes},
};

const CLIP: f32 = 10.0;

pub struct Sgd;

impl Sgd {
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
        _m: &mut Matrix,
        _v: &mut Matrix,
        _step: usize,
        lr: f32,
    ) {
        Self::apply_matrix(weights, grads, lr);
    }

    pub fn apply_vec_state(
        weights: &mut [f32],
        grads: &[f32],
        _m: &mut [f32],
        _v: &mut [f32],
        _step: usize,
        lr: f32,
    ) {
        Self::apply_vec(weights, grads, lr);
    }
}

pub struct SgdGradMatrix {
    grads: Matrix,
}

pub struct SgdGradVec {
    grads: Box<[f32]>,
}

// SGD: nur die Grads (Basis-Typen)
impl GradMatrixOps for SgdGradMatrix {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            grads: Matrix::zeros(rows, cols),
        }
    }

    fn apply_to(&mut self, weights: &mut Matrix, lr: f32) {
        Sgd::apply_matrix(weights, &self.grads, lr);
    }

    fn clear(&mut self) {
        self.grads.clear();
    }

    fn clip(&mut self) {
        Sgd::clip_matrix(&mut self.grads);
    }

    fn matrix(&mut self) -> &mut Matrix {
        &mut self.grads
    }
}

impl GradVecOps for SgdGradVec {
    fn zeros(len: usize) -> Self {
        Self {
            grads: vec![0.0; len].into(),
        }
    }

    fn apply_to(&mut self, weights: &mut [f32], lr: f32) {
        Sgd::apply_vec(weights, &self.grads, lr);
    }

    fn clear(&mut self) {
        self.grads.fill(0.0);
    }

    fn clip(&mut self) {
        Sgd::clip_vec(&mut self.grads);
    }

    fn vec(&mut self) -> &mut [f32] {
        &mut self.grads
    }
}

impl OptimizerGradTypes for Sgd {
    type GradMatrix = Matrix;
    type GradVec = Box<[f32]>;
}
