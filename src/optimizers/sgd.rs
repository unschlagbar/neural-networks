use iron_oxide::collections::Matrix;

use crate::{
    nn::{sub_in_place, sub_vec_in_place},
    optimizers::{GradMatrixOps, GradVecOps, OptimizerGradTypes},
};

const CLIP: f32 = 30.0;

#[derive(Debug)]
pub struct Sgd;

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
        sub_in_place(weights, &self.grads, lr);
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

impl GradVecOps for SgdGradVec {
    fn zeros(len: usize) -> Self {
        Self {
            grads: vec![0.0; len].into(),
        }
    }

    fn apply_to(&mut self, weights: &mut [f32], lr: f32) {
        sub_vec_in_place(weights, &self.grads, lr);
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

impl OptimizerGradTypes for Sgd {
    type GradMatrix = SgdGradMatrix;
    type GradVec = SgdGradVec;
}
