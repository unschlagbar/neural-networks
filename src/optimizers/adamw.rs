use crate::optimizers::{
    OptimizerGradTypes,
    adam::{AdamGradMatrix, AdamGradVec},
};

/// Adam with decoupled weight decay on the hidden matrices; biases and the
/// no-decay tables are not decayed. The decay itself lives in
/// [`AdamGradMatrix::apply_to`], which takes the coefficient per call.
#[derive(Debug)]
pub struct AdamW;

impl OptimizerGradTypes for AdamW {
    type GradMatrix = AdamGradMatrix;
    type GradMatrixNoDecay = AdamGradMatrix;
    type GradVec = AdamGradVec;
}
