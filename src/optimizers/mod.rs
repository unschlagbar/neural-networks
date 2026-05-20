use iron_oxide::collections::Matrix;

pub mod adam;
pub mod adamw;
pub mod sgd;
pub use self::adam::Adam;
pub use self::adamw::AdamW;
pub use self::sgd::Sgd;
pub type Optimizer = AdamW;

pub trait OptimizerGradTypes {
    type GradMatrix;
    type GradMatrixNoDecay;
    type GradVec;
}

pub trait GradMatrixOps {
    fn zeros(rows: usize, cols: usize) -> Self;
    fn apply_to(&mut self, weights: &mut Matrix, lr: f32);
    fn clear(&mut self);
    fn clip(&mut self);
    fn matrix(&mut self) -> &mut Matrix;
}

pub trait GradVecOps {
    fn zeros(len: usize) -> Self;
    fn apply_to(&mut self, weights: &mut [f32], lr: f32);
    fn clear(&mut self);
    fn clip(&mut self);
    fn vec(&mut self) -> &mut [f32];
}

pub type GradMatrix = <Optimizer as OptimizerGradTypes>::GradMatrix;
pub type GradMatrixNoDecay = <Optimizer as OptimizerGradTypes>::GradMatrixNoDecay;
pub type GradVec = <Optimizer as OptimizerGradTypes>::GradVec;
