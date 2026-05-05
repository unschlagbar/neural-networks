use iron_oxide::collections::Matrix;

pub mod adam;
pub mod sgd;
pub use self::adam::Adam;
pub use self::sgd::Sgd;
pub type Optimizer = Adam;

// ── Optimizer Grad Types (statisch bestimmt zur Compile-Zeit) ────────────────

pub trait OptimizerGradTypes {
    type GradMatrix;
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

// Type-Aliase für Layer
pub type OptGradMatrix = <Optimizer as OptimizerGradTypes>::GradMatrix;
pub type OptGradVec = <Optimizer as OptimizerGradTypes>::GradVec;
