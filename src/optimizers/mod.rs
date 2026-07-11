use iron_oxide::collections::Matrix;

pub mod adam;
pub mod adamw;
pub mod muon;
pub mod sgd;
pub use self::adam::Adam;
pub use self::adamw::AdamW;
pub use self::muon::Muon;
pub use self::sgd::Sgd;

pub const WEIGHT_DECAY: f32 = 0.03;
// Active optimizer. Swap to `Muon` to orthogonalize the 2D hidden-weight updates
// (embeddings + biases stay on Adam automatically); retune the lr schedule if so.
pub type Optimizer = Adam;

pub trait OptimizerGradTypes {
    type GradMatrix;
    type GradMatrixNoDecay;
    type GradVec;
}

pub trait GradMatrixOps {
    fn zeros(rows: usize, cols: usize) -> Self;
    /// Apply the accumulated gradient to `weights`. `weight_decay` is the
    /// per-step decoupled decay coefficient (λ) for THIS matrix: pass `0.0` to
    /// get plain Adam (no decay), or a positive λ for AdamW-style decay. This is
    /// a runtime argument (not the compile-time `WEIGHT_DECAY` constant) so that
    /// different stacks — e.g. the hierarchical encoder/decoder vs. the backbone
    /// — can be decayed independently while sharing one optimizer type.
    fn apply_to(&mut self, weights: &mut Matrix, lr: f32, weight_decay: f32);
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

/// Add `src`'s accumulated raw gradients into `dst` element-wise. Optimizer
/// moments are untouched — used to reduce per-thread replica gradients.
pub fn add_grad_matrix<G: GradMatrixOps>(dst: &mut G, src: &mut G) {
    let src = src.matrix().as_slice();
    for (d, &s) in dst.matrix().as_slice_mut().iter_mut().zip(src) {
        *d += s;
    }
}

/// Vector counterpart of [`add_grad_matrix`].
pub fn add_grad_vec<G: GradVecOps>(dst: &mut G, src: &mut G) {
    let src = src.vec();
    for (d, s) in dst.vec().iter_mut().zip(src) {
        *d += *s;
    }
}
