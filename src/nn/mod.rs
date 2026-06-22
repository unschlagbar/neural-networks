use iron_oxide::collections::Matrix;

pub mod activations;
pub mod causal_conv1d;
pub mod dropout;
pub mod embedding;
pub mod headwise_rms_norm;
pub mod linear;
pub mod linear_nb;
pub mod lstm;
pub mod mlstm;
pub mod mlstm_block;
pub mod rms_norm;
pub mod silu_dense;
pub mod slstm;
pub mod slstm_block;
pub mod softmax;

pub fn add_vec_in_place(x: &mut [f32], y: &[f32]) {
    debug_assert_eq!(x.len(), y.len());
    x.iter_mut().zip(y).for_each(|(x, y)| *x += y);
}

pub fn sub_in_place(a: &mut Matrix, b: &Matrix, lr: f32) {
    debug_assert_eq!(a.rows(), b.rows());
    debug_assert_eq!(a.cols(), b.cols());
    a.as_slice_mut()
        .iter_mut()
        .zip(b.as_slice())
        .for_each(|(a, b)| *a -= lr * b);
}

pub fn sub_vec_in_place(a: &mut [f32], b: &[f32], lr: f32) {
    a.iter_mut().zip(b).for_each(|(a, b)| *a -= lr * b);
}

pub fn one_hot(index: usize, size: usize) -> Vec<f32> {
    let mut out = vec![0.0; size];
    out[index] = 1.0;
    out
}
