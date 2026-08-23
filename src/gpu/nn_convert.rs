//! Conversion between the device layers and the CPU `nn::*` checkpoint format.
//!
//! Checkpoints are written and read as CPU layers, so every GPU layer round-trips
//! through them. The two representations disagree on two points, and both are settled
//! here rather than inside the layer:
//!
//! - a GPU cell holds its projections **fused** (one matrix per group), while the
//!   checkpoint keeps every projection apart — [`concat_cols`] / [`split_cols`];
//! - the `1/√dqk` of `q·kᵀ` is folded into `wk`/`bk` on the device, but the
//!   checkpoint stores the unscaled weights.

use iron_oxide::collections::Matrix;

use super::mlstm::MLstm;
use super::{Gpu, dt_matrix, dt_vec, tensor_from_matrix, tensor_from_slice};
use crate::tensor::Tensor;

/// Lay several `[rows, ·]` weight (or `[·]` bias) tensors side by side into the one
/// matrix a fused projection holds. The inverse of [`split_cols`].
pub(super) fn concat_cols(parts: &[&Tensor]) -> Tensor {
    let rows = if parts[0].dims().len() == 1 {
        1
    } else {
        parts[0].dims()[0]
    };
    let cols: Vec<usize> = parts.iter().map(|p| p.data.len() / rows).collect();
    let total: usize = cols.iter().sum();
    let mut data = Vec::with_capacity(rows * total);
    for r in 0..rows {
        for (p, c) in parts.iter().zip(&cols) {
            data.extend_from_slice(&p.data[r * c..(r + 1) * c]);
        }
    }
    let dims: &[usize] = if parts[0].dims().len() == 1 {
        &[total]
    } else {
        &[rows, total]
    };
    Tensor::new(dims, data)
}

/// Cut column block `[at, at + cols)` back out of a fused projection's matrix.
fn split_cols(m: &Matrix, at: usize, cols: usize) -> Matrix {
    let (rows, w) = (m.rows(), m.cols());
    let src = m.as_slice();
    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        data.extend_from_slice(&src[r * w + at..r * w + at + cols]);
    }
    Matrix::from_vec(data, rows, cols)
}

/// Column block `[at, at + n)` of a fused bias vector.
fn split_vec(v: &[f32], at: usize, n: usize) -> Box<[f32]> {
    v[at..at + n].into()
}

impl MLstm {
    /// Export this cell into the CPU `nn::MLSTMLayer` format. Used to write a
    /// `HIER` checkpoint from a GPU model.
    pub fn to_nn_cell(&self, gpu: &Gpu) -> crate::nn::mlstm::MLSTMLayer {
        let unfold_m = |m: Matrix| {
            let (r, c) = (m.rows(), m.cols());
            let d = m.as_slice().iter().map(|v| v / self.inv_sqrt_dqk).collect();
            Matrix::from_vec(d, r, c)
        };
        let unfold_v =
            |v: Box<[f32]>| -> Box<[f32]> { v.iter().map(|x| x / self.inv_sqrt_dqk).collect() };
        let w_out = crate::nn::linear::LinearLayer::from_loaded(
            self.d,
            self.d,
            dt_matrix(gpu, &self.lin_out.w),
            dt_vec(gpu, &self.lin_out.b),
        );
        let (h, wqk) = (self.heads, self.heads * self.dqk);
        let (xw, xb) = (
            dt_matrix(gpu, &self.lin_qkvo.w),
            dt_vec(gpu, &self.lin_qkvo.b),
        );
        let (gw, gb) = (
            dt_matrix(gpu, &self.lin_gates.w),
            dt_vec(gpu, &self.lin_gates.b),
        );
        let o_at = 2 * wqk + self.d;
        crate::nn::mlstm::MLSTMLayer::from_loaded(
            self.input_size,
            self.d,
            h,
            self.dqk,
            split_cols(&xw, 0, wqk),
            unfold_m(split_cols(&xw, wqk, wqk)),
            split_cols(&xw, 2 * wqk, self.d),
            split_cols(&xw, o_at, self.d),
            split_cols(&gw, 0, h),
            split_cols(&gw, h, h),
            split_vec(&xb, 0, wqk),
            unfold_v(split_vec(&xb, wqk, wqk)),
            split_vec(&xb, 2 * wqk, self.d),
            split_vec(&xb, o_at, self.d),
            split_vec(&gb, 0, h),
            split_vec(&gb, h, h),
            w_out,
            dt_vec(gpu, &self.headnorm.gamma),
        )
    }

    /// Rebuild a GPU cell from a CPU `nn::MLSTMLayer` (inverse of `to_nn_cell`).
    pub fn from_nn_cell(gpu: &Gpu, c: &crate::nn::mlstm::MLSTMLayer) -> Self {
        use tensor_from_matrix as m;
        use tensor_from_slice as v;
        Self::from_parts(
            gpu,
            c.input_size,
            c.hidden_size,
            c.num_heads,
            c.dqk,
            &m(&c.wq),
            &m(&c.wk),
            &m(&c.wv),
            &m(&c.wo),
            &m(&c.wi),
            &m(&c.wf),
            &v(&c.bq),
            &v(&c.bk),
            &v(&c.bv),
            &v(&c.bo),
            &v(&c.bi),
            &v(&c.bf),
            &m(&c.w_out.weights),
            &v(&c.w_out.biases),
            &v(&c.head_norm.gamma),
        )
    }
}
