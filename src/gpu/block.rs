//! Device-resident xLSTM-style residual block, the GPU counterpart of
//! [`nn2::block::Block`](crate::nn2::block::Block).
//!
//!   z = x + post_cell_norm(cell(pre_norm1(x)))
//!   y = z + lin_down( SiLU(lin_gate·pre_norm2(z)) ⊙ (lin_value·pre_norm2(z)) )
//!
//! The norms and the SwiGLU MLP are position-wise and run on the flattened
//! `[N, H]` view (`N = B·T`); only the recurrent `cell` sees the `[B, T, H]`
//! sequence. Since a `DTensor` is contiguous row-major, the `[B,T,H] ↔ [N,H]`
//! reshapes are metadata-only (`DTensor::reshaped`), no copy. The block composes
//! the already-parity-tested `gpu::Linear` / `gpu::RmsNorm` sub-layers plus three
//! small elementwise kernels (`add`, `swiglu_forward`, `swiglu_backward`) around
//! a generic GPU `Cell`.

use super::{
    Buf, DTensor, Gpu, Pool, linear::Linear, mlstm::MLstm, ops, rms_norm::RmsNorm, slstm::SLstm,
};
use crate::{
    nn::{linear::LinearLayer, rms_norm::RMSNorm, slstm_block::SLSTMBlock},
    nn_layer::NnLayer,
    nn2::optim::AdamCfg,
    tensor::Tensor,
};

/// A recurrent cell operating on `[B, T, H]` device sequences (H in == H out).
pub trait Cell {
    fn forward(&mut self, gpu: &Gpu, x: &DTensor, out: &mut DTensor);
    fn backward(&mut self, gpu: &Gpu, dy: &DTensor, dx: &mut DTensor);
    fn zero_grad(&mut self, gpu: &Gpu);
    fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg);
    /// Learnable tensors in a fixed order (checkpoint save/load).
    fn params_mut(&mut self) -> Vec<&mut DTensor>;
    /// Whether the surrounding block applies a `post_cell_norm` before the
    /// residual. sLSTM does; mLSTM doesn't (see `nn2::block::Cell`).
    fn wants_post_cell_norm(&self) -> bool;
    /// Build the matching CPU `nn` block (`SLSTMBlock` / `MLSTMBlock`) from this
    /// cell plus the already-exported surrounding norms and projections.
    #[allow(clippy::too_many_arguments)]
    fn to_nn_block(
        &self,
        gpu: &Gpu,
        hidden: usize,
        up: usize,
        pre_norm1: RMSNorm,
        post_cell_norm: Option<RMSNorm>,
        pre_norm2: RMSNorm,
        lin_gate: LinearLayer,
        lin_value: LinearLayer,
        lin_down: LinearLayer,
    ) -> Box<dyn NnLayer>;
}

impl Cell for SLstm {
    fn forward(&mut self, gpu: &Gpu, x: &DTensor, out: &mut DTensor) {
        SLstm::forward(self, gpu, x, out)
    }
    fn backward(&mut self, gpu: &Gpu, dy: &DTensor, dx: &mut DTensor) {
        SLstm::backward(self, gpu, dy, dx)
    }
    fn zero_grad(&mut self, gpu: &Gpu) {
        SLstm::zero_grad(self, gpu)
    }
    fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        SLstm::step(self, gpu, cfg)
    }
    fn params_mut(&mut self) -> Vec<&mut DTensor> {
        SLstm::params_mut(self)
    }
    fn wants_post_cell_norm(&self) -> bool {
        true
    }
    fn to_nn_block(
        &self,
        gpu: &Gpu,
        hidden: usize,
        up: usize,
        pre_norm1: RMSNorm,
        post_cell_norm: Option<RMSNorm>,
        pre_norm2: RMSNorm,
        lin_gate: LinearLayer,
        lin_value: LinearLayer,
        lin_down: LinearLayer,
    ) -> Box<dyn NnLayer> {
        let post = post_cell_norm.expect("sLSTM block requires a post_cell_norm");
        Box::new(SLSTMBlock::from_loaded(
            hidden,
            up,
            pre_norm1,
            post,
            pre_norm2,
            self.to_nn_cell(gpu),
            lin_gate,
            lin_value,
            lin_down,
        ))
    }
}

/// Type-erased `Block`, so a model can hold a heterogeneous stack (alternating
/// sLSTM / mLSTM blocks) as `Vec<Box<dyn BlockLike>>`. `Block<C>` is generic over
/// its cell, so the concrete types differ; this is the common interface.
pub trait BlockLike {
    fn forward(&mut self, gpu: &Gpu, x: &DTensor, out: &mut DTensor);
    fn backward(&mut self, gpu: &Gpu, dy: &DTensor, dx: &mut DTensor);
    /// Forward into a freshly allocated `[B, T, H]`. Blocks are H-in == H-out, so
    /// the shape follows the input. For benchmarks and one-shot call sites; a
    /// training loop passes its own buffer to [`forward`](Self::forward).
    fn forward_alloc(&mut self, gpu: &Gpu, x: &DTensor) -> DTensor {
        let mut y = DTensor::uninit(gpu, x.dims());
        self.forward(gpu, x, &mut y);
        y
    }
    /// Backward into a freshly allocated `dx`, shaped like `dy`.
    fn backward_alloc(&mut self, gpu: &Gpu, dy: &DTensor) -> DTensor {
        let mut dx = DTensor::uninit(gpu, dy.dims());
        self.backward(gpu, dy, &mut dx);
        dx
    }
    fn zero_grad(&mut self, gpu: &Gpu);
    fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg);
    /// Learnable tensors in a fixed order (checkpoint save/load).
    fn params_mut(&mut self) -> Vec<&mut DTensor>;
    /// Export the block into the matching CPU `nn` block (`SLSTMBlock` /
    /// `MLSTMBlock`) for a `HIER` checkpoint.
    fn to_nn_layer(&mut self, gpu: &Gpu) -> Box<dyn NnLayer>;
}

impl<C: Cell> BlockLike for Block<C> {
    fn forward(&mut self, gpu: &Gpu, x: &DTensor, out: &mut DTensor) {
        Block::forward(self, gpu, x, out)
    }
    fn backward(&mut self, gpu: &Gpu, dy: &DTensor, dx: &mut DTensor) {
        Block::backward(self, gpu, dy, dx)
    }
    fn zero_grad(&mut self, gpu: &Gpu) {
        Block::zero_grad(self, gpu)
    }
    fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        Block::step(self, gpu, cfg)
    }
    fn params_mut(&mut self) -> Vec<&mut DTensor> {
        Block::params_mut(self)
    }
    fn to_nn_layer(&mut self, gpu: &Gpu) -> Box<dyn NnLayer> {
        Block::to_nn_layer(self, gpu)
    }
}

pub struct Block<C: Cell> {
    pub hidden: usize,
    pub up: usize,
    pub pre_norm1: RmsNorm,
    pub cell: C,
    /// Present only when `cell.wants_post_cell_norm()` (sLSTM); `None` for mLSTM.
    pub post_cell_norm: Option<RmsNorm>,
    pub pre_norm2: RmsNorm,
    pub lin_gate: Linear,
    pub lin_value: Linear,
    pub lin_down: Linear,

    /// This block's activations, owned across calls.
    act: Act,
    seq: (usize, usize), // (B, T) of the last forward
}

/// A block's activations.
///
/// Only three values have to survive the forward: the SwiGLU operands its
/// backward differentiates. Those get a permanent [`Buf`] each. Everything else
/// — the residual chain, the norm outputs, every `d_*` — is a temporary consumed
/// within the same call, and lives in a [`Pool`] instead.
///
/// That split is what keeps the memory honest. Giving all 23 intermediates a
/// permanent buffer pins every one of them at once: measured at the backbone's
/// shape (16 blocks, N=2048, H=1024) that is 4–6 GB of retained activations, and
/// it pushed an 11 GB step to an out-of-memory abort. Pooling the temporaries
/// keeps only as many buffers as are simultaneously live — a handful — while
/// still allocating nothing once the shapes are steady.
#[derive(Default)]
struct Act {
    /// Scratch for every temporary, recycled by size within the call.
    pool: Pool,
    // The SwiGLU operands, read by backward, so they outlive the forward.
    gate_pre: Buf, // [N, U] pre-activation for SiLU'
    gate_act: Buf, // [N, U] SiLU(gate_pre)
    value: Buf,    // [N, U]
}

impl<C: Cell> Block<C> {
    /// Assemble a block around a cell, with fresh norms (γ=1) and Xavier `Linear`
    /// weights. `hidden` is the model width, `up` the SwiGLU inner width.
    pub fn from_cell(gpu: &Gpu, hidden: usize, up: usize, cell: C) -> Self {
        let post_cell_norm = cell
            .wants_post_cell_norm()
            .then(|| RmsNorm::new(gpu, hidden));
        Self {
            hidden,
            up,
            pre_norm1: RmsNorm::new(gpu, hidden),
            cell,
            post_cell_norm,
            pre_norm2: RmsNorm::new(gpu, hidden),
            lin_gate: Linear::from_parts(gpu, &Tensor::xavier(hidden, up), &Tensor::zeros(&[up])),
            lin_value: Linear::from_parts(gpu, &Tensor::xavier(hidden, up), &Tensor::zeros(&[up])),
            lin_down: Linear::from_parts(
                gpu,
                &Tensor::xavier(up, hidden),
                &Tensor::zeros(&[hidden]),
            ),
            act: Act::default(),
            seq: (0, 0),
        }
    }

    /// Assemble around a cell, taking the surrounding norms/projections from a
    /// CPU block (the cell is uploaded by the caller). Shared by the `from_cpu`
    /// constructors below.
    fn from_cpu_parts<D>(gpu: &Gpu, cpu: &crate::nn2::Block<D>, cell: C) -> Self
    where
        D: crate::nn2::block::Cell,
    {
        Self {
            hidden: cpu.hidden,
            up: cpu.up,
            pre_norm1: RmsNorm::from_parts(gpu, &cpu.pre_norm1.gamma),
            cell,
            post_cell_norm: cpu
                .post_cell_norm
                .as_ref()
                .map(|n| RmsNorm::from_parts(gpu, &n.gamma)),
            pre_norm2: RmsNorm::from_parts(gpu, &cpu.pre_norm2.gamma),
            lin_gate: Linear::from_parts(gpu, &cpu.lin_gate.w, &cpu.lin_gate.b),
            lin_value: Linear::from_parts(gpu, &cpu.lin_value.w, &cpu.lin_value.b),
            lin_down: Linear::from_parts(gpu, &cpu.lin_down.w, &cpu.lin_down.b),
            act: Act::default(),
            seq: (0, 0),
        }
    }

    /// Forward over `[B, T, H]` → `y` `[B, T, H]`.
    pub fn forward(&mut self, gpu: &Gpu, x: &DTensor, y: &mut DTensor) {
        assert_eq!(x.rank, 3, "Block::forward expects [B, T, H]");
        let (b, t, h) = (x.shape[0], x.shape[1], x.shape[2]);
        assert_eq!(h, self.hidden, "Block::forward — hidden mismatch");
        assert_eq!(y.dims(), x.dims(), "Block::forward — output shape");
        let (n, u) = (b * t, self.up);
        self.seq = (b, t);
        let a = &mut self.act;

        // Owned [N, H] copy of the input: it feeds both the norm path and the
        // residual, and the caller's `x` is only lent to us.
        let mut x_flat = a.pool.take(gpu, &[n, h]);
        gpu.stream
            .memcpy_dtod(&x.buf, &mut x_flat.buf)
            .expect("copy x");

        // Residual 1: z = x + post_cell_norm(cell(pre_norm1(x))). The post-cell
        // norm is skipped for cells that don't want it (mLSTM).
        let mut xn1 = a.pool.take(gpu, &[n, h]);
        self.pre_norm1.forward(gpu, &x_flat, &mut xn1);
        xn1.reshape_to(&[b, t, h]);
        let mut cell_out = a.pool.take(gpu, &[b, t, h]);
        self.cell.forward(gpu, &xn1, &mut cell_out);
        a.pool.put(xn1);

        // Downstream is position-wise [N, H]. With a post-cell norm the normalized
        // result lands in `cn`; without one the cell output *is* `cn`.
        cell_out.reshape_to(&[n, h]);
        let cn = match &mut self.post_cell_norm {
            Some(norm) => {
                let mut cn = a.pool.take(gpu, &[n, h]);
                norm.forward(gpu, &cell_out, &mut cn);
                a.pool.put(cell_out);
                cn
            }
            None => cell_out,
        };
        let mut z = a.pool.take(gpu, &[n, h]);
        ops::add_into(gpu, &x_flat, &cn, &mut z);
        a.pool.put_all([x_flat, cn]);

        // Residual 2: y = z + SwiGLU(pre_norm2(z)). The three SwiGLU operands are
        // the only values backward needs, so they alone go to permanent buffers.
        let mut zn = a.pool.take(gpu, &[n, h]);
        self.pre_norm2.forward(gpu, &z, &mut zn);
        self.lin_gate
            .forward(gpu, &zn, a.gate_pre.get(gpu, &[n, u]));
        self.lin_value.forward(gpu, &zn, a.value.get(gpu, &[n, u]));
        a.pool.put(zn);
        let mut mixed = a.pool.take(gpu, &[n, u]);
        ops::swiglu_forward_into(
            gpu,
            a.gate_pre.expect("projected"),
            a.value.expect("projected"),
            a.gate_act.get(gpu, &[n, u]),
            &mut mixed,
        );
        let mut down = a.pool.take(gpu, &[n, h]);
        self.lin_down.forward(gpu, &mixed, &mut down);
        y.reshape_to(&[n, h]);
        ops::add_into(gpu, &z, &down, y);
        y.reshape_to(&[b, t, h]);
        a.pool.put_all([mixed, down, z]);
    }

    /// Backward over `[B, T, H]` → `dx` `[B, T, H]`.
    pub fn backward(&mut self, gpu: &Gpu, dy: &DTensor, dx: &mut DTensor) {
        let (b, t) = self.seq;
        let (h, u) = (self.hidden, self.up);
        let n = b * t;
        assert_eq!(dx.dims(), [b, t, h], "Block::backward — dx shape");
        let a = &mut self.act;

        // Owned [N, H]: read by lin_down.backward and again by the d_z residual.
        let mut dy_flat = a.pool.take(gpu, &[n, h]);
        gpu.stream
            .memcpy_dtod(&dy.buf, &mut dy_flat.buf)
            .expect("copy dy");

        // Residual 2.
        let mut d_mixed = a.pool.take(gpu, &[n, u]);
        self.lin_down.backward(gpu, &dy_flat, &mut d_mixed);
        let mut d_gate = a.pool.take(gpu, &[n, u]);
        let mut d_value = a.pool.take(gpu, &[n, u]);
        ops::swiglu_backward_into(
            gpu,
            &d_mixed,
            a.gate_act.expect("forward before backward"),
            a.value.expect("forward before backward"),
            a.gate_pre.expect("forward before backward"),
            &mut d_gate,
            &mut d_value,
        );
        a.pool.put(d_mixed);
        let mut d_zn_g = a.pool.take(gpu, &[n, h]);
        self.lin_gate.backward(gpu, &d_gate, &mut d_zn_g);
        let mut d_zn_v = a.pool.take(gpu, &[n, h]);
        self.lin_value.backward(gpu, &d_value, &mut d_zn_v);
        let mut d_zn = a.pool.take(gpu, &[n, h]);
        ops::add_into(gpu, &d_zn_g, &d_zn_v, &mut d_zn);
        a.pool.put_all([d_gate, d_value, d_zn_g, d_zn_v]);

        // z feeds pre_norm2 (the MLP path) and the y = z + down residual, so the
        // norm's dx and the incoming dy sum into d_z. `d_z_mlp` is separate because
        // `add_into`'s destination must not be one of its operands.
        let mut d_z_mlp = a.pool.take(gpu, &[n, h]);
        self.pre_norm2.backward(gpu, &d_zn, &mut d_z_mlp);
        let mut d_z = a.pool.take(gpu, &[n, h]);
        ops::add_into(gpu, &d_z_mlp, &dy_flat, &mut d_z);
        a.pool.put_all([d_zn, d_z_mlp, dy_flat]);

        // Residual 1. Without a post-cell norm, d_cell_out is a copy of d_z — the
        // cell's backward must not clobber d_z, which the dx residual still needs.
        let mut d_cell_out = a.pool.take(gpu, &[n, h]);
        match &mut self.post_cell_norm {
            Some(norm) => norm.backward(gpu, &d_z, &mut d_cell_out),
            None => gpu
                .stream
                .memcpy_dtod(&d_z.buf, &mut d_cell_out.buf)
                .expect("seed d_cell_out"),
        }
        d_cell_out.reshape_to(&[b, t, h]);
        let mut d_cell_in = a.pool.take(gpu, &[b, t, h]);
        self.cell.backward(gpu, &d_cell_out, &mut d_cell_in);
        a.pool.put(d_cell_out);
        d_cell_in.reshape_to(&[n, h]);
        let mut d_xn1 = a.pool.take(gpu, &[n, h]);
        self.pre_norm1.backward(gpu, &d_cell_in, &mut d_xn1);
        // x feeds pre_norm1 (cell path) and the z = x + cn residual.
        dx.reshape_to(&[n, h]);
        ops::add_into(gpu, &d_xn1, &d_z, dx);
        dx.reshape_to(&[b, t, h]);
        a.pool.put_all([d_cell_in, d_xn1, d_z]);
    }

    /// Learnable tensors in a fixed order (checkpoint save/load).
    pub fn params_mut(&mut self) -> Vec<&mut DTensor> {
        let mut v = Vec::new();
        v.extend(self.pre_norm1.params_mut());
        v.extend(self.cell.params_mut());
        if let Some(norm) = &mut self.post_cell_norm {
            v.extend(norm.params_mut());
        }
        v.extend(self.pre_norm2.params_mut());
        v.extend(self.lin_gate.params_mut());
        v.extend(self.lin_value.params_mut());
        v.extend(self.lin_down.params_mut());
        v
    }

    /// Export the block into the matching CPU `nn` block for a `HIER` checkpoint.
    /// Downloads every surrounding norm/projection, then lets the cell assemble
    /// the concrete `SLSTMBlock` / `MLSTMBlock`.
    pub fn to_nn_layer(&mut self, gpu: &Gpu) -> Box<dyn crate::nn_layer::NnLayer> {
        use super::{dt_matrix, dt_vec};
        let (h, u) = (self.hidden, self.up);
        let pre1 = RMSNorm::from_loaded(h, dt_vec(gpu, &self.pre_norm1.gamma));
        let pre2 = RMSNorm::from_loaded(h, dt_vec(gpu, &self.pre_norm2.gamma));
        let post = self
            .post_cell_norm
            .as_ref()
            .map(|nm| RMSNorm::from_loaded(h, dt_vec(gpu, &nm.gamma)));
        let gate = LinearLayer::from_loaded(
            h,
            u,
            dt_matrix(gpu, &self.lin_gate.w),
            dt_vec(gpu, &self.lin_gate.b),
        );
        let value = LinearLayer::from_loaded(
            h,
            u,
            dt_matrix(gpu, &self.lin_value.w),
            dt_vec(gpu, &self.lin_value.b),
        );
        let down = LinearLayer::from_loaded(
            u,
            h,
            dt_matrix(gpu, &self.lin_down.w),
            dt_vec(gpu, &self.lin_down.b),
        );
        self.cell
            .to_nn_block(gpu, h, u, pre1, post, pre2, gate, value, down)
    }

    pub fn zero_grad(&mut self, gpu: &Gpu) {
        self.pre_norm1.zero_grad(gpu);
        self.cell.zero_grad(gpu);
        if let Some(norm) = &mut self.post_cell_norm {
            norm.zero_grad(gpu);
        }
        self.pre_norm2.zero_grad(gpu);
        self.lin_gate.zero_grad(gpu);
        self.lin_value.zero_grad(gpu);
        self.lin_down.zero_grad(gpu);
    }

    /// AdamW step across every sub-layer.
    pub fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        self.pre_norm1.step(gpu, cfg);
        self.cell.step(gpu, cfg);
        if let Some(norm) = &mut self.post_cell_norm {
            norm.step(gpu, cfg);
        }
        self.pre_norm2.step(gpu, cfg);
        self.lin_gate.step(gpu, cfg);
        self.lin_value.step(gpu, cfg);
        self.lin_down.step(gpu, cfg);
    }
}

/// Upload an `nn::LinearLayer` to the device.
fn lin_from_nn(gpu: &Gpu, l: &LinearLayer) -> Linear {
    use super::{tensor_from_matrix as m, tensor_from_slice as v};
    Linear::from_parts(gpu, &m(&l.weights), &v(&l.biases))
}

impl Block<SLstm> {
    /// Upload a whole CPU sLSTM block (norms, SwiGLU projections and the cell).
    pub fn from_cpu(gpu: &Gpu, cpu: &crate::nn2::SLstmBlock) -> Self {
        Self::from_cpu_parts(gpu, cpu, SLstm::from_cpu(gpu, &cpu.cell))
    }

    /// Import an `nn::SLSTMBlock` (from a `HIER` checkpoint) onto the device.
    pub fn from_nn_block(gpu: &Gpu, cpu: &crate::nn::slstm_block::SLSTMBlock) -> Self {
        use super::tensor_from_slice as v;
        Self {
            hidden: cpu.hidden_size,
            up: cpu.up_size,
            pre_norm1: RmsNorm::from_parts(gpu, &v(&cpu.pre_norm1.gamma)),
            cell: SLstm::from_nn_cell(gpu, &cpu.cell),
            post_cell_norm: Some(RmsNorm::from_parts(gpu, &v(&cpu.post_cell_norm.gamma))),
            pre_norm2: RmsNorm::from_parts(gpu, &v(&cpu.pre_norm2.gamma)),
            lin_gate: lin_from_nn(gpu, &cpu.lin_gate),
            lin_value: lin_from_nn(gpu, &cpu.lin_value),
            lin_down: lin_from_nn(gpu, &cpu.lin_down),
            act: Act::default(),
            seq: (0, 0),
        }
    }
}

impl Block<MLstm> {
    /// Upload a whole CPU mLSTM block (norms, SwiGLU projections and the cell).
    pub fn from_cpu(gpu: &Gpu, cpu: &crate::nn2::MLstmBlock) -> Self {
        Self::from_cpu_parts(gpu, cpu, MLstm::from_cpu(gpu, &cpu.cell))
    }

    /// Import an `nn::MLSTMBlock` (from a `HIER` checkpoint) onto the device.
    pub fn from_nn_block(gpu: &Gpu, cpu: &crate::nn::mlstm_block::MLSTMBlock) -> Self {
        use super::tensor_from_slice as v;
        Self {
            hidden: cpu.hidden_size,
            up: cpu.up_size,
            pre_norm1: RmsNorm::from_parts(gpu, &v(&cpu.pre_norm1.gamma)),
            cell: MLstm::from_nn_cell(gpu, &cpu.cell),
            post_cell_norm: None,
            pre_norm2: RmsNorm::from_parts(gpu, &v(&cpu.pre_norm2.gamma)),
            lin_gate: lin_from_nn(gpu, &cpu.lin_gate),
            lin_value: lin_from_nn(gpu, &cpu.lin_value),
            lin_down: lin_from_nn(gpu, &cpu.lin_down),
            act: Act::default(),
            seq: (0, 0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn2::block::{MLstmBlock as CpuMLstmBlock, SLstmBlock as CpuSLstmBlock};
    use crate::nn2::optim::AdamCfg;
    use crate::tensor::Tensor;

    fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length mismatch");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < tol, "{what}[{i}]: gpu {g} vs cpu {w}");
        }
    }

    /// GPU `Block<SLstm>` must match `nn2::SLstmBlock` for forward → backward →
    /// AdamW-step from identical parameters.
    #[test]
    fn slstm_block_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let (b, t, h, u) = (2, 4, 8, 12);

        let mut cpu = CpuSLstmBlock::new_slstm(h, u);
        let mut dev = Block::<SLstm>::from_cpu(&gpu, &cpu);

        let x = Tensor::random(&[b, t, h], 0.5);
        let g = Tensor::random(&[b, t, h], 1.0);

        // Forward
        let y_cpu = cpu.forward(&x);
        let y_dev = dev.forward_alloc(&gpu, &DTensor::from_host(&gpu, &x));
        assert_close(&y_dev.to_host(&gpu).data, &y_cpu.data, 3e-3, "y");

        // Backward
        let dx_cpu = cpu.backward(&g);
        let dx_dev = dev.backward_alloc(&gpu, &DTensor::from_host(&gpu, &g));
        assert_close(&dx_dev.to_host(&gpu).data, &dx_cpu.data, 3e-3, "dx");

        // One AdamW step; compare a representative parameter from each path.
        let mut cfg = AdamCfg::new(1e-3, 0.01);
        cfg.t = 1;
        cpu.step(&cfg);
        dev.step(&gpu, &cfg);
        assert_close(
            &dev.lin_down.w.to_host(&gpu).data,
            &cpu.lin_down.w.data,
            3e-3,
            "lin_down.w",
        );
        assert_close(
            &dev.pre_norm1.gamma.to_host(&gpu).data,
            &cpu.pre_norm1.gamma.data,
            3e-3,
            "pre_norm1.gamma",
        );
        assert_close(
            &dev.cell.w[0].to_host(&gpu).data,
            &cpu.cell.wz.data,
            3e-3,
            "cell.wz",
        );
    }

    /// GPU `Block<MLstm>` (parallel-form cell) must match `nn2::MLstmBlock` (scalar
    /// recurrence) for forward → backward → AdamW-step from identical parameters.
    /// This closes Phase C: the block wiring is shared, so it also re-checks that
    /// the mLSTM cell composes correctly inside the two residuals.
    #[test]
    fn mlstm_block_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let (b, t, h, u, heads, dqk) = (2, 5, 8, 12, 2, 4); // dhv = 4

        let mut cpu = CpuMLstmBlock::new_mlstm(h, u, heads, dqk);
        // Non-trivial gate weights so the decay/stabilizer path is exercised
        // (nn2::MLstm::new zero-inits wi/wf).
        cpu.cell.wi = Tensor::random(&[h, heads], 0.3);
        cpu.cell.wf = Tensor::random(&[h, heads], 0.3);
        let mut dev = Block::<MLstm>::from_cpu(&gpu, &cpu);

        let x = Tensor::random(&[b, t, h], 0.5);
        let g = Tensor::random(&[b, t, h], 1.0);

        let y_cpu = cpu.forward(&x);
        let y_dev = dev.forward_alloc(&gpu, &DTensor::from_host(&gpu, &x));
        assert_close(&y_dev.to_host(&gpu).data, &y_cpu.data, 3e-3, "y");

        let dx_cpu = cpu.backward(&g);
        let dx_dev = dev.backward_alloc(&gpu, &DTensor::from_host(&gpu, &g));
        assert_close(&dx_dev.to_host(&gpu).data, &dx_cpu.data, 3e-3, "dx");

        let mut cfg = AdamCfg::new(1e-3, 0.01);
        cfg.t = 1;
        cpu.step(&cfg);
        dev.step(&gpu, &cfg);
        assert_close(
            &dev.lin_down.w.to_host(&gpu).data,
            &cpu.lin_down.w.data,
            3e-3,
            "lin_down.w",
        );
        assert_close(
            &dev.pre_norm1.gamma.to_host(&gpu).data,
            &cpu.pre_norm1.gamma.data,
            3e-3,
            "pre_norm1.gamma",
        );
    }
}
