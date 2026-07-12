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

use super::{DTensor, Gpu, linear::Linear, mlstm::MLstm, ops, rms_norm::RmsNorm, slstm::SLstm};
use crate::nn2::optim::AdamCfg;

/// A recurrent cell operating on `[B, T, H]` device sequences (H in == H out).
pub trait Cell {
    fn forward(&mut self, gpu: &Gpu, x: &DTensor) -> DTensor;
    fn backward(&mut self, gpu: &Gpu, dy: &DTensor) -> DTensor;
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
        pre_norm1: crate::nn::rms_norm::RMSNorm,
        post_cell_norm: Option<crate::nn::rms_norm::RMSNorm>,
        pre_norm2: crate::nn::rms_norm::RMSNorm,
        lin_gate: crate::nn::linear::LinearLayer,
        lin_value: crate::nn::linear::LinearLayer,
        lin_down: crate::nn::linear::LinearLayer,
    ) -> Box<dyn crate::nn_layer::NnLayer>;
}

impl Cell for SLstm {
    fn forward(&mut self, gpu: &Gpu, x: &DTensor) -> DTensor { SLstm::forward(self, gpu, x) }
    fn backward(&mut self, gpu: &Gpu, dy: &DTensor) -> DTensor { SLstm::backward(self, gpu, dy) }
    fn zero_grad(&mut self, gpu: &Gpu) { SLstm::zero_grad(self, gpu) }
    fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) { SLstm::step(self, gpu, cfg) }
    fn params_mut(&mut self) -> Vec<&mut DTensor> { SLstm::params_mut(self) }
    fn wants_post_cell_norm(&self) -> bool { true }
    fn to_nn_block(
        &self,
        gpu: &Gpu,
        hidden: usize,
        up: usize,
        pre_norm1: crate::nn::rms_norm::RMSNorm,
        post_cell_norm: Option<crate::nn::rms_norm::RMSNorm>,
        pre_norm2: crate::nn::rms_norm::RMSNorm,
        lin_gate: crate::nn::linear::LinearLayer,
        lin_value: crate::nn::linear::LinearLayer,
        lin_down: crate::nn::linear::LinearLayer,
    ) -> Box<dyn crate::nn_layer::NnLayer> {
        let post = post_cell_norm.expect("sLSTM block requires a post_cell_norm");
        Box::new(crate::nn::slstm_block::SLSTMBlock::from_loaded(
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
    fn forward(&mut self, gpu: &Gpu, x: &DTensor) -> DTensor;
    fn backward(&mut self, gpu: &Gpu, dy: &DTensor) -> DTensor;
    fn zero_grad(&mut self, gpu: &Gpu);
    fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg);
    /// Learnable tensors in a fixed order (checkpoint save/load).
    fn params_mut(&mut self) -> Vec<&mut DTensor>;
    /// Export the block into the matching CPU `nn` block (`SLSTMBlock` /
    /// `MLSTMBlock`) for a `HIER` checkpoint.
    fn to_nn_layer(&mut self, gpu: &Gpu) -> Box<dyn crate::nn_layer::NnLayer>;
}

impl<C: Cell> BlockLike for Block<C> {
    fn forward(&mut self, gpu: &Gpu, x: &DTensor) -> DTensor { Block::forward(self, gpu, x) }
    fn backward(&mut self, gpu: &Gpu, dy: &DTensor) -> DTensor { Block::backward(self, gpu, dy) }
    fn zero_grad(&mut self, gpu: &Gpu) { Block::zero_grad(self, gpu) }
    fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) { Block::step(self, gpu, cfg) }
    fn params_mut(&mut self) -> Vec<&mut DTensor> { Block::params_mut(self) }
    fn to_nn_layer(&mut self, gpu: &Gpu) -> Box<dyn crate::nn_layer::NnLayer> {
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

    // Saved for backward (SwiGLU is the only path not owned by a sub-layer).
    gate_pre: Option<DTensor>, // [N, U] pre-activation for SiLU'
    gate_act: Option<DTensor>, // [N, U] SiLU(gate_pre)
    value: Option<DTensor>,    // [N, U]
    seq: (usize, usize),       // (B, T) of the last forward
}

impl<C: Cell> Block<C> {
    /// Assemble a block around a cell, with fresh norms (γ=1) and Xavier `Linear`
    /// weights. `hidden` is the model width, `up` the SwiGLU inner width.
    pub fn from_cell(gpu: &Gpu, hidden: usize, up: usize, cell: C) -> Self {
        let post_cell_norm = cell.wants_post_cell_norm().then(|| RmsNorm::new(gpu, hidden));
        Self {
            hidden,
            up,
            pre_norm1: RmsNorm::new(gpu, hidden),
            cell,
            post_cell_norm,
            pre_norm2: RmsNorm::new(gpu, hidden),
            lin_gate: Linear::from_parts(gpu, &crate::tensor::Tensor::xavier(hidden, up), &crate::tensor::Tensor::zeros(&[up])),
            lin_value: Linear::from_parts(gpu, &crate::tensor::Tensor::xavier(hidden, up), &crate::tensor::Tensor::zeros(&[up])),
            lin_down: Linear::from_parts(gpu, &crate::tensor::Tensor::xavier(up, hidden), &crate::tensor::Tensor::zeros(&[hidden])),
            gate_pre: None,
            gate_act: None,
            value: None,
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
            gate_pre: None,
            gate_act: None,
            value: None,
            seq: (0, 0),
        }
    }

    /// Forward over `[B, T, H]` → `[B, T, H]`.
    pub fn forward(&mut self, gpu: &Gpu, x: &DTensor) -> DTensor {
        assert_eq!(x.rank, 3, "Block::forward expects [B, T, H]");
        let (b, t, h) = (x.shape[0], x.shape[1], x.shape[2]);
        assert_eq!(h, self.hidden, "Block::forward — hidden mismatch");
        let n = b * t;
        self.seq = (b, t);

        // Owned [N, H] copy of the input: needed for both the norm path and the
        // residual (read twice), and the caller's `x` is only borrowed.
        let x_flat = x.dup(gpu).reshaped(&[n, h]);

        // Residual 1: z = x + post_cell_norm(cell(pre_norm1(x))). The post-cell
        // norm is skipped for cells that don't want it (mLSTM).
        let xn1 = self.pre_norm1.forward(gpu, &x_flat);
        let cell_out = self.cell.forward(gpu, &xn1.reshaped(&[b, t, h]));
        let cell_flat = cell_out.reshaped(&[n, h]);
        let cn = match &mut self.post_cell_norm {
            Some(norm) => norm.forward(gpu, &cell_flat),
            None => cell_flat,
        };
        let z = ops::add(gpu, &x_flat, &cn);

        // Residual 2: y = z + SwiGLU(pre_norm2(z)).
        let zn = self.pre_norm2.forward(gpu, &z);
        let gate_pre = self.lin_gate.forward(gpu, &zn);
        let value = self.lin_value.forward(gpu, &zn);
        let (gate_act, mixed) = ops::swiglu_forward(gpu, &gate_pre, &value);
        let down = self.lin_down.forward(gpu, &mixed);
        let y = ops::add(gpu, &z, &down);

        self.gate_pre = Some(gate_pre);
        self.gate_act = Some(gate_act);
        self.value = Some(value);
        y.reshaped(&[b, t, h])
    }

    /// Backward over `[B, T, H]` → `dx` `[B, T, H]`.
    pub fn backward(&mut self, gpu: &Gpu, dy: &DTensor) -> DTensor {
        let (b, t) = self.seq;
        let h = self.hidden;
        let n = b * t;
        // Owned [N, H]: used by lin_down.backward and the d_z residual (twice).
        let dy_flat = dy.dup(gpu).reshaped(&[n, h]);

        // Residual 2.
        let d_mixed = self.lin_down.backward(gpu, &dy_flat); // [n, u]
        // Taken, not borrowed: these [N, U] activations are rebuilt by every forward,
        // so releasing them here frees the device memory across the optimizer step
        // instead of holding it until the next forward overwrites the fields.
        let gate_pre = self.gate_pre.take().expect("forward before backward");
        let gate_act = self.gate_act.take().expect("forward before backward");
        let value = self.value.take().expect("forward before backward");
        let (d_gate, d_value) = ops::swiglu_backward(gpu, &d_mixed, &gate_act, &value, &gate_pre);
        drop((gate_pre, gate_act, value));
        let d_zn_g = self.lin_gate.backward(gpu, &d_gate);
        let d_zn_v = self.lin_value.backward(gpu, &d_value);
        let d_zn = ops::add(gpu, &d_zn_g, &d_zn_v);
        let d_z_mlp = self.pre_norm2.backward(gpu, &d_zn);
        // z feeds pre_norm2 (MLP path) and the y = z + down residual.
        let d_z = ops::add(gpu, &d_z_mlp, &dy_flat);

        // Residual 1.
        let d_cell_out = match &mut self.post_cell_norm {
            Some(norm) => norm.backward(gpu, &d_z),
            None => d_z.dup(gpu),
        };
        let d_cell_in = self.cell.backward(gpu, &d_cell_out.reshaped(&[b, t, h]));
        let d_xn1 = self.pre_norm1.backward(gpu, &d_cell_in.reshaped(&[n, h]));
        // x feeds pre_norm1 (cell path) and the z = x + cn residual.
        let dx = ops::add(gpu, &d_xn1, &d_z);
        dx.reshaped(&[b, t, h])
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
        use crate::nn::{linear::LinearLayer, rms_norm::RMSNorm};
        let (h, u) = (self.hidden, self.up);
        let pre1 = RMSNorm::from_loaded(h, dt_vec(gpu, &self.pre_norm1.gamma));
        let pre2 = RMSNorm::from_loaded(h, dt_vec(gpu, &self.pre_norm2.gamma));
        let post = self
            .post_cell_norm
            .as_ref()
            .map(|nm| RMSNorm::from_loaded(h, dt_vec(gpu, &nm.gamma)));
        let gate = LinearLayer::from_loaded(h, u, dt_matrix(gpu, &self.lin_gate.w), dt_vec(gpu, &self.lin_gate.b));
        let value = LinearLayer::from_loaded(h, u, dt_matrix(gpu, &self.lin_value.w), dt_vec(gpu, &self.lin_value.b));
        let down = LinearLayer::from_loaded(u, h, dt_matrix(gpu, &self.lin_down.w), dt_vec(gpu, &self.lin_down.b));
        self.cell.to_nn_block(gpu, h, u, pre1, post, pre2, gate, value, down)
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
fn lin_from_nn(gpu: &Gpu, l: &crate::nn::linear::LinearLayer) -> Linear {
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
            gate_pre: None,
            gate_act: None,
            value: None,
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
            gate_pre: None,
            gate_act: None,
            value: None,
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
        let Some(gpu) = super::super::test_gpu() else { return };
        let (b, t, h, u) = (2, 4, 8, 12);

        let mut cpu = CpuSLstmBlock::new_slstm(h, u);
        let mut dev = Block::<SLstm>::from_cpu(&gpu, &cpu);

        let x = Tensor::random(&[b, t, h], 0.5);
        let g = Tensor::random(&[b, t, h], 1.0);

        // Forward
        let y_cpu = cpu.forward(&x);
        let y_dev = dev.forward(&gpu, &DTensor::from_host(&gpu, &x));
        assert_close(&y_dev.to_host(&gpu).data, &y_cpu.data, 3e-3, "y");

        // Backward
        let dx_cpu = cpu.backward(&g);
        let dx_dev = dev.backward(&gpu, &DTensor::from_host(&gpu, &g));
        assert_close(&dx_dev.to_host(&gpu).data, &dx_cpu.data, 3e-3, "dx");

        // One AdamW step; compare a representative parameter from each path.
        let mut cfg = AdamCfg::new(1e-3, 0.01);
        cfg.t = 1;
        cpu.step(&cfg);
        dev.step(&gpu, &cfg);
        assert_close(&dev.lin_down.w.to_host(&gpu).data, &cpu.lin_down.w.data, 3e-3, "lin_down.w");
        assert_close(&dev.pre_norm1.gamma.to_host(&gpu).data, &cpu.pre_norm1.gamma.data, 3e-3, "pre_norm1.gamma");
        assert_close(&dev.cell.w[0].to_host(&gpu).data, &cpu.cell.wz.data, 3e-3, "cell.wz");
    }

    /// GPU `Block<MLstm>` (parallel-form cell) must match `nn2::MLstmBlock` (scalar
    /// recurrence) for forward → backward → AdamW-step from identical parameters.
    /// This closes Phase C: the block wiring is shared, so it also re-checks that
    /// the mLSTM cell composes correctly inside the two residuals.
    #[test]
    fn mlstm_block_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else { return };
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
        let y_dev = dev.forward(&gpu, &DTensor::from_host(&gpu, &x));
        assert_close(&y_dev.to_host(&gpu).data, &y_cpu.data, 3e-3, "y");

        let dx_cpu = cpu.backward(&g);
        let dx_dev = dev.backward(&gpu, &DTensor::from_host(&gpu, &g));
        assert_close(&dx_dev.to_host(&gpu).data, &dx_cpu.data, 3e-3, "dx");

        let mut cfg = AdamCfg::new(1e-3, 0.01);
        cfg.t = 1;
        cpu.step(&cfg);
        dev.step(&gpu, &cfg);
        assert_close(&dev.lin_down.w.to_host(&gpu).data, &cpu.lin_down.w.data, 3e-3, "lin_down.w");
        assert_close(&dev.pre_norm1.gamma.to_host(&gpu).data, &cpu.pre_norm1.gamma.data, 3e-3, "pre_norm1.gamma");
    }
}
