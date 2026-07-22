//! xLSTM-style residual block, batched.
//!
//!   x ─► RMSNorm ─► cell ─► RMSNorm ─┬─► z ─► RMSNorm ─► SwiGLU ─┬─► y
//!        (pre1)            (postcell) │        (pre2)             │
//!        └──────── residual 1 ────────┘        └──── residual 2 ──┘
//!
//!   z = x + post_cell_norm(cell(pre_norm1(x)))
//!   y = z + lin_down( SiLU(lin_gate·pre_norm2(z)) ⊙ (lin_value·pre_norm2(z)) )
//!
//! Same architecture as `nn::slstm_block` / `nn::mlstm_block`. The norms and the
//! SwiGLU MLP are position-wise, so they run once on the flattened `[B·T, H]`
//! view; only the recurrent `cell` sees the `[B, T, H]` sequence. This lets the
//! block compose the already-FD-verified `Linear` / `RmsNorm` sub-layers (each
//! called exactly once per block forward) around a generic cell.

use crate::nn2::optim::AdamCfg;
use crate::nn2::{Linear, MLstm, RmsNorm, SLstm};
use crate::tensor::Tensor;

/// A recurrent cell operating on `[B, T, H]` sequences (H in == H out).
pub trait Cell {
    fn forward(&mut self, x: &Tensor) -> Tensor;
    fn backward(&mut self, dy: &Tensor) -> Tensor;
    fn zero_grad(&mut self);
    fn step(&mut self, cfg: &AdamCfg);
    /// Whether the block wraps this cell with a `post_cell_norm` before the
    /// residual. sLSTM does (like `nn::slstm_block`); mLSTM does not — its own
    /// head-wise norm already normalizes the cell output, so a second norm here
    /// is redundant and absent from `nn::mlstm_block`.
    fn wants_post_cell_norm(&self) -> bool;
}

impl Cell for SLstm {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        SLstm::forward(self, x)
    }
    fn backward(&mut self, dy: &Tensor) -> Tensor {
        SLstm::backward(self, dy)
    }
    fn zero_grad(&mut self) {
        SLstm::zero_grad(self)
    }
    fn step(&mut self, cfg: &AdamCfg) {
        SLstm::step(self, cfg)
    }
    fn wants_post_cell_norm(&self) -> bool {
        true
    }
}

impl Cell for MLstm {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        MLstm::forward(self, x)
    }
    fn backward(&mut self, dy: &Tensor) -> Tensor {
        MLstm::backward(self, dy)
    }
    fn zero_grad(&mut self) {
        MLstm::zero_grad(self)
    }
    fn step(&mut self, cfg: &AdamCfg) {
        MLstm::step(self, cfg)
    }
    fn wants_post_cell_norm(&self) -> bool {
        false
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
#[inline]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
#[inline]
fn silu_prime(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
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

    // Saved for backward (single call per forward).
    gate_pre: Tensor,    // [N, U] pre-activation for SiLU'
    gate_act: Tensor,    // [N, U] SiLU(gate_pre)
    value: Tensor,       // [N, U]
    seq: (usize, usize), // (B, T) of the last forward
}

impl<C: Cell> Block<C> {
    pub fn from_cell(hidden: usize, up: usize, cell: C) -> Self {
        let post_cell_norm = cell.wants_post_cell_norm().then(|| RmsNorm::new(hidden));
        Self {
            hidden,
            up,
            pre_norm1: RmsNorm::new(hidden),
            cell,
            post_cell_norm,
            pre_norm2: RmsNorm::new(hidden),
            lin_gate: Linear::new(hidden, up),
            lin_value: Linear::new(hidden, up),
            lin_down: Linear::new(up, hidden),
            gate_pre: Tensor::zeros(&[0, up]),
            gate_act: Tensor::zeros(&[0, up]),
            value: Tensor::zeros(&[0, up]),
            seq: (0, 0),
        }
    }

    /// Forward over `[B, T, H]` → `[B, T, H]`.
    pub fn forward(&mut self, x: &Tensor) -> Tensor {
        assert_eq!(x.rank(), 3, "Block::forward expects [B, T, H]");
        let (b, t, h) = (x.shape[0], x.shape[1], x.shape[2]);
        assert_eq!(h, self.hidden, "Block::forward — hidden mismatch");
        let n = b * t;
        self.seq = (b, t);

        let x_flat = x.reshape(&[n, h]);

        // Residual 1: z = x + post_cell_norm(cell(pre_norm1(x))).  The post-cell
        // norm is skipped for cells that don't want it (mLSTM).
        let xn1 = self.pre_norm1.forward(&x_flat);
        let cell_out = self.cell.forward(&xn1.reshape(&[b, t, h]));
        let cell_flat = cell_out.reshape(&[n, h]);
        let cn = match &mut self.post_cell_norm {
            Some(norm) => norm.forward(&cell_flat),
            None => cell_flat,
        };
        let mut z = Tensor::zeros(&[n, h]);
        for i in 0..n * h {
            z.data[i] = x_flat.data[i] + cn.data[i];
        }

        // Residual 2: y = z + SwiGLU(pre_norm2(z)).
        let zn = self.pre_norm2.forward(&z);
        self.gate_pre = self.lin_gate.forward(&zn);
        self.value = self.lin_value.forward(&zn);
        let u = self.up;
        self.gate_act = Tensor::zeros(&[n, u]);
        let mut mixed = Tensor::zeros(&[n, u]);
        for i in 0..n * u {
            let ga = silu(self.gate_pre.data[i]);
            self.gate_act.data[i] = ga;
            mixed.data[i] = ga * self.value.data[i];
        }
        let down = self.lin_down.forward(&mixed);

        let mut y = Tensor::zeros(&[n, h]);
        for i in 0..n * h {
            y.data[i] = z.data[i] + down.data[i];
        }
        y.reshape(&[b, t, h])
    }

    /// Backward over `[B, T, H]` → `dx` `[B, T, H]`.
    pub fn backward(&mut self, dy: &Tensor) -> Tensor {
        let (b, t) = self.seq;
        let h = self.hidden;
        let u = self.up;
        let n = b * t;
        let dy_flat = dy.reshape(&[n, h]);

        // Residual 2: y = z + down.
        let d_mixed = self.lin_down.backward(&dy_flat); // [n, u]
        // mixed = gate_act ⊙ value.
        let mut d_gate = Tensor::zeros(&[n, u]);
        let mut d_value = Tensor::zeros(&[n, u]);
        for i in 0..n * u {
            let dm = d_mixed.data[i];
            d_value.data[i] = dm * self.gate_act.data[i];
            d_gate.data[i] = dm * self.value.data[i] * silu_prime(self.gate_pre.data[i]);
        }
        let d_zn_g = self.lin_gate.backward(&d_gate);
        let d_zn_v = self.lin_value.backward(&d_value);
        let mut d_zn = Tensor::zeros(&[n, h]);
        for i in 0..n * h {
            d_zn.data[i] = d_zn_g.data[i] + d_zn_v.data[i];
        }
        // z feeds pre_norm2 (MLP path) and the y = z + down residual.
        let d_z_mlp = self.pre_norm2.backward(&d_zn);
        let mut d_z = Tensor::zeros(&[n, h]);
        for i in 0..n * h {
            d_z.data[i] = d_z_mlp.data[i] + dy_flat.data[i];
        }

        // Residual 1: z = x + cn.
        let d_cell_out = match &mut self.post_cell_norm {
            Some(norm) => norm.backward(&d_z),
            None => d_z.clone(),
        };
        let d_cell_in = self.cell.backward(&d_cell_out.reshape(&[b, t, h]));
        let d_xn1 = self.pre_norm1.backward(&d_cell_in.reshape(&[n, h]));
        // x feeds pre_norm1 (cell path) and the z = x + cn residual.
        let mut dx = Tensor::zeros(&[n, h]);
        for i in 0..n * h {
            dx.data[i] = d_xn1.data[i] + d_z.data[i];
        }
        dx.reshape(&[b, t, h])
    }

    pub fn zero_grad(&mut self) {
        self.pre_norm1.zero_grad();
        self.cell.zero_grad();
        if let Some(norm) = &mut self.post_cell_norm {
            norm.zero_grad();
        }
        self.pre_norm2.zero_grad();
        self.lin_gate.zero_grad();
        self.lin_value.zero_grad();
        self.lin_down.zero_grad();
    }

    /// AdamW step across every sub-layer.
    pub fn step(&mut self, cfg: &AdamCfg) {
        self.pre_norm1.step(cfg);
        self.cell.step(cfg);
        if let Some(norm) = &mut self.post_cell_norm {
            norm.step(cfg);
        }
        self.pre_norm2.step(cfg);
        self.lin_gate.step(cfg);
        self.lin_value.step(cfg);
        self.lin_down.step(cfg);
    }
}

/// sLSTM block: `hidden → hidden` cell.
pub type SLstmBlock = Block<SLstm>;
/// mLSTM block: multi-head cell.
pub type MLstmBlock = Block<MLstm>;

impl SLstmBlock {
    pub fn new_slstm(hidden: usize, up: usize) -> Self {
        Block::from_cell(hidden, up, SLstm::new(hidden, hidden))
    }
}
impl MLstmBlock {
    pub fn new_mlstm(hidden: usize, up: usize, heads: usize, dqk: usize) -> Self {
        Block::from_cell(hidden, up, MLstm::new(hidden, hidden, heads, dqk))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Directional whole-tensor FD across the full block (contains a recurrent
    /// cell with stabilizer kinks, so loose tol like the cell tests).
    fn check(
        cell_forward: impl Fn(&mut MLstmBlock, &Tensor) -> Tensor,
        grad_of: impl Fn(&MLstmBlock) -> Vec<f32>,
        mut perturb: impl FnMut(&mut MLstmBlock, f32, &[f32]),
        name: &str,
    ) {
        let (b, t, h, u) = (2, 3, 8, 12);
        let mut blk = MLstmBlock::new_mlstm(h, u, 2, 4);
        let x = Tensor::random(&[b, t, h], 0.5);
        let g = Tensor::random(&[b, t, h], 1.0);

        let _ = cell_forward(&mut blk, &x);
        let _dx = blk.backward(&g);
        let grad = grad_of(&blk);

        let norm: f32 = grad.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm > 1e-6, "{name}: analytic grad ~zero");
        let u_dir: Vec<f32> = grad.iter().map(|v| v / norm).collect();
        let loss = |blk: &mut MLstmBlock| -> f32 {
            let y = blk.forward(&x);
            y.data.iter().zip(&g.data).map(|(a, b)| a * b).sum()
        };
        let eps = 2e-4;
        perturb(&mut blk, eps, &u_dir);
        let plus = loss(&mut blk);
        perturb(&mut blk, -2.0 * eps, &u_dir);
        let minus = loss(&mut blk);
        perturb(&mut blk, eps, &u_dir);
        let fd = (plus - minus) / (2.0 * eps);
        assert!(
            (fd - norm).abs() <= 0.3 * norm + 1e-3,
            "{name}: ‖G‖ {norm} vs fd {fd}"
        );
    }

    #[test]
    fn lin_down_grad_matches_fd() {
        check(
            |blk, x| blk.forward(x),
            |blk| blk.lin_down.dw.data.clone(),
            |blk, s, u| {
                for (w, &ui) in blk.lin_down.w.data.iter_mut().zip(u) {
                    *w += s * ui;
                }
            },
            "lin_down.w",
        );
    }

    #[test]
    fn pre_norm1_gamma_grad_matches_fd() {
        check(
            |blk, x| blk.forward(x),
            |blk| blk.pre_norm1.dgamma.data.clone(),
            |blk, s, u| {
                for (w, &ui) in blk.pre_norm1.gamma.data.iter_mut().zip(u) {
                    *w += s * ui;
                }
            },
            "pre_norm1.gamma",
        );
    }
}
