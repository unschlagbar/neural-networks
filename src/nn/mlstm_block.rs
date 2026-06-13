// mlstm_block.rs ── Multi-head mLSTM-Block aus xLSTM 7B (Beck et al. 2024)
//
// Per-Timestep-Architektur (zwei separate Residuals, Transformer-Stil):
//
//     x ──┬─► RMSNorm(1) ─► mLSTM cell ─┬─► z ──┬─► RMSNorm(2) ─► SwiGLU ─┬─►
//         │                              │       │                         │
//         └──────────────────────────────┘       └─────────────────────────┘
//           1. Residual: z = x + cell(x)         2. Residual: y = z + MLP(z)
//
// SwiGLU-MLP(h) = lin_down · ( SiLU(lin_gate · h) ⊙ (lin_value · h) )
//
// Components:
//   pre_norm  : RMSNorm   (H_dim)
//   cell      : MLSTMLayer  (multi-head, H_dim → H_dim)
//   pre_norm2 : RMSNorm   (H_dim)
//   lin_gate  : LinearLayer (H_dim → U)
//   lin_value : LinearLayer (H_dim → U)
//   lin_down  : LinearLayer (U → H_dim)
//
// (H_dim is used for hidden_size / block dim and H_n for num_heads,
//  to avoid confusion between "hidden" and "num heads".)
//
// Save-Format (Tag 14):
//   up_size : u32
//   pre_norm.gamma, pre_norm2.gamma : f32_vec(H_dim)
//   cell-Daten   (mit num_heads, dqk vorne — siehe MLSTMLayer::save)
//   lin_gate, lin_value, lin_down  (jeweils Matrix + Bias)

use std::{any::Any, io};

use crate::{
    nn::{
        linear::{LinearCache, LinearLayer},
        mlstm::{MLSTMCache, MLSTMLayer},
        rms_norm::{RMSNorm, RMSNormCache},
    },
    nn_layer::{DynCache, NnLayer},
    saving::{write_f32_slice, write_matrix, write_u32},
};

#[inline]
fn stable_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x * stable_sigmoid(x)
}

#[inline]
fn silu_prime(pre: f32) -> f32 {
    let s = stable_sigmoid(pre);
    s * (1.0 + pre * (1.0 - s))
}

pub struct MLSTMBlockCache {
    pub pre_norm1: RMSNormCache,
    pub cell: MLSTMCache,
    pub z: Box<[f32]>, // (H_dim) z = x + cell.h
    pub pre_norm2: RMSNormCache,
    pub lin_gate: LinearCache,
    pub gate_act: Box<[f32]>, // (U) SiLU(gate_pre)
    pub lin_value: LinearCache,
    pub mixed: Box<[f32]>, // (U) gate_act ⊙ value
    pub lin_down: LinearCache,
    pub output: Box<[f32]>, // (H_dim)
    pub dx: Box<[f32]>,     // (H_dim)
}

impl DynCache for MLSTMBlockCache {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn output(&self) -> &[f32] {
        &self.output
    }
    fn input_grad(&self) -> &[f32] {
        &self.dx
    }
}

pub struct MLSTMBlock {
    pub hidden_size: usize,
    pub up_size: usize,

    pub pre_norm1: RMSNorm,
    pub cell: MLSTMLayer,
    pub pre_norm2: RMSNorm,
    pub lin_gate: LinearLayer,
    pub lin_value: LinearLayer,
    pub lin_down: LinearLayer,

    // Backward scratch buffers (no allocation in the hot path)
    pub sc_h1: Box<[f32]>, // (H_dim)
    pub sc_h2: Box<[f32]>, // (H_dim)
    pub sc_u2: Box<[f32]>, // (U)
    pub sc_u3: Box<[f32]>, // (U)
}

impl MLSTMBlock {
    pub fn new(hidden_size: usize, num_heads: usize, dqk: usize, up_size: usize) -> Self {
        let h = hidden_size;
        let u = up_size;
        let scale_up = (6.0 / (h as f32 + u as f32)).sqrt();
        let scale_dn = (6.0 / (u as f32 + h as f32)).sqrt();

        let make_lin = |rows: usize, cols: usize, scale: f32| {
            use iron_oxide::collections::Matrix;
            LinearLayer::from_loaded(
                rows,
                cols,
                Matrix::random(rows, cols, scale),
                vec![0.0; cols].into(),
            )
        };

        Self {
            hidden_size: h,
            up_size: u,

            pre_norm1: RMSNorm::new(h),
            cell: MLSTMLayer::new(h, h, num_heads, dqk),
            pre_norm2: RMSNorm::new(h),
            lin_gate: make_lin(h, u, scale_up),
            lin_value: make_lin(h, u, scale_up),
            lin_down: make_lin(u, h, scale_dn),

            sc_h1: vec![0.0; h].into(),
            sc_h2: vec![0.0; h].into(),
            sc_u2: vec![0.0; u].into(),
            sc_u3: vec![0.0; u].into(),
        }
    }

    pub fn from_loaded(
        hidden_size: usize,
        up_size: usize,
        pre_norm1: RMSNorm,
        pre_norm2: RMSNorm,
        cell: MLSTMLayer,
        lin_gate: LinearLayer,
        lin_value: LinearLayer,
        lin_down: LinearLayer,
    ) -> Self {
        let h = hidden_size;
        let u = up_size;
        Self {
            hidden_size: h,
            up_size: u,
            pre_norm1,
            cell,
            pre_norm2,
            lin_gate,
            lin_value,
            lin_down,
            sc_h1: vec![0.0; h].into(),
            sc_h2: vec![0.0; h].into(),
            sc_u2: vec![0.0; u].into(),
            sc_u3: vec![0.0; u].into(),
        }
    }

    pub fn forward(&mut self, input: &[f32], cache: &mut MLSTMBlockCache) {
        let h = self.hidden_size;
        let u = self.up_size;

        self.pre_norm1.forward_into(input, &mut cache.pre_norm1);

        self.cell.forward(&cache.pre_norm1.output, &mut cache.cell);

        for i in 0..h {
            cache.z[i] = input[i] + cache.cell.w_out.output[i];
        }

        self.pre_norm2.forward_into(&cache.z, &mut cache.pre_norm2);

        self.lin_gate
            .forward(&cache.pre_norm2.output, &mut cache.lin_gate);
        self.lin_value
            .forward(&cache.pre_norm2.output, &mut cache.lin_value);

        for j in 0..u {
            cache.gate_act[j] = silu(cache.lin_gate.output[j]);
            cache.mixed[j] = cache.gate_act[j] * cache.lin_value.output[j];
        }

        self.lin_down.forward(&cache.mixed, &mut cache.lin_down);

        for i in 0..h {
            cache.output[i] = cache.z[i] + cache.lin_down.output[i];
        }
    }

    pub fn backward(&mut self, delta: &mut [f32], cache: &mut MLSTMBlockCache) {
        let h = self.hidden_size;
        let u = self.up_size;

        cache.dx.copy_from_slice(delta);

        self.lin_down.backward(delta, &mut cache.lin_down);

        // mixed = gate_act ⊙ value
        for j in 0..u {
            self.sc_u2[j] = cache.lin_down.dx[j] * cache.lin_value.output[j]; // d(gate_act)
            self.sc_u3[j] = cache.lin_down.dx[j] * cache.gate_act[j]; // d(value)
        }

        self.lin_value
            .backward(&mut self.sc_u3, &mut cache.lin_value);

        for j in 0..u {
            self.sc_u3[j] = self.sc_u2[j] * silu_prime(cache.lin_gate.output[j]);
        }
        self.lin_gate.backward(&mut self.sc_u3, &mut cache.lin_gate);

        for i in 0..h {
            self.sc_h1[i] = cache.lin_value.dx[i] + cache.lin_gate.dx[i];
        }

        self.pre_norm2
            .backward_into(&self.sc_h1, &mut cache.pre_norm2);

        for i in 0..h {
            cache.dx[i] += cache.pre_norm2.dx[i];
        }

        self.sc_h2.copy_from_slice(&cache.dx);
        self.cell.backward(&mut self.sc_h2, &mut cache.cell);

        let d_pre_normed = &cache.cell.dx[..h];
        self.pre_norm1
            .backward_into(d_pre_normed, &mut cache.pre_norm1);

        for i in 0..h {
            cache.dx[i] += cache.pre_norm1.dx[i];
        }
    }

    pub fn alloc_cache(&self) -> MLSTMBlockCache {
        let h = self.hidden_size;
        let u = self.up_size;
        MLSTMBlockCache {
            pre_norm1: self.pre_norm1.alloc_cache(),
            cell: self.cell.alloc_cache(),
            z: vec![0.0; h].into(),
            pre_norm2: self.pre_norm2.alloc_cache(),
            lin_gate: LinearCache::new(h, u),
            gate_act: vec![0.0; u].into(),
            lin_value: LinearCache::new(h, u),
            mixed: vec![0.0; u].into(),
            lin_down: LinearCache::new(u, h),
            output: vec![0.0; h].into(),
            dx: vec![0.0; h].into(),
        }
    }
}

impl NnLayer for MLSTMBlock {
    //type Cache = MLSTMBlockCache;
    fn input_size(&self) -> usize {
        self.hidden_size
    }
    fn output_size(&self) -> usize {
        self.hidden_size
    }

    fn forward(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<MLSTMBlockCache>()
            .expect("MLSTMBlock::forward — expected MLSTMBlockCache");
        MLSTMBlock::forward(self, input, c);
    }

    fn backward(&mut self, delta: &mut [f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<MLSTMBlockCache>()
            .expect("MLSTMBlock::backward — expected MLSTMBlockCache");
        MLSTMBlock::backward(self, delta, c);
    }

    fn make_cache(&self) -> Box<dyn DynCache> {
        Box::new(MLSTMBlock::alloc_cache(self))
    }

    fn layer_tag(&self) -> u8 {
        14
    }

    fn save(&self, w: &mut dyn io::Write) -> io::Result<()> {
        write_u32(w, self.up_size as u32)?;
        write_f32_slice(w, &self.pre_norm1.gamma)?;
        write_f32_slice(w, &self.pre_norm2.gamma)?;
        // cell.save writes num_heads, dqk + all weights itself
        self.cell.save(w)?;
        write_matrix(w, &self.lin_gate.weights)?;
        write_f32_slice(w, &self.lin_gate.biases)?;
        write_matrix(w, &self.lin_value.weights)?;
        write_f32_slice(w, &self.lin_value.biases)?;
        write_matrix(w, &self.lin_down.weights)?;
        write_f32_slice(w, &self.lin_down.biases)?;
        Ok(())
    }

    fn reset_state(&mut self) {
        self.cell.reset_state();
    }

    fn reset_bptt_state(&mut self) {
        self.cell.reset_bptt_state();
    }

    fn state_size(&self) -> usize {
        0
    }

    fn inject_state(&mut self, _buf: &[f32], offset: usize) -> usize {
        self.cell.reset_state();
        offset
    }

    fn collect_bptt_grad(&mut self, _buf: &mut [f32], offset: usize) -> usize {
        //self.cell.collect_bptt_grad(buf, offset)
        offset
    }

    fn apply_grads(&mut self, lr: f32) {
        self.pre_norm1.apply_grads(lr);
        self.cell.apply_grads(lr);
        self.pre_norm2.apply_grads(lr);
        self.lin_gate.apply_grads(lr);
        self.lin_value.apply_grads(lr);
        self.lin_down.apply_grads(lr);
    }

    fn clear_grads(&mut self) {
        self.pre_norm1.clear_grads();
        self.cell.clear_grads();
        self.pre_norm2.clear_grads();
        self.lin_gate.clear_grads();
        self.lin_value.clear_grads();
        self.lin_down.clear_grads();
    }
}
