// slstm_block.rs ── xLSTM-style sLSTM block
//
// Per-Timestep Architektur (Paper-konform, zwei getrennte Residuals):
//
//     x ──┬─► RMSNorm(1) ─► sLSTM-Zelle ─► RMSNorm(post) ─┬─► z ──┬─► RMSNorm(2) ─► SwiGLU ─┬─►
//         │                                               │       │                         │
//         └───────────────────────────────────────────────┘       └─────────────────────────┘
//           1. Residual: z = x + post_norm(cell(x))               2. Residual: y = z + MLP(z)
//
// SwiGLU-MLP(h) = lin_down · ( SiLU(lin_gate · h) ⊙ (lin_value · h) )
//
// Bausteine (alle statisch, kein dyn):
//   pre_norm1      : RMSNorm  (H)
//   cell           : SLSTMLayer  (H→H)
//   post_cell_norm : RMSNorm  (H)   ← stabilisiert cell-Output vor Residual (Paper §3)
//   pre_norm2      : RMSNorm  (H)
//   lin_gate       : LinearLayer  (H→U)
//   lin_value      : LinearLayer  (H→U)
//   lin_down       : LinearLayer  (U→H)
//
// Shape-Konvention:  H = hidden_size,  U = up_size  (typisch U = 8·H/3).
//
// Binär-Format (`save`):
//   up_size : u32
//   pre_norm1.gamma, post_cell_norm.gamma, pre_norm2.gamma : f32_vec(H)
//   cell: wz, wi, wf, wo, b, h_init, c_init
//   lin_gate:  weights (H×U), biases (U)
//   lin_value: weights (H×U), biases (U)
//   lin_down:  weights (U×H), biases (H)

use std::{any::Any, io};

use crate::{
    nn::{
        linear::{LinearCache, LinearLayer},
        rms_norm::{RMSNorm, RMSNormCache},
        slstm::{SLSTMCache, SLSTMLayer},
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

pub struct SLSTMBlockCache {
    // Pre-Norm
    pub pre_norm1: RMSNormCache, // .output = pre_normed (H)

    // sLSTM-Zelle
    pub cell: SLSTMCache,

    // Post-Cell-Norm: normalisiert cell.h vor dem Residual
    pub post_cell_norm: RMSNormCache, // .output = normed cell.h (H)

    // Erstes Residual: z = x + post_cell_norm.output
    pub z: Box<[f32]>, // (H)

    // Pre-Norm2 (auf z angewendet)
    pub pre_norm2: RMSNormCache, // .output = pre_normed2 (H)

    // SwiGLU
    pub lin_gate: LinearCache,  // .output = gate_pre  (U)
    pub gate_act: Box<[f32]>,   // SiLU(gate_pre)       (U)
    pub lin_value: LinearCache, // .output = value       (U)
    pub mixed: Box<[f32]>,      // gate_act ⊙ value      (U)
    pub lin_down: LinearCache,  // .output = down_out    (H)

    pub output: Box<[f32]>, // z + down_out          (H)
    pub dx: Box<[f32]>,     // dL/d(input)           (H)
}

impl DynCache for SLSTMBlockCache {
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

pub struct SLSTMBlock {
    pub hidden_size: usize,
    pub up_size: usize,

    pub pre_norm1: RMSNorm,
    pub cell: SLSTMLayer,
    pub post_cell_norm: RMSNorm,
    pub pre_norm2: RMSNorm,
    pub lin_gate: LinearLayer,
    pub lin_value: LinearLayer,
    pub lin_down: LinearLayer,

    // Backward-Scratch (keine Allokation im Hot Path)
    pub sc_h1: Box<[f32]>, // (H)  d_pre_normed2
    pub sc_h2: Box<[f32]>, // (H)  dz + bptt, Input für cell.backward
    pub sc_u2: Box<[f32]>, // (U)  d_gate_act
    pub sc_u3: Box<[f32]>, // (U)  d_value  →  d_gate_pre
}

impl SLSTMBlock {
    pub fn new(hidden_size: usize, up_size: usize) -> Self {
        let h = hidden_size;
        let u = up_size;
        // Xavier-ähnliche Skalierung, identisch zur alten Initialisierung.
        let scale_up = (6.0 / (h as f32 + u as f32)).sqrt();
        let scale_dn = (6.0 / (u as f32 + h as f32)).sqrt(); // gleich, aber explizit

        // LinearLayer::new würde zufällige Biases erzeugen — wir wollen Nullbiases.
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
            cell: SLSTMLayer::new(h, h),
            post_cell_norm: RMSNorm::new(h),
            pre_norm2: RMSNorm::new(h),
            lin_gate: make_lin(h, u, scale_up),
            lin_value: make_lin(h, u, scale_up),
            // Down-Projektion kleiner init → Block startet nahe Identität.
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
        post_cell_norm: RMSNorm,
        pre_norm2: RMSNorm,
        cell: SLSTMLayer,
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
            post_cell_norm,
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

    pub fn forward(&mut self, input: &[f32], cache: &mut SLSTMBlockCache) {
        let u = self.up_size;

        self.pre_norm1.forward_into(input, &mut cache.pre_norm1);
        self.cell.forward(&cache.pre_norm1.output, &mut cache.cell);
        self.post_cell_norm
            .forward_into(&cache.cell.h, &mut cache.post_cell_norm);

        for i in 0..self.hidden_size {
            cache.z[i] = input[i] + cache.post_cell_norm.output[i];
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

        for i in 0..self.hidden_size {
            cache.output[i] = cache.z[i] + cache.lin_down.output[i];
        }
    }

    pub fn backward(&mut self, delta: &mut [f32], cache: &mut SLSTMBlockCache) {
        let h = self.hidden_size;
        let u = self.up_size;

        cache.dx.copy_from_slice(delta);

        self.lin_down.backward(delta, &mut cache.lin_down);

        for j in 0..u {
            self.sc_u2[j] = cache.lin_down.dx[j] * cache.lin_value.output[j];
            self.sc_u3[j] = cache.lin_down.dx[j] * cache.gate_act[j];
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

        self.post_cell_norm
            .backward_into(&cache.dx, &mut cache.post_cell_norm);
        self.sc_h2.copy_from_slice(&cache.post_cell_norm.dx);
        self.cell.backward(&mut self.sc_h2, &mut cache.cell);

        let d_pre_normed: &[f32] = &cache.cell.dconcat[..h];
        self.pre_norm1
            .backward_into(d_pre_normed, &mut cache.pre_norm1);

        for i in 0..h {
            cache.dx[i] += cache.pre_norm1.dx[i];
        }
    }

    pub fn alloc_cache(&self) -> SLSTMBlockCache {
        let h = self.hidden_size;
        let u = self.up_size;
        SLSTMBlockCache {
            pre_norm1: self.pre_norm1.alloc_cache(),
            cell: self.cell.alloc_cache(),
            post_cell_norm: self.post_cell_norm.alloc_cache(),
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

impl NnLayer for SLSTMBlock {
    //type Cache = SLSTMBlockCache;
    fn input_size(&self) -> usize {
        self.hidden_size
    }
    fn output_size(&self) -> usize {
        self.hidden_size
    }

    fn forward(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<SLSTMBlockCache>()
            .expect("SLSTMBlock::forward — expected SLSTMBlockCache");
        SLSTMBlock::forward(self, input, c);
    }

    fn backward(&mut self, delta: &mut [f32], cache: &mut dyn DynCache) {
        let c = cache
            .as_any_mut()
            .downcast_mut::<SLSTMBlockCache>()
            .expect("SLSTMBlock::backward — expected SLSTMBlockCache");
        SLSTMBlock::backward(self, delta, c);
    }

    fn make_cache(&self) -> Box<dyn DynCache> {
        Box::new(SLSTMBlock::alloc_cache(self))
    }

    fn layer_tag(&self) -> u8 {
        11
    }

    fn save(&self, w: &mut dyn io::Write) -> io::Result<()> {
        write_u32(w, self.up_size as u32)?;
        write_f32_slice(w, &self.pre_norm1.gamma)?;
        write_f32_slice(w, &self.post_cell_norm.gamma)?;
        write_f32_slice(w, &self.pre_norm2.gamma)?;
        self.cell.save(w)?;
        write_matrix(w, &self.lin_gate.weights)?;
        write_f32_slice(w, &self.lin_gate.biases)?;
        write_matrix(w, &self.lin_value.weights)?;
        write_f32_slice(w, &self.lin_value.biases)?;
        write_matrix(w, &self.lin_down.weights)?;
        write_f32_slice(w, &self.lin_down.biases)
    }

    fn reset_state(&mut self) {
        self.cell.reset_state();
    }

    fn reset_bptt_state(&mut self) {
        self.cell.reset_bptt_state();
    }

    fn accumulate_init_grad(&mut self) {
        self.cell.accumulate_init_grad();
    }

    fn state_size(&self) -> usize { self.cell.state_size() }

    fn inject_state(&mut self, buf: &[f32], offset: usize) -> usize {
        self.cell.inject_state(buf, offset)
    }

    fn collect_bptt_grad(&self, buf: &mut [f32], offset: usize) -> usize {
        self.cell.collect_bptt_grad(buf, offset)
    }

    fn apply_grads(&mut self, lr: f32) {
        self.pre_norm1.apply_grads(lr);
        self.cell.apply_grads(lr);
        self.post_cell_norm.apply_grads(lr);
        self.pre_norm2.apply_grads(lr);
        self.lin_gate.apply_grads(lr);
        self.lin_value.apply_grads(lr);
        self.lin_down.apply_grads(lr);
    }

    fn clear_grads(&mut self) {
        self.pre_norm1.clear_grads();
        self.cell.clear_grads();
        self.post_cell_norm.clear_grads();
        self.pre_norm2.clear_grads();
        self.lin_gate.clear_grads();
        self.lin_value.clear_grads();
        self.lin_down.clear_grads();
    }
}
