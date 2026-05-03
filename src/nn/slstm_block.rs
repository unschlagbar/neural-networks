// slstm_block.rs ── xLSTM-style sLSTM block
//
// Per-Timestep Architektur:
//
//     x ──┬─► RMSNorm(pre) ─► sLSTM-Zelle ─► RMSNorm(post) ─► SwiGLU ─┐
//         │                                                            │
//         └───────────────────────────────── (+) ◄────────────────────┘
//
// SwiGLU-MLP(h) = lin_down · ( SiLU(lin_gate · h) ⊙ (lin_value · h) )
//
// Bausteine (alle statisch, kein dyn):
//   pre_norm  : RMSNorm  (H)
//   cell      : SLSTMLayer  (H→H)
//   post_norm : RMSNorm  (H)
//   lin_gate  : LinearLayer  (H→U)
//   lin_value : LinearLayer  (H→U)
//   lin_down  : LinearLayer  (U→H)
//
// Shape-Konvention:  H = hidden_size,  U = up_size  (typisch U = 8·H/3).
//
// BPTT-Protokoll:
//   cell.dh_bptt wird intern verwaltet — nach der Post-Norm-Rückwärtsphase
//   addieren wir es zu dh, bevor cell.backward aufgerufen wird.
//   Nach außen gibt der Block `bptt_hidden_grad() = None` zurück.
//
// Binär-Format (`save`) identisch zum alten Layout — kein Checkpoint-Bruch:
//   up_size : u32
//   pre_norm.gamma, post_norm.gamma : f32_vec(H)
//   cell: wz, wi, wf, wo, b, h_init, c_init
//   lin_gate:  weights (H×U), biases (U)
//   lin_value: weights (H×U), biases (U)
//   lin_down:  weights (U×H), biases (H)

use std::{any::Any, io};

use crate::{
    nn::{
        add_vec_in_place,
        linear::{LinearCache, LinearLayer},
        rms_norm::{RMSNorm, RMSNormCache},
        slstm::{SLSTMCache, SLSTMLayer},
        sub_vec_in_place,
    },
    nn_layer::{DynCache, NnLayer},
    saving::{write_f32_slice, write_matrix, write_u32},
};

// ── SiLU-Helfer ──────────────────────────────────────────────────────────────

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

// ── Cache ─────────────────────────────────────────────────────────────────────

pub struct SLSTMBlockCache {
    // Pre-Norm
    pub pre_norm: RMSNormCache, // .output = pre_normed (H)

    // sLSTM-Zelle
    pub cell: SLSTMCache,

    // Post-Norm
    pub post_norm: RMSNormCache, // .output = post_normed (H)

    // SwiGLU
    pub lin_gate: LinearCache,  // .output = gate_pre  (U)
    pub gate_act: Box<[f32]>,   // SiLU(gate_pre)       (U)
    pub lin_value: LinearCache, // .output = value       (U)
    pub mixed: Box<[f32]>,      // gate_act ⊙ value      (U)
    pub lin_down: LinearCache,  // .output = down_out    (H)

    pub output: Box<[f32]>, // input + down_out      (H)
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

// ── Layer ─────────────────────────────────────────────────────────────────────

pub struct SLSTMBlock {
    pub hidden_size: usize,
    pub up_size: usize,

    pub pre_norm: RMSNorm,      // (H)
    pub cell: SLSTMLayer,       // (H→H)
    pub post_norm: RMSNorm,     // (H)
    pub lin_gate: LinearLayer,  // (H→U)
    pub lin_value: LinearLayer, // (H→U)
    pub lin_down: LinearLayer,  // (U→H)

    // Backward-Scratch (keine Allokation im Hot Path)
    pub sc_h1: Box<[f32]>, // (H)  d_post_normed / dh
    pub sc_h2: Box<[f32]>, // (H)  dh + bptt, Input für cell.backward
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

            pre_norm: RMSNorm::new(h),
            cell: SLSTMLayer::new(h, h),
            post_norm: RMSNorm::new(h),
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
        pre_norm: RMSNorm,
        post_norm: RMSNorm,
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
            pre_norm,
            cell,
            post_norm,
            lin_gate,
            lin_value,
            lin_down,
            sc_h1: vec![0.0; h].into(),
            sc_h2: vec![0.0; h].into(),
            sc_u2: vec![0.0; u].into(),
            sc_u3: vec![0.0; u].into(),
        }
    }

    // ── forward ──────────────────────────────────────────────────────────────

    pub fn forward(&mut self, input: &[f32], cache: &mut SLSTMBlockCache) {
        let u = self.up_size;

        // 1. Pre-Norm  →  cache.pre_norm.output
        self.pre_norm.forward_into(input, &mut cache.pre_norm);

        // 2. sLSTM-Zelle  →  cache.cell.h
        self.cell.forward(&cache.pre_norm.output, &mut cache.cell);

        // 3. Post-Norm  →  cache.post_norm.output
        self.post_norm
            .forward_into(&cache.cell.h, &mut cache.post_norm);

        // 4. SwiGLU
        //    gate_pre = lin_gate(post_normed)      → cache.lin_gate.output
        //    value    = lin_value(post_normed)      → cache.lin_value.output
        self.lin_gate
            .forward(&cache.post_norm.output, &mut cache.lin_gate);
        self.lin_value
            .forward(&cache.post_norm.output, &mut cache.lin_value);

        //    gate_act = SiLU(gate_pre);  mixed = gate_act ⊙ value
        for j in 0..u {
            cache.gate_act[j] = silu(cache.lin_gate.output[j]);
            cache.mixed[j] = cache.gate_act[j] * cache.lin_value.output[j];
        }

        //    down_out = lin_down(mixed)             → cache.lin_down.output
        self.lin_down.forward(&cache.mixed, &mut cache.lin_down);

        // 5. Residual
        for i in 0..self.hidden_size {
            cache.output[i] = input[i] + cache.lin_down.output[i];
        }
    }

    // ── backward ─────────────────────────────────────────────────────────────
    //
    // `delta` = dL/d(output).
    //
    // (a) Residual-Anteil:  cache.dx  = delta
    // (b) SwiGLU rückwärts: lin_down → lin_value/lin_gate
    //     d_mixed       = lin_down.backward(delta)           → cache.lin_down.dx
    //     d_gate_act    = d_mixed ⊙ value                    → sc_u2
    //     d_value       = d_mixed ⊙ gate_act                 → sc_u3
    //     lin_value.backward(d_value)  → cache.lin_value.dx  (erster Teil d_post_normed)
    //     d_gate_pre    = d_gate_act ⊙ SiLU'(gate_pre)       → sc_u3
    //     lin_gate.backward(d_gate_pre) → cache.lin_gate.dx  (zweiter Teil d_post_normed)
    //     d_post_normed = cache.lin_value.dx + cache.lin_gate.dx → sc_h1
    // (c) Post-Norm rückwärts: dh  = post_norm.backward(d_post_normed) → cache.post_norm.dx
    // (d) BPTT-Injektion + cell.backward: sc_h2 = dh + cell.dh_bptt
    //     cache.cell.dconcat[..H] = d(pre_normed)
    // (e) Pre-Norm rückwärts: dx_rms = pre_norm.backward(d_pre_normed) → cache.pre_norm.dx
    // (f) cache.dx += dx_rms

    pub fn backward(&mut self, delta: &mut [f32], cache: &mut SLSTMBlockCache) {
        let h = self.hidden_size;
        let u = self.up_size;

        // (a)
        cache.dx.copy_from_slice(delta);

        // (b) SwiGLU rückwärts ────────────────────────────────────────────────

        // lin_down: Gradienten + d_mixed in cache.lin_down.dx
        self.lin_down.backward(delta, &mut cache.lin_down);

        // Elementweise Aufspaltung d_mixed → d_gate_act (sc_u2) und d_value (sc_u3)
        for j in 0..u {
            self.sc_u2[j] = cache.lin_down.dx[j] * cache.lin_value.output[j]; // d_gate_act
            self.sc_u3[j] = cache.lin_down.dx[j] * cache.gate_act[j]; // d_value
        }

        // lin_value: Gradienten + W_valueᵀ · d_value → cache.lin_value.dx
        self.lin_value
            .backward(&mut self.sc_u3, &mut cache.lin_value);

        // SiLU-Ableitung: d_gate_pre = d_gate_act · SiLU'(gate_pre)   → sc_u3
        for j in 0..u {
            self.sc_u3[j] = self.sc_u2[j] * silu_prime(cache.lin_gate.output[j]);
        }

        // lin_gate: Gradienten + W_gateᵀ · d_gate_pre → cache.lin_gate.dx
        self.lin_gate.backward(&mut self.sc_u3, &mut cache.lin_gate);

        // d_post_normed = Summe beider Rückwärtspfade
        for i in 0..h {
            self.sc_h1[i] = cache.lin_value.dx[i] + cache.lin_gate.dx[i];
        }

        // (c) Post-Norm rückwärts → dh in cache.post_norm.dx
        self.post_norm
            .backward_into(&self.sc_h1, &mut cache.post_norm);

        // (d) BPTT-Injektion + cell.backward
        self.sc_h2.copy_from_slice(&cache.post_norm.dx);
        add_vec_in_place(&mut self.sc_h2, &self.cell.dh_bptt);
        self.cell.backward(&mut self.sc_h2, &mut cache.cell);
        // → cache.cell.dconcat[..H] = d(pre_normed)

        // (e) Pre-Norm rückwärts → dx_rms in cache.pre_norm.dx
        let d_pre_normed: &[f32] = &cache.cell.dconcat[..h];
        self.pre_norm
            .backward_into(d_pre_normed, &mut cache.pre_norm);

        // (f) Residual + RMSNorm-Pfad zusammenführen
        for i in 0..h {
            cache.dx[i] += cache.pre_norm.dx[i];
        }
    }

    pub fn alloc_cache(&self) -> SLSTMBlockCache {
        let h = self.hidden_size;
        let u = self.up_size;
        SLSTMBlockCache {
            pre_norm: self.pre_norm.alloc_cache(),
            cell: self.cell.alloc_cache(),
            post_norm: self.post_norm.alloc_cache(),
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

// ── NnLayer-Impl ─────────────────────────────────────────────────────────────

impl NnLayer for SLSTMBlock {
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

    fn layer_tag(&self) -> u8 {
        11
    } // TAG_SLSTM_BLOCK

    fn save(&self, w: &mut dyn io::Write) -> io::Result<()> {
        // Binärformat identisch zum alten Layout.
        write_u32(w, self.up_size as u32)?;

        write_f32_slice(w, &self.pre_norm.gamma)?;
        write_f32_slice(w, &self.post_norm.gamma)?;

        // Zell-Gewichte (Reihenfolge wie SLSTMLayer::save / load_slstm_block)
        write_matrix(w, &self.cell.wz)?;
        write_matrix(w, &self.cell.wi)?;
        write_matrix(w, &self.cell.wf)?;
        write_matrix(w, &self.cell.wo)?;
        write_matrix(w, &self.cell.b)?;
        write_f32_slice(w, &self.cell.h_init)?;
        write_f32_slice(w, &self.cell.c_init)?;

        // SwiGLU-Projektionen
        write_matrix(w, &self.lin_gate.weights)?;
        write_f32_slice(w, &self.lin_gate.biases)?;
        write_matrix(w, &self.lin_value.weights)?;
        write_f32_slice(w, &self.lin_value.biases)?;
        write_matrix(w, &self.lin_down.weights)?;
        write_f32_slice(w, &self.lin_down.biases)
    }

    fn make_cache(&self) -> Box<dyn DynCache> {
        Box::new(self.alloc_cache())
    }

    fn input_size(&self) -> usize {
        self.hidden_size
    }
    fn output_size(&self) -> usize {
        self.hidden_size
    }

    // ── Gradient-Bookkeeping ──────────────────────────────────────────────────

    fn apply_grads(&mut self, lr: f32) {
        sub_vec_in_place(&mut self.pre_norm.gamma, &self.pre_norm.grads_gamma, lr);
        sub_vec_in_place(&mut self.post_norm.gamma, &self.post_norm.grads_gamma, lr);
        self.cell.apply_grads(lr);
        self.lin_gate.apply_grads(lr);
        self.lin_value.apply_grads(lr);
        self.lin_down.apply_grads(lr);
    }

    fn clear_grads(&mut self) {
        self.pre_norm.grads_gamma.fill(0.0);
        self.post_norm.grads_gamma.fill(0.0);
        self.cell.clear_grads();
        self.lin_gate.clear_grads();
        self.lin_value.clear_grads();
        self.lin_down.clear_grads();
    }

    fn scale_grads(&mut self, scale: f32) {
        self.pre_norm
            .grads_gamma
            .iter_mut()
            .for_each(|v| *v *= scale);
        self.post_norm
            .grads_gamma
            .iter_mut()
            .for_each(|v| *v *= scale);
        self.cell.scale_grads(scale);
        self.lin_gate.scale_grads(scale);
        self.lin_value.scale_grads(scale);
        self.lin_down.scale_grads(scale);
    }

    // ── Zustand ───────────────────────────────────────────────────────────────

    fn reset_state(&mut self) {
        self.cell.reset_state();
    }
    fn zero_bptt_state(&mut self) {
        self.cell.zero_bptt_state();
    }
    fn accumulate_init_grad(&mut self) {
        self.cell.accumulate_init_grad();
    }
}
