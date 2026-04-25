// slstm_block.rs ── xLSTM-style sLSTM block
//
// Per-Timestep Architektur:
//
//     x ──┬─► RMSNorm(pre) ─► sLSTM-Zelle ─► RMSNorm(post) ─► SwiGLU ─┐
//         │                                                            │
//         └───────────────────────────────── (+) ◄────────────────────┘
//
// SwiGLU-MLP(h) = W_down · ( SiLU(W_gate · h + b_gate) ⊙ (W_value · h + b_value) )
//                 + b_down                                          (⊙ = elementwise)
//
// Das ist der Teil der im xLSTM-Paper (Beck et al. 2024) als "post up-projection"
// beschrieben wird — er liefert die eigentliche Channel-Mixing-Schicht UND das
// Gating zwischen zwei rekurrenten Zellen. Ohne ihn stapelst du nur nackte
// sLSTM-Zellen hintereinander, was praktisch null Repräsentations-Gewinn bringt.
//
// Verglichen mit dem alten `build_new_normal_model`:
//   • pre-norm stabilisiert die Gradienten durch tiefe Stapel
//   • post-norm begrenzt die Varianz des Zellausgangs (wichtig bei sLSTM,
//     wo h = o·c/max(|n|,1) stark streuen kann)
//   • SwiGLU macht echtes Channel-Mixing MIT gesteuertem Gate
//   • Residual macht ≥ 4 Block-Stapel trainierbar
//
// Die Zelle selbst wird via Komposition wiederverwendet (`SLSTMLayer` +
// `SLSTMCache`), es gibt keinen duplizierten BPTT-Code.
//
// Shape-Konvention:  H = hidden_size,  U = up_size  (typisch U = 4·H / 3).
//
// Wichtig zum BPTT-Protokoll:
//   Die Zelle schreibt ihr `dh_bptt` (dL/d h_{t-1}) bei jedem Backward — das
//   MUSS beim nächsten (früheren) Backward zurück ins Delta injiziert werden.
//   Sequential macht das automatisch nur für Zellen, die direkt im Layer-
//   Stack liegen. Weil unsere Zelle in einem Block gekapselt ist, erledigen
//   wir diese Injektion hier selbst: nach der Post-Norm-Rückwärtsphase
//   addieren wir `cell.dh_bptt` auf `dh` bevor wir `cell.backward` rufen.
//   Nach außen gibt der Block daher `bptt_hidden_grad() = None` zurück.

use iron_oxide::collections::Matrix;
use std::{any::Any, io};

use crate::{
    lstm::{add_vec_in_place, sub_in_place, sub_vec_in_place},
    nn_layer::{DynCache, NnLayer},
    saving::{write_f32_slice, write_matrix, write_u32},
    slstm::{SLSTMCache, SLSTMLayer},
};

const EPS: f32 = 1e-6;

// ── Cache ────────────────────────────────────────────────────────────────────

pub struct SLSTMBlockCache {
    // rohe Eingabe (für Residual + Backward durch Pre-Norm)
    pub input: Box<[f32]>, // (H)

    // Pre-Norm
    pub pre_x_hat: Box<[f32]>, // (H)  x / rms
    pub pre_inv_rms: f32,
    pub pre_normed: Box<[f32]>, // (H)  pre_gamma · pre_x_hat  (Input der Zelle)

    // sLSTM-Zelle
    pub cell: SLSTMCache,

    // Post-Norm auf cell.h
    pub post_x_hat: Box<[f32]>,
    pub post_inv_rms: f32,
    pub post_normed: Box<[f32]>, // (H)  → Input der SwiGLU

    // SwiGLU
    pub gate_pre: Box<[f32]>, // (U)  W_gate · post_normed + b_gate
    pub gate_act: Box<[f32]>, // (U)  SiLU(gate_pre)
    pub value: Box<[f32]>,    // (U)  W_value · post_normed + b_value
    pub mixed: Box<[f32]>,    // (U)  gate_act ⊙ value
    pub down_out: Box<[f32]>, // (H)  W_down · mixed + b_down

    pub output: Box<[f32]>, // (H)  = input + down_out
    pub dx: Box<[f32]>,     // (H)  dL/d(input)
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

// ── Layer ────────────────────────────────────────────────────────────────────

pub struct SLSTMBlock {
    pub hidden_size: usize,
    pub up_size: usize,

    // Pre-Norm
    pub pre_gamma: Box<[f32]>,
    pub pre_gamma_grad: Box<[f32]>,

    // sLSTM-Zelle (input_size = hidden_size = H, hidden_size = H)
    pub cell: SLSTMLayer,

    // Post-Norm
    pub post_gamma: Box<[f32]>,
    pub post_gamma_grad: Box<[f32]>,

    // SwiGLU: zwei parallele Up-Projektionen + eine Down-Projektion
    pub w_gate: Matrix, // (H, U)
    pub b_gate: Box<[f32]>,
    pub w_gate_grad: Matrix,
    pub b_gate_grad: Box<[f32]>,

    pub w_value: Matrix, // (H, U)
    pub b_value: Box<[f32]>,
    pub w_value_grad: Matrix,
    pub b_value_grad: Box<[f32]>,

    pub w_down: Matrix, // (U, H)
    pub b_down: Box<[f32]>,
    pub w_down_grad: Matrix,
    pub b_down_grad: Box<[f32]>,

    // Scratch-Puffer für Backward (keine Allokation im Hot Path)
    pub sc_h1: Box<[f32]>, // (H)  d_post_normed  /  dh
    pub sc_h2: Box<[f32]>, // (H)  dh-Scratch für cell.backward
    pub sc_u1: Box<[f32]>, // (U)  d_mixed
    pub sc_u2: Box<[f32]>, // (U)  d_gate_act
    pub sc_u3: Box<[f32]>, // (U)  d_value → später d_gate_pre
}

// ── kleine Nichtlinearitäts-Helfer ──────────────────────────────────────────

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
    // silu'(z) = σ(z) · (1 + z·(1 − σ(z)))
    let s = stable_sigmoid(pre);
    s * (1.0 + pre * (1.0 - s))
}

impl SLSTMBlock {
    pub fn new(hidden_size: usize, up_size: usize) -> Self {
        let scale_up = (6.0 / (hidden_size as f32 + up_size as f32)).sqrt();
        // Down-Projektion etwas kleiner initialisieren → Residual-Pfad dominiert
        // am Anfang, der Block startet quasi als Identität. (Standard-Trick.)
        let scale_dn = (6.0 / (up_size as f32 + hidden_size as f32)).sqrt() * 0.5;

        Self {
            hidden_size,
            up_size,

            pre_gamma: vec![1.0; hidden_size].into(),
            pre_gamma_grad: vec![0.0; hidden_size].into(),

            cell: SLSTMLayer::new(hidden_size, hidden_size),

            post_gamma: vec![1.0; hidden_size].into(),
            post_gamma_grad: vec![0.0; hidden_size].into(),

            w_gate: Matrix::random(hidden_size, up_size, scale_up),
            b_gate: vec![0.0; up_size].into(),
            w_gate_grad: Matrix::zeros(hidden_size, up_size),
            b_gate_grad: vec![0.0; up_size].into(),

            w_value: Matrix::random(hidden_size, up_size, scale_up),
            b_value: vec![0.0; up_size].into(),
            w_value_grad: Matrix::zeros(hidden_size, up_size),
            b_value_grad: vec![0.0; up_size].into(),

            w_down: Matrix::random(up_size, hidden_size, scale_dn),
            b_down: vec![0.0; hidden_size].into(),
            w_down_grad: Matrix::zeros(up_size, hidden_size),
            b_down_grad: vec![0.0; hidden_size].into(),

            sc_h1: vec![0.0; hidden_size].into(),
            sc_h2: vec![0.0; hidden_size].into(),
            sc_u1: vec![0.0; up_size].into(),
            sc_u2: vec![0.0; up_size].into(),
            sc_u3: vec![0.0; up_size].into(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_loaded(
        hidden_size: usize,
        up_size: usize,
        pre_gamma: Box<[f32]>,
        post_gamma: Box<[f32]>,
        cell: SLSTMLayer,
        w_gate: Matrix,
        b_gate: Box<[f32]>,
        w_value: Matrix,
        b_value: Box<[f32]>,
        w_down: Matrix,
        b_down: Box<[f32]>,
    ) -> Self {
        Self {
            hidden_size,
            up_size,
            pre_gamma,
            pre_gamma_grad: vec![0.0; hidden_size].into(),
            cell,
            post_gamma,
            post_gamma_grad: vec![0.0; hidden_size].into(),
            w_gate,
            b_gate,
            w_gate_grad: Matrix::zeros(hidden_size, up_size),
            b_gate_grad: vec![0.0; up_size].into(),
            w_value,
            b_value,
            w_value_grad: Matrix::zeros(hidden_size, up_size),
            b_value_grad: vec![0.0; up_size].into(),
            w_down,
            b_down,
            w_down_grad: Matrix::zeros(up_size, hidden_size),
            b_down_grad: vec![0.0; hidden_size].into(),
            sc_h1: vec![0.0; hidden_size].into(),
            sc_h2: vec![0.0; hidden_size].into(),
            sc_u1: vec![0.0; up_size].into(),
            sc_u2: vec![0.0; up_size].into(),
            sc_u3: vec![0.0; up_size].into(),
        }
    }

    // ── forward ─────────────────────────────────────────────────────────────

    pub fn forward(&mut self, input: &[f32], cache: &mut SLSTMBlockCache) {
        let h = self.hidden_size;
        let u = self.up_size;

        cache.input.copy_from_slice(input);

        // 1. Pre-Norm
        let ss: f32 = input.iter().map(|&v| v * v).sum();
        let pre_inv_rms = 1.0 / (ss / h as f32 + EPS).sqrt();
        cache.pre_inv_rms = pre_inv_rms;
        for i in 0..h {
            cache.pre_x_hat[i] = input[i] * pre_inv_rms;
            cache.pre_normed[i] = self.pre_gamma[i] * cache.pre_x_hat[i];
        }

        // 2. sLSTM-Zelle — Input ist das pre-normierte x
        self.cell.forward(&cache.pre_normed, &mut cache.cell);
        // Output der Zelle sitzt in cache.cell.h  (Shape H)

        // 3. Post-Norm auf cell.h
        let ss2: f32 = cache.cell.h.iter().map(|&v| v * v).sum();
        let post_inv_rms = 1.0 / (ss2 / h as f32 + EPS).sqrt();
        cache.post_inv_rms = post_inv_rms;
        for i in 0..h {
            cache.post_x_hat[i] = cache.cell.h[i] * post_inv_rms;
            cache.post_normed[i] = self.post_gamma[i] * cache.post_x_hat[i];
        }

        // 4. SwiGLU
        //   gate_pre = W_gate · post_normed + b_gate
        cache.gate_pre.copy_from_slice(&self.b_gate);
        for (i, &x) in cache.post_normed.iter().enumerate() {
            let row = &self.w_gate[i];
            for j in 0..u {
                cache.gate_pre[j] += x * row[j];
            }
        }
        //   value = W_value · post_normed + b_value
        cache.value.copy_from_slice(&self.b_value);
        for (i, &x) in cache.post_normed.iter().enumerate() {
            let row = &self.w_value[i];
            for j in 0..u {
                cache.value[j] += x * row[j];
            }
        }
        //   gate_act = SiLU(gate_pre);  mixed = gate_act ⊙ value
        for j in 0..u {
            cache.gate_act[j] = silu(cache.gate_pre[j]);
            cache.mixed[j] = cache.gate_act[j] * cache.value[j];
        }
        //   down_out = W_down · mixed + b_down
        cache.down_out.copy_from_slice(&self.b_down);
        for (i, &m) in cache.mixed.iter().enumerate() {
            let row = &self.w_down[i];
            for j in 0..h {
                cache.down_out[j] += m * row[j];
            }
        }

        // 5. Residual
        for i in 0..h {
            cache.output[i] = input[i] + cache.down_out[i];
        }
    }

    // ── backward ────────────────────────────────────────────────────────────
    //
    // `delta` = dL/d(output). Wir:
    //   (a) nehmen delta direkt als Residual-Beitrag zu dx,
    //   (b) gehen durch SwiGLU (down → mixed → gate_act/value → W_gate/W_value),
    //   (c) Post-Norm rückwärts → dh,
    //   (d) cell.dh_bptt addieren, cell.backward rufen,
    //   (e) Pre-Norm rückwärts → dx_rms,
    //   (f) cache.dx = delta + dx_rms.

    pub fn backward(&mut self, delta: &mut [f32], cache: &mut SLSTMBlockCache) {
        let h = self.hidden_size;
        let u = self.up_size;

        // (a) Residual-Beitrag schon jetzt in cache.dx parken — am Ende += dx_rms.
        cache.dx.copy_from_slice(delta);

        // ── (b) SwiGLU rückwärts ────────────────────────────────────────────

        // down: d_mixed = W_downᵀ · delta
        for (i, m) in self.sc_u1.iter_mut().enumerate() {
            let row = &self.w_down[i];
            let mut s = 0.0;
            for j in 0..h {
                s += row[j] * delta[j];
            }
            *m = s;
        }
        self.w_down_grad.add_outer(&cache.mixed, delta);
        add_vec_in_place(&mut self.b_down_grad, delta);

        // mixed = gate_act ⊙ value
        //   d_gate_act = d_mixed · value       → sc_u2
        //   d_value    = d_mixed · gate_act    → sc_u3
        for j in 0..u {
            self.sc_u2[j] = self.sc_u1[j] * cache.value[j];
            self.sc_u3[j] = self.sc_u1[j] * cache.gate_act[j];
        }

        // value = W_value · post_normed + b_value
        //   W_value_grad += outer(post_normed, d_value)
        //   b_value_grad += d_value
        //   d_post_normed (init) = W_valueᵀ · d_value       → sc_h1
        self.w_value_grad.add_outer(&cache.post_normed, &self.sc_u3);
        add_vec_in_place(&mut self.b_value_grad, &self.sc_u3);
        for (i, dp) in self.sc_h1.iter_mut().enumerate() {
            let row = &self.w_value[i];
            let mut s = 0.0;
            for j in 0..u {
                s += row[j] * self.sc_u3[j];
            }
            *dp = s;
        }

        // gate_act = SiLU(gate_pre)
        //   d_gate_pre = d_gate_act · SiLU'(gate_pre)   — überschreibt sc_u3
        for j in 0..u {
            self.sc_u3[j] = self.sc_u2[j] * silu_prime(cache.gate_pre[j]);
        }

        // gate_pre = W_gate · post_normed + b_gate
        //   W_gate_grad += outer(post_normed, d_gate_pre)
        //   b_gate_grad += d_gate_pre
        //   d_post_normed += W_gateᵀ · d_gate_pre
        self.w_gate_grad.add_outer(&cache.post_normed, &self.sc_u3);
        add_vec_in_place(&mut self.b_gate_grad, &self.sc_u3);
        for (i, dp) in self.sc_h1.iter_mut().enumerate() {
            let row = &self.w_gate[i];
            let mut s = 0.0;
            for j in 0..u {
                s += row[j] * self.sc_u3[j];
            }
            *dp += s;
        }
        // sc_h1 enthält jetzt das vollständige d_post_normed.

        // ── (c) Post-Norm rückwärts ─────────────────────────────────────────
        //
        //   post_normed[i] = post_gamma[i] · post_x_hat[i]
        //   post_x_hat[i]  = cell.h[i] · post_inv_rms
        //
        //   post_gamma_grad[i] += d_post_normed[i] · post_x_hat[i]
        //   S = Σⱼ post_gamma[j] · d_post_normed[j] · post_x_hat[j]
        //   dh[i] = post_inv_rms · ( post_gamma[i]·d_post_normed[i]
        //                            − post_x_hat[i] · S/H )
        let mut s_post = 0.0;
        for i in 0..h {
            self.post_gamma_grad[i] += self.sc_h1[i] * cache.post_x_hat[i];
            s_post += self.post_gamma[i] * self.sc_h1[i] * cache.post_x_hat[i];
        }
        let s_post_over_h = s_post / h as f32;
        let post_inv_rms = cache.post_inv_rms;
        for i in 0..h {
            self.sc_h2[i] = post_inv_rms
                * (self.post_gamma[i] * self.sc_h1[i] - cache.post_x_hat[i] * s_post_over_h);
        }

        // ── (d) cell.dh_bptt einfüttern, dann cell.backward ─────────────────
        //
        // Der rekurrente Pfad (cell h_{t-1} → cell xh_t → cell h_t) lebt
        // VOLLSTÄNDIG innerhalb des Blocks. Sequential sieht den Block als
        // nicht-rekurrent (bptt_hidden_grad() = None unten), also müssen wir
        // dh_bptt hier selbst aufsummieren, sonst ignorieren wir BPTT komplett.
        add_vec_in_place(&mut self.sc_h2, &self.cell.dh_bptt);
        self.cell.backward(&mut self.sc_h2, &mut cache.cell);
        // cache.cell.dconcat[..H] enthält jetzt d(pre_normed) = dL/d(Zelleneingang)

        // ── (e) Pre-Norm rückwärts ──────────────────────────────────────────
        let mut s_pre = 0.0;
        for i in 0..h {
            let d_pn = cache.cell.dconcat[i];
            self.pre_gamma_grad[i] += d_pn * cache.pre_x_hat[i];
            s_pre += self.pre_gamma[i] * d_pn * cache.pre_x_hat[i];
        }
        let s_pre_over_h = s_pre / h as f32;
        let pre_inv_rms = cache.pre_inv_rms;

        // ── (f) Pre-Norm-dx + Residual-dx zusammenführen ────────────────────
        for i in 0..h {
            let d_pn = cache.cell.dconcat[i];
            let dx_rms =
                pre_inv_rms * (self.pre_gamma[i] * d_pn - cache.pre_x_hat[i] * s_pre_over_h);
            cache.dx[i] += dx_rms; // cache.dx war schon = delta (Residual)
        }
    }

    pub fn alloc_cache(&self) -> SLSTMBlockCache {
        let h = self.hidden_size;
        let u = self.up_size;
        SLSTMBlockCache {
            input: vec![0.0; h].into(),
            pre_x_hat: vec![0.0; h].into(),
            pre_inv_rms: 0.0,
            pre_normed: vec![0.0; h].into(),
            cell: self.cell.alloc_cache(),
            post_x_hat: vec![0.0; h].into(),
            post_inv_rms: 0.0,
            post_normed: vec![0.0; h].into(),
            gate_pre: vec![0.0; u].into(),
            gate_act: vec![0.0; u].into(),
            value: vec![0.0; u].into(),
            mixed: vec![0.0; u].into(),
            down_out: vec![0.0; h].into(),
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
    } // TAG_SLSTM_BLOCK — neu, siehe loading.rs/new_layer

    fn save(&self, w: &mut dyn io::Write) -> io::Result<()> {
        // hidden_size ist bereits im Sequential-Architekturheader (input_size
        // == output_size == H), daher NICHT nochmal hier. up_size ist block-
        // intern und MUSS geschrieben werden.
        write_u32(w, self.up_size as u32)?;

        write_f32_slice(w, &self.pre_gamma)?;
        write_f32_slice(w, &self.post_gamma)?;

        // Zell-Gewichte — exakt dieselbe Reihenfolge wie `SLSTMLayer::save`
        // damit load_slstm_block den identischen Reader-Code nutzen kann.
        write_matrix(w, &self.cell.wz)?;
        write_matrix(w, &self.cell.wi)?;
        write_matrix(w, &self.cell.wf)?;
        write_matrix(w, &self.cell.wo)?;
        write_matrix(w, &self.cell.b)?;
        write_f32_slice(w, &self.cell.h_init)?;
        write_f32_slice(w, &self.cell.c_init)?;

        write_matrix(w, &self.w_gate)?;
        write_f32_slice(w, &self.b_gate)?;
        write_matrix(w, &self.w_value)?;
        write_f32_slice(w, &self.b_value)?;
        write_matrix(w, &self.w_down)?;
        write_f32_slice(w, &self.b_down)?;
        Ok(())
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

    fn apply_grads(&mut self, lr: f32) {
        sub_vec_in_place(&mut self.pre_gamma, &self.pre_gamma_grad, lr);
        sub_vec_in_place(&mut self.post_gamma, &self.post_gamma_grad, lr);
        self.cell.apply_grads(lr);
        sub_in_place(&mut self.w_gate, &self.w_gate_grad, lr);
        sub_vec_in_place(&mut self.b_gate, &self.b_gate_grad, lr);
        sub_in_place(&mut self.w_value, &self.w_value_grad, lr);
        sub_vec_in_place(&mut self.b_value, &self.b_value_grad, lr);
        sub_in_place(&mut self.w_down, &self.w_down_grad, lr);
        sub_vec_in_place(&mut self.b_down, &self.b_down_grad, lr);
    }

    fn clear_grads(&mut self) {
        self.pre_gamma_grad.fill(0.0);
        self.post_gamma_grad.fill(0.0);
        self.cell.clear_grads();
        self.w_gate_grad.clear();
        self.b_gate_grad.fill(0.0);
        self.w_value_grad.clear();
        self.b_value_grad.fill(0.0);
        self.w_down_grad.clear();
        self.b_down_grad.fill(0.0);
    }

    fn scale_grads(&mut self, scale: f32) {
        self.pre_gamma_grad.iter_mut().for_each(|v| *v *= scale);
        self.post_gamma_grad.iter_mut().for_each(|v| *v *= scale);
        self.cell.scale_grads(scale);
        self.w_gate_grad.scale(scale);
        self.b_gate_grad.iter_mut().for_each(|v| *v *= scale);
        self.w_value_grad.scale(scale);
        self.b_value_grad.iter_mut().for_each(|v| *v *= scale);
        self.w_down_grad.scale(scale);
        self.b_down_grad.iter_mut().for_each(|v| *v *= scale);
    }

    fn reset_state(&mut self) {
        self.cell.reset();
    }

    fn zero_bptt_state(&mut self) {
        self.cell.dh_bptt.fill(0.0);
        self.cell.dc_bptt.fill(0.0);
        self.cell.dn_bptt.fill(0.0);
    }

    fn accumulate_init_grad(&mut self) {
        self.cell.accumulate_init_grad();
    }
}
