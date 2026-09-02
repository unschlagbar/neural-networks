//! Gated Recurrent Transformer modulation (arXiv 2608.15062, §3.3).
//!
//! The backbone's depth is factored into `n_pre + n_rec × R + n_coda` block
//! executions over only `n_pre + n_rec + n_coda` sets of weights. This module owns
//! everything that sits *between* two applications of the shared core — the pieces
//! that make repeating one core R times behave differently at each step instead of
//! collapsing to a fixed point:
//!
//! ```text
//!   h̃⁽ʳ⁾ = W_proj [ h⁽ʳ⁻¹⁾ + εx , h⁽ᵖʳᵉ⁾ ]                              (2)
//!   o⁽ʳ⁾ = B_shared( h̃⁽ʳ⁾ )                                             (3)
//!   h⁽ʳ⁾ = g⁽ʳ⁾ ⊙ h⁽ʳ⁻¹⁾ + (1 − g⁽ʳ⁾) ⊙ o⁽ʳ⁾                            (4)
//!   g⁽ʳ⁾ = σ( f_g([ LN(h⁽ʳ⁻¹⁾), LN(h⁽ᵖʳᵉ⁾) ]) / τ + εg )                (5)
//! ```
//!
//! with `h⁽⁰⁾ = h⁽ᵖʳᵉ⁾`, `f_g = LayerNorm(2d) → Linear(2d→d) → SiLU → Linear(d→d)`,
//! the second Linear's bias initialised to +4 (so `g ≈ 0.98` and the copy branch
//! dominates at the start of training), `τ = 1`, and `εx`, `εg ~ N(0, σ²)` resampled
//! at every step. Eq. (3) is not here — the core blocks belong to the backbone.
//!
//! **Nothing is saved across the forward.** Every intermediate above is a pure
//! function of `(h⁽ʳ⁻¹⁾, h⁽ᵖʳᵉ⁾, seed)`, so the backward recomputes the whole gate
//! rather than keeping it: at `R` recurrences over a chunked sweep, storing it would
//! be eight tensors per (chunk, step) where the backbone already stores the core's
//! own activations. That is why the noise is counter-based (`ops::grt_noise_add`)
//! and why [`LayerNorm`] hands its `inv_std` to the caller — a layer with an internal
//! per-call slot could not be run twice over the same step.
//!
//! Faithful to the reference implementation on three points the paper's prose leaves
//! open: `LN(h)` and `LN(h_pre)` are *separate* norms, `f_g` opens with a third
//! LayerNorm over the concatenation, and `εg` is **one draw per row** broadcast
//! across the feature axis (`randn_like(features[..., :1])`), not per element.

use super::arena::{ParamKind, ParamSlot, TrainingCache};
use super::layer_norm::LayerNorm;
use super::{GTensor, Gpu, linear::Linear, ops};
use crate::tensor::Tensor;

/// Layout and hyperparameters of the recurrent-depth backbone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GrtCfg {
    /// `n_pre` — prelude blocks, run once, producing the anchor `h⁽ᵖʳᵉ⁾`.
    pub pre: usize,
    /// `n_rec` — blocks in the shared core, one set of weights for all `R` steps.
    pub core: usize,
    /// `R` — maximum recurrence depth. Training samples `r ~ U{r_min..=R}` per
    /// window; inference runs `R` unless told otherwise, and any `r ≤ R` is valid,
    /// which is the early-exit property.
    pub r: usize,
    /// Lower end of the training depth sample. The paper's §3.4 says 1; the
    /// reference's shipped configs use 2.
    pub r_min: usize,
    /// `n_coda` — coda blocks, run once on `h⁽ᴿ⁾`.
    pub coda: usize,
    /// Blocks in the slow "anchor" module. `0` is the paper's frozen anchor, where
    /// `h⁽ᵖʳᵉ⁾` is the prelude's output for the whole recurrence.
    pub anchor: usize,
    /// Core applications per anchor update — HRM's `T`. Ignored when `anchor == 0`.
    pub t: usize,
    /// `σ` for both `εx` and `εg`. The reference uses one value for both.
    pub noise_std: f32,
    /// Gate temperature `τ`.
    pub tau: f32,
    /// Init of `f_g`'s output bias in the TIED form. +4 puts `g ≈ 0.98` at step 0.
    pub gate_bias: f32,
    /// Untie the gate into independent forget and input branches (LSTM) instead of
    /// the paper's tied convex blend (GRU). See [`crate::config::GRT_LSTM_GATE`].
    pub lstm_gate: bool,
    /// Untied form: the per-channel forget-gate bias is drawn from `[.0, .1)`.
    pub f_bias: (f32, f32),
    /// Untied form: the input-gate bias.
    pub i_bias: f32,
}

impl GrtCfg {
    /// Anchor updates at a given depth.
    ///
    /// Strictly BETWEEN cycles: the update that would follow the last application is
    /// dead compute, because nothing reads the anchor after it. So a depth of `t` or
    /// less never moves the anchor, which is also what makes `anchor = 0` and
    /// `t >= r` the same model.
    pub fn anchor_updates(&self, steps: usize) -> usize {
        if self.anchor == 0 || self.t == 0 || steps == 0 {
            return 0;
        }
        steps.div_ceil(self.t) - 1
    }

    /// Anchor updates at full depth — the number of weight-distinct slow stages, and
    /// so how many replicas the model holds.
    pub fn max_anchor_updates(&self) -> usize {
        self.anchor_updates(self.r)
    }

    /// Which cycle application `r` belongs to — which anchor it reads.
    ///
    /// Clamped to the updates that actually fire, which is what makes a FROZEN anchor
    /// work: there is only ever anchor 0, while `r / t` keeps counting and would run
    /// off the end of a one-element list. With a moving anchor the clamp is a no-op,
    /// because `(steps - 1) / t == ceil(steps / t) - 1`.
    ///
    /// The forward and the backward must agree on this exactly, so both call it rather
    /// than each computing it.
    pub fn cycle_of(&self, r: usize, steps: usize) -> usize {
        (r / self.t.max(1)).min(self.anchor_updates(steps))
    }

    /// Block executions per forward pass — what an isoFLOPs comparison against a
    /// dense backbone of this many blocks is matched on.
    pub fn executions(&self) -> usize {
        self.pre + self.core * self.r + self.anchor * self.max_anchor_updates() + self.coda
    }

    /// Blocks whose weights actually exist.
    pub fn unique_blocks(&self) -> usize {
        self.pre + self.core + self.anchor + self.coda
    }
}

/// Which stream a noise draw comes from. Keeps `εx` and `εg` from sharing a
/// sequence at the same `(chunk, step)`.
const KIND_STATE: u64 = 0x51;
const KIND_GATE: u64 = 0xA7;
/// `εa` — the noise on the fast state as the slow one reads it (GRAM, arXiv
/// 2605.19376: a stochastic transition on the slow state rather than a deterministic
/// one). Keyed on the UPDATE index, not the recurrence step.
const KIND_ANCHOR: u64 = 0x3D;

/// Seed for one `(window base, recurrence step, stream)`.
///
/// A pure function of its inputs, because the backward has to reproduce the forward's
/// draws exactly — see the module note. SplitMix64's finalizer, so neighbouring steps
/// do not produce correlated streams.
///
/// Deliberately NOT keyed on the backbone chunk: the chunk is a memory-layout choice,
/// and the position WITHIN the window is passed to the kernels as an index offset
/// instead. Keying here would make a run's noise a function of `BACKBONE_CHUNK`.
fn mix_seed(base: u64, step: usize, kind: u64) -> u64 {
    let mut z = base
        .wrapping_add((step as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(kind.wrapping_mul(0x94D0_49BB_1331_11EB));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The prelude anchor's normalization, computed once per chunk and read by every
/// recurrence step's gate.
///
/// `LN(h⁽ᵖʳᵉ⁾)` does not depend on `r`, so running it once and accumulating its
/// gradient over the `R` uses is both cheaper and exactly equivalent to running it
/// per step.
pub struct AnchorNorm {
    /// `LN_p(h_pre)` `[n, d]`.
    pub np: GTensor<f32>,
    /// Per-row `inv_std` of that norm.
    inv: GTensor<f32>,
}

/// One recurrence step's gate forward, as the backward needs it.
///
/// Produced by [`GrtMod::gate`] in the forward (where only `g` is read) and produced
/// *again* by the backward from the same inputs.
pub struct GateFwd {
    /// The gate. `[n, d]` in the tied form; `[n, 2d]` in the untied one, forget half
    /// first.
    pub g: GTensor<f32>,
    /// `LN_x(h⁽ʳ⁻¹⁾)` and its per-row `inv_std`.
    nx: GTensor<f32>,
    inv_x: GTensor<f32>,
    /// `LN_c([LN_x(h), LN_p(h_pre)])` and its `inv_std`. The concatenation itself
    /// is not kept: `LN_c`'s backward rebuilds `x̂` from this output.
    nc: GTensor<f32>,
    inv_c: GTensor<f32>,
    /// `f_g`'s hidden pre-activation and post-activation, `[n, d]`.
    z1: GTensor<f32>,
    a1: GTensor<f32>,
}

/// The parameters of Eqs. (2) and (5). The blocks of Eq. (3) live in the backbone.
pub struct GrtMod {
    /// Backbone width `d`.
    d: usize,
    pub cfg: GrtCfg,
    /// `W_proj ∈ R^{2d×d}` — Eq. (2). Bias frozen at zero: the reference builds it
    /// `bias=False`.
    pub w_proj: Linear,
    /// `LN(h⁽ʳ⁻¹⁾)` and `LN(h⁽ᵖʳᵉ⁾)` — separate norms, as in the reference.
    pub ln_x: LayerNorm,
    pub ln_p: LayerNorm,
    /// `f_g`: LayerNorm(2d) → Linear(2d→d) → SiLU → Linear(d→d).
    pub ln_c: LayerNorm,
    pub fc1: Linear,
    pub fc2: Linear,
    /// `W_inj ∈ R^{d×d}` — HRM's input injection, folding the fast state into the
    /// slow one: `h_pre' = anchor_blocks(h_pre + W_inj(h + εa))`. `Some` iff
    /// `cfg.anchor > 0`.
    ///
    /// **Zero-initialised**, so the first step of training is exactly the frozen-anchor
    /// model plus one more block execution and the model learns how much of `h` to
    /// fold in from nothing. A zero weight still receives gradient (`dW = xᵀdy`); it is
    /// only the output that starts at zero.
    pub w_inj: Option<Linear>,
}

/// Uniform half-width whose standard deviation is `std`.
///
/// The reference initialises every Linear `N(0, 0.02)`; [`Tensor::random`] draws
/// uniform, so match the second moment rather than the shape — what the gate bias of
/// +4 has to dominate at init is the *scale* of `z₂`.
fn uniform_for_std(std: f32) -> f32 {
    std * 3.0f32.sqrt()
}

/// `(output width, bias init)` for the gate head.
///
/// Tied: one output per channel at a constant bias. Untied: forget half drawn per
/// channel from `cfg.f_bias`, input half at `cfg.i_bias` — the sLSTM convention
/// (`nn::slstm`), where the forget bias is `random_range(3.0..6.0)` rather than a
/// constant.
fn gate_head_init(d: usize, cfg: &GrtCfg) -> (usize, Tensor) {
    if !cfg.lstm_gate {
        return (d, Tensor::new(&[d], vec![cfg.gate_bias; d]));
    }
    let (lo, hi) = cfg.f_bias;
    let mid = 0.5 * (lo + hi);
    let f = Tensor::random(&[d], 0.5 * (hi - lo));
    let mut b = vec![cfg.i_bias; 2 * d];
    for (i, v) in f.data.iter().enumerate() {
        b[i] = mid + v;
    }
    (2 * d, Tensor::new(&[2 * d], b))
}

impl GrtMod {
    pub fn new(gpu: &Gpu, d: usize, cfg: GrtCfg) -> Self {
        let init = uniform_for_std(0.02);
        let lin = |input: usize, output: usize, bias: Tensor| {
            let w = Tensor::random(&[input, output], init);
            Linear::from_parts(gpu, &w, &bias)
        };
        let flat = |n: usize, v: f32| Tensor::new(&[n], vec![v; n]);
        // W_proj and f_g's first Linear are `bias=False` in the reference; a frozen
        // zero bias is the same function and keeps them on one Linear type.
        let mut w_proj = lin(2 * d, d, flat(d, 0.0));
        w_proj.freeze_bias();
        let mut fc1 = lin(2 * d, d, flat(d, 0.0));
        fc1.freeze_bias();
        // The gate head: one output per channel in the tied form, two (forget then
        // input) in the untied one. Its bias is the one that trains and the one that
        // is not zero at init.
        let (gate_out, gate_bias) = gate_head_init(d, &cfg);
        Self {
            d,
            cfg,
            w_proj,
            ln_x: LayerNorm::new(gpu, d),
            ln_p: LayerNorm::new(gpu, d),
            ln_c: LayerNorm::new(gpu, 2 * d),
            fc1,
            fc2: lin(d, gate_out, gate_bias),
            w_inj: (cfg.anchor > 0).then(|| {
                let mut l =
                    Linear::from_parts(gpu, &Tensor::new(&[d, d], vec![0.0; d * d]), &flat(d, 0.0));
                l.freeze_bias();
                l
            }),
        }
    }

    /// Width of the gate head: `d` tied, `2d` untied.
    fn gate_width(&self) -> usize {
        if self.cfg.lstm_gate { 2 * self.d } else { self.d }
    }

    /// Rebuild from loaded weights, in [`param_slots`](Self::param_slots) order.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        gpu: &Gpu,
        cfg: GrtCfg,
        w_proj: (&Tensor, &Tensor),
        ln_x: &Tensor,
        ln_p: &Tensor,
        ln_c: &Tensor,
        fc1: (&Tensor, &Tensor),
        fc2: (&Tensor, &Tensor),
        inj: Option<(&Tensor, &Tensor)>,
    ) -> Self {
        let d = ln_x.len();
        // The stored gate head's width IS the mode: `d` tied, `2d` untied. Taking it
        // from the file rather than from `config.rs` is what lets a checkpoint trained
        // under either form keep loading after the constant is flipped.
        let mut cfg = cfg;
        cfg.lstm_gate = fc2.0.cols() == 2 * d;
        // Whether the anchor moves is layout too: the slow module's weights either
        // exist in the file or they do not, and a checkpoint keeps its own form
        // whatever `config.rs` says. `cfg.anchor` (how many blocks) comes from the
        // `grt_anchor` section's length, set by the caller.
        if inj.is_none() {
            cfg.anchor = 0;
        }
        let mut w = Linear::from_parts(gpu, w_proj.0, w_proj.1);
        w.freeze_bias();
        let mut f1 = Linear::from_parts(gpu, fc1.0, fc1.1);
        f1.freeze_bias();
        Self {
            d,
            cfg,
            w_proj: w,
            ln_x: LayerNorm::from_parts(gpu, ln_x),
            ln_p: LayerNorm::from_parts(gpu, ln_p),
            ln_c: LayerNorm::from_parts(gpu, ln_c),
            fc1: f1,
            fc2: Linear::from_parts(gpu, fc2.0, fc2.1),
            w_inj: inj.map(|(w, b)| {
                let mut l = Linear::from_parts(gpu, w, b);
                l.freeze_bias();
                l
            }),
        }
    }

    pub fn width(&self) -> usize {
        self.d
    }

    /// `σ` to use for this call: zero when noise is off, which is what an evaluation
    /// or a sampling pass wants. The reference leaves the draw in at eval; a
    /// forward-only pass here has to be reproducible, so it is switched off instead.
    fn sigma(&self, noise: bool) -> f32 {
        if noise { self.cfg.noise_std } else { 0.0 }
    }

    /// `LN_p(h⁽ᵖʳᵉ⁾)`, once per chunk.
    pub fn anchor(&self, gpu: &Gpu, h_pre: &GTensor<f32>) -> AnchorNorm {
        let (n, d) = h_pre.as_2d();
        assert_eq!(d, self.d, "GrtMod::anchor — width mismatch");
        let mut np = GTensor::uninit(gpu, &[n, d]);
        let mut inv = GTensor::uninit(gpu, &[n]);
        self.ln_p.forward(gpu, h_pre, &mut np, &mut inv);
        AnchorNorm { np, inv }
    }

    /// Eq. (5) — the gate for one recurrence step.
    ///
    /// Deterministic in `(h_prev, anchor, seed)`, which is what lets the backward
    /// call it again instead of the forward keeping the result.
    #[allow(clippy::too_many_arguments)]
    pub fn gate(
        &mut self,
        gpu: &Gpu,
        h_prev: &GTensor<f32>,
        anchor: &AnchorNorm,
        base_seed: u64,
        row0: usize,
        step: usize,
        noise: bool,
    ) -> GateFwd {
        let (n, d) = h_prev.as_2d();
        assert_eq!(d, self.d, "GrtMod::gate — width mismatch");
        let mut nx = GTensor::uninit(gpu, &[n, d]);
        let mut inv_x = GTensor::uninit(gpu, &[n]);
        self.ln_x.forward(gpu, h_prev, &mut nx, &mut inv_x);

        let mut cat = GTensor::uninit(gpu, &[n, 2 * d]);
        ops::grt_cat2(gpu, &mut cat, &nx, &anchor.np);

        let mut nc = GTensor::uninit(gpu, &[n, 2 * d]);
        let mut inv_c = GTensor::uninit(gpu, &[n]);
        self.ln_c.forward(gpu, &cat, &mut nc, &mut inv_c);

        let mut z1 = GTensor::uninit(gpu, &[n, d]);
        self.fc1.forward_shared(gpu, &nc, &mut z1);
        let mut a1 = GTensor::uninit(gpu, &[n, d]);
        ops::grt_silu_forward(gpu, &mut a1, &z1);

        let gw = self.gate_width();
        let mut z2 = GTensor::uninit(gpu, &[n, gw]);
        self.fc2.forward_shared(gpu, &a1, &mut z2);
        let mut g = GTensor::uninit(gpu, &[n, gw]);
        ops::grt_gate_apply(
            gpu,
            &mut g,
            &z2,
            self.cfg.tau,
            self.sigma(noise),
            mix_seed(base_seed, step, KIND_GATE),
            row0,
        );

        GateFwd {
            g,
            nx,
            inv_x,
            nc,
            inv_c,
            z1,
            a1,
        }
    }

    /// `[h⁽ʳ⁻¹⁾ + εx, h⁽ᵖʳᵉ⁾]` — the concatenated input of Eq. (2).
    ///
    /// Split out from [`project`](Self::project) because the backward needs exactly
    /// this and not the GEMM that follows it: `W_proj`'s `dW = XᵀdY` wants `X`, and
    /// re-running the projection to get it would be a wasted matmul per (chunk, step).
    #[allow(clippy::too_many_arguments)]
    pub fn project_input(
        &self,
        gpu: &Gpu,
        h_prev: &GTensor<f32>,
        h_pre: &GTensor<f32>,
        base_seed: u64,
        row0: usize,
        step: usize,
        noise: bool,
    ) -> GTensor<f32> {
        let (n, d) = h_prev.as_2d();
        assert_eq!(d, self.d, "GrtMod::project_input — width mismatch");
        let mut xs = GTensor::uninit(gpu, &[n, d]);
        ops::grt_noise_add(
            gpu,
            &mut xs,
            h_prev,
            self.sigma(noise),
            mix_seed(base_seed, step, KIND_STATE),
            row0 * d,
        );
        let mut pin = GTensor::uninit(gpu, &[n, 2 * d]);
        ops::grt_cat2(gpu, &mut pin, &xs, h_pre);
        pin
    }

    /// Eq. (2) — `h̃⁽ʳ⁾ = W_proj[h⁽ʳ⁻¹⁾ + εx, h⁽ᵖʳᵉ⁾]`, returning `h̃` and the
    /// concatenated input `W_proj`'s backward needs.
    #[allow(clippy::too_many_arguments)]
    pub fn project(
        &mut self,
        gpu: &Gpu,
        h_prev: &GTensor<f32>,
        h_pre: &GTensor<f32>,
        base_seed: u64,
        row0: usize,
        step: usize,
        noise: bool,
    ) -> (GTensor<f32>, GTensor<f32>) {
        let (n, d) = h_prev.as_2d();
        let pin = self.project_input(gpu, h_prev, h_pre, base_seed, row0, step, noise);
        let mut ht = GTensor::uninit(gpu, &[n, d]);
        self.w_proj.forward_shared(gpu, &pin, &mut ht);
        (ht, pin)
    }

    /// Eq. (4) — the gated write-back.
    pub fn blend(
        &self,
        gpu: &Gpu,
        g: &GTensor<f32>,
        h_prev: &GTensor<f32>,
        o: &GTensor<f32>,
    ) -> GTensor<f32> {
        let (n, d) = h_prev.as_2d();
        let mut h = GTensor::uninit(gpu, &[n, d]);
        if self.cfg.lstm_gate {
            ops::grt_blend_lstm(gpu, &mut h, g, h_prev, o);
        } else {
            ops::grt_blend(gpu, &mut h, g, h_prev, o);
        }
        h
    }

    /// Backward of Eq. (4): `d_o` and `d_g` are returned, `dh_prev` is accumulated.
    pub fn blend_backward(
        &self,
        gpu: &Gpu,
        dh: &GTensor<f32>,
        g: &GTensor<f32>,
        h_prev: &GTensor<f32>,
        o: &GTensor<f32>,
        dh_prev: &mut GTensor<f32>,
    ) -> (GTensor<f32>, GTensor<f32>) {
        let (n, d) = h_prev.as_2d();
        let mut d_o = GTensor::uninit(gpu, &[n, d]);
        let mut d_g = GTensor::uninit(gpu, &[n, self.gate_width()]);
        if self.cfg.lstm_gate {
            ops::grt_blend_lstm_bwd(gpu, dh, g, h_prev, o, dh_prev, &mut d_o, &mut d_g);
        } else {
            ops::grt_blend_bwd(gpu, dh, g, h_prev, o, dh_prev, &mut d_o, &mut d_g);
        }
        (d_o, d_g)
    }

    /// Backward of Eq. (2), given `dh̃`. Accumulates `dW_proj`, and adds this step's
    /// taps to `dh_prev` (the noise is additive, so it passes the gradient straight
    /// through) and to `d_h_pre`.
    #[allow(clippy::too_many_arguments)]
    pub fn project_backward(
        &mut self,
        gpu: &Gpu,
        dht: &GTensor<f32>,
        pin: &GTensor<f32>,
        dh_prev: &mut GTensor<f32>,
        d_h_pre: &mut GTensor<f32>,
        cache: &TrainingCache,
    ) {
        let (n, d) = dht.as_2d();
        let mut dpin = GTensor::uninit(gpu, &[n, 2 * d]);
        self.w_proj
            .backward_with_x(gpu, pin, dht, &mut dpin, cache);
        let mut d_xs = GTensor::uninit(gpu, &[n, d]);
        let mut d_hp = GTensor::uninit(gpu, &[n, d]);
        ops::grt_split2(gpu, &dpin, &mut d_xs, &mut d_hp);
        ops::add_assign(gpu, dh_prev, &d_xs);
        ops::add_assign(gpu, d_h_pre, &d_hp);
    }

    /// Backward of Eq. (5), given `dg`. Accumulates `dγ` for all three norms and the
    /// two `f_g` weight matrices, adds the `LN_x` tap to `dh_prev`, and adds the
    /// `LN_p` tap to `d_np` — the anchor norm is shared across the `R` steps, so its
    /// own backward runs once at the end of the chunk
    /// ([`anchor_backward`](Self::anchor_backward)).
    #[allow(clippy::too_many_arguments)]
    pub fn gate_backward(
        &mut self,
        gpu: &Gpu,
        fwd: &GateFwd,
        dg: &GTensor<f32>,
        h_prev: &GTensor<f32>,
        dh_prev: &mut GTensor<f32>,
        d_np: &mut GTensor<f32>,
        cache: &TrainingCache,
    ) {
        let (n, d) = h_prev.as_2d();
        let mut dz2 = GTensor::uninit(gpu, &[n, self.gate_width()]);
        ops::grt_gate_bwd(gpu, &mut dz2, dg, &fwd.g, self.cfg.tau);

        let mut da1 = GTensor::uninit(gpu, &[n, d]);
        self.fc2
            .backward_with_x(gpu, &fwd.a1, &dz2, &mut da1, cache);
        let mut dz1 = GTensor::uninit(gpu, &[n, d]);
        ops::grt_silu_backward(gpu, &mut dz1, &da1, &fwd.z1);

        let mut dnc = GTensor::uninit(gpu, &[n, 2 * d]);
        self.fc1
            .backward_with_x(gpu, &fwd.nc, &dz1, &mut dnc, cache);

        let mut dcat = GTensor::uninit(gpu, &[n, 2 * d]);
        self.ln_c
            .backward(gpu, &dnc, &fwd.nc, &fwd.inv_c, &mut dcat, cache);

        let mut dnx = GTensor::uninit(gpu, &[n, d]);
        let mut dnp = GTensor::uninit(gpu, &[n, d]);
        ops::grt_split2(gpu, &dcat, &mut dnx, &mut dnp);
        ops::add_assign(gpu, d_np, &dnp);

        let mut dh_x = GTensor::uninit(gpu, &[n, d]);
        self.ln_x
            .backward(gpu, &dnx, &fwd.nx, &fwd.inv_x, &mut dh_x, cache);
        ops::add_assign(gpu, dh_prev, &dh_x);
    }

    /// Backward of the once-per-chunk anchor norm, folding the `R` steps' summed
    /// `d_np` into `d_h_pre`.
    pub fn anchor_backward(
        &mut self,
        gpu: &Gpu,
        anchor: &AnchorNorm,
        d_np: &GTensor<f32>,
        d_h_pre: &mut GTensor<f32>,
        cache: &TrainingCache,
    ) {
        let (n, d) = anchor.np.as_2d();
        let mut dh = GTensor::uninit(gpu, &[n, d]);
        self.ln_p
            .backward(gpu, d_np, &anchor.np, &anchor.inv, &mut dh, cache);
        ops::add_assign(gpu, d_h_pre, &dh);
    }

    /// The slow module's input: `h⁽ᵖʳᵉ⁾ + W_inj(h⁽ʳ⁾ + εa)` — HRM's input injection
    /// (arXiv 2506.21734 §3.1), with the fast state entering through a learned
    /// projection rather than a concatenation.
    ///
    /// Returns the input and the noised fast state, which `W_inj`'s `dW = XᵀdY` needs
    /// and which re-running the noise to recover would be a second draw.
    ///
    /// A projection rather than `W[h_pre, h]` because it makes the identity the init:
    /// with `W_inj = 0` the slow module opens on `h_pre` itself, so the untrained model
    /// is the frozen-anchor one and nothing has to be unlearned. Concatenation would
    /// put a random map on the anchor path at step 0 and destroy the prelude's output.
    #[allow(clippy::too_many_arguments)]
    pub fn anchor_input(
        &mut self,
        gpu: &Gpu,
        h_pre: &GTensor<f32>,
        h: &GTensor<f32>,
        base_seed: u64,
        row0: usize,
        update: usize,
        noise: bool,
    ) -> (GTensor<f32>, GTensor<f32>) {
        let (n, d) = h.as_2d();
        assert_eq!(d, self.d, "GrtMod::anchor_input — width mismatch");
        let sigma = self.sigma(noise);
        let inj = self
            .w_inj
            .as_mut()
            .expect("GrtMod::anchor_input without an anchor module");
        let mut xs = GTensor::uninit(gpu, &[n, d]);
        ops::grt_noise_add(
            gpu,
            &mut xs,
            h,
            sigma,
            mix_seed(base_seed, update, KIND_ANCHOR),
            row0 * d,
        );
        let mut delta = GTensor::uninit(gpu, &[n, d]);
        inj.forward_shared(gpu, &xs, &mut delta);
        let mut ain = GTensor::uninit(gpu, &[n, d]);
        ops::add_into(gpu, h_pre, &delta, &mut ain);
        (ain, xs)
    }

    /// Backward of [`anchor_input`](Self::anchor_input). The residual sends `d_ain`
    /// straight to the previous anchor; the injection tap lands on the fast state the
    /// slow module read, which is the state at the end of the cycle.
    #[allow(clippy::too_many_arguments)]
    pub fn anchor_input_backward(
        &mut self,
        gpu: &Gpu,
        d_ain: &GTensor<f32>,
        xs: &GTensor<f32>,
        d_h_pre: &mut GTensor<f32>,
        dh: &mut GTensor<f32>,
        cache: &TrainingCache,
    ) {
        let (n, d) = xs.as_2d();
        ops::add_assign(gpu, d_h_pre, d_ain);
        let mut d_xs = GTensor::uninit(gpu, &[n, d]);
        self.w_inj
            .as_mut()
            .expect("GrtMod::anchor_input_backward without an anchor module")
            .backward_with_x(gpu, xs, d_ain, &mut d_xs, cache);
        // εa is additive, so the gradient passes straight through it.
        ops::add_assign(gpu, dh, &d_xs);
    }

    /// Every parameter with its gradient and AdamW moments, in a fixed order.
    ///
    /// The two projection matrices decay; every norm scale and the gate bias do not
    /// — the reference's optimizer splits on `dim >= 2`, which is the same rule.
    pub fn param_slots(&mut self) -> Vec<ParamSlot<'_>> {
        let mut v = Vec::new();
        v.extend(self.w_proj.param_slots());
        v.extend(self.ln_x.param_slots());
        v.extend(self.ln_p.param_slots());
        v.extend(self.ln_c.param_slots());
        v.extend(self.fc1.param_slots());
        v.extend(self.fc2.param_slots());
        if let Some(l) = self.w_inj.as_mut() {
            v.extend(l.param_slots());
        }
        debug_assert!(
            v.iter().all(|s| s.kind != ParamKind::Decay || s.param.rank == 2),
            "GrtMod: only the 2D projections may be weight-decayed"
        );
        v
    }

    /// Release anything the forward left resident. The GRT modulation saves no
    /// activations of its own; this only clears the `Linear`s' staging.
    pub fn drop_saved_act(&mut self, gpu: &Gpu) {
        for l in [&mut self.w_proj, &mut self.fc1, &mut self.fc2]
            .into_iter()
            .chain(self.w_inj.as_mut())
        {
            l.drop_saved_act(gpu);
        }
    }

    /// Run the three projections in fp32.
    ///
    /// For the gradient check below: with the bf16 GEMM path a finite difference and
    /// the analytic gradient disagree at ~4e-3 relative for reasons that have nothing
    /// to do with whether the backward is right, and a tolerance loose enough to
    /// absorb that would pass a genuinely wrong sign on a small term.
    pub fn set_fp32(&mut self) {
        for l in [&mut self.w_proj, &mut self.fc1, &mut self.fc2]
            .into_iter()
            .chain(self.w_inj.as_mut())
        {
            l.set_fp32();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::arena::TrainingCache;

    /// Updates fire strictly BETWEEN cycles, so a depth of `t` or less never moves
    /// the anchor and `t >= r` is the same model as `anchor = 0`.
    #[test]
    fn anchor_updates_count_the_gaps_between_cycles() {
        let c = |anchor, t, r| GrtCfg {
            anchor,
            t,
            r,
            ..cfg(0.0)
        };
        for (anchor, t, r, want) in [
            (1, 2, 1, 0),
            (1, 2, 2, 0),
            (1, 2, 3, 1),
            (1, 2, 4, 1),
            (1, 2, 5, 2),
            (1, 1, 4, 3),
            (1, 9, 4, 0),
            (0, 2, 4, 0),
        ] {
            let g = c(anchor, t, r);
            assert_eq!(
                g.max_anchor_updates(),
                want,
                "anchor {anchor}, t {t}, r {r}"
            );
        }
        // Executions count the slow module's runs, and unique blocks count its weights.
        let g = c(1, 2, 4);
        assert_eq!(g.executions(), g.pre + g.core * 4 + 1 + g.coda);
        assert_eq!(g.unique_blocks(), g.pre + g.core + 1 + g.coda);
    }

    fn cfg(noise: f32) -> GrtCfg {
        cfg_gate(noise, false)
    }

    fn cfg_gate(noise: f32, lstm_gate: bool) -> GrtCfg {
        GrtCfg {
            pre: 1,
            core: 1,
            r: 1,
            r_min: 1,
            coda: 1,
            anchor: 0,
            t: 2,
            noise_std: noise,
            tau: 1.0,
            gate_bias: 4.0,
            lstm_gate,
            f_bias: (3.0, 6.0),
            i_bias: 0.0,
        }
    }

    /// A module whose gradients a finite difference can actually resolve.
    ///
    /// Two departures from the training init, both so the difference quotient has
    /// something to measure: the gate head's bias is zeroed (at +4 every `g` is 0.98,
    /// `1 - g` is 0.02, and the whole `o` branch is nearly invisible) and the three
    /// projections are widened from std 0.02, which otherwise puts the gate path's
    /// gradients four orders below the blend's.
    fn probe_module(gpu: &Gpu, d: usize, lstm_gate: bool, seed: u64) -> GrtMod {
        let mut m = GrtMod::new(gpu, d, cfg_gate(0.0, lstm_gate));
        m.set_fp32();
        let mut slots = m.param_slots();
        let bias_len = slots[8].param.len();
        *slots[8].param = GTensor::zeros(gpu, &[bias_len]);
        for (k, slot) in [0usize, 5, 7].iter().enumerate() {
            let dims = slots[*slot].param.dims().to_vec();
            *slots[*slot].param =
                GTensor::from_host(gpu, &Tensor::random_seeded(&dims, 0.5, seed + k as u64));
        }
        drop(slots);
        m
    }

    /// Everything the modulation does around ONE application of the shared core,
    /// with the core stood in for by the identity so `W_proj` is on the gradient path:
    ///
    ///   o = h_tilde = W_proj[h + eps, h_pre],  h' = g*h + (1-g)*o,  L = sum(h' * w)
    ///
    /// Identity is the right stand-in precisely because it is transparent: any real
    /// core sits between `d_o` and `dh_tilde` and is unwound by its own tested
    /// backward, so what is left to check here is the modulation.
    fn forward_loss(
        gpu: &Gpu,
        m: &mut GrtMod,
        h: &GTensor<f32>,
        h_pre: &GTensor<f32>,
        w: &GTensor<f32>,
    ) -> f64 {
        let anchor = m.anchor(gpu, h_pre);
        let gate = m.gate(gpu, h, &anchor, 0, 0, 0, false);
        let (ht, _) = m.project(gpu, h, h_pre, 0, 0, 0, false);
        let h_next = m.blend(gpu, &gate.g, h, &ht);
        let (a, b) = (h_next.to_host(gpu).data, w.to_host(gpu).data);
        // f64: the gate path's gradients are three small matrices below the blend's,
        // so a central difference on the f32 sum of ~n*d terms cannot resolve them.
        a.iter().zip(b.iter()).map(|(x, y)| *x as f64 * *y as f64).sum()
    }

    /// The analytic backward of [`forward_loss`], leaving `dgamma`/`dW` accumulated in
    /// `m` and returning `(dh, dh_pre)`.
    fn backward(
        gpu: &Gpu,
        m: &mut GrtMod,
        h: &GTensor<f32>,
        h_pre: &GTensor<f32>,
        w: &GTensor<f32>,
        cache: &TrainingCache,
    ) -> (GTensor<f32>, GTensor<f32>) {
        let dims = h.dims().to_vec();
        let anchor = m.anchor(gpu, h_pre);
        let gate = m.gate(gpu, h, &anchor, 0, 0, 0, false);
        let (ht, pin) = m.project(gpu, h, h_pre, 0, 0, 0, false);

        let mut dh = GTensor::zeros(gpu, &dims);
        let mut d_h_pre = GTensor::zeros(gpu, &dims);
        let mut d_np = GTensor::zeros(gpu, &dims);
        // dL/dh' = w.
        let (d_o, d_g) = m.blend_backward(gpu, w, &gate.g, h, &ht, &mut dh);
        // The identity core: dL/dh_tilde IS d_o.
        m.project_backward(gpu, &d_o, &pin, &mut dh, &mut d_h_pre, cache);
        m.gate_backward(gpu, &gate, &d_g, h, &mut dh, &mut d_np, cache);
        m.anchor_backward(gpu, &anchor, &d_np, &mut d_h_pre, cache);
        (dh, d_h_pre)
    }

    /// The forward of `steps` chained recurrence steps with the core stood in for by
    /// the identity, and `h(0) = h_pre` exactly as the real backbone has it. Returns
    /// the loss and everything the backward needs.
    #[allow(clippy::type_complexity)]
    fn chain_forward(
        gpu: &Gpu,
        m: &mut GrtMod,
        h_pre: &GTensor<f32>,
        w: &GTensor<f32>,
        steps: usize,
    ) -> (f64, Vec<GTensor<f32>>, Vec<GTensor<f32>>, Vec<GTensor<f32>>) {
        let anchor = m.anchor(gpu, h_pre);
        let mut h = vec![h_pre.dup(gpu)];
        let (mut prop, mut pins) = (Vec::new(), Vec::new());
        for r in 0..steps {
            let gate = m.gate(gpu, &h[r], &anchor, 0, 0, r, false);
            let (ht, pin) = m.project(gpu, &h[r], h_pre, 0, 0, r, false);
            // Identity core: o(r) IS h~(r).
            let next = m.blend(gpu, &gate.g, &h[r], &ht);
            prop.push(ht);
            pins.push(pin);
            h.push(next);
        }
        let (a, b) = (h[steps].to_host(gpu).data, w.to_host(gpu).data);
        let loss = a.iter().zip(b.iter()).map(|(x, y)| *x as f64 * *y as f64).sum();
        (loss, h, prop, pins)
    }

    /// Every parameter of Eqs. (2) and (5), and both inputs, against a central finite
    /// difference.
    ///
    /// This is the whole reason the module can be trusted: the chain runs through
    /// three LayerNorms, a concatenation, a SiLU, a sigmoid and a convex blend, and
    /// several of those contribute terms that a plausible-looking implementation drops
    /// silently — the gate's `(h_prev - o)` factor, the anchor's two separate taps, the
    /// LayerNorm mean terms. Noise is off so the function is deterministic; the noise
    /// itself is additive and carries gradient 1 by construction.
    #[test]
    fn gradients_match_finite_differences() {
        for lstm_gate in [false, true] {
            single_step_fd(lstm_gate);
        }
    }

    fn single_step_fd(lstm_gate: bool) {
        let Some(gpu) = crate::gpu::test_gpu() else {
            return;
        };
        let tc = TrainingCache::new(&gpu, 1 << 20, 1 << 16, 1 << 20);
        let (n, d) = (5usize, 8usize);
        let mut m = probe_module(&gpu, d, lstm_gate, 21);
        let h = GTensor::from_host(&gpu, &Tensor::random_seeded(&[n, d], 1.0, 7));
        let h_pre = GTensor::from_host(&gpu, &Tensor::random_seeded(&[n, d], 1.0, 11));
        let w = GTensor::from_host(&gpu, &Tensor::random_seeded(&[n, d], 1.0, 13));

        let (dh, d_h_pre) = backward(&gpu, &mut m, &h, &h_pre, &w, &tc);
        let dh = dh.to_host(&gpu).data;
        let d_h_pre = d_h_pre.to_host(&gpu).data;

        let eps = 1e-3;
        // (slot index, name). The two frozen biases are not on the gradient path.
        for (slot, name) in [
            (0usize, "w_proj.w"),
            (2, "ln_x.gamma"),
            (3, "ln_p.gamma"),
            (4, "ln_c.gamma"),
            (5, "fc1.w"),
            (7, "fc2.w"),
            (8, "fc2.b"),
        ] {
            let (len, analytic) = {
                let slots = m.param_slots();
                let g = slots[slot].grad.to_host(&gpu).data;
                (slots[slot].param.len(), g)
            };
            // A spread of indices rather than all of them: one central difference is
            // two full forwards, and a wrong term shows up in any of them.
            for k in 0..5.min(len) {
                let i = k * len / 5.min(len);
                let mut probe = |delta: f32| {
                    let mut slots = m.param_slots();
                    let t = &mut *slots[slot].param;
                    let mut host = t.to_host(&gpu);
                    host.data[i] += delta;
                    *t = GTensor::from_host(&gpu, &host);
                    drop(slots);
                    let l = forward_loss(&gpu, &mut m, &h, &h_pre, &w);
                    let mut slots = m.param_slots();
                    let t = &mut *slots[slot].param;
                    let mut host = t.to_host(&gpu);
                    host.data[i] -= delta;
                    *t = GTensor::from_host(&gpu, &host);
                    l
                };
                let fd = ((probe(eps) - probe(-eps)) / (2.0 * eps as f64)) as f32;
                // Scaled by the LARGEST gradient in the tensor, not by this element's:
                // a component four orders below its neighbours is below what a
                // difference quotient in f32 activations can resolve at all, and a
                // per-element tolerance would be asking the test to measure noise.
                let scale = analytic.iter().fold(0.0f32, |a, v| a.max(v.abs()));
                let tol = 3e-2 * scale.max(1e-6);
                assert!(
                    (analytic[i] - fd).abs() <= tol,
                    "{name}[{i}]: analytic {} vs finite difference {fd}",
                    analytic[i]
                );
            }
        }

        // The two inputs. `h` is the interesting one: three separate paths reach it
        // (the gate's copy branch, `W_proj`, `LN_x`) and they are summed, not chosen.
        for (name, grad, base) in [("dh", &dh, &h), ("dh_pre", &d_h_pre, &h_pre)] {
            for i in (0..n * d).step_by(7) {
                let mut probe = |delta: f32| {
                    let mut host = base.to_host(&gpu);
                    host.data[i] += delta;
                    let moved = GTensor::from_host(&gpu, &host);
                    if name == "dh" {
                        forward_loss(&gpu, &mut m, &moved, &h_pre, &w)
                    } else {
                        forward_loss(&gpu, &mut m, &h, &moved, &w)
                    }
                };
                let fd = ((probe(eps) - probe(-eps)) / (2.0 * eps as f64)) as f32;
                let scale = grad.iter().fold(0.0f32, |a, v| a.max(v.abs()));
                let tol = 3e-2 * scale.max(1e-6);
                assert!(
                    (grad[i] - fd).abs() <= tol,
                    "{name}[{i}]: analytic {} vs finite difference {fd}",
                    grad[i]
                );
            }
        }
    }

    /// The same check across THREE chained recurrence steps.
    ///
    /// This is the part `gradients_match_finite_differences` cannot reach and the
    /// chunk-parity test cannot either: the gradient that flows from step `r` back
    /// into step `r-1`. A wrong copy-branch term, a `dh_prev` that overwrote instead
    /// of accumulating, or an anchor tap counted once instead of `R` times would all
    /// leave a single step exactly right and starve the deeper ones — which is
    /// indistinguishable, from a loss curve, from a model that has simply chosen not
    /// to use its depth.
    ///
    /// The structure mirrors `hierarchical::grt_chunk_backward` exactly, with an empty
    /// prelude and coda and an identity core, so what passes here is the real
    /// unwinding order and not a test-only rewrite of it.
    #[test]
    fn chained_step_gradients_match_finite_differences() {
        for lstm_gate in [false, true] {
            chained_fd(lstm_gate);
        }
    }

    fn chained_fd(lstm_gate: bool) {
        let Some(gpu) = crate::gpu::test_gpu() else {
            return;
        };
        let tc = TrainingCache::new(&gpu, 1 << 20, 1 << 16, 1 << 20);
        let (n, d, steps) = (5usize, 8usize, 3usize);
        let mut m = probe_module(&gpu, d, lstm_gate, 31);
        let h_pre = GTensor::from_host(&gpu, &Tensor::random_seeded(&[n, d], 1.0, 41));
        let w = GTensor::from_host(&gpu, &Tensor::random_seeded(&[n, d], 1.0, 43));

        // Backward, in `grt_chunk_backward`'s order.
        let (_, h, prop, pins) = chain_forward(&gpu, &mut m, &h_pre, &w, steps);
        let anchor = m.anchor(&gpu, &h_pre);
        let mut d_h_pre = GTensor::zeros(&gpu, &[n, d]);
        let mut d_np = GTensor::zeros(&gpu, &[n, d]);
        let mut dh = w.dup(&gpu);
        for r in (0..steps).rev() {
            let gate = m.gate(&gpu, &h[r], &anchor, 0, 0, r, false);
            let mut dh_prev = GTensor::zeros(&gpu, &[n, d]);
            let (d_o, d_g) = m.blend_backward(&gpu, &dh, &gate.g, &h[r], &prop[r], &mut dh_prev);
            // Identity core: dL/dh~(r) IS d_o.
            m.project_backward(&gpu, &d_o, &pins[r], &mut dh_prev, &mut d_h_pre, &tc);
            m.gate_backward(&gpu, &gate, &d_g, &h[r], &mut dh_prev, &mut d_np, &tc);
            dh = dh_prev;
        }
        // h(0) IS the anchor, so the chain's remaining gradient lands on it too.
        ops::add_assign(&gpu, &mut d_h_pre, &dh);
        m.anchor_backward(&gpu, &anchor, &d_np, &mut d_h_pre, &tc);
        let d_h_pre = d_h_pre.to_host(&gpu).data;

        let eps = 1e-3;
        for (slot, name) in [
            (0usize, "w_proj.w"),
            (2, "ln_x.gamma"),
            (3, "ln_p.gamma"),
            (4, "ln_c.gamma"),
            (5, "fc1.w"),
            (7, "fc2.w"),
            (8, "fc2.b"),
        ] {
            let (len, analytic) = {
                let slots = m.param_slots();
                let g = slots[slot].grad.to_host(&gpu).data;
                (slots[slot].param.len(), g)
            };
            let scale = analytic.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            for k in 0..5.min(len) {
                let i = k * len / 5.min(len);
                let mut probe = |delta: f32| {
                    let mut slots = m.param_slots();
                    let t = &mut *slots[slot].param;
                    let mut host = t.to_host(&gpu);
                    host.data[i] += delta;
                    *t = GTensor::from_host(&gpu, &host);
                    drop(slots);
                    let l = chain_forward(&gpu, &mut m, &h_pre, &w, steps).0;
                    let mut slots = m.param_slots();
                    let t = &mut *slots[slot].param;
                    let mut host = t.to_host(&gpu);
                    host.data[i] -= delta;
                    *t = GTensor::from_host(&gpu, &host);
                    l
                };
                let fd = ((probe(eps) - probe(-eps)) / (2.0 * eps as f64)) as f32;
                assert!(
                    (analytic[i] - fd).abs() <= 3e-2 * scale.max(1e-6),
                    "{name}[{i}] over {steps} steps (lstm_gate {lstm_gate}): analytic {} vs finite difference {fd}",
                    analytic[i]
                );
            }
        }

        // `h_pre` is the strongest single check: it is read by the anchor norm, by
        // `W_proj` at every step, and it IS `h(0)`, so its gradient is the sum of
        // `3 * 2 + 1` separate paths through the chain.
        let scale = d_h_pre.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        for i in (0..n * d).step_by(3) {
            let mut probe = |delta: f32| {
                let mut host = h_pre.to_host(&gpu);
                host.data[i] += delta;
                let moved = GTensor::from_host(&gpu, &host);
                chain_forward(&gpu, &mut m, &moved, &w, steps).0
            };
            let fd = ((probe(eps) - probe(-eps)) / (2.0 * eps as f64)) as f32;
            assert!(
                (d_h_pre[i] - fd).abs() <= 3e-2 * scale.max(1e-6),
                "d_h_pre[{i}] over {steps} steps (lstm_gate {lstm_gate}): analytic {} vs finite difference {fd}",
                d_h_pre[i]
            );
        }
    }

    /// The counter-based noise has to be a pure function of its seed, or the
    /// backward's recomputation reads different `εx` than the forward wrote and every
    /// `dW_proj` is quietly wrong.
    #[test]
    fn noise_is_reproducible_and_has_the_right_scale() {
        let Some(gpu) = crate::gpu::test_gpu() else {
            return;
        };
        let (n, d) = (64usize, 64usize);
        let m = GrtMod::new(&gpu, d, cfg(0.1));
        let h = GTensor::zeros(&gpu, &[n, d]);
        let h_pre = GTensor::zeros(&gpu, &[n, d]);
        let a = m.project_input(&gpu, &h, &h_pre, 99, 1, 2, true);
        let b = m.project_input(&gpu, &h, &h_pre, 99, 1, 2, true);
        let c = m.project_input(&gpu, &h, &h_pre, 99, 1, 3, true);
        let (a, b, c) = (
            a.to_host(&gpu).data,
            b.to_host(&gpu).data,
            c.to_host(&gpu).data,
        );
        assert_eq!(a, b, "same seed must give the same draw");
        assert_ne!(a, c, "a different recurrence step must give a different draw");
        // `h` is zero, so the first half of the concatenation IS eps_x.
        let eps: Vec<f32> = (0..n)
            .flat_map(|r| a[r * 2 * d..r * 2 * d + d].to_vec())
            .collect();
        let mean = eps.iter().sum::<f32>() / eps.len() as f32;
        let var = eps.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / eps.len() as f32;
        assert!(mean.abs() < 0.02, "eps_x mean {mean} is not ~0");
        assert!(
            (var.sqrt() - 0.1).abs() < 0.01,
            "eps_x std {} is not ~0.1",
            var.sqrt()
        );
        // Noise off must be exactly off, not merely small.
        let off = m.project_input(&gpu, &h, &h_pre, 99, 1, 2, false);
        assert!(off.to_host(&gpu).data.iter().all(|&v| v == 0.0));
    }
}
