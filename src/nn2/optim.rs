//! AdamW for the `nn2` layers.
//!
//! Each layer owns its per-parameter optimizer state (`m`, `v`) and steps
//! itself; the model passes a shared `AdamCfg` carrying the hyperparameters and
//! the global step count `t`. Decoupled weight decay (AdamW): the decay term
//! acts on the raw parameter, not through the gradient, and is applied only to
//! `decay = true` params — the project convention is interior projection
//! matrices decay, while embeddings, logit heads, biases and norm scales do not.

/// Shared optimizer configuration. `t` is the global step count (>= 1 at step
/// time); the model increments it once per optimizer step.
#[derive(Clone, Copy)]
pub struct AdamCfg {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub t: u64,
}

impl AdamCfg {
    pub fn new(lr: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            t: 0,
        }
    }
}

/// Per-parameter Adam moments. Lazily sized on the first step.
#[derive(Default)]
pub struct AdamState {
    m: Vec<f32>,
    v: Vec<f32>,
}

impl AdamState {
    pub fn new() -> Self {
        Self::default()
    }

    /// One AdamW update of `param` from `grad`. `decay` toggles the decoupled
    /// weight-decay term for this parameter.
    pub fn step(&mut self, param: &mut [f32], grad: &[f32], cfg: &AdamCfg, decay: bool) {
        let n = param.len();
        debug_assert_eq!(grad.len(), n);
        if self.m.len() != n {
            self.m = vec![0.0; n];
            self.v = vec![0.0; n];
        }
        let (b1, b2) = (cfg.beta1, cfg.beta2);
        let bc1 = 1.0 - b1.powi(cfg.t as i32);
        let bc2 = 1.0 - b2.powi(cfg.t as i32);
        let wd = if decay { cfg.weight_decay } else { 0.0 };
        for k in 0..n {
            let g = grad[k];
            self.m[k] = b1 * self.m[k] + (1.0 - b1) * g;
            self.v[k] = b2 * self.v[k] + (1.0 - b2) * g * g;
            let mh = self.m[k] / bc1;
            let vh = self.v[k] / bc2;
            let mut p = param[k];
            p -= cfg.lr * wd * p; // decoupled decay on the raw parameter
            p -= cfg.lr * mh / (vh.sqrt() + cfg.eps);
            param[k] = p;
        }
    }
}

/// Grow a lazily-managed pool of `AdamState` to at least `n` entries.
pub fn ensure_states(states: &mut Vec<AdamState>, n: usize) {
    while states.len() < n {
        states.push(AdamState::new());
    }
}
