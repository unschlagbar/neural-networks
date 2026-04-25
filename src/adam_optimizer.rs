// ── adam_optimizer.rs ─────────────────────────────────────────────────────────
//
// Adam Optimizer Implementierung für Neural Networks.
//
// Verwendung:
//   1. Jeder Layer speichert zusätzlich zu `grads` auch Momente in `adam_moments`
//   2. In Sequential::sgd_step() → Sequential::adam_step() aufrufen
//   3. Momente werden im State des Layers selbst gehalten
//
// Mathematik:
//   m_t = β₁ * m_{t-1} + (1 - β₁) * g_t           (erstes Moment — Mittelwert)
//   v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²          (zweites Moment — Varianz)
//   m̂_t = m_t / (1 - β₁^t)                        (Bias-Korrektur für m)
//   v̂_t = v_t / (1 - β₂^t)                        (Bias-Korrektur für v)
//   θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)         (Update-Schritt)

use iron_oxide::collections::Matrix;

/// Adam Konfiguration.
/// Standardwerte nach Kingma & Ba, 2014:
///   β₁ = 0.9    → Exponential decay für Gradient-Mittelwert
///   β₂ = 0.999  → Exponential decay für Gradient²-Mittelwert (Varianz)
///   ε  = 1e-8   → Kleine Konstante zur Numerik-Stabilität
#[derive(Clone, Debug)]
pub struct AdamConfig {
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl AdamConfig {
    /// Standard-Hyperparameter
    pub fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    /// Custom Hyperparameter
    pub fn new(beta1: f32, beta2: f32, eps: f32) -> Self {
        Self { beta1, beta2, eps }
    }
}

/// Momente für einen einzelnen Skalar-Parameter.
///
/// Speichert m (erstes Moment) und v (zweites Moment) für Adam.
#[derive(Clone, Copy, Debug)]
pub struct ScalarMoment {
    pub m: f32, // erstes Moment
    pub v: f32, // zweites Moment (v = E[g²])
}

impl ScalarMoment {
    pub fn new() -> Self {
        Self { m: 0.0, v: 0.0 }
    }

    /// Akkumuliere neuen Gradienten in die Momente.
    ///
    /// Formel:
    ///   m ← β₁ * m + (1 - β₁) * grad
    ///   v ← β₂ * v + (1 - β₂) * grad²
    #[inline]
    pub fn accumulate(&mut self, grad: f32, beta1: f32, beta2: f32) {
        self.m = beta1 * self.m + (1.0 - beta1) * grad;
        self.v = beta2 * self.v + (1.0 - beta2) * grad * grad;
    }

    /// Berechne den Parameter-Update mit Bias-Korrektur.
    ///
    /// Rückgabe: Δθ = lr * m̂ / (√v̂ + ε)
    ///
    /// `t` ist der Schrittzähler (1-basiert).
    #[inline]
    pub fn update_delta(&self, lr: f32, t: u32, beta1: f32, beta2: f32, eps: f32) -> f32 {
        let t_f32 = t as f32;
        let m_hat = self.m / (1.0 - beta1.powf(t_f32));
        let v_hat = self.v / (1.0 - beta2.powf(t_f32));
        lr * m_hat / (v_hat.sqrt() + eps)
    }

    pub fn zero(&mut self) {
        self.m = 0.0;
        self.v = 0.0;
    }
}

/// Momente für eine Matrix (Gewichte in Dense, LSTM, etc.)
pub struct MatrixMoments {
    pub m: Matrix,
    pub v: Matrix,
}

impl MatrixMoments {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            m: Matrix::zeros(rows, cols),
            v: Matrix::zeros(rows, cols),
        }
    }

    /// Akkumuliere eine Gradient-Matrix in die Momente.
    pub fn accumulate(&mut self, grad_matrix: &Matrix, beta1: f32, beta2: f32) {
        for i in 0..self.m.rows() {
            for j in 0..self.m.cols() {
                let g = grad_matrix[i][j];
                self.m[i][j] = beta1 * self.m[i][j] + (1.0 - beta1) * g;
                self.v[i][j] = beta2 * self.v[i][j] + (1.0 - beta2) * g * g;
            }
        }
    }

    pub fn zero(&mut self) {
        for i in 0..self.m.rows() {
            for j in 0..self.m.cols() {
                self.m[i][j] = 0.0;
                self.v[i][j] = 0.0;
            }
        }
    }
}

/// Momente für einen Vektor (Bias-Terme, etc.)
pub struct VectorMoments {
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

impl VectorMoments {
    pub fn zeros(len: usize) -> Self {
        Self {
            m: vec![0.0; len],
            v: vec![0.0; len],
        }
    }

    /// Akkumuliere einen Gradient-Vektor.
    pub fn accumulate(&mut self, grad_vec: &[f32], beta1: f32, beta2: f32) {
        for i in 0..self.m.len() {
            let g = grad_vec[i];
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g;
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * g * g;
        }
    }

    pub fn zero(&mut self) {
        self.m.fill(0.0);
        self.v.fill(0.0);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// PRAKTISCHE HELPER-FUNKTIONEN FÜR DIE INTEGRATION
// ──────────────────────────────────────────────────────────────────────────────

/// Führe Adam-Update für eine einzelne Gewichtmatrix durch.
///
/// Diese Funktion wird in Layer::apply_grads_adam aufgerufen.
#[inline]
pub fn apply_adam_update_matrix(
    weights: &mut Matrix,
    grads: &Matrix,
    moments: &mut MatrixMoments,
    lr: f32,
    t: u32,
    config: &AdamConfig,
) {
    moments.accumulate(grads, config.beta1, config.beta2);

    let t_f32 = t as f32;
    let bias_correction1 = 1.0 - config.beta1.powf(t_f32);
    let bias_correction2 = 1.0 - config.beta2.powf(t_f32);

    for i in 0..weights.rows() {
        for j in 0..weights.cols() {
            let m_hat = moments.m[i][j] / bias_correction1;
            let v_hat = moments.v[i][j] / bias_correction2;
            let delta = lr * m_hat / (v_hat.sqrt() + config.eps);
            weights[i][j] -= delta;
        }
    }
}

/// Führe Adam-Update für einen Bias-Vektor durch.
#[inline]
pub fn apply_adam_update_vector(
    biases: &mut [f32],
    grads: &[f32],
    moments: &mut VectorMoments,
    lr: f32,
    t: u32,
    config: &AdamConfig,
) {
    moments.accumulate(grads, config.beta1, config.beta2);

    let t_f32 = t as f32;
    let bias_correction1 = 1.0 - config.beta1.powf(t_f32);
    let bias_correction2 = 1.0 - config.beta2.powf(t_f32);

    for i in 0..biases.len() {
        let m_hat = moments.m[i] / bias_correction1;
        let v_hat = moments.v[i] / bias_correction2;
        let delta = lr * m_hat / (v_hat.sqrt() + config.eps);
        biases[i] -= delta;
    }
}

/// Führe Adam-Update für einen einzelnen Skalar-Parameter durch.
#[inline]
pub fn apply_adam_update_scalar(
    param: &mut f32,
    grad: f32,
    moment: &mut ScalarMoment,
    lr: f32,
    t: u32,
    config: &AdamConfig,
) {
    moment.accumulate(grad, config.beta1, config.beta2);
    let delta = moment.update_delta(lr, t, config.beta1, config.beta2, config.eps);
    *param -= delta;
}
