// INTEGRATIONS-GUIDE: Adam Optimizer in dein Netzwerk einbauen
// 
// Dieser Guide zeigt Schritt für Schritt, wie du SGD durch Adam ersetzt.

// ─── SCHRITT 1: Konfiguration (config.rs) ────────────────────────────────────

// In deiner config.rs hinzufügen:

pub enum OptimizerType {
    SGD,
    Adam,
}

pub const OPTIMIZER: OptimizerType = OptimizerType::Adam;

// Optional für Adam-Hyperparameter:
pub const ADAM_BETA1: f32 = 0.9;
pub const ADAM_BETA2: f32 = 0.999;
pub const ADAM_EPS: f32 = 1e-8;


// ─── SCHRITT 2: Momente in den Gradient-Strukturen speichern ─────────────────

// In dense.rs (und ähnlich in anderen Layern):

use crate::adam_optimizer::{MatrixMoments, VectorMoments};

pub struct DenseGrads {
    pub weights: Matrix,
    pub biases: Box<[f32]>,
    
    // NEU: Momente für Adam
    pub weights_moments: MatrixMoments,
    pub biases_moments: VectorMoments,
}

impl DenseGrads {
    pub fn zeros(input_size: usize, output_size: usize) -> Self {
        Self {
            weights: Matrix::zeros(input_size, output_size),
            biases: vec![0.0; output_size].into(),
            // NEU:
            weights_moments: MatrixMoments::zeros(input_size, output_size),
            biases_moments: VectorMoments::zeros(output_size),
        }
    }
    
    pub fn clear(&mut self) {
        for row in 0..self.weights.rows() {
            for col in 0..self.weights.cols() {
                self.weights[row][col] = 0.0;
            }
        }
        self.biases.iter_mut().for_each(|b| *b = 0.0);
    }
}


// ─── SCHRITT 3: apply_grads in Dense Layer anpassen ───────────────────────────

// In dense.rs, ersetze die alte apply_grads:

impl<A: Activate> NnLayer for DenseLayer<A> {
    fn apply_grads(&mut self, lr: f32) {
        match crate::config::OPTIMIZER {
            crate::config::OptimizerType::SGD => {
                // Alte SGD-Logik
                for i in 0..self.weights.rows() {
                    for j in 0..self.weights.cols() {
                        self.weights[i][j] -= lr * self.grads.weights[i][j];
                    }
                }
                for i in 0..self.biases.len() {
                    self.biases[i] -= lr * self.grads.biases[i];
                }
            }
            crate::config::OptimizerType::Adam => {
                use crate::adam_optimizer::*;
                
                let config = AdamConfig::new(
                    crate::config::ADAM_BETA1,
                    crate::config::ADAM_BETA2,
                    crate::config::ADAM_EPS,
                );
                
                // Adam Update — beachte dass wir die Step-Zahl brauchen!
                // Das ist problematisch, wir müssen die in Sequential tracking.
                // Siehe SCHRITT 4.
                
                apply_adam_update_matrix(
                    &mut self.weights,
                    &self.grads.weights,
                    &mut self.grads.weights_moments,
                    lr,
                    self.get_adam_step(),  // ← Siehe SCHRITT 4
                    &config,
                );
                
                apply_adam_update_vector(
                    &mut self.biases,
                    &self.grads.biases,
                    &mut self.grads.biases_moments,
                    lr,
                    self.get_adam_step(),
                    &config,
                );
            }
        }
        self.clear_grads();
    }
}


// ─── SCHRITT 4: Step-Zähler für Adam Bias-Korrektur ──────────────────────────

// Das ist der trickvoll Teil! Adam braucht einen globalen Schrittzähler (t)
// für die Bias-Korrektur. Mögliche Lösungen:

// OPTION A: Globale static Variable (einfach, aber nicht thread-safe)
use std::sync::atomic::{AtomicU32, Ordering};
static ADAM_STEP: AtomicU32 = AtomicU32::new(0);

pub fn increment_adam_step() {
    ADAM_STEP.fetch_add(1, Ordering::Relaxed);
}

pub fn get_adam_step() -> u32 {
    ADAM_STEP.load(Ordering::Relaxed)
}

pub fn reset_adam_step() {
    ADAM_STEP.store(0, Ordering::Relaxed);
}

// OPTION B: State in Sequential speichern (besser!)
// In sequential.rs:
pub struct Sequential {
    // ... existing fields ...
    adam_step: u32,
}

impl Sequential {
    pub fn new() -> Self {
        Self {
            // ...
            adam_step: 0,
        }
    }
    
    fn sgd_step(&mut self, lr: f32) {
        // Old SGD:
        for layer in &mut self.layers {
            layer.apply_grads(lr);
        }
        self.clear_grads();
    }
    
    // NEU:
    fn adam_step(&mut self, lr: f32) {
        // Setze den aktuellen Step in allen Layern
        for layer in &mut self.layers {
            layer.set_adam_step(self.adam_step);
            layer.apply_grads(lr);
        }
        self.adam_step += 1;
        self.clear_grads();
    }
}

// Dann in train():
pub fn train<B: Batches>(&mut self, batches: B, lr: f32, iteration: &mut usize, j: &mut usize, batch_size: usize) {
    // ... forward/backward ...
    
    *iteration += 1;
    if *iteration % batch_size == 0 {
        *j += 1;
        match crate::config::OPTIMIZER {
            crate::config::OptimizerType::SGD => self.sgd_step(lr),
            crate::config::OptimizerType::Adam => self.adam_step(lr),
        }
    }
}


// ─── SCHRITT 5: NnLayer Trait erweitern ──────────────────────────────────────

// In nn_layer.rs, ergänze neue Methoden:

pub trait NnLayer {
    // ... existing methods ...
    
    fn set_adam_step(&mut self, _t: u32) {}
}

// Dann implementiere in jedem Layer:
impl<A: Activate> NnLayer for DenseLayer<A> {
    fn set_adam_step(&mut self, t: u32) {
        self.adam_step = t;
    }
    // ...
}


// ─── SCHRITT 6: Momente speichern / clearen ──────────────────────────────────

// In dense.rs, add in apply_grads:

pub fn clear_grads(&mut self) {
    self.grads.clear();
    // Momente NICHT clearen — sie sind persistent!
    // self.grads.weights_moments.zero();  ← NICHT HIER
}

// Momente sollten NUR am Start eines neuen Training gelöscht werden:
impl Sequential {
    pub fn reset_optimizer(&mut self) {
        if let OptimizerType::Adam = OPTIMIZER {
            for layer in &mut self.layers {
                layer.reset_adam_moments();
            }
        }
    }
}


// ─── SCHRITT 7: Ähnlich für andere Layer (LSTM, etc.) ───────────────────────

// Die selbe Logik gilt für:
//   - lstm.rs: LSTM Gewichte haben 4 Gates + Peephole
//   - linear.rs: Nur Gewichte, keine Bias
//   - silu_dense.rs: Dense mit SiLU Aktivation
//   - Alle anderen Layer...

// Beispiel für LSTM:
pub struct LSTMGrads {
    pub weights: Matrix,
    pub biases: Box<[f32]>,
    pub peephole_in: Vec<f32>,
    pub peephole_forget: Vec<f32>,
    pub peephole_out: Vec<f32>,
    
    // NEU: Momente
    pub weights_moments: MatrixMoments,
    pub biases_moments: VectorMoments,
    pub peephole_in_moments: VectorMoments,
    pub peephole_forget_moments: VectorMoments,
    pub peephole_out_moments: VectorMoments,
}


// ─── VERWENDUNG ──────────────────────────────────────────────────────────────

// Nach diesen Schritten kannst du einfach in config.rs umschalten:

pub const OPTIMIZER: OptimizerType = OptimizerType::Adam;  // Statt SGD

// Und das Training läuft mit Adam statt SGD!
// Kein anderer Code muss geändert werden.

// Wenn du die Learning Rate anpassen willst (Adam ist typisch 1e-3):
pub const LR: f32 = 1e-3;  // Statt 3e-4 für SGD


// ─── TESTING ──────────────────────────────────────────────────────────────────

// Kleine Änderung in training.rs zum Vergleich:

println!(
    "Training mit: {} Optimizer",
    match crate::config::OPTIMIZER {
        crate::config::OptimizerType::SGD => "SGD",
        crate::config::OptimizerType::Adam => "Adam",
    }
);

// Dann trainieren und die Konvergenz vergleichen!
