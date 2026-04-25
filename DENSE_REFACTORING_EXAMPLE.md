// ─ REFACTORING EXAMPLE: Dense Layer mit Adam ──────────────────────────────────
//
// Dies zeigt genau, wie du dense.rs anpassen musst.
// Kopiere diese Code-Blöcke in deine dense.rs!

// ──────────────────────────────────────────────────────────────────────────────
// 1. UPDATE: DenseGrads — Momente hinzufügen
// ──────────────────────────────────────────────────────────────────────────────

// ← ALTER CODE:
//
// pub struct DenseGrads {
//     pub weights: Matrix,
//     pub biases: Vec<f32>,
// }
//
// impl DenseGrads {
//     pub fn zeros(input_size: usize, output_size: usize) -> Self {
//         Self {
//             weights: Matrix::zeros(input_size, output_size),
//             biases: vec![0.0; output_size],
//         }
//     }
// }

// ← NEUER CODE:

use crate::adam_optimizer::{MatrixMoments, VectorMoments};

pub struct DenseGrads {
    pub weights: Matrix,
    pub biases: Vec<f32>,
    
    // NEU: Momente für Adam Optimizer
    pub weights_moments: MatrixMoments,
    pub biases_moments: VectorMoments,
}

impl DenseGrads {
    pub fn zeros(input_size: usize, output_size: usize) -> Self {
        Self {
            weights: Matrix::zeros(input_size, output_size),
            biases: vec![0.0; output_size],
            // NEU:
            weights_moments: MatrixMoments::zeros(input_size, output_size),
            biases_moments: VectorMoments::zeros(output_size),
        }
    }
}


// ──────────────────────────────────────────────────────────────────────────────
// 2. UPDATE: DenseLayer — Step-Zähler speichern
// ──────────────────────────────────────────────────────────────────────────────

// ← ALTER CODE:
//
// pub struct DenseLayer<A: Activate> {
//     pub weights: Matrix,
//     pub biases: Box<[f32]>,
//     pub activation: A,
//     pub grads: DenseGrads,
// }

// ← NEUER CODE:

pub struct DenseLayer<A: Activate> {
    pub weights: Matrix,
    pub biases: Box<[f32]>,
    pub activation: A,
    pub grads: DenseGrads,
    
    // NEU: Schrittzähler für Adam Bias-Korrektur
    adam_step: u32,
}

// Dann in impl<A: Activate> DenseLayer<A>:
impl<A: Activate> DenseLayer<A> {
    pub fn new(input_size: usize, hidden_size: usize, activation: A) -> Self {
        let scale = (6.0 / (input_size as f32 + hidden_size as f32)).sqrt();
        let weights = Matrix::random(input_size, hidden_size, scale);
        let scale = (1.0 / input_size as f32).sqrt();
        let biases = (0..hidden_size)
            .map(|_| random_range(-scale..scale))
            .collect();
        Self {
            weights,
            biases,
            activation,
            grads: DenseGrads::zeros(input_size, hidden_size),
            adam_step: 0,  // NEU
        }
    }
    
    pub fn from_loaded(
        input_size: usize,
        output_size: usize,
        activation: A,
        weights: Matrix,
        biases: Box<[f32]>,
    ) -> Self {
        debug_assert_eq!(weights.rows(), input_size);
        debug_assert_eq!(weights.cols(), output_size);
        debug_assert_eq!(biases.len(), output_size);

        Self {
            weights,
            biases,
            activation,
            grads: DenseGrads::zeros(input_size, output_size),
            adam_step: 0,  // NEU
        }
    }
}


// ──────────────────────────────────────────────────────────────────────────────
// 3. UPDATE: apply_grads — SGD ↔ Adam Switch
// ──────────────────────────────────────────────────────────────────────────────

// ← ALTER CODE (in impl NnLayer for DenseLayer):
//
// fn apply_grads(&mut self, lr: f32) {
//     sub_in_place(&mut self.weights, &self.grads.weights, lr);
//     sub_vec_in_place(&mut self.biases, &self.grads.biases, lr);
// }

// ← NEUER CODE:

// Zuerst add in impl<A: Activate> DenseLayer<A>:

impl<A: Activate> DenseLayer<A> {
    fn apply_grads_sgd(&mut self, lr: f32) {
        // Alt: theta -= lr * grad
        sub_in_place(&mut self.weights, &self.grads.weights, lr);
        sub_vec_in_place(&mut self.biases, &self.grads.biases, lr);
    }
    
    fn apply_grads_adam(&mut self, lr: f32) {
        use crate::adam_optimizer::*;
        
        let config = AdamConfig::new(
            crate::config::ADAM_BETA1,
            crate::config::ADAM_BETA2,
            crate::config::ADAM_EPS,
        );
        
        // Accumulate gradients into moments
        self.grads.weights_moments.accumulate(&self.grads.weights, config.beta1, config.beta2);
        self.grads.biases_moments.accumulate(&self.grads.biases, config.beta1, config.beta2);
        
        // Compute update mit Bias-Korrektur
        let t = self.adam_step;
        let t_f32 = t as f32;
        let bias_correction1 = 1.0 - config.beta1.powf(t_f32);
        let bias_correction2 = 1.0 - config.beta2.powf(t_f32);
        
        // Update weights
        for i in 0..self.weights.rows() {
            for j in 0..self.weights.cols() {
                let m_hat = self.grads.weights_moments.m[i][j] / bias_correction1;
                let v_hat = self.grads.weights_moments.v[i][j] / bias_correction2;
                let delta = lr * m_hat / (v_hat.sqrt() + config.eps);
                self.weights[i][j] -= delta;
            }
        }
        
        // Update biases
        for i in 0..self.biases.len() {
            let m_hat = self.grads.biases_moments.m[i] / bias_correction1;
            let v_hat = self.grads.biases_moments.v[i] / bias_correction2;
            let delta = lr * m_hat / (v_hat.sqrt() + config.eps);
            self.biases[i] -= delta;
        }
    }
}

// Dann in impl NnLayer for DenseLayer:

impl<A: Activate> NnLayer for DenseLayer<A> {
    // ... alle anderen Methods wie gehabt ...
    
    fn apply_grads(&mut self, lr: f32) {
        match crate::config::OPTIMIZER {
            crate::config::OptimizerType::SGD => self.apply_grads_sgd(lr),
            crate::config::OptimizerType::Adam => self.apply_grads_adam(lr),
        }
    }
    
    fn set_adam_step(&mut self, t: u32) {
        self.adam_step = t;
    }
    
    // ... rest of methods ...
}


// ──────────────────────────────────────────────────────────────────────────────
// 4. UPDATE: nn_layer.rs — Trait mit set_adam_step erweitern
// ──────────────────────────────────────────────────────────────────────────────

// In pub trait NnLayer (in nn_layer.rs):
// Füge diese Methode hinzu:

pub trait NnLayer {
    // ... existing methods ...
    
    /// Setze den Adam-Step-Zähler für Bias-Korrektur.
    fn set_adam_step(&mut self, _t: u32) {}
    
    /// Resette Adam Moments (am Anfang des Trainings).
    fn reset_adam_moments(&mut self) {}
}


// ──────────────────────────────────────────────────────────────────────────────
// 5. UPDATE: config.rs — Optimizer wählen
// ──────────────────────────────────────────────────────────────────────────────

// Füge in config.rs diese Konstanten hinzu:

pub enum OptimizerType {
    SGD,
    Adam,
}

pub const OPTIMIZER: OptimizerType = OptimizerType::Adam;  // ← Hier umschalten!

// Adam-spezifische Hyperparameter:
pub const ADAM_BETA1: f32 = 0.9;
pub const ADAM_BETA2: f32 = 0.999;
pub const ADAM_EPS: f32 = 1e-8;

// Learning Rate (für Adam typisch höher):
// pub const LR: f32 = 1e-3;  // Statt 3e-4


// ──────────────────────────────────────────────────────────────────────────────
// 6. UPDATE: sequential.rs — Adam-Step-Tracking
// ──────────────────────────────────────────────────────────────────────────────

// In struct Sequential:

pub struct Sequential {
    pub layers: Vec<Box<dyn NnLayer>>,
    pub cache: Vec<Vec<Box<dyn DynCache>>>,
    adam_step: u32,  // ← NEU: Schrittzähler
}

impl Sequential {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            cache: Vec::new(),
            adam_step: 0,  // ← NEU
        }
    }
    
    // ... existing methods ...
    
    fn sgd_step(&mut self, lr: f32) {
        for layer in &mut self.layers {
            layer.apply_grads(lr);
        }
        self.clear_grads();
    }
    
    // ← NEU: Separate Adam-Step Methode
    fn adam_step(&mut self, lr: f32) {
        // Setze den aktuellen Schritt in allen Layern
        for layer in &mut self.layers {
            layer.set_adam_step(self.adam_step);
        }
        // Apply gradients (benutzt set_adam_step)
        for layer in &mut self.layers {
            layer.apply_grads(lr);
        }
        // Inkrementiere für nächsten Update
        self.adam_step += 1;
        self.clear_grads();
    }
}

// In der train() Methode:

pub fn train<B: Batches>(&mut self, batches: B, lr: f32, iteration: &mut usize, j: &mut usize, batch_size: usize) {
    // ... forward pass und backward pass ...
    
    *iteration += 1;
    if *iteration % batch_size == 0 {
        *j += 1;
        // ← ÄNDERUNG: Nutze Adam oder SGD basierend auf config
        match crate::config::OPTIMIZER {
            crate::config::OptimizerType::SGD => self.sgd_step(lr),
            crate::config::OptimizerType::Adam => self.adam_step(lr),
        }
    }
}

// Optional: Methode um Optimizer zu resetten
impl Sequential {
    pub fn reset_optimizer(&mut self) {
        match crate::config::OPTIMIZER {
            crate::config::OptimizerType::SGD => {
                // Nichts zu tun
            }
            crate::config::OptimizerType::Adam => {
                self.adam_step = 0;
                for layer in &mut self.layers {
                    layer.reset_adam_moments();
                }
            }
        }
    }
}


// ──────────────────────────────────────────────────────────────────────────────
// ANWENDUNG
// ──────────────────────────────────────────────────────────────────────────────
//
// Das wars! Jetzt kannst du in config.rs umschalten:
//
//   pub const OPTIMIZER: OptimizerType = OptimizerType::Adam;
//
// Und dann trainiert dein Netzwerk mit Adam statt SGD!
//
// Tipp: Die anderen Layer (LSTM, Linear, etc.) brauchen das selbe Update-Pattern.
// Kopiere einfach die gleiche Logik für die anderen Layers.
