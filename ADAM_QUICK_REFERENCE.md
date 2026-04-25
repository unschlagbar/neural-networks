// ═══════════════════════════════════════════════════════════════════════════════
// ADAM OPTIMIZER — QUICK REFERENCE
// ═══════════════════════════════════════════════════════════════════════════════

// MATHEMATIK:
//
// Adam (Adaptive Moment Estimation) benutzt zwei exponential moving averages:
//
//   m_t = β₁ * m_{t-1} + (1 - β₁) * g_t        ← erstes Moment (mean)
//   v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²       ← zweites Moment (second moment)
//
//   m̂_t = m_t / (1 - β₁^t)                     ← Bias-korrigierter Mean
//   v̂_t = v_t / (1 - β₂^t)                     ← Bias-korrigierte Varianz
//
//   θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)      ← Parameter Update
//
// Hyperparameter:
//   β₁ = 0.9      (decay für Gradient-Mean, typisch 0.9)
//   β₂ = 0.999    (decay für Gradient-Varianz, typisch 0.999)
//   ε  = 1e-8     (small constant for numerical stability)
//   α  = lr       (learning rate, typisch 1e-3 oder 1e-4 für Adam)


// ═══════════════════════════════════════════════════════════════════════════════
// INTEGRATION CHECKLIST
// ═══════════════════════════════════════════════════════════════════════════════

// ✓ [1] Erstelle src/adam_optimizer.rs
//       - AdamConfig, ScalarMoment, MatrixMoments, VectorMoments
//       - Helper Funktionen für Updates
//       - BEREITS ERLEDIGT ✓

// ✓ [2] Füge in lib.rs ein:
//       pub mod adam_optimizer;
//       - BEREITS ERLEDIGT ✓

// [ ] [3] Update config.rs:
//   pub enum OptimizerType { SGD, Adam }
//   pub const OPTIMIZER: OptimizerType = OptimizerType::Adam;
//   pub const ADAM_BETA1: f32 = 0.9;
//   pub const ADAM_BETA2: f32 = 0.999;
//   pub const ADAM_EPS: f32 = 1e-8;

// [ ] [4] Update JEDEN Layer (dense.rs, lstm.rs, linear.rs, ...):
//   a) DenseGrads mit MatrixMoments, VectorMoments erweitern
//   b) DenseLayer mit adam_step: u32 Feld
//   c) apply_grads_sgd() und apply_grads_adam() Methoden
//   d) apply_grads() → Match zwischen SGD und Adam
//   e) set_adam_step() implementieren in NnLayer trait

// [ ] [5] Update nn_layer.rs Trait:
//   fn set_adam_step(&mut self, _t: u32) {}
//   fn reset_adam_moments(&mut self) {}

// [ ] [6] Update sequential.rs:
//   a) Feld adam_step: u32 hinzufügen
//   b) adam_step() Methode schreiben (wie sgd_step aber mit set_adam_step)
//   c) train() → Match OPTIMIZER im Gradient-Update
//   d) reset_optimizer() Optional

// [ ] [7] Ähnlich für hierarchical_sequential.rs


// ═══════════════════════════════════════════════════════════════════════════════
// TIPPS & BEST PRACTICES
// ═══════════════════════════════════════════════════════════════════════════════

// 1. Learning Rate:
//    SGD:  Typisch 1e-4 bis 1e-3
//    Adam: Typisch 1e-3 bis 1e-4 (höher als SGD!)
//
//    Wenn dein Netzwerk divergiert:
//    - Learning Rate reduzieren (z.B. 1e-4)
//    - Batch Size erhöhen

// 2. Batch Normalization & Adam:
//    Adam funktioniert besser mit Batch Normalization.
//    Dein Code verwendet RMSNorm — auch gut!

// 3. Gradient Clipping (optional, aber empfohlen):
//    Falls Gradienten explodieren, clipp sie:
//    
//    grad.clip(-1.0, 1.0);  // max |gradient| = 1.0
//    
//    Dann kann Adam besser stabilisieren.

// 4. Weight Decay (L2 Regularization):
//    Adam kann mit Weight Decay kombiniert werden:
//    
//    // In apply_grads_adam:
//    param -= lr * weight_decay * param;
//    
//    Starke Version ("AdamW") ist für viele Modelle besser.

// 5. Momentum Schnittstelle (SGD mit Momentum):
//    Falls du auch SGD mit Momentum willst, erweitere das selbe Pattern:
//    
//    pub enum OptimizerType {
//        SGD,
//        SGDMomentum,
//        Adam,
//        AdamW,
//    }


// ═══════════════════════════════════════════════════════════════════════════════
// TESTING & DEBUGGING
// ═══════════════════════════════════════════════════════════════════════════════

// 1. Prüfe dass die Momente akkumuliert werden:
//
//    println!("Moment m: {:.6}", layer.grads.weights_moments.m[0][0]);
//    println!("Moment v: {:.6}", layer.grads.weights_moments.v[0][0]);

// 2. Vergleich SGD vs Adam:
//
//    pub const OPTIMIZER: OptimizerType = OptimizerType::SGD;  // Test
//    pub const LR: f32 = 3e-4;
//    // vs
//    pub const OPTIMIZER: OptimizerType = OptimizerType::Adam;
//    pub const LR: f32 = 1e-3;
//
//    Adam sollte schneller konvergieren!

// 3. Prüfe die Gradienten:
//
//    if iteration % 100 == 0 {
//        let mean_grad = self.layers[0].grads.weights.mean();
//        println!("Mean gradient: {:.6}", mean_grad);
//    }
//
//    Typisch: Mean gradient sinkt während Training → Good!

// 4. Numerik-Stabilität:
//
//    Falls NaN/Inf auftauchen:
//    - eps erhöhen (z.B. 1e-7 statt 1e-8)
//    - Learning Rate reduzieren
//    - Gradienten clippen


// ═══════════════════════════════════════════════════════════════════════════════
// BEFEHLE ZUM SOFORT STARTEN
// ═══════════════════════════════════════════════════════════════════════════════

// 1. Kompilierungstest:
//    cargo build --release
//    → Sollte ohne Fehler durchlaufen (außer der fehlenden Implementierungen)

// 2. Nach Implementierung testen:
//    cargo build --release
//    cargo run --release --bin neural-networks

// 3. Schneller Benchmark:
//    cargo bench --bench lstm_training

// 4. Mit Debugging Logs starten:
//    RUST_LOG=debug cargo run --bin neural-networks


// ═══════════════════════════════════════════════════════════════════════════════
// WEITERE RESOURCES
// ═══════════════════════════════════════════════════════════════════════════════

// Original Adam Paper:
//   Kingma, D. P., & Ba, J. (2014).
//   "Adam: A Method for Stochastic Optimization"
//   https://arxiv.org/abs/1412.6980

// AdamW (improved variant):
//   Loshchilov, I., & Hutter, F. (2019).
//   "Decoupled Weight Decay Regularization"
//   https://arxiv.org/abs/1711.05101

// PyTorch Adam Impl (für Vergleich):
//   https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

// TensorFlow Adam:
//   https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
