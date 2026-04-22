// ── model.rs ─────────────────────────────────────────────────────────────────
//
// Alle Modell-Architekturen an einer Stelle. Wenn du eine neue Architektur
// ausprobieren willst, mach das hier — nicht in main.rs, nicht in training.rs.
//
// Beide Builder sind so aufgebaut, dass sie fail-fast sind: wenn du eine
// inkompatible Dimension nimmst (z.B. slstm_block mit input != hidden), knallt
// es im Builder und nicht erst beim Training.

use crate::{
    activations::Linear,
    config::{CHAR_HIDDEN, CONTEXT_DIM},
    hierarchical_sequential::HierarchicalSequential,
    nn_layer::SequentialBuilder,
    sequential::Sequential,
};

// ── „normales" Sequential-Modell (Char-Level) ────────────────────────────────
//
// Alte Architektur:
//     vocab → Projection(C) → [ slstm+rms · slstm+rms · silu_dense+rms ]×2
//             → Dense(vocab) → Softmax
//
// Probleme daran:
//   • slstm-Zellen mit res_rms_norm ≠ „sLSTM-Block" aus dem xLSTM-Paper.
//     Es wurde zwar Normierung addiert, aber kein richtiges Channel-Mixing
//     MIT Gating zwischen den Zellen — das Paper macht das explizit mit
//     post-up-projection / SwiGLU.
//   • Die silu_dense als ersatz für den SwiGLU-Teil: SiluDense ist einfach
//     `x → SiLU(Wx+b)`, also nur eine aktivierte Dense-Schicht ohne Gate.
//     Das Gate ist aber genau der Teil, der dem MLP die Selektivität gibt.
//
// Neue Architektur:
//     vocab → Projection(C) → [ SLSTMBlock(C) ]×3 → Dense(vocab) → Softmax
//
// Ein SLSTMBlock kapselt bereits:
//     Pre-RMSNorm → sLSTM-Zelle → Post-RMSNorm → SwiGLU-MLP → Residual
// Also übernimmt er die Rolle der alten 3-Zeilen-Sequenz mit korrekter
// Semantik. 3 Blöcke ≈ die vorige Tiefe, aber mit weniger Parametern pro
// Block und höherer effektiver Kapazität durch das Gate.
pub fn build_normal_model(vocab: usize) -> Sequential {
    let mut model = SequentialBuilder::new(vocab).linear(CONTEXT_DIM);
    for _ in 0..6 {
        model = model.slstm_block(CONTEXT_DIM);
    }
    model.linear(vocab).softmax().build()
}

// ── Hierarchisches Modell ────────────────────────────────────────────────────
//
// Char-Stack und High-Stack werden ebenfalls auf slstm_block umgestellt, weil
// dort dasselbe Problem auftritt — nackte Zellen mit Residual-Norm reichen
// nicht. Die Projection am Anfang/Ende bleibt, sie ist die dim-Anpassung
// Zwischen Vocab/Context.
pub fn build_hierarchical_model(vocab: usize, boundary_ids: Vec<u16>) -> HierarchicalSequential {
    // ── Char-Level-Teilmodell ─────────────────────────────────────────────
    // Input: one-hot(vocab) ⊕ context(CONTEXT_DIM)   (siehe HierarchicalSequential)
    let mut char_model = SequentialBuilder::new(vocab + CONTEXT_DIM).project(CHAR_HIDDEN, Linear);
    for _ in 0..4 {
        char_model = char_model.slstm_block(CHAR_HIDDEN);
    }
    let char_model = char_model.project(vocab, Linear).softmax().build();

    // ── High-Level-Teilmodell ─────────────────────────────────────────────
    let mut high_model = SequentialBuilder::new(CHAR_HIDDEN).project(CONTEXT_DIM, Linear);
    for _ in 0..4 {
        high_model = high_model.slstm_block(CONTEXT_DIM);
    }
    let high_model = high_model.project(CONTEXT_DIM, Linear).build();

    HierarchicalSequential::new(char_model, high_model, vocab, boundary_ids, 5)
}
