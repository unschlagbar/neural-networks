// ── model.rs ─────────────────────────────────────────────────────────────────
//
// Alle Modell-Architekturen an einer Stelle. Wenn du eine neue Architektur
// ausprobieren willst, mach das hier — nicht in main.rs, nicht in training.rs.
//
// Beide Builder sind so aufgebaut, dass sie fail-fast sind: wenn du eine
// inkompatible Dimension nimmst (z.B. slstm_block mit input != hidden), knallt
// es im Builder und nicht erst beim Training.

use crate::{
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
    let mut model = SequentialBuilder::new(vocab);
    for _ in 0..1 {
        model = model.slstm_block(vocab);
        // model = model.lstm(CONTEXT_DIM);
    }
    model.softmax().build()
}

// ── Hierarchisches Modell (HierarchicalSequential) ───────────────────────────
//
// Drei gekoppelte Sub-Modelle mit FESTEN (token-basierten) Wortgrenzen:
pub fn build_hierarchical_model(
    vocab: usize,
    boundary_token_ids: Vec<u16>,
) -> HierarchicalSequential {
    // ── char_model (vocab → CHAR_HIDDEN) ──────────────────────────────────
    // Kurz (2 Blöcke): wird oft zurückgesetzt, mehr Tiefe bringt hier wenig.
    let char_model = SequentialBuilder::new(vocab)
        .linear(CHAR_HIDDEN)
        .slstm_block(CHAR_HIDDEN)
        .slstm_block(CHAR_HIDDEN)
        .build();

    // ── high_model (CHAR_HIDDEN → CONTEXT_DIM) ────────────────────────────
    // Ein Block reicht: der Informationsfluss kommt kondensiert aus char_model.
    let high_model = SequentialBuilder::new(vocab + CHAR_HIDDEN)
        .linear(CONTEXT_DIM)
        .slstm_block(CONTEXT_DIM)
        .slstm_block(CONTEXT_DIM)
        .build();

    // ── char2_model ([CHAR_HIDDEN + CONTEXT_DIM] → vocab) ─────────────────
    // Prediction-Head: SiluDense für nichtlineare Interaktion, dann auf vocab.
    let char2_model = SequentialBuilder::new(CHAR_HIDDEN + CONTEXT_DIM)
        .slstm_block(CHAR_HIDDEN + CONTEXT_DIM)
        .linear(vocab)
        .softmax()
        .build();

    HierarchicalSequential::new(
        char_model,
        char2_model,
        high_model,
        vocab,
        boundary_token_ids,
    )
}
