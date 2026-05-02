use crate::{
    config::{CHAR_HIDDEN, WORD_HIDDEN},
    hierarchical::HierarchicalSequential,
    nn_layer::SequentialBuilder,
    sequential::Sequential,
};

pub fn build_normal_model(vocab: usize) -> Sequential {
    let mut model = SequentialBuilder::new(vocab).embedding(WORD_HIDDEN);
    for _ in 0..2 {
        model = model.slstm_block(WORD_HIDDEN);
    }
    model.linear(vocab).softmax().build()
}

// ── Hierarchisches Modell (HierarchicalSequential) ───────────────────────────
//
// Drei gekoppelte Sub-Modelle mit FESTEN (token-basierten) Wortgrenzen:
pub fn build_hierarchical_model(
    vocab: usize,
    boundary_token_ids: Vec<u16>,
) -> HierarchicalSequential {
    let char_model = SequentialBuilder::new(vocab)
        .embedding(CHAR_HIDDEN)
        .slstm_block(CHAR_HIDDEN)
        .rms_norm()
        .build();

    let high_model = SequentialBuilder::new(vocab + CHAR_HIDDEN)
        .linear(WORD_HIDDEN)
        .slstm_block(WORD_HIDDEN)
        .slstm_block(WORD_HIDDEN)
        .rms_norm()
        .build();

    let char2_model = SequentialBuilder::new(CHAR_HIDDEN + WORD_HIDDEN)
        .linear(CHAR_HIDDEN * 2)
        .slstm_block(CHAR_HIDDEN * 2)
        .rms_norm()
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
