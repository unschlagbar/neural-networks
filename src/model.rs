use crate::{
    config::{CHAR_HIDDEN, OUT_HIDDEN, WORD_HIDDEN},
    hierarchical::Hierarchical,
    nn_layer::SequentialBuilder,
    sequential::Sequential,
};

pub fn build_normal_model(vocab: usize) -> Sequential {
    pub const NUM_HEADS: usize = 16;
    pub const DQK: usize = WORD_HIDDEN / NUM_HEADS / 2;
    SequentialBuilder::new(vocab)
        .embedding(WORD_HIDDEN)
        .mlstm_block(NUM_HEADS, DQK)
        .mlstm_block(NUM_HEADS, DQK)
        .mlstm_block(NUM_HEADS, DQK)
        .rms_norm()
        .linear(vocab)
        .build()
}

// Drei gekoppelte Sub-Modelle mit FESTEN (token-basierten) Wortgrenzen:
pub fn build_hierarchical_model(vocab: usize, boundary_token_ids: Vec<u16>) -> Hierarchical {
    let char_model = SequentialBuilder::new(vocab)
        .embedding(CHAR_HIDDEN)
        .slstm_block(CHAR_HIDDEN)
        .rms_norm()
        .build();

    let heads: usize = 16;
    let dqk: usize = WORD_HIDDEN / heads / 2;

    let mut high_model = SequentialBuilder::new(CHAR_HIDDEN).linear(WORD_HIDDEN);
    for _ in 0..1 {
        high_model = high_model.mlstm_block(heads, dqk)
    }
    high_model = high_model.slstm_block(WORD_HIDDEN);

    for _ in 0..4 {
        high_model = high_model.mlstm_block(heads, dqk)
    }
    let high_model = high_model.rms_norm().build();

    let char2_model = SequentialBuilder::new(WORD_HIDDEN + CHAR_HIDDEN)
        .linear(OUT_HIDDEN)
        .slstm_block(OUT_HIDDEN)
        .rms_norm()
        .linear_no_bias(vocab)
        .build();

    Hierarchical::new(
        char_model,
        char2_model,
        high_model,
        vocab,
        boundary_token_ids,
    )
}
