use crate::{
    config::{CHAR_HIDDEN, OUT_HIDDEN, WORD_HIDDEN},
    hierarchical::HierarchicalSequential,
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
pub fn build_hierarchical_model(
    vocab: usize,
    boundary_token_ids: Vec<u16>,
) -> HierarchicalSequential {
    let char_model = SequentialBuilder::new(vocab)
        .embedding(CHAR_HIDDEN)
        .slstm_block(CHAR_HIDDEN)
        .slstm_block(CHAR_HIDDEN)
        .rms_norm()
        .build();

    let heads: usize = 16;
    let dqk: usize = WORD_HIDDEN / heads / 2;

    let high_model = SequentialBuilder::new(CHAR_HIDDEN)
        .linear(WORD_HIDDEN)
        .mlstm_block(heads, dqk)
        .slstm_block(WORD_HIDDEN)
        .mlstm_block(heads, dqk)
        .mlstm_block(heads, dqk)
        .mlstm_block(heads, dqk)
        .rms_norm()
        .build();

    let heads: usize = 16;
    let dqk: usize = CHAR_HIDDEN / heads / 2;

    let char2_model = SequentialBuilder::new(CHAR_HIDDEN + WORD_HIDDEN)
        .linear(OUT_HIDDEN)
        .slstm_block(OUT_HIDDEN)
        .mlstm_block(heads, dqk)
        .slstm_block(OUT_HIDDEN)
        .rms_norm()
        .linear(vocab)
        .build();

    HierarchicalSequential::new(
        char_model,
        char2_model,
        high_model,
        vocab,
        boundary_token_ids,
    )
}
