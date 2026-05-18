use crate::{
    config::{CHAR_HIDDEN, DQK, NUM_HEADS, OUT_HIDDEN, WORD_HIDDEN},
    hierarchical::HierarchicalSequential,
    nn_layer::SequentialBuilder,
    sequential::Sequential,
};

pub fn build_normal_model(vocab: usize) -> Sequential {
    SequentialBuilder::new(vocab)
        .embedding(WORD_HIDDEN)
        .mlstm_block(NUM_HEADS, DQK)
        .slstm_block(WORD_HIDDEN)
        .mlstm_block(NUM_HEADS, DQK)
        .mlstm_block(NUM_HEADS, DQK)
        .rms_norm()
        .linear(vocab)
        .softmax()
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

    let high_model = SequentialBuilder::new(CHAR_HIDDEN)
        .linear(WORD_HIDDEN)
        .mlstm_block(NUM_HEADS, DQK)
        .slstm_block(WORD_HIDDEN)
        .mlstm_block(NUM_HEADS, DQK)
        .mlstm_block(NUM_HEADS, DQK)
        .mlstm_block(NUM_HEADS, DQK)
        .mlstm_block(NUM_HEADS, DQK)
        .mlstm_block(NUM_HEADS, DQK)
        .mlstm_block(NUM_HEADS, DQK)
        .rms_norm()
        .build();

    let char2_model = SequentialBuilder::new(CHAR_HIDDEN + WORD_HIDDEN)
        .linear(OUT_HIDDEN)
        .slstm_block(OUT_HIDDEN)
        .slstm_block(OUT_HIDDEN)
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
