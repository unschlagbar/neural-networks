use std::rc::Rc;

use crate::{
    config::{CHAR_HIDDEN, OUT_HIDDEN, WORD_BLOCKS, WORD_HIDDEN},
    hierarchical::Hierarchical,
    nn_layer::SequentialBuilder,
    sequential::Sequential,
    tokenizer::Tokenizer,
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

// Hierarchical Autoregressive Transformer (HAT)-style three-part model.
//
//   encoder  (char_model)  — embeds the complete characters of one word into a
//                            single CHAR_HIDDEN vector (final hidden state).
//   backbone (word_model)  — autoregressive over word embeddings; its output is
//                            the model-space prediction for the *next* word.
//   decoder  (char2_model) — generates the characters of a word one at a time,
//                            conditioned (per step) on the backbone context that
//                            was concatenated onto its input. Reset per word.
pub fn build_hierarchical_model(
    vocab: usize,
    boundary_token_ids: Vec<u16>,
    tokenizer: Rc<Tokenizer>,
) -> Hierarchical {
    let char_model = SequentialBuilder::new(vocab)
        .embedding(CHAR_HIDDEN)
        .slstm_block(CHAR_HIDDEN)
        .slstm_block(CHAR_HIDDEN)
        .linear(CHAR_HIDDEN)
        .build();

    let heads: usize = 8;
    let dqk: usize = WORD_HIDDEN / heads;

    let mut word_model = SequentialBuilder::new(CHAR_HIDDEN).linear(WORD_HIDDEN);
    for _ in 0..WORD_BLOCKS {
        word_model = word_model.mlstm_block(heads, dqk)
    }
    let word_model = word_model.linear(OUT_HIDDEN).build();

    let char2_builder = SequentialBuilder::new(OUT_HIDDEN + vocab);
    let char2_model = char2_builder
        .linear(OUT_HIDDEN)
        .slstm_block(OUT_HIDDEN)
        .slstm_block(OUT_HIDDEN)
        .linear(OUT_HIDDEN)
        .rms_norm()
        .linear_no_bias(vocab)
        .build();

    Hierarchical::new(
        char_model,
        char2_model,
        word_model,
        vocab,
        boundary_token_ids,
        tokenizer,
    )
}
