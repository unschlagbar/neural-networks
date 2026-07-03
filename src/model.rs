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
        .slstm_block(WORD_HIDDEN)
        .mlstm_block(NUM_HEADS, DQK)
        .rms_norm()
        .linear(vocab)
        .build()
}

// Hierarchical Autoregressive Transformer (HAT)-style model (arXiv 2501.10322).
//
//   encoder  (word_encoder) — a normal forward sLSTM stack over the characters
//                            of one word. The word embedding e_w is the RMSNorm'd
//                            hidden state at the LAST character, where the state
//                            has seen the whole word. Reset per word.
//   backbone (word_model)  — autoregressive over word embeddings; its output is
//                            the model-space prediction for the *next* word.
//   decoder  (char2_model) — generates the characters of a word one at a time,
//                            conditioned (per step) on the backbone context that
//                            was concatenated onto its input. The decoder is fed a
//                            leading `[W]` (BOS) and predicts the word's chars
//                            followed by a trailing `[W]` (EOS). Reset per word.
pub fn build_hierarchical_model(
    vocab: usize,
    boundary_token_ids: Vec<u16>,
    tokenizer: Rc<Tokenizer>,
) -> Hierarchical {
    // The trailing RMSNorm keeps e_w at a stable scale for the backbone.
    let encoder = SequentialBuilder::new(vocab)
        .embedding(CHAR_HIDDEN)
        .slstm_block(CHAR_HIDDEN)
        .slstm_block(CHAR_HIDDEN)
        .rms_norm()
        .build();

    let heads: usize = 8;
    let dqk: usize = WORD_HIDDEN / heads;

    let mut word_model = SequentialBuilder::new(CHAR_HIDDEN).linear_no_bias(WORD_HIDDEN);
    for i in 0..WORD_BLOCKS {
        if i.is_multiple_of(2) {
            word_model = word_model.slstm_block(WORD_HIDDEN)
        } else {
            word_model = word_model.mlstm_block(heads, dqk)
        }
    }
    let word_model = word_model.rms_norm().linear_no_bias(OUT_HIDDEN).build();

    let char2_builder = SequentialBuilder::new(OUT_HIDDEN + vocab);
    let char2_model = char2_builder
        .linear_no_bias(OUT_HIDDEN)
        .slstm_block(OUT_HIDDEN)
        .slstm_block(OUT_HIDDEN)
        .rms_norm()
        .linear_no_bias(vocab)
        .build();

    Hierarchical::new(
        encoder,
        char2_model,
        word_model,
        vocab,
        boundary_token_ids,
        tokenizer,
    )
}
