use std::rc::Rc;

use crate::{
    config::{CHAR_HIDDEN, OUT_HIDDEN, WORD_BLOCKS, WORD_HIDDEN},
    hierarchical::Hierarchical,
    nn::{linear_nb::LinearNBLayer, rms_norm::RMSNorm},
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
//   encoder  (char_fwd + char_bwd) — a bidirectional sLSTM over the characters of
//                            one word, prefixed with the `[W]` marker. The word
//                            embedding is read off the `[W]` position in each
//                            direction and merged by `combine` into a CHAR_HIDDEN
//                            vector.
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
    // One sLSTM encoder stack per direction (no shared weights). The word embedding
    // is the raw sLSTM hidden state at the `[W]` position, so no per-step linear /
    // norm head here — `combine` merges the two directions instead.
    let encoder = || {
        SequentialBuilder::new(vocab)
            .embedding(CHAR_HIDDEN)
            .slstm_block(CHAR_HIDDEN)
            .slstm_block(CHAR_HIDDEN)
            .build()
    };
    let char_fwd = encoder();
    let char_bwd = encoder();

    // Merges concat(fwd@[W], bwd@[W]) ∈ R^{2H} → the word embedding e_w ∈ R^H.
    let combine = LinearNBLayer::new(2 * CHAR_HIDDEN, CHAR_HIDDEN);
    let combine_norm = RMSNorm::new(CHAR_HIDDEN);

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
        char_fwd,
        char_bwd,
        combine,
        combine_norm,
        char2_model,
        word_model,
        vocab,
        boundary_token_ids,
        tokenizer,
    )
}
