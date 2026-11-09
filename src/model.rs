use crate::{
    config::{CHAR_HIDDEN, LOGIT_SOFTCAP, OUT_HIDDEN, WORD_BLOCKS, WORD_HIDDEN},
    hierarchical::Hierarchical,
    nn_layer::SequentialBuilder,
    tokenizer_utf8::Utf8Tokenizer,
};

// Hierarchical Autoregressive Transformer (HAT)-style model (arXiv 2501.10322).
//
//   encoder  (word_encoder) — a normal forward sLSTM stack over the characters
//                            of one word plus a closing [W] step. The word
//                            embedding e_w is the RMSNorm'd hidden state at the
//                            [W] step, where the state has seen the whole word
//                            and knows it is complete. Reset per word.
//   backbone (word_model)  — autoregressive over word embeddings; its output is
//                            the model-space prediction for the *next* word.
//   decoder  (char2_model) — generates the characters of a word one at a time.
//                            The backbone context is injected as the first
//                            sequence step (paper eq. 3–4: p_i takes the BOS
//                            slot); every following step feeds the previous
//                            char through the ENCODER's char embedding (tied
//                            table, held by `Hierarchical`). Predicts the
//                            word's chars followed by a trailing `[W]` (EOS).
//                            Reset per word.
pub fn build_hierarchical_model(vocab: usize, tokenizer: Utf8Tokenizer) -> Hierarchical {
    // No trailing norm: the decoder's pre-head RMSNorm is the only stage-level
    // norm in the whole hierarchical stack (the blocks keep their internal ones).
    let encoder = SequentialBuilder::new(vocab)
        .embedding(CHAR_HIDDEN)
        .slstm_block(CHAR_HIDDEN)
        .mlstm_block(16, OUT_HIDDEN / 16)
        .slstm_block(CHAR_HIDDEN)
        .build();

    let heads: usize = 8;
    let dqk: usize = WORD_HIDDEN / heads;

    // The in/out projections are hidden layers, so they use `linear` (Muon with
    // weight decay) — `linear_no_bias` trains on plain Adam and is reserved for
    // embedding-like tables and logit heads.
    let mut word_model = SequentialBuilder::new(CHAR_HIDDEN).linear(WORD_HIDDEN);
    for i in 0..WORD_BLOCKS {
        if i.is_multiple_of(4) {
            word_model = word_model.slstm_block(WORD_HIDDEN)
        } else {
            word_model = word_model.mlstm_block(heads, dqk)
        }
    }
    let word_model = word_model.linear(OUT_HIDDEN).build();

    // Decoder: no front layer — `Hierarchical` builds the inputs itself
    // (the injected context for slot 0, rows of the tied encoder char
    // embedding after that), so the stack starts at decoder width. Requires
    // CHAR_HIDDEN == OUT_HIDDEN for the tie; asserted in `Hierarchical::new`.
    let char2_model = SequentialBuilder::new(OUT_HIDDEN)
        .slstm_block(OUT_HIDDEN)
        .mlstm_block(16, OUT_HIDDEN / 16)
        .slstm_block(OUT_HIDDEN)
        .rms_norm()
        .linear_no_bias(vocab)
        .soft_cap(LOGIT_SOFTCAP)
        .build();

    Hierarchical::new(encoder, char2_model, word_model, vocab, tokenizer)
}
