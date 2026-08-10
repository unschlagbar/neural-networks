// Grow a pretrained hierarchical checkpoint's vocabulary.
//
// A model trained before SFT knows only `<W>` and `<END>` on top of the 256
// byte tokens (vocab 258). Post-training adds chat markers (`<CONTEXT>`,
// `<SEP>`, …) to `SPECIAL_TOKENS`, which raises `vocab_size()`. Those new ids
// have no rows in the checkpoint's two vocab-sized tables:
//
//   - the encoder's char **embedding** — `[vocab, HC]`, one row per token id
//     (this table is tied, so it also feeds the decoder's char slots), and
//   - the decoder's logit **head** (`LinearNoBias`) — `[HC, vocab]`, one column
//     per token id.
//
// Growing the model = appending fresh rows/columns for the new ids and leaving
// every existing weight byte-for-byte unchanged, so the pretrained behaviour is
// preserved and only the new markers start untrained. New embedding rows get a
// small random init (like a fresh table row); new head columns start at zero so
// the untrained markers begin with a neutral logit and cannot dominate the
// softmax before they have learned anything.
//
// This rewrites the checkpoint's `encoder` and `char2_model` sections in place
// (via the NNM1 container) and bumps the stored `vocab`. The backbone is
// untouched — it never sees token ids. A model already at the target vocab is
// left alone.

use std::io;

use iron_oxide::collections::Matrix;

use crate::{
    hierarchical::HierStacks,
    nn::{embedding::EmbeddingLayer, linear_nb::LinearNBLayer, soft_cap::SoftCapLayer},
    format::{Meta, ModelKind, Writer},
    tokenizer_utf8::Utf8Tokenizer,
};

/// Widen a `[rows, cols]` matrix to `[new_rows, cols]` (append rows), or to
/// `[rows, new_cols]` (append cols) — whichever `grow_rows` selects. Existing
/// entries keep their exact value; new entries come from `fill(r, c)`.
fn grow_matrix(
    old: &Matrix,
    new_rows: usize,
    new_cols: usize,
    fill: impl Fn(usize, usize) -> f32,
) -> Matrix {
    let (r, c) = (old.rows(), old.cols());
    debug_assert!(new_rows >= r && new_cols >= c);
    let mut out = Matrix::zeros(new_rows, new_cols);
    for i in 0..new_rows {
        for j in 0..new_cols {
            let v = if i < r && j < c {
                old[i][j]
            } else {
                fill(i, j)
            };
            out.set(i, j, v);
        }
    }
    out
}

/// Grow the encoder embedding (`[vocab, HC]`) — append rows for the new ids with
/// a small random init matching a fresh embedding row's scale.
fn grow_embedding(emb: &EmbeddingLayer, new_vocab: usize) -> EmbeddingLayer {
    let hc = emb.output_size();
    let scale = (6.0 / (new_vocab as f32 + hc as f32)).sqrt();
    let weights = grow_matrix(&emb.weights, new_vocab, hc, |_, _| {
        rand::random_range(-scale..scale)
    });
    EmbeddingLayer::from_loaded(new_vocab, hc, weights)
}

/// Grow the decoder logit head (`[HC, vocab]`) — append zero columns for the new
/// ids so their initial logit is neutral.
fn grow_head(head: &LinearNBLayer, new_vocab: usize) -> LinearNBLayer {
    let hc = head.weights.rows();
    let weights = grow_matrix(&head.weights, hc, new_vocab, |_, _| 0.0);
    LinearNBLayer::from_loaded(hc, new_vocab, weights)
}

/// Load a hierarchical checkpoint, grow its two vocab-sized tables to
/// `new_vocab`, and write it back to `out_path` (same NNM1 container).
///
/// Errors if the file is not a hierarchical model, if it is already *larger*
/// than `new_vocab`, or if the encoder/decoder do not have the expected
/// embedding/head layers. A checkpoint already exactly at `new_vocab` is copied
/// through unchanged.
pub fn grow_checkpoint(in_path: &str, out_path: &str, new_vocab: usize) -> io::Result<()> {
    let stacks = HierStacks::load(in_path)?;
    let old_vocab = stacks.vocab_size;

    if old_vocab > new_vocab {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("checkpoint vocab {old_vocab} is already larger than target {new_vocab}"),
        ));
    }

    let HierStacks {
        mut encoder_chars,
        word_model,
        mut char2_model,
        context_size,
        step,
        seen,
        ..
    } = stacks;

    if old_vocab < new_vocab {
        // Encoder: layer 0 must be the embedding.
        let emb = encoder_chars.layers[0]
            .as_any()
            .downcast_ref::<EmbeddingLayer>()
            .ok_or_else(|| invalid("encoder must start with an EmbeddingLayer"))?;
        encoder_chars.layers[0] = Box::new(grow_embedding(emb, new_vocab));

        // Decoder: the LinearNoBias head, followed by its SoftCap. Grow both
        // (the SoftCap is just an elementwise cap sized to vocab).
        let head_idx = char2_model
            .layers
            .iter()
            .position(|l| l.as_any().is::<LinearNBLayer>())
            .ok_or_else(|| invalid("decoder is missing its LinearNoBias head"))?;
        let head = char2_model.layers[head_idx]
            .as_any()
            .downcast_ref::<LinearNBLayer>()
            .unwrap();
        let cap = char2_model
            .layers
            .iter()
            .find_map(|l| l.as_any().downcast_ref::<SoftCapLayer>())
            .map(|s| s.cap)
            .unwrap_or(crate::config::LOGIT_SOFTCAP);
        char2_model.layers[head_idx] = Box::new(grow_head(head, new_vocab));
        if let Some(sc_idx) = char2_model
            .layers
            .iter()
            .position(|l| l.as_any().is::<SoftCapLayer>())
        {
            char2_model.layers[sc_idx] = Box::new(SoftCapLayer::new(new_vocab, cap));
        }
    }

    Writer::new(
        ModelKind::Hierarchical,
        Meta {
            vocab_size: new_vocab as u32,
            context_size: context_size as u32,
            step: step as u64,
            seen,
        },
    )
    .section("encoder", &encoder_chars.layers)
    .section("word_model", &word_model.layers)
    .section("char2_model", &char2_model.layers)
    .save(out_path)?;

    println!(
        "grew vocab {old_vocab} → {new_vocab}: '{in_path}' → '{out_path}'  (step {step})"
    );
    Ok(())
}

/// Interactive entry point (`av` mode): prompt for an input model, grow it to
/// the current `Utf8Tokenizer::vocab_size()`, and write `<in>_sft`.
pub fn grow_model_interactive() {
    use std::io::{Write, stdin, stdout};

    let read = |prompt: &str, default: &str| -> String {
        print!("{prompt} [{default}]: ");
        stdout().flush().ok();
        let mut line = String::new();
        stdin().read_line(&mut line).ok();
        let name = line.trim();
        let name = if name.is_empty() { default } else { name };
        if name.contains('/') {
            name.to_string()
        } else {
            format!("models/{name}")
        }
    };

    let in_path = read("Pretrained model to grow", "models/hier_gpu");
    let out_default = format!("{in_path}_sft");
    let out_path = read("Output model", &out_default);
    let new_vocab = Utf8Tokenizer::new().vocab_size();

    match grow_checkpoint(&in_path, &out_path, new_vocab) {
        Ok(()) => println!("done."),
        Err(e) => eprintln!("grow failed: {e}"),
    }
}

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn_layer::SequentialBuilder;

    /// Growing preserves every existing weight and only appends new rows/cols.
    #[test]
    fn grow_preserves_existing_and_appends() {
        let old_vocab = 10;
        let new_vocab = 13;
        let hc = 4;

        let emb = EmbeddingLayer::new(old_vocab, hc);
        let grown = grow_embedding(&emb, new_vocab);
        assert_eq!(grown.input_size(), new_vocab);
        assert_eq!(grown.output_size(), hc);
        for i in 0..old_vocab {
            for j in 0..hc {
                assert_eq!(grown.weights[i][j], emb.weights[i][j]);
            }
        }

        let head = LinearNBLayer::new(hc, old_vocab);
        let ghead = grow_head(&head, new_vocab);
        assert_eq!(ghead.weights.rows(), hc);
        assert_eq!(ghead.weights.cols(), new_vocab);
        for i in 0..hc {
            for j in 0..old_vocab {
                assert_eq!(ghead.weights[i][j], head.weights[i][j]);
            }
            // New columns start at zero.
            for j in old_vocab..new_vocab {
                assert_eq!(ghead.weights[i][j], 0.0);
            }
        }
    }

    /// A full checkpoint round-trip: grow a saved hierarchical model and confirm
    /// the reloaded stacks carry the new vocab with the backbone untouched.
    #[test]
    fn grow_checkpoint_roundtrips() {
        let tok = Utf8Tokenizer::new();
        let old_vocab = 258;
        let hc = 8;
        let wh = 8;

        let encoder = SequentialBuilder::new(old_vocab).embedding(hc).rms_norm().build();
        let word_model = SequentialBuilder::new(hc).rms_norm().linear(wh).linear(hc).build();
        let char2_model = SequentialBuilder::new(hc)
            .rms_norm()
            .linear_no_bias(old_vocab)
            .soft_cap(30.0)
            .build();
        let model = crate::hierarchical::Hierarchical::new(
            encoder, char2_model, word_model, old_vocab, tok,
        );
        let dir = std::env::temp_dir();
        let src = dir.join("grow_src.model");
        let dst = dir.join("grow_dst.model");
        model.save(src.to_str().unwrap()).unwrap();

        let new_vocab = 261;
        grow_checkpoint(src.to_str().unwrap(), dst.to_str().unwrap(), new_vocab).unwrap();

        let stacks = HierStacks::load(dst.to_str().unwrap()).unwrap();
        assert_eq!(stacks.vocab_size, new_vocab);
        assert_eq!(stacks.encoder_chars.layers[0].input_size(), new_vocab);
        assert_eq!(stacks.char2_model.output_size, new_vocab);
        let _ = std::fs::remove_file(src);
        let _ = std::fs::remove_file(dst);
    }
}
