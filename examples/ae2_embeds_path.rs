//! Is continuing from given word embeddings the same as continuing from text?
//! Samples s4 (CPU) from a prompt as text and from the text encoder's own
//! embeddings of the same words fed through `sample_from_embeddings`.
//!
//!   cargo run --release --example ae2_embeds_path -- models/s4 "prompt text"

use neural_networks::config::{TEMPERATURE, TOP_P};
use neural_networks::hierarchical::Hierarchical;
use neural_networks::segment::word_ends;
use neural_networks::tokenizer_utf8::Utf8Tokenizer;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let tok = Utf8Tokenizer::new();
    let mut model = Hierarchical::load(&args[0], Utf8Tokenizer::new()).unwrap();
    let text = format!("{} ", args[1]);
    let raw = tok.to_tokens(&text);
    let mut embeds = Vec::new();
    let mut start = 0;
    model.make_cache(1, neural_networks::config::MAX_SEQ_LEN);
    for end in word_ends(&raw) {
        embeds.extend_from_slice(model.encoder.encode_word_sample(&raw[start..end as usize]));
        start = end as usize;
    }
    for k in 0..3 {
        let out = model.sample(&raw, 200, TEMPERATURE, TOP_P, |_| true);
        println!(
            "text  {k} → {}",
            String::from_utf8_lossy(
                &out.iter()
                    .filter(|&&t| t < 256)
                    .map(|&t| t as u8)
                    .collect::<Vec<_>>()
            )
        );
        let out = model.sample_from_embeddings(&embeds, 200, TEMPERATURE, TOP_P, |_| true);
        println!(
            "embed {k} → {}",
            String::from_utf8_lossy(
                &out.iter()
                    .filter(|&&t| t < 256)
                    .map(|&t| t as u8)
                    .collect::<Vec<_>>()
            )
        );
    }
}
