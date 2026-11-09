//! Is continuing from given word embeddings the same as continuing from text?
//! Samples s4 (CPU) from a prompt as text and from the text encoder's own
//! embeddings of the same words fed through `sample_from_embeddings`.

use neural_networks::config::{TEMPERATURE, TOP_P};
use neural_networks::infer::Hierarchical;
use neural_networks::segment::word_ends;
use neural_networks::tokenizer_utf8::Utf8Tokenizer;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let tok = Utf8Tokenizer::new();
    let mut model = Hierarchical::load(&args[0]).unwrap();
    let text = format!("{} ", args[1]);
    let raw = tok.to_tokens(&text);
    let mut words = Vec::new();
    let mut start = 0;
    for end in word_ends(&raw) {
        words.push(&raw[start..end as usize]);
        start = end as usize;
    }
    let embeds = model.encode(&words);
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
