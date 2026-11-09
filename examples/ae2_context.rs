//! How much the text model gains from context on LibriSpeech transcripts: each
//! held-out utterance alone versus the utterances of its chapter run as one
//! window, scored per word.
//!
//!   cargo run --release --features cuda --example ae2_context -- s4

use std::range::Range;

use neural_networks::audio::librispeech::LibriSpeech;
use neural_networks::config::AUDIO_DATA;
use neural_networks::gpu::{Gpu, hierarchical::Hierarchical};
use neural_networks::segment::word_ends;
use neural_networks::tokenizer_utf8::Utf8Tokenizer;

fn words_of(tok: &Utf8Tokenizer, text: &str) -> (Vec<usize>, Vec<Range<usize>>) {
    let raw = tok.to_tokens(text);
    let mut words = Vec::new();
    let mut start = 0;
    for end in word_ends(&raw) {
        words.push(Range::from(start..end as usize));
        start = end as usize;
    }
    (raw.iter().map(|&t| t as usize).collect(), words)
}

fn main() {
    let name = std::env::args().nth(1).expect("model name");
    let gpu = Gpu::new().unwrap();
    let tok = Utf8Tokenizer::new();
    let mut model = Hierarchical::load(&gpu, &format!("models/{name}"), tok.w_token() as usize).unwrap();
    model.set_offload(&gpu, false);
    let corpus = LibriSpeech::load(AUDIO_DATA, "dev-clean").unwrap();
    let speakers: Vec<&str> = corpus.speakers().into_iter().rev().take(4).collect();

    let (mut alone, mut alone_words, mut joined, mut joined_words) = (0.0, 0, 0.0, 0);
    let mut chapter: Vec<&str> = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    let flush = |texts: &mut Vec<String>, model: &mut Hierarchical, joined: &mut f64, joined_words: &mut usize| {
        if texts.is_empty() {
            return;
        }
        let (tokens, words) = words_of(&tok, &texts.join(" "));
        model.eval_loss(&gpu, &tokens, &words);
        // Mean over the scored words, which are all but the first.
        *joined += model.last_word_loss() as f64 * (words.len() - 1) as f64;
        *joined_words += words.len() - 1;
        texts.clear();
    };
    for u in corpus.utterances.iter().filter(|u| speakers.contains(&u.speaker.as_str())) {
        let ch: Vec<&str> = u.id.split('-').take(2).collect();
        if ch != chapter {
            flush(&mut texts, &mut model, &mut joined, &mut joined_words);
            chapter = ch;
        }
        let text = u.text();
        let (tokens, words) = words_of(&tok, &text);
        if words.len() < 2 {
            continue;
        }
        model.eval_loss(&gpu, &tokens, &words);
        alone += model.last_word_loss() as f64 * (words.len() - 1) as f64;
        alone_words += words.len() - 1;
        texts.push(text);
    }
    flush(&mut texts, &mut model, &mut joined, &mut joined_words);
    println!(
        "per word: each utterance alone {:.3} ({alone_words} words), chapter as one window {:.3} ({joined_words} words)",
        alone / alone_words as f64,
        joined / joined_words as f64
    );
}
