//! Where the next-word gradient lands on the audio embeddings: per row, for an
//! utterance read alone and for the same utterance followed by the first words
//! of the next one as text, plus the stem's gradient norm in each case.
//!
//!   cargo run --release --features cuda --example ae2_cont_grad -- s4 models/s4.ae2.E-ctc

use std::range::Range;

use neural_networks::audio::frontend::LogMel;
use neural_networks::audio::io::Clip;
use neural_networks::audio::librispeech::LibriSpeech;
use neural_networks::audio::segments::WordSegments;
use neural_networks::config::{AUDIO_DATA, AUDIO_FEATS, AUDIO_SR};
use neural_networks::gpu::arena::TrainingCache;
use neural_networks::gpu::audio::AudioEncoder;
use neural_networks::gpu::{GTensor, Gpu, hierarchical::Hierarchical};
use neural_networks::segment::word_ends;
use neural_networks::tensor::Tensor;
use neural_networks::tokenizer_utf8::Utf8Tokenizer;

const AFTER: usize = 4;

fn split(raw: &[u16]) -> Vec<Range<usize>> {
    let mut words = Vec::new();
    let mut start = 0;
    for end in word_ends(raw) {
        words.push(Range::from(start..end as usize));
        start = end as usize;
    }
    words
}

fn row_norms(d: &[f32], hc: usize) -> Vec<f32> {
    d.chunks(hc).map(|r| r.iter().map(|x| x * x).sum::<f32>().sqrt()).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = format!("models/{}", args[0]);
    let gpu = Gpu::new().unwrap();
    let tok = Utf8Tokenizer::new();
    let mut model = Hierarchical::load(&gpu, &path, tok.w_token() as usize).unwrap();
    model.set_offload(&gpu, false);
    let (mut enc, ck) = AudioEncoder::load(&gpu, &args[1]).unwrap();
    let c = enc.cfg;
    let hc = c.out;
    println!("{} at step {}", args[1], ck.step);
    let corpus = LibriSpeech::load(AUDIO_DATA, "dev-clean").unwrap();
    let ids = ["8297-275156-0000", "8297-275156-0001", "7976-110124-0000", "7850-111771-0001", "8842-302196-0002"];
    for id in ids {
        let Some(at) = corpus.utterances.iter().position(|u| u.id == id) else {
            continue;
        };
        let u = &corpus.utterances[at];
        let next = &corpus.utterances[at + 1];
        let own = u.text();
        let mut after: String = next.text().split(' ').take(AFTER).collect::<Vec<_>>().join(" ");
        if !after.starts_with("I ") {
            after = after[..1].to_lowercase() + &after[1..];
        }

        let clip = Clip::read_at(&u.audio, AUDIO_SR).unwrap();
        let mut feats = LogMel::new().utterance(&clip.samples);
        ck.stats.apply(&mut feats);
        let t_in = feats.len() / AUDIO_FEATS;
        let t_out = c.stem().frames(t_in);
        let segs = WordSegments::from_word_times(&u.times(), t_out);
        let n = segs.len();
        let cache: TrainingCache = enc.training_cache(&gpu, t_out, t_out, t_out);

        let mut report = Vec::new();
        for text in [own.clone(), format!("{own} {after}")] {
            let raw = tok.to_tokens(&text);
            let words = split(&raw);
            let tokens: Vec<usize> = raw.iter().map(|&t| t as usize).collect();
            let cont = words.len() - n;
            enc.zero_grad(&gpu);
            let out = enc.forward(&gpu, &feats, 1, t_in, std::slice::from_ref(&segs), &cache);
            let embeds = out.embeddings.as_ref().unwrap().to_host(&gpu).data;
            let audio_rows = if cont > 0 { n } else { n - 1 };
            let mut rows = embeds[..audio_rows * hc].to_vec();
            if cont > 1 {
                let text_words: Vec<Range<usize>> = words[n..].to_vec();
                rows.extend(model.encode_embeds(&gpu, &tokens, &text_words).to_host(&gpu).data);
            }
            let dw = words.len() - 1;
            let given = GTensor::from_host(&gpu, &Tensor::new(&[dw, hc], rows));
            let mut d = GTensor::zeros(&gpu, &[dw, hc]);
            let loss = model.forward_backward_embeds(&gpu, &given, &tokens, &words, None, &mut d);
            let d = d.to_host(&gpu).data;
            let mut d_e = vec![0.0; embeds.len()];
            d_e[..audio_rows * hc].copy_from_slice(&d[..audio_rows * hc]);
            let norms = row_norms(&d_e, hc);
            let d_logits = GTensor::zeros(&gpu, &[t_out, c.delay + 1]);
            let d_e = GTensor::from_host(&gpu, &Tensor::new(&[n, hc], d_e));
            enc.backward(&gpu, &d_logits, Some(&d_e), None, &cache);
            let stem = enc.stem.dw.to_host(&gpu).data.iter().map(|x| x * x).sum::<f32>().sqrt();
            report.push((cont, loss, norms, stem));
        }
        enc.zero_grad(&gpu);
        println!("{id}: {n} words");
        for (cont, loss, norms, stem) in report {
            let fmt: Vec<String> = norms.iter().map(|x| format!("{:.1}", x * 1e4)).collect();
            println!(
                "  {cont} after: loss {loss:.3}, stem |grad| {stem:.3e}, rows ×1e-4: {}",
                fmt.join(" ")
            );
        }
    }
}
