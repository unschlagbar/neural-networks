//! How hard each loss pulls on the audio encoder's shared frame stack: the stem's
//! gradient norm from the boundary loss alone and from the next-word loss alone,
//! on a few held-out utterances, plus the raw (pre-norm) embedding norm the
//! output RMSNorm divides by.
//!
//!   cargo run --release --features cuda --example ae2_grad_balance -- s4

use std::range::Range;

use neural_networks::audio::frontend::LogMel;
use neural_networks::audio::io::Clip;
use neural_networks::audio::librispeech::LibriSpeech;
use neural_networks::audio::segments::{Bce, BoundaryTargets, WordSegments};
use neural_networks::config::{
    AUDIO_DATA, AUDIO_FEATS, AUDIO_SR, BOUNDARY_POS_WEIGHT, BOUNDARY_TOLERANCE,
};
use neural_networks::gpu::arena::TrainingCache;
use neural_networks::gpu::audio::AudioEncoder;
use neural_networks::gpu::{GTensor, Gpu, hierarchical::Hierarchical};
use neural_networks::segment::word_ends;
use neural_networks::tensor::Tensor;
use neural_networks::tokenizer_utf8::Utf8Tokenizer;

fn norm(gpu: &Gpu, t: &GTensor<f32>) -> f32 {
    t.to_host(gpu).data.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn main() {
    let name = std::env::args().nth(1).expect("model name");
    let path = format!("models/{name}");
    let gpu = Gpu::new().unwrap();
    let tok = Utf8Tokenizer::new();
    let mut model = Hierarchical::load(&gpu, &path, tok.w_token() as usize).unwrap();
    model.set_offload(&gpu, false);
    let (mut enc, ck) = AudioEncoder::load(&gpu, &format!("{path}.ae2")).unwrap();
    let c = enc.cfg;
    println!("{path}.ae2 at step {}", ck.step);
    let corpus = LibriSpeech::load(AUDIO_DATA, "dev-clean").unwrap();
    let bce = Bce {
        pos_weight: BOUNDARY_POS_WEIGHT,
    };

    let ids = ["8297-275156-0000", "8297-275156-0001", "84-121123-0001", "7976-110124-0000", "7976-110124-0001"];
    for id in ids {
        let u = corpus.utterances.iter().find(|u| u.id == id).unwrap();
        let raw = tok.to_tokens(&u.text());
        let mut words: Vec<Range<usize>> = Vec::new();
        let mut start = 0;
        for end in word_ends(&raw) {
            words.push(Range::from(start..end as usize));
            start = end as usize;
        }
        let tokens: Vec<usize> = raw.iter().map(|&t| t as usize).collect();
        let clip = Clip::read_at(&u.audio, AUDIO_SR).unwrap();
        let mut feats = LogMel::new().utterance(&clip.samples);
        ck.stats.apply(&mut feats);
        let t_in = feats.len() / AUDIO_FEATS;
        let t_out = c.stem().frames(t_in);
        let segs = WordSegments::from_word_times(&u.times(), t_out);
        let n = segs.len();
        if n != words.len() {
            continue;
        }
        let cache: TrainingCache = enc.training_cache(&gpu, t_out, t_out, t_out);
        let targets = BoundaryTargets::new(&segs, t_out, c.delay, BOUNDARY_TOLERANCE);
        let cols = c.delay + 1;

        let mut pulls = [0.0; 2];
        for (k, pull) in pulls.iter_mut().enumerate() {
            enc.zero_grad(&gpu);
            let out = enc.forward(&gpu, &feats, 1, t_in, std::slice::from_ref(&segs), &cache);
            let logits = out.boundary.to_host(&gpu).data;
            let (_, d_logits) = bce.loss_and_grad(&logits, &targets);
            let embeds = out.embeddings.as_ref().unwrap().to_host(&gpu).data;
            let hc = c.out;
            let mut d_e = vec![0.0; embeds.len()];
            if k == 1 {
                let given = GTensor::from_host(&gpu, &Tensor::new(&[n - 1, hc], embeds[..(n - 1) * hc].to_vec()));
                let mut d = GTensor::zeros(&gpu, &[n - 1, hc]);
                model.forward_backward_embeds(&gpu, &given, &tokens, &words, None, &mut d);
                d_e[..(n - 1) * hc].copy_from_slice(&d.to_host(&gpu).data);
            }
            let d_logits = if k == 0 { d_logits } else { vec![0.0; logits.len()] };
            let d_logits = GTensor::from_host(&gpu, &Tensor::new(&[t_out, cols], d_logits));
            let d_e = GTensor::from_host(&gpu, &Tensor::new(&[n, hc], d_e));
            enc.backward(&gpu, &d_logits, Some(&d_e), None, &cache);
            *pull = norm(&gpu, &enc.stem.dw);
        }
        enc.zero_grad(&gpu);
        println!(
            "{id}: stem |grad| from boundary {:.3e}, from next word {:.3e} (ratio {:.1}x)",
            pulls[0],
            pulls[1],
            pulls[1] / pulls[0]
        );
    }
}
