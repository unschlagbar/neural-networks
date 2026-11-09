//! Checks that word embeddings mean the same to the CPU and GPU models: the CPU
//! text encoder's e_w, fed to the GPU backbone as given embeddings, must score
//! like the GPU model reading the text itself.

use std::range::Range;

use neural_networks::audio::frontend::LogMel;
use neural_networks::audio::io::Clip;
use neural_networks::audio::librispeech::LibriSpeech;
use neural_networks::audio::segments::WordSegments;
use neural_networks::config::{AUDIO_DATA, AUDIO_FEATS, AUDIO_SR};
use neural_networks::gpu::audio::AudioEncoder;
use neural_networks::gpu::{GTensor, Gpu, hierarchical::Hierarchical as GpuHier};
use neural_networks::infer::Hierarchical;
use neural_networks::segment::word_ends;
use neural_networks::tensor::Tensor;
use neural_networks::tokenizer_utf8::Utf8Tokenizer;

fn main() {
    let name = std::env::args().nth(1).expect("model name");
    let path = format!("models/{name}");
    let text = "Long ago there lived a merchant who had three daughters and every year he went away to a distant city";
    let tok = Utf8Tokenizer::new();
    let raw = tok.to_tokens(text);
    let mut words: Vec<Range<usize>> = Vec::new();
    let mut start = 0;
    for end in word_ends(&raw) {
        words.push(Range::from(start..end as usize));
        start = end as usize;
    }
    let tokens: Vec<usize> = raw.iter().map(|&t| t as usize).collect();

    let gpu = Gpu::new().unwrap();
    let mut g = GpuHier::load(&gpu, &path, tok.w_token() as usize).unwrap();
    g.set_offload(&gpu, false);
    g.eval_loss(&gpu, &tokens, &words);
    let teacher = g.last_word_loss();

    let c = Hierarchical::load(&path).unwrap();
    let slices = |words: &[Range<usize>], raw: &[u16]| -> Vec<Vec<u16>> {
        words.iter().map(|w| raw[w.start..w.end].to_vec()).collect()
    };
    let encode = |words: &[Vec<u16>]| {
        let refs: Vec<&[u16]> = words.iter().map(|w| w.as_slice()).collect();
        c.encode(&refs)
    };
    let embeds = encode(&slices(&words[..words.len() - 1], &raw));
    let hc = embeds.len() / (words.len() - 1);
    let given = GTensor::from_host(&gpu, &Tensor::new(&[words.len() - 1, hc], embeds));
    g.eval_loss_embeds(&gpu, &given, &tokens, &words, None);
    println!("GPU reading text: {teacher:.4}   GPU reading the CPU encoder's e_w: {:.4}", g.last_word_loss());

    let corpus = LibriSpeech::load(AUDIO_DATA, "dev-clean").unwrap();
    let (mut enc, ck) = AudioEncoder::load(&gpu, &format!("{path}.ae2")).unwrap();
    for id in ["7976-110124-0000", "8297-275156-0001", "84-121123-0001"] {
        let u = corpus.utterances.iter().find(|u| u.id == id).unwrap();
        let text = u.text();
        let raw = tok.to_tokens(&text);
        let mut words: Vec<Range<usize>> = Vec::new();
        let mut start = 0;
        for end in word_ends(&raw) {
            words.push(Range::from(start..end as usize));
            start = end as usize;
        }
        let tokens: Vec<usize> = raw.iter().map(|&t| t as usize).collect();
        let n = words.len();
        let clip = Clip::read_at(&u.audio, AUDIO_SR).unwrap();
        let mut feats = LogMel::new().utterance(&clip.samples);
        ck.stats.apply(&mut feats);
        let t_in = feats.len() / AUDIO_FEATS;
        let t_out = enc.cfg.stem().frames(t_in);
        let segs = WordSegments::from_word_times(&u.times(), t_out);
        assert_eq!(segs.len(), n);
        let cache = enc.training_cache(&gpu, t_out, t_out, t_out);
        let out = enc.forward(&gpu, &feats, 1, t_in, std::slice::from_ref(&segs), &cache);
        let audio = out.embeddings.as_ref().unwrap().to_host(&gpu).data;
        enc.discard();

        let text_e = encode(&slices(&words, &raw));
        let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("\n{id}: {text}");
        for (i, w) in words.iter().enumerate() {
            let (a, t) = (&audio[i * hc..(i + 1) * hc], &text_e[i * hc..(i + 1) * hc]);
            let dot: f32 = a.iter().zip(t).map(|(x, y)| x * y).sum();
            println!(
                "  {:>14}  |audio| {:7.2}  |text| {:7.2}  cos {:+.3}",
                tok.to_text(&raw[w.start..w.end]).trim(),
                norm(a),
                norm(t),
                dot / (norm(a) * norm(t))
            );
        }
        g.eval_loss(&gpu, &tokens, &words);
        let teacher = g.last_word_loss();
        let given = GTensor::from_host(&gpu, &Tensor::new(&[n - 1, hc], audio[..(n - 1) * hc].to_vec()));
        g.eval_loss_embeds(&gpu, &given, &tokens, &words, None);
        println!("  GPU loss: text {teacher:.3}, audio {:.3}", g.last_word_loss());
    }
}

