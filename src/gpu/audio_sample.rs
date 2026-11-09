//! Mode `ae2s`: speak to the text model. An utterance goes through the audio
//! encoder of `<model>.ae2`, its word embeddings step the text model's backbone
//! in place of the text encoder's, and the model writes on from there. The text
//! model samples on the CPU, so this runs beside a training job.
//!
//! A LibriSpeech id runs three ways — MFA's word boundaries, boundaries aligned
//! to the transcript, the boundary head alone — next to the text model reading
//! the transcript itself. Any other WAV/FLAC runs on the boundary head, or
//! aligned to a transcript given after `|`.
//!
//! [`AudioEncoder::hear`] and the calls after it are the same path for any
//! other front end, e.g. a live microphone.

use std::io::{Write, stdin, stdout};
use std::path::PathBuf;

use super::Gpu;
use super::audio::AudioEncoder;
use super::audio_train::{BOUNDARY_THRESHOLD, PEAK_RADIUS};
use crate::audio::ctc::{Ctc, CtcAlphabet};
use crate::audio::frontend::{FeatureStats, LogMel};
use crate::audio::io::Clip;
use crate::audio::librispeech::LibriSpeech;
use crate::audio::segments::{CountAligner, DurationPrior, PeakDecoder, WordSegments};
use crate::config::{AUDIO_DATA, AUDIO_FEATS, AUDIO_SPLITS, AUDIO_SR, TEMPERATURE, TOP_P};
use crate::infer::Hierarchical;
use crate::sampling::Utf8Printer;
use crate::segment::word_ends;
use crate::tokenizer_utf8::Utf8Tokenizer;

/// Bytes generated per continuation.
const CONTINUATION: usize = 300;

/// One utterance through the encoder, before a word segmentation is chosen:
/// what the boundary and character heads made of it, and what
/// [`AudioEncoder::embed`] needs to run it again under a segmentation.
pub struct Heard {
    feats: Vec<f32>,
    t_in: usize,
    /// Encoder frames.
    pub t_out: usize,
    /// Boundary logits, `[t_out, delay + 1]`.
    pub boundary: Vec<f32>,
    /// The character head's greedy reading.
    pub ctc: String,
}

impl AudioEncoder {
    /// Runs `samples` (mono, `AUDIO_SR`) through the encoder under the feature
    /// normalization `stats` it was trained with. `None` when the audio is
    /// shorter than one encoder frame.
    pub fn hear(&mut self, gpu: &Gpu, stats: &FeatureStats, samples: &[f32]) -> Option<Heard> {
        let mut feats = LogMel::new().utterance(samples);
        stats.apply(&mut feats);
        let t_in = feats.len() / AUDIO_FEATS;
        let t_out = self.cfg.stem().frames(t_in);
        if t_out == 0 {
            return None;
        }
        let cache = self.training_cache(gpu, t_out, t_out, 0);
        let out = self.forward(gpu, &feats, 1, t_in, &[WordSegments::default()], &cache);
        let boundary = out.boundary.to_host(gpu).data;
        let chars = out.ctc.to_host(gpu).data;
        self.discard();
        let heard = Ctc {
            vocab: CtcAlphabet::SIZE,
        }
        .greedy(&chars);
        Some(Heard {
            feats,
            t_in,
            t_out,
            boundary,
            ctc: CtcAlphabet::decode(&heard),
        })
    }

    /// Words ending where the boundary head peaks.
    pub fn boundary_words(&self, heard: &Heard) -> WordSegments {
        let ends = PeakDecoder {
            delay: self.cfg.delay,
            threshold: BOUNDARY_THRESHOLD,
            radius: PEAK_RADIUS,
        }
        .decide(&heard.boundary, heard.t_out);
        WordSegments::from_ends(&ends)
    }

    /// The words of `text`, in the text model's segmentation, placed on the
    /// boundary logits. `None` when they do not fit.
    pub fn aligned_words(
        &self,
        heard: &Heard,
        text: &str,
        tok: &Utf8Tokenizer,
    ) -> Option<WordSegments> {
        let tokens = tok.to_tokens(text);
        let mut lens = Vec::new();
        let mut start = 0;
        for end in word_ends(&tokens) {
            lens.push(end as usize - start);
            start = end as usize;
        }
        let aligner = CountAligner {
            delay: self.cfg.delay,
            from_col: 0,
            off_peak: 3.0,
        };
        let prior = DurationPrior {
            sigma: 0.5,
            weight: 1.0,
            max_frames: 100,
        };
        aligner
            .align_words(&heard.boundary, heard.t_out, &lens, prior)
            .map(|ends| WordSegments::from_ends(&ends))
    }

    /// One word embedding per segment, `[segs.len(), out]` row-major: what
    /// [`Hierarchical::sample_from_embeddings`] reads.
    pub fn embed(&mut self, gpu: &Gpu, heard: &Heard, segs: &WordSegments) -> Vec<f32> {
        if segs.is_empty() {
            return Vec::new();
        }
        let cache = self.training_cache(gpu, heard.t_out, heard.t_out, segs.len());
        let out = self.forward(
            gpu,
            &heard.feats,
            1,
            heard.t_in,
            std::slice::from_ref(segs),
            &cache,
        );
        let embeds = out
            .embeddings
            .as_ref()
            .map(|e| e.to_host(gpu).data)
            .unwrap_or_default();
        self.discard();
        embeds
    }
}

pub fn sample_audio(model_path: &str) {
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No CUDA GPU available ({e}).");
            return;
        }
    };
    print!("Encoder checkpoint (empty = {model_path}.ae2): ");
    std::io::Write::flush(&mut std::io::stdout()).ok();
    let mut line = String::new();
    stdin().read_line(&mut line).ok();
    let path = match line.trim() {
        "" => format!("{model_path}.ae2"),
        p => p.to_string(),
    };
    let (mut enc, ck) = match AudioEncoder::load(&gpu, &path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Could not load {path}: {e}");
            return;
        }
    };
    let tok = Utf8Tokenizer::new();
    let mut model = match Hierarchical::load(model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Could not load the text model '{model_path}': {e}");
            return;
        }
    };
    let split = AUDIO_SPLITS[0];
    let corpus = LibriSpeech::load(AUDIO_DATA, split)
        .map(|c| c.utterances)
        .unwrap_or_default();
    println!(
        "{path}: step {}, {} {split} utterances to pick from",
        ck.step,
        corpus.len()
    );

    loop {
        println!(
            "\nAudio — a {split} id (e.g. {}), or a WAV/FLAC path, optionally `path | transcript` (Ctrl+D = quit):",
            corpus.first().map_or("1272-128104-0000", |u| u.id.as_str())
        );
        let mut line = String::new();
        if stdin().read_line(&mut line).unwrap_or(0) == 0 {
            println!();
            return;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let utterance = corpus.iter().find(|u| u.id == line);
        let (audio, transcript) = match utterance {
            Some(u) => (u.audio.clone(), Some(u.text())),
            None => match line.split_once('|') {
                Some((p, t)) => (PathBuf::from(p.trim()), Some(t.trim().to_string())),
                None => (PathBuf::from(line), None),
            },
        };
        let clip = match Clip::read_at(&audio, AUDIO_SR) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("{}: {e}", audio.display());
                continue;
            }
        };
        let Some(heard) = enc.hear(&gpu, &ck.stats, &clip.samples) else {
            eprintln!("shorter than one encoder frame");
            continue;
        };
        println!("character head hears: {}", heard.ctc);

        let mut runs: Vec<(&str, WordSegments)> = Vec::new();
        if let Some(u) = utterance {
            runs.push((
                "MFA boundaries",
                WordSegments::from_word_times(&u.times(), heard.t_out),
            ));
        }
        if let Some(segs) = transcript
            .as_deref()
            .and_then(|text| enc.aligned_words(&heard, text, &tok))
        {
            runs.push(("aligned to the transcript", segs));
        }
        runs.push(("boundary head alone", enc.boundary_words(&heard)));

        if let Some(text) = &transcript {
            println!("transcript: {text}");
            print!("  text model reading it   → ");
            stdout().flush().ok();
            let mut printer = Utf8Printer::default();
            model.sample(
                &tok.to_tokens(&format!("{text} ")),
                CONTINUATION,
                TEMPERATURE,
                TOP_P,
                |t| printer.print(t, &tok),
            );
            println!();
        }
        for (name, segs) in runs {
            if segs.is_empty() {
                continue;
            }
            let embeds = enc.embed(&gpu, &heard, &segs);
            print!("  audio, {name} ({} words) → ", segs.len());
            stdout().flush().ok();
            let mut printer = Utf8Printer::default();
            model.sample_from_embeddings(&embeds, CONTINUATION, TEMPERATURE, TOP_P, |t| {
                printer.print(t, &tok)
            });
            println!();
        }
    }
}
