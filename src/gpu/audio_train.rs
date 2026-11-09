//! Mode `ae2`: the audio encoder trained against a frozen text model (stage E2).
//!
//! The encoder learns word embeddings from which the pretrained `Hierarchical`
//! predicts the next word as well as it does from the transcript's own, and a
//! boundary head that finds the words. The text model is only read, never
//! stepped. Data is LibriSpeech with MFA word alignments (`audio::librispeech`).
//!
//! Each utterance is one text-model window: optionally some of the chapter's
//! transcript before it, its own words as audio embeddings, then the
//! transcript that follows it, so the audio has to leave the backbone where
//! the text can continue. The decoder's target mixes the transcript with the
//! text model's own distribution reading the whole window as text
//! (distillation, as in DiVA, arXiv 2410.02678).
//!
//! Logged: the student's next-word loss (nats per word, reading audio) against
//! the teacher's (the same model reading the transcript) — the gap is what the
//! audio path still loses — on the utterance's words and on the text after it,
//! plus the boundary head's loss and F1 and the character head. The encoder is
//! saved next to the text model as `<model>.ae2`, and a run resumes from it.

use std::collections::HashMap;
use std::path::PathBuf;
use std::range::Range;
use std::time::Instant;

use rand::random_range;
use rand::seq::SliceRandom;
use rayon::prelude::*;

use super::arena::TrainingCache;
use super::audio::{AudioCfg, AudioCheckpoint, AudioEncoder};
use super::hierarchical::Hierarchical;
use super::{GTensor, Gpu};
use crate::audio::augment::{Augment, AugmentCfg, NoiseBank};
use crate::audio::ctc::{Ctc, CtcAlphabet, edit_distance};
use crate::audio::frontend::{FeatureStats, LogMel, StatsAccumulator};
use crate::audio::io::Clip;
use crate::audio::librispeech::{LibriSpeech, Utterance};
use crate::audio::voice;
use crate::audio::segments::{
    Bce, BoundaryTargets, CountAligner, CountLoss, Segment, DurationPrior, ENC_FRAME_S, PeakDecoder, WordSegments,
};
use crate::config::{
    AUDIO_ALIGN_ONLINE, AUDIO_HEAD_CUT_P, AUDIO_TAKE_SET_POWER, AUDIO_AUGMENT, AUDIO_SEG_MAX_FRAMES, AUDIO_BATCH, AUDIO_CONTEXT_WORDS, AUDIO_DATA, AUDIO_E2_STEPS, AUDIO_FEATS,
    AUDIO_HELD_OUT_SPEAKERS, AUDIO_MAX_WORDS, AUDIO_BABBLE_SPLIT, AUDIO_E2_FROM, AUDIO_MAX_TAKE_SECONDS, AUDIO_REALIGN_MAX_CER, AUDIO_PRETRAIN_DISTILL, AUDIO_PRETRAIN_EVAL_EVERY, AUDIO_TAKE_SHARE, AUDIO_PRETRAIN_TTS_P, AUDIO_PRETRAIN_VOICE_P, AUDIO_HOP, AUDIO_LR, AUDIO_NOISE_DIR, AUDIO_PREFIX_P, AUDIO_SPLITS, AUDIO_TRAIN_AFTER_WORDS, AUDIO_TAKE_DIRS, AUDIO_VOICE_DIR, AUDIO_VOICE_DOCS, AUDIO_VOICE_EPOCHS,
    AUDIO_VOICE_LAST, AUDIO_VOICE_VAL_SESSIONS,
    AUDIO_SR, AUDIO_VAL_EVERY, AUDIO_WARMUP_STEPS, BOUNDARY_COUNT_WEIGHT, BOUNDARY_POS_WEIGHT, BOUNDARY_TOLERANCE,
    CTC_WEIGHT, DISTILL_ALPHA, DISTILL_ALPHA_PLAIN, EMBED_ALIGN_WEIGHT, EMBED_CONTRAST_TEMP, EMBED_CONTRAST_WEIGHT, STATE_DISTILL_BETA, MAX_SEQ_LEN, AUDIO_PRETRAIN_EVERY, AUDIO_PRETRAIN_HEARD,
    AUDIO_PRETRAIN_LR, AUDIO_PRETRAIN_START, AUDIO_PRETRAIN_WORDS, SFT_AUDIO_DISTILL, SFT_AUDIO_LR, SFT_AUDIO_START,
    SFT_SPEECH_DIRS,
};
use crate::nn2::optim::AdamCfg;
use crate::segment::word_ends;
use crate::tensor::Tensor;
use crate::tokenizer_utf8::Utf8Tokenizer;

const LOG_EVERY: usize = 20;
/// Words per second past which a transcript cannot be its audio's.
const MAX_WORDS_PER_SECOND: f32 = 6.0;
/// Consecutive takes of a set drawn as a run (see [`Epochs::blocked`]): the
/// large sets live on a spinning disk, where a random take costs a seek.
const TAKE_BLOCK: usize = 32;
/// Training batches loading at once.
const LOAD_AHEAD: usize = 4;
/// Words of the training windows the text model reads in one packed sequence:
/// the backbone's resident frames grow to the longest it has seen.
const PACK_WORDS: usize = 256;
/// Held-out TTS takes `ae2` validates on.
const TTS_VAL: usize = 256;
/// Positions after an utterance `ae2e`'s context report scores.
const CONTEXT_AT: [usize; 5] = [1, 2, 4, 8, 16];
/// Training utterances the feature statistics are measured on.
const STATS_UTTERANCES: usize = 500;
/// Transcripts the output scale is measured on.
const SCALE_UTTERANCES: usize = 300;
/// Training utterances `ae2a` scores beside the held-out ones.
const EVAL_TRAIN_SAMPLE: usize = 500;
/// Boundary probability at which the decoder commits to a word end.
pub(crate) const BOUNDARY_THRESHOLD: f32 = 0.5;
/// Frames on each side a word end must top; best F1 on held-out speakers.
pub(crate) const PEAK_RADIUS: usize = 1;

pub fn train_audio_e2(model_path: &str) {
    run_audio_e2(model_path, None);
}

/// Mode `ae2e`: the validation lines of an audio encoder checkpoint, never a
/// training step. `encoder` is its file; empty means `<model>.ae2`.
pub fn eval_audio_e2(model_path: &str) {
    print!("Encoder checkpoint (empty = {model_path}.ae2): ");
    std::io::Write::flush(&mut std::io::stdout()).ok();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).ok();
    let path = match line.trim() {
        "" => format!("{model_path}.ae2"),
        p => p.to_string(),
    };
    run_audio_e2(model_path, Some(&path));
}

/// Trains `<model>.ae2`, or with `evaluate` only validates that checkpoint.
fn run_audio_e2(model_path: &str, evaluate: Option<&str>) {
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No CUDA GPU available ({e}).");
            return;
        }
    };
    let tok = Utf8Tokenizer::new();
    let mut model = match Hierarchical::load(&gpu, model_path, tok.w_token() as usize) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Could not load the text model '{model_path}': {e}");
            return;
        }
    };
    // Every window is read back by its own backward right away; parking the
    // backbone's activations in host memory would only cost the copies.
    model.set_offload(&gpu, false);
    // Its windows are an utterance and some text (packs of at most
    // `PACK_WORDS` in training), never near the pretraining window the
    // backbone's resident frames are sized for; an evaluation goes smaller
    // still, to share the device with a training run.
    model.set_bb_chunk(Some(if evaluate.is_some() { 128 } else { PACK_WORDS }));

    let save_path = evaluate.map_or_else(|| format!("{model_path}.ae2"), str::to_string);
    let mut resumed = AudioEncoder::load(&gpu, &save_path).ok().or_else(|| {
        let from = (evaluate.is_none() && !AUDIO_E2_FROM.is_empty()).then_some(AUDIO_E2_FROM)?;
        let (enc, mut ck) = AudioEncoder::load(&gpu, from).ok()?;
        println!("Starting {save_path} from {from} (its step {}), at step 0.", ck.step);
        ck.step = 0;
        Some((enc, ck))
    });
    if evaluate.is_some() && resumed.is_none() {
        eprintln!("Could not load {save_path}.");
        return;
    }
    // How many frames of a word the segment encoder reads changes no weight, so
    // a training run takes it from the config rather than from the checkpoint.
    if let Some((enc, _)) = resumed.as_mut().filter(|_| evaluate.is_none()) {
        if enc.cfg.seg_max != AUDIO_SEG_MAX_FRAMES {
            println!("segment encoder reads {} frames per word, was {}", AUDIO_SEG_MAX_FRAMES, enc.cfg.seg_max);
            enc.cfg.seg_max = AUDIO_SEG_MAX_FRAMES;
        }
    }
    let cfg = resumed
        .as_ref()
        .map_or_else(|| AudioCfg::from_config(model.cfg.hc), |(e, _)| e.cfg);
    assert_eq!(
        cfg.out, model.cfg.hc,
        "{save_path} was trained against a different text model width"
    );

    let t0 = Instant::now();
    let mut corpus = Corpus::load(&tok, cfg.stride);
    println!(
        "{} of {} utterances map one to one ({:.1?})",
        corpus.items.len(),
        corpus.total,
        t0.elapsed()
    );
    let (mut enc, mut ck) = match resumed {
        Some((enc, ck)) => {
            println!(
                "Resuming {save_path} at step {} ({:.1} h of audio heard).",
                ck.step,
                ck.seconds / 3600.0
            );
            if !enc.missing.is_empty() {
                println!(
                    "  not in the checkpoint, starting fresh: {}",
                    enc.missing.join(", ")
                );
            }
            (enc, ck)
        }
        None => {
            println!("No {save_path} yet — starting a new audio encoder.");
            let ck = AudioCheckpoint {
                step: 0,
                seconds: 0.0,
                stats: corpus.train_stats(),
            };
            let mut enc = AudioEncoder::new(&gpu, cfg);
            enc.missing = vec!["output scale"];
            (enc, ck)
        }
    };
    corpus.stats = ck.stats.clone();
    if enc.missing.contains(&"output scale") {
        let scale = corpus.text_scale(model_path);
        let norm = scale.iter().map(|s| s * s).sum::<f32>().sqrt();
        println!("output scale from the text encoder: {norm:.2} per embedding");
        enc.out_norm.gamma = GTensor::from_host(&gpu, &Tensor::new(&[scale.len()], scale));
    }

    let cache = corpus.training_cache(&gpu, &enc);
    let mut trainer = E2Trainer {
        gpu,
        model,
        enc,
        cache,
        bce: Bce {
            pos_weight: BOUNDARY_POS_WEIGHT,
        },
        teacher: vec![None; corpus.items.len()],
        state_mse: 0.0,
        state_n: 0,
        embed_mse: 0.0,
        embed_n: 0,
    };
    corpus.report(&mut trainer);
    if evaluate.is_some() || ck.step as usize >= AUDIO_E2_STEPS {
        let val: Vec<usize> = corpus.val().iter().map(|p| p.index).collect();
        println!("  val       | {}", trainer.validate(&corpus, &val).summary());
        let voice_val: Vec<usize> = corpus.voice(true).iter().map(|p| p.index).collect();
        if !voice_val.is_empty() {
            println!("  val voice | {}", trainer.validate(&corpus, &voice_val).summary());
        }
        // The takes it trained on, scored the same way: against `val voice`,
        // how much of the voice it learned and how much only the sentences.
        let voice_train: Vec<usize> = corpus.voice(false).iter().map(|p| p.index).collect();
        if evaluate.is_some() && !voice_train.is_empty() {
            println!("  voice trained | {}", trainer.validate(&corpus, &voice_train).summary());
        }
        // Each take set on its own held-out takes (`tts_held_out`).
        if evaluate.is_some() {
            for (dir, range) in AUDIO_TAKE_DIRS.iter().zip(&corpus.take_sets) {
                let held: Vec<usize> = corpus.items[range.clone()]
                    .iter()
                    .filter(|p| tts_held_out(p))
                    .take(TTS_VAL)
                    .map(|p| p.index)
                    .collect();
                if !held.is_empty() {
                    let name = dir.rsplit('/').next().unwrap_or(dir);
                    println!("  val {name} ({}) | {}", held.len(), trainer.validate(&corpus, &held).summary());
                }
            }
        }
        if evaluate.is_some() && !voice_train.is_empty() {
            println!("  the user's words heard wrong (all voice takes):");
            let mut voice_all = voice_train.clone();
            voice_all.extend(&voice_val);
            if let Some(encoder) = evaluate {
                trainer.misheard_report(&corpus, &voice_all, &format!("{encoder}.misheard.tsv"));
            }
        }
        if evaluate.is_some() {
            println!("  LibriSpeech val, words cut three ways:");
            trainer.segmentation_report(&corpus, &val);
            println!("  LibriSpeech val:");
            trainer.context_report(&corpus, &val);
            println!("  voice, held out:");
            trainer.context_report(&corpus, &voice_val);
        }
        if evaluate.is_none() {
            println!(
                "Already at step {} of AUDIO_E2_STEPS = {AUDIO_E2_STEPS}; raise it to train on.",
                ck.step
            );
        }
        return;
    }
    trainer.run(&corpus, &mut ck, &save_path);
}

/// The audio stage (mode `hga`), once text pretraining is through: every
/// `AUDIO_PRETRAIN_EVERY` text windows the pretraining loop runs one more, a run of consecutive LibriSpeech utterances read as
/// a document of written text in which `AUDIO_PRETRAIN_HEARD` of them are heard
/// instead of read — their backbone rows come from the audio encoder. Backbone,
/// decoder and text encoder accumulate into the same step as the text windows;
/// `AUDIO_PRETRAIN_VOICE_P` of the windows are instead one of the pretraining
/// documents the voice takes were read out of, with those takes heard where
/// their sentences stand.
/// With `AUDIO_PRETRAIN_DISTILL` the window first runs read, every word as
/// text, and the heard run trains towards that reading's predictions.
/// The audio encoder steps on its own after each window, with its character
/// and boundary heads alongside. The backbone's carried state is stashed
/// around the window, so the text stream runs exactly as it would without it.
pub struct AudioPretrain {
    enc: AudioEncoder,
    ck: AudioCheckpoint,
    corpus: Corpus,
    train: Vec<usize>,
    voice: DocSource,
    tts: DocSource,
    /// Fixed windows nothing trains on: held-out LibriSpeech speakers, the
    /// held-out voice sessions, held-out TTS questions.
    val: Vec<(&'static str, Vec<(Vec<usize>, Document)>)>,
    cache: TrainingCache,
    bce: Bce,
    adam: AdamCfg,
    save_path: String,
    loss: f32,
    read_loss: f32,
    windows: usize,
    voice_windows: usize,
    tts_windows: usize,
    ctc: CtcScore,
}

impl AudioPretrain {
    /// The encoder at `<model>.ae2`, or a fresh copy of `AUDIO_PRETRAIN_START`
    /// saved there; `None` (with a note) when there is neither or audio is off.
    pub fn open(gpu: &Gpu, model_path: &str, model_hc: usize) -> Option<Self> {
        if AUDIO_PRETRAIN_EVERY == 0 {
            return None;
        }
        let save_path = format!("{model_path}.ae2");
        let from = if std::path::Path::new(&save_path).is_file() {
            save_path.clone()
        } else {
            AUDIO_PRETRAIN_START.to_string()
        };
        let (enc, ck) = match AudioEncoder::load(gpu, &from) {
            Ok(v) => v,
            Err(e) => {
                println!("audio windows off: no encoder at {from} ({e})");
                return None;
            }
        };
        if enc.cfg.out != model_hc {
            println!("audio windows off: {from} was trained against another text model width");
            return None;
        }
        let mut corpus = Corpus::load(&Utf8Tokenizer::new(), enc.cfg.stride);
        corpus.stats = ck.stats.clone();
        let train: Vec<usize> = corpus.train().iter().map(|p| p.index).collect();
        let voice = DocSource::new(&corpus, corpus.voice(false).iter().map(|p| p.index));
        let tts = DocSource::new(&corpus, corpus.tts().iter().filter(|p| !tts_held_out(p)).map(|p| p.index));
        let cache = Self::cache(gpu, &enc, &corpus);
        println!(
            "audio windows: every {AUDIO_PRETRAIN_EVERY} text windows, {} utterances to draw from, encoder {from} (step {})",
            train.len(),
            ck.step
        );
        println!("  {:.0}% of them voice: {}", AUDIO_PRETRAIN_VOICE_P * 100.0, voice.summary(&corpus));
        println!("  {:.0}% of them TTS questions: {}", AUDIO_PRETRAIN_TTS_P * 100.0, tts.summary(&corpus));
        let mut a = Self {
            enc,
            ck,
            corpus,
            train,
            voice,
            tts,
            val: Vec::new(),
            cache,
            bce: Bce {
                pos_weight: BOUNDARY_POS_WEIGHT,
            },
            adam: AdamCfg::new(AUDIO_PRETRAIN_LR, 0.0),
            save_path,
            loss: 0.0,
            read_loss: 0.0,
            windows: 0,
            voice_windows: 0,
            tts_windows: 0,
            ctc: CtcScore::default(),
        };
        a.val = a.val_windows();
        println!(
            "  held-out windows: {}",
            a.val.iter().map(|(n, w)| format!("{} {n}", w.len())).collect::<Vec<_>>().join(", ")
        );
        Some(a)
    }

    fn cache(gpu: &Gpu, enc: &AudioEncoder, corpus: &Corpus) -> TrainingCache {
        let c = enc.cfg;
        let t_out = corpus.items.iter().map(|p| p.t_in / c.stride).max().unwrap_or(1);
        let words = corpus.items.iter().map(|p| p.segments.len()).max().unwrap_or(1);
        let t_out = slowed(t_out);
        // A window's heard utterances run as one row.
        let n = AUDIO_PRETRAIN_HEARD.max(corpus.voice_docs.iter().map(|d| d.takes.len()).max().unwrap_or(0));
        enc.training_cache(gpu, t_out * n, t_out * n, words * n)
    }

    /// Whether text window number `n` (counted from 1) is followed by an audio window.
    pub fn due(n: usize) -> bool {
        AUDIO_PRETRAIN_EVERY > 0 && n % AUDIO_PRETRAIN_EVERY == 0
    }

    /// Consecutive utterances from a random one, up to `AUDIO_PRETRAIN_WORDS`.
    fn draw_run(&self) -> Vec<usize> {
        self.run_from(self.train[random_range(0..self.train.len())])
    }

    /// Consecutive utterances from `start`, up to `AUDIO_PRETRAIN_WORDS`.
    fn run_from(&self, start: usize) -> Vec<usize> {
        let mut run = vec![start];
        let mut words = self.corpus.items[run[0]].written_words.len();
        while let Some(next) = self.corpus.next.get(*run.last().unwrap()).copied().flatten() {
            words += self.corpus.items[next].written_words.len();
            if words > AUDIO_PRETRAIN_WORDS {
                break;
            }
            run.push(next);
        }
        run
    }

    /// A LibriSpeech run as a document: the items it hears, in order, and the
    /// document.
    fn librispeech_window(&self) -> Option<(Vec<usize>, Document)> {
        let run = self.draw_run();
        if run.len() < 2 {
            return None;
        }
        let mut heard: Vec<usize> = (0..run.len()).collect();
        heard.shuffle(&mut rand::rng());
        heard.truncate(AUDIO_PRETRAIN_HEARD.min(run.len() - 1));
        heard.sort();
        let items: Vec<&Prepared> = run.iter().map(|&i| &self.corpus.items[i]).collect();
        let doc = Document::new(&items, &heard)?;
        Some((heard.iter().map(|&k| run[k]).collect(), doc))
    }

    /// A random document of `src` cut to `AUDIO_PRETRAIN_WORDS` words around
    /// one of its training takes, every training take inside the cut heard.
    fn voice_window(&self, src: &DocSource) -> Option<(Vec<usize>, Document)> {
        let d = src.docs[random_range(0..src.docs.len())];
        let n = self.corpus.voice_docs[d].takes.iter().filter(|(i, _)| src.train.contains(i)).count();
        self.cut(d, &src.train, random_range(0..n), random_range(0.0..=1.0))
    }

    /// Voice document `d` cut to `AUDIO_PRETRAIN_WORDS` words around its
    /// `anchor`-th take from `takes`, `place` (0 to 1) of the room left over
    /// before the take; every take from `takes` inside the cut heard.
    fn cut(
        &self,
        d: usize,
        takes: &std::collections::HashSet<usize>,
        anchor: usize,
        place: f32,
    ) -> Option<(Vec<usize>, Document)> {
        let doc = &self.corpus.voice_docs[d];
        let takes: Vec<&(usize, Range<usize>)> = doc.takes.iter().filter(|(i, _)| takes.contains(i)).collect();
        let anchor = takes.get(anchor)?.1;
        let bytes: Vec<u16> = doc.text.bytes().map(u16::from).collect();
        let words = split_words(&bytes);
        let first = words.iter().position(|w| w.end > anchor.start)?;
        let last = words.iter().rposition(|w| w.start < anchor.end)?;
        let budget = AUDIO_PRETRAIN_WORDS;
        let lo = if words.len() <= budget {
            0
        } else {
            let slack = budget.saturating_sub(last + 1 - first).min(first);
            let lo = first - (place * slack as f32).round() as usize;
            lo.min(words.len() - budget)
        };
        let hi = (lo + budget).min(words.len());
        let span = words[lo].start..words[hi - 1].end;
        let mut inside: Vec<(usize, Range<usize>)> = Vec::new();
        for &&(i, r) in &takes {
            let after_last = inside.last().is_none_or(|(_, l)| r.start >= l.end);
            if r.start >= span.start && r.end <= span.end && after_last {
                inside.push((i, r));
            }
        }
        let pieces: Vec<(&Prepared, Range<usize>)> =
            inside.iter().map(|&(i, r)| (&self.corpus.items[i], r)).collect();
        let doc = Document::voice(&doc.text, span, &pieces)?;
        Some((inside.iter().map(|&(i, _)| i).collect(), doc))
    }

    /// One audio window into `model`'s accumulating step, and a step of the
    /// encoder.
    pub fn window(&mut self, gpu: &Gpu, model: &mut Hierarchical) {
        if self.adam.t % AUDIO_PRETRAIN_EVAL_EVERY as u64 == 0 {
            println!("{}", self.evaluate(gpu, model));
        }
        let r = random_range(0.0..1.0);
        let voice = !self.voice.docs.is_empty() && r < AUDIO_PRETRAIN_VOICE_P;
        let tts = !voice && !self.tts.docs.is_empty() && r < AUDIO_PRETRAIN_VOICE_P + AUDIO_PRETRAIN_TTS_P;
        let drawn = if voice {
            self.voice_window(&self.voice)
        } else if tts {
            self.voice_window(&self.tts)
        } else {
            self.librispeech_window()
        };
        let Some((heard, doc)) = drawn.filter(|(heard, _)| !heard.is_empty()) else {
            return;
        };
        let c = self.enc.cfg;
        let hc = c.out;
        let heard_items: Vec<&Prepared> = heard.iter().map(|&i| &self.corpus.items[i]).collect();
        let augment = self.corpus.augment.as_ref();
        // One fresh encoder state for the whole window, carried through every
        // utterance it hears.
        let batch = Batch::chained(heard_items, &self.corpus.stats, augment, c.stride);
        let out = self.enc.forward(
            gpu,
            &batch.feats,
            batch.rows(),
            batch.t_in,
            &batch.segments(),
            &self.cache,
        );
        let logits = out.boundary.to_host(gpu).data;
        let (_, d_logits) = self
            .bce
            .loss_and_grad(&logits, &batch.targets(out.t_out, c.stride, c.delay));
        let ctc_logits = out.ctc.to_host(gpu).data;
        let (ctc, d_ctc) = ctc_batch(&ctc_logits, &batch, out.t_out, c);
        let embeds = out
            .embeddings
            .as_ref()
            .expect("every utterance has words")
            .to_host(gpu)
            .data;

        // Rows of the window the backbone reads from the audio, and which
        // encoder embedding each is.
        let dw = doc.words.len() - 1;
        let (rows, from): (Vec<usize>, Vec<usize>) =
            doc.heard.iter().copied().filter(|&(r, _)| r < dw).unzip();
        let given: Vec<f32> = from
            .iter()
            .flat_map(|&e| embeds[e * hc..(e + 1) * hc].iter().copied())
            .collect();
        let given = GTensor::from_host(gpu, &Tensor::new(&[rows.len(), hc], given));
        let mut d = GTensor::zeros(gpu, &[rows.len(), hc]);

        let carried = model.stash_carried(gpu);
        // The same tokens and words read: the teacher, and the loss hearing is
        // measured against. Each run starts from zero state.
        if AUDIO_PRETRAIN_DISTILL > 0.0 {
            model.record_distill();
        }
        model.set_continues(false);
        model.set_doc_starts(&[]);
        model.eval_loss_masked(gpu, &doc.tokens, &doc.words, None);
        self.read_loss += model.last_word_loss();
        if AUDIO_PRETRAIN_DISTILL > 0.0 {
            let targets = model.take_distill();
            model.apply_distill(targets, AUDIO_PRETRAIN_DISTILL);
        }
        model.set_continues(false);
        model.set_doc_starts(&[]);
        let _ = model.forward_backward_mixed(gpu, &doc.tokens, &doc.words, &given, &rows, &mut d);
        model.end_distill();
        model.restore_carried(carried);
        self.loss += model.last_word_loss();
        self.windows += 1;
        self.voice_windows += voice as usize;
        self.tts_windows += tts as usize;
        if self.adam.t == 0 {
            println!(
                "  first audio window: {} words, {} of them heard, word loss {:.3}",
                doc.words.len(),
                rows.len(),
                model.last_word_loss()
            );
        }
        self.ctc.merge(&ctc);

        let d = d.to_host(gpu).data;
        let mut d_e = vec![0.0; embeds.len()];
        for (i, &e) in from.iter().enumerate() {
            d_e[e * hc..(e + 1) * hc].copy_from_slice(&d[i * hc..(i + 1) * hc]);
        }
        let cols = c.delay + 1;
        self.enc.backward(
            gpu,
            &GTensor::from_host(gpu, &Tensor::new(&[logits.len() / cols, cols], d_logits)),
            Some(&GTensor::from_host(gpu, &Tensor::new(&[embeds.len() / hc, hc], d_e))),
            Some((
                &GTensor::from_host(gpu, &Tensor::new(&[ctc_logits.len() / c.ctc, c.ctc], d_ctc)),
                CTC_WEIGHT,
            )),
            &self.cache,
        );
        self.adam.t += 1;
        self.enc.step(gpu, &self.adam);
        self.ck.step += 1;
        self.ck.seconds += batch.seconds();
    }

    /// The held-out windows: from every `k`-th validation utterance a run with
    /// its second, fourth and sixth utterance heard; every held-out voice and
    /// TTS document around its first take.
    fn val_windows(&self) -> Vec<(&'static str, Vec<(Vec<usize>, Document)>)> {
        const PER_SOURCE: usize = 32;
        let val = self.corpus.val();
        let step = (val.len() / PER_SOURCE).max(1);
        let libri = val
            .iter()
            .step_by(step)
            .filter_map(|p| {
                let run = self.run_from(p.index);
                let heard: Vec<usize> = (1..run.len()).step_by(2).take(AUDIO_PRETRAIN_HEARD).collect();
                let items: Vec<&Prepared> = run.iter().map(|&i| &self.corpus.items[i]).collect();
                let doc = Document::new(&items, &heard)?;
                Some((heard.iter().map(|&k| run[k]).collect::<Vec<_>>(), doc))
            })
            .filter(|(heard, _)| !heard.is_empty())
            .take(PER_SOURCE)
            .collect();
        let docs = |held: std::collections::HashSet<usize>| -> Vec<(Vec<usize>, Document)> {
            let src = DocSource::new(&self.corpus, held.into_iter());
            let step = (src.docs.len() / PER_SOURCE).max(1);
            src.docs
                .iter()
                .step_by(step)
                .filter_map(|&d| self.cut(d, &src.train, 0, 0.5))
                .filter(|(heard, _)| !heard.is_empty())
                .take(PER_SOURCE)
                .collect()
        };
        let voice = docs(self.corpus.voice(true).iter().map(|p| p.index).collect());
        let tts = docs(self.corpus.tts().iter().filter(|p| tts_held_out(p)).map(|p| p.index).collect());
        vec![("librispeech", libri), ("voice", voice), ("tts", tts)]
    }

    /// Word loss on the held-out windows heard and read, per source, over every
    /// word from the first heard one to the end of the window: the heard words
    /// and all the text that follows them. Forward only, no augmentation; the
    /// backbone's carried state is left as it was.
    fn evaluate(&mut self, gpu: &Gpu, model: &mut Hierarchical) -> String {
        let c = self.enc.cfg;
        let hc = c.out;
        let carried = model.stash_carried(gpu);
        let mut parts = Vec::new();
        for (name, windows) in &self.val {
            if windows.is_empty() {
                continue;
            }
            let (mut heard_loss, mut read_loss) = (0.0, 0.0);
            for (heard, doc) in windows {
                let items: Vec<&Prepared> = heard.iter().map(|&i| &self.corpus.items[i]).collect();
                let batch = Batch::chained(items, &self.corpus.stats, None, c.stride);
                let out = self.enc.forward(gpu, &batch.feats, batch.rows(), batch.t_in, &batch.segments(), &self.cache);
                let embeds = out.embeddings.as_ref().expect("every utterance has words").to_host(gpu).data;
                self.enc.discard();
                let dw = doc.words.len() - 1;
                let (rows, from): (Vec<usize>, Vec<usize>) =
                    doc.heard.iter().copied().filter(|&(r, _)| r < dw).unzip();
                let given: Vec<f32> = from.iter().flat_map(|&e| embeds[e * hc..(e + 1) * hc].iter().copied()).collect();
                let given = GTensor::from_host(gpu, &Tensor::new(&[rows.len(), hc], given));
                // Decoded word `i` is window word `i + 1`.
                let first = rows.first().copied().unwrap_or(0);
                let mask: Vec<bool> = (0..dw).map(|i| i + 1 >= first).collect();
                model.set_continues(false);
                model.set_doc_starts(&[]);
                model.eval_loss_masked(gpu, &doc.tokens, &doc.words, Some(&mask));
                read_loss += model.last_word_loss();
                model.set_continues(false);
                model.set_doc_starts(&[]);
                model.eval_loss_mixed(gpu, &doc.tokens, &doc.words, &given, &rows, Some(&mask));
                heard_loss += model.last_word_loss();
            }
            let n = windows.len() as f32;
            parts.push(format!(
                "{name} heard {:.3} read {:.3} gap {:.3}",
                heard_loss / n,
                read_loss / n,
                (heard_loss - read_loss) / n
            ));
        }
        model.restore_carried(carried);
        format!("  held out | {}", parts.join(" | "))
    }

    /// The audio windows since the last report, or `None` if there were none.
    pub fn report(&mut self) -> Option<String> {
        if self.windows == 0 {
            return None;
        }
        let line = format!(
            "  audio windows {} ({} voice, {} tts) | word loss heard {:.3} read {:.3} | {}",
            self.windows,
            self.voice_windows,
            self.tts_windows,
            self.loss / self.windows as f32,
            self.read_loss / self.windows as f32,
            self.ctc.summary()
        );
        (self.loss, self.read_loss, self.windows, self.voice_windows, self.tts_windows, self.ctc) =
            (0.0, 0.0, 0, 0, 0, CtcScore::default());
        Some(line)
    }

    pub fn save(&mut self, gpu: &Gpu) {
        if let Err(e) = self.enc.save(gpu, &self.save_path, &self.ck) {
            eprintln!("saving {} failed: {e}", self.save_path);
        }
    }
}

/// The heard user turns of audio SFT (mode `hqg`): each spoken turn of an
/// example runs through the encoder as its own utterance, half of them pooled
/// as the boundary head cuts them; the backbone reads those words from the
/// audio and trains on the reply after them, at `SFT_AUDIO_DISTILL` towards
/// what it predicts after reading the same words. The encoder steps after
/// every heard example, CTC and boundary heads alongside.
pub struct SftAudio {
    enc: AudioEncoder,
    ck: AudioCheckpoint,
    corpus: Corpus,
    by_id: HashMap<String, usize>,
    cache: Option<TrainingCache>,
    /// Heard turns per example `cache` holds.
    reserved: usize,
    bce: Bce,
    adam: AdamCfg,
    save_path: String,
    loss: f32,
    read_loss: f32,
    examples: usize,
    unmatched: usize,
    head_cut: (usize, usize),
    ctc: CtcScore,
}

impl SftAudio {
    /// The encoder at `<model>.ae2`, or a fresh copy of `SFT_AUDIO_START`
    /// saved there, and the takes of `SFT_SPEECH_DIRS`; `None` (with a note)
    /// when there is no encoder or no take.
    pub fn open(gpu: &Gpu, model_path: &str, model_hc: usize) -> Option<Self> {
        let save_path = format!("{model_path}.ae2");
        let from = if std::path::Path::new(&save_path).is_file() {
            save_path.clone()
        } else {
            SFT_AUDIO_START.to_string()
        };
        let (enc, ck) = match AudioEncoder::load(gpu, &from) {
            Ok(v) => v,
            Err(e) => {
                println!("heard turns off: no encoder at {from} ({e})");
                return None;
            }
        };
        if enc.cfg.out != model_hc {
            println!("heard turns off: {from} was trained against another text model width");
            return None;
        }
        let mut corpus = Corpus::takes(&Utf8Tokenizer::new(), enc.cfg.stride, SFT_SPEECH_DIRS);
        corpus.stats = ck.stats.clone();
        if corpus.items.is_empty() {
            println!("heard turns off: no aligned take in {SFT_SPEECH_DIRS:?}");
            return None;
        }
        // A take's id is `<speaker dir>/<session>/<id>`; the corpus names it
        // without the speaker dir.
        let by_id = corpus
            .items
            .iter()
            .map(|p| (p.id.split_once('/').map_or(p.id.as_str(), |(_, id)| id).to_string(), p.index))
            .collect();
        println!(
            "heard turns: {} takes ({:.0} h), encoder {from} (step {})",
            corpus.items.len(),
            corpus.items.iter().map(|p| p.seconds()).sum::<f64>() / 3600.0,
            ck.step
        );
        Some(Self {
            enc,
            ck,
            corpus,
            by_id,
            cache: None,
            reserved: 0,
            bce: Bce {
                pos_weight: BOUNDARY_POS_WEIGHT,
            },
            adam: AdamCfg::new(SFT_AUDIO_LR, 0.0),
            save_path,
            loss: 0.0,
            read_loss: 0.0,
            examples: 0,
            unmatched: 0,
            head_cut: (0, 0),
            ctc: CtcScore::default(),
        })
    }

    /// The words take `id` is heard as, in the written spelling: what the
    /// SFT loader writes in place of the turn.
    pub fn heard_tokens(&self, id: &str) -> Option<Vec<u16>> {
        let p = &self.corpus.items[*self.by_id.get(id)?];
        Some(p.tokens.iter().map(|&t| t as u16).collect())
    }

    /// Sizes the encoder's scratch for examples of at most `turns` heard turns.
    pub fn reserve(&mut self, gpu: &Gpu, turns: usize) {
        let c = self.enc.cfg;
        let t_out = slowed(self.corpus.items.iter().map(|p| p.t_in / c.stride).max().unwrap_or(1));
        let words = self.corpus.items.iter().map(|p| p.segments.len()).max().unwrap_or(1);
        self.cache = Some(self.enc.training_cache(gpu, t_out * turns.max(1), t_out, words));
        self.reserved = turns.max(1);
    }

    /// One SFT example with its `heard` turns heard, into `model`'s
    /// accumulating step, and a step of the encoder. The returned loss is the
    /// reply's CE, as `forward_backward_masked` returns it.
    pub fn example(
        &mut self,
        gpu: &Gpu,
        model: &mut Hierarchical,
        tokens: &[usize],
        words: &[Range<usize>],
        word_loss: &[bool],
        heard: &[(String, Range<usize>)],
    ) -> f32 {
        if heard.len() > self.reserved {
            self.reserve(gpu, heard.len());
        }
        let items: Vec<&Prepared> = heard.iter().map(|(id, _)| &self.corpus.items[self.by_id[id]]).collect();
        // Each heard word's window index, in the order of the encoder's
        // embedding rows (the turns' words one after another).
        let mut at = Vec::new();
        for (p, (_, r)) in items.iter().zip(heard) {
            let Some(w) = words.iter().position(|w| w.start == r.start) else {
                at.clear();
                break;
            };
            let fits = p.words.iter().enumerate().all(|(j, pw)| words.get(w + j).is_some_and(|x| x.start == r.start + pw.start));
            if !fits {
                at.clear();
                break;
            }
            at.extend(w..w + p.words.len());
        }
        if at.is_empty() {
            self.unmatched += 1;
            return model.forward_backward_masked(gpu, tokens, words, word_loss);
        }
        let cache = self.cache.as_ref().expect("reserved");
        let c = self.enc.cfg;
        let hc = c.out;
        let batch = Batch::new(items, &self.corpus.stats, self.corpus.augment.as_ref());
        let mut cut = 0;
        let (out, ctc_logits) = self.enc.forward_aligned(gpu, &batch.feats, batch.rows(), batch.t_in, cache, |_, boundary, t_out| {
            let (pooled, n) = head_cut_segments(&batch, boundary, t_out, c);
            cut = n;
            pooled
        });
        self.head_cut.0 += cut;
        self.head_cut.1 += batch.items.len();
        let logits = out.boundary.to_host(gpu).data;
        let (_, d_logits) = self.bce.loss_and_grad(&logits, &batch.targets(out.t_out, c.stride, c.delay));
        let (ctc, d_ctc) = ctc_batch(&ctc_logits, &batch, out.t_out, c);
        let embeds = out.embeddings.as_ref().expect("every turn has words").to_host(gpu).data;

        // The last word is only decoded: it has no backbone row.
        let dw = words.len() - 1;
        let (rows, from): (Vec<usize>, Vec<usize>) = at.iter().enumerate().map(|(e, &w)| (w, e)).filter(|&(w, _)| w < dw).unzip();
        let given: Vec<f32> = from.iter().flat_map(|&e| embeds[e * hc..(e + 1) * hc].iter().copied()).collect();
        let given = GTensor::from_host(gpu, &Tensor::new(&[rows.len(), hc], given));
        let mut d = GTensor::zeros(gpu, &[rows.len(), hc]);

        if SFT_AUDIO_DISTILL > 0.0 {
            model.record_distill();
        }
        model.eval_loss_masked(gpu, tokens, words, Some(word_loss));
        self.read_loss += model.last_word_loss();
        if SFT_AUDIO_DISTILL > 0.0 {
            let targets = model.take_distill();
            model.apply_distill(targets, SFT_AUDIO_DISTILL);
        }
        let loss = model.forward_backward_mixed_masked(gpu, tokens, words, Some(word_loss), &given, &rows, &mut d);
        model.end_distill();
        self.loss += model.last_word_loss();
        self.examples += 1;
        self.ctc.merge(&ctc);

        let d = d.to_host(gpu).data;
        let mut d_e = vec![0.0; embeds.len()];
        for (i, &e) in from.iter().enumerate() {
            d_e[e * hc..(e + 1) * hc].copy_from_slice(&d[i * hc..(i + 1) * hc]);
        }
        let cols = c.delay + 1;
        self.enc.backward(
            gpu,
            &GTensor::from_host(gpu, &Tensor::new(&[logits.len() / cols, cols], d_logits)),
            Some(&GTensor::from_host(gpu, &Tensor::new(&[embeds.len() / hc, hc], d_e))),
            Some((
                &GTensor::from_host(gpu, &Tensor::new(&[ctc_logits.len() / c.ctc, c.ctc], d_ctc)),
                CTC_WEIGHT,
            )),
            cache,
        );
        self.adam.t += 1;
        self.enc.step(gpu, &self.adam);
        self.ck.step += 1;
        self.ck.seconds += batch.seconds();
        loss
    }

    /// The heard examples since the last report: the reply's loss after
    /// hearing against after reading the same words, and the character head's
    /// error rate; `None` when there were none.
    pub fn report(&mut self) -> Option<String> {
        if self.examples == 0 {
            return None;
        }
        let n = self.examples as f32;
        let line = format!(
            "  heard {} | reply loss heard {:.3} read {:.3} | CER {} | head cut {}/{} | unmatched {}",
            self.examples,
            self.loss / n,
            self.read_loss / n,
            self.ctc.summary(),
            self.head_cut.0,
            self.head_cut.1,
            self.unmatched,
        );
        (self.loss, self.read_loss, self.examples, self.unmatched) = (0.0, 0.0, 0, 0);
        self.head_cut = (0, 0);
        self.ctc = CtcScore::default();
        Some(line)
    }

    pub fn save(&mut self, gpu: &Gpu) {
        if let Err(e) = self.enc.save(gpu, &self.save_path, &self.ck) {
            eprintln!("saving {} failed: {e}", self.save_path);
        }
    }
}

/// A run of utterances as one document of written text, with the utterances
/// at positions `heard` as their spoken words. `heard` pairs each of those
/// words' window index with its row in the encoder's embeddings (the heard
/// utterances' words in order).
struct Document {
    tokens: Vec<usize>,
    words: Vec<Range<usize>>,
    heard: Vec<(usize, usize)>,
}

impl Document {
    fn new(items: &[&Prepared], heard: &[usize]) -> Option<Self> {
        let mut raw: Vec<u16> = Vec::new();
        let (mut expected, mut heard_words, mut embed_row) = (0, Vec::new(), 0);
        // Whether the written text so far ended a sentence: a heard utterance
        // carries its full stop in its melody, not in the tokens.
        let mut ended = false;
        for (k, p) in items.iter().enumerate() {
            let is_heard = heard.contains(&k);
            let toks = if is_heard { &p.tokens } else { &p.written };
            if !raw.is_empty() {
                raw.push(b' ' as u16);
            }
            let first = raw.len();
            raw.extend(toks.iter().map(|&c| c as u16));
            if first > 0 && !ended && !pronoun_i(toks) {
                raw[first] = (raw[first] as u8).to_ascii_lowercase() as u16;
            }
            ended = p
                .written
                .iter()
                .rev()
                .find(|&&c| !(c as u8).is_ascii_whitespace())
                .is_some_and(|&c| matches!(c as u8, b'.' | b'!' | b'?'));
            let n = if is_heard { p.words.len() } else { p.written_words.len() };
            if is_heard {
                heard_words.extend((0..n).map(|j| (expected + j, embed_row + j)));
                embed_row += n;
            }
            expected += n;
        }
        let words = split_words(&raw);
        (words.len() == expected).then(|| Self {
            tokens: raw.iter().map(|&c| c as usize).collect(),
            words,
            heard: heard_words,
        })
    }

    /// The bytes `span` of the document `text` as written text, with each of
    /// `takes` (in order, inside `span`) heard in place of the sentence at its
    /// range. `None` when a take's words do not stay whole words in the text
    /// around them.
    fn voice(text: &str, span: std::ops::Range<usize>, takes: &[(&Prepared, Range<usize>)]) -> Option<Self> {
        let mut raw: Vec<u16> = Vec::new();
        let mut starts = Vec::new();
        let mut cursor = span.start;
        for (p, r) in takes {
            raw.extend(text[cursor..r.start].bytes().map(u16::from));
            starts.push(raw.len());
            raw.extend(p.tokens.iter().map(|&c| c as u16));
            cursor = r.end;
        }
        raw.extend(text[cursor..span.end].bytes().map(u16::from));
        let words = split_words(&raw);
        let (mut heard, mut embed_row) = (Vec::new(), 0);
        for ((p, _), &at) in takes.iter().zip(&starts) {
            let w = words.iter().position(|w| w.start == at)?;
            for (j, pw) in p.words.iter().enumerate() {
                (words.get(w + j)?.start == at + pw.start).then_some(())?;
                heard.push((w + j, embed_row + j));
            }
            embed_row += p.words.len();
        }
        Some(Self {
            tokens: raw.iter().map(|&c| c as usize).collect(),
            words,
            heard,
        })
    }
}

/// Mode `ae2a`: how well `<model>.ae2` finds the words against the MFA
/// alignment, for held-out and training speakers: the boundary head streaming
/// on its own, the boundary head told how many words were spoken, and the
/// character head forced onto the transcript. Needs no text model.
pub fn evaluate_alignment(model_path: &str) {
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No CUDA GPU available ({e}).");
            return;
        }
    };
    let path = format!("{model_path}.ae2");
    let (mut enc, ck) = match AudioEncoder::load(&gpu, &path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Could not load {path}: {e}");
            return;
        }
    };
    println!("{path}: step {}", ck.step);
    let mut corpus = Corpus::load(&Utf8Tokenizer::new(), enc.cfg.stride);
    corpus.stats = ck.stats.clone();
    let cache = corpus.training_cache(&gpu, &enc);
    let c = enc.cfg;
    let aligner = CountAligner {
        delay: c.delay,
        from_col: 0,
        off_peak: 3.0,
    };
    let variants = [
        ("no duration prior", None),
        (
            "duration prior",
            Some(DurationPrior {
                sigma: 0.5,
                weight: 1.0,
                max_frames: 100,
            }),
        ),
    ];
    let train = corpus.train();
    let sample: Vec<&Prepared> = train
        .iter()
        .step_by((train.len() / EVAL_TRAIN_SAMPLE).max(1))
        .copied()
        .collect();
    let peak_decoders: Vec<PeakDecoder> = [(0.3, 1), (0.5, 1), (0.7, 1)]
        .into_iter()
        .map(|(threshold, radius)| PeakDecoder {
            delay: c.delay,
            threshold,
            radius,
        })
        .collect();
    for (name, part) in [("val", corpus.val()), ("train sample", sample)] {
        let mut scores: Vec<AlignScore> = variants.iter().map(|_| AlignScore::default()).collect();
        let mut forced = AlignScore::default();
        // The same ends moved 1 and 2 frames earlier: the causal head emits a
        // character only after it has heard it.
        let mut earlier = [AlignScore::default(), AlignScore::default()];
        let mut peak_scores: Vec<BoundaryScore> = peak_decoders
            .iter()
            .map(|_| BoundaryScore::default())
            .collect();
        for chunk in part.chunks(AUDIO_BATCH) {
            let batch = Batch::new(chunk.to_vec(), &corpus.stats, None);
            let none = vec![WordSegments::default(); chunk.len()];
            let out = enc.forward(&gpu, &batch.feats, chunk.len(), batch.t_in, &none, &cache);
            let logits = out.boundary.to_host(&gpu).data;
            let chars = out.ctc.to_host(&gpu).data;
            enc.discard();
            let cols = c.delay + 1;
            let per_frame = c.ctc / CtcAlphabet::SIZE;
            for (b, p) in chunk.iter().enumerate() {
                let own = p.t_in / c.stride;
                let rows = &logits[b * out.t_out * cols..(b * out.t_out + own) * cols];
                let lens: Vec<usize> = p.words.iter().map(|r| r.end - r.start).collect();
                for (score, (_, prior)) in scores.iter_mut().zip(&variants) {
                    let ends = match prior {
                        Some(prior) => aligner.align_words(rows, own, &lens, *prior),
                        None => aligner
                            .align(&aligner.peaks(&aligner.end_scores(rows, own)), lens.len()),
                    };
                    score.add(ends.as_deref(), &p.segments);
                }
                let steps = &chars[b * out.t_out * c.ctc..(b * out.t_out + own) * c.ctc];
                let ends = Ctc {
                    vocab: CtcAlphabet::SIZE,
                }
                .align(steps, &p.chars)
                .map(|last| {
                    CtcAlphabet::word_ends(&p.chars)
                        .iter()
                        .map(|&k| last[k] / per_frame)
                        .collect::<Vec<usize>>()
                });
                forced.add(ends.as_deref(), &p.segments);
                for (k, score) in earlier.iter_mut().enumerate() {
                    let moved = ends.as_ref().map(|e| e.iter().map(|&f| f.saturating_sub(k + 1)).collect::<Vec<_>>());
                    score.add(moved.as_deref(), &p.segments);
                }
                for (score, dec) in peak_scores.iter_mut().zip(&peak_decoders) {
                    score.add(&dec.decide(rows, own), &p.segments);
                }
            }
        }
        println!("{name} ({} utterances):", part.len());
        for (score, dec) in peak_scores.iter().zip(&peak_decoders) {
            println!(
                "  peak decoder p > {:.2}, radius {}: {}",
                dec.threshold,
                dec.radius,
                score.summary()
            );
        }
        for (score, (variant, ..)) in scores.iter().zip(&variants) {
            println!("  count-constrained, {variant}: {}", score.summary());
        }
        println!("  CTC forced alignment: {}", forced.summary());
        for (k, score) in earlier.iter().enumerate() {
            println!("  CTC forced alignment, {} frame(s) earlier: {}", k + 1, score.summary());
        }
    }
}

/// Takes whose features are computed at once, and utterances per encoder batch.
const ALIGN_CHUNK: usize = 1024;
const ALIGN_BATCH: usize = 16;

/// Mode `ae2v`: word ends for the recorded takes (`AUDIO_VOICE_DIR` and
/// `AUDIO_TAKE_DIRS`) by CTC forced alignment of what was said, written to each
/// speaker's `alignment.jsonl`. `path` is an audio encoder checkpoint
/// (`<model>.ae2`, or the file itself). Takes already aligned keep their
/// alignment unless `AE2V_ALL` is set.
pub fn align_voice(path: &str) {
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No CUDA GPU available ({e}).");
            return;
        }
    };
    let path = if std::path::Path::new(path).is_file() {
        path.to_string()
    } else {
        format!("{path}.ae2")
    };
    let (mut enc, ck) = match AudioEncoder::load(&gpu, &path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Could not load {path}: {e}");
            return;
        }
    };
    println!("{path}: step {}", ck.step);
    for &dir in std::iter::once(&AUDIO_VOICE_DIR).chain(AUDIO_TAKE_DIRS) {
        let mut takes = match voice::load(dir) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("{dir}: {e}");
                continue;
            }
        };
        println!("{dir}: {} takes", takes.len());
        let c = enc.cfg;
        let per_frame = c.ctc / CtcAlphabet::SIZE;
        let ctc = Ctc {
            vocab: CtcAlphabet::SIZE,
        };
        let (mut aligned, mut errors, mut chars) = (0, 0, 0);
        let t0 = Instant::now();
        // Features in parallel a chunk at a time, then the chunk through the
        // encoder in batches of similar length.
        // Only takes with no alignment yet, unless `AE2V_ALL` asks for all of
        // them: training aligns them again online anyway.
        let all = std::env::var_os("AE2V_ALL").is_some();
        let mut todo: Vec<&mut voice::Take> = takes.iter_mut().filter(|t| all || t.ends.is_none()).collect();
        println!("  {} to align", todo.len());
        for chunk in todo.chunks_mut(ALIGN_CHUNK) {
            let loaded: Vec<Option<(Vec<u32>, Vec<f32>, f32)>> = chunk
                .par_iter()
                .map(|t| {
                    let target = CtcAlphabet::encode(&t.spoken)?;
                    let clip = Clip::read_at(&t.audio, AUDIO_SR).ok()?;
                    let mut feats = LogMel::new().utterance(&clip.samples);
                    ck.stats.apply(&mut feats);
                    (c.stem().frames(feats.len() / AUDIO_FEATS) > 0).then(|| (target, feats, clip.seconds()))
                })
                .collect();
            let mut order: Vec<usize> = (0..chunk.len()).filter(|&i| loaded[i].is_some()).collect();
            order.sort_by_key(|&i| loaded[i].as_ref().unwrap().1.len());
            for group in order.chunks(ALIGN_BATCH) {
                let t_in = group.iter().map(|&i| loaded[i].as_ref().unwrap().1.len() / AUDIO_FEATS).max().unwrap();
                let t_out = c.stem().frames(t_in);
                let mut feats = vec![0.0; group.len() * t_in * AUDIO_FEATS];
                for (b, &i) in group.iter().enumerate() {
                    let f = &loaded[i].as_ref().unwrap().1;
                    feats[b * t_in * AUDIO_FEATS..][..f.len()].copy_from_slice(f);
                }
                let cache = enc.training_cache(&gpu, group.len() * t_out, t_out, 0);
                let none = vec![WordSegments::default(); group.len()];
                let out = enc.forward(&gpu, &feats, group.len(), t_in, &none, &cache);
                let steps = out.ctc.to_host(&gpu).data;
                enc.discard();
                for (b, &i) in group.iter().enumerate() {
                    let (target, f, seconds) = loaded[i].as_ref().unwrap();
                    let own = f.len() / AUDIO_FEATS / c.stride;
                    let rows = &steps[b * t_out * c.ctc..(b * t_out + own) * c.ctc];
                    errors += edit_distance(&ctc.greedy(rows), target);
                    chars += target.len();
                    // A take the encoder cannot align keeps the alignment it had.
                    let Some(last) = ctc.align(rows, target) else {
                        continue;
                    };
                    // A word ends with its encoder frame.
                    chunk[i].ends = Some(
                        CtcAlphabet::word_ends(target)
                            .iter()
                            .map(|&k| ((last[k] / per_frame + 1) as f32 * ENC_FRAME_S).min(*seconds))
                            .collect(),
                    );
                    aligned += 1;
                }
            }
            print!("\r  {aligned} aligned ({:.0?})", t0.elapsed());
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
        println!();
        match voice::save_alignments(dir, &takes) {
            Ok(()) => println!(
                "aligned {aligned} of {} takes; the character head hears them at CER {:.1}%",
                takes.len(),
                100.0 * errors as f32 / chars.max(1) as f32
            ),
            Err(e) => eprintln!("writing the alignments failed: {e}"),
        }
    }
}

/// Distance of aligned word ends from the labelled ones, in encoder frames.
#[derive(Default)]
struct AlignScore {
    /// Words off by 0, 1, 2, 3..=5 and more than 5 frames.
    buckets: [usize; 5],
    abs: usize,
    signed: i64,
    words: usize,
    failed: usize,
}

impl AlignScore {
    fn add(&mut self, ends: Option<&[usize]>, segs: &WordSegments) {
        let Some(ends) = ends else {
            self.failed += 1;
            return;
        };
        for (&e, s) in ends.iter().zip(&segs.0) {
            let d = e.abs_diff(s.last());
            self.buckets[match d {
                0..=2 => d,
                3..=5 => 3,
                _ => 4,
            }] += 1;
            self.abs += d;
            self.signed += e as i64 - s.last() as i64;
            self.words += 1;
        }
    }

    fn summary(&self) -> String {
        let w = self.words.max(1) as f32;
        let pct = |n: usize| 100.0 * n as f32 / w;
        let within1 = self.buckets[0] + self.buckets[1];
        format!(
            "exact {:.1}% ±1 {:.1}% ±2 {:.1}% ±3-5 {:.1}% >5 {:.1}% | mean |err| {:.2} frames ({:.0} ms), bias {:+.2} | failed {}",
            pct(self.buckets[0]),
            pct(within1),
            pct(within1 + self.buckets[2]),
            pct(self.buckets[3]),
            pct(self.buckets[4]),
            self.abs as f32 / w,
            1000.0 * ENC_FRAME_S * self.abs as f32 / w,
            self.signed as f32 / w,
            self.failed
        )
    }
}

/// One utterance ready for the encoder and the text model. Its features are
/// computed when a batch needs them: a 100 h corpus of them would not fit in
/// memory.
#[derive(Clone)]
struct Prepared {
    /// Position in `Corpus::items`.
    index: usize,
    id: String,
    speaker: String,
    audio: PathBuf,
    t_in: usize,
    segments: WordSegments,
    /// Each spoken word's `(start, end)` in seconds, and the input frames per
    /// encoder frame `segments` were laid out with: what a change of speed
    /// lays them out again from.
    times: Vec<(f32, f32)>,
    stride: usize,
    /// The spoken words, one per segment, spelled and capitalized as written
    /// where a written transcript maps onto them. Punctuation is not among
    /// them: it is heard as the pause and melody of the word before it.
    tokens: Vec<usize>,
    words: Vec<Range<usize>>,
    /// The written transcript, punctuation included, as the text model reads
    /// it; `spoken_at[i]` is the written word that spoken word `i` is.
    written: Vec<usize>,
    written_words: Vec<Range<usize>>,
    spoken_at: Vec<usize>,
    /// The transcript in the character head's alphabet.
    chars: Vec<u32>,
}

impl Prepared {
    /// `None` when the transcript's words do not map one to one onto the spoken
    /// ones, or a word falls outside the audio.
    /// `length` is the audio's `(samples, rate)`, read from its header when
    /// not given.
    fn new(
        u: &Utterance,
        tok: &Utf8Tokenizer,
        stride: usize,
        length: Option<(usize, usize)>,
    ) -> Option<Self> {
        let (samples, rate) = match length {
            Some(l) => l,
            None => Clip::length(&u.audio).ok()?,
        };
        if rate != AUDIO_SR {
            return None;
        }
        let t_in = samples / AUDIO_HOP;
        let segments = WordSegments::from_word_times(&u.times(), t_in / stride);
        let spoken: Vec<&str> = u.words.iter().map(|w| w.text.as_str()).collect();
        let n = spoken.len();
        let chars = CtcAlphabet::encode(&spoken.join(" "))?;
        let (raw, written, spoken_at) = u
            .written
            .as_deref()
            .and_then(|w| spelled(&tok.to_tokens(w), &spoken))
            .unwrap_or_else(|| {
                let raw = tok.to_tokens(&u.text());
                (raw.clone(), raw, (0..n).collect())
            });
        let words = split_words(&raw);
        // Faster than anyone speaks: the transcript is not this audio's.
        let plausible = n as f32 <= MAX_WORDS_PER_SECOND * (samples as f32 / AUDIO_SR as f32);
        (n >= 2 && n <= AUDIO_MAX_WORDS && plausible && words.len() == n && segments.len() == n).then(|| Self {
            index: 0,
            id: u.id.clone(),
            speaker: u.speaker.clone(),
            audio: u.audio.clone(),
            t_in,
            segments,
            times: u.times(),
            stride,
            tokens: raw.iter().map(|&t| t as usize).collect(),
            words,
            written_words: split_words(&written),
            written: written.iter().map(|&t| t as usize).collect(),
            spoken_at,
            chars,
        })
    }

    /// `[t_in, AUDIO_FEATS]`, normalized by `stats`.
    fn features(&self, stats: &FeatureStats, augment: Option<&Augment>) -> Vec<f32> {
        self.features_at(stats, augment, 1.0, None).0
    }

    /// The features with the audio played `speed` times as fast, and this item
    /// with its length and word timings stretched to match; its words laid out
    /// from `times` (seconds, as recorded) in place of its own when given.
    fn features_at(
        &self,
        stats: &FeatureStats,
        augment: Option<&Augment>,
        speed: f32,
        times: Option<&[(f32, f32)]>,
    ) -> (Vec<f32>, Prepared) {
        let mut clip = Clip::read_at(&self.audio, AUDIO_SR)
            .unwrap_or_else(|e| panic!("{}: {e}", self.audio.display()));
        let mut item = self.clone();
        if let Some(times) = times {
            let segments = WordSegments::from_word_times(times, self.t_in / self.stride);
            if segments.len() == self.segments.len() {
                (item.segments, item.times) = (segments, times.to_vec());
            }
        }
        if speed != 1.0 {
            let fast = Augment::speed(&clip.samples, AUDIO_SR, speed);
            let t_in = fast.len() / AUDIO_HOP;
            let times: Vec<(f32, f32)> = item.times.iter().map(|&(a, b)| (a / speed, b / speed)).collect();
            let segments = WordSegments::from_word_times(&times, t_in / self.stride);
            // Played too fast for every word to keep a frame: as recorded.
            if segments.len() == item.segments.len() {
                clip.samples = fast;
                (item.t_in, item.segments, item.times) = (t_in, segments, times);
            }
        }
        if let Some(a) = augment {
            a.apply(&mut clip.samples);
        }
        let mut feats = LogMel::new().utterance(&clip.samples);
        assert_eq!(
            feats.len(),
            item.t_in * AUDIO_FEATS,
            "{}: length differs from its header",
            self.audio.display()
        );
        stats.apply(&mut feats);
        (feats, item)
    }

    fn seconds(&self) -> f64 {
        (self.t_in * AUDIO_HOP) as f64 / AUDIO_SR as f64
    }
}

/// The written transcript `written` laid over the spoken words: the spoken
/// words in the written spelling, joined by spaces, the written tokens, and the
/// written word each spoken word is. A written word with no letter or digit is
/// punctuation and belongs to no spoken word; every other one must be the next
/// spoken word, or `None` (a number written as digits, an abbreviation).
fn spelled(written: &[u16], spoken: &[&str]) -> Option<(Vec<u16>, Vec<u16>, Vec<usize>)> {
    let fold = |t: &[u16]| -> String {
        let s: String = t
            .iter()
            .map(|&b| (b as u8 as char).to_ascii_lowercase())
            .filter(|c| c.is_ascii_alphanumeric() || *c == '\'')
            .collect();
        s.trim_matches('\'').to_string()
    };
    let mut raw: Vec<u16> = Vec::new();
    let mut spoken_at = Vec::with_capacity(spoken.len());
    for (i, w) in split_words(written).iter().enumerate() {
        let toks = &written[w.start..w.end];
        if toks.iter().any(|&t| t >= 128) {
            return None;
        }
        let word = fold(toks);
        if word.is_empty() {
            continue;
        }
        if spoken.get(spoken_at.len()).map(|s| s.trim_matches('\'')) != Some(word.as_str()) {
            return None;
        }
        if !raw.is_empty() {
            raw.push(b' ' as u16);
        }
        // The splitter keeps a hyphen on the word before it ("heavy-" of
        // "heavy-tailed"), which would split off as a word of its own here.
        let end = toks.iter().rposition(|&t| !(t as u8).is_ascii_whitespace() && t != b'-' as u16)? + 1;
        raw.extend_from_slice(&toks[..end]);
        spoken_at.push(i);
    }
    (spoken_at.len() == spoken.len()).then(|| (raw, written.to_vec(), spoken_at))
}

/// Audio lengths `(samples, rate)` by file, kept in `<AUDIO_DATA>/lengths.tsv`:
/// reading a hundred thousand FLAC headers off a hard disk takes a quarter of
/// an hour, the file a second.
struct AudioLengths(HashMap<PathBuf, (usize, usize)>);

impl AudioLengths {
    fn path() -> PathBuf {
        PathBuf::from(AUDIO_DATA).join("lengths.tsv")
    }

    fn load() -> Self {
        let text = std::fs::read_to_string(Self::path()).unwrap_or_default();
        Self(
            text.lines()
                .filter_map(|l| {
                    let mut f = l.split('\t');
                    let path = PathBuf::from(f.next()?);
                    Some((path, (f.next()?.parse().ok()?, f.next()?.parse().ok()?)))
                })
                .collect(),
        )
    }

    fn get(&self, audio: &PathBuf) -> Option<(usize, usize)> {
        self.0.get(audio).copied()
    }

    /// Adds what was not known yet and rewrites the file if anything was.
    fn update<'a>(&mut self, found: impl Iterator<Item = (&'a PathBuf, &'a Option<(usize, usize)>)>) {
        let before = self.0.len();
        for (path, l) in found {
            if let Some(l) = l {
                self.0.entry(path.clone()).or_insert(*l);
            }
        }
        if self.0.len() == before {
            return;
        }
        let mut out = String::new();
        for (path, (samples, rate)) in &self.0 {
            out.push_str(&format!("{}\t{samples}\t{rate}\n", path.display()));
        }
        let tmp = Self::path().with_extension("tsv.tmp");
        if std::fs::write(&tmp, out).and_then(|()| std::fs::rename(&tmp, Self::path())).is_err() {
            eprintln!("could not write {}", Self::path().display());
        }
    }
}

/// Speech to play under other speech: `BABBLE_PER_SPEAKER` utterances of
/// each of the first `BABBLE_SPEAKERS` speakers of `AUDIO_BABBLE_SPLIT`.
fn babble_files() -> Vec<PathBuf> {
    const BABBLE_SPEAKERS: usize = 30;
    const BABBLE_PER_SPEAKER: usize = 10;
    let root = PathBuf::from(AUDIO_DATA).join("LibriSpeech").join(AUDIO_BABBLE_SPLIT);
    let sorted = |d: &std::path::Path| -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = std::fs::read_dir(d)
            .map(|r| r.flatten().map(|e| e.path()).collect())
            .unwrap_or_default();
        v.sort();
        v
    };
    sorted(&root)
        .into_iter()
        .take(BABBLE_SPEAKERS)
        .flat_map(|speaker| {
            sorted(&speaker)
                .into_iter()
                .flat_map(|chapter| sorted(&chapter))
                .filter(|p| p.extension().is_some_and(|e| e == "flac"))
                .take(BABBLE_PER_SPEAKER)
        })
        .collect()
}

/// `frames` of audio as long as they get at the slowest augmentation speed.
fn slowed(frames: usize) -> usize {
    (frames as f32 / AugmentCfg::speech(AUDIO_SR).speed.0).ceil() as usize + 1
}

/// `raw` cut into its words, as token ranges.
fn split_words(raw: &[u16]) -> Vec<Range<usize>> {
    word_ends(raw)
        .iter()
        .scan(0usize, |start, &end| {
            let r = Range::from(*start..end as usize);
            *start = end as usize;
            Some(r)
        })
        .collect()
}

/// Whether `toks` starts with the pronoun "I", which stays capitalized.
fn pronoun_i(toks: &[usize]) -> bool {
    toks.first() == Some(&(b'I' as usize))
        && toks.get(1).is_none_or(|&t| t == b' ' as usize || t == b'\'' as usize)
}

/// One TTS question in twenty, by its id, is never trained on.
fn tts_held_out(p: &Prepared) -> bool {
    use std::hash::{Hash, Hasher};
    let mut h = std::hash::DefaultHasher::new();
    p.id.hash(&mut h);
    h.finish() % 20 == 0
}

/// The documents in `Corpus::voice_docs` holding a take from `train`, the
/// takes an audio window may hear.
struct DocSource {
    docs: Vec<usize>,
    train: std::collections::HashSet<usize>,
}

impl DocSource {
    fn new(corpus: &Corpus, train: impl Iterator<Item = usize>) -> Self {
        let train: std::collections::HashSet<usize> = train.collect();
        let docs = (0..corpus.voice_docs.len())
            .filter(|&d| corpus.voice_docs[d].takes.iter().any(|(i, _)| train.contains(i)))
            .collect();
        Self { docs, train }
    }

    fn summary(&self, corpus: &Corpus) -> String {
        let heard = self
            .docs
            .iter()
            .flat_map(|&d| &corpus.voice_docs[d].takes)
            .filter(|(i, _)| self.train.contains(i))
            .count();
        format!("{} documents holding {heard} of {} training takes", self.docs.len(), self.train.len())
    }
}

/// A document voice prompts were read out of, and the takes read from it:
/// each take's item and the byte range of its sentence in `text`, in order.
struct VoiceDoc {
    text: String,
    takes: Vec<(usize, Range<usize>)>,
}

/// Where the sentence `text` stands in `doc`, as a byte range from its first
/// letter to the end of its last word, punctuation included. Compared on
/// letters and digits alone: the prompt was normalized (plain quotes and
/// dashes, one line) and the document was not.
fn locate(doc: &str, text: &str) -> Option<Range<usize>> {
    // Lower-cased letters and digits, and for each of their bytes the byte of
    // `s` the character came from.
    let skeleton = |s: &str| {
        let (mut out, mut at) = (String::new(), Vec::new());
        for (i, c) in s.char_indices().filter(|(_, c)| c.is_alphanumeric()) {
            for l in c.to_lowercase() {
                out.push(l);
                at.extend(std::iter::repeat_n(i, l.len_utf8()));
            }
        }
        (out, at)
    };
    let (d, at) = skeleton(doc);
    let (t, _) = skeleton(text);
    let k = d.find(&t).filter(|_| !t.is_empty())?;
    let last = at[k + t.len() - 1];
    let end = doc[last..]
        .find(char::is_whitespace)
        .map_or(doc.len(), |e| last + e);
    Some(Range::from(at[k]..end))
}

/// The text model's window around one utterance: `prefix` words of the
/// chapter's transcript before it, its own `audio` words, which the backbone
/// reads from the audio encoder, and `cont` words of transcript after it.
/// The teacher reads the same window as written text: the utterance with its
/// punctuation, so more words; `map[i]` is the teacher's word for word `i`.
struct Window {
    tokens: Vec<usize>,
    words: Vec<Range<usize>>,
    teacher_tokens: Vec<usize>,
    teacher_words: Vec<Range<usize>>,
    map: Vec<usize>,
    prefix: usize,
    audio: usize,
    cont: usize,
}

impl Window {
    /// Up to `AUDIO_CONTEXT_WORDS` words from the end of the written text
    /// `prev` and up to `after` from the start of `next` around `p`; `p` alone
    /// when joining moves a word border.
    fn new(prev: Option<&[usize]>, p: &Prepared, next: Option<&[usize]>, after: usize) -> Self {
        let as_u16 = |t: &[usize]| -> Vec<u16> { t.iter().map(|&c| c as u16).collect() };
        let blank = |c: &usize| (*c as u8).is_ascii_whitespace();
        // Whitespace between the pieces is the window's own; a line break
        // before the utterance ends what came before it like a full stop.
        let prev = prev.map(|t| {
            let end = t.iter().rposition(|c| !blank(c)).map_or(0, |i| i + 1);
            (&t[..end], t[end..].contains(&(b'\n' as usize)))
        });
        let next = next.map(|t| &t[t.iter().position(|c| !blank(c)).unwrap_or(t.len())..]);
        let prev_words = prev.map(|(t, _)| split_words(&as_u16(t))).unwrap_or_default();
        let next_words = next.map(|t| split_words(&as_u16(t))).unwrap_or_default();
        let before = prev_words.len().min(AUDIO_CONTEXT_WORDS);
        let after = next_words.len().min(after);
        let (mut student, mut teacher): (Vec<u16>, Vec<u16>) = (Vec::new(), Vec::new());
        // A transcript starts capitalized; inside the window it is running text
        // unless the written text before it ended a sentence.
        let mut push = |s: &[usize], t: &[usize], broke: bool| {
            let inside = !teacher.is_empty();
            let ended = broke
                || teacher
                    .iter()
                    .rev()
                    .find(|&&c| !(c as u8).is_ascii_whitespace())
                    .is_some_and(|&c| matches!(c as u8, b'.' | b'!' | b'?'));
            for (raw, toks) in [(&mut student, s), (&mut teacher, t)] {
                if raw.last().is_some_and(|&c| !(c as u8).is_ascii_whitespace()) {
                    raw.push(b' ' as u16);
                }
                let first = raw.len();
                raw.extend(toks.iter().map(|&c| c as u16));
                if inside && !ended && !pronoun_i(toks) {
                    if let Some(c) = raw.get_mut(first) {
                        *c = (*c as u8).to_ascii_lowercase() as u16;
                    }
                }
            }
        };
        let mut broke = false;
        if let Some((t, b)) = prev.filter(|_| before > 0) {
            let tail = &t[prev_words[prev_words.len() - before].start..];
            push(tail, tail, false);
            broke = b;
        }
        push(&p.tokens, &p.written, broke);
        if let Some(t) = next.filter(|_| after > 0) {
            let head = &t[..next_words[after - 1].end];
            push(head, head, false);
        }
        let (words, teacher_words) = (split_words(&student), split_words(&teacher));
        if words.len() != before + p.words.len() + after
            || teacher_words.len() != before + p.written_words.len() + after
        {
            return Self {
                tokens: p.tokens.clone(),
                words: p.words.clone(),
                teacher_tokens: p.written.clone(),
                teacher_words: p.written_words.clone(),
                map: p.spoken_at.clone(),
                prefix: 0,
                audio: p.words.len(),
                cont: 0,
            };
        }
        let map = (0..before)
            .chain(p.spoken_at.iter().map(|&i| before + i))
            .chain((0..after).map(|j| before + p.written_words.len() + j))
            .collect();
        Self {
            tokens: student.iter().map(|&t| t as usize).collect(),
            words,
            teacher_tokens: teacher.iter().map(|&t| t as usize).collect(),
            teacher_words,
            map,
            prefix: before,
            audio: p.words.len(),
            cont: after,
        }
    }

    /// `mask` over the student's predictions as the same predictions of the
    /// teacher; the teacher's own predictions of punctuation are not scored.
    fn teacher_mask(&self, mask: &[bool]) -> Vec<bool> {
        let mut out = vec![false; self.teacher_words.len() - 1];
        for (w, _) in mask.iter().enumerate().filter(|(_, m)| **m) {
            out[self.map[w + 1] - 1] = true;
        }
        out
    }

    /// Words the decoder predicts: every one but the first.
    fn decoded(&self) -> usize {
        self.words.len() - 1
    }

    /// Backbone rows read from the audio: every word of the utterance, the last
    /// only when text follows it (a window's last word is never read).
    fn audio_rows(&self) -> usize {
        if self.cont > 0 { self.audio } else { self.audio - 1 }
    }

    /// Decoded word `w` predicts word `w + 1` from the backbone's state after
    /// word `w`. Trained once the audio has begun; before it the encoder takes
    /// no part.
    fn train_mask(&self) -> Option<Vec<bool>> {
        (self.prefix > 0).then(|| (0..self.decoded()).map(|w| w >= self.prefix).collect())
    }

    /// Predictions of the utterance's own words from its audio.
    fn audio_mask(&self) -> Vec<bool> {
        let end = self.prefix + self.audio;
        (0..self.decoded())
            .map(|w| w >= self.prefix && w + 1 < end)
            .collect()
    }

    /// The prediction of the first word after the utterance: from the state
    /// the audio alone leaves, where a continuation starts.
    fn next_mask(&self) -> Vec<bool> {
        let end = self.prefix + self.audio;
        (0..self.decoded()).map(|w| w + 1 == end).collect()
    }

    /// Predictions of the text after the utterance.
    fn cont_mask(&self) -> Vec<bool> {
        let end = self.prefix + self.audio;
        (0..self.decoded()).map(|w| w + 1 >= end).collect()
    }
}

/// Every usable utterance of `AUDIO_SPLITS`. The held-out speakers come from
/// the first split, so adding training data leaves the validation set as it
/// was.
struct Corpus {
    items: Vec<Prepared>,
    /// For each LibriSpeech item, the items read just before and just after
    /// it in its chapter, when those are usable.
    prev: Vec<Option<usize>>,
    next: Vec<Option<usize>>,
    /// The written text before and after a voice take that was read out of a
    /// document.
    around: HashMap<usize, (Vec<usize>, Vec<usize>)>,
    /// The documents the voice takes were read out of.
    voice_docs: Vec<VoiceDoc>,
    /// Items from here on are voice takes, from `tts_from` on TTS takes;
    /// `train` and `val` are LibriSpeech.
    voice_from: usize,
    tts_from: usize,
    /// Each of `AUDIO_TAKE_DIRS` as a range of `items`.
    take_sets: Vec<std::ops::Range<usize>>,
    /// Word times of takes (from `voice_from` on) aligned again by the encoder
    /// as it trains, in seconds as recorded; training batches lay their words
    /// out from these.
    realigned: std::sync::Mutex<HashMap<usize, Vec<(f32, f32)>>>,
    held_out: Vec<String>,
    total: usize,
    stats: FeatureStats,
    /// Applied to training batches only, when `AUDIO_AUGMENT` is on.
    augment: Option<Augment>,
}

impl Corpus {
    fn load(tok: &Utf8Tokenizer, stride: usize) -> Self {
        let mut utterances = Vec::new();
        let mut held_out = Vec::new();
        for (i, split) in AUDIO_SPLITS.iter().enumerate() {
            match LibriSpeech::load(AUDIO_DATA, split) {
                Ok(c) => {
                    if i == 0 {
                        held_out = c
                            .speakers()
                            .iter()
                            .rev()
                            .take(AUDIO_HELD_OUT_SPEAKERS)
                            .map(|s| s.to_string())
                            .collect();
                    }
                    utterances.extend(c.utterances)
                }
                Err(e) => eprintln!("skipping {AUDIO_DATA}/{split}: {e}"),
            }
        }
        let mut lengths = AudioLengths::load();
        let known: Vec<Option<(usize, usize)>> = utterances
            .par_iter()
            .map(|u| lengths.get(&u.audio).or_else(|| Clip::length(&u.audio).ok()))
            .collect();
        lengths.update(utterances.iter().map(|u| &u.audio).zip(&known));
        let mut items: Vec<Prepared> = utterances
            .par_iter()
            .zip(&known)
            .filter_map(|(u, &l)| l.and_then(|l| Prepared::new(u, tok, stride, Some(l))))
            .collect();
        for (i, p) in items.iter_mut().enumerate() {
            p.index = i;
        }
        let at: HashMap<&str, usize> = items.iter().map(|p| (p.id.as_str(), p.index)).collect();
        // LibriSpeech ids are `speaker-chapter-number`, numbered in reading order.
        let neighbour = |p: &Prepared, ahead: bool| -> Option<usize> {
            let (chapter, n) = p.id.rsplit_once('-')?;
            let k = n.parse::<usize>().ok()?;
            let k = if ahead { k + 1 } else { k.checked_sub(1)? };
            at.get(format!("{chapter}-{k:0w$}", w = n.len()).as_str())
                .copied()
        };
        let prev = items.iter().map(|p| neighbour(p, false)).collect();
        let next = items.iter().map(|p| neighbour(p, true)).collect();
        drop(at);
        let voice_from = items.len();
        let (mut around, mut voice_docs) =
            Self::load_voice(tok, stride, &mut items, AUDIO_VOICE_DIR, AUDIO_VOICE_DOCS);
        let tts_from = items.len();
        let mut take_sets = Vec::new();
        for dir in AUDIO_TAKE_DIRS {
            let docs = format!("{dir}/docs.jsonl");
            let from = items.len();
            let (tts_around, tts_docs) = Self::load_voice(tok, stride, &mut items, dir, &docs);
            around.extend(tts_around);
            voice_docs.extend(tts_docs);
            take_sets.push(from..items.len());
        }
        Self {
            items,
            prev,
            next,
            around,
            voice_docs,
            voice_from,
            tts_from,
            take_sets,
            realigned: std::sync::Mutex::new(HashMap::new()),
            held_out,
            total: utterances.len(),
            stats: FeatureStats::identity(),
            augment: AUDIO_AUGMENT.then(|| Augment {
                cfg: AugmentCfg::speech(AUDIO_SR),
                noise: NoiseBank::load(AUDIO_NOISE_DIR, AUDIO_SR),
                babble: NoiseBank::from_files(&babble_files(), AUDIO_SR),
            }),
        }
    }

    /// Only the takes of `dirs`, as one set each: no LibriSpeech, no
    /// documents, nothing held out.
    fn takes(tok: &Utf8Tokenizer, stride: usize, dirs: &[&str]) -> Self {
        let mut items = Vec::new();
        let mut take_sets = Vec::new();
        for dir in dirs {
            let from = items.len();
            Self::load_voice(tok, stride, &mut items, dir, "");
            take_sets.push(from..items.len());
        }
        Self {
            items,
            prev: Vec::new(),
            next: Vec::new(),
            around: HashMap::new(),
            voice_docs: Vec::new(),
            voice_from: 0,
            tts_from: 0,
            take_sets,
            realigned: std::sync::Mutex::new(HashMap::new()),
            held_out: Vec::new(),
            total: 0,
            stats: FeatureStats::identity(),
            augment: AUDIO_AUGMENT.then(|| Augment {
                cfg: AugmentCfg::speech(AUDIO_SR),
                noise: NoiseBank::load(AUDIO_NOISE_DIR, AUDIO_SR),
                babble: NoiseBank::from_files(&babble_files(), AUDIO_SR),
            }),
        }
    }

    /// The aligned voice takes appended to `items`, the document text around
    /// each one read out of a document, and those documents.
    fn load_voice(
        tok: &Utf8Tokenizer,
        stride: usize,
        items: &mut Vec<Prepared>,
        dir: &str,
        docs: &str,
    ) -> (HashMap<usize, (Vec<usize>, Vec<usize>)>, Vec<VoiceDoc>) {
        let takes = voice::load(dir).unwrap_or_default();
        let docs: HashMap<String, String> = std::fs::read_to_string(docs)
            .unwrap_or_default()
            .lines()
            .filter_map(|l| {
                Some((
                    crate::sft::json_string_field(l, "doc")?,
                    crate::sft::json_string_field(l, "text")?,
                ))
            })
            .collect();
        let tokens = |t: &str| -> Vec<usize> { tok.to_tokens(t).iter().map(|&c| c as usize).collect() };
        // Lengths come from the cache LibriSpeech's use, headers are read
        // only for takes new to it, and each take is prepared in parallel.
        let mut lengths = AudioLengths::load();
        let utterances: Vec<Option<Utterance>> = takes.par_iter().map(|t| t.utterance()).collect();
        let known: Vec<Option<(usize, usize)>> = takes
            .par_iter()
            .zip(&utterances)
            .map(|(t, u)| u.as_ref().and_then(|_| lengths.get(&t.audio).or_else(|| Clip::length(&t.audio).ok())))
            .collect();
        lengths.update(takes.iter().map(|t| &t.audio).zip(&known));
        let longest = (AUDIO_MAX_TAKE_SECONDS * AUDIO_SR as f32) as usize;
        type Ready = (Prepared, Option<(Range<usize>, Vec<usize>, Vec<usize>)>);
        let ready: Vec<Option<Ready>> = takes
            .par_iter()
            .zip(&utterances)
            .zip(&known)
            .map(|((take, u), &l)| {
                let p = Prepared::new(u.as_ref()?, tok, stride, Some(l?))?;
                if p.t_in * AUDIO_HOP > longest {
                    return None;
                }
                let doc = take.doc.as_ref().and_then(|d| docs.get(d));
                let context = doc.and_then(|d| {
                    let at = locate(d, &take.text)?;
                    Some((at, tokens(&d[..at.start]), tokens(&d[at.end..])))
                });
                Some((p, context))
            })
            .collect();
        let mut around = HashMap::new();
        let (mut voice_docs, mut doc_at) = (Vec::<VoiceDoc>::new(), HashMap::new());
        for (take, ready) in takes.iter().zip(ready) {
            let Some((mut p, context)) = ready else {
                continue;
            };
            p.index = items.len();
            if let Some((at, before, after)) = context {
                let id = take.doc.as_ref().expect("a located take has a document");
                around.insert(p.index, (before, after));
                let k = *doc_at.entry(id.clone()).or_insert_with(|| {
                    voice_docs.push(VoiceDoc {
                        text: docs[id].clone(),
                        takes: Vec::new(),
                    });
                    voice_docs.len() - 1
                });
                voice_docs[k].takes.push((p.index, at));
            }
            items.push(p);
        }
        for d in &mut voice_docs {
            d.takes.sort_by_key(|(_, r)| r.start);
        }
        (around, voice_docs)
    }

    fn is_val(&self, p: &Prepared) -> bool {
        self.held_out.contains(&p.speaker)
    }

    fn train(&self) -> Vec<&Prepared> {
        self.items[..self.voice_from].iter().filter(|p| !self.is_val(p)).collect()
    }

    fn val(&self) -> Vec<&Prepared> {
        self.items[..self.voice_from].iter().filter(|p| self.is_val(p)).collect()
    }

    /// Voice takes, of the held-out sessions or the newest `AUDIO_VOICE_LAST`
    /// of the others.
    fn voice(&self, held_out: bool) -> Vec<&Prepared> {
        let takes: Vec<&Prepared> = self.items[self.voice_from..self.tts_from]
            .iter()
            .filter(|p| {
                let session = p.speaker.rsplit('/').next().unwrap_or("");
                AUDIO_VOICE_VAL_SESSIONS.contains(&session) == held_out
            })
            .collect();
        if held_out {
            return takes;
        }
        takes[takes.len().saturating_sub(AUDIO_VOICE_LAST)..].to_vec()
    }

    fn tts(&self) -> &[Prepared] {
        &self.items[self.tts_from..]
    }

    /// Item `i`'s window: for training with the transcript before it when
    /// `prefix`, for validation never, and with the validation's longer text
    /// after it.
    fn window(&self, i: usize, train: bool, prefix: bool) -> Window {
        let (prev, next) = match self.around.get(&i) {
            Some((before, after)) => (Some(&before[..]), Some(&after[..])),
            None => (
                self.prev.get(i).copied().flatten().map(|j| &self.items[j].written[..]),
                self.next.get(i).copied().flatten().map(|j| &self.items[j].written[..]),
            ),
        };
        let prev = prev.filter(|_| train && prefix);
        let after = match (train, self.around.contains_key(&i)) {
            (true, true) => AUDIO_TRAIN_AFTER_WORDS,
            (true, false) => 0,
            (false, _) => AUDIO_CONTEXT_WORDS,
        };
        Window::new(prev, &self.items[i], next, after)
    }

    /// `items` loaded as one batch. A training batch is augmented and draws
    /// which windows get the transcript before them; a validation batch never
    /// does either.
    fn batch(&self, items: &[usize], train: bool) -> Batch<'_> {
        let windows = items
            .iter()
            .map(|&i| self.window(i, train, random_range(0.0..1.0) < AUDIO_PREFIX_P))
            .collect();
        let augment = self.augment.as_ref().filter(|_| train);
        let refreshed = |i: usize| train.then(|| self.realigned.lock().unwrap().get(&i).cloned()).flatten();
        let mut batch =
            Batch::with_times(items.iter().map(|&i| &self.items[i]).collect(), &self.stats, augment, &refreshed);
        batch.windows = windows;
        batch
    }

    /// Feature statistics over `STATS_UTTERANCES` training utterances spread
    /// over the whole set.
    fn train_stats(&self) -> FeatureStats {
        let train = self.train();
        let step = (train.len() / STATS_UTTERANCES).max(1);
        let identity = FeatureStats::identity();
        let feats: Vec<Vec<f32>> = train
            .par_iter()
            .step_by(step)
            .map(|p| p.features(&identity, None))
            .collect();
        let mut acc = StatsAccumulator::new();
        for f in &feats {
            acc.add(f);
        }
        acc.finish()
    }

    /// Sizes and the teacher on the held-out set; the training set's teacher
    /// runs beside every training window.
    fn report(&self, trainer: &mut E2Trainer) {
        let hours = |v: &[&Prepared]| v.iter().map(|p| p.seconds()).sum::<f64>() / 3600.0;
        let (train, val) = (self.train(), self.val());
        let (mut own, mut after, mut conts) = (0.0, 0.0, 0);
        for p in &val {
            let t = trainer.val_teacher(p.index, &self.window(p.index, false, false));
            own += t.own;
            if let Some((_, c)) = t.after {
                after += c;
                conts += 1;
            }
        }
        let (voice_train, voice_val) = (self.voice(false), self.voice(true));
        println!(
            "voice: train {} takes ({:.1} min), val {} ({:.1} min), {} with document text around them",
            voice_train.len(),
            60.0 * hours(&voice_train),
            voice_val.len(),
            60.0 * hours(&voice_val),
            self.around.len()
        );
        println!(
            "train {} utterances ({:.1} h), val {} ({:.1} h, {} speakers, {conts} with text after them); teacher next-word loss on val {:.3} on the utterances, {:.3} on the text after them",
            train.len(),
            hours(&train),
            val.len(),
            hours(&val),
            self.held_out.len(),
            own / val.len().max(1) as f32,
            after / conts.max(1) as f32
        );
    }

    /// Per-dimension RMS of the text encoder's embeddings of the words of
    /// `SCALE_UTTERANCES` training transcripts: where the backbone is used to
    /// reading its input. On the CPU, so it needs no room on the device.
    fn text_scale(&self, model_path: &str) -> Vec<f32> {
        let mut text = crate::hierarchical::Hierarchical::load(model_path, Utf8Tokenizer::new())
            .unwrap_or_else(|e| panic!("{model_path}: {e}"));
        text.make_cache(1, MAX_SEQ_LEN);
        let mut sum_sq: Vec<f64> = Vec::new();
        let mut words = 0;
        for p in self.train().into_iter().take(SCALE_UTTERANCES) {
            for w in &p.words {
                let toks: Vec<u16> = p.tokens[w.start..w.end].iter().map(|&t| t as u16).collect();
                let e = text.encoder.encode_word_sample(&toks);
                sum_sq.resize(e.len(), 0.0);
                for (s, &x) in sum_sq.iter_mut().zip(e) {
                    *s += (x as f64) * (x as f64);
                }
                words += 1;
            }
        }
        sum_sq
            .iter()
            .map(|s| (s / words as f64).sqrt() as f32)
            .collect()
    }

    /// Scratch sized for the longest utterances this corpus can batch.
    fn training_cache(&self, gpu: &Gpu, enc: &AudioEncoder) -> TrainingCache {
        let c = enc.cfg;
        let t_out = self
            .items
            .iter()
            .map(|p| p.t_in / c.stride)
            .max()
            .unwrap_or(1);
        let words = self.items.iter().map(|p| p.segments.len()).max().unwrap_or(1);
        println!("encoder scratch for {t_out} frames and {words} words per utterance");
        enc.training_cache(gpu, slowed(t_out) * AUDIO_BATCH, slowed(t_out), words * AUDIO_BATCH)
    }
}

/// The training items in a fresh random order every epoch.
struct Epochs {
    /// The items in the order they lie on disk.
    items: Vec<usize>,
    order: Vec<usize>,
    cursor: usize,
    /// Each epoch shuffles runs of this many consecutive items, not single
    /// ones: on a spinning disk, neighbouring takes are read without a seek.
    block: usize,
}

impl Epochs {
    fn new(order: Vec<usize>) -> Self {
        Self::blocked(order, 1)
    }

    fn blocked(items: Vec<usize>, block: usize) -> Self {
        let cursor = items.len();
        Self {
            order: items.clone(),
            items,
            cursor,
            block: block.max(1),
        }
    }

    /// The next `n` items, fewer at the end of an epoch.
    fn next_batch(&mut self, n: usize) -> Vec<usize> {
        if self.order.is_empty() {
            return Vec::new();
        }
        if self.cursor >= self.order.len() {
            if self.block == 1 {
                self.order.shuffle(&mut rand::rng());
            } else {
                let mut runs: Vec<&[usize]> = self.items.chunks(self.block).collect();
                runs.shuffle(&mut rand::rng());
                self.order = runs.concat();
            }
            self.cursor = 0;
        }
        let end = (self.cursor + n).min(self.order.len());
        let out = self.order[self.cursor..end].to_vec();
        self.cursor = end;
        out
    }
}

/// Utterances padded to one length, as the encoder takes them.
struct Batch<'a> {
    /// The items as heard: a speed-perturbed or realigned one is a copy.
    items: Vec<std::borrow::Cow<'a, Prepared>>,
    /// The speed each item was played at; empty when every one was as recorded.
    speeds: Vec<f32>,
    /// The text model's window around each of `items`; empty where only the
    /// audio is used.
    windows: Vec<Window>,
    feats: Vec<f32>,
    t_in: usize,
    /// For a chained batch, the encoder frame each utterance starts at in its
    /// single row; empty when every utterance is a row of its own.
    chain: Vec<usize>,
}

impl<'a> Batch<'a> {
    /// `items` heard as they are, or under `augment` each at a speed of its own,
    /// with the word times `times` gives for an item in place of its own.
    fn load(
        items: Vec<&'a Prepared>,
        stats: &FeatureStats,
        augment: Option<&Augment>,
        times: &(dyn Fn(usize) -> Option<Vec<(f32, f32)>> + Sync),
    ) -> (Vec<std::borrow::Cow<'a, Prepared>>, Vec<Vec<f32>>, Vec<f32>) {
        let speeds: Vec<f32> = items.iter().map(|_| augment.map_or(1.0, Augment::draw_speed)).collect();
        let loaded: Vec<(Vec<f32>, Prepared, bool)> = items
            .par_iter()
            .zip(&speeds)
            .map(|(p, &speed)| {
                let t = times(p.index);
                let (f, item) = p.features_at(stats, augment, speed, t.as_deref());
                (f, item, t.is_some())
            })
            .collect();
        let (items, feats) = loaded
            .into_iter()
            .zip(items)
            .zip(&speeds)
            .map(|(((f, changed, realigned), p), &speed)| {
                let item = if speed == 1.0 && !realigned {
                    std::borrow::Cow::Borrowed(p)
                } else {
                    std::borrow::Cow::Owned(changed)
                };
                (item, f)
            })
            .unzip();
        (items, feats, speeds)
    }

    fn new(items: Vec<&'a Prepared>, stats: &FeatureStats, augment: Option<&Augment>) -> Self {
        Self::with_times(items, stats, augment, &|_| None)
    }

    /// [`new`](Self::new) with refreshed word times (see [`Batch::load`]).
    fn with_times(
        items: Vec<&'a Prepared>,
        stats: &FeatureStats,
        augment: Option<&Augment>,
        times: &(dyn Fn(usize) -> Option<Vec<(f32, f32)>> + Sync),
    ) -> Self {
        let (items, loaded, speeds) = Self::load(items, stats, augment, times);
        let t_in = items.iter().map(|p| p.t_in).max().unwrap_or(0);
        let mut feats = vec![0.0; items.len() * t_in * AUDIO_FEATS];
        for (b, f) in loaded.iter().enumerate() {
            let at = b * t_in * AUDIO_FEATS;
            feats[at..at + f.len()].copy_from_slice(f);
        }
        Self {
            items,
            speeds,
            windows: Vec::new(),
            feats,
            t_in,
            chain: Vec::new(),
        }
    }

    /// `items` one after another as a single recording, so the encoder runs
    /// one state through all of them from a fresh start. Each keeps whole
    /// encoder frames of its own: `stride` feature frames each.
    fn chained(
        items: Vec<&'a Prepared>,
        stats: &FeatureStats,
        augment: Option<&Augment>,
        stride: usize,
    ) -> Self {
        let (items, loaded, speeds) = Self::load(items, stats, augment, &|_| None);
        let (mut feats, mut chain, mut frames) = (Vec::new(), Vec::new(), 0);
        for (p, f) in items.iter().zip(&loaded) {
            let own = p.t_in / stride;
            chain.push(frames);
            feats.extend_from_slice(&f[..own * stride * AUDIO_FEATS]);
            frames += own;
        }
        Self {
            items,
            speeds,
            windows: Vec::new(),
            feats,
            t_in: frames * stride,
            chain,
        }
    }

    /// Rows the encoder runs.
    fn rows(&self) -> usize {
        if self.chain.is_empty() { self.items.len() } else { 1 }
    }

    /// Utterance `b`'s first encoder frame among the batch's `rows · t_out`,
    /// and how many frames are its own.
    fn frames(&self, b: usize, t_out: usize, stride: usize) -> (usize, usize) {
        let own = self.items[b].t_in / stride;
        match self.chain.get(b) {
            Some(&start) => (start, own),
            None => (b * t_out, own),
        }
    }

    fn segments(&self) -> Vec<WordSegments> {
        if self.chain.is_empty() {
            return self.items.iter().map(|p| p.segments.clone()).collect();
        }
        let shifted = self.items.iter().zip(&self.chain).flat_map(|(p, &start)| {
            p.segments.0.iter().map(move |s| Segment {
                start: s.start + start,
                end: s.end + start,
            })
        });
        vec![WordSegments(shifted.collect())]
    }

    /// Boundary targets at the padded length, with no weight past each
    /// utterance's own end.
    fn targets(&self, t_out: usize, stride: usize, delay: usize) -> BoundaryTargets {
        if !self.chain.is_empty() {
            return BoundaryTargets::new(&self.segments()[0], t_out, delay, BOUNDARY_TOLERANCE);
        }
        BoundaryTargets::concat(self.items.iter().map(|p| {
            let mut bt = BoundaryTargets::new(&p.segments, t_out, delay, BOUNDARY_TOLERANCE);
            bt.weights[(p.t_in / stride) * (delay + 1)..].fill(0.0);
            bt
        }))
    }

    fn seconds(&self) -> f64 {
        self.items.iter().map(|p| p.seconds()).sum()
    }
}

/// Word ends found within the tolerance, for precision and recall.
#[derive(Default)]
struct BoundaryScore {
    true_pos: usize,
    predicted: usize,
    labelled: usize,
    /// Relative error of the boundary probability mass against the word
    /// count, summed over utterances.
    count_err: f32,
    counted: usize,
}

impl BoundaryScore {
    fn add(&mut self, decided: &[usize], segs: &WordSegments) {
        self.predicted += decided.len();
        self.labelled += segs.len();
        self.true_pos += segs
            .0
            .iter()
            .filter(|s| {
                decided
                    .iter()
                    .any(|&e| e.abs_diff(s.last()) <= BOUNDARY_TOLERANCE)
            })
            .count();
    }

    fn merge(&mut self, other: &Self) {
        self.true_pos += other.true_pos;
        self.predicted += other.predicted;
        self.labelled += other.labelled;
        self.count_err += other.count_err;
        self.counted += other.counted;
    }

    fn precision(&self) -> f32 {
        self.true_pos as f32 / self.predicted.max(1) as f32
    }

    fn recall(&self) -> f32 {
        self.true_pos as f32 / self.labelled.max(1) as f32
    }

    fn f1(&self) -> f32 {
        let (p, r) = (self.precision(), self.recall());
        if p + r > 0.0 {
            2.0 * p * r / (p + r)
        } else {
            0.0
        }
    }

    fn summary(&self) -> String {
        format!(
            "F1 {:.3} (precision {:.3}, recall {:.3}, {:.2} ends per word)",
            self.f1(),
            self.precision(),
            self.recall(),
            self.predicted as f32 / self.labelled.max(1) as f32
        )
    }
}

/// Next-word losses of the student, its control and the teacher, summed over
/// windows.
#[derive(Default)]
struct Losses {
    student: f32,
    control: f32,
    controls: usize,
    teacher: f32,
    teachers: usize,
    windows: usize,
}

impl Losses {
    fn add(&mut self, l: &WindowLosses) {
        self.student += l.student.iter().sum::<f32>();
        self.control += l.controls.iter().sum::<f32>();
        self.controls += l.controls.len();
        self.teacher += l.teachers.iter().sum::<f32>();
        self.teachers += l.teachers.len();
        self.windows += l.student.len();
    }

    fn summary(&self, rotated: &str) -> String {
        let w = self.windows.max(1) as f32;
        let control = if self.controls > 0 {
            format!(" ({rotated} {:.3})", self.control / self.controls as f32)
        } else {
            String::new()
        };
        if self.teachers == 0 {
            return format!("student {:.3}{control}", self.student / w);
        }
        format!(
            "student {:.3}{control} teacher {:.3} gap {:+.3}",
            self.student / w,
            self.teacher / w,
            (self.student - self.teacher) / w,
        )
    }
}

/// Losses summed over windows, and the boundary loss over batches.
#[derive(Default)]
struct Meter {
    /// Seconds loading batches and in the text model.
    load_s: f32,
    text_s: f32,
    /// Training: every trained prediction. Validation: the utterances' own
    /// words.
    own: Losses,
    /// Validation: the first word after the utterances, and all the text after
    /// them.
    next: Losses,
    after: Losses,
    bce: f32,
    batches: usize,
    boundaries: BoundaryScore,
    ctc: CtcScore,
    /// Heard words whose nearest text word in the batch is their own, of all.
    matched: (usize, usize),
    head_cut: (usize, usize),
}

impl Meter {
    fn add(&mut self, out: &WindowOut) {
        self.head_cut.0 += out.head_cut.0;
        self.head_cut.1 += out.head_cut.1;
        self.text_s += out.text_s;
        self.own.add(&out.own);
        self.next.add(&out.next);
        self.after.add(&out.after);
        self.bce += out.bce;
        self.batches += 1;
        self.boundaries.merge(&out.boundaries);
        self.ctc.merge(&out.ctc);
        self.matched.0 += out.matched.0;
        self.matched.1 += out.matched.1;
    }

    fn summary(&self) -> String {
        let after = if self.after.windows > 0 {
            format!(
                " | next word: {} | text after: {}",
                self.next.summary("audio rotated"),
                self.after.summary("audio rotated")
            )
        } else {
            String::new()
        };
        let matched = if self.matched.1 > 0 {
            format!(" | word match {:.1}%", 100.0 * self.matched.0 as f32 / self.matched.1 as f32)
        } else {
            String::new()
        };
        format!(
            "{}{after} | boundary bce {:.3} F1 {:.3} count ±{:.1}% | {}{matched}",
            self.own.summary("words rotated"),
            self.bce / self.batches.max(1) as f32,
            self.boundaries.f1(),
            100.0 * self.boundaries.count_err / self.boundaries.counted.max(1) as f32,
            self.ctc.summary()
        )
    }
}

/// One loss per window: the student's, the control's (its audio words
/// rotated, validation only) and the teacher's, reading the transcript.
#[derive(Default)]
struct WindowLosses {
    student: Vec<f32>,
    controls: Vec<f32>,
    teachers: Vec<f32>,
}

/// What one batch produced.
#[derive(Default)]
struct WindowOut {
    /// Seconds spent in the text model.
    text_s: f32,
    /// Utterances pooled at the boundary head's own cut, of all in the batch.
    head_cut: (usize, usize),
    /// Items the character head heard well enough to align again, with their
    /// word times.
    realigned: Vec<(usize, Vec<(f32, f32)>)>,
    /// Heard words whose nearest text word in the batch is their own, of all.
    matched: (usize, usize),
    own: WindowLosses,
    next: WindowLosses,
    after: WindowLosses,
    bce: f32,
    boundaries: BoundaryScore,
    ctc: CtcScore,
}

/// The character head: CTC loss per character and greedy-decoded errors.
#[derive(Default)]
struct CtcScore {
    loss: f32,
    /// Characters of the utterances `loss` covers.
    loss_chars: usize,
    errors: usize,
    chars: usize,
    /// Utterances whose transcript did not fit in their CTC steps.
    skipped: usize,
}

impl CtcScore {
    fn merge(&mut self, other: &Self) {
        self.loss += other.loss;
        self.loss_chars += other.loss_chars;
        self.errors += other.errors;
        self.chars += other.chars;
        self.skipped += other.skipped;
    }

    fn summary(&self) -> String {
        let skipped = if self.skipped > 0 {
            format!(" ({} skipped)", self.skipped)
        } else {
            String::new()
        };
        format!(
            "ctc {:.3}/char CER {:.1}%{skipped}",
            self.loss / self.loss_chars.max(1) as f32,
            100.0 * self.errors as f32 / self.chars.max(1) as f32
        )
    }
}

/// The contrastive term between heard words and text words: for each heard
/// word, a softmax over the cosines (at temperature `EMBED_CONTRAST_TEMP`) to
/// the text embeddings of every word heard in the batch, trained towards its
/// own; other occurrences of the same word are left out of its softmax. With
/// `d`, adds `EMBED_CONTRAST_WEIGHT` times the gradient of the mean loss to
/// the rows of `d`. Returns how many heard words have their own text word
/// nearest, of all.
fn contrast(
    embeds: &[f32],
    hc: usize,
    words: &[(usize, Vec<f32>, String)],
    d: Option<&mut Vec<f32>>,
) -> (usize, usize) {
    let n = words.len();
    if n < 2 {
        return (0, 0);
    }
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    let audio: Vec<&[f32]> = words.iter().map(|(r, _, _)| &embeds[r * hc..(r + 1) * hc]).collect();
    let a_norm: Vec<f32> = audio.iter().map(|a| norm(a)).collect();
    let t_norm: Vec<f32> = words.iter().map(|(_, t, _)| norm(t)).collect();
    let cos: Vec<Vec<f32>> = (0..n)
        .into_par_iter()
        .map(|i| {
            (0..n)
                .map(|j| {
                    let dot: f32 = audio[i].iter().zip(&words[j].1).map(|(a, t)| a * t).sum();
                    dot / (a_norm[i] * t_norm[j])
                })
                .collect()
        })
        .collect();
    let others = |i: usize, j: usize| j == i || words[j].2 != words[i].2;
    let hits = (0..n)
        .filter(|&i| (0..n).filter(|&j| others(i, j)).all(|j| cos[i][j] <= cos[i][i]))
        .count();
    if let Some(d) = d {
        let tau = EMBED_CONTRAST_TEMP;
        let g_scale = EMBED_CONTRAST_WEIGHT / (n as f32 * tau);
        for i in 0..n {
            let logits: Vec<f32> = (0..n).map(|j| if others(i, j) { cos[i][j] / tau } else { f32::NEG_INFINITY }).collect();
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
            let sum: f32 = exp.iter().sum();
            let row = words[i].0;
            for j in 0..n {
                let w = exp[j] / sum - if j == i { 1.0 } else { 0.0 };
                if w == 0.0 {
                    continue;
                }
                // ∂cos(a, t)/∂a = t / (|a||t|) − cos · a / |a|².
                let t = &words[j].1;
                for k in 0..hc {
                    let grad = t[k] / (a_norm[i] * t_norm[j]) - cos[i][j] * audio[i][k] / (a_norm[i] * a_norm[i]);
                    d[row * hc + k] += g_scale * w * grad;
                }
            }
        }
    }
    (hits, n)
}

/// CTC over each utterance's own frames of `logits` (`[batch · t_out, ctc]`):
/// the score, and the gradient averaged over utterances and characters.
fn ctc_batch(logits: &[f32], batch: &Batch, t_out: usize, c: AudioCfg) -> (CtcScore, Vec<f32>) {
    let (score, grad, _) = ctc_batch_aligned(logits, batch, t_out, c);
    (score, grad)
}

/// [`ctc_batch`], and for each item the character head heard well enough
/// (`AUDIO_REALIGN_MAX_CER`) its words' `(start, end)` in seconds as heard,
/// forced-aligned on the same logits.
fn ctc_batch_aligned(
    logits: &[f32],
    batch: &Batch,
    t_out: usize,
    c: AudioCfg,
) -> (CtcScore, Vec<f32>, Vec<Option<Vec<(f32, f32)>>>) {
    let ctc = Ctc {
        vocab: CtcAlphabet::SIZE,
    };
    let per_frame = c.ctc / CtcAlphabet::SIZE;
    let scale = 1.0 / batch.items.len() as f32;
    let per_item: Vec<(CtcScore, Option<Vec<f32>>, Option<Vec<(f32, f32)>>)> = batch
        .items
        .par_iter()
        .enumerate()
        .map(|(b, p)| {
            let (first, own) = batch.frames(b, t_out, c.stride);
            let rows = &logits[first * c.ctc..(first + own) * c.ctc];
            let heard = ctc.greedy(rows);
            let mut score = CtcScore {
                errors: edit_distance(&heard, &p.chars),
                chars: p.chars.len(),
                ..Default::default()
            };
            let heard_well = (score.errors as f32) <= AUDIO_REALIGN_MAX_CER * score.chars as f32;
            let times = heard_well
                .then(|| ctc.align(rows, &p.chars))
                .flatten()
                .map(|last| {
                    let seconds = (p.t_in * AUDIO_HOP) as f32 / AUDIO_SR as f32;
                    let mut start = 0.0;
                    CtcAlphabet::word_ends(&p.chars)
                        .iter()
                        .map(|&k| {
                            let end = ((last[k] / per_frame + 1) as f32 * ENC_FRAME_S).min(seconds);
                            let word = (start, end);
                            start = end;
                            word
                        })
                        .collect()
                });
            match ctc.loss_and_grad(rows, &p.chars) {
                Some((loss, grad)) => {
                    score.loss = loss;
                    score.loss_chars = p.chars.len();
                    let g = scale / p.chars.len() as f32;
                    (score, Some(grad.into_iter().map(|x| x * g).collect()), times)
                }
                None => {
                    score.skipped = 1;
                    (score, None, times)
                }
            }
        })
        .collect();
    let mut total = CtcScore::default();
    let mut grad = vec![0.0; logits.len()];
    let mut aligned = Vec::with_capacity(per_item.len());
    for (b, (score, g, times)) in per_item.into_iter().enumerate() {
        aligned.push(times);
        total.merge(&score);
        if let Some(g) = g {
            let at = batch.frames(b, t_out, c.stride).0 * c.ctc;
            grad[at..at + g.len()].copy_from_slice(&g);
        }
    }
    (total, grad, aligned)
}

/// The teacher on one held-out window: the utterance, and when text follows
/// it, the first word after it and all of that text.
#[derive(Clone, Copy)]
struct ValTeacher {
    own: f32,
    after: Option<(f32, f32)>,
}

struct E2Trainer {
    gpu: Gpu,
    model: Hierarchical,
    enc: AudioEncoder,
    cache: TrainingCache,
    bce: Bce,
    /// The text model's next-word loss reading each held-out item's window, on
    /// the utterance and on the text after it, once it has been measured.
    teacher: Vec<Option<ValTeacher>>,
    /// Relative backbone-context distance to the reading, summed since the last log.
    state_mse: f32,
    state_n: usize,
    /// Relative distance of the audio words' embeddings to the text encoder's,
    /// summed since the last log.
    embed_mse: f32,
    embed_n: usize,
}

impl E2Trainer {
    fn run(&mut self, corpus: &Corpus, ck: &mut AudioCheckpoint, save_path: &str) {
        let mut epochs = Epochs::new(corpus.train().iter().map(|p| p.index).collect());
        let mut voice = Epochs::new(corpus.voice(false).iter().map(|p| p.index).collect());
        let voice_share = AUDIO_VOICE_EPOCHS * voice.order.len() as f32 / AUDIO_E2_STEPS as f32;
        let mut sets: Vec<Epochs> = corpus
            .take_sets
            .iter()
            .map(|r| {
                let held_in = corpus.items[r.clone()].iter().filter(|p| !tts_held_out(p)).map(|p| p.index).collect();
                Epochs::blocked(held_in, TAKE_BLOCK)
            })
            .filter(|e| !e.order.is_empty())
            .collect();
        let tts_n = if sets.is_empty() { 0 } else { (AUDIO_TAKE_SHARE * AUDIO_BATCH as f32).round() as usize };
        // A set's share of the take slots grows with its size to the power
        // `AUDIO_TAKE_SET_POWER`: a small set is not repeated many times over
        // while a large one goes half unheard.
        let weights: Vec<f32> = sets.iter().map(|e| (e.order.len() as f32).powf(AUDIO_TAKE_SET_POWER)).collect();
        let total: f32 = weights.iter().sum();
        for ((w, e), dir) in weights.iter().zip(&sets).zip(AUDIO_TAKE_DIRS) {
            let draws = w / total * (tts_n * AUDIO_E2_STEPS) as f32;
            println!("  {dir}: {} takes, {:.1} epochs over the run", e.order.len(), draws / e.order.len() as f32);
        }
        let mut next_items = move || {
            let voice_n = usize::from(random_range(0.0..1.0) < voice_share);
            let mut items = epochs.next_batch(AUDIO_BATCH - voice_n - tts_n);
            items.extend(voice.next_batch(voice_n));
            for _ in 0..tts_n {
                let mut x = random_range(0.0..total);
                let k = weights.iter().position(|w| {
                    x -= w;
                    x < 0.0
                });
                let k = k.unwrap_or(sets.len() - 1);
                items.extend(sets[k].next_batch(1));
            }
            items
        };
        let tts_val: Vec<usize> =
            corpus.tts().iter().filter(|p| tts_held_out(p)).take(TTS_VAL).map(|p| p.index).collect();
        let val: Vec<usize> = corpus.val().iter().map(|p| p.index).collect();
        let voice_val: Vec<usize> = corpus.voice(true).iter().map(|p| p.index).collect();
        let mut adam = AdamCfg::new(AUDIO_LR, 0.0);
        let mut meter = Meter::default();
        let t_run = Instant::now();
        let mut t_log = Instant::now();
        if ck.step > 0 {
            println!(
                "  val       | {} (at resume)",
                self.validate(corpus, &val).summary()
            );
            if !voice_val.is_empty() {
                println!("  val voice | {}", self.validate(corpus, &voice_val).summary());
            }
        }
        if !tts_val.is_empty() {
            println!("  val tts   | {}", self.validate(corpus, &tts_val).summary());
        }
        // The next `LOAD_AHEAD` batches decode and featurize on the CPU while
        // the GPU runs this one: one loader alone took longer than a step.
        std::thread::scope(|scope| {
            let mut ahead = std::collections::VecDeque::new();
            for _ in 0..LOAD_AHEAD {
                let items = next_items();
                ahead.push_back(scope.spawn(move || corpus.batch(&items, true)));
            }
            for step in ck.step as usize + 1..=AUDIO_E2_STEPS {
                let t_load = Instant::now();
                let mut batch = ahead.pop_front().unwrap().join().expect("batch loader panicked");
                meter.load_s += t_load.elapsed().as_secs_f32();
                let upcoming = next_items();
                ahead.push_back(scope.spawn(move || corpus.batch(&upcoming, true)));

                let out = self.window(&mut batch, true, corpus.voice_from);
                {
                    let mut realigned = corpus.realigned.lock().unwrap();
                    for (i, times) in &out.realigned {
                        if *i >= corpus.voice_from {
                            realigned.insert(*i, times.clone());
                        }
                    }
                }
                adam.t += 1;
                adam.lr = Self::lr(step);
                self.enc.step(&self.gpu, &adam);
                ck.step = step as u64;
                ck.seconds += batch.seconds();
                meter.add(&out);

                if step % LOG_EVERY == 0 {
                    let wall = t_log.elapsed().as_secs_f32();
                    let state = std::mem::take(&mut self.state_mse) / std::mem::take(&mut self.state_n).max(1) as f32;
                    let embed = std::mem::take(&mut self.embed_mse) / std::mem::take(&mut self.embed_n).max(1) as f32;
                    println!(
                        "step {step:5} | {} | state {state:.3} embed {embed:.3} | head cut {:.0}% | realigned {} | lr {:.1e} | {:.0} ms/step (load {:.0}%, text model {:.0}%) | {:.1?}",
                        meter.summary(),
                        100.0 * meter.head_cut.0 as f32 / meter.head_cut.1.max(1) as f32,
                        corpus.realigned.lock().unwrap().len(),
                        adam.lr,
                        1000.0 * wall / LOG_EVERY as f32,
                        100.0 * meter.load_s / wall,
                        100.0 * meter.text_s / wall,
                        t_run.elapsed()
                    );
                    meter = Meter::default();
                    t_log = Instant::now();
                }
                if step % AUDIO_VAL_EVERY == 0 || step == AUDIO_E2_STEPS {
                    println!("  val       | {}", self.validate(corpus, &val).summary());
                    if !voice_val.is_empty() {
                        println!("  val voice | {}", self.validate(corpus, &voice_val).summary());
                    }
                    if !tts_val.is_empty() {
                        println!("  val tts   | {}", self.validate(corpus, &tts_val).summary());
                    }
                    match self.enc.save(&self.gpu, save_path, ck) {
                        Ok(()) => println!("  saved {save_path} (step {step})"),
                        Err(e) => eprintln!("  save failed: {e}"),
                    }
                }
            }
        });
    }

    fn validate(&mut self, corpus: &Corpus, val: &[usize]) -> Meter {
        let mut meter = Meter::default();
        for chunk in val.chunks(AUDIO_BATCH) {
            let out = self.window(&mut corpus.batch(chunk, false), false, usize::MAX);
            meter.add(&out);
        }
        meter
    }

    /// The same utterances scored with their words cut three ways: at the
    /// MFA alignment, at this encoder's own CTC forced alignment (how the takes
    /// are cut for training), and where its boundary head puts the known
    /// number of words (how a take is cut live). What each way costs is what
    /// the segmentation costs.
    /// The spoken words of `items` the character head gets wrong most often:
    /// each greedy decode aligned word by word against what was said, every
    /// said word that was replaced or dropped counted. What the user's
    /// recording prompts should contain more of.
    /// Every said word's count heard wrong and said also goes to `table` as
    /// `word\twrong\tsaid` lines, the input of `scripts/error_prompts.py`.
    fn misheard_report(&mut self, corpus: &Corpus, items: &[usize], table: &str) {
        let c = self.enc.cfg;
        let ctc = Ctc {
            vocab: CtcAlphabet::SIZE,
        };
        let (mut wrong, mut said): (HashMap<String, usize>, HashMap<String, usize>) = (HashMap::new(), HashMap::new());
        for chunk in items.chunks(AUDIO_BATCH) {
            let refs: Vec<&Prepared> = chunk.iter().map(|&i| &corpus.items[i]).collect();
            let batch = Batch::new(refs, &corpus.stats, None);
            let none = vec![WordSegments::default(); chunk.len()];
            let out = self.enc.forward(&self.gpu, &batch.feats, chunk.len(), batch.t_in, &none, &self.cache);
            let chars = out.ctc.to_host(&self.gpu).data;
            self.enc.discard();
            for (b, p) in batch.items.iter().enumerate() {
                let own = p.t_in / c.stride;
                let heard = CtcAlphabet::decode(&ctc.greedy(&chars[b * out.t_out * c.ctc..(b * out.t_out + own) * c.ctc]));
                let reference = CtcAlphabet::decode(&p.chars);
                let (r, h): (Vec<&str>, Vec<&str>) = (reference.split_whitespace().collect(), heard.split_whitespace().collect());
                // Word-level edit distance; walking back marks each said word kept or not.
                let mut d = vec![vec![0usize; h.len() + 1]; r.len() + 1];
                for i in 0..=r.len() {
                    for j in 0..=h.len() {
                        d[i][j] = if i == 0 || j == 0 {
                            i + j
                        } else {
                            (d[i - 1][j - 1] + usize::from(r[i - 1] != h[j - 1])).min(d[i - 1][j] + 1).min(d[i][j - 1] + 1)
                        };
                    }
                }
                let (mut i, mut j) = (r.len(), h.len());
                while i > 0 {
                    if j > 0 && d[i][j] == d[i - 1][j - 1] + usize::from(r[i - 1] != h[j - 1]) {
                        if r[i - 1] != h[j - 1] {
                            *wrong.entry(r[i - 1].to_string()).or_default() += 1;
                        }
                        i -= 1;
                        j -= 1;
                    } else if j > 0 && d[i][j] == d[i][j - 1] + 1 {
                        j -= 1;
                    } else {
                        *wrong.entry(r[i - 1].to_string()).or_default() += 1;
                        i -= 1;
                    }
                }
                for w in &r {
                    *said.entry(w.to_string()).or_default() += 1;
                }
            }
        }
        let total: usize = said.values().sum();
        let missed: usize = wrong.values().sum();
        println!("    {missed} of {total} said words heard wrong ({:.1}%); most often:", 100.0 * missed as f32 / total.max(1) as f32);
        let mut ranked: Vec<(&String, &usize)> = wrong.iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        let line: Vec<String> = ranked.iter().take(60).map(|(w, n)| format!("{w} {n}/{}", said[*w])).collect();
        for row in line.chunks(10) {
            println!("      {}", row.join(", "));
        }
        let mut rows: Vec<(&String, &usize)> = said.iter().collect();
        rows.sort();
        let tsv: String = rows.iter().map(|(w, n)| format!("{w}\t{}\t{n}\n", wrong.get(*w).unwrap_or(&0))).collect();
        if let Err(e) = std::fs::write(table, tsv) {
            println!("    {table}: {e}");
        }
    }

    fn segmentation_report(&mut self, corpus: &Corpus, items: &[usize]) {
        let c = self.enc.cfg;
        let ctc = Ctc {
            vocab: CtcAlphabet::SIZE,
        };
        let per_frame = c.ctc / CtcAlphabet::SIZE;
        let cols = c.delay + 1;
        let aligner = CountAligner {
            delay: c.delay,
            from_col: 0,
            off_peak: 3.0,
        };
        let prior = DurationPrior {
            sigma: 0.5,
            weight: 1.0,
            max_frames: 100,
        };
        let (mut by_ctc, mut by_head) = (HashMap::new(), HashMap::new());
        for chunk in items.chunks(AUDIO_BATCH) {
            let refs: Vec<&Prepared> = chunk.iter().map(|&i| &corpus.items[i]).collect();
            let batch = Batch::new(refs, &corpus.stats, None);
            let none = vec![WordSegments::default(); chunk.len()];
            let out = self.enc.forward(&self.gpu, &batch.feats, chunk.len(), batch.t_in, &none, &self.cache);
            let boundary = out.boundary.to_host(&self.gpu).data;
            let chars = out.ctc.to_host(&self.gpu).data;
            self.enc.discard();
            for (b, p) in batch.items.iter().enumerate() {
                let own = p.t_in / c.stride;
                let seconds = (p.t_in * AUDIO_HOP) as f32 / AUDIO_SR as f32;
                let contiguous = |ends: Vec<f32>| -> Vec<(f32, f32)> {
                    let mut start = 0.0;
                    ends.into_iter()
                        .map(|e| {
                            let w = (start, e.min(seconds));
                            start = w.1;
                            w
                        })
                        .collect()
                };
                let rows = &chars[b * out.t_out * c.ctc..(b * out.t_out + own) * c.ctc];
                if let Some(last) = ctc.align(rows, &p.chars) {
                    let ends = CtcAlphabet::word_ends(&p.chars)
                        .iter()
                        .map(|&k| (last[k] / per_frame + 1) as f32 * ENC_FRAME_S)
                        .collect();
                    by_ctc.insert(p.index, contiguous(ends));
                }
                let logits = &boundary[b * out.t_out * cols..(b * out.t_out + own) * cols];
                let lens: Vec<usize> = p.words.iter().map(|r| r.end - r.start).collect();
                if let Some(ends) = aligner.align_words(logits, own, &lens, prior) {
                    by_head.insert(p.index, contiguous(ends.iter().map(|&e| (e + 1) as f32 * ENC_FRAME_S).collect()));
                }
            }
        }
        for (name, cut) in [("MFA", None), ("CTC aligned", Some(&by_ctc)), ("boundary head", Some(&by_head))] {
            let mut meter = Meter::default();
            for chunk in items.chunks(AUDIO_BATCH) {
                let refs: Vec<&Prepared> = chunk.iter().map(|&i| &corpus.items[i]).collect();
                let times = |i: usize| cut.and_then(|m: &HashMap<usize, Vec<(f32, f32)>>| m.get(&i).cloned());
                let mut batch = Batch::with_times(refs, &corpus.stats, None, &times);
                batch.windows = chunk.iter().map(|&i| corpus.window(i, false, false)).collect();
                meter.add(&self.window(&mut batch, false, usize::MAX));
            }
            let covered = cut.map_or(items.len(), |m| m.len());
            println!("    {name:13} ({covered} of {}) | {}", items.len(), meter.summary());
        }
    }

    /// How much of what an utterance tells the text model about the words
    /// after it arrives through the audio. At each position `k` of
    /// `CONTEXT_AT` after the utterance, the next-word loss from the text after
    /// it alone (no context), with the transcript (teacher), with the audio
    /// (student) and with the audio words in scrambled order; "recovered" is
    /// the share of the context's benefit the audio delivers,
    /// (none − student) / (none − teacher).
    fn context_report(&mut self, corpus: &Corpus, items: &[usize]) {
        let last = *CONTEXT_AT.last().unwrap();
        // [none, teacher, student, scrambled] per position.
        let mut sums = [[0.0f64; 4]; CONTEXT_AT.len()];
        let mut windows = 0;
        let hc = self.enc.cfg.out;
        for chunk in items.chunks(AUDIO_BATCH) {
            let batch = corpus.batch(chunk, false);
            let out = self.enc.forward(
                &self.gpu,
                &batch.feats,
                batch.items.len(),
                batch.t_in,
                &batch.segments(),
                &self.cache,
            );
            let embeds = out
                .embeddings
                .as_ref()
                .expect("every utterance has words")
                .to_host(&self.gpu)
                .data;
            self.enc.discard();
            let mut row = 0;
            for w in &batch.windows {
                let n = w.audio;
                let audio = embeds[row * hc..(row + n) * hc].to_vec();
                row += n;
                if w.cont < last {
                    continue;
                }
                windows += 1;
                let text = self.text_rows(w);
                let scrambled: Vec<f32> = (0..n)
                    .flat_map(|i| {
                        let j = (i + n / 2) % n;
                        audio[j * hc..(j + 1) * hc].to_vec()
                    })
                    .collect();
                let heard = self.given(w, &audio, &text);
                let scrambled = self.given(w, &scrambled, &text);
                // The text after the utterance as a window of its own.
                let start = w.prefix + w.audio;
                let off = w.words[start].start;
                let tail_tokens = w.tokens[off..].to_vec();
                let tail_words: Vec<Range<usize>> = w.words[start..]
                    .iter()
                    .map(|r| Range::from(r.start - off..r.end - off))
                    .collect();
                for (i, &k) in CONTEXT_AT.iter().enumerate() {
                    // Decoded word `d` predicts word `d + 1`.
                    let mut mask = vec![false; w.decoded()];
                    mask[start + k - 2] = true;
                    let none = if k >= 2 {
                        let mut m = vec![false; tail_words.len() - 1];
                        m[k - 2] = true;
                        self.model
                            .eval_loss_masked(&self.gpu, &tail_tokens, &tail_words, Some(&m));
                        self.model.last_word_loss()
                    } else {
                        0.0
                    };
                    let row = [
                        none,
                        self.teacher_on(w, &mask),
                        self.student_on(&heard, w, &mask),
                        self.student_on(&scrambled, w, &mask),
                    ];
                    for (s, x) in sums[i].iter_mut().zip(row) {
                        *s += x as f64;
                    }
                }
            }
        }
        let n = windows.max(1) as f64;
        println!("    context over {windows} windows with ≥ {last} words after the utterance:");
        println!("      k   no context  teacher  student  scrambled  recovered");
        for (i, &k) in CONTEXT_AT.iter().enumerate() {
            let [none, teacher, student, scrambled] = sums[i].map(|s| s / n);
            if k < 2 {
                println!("      {k:2}       —      {teacher:7.3}  {student:7.3}  {scrambled:9.3}      —");
            } else {
                println!(
                    "      {k:2}  {none:9.3}  {teacher:7.3}  {student:7.3}  {scrambled:9.3}  {:7.0}%",
                    100.0 * (none - student) / (none - teacher)
                );
            }
        }
    }

    /// The teacher on held-out item `item`'s window `w`.
    fn val_teacher(&mut self, item: usize, w: &Window) -> ValTeacher {
        if let Some(t) = self.teacher[item] {
            return t;
        }
        let t = ValTeacher {
            own: self.teacher_on(w, &w.audio_mask()),
            after: (w.cont > 0)
                .then(|| (self.teacher_on(w, &w.next_mask()), self.teacher_on(w, &w.cont_mask()))),
        };
        self.teacher[item] = Some(t);
        t
    }

    fn teacher_on(&mut self, w: &Window, mask: &[bool]) -> f32 {
        let mask = w.teacher_mask(mask);
        self.model
            .eval_loss_masked(&self.gpu, &w.teacher_tokens, &w.teacher_words, Some(&mask));
        self.model.last_word_loss()
    }

    /// The student reading `given`, scored on `mask`.
    fn student_on(&mut self, given: &GTensor<f32>, w: &Window, mask: &[bool]) -> f32 {
        self.model
            .eval_loss_embeds(&self.gpu, given, &w.tokens, &w.words, Some(mask));
        self.model.last_word_loss()
    }

    /// The text encoder's embeddings of the text words of `w` the backbone
    /// reads, `[prefix + cont − 1, HC]` flattened: the transcript before the
    /// audio, then every word after it but the last.
    fn text_rows(&mut self, w: &Window) -> Vec<f32> {
        let mut words: Vec<Range<usize>> = w.words[..w.prefix].to_vec();
        if w.cont > 0 {
            words.extend_from_slice(&w.words[w.prefix + w.audio..]);
        } else if w.prefix > 0 {
            // Stands in as the last word, which is not encoded.
            words.push(w.words[w.prefix]);
        }
        if words.len() < 2 {
            return Vec::new();
        }
        self.model
            .encode_embeds(&self.gpu, &w.tokens, &words)
            .to_host(&self.gpu)
            .data
    }

    /// The text encoder's embedding of each of `w`'s audio words, as written.
    fn word_rows(&mut self, w: &Window) -> Vec<f32> {
        let mut words: Vec<Range<usize>> = w.words[w.prefix..w.prefix + w.audio].to_vec();
        // Stands in as the last word, which is not encoded.
        words.push(w.words[w.prefix]);
        self.model
            .encode_embeds(&self.gpu, &w.tokens, &words)
            .to_host(&self.gpu)
            .data
    }

    /// The backbone's input for `w`: text rows around the audio rows.
    fn given(&self, w: &Window, audio: &[f32], text: &[f32]) -> GTensor<f32> {
        let hc = self.enc.cfg.out;
        let mut rows = Vec::with_capacity(w.decoded() * hc);
        rows.extend_from_slice(&text[..w.prefix * hc]);
        rows.extend_from_slice(&audio[..w.audio_rows() * hc]);
        rows.extend_from_slice(&text[w.prefix * hc..]);
        assert_eq!(rows.len(), w.decoded() * hc, "window rows do not add up");
        GTensor::from_host(&self.gpu, &Tensor::new(&[w.decoded(), hc], rows))
    }

    /// Linear warmup, then cosine decay to zero at `AUDIO_E2_STEPS`.
    fn lr(step: usize) -> f32 {
        let warm = (step as f32 / AUDIO_WARMUP_STEPS as f32).min(1.0);
        let progress = step as f32 / AUDIO_E2_STEPS as f32;
        AUDIO_LR * warm * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
    }

    /// One batch through the encoder and the text model; with `train`, the
    /// gradient back into the encoder. A training batch aligns the words of
    /// every item from `realign_from` on with this step's own character head
    /// where it heard the item well enough, and reads all its windows as one
    /// packed sequence; validation reads them one by one, as aligned on load.
    fn window(&mut self, batch: &mut Batch, train: bool, realign_from: usize) -> WindowOut {
        let c = self.enc.cfg;
        let hc = c.out;
        let cols = c.delay + 1;
        let feats = std::mem::take(&mut batch.feats);
        let (rows, t_in) = (batch.rows(), batch.t_in);
        let mut aligned = Vec::new();
        let mut head_cut = 0;
        let mut ctc_out = (CtcScore::default(), Vec::new());
        let (out, ctc_logits) = self.enc.forward_aligned(&self.gpu, &feats, rows, t_in, &self.cache, |logits, boundary, t_out| {
            let (score, grad, heard) = ctc_batch_aligned(logits, batch, t_out, c);
            ctc_out = (score, grad);
            for (b, times) in heard.into_iter().enumerate() {
                let Some(times) = times.filter(|_| train && AUDIO_ALIGN_ONLINE && batch.items[b].index >= realign_from) else {
                    continue;
                };
                let own = batch.items[b].t_in / c.stride;
                let segments = WordSegments::from_word_times(&times, own);
                if segments.len() != batch.items[b].segments.len() {
                    continue;
                }
                let speed = batch.speeds.get(b).copied().unwrap_or(1.0);
                aligned.push((batch.items[b].index, times.iter().map(|&(s, e)| (s * speed, e * speed)).collect()));
                let item = batch.items[b].to_mut();
                (item.segments, item.times) = (segments, times);
            }
            // Some words are pooled as the boundary head cuts them, as they
            // will be live; its own targets stay the reference cut.
            let (pooled, n) = if train && batch.chain.is_empty() {
                head_cut_segments(batch, boundary, t_out, c)
            } else {
                (batch.segments(), 0)
            };
            head_cut = n;
            pooled
        });
        batch.feats = feats;
        let (ctc, d_ctc) = ctc_out;
        let logits = out.boundary.to_host(&self.gpu).data;
        let (bce, mut d_logits) = self
            .bce
            .loss_and_grad(&logits, &batch.targets(out.t_out, c.stride, c.delay));
        let count = CountLoss {
            tolerance: BOUNDARY_TOLERANCE,
            weight: BOUNDARY_COUNT_WEIGHT,
        };

        let mut boundaries = BoundaryScore::default();
        let decoder = PeakDecoder {
            delay: c.delay,
            threshold: BOUNDARY_THRESHOLD,
            radius: PEAK_RADIUS,
        };
        for (b, p) in batch.items.iter().enumerate() {
            let own = p.t_in / c.stride;
            let at = b * out.t_out * cols;
            let rows = &logits[at..at + own * cols];
            boundaries.add(&decoder.decide(rows, own), &p.segments);
            let scale = 1.0 / batch.items.len() as f32;
            let err = count.add(rows, cols, own, p.words.len(), scale, &mut d_logits[at..]);
            boundaries.count_err += err.abs();
            boundaries.counted += 1;
        }

        let embeds = out
            .embeddings
            .as_ref()
            .expect("every utterance has words")
            .to_host(&self.gpu)
            .data;
        let mut d_embeds = vec![0.0; embeds.len()];
        let mut own = WindowLosses::default();
        let mut next = WindowLosses::default();
        let mut after = WindowLosses::default();
        let t_text = Instant::now();
        let heard_words = if train {
            self.train_text(batch, &embeds, &mut d_embeds, &mut own)
        } else {
            self.validate_text(batch, &embeds, &mut own, &mut next, &mut after)
        };
        let matched = contrast(&embeds, hc, &heard_words, train.then_some(&mut d_embeds));
        let text_s = t_text.elapsed().as_secs_f32();

        let n_embeds = embeds.len() / hc;
        if train {
            let d_logits = GTensor::from_host(
                &self.gpu,
                &Tensor::new(&[logits.len() / cols, cols], d_logits),
            );
            let d_e = GTensor::from_host(&self.gpu, &Tensor::new(&[n_embeds, hc], d_embeds));
            let d_ctc = GTensor::from_host(
                &self.gpu,
                &Tensor::new(&[ctc_logits.len() / c.ctc, c.ctc], d_ctc),
            );
            self.enc.backward(
                &self.gpu,
                &d_logits,
                Some(&d_e),
                Some((&d_ctc, CTC_WEIGHT)),
                &self.cache,
            );
        } else {
            self.enc.discard();
        }
        WindowOut {
            head_cut: (head_cut, batch.items.len()),
            realigned: aligned,
            matched,
            text_s,
            own,
            next,
            after,
            bce,
            boundaries,
            ctc,
        }
    }

    /// The training windows of `batch` packed into sequences of at most
    /// `PACK_WORDS` words (see [`train_pack`](Self::train_pack)), each pack's
    /// gradient weighted by its share of the batch's words. Returns each heard
    /// word with the text encoder's embedding of it, for the contrastive term.
    fn train_text(
        &mut self,
        batch: &Batch,
        embeds: &[f32],
        d_embeds: &mut [f32],
        own: &mut WindowLosses,
    ) -> Vec<(usize, Vec<f32>, String)> {
        let total: usize = batch.windows.iter().map(|w| w.words.len()).sum();
        // Each window with the row of `embeds` its audio words start at.
        let mut at = 0;
        let placed: Vec<(&Window, usize)> = batch
            .windows
            .iter()
            .map(|w| {
                at += w.audio;
                (w, at - w.audio)
            })
            .collect();
        // Takes read out of a document distill at `DISTILL_ALPHA`; windows with
        // nothing after them (books, commands, speeches) at
        // `DISTILL_ALPHA_PLAIN`, so their genre is nothing to learn.
        let mut heard = Vec::new();
        for (documents, alpha) in [(true, DISTILL_ALPHA), (false, DISTILL_ALPHA_PLAIN)] {
            let class: Vec<(&Window, usize)> = placed.iter().copied().filter(|(w, _)| (w.cont > 0) == documents).collect();
            let mut from = 0;
            while from < class.len() {
                let mut to = from + 1;
                let mut words = class[from].0.words.len();
                while to < class.len() && words + class[to].0.words.len() <= PACK_WORDS {
                    words += class[to].0.words.len();
                    to += 1;
                }
                let weight = words as f32 / total as f32;
                heard.extend(self.train_pack(&class[from..to], weight, alpha, embeds, d_embeds, own));
                from = to;
            }
        }
        heard
    }

    /// `windows` as one sequence, each closed by an `<END>` and its backbone
    /// restarting at the next: the reading teacher once, then the heard
    /// student distilled at `alpha`, whose gradient on each window's audio
    /// rows (from the row of `embeds` given with it) lands in `d_embeds`,
    /// times `weight`.
    fn train_pack(
        &mut self,
        windows: &[(&Window, usize)],
        weight: f32,
        alpha: f32,
        embeds: &[f32],
        d_embeds: &mut [f32],
        own: &mut WindowLosses,
    ) -> Vec<(usize, Vec<f32>, String)> {
        let hc = self.enc.cfg.out;
        let end = Utf8Tokenizer::new().end_token() as usize;
        let (mut tokens, mut words, mut mask, mut starts) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        // (row of the packed sequence, row of `embeds`, the word as written)
        let mut heard = Vec::new();
        for &(w, row) in windows {
            let (t0, w0) = (tokens.len(), words.len());
            if w0 > 0 {
                starts.push(w0);
            }
            tokens.extend_from_slice(&w.tokens);
            words.extend(w.words.iter().map(|r| Range { start: r.start + t0, end: r.end + t0 }));
            tokens.push(end);
            words.push(Range { start: tokens.len() - 1, end: tokens.len() });
            match w.train_mask() {
                Some(m) => mask.extend(m),
                None => mask.extend(std::iter::repeat_n(true, w.decoded())),
            }
            // The window's last word predicts its `<END>`, the `<END>` the next
            // window's first word: neither is scored.
            mask.extend([false, false]);
            for j in 0..w.audio {
                let r = w.words[w.prefix + j];
                let word: String = w.tokens[r.start..r.end]
                    .iter()
                    .map(|&c| (c as u8 as char).to_ascii_lowercase())
                    .filter(|c| c.is_alphanumeric() || *c == '\'')
                    .collect();
                heard.push((w0 + w.prefix + j, row + j, word));
            }
        }
        let rows = words.len() - 1;
        mask.truncate(rows);
        let text = self.model.encode_embeds(&self.gpu, &tokens, &words).to_host(&self.gpu).data;
        let mut given = text.clone();
        for &(at, r, _) in &heard {
            given[at * hc..(at + 1) * hc].copy_from_slice(&embeds[r * hc..(r + 1) * hc]);
        }

        if alpha > 0.0 || STATE_DISTILL_BETA > 0.0 {
            if alpha > 0.0 {
                self.model.record_distill();
            }
            if STATE_DISTILL_BETA > 0.0 {
                self.model.record_state();
            }
            self.model.set_doc_starts(&starts);
            self.model.eval_loss_masked(&self.gpu, &tokens, &words, Some(&mask));
            own.teachers.push(self.model.last_word_loss());
            if alpha > 0.0 {
                let targets = self.model.take_distill();
                self.model.apply_distill(targets, alpha);
            }
            if STATE_DISTILL_BETA > 0.0 {
                let target = self.model.take_state();
                self.model.apply_state(target, STATE_DISTILL_BETA);
            }
        }
        let given = GTensor::from_host(&self.gpu, &Tensor::new(&[rows, hc], given));
        let mut d = GTensor::zeros(&self.gpu, &[rows, hc]);
        self.model.set_doc_starts(&starts);
        self.model
            .forward_backward_embeds(&self.gpu, &given, &tokens, &words, Some(&mask), &mut d);
        self.model.end_distill();
        if STATE_DISTILL_BETA > 0.0 {
            self.model.end_state();
            self.state_mse += self.model.last_state_mse();
            self.state_n += 1;
        }
        own.student.push(self.model.last_word_loss());
        let d = d.to_host(&self.gpu).data;
        for &(at, r, _) in &heard {
            for (o, x) in d_embeds[r * hc..(r + 1) * hc].iter_mut().zip(&d[at * hc..(at + 1) * hc]) {
                *o = x * weight;
            }
        }
        if EMBED_ALIGN_WEIGHT > 0.0 {
            // γ/2 · Σ‖a − t‖² / Σ‖t‖² against the text encoder's embedding of
            // each heard word.
            let energy = heard
                .iter()
                .map(|&(at, _, _)| text[at * hc..(at + 1) * hc].iter().map(|x| x * x).sum::<f32>())
                .sum::<f32>()
                .max(f32::MIN_POSITIVE);
            let g = EMBED_ALIGN_WEIGHT / energy;
            for &(at, r, _) in &heard {
                for k in 0..hc {
                    let diff = embeds[r * hc + k] - text[at * hc + k];
                    self.embed_mse += diff * diff / energy;
                    d_embeds[r * hc + k] += g * diff;
                }
            }
            self.embed_n += 1;
        }
        heard
            .into_iter()
            .map(|(at, r, word)| (r, text[at * hc..(at + 1) * hc].to_vec(), word))
            .collect()
    }

    /// The held-out windows of `batch` one by one: the student on its own
    /// words, the first word after them and the text after them, the same
    /// with the audio words rotated as the control, and the teacher reading.
    fn validate_text(
        &mut self,
        batch: &Batch,
        embeds: &[f32],
        own: &mut WindowLosses,
        next: &mut WindowLosses,
        after: &mut WindowLosses,
    ) -> Vec<(usize, Vec<f32>, String)> {
        let hc = self.enc.cfg.out;
        let mut heard_words = Vec::new();
        let mut row = 0;
        for (p, w) in batch.items.iter().zip(&batch.windows) {
            let n = w.audio;
            let audio = &embeds[row * hc..(row + n) * hc];
            let text = self.text_rows(w);
            let target = self.word_rows(w);
            for j in 0..n {
                let r = w.words[w.prefix + j];
                let word: String = w.tokens[r.start..r.end]
                    .iter()
                    .map(|&c| (c as u8 as char).to_ascii_lowercase())
                    .filter(|c| c.is_alphanumeric() || *c == '\'')
                    .collect();
                heard_words.push((row + j, target[j * hc..(j + 1) * hc].to_vec(), word));
            }
            let t = self.val_teacher(p.index, w);
            own.teachers.push(t.own);
            if let Some((t_next, t_after)) = t.after {
                next.teachers.push(t_next);
                after.teachers.push(t_after);
            }
            // The control: each audio word handed the embedding from half an
            // utterance away. Close to the real loss means the embeddings say
            // little about which word was spoken.
            let rotated: Vec<f32> = (0..n)
                .flat_map(|i| {
                    let j = (i + n / 2) % n;
                    audio[j * hc..(j + 1) * hc].iter().copied()
                })
                .collect();
            for (rows, control) in [(audio, false), (&rotated[..], true)] {
                let given = self.given(w, rows, &text);
                let o = self.student_on(&given, w, &w.audio_mask());
                let a = (w.cont > 0).then(|| {
                    (
                        self.student_on(&given, w, &w.next_mask()),
                        self.student_on(&given, w, &w.cont_mask()),
                    )
                });
                let (o_l, n_l, a_l) = if control {
                    (&mut own.controls, &mut next.controls, &mut after.controls)
                } else {
                    (&mut own.student, &mut next.student, &mut after.student)
                };
                o_l.push(o);
                if let Some((n, a)) = a {
                    n_l.push(n);
                    a_l.push(a);
                }
            }
            row += n;
        }
        heard_words
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utterance(text: &str) -> Prepared {
        written(text, text)
    }

    /// An utterance whose spoken words (`said`, space-separated, lower case)
    /// are written as `text`.
    fn written(said: &str, text: &str) -> Prepared {
        let tok = Utf8Tokenizer::new();
        let spoken: Vec<String> = said.split(' ').map(|w| w.to_lowercase()).collect();
        let spoken: Vec<&str> = spoken.iter().map(String::as_str).collect();
        let (raw, written, spoken_at) =
            spelled(&tok.to_tokens(text), &spoken).expect("the text maps onto the spoken words");
        Prepared {
            index: 0,
            id: String::new(),
            speaker: String::new(),
            audio: PathBuf::new(),
            t_in: 0,
            segments: WordSegments::default(),
            times: Vec::new(),
            stride: 4,
            tokens: raw.iter().map(|&t| t as usize).collect(),
            words: split_words(&raw),
            written_words: split_words(&written),
            written: written.iter().map(|&t| t as usize).collect(),
            spoken_at,
            chars: Vec::new(),
        }
    }

    fn text(t: &[usize]) -> String {
        t.iter().map(|&c| c as u8 as char).collect()
    }

    #[test]
    fn the_contrastive_gradient_matches_finite_differences() {
        let hc = 4;
        let words: Vec<(usize, Vec<f32>, String)> = (0..5)
            .map(|i| {
                let t: Vec<f32> = (0..hc).map(|k| ((i * 7 + k * 3) as f32 * 0.9).sin()).collect();
                let w = if i == 3 { "a".to_string() } else { format!("w{i}") };
                (i, t, w)
            })
            .collect();
        let mut words = words;
        words[1].2 = "a".into();
        let embeds: Vec<f32> = (0..5 * hc).map(|x| (x as f32 * 1.3).cos()).collect();
        let loss = |e: &[f32]| -> f64 {
            let n = words.len();
            let tau = EMBED_CONTRAST_TEMP as f64;
            let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
            let mut total = 0.0;
            for i in 0..n {
                let a: Vec<f64> = e[words[i].0 * hc..(words[i].0 + 1) * hc].iter().map(|&x| x as f64).collect();
                let logit = |j: usize| {
                    let t: Vec<f64> = words[j].1.iter().map(|&x| x as f64).collect();
                    a.iter().zip(&t).map(|(x, y)| x * y).sum::<f64>() / (norm(&a) * norm(&t)) / tau
                };
                let lse = (0..n)
                    .filter(|&j| j == i || words[j].2 != words[i].2)
                    .map(|j| logit(j).exp())
                    .sum::<f64>()
                    .ln();
                total += lse - logit(i);
            }
            EMBED_CONTRAST_WEIGHT as f64 * total / n as f64
        };
        let mut d = vec![0.0; embeds.len()];
        contrast(&embeds, hc, &words, Some(&mut d));
        for k in 0..embeds.len() {
            let (mut up, mut down) = (embeds.clone(), embeds.clone());
            up[k] += 1e-3;
            down[k] -= 1e-3;
            let fd = ((loss(&up) - loss(&down)) / 2e-3) as f32;
            assert!((fd - d[k]).abs() < 1e-4 + 1e-2 * fd.abs(), "row {k}: analytic {} vs fd {fd}", d[k]);
        }
    }

    #[test]
    fn a_faster_take_is_shorter_with_its_words_laid_out_again() {
        let dir = std::env::temp_dir().join(format!("nn-speed-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("take.wav");
        let samples: Vec<f32> = (0..2 * AUDIO_SR).map(|i| 0.3 * (i as f32 * 0.05).sin()).collect();
        Clip::new(samples, AUDIO_SR).write(&path).unwrap();
        let mut p = utterance("one two three");
        p.audio = path;
        p.t_in = 2 * AUDIO_SR / AUDIO_HOP;
        p.times = vec![(0.1, 0.5), (0.5, 1.1), (1.1, 1.9)];
        p.segments = WordSegments::from_word_times(&p.times, p.t_in / p.stride);
        let stats = FeatureStats::identity();
        let (same, plain) = p.features_at(&stats, None, 1.0, None);
        assert_eq!((same.len(), plain.t_in), (p.t_in * AUDIO_FEATS, p.t_in));
        let (feats, fast) = p.features_at(&stats, None, 1.1, None);
        assert!(fast.t_in < p.t_in && fast.t_in > p.t_in * 85 / 100, "{} from {}", fast.t_in, p.t_in);
        assert_eq!(feats.len(), fast.t_in * AUDIO_FEATS);
        assert_eq!(fast.segments.len(), 3);
        let ends: Vec<usize> = fast.segments.0.iter().map(|s| s.end).collect();
        assert!(ends.windows(2).all(|w| w[0] < w[1]) && *ends.last().unwrap() <= fast.t_in / fast.stride);
        let slow_ends: Vec<usize> = p.segments.0.iter().map(|s| s.end).collect();
        assert!(ends.iter().zip(&slow_ends).all(|(f, s)| f <= s), "{ends:?} vs {slow_ends:?}");

        // Word times aligned again replace the item's own, before the speed.
        let moved = [(0.1, 0.7), (0.7, 1.2), (1.2, 1.9)];
        let (_, realigned) = p.features_at(&stats, None, 1.0, Some(&moved));
        assert_eq!(realigned.segments, WordSegments::from_word_times(&moved, p.t_in / p.stride));
        assert_ne!(realigned.segments, p.segments);
        let (_, both) = p.features_at(&stats, None, 1.1, Some(&moved));
        assert_eq!(both.times, moved.iter().map(|&(a, b)| (a / 1.1, b / 1.1)).collect::<Vec<_>>());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn punctuation_rides_on_the_spoken_word_before_it() {
        let p = written("with a shout the boys ran", "With a shout, the boys ran.");
        assert_eq!(text(&p.tokens), "With a shout the boys ran");
        assert_eq!(p.words.len(), 6);
        assert_eq!(p.spoken_at, vec![0, 1, 2, 4, 5, 6]);
        let tok = Utf8Tokenizer::new();
        assert!(spelled(&tok.to_tokens("Mr. Smith"), &["mister", "smith"]).is_none());

        // The teacher reads the punctuation and is scored on the same words.
        let next = written("then he stopped", "Then he stopped.");
        let w = Window::new(None, &p, Some(&next.written), 24);
        assert_eq!(text(&w.tokens), "With a shout the boys ran Then he stopped.");
        assert_eq!(text(&w.teacher_tokens), "With a shout, the boys ran. Then he stopped.");
        let t = |m: &[bool]| m.iter().map(|&b| if b { '1' } else { '.' }).collect::<String>();
        assert_eq!(t(&w.audio_mask()), "11111....");
        assert_eq!(t(&w.teacher_mask(&w.audio_mask())), "11.111.....");
        assert_eq!(t(&w.teacher_mask(&w.next_mask())), ".......1...");
    }

    #[test]
    fn a_window_reads_text_audio_text_and_scores_each_part() {
        let (prev, p, next) = (utterance("Once upon a"), utterance("Time there"), utterance("Was a king"));
        let w = Window::new(Some(&prev.written), &p, Some(&next.written), 24);
        assert_eq!(text(&w.tokens), "Once upon a time there was a king");
        let i = Window::new(Some(&prev.written), &utterance("I think"), None, 24);
        assert_eq!(text(&i.tokens), "Once upon a I think");
        assert_eq!((w.prefix, w.audio, w.cont), (3, 2, 3));
        assert_eq!(w.audio_rows(), 2, "the last audio word is read when text follows");
        // Decoded word w predicts word w + 1.
        let t = |m: &[bool]| m.iter().map(|&b| if b { '1' } else { '.' }).collect::<String>();
        assert_eq!(t(&w.train_mask().unwrap()), "...1111");
        assert_eq!(t(&w.audio_mask()), "...1...");
        assert_eq!(t(&w.cont_mask()), "....111");

        let alone = Window::new(None, &p, None, 24);
        assert_eq!((alone.prefix, alone.audio, alone.cont), (0, 2, 0));
        assert_eq!(alone.audio_rows(), 1);
        assert_eq!(t(&alone.audio_mask()), "1");
        assert!(alone.train_mask().is_none());
    }

    #[test]
    fn a_line_break_before_the_audio_ends_the_sentence() {
        let tok = Utf8Tokenizer::new();
        let t = |s: &str| -> Vec<usize> { tok.to_tokens(s).iter().map(|&c| c as usize).collect() };
        let p = written("the crew counts", "The crew counts.");
        let w = Window::new(Some(&t("A Title\n")), &p, Some(&t(" Then it ends.")), 24);
        assert_eq!(text(&w.teacher_tokens), "A Title The crew counts. Then it ends.");
        let w = Window::new(Some(&t("and so")), &p, None, 24);
        assert_eq!(text(&w.tokens), "and so the crew counts");
    }

    /// The recorded takes on this machine: how many map onto their written
    /// prompt, and how many find their document.
    #[test]
    #[ignore]
    fn voice_takes_load() {
        let tok = Utf8Tokenizer::new();
        let mut items = Vec::new();
        let (around, docs) = Corpus::load_voice(&tok, 4, &mut items, AUDIO_VOICE_DIR, AUDIO_VOICE_DOCS);
        let (mut built, mut takes) = (0, 0);
        for d in &docs {
            let mut pieces: Vec<(&Prepared, Range<usize>)> = Vec::new();
            for &(i, r) in &d.takes {
                if pieces.last().is_none_or(|(_, l)| r.start >= l.end) {
                    pieces.push((&items[i], r));
                }
            }
            if Document::voice(&d.text, 0..d.text.len(), &pieces).is_some() {
                built += 1;
                takes += pieces.len();
            }
        }
        println!("{} documents, {built} build as voice windows with {takes} takes heard", docs.len());
        let spelled = items.iter().filter(|p| p.written != p.tokens).count();
        println!("{} takes usable, {spelled} with punctuation, {} with document text", items.len(), around.len());
        for p in items.iter().take(3) {
            println!("  {:?} | {:?}", text(&p.tokens), text(&p.written));
        }
    }

    /// Loads the whole corpus on the CPU, which also writes the length cache.
    #[test]
    #[ignore]
    fn corpus_loads() {
        let t = Instant::now();
        let corpus = Corpus::load(&Utf8Tokenizer::new(), 4);
        println!("{} items in {:.1?}", corpus.items.len(), t.elapsed());
    }

    #[test]
    fn a_document_reads_text_and_hears_its_heard_utterances() {
        let items = [
            written("once upon a time", "Once upon a time,"),
            written("there was a king", "there was a king."),
            written("he had a crown", "He had a crown."),
        ];
        let refs: Vec<&Prepared> = items.iter().collect();
        let doc = Document::new(&refs, &[1]).unwrap();
        assert_eq!(text(&doc.tokens), "Once upon a time, there was a king He had a crown.");
        // "Once upon a time" "," then the heard words, each with its embedding row.
        assert_eq!(doc.heard, vec![(5, 0), (6, 1), (7, 2), (8, 3)]);
        let words: Vec<String> = doc.heard.iter().map(|&(w, _)| text(&doc.tokens[doc.words[w].start..doc.words[w].end])).collect();
        assert_eq!(words, ["there ", "was ", "a ", "king "]);
    }

    #[test]
    fn a_prompt_is_found_in_its_typographic_document() {
        let doc = "Intro.\nA disk \u{2014} even \u{2018}Windows\u{2019} \u{2014} might crash. You\u{2019}ll see.";
        let at = locate(doc, "A disk - even 'Windows' - might crash.").unwrap();
        assert_eq!(&doc[at.start..at.end], "A disk \u{2014} even \u{2018}Windows\u{2019} \u{2014} might crash.");
        assert!(locate(doc, "a cat").is_none());
    }

    #[test]
    fn a_voice_document_hears_its_takes_where_they_stand() {
        let page = "Intro text here. There was a king. He had a crown. The end.";
        let a = written("there was a king", "There was a king.");
        let b = written("he had a crown", "He had a crown.");
        let ra = locate(page, "There was a king.").unwrap();
        let rb = locate(page, "He had a crown.").unwrap();
        let doc = Document::voice(page, 6..page.len(), &[(&a, ra), (&b, rb)]).unwrap();
        assert_eq!(text(&doc.tokens), "text here. There was a king He had a crown The end.");
        let heard: Vec<String> = doc
            .heard
            .iter()
            .map(|&(w, _)| text(&doc.tokens[doc.words[w].start..doc.words[w].end]))
            .collect();
        assert_eq!(heard, ["There ", "was ", "a ", "king ", "He ", "had ", "a ", "crown "]);
        assert_eq!(doc.heard.iter().map(|&(_, e)| e).collect::<Vec<_>>(), (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn a_chained_batch_is_one_row_with_each_utterance_at_its_offset() {
        let mut a = utterance("one two");
        a.t_in = 41;
        a.segments = WordSegments::from_ends(&[3, 9]);
        let mut b = utterance("three four five");
        b.t_in = 24;
        b.segments = WordSegments::from_ends(&[1, 4, 5]);
        let batch = Batch {
            items: vec![std::borrow::Cow::Borrowed(&a), std::borrow::Cow::Borrowed(&b)],
            speeds: Vec::new(),
            windows: Vec::new(),
            feats: Vec::new(),
            t_in: (10 + 6) * 4,
            chain: vec![0, 10],
        };
        assert_eq!(batch.rows(), 1);
        assert_eq!(batch.frames(1, 16, 4), (10, 6));
        let segs = &batch.segments()[0];
        let ends: Vec<usize> = segs.0.iter().map(|s| s.last()).collect();
        assert_eq!(ends, vec![3, 9, 11, 14, 15]);
    }
}

/// `batch`'s word segments with each item, at chance `AUDIO_HEAD_CUT_P`, cut
/// where the boundary head (`boundary`, `[rows, t_out, delay + 1]`) puts its
/// words given their count; and how many items were cut so.
fn head_cut_segments(batch: &Batch, boundary: &[f32], t_out: usize, c: AudioCfg) -> (Vec<WordSegments>, usize) {
    let mut pooled = batch.segments();
    if AUDIO_HEAD_CUT_P <= 0.0 {
        return (pooled, 0);
    }
    let cols = c.delay + 1;
    let aligner = CountAligner {
        delay: c.delay,
        from_col: 0,
        off_peak: 3.0,
    };
    let prior = DurationPrior {
        sigma: 0.5,
        weight: 1.0,
        max_frames: 100,
    };
    let chosen: Vec<bool> = batch.items.iter().map(|_| random_range(0.0..1.0) < AUDIO_HEAD_CUT_P).collect();
    let cut: Vec<Option<WordSegments>> = batch
        .items
        .par_iter()
        .zip(&chosen)
        .enumerate()
        .map(|(b, (p, &chosen))| {
            if !chosen {
                return None;
            }
            let own = p.t_in / c.stride;
            let rows = &boundary[b * t_out * cols..(b * t_out + own) * cols];
            let lens: Vec<usize> = p.words.iter().map(|r| r.end - r.start).collect();
            let ends = aligner.align_words(rows, own, &lens, prior)?;
            let seconds = (p.t_in * AUDIO_HOP) as f32 / AUDIO_SR as f32;
            let mut start = 0.0;
            let times: Vec<(f32, f32)> = ends
                .iter()
                .map(|&e| {
                    let w = (start, ((e + 1) as f32 * ENC_FRAME_S).min(seconds));
                    start = w.1;
                    w
                })
                .collect();
            let segments = WordSegments::from_word_times(&times, own);
            (segments.len() == p.segments.len()).then_some(segments)
        })
        .collect();
    let mut n = 0;
    for (b, s) in cut.into_iter().enumerate() {
        if let Some(s) = s {
            pooled[b] = s;
            n += 1;
        }
    }
    (pooled, n)
}
