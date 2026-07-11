use std::{fs, time::Instant};

use rand::random_range;

use crate::{
    config::{
        WAKE_DATA_NEG, WAKE_DATA_POS, WAKE_EPOCHS, WAKE_FRAME_LEN, WAKE_FRAME_SHIFT, WAKE_LR,
        WAKE_MODEL_LOC, WAKE_SR,
    },
    wake_word::{
        mfcc::{MelExtractor, resample, sigmoid},
        model::load_or_build_wake_model,
    },
};

const AUG_PER_POS: usize = 4;
const PREFIX_MIN_S: f32 = 0.3;
const PREFIX_MAX_S: f32 = 25.0;

pub struct Frame {
    pub features: Vec<f32>,
    pub label: f32,
}

impl AsRef<[f32]> for Frame {
    fn as_ref(&self) -> &[f32] {
        &self.features
    }
}

pub struct Sequence {
    pub frames: Vec<Frame>,
}

pub fn train_wake() {
    fs::create_dir_all(WAKE_DATA_POS).ok();
    fs::create_dir_all(WAKE_DATA_NEG).ok();

    let extractor = MelExtractor::new();
    let pos_raw = load_raw_dir(WAKE_DATA_POS);
    let neg_raw = load_raw_dir(WAKE_DATA_NEG);

    if pos_raw.is_empty() {
        eprintln!("no positive training data found — record samples first with 'wr'");
        return;
    }

    // Pre-allocate cache large enough for the longest possible sequence:
    // up to PREFIX_MAX_S of neg audio (split into prefix + suffix) plus the positive clip.
    let max_pos_samples = pos_raw.iter().map(|a| a.len()).max().unwrap_or(0);
    let max_seq_samples =
        (PREFIX_MAX_S * WAKE_SR as f32) as usize + max_pos_samples + WAKE_FRAME_LEN;
    let max_seq_frames = 2 + max_seq_samples / WAKE_FRAME_SHIFT;

    println!(
        "{} positive clips, {} negative clips  ({} pos + {} neg sequences/epoch, max {} frames/seq)",
        pos_raw.len(),
        neg_raw.len(),
        pos_raw.len() * AUG_PER_POS,
        pos_raw.len() * AUG_PER_POS,
        max_seq_frames,
    );

    let mut model = load_or_build_wake_model();
    model.make_cache(max_seq_frames);

    for epoch in 0..WAKE_EPOCHS {
        // Every epoch: build fresh sequences with new random prefixes and shuffling.
        let mut seqs = build_sequences(&pos_raw, &neg_raw, &extractor);
        shuffle_seqs(&mut seqs);

        // State starts clean each epoch; carries across sequences within the epoch.
        for layer in &mut model.layers {
            layer.reset_state();
        }

        let mut total_loss = 0.0;
        let mut correct = 0;
        let mut total_frames = 0;
        let t0 = Instant::now();

        let lr = lr_cosine(epoch, WAKE_EPOCHS, WAKE_LR);

        for s in &seqs {
            // State carries from the previous sequence — no reset here.
            // Grow cache if a sequence is somehow longer than pre-allocated.
            if model.cache.len() < s.frames.len() {
                panic!(
                    "sequence with {} frames exceeds model cache of {} frames",
                    s.frames.len(),
                    model.cache.len()
                );
            }

            // Full forward pass through all frames (neg prefix + word).
            // BPTT will flow back through every frame so the model learns
            // "Jarvis" as a multi-frame temporal pattern (~20-30 frames).
            model.forward_raw_seq(&s.frames);

            let n_frames = s.frames.len();

            let mut num_neg = 0;
            let mut num_pos = 0;

            s.frames.iter().for_each(|f| {
                if f.label == 0.0 {
                    num_neg += 1
                } else {
                    num_pos += 1
                }
            });

            let weight_neg = 1.0 / num_neg as f32;
            let weight_pos = 1.0 / num_pos as f32;

            for t in 0..n_frames {
                let p = sigmoid(model.output(t)[0]);
                let label = s.frames[t].label;
                let weight = if label > 0.5 { weight_pos } else { weight_neg };

                total_loss -=
                    weight * (label * (p + 1e-9).ln() + (1.0 - label) * (1.0 - p + 1e-9).ln());

                if (p > 0.5) == (label > 0.5) {
                    correct += 1;
                }
                total_frames += 1;
            }

            model.backwards_wake_bce(s, weight_pos, weight_neg);

            for layer in &mut model.layers {
                layer.accumulate_init_grad();
                layer.reset_bptt_state();
                layer.reset_state();
            }

            model.sgd_step(lr, crate::config::FLAT_WEIGHT_DECAY);
        }

        println!(
            "epoch {:2} | loss {:.4} | acc {:.1}% | {:.1?}",
            epoch + 1,
            total_loss / seqs.len() as f32,
            correct as f32 / total_frames as f32 * 100.0,
            t0.elapsed(),
        );

        if let Err(e) = model.save(WAKE_MODEL_LOC) {
            eprintln!("save failed: {e}");
        }
    }

    println!("training done — model saved to {WAKE_MODEL_LOC}");
}

// ── sequence construction ─────────────────────────────────────────────────────

fn build_sequences(
    pos_raw: &[Vec<f32>],
    neg_raw: &[Vec<f32>],
    extractor: &MelExtractor,
) -> Vec<Sequence> {
    let mut out = Vec::new();
    for pos in pos_raw {
        for _ in 0..AUG_PER_POS {
            let total_neg = random_prefix_len();
            let prefix_len = random_range(0..=total_neg);
            let suffix_len = total_neg - prefix_len;

            let mut audio = build_prefix(neg_raw, prefix_len);
            audio.extend_from_slice(pos);
            let pos_end = audio.len();
            audio.extend(build_prefix(neg_raw, suffix_len));

            if random_range(0..4) != 0 {
                augment(&mut audio);
            }

            out.push(Sequence {
                frames: extract_seq(extractor, &audio, pos_end),
            });
        }
    }
    out
}

fn random_prefix_len() -> usize {
    let s = random_range(PREFIX_MIN_S..PREFIX_MAX_S);
    (s * WAKE_SR as f32) as usize
}

/// Fill `target_len` samples with random segments from neg clips,
/// randomly interspersed with short silence gaps.
fn build_prefix(neg_raw: &[Vec<f32>], target_len: usize) -> Vec<f32> {
    let mut out = vec![0.0; target_len];
    if neg_raw.is_empty() {
        return out; // pure silence if no negative clips
    }
    let mut pos = 0;
    while pos < target_len {
        // 25% chance: short silence gap instead of audio
        if random_range(0..4) == 0 {
            pos += random_range(0..WAKE_SR / 10).min(target_len - pos);
            continue;
        }
        let clip = &neg_raw[random_range(0..neg_raw.len())];
        let available = target_len - pos;
        if clip.len() <= available {
            out[pos..pos + clip.len()].copy_from_slice(clip);
            pos += clip.len();
        } else {
            let off = random_range(0..=clip.len() - available);
            out[pos..target_len].copy_from_slice(&clip[off..off + available]);
            pos = target_len;
        }
    }
    out
}

fn extract_seq(extractor: &MelExtractor, audio: &[f32], pos_end: usize) -> Vec<Frame> {
    let n = 1 + (audio.len().saturating_sub(WAKE_FRAME_LEN)) / WAKE_FRAME_SHIFT;
    let pos_trigger = pos_end.saturating_sub(1) / WAKE_FRAME_SHIFT;
    let mut buf = [0.0; WAKE_FRAME_LEN];
    (0..n)
        .map(|i| {
            let start = i * WAKE_FRAME_SHIFT;
            let end = (start + WAKE_FRAME_LEN).min(audio.len());
            buf.fill(0.0);
            buf[..end - start].copy_from_slice(&audio[start..end]);
            let features = extractor.extract_frame(&buf);
            Frame {
                features,
                label: if i == pos_trigger { 1.0 } else { 0.0 },
            }
        })
        .collect()
}

// ── I/O ───────────────────────────────────────────────────────────────────────

fn load_raw_dir(dir: &str) -> Vec<Vec<f32>> {
    let mut out = Vec::new();
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return out,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("wav") {
            continue;
        }
        if let Some(audio) = load_wav(path.to_str().unwrap()) {
            out.push(audio);
        }
    }
    out
}

fn load_wav(path: &str) -> Option<Vec<f32>> {
    let reader = hound::WavReader::open(path).ok()?;
    let spec = reader.spec();

    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .collect(),
        (hound::SampleFormat::Int, 16) => reader
            .into_samples::<i16>()
            .filter_map(|s| s.ok())
            .map(|s| s as f32 / 32768.0)
            .collect(),
        (hound::SampleFormat::Int, 24) | (hound::SampleFormat::Int, 32) => reader
            .into_samples::<i32>()
            .filter_map(|s| s.ok())
            .map(|s| s as f32 / 2_147_483_648.0)
            .collect(),
        _ => {
            eprintln!("unsupported WAV format in {path}");
            return None;
        }
    };

    let ch = spec.channels as usize;
    let mono: Vec<f32> = if ch == 1 {
        samples
    } else {
        samples
            .chunks(ch)
            .map(|c| c.iter().sum::<f32>() / ch as f32)
            .collect()
    };

    if spec.sample_rate != WAKE_SR as u32 {
        Some(resample(&mono, spec.sample_rate as usize, WAKE_SR))
    } else {
        Some(mono)
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn augment(audio: &mut Vec<f32>) {
    let vol = random_range(0.5..1.5);
    let noise_amp = random_range(0.0..0.05);
    for s in audio.iter_mut() {
        *s = (*s * vol + random_range(-noise_amp..noise_amp)).clamp(-1.0, 1.0);
    }
}

fn lr_cosine(epoch: usize, total: usize, lr_max: f32) -> f32 {
    use std::f32::consts::PI;
    let t = epoch as f32 / total as f32;
    lr_max * 0.5 * (1.0 + (PI * t).cos())
}

fn shuffle_seqs(v: &mut Vec<Sequence>) {
    let n = v.len();
    for i in (1..n).rev() {
        let j = random_range(0..=i);
        v.swap(i, j);
    }
}
