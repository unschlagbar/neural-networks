use std::{
    fs,
    io::{self, BufRead, Write},
    path::Path,
    sync::{Arc, Mutex},
    time::Duration,
};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait as _};

use crate::{
    config::{WAKE_DATA_NEG, WAKE_DATA_POS, WAKE_SR},
    wake_word::mfcc::resample,
};

const SILENCE_SECS: f32 = 0.75;
const MAX_RECORD_SECS: f32 = 10.0;
const VOICE_THRESHOLD: f32 = 0.025; // RMS energy; raise if mic is loud
const VAD_CHUNK_MS: f32 = 20.0; // VAD frame size in ms
const TAIL_SECS: f32 = 0.30; // keep 50ms after last speech frame

pub fn record_samples() {
    fs::create_dir_all(WAKE_DATA_POS).expect("cannot create positive dir");
    fs::create_dir_all(WAKE_DATA_NEG).expect("cannot create negative dir");

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("no input device available");
    let config = crate::wake_word::preferred_input_config(&device);

    println!("input device: {}", device.description().unwrap().name());
    println!("sample rate:  {} Hz", config.sample_rate());
    println!("silence stop: {SILENCE_SECS}s  threshold: {VOICE_THRESHOLD}");
    println!();
    println!("commands: [p] positive  [n] negative  [q] quit");

    let stdin = io::stdin();
    loop {
        print!("> ");
        io::stdout().flush().ok();
        let mut line = String::new();
        stdin.lock().read_line(&mut line).unwrap();
        match line.trim() {
            "p" | "positive" => record_one(&device, &config, WAKE_DATA_POS),
            "n" | "negative" => record_one(&device, &config, WAKE_DATA_NEG),
            "q" | "quit" => break,
            other => println!("unknown command '{other}'"),
        }
    }
}

fn record_one(device: &cpal::Device, config: &cpal::SupportedStreamConfig, dir: &str) {
    let sr = config.sample_rate() as usize;
    let channels = config.channels() as usize;

    let vad_chunk = ((VAD_CHUNK_MS / 1000.0) * sr as f32) as usize;
    let silence_limit = (SILENCE_SECS * sr as f32) as usize;
    let max_samples = (MAX_RECORD_SECS * sr as f32) as usize;

    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let buf_cb = buffer.clone();
    let buf_cb2 = buffer.clone();

    let stream: cpal::Stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device
            .build_input_stream(
                &config.config(),
                move |data: &[f32], _| push_mono(data, channels, &buf_cb),
                |e| eprintln!("stream error: {e}"),
                None,
            )
            .expect("failed to build stream"),
        cpal::SampleFormat::I16 => device
            .build_input_stream(
                &config.config(),
                move |data: &[i16], _| {
                    let f: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
                    push_mono(&f, channels, &buf_cb2);
                },
                |e| eprintln!("stream error: {e}"),
                None,
            )
            .expect("failed to build stream"),
        fmt => panic!("unsupported sample format {fmt:?}"),
    };

    stream.play().expect("stream play failed");
    println!("ready — speak now…");

    let mut processed = 0;
    let mut speech_started = false;
    let mut silence_accum = 0;

    'vad: loop {
        std::thread::sleep(Duration::from_millis(10));
        let current_len = buffer.lock().unwrap().len();

        while processed + vad_chunk <= current_len {
            let rms = {
                let buf = buffer.lock().unwrap();
                rms_energy(&buf[processed..processed + vad_chunk])
            };

            if rms >= VOICE_THRESHOLD {
                if !speech_started {
                    speech_started = true;
                }
                silence_accum = 0;
            } else if speech_started {
                silence_accum += vad_chunk;
                if silence_accum >= silence_limit {
                    break 'vad;
                }
            }

            processed += vad_chunk;

            if processed >= max_samples {
                break 'vad;
            }
        }
    }

    drop(stream);

    if !speech_started {
        println!("no speech detected — discarding");
        return;
    }

    let raw = buffer.lock().unwrap().clone();

    // Trim trailing silence, keep TAIL_SECS after last voiced frame
    let trimmed = trim_trailing_silence(&raw, vad_chunk, sr);

    if trimmed.is_empty() {
        println!("empty after trimming — discarding");
        return;
    }

    let final_samples = if sr != WAKE_SR {
        resample(&trimmed, sr, WAKE_SR)
    } else {
        trimmed
    };

    let path = next_wav_path(dir);
    save_wav(&path, &final_samples);
    println!(
        "saved {:.2}s → {path}",
        final_samples.len() as f32 / WAKE_SR as f32
    );
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn push_mono(data: &[f32], channels: usize, buf: &Arc<Mutex<Vec<f32>>>) {
    let mut b = buf.lock().unwrap();
    if channels == 1 {
        b.extend_from_slice(data);
    } else {
        let inv = 1.0 / channels as f32;
        for chunk in data.chunks(channels) {
            b.push(chunk.iter().sum::<f32>() * inv);
        }
    }
}

fn rms_energy(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Keep audio up to (last voiced frame + TAIL_SECS), discard everything after.
fn trim_trailing_silence(samples: &[f32], chunk: usize, sr: usize) -> Vec<f32> {
    let tail = (TAIL_SECS * sr as f32) as usize;
    let mut last_speech_end = 0usize;

    let mut i = 0;
    while i + chunk <= samples.len() {
        if rms_energy(&samples[i..i + chunk]) >= VOICE_THRESHOLD {
            last_speech_end = i + chunk;
        }
        i += chunk;
    }

    if last_speech_end == 0 {
        return Vec::new();
    }

    let end = (last_speech_end + tail).min(samples.len());
    samples[..end].to_vec()
}

fn next_wav_path(dir: &str) -> String {
    let mut idx = 1;
    loop {
        let p = format!("{dir}/{:06}.wav", idx);
        if !Path::new(&p).exists() {
            return p;
        }
        idx += 1;
    }
}

fn save_wav(path: &str, samples: &[f32]) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: WAKE_SR as u32,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("cannot create WAV");
    for &s in samples {
        writer.write_sample(s).unwrap();
    }
    writer.finalize().unwrap();
}
