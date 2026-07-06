use std::{
    sync::mpsc,
    time::{Duration, Instant},
};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait as _};

use crate::{
    config::{WAKE_FRAME_LEN, WAKE_FRAME_SHIFT, WAKE_MODEL_LOC, WAKE_SR, WAKE_THRESHOLD},
    wake_word::mfcc::{MelExtractor, resample, sigmoid},
};

pub fn run_detector() {
    let mut model = match crate::sequential::Sequential::load(WAKE_MODEL_LOC) {
        Ok(m) => m,
        Err(e) => {
            panic!("failed to load wake-word model from {WAKE_MODEL_LOC}: {e}");
        }
    };
    model.make_cache(1);

    let extractor = MelExtractor::new();

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("no input device available");
    let config = crate::wake_word::preferred_input_config(&device);

    let native_sr = config.sample_rate() as usize;
    let channels = config.channels() as usize;

    let frame_len_native = WAKE_FRAME_LEN * native_sr / WAKE_SR;
    let frame_shift_native = WAKE_FRAME_SHIFT * native_sr / WAKE_SR;

    // Channel carries complete frame-shift-sized hops — the callback is the threshold gate.
    let (audio_sender, audio_receiver) = mpsc::sync_channel::<Vec<f32>>(1024);

    let stream: cpal::Stream = match config.sample_format() {
        cpal::SampleFormat::F32 => {
            let mut acc: Vec<f32> = Vec::with_capacity(frame_shift_native * 2);
            device
                .build_input_stream(
                    config.config(),
                    move |data: &[f32], _| {
                        push_mono(&mut acc, data, channels);
                        while acc.len() >= frame_shift_native {
                            let _ = audio_sender.send(acc.drain(..frame_shift_native).collect());
                        }
                    },
                    |e| eprintln!("stream error: {e}"),
                    None,
                )
                .expect("failed to build stream")
        }
        cpal::SampleFormat::I16 => {
            let mut acc: Vec<f32> = Vec::with_capacity(frame_shift_native * 2);
            device
                .build_input_stream(
                    config.config(),
                    move |data: &[i16], _| {
                        push_mono_i16(&mut acc, data, channels);
                        while acc.len() >= frame_shift_native {
                            let _ = audio_sender.send(acc.drain(..frame_shift_native).collect());
                        }
                    },
                    |e| eprintln!("stream error: {e}"),
                    None,
                )
                .expect("failed to build stream")
        }
        fmt => panic!("unsupported sample format {fmt:?}"),
    };

    stream.play().expect("stream play failed");

    for layer in &mut model.layers {
        layer.reset_state();
    }
    println!("listening for 'Jarvis'…");

    // Sliding window: overlap samples stay, new hop fills the tail.
    let overlap = frame_len_native - frame_shift_native;
    let mut window = vec![0.0; frame_len_native];

    let mut frame_buf = [0.0; WAKE_FRAME_LEN];
    let mut last_trigger = Instant::now();
    const COOLDOWN: Duration = Duration::from_millis(1000);

    loop {
        let Ok(hop) = audio_receiver.recv() else {
            break;
        };

        window.copy_within(frame_shift_native.., 0);
        window[overlap..].copy_from_slice(&hop);

        if native_sr != WAKE_SR {
            let resampled = resample(&window, native_sr, WAKE_SR);
            let copy = resampled.len().min(WAKE_FRAME_LEN);
            frame_buf[..copy].copy_from_slice(&resampled[..copy]);
            frame_buf[copy..].fill(0.0);
        } else {
            frame_buf.copy_from_slice(&window);
        }

        let features = extractor.extract_frame(&frame_buf);
        let out = model.forward_raw(&features);
        let p = sigmoid(out[0]);

        if p >= WAKE_THRESHOLD && last_trigger.elapsed() > COOLDOWN {
            println!("Jarvis  (p={:.3})", p);
            last_trigger = Instant::now();
            for layer in &mut model.layers {
                layer.reset_state();
            }
        }
    }
}

fn push_mono(acc: &mut Vec<f32>, data: &[f32], channels: usize) {
    if channels == 1 {
        acc.extend_from_slice(data);
    } else {
        let inv = 1.0 / channels as f32;
        acc.extend(data.chunks(channels).map(|c| c.iter().sum::<f32>() * inv));
    }
}

fn push_mono_i16(acc: &mut Vec<f32>, data: &[i16], channels: usize) {
    if channels == 1 {
        acc.extend(data.iter().map(|&s| s as f32 / 32768.0));
    } else {
        let inv = 1.0 / channels as f32;
        acc.extend(
            data.chunks(channels)
                .map(|c| c.iter().map(|&s| s as f32 / 32768.0).sum::<f32>() * inv),
        );
    }
}
