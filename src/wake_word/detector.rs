use std::{
    sync::mpsc,
    time::{Duration, Instant},
};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait as _};

use crate::{
    config::{WAKE_FRAME_LEN, WAKE_FRAME_SHIFT, WAKE_MODEL_LOC, WAKE_SR, WAKE_THRESHOLD},
    wake_word::{
        buffer::RingBuffer,
        mfcc::{MelExtractor, resample, sigmoid},
    },
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

    // Frame sizes at native sample rate
    let frame_len_native = WAKE_FRAME_LEN * native_sr / WAKE_SR;
    let frame_shift_native = WAKE_FRAME_SHIFT * native_sr / WAKE_SR;

    let (audio_sender, audio_receiver) = mpsc::sync_channel(native_sr as usize * 3);

    let stream: cpal::Stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device
            .build_input_stream(
                &config.config(),
                move |data: &[f32], _| {
                    let mono = to_mono_f32(data, channels);
                    audio_sender.send(mono).unwrap();
                },
                |e| eprintln!("stream error: {e}"),
                None,
            )
            .expect("failed to build stream"),
        cpal::SampleFormat::I16 => device
            .build_input_stream(
                &config.config(),
                move |data: &[i16], _| {
                    let f: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
                    let mono = to_mono_f32(&f, channels);
                    audio_sender.send(mono).unwrap();
                },
                |e| eprintln!("stream error: {e}"),
                None,
            )
            .expect("failed to build stream"),
        fmt => panic!("unsupported sample format {fmt:?}"),
    };

    stream.play().expect("stream play failed");

    // Local ring buffer fed from the channel: fixed allocation, O(1) push and indexed reads.
    let mut ring = RingBuffer::new(native_sr as usize * 3);

    // Start from the current write position — don't process stale data.
    let mut next_sample = 0;

    // Model state runs continuously; only reset on a positive detection.
    for layer in &mut model.layers {
        layer.reset_state();
    }
    println!("listening for 'Jarvis'…");

    let mut frame_buf = [0.0; WAKE_FRAME_LEN];
    let mut last_trigger = Instant::now();
    const COOLDOWN: Duration = Duration::from_millis(1000);

    loop {
        std::thread::sleep(Duration::from_millis(50));

        // Drain the channel into the local ring buffer.
        while let Ok(chunk) = audio_receiver.try_recv() {
            ring.push(&chunk);
        }

        // If too far behind (e.g. process was paused), skip ahead.
        let head = ring.total_written();
        let safe = head.saturating_sub(native_sr as usize * 2);
        if next_sample < safe {
            next_sample = safe;
        }

        // Drain all fully-available frames in one burst.
        while let Some(frame) = ring.get_range(next_sample, frame_len_native) {
            // Resample to WAKE_SR if device couldn't do 16 kHz natively.
            let frame_ws = if native_sr != WAKE_SR {
                print!("resampling from {native_sr} Hz… ");
                resample(&frame, native_sr, WAKE_SR)
            } else {
                frame
            };

            // Copy into fixed-size buffer (trim or zero-pad to WAKE_FRAME_LEN).
            let copy = frame_ws.len().min(WAKE_FRAME_LEN);
            frame_buf[..copy].copy_from_slice(&frame_ws[..copy]);
            frame_buf[copy..].fill(0.0);

            let features = extractor.extract_frame(&frame_buf);
            let out = model.forward_raw(&features);
            let p = sigmoid(out[0]);

            if p >= WAKE_THRESHOLD && last_trigger.elapsed() > COOLDOWN {
                println!("JARVIS  (p={:.3})", p);
                last_trigger = Instant::now();

                for layer in &mut model.layers {
                    layer.reset_state();
                }
            }

            next_sample += frame_shift_native;
        }
    }
}

fn to_mono_f32(data: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        data.to_vec()
    } else {
        let inv = 1.0 / channels as f32;
        data.chunks(channels)
            .map(|c| c.iter().sum::<f32>() * inv)
            .collect()
    }
}
