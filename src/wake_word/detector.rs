use std::{
    sync::{Arc, Mutex},
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
            eprintln!("failed to load wake-word model from {WAKE_MODEL_LOC}: {e}");
            eprintln!("train a model first with 'wt'");
            std::process::exit(1);
        }
    };
    model.make_cache(1);

    let extractor = MelExtractor::new();

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("no input device available");
    let config = crate::wake_word::preferred_input_config(&device);

    let native_sr = config.sample_rate().0;
    let channels = config.channels() as usize;

    // Frame sizes at native sample rate
    let frame_len_native =
        (WAKE_FRAME_LEN as f32 * native_sr as f32 / WAKE_SR as f32).round() as usize;
    let frame_shift_native =
        (WAKE_FRAME_SHIFT as f32 * native_sr as f32 / WAKE_SR as f32).round() as usize;

    let ring = Arc::new(Mutex::new(RingBuffer::new(native_sr as usize * 3)));
    let ring_cb = ring.clone();
    let ring_cb2 = ring.clone();

    let stream: cpal::Stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device
            .build_input_stream(
                &config.into(),
                move |data: &[f32], _| {
                    let mono = to_mono_f32(data, channels);
                    ring_cb.lock().unwrap().push(&mono);
                },
                |e| eprintln!("stream error: {e}"),
                None,
            )
            .expect("failed to build stream"),
        cpal::SampleFormat::I16 => device
            .build_input_stream(
                &config.into(),
                move |data: &[i16], _| {
                    let f: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
                    let mono = to_mono_f32(&f, channels);
                    ring_cb2.lock().unwrap().push(&mono);
                },
                |e| eprintln!("stream error: {e}"),
                None,
            )
            .expect("failed to build stream"),
        fmt => panic!("unsupported sample format {fmt:?}"),
    };

    stream.play().expect("stream play failed");

    // Start from the current write position — don't process stale data.
    let mut next_sample = ring.lock().unwrap().total_written();

    // Model state runs continuously; only reset on a positive detection.
    for layer in &mut model.layers {
        layer.reset_state();
    }
    println!("listening for 'Jarvis'…  (Ctrl+C to stop)");

    let mut frame_buf = vec![0.0; WAKE_FRAME_LEN];
    let mut last_trigger = Instant::now();
    const COOLDOWN: Duration = Duration::from_millis(1000);

    loop {
        // Sleep approximately one MFCC frame shift (10 ms at 16 kHz).
        std::thread::sleep(Duration::from_millis(10));

        // If too far behind (e.g. process was paused), skip ahead.
        {
            let head = ring.lock().unwrap().total_written();
            let safe = head.saturating_sub(native_sr as usize * 2);
            if next_sample < safe {
                next_sample = safe;
            }
        }

        // Drain all fully-available frames in one burst.
        loop {
            let maybe_frame = {
                let rb = ring.lock().unwrap();
                rb.get_range(next_sample, frame_len_native)
            };

            let frame_native = match maybe_frame {
                Some(f) => f,
                None => break,
            };

            // Resample to WAKE_SR if device couldn't do 16 kHz natively.
            let frame_ws = if native_sr != WAKE_SR as u32 {
                resample(&frame_native, native_sr, WAKE_SR as u32)
            } else {
                frame_native
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
