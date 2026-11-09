//! Writes one utterance through a few random augmentations as WAVs to listen
//! to, and times the augmentation.
//!
//!   cargo run --release --example augment_listen -- <utterance.flac> <out dir> [noise dir] [other speakers' clips...]

use std::time::Instant;

use neural_networks::audio::augment::{Augment, AugmentCfg, NoiseBank};
use neural_networks::audio::io::Clip;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (input, out_dir) = (&args[0], &args[1]);
    let sr = 16_000;
    let noise = NoiseBank::load(args.get(2).map_or("", String::as_str), sr);
    println!("{} noise recordings", noise.len());
    let aug = Augment {
        cfg: AugmentCfg::speech(sr),
        noise,
        babble: NoiseBank::from_files(&args[3..].iter().map(Into::into).collect::<Vec<_>>(), sr),
    };
    let clean = Clip::read_at(input, sr).unwrap();
    std::fs::create_dir_all(out_dir).unwrap();
    clean.write(format!("{out_dir}/clean.wav")).unwrap();
    for i in 0..6 {
        let t = Instant::now();
        let speed = aug.draw_speed();
        let mut audio = Augment::speed(&clean.samples, sr, speed);
        aug.apply(&mut audio);
        println!("augmented {i}: speed {speed:.2}, {:.1?} for {:.1} s of audio", t.elapsed(), clean.seconds());
        Clip::new(audio, sr).write(format!("{out_dir}/augmented_{i}.wav")).unwrap();
    }
}
