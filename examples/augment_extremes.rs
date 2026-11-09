//! Writes one utterance through the edges of `AugmentCfg::speech` as WAVs to
//! listen to: each knob alone at its harshest, then all of them at once.

use neural_networks::audio::augment::{Augment, AugmentCfg, NoiseBank};
use neural_networks::audio::io::Clip;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (input, out_dir, noise_dir) = (&args[0], &args[1], &args[2]);
    let sr = 16_000;
    let base = AugmentCfg::speech(sr);
    let at = |r: (f32, f32), hi: bool| if hi { (r.1 - 1e-3, r.1) } else { (r.0, r.0 + 1e-3) };
    let off = AugmentCfg {
        p_room: 0.0,
        p_noise: 0.0,
        p_mic: 0.0,
        p_babble: 0.0,
        gain_db: (0.0, 1e-3),
        floor_dbfs: (-140.0, -139.9),
        ..base
    };
    let cases: Vec<(&str, AugmentCfg)> = vec![
        ("1_quietest", AugmentCfg { gain_db: at(base.gain_db, false), floor_dbfs: at(base.floor_dbfs, true), ..off }),
        ("2_loudest", AugmentCfg { gain_db: at(base.gain_db, true), ..off }),
        ("3_noise_0db", AugmentCfg { p_noise: 1.0, snr_db: at(base.snr_db, false), ..off }),
        ("4_other_speaker_5db", AugmentCfg { p_babble: 1.0, babble_snr_db: at(base.babble_snr_db, false), ..off }),
        (
            "5_far_reverberant_room",
            AugmentCfg { p_room: 1.0, rt60: at(base.rt60, true), drr_db: at(base.drr_db, false), ..off },
        ),
        (
            "6_everything_at_once",
            AugmentCfg {
                p_room: 1.0,
                p_noise: 1.0,
                p_mic: 1.0,
                p_babble: 1.0,
                rt60: at(base.rt60, true),
                drr_db: at(base.drr_db, false),
                snr_db: at(base.snr_db, false),
                babble_snr_db: at(base.babble_snr_db, false),
                gain_db: at(base.gain_db, false),
                floor_dbfs: at(base.floor_dbfs, true),
                ..base
            },
        ),
    ];
    let clean = Clip::read_at(input, sr).unwrap();
    std::fs::create_dir_all(out_dir).unwrap();
    clean.write(format!("{out_dir}/0_clean.wav")).unwrap();
    let babble: Vec<std::path::PathBuf> = args[3..].iter().map(Into::into).collect();
    for (name, cfg) in cases {
        let aug = Augment {
            cfg,
            noise: NoiseBank::load(noise_dir, sr),
            babble: NoiseBank::from_files(&babble, sr),
        };
        let mut audio = clean.samples.clone();
        aug.apply(&mut audio);
        let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt();
        println!("{name}: {:.1} dBFS", 20.0 * rms.max(1e-9).log10());
        Clip::new(audio, sr).write(format!("{out_dir}/{name}.wav")).unwrap();
    }
}
