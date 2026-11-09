//! What the character head hears on the recorded voice takes: transcript,
//! greedy CTC decode and per-take CER, held-out sessions first.
//!
//!   cargo run --release --features cuda --example ae2_voice_ctc -- models/s4.ae2.G-460h [takes per group]

use neural_networks::audio::ctc::{Ctc, CtcAlphabet, edit_distance};
use neural_networks::audio::frontend::LogMel;
use neural_networks::audio::io::Clip;
use neural_networks::audio::segments::WordSegments;
use neural_networks::audio::voice;
use neural_networks::config::{AUDIO_FEATS, AUDIO_SR, AUDIO_VOICE_DIR, AUDIO_VOICE_VAL_SESSIONS};
use neural_networks::gpu::Gpu;
use neural_networks::gpu::audio::AudioEncoder;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let per_group: usize = args.get(1).and_then(|n| n.parse().ok()).unwrap_or(15);
    let gpu = Gpu::new().unwrap();
    let (mut enc, ck) = AudioEncoder::load(&gpu, &args[0]).unwrap();
    let c = enc.cfg;
    let ctc = Ctc { vocab: CtcAlphabet::SIZE };
    let takes = voice::load(AUDIO_VOICE_DIR).unwrap();
    println!("{} at step {}", args[0], ck.step);
    for held_out in [true, false] {
        let group: Vec<_> = takes
            .iter()
            .filter(|t| AUDIO_VOICE_VAL_SESSIONS.contains(&t.session.as_str()) == held_out)
            .collect();
        let (mut errors, mut chars) = (0, 0);
        println!("\n{}", if held_out { "HELD-OUT session (never trained on)" } else { "TRAINED sessions" });
        for (k, t) in group.iter().enumerate() {
            let Some(target) = CtcAlphabet::encode(&t.spoken) else { continue };
            let clip = Clip::read_at(&t.audio, AUDIO_SR).unwrap();
            let mut feats = LogMel::new().utterance(&clip.samples);
            ck.stats.apply(&mut feats);
            let t_in = feats.len() / AUDIO_FEATS;
            let t_out = c.stem().frames(t_in);
            let cache = enc.training_cache(&gpu, t_out, t_out, t_out);
            let out = enc.forward(&gpu, &feats, 1, t_in, &[WordSegments::default()], &cache);
            let heard = ctc.greedy(&out.ctc.to_host(&gpu).data);
            enc.discard();
            let e = edit_distance(&heard, &target);
            errors += e;
            chars += target.len();
            if k < per_group {
                println!("  said:  {}", t.spoken);
                println!("  heard: {}   ({:.0}%)", CtcAlphabet::decode(&heard), 100.0 * e as f32 / target.len() as f32);
            }
        }
        println!("  -> {} takes, CER {:.1}%", group.len(), 100.0 * errors as f32 / chars.max(1) as f32);
    }
}
