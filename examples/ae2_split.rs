//! Cuts an utterance where the boundary head of an `.ae2` checkpoint puts word
//! ends and writes every piece as its own WAV, to hear what the head calls a word.

use std::fmt::Write as _;
use std::path::PathBuf;

use neural_networks::audio::io::Clip;
use neural_networks::audio::librispeech::LibriSpeech;
use neural_networks::audio::segments::{ENC_FRAME_S, Segment};
use neural_networks::config::{AUDIO_DATA, AUDIO_SPLITS, AUDIO_SR};
use neural_networks::gpu::Gpu;
use neural_networks::gpu::audio::AudioEncoder;

/// Silence between words in `gaps.wav`, seconds.
const GAP_S: f32 = 0.4;
/// Fade at each cut, seconds, so a cut does not click.
const FADE_S: f32 = 0.004;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [ckpt, input, out_dir] = &args[..] else {
        eprintln!("usage: ae2_split <ckpt.ae2> <dev-clean id | wav/flac> <out dir>");
        return;
    };
    let gpu = Gpu::new().expect("CUDA GPU");
    let (mut enc, ck) = AudioEncoder::load(&gpu, ckpt).expect("checkpoint");

    let utterance = if PathBuf::from(input).exists() {
        None
    } else {
        let corpus = LibriSpeech::load(AUDIO_DATA, AUDIO_SPLITS[0]).expect("dev-clean");
        Some(
            corpus
                .utterances
                .into_iter()
                .find(|u| &u.id == input)
                .unwrap_or_else(|| panic!("{input} is neither a file nor a dev-clean id")),
        )
    };
    let audio = utterance
        .as_ref()
        .map_or_else(|| PathBuf::from(input), |u| u.audio.clone());
    let clip = Clip::read_at(&audio, AUDIO_SR).expect("audio");
    let heard = enc
        .hear(&gpu, &ck.stats, &clip.samples)
        .expect("shorter than one encoder frame");
    let mut segs = enc.boundary_words(&heard).0;
    // Audio after the last end has not been called a word yet.
    let tail = segs.last().map_or(0, |s| s.end);
    if tail < heard.t_out {
        segs.push(Segment {
            start: tail,
            end: heard.t_out,
        });
    }

    std::fs::create_dir_all(out_dir).unwrap();
    let to_sample = |frame: usize| ((frame as f32 * ENC_FRAME_S * AUDIO_SR as f32) as usize).min(clip.samples.len());
    let fade = (FADE_S * AUDIO_SR as f32) as usize;
    let gap = vec![0.0; (GAP_S * AUDIO_SR as f32) as usize];
    let mut all = Vec::new();
    let mut listing = String::new();
    writeln!(listing, "{} at step {}, {}", ckpt, ck.step, audio.display()).unwrap();
    writeln!(listing, "character head: {}", heard.ctc).unwrap();
    if let Some(u) = &utterance {
        writeln!(listing, "transcript: {} ({} words)", u.text(), u.words.len()).unwrap();
    }
    writeln!(listing, "{} pieces (last = audio after the last end)\n", segs.len()).unwrap();

    for (i, s) in segs.iter().enumerate() {
        let (a, b) = (to_sample(s.start), to_sample(s.end));
        let mut piece = clip.samples[a..b].to_vec();
        let n = piece.len();
        for k in 0..fade.min(n / 2) {
            let g = k as f32 / fade as f32;
            piece[k] *= g;
            piece[n - 1 - k] *= g;
        }
        let (t0, t1) = (s.start as f32 * ENC_FRAME_S, s.end as f32 * ENC_FRAME_S);
        let name = format!("{i:02}_{t0:.2}-{t1:.2}.wav");
        let words = utterance.as_ref().map_or(String::new(), |u| {
            let inside: Vec<&str> = u
                .words
                .iter()
                .filter(|w| (t0..t1).contains(&(0.5 * (w.start + w.end))))
                .map(|w| w.text.as_str())
                .collect();
            match inside.len() {
                0 => "  (no MFA word)".into(),
                1 => format!("  {}", inside[0]),
                _ => format!("  {}  <- {} words", inside.join(" "), inside.len()),
            }
        });
        writeln!(listing, "{name}{words}").unwrap();
        all.extend_from_slice(&piece);
        all.extend_from_slice(&gap);
        Clip::new(piece, AUDIO_SR).write(format!("{out_dir}/{name}")).unwrap();
    }
    Clip::new(all, AUDIO_SR).write(format!("{out_dir}/gaps.wav")).unwrap();
    std::fs::write(format!("{out_dir}/segments.txt"), &listing).unwrap();
    print!("{listing}");
}
