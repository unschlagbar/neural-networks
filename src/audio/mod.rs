//! Audio shared by everything that listens: the wake word (`smart-light`) and
//! the speech encoder. Signal basics (`fft`, `spectrum`, `mel`, `io`) are used by
//! both; `frontend` and `segments` are the speech encoder's features and layout,
//! whose device side is `gpu::audio`; `librispeech` reads its first corpus and
//! `voice` the takes recorded on the harness VOICE page.

pub mod augment;
pub mod ctc;
pub mod fft;
pub mod frontend;
pub mod io;
pub mod librispeech;
pub mod mel;
pub mod segments;
pub mod spectrum;
pub mod voice;
