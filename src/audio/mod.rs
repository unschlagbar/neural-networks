//! Audio shared by everything that listens: the wake word (`smart-light`) and
//! the speech encoder. Signal basics (`fft`, `spectrum`, `mel`, `io`) are used by
//! both; `frontend` and `segments` are the speech encoder's features and layout,
//! whose device side is `gpu::audio`. All of it runs without a GPU.

pub mod fft;
pub mod frontend;
pub mod io;
pub mod mel;
pub mod segments;
pub mod spectrum;
