// ── config.rs ────────────────────────────────────────────────────────────────
//
// Zentrale Stelle für alle Hyperparameter.  Ändere sie hier — NICHT verstreut
// über main.rs. Die Begründungen stehen als Kommentare direkt bei den Werten,
// damit man beim Tuning weiß, was man verändert.

// ── Model-Pfade ──────────────────────────────────────────────────────────────

/// Pfad für das hierarchische Modell (char- + sentence-level).
pub const MODEL_LOC: &str = "models/hierarchical";
/// Pfad für das „normale" Char-Level-Sequential-Modell.
pub const SEQ_LOC: &str = "models/seq";

// ── Sequenz-Längen ───────────────────────────────────────────────────────────

/// Trainings-Sequenzlänge (Anzahl Tokens pro BPTT-Chunk).
pub const SEQ_LEN: usize = 1024;
/// Reserve für den Forward-Cache (unsere BatchIter kann länger werden als SEQ_LEN).
pub const MAX_SEQ_LEN: usize = SEQ_LEN + 1024;

pub const LR: f32 = 1e-4;
pub const BATCH_SIZE: usize = 1;

// ── Training-Schedule ────────────────────────────────────────────────────────

pub const EPOCHS: usize = 100;

/// Save nach jeweils N abgeschlossenen Files (0 = nur am Ende jeder Epoche).
pub const SAVE_EVERY: usize = 30;

/// Loss-Ausgabe alle N Trainings-Steps (= Forward-/Backward-Durchgänge über
/// ein Window). 0 = nur am Ende jeder Epoche flushen.
pub const PRINT_EVERY: usize = 10;

// ── Sampling ─────────────────────────────────────────────────────────────────

pub const MAX_LEN: usize = 1000;
pub const TEMPERATURE: f32 = 0.2;
pub const TOP_P: f32 = 1.;

// ── Modell-Dimensionen ───────────────────────────────────────────────────────

pub const CHAR_HIDDEN: usize = 64;
pub const OUT_HIDDEN: usize = 128;
pub const WORD_HIDDEN: usize = 384;

// ── Dataset ──────────────────────────────────────────────────────────────────

pub const DATA_DIR: &str = "data/political_speeches/";
pub const CHARSET: &str = "charset.txt";
