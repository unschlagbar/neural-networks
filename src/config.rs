// ── config.rs ────────────────────────────────────────────────────────────────
//
// Zentrale Stelle für alle Hyperparameter.  Ändere sie hier — NICHT verstreut
// über main.rs. Die Begründungen stehen als Kommentare direkt bei den Werten,
// damit man beim Tuning weiß, was man verändert.

// ── Model-Pfade ──────────────────────────────────────────────────────────────

/// Pfad für das hierarchische Modell (char- + sentence-level).
pub const MODEL_LOC: &str = "models/hric";
/// Pfad für das „normale" Char-Level-Sequential-Modell.
pub const SEQ_LOC: &str = "models/seq";

// ── Sequenz-Längen ───────────────────────────────────────────────────────────

/// Trainings-Sequenzlänge (Anzahl Tokens pro BPTT-Chunk).
pub const SEQ_LEN: usize = u16::MAX as usize;
/// Reserve für den Forward-Cache (unsere BatchIter kann länger werden als SEQ_LEN).
pub const MAX_SEQ_LEN: usize = SEQ_LEN + 1024;

// ── Optimizer ────────────────────────────────────────────────────────────────
//
// Die alte LR = 1e-5 ist für ein Netz mit Residual-Pfaden und RMSNorm einfach
// zu klein. Deep-RNNs mit Pre-Norm laufen typisch bei 3e-4 – 1e-3. Weil wir
// SGD (keinen Adam) haben, bleiben wir konservativ am unteren Ende.
pub const LR: f32 = 4e-5;

/// Gradienten-Akkumulation: wir rufen `apply_grads` erst nach BATCH_SIZE
/// Sequenzen (Sequential skaliert automatisch mit 1/BATCH_SIZE).
/// BATCH_SIZE = 1 war ein Hauptgrund für das Rauschen. 8 ist ein guter
/// Kompromiss zwischen Stabilität und Update-Frequenz.
pub const BATCH_SIZE: usize = 1;

// ── Training-Schedule ────────────────────────────────────────────────────────

pub const EPOCHS: usize = 1;

/// Save nach jeweils N abgeschlossenen Files (0 = nur am Ende jeder Epoche).
pub const SAVE_EVERY: usize = 30;

/// Loss-Ausgabe alle N Trainings-Steps (= Forward-/Backward-Durchgänge über
/// ein Window). 0 = nur am Ende jeder Epoche flushen.
pub const PRINT_EVERY: usize = 10;

// ── Sampling ─────────────────────────────────────────────────────────────────

pub const MAX_LEN: usize = 1000;
pub const TEMPERATURE: f32 = 0.0;

// ── Modell-Dimensionen ───────────────────────────────────────────────────────

pub const CHAR_HIDDEN: usize = 64;
pub const OUT_HIDDEN: usize = 64;
pub const WORD_HIDDEN: usize = 128;

// ── Dataset ──────────────────────────────────────────────────────────────────

pub const DATA_DIR: &str = "political_speeches/";
pub const CHARSET: &str = "charset.txt";
