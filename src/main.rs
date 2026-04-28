// ── main.rs ──────────────────────────────────────────────────────────────────
//
// Dünner Dispatcher. Die eigentliche Logik liegt in config/model/training/
// sampling. Argumente:
//
//   (nichts / leere Zeile)  → train_normal     (sLSTM-Block-Stapel)
//   "h"                     → train_hierarchical
//   "s"                     → sample_normal
//   "hs"                    → sample_hierarchical
//
// Die alte Variante „leere Zeile = train, sonst = sample_old" war zu
// implizit, man konnte den hierarchischen Pfad gar nicht direkt ansteuern.

fn main() {
    neural_networks::run();
}
