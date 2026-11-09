# Where the audio encoder loses (2026-10-22, encoder O, run Q in flight)

All numbers are `ae2e` on held-out takes (`models/s4_O.sets_eval.log`): CER of the character head, "own" = gap of the heard words' next-word loss to the frozen text model reading the transcript, "after" = the same on the text after the audio.

| set | CER | own gap | after gap | word match | boundary F1 |
|---|---|---|---|---|---|
| TTS questions | 4.0 | +0.42 | +0.25 | 97.9 | 0.98 |
| TTS text | 5.9 | +0.69 | +0.39 | 98.0 | 0.97 |
| LibriSpeech | 7.4 | +0.23 | +0.37 | 96.8 | 0.96 |
| clone of the user | 8.5 | +0.77 | +0.40 | 98.1 | 0.96 |
| VoxPopuli | 15.1 | +0.66 | – | 92.2 | 0.92 |
| **user's voice** | **18.4** | **+1.34** | +0.41 | 94.2 | 0.90 |
| Common Voice | 19.1 | +0.74 | +0.56 | 94.7 | 0.89 |
| SLURP | 24.5 | +0.61 | – | 89.0 | 0.84 |

## 1. Live segmentation (largest measured gap at inference)

Training pools every word at reference ends (MMS/MFA); live, the boundary head cuts. On LibriSpeech val the same encoder loses **+0.37** on the heard words under its own cut (gap +0.23 → +0.60) and word match falls 96.8 → 85.7%; the text after barely moves (+0.37 → +0.40). Boundary F1 is 0.84-0.90 on real microphones.

Run **R** (from Q, `AUDIO_HEAD_CUT_P` 0.5): half the training utterances are pooled at the head's own count-constrained cut, the head itself still learns from the reference ends. Judge it by `segmentation_report`'s "boundary head" line against Q's.

Still open: the cut used here knows the word count; live, the peak decoder decides how many words there are. A count error shifts every later word against the text; that path is untrained.

## 2. The user's voice: twice the gap at the same CER

Common Voice and the user's voice have the same CER (19.1 vs 18.4%) but the heard words' gap differs by 2x (+0.74 vs +1.34). Recognising the characters is not the whole problem: the embeddings of this speaker carry less of the word than for an average Common Voice speaker. The clone (synthetic, same voice) is at +0.77 with 8.5% CER, so it is the real recordings — accent, microphone, room, speaking style — not the timbre.

What helps according to the literature: personalisation works with very little data when the data is chosen by the model's errors. Awasthi et al. (arXiv 2103.03142) cut accented WER from 53 to 19 on one speaker with 150 sentences selected by an error model; random sentences of the same count helped less. The user has 388 takes (33 min), chosen at random.

Next steps:
- **Error-driven prompts**: the VOICE page should offer pretraining sentences whose words the current encoder mishears most (from CTC errors on the user's takes and on the clone), not random ones.
- More takes per session in the setups the user actually uses (distance, quiet speech); the augmentation now covers quiet speech, real recordings of it are better.
- Validation of the voice is 38 takes; differences under ~0.1 between checkpoints are noise there.

## 3. Real microphones in general (CER 15-25% vs 4-8% on TTS)

Every step that added real speech paid most: Common Voice took the user's CER 27 → 21% (M → N). Q doubles Common Voice (733k takes). Sets on the HDD are drawn in runs of 32 (`TAKE_BLOCK`) so they do not cost a seek each.

Not done yet: self-supervised pretraining. BEST-RQ (arXiv 2202.01855) — a frozen random-projection quantizer with masked prediction — is the simplest such objective and works for streaming encoders; its gains are largest when labels are scarce. We have ~4000 h aligned, but MLS (~600 h, no alignment needed for SSL) and People's Speech are unlabeled-usable. A larger project: a pretraining phase for the frame stack before `ae2`.

## 4. The text after the audio

+0.25 to +0.56 everywhere and moving slowly across M → Q. The text model is frozen; how a backbone continues after heard words is what the joint stage (`hga`) trains. Within `ae2`, α 0.5 on document windows (O) recovered what pure distillation (N) had lost.
