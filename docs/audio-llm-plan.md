# Audio-text model plan

Goal: one model that reads text, speech or any mix of the two, and writes text,
speech or a mix, with the speech carrying more than its words (tone, emphasis,
pauses, emotion) in both directions. It thinks in text and decides what, and
how much, it says out loud.

## The shared unit is the word

The backbone autoregresses over words. Every word reaches it as one embedding
`e_w`, from the text encoder or from the audio encoder, and every word it
predicts leaves through the character decoder and, when spoken, a speech
decoder. Modality is a property of each word, not of the window, so mixing is
free: a window can be text, then a spoken sentence, then text.

```
input  word i:  text encoder (chars)  ──┐
                audio encoder (frames) ─┴─> e_w ─> backbone ─> state_i
output word i+1: char decoder (always) ─> text
                 speech decoder (when spoken) ─> audio frames, conditioned on
                 state_i and the word's chars
```

An audio word is a written word whose spoken span is cut by the boundary head
(streaming) or by CTC forced alignment (training). Where written and spoken
differ ("$23" ↔ "twenty three dollars"), the written word is one alignment
group and its embedding reads the whole spoken span; punctuation joins the
word before it.

## Audio encoder state

The frame encoder is recurrent and keeps one state per continuous recording;
the segment encoder restarts at every word. A window of audio, text, audio from
two recordings therefore starts the frame encoder twice — the backbone, not the
encoder, carries what was said.

Live, a recording is one conversation session:

1. The wake word (smart-light's detector) starts the encoder from a zero state.
2. The user's question streams in word by word; the end of the turn is
   predicted by the model itself — after the last spoken word the backbone
   predicts the turn marker (`<SEP>`), from the falling intonation and the
   pause, rather than a fixed silence timeout. Needs spoken-question data
   with the marker after the last word (phase 4); a VAD timeout stays as the
   fallback.
3. Jarvis answers; the encoder keeps running and keeps its state (speaker,
   room), and its input is echo-cancelled against Jarvis's own output.
4. A follow-up within ~20 s continues the same session and backbone context;
   after ~20 s without speech the encoder stops and its state is dropped until
   the next wake word.

So a session is seconds to a few minutes of audio. The encoder has only ever
seen single utterances (≤ 35 s) from a zero state; it should also train on
chains of consecutive utterances up to a few minutes, with pauses between them,
so its state holds over a whole session.

## What is already known (stage E2, frozen `s4`)

- Per-word audio embeddings through a frozen RMSNorm at text scale drive the
  backbone like text (`docs/ae2-runs.md`). Off-manifold embeddings looked fine
  teacher-forced and derailed generation: check free-running output and norms.
- The CTC head is the strongest single lever (E) and its forced alignment is
  MFA quality (±1 frame 84 %, mean error 37 ms). New data needs a transcript,
  no timestamps.
- After an utterance the audio predicts the following text better than the
  transcript (E: next word 6.05 vs 8.18): the transcript has no punctuation, the
  audio has its pauses and intonation. A text teacher is not an upper bound
  where speech knows more, so no distillation towards it there.
- Scoring more continuation than own words costs the encoder (F).

## Phase 1 — finish the frozen-backbone encoder

1. F2: context windows with 4 scored words after the audio.
2. G: train-clean-360 (460 h) plus augmentation (room, ESC-50 noise, mic); one
   change at a time if G disappoints.
3. Self-alignment: replace MFA by the encoder's own CTC forced alignment, so any
   transcribed corpus can be added — LibriSpeech other-500, then large
   transcribed sets (GigaSpeech, MLS English, People's Speech; all on the HDD).
   Long recordings are cut by CTC segmentation. The boundary head keeps
   training on these CTC-derived ends; agreement between the two is logged.
   Where the CTC alignment fails or is unsure, a count loss alone (the
   boundary probabilities of an utterance summing to its word count, as in
   CIF's quantity loss) still trains the head without any timing.
4. Written-vs-spoken groups: normalize the transcript to the spoken form for
   CTC, keep the written words for the text model, map the aligned spoken spans
   back onto written words.

Exit: on held-out speakers, own-word gap small, "next word" and "text after"
at or below the teacher, CER and boundary F1 still improving with data, and
free-running continuations as good as from the transcript.

## Phase 2 — interleaved joint pretraining

The backbone is trained again, together with the audio encoder, inside normal
pretraining.

Window kinds, mixed per batch:

- **text** — the existing corpus, unchanged; by far the most windows. It is
  what keeps the text ability.
- **text with spoken spans** — a text document where random sentences are
  replaced by speech of the same sentence. Speech comes from a TTS rendering of
  the corpus text (many voices, rates and styles, then augmentation), so the
  text around it keeps its punctuation and the span's text is known exactly
  (no alignment needed: the TTS gives durations).
- **speech with text spans** — real transcribed speech (phase 1 corpora) as
  running chapters, with random spans given as text instead. Real voices and
  real prosody; alignment from CTC.

Losses: next-word CE on every text target (as now); the boundary and CTC heads
on every audio span; and, on TTS spans only, a consistency term: the same window
once with the span as text and once as speech, KL from the text view's
predictions on the words after the span to the speech view's. That is the
direct "audio gives the same context as text" constraint, and it is valid there
because the text view has the punctuation. On real speech there is no such
term: the speech may know more than its transcript.

Guards: the text validation (`hvg` on `VAL_DATA`) must not degrade; the
backbone learns at a fraction of the audio encoder's rate at first; text
windows stay the majority.

Code it needs: a window whose backbone rows come per word from the text
encoder or from given rows, with the backward reaching both encoders (today
`given` replaces the encoder for the whole window); the mixed window builder;
the paired-view KL (the distillation mechanism in `gpu::Hierarchical` already
does the decoder side).

## Phase 3 — speech output

A speech decoder per word, beside the character decoder: input the backbone
state (which carries the conversation's tone) and the word's characters, output
the word's audio frames, with its duration predicted. Speech is always
rendered from text the model has already written, so what is said is exactly
what was decided (text-first, like Moshi's inner monologue).

- Feedback: the backbone reads back each spoken word as its text embedding
  (exact, so content errors cannot accumulate) plus the speech decoder's own
  compact latent for that word (how it was said, so tone stays consistent
  across words) — never the waveform re-encoded by the audio encoder, which
  would be slow and feed rendering errors back as content.
- What is spoken: text inside `<think>…</think>` (already in the vocabulary) is
  silent; the rest of a reply is spoken unless marked otherwise. Instructions
  like "say only the number" are SFT data, not architecture.
- How it is spoken: the decoder sees the backbone state, so the same words come
  out differently in different contexts; style can be steered by text in
  context.
- Representation: mel frames plus a vocoder, or a neural codec (open weights,
  12.5–25 Hz). Decision below.
- Data: LibriTTS-R and the phase 1 corpora for voice; expressive sets for
  emotion later. Backbone frozen first, then joint.

## Phase 4 — spoken assistant

SFT where the user turns are speech (real recordings where there are any; the
existing SFT corpus rendered by TTS with many voices plus augmentation
otherwise) and the replies are thought in text and spoken. Tasks whose answer
depends on how something was said (tone, emotion, emphasis) so the backbone
learns to use what audio adds. Later: full duplex (listening while speaking).

### First audio SFT (2026-10-29)

Started before phase 2, straight from pretrained `s4`, with the `s4_T` encoder
(trained against frozen `s4`, so its embeddings fit this backbone exactly).

- Corpus `data/mix/assistant_audio.jsonl` = `assistant_full` (815k conversations)
  with the user turns of 20% of them heard (`scripts/sft_speech.py`, then
  `scripts/sft_audio_mix.py`). A conversation is typed or spoken, never both,
  so no reply is trained twice. Drawn among the 385k conversations that have a
  speakable user turn (one paragraph, ≤ 55 words, no code or markup; numbers,
  amounts, percentages, ordinals and years written out); every speakable user
  turn of a drawn conversation is spoken by one Piper LibriTTS-R speaker
  (904), the rest stay typed, so text and speech also mix inside one
  conversation. ~321k takes, MMS-aligned; FLAC on the data SSD
  (`~/training_data/sft_speech`). The replies are the text corpus's own.
- A message carries `"audio": "<session>/<id>"`; `sft::load_jsonl_heard` writes
  the take's spoken words in place of the turn and records their token range,
  and a loader without the takes reads the turn's text.
- `SftAudio` (`gpu/audio_train.rs`): each heard turn is one encoder row, half
  pooled at the boundary head's own cuts; the backbone reads those rows from
  the audio, the reply trains with KL `SFT_AUDIO_DISTILL` 0.5 towards the
  reply's predictions after reading the same words; the encoder steps at
  `SFT_AUDIO_LR` 2e-5 with its CTC and boundary heads. Smoke test: heard and
  read reply loss equal (2.40 / 2.40), CER 6–8%, 7.5 GB, no slower than text.
- Run `s4_sft_audio` (`models/after_speech.sh` as the systemd user unit
  `after-speech`): ~330 examples/min, the full epoch ~41 h. Until 50% of the
  epoch only 291k of the 321k takes loaded (19.0% heard): `spelled` rebuilt
  "heavy-tailed" as "heavy- tailed", one word more than spoken, and dropped the
  take. With that fixed 318k load (19.9% heard); the rest are words longer than
  `MAX_WORD_BYTES` (~1.7k) and takes faster than 6 words/s (~1.8k).
- Inference: the harness talks to it. `Hierarchical::chat_feed_embeddings`
  steps the backbone on heard words; `harness/src/ears.rs` runs a take through
  `<model>.ae2` (boundary-head cuts, CTC transcript as the turn's text);
  `Session::push_heard` feeds the embeddings in place of the turn's tokens.
  Window: TALK chip or hold F2 (Space when no field is focused); terminal:
  `/hear <wav>`. At 2% of the epoch: the user's own voice is heard nearly
  verbatim and answered on topic; VoiceBench commoneval (accented, noisy) is
  heard badly (CTC "tho she earthmobal is i emoba").
- Still to add: real voices (HeySQuAD questions with a paragraph as typed
  context and generated answers; the user's cloned voice), and spoken-QA
  evaluation on VoiceBench (real recordings, on the HDD, not trained on).

## Measures carried through every phase

- own-word gap, "next word", "text after", rotated-audio controls (`ae2` val)
- CER, boundary F1, CTC-vs-boundary agreement on unaligned data
- text-only validation loss of the backbone
- free-running continuations from speech vs from the transcript (`ae2s`)
- a prosody probe: emotion / question-vs-statement read from the backbone state

## Open decisions

1. TTS-rendered corpus text for the spoken spans: it is the only source of
   unlimited speech whose text has punctuation, but it means an external TTS as
   a data tool (never at run time).
2. Speech output as mel + vocoder (all ours, more work, slower to sound good)
   or a pretrained neural codec (sounds good early, an external component).
3. How much text quality joint training may cost, if any.
