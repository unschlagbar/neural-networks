# Voice recorder and everyday-writing data — plan for a new session

Two related additions:

1. a page in the harness that shows a sentence, records the user speaking it,
   and writes a speech dataset with exact transcripts, so the audio encoder
   learns the user's voice, microphone and room;
2. new pretraining text in everyday human writing (chat, forum, messages: typos,
   no capitals, slang, run-on sentences), so the text model reads the user's
   own style.

The two share one rule: the sentences the user reads aloud come from a
held-out split that never enters pretraining or SFT, so recordings stay usable
as an honest test of the user's voice.

## Part A — sentence pool

`datamix` builds it, so the filters and the report are the ones every other
corpus goes through.

- Sources, in rough shares:
  - **Jarvis commands** (~30 %): expansions of `mixes/synth/*.syn` (lamp,
    apps, math, media, browser) — the sentences the model will actually hear.
    Expanded fresh, never the exact strings in the SFT corpus.
  - **everyday writing** (~40 %): sentences from the held-out split of the
    Part B corpus.
  - **general text** (~30 %): sentences from a corpus not in `pretrain_v2`
    (e.g. a Wikipedia or news dump; check against the pretraining sources).
- Split documents into sentences (punctuation + capital heuristics), keep 4–25
  words, drop URLs, code, emoji-only, numbers-heavy and non-English lines, and
  exact-dedup against every pretraining and SFT shard (hash of the normalized
  sentence).
- Output `data/voice/prompts.jsonl`: `{"id", "source", "text", "spoken"}`.
  `spoken` is the text as it is said — digits, symbols and abbreviations written
  out ("$23" → "twenty three dollars", "u" → "you"), lower-case, no punctuation:
  what CTC aligns against. `text` stays as written, for the text model.
  Normalization is a small rule set in `datamix`; a prompt it cannot normalize
  is dropped rather than guessed.

## Part B — everyday-writing pretraining corpus

- Candidates (check licence and size before fetching):
  - **WildChat-1M user turns** — already on disk
    (`~/training_data/wildchat`); only the GPT-4 half's conversations are in
    SFT today. User turns alone, both halves, as plain text documents: real
    people typing to a chatbot, exactly the register the user writes in.
  - **Reddit comments** (a curated HF dump), **forum/Q&A threads** (Stack
    Exchange non-technical sites), **IRC/Discord-style chat logs** (e.g. the
    Ubuntu dialogue corpus) — multi-person informal dialogue.
  - The user's own writing, if they want: exported chats or notes, kept
    private, never published.
- Filters: `min_alpha_ratio`, `max_dup_line_ratio`, near-dedup, toxicity /
  personal-data blocklists; keep typos and lower case — that is the point.
  Strip quoted replies and bot boilerplate.
- A new mixture (`mixes/everyday.toml`) writing a `text` corpus with
  `<|endoftext|>` separators, plus the held-out split for Part A
  (deterministic by document hash, ~1 %).
- Then a pretraining mixture that adds it at a modest share (5–10 % of tokens)
  to continue `s4`/`s5` pretraining; compare text val on `VAL_DATA` and on a
  held-out everyday split before and after.

## Part C — recorder page in the harness

A new page beside the OCR and SFT pages (`harness/src/ui/voice.rs` +
`harness/src/voice.rs` for the worker), same message-range pattern
(`MSG_FIRST..MSG_LAST`).

- **Capture**: port `smart-light/src/mic.rs` (cpal, pooled hop buffers) into
  the harness or a small shared module; record at the device rate, resample
  to 16 kHz mono with `neural_networks::audio::io::Clip::resampled`, never
  normalize loudness (it is part of how something was said).
- **Flow**: show the prompt's `text` large, `spoken` small beneath it; hold
  space (or a button) to record, release to stop; auto-trim leading/trailing
  silence beyond 300 ms; level meter while recording, clip warning above
  −1 dBFS; keys for play back, re-record, skip (marks the prompt skipped, never
  shown again), next. A free-speech mode records something the user says on
  their own and asks for the transcript afterwards.
- **Metadata per session**: speaker, microphone (device name from cpal),
  distance (near / desk / across the room), background (quiet / music / TV /
  birds…), set once and changed when the setup changes — the encoder's
  augmentation is judged against exactly these.
- **Storage**: `~/training_data/voice/<speaker>/<session>/<id>.wav` (16 kHz,
  16-bit) plus one `manifest.jsonl` line per take: `{"id", "prompt", "text",
  "spoken", "seconds", "device", "distance", "background", "session", "time"}`.
  On the HDD symlink like the LibriSpeech splits if the SSD fills.
- **Progress**: count and minutes recorded, per source and per condition, so
  the user can aim for balance. 10 minutes is a voice test, 1–2 hours makes
  fine-tuning worthwhile.

## Part D — using the recordings in `ae2`

- A corpus reader for the manifest beside `audio::librispeech`, producing the
  same `Utterance` shape. There is no MFA: word segments come from CTC forced
  alignment by the current encoder (MFA quality on LibriSpeech, ±1 frame 84 %),
  re-aligned when the encoder improves; the boundary head trains on those ends
  plus `CountLoss`, which needs only the word count.
- Speaker split: hold out whole sessions, never single takes, so the val set
  is a new day / room.
- Metrics on the user's voice: CER, boundary F1 against the CTC ends, and the
  own-word / next-word losses, reported apart from LibriSpeech val.
- Training: mixed into the LibriSpeech batches at a small share (one
  utterance per batch), clean and augmented.
- Later: the same page records spoken commands for the SFT stage (Phase 4 of
  `docs/audio-llm-plan.md`).

## Order for the session

1. Part C capture + page with a stub prompt list (to hear the recording path
   end to end on the real microphone).
2. Part A pool from `.syn` expansions only, then the other sources.
3. Part D reader + CTC alignment, a first val number on ~10 minutes of voice.
4. Part B corpus and its pretraining mixture.

## Status (2026-10-17)

- **Part C done (capture + page).** `harness/src/mic.rs` (cpal capture at the
  device rate, playback), `harness/src/voice.rs` (worker), `harness/src/ui/voice.rs`
  (VOICE chip, messages 500-599, `HARNESS_PAGE=voice`). 300 ms pre-roll, trim
  to 300 ms of silence around speech, clip warning at -1 dBFS, setup persisted
  in `<speaker>/setup.txt`, skips in `skipped.txt`. Each take is
  `<session>/<id>.wav` with its transcript beside it as `<id>.txt`, and one
  line in `<speaker>/metadata.jsonl` whose `file_name` column makes the speaker
  folder a Hugging Face `audiofolder` dataset as it stands (the plan's
  `manifest.jsonl`). Downsampling in
  `Clip::resampled` now low-passes first (the linear `Resampler` alone folded
  everything above 8 kHz into the band).
- **Part A started.** `datamix` output `kind = "prompts"` (`datamix/src/prompts.rs`):
  sentence split, gates, spoken normalization, held out against `[output]
  exclude` by streaming the corpora against the staged prompts (33 GB
  `pretrain_v2` in ~100 s; `exclude` takes `shard.*.txt` patterns).
  `mixes/voice_prompts.toml` reads the 3253 documents `pretrain_v2` held out
  (`shard.txt.eval`, never trained on; FineWeb itself was used up) and the
  `.syn` sources: 81.3k prompts, 143 of them commands, because the templates'
  phrase space is ~all in `tools.jsonl` already.
  Every prompt carries `doc` + `index`: the record it was cut from (kept in
  `data/voice/prompts_docs.jsonl`) and its position among that record's
  `prompts::sentences`, and the recorder copies both into each take's metadata.
  That is what lets Phase 2 of `docs/audio-llm-plan.md` put a take back between
  its neighbours (text, spoken span, text) instead of using it only alone. Prompts are 4-100 words, with those over 25 kept at
  10 % so the encoder sees every length. Commands are meant to be a small share:
  the pool needs everyday and general text next.
- **Not yet:** an `exclude` for SFT/text outputs, so a rebuilt SFT corpus cannot
  pick up a prompt sentence; the user turns of generated tool data that happen
  to match.

## Typos: pretraining vs post-training

Everyone pretrains on noisy text (Common Crawl, Reddit, forums). Output
register comes from post-training, whose assistant turns are clean. Here SFT
loss covers assistant turns only, so typo-heavy text as *user* turns teaches
reading that style without writing it. That is the place for WildChat user
turns and the user's own chats. Pretraining keeps everyday writing at the
modest 5-10 % share above, and SFT assistant turns stay clean.

## Open questions for the user

- Which microphone(s) and rooms Jarvis will live with — record those.
- Whether their own chats/notes may go into Part B.
- Prompt display: as written (practises reading typos aloud, transcript fixed
  by `spoken`) or a cleaned version.
