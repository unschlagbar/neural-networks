# ae2 runs

Audio encoder against frozen `s4` (mode `ae2`). Val = the 4 held-out dev-clean
speakers (250 utterances, 0.5 h); teacher next-word loss on val 6.341. Train =
dev-clean rest + train-clean-100 (28 838 utterances, 97.7 h), batch 8,
`AUDIO_LR` 3e-4 with cosine to `AUDIO_E2_STEPS` = 10 000.

"rotated" is the control: the same embeddings, each word handed the one from
half an utterance away. `rotated − student` is how much word identity the
embeddings carry. Boundary F1 is the peak decoder (p > 0.5, radius 1). CTC is a
probe (`CTC_WEIGHT` 0): a linear head on the frames, no gradient into them.

## A — stem stack 7, stride 4 (30 ms overlap), output RMSNorm at text scale

| step | student | rotated | rotated − student | gap | boundary F1 | CTC CER |
|---|---|---|---|---|---|---|
| ~3000 | 7.600 | 8.098 | 0.50 | +1.263 | — | — |
| ~3200 | 7.569 | 8.035 | 0.47 | +1.228 | 0.696 | 72.0 % |
| 4000 | 7.463 | 8.112 | 0.65 | +1.122 | 0.770 | 68.2 % |

At 4000 (`ae2a`, val): peak F1 0.770 (P 0.796, R 0.746, 0.94 ends per word);
count-constrained alignment with duration prior ±1 frame 46.9 %.

## B — stem stack 4, stride 4 (no overlap), otherwise as A

| step | student | rotated | rotated − student | gap | boundary F1 | CTC CER |
|---|---|---|---|---|---|---|
| 4000 | 7.498 | 8.099 | 0.60 | +1.157 | 0.779 | 69.9 % |
| 7800 | 7.294 | 8.243 | 0.95 | +0.953 | 0.825 | 65.0 % |

A vs B at 4000: gap +0.035 worse, word identity 0.05 lower, boundary F1 0.009
higher, CER 1.7 points higher — within what a second seed would move, so the
stem overlap buys nothing measurable here. B kept (smaller stem).

At 7800 `ae2s` shows content for the first time: "What are you doing here he
asked" continues `you came to my house and said to me, "I am the one who has
the right to do this." I said to him, "What are you doing here?"`; the CTC probe
hears "you have been to the hotel he burst out you have seen catherine" as
`uuticothe otll he prsohh itssemn catr e`.

## C — STFT 20 ms windows without overlap (320/320, 50 Hz), stem stack 2 stride 2, otherwise as B

Encoder frames stay 40 ms. B's checkpoint kept as `models/s4.ae2.B-stack4`.

| step | student | rotated | rotated − student | gap | boundary F1 | CTC CER |
|---|---|---|---|---|---|---|
| 4000 | 7.547 | 8.014 | 0.47 | +1.206 | 0.742 | 75.8 % |

C vs B at 4000: worse on every column — gap +0.049, word identity −0.13,
boundary F1 −0.037, CER +5.9 points. Larger than A vs B and all one way, so
dropping the STFT overlap costs; back to 400/160 with the stack-4 stem.

## D — B plus passages: up to 4 consecutive utterances of a chapter as one text window

The text model reads a passage as running text instead of each utterance
alone; the encoder still hears every utterance on its own. Teacher on val
5.934 per passage (6.341 per utterance before), so student, rotated and gap are
on a new scale and only compare within D onward. Step ~220 ms (B ~480 ms):
fewer, larger text windows and the next batch loaded on a thread.

| step | student | rotated | rotated − student | gap | boundary F1 | CTC CER |
|---|---|---|---|---|---|---|
| 4000 | 7.455 | 7.885 | 0.43 | +1.521 | 0.712 | 73.8 % |

D vs B at 4000: the teacher gained 0.41 from context, the student nothing
(7.455 vs 7.498), so the gap widened; word identity (0.43 vs 0.60), boundary F1
(0.712 vs 0.779) and CER (73.8 vs 69.9 %) are all worse — and the last two never
see the text model. The likely cause is the batch: four utterances of a passage
share speaker, chapter and recording, so a batch of 8 holds about 2 voices
instead of 8. Context should come without that — e.g. the preceding transcript
as text in front of one utterance's audio. Passages off (`AUDIO_PASSAGE` = 1)
until then; the faster pipeline stays.

## E — B (one utterance per window) plus `CTC_WEIGHT` 0.1, with D's faster pipeline

| step | student | rotated | rotated − student | gap | boundary F1 | CTC CER |
|---|---|---|---|---|---|---|
| 4000 | 7.247 | 8.362 | 1.12 | +0.906 | 0.827 | 47.2 % |
| 8600 | 6.861 | 8.685 | 1.82 | +0.520 | 0.891 | 31.9 % |
| 10000 | 6.840 | 8.725 | 1.89 | +0.499 | 0.894 | 31.4 % |

E vs B at 4000: better on every column and by far more than seed noise — gap
−0.25, word identity almost doubled (1.12 vs 0.60), boundary F1 +0.048, CER
−22.7 points; at 4000 E already beats B at 7800 everywhere. The dense
per-frame character signal is what the frame stack was missing; the word
embeddings gain from it, not only the character head. `CTC_WEIGHT` 0.1 stays.

`ae2s` at 4000: the character head hears "nor is mister quilter's manner less
interesting than his matter" as `noresmetce coters manr les intr sttin hen his
mater` and "you have been to the hotel he burst out you have seen catherine" as
`ho hentethottell he pars d he ot seen tathervee`. The audio continuations are
fluent but not yet on topic.

At 8600 (2.4 epochs) the gap has nearly halved again (+0.520, B at 7800
+0.953) and word identity is 1.82. `ae2s`: `nores miscer clers manara less
interesting than his mateer`, `you ae hento th hot hal he barstout hou of
sseeme catevren`; names and run-together words are most of what is left.

At 10 000 (`ae2a`, val): peak F1 0.894 (P 0.893, R 0.895, 1.00 ends per word);
CTC forced alignment ±1 frame 83.7 %, ±2 95.8 %, mean error 37 ms, bias +0.19
frames — no better than the probe-only head of B (84.5 %), although the CER fell
from 65 to 31 %: the timing of the last character does not depend on reading it
right. Train sample ±1 85.7 %, so no overfit in the timing either. Checkpoint
kept as `models/s4.ae2.E-ctc`.

### E under the context windows (validation only, `ae2` on the finished checkpoint)

From here a val window is the utterance's audio followed by up to 24 words of
the chapter's next utterance as text (218 of 250 have one), its first letter
lower-cased: "next word" is the first word after the audio, predicted from the
state the audio alone leaves, "text after" all of them. The utterance's own
words now include its last word's trailing space, so "student/teacher" moved
(6.840 → 6.671, 6.341 → 6.257).

| | student | rotated | teacher | gap |
|---|---|---|---|---|
| own words | 6.671 | 8.537 | 6.257 | +0.414 |
| next word | 6.048 | 7.272 | 8.176 | −2.127 |
| text after | 5.975 | 6.225 | 6.163 | −0.188 |

After an utterance the audio beats the transcript: the transcript has no
punctuation, the audio has its pause and intonation, and the last word's
embedding can carry them. So the off-topic `ae2s` continuations are the text
model's, not the audio's, and distilling towards the transcript-reading teacher
would pull the student the wrong way exactly there (`DISTILL_ALPHA` 0).

## F — E plus context windows: up to 24 words of transcript after the audio, and before it with chance 0.5

Training scores every prediction from the first audio word on, the last audio
word is read whenever text follows, and the train log's student covers the
utterance and the text after it together (no teacher pass while
`DISTILL_ALPHA` is 0). Val as above, without the text before.

| step | own student | rotated | own gap | next word | text after | boundary F1 | CTC CER |
|---|---|---|---|---|---|---|---|
| 4000 | 7.364 | 8.004 | +1.107 | 6.186 (rotated 7.009) | 5.873 (rotated 5.984) | 0.783 | 58.9 % |
| 10000 | 7.090 | 8.153 | +0.833 | 6.092 (rotated 7.105) | 5.838 (rotated 5.973) | 0.847 | 47.2 % |

F vs E at 4000 (E then: own gap +0.906 against the old teacher, rotated −
student 1.12, F1 0.827, CER 47.2 %): the text after the audio is already better
than E ever got (5.873 vs E's 5.975 at 10 000), but everything the encoder
itself learns is worse — word identity 0.64 vs 1.12, F1 −0.044, CER +11.7
points — and the boundary and character heads never see the text model. The
loss is a mean over every scored word, so 24 words of continuation outweighed
the utterance's ~18 and took the encoder's capacity. Next: score only
`AUDIO_TRAIN_AFTER_WORDS` = 4 words after the audio in training (val still
reads 24).

F vs E at 10 000 (E: own 6.671 / gap +0.414 / rotated − student 1.87, next
word 6.048, text after 5.975, F1 0.894, CER 31.4 %): the text after is 0.14
better and the next word about level, but the encoder ends far behind — own
gap twice E's, word identity 1.06 vs 1.87, CER 47 vs 31 % — with the learning
rate already at zero, so it is not a slower start that would catch up.

## F2 — F with 4 scored words after the audio, plus the boundary word-count loss

`AUDIO_TRAIN_AFTER_WORDS` 4 (val still 24) and `CountLoss` on the boundary
head (`BOUNDARY_COUNT_WEIGHT` 0.1): its probability mass over an utterance
pulled towards word count × mass per word, CIF's quantity loss, which needs no
timing. The val line logs the relative count error.

| step | own student | rotated | own gap | next word | text after | boundary F1 | count err | CTC CER |
|---|---|---|---|---|---|---|---|---|
| 4000 | 7.341 | 7.959 | +1.084 | 6.189 (rotated 6.815) | 5.907 (rotated 6.030) | 0.779 | ±16.5 % | 61.7 % |
| 10000 | 7.098 | 8.069 | +0.841 | 6.119 (rotated 6.878) | 5.865 (rotated 6.014) | 0.834 | ±11.3 % | 48.5 % |

F2 at 4000 is F at 4000 within noise (own gap +1.084 vs +1.107, word identity
0.62 vs 0.64, F1 0.779 vs 0.783, CER 61.7 vs 58.9 %): the 24 scored words were
not what cost the encoder, and the count loss changes nothing yet. Left from
F's changes: the transcript before the audio in half the windows, and the last
audio word read with text after it. Next: F3 = F2 without the text before
(`AUDIO_PREFIX_P` 0).

At 10 000 F2 is F again (own gap +0.841 vs +0.833, CER 48.5 vs 47.2 %, F1
0.834 vs 0.847). Checkpoint kept as `models/s4.ae2.F2-cont4-count`.

## F3 — F2 without the transcript before the audio (`AUDIO_PREFIX_P` 0) and without the count loss

| step | own student | rotated | own gap | next word | text after | boundary F1 | count err | CTC CER |
|---|---|---|---|---|---|---|---|---|
| 4000 | 7.305 | 7.939 | +1.048 | 6.180 (rotated 6.760) | 5.938 (rotated 6.050) | 0.766 | ±41.3 % (no loss) | 63.0 % |

F3 at 4000 is F2 again: the transcript before the audio was not it either.
What is left is reading the last audio word with text after it and scoring
what follows — and that hurts even the boundary and character heads, which
never see the text model, so the suspect is the size of that gradient
(`examples/ae2_cont_grad.rs` measures it per audio row).

`ae2_cont_grad` on E's checkpoint, five held-out utterances: with 4 words of
text after the audio the stem's gradient is 0.75–0.85× what the utterance alone
gives (the loss is normalized over more rows), the last audio row gets about
as much as its neighbours, and no row is singled out. So the continuation's
gradient is not too large — the size hypothesis is wrong.

## E′ — E reproduced on the current code (`AUDIO_TRAIN_AFTER_WORDS` 0, no prefix, no count loss)

Training windows are then exactly E's (the utterance alone, its last word not
read). If E′ matches E, the continuation itself costs the encoder; if it
matches F, something else in the new code does.

| step | own student | rotated | own gap | next word | text after | boundary F1 | CTC CER |
|---|---|---|---|---|---|---|---|

## Built for G (not yet trained)

- **Written text (LibriSpeech-PC, openslr 145, `<root>/pc/<split>.json`).**
  The student still steps once per spoken word and its targets are the spoken
  words, now spelled and capitalized as written ("The" after a sentence end);
  punctuation is never a step of its own — it is heard as the pause and
  melody of the word before it. The teacher reads the punctuated text and is
  scored on the same spoken words (`Window::teacher_mask`), so it gets the
  punctuation the student has to hear: a fair target again. Maps cleanly for
  88 % of dev-clean and 82 % of train-clean-100 (the rest: MFA `<unk>`,
  "missus"/"misses"); those keep the unpunctuated transcript.
- **The user's voice (`audio::voice`, harness VOICE page).** 257 takes, 23.7
  min, one microphone (FIFINE 683, near, fan). Aligned by the encoder's own
  CTC (mode `ae2v`, E's checkpoint: CER 55 % on this voice against 31 % on
  LibriSpeech). 234 usable, 200 with punctuation, 154 with the text of the
  document they were read from around them. Session `2026-10-17_1534` (64
  takes, 57 usable) is held out as its own `val voice` line; a training batch
  carries one take with chance `AUDIO_VOICE_SHARE` 0.25.

## E′ — E reproduced on the current code (finished, 10 000 steps)

Validated with `ae2e` (eval only) on the code with written-text targets, so
the teacher now reads punctuation and the numbers are a new scale: E and E′
measured the same way.

| | own student | rotated | own gap | next word | text after | F1 | CER |
|---|---|---|---|---|---|---|---|
| E | 6.914 | 8.782 | +0.628 | 10.882 | 5.944 | 0.894 | 31.4 % |
| E′ | 7.008 | 8.345 | +0.722 | 10.574 | 5.868 | 0.878 | 36.2 % |

E′ lands near E, not near F (CER 47–48 %, F1 0.83–0.85 at 10 000): most of what
F lost is the continuation itself; the rest (CER +5, word identity −0.5) is a
second seed or something small in the new code, to keep an eye on. The next
word is now 4+ nats behind the teacher because the targets are capitalized
after a sentence end and neither encoder was trained on that — G is.
`AUDIO_TRAIN_AFTER_WORDS` stays 0.

**The user's voice** (`val voice`, held-out session, 57 takes):

| | own student | rotated | own gap | F1 (vs CTC ends) | CER |
|---|---|---|---|---|---|
| E | 8.778 | 9.487 | +4.090 | 0.646 | 53.8 % |
| E′ | 8.769 | 8.817 | +4.081 | 0.631 | 56.6 % |

On this microphone and voice the embeddings carry almost no word identity
(rotated − student 0.7 and 0.05): the LibriSpeech-only encoder does not
transfer. That is what G's own-voice takes and augmentation are for.

## G — 460 h (train-clean-100 + 360), written-text targets, the user's voice (one take per 4 batches), augmentation, no continuation, 40 000 steps

Same val scale as the E/E′ table above (punctuated teacher). Log:
`models/s4.ae2.G.log`; checkpoint `models/s4.ae2.G-460h`. ~285 ms/step.

| step | own student | rotated | own gap | next word gap | text after gap | F1 | CER |
|---|---|---|---|---|---|---|---|
| 29 400 | 6.375 | 8.623 | +0.089 | +0.662 | +0.163 | 0.939 | 18.3 % |
| 40 000 | 6.313 | 8.659 | +0.027 | +0.635 | +0.161 | 0.945 | 16.3 % |

Against E (gap +0.628, next word +4.6, F1 0.894, CER 31.4 %): the audio now
predicts the next word as well as the punctuated transcript does, and hears the
sentence ends the transcript writes as punctuation (next word +0.64 from +4.6).

**The user's voice** (held-out session, re-aligned with G's own CTC: F1 0.77 →
0.81 against those ends, gap unchanged — alignment was not the problem):

| | own gap | rotated − student | next word gap | CER |
|---|---|---|---|---|
| held-out session | +4.213 | 0.42 | +4.818 | 39.9 % |
| sessions it trained on | −0.836 | 5.29 | +5.063 | 10.8 % |

It learned the 177 training takes — better than the teacher on them, each was
heard ~55 times — and that does not carry over to new sentences in the same
voice. Voice, accent, microphone and fan together are far from LibriSpeech,
and 18 minutes cover them only for the sentences read. Needs more of this
voice and more accented, noisier speech in general, not more passes over the
same takes.

## G-2ep — G with every voice take heard twice (`AUDIO_VOICE_EPOCHS` 2)

All 426 takes re-aligned with G's CTC first (325 usable for training, 29.9 min; the same 57
held-out); the user had recorded ~140 more since G. So against G two things
change: two passes per take instead of ~56, and twice the sentences. Log
`models/s4.ae2.G2ep.log`.

| G-2ep | own gap | rotated − student | next word gap | F1 | CER |
|---|---|---|---|---|---|
| LibriSpeech val | +0.013 | 2.25 | +0.574 | 0.947 | 15.6 % |
| voice, held-out session | +3.648 | 0.72 | +3.256 | 0.655 | 44.2 % |
| voice, trained sessions | +3.272 | 0.86 | +2.827 | 0.677 | 43.2 % |

(F1 on the voice is against G's CTC ends, which G-2ep did not make.) Against G:
LibriSpeech a little better (gap +0.013 vs +0.027, CER 15.6 vs 16.3 %); on the
voice nothing memorized any more — trained and held-out takes score alike — and
the held-out words improve (gap +3.65 vs +4.21, word identity 0.72 vs 0.42), but
the character head hears the voice worse (CER 44 vs 40 %): two passes teach the
acoustics less than 56 did. The voice is data-limited either way.

## H — G-2ep with 8 voice epochs, 50 000 steps

Between the two: enough passes for the acoustics, few enough not to learn the
sentences; and longer, since G was still improving at its last step. Log
`models/s4.ae2.H.log`.

### Data after the night

train-other-500 downloaded (30.6 GB, MD5 checked) and unpacked on the HDD,
148 688 utterances with 148 645 MFA alignments; LibriSpeech-PC covers it too.
In `AUDIO_SPLITS` from the next run on (not in H): 265 445 usable utterances,
about 960 h, half of them the harder "other" speakers. The length cache holds
all of them, so a start reads no FLAC headers.

| H (50 000) | own gap | rotated − student | next word gap | F1 | CER |
|---|---|---|---|---|---|
| LibriSpeech val | −0.146 | 2.42 | +0.568 | 0.953 | 13.9 % |
| voice, held-out session | +3.171 | 0.92 | +3.857 | 0.815 | 39.0 % |
| voice, trained sessions | +2.007 | 1.96 | +3.566 | 0.861 | 34.0 % |

Best on everything that matters: the voice has G's acoustics (CER 39 %) and
better than G-2ep's words (gap +3.17), with a moderate train/held-out split
(+2.0 vs +3.2) instead of G's memorization. On LibriSpeech the audio now beats
the punctuated teacher on the utterance's own words (−0.15). The embeddings
still read only their own frames; what the audio has beyond the punctuation is
stress, pauses, rate, and the coarticulation at the end of a word with the next
one, all of it in the audio up to the word's end. Checkpoint
`models/s4.ae2.H-8ep`.

## I — H on all of LibriSpeech (960 h with train-other-500), 80 000 steps

Log `models/s4.ae2.I.log`.

I was paused at step 11 400 (`models/s4.ae2.I-paused`) to free the device for
the analysis below.

## Does the meaning arrive? (`ae2e` context report, H)

Next-word loss of the k-th word after the utterance: from the text after it
alone, after the transcript (teacher), after the audio (student), after the
audio words in rotated order. "Recovered" = (none − student) / (none − teacher).

| LibriSpeech, k | none | teacher | student | scrambled | recovered |
|---|---|---|---|---|---|
| 1 | — | 5.915 | 6.687 | 10.586 | — |
| 2 | 6.681 | 5.903 | 6.018 | 6.152 | 85 % |
| 4 | 6.413 | 5.480 | 5.963 | 5.924 | 48 % |
| 8 | 6.117 | 5.500 | 5.668 | 5.678 | 73 % |
| 16 | 5.224 | 4.976 | 5.098 | 5.027 | 51 % |

| voice (held out), k | none | teacher | student | scrambled | recovered |
|---|---|---|---|---|---|
| 2 | 5.317 | 4.045 | 5.770 | 5.932 | −36 % |
| 4 | 5.114 | 3.922 | 5.850 | 5.968 | −62 % |
| 8 | 4.462 | 4.135 | 4.972 | 4.913 | −156 % |
| 16 | 3.211 | 3.114 | 3.452 | 3.432 | −251 % |

On LibriSpeech about half of what the sentence tells the text model gets
through, and beyond the second word it does not depend on the order of the
audio words: the backbone gets which words were there, not the sentence. Only
the next word or two use the order, which is what the next-word loss rewards.
On the user's voice the state after the audio is worse than no context at
all. The next-word objective does not ask for the meaning; the next steps
are objectives that do: matching the backbone state after the audio to the
state after the transcript at the end of each word (the embedding stays free,
the understanding must agree), and KL to the teacher's next-word
distribution. A frozen-backbone path (input gradient only, no weight
gradients or their memory) would make both cheaper and let an evaluation run
beside training.

## The audio stage (mode `hga`, `AudioPretrain`) — built and smoke-tested

A backbone stage of its own, after text pretraining is through: `hg` stays
text only. Every `AUDIO_PRETRAIN_EVERY` (10) text windows the loop runs one
extra window: a run of consecutive LibriSpeech utterances (≤ 1024 words) as a
document of written text in which 3 utterances are heard instead of read. The
heard utterances run through the audio encoder as one recording from a fresh
state; their words' backbone rows come from it (`Hierarchical::
forward_backward_mixed`), everything else from the text encoder. Backbone,
decoder and text encoder accumulate into the same optimizer step as the text
windows; the encoder steps after each audio window (CTC and boundary heads
alongside); the backbone trains at the constant `AUDIO_STAGE_LR` (2e-5). The
backbone's carried state is stashed around the window
(`stash_carried`/`restore_carried`), so the text windows run exactly as
without it — both pinned by GPU tests (bit-exact). Encoder saved to
`<model>.ae2` with the model; it starts from `models/s4.ae2.H-8ep`.

Smoke test on a throwaway copy of s4 (deleted after): 7.6 GB on the device,
first audio window 997 words with 80 heard, word loss 4.3, CER 16 %.

Later additions to the audio stage: voice windows (a pretraining-holdout document with the user's takes heard where their sentences stand, `AUDIO_PRETRAIN_VOICE_P`), TTS windows (`AUDIO_PRETRAIN_TTS_P`), distillation towards the same window read (`AUDIO_PRETRAIN_DISTILL` 0.5), and a held-out heard-vs-read report scored from the first heard word to the end of the window. A short `hga` run on `s4_audio` (to step 1713000) showed the heard−read gap on training windows falling from ~0.26 to ~0.18. Paused: the user wants the encoder good on its own first.

## A live test: the acoustics are the bottleneck

The harness listen mode (same path as `ae2s`) gave fluent but unrelated continuations. On held-out takes of the user's voice the character head reads e.g. "this may be suprizing bes an caon ge o poletyq xs pottasha" for "This may be surprising based on current geopolitics": CER ~36-40% on the German-accented voice against ~19% on LibriSpeech. Even LibriSpeech dev utterances continue off-topic.

## New data (2026-10-18)

- `scripts/tts_questions.py`: Piper `en_US-libritts_r-medium` (904 speakers, CPU, ~100 takes/min on 8 cores). `--sft` reads first user turns of `assistant_full` (`~/training_data/tts_questions`, document = question + answer); `--text` one plain sentence per random `pretrain_v2` document (`~/training_data/tts_text`, document = the document).
- `scripts/tts_clone.py`: Chatterbox (torch 2.8 cu128 in `~/training_data/venv-tts`, Python 3.12 standalone) clones the user's voice from training-session takes onto pretraining sentences (`~/training_data/tts_clone`). ~1.4 h of audio per hour next to a training run.
- Character-head CER of J at step ~3000: real voice 32.5% (held-out session ~40%), Piper questions 10.3% (partly trained), Piper sentences 16.6% (never trained), **cloned voice 26.4%** (never trained): the clone is much harder than Piper, closer to the real voice.
- SLURP real (assistant commands, CC BY-NC) downloading; `scripts/slurp_takes.py` converts it to takes. MLS English: ~20 shards (~600 h) on the HDD, paused (audiobooks again).

## J — encoder only, CTC 1:1, continuation on documents, TTS (from H-8ep, 40 000 steps)

The user's direction: the encoder must hear well without the backbone's help, the text after the audio must continue well, and audio must not be tied to stories. So: `CTC_WEIGHT` 1.0 (was 0.1), `DISTILL_ALPHA` 0.5 (KL towards the same window read), `AUDIO_TRAIN_AFTER_WORDS` 16 but only for takes read out of a document (voice, TTS) — a LibriSpeech utterance scores no book text — and 30% of each batch from the TTS sets (`AUDIO_TTS_SHARE`). Started from H-8ep at step 0 of a fresh schedule (`AUDIO_E2_FROM`). Validation adds `val tts` (1 in 20 TTS takes held out by id).

| step | tts own | tts text after (rotated) | libri own | voice own | voice text after (rotated) | voice CER |
|---|---|---|---|---|---|---|
| 0 (H-8ep) | 7.590 | 4.248 (4.371) | 6.451 | 7.818 | 4.591 (4.691) | 42.8% |
| 2800 | 5.867 | 3.660 (4.000) | 6.537 | 7.702 | 4.546 (4.675) | 40.1% |
| 40 000 (saved as `s4.ae2.J-40k`) | 5.255 | 3.593 (3.982) | 6.413 | 7.180 | 4.312 (4.595) | 24.9% |

Teachers: tts own 4.305, after 2.814; voice own 4.681, after 3.864. At 2800 the real voice had barely moved; by 40 000, with SLURP and the clone in the batches from step 14 600, its CER fell from 42.8% to 24.9% and the text after a held-out take beats rotated audio by 0.28 (from 0.10). LibriSpeech CER 19.4% → 10.4%, TTS 21.0% → 5.3%. The reader is still well ahead (voice text after 4.31 vs 3.86). One OOM at step ~21 000 while Chatterbox shared the GPU: generation and training now run one after the other.

## J2 — J continued to 70 000 steps with backbone-context distillation and more sets (queued overnight)

`STATE_DISTILL_BETA` 1.0: besides the KL towards the reading's predictions, the backbone's per-word context (`[dw, HC]`, what the decoder is given) after the audio is pulled towards the context reading the transcript gives, `β/2 · Σ‖o − o_read‖² / Σ‖o_read‖²` through the backbone's output gradient (`Hierarchical::record_state` / `apply_state`; test `state_distillation_pulls_the_backbone_context_towards_the_teacher`). The 2025-26 frozen-LLM speech papers (SALAD, FreezeEmpath, language-aware distillation) align hidden states under speech vs transcript conditioning the same way. The teacher is the same backbone reading, so the term stays meaningful when the backbone trains too (the reading pass stays gradient-free): nothing here depends on the backbone being frozen. The log line reports the relative distance as `state`.

Take sets in turn per batch (`AUDIO_TAKE_SHARE` 0.625 = 5 of 8): TTS questions, TTS pretraining sentences, the cloned voice (no accent filter: the user wants the variation), SLURP (72k real assistant commands, CER 51% at J step 14 600), VoxPopuli English (downloading overnight). The schedule resumes at step 40 000 of 70 000 (lr ≈ 1.2e-4, cosine to zero).

J2 result (step 70 000, saved as `s4.ae2.J2-70k`; one OOM at ~41 000 from a 229 s VoxPopuli speech, takes now capped at `AUDIO_MAX_TAKE_SECONDS` 25):

| val | J-40k | J2-70k |
|---|---|---|
| voice CER / own / text after (rotated) | 24.9% / 7.180 / 4.312 (4.595) | **23.2%** / **7.082** / 4.327 (4.613) |
| LibriSpeech CER / own / text after | 10.4% / 6.413 / 5.615 | 9.9% / 6.460 / 5.664 |
| TTS CER / own / text after | 5.3% / 5.255 / 3.593 | 4.8% / 5.265 / 3.602 |
| state distance to the reading | — | 0.24 → 0.12 |

The acoustics improved again; the text after the audio did not. The backbone-context distance halved without moving the next-word loss on the text after: matching the context in the mean-squared sense is not what the predictions after the take depend on, at least at β 1. The large remaining gap is on the heard words themselves (voice own 7.08 against the reader's 4.68, TTS 5.27 against 4.36, at TTS CER 4.8%): a word's audio embedding does not stand for the word the way the text encoder's embedding does.

## J3 — J2 plus per-word embedding alignment (`EMBED_ALIGN_WEIGHT` 1.0): worse, stopped at 72 600

Each heard word's audio embedding pulled onto the text encoder's embedding of the same word, `γ/2 · Σ‖a − t‖² / Σ‖t‖²`. The relative distance started at 1.52 (with both at the same fixed norm: cosine ≈ 0.24) and settled at exactly 1.00 (cosine 0.5) within 2000 steps, while every prediction got worse: TTS own 5.27 → 5.90, text after 3.60 → 3.80; voice own 7.08 → 7.51, text after 4.33 → 4.47; LibriSpeech text after 5.66 → 5.73. CER unchanged. The text encoder's embedding is spelling-centred (see the orthographic clustering of e_w); the audio embedding carries what the backbone reads differently, and squeezing it into the text one removes information. A control from J2 with the same restart and no embedding loss stayed flat (TTS own 5.28, text after 3.61, voice own 7.12 at step 70 800, against 5.65 / 3.69 / 7.41 with it), so the damage is the loss, not the learning-rate jump. Reverted: `s4.ae2` is `J2-70k` again, the J3 encoder kept as `s4.ae2.J3-72k-embed`, the weight set to 0.

## J5 — contrastive word identity (`EMBED_CONTRAST_WEIGHT` 0.1), stopped at 95 840

InfoNCE between each heard word and the text encoder's embedding of every word heard in the batch (cosine, τ 0.1, other occurrences of the same word left out). Word match (heard word nearest its own text word) rose TTS 26 → 56%, voice 18 → 44%, LibriSpeech 22 → 53%, while the frozen backbone's losses got slightly worse (TTS own 5.28 → 5.61, after 3.61 → 3.66; voice own 7.11 → 7.30, after 4.31 → 4.35), voice CER 23.3%. Kept as `s4.ae2.J5-last`. Stopped because an unknown share of the user's earlier takes lost their last word to the harness's input buffer (fixed since).

## Word ends: our CTC alignment against MFA and MMS (2026-10-19)

On LibriSpeech val (250 utterances, `ae2a`), J5's own CTC forced alignment ends words **1.7 encoder frames (~70 ms) late on average**: exact 7.7%, ±1 frame 28%, ±2 76%, mean |err| 76 ms. Every take aligned by it (and by online realignment, same formula) therefore pools the onset of the next word into the current one. The boundary head given the word count does better (duration prior: ±1 88%, mean 40 ms), having learned from MFA. torchaudio's `MMS_FA` (wav2vec2 trained for forced alignment, `scripts/mms_align.py`) on 400 dev-clean utterances: mean |err| 37 ms, median 24 ms, within 40 ms 69%, within 80 ms 88%, 14 utterances/s next to a training run. Its ends come **32 ms early** on average (median 21 ms): the last character's CTC emission precedes the word's acoustic end. `--shift 0.03` (never past the next word's start) brings the bias to −3 ms, mean |err| 30 ms, within 40 ms 78%, within 80 ms 94% (0.02: 77.5% / 92%; 0.04: 74% / 95%).

## K — new encoder from scratch, careful takes only, online alignment, packed text model

- Encoder from scratch (`AUDIO_E2_FROM` empty), saved as `models/s4_K.ae2` (`s4_K` is a symlink to `s4`).
- Voice: only the newest 80 takes (`AUDIO_VOICE_LAST`), recorded after the cut-off fix; validation stays the held-out session.
- Takes pre-aligned by J5 (`ae2v`, now incremental: only takes without an alignment unless `AE2V_ALL`), then aligned online: the frame stack and CTC head run first (`AudioEncoder::forward_aligned`), each take the head hears at ≤ 35% CER is forced-aligned on those logits and pooled with those boundaries in the same step (boundary head targets too); otherwise its last such alignment stands. LibriSpeech keeps MFA.
- Contrastive word identity 0.1, KL 0.5, state 1.0, CTC 1.0 as J5.
- The training windows of a batch run as one packed sequence, each closed by `<END>` with a document reset (`set_doc_starts`), so the text model is 3 calls per step instead of 24; 4 batches load ahead. 410 → 207 ms/step.

| step | LibriSpeech CER / word match / own gap / after gap | voice CER / match / own gap / after gap | TTS CER / match / own gap / after gap |
|---|---|---|---|
| 15 400 | 32.2% / 73.4% / +1.02 / +0.34 | 45.4% / 59.3% / +3.27 / +0.66 | 23.2% / 78.2% / +2.26 / +0.94 |

Word match at step 15 400 is already above J5's end. Caveat found afterwards: the online alignment inherits the CTC lateness above.

## L — K with MMS word ends, no online alignment (planned after K)

Every take set aligned by `scripts/mms_align.py --shift 0.03` (J5's alignments kept as `alignment.j5.jsonl`), `AUDIO_ALIGN_ONLINE` false so those ends stand for the whole run. The question: do accurate boundaries let the meaning come through more (word match, own and after gaps) than K's?

## L, M — the user's segment encoder (2026-10-20)

The user reworked the segment encoder: a configurable stack (`AUDIO_SEG_STACK`), a `[W]` step after each word's frames, each word one row of a `[words, steps]` batch padded to the batch's longest. L (`smsm`, 24 frames) and M (`ss`) from scratch on MMS word ends, online alignment off. Fixes on the way: the trainer packs at most `PACK_WORDS` = 256 words per backbone call with the backbone chunk matched (the resident buffers had grown to an OOM), utterances over `AUDIO_MAX_WORDS` = 100 or faster than 6 words/s are dropped (VoxPopuli had transcripts of up to 501 words on ≤ 25 s), and the segment encoder runs at most `AUDIO_SEG_MAX_ROWS` = 20 000 rows (a full batch reads fewer frames per word instead of running out of memory).

At equal steps M (`ss`) heard better than L and K early on; by 17k all three were within a validation's noise. Raising M's frames per word from 24 to 64 at step 29 400 (segments include the pause before a word: 2.5% exceed 24 frames) improved the heard words clearly (voice own 7.86 → 7.33 in 7k steps, TTS 5.75 → 5.36). At step 54 800: LibriSpeech own +0.35 / after +0.34 / CER 15.5%; voice own +2.26 / after +0.55 / CER 32.6%; TTS own +0.66 / after +0.39 / CER 8.0%. The text after the audio moved little after step 40k while the heard words kept improving.

From step 55 200 M trains on all 388 of the user's reviewed takes (33 min, was the newest 80); the held-out session lost takes in the review (57 → 38), so voice numbers from there on are on a new set (teacher 4.745 / 3.693).

## N — M continued with pure distillation and Common Voice (planned overnight 2026-10-20)

- `DISTILL_ALPHA` 1.0: the decoder's target is the reading's distribution alone. With 0.5 half the target was the true next word, which rewards audio that beats the text (genre cues, the next word's onset) instead of audio that equals it; DiVA (arXiv 2410.02678) finds the KL to the text model's output the essential loss, Ultravox matches the text backbone's logits the same way.
- Common Voice 17 English (`scripts/fetch_common_voice.py`, fixie-ai's copy with an LLM-written continuation per sentence, used as the take's document so the 16 words after it are scored): crowd-sourced, home microphones, many accents. 30 shards (~360k takes, ~66 GB as 16 kHz FLAC on the HDD), MMS-aligned.
- Take sets drawn with probability ∝ size^0.5 (`AUDIO_TAKE_SET_POWER`) instead of in turn: the clone was heard ~14 times per run, VoxPopuli not once.
- Starts from M's final encoder (`AUDIO_E2_FROM`), 100k steps. Chain: `models/overnight.sh`, log `models/overnight.log`.
- From step 59 800 (resumed): louder and quieter speech. `AugmentCfg::speech` gain −18…+6 dB (was ±6; the user's takes sit near −25 dBFS and were never heard below −31, while speaking quietly or from further away lost accuracy), and a white microphone floor at −80…−60 dBFS added after the gain, never within 15 dB of the speech (`the_noise_floor_stays_below_quiet_speech`). Samples of the edges: `examples/augment_extremes.rs`, `~/training_data/augment_demo/extremes/`.

### N result (100k steps from M)

| CER | LibriSpeech | voice | TTS q | TTS text | clone | SLURP | VoxPopuli | Common Voice |
|---|---|---|---|---|---|---|---|---|
| M | 11.5 | 27.1 | 5.1 | 7.4 | 10.7 | 30.5 | 20.1 | 29.2 |
| N | **8.7** | **20.8** | **4.6** | **6.6** | **9.6** | **27.0** | **15.9** | **21.0** |

Text after the audio (gap to the teacher) got worse on every document set but Common Voice (voice +0.48 → +0.54, TTS questions +0.26 → +0.42); SLURP's heard words too (+0.52 → +0.70) while its CER fell. Pure distillation costs the continuation here; DiVA's finding was with a pretrained Whisper encoder. Per-set lines come from `ae2e` (one per `AUDIO_TAKE_DIRS` set, 256 held-out takes).

## O — N with α per window: 0.5 with a document, 1.0 without (`DISTILL_ALPHA_PLAIN`)

The two kinds of window are packed separately, each distilled at its own α. Take sets are drawn in runs of 32 consecutive takes (`TAKE_BLOCK`): the large sets live on the HDD, where a random take costs a ~130 ms seek; load went from 30-38% to 0% of a step (~470 → ~270 ms/step). Restarting from N at the full 3e-4 cost 3-4 points of CER first; by step 34 800 the text after already beat both parents on the user's voice (4.15 vs N 4.23, M 4.17), TTS 3.15 (M 3.10, N 3.27). Continuations now use `AUDIO_LR` 1e-4.

## Q (queued after O): O + Common Voice shards 30-59, peak lr 1e-4

`models/after_o.sh`: keeps O's final encoder, aligns only the new takes (`mms_align.py --only-new`), then trains `s4_Q` from O under `models/run_until_done.sh` (after a crash, waits for a free GPU and resumes; per-set evaluation at the end).

## R (queued after Q): Q + pooling at the boundary head's own cut

`AUDIO_HEAD_CUT_P` 0.5: per training utterance, the words are pooled at the boundary head's count-constrained cut (`CountAligner::align_words`, as the evaluation's "boundary head" line) instead of the reference ends; the head's BCE targets stay the reference. The step log shows the share as `head cut`. From Q's final encoder at peak lr 1e-4, supervisor `models/run_until_done2.sh` (gives up after three crashes without a new checkpoint). `ae2e` now also lists the user's most misheard words (`misheard_report`), the input for error-driven recording prompts. Analysis of where the encoder loses: `docs/encoder-bottlenecks.md`.

### R result (100k steps from Q)

| | own / after / CER | Q | R |
|---|---|---|---|
| voice | | +1.28 / +0.35 / 17.7 | **+1.16 / +0.31 / 16.2** |
| TTS questions | | +0.39 / +0.23 / 3.6 | **+0.35 / +0.21 / 3.4** |
| LibriSpeech | | +0.21 / +0.38 / 6.6 | **+0.19 / +0.38 / 6.4** |
| Common Voice | | +0.64 / +0.54 / 17.5 | **+0.61 / +0.50 / 16.6** |

Under its own boundary head's cut (LibriSpeech val) the heard words lose +0.23 against MFA's ends (Q +0.35), word match 88.5% (Q 87.3%), and the text after is as good as under the reference cut (+0.37). Misheard words of the user: 36.6% (Q 40.7%), th-words on top.

## S: R + HeySQuAD's human-read questions

`scripts/heysquad_takes.py`: 76k SQuAD questions read by crowd workers on their own microphones, the `question` as text (the shipped `transcription` is ASR output), digits dropped, ~64k takes. A seventh take set; nothing else changes. `models/after_r.sh` aligns them with MMS, scores R on the new set as the baseline (`models/s4_R.sets_eval2.log`), then trains `s4_S` from R's final encoder at peak lr 1e-4.

R on HeySQuAD before S (`models/s4_R.sets_eval2.log`): own gap +0.87, CER 25.3%, word match 84.3%, boundary F1 0.84: the hardest set but the user's voice. S's first start crashed in the loader: ~19k of the WAVs had a data chunk ending mid-sample (some as shipped, the stereo ones because ffmpeg wrote them to a pipe and could not fill in the header); all re-encoded, restarted from step 0 at 23:50.

## T (queued after S): S continued

Nothing changes but the starting point (`models/after_s.sh`): S sees Common Voice 0.3 times over its run, so the encoder is limited by steps, not data. Before T starts, `scripts/error_prompts.py` re-ranks the voice prompt pool by S's full misheard-word table (`<encoder>.misheard.tsv`, now written by `ae2e`); the pool as it was is `data/voice/prompts.random.jsonl`.
