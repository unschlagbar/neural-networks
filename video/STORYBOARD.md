# Storyboard: a chatbot from scratch

Target: ~12 min, English voiceover. Audience: programmers who know roughly what a
neural network is. No LSTM theory: the xLSTM block is shown as data flow only.

Columns per beat: **say** (what the narration covers), **show** (what is on screen),
**data** (where real numbers come from). `scene:` names a Motion Canvas scene in
`src/scenes/`; `rec:` is a screen recording; `still:` is an image.

Throughline: one line of real Rust code (`data/snippet.rs`) and one real chat turn
travel through the whole video, so every stage is shown on the same input.

## 0. Cold open (0:00 – 0:45)

| time | say | show | data |
|---|---|---|---|
| 0:00 | none, just the demo | rec: `hqs` session: "turn on the lamp" → `<tool>lamp.set(on=true)</tool>` → result → answer | real session |
| 0:20 | "No PyTorch, no pretrained weights. Every layer, every GPU kernel, the tokenizer, the data pipeline, written from scratch in Rust." | fast montage: `src/gpu/*.cu`, `cargo build`, loss curve | `logs/*.csv` |
| 0:40 | "This is how it works." | title card | |

## 1. The problem (0:45 – 2:00)

| time | say | show | data |
|---|---|---|---|
| 0:45 | A language model predicts the next piece of text, over and over | scene: `next_token`: a sentence grows one piece at a time, probability bars above the next slot | sampled logits dump |
| 1:20 | Characters make long sequences, subword tokens are an arbitrary vocab. Ours: bytes in, but the model thinks in words | same text as a long row of chars, then regrouped into words; three-stage diagram teased, greyed out | |

## 2. Bytes and words (2:00 – 3:30)

| time | say | show | data |
|---|---|---|---|
| 2:00 | Any text is UTF-8 bytes: 256 ids plus a few special tokens, nothing is out of vocabulary | scene: `words` part 1–2 (cells + byte values) | `viz_words` |
| 2:40 | A segmenter groups bytes into words: identifiers whole, operators whole, whitespace attached to the word before. ~3–4 bytes per word, so the big model runs ~4x fewer steps | scene: `words` part 3 (the split + counter) | `viz_words` |

## 3. The hierarchical architecture (3:30 – 6:00)

| time | say | show | data |
|---|---|---|---|
| 3:30 | Three stages in one loop: the encoder turns a word into a vector, the backbone folds it into its memory, the decoder spells out the next word, and that word goes back in. Encoder and decoder start fresh for every word; only the backbone keeps memory (HAT, arXiv 2501.10322) | scene: `hier_overview` (markers `encoder`, `backbone`, `decoder`, `loop`, `repeat`, `zoom_encoder`): one stack, first word slow, then three more rounds while the text grows at the top; ends zooming into the encoder | `viz_words` |
| 4:00 | Encoder: each byte looks up a vector, three blocks (sLSTM, mLSTM, sLSTM) run over the letters with memory carried along, and the output at `[W]` is the word vector. Every word becomes one vector; that sequence is what the backbone reads | scene: `encoder` (markers `lookup`, `letters`, `end`, `word_vector`, `all_words`): unrolled grid over `println` + `[W]`, then a row of words collapsing into vectors | `viz_words` |
| 4:25 | Words don't depend on each other, so all of a window's words are encoded at once: grouped by length into dense batches, no padding | scene: `parallel_encode` (markers `pad`, `groups`, `run_groups`, `scatter`): padded batch with wasted cells flashing red, rows regroup by length, groups sweep, vectors return to window order | `viz_words`, `src/gpu/word_groups.rs` |
| 4:35 | Backbone: the big model, one step per word, carries memory across the whole text | words become dots moving along a long lane, a state vector travelling with them | |
| 5:10 | Decoder: gets the backbone's context as its first input, spells out the *next* word char by char, ends with `[W]` | chars appear one by one with probability bars | decoder logits dump |
| 5:40 | Sizes: backbone 1024 wide × 28 blocks; encoder/decoder 256 wide. Compute goes where the meaning is | boxes rescale to their real width/depth | `config.rs` |

## 4. The xLSTM block (6:00 – 7:30)

All three stages are stacks of the same block; only width, depth and cell type differ.
One animation, no equations: scene `xlstm_block`. Order as in `src/gpu/block.rs`; the
backbone has an sLSTM every 4th block and mLSTMs in between (`src/model.rs`).
Each beat starts at a `waitUntil` marker of the same name.

| time | marker | say | show |
|---|---|---|---|
| 6:00 | (start) | The backbone is 28 blocks | row of 28 blocks, `s`/`m` labelled |
| 6:10 | `zoom` | Look inside one | camera dives into block 6, the diagram draws in |
| 6:20 | `input` | A word's vector enters and gets normalized | column of coloured cells slides in, RMSNorm evens it out |
| 6:30 | `memory` | The mLSTM keeps a memory matrix. It fades a little, the new word is written in, the output is read back out, and the memory stays for the next word | grid stays inside the cell; key row and value column appear on its edges, grid dims, new pattern painted in row by row; caption names each step |
| 7:05 | `residual` | The input skips around the cell and is added back | bypass wire lights up, the copy flies over, (+) merges |
| 7:12 | `ffn` | A feed-forward part handles each word on its own: two linear layers widen it, one goes through SiLU, they multiply, a third linear narrows it back | vector forks into linear (gate) and linear (value), gate passes SiLU, both meet at ⊙, linear (down), second residual |
| 7:25 | `stack` | Stack 28 of these | zoom out, the vector runs through all 28 blocks |

## 5. Inside the trained model (7:30 – 8:30)

| time | say | show | data |
|---|---|---|---|
| 7:30 | The encoder's word vectors cluster by spelling: pumpkin sits next to puddle | scene: `embeddings`: 2D projection of word vectors, a few words labelled | embedding dump + PCA |
| 8:00 | The memory split itself: some heads keep things for the whole text, others only look a few words back | scene: `memory_span`: per-head bars of how long memory survives, sorting into two groups | forget-gate dump |

## 6. Training (8:30 – 10:15)

| time | say | show | data |
|---|---|---|---|
| 8:30 | Pretraining on one 16 GB GPU: packed documents, predict every next char, memory carried between 4096-word windows | windows sliding over a long stream, `<END>` markers resetting memory | |
| 9:10 | Loss curve and what samples look like along the way | loss plot + three sample snippets from different checkpoints | `logs/*.csv`, `hs` samples |
| 9:45 | Data matters: filters and dedup (rust-lib was 98% duplicates) | bar of sources shrinking through filters | `datamix check` report |

## 7. Making it a chatbot (10:15 – 12:00)

| time | say | show | data |
|---|---|---|---|
| 10:15 | SFT: the same model, trained on conversations; loss only on the assistant's words | scene: `sft_mask`: a chat in wire form, `<SEP>`/`<END>` chips, loss-carrying span highlighted | real record from `data/sources/tools.jsonl` |
| 11:00 | Tools: `<tool>` is one token, the harness answers the call | the cold-open lamp call replayed as tokens | |
| 11:30 | Back to the demo; what's next; repo link | rec: another `hqs` session, outro card | |

## Production

1. **Script first.** Write the narration per beat, read it aloud, then fix the times above.
2. **Record the voiceover** before animating. Motion Canvas takes an audio track
   (`makeProject({audio: ...})`) and scenes wait on named events
   (`yield* waitUntil('zoom')`), which show up as draggable markers on the timeline,
   so animation is lined up to the voice by dragging, not by editing durations.
3. **One scene per section** above; stills and recordings are cut in a video editor
   (Kdenlive) together with the rendered scenes.
4. **Data dumps** are `examples/viz_*.rs`, each writing JSON into `src/data/`, so a
   scene never contains hand-typed numbers. Shared shapes and colours live in
   `src/lib/viz.ts`.

## Scene backlog

| scene | status | needs dump |
|---|---|---|
| `words` | done (needs `waitUntil` markers) | `viz_words` ✓ |
| `encoder` | first draft | `viz_words` ✓ (vector values illustrative) |
| `parallel_encode` | first draft | `viz_words` ✓ |
| `xlstm_block` | first draft | none (illustrative values) |
| `next_token` | todo | sampled logits |
| `hier_overview` | first draft | `viz_words` ✓ |
| `embeddings` | todo | word vectors + projection |
| `memory_span` | todo | forget-gate values |
| `sft_mask` | todo | one SFT record + mask |
