# Plan: paper-style context injection — sequence concat, not feature concat

> **Executed 2026-07-05, with one deviation:** no `EmbedInject` layer (tag 18)
> was added. The decoder instead reuses the **encoder's char embedding** (tied
> table, CHAR_HIDDEN == OUT_HIDDEN): `Hierarchical` builds the decoder inputs
> itself — slot 0 gets the context vector, char slots get embedding rows — and
> reduces the decoder-side embedding grads back into the encoder's table. The
> decoder stack starts directly at the sLSTM blocks (no vocab-wide input, no
> new tag, old checkpoints stay inspectable via `Hierarchical::load_stacks`).
> Verified by FD gradient tests in `hierarchical.rs` plus `parallel_parity`.

Implement the HAT paper's decoder coupling (arXiv 2501.10322, eq. 3–4): the
backbone output is injected as the **first sequence step** of the decoder,
`(p_i, x_1, …, x_ℓ)`, instead of feature-concatenated onto every char step.
Chars are embedded at decoder width; there is no per-step context concat and no
combining linear. Weight-decay experiments are closed: batch4 was resumed with
a decaying embedding and nothing changed — decay is ruled out as a lever, drop
that thread entirely.

## Target architecture (decoder only — encoder & backbone unchanged)

Current decoder:

```
EmbedConcat(vocab→OUT_HIDDEN ‖ ctx) → Linear(2·OUT_HIDDEN→OUT_HIDDEN)
→ sLSTMBlock(OUT_HIDDEN) ×2 → RMSNorm → LinearNoBias(vocab) → SoftCap(30)
```

New decoder:

```
EmbedInject(vocab→OUT_HIDDEN | inject ctx)
→ sLSTMBlock(OUT_HIDDEN) ×2 → RMSNorm → LinearNoBias(vocab) → SoftCap(30)
```

Per word with ℓ chars (decoder state reset per word, as now):

| slot | input | target |
|---|---|---|
| 0 | **context vector** (backbone output, replaces the `[W]` BOS — paper: `p_i` takes the BOS slot) | char 1 |
| j = 1…ℓ−1 | embedded char j | char j+1 |
| ℓ | embedded char ℓ | `[W]` (EOS, unchanged) |

Slot count per word stays ℓ+1 → `dec_ranges` / `dec_targets` logic in
`hierarchical.rs` is unchanged; only the *inputs* change.

The backbone already ends in `Linear(OUT_HIDDEN)`, which is the paper's `W_D`
downscale — context arrives at decoder width, no extra projection needed.

## New layer: `EmbedInjectLayer` (tag 18, `src/nn/embed_inject.rs`)

Keeps the decoder's outer interface identical to today so the buffer and cache
machinery in `hierarchical.rs` keeps working:

- `input_size = vocab + ctx`, `output_size = embed_dim` — assert
  `embed_dim == ctx` in the builder (both are OUT_HIDDEN).
- **forward**: if the one-hot part `input[..vocab]` has a set entry → output =
  `weights[tok]` (embedding lookup, ignore tail). If the one-hot is all zero →
  output = `input[vocab..]` (identity pass of the context). Cache must record
  which mode ran (store the token index as `Option<usize>` or `usize::MAX`).
- **backward**: token step → `grads.row(tok) += delta`, `dx` all zero. Inject
  step → `dx[..vocab] = 0`, `dx[vocab..] = delta` (identity). This preserves
  the existing context-grad accumulation `d_o_buf[i] += dx[vocab + i]` in
  `hierarchical.rs` (~line 460): char slots now contribute exactly zero, slot 0
  carries the whole word-context gradient.
- Weights: `vocab × embed_dim` table on the no-decay Adam grad type (same as
  `EmbeddingLayer`), init scale `sqrt(6/(vocab + embed_dim))`.
- Save = `write_matrix(weights)`; load derives `passthrough = input − vocab`
  (mirror `load_embed_concat` in `src/loading.rs`).
- Must implement `add_grads_from` + `copy_weights_from` (decoder is
  parallel-trained via `ReplicaPool`) — mirror `EmbedConcatLayer`.

## Implementation steps

1. `src/nn/embed_inject.rs` — layer + cache + a unit test: finite-difference
   grad check for a token step, exact identity check (fwd and bwd) for an
   inject step.
2. Register: `src/nn/mod.rs`; builder method `embed_inject(vocab, embed_dim)`
   in `src/nn_layer.rs` (assert `self.output_size − vocab == embed_dim`);
   `18 => load_embed_inject(...)` in `src/loading.rs`; describe-arm in
   `src/inspect.rs`; a `row(...)` arm in `examples/weight_stats.rs`.
3. `src/model.rs`: swap the decoder front — `embed_concat + linear` →
   `embed_inject`, blocks unchanged.
4. `src/hierarchical.rs` — training input construction (`forward_over`, decode
   phase, ~line 340): slot 0 gets zero one-hot + context in `char2_input[vocab..]`;
   slots ≥ 1 get the char one-hot + **zeroed** context tail. Today every slot
   gets `[W]`-or-char one-hot + context; the leading `[W]` feed disappears.
5. `src/hierarchical.rs` — backward: no change expected; verify slot 0 is the
   only slot writing into `d_o_buf` (assert once during bring-up if unsure).
6. `src/hierarchical.rs` — sampling (`encode_word_advance` + the word-recursion
   decode loop): first decoder step feeds the injected context, then feed the
   sampled char per step (context tail zeroed) until `[W]` is sampled.
7. `CLAUDE.md`: update the decoder line and remove the EmbedConcat/combining-
   linear description.

## Verification

- `cargo test --lib` — new layer tests + existing SoftCap grad check.
- `cargo test --test parallel_parity` (~280 s) — covers the new layer's replica
  save/load round-trip and grad reduction end-to-end.
- Fresh run (new tag → old checkpoints load for inspection but cannot resume);
  compare against fable10 on the same log columns: `loss`, `word_loss`,
  `delta_word`, `delta_char1`, `dew_d1..dew_d128`.

## What this experiment actually tests / risk

The paper's decoder is a transformer: attention re-reads the injected first
token at every char position for free. An sLSTM decoder must *carry* the
context in its recurrent state across the whole word. Failure mode to watch:
word_loss degrading specifically on long words (state washes the context out)
while short words match fable10. If that happens, the per-step feature concat
(current design) is the better RNN translation and this becomes a negative
result — still worth having, since it isolates how much of the decoder's
capacity is spent re-reading context.

Side effects to expect in the metrics: `delta_word` changes meaning slightly —
the context gradient now arrives once per word (slot 0) instead of summed over
ℓ+1 slots, so its absolute level will drop; compare trends, not levels, against
fable10.
