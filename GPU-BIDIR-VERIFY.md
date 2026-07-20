# GPU bidirectional encoder — verification notes

Implemented the bidirectional encoder on the GPU (`src/gpu/hierarchical.rs`),
mirroring the CPU model. This machine has **no CUDA runtime**, so none of the
GPU tests could actually execute here — they compile but early-return / panic at
the cudarc library load. **Run the steps below on the laptop with the GPU.**

## What changed (GPU)

- `gpu::WordEncoder` now holds two block stacks `fwd` + `bwd` and a `combine`
  Linear (`2·hc → hc`), instead of one stack.
  - `fwd` reads `c1 … cn [W]` (readout at the `[W]` step), embeds through the
    tied `table`.
  - `bwd` reads the reversed `[W] cn … c1` (readout at the last real step, `c1`),
    embeds through its **own** new table `bwd_table` (matches CPU: only fwd is
    tied).
  - `e_w = combine([fwd_ro ; bwd_ro])`.
- New device kernels `concat_cols` / `split_cols` (`src/gpu/kernels.rs`,
  registered in `NAMES`) + op wrappers in `src/gpu/ops.rs`.
- New model fields: `bwd_table` + its grad/moments (`d_bwd_table`, `m_bwd_tbl`,
  `v_bwd_tbl`). Optimizer `step` updates `bwd_table` (undecayed, like the tied
  table), both encoder stacks, and `combine` (decayed).
- Encoder backward re-forwards **both** stacks + combine (activation
  checkpointing per length group), splits `d_e_w` through combine, seeds each
  stack at its readout row, backprops, scatter-adds grads to `table` /
  `bwd_table` respectively.
- Checkpoint `save`/`load` now export/import the real `encoder_bwd` + `combine`
  sections (no longer the synthesized copy). `to_sequentials` returns
  `(encoder_fwd, encoder_bwd, combine, word_model, char2_model)`.

## Run these on the laptop (with `--features cuda`)

```bash
# 1. The GPU hierarchical tests — these NOW exercise the bidirectional path
#    (forward_backward runs both stacks; the checkpoint test round-trips
#    bwd_table + combine through the NNM1 container).
cargo test --features cuda gpu::hierarchical:: -- --nocapture

#    Expect: hierarchical_memorizes_and_checkpoints  ... ok
#            grouping_matches_single_rectangle        ... ok
#    (grouping_matches_single_rectangle is the key correctness check: grouped
#     vs single-rectangle must give identical loss AND identical post-step
#     weights — it now covers the bidirectional encoder's grads too.)

# 2. Full GPU test suite (kernels, blocks, ops, parity vs nn2 for the shared
#    pieces). concat_cols/split_cols have no dedicated parity test yet — see
#    "Gaps" below; grouping_matches_single_rectangle exercises them indirectly.
cargo test --features cuda gpu::

# 3. Occupancy sanity after adding kernels (optional, just confirms nothing
#    spilled): 
cargo run --release --features cuda --example gpu_occupancy
```

## If a test fails — likely suspects, in order

1. **`grouping_matches_single_rectangle` diverges** → the bidirectional grad
   path. Most likely a `bwd` readout-row or reversed-id indexing bug in
   `enc_group_rows(.., reversed=true)`, or `split_cols` column order swapped
   (should be `da` = first C cols = fwd, `db` = last C = bwd, matching
   `concat_cols`).
2. **Checkpoint round-trip loss changes** → `to_sequentials` / `load` mismatch
   for `encoder_bwd` or `combine` (e.g. bwd_table not swapped in, or combine
   weights not uploaded). Check `model.encoder.bwd`, `model.bwd_table`,
   `model.encoder.combine` are all assigned in `load`.
3. **Loss doesn't fall (`memorizes` test)** → combine not in the optimizer
   `step`, or `d_bwd_table` not zeroed after the step.

## Gaps / follow-ups (decide after the above passes)

- **No CPU↔GPU parity test for the bidirectional encoder.** The `nn2` twin
  (`src/nn2/hierarchical.rs`) is still **unidirectional**, so the GPU encoder no
  longer has an exact CPU reference of the same shape. The doc comment at the
  top of `gpu/hierarchical.rs` still says "GPU counterpart of nn2::Hierarchical"
  — that's now only true for the backbone/decoder, not the encoder. Options:
  (a) port nn2 to bidirectional too so the parity test covers it, or (b) add a
  standalone FD test for `concat_cols`/`split_cols` + the combine split. Left
  undone pending your call.
- `concat_cols` / `split_cols` have no direct unit test (only exercised via
  `grouping_matches_single_rectangle`). A tiny `x/roundtrip` test would be cheap
  if you want it.

## Report back

Paste the output of step 1 (and step 2 if anything else fails) and I'll fix from
there.
