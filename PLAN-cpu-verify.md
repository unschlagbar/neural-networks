# Verifying the `forward_backward_window` host-side optimization

Run this on the laptop. Every command is read-only with respect to the working
tree — the A/B baseline comes from a separate git worktree, so nothing touches
the working copy or any stash.

## What changed

Per-window host work in `gpu::hierarchical::forward_backward_window` that was
either redundant or didn't need doing at all:

| # | Before | After |
|---|--------|-------|
| 1 | 4 `std::env::var` calls per window (`GPU_PROF`, `GPU_MEM`, `GPU_NO_GROUP` twice via `group_by_len`) — each takes a process-wide lock and allocates a `String` | Read once into `Hierarchical::flags` at construction |
| 2 | `enc_group_rows` ran **twice** per window — once in the encoder forward, again in the encoder backward's re-forward, rebuilding byte-identical rectangles | Built once into `Scratch::enc_layout`, reused by the backward |
| 3 | Ids built as `Vec<usize>`, then `upload_ids` re-allocated and re-mapped to `Vec<u32>` — a second alloc + second pass on each of ~8 uploads per group | Built directly as `Vec<u32>`; new `*_u32` ops entry points upload them as-is |
| 4 | CE mask went `Vec<bool>` → `Vec<usize>` → `Vec<u32>` | One `Vec<u32>`, reused |
| 5 | `group_by_len` allocated a fresh `BTreeMap` + a fresh `Vec` per bucket, twice per window | Writes into reused buffers on the model |

All the per-window index buffers now live in a `Scratch` struct owned by the
model, so a training run allocates them once instead of once per window.

**Group ordering was deliberately kept bit-identical** to the old
`BTreeMap::into_values()` order. Group order decides the order the per-group
losses are summed and the grads are scattered, so preserving it means the loss
comparisons below should match to the same tolerances as before, not merely
"close".

## What was already verified (on a machine with no GPU)

- Compiles clean with and without `--features cuda`, across lib / tests /
  examples / benches. No new clippy warnings.
- The two pieces of rewritten host logic were extracted into standalone
  programs and diff-tested against the original implementations:
  - `group_by_len` — 4000 randomized cases, including buffer reuse across
    calls and the empty-input case. Exact match.
  - decoder index construction (`o_rows` / `char_rows` / `char_ids` /
    `targets` / `mask`) — 3000 randomized cases with buffer reuse. Exact match.

That covers the index math. It does **not** cover the device side — that is
what Step 1 is for.

## Step 0 — set the corpus path

```fish
set -x CORPUS ../../training_data/000_00000.parquet   # or any parquet/text corpus
```

## Step 1 — correctness: the parity tests (fast; do this first)

`grouping_matches_single_rectangle` is the important one — it exercises both
`GPU_NO_GROUP` branches of the rewritten `group_by_len` **and** compares
grouped vs single-rectangle through a full optimizer step (identical weights
afterwards means the reduced grads agreed, not just the loss).

```fish
cargo test --release --features cuda gpu::hierarchical:: -- --nocapture --test-threads=1
```

`--test-threads=1` matters: that test mutates the `GPU_NO_GROUP` env var, and
the flag is now read at **construction** time, so concurrent tests could
otherwise race on it. (The test sets the var before `new`/`load` in both
branches, so it is correct — but only single-threaded.)

Then the rest of the GPU suite, to catch anything the `ops.rs` signature
changes disturbed:

```fish
cargo test --release --features cuda gpu:: -- --test-threads=1
```

> Expect `mlstm_fused_matches_legacy` to fail ~50% of runs — the known flake
> documented in CLAUDE.md (hypersensitive post-Adam-step weight comparison),
> unrelated to this change. Re-run it alone to confirm it's the flake and not a
> real break.

## Step 2 — build the baseline in a worktree

This is what makes the speed number trustworthy: the baseline is the *old*
code, compiled the same way. The worktree lands beside `iron_oxide` so the
`../iron_oxide` path dep resolves.

```fish
git worktree add ../nn-baseline HEAD
cd ../nn-baseline; and cargo build --release --features cuda --example gpu_soak
cd ../neural-networks; and cargo build --release --features cuda --example gpu_soak
```

## Step 3 — speed A/B

`gpu_soak` is the right vehicle: real streamed windows, many of them, no
checkpoint writing, and it reports `s/window`.

Run each three times and take the **best**, not the mean — a laptop thermally
throttles, and the first run pays page-cache cost for the corpus.

```fish
for i in 1 2 3
    ../nn-baseline/target/release/examples/gpu_soak $CORPUS 300
end
for i in 1 2 3
    ./target/release/examples/gpu_soak $CORPUS 300
end
```

300 windows averages over window-shape variation while staying a couple of
minutes.

> **Do not compare the mean loss between the two runs.** `Tensor::random` is
> unseeded, so the two runs start from different weights and their losses are
> not comparable. Loss parity is Step 1's job, via the save/load seeding the
> tests already do.

## Step 4 — confirm it's actually the CPU that got freed

This is the original question — whether the laptop is GPU-bound or CPU-bound.
Compare **user CPU seconds**, not wall clock:

```fish
command time -v ./target/release/examples/gpu_soak $CORPUS 300 2>&1 | grep -E 'User time|System time|Elapsed'
```

Run it against `../nn-baseline/target/release/examples/gpu_soak` too, and read
the result:

- **User time drops, wall clock barely moves** → GPU-bound. The change freed a
  core but the GPU is still the wall. Still a win on a laptop (less heat, less
  throttling), just not a throughput one.
- **Both drop** → genuinely CPU-bound; a real speedup.
- **User time ≈ wall time in both runs** → one core is saturated and the host
  is the bottleneck. The remaining cost is then kernel-launch overhead, not
  allocation, and the next move is batching the backbone over windows (see
  the `waves` ≈ 0.15 note in CLAUDE.md) rather than more host micro-optimization.

Per-phase split, if you want to see which phase dominates:

```fish
GPU_PROF=1 ./target/release/examples/gpu_soak $CORPUS 5
```

## Step 5 — clean up

```fish
git worktree remove ../nn-baseline
```
