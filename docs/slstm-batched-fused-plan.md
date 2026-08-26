# Implementation spec: batched time-fused sLSTM kernel (`slstm_fused_time_batched`)

Written 2026-08-25. Goal: a second time-fused sLSTM kernel that exploits large batch, so
encoder/decoder stop using the per-step path. Dispatch on **B**, not T.

## Status (2026-08-26): built, landed, measured

Shipped as `slstm_batched_fwd` / `slstm_batched_bwd` in `src/gpu/cuda/slstm_batched.cu`,
with the shared `mma.sync` primitives factored out of `mlstm_fused.cu` into `mma.cu`.
Measured over the twenty real encoder groups (`examples/encdec_baseline.rs`, which now
interleaves the arms in one process and reports what the dispatch actually picks):

| | fwd | fwd+bwd | encoder+decoder |
|---|---|---|---|
| per-step | 1.66 ms | 3.85-4.03 ms | 15.4-16.1 ms |
| dispatched | 1.39 ms | 3.36-3.48 ms | 13.4-13.9 ms |

**~13% on the encoder/decoder sLSTM, ~1.03-1.67x on the groups that take it.** Below the
"honestly" line, and for a reason the plan did not anticipate — see the next section.

### What the plan got wrong

**The contraction was never the cost.** The plan's whole argument is scalar-vs-tensor-core
on `h·Wh`, and that part is real (the scalar kernel measures 3x SLOWER than per-step at
B=120/T=64). But once the product is on the mma unit it disappears into the noise: at
B=120/T=32 the forward moves 45 MB of saved activations and `g` against 63 MFLOP per
timestep of contraction. `ncu` puts the kernel at 422 GB/s DRAM and 25-33% of the warp
slots — **memory- and latency-bound, with the tensor cores idle most of the time**. The
per-timestep cost is the slabs (`h_prev`, `z`, `o`, `i'`, `f'`, `c`, `n`, `out`) plus the
four `g` loads, and the per-step path pays exactly the same bytes. Fusing time removes
launches, not work.

**A new cost the plan does not mention: operand re-reads.** A block owns `SB_NJ` units, so
the grid needs `cb = H / SB_NJ` column blocks, and EVERY one of them reads the whole `h`
row (forward) or `dg` row (backward) for its batch rows, every timestep. That is `cb`x the
traffic cuBLAS pays. It is proportional to B, while the launches it saves are not — which
is what sets the ceiling in `ops::slstm_batched_pays` and why the widest encoder groups
(B >= 292) keep the per-step path. The backward is 4x worse than the forward here, because
its reduction axis is `4H` rather than `H`.

`SB_NJ` is therefore the one parameter that matters, and it is a re-read/occupancy trade,
not a tiling one: 16 / 32 / 64 measure 4.38 / 4.00 / 4.08 ms per layer per window. 64
gives it back — the block is then 1024 threads, whose 64-register budget spills the
backward's `Wh` fragments (`ptxas -v`, exactly as warned below).

### The grid must fit ONE wave of SMs (2026-08-26, from ncu on a real training run)

ncu's "only 32 of 84 multiprocessors used, Est. Speedup 61.9%" and its "one or more L1
slices have much lower active cycles" are the same fact: `grid = (H / SB_NJ) * ceil(B /
16)`, and at the batch a real corpus produces (B ~ 64, `SB_NJ` = 32) that is 8 x 4 = 32
blocks. The fix is `SB_NJ`, and the rule is not "more blocks is better" — measured at
H=256, T=16, `rpt` = 1 (us, forward):

| grid blocks | 32 | 48 | 64 | 80 | 96 | 128 | 160 |
|---|---|---|---|---|---|---|---|
| `SB_NJ`=16 | 54.0 | 53.9 | 54.6 | 54.7 | 62.3 | 63.4 | 64.1 |
| `SB_NJ`=32 | 58.1 | 58.2 | 58.6 | 58.3 | 58.1 | 58.8 | 60.1 |

**Flat while the grid fits the 84 SMs, then a step.** A cooperative grid must be fully
resident, so crossing `sm_count` does not queue a second wave — it doubles up on a few
SMs while the rest hold one block, and every `grid.sync()` waits on the doubled ones. At
B=186 the `SB_NJ`=32 grid is 96 blocks and costs 77.3 us against 60.1 at 80 blocks, for
4% more work.

Inside the plateau the NARROWER block wins by ~7%: twice the blocks for the same work
hides latency better and the extra operand re-reads stay in L2. So `sb_nj_for` takes the
narrowest `SB_NJ` whose grid still fits one wave, else the widest. Only 16 and 32 are
ever worth it — 8 loses even where its grid fits (67.6 vs 54.0 us at B=32) and 64 needs a
1024-thread block whose 64-register budget spills the backward's fragments. The backward
measures the same crossover at the same batch.

`SB_RPT` is NOT a second handle on the grid: the M-tiles inside a block run in sequence
with a barrier between them, so halving the grid doubles the serial chain. 1 / 2 / 4
measure 54.6 / 85.4 / 158.9 us at B=64. The search takes `rpt` = 1 and lets the host loop
row chunks rather than widening the block to save a launch.

**The batch a real corpus produces is not the one `_grp_probe` reports on Rust source.**
The default group table here (B = 120..1024) came from `src/gpu/ops.rs`; a natural-language
corpus splits into much narrower buckets. At B=64 across the same lengths the batched path
is **1.40-1.57x on every group** and nothing declines — 2.19 -> 1.42 ms fwd+bwd per layer
per window, against the 13% the Rust-source table showed. `GROUPS=T:B:pieces,...` feeds
`encdec_baseline` a real window's buckets.

### Where the forward's time actually goes

Delete-one-thing probes at B=120/T=16 (58.5 us total): `grid.sync()` 9.4 us (16%, 0.63 us
per sync, irreducible — it IS the recurrence), the mma loop 10.6 us (18%, ~74% of the
achievable rate for a 64-block grid), the `h` tile load 9.2 us. Stall breakdown per
issue-active cycle: barrier 4.36, long scoreboard 4.09, wait 1.92, short scoreboard 1.55.
DRAM sits at 39% of peak, so it is **barrier- and latency-bound, not bandwidth-bound** —
and with fewer blocks than SMs, a block at a barrier is an idle SM, which is why the grid
shape is worth more than anything inside the block.

### Things that cost real time to find

* **`cargo build --release` does not rebuild examples.** Two rounds of "the change did
  nothing" were the profiler reading a stale binary. `--examples`.
* **A `Wh` fragment walks a COLUMN**, whose global stride is `4H` floats. Loading fragments
  straight from global was 27 us of the forward's ~40; staging a slice through shared
  memory, where the read coalesces and the transpose is free, removed it.
* **The staging area must not size the block's shared allocation.** It is dead the moment
  the fragments are built, but sized in one piece at `SB_NJ = 32` it was 64 KB and cost
  one block per SM for the whole T-loop. It runs in passes instead.
* **A scratch buffer keyed on exact dims reallocates twenty times per window.** The
  forward's `h` mirror first shared `FUSED_ALT`, whose `zeros(...)` fired on every group.
  Grow-only and uninitialised: neither kernel reads anything it did not write.
* **The measurement discipline below is not optional.** Wall clock at ROUNDS=4 put the
  same arm anywhere from 3.5 to 5.3 ms; the per-group ratios in an interleaved run were
  the only stable signal, and the totals only settled at ROUNDS=12.

## The key structural fact

**sLSTM batch rows are completely independent.** Each row carries its own `h/c/n/m`; the
only coupling is along `H` (every unit feeds every gate of the SAME row). Therefore:

- the batch can be **chunked freely** into whatever size fits the thread/register budget
- chunks need **no** communication, no grid.sync between them, no state exchange
- a thread-count or smem limit is a **chunk-size parameter**, never a blocker

Any statement of the form "B=2048 doesn't fit so fused can't serve encoder/decoder" is
wrong. It means "process it in chunks of BC".

## Why chunking ALONE is not enough (do not stop there)

You could chunk B to <=256 and reuse `slstm_fused_time` today. **It would lose.** The
existing kernel does the recurrent product with scalar `fmaf` + `__shfl_down_sync`, so
its cost is **linear in B**; the per-step path uses cuBLAS (tensor cores) which is
**flat** until the tiles fill. Measured, fwd+bwd, T=512, H=1024 (ms):

| B | ours fused (scalar) | ours per-step (cuBLAS) | FlashRNN fused (wmma) |
|---|---|---|---|
| 1 | **3.31** | 11.50 | 5.67 |
| 8 | 29.32 | **16.62** | **5.55** |

Ours 3.31 -> 29.32 = 8.9x for 8x batch (perfectly linear). Theirs 5.67 -> 5.55 (flat).
**The tensor-core path is the whole point; chunking is just how you make it fit.**

## Target design (mirrors what FlashRNN does, ncu-verified)

ncu, B=1/T=512/H=1024, dense R:

| | ours fwd | theirs fwd | ours bwd | theirs bwd |
|---|---|---|---|---|
| grid x block | 79 x 640 | 64 x 256 | 79 x 416 | 32 x 256 |
| regs/thread | 90 | 173 | 128 | **255** |
| smem | **78.2 KB** | 9.0 KB | 0.1 KB | 26.5 KB |
| tensor-pipe inst | **0** | 8,388,608 | **0** | 8,388,608 |

**They hold `R` in registers and use wmma; we stage `Whr` in 78 KB of smem and use
scalar FMA.** Our smem footprint is what forces one block/SM and what makes geometry
decline at large B. Register-resident R attacks both at once.

Register budget check: 256 threads x 255 regs = 65280 regs = **261 KB register file per
block**. A block owning `NCOL` gate columns holds an `[H, NCOL]` bf16 slice =
`2*H*NCOL` bytes. At H=1024: NCOL=128 -> 256 KB, i.e. ~32 blocks over 4H=4096 columns.
**That matches their observed grid of 32 (bwd) / 64 (fwd) exactly** -- good evidence the
model is right.

### Proposed structure

```
grid  = (4H / NCOL) blocks, must be <= sm_count (84) for cooperative co-residency
block = 256 threads (8 warps)
each block owns NCOL contiguous gate columns, ALL of the chunk's BC batch rows
registers: R slice [H, NCOL] bf16, distributed across the block's threads
smem: h tile [BC, H] bf16 for the current timestep (small: BC=16, H=1024 -> 32 KB)

per timestep t:
  1. load h_{t-1} tile [BC, H] (from the h mirror written last step)
  2. mma.sync.m16n8k16 over K=H: acc[BC, NCOL] += h_tile x R_regs
     - M=16 tiles the batch (BC=16 fills M exactly; BC=32/64 -> 2/4 M-tiles)
     - N=8  tiles the gate columns
     - K=16 tiles the reduction over H
  3. add Wx[t] slice + bias -> gate pre-activations for owned columns
  4. pointwise recurrence for owned (row, unit) pairs -> h_t, c_t, n_t, m_t
  5. write h_t to the mirror; grid.sync()
```

Batch chunking: host loops chunks of `BC` rows; each chunk is an independent sweep. State
tensors are `[B, H]`, so a chunk is a row-slice -- no repacking.

`BC` is a tuning knob: larger BC amortises the R register load over more rows (R is read
once per timestep regardless of BC), so **prefer the largest BC that fits**. Sweep it.

## Constraints that are real (design inputs, not blockers)

1. **sm_120 has nothing better than `mma.sync.aligned.m16n8k16` for bf16** -- no
   tcgen05, no wgmma. Verified against two independent sources. We already have this
   working and bit-exact on bf16 slabs in `src/gpu/cuda/mlstm_fused.cu`. **Port from
   there; do not import an external repo.**
2. Cooperative launch requires the whole grid co-resident -> blocks <= 84. With
   `NCOL = 4H/blocks` this is satisfied by construction.
3. `ptxas -v` is the ONLY way to see register-array spills here; a spilled R slice
   silently destroys the design. Check it every build (see [[ptxas-v-shows-spills]]).
4. Keep the scalar `slstm_fused_time` for **B=1**. It is not legacy: at B=1 the
   recurrent op is a mat-VEC, tensor cores have nothing to fill, and we beat FlashRNN
   ~1.6x there with ZERO tensor instructions.

## Work order

1. **B-aware dispatch guard** + a test asserting which path each production shape picks.
   `FUSED_MIN_T` gates on T alone; T decides whether launch amortisation has enough
   steps, **B decides the sign of the trade**. Crossover measured at **B ~= 32**.
   Immediate: at B=8/T=512 we currently pick 29.32 ms when per-step does 16.62 ms.
2. **`slstm_fused_time_batched` forward** (new file `src/gpu/cuda/slstm_batched.cu`),
   parity-tested against the per-step path at B=1/64/256/1024.
3. **Backward.** Same structure; expect higher register pressure (theirs hits the 255
   cap). Parity-test dx AND all three weight grads.
4. **Batch chunking in the host launcher** so any B works.
5. Only then remove `slstm_step_fused` -- it is the mandatory fallback while any
   geometry can decline, and correctness tests pin math, not which path ran.

## Shapes it must serve (measured, `examples/_grp_probe.rs`)

One 4096-word window -> **20 encoder groups**, `B*T` roughly constant (~2048):

| len (T) | rows (B) | pieces |
|---|---|---|
| 1 | 1024 | 2 |
| 2 | 682 | 2 |
| 3 | 512 | 1 |
| 4 | 409 | 2 |
| 5..7 | 341, 292, 256 | 1-2 |
| 8..16 | 227 down to 120 | 1 each |

~149 timesteps total per layer per window; T>=8 groups carry ~108 of them. Long words
have the SMALLEST batch. `GROUP_MAX_ROWS = 2048`, `MAX_WORD_BYTES = 16`,
`CHAR_HIDDEN = 256`, `WORD_HIDDEN = 1024`, `BACKBONE_CHUNK = 512`.

Backbone (keep scalar kernel): B=1, T=512, H=1024.

Current per-step cost at these shapes is ~100% host-issue-bound: 5.5 us cuBLAS
(`run_staged_lhs`, measured `examples/gemm_issue.rs`, `issue/wall = 1.00` at B=120) +
1.73 us kernel, per timestep.

**Prize, honestly:** encoder+decoder sLSTM is ~16 ms of a ~101 ms window; backbone sLSTM
is 76 ms. Single-digit percent of a step. Worth doing; not a multiple.

## Measurement discipline (violating this cost hours)

- SM clock swings **1372-2880 MHz**. Interleave arms within a round, score the
  **minimum** over rounds, never the mean.
- Anything **< 2% of a step is invisible to wall clock**. Use
  `ncu --csv --metrics gpu__time_duration.sum` and diff per-kernel sums between arms.
- **Never read Duration from an ncu run with rule sections enabled** -- replay inflates
  it ~300x and *unevenly*, reordering the ranking (8610 ms reported against a real 24).
- ncu's "Est. Speedup %" is **per-kernel in isolation**, never step-level.
- **ncu's occupancy advice is wrong for cooperative kernels** -- it ranks
  `slstm_fused_time` first on "50% occupancy"; that kernel is one block/SM by design and
  raising occupancy measured **2x slower**.
- ncu runs unprivileged since `/etc/modprobe.d/nvidia-profiling.conf` set
  `NVreg_RestrictProfilingToAdminUsers=0`.
- zsh does **not** word-split unquoted vars: `for C in "1 512"; do prog $C` passes ONE
  arg and examples silently fall back to defaults. Use `${=C}`.
- NVRTC failures are SILENT: a broken `.cu` makes GPU tests SKIP, not fail. Grep for
  "skipping GPU test". NVRTC has no `INFINITY`/`uintptr_t` (use `-1e30f`).
- **Never script-edit `.cu` files** -- a python slice once dropped a brace and the silent
  NVRTC failure faked a 40% slowdown.

## Tools

| | |
|---|---|
| `examples/tloop_only.rs [B T H]` | our fused T-loop alone (span comparable to `flashrnn()`) |
| `examples/slstm_path_ab.rs` | fused vs per-step, interleaved, min-of-rounds, reports declines |
| `examples/gemm_issue.rs` | host issue cost of the per-timestep recurrent GEMM |
| `examples/_grp_probe.rs <file>` | real encoder length buckets |
| `python/run.sh <script> ...` | FlashRNN with a CUDA runtime matching torch |
| `python/_one.py <backend> B T N D` | one FlashRNN config per process |

FlashRNN needed three fixes to run here (`python/patch_flashrnn.py` + `run.sh`,
idempotent): CUDA 13 removed 8 `cudaDeviceProp` fields; `cuda_init.py` hardcodes
`arch=compute_80` ignoring `TORCH_CUDA_ARCH_LIST`; system CUDA 13.3 vs torch cu128 put
two `libcudart` in one process -> SIGSEGV with **0 compute-sanitizer errors** (host-side).
`N=1, D=H` makes their block-diagonal `R` dense, matching our `whr`.
**A segfault with 0 compute-sanitizer errors is host-side: `gdb -batch -ex run -ex bt`.**
