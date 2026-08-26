# `python/` — reference-implementation benchmarks

A side-project for measuring our CUDA kernels against NX-AI's reference
implementations. Nothing in the Rust crate depends on this; it exists to answer
"is there headroom left?" with a number instead of an argument.

## Setup

```bash
python3 -m venv python/.venv
python/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
```

`cu128` is required: this box is an RTX 5080 (sm_120, compute cap 12.0), and older
CUDA wheels have no sm_120 kernels. The download is ~8 GB and slow; the wheels cache
in `~/.cache/pip`, so a re-run after an interrupt resumes cheaply.

FlashRNN itself is vendored in `flashrnn/` and **git-ignored** (NXAI Community
License — vendoring it here is a licensing decision, not a default). Obtain it from
https://github.com/NX-AI/flashrnn if the directory is missing.

## Running

```bash
python/.venv/bin/python python/frnn_bench.py
```

FlashRNN JIT-compiles its CUDA on first use, so the first run is slow. It needs
`TORCH_CUDA_ARCH_LIST` to include this card; the script sets `12.0` itself.

## What the benchmark compares

`frnn_bench.py` times FlashRNN's sLSTM at **our** shapes. `flashrnn()` takes `Wx`
already computed, so the comparable span on our side is `slstm_fused_time` /
`_bwd` alone — **not** `SLstm::forward`, which also runs the whole-sequence `x·Wx`
GEMM. Timing the wrong span hands FlashRNN a ~30% head start it did not earn.
`examples/tloop_only.rs` measures exactly that span on the Rust side.

`N=1, D=H` makes their block-diagonal `R` dense, which is exactly our `whr [H, 4H]`.

### Our baseline to beat (2026-08-25, RTX 5080)

`cargo run --release --features cuda --example tloop_only`

| shape | fwd ms | fwd+bwd ms |
|---|---|---|
| backbone B=1 T=512 H=1024 | 1.411 | 3.357 |

That is 2.76 µs per timestep forward.

### Why this comparison is worth running

Their fused kernel does the recurrent matmul with `wmma` (tensor cores); ours does it
with scalar `fmaf` plus a warp shuffle reduction. Measured consequence: our fused path
beats the per-step path 5.3× at B=1 but loses 2× at B≥32, because scalar cost scales
with the batch and tensor-core cost does not. At B=1 — the backbone, and the only
regime production runs — scalar is the *correct* choice and SOTA dispatch practice
agrees. This benchmark establishes whether that is still true against a real wmma
implementation, or whether we are leaving time on the table.

## Measurement discipline

SM clock swings 1372–2880 MHz on this card, which is larger than most effects being
measured. Interleave arms within a round and score on the **minimum** over rounds,
never the mean. For anything worth under ~2% of a step, wall clock cannot resolve it
at all — use `ncu --csv --metrics gpu__time_duration.sum` and diff per-kernel sums.
Do **not** read Duration from an ncu run with rule sections enabled: replay inflates
it ~300× and unevenly, which reorders the ranking.
