# Plan: GPU port of `nn2` via real CUDA (`cudarc`)

> **Status: seam started 2026-07-11.** This doc lives in the repo (not in
> Claude's local `~/.claude` memory, which does not sync between machines) so it
> is visible on any machine that pulls the repo — e.g. the laptop with the Nvidia
> GPU where the CUDA kernels actually get compiled and run.

## Decisions (user-chosen)

- **Backend: real CUDA via `cudarc`.** Not wgpu, not candle — the project keeps
  its own hand-written primitives, so we write our own kernels rather than adopt
  someone else's tensor library.
- This dev machine (Intel Arrow Lake iGPU, Vulkan only) **has no Nvidia GPU and
  no CUDA toolkit**, so CUDA code is written here blind and **compiled/run on the
  laptop**. Keep the CPU path as the always-available reference + correctness
  oracle (the finite-difference tests).

## Why the rework already made this tractable

The `nn2`/`tensor` rework was the hard part and it is done:

- Flat, contiguous `f32` tensors (`Tensor { data: Vec<f32>, shape, rank }`) — a
  GPU buffer is literally this. Direct upload, no marshalling.
- Batched everything — the leading batch axis turns per-timestep matrix×vector
  into matrix×matrix GEMM, which is what the hardware wants.
- No autograd graph — manual forward/backward means every op is already an
  explicit, ordered kernel dispatch. Nothing to port but the kernels.
- Pooled, explicit dataflow — no hidden per-step allocations.

## The enabling refactor: the Backend/ops seam

Rule: **no layer touches `Tensor.data` for math.** Every elementwise / reduction
/ gather op is a named free function in [`src/nn2/ops.rs`](src/nn2/ops.rs) taking
`Tensor`s (so a `Tensor` can later hold a *device* buffer). Each such function =
**one future CUDA kernel**. Adding the GPU backend = implement these bodies
against device buffers, one kernel at a time; the layer code above them does not
change. The FD tests tell you immediately whether each kernel is correct.

GEMM (`tensor::gemm`), fused softmax/CE (`nn2::loss`) and AdamW (`nn2::optim`)
are already single-choke-point free functions — already "kernels" in this sense.

## Progress

### Done (2026-07-11, CPU impls, behaviour-preserving, all 16 nn2 tests green)

`src/nn2/ops.rs` created and these layers routed through it:

| Layer | Ops it now calls |
|-------|------------------|
| SoftCap | `softcap_forward`, `softcap_backward` |
| Embedding | `embedding_gather`, `embedding_scatter_add` |
| RmsNorm + HeadwiseRmsNorm | unified `rms_norm_forward` / `rms_norm_backward` (grouped: `group == width` for plain, `group == dhv` for head-wise) |
| Linear | `broadcast_row` (bias seed), `add_col_sum` (bias grad); GEMM/transpose stay in `tensor::gemm` |

### Not yet converted — the hard part, next pass

**`slstm` / `mlstm` / `block`** — ~218 `.data` sites, the max-stabilized
recurrence. These are the fused-recurrence kernels. Notes for that pass:

- The **chunkwise-mLSTM rewrite belongs here**: "chunkwise mLSTM" and "GPU" are
  the same project — it turns the per-head scalar recurrence into attention-like
  batched matmuls that GPUs love, and fixes the current mLSTM sub-1x CPU perf
  (matrix-valued state C[B,H,dhv,dqk] has poor recurrent-state cache locality).
- The recurrence is sequential but fine on GPU **as long as state stays in device
  buffers across the T-loop**. The trap is per-step CPU↔GPU transfer — never do
  that. sLSTM state is tiny (`[B,H]`) so per-step launches are cheap; the mLSTM
  per-head work is launch-bound and needs a fused/chunkwise kernel.

## Suggested GPU bring-up order (on the laptop)

1. Add `cudarc`; give `Tensor` an optional device buffer + up/download.
2. Port the already-seam kernels first: GEMM (cuBLAS or own), elementwise +
   reductions (Linear, norms, softmax, softcap, embedding gather/scatter, AdamW).
   This alone GPU-accelerates the flat model and the backbone.
3. Convert sLSTM/block to the ops seam (CPU first, FD-verified), then port.
4. Tackle mLSTM via the **chunkwise** formulation rather than porting the
   per-head scalar recurrence.
