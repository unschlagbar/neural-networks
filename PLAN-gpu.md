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

## Laptop environment (the box with the RTX 4050)

Confirmed working 2026-07-11 on Manjaro:

- GPU: RTX 4050 Mobile (Ada, 6 GB). Driver present, `nvidia-smi` reports CUDA UMD 13.3.
- Toolkit: `sudo pacman -S cuda` -> `/opt/cuda` (nvcc/nvrtc/cuBLAS 13.3). `/opt/cuda/bin`
  is on PATH system-wide after install, so plain `cargo` picks up `nvcc`.
- Also required to build the crate at all: `sudo pacman -S pkgconf` (the `cpal`
  audio dep links ALSA via pkg-config; unrelated to CUDA but blocks the build).
- **cudarc version pin:** cudarc 0.17.8's auto-detect panics on toolkit 13.3
  ("Unsupported cuda toolkit version"). We pin the `cuda-13000` feature instead
  of `cuda-version-from-build-system`; with `dynamic-loading` the real 13.3 .so's
  are dlopen'd at runtime, so 13.0 bindings are fine.

Build/test the GPU path with: `cargo test --features cuda gpu:: -- --nocapture`.

## Progress

### Done (2026-07-11, laptop): CUDA toolchain proven

`Cargo.toml` gained a default-off `cuda` feature + optional `cudarc` dep; the CPU
reference build is unchanged. `src/gpu/mod.rs` has a lazy `Gpu` (context + default
stream) and a `vector_add_roundtrip` smoke test (NVRTC-compile -> upload -> launch
-> download -> verify) that **passes** — cudarc + NVRTC + driver + cuBLAS all link
and run on the laptop.

### Done (2026-07-11, laptop): device tensor + cuBLAS GEMM (all 3 forms)

- `src/gpu/dtensor.rs` — `DTensor` (resident device buffer + shape), `from_host`/
  `to_host`/`zeros`. A parallel type to host `Tensor` rather than surgery on it
  (keeps `Tensor`'s `Clone`/`PartialEq`/serialization untouched); data stays on
  the GPU across an op chain, crossing PCIe only at the explicit boundaries.
- `src/gpu/ops.rs` — `matmul` / `matmul_nt` / `matmul_tn` via cuBLAS, mirroring
  `tensor::gemm`. Row-major-over-column-major handled by the operand-swap trick
  (compute `Cᵀ = Bᵀ·Aᵀ`). **All three parity tests vs the CPU gemm pass.**
- `Gpu` now also carries an `Arc<CudaBlas>` handle.

### Done (2026-07-11, laptop): NVRTC elementwise/reduction kernel batch

`src/gpu/kernels.rs` — one NVRTC source string, compiled once at `Gpu::new` and
cached as `Gpu.kernels`. `src/gpu/ops.rs` gained thin launcher wrappers, each
parity-checked vs the CPU reference (all pass):

| Kernel(s) | CPU reference |
|-----------|---------------|
| `softcap_forward/backward` | `nn2::ops` |
| `broadcast_row`, `add_col_sum` (Linear bias) | `nn2::ops` |
| `embedding_gather`, `embedding_scatter_add` (atomic) | `nn2::ops` |
| `rms_norm_forward/backward` (grouped; `dgamma` atomic) | `nn2::ops` |
| `softmax_ce` (fused, per-row) | `nn2::loss` |
| `adamw` (bias-corrections precomputed host-side) | `nn2::optim` |

**11 GPU tests green** (`cargo test --features cuda gpu::`); CPU-only build
unaffected. Together with the cuBLAS GEMM this is the full kernel set the flat
model needs.

### Done (2026-07-11, laptop): Linear layer end-to-end on the GPU

`src/gpu/linear.rs` — a fully device-resident `Linear` (weights, grad
accumulators, AdamW moments, saved input all `DTensor`). Composes the ops:
forward = `broadcast_row` (bias seed) + `matmul_nn_into(beta=1)`; backward =
`matmul_tn_into(beta=1)` (dW) + `add_col_sum` (db) + `matmul_nt` (dX = dY·Wᵀ, no
host transpose — cuBLAS transposes W internally); step = `adamw` ×2. The GEMM ops
gained `_into(beta)` forms; `DTensor::dup` (device→device `clone_dtod`) saves the
forward input. **Parity test vs `nn2::Linear` passes** for forward → backward →
one AdamW step from identical weights (12 GPU tests total green).

### Done (2026-07-11, laptop): allocation cleanup + flat GPU stack + benchmark

**Alloc cleanup** (`DTensor` gained `uninit` = alloc without memset, and `zero_` =
in-place memset). cudarc already stream-orders allocation via `cuMemAllocAsync` +
`free_async`, so raw malloc was never the serializer; the waste was memsets on
fully-overwritten outputs and per-step grad reallocs. Fixed: all op outputs that a
kernel/GEMM writes in full now use `uninit`; `Linear::zero_grad` memsets in place;
`Linear::forward` reuses its saved-input buffer when the batch size is stable.

**Flat GPU stack** — `src/gpu/flat.rs`: `Embedding → Linear → RMSNorm →
Linear(head) → SoftCap → softmax/CE`, fully device-resident, composing every
ported op. `Flat::train_step` does forward→loss→backward→AdamW-step. Parity test
vs the identical `nn2` CPU stack passes: loss **and** every updated parameter
match after one step. **13 GPU tests green.** (No recurrent core yet — this proves
composition/wiring, not a real LM.)

**Benchmark** — `examples/gpu_bench.rs` (`cargo run --release --features cuda
--example gpu_bench`). GEMM: GPU ~6.6 TFLOP/s when boosted (~75% of the 4050's
FP32 peak) vs ~25 GF/s CPU → 100–290×. Linear fwd+bwd: 45–220×. **Caveat: this is
a Max-Q laptop part with aggressive power gating** — the *same binary* swings 10×
between runs (0.33 ms vs 1.84 ms for 1024³) as the SM clock bounces 2.1–3.1 GHz.
Pin with `sudo nvidia-smi -lgc <freq>` for stable numbers.

Remaining perf-debt (correctness-first, fine for now):
- `forward`/`backward` still allocate fresh output `DTensor`s per call.
- `from_host`/`to_host` copies block on the default stream.

## Recurrent core (sLSTM / mLSTM / block) — the real LM

The flat stack proves the seam works but has no recurrent core. This is the plan
for porting it. Everything below is grounded in the current CPU code
(`src/nn2/slstm.rs` 518 L, `src/nn2/mlstm.rs` 622 L, `src/nn2/block.rs` 264 L).

### Done (2026-07-11, dev box — WRITTEN BLIND, needs laptop verification): Phase A sLSTM cell

Phase A (below) is implemented but **has not yet run on a GPU** — it was written
on the no-CUDA dev box. The Rust plumbing typechecks under `--features cuda`
(cudarc `dynamic-loading` compiles without a toolkit); the NVRTC kernel **C**
source has never been compiled. **First action on the laptop: run
`cargo test --features cuda gpu::slstm -- --nocapture`** and fix any NVRTC compile
errors / parity failures.

- `src/gpu/kernels.rs` — 4 new kernels + 2 `__device__` helpers (`stable_sigmoid`,
  `log_sigmoid`): `concat_xh` (build `xh = concat(x_t, h_{t-1})`), `split_dxh`
  (its backward inverse → `dx` + `dh_bptt`), `slstm_cell_step` (elementwise
  recurrence over B·H, advances resident `(c,n,m,h)` state + fills saved tensors +
  writes `out[:,t,:]`), `slstm_cell_step_bwd` (mirror: gate deltas + BPTT
  channels). All added to `NAMES`.
- `src/gpu/ops.rs` — launcher wrappers for the four, plus `SlstmSaved` (the 9
  per-step `[B,H]` saved `DTensor`s, grouped so the layer holds a `Vec`).
- `src/gpu/slstm.rs` — `gpu::SLstm`: gates as `[DTensor;4]` (z,i,f,o), all
  weights/grads/moments/state/saved device-resident; state stays resident across
  the whole T-loop (no per-step host transfer). `from_parts` builds from a CPU
  cell's 8 host tensors. Parity test `slstm_matches_cpu_layer` vs `nn2::SLstm`
  (fwd/bwd/step, tol 2e-3 — cuBLAS vs CPU-gemm reduction order).

**Milestone A reached on the dev box only.** When the laptop confirms parity,
this becomes truly Done.

### Done (2026-07-11, dev box — WRITTEN BLIND, needs laptop verification): Phase B Block

Also written on the no-CUDA box; typechecks + clippy-clean under `--features cuda`,
NVRTC C never compiled. **Verify on the laptop with
`cargo test --features cuda gpu::block -- --nocapture` (and `gpu::rms_norm`,
`gpu::slstm`).**

- `src/gpu/kernels.rs` — 3 new elementwise kernels: `add` (residual/grad-sum),
  `swiglu_forward` (`gate_act = SiLU(gate_pre)`, `mixed = gate_act⊙value`),
  `swiglu_backward` (reuses the `stable_sigmoid` device helper). Added to `NAMES`.
- `src/gpu/ops.rs` — wrappers `add` / `swiglu_forward` / `swiglu_backward`. Also
  **refactored `GpuRmsForward`**: `rms_norm_forward` now returns `(out, saved)`
  and the saved struct no longer carries `out` (so a stateful norm can return its
  output while stashing only `x̂`/`inv_rms`). `flat.rs` + the ops test updated.
- `src/gpu/dtensor.rs` — `DTensor::reshaped(self, dims)`: metadata-only by-value
  reshape (zero copy; `[B,T,H] ↔ [N,H]` for the cell↔norm boundary).
- `src/gpu/rms_norm.rs` — device-resident `gpu::RmsNorm` (γ/dγ/moments + saved
  fwd; undecayed AdamW), wrapping the grouped ops with `group == size`.
- `src/gpu/block.rs` — `gpu::Cell` trait (`impl` for `SLstm`) + `gpu::Block<C>`:
  composes 3 `RmsNorm` + 3 `Linear` + the SwiGLU kernels + a generic cell exactly
  as `nn2::block`. Norms/MLP run on the flat `[N,H]` view; only the cell sees
  `[B,T,H]`. Parity test `slstm_block_matches_cpu` vs `nn2::SLstmBlock`
  (fwd/bwd/step, tol 3e-3). `SLstm.w`/`.bias` made `pub` for the test.

Correctness-first debt (fine for now): `Block::forward`/`backward` each `dup` the
input once (borrowed `&DTensor` → owned `[N,H]`); `ops::add`/`swiglu_*` allocate
fresh outputs per call.

### Done (2026-07-11, dev box): sLSTM speed test + Phase C groundwork primitives

- `examples/gpu_bench.rs` gained an **sLSTM cell fwd+bwd+step** row (CPU vs GPU
  over `[B,T,H]`) — the launch-bound recurrent baseline the chunkwise mLSTM must
  beat, so we don't repeat the old sub-1× scalar-recurrence mistake.
- Phase C primitives landed + parity-tested (see the Phase C section): strided-
  batched GEMM (`matmul_batched_{nn,nt,tn}`) and the projection kernels
  (`scale_`, `sigmoid_`). The chunkwise mLSTM's full math is now derived in the
  Phase C blueprint below.

**Phase C (steps 1–3, 5) and Phase D are now implemented blind.** The port is
functionally complete: `gpu::Lm` is a real, fully device-resident language model
with a recurrent core (sLSTM + mLSTM blocks), trainable end to end.

**Phase C is complete as of 2026-07-12** — including step 4 (chunking, O(T)), done
and verified on hardware. `cargo test --features cuda gpu::` (24 tests) verifies the
mLSTM cell (single-chunk AND chunked), both block flavours and the whole LM against
their CPU twins.

### Orientation (what already exists to build on)

- **GPU ops ready to reuse:** cuBLAS `matmul{,_nt,_tn}{,_into}`; `broadcast_row`
  (== per-step bias add), `add_col_sum` (== `accum_bias`, batch-summed bias grad);
  grouped `rms_norm_forward/backward` (**already covers head-wise RMSNorm** — call
  with `group = dhv`); `softcap`, `embedding`, `adamw`. `DTensor` supports rank ≤ 4
  (`MAX_RANK`), so mLSTM `C[B,H,dhv,dqk]` (rank 4) and `n[B,H,dqk]` (rank 3) fit.
- **`gpu::Linear`** is the model of how to port a layer (weights/grads/moments/
  saved-input all `DTensor`, parity-tested). sLSTM/mLSTM/block follow the same shape.
- **CPU reference & test style:** cells are FD-checked with a *directional whole-
  tensor* finite difference (project loss along `G/‖G‖`, expect `‖G‖`), **loose tol
  ~0.3** because the max-stabilizer `m_t` is treated as constant in backward (kinks).
  Reuse this exact harness for the GPU cells (compare GPU vs CPU cell output/grads
  directly — tighter — AND keep an FD check). `AdamCfg.t` starts at 1.

### Architecture recap (from block.rs)

`Block<C: Cell>` wraps a generic recurrent `Cell` in two residuals:
```
z = x + post_cell_norm( cell( pre_norm1(x) ) )
y = z + lin_down( SiLU(lin_gate·pre_norm2(z)) ⊙ (lin_value·pre_norm2(z)) )   // SwiGLU
```
Norms + SwiGLU MLP are position-wise → run on the flat `[B·T, H]` view; only
`cell` sees `[B,T,H]`. So `Block` on GPU = compose existing `gpu::Linear` +
`rms_norm` ops + **three new elementwise kernels** (`silu_forward`,
`silu_backward` i.e. `silu'`, and a fused `mul`/residual-`add`) around a GPU cell.
`Cell` trait: `forward([B,T,H])->[B,T,H]`, `backward`, `zero_grad`, `step`.

### Phase A — sLSTM cell (do first; simplest, state is tiny `[B,H]`)

Weights: 4 gates `w{z,i,f,o}` each `[rows,H]`, `rows = in+H`; biases `[H]`;
`bf` init 4.5. Forward is a serial T-loop; batch is the parallel axis; state
`(h,c,n,m)` is `[B,H]` and **must stay in device buffers across the whole T-loop**
(never download per step). Per step:
1. Build `xh = concat(x_t, h_{t-1})` `[B,rows]` — **new kernel** `concat_xh`
   (gather `x[:,t,:]` + current `h_state`), and its inverse `split_dxh` in backward.
2. 4 gate GEMMs `[B,rows]·[rows,H]` (cuBLAS) + `broadcast_row` bias.
3. **New kernel** `slstm_cell_step`: the elementwise recurrence over `B·H`
   (lines slstm.rs 241–271) — reads gate pre-acts + `(c,n,m)_state`, writes
   `(c,n,m,h)_state` and the step's saved tensors + `out[:,t,:]`. All state buffers
   resident; one launch per step (B·H threads).
Backward mirrors it: **new kernel** `slstm_cell_step_bwd` (lines 321–353) producing
gate deltas `dz,di,df,dob` + BPTT channels `dc/dn/dh_bptt` (resident), then
`matmul_tn_into(beta=1)` for `dW`, `add_col_sum` for `db`, `matmul_nt` for `dxh`,
`split_dxh` into `dx[:,t,:]` and `dh_bptt`. 8 AdamW calls in `step` (gates decay,
biases don't).
**Milestone A:** `gpu::SLstm` parity vs `nn2::SLstm` (cell alone) + FD check.

### Phase B — Block (wire A into the residual block)

New elementwise kernels: `silu_forward`, `silu_backward` (`silu'(x)=σ(1+x(1-σ))`),
`ew_mul` (+ its backward), and an `axpy`/residual-add (or reuse a generic
`add`). Compose `gpu::Linear` ×3 + `rms_norm` ×3 + the sLSTM cell exactly as
block.rs. **Milestone B:** `gpu::Block<SLstm>` parity vs `nn2::SLstmBlock`.

### Phase C — mLSTM cell via the **chunkwise / parallel** formulation (the hard one)

Do **not** port the scalar per-head recurrence (mlstm.rs loops B·H doing outer
products `v⊗k` and `C·q` — poor GPU locality; it's the current sub-1× CPU path).
Instead do the attention-like parallel form. The `gpu_bench.rs` "sLSTM cell" row
is the launch-bound recurrent baseline to beat.

#### Groundwork landed (2026-07-11, dev box, parity-tested primitives)

Everything the parallel form is built from is already ported + parity-tested vs
its CPU twin (so the hard layer only has to get the *wiring* right):
- **Strided-batched GEMM** (`ops::matmul_batched_{nn,nt,tn}`, cudarc
  `gemm_strided_batched`, test `matmul_batched_matches_cpu`) — the per-(B·H)
  chunk matmuls. Lay tensors out `[B*H, ·, ·]` (head-major) so one batched call
  covers all heads.
- **Projection kernels** `ops::scale_` (k's 1/√dqk) and `ops::sigmoid_` (o-gate),
  test `scale_and_sigmoid_match_cpu`. Projections themselves are flat GEMMs on the
  `[N=B·T, in]` view (reuse `Linear`/`broadcast_row`/`matmul_nn_into`).
- Head-wise RMSNorm = existing `rms_norm` with `group == dhv`; `add`,
  `softcap`, `adamw` all ready.

#### The exact math (derived from the CPU recurrence — this is the parity target)

Key fact: the CPU stores the **stabilized** state `C_t = C_t^true·exp(-m_t)`,
`n_t = n_t^true·exp(-m_t)` (verified: with `f'=exp(logσ(f̃)+m_{t-1}-m_t)`,
`i'=exp(ĩ-m_t)` the `exp(-m_t)` factors telescope). And the running stabilizer
unrolls to a **row-max over the decay matrix**:
```
fc_t = Σ_{s≤t} logσ(f̃_s)                       (inclusive cumsum of log-forget)
logD_{tj} = fc_t - fc_j + ĩ_j     for j ≤ t     (0 / −∞ above the diagonal)
m_t = max( max_{j≤t} logD_{tj},  fc_t + m_prev )   (m_prev carried; init m_0 = 0)
```
Then per head (Q,K,V are `[L,dqk]`,`[L,dqk]`,`[L,dhv]`; K already ×1/√dqk):
```
S = Q·Kᵀ                                   (L×L, matmul_batched_nt)
D̄_{tj} = exp(logD_{tj} - m_t)  (j≤t else 0)     (a mask+exp kernel)
num_t   = Σ_j (D̄⊙S)_{tj} v_j = ((D̄⊙S)·V)_t      +  exp(fc_t+m_prev-m_t)·(C_prev·q_t)
qn_t    = Σ_j (D̄⊙S)_{tj}      = rowsum(D̄⊙S)_t   +  exp(fc_t+m_prev-m_t)·(q_t·n_prev)
ψ_t = max(|qn_t|, 1) ;  ỹ_t = num_t / ψ_t   ( = st.cq/ψ in CPU terms)
```
`inter` terms vanish in the **single-chunk** (whole-sequence, C_prev=n_prev=0,
m_prev=0) form — but the `fc_t + m_prev` term still enters `m_t`. Then the tail is
exactly the CPU's: head-norm(ỹ)→ŷ, `y = o⊙ŷ`, `h = y·W_out + b_out`.

Multi-chunk O(T): carry `C[B·H,dhv,dqk]`, `n[B·H,dqk]`, `m[B·H]` across chunks;
end-of-chunk update `C_L = exp(fc_L+m_prev-m_L)·C_prev + (Vscaled)ᵀ·K` with
`Vscaled_j = exp(fc_L-fc_j+ĩ_j-m_L)·v_j` (a `matmul_batched_tn`), likewise `n_L`.

#### Build order (each a parity checkpoint vs `nn2::MLstm`, tol ~2–3e-3)

1. ~~**Projections**~~ **DONE (dev box, blind).** Composed from `gpu::Linear`
   (six projections + `W_out` — same layout/AdamW convention as CPU `project`),
   with `scale_` (k's 1/√dqk) and `sigmoid_` (o-gate).
2. ~~**Single-chunk forward**~~ **DONE (dev box, blind — needs laptop verify).**
   `src/gpu/mlstm.rs` `gpu::MLstm::{from_parts,forward}` + 7 new kernels
   (`head_gather`/`head_scatter`, `cumsum_logsig`, `mlstm_rowmax_m`, `mlstm_ds`,
   `div_rows`, `mul`) + `gpu::RmsNorm::from_parts_grouped` (head-wise, group=dhv).
   Forward-parity test `mlstm_forward_matches_cpu` vs `nn2::MLstm::forward`
   (single chunk, L=T). Math verified by hand: `D̄_{tj}` telescopes to the CPU's
   stabilized `i'_j·Πf'_s`, and `m_t` = the row-max form of the running max.
   **← laptop: run `cargo test --features cuda gpu::mlstm` first.**
3. ~~**Single-chunk backward**~~ **DONE (dev box, blind — needs laptop verify).**
   `gpu::MLstm::{backward,step,zero_grad}` + 5 new kernels (`ogate_bwd`,
   `div_rows_bwd`, `mlstm_ds_bwd`, `mlstm_dfc_dig`, `revcumsum_dlogsig`); `mlstm_ds`
   now also emits `D̄`. Chain: `lin_out.backward` → `ogate_bwd` (fold σ') →
   head-norm bwd (group=dhv) → `head_gather` → `div_rows_bwd` → dV/dDS via
   batched `tn`/`nt` → `mlstm_ds_bwd` (dS + P=dD̄⊙D̄) → dQ/dK via batched `nn`/`tn`
   → `mlstm_dfc_dig` (P → dfc,dig) → `revcumsum_dlogsig` (dfc → d f-logit) →
   `head_scatter` all grads → sum of six `lin_*.backward`. `m` held const (ref
   approximation). Parity test `mlstm_matches_cpu` (fwd+bwd+step, tol 3e-3); the
   CPU backward is itself FD-verified so GPU-vs-CPU is the tighter check.
   `impl Cell for gpu::MLstm` added. mLSTM row added to `gpu_bench.rs` (scalar CPU
   vs parallel GPU, ms/iter). **← laptop: run `cargo test --features cuda gpu::mlstm`.**
4. ~~**Chunking** (L < T, inter-chunk state carry) for O(T) + long sequences~~
   **DONE (2026-07-12, laptop, on hardware).** `config::MLSTM_CHUNK` (default 256,
   `MLSTM_CHUNK=<L>` env override, 0 = single-chunk) + 10 kernels (`slice_t`,
   `unslice_t`, `mlstm_chunk_ab`, `mul_rows`, `mul_rows_add`, `psi_from_qn`,
   `row_dot_add`, `group_dot_add`, `mlstm_chunk_ab_bwd`). Forward carries
   `C[BH,dhv,dqk]`, `n[BH,1,dqk]`, `m[BH]`; backward sweeps chunks in reverse
   carrying `dC`/`dn` (BPTT over chunks, parallel form within each).

   Two things fell out of the derivation and made it much easier than feared:
   - **`fc` must be the chunk-LOCAL cumsum.** Then the existing `mlstm_rowmax_m`'s
     `fc_t + m_prev` branch (written for exactly this and previously fed a zero
     `m_prev`) makes the chunk-local row-max telescope to the *global* one — so
     chunking is an exact refactoring, not an approximation, and chunked vs
     single-chunk agree to fp tolerance in forward AND backward.
   - **`a_j` is just the last row of D̄, and `g` is `b_last`** — the end-of-chunk
     state update needs no new exponentials.

   Parity: `mlstm_chunking_matches_single_chunk` (L ∈ {1,3,8,16,32} incl. a short
   final chunk, vs the single-chunk path, 2e-6 on the weight update) and
   `mlstm_chunked_matches_cpu` (multi-chunk vs the CPU scalar recurrence).

   **Measured (RTX 4050, B=1 d=512 16 heads, fwd+bwd ms/iter):**

   | T | single-chunk | L=256 | |
   |---|---|---|---|
   | 512 | 5.08 | 4.69 | |
   | 1024 | 17.32 | 9.31 | |
   | 2048 | 63.53 | **18.90** | **3.4x**, and linear in T (was quadratic) |

   End-to-end (`gpu_fit`, worst window of political_speeches.xml, 1236 words) the
   step is only ~4% faster — the mLSTM blocks are ~20 ms of a ~550 ms window, which
   the sLSTM backbone blocks and the encoder/decoder phases dominate. The real
   end-to-end win is **VRAM: peak 3132 → 2332 MB (−26%)**, which is headroom for a
   longer `WORDS_PER_SEQ` (the whole point of O(T)).
5. ~~`gpu::Block<MLstm>` vs `nn2::MLstmBlock`~~ **DONE (blind).** Parity test
   `mlstm_block_matches_cpu`. Added `BlockLike` (type-erased `Block`, so a model can
   hold a heterogeneous `Vec<Box<dyn BlockLike>>`) and `from_cpu` uploaders:
   `SLstm::from_cpu`, `MLstm::from_cpu`, `Block::<SLstm>::from_cpu`,
   `Block::<MLstm>::from_cpu` (useful beyond tests — loading a trained CPU model
   onto the GPU).

**Milestone C COMPLETE (2026-07-12, on hardware):** `gpu::MLstm` parity vs
`nn2::MLstm` in both the single-chunk and the chunked form, `gpu::Block<MLstm>` vs
`nn2::MLstmBlock`, and step 4 (chunking → O(T)) landed and benchmarked.

### Phase D — assemble the real GPU LM + retire the flat toy

**DONE (2026-07-11, dev box, blind — needs laptop verify).** `src/gpu/lm.rs`:
`gpu::Lm` = `Embedding → Block×N (alt sLSTM/mLSTM) → RMSNorm → Linear head →
SoftCap → fused softmax/CE`, fully device-resident (table, every block, final
norm, head + all grads/moments are `DTensor`s). A forward → loss → backward →
AdamW-step cycle crosses PCIe only for the token ids in and the scalar loss out.
Blocks are `Vec<Box<dyn BlockLike>>` (the two `Block<C>` types differ).
`Lm::train_step` returns the mean CE loss.

Optimizer convention held: table / norm scales / logit head undecayed; the blocks'
interior projections decay.

Parity test `lm_matches_cpu` — the **end-to-end Phase-D check**: an LM with *both*
an sLSTM and an mLSTM block, vs the identical CPU `nn2` stack, comparing loss and
every updated parameter after one AdamW step. This exercises every ported kernel
composed in the real architecture.

**VERIFIED ON HARDWARE (2026-07-12, laptop): all 20 `gpu::` tests pass**, including
the mLSTM parallel-form cell (fwd+bwd), both block flavours and `lm_matches_cpu`.
The blind-written chunkwise derivation was correct on the first hardware run.

#### Benchmark results (RTX 4050 Mobile vs AVX2 CPU) — reproducible

**Benchmark harness caveat (important, cost us a bogus first result):** this is a
Max-Q part that idles at **210 MHz** and boosts to **3105 MHz**. At `iters=10` the
timed region was 10–250 ms — mostly clock ramp-up, so the *same* config swung
12×↔75× between runs. Fixed by decoupling the iteration counts (`CPU_ITERS=3`,
`GPU_ITERS=100`, `GPU_WARMUP=30`): the GPU now gets a long, boosted, steady-state
region and results reproduce to <1%. For absolute stability: `sudo nvidia-smi -lgc 3105`.

| Bench | CPU | GPU | Speedup |
|---|---|---|---|
| GEMM 1024³ | 24 GF/s | 6600 GF/s | **271×** |
| Linear fwd+bwd+step (4096,1024,2048) | 23 GF/s | 6040 GF/s | **258×** |
| sLSTM cell fwd+bwd+step (64,64,512) | 26 GF/s | 3155 GF/s | **119×** |
| mLSTM cell fwd+bwd+step (32,64,512,8h,64) | 787 ms | 6.2 ms | **128×** |
| **LM train_step** (vocab 4096, B16 T64 H512) | **2129 ms** | **52.7 ms** | **40×** |
| LM train_step (vocab 256, B16 T32 H256) | 205 ms | 11.1 ms | 18.5× |

The mLSTM headline: the CPU scalar per-head recurrence takes **787 ms** for one
fwd+bwd+step where the GPU parallel form takes **6.2 ms**, and the speedup *grows*
with size (38× → 103× → 128×) — the opposite of the old sub-1× scalar path. The
parallel/attention reformulation was the whole point and it worked.

Remaining for Phase D: retire `gpu::flat` (the recurrent-core-less toy) now that
the real LM is verified.

### Done (2026-07-12, laptop, VERIFIED): Phase E — GPU hierarchical + train + checkpoint

The thing you can actually run. `src/gpu/hierarchical.rs` + `src/gpu/train.rs`,
wired into `run()` as stdin mode **`hg`**.

- **Architecture fix (user-caught):** `nn2::Hierarchical` had grown *three*
  stage-level RMSNorms. The real model (`model.rs::build_hierarchical_model`) has
  exactly **one** — the decoder's, before the logit head. The encoder ends at its
  last `slstm_block` and the backbone is `Linear → blocks → Linear`, neither with
  a norm. Removed the encoder + backbone norms from `nn2` (all 16 CPU tests still
  green) and matched it in the GPU port. CLAUDE.md and a stale `model.rs` comment
  documented the wrong architecture; both fixed. SwiGLU `up = 8·h/3` is derived
  **per block** from its own width (so it differs per stage), as the builder does.
- `gpu::Hierarchical`: tied char table (encoder input + decoder char slots),
  encoder → backbone → decoder, all device-resident. Only 2 new kernels were
  needed (`scatter_rows`, `masked_softmax_ce`) — the `[W]`-step readout and the
  decoder-slot grads reuse `embedding_gather`/`scatter_add` as row gather/scatter.
- **Checkpointing (`GHIR`)**: `params_mut()` on every layer gives a fixed param
  order; save writes config header + step count + every tensor, and renames only
  after a complete write. Test `hierarchical_memorizes_and_checkpoints` proves it
  learns *and* that a reloaded model reproduces the exact same loss.
- `train_hierarchical_gpu` reuses `TrainingState` (LR warmup/cosine schedule, CSV
  logging, print/save intervals) and streams `ChunkedWordDataSet`, accumulating
  grads over `BATCH_SIZE` windows.

**Smoke-tested end to end on a synthetic corpus:** loss 5.41 → 3.61 (ppl 223 → 37)
in 180 steps, checkpoint written, **resumed from step 180**, peak GPU memory
629 MiB / 6141. Note the O(T²) mLSTM is a non-issue in practice because windows
never cross document borders, so the backbone length is the *document's* word
count (~100s), not `WORDS_PER_SEQ`.

**Not done: GPU sampling/inference.** Training + checkpointing work, but there is
no `hgs` sampling mode yet. The parallel mLSTM is a *training* form (it consumes a
whole sequence); autoregressive generation needs either a re-run-the-prefix loop
or the incremental recurrent single-step form.

### Ground rules (carry into every phase)

- **State residency:** recurrent state lives in `DTensor`s across the T-loop /
  chunk-loop; the only host transfer is the final output (and inputs in). A
  per-step `to_host`/`from_host` would erase the win (see 4.5× residency measurement).
- Correctness-first: new cell kernels can be one-thread-per-`(b,h)` naive launches;
  optimize after parity. Use `uninit` for fully-written outputs, `zero_`/memset for
  accumulators and BPTT channels.
- Every new kernel gets a parity test vs its `nn2` CPU counterpart before it's wired
  into a layer; every new layer gets a parity + FD test before the next phase.
- Run `cargo test --features cuda gpu::` after each kernel; keep the CPU build green.

### Kernel-launch gotcha (for the next pass)

`launch_builder(...).arg(&x)` **borrows** each arg for the life of the builder, so
scalar kernel args must be bound to a `let` first — `.arg(&(n as i32))` fails to
compile ("temporary dropped while borrowed"). Bind `let n_i = n as i32;` etc.

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
