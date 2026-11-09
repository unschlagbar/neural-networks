# Refactor plan

Goal: one CPU implementation, one checkpoint path, no dead code, short comments.
Baseline commit: `448e5f5`. Do one step per session, and finish each with a green
build plus the tests named in the step. Tick a step off here when it is done.

## Decisions

- **Keep `src/nn2`, delete `src/nn`, then rename `nn2` → `cpu`** (beside `gpu`:
  `cpu::Hierarchical`, `cpu::MLstm`, `cpu::Block`). No "2" or "new"/"old" in any
  name; rename types the same way where they carry such a suffix. nn2 is batched and is already the GPU's
  reference and init source. nn is per-timestep, and the checkpoint round trip
  through it is what cost 17 GB of host RAM.
- CPU modes that stay: `h`, `hq`, `hv` (training/validation), `hs`, `hqs`
  (sampling), `i` (inspect). GPU modes all stay.
- Removed: `""` + `s` (flat model), `av` (grow_vocab), `hp` (probe), `ht` (trace).
- Inference gets its own small module (no training state, no grads): it is what
  `harness` links against.

## Steps

- [x] **1. Dead modes and examples.** Delete the `""`, `s`, `av`, `hp`, `ht` modes
  and the code only they reach (`grow_vocab.rs`, flat-model parts of `model.rs` /
  `training.rs` / `sampling.rs`, `gpu/flat.rs` + `gpu/lm.rs` if nothing else uses
  them). Delete the examples listed below. Fix CLAUDE.md's mode table.
  Check: `cargo check --features cuda --examples --tests`, `cargo check -p harness`.

- [x] **2. Checkpoint format v2.** A header (`ModelCfg` + step/seen counters) plus the
  parameter tensors in `param_slots` order, written straight from device
  tensors by the GPU model and read by nn2. No CPU layer objects in between.
  Add a one-shot `convert` mode (old HIER → v2), which still uses `src/nn`, and
  convert `models/s4`–`s9` and `s4_sft_audio`. Keep the originals until step 5.
  Check: GPU save→load is bit-exact; a converted model gives the same `hvg` loss
  as the original.
  Done: the header carries a per-stage `Layout` of block-kind tags (not just counts),
  because s4/s5 have an sLSTM block at every backbone index ≡ 4 mod 6. All seven are
  converted to `models/<name>.v2`; `hvg` on s4 vs s4.v2: char 0.501 / word 2.595 both.

- [x] **3. nn2 inference.** Stateful `feed` / `continue` / `sample` over the nn2
  hierarchical model (backbone state carried, encoder/decoder reset per word),
  loading v2. Port `hs`, `hqs` and `harness` (`session.rs`, `listen.rs`) to it.
  Check: identical sampled tokens (greedy) vs the old `Hierarchical` on `s4`.
  Done: `src/infer.rs`, standalone (no nn/nn2 types, no grads), weights kept in the
  v2/GPU layouts, `feed` / `feed_embeddings` / `continue_chat` / `sample` /
  `sample_chat`. Greedy output is identical to the old model on `s4` and
  `s4_sft_audio` (three prompts × 300 tokens plus a chat reply), at ~2x its speed
  (~32 ms per backbone word, bandwidth-bound). `examples/infer_parity.rs` is the
  check; it goes with the old model in step 5.

- [x] **4. CPU training on nn2.** `h`, `hq`, `hv` on nn2 + v2 checkpoints, with
  the same masking (SFT loss mask, document borders) as the GPU path.
  Check: the hierarchical mask tests ported to nn2, and a short `h` run lowers
  the loss.
  Done: `cpu::hierarchical` tests cover the full-mask and per-word-mask gradients,
  packed document borders, both FD checks of the old model (tied table, injected
  context), v2 round trip and `infer` loading. The full config does not fit in host
  RAM, so the short run was the `h` step on a small model (hc 128, wh 256, 256-word
  windows of `pretrain_v2`): held-out loss 5.86 → 2.08 over 200 windows.

- [x] **5. Delete the old CPU stack.** `src/nn/`, `src/hierarchical.rs`,
  `sequential.rs`, `nn_layer.rs`, `word_encoder.rs`, `parallel.rs`, `format.rs`
  (HIER), `loading.rs`/`saving.rs`, `gpu/nn_convert.rs`, the optimizers nn2 does
  not use (Muon, SGD, the `GradMatrix` aliases), `tests/parallel_parity.rs`
  (already failing, see below), and the `convert` mode. Keep sLSTM backbone
  blocks: s4 (deployed) has them. Rename `src/nn2` → `src/cpu` (and every `nn2::` path,
  including the GPU side and the `.cu` comments).
  Check: `cargo test --release --features cuda --lib` with a narrow filter per
  module; `harness` builds.
  Done: `Sequential` stays (smart-light's wake-word detector runs on it), moved to
  `src/sequential/` with only its general layers (linear, no-bias linear, embedding,
  SiLU dense, LSTM, sLSTM, RMSNorm, conv1d, dropout, soft cap), one Adam(W) in
  `optim.rs` instead of the swappable optimizers, and its `NNM1` flat container in
  `io.rs` (byte-identical: a file saved at `d525d91` loads and re-saves to the same
  bytes). The mLSTM / block layers, replica and parallel code went with the old
  model. Byte primitives live in `src/bytes.rs`. `i` reads v2 (or a flat file), the
  `ae2` text scale and `ae2s` use `infer::Hierarchical`. `models/*` `NNM1`
  originals are still on disk and no longer readable.

- [ ] **6. Comment and doc pass.** File by file, per CLAUDE.md style: cut
  restating comments, history ("was X"), paragraphs where one line will do,
  German comments. Shrink CLAUDE.md to what is true after steps 1–5. Move
  measurement history out of doc comments (it lives in memory/commits).
  Check: no code change (`git diff --stat` shows only comment lines; build green).

## Examples

Keep (current tools):
`gpu_soak`, `vram_audit`, `offload_vram`, `phase_table`, `gpu_prof`, `kern_prof`,
`ncu_blocks`, `step_time`, `eval_win`, `determinism`, `window_determinism`,
`grad_coverage`, `gpu_cpu_parity`, `param_count`, `parquet_demo`, `seg_demo`,
`win_carry_stats`, and the audio ones from 2026-11 (`ae2_*`, `augment_*`).

Delete (one-off probes and benches whose finding is recorded):
`bgrad_probe`, `chunk_ab`, `colsum_bench`, `ctx_ablation`, `encdec_baseline`,
`enc_shapes`, `eval_leak`, `e_w_cosine`, `forget_collapse`, `fused_bf16_parity`,
`gemm_issue`, `geom_probe`, `gpu_bench`, `gpu_fit`, `gpu_memcheck`,
`gpu_occupancy`, `gpu_win`, `grad_dump`, `grad_norm_probe`, `grad_seed`,
`_grp_probe`, `launch_cost`, `logit_scale`, `loop_confidence`, `mlstm_bench`,
`mlstm_bw_bench`, `mlstm_chunk_bench`, `mlstm_determinism`, `mlstm_fw_bench`,
`mlstm_oracle`, `mlstm_path_check`, `mlstm_spec_ab`, `mlstm_stage_prof`,
`mlstm_xtree`, `nn2_bench`, `path_parity`, `proj_merge_bench`, `rms_bench`,
`sample_parity`, `sft_mask_audit`, `slstm_backbone_probe`, `slstm_bench`,
`slstm_bwd_bench`, `slstm_fit`, `slstm_fwd_bench`, `slstm_launch_bench`,
`slstm_path_ab`, `slstm_roofline`, `slstm_state_prec`, `smoltalk_sources`,
`temp_sweep`, `tloop_only`, `train_determinism`, `vary_win`, `viz_words`,
`weight_delta`, `weight_stats`, `window_count`, `window_spans`, `win_seq`.

## Known state

- Every hierarchical mode reads v2 only (`models/<name>.v2`).
- The CPU trainer does not fit the full config at 4096-word windows: measured
  16.9 GB peak at 1024 words (fwd+bwd, before the 8 GB of Adam moments), 67 s per
  window. It is for small models, SFT and checks.
- `harness` uses `infer::Hierarchical` for chat and `gpu::audio` for the ear. `smart-light` uses `audio` and `sequential`.
