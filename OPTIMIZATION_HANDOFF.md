# GPU optimization reference — hierarchical xLSTM on RTX 5080

Researched and measured knowledge for optimizing this trainer. Reference material, not
a changelog: everything here is a fact about the hardware, the tools, or the SOTA
kernels, gathered so a session doesn't have to re-derive or re-search it.

---

# 1. Hardware facts (sm_120 / RTX 5080)

Verify with `gpu_occupancy` rather than trusting these; the user changes config.

| | |
|---|---|
| Compute capability | **12.0 (sm_120, consumer Blackwell)** |
| SMs | 84 |
| Peak DRAM bandwidth | **~960 GB/s** (256-bit GDDR7 @ 15001 MHz) |
| Max shared memory per block | **99 KB** (opt-in), ~100 KB/SM |
| Registers | 65536/SM, 255/thread max |
| Threads/SM | 1536 (48 warps), 12 warps/scheduler × 4 schedulers |
| L2 | large (~64 MB class); measured non-uniform: ~79 cyc near vs ~180 cyc far |
| Issue | 2/cycle, no triple-issue; 4 static sub-cores, warps bound by `warpid % 4` |
| Outstanding loads | ≥32 tracked per warp, scoreboard depth ≥12 |

**sm_120 keeps:** thread-block clusters + DSMEM, TMA/`cp.async.bulk`, `cp.async`,
`mbarrier`, `mma.sync`.
**sm_120 removes:** `tcgen05`, `wgmma`, Tensor Memory (TMEM), 2-SM cooperative GEMM.
Hopper/datacenter-Blackwell GEMM code will not assemble for sm_120.

**Practical consequences**
- Block sizes that aren't multiples of 128 threads leave sub-cores unevenly loaded.
- Deep scoreboard + many outstanding loads make the ILP strategy (§3.2) work *better*
  here than on older parts.
- The big L2 means working sets that were DRAM-resident on Ampere may now be L2-
  resident — check `lts__t_sector_hit_rate.pct` before assuming DRAM is the constraint,
  and prefer the *hierarchical* roofline over the DRAM-only one.
- **TMA multicast is ~10000× degraded on sm_120** — cluster-multicast patterns from
  Hopper GEMM literature are actively harmful.

## Model dimensions

`WORDS_PER_SEQ = 4096`, `MAX_WINDOW_TOKENS = 24576`, `MLSTM_CHUNK = 256`.
Real window: **4069 words / 24575 tokens**. Always print the actual `words, tokens`
alongside any step time — figures from smaller windows are not comparable.

`WORD_HIDDEN=768` → backbone dqk=dhv=96. `CHAR_HIDDEN=OUT_HIDDEN=256` → enc/dec
dqk=dhv=32. `WORD_BLOCKS=32`, 8 heads.

---

# 2. Tooling

## Harnesses

```bash
# Profiling target: ONE model, one window shape, warmup then N timed steps.
cargo run --release --features cuda --example kern_prof [words] [steps]

# Correctness: same window N times through a never-stepped model, comparing loss BITS.
cargo run --release --features cuda --example determinism [words] [reps]

# VRAM by phase (the user's hard constraint)
cargo run --release --features cuda --example vram_audit [words]

# Driver-reported occupancy / registers / shared memory per kernel
cargo run --release --features cuda --example gpu_occupancy
```

`chunk_ab` builds **four** models in one process — its traces mix four configurations
under the same kernel names. Never profile with it.

## nsys

```bash
nsys profile --trace=cuda --sample=none --cpuctxsw=none -o out --force-overwrite true \
  ./target/release/examples/kern_prof 4096 4
nsys stats --report cuda_gpu_kern_sum --format csv out.nsys-rep | head -15
```

For anything the canned reports don't cover, query the `.sqlite` directly:

```python
import sqlite3
c     = sqlite3.connect('out.sqlite')
names = {r[0]: r[1] for r in c.execute('select id,value from StringIds')}
rows  = sorted(c.execute('select start,end from CUPTI_ACTIVITY_KIND_KERNEL order by start'))
rows  = rows[len(rows)//4:]              # drop JIT/startup or utilisation reads too low
busy  = sum(e - s for s, e in rows)
span  = rows[-1][1] - rows[0][0]         # busy/span = real GPU utilisation
# per-kernel: group by shortName -> (count, total ns); diff two runs to find regressions
```

Whole-trace utilisation includes `cuLibraryLoadData`/JIT and **lies** — always cut to
steady state.

## ncu — requires the user

`ncu` returns `ERR_NVGPUCTRPERM` for a normal user. **Ask the user to run it and paste
output**; batch several kernels per request.

```bash
sudo ncu --kernel-name "regex:^(kernelA|kernelB)$" --launch-count 6 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats \
  ./target/release/examples/kern_prof 512 1 2>&1 | grep -v "^==PROF== Profiling"
```

Permanent fix (user, needs reboot): `options nvidia NVreg_RestrictProfilingToAdminUsers=0`
in `/etc/modprobe.d/`, then `sudo mkinitcpio -P` (CachyOS/Arch) and reboot.

### Metrics worth asking for

```
gpu__time_duration.sum
dram__bytes_read.sum, dram__bytes_write.sum
dram__throughput.avg.pct_of_peak_sustained_elapsed
sm__throughput.avg.pct_of_peak_sustained_elapsed
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
sm__warps_active.avg.pct_of_peak_sustained_active
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
lts__t_sector_hit_rate.pct
```

### Interpreting the output

- **Memory% high (>60), SM% low** → memory-bound. Move fewer *bytes* (fusion), not
  more threads.
- **SM% high, memory% low** → compute-bound.
- **Both below 60%** → **latency-bound**. The most commonly misdiagnosed case. Fix with
  more in flight (ILP, vector width), not more occupancy.
- "Memory Throughput" in SoL is a **max over L1/L2/DRAM sub-metrics** — it can read
  >100% while DRAM sits at 8%. Always read the DRAM line; expand the Memory Throughput
  Breakdown to see which level actually binds.
- **Sectors per request** is the coalescing diagnostic. A sector is 32 B. For a warp of
  32 threads × 4 B = 128 B contiguous, **4 sectors/request is ideal**; **32 means every
  lane hit a different sector** (8× wasted bandwidth). For `float4`, 16 is ideal.
  Interpret against the access width, not against 1.0.
- Stall breakdown: high `long_scoreboard` = waiting on global memory (add ILP/width).
  High `barrier` = `__syncthreads` with uneven work before it.
- **Nsight's "Est. Speedup: N%" is LOCAL to that kernel.** It knows nothing about the
  surrounding algorithm and can point the wrong way (see §5, the occupancy trap).

---

# 3. Optimization techniques

## 3.1 Coalescing — highest leverage, cheapest diagnosis

Official guidance ranks this first. On CC 6.0+ coalescing resolves in **32-byte
sectors**:

| pattern | sectors/warp | efficiency |
|---|---|---|
| aligned unit-stride 4 B/thread | 4 | 100% |
| misaligned start | 5 | ~80% |
| stride 2 | 8 | 50% |
| stride ≥ 8 (≥32 B) | up to 32 | ~1/32 |

**The antipattern to grep for:** `blockIdx.x * blockDim.x + threadIdx.x` followed by an
inner `for` over a contiguous dimension. That is one thread per row walking the row —
adjacent lanes land a full row apart, bottom row of the table.

**The fix:** one **BLOCK** per row, `for (i = threadIdx.x; i < W; i += blockDim.x)`,
then a shuffle reduction. Helpers `block_reduce_sum`/`block_reduce_max` live at the top
of `src/gpu/cuda/mlstm_ops.cu`; launch with `ops::row_block_cfg(rows, row_len)`.

**Two traps when writing block reductions here:**
1. The `parts[32]` staging array **must be padded with the reduction's identity**.
   Blocks can be as narrow as 32 threads, so the final warp-0
   `__shfl_down_sync(0xffffffff, …)` includes lanes past `nwarps`; those read unwritten
   shared memory and fold a previous block's residue into the result — silent,
   nondeterministic corruption, not a crash.
2. No early `return` before a `__syncthreads()`, and give the helper a **trailing**
   `__syncthreads()` so a caller may reduce twice from the same scratch.

Alternative when the layout can't change: stage through shared memory to transpose —
the canonical use of shared memory per the Best Practices Guide.

## 3.2 Occupancy vs ILP (Volkov)

Governing rule is **Little's Law: bytes in flight = latency × bandwidth.** You reach it
with many threads × few bytes, *or* few threads × many bytes — the product is what
matters. Volkov's memory-bound copy measurements:

| per-thread work | occupancy needed for ~peak |
|---|---|
| 1 float | ~100% (and only reaches 85%) |
| 4 floats | 25% |
| 8 × `float4` | **8%** (87% of peak) |
| 14 × `float4` | 4% (84%) |

For this GPU: ~960 GB/s and ~372–877 cycle DRAM latency → order **135–320 KB in flight
device-wide**, i.e. ~1.6–3.8 KB/SM. At 16 B/thread that's only **2–8 warps per SM**.
(The two published latency figures disagree by 2.4×, so treat this as a hypothesis to
validate, not a spec.)

Key mechanism: **threads don't stall on memory access, only on data dependency.** So
structure grid-stride bodies as load-all-then-store-all:

```cuda
float4 a0 = src[i];
float4 a1 = src[i + stride];      // issues immediately
float4 a2 = src[i + 2*stride];
dst[i] = f(a0);                   // first stall is here
```

**Consequence: do not reflexively raise occupancy on a bandwidth-bound kernel.**
Measured here: raising occupancy on the mLSTM backward made the step **2× slower**.

## 3.3 Vectorized loads

`float4`/`ushort4`, 4 elements/thread, grid-stride, with a runtime 16-B alignment check
and scalar fallback (pattern in `bf16_cast.cu`). `cudaMalloc` guarantees 256-B
alignment but any *offset* pointer into a sub-buffer must be checked.

Helps when launches are **too small to saturate** — a scalar thread has 4 B in flight
and needs enormous occupancy to reach the roof. **Neutral on already-saturated
kernels**, where a warp's scalar accesses already coalesce into full sectors.

`__ldg` is obsolete; the modern spelling is `const T* __restrict__` on the parameter,
which is free and sometimes unlocks vectorization.

## 3.4 Kernel fusion — the model is exact

NVIDIA's own benchmark (`sum(abs(x))`, 1 GB, RTX 4090):

| | traffic | time |
|---|---|---|
| 2 kernels | 3 GB | 3.51 ms |
| fused | 1 GB | **1.18 ms** |

Both ran at ~850 GiB/s (~90% of peak) — **the win was purely fewer bytes.**

**Rule: speedup = (bytes before) / (bytes after) = passes eliminated.** Two chained
elementwise ops = 4 passes → 2 = **2×**. Three ops = 6 → 2 = **3×**. Ceiling is the
unavoidable input read + output write.

Caveats: only pays when ops share data **through DRAM** (fusing over different arrays
saves only launch overhead); elementwise-before-reduction fuses cleanly into the
reduction's load, reduction-then-elementwise does not.

Canonical targets: cast + accumulate (fuse the cast into the producer's *store* — free);
column-sum + elementwise on the same matrix; a multi-kernel optimizer step (Adam's
read grad/m/v/w + write m/v/w is a fixed 7-pass footprint that should be one kernel).

## 3.5 Grid-stride vs one-thread-per-element

No published crossover point; the mechanism is what decides.

**Grid-stride wins** when per-thread setup is amortizable, when you want a fixed tuned
grid, when fusing a chain, when element counts vary, or when you want the ILP of §3.2.
**One-thread-per-element wins** when the body is genuinely one load-op-store.

Size a fixed grid from `cudaOccupancyMaxActiveBlocksPerMultiprocessor × 84`, then sweep
*downward* — max occupancy is often not fastest.

**Common bug:** striding by a fixed chunk per thread (thread *i* takes elements
`i*K … i*K+K`) destroys coalescing. Stride by `gridDim.x * blockDim.x`.

## 3.6 L2 residency control (`cudaAccessPolicyWindow`)

Available on sm_120 (CC 8.0+). Up to **+50%** when the persistent set fits the
set-aside region; **−10%** from thrashing when it doesn't. Set
`hitRatio ≈ set_aside / working_set` rather than 1.0.

```c
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prop.persistingL2CacheMaxSize);
attr.accessPolicyWindow = { base_ptr, num_bytes, hitRatio,
                            cudaAccessPropertyPersisting, cudaAccessPropertyStreaming };
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

For **streaming** kernels (read once, write once — most elementwise work here) the
correct policy is the opposite: mark them `cudaAccessPropertyStreaming` so they don't
evict genuinely reused data (optimizer moments, recurrent state).

## 3.7 `__launch_bounds__`

Always give at least the 1-arg form matching the largest launch — caps registers so the
block is guaranteed launchable on future hardware. **The 2-arg form may *increase*
register use** up to the derived limit to cut instruction count. Optimal values differ
across architectures; don't port a value tuned on Ampere. For bandwidth-bound kernels
don't reflexively squeeze occupancy up (§3.2).

## 3.8 What NOT to reach for on sm_120

- **TMA** — designed for bulk tiles feeding tensor cores; adds descriptor setup and an
  `mbarrier` wait to save nothing on a flat elementwise stream. Multicast is broken here.
- **`cp.async` / `memcpy_async`** — exists to overlap global→shared staging with
  compute; these kernels have almost no compute to overlap. A published study found a
  41.78% kernel-time reduction yielded **<1% overall** on memory-bound workloads, and
  naive substitution *degraded* GEMM.
- **Clusters/DSMEM** — the one Blackwell feature with plausible upside (cluster-scoped
  reduction avoiding a second launch or global atomics), but unmeasured on sm_120.
  Speculative; try §3.4 first.

---

# 4. SOTA reference: NX-AI mlstm_kernels / TFLA / FlashLinearAttention

From reading the actual Triton sources (`mlstm_kernels/triton/chunkwise/xl_chunk/`,
`fla/ops/common/chunk_h.py`) and TFLA (arXiv 2503.14376).

**Backward is 4 kernels in strict order:** recompute forward C states → dC recurrent →
dV → dK → dQ.

**dC is its own sequential-over-chunks kernel**, never fused with the intra-chunk part.
Grid `(num_b_DHQK, num_b_DHHV, B*NH)`; the chunk loop is *inside*, accumulator in
registers:

```python
matDeltaC_k_val = tl.zeros((siz_b_DHQK, siz_b_DHHV), dtype=tl.float32)
for k in range(NC, 0, -1):
    if k % save_states_every_nth_chunk == 0:      # store BEFORE updating
        tl.store(matDeltaCstates_k_ptr, ...)
    scaGbar_k_val   = tl.exp(scaG_k_val + scaM_inter_km1_val - scaM_inter_k_val)
    matDeltaH_k_val = matDeltaH_k_val / (vecN_out_k_val[:, None] + EPS)
    matDeltaC_k_val = scaGbar_k_val * matDeltaC_k_val + tl.dot(matQbar_k_val, matDeltaH_k_val)
```

**TFLA's core trick — a second level of parallelism inside a chunk.** dQ/dK/dV each
parallelize one L dimension and *loop* the other, permuting roles (which is why they
can't be one kernel):

| kernel | grid | parallel L | looped L | inner |
|---|---|---|---|---|
| dQ | `(num_b_DHQK, num_b_LQ, NC*B*NH)` | `siz_b_LQ` | `siz_b_LKV` | DHHV |
| dK | `(num_b_DHQK, num_b_LKV, NC*B*NH)` | `siz_b_LKV` | `siz_b_LQ` | DHHV |
| dV | `(num_b_DHHV, num_b_LKV, NC*B*NH)` | `siz_b_LKV` | `siz_b_LQ` | DHQK |

**Causality trims the loop instead of masking it** — dQ runs
`for idx_b_LKV in range(((idx_b_LQ+1)*siz_b_LQ)//siz_b_LKV)`; explicit `tl.where` only
on diagonal tiles.

**Concrete sizes:** `DEFAULT_CHUNK_SIZE=128`, `DEFAULT_CHUNK_BLOCK_SIZE=64`, head-dim
blocks `min(64, next_pow2(head_dim))`, **`num_stages=1` always** (they deliberately do
not rely on software pipelining), `num_warps = 4 if (siz_b_DHQK>=64 or siz_b_DHHV>=64)
else 2`. **No `triton.autotune` anywhere** in the chunkwise kernels — fixed heuristics.
FLA does autotune, and its space is a reasonable one to steal:
`BK, BV ∈ {32,64}`, `num_warps ∈ {1,2,4,8}`, `num_stages ∈ {2,3,4}`.

**Stabilizer:** *"For the backward pass there is no rescaling necessary as we store the
max states in the forward pass and reuse them."* Backward only ever **loads**
`vecM_out`, `scaMstate_all`, `vecN_out`; rescaling is always a difference of stored
maxes fed to `exp`. Two levels: `scaM_inter` (per chunk) and `vecM_out` (per timestep).
`vecA` is built with a stable reverse-cumsum, not the subtraction `vecB[-1] - vecB`.

**Deliberate reassociation worth copying:** hoist the normalizer division out of the
inner loop — *"we change the order of matrix multiply and division here … to avoid the
division in the inner loop, for better performance."*

**Recomputation:** `S = QKᵀ` is **fully recomputed** in every backward kernel, never
stored (FlashAttention-style). The gate matrix D is recomputed from `vecB`/`vecI`.
Forward C states are recomputed by default (`recompute_states_in_bw=True`) — the
biggest memory lever, since C is `[DHQK, DHHV]` per chunk. Only Q, K, V, i, f,
`vecN_out`, `vecM_out` are stored.

**Gate gradients are not in the kernels at all** — a compiled epilogue derived from
already-computed dQ/dK:
`dfbar = rev_cumsum((q*dq - k*dk).sum(-1))`, `vecDeltaF = vecDeltaFbar * sigmoid(-vecF)`,
`vecDeltaI = (matK*matDeltaK).sum(-1)`. Fusing it took **2.5 ms → 0.2 ms** at 1.3B/ctx8192.

**Bottleneck framing:** chunk size L is the dial between memory-bound and compute-bound.
FLA's limited chunk size gives *"low arithmetic intensity and high memory consumption
and IO cost"*; TFLA's second parallelism level exists to allow *"arbitrary large chunk
sizes and high arithmetic intensity."* Gains are relative (~25% over `limit_chunk`,
~2× vs Mamba2 on H100); no absolute achieved-TFLOP/s or bandwidth numbers found.

**Important caveat for this model:** at head dims 32 and 64 the paper notes
`limit_chunk` is **as fast as** TFLA — the extra machinery pays off at larger head dims.
This model is dqk=32 (enc/dec) and 96 (backbone). The older
`limit_chunk/bw_kernel_parallel.py` computes dQ/dK/dV in **one fused kernel** (grid
`(num_b_DHQK, NC, B*NH)`, materializing the full `L×L` S) and may be the better model
to copy; note it writes `matDeltaV` as `(num_b_DHQK, B, NH, S, DHHV)` needing a post-hoc
sum reduction over DHQK blocks.

## Sources

- CUDA C++ Best Practices Guide — memory optimization chapter
- [Kernel Fusion in NVIDIA CUDA](https://developer.nvidia.com/blog/kernel-fusion-in-nvidia-cuda-optimizing-memory-traffic-and-launch-overhead)
- [Volkov, "Better Performance at Lower Occupancy", GTC 2010](https://www.nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf)
- [CUDA Pro Tip: Grid-Stride Loops](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/)
- Nsight Compute Kernel Profiling Guide
- [Dissecting Blackwell with Microbenchmarks (arXiv:2507.10789)](https://arxiv.org/abs/2507.10789);
  [SM_120 microarchitecture](https://zartbot.github.io/micro_arch/nvidia/sm_120/paper.html)
- TFLA (arXiv:2503.14376); github.com/NX-AI/mlstm_kernels; github.com/fla-org/flash-linear-attention

---

# 5. Measured results on THIS model — do not re-derive

## Rejected, with the numbers

| idea | verdict |
|---|---|
| Apex multi-tensor-apply for adamw | 66% of launches are <4 µs but only **10%** of adamw time; ceiling ~1 ms/step. adamw is 1.8% of kernel time. |
| `float4` adamw | **Neutral** (71.2 → 71.1 ms). Already bandwidth-saturated. |
| Raising occupancy on `mlstm_bw_parallel_mma` | ncu says 33% occupancy, 88% "No Eligible", 73% of stalls at a CTA barrier, "Est. Speedup 73%" — chasing it was **2× slower**. |
| CPU/API-side work (pinned-slot reuse, device-side loss accumulation) | Real improvements, **0 ms**. Host cost was already hidden. |
| bf16 weights without fp32 master | **Dropped by the user.** Do not re-propose. |

## The occupancy trap, in full

`MLSTM_CHUNK` sweep at 4069 words — lower chunk = less shared memory = more blocks/SM:

| chunk | 16 | 32 | 48 | 64 | 128 | **256** | 384 |
|---|---|---|---|---|---|---|---|
| ms/step | 3151 | 1902 | 1534 | 1314 | 1040 | **940** | 965 |
| occupancy | 133% | 100% | 66% | 33% | — | — | — |

**100% occupancy is 2× slower than 33%.** Chunk *count* dominates: fewer chunks = fewer
sequential inter-chunk steps and far fewer launches. 512 and 0 (single-chunk) OOM.
`MLSTM_CHUNK = 256` is optimal. Nsight's per-kernel estimate knew nothing about this.

## The fused mLSTM path does not fit the backbone

`mlstm_fused_smem_bytes` vs the 99 KB cap:

| | dqk | L=16 | L=64 | L=128 | L=256 |
|---|---|---|---|---|---|
| enc/dec (h=256) | 32 | 18.5 KB | **60.5 fits** | 144.5 | 408.5 |
| backbone (h=768) | 96 | **99.5 NO** | 189.5 | 337.5 | 729.5 |

**The 32-block backbone cannot use the fused kernels at any chunk length** and runs
op-at-a-time — which is why `slice_t`/`mlstm_ds`/etc. dominate launch counts. A
shared-memory-light backbone kernel is the structural fix and likely the biggest lever
left. `mlstm_bw_parallel_mma` on the profile is entirely enc/dec work
(95.2 KB/block → 1 block/SM → 16 warps → 33%).

## Pinned-slot churn — fixed, 931 → 635 ms/step

The 48% idle was **not** launch overhead. Attributing each gap to the kernel that
followed it put **74% of all idle time (1804 of 2427 ms) in one seam**: `add` →
`rms_norm_forward`, 4860 gaps averaging **369 µs**. Launch overhead is ~1–2 µs, so
something was blocking. Cross-referencing the runtime API against those windows named
it: **`cuMemHostAlloc`, 780 of 855 ms sampled.** Page-locking is a synchronizing kernel
call, and it was on the critical path 256 times per step, every step — steady state,
not warmup.

Two independent bugs in `HostPark`'s slot pool, both in `evict`:

1. **`SPARE_MAX = 32` was below actual demand.** It was derived from `IN_FLIGHT_DEPTH`,
   but a chunked sweep keeps **one generation per chunk**, so peak demand scales with
   sequence length — measured 40 slots per park at 4069 words. Every step reclaimed 1280
   slots, dropped the 256 that overflowed, and page-locked them again (`spare_full` and
   `miss` were equal, at 256 each). The bound now tracks the park's own observed peak
   (`peak_slots + SPARE_SLACK`).
2. **First-fit matching fragmented the pool.** `position(|s| s.len() >= need)` let a
   small request take a large slot; the large requests then found nothing and allocated,
   once per sweep, forever. Now best-fit (`min_by_key(|s| s.len())`).

Each is independently necessary — `park_reuses_slots_across_a_chunked_sweep` fails with
either one reverted.

| | before | after |
|---|---|---|
| ms/step (4069 words) | 931 | **635** |
| GPU busy | 52.6% | **76.2%** |
| kernel busy time | 2690 ms | 2667 ms (unchanged) |
| span | 5116 ms | 3499 ms |
| `cuMemHostAlloc` / steady step | 256 | **0** |
| peak host RSS | 5557 MB | 5242 MB |

Kernel time did not move — **only the stalls closed**, which is the proof the diagnosis
was right. Host RSS went *down*: retaining slots beats alloc/free churn.

**Generalisable lesson:** attribute idle to the *following* kernel and then intersect
those windows with the runtime API trace. A per-kernel time ranking cannot see a host-
side stall at all — the kernel after the gap looks innocent, and the real cost has no
kernel row.

## Current profile

Numbers below predate the pinned-slot fix (they were taken at 52% busy / 931 ms). The
kernel-time *shares* still hold, but idle is now 24%, not 48%.

**3.52 s kernel time / 4 steps, 147k launches per step, 52% GPU-busy** — roughly half
the wall time is gaps between kernels. **The step is launch/idle-bound, not
kernel-bound.**

By share of kernel time: `mlstm_bw_parallel_mma` 7.4%, `cast_f32_to_bf16` 6.2%,
`rms_norm_backward` 4.7%, cuBLAS `nvjet_*` ~3.8%, `cumsum_logsig` 3.5%,
`revcumsum_dlogsig` 3.5%.

By **launch count** (per 4 steps) — the more actionable ranking now:

| kernel | launches | time | avg |
|---|---|---|---|
| `cast_f32_to_bf16` | 119,826 | 218 ms | 1.8 µs |
| cuBLAS `Kernel2` | 46,788 | 536 ms | 11.5 µs |
| `slice_t` | 39,424 | 90 ms | 2.3 µs |
| `broadcast_row` | 30,023 | 96 ms | 3.2 µs |
| `add_col_sum` | 25,403 | 65 ms | 2.5 µs |
| `unslice_t` | 21,504 | 39 ms | 1.8 µs |
| `add` | 19,264 | 59 ms | 3.1 µs |
| `add_assign` | 17,136 | 38 ms | 2.2 µs |

ncu on `add`/`broadcast_row`: DRAM **0.2–8.7%**, SM **2–5%**, *"grid too small — only
0.25 full waves across all SMs"* (21 blocks on an 84-SM GPU). They do essentially no
work and pay full launch cost.

## Block-per-row rewrites (done)

Four kernels used the thread-per-row antipattern — one thread walks a row serially,
so adjacent lanes sit a full row apart (uncoalesced) and the whole launch fits in
one or two blocks on an 84-SM GPU. Rewritten block-per-row with warp-shuffle scans
and `block_reduce_sum`; `GPU_NO_BLOCK_SCAN=1` reverts all four.

| kernel | before | after |
|---|---|---|
| `cumsum_logsig` | 17.76 ms | 0.49 ms |
| `revcumsum_dlogsig` | 19.62 ms | 0.57 ms |
| `masked_softmax_ce` | 6.83 ms | 0.16 ms |
| `mlstm_chunk_ab_bwd` | 2.12 ms | 0.73 ms |

Whole-step kernel time **529.0 → 447.9 ms**. Occupancy is the tell: `grid=1` or
`grid=2` at `block=1024` is one or two SMs of 84 — check `gridX` against `sm_count`
before assuming a slow kernel is doing real work.

Batched `slice_t`/`unslice_t` (`slice_t_batch`, `GPU_NO_SLICE_BATCH=1` to revert)
also landed: 8,704 → 2,560 launches/step, ~8 ms.

## Batched-GEMM TF32 (done, −183 ms/10 steps)

`matmul_batched_{nn,nt,tn}` (the chunkwise-mLSTM per-head GEMMs) ran on the **CUDA
cores** in fp32 while the *same algorithm's* fused-kernel dots were already TF32
(`MMA_TF32`, matching the reference, where Triton's `tl.dot` is TF32 on fp32 inputs).
Switching them to `CUBLAS_COMPUTE_32F_FAST_TF32` via `gemm_strided_batched_ex` moved
them to the tensor cores. `GPU_NO_BATCHED_TF32=1` reverts.

Measured over 10 steps, and the diff is **entirely GEMM kernels** — nothing unrelated
moved, so this is not clock drift:

| | on | off |
|---|---|---|
| `simt_sgemm` + `magma_sgemmEx` (CUDA cores) | 119.8 ms | 520.1 ms |
| `tensorop` + `tf32_mma` (tensor cores) | 217.0 ms | 0 ms |
| **total kernel time** | **4765.1 ms** | 4948.2 ms |

Accuracy at production shapes (K=96/256): max abs error **4.3e-3 against a max |ref| of
15.4**, i.e. ~2.8e-4 relative — exactly TF32's ~10-bit mantissa. fp32 is ~1000× tighter
(3.8e-6), which is the A/B that proves the path is actually engaged.

This is distinct from the handle-wide `gpu::set_tf32` (`GPU_TF32=1`), which stays
opt-in because it also reaches the GEMMs the `gemm_*_matches_cpu` tests use as an
exact-fp32 oracle. These three wrappers are not on that path.

**`matmul_batched_matches_cpu` cannot catch a regression here** — it runs K=6, where
the TF32 error is 2.4e-7. Do not read it as evidence about production accuracy.

## `db` is not bit-reproducible (`add_col_sum`)

`add_col_sum` splits the row axis over `blockIdx.y` and combines slices with
`atomicAdd`, so summation order varies with scheduling; float addition is not
associative. Any test asserting **exact equality** on a `db` / bias gradient is
unsound and will flake. `forward_shared_matches_forward_and_saves_nothing` asserted
this and failed ~1-in-3 once TF32 changed GEMM timing enough to perturb scheduling —
the test was wrong, not the code. GEMM outputs (`y`, `dx`, `dw`) *are* exact.

## CUDA graphs: blocked, and why

Whole-step graph capture would be worth ~24% (the CPU sits 74% inside CUDA API calls
while the GPU is 76% busy), and the launch structure is ideal — **all ~78k launches
per step are bit-identical across steps**, 87% of them inside 67 repeating signatures.

It is blocked by **allocation instability**. A captured graph bakes in raw device
pointers; `GPU_PTR_PROBE=1` (see `gpu::dtensor::ptr_probe`) measures 44,066 allocations
per step of which only **0.8–1.3% reuse the previous step's address**, at both 4096 and
512 words. Replaying a graph would read and write reassigned memory — silent
corruption.

The sLSTM used to capture its T-loop — a loop over *persistent* state buffers, keyed
on `(b,t)` — and that was the one place the precondition held. It has been removed:
real training never repeats a window shape (windows never cross a document border),
so the cache missed constantly, and the time-fused cooperative kernel now covers the
one shape that made the capture worth having.

Consequence: **the flat parameter tensor is the gate on the graph win**, not a ~2%
optimization in its own right. That reprices it well above where it sits below.

## sLSTM: three saved slabs are shifted copies of three others

`slstm_step_fused` writes ten `[B, T, H]` slabs per timestep, and three of them are
redundant by construction:

| written           | equals                                    |
|-------------------|-------------------------------------------|
| `c_prev[:, t, :]` | `c[:, t-1, :]`                            |
| `n_prev[:, t, :]` | `n[:, t-1, :]`                            |
| `h_prev[:, t, :]` | `out[:, t-1, :]` (the pre-norm `h`)       |

with only the `t == 0` slot — the state the call carried in — genuinely new. Dropping
them costs one `[B, H]` buffer for that slot plus a `t == 0` branch in every backward
reader, and buys 30% of the step kernel's store traffic and **two fp32 plus one bf16
`[B, T, H]` buffer of activation memory per cell**, which at the backbone's shape is
~100 MB across a chunked window.

It is a forward *and* backward change (both fused kernels, both per-step kernels,
`SlstmSlabs`, `park_order`), and the time it saves is small — the memory is the reason
to do it.

## Ranked next steps

With idle down to 24%, the remaining wall time is mostly real kernel time, so this list
now reads more like a kernel-time ranking than it did. Re-measure busy-vs-idle before
assuming any of it.

1. **Fuse the small elementwise chains.** Attacks both kernel time and the residual idle.
   `slice_t`/`unslice_t` (61k launches) exist only to hand cuBLAS a contiguous chunk —
   TFLA indexes into the full tensor with block offsets instead of materializing copies.
2. **A shared-memory-light backbone kernel** so the backbone gets a fused path at all.
3. **`cumsum_logsig` / `revcumsum_dlogsig`** (7% combined) — still thread-per-row, but
   they are sequential *scans*, so they need a parallel scan (Hillis-Steele/Blelloch),
   not a reduction.
4. **`mlstm_bw_parallel_mma`** (7.4%) — apply §4: recompute `S = QKᵀ`, hoist the `n`
   division out of the inner loop.
5. **Flat parameter tensor** (user's idea). Fusing adamw is only ~1%, but collapsing
   6,181 launches into ~20 removes real *gap* time — frame it as launch-count, not
   bandwidth. Most invasive change on the list: layer ownership, the NNFW round-trip,
   `add_grads_from`.

---

# 6. Working rules for this repo

1. **Measure GPU busy-vs-idle before optimizing anything.** If busy% is low, kernel
   time is not the problem.
2. **Wall clock cannot measure a sub-50 ms change on this machine — use nsys kernel
   time.** `nvidia-smi` reports the SM clock idling at **1612 MHz against a 3090 MHz
   max**, so a run's speed depends on where it catches the boost ramp. Measured:
   interleaved same-process A/B of a change worth a known ~45 ms of GPU time gave
   711/522, 548/703, 548/537 ms — the sign flipped between reps. Thermal drift is
   real too (the same binary reads ~1150 ms warm and ~880 ms settled), but the clock
   swing is the larger effect and back-to-back A/B does **not** control for it.
   The reliable signal is nsys `sum(end-start)` per kernel over the measured region;
   a change in *kernel* time is trustworthy, a change in wall clock is not. A useful
   cross-check that a "regression" is really clock state: if kernels you did not
   touch moved by a uniform percentage at identical launch counts, it is the clock.
3. **NVRTC failures are silent.** A broken `.cu` makes GPU tests *skip* and report `ok`;
   the tell is the suite finishing suspiciously fast (~1 s instead of ~20 s). Verify:
   `cargo test … -- --nocapture 2>&1 | grep -c "skipping GPU test"` must print **0**.
   NVRTC has **no** `INFINITY`, `uintptr_t`, or `__builtin_huge_valf`; `size_t` is
   builtin; this codebase's −inf idiom is **`-1e30f`**.
4. **A new test must be verified to FAIL without its fix.** Race tests need the **real
   model widths** (hc=256, wh=768) — at toy sizes (hc=16, wh=24) the CPU never laps the
   device and races do not reproduce at any word count.
5. **Never `git checkout <file>`.** The tree carries large uncommitted user changes.
   Back up to the scratchpad instead. (Recovery, if it happens: `git fsck --lost-found`
   and grep the dangling blobs.)
6. **Run narrow test filters**, not the full ~85 s suite repeatedly. It is flaky under
   concurrency — `slstm_fused_time_matches_per_step`, `group_cap_matches_uncapped`
   have each failed once and passed 3/3 in isolation. Re-run the single filter before
   believing a failure. **But do not assume a failure is the known flake**: a repeat
   run under the A/B toggle told apart "flaky" from "my change broke it" for
   `forward_shared_…` — it passed 3/3 with the change off and failed 1/3 with it on,
   which is a real defect, not noise. `--test-threads=4` gives a clean 70/70.
7. **No long training runs.** Validate with `vram_audit` / `kern_prof` / `determinism`,
   never `hg`. Kill background GPU jobs and check `nvidia-smi` before handing back.
8. **Bisect toggles** (these localized a race in minutes): `GPU_NO_GROUP`,
   `GPU_NO_OFFLOAD`, `GPU_NO_WCACHE`, `MLSTM_NO_MMA`, `GPU_NO_BF16`, `MLSTM_CHUNK`,
   `GPU_BACKBONE_CHUNK`, `RAYON_NUM_THREADS`, `GPU_PHASE=1`, `GPU_MEM=1`.
9. **Async H2D from pinned memory needs a per-slot event the CPU waits on**
   (`ev.synchronize()`, *not* `stream.wait(ev)` — the racing writer is the CPU).
   Rotating N buffers alone only buys N−1 iterations of slack, and the CPU laps it
   inside a loop over groups. This has caused silent training divergence twice.

## Style (CLAUDE.md, enforced)

- Never `0.0_f32` — write `0.0`. Comments in English.
- **Comments describe the code, not the edit.** No "was X, now Y", no "NEW:", no
  "kept for compatibility", no "moved here from foo.rs".
- Short comments; explain only what isn't obvious (an invariant, a unit, why the naive
  approach fails, a paper reference). No banner lines (`// =====`).
