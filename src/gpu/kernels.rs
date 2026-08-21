//! NVRTC-compiled CUDA kernels for the backend op seam.
//!
//! The device source lives in `src/gpu/cuda/*.cu` — real `.cu` files so an editor
//! or clangd can lint and highlight them — and is stitched back into ONE string by
//! [`SRC`] at Rust compile time. It is compiled once at [`Kernels::load`] and
//! cached on the [`Gpu`](super::Gpu). Each `extern "C"` kernel is the device
//! counterpart of a function in `src/nn2/ops.rs` / `loss.rs` / `optim.rs`, and is
//! launched by a thin wrapper in `src/gpu/ops.rs` that owns the grid/block config
//! and is parity-checked against the CPU reference.
//!
//! Conventions: tensors are the same row-major contiguous `f32` as on the host;
//! embedding ids and CE targets are uploaded as `unsigned int`; reductions that
//! accumulate into shared outputs (`embedding_scatter_add`, RMSNorm `dgamma`)
//! use `atomicAdd`.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction};
use cudarc::nvrtc::{CompileOptions, compile_ptx, compile_ptx_with_opts};

/// The whole device source, as one translation unit.
///
/// The pieces are separate files purely for tooling — NVRTC still sees a single
/// concatenated string, and it has to: `prelude.cu`'s `slab_t` typedef is at file
/// scope and every kernel that touches an sLSTM slab must be compiled against the
/// same width, and the `#ifdef` fences (`SLAB_BF16`, `BF16_CAST`, `COOP_KERNELS`,
/// `MMA_TF32`, `SLSTM_H`/`SLSTM_B`) select *across* the pieces. So the order below
/// is load-bearing: `prelude.cu` first, and each file must stay self-contained
/// enough that its `#ifdef`/`#endif` pairs balance within itself.
///
/// `include_str!` also means `cargo` reruns on any `.cu` edit, and a missing file
/// is a Rust compile error rather than an NVRTC error at first GPU touch.
const SRC: &str = concat!(
    include_str!("cuda/prelude.cu"),
    include_str!("cuda/common.cu"),
    include_str!("cuda/bf16_cast.cu"),
    include_str!("cuda/slstm_coop.cu"),
    include_str!("cuda/slstm_fused.cu"),
    include_str!("cuda/block.cu"),
    include_str!("cuda/mlstm_ops.cu"),
    include_str!("cuda/mlstm_fused.cu"),
);


/// Names of every kernel in [`SRC`], loaded into [`Kernels`].
const NAMES: &[&str] = &[
    "softcap_forward",
    "softcap_backward",
    "broadcast_row",
    "broadcast_row_resid",
    "add_col_sum",
    "embedding_gather",
    "embedding_scatter_add",
    "embedding_scatter_merge",
    "rms_norm_forward",
    "rms_norm_backward",
    "softmax_ce",
    "adamw",
    "adamw_batch",
    "concat_xh",
    "split_dxh",
    "slstm_cell_step",
    "slstm_cell_step_bwd",
    "slstm_pack_w",
    "slstm_pack_b",
    "slstm_unpack_dw",
    "slstm_unpack_db",
    "fill_const",
    "state_slot_copy",
    "sum_accum",
    "add_assign",
    "slstm_gate_matvec",
    "slstm_gate_matvec_t",
    "slstm_step_fused",
    "slstm_step_fused_bwd",
    "add",
    "swiglu_forward",
    "swiglu_backward",
    "scale_inplace",
    "sigmoid_inplace",
    "cumsum_logsig",
    "cumsum_logsig_block",
    "mul",
    "ogate_fwd",
    "slice_t",
    "slice_t_batch",
    "unslice_t",
    "unslice_t_batch",
    "row_dot_add",
    "group_dot_add",
    "mlstm_chunk_ab_bwd",
    "mlstm_chunk_ab_bwd_block",
    "ogate_bwd",
    "mlstm_dfc_dig",
    "revcumsum_dlogsig",
    "revcumsum_dlogsig_block",
    "mlstm_fw_gates",
    "mlstm_state_scan",
    "mlstm_bw_dqn",
    "scatter_rows",
    "masked_softmax_ce",
    "masked_softmax_ce_block",
];

/// Kernels compiled with `MMA_TF32` against the real device arch — the ones that
/// issue `mma.sync`, which does not exist at NVRTC's default target.
const MMA_NAMES: &[&str] =
    &["mlstm_fw_dC", "mlstm_fw_parallel", "mlstm_bw_dC", "mlstm_bw_parallel"];


/// Cooperative-launch kernels; need `<cooperative_groups.h>`, so they share the
/// bf16 module's include-path requirement.
///
/// `slstm_fused_time` is deliberately absent: the forward exists only in the
/// shape-specialized build (it holds its slice of `h` in a register array, which
/// needs H at compile time), so it is reached through [`Kernels::specialized`] and
/// this module's success is what tells the caller the specialized one will build.
const COOP_NAMES: &[&str] = &["slstm_fused_time_bwd"];

/// fp32 <-> bf16 casts. Need `<cuda_bf16.h>` only — a strictly smaller include
/// requirement than [`COOP_NAMES`], hence their own module: a machine that cannot
/// build the cooperative kernels can still store activations in bf16.
const BF16_NAMES: &[&str] = &["cast_f32_to_bf16", "cast_bf16_to_f32"];

/// Directory holding `cuda_bf16.h`, or `None` if it cannot be found.
///
/// `dynamic-loading` means the crate never links a fixed CUDA install, so there is
/// no build-time constant to fall back on; NVRTC still needs a real `-I` to read
/// the bf16 intrinsics. `CUDA_INCLUDE_DIR` overrides, otherwise the usual install
/// locations are probed. Missing headers are not an error — the bf16 path is
/// simply unavailable and callers fall back to fp32.
fn cuda_include_dir() -> Option<String> {
    if let Ok(d) = std::env::var("CUDA_INCLUDE_DIR")
        && std::path::Path::new(&d).join("cuda_bf16.h").is_file()
    {
        return Some(d);
    }
    let mut roots: Vec<String> = Vec::new();
    if let Ok(home) = std::env::var("CUDA_HOME") {
        roots.push(format!("{home}/include"));
    }
    if let Ok(path) = std::env::var("CUDA_PATH") {
        roots.push(format!("{path}/include"));
    }
    roots.extend(
        ["/opt/cuda/include", "/usr/local/cuda/include", "/usr/include"]
            .iter()
            .map(|s| s.to_string()),
    );
    roots
        .into_iter()
        .find(|d| std::path::Path::new(d).join("cuda_bf16.h").is_file())
}

/// Extra `-I` directories needed alongside [`cuda_include_dir`].
///
/// `<cooperative_groups.h>` includes `<cuda/std/type_traits>` from libcu++, which
/// CUDA 13 ships in a `cccl/` subtree that is *not* on NVRTC's default search
/// path — without this the cooperative kernels fail to compile with "cannot open
/// source file". Probed next to the main include dir and in the system location
/// the distro packages use.
fn extra_include_dirs(main: &str) -> Vec<String> {
    let mut out = Vec::new();
    for cand in [
        format!("{main}/cccl"),
        "/usr/include/cccl".to_string(),
        "/usr/local/cuda/include/cccl".to_string(),
    ] {
        if std::path::Path::new(&cand)
            .join("cuda/std/type_traits")
            .is_file()
        {
            out.push(cand);
            break;
        }
    }
    out
}

/// All compiled kernels, held by name. Cloneable (each `CudaFunction` is an
/// `Arc`-backed handle) and cheap to look up.
pub struct Kernels {
    funcs: std::collections::HashMap<&'static str, CudaFunction>,
    /// Whether [`COOP_NAMES`] were compiled (same include-path requirement as bf16).
    pub has_coop: bool,
    /// Whether [`BF16_NAMES`] were compiled, i.e. whether `<cuda_bf16.h>` was found.
    /// When false, bf16 activation storage is unavailable and callers stay fp32.
    pub has_bf16: bool,
    /// Whether the kernels were built with `-DSLAB_BF16`, i.e. whether the sLSTM's
    /// saved `zt`/`ot`/`h_prev` slabs are bf16 in global memory.
    ///
    /// **This is the authority the Rust side must allocate against**: the kernels
    /// index those buffers at a compile-time width, so a `BTensor` where the kernel
    /// expects `float` (or the reverse) is a silent out-of-bounds walk, not a type
    /// error. Every slab allocation reads this flag rather than the env var.
    pub slab_bf16: bool,
    /// Include flags the optional modules were compiled with, kept so the
    /// shape-specialized variants can reuse them. `None` if the headers were absent.
    coop_includes: Option<Vec<String>>,
    /// Device arch, for the same reason.
    arch: (i32, i32),
    /// Shape-specialized cooperative kernels, keyed by `(name, H, B)`.
    ///
    /// The FlashRNN model (`flashrnn.py` emits `-DFLASHRNN_HIDDEN_SIZE=...`
    /// `-DFLASHRNN_BATCH_SIZE=...` and rebuilds the module): with H and B known at
    /// compile time the reduction gets a static trip count and the batch indexing
    /// folds away. A miss costs one NVRTC compile (a few hundred ms); the backbone
    /// runs one fixed shape, so it is paid once at startup and hit forever after.
    ///
    /// `Mutex` because `Kernels` lives behind an `Arc` and this fills in lazily.
    specialized: std::sync::Mutex<HashMap<(&'static str, usize, usize, bool), Option<CudaFunction>>>,
    /// Shape-specialized fused mLSTM kernels, keyed by `(name, L, dqk, dhv, H)`.
    ///
    /// The reference (`nx-ai/mlstm_kernels`) passes every one of these as a Triton
    /// `tl.constexpr`, so the tile arithmetic folds at compile time; here they arrive
    /// as runtime `int`s and each shared-memory index costs a multiply. With them
    /// fixed, `LP`/`KP`/`VP` become literals, the tile loops get static trip counts,
    /// and the pad branches resolve away.
    mlstm_spec: std::sync::Mutex<HashMap<MlstmSpecKey, Option<CudaFunction>>>,
}

/// The shape constants a fused mLSTM kernel is built at. `kt`/`vt` are the head-dim
/// slice widths and are 0 for the two kernels that do not tile them, so a ΔC build
/// and a parallel build never share a cache slot.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MlstmSpec {
    pub l: usize,
    pub dqk: usize,
    pub dhv: usize,
    pub h: usize,
    pub threads: u32,
    /// `dqk` slice width; 0 for the kernels that do not tile it.
    pub kt: usize,
    /// `dhv` slice width; 0 for the kernels that do not tile it.
    pub vt: usize,
}

/// `(name, shape)` — what a specialized build is cached under.
pub type MlstmSpecKey = (&'static str, MlstmSpec);

impl Kernels {
    /// Compile [`SRC`] with NVRTC and load every kernel in [`NAMES`], plus — on a
    /// device that has tensor cores — the [`MMA_NAMES`] kernels.
    ///
    /// [`SRC`] is compiled **twice**, and deliberately so. `mma.sync` does not exist
    /// at NVRTC's default target arch, so the tensor-core kernels need
    /// `--gpu-architecture` pointed at the real device. But that flag is not free of
    /// consequences for the kernels around them: at a newer arch ptxas contracts
    /// FMAs differently, which shifts every scalar kernel's rounding in the last
    /// bits. That is harmless in itself — but `mlstm_fused_matches_legacy` compares
    /// weights after ONE Adam step, where the update is `lr·g/(√(g²)+ε)` and so
    /// amplifies a 1e-7 gradient wobble on a near-zero-gradient weight into ~1e-4.
    /// The test is right to be tight; the arch flag simply has no business changing
    /// the numerics of kernels that did not ask for it.
    ///
    /// So: module `base` is compiled exactly as before (default arch, no define) and
    /// every pre-existing kernel is taken from it, bit-for-bit unchanged. Module
    /// `mma` is compiled for the device arch with `MMA_TF32` defined, and *only* the
    /// two tensor-core kernels are taken from it. The second compile costs a few
    /// hundred ms once, at startup.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self, String> {
        let (major, minor) = ctx
            .compute_capability()
            .map_err(|e| format!("compute capability query failed: {e:?}"))?;
        // The mLSTM core is tensor-core only: every contraction in the parallel pair
        // is an `mma.sync`, which needs sm_80. There is no scalar twin to fall back
        // to, so this is a hard requirement rather than a dispatch.
        if major < 8 {
            return Err(format!(
                "GPU training needs tensor cores (sm_80 or later); this device is sm_{major}{minor}"
            ));
        }

        // Slab storage dtype has to be decided BEFORE the base module is compiled:
        // `slab_t` appears in the signature of `slstm_cell_step` and friends, which
        // live in this module, and every module that touches the same slab buffers
        // must agree on their width. bf16 needs `<cuda_bf16.h>`, so it is only
        // attempted when the header was found and the user has not opted out; the
        // compile is still allowed to fail, in which case we fall back to fp32 below.
        let inc = cuda_include_dir();
        let want_bf16 = super::bf16::enabled() && inc.is_some();
        let base_opts = |bf16: bool| -> CompileOptions {
            let mut options = Vec::new();
            if bf16 {
                let inc = inc.as_deref().expect("checked is_some");
                options.push("-DSLAB_BF16=1".to_string());
                options.push(format!("-I{inc}"));
                options.extend(extra_include_dirs(inc).iter().map(|d| format!("-I{d}")));
            }
            CompileOptions { options, ..Default::default() }
        };

        // Try bf16 first, fall back to fp32 if NVRTC cannot build it. `slab_bf16`
        // records what actually got compiled — the Rust side allocates to match.
        let mut slab_bf16 = want_bf16;
        let base = match want_bf16
            .then(|| compile_ptx_with_opts(SRC, base_opts(true)))
            .and_then(|r| r.ok())
        {
            Some(ptx) => ptx,
            None => {
                if want_bf16 {
                    eprintln!("bf16 slab storage unavailable (NVRTC); falling back to fp32");
                }
                slab_bf16 = false;
                compile_ptx(SRC).map_err(|e| format!("NVRTC compile failed: {e:?}"))?
            }
        };
        let base = ctx
            .load_module(base)
            .map_err(|e| format!("load_module failed: {e:?}"))?;

        let mut funcs = HashMap::new();
        for &name in NAMES {
            let f = base
                .load_function(name)
                .map_err(|e| format!("load_function {name} failed: {e:?}"))?;
            funcs.insert(name, f);
        }

        {
            let mut options = vec![
                format!("--gpu-architecture=compute_{major}{minor}"),
                "-DMMA_TF32=1".to_string(),
            ];
            // Same slab dtype as the base module: these kernels do not touch the
            // sLSTM slabs, but `slab_t` is at file scope and the two must agree so
            // that a future kernel moved between modules cannot silently disagree.
            if slab_bf16 {
                let inc = inc.as_deref().expect("slab_bf16 implies an include dir");
                options.push("-DSLAB_BF16=1".to_string());
                options.push(format!("-I{inc}"));
                options.extend(extra_include_dirs(inc).iter().map(|d| format!("-I{d}")));
            }
            let opts = CompileOptions {
                options,
                ..Default::default()
            };
            let ptx = compile_ptx_with_opts(SRC, opts)
                .map_err(|e| format!("NVRTC compile (mma) failed: {e:?}"))?;
            let module = ctx
                .load_module(ptx)
                .map_err(|e| format!("load_module (mma) failed: {e:?}"))?;
            for &name in MMA_NAMES {
                let f = module
                    .load_function(name)
                    .map_err(|e| format!("load_function {name} failed: {e:?}"))?;
                funcs.insert(name, f);
            }
        }

        // Two optional modules, compiled SEPARATELY so one cannot disable the other:
        // the bf16 kernels need only <cuda_bf16.h>, while the cooperative ones drag
        // in libcu++ through <cooperative_groups.h> and so have a strictly larger
        // include requirement. Either may be absent; neither is fatal.
        let mut has_coop = false;
        let mut has_bf16 = false;
        let mut coop_includes: Option<Vec<String>> = None;
        if let Some(inc) = cuda_include_dir() {
            let extra = extra_include_dirs(&inc);
            coop_includes = Some(
                std::iter::once(format!("-I{inc}"))
                    .chain(extra.iter().map(|d| format!("-I{d}")))
                    .collect(),
            );
            let mut compile_module = |define: &str, names: &[&'static str]| -> bool {
                let mut options = vec![
                    format!("--gpu-architecture=compute_{major}{minor}"),
                    format!("-D{define}=1"),
                    format!("-I{inc}"),
                ];
                // The cooperative sLSTM kernels write the same slabs as the base
                // module's eager ones, so they MUST be built at the same width.
                if slab_bf16 {
                    options.push("-DSLAB_BF16=1".to_string());
                }
                options.extend(extra.iter().map(|d| format!("-I{d}")));
                let opts = CompileOptions {
                    options,
                    ..Default::default()
                };
                match compile_ptx_with_opts(SRC, opts)
                    .map_err(|e| format!("{e:?}"))
                    .and_then(|ptx| ctx.load_module(ptx).map_err(|e| format!("{e:?}")))
                {
                    Ok(module) => {
                        let mut loaded = Vec::new();
                        for &name in names {
                            match module.load_function(name) {
                                Ok(f) => loaded.push((name, f)),
                                Err(e) => {
                                    eprintln!("{define}: kernel {name} unavailable: {e:?}");
                                    return false;
                                }
                            }
                        }
                        funcs.extend(loaded);
                        true
                    }
                    Err(e) => {
                        eprintln!("{define}: kernels unavailable (NVRTC): {e}");
                        false
                    }
                }
            };
            has_bf16 = compile_module("BF16_CAST", BF16_NAMES);
            has_coop = compile_module("COOP_KERNELS", COOP_NAMES);
        }

        Ok(Self {
            funcs,
            has_bf16,
            slab_bf16,
            has_coop,
            coop_includes,
            arch: (major, minor),
            specialized: std::sync::Mutex::new(HashMap::new()),
            mlstm_spec: std::sync::Mutex::new(HashMap::new()),
        })
    }

    /// A fused mLSTM kernel with `(L, dqk, dhv, H)` baked in as compile-time
    /// constants, or `None` to fall back to the generic one.
    ///
    /// Mirrors [`Kernels::specialized`]: one NVRTC compile per distinct shape, cached
    /// (failures included). The backbone runs a single shape, so this is paid once at
    /// startup. `MLSTM_NO_SPECIALIZE=1` forces the generic path for an A/B.
    ///
    /// Every specialized build is compiled for the device arch, which the `mma.sync`
    /// in the parallel pair needs anyway.
    pub fn mlstm_specialized(
        &self,
        ctx: &Arc<CudaContext>,
        name: &'static str,
        spec: MlstmSpec,
    ) -> Option<CudaFunction> {
        if std::env::var("MLSTM_NO_SPECIALIZE").is_ok() {
            return None;
        }
        let mut cache = self.mlstm_spec.lock().ok()?;
        let key = (name, spec);
        if let Some(hit) = cache.get(&key) {
            return hit.clone();
        }
        let (major, minor) = self.arch;
        let MlstmSpec { l, dqk, dhv, h, threads, kt, vt } = spec;
        let mut options = vec![
            format!("--gpu-architecture=compute_{major}{minor}"),
            "-DMMA_TF32=1".to_string(),
            "-DMLSTM_SPEC=1".to_string(),
            format!("-DMLSTM_L={l}"),
            format!("-DMLSTM_DQK={dqk}"),
            format!("-DMLSTM_DHV={dhv}"),
            format!("-DMLSTM_H={h}"),
            format!("-DMLSTM_THREADS={}", threads.max(1)),
        ];
        if kt > 0 {
            options.push(format!("-DMLSTM_KT={kt}"));
        }
        if vt > 0 {
            options.push(format!("-DMLSTM_VT={vt}"));
        }
        // Same slab width as every other module — see `load`.
        if self.slab_bf16 {
            options.push("-DSLAB_BF16=1".to_string());
            if let Some(inc) = cuda_include_dir() {
                options.push(format!("-I{inc}"));
                options.extend(extra_include_dirs(&inc).iter().map(|d| format!("-I{d}")));
            }
        }
        let built = compile_ptx_with_opts(SRC, CompileOptions {
            options,
            ..Default::default()
        })
        .map_err(|e| format!("{e:?}"))
        .and_then(|ptx| ctx.load_module(ptx).map_err(|e| format!("{e:?}")))
        .and_then(|m| m.load_function(name).map_err(|e| format!("{e:?}")));
        let f = match built {
            Ok(f) => Some(f),
            Err(e) => {
                eprintln!("mlstm specialized {name} ({spec:?}) unavailable: {e}");
                None
            }
        };
        cache.insert(key, f.clone());
        f
    }

    /// A cooperative kernel specialized to `(h, b)`, or `None` to use the generic
    /// one. Compiles on first request for a shape and caches the result — including
    /// a cached *failure*, so a shape that cannot be specialized is not retried on
    /// every call.
    ///
    /// Mirrors FlashRNN's parametric build: same source, recompiled with the shape
    /// as `-D` constants. Sequence length is not among them, there or here.
    pub fn specialized(
        &self,
        ctx: &Arc<CudaContext>,
        name: &'static str,
        h: usize,
        b: usize,
        bf16: bool,
    ) -> Option<CudaFunction> {
        if std::env::var("SLSTM_NO_SPECIALIZE").is_ok() {
            return None;
        }
        let includes = self.coop_includes.as_ref()?;
        let mut cache = self.specialized.lock().ok()?;
        let key = (name, h, b, bf16);
        if let Some(hit) = cache.get(&key) {
            return hit.clone();
        }
        let (major, minor) = self.arch;
        let mut options = vec![
            format!("--gpu-architecture=compute_{major}{minor}"),
            "-DCOOP_KERNELS=1".to_string(),
            format!("-DSLSTM_H={h}"),
            format!("-DSLSTM_B={b}"),
        ];
        if bf16 {
            options.push("-DFUSED_BF16=1".to_string());
        }
        // Must match the slab width the rest of the backend was built at, or this
        // specialized kernel would read the saved slabs at the wrong stride.
        if self.slab_bf16 {
            options.push("-DSLAB_BF16=1".to_string());
        }
        options.extend(includes.iter().cloned());
        let built = compile_ptx_with_opts(SRC, CompileOptions {
            options,
            ..Default::default()
        })
        .map_err(|e| format!("{e:?}"))
        .and_then(|ptx| ctx.load_module(ptx).map_err(|e| format!("{e:?}")))
        .and_then(|m| m.load_function(name).map_err(|e| format!("{e:?}")));
        let f = match built {
            Ok(f) => Some(f),
            Err(e) => {
                    eprintln!("specialized {name} (H={h}, B={b}, bf16={bf16}) unavailable: {e}");
                None
            }
        };
        cache.insert(key, f.clone());
        f
    }

    /// Look up a kernel by name (panics if it was not in [`NAMES`]).
    pub fn get(&self, name: &str) -> CudaFunction {
        self.funcs
            .get(name)
            .unwrap_or_else(|| panic!("unknown kernel {name}"))
            .clone()
    }
}
