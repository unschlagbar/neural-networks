//! Device-resident batched sLSTM cell, the GPU counterpart of
//! [`nn2::slstm::SLstm`](crate::nn2::slstm::SLstm).
//!
//! Same equations and the same AdamW convention (gate matrices decay, biases do
//! not), so a GPU cell built from a CPU cell's weights matches it for
//! forward → backward → step — which the parity test checks against `nn2::SLstm`.
//! The weights are stored fused rather than as the CPU's 4 gates `[rows, H]`; see
//! the note on the layout below.
//!
//! The cell also owns its **post-cell RMSNorm**, applied to the output on the way
//! out of `forward` and undone first in `backward`. `nn2` and the checkpoint format
//! still keep that norm on the surrounding block, so its γ crosses the boundary as
//! an argument (`from_parts`) and a getter (`post_norm_gamma`) — but on the GPU
//! path the block holds no norm at all, because how a cell normalizes its own
//! output is the cell's business: this one uses a plain row-wise norm, the mLSTM a
//! head-wise one. The parity tests therefore compare against `nn2::SLstm` composed
//! with an `nn2::RmsNorm`, not against the bare cell.
//!
//! Time is a serial loop; the batch is the parallel axis. **The whole recurrent
//! state `(h,c,n,m)` stays resident in `GTensor<f32>`s across the entire T-loop** — no
//! per-step host transfer.
//!
//! The four gates run **fused**: they are one `[·, 4H]` column block, so a timestep
//! is one matmul plus one elementwise pass rather than four of each. `x·Wx` for
//! **all** timesteps is a single GEMM hoisted out of the loop (it has no recurrent
//! dependency), landing in a `[B, T, 4H]` gate buffer `g`; only the recurrent half
//! `g[:, t, :] += h_{t-1}·Wh` and the elementwise recurrence stay inside it.
//!
//! That inner loop runs one of two ways, chosen by [`SLstm::fwd_loop`]:
//!
//!   * **long T** (the backbone, B=1 over a whole chunk of words) — `slstm_fused_time`
//!     runs the entire T-loop as ONE cooperative launch, with `Wh` staged in shared
//!     memory and `grid.sync()` in place of the per-step launches. See the kernel in
//!     `cuda/slstm_coop.cu`; it is where essentially all of the forward's GPU time is.
//!   * **wide B** (the encoder/decoder, one word per sequence, 120-2048 of them) —
//!     `slstm_batched_fwd` runs the whole T-loop as one cooperative launch too, but
//!     does `h_{t-1}·Wh` on the bf16 MMA unit instead of scalar FMA, so the batch fills
//!     a tensor-core tile rather than multiplying the work. See
//!     `cuda/slstm_batched.cu`. It re-reads `h` once per column block, which is what
//!     eventually loses to cuBLAS — hence the ceiling in `ops::slstm_batched_pays`.
//!   * **neither** — two launches per timestep, a cuBLAS matmul for `h_{t-1}·Wh` plus
//!     `slstm_step_fused`. The fallback, and the reference both fused paths are pinned
//!     against.
//!
//! Backward mirrors it, on the same split: `slstm_fused_time_bwd` for a long sequence,
//! `slstm_batched_bwd` for a wide batch, two launches per timestep otherwise. Either way the gate deltas go
//! back into `g` (its forward contents are dead by then) and the loop carries only the
//! BPTT channels — `dh = dg[:, t, :]·Whᵀ` — so `dx`, `dWx`, `dWh` and the bias grads
//! all fall out of three whole-sequence GEMMs plus one reduction *after* the loop.
//! Those three GEMMs read the same gate deltas, so they narrow them once between them
//! (`GemmBf16::run_slstm_backward`).
//!
//! The fused operands **are** the parameters of record: `wx [in, 4H]`,
//! `whr [H, 4H]` and `bcat [4H]` are what the optimizer steps and what the grads
//! accumulate into, so no per-forward repacking happens at all. The four `[rows, H]`
//! gate matrices `nn2::SLstm` and the checkpoints use are a *serialization* layout,
//! converted on the host only in `from_parts` / `to_nn_cell` — once per checkpoint,
//! never per step. Gate order stays z=0, i=1, f=2, o=3: the column blocks of the
//! fused `[·, 4H]`, with the input rows above the recurrent rows in `[rows, H]`.

use super::arena::{self, ParamKind, ParamSlot};
use super::block::phase;
use super::ops::{self, SlabBuf, SlstmSlabs};
use super::rms_norm::RmsNorm;
use super::{GTensor, Gpu};
use crate::gpu::arena::TrainingCache;
use crate::nn2::optim::AdamCfg;
use crate::tensor::Tensor;

/// Below this sequence length the T-loop runs step by step instead of as one
/// time-fused cooperative launch. The fused kernel amortises its `Wh` staging over
/// the whole sequence, so a handful of timesteps never earns it back — and the
/// encoder/decoder call this cell with T = a word length (<= MAX_WORD_BYTES + 1),
/// which is exactly that case. The backbone, where T is a whole chunk of words,
/// fuses.
const FUSED_MIN_T: usize = 32;

/// Above this batch the time-fused path stops paying and the per-step path wins.
///
/// `FUSED_MIN_T` is only half the predicate. `T` decides whether there are enough
/// timesteps to amortise the `Wh` staging; **`B` decides the sign of the trade**. The
/// fused kernel does the recurrent product with scalar `fmaf` + a warp shuffle, so its
/// cost is *linear* in the batch, while the per-step path hands the same product to
/// cuBLAS, whose tensor-core tiles are underfilled at small `B` and therefore *flat*
/// until they fill. Measured crossover is B ~= 32 (fwd+bwd, T=512, H=1024):
///
/// | B | fused | per-step |
/// |---|-------|----------|
/// | 1 | 3.31  | 11.50    |
/// | 8 | 29.32 | 16.62    |
///
/// Until `slstm_fused_time_batched` lands (mma-based, flat in B — see
/// `docs/slstm-batched-fused-plan.md`) a wide batch must take the per-step path.
/// `force_fused_time` still overrides this, so benchmarks can measure the losing arm.
const FUSED_MAX_B: usize = 32;

/// Whether the sLSTM's own weight gradients are held at bf16 precision
/// (`SLSTM_BF16_GRAD=1`), matching FlashRNN's `dR`/`db`.
///
/// A measurement switch, not a memory saving: the gradients stay fp32 arena windows
/// and are merely rounded. See `ops::quantize_bf16_`.
fn grad_bf16(gpu: &Gpu) -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SLSTM_BF16_GRAD").as_deref() == Ok("1")) && gpu.kernels.has_bf16
}


/// The time-fused T-loop (`slstm_fused_time` / `_bwd`): the whole forward or
/// backward sequence as ONE cooperative launch instead of two launches per
/// timestep. **On by default**; `SLSTM_NO_FUSED_TIME=1` forces the per-step path,
/// which is the A/B baseline and the fallback if a driver ever mis-schedules a
/// cooperative grid.
///
/// Measured at the backbone's shape (B=1, T=2047, H=512): forward 9.35 -> 6.50ms,
/// backward 10.24 -> 7.64ms, i.e. 1.38x on the layer and 267 -> 217ms on the
/// backbone's eight sLSTM halves. Parity with the per-step path (output, dx, every
/// dW and db) is pinned by `slstm_fused_time_matches_per_step`.
///
/// It declines on its own when the shape does not fit — notably at H >= 1024,
/// where the grid would need more blocks than the device has SMs and a cooperative
/// launch requires the whole grid to be co-resident — so enabling it by default
/// never removes a working path, it only takes the faster one where it exists.
fn fused_time_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SLSTM_NO_FUSED_TIME").is_err())
}

/// Narrowest batch the batch-parallel fused path takes. The mma M axis is 16 rows
/// wide, so below a couple of tiles the contraction is mostly padding and the
/// per-step cuBLAS call is doing the same thing with less setup.
const SB_MIN_B: usize = 32;

/// The batch-parallel fused T-loop (`slstm_batched_fwd`). **On by default**;
/// `SLSTM_NO_BATCHED=1` forces the per-step path, which is the A/B baseline.
fn batched_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SLSTM_NO_BATCHED").is_err())
}

pub struct SLstm {
    input: usize,
    hidden: usize,

    /// Post-cell RMSNorm, applied to the cell's output before it leaves `forward`.
    ///
    /// It belongs here rather than in the surrounding block: normalizing its own
    /// output is the cell's business, and the two cells do it differently — this one
    /// with a plain row-wise norm, the mLSTM with its head-wise `headnorm`. Neither
    /// shape is something the block should have to know about.
    post_norm: RmsNorm,

    // The parameters of record, in the fused layout the GEMMs consume directly:
    // gates z=0, i=1, f=2, o=3 occupy the four column blocks of the `4H` axis.
    // The optimizer steps these, the grads accumulate into them, and nothing is
    // repacked per forward — the `[rows, H]` gate matrices exist only at the
    // checkpoint boundary (`from_parts` / `to_nn_cell`).
    pub wx: GTensor<f32>,   // [in, 4H]   input half
    pub whr: GTensor<f32>,  // [H, 4H]    recurrent half
    pub bcat: GTensor<f32>, // [4H]
    dwx: GTensor<f32>,
    dwhr: GTensor<f32>,
    dbcat: GTensor<f32>,
    mwx: GTensor<f32>,
    vwx: GTensor<f32>,
    mwhr: GTensor<f32>,
    vwhr: GTensor<f32>,
    mbcat: GTensor<f32>,
    vbcat: GTensor<f32>,

    /// Per-instance override of the time-fused path. `None` follows the global
    /// default (see [`fused_time_enabled`]); `Some(false)` pins this cell to the
    /// per-step loop and `Some(true)` to the fused one. The env flag is read through
    /// a `OnceLock`, so a test that needs BOTH paths in one process sets this — and
    /// it must set it on both cells, since the default alone no longer distinguishes
    /// them.
    pub force_fused_time: Option<bool>,

    /// Per-instance override of the batch-parallel fused path (`slstm_batched_fwd`),
    /// the mma-based twin of the time-fused one for wide batches. `None` follows the
    /// global default (see [`batched_enabled`]).
    pub force_batched: Option<bool>,

    /// Continue the previous call's recurrence instead of starting from zero.
    ///
    /// Off by default: a `forward` is a whole sequence, and its state is private to
    /// the call. Set for a **chunked** sweep, where one sequence is split across
    /// several calls and the state has to cross the chunk borders — forward carries
    /// `h/c/n/m`, backward carries the BPTT channels (in reverse chunk order). See
    /// [`set_carry`](Self::set_carry).
    carry: bool,

    // Recurrent state carried across timesteps within one call, [B, H].
    h_state: GTensor<f32>,
    c_state: GTensor<f32>,
    n_state: GTensor<f32>,
    m_state: GTensor<f32>,
    /// Contiguous `[B, 4H]` scratch for the current timestep's recurrent gate half
    /// (`h_{t-1}·Wh`). It exists so that GEMM stays dense at any batch size — see
    /// `slstm_step_fused` in `kernels.rs`.
    gh: GTensor<f32>,
    /// The backward twin of [`gh`](Self::gh): this timestep's gate deltas, `[B, 4H]`,
    /// at the slab width. `slstm_step_fused_bwd` writes it, `dh = dg·Whᵀ` reads it —
    /// nothing else — so it is narrow for the same reason [`h_narrow`](Self::h_narrow)
    /// is, and that GEMM gets tensor-core operands with no cast launch of its own.
    dgh: ops::SlabBuf,
    /// The left operand of the per-timestep recurrent GEMM: `h_{t-1}` at the slab
    /// width, `[B, H]`.
    ///
    /// `slstm_step_fused` writes it alongside the fp32 `h_state`, so on the bf16 path
    /// that GEMM gets tensor-core operands without a narrowing launch of its own —
    /// and it is worth having: at the encoder's shape (`[512,256]x[256,1024]`) the
    /// fp32 SIMT kernel cuBLAS picks runs at 22 TFLOP/s, 39% of this card's fp32 peak
    /// and a small fraction of what the same matmul does in bf16.
    h_narrow: ops::SlabBuf,
    // BPTT channels, [B, H].
    dh_bptt: GTensor<f32>,
    dc_bptt: GTensor<f32>,
    dn_bptt: GTensor<f32>,

    // Handed from forward to backward: the gate buffer [B, T, 4H], the saved
    // [B, T, H] slabs, and the flattened input [B·T, in] (needed for dWx).
    //
    // These are *reused* across calls rather than reallocated, and `out` / `dy_buf`
    // exist for the same reason: a stack runs the same handful of shapes over and
    // over (one rectangle per length bucket, one chunk length on the backbone), so
    // `take_uninit` turns what would be an allocate/free pair per call per buffer
    // into a pointer move.
    g: Option<GTensor<f32>>,
    slabs: Option<SlstmSlabs>,
    /// The forward's input at the width the GEMMs consume it: `[B·T, in]`.
    ///
    /// Not a copy of `x` — a *narrowing* of it. `x` itself cannot be held (the caller
    /// returns its buffer to the pool the moment forward returns, and `dWx = xᵀ·dg`
    /// needs it again in backward), and both GEMMs that read it want bf16 anyway. So
    /// the one cast the forward GEMM was doing per call is hoisted here and the
    /// backward GEMM reuses its result — one narrowing instead of a copy plus two.
    x_saved: Option<SlabBuf>,
    out_buf: Option<GTensor<f32>>,
    /// Forward caches of earlier chunks of a chunked sweep, oldest first.
    ///
    /// The buffers above are reused call to call, which is exactly what a chunked
    /// sweep cannot have: chunk c+1's forward would overwrite what chunk c's backward
    /// reads. So each chunk's `(g, slabs, x_saved)` is moved aside here when the next
    /// chunk's forward takes fresh buffers, and backward pops them right to left.
    /// Only what backward *reads* moves; `out_buf`/`dy_buf` are written through.
    ///
    /// Empty on the unchunked path, where the reuse above is untouched.
    chunk_saved: Vec<SlstmChunk>,
    /// Host staging for the chunk caches above, when the surrounding block opted in.
    ///
    /// Only the *set-aside* chunks ride: the live `g`/`slabs`/`x_saved` slots are
    /// about to be written again, so parking them would buy nothing.
    park: Option<super::offload::HostPark>,
    /// Where the post-cell norm's backward lands, so the loop reads a buffer this
    /// cell owns and reuses instead of allocating one per call.
    dy_buf: Option<GTensor<f32>>,
    /// Scratch for widening a bf16 `h_prev` slab back to fp32 for the `dWh` GEMM.
    /// Only allocated when the kernels and the whole-sequence GEMMs were built at
    /// different widths — normally that GEMM takes the slab as it stands.
    h_prev_f32: Option<GTensor<f32>>,
    batch: usize,
    /// bf16 staging for the **whole-sequence** GEMMs (`x·Wx` forward; `dg·Wxᵀ`,
    /// `xᵀ·dg` and `h_prevᵀ·dg` backward). Those run once per call over `[N, ·]`,
    /// exactly like a `Linear`'s, and were the last fp32 SIMT matmuls left on the
    /// profile. The recurrent `Wh` GEMM is deliberately NOT included: it runs per
    /// timestep inside the loop, where the staging would cost a cast per step, and the
    /// fused kernel owns that path anyway.
    gemm_x: ops::GemmBf16,
    gemm_dx: ops::GemmBf16,
    /// Weight cache for the **per-timestep** recurrent GEMMs `h_{t-1}·Whr` and
    /// `dg_t·Whrᵀ`. Separate from the two above only because `GemmBf16` holds one
    /// cached weight and this one is `whr`, not `wx`; both GEMMs read `whr` as their
    /// right operand, so one narrowing per optimizer step serves the whole sweep. The
    /// left operand ([`h_narrow`](Self::h_narrow) forward, [`dgh`](Self::dgh)
    /// backward) arrives already narrowed, so a step of either loop is still one
    /// launch per GEMM.
    gemm_h: ops::GemmBf16,
    /// Whether the whole-sequence GEMMs take the bf16 path. Pinned at construction so
    /// forward and backward cannot disagree.
    bf16: bool,
}

/// One chunk's forward cache, set aside so a later chunk's forward can take fresh
/// buffers without destroying it. See [`SLstm::chunk_saved`].
///
/// `Resident` holds the device buffers directly. `Parked` means the same set has been
/// handed to the [`HostPark`](super::offload::HostPark) and lives in host memory; the
/// park's generation stack is popped in the same right-to-left order backward unwinds
/// the chunks, so the two stay in step without storing an index here.
enum SlstmChunk {
    Resident {
        g: GTensor<f32>,
        slabs: SlstmSlabs,
        x_saved: SlabBuf,
    },
    Parked,
}

/// The eleven buffers of one chunk cache, in the fixed order `evict`/`restore` share.
///
/// Written out rather than derived so the two directions cannot drift: a mismatch here
/// is a silent shape/width swap, not a compile error.
fn park_order(g: GTensor<f32>, slabs: SlstmSlabs, x_saved: SlabBuf) -> Vec<super::offload::Parked> {
    use super::offload::Parked;
    vec![
        Parked::from(g),
        Parked::from(x_saved),
        Parked::from(slabs.c_entry),
        Parked::from(slabs.n_entry),
        Parked::from(slabs.i_prime),
        Parked::from(slabs.f_prime),
        Parked::from(slabs.c),
        Parked::from(slabs.n),
        Parked::from(slabs.zt),
        Parked::from(slabs.ot),
        Parked::from(slabs.h_prev),
    ]
}

/// Rebuild a chunk cache from `park_order`'s output, in the same order.
fn park_unorder(p: Vec<super::offload::Parked>) -> (GTensor<f32>, SlstmSlabs, SlabBuf) {
    use super::offload::Parked;
    assert_eq!(p.len(), 11, "slstm park: restored buffer count");
    let mut it = p.into_iter();
    // The stabilizer group is fp32 by construction (`gpu::bf16`), so a bf16 buffer
    // arriving here is a park/restore order mismatch, not a precision choice.
    fn wide(it: &mut std::vec::IntoIter<super::offload::Parked>, what: &str) -> GTensor<f32> {
        match it.next().expect("slstm park: short restore") {
            Parked::F32(t) => t,
            Parked::Bf16(_) => panic!("slstm park: {what} came back bf16"),
        }
    }
    let g = wide(&mut it, "g");
    let slab = |it: &mut std::vec::IntoIter<super::offload::Parked>| {
        SlabBuf::from(it.next().expect("slstm park: short restore"))
    };
    let x_saved = slab(&mut it);
    let (c_entry, n_entry) = (slab(&mut it), slab(&mut it));
    let (i_prime, f_prime) = (slab(&mut it), slab(&mut it));
    let (c, n) = (slab(&mut it), slab(&mut it));
    let slabs = SlstmSlabs {
        c_entry,
        n_entry,
        i_prime,
        f_prime,
        c,
        n,
        zt: slab(&mut it),
        ot: slab(&mut it),
        h_prev: slab(&mut it),
    };
    (g, slabs, x_saved)
}

/// Keep `slot`'s buffer when it already has the wanted shape, else allocate a
/// fresh (uninitialised) one — so a stack that repeats a handful of shapes keeps
/// the allocator off the hot path.
fn take_uninit(gpu: &Gpu, slot: Option<GTensor<f32>>, dims: &[usize]) -> GTensor<f32> {
    match slot {
        Some(t) if t.dims() == dims => t,
        _ => GTensor::uninit(gpu, dims),
    }
}

/// [`take_uninit`] for the narrowed input: keep `slot` when it is wide enough, else
/// allocate at the width the GEMMs will read it at. `bf16` is the cell's GEMM path,
/// not the slab flag — this buffer feeds cuBLAS, not the fused kernels.
fn fit_saved(gpu: &Gpu, slot: Option<SlabBuf>, bf16: bool, dims: &[usize]) -> SlabBuf {
    let n: usize = dims.iter().product();
    match slot {
        Some(mut s) if matches!(s, SlabBuf::Bf16(_)) == bf16 && s.capacity() >= n => {
            s.shrink_to(dims);
            s
        }
        _ if bf16 => SlabBuf::Bf16(super::GTensor::uninit(gpu, dims)),
        _ => SlabBuf::F32(GTensor::uninit(gpu, dims)),
    }
}

/// [`take_uninit`] for the whole saved set. fp32 for the stabilizer-carrying slabs,
/// kernel-matched width for the plain activations — see `SlstmSlabs` and `gpu::bf16`.
fn fit_slabs(gpu: &Gpu, slot: Option<SlstmSlabs>, dims: &[usize]) -> SlstmSlabs {
    if let Some(s) = slot
        && s.c.dims() == dims
    {
        return s;
    }
    let slab = || SlabBuf::new(gpu, dims);
    // The stabilizer group follows `state_bf16`, a different switch from the plain
    // slabs' `slab_bf16` — see `SlstmSlabs` and `gpu::bf16`.
    let st = || SlabBuf::new_width(gpu, dims, gpu.kernels.state_bf16);
    // `c_entry`/`n_entry` hold one timestep, not the sweep: [B, H], not [B, T, H].
    let entry = || SlabBuf::new_width(gpu, &[dims[0], dims[2]], gpu.kernels.state_bf16);
    SlstmSlabs {
        c_entry: entry(),
        n_entry: entry(),
        i_prime: st(),
        f_prime: st(),
        c: st(),
        n: st(),
        zt: slab(),
        ot: slab(),
        h_prev: slab(),
    }
}

/// Reuse `t`'s device buffer when the shape matches, zeroing it in place; else
/// (re)allocate a zeroed buffer. For state / BPTT channels that must start at 0.
fn fit_zeros(gpu: &Gpu, t: &mut GTensor<f32>, dims: &[usize]) {
    if t.dims() == dims {
        t.zero_(gpu);
    } else {
        *t = GTensor::zeros(gpu, dims);
    }
}

/// Reuse `t`'s device buffer when the shape matches (leaving its contents); else
/// (re)allocate uninitialised. For outputs a kernel/GEMM overwrites in full.
fn fit_uninit(gpu: &Gpu, t: &mut GTensor<f32>, dims: &[usize]) {
    if t.dims() != dims {
        *t = GTensor::uninit(gpu, dims);
    }
}

impl SLstm {
    /// Build from a CPU cell's host weights (gate order z, i, f, o). The `w{*}`
    /// are `[rows, H]` and the `b{*}` are `[H]`; they are uploaded to the device.
    ///
    /// `post_gamma` is the post-cell norm's scale `[H]`; `None` starts it at γ=1,
    /// which is what a freshly-built cell wants. A checkpoint passes the saved one.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        gpu: &Gpu,
        input: usize,
        hidden: usize,
        wz: &Tensor,
        wi: &Tensor,
        wf: &Tensor,
        wo: &Tensor,
        bz: &Tensor,
        bi: &Tensor,
        bf: &Tensor,
        bo: &Tensor,
        post_gamma: Option<&Tensor>,
    ) -> Self {
        let (h, h4) = (hidden, 4 * hidden);
        let rows = input + h;
        let w = [wz, wi, wf, wo];
        let b = [bz, bi, bf, bo];
        for (g, t) in w.iter().enumerate() {
            assert_eq!(t.dims(), [rows, h], "SLstm::from_parts — gate {g} weight");
            assert_eq!(b[g].dims(), [h], "SLstm::from_parts — gate {g} bias");
        }

        // Host-side pack into the fused layout: gate `g` becomes column block
        // `g·H..(g+1)·H`, with `[rows, H]`'s top `input` rows going to `wx` and its
        // remaining `H` rows to `whr`. This is the only place the two layouts meet.
        let mut wx = vec![0.0; input * h4];
        let mut whr = vec![0.0; h * h4];
        let mut bcat = vec![0.0; h4];
        for (g, gw) in w.iter().enumerate() {
            let src = gw.data.as_slice();
            for r in 0..rows {
                let (dst, dr) = if r < input {
                    (&mut wx, r)
                } else {
                    (&mut whr, r - input)
                };
                let (from, to) = (r * h, dr * h4 + g * h);
                dst[to..to + h].copy_from_slice(&src[from..from + h]);
            }
            bcat[g * h..(g + 1) * h].copy_from_slice(&b[g].data);
        }

        let up = |d: Vec<f32>, dims: &[usize]| GTensor::from_host(gpu, &Tensor::new(dims, d));
        Self {
            input,
            hidden,
            post_norm: match post_gamma {
                Some(g) => {
                    assert_eq!(g.dims(), [h], "SLstm::from_parts — post_norm gamma");
                    RmsNorm::from_parts(gpu, g)
                }
                None => RmsNorm::new(gpu, h),
            },
            wx: up(wx, &[input, h4]),
            whr: up(whr, &[h, h4]),
            bcat: up(bcat, &[h4]),
            dwx: GTensor::zeros(gpu, &[input, h4]),
            dwhr: GTensor::zeros(gpu, &[h, h4]),
            dbcat: GTensor::zeros(gpu, &[h4]),
            mwx: GTensor::zeros(gpu, &[input, h4]),
            vwx: GTensor::zeros(gpu, &[input, h4]),
            mwhr: GTensor::zeros(gpu, &[h, h4]),
            vwhr: GTensor::zeros(gpu, &[h, h4]),
            mbcat: GTensor::zeros(gpu, &[h4]),
            vbcat: GTensor::zeros(gpu, &[h4]),
            force_fused_time: None,
            force_batched: None,
            carry: false,
            h_state: GTensor::zeros(gpu, &[0, 0]),
            c_state: GTensor::zeros(gpu, &[0, 0]),
            n_state: GTensor::zeros(gpu, &[0, 0]),
            m_state: GTensor::zeros(gpu, &[0, 0]),
            gh: GTensor::zeros(gpu, &[0, 0]),
            dgh: ops::SlabBuf::new(gpu, &[0, 0]),
            h_narrow: ops::SlabBuf::new(gpu, &[0, 0]),
            dh_bptt: GTensor::zeros(gpu, &[0, 0]),
            dc_bptt: GTensor::zeros(gpu, &[0, 0]),
            dn_bptt: GTensor::zeros(gpu, &[0, 0]),
            g: None,
            slabs: None,
            x_saved: None,
            out_buf: None,
            dy_buf: None,
            h_prev_f32: None,
            chunk_saved: Vec::new(),
            park: None,
            batch: 0,
            gemm_x: ops::GemmBf16::new(),
            gemm_dx: ops::GemmBf16::new(),
            gemm_h: ops::GemmBf16::new(),
            bf16: ops::gemm_bf16_enabled(gpu),
        }
    }

    /// Freshly-initialised cell, matching `nn2::SLstm::new`'s init exactly
    /// (including the +4.5 forget-gate bias). Post-norm starts at γ=1.
    pub fn new_rand(gpu: &Gpu, input: usize, hidden: usize) -> Self {
        Self::from_cpu(gpu, &crate::nn2::SLstm::new(input, hidden), None)
    }

    /// Upload a CPU cell (weights are copied; grads/moments start at zero).
    ///
    /// The post-cell norm's γ comes from the surrounding CPU *block* — `nn2` still
    /// keeps it there — so the caller passes it in; `None` starts it at 1.
    pub fn from_cpu(gpu: &Gpu, cpu: &crate::nn2::SLstm, post_gamma: Option<&Tensor>) -> Self {
        Self::from_parts(
            gpu,
            cpu.input_size,
            cpu.hidden_size,
            &cpu.wz,
            &cpu.wi,
            &cpu.wf,
            &cpu.wo,
            &cpu.bz,
            &cpu.bi,
            &cpu.bf,
            &cpu.bo,
            post_gamma,
        )
    }

    #[inline]
    pub fn input_size(&self) -> usize {
        self.input
    }
    #[inline]
    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    /// Export this cell into the CPU `nn::SLSTMLayer` format (weights only; the
    /// `h_init`/`c_init` the CPU format carries are always zero here). Used to
    /// write a `HIER` checkpoint from a GPU model.
    pub fn to_nn_cell(&self, gpu: &Gpu) -> crate::nn::slstm::SLSTMLayer {
        let (h, inp) = (self.hidden, self.input);
        let [wz, wi, wf, wo] = self.gate_matrices(gpu);
        let b = self.bcat.to_host(gpu).data;
        let bias = |g: usize| b[g * h..(g + 1) * h].to_vec().into_boxed_slice();
        crate::nn::slstm::SLSTMLayer::from_loaded(
            inp,
            h,
            wz,
            wi,
            wf,
            wo,
            bias(0),
            bias(1),
            bias(2),
            bias(3),
            vec![0.0; h].into(),
            vec![0.0; h].into(),
        )
    }

    /// Gate `g`'s weights as the `[rows, H]` matrix the CPU cell holds, downloaded
    /// and unpacked. For parity tests, which compare per gate.
    #[cfg(test)]
    pub(crate) fn gate_w(&self, gpu: &Gpu, g: usize) -> Vec<f32> {
        let m = &self.gate_matrices(gpu)[g];
        m.as_slice().to_vec()
    }

    /// Gate `g`'s bias `[H]`, sliced out of the fused `bcat`. Test companion to
    /// [`gate_w`](Self::gate_w).
    #[cfg(test)]
    fn gate_b(&self, gpu: &Gpu, g: usize) -> Vec<f32> {
        let h = self.hidden;
        self.bcat.to_host(gpu).data[g * h..(g + 1) * h].to_vec()
    }

    /// Gate `g`'s weight gradient, unpacked from the fused `dwx`/`dwhr` into the
    /// `[rows, H]` layout the CPU cell's `dw*` use. Test-only.
    #[cfg(test)]
    fn gate_dw(&self, gpu: &Gpu, g: usize) -> Vec<f32> {
        let (h, h4, inp) = (self.hidden, 4 * self.hidden, self.input);
        let rows = inp + h;
        let dwx = self.dwx.to_host(gpu).data;
        let dwhr = self.dwhr.to_host(gpu).data;
        let mut out = vec![0.0; rows * h];
        for r in 0..rows {
            let (src, sr) = if r < inp { (&dwx, r) } else { (&dwhr, r - inp) };
            let from = sr * h4 + g * h;
            out[r * h..(r + 1) * h].copy_from_slice(&src[from..from + h]);
        }
        out
    }

    /// Gate `g`'s bias gradient `[H]`. Test companion to [`gate_dw`](Self::gate_dw).
    #[cfg(test)]
    fn gate_db(&self, gpu: &Gpu, g: usize) -> Vec<f32> {
        let h = self.hidden;
        self.dbcat.to_host(gpu).data[g * h..(g + 1) * h].to_vec()
    }

    /// Unpack the fused `wx`/`whr` back into the four `[rows, H]` gate matrices the
    /// CPU cells and checkpoints use — the inverse of the pack in
    /// [`from_parts`](Self::from_parts). Host-side and once per checkpoint.
    fn gate_matrices(&self, gpu: &Gpu) -> [iron_oxide::collections::Matrix; 4] {
        let (h, h4, inp) = (self.hidden, 4 * self.hidden, self.input);
        let rows = inp + h;
        let wx = self.wx.to_host(gpu).data;
        let whr = self.whr.to_host(gpu).data;
        std::array::from_fn(|g| {
            let mut m = vec![0.0; rows * h];
            for r in 0..rows {
                let (src, sr) = if r < inp { (&wx, r) } else { (&whr, r - inp) };
                let from = sr * h4 + g * h;
                m[r * h..(r + 1) * h].copy_from_slice(&src[from..from + h]);
            }
            iron_oxide::collections::Matrix::from_vec(m, rows, h)
        })
    }

    /// Rebuild a GPU cell from a CPU `nn::SLSTMLayer` (inverse of `to_nn_cell`).
    /// `post_gamma` is the enclosing `SLSTMBlock`'s `post_cell_norm.gamma`.
    pub fn from_nn_cell(
        gpu: &Gpu,
        c: &crate::nn::slstm::SLSTMLayer,
        post_gamma: Option<&Tensor>,
    ) -> Self {
        use super::{tensor_from_matrix as m, tensor_from_slice as v};
        Self::from_parts(
            gpu,
            c.input_size,
            c.hidden_size,
            &m(&c.wz),
            &m(&c.wi),
            &m(&c.wf),
            &m(&c.wo),
            &v(&c.bz),
            &v(&c.bi),
            &v(&c.bf),
            &v(&c.bo),
            post_gamma,
        )
    }

    /// The post-cell norm's scale, for exporting the enclosing block to a CPU
    /// `SLSTMBlock` (which still keeps the norm at block level).
    pub fn post_norm_gamma(&self) -> &GTensor<f32> {
        &self.post_norm.gamma
    }

    /// Move the live forward cache into [`chunk_saved`](Self::chunk_saved), so the
    /// call about to run can take fresh buffers without destroying it.
    ///
    /// A chunked sweep forwards every chunk before unwinding any, so the previous
    /// chunk's `(g, slabs, x_saved)` is still owed a backward. Unchunked there is
    /// nothing to preserve and this is never called.
    fn set_aside_chunk(&mut self, gpu: &Gpu) {
        let (Some(g), Some(slabs), Some(x_saved)) =
            (self.g.take(), self.slabs.take(), self.x_saved.take())
        else {
            return; // first chunk of the sweep: nothing forwarded yet
        };
        // With offload on, the cache goes to the host instead of staying resident. The
        // device tensors are handed to the park, which holds them until its D2H has
        // landed — the next chunk's eviction releases them, so the copy overlaps that
        // chunk's compute.
        match &mut self.park {
            Some(park) => {
                park.evict(gpu, park_order(g, slabs, x_saved));
                self.chunk_saved.push(SlstmChunk::Parked);
            }
            None => self
                .chunk_saved
                .push(SlstmChunk::Resident { g, slabs, x_saved }),
        }
    }

    /// Forward over a whole `[B, T, in]` sequence into `y` `[B, T, H]`.
    ///
    /// The recurrence starts from zero unless [`set_carry`](Self::set_carry) says this
    /// call continues the previous one's sequence, and the whole state stays
    /// device-resident across the T-loop either way.
    pub fn forward(
        &mut self,
        gpu: &Gpu,
        x: &GTensor<f32>,
        y: &mut GTensor<f32>,
        cache: &mut TrainingCache,
    ) {
        // Release the previous eviction before allocating anything here: freeing
        // returns memory to the CUDA allocator, which must not hand it back while a
        // copy is still reading it. See `InFlight::release`.
        if let Some(park) = &self.park {
            park.release_previous();
        }
        assert_eq!(x.rank, 3, "SLstm::forward expects [B, T, in]");
        let (b, t, inp) = (x.shape[0], x.shape[1], x.shape[2]);
        assert_eq!(inp, self.input, "SLstm::forward — input width mismatch");
        assert_eq!(
            y.dims(),
            [b, t, self.hidden],
            "SLstm::forward — output shape"
        );
        let h = self.hidden;
        let h4 = 4 * h;
        let n = b * t;
        self.batch = b;

        // `wx`/`whr`/`bcat` are the parameters themselves — already in the layout the
        // GEMMs below want, so there is nothing to pack here.

        // Whether this call continues the sequence the previous one left off (see
        // `set_carry`), which is what makes a chunked sweep reproduce the unchunked
        // recurrence exactly instead of resetting at every chunk border. A shape change
        // ends the carry regardless: a carried state is only meaningful for the batch
        // it was produced at.
        //
        // Without a carry the buffers below are *not* zeroed — the kernels take `carry`
        // and start from literal zeros, which is the same arithmetic without four
        // memsets and, on the per-step path, without the t=0 GEMM whose operand they
        // would have been.
        let carry = self.carry && self.h_state.dims() == [b, h];
        for s in [
            &mut self.h_state,
            &mut self.c_state,
            &mut self.n_state,
            &mut self.m_state,
        ] {
            fit_uninit(gpu, s, &[b, h]);
        }

        if carry {
            self.set_aside_chunk(gpu);
        }

        // Narrow `x` into the cell's own `[N, in]` buffer. This is the only place the
        // input is read at full width: the forward GEMM below and backward's
        // `dWx = xᵀ·dg` both consume the narrowed copy, and `x` itself is gone by then
        // (the caller returns its buffer to the pool the moment this returns).
        //
        // `store` takes the leading `N·in` elements, which is what `x` holds — a
        // pooled buffer may be *larger* than [B, T, in] (`Buf`/`Pool` reuse by
        // capacity), and copying its whole allocation would move capacity, not content.
        let mut x_flat = fit_saved(gpu, self.x_saved.take(), self.bf16, &[n, inp]);
        phase::timed(gpu, phase::Bucket::SlstmCopyFwd, || x_flat.store(gpu, x));

        // The input half of every gate pre-activation, for all timesteps at once —
        // it has no recurrent dependency, so it is one GEMM outside the loop.
        //
        // One buffer, two views: the GEMM wants [N, 4H], the time loop wants
        // [B, T, 4H]. `reshaped` is metadata-only, so the allocation is untouched.
        let mut g = take_uninit(gpu, self.g.take(), &[b, t, h4]).reshaped(&[n, h4]);
        let (gemm_x, wx_w) = (&mut self.gemm_x, &self.wx);
        phase::timed(gpu, phase::Bucket::SlstmGemmFwd, || match &x_flat {
            SlabBuf::Bf16(xb) => gemm_x.run_staged_lhs(gpu, ops::MmForm::Nn, xb, wx_w, &mut g, 0.0),
            SlabBuf::F32(xf) => ops::matmul_nn_into(gpu, xf, wx_w, &mut g, 0.0),
        });
        let mut g = g.reshaped(&[b, t, h4]);

        let mut slabs = fit_slabs(gpu, self.slabs.take(), &[b, t, h]);
        let mut out = take_uninit(gpu, self.out_buf.take(), &[b, t, h]);

        phase::timed(gpu, phase::Bucket::SlstmLoopFwd, || {
            self.fwd_loop(gpu, &mut g, &mut slabs, &mut out, t, carry);
        });

        // The loop writes `out`, and the result reaches the caller's buffer through
        // the post-cell norm — which is also what moves it, so there is no separate
        // copy.
        //
        // The norm is position-wise and folds the leading axes itself, so both sides
        // stay [B, T, H]. `y` was asserted that shape on entry, so the write covers
        // exactly the caller's buffer.
        phase::timed(gpu, phase::Bucket::SlstmCopyFwd, || {
            self.post_norm.forward(gpu, &out, y);
        });
        self.g = Some(g);
        self.slabs = Some(slabs);
        self.x_saved = Some(x_flat);
        self.out_buf = Some(out);
    }

    /// The forward time loop: one cooperative launch when T is long enough, else
    /// step by step.
    ///
    /// The split exists because a long loop at batch 1 is pure launch latency: a
    /// timestep there is a `[1,H]x[H,4H]` matvec — a couple of us of GPU work — while
    /// *submitting* it costs the host far more, so the card idles on the driver and no
    /// faster card can help. `slstm_fused_time` removes the submissions entirely. A
    /// short T (a word, in the encoder/decoder) has too few of them to be worth the
    /// kernel's `Wh` staging, and its batch is wide enough that the per-step matmul is
    /// real work rather than latency.
    fn fwd_loop(
        &mut self,
        gpu: &Gpu,
        g: &mut GTensor<f32>,
        slabs: &mut SlstmSlabs,
        out: &mut GTensor<f32>,
        t: usize,
        carry: bool,
    ) {
        // The kernel declines by returning false when the shape does not fit or the
        // shape-specialized build is unavailable, leaving the per-step path intact.
        let b = self.h_state.rows();
        if self.fuses_at(b, t)
            && ops::slstm_fused_time(
                gpu,
                &self.whr,
                g,
                &self.bcat,
                &mut self.c_state,
                &mut self.n_state,
                &mut self.m_state,
                &mut self.h_state,
                slabs,
                out,
                t,
                carry,
            )
        {
            return;
        }
        if self.batches_at(gpu, b)
            && ops::slstm_batched_fwd(
                gpu,
                &self.whr,
                g,
                &self.bcat,
                &mut self.c_state,
                &mut self.n_state,
                &mut self.m_state,
                &mut self.h_state,
                slabs,
                out,
                t,
                carry,
            )
        {
            return;
        }
        self.fwd_steps(gpu, g, slabs, out, t, carry);
    }

    /// The loop body, one timestep at a time.
    fn fwd_steps(
        &mut self,
        gpu: &Gpu,
        g: &mut GTensor<f32>,
        slabs: &mut SlstmSlabs,
        out: &mut GTensor<f32>,
        t: usize,
        carry: bool,
    ) {
        // The scratch only this path needs: the contiguous `[B, 4H]` gate half and the
        // narrowed `h` its GEMM reads. Fitted here rather than in `forward` because
        // which path runs is not known until the fused one has been *tried* — it can
        // decline at launch, not just at geometry.
        let (b, h) = (self.h_state.rows(), self.h_state.cols());
        fit_uninit(gpu, &mut self.gh, &[b, 4 * h]);
        self.h_narrow.fit(gpu, &[b, h]);
        if carry {
            // `slstm_step_fused` refreshes `h_narrow` every step, so only a carried
            // starting value has to be staged; without a carry the loop skips step 0's
            // GEMM outright and never reads it.
            let (h_narrow, h_state) = (&mut self.h_narrow, &self.h_state);
            h_narrow.store(gpu, h_state);
        }
        for step in 0..t {
            // Recurrent half of the gates (one dense GEMM into the contiguous
            // scratch), then the elementwise recurrence: two launches per timestep.
            // The GEMM's left operand is the narrowed `h` the previous step's kernel
            // wrote, so the bf16 path adds no third launch.
            //
            // At a sequence start it is skipped outright: `h_{-1}` is zero, so the
            // product is, and the kernel substitutes that. At the encoder's shape that
            // is a `[512,256]x[256,1024]` GEMM saved out of every group's T of them.
            let first = step == 0 && !carry;
            if !first {
                let Self {
                    gemm_h,
                    h_narrow,
                    whr,
                    gh,
                    ..
                } = self;
                match h_narrow {
                    ops::SlabBuf::Bf16(h) => {
                        gemm_h.run_staged_lhs(gpu, ops::MmForm::Nn, h, whr, gh, 0.0)
                    }
                    ops::SlabBuf::F32(h) => ops::matmul_nn_into(gpu, h, whr, gh, 0.0),
                }
            }
            ops::slstm_step_fused(
                gpu,
                g,
                &self.gh,
                &self.bcat,
                &mut self.c_state,
                &mut self.n_state,
                &mut self.m_state,
                &mut self.h_state,
                &mut self.h_narrow,
                slabs,
                out,
                step,
                first,
            );
        }
    }

    /// Forward into a freshly allocated `[B, T, H]` — the by-value companion to
    /// [`forward`](Self::forward), used by tests and one-shot call sites.
    pub fn forward_alloc(&mut self, gpu: &Gpu, x: &GTensor<f32>) -> GTensor<f32> {
        let mut y: GTensor<f32> = GTensor::uninit(gpu, &[x.shape[0], x.shape[1], self.hidden]);
        self.forward(gpu, x, &mut y, &mut TrainingCache::new());
        y
    }

    /// Backward into a freshly allocated `dx` `[B, T, in]`. `y` is the forward output,
    /// as for [`backward`](Self::backward).
    pub fn backward_alloc(
        &mut self,
        gpu: &Gpu,
        y: &GTensor<f32>,
        dy: &GTensor<f32>,
    ) -> GTensor<f32> {
        let mut dx = GTensor::uninit(gpu, &[dy.shape[0], dy.shape[1], self.input]);
        self.backward(gpu, y, dy, &mut dx);
        dx
    }

    /// Backward over the whole sequence. `dy` is `[B, T, H]`, `dx` is the
    /// caller's `[B, T, in]` output. Accumulates weight/bias grads.
    /// `y` is this cell's forward output — the post-cell norm's output, which that
    /// norm's backward divides by γ to recover `x̂`. The caller keeps it; the norm
    /// itself stores only `inv_rms`.
    pub fn backward(
        &mut self,
        gpu: &Gpu,
        y: &GTensor<f32>,
        dy: &GTensor<f32>,
        dx: &mut GTensor<f32>,
    ) {
        assert_eq!(dy.rank, 3, "SLstm::backward expects [B, T, H]");
        let (b, t, h) = (dy.shape[0], dy.shape[1], dy.shape[2]);
        assert_eq!(b, self.batch, "SLstm::backward — batch mismatch");
        assert_eq!(h, self.hidden, "SLstm::backward — hidden mismatch");
        assert_eq!(dx.dims(), [b, t, self.input], "SLstm::backward — dx shape");
        let inp = self.input;
        let h4 = 4 * h;
        let n = b * t;

        // Taken, not borrowed: these are rebuilt by every forward, so releasing them
        // here frees the device memory across the optimizer step.
        let mut g = self.g.take().expect("forward before backward");
        let mut slabs = self.slabs.take().expect("forward before backward");
        let x_flat = self.x_saved.take().expect("forward before backward");

        // BPTT channels start at zero — unless this call continues the backward of a
        // sequence whose later chunk ran first (see `set_carry`), where they start at
        // the gradient flowing back across the chunk border. Chunks run in reverse, so
        // "the previous call" is the chunk to the right.
        let carry = self.carry && self.dh_bptt.dims() == [b, h];
        for buf in [&mut self.dh_bptt, &mut self.dc_bptt, &mut self.dn_bptt] {
            if !carry {
                fit_zeros(gpu, buf, &[b, h]);
            }
        }

        // The post-cell norm is the last thing forward applied, so it is the first
        // thing to undo: its `dx` is what the recurrence actually receives.
        //
        // It lands in `dy_buf`, the buffer this cell reuses call to call, so undoing
        // the norm doubles as the staging the loop would otherwise need.
        let mut dy_buf = take_uninit(gpu, self.dy_buf.take(), &[b, t, h]);
        phase::timed(gpu, phase::Bucket::SlstmCopyBwd, || {
            self.post_norm.backward(gpu, dy, y, &mut dy_buf);
        });

        // The only thing the loop must carry is BPTT: the gate deltas go straight
        // back into `g`, and everything derived from them waits until the loop ends.
        phase::timed(gpu, phase::Bucket::SlstmLoopBwd, || {
            self.bwd_loop(gpu, &dy_buf, &mut g, &slabs, t);
        });
        self.dy_buf = Some(dy_buf);

        // `g` now holds the gate deltas for the whole sequence: dx, dWx, dWh and the
        // bias grads are three GEMMs and one reduction over it.
        let dg = g.reshaped(&[n, h4]);
        dx.reshape_to(&[n, inp]);
        // `dx = dg·Wxᵀ`, `dWx = x_flatᵀ·dg` and `dWh = h_prevᵀ·dg` all read `dg`, so
        // where the operands are narrow all three go through one call that casts it
        // once. The gate grads land in the parameter layout directly, so `dWx`/`dWh`
        // write with beta = 1: cuBLAS does the accumulation across windows that the
        // unpack kernel used to do by hand.
        slabs.h_prev.shrink_to(&[n, h]);
        let wx = &self.wx;
        let dwx = &mut self.dwx;
        let dwhr = &mut self.dwhr;
        let gemm_dx = &mut self.gemm_dx;
        let h_prev_f32 = &mut self.h_prev_f32;
        phase::timed(gpu, phase::Bucket::SlstmGemmBwd, || {
            match (&x_flat, &slabs.h_prev) {
                (SlabBuf::Bf16(xb), SlabBuf::Bf16(hb)) => {
                    gemm_dx.run_slstm_backward(gpu, xb, hb, &dg, wx, dwx, dwhr, dx)
                }
                // Either the GEMMs or the kernels were built fp32, so `dWh` goes to
                // cuBLAS wide — widening a narrow `h_prev` into reusable scratch first.
                // The scratch is transient (one GEMM) while the slab is pinned across
                // the whole forward and backward, so it still gives memory back.
                (x, hp) => {
                    match x {
                        SlabBuf::Bf16(xb) => {
                            gemm_dx.run_backward_staged_x(gpu, xb, &dg, wx, dwx, dx)
                        }
                        SlabBuf::F32(xf) => {
                            ops::matmul_nt_into(gpu, &dg, wx, dx, 0.0);
                            ops::matmul_tn_into(gpu, xf, &dg, dwx, 1.0);
                        }
                    }
                    let mut scratch = take_uninit(gpu, h_prev_f32.take(), &[n, h]);
                    let hf = hp.as_f32(gpu, &mut scratch);
                    ops::matmul_tn_into(gpu, hf, &dg, dwhr, 1.0);
                    *h_prev_f32 = Some(scratch);
                }
            }
        });
        slabs.h_prev.shrink_to(&[b, t, h]);

        // The bias gradient is the column sum of the gate deltas, accumulating straight
        // into the fused `dbcat`. Nothing to scatter afterwards.
        let dbcat = &mut self.dbcat;
        phase::timed(gpu, phase::Bucket::SlstmGemmBwd, || {
            ops::add_col_sum(gpu, dbcat, &dg);
        });

        // FlashRNN keeps `dR`/`db` at bf16 (`Ctype = CUDA_R_16BF`, `beta = 1`), so every
        // accumulation into them round-trips through 8 mantissa bits. Ours are fp32
        // arena windows; rounding them here reproduces that precision exactly, which is
        // the half of the question that can be answered without narrowing the whole
        // arena. Off unless `SLSTM_BF16_GRAD=1`.
        if grad_bf16(gpu) {
            let Self {
                dwx, dwhr, dbcat, ..
            } = self;
            for t in [dwx, dwhr, dbcat] {
                ops::quantize_bf16_(gpu, t);
            }
        }

        // Give the buffers back at their original shapes so the next forward reuses
        // the same allocations.
        self.g = Some(dg.reshaped(&[b, t, h4]));
        self.slabs = Some(slabs);
        self.x_saved = Some(x_flat);

        // Chunked sweep: this chunk is done, so the chunk to its left — the next one
        // to unwind — takes the live slots. Its buffers are the ones its own forward
        // wrote, so backward reads exactly what that chunk produced. Dropping what was
        // just handed back releases this chunk's activations now rather than at the
        // next forward, which is what keeps only the chunks still owed a backward
        // resident.
        match self.chunk_saved.pop() {
            Some(SlstmChunk::Resident { g, slabs, x_saved }) => {
                self.g = Some(g);
                self.slabs = Some(slabs);
                self.x_saved = Some(x_saved);
            }
            // The park's generations pop in the same right-to-left order, so this
            // restores the chunk to the left — exactly the one that unwinds next.
            Some(SlstmChunk::Parked) => {
                let park = self.park.as_mut().expect("parked chunk without a park");
                let (g, slabs, x_saved) = park_unorder(park.restore(gpu));
                self.g = Some(g);
                self.slabs = Some(slabs);
                self.x_saved = Some(x_saved);
            }
            None => {}
        }

        dx.reshape_to(&[b, t, inp]);
    }

    /// The backward time loop — time-fused on the same terms as [`Self::fwd_loop`].
    fn bwd_loop(
        &mut self,
        gpu: &Gpu,
        dy: &GTensor<f32>,
        g: &mut GTensor<f32>,
        slabs: &SlstmSlabs,
        t: usize,
    ) {
        // One cooperative launch for the whole reverse loop. It carries the gate deltas
        // through `g` itself, so it needs no scratch of its own.
        let b = dy.shape[0];
        if self.fuses_at(b, t)
            && ops::slstm_fused_time_bwd(
                gpu,
                &self.whr,
                dy,
                g,
                &mut self.dh_bptt,
                &mut self.dc_bptt,
                &mut self.dn_bptt,
                slabs,
                t,
            )
        {
            return;
        }
        if self.batches_at(gpu, b)
            && ops::slstm_batched_bwd(
                gpu,
                &self.whr,
                dy,
                g,
                &mut self.dh_bptt,
                &mut self.dc_bptt,
                &mut self.dn_bptt,
                slabs,
                t,
            )
        {
            return;
        }
        self.bwd_steps(gpu, dy, g, slabs, t);
    }

    fn bwd_steps(
        &mut self,
        gpu: &Gpu,
        dy: &GTensor<f32>,
        g: &mut GTensor<f32>,
        slabs: &SlstmSlabs,
        t: usize,
    ) {
        // The contiguous `[B, 4H]` gate-delta scratch only this path needs, fitted here
        // rather than in `backward` because which path runs is not known until the
        // fused one has been *tried* — it can decline at launch, not just at geometry.
        let (b, h) = (self.dc_bptt.rows(), self.dc_bptt.cols());
        self.dgh.fit(gpu, &[b, 4 * h]);
        for step in (0..t).rev() {
            ops::slstm_step_fused_bwd(
                gpu,
                dy,
                g,
                &mut self.dgh,
                &self.dh_bptt,
                slabs,
                &mut self.dc_bptt,
                &mut self.dn_bptt,
                step,
            );
            // dh_{t-1} = dgates_t · Whᵀ — the one gradient BPTT cannot defer. The
            // weight comes from the same cache the forward's `h·Wh` fills: it is the
            // same `whr`, narrowed once per optimizer step rather than once per GEMM.
            let Self {
                gemm_h,
                dgh,
                whr,
                dh_bptt,
                ..
            } = self;
            match dgh {
                ops::SlabBuf::Bf16(d) => {
                    gemm_h.run_staged_lhs(gpu, ops::MmForm::Nt, d, whr, dh_bptt, 0.0)
                }
                ops::SlabBuf::F32(d) => ops::matmul_nt_into(gpu, d, whr, dh_bptt, 0.0),
            }
        }
    }

    /// Every parameter with its gradient and AdamW moments. The gate matrices decay,
    /// the fused bias does not; the post-cell norm's γ comes last, where the enclosing
    /// block used to emit it.
    ///
    /// AdamW is elementwise, so stepping the fused `[in, 4H]` / `[H, 4H]` operands is
    /// numerically identical to stepping the four `[rows, H]` gate matrices they hold
    /// — the decay split that matters is weights vs. bias, and that survives the
    /// fusion because `bcat` is still its own tensor.
    pub fn param_slots(&mut self) -> Vec<ParamSlot<'_>> {
        // The caller gets a mutable handle on `wx` (checkpoint load overwrites it), so
        // the cached bf16 copy must be assumed stale from here on.
        self.gemm_x.invalidate_w();
        self.gemm_dx.invalidate_w();
        self.gemm_h.invalidate_w();
        let mut v = vec![
            ParamSlot::new(
                &mut self.wx,
                &mut self.dwx,
                &mut self.mwx,
                &mut self.vwx,
                ParamKind::Decay,
            ),
            ParamSlot::new(
                &mut self.whr,
                &mut self.dwhr,
                &mut self.mwhr,
                &mut self.vwhr,
                ParamKind::Decay,
            ),
            ParamSlot::new(
                &mut self.bcat,
                &mut self.dbcat,
                &mut self.mbcat,
                &mut self.vbcat,
                ParamKind::NoDecay,
            ),
        ];
        v.extend(self.post_norm.param_slots());
        v
    }

    /// Every learnable tensor, in a fixed order (used by checkpoint save/load).
    pub fn params_mut(&mut self) -> Vec<&mut GTensor<f32>> {
        self.param_slots().into_iter().map(|s| s.param).collect()
    }

    /// Gradient accumulators, in the same order as `params_mut`. Diagnostic.
    pub fn grads(&mut self) -> Vec<&GTensor<f32>> {
        self.param_slots().into_iter().map(|s| &*s.grad).collect()
    }

    /// Forward-cache extremes of the last sweep: `(min |n|, max |c|, max |c/n|)`.
    ///
    /// The backward divides by `n` and by `n²`, so a normalizer that collapses is the
    /// difference between a finite gradient and a NaN. Diagnostic — downloads the
    /// whole `[B, T, H]` slabs, so it belongs in a probe.
    pub fn state_extremes(&self, gpu: &Gpu) -> Option<(f32, f32, f32)> {
        let slabs = self.slabs.as_ref()?;
        // Widened into scratch: under `SLSTM_BF16_STATE` these are the narrow copies,
        // and the extremes we are after are precisely what narrowing might move.
        let mut scratch = GTensor::uninit(gpu, slabs.n.dims());
        let n = slabs.n.as_f32(gpu, &mut scratch).to_host(gpu).data.to_vec();
        let mut scratch = GTensor::uninit(gpu, slabs.c.dims());
        let c = slabs.c.as_f32(gpu, &mut scratch).to_host(gpu).data.to_vec();
        let min_n = n.iter().map(|v| v.abs()).fold(f32::INFINITY, f32::min);
        let max_c = c.iter().map(|v| v.abs()).fold(0.0, f32::max);
        let max_ratio = c
            .iter()
            .zip(&n)
            .map(|(a, b)| (a / b).abs())
            .fold(0.0, f32::max);
        Some((min_n, max_c, max_ratio))
    }

    /// Release the forward cache — the `[B, T, ·]` slabs and the saved input — without
    /// reading it.
    ///
    /// For a stack that re-forwards rather than unwinding; see
    /// `Block::drop_saved_act`. These are the cell's largest buffers, and in the
    /// encoder every group but the last leaves them holding activations no backward
    /// will ever read.
    pub fn drop_saved_act(&mut self) {
        self.slabs = None;
        self.x_saved = None;
        self.chunk_saved.clear();
        self.discard_parked();
        // The GEMM staging deliberately stays: `drop_saved_act` runs between the chunks
        // of a sweep, and dropping the cached bf16 `Wx` there costs a re-narrow per
        // chunk. It is bounded by the window, not the corpus, and `clear` on the layer
        // is what releases it.
    }

    /// Continue the previous call's recurrence rather than starting from zero.
    ///
    /// For a **chunked** sweep: one sequence split across several `forward` calls,
    /// where the state must cross the chunk borders for the result to match the
    /// unchunked recurrence. Forward carries `h/c/n/m`; backward carries the BPTT
    /// channels, and because chunks unwind right-to-left "the previous call" there is
    /// the chunk to its right.
    ///
    /// The caller is responsible for the boundaries: clear it (or call
    /// [`reset_state`](Self::reset_state)) before the **first** forward chunk and
    /// before the **last** backward chunk, so each sweep starts from zero. Leaving it
    /// set across a window silently seeds the next window with the previous one's
    /// final state.
    /// Whether a `[B, T, H]` sweep takes the time-fused path rather than the per-step
    /// one. Both halves matter: `T` must be long enough to amortise the `Wh` staging,
    /// and `B` small enough that the fused kernel's scalar recurrent product still beats
    /// cuBLAS. `force_fused_time` overrides both so a benchmark can measure either arm.
    ///
    /// The kernel can still decline at launch (shared memory, registers, cooperative
    /// co-residency), so a `true` here is necessary, not sufficient.
    pub fn fuses_at(&self, b: usize, t: usize) -> bool {
        t >= FUSED_MIN_T
            && self
                .force_fused_time
                .unwrap_or_else(|| fused_time_enabled() && b <= FUSED_MAX_B)
    }

    /// Whether a `[B, T, H]` sweep takes the **batch-parallel** fused path.
    ///
    /// The complement of [`fuses_at`](Self::fuses_at): that one is the scalar
    /// contraction, which is right exactly where the batch cannot fill a tensor-core
    /// tile; this one puts `h·Wh` on the mma unit and takes over once the batch is wide
    /// enough to fill one. `SB_MIN_B` is that width — below it the M axis is mostly
    /// padding — and `ops::slstm_batched_pays` is the ceiling, where the kernels'
    /// per-column-block operand re-reads outgrow the launches they save.
    ///
    /// `T` does not appear. The kernel amortises nothing over the sequence that a
    /// wide batch does not already amortise over the rows, and the encoder's whole
    /// range (T = 1..17) is a case the per-step path loses outright: it pays two
    /// launches per timestep for a matmul the GPU finishes in under a microsecond.
    ///
    /// The kernel can still decline at launch (shared memory, registers, cooperative
    /// co-residency), so a `true` here is necessary, not sufficient.
    pub fn batches_at(&self, gpu: &Gpu, b: usize) -> bool {
        self.force_batched.unwrap_or_else(|| {
            batched_enabled() && b >= SB_MIN_B && ops::slstm_batched_pays(gpu, self.hidden, b)
        })
    }

    pub fn set_carry(&mut self, carry: bool) {
        self.carry = carry;
        self.post_norm.set_carry(carry);
    }

    /// Zero the carried **forward** state, so the next `forward` starts the recurrence
    /// from scratch whatever `carry` says.
    ///
    /// Call before the first chunk of a sweep. Separate from
    /// [`reset_bptt`](Self::reset_bptt) because the two are reset at opposite ends: a
    /// chunked forward runs left-to-right and resets here, while its backward runs
    /// right-to-left and resets at the *last* chunk.
    pub fn reset_state(&mut self, gpu: &Gpu) {
        for s in [
            &mut self.h_state,
            &mut self.c_state,
            &mut self.n_state,
            &mut self.m_state,
        ] {
            if !s.is_empty() {
                s.zero_(gpu);
            }
        }
        // A sweep that ended early (a caller that forwarded chunks and never unwound
        // them) would otherwise leave its caches to accumulate across steps.
        self.chunk_saved.clear();
        self.discard_parked();
    }

    /// Drop any host generations left over from a sweep that was abandoned before its
    /// backward consumed them, so they do not accumulate across steps.
    fn discard_parked(&mut self) {
        if let Some(park) = &mut self.park {
            park.discard_all();
        }
    }

    /// Park this cell's set-aside chunk caches on the host between forward and
    /// backward.
    ///
    /// Opted into by the surrounding [`Block`](super::block::Block), and subject to the
    /// same constraint: only for a stack whose whole forward precedes its backward.
    /// See `Block::enable_offload`.
    pub fn enable_offload(&mut self, gpu: &Gpu, in_flight: super::offload::SharedInFlight) {
        self.park =
            Some(super::offload::HostPark::new(gpu, in_flight).expect("offload: host park"));
    }

    /// Start the parked chunk on its way back, without waiting. Called one block ahead
    /// of this cell's backward so the upload overlaps compute.
    pub fn prefetch_saved(&mut self, gpu: &Gpu) {
        if let Some(park) = &mut self.park {
            park.prefetch(gpu);
        }
    }

    /// Zero the carried **BPTT** channels, so the next `backward` starts with no
    /// incoming gradient from the right. Call before the rightmost chunk's backward.
    pub fn reset_bptt(&mut self, gpu: &Gpu) {
        for s in [&mut self.dh_bptt, &mut self.dc_bptt, &mut self.dn_bptt] {
            if !s.is_empty() {
                s.zero_(gpu);
            }
        }
    }

    /// Retained activation bytes split `(saved_cache, other)`.
    ///
    /// `saved_cache` is the `[B, T, ·]` slabs and saved input that
    /// [`drop_saved_act`](Self::drop_saved_act) releases. `other` is everything it
    /// keeps: the gate buffer, the stable `out`/`dy` buffers, the widening scratch,
    /// the per-batch state and BPTT channels, and the post-norm's saved `x̂`.
    pub fn act_split(&self) -> (usize, usize) {
        let saved = self.slabs.as_ref().map_or(0, |s| s.retained_bytes())
            + self.x_saved.as_ref().map_or(0, |t| t.retained_bytes())
            + self
                .chunk_saved
                .iter()
                .map(|c| match c {
                    SlstmChunk::Resident { g, slabs, x_saved } => {
                        slabs.retained_bytes() + x_saved.retained_bytes() + g.capacity() * 4
                    }
                    // On the host, so it holds no device bytes — which is the point.
                    SlstmChunk::Parked => 0,
                })
                .sum::<usize>();
        let (_, all) = self.retained_bytes();
        (saved, all - saved)
    }

    /// Release every activation this cell holds — the saved slabs and input, the
    /// gate/output/dy buffers and the widening scratch.
    ///
    /// Broader than [`drop_saved_act`](Self::drop_saved_act), which keeps the reused
    /// buffers on purpose. For a window boundary, not the hot path.
    pub fn drop_all_act(&mut self) {
        // The big per-`[B, T, ·]` buffers: these are what scale with the rectangle and
        // what a group boundary needs back.
        self.slabs = None;
        self.x_saved = None;
        self.chunk_saved.clear();
        self.post_norm.drop_saved_act();

        // `g`, `out_buf`, `dy_buf` and `h_prev_f32` are deliberately KEPT: the
        // encoder and decoder run one rectangle per length bucket and the buckets
        // repeat window after window, so dropping them per group would mean a fresh
        // allocation for every group of every window.
        //
        // Keeping them costs the `[B, T, 4H]` gate buffer and two `[B, T, H]` staging
        // buffers at the LARGEST bucket's shape, which at the encoder/decoder's
        // CHAR_HIDDEN=256 is single-digit MB — against the ~1.2 GB that releasing the
        // slabs and saved input recovers.
    }

    /// Device bytes held, split `(params, activations)`. Diagnostic — see
    /// [`Hierarchical::retained_report`](super::hierarchical::Hierarchical::retained_report).
    ///
    /// `activations` covers everything that scales with the window: the saved slabs
    /// and input, the gate buffer, the stable `out`/`dy` buffers, the per-batch
    /// recurrent state and BPTT channels, and the `h_prev` widening scratch. Only the
    /// first two are released by [`drop_saved_act`](Self::drop_saved_act) — the rest
    /// are kept deliberately, so the next call at the same shape reuses them.
    pub fn retained_bytes(&self) -> (usize, usize) {
        let params: usize = [
            &self.wx,
            &self.whr,
            &self.bcat,
            &self.dwx,
            &self.dwhr,
            &self.dbcat,
            &self.mwx,
            &self.vwx,
            &self.mwhr,
            &self.vwhr,
            &self.mbcat,
            &self.vbcat,
        ]
        .iter()
        .map(|t| t.capacity() * 4)
        .sum();
        let opt: usize = [&self.g, &self.out_buf, &self.dy_buf, &self.h_prev_f32]
            .iter()
            .filter_map(|s| s.as_ref())
            .map(|t| t.capacity() * 4)
            .sum::<usize>()
            + self.x_saved.as_ref().map_or(0, |t| t.retained_bytes());
        let live: usize = [
            &self.h_state,
            &self.c_state,
            &self.n_state,
            &self.m_state,
            &self.gh,
            &self.dh_bptt,
            &self.dc_bptt,
            &self.dn_bptt,
        ]
        .iter()
        .map(|t| t.capacity() * 4)
        .sum::<usize>()
            + self.h_narrow.retained_bytes()
            + self.dgh.retained_bytes();
        let slabs = self.slabs.as_ref().map_or(0, |s| s.retained_bytes());
        let staging = self.gemm_x.retained_bytes()
            + self.gemm_dx.retained_bytes()
            + self.gemm_h.retained_bytes();
        let (pn_p, _) = self.post_norm.retained_bytes();
        (params + pn_p, opt + live + slabs + staging)
    }

    pub fn zero_grad(&mut self, gpu: &Gpu) {
        for g in [&mut self.dwx, &mut self.dwhr, &mut self.dbcat] {
            g.zero_(gpu);
        }
        self.post_norm.zero_grad(gpu);
    }

    /// AdamW over this cell's own parameters, then clear the grads. A model steps
    /// its whole `ParamArena` in one launch instead.
    pub fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        arena::step_slots(gpu, &mut self.param_slots(), cfg);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn2::optim::AdamCfg;
    use crate::nn2::slstm::SLstm as CpuSLstm;

    /// GPU-vs-CPU tolerance, scaled for the slab storage dtype in use.
    ///
    /// The CPU reference is all-fp32, so with bf16 slabs the gap is one quantization
    /// of `zt`/`ot` (~2^-8 relative) propagated through the recurrence. Loosening is
    /// expected here; what would be a regression is the error GROWING with T, which
    /// `bf16_slab_error_does_not_compound_with_t` pins separately.
    fn tol(gpu: &Gpu, base: f32) -> f32 {
        // x128, not x8. bf16's half-ulp is ~2e-3 relative at these magnitudes and a
        // T-loop accumulates several of them, so x8 sat right ON the observed spread
        // (a 2.0e-3 diff against a 2.0e-3 bound) and failed about one run in five.
        // A bound a correct implementation trips intermittently is worse than none;
        // the property that would actually signal a bug — error GROWING with T — is
        // pinned separately by `bf16_slab_error_does_not_compound_with_t`.
        //
        // x16 was enough while the post-cell norm sat in the block. Now that the cell
        // owns it, `backward` runs that noise through the norm's `γ·dY − x̂·S/F`
        // reduction before it reaches `dx`, which amplifies it — and, because the
        // reduction is over the whole row, makes the worst element swing a lot from
        // run to run. Ten measured runs of `slstm_long_t_matches_cpu` at T=64
        // spread 2.2e-2 to 5.9e-2 on `dx`, so x32 (6.4e-2) was back to clearing by a
        // hair and failing intermittently. x128 (2.6e-1) sits ~4x above the observed
        // maximum, which is the margin this bound needs to be worth having.
        //
        // That this is bf16 and not a logic error is directly checkable:
        // `GPU_NO_BF16=1` makes both long-T tests pass at the base tolerance, two
        // orders tighter.
        if gpu.kernels.slab_bf16 {
            base * 128.0
        } else {
            base
        }
    }

    /// Tolerance for a parameter compared AFTER an Adam step. Wider than [`tol`]:
    /// `lr·ĝ/(√v̂+ε)` is scale-invariant in the gradient, so a near-zero-gradient
    /// weight turns a ~1e-7 difference into ~1e-4.
    fn step_tol(gpu: &Gpu, base: f32) -> f32 {
        // A MULTIPLE, not `base.max(2e-3)` — with base == 2e-3 that was a no-op and
        // left the post-Adam check at its fp32 bound, which failed ~2 runs in 10.
        // Adam's `lr·ĝ/(√v̂+ε)` is scale-invariant in the gradient, so a weight whose
        // gradient is near zero divides a bf16-sized difference by an equally small
        // √v̂ and lands far wider than the activations it came from.
        if gpu.kernels.slab_bf16 {
            base * 16.0
        } else {
            base
        }
    }

    fn assert_close(got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len(), "length mismatch");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < tol, "index {i}: gpu {g} vs cpu {w}");
        }
    }

    fn from_cpu(gpu: &Gpu, cpu: &CpuSLstm) -> SLstm {
        SLstm::from_parts(
            gpu,
            cpu.input_size,
            cpu.hidden_size,
            &cpu.wz,
            &cpu.wi,
            &cpu.wf,
            &cpu.wo,
            &cpu.bz,
            &cpu.bi,
            &cpu.bf,
            &cpu.bo,
            None,
        )
    }

    /// The CPU reference for a GPU cell: `nn2::SLstm` is the bare recurrence, while
    /// the GPU cell now also owns the post-cell norm, so the comparison composes the
    /// two by hand. γ starts at 1 on both sides (`from_cpu` above passes `None`).
    ///
    /// Only the *cell's* gradients are compared in these tests, and the norm sits
    /// downstream of every one of them — so routing dy through it is exactly what
    /// makes the two sides see the same delta at the recurrence.
    struct CpuRef {
        cell: CpuSLstm,
        norm: crate::nn2::rms_norm::RmsNorm,
    }

    impl CpuRef {
        fn new(inp: usize, h: usize) -> Self {
            Self {
                cell: CpuSLstm::new(inp, h),
                norm: crate::nn2::rms_norm::RmsNorm::new(h),
            }
        }
        /// `nn2::RmsNorm` is strictly rank 2 (its GPU twin folds leading axes
        /// itself), so the `[B, T, H]` sequences are flattened around it here.
        fn flat(t: &Tensor) -> Tensor {
            let f = t.shape[t.rank - 1];
            t.reshape(&[t.len() / f, f])
        }
        fn forward(&mut self, x: &Tensor) -> Tensor {
            let y = self.cell.forward(x);
            let dims = y.dims().to_vec();
            self.norm.forward(&Self::flat(&y)).reshape(&dims)
        }
        fn backward(&mut self, dy: &Tensor) -> Tensor {
            let dims = dy.dims().to_vec();
            let d = self.norm.backward(&Self::flat(dy)).reshape(&dims);
            self.cell.backward(&d)
        }
        fn step(&mut self, cfg: &AdamCfg) {
            self.cell.step(cfg);
            self.norm.step(cfg);
        }
    }

    /// The whole-sequence GEMMs cache a bf16 copy of `Wx`, so every write to `Wx` must
    /// invalidate it. A stale cache does not error and does not go non-finite — the
    /// forward simply keeps reading the weight the optimizer has already moved past, so
    /// the observable is that the output stops tracking `Wx`.
    ///
    /// An end-to-end "does the loss fall" check does NOT catch this: one step of
    /// staleness is a small perturbation, and a small model converges either way
    /// (measured — identical final loss with the invalidation removed). Hence this
    /// direct test.
    #[test]
    fn wx_cache_follows_optimizer_steps() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let (b, t, inp, h) = (2, 4, 5, 6);
        let mut cell = SLstm::new_rand(&gpu, inp, h);
        let x = GTensor::from_host(&gpu, &Tensor::random(&[b, t, inp], 0.5));

        // `Wx` is overwritten DIRECTLY rather than stepped: a real optimizer step also
        // moves `Whr`, `bcat` and the norm's γ, and those alone change the output — so
        // a stale `Wx` stays hidden and the test passes with the invalidation removed
        // (measured). Writing only `Wx` makes the cache the single variable.
        for round in 1..=3 {
            let before = cell.forward_alloc(&gpu, &x).to_host(&gpu).data;

            let mut hw = cell.wx.to_host(&gpu);
            for (i, v) in hw.data.iter_mut().enumerate() {
                *v += 0.25 * ((i % 7) as f32 - 3.0) * round as f32;
            }
            cell.wx = GTensor::from_host(&gpu, &hw);
            // The write above is exactly what `params_mut` exists to guard.
            cell.params_mut();

            let after = cell.forward_alloc(&gpu, &x).to_host(&gpu).data;
            let changed = before
                .iter()
                .zip(&after)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                changed > 1e-4,
                "round {round}: Wx was overwritten but the output did not change — the \
                 bf16 weight cache was not invalidated"
            );
        }
    }

    /// GPU sLSTM must match `nn2::SLstm` (cell alone) for a full
    /// forward → backward → AdamW-step cycle, from identical weights. Tolerance
    /// is loose-ish because the two paths differ in float reduction order (cuBLAS
    /// vs the CPU gemm), but the recurrence math is identical.
    #[test]
    fn slstm_matches_cpu_layer() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let (b, t, inp, h) = (2, 5, 4, 6);

        let mut cpu = CpuRef::new(inp, h);
        let mut dev = from_cpu(&gpu, &cpu.cell);

        let x = Tensor::random(&[b, t, inp], 0.5);
        let g = Tensor::random(&[b, t, h], 1.0);

        // Forward
        let y_cpu = cpu.forward(&x);
        let y_dev = dev.forward_alloc(&gpu, &GTensor::from_host(&gpu, &x));
        assert_close(&y_dev.to_host(&gpu).data, &y_cpu.data, tol(&gpu, 2e-3));

        // Backward
        let dx_cpu = cpu.backward(&g);
        let dx_dev = dev.backward_alloc(&gpu, &y_dev, &GTensor::from_host(&gpu, &g));
        assert_close(&dx_dev.to_host(&gpu).data, &dx_cpu.data, tol(&gpu, 2e-3));
        assert_close(&dev.gate_dw(&gpu, 0), &cpu.cell.dwz.data, tol(&gpu, 2e-3));
        assert_close(&dev.gate_dw(&gpu, 2), &cpu.cell.dwf.data, tol(&gpu, 2e-3));
        assert_close(&dev.gate_db(&gpu, 2), &cpu.cell.dbf.data, tol(&gpu, 2e-3));

        // One AdamW step, then compare the updated gate weights.
        let mut cfg = AdamCfg::new(1e-3, 0.01);
        cfg.t = 1;
        cpu.step(&cfg);
        dev.step(&gpu, &cfg);
        // Looser than Linear's 1e-5: the AdamW update is ~lr in magnitude, and a
        // near-zero grad element can sign-flip between the cuBLAS and CPU gemm
        // reduction orders, swinging its update by ~2·lr. A plumbing bug misses by
        // O(weight), far more than this.
        assert_close(
            &dev.gate_w(&gpu, 0),
            &cpu.cell.wz.data,
            step_tol(&gpu, 2e-3),
        );
        assert_close(
            &dev.gate_w(&gpu, 2),
            &cpu.cell.wf.data,
            step_tol(&gpu, 2e-3),
        );
        assert_close(
            &dev.gate_b(&gpu, 2),
            &cpu.cell.bf.data,
            step_tol(&gpu, 2e-3),
        );
    }

    /// The same parity check at `T > FUSED_MIN_T`, so both time loops run as **one
    /// cooperative launch** — the path `slstm_matches_cpu_layer` (T=5) never reaches.
    ///
    /// Two full cycles with an optimizer step between them, checked separately. The
    /// fused kernels read `whr`/`bcat` live out of the parameters, and the whole-
    /// sequence GEMMs read a *cached* bf16 copy of `wx`; a second pass against
    /// changed weights is what catches a cache that was not invalidated.
    #[test]
    fn slstm_long_t_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        assert!(
            64 > FUSED_MIN_T,
            "this test must exercise the time-fused path"
        );
        let (b, t, inp, h) = (2, 64, 8, 12);

        let mut cpu = CpuRef::new(inp, h);
        let mut dev = from_cpu(&gpu, &cpu.cell);
        let mut cfg = AdamCfg::new(1e-3, 0.01);

        for pass in 0..2 {
            let x = Tensor::random(&[b, t, inp], 0.5);
            let g = Tensor::random(&[b, t, h], 1.0);

            let y_cpu = cpu.forward(&x);
            let y_dev = dev.forward_alloc(&gpu, &GTensor::from_host(&gpu, &x));
            assert_close(&y_dev.to_host(&gpu).data, &y_cpu.data, tol(&gpu, 2e-3));

            let dx_cpu = cpu.backward(&g);
            let dx_dev = dev.backward_alloc(&gpu, &y_dev, &GTensor::from_host(&gpu, &g));
            assert_close(&dx_dev.to_host(&gpu).data, &dx_cpu.data, tol(&gpu, 2e-3));
            assert_close(&dev.gate_dw(&gpu, 0), &cpu.cell.dwz.data, tol(&gpu, 2e-3));
            assert_close(&dev.gate_dw(&gpu, 2), &cpu.cell.dwf.data, tol(&gpu, 2e-3));

            // Step between passes, so the second cycle runs against *changed*
            // weights — the loop must read the packed operands live, not a stale copy.
            cfg.t = pass + 1;
            cpu.step(&cfg);
            dev.step(&gpu, &cfg);
            assert_close(
                &dev.gate_w(&gpu, 0),
                &cpu.cell.wz.data,
                step_tol(&gpu, 2e-3),
            );
        }
    }

    /// The cell's per-call buffers are *reused* whenever the shape matches and
    /// reallocated when it does not, so a run of differently shaped windows walks
    /// them through allocate/reuse/free repeatedly. This is the real training
    /// pattern — windows never cross a document border, so every short document
    /// yields a short window — and anything that survived a call it should not
    /// (a stale slab, a scratch sized for the previous shape) shows up here.
    ///
    /// The sequence matters: long, short, long-again, so a shape is revisited after
    /// its buffers have been handed back once.
    #[test]
    fn slstm_survives_shape_changes() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let (b, inp, h) = (1, 64, 64);
        let mut cpu = CpuRef::new(inp, h);
        let mut dev = from_cpu(&gpu, &cpu.cell);

        // Freed device memory only *shows* it was reused if somebody reuses it. In
        // the real model the encoder/decoder's per-group temporaries do that between
        // two backbone sweeps; here nothing else allocates, so the pool would hand
        // the very same addresses back and a stale read would still see plausible
        // data. Grab (and dirty) a block between windows to stand in for that churn.
        let poison = |floats: usize| {
            let mut d = GTensor::uninit(&gpu, &[floats]);
            ops::fill(&gpu, &mut d, 1e30);
        };

        for &t in &[256usize, 8, 256, 192, 5, 256] {
            let x = Tensor::random(&[b, t, inp], 0.5);
            let g = Tensor::random(&[b, t, h], 1.0);

            let y_cpu = cpu.forward(&x);
            let y_dev = dev.forward_alloc(&gpu, &GTensor::from_host(&gpu, &x));
            assert_close(&y_dev.to_host(&gpu).data, &y_cpu.data, tol(&gpu, 2e-3));
            let dx_cpu = cpu.backward(&g);
            let dx_dev = dev.backward_alloc(&gpu, &y_dev, &GTensor::from_host(&gpu, &g));
            assert_close(&dx_dev.to_host(&gpu).data, &dx_cpu.data, tol(&gpu, 2e-3));

            poison(256 * 4 * h * b);
        }
    }

    /// A shape long enough to *ask* for the time-fused path but too wide for it must
    /// still run, and run correctly.
    ///
    /// `ops::slstm_fused_time` can decline in two places — the geometry, and the launch
    /// itself, where the driver refuses a cooperative grid it cannot make co-resident.
    /// So `forward` cannot know which loop will run, and anything the per-step loop
    /// needs has to be prepared by that loop rather than guessed at in advance. It was
    /// guessed at once: the per-step scratch was skipped whenever T was long, and the
    /// first shape that declined at launch reached the fallback with an unallocated
    /// operand and panicked inside cuBLAS.
    ///
    /// The batch here is what declines: the pointwise phase is one thread per (batch
    /// row, owned unit), so a batch this wide needs a block wider than the hardware
    /// allows. `T` is above `FUSED_MIN_T` all the same.
    #[test]
    fn wide_batch_falls_back_to_the_per_step_loop() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let (b, t, h) = (1088usize, 32usize, 32usize);
        let s = 1.0 / (h as f32).sqrt();
        let w: Vec<Tensor> = (0..4)
            .map(|g| Tensor::random(&[2 * h, h], s * (1.0 + g as f32 * 0.05)))
            .collect();
        let bi: Vec<Tensor> = (0..4).map(|g| Tensor::random(&[h], 0.2)).collect();
        let x = GTensor::from_host(&gpu, &Tensor::random(&[b, t, h], 0.5));

        let build = || {
            SLstm::from_parts(
                &gpu, h, h, &w[0], &w[1], &w[2], &w[3], &bi[0], &bi[1], &bi[2], &bi[3], None,
            )
        };
        // Default selection: asks for the fused path, gets told no, falls back.
        let got = build().forward_alloc(&gpu, &x).to_host(&gpu);
        // The path it lands on, pinned directly.
        let mut forced = build();
        forced.force_fused_time = Some(false);
        let want = forced.forward_alloc(&gpu, &x).to_host(&gpu);
        assert_eq!(
            got.data, want.data,
            "fallback must be the per-step loop exactly"
        );
        assert!(got.data.iter().all(|v| v.is_finite()), "non-finite output");
    }

    /// The time-fused forward (one cooperative launch for the whole T-loop) must
    /// agree with the per-step path it replaces.
    ///
    /// `fused_time_enabled()` reads its env var through a `OnceLock`, so this
    /// drives `ops::slstm_fused_time` directly rather than toggling the flag: the
    /// point is that the kernel computes the same thing, not how it is selected.
    /// T is above `FUSED_MIN_T`, the length at which the cell actually selects it.
    ///
    /// Both fused kernels stage `Wh` in bf16 (FlashRNN's `DTYPE_R`, with an fp32
    /// accumulator), which keeps an 8-bit mantissa on the operand — so agreement with
    /// the all-fp32 per-step path is to ~1e-2 relative, not fp32's ~1e-6.
    #[test]
    fn slstm_fused_time_matches_per_step() {
        // B=1 is the backbone's shape, where the backward gives each unit one warp;
        // the larger batches are the encoder/decoder's, where its warps split the batch
        // instead. Both are the same arithmetic and both must match the per-step path.
        for b in [1usize, 64, 256] {
            fused_time_matches_per_step_at(b, 64);
        }
        // A width whose `Wh` slice no longer fits shared memory, so the forward stages
        // part of it and reads the rest from the global tail. Nothing below H ~ 1000
        // reaches that branch — it is `#if`-ed out of every narrower build.
        fused_time_matches_per_step_at(1, 1024);
    }

    fn fused_time_matches_per_step_at(b: usize, h: usize) {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        if !gpu.kernels.has_coop {
            return; // cooperative kernels unavailable (no CUDA headers)
        }
        let t = 64usize;
        if ops::slstm_fused_time_geometry(&gpu, h, b).is_none()
            || ops::slstm_fused_time_bwd_geometry(&gpu, h, b).is_none()
        {
            return; // shape does not fit the fused path on this device
        }
        // 1/sqrt(H) weight scale, i.e. the initialization the cell actually trains at.
        // At a fixed 0.3 the gates saturate, and a saturated input gate drives its bias
        // gradient to a near-total cancellation — a vector whose largest entry is ~1e-2
        // and whose individual entries are then pure reassociation noise, which no
        // parity bound can be written against.
        let s = 1.0 / (h as f32).sqrt();
        let w: Vec<Tensor> = (0..4)
            .map(|g| Tensor::random(&[2 * h, h], s * (1.0 + g as f32 * 0.05)))
            .collect();
        let bi: Vec<Tensor> = (0..4)
            .map(|g| Tensor::random(&[h], 0.2 + g as f32 * 0.01))
            .collect();
        let x = Tensor::random(&[b, t, h], 0.5);
        let dx = GTensor::from_host(&gpu, &x);

        let mut per_step = SLstm::from_parts(
            &gpu, h, h, &w[0], &w[1], &w[2], &w[3], &bi[0], &bi[1], &bi[2], &bi[3], None,
        );
        per_step.force_fused_time = Some(false);
        let y_per = per_step.forward_alloc(&gpu, &dx);
        let want = y_per.to_host(&gpu);

        let mut fused = SLstm::from_parts(
            &gpu, h, h, &w[0], &w[1], &w[2], &w[3], &bi[0], &bi[1], &bi[2], &bi[3], None,
        );
        fused.force_fused_time = Some(true);
        let y_fused = fused.forward_alloc(&gpu, &dx);
        let got = y_fused.to_host(&gpu);

        // bf16 staging costs ~3 decimal digits on the operand, and a T-loop
        // accumulates several of them.
        let rel_tol = 1e-2;
        assert_eq!(want.data.len(), got.data.len());
        for (i, (a, c)) in want.data.iter().zip(got.data.iter()).enumerate() {
            assert!(
                (a - c).abs() <= rel_tol * a.abs().max(1.0),
                "fused vs per-step forward diverged at {i} (B={b}, H={h}): {a} vs {c}"
            );
        }

        // ...and the backward: dx plus every gate's weight gradient. The fused
        // backward carries the BPTT channels inside one launch, so an error there
        // shows up as drift that grows toward t = 0 rather than a single bad slot.
        let gy = GTensor::from_host(&gpu, &Tensor::random(&[b, t, h], 0.7));
        // Scaled to each tensor's own magnitude: the gradients here run to ~1e2, so a
        // fixed absolute bound would be far tighter on `dx` than on `dW` and would say
        // nothing consistent about either.
        //
        // The scale is the tensor's max |entry|, with no floor. Flooring it at 1.0
        // would make the bound absolute for any tensor that never reaches 1 — and the
        // bias gradients are exactly that: sums over T of terms that largely cancel,
        // so a single entry can sit three orders below the vector's own spread and
        // differ by 2x on noise that is irrelevant to the weight it updates.
        // A slice whose largest entry is this small carries no signal to compare: the
        // input gate saturates under the forget-bias init, so its bias gradient is a
        // sum over T of terms that cancel to ~1e-2 against per-term magnitudes orders
        // larger. What survives is reassociation noise, and the two paths reassociate
        // differently by construction. Every other slice here is O(1)-O(100), so this
        // skips the degenerate case without weakening the real checks.
        //
        // The floor is relative to the largest bias gradient in the same backward, not
        // an absolute constant: every gradient here scales with B, so a fixed floor
        // that skips the degenerate slice at B=1 stops skipping it at B=256 while the
        // slice is just as degenerate — same ~1e-3 ratio to its siblings, same pure
        // cancellation noise.
        let close_rel = |dead: f32, want: &[f32], got: &[f32], what: &str| {
            let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            if scale < dead {
                return;
            }
            for (i, (a, c)) in want.iter().zip(got).enumerate() {
                assert!(
                    (a - c).abs() <= rel_tol * scale,
                    "{what} diverged at {i} (B={b}, H={h}): {a} vs {c} (scale {scale})"
                );
            }
        };
        let want_dx = per_step.backward_alloc(&gpu, &y_per, &gy).to_host(&gpu);
        let got_dx = fused.backward_alloc(&gpu, &y_fused, &gy).to_host(&gpu);
        close_rel(0.0, &want_dx.data, &got_dx.data, "dx");
        // Measured separation at B = 1/64/256: the degenerate input-gate slice sits at
        // 2-4e-4 of the largest bias gradient, the healthy ones at 0.05-0.14. Two
        // orders of gap, stable across batch, so 1e-3 skips exactly the one slice.
        // The bias gradients only exist once the backward above has run.
        let db_scale = (0..4)
            .flat_map(|gi| per_step.gate_db(&gpu, gi))
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        let dead = db_scale * 1e-3;
        for gi in 0..4 {
            // `dW` runs orders above the bias floor, so it is checked unconditionally.
            close_rel(
                0.0,
                &per_step.gate_dw(&gpu, gi),
                &fused.gate_dw(&gpu, gi),
                &format!("dW[{gi}]"),
            );
            close_rel(
                dead,
                &per_step.gate_db(&gpu, gi),
                &fused.gate_db(&gpu, gi),
                &format!("db[{gi}]"),
            );
        }
    }

    /// The batch-parallel fused forward (`slstm_batched_fwd`) must agree with the
    /// per-step path it replaces.
    ///
    /// The batch sizes are the encoder's real length buckets — a word of T characters
    /// arrives as B = ~2048/T rows — and T is deliberately short, which is the regime
    /// the scalar time-fused kernel refuses and this one is built for. Rows that do not
    /// fill the last 16-row mma tile are the interesting case: 120 and 341 are neither
    /// a multiple of the tile nor of a block's row range.
    ///
    /// The backward runs per-step in BOTH arms, so the gradients here check the saved
    /// slabs the fused forward wrote, not a second kernel.
    #[test]
    fn slstm_batched_matches_per_step() {
        for (b, t) in [(120usize, 16usize), (341, 5), (1024, 1), (682, 2), (64, 33)] {
            batched_matches_per_step_at(b, t, 256);
        }
    }

    fn batched_matches_per_step_at(b: usize, t: usize, h: usize) {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        if !gpu.kernels.has_coop || ops::slstm_batched_geometry(&gpu, h, b).is_none() {
            return;
        }
        let s = 1.0 / (h as f32).sqrt();
        let w: Vec<Tensor> = (0..4)
            .map(|g| Tensor::random_seeded(&[2 * h, h], s * (1.0 + g as f32 * 0.05), 11 + g as u64))
            .collect();
        let bi: Vec<Tensor> = (0..4)
            .map(|g| Tensor::random_seeded(&[h], 0.2 + g as f32 * 0.01, 31 + g as u64))
            .collect();
        let x = GTensor::from_host(&gpu, &Tensor::random_seeded(&[b, t, h], 0.5, 7));
        let gy = GTensor::from_host(&gpu, &Tensor::random_seeded(&[b, t, h], 0.7, 9));

        let build = |batched: bool| {
            let mut c = SLstm::from_parts(
                &gpu, h, h, &w[0], &w[1], &w[2], &w[3], &bi[0], &bi[1], &bi[2], &bi[3], None,
            );
            c.force_fused_time = Some(false);
            c.force_batched = Some(batched);
            c
        };
        let mut per_step = build(false);
        let y_per = per_step.forward_alloc(&gpu, &x);
        let want = y_per.to_host(&gpu);
        let mut batched = build(true);
        let y_bat = batched.forward_alloc(&gpu, &x);
        let got = y_bat.to_host(&gpu);

        // Both paths contract on bf16 operands — cuBLAS on a narrowed `h`/`Whr`, this
        // one on mma fragments of the same — so they agree to bf16's ~3 decimal digits,
        // not fp32's ~6.
        let rel_tol = 1e-2;
        for (i, (a, c)) in want.data.iter().zip(got.data.iter()).enumerate() {
            assert!(
                (a - c).abs() <= rel_tol * a.abs().max(1.0),
                "batched vs per-step forward diverged at {i} (B={b}, T={t}): {a} vs {c}"
            );
        }

        let close_rel = |want: &[f32], got: &[f32], what: &str| {
            let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            for (i, (a, c)) in want.iter().zip(got).enumerate() {
                assert!(
                    (a - c).abs() <= rel_tol * scale,
                    "{what} diverged at {i} (B={b}, T={t}): {a} vs {c} (scale {scale})"
                );
            }
        };
        let want_dx = per_step.backward_alloc(&gpu, &y_per, &gy).to_host(&gpu);
        let got_dx = batched.backward_alloc(&gpu, &y_bat, &gy).to_host(&gpu);
        close_rel(&want_dx.data, &got_dx.data, "dx");
        for gi in 0..4 {
            close_rel(
                &per_step.gate_dw(&gpu, gi),
                &batched.gate_dw(&gpu, gi),
                &format!("dW[{gi}]"),
            );
        }
    }

    /// The batched path must honour `carry` — the state a chunk hands the next one.
    ///
    /// Nothing in production reaches this yet (the encoder and decoder run one word per
    /// sequence, and the backbone, which does chunk, is at B=1 on the scalar path), but
    /// the forward implements it — step 0 of a carried sweep takes `h_{-1}` from
    /// `h_state` rather than from the mirror, which is a branch nothing else covers.
    ///
    /// The check is against the same chunked sweep on the per-step path rather than
    /// against an unchunked run, so a carry that is dropped or read from the wrong place
    /// shows up as a difference in the kernel and not in the chunking.
    #[test]
    fn slstm_batched_carry_matches_per_step() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let (b, h, chunks) = (64usize, 256usize, [6usize, 10]);
        if !gpu.kernels.has_coop || ops::slstm_batched_geometry(&gpu, h, b).is_none() {
            return;
        }
        let t: usize = chunks.iter().sum();
        let sc = 1.0 / (h as f32).sqrt();
        let w: Vec<Tensor> = (0..4)
            .map(|g| Tensor::random_seeded(&[2 * h, h], sc * (1.0 + g as f32 * 0.05), 41 + g as u64))
            .collect();
        let bi: Vec<Tensor> = (0..4)
            .map(|g| Tensor::random_seeded(&[h], 0.2, 61 + g as u64))
            .collect();
        let x = Tensor::random_seeded(&[b, t, h], 0.5, 13);
        // A time chunk of a `[B, T, H]` tensor is one row range per batch row, so this
        // gathers rather than slicing.
        let cut = |off: usize, len: usize| {
            let mut d = Vec::with_capacity(b * len * h);
            for r in 0..b {
                let s = (r * t + off) * h;
                d.extend_from_slice(&x.data[s..s + len * h]);
            }
            Tensor::new(&[b, len, h], d)
        };

        let sweep = |batched: bool| -> Vec<f32> {
            let mut cell = SLstm::from_parts(
                &gpu, h, h, &w[0], &w[1], &w[2], &w[3], &bi[0], &bi[1], &bi[2], &bi[3], None,
            );
            cell.force_fused_time = Some(false);
            cell.force_batched = Some(batched);
            cell.set_carry(true);
            cell.reset_state(&gpu);
            let mut y = Vec::with_capacity(b * t * h);
            let mut off = 0;
            for &len in &chunks {
                let xc = GTensor::from_host(&gpu, &cut(off, len));
                y.extend(cell.forward_alloc(&gpu, &xc).to_host(&gpu).data);
                off += len;
            }
            y
        };
        let (want, got) = (sweep(false), sweep(true));
        let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        for (i, (a, c)) in want.iter().zip(got.iter()).enumerate() {
            assert!(
                (a - c).abs() <= 1e-2 * scale,
                "carried forward diverged at {i}: {a} vs {c} (scale {scale})"
            );
        }
    }

    /// A chunked sweep with `carry` must reproduce the unchunked one.
    ///
    /// This is the load-bearing property of chunked training: splitting the sequence
    /// is supposed to change only *when* memory is live, never the arithmetic. Forward
    /// carries `h/c/n/m` left-to-right; backward carries the BPTT channels
    /// right-to-left, so the chunks unwind in reverse. The weight gradients accumulate
    /// across chunks, so comparing them checks the whole sweep at once rather than any
    /// single chunk.
    ///
    /// The tolerance is not zero: chunking re-associates the `dWx = xᵀ·dg` GEMM into
    /// per-chunk partial sums, and fp32 addition is not associative. What must hold is
    /// that the difference stays at rounding, not that the bits match.
    #[test]
    fn chunked_carry_matches_unchunked() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let (b, inp, h) = (1, 8, 12);
        let chunks = [48usize, 48, 32]; // ragged: the last chunk is a different length
        let t: usize = chunks.iter().sum();

        let cpu = CpuSLstm::new(inp, h);
        let x = Tensor::random(&[b, t, inp], 0.5);
        let dy = Tensor::random(&[b, t, h], 1.0);

        // Reference: one call over the whole sequence.
        let mut whole = from_cpu(&gpu, &cpu);
        let y_whole_dev = whole.forward_alloc(&gpu, &GTensor::from_host(&gpu, &x));
        let y_whole = y_whole_dev.to_host(&gpu);
        let dx_whole = whole
            .backward_alloc(&gpu, &y_whole_dev, &GTensor::from_host(&gpu, &dy))
            .to_host(&gpu);

        // Chunked: same weights, same input, one call per chunk.
        let mut part = from_cpu(&gpu, &cpu);
        part.set_carry(true);
        part.reset_state(&gpu);

        // `[B, T, F]` with B=1 means a time chunk is a contiguous row range, so the
        // slices are plain host sub-vectors — no device-side time slicing needed.
        let cut = |src: &Tensor, f: usize, off: usize, len: usize| {
            Tensor::new(&[b, len, f], src.data[off * f..(off + len) * f].to_vec())
        };

        let mut y_parts = Vec::with_capacity(t * h);
        let mut off = 0;
        for &c in &chunks {
            let xc = GTensor::from_host(&gpu, &cut(&x, inp, off, c));
            y_parts.extend(part.forward_alloc(&gpu, &xc).to_host(&gpu).data);
            off += c;
        }
        assert_close(&y_parts, &y_whole.data, 1e-4);

        // Backward runs the chunks in REVERSE, so the BPTT channels carry leftward.
        //
        // A chunk's backward reads the forward cache its own forward left behind, and
        // a cell holds exactly one such cache — so the chunks are re-forwarded here,
        // rightmost first, each from the state its predecessors produced. That is the
        // ordering a real chunked sweep has to arrange too (by keeping per-chunk
        // caches rather than replaying, which is what the offload path is for); the
        // point of the test is the arithmetic, not the scheduling.
        let mut ends: Vec<usize> = Vec::with_capacity(chunks.len());
        let mut acc = 0;
        for &c in &chunks {
            ends.push(acc);
            acc += c;
        }
        // Grads accumulate at beta=1 across chunks, so start from a clean slate: the
        // forward pass above touched none of them, but `whole` is a separate cell and
        // the comparison is against ITS single-call totals.
        part.zero_grad(&gpu);

        let mut dx_parts: Vec<Vec<f32>> = Vec::with_capacity(chunks.len());
        for (i, &c) in chunks.iter().enumerate().rev() {
            // Rebuild this chunk's cache: replay the recurrence from zero up to the
            // chunk's start, then forward the chunk itself. Only the last of those
            // forwards leaves the cache backward will read.
            part.set_carry(true);
            part.reset_state(&gpu);
            let mut o = 0;
            for &pc in &chunks[..i] {
                let xp = GTensor::from_host(&gpu, &cut(&x, inp, o, pc));
                part.forward_alloc(&gpu, &xp);
                o += pc;
            }
            let xc = GTensor::from_host(&gpu, &cut(&x, inp, ends[i], c));
            let yc = part.forward_alloc(&gpu, &xc);

            // The BPTT channels must carry from the chunk to the right, so they are
            // NOT reset here — except for the rightmost chunk, which starts at zero.
            if i + 1 == chunks.len() {
                part.reset_bptt(&gpu);
            }
            let dyc = GTensor::from_host(&gpu, &cut(&dy, h, ends[i], c));
            dx_parts.push(part.backward_alloc(&gpu, &yc, &dyc).to_host(&gpu).data);
        }
        dx_parts.reverse();
        let dx_flat: Vec<f32> = dx_parts.concat();
        assert_close(&dx_flat, &dx_whole.data, 1e-4);

        // The gate weight gradients summed over the chunks must equal the single
        // call's: chunking re-associates that sum, it does not change it.
        for gi in 0..4 {
            assert_close(&part.gate_dw(&gpu, gi), &whole.gate_dw(&gpu, gi), 1e-3);
            assert_close(&part.gate_db(&gpu, gi), &whole.gate_db(&gpu, gi), 1e-3);
        }
    }

    /// The GPU cell (`hg` training) against `nn::slstm` (`hs` inference) — the two
    /// implementations a checkpoint actually round-trips through. The other parity
    /// tests compare the GPU to `nn2`, which inference never runs, so this is the
    /// pair that has to agree. State is reset per word in the encoder/decoder, so
    /// the sequence start (`n_prev == 0`) is exercised on every call.
    #[test]
    fn gpu_matches_nn_slstm_inference_cell() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        use crate::nn::slstm::SLSTMLayer as InferSLstm;
        use crate::nn_layer::NnLayer;

        let (t, inp, h) = (64, 4, 6);
        let mut cpu = InferSLstm::new(inp, h);

        // `from_nn_cell` is the same converter the real GPU stack uses to load an
        // `nn::slstm` checkpoint, so the weights land exactly as training saw them.
        let mut dev = SLstm::from_nn_cell(&gpu, &cpu, None);
        dev.set_carry(true);
        dev.reset_state(&gpu);
        cpu.reset_state();

        let x = Tensor::random(&[1, t, inp], 0.5);
        let y_dev = dev
            .forward_alloc(&gpu, &GTensor::from_host(&gpu, &x))
            .to_host(&gpu)
            .data;

        // Drive the scalar CPU cell one step at a time over the same input, then
        // apply the post-cell norm the GPU cell owns internally (γ = 1, since
        // from_parts got None) so both sides end at the same point.
        let mut cache = cpu.alloc_cache();
        let mut norm = crate::nn2::rms_norm::RmsNorm::new(h);
        let mut y_cpu = Vec::with_capacity(t * h);
        for step in 0..t {
            let xt = &x.data[step * inp..(step + 1) * inp];
            cpu.forward(xt, &mut cache);
            let mut row = Tensor::zeros(&[1, h]);
            row.data.copy_from_slice(&cache.h);
            y_cpu.extend_from_slice(&norm.forward(&row).data);
        }

        assert_eq!(y_dev.len(), y_cpu.len());
        for (i, (g, c)) in y_dev.iter().zip(&y_cpu).enumerate() {
            assert!(
                (g - c).abs() < tol(&gpu, 8e-3),
                "step {}/unit {}: gpu {g} vs cpu {c}",
                i / h,
                i % h
            );
        }
    }

    /// The dispatch predicate is two-dimensional, and production depends on it landing
    /// the right way for shapes the model actually runs. `T` alone is not enough: the
    /// fused kernel's recurrent product is scalar and therefore linear in `B`, while the
    /// per-step path's cuBLAS call is flat until its tensor-core tiles fill, so a wide
    /// batch must NOT fuse (measured crossover B ~= 32).
    ///
    /// Correctness tests pin the math, not which path ran, so without this a dispatch
    /// regression is silent — and at B=8/T=512 it costs 29.32 ms against 16.62 ms.
    ///
    /// The same goes for the batched path at the other end of the batch axis: it is
    /// worth 1.03-1.67x on the encoder's narrow groups and a loss on its widest, and
    /// nothing else would notice it being taken at the wrong shape.
    #[test]
    fn dispatch_picks_the_right_path_for_production_shapes() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let cell = SLstm::new_rand(&gpu, 6, 6);

        // Backbone: one sequence, a whole chunk of words. Fuses.
        assert!(cell.fuses_at(1, crate::config::BACKBONE_CHUNK), "backbone must fuse");

        // Encoder/decoder: a word is at most MAX_WORD_BYTES + 1 steps, and the batch is
        // the whole length group. Must not fuse — on either count.
        for t in 1..=(crate::config::MAX_WORD_BYTES + 1) {
            for b in [120, 227, 512, 1024, crate::config::GROUP_MAX_ROWS] {
                assert!(!cell.fuses_at(b, t), "encoder shape B={b} T={t} must not fuse");
            }
        }

        // The batch axis alone decides it at a length that clears FUSED_MIN_T.
        assert!(cell.fuses_at(FUSED_MAX_B, 512), "B at the cap still fuses");
        assert!(!cell.fuses_at(FUSED_MAX_B + 1, 512), "one past the cap must not");

        // ...and the batched path is the other half of that split: it takes the
        // encoder/decoder groups the scalar one refuses, up to the batch where its
        // operand re-reads stop paying. At CHAR_HIDDEN the narrow groups must take it
        // and the widest must not — a regression either way is otherwise silent.
        let enc = SLstm::new_rand(&gpu, crate::config::CHAR_HIDDEN, crate::config::CHAR_HIDDEN);
        for b in [120, 128, 186] {
            assert!(enc.batches_at(&gpu, b), "encoder group B={b} must take the batched path");
        }
        for b in [292, 512, 1024, crate::config::GROUP_MAX_ROWS] {
            assert!(!enc.batches_at(&gpu, b), "B={b} re-reads too much to be worth it");
        }
        assert!(!enc.batches_at(&gpu, SB_MIN_B - 1), "a batch that cannot fill an mma tile");
    }
}
