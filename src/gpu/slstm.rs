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
//! state `(h,c,n,m)` stays resident in `DTensor`s across the entire T-loop** — no
//! per-step host transfer.
//!
//! The four gates run **fused**: a timestep is one cuBLAS GEMM plus one kernel.
//! That matters because the backbone runs this cell at batch 1 over ~2000 words,
//! where every launch is pure latency — the per-timestep GEMM there is a
//! matrix-vector product that takes far less time than the launch itself, so the
//! step cost is simply the number of launches.
//!
//! Concretely, per timestep `t`:
//!   * `x·Wx` for **all** timesteps is one GEMM hoisted out of the loop (it has no
//!     recurrent dependency), landing in a `[B, T, 4H]` gate buffer `g`;
//!   * the loop only adds the recurrent half, `g[:, t, :] += h_{t-1}·Wh`;
//!   * `slstm_step_fused` adds the biases and runs the elementwise recurrence.
//!
//! Backward mirrors it: the per-step kernel writes the four gate deltas back into
//! `g` (its forward contents are dead by then), the loop carries only the BPTT
//! channels — `dh = dg[:, t, :]·Whᵀ` — and `dx`, `dWx`, `dWh` and the bias grads
//! all fall out of three whole-sequence GEMMs plus one reduction *after* the loop.
//!
//! The fused operands **are** the parameters of record: `wx [in, 4H]`,
//! `whr [H, 4H]` and `bcat [4H]` are what the optimizer steps and what the grads
//! accumulate into, so no per-forward repacking happens at all. The four `[rows, H]`
//! gate matrices `nn2::SLstm` and the checkpoints use are a *serialization* layout,
//! converted on the host only in `from_parts` / `to_nn_cell` — once per checkpoint,
//! never per step. Gate order stays z=0, i=1, f=2, o=3: the column blocks of the
//! fused `[·, 4H]`, with the input rows above the recurrent rows in `[rows, H]`.

use cudarc::driver::CudaGraph;
use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};

use super::block::phase;
use super::ops::{self, SlstmSlabs};
use super::rms_norm::RmsNorm;
use super::{DTensor, Gpu};
use crate::nn2::optim::AdamCfg;
use crate::tensor::Tensor;

/// Below this sequence length the T-loop runs eagerly instead of as a captured
/// CUDA graph. Capturing costs one `cuGraphInstantiate` (hundreds of us), which a
/// short loop never earns back — and the encoder/decoder call this cell with T =
/// a word length (<= MAX_WORD_BYTES + 1), a shape that also changes from group to
/// group, so they would re-instantiate constantly. The backbone, which is where
/// the launch cost actually hurts (T = the window's word count, ~1000), captures.
const GRAPH_MIN_T: usize = 32;

/// A captured T-loop plus the shape it was captured at. A graph bakes in the
/// device pointer of every buffer its nodes touch, so it may only be replayed
/// when those buffers are still the same allocations — which is exactly when
/// `(b, t)` is unchanged, since that is what decides whether the activation
/// buffers below were reallocated.
struct LoopGraph {
    b: usize,
    t: usize,
    /// Device addresses of every buffer the capture baked in. A graph replays the
    /// pointers it was captured against, so a matching `(b, t)` is not enough: the
    /// chunked sweep hands each chunk its own `g`/`slabs` (see `chunk_saved`), and
    /// replaying across that swap reads the previous chunk's freed allocations.
    ptrs: Vec<u64>,
    graph: CudaGraph,
}

/// The device addresses a captured sLSTM loop depends on, in a fixed order.
fn loop_ptrs(gpu: &Gpu, g: &DTensor, slabs: &SlstmSlabs, dy: &DTensor) -> Vec<u64> {
    use cudarc::driver::DevicePtr;
    let f32_ptr = |t: &DTensor| t.buf.device_ptr(&gpu.stream).0;
    let slab_ptr = |s: &ops::SlabBuf| match s {
        ops::SlabBuf::F32(t) => t.buf.device_ptr(&gpu.stream).0,
        ops::SlabBuf::Bf16(t) => t.buf.device_ptr(&gpu.stream).0,
    };
    vec![
        f32_ptr(g),
        f32_ptr(dy),
        f32_ptr(&slabs.c_prev),
        f32_ptr(&slabs.n_prev),
        f32_ptr(&slabs.i_prime),
        f32_ptr(&slabs.f_prime),
        f32_ptr(&slabs.c),
        f32_ptr(&slabs.n),
        slab_ptr(&slabs.zt),
        slab_ptr(&slabs.ot),
        slab_ptr(&slabs.h_prev),
    ]
}

/// `GPU_NO_GRAPH=1` forces the eager per-timestep launch path — the A/B baseline
/// for `slstm_launch_bench`, and the fallback if a driver ever mis-captures.
fn graphs_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("GPU_NO_GRAPH").is_ok())
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
    pub wx: DTensor,   // [in, 4H]   input half
    pub whr: DTensor,  // [H, 4H]    recurrent half
    pub bcat: DTensor, // [4H]
    dwx: DTensor,
    dwhr: DTensor,
    dbcat: DTensor,
    mwx: DTensor,
    vwx: DTensor,
    mwhr: DTensor,
    vwhr: DTensor,
    mbcat: DTensor,
    vbcat: DTensor,

    /// Per-instance override of the time-fused path. `None` follows the global
    /// default (see [`fused_time_enabled`]); `Some(false)` pins this cell to the
    /// per-step loop and `Some(true)` to the fused one. The env flag is read through
    /// a `OnceLock`, so a test that needs BOTH paths in one process sets this — and
    /// it must set it on both cells, since the default alone no longer distinguishes
    /// them.
    pub force_fused_time: Option<bool>,

    /// Continue the previous call's recurrence instead of starting from zero.
    ///
    /// Off by default: a `forward` is a whole sequence, and its state is private to
    /// the call. Set for a **chunked** sweep, where one sequence is split across
    /// several calls and the state has to cross the chunk borders — forward carries
    /// `h/c/n/m`, backward carries the BPTT channels (in reverse chunk order). See
    /// [`set_carry`](Self::set_carry).
    carry: bool,

    // Recurrent state carried across timesteps within one call, [B, H].
    h_state: DTensor,
    c_state: DTensor,
    n_state: DTensor,
    m_state: DTensor,
    /// Contiguous `[B, 4H]` scratch for the current timestep's recurrent gate half
    /// (`h_{t-1}·Wh` forward, the gate deltas backward). It exists so both of those
    /// GEMMs stay dense at any batch size — see `slstm_step_fused` in `kernels.rs`.
    gh: DTensor,
    /// `[1, N]` of ones: the bias gradient is the column sum of the gate deltas,
    /// which cuBLAS reduces as a `ones · dgates` GEMM straight into `dbcat`.
    ones: DTensor,
    // BPTT channels, [B, H].
    dh_bptt: DTensor,
    dc_bptt: DTensor,
    dn_bptt: DTensor,

    // Handed from forward to backward: the gate buffer [B, T, 4H], the saved
    // [B, T, H] slabs, and the flattened input [B·T, in] (needed for dWx).
    //
    // These are *reused* across calls rather than reallocated, and `out` / `dy_buf`
    // exist for the same reason: a captured graph's nodes hold raw device pointers,
    // so replaying it is only correct if every buffer the loop touches is still at
    // the address it had at capture time. `take_uninit` keeps the allocation
    // whenever the shape matches, and a shape that matches is precisely a shape the
    // graph cache hits — the two conditions cannot drift apart.
    g: Option<DTensor>,
    slabs: Option<SlstmSlabs>,
    x_saved: Option<DTensor>,
    out_buf: Option<DTensor>,
    /// Forward caches of earlier chunks of a chunked sweep, oldest first.
    ///
    /// The buffers above are reused call to call to keep the graphs' device pointers
    /// valid, which is exactly what a chunked sweep cannot have: chunk c+1's forward
    /// would overwrite what chunk c's backward reads. So each chunk's `(g, slabs,
    /// x_saved)` is moved aside here when the next chunk's forward takes fresh
    /// buffers, and backward pops them right to left. Only what backward *reads*
    /// moves; `out_buf`/`dy_buf` are written through and stay stable, so replay
    /// survives.
    ///
    /// Empty on the unchunked path, where the reuse above is untouched.
    chunk_saved: Vec<SlstmChunk>,
    /// Backward's incoming `dy`, copied into a stable buffer: the caller hands us a
    /// fresh `DTensor` every time, whose pointer a graph cannot depend on.
    dy_buf: Option<DTensor>,
    /// Scratch for widening a bf16 `h_prev` slab back to fp32 for the `dWh` GEMM
    /// (cuBLAS has no bf16 operand here). Unused on the fp32 slab path.
    h_prev_f32: Option<DTensor>,
    /// The `(b, t)` the buffers above are currently allocated for. A captured graph
    /// is only valid for the allocation it was captured against, so the graphs are
    /// dropped whenever this changes — see [`Self::forward`].
    buf_shape: Option<(usize, usize)>,
    fwd_graph: Option<LoopGraph>,
    bwd_graph: Option<LoopGraph>,
    batch: usize,
}

/// One chunk's forward cache, set aside so a later chunk's forward can take fresh
/// buffers without destroying it. See [`SLstm::chunk_saved`].
struct SlstmChunk {
    g: DTensor,
    slabs: SlstmSlabs,
    x_saved: DTensor,
}

/// Keep `slot`'s buffer when it already has the wanted shape, else allocate a
/// fresh (uninitialised) one. The reuse is what makes the device pointers stable
/// across calls, which is what makes graph replay legal.
fn take_uninit(gpu: &Gpu, slot: Option<DTensor>, dims: &[usize]) -> DTensor {
    match slot {
        Some(t) if t.dims() == dims => t,
        _ => DTensor::uninit(gpu, dims),
    }
}

/// Reuse `t`'s device buffer when the shape matches, zeroing it in place; else
/// (re)allocate a zeroed buffer. For state / BPTT channels that must start at 0.
fn fit_zeros(gpu: &Gpu, t: &mut DTensor, dims: &[usize]) {
    if t.dims() == dims {
        t.zero_(gpu);
    } else {
        *t = DTensor::zeros(gpu, dims);
    }
}

/// Reuse `t`'s device buffer when the shape matches (leaving its contents); else
/// (re)allocate uninitialised. For outputs a kernel/GEMM overwrites in full.
fn fit_uninit(gpu: &Gpu, t: &mut DTensor, dims: &[usize]) {
    if t.dims() != dims {
        *t = DTensor::uninit(gpu, dims);
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

        let up = |d: Vec<f32>, dims: &[usize]| DTensor::from_host(gpu, &Tensor::new(dims, d));
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
            dwx: DTensor::zeros(gpu, &[input, h4]),
            dwhr: DTensor::zeros(gpu, &[h, h4]),
            dbcat: DTensor::zeros(gpu, &[h4]),
            mwx: DTensor::zeros(gpu, &[input, h4]),
            vwx: DTensor::zeros(gpu, &[input, h4]),
            mwhr: DTensor::zeros(gpu, &[h, h4]),
            vwhr: DTensor::zeros(gpu, &[h, h4]),
            mbcat: DTensor::zeros(gpu, &[h4]),
            vbcat: DTensor::zeros(gpu, &[h4]),
            force_fused_time: None,
            carry: false,
            h_state: DTensor::zeros(gpu, &[0, 0]),
            c_state: DTensor::zeros(gpu, &[0, 0]),
            n_state: DTensor::zeros(gpu, &[0, 0]),
            m_state: DTensor::zeros(gpu, &[0, 0]),
            gh: DTensor::zeros(gpu, &[0, 0]),
            ones: DTensor::zeros(gpu, &[0, 0]),
            dh_bptt: DTensor::zeros(gpu, &[0, 0]),
            dc_bptt: DTensor::zeros(gpu, &[0, 0]),
            dn_bptt: DTensor::zeros(gpu, &[0, 0]),
            g: None,
            slabs: None,
            x_saved: None,
            out_buf: None,
            dy_buf: None,
            h_prev_f32: None,
            buf_shape: None,
            chunk_saved: Vec::new(),
            fwd_graph: None,
            bwd_graph: None,
            batch: 0,
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
    pub fn post_norm_gamma(&self) -> &DTensor {
        &self.post_norm.gamma
    }

    /// Forward over a whole `[B, T, in]` sequence into `y` `[B, T, H]`. State
    /// resets to zero at t=0 and stays device-resident across the T-loop.
    pub fn forward(&mut self, gpu: &Gpu, x: &DTensor, y: &mut DTensor) {
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

        // A captured graph holds raw device pointers, so it is bound to the exact
        // ALLOCATIONS it was captured against — not merely to a shape. Every buffer
        // below is refit (and thus possibly reallocated) whenever `(b, t)` changes,
        // so that is the moment both graphs die.
        //
        // This has to happen here, unconditionally, rather than inside `fwd_loop`:
        // a window shorter than GRAPH_MIN_T takes the eager path and never consults
        // the cache, but it still refits the buffers underneath it. Skipping the
        // invalidation there let a long window -> short window -> long window
        // sequence (which the dataset produces constantly, since windows never cross
        // document borders) find a cached graph whose `(b, t)` matched again while
        // its nodes pointed at memory the short window had already handed back to
        // the pool. That is a use-after-free on the device: it shows up as an
        // illegal access, and then as a sticky CUBLAS_STATUS_EXECUTION_FAILED on
        // whatever GEMM runs next.
        if self.buf_shape != Some((b, t)) {
            self.fwd_graph = None;
            self.bwd_graph = None;
            self.buf_shape = Some((b, t));
        }

        // `wx`/`whr`/`bcat` are the parameters themselves — already in the layout the
        // GEMMs below want, so there is nothing to pack here.

        // Recurrent state starts at zero — unless this call continues a sequence the
        // previous call left off (see `set_carry`), where it starts at whatever that
        // call ended with. Carrying is what makes a chunked sweep reproduce the
        // unchunked recurrence exactly rather than resetting at every chunk border.
        //
        // A shape change forces zeros regardless: a carried state is only meaningful
        // for the batch it was produced at, and `fit_zeros` would have reallocated it
        // anyway.
        let carry = self.carry && self.h_state.dims() == [b, h];
        for s in [
            &mut self.h_state,
            &mut self.c_state,
            &mut self.n_state,
            &mut self.m_state,
        ] {
            if !carry {
                fit_zeros(gpu, s, &[b, h]);
            }
        }

        // A chunked sweep continues the recurrence (`carry`) and forwards every chunk
        // before unwinding any, so the previous chunk's cache is still owed a backward:
        // set it aside instead of letting `take_uninit` hand its buffers to this chunk.
        // Unchunked, there is nothing to preserve and the buffers are reused as before.
        if carry {
            if let (Some(g), Some(slabs), Some(x_saved)) =
                (self.g.take(), self.slabs.take(), self.x_saved.take())
            {
                self.chunk_saved.push(SlstmChunk { g, slabs, x_saved });
            }
        }

        // The input half of every gate pre-activation, for all timesteps at once —
        // it has no recurrent dependency, so it is one GEMM outside the loop.
        let mut x_flat = take_uninit(gpu, self.x_saved.take(), &[n, inp]);
        // The copy is for LIFETIME, not layout: `backward` needs `x` again for
        // `dWx = xᵀ·dg`, and by then the caller has returned its buffer to the pool
        // and someone else owns that memory. `x_flat` is the cell's own copy, held
        // across the forward→backward boundary as `x_saved`.
        //
        // Slice both sides because `x` may be a pooled buffer larger than [B, T, in]
        // (`Buf`/`Pool` reuse by capacity) while `x_flat` is exactly [N, in]:
        // `memcpy_dtod` asserts dst >= src, so copying the raw `buf`s would trip on
        // any oversized input — and would move capacity rather than content.
        let n_x = x_flat.len();
        phase::timed(gpu, phase::Bucket::SlstmCopyFwd, || {
            gpu.stream
                .memcpy_dtod(&x.buf.slice(..n_x), &mut x_flat.buf.slice_mut(..n_x))
                .expect("copy x");
        });
        // One buffer, two views: the GEMM wants [N, 4H], the time loop wants
        // [B, T, 4H]. `reshaped` is metadata-only, so the allocation is untouched.
        let mut g = take_uninit(gpu, self.g.take(), &[b, t, h4]).reshaped(&[n, h4]);
        phase::timed(gpu, phase::Bucket::SlstmGemmFwd, || {
            ops::matmul_nn_into(gpu, &x_flat, &self.wx, &mut g, 0.0);
        });
        let mut g = g.reshaped(&[b, t, h4]);

        let mut slabs = match self.slabs.take() {
            Some(s) if s.c.dims() == [b, t, h].as_slice() => s,
            _ => {
                // fp32 for the stabilizer-carrying slabs, kernel-matched width for
                // the plain activations — see `SlstmSlabs` and `gpu::bf16`.
                let f32_slab = || DTensor::uninit(gpu, &[b, t, h]);
                let act_slab = || ops::SlabBuf::new(gpu, &[b, t, h]);
                SlstmSlabs {
                    c_prev: f32_slab(),
                    n_prev: f32_slab(),
                    i_prime: f32_slab(),
                    f_prime: f32_slab(),
                    c: f32_slab(),
                    n: f32_slab(),
                    zt: act_slab(),
                    ot: act_slab(),
                    h_prev: act_slab(),
                }
            }
        };
        let mut out = take_uninit(gpu, self.out_buf.take(), &[b, t, h]);
        fit_uninit(gpu, &mut self.gh, &[b, h4]);

        phase::timed(gpu, phase::Bucket::SlstmLoopFwd, || {
            self.fwd_loop(gpu, &mut g, &mut slabs, &mut out, b, t);
        });

        // `out` is the graph's write target and must keep its address, so the loop
        // writes there and the result reaches the caller's buffer through the
        // post-cell norm — which is also what moves it, so there is no separate copy.
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

    /// The forward time loop: replayed from a captured CUDA graph when T is long
    /// enough to be worth it, else issued step by step.
    ///
    /// At the backbone's shape (B=1, H=512) a timestep is a `[1,512]x[512,2048]`
    /// matvec — ~2 us of GPU work — while *submitting* it costs the host 18 us for
    /// the cuBLAS call plus 7 us for the kernel. The card therefore idles waiting on
    /// the driver, and no faster card can help. Capturing the loop once and replaying
    /// it turns those 2·T submissions into a single `cuGraphLaunch`.
    fn fwd_loop(
        &mut self,
        gpu: &Gpu,
        g: &mut DTensor,
        slabs: &mut SlstmSlabs,
        out: &mut DTensor,
        b: usize,
        t: usize,
    ) {
        // The time-fused kernel replaces the whole loop with one cooperative launch,
        // so it comes before the graph path (a cooperative launch cannot be stream
        // captured anyway) and before the eager path. It declines by returning false
        // when the shape does not fit, leaving both fallbacks intact.
        if self.force_fused_time.unwrap_or_else(fused_time_enabled)
            && t >= GRAPH_MIN_T
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
            )
        {
            return;
        }
        if t < GRAPH_MIN_T || graphs_disabled() {
            self.fwd_steps(gpu, g, slabs, out, t);
            return;
        }
        let ptrs = loop_ptrs(gpu, g, slabs, out);
        if self
            .fwd_graph
            .as_ref()
            .map_or(true, |c| (c.b, c.t) != (b, t) || c.ptrs != ptrs)
        {
            // Drop the stale exec first: its nodes point into buffers that the shape
            // change above has just reallocated.
            self.fwd_graph = None;
            gpu.stream
                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .expect("begin capture");
            self.fwd_steps(gpu, g, slabs, out, t);
            // Capture records the launches instead of running them, so the recurrent
            // state is untouched here and the `launch` below is what executes them.
            //
            // AUTO_FREE_ON_LAUNCH is the only flag cudarc's enum exposes (it has no
            // zero variant); it only concerns memory *allocated by graph nodes*, and
            // the loop allocates nothing, so it is a no-op for us.
            let graph = gpu
                .stream
                .end_capture(
                    CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )
                .expect("end capture")
                .expect("stream was not capturing");
            self.fwd_graph = Some(LoopGraph { b, t, ptrs, graph });
        }
        self.fwd_graph
            .as_ref()
            .unwrap()
            .graph
            .launch()
            .expect("graph launch");
    }

    /// The loop body, issued eagerly. Called both to run the steps and — under
    /// stream capture — to record them into a graph.
    fn fwd_steps(
        &mut self,
        gpu: &Gpu,
        g: &mut DTensor,
        slabs: &mut SlstmSlabs,
        out: &mut DTensor,
        t: usize,
    ) {
        for step in 0..t {
            // Recurrent half of the gates (one dense GEMM into the contiguous
            // scratch), then the elementwise recurrence: two launches per timestep.
            // The bf16 variant adds a third (the h round-trip) but halves the Wh
            // traffic, which at B=1 is what the GEMM's time is actually made of.
            ops::matmul_nn_into(gpu, &self.h_state, &self.whr, &mut self.gh, 0.0);
            ops::slstm_step_fused(
                gpu,
                g,
                &self.gh,
                &self.bcat,
                &mut self.c_state,
                &mut self.n_state,
                &mut self.m_state,
                &mut self.h_state,
                slabs,
                out,
                step,
            );
        }
    }

    /// Forward into a freshly allocated `[B, T, H]` — the by-value companion to
    /// [`forward`](Self::forward), used by tests and one-shot call sites.
    pub fn forward_alloc(&mut self, gpu: &Gpu, x: &DTensor) -> DTensor {
        let mut y: DTensor = DTensor::uninit(gpu, &[x.shape[0], x.shape[1], self.hidden]);
        self.forward(gpu, x, &mut y);
        y
    }

    /// Backward into a freshly allocated `dx` `[B, T, in]`.
    pub fn backward_alloc(&mut self, gpu: &Gpu, dy: &DTensor) -> DTensor {
        let mut dx = DTensor::uninit(gpu, &[dy.shape[0], dy.shape[1], self.input]);
        self.backward(gpu, dy, &mut dx);
        dx
    }

    /// Backward over the whole sequence. `dy` is `[B, T, H]`, `dx` is the
    /// caller's `[B, T, in]` output. Accumulates weight/bias grads.
    pub fn backward(&mut self, gpu: &Gpu, dy: &DTensor, dx: &mut DTensor) {
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
        fit_uninit(gpu, &mut self.gh, &[b, h4]);

        // The post-cell norm is the last thing forward applied, so it is the first
        // thing to undo: its `dx` is what the recurrence actually receives.
        //
        // It lands directly in `dy_buf` — the loop reads that every step, and the
        // caller hands us a different `dy` each time, a pointer a captured graph
        // cannot follow. Undoing the norm into our own buffer therefore costs nothing
        // extra: it replaces the copy that was there for the same reason.
        let mut dy_buf = take_uninit(gpu, self.dy_buf.take(), &[b, t, h]);
        phase::timed(gpu, phase::Bucket::SlstmCopyBwd, || {
            self.post_norm.backward(gpu, dy, &mut dy_buf);
        });

        // The only thing the loop must carry is BPTT: the gate deltas go straight
        // back into `g`, and everything derived from them waits until the loop ends.
        phase::timed(gpu, phase::Bucket::SlstmLoopBwd, || {
            self.bwd_loop(gpu, &dy_buf, &mut g, &slabs, b, t);
        });
        self.dy_buf = Some(dy_buf);

        // `g` now holds the gate deltas for the whole sequence: dx, dWx, dWh and the
        // bias grads are three GEMMs and one reduction over it.
        let dg = g.reshaped(&[n, h4]);
        dx.reshape_to(&[n, inp]);
        let wx = &self.wx;
        phase::timed(gpu, phase::Bucket::SlstmGemmBwd, || {
            ops::matmul_nt_into(gpu, &dg, wx, dx, 0.0);
        });

        // The gate grads land in the parameter layout directly, so these GEMMs write
        // into `dwx`/`dwhr` with beta = 1: cuBLAS does the accumulation across
        // windows that the unpack kernel used to do by hand.
        let dwx = &mut self.dwx;
        phase::timed(gpu, phase::Bucket::SlstmGemmBwd, || {
            ops::matmul_tn_into(gpu, &x_flat, &dg, dwx, 1.0);
        });
        // dWh = h_prevᵀ · dg goes through cuBLAS, which needs an fp32 operand, so a
        // bf16 `h_prev` slab is widened into reusable scratch first. The scratch is
        // transient (one GEMM) while the slab it replaced was pinned across the whole
        // forward AND backward, so this still gives memory back.
        slabs.h_prev.shrink_to(&[n, h]);
        let dwhr = &mut self.dwhr;
        let h_prev_f32 = &mut self.h_prev_f32;
        phase::timed(gpu, phase::Bucket::SlstmGemmBwd, || match &slabs.h_prev {
            ops::SlabBuf::F32(t) => ops::matmul_tn_into(gpu, t, &dg, dwhr, 1.0),
            ops::SlabBuf::Bf16(b16) => {
                let mut scratch = take_uninit(gpu, h_prev_f32.take(), &[n, h]);
                b16.load(gpu, &mut scratch);
                ops::matmul_tn_into(gpu, &scratch, &dg, dwhr, 1.0);
                *h_prev_f32 = Some(scratch);
            }
        });
        slabs.h_prev.shrink_to(&[b, t, h]);

        // The bias gradient is the column sum of the gate deltas — a `ones · dg` GEMM,
        // accumulating straight into the fused `dbcat` (viewed as the [1, 4H] row it
        // is). Nothing to scatter afterwards.
        fit_uninit(gpu, &mut self.ones, &[1, n]);
        ops::fill(gpu, &mut self.ones, 1.0);
        let mut dbcat =
            std::mem::replace(&mut self.dbcat, DTensor::zeros(gpu, &[0])).reshaped(&[1, h4]);
        let ones = &self.ones;
        phase::timed(gpu, phase::Bucket::SlstmGemmBwd, || {
            ops::matmul_nn_into(gpu, ones, &dg, &mut dbcat, 1.0);
        });
        self.dbcat = dbcat.reshaped(&[h4]);

        // Give the buffers back (same allocations, original shapes) so the next
        // forward reuses them — and so the captured graphs stay valid.
        self.g = Some(dg.reshaped(&[b, t, h4]));
        self.slabs = Some(slabs);
        self.x_saved = Some(x_flat);

        // Chunked sweep: this chunk is done, so the chunk to its left — the next one
        // to unwind — takes the live slots. Its buffers are the ones its own forward
        // wrote, so backward reads exactly what that chunk produced. Dropping what was
        // just handed back releases this chunk's activations now rather than at the
        // next forward, which is what keeps only the chunks still owed a backward
        // resident.
        if let Some(prev) = self.chunk_saved.pop() {
            self.g = Some(prev.g);
            self.slabs = Some(prev.slabs);
            self.x_saved = Some(prev.x_saved);
        }

        dx.reshape_to(&[b, t, inp]);
    }

    /// The backward time loop — graph-replayed on the same terms as [`Self::fwd_loop`].
    fn bwd_loop(
        &mut self,
        gpu: &Gpu,
        dy: &DTensor,
        g: &mut DTensor,
        slabs: &SlstmSlabs,
        b: usize,
        t: usize,
    ) {
        // One cooperative launch for the whole reverse loop; see `fwd_loop` for why
        // this precedes the graph path. `gh` doubles as the grid-visible [B, 4H]
        // gate-delta scratch, which is exactly what the per-step path uses it for.
        if self.force_fused_time.unwrap_or_else(fused_time_enabled)
            && t >= GRAPH_MIN_T
            && ops::slstm_fused_time_bwd(
                gpu,
                &self.whr,
                dy,
                g,
                &mut self.gh,
                &mut self.dh_bptt,
                &mut self.dc_bptt,
                &mut self.dn_bptt,
                slabs,
                t,
            )
        {
            return;
        }
        if t < GRAPH_MIN_T || graphs_disabled() {
            self.bwd_steps(gpu, dy, g, slabs, t);
            return;
        }
        let ptrs = loop_ptrs(gpu, g, slabs, dy);
        if self
            .bwd_graph
            .as_ref()
            .map_or(true, |c| (c.b, c.t) != (b, t) || c.ptrs != ptrs)
        {
            self.bwd_graph = None;
            gpu.stream
                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .expect("begin capture");
            self.bwd_steps(gpu, dy, g, slabs, t);
            let graph = gpu
                .stream
                .end_capture(
                    CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )
                .expect("end capture")
                .expect("stream was not capturing");
            self.bwd_graph = Some(LoopGraph { b, t, ptrs, graph });
        }
        self.bwd_graph
            .as_ref()
            .unwrap()
            .graph
            .launch()
            .expect("graph launch");
    }

    fn bwd_steps(
        &mut self,
        gpu: &Gpu,
        dy: &DTensor,
        g: &mut DTensor,
        slabs: &SlstmSlabs,
        t: usize,
    ) {
        for step in (0..t).rev() {
            ops::slstm_step_fused_bwd(
                gpu,
                dy,
                g,
                &mut self.gh,
                &self.dh_bptt,
                slabs,
                &mut self.dc_bptt,
                &mut self.dn_bptt,
                step,
            );
            // dh_{t-1} = dgates_t · Whᵀ — the one gradient BPTT cannot defer.
            ops::matmul_nt_into(gpu, &self.gh, &self.whr, &mut self.dh_bptt, 0.0);
        }
    }

    /// Every learnable tensor, in a fixed order (used by checkpoint save/load).
    /// The post-cell norm's γ comes last, where the enclosing block used to emit it.
    pub fn params_mut(&mut self) -> Vec<&mut DTensor> {
        let mut v = vec![&mut self.wx, &mut self.whr, &mut self.bcat];
        v.extend(self.post_norm.params_mut());
        v
    }

    /// Gradient accumulators, in the same order as `params_mut`. Diagnostic.
    pub fn grads(&self) -> Vec<&DTensor> {
        vec![&self.dwx, &self.dwhr, &self.dbcat, &self.post_norm.dgamma]
    }

    /// Forward-cache extremes of the last sweep: `(min |n|, max |c|, max |c/n|)`.
    ///
    /// The backward divides by `n` and by `n²`, so a normalizer that collapses is the
    /// difference between a finite gradient and a NaN. Diagnostic — downloads the
    /// whole `[B, T, H]` slabs, so it belongs in a probe.
    pub fn state_extremes(&self, gpu: &Gpu) -> Option<(f32, f32, f32)> {
        let slabs = self.slabs.as_ref()?;
        let n = slabs.n.to_host(gpu).data.to_vec();
        let c = slabs.c.to_host(gpu).data.to_vec();
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
    }

    /// Zero the carried **BPTT** channels, so the next `backward` starts with no
    /// incoming gradient from the right. Call before the rightmost chunk's backward.
    pub fn reset_bptt(&mut self, gpu: &Gpu) {
        for s in [
            &mut self.dh_bptt,
            &mut self.dc_bptt,
            &mut self.dn_bptt,
        ] {
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
            + self.x_saved.as_ref().map_or(0, |t| t.capacity() * 4)
            + self
                .chunk_saved
                .iter()
                .map(|c| {
                    c.slabs.retained_bytes()
                        + c.x_saved.capacity() * 4
                        + c.g.capacity() * 4
                })
                .sum::<usize>();
        let (_, all) = self.retained_bytes();
        (saved, all - saved)
    }

    /// Release every activation this cell holds — the saved slabs and input, the
    /// gate/output/dy buffers and the widening scratch.
    ///
    /// Broader than [`drop_saved_act`](Self::drop_saved_act), which keeps the stable
    /// buffers on purpose: their addresses are what a captured graph replays against.
    /// Dropping them invalidates the graphs, so this also clears those — the next
    /// forward recaptures. For a window boundary, not the hot path.
    pub fn drop_all_act(&mut self) {
        // The big per-`[B, T, ·]` buffers: these are what scale with the rectangle and
        // what a group boundary needs back.
        self.slabs = None;
        self.x_saved = None;
        self.chunk_saved.clear();
        self.post_norm.drop_saved_act();

        // `g`, `out_buf`, `dy_buf`, `h_prev_f32` and the two graphs are deliberately
        // KEPT.
        //
        // A captured graph bakes in the raw device pointers of every buffer its nodes
        // touch, so dropping those buffers means dropping the graphs, and the next call
        // at the same `(b, t)` has to re-capture. That is not a rare event: the
        // encoder and decoder run one rectangle per length bucket and the buckets
        // repeat window after window, so the graphs are hit constantly — clearing them
        // per group turned every one of those hits into a re-capture and roughly halved
        // the step rate.
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
    /// are kept deliberately, because a captured graph holds their raw pointers.
    pub fn retained_bytes(&self) -> (usize, usize) {
        let params: usize = [
            &self.wx, &self.whr, &self.bcat, &self.dwx, &self.dwhr, &self.dbcat, &self.mwx,
            &self.vwx, &self.mwhr, &self.vwhr, &self.mbcat, &self.vbcat,
        ]
        .iter()
        .map(|t| t.capacity() * 4)
        .sum();
        let opt: usize = [
            &self.g,
            &self.x_saved,
            &self.out_buf,
            &self.dy_buf,
            &self.h_prev_f32,
        ]
        .iter()
        .filter_map(|s| s.as_ref())
        .map(|t| t.capacity() * 4)
        .sum();
        let live: usize = [
            &self.h_state,
            &self.c_state,
            &self.n_state,
            &self.m_state,
            &self.gh,
            &self.ones,
            &self.dh_bptt,
            &self.dc_bptt,
            &self.dn_bptt,
        ]
        .iter()
        .map(|t| t.capacity() * 4)
        .sum();
        let slabs = self.slabs.as_ref().map_or(0, |s| s.retained_bytes());
        let (pn_p, _) = self.post_norm.retained_bytes();
        (params + pn_p, opt + live + slabs)
    }

    pub fn zero_grad(&mut self, gpu: &Gpu) {
        for g in [&mut self.dwx, &mut self.dwhr, &mut self.dbcat] {
            g.zero_(gpu);
        }
        self.post_norm.zero_grad(gpu);
    }

    /// AdamW step: gate matrices decay, biases don't. Clears the grads.
    ///
    /// AdamW is elementwise, so stepping the fused `[in, 4H]` / `[H, 4H]` operands is
    /// numerically identical to stepping the four `[rows, H]` gate matrices they hold
    /// — the decay/no-decay split that actually matters is weights vs. bias, and that
    /// survives the fusion because `bcat` is still its own tensor.
    pub fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        self.step_q(gpu, cfg, None);
    }

    /// [`step`](Self::step), optionally queueing instead of launching.
    pub fn step_q(&mut self, gpu: &Gpu, cfg: &AdamCfg, q: Option<&mut ops::AdamwQueue>) {
        if let Some(q) = q {
            q.push(gpu, &mut self.wx, &self.dwx, &mut self.mwx, &mut self.vwx, cfg, true);
            q.push(gpu, &mut self.whr, &self.dwhr, &mut self.mwhr, &mut self.vwhr, cfg, true);
            q.push(gpu, &mut self.bcat, &self.dbcat, &mut self.mbcat, &mut self.vbcat, cfg, false);
            self.post_norm.step_q(gpu, cfg, Some(q));
            return;
        }
        ops::adamw(
            gpu,
            &mut self.wx,
            &self.dwx,
            &mut self.mwx,
            &mut self.vwx,
            cfg,
            true,
        );
        ops::adamw(
            gpu,
            &mut self.whr,
            &self.dwhr,
            &mut self.mwhr,
            &mut self.vwhr,
            cfg,
            true,
        );
        ops::adamw(
            gpu,
            &mut self.bcat,
            &self.dbcat,
            &mut self.mbcat,
            &mut self.vbcat,
            cfg,
            false,
        );
        // Before `zero_grad` below, which would otherwise clear dγ unstepped.
        self.post_norm.step(gpu, cfg);
        self.zero_grad(gpu);
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
        // run to run. Ten measured runs of `slstm_graph_path_matches_cpu` at T=64
        // spread 2.2e-2 to 5.9e-2 on `dx`, so x32 (6.4e-2) was back to clearing by a
        // hair and failing intermittently. x128 (2.6e-1) sits ~4x above the observed
        // maximum, which is the margin this bound needs to be worth having.
        //
        // That this is bf16 and not a logic error is directly checkable:
        // `GPU_NO_BF16=1` makes both graph tests pass at the base tolerance, two
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
        let y_dev = dev.forward_alloc(&gpu, &DTensor::from_host(&gpu, &x));
        assert_close(&y_dev.to_host(&gpu).data, &y_cpu.data, tol(&gpu, 2e-3));

        // Backward
        let dx_cpu = cpu.backward(&g);
        let dx_dev = dev.backward_alloc(&gpu, &DTensor::from_host(&gpu, &g));
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

    /// The same parity check, but at `T > GRAPH_MIN_T` so the time loops run as
    /// **captured CUDA graphs** — the path `slstm_matches_cpu_layer` (T=5) never
    /// reaches. This is what pins the graph rewrite: the buffers a graph's nodes
    /// point at are now reused across calls, so a stale pointer or a buffer that
    /// silently moved would show up here as a numeric mismatch.
    ///
    /// Two full cycles, checked separately: the first captures and instantiates,
    /// the second **replays**. A replay reading a wrong address is the failure mode
    /// that a single-pass test would miss entirely.
    #[test]
    fn slstm_graph_path_matches_cpu() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        assert!(64 > GRAPH_MIN_T, "this test must exercise the graph path");
        let (b, t, inp, h) = (2, 64, 8, 12);

        let mut cpu = CpuRef::new(inp, h);
        let mut dev = from_cpu(&gpu, &cpu.cell);
        let mut cfg = AdamCfg::new(1e-3, 0.01);

        for pass in 0..2 {
            let x = Tensor::random(&[b, t, inp], 0.5);
            let g = Tensor::random(&[b, t, h], 1.0);

            let y_cpu = cpu.forward(&x);
            let y_dev = dev.forward_alloc(&gpu, &DTensor::from_host(&gpu, &x));
            assert_close(&y_dev.to_host(&gpu).data, &y_cpu.data, tol(&gpu, 2e-3));

            let dx_cpu = cpu.backward(&g);
            let dx_dev = dev.backward_alloc(&gpu, &DTensor::from_host(&gpu, &g));
            assert_close(&dx_dev.to_host(&gpu).data, &dx_cpu.data, tol(&gpu, 2e-3));
            assert_close(&dev.gate_dw(&gpu, 0), &cpu.cell.dwz.data, tol(&gpu, 2e-3));
            assert_close(&dev.gate_dw(&gpu, 2), &cpu.cell.dwf.data, tol(&gpu, 2e-3));

            // Step between passes, so the replay runs against *changed* weights —
            // the graph must read the packed operands live, not a stale copy.
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

    /// A cell whose shape changes must not replay a graph captured at the old shape:
    /// refitting the buffers reallocates them, and the old graph's nodes still point
    /// at the memory that was handed back.
    ///
    /// The sequence matters. The short T here (`8`, below GRAPH_MIN_T) is the one
    /// that broke a real training run: it takes the *eager* path, so it consults no
    /// graph cache — but it reallocates the buffers all the same. Coming back to a
    /// long T that was captured earlier then found a cache entry whose `(b, t)` key
    /// matched while its nodes addressed freed memory. The backbone meets this
    /// constantly, because windows never cross document borders and every short
    /// document yields a short window. So: long, short-eager, long-again.
    #[test]
    fn slstm_graph_survives_shape_changes() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let (b, inp, h) = (1, 64, 64);
        let mut cpu = CpuRef::new(inp, h);
        let mut dev = from_cpu(&gpu, &cpu.cell);

        // Freed device memory only *shows* it was reused if somebody reuses it. In
        // the real model the encoder/decoder's per-group temporaries do that between
        // two backbone sweeps; here nothing else allocates, so the pool would hand
        // the very same addresses back and a stale graph would still read plausible
        // data. Grab (and dirty) a block between windows to stand in for that churn.
        let poison = |floats: usize| {
            let mut d = DTensor::uninit(&gpu, &[floats]);
            ops::fill(&gpu, &mut d, 1e30);
        };

        for &t in &[256usize, 8, 256, 192, 5, 256] {
            let x = Tensor::random(&[b, t, inp], 0.5);
            let g = Tensor::random(&[b, t, h], 1.0);

            let y_cpu = cpu.forward(&x);
            let y_dev = dev.forward_alloc(&gpu, &DTensor::from_host(&gpu, &x));
            assert_close(&y_dev.to_host(&gpu).data, &y_cpu.data, tol(&gpu, 2e-3));
            let dx_cpu = cpu.backward(&g);
            let dx_dev = dev.backward_alloc(&gpu, &DTensor::from_host(&gpu, &g));
            assert_close(&dx_dev.to_host(&gpu).data, &dx_cpu.data, tol(&gpu, 2e-3));

            poison(256 * 4 * h * b);
        }
    }

    /// The time-fused forward (one cooperative launch for the whole T-loop) must
    /// agree with the per-step path it replaces.
    ///
    /// `fused_time_enabled()` reads its env var through a `OnceLock`, so this
    /// drives `ops::slstm_fused_time` directly rather than toggling the flag: the
    /// point is that the kernel computes the same thing, not how it is selected.
    /// T is above `GRAPH_MIN_T` so the comparison is against the graph path that
    /// actually runs at backbone shapes.
    #[test]
    fn slstm_fused_time_matches_per_step() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        if !gpu.kernels.has_coop {
            return; // cooperative kernels unavailable (no CUDA headers)
        }
        let (b, t, h) = (1usize, 64usize, 64usize);
        if ops::slstm_fused_time_geometry(&gpu, h, b).is_none() {
            return; // shape does not fit the fused path on this device
        }
        let w: Vec<Tensor> = (0..4)
            .map(|g| Tensor::random(&[2 * h, h], 0.3 + g as f32 * 0.01))
            .collect();
        let bi: Vec<Tensor> = (0..4)
            .map(|g| Tensor::random(&[h], 0.2 + g as f32 * 0.01))
            .collect();
        let x = Tensor::random(&[b, t, h], 0.5);
        let dx = DTensor::from_host(&gpu, &x);

        let mut per_step = SLstm::from_parts(
            &gpu, h, h, &w[0], &w[1], &w[2], &w[3], &bi[0], &bi[1], &bi[2], &bi[3], None,
        );
        per_step.force_fused_time = Some(false);
        let want = per_step.forward_alloc(&gpu, &dx).to_host(&gpu);

        let mut fused = SLstm::from_parts(
            &gpu, h, h, &w[0], &w[1], &w[2], &w[3], &bi[0], &bi[1], &bi[2], &bi[3], None,
        );
        fused.force_fused_time = Some(true);
        let got = fused.forward_alloc(&gpu, &dx).to_host(&gpu);

        assert_eq!(want.data.len(), got.data.len());
        for (i, (a, c)) in want.data.iter().zip(got.data.iter()).enumerate() {
            assert!(
                (a - c).abs() <= 1e-5 * a.abs().max(1.0),
                "fused vs per-step forward diverged at {i}: {a} vs {c}"
            );
        }

        // ...and the backward: dx plus every gate's weight gradient. The fused
        // backward carries the BPTT channels inside one launch, so an error there
        // shows up as drift that grows toward t = 0 rather than a single bad slot.
        let gy = DTensor::from_host(&gpu, &Tensor::random(&[b, t, h], 0.7));
        let want_dx = per_step.backward_alloc(&gpu, &gy).to_host(&gpu);
        let got_dx = fused.backward_alloc(&gpu, &gy).to_host(&gpu);
        assert_close(&want_dx.data, &got_dx.data, 1e-4);
        for gi in 0..4 {
            assert_close(&per_step.gate_dw(&gpu, gi), &fused.gate_dw(&gpu, gi), 1e-4);
            assert_close(&per_step.gate_db(&gpu, gi), &fused.gate_db(&gpu, gi), 1e-4);
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
        let y_whole = whole
            .forward_alloc(&gpu, &DTensor::from_host(&gpu, &x))
            .to_host(&gpu);
        let dx_whole = whole
            .backward_alloc(&gpu, &DTensor::from_host(&gpu, &dy))
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
            let xc = DTensor::from_host(&gpu, &cut(&x, inp, off, c));
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
                let xp = DTensor::from_host(&gpu, &cut(&x, inp, o, pc));
                part.forward_alloc(&gpu, &xp);
                o += pc;
            }
            let xc = DTensor::from_host(&gpu, &cut(&x, inp, ends[i], c));
            part.forward_alloc(&gpu, &xc);

            // The BPTT channels must carry from the chunk to the right, so they are
            // NOT reset here — except for the rightmost chunk, which starts at zero.
            if i + 1 == chunks.len() {
                part.reset_bptt(&gpu);
            }
            let dyc = DTensor::from_host(&gpu, &cut(&dy, h, ends[i], c));
            dx_parts.push(part.backward_alloc(&gpu, &dyc).to_host(&gpu).data);
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
}
