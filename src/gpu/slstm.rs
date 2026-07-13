//! Device-resident batched sLSTM cell, the GPU counterpart of
//! [`nn2::slstm::SLstm`](crate::nn2::slstm::SLstm).
//!
//! Same equations, same weight layout (4 gates `[rows, H]`, `rows = in + H`,
//! concat-trick), and the same AdamW convention (gate matrices decay, biases do
//! not), so a GPU cell built from a CPU cell's weights matches it for
//! forward → backward → step — which the parity test checks against `nn2::SLstm`.
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
//! The gate weights of record stay the four `[rows, H]` matrices `nn2::SLstm` and
//! the checkpoints use; `slstm_pack` derives the fused `Wx`/`Wh`/`bias` operands
//! from them each forward, and `slstm_unpack_dw` folds the fused gradients back.

use cudarc::driver::CudaGraph;
use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};

use super::ops::{self, SlstmSlabs};
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
    graph: CudaGraph,
}

/// `GPU_NO_GRAPH=1` forces the eager per-timestep launch path — the A/B baseline
/// for `slstm_launch_bench`, and the fallback if a driver ever mis-captures.
fn graphs_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("GPU_NO_GRAPH").is_ok())
}

pub struct SLstm {
    input: usize,
    hidden: usize,

    // Gate weights/biases and their grads/moments, indexed z=0, i=1, f=2, o=3.
    // These are the parameters of record: the optimizer steps them and the
    // checkpoint stores them. The fused operands below are derived from them.
    pub w: [DTensor; 4],    // [rows, H]
    pub bias: [DTensor; 4], // [H]
    dw: [DTensor; 4],
    db: [DTensor; 4],
    mw: [DTensor; 4],
    vw: [DTensor; 4],
    mb: [DTensor; 4],
    vb: [DTensor; 4],

    // Fused operands, repacked from `w`/`bias` at the top of every forward.
    wx: DTensor,   // [in, 4H]
    whr: DTensor,  // [H, 4H]  (recurrent half)
    bcat: DTensor, // [4H]

    // Recurrent state carried across timesteps within one call, [B, H].
    h_state: DTensor,
    c_state: DTensor,
    n_state: DTensor,
    m_state: DTensor,
    /// Contiguous `[B, 4H]` scratch for the current timestep's recurrent gate half
    /// (`h_{t-1}·Wh` forward, the gate deltas backward). It exists so both of those
    /// GEMMs stay dense at any batch size — see `slstm_step_fused` in `kernels.rs`.
    gh: DTensor,
    /// `[1, N]` of ones and a `[1, 4H]` landing pad: the bias gradient is the column
    /// sum of the gate deltas, which cuBLAS reduces as a `ones · dgates` GEMM.
    ones: DTensor,
    dbcat: DTensor,
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
    /// Backward's incoming `dy`, copied into a stable buffer: the caller hands us a
    /// fresh `DTensor` every time, whose pointer a graph cannot depend on.
    dy_buf: Option<DTensor>,
    /// The `(b, t)` the buffers above are currently allocated for. A captured graph
    /// is only valid for the allocation it was captured against, so the graphs are
    /// dropped whenever this changes — see [`Self::forward`].
    buf_shape: Option<(usize, usize)>,
    fwd_graph: Option<LoopGraph>,
    bwd_graph: Option<LoopGraph>,
    batch: usize,
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
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        gpu: &Gpu,
        input: usize,
        hidden: usize,
        wz: &Tensor, wi: &Tensor, wf: &Tensor, wo: &Tensor,
        bz: &Tensor, bi: &Tensor, bf: &Tensor, bo: &Tensor,
    ) -> Self {
        let rows = input + hidden;
        let up = |t: &Tensor| DTensor::from_host(gpu, t);
        let zw = || DTensor::zeros(gpu, &[rows, hidden]);
        let zb = || DTensor::zeros(gpu, &[hidden]);
        Self {
            input,
            hidden,
            w: [up(wz), up(wi), up(wf), up(wo)],
            bias: [up(bz), up(bi), up(bf), up(bo)],
            dw: [zw(), zw(), zw(), zw()],
            db: [zb(), zb(), zb(), zb()],
            mw: [zw(), zw(), zw(), zw()],
            vw: [zw(), zw(), zw(), zw()],
            mb: [zb(), zb(), zb(), zb()],
            vb: [zb(), zb(), zb(), zb()],
            wx: DTensor::zeros(gpu, &[0, 0]),
            whr: DTensor::zeros(gpu, &[0, 0]),
            bcat: DTensor::zeros(gpu, &[0]),
            h_state: DTensor::zeros(gpu, &[0, 0]),
            c_state: DTensor::zeros(gpu, &[0, 0]),
            n_state: DTensor::zeros(gpu, &[0, 0]),
            m_state: DTensor::zeros(gpu, &[0, 0]),
            gh: DTensor::zeros(gpu, &[0, 0]),
            ones: DTensor::zeros(gpu, &[0, 0]),
            dbcat: DTensor::zeros(gpu, &[0, 0]),
            dh_bptt: DTensor::zeros(gpu, &[0, 0]),
            dc_bptt: DTensor::zeros(gpu, &[0, 0]),
            dn_bptt: DTensor::zeros(gpu, &[0, 0]),
            g: None,
            slabs: None,
            x_saved: None,
            out_buf: None,
            dy_buf: None,
            buf_shape: None,
            fwd_graph: None,
            bwd_graph: None,
            batch: 0,
        }
    }

    /// Freshly-initialised cell, matching `nn2::SLstm::new`'s init exactly
    /// (including the +4.5 forget-gate bias).
    pub fn new_rand(gpu: &Gpu, input: usize, hidden: usize) -> Self {
        Self::from_cpu(gpu, &crate::nn2::SLstm::new(input, hidden))
    }

    /// Upload a CPU cell (weights are copied; grads/moments start at zero).
    pub fn from_cpu(gpu: &Gpu, cpu: &crate::nn2::SLstm) -> Self {
        Self::from_parts(
            gpu, cpu.input_size, cpu.hidden_size,
            &cpu.wz, &cpu.wi, &cpu.wf, &cpu.wo,
            &cpu.bz, &cpu.bi, &cpu.bf, &cpu.bo,
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
        use super::{dt_matrix, dt_vec};
        let h = self.hidden;
        crate::nn::slstm::SLSTMLayer::from_loaded(
            self.input,
            h,
            dt_matrix(gpu, &self.w[0]),
            dt_matrix(gpu, &self.w[1]),
            dt_matrix(gpu, &self.w[2]),
            dt_matrix(gpu, &self.w[3]),
            dt_vec(gpu, &self.bias[0]),
            dt_vec(gpu, &self.bias[1]),
            dt_vec(gpu, &self.bias[2]),
            dt_vec(gpu, &self.bias[3]),
            vec![0.0; h].into(),
            vec![0.0; h].into(),
        )
    }

    /// Rebuild a GPU cell from a CPU `nn::SLSTMLayer` (inverse of `to_nn_cell`).
    pub fn from_nn_cell(gpu: &Gpu, c: &crate::nn::slstm::SLSTMLayer) -> Self {
        use super::{tensor_from_matrix as m, tensor_from_slice as v};
        Self::from_parts(
            gpu,
            c.input_size,
            c.hidden_size,
            &m(&c.wz), &m(&c.wi), &m(&c.wf), &m(&c.wo),
            &v(&c.bz), &v(&c.bi), &v(&c.bf), &v(&c.bo),
        )
    }

    /// Forward over a whole `[B, T, in]` sequence → `[B, T, H]`. State resets to
    /// zero at t=0 and stays device-resident across the T-loop.
    pub fn forward(&mut self, gpu: &Gpu, x: &DTensor) -> DTensor {
        assert_eq!(x.rank, 3, "SLstm::forward expects [B, T, in]");
        let (b, t, inp) = (x.shape[0], x.shape[1], x.shape[2]);
        assert_eq!(inp, self.input, "SLstm::forward — input width mismatch");
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

        // Rebuild the fused operands from the gate weights the optimizer owns.
        fit_uninit(gpu, &mut self.wx, &[inp, h4]);
        fit_uninit(gpu, &mut self.whr, &[h, h4]);
        fit_uninit(gpu, &mut self.bcat, &[h4]);
        ops::slstm_pack(
            gpu, &self.w, &self.bias, &mut self.wx, &mut self.whr, &mut self.bcat, inp, h,
        );

        // Recurrent state starts at zero.
        for s in [&mut self.h_state, &mut self.c_state, &mut self.n_state, &mut self.m_state] {
            fit_zeros(gpu, s, &[b, h]);
        }

        // The input half of every gate pre-activation, for all timesteps at once —
        // it has no recurrent dependency, so it is one GEMM outside the loop.
        let mut x_flat = take_uninit(gpu, self.x_saved.take(), &[n, inp]);
        gpu.stream.memcpy_dtod(&x.buf, &mut x_flat.buf).expect("copy x");
        // One buffer, two views: the GEMM wants [N, 4H], the time loop wants
        // [B, T, 4H]. `reshaped` is metadata-only, so the allocation is untouched.
        let mut g = take_uninit(gpu, self.g.take(), &[b, t, h4]).reshaped(&[n, h4]);
        ops::matmul_nn_into(gpu, &x_flat, &self.wx, &mut g, 0.0);
        let mut g = g.reshaped(&[b, t, h4]);

        let mut slabs = match self.slabs.take() {
            Some(s) if s.c.dims() == [b, t, h].as_slice() => s,
            _ => {
                let slab = || DTensor::uninit(gpu, &[b, t, h]);
                SlstmSlabs {
                    c_prev: slab(), n_prev: slab(), zt: slab(), ot: slab(), i_prime: slab(),
                    f_prime: slab(), c: slab(), n: slab(), h_prev: slab(),
                }
            }
        };
        let mut out = take_uninit(gpu, self.out_buf.take(), &[b, t, h]);
        fit_uninit(gpu, &mut self.gh, &[b, h4]);

        self.fwd_loop(gpu, &mut g, &mut slabs, &mut out, b, t);

        // `out` is the graph's write target and must keep its address, so hand the
        // caller a copy rather than the buffer itself. One [B, T, H] device-to-device
        // copy against a loop that was costing tens of milliseconds of launch time.
        let y = out.dup(gpu);
        self.g = Some(g);
        self.slabs = Some(slabs);
        self.x_saved = Some(x_flat);
        self.out_buf = Some(out);
        y
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
        if t < GRAPH_MIN_T || graphs_disabled() {
            self.fwd_steps(gpu, g, slabs, out, t);
            return;
        }
        if self.fwd_graph.as_ref().map_or(true, |c| (c.b, c.t) != (b, t)) {
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
                .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
                .expect("end capture")
                .expect("stream was not capturing");
            self.fwd_graph = Some(LoopGraph { b, t, graph });
        }
        self.fwd_graph.as_ref().unwrap().graph.launch().expect("graph launch");
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
            ops::matmul_nn_into(gpu, &self.h_state, &self.whr, &mut self.gh, 0.0);
            ops::slstm_step_fused(
                gpu, g, &self.gh, &self.bcat,
                &mut self.c_state, &mut self.n_state, &mut self.m_state, &mut self.h_state,
                slabs, out, step,
            );
        }
    }

    /// Backward over the whole sequence. `dy` is `[B, T, H]`; returns
    /// `dx` `[B, T, in]`. Accumulates weight/bias grads.
    pub fn backward(&mut self, gpu: &Gpu, dy: &DTensor) -> DTensor {
        assert_eq!(dy.rank, 3, "SLstm::backward expects [B, T, H]");
        let (b, t, h) = (dy.shape[0], dy.shape[1], dy.shape[2]);
        assert_eq!(b, self.batch, "SLstm::backward — batch mismatch");
        assert_eq!(h, self.hidden, "SLstm::backward — hidden mismatch");
        let inp = self.input;
        let h4 = 4 * h;
        let n = b * t;

        // Taken, not borrowed: these are rebuilt by every forward, so releasing them
        // here frees the device memory across the optimizer step.
        let mut g = self.g.take().expect("forward before backward");
        let mut slabs = self.slabs.take().expect("forward before backward");
        let x_flat = self.x_saved.take().expect("forward before backward");

        for buf in [&mut self.dh_bptt, &mut self.dc_bptt, &mut self.dn_bptt] {
            fit_zeros(gpu, buf, &[b, h]);
        }
        fit_uninit(gpu, &mut self.gh, &[b, h4]);

        // The loop reads `dy` every step, and the caller hands us a different
        // `DTensor` each time — a pointer a captured graph cannot follow. Copy it
        // into a buffer whose address we own.
        let mut dy_buf = take_uninit(gpu, self.dy_buf.take(), &[b, t, h]);
        gpu.stream.memcpy_dtod(&dy.buf, &mut dy_buf.buf).expect("copy dy");

        // The only thing the loop must carry is BPTT: the gate deltas go straight
        // back into `g`, and everything derived from them waits until the loop ends.
        self.bwd_loop(gpu, &dy_buf, &mut g, &slabs, b, t);
        self.dy_buf = Some(dy_buf);

        // `g` now holds the gate deltas for the whole sequence: dx, dWx, dWh and the
        // bias grads are three GEMMs and one reduction over it.
        let dg = g.reshaped(&[n, h4]);
        let dx = ops::matmul_nt(gpu, &dg, &self.wx); // [N, in]

        let mut dwx = DTensor::uninit(gpu, &[inp, h4]);
        ops::matmul_tn_into(gpu, &x_flat, &dg, &mut dwx, 0.0);
        let h_prev = slabs.h_prev.reshaped(&[n, h]);
        let mut dwh = DTensor::uninit(gpu, &[h, h4]);
        ops::matmul_tn_into(gpu, &h_prev, &dg, &mut dwh, 0.0);

        ops::slstm_unpack_dw(gpu, &dwx, &dwh, &mut self.dw, inp, h);

        fit_uninit(gpu, &mut self.ones, &[1, n]);
        ops::fill(gpu, &mut self.ones, 1.0);
        fit_uninit(gpu, &mut self.dbcat, &[1, h4]);
        ops::slstm_db_from_dg(gpu, &dg, &self.ones, &mut self.dbcat, &mut self.db, h);

        // Give the buffers back (same allocations, original shapes) so the next
        // forward reuses them — and so the captured graphs stay valid.
        slabs.h_prev = h_prev.reshaped(&[b, t, h]);
        self.g = Some(dg.reshaped(&[b, t, h4]));
        self.slabs = Some(slabs);
        self.x_saved = Some(x_flat);

        dx.reshaped(&[b, t, inp])
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
        if t < GRAPH_MIN_T || graphs_disabled() {
            self.bwd_steps(gpu, dy, g, slabs, t);
            return;
        }
        if self.bwd_graph.as_ref().map_or(true, |c| (c.b, c.t) != (b, t)) {
            self.bwd_graph = None;
            gpu.stream
                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .expect("begin capture");
            self.bwd_steps(gpu, dy, g, slabs, t);
            let graph = gpu
                .stream
                .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
                .expect("end capture")
                .expect("stream was not capturing");
            self.bwd_graph = Some(LoopGraph { b, t, graph });
        }
        self.bwd_graph.as_ref().unwrap().graph.launch().expect("graph launch");
    }

    fn bwd_steps(&mut self, gpu: &Gpu, dy: &DTensor, g: &mut DTensor, slabs: &SlstmSlabs, t: usize) {
        for step in (0..t).rev() {
            ops::slstm_step_fused_bwd(
                gpu, dy, g, &mut self.gh, &self.dh_bptt, slabs,
                &mut self.dc_bptt, &mut self.dn_bptt, step,
            );
            // dh_{t-1} = dgates_t · Whᵀ — the one gradient BPTT cannot defer.
            ops::matmul_nt_into(gpu, &self.gh, &self.whr, &mut self.dh_bptt, 0.0);
        }
    }

    /// Every learnable tensor, in a fixed order (used by checkpoint save/load).
    pub fn params_mut(&mut self) -> Vec<&mut DTensor> {
        self.w.iter_mut().chain(self.bias.iter_mut()).collect()
    }

    pub fn zero_grad(&mut self, gpu: &Gpu) {
        for g in self.dw.iter_mut().chain(self.db.iter_mut()) {
            g.zero_(gpu);
        }
    }

    /// AdamW step: gate matrices decay, biases don't. Clears the grads.
    pub fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        for i in 0..4 {
            ops::adamw(gpu, &mut self.w[i], &self.dw[i], &mut self.mw[i], &mut self.vw[i], cfg, true);
        }
        for i in 0..4 {
            ops::adamw(gpu, &mut self.bias[i], &self.db[i], &mut self.mb[i], &mut self.vb[i], cfg, false);
        }
        self.zero_grad(gpu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn2::optim::AdamCfg;
    use crate::nn2::slstm::SLstm as CpuSLstm;

    fn assert_close(got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len(), "length mismatch");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < tol, "index {i}: gpu {g} vs cpu {w}");
        }
    }

    fn from_cpu(gpu: &Gpu, cpu: &CpuSLstm) -> SLstm {
        SLstm::from_parts(
            gpu, cpu.input_size, cpu.hidden_size,
            &cpu.wz, &cpu.wi, &cpu.wf, &cpu.wo,
            &cpu.bz, &cpu.bi, &cpu.bf, &cpu.bo,
        )
    }

    /// GPU sLSTM must match `nn2::SLstm` (cell alone) for a full
    /// forward → backward → AdamW-step cycle, from identical weights. Tolerance
    /// is loose-ish because the two paths differ in float reduction order (cuBLAS
    /// vs the CPU gemm), but the recurrence math is identical.
    #[test]
    fn slstm_matches_cpu_layer() {
        let Some(gpu) = super::super::test_gpu() else { return };
        let (b, t, inp, h) = (2, 5, 4, 6);

        let mut cpu = CpuSLstm::new(inp, h);
        let mut dev = from_cpu(&gpu, &cpu);

        let x = Tensor::random(&[b, t, inp], 0.5);
        let g = Tensor::random(&[b, t, h], 1.0);

        // Forward
        let y_cpu = cpu.forward(&x);
        let y_dev = dev.forward(&gpu, &DTensor::from_host(&gpu, &x));
        assert_close(&y_dev.to_host(&gpu).data, &y_cpu.data, 2e-3);

        // Backward
        let dx_cpu = cpu.backward(&g);
        let dx_dev = dev.backward(&gpu, &DTensor::from_host(&gpu, &g));
        assert_close(&dx_dev.to_host(&gpu).data, &dx_cpu.data, 2e-3);
        assert_close(&dev.dw[0].to_host(&gpu).data, &cpu.dwz.data, 2e-3);
        assert_close(&dev.dw[2].to_host(&gpu).data, &cpu.dwf.data, 2e-3);
        assert_close(&dev.db[2].to_host(&gpu).data, &cpu.dbf.data, 2e-3);

        // One AdamW step, then compare the updated gate weights.
        let mut cfg = AdamCfg::new(1e-3, 0.01);
        cfg.t = 1;
        cpu.step(&cfg);
        dev.step(&gpu, &cfg);
        // Looser than Linear's 1e-5: the AdamW update is ~lr in magnitude, and a
        // near-zero grad element can sign-flip between the cuBLAS and CPU gemm
        // reduction orders, swinging its update by ~2·lr. A plumbing bug misses by
        // O(weight), far more than this.
        assert_close(&dev.w[0].to_host(&gpu).data, &cpu.wz.data, 2e-3);
        assert_close(&dev.w[2].to_host(&gpu).data, &cpu.wf.data, 2e-3);
        assert_close(&dev.bias[2].to_host(&gpu).data, &cpu.bf.data, 2e-3);
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
        let Some(gpu) = super::super::test_gpu() else { return };
        assert!(64 > GRAPH_MIN_T, "this test must exercise the graph path");
        let (b, t, inp, h) = (2, 64, 8, 12);

        let mut cpu = CpuSLstm::new(inp, h);
        let mut dev = from_cpu(&gpu, &cpu);
        let mut cfg = AdamCfg::new(1e-3, 0.01);

        for pass in 0..2 {
            let x = Tensor::random(&[b, t, inp], 0.5);
            let g = Tensor::random(&[b, t, h], 1.0);

            let y_cpu = cpu.forward(&x);
            let y_dev = dev.forward(&gpu, &DTensor::from_host(&gpu, &x));
            assert_close(&y_dev.to_host(&gpu).data, &y_cpu.data, 2e-3);

            let dx_cpu = cpu.backward(&g);
            let dx_dev = dev.backward(&gpu, &DTensor::from_host(&gpu, &g));
            assert_close(&dx_dev.to_host(&gpu).data, &dx_cpu.data, 2e-3);
            assert_close(&dev.dw[0].to_host(&gpu).data, &cpu.dwz.data, 2e-3);
            assert_close(&dev.dw[2].to_host(&gpu).data, &cpu.dwf.data, 2e-3);

            // Step between passes, so the replay runs against *changed* weights —
            // the graph must read the packed operands live, not a stale copy.
            cfg.t = pass + 1;
            cpu.step(&cfg);
            dev.step(&gpu, &cfg);
            assert_close(&dev.w[0].to_host(&gpu).data, &cpu.wz.data, 2e-3);
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
        let Some(gpu) = super::super::test_gpu() else { return };
        let (b, inp, h) = (1, 64, 64);
        let mut cpu = CpuSLstm::new(inp, h);
        let mut dev = from_cpu(&gpu, &cpu);

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
            let y_dev = dev.forward(&gpu, &DTensor::from_host(&gpu, &x));
            assert_close(&y_dev.to_host(&gpu).data, &y_cpu.data, 2e-3);
            let dx_cpu = cpu.backward(&g);
            let dx_dev = dev.backward(&gpu, &DTensor::from_host(&gpu, &g));
            assert_close(&dx_dev.to_host(&gpu).data, &dx_cpu.data, 2e-3);

            poison(256 * 4 * h * b);
        }
    }
}
