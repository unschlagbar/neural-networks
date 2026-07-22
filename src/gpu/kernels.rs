//! NVRTC-compiled CUDA kernels for the backend op seam.
//!
//! One source string, compiled once at [`Kernels::load`] and cached on the
//! [`Gpu`](super::Gpu). Each `extern "C"` kernel is the device counterpart of a
//! function in `src/nn2/ops.rs` / `loss.rs` / `optim.rs`, and is launched by a
//! thin wrapper in `src/gpu/ops.rs` that owns the grid/block config and is
//! parity-checked against the CPU reference.
//!
//! Conventions: tensors are the same row-major contiguous `f32` as on the host;
//! embedding ids and CE targets are uploaded as `unsigned int`; reductions that
//! accumulate into shared outputs (`embedding_scatter_add`, RMSNorm `dgamma`)
//! use `atomicAdd`.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction};
use cudarc::nvrtc::{CompileOptions, compile_ptx, compile_ptx_with_opts};

const SRC: &str = r#"
extern "C" __global__ void softcap_forward(const float* x, float* y, float cap, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = cap * tanhf(x[i] / cap);
}

extern "C" __global__ void softcap_backward(const float* dy, const float* y, float* dx, float cap, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float t = y[i] / cap; dx[i] = dy[i] * (1.0f - t * t); }
}

// Copy bias[n] into every row of out[rows, n].
extern "C" __global__ void broadcast_row(float* out, const float* bias, int rows, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * n) out[i] = bias[i % n];
}

// db[o] += sum over rows of dy[r*n + o]. One thread per column.
extern "C" __global__ void add_col_sum(float* db, const float* dy, int rows, int n) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o < n) {
        float s = 0.0f;
        for (int r = 0; r < rows; ++r) s += dy[r * n + o];
        db[o] += s;
    }
}

// out[r, :] = table[ids[r], :]. One thread per output element.
extern "C" __global__ void embedding_gather(const float* table, const unsigned* ids,
                                            float* out, int dim, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * dim) {
        int r = i / dim, c = i % dim;
        out[i] = table[ids[r] * dim + c];
    }
}

// dtable[ids[r], :] += dy[r, :]. Ids may repeat -> atomicAdd.
extern "C" __global__ void embedding_scatter_add(float* dtable, const unsigned* ids,
                                                 const float* dy, int dim, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * dim) {
        int r = i / dim, c = i % dim;
        atomicAdd(&dtable[ids[r] * dim + c], dy[i]);
    }
}

// Grouped RMSNorm forward. One thread per (row, group); f = groups_per_row*group.
extern "C" __global__ void rms_norm_forward(const float* x, const float* gamma, float* out,
                                            float* x_hat, float* inv_rms,
                                            int groups_per_row, int group, float eps,
                                            int total_groups) {
    int gi = blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= total_groups) return;
    int row = gi / groups_per_row;
    int grp = gi % groups_per_row;
    int f = groups_per_row * group;
    int off = row * f + grp * group;
    int g_off = grp * group;
    float ss = 0.0f;
    for (int i = 0; i < group; ++i) { float v = x[off + i]; ss += v * v; }
    float inv = rsqrtf(ss / (float)group + eps);
    inv_rms[gi] = inv;
    for (int i = 0; i < group; ++i) {
        float xh = x[off + i] * inv;
        x_hat[off + i] = xh;
        out[off + i] = gamma[g_off + i] * xh;
    }
}

// Grouped RMSNorm backward. dgamma is shared across rows -> atomicAdd.
extern "C" __global__ void rms_norm_backward(const float* dy, const float* x_hat,
                                             const float* inv_rms, const float* gamma,
                                             float* dgamma, float* dx,
                                             int groups_per_row, int group, int total_groups) {
    int gi = blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= total_groups) return;
    int row = gi / groups_per_row;
    int grp = gi % groups_per_row;
    int f = groups_per_row * group;
    int off = row * f + grp * group;
    int g_off = grp * group;
    float inv = inv_rms[gi];
    float s = 0.0f;
    for (int i = 0; i < group; ++i) {
        float dyxh = dy[off + i] * x_hat[off + i];
        atomicAdd(&dgamma[g_off + i], dyxh);
        s += gamma[g_off + i] * dyxh;
    }
    float s_over_g = s / (float)group;
    for (int i = 0; i < group; ++i) {
        dx[off + i] = inv * (gamma[g_off + i] * dy[off + i] - x_hat[off + i] * s_over_g);
    }
}

// Fused softmax + cross-entropy. One thread per row. Writes dlogits = (p - onehot)/B
// in place and the per-row loss -ln p_target into row_loss (host sums * inv_b).
extern "C" __global__ void softmax_ce(const float* logits, const unsigned* targets,
                                      float* dlogits, float* row_loss,
                                      int c, float inv_b, int b) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= b) return;
    const float* z = logits + r * c;
    float* d = dlogits + r * c;
    float mx = -1e30f;
    for (int j = 0; j < c; ++j) mx = fmaxf(mx, z[j]);
    float sum = 0.0f;
    for (int j = 0; j < c; ++j) sum += expf(z[j] - mx);
    unsigned t = targets[r];
    float p_t = expf(z[t] - mx) / sum;
    row_loss[r] = -logf(fmaxf(p_t, 1e-30f));
    for (int j = 0; j < c; ++j) {
        float p = expf(z[j] - mx) / sum;
        if ((unsigned)j == t) p -= 1.0f;
        d[j] = p * inv_b;
    }
}

// AdamW, one thread per parameter. bc1/bc2 (bias corrections) and the effective
// weight decay (0 if this param doesn't decay) are precomputed on the host.
extern "C" __global__ void adamw(float* param, const float* grad, float* m, float* v,
                                 float lr, float b1, float b2, float eps, float wd,
                                 float bc1, float bc2, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    float g = grad[k];
    float mk = b1 * m[k] + (1.0f - b1) * g;
    float vk = b2 * v[k] + (1.0f - b2) * g * g;
    m[k] = mk; v[k] = vk;
    float mh = mk / bc1;
    float vh = vk / bc2;
    float p = param[k];
    p -= lr * wd * p;
    p -= lr * mh / (sqrtf(vh) + eps);
    param[k] = p;
}

// sLSTM cell (recurrent core)
// Numerically-stable sigmoid / log-sigmoid, matching the CPU helpers in
// nn2/slstm.rs (branch on the sign to avoid overflow of exp).
__device__ __forceinline__ float stable_sigmoid(float x) {
    if (x >= 0.0f) { return 1.0f / (1.0f + expf(-x)); }
    float e = expf(x);
    return e / (1.0f + e);
}
__device__ __forceinline__ float log_sigmoid(float x) {
    if (x >= 0.0f) { return -log1pf(expf(-x)); }
    return x - log1pf(expf(x));
}

// Build xh = concat(x_t, h_state) as [B, rows], rows = inp + H. One thread per
// output element. The first `inp` columns come from timestep `t` of x[B,T,inp];
// the remaining `H` columns from the current recurrent state h_state[B,H].
extern "C" __global__ void concat_xh(float* xh, const float* x, const float* h_state,
                                     int t, int T, int inp, int H, int BR) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BR) return;
    int rows = inp + H;
    int b = i / rows, col = i % rows;
    if (col < inp) {
        xh[i] = x[(b * T + t) * inp + col];
    } else {
        xh[i] = h_state[b * H + (col - inp)];
    }
}

// Inverse of concat_xh in the backward pass: split dxh[B, rows] into dx[:,t,:]
// (first `inp` columns) and dh_bptt[B,H] (last `H` columns). One thread per
// element of [B, rows].
extern "C" __global__ void split_dxh(const float* dxh, float* dx, float* dh_bptt,
                                     int t, int T, int inp, int H, int BR) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BR) return;
    int rows = inp + H;
    int b = i / rows, col = i % rows;
    if (col < inp) {
        dx[(b * T + t) * inp + col] = dxh[i];
    } else {
        dh_bptt[b * H + (col - inp)] = dxh[i];
    }
}

// Elementwise sLSTM recurrence over B*H (nn2/slstm.rs lines 241-271). Reads the
// four gate pre-activations and the running (c,n,m) state; writes the advanced
// (c,n,m,h) state, the per-step saved tensors for backward, and out[:,t,:].
// c_state/n_state are read (previous) then overwritten; the previous values are
// saved into c_prev/n_prev for the weight/BPTT gradients.
extern "C" __global__ void slstm_cell_step(
        const float* zt_pre, const float* it_pre, const float* ft_pre, const float* ot_pre,
        float* c_state, float* n_state, float* m_state, float* h_state,
        float* c_prev, float* n_prev, float* zt, float* ot,
        float* i_prime, float* f_prime, float* c_out, float* n_out,
        float* out, int t, int T, int H, int BH) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= BH) return;
    float z = tanhf(zt_pre[k]);
    float o = stable_sigmoid(ot_pre[k]);
    float log_f = log_sigmoid(ft_pre[k]);
    float fm = log_f + m_state[k];
    float it = it_pre[k];
    float np = n_state[k];
    // First step of a sequence (n == 0): take m = ĩ, so i' is exactly 1 and n
    // starts at 1. Otherwise max(logσ(f̃)+m_prev, ĩ) could make i' underflow to 0
    // and leave h = c/n as 0/0. This is the reference's `if all(n == 0)` guard, and
    // the state resets per word here, so it is hit constantly.
    float m = (np == 0.0f) ? it : fmaxf(fm, it);
    float ip = expf(it - m);
    float fp = expf(fm - m);
    float cp = c_state[k];
    float c = fp * cp + ip * z;
    float n = fp * np + ip;
    c_prev[k] = cp;
    n_prev[k] = np;
    zt[k] = z; ot[k] = o; i_prime[k] = ip; f_prime[k] = fp;
    c_out[k] = c; n_out[k] = n;
    c_state[k] = c; n_state[k] = n; m_state[k] = m;
    // h = o·c/n, NOT o·c/max(|n|,1). c and n both carry exp(−m), so it cancels in
    // the ratio — the sLSTM normalizer is stabilizer-invariant by construction, and
    // clamping the STABILIZED n at 1 would let m leak into the model's output.
    float hh = o * c / n;
    h_state[k] = hh;
    int b = k / H, j = k % H;
    out[(b * T + t) * H + j] = hh;
}

// Backward of slstm_cell_step (nn2/slstm.rs lines 321-353). Produces the four
// gate deltas (dz,di,df,dob) and carries the BPTT channels dc_bptt/dn_bptt back
// one step (read = incoming from the later step, write = outgoing to the earlier
// step). dh_bptt (incoming) is read here; it is rewritten by split_dxh afterward.
extern "C" __global__ void slstm_cell_step_bwd(
        const float* dy, int t, int T, const float* dh_bptt,
        const float* ot, const float* c, const float* n,
        const float* c_prev, const float* n_prev, const float* zt,
        const float* i_prime, const float* f_prime, const float* ft_pre,
        float* dc_bptt, float* dn_bptt,
        float* dz, float* di, float* df, float* dob, int H, int BH) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= BH) return;
    int b = k / H, j = k % H;
    float d = dy[(b * T + t) * H + j] + dh_bptt[k];
    float o = ot[k];
    float cc = c[k];
    float nn = n[k];
    // h = o·c/n — no clamp, so dn has no branch: ∂h/∂n = −o·c/n².
    dob[k] = d * (cc / nn) * o * (1.0f - o);
    float dc = d * o / nn + dc_bptt[k];
    float dn = d * o * (-cc) / (nn * nn) + dn_bptt[k];
    float fp = f_prime[k];
    float df_prime = dc * c_prev[k] + dn * n_prev[k];
    float ztk = zt[k];
    float di_prime = dc * ztk + dn;
    float dz_post = dc * i_prime[k];
    dz[k] = dz_post * (1.0f - ztk * ztk);
    di[k] = di_prime * i_prime[k];
    float sig_f = stable_sigmoid(ft_pre[k]);
    df[k] = df_prime * fp * (1.0f - sig_f);
    dc_bptt[k] = dc * fp;
    dn_bptt[k] = dn * fp;
}

// fused-gate sLSTM (see gpu/slstm.rs)
// The kernels above run one gate GEMM per gate per timestep. These run the four
// gates as one [.., 4H] block, so a timestep costs one GEMM + one kernel. The
// weights of record stay the four [rows, H] gate matrices (checkpoint layout);
// these pack them into the fused operands the GEMMs want, and unpack the grads
// back. Gate order is z=0, i=1, f=2, o=3 — the column blocks of the fused [.,4H].

// Pack the four gate matrices [rows, H] (rows = inp + H, input part on top of
// the recurrent part) into wx [inp, 4H] and wh [H, 4H]. One thread per element
// of the [rows, 4H] fused layout.
extern "C" __global__ void slstm_pack_w(const float* w0, const float* w1,
                                        const float* w2, const float* w3,
                                        float* wx, float* wh, int inp, int H, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int H4 = 4 * H;
    if (i >= rows * H4) return;
    int r = i / H4, cc = i % H4;
    int g = cc / H, j = cc % H;
    const float* w = (g == 0) ? w0 : (g == 1) ? w1 : (g == 2) ? w2 : w3;
    float v = w[r * H + j];
    if (r < inp) wx[r * H4 + cc] = v;
    else         wh[(r - inp) * H4 + cc] = v;
}

// Pack the four bias vectors [H] into bcat [4H].
extern "C" __global__ void slstm_pack_b(const float* b0, const float* b1,
                                        const float* b2, const float* b3,
                                        float* bcat, int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 4 * H) return;
    int g = i / H, j = i % H;
    const float* b = (g == 0) ? b0 : (g == 1) ? b1 : (g == 2) ? b2 : b3;
    bcat[i] = b[j];
}

// Inverse of slstm_pack_w for the gradients: dw[g] += the g-th column block of
// the fused dwx / dwh. Accumulates, so gradients survive across windows.
extern "C" __global__ void slstm_unpack_dw(const float* dwx, const float* dwh,
                                           float* dw0, float* dw1, float* dw2, float* dw3,
                                           int inp, int H, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int H4 = 4 * H;
    if (i >= rows * H4) return;
    int r = i / H4, cc = i % H4;
    int g = cc / H, j = cc % H;
    float v = (r < inp) ? dwx[r * H4 + cc] : dwh[(r - inp) * H4 + cc];
    float* dw = (g == 0) ? dw0 : (g == 1) ? dw1 : (g == 2) ? dw2 : dw3;
    dw[r * H + j] += v;
}

// Scatter the already-reduced fused bias gradient dbcat [4H] into the four db[g].
// The reduction itself (summing the N rows of the gate-delta buffer) is left to a
// ones-vector GEMM: doing it here, one thread per column looping over N, would put
// a 2048-thread serial scan on the critical path — measurably slower than every
// other part of the backward put together.
extern "C" __global__ void slstm_unpack_db(const float* dbcat,
                                           float* db0, float* db1, float* db2, float* db3,
                                           int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 4 * H) return;
    int g = i / H, j = i % H;
    float* db = (g == 0) ? db0 : (g == 1) ? db1 : (g == 2) ? db2 : db3;
    db[j] += dbcat[i];
}

// out[i] = v, for the ones vector that drives the bias-gradient reduction.
extern "C" __global__ void fill_const(float* out, float v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = v;
}

// Fused-gate forward step. Same recurrence as slstm_cell_step, but a gate
// pre-activation is assembled here from three pieces instead of arriving as one
// tensor: the input half `g` (a [B, T, 4H] buffer holding x·Wx for every
// timestep), the recurrent half `gh` (a [B, 4H] scratch holding h_{t-1}·Wh for
// this timestep) and the bias. The saved tensors are [B, T, H] slabs indexed by
// (b·T + t)·H + j rather than one small tensor per timestep.
//
// Keeping the recurrent half in its own *contiguous* [B, 4H] scratch is what lets
// its GEMM stay one dense matmul at any batch size — accumulating straight into
// g's strided [:, t, :] rows would force a batched GEMM of one-row matrices, which
// is fine at batch 1 (the backbone) and disastrous at batch 2047 (the encoder).
//
// `g`'s forget block is overwritten in place with the *biased* forget
// pre-activation: backward needs it, and the slot is dead once this step is done.
// `h_prev` records h_{t-1} for the deferred dWh GEMM.
extern "C" __global__ void slstm_step_fused(
        float* g, const float* gh, const float* bcat, float* h_prev,
        float* c_state, float* n_state, float* m_state, float* h_state,
        float* c_prev, float* n_prev, float* zt, float* ot,
        float* i_prime, float* f_prime, float* c_out, float* n_out,
        float* out, int t, int T, int H, int BH) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= BH) return;
    int b = k / H, j = k % H;
    long long go = (long long)(b * T + t) * 4 * H + j;
    long long s = (long long)(b * T + t) * H + j;
    int ho = b * 4 * H + j; // gh row for this batch element

    float z_pre = g[go]         + gh[ho]           + bcat[j];
    float i_pre = g[go + H]     + gh[ho + H]       + bcat[H + j];
    float f_pre = g[go + 2 * H] + gh[ho + 2 * H]   + bcat[2 * H + j];
    float o_pre = g[go + 3 * H] + gh[ho + 3 * H]   + bcat[3 * H + j];
    g[go + 2 * H] = f_pre; // biased forget pre-activation, saved for backward

    h_prev[s] = h_state[k];

    float z = tanhf(z_pre);
    float o = stable_sigmoid(o_pre);
    float log_f = log_sigmoid(f_pre);
    float fm = log_f + m_state[k];
    float np = n_state[k];
    // See `slstm_cell_step`: n == 0 is the first step of a sequence, where m must be
    // ĩ so that i' is exactly 1 and h = c/n cannot become 0/0.
    float m = (np == 0.0f) ? i_pre : fmaxf(fm, i_pre);
    float ip = expf(i_pre - m);
    float fp = expf(fm - m);
    float cp = c_state[k];
    float c = fp * cp + ip * z;
    float n = fp * np + ip;

    c_prev[s] = cp;
    n_prev[s] = np;
    zt[s] = z; ot[s] = o; i_prime[s] = ip; f_prime[s] = fp;
    c_out[s] = c; n_out[s] = n;
    c_state[k] = c; n_state[k] = n; m_state[k] = m;

    // h = o·c/n — the exp(−m) in c and n cancels. See `slstm_cell_step`.
    float hh = o * c / n;
    h_state[k] = hh;
    out[s] = hh;
}

// Backward of slstm_step_fused: the four gate deltas are written back into `g`
// (whose forward contents are dead by now), so one buffer carries the gate
// pre-activations forward and the gate deltas backward. Reads the biased forget
// pre-activation out of g's forget block before overwriting it.
//
// The deltas are also written to the contiguous [B, 4H] scratch `dgh`, the mirror
// of the forward's `gh`: this timestep's dh = dgh·Whᵀ is the one thing BPTT cannot
// defer, and going through the scratch keeps that a dense GEMM at any batch size.
extern "C" __global__ void slstm_step_fused_bwd(
        const float* dy, float* g, float* dgh, const float* dh_bptt,
        const float* ot, const float* c, const float* n,
        const float* c_prev, const float* n_prev, const float* zt,
        const float* i_prime, const float* f_prime,
        float* dc_bptt, float* dn_bptt, int t, int T, int H, int BH) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= BH) return;
    int b = k / H, j = k % H;
    long long go = (long long)(b * T + t) * 4 * H + j;
    long long s = (long long)(b * T + t) * H + j;
    int ho = b * 4 * H + j;

    float ft_pre = g[go + 2 * H];
    float d = dy[s] + dh_bptt[k];
    float o = ot[s];
    float cc = c[s];
    float nn = n[s];
    // h = o·c/n — no clamp, so dn has no branch: ∂h/∂n = −o·c/n².
    float dob = d * (cc / nn) * o * (1.0f - o);
    float dc = d * o / nn + dc_bptt[k];
    float dn = d * o * (-cc) / (nn * nn) + dn_bptt[k];
    float fp = f_prime[s];
    float df_prime = dc * c_prev[s] + dn * n_prev[s];
    float ztk = zt[s];
    float di_prime = dc * ztk + dn;
    float dz_post = dc * i_prime[s];
    float sig_f = stable_sigmoid(ft_pre);

    float dz = dz_post * (1.0f - ztk * ztk);
    float di = di_prime * i_prime[s];
    float df = df_prime * fp * (1.0f - sig_f);
    g[go]         = dz;  dgh[ho]         = dz;
    g[go + H]     = di;  dgh[ho + H]     = di;
    g[go + 2 * H] = df;  dgh[ho + 2 * H] = df;
    g[go + 3 * H] = dob; dgh[ho + 3 * H] = dob;

    dc_bptt[k] = dc * fp;
    dn_bptt[k] = dn * fp;
}

// residual block / SwiGLU (nn2/block.rs)
// Elementwise add: out = a + b. Used for the two residual adds and the grad
// accumulations that are plain sums (d_zn, d_z, dx).
extern "C" __global__ void add(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// SwiGLU forward: gate_act = SiLU(gate_pre); mixed = gate_act ⊙ value.
// SiLU(x) = x·σ(x). One thread per element of the [N, U] tensors.
extern "C" __global__ void swiglu_forward(const float* gate_pre, const float* value,
                                          float* gate_act, float* mixed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gp = gate_pre[i];
    float ga = gp * stable_sigmoid(gp);
    gate_act[i] = ga;
    mixed[i] = ga * value[i];
}

// mLSTM projections (nn2/mlstm.rs `project`)
// In-place multiply by a scalar (the k-projection's 1/√dqk scale).
extern "C" __global__ void scale_inplace(float* x, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

// In-place numerically-stable sigmoid (the o-gate projection).
extern "C" __global__ void sigmoid_inplace(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = stable_sigmoid(x[i]);
}

// mLSTM parallel/chunkwise core (nn2/mlstm.rs)
// Reorganize a position-major [N, H*W] tensor (N=B*T, row=b*T+t) into a
// head-major [B*H, T, W] tensor for the per-(batch,head) batched matmuls.
// The flat output index idx == ((b*H+h)*T+t)*W+c, so it decomposes cleanly.
extern "C" __global__ void head_gather(const float* x, float* out, int B, int H, int T, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * T * W) return;
    int c = idx % W;
    int t = (idx / W) % T;
    int h = (idx / (W * T)) % H;
    int b = idx / (W * T * H);
    out[idx] = x[(b * T + t) * (H * W) + h * W + c];
}

// Inverse of head_gather: head-major [B*H, T, W] → position-major [N, H*W].
// Plain write (each destination element hit once).
extern "C" __global__ void head_scatter(const float* in, float* x, int B, int H, int T, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * T * W) return;
    int c = idx % W;
    int t = (idx / W) % T;
    int h = (idx / (W * T)) % H;
    int b = idx / (W * T * H);
    x[(b * T + t) * (H * W) + h * W + c] = in[idx];
}

// Inclusive cumulative sum of logσ(f) along T, per row g of [BH, T]. One thread
// per row (the scan is serial but each row is independent; T is small).
extern "C" __global__ void cumsum_logsig(const float* f, float* fc, int T, int BH) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= BH) return;
    float acc = 0.0f;
    for (int t = 0; t < T; ++t) { acc += log_sigmoid(f[g * T + t]); fc[g * T + t] = acc; }
}

// Per-row stabilizer m[g,t] = max( max_{j<=t}(fc_t - fc_j + ig_j), fc_t + m_prev_g ).
// One thread per (g,t) over BH*T. `ig` is the input-gate logit [BH,T].
extern "C" __global__ void mlstm_rowmax_m(const float* fc, const float* ig, const float* m_prev,
                                          float* m, int T, int BHT) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BHT) return;
    int t = idx % T, g = idx / T;
    float fct = fc[g * T + t];
    float mx = fct + m_prev[g];
    for (int j = 0; j <= t; ++j) {
        float ld = fct - fc[g * T + j] + ig[g * T + j];
        mx = fmaxf(mx, ld);
    }
    m[idx] = mx;
}

// DS = D̄ ⊙ S with D̄_{tj}=exp(fc_t-fc_j+ig_j-m_t) (j<=t else 0); also the row
// normalizer qn_t = Σ_j DS_{tj} and ψ_t = max(|qn_t|,1). One thread per (g,t).
// S/DS are [BH,T,T] row-major (row t of head g at g*T*T + t*T).
extern "C" __global__ void mlstm_ds(const float* S, const float* fc, const float* ig, const float* m,
                                    float* Dbar, float* DS, float* qn, float* psi, int T, int BHT) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BHT) return;
    int t = idx % T, g = idx / T;
    float fct = fc[g * T + t], mt = m[idx];
    long base = (long)g * T * T + (long)t * T;
    float acc = 0.0f;
    for (int j = 0; j < T; ++j) {
        if (j <= t) {
            float dbar = expf(fct - fc[g * T + j] + ig[g * T + j] - mt);
            float val = dbar * S[base + j];
            Dbar[base + j] = dbar;
            DS[base + j] = val;
            acc += val;
        } else {
            Dbar[base + j] = 0.0f;
            DS[base + j] = 0.0f;
        }
    }
    qn[idx] = acc;
    // ψ = max(|qn|, exp(−m)): qn is the STABILIZED normalizer, so xLSTM's
    // max(|n_trueᵀq|, 1) becomes this once exp(m) cancels. See `nn2::mlstm`.
    psi[idx] = fmaxf(fabsf(acc), expf(-mt));
}

// Row-normalize num by ψ: ytil[g,t,i] = num[g,t,i] / psi[g,t]. num:[BH,T,dhv].
extern "C" __global__ void div_rows(const float* num, const float* psi, float* ytil,
                                    int dhv, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    ytil[idx] = num[idx] / psi[idx / dhv];
}

// Elementwise product out = a ⊙ b (the o-gate: hconcat = o ⊙ yhat).
extern "C" __global__ void mul(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

// mLSTM chunking (inter-chunk state carry; see gpu/mlstm.rs)
// A chunk is a contiguous T-range [c0, c0+L) of a [BH, T, W] head-major tensor.
// Within a group g the range is contiguous (g*T*W + c0*W, length L*W), so both
// directions are plain index math.

// Extract a chunk: out[BH,L,W] = x[BH, c0..c0+L, W].
extern "C" __global__ void slice_t(const float* x, float* out, int T, int L, int c0,
                                   int W, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int w = idx % W;
    int t = (idx / W) % L;
    int g = idx / (W * L);
    out[idx] = x[((long)g * T + c0 + t) * W + w];
}

// Write a chunk back: dst[BH, c0..c0+L, W] = src[BH,L,W]. Chunks partition T, so
// every destination element is written exactly once — a plain store, not an add.
extern "C" __global__ void unslice_t(float* dst, const float* src, int T, int L, int c0,
                                     int W, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int w = idx % W;
    int t = (idx / W) % L;
    int g = idx / (W * L);
    dst[((long)g * T + c0 + t) * W + w] = src[idx];
}

// The two inter-chunk decay weights, both [BH, L] (fc is the chunk-LOCAL cumsum):
//   b_t = exp(fc_t + m_prev − m_t)               — scales the carried state into row t
//   a_j = exp(fc_last − fc_j + ig_j − m_last)    — scales row j into the outgoing state
// a_j is the last row of D̄ and b_last is the state-decay scalar g, so the
// end-of-chunk update needs no further exponentials. One thread per (g,t).
extern "C" __global__ void mlstm_chunk_ab(const float* fc, const float* ig, const float* m,
                                          const float* m_prev, float* b, float* a,
                                          int L, int BHL) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BHL) return;
    int t = idx % L, g = idx / L;
    int last = g * L + L - 1;
    b[idx] = expf(fc[idx] + m_prev[g] - m[idx]);
    a[idx] = expf(fc[last] - fc[idx] + ig[idx] - m[last]);
}

// out[i] = s[i/W] · x[i] — scale each row of a [·, W] tensor by a per-row scalar.
extern "C" __global__ void mul_rows(float* out, const float* x, const float* s,
                                    int W, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = s[i / W] * x[i];
}

// out[i] += s[i/W] · x[i]. Used both for row scaling (W = dhv) and for the
// per-head state decay dst += g[head]·src (W = the head's element count).
extern "C" __global__ void mul_rows_add(float* out, const float* x, const float* s,
                                        int W, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] += s[i / W] * x[i];
}

// ψ = max(|qn|, 1), recomputed once the inter-chunk term has been added into qn
// (the single-chunk path gets ψ straight out of `mlstm_ds`).
extern "C" __global__ void psi_from_qn(const float* qn, const float* m, float* psi, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) psi[i] = fmaxf(fabsf(qn[i]), expf(-m[i]));
}

// out[r] += Σ_w x[r,W+w]·y[r,W+w] — row-wise dot of two [R, W] tensors.
extern "C" __global__ void row_dot_add(float* out, const float* x, const float* y,
                                       int W, int R) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;
    long base = (long)r * W;
    float acc = 0.0f;
    for (int w = 0; w < W; ++w) acc += x[base + w] * y[base + w];
    out[r] += acc;
}

// out[g] += Σ_e x[g,e]·y[g,e] — the per-head reduction behind dg (E = dhv·dqk for
// the C state, dqk for n).
extern "C" __global__ void group_dot_add(float* out, const float* x, const float* y,
                                         int E, int G) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= G) return;
    long base = (long)g * E;
    float acc = 0.0f;
    for (int e = 0; e < E; ++e) acc += x[base + e] * y[base + e];
    out[g] += acc;
}

// Backward of `mlstm_chunk_ab`, ACCUMULATING into the dfc/dig that `mlstm_dfc_dig`
// already wrote from the intra-chunk D̄ (m held const, as everywhere):
//   b_t = exp(fc_t + m_prev − m_t)            → dfc_t += db_t·b_t
//   a_j = exp(fc_last − fc_j + ig_j − m_last) → Pa_j = da_j·a_j;
//                                               dig_j += Pa_j; dfc_j −= Pa_j;
//                                               dfc_last += Σ_j Pa_j
// The Σ_j term lands on the last row, so the thread that owns it also runs the
// (serial, L-long) reduction — no cross-thread race on dfc[last].
extern "C" __global__ void mlstm_chunk_ab_bwd(const float* db, const float* da,
                                              const float* b, const float* a,
                                              float* dfc, float* dig, int L, int BHL) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BHL) return;
    int t = idx % L, g = idx / L;
    float pa = da[idx] * a[idx];
    float acc = db[idx] * b[idx] - pa;
    dig[idx] += pa;
    if (t == L - 1) {
        for (int j = 0; j < L; ++j) acc += da[g * L + j] * a[g * L + j];
    }
    dfc[idx] += acc;
}

// hierarchical model
// Copy rows of `src` into arbitrary rows of `dst`: dst[row_ids[i], :] = src[i, :].
// The inverse (gathering those rows back out) is just `embedding_gather` with the
// same row ids, so the hierarchical model needs no separate gather kernel.
extern "C" __global__ void scatter_rows(float* dst, const float* src, const unsigned* row_ids,
                                        int dim, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * dim) return;
    int r = i / dim, c = i % dim;
    dst[(long)row_ids[r] * dim + c] = src[i];
}

// Masked softmax cross-entropy (the hierarchical decode loss). Rows with mask==0
// are padding: zero grad, zero loss. `inv` = 1/num_valid (computed host-side), so
// the caller's loss is sum(row_loss)*inv and dlogits = (p − onehot)*inv.
extern "C" __global__ void masked_softmax_ce(const float* logits, const unsigned* targets,
                                             const unsigned* mask, float* dlogits, float* row_loss,
                                             int C, float inv, int R) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;
    const float* z = logits + (long)r * C;
    float* d = dlogits + (long)r * C;
    if (!mask[r]) {
        for (int j = 0; j < C; ++j) d[j] = 0.0f;
        row_loss[r] = 0.0f;
        return;
    }
    float mx = -1e30f;
    for (int j = 0; j < C; ++j) mx = fmaxf(mx, z[j]);
    float sum = 0.0f;
    for (int j = 0; j < C; ++j) sum += expf(z[j] - mx);
    unsigned t = targets[r];
    float pt = expf(z[t] - mx) / sum;
    row_loss[r] = -logf(fmaxf(pt, 1e-30f));
    for (int j = 0; j < C; ++j) {
        float p = expf(z[j] - mx) / sum;
        if ((unsigned)j == t) p -= 1.0f;
        d[j] = p * inv;
    }
}

// mLSTM parallel-form backward
// o-gate backward: hconcat = o ⊙ yhat with o = σ(o_pre).
//   d_yhat = d_hconcat ⊙ o ;  do_pre = d_hconcat ⊙ yhat ⊙ o(1-o).
extern "C" __global__ void ogate_bwd(const float* d_hconcat, const float* o, const float* yhat,
                                     float* do_pre, float* d_yhat, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float dch = d_hconcat[i], oi = o[i];
    do_pre[i] = dch * yhat[i] * oi * (1.0f - oi);
    d_yhat[i] = dch * oi;
}

// Backward of ytil = num/ψ (with ψ = max(|qn|,1)). One thread per row (g,t):
//   d_num = d_ytil/ψ ;  dψ = −(Σ_i d_ytil·num)/ψ² ;  d_qn = (|qn|>1? sign(qn):0)·dψ.
extern "C" __global__ void div_rows_bwd(const float* d_ytil, const float* num, const float* psi,
                                        const float* qn, const float* m,
                                        float* d_num, float* d_qn,
                                        int dhv, int BHT) {
    int gt = blockIdx.x * blockDim.x + threadIdx.x;
    if (gt >= BHT) return;
    float inv = 1.0f / psi[gt];
    long base = (long)gt * dhv;
    float red = 0.0f;
    for (int i = 0; i < dhv; ++i) {
        float dy = d_ytil[base + i];
        d_num[base + i] = dy * inv;
        red += dy * num[base + i];
    }
    float dpsi = -red * inv * inv;
    float q = qn[gt];
    // Grad flows through qn only where it, not the exp(−m) floor, won the max.
    d_qn[gt] = (fabsf(q) > expf(-m[gt])) ? ((q > 0.0f ? 1.0f : -1.0f) * dpsi) : 0.0f;
}

// Backward of DS = D̄⊙S plus the qn row-sum. Given dDS_num (= d_num·Vᵀ, the num
// path) and d_qn (the qn path), form the full dDS = dDS_num + d_qn (masked j≤t),
// then split: dS = dDS⊙D̄ and P = dDS⊙DS (P feeds the fc/ig grads, since
// dD̄ = dDS⊙S and P = dD̄⊙D̄ = dDS⊙DS). One thread per (g,t,j).
extern "C" __global__ void mlstm_ds_bwd(const float* dDS_num, const float* d_qn,
                                        const float* Dbar, const float* DS,
                                        float* dS, float* P, int T, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int j = idx % T;
    int t = (idx / T) % T;
    if (j <= t) {
        int g = idx / (T * T);
        float full = dDS_num[idx] + d_qn[g * T + t];
        dS[idx] = full * Dbar[idx];
        P[idx] = full * DS[idx];
    } else {
        dS[idx] = 0.0f;
        P[idx] = 0.0f;
    }
}

// Reduce P into the cumulative-log-forget and input-gate grads (m held const):
//   dfc[g,r] = Σ_{j≤r} P[g,r,j] − Σ_{t≥r} P[g,t,r]      (fc_t: +D̄, fc_j: −D̄)
//   dig[g,r] = Σ_{t≥r} P[g,t,r]                          (ig_j: +D̄)
// One thread per (g,r) over BH·T.
extern "C" __global__ void mlstm_dfc_dig(const float* P, float* dfc, float* dig, int T, int BHT) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BHT) return;
    int r = idx % T, g = idx / T;
    long gb = (long)g * T * T;
    float rowsum = 0.0f;
    for (int j = 0; j <= r; ++j) rowsum += P[gb + (long)r * T + j];
    float colsum = 0.0f;
    for (int t = r; t < T; ++t) colsum += P[gb + (long)t * T + r];
    dfc[idx] = rowsum - colsum;
    dig[idx] = colsum;
}

// Backward of fc = cumsum_t(logσ(f)): dL_s = Σ_{t≥s} dfc[g,t] (reverse cumsum),
// then chain through logσ':  d_f[g,s] = dL_s · (1 − σ(f[g,s])). One thread/row.
extern "C" __global__ void revcumsum_dlogsig(const float* dfc, const float* f, float* df,
                                             int T, int BH) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= BH) return;
    float acc = 0.0f;
    for (int t = T - 1; t >= 0; --t) {
        acc += dfc[g * T + t];
        df[g * T + t] = acc * (1.0f - stable_sigmoid(f[g * T + t]));
    }
}

// SwiGLU backward: from d_mixed (grad wrt gate_act⊙value),
//   d_value = d_mixed ⊙ gate_act
//   d_gate  = d_mixed ⊙ value ⊙ SiLU'(gate_pre),  SiLU'(x) = σ(x)(1 + x(1-σ(x))).
extern "C" __global__ void swiglu_backward(const float* d_mixed, const float* gate_act,
                                           const float* value, const float* gate_pre,
                                           float* d_gate, float* d_value, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float dm = d_mixed[i];
    d_value[i] = dm * gate_act[i];
    float gp = gate_pre[i];
    float s = stable_sigmoid(gp);
    float sp = s * (1.0f + gp * (1.0f - s));
    d_gate[i] = dm * value[i] * sp;
}

// ===========================================================================
// mLSTM chunkwise, FUSED (TFLA — after nx-ai/mlstm_kernels).
// ===========================================================================
//
// The op-at-a-time path in `gpu::mlstm` runs this same math as ~25 launches per
// chunk inside a host-side chunk loop: ~600 launches for one fwd+bwd at the
// backbone's shape, which is ~1 ms of arithmetic stretched over 14 ms of driver
// latency (see `examples/mlstm_stage_prof.rs`). These kernels do a whole sequence
// in three launches forward and two backward, and the [L, L] decay matrix never
// reaches HBM — it lives in shared memory for the lifetime of a block.
//
// Notation follows the reference. Within a chunk of length `len` (<= L):
//   fc[j]      = cumsum_{j'<=j} logsigmoid(f[j'])           (their vecB)
//   logD[t][j] = fc[t] - fc[j] + i[j]   for j <= t          (their matD)
//   m[t]       = max( max_{j<=t} logD[t][j], fc[t] + m_prev )
//   b[t]       = exp(fc[t] + m_prev - m[t])                 (their vecBbar)
//   a[j]       = exp(fc[last] - fc[j] + i[j] - m[last])     (their vecAbar)
//   g          = exp(fc[last] + m_prev - m[last])           (their scaGbar)
// which is exactly the (fc, m, bvec, avec) the op-at-a-time path builds, so the
// two agree elementwise — `mlstm_fused_matches_legacy` pins them together.
//
// The only sequential axis is the CHUNK STATE (C, n, m), and it is small. So the
// work splits into a serial-over-chunks kernel that carries a [dhv, dqk] state
// and does no per-timestep launch, and a parallel-over-chunks kernel that holds
// all the FLOPs and is embarrassingly parallel:
//   mlstm_fw_gates    -> fc, per chunk, independent
//   mlstm_fw_C        -> the chunk states, looping chunks INSIDE the kernel
//   mlstm_fw_parallel -> every chunk independently, one block each
// Backward mirrors it (mlstm_bw_dC walks chunks in reverse, mlstm_bw_parallel is
// per-chunk). All five take a `[BH, T, ·]` head-major layout and a chunk length
// `L`; the last chunk may be short and is masked by `len` everywhere.

// fc: the chunk-local cumulative log-forget. One thread per (bh, chunk) — the
// scan is serial but L is tiny and there are BH*NC of them. Positions past `len`
// hold the last valid prefix (they are always masked out by a `j < len` guard,
// but leaving them undefined would poison the exp()s below).
extern "C" __global__ void mlstm_fw_gates(const float* fg, float* fcb,
                                          int T, int L, int NC, int BH) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BH * NC) return;
    int k = idx % NC, bh = idx / NC;
    int c0 = k * L;
    int len = min(L, T - c0);
    float acc = 0.0f;
    float* out = fcb + (long)(bh * NC + k) * L;
    for (int j = 0; j < L; ++j) {
        if (j < len) acc += log_sigmoid(fg[(long)bh * T + c0 + j]);
        out[j] = acc;
    }
}

// The chunk states. ONE block per (bh); the chunk loop is inside the kernel, so
// the serial dependency costs iterations, not launches.
//   C_k = g·C_{k-1} + Σ_j a[j]·V[j]⊗K[j]      (C is [dhv, dqk])
//   n_k = g·n_{k-1} + Σ_j a[j]·K[j]
//   m_k = max(g_exp + m_{k-1}, max_j a_exp[j])
// State k (the state ENTERING chunk k) is published at index k, so index 0 is the
// zero initial state and index NC is the final one.
//
// m is carried in a REGISTER, not shared memory: every thread derives it from the
// same block-wide max reduction, so they all hold the identical value and no
// broadcast is needed.
//
// Every 2D shared array is stored with its row stride PADDED by one float (LQ, LV,
// LS below). Without that, a row stride of dqk = 64 floats puts `s[row*64 + c]` on
// bank (row*64 + c) % 32 = c % 32 for every row — so the warps below, which walk
// `row` across threads at fixed `c`, would hit one bank 32 ways and serialize. The
// pad makes consecutive rows land on consecutive banks. This is the single biggest
// factor in these kernels; measured, it is worth ~4x.
// This kernel and `mlstm_bw_dC` are TILED over the value dimension: grid is
// (ceil(dhv/TV), BH), and a block owns the `v` slice [v0, v0+tv) of the state.
// Untiled, the grid would be BH alone — 8 blocks at the backbone's shape (B=1,
// 8 heads), i.e. 8 of the GPU's 48 SMs doing a 64x64 state update over every chunk
// in sequence. Slicing `v` is free: C's update is an outer product, so row v of C
// only ever needs column v of V, and only the `n` update (which does not depend on
// v at all) has to be assigned to a single tile — tile 0.
extern "C" __global__ void mlstm_fw_C(const float* kk, const float* vv, const float* ig,
                                      const float* fcb,
                                      float* cst, float* nst, float* mst,
                                      int T, int L, int NC, int dqk, int dhv, int TV) {
    int v0 = blockIdx.x * TV, bh = blockIdx.y;
    int tv = min(TV, dhv - v0);
    int tid = threadIdx.x, nthreads = blockDim.x;
    int LQ = dqk + 1, LV = tv + 1;
    int lead = (blockIdx.x == 0); // the tile that also owns `n` and `m`

    extern __shared__ float sh[];
    float* sK  = sh;                  // [L, LQ]
    float* sV  = sK + L * LQ;         // [L, LV]   the v-slice only
    float* sC  = sV + L * LV;         // [tv, LQ]
    float* sN  = sC + tv * LQ;        // [dqk]
    float* sFc = sN + dqk;            // [L]
    float* sIg = sFc + L;             // [L]
    float* sA  = sIg + L;             // [L]
    __shared__ float sRed[256];

    for (int e = tid; e < tv * dqk; e += nthreads) {
        int v = e / dqk, q = e - v * dqk;
        sC[v * LQ + q] = 0.0f;
    }
    for (int e = tid; e < dqk; e += nthreads) sN[e] = 0.0f;
    float m_run = 0.0f;
    __syncthreads();

    for (int k = 0; k < NC; ++k) {
        int c0 = k * L;
        int len = min(L, T - c0);

        // Publish the state entering chunk k. Each element is owned by one thread
        // for the whole kernel, so reading it here and updating it below is a
        // same-thread sequence — no barrier needed between the two.
        float* cout = cst + ((long)bh * (NC + 1) + k) * dhv * dqk;
        for (int e = tid; e < tv * dqk; e += nthreads) {
            int v = e / dqk, q = e - v * dqk;
            cout[(long)(v0 + v) * dqk + q] = sC[v * LQ + q];
        }
        if (lead) {
            float* nout = nst + ((long)bh * (NC + 1) + k) * dqk;
            for (int e = tid; e < dqk; e += nthreads) nout[e] = sN[e];
            if (tid == 0) mst[(long)bh * (NC + 1) + k] = m_run;
        }

        for (int j = tid; j < L; j += nthreads) {
            sFc[j] = fcb[((long)bh * NC + k) * L + j];
            sIg[j] = (j < len) ? ig[(long)bh * T + c0 + j] : 0.0f;
        }
        for (int e = tid; e < len * dqk; e += nthreads) {
            int j = e / dqk, q = e - j * dqk;
            sK[j * LQ + q] = kk[((long)bh * T + c0) * dqk + e];
        }
        for (int e = tid; e < len * tv; e += nthreads) {
            int j = e / tv, v = e - j * tv;
            sV[j * LV + v] = vv[((long)bh * T + c0 + j) * dhv + v0 + v];
        }
        __syncthreads();

        float fc_last = sFc[len - 1];

        // m_new = max( max_j (fc_last - fc_j + ig_j), fc_last + m_prev ). Every tile
        // recomputes it — it is a handful of flops, and the alternative is a grid-wide
        // dependency between blocks.
        float local = -1e30f;
        for (int j = tid; j < len; j += nthreads)
            local = fmaxf(local, fc_last - sFc[j] + sIg[j]);
        sRed[tid] = local;
        __syncthreads();
        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) sRed[tid] = fmaxf(sRed[tid], sRed[tid + s]);
            __syncthreads();
        }
        float m_new = fmaxf(sRed[0], fc_last + m_run);
        float gbar = expf(fc_last + m_run - m_new);

        for (int j = tid; j < L; j += nthreads)
            sA[j] = (j < len) ? expf(fc_last - sFc[j] + sIg[j] - m_new) : 0.0f;
        __syncthreads();

        for (int e = tid; e < tv * dqk; e += nthreads) {
            int v = e / dqk, q = e - v * dqk;
            float acc = 0.0f;
            for (int j = 0; j < len; ++j) acc += sA[j] * sV[j * LV + v] * sK[j * LQ + q];
            sC[v * LQ + q] = gbar * sC[v * LQ + q] + acc;
        }
        if (lead) {
            for (int q = tid; q < dqk; q += nthreads) {
                float acc = 0.0f;
                for (int j = 0; j < len; ++j) acc += sA[j] * sK[j * LQ + q];
                sN[q] = gbar * sN[q] + acc;
            }
        }
        m_run = m_new;
        __syncthreads();
    }

    float* cout = cst + ((long)bh * (NC + 1) + NC) * dhv * dqk;
    for (int e = tid; e < tv * dqk; e += nthreads) {
        int v = e / dqk, q = e - v * dqk;
        cout[(long)(v0 + v) * dqk + q] = sC[v * LQ + q];
    }
    if (lead) {
        float* nout = nst + ((long)bh * (NC + 1) + NC) * dqk;
        for (int e = tid; e < dqk; e += nthreads) nout[e] = sN[e];
        if (tid == 0) mst[(long)bh * (NC + 1) + NC] = m_run;
    }
}

// One block per (chunk, bh) — all chunks at once. Intra-chunk attention plus the
// read-out of the incoming state:
//   num[t] = Σ_j (D̄⊙S)[t][j]·V[j] + b[t]·(Q[t]·C_prevᵀ)
//   qn[t]  = Σ_j (D̄⊙S)[t][j]      + b[t]·(Q[t]·n_prev)
//   ỹ[t]   = num[t] / max(|qn[t]|, 1)
// Chunk 0 needs no special case: its incoming state is zero, so the inter terms
// vanish on their own.
extern "C" __global__ void mlstm_fw_parallel(
    const float* qq, const float* kk, const float* vv, const float* ig, const float* fcb,
    const float* cst, const float* nst, const float* mst,
    float* ytil, float* msv, float* psiv, float* qnv,
    int T, int L, int NC, int dqk, int dhv) {
    int k = blockIdx.x, bh = blockIdx.y;
    int tid = threadIdx.x, nthreads = blockDim.x;
    int c0 = k * L;
    int len = min(L, T - c0);
    int LQ = dqk + 1, LV = dhv + 1, LS = L + 1;

    extern __shared__ float sh[];
    float* sQ  = sh;                  // [L, LQ]
    float* sK  = sQ + L * LQ;         // [L, LQ]
    float* sV  = sK + L * LQ;         // [L, LV]
    float* sC  = sV + L * LV;         // [dhv, LQ]
    float* sDS = sC + dhv * LQ;       // [L, LS]
    float* sN  = sDS + L * LS;        // [dqk]
    float* sFc = sN + dqk;            // [L]
    float* sIg = sFc + L;             // [L]
    float* sM  = sIg + L;             // [L]
    float* sB  = sM + L;              // [L]
    float* sQn = sB + L;              // [L]

    for (int e = tid; e < len * dqk; e += nthreads) {
        int t = e / dqk, q = e - t * dqk;
        sQ[t * LQ + q] = qq[((long)bh * T + c0) * dqk + e];
        sK[t * LQ + q] = kk[((long)bh * T + c0) * dqk + e];
    }
    for (int e = tid; e < len * dhv; e += nthreads) {
        int t = e / dhv, v = e - t * dhv;
        sV[t * LV + v] = vv[((long)bh * T + c0) * dhv + e];
    }
    for (int e = tid; e < dhv * dqk; e += nthreads) {
        int v = e / dqk, q = e - v * dqk;
        sC[v * LQ + q] = cst[((long)bh * (NC + 1) + k) * dhv * dqk + e];
    }
    for (int e = tid; e < dqk; e += nthreads)
        sN[e] = nst[((long)bh * (NC + 1) + k) * dqk + e];
    for (int j = tid; j < L; j += nthreads) {
        sFc[j] = fcb[((long)bh * NC + k) * L + j];
        sIg[j] = (j < len) ? ig[(long)bh * T + c0 + j] : 0.0f;
    }
    float m_prev = mst[(long)bh * (NC + 1) + k];
    __syncthreads();

    for (int t = tid; t < len; t += nthreads) {
        float fct = sFc[t];
        float mx = fct + m_prev;
        for (int j = 0; j <= t; ++j) mx = fmaxf(mx, fct - sFc[j] + sIg[j]);
        sM[t] = mx;
        sB[t] = expf(fct + m_prev - mx);
    }
    __syncthreads();

    // DS = D̄ ⊙ (Q·Kᵀ), the whole [len, len] block, kept in shared memory.
    for (int e = tid; e < len * len; e += nthreads) {
        int t = e / len, j = e - t * len;
        float val = 0.0f;
        if (j <= t) {
            float s = 0.0f;
            for (int q = 0; q < dqk; ++q) s += sQ[t * LQ + q] * sK[j * LQ + q];
            val = expf(sFc[t] - sFc[j] + sIg[j] - sM[t]) * s;
        }
        sDS[t * LS + j] = val;
    }
    __syncthreads();

    for (int t = tid; t < len; t += nthreads) {
        float acc = 0.0f;
        for (int j = 0; j <= t; ++j) acc += sDS[t * LS + j];
        float qi = 0.0f;
        for (int q = 0; q < dqk; ++q) qi += sQ[t * LQ + q] * sN[q];
        acc += sB[t] * qi;
        sQn[t] = acc;
        long gt = (long)bh * T + c0 + t;
        qnv[gt] = acc;
        msv[gt] = sM[t];
        // ψ = max(|qn|, exp(−m)) — qn is the stabilized normalizer. See `nn2::mlstm`.
        psiv[gt] = fmaxf(fabsf(acc), expf(-sM[t]));
    }
    __syncthreads();

    for (int e = tid; e < len * dhv; e += nthreads) {
        int t = e / dhv, v = e - t * dhv;
        float acc = 0.0f;
        for (int j = 0; j <= t; ++j) acc += sDS[t * LS + j] * sV[j * LV + v];
        float inter = 0.0f;
        for (int q = 0; q < dqk; ++q) inter += sQ[t * LQ + q] * sC[v * LQ + q];
        acc += sB[t] * inter;
        ytil[((long)bh * T + c0 + t) * dhv + v] =
            acc / fmaxf(fabsf(sQn[t]), expf(-sM[t]));
    }
}
// Tensor-core dot (MMA_TF32, sm_80+)
//
// The scalar kernels above give every output element to one thread, which then
// walks the contraction with an FMA loop. That is the one thing our chunkwise core
// does differently from the reference (nx-ai/mlstm_kernels): every contraction
// there is a `tl.dot`, and Triton lowers `tl.dot` on fp32 inputs to the tensor
// cores in TF32 (`allow_tf32` defaults to true). Below is that same `dot`, written
// out as the PTX Triton would emit.
//
// The unit is `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`: a whole WARP
// cooperates to compute D(16x8) += A(16x8)·B(8x8), with the operands rounded to
// TF32 (fp32's 8 exponent bits, 10 mantissa bits) and the product accumulated in
// full fp32. Precision-wise it sits exactly where cuBLAS's TF32 math mode does,
// and where the reference already is.
//
// A warp's 32 lanes each hold a fixed slice of every fragment. With
//   g = lane / 4   (the "group")      c = lane % 4   (the index within it)
// the layouts the instruction requires are:
//   A (16x8, row): a0=(g, c)   a1=(g+8, c)   a2=(g, c+4)   a3=(g+8, c+4)
//   B (8x8, col):  b0=(c, g)   b1=(c+4, g)                  [(row=k, col=n)]
//   D (16x8):      d0=(g, 2c)  d1=(g, 2c+1)  d2=(g+8, 2c)  d3=(g+8, 2c+1)
// The `ld_*` helpers below are just those tables, applied to a shared-memory tile.
// Nothing here is allowed to diverge inside a warp — `mma.sync` is warp-wide.
#if MMA_TF32

__device__ __forceinline__ unsigned tf32_of(float x) {
    unsigned r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}

// D += A·B for one warp. Accumulates in place, so a K-loop just calls it again.
__device__ __forceinline__ void mma_16x8x8(float* d, const unsigned* a, const unsigned* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// A from a row-major [M, K] tile: A[m][k] = s[m*ld + k].
__device__ __forceinline__ void ld_a_mk(unsigned* a, const float* s, int ld, int m0, int k0) {
    int lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    a[0] = tf32_of(s[(m0 + g)     * ld + k0 + c]);
    a[1] = tf32_of(s[(m0 + g + 8) * ld + k0 + c]);
    a[2] = tf32_of(s[(m0 + g)     * ld + k0 + c + 4]);
    a[3] = tf32_of(s[(m0 + g + 8) * ld + k0 + c + 4]);
}

// A from a row-major [K, M] tile, i.e. Aᵀ is what is in memory: A[m][k] = s[k*ld + m].
// This is how `dS` is contracted over `t` for dK without ever transposing it.
__device__ __forceinline__ void ld_a_km(unsigned* a, const float* s, int ld, int m0, int k0) {
    int lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    a[0] = tf32_of(s[(k0 + c)     * ld + m0 + g]);
    a[1] = tf32_of(s[(k0 + c)     * ld + m0 + g + 8]);
    a[2] = tf32_of(s[(k0 + c + 4) * ld + m0 + g]);
    a[3] = tf32_of(s[(k0 + c + 4) * ld + m0 + g + 8]);
}

// B from a row-major [N, K] tile: B[k][n] = s[n*ld + k]. (`Q·Kᵀ`: K is stored
// [j, q] and is wanted as [q, j].)
__device__ __forceinline__ void ld_b_nk(unsigned* b, const float* s, int ld, int k0, int n0) {
    int lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    b[0] = tf32_of(s[(n0 + g) * ld + k0 + c]);
    b[1] = tf32_of(s[(n0 + g) * ld + k0 + c + 4]);
}

// B from a row-major [K, N] tile: B[k][n] = s[k*ld + n]. (`(D̄⊙S)·V`.)
__device__ __forceinline__ void ld_b_kn(unsigned* b, const float* s, int ld, int k0, int n0) {
    int lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    b[0] = tf32_of(s[(k0 + c)     * ld + n0 + g]);
    b[1] = tf32_of(s[(k0 + c + 4) * ld + n0 + g]);
}

// Where accumulator register `i` lands in the 16x8 output tile.
__device__ __forceinline__ int mma_row(int i) { return ((threadIdx.x & 31) >> 2) + ((i & 2) ? 8 : 0); }
__device__ __forceinline__ int mma_col(int i) { return (((threadIdx.x & 31) & 3) << 1) + (i & 1); }

// The tensor-core twin of `mlstm_fw_parallel`. Same algorithm, same shared-memory
// plan, same numbers up to TF32 rounding of the three contractions:
//   S    = Q·Kᵀ        (over dqk)   -> masked/decayed in the mma epilogue
//   H    = (D̄⊙S)·V    (over j)
//   Hinter = Q·C_prevᵀ (over dqk)   -> scaled by b[t] and added to H
// The decay mask is applied as the epilogue of the first mma rather than in a pass
// of its own, so `S` never lands in shared memory unmasked.
//
// Shapes are padded UP to the mma tile (rows to 16, contractions to 8, columns to
// 8) and the pad is zero-filled, so a short last chunk, an odd `dqk` or an odd
// `dhv` all fall out for free: a zero row contributes nothing to a dot, and the
// out-of-range outputs are simply not written. `LP`/`KP`/`VP` are those padded
// dims and must match `fused_smem("fw_parallel_mma", ..)` on the host exactly.
extern "C" __global__ void mlstm_fw_parallel_mma(
    const float* qq, const float* kk, const float* vv, const float* ig, const float* fcb,
    const float* cst, const float* nst, const float* mst,
    float* ytil, float* msv, float* psiv, float* qnv,
    int T, int L, int NC, int dqk, int dhv) {
    int k = blockIdx.x, bh = blockIdx.y;
    int tid = threadIdx.x, nthreads = blockDim.x;
    int warp = tid >> 5, nwarps = nthreads >> 5;
    int c0 = k * L;
    int len = min(L, T - c0);

    int LP = (L + 15) & ~15;    // rows      -> multiple of the mma M
    int KP = (dqk + 7) & ~7;    // dqk       -> multiple of the mma K
    int VP = (dhv + 7) & ~7;    // dhv       -> multiple of the mma N
    int LQ = KP + 1, LV = VP + 1, LS = LP + 1;   // +1: the bank-conflict pad

    extern __shared__ float sh[];
    float* sQ  = sh;                  // [LP, LQ]
    float* sK  = sQ + LP * LQ;        // [LP, LQ]
    float* sV  = sK + LP * LQ;        // [LP, LV]
    float* sC  = sV + LP * LV;        // [VP, LQ]
    float* sDS = sC + VP * LQ;        // [LP, LS]
    float* sN  = sDS + LP * LS;       // [KP]
    float* sFc = sN + KP;             // [LP]
    float* sIg = sFc + LP;            // [LP]
    float* sM  = sIg + LP;            // [LP]
    float* sB  = sM + LP;             // [LP]
    float* sQn = sB + LP;             // [LP]

    for (int e = tid; e < LP * KP; e += nthreads) {
        int t = e / KP, q = e - t * KP;
        int ok = (t < len) && (q < dqk);
        sQ[t * LQ + q] = ok ? qq[((long)bh * T + c0 + t) * dqk + q] : 0.0f;
        sK[t * LQ + q] = ok ? kk[((long)bh * T + c0 + t) * dqk + q] : 0.0f;
    }
    for (int e = tid; e < LP * VP; e += nthreads) {
        int t = e / VP, v = e - t * VP;
        sV[t * LV + v] = ((t < len) && (v < dhv))
            ? vv[((long)bh * T + c0 + t) * dhv + v] : 0.0f;
    }
    for (int e = tid; e < VP * KP; e += nthreads) {
        int v = e / KP, q = e - v * KP;
        sC[v * LQ + q] = ((v < dhv) && (q < dqk))
            ? cst[((long)bh * (NC + 1) + k) * dhv * dqk + (long)v * dqk + q] : 0.0f;
    }
    for (int e = tid; e < KP; e += nthreads)
        sN[e] = (e < dqk) ? nst[((long)bh * (NC + 1) + k) * dqk + e] : 0.0f;
    for (int j = tid; j < LP; j += nthreads) {
        sFc[j] = (j < L)   ? fcb[((long)bh * NC + k) * L + j] : 0.0f;
        sIg[j] = (j < len) ? ig[(long)bh * T + c0 + j]        : 0.0f;
    }
    float m_prev = mst[(long)bh * (NC + 1) + k];
    __syncthreads();

    for (int t = tid; t < LP; t += nthreads) {
        float mx = 0.0f, b = 0.0f;
        if (t < len) {
            float fct = sFc[t];
            mx = fct + m_prev;
            for (int j = 0; j <= t; ++j) mx = fmaxf(mx, fct - sFc[j] + sIg[j]);
            b = expf(fct + m_prev - mx);
        }
        sM[t] = mx;
        sB[t] = b;
    }
    __syncthreads();

    // S = Q·Kᵀ, with D̄ and the causal mask folded into the epilogue. Rows/cols in
    // the pad get 0, which is what the next dot needs anyway.
    int mtile = LP >> 4, ntile = LP >> 3;
    for (int tile = warp; tile < mtile * ntile; tile += nwarps) {
        int m0 = (tile / ntile) << 4, n0 = (tile % ntile) << 3;
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        unsigned a[4], b[2];
        for (int k0 = 0; k0 < KP; k0 += 8) {
            ld_a_mk(a, sQ, LQ, m0, k0);
            ld_b_nk(b, sK, LQ, k0, n0);
            mma_16x8x8(d, a, b);
        }
        for (int i = 0; i < 4; ++i) {
            int t = m0 + mma_row(i), j = n0 + mma_col(i);
            float val = 0.0f;
            if (t < len && j <= t) val = expf(sFc[t] - sFc[j] + sIg[j] - sM[t]) * d[i];
            sDS[t * LS + j] = val;
        }
    }
    __syncthreads();

    // qn: the row sum of D̄⊙S plus the b·(Q·n_prev) read-out. A matrix-VECTOR
    // product, so it stays scalar — there is no dot for the tensor cores here.
    for (int t = tid; t < len; t += nthreads) {
        float acc = 0.0f;
        for (int j = 0; j <= t; ++j) acc += sDS[t * LS + j];
        float qi = 0.0f;
        for (int q = 0; q < dqk; ++q) qi += sQ[t * LQ + q] * sN[q];
        acc += sB[t] * qi;
        sQn[t] = acc;
        long gt = (long)bh * T + c0 + t;
        qnv[gt] = acc;
        msv[gt] = sM[t];
        psiv[gt] = fmaxf(fabsf(acc), expf(-sM[t]));
    }
    __syncthreads();

    // The two output dots share an output tile, so they share a warp: intra over j,
    // inter over q, combined as num = intra + b[t]·inter in the epilogue.
    int vtile = VP >> 3;
    for (int tile = warp; tile < mtile * vtile; tile += nwarps) {
        int m0 = (tile / vtile) << 4, n0 = (tile % vtile) << 3;
        float dintra[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float dinter[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        unsigned a[4], b[2];
        for (int k0 = 0; k0 < LP; k0 += 8) {   // (D̄⊙S)·V, contracting j
            ld_a_mk(a, sDS, LS, m0, k0);
            ld_b_kn(b, sV, LV, k0, n0);
            mma_16x8x8(dintra, a, b);
        }
        for (int k0 = 0; k0 < KP; k0 += 8) {   // Q·C_prevᵀ, contracting q
            ld_a_mk(a, sQ, LQ, m0, k0);
            ld_b_nk(b, sC, LQ, k0, n0);
            mma_16x8x8(dinter, a, b);
        }
        for (int i = 0; i < 4; ++i) {
            int t = m0 + mma_row(i), v = n0 + mma_col(i);
            if (t < len && v < dhv) {
                float acc = dintra[i] + sB[t] * dinter[i];
                ytil[((long)bh * T + c0 + t) * dhv + v] =
                    acc / fmaxf(fabsf(sQn[t]), expf(-sM[t]));
            }
        }
    }
}

#endif // MMA_TF32

// Backward over the chunk states: the mirror of `mlstm_fw_C`, walking chunks in
// reverse with the chunk loop inside the kernel. `dcst[k]` is the gradient wrt the
// state ENTERING chunk k (so it lines up index-for-index with `cst`):
//   dcst[k] = g_k · dcst[k+1] + Σ_t b_k[t]·d_num_k[t]⊗Q_k[t]
// dcst[NC] is zero — the state the last chunk produces is never read.
extern "C" __global__ void mlstm_bw_dC(
    const float* qq, const float* dytil, const float* ytil,
    const float* psiv, const float* qnv, const float* msv,
    const float* fcb, const float* mst,
    float* dcst, float* dnst,
    int T, int L, int NC, int dqk, int dhv, int TV) {
    int v0 = blockIdx.x * TV, bh = blockIdx.y;
    int tv = min(TV, dhv - v0);
    int tid = threadIdx.x, nthreads = blockDim.x;
    int LQ = dqk + 1, LV = tv + 1;
    int lead = (blockIdx.x == 0); // the tile that also owns `dn`

    extern __shared__ float sh[];
    float* sQ   = sh;                 // [L, LQ]
    float* sDN  = sQ + L * LQ;        // [L, LV]   d_num, the v-slice only
    float* sdC  = sDN + L * LV;       // [tv, LQ]
    float* sdN  = sdC + tv * LQ;      // [dqk]
    float* sB   = sdN + dqk;          // [L]
    float* sDQn = sB + L;             // [L]

    for (int e = tid; e < tv * dqk; e += nthreads) {
        int v = e / dqk, q = e - v * dqk;
        sdC[v * LQ + q] = 0.0f;
    }
    for (int e = tid; e < dqk; e += nthreads) sdN[e] = 0.0f;
    __syncthreads();

    for (int k = NC - 1; k >= 0; --k) {
        int c0 = k * L;
        int len = min(L, T - c0);

        // What is in sdC right now is the gradient wrt C_k — i.e. wrt the state
        // entering chunk k+1. Publish it there before folding chunk k in.
        float* dcout = dcst + ((long)bh * (NC + 1) + (k + 1)) * dhv * dqk;
        for (int e = tid; e < tv * dqk; e += nthreads) {
            int v = e / dqk, q = e - v * dqk;
            dcout[(long)(v0 + v) * dqk + q] = sdC[v * LQ + q];
        }
        if (lead) {
            float* dnout = dnst + ((long)bh * (NC + 1) + (k + 1)) * dqk;
            for (int e = tid; e < dqk; e += nthreads) dnout[e] = sdN[e];
        }

        float m_prev = mst[(long)bh * (NC + 1) + k];
        float fc_last = fcb[((long)bh * NC + k) * L + (len - 1)];
        float m_last = msv[(long)bh * T + c0 + len - 1];
        float gk = expf(fc_last + m_prev - m_last);

        for (int e = tid; e < len * dqk; e += nthreads) {
            int t = e / dqk, q = e - t * dqk;
            sQ[t * LQ + q] = qq[((long)bh * T + c0) * dqk + e];
        }

        // d_num = d_ytil/ψ and d_qn — the backward of ỹ = num/ψ, ψ = max(|qn|,1).
        // num is not saved: num = ỹ·ψ, so Σ_v d_ytil·num = ψ·Σ_v d_ytil·ỹ and the
        // ψ² cancels down to one division. d_qn contracts over ALL of dhv, so every
        // tile computes it (each reading the full d_ytil row) — only the v-slice of
        // d_num goes to shared memory.
        for (int t = tid; t < len; t += nthreads) {
            long gt = (long)bh * T + c0 + t;
            float inv = 1.0f / psiv[gt];
            float red = 0.0f;
            for (int v = 0; v < dhv; ++v) red += dytil[gt * dhv + v] * ytil[gt * dhv + v];
            for (int v = 0; v < tv; ++v)
                sDN[t * LV + v] = dytil[gt * dhv + v0 + v] * inv;
            float dpsi = -red * inv;
            float qn = qnv[gt];
            // Grad flows through qn only where it, not the exp(−m) floor, won the max.
            sDQn[t] = (fabsf(qn) > expf(-msv[gt]))
                          ? ((qn > 0.0f ? 1.0f : -1.0f) * dpsi)
                          : 0.0f;
            sB[t] = expf(fcb[((long)bh * NC + k) * L + t] + m_prev - msv[gt]);
        }
        __syncthreads();

        for (int e = tid; e < tv * dqk; e += nthreads) {
            int v = e / dqk, q = e - v * dqk;
            float acc = 0.0f;
            for (int t = 0; t < len; ++t) acc += sB[t] * sDN[t * LV + v] * sQ[t * LQ + q];
            sdC[v * LQ + q] = gk * sdC[v * LQ + q] + acc;
        }
        if (lead) {
            for (int q = tid; q < dqk; q += nthreads) {
                float acc = 0.0f;
                for (int t = 0; t < len; ++t) acc += sB[t] * sDQn[t] * sQ[t * LQ + q];
                sdN[q] = gk * sdN[q] + acc;
            }
        }
        __syncthreads();
    }

    float* dcout = dcst + (long)bh * (NC + 1) * dhv * dqk;
    for (int e = tid; e < tv * dqk; e += nthreads) {
        int v = e / dqk, q = e - v * dqk;
        dcout[(long)(v0 + v) * dqk + q] = sdC[v * LQ + q];
    }
    if (lead) {
        float* dnout = dnst + (long)bh * (NC + 1) * dqk;
        for (int e = tid; e < dqk; e += nthreads) dnout[e] = sdN[e];
    }
}

// One block per (chunk, bh): everything a chunk owes its inputs, given the state
// gradients the two recurrent kernels produced. S and D̄ are RECOMPUTED here from
// (Q, K, fc, ig, m) rather than saved — that is what keeps the [L, L] matrices off
// HBM. The shared DS buffer is read as DS (for dV) and then overwritten in place
// with dS, since each (t, j) is owned by exactly one thread.
//
// The gate gradients close out in the same block: `a`, `b` and `g` are chunk-local,
// so dfc/dig need no cross-block reduction, and the reverse cumsum that turns dfc
// into the forget-logit grad is a within-chunk scan.
extern "C" __global__ void mlstm_bw_parallel(
    const float* qq, const float* kk, const float* vv,
    const float* ig, const float* fg, const float* fcb,
    const float* cst, const float* nst, const float* mst,
    const float* dcst, const float* dnst,
    const float* ytil, const float* dytil, const float* psiv,
    const float* qnv, const float* msv,
    float* dq, float* dk, float* dv, float* dig, float* dfg,
    int T, int L, int NC, int dqk, int dhv) {
    int k = blockIdx.x, bh = blockIdx.y;
    int tid = threadIdx.x, nthreads = blockDim.x;
    int c0 = k * L;
    int len = min(L, T - c0);
    int is_last = (k == NC - 1);
    int LQ = dqk + 1, LV = dhv + 1, LS = L + 1;

    extern __shared__ float sh[];
    float* sQ   = sh;                 // [L, LQ]
    float* sK   = sQ + L * LQ;        // [L, LQ]
    float* sV   = sK + L * LQ;        // [L, LV]
    float* sDN  = sV + L * LV;        // [L, LV]    d_num
    float* sDS  = sDN + L * LV;       // [L, LS]    DS, then dS
    float* sC   = sDS + L * LS;       // [dhv, LQ]  C_{k-1}
    float* sdC  = sC + dhv * LQ;      // [dhv, LQ]  dC_k
    float* sN   = sdC + dhv * LQ;     // [dqk]      n_{k-1}
    float* sdN  = sN + dqk;           // [dqk]      dn_k
    float* sFc  = sdN + dqk;          // [L]
    float* sIg  = sFc + L;            // [L]
    float* sM   = sIg + L;            // [L]
    float* sB   = sM + L;             // [L]
    float* sA   = sB + L;             // [L]
    float* sQn  = sA + L;             // [L]
    float* sDQn = sQn + L;            // [L]
    float* sDfc = sDQn + L;           // [L]
    float* sDig = sDfc + L;           // [L]
    float* sDa  = sDig + L;           // [L]
    float* sDb  = sDa + L;            // [L]
    __shared__ float sRed[512]; // must cover FUSED_THREADS_PAR

    for (int e = tid; e < len * dqk; e += nthreads) {
        int t = e / dqk, q = e - t * dqk;
        sQ[t * LQ + q] = qq[((long)bh * T + c0) * dqk + e];
        sK[t * LQ + q] = kk[((long)bh * T + c0) * dqk + e];
    }
    for (int e = tid; e < len * dhv; e += nthreads) {
        int t = e / dhv, v = e - t * dhv;
        sV[t * LV + v] = vv[((long)bh * T + c0) * dhv + e];
    }
    for (int e = tid; e < dhv * dqk; e += nthreads) {
        int v = e / dqk, q = e - v * dqk;
        sC[v * LQ + q] = cst[((long)bh * (NC + 1) + k) * dhv * dqk + e];
        // The last chunk's outgoing state is never read, so its gradient is zero.
        sdC[v * LQ + q] =
            is_last ? 0.0f : dcst[((long)bh * (NC + 1) + (k + 1)) * dhv * dqk + e];
    }
    for (int e = tid; e < dqk; e += nthreads) {
        sN[e] = nst[((long)bh * (NC + 1) + k) * dqk + e];
        sdN[e] = is_last ? 0.0f : dnst[((long)bh * (NC + 1) + (k + 1)) * dqk + e];
    }
    for (int j = tid; j < L; j += nthreads) {
        sFc[j] = fcb[((long)bh * NC + k) * L + j];
        sIg[j] = (j < len) ? ig[(long)bh * T + c0 + j] : 0.0f;
        sDfc[j] = 0.0f;
        sDig[j] = 0.0f;
        sDa[j] = 0.0f;
        sDb[j] = 0.0f;
    }
    float m_prev = mst[(long)bh * (NC + 1) + k];
    __syncthreads();

    float fc_last = sFc[len - 1];
    float m_last = msv[(long)bh * T + c0 + len - 1];
    float gsca = expf(fc_last + m_prev - m_last);

    for (int t = tid; t < len; t += nthreads) {
        long gt = (long)bh * T + c0 + t;
        sM[t] = msv[gt];
        sB[t] = expf(sFc[t] + m_prev - sM[t]);
        sA[t] = expf(fc_last - sFc[t] + sIg[t] - m_last);
        sQn[t] = qnv[gt];

        float inv = 1.0f / psiv[gt];
        float red = 0.0f;
        for (int v = 0; v < dhv; ++v) {
            float dy = dytil[gt * dhv + v];
            sDN[t * LV + v] = dy * inv;
            red += dy * ytil[gt * dhv + v];
        }
        float dpsi = -red * inv;
        float qn = sQn[t];
        // Grad flows through qn only where it, not the exp(−m) floor, won the max.
        sDQn[t] = (fabsf(qn) > expf(-sM[t])) ? ((qn > 0.0f ? 1.0f : -1.0f) * dpsi) : 0.0f;
    }
    __syncthreads();

    // Recompute DS = D̄ ⊙ (Q·Kᵀ).
    for (int e = tid; e < len * len; e += nthreads) {
        int t = e / len, j = e - t * len;
        float val = 0.0f;
        if (j <= t) {
            float s = 0.0f;
            for (int q = 0; q < dqk; ++q) s += sQ[t * LQ + q] * sK[j * LQ + q];
            val = expf(sFc[t] - sFc[j] + sIg[j] - sM[t]) * s;
        }
        sDS[t * LS + j] = val;
    }
    __syncthreads();

    // dV: the num path (needs DS) plus the state-update path C_k = g·C + (a⊙V)ᵀ·K.
    //
    // `da` rides along here rather than in a loop of its own. da[j] contracts dC_k
    // over BOTH v and q — as its own loop that is `len` threads (32) each grinding
    // dhv·dqk (4096) iterations while the other 224 idle. But the inner q-contraction
    // is exactly the `st` this loop already computes, so accumulating V[j][v]·st into
    // a shared da[j] reuses it and spreads the work over all len·dhv elements.
    for (int e = tid; e < len * dhv; e += nthreads) {
        int j = e / dhv, v = e - j * dhv;
        float acc = 0.0f;
        for (int t = j; t < len; ++t) acc += sDS[t * LS + j] * sDN[t * LV + v];
        float st = 0.0f;
        for (int q = 0; q < dqk; ++q) st += sdC[v * LQ + q] * sK[j * LQ + q];
        atomicAdd(&sDa[j], sV[j * LV + v] * st);
        acc += sA[j] * st;
        dv[((long)bh * T + c0 + j) * dhv + v] = acc;
    }

    // db[t] contracts d_num against the pre-b inter read-out Q[t]·C_prevᵀ, which is
    // an [len, dhv] product — so it parallelizes over the same len·dhv grid instead
    // of over `len` alone.
    for (int e = tid; e < len * dhv; e += nthreads) {
        int t = e / dhv, v = e - t * dhv;
        float pre = 0.0f;
        for (int q = 0; q < dqk; ++q) pre += sQ[t * LQ + q] * sC[v * LQ + q];
        atomicAdd(&sDb[t], sDN[t * LV + v] * pre);
    }
    __syncthreads();

    // The n-side of da/db: both contract over dqk only, so `len` threads is enough.
    for (int j = tid; j < len; j += nthreads) {
        float acc = 0.0f, pre_qn = 0.0f;
        for (int q = 0; q < dqk; ++q) {
            acc += sdN[q] * sK[j * LQ + q];
            pre_qn += sQ[j * LQ + q] * sN[q];
        }
        sDa[j] += acc;
        sDb[j] += sDQn[j] * pre_qn;
    }

    // dDS -> (dS, P). P = dDS⊙DS feeds the decay grads; dS overwrites DS in place
    // (each (t, j) is owned by exactly one thread, so the read-then-write is safe).
    // dfc[t] += Σ_{j<=t} P[t][j] is a row sum; dfc[j] -= Σ_t P[t][j] and
    // dig[j] += Σ_t P[t][j] are column sums, so they go through shared atomics.
    for (int e = tid; e < len * len; e += nthreads) {
        int t = e / len, j = e - t * len;
        if (j <= t) {
            float ds_val = sDS[t * LS + j];
            float dds = sDQn[t];
            for (int v = 0; v < dhv; ++v) dds += sDN[t * LV + v] * sV[j * LV + v];
            float p = dds * ds_val;
            atomicAdd(&sDfc[t], p);
            atomicAdd(&sDfc[j], -p);
            atomicAdd(&sDig[j], p);
            float dbar = expf(sFc[t] - sFc[j] + sIg[j] - sM[t]);
            sDS[t * LS + j] = dds * dbar;
        } else {
            sDS[t * LS + j] = 0.0f;
        }
    }
    __syncthreads();

    // dQ: the intra path (dS·K) plus the two inter read-outs of the incoming state.
    for (int e = tid; e < len * dqk; e += nthreads) {
        int t = e / dqk, q = e - t * dqk;
        float acc = 0.0f;
        for (int j = 0; j <= t; ++j) acc += sDS[t * LS + j] * sK[j * LQ + q];
        float inter = 0.0f;
        for (int v = 0; v < dhv; ++v) inter += sDN[t * LV + v] * sC[v * LQ + q];
        acc += sB[t] * (inter + sDQn[t] * sN[q]);
        dq[((long)bh * T + c0 + t) * dqk + q] = acc;
    }

    // dK: the intra path (dSᵀ·Q) plus both state-update paths (C and n).
    for (int e = tid; e < len * dqk; e += nthreads) {
        int j = e / dqk, q = e - j * dqk;
        float acc = 0.0f;
        for (int t = j; t < len; ++t) acc += sDS[t * LS + j] * sQ[t * LQ + q];
        float st = 0.0f;
        for (int v = 0; v < dhv; ++v) st += sV[j * LV + v] * sdC[v * LQ + q];
        acc += sA[j] * (st + sdN[q]);
        dk[((long)bh * T + c0 + j) * dqk + q] = acc;
    }
    __syncthreads();

    // dg = Σ dC_k⊙C_{k-1} + Σ dn_k⊙n_{k-1}, a block-wide reduction.
    float loc = 0.0f;
    for (int e = tid; e < dhv * dqk; e += nthreads) {
        int v = e / dqk, q = e - v * dqk;
        loc += sdC[v * LQ + q] * sC[v * LQ + q];
    }
    for (int e = tid; e < dqk; e += nthreads) loc += sdN[e] * sN[e];
    sRed[tid] = loc;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sRed[tid] += sRed[tid + s];
        __syncthreads();
    }
    float dg = sRed[0];

    // a/b/g -> (dfc, dig), accumulating onto the intra-chunk D̄ contribution
    // (m held constant, as everywhere).
    for (int j = tid; j < len; j += nthreads) {
        float pa = sDa[j] * sA[j];
        atomicAdd(&sDig[j], pa);
        atomicAdd(&sDfc[j], sDb[j] * sB[j] - pa);
    }
    __syncthreads();
    if (tid == 0) {
        float acc = dg * gsca;
        for (int j = 0; j < len; ++j) acc += sDa[j] * sA[j];
        sDfc[len - 1] += acc;
    }
    __syncthreads();

    // dfc -> d(forget logit): reverse cumsum within the chunk, times logσ'.
    // Serial over len on one thread — len is 32 and this is the tail of the kernel.
    if (tid == 0) {
        float acc = 0.0f;
        for (int j = len - 1; j >= 0; --j) {
            acc += sDfc[j];
            long gt = (long)bh * T + c0 + j;
            dfg[gt] = acc * (1.0f - stable_sigmoid(fg[gt]));
            dig[gt] = sDig[j];
        }
    }
}

#if MMA_TF32

// The tensor-core twin of `mlstm_bw_parallel`. Every contraction in the backward
// is a dot, so every one of them goes to the tensor cores:
//
//   phase 1   S     = Q·Kᵀ         over dqk   -> masked+decayed into sDS (as fwd)
//   phase 2   dVnum = DSᵀ·dN       over t     |
//             st    = K·dCᵀ        over dqk   |- all three land on [L, dhv], so one
//             pre   = Q·Cᵀ         over dqk   |  warp pass computes all three
//   phase 4   dds   = dN·Vᵀ        over dhv   -> [L, L], epilogue makes dS in place
//   phase 5   dQint = dS·K         over j     |
//             dQinx = dN·C         over dhv   |- all four land on [L, dqk], so again
//             dKint = dSᵀ·Q        over t     |  one warp pass
//             dKst  = V·dC         over dhv   |
//
// Grouping by output tile is the point: an accumulator lives in registers for the
// whole K-loop, so two dots that write the same tile cost one tile's worth of
// epilogue and one set of shared-memory reads for the shared operand.
//
// `da`/`db` still reduce with shared atomics — they contract the SAME products the
// tiles already hold (`st` over v, `pre` over v), so they ride along in the epilogue
// instead of getting a pass of their own, exactly as in the scalar kernel.
//
// Everything else — the dg reduction, the a/b/g -> (dfc, dig) fold, the reverse
// cumsum tail — is elementwise or a scan, has no dot in it, and is unchanged.
extern "C" __global__ void mlstm_bw_parallel_mma(
    const float* qq, const float* kk, const float* vv,
    const float* ig, const float* fg, const float* fcb,
    const float* cst, const float* nst, const float* mst,
    const float* dcst, const float* dnst,
    const float* ytil, const float* dytil, const float* psiv,
    const float* qnv, const float* msv,
    float* dq, float* dk, float* dv, float* dig, float* dfg,
    int T, int L, int NC, int dqk, int dhv) {
    int k = blockIdx.x, bh = blockIdx.y;
    int tid = threadIdx.x, nthreads = blockDim.x;
    int warp = tid >> 5, nwarps = nthreads >> 5;
    int c0 = k * L;
    int len = min(L, T - c0);
    int is_last = (k == NC - 1);

    int LP = (L + 15) & ~15;
    int KP = (dqk + 7) & ~7;
    int VP = (dhv + 7) & ~7;
    int LQ = KP + 1, LV = VP + 1, LS = LP + 1;

    extern __shared__ float sh[];
    float* sQ   = sh;                  // [LP, LQ]
    float* sK   = sQ + LP * LQ;        // [LP, LQ]
    float* sV   = sK + LP * LQ;        // [LP, LV]
    float* sDN  = sV + LP * LV;        // [LP, LV]   d_num
    float* sDS  = sDN + LP * LV;       // [LP, LS]   DS, then dS
    float* sC   = sDS + LP * LS;       // [VP, LQ]   C_{k-1}
    float* sdC  = sC + VP * LQ;        // [VP, LQ]   dC_k
    float* sN   = sdC + VP * LQ;       // [KP]
    float* sdN  = sN + KP;             // [KP]
    float* sFc  = sdN + KP;            // [LP]
    float* sIg  = sFc + LP;            // [LP]
    float* sM   = sIg + LP;            // [LP]
    float* sB   = sM + LP;             // [LP]
    float* sA   = sB + LP;             // [LP]
    float* sQn  = sA + LP;             // [LP]
    float* sDQn = sQn + LP;            // [LP]
    float* sDfc = sDQn + LP;           // [LP]
    float* sDig = sDfc + LP;           // [LP]
    float* sDa  = sDig + LP;           // [LP]
    float* sDb  = sDa + LP;            // [LP]
    __shared__ float sRed[512]; // must cover FUSED_THREADS_PAR

    for (int e = tid; e < LP * KP; e += nthreads) {
        int t = e / KP, q = e - t * KP;
        int ok = (t < len) && (q < dqk);
        sQ[t * LQ + q] = ok ? qq[((long)bh * T + c0 + t) * dqk + q] : 0.0f;
        sK[t * LQ + q] = ok ? kk[((long)bh * T + c0 + t) * dqk + q] : 0.0f;
    }
    for (int e = tid; e < LP * VP; e += nthreads) {
        int t = e / VP, v = e - t * VP;
        sV[t * LV + v] = ((t < len) && (v < dhv))
            ? vv[((long)bh * T + c0 + t) * dhv + v] : 0.0f;
        sDN[t * LV + v] = 0.0f; // filled below for t < len
    }
    for (int e = tid; e < VP * KP; e += nthreads) {
        int v = e / KP, q = e - v * KP;
        int ok = (v < dhv) && (q < dqk);
        sC[v * LQ + q] =
            ok ? cst[((long)bh * (NC + 1) + k) * dhv * dqk + (long)v * dqk + q] : 0.0f;
        // The last chunk's outgoing state is never read, so its gradient is zero.
        sdC[v * LQ + q] = (ok && !is_last)
            ? dcst[((long)bh * (NC + 1) + (k + 1)) * dhv * dqk + (long)v * dqk + q] : 0.0f;
    }
    for (int e = tid; e < KP; e += nthreads) {
        int ok = e < dqk;
        sN[e] = ok ? nst[((long)bh * (NC + 1) + k) * dqk + e] : 0.0f;
        sdN[e] = (ok && !is_last) ? dnst[((long)bh * (NC + 1) + (k + 1)) * dqk + e] : 0.0f;
    }
    for (int j = tid; j < LP; j += nthreads) {
        sFc[j] = (j < L)   ? fcb[((long)bh * NC + k) * L + j] : 0.0f;
        sIg[j] = (j < len) ? ig[(long)bh * T + c0 + j]        : 0.0f;
        sM[j] = 0.0f; sB[j] = 0.0f; sA[j] = 0.0f; sQn[j] = 0.0f; sDQn[j] = 0.0f;
        sDfc[j] = 0.0f; sDig[j] = 0.0f; sDa[j] = 0.0f; sDb[j] = 0.0f;
    }
    float m_prev = mst[(long)bh * (NC + 1) + k];
    __syncthreads();

    float fc_last = sFc[len - 1];
    float m_last = msv[(long)bh * T + c0 + len - 1];
    float gsca = expf(fc_last + m_prev - m_last);

    for (int t = tid; t < len; t += nthreads) {
        long gt = (long)bh * T + c0 + t;
        sM[t] = msv[gt];
        sB[t] = expf(sFc[t] + m_prev - sM[t]);
        sA[t] = expf(fc_last - sFc[t] + sIg[t] - m_last);
        sQn[t] = qnv[gt];

        float inv = 1.0f / psiv[gt];
        float red = 0.0f;
        for (int v = 0; v < dhv; ++v) {
            float dy = dytil[gt * dhv + v];
            sDN[t * LV + v] = dy * inv;
            red += dy * ytil[gt * dhv + v];
        }
        float dpsi = -red * inv;
        float qn = sQn[t];
        // Grad flows through qn only where it, not the exp(−m) floor, won the max.
        sDQn[t] = (fabsf(qn) > expf(-sM[t])) ? ((qn > 0.0f ? 1.0f : -1.0f) * dpsi) : 0.0f;
    }
    __syncthreads();

    int mtile = LP >> 4, ltile = LP >> 3, vtile = VP >> 3, ktile = KP >> 3;

    // Phase 1: recompute DS = D̄ ⊙ (Q·Kᵀ), exactly as the forward built it.
    for (int tile = warp; tile < mtile * ltile; tile += nwarps) {
        int m0 = (tile / ltile) << 4, n0 = (tile % ltile) << 3;
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        unsigned a[4], b[2];
        for (int k0 = 0; k0 < KP; k0 += 8) {
            ld_a_mk(a, sQ, LQ, m0, k0);
            ld_b_nk(b, sK, LQ, k0, n0);
            mma_16x8x8(d, a, b);
        }
        for (int i = 0; i < 4; ++i) {
            int t = m0 + mma_row(i), j = n0 + mma_col(i);
            float val = 0.0f;
            if (t < len && j <= t) val = expf(sFc[t] - sFc[j] + sIg[j] - sM[t]) * d[i];
            sDS[t * LS + j] = val;
        }
    }
    __syncthreads();

    // Phase 2: dV, plus the `st` and `pre` products that `da`/`db` reduce over v.
    // Three dots, one [L, dhv] output tile, one warp pass.
    for (int tile = warp; tile < mtile * vtile; tile += nwarps) {
        int m0 = (tile / vtile) << 4, n0 = (tile % vtile) << 3;
        float dnum[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Σ_t DS[t][j]·dN[t][v]
        float dst[4]  = {0.0f, 0.0f, 0.0f, 0.0f};  // Σ_q dC[v][q]·K[j][q]
        float dpre[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Σ_q Q[t][q]·C[v][q]
        unsigned a[4], b[2];
        // DS[t][j] is zero for j > t, so contracting over ALL t is the same as t >= j.
        for (int k0 = 0; k0 < LP; k0 += 8) {
            ld_a_km(a, sDS, LS, m0, k0);   // Aᵀ in memory: DS is [t, j], we want [j, t]
            ld_b_kn(b, sDN, LV, k0, n0);
            mma_16x8x8(dnum, a, b);
        }
        for (int k0 = 0; k0 < KP; k0 += 8) {
            ld_a_mk(a, sK, LQ, m0, k0);
            ld_b_nk(b, sdC, LQ, k0, n0);
            mma_16x8x8(dst, a, b);
            ld_a_mk(a, sQ, LQ, m0, k0);
            ld_b_nk(b, sC, LQ, k0, n0);
            mma_16x8x8(dpre, a, b);
        }
        for (int i = 0; i < 4; ++i) {
            int r = m0 + mma_row(i), v = n0 + mma_col(i); // r is `j` for dv, `t` for db
            if (r < len && v < dhv) {
                float st = dst[i];
                dv[((long)bh * T + c0 + r) * dhv + v] = dnum[i] + sA[r] * st;
                atomicAdd(&sDa[r], sV[r * LV + v] * st);
                atomicAdd(&sDb[r], sDN[r * LV + v] * dpre[i]);
            }
        }
    }
    __syncthreads();

    // The n-side of da/db: both contract over dqk only — a matrix-vector product, so
    // there is no dot here for the tensor cores and `len` threads is enough.
    for (int j = tid; j < len; j += nthreads) {
        float acc = 0.0f, pre_qn = 0.0f;
        for (int q = 0; q < dqk; ++q) {
            acc += sdN[q] * sK[j * LQ + q];
            pre_qn += sQ[j * LQ + q] * sN[q];
        }
        sDa[j] += acc;
        sDb[j] += sDQn[j] * pre_qn;
    }
    __syncthreads();

    // Phase 4: dDS -> (dS, P). dS overwrites DS in place; the warp that owns an
    // output tile is the only one that reads or writes those elements, and the
    // barrier above guarantees phase 2 is done reading DS.
    for (int tile = warp; tile < mtile * ltile; tile += nwarps) {
        int m0 = (tile / ltile) << 4, n0 = (tile % ltile) << 3;
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};   // Σ_v dN[t][v]·V[j][v]
        unsigned a[4], b[2];
        for (int k0 = 0; k0 < VP; k0 += 8) {
            ld_a_mk(a, sDN, LV, m0, k0);
            ld_b_nk(b, sV, LV, k0, n0);
            mma_16x8x8(d, a, b);
        }
        for (int i = 0; i < 4; ++i) {
            int t = m0 + mma_row(i), j = n0 + mma_col(i);
            float out = 0.0f;
            if (t < len && j <= t) {
                float dds = d[i] + sDQn[t];
                float p = dds * sDS[t * LS + j];
                atomicAdd(&sDfc[t], p);
                atomicAdd(&sDfc[j], -p);
                atomicAdd(&sDig[j], p);
                out = dds * expf(sFc[t] - sFc[j] + sIg[j] - sM[t]);
            }
            sDS[t * LS + j] = out;
        }
    }
    __syncthreads();

    // Phase 5: dQ and dK. Four dots, one [L, dqk] output tile, one warp pass.
    for (int tile = warp; tile < mtile * ktile; tile += nwarps) {
        int m0 = (tile / ktile) << 4, n0 = (tile % ktile) << 3;
        float dqi[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Σ_j dS[t][j]·K[j][q]
        float dqx[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Σ_v dN[t][v]·C[v][q]
        float dki[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Σ_t dS[t][j]·Q[t][q]
        float dks[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Σ_v V[j][v]·dC[v][q]
        unsigned a[4], b[2];
        for (int k0 = 0; k0 < LP; k0 += 8) {
            ld_a_mk(a, sDS, LS, m0, k0);   // dS as [t, j], contracting j
            ld_b_kn(b, sK, LQ, k0, n0);
            mma_16x8x8(dqi, a, b);
            ld_a_km(a, sDS, LS, m0, k0);   // dSᵀ, contracting t
            ld_b_kn(b, sQ, LQ, k0, n0);
            mma_16x8x8(dki, a, b);
        }
        for (int k0 = 0; k0 < VP; k0 += 8) {
            ld_a_mk(a, sDN, LV, m0, k0);
            ld_b_kn(b, sC, LQ, k0, n0);
            mma_16x8x8(dqx, a, b);
            ld_a_mk(a, sV, LV, m0, k0);
            ld_b_kn(b, sdC, LQ, k0, n0);
            mma_16x8x8(dks, a, b);
        }
        for (int i = 0; i < 4; ++i) {
            int r = m0 + mma_row(i), q = n0 + mma_col(i); // r is `t` for dq, `j` for dk
            if (r < len && q < dqk) {
                dq[((long)bh * T + c0 + r) * dqk + q] =
                    dqi[i] + sB[r] * (dqx[i] + sDQn[r] * sN[q]);
                dk[((long)bh * T + c0 + r) * dqk + q] =
                    dki[i] + sA[r] * (dks[i] + sdN[q]);
            }
        }
    }
    __syncthreads();

    // dg = Σ dC_k⊙C_{k-1} + Σ dn_k⊙n_{k-1}, a block-wide reduction. The pad is zero
    // in both operands, so it contributes nothing and needs no masking.
    float loc = 0.0f;
    for (int e = tid; e < VP * KP; e += nthreads) {
        int v = e / KP, q = e - v * KP;
        loc += sdC[v * LQ + q] * sC[v * LQ + q];
    }
    for (int e = tid; e < KP; e += nthreads) loc += sdN[e] * sN[e];
    sRed[tid] = loc;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sRed[tid] += sRed[tid + s];
        __syncthreads();
    }
    float dg = sRed[0];

    // a/b/g -> (dfc, dig), accumulating onto the intra-chunk D̄ contribution
    // (m held constant, as everywhere).
    for (int j = tid; j < len; j += nthreads) {
        float pa = sDa[j] * sA[j];
        atomicAdd(&sDig[j], pa);
        atomicAdd(&sDfc[j], sDb[j] * sB[j] - pa);
    }
    __syncthreads();
    if (tid == 0) {
        float acc = dg * gsca;
        for (int j = 0; j < len; ++j) acc += sDa[j] * sA[j];
        sDfc[len - 1] += acc;
    }
    __syncthreads();

    if (tid == 0) {
        float acc = 0.0f;
        for (int j = len - 1; j >= 0; --j) {
            acc += sDfc[j];
            long gt = (long)bh * T + c0 + j;
            dfg[gt] = acc * (1.0f - stable_sigmoid(fg[gt]));
            dig[gt] = sDig[j];
        }
    }
}

#endif // MMA_TF32
"#;

/// Names of every kernel in [`SRC`], loaded into [`Kernels`].
const NAMES: &[&str] = &[
    "softcap_forward",
    "softcap_backward",
    "broadcast_row",
    "add_col_sum",
    "embedding_gather",
    "embedding_scatter_add",
    "rms_norm_forward",
    "rms_norm_backward",
    "softmax_ce",
    "adamw",
    "concat_xh",
    "split_dxh",
    "slstm_cell_step",
    "slstm_cell_step_bwd",
    "slstm_pack_w",
    "slstm_pack_b",
    "slstm_unpack_dw",
    "slstm_unpack_db",
    "fill_const",
    "slstm_step_fused",
    "slstm_step_fused_bwd",
    "add",
    "swiglu_forward",
    "swiglu_backward",
    "scale_inplace",
    "sigmoid_inplace",
    "head_gather",
    "head_scatter",
    "cumsum_logsig",
    "mlstm_rowmax_m",
    "mlstm_ds",
    "div_rows",
    "mul",
    "slice_t",
    "unslice_t",
    "mlstm_chunk_ab",
    "mul_rows",
    "mul_rows_add",
    "psi_from_qn",
    "row_dot_add",
    "group_dot_add",
    "mlstm_chunk_ab_bwd",
    "ogate_bwd",
    "div_rows_bwd",
    "mlstm_ds_bwd",
    "mlstm_dfc_dig",
    "revcumsum_dlogsig",
    "mlstm_fw_gates",
    "mlstm_fw_C",
    "mlstm_fw_parallel",
    "mlstm_bw_dC",
    "mlstm_bw_parallel",
    "scatter_rows",
    "masked_softmax_ce",
];

/// Kernels that exist only when the device has tensor cores (`MMA_TF32`, sm_80+).
/// Each is a drop-in twin of the same-named kernel without the suffix.
const MMA_NAMES: &[&str] = &["mlstm_fw_parallel_mma", "mlstm_bw_parallel_mma"];

/// All compiled kernels, held by name. Cloneable (each `CudaFunction` is an
/// `Arc`-backed handle) and cheap to look up.
pub struct Kernels {
    funcs: std::collections::HashMap<&'static str, CudaFunction>,
    /// Whether [`MMA_NAMES`] were compiled — i.e. the device is sm_80 or later.
    pub has_mma: bool,
}

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
        let has_mma = major >= 8;

        let base = compile_ptx(SRC).map_err(|e| format!("NVRTC compile failed: {e:?}"))?;
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

        if has_mma {
            let opts = CompileOptions {
                options: vec![
                    format!("--gpu-architecture=compute_{major}{minor}"),
                    "-DMMA_TF32=1".to_string(),
                ],
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

        Ok(Self { funcs, has_mma })
    }

    /// Look up a kernel by name (panics if it was not in [`NAMES`]).
    pub fn get(&self, name: &str) -> CudaFunction {
        self.funcs
            .get(name)
            .unwrap_or_else(|| panic!("unknown kernel {name}"))
            .clone()
    }
}
