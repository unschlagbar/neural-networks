// Elementwise ops, embedding gather/scatter, RMSNorm, softmax-CE, AdamW and the
// eager sLSTM cell (step + backward, gate pack/unpack).

extern "C" __global__ void softcap_forward(const float* x, float* y, float cap, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = cap * tanhf(x[i] / cap);
}

extern "C" __global__ void softcap_backward(const float* dy, const float* y, float* dx, float cap, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float t = y[i] / cap; dx[i] = dy[i] * (1.0f - t * t); }
}

// Copy bias[n] into every row of out[rows, n].
//
// Grid-stride over rows with the column from `threadIdx.x`, rather than one flat index
// and `bias[i % n]`: the modulo was an integer division per element, and this shape is
// pure bandwidth (every projection seeds its output here before the GEMM accumulates
// on top). Threads of a warp still hold adjacent columns, so stores stay coalesced.
extern "C" __global__ void broadcast_row(float* out, const float* bias, int rows, int n) {
    for (int r = blockIdx.x; r < rows; r += gridDim.x) {
        const float* b = bias;
        float* o = out + (size_t)r * n;
        for (int c = threadIdx.x; c < n; c += blockDim.x) o[c] = b[c];
    }
}

// out[r, c] = resid[r, c] + bias[c] — `broadcast_row` with a residual folded in.
//
// A projection that feeds a residual seeds its output here instead, so the trailing
// `y = resid + proj(x)` add costs no kernel of its own: the GEMM then accumulates on
// top at beta = 1 exactly as before. The add it replaces was a separate [N, H] pass
// running at ~92 GB/s of a ~900 GB/s card, i.e. launch-bound rather than
// bandwidth-bound, so what this saves is the launch, not the traffic.
extern "C" __global__ void broadcast_row_resid(
        float* out, const float* resid, const float* bias, int rows, int n) {
    for (int r = blockIdx.x; r < rows; r += gridDim.x) {
        const float* s = resid + (size_t)r * n;
        float* o = out + (size_t)r * n;
        for (int c = threadIdx.x; c < n; c += blockDim.x) o[c] = s[c] + bias[c];
    }
}

// db[o] += sum over rows of dy[r*n + o], and of dy[r*n + o]*mul[r*n + o] under
// `use_mul` (`dgamma` for RMSNorm, which is the same reduction over an elementwise
// product). `mul` is a live pointer either way — the caller passes `dy` again when it
// has no second operand, which costs nothing and keeps the argument non-null.
//
// The row axis is split across `threadIdx.y` and folded by a fixed-order tree — NOT
// by an atomicAdd across blocks. Float addition is not associative, so an atomic made
// the last bits of every bias gradient depend on the order the blocks happened to be
// scheduled in; one training step later that is a different model. Here ONE block
// owns a column tile and every row of it, so the summation order is a property of the
// shape alone and two runs of the same shape agree bit for bit.
//
// A thread still owns one column and a slice of the rows, so the parallelism the
// atomic bought is kept: it moved from `blockIdx.y` into `threadIdx.y`. Threads of a
// warp hold adjacent `o`, so each row read stays coalesced.
//
// `db` is an accumulator (`+=`) across calls; the caller zeroes it between steps.
extern "C" __global__ void add_col_sum(float* db, const float* dy, const float* mul,
                                       int use_mul, int rows, int n) {
    extern __shared__ float shcs[];
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    float s = 0.0f;
    if (o < n) {
        for (int r = threadIdx.y; r < rows; r += blockDim.y) {
            long i = (long)r * n + o;
            s += use_mul ? dy[i] * mul[i] : dy[i];
        }
    }
    shcs[tid] = s;
    __syncthreads();
    // blockDim.y is a power of two (the launcher picks it), so the tree is exact.
    for (int half = blockDim.y >> 1; half > 0; half >>= 1) {
        if (threadIdx.y < half) shcs[tid] += shcs[tid + half * blockDim.x];
        __syncthreads();
    }
    if (threadIdx.y == 0 && o < n) db[o] += shcs[tid];
}

// `add_col_sum` with the row axis cut into `bands` of `band` rows, one band per
// `blockIdx.y`, each writing its own `part[band, n]` row.
//
// The single-block form above owns a column tile and EVERY row of it, so its grid is
// `ceil(n / 32)` blocks — 8 blocks at `[2045, 256]`, one at `[rows, 16]`. That reads
// at a tenth of the machine's bandwidth: the work is there, the parallelism is not.
// Banding puts the row axis back into the grid.
//
// Still not an atomicAdd: `band` is a function of the shape alone, so which rows a
// band holds, the tree inside it, and the order `col_sum_merge` folds the bands in
// are all fixed by the shape, and two runs of it agree bit for bit.
extern "C" __global__ void col_sum_part(float* part, const float* dy, const float* mul,
                                        int use_mul, int rows, int n, int band) {
    extern __shared__ float shcs[];
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int r0 = blockIdx.y * band;
    const int r1 = min(r0 + band, rows);
    float s = 0.0f;
    if (o < n) {
        for (int r = r0 + threadIdx.y; r < r1; r += blockDim.y) {
            long i = (long)r * n + o;
            s += use_mul ? dy[i] * mul[i] : dy[i];
        }
    }
    shcs[tid] = s;
    __syncthreads();
    for (int half = blockDim.y >> 1; half > 0; half >>= 1) {
        if (threadIdx.y < half) shcs[tid] += shcs[tid + half * blockDim.x];
        __syncthreads();
    }
    if (threadIdx.y == 0 && o < n) part[(long)blockIdx.y * n + o] = shcs[tid];
}

// db[o] += sum of the bands `col_sum_part` wrote, in ascending band order.
extern "C" __global__ void col_sum_merge(float* db, const float* part, int bands, int n) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n) return;
    float s = 0.0f;
    for (int p = 0; p < bands; ++p) s += part[(long)p * n + o];
    db[o] += s;
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

// dtable[ids[r], :] += dy[r, :], where ids REPEAT — the whole difficulty. An
// atomicAdd per element is the obvious answer and is not reproducible: a token that
// occurs 3000 times in a window has its 3000 contributions summed in whatever order
// the blocks were scheduled in, and float addition does not associate.
//
// Instead the row axis is cut into `slices`, and one thread owns one (slice, column)
// for the whole slice: it walks its rows in ASCENDING order into its slice's private
// table, so no two threads ever touch the same slot. `embedding_scatter_merge` then
// folds the slices in slice order. Both orders are properties of the shape.
//
// With one slice the "private table" is `dtable` itself and no merge is needed.
extern "C" __global__ void embedding_scatter_add(float* part, const unsigned* ids,
                                                 const float* dy, int dim, int rows,
                                                 int vocab, int slices) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= dim) return;
    const int s = blockIdx.y;
    float* p = part + (long)s * vocab * dim + c;
    // Balanced to the last row: `rows` need not divide `slices`.
    const int lo = (int)((long)rows * s / slices);
    const int hi = (int)((long)rows * (s + 1) / slices);
    for (int r = lo; r < hi; ++r) p[(long)ids[r] * dim] += dy[(long)r * dim + c];
}

// Fold `embedding_scatter_add`'s per-slice tables into `dtable`, slices ascending.
extern "C" __global__ void embedding_scatter_merge(float* dtable, const float* part,
                                                   int n, int slices) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0.0f;
    for (int k = 0; k < slices; ++k) s += part[(long)k * n + i];
    dtable[i] += s;
}


// Grouped RMSNorm backward. dgamma is shared across rows -> atomicAdd.
// Block-per-group RMSNorm. One CUDA BLOCK owns one (row, group) and reduces over
// the group cooperatively, instead of one thread walking it serially.
//
// The thread-per-group kernels below are correct but pathologically parallel-poor at
// the shape this model actually runs: a block's norms are ungrouped, so
// `group == hidden == 1024` and `total_groups == rows`. At T=1024 that launched
// 1024 threads for the whole kernel — on an 84-SM card — with each thread doing a
// 1024-element serial pass, twice. Measured, the three norms plus the residual adds
// ("block glue") came to 44.7% of a training step, MORE than every recurrent cell
// combined. This is that fix.
//
// One block per group turns the group loop into a strided read plus a tree
// reduction: 256 threads each handle group/256 elements, a warp shuffle folds each
// warp, and one shared slot per warp finishes it. The launch goes from
// `total_groups` threads to `total_groups` BLOCKS.
//
// Grid: total_groups blocks. Block: RMSN_THREADS threads. Shared: one float per warp.
#define RMSN_THREADS 256

__device__ __forceinline__ float rmsn_block_sum(float v, float* sh) {
    // Fold within each warp first (no shared traffic), then across warps.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    int nwarps = blockDim.x >> 5;
    // The first warp reduces the per-warp partials; broadcast through sh[0].
    if (warp == 0) {
        float t = (lane < nwarps) ? sh[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) t += __shfl_down_sync(0xffffffff, t, off);
        if (lane == 0) sh[0] = t;
    }
    __syncthreads();
    return sh[0];
}

extern "C" __global__ void rms_norm_forward(const float* x, const float* gamma, float* out,
                                                float* x_hat, float* inv_rms,
                                                int groups_per_row, int group, float eps,
                                                int total_groups) {
    int gi = blockIdx.x;
    if (gi >= total_groups) return;
    extern __shared__ float sh[];
    int row = gi / groups_per_row;
    int grp = gi - row * groups_per_row;
    long long off = (long long)row * groups_per_row * group + (long long)grp * group;
    int g_off = grp * group;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < group; i += blockDim.x) {
        float v = x[off + i];
        ss += v * v;
    }
    ss = rmsn_block_sum(ss, sh);
    float inv = rsqrtf(ss / (float)group + eps);
    if (threadIdx.x == 0) inv_rms[gi] = inv;

    for (int i = threadIdx.x; i < group; i += blockDim.x) {
        float xh = x[off + i] * inv;
        x_hat[off + i] = xh;
        out[off + i] = gamma[g_off + i] * xh;
    }
}

// Backward twin — `dx` only. `dgamma` is a sum over ROWS of `dy ⊙ x_hat`, which
// every block of this grid would have to contribute to; that is `add_col_sum`'s
// reduction, and doing it here needed an atomicAdd per element, which is not
// reproducible. The caller runs `add_col_sum(dgamma, dy, x_hat, ..)` instead.
extern "C" __global__ void rms_norm_backward(const float* dy, const float* x_hat,
                                                 const float* inv_rms, const float* gamma,
                                                 float* dx,
                                                 int groups_per_row, int group,
                                                 int total_groups) {
    int gi = blockIdx.x;
    if (gi >= total_groups) return;
    extern __shared__ float sh[];
    int row = gi / groups_per_row;
    int grp = gi - row * groups_per_row;
    long long off = (long long)row * groups_per_row * group + (long long)grp * group;
    int g_off = grp * group;
    float inv = inv_rms[gi];

    float s = 0.0f;
    for (int i = threadIdx.x; i < group; i += blockDim.x)
        s += gamma[g_off + i] * dy[off + i] * x_hat[off + i];
    s = rmsn_block_sum(s, sh);
    float s_over_g = s / (float)group;

    for (int i = threadIdx.x; i < group; i += blockDim.x) {
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

// AdamW over a whole parameter arena in one launch (see gpu/arena.rs). One thread
// per element; bc1/bc2 (the bias corrections) are precomputed on the host.
//
// The arena packs decayed parameters first and frozen ones last, so `decay_end` and
// `n` replace what would otherwise be a per-tensor decay flag and bounds.
//
// What this buys is launches, not bandwidth. Measured against the two things Apex's
// fused AdamW does:
//   - `float4` (its ILP=4 vectorized path): total adamw time 71.2 -> 71.1 ms, i.e. no
//     change. A warp's scalar accesses already coalesce into full sectors.
//   - multi-tensor apply: 66% of this model's per-tensor adamw launches were under
//     4 us but only 10% of its time — the kernel time sits in mid-sized tensors that
//     are bandwidth-bound, and no amount of batching moves that.
// The step still drops from ~900 launches and ~900 memsets to one of each, and the
// arena is what gives every parameter a stable address.
extern "C" __global__ void adamw_arena(float* param, const float* grad, float* m, float* v,
                                       float lr, float b1, float b2, float eps, float wd,
                                       float bc1, float bc2, int decay_end, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    float g = grad[k];
    float mk = b1 * m[k] + (1.0f - b1) * g;
    float vk = b2 * v[k] + (1.0f - b2) * g * g;
    m[k] = mk; v[k] = vk;
    float mh = mk / bc1;
    float vh = vk / bc2;
    float p = param[k];
    p -= lr * (k < decay_end ? wd : 0.0f) * p;
    p -= lr * mh / (sqrtf(vh) + eps);
    param[k] = p;
}

// The same update over one tensor, with `wd` fixed for all of it: what a layer used
// on its own and the parity tests step through.
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
        float* c_prev, float* n_prev, slab_t* zt, slab_t* ot,
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
    slab_st(zt, k, z); slab_st(ot, k, o);
    i_prime[k] = ip; f_prime[k] = fp;
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
        const slab_t* ot, const float* c, const float* n,
        const float* c_prev, const float* n_prev, const slab_t* zt,
        const float* i_prime, const float* f_prime, const float* ft_pre,
        float* dc_bptt, float* dn_bptt,
        float* dz, float* di, float* df, float* dob, int H, int BH) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= BH) return;
    int b = k / H, j = k % H;
    float d = dy[(b * T + t) * H + j] + dh_bptt[k];
    float o = slab_ld(ot, k);
    float cc = c[k];
    float nn = n[k];
    // h = o·c/n — no clamp, so dn has no branch: ∂h/∂n = −o·c/n².
    dob[k] = d * (cc / nn) * o * (1.0f - o);
    float dc = d * o / nn + dc_bptt[k];
    float dn = d * o * (-cc) / (nn * nn) + dn_bptt[k];
    float fp = f_prime[k];
    float df_prime = dc * c_prev[k] + dn * n_prev[k];
    float ztk = slab_ld(zt, k);
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

// Copy one slot per `bh` between a flat `[BH, stride]` state and slot `idx` of a
// `[BH, slots, stride]` chunk-state array.
//
// This replaces a host-side `for b in 0..bh { memcpy_dtod(...) }`. That loop issued
// one tiny async copy per `bh`, and the encoder runs at `bh = words_in_group * heads`
// — thousands of launches for a few hundred KB. `cuMemcpyDtoDAsync` was 53% of all
// CUDA API time in an nsys trace, ~54k calls per step, almost all of it from here.
//
// `gather` picks the direction: 0 writes the slot (seed), 1 reads it (extract).
extern "C" __global__ void state_slot_copy(const float* src, float* dst,
                                           int bh, int slots, int idx, int stride,
                                           int gather) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)bh * stride;
    if (i >= total) return;
    long b = i / stride, e = i - b * stride;
    long slotted = (b * slots + idx) * stride + e; // the [BH, slots, stride] side
    long flat = i;                                 // the [BH, stride] side
    if (gather) dst[flat] = src[slotted];
    else dst[slotted] = src[flat];
}

// `*dst += inv * sum(src[..n])` — a whole-array reduction into a single accumulator.
//
// One block, grid-stride load then a shared-memory tree. The caller reduces at most a
// decoder group's rows, and the point is to keep the per-group loss on the device: the
// host-side alternative was a blocking `clone_dtoh` per group, which drained the stream
// and page-locked a staging buffer in the middle of the decode loop.
extern "C" __global__ void sum_accum(const float* src, float* dst, int n, float inv) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) acc += src[i];
    sdata[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) *dst += inv * sdata[0];
}
