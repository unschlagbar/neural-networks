// mLSTM parallel/chunkwise core (nn2/mlstm.rs), op-at-a-time: head gather/scatter,
// the gate scans, the chunk A/B terms and their backward twins. One launch per op —
// the reference the fused TFLA kernels in mlstm_fused.cu are checked against.

// Block-wide sum of `v`, returned on thread 0 (garbage on the others).
// Assumes at most 32 warps per block.
//
// `parts` is padded to a full warp with the reduction's identity. Blocks here are
// as narrow as 32 threads, so the final warp-0 shuffle has lanes beyond `nwarps`;
// with `0xffffffff` they participate, and reading an unwritten `parts` slot folds
// whatever the previous block left in that shared memory into the result. That is
// silent, nondeterministic gradient corruption, not a crash.
//
// Every thread runs to the end — no early return — so the caller may reduce twice.
// The trailing __syncthreads() is what makes a second call safe: it keeps a fast
// warp from overwriting `parts` while a slow one is still reading the previous
// round out of it.
__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float parts[32];
    if (threadIdx.x < 32) parts[threadIdx.x] = 0.0f;
    __syncthreads();
    for (int off = warpSize / 2; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) parts[wid] = v;
    __syncthreads();
    float t = parts[threadIdx.x & 31];
    if (wid == 0) {
        for (int off = warpSize / 2; off > 0; off >>= 1) t += __shfl_down_sync(0xffffffff, t, off);
    }
    __syncthreads();
    return t;
}

// Block-wide max of `v`, returned on thread 0. Same contract as block_reduce_sum,
// including the identity padding — see the note there.
__device__ __forceinline__ float block_reduce_max(float v) {
    __shared__ float parts[32];
    if (threadIdx.x < 32) parts[threadIdx.x] = -1e30f;
    __syncthreads();
    for (int off = warpSize / 2; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) parts[wid] = v;
    __syncthreads();
    float t = parts[threadIdx.x & 31];
    if (wid == 0) {
        for (int off = warpSize / 2; off > 0; off >>= 1)
            t = fmaxf(t, __shfl_down_sync(0xffffffff, t, off));
    }
    __syncthreads();
    return t;
}

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

// `head_gather` writing a bf16 destination, for the mLSTM's saved q/k/v.
//
// This must narrow AT THE GATHER, not afterwards. Producing the fp32 tensor first
// and converting it costs the full-width allocation anyway — and because cudarc
// frees through an async memory pool that retains blocks, that fp32 buffer stays
// resident, so the pair costs MORE than fp32 alone. Measured: doing it the naive
// way made a step 32 MB larger, not smaller. Writing narrow from the start is what
// actually removes the wide buffer.
extern "C" __global__ void head_gather_slab(const float* x, slab_t* out, int B, int H, int T, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * T * W) return;
    int c = idx % W;
    int t = (idx / W) % T;
    int h = (idx / (W * T)) % H;
    int b = idx / (W * T * H);
    slab_st(out, idx, x[(b * T + t) * (H * W) + h * W + c]);
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

// `head_scatter` reading a bf16 source. The mLSTM's `ytil` is stored narrow, and
// this is the one consumer that reads it outside the fused kernels; widening it
// into a temporary first would allocate exactly the fp32 buffer the narrow storage
// exists to avoid, so the conversion happens here, on the fly. The destination is
// fp32 either way — it feeds the head norm.
extern "C" __global__ void head_scatter_slab(const slab_t* in, float* x, int B, int H, int T, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * T * W) return;
    int c = idx % W;
    int t = (idx / W) % T;
    int h = (idx / (W * T)) % H;
    int b = idx / (W * T * H);
    x[(b * T + t) * (H * W) + h * W + c] = slab_ld(in, idx);
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
// One BLOCK per (g,t) over BH*T. `ig` is the input-gate logit [BH,T].
extern "C" __global__ void mlstm_rowmax_m(const float* fc, const float* ig, const float* m_prev,
                                          float* m, int T, int BHT) {
    int idx = blockIdx.x;
    if (idx >= BHT) return;
    int t = idx % T, g = idx / T;
    float fct = fc[g * T + t];
    float mx = -1e30f;
    for (int j = threadIdx.x; j <= t; j += blockDim.x) {
        float ld = fct - fc[g * T + j] + ig[g * T + j];
        mx = fmaxf(mx, ld);
    }
    mx = block_reduce_max(mx);
    if (threadIdx.x != 0) return;
    m[idx] = fmaxf(mx, fct + m_prev[g]);
}

// DS = D̄ ⊙ S with D̄_{tj}=exp(fc_t-fc_j+ig_j-m_t) (j<=t else 0); also the row
// normalizer qn_t = Σ_j DS_{tj} and ψ_t = max(|qn_t|,1).
// S/DS are [BH,T,T] row-major (row t of head g at g*T*T + t*T).
//
// One BLOCK per row (g,t), lanes walking j. A thread per row makes adjacent lanes
// read S a full row apart, so no access in a warp shares a sector.
//
// Only DS is zeroed above the diagonal: it is consumed by a cuBLAS GEMM, which
// reads the whole rectangle. `mlstm_ds_bwd` re-derives the j<=t mask itself and
// never reads D̄ above the diagonal, so writing those zeros is pure traffic.
extern "C" __global__ void mlstm_ds(const float* S, const float* fc, const float* ig, const float* m,
                                    float* Dbar, float* DS, float* qn, float* psi, int T, int BHT) {
    int idx = blockIdx.x;
    if (idx >= BHT) return;
    int t = idx % T, g = idx / T;
    float fct = fc[g * T + t], mt = m[idx];
    long base = (long)g * T * T + (long)t * T;
    float acc = 0.0f;
    for (int j = threadIdx.x; j <= t; j += blockDim.x) {
        float dbar = expf(fct - fc[g * T + j] + ig[g * T + j] - mt);
        float val = dbar * S[base + j];
        Dbar[base + j] = dbar;
        DS[base + j] = val;
        acc += val;
    }
    for (int j = t + 1 + threadIdx.x; j < T; j += blockDim.x) DS[base + j] = 0.0f;

    acc = block_reduce_sum(acc);
    if (threadIdx.x != 0) return;
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
// One block per row: a thread per row would put adjacent lanes W floats apart.
extern "C" __global__ void row_dot_add(float* out, const float* x, const float* y,
                                       int W, int R) {
    int r = blockIdx.x;
    if (r >= R) return;
    long base = (long)r * W;
    float acc = 0.0f;
    for (int w = threadIdx.x; w < W; w += blockDim.x) acc += x[base + w] * y[base + w];
    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0) out[r] += acc;
}

// out[g] += Σ_e x[g,e]·y[g,e] — the per-head reduction behind dg (E = dhv·dqk for
// the C state, dqk for n). One block per group; E is a whole state matrix, so a
// thread per group would walk it alone.
extern "C" __global__ void group_dot_add(float* out, const float* x, const float* y,
                                         int E, int G) {
    int g = blockIdx.x;
    if (g >= G) return;
    long base = (long)g * E;
    float acc = 0.0f;
    for (int e = threadIdx.x; e < E; e += blockDim.x) acc += x[base + e] * y[base + e];
    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0) out[g] += acc;
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

// Backward of ytil = num/ψ (with ψ = max(|qn|,1)). One BLOCK per row (g,t):
//   d_num = d_ytil/ψ ;  dψ = −(Σ_i d_ytil·num)/ψ² ;  d_qn = (|qn|>1? sign(qn):0)·dψ.
// A thread per row would stride the row length between adjacent lanes, so every
// access sits in its own sector; lanes must walk the contiguous i to coalesce.
extern "C" __global__ void div_rows_bwd(const float* d_ytil, const float* num, const float* psi,
                                        const float* qn, const float* m,
                                        float* d_num, float* d_qn,
                                        int dhv, int BHT) {
    int gt = blockIdx.x;
    if (gt >= BHT) return;
    float inv = 1.0f / psi[gt];
    long base = (long)gt * dhv;
    float red = 0.0f;
    for (int i = threadIdx.x; i < dhv; i += blockDim.x) {
        float dy = d_ytil[base + i];
        d_num[base + i] = dy * inv;
        red += dy * num[base + i];
    }
    red = block_reduce_sum(red);
    if (threadIdx.x != 0) return;
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
// One BLOCK per (g,r) over BH·T: both sums walk a full T, and a thread per (g,r)
// would give each lane its own sector on the row sum.
extern "C" __global__ void mlstm_dfc_dig(const float* P, float* dfc, float* dig, int T, int BHT) {
    int idx = blockIdx.x;
    if (idx >= BHT) return;
    int r = idx % T, g = idx / T;
    long gb = (long)g * T * T;
    float rowsum = 0.0f;
    for (int j = threadIdx.x; j <= r; j += blockDim.x) rowsum += P[gb + (long)r * T + j];
    float colsum = 0.0f;
    for (int t = r + threadIdx.x; t < T; t += blockDim.x) colsum += P[gb + (long)t * T + r];
    rowsum = block_reduce_sum(rowsum);
    colsum = block_reduce_sum(colsum);
    if (threadIdx.x != 0) return;
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

