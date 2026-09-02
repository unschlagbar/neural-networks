// Gated Recurrent Transformer (arXiv 2608.15062) device kernels: LayerNorm, the
// stateless Gaussian noise, the elementwise depth gate and the SiLU the gate MLP
// uses. Everything here is position-wise over `[rows, n]` — the recurrence these
// serve runs over DEPTH, not over the row axis, so no kernel below carries state.

// LayerNorm — mean-subtracting, weight-only (the reference runs `bias=False`).
//
// Block per row, exactly like the RMSNorm pair above and for the same reason: a
// thread-per-row norm walks the whole width serially and leaves the machine idle.
// Two reductions per row (Sigma x, then Sigma (x-mu)^2) rather than one pass over
// x and x^2: the widths here are 512-2048, the row is in L1 after the first pass,
// and the subtracted form does not lose the variance to cancellation.
//
// Saves `inv_std` and nothing else. `x_hat` is recovered in backward as `y/gamma`,
// the same trick `rms_norm_backward` uses.
extern "C" __global__ void layer_norm_forward(const float* x, const float* gamma,
                                              float* out, float* inv_std,
                                              int n, float eps, int rows) {
    int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float sh[];
    long long off = (long long)row * n;

    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += x[off + i];
    s = rmsn_block_sum(s, sh);
    float mean = s / (float)n;
    __syncthreads();

    float q = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float d = x[off + i] - mean;
        q += d * d;
    }
    q = rmsn_block_sum(q, sh);
    float inv = rsqrtf(q / (float)n + eps);
    if (threadIdx.x == 0) inv_std[row] = inv;
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[off + i] = gamma[i] * (x[off + i] - mean) * inv;
}

// Backward twin — `dx` only; `dgamma` is a column sum the caller runs through
// `add_col_sum_mul_div` (deterministic, unlike an atomic across these blocks).
//
//   u      = gamma * dy                    (dL/dx_hat)
//   dx     = inv_std * (u - mean(u) - x_hat * mean(u * x_hat))
//
// with `x_hat = y / gamma`. gamma is undecayed and starts at 1, so the divisor
// stays away from zero — the standing caveat this shares with `rms_norm_backward`.
extern "C" __global__ void layer_norm_backward(const float* dy, const float* y,
                                               const float* inv_std, const float* gamma,
                                               float* dx, int n, int rows) {
    int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float sh[];
    long long off = (long long)row * n;
    float inv = inv_std[row];

    float su = 0.0f, sux = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float g = gamma[i];
        float u = g * dy[off + i];
        su += u;
        sux += u * (y[off + i] / g);
    }
    su = rmsn_block_sum(su, sh);
    float mean_u = su / (float)n;
    __syncthreads();
    sux = rmsn_block_sum(sux, sh);
    float mean_ux = sux / (float)n;
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float g = gamma[i];
        float u = g * dy[off + i];
        float xh = y[off + i] / g;
        dx[off + i] = inv * (u - mean_u - xh * mean_ux);
    }
}

// Counter-based standard normal.
//
// Stateless on purpose: the backward re-derives the same eps_x from the same key
// instead of the forward storing it, which at R recurrences over a chunked sweep is
// the difference between keeping one [rows, d] tensor per step and keeping none.
// A stateful RNG could not do that — the draw has to be a pure function of
// (seed, index).
__device__ __forceinline__ unsigned grt_mix(unsigned x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ float grt_normal(unsigned seed_hi, unsigned seed_lo, unsigned idx) {
    unsigned a = grt_mix(idx ^ seed_lo ^ 0x9e3779b9u);
    unsigned b = grt_mix(a ^ seed_hi ^ 0x85ebca6bu);
    // u1 in (0, 1] so the log never sees zero; u2 in [0, 1).
    float u1 = (float)((a >> 8) + 1u) * (1.0f / 16777216.0f);
    float u2 = (float)(b >> 8) * (1.0f / 16777216.0f);
    return sqrtf(-2.0f * logf(u1)) * cospif(2.0f * u2);
}

// `out = x + sigma * eps`, eps ~ N(0, 1) drawn from (seed, idx0 + i). Eq. (2)'s
// state noise.
//
// `idx0` is the element index of this tensor's first element in the WINDOW, not in
// the launch. The backbone sweeps the word axis in chunks, and keying the draw on
// the position within a chunk would make the noise a function of BACKBONE_CHUNK —
// a memory knob would change what the model trains on, and the chunk-invariance the
// backbone is pinned against would stop holding.
extern "C" __global__ void grt_noise_add(float* out, const float* x, float sigma,
                                         unsigned seed_hi, unsigned seed_lo,
                                         unsigned idx0, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = x[i] + sigma * grt_normal(seed_hi, seed_lo, idx0 + (unsigned)i);
}

// Feature-axis concatenation `out[r, :] = [a[r, :] | b[r, :]]`, and the split that
// undoes it on the gradient. Both halves are the same width here (d), but the
// kernels take them separately so a caller never has to assume that.
extern "C" __global__ void grt_cat2(float* out, const float* a, const float* b,
                                    int na, int nb, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * (na + nb);
    if (i >= total) return;
    int w = na + nb;
    int row = i / w, col = i - row * w;
    out[i] = (col < na) ? a[(long long)row * na + col] : b[(long long)row * nb + (col - na)];
}

// `da`/`db` are OVERWRITTEN. Every caller here owns freshly presented buffers.
extern "C" __global__ void grt_split2(const float* d, float* da, float* db,
                                      int na, int nb, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * (na + nb);
    if (i >= total) return;
    int w = na + nb;
    int row = i / w, col = i - row * w;
    if (col < na) da[(long long)row * na + col] = d[i];
    else db[(long long)row * nb + (col - na)] = d[i];
}

// SiLU, the gate MLP's activation. `y = z * sigmoid(z)`.
extern "C" __global__ void grt_silu_forward(float* y, const float* z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = z[i];
    y[i] = v / (1.0f + expf(-v));
}

// `dz = dy * (s + z * s * (1 - s))` with `s = sigmoid(z)`.
extern "C" __global__ void grt_silu_backward(float* dz, const float* dy, const float* z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = z[i];
    float s = 1.0f / (1.0f + expf(-v));
    dz[i] = dy[i] * (s + v * s * (1.0f - s));
}

// Eq. (5): `g = sigmoid(z2 / tau + eps_g)`.
//
// `eps_g` is ONE draw per row, broadcast across the feature axis — the reference
// samples `randn_like(gate_features[..., :1])`, a per-token scalar, not a per-element
// tensor. Writing it per element would be a different (and much stronger) regulariser.
extern "C" __global__ void grt_gate_apply(float* g, const float* z2, float inv_tau,
                                          float sigma, unsigned seed_hi, unsigned seed_lo,
                                          unsigned row0, int n, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * n;
    if (i >= total) return;
    int row = i / n;
    // `row0`: the window-global row, for the reason `grt_noise_add` gives.
    float noise = (sigma > 0.0f)
                      ? sigma * grt_normal(seed_hi, seed_lo, row0 + (unsigned)row)
                      : 0.0f;
    g[i] = 1.0f / (1.0f + expf(-(z2[i] * inv_tau + noise)));
}

// `dz2 = dg * g * (1 - g) / tau`. The noise is additive, so it drops out.
extern "C" __global__ void grt_gate_bwd(float* dz2, const float* dg, const float* g,
                                        float inv_tau, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = g[i];
    dz2[i] = dg[i] * gv * (1.0f - gv) * inv_tau;
}

// Eq. (4): `h_out = g * h_prev + (1 - g) * o`.
extern "C" __global__ void grt_blend(float* h_out, const float* g, const float* h_prev,
                                     const float* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = g[i];
    h_out[i] = gv * h_prev[i] + (1.0f - gv) * o[i];
}

// Backward of Eq. (4), all three branches in one pass over `dh`:
//   d_o      = (1 - g) * dh          (overwritten — the core's incoming gradient)
//   d_g      = (h_prev - o) * dh     (overwritten)
//   dh_prev += g * dh                (ACCUMULATED: h_prev is also read by the gate
//                                     MLP and by W_proj, whose taps land here too)
extern "C" __global__ void grt_blend_bwd(const float* dh, const float* g, const float* h_prev,
                                         const float* o, float* dh_prev, float* d_o,
                                         float* d_g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = g[i], d = dh[i];
    d_o[i] = (1.0f - gv) * d;
    d_g[i] = (h_prev[i] - o[i]) * d;
    dh_prev[i] += gv * d;
}

// Untied LSTM-style depth gate — the alternative to Eq. (4)'s tied convex blend:
//
//   h_out = f * h_prev + i * o
//
// `fi` is `[rows, 2d]`, the forget half first. The tied form ties the write magnitude
// to the copy magnitude (`i == 1 - f`), so keeping the state and letting the core
// contribute are the same knob; untying them is what an LSTM cell does over time, and
// what our own sLSTM/mLSTM cells already do.
extern "C" __global__ void grt_blend_lstm(float* h_out, const float* fi,
                                          const float* h_prev, const float* o,
                                          int d, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int row = idx / d, c = idx - row * d;
    long long fo = (long long)row * 2 * d + c;
    h_out[idx] = fi[fo] * h_prev[idx] + fi[fo + d] * o[idx];
}

// Backward of the untied blend. `d_o` and `d_fi` are overwritten, `dh_prev` is
// ACCUMULATED into — the gate MLP and W_proj tap `h_prev` as well.
extern "C" __global__ void grt_blend_lstm_bwd(const float* dh, const float* fi,
                                              const float* h_prev, const float* o,
                                              float* dh_prev, float* d_o, float* d_fi,
                                              int d, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int row = idx / d, c = idx - row * d;
    long long fo = (long long)row * 2 * d + c;
    float g = dh[idx];
    d_fi[fo] = h_prev[idx] * g;
    d_fi[fo + d] = o[idx] * g;
    d_o[idx] = fi[fo + d] * g;
    dh_prev[idx] += fi[fo] * g;
}
