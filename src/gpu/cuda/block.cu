// Residual block / SwiGLU (nn2/block.rs): elementwise add, SwiGLU forward,
// in-place scale and sigmoid.

// Elementwise add: out = a + b. Used for the two residual adds and the grad
// accumulations that are plain sums (d_zn, d_z, dx).
extern "C" __global__ void add(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// In-place accumulate: acc += b. The two-operand `add` needs a distinct output,
// so a running sum through it costs a fresh buffer per term (and, where the old
// buffer was rebound, dropped the previous one). This writes back into `acc`.
extern "C" __global__ void add_assign(float* acc, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += b[i];
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
