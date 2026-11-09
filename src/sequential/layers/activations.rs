// Shared activation functions for the xLSTM family (mLSTM / sLSTM and their
// blocks). Kept in one place so the numerically-stable variants are defined
// exactly once instead of being copy-pasted into every cell and block.

/// Numerically stable logistic sigmoid: avoids `exp` overflow for large |x|.
#[inline]
pub fn stable_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Stable `log σ(x) = −softplus(−x)`, using `ln_1p` for accuracy near 0.
#[inline]
pub fn log_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        -(-x).exp().ln_1p()
    } else {
        x - x.exp().ln_1p()
    }
}

/// SiLU / swish activation: `x · σ(x)`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x * stable_sigmoid(x)
}

/// Derivative of SiLU w.r.t. its pre-activation input.
#[inline]
pub fn silu_prime(pre: f32) -> f32 {
    let s = stable_sigmoid(pre);
    s * (1.0 + pre * (1.0 - s))
}
