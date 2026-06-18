use crate::{
    config::{WAKE_HIDDEN, WAKE_INPUT_DIM, WAKE_MODEL_LOC},
    nn_layer::SequentialBuilder,
    sequential::Sequential,
};

/// SiLU-Dense(INPUT_DIM→H) → CausalConv1d(k=7) → RMSNorm → CausalConv1d(k=5) → RMSNorm
/// → sLSTM(H) → RMSNorm → sLSTM(H) → RMSNorm → Linear(H→1)
pub fn build_wake_model() -> Sequential {
    SequentialBuilder::new(WAKE_INPUT_DIM)
        .silu_dense(128)
        .silu_dense(128)
        .slstm(WAKE_HIDDEN)
        .rms_norm()
        .slstm(WAKE_HIDDEN)
        .rms_norm()
        .linear_no_bias(1)
        .build()
}

pub fn load_or_build_wake_model() -> Sequential {
    match Sequential::load(WAKE_MODEL_LOC) {
        Ok(m) => {
            println!("loaded wake-word model from {WAKE_MODEL_LOC}");
            m
        }
        Err(_) => {
            println!("no model found — building fresh wake-word model");
            build_wake_model()
        }
    }
}
