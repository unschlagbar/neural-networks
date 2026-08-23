//! One mLSTM cell, same input and weights, N times — every gradient must come back
//! bit-identical.
//!
//!   cargo run --release --features cuda --example mlstm_determinism
//!
//! `determinism.rs` compares the whole model's LOSS, which is a forward quantity: a
//! nondeterministic backward leaves it alone and only shows up a step later. This
//! compares the gradients themselves, at the backbone's real shape and with the
//! chunked state carry the backbone sweep uses. `OFFLOAD=1` parks the saved
//! activations on the host between the forward and the backward, as the backbone
//! does, so the transfer's stream ordering is on trial too.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use neural_networks::gpu::arena::TrainingCache;
    use neural_networks::gpu::{GTensor, Gpu, mlstm::MLstm};
    use neural_networks::tensor::Tensor;

    let mut cache = TrainingCache::new();

    let gpu = Gpu::new().expect("gpu");
    let (b, t, heads, dqk) = (1usize, 512usize, 8usize, 96usize);
    let d = heads * dqk;
    let reps = 6;

    let x = GTensor::from_host(&gpu, &Tensor::random_seeded(&[b, t, d], 0.5, 0xA1));
    let g = GTensor::from_host(&gpu, &Tensor::random_seeded(&[b, t, d], 1.0, 0xA2));
    let mut cell = MLstm::new_rand(&gpu, d, d, heads, dqk);
    if std::env::var("OFFLOAD").is_ok() {
        cell.enable_offload(&gpu, neural_networks::gpu::offload::InFlight::shared());
        println!("(activation offload enabled)");
    }
    // Positional labels for `grads()`, so a mismatch names something.
    const GRAD_NAMES: [&str; 16] = [
        "dWq", "dbq", "dWk", "dbk", "dWv", "dbv", "dWo", "dbo", "dWi", "dbi", "dWf",
        "dbf", "dWout", "dbout", "dgamma", "d?",
    ];

    // Sum of bits, not of values: a reordered fp32 sum would hide behind a float
    // total, and what is on trial is bit-identity.
    let hash = |v: &[f32]| -> u64 {
        v.iter().fold(0xcbf29ce484222325u64, |h, x| {
            (h ^ x.to_bits() as u64).wrapping_mul(0x100000001b3)
        })
    };

    let mut first: Option<Vec<(&str, u64)>> = None;
    let mut bad = false;
    for r in 0..reps {
        cell.zero_grad(&gpu);
        let mut y = GTensor::uninit(&gpu, &[b, t, d]);
        cell.forward(&gpu, &x, &mut y, &mut cache);
        let dx = cell.backward_alloc(&gpu, &g);
        let mut sig = vec![
            ("y", hash(&y.to_host(&gpu).data)),
            ("dx", hash(&dx.to_host(&gpu).data)),
        ];
        // `grads()` is the cell's own order: the seven projections' dW/db, then the
        // head norm's dγ.
        for (i, gr) in cell.grads().iter().enumerate() {
            sig.push((GRAD_NAMES[i.min(GRAD_NAMES.len() - 1)], hash(&gr.to_host(&gpu).data)));
        }
        match &first {
            None => {
                for (n, h) in &sig {
                    println!("  {n:>6}: {h:#018x}");
                }
                first = Some(sig);
            }
            Some(f) => {
                for ((n, h0), (_, h)) in f.iter().zip(&sig) {
                    if h0 != h {
                        println!("  rep {r}: {n} DIFFERS {h0:#018x} != {h:#018x}");
                        bad = true;
                    }
                }
            }
        }
    }
    println!("\n{}", if bad { "NONDETERMINISTIC" } else { "DETERMINISTIC" });
    if bad {
        std::process::exit(1);
    }
}
