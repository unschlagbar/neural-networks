//! `fused_tv` / `FUSED_THREADS_REC` across the shapes the model actually runs,
//! not just the backbone's.
//!
//!   cargo run --release --features cuda --example mlstm_tv_shapes
//!
//! `mlstm_bw_dC` gets its grid from `(ceil(dhv/tv), B*H)`. Tuning it on one shape is
//! misleading: at B=1 the batch axis contributes nothing and the
//! `v` split is the only source of blocks, while at the encoder/decoder's batched
//! shapes `B*H` alone already oversubscribes the machine and splitting `v` only
//! shrinks the per-block accumulation below one element per thread.
//!
//! Sweeps both knobs over both regimes so the defaults can be chosen against the
//! whole model rather than one cell.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda` to run this benchmark");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;

    use neural_networks::gpu::{DTensor, Gpu, mlstm::MLstm};
    use neural_networks::tensor::Tensor;

    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU: {e}");
            return;
        }
    };

    // (label, B, T, heads, dqk) — the cells this model builds.
    let shapes = [
        ("backbone dqk96", 1, 4096, 8, 96),
        ("backbone dqk64", 1, 4096, 8, 64),
        ("enc/dec B=64", 64, 16, 16, 16),
        ("enc/dec B=256", 256, 16, 16, 16),
    ];

    let tv = std::env::var("MLSTM_TV").unwrap_or_else(|_| "default".into());
    let thr = std::env::var("MLSTM_THREADS_REC").unwrap_or_else(|_| "default".into());
    println!("MLSTM_TV={tv}  MLSTM_THREADS_REC={thr}");
    println!("{:>14} {:>5} {:>6} {:>6} {:>5} {:>12}", "shape", "B", "T", "heads", "dqk", "ms/iter");

    for (label, b, t, heads, dqk) in shapes {
        let d = dqk * heads;
        let mut dev = MLstm::new_rand(&gpu, d, d, heads, dqk);
        let x = DTensor::from_host(&gpu, &Tensor::random(&[b, t, d], 0.5));
        let g = DTensor::from_host(&gpu, &Tensor::random(&[b, t, d], 1.0));
        let mut y = DTensor::uninit(&gpu, &[b, t, d]);

        let iters = if t > 1000 { 10 } else { 50 };
        for _ in 0..5 {
            dev.forward(&gpu, &x, &mut y);
            let _ = dev.backward_alloc(&gpu, &g);
        }
        gpu.stream.synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            dev.forward(&gpu, &x, &mut y);
            let _ = dev.backward_alloc(&gpu, &g);
        }
        gpu.stream.synchronize().unwrap();
        let ms = t0.elapsed().as_secs_f64() / iters as f64 * 1e3;
        println!("{label:>14} {b:>5} {t:>6} {heads:>6} {dqk:>5} {ms:>12.3}");
    }
}
