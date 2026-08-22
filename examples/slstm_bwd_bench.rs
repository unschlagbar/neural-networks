//! sLSTM BACKWARD only, at the four shapes the hierarchical model actually runs it
//! at: one backbone chunk (B=1, long T, the time-fused kernel) and three
//! encoder/decoder length groups (wide B, a word's worth of T, the per-step loop).
//!
//! The forward is re-run before each timed backward (a backward consumes the saved
//! activations), but only the backward region is timed.
//!
//! Reports the BEST of `REPS` timed rounds: the SM clock on this card swings between
//! ~1.4 and ~2.9 GHz, which is wider than most wins being measured.
//!
//!   cargo run --release --features cuda --example slstm_bwd_bench

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda`");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;

    use neural_networks::gpu::{DTensor, Gpu, slstm::SLstm};
    use neural_networks::tensor::Tensor;

    let env = |k: &str, d: usize| -> usize {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let (reps, iters) = (env("REPS", 5), env("ITERS", 20));

    let gpu = Gpu::new().expect("gpu");
    println!(
        "SMs {}  smem_optin {}  best of {reps} x {iters}\n",
        gpu.sm_count, gpu.max_shared_optin
    );

    let shapes = [
        ("backbone  B=1   T=512 H=768", 1, 512, 768),
        ("encoder   B=512 T=4   H=256", 512, 4, 256),
        ("encoder   B=256 T=8   H=256", 256, 8, 256),
        ("decoder   B=128 T=16  H=256", 128, 16, 256),
    ];
    // ONLY=<index> restricts the run to one shape, so a profiler sees just that one.
    let only = env("ONLY", shapes.len());

    for (i, (label, b, t, h)) in shapes.into_iter().enumerate() {
        if only < shapes.len() && i != only {
            continue;
        }
        let mut cell = SLstm::new_rand(&gpu, h, h);
        let x = DTensor::from_host(&gpu, &Tensor::random(&[b, t, h], 0.5));
        let dy = DTensor::from_host(&gpu, &Tensor::random(&[b, t, h], 0.05));
        let mut y = DTensor::uninit(&gpu, &[b, t, h]);
        let mut dx = DTensor::uninit(&gpu, &[b, t, h]);
        let (mut best_wall, mut best_issue) = (f64::MAX, f64::MAX);

        for _ in 0..reps {
            for _ in 0..3 {
                cell.forward(&gpu, &x, &mut y);
                cell.backward(&gpu, &dy, &mut dx);
            }
            gpu.stream.synchronize().unwrap();

            let mut issue = 0.0;
            let mut wall = 0.0;
            for _ in 0..iters {
                cell.forward(&gpu, &x, &mut y);
                gpu.stream.synchronize().unwrap();
                let t0 = Instant::now();
                cell.backward(&gpu, &dy, &mut dx);
                issue += t0.elapsed().as_secs_f64();
                gpu.stream.synchronize().unwrap();
                wall += t0.elapsed().as_secs_f64();
            }
            best_wall = best_wall.min(wall / iters as f64);
            best_issue = best_issue.min(issue / iters as f64);
        }
        println!(
            "{label}   wall {:>8.2} us   {:>6.3} us/step   issue {:>7.2} us ({:.0}% host)",
            best_wall * 1e6,
            best_wall * 1e6 / t as f64,
            best_issue * 1e6,
            best_issue / best_wall * 100.0
        );
    }
}
