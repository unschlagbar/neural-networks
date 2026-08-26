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

    use neural_networks::config::{BACKBONE_CHUNK, CHAR_HIDDEN, WORD_HIDDEN};
    use neural_networks::gpu::arena::TrainingCache;
    use neural_networks::gpu::{GTensor, Gpu, slstm::SLstm};
    use neural_networks::tensor::Tensor;

    let env = |k: &str, d: usize| -> usize {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let (reps, iters) = (env("REPS", 5), env("ITERS", 20));

    let mut cache = TrainingCache::new();

    let gpu = Gpu::new().expect("gpu");
    println!(
        "SMs {}  smem_optin {}  best of {reps} x {iters}\n",
        gpu.sm_count, gpu.max_shared_optin
    );

    let named = |kind: &str, b: usize, t: usize, h: usize| {
        (format!("{kind:<9} B={b:<5}T={t:<5}H={h}"), b, t, h)
    };
    let default_shapes = vec![
        named("backbone", 1, BACKBONE_CHUNK, WORD_HIDDEN),
        named("encoder", 512, 4, CHAR_HIDDEN),
        named("encoder", 256, 8, CHAR_HIDDEN),
        named("decoder", 128, 16, CHAR_HIDDEN),
    ];
    // `SLSTM_SHAPES=B,T,H[;B,T,H...]` replaces the table, for sweeping a width the
    // config does not currently build.
    let shapes: Vec<(String, usize, usize, usize)> = match std::env::var("SLSTM_SHAPES") {
        Ok(spec) => spec
            .split(';')
            .map(|row| {
                let n: Vec<usize> = row.split(',').map(|v| v.trim().parse().unwrap()).collect();
                named("custom", n[0], n[1], n[2])
            })
            .collect(),
        Err(_) => default_shapes,
    };
    // ONLY=<index> restricts the run to one shape, so a profiler sees just that one.
    let n_shapes = shapes.len();
    let only = env("ONLY", n_shapes);

    for (i, (label, b, t, h)) in shapes.into_iter().enumerate() {
        if only < n_shapes && i != only {
            continue;
        }
        let mut cell = SLstm::new_rand(&gpu, h, h);
        let x = GTensor::from_host(&gpu, &Tensor::random(&[b, t, h], 0.5));
        let dy = GTensor::from_host(&gpu, &Tensor::random(&[b, t, h], 0.05));
        let mut y = GTensor::uninit(&gpu, &[b, t, h]);
        let mut dx = GTensor::uninit(&gpu, &[b, t, h]);
        let (mut best_wall, mut best_issue) = (f64::MAX, f64::MAX);

        for _ in 0..reps {
            for _ in 0..3 {
                cell.forward(&gpu, &x, &mut y, &mut cache);
                cell.backward(&gpu, &y, &dy, &mut dx);
            }
            gpu.stream.synchronize().unwrap();

            let mut issue = 0.0;
            let mut wall = 0.0;
            for _ in 0..iters {
                cell.forward(&gpu, &x, &mut y, &mut cache);
                gpu.stream.synchronize().unwrap();
                let t0 = Instant::now();
                cell.backward(&gpu, &y, &dy, &mut dx);
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
