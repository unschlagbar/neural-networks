//! mLSTM forward and backward at the shapes the model actually runs.
//!
//!   cargo run --release --features cuda --example mlstm_bw_bench
//!
//! Backward is not timed on its own — it needs a forward's cache — so each row is
//! measured twice: a forward-only loop and a forward+backward loop, with the
//! backward reported as the difference. The `bwd` column is what this benchmark
//! exists for.
//!
//! SM clock swings by more than 2x under load, so every row prints the clock it was
//! measured at and the whole table is repeated — compare rows across repeats, not a
//! single pair.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda` to run this benchmark");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;

    use neural_networks::config::{BACKBONE_CHUNK, OUT_HIDDEN, WORD_HIDDEN};
    use neural_networks::gpu::arena::TrainingCache;
    use neural_networks::gpu::{GTensor, Gpu, mlstm::MLstm};
    use neural_networks::tensor::Tensor;

    let mut cache = TrainingCache::new();

    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU: {e}");
            return;
        }
    };

    let clock = || -> String {
        std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=clocks.sm", "--format=csv,noheader,nounits"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "?".into())
    };

    // (B, T, heads, dqk). Row 0/1 are the backbone cell as the trainer builds it: one
    // sequence, BACKBONE_CHUNK long, WORD_HIDDEN wide. The rest are the encoder's and
    // decoder's cell — CHAR_HIDDEN wide at 16 heads, so dqk = 16, over a batch of
    // words whose length is the sequence axis.
    let default_shapes: Vec<(usize, usize, usize, usize)> = vec![
        (1, BACKBONE_CHUNK, 8, WORD_HIDDEN / 8),
        (1, 2 * BACKBONE_CHUNK, 8, WORD_HIDDEN / 8),
        (512, 4, 16, OUT_HIDDEN / 16),
        (256, 8, 16, OUT_HIDDEN / 16),
        (128, 16, 16, OUT_HIDDEN / 16),
    ];
    // `MLSTM_SHAPES=B,T,H,dqk[;B,T,H,dqk...]` replaces the table — the rows stay
    // interleaved inside a repeat, so two widths listed here are comparable.
    let shapes: Vec<(usize, usize, usize, usize)> = match std::env::var("MLSTM_SHAPES") {
        Ok(spec) => spec
            .split(';')
            .map(|row| {
                let n: Vec<usize> = row.split(',').map(|v| v.trim().parse().unwrap()).collect();
                (n[0], n[1], n[2], n[3])
            })
            .collect(),
        Err(_) => default_shapes,
    };
    let shapes: &[(usize, usize, usize, usize)] = &shapes;
    let only: Option<usize> = std::env::var("MLSTM_BENCH_ONLY")
        .ok()
        .and_then(|v| v.parse().ok());
    let shapes: Vec<(usize, usize, usize, usize)> = match only {
        Some(i) => vec![shapes[i]],
        None => shapes.to_vec(),
    };
    let env = |k: &str, d: usize| -> usize {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    // The SM clock ramps for the first few hundred iterations, so a short warmup
    // measures the ramp. `MLSTM_BENCH_WARMUP` raises it when a shape is being swept.
    let iters = env("MLSTM_BENCH_ITERS", 30);
    let repeats = env("MLSTM_BENCH_REPEATS", 3);
    let warmup = env("MLSTM_BENCH_WARMUP", 10);

    let mut cells: Vec<_> = shapes
        .iter()
        .map(|&(b, t, h, dqk)| {
            let d = h * dqk;
            let x = GTensor::from_host(&gpu, &Tensor::random(&[b, t, d], 0.5));
            let g = GTensor::from_host(&gpu, &Tensor::random(&[b, t, d], 1.0));
            let cell = MLstm::new_rand(&gpu, d, d, h, dqk);
            let y = GTensor::uninit(&gpu, &[b, t, d]);
            (x, g, cell, y)
        })
        .collect();

    for (x, g, cell, y) in cells.iter_mut() {
        for _ in 0..warmup {
            cell.forward(&gpu, x, y, &mut cache);
            let _ = cell.backward_alloc(&gpu, g);
        }
    }
    gpu.stream.synchronize().unwrap();

    println!(
        "{:>3} {:>5} {:>4} {:>4} {:>9} {:>9} {:>9}",
        "B", "T", "H", "dqk", "ms/fwd", "ms/bwd", "clocks"
    );
    for _ in 0..repeats {
        for (&(b, t, h, dqk), (x, g, cell, y)) in shapes.iter().zip(cells.iter_mut()) {
            gpu.stream.synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..iters {
                cell.forward(&gpu, x, y, &mut cache);
                cell.drop_saved();
            }
            gpu.stream.synchronize().unwrap();
            let fwd = t0.elapsed().as_secs_f64() / iters as f64 * 1e3;

            gpu.stream.synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..iters {
                cell.forward(&gpu, x, y, &mut cache);
                let _ = cell.backward_alloc(&gpu, g);
            }
            gpu.stream.synchronize().unwrap();
            let both = t0.elapsed().as_secs_f64() / iters as f64 * 1e3;
            println!(
                "{b:>3} {t:>5} {h:>4} {dqk:>4} {fwd:>9.4} {:>9.4} {:>9}",
                both - fwd,
                clock()
            );
        }
    }
}
