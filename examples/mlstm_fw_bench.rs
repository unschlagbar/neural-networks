//! mLSTM forward at the shapes the model actually runs, timed on its own.
//!
//!   cargo run --release --features cuda --example mlstm_fw_bench
//!
//! The backbone is the only place an mLSTM cell appears (B=1, T=BACKBONE_CHUNK,
//! WORD_HIDDEN wide, 8 heads), so that row is the one that decides a step. The
//! others bracket it.
//!
//! SM clock swings by more than 2x under load, so every row prints the clock it
//! was measured at and the whole table is repeated — compare rows across repeats,
//! not a single pair.

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

    let clock = || -> String {
        std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=clocks.sm", "--format=csv,noheader,nounits"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "?".into())
    };

    // (B, T, heads, dqk) — row 1 is the backbone cell as the trainer builds it.
    let shapes: &[(usize, usize, usize, usize)] = &[
        (1, 512, 8, 96),
        (1, 1024, 8, 96),
        (1, 512, 8, 64),
        (4, 512, 8, 96),
    ];
    // `MLSTM_BENCH_ONLY=<row>` narrows the table to one shape, which is what a
    // profiler run wants — otherwise every kernel statistic mixes all four.
    let only: Option<usize> = std::env::var("MLSTM_BENCH_ONLY").ok().and_then(|v| v.parse().ok());
    let shapes: Vec<(usize, usize, usize, usize)> = match only {
        Some(i) => vec![shapes[i]],
        None => shapes.to_vec(),
    };
    let iters = 50;
    let repeats = 3;

    let mut cells: Vec<_> = shapes
        .iter()
        .map(|&(b, t, h, dqk)| {
            let d = h * dqk;
            let x = DTensor::from_host(&gpu, &Tensor::random(&[b, t, d], 0.5));
            let cell = MLstm::new_rand(&gpu, d, d, h, dqk);
            let y = DTensor::uninit(&gpu, &[b, t, d]);
            (x, cell, y)
        })
        .collect();

    for (i, (x, cell, y)) in cells.iter_mut().enumerate() {
        let _ = i;
        for _ in 0..10 {
            cell.forward(&gpu, x, y);
            cell.drop_saved();
        }
    }
    gpu.stream.synchronize().unwrap();

    println!("{:>3} {:>5} {:>4} {:>4} {:>10} {:>9}", "B", "T", "H", "dqk", "ms/fwd", "clocks");
    for _ in 0..repeats {
        for (&(b, t, h, dqk), (x, cell, y)) in shapes.iter().zip(cells.iter_mut()) {
            gpu.stream.synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..iters {
                cell.forward(&gpu, x, y);
                cell.drop_saved();
            }
            gpu.stream.synchronize().unwrap();
            let ms = t0.elapsed().as_secs_f64() / iters as f64 * 1e3;
            println!("{b:>3} {t:>5} {h:>4} {dqk:>4} {ms:>10.4} {:>9}", clock());
        }
    }
}
