//! Which of the sLSTM's two time loops is actually faster, and where does the
//! crossover sit?
//!
//! The cell has a per-step path (`fwd_steps`/`bwd_steps`, one launch per timestep)
//! and a time-fused cooperative path (`slstm_fused_time`/`_bwd`, the whole T-loop in
//! one launch), selected at `T >= FUSED_MIN_T`. That threshold is a guess that was
//! never re-checked after the backward was rewritten warp-per-unit, the slabs went
//! bf16, and the block width was refitted. This measures it.
//!
//! Two traps this harness is built around:
//!
//!   * The fused path DECLINES silently when the shape does not fit its shared-memory
//!     or register budget, falling back to per-step. Forcing it on then measures the
//!     same code twice and reports a meaningless 1.00x. Both geometry helpers are
//!     consulted up front and the verdict is printed per shape.
//!   * SM clock swings 1372-2880 MHz on this card, which is larger than most of the
//!     effects being measured. The arms are therefore INTERLEAVED within a round and
//!     scored on the MINIMUM over rounds, not the mean.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda` to run this benchmark");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;

    use neural_networks::gpu::arena::TrainingCache;
    use neural_networks::gpu::{GTensor, Gpu, ops, slstm::SLstm};
    use neural_networks::tensor::Tensor;

    let gpu = Gpu::new().expect("gpu");
    let rounds: usize = env_usize("ROUNDS", 9);
    let iters: usize = env_usize("ITERS", 6);

    // (label, B, T, H). The backbone sweep walks T across FUSED_MIN_T at the width and
    // batch the backbone really runs (B=1, H=1024, chunk 512). The batched sweep is the
    // opposite corner: the encoder's shape, but at T values high enough that the fused
    // path is even eligible.
    let shapes: &[(&str, usize, usize, usize)] = &[
        // B sweep at fixed T, H: the recurrent product `h.Whr` goes from a mat-VEC
        // (M=1, where cuBLAS cannot fill a tensor-core tile) to a real GEMM. If the
        // fused path loses because it does that product as scalar FMA, the crossover
        // sits where the tiles start filling, not anywhere related to T.
        ("Bsweep    B=1   T=64   H=256", 1, 64, 256),
        ("Bsweep    B=2   T=64   H=256", 2, 64, 256),
        ("Bsweep    B=4   T=64   H=256", 4, 64, 256),
        ("Bsweep    B=8   T=64   H=256", 8, 64, 256),
        ("Bsweep    B=16  T=64   H=256", 16, 64, 256),
        ("Bsweep    B=32  T=64   H=256", 32, 64, 256),
        ("backbone  B=1   T=16   H=1024", 1, 16, 1024),
        ("backbone  B=1   T=32   H=1024", 1, 32, 1024),
        ("backbone  B=1   T=64   H=1024", 1, 64, 1024),
        ("backbone  B=1   T=128  H=1024", 1, 128, 1024),
        ("backbone  B=1   T=512  H=1024", 1, 512, 1024),
        ("batched   B=8   T=512  H=1024", 8, 512, 1024),
        ("backbone  B=1   T=512  H=768", 1, 512, 768),
        ("batched   B=64  T=64   H=256", 64, 64, 256),
        ("batched   B=256 T=64   H=256", 256, 64, 256),
        ("batched   B=256 T=32   H=256", 256, 32, 256),
    ];

    // Optional CLI override: `slstm_path_ab B T H` measures one shape.
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let own;
    let shapes: &[(&str, usize, usize, usize)] = if argv.len() == 3 {
        let p = |i: usize| argv[i].parse::<usize>().expect("B T H");
        own = vec![("cli", p(0), p(1), p(2))];
        &own
    } else {
        shapes
    };

    println!("rounds {rounds}  iters/sample {iters}   (min over rounds)");
    println!("{}", clocks());
    println!(
        "\n{:<30} {:>10} {:>10} {:>10} {:>8}  {}",
        "shape", "fused ms", "batched ms", "steps ms", "ratio", "fused geometry"
    );

    for &(label, b, t, h) in shapes {
        let s = 1.0 / (h as f32).sqrt();
        let w: Vec<Tensor> = (0..4)
            .map(|g| Tensor::random_seeded(&[2 * h, h], s * (1.0 + g as f32 * 0.05), 100 + g as u64))
            .collect();
        let bi: Vec<Tensor> = (0..4)
            .map(|g| Tensor::random_seeded(&[h], 0.2 + g as f32 * 0.01, 200 + g as u64))
            .collect();
        let mut cell = SLstm::from_parts(
            &gpu, h, h, &w[0], &w[1], &w[2], &w[3], &bi[0], &bi[1], &bi[2], &bi[3], None,
        );

        // Does the fused path actually accept this shape? Forward and backward decide
        // separately, so a shape can fuse one half and fall back on the other.
        let gf = ops::slstm_fused_time_geometry(&gpu, h, b).is_some();
        let gb = ops::slstm_fused_time_bwd_geometry(&gpu, h, b).is_some();
        let verdict = match (t >= 32, gf, gb) {
            (false, _, _) => "declined: T < FUSED_MIN_T".to_string(),
            (true, true, true) => "fwd+bwd".to_string(),
            (true, true, false) => "fwd only (bwd declined)".to_string(),
            (true, false, true) => "bwd only (fwd declined)".to_string(),
            (true, false, false) => "declined: both".to_string(),
        };

        let x = GTensor::from_host(&gpu, &Tensor::random_seeded(&[b, t, h], 0.5, 7));
        let gy = GTensor::from_host(&gpu, &Tensor::random_seeded(&[b, t, h], 0.7, 9));
        let mut y: GTensor<f32> = GTensor::uninit(&gpu, &[b, t, h]);
        let mut dx: GTensor<f32> = GTensor::uninit(&gpu, &[b, t, h]);
        let mut cache = TrainingCache::new();

        // Three arms: the scalar time-fused kernel, the mma batched one, and the
        // per-step loop both replace.
        let mut run = |cell: &mut SLstm, arm: usize, n: usize| -> f64 {
            cell.force_fused_time = Some(arm == 0);
            cell.force_batched = Some(arm == 1);
            gpu.stream.synchronize().ok();
            let t0 = Instant::now();
            for _ in 0..n {
                cell.forward(&gpu, &x, &mut y, &mut cache);
                cell.backward(&gpu, &y, &gy, &mut dx);
            }
            gpu.stream.synchronize().ok();
            t0.elapsed().as_secs_f64() * 1e3 / n as f64
        };

        // Warm every arm: NVRTC compile, allocator, weight cache, clocks.
        for arm in 0..3 {
            run(&mut cell, arm, 3);
        }

        let mut best = [f64::MAX; 3];
        for _ in 0..rounds {
            for arm in 0..3 {
                best[arm] = best[arm].min(run(&mut cell, arm, iters));
            }
        }
        println!(
            "{label:<30} {:>10.3} {:>10.3} {:>10.3} {:>7.2}x  {verdict}",
            best[0], best[1], best[2],
            best[2] / best[0].min(best[1])
        );
    }
    println!("\n{}", clocks());
}

#[cfg(feature = "cuda")]
fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// SM and memory clock, so a suspicious run can be checked against thermal drift.
#[cfg(feature = "cuda")]
fn clocks() -> String {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=clocks.sm,clocks.mem,temperature.gpu", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| format!("clocks (sm, mem, temp): {}", s.trim()))
        .unwrap_or_default()
}
