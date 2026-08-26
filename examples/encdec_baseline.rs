//! Baseline cost of the encoder/decoder sLSTM sweeps, at the REAL length buckets.
//!
//! `gpu_prof` runs one synthetic group (B=2047, T=8), which hides the thing that
//! actually costs: a window is ~20 SEPARATE groups, each paying its own per-timestep
//! launches. `_grp_probe` says a 4096-word window over real source splits like the
//! table below -- B*T roughly constant, and the LONG words carry most of the timesteps
//! while having the SMALLEST batch.
//!
//! This is the number `slstm_fused_time_batched` has to beat
//! (see docs/slstm-batched-fused-plan.md).

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda`");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;

    use neural_networks::config::CHAR_HIDDEN;
    use neural_networks::gpu::arena::TrainingCache;
    use neural_networks::gpu::{GTensor, Gpu, slstm::SLstm};
    use neural_networks::tensor::Tensor;

    let gpu = Gpu::new().expect("gpu");
    let h = CHAR_HIDDEN;
    let rounds: usize = std::env::var("ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(7);
    // `ARM=step` or `ARM=bat` runs one arm only, so a profiler sees just that one.
    let arms: &[bool] = match std::env::var("ARM").as_deref() {
        Ok("step") => &[false],
        Ok("bat") => &[true],
        _ => &[false, true],
    };

    // (T, B, pieces) -- measured by `examples/_grp_probe.rs` on src/gpu/ops.rs. A
    // corpus with a different word-length distribution splits differently, and the
    // batch is what decides both the launch geometry and whether the batched path is
    // taken at all -- so `GROUPS=T:B:pieces,...` takes a real window's buckets instead.
    let own: Vec<(usize, usize, usize)>;
    let default_groups: &[(usize, usize, usize)] = &[
        (1, 1024, 2), (2, 682, 2), (3, 512, 1), (4, 409, 2),
        (5, 341, 1), (6, 292, 2), (7, 256, 1), (8, 227, 1),
        (9, 204, 1), (10, 186, 1), (11, 170, 1), (12, 157, 1),
        (13, 146, 1), (14, 136, 1), (15, 128, 1), (16, 120, 1),
    ];

    let groups: &[(usize, usize, usize)] = match std::env::var("GROUPS") {
        Ok(spec) => {
            own = spec
                .split(',')
                .map(|g| {
                    let f: Vec<usize> = g
                        .split(':')
                        .map(|v| v.trim().parse().expect("GROUPS=T:B:pieces,..."))
                        .collect();
                    assert_eq!(f.len(), 3, "GROUPS=T:B:pieces,...");
                    (f[0], f[1], f[2])
                })
                .collect();
            &own
        }
        Err(_) => default_groups,
    };

    let mut cell = SLstm::new_rand(&gpu, h, h);
    let mut cache = TrainingCache::new();
    println!("H={h}, {} groups, min of {rounds} rounds\n", groups.iter().map(|g| g.2).sum::<usize>());
    println!("{:>3} {:>6} {:>6} {:>9} {:>9} {:>9} {:>9} {:>8}",
             "T", "B", "pieces", "fwd step", "fwd bat", "f+b step", "f+b bat", "speedup");

    let mut tot = [[0.0f64; 2]; 3];
    let (mut tot_steps, mut declined) = (0usize, 0usize);
    for &(t, b, pieces) in groups {
        let x = GTensor::from_host(&gpu, &Tensor::random_seeded(&[b, t, h], 0.5, 7));
        let gy = GTensor::from_host(&gpu, &Tensor::random_seeded(&[b, t, h], 0.7, 9));
        let mut y: GTensor<f32> = GTensor::uninit(&gpu, &[b, t, h]);
        let mut dx: GTensor<f32> = GTensor::uninit(&gpu, &[b, t, h]);
        assert!(!cell.fuses_at(b, t), "this shape should be on the per-step path");
        // Printed rather than asserted: the batched path declines a batch too wide for
        // its operand re-reads, and those groups are exactly what the table compares.
        // Which arm the cell would pick on its own. The `bat` column below forces the
        // batched kernels either way, so the table shows the losing arm too.
        cell.force_batched = None;
        let picks_batched = cell.batches_at(&gpu, b);

        let mut run = |cell: &mut SLstm, batched: bool, bwd: bool, n: usize| -> f64 {
            cell.force_batched = Some(batched);
            gpu.stream.synchronize().ok();
            let t0 = Instant::now();
            for _ in 0..n {
                cell.forward(&gpu, &x, &mut y, &mut cache);
                if bwd {
                    cell.backward(&gpu, &y, &gy, &mut dx);
                }
            }
            gpu.stream.synchronize().ok();
            t0.elapsed().as_secs_f64() * 1e3 / n as f64
        };
        for arm in arms.iter() {
            run(&mut cell, *arm, true, 3);
        }

        // The SM clock swings wider than most of what is being measured here, so the
        // arms are interleaved within a round and scored on the MINIMUM over rounds.
        let mut best = [[f64::MAX; 2]; 2]; // [batched][with backward]
        for _ in 0..rounds {
            for &arm in arms.iter() {
                for bwd in 0..2 {
                    let i = usize::from(arm);
                    best[i][bwd] = best[i][bwd].min(run(&mut cell, arm, bwd == 1, 5));
                }
            }
        }
        // One group of this length runs `pieces` times per window.
        let p = pieces as f64;
        for arm in 0..2 {
            tot[arm][0] += best[arm][0] * p;
            tot[arm][1] += best[arm][1] * p;
        }
        tot_steps += t * pieces;
        println!(
            "{t:>3} {b:>6} {pieces:>6} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>7.2}x",
            best[0][0] * p, best[1][0] * p, best[0][1] * p, best[1][1] * p,
            best[0][1] / best[1][1]
        );
        tot[2][0] += best[usize::from(picks_batched)][0] * p;
        tot[2][1] += best[usize::from(picks_batched)][1] * p;
        if !picks_batched {
            declined += pieces;
        }
    }
    println!("\n{declined} of 20 groups declined the batched path (batch too wide)");
    for (arm, name) in [(0, "per-step  "), (1, "batched   "), (2, "dispatched")].into_iter() {
        println!(
            "\n{name}: fwd {:.2} ms, fwd+bwd {:.2} ms per sLSTM layer per window \
             ({tot_steps} timesteps) -> enc+dec {:.2} ms",
            tot[arm][0], tot[arm][1], tot[arm][1] * 4.0
        );
    }
}
