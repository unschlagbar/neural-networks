//! fp32 vs slab RMSNorm, at the shapes the model actually normalizes.
//!
//! The question is narrow: the norm's arithmetic is fp32 either way, so this measures
//! only what changes when the OUTPUT is stored bf16 — half the store in forward, half
//! the `y` read in backward (twice: the kernel and the `dγ` reduction).
//!
//! It deliberately does NOT measure the larger half of the win, which is the
//! `cast_f32_to_bf16` pass this deletes downstream: that cast belongs to the consumer,
//! not to the norm. Read this as the floor.
//!
//! Two things this has to get right or it measures nothing:
//!
//!   * **One sync per REPS launches, not per launch.** These kernels run 1-20 us. A
//!     `synchronize` on either side of a single launch costs ~380 us and reports it as
//!     the kernel — every shape then looks identical, which is exactly what the first
//!     version of this file said.
//!   * **A working set past L2.** One `[512, 1024]` fp32 tensor is 2 MB and this card's
//!     L2 is ~48 MB, so a tight loop over one buffer never leaves cache and halving the
//!     bytes in HBM changes nothing. The bench cycles `SETS` distinct buffers.
//!
//! Run: cargo run --release --features cuda --example rms_bench

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;

    use neural_networks::config::{BACKBONE_CHUNK, CHAR_HIDDEN, GROUP_MAX_ROWS, WORD_HIDDEN};
    use neural_networks::gpu::rms_norm::RmsNorm;
    use neural_networks::gpu::{GTensor, Gpu, ops};
    use neural_networks::tensor::Tensor;

    let gpu = Gpu::new().expect("gpu");
    use neural_networks::gpu::arena::TrainingCache;
    let clk = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=clocks.sm", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();
    println!(
        "slab_bf16 = {}   sm clock at start: {}",
        gpu.kernels.slab_bf16,
        clk.trim()
    );

    // (label, rows, width, group). The three block norms are group == width; the
    // mLSTM head norm splits the row into `dhv`-wide groups, a different launch shape.
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("backbone block", BACKBONE_CHUNK, WORD_HIDDEN, WORD_HIDDEN),
        ("backbone head", BACKBONE_CHUNK, WORD_HIDDEN, WORD_HIDDEN / 8),
        ("enc/dec block", GROUP_MAX_ROWS, CHAR_HIDDEN, CHAR_HIDDEN),
        ("enc/dec head", GROUP_MAX_ROWS, CHAR_HIDDEN, CHAR_HIDDEN / 8),
    ];
    const REPS: usize = 200;
    const ROUNDS: usize = 15;
    const SETS: usize = 24;

    println!(
        "\n{:<16} {:>13} {:>10} {:>10} {:>8}  {}",
        "shape", "shape", "fp32 us", "slab us", "ratio", "phase"
    );

    for &(label, rows, width, group) in shapes {
        let g = Tensor::random(&[width], 0.4);
        let xs: Vec<GTensor<f32>> = (0..SETS)
            .map(|i| GTensor::from_host(&gpu, &Tensor::random(&[rows, width], 0.7 + i as f32 * 0.01)))
            .collect();
        let dys: Vec<GTensor<f32>> = (0..SETS)
            .map(|i| GTensor::from_host(&gpu, &Tensor::random(&[rows, width], 0.9 + i as f32 * 0.01)))
            .collect();
        let mut dxs: Vec<GTensor<f32>> = (0..SETS)
            .map(|_| GTensor::uninit(&gpu, &[rows, width]))
            .collect();
        let cache = TrainingCache::for_shape(&gpu, rows, width);
        let mut ys_w: Vec<GTensor<f32>> = (0..SETS)
            .map(|_| GTensor::uninit(&gpu, &[rows, width]))
            .collect();
        let mut ys_n: Vec<ops::SlabBuf> = (0..SETS)
            .map(|_| ops::SlabBuf::new(&gpu, &[rows, width]))
            .collect();

        let mut wide = RmsNorm::from_parts_grouped(&gpu, &g, group);
        let mut narrow = RmsNorm::from_parts_grouped(&gpu, &g, group);
        // Seed every `y` once, so backward has something to read on its first timed rep.
        for i in 0..SETS {
            wide.forward(&gpu, &xs[i], &mut ys_w[i]);
            narrow.forward_slab(&gpu, &xs[i], &mut ys_n[i]);
        }

        // A/B interleaved per round rather than one long A then one long B: the SM
        // clock swings 1372-2880 MHz on this card, and a back-to-back A-then-B
        // attributes the whole ramp to whichever ran second.
        // Per-round times, reported as the MINIMUM rather than the mean. Anything
        // else running on the card only ever makes a round slower, so the fastest
        // round is the one least contaminated by it — the usual defence on a machine
        // that is not exclusively yours. The spread is printed so a run where even the
        // minimum is contended is visible rather than silently believed.
        let mut t_fw = [Vec::new(), Vec::new()];
        let mut t_bw = [Vec::new(), Vec::new()];
        for round in 0..ROUNDS + 1 {
            for variant in 0..2 {
                gpu.stream.synchronize().expect("sync");
                let t0 = Instant::now();
                for r in 0..REPS {
                    let i = r % SETS;
                    if variant == 0 {
                        wide.forward(&gpu, &xs[i], &mut ys_w[i]);
                    } else {
                        narrow.forward_slab(&gpu, &xs[i], &mut ys_n[i]);
                    }
                }
                gpu.stream.synchronize().expect("sync");
                let fw = t0.elapsed().as_secs_f64();

                let t1 = Instant::now();
                for r in 0..REPS {
                    let i = r % SETS;
                    if variant == 0 {
                        wide.backward(&gpu, &dys[i], &ys_w[i], &mut dxs[i], &cache);
                    } else {
                        narrow.backward_slab(&gpu, &dys[i], &ys_n[i], &mut dxs[i], &cache);
                    }
                }
                gpu.stream.synchronize().expect("sync");
                let bw = t1.elapsed().as_secs_f64();

                // Round 0 is warmup: JIT, allocator growth and the clock ramp.
                if round > 0 {
                    t_fw[variant].push(fw);
                    t_bw[variant].push(bw);
                }
            }
        }
        let stats = |v: &[f64]| {
            let us: Vec<f64> = v.iter().map(|t| t / REPS as f64 * 1e6).collect();
            let lo = us.iter().cloned().fold(f64::MAX, f64::min);
            let hi = us.iter().cloned().fold(0.0, f64::max);
            (lo, hi / lo)
        };
        let ((fw0, s0), (fw1, s1)) = (stats(&t_fw[0]), stats(&t_fw[1]));
        let ((bw0, s2), (bw1, s3)) = (stats(&t_bw[0]), stats(&t_bw[1]));
        println!(
            "{label:<16} {:>13} {fw0:>10.2} {fw1:>10.2} {:>7.2}x  forward   spread {:.2}/{:.2}",
            format!("{rows}x{width}"),
            fw0 / fw1,
            s0,
            s1
        );
        println!(
            "{:<16} {:>13} {bw0:>10.2} {bw1:>10.2} {:>7.2}x  backward  spread {:.2}/{:.2}",
            "",
            format!("group {group}"),
            bw0 / bw1,
            s2,
            s3
        );
    }

    // The chain the block actually runs, which is the case the change is for. The
    // norm's output is consumed by a GEMM that wants bf16, so the fp32 path pays for
    // the norm AND for a `cast_f32_to_bf16` pass that reads the result straight back
    // out of HBM. The slab path produces those bits once.
    println!(
        "\n--- norm + the narrowing its consumer needs ---\n{:<16} {:>13} {:>10} {:>10} {:>8}  {}",
        "shape", "shape", "fp32 us", "slab us", "ratio", "phase"
    );
    for &(label, rows, width, group) in shapes {
        if group != width {
            continue; // head norms feed an elementwise op, not a GEMM
        }
        let g = Tensor::random(&[width], 0.4);
        let xs: Vec<GTensor<f32>> = (0..SETS)
            .map(|i| GTensor::from_host(&gpu, &Tensor::random(&[rows, width], 0.7 + i as f32 * 0.01)))
            .collect();
        let dys: Vec<GTensor<f32>> = (0..SETS)
            .map(|i| GTensor::from_host(&gpu, &Tensor::random(&[rows, width], 0.9 + i as f32 * 0.01)))
            .collect();
        let mut dxs: Vec<GTensor<f32>> = (0..SETS)
            .map(|_| GTensor::uninit(&gpu, &[rows, width]))
            .collect();
        let cache = TrainingCache::for_shape(&gpu, rows, width);
        let mut xb = cache.temps.get::<u16>(&gpu, &[rows, width]);
        let mut ys_w: Vec<GTensor<f32>> = (0..SETS)
            .map(|_| GTensor::uninit(&gpu, &[rows, width]))
            .collect();
        let mut ys_n: Vec<ops::SlabBuf> = (0..SETS)
            .map(|_| ops::SlabBuf::new(&gpu, &[rows, width]))
            .collect();
        let mut wide = RmsNorm::from_parts_grouped(&gpu, &g, group);
        let mut narrow = RmsNorm::from_parts_grouped(&gpu, &g, group);
        for i in 0..SETS {
            wide.forward(&gpu, &xs[i], &mut ys_w[i]);
            narrow.forward_slab(&gpu, &xs[i], &mut ys_n[i]);
        }

        let mut t_fw = [Vec::new(), Vec::new()];
        let mut t_bw = [Vec::new(), Vec::new()];
        for round in 0..ROUNDS + 1 {
            for variant in 0..2 {
                gpu.stream.synchronize().expect("sync");
                let t0 = Instant::now();
                for r in 0..REPS {
                    let i = r % SETS;
                    if variant == 0 {
                        wide.forward(&gpu, &xs[i], &mut ys_w[i]);
                        // What the FFN does today: narrow the fp32 result once for the
                        // pair of projections that read it.
                        xb.store(&gpu, &ys_w[i]);
                    } else {
                        narrow.forward_slab(&gpu, &xs[i], &mut ys_n[i]);
                    }
                }
                gpu.stream.synchronize().expect("sync");
                t0.elapsed();
                let fw = t0.elapsed().as_secs_f64();

                let t1 = Instant::now();
                for r in 0..REPS {
                    let i = r % SETS;
                    if variant == 0 {
                        xb.store(&gpu, &ys_w[i]);
                        wide.backward(&gpu, &dys[i], &ys_w[i], &mut dxs[i], &cache);
                    } else {
                        narrow.backward_slab(&gpu, &dys[i], &ys_n[i], &mut dxs[i], &cache);
                    }
                }
                gpu.stream.synchronize().expect("sync");
                let bw = t1.elapsed().as_secs_f64();
                if round > 0 {
                    t_fw[variant].push(fw);
                    t_bw[variant].push(bw);
                }
            }
        }
        let stats = |v: &[f64]| {
            let us: Vec<f64> = v.iter().map(|t| t / REPS as f64 * 1e6).collect();
            let lo = us.iter().cloned().fold(f64::MAX, f64::min);
            let hi = us.iter().cloned().fold(0.0, f64::max);
            (lo, hi / lo)
        };
        let ((fw0, s0), (fw1, s1)) = (stats(&t_fw[0]), stats(&t_fw[1]));
        let ((bw0, s2), (bw1, s3)) = (stats(&t_bw[0]), stats(&t_bw[1]));
        println!(
            "{label:<16} {:>13} {fw0:>10.2} {fw1:>10.2} {:>7.2}x  fwd+cast  spread {:.2}/{:.2}",
            format!("{rows}x{width}"),
            fw0 / fw1,
            s0,
            s1
        );
        println!(
            "{:<16} {:>13} {bw0:>10.2} {bw1:>10.2} {:>7.2}x  bwd+cast  spread {:.2}/{:.2}",
            "",
            format!("group {group}"),
            bw0 / bw1,
            s2,
            s3
        );
    }
}
