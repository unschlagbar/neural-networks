//! What does the per-timestep recurrent GEMM actually cost the HOST to submit?
//!
//! `launch_cost` measured a plain fp32 cuBLAS call at backbone shape. The per-step
//! sLSTM loop does not use that: it calls `GemmBf16::run_staged_lhs`, which may take a
//! different cuBLAS entry point with a cached algorithm. If that one is cheap, the
//! whole "kill the submission cost" plan is built on the wrong number.
//!
//! Issue time is measured with NO sync in the loop — what the CPU pays to submit.
//! Wall is with a sync at the end. issue/wall near 1 means host-bound.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda`");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;

    use neural_networks::gpu::{GTensor, Gpu, ops};
    use neural_networks::tensor::Tensor;

    let gpu = Gpu::new().expect("gpu");
    let n: usize = 4000;

    // (label, B, H) — the recurrent product is [B,H] x [H,4H], once per timestep.
    let shapes: &[(&str, usize, usize)] = &[
        ("backbone  B=1    H=1024", 1, 1024),
        ("encoder   B=120  H=256", 120, 256),
        ("encoder   B=409  H=256", 409, 256),
        ("encoder   B=1024 H=256", 1024, 256),
        ("encoder   B=2048 H=256", 2048, 256),
    ];

    println!("{n} launches each\n");
    println!(
        "{:<26} {:>10} {:>10} {:>9} {:>11}",
        "shape", "issue us", "wall us", "issue/w", "gpu GFLOP/s"
    );

    for &(label, b, h) in shapes {
        let h4 = 4 * h;
        let whr = GTensor::from_host(&gpu, &Tensor::random_seeded(&[h, h4], 0.05, 1));
        let hn = GTensor::from_host(&gpu, &Tensor::random_seeded(&[b, h], 0.5, 2));
        let mut hb: ops::SlabBuf = ops::SlabBuf::new_width(&gpu, &[b, h], true);
        hb.store(&gpu, &hn);
        let mut gh: GTensor<f32> = GTensor::uninit(&gpu, &[b, h4]);
        let mut gemm = ops::GemmBf16::default();

        let mut once = |gemm: &mut ops::GemmBf16, gh: &mut GTensor<f32>| match &hb {
            ops::SlabBuf::Bf16(x) => gemm.run_staged_lhs(&gpu, ops::MmForm::Nn, x, &whr, gh, 0.0),
            ops::SlabBuf::F32(x) => ops::matmul_nn_into(&gpu, x, &whr, gh, 0.0),
        };

        for _ in 0..200 {
            once(&mut gemm, &mut gh);
        }
        gpu.stream.synchronize().ok();

        // Issue: no sync inside the loop.
        let t0 = Instant::now();
        for _ in 0..n {
            once(&mut gemm, &mut gh);
        }
        let issue = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
        gpu.stream.synchronize().ok();

        // Wall: same loop, but timed through to completion.
        let t1 = Instant::now();
        for _ in 0..n {
            once(&mut gemm, &mut gh);
        }
        gpu.stream.synchronize().ok();
        let wall = t1.elapsed().as_secs_f64() * 1e6 / n as f64;

        let gflop = 2.0 * b as f64 * h as f64 * h4 as f64 / 1e9;
        println!(
            "{label:<26} {issue:>10.2} {wall:>10.2} {:>9.2} {:>11.0}",
            issue / wall,
            gflop / (wall * 1e-6)
        );
    }
}
