//! `add_col_sum` at the shapes a real step runs it at.
//!
//! The kernel is 10 % of GPU time across ~4.7k calls per step. It reads one fp32
//! `[rows, n]` activation and produces `[n]`, so it is pure bandwidth — this prints
//! the achieved fraction of it, plus the block/grid the launcher picks, to show
//! whether the shortfall is bandwidth or occupancy.
//!
//!   cargo run --release --features cuda --example colsum_bench

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda` to run this benchmark");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;

    use neural_networks::gpu::{GTensor, Gpu, ops};
    use neural_networks::tensor::Tensor;

    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU: {e}");
            return;
        }
    };

    // (rows, n, use_mul, calls per step) — measured with COLSUM_HIST over one step.
    let shapes: &[(usize, usize, bool, usize)] = &[
        (512, 768, false, 1204),
        (512, 768, true, 672),
        (512, 2048, false, 448),
        (512, 16, false, 392),
        (511, 768, false, 172),
        (2045, 256, false, 140),
        (2045, 256, true, 130),
        (511, 768, true, 96),
        (2045, 682, false, 80),
        (511, 2048, false, 64),
        (511, 16, false, 56),
        (2045, 16, false, 40),
    ];

    const ITERS: usize = 300;
    let mut total_us = 0.0;

    println!("== add_col_sum, {ITERS} iters/shape ==\n");
    println!(
        "{:>6} {:>6} {:>4}  {:>9} {:>9} {:>8}  {:>10} {:>6}",
        "rows", "n", "mul", "MB read", "us/call", "GB/s", "grid,block", "share"
    );

    for &(rows, n, use_mul, calls) in shapes {
        let dy = GTensor::from_host(&gpu, &Tensor::random(&[rows, n], 0.5));
        let mul = GTensor::from_host(&gpu, &Tensor::random(&[rows, n], 0.5));
        let mut db = GTensor::zeros(&gpu, &[n]);

        for _ in 0..30 {
            if use_mul {
                ops::add_col_sum_mul(&gpu, &mut db, &dy, &mul);
            } else {
                ops::add_col_sum(&gpu, &mut db, &dy);
            }
        }
        gpu.stream.synchronize().unwrap();

        let t0 = Instant::now();
        for _ in 0..ITERS {
            if use_mul {
                ops::add_col_sum_mul(&gpu, &mut db, &dy, &mul);
            } else {
                ops::add_col_sum(&gpu, &mut db, &dy);
            }
        }
        gpu.stream.synchronize().unwrap();
        let us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

        let operands = if use_mul { 2 } else { 1 };
        let bytes = (rows * n * 4 * operands) as f64;
        let mb = bytes / 1e6;
        let gbs = bytes / (us * 1e3);

        // Mirror the launcher's own block/grid choice (ops::col_sum_into).
        let bx = n.next_power_of_two().min(32).max(1);
        let by = (512 / bx).min(rows.next_power_of_two()).max(1);
        let grid = n.div_ceil(bx);

        let share = us * calls as f64 / 1e3;
        total_us += share;
        println!(
            "{rows:>6} {n:>6} {:>4}  {mb:>9.2} {us:>9.2} {gbs:>8.0}  {:>10} {share:>5.1}ms",
            use_mul as u8,
            format!("{grid},{bx}x{by}"),
        );
    }
    println!("\nper step over these shapes: {total_us:.1} ms");
}
