//! What does ONE launch cost the host, broken down by kind?
//!
//!   cargo run --release --features cuda --example launch_cost
//!
//! `slstm_launch_bench` shows the backbone sLSTM is host-bound at ~12 us per
//! launch. The cell issues two things per timestep — a cuBLAS GEMM (the recurrent
//! half, a [1, H] x [H, 4H] matvec at backbone shape) and the fused step kernel —
//! so this splits that 12 us between them, against an empty-kernel baseline that
//! measures the driver's own floor.
//!
//! All timings are HOST issue time (no sync inside the loop): the question is what
//! the CPU pays to submit the work, not what the GPU pays to run it.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda` to run this benchmark");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;

    use cudarc::driver::{LaunchConfig, PushKernelArg};
    use neural_networks::gpu::{DTensor, Gpu, ops};
    use neural_networks::tensor::Tensor;

    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU: {e}");
            return;
        }
    };

    const N: usize = 20_000;
    let h = neural_networks::config::WORD_HIDDEN; // 512
    let h4 = 4 * h;

    // The exact operands the backbone's sLSTM loop uses at B=1.
    let hs = DTensor::from_host(&gpu, &Tensor::random(&[1, h], 0.5)); // h_{t-1}
    let whr = DTensor::from_host(&gpu, &Tensor::random(&[h, h4], 0.05)); // recurrent W
    let mut gh = DTensor::zeros(&gpu, &[1, h4]);

    println!("== host issue cost per launch, {N} launches each (B=1, H={h}) ==\n");

    // 1. cuBLAS GEMM: the recurrent half, [1,H] x [H,4H].
    let t0 = Instant::now();
    for _ in 0..N {
        ops::matmul_nn_into(&gpu, &hs, &whr, &mut gh, 0.0);
    }
    let gemm = t0.elapsed().as_secs_f64();
    gpu.stream.synchronize().unwrap();
    let gemm_wall = t0.elapsed().as_secs_f64();

    // 2. A trivial elementwise kernel from our own module (`scale_`), standing in
    //    for the fused step kernel: same launch path (kernels.get + launch_builder),
    //    negligible GPU work, so what it measures is our per-launch host overhead.
    let mut scratch = DTensor::zeros(&gpu, &[1, h4]);
    let t0 = Instant::now();
    for _ in 0..N {
        ops::scale_(&gpu, &mut scratch, 1.0);
    }
    let kern = t0.elapsed().as_secs_f64();
    gpu.stream.synchronize().unwrap();
    let kern_wall = t0.elapsed().as_secs_f64();

    // 3. Raw driver floor: the same kernel, but hoist the hashmap lookup and the
    //    CudaFunction clone out of the loop. The delta against (2) is what our
    //    `kernels.get(name)` per-launch bookkeeping costs.
    let f = gpu.kernels.get("scale_inplace");
    let n_i = (h4) as i32;
    let s = 1.0f32;
    let cfg = LaunchConfig::for_num_elems(h4 as u32);
    let t0 = Instant::now();
    for _ in 0..N {
        let mut lb = gpu.stream.launch_builder(&f);
        lb.arg(&mut scratch.buf).arg(&s).arg(&n_i);
        unsafe { lb.launch(cfg) }.expect("launch");
    }
    let raw = t0.elapsed().as_secs_f64();
    gpu.stream.synchronize().unwrap();
    let raw_wall = t0.elapsed().as_secs_f64();

    let us = |t: f64| t / N as f64 * 1e6;
    println!("{:<34} {:>10} {:>10} {:>9}", "", "issue us", "wall us", "issue/w");
    println!(
        "{:<34} {:>10.2} {:>10.2} {:>9.2}",
        "cuBLAS gemm [1,H]x[H,4H]",
        us(gemm),
        us(gemm_wall),
        gemm / gemm_wall
    );
    println!(
        "{:<34} {:>10.2} {:>10.2} {:>9.2}",
        "our kernel (via kernels.get)",
        us(kern),
        us(kern_wall),
        kern / kern_wall
    );
    println!(
        "{:<34} {:>10.2} {:>10.2} {:>9.2}",
        "same kernel, fn hoisted (floor)",
        us(raw),
        us(raw_wall),
        raw / raw_wall
    );
    println!(
        "\nper timestep the cell issues 1 gemm + 1 kernel = {:.1} us of host time.",
        us(gemm) + us(kern)
    );
}
