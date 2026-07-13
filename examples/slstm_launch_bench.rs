//! sLSTM cell at the BACKBONE shape — is it GPU-bound or launch-bound?
//!
//!   cargo run --release --features cuda --example slstm_launch_bench
//!
//! The backbone runs the sLSTM at batch 1 over T = the window's word count, so a
//! timestep's GEMM is a `[1, H] x [H, 4H]` matvec — a few hundred kFLOP, which any
//! modern card finishes in ~2 us. But the cell's time loop issues TWO launches per
//! timestep (the recurrent GEMM + the fused step kernel), and a launch costs the
//! HOST 5-10 us regardless of which card is on the other end. If that is what we
//! are paying, the GPU is idle most of the time and a faster GPU buys nothing.
//!
//! This bench separates the two costs:
//!
//!   * `issue`  — wall time of the fwd+bwd call itself, WITHOUT a trailing sync.
//!                CUDA launches are async, so this is (almost) pure host time:
//!                the cost of the driver calls, not of the kernels.
//!   * `wall`   — the same region with a `synchronize()` at the end: host issue +
//!                whatever GPU work had not finished by then.
//!
//! `issue / wall ~ 1.0` means the host cannot feed the GPU fast enough — the queue
//! never gets ahead, the card waits on the CPU, and the fix is to launch less (CUDA
//! graphs / fusing steps), not to compute faster. `issue / wall << 1` means the GPU
//! is genuinely the bottleneck and the kernels are what to optimize.
//!
//! `launches` is the count the cell must issue for that T (2 per step each way plus
//! the fixed per-call ones), and `us/launch` divides the host issue time by it —
//! that number should land near the driver's per-launch floor if we are launch-bound.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda` to run this benchmark");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;

    use neural_networks::gpu::{DTensor, Gpu, slstm::SLstm};
    use neural_networks::tensor::Tensor;

    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU: {e}");
            return;
        }
    };

    // Backbone shape: one window (B=1), width WORD_HIDDEN.
    let (b, d) = (1, neural_networks::config::WORD_HIDDEN);

    println!("== sLSTM cell fwd+bwd, B={b} d={d} (backbone shape) ==");
    println!(
        "{:>7} {:>6} {:>10} {:>10} {:>8} {:>10} {:>11}",
        "T", "iters", "wall ms", "issue ms", "issue/w", "launches", "us/launch",
    );

    for &t in &[64, 128, 256, 512, 1024] {
        let mut dev = SLstm::new_rand(&gpu, d, d);
        let x = DTensor::from_host(&gpu, &Tensor::random(&[b, t, d], 0.5));
        let g = DTensor::from_host(&gpu, &Tensor::random(&[b, t, d], 1.0));

        let iters = if t <= 256 { 30 } else { 10 };
        let warmup = (iters / 5).max(2);

        for _ in 0..warmup {
            let y = dev.forward(&gpu, &x);
            drop(y);
            let _ = dev.backward(&gpu, &g);
        }
        gpu.stream.synchronize().unwrap();

        // Host issue time: no sync inside the loop, so the elapsed time is what the
        // CPU spent pushing driver calls (plus any stall once the queue fills).
        let t0 = Instant::now();
        for _ in 0..iters {
            let y = dev.forward(&gpu, &x);
            drop(y);
            let _ = dev.backward(&gpu, &g);
        }
        let issue = t0.elapsed().as_secs_f64();
        gpu.stream.synchronize().unwrap();
        let wall = t0.elapsed().as_secs_f64();

        // Per timestep: forward does 1 GEMM + 1 fused kernel, backward the same.
        // Fixed per call: pack (2), the x-half GEMM, and the 5 post-loop GEMMs/kernels.
        let per_iter_launches = 4 * t + 11;
        let launches = (per_iter_launches * iters) as f64;

        println!(
            "{:>7} {:>6} {:>10.2} {:>10.2} {:>8.2} {:>10} {:>11.2}",
            t,
            iters,
            wall / iters as f64 * 1e3,
            issue / iters as f64 * 1e3,
            issue / wall,
            per_iter_launches,
            issue / launches * 1e6,
        );
    }

    println!(
        "\nissue/w near 1.00 => host-bound: the GPU is waiting on driver calls, \n\
         and the win is in issuing fewer of them (CUDA graphs / fused steps)."
    );
}
