//! Now that the backbone sLSTM's launch cost is gone (CUDA graphs), what is the
//! GPU side actually limited by? Three candidates, and they want different fixes:
//!
//!   a) MEMORY BANDWIDTH. A timestep multiplies h[B, H] by Whr[H, 4H]. At B=1 that
//!      is a mat-VEC: it streams the entire 4H*H weight matrix (4 MB at H=512) out
//!      of VRAM to do 2*H*4H flops. Arithmetic intensity ~= 2 flops/byte * B, i.e.
//!      it is set by the BATCH and nothing else. More SMs cannot help.
//!   b) NODE LATENCY. The recurrence is a strictly dependent chain, so the T steps
//!      cannot overlap; each node pays its own launch latency on the device.
//!   c) OCCUPANCY. At B=1 the fused step kernel has B*H = 512 elements of work —
//!      one block, one SM out of ~48. The card is 98% idle.
//!
//! The two sweeps below separate them:
//!
//!   * SWEEP B at fixed T, H — the weight traffic per step and the node count are
//!     BOTH unchanged, only the useful work grows. If ms/iter stays flat as B rises,
//!     the step is bandwidth- or latency-bound and the batch is free real estate:
//!     the fix is to feed the backbone more sequences at once. If it rises linearly,
//!     we were already saturating the machine.
//!   * SWEEP H at fixed T, B=1 — the node count is unchanged, the weight traffic
//!     grows as H^2. Time flat => latency-bound (b). Time ~ H^2 => bandwidth-bound (a).
//!
//! `eff GB/s` is the weight traffic alone (Whr read once per step, forward and
//! backward) over the elapsed time — compare it to the card's spec bandwidth.

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

    let t = 1024; // the backbone's unroll (words per window)

    // fwd+bwd each read Whr [H, 4H] once per timestep.
    let run = |b: usize, h: usize| -> (f64, f64) {
        let mut dev = SLstm::new_rand(&gpu, h, h);
        let x = DTensor::from_host(&gpu, &Tensor::random(&[b, t, h], 0.5));
        let g = DTensor::from_host(&gpu, &Tensor::random(&[b, t, h], 1.0));
        let iters = 10;

        for _ in 0..2 {
            let y = dev.forward(&gpu, &x);
            drop(y);
            let _ = dev.backward(&gpu, &g);
        }
        gpu.stream.synchronize().unwrap();

        let t0 = Instant::now();
        for _ in 0..iters {
            let y = dev.forward(&gpu, &x);
            drop(y);
            let _ = dev.backward(&gpu, &g);
        }
        gpu.stream.synchronize().unwrap();
        let secs = t0.elapsed().as_secs_f64() / iters as f64;

        let whr_bytes = (h * 4 * h * 4) as f64; // [H, 4H] f32
        let traffic = whr_bytes * (2 * t) as f64; // fwd step + bwd step
        (secs * 1e3, traffic / secs / 1e9)
    };

    let h0 = neural_networks::config::WORD_HIDDEN;
    println!("== sLSTM fwd+bwd, T={t} (graphs on) ==\n");

    println!("SWEEP B  (H={h0}; weight traffic & node count CONSTANT — only work grows)");
    println!("{:>5} {:>10} {:>12} {:>12}", "B", "ms/iter", "vs B=1", "eff GB/s");
    let mut base = 0.0;
    for &b in &[1usize, 2, 4, 8, 16, 32] {
        let (ms, gbs) = run(b, h0);
        if b == 1 {
            base = ms;
        }
        println!("{:>5} {:>10.2} {:>11.2}x {:>12.0}", b, ms, ms / base, gbs);
    }

    println!("\nSWEEP H  (B=1; node count CONSTANT — weight traffic grows as H^2)");
    println!("{:>5} {:>10} {:>12} {:>12}", "H", "ms/iter", "vs H=128", "eff GB/s");
    let mut base = 0.0;
    for &h in &[128usize, 256, 512, 1024] {
        let (ms, gbs) = run(1, h);
        if h == 128 {
            base = ms;
        }
        println!("{:>5} {:>10.2} {:>11.2}x {:>12.0}", h, ms, ms / base, gbs);
    }

    println!(
        "\nB flat  => batching the backbone is ~free: the step is bandwidth/latency bound.\n\
         H flat  => latency bound (fuse nodes). H ~ x4 per doubling => bandwidth bound (batch)."
    );
}
