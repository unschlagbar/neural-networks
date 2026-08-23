//! Specialized-vs-generic fused mLSTM kernels, interleaved.
//!
//!   cargo run --release --features cuda --example mlstm_spec_ab
//!
//! The two parallel mma kernels can be built with `(L, dqk, dhv, H)` as NVRTC
//! constants instead of runtime arguments. This measures whether that folding
//! reaches wall clock at a fused-eligible shape.
//!
//! Rounds ALTERNATE the two paths rather than timing one after the other: the SM
//! clock on this part swings by more than 2x under load, so two consecutive blocks
//! of samples measure the clock ramp as much as the kernels. `clocks.sm` is printed
//! per round so a run taken during a ramp is visible rather than silently averaged
//! in.
//!
//! `dqk` is swept over the fused-eligible widths only — 96 (the backbone) does not
//! fit in shared memory and falls back to the op-at-a-time path, where neither
//! kernel runs at all.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda` to run this benchmark");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;

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

    // Reading the clock is what makes a suspicious round attributable rather than
    // just noisy; absent nvidia-smi it is simply not reported.
    let sm_clock = || -> String {
        std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=clocks.sm", "--format=csv,noheader,nounits"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "?".into())
    };

    let (b, t, heads) = (1, 4096, 8);
    let rounds = 5;
    let iters = 10;

    println!("== fused mLSTM: specialized vs generic, B={b} T={t} heads={heads} ==");
    println!("interleaved, {rounds} rounds x {iters} iters");
    println!("MLSTM_TV = {}\n", std::env::var("MLSTM_TV").unwrap_or_else(|_| "unset (built-in default)".into()));

    for &dqk in &[32, 64] {
        let d = dqk * heads;
        let x = GTensor::from_host(&gpu, &Tensor::random(&[b, t, d], 0.5));
        let g = GTensor::from_host(&gpu, &Tensor::random(&[b, t, d], 1.0));

        // One cell per path: the specialized kernel is chosen inside the launch, so
        // the two differ only in the env var read at that point.
        let mut run = |spec: bool| -> f64 {
            // SAFETY: single-threaded, and the kernel cache keys on the flag's
            // effect (the compiled variant), not on the variable itself.
            unsafe {
                if spec {
                    std::env::remove_var("MLSTM_NO_SPECIALIZE");
                } else {
                    std::env::set_var("MLSTM_NO_SPECIALIZE", "1");
                }
            }
            let mut dev = MLstm::new_rand(&gpu, d, d, heads, dqk);
            let mut y = GTensor::uninit(&gpu, &[b, t, d]);
            for _ in 0..3 {
                dev.forward(&gpu, &x, &mut y, &mut cache);
                let _ = dev.backward_alloc(&gpu, &g);
            }
            gpu.stream.synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..iters {
                dev.forward(&gpu, &x, &mut y, &mut cache);
                let _ = dev.backward_alloc(&gpu, &g);
            }
            gpu.stream.synchronize().unwrap();
            t0.elapsed().as_secs_f64() / iters as f64 * 1e3
        };

        println!("-- dqk = dhv = {dqk} (d = {d}) --");
        println!("{:>6} {:>12} {:>12} {:>9} {:>10}", "round", "spec ms", "generic ms", "ratio", "clocks.sm");
        let (mut sum_s, mut sum_g) = (0.0, 0.0);
        for r in 0..rounds {
            let s = run(true);
            let gm = run(false);
            sum_s += s;
            sum_g += gm;
            println!("{:>6} {:>12.2} {:>12.2} {:>9.3} {:>10}", r, s, gm, gm / s, sm_clock());
        }
        println!(
            "{:>6} {:>12.2} {:>12.2} {:>9.3}\n",
            "mean",
            sum_s / rounds as f64,
            sum_g / rounds as f64,
            sum_g / sum_s,
        );
    }
}
