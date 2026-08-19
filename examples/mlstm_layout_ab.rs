//! Head-major vs position-major addressing in the fused mLSTM kernels.
//!
//!   cargo run --release --features cuda --example mlstm_layout_ab
//!
//! Removing the head-major reorg deletes a streaming pass over q/k/v but costs the
//! kernels their locality: under position-major a timestep's row is `H*W` apart
//! instead of `W`, and the fused kernels re-read q/k/v once per chunk. This measures
//! which side wins at the two shapes the model actually runs.
//!
//! The two arms are INTERLEAVED within a run and the SM clock is printed, because a
//! 1372-2880 MHz swing is larger than the effect being measured.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda` to run this benchmark");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;

    use neural_networks::gpu::{DTensor, Gpu, ops};
    use neural_networks::tensor::Tensor;

    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU: {e}");
            return;
        }
    };

    fn sm_clock() -> String {
        std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=clocks.sm", "--format=csv,noheader"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "?".into())
    }

    /// One timed pass: `iters` fwd+bwd at the given strides.
    fn run(
        gpu: &Gpu,
        q: &ops::SlabBuf,
        k: &ops::SlabBuf,
        v: &ops::SlabBuf,
        ig: &DTensor,
        fg: &DTensor,
        dy: &DTensor,
        l: usize,
        st: ops::MlstmStrides,
        iters: usize,
    ) -> f64 {
        for _ in 0..3 {
            let sv = ops::mlstm_fused_fw(gpu, q, k, v, ig, fg, l, None, st);
            let _ = ops::mlstm_fused_bw(gpu, &sv, q, k, v, ig, fg, dy, None, st);
        }
        gpu.stream.synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            let sv = ops::mlstm_fused_fw(gpu, q, k, v, ig, fg, l, None, st);
            let _ = ops::mlstm_fused_bw(gpu, &sv, q, k, v, ig, fg, dy, None, st);
        }
        gpu.stream.synchronize().unwrap();
        t0.elapsed().as_secs_f64() / iters as f64
    }

    // (label, b, h, t, dqk, dhv) — the backbone runs B=1 at WORD_HIDDEN=768 over a
    // 4096-word window; the encoder/decoder batch short words at OUT_HIDDEN=256.
    let shapes = [
        ("backbone  B=1  T=4096 d=768 H=8", 1usize, 8usize, 4096usize, 96usize, 96usize),
        ("encoder   B=256 T=8    d=256 H=16", 256, 16, 8, 16, 16),
        ("decoder   B=256 T=12   d=256 H=16", 256, 16, 12, 16, 16),
        // H sweep at FIXED head dim (16) and fixed total work: only the row gap
        // changes (position-major scatters a block's L rows by H*W). If the gap is
        // the cause, the ratio must grow with H.
        ("H-sweep   B=256 T=12  H=2  dh=16", 256, 2, 12, 16, 16),
        ("H-sweep   B=256 T=12  H=4  dh=16", 256, 4, 12, 16, 16),
        ("H-sweep   B=256 T=12  H=8  dh=16", 256, 8, 12, 16, 16),
        ("H-sweep   B=256 T=12  H=32 dh=16", 256, 32, 12, 16, 16),
    ];
    let l = 256;
    let iters = 20;

    println!("SM clock before: {}", sm_clock());
    println!(
        "\n{:<34} {:>10} {:>10} {:>8}",
        "shape", "head-maj", "pos-maj", "ratio"
    );

    for (label, b, h, t, dqk, dhv) in shapes {
        let bh = b * h;
        if !ops::mlstm_fused_supported(l.min(t), dqk, dhv) {
            println!("{label:<34} unsupported at this shape (smem)");
            continue;
        }
        let mk = |dims: &[usize], seed: u64| {
            DTensor::from_host(&gpu, &Tensor::random_seeded(dims, 0.5, seed))
        };
        // Same bytes both ways: only the strides differ, so the arms are comparable.
        let slab = |d: &DTensor| ops::SlabBuf::from_f32(&gpu, d.dup(&gpu));
        let (q, k, v) = (
            slab(&mk(&[bh, t, dqk], 1)),
            slab(&mk(&[bh, t, dqk], 2)),
            slab(&mk(&[bh, t, dhv], 3)),
        );
        let ig = mk(&[bh, t], 4);
        let fg = mk(&[bh, t], 5);
        let dy = mk(&[bh, t, dhv], 6);

        let hm = ops::MlstmStrides::head_major(b, h, t, dqk, dhv);
        let pm = ops::MlstmStrides::position_major(b, h, t, dqk, dhv);

        // `ONLY=hm|pm` runs a single arm, so a profiler attributes kernels to one
        // layout instead of interleaving them.
        let only = std::env::var("ONLY").unwrap_or_default();
        if only == "hm" {
            let a = run(&gpu, &q, &k, &v, &ig, &fg, &dy, l, hm, iters);
            println!("{label:<34} head-major {:>9.3}ms", a * 1e3);
            continue;
        }
        if only == "pm" {
            let c = run(&gpu, &q, &k, &v, &ig, &fg, &dy, l, pm, iters);
            println!("{label:<34} pos-major  {:>9.3}ms", c * 1e3);
            continue;
        }
        // Interleave: alternate arms so a clock drift hits both equally.
        let (mut a, mut c) = (0.0, 0.0);
        let rounds = 3;
        for _ in 0..rounds {
            a += run(&gpu, &q, &k, &v, &ig, &fg, &dy, l, hm, iters);
            c += run(&gpu, &q, &k, &v, &ig, &fg, &dy, l, pm, iters);
        }
        let (a, c) = (a / rounds as f64, c / rounds as f64);
        println!(
            "{label:<34} {:>9.3}ms {:>9.3}ms {:>7.3}x",
            a * 1e3,
            c * 1e3,
            c / a
        );
    }
    println!("\nSM clock after:  {}", sm_clock());
    println!("ratio > 1 means position-major (the gather-free path) is SLOWER");
}
