//! GPU (cuBLAS / NVRTC) vs CPU (hand-written AVX2) throughput.
//!
//!   cargo run --release --features cuda --example gpu_bench
//!
//! Measures the training-relevant number: compute throughput with operands
//! already resident (weights/activations stay on the device across a step — the
//! whole point of `DTensor`), so per-op host<->device transfer is excluded. One
//! GEMM size is additionally reported *with* a full host round-trip to show why
//! residency matters.
//!
//! GPU timing queues all `iters` launches and synchronizes the stream **once** at
//! the end, so the async launch pipeline is drained inside the timed region and
//! we measure throughput, not per-launch latency.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda` to run this benchmark");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;

    use neural_networks::gpu::{DTensor, Gpu, ops};
    use neural_networks::nn2::linear::Linear as CpuLinear;
    use neural_networks::nn2::optim::AdamCfg;
    use neural_networks::nn2::slstm::SLstm as CpuSLstm;
    use neural_networks::tensor::{Tensor, gemm};

    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU: {e}");
            return;
        }
    };

    /// CPU timer: untimed `warmup` runs, then `iters` timed; returns seconds.
    fn cpu_time(warmup: usize, iters: usize, mut f: impl FnMut()) -> f64 {
        for _ in 0..warmup {
            f();
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            f();
        }
        t0.elapsed().as_secs_f64()
    }

    /// GPU timer: queue `iters` launches, synchronize once at the end (drains the
    /// async pipeline inside the timed region -> throughput, not latency).
    fn gpu_time(gpu: &Gpu, warmup: usize, iters: usize, mut f: impl FnMut()) -> f64 {
        for _ in 0..warmup {
            f();
        }
        gpu.stream.synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            f();
        }
        gpu.stream.synchronize().unwrap();
        t0.elapsed().as_secs_f64()
    }
    let gflops = |flops: f64, iters: usize, secs: f64| flops * iters as f64 / secs / 1e9;

    // This is a Max-Q laptop part: it idles at ~210 MHz and boosts to ~3.1 GHz, so
    // a short timed region measures clock ramp-up, not compute (the same config
    // swung 10x between runs at iters=10). The recurrent benches below therefore
    // use many more GPU iterations than CPU ones — the GPU needs a long, boosted,
    // steady-state region; the CPU is slow enough that a few iterations suffice.
    // For fully stable numbers pin the clocks: `sudo nvidia-smi -lgc 3105`.
    const CPU_ITERS: usize = 3;
    const GPU_ITERS: usize = 100;
    const GPU_WARMUP: usize = 30;

    println!("== GEMM: C[M,N] = A[M,K] · B[K,N]  (2·M·K·N flops) ==");
    println!(
        "{:>16} {:>7} {:>11} {:>11} {:>9}",
        "M,K,N", "iters", "CPU GF/s", "GPU GF/s", "speedup"
    );
    for &(m, k, n) in &[
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 1024, 1024),
    ] {
        let a = Tensor::random(&[m, k], 1.0);
        let b = Tensor::random(&[k, n], 1.0);
        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        let iters = if m * k * n <= 512 * 512 * 512 { 50 } else { 30 };

        let mut c = Tensor::zeros(&[m, n]);
        let cpu_s = cpu_time(3, iters, || {
            gemm::gemm_nn(m, k, n, &a.data, &b.data, &mut c.data, 0.0)
        });

        let da = DTensor::from_host(&gpu, &a);
        let db = DTensor::from_host(&gpu, &b);
        let mut dc = DTensor::zeros(&gpu, &[m, n]);
        let gpu_s = gpu_time(&gpu, 3, iters, || {
            ops::matmul_nn_into(&gpu, &da, &db, &mut dc, 0.0)
        });

        let (cpu_g, gpu_g) = (gflops(flops, iters, cpu_s), gflops(flops, iters, gpu_s));
        println!(
            "{:>6},{:>4},{:>4} {:>7} {:>11.1} {:>11.1} {:>8.1}x",
            m,
            k,
            n,
            iters,
            cpu_g,
            gpu_g,
            gpu_g / cpu_g
        );
    }

    // Transfer-cost illustration on the 1024³ case.
    {
        let (m, k, n) = (1024, 1024, 1024);
        let a = Tensor::random(&[m, k], 1.0);
        let b = Tensor::random(&[k, n], 1.0);
        let iters = 30;
        let da = DTensor::from_host(&gpu, &a);
        let db = DTensor::from_host(&gpu, &b);
        let mut dc = DTensor::zeros(&gpu, &[m, n]);
        let resident = gpu_time(&gpu, 3, iters, || {
            ops::matmul_nn_into(&gpu, &da, &db, &mut dc, 0.0)
        });
        // With a full host round-trip each call, sync is implicit in to_host.
        let with_xfer = cpu_time(3, iters, || {
            let da = DTensor::from_host(&gpu, &a);
            let db = DTensor::from_host(&gpu, &b);
            let c = ops::matmul(&gpu, &da, &db);
            let _ = c.to_host(&gpu);
        });
        println!(
            "\n1024³ per-call: resident {:.2} ms  vs  host round-trip {:.2} ms  ({:.1}x slower with transfer)",
            resident / iters as f64 * 1e3,
            with_xfer / iters as f64 * 1e3,
            with_xfer / resident
        );
    }

    println!("\n== Linear fwd+bwd+step  (~6·B·in·out flops for fwd+bwd) ==");
    println!(
        "{:>16} {:>7} {:>11} {:>11} {:>9}",
        "B,in,out", "iters", "CPU GF/s", "GPU GF/s", "speedup"
    );
    for &(batch, inp, out) in &[(512, 512, 512), (2048, 1024, 1024), (4096, 1024, 2048)] {
        let w = Tensor::xavier(inp, out);
        let bias = Tensor::random(&[out], 0.1);
        let x = Tensor::random(&[batch, inp], 1.0);
        let dy = Tensor::random(&[batch, out], 1.0);
        let flops = 6.0 * batch as f64 * inp as f64 * out as f64;
        let iters = 20;
        let mut cfg = AdamCfg::new(1e-3, 0.01);
        cfg.t = 1;

        let mut cpu = CpuLinear::from_parts(w.clone(), bias.clone());
        let cpu_s = cpu_time(3, iters, || {
            let _y = cpu.forward(&x);
            let _dx = cpu.backward(&dy);
            cpu.step(&cfg);
        });

        let mut dev = neural_networks::gpu::linear::Linear::from_parts(&gpu, &w, &bias);
        let dx_in = DTensor::from_host(&gpu, &x);
        let ddy = DTensor::from_host(&gpu, &dy);
        let gpu_s = gpu_time(&gpu, 3, iters, || {
            let _y = dev.forward(&gpu, &dx_in);
            let _dx = dev.backward(&gpu, &ddy);
            dev.step(&gpu, &cfg);
        });

        let (cpu_g, gpu_g) = (gflops(flops, iters, cpu_s), gflops(flops, iters, gpu_s));
        println!(
            "{:>6},{:>4},{:>4} {:>7} {:>11.1} {:>11.1} {:>8.1}x",
            batch,
            inp,
            out,
            iters,
            cpu_g,
            gpu_g,
            gpu_g / cpu_g
        );
    }

    // sLSTM cell: forward + backward + AdamW step over a whole [B, T, H] sequence.
    // The cell is SEQUENTIAL over T (T serial kernel launches per phase), so this
    // is the launch-bound recurrent baseline the chunkwise mLSTM must beat — the
    // point of measuring it now is to not repeat the old sub-1x scalar-recurrence
    // mistake in mLSTM. rows = in+H = 2H (hidden->hidden). Dominant work is the
    // 4 gate GEMMs fwd + (dW, dxh) bwd ≈ 24·B·rows·H per step.
    println!("\n== sLSTM cell fwd+bwd+step over [B,T,H]  (~24·B·rows·H·T flops, rows=2H) ==");
    println!(
        "{:>16} {:>7} {:>11} {:>11} {:>9}",
        "B,T,H", "iters", "CPU GF/s", "GPU GF/s", "speedup"
    );
    for &(b, t, h) in &[(32, 32, 256), (32, 64, 512), (64, 64, 512)] {
        let rows = 2 * h;
        let x = Tensor::random(&[b, t, h], 0.5);
        let dy = Tensor::random(&[b, t, h], 1.0);
        let flops = 24.0 * b as f64 * rows as f64 * h as f64 * t as f64;
        let mut cfg = AdamCfg::new(1e-3, 0.01);
        cfg.t = 1;

        let mut cpu = CpuSLstm::new(h, h);
        let cpu_s = cpu_time(1, CPU_ITERS, || {
            let _y = cpu.forward(&x);
            let _dx = cpu.backward(&dy);
            cpu.step(&cfg);
        });

        let mut dev = neural_networks::gpu::slstm::SLstm::from_parts(
            &gpu, h, h, &cpu.wz, &cpu.wi, &cpu.wf, &cpu.wo, &cpu.bz, &cpu.bi, &cpu.bf, &cpu.bo,
        );
        let dx_in = DTensor::from_host(&gpu, &x);
        let ddy = DTensor::from_host(&gpu, &dy);
        let gpu_s = gpu_time(&gpu, GPU_WARMUP, GPU_ITERS, || {
            let _y = dev.forward(&gpu, &dx_in);
            let _dx = dev.backward(&gpu, &ddy);
            dev.step(&gpu, &cfg);
        });

        let (cpu_g, gpu_g) = (
            gflops(flops, CPU_ITERS, cpu_s),
            gflops(flops, GPU_ITERS, gpu_s),
        );
        println!(
            "{:>6},{:>3},{:>4} {:>7} {:>11.1} {:>11.1} {:>8.1}x",
            b,
            t,
            h,
            GPU_ITERS,
            cpu_g,
            gpu_g,
            gpu_g / cpu_g
        );
    }

    // mLSTM cell: the payoff measurement. CPU is the scalar per-head recurrence
    // (matrix state C[dhv,dqk] per B·H — poor locality, the sub-1x path); GPU is
    // the single-chunk parallel form (attention-like batched GEMMs). Reported as
    // wall-clock ms/iter + speedup — the flop counts differ between the two
    // formulations (O(T·dhv·dqk) recurrence vs O(T²) parallel), so GF/s would not
    // be comparable. d = heads·dhv, dqk per head.
    println!(
        "\n== mLSTM cell fwd+bwd+step over [B,T,d]  (scalar-recurrence CPU vs parallel GPU) =="
    );
    println!(
        "{:>18} {:>7} {:>13} {:>13} {:>9}",
        "B,T,d,heads,dqk", "iters", "CPU ms/it", "GPU ms/it", "speedup"
    );
    for &(b, t, d, heads, dqk) in &[
        (16, 32, 256, 4, 32),
        (16, 64, 512, 8, 32),
        (32, 64, 512, 8, 64),
    ] {
        let x = Tensor::random(&[b, t, d], 0.5);
        let dy = Tensor::random(&[b, t, d], 1.0);
        let mut cfg = AdamCfg::new(1e-3, 0.01);
        cfg.t = 1;

        let mut cpu = neural_networks::nn2::mlstm::MLstm::new(d, d, heads, dqk);
        let cpu_s = cpu_time(1, CPU_ITERS, || {
            let _y = cpu.forward(&x);
            let _dx = cpu.backward(&dy);
            cpu.step(&cfg);
        });

        let mut dev = neural_networks::gpu::mlstm::MLstm::from_parts(
            &gpu, d, d, heads, dqk, &cpu.wq, &cpu.wk, &cpu.wv, &cpu.wo, &cpu.wi, &cpu.wf, &cpu.bq,
            &cpu.bk, &cpu.bv, &cpu.bo, &cpu.bi, &cpu.bf, &cpu.w_out, &cpu.b_out, &cpu.gamma,
        );
        let dx_in = DTensor::from_host(&gpu, &x);
        let ddy = DTensor::from_host(&gpu, &dy);
        let gpu_s = gpu_time(&gpu, GPU_WARMUP, GPU_ITERS, || {
            let _y = dev.forward(&gpu, &dx_in);
            let _dx = dev.backward(&gpu, &ddy);
            dev.step(&gpu, &cfg);
        });

        let (cpu_ms, gpu_ms) = (
            cpu_s / CPU_ITERS as f64 * 1e3,
            gpu_s / GPU_ITERS as f64 * 1e3,
        );
        println!(
            "{:>6},{:>3},{:>4},{:>2},{:>3} {:>7} {:>13.2} {:>13.2} {:>8.1}x",
            b,
            t,
            d,
            heads,
            dqk,
            GPU_ITERS,
            cpu_ms,
            gpu_ms,
            cpu_ms / gpu_ms
        );
    }

    // The end-to-end number: one full training step of the real LM
    // (Embedding -> sLSTM block -> mLSTM block -> RMSNorm -> head -> SoftCap -> CE
    // -> backward -> AdamW), CPU nn2 stack vs the device-resident `gpu::Lm`.
    println!("\n== LM train_step: Embedding -> sLSTM blk -> mLSTM blk -> norm -> head -> CE ==");
    println!(
        "{:>20} {:>7} {:>13} {:>13} {:>9}",
        "vocab,B,T,H,up", "iters", "CPU ms/it", "GPU ms/it", "speedup"
    );
    for &(vocab, b, t, hidden, up, heads, dqk) in &[
        (
            256usize, 16usize, 32usize, 256usize, 512usize, 4usize, 32usize,
        ),
        (4096, 16, 64, 512, 1024, 8, 64),
    ] {
        use neural_networks::gpu::block::{Block, BlockLike};
        use neural_networks::gpu::lm::Lm;
        use neural_networks::gpu::{mlstm::MLstm, slstm::SLstm};
        use neural_networks::nn2::loss::softmax_cross_entropy;
        use neural_networks::nn2::{Embedding, Linear, MLstmBlock, RmsNorm, SLstmBlock, SoftCap};

        let n = b * t;
        let cap = 30.0;
        let ids: Vec<usize> = (0..n).map(|i| (i * 7 + 1) % vocab).collect();
        let targets: Vec<usize> = (0..n).map(|i| (i * 5 + 2) % vocab).collect();
        let mut cfg = AdamCfg::new(1e-3, 0.01);
        cfg.t = 1;

        // CPU stack.
        let mut emb = Embedding::new(vocab, hidden);
        let mut blk1 = SLstmBlock::new_slstm(hidden, up);
        let mut blk2 = MLstmBlock::new_mlstm(hidden, up, heads, dqk);
        let mut norm = RmsNorm::new(hidden);
        let mut head = Linear::new(hidden, vocab);
        let mut sc = SoftCap::new(cap);

        let cpu_s = cpu_time(1, CPU_ITERS, || {
            let e = emb.forward(&ids);
            let h1 = blk1.forward(&e.reshape(&[b, t, hidden]));
            let h2 = blk2.forward(&h1);
            let nr = norm.forward(&h2.reshape(&[n, hidden]));
            let logits = sc.forward(&head.forward(&nr));
            let (_l, dlog) = softmax_cross_entropy(&logits, &targets);
            let d_nr = head.backward(&sc.backward(&dlog));
            let d_h2 = norm.backward(&d_nr);
            let d_e = blk1.backward(&blk2.backward(&d_h2.reshape(&[b, t, hidden])));
            emb.backward(&d_e.reshape(&[n, hidden]));
            emb.step(&cfg);
            blk1.step(&cfg);
            blk2.step(&cfg);
            norm.step(&cfg);
            head.step_wd(&cfg, false);
        });

        // GPU stack, same architecture.
        let blocks: Vec<Box<dyn BlockLike>> = vec![
            Box::new(Block::<SLstm>::from_cpu(
                &gpu,
                &SLstmBlock::new_slstm(hidden, up),
            )),
            Box::new(Block::<MLstm>::from_cpu(
                &gpu,
                &MLstmBlock::new_mlstm(hidden, up, heads, dqk),
            )),
        ];
        let mut dev = Lm::from_parts(
            &gpu,
            &Tensor::random(&[vocab, hidden], 0.1),
            blocks,
            &Tensor::new(&[hidden], vec![1.0; hidden]),
            &Tensor::xavier(hidden, vocab),
            &Tensor::zeros(&[vocab]),
            cap,
        );
        let gpu_s = gpu_time(&gpu, GPU_WARMUP, GPU_ITERS, || {
            let _loss = dev.train_step(&gpu, &ids, &targets, b, t, &cfg);
        });

        let (cpu_ms, gpu_ms) = (
            cpu_s / CPU_ITERS as f64 * 1e3,
            gpu_s / GPU_ITERS as f64 * 1e3,
        );
        println!(
            "{:>5},{:>3},{:>3},{:>4},{:>4} {:>7} {:>13.2} {:>13.2} {:>8.1}x",
            vocab,
            b,
            t,
            hidden,
            up,
            GPU_ITERS,
            cpu_ms,
            gpu_ms,
            cpu_ms / gpu_ms
        );
    }
}
