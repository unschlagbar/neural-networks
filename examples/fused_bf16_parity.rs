//! Numeric parity of the bf16-staged time-fused sLSTM against the per-step loop,
//! at the backbone's real width.
//!
//! The fp32 fused path does not exist at H=768 (its shared slice needs more blocks
//! than the device has SMs), so the reference here is the per-step loop — the path
//! the cell falls back to today. Reports max absolute and relative error on the
//! output, on dx, and on every gate's weight gradient, rather than asserting a
//! tolerance: the point is to see the size of the bf16 staging error, and decide
//! whether it is mantissa noise or a bug.
//!
//!   SLSTM_BF16=1 cargo run --release --features cuda --example fused_bf16_parity

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use neural_networks::gpu::{DTensor, Gpu, ops, slstm::SLstm};
    use neural_networks::tensor::Tensor;

    let gpu = Gpu::new().expect("gpu");
    if !gpu.kernels.has_coop {
        eprintln!("no cooperative kernels");
        return;
    }
    let h: usize = std::env::var("H").ok().and_then(|v| v.parse().ok()).unwrap_or(768);
    let t: usize = std::env::var("T").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    let b = 1usize;

    println!("H={h} T={t} B={b}, bf16 staging {}", ops::fused_bf16_enabled());
    println!(
        "fwd geometry {:?}\nbwd geometry {:?}",
        ops::slstm_fused_time_geometry(&gpu, h, b),
        ops::slstm_fused_time_bwd_geometry(&gpu, h, b)
    );
    if ops::slstm_fused_time_geometry(&gpu, h, b).is_none() {
        eprintln!("fused path declines this shape — nothing to compare");
        return;
    }

    // Scale the recurrent weights by 1/sqrt(H): at a fixed scale the spectral radius
    // grows with width and an H=768 recurrence diverges within a few dozen steps, so
    // both paths would overflow and the comparison would measure nothing.
    let s = 1.0 / (h as f32).sqrt();
    let w: Vec<Tensor> = (0..4)
        .map(|g| Tensor::random(&[2 * h, h], s * (1.0 + g as f32 * 0.05)))
        .collect();
    let bi: Vec<Tensor> = (0..4)
        .map(|g| Tensor::random(&[h], 0.2 + g as f32 * 0.01))
        .collect();
    let x = DTensor::from_host(&gpu, &Tensor::random(&[b, t, h], 0.5));
    let gy = DTensor::from_host(&gpu, &Tensor::random(&[b, t, h], 0.7));

    let build = |fused: bool| {
        let mut c = SLstm::from_parts(
            &gpu, h, h, &w[0], &w[1], &w[2], &w[3], &bi[0], &bi[1], &bi[2], &bi[3], None,
        );
        c.force_fused_time = Some(fused);
        c
    };
    let mut per_step = build(false);
    let mut fused = build(true);

    let want = per_step.forward_alloc(&gpu, &x).to_host(&gpu);
    let got = fused.forward_alloc(&gpu, &x).to_host(&gpu);
    let want_dx = per_step.backward_alloc(&gpu, &gy).to_host(&gpu);
    let got_dx = fused.backward_alloc(&gpu, &gy).to_host(&gpu);

    // Relative to the tensor's own scale: a max-abs alone says nothing without
    // knowing how big the values are.
    let report = |name: &str, a: &[f32], c: &[f32]| {
        let scale = a.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-30);
        let maxabs = a
            .iter()
            .zip(c)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let nonfinite = c.iter().filter(|v| !v.is_finite()).count();
        println!(
            "{name:<10} max|err| {maxabs:>12.3e}   scale {scale:>10.3e}   rel {:>10.3e}{}",
            maxabs / scale,
            if nonfinite > 0 { format!("   NON-FINITE {nonfinite}") } else { String::new() }
        );
    };
    println!();
    report("out", &want.data, &got.data);
    report("dx", &want_dx.data, &got_dx.data);
    // grads() order: dwx, dwhr, dbcat, post_norm.dgamma.
    let names = ["dWx", "dWh", "dbcat", "dgamma"];
    let (pg, fg) = (per_step.grads(), fused.grads());
    for (i, name) in names.iter().enumerate() {
        let a = pg[i].to_host(&gpu).data.to_vec();
        let c = fg[i].to_host(&gpu).data.to_vec();
        report(name, &a, &c);
    }
    println!("\nbf16 has an 8-bit mantissa: ~4e-3 relative is the noise floor of the\nstaged operand, and error accumulated over T recurrent steps sits above it.");
}
