//! Our fused sLSTM T-loop in isolation, for a head-to-head against FlashRNN.
//!
//! `flashrnn()` takes `Wx` already computed and returns the state sequence, so the
//! comparable span on our side is `slstm_fused_time` / `_bwd` alone -- NOT
//! `SLstm::forward`, which also runs the whole-sequence `x.Wx` GEMM. Timing the
//! wrong span would hand FlashRNN a ~30% head start it did not earn.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda`");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;

    use neural_networks::gpu::ops::{SlabBuf, SlstmSlabs};
    use neural_networks::gpu::{GTensor, Gpu, ops};
    use neural_networks::tensor::Tensor;

    let gpu = Gpu::new().expect("gpu");
    let iters: usize = 20;

    // (label, B, T, H)
    // Override with `-- B T H` to match a FlashRNN run shape for shape.
    let a: Vec<usize> = std::env::args().skip(1).filter_map(|v| v.parse().ok()).collect();
    let owned: Vec<(String, usize, usize, usize)> = if a.len() == 3 {
        vec![(format!("B={} T={} H={}", a[0], a[1], a[2]), a[0], a[1], a[2])]
    } else {
        vec![
            ("backbone B=1 T=512 H=1024".into(), 1, 512, 1024),
            ("encoder  B=227 T=8 H=256".into(), 227, 8, 256),
        ]
    };
    let shapes: Vec<(&str, usize, usize, usize)> =
        owned.iter().map(|(l, b, t, h)| (l.as_str(), *b, *t, *h)).collect();
    let shapes = shapes.as_slice();

    println!("{:<30} {:>9} {:>12}", "shape", "fwd ms", "fwd+bwd ms");
    for &(label, b, t, h) in shapes {
        let h4 = 4 * h;
        let act = || SlabBuf::new(&gpu, &[b, t, h]);
        let state = || SlabBuf::new_width(&gpu, &[b, t, h], gpu.kernels.state_bf16);
        let entry = || SlabBuf::new_width(&gpu, &[b, h], gpu.kernels.state_bf16);
        let mut slabs = SlstmSlabs {
            c_entry: entry(),
            n_entry: entry(),
            i_prime: state(),
            f_prime: state(),
            c: state(),
            n: state(),
            zt: act(),
            ot: act(),
            h_prev: act(),
        };
        let whr = GTensor::from_host(
            &gpu,
            &Tensor::random_seeded(&[h, h4], (h as f32).powf(-0.5), 1),
        );
        let bcat = GTensor::zeros(&gpu, &[h4]);
        let mut g = GTensor::from_host(&gpu, &Tensor::random_seeded(&[b, t, h4], 0.5, 2));
        let dy = GTensor::from_host(&gpu, &Tensor::random_seeded(&[b, t, h], 0.5, 3));
        let mut out: GTensor<f32> = GTensor::uninit(&gpu, &[b, t, h]);
        let (mut cs, mut ns, mut ms, mut hs) = (
            GTensor::zeros(&gpu, &[b, h]),
            GTensor::zeros(&gpu, &[b, h]),
            GTensor::zeros(&gpu, &[b, h]),
            GTensor::zeros(&gpu, &[b, h]),
        );
        let (mut dh, mut dc, mut dn) = (
            GTensor::zeros(&gpu, &[b, h]),
            GTensor::zeros(&gpu, &[b, h]),
            GTensor::zeros(&gpu, &[b, h]),
        );

        let mut fwd = |g: &mut GTensor<f32>, slabs: &mut SlstmSlabs, out: &mut GTensor<f32>| {
            ops::slstm_fused_time(
                &gpu, &whr, g, &bcat, &mut cs, &mut ns, &mut ms, &mut hs, slabs, out, t, false,
            )
        };
        let ok = fwd(&mut g, &mut slabs, &mut out);
        if !ok {
            println!("{label:<30} {:>9} (fused declined)", "-");
            continue;
        }
        for _ in 0..5 {
            fwd(&mut g, &mut slabs, &mut out);
            ops::slstm_fused_time_bwd(
                &gpu, &whr, &dy, &mut g, &mut dh, &mut dc, &mut dn, &slabs, t,
            );
        }
        gpu.stream.synchronize().ok();

        let mut best_f = f64::MAX;
        let mut best_fb = f64::MAX;
        for _ in 0..5 {
            gpu.stream.synchronize().ok();
            let t0 = Instant::now();
            for _ in 0..iters {
                fwd(&mut g, &mut slabs, &mut out);
            }
            gpu.stream.synchronize().ok();
            best_f = best_f.min(t0.elapsed().as_secs_f64() * 1e3 / iters as f64);

            let t1 = Instant::now();
            for _ in 0..iters {
                fwd(&mut g, &mut slabs, &mut out);
                ops::slstm_fused_time_bwd(
                    &gpu, &whr, &dy, &mut g, &mut dh, &mut dc, &mut dn, &slabs, t,
                );
            }
            gpu.stream.synchronize().ok();
            best_fb = best_fb.min(t1.elapsed().as_secs_f64() * 1e3 / iters as f64);
        }
        println!("{label:<30} {best_f:>9.3} {best_fb:>12.3}");
    }
}
