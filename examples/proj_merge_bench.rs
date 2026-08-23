//! What the mLSTM cell's input projections cost at two, three or six GEMMs.
//!
//! The cell runs `q ‖ k ‖ v` and `ĩ ‖ f̃` as one `Linear` each, with `o` on its own —
//! the reference's fused weight mode (NX-AI xlstm, `xlstm_large/model.py`,
//! `weight_mode="fused"`: `qkv_opreact` + `ifgate_preact`) minus `o`, which must
//! reach `ogate_fwd` as fp32 while q/k/v travel as a bf16 slab. This times that
//! arrangement against the six separate projections it replaced, and against the
//! full four-way merge, at the shapes the model actually runs.
//!
//!   cargo run --release --features cuda --example proj_merge_bench

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;

    use neural_networks::gpu::linear::Linear;
    use neural_networks::gpu::{GTensor, Gpu, ops};

    let gpu = Gpu::new().expect("gpu");

    fn timed(gpu: &Gpu, warm: usize, iters: usize, mut f: impl FnMut()) -> f64 {
        for _ in 0..warm {
            f();
        }
        gpu.stream.synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            f();
        }
        gpu.stream.synchronize().unwrap();
        t0.elapsed().as_secs_f64() * 1e6 / iters as f64
    }

    // (label, rows, in, heads, dqk, dhv)
    let shapes = [
        ("backbone  ", 512usize, 1024usize, 16usize, 64usize, 64usize),
        ("enc/dec   ", 2048, 256, 16, 16, 16),
        ("enc/dec sm", 256, 256, 16, 16, 16),
    ];

    for (name, rows, inp, h, dqk, dhv) in shapes {
        let (wqk, wv, wg) = (h * dqk, h * dhv, h);
        let x = GTensor::from_host(
            &gpu,
            &neural_networks::tensor::Tensor::random(&[rows, inp], 0.05),
        );

        // Six separate projections, exactly as `MLstm::forward` runs them.
        let mut six: Vec<Linear> = [wqk, wqk, wv, wv, wg, wg]
            .iter()
            .map(|&o| Linear::new_rand(&gpu, inp, o))
            .collect();
        let mut outs: Vec<GTensor<f32>> = [wqk, wqk, wv, wv, wg, wg]
            .iter()
            .map(|&o| GTensor::uninit(&gpu, &[rows, o]))
            .collect();

        // Merged: one `qkvo` and one `if`, the reference's fused weight mode.
        let wall = 2 * wqk + 2 * wv;
        let mut lin_qkvo = Linear::new_rand(&gpu, inp, wall);
        let mut lin_if = Linear::new_rand(&gpu, inp, 2 * wg);
        let mut out_qkvo = GTensor::uninit(&gpu, &[rows, wall]);
        let mut out_if = GTensor::uninit(&gpu, &[rows, 2 * wg]);

        // Merged, q/k/v only — the variant that leaves `o` fp32 and separate.
        let wqkv = 2 * wqk + wv;
        let mut lin_qkv = Linear::new_rand(&gpu, inp, wqkv);
        let mut lin_o = Linear::new_rand(&gpu, inp, wv);
        let mut out_qkv = GTensor::uninit(&gpu, &[rows, wqkv]);
        let mut out_o = GTensor::uninit(&gpu, &[rows, wv]);

        // The slab variants are what the cell actually runs: q/k/v (and o, in the
        // four-way merge) come out of `forward_staged_slab` at the kernels' width with
        // the bias fused into the GEMM epilogue. That is a different cuBLASLt call
        // from the fp32-output one below, and it picks its tiles differently — which
        // is the whole question here, so it has to be measured on the real path.
        let mut slab_qkv = ops::SlabBuf::new(&gpu, &[rows, wqkv]);
        let mut slab_qkvo = ops::SlabBuf::new(&gpu, &[rows, wall]);
        let fw_slab3 = timed(&gpu, 20, 200, || {
            ops::with_shared_lhs(&gpu, &x, |xb| {
                lin_qkv.forward_staged_slab(&gpu, &x, xb, &mut slab_qkv);
                lin_o.forward_staged(&gpu, &x, xb, &mut out_o);
                lin_if.forward_staged(&gpu, &x, xb, &mut out_if);
            });
        });
        let fw_slab2 = timed(&gpu, 20, 200, || {
            ops::with_shared_lhs(&gpu, &x, |xb| {
                lin_qkvo.forward_staged_slab(&gpu, &x, xb, &mut slab_qkvo);
                lin_if.forward_staged(&gpu, &x, xb, &mut out_if);
            });
        });
        println!(
            "{name} rows={rows:<5} in={inp:<5}  SLAB fw  qkv+o+if={fw_slab3:7.1}us  qkvo+if={fw_slab2:7.1}us"
        );

        let fw6 = timed(&gpu, 20, 200, || {
            ops::with_shared_lhs(&gpu, &x, |xb| {
                for (l, o) in six.iter_mut().zip(outs.iter_mut()) {
                    l.forward_staged(&gpu, &x, xb, o);
                }
            });
        });
        let fw2 = timed(&gpu, 20, 200, || {
            ops::with_shared_lhs(&gpu, &x, |xb| {
                lin_qkvo.forward_staged(&gpu, &x, xb, &mut out_qkvo);
                lin_if.forward_staged(&gpu, &x, xb, &mut out_if);
            });
        });
        let fw3 = timed(&gpu, 20, 200, || {
            ops::with_shared_lhs(&gpu, &x, |xb| {
                lin_qkv.forward_staged(&gpu, &x, xb, &mut out_qkv);
                lin_o.forward_staged(&gpu, &x, xb, &mut out_o);
                lin_if.forward_staged(&gpu, &x, xb, &mut out_if);
            });
        });

        // Backward: the six accumulate into one `dx` with an add per extra term.
        let mut acc = GTensor::uninit(&gpu, &[rows, inp]);
        let mut part = GTensor::uninit(&gpu, &[rows, inp]);
        let bw6 = timed(&gpu, 20, 200, || {
            ops::with_shared_lhs(&gpu, &x, |xb| {
                for (i, (l, o)) in six.iter_mut().zip(outs.iter()).enumerate() {
                    if i == 0 {
                        l.backward_staged_x(&gpu, &x, xb, o, &mut acc);
                    } else {
                        l.backward_staged_x(&gpu, &x, xb, o, &mut part);
                        ops::add_assign(&gpu, &mut acc, &part);
                    }
                }
            });
        });
        let bw2 = timed(&gpu, 20, 200, || {
            ops::with_shared_lhs(&gpu, &x, |xb| {
                lin_qkvo.backward_staged_x(&gpu, &x, xb, &out_qkvo, &mut acc);
                lin_if.backward_staged_x(&gpu, &x, xb, &out_if, &mut part);
                ops::add_assign(&gpu, &mut acc, &part);
            });
        });
        let bw3 = timed(&gpu, 20, 200, || {
            ops::with_shared_lhs(&gpu, &x, |xb| {
                lin_qkv.backward_staged_x(&gpu, &x, xb, &out_qkv, &mut acc);
                lin_o.backward_staged_x(&gpu, &x, xb, &out_o, &mut part);
                ops::add_assign(&gpu, &mut acc, &part);
                lin_if.backward_staged_x(&gpu, &x, xb, &out_if, &mut part);
                ops::add_assign(&gpu, &mut acc, &part);
            });
        });

        println!(
            "{name} rows={rows:<5} in={inp:<5}  fw 6x={fw6:7.1}us  qkv+o+if={fw3:7.1}us  qkvo+if={fw2:7.1}us   \
             bw 6x={bw6:7.1}us  qkv+o+if={bw3:7.1}us  qkvo+if={bw2:7.1}us"
        );
    }
}
