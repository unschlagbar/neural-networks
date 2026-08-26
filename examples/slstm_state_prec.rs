//! What does storing the sLSTM's stabilizer group in bf16 actually cost?
//!
//! `c`, `n`, `i_prime`, `f_prime` carry the `exp(-m)` scale that keeps the recurrence
//! bounded, and `gpu::bf16` argues they must stay fp32 on the grounds that an absolute
//! error in an exponent is a multiplicative error in the value it guards. FlashRNN
//! (NX-AI) stores exactly this group in bf16 by default. This measures which reading
//! survives contact with our shapes.
//!
//! The storage width is a process-wide `OnceLock` — a forward and its backward must
//! agree on it — so the two arms cannot coexist in one process. Run twice and diff:
//!
//!   cargo run --release --features cuda --example slstm_state_prec -- /tmp/fp32.bin
//!   SLSTM_BF16_STATE=1 cargo run --release --features cuda --example slstm_state_prec -- /tmp/bf16.bin
//!   python3 -c "..."   # or the printed one-liner
//!
//! Every tensor is written as raw little-endian f32, in the order the header lists.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::io::Write;

    use neural_networks::gpu::slstm::SLstm;
    use neural_networks::gpu::{GTensor, Gpu};
    use neural_networks::tensor::Tensor;

    let out = std::env::args().nth(1).unwrap_or_else(|| "/tmp/slstm.bin".into());
    let gpu = Gpu::new().expect("gpu");
    println!(
        "slab_bf16 = {}   state_bf16 = {}",
        gpu.kernels.slab_bf16, gpu.kernels.state_bf16
    );

    // (label, B, T, H). The backbone is the compounding worst case: batch 1, and the
    // longest unbroken chain the model ever runs (one BACKBONE_CHUNK). The encoder
    // shape is the opposite corner — wide batch, a handful of timesteps.
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("backbone B=1 T=512 H=1024", 1, 512, 1024),
        ("encoder  B=512 T=8 H=256", 512, 8, 256),
    ];

    let mut f = std::fs::File::create(&out).expect("create dump");
    let mut dump = |t: &[f32]| {
        let mut bytes = Vec::with_capacity(t.len() * 4);
        for v in t {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        f.write_all(&bytes).expect("write dump");
    };

    for &(label, b, t, h) in shapes {
        // Deterministic init at the scale the cell actually trains at (1/sqrt(H)); a
        // fixed larger scale saturates the gates and the comparison degenerates.
        let s = 1.0 / (h as f32).sqrt();
        // SEEDED, always: the two arms are separate processes (the width is a
        // process-wide OnceLock), so unseeded data means comparing two different
        // models and every relative error comes out at sqrt(2) — which is exactly
        // what the first version of this probe reported.
        let w: Vec<Tensor> = (0..4)
            .map(|g| Tensor::random_seeded(&[2 * h, h], s * (1.0 + g as f32 * 0.05), 100 + g as u64))
            .collect();
        let bi: Vec<Tensor> = (0..4)
            .map(|g| Tensor::random_seeded(&[h], 0.2 + g as f32 * 0.01, 200 + g as u64))
            .collect();
        let mut cell = SLstm::from_parts(
            &gpu, h, h, &w[0], &w[1], &w[2], &w[3], &bi[0], &bi[1], &bi[2], &bi[3], None,
        );

        // ACC forward/backward pairs accumulating into the SAME gradient buffers, which
        // is what a real step does: BATCH_SIZE windows x one call per chunk (backbone)
        // or per length group (encoder/decoder) — depth 16-32, not 1. A low-precision
        // accumulator loses small contributions, and that failure mode only shows up
        // with depth, so a single-shot comparison understates it.
        let acc: usize = std::env::var("ACC")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        let mut y = cell.forward_alloc(&gpu, &GTensor::from_host(&gpu, &Tensor::random_seeded(&[b, t, h], 0.5, 7)));
        let mut extremes = cell.state_extremes(&gpu);
        let mut dx = {
            let gy = GTensor::from_host(&gpu, &Tensor::random_seeded(&[b, t, h], 0.7, 9));
            cell.backward_alloc(&gpu, &y, &gy)
        };
        for i in 1..acc {
            // Each pair gets its own draw, so the contributions differ in magnitude the
            // way successive windows do.
            let x = GTensor::from_host(
                &gpu,
                &Tensor::random_seeded(&[b, t, h], 0.5, 7 + i as u64 * 13),
            );
            y = cell.forward_alloc(&gpu, &x);
            extremes = cell.state_extremes(&gpu);
            let gy = GTensor::from_host(
                &gpu,
                &Tensor::random_seeded(&[b, t, h], 0.7, 9 + i as u64 * 13),
            );
            dx = cell.backward_alloc(&gpu, &y, &gy);
        }

        let yh = y.to_host(&gpu).data;
        let dxh = dx.to_host(&gpu).data;
        // Every gradient this cell owns, in `param_slots` order: dWx, dWhr, dbcat,
        // then the post-norm's dgamma if it had one (it does not here).
        let grads: Vec<Vec<f32>> = cell
            .param_slots()
            .iter()
            .map(|s| s.grad.to_host(&gpu).data.to_vec())
            .collect();
        println!(
            "{label} (acc {acc}): |y|max {:.4}  |dx|max {:.4}  extremes(min|n|, max|c|, max|c/n|) = {:?}",
            yh.iter().fold(0.0f32, |m, v| m.max(v.abs())),
            dxh.iter().fold(0.0f32, |m, v| m.max(v.abs())),
            extremes
        );
        dump(&yh);
        dump(&dxh);
        for g in &grads {
            dump(g);
        }
    }
    println!("\nwrote {out}");
}
