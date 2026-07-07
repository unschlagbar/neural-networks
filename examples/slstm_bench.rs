// slstm_bench.rs — deterministic single-layer sLSTM timing + parity checksums.
//
// Times forward and backward of one SLSTMLayer at the hierarchical-stack
// width (384) over a BPTT window, with all weights and inputs filled from a
// fixed hash so runs are reproducible. The printed checksums must stay
// bit-identical across pure refactors of the layer — any change there means
// the math was reordered, not just sped up.
//
//   cargo run --release --example slstm_bench

use std::time::Instant;

use neural_networks::nn::slstm::SLSTMLayer;
use neural_networks::nn_layer::NnLayer;
use neural_networks::optimizers::{GradMatrixOps, GradVecOps};

// Width via env: SLSTM_H=192 for the encoder/decoder dims, default 384 (backbone).
const SEQ_LEN: usize = 64;
const ITERS: usize = 60;

/// Deterministic pseudo-random f32 in (-scale, scale) from an index.
fn det(i: usize, scale: f32) -> f32 {
    let mut x = i as u64 ^ 0x9E37_79B9_7F4A_7C15;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x as f32 / u64::MAX as f32 * 2.0 - 1.0) * scale
}

fn fill(slice: &mut [f32], seed: usize, scale: f32) {
    for (i, v) in slice.iter_mut().enumerate() {
        *v = det(seed.wrapping_mul(0x1000_0000) + i, scale);
    }
}

fn main() {
    let width: usize = std::env::var("SLSTM_H")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(384);
    println!("width {width} (input == hidden), seq {SEQ_LEN}");
    let mut layer = SLSTMLayer::new(width, width);

    fill(layer.wz.as_slice_mut(), 1, 0.08);
    fill(layer.wi.as_slice_mut(), 2, 0.05);
    fill(layer.wf.as_slice_mut(), 3, 0.05);
    fill(layer.wo.as_slice_mut(), 4, 0.08);
    fill(&mut layer.bz, 5, 0.02);
    fill(&mut layer.bi, 6, 0.02);
    for (j, b) in layer.bf.iter_mut().enumerate() {
        *b = 4.5 + det(200 + j, 1.0);
    }
    fill(&mut layer.bo, 7, 0.02);

    // Per-timestep inputs and output deltas, fixed across iterations.
    let inputs: Vec<Vec<f32>> = (0..SEQ_LEN)
        .map(|t| {
            let mut v = vec![0.0; width];
            fill(&mut v, 1000 + t, 0.8);
            v
        })
        .collect();
    let deltas: Vec<Vec<f32>> = (0..SEQ_LEN)
        .map(|t| {
            let mut v = vec![0.0; width];
            fill(&mut v, 2000 + t, 0.05);
            v
        })
        .collect();

    let mut caches: Vec<_> = (0..SEQ_LEN).map(|_| layer.alloc_cache()).collect();
    let mut delta_buf = vec![0.0; width];

    // Warmup + parity pass.
    layer.reset_state();
    layer.reset_bptt_state();
    for t in 0..SEQ_LEN {
        layer.forward(&inputs[t], &mut caches[t]);
    }
    let out_sum: f64 = caches
        .iter()
        .map(|c| c.h.iter().map(|&x| x as f64).sum::<f64>())
        .sum();
    for t in (0..SEQ_LEN).rev() {
        delta_buf.copy_from_slice(&deltas[t]);
        layer.backward(&mut delta_buf, &mut caches[t]);
    }
    let dx_sum: f64 = caches
        .iter()
        .map(|c| c.dconcat[..width].iter().map(|&x| x as f64).sum::<f64>())
        .sum();
    let gsum = |s: &[f32]| s.iter().map(|&x| x as f64).sum::<f64>();
    let gwz_sum = gsum(layer.grads.wz.matrix().as_slice());
    let gwi_sum = gsum(layer.grads.wi.matrix().as_slice());
    let gwf_sum = gsum(layer.grads.wf.matrix().as_slice());
    let gwo_sum = gsum(layer.grads.wo.matrix().as_slice());
    let gbz_sum = gsum(layer.grads.bz.vec());

    println!("parity checksums (must not change across refactors):");
    println!("  out  {out_sum:+.9e}");
    println!("  dx   {dx_sum:+.9e}");
    println!("  gwz  {gwz_sum:+.9e}  gwi {gwi_sum:+.9e}");
    println!("  gwf  {gwf_sum:+.9e}  gwo {gwo_sum:+.9e}");
    println!("  gbz  {gbz_sum:+.9e}");

    // Timed iterations.
    let mut fwd_ns = 0u128;
    let mut bwd_ns = 0u128;
    for _ in 0..ITERS {
        layer.clear_grads();
        layer.reset_state();
        layer.reset_bptt_state();

        let t0 = Instant::now();
        for t in 0..SEQ_LEN {
            layer.forward(&inputs[t], &mut caches[t]);
        }
        fwd_ns += t0.elapsed().as_nanos();

        let t1 = Instant::now();
        for t in (0..SEQ_LEN).rev() {
            delta_buf.copy_from_slice(&deltas[t]);
            layer.backward(&mut delta_buf, &mut caches[t]);
        }
        layer.accumulate_init_grad(); // window-end flush belongs to backward
        bwd_ns += t1.elapsed().as_nanos();
    }

    let steps = (ITERS * SEQ_LEN) as u128;
    println!("\ntiming over {ITERS} iters x {SEQ_LEN} steps:");
    println!(
        "  forward   {:>7} ns/step   ({:.3} ms total)",
        fwd_ns / steps,
        fwd_ns as f64 / 1e6
    );
    println!(
        "  backward  {:>7} ns/step   ({:.3} ms total)",
        bwd_ns / steps,
        bwd_ns as f64 / 1e6
    );
    println!("  fwd+bwd   {:>7} ns/step", (fwd_ns + bwd_ns) / steps);
}
