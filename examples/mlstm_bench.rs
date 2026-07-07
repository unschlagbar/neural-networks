// mlstm_bench.rs — deterministic single-layer mLSTM timing + parity checksums.
//
// Times forward and backward of one MLSTMLayer at the hierarchical-backbone
// dims (384 wide, 8 heads, dqk 48) over a BPTT window, with all weights and
// inputs filled from a fixed hash so runs are reproducible. The printed
// checksums must stay bit-identical across pure refactors of the layer —
// any change there means the math was reordered, not just sped up.
//
//   cargo run --release --example mlstm_bench

use std::time::Instant;

use neural_networks::nn::mlstm::MLSTMLayer;
use neural_networks::nn_layer::NnLayer;
use neural_networks::optimizers::{GradMatrixOps, GradVecOps};

const INPUT: usize = 384;
const HIDDEN: usize = 384;
const HEADS: usize = 8;
const DQK: usize = 48;
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
    let mut layer = MLSTMLayer::new(INPUT, HIDDEN, HEADS, DQK);

    fill(layer.wq.as_slice_mut(), 1, 0.08);
    fill(layer.wk.as_slice_mut(), 2, 0.08);
    fill(layer.wv.as_slice_mut(), 3, 0.08);
    fill(layer.wo.as_slice_mut(), 4, 0.08);
    fill(layer.wi.as_slice_mut(), 5, 0.05);
    fill(layer.wf.as_slice_mut(), 6, 0.05);
    fill(&mut layer.bq, 7, 0.02);
    fill(&mut layer.bk, 8, 0.02);
    fill(&mut layer.bv, 9, 0.02);
    fill(&mut layer.bo, 10, 0.02);
    for (hd, b) in layer.bi.iter_mut().enumerate() {
        *b = -4.0 + det(100 + hd, 1.0);
    }
    for (hd, b) in layer.bf.iter_mut().enumerate() {
        *b = 4.5 + det(200 + hd, 1.0);
    }
    fill(layer.w_out.weights.as_slice_mut(), 11, 0.06);
    fill(&mut layer.w_out.biases, 12, 0.02);

    // Per-timestep inputs and output deltas, fixed across iterations.
    let inputs: Vec<Vec<f32>> = (0..SEQ_LEN)
        .map(|t| {
            let mut v = vec![0.0; INPUT];
            fill(&mut v, 1000 + t, 0.8);
            v
        })
        .collect();
    let deltas: Vec<Vec<f32>> = (0..SEQ_LEN)
        .map(|t| {
            let mut v = vec![0.0; HIDDEN];
            fill(&mut v, 2000 + t, 0.05);
            v
        })
        .collect();

    let mut caches: Vec<_> = (0..SEQ_LEN).map(|_| layer.alloc_cache()).collect();
    let mut delta_buf = vec![0.0; HIDDEN];

    // Warmup + parity pass.
    layer.reset_state();
    layer.reset_bptt_state();
    for t in 0..SEQ_LEN {
        layer.forward(&inputs[t], &mut caches[t]);
    }
    let out_sum: f64 = caches
        .iter()
        .map(|c| c.w_out.output.iter().map(|&x| x as f64).sum::<f64>())
        .sum();
    for t in (0..SEQ_LEN).rev() {
        delta_buf.copy_from_slice(&deltas[t]);
        layer.backward(&mut delta_buf, &mut caches[t]);
    }
    layer.accumulate_init_grad(); // window-end flush of deferred weight grads
    let dx_sum: f64 = caches
        .iter()
        .map(|c| c.dx.iter().map(|&x| x as f64).sum::<f64>())
        .sum();
    let gsum = |s: &[f32]| s.iter().map(|&x| x as f64).sum::<f64>();
    let gwq_sum = gsum(layer.grads.wq.matrix().as_slice());
    let gwk_sum = gsum(layer.grads.wk.matrix().as_slice());
    let gwv_sum = gsum(layer.grads.wv.matrix().as_slice());
    let gwo_sum = gsum(layer.grads.wo.matrix().as_slice());
    let gwf_sum = gsum(layer.grads.wf.matrix().as_slice());
    let gbq_sum = gsum(layer.grads.bq.vec());

    println!("parity checksums (must not change across refactors):");
    println!("  out  {out_sum:+.9e}");
    println!("  dx   {dx_sum:+.9e}");
    println!("  gwq  {gwq_sum:+.9e}  gwk {gwk_sum:+.9e}");
    println!("  gwv  {gwv_sum:+.9e}  gwo {gwo_sum:+.9e}");
    println!("  gwf  {gwf_sum:+.9e}  gbq {gbq_sum:+.9e}");

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
    println!(
        "  fwd+bwd   {:>7} ns/step",
        (fwd_ns + bwd_ns) / steps
    );
}
