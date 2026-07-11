//! Head-to-head: old per-timestep `nn` path vs new batched `nn2` path.
//!
//!   cargo run --release --example nn2_bench
//!
//! The thesis under test: batching turns a matrix×vector (arithmetic intensity
//! ~1, memory-bandwidth bound) into a matrix×matrix GEMM (intensity ~B). Both
//! paths do the *same* total FLOPs; we compare wall-clock throughput.
//!
//! Caveat for reading the numbers: the old Linear uses a hand-tuned
//! `matvec_acc`; the new GEMM is still the correct-but-simple version (good loop
//! order, no tiling/SIMD). So this measures "batched-simple-GEMM vs
//! tuned-per-vector-matvec" — the honest question of whether batching alone wins
//! before any kernel tuning.

use std::time::Instant;

use neural_networks::nn::linear::{LinearCache, LinearLayer};
use neural_networks::nn::mlstm::MLSTMLayer;
use neural_networks::nn::slstm::SLSTMLayer;
use neural_networks::nn2;
use neural_networks::nn_layer::NnLayer; // reset_state / clear_grads (trait methods)
use neural_networks::tensor::Tensor;

/// Run `f` `iters` times, return seconds elapsed. `warmup` runs are untimed.
fn time(warmup: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    t0.elapsed().as_secs_f64()
}

fn gflops(flops: f64, secs: f64) -> f64 {
    flops / secs / 1e9
}

fn bench_linear(inp: usize, out: usize, batch: usize) {
    println!("\n== Linear  in={inp} out={out} batch(B)={batch} ==");
    let iters = 200;
    // FLOPs for one forward over the whole batch: 2·B·in·out (multiply-add).
    let flops = 2.0 * batch as f64 * inp as f64 * out as f64;

    // --- old: B separate matrix-vector products (per-timestep path) ---------
    let old = LinearLayer::new(inp, out);
    let mut cache = LinearCache::new(inp, out);
    let x: Vec<f32> = (0..batch * inp).map(|i| (i as f32 * 0.001).sin()).collect();
    let secs_old = time(20, iters, || {
        for b in 0..batch {
            old.forward(&x[b * inp..(b + 1) * inp], &mut cache);
            std::hint::black_box(&cache.output);
        }
    });

    // --- new: one batched [B,in]·[in,out] GEMM -----------------------------
    let mut new = nn2::Linear::new(inp, out);
    let xt = Tensor::new(&[batch, inp], x.clone());
    let secs_new = time(20, iters, || {
        let y = new.forward(&xt);
        std::hint::black_box(&y);
    });

    report("forward", secs_old, secs_new, iters, flops);

    // --- forward + backward -------------------------------------------------
    let mut oldb = LinearLayer::new(inp, out);
    let mut delta = vec![0.1f32; out];
    let secs_old_b = time(20, iters, || {
        for b in 0..batch {
            oldb.forward(&x[b * inp..(b + 1) * inp], &mut cache);
            delta.iter_mut().for_each(|d| *d = 0.1);
            oldb.backward(&mut delta, &mut cache);
        }
        oldb.clear_grads();
    });
    let dy = Tensor::new(&[batch, out], vec![0.1; batch * out]);
    let secs_new_b = time(20, iters, || {
        let _y = new.forward(&xt);
        let dx = new.backward(&dy);
        std::hint::black_box(&dx);
        new.zero_grad();
    });
    report("fwd+bwd", secs_old_b, secs_new_b, iters, 3.0 * flops);
}

fn bench_slstm(inp: usize, hidden: usize, batch: usize, seq: usize) {
    println!("\n== sLSTM  in={inp} hidden={hidden} B={batch} T={seq}  (forward only) ==");
    let iters = 30;
    // 4 gate matmuls, each 2·B·rows·H, over T steps.
    let rows = inp + hidden;
    let flops = 4.0 * 2.0 * batch as f64 * rows as f64 * hidden as f64 * seq as f64;

    // --- old: B sequences × T steps of single-vector forward ---------------
    let mut old = SLSTMLayer::new(inp, hidden);
    let mut cache = old.alloc_cache();
    let x: Vec<f32> = (0..batch * seq * inp).map(|i| (i as f32 * 0.001).sin()).collect();
    let secs_old = time(3, iters, || {
        for b in 0..batch {
            old.reset_state();
            for t in 0..seq {
                let off = (b * seq + t) * inp;
                old.forward(&x[off..off + inp], &mut cache);
                std::hint::black_box(&cache.h);
            }
        }
    });

    // --- new: one batched [B,T,in] sequence forward ------------------------
    let mut new = nn2::SLstm::new(inp, hidden);
    let xt = Tensor::new(&[batch, seq, inp], x.clone());
    let secs_new = time(3, iters, || {
        let y = new.forward(&xt);
        std::hint::black_box(&y);
    });

    report("forward", secs_old, secs_new, iters, flops);

    // --- forward + backward (BPTT) -----------------------------------------
    // Old path: T caches per sequence (backward walks them in reverse).
    let mut oldb = SLSTMLayer::new(inp, hidden);
    let mut caches: Vec<_> = (0..seq).map(|_| oldb.alloc_cache()).collect();
    let mut delta = vec![0.1f32; hidden];
    let secs_old_b = time(3, iters, || {
        for b in 0..batch {
            oldb.reset_state();
            oldb.reset_bptt_state();
            for t in 0..seq {
                let off = (b * seq + t) * inp;
                oldb.forward(&x[off..off + inp], &mut caches[t]);
            }
            for t in (0..seq).rev() {
                delta.iter_mut().for_each(|d| *d = 0.1);
                oldb.backward(&mut delta, &mut caches[t]);
            }
        }
        oldb.clear_grads();
    });
    let dy = Tensor::new(&[batch, seq, hidden], vec![0.1; batch * seq * hidden]);
    let secs_new_b = time(3, iters, || {
        let _y = new.forward(&xt);
        let dx = new.backward(&dy);
        std::hint::black_box(&dx);
        new.zero_grad();
    });
    // fwd+bwd ≈ 3× the forward matmul work.
    report("fwd+bwd", secs_old_b, secs_new_b, iters, 3.0 * flops);
}

fn bench_mlstm(inp: usize, d: usize, heads: usize, dqk: usize, batch: usize, seq: usize) {
    println!("\n== mLSTM  in={inp} d={d} H={heads} dqk={dqk} B={batch} T={seq}  (fwd+bwd) ==");
    let iters = 10;
    // Rough proxy: projection + output GEMMs only (ignores the per-head recurrence),
    // so GFLOP/s understates work — read the ms and speedup as the real comparison.
    let d_qk = heads * dqk;
    let proj = 2.0 * batch as f64 * inp as f64 * (2.0 * d_qk as f64 + 2.0 * d as f64 + 2.0 * heads as f64);
    let wout = 2.0 * batch as f64 * d as f64 * d as f64;
    let flops = 3.0 * (proj + wout) * seq as f64;

    let x: Vec<f32> = (0..batch * seq * inp).map(|i| (i as f32 * 0.001).sin()).collect();

    // Old: B sequences × T steps, T caches per sequence for BPTT.
    let mut old = MLSTMLayer::new(inp, d, heads, dqk);
    let mut caches: Vec<_> = (0..seq).map(|_| old.alloc_cache()).collect();
    let mut delta = vec![0.1f32; d];
    let secs_old = time(2, iters, || {
        for bb in 0..batch {
            old.reset_state();
            old.reset_bptt_state();
            for s in 0..seq {
                let off = (bb * seq + s) * inp;
                old.forward(&x[off..off + inp], &mut caches[s]);
            }
            for s in (0..seq).rev() {
                delta.iter_mut().for_each(|dd| *dd = 0.1);
                old.backward(&mut delta, &mut caches[s]);
            }
        }
        old.clear_grads();
    });

    // New: one batched [B,T,in] sequence.
    let mut new = nn2::MLstm::new(inp, d, heads, dqk);
    let xt = Tensor::new(&[batch, seq, inp], x.clone());
    let dy = Tensor::new(&[batch, seq, d], vec![0.1; batch * seq * d]);
    let secs_new = time(2, iters, || {
        let _y = new.forward(&xt);
        let dxg = new.backward(&dy);
        std::hint::black_box(&dxg);
        new.zero_grad();
    });

    report("fwd+bwd", secs_old, secs_new, iters, flops);
}

fn report(label: &str, secs_old: f64, secs_new: f64, iters: usize, flops: f64) {
    let ms_old = secs_old / iters as f64 * 1e3;
    let ms_new = secs_new / iters as f64 * 1e3;
    let g_old = gflops(flops, secs_old / iters as f64);
    let g_new = gflops(flops, secs_new / iters as f64);
    println!(
        "  {label:8}  old {ms_old:8.3} ms ({g_old:6.2} GFLOP/s)   new {ms_new:8.3} ms ({g_new:6.2} GFLOP/s)   speedup {:.2}x",
        secs_old / secs_new
    );
}

fn main() {
    println!("nn (per-timestep) vs nn2 (batched) — lower ms / higher GFLOP/s is better");
    // Backbone-ish projection.
    bench_linear(512, 512, 256);
    // Logit head: wide output, tall batch.
    bench_linear(128, 260, 512);
    // Recurrent cell at a realistic char-model width.
    bench_slstm(128, 128, 64, 64);
    bench_slstm(256, 256, 64, 32);
    // Multi-head mLSTM at a char-model-ish width.
    bench_mlstm(128, 128, 4, 32, 32, 32);
    bench_mlstm(256, 256, 4, 64, 32, 16);
}
