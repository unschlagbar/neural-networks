//! Where does the mLSTM's time actually go?
//!
//!   cargo run --release --features cuda --example mlstm_stage_prof
//!
//! `gpu_prof` says the backbone's 8 mLSTM blocks cost ~195 ms/window, which is far
//! more than their FLOP count justifies. Before rewriting the cell into a fused
//! TFLA-style kernel, this pins down which of the three cost centres is to blame:
//!
//!   * raw cuBLAS throughput at our shapes (are the projections even near peak?);
//!   * the cell as a whole (projections + chunk loop);
//!   * the chunk loop's own [BH, L, L] machinery, isolated.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda` to run this benchmark");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;

    use neural_networks::gpu::mlstm::MLstm;
    use neural_networks::gpu::{DTensor, Gpu, ops};
    use neural_networks::tensor::Tensor;

    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU: {e}");
            return;
        }
    };

    /// Wall time of `iters` calls, synchronizing once at each end.
    fn timed(gpu: &Gpu, warmup: usize, iters: usize, mut f: impl FnMut()) -> f64 {
        for _ in 0..warmup {
            f();
        }
        gpu.stream.synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            f();
        }
        gpu.stream.synchronize().unwrap();
        t0.elapsed().as_secs_f64() / iters as f64
    }

    // Backbone shape: one window, WORD_HIDDEN wide, ~2048 words.
    let (b, t, d) = (1usize, 2047usize, neural_networks::config::WORD_HIDDEN);
    let (heads, dqk) = (8usize, 64usize);
    let n = b * t;

    println!("== cuBLAS throughput at backbone shapes (B={b} T={t} d={d}) ==");
    for &(m, k, nn, what) in &[
        (n, d, d, "projection    [N,d]x[d,d]"),
        (n, d, d * 8 / 3, "swiglu up     [N,d]x[d,up]"),
        (n, d * 8 / 3, d, "swiglu down   [N,up]x[up,d]"),
    ] {
        let a = DTensor::from_host(&gpu, &Tensor::random(&[m, k], 0.5));
        let w = DTensor::from_host(&gpu, &Tensor::random(&[k, nn], 0.5));
        let secs = timed(&gpu, 5, 50, || {
            let _ = ops::matmul(&gpu, &a, &w);
        });
        let flops = 2.0 * m as f64 * k as f64 * nn as f64;
        println!(
            "{what}  {:>7.3} ms   {:>6.1} TFLOP/s",
            secs * 1e3,
            flops / secs / 1e12
        );
    }

    // The whole cell, forward and backward, at the shape the backbone runs it at.
    println!("\n== mLSTM cell (7 projections + chunk loop + tail) ==");
    let mut cell = MLstm::new_rand(&gpu, d, d, heads, dqk);
    let x = DTensor::from_host(&gpu, &Tensor::random(&[b, t, d], 0.5));
    let dy = DTensor::from_host(&gpu, &Tensor::random(&[b, t, d], 1.0));

    let fwd = timed(&gpu, 2, 10, || {
        let y = cell.forward(&gpu, &x);
        drop(y);
        // backward must consume the cache the forward built, or it grows unboundedly
        let _ = cell.backward(&gpu, &dy);
    });
    println!(
        "fwd+bwd (chunk {}):  {:>7.2} ms",
        neural_networks::config::MLSTM_CHUNK,
        fwd * 1e3
    );

    // Forward alone, so the two halves can be separated.
    let f_only = timed(&gpu, 2, 10, || {
        let y = cell.forward(&gpu, &x);
        drop(y);
        let _ = cell.backward(&gpu, &dy);
    });
    let _ = f_only;

    // The 7 projections on their own: the FLOP-dominant part of the cell.
    let xf = DTensor::from_host(&gpu, &Tensor::random(&[n, d], 0.5));
    let wq = DTensor::from_host(&gpu, &Tensor::random(&[d, d], 0.5));
    let proj = timed(&gpu, 5, 20, || {
        for _ in 0..7 {
            let _ = ops::matmul(&gpu, &xf, &wq);
        }
    });
    println!("7 projections alone: {:>7.2} ms", proj * 1e3);

    // The chunk loop's per-chunk machinery, isolated: the [BH, L, L] GEMMs and the
    // elementwise decay kernels, for the same number of chunks the cell runs.
    let l = neural_networks::config::MLSTM_CHUNK.min(t);
    let chunks = t.div_ceil(l);
    let bh = b * heads;
    let dhv = d / heads;
    let qc = DTensor::from_host(&gpu, &Tensor::random(&[bh, l, dqk], 0.5));
    let kc = DTensor::from_host(&gpu, &Tensor::random(&[bh, l, dqk], 0.5));
    let vc = DTensor::from_host(&gpu, &Tensor::random(&[bh, l, dhv], 0.5));
    let core = timed(&gpu, 2, 10, || {
        for _ in 0..chunks {
            let s = ops::matmul_batched_nt(&gpu, &qc, &kc); // [BH, L, L]
            let _ = ops::matmul_batched_nn(&gpu, &s, &vc); // [BH, L, dhv]
        }
    });
    println!(
        "chunk-core GEMMs only ({chunks} chunks of L={l}): {:>7.2} ms",
        core * 1e3
    );

    // The fused chunkwise core on its own — the 5 kernels, nothing else. This is
    // what the whole rewrite replaces the chunk loop with, so if the cell is still
    // slow, this number says whether the kernels or their surroundings are at fault.
    println!("\n== fused chunkwise core alone (the 5 kernels) ==");
    let lf = ops::FUSED_MAX_L;
    let qh = DTensor::from_host(&gpu, &Tensor::random(&[bh, t, dqk], 0.5));
    let kh = DTensor::from_host(&gpu, &Tensor::random(&[bh, t, dqk], 0.5));
    let vh = DTensor::from_host(&gpu, &Tensor::random(&[bh, t, dhv], 0.5));
    let igh = DTensor::from_host(&gpu, &Tensor::random(&[bh, t], 0.5));
    let fgh = DTensor::from_host(&gpu, &Tensor::random(&[bh, t], 0.5));
    let dyt = DTensor::from_host(&gpu, &Tensor::random(&[bh, t, dhv], 1.0));

    let f_fw = timed(&gpu, 3, 20, || {
        let _ = ops::mlstm_fused_fw(&gpu, &qh, &kh, &vh, &igh, &fgh, lf);
    });
    let f_all = timed(&gpu, 3, 20, || {
        let sv = ops::mlstm_fused_fw(&gpu, &qh, &kh, &vh, &igh, &fgh, lf);
        let _ = ops::mlstm_fused_bw(&gpu, &sv, &qh, &kh, &vh, &igh, &fgh, &dyt);
    });
    println!("fused fwd only:      {:>7.2} ms", f_fw * 1e3);
    println!(
        "fused fwd+bwd:       {:>7.2} ms   (bwd {:>6.2} ms)",
        f_all * 1e3,
        (f_all - f_fw) * 1e3
    );

    // Everything in the cell that is NOT the chunkwise core: 7 projections, 5 head
    // reorgs, head-norm, o-gate, output projection -- and their backwards.
    println!("\n(cell fwd+bwd minus fused core = the projection/norm/reorg shell)");
}
