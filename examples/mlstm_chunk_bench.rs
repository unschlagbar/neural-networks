//! mLSTM cell scaling in sequence length T — the measurement that decides whether
//! chunking pays off.
//!
//!   cargo run --release --features cuda --example mlstm_chunk_bench
//!
//! The single-chunk parallel form is O(T²) in both time and memory (the [BH,T,T]
//! decay matrices); the chunkwise form is O(T·L). The backbone is where this bites:
//! it autoregresses over WORDS, so T is the window's word count (up to ~2048) at
//! width WORD_HIDDEN. Shape below mirrors that, not the small LM-toy shape in
//! `gpu_bench`.
//!
//! Reports forward+backward ms/iter and the device memory the cell holds live at
//! peak (measured between forward and backward, when the decay matrices and the
//! rest of the activation cache are all resident).
//!
//! `MLSTM_CHUNK=<L>` overrides the chunk length (0 = single-chunk). Set it to
//! compare the two paths at the same shape.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda` to run this benchmark");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;

    use neural_networks::gpu::{DTensor, Gpu, mlstm::MLstm};
    use neural_networks::tensor::Tensor;

    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU: {e}");
            return;
        }
    };

    let mem_used = || {
        let (free, total) = cudarc::driver::result::mem_get_info().expect("mem_get_info");
        (total - free) as f64 / 1e6
    };

    // Backbone shape: one window (B=1), width WORD_HIDDEN, 16 heads.
    let (b, d, heads) = (1, neural_networks::config::WORD_HIDDEN, 16);
    let dqk = d / heads;

    println!(
        "== mLSTM cell fwd+bwd, B={b} d={d} heads={heads} dqk={dqk} (backbone shape) ==\n\
         chunk = {}",
        std::env::var("MLSTM_CHUNK").unwrap_or_else(|_| "default".into()),
    );
    println!("{:>7} {:>7} {:>12} {:>14} {:>12}", "T", "iters", "ms/iter", "live MB @peak", "MB/step");

    for &t in &[128, 256, 512, 1024, 2048] {
        let mut dev = MLstm::new_rand(&gpu, d, d, heads, dqk);
        let x = DTensor::from_host(&gpu, &Tensor::random(&[b, t, d], 0.5));
        let g = DTensor::from_host(&gpu, &Tensor::random(&[b, t, d], 1.0));

        // Iteration count scales down with T so the big shapes don't take minutes;
        // this part boosts to 3.1 GHz, so keep the timed region long (see gpu_bench).
        let iters = if t <= 512 { 50 } else { 10 };
        let warmup = (iters / 5).max(2);

        // Peak: memory live between forward and backward (decay matrices resident).
        let before = mem_used();
        let y = dev.forward(&gpu, &x);
        gpu.stream.synchronize().unwrap();
        let peak = mem_used();
        drop(y);
        let _ = dev.backward(&gpu, &g);
        gpu.stream.synchronize().unwrap();

        for _ in 0..warmup {
            let _ = dev.forward(&gpu, &x);
            let _ = dev.backward(&gpu, &g);
        }
        gpu.stream.synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = dev.forward(&gpu, &x);
            let _ = dev.backward(&gpu, &g);
        }
        gpu.stream.synchronize().unwrap();
        let secs = t0.elapsed().as_secs_f64();

        println!(
            "{:>7} {:>7} {:>12.2} {:>14.0} {:>12.0}",
            t,
            iters,
            secs / iters as f64 * 1e3,
            peak,
            peak - before,
        );
    }
}
