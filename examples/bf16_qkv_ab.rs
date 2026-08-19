//! Does producing q/k/v bf16 straight out of the projection pay?
//!
//!   cargo run --release --features cuda --example bf16_qkv_ab
//!
//! The bf16 path skips the fp32 projection buffer and the `SlabBuf::from_f32` pass
//! that narrows it, and folds the bias into the GEMM epilogue instead of seeding it
//! with `broadcast_row`.
//!
//! Both cells live in one process and alternate round by round, because the SM clock
//! ramps over the first seconds of load and drifts after: measuring one path then the
//! other attributes the ramp to whichever ran second. Interleaving makes the drift
//! common-mode. `nvidia-smi` is never called inside the timing loop — spawning it
//! stalls the CPU long enough to let the clock settle back down.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda` to run this benchmark");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;

    use neural_networks::config::WORD_HIDDEN;
    use neural_networks::gpu::mlstm::MLstm;
    use neural_networks::gpu::{DTensor, Gpu};
    use neural_networks::tensor::Tensor;

    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU: {e}");
            return;
        }
    };

    // The backbone's actual shape: one window, WORD_HIDDEN wide, dqk = d/heads.
    let (b, t, d) = (1usize, 4096usize, WORD_HIDDEN);
    let (heads, dqk) = (8usize, WORD_HIDDEN / 8);
    println!("shape B={b} T={t} d={d} heads={heads} dqk={dqk}");

    // Two cells, identical weights, one per path.
    let cpu = neural_networks::nn2::MLstm::new(d, d, heads, dqk);
    let mut off = MLstm::from_cpu(&gpu, &cpu);
    off.set_bf16_qkv(&gpu, false);
    let mut on = MLstm::from_cpu(&gpu, &cpu);
    on.set_bf16_qkv(&gpu, true);
    if !on.uses_bf16_qkv() {
        eprintln!("bf16_qkv is unavailable here (head-major, fp32 slabs, or pinned)");
        return;
    }

    let x = DTensor::from_host(&gpu, &Tensor::random(&[b, t, d], 0.5));

    let one = |cell: &mut MLstm, gpu: &Gpu, iters: usize| -> f64 {
        let t0 = Instant::now();
        for _ in 0..iters {
            let y = cell.forward_alloc(gpu, &x);
            std::hint::black_box(&y);
            cell.drop_saved();
        }
        gpu.stream.synchronize().unwrap();
        t0.elapsed().as_secs_f64() * 1e3 / iters as f64
    };

    // Warm up both, and let the clock reach its loaded steady state before timing.
    for _ in 0..20 {
        one(&mut on, &gpu, 1);
        one(&mut off, &gpu, 1);
    }

    let (rounds, iters) = (12, 20);
    let (mut t_on, mut t_off) = (Vec::new(), Vec::new());
    for _ in 0..rounds {
        // Alternate the order too, so neither path always follows the other.
        t_on.push(one(&mut on, &gpu, iters));
        t_off.push(one(&mut off, &gpu, iters));
        t_off.push(one(&mut off, &gpu, iters));
        t_on.push(one(&mut on, &gpu, iters));
    }
    t_on.sort_by(f64::total_cmp);
    t_off.sort_by(f64::total_cmp);
    let (m_on, m_off) = (t_on[t_on.len() / 2], t_off[t_off.len() / 2]);
    println!("bf16 qkv ON   median {m_on:7.3} ms/fwd   min {:7.3}", t_on[0]);
    println!("bf16 qkv OFF  median {m_off:7.3} ms/fwd   min {:7.3}", t_off[0]);
    println!(
        "delta {:+.3} ms ({:+.1}%)",
        m_on - m_off,
        (m_on - m_off) / m_off * 100.0
    );
}
