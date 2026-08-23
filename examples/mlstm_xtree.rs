//! One mLSTM cell at the BACKBONE's shape, with every weight written from a fixed
//! seed, so two source trees run the same problem instance and their gradients can be
//! compared by magnitude:
//!
//!   cargo run --release --features cuda --example mlstm_xtree [t] [chunks]
//!
//! `mlstm_determinism` compares one tree against itself and can use `new_rand`;
//! across trees the weights must be pinned, or the two runs are different problems.
//! `chunks` > 1 splits the sequence and carries the state, which is the path the
//! backbone sweep takes and the one no CPU-oracle test covers at this width.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use neural_networks::gpu::arena::TrainingCache;
    use neural_networks::gpu::{GTensor, Gpu, mlstm::MLstm};
    use neural_networks::tensor::Tensor;

    let mut args = std::env::args().skip(1);
    let t: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(512);
    let chunks: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    assert!(t % chunks == 0, "t must divide into chunks");

    let mut cache = TrainingCache::new();

    let gpu = Gpu::new().expect("gpu");
    let (b, heads, dqk) = (1usize, 8usize, 96usize);
    let d = heads * dqk;

    let xh = Tensor::random_seeded(&[b, t, d], 0.5, 0xA1);
    let gh = Tensor::random_seeded(&[b, t, d], 1.0, 0xA2);
    let mut cell = MLstm::new_rand(&gpu, d, d, heads, dqk);

    // Pin every parameter: `new_rand` redraws per process, so without this the two
    // trees solve different problems and any difference is meaningless.
    for (i, p) in cell.params_mut().into_iter().enumerate() {
        let dims: Vec<usize> = p.dims().to_vec();
        let w = Tensor::random_seeded(&dims, 0.05, 0x5EED + i as u64);
        p.copy_from(&gpu, &GTensor::from_host(&gpu, &w));
    }

    const NAMES: [&str; 16] = [
        "dWq", "dbq", "dWk", "dbk", "dWv", "dbv", "dWo", "dbo", "dWi", "dbi", "dWf",
        "dbf", "dWout", "dbout", "dgamma", "d?",
    ];
    let l2 = |v: &[f32]| -> f64 { v.iter().map(|&a| (a as f64) * (a as f64)).sum::<f64>().sqrt() };

    cell.zero_grad(&gpu);
    let step = t / chunks;
    cell.set_carry(chunks > 1);
    let part = |src: &Tensor, c: usize| -> GTensor<f32> {
        let lo = c * step * d;
        let sl = Tensor::new(&[b, step, d], src.data[lo..lo + step * d].to_vec());
        GTensor::from_host(&gpu, &sl)
    };
    let mut ys = Vec::new();
    let mut dxs = Vec::new();
    for c in 0..chunks {
        let mut y = GTensor::uninit(&gpu, &[b, step, d]);
        cell.forward(&gpu, &part(&xh, c), &mut y, &mut cache);
        ys.push(y);
    }
    for c in (0..chunks).rev() {
        dxs.push(cell.backward_alloc(&gpu, &part(&gh, c)));
    }

    println!("t={t} chunks={chunks} d={d} heads={heads} dqk={dqk}");
    let ynorm: f64 = ys.iter().map(|y| l2(&y.to_host(&gpu).data).powi(2)).sum::<f64>().sqrt();
    let dxnorm: f64 = dxs.iter().map(|v| l2(&v.to_host(&gpu).data).powi(2)).sum::<f64>().sqrt();
    println!("{:>7} {:.9e}", "y", ynorm);
    println!("{:>7} {:.9e}", "dx", dxnorm);
    for (i, gr) in cell.grads().iter().enumerate() {
        println!("{:>7} {:.9e}", NAMES[i.min(NAMES.len() - 1)], l2(&gr.to_host(&gpu).data));
    }
}
