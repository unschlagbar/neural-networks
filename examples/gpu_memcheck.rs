//! Temporary: watch device memory across many forward_backward/step cycles with
//! varying window shapes. Free memory must plateau, not drift downward.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use neural_networks::gpu::Gpu;
    use neural_networks::gpu::hierarchical::{HierCfg, Hierarchical};
    use neural_networks::nn2::optim::AdamCfg;

    let gpu = Gpu::new().expect("gpu");
    let cfg = HierCfg {
        vocab: 100,
        hc: 256,
        wh: 512,
        enc_blocks: 2,
        bb_blocks: 16,
        dec_blocks: 2,
        heads: 8,
        dqk: 64,
        w_token: 99,
        cap: 30.0,
    };
    let mut model = Hierarchical::new(&gpu, &cfg);
    let mut opt = AdamCfg::new(3e-4, 0.01);

    let free_mb = || cudarc::driver::result::mem_get_info().unwrap().0 / (1024 * 1024);
    println!("after model init: {} MB free", free_mb());

    // Vary the window: word count cycles, word lengths cycle. This is what the real
    // dataset does, and it is what made the pooled caches ratchet.
    for it in 0..300 {
        let n_words = 40 + (it % 7) * 30; // 40..220 words
        let wlen = 3 + (it % 5) * 4; // 3..19 chars per word
        let mut tokens = Vec::new();
        let mut words = Vec::new();
        for _ in 0..n_words {
            let s = tokens.len();
            for k in 0..wlen {
                tokens.push(1 + (k % 90));
            }
            words.push((s, tokens.len()));
        }
        let loss = model.forward_backward(&gpu, &tokens, &words);
        opt.t += 1;
        model.step(&gpu, &opt);
        if it % 25 == 0 {
            println!(
                "iter {it:2} | words {n_words:3} wlen {wlen:2} | loss {loss:.3} | {} MB free",
                free_mb()
            );
        }
    }
}
