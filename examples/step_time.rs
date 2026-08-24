//! Uninstrumented step time for a real window — the A/B instrument.
//!
//! `phase_table` syncs around every span, which inflates small kernels; this only
//! syncs once per measured block. Prints the median of per-step times so one thermal
//! excursion cannot move the answer, plus the SM clock, which varies enough on this
//! box to swamp a small win.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use neural_networks::config::{WORD_BLOCKS, WORD_HIDDEN};
    use neural_networks::gpu::Gpu;
    use neural_networks::gpu::hierarchical::{Hierarchical, ModelCfg};
    use neural_networks::nn2::optim::AdamCfg;

    let path = std::env::args().nth(1).unwrap_or("src/gpu/ops.rs".into());
    let steps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);

    let gpu = Gpu::new().expect("gpu");
    let cfg = ModelCfg {
        vocab: 260,
        hc: 256,
        wh: WORD_HIDDEN,
        enc_blocks: 2,
        bb_blocks: WORD_BLOCKS,
        dec_blocks: 2,
        heads: 16,
        dqk: WORD_HIDDEN / 16,
        w_token: 256,
        cap: 30.0,
    };

    let text = std::fs::read_to_string(&path).expect("read source");
    let ids = neural_networks::tokenizer_utf8::Utf8Tokenizer::new().to_tokens(&text);
    let mut words: Vec<std::range::Range<usize>> = Vec::new();
    let mut start = 0usize;
    for e in neural_networks::segment::word_ends(&ids) {
        words.push((start..e as usize).into());
        start = e as usize;
        if words.len() == 1024 {
            break;
        }
    }
    let tokens: Vec<usize> = ids[..start].iter().map(|&t| t as usize).collect();

    let mut model = Hierarchical::new(&gpu, cfg);
    let mut opt = AdamCfg::new(3e-4, 0.01);

    // Warm up allocations and lazy kernel specialization before anything is timed.
    for it in 0..5 {
        model.forward_backward(&gpu, &tokens, &words);
        opt.t = it + 1;
        model.step(&gpu, &opt);
    }

    let mut ms: Vec<f64> = Vec::with_capacity(steps);
    for it in 0..steps {
        gpu.stream.synchronize().unwrap();
        let t0 = std::time::Instant::now();
        model.forward_backward(&gpu, &tokens, &words);
        opt.t = 6 + it as u64;
        model.step(&gpu, &opt);
        gpu.stream.synchronize().unwrap();
        ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let clock = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=clocks.sm", "--format=csv,noheader"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    println!(
        "{} words | median {:.1} ms | min {:.1} | max {:.1} | n={} | sm {}",
        words.len(),
        ms[ms.len() / 2],
        ms[0],
        ms[ms.len() - 1],
        steps,
        clock
    );
}
