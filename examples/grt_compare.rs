//! Dense backbone against the recurrent-depth (GRT) one, at the production config.
//!
//! The two arms of this comparison are supposed to be matched on COMPUTE — the same
//! number of backbone block executions per forward — while the GRT arm stores far
//! fewer weights. Neither `config.rs` alone nor a parameter count alone shows whether
//! that actually holds, so this prints both sides together: unique blocks, executions
//! per forward, parameter counts by stage, peak device memory, and the median step
//! time on a real window.
//!
//! Run it before starting a run, not after: if the step times are far apart, the two
//! loss curves are not comparable and the layout in `config.rs` needs adjusting
//! (raise or lower `GRT_R`, or move a block between the core and the prelude).
//!
//! ```text
//! cargo run --release --features cuda --example grt_compare -- [file] [steps]
//! ```

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use neural_networks::config::{CHAR_HIDDEN, LOGIT_SOFTCAP, WORD_BLOCKS, WORD_HIDDEN};
    use neural_networks::gpu::Gpu;
    use neural_networks::gpu::hierarchical::{Hierarchical, ModelCfg};
    use neural_networks::gpu::train::grt_cfg_from_config;
    use neural_networks::nn2::optim::AdamCfg;

    let path = std::env::args().nth(1).unwrap_or("src/gpu/ops.rs".into());
    let steps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    let gpu = Gpu::new().expect("gpu");
    let tokenizer = neural_networks::tokenizer_utf8::Utf8Tokenizer::new();
    let heads = 8;
    let base = ModelCfg {
        vocab: tokenizer.vocab_size(),
        hc: CHAR_HIDDEN,
        wh: WORD_HIDDEN,
        enc_blocks: 4,
        bb_blocks: WORD_BLOCKS,
        dec_blocks: 4,
        heads,
        dqk: WORD_HIDDEN / heads,
        w_token: tokenizer.w_token() as usize,
        cap: LOGIT_SOFTCAP,
        grt: None,
    };

    // A real window, so the word-length histogram is the one training sees.
    let text = std::fs::read_to_string(&path).expect("read source");
    let ids = tokenizer.to_tokens(&text);
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
    println!("window: {} words, {} tokens\n", words.len(), tokens.len());

    let mut rows: Vec<(String, usize, usize, usize, f64, f64)> = Vec::new();
    for (name, cfg) in [
        ("dense", base),
        (
            "grt",
            ModelCfg {
                grt: Some(grt_cfg_from_config()),
                ..base
            },
        ),
    ] {
        let mut model = Hierarchical::new(&gpu, cfg);
        // The deepest recurrence: an isoFLOPs comparison is against the depth the
        // layout is named for, not against a sampled one.
        model.set_recurrence(cfg.grt.map(|g| g.r));
        let (enc, bb, dec, other) = model.param_counts();
        let mut opt = AdamCfg::new(3e-4, 0.01);

        // Warm up allocations and the lazy per-shape kernel specialization first —
        // the first steps of a fresh model are several times the steady-state cost.
        for it in 0..5 {
            model.forward_backward(&gpu, &tokens, &words);
            opt.t = it + 1;
            model.step(&gpu, &opt);
        }
        let (free, total) = cudarc::driver::result::mem_get_info().expect("mem_get_info");
        let mem_mb = (total - free) as f64 / 1e6;

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
        let (unique, execs) = match cfg.grt {
            Some(g) => (g.unique_blocks(), g.executions()),
            None => (cfg.bb_blocks, cfg.bb_blocks),
        };
        rows.push((
            name.into(),
            unique,
            execs,
            enc + bb + dec + other,
            bb as f64,
            ms[ms.len() / 2],
        ));
        println!(
            "{name:<6} unique {unique:>2} blocks, {execs:>2} executions/forward   \
             params {:>10}  backbone {:>10}   step {:>7.1} ms   device {mem_mb:>6.0} MB",
            neural_networks::format::compact((enc + bb + dec + other) as u64),
            neural_networks::format::compact(bb as u64),
            ms[ms.len() / 2],
        );
    }

    if rows.len() == 2 {
        let (d, g) = (&rows[0], &rows[1]);
        println!(
            "\ngrt / dense:  executions {:.2}x   backbone params {:.2}x   step {:.2}x",
            g.2 as f64 / d.2 as f64,
            g.4 / d.4,
            g.5 / d.5,
        );
        println!(
            "An isoFLOPs comparison wants executions at 1.00x. Step time will not match \
             it exactly:\nthe modulation adds W_proj and the gate MLP per recurrence \
             step, and the two\nbackbones do not run the same sLSTM/mLSTM mix."
        );
    }

    let clock = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=clocks.sm", "--format=csv,noheader"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    if !clock.is_empty() {
        println!("SM clock {clock}");
    }
}
