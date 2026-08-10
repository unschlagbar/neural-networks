//! Where a hierarchical training step's time actually goes, split by component.
//!
//! Reports, separately for forward and backward, the share taken by the sLSTM
//! cells, the mLSTM cells, the SwiGLU FFNs, and the per-block glue (norms, residual
//! adds, buffer copies) — plus everything outside the blocks entirely (embedding,
//! the encoder/decoder plumbing, loss, the optimizer step).
//!
//! Run with `GPU_PHASE=1`, which is what enables the per-phase accumulators. Note
//! that phase timing synchronizes around every span, so the TOTAL here is longer
//! than an uninstrumented step — read the percentages, not the absolute step time.
//! The uninstrumented time is printed alongside for reference.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use neural_networks::config::{WORD_BLOCKS, WORD_HIDDEN};
    use neural_networks::gpu::Gpu;
    use neural_networks::gpu::block::phase::{self, Bucket};
    use neural_networks::gpu::hierarchical::{HierCfg, Hierarchical};
    use neural_networks::nn2::optim::AdamCfg;

    if !phase::enabled() {
        eprintln!("set GPU_PHASE=1 to enable the per-phase accumulators");
        return;
    }
    let gpu = Gpu::new().expect("gpu");

    let cfg = HierCfg {
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
    let mut model = Hierarchical::new(&gpu, &cfg);
    let mut opt = AdamCfg::new(3e-4, 0.01);

    let n_words = 1024;
    let mut tokens = Vec::new();
    let mut words: Vec<std::range::Range<usize>> = Vec::new();
    for w in 0..n_words {
        let start = tokens.len();
        for c in 0..4 {
            tokens.push((w * 7 + c * 13) % 256);
        }
        words.push((start..tokens.len()).into());
    }

    // Warm up (allocations, lazy kernel specialization) with recording off.
    phase::set_recording(false);
    for it in 0..3 {
        model.forward_backward(&gpu, &tokens, &words);
        opt.t = it + 1;
        model.step(&gpu, &opt);
    }

    let iters = 5u64;
    phase::reset();
    phase::set_recording(true);
    gpu.stream.synchronize().unwrap();
    let t0 = std::time::Instant::now();
    for it in 0..iters {
        model.forward_backward(&gpu, &tokens, &words);
        opt.t = 4 + it;
        model.step(&gpu, &opt);
    }
    gpu.stream.synchronize().unwrap();
    let wall = t0.elapsed();
    phase::set_recording(false);

    let per = |ns: u64| ns as f64 / iters as f64 / 1e6; // ms per step
    let total_ms = wall.as_secs_f64() * 1e3 / iters as f64;

    let rows: [(&str, f64, f64); 4] = [
        (
            "sLSTM cell",
            per(phase::get(Bucket::SlstmCellFwd)),
            per(phase::get(Bucket::SlstmCellBwd)),
        ),
        (
            "mLSTM cell",
            per(phase::get(Bucket::MlstmCellFwd)),
            per(phase::get(Bucket::MlstmCellBwd)),
        ),
        (
            "SwiGLU FFN",
            per(phase::get(Bucket::FfnFwd)),
            per(phase::get(Bucket::FfnBwd)),
        ),
        (
            "block glue (norms/resid/copies)",
            per(phase::get(Bucket::GlueFwd)),
            per(phase::get(Bucket::GlueBwd)),
        ),
    ];
    let acc_fwd: f64 = rows.iter().map(|r| r.1).sum();
    let acc_bwd: f64 = rows.iter().map(|r| r.2).sum();
    let outside = total_ms - acc_fwd - acc_bwd;

    println!(
        "\nhierarchical step: WH={}, {} backbone blocks, {} words\n",
        cfg.wh, cfg.bb_blocks, n_words
    );
    println!(
        "{:<34} {:>10} {:>8} {:>10} {:>8}",
        "component", "fwd (ms)", "% step", "bwd (ms)", "% step"
    );
    println!("{}", "-".repeat(74));
    for (name, f, b) in rows {
        println!(
            "{name:<34} {f:>10.1} {:>7.1}% {b:>10.1} {:>7.1}%",
            100.0 * f / total_ms,
            100.0 * b / total_ms
        );
    }
    println!("{}", "-".repeat(74));
    println!(
        "{:<34} {acc_fwd:>10.1} {:>7.1}% {acc_bwd:>10.1} {:>7.1}%",
        "in-block subtotal",
        100.0 * acc_fwd / total_ms,
        100.0 * acc_bwd / total_ms
    );
    println!(
        "{:<34} {outside:>10.1} {:>7.1}%   (embedding, encoder/decoder plumbing,",
        "everything else",
        100.0 * outside / total_ms
    );
    println!(
        "{:<34} {:>10} {:>8}    loss, optimizer step, host gaps)",
        "", "", ""
    );
    println!("{}", "-".repeat(74));
    println!(
        "{:<34} {total_ms:>10.1} {:>7.1}%",
        "TOTAL (instrumented)", 100.0
    );
    // The sLSTM dominates, so break it down: is it the serial T-loop, the
    // whole-sequence GEMMs, or the device-to-device staging copies?
    let sl_rows: [(&str, f64, f64); 3] = [
        (
            "  dtod staging copies",
            per(phase::get(Bucket::SlstmCopyFwd)),
            per(phase::get(Bucket::SlstmCopyBwd)),
        ),
        (
            "  whole-sequence GEMMs",
            per(phase::get(Bucket::SlstmGemmFwd)),
            per(phase::get(Bucket::SlstmGemmBwd)),
        ),
        (
            "  serial T-loop",
            per(phase::get(Bucket::SlstmLoopFwd)),
            per(phase::get(Bucket::SlstmLoopBwd)),
        ),
    ];
    let sl_f: f64 = sl_rows.iter().map(|r| r.1).sum();
    let sl_b: f64 = sl_rows.iter().map(|r| r.2).sum();
    let cell_f = per(phase::get(Bucket::SlstmCellFwd));
    let cell_b = per(phase::get(Bucket::SlstmCellBwd));

    println!("\nsLSTM cell breakdown (share of the cell, and of the whole step):\n");
    println!(
        "{:<34} {:>10} {:>8} {:>10} {:>8}",
        "sub-phase", "fwd (ms)", "% cell", "bwd (ms)", "% cell"
    );
    println!("{}", "-".repeat(74));
    for (name, f, b) in sl_rows {
        println!(
            "{name:<34} {f:>10.1} {:>7.1}% {b:>10.1} {:>7.1}%",
            100.0 * f / cell_f.max(1e-9),
            100.0 * b / cell_b.max(1e-9)
        );
    }
    println!(
        "{:<34} {:>10.1} {:>7.1}% {:>10.1} {:>7.1}%",
        "  unattributed (launch gaps)",
        cell_f - sl_f,
        100.0 * (cell_f - sl_f) / cell_f.max(1e-9),
        cell_b - sl_b,
        100.0 * (cell_b - sl_b) / cell_b.max(1e-9)
    );
    println!("{}", "-".repeat(74));
    println!(
        "{:<34} {cell_f:>10.1} {:>7.1}% {cell_b:>10.1} {:>7.1}%   (% of STEP)",
        "sLSTM cell total",
        100.0 * cell_f / total_ms,
        100.0 * cell_b / total_ms
    );

    println!(
        "\nnote: phase timing syncs around every span, so this total exceeds an\n\
         uninstrumented step. Read the percentages."
    );
}
