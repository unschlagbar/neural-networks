//! Where the device memory actually goes, phase by phase.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use cudarc::driver::sys;
    use neural_networks::config::{
        CHAR_HIDDEN, OUT_HIDDEN, WORD_BLOCKS, WORD_HIDDEN, WORDS_PER_SEQ,
    };
    use neural_networks::gpu::Gpu;
    use neural_networks::gpu::hierarchical::{Hierarchical, ModelCfg};
    use neural_networks::cpu::optim::AdamCfg;
    use neural_networks::tokenizer_utf8::Utf8Tokenizer;

    let words: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(WORDS_PER_SEQ);

    let gpu = Gpu::new().expect("gpu");

    let pool = unsafe {
        let dev = cudarc::driver::result::device::get(0).expect("device");
        cudarc::driver::result::device::get_default_mem_pool(dev).expect("pool")
    };
    let attr = |a: sys::CUmemPool_attribute| -> f64 {
        let mut v: u64 = 0;
        unsafe {
            cudarc::driver::result::mem_pool::get_attribute(
                pool,
                a,
                &mut v as *mut u64 as *mut std::ffi::c_void,
            )
            .expect("pool attr");
        }
        v as f64 / (1024.0 * 1024.0)
    };
    let used = || attr(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT);
    let reserved = || attr(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT);
    let driver_mb = || {
        let (free, total) = cudarc::driver::result::mem_get_info().expect("mem_get_info");
        (
            (total - free) as f64 / (1024.0 * 1024.0),
            total as f64 / (1024.0 * 1024.0),
        )
    };

    let (_, total) = driver_mb();
    println!("device total {total:.0} MB\n");

    let tok = Utf8Tokenizer::new();
    let cfg = ModelCfg {
        vocab: tok.vocab_size(),
        hc: CHAR_HIDDEN,
        wh: WORD_HIDDEN,
        enc_blocks: 4,
        bb_blocks: WORD_BLOCKS,
        dec_blocks: 4,
        heads: 8,
        dqk: WORD_HIDDEN / 8,
        w_token: neural_networks::tokenizer_utf8::W_TOKEN as usize,
        cap: 30.0,
    };
    assert_eq!(
        CHAR_HIDDEN, OUT_HIDDEN,
        "decoder ties the encoder char table"
    );

    let row = |label: &str| {
        let (drv, _) = driver_mb();
        println!(
            "{label:<26} used {:8.0}  reserved {:8.0}  driver {:8.0}  cache {:8.0}",
            used(),
            reserved(),
            drv,
            reserved() - used()
        );
    };

    row("before model");
    let mut model = Hierarchical::new(&gpu, cfg);
    let mut opt = AdamCfg::new(3e-4, 0.01);
    gpu.stream.synchronize().unwrap();
    row("after model init");
    let weights = used();

    let budget = std::env::var("AUDIT_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(neural_networks::config::MAX_WINDOW_TOKENS);
    let wlen_mode = std::env::var("AUDIT_WLEN")
        .ok()
        .and_then(|s| s.parse().ok());
    let mut tokens = Vec::with_capacity(words * 4);
    let mut spans: Vec<std::range::Range<usize>> = Vec::with_capacity(words);
    for w in 0..words {
        let s = tokens.len();
        let want = wlen_mode.unwrap_or_else(|| {
            let r = (w * 2654435761) % 100;
            match r {
                0..=24 => 1 + (w % 3),  // 1-3
                25..=59 => 3 + (w % 4), // 3-6
                60..=84 => 6 + (w % 5), // 6-10
                _ => 10 + (w % 7),      // 10-16
            }
        });
        // Stop growing the window once the loader's cap would have cut it.
        if s + want > budget {
            break;
        }
        for k in 0..want {
            tokens.push(1 + (k % 90));
        }
        spans.push((s..tokens.len()).into());
    }
    let words = spans.len();
    println!("\nwindow: {words} words, {} tokens\n", tokens.len());

    for it in 0..3 {
        model.forward_backward(&gpu, &tokens, &spans);
        gpu.stream.synchronize().unwrap();
        let after_fb = used();
        opt.t += 1;
        model.step(&gpu, &opt);
        gpu.stream.synchronize().unwrap();
        println!(
            "iter {it}: after fwd+bwd {after_fb:8.0} MB (over weights {:8.0})  \
             after step {:8.0}",
            after_fb - weights,
            used()
        );
    }

    if std::env::var("AUDIT_SWEEP").is_ok() {
        println!("\n--- varying window shapes (the real loader's behaviour) ---");
        let reps: usize = std::env::var("AUDIT_REPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(40);
        let mut peak = used();
        let mut worst_at = (0usize, 0usize);
        for it in 0..reps {
            let target: usize = match it % 5 {
                0 => 2048,
                1 => 400 + (it * 137) % 900,
                2 => 1700 + (it * 61) % 348,
                3 => 150 + (it * 89) % 400,
                _ => 900 + (it * 211) % 1100,
            };
            let mut tk = Vec::with_capacity(target * 6);
            let mut sp: Vec<std::range::Range<usize>> = Vec::with_capacity(target);
            for w in 0..target {
                let s = tk.len();
                let r = ((w + it * 7919) * 2654435761) % 100;
                let want = match r {
                    0..=24 => 1 + (w % 3),
                    25..=59 => 3 + (w % 4),
                    60..=84 => 6 + (w % 5),
                    _ => 10 + (w % 7),
                };
                if s + want > budget {
                    break;
                }
                for k in 0..want {
                    tk.push(1 + (k % 90));
                }
                sp.push((s..tk.len()).into());
            }
            if sp.len() < 2 {
                continue;
            }
            model.forward_backward(&gpu, &tk, &sp);
            opt.t += 1;
            model.step(&gpu, &opt);
            gpu.stream.synchronize().unwrap();
            let now = used();
            if now > peak {
                peak = now;
                worst_at = (sp.len(), tk.len());
            }
            if it % 4 == 0 || now >= peak {
                let (high, slots) = model.temp_high_water();
                println!(
                    "  it {it:3}: {:5} words, {:6} tokens -> live {now:8.0} MB  \
                     (peak {peak:8.0})  temp slots: {high}/{slots}",
                    sp.len(),
                    tk.len()
                );
            }
        }
        println!(
            "  worst live across the sweep: {peak:.0} MB  (at {} words / {} tokens)",
            worst_at.0, worst_at.1
        );

        let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
        println!("\n--- after the sweep, by owner (MB) ---");
        println!(
            "{:<10} {:>8} {:>8} {:>8} {:>10} {:>10}",
            "stage", "ffn", "norms", "proj", "cell_saved", "cell_other"
        );
        for (label, c) in model.act_breakdown() {
            println!(
                "{label:<10} {:>8.0} {:>8.0} {:>8.0} {:>10.0} {:>10.0}",
                mb(c[0]),
                mb(c[1]),
                mb(c[2]),
                mb(c[3]),
                mb(c[4])
            );
        }
    }

    println!("\n--- what survives a completed step ---");
    let after_step = used();
    println!("live after step        {after_step:8.0} MB");
    println!("  of which weights     {weights:8.0} MB");
    println!("  retained by layers   {:8.0} MB", after_step - weights);

    println!("\n--- retained bytes, by layer (capacity, not shape) ---");
    let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
    let report = model.retained_report();
    println!("{:<18} {:>12} {:>12}", "", "params MB", "activs MB");
    let (mut tp, mut ta) = (0, 0);
    for (label, p, a) in &report {
        println!("{label:<18} {:>12.0} {:>12.0}", mb(*p), mb(*a));
        tp += p;
        ta += a;
    }
    println!("{:<18} {:>12.0} {:>12.0}", "TOTAL", mb(tp), mb(ta));
    println!(
        "\nwalked total {:.0} MB vs pool-reported live {after_step:.0} MB",
        mb(tp + ta)
    );

    println!("\n--- activations by owner (MB) ---");
    println!(
        "{:<10} {:>8} {:>8} {:>8} {:>10} {:>10}",
        "stage", "ffn", "norms", "proj", "cell_saved", "cell_other"
    );
    for (label, c) in model.act_breakdown() {
        println!(
            "{label:<10} {:>8.0} {:>8.0} {:>8.0} {:>10.0} {:>10.0}",
            mb(c[0]),
            mb(c[1]),
            mb(c[2]),
            mb(c[3]),
            mb(c[4])
        );
    }

    let mut prev = after_step;
    for stage in ["encoder", "backbone", "decoder"] {
        model.release_stage(stage);
        gpu.stream.synchronize().unwrap();
        let now = used();
        println!(
            "  release {stage:<9}      freed {:8.0} MB  (live {now:8.0})",
            prev - now
        );
        prev = now;
    }
    model.release_activation_buffers();
    gpu.stream.synchronize().unwrap();
    let after_release = used();
    println!(
        "after releasing all    {after_release:8.0} MB   (freed {:8.0} MB total)",
        after_step - after_release
    );
    println!(
        "  STILL retained       {:8.0} MB  <- not reachable via drop_saved_act/trim",
        after_release - weights
    );

    model.drop_all_act(&gpu);
    gpu.stream.synchronize().unwrap();
    let after_deep = used();
    println!(
        "after drop_all_act     {after_deep:8.0} MB   (freed a further {:8.0} MB)",
        after_release - after_deep
    );
    println!("  STILL retained       {:8.0} MB", after_deep - weights);

    drop(model);
    gpu.stream.synchronize().unwrap();
    println!(
        "\nafter dropping model   {:8.0} MB live  (pool reserved {:8.0})",
        used(),
        reserved()
    );

    println!();
    println!(
        "\nweights+moments {weights:.0} MB | live activations are the 'activations' column above\
         \nthe gap between reserved and used is allocator cache — reclaimable, not a leak"
    );
}
