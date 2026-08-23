//! Where the device memory actually goes, phase by phase.
//!
//! `mem_get_info` — what `GPU_MEM=1` prints — reports memory the *driver* has handed
//! to the process, which includes every block the CUDA async allocator is caching for
//! reuse. That number cannot distinguish "the model needs this" from "the allocator is
//! sitting on it", and estimating from tensor shapes has repeatedly disagreed with it.
//!
//! So ask the allocator directly. `CU_MEMPOOL_ATTR_USED_MEM_CURRENT` is what is
//! genuinely live; `RESERVED_MEM_CURRENT` is what the pool holds. The gap between them
//! is cache, which is reclaimable under pressure and is *not* a reason to OOM.
//!
//! Run: cargo run --release --features cuda --example vram_audit [words]

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
    use neural_networks::nn2::optim::AdamCfg;
    use neural_networks::tokenizer_utf8::Utf8Tokenizer;

    let words: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(WORDS_PER_SEQ);

    let gpu = Gpu::new().expect("gpu");

    // The device's default async pool — what `cuMemAllocAsync` (and hence every
    // `GTensor<f32>`) draws from.
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
        // Must match `gpu::train::cfg_from_config` — that is what `hg` actually
        // builds. A probe with fewer/narrower blocks measures a different model and
        // will happily report a footprint that fits when the real one does not.
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

    // Mixed word lengths, as a real corpus produces — but respecting the SAME token
    // ceiling the loader enforces (`MAX_WINDOW_TOKENS`). Cycling 1..=MAX_WORD_BYTES
    // averages 8.5 bytes/word, which for 2048 words is 17408 tokens: 2.1x more than
    // training can ever hand the model, and the audit then reports a footprint no real
    // window produces. `word_ends` averages 3-4 bytes/word on Rust source anyway.
    let budget = std::env::var("AUDIT_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(neural_networks::config::MAX_WINDOW_TOKENS);
    // Word-length histogram. `word_ends` averages 3-4 bytes/word on real text, so a
    // real window hits BOTH caps at once: 2048 words AND ~8192 tokens. Cycling
    // 1..=MAX_WORD_BYTES averages 8.5 and runs out of token budget at ~967 words —
    // half the word count, and the encoder/decoder cost scales with WORDS, not just
    // tokens. `AUDIT_WLEN=n` pins every word to n bytes; the default mimics real text.
    let wlen_mode = std::env::var("AUDIT_WLEN")
        .ok()
        .and_then(|s| s.parse().ok());
    let mut tokens = Vec::with_capacity(words * 4);
    let mut spans: Vec<std::range::Range<usize>> = Vec::with_capacity(words);
    for w in 0..words {
        let s = tokens.len();
        // Default: the distribution `word_ends` actually produces, measured off a real
        // `hg` run — 14-16 distinct lengths per window spanning 1..=16, averaging ~5.
        // The spread matters as much as the mean: the encoder and decoder run ONE
        // RECTANGLE PER LENGTH BUCKET, so a window with 16 buckets allocates 16 sets of
        // per-group buffers where a uniform window allocates one.
        //
        // Skewed short (most words are 2-6 bytes, a few run to 16) rather than uniform,
        // which is what the measured histogram looks like.
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

    // Real training never repeats one shape: windows never cross a document border,
    // so their word counts and length histograms vary constantly, and every `Buf`/
    // `Pool` reuses by capacity. A fixed-shape probe therefore measures the steady
    // state of ONE shape and misses the ratchet entirely — which is the failure mode
    // that actually aborts a run. Sweep a spread of shapes and report the worst.
    if std::env::var("AUDIT_SWEEP").is_ok() {
        println!("\n--- varying window shapes (the real loader's behaviour) ---");
        let reps: usize = std::env::var("AUDIT_REPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(40);
        // Each iteration builds a FRESH window with its own word count and its own
        // length histogram. Slicing one fixed window (the previous version of this
        // sweep) reuses the same rectangles every time and so never exercises the
        // ratchet: real training never sees the same shape twice, and it is the
        // arrival of a shape no buffer yet fits that grows the footprint.
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
                // Re-seeded per iteration, so the length histogram itself differs
                // window to window — not just the word count.
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
                let ps = model.pool_shapes();
                let sizes: usize = ps.iter().map(|(_, s, _)| s).sum();
                let bufs: usize = ps.iter().map(|(_, _, b)| b).sum();
                println!(
                    "  it {it:3}: {:5} words, {:6} tokens -> live {now:8.0} MB  \
                     (peak {peak:8.0})  pool: {sizes} sizes / {bufs} bufs",
                    sp.len(),
                    tk.len()
                );
            }
        }
        println!(
            "  worst live across the sweep: {peak:.0} MB  (at {} words / {} tokens)",
            worst_at.0, worst_at.1
        );

        // Which owner grew? Compare against the same breakdown taken after the very
        // first window: whatever is bigger here ratcheted, and by how much.
        let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
        println!("\n--- after the sweep, by owner (MB) ---");
        println!(
            "{:<10} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
            "stage", "ffn", "pool", "norms", "proj", "cell_saved", "cell_other"
        );
        for (label, c) in model.act_breakdown() {
            println!(
                "{label:<10} {:>8.0} {:>8.0} {:>8.0} {:>8.0} {:>10.0} {:>10.0}",
                mb(c[0]),
                mb(c[1]),
                mb(c[2]),
                mb(c[3]),
                mb(c[4]),
                mb(c[5])
            );
        }
    }

    // What is still live once a whole step has finished? Nothing should be: every
    // activation is dead, and the optimizer has consumed the gradients. Whatever
    // remains is retained by a layer's own buffers, and THAT is what scales with the
    // window and decides whether a longer sequence fits.
    println!("\n--- what survives a completed step ---");
    let after_step = used();
    println!("live after step        {after_step:8.0} MB");
    println!("  of which weights     {weights:8.0} MB");
    println!("  retained by layers   {:8.0} MB", after_step - weights);

    // Walk the actual buffers. `release_stage` only reaches a block's own `Buf`s and
    // pool, so anything it leaves behind is invisible to a release-and-diff — which
    // is exactly the memory that went unaccounted for. This counts it directly.
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

    // Which owner inside each stage holds it. Only `ffn` and `pool` are reachable
    // from drop_saved_act + trim_to; `norms`, `proj` and `cell_other` are held inside
    // the sub-layers and survive both.
    println!("\n--- activations by owner (MB) ---");
    println!(
        "{:<10} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
        "stage", "ffn", "pool", "norms", "proj", "cell_saved", "cell_other"
    );
    for (label, c) in model.act_breakdown() {
        println!(
            "{label:<10} {:>8.0} {:>8.0} {:>8.0} {:>8.0} {:>10.0} {:>10.0}",
            mb(c[0]),
            mb(c[1]),
            mb(c[2]),
            mb(c[3]),
            mb(c[4]),
            mb(c[5])
        );
    }

    // Release one stage at a time, so the retained memory is attributed rather than
    // estimated — every shape-based estimate so far has been off by ~10x.
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

    // Now the deep release, which also clears what lives inside the norms and
    // projections. The gap between this and the line above is the answer.
    model.drop_all_act(&gpu);
    gpu.stream.synchronize().unwrap();
    let after_deep = used();
    println!(
        "after drop_all_act     {after_deep:8.0} MB   (freed a further {:8.0} MB)",
        after_release - after_deep
    );
    println!("  STILL retained       {:8.0} MB", after_deep - weights);

    // Attribute the remainder: drop the whole model and see what the pool still holds.
    // Anything left is either not owned by the model or leaked outright.
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
