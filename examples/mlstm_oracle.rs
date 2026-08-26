//! One mLSTM cell against the CPU scalar recurrence at the **backbone's** shape.
//!
//!   cargo run --release --features cuda --example mlstm_oracle [t] [chunks] [dqk] [heads]
//!
//! `mlstm_matches_cpu` and friends run the same comparison at dqk=8, d=16, t=12. The
//! fused kernels are constexpr-specialised on the shape (tile counts and block width
//! are compile-time constants derived from dqk/dhv), so those tests exercise different
//! generated code than the backbone's dqk=96 — a slice or tiling error that only
//! appears once dqk exceeds a tile width is invisible to them.
//!
//! `chunks` > 1 runs the GPU cell the way the backbone sweep does: split the sequence,
//! carry (C, n, m) across the borders, unwind the backward right to left. The CPU cell
//! zeroes its state per call and therefore only ever runs the whole sequence — which is
//! exactly the reference the chunked run has to reproduce.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use neural_networks::gpu::arena::TrainingCache;
    use neural_networks::gpu::{GTensor, Gpu, mlstm::MLstm};
    use neural_networks::nn2::MLstm as CpuMLstm;
    use neural_networks::tensor::Tensor;

    let mut args = std::env::args().skip(1);
    let mut next = |d: usize| args.next().and_then(|s| s.parse().ok()).unwrap_or(d);
    let t = next(512);
    let chunks = next(4);
    let dqk = next(96);
    let heads = next(8);
    let (b, d) = (1usize, heads * dqk);
    let inp = d;
    assert!(t % chunks == 0, "t must divide into chunks");


    let gpu = Gpu::new().expect("gpu");
    let cache = TrainingCache::new(&gpu, 1 << 23, 1 << 18, 1 << 23);
    println!("t={t} chunks={chunks} b={b} inp={inp} d={d} heads={heads} dqk={dqk}");

    // Fan-in-scaled like `MLstm::new`, but seeded, so a second source tree runs the
    // identical problem instance. A flat `random(0.3)` would not do: at inp=768 it puts
    // the gate logits at std ~4, where the stabilizer swings over decades and the
    // recurrence is chaotic rather than representative of anything a trained model runs
    // at.
    //
    // `new` leaves wi/wf at zero and bi at -10, which pins the gates near-constant and
    // leaves the stabilizer untested. `gate` (arg 5, tenths) reopens them: the gate
    // logits get std ~ gate/10, so 0 reproduces the init regime and 10 a trained one.
    let gate = next(10) as f32 / 10.0;
    let mut cpu = CpuMLstm::new(inp, d, heads, dqk);
    let sq = |fi: usize, fo: usize| (6.0 / (fi as f32 + fo as f32)).sqrt();
    let mut s = 0x5EEDu64;
    let mut seeded = |dims: &[usize], sc: f32| {
        s = s.wrapping_add(1);
        Tensor::random_seeded(dims, sc, s)
    };
    cpu.wq = seeded(&[inp, d], sq(inp, d));
    cpu.wk = seeded(&[inp, d], sq(inp, d));
    cpu.wv = seeded(&[inp, d], sq(inp, d));
    cpu.wo = seeded(&[inp, d], sq(inp, d));
    cpu.w_out = seeded(&[d, d], sq(d, d));
    cpu.wi = seeded(&[inp, heads], gate / (inp as f32).sqrt());
    cpu.wf = seeded(&[inp, heads], gate / (inp as f32).sqrt());
    cpu.bi = seeded(&[heads], gate);
    cpu.bf = Tensor::new(
        &[heads],
        (0..heads).map(|h| 4.0 + h as f32 / heads as f32).collect(),
    );

    let x = Tensor::random_seeded(&[b, t, inp], 0.5, 0xA1);
    let g = Tensor::random_seeded(&[b, t, d], 1.0, 0xA2);

    // Second reference: the same cell run chunk by chunk with the state RESET at every
    // border (the CPU cell zeroes its state per call, so this is free). A chunked GPU
    // run that matches this one rather than the carried whole-sequence run is not
    // "imprecise" — it is discarding the cross-border state outright, i.e. training on
    // `t/chunks` of context instead of `t`.
    let mut reset_ref: Vec<f32> = Vec::new();
    {
        let step = t / chunks;
        for c in 0..chunks {
            let lo = c * step * inp;
            let sl = Tensor::new(&[b, step, inp], x.data[lo..lo + step * inp].to_vec());
            reset_ref.extend_from_slice(&cpu.forward(&sl).data);
        }
    }

    cpu.zero_grad();
    let y_cpu = cpu.forward(&x);
    let dx_cpu = cpu.backward(&g);
    // wk/bk carry a folded 1/√dqk on the GPU, so their gradients come back scaled.
    let inv = 1.0 / (dqk as f32).sqrt();
    let want: Vec<(&str, Vec<f32>, f32)> = vec![
        ("y", y_cpu.data.clone(), 1.0),
        ("dx", dx_cpu.data.clone(), 1.0),
        ("dWq", cpu.dwq.data.clone(), 1.0),
        ("dbq", cpu.dbq.data.clone(), 1.0),
        ("dWk", cpu.dwk.data.clone(), inv),
        ("dbk", cpu.dbk.data.clone(), inv),
        ("dWv", cpu.dwv.data.clone(), 1.0),
        ("dbv", cpu.dbv.data.clone(), 1.0),
        ("dWo", cpu.dwo.data.clone(), 1.0),
        ("dbo", cpu.dbo.data.clone(), 1.0),
        ("dWi", cpu.dwi.data.clone(), 1.0),
        ("dbi", cpu.dbi.data.clone(), 1.0),
        ("dWf", cpu.dwf.data.clone(), 1.0),
        ("dbf", cpu.dbf.data.clone(), 1.0),
        ("dWout", cpu.dw_out.data.clone(), 1.0),
        ("dbout", cpu.db_out.data.clone(), 1.0),
        ("dgamma", cpu.dgamma.data.clone(), 1.0),
    ];

    // Worst |a-b| against the reference's own magnitude: a pointwise relative error
    // explodes on entries near zero, an absolute one says nothing about scale.
    let err = |got: &[f32], w: &[f32]| -> f64 {
        assert_eq!(got.len(), w.len(), "length mismatch");
        let scale = w.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
        got.iter()
            .zip(w)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max) as f64
            / scale as f64
    };

    let mut run = |nchunk: usize| -> Vec<Vec<f32>> {
        let mut dev = MLstm::from_cpu(&gpu, &cpu);
        dev.zero_grad(&gpu);
        let step = t / nchunk;
        dev.set_carry(nchunk > 1 || std::env::var("FORCE_CARRY").is_ok());
        dev.reset_state(&gpu);
        let cut = |src: &Tensor, c: usize, w: usize| {
            let lo = c * step * w;
            GTensor::from_host(
                &gpu,
                &Tensor::new(&[b, step, w], src.data[lo..lo + step * w].to_vec()),
            )
        };
        let mut ys = Vec::new();
        for c in 0..nchunk {
            let mut y = GTensor::uninit(&gpu, &[b, step, d]);
            dev.forward(&gpu, &cut(&x, c, inp), &mut y, &cache);
            ys.push(y.to_host(&gpu).data.to_vec());
        }
        dev.reset_bptt(&gpu);
        let mut dxs = vec![Vec::new(); nchunk];
        for c in (0..nchunk).rev() {
            dxs[c] = dev
                .backward_alloc(&gpu, &cut(&g, c, d), &cache)
                .to_host(&gpu)
                .data
                .to_vec();
        }
        if std::env::var("PERCHUNK").is_ok() {
            eprintln!(
                "  (per-chunk y first element: {:?})",
                ys.iter().map(|c| c[0]).collect::<Vec<_>>()
            );
        }
        let mut out = vec![ys.concat(), dxs.concat()];
        for gr in dev.grads() {
            out.push(gr.to_host(&gpu).data.to_vec());
        }
        out
    };

    let whole = run(1);
    let split = run(chunks);

    // Raw values, not just an aggregate: rounding noise is symmetric about zero and
    // tracks the reference 1:1, while different algebra shows up as a bias or as a few
    // elements that are simply elsewhere. `DUMP=<tensor>` prints the elements.
    if let Ok(want_name) = std::env::var("DUMP") {
        for (i, (name, w, sc)) in want.iter().enumerate() {
            if *name != want_name {
                continue;
            }
            println!("\n{name}: cpu / gpu whole / gpu x{chunks}");
            let n = w.len().min(16);
            for j in 0..n {
                let c = w[j] / sc;
                println!(
                    "  [{j:>3}] {c:>14.7e} {:>14.7e} {:>14.7e}   d={:>10.2e}",
                    whole[i][j],
                    split[i][j],
                    whole[i][j] - c
                );
            }
        }
    }

    // Signed mean of the error against the mean magnitude. Rounding cancels, so this
    // stays far below 1; a systematically different quantity does not.
    let bias = |got: &[f32], w: &[f32]| -> f64 {
        let n = w.len() as f64;
        let m: f64 = got.iter().zip(w).map(|(a, b)| (a - b) as f64).sum::<f64>() / n;
        let mag: f64 = w.iter().map(|v| v.abs() as f64).sum::<f64>() / n;
        m / mag.max(1e-30)
    };
    // Pearson r against the reference: pure precision keeps this at 1 - O(eps^2).
    let corr = |got: &[f32], w: &[f32]| -> f64 {
        let n = w.len() as f64;
        let (mx, my) = (
            w.iter().map(|v| *v as f64).sum::<f64>() / n,
            got.iter().map(|v| *v as f64).sum::<f64>() / n,
        );
        let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
        for (a, b) in w.iter().zip(got) {
            let (dx, dy) = (*a as f64 - mx, *b as f64 - my);
            sxy += dx * dy;
            sxx += dx * dx;
            syy += dy * dy;
        }
        sxy / (sxx * syy).sqrt().max(1e-300)
    };
    println!(
        "{:>8}  {:>11} {:>11}  {:>10} {:>10}  {:>12}",
        "tensor",
        "relerr wh",
        format!("relerr x{chunks}"),
        "bias wh",
        format!("bias x{chunks}"),
        "1-corr wh"
    );
    let mut worst = (0.0f64, "");
    for (i, (name, w, sc)) in want.iter().enumerate() {
        let scaled: Vec<f32> = w.iter().map(|v| v / sc).collect();
        let (a, c) = (err(&whole[i], &scaled), err(&split[i], &scaled));
        println!(
            "{name:>8}  {a:>11.3e} {c:>11.3e}  {:>10.2e} {:>10.2e}  {:>12.3e}",
            bias(&whole[i], &scaled),
            bias(&split[i], &scaled),
            1.0 - corr(&whole[i], &scaled)
        );
        for e in [a, c] {
            if e > worst.0 {
                worst = (e, name);
            }
        }
    }
    println!("\nworst {:.3e} on {}", worst.0, worst.1);
    let scaled_y: Vec<f32> = want[0].1.clone();
    println!(
        "\ny, gpu x{chunks} vs cpu carried  : {:.3e}\ny, gpu x{chunks} vs cpu RESET-at-border: {:.3e}",
        err(&split[0], &scaled_y),
        err(&split[0], &reset_ref)
    );
}
