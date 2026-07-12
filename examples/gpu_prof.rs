//! Phase profile of one hierarchical GPU training window.
//!
//!   cargo run --release --features cuda --example gpu_prof
//!
//! Times the three stages at their real training shapes (encoder/decoder: words
//! are the batch axis, T is short; backbone: batch 1, T = words) so we can see
//! which one owns the step time before optimizing anything.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda`");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;

    use neural_networks::gpu::block::{Block, BlockLike};
    use neural_networks::gpu::mlstm::MLstm;
    use neural_networks::gpu::slstm::SLstm;
    use neural_networks::gpu::{DTensor, Gpu, ops};

    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU: {e}");
            return;
        }
    };

    /// GPU timer: queue `iters` launches, synchronize once at the end.
    fn gpu_time(gpu: &Gpu, warmup: usize, iters: usize, mut f: impl FnMut()) -> f64 {
        for _ in 0..warmup { f(); }
        gpu.stream.synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..iters { f(); }
        gpu.stream.synchronize().unwrap();
        t0.elapsed().as_secs_f64()
    }

    let up = |h: usize| h * 8 / 3;

    // The real training step: one full hierarchical forward+backward window at
    // config.rs shapes (2048 words, ~5 chars each).
    {
        use neural_networks::gpu::hierarchical::{HierCfg, Hierarchical};
        let words_n = 2048usize;
        let cfg = HierCfg {
            vocab: 100, hc: 256, wh: 512, enc_blocks: 2, bb_blocks: 16, dec_blocks: 2,
            heads: 8, dqk: 64, w_token: 99, cap: 30.0,
        };
        let mut model = Hierarchical::new(&gpu, &cfg);
        let mut tokens = Vec::new();
        let mut words = Vec::new();
        for w in 0..words_n {
            let start = tokens.len();
            let len = 3 + (w % 5);
            for k in 0..len { tokens.push(1 + (w + k) % 90); }
            words.push((start, tokens.len()));
        }
        let _ = model.forward_backward(&gpu, &tokens, &words);
        let t0 = Instant::now();
        let n_iter = 3;
        for _ in 0..n_iter { let _ = model.forward_backward(&gpu, &tokens, &words); }
        println!(">>> hierarchical window ({words_n} words, {} tokens): {:.1?} / window\n",
                 tokens.len(), t0.elapsed() / n_iter);
    }

    // Time a forward+backward over `iters` runs, synchronizing once at the end.
    let run = |name: &str, blocks: &mut Vec<Box<dyn BlockLike>>, b: usize, t: usize, h: usize| {
        let x = DTensor::zeros(&gpu, &[b, t, h]);
        let g = DTensor::zeros(&gpu, &[b, t, h]);
        // warmup
        for blk in blocks.iter_mut() {
            let _ = blk.forward(&gpu, &x);
        }
        for blk in blocks.iter_mut().rev() {
            let _ = blk.backward(&gpu, &g);
        }
        gpu.stream.synchronize().unwrap();

        let t0 = Instant::now();
        let mut h_ = x.dup(&gpu);
        for blk in blocks.iter_mut() {
            h_ = blk.forward(&gpu, &h_);
        }
        gpu.stream.synchronize().unwrap();
        let fwd = t0.elapsed();

        let t1 = Instant::now();
        let mut d = g.dup(&gpu);
        for blk in blocks.iter_mut().rev() {
            d = blk.backward(&gpu, &d);
        }
        gpu.stream.synchronize().unwrap();
        let bwd = t1.elapsed();

        println!(
            "{name:<28} B={b:<5} T={t:<5} H={h:<4} fwd {:>8.1?}  bwd {:>8.1?}  total {:>8.1?}",
            fwd,
            bwd,
            fwd + bwd
        );
    };

    // Shapes from config.rs: hc=256, wh=512, 2 enc + 16 bb + 2 dec, 2048 words.
    let (hc, wh, words) = (256usize, 512usize, 2047usize);

    // Isolate the two per-timestep GEMM shapes of the backbone cell (B=1, H=512).
    {
        let h = 512usize;
        let h4 = 4 * h;
        let hs = DTensor::zeros(&gpu, &[1, h]);
        let whr = DTensor::zeros(&gpu, &[h, h4]);
        let whr_t = DTensor::zeros(&gpu, &[h4, h]);
        let mut gh = DTensor::zeros(&gpu, &[1, h4]);
        let mut dh = DTensor::zeros(&gpu, &[1, h]);
        let iters = 2047;
        let t = gpu_time(&gpu, 10, iters, || ops::matmul_nn_into(&gpu, &hs, &whr, &mut gh, 0.0));
        println!("fwd gemm  [1,{h}]x[{h},{h4}]        {:>8.1?} for {iters}", std::time::Duration::from_secs_f64(t));
        let t = gpu_time(&gpu, 10, iters, || ops::matmul_nn_into(&gpu, &gh, &whr_t, &mut dh, 0.0));
        println!("bwd gemm  [1,{h4}]x[{h4},{h}]  (nn) {:>8.1?} for {iters}", std::time::Duration::from_secs_f64(t));
        let t = gpu_time(&gpu, 10, iters, || ops::matmul_nt_into(&gpu, &gh, &whr, &mut dh, 0.0));
        println!("bwd gemm  [1,{h4}]x[{h},{h4}]T (nt) {:>8.1?} for {iters}", std::time::Duration::from_secs_f64(t));
    }

    // Isolate the two per-timestep kernels (B=1, H=512, T=2047).
    {
        use neural_networks::gpu::ops::SlstmSlabs;
        let (b, t, h) = (1usize, 2047usize, 512usize);
        let h4 = 4 * h;
        let slab = || DTensor::zeros(&gpu, &[b, t, h]);
        let mut slabs = SlstmSlabs { c_prev: slab(), n_prev: slab(), zt: slab(), ot: slab(),
            i_prime: slab(), f_prime: slab(), c: slab(), n: slab(), psi: slab(), h_prev: slab() };
        let mut g = DTensor::zeros(&gpu, &[b, t, h4]);
        let mut gh = DTensor::zeros(&gpu, &[b, h4]);
        let bcat = DTensor::zeros(&gpu, &[h4]);
        let (mut cs, mut ns, mut ms, mut hs2) = (DTensor::zeros(&gpu, &[b, h]), DTensor::zeros(&gpu, &[b, h]), DTensor::zeros(&gpu, &[b, h]), DTensor::zeros(&gpu, &[b, h]));
        let (mut dc, mut dn) = (DTensor::zeros(&gpu, &[b, h]), DTensor::zeros(&gpu, &[b, h]));
        let dhb = DTensor::zeros(&gpu, &[b, h]);
        let mut out = DTensor::zeros(&gpu, &[b, t, h]);
        let dy = DTensor::zeros(&gpu, &[b, t, h]);
        let mut step = 0usize;
        let tf = gpu_time(&gpu, 10, t, || {
            ops::slstm_step_fused(&gpu, &mut g, &gh, &bcat, &mut cs, &mut ns, &mut ms, &mut hs2, &mut slabs, &mut out, step % t);
            step += 1;
        });
        let mut step = 0usize;
        let tb = gpu_time(&gpu, 10, t, || {
            ops::slstm_step_fused_bwd(&gpu, &dy, &mut g, &mut gh, &dhb, &slabs, &mut dc, &mut dn, step % t);
            step += 1;
        });
        println!("kernel fwd x{t}                    {:>8.1?}", std::time::Duration::from_secs_f64(tf));
        println!("kernel bwd x{t}                    {:>8.1?}", std::time::Duration::from_secs_f64(tb));
    }

    // Bare cell (no block wrapper) at backbone shape, to separate cell cost from
    // the surrounding norms/SwiGLU.
    {
        let mut cell = SLstm::new_rand(&gpu, wh, wh);
        let x = DTensor::zeros(&gpu, &[1, words, wh]);
        let g = DTensor::zeros(&gpu, &[1, words, wh]);
        let _ = cell.forward(&gpu, &x);
        let _ = cell.backward(&gpu, &g);
        gpu.stream.synchronize().unwrap();
        let t0 = Instant::now();
        let _ = cell.forward(&gpu, &x);
        gpu.stream.synchronize().unwrap();
        let f = t0.elapsed();
        let t1 = Instant::now();
        let _ = cell.backward(&gpu, &g);
        gpu.stream.synchronize().unwrap();
        println!("bare sLSTM cell x1          B=1     T={words}  H={wh}  fwd {:>8.1?}  bwd {:>8.1?}", f, t1.elapsed());
    }

    let mut enc: Vec<Box<dyn BlockLike>> = (0..2)
        .map(|_| {
            Box::new(Block::from_cell(&gpu, hc, up(hc), SLstm::new_rand(&gpu, hc, hc)))
                as Box<dyn BlockLike>
        })
        .collect();
    run("encoder (2x sLSTM)", &mut enc, words, 8, hc);

    let mut dec: Vec<Box<dyn BlockLike>> = (0..2)
        .map(|_| {
            Box::new(Block::from_cell(&gpu, hc, up(hc), SLstm::new_rand(&gpu, hc, hc)))
                as Box<dyn BlockLike>
        })
        .collect();
    run("decoder (2x sLSTM)", &mut dec, words, 8, hc);

    let mut bb_s: Vec<Box<dyn BlockLike>> = (0..8)
        .map(|_| {
            Box::new(Block::from_cell(&gpu, wh, up(wh), SLstm::new_rand(&gpu, wh, wh)))
                as Box<dyn BlockLike>
        })
        .collect();
    run("backbone sLSTM half (8x)", &mut bb_s, 1, words, wh);

    if std::env::var("SKIP_M").is_ok() { return; }
    let mut bb_m: Vec<Box<dyn BlockLike>> = (0..8)
        .map(|_| {
            Box::new(Block::from_cell(
                &gpu,
                wh,
                up(wh),
                MLstm::new_rand(&gpu, wh, wh, 8, wh / 8),
            )) as Box<dyn BlockLike>
        })
        .collect();
    run("backbone mLSTM half (8x)", &mut bb_m, 1, words, wh);
}
