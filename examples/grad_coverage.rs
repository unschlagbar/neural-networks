//! Does every word of a window reach the gradients, and does a window start clean?
//!
//!   cargo run --release --features cuda --example grad_coverage [segments] [full]
//!
//! Two independent failure modes, both invisible in a loss curve.
//!
//! **Coverage.** The window is swept phase by phase and stage by stage: encoder and
//! decoder run one rectangle per length bucket, the backbone one chunk at a time.
//! Every one of those loops reuses the same buffers, so a layer that *overwrites*
//! its gradient instead of accumulating, or a saved input a later pass clobbers,
//! silently drops whole spans of the sequence — the loss still falls, and the model
//! only learns the words that landed in the surviving pass. The check is
//! additivity: the window loss is a sum over decoder rows, so restricting it to a
//! contiguous slice of the words (the SFT loss mask gates exactly those rows) must
//! give gradients that sum back to the whole window's. Reported per shape:
//!
//!   * each segment's share of the total gradient norm — a discarded span
//!     contributes 0 here, whatever the tolerance,
//!   * `‖Σ segments − whole‖ / ‖whole‖` per tensor, against the run-to-run noise of
//!     an identical repeat, which catches a partial drop the norms would average
//!     away.
//!
//! Segment runs are rescaled before summing: the loss normalizes by the unmasked
//! row count, so a segment covering a quarter of the rows returns gradients ~4x too
//! large.
//!
//! **Leakage.** A cell carries recurrent state forward and a BPTT state gradient
//! backward, and the backbone only resets the latter on the chunked path. So a
//! window can inherit the previous window's recurrence: window B run after window A
//! must give exactly what B gives on a fresh model. Every ordered pair of shapes is
//! checked, which is what puts a chunked window in front of an unchunked one.
//!
//! Both stages of both cell kinds are exercised: the encoder and decoder alternate
//! sLSTM/mLSTM every other block, the backbone is sLSTM every 8th, and residuals are
//! reported per (stage, cell kind).

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use neural_networks::config::{
        BACKBONE_CHUNK, CHAR_HIDDEN, MAX_WINDOW_TOKENS, MAX_WORD_BYTES, OUT_HIDDEN, WORD_BLOCKS,
        WORD_HIDDEN,
    };
    use neural_networks::gpu::Gpu;
    use neural_networks::gpu::hierarchical::{Hierarchical, ModelCfg};
    use neural_networks::nn2::optim::AdamCfg;
    use neural_networks::tokenizer_utf8::Utf8Tokenizer;
    use std::collections::BTreeMap;
    use std::range::Range;

    let mut args = std::env::args().skip(1);
    let segs: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    let full = args.next().is_some_and(|s| s == "full");

    let gpu = Gpu::new().expect("gpu");
    let tok = Utf8Tokenizer::new();
    assert_eq!(CHAR_HIDDEN, OUT_HIDDEN, "decoder ties the encoder char table");
    let cfg = ModelCfg {
        vocab: tok.vocab_size(),
        hc: CHAR_HIDDEN,
        wh: if full { WORD_HIDDEN } else { 256 },
        enc_blocks: if full { 4 } else { 2 }, // alternates sLSTM/mLSTM
        // 9, not 4: the backbone is sLSTM only every 8th block, so a shorter stack
        // would run this whole probe against mLSTM cells alone.
        bb_blocks: if full { WORD_BLOCKS } else { 9 },
        dec_blocks: if full { 4 } else { 2 }, // alternates sLSTM/mLSTM
        heads: if full { 8 } else { 2 },
        dqk: 128,
        w_token: neural_networks::tokenizer_utf8::W_TOKEN as usize,
        cap: 30.0,
    };

    // (name, words, token count of word w, backbone chunk, group row cap)
    type Len = fn(usize) -> usize;
    type Shape = (&'static str, usize, Len, Option<usize>, Option<usize>);
    let ragged: Len = |w| 1 + (w * 2654435761 >> 13) % MAX_WORD_BYTES;
    let shapes: Vec<Shape> = if full {
        vec![
            ("one-chunk", BACKBONE_CHUNK / 2, ragged, None, None),
            ("ragged-odd", BACKBONE_CHUNK * 2 + 37, ragged, None, None),
            ("bimodal-wide", 4400, |w| if w % 2 == 0 { 2 } else { 11 }, None, None),
        ]
    } else {
        vec![
            // one bucket, one chunk: the plain case, nothing to drop
            ("uniform-1chunk", 48, |_| 3, None, None),
            // many buckets, still one backbone chunk: isolates enc/dec grouping
            ("fanned-1chunk", 64, |w| 1 + w % MAX_WORD_BYTES, None, None),
            // several backbone chunks, word count not a multiple of the chunk
            ("ragged-4chunk", 133, ragged, Some(32), None),
            // chunked backbone AND buckets split by the row cap: everything at once
            ("bimodal-split", 96, |w| if w % 2 == 0 { 2 } else { 11 }, Some(24), Some(32)),
        ]
    };

    let build = |words: usize, len: Len| -> (Vec<usize>, Vec<Range<usize>>) {
        let mut tokens = Vec::new();
        let mut spans = Vec::new();
        for w in 0..words {
            let s = tokens.len();
            let n = len(w).clamp(1, MAX_WORD_BYTES);
            if s + n > MAX_WINDOW_TOKENS {
                break;
            }
            for k in 0..n {
                tokens.push(1 + (w + k) % 90);
            }
            spans.push((s..tokens.len()).into());
        }
        (tokens, spans)
    };
    let windows: Vec<(&str, Vec<usize>, Vec<Range<usize>>, Option<usize>, Option<usize>)> = shapes
        .iter()
        .map(|(n, w, l, c, cap)| {
            let (t, s) = build(*w, *l);
            (*n, t, s, *c, *cap)
        })
        .collect();

    let seed = std::env::temp_dir().join("grad_coverage_seed.hier");
    let seed = seed.to_str().unwrap();
    Hierarchical::new(&gpu, cfg).save(&gpu, seed, &[]).expect("save seed");
    let fresh = || {
        // The pool caches the previous model's blocks, and at production width
        // that cache alone is enough to fail the next model's arena allocation.
        neural_networks::gpu::trim_pool(&gpu);
        Hierarchical::load(&gpu, seed, cfg.w_token).expect("load seed")
    };
    // lr 0: zeroes the gradient accumulators without touching the weights, so every
    // run below starts from zero against identical parameters.
    let clear = {
        let mut c = AdamCfg::new(0.0, 0.0);
        c.t = 1;
        c
    };

    let l2 = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let rel = |a: &[f32], b: &[f32]| -> f32 {
        let na = l2(a);
        if na < 1e-12 {
            return 0.0;
        }
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt() / na
    };

    let mut model = fresh();

    // Cell kind per block, from its slot count: an sLSTM cell contributes 4 slots to
    // its block, an mLSTM cell 15. Derived rather than hardcoded so a change to
    // either cell's parameter list shows up as an unknown kind, not a wrong label.
    let kinds: BTreeMap<String, &'static str> = {
        let mut per_block: BTreeMap<String, usize> = BTreeMap::new();
        for (name, _) in model.grad_signature(&gpu) {
            let (blk, _) = name.split_once('.').unwrap_or((name.as_str(), ""));
            *per_block.entry(blk.to_string()).or_default() += 1;
        }
        per_block
            .into_iter()
            // Blocks only: `bb_front`, `dec_head` and `dec_norm` share the prefixes.
            .filter(|(b, _)| {
                b.ends_with(|c: char| c.is_ascii_digit())
                    && ["enc", "bb", "dec"]
                        .iter()
                        .any(|p| b.trim_end_matches(|c: char| c.is_ascii_digit()) == *p)
            })
            .map(|(b, n)| {
                (
                    b,
                    match n {
                        12 => "sLSTM",
                        // q/k/v and the two gate logits are one projection each, so
                        // the cell holds four Linears, not seven.
                        17 => "mLSTM",
                        _ => "?",
                    },
                )
            })
            .collect()
    };
    let kind_of = |name: &str| -> String {
        let blk = name.split_once('.').map(|(b, _)| b).unwrap_or(name);
        let stage = blk.trim_end_matches(|c: char| c.is_ascii_digit());
        match kinds.get(blk) {
            Some(k) => format!("{stage} {k}"),
            None => stage.to_string(),
        }
    };
    {
        let mut n = BTreeMap::new();
        for (_, k) in &kinds {
            *n.entry(*k).or_insert(0) += 1;
        }
        println!("{segs} segments, {} model", if full { "production" } else { "small" });
        println!("blocks by cell kind: {n:?}\n");
        assert!(!kinds.values().any(|k| *k == "?"), "unrecognized block layout");
    }

    let mut any_bad = false;

    println!("=== gradient coverage: does every word reach every gradient?\n");
    for (name, tokens, spans, chunk, cap) in &windows {
        let dw = spans.len() - 1; // word 0 is encode-only
        model.set_bb_chunk(*chunk);
        model.set_group_cap(*cap);

        // Whole window, every decoded word scored.
        let whole_loss = model.forward_backward_masked(&gpu, tokens, spans, &vec![true; dw]);
        let rows_whole = model.last_rows() as f32;
        let whole = model.grad_values(&gpu);
        model.step(&gpu, &clear);

        // The mask path itself is the control: an all-true mask must reproduce the
        // plain path, or the comparison below measures the mask, not the sweep.
        model.forward_backward(&gpu, tokens, spans);
        let plain = model.grad_values(&gpu);
        model.step(&gpu, &clear);

        // The noise floor: the same whole-window gradient a second time. Every
        // residual below comes off a bf16 pipeline and only means something next to
        // what an identical repeat already moves.
        model.forward_backward_masked(&gpu, tokens, spans, &vec![true; dw]);
        let again = model.grad_values(&gpu);
        model.step(&gpu, &clear);

        // Contiguous segments of the decoded-word axis, each run's own normalizer
        // undone before summing.
        let mut sum: Vec<Vec<f32>> = whole.iter().map(|(_, v)| vec![0.0; v.len()]).collect();
        // Per tensor, the summed magnitude of the segments. Against `‖whole‖` this is
        // how much the segments cancel each other — the factor by which any roundoff
        // in a segment is amplified when the residual is measured relative to the
        // whole. A tensor with heavy cancellation cannot be judged on the residual
        // alone.
        let mut mag: Vec<f32> = vec![0.0; whole.len()];
        // (tensor, segment) pairs where the segment produced literally no gradient.
        let mut zeroed: Vec<Vec<(String, usize, usize)>> =
            whole.iter().map(|_| Vec::new()).collect();
        let mut share = Vec::new();
        for s in 0..segs {
            let lo = s * dw / segs;
            let hi = (s + 1) * dw / segs;
            if lo == hi {
                continue;
            }
            let mask: Vec<bool> = (0..dw).map(|w| w >= lo && w < hi).collect();
            model.forward_backward_masked(&gpu, tokens, spans, &mask);
            let scale = model.last_rows() as f32 / rows_whole;
            let g = model.grad_values(&gpu);
            model.step(&gpu, &clear);
            let mut sq = 0.0;
            for (((dst, m), z), (n, src)) in sum
                .iter_mut()
                .zip(mag.iter_mut())
                .zip(zeroed.iter_mut())
                .zip(&g)
            {
                let mut t = 0.0;
                for (d, s) in dst.iter_mut().zip(src) {
                    *d += s * scale;
                    t += (s * scale) * (s * scale);
                }
                // Exact, not a tolerance: with this segment the only scored rows, a
                // tensor that ends at literal zero received nothing from these words.
                // That is what a dropped span looks like — the pass that would have
                // accumulated it was overwritten, so nothing was ever added.
                if src.iter().all(|v| *v == 0.0) {
                    z.push(((*n).clone(), lo, hi));
                }
                *m += t.sqrt();
                sq += t;
            }
            share.push((lo, hi, sq.sqrt()));
        }

        // (residual, repeat noise, cancellation, name) per tensor.
        let mut resid: Vec<(f32, f32, f32, &str)> = whole
            .iter()
            .zip(&sum)
            .zip(&again)
            .zip(&mag)
            .map(|((((n, w), s), (_, r)), m)| {
                (rel(w, s), rel(w, r), m / l2(w).max(1e-30), n.as_str())
            })
            .collect();
        resid.sort_by(|a, b| b.0.total_cmp(&a.0));

        let total: f32 = share.iter().map(|(_, _, n)| n).sum();
        println!(
            "{name:<15}{:>5} words {:>6} tokens  loss {whole_loss:.4}  rows {rows_whole:.0}",
            spans.len(),
            tokens.len()
        );
        let pct: Vec<String> = share
            .iter()
            .map(|(lo, hi, n)| format!("[{lo}..{hi}) {:.1}%", 100.0 * n / total.max(1e-30)))
            .collect();
        println!("{:<15}  segment share  {}", "", pct.join("  "));

        // Worst residual per (stage, cell kind), each next to that tensor's own
        // repeat noise: a real drop is orders above its noise, arithmetic is not.
        let mut per_kind: BTreeMap<String, (f32, f32, f32, &str)> = BTreeMap::new();
        for &(d, noise, cancel, n) in &resid {
            let e = per_kind.entry(kind_of(n)).or_insert((0.0, 0.0, 0.0, n));
            if d > e.0 {
                *e = (d, noise, cancel, n);
            }
        }
        for (k, (d, noise, cancel, n)) in &per_kind {
            // `per part` is the residual measured against what the segments actually
            // carried rather than against what survived their cancellation.
            println!(
                "{:<15}  {k:<12} worst {d:.2e}  per part {:.1e}  cancel {cancel:>6.1}x  noise {noise:.0e}  ({n})",
                "",
                d / cancel.max(1.0)
            );
        }
        // The mask path against the plain path is one run against another of the same
        // arithmetic, so this one IS exact: same kernels, same order, mask all-ones.
        let ctrl = whole
            .iter()
            .zip(&plain)
            .filter(|((_, a), (_, b))| a.iter().zip(b.iter()).any(|(x, y)| x.to_bits() != y.to_bits()))
            .count();

        let dead: Vec<&(String, usize, usize)> = zeroed.iter().flatten().collect();
        if dead.is_empty() {
            println!(
                "{:<15}  exact: every one of {} tensors takes gradient from all {} segments",
                "",
                whole.len(),
                share.len()
            );
        } else {
            println!("{:<15}  *** {} (tensor, segment) pairs got zero gradient", "", dead.len());
            for (n, lo, hi) in dead.iter().take(6) {
                println!("{:<15}      {n} sees nothing from words [{lo}..{hi})", "");
            }
            any_bad = true;
        }
        // A dropped gradient is a whole missing term, not a rounding difference: flag
        // only what stands well clear of the same tensor's repeat noise.
        if let Some(&(d, _, cancel, n)) =
            resid.iter().find(|&&(d, _, cancel, _)| d / cancel.max(1.0) > 0.02)
        {
            println!("{:<15}  *** {n}: residual {d:.2e} at {cancel:.1}x cancellation", "");
            any_bad = true;
        }
        if ctrl > 0 {
            println!("{:<15}  *** all-true mask differs from the plain path in {ctrl} tensors", "");
            any_bad = true;
        }
        println!();
    }

    // The coverage model, its pools and its arena are dead from here on. At
    // production width they are gigabytes, and the sweep below holds two models at a
    // time — without this the second load runs the device out of memory.
    drop(model);
    neural_networks::gpu::trim_pool(&gpu);

    println!("=== leakage: does a window inherit the one before it?\n");
    let run = |m: &mut Hierarchical,
               w: &(&str, Vec<usize>, Vec<Range<usize>>, Option<usize>, Option<usize>)| {
        m.set_bb_chunk(w.3);
        m.set_group_cap(w.4);
        m.forward_backward(&gpu, &w.1, &w.2)
    };
    // Reference: each shape on a model that has seen nothing.
    let refs: Vec<(f32, Vec<(String, Vec<f32>)>)> = windows
        .iter()
        .map(|w| {
            let mut m = fresh();
            let loss = run(&mut m, w);
            let g = m.grad_values(&gpu);
            m.release_activation_buffers();
            (loss, g)
        })
        .collect();

    // Bit-exact: the pipeline is deterministic, so window B after window A must
    // reproduce B on a fresh model exactly. Anything at all carried across the window
    // border — a cell's recurrent state, a BPTT state gradient the unchunked path
    // never resets — moves some bit here.
    println!("  (a bit-exact comparison: '.' is identical, 'X' is not)");
    for wa in windows.iter() {
        let mut row = Vec::new();
        for (b, wb) in windows.iter().enumerate() {
            let mut m = fresh();
            run(&mut m, wa);
            m.step(&gpu, &clear); // clears grads, leaves weights AND cell state
            let loss = run(&mut m, wb);
            let g = m.grad_values(&gpu);
            m.release_activation_buffers();
            let (ref_loss, ref_g) = &refs[b];
            let bad: Vec<&str> = ref_g
                .iter()
                .zip(&g)
                .filter(|((_, x), (_, y))| {
                    x.iter().zip(y.iter()).any(|(p, q)| p.to_bits() != q.to_bits())
                })
                .map(|((n, _), _)| n.as_str())
                .collect();
            // Forward state leaking shows in the loss too; a BPTT state gradient
            // leaking does not touch the loss at all and only shows in the gradients.
            let loss_moved = loss.to_bits() != ref_loss.to_bits();
            if !bad.is_empty() || loss_moved {
                println!(
                    "  {:<15} -> {:<15} loss {} {} tensors differ, e.g. {} ({})",
                    wa.0,
                    wb.0,
                    if loss_moved { "moved," } else { "same," },
                    bad.len(),
                    bad[0],
                    kind_of(bad[0])
                );
                any_bad = true;
            }
            row.push(if bad.is_empty() && !loss_moved { '.' } else { 'X' });
        }
        println!("  {:<15} {}", wa.0, row.iter().collect::<String>());
    }

    println!(
        "\n{}",
        if any_bad {
            "FAIL"
        } else {
            "every word reaches every gradient; no window inherits the one before it"
        }
    );
}
