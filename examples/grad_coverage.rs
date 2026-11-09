//! Does every word of a window reach the gradients, and does a window start clean?

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
    use neural_networks::cpu::optim::AdamCfg;
    use neural_networks::tokenizer_utf8::Utf8Tokenizer;
    use std::collections::BTreeMap;
    use std::range::Range;

    let mut args = std::env::args().skip(1);
    let segs: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    let full = args.next().is_some_and(|s| s == "full");

    let gpu = Gpu::new().expect("gpu");
    let tok = Utf8Tokenizer::new();
    assert_eq!(
        CHAR_HIDDEN, OUT_HIDDEN,
        "decoder ties the encoder char table"
    );
    let cfg = ModelCfg {
        vocab: tok.vocab_size(),
        hc: CHAR_HIDDEN,
        wh: if full { WORD_HIDDEN } else { 256 },
        enc_blocks: if full { 4 } else { 2 }, // alternates sLSTM/mLSTM
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
            (
                "bimodal-wide",
                4400,
                |w| if w % 2 == 0 { 2 } else { 11 },
                None,
                None,
            ),
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
            (
                "bimodal-split",
                96,
                |w| if w % 2 == 0 { 2 } else { 11 },
                Some(24),
                Some(32),
            ),
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
    let windows: Vec<(
        &str,
        Vec<usize>,
        Vec<Range<usize>>,
        Option<usize>,
        Option<usize>,
    )> = shapes
        .iter()
        .map(|(n, w, l, c, cap)| {
            let (t, s) = build(*w, *l);
            (*n, t, s, *c, *cap)
        })
        .collect();

    let seed = std::env::temp_dir().join("grad_coverage_seed.hier");
    let seed = seed.to_str().unwrap();
    Hierarchical::new(&gpu, cfg)
        .save(&gpu, seed)
        .expect("save seed");
    let fresh = || {
        neural_networks::gpu::trim_pool(&gpu);
        Hierarchical::load(&gpu, seed, cfg.w_token).expect("load seed")
    };
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
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
            / na
    };

    let mut model = fresh();

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
                        15 => "mLSTM",
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
        println!(
            "{segs} segments, {} model",
            if full { "production" } else { "small" }
        );
        println!("blocks by cell kind: {n:?}\n");
        assert!(
            !kinds.values().any(|k| *k == "?"),
            "unrecognized block layout"
        );
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

        model.forward_backward(&gpu, tokens, spans);
        let plain = model.grad_values(&gpu);
        model.step(&gpu, &clear);

        model.forward_backward_masked(&gpu, tokens, spans, &vec![true; dw]);
        let again = model.grad_values(&gpu);
        model.step(&gpu, &clear);

        let mut sum: Vec<Vec<f32>> = whole.iter().map(|(_, v)| vec![0.0; v.len()]).collect();
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
            "{name:<15}{:>5} words {:>6} tokens  loss {whole_loss:.3}  rows {rows_whole:.0}",
            spans.len(),
            tokens.len()
        );
        let pct: Vec<String> = share
            .iter()
            .map(|(lo, hi, n)| format!("[{lo}..{hi}) {:.1}%", 100.0 * n / total.max(1e-30)))
            .collect();
        println!("{:<15}  segment share  {}", "", pct.join("  "));

        let mut per_kind: BTreeMap<String, (f32, f32, f32, &str)> = BTreeMap::new();
        for &(d, noise, cancel, n) in &resid {
            let e = per_kind.entry(kind_of(n)).or_insert((0.0, 0.0, 0.0, n));
            if d > e.0 {
                *e = (d, noise, cancel, n);
            }
        }
        for (k, (d, noise, cancel, n)) in &per_kind {
            println!(
                "{:<15}  {k:<12} worst {d:.2e}  per part {:.1e}  cancel {cancel:>6.1}x  noise {noise:.0e}  ({n})",
                "",
                d / cancel.max(1.0)
            );
        }
        let ctrl = whole
            .iter()
            .zip(&plain)
            .filter(|((_, a), (_, b))| {
                a.iter()
                    .zip(b.iter())
                    .any(|(x, y)| x.to_bits() != y.to_bits())
            })
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
            println!(
                "{:<15}  *** {} (tensor, segment) pairs got zero gradient",
                "",
                dead.len()
            );
            for (n, lo, hi) in dead.iter().take(6) {
                println!("{:<15}      {n} sees nothing from words [{lo}..{hi})", "");
            }
            any_bad = true;
        }
        if let Some(&(d, _, cancel, n)) = resid
            .iter()
            .find(|&&(d, _, cancel, _)| d / cancel.max(1.0) > 0.02)
        {
            println!(
                "{:<15}  *** {n}: residual {d:.2e} at {cancel:.1}x cancellation",
                ""
            );
            any_bad = true;
        }
        if ctrl > 0 {
            println!(
                "{:<15}  *** all-true mask differs from the plain path in {ctrl} tensors",
                ""
            );
            any_bad = true;
        }
        println!();
    }

    drop(model);
    neural_networks::gpu::trim_pool(&gpu);

    println!("=== leakage: does a window inherit the one before it?\n");
    let run = |m: &mut Hierarchical,
               w: &(
        &str,
        Vec<usize>,
        Vec<Range<usize>>,
        Option<usize>,
        Option<usize>,
    )| {
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
                    x.iter()
                        .zip(y.iter())
                        .any(|(p, q)| p.to_bits() != q.to_bits())
                })
                .map(|((n, _), _)| n.as_str())
                .collect();
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
            row.push(if bad.is_empty() && !loss_moved {
                '.'
            } else {
                'X'
            });
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
