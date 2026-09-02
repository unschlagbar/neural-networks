//! What a trained recurrent-depth checkpoint is actually doing with its recurrence.
//!
//! Two measurements, both against real windows from a text file:
//!
//!   * **Loss by recurrence depth.** Depth sampling trains every exit, so `r < R` is a
//!     valid forward. A model whose loss falls as `r` rises is spending the extra
//!     compute on something; a flat curve means the recurrence is decoration and the
//!     quality is coming from the prelude and coda alone.
//!   * **The gate's distribution per step.** The failure mode that looks exactly like
//!     success from a loss curve is a gate that never left its copy-dominant init, so
//!     the shared core contributes a couple of percent per step and the prelude and
//!     coda carry the model. `channels <.5` is the sharpest tell — per-element
//!     specialization splits the channels, it does not merely shift the mean. Under
//!     the untied gate the two branches are reported separately: `f` near 1 with `i`
//!     near 0 is the same inert state, reached by a different route.
//!
//! Forward only and it releases its activations per window, so it is safe to run
//! against a checkpoint while a training job holds the rest of the device.
//!
//! ```text
//! cargo run --release --features cuda --example grt_probe -- models/tiny2_grt [text] [windows]
//! ```

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use neural_networks::config::{
        CHUNK_BYTES, MAX_WINDOW_TOKENS, MIN_WORDS_PER_SEQ, VAL_DATA, WORDS_PER_SEQ,
    };
    use neural_networks::batches::ChunkedWordDataSet;
    use neural_networks::gpu::Gpu;
    use neural_networks::gpu::hierarchical::Hierarchical;
    use neural_networks::tokenizer_utf8::Utf8Tokenizer;

    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or("models/tiny2_grt".into());
    let data = args.next().unwrap_or(VAL_DATA.to_string());
    let want: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);

    let gpu = Gpu::new().expect("gpu");
    let tokenizer = Utf8Tokenizer::new();
    let w_token = tokenizer.w_token() as usize;
    let mut model = match Hierarchical::load(&gpu, &path, w_token) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("could not load '{path}': {e}");
            std::process::exit(2);
        }
    };
    let Some(g) = model.cfg.grt else {
        eprintln!("'{path}' has a dense backbone — nothing to probe");
        std::process::exit(2);
    };
    println!(
        "{path}  step {}  layout {}+{}x{}+{}  ({} executions over {} unique blocks)\n\
         anchor: {}",
        model.step_count,
        g.pre,
        g.core,
        g.r,
        g.coda,
        g.executions(),
        g.unique_blocks(),
        if g.anchor == 0 {
            "frozen (the paper's form)".to_string()
        } else {
            format!(
                "{} block(s), moved every {} applications — {} updates at r = {}",
                g.anchor,
                g.t,
                g.max_anchor_updates(),
                g.r
            )
        },
    );
    // Nothing unwinds these passes, so parking activations would be a pure D2H cost.
    model.set_offload(&gpu, false);

    // Real windows, so the word-length histogram is the one training sees.
    let mut set = ChunkedWordDataSet::open(
        tokenizer,
        &data,
        WORDS_PER_SEQ,
        MIN_WORDS_PER_SEQ,
        MAX_WINDOW_TOKENS,
        CHUNK_BYTES,
    );
    let mut windows: Vec<(Vec<usize>, Vec<std::range::Range<usize>>)> = Vec::new();
    'outer: while let Some(chunk) = set.next_chunk() {
        for batch in chunk.iter() {
            if batch.words.len() < 2 {
                continue;
            }
            windows.push((
                batch.tokens.iter().map(|&t| t as usize).collect(),
                batch.words.to_vec(),
            ));
            if windows.len() == want {
                break 'outer;
            }
        }
    }
    if windows.is_empty() {
        eprintln!("no windows in '{data}'");
        std::process::exit(2);
    }
    println!(
        "{} windows from {data}, {} words each\n",
        windows.len(),
        windows[0].1.len()
    );

    // Depth sampling is `r ~ U{r_min..=R}`, so an exit BELOW `r_min` was never
    // trained and its loss says nothing about what depth is worth — reading the span
    // from r = 1 on an `r_min = 2` model reports the untrained exit as the depth
    // gain, which is several times the real one.
    let lo = g.r_min.clamp(1, g.r);
    println!("loss by recurrence depth  (exits below r = {lo} were never trained)");
    let mut by_depth = Vec::new();
    for r in 1..=g.r {
        model.set_recurrence(Some(r));
        let (mut c, mut w) = (0.0, 0.0);
        for (tokens, words) in &windows {
            c += model.eval_loss(&gpu, tokens, words);
            w += model.last_word_loss();
        }
        let (c, w) = (c / windows.len() as f32, w / windows.len() as f32);
        by_depth.push(c);
        println!(
            "  r = {r:<2} char {c:.4}  ppl {:7.3}   word {w:.4}   {}",
            c.exp(),
            match r.cmp(&lo) {
                std::cmp::Ordering::Less => "UNTRAINED exit".to_string(),
                std::cmp::Ordering::Equal => "(baseline)".to_string(),
                std::cmp::Ordering::Greater => format!("{:+.4} vs r={lo}", c - by_depth[lo - 1]),
            }
        );
    }
    let span = by_depth[lo - 1] - by_depth[by_depth.len() - 1];
    println!(
        "\n  depth is worth {span:+.4} nats from r={lo} to r={}{}",
        g.r,
        if span.abs() < 5e-3 {
            "  — flat: the recurrence is not doing work"
        } else if span < 0.0 {
            "  — NEGATIVE: deeper is worse"
        } else {
            ""
        }
    );

    println!(
        "\ngate distribution per recurrence step  ({})",
        if g.lstm_gate {
            "untied: h = f*h_prev + i*o, so f -> 1 keeps the state and i -> 1 writes"
        } else {
            "tied: g -> 1 copies, g -> 0 overwrites"
        }
    );
    model.set_recurrence(Some(g.r));
    // Keyed by (step, branch) so the untied gate's two halves stay apart — they are
    // different quantities and their average describes neither.
    let mut acc: std::collections::BTreeMap<(usize, &str), [f32; 6]> = Default::default();
    let mut channels = 0;
    for (tokens, words) in &windows {
        for s in model.grt_gate_stats(&gpu, tokens, words) {
            let a = acc.entry((s.step, s.branch)).or_insert([0.0; 6]);
            a[0] += s.mean;
            a[1] += s.std;
            a[2] += s.open;
            a[3] += s.channels_open as f32;
            a[4] += s.channel_min;
            a[5] += s.channel_max;
            channels = s.channels;
        }
    }
    let n = windows.len() as f32;
    println!("        mean     std   below .5   channels <.5   per-channel mean range");
    for ((r, branch), a) in &acc {
        println!(
            "  r={r:<2} {branch}  {:.4}  {:.4}   {:5.1}%    {:5.1} / {channels:<5}   {:.3} .. {:.3}",
            a[0] / n,
            a[1] / n,
            100.0 * a[2] / n,
            a[3] / n,
            a[4] / n,
            a[5] / n,
        );
    }
    if g.lstm_gate {
        // What the untied form buys is the ability to hold the state AND write at the
        // same time, which the tied gate cannot express. If `i` collapses while `f`
        // stays near 1, the deeper steps have gone inert exactly as they did under the
        // tied gate, and the change bought nothing.
        let i_means: Vec<f32> = acc
            .iter()
            .filter(|((_, b), _)| *b == "i")
            .map(|(_, a)| a[0] / n)
            .collect();
        let dead = i_means.iter().skip(1).all(|&m| m < 0.05);
        println!(
            "\n  input gate by step: {}\n  {}",
            i_means
                .iter()
                .map(|m| format!("{m:.3}"))
                .collect::<Vec<_>>()
                .join("  "),
            if dead {
                "collapsed after step 0 — the deeper steps write nothing."
            } else {
                "the deeper steps are still writing."
            }
        );
    }
}
