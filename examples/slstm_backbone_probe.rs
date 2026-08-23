// Backbone gradient probe: runs synthetic windows of growing word count through
// the hierarchical GPU model and prints the per-block gradient norm, split by
// cell type. The backbone is the only stack that carries sLSTM state across
// thousands of steps (encoder/decoder reset per word), so a stabilizer or BPTT
// problem that is invisible there shows up here as a norm that grows with the
// sweep length.
//
//   cargo run --release --features cuda --example slstm_backbone_probe [words...]
use neural_networks::{
    batches::ChunkedWordDataSet,
    config::{CHUNK_BYTES, MAX_WINDOW_TOKENS, MIN_WORDS_PER_SEQ},
    tokenizer_utf8::Utf8Tokenizer,
};
use neural_networks::{
    config::{CHAR_HIDDEN, LOGIT_SOFTCAP, LR, WORD_BLOCKS, WORD_HIDDEN},
    gpu::{
        Gpu,
        hierarchical::{Hierarchical, ModelCfg},
    },
    nn2::optim::AdamCfg,
};
use std::range::Range;

fn main() {
    let lens: Vec<usize> = {
        let a: Vec<usize> = std::env::args()
            .skip(1)
            .filter_map(|s| s.parse().ok())
            .collect();
        if a.is_empty() {
            vec![32, 128, 512, 2048]
        } else {
            a
        }
    };

    let gpu = Gpu::new().expect("no GPU");
    // Defaults to the real training shape; REAL=0 shrinks it for a quick smoke run.
    let real = std::env::var("REAL").map_or(true, |v| v != "0");
    let cfg = if real {
        ModelCfg {
            vocab: 260,
            hc: CHAR_HIDDEN,
            wh: WORD_HIDDEN,
            enc_blocks: 3,
            bb_blocks: WORD_BLOCKS,
            dec_blocks: 3,
            heads: 8,
            dqk: WORD_HIDDEN / 8,
            w_token: 256,
            cap: LOGIT_SOFTCAP,
        }
    } else {
        ModelCfg {
            vocab: 32,
            hc: 64,
            wh: 128,
            enc_blocks: 2,
            bb_blocks: 8,
            dec_blocks: 2,
            heads: 8,
            dqk: 16,
            w_token: 31,
            cap: 30.0,
        }
    };

    let steps: usize = std::env::var("STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    // Real corpus windows when CORPUS is set. Synthetic text is periodic enough that
    // the model memorizes it before any instability can build, so it cannot reproduce
    // a divergence that only real data triggers.
    let corpus: Option<Vec<(Vec<usize>, Vec<Range<usize>>)>> =
        std::env::var("CORPUS").ok().map(|path| {
            let tok = Utf8Tokenizer::new();
            let mut data = ChunkedWordDataSet::open(
                tok,
                &path,
                lens[0],
                MIN_WORDS_PER_SEQ,
                MAX_WINDOW_TOKENS,
                CHUNK_BYTES,
            );
            let chunk = data.next_chunk().expect("corpus chunk");
            let skip: usize = std::env::var("SKIP")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            chunk
                .iter()
                .skip(skip)
                .take(steps.max(1))
                .map(|b| {
                    (
                        b.tokens.iter().map(|&t| t as usize).collect(),
                        b.words.clone(),
                    )
                })
                .collect()
        });

    for &nw in &lens {
        let mut model = Hierarchical::new(&gpu, cfg);
        // 3 chars per word, deterministic, so the only thing varying is length.
        let mut tokens = Vec::with_capacity(nw * 3);
        let mut words = Vec::with_capacity(nw);
        for w in 0..nw {
            let s = tokens.len();
            for k in 0..3 {
                tokens.push(1 + (w * 3 + k) % 20);
            }
            words.push(Range {
                start: s,
                end: tokens.len(),
            });
        }

        println!("\n== {nw} words ==");
        if steps == 0 {
            let loss = model.forward_backward(&gpu, &tokens, &words);
            println!("loss {loss:.4}");
            report(&mut model, &gpu);
            continue;
        }

        // Train in place: a divergence that only appears after the weights move is
        // invisible in a single backward at init.
        let mut acfg = AdamCfg::new(LR, neural_networks::optimizers::WEIGHT_DECAY);
        for s in 0..steps {
            let (tk, wd) = match &corpus {
                Some(w) => {
                    let (t, r) = &w[s % w.len()];
                    (t.as_slice(), r.as_slice())
                }
                None => (tokens.as_slice(), words.as_slice()),
            };
            let loss = model.forward_backward(&gpu, tk, wd);
            let norms = model.grad_norms_by_block(&gpu, "backbone");
            let sl: f32 = norms.iter().step_by(4).sum();
            let ml: f32 = norms
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 4 != 0)
                .map(|(_, v)| v)
                .sum();
            println!("  step {s:3} loss {loss:8.4}  |g| sLSTM {sl:>11.3e}  mLSTM {ml:>11.3e}");
            if norms.iter().any(|v| !v.is_finite()) {
                println!("  -- non-finite gradient, per-stage breakdown:");
                for stage in ["encoder", "backbone", "decoder"] {
                    let n = model.grad_norms_by_block(&gpu, stage);
                    let ext = model.state_extremes_by_block(&gpu, stage);
                    let bad: Vec<usize> = n
                        .iter()
                        .enumerate()
                        .filter(|(_, v)| !v.is_finite())
                        .map(|(i, _)| i)
                        .collect();
                    println!("     {stage:8} {} blocks, non-finite at {bad:?}", n.len());
                    for (i, v) in n.iter().enumerate() {
                        match ext[i] {
                            Some((mn, mc, mr)) => println!(
                                "        [{i:2}] |g| {v:>11.3e}  min|n| {mn:>10.3e}  max|c| {mc:>10.3e}  max|c/n| {mr:>10.3e}"
                            ),
                            None => println!("        [{i:2}] |g| {v:>11.3e}"),
                        }
                    }
                }
                break;
            }
            acfg.t += 1;
            model.step(&gpu, &acfg);
        }
    }
}

fn report(model: &mut Hierarchical, gpu: &Gpu) {
    for (i, gn) in model
        .grad_norms_by_block(gpu, "backbone")
        .iter()
        .enumerate()
    {
        let kind = if i % 4 == 0 { "sLSTM" } else { "mLSTM" };
        println!("  block {i:2} {kind}  |grad| {gn:>12.4e}");
    }
}
