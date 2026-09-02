//! Which recurrent-depth layouts have the same parameter count as the dense backbone.
//!
//! The paper's isoFLOPs arms match block EXECUTIONS and store fewer weights; its
//! isoParam arms do the opposite — same weights, far more compute (`2+20x4+2` is 24
//! unique blocks and 84 executions). This finds the isoParam layouts for our config,
//! because "is it better at the same parameters" is a different question from "is it
//! better at the same compute" and they want different layouts.
//!
//! Only three models are built. Every valid layout has exactly two sLSTM blocks (index
//! 0 of the prelude and of the coda) and the rest mLSTM, so the backbone's parameter
//! count is affine in the unique-block count:
//!
//! ```text
//!   params(pre, core, coda) = K + m * (pre + core + coda - 2)
//! ```
//!
//! with `m` one mLSTM block and `K` everything fixed — the two sLSTM blocks,
//! `bb_front`, `bb_back` and the GRT modulation. Two probes give `m`, a third checks
//! the model rather than trusting it.
//!
//! ```text
//! cargo run --release --features cuda --example grt_isoparam
//! ```

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use neural_networks::config::{
        CHAR_HIDDEN, GRT_R, LOGIT_SOFTCAP, WORD_BLOCKS, WORD_HIDDEN,
    };
    use neural_networks::gpu::Gpu;
    use neural_networks::gpu::grt::GrtCfg;
    use neural_networks::gpu::hierarchical::{Hierarchical, ModelCfg};
    use neural_networks::gpu::train::grt_cfg_from_config;
    use neural_networks::tokenizer_utf8::Utf8Tokenizer;

    let gpu = Gpu::new().expect("gpu");
    let tok = Utf8Tokenizer::new();
    let heads = 8;
    let base = ModelCfg {
        vocab: tok.vocab_size(),
        hc: CHAR_HIDDEN,
        wh: WORD_HIDDEN,
        enc_blocks: 5,
        bb_blocks: WORD_BLOCKS,
        dec_blocks: 5,
        heads,
        dqk: WORD_HIDDEN / heads,
        w_token: neural_networks::tokenizer_utf8::W_TOKEN as usize,
        cap: LOGIT_SOFTCAP,
        grt: None,
    };

    // Backbone parameters of one model, built and dropped.
    let backbone = |grt: Option<GrtCfg>| -> usize {
        let mut m = Hierarchical::new(
            &gpu,
            ModelCfg {
                grt,
                ..base
            },
        );
        m.param_counts().1
    };
    let layout = |pre: usize, core: usize, coda: usize| -> Option<GrtCfg> {
        Some(GrtCfg {
            pre,
            core,
            coda,
            ..grt_cfg_from_config()
        })
    };

    let target = backbone(None);
    let p111 = backbone(layout(1, 1, 1));
    let p211 = backbone(layout(2, 1, 1));
    let m = p211 - p111;
    let k = p111 - m;

    // The affine model has to predict a layout it was not fitted on.
    let check = backbone(layout(2, 3, 2));
    let predicted = k + m * (2 + 3 + 2 - 2);
    assert_eq!(
        check, predicted,
        "the affine parameter model is wrong: 2+3x?+2 measured {check}, predicted {predicted}"
    );

    println!(
        "dense {WORD_BLOCKS} blocks: backbone {:.2}M   (wh {WORD_HIDDEN})",
        target as f64 / 1e6
    );
    println!(
        "one mLSTM block {:.2}M, fixed cost (2 sLSTM + bb_front/back + modulation) {:.2}M\n",
        m as f64 / 1e6,
        k as f64 / 1e6
    );

    // The unique-block count that lands on the dense backbone's parameters.
    let want = 2.0 + (target as f64 - k as f64) / m as f64;
    println!("isoParam wants {want:.2} unique blocks (dense has {WORD_BLOCKS})\n");
    let u = want.round() as usize;

    println!("layouts at {u} unique blocks, R = {GRT_R}:");
    println!("  pre  core  coda   executions   backbone     vs dense");
    let mut rows: Vec<(usize, usize, usize, usize)> = Vec::new();
    for pre in 1..=u.saturating_sub(2) {
        for core in 1..=u.saturating_sub(pre + 1) {
            let coda = u - pre - core;
            if coda == 0 {
                continue;
            }
            rows.push((pre, core, coda, pre + core * GRT_R + coda));
        }
    }
    rows.sort_by_key(|r| r.3);
    let params = k + m * (u - 2);
    for (pre, core, coda, execs) in rows.iter().take(12) {
        println!(
            "  {pre:>3}  {core:>4}  {coda:>4}   {execs:>10}   {:>7.2}M   {:>+.2}M  ({:.2}x compute)",
            params as f64 / 1e6,
            (params as f64 - target as f64) / 1e6,
            *execs as f64 / WORD_BLOCKS as f64,
        );
    }
    println!(
        "\nEvery row stores the same weights; they differ only in how much compute the\n\
         recurrence spends. A big core is the paper's isoParam shape and costs R times\n\
         its size in executions; putting the blocks in the prelude and coda costs one\n\
         execution each but leaves less for the recurrence to iterate on."
    );
}
