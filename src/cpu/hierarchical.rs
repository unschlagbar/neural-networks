//! The hierarchical (HAT-style) model on the CPU, read from and written to v2
//! checkpoints (`crate::checkpoint`).

use std::{io, mem, range::Range};

use crate::checkpoint::{self, BlockKind, Layout, ModelCfg, Seen};
use crate::config::{
    BACKBONE_WEIGHT_DECAY, DECODER_LR_SCALE, DECODER_WEIGHT_DECAY, ENCODER_LR_SCALE,
    ENCODER_WEIGHT_DECAY, GROUP_MAX_ROWS, up_of,
};
use crate::cpu::block::{MLstmBlock, SLstmBlock};
use crate::cpu::loss::softmax_rows;
use crate::cpu::optim::AdamCfg;
use crate::cpu::{Embedding, Linear, RmsNorm, SoftCap};
use crate::tensor::Tensor;
use crate::tokenizer_utf8::Utf8Tokenizer;

pub enum AnyBlock {
    S(SLstmBlock),
    M(MLstmBlock),
}

impl AnyBlock {
    fn new(kind: BlockKind, h: usize) -> Self {
        match kind {
            BlockKind::SLstm => AnyBlock::S(SLstmBlock::new_slstm(h, up_of(h))),
            BlockKind::MLstm { heads, dqk } => {
                AnyBlock::M(MLstmBlock::new_mlstm(h, up_of(h), heads, dqk))
            }
        }
    }

    pub fn forward(&mut self, x: &Tensor) -> Tensor {
        match self {
            AnyBlock::S(b) => b.forward(x),
            AnyBlock::M(b) => b.forward(x),
        }
    }

    pub fn backward(&mut self, dy: &Tensor) -> Tensor {
        match self {
            AnyBlock::S(b) => b.backward(dy),
            AnyBlock::M(b) => b.backward(dy),
        }
    }

    fn step(&mut self, cfg: &AdamCfg) {
        match self {
            AnyBlock::S(b) => b.step(cfg),
            AnyBlock::M(b) => b.step(cfg),
        }
    }

    fn set_resets(&mut self, steps: &[usize]) {
        match self {
            AnyBlock::S(b) => b.set_resets(steps),
            AnyBlock::M(b) => b.set_resets(steps),
        }
    }

    fn grads_mut(&mut self) -> Vec<&mut Tensor> {
        match self {
            AnyBlock::S(b) => b.grads_mut(),
            AnyBlock::M(b) => b.grads_mut(),
        }
    }
}

/// `cfg` at a stage's own rate multiple and weight decay.
fn stage_cfg(cfg: &AdamCfg, lr_scale: f32, weight_decay: f32) -> AdamCfg {
    AdamCfg {
        lr: cfg.lr * lr_scale,
        weight_decay,
        ..*cfg
    }
}

/// Words grouped by their step count (`steps(w)`, only words `keep` passes),
/// each group at most `GROUP_MAX_ROWS` rows of steps × words.
fn groups(n: usize, steps: impl Fn(usize) -> usize, keep: impl Fn(usize) -> bool) -> Vec<(usize, Vec<usize>)> {
    let mut ws: Vec<(usize, usize)> = (0..n).filter(|&w| keep(w)).map(|w| (steps(w), w)).collect();
    ws.sort_unstable();
    let mut out: Vec<(usize, Vec<usize>)> = Vec::new();
    for (s, w) in ws {
        let cap = (GROUP_MAX_ROWS / s).max(1);
        match out.last_mut() {
            Some((gs, g)) if *gs == s && g.len() < cap => g.push(w),
            _ => out.push((s, vec![w])),
        }
    }
    out
}

pub struct Hierarchical {
    pub cfg: ModelCfg,
    pub layout: Layout,
    pub step_count: usize,
    pub seen: Seen,
    /// tied char table `[vocab, hc]`
    pub table: Embedding,
    pub encoder: Vec<AnyBlock>,
    pub bb_front: Linear,
    pub backbone: Vec<AnyBlock>,
    pub bb_back: Linear,
    pub decoder: Vec<AnyBlock>,
    pub dec_norm: RmsNorm,
    pub head: Linear,
    softcap: SoftCap,
    doc_starts: Vec<usize>,
    last_word_loss: f32,
    last_rows: usize,
    last_scored_words: usize,
}

impl Hierarchical {
    /// A new model in the [standard](Layout::standard) layout.
    pub fn new(cfg: ModelCfg) -> Self {
        Self::with_layout(cfg, Layout::standard(&cfg))
    }

    pub fn with_layout(cfg: ModelCfg, layout: Layout) -> Self {
        if let Err(e) = layout.check(&cfg) {
            panic!("layout does not fit {cfg:?}: {e}");
        }
        let stage = |kinds: &[BlockKind], h: usize| kinds.iter().map(|&k| AnyBlock::new(k, h)).collect();
        Self {
            table: Embedding::new(cfg.vocab, cfg.hc),
            encoder: stage(&layout.encoder, cfg.hc),
            bb_front: Linear::new(cfg.hc, cfg.wh),
            backbone: stage(&layout.backbone, cfg.wh),
            bb_back: Linear::new(cfg.wh, cfg.hc),
            decoder: stage(&layout.decoder, cfg.hc),
            dec_norm: RmsNorm::new(cfg.hc),
            head: {
                let mut h = Linear::new(cfg.hc, cfg.vocab);
                h.b.zero_();
                h
            },
            softcap: SoftCap::new(cfg.cap),
            cfg,
            layout,
            step_count: 0,
            seen: Seen::default(),
            doc_starts: Vec::new(),
            last_word_loss: 0.0,
            last_rows: 0,
            last_scored_words: 0,
        }
    }

    /// Where the documents packed into the next window begin, as indices into its
    /// words (see [`WordBatch::doc_starts`](crate::batches::WordBatch)). The backbone
    /// restarts at each of them, and the document's first word and the `<END>`
    /// before it carry no loss. Consumed by the window it applies to.
    pub fn set_doc_starts(&mut self, starts: &[usize]) {
        self.doc_starts.clear();
        self.doc_starts.extend_from_slice(starts);
    }

    /// Mean NLL per scored word of the last window.
    pub fn last_word_loss(&self) -> f32 {
        self.last_word_loss
    }

    /// Decoder rows the last window's loss was averaged over.
    pub fn last_rows(&self) -> usize {
        self.last_rows
    }

    /// Tokens of the words the last window scored (its rows minus their `[W]`s).
    pub fn last_scored_tokens(&self) -> usize {
        self.last_rows - self.last_scored_words
    }

    pub fn last_scored_words(&self) -> usize {
        self.last_scored_words
    }

    /// Forward + backward over one window; accumulates every gradient and returns
    /// the mean decode cross-entropy. Word 0 is encode-only, words 1.. are decoded.
    pub fn forward_backward(&mut self, tokens: &[usize], words: &[Range<usize>]) -> f32 {
        self.run(tokens, words, None, true)
    }

    /// [`forward_backward`](Self::forward_backward) with one loss flag per decoded
    /// word (`word_loss[w]` for `words[w + 1]`). Every word is still encoded and the
    /// backbone reads all of them; only flagged words carry loss and gradient, and
    /// the loss is the mean over their rows.
    pub fn forward_backward_masked(&mut self, tokens: &[usize], words: &[Range<usize>], word_loss: &[bool]) -> f32 {
        self.run(tokens, words, Some(word_loss), true)
    }

    /// The decode loss without any backward.
    pub fn eval_loss(&mut self, tokens: &[usize], words: &[Range<usize>]) -> f32 {
        self.run(tokens, words, None, false)
    }

    fn run(&mut self, tokens: &[usize], words: &[Range<usize>], word_loss: Option<&[bool]>, grad: bool) -> f32 {
        let doc_starts = mem::take(&mut self.doc_starts);
        self.last_word_loss = 0.0;
        self.last_rows = 0;
        self.last_scored_words = 0;
        if words.len() < 2 {
            return 0.0;
        }
        let dw = words.len() - 1;
        let len = |w: usize| words[w].end - words[w].start;

        let mut mask = word_loss.map_or_else(|| vec![true; dw], <[bool]>::to_vec);
        assert_eq!(mask.len(), dw, "one loss flag per decoded word");
        for &d in &doc_starts {
            for i in [d.checked_sub(1), d.checked_sub(2)].into_iter().flatten() {
                if i < dw {
                    mask[i] = false;
                }
            }
        }
        let rows: usize = (0..dw).filter(|&w| mask[w]).map(|w| len(w + 1) + 1).sum();
        let scored_words = mask.iter().filter(|&&m| m).count();
        if rows == 0 {
            return 0.0;
        }
        let (hc, wh) = (self.cfg.hc, self.cfg.wh);

        // PHASE 1: ENCODER
        let enc_groups = groups(dw, |w| len(w) + 1, |_| true);
        let mut e = Tensor::zeros(&[dw, hc]);
        for (steps, ws) in &enc_groups {
            let h = self.encode_group(tokens, words, *steps, ws);
            for (i, &w) in ws.iter().enumerate() {
                let src = (i * steps + steps - 1) * hc;
                e.data[w * hc..(w + 1) * hc].copy_from_slice(&h.data[src..src + hc]);
            }
        }

        // PHASE 2: BACKBONE
        let resets: Vec<usize> = doc_starts.iter().copied().filter(|&d| d > 0 && d < dw).collect();
        let mut x = self.bb_front.forward(&e).reshape(&[1, dw, wh]);
        for blk in &mut self.backbone {
            blk.set_resets(&resets);
            x = blk.forward(&x);
        }
        let o = self.bb_back.forward(&x.reshape(&[dw, wh]));
        drop(x);

        // PHASE 3: DECODER, forward and back per group
        let inv_rows = 1.0 / rows as f32;
        let mut nll = 0.0f64;
        let mut d_o = Tensor::zeros(&[if grad { dw } else { 0 }, hc]);
        for (steps, ws) in groups(dw, |w| len(w + 1) + 1, |w| mask[w]) {
            let (nw, n) = (ws.len(), ws.len() * steps);
            let mut x = Tensor::zeros(&[nw, steps, hc]);
            let mut targets = vec![self.cfg.w_token; n];
            for (i, &w) in ws.iter().enumerate() {
                let start = words[w + 1].start;
                let base = i * steps;
                x.data[base * hc..(base + 1) * hc].copy_from_slice(&o.data[w * hc..(w + 1) * hc]);
                for k in 1..steps {
                    let c = tokens[start + k - 1];
                    x.data[(base + k) * hc..(base + k + 1) * hc].copy_from_slice(self.table.table.row(c));
                    targets[base + k - 1] = c;
                }
            }
            for blk in &mut self.decoder {
                x = blk.forward(&x);
            }
            let hn = self.dec_norm.forward(&x.reshape(&[n, hc]));
            let capped = self.softcap.forward(&self.head.forward(&hn));
            let (sum, d_capped) = cross_entropy(&capped, &targets, inv_rows);
            nll += sum;
            if !grad {
                continue;
            }
            let d_hn = self.head.backward(&self.softcap.backward(&d_capped));
            let mut d = self.dec_norm.backward(&d_hn).reshape(&[nw, steps, hc]);
            for blk in self.decoder.iter_mut().rev() {
                d = blk.backward(&d);
            }
            for (i, &w) in ws.iter().enumerate() {
                let start = words[w + 1].start;
                let base = i * steps;
                d_o.data[w * hc..(w + 1) * hc].copy_from_slice(&d.data[base * hc..(base + 1) * hc]);
                for k in 1..steps {
                    let row = tokens[start + k - 1] * hc;
                    let src = &d.data[(base + k) * hc..(base + k + 1) * hc];
                    for (t, &g) in self.table.dtable.data[row..row + hc].iter_mut().zip(src) {
                        *t += g;
                    }
                }
            }
        }
        self.last_rows = rows;
        self.last_scored_words = scored_words;
        self.last_word_loss = (nll / scored_words.max(1) as f64) as f32;
        let loss = (nll * inv_rows as f64) as f32;
        if !grad {
            return loss;
        }

        // BACKBONE BACKWARD
        let mut d = self.bb_back.backward(&d_o).reshape(&[1, dw, wh]);
        for blk in self.backbone.iter_mut().rev() {
            d = blk.backward(&d);
        }
        let d_e = self.bb_front.backward(&d.reshape(&[dw, wh]));
        drop(d);

        // ENCODER BACKWARD, the last group forwarded still holding its activations
        for (gi, (steps, ws)) in enc_groups.iter().enumerate().rev() {
            if gi + 1 != enc_groups.len() {
                self.encode_group(tokens, words, *steps, ws);
            }
            let nw = ws.len();
            let mut d = Tensor::zeros(&[nw, *steps, hc]);
            for (i, &w) in ws.iter().enumerate() {
                let dst = (i * steps + steps - 1) * hc;
                d.data[dst..dst + hc].copy_from_slice(&d_e.data[w * hc..(w + 1) * hc]);
            }
            for blk in self.encoder.iter_mut().rev() {
                d = blk.backward(&d);
            }
            self.table.backward(&d.reshape(&[nw * steps, hc]));
        }
        loss
    }

    /// The encoder over words `ws`, each `steps − 1` chars plus `[W]`. Returns the
    /// last block's output `[words, steps, hc]`.
    fn encode_group(&mut self, tokens: &[usize], words: &[Range<usize>], steps: usize, ws: &[usize]) -> Tensor {
        let mut ids = Vec::with_capacity(ws.len() * steps);
        for &w in ws {
            ids.extend_from_slice(&tokens[words[w].start..words[w].end]);
            ids.push(self.cfg.w_token);
        }
        let mut x = self.table.forward(&ids).reshape(&[ws.len(), steps, self.cfg.hc]);
        for blk in &mut self.encoder {
            x = blk.forward(&x);
        }
        x
    }

    /// Every gradient accumulator, in no particular order.
    pub fn grads_mut(&mut self) -> Vec<&mut Tensor> {
        let mut g = self.table.grads_mut();
        for blk in self.encoder.iter_mut().chain(&mut self.backbone).chain(&mut self.decoder) {
            g.extend(blk.grads_mut());
        }
        for lin in [&mut self.bb_front, &mut self.bb_back, &mut self.head] {
            g.extend(lin.grads_mut());
        }
        g.extend(self.dec_norm.grads_mut());
        g
    }

    /// Scale the accumulated gradients down to a global L2 norm of at most
    /// `max_norm`, like `torch.nn.utils.clip_grad_norm_`. Returns the norm before
    /// clipping.
    pub fn clip_grad_norm(&mut self, max_norm: f32) -> f32 {
        let mut grads = self.grads_mut();
        let sq: f64 = grads
            .iter()
            .map(|g| g.data.iter().map(|&v| v as f64 * v as f64).sum::<f64>())
            .sum();
        let norm = sq.sqrt() as f32;
        if norm > max_norm {
            let s = max_norm / norm;
            for g in grads.iter_mut() {
                g.data.iter_mut().for_each(|v| *v *= s);
            }
        }
        norm
    }

    /// AdamW across every stage, each at its own rate multiple and decay. Clears
    /// the gradients.
    pub fn step(&mut self, cfg: &AdamCfg) {
        let enc = stage_cfg(cfg, ENCODER_LR_SCALE, ENCODER_WEIGHT_DECAY);
        self.table.step(&enc);
        for blk in &mut self.encoder {
            blk.step(&enc);
        }
        let bb = stage_cfg(cfg, 1.0, BACKBONE_WEIGHT_DECAY);
        self.bb_front.step(&bb);
        for blk in &mut self.backbone {
            blk.step(&bb);
        }
        self.bb_back.step(&bb);
        let dec = stage_cfg(cfg, DECODER_LR_SCALE, DECODER_WEIGHT_DECAY);
        for blk in &mut self.decoder {
            blk.step(&dec);
        }
        self.dec_norm.step(&dec);
        // The logit head trains undecayed with its bias frozen, as on the GPU.
        self.head.step_frozen_bias(&dec, false);
        self.step_count += 1;
    }

    /// Write a v2 checkpoint.
    pub fn save(&self, path: &str) -> io::Result<()> {
        let header = checkpoint::Header {
            cfg: self.cfg,
            layout: self.layout.clone(),
            step: self.step_count as u64,
            seen: self.seen,
        };
        let blocks = self.encoder.iter().chain(&self.backbone).chain(&self.decoder);
        let count = 1 + 2 + 2 + 1 + 2 + blocks.map(v2::tensor_count).sum::<usize>();
        let mut w = checkpoint::Writer::create(path, &header, count)?;
        w.tensor(self.table.table.dims(), &self.table.table.data)?;
        for blk in &self.encoder {
            v2::write_block(&mut w, blk)?;
        }
        v2::write_linear(&mut w, &self.bb_front)?;
        for blk in &self.backbone {
            v2::write_block(&mut w, blk)?;
        }
        v2::write_linear(&mut w, &self.bb_back)?;
        for blk in &self.decoder {
            v2::write_block(&mut w, blk)?;
        }
        w.tensor(self.dec_norm.gamma.dims(), &self.dec_norm.gamma.data)?;
        v2::write_linear(&mut w, &self.head)?;
        w.finish()
    }

    /// Read a v2 checkpoint.
    pub fn load(path: &str) -> io::Result<Self> {
        let err = |e: io::Error| io::Error::new(e.kind(), format!("{path}: {e}"));
        let r = checkpoint::Reader::open(path).map_err(err)?;
        let checkpoint::Header {
            cfg,
            layout,
            step,
            seen,
        } = r.header.clone();
        let w_token = Utf8Tokenizer::new().w_token() as usize;
        if cfg.w_token != w_token {
            return Err(err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("checkpoint [W] token {} differs from the tokenizer's {w_token}", cfg.w_token),
            )));
        }
        let mut m = Self::with_layout(cfg, layout);
        let mut r = v2::Load { r, buf: Vec::new() };
        let load = |m: &mut Self, r: &mut v2::Load| -> io::Result<()> {
            r.into(&mut m.table.table)?;
            for blk in &mut m.encoder {
                r.block(blk)?;
            }
            r.linear(&mut m.bb_front)?;
            for blk in &mut m.backbone {
                r.block(blk)?;
            }
            r.linear(&mut m.bb_back)?;
            for blk in &mut m.decoder {
                r.block(blk)?;
            }
            r.into(&mut m.dec_norm.gamma)?;
            r.linear(&mut m.head)
        };
        load(&mut m, &mut r).map_err(err)?;
        r.r.finish().map_err(err)?;
        m.step_count = step as usize;
        m.seen = seen;
        Ok(m)
    }
}

/// Summed cross-entropy of every row and its gradient `(softmax − onehot) · scale`.
fn cross_entropy(capped: &Tensor, targets: &[usize], scale: f32) -> (f64, Tensor) {
    let vocab = capped.cols();
    let mut probs = softmax_rows(capped);
    let mut loss = 0.0f64;
    for (row, &t) in probs.data.chunks_mut(vocab).zip(targets) {
        loss -= (row[t].max(1e-30) as f64).ln();
        row[t] -= 1.0;
        row.iter_mut().for_each(|v| *v *= scale);
    }
    (loss, probs)
}

/// The cpu layers in the v2 checkpoint's layouts, which are the GPU model's: the
/// mLSTM's fused `q ‖ k/√dqk ‖ v ‖ o` and `ĩ ‖ f̃` projections, the sLSTM's gates
/// concatenated `z ‖ i ‖ f ‖ o` and split into input and recurrent halves.
mod v2 {
    use super::*;
    use crate::cpu::block::{Block, Cell};
    use crate::cpu::{MLstm, SLstm};

    pub fn tensor_count(b: &AnyBlock) -> usize {
        match b {
            AnyBlock::S(_) => 12,
            AnyBlock::M(_) => 15,
        }
    }

    pub fn write_linear(w: &mut checkpoint::Writer, l: &Linear) -> io::Result<()> {
        w.tensor(l.w.dims(), &l.w.data)?;
        w.tensor(l.b.dims(), &l.b.data)
    }

    /// Columns of `parts` (each `[rows, ·]`) side by side.
    fn cat_cols(parts: &[&[f32]], rows: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(parts.iter().map(|p| p.len()).sum());
        for r in 0..rows {
            for p in parts {
                let c = p.len() / rows;
                out.extend_from_slice(&p[r * c..(r + 1) * c]);
            }
        }
        out
    }

    fn scaled(t: &Tensor, s: f32) -> Vec<f32> {
        t.data.iter().map(|v| v * s).collect()
    }

    fn slstm_gates(c: &SLstm) -> [(&Tensor, &Tensor); 4] {
        [(&c.wz, &c.bz), (&c.wi, &c.bi), (&c.wf, &c.bf), (&c.wo, &c.bo)]
    }

    fn write_ffn<C: Cell>(w: &mut checkpoint::Writer, b: &Block<C>) -> io::Result<()> {
        w.tensor(b.pre_norm2.gamma.dims(), &b.pre_norm2.gamma.data)?;
        write_linear(w, &b.lin_gate)?;
        write_linear(w, &b.lin_value)?;
        write_linear(w, &b.lin_down)
    }

    pub fn write_block(w: &mut checkpoint::Writer, blk: &AnyBlock) -> io::Result<()> {
        match blk {
            AnyBlock::S(b) => {
                w.tensor(b.pre_norm1.gamma.dims(), &b.pre_norm1.gamma.data)?;
                let c = &b.cell;
                let (inp, h) = (c.input_size, c.hidden_size);
                let gates = slstm_gates(c);
                let half = |r0: usize, rows: usize| {
                    let parts: Vec<&[f32]> = gates.iter().map(|(g, _)| &g.data[r0 * h..(r0 + rows) * h]).collect();
                    cat_cols(&parts, rows)
                };
                w.tensor(&[inp, 4 * h], &half(0, inp))?;
                w.tensor(&[h, 4 * h], &half(inp, h))?;
                let bias: Vec<f32> = gates.iter().flat_map(|(_, b)| b.data.iter().copied()).collect();
                w.tensor(&[4 * h], &bias)?;
                let post = b.post_cell_norm.as_ref().expect("an sLSTM block has a post-cell norm");
                w.tensor(post.gamma.dims(), &post.gamma.data)?;
                write_ffn(w, b)
            }
            AnyBlock::M(b) => {
                w.tensor(b.pre_norm1.gamma.dims(), &b.pre_norm1.gamma.data)?;
                let c = &b.cell;
                let (inp, s) = (c.input_size, 1.0 / (c.dqk as f32).sqrt());
                let (wk, bk) = (scaled(&c.wk, s), scaled(&c.bk, s));
                let qkvo = cat_cols(&[&c.wq.data, &wk, &c.wv.data, &c.wo.data], inp);
                w.tensor(&[inp, qkvo.len() / inp], &qkvo)?;
                let bias = cat_cols(&[&c.bq.data, &bk, &c.bv.data, &c.bo.data], 1);
                w.tensor(&[bias.len()], &bias)?;
                w.tensor(&[inp, 2 * c.heads], &cat_cols(&[&c.wi.data, &c.wf.data], inp))?;
                w.tensor(&[2 * c.heads], &cat_cols(&[&c.bi.data, &c.bf.data], 1))?;
                w.tensor(c.w_out.dims(), &c.w_out.data)?;
                w.tensor(c.b_out.dims(), &c.b_out.data)?;
                w.tensor(c.gamma.dims(), &c.gamma.data)?;
                write_ffn(w, b)
            }
        }
    }

    pub struct Load {
        pub r: checkpoint::Reader,
        pub buf: Vec<f32>,
    }

    impl Load {
        fn next(&mut self, dims: &[usize]) -> io::Result<&[f32]> {
            self.r.next_into(dims, &mut self.buf)?;
            Ok(&self.buf)
        }

        pub fn into(&mut self, t: &mut Tensor) -> io::Result<()> {
            let dims = t.dims().to_vec();
            t.data.copy_from_slice(self.next(&dims)?);
            Ok(())
        }

        pub fn linear(&mut self, l: &mut Linear) -> io::Result<()> {
            self.into(&mut l.w)?;
            self.into(&mut l.b)
        }

        /// The columns `c0..c0 + t.cols()` of the next `[rows, width]` tensor into `t`.
        fn split_cols(src: &[f32], width: usize, c0: usize, t: &mut Tensor, scale: f32) {
            let c = t.data.len() / (src.len() / width);
            for (r, dst) in t.data.chunks_mut(c).enumerate() {
                for (d, &v) in dst.iter_mut().zip(&src[r * width + c0..r * width + c0 + c]) {
                    *d = v * scale;
                }
            }
        }

        fn ffn<C: Cell>(&mut self, b: &mut Block<C>) -> io::Result<()> {
            self.into(&mut b.pre_norm2.gamma)?;
            self.linear(&mut b.lin_gate)?;
            self.linear(&mut b.lin_value)?;
            self.linear(&mut b.lin_down)
        }

        pub fn block(&mut self, blk: &mut AnyBlock) -> io::Result<()> {
            match blk {
                AnyBlock::S(b) => {
                    self.into(&mut b.pre_norm1.gamma)?;
                    let c = &mut b.cell;
                    let (inp, h) = (c.input_size, c.hidden_size);
                    let wx = self.next(&[inp, 4 * h])?.to_vec();
                    let whr = self.next(&[h, 4 * h])?.to_vec();
                    let bias = self.next(&[4 * h])?.to_vec();
                    let gates = [
                        (&mut c.wz, &mut c.bz),
                        (&mut c.wi, &mut c.bi),
                        (&mut c.wf, &mut c.bf),
                        (&mut c.wo, &mut c.bo),
                    ];
                    for (g, (w, b)) in gates.into_iter().enumerate() {
                        for r in 0..inp + h {
                            let (src, sr) = if r < inp { (&wx, r) } else { (&whr, r - inp) };
                            w.data[r * h..(r + 1) * h]
                                .copy_from_slice(&src[sr * 4 * h + g * h..sr * 4 * h + (g + 1) * h]);
                        }
                        b.data.copy_from_slice(&bias[g * h..(g + 1) * h]);
                    }
                    let post = b.post_cell_norm.as_mut().expect("an sLSTM block has a post-cell norm");
                    self.into(&mut post.gamma)?;
                    self.ffn(b)
                }
                AnyBlock::M(b) => {
                    self.into(&mut b.pre_norm1.gamma)?;
                    let c: &mut MLstm = &mut b.cell;
                    let (inp, d, heads) = (c.input_size, c.d, c.heads);
                    let dq = heads * c.dqk;
                    let s = (c.dqk as f32).sqrt();
                    let width = 2 * dq + 2 * d;
                    let w = self.next(&[inp, width])?.to_vec();
                    let bias = self.next(&[width])?.to_vec();
                    for (src, rows) in [(&w, inp), (&bias, 1)] {
                        let wide = rows == inp;
                        let (q, k, v, o) = if wide {
                            (&mut c.wq, &mut c.wk, &mut c.wv, &mut c.wo)
                        } else {
                            (&mut c.bq, &mut c.bk, &mut c.bv, &mut c.bo)
                        };
                        Self::split_cols(src, width, 0, q, 1.0);
                        Self::split_cols(src, width, dq, k, s);
                        Self::split_cols(src, width, 2 * dq, v, 1.0);
                        Self::split_cols(src, width, 2 * dq + d, o, 1.0);
                    }
                    let w = self.next(&[inp, 2 * heads])?.to_vec();
                    Self::split_cols(&w, 2 * heads, 0, &mut c.wi, 1.0);
                    Self::split_cols(&w, 2 * heads, heads, &mut c.wf, 1.0);
                    let bias = self.next(&[2 * heads])?.to_vec();
                    Self::split_cols(&bias, 2 * heads, 0, &mut c.bi, 1.0);
                    Self::split_cols(&bias, 2 * heads, heads, &mut c.bf, 1.0);
                    self.into(&mut c.w_out)?;
                    self.into(&mut c.b_out)?;
                    self.into(&mut c.gamma)?;
                    self.ffn(b)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment;
    use crate::tokenizer_utf8::END_TOKEN;

    fn cfg() -> ModelCfg {
        let tok = Utf8Tokenizer::new();
        ModelCfg {
            vocab: tok.vocab_size(),
            hc: 64,
            wh: 24,
            enc_blocks: 2,
            bb_blocks: 3,
            dec_blocks: 2,
            heads: 2,
            dqk: 4,
            w_token: tok.w_token() as usize,
            cap: 30.0,
        }
    }

    /// Every block kind in every stage, and a backbone sLSTM like s4/s5 have.
    fn model() -> Hierarchical {
        let c = cfg();
        let mut layout = Layout::standard(&c);
        layout.backbone[1] = BlockKind::SLstm;
        let mut m = Hierarchical::with_layout(c, layout);
        // Gate weights start at zero; give them something to differentiate.
        for blk in m.encoder.iter_mut().chain(&mut m.backbone).chain(&mut m.decoder) {
            if let AnyBlock::M(b) = blk {
                b.cell.wi = Tensor::random(b.cell.wi.dims(), 0.3);
                b.cell.wf = Tensor::random(b.cell.wf.dims(), 0.3);
            }
        }
        m
    }

    fn window(text: &str) -> (Vec<usize>, Vec<Range<usize>>) {
        let tok = Utf8Tokenizer::new();
        let tokens = tok.to_tokens(text);
        window_of(&tokens)
    }

    fn window_of(tokens: &[u16]) -> (Vec<usize>, Vec<Range<usize>>) {
        let mut words = Vec::new();
        let mut start = 0;
        for e in segment::word_ends(tokens) {
            words.push(Range { start, end: e as usize });
            start = e as usize;
        }
        (tokens.iter().map(|&t| t as usize).collect(), words)
    }

    fn zero_grads(m: &mut Hierarchical) {
        for g in m.grads_mut() {
            g.zero_();
        }
    }

    fn grads(m: &mut Hierarchical) -> Vec<f32> {
        let mut out = Vec::new();
        for g in m.grads_mut() {
            out.extend_from_slice(&g.data);
        }
        out
    }

    #[test]
    fn memorizes_a_window() {
        let mut m = model();
        let (tokens, words) = window("let a = b(c); fn d() -> e { f }");
        let mut opt = AdamCfg::new(3e-3, 0.0);
        let first = m.eval_loss(&tokens, &words);
        for _ in 0..150 {
            m.forward_backward(&tokens, &words);
            opt.t += 1;
            m.step(&opt);
        }
        let last = m.eval_loss(&tokens, &words);
        assert!(last < first * 0.3, "decode loss did not fall enough: {first} -> {last}");
    }

    /// The tied table's gradient, which comes from the encoder (run again in
    /// backward) and from the decoder's char slots, against a finite difference
    /// along it.
    #[test]
    fn tied_table_grad_matches_fd() {
        let mut m = model();
        let (tokens, words) = window("use std::io; let x = y.z(1, 22, 333);");
        zero_grads(&mut m);
        m.forward_backward(&tokens, &words);
        let g = m.table.dtable.data.clone();
        let norm: f32 = g.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm > 1e-6);
        let eps = 1e-3;
        let shift = |m: &mut Hierarchical, s: f32| {
            for (w, &gi) in m.table.table.data.iter_mut().zip(&g) {
                *w += s * gi / norm;
            }
        };
        shift(&mut m, eps);
        let plus = m.eval_loss(&tokens, &words);
        shift(&mut m, -2.0 * eps);
        let minus = m.eval_loss(&tokens, &words);
        let fd = (plus - minus) / (2.0 * eps);
        assert!((fd - norm).abs() <= 0.3 * norm, "‖G‖ {norm} vs fd {fd}");
    }

    /// The backbone's output projection gets gradient only through the context
    /// injected at each decoded word's first slot, so this checks the `d_o` routing.
    #[test]
    fn injected_context_grad_matches_fd() {
        let mut m = model();
        let (tokens, words) = window("use std::io; let x = y.z(1, 22, 333);");
        zero_grads(&mut m);
        m.forward_backward(&tokens, &words);
        let g = m.bb_back.dw.data.clone();
        let norm: f32 = g.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm > 1e-6);
        let eps = 1e-3;
        let shift = |m: &mut Hierarchical, s: f32| {
            for (w, &gi) in m.bb_back.w.data.iter_mut().zip(&g) {
                *w += s * gi / norm;
            }
        };
        shift(&mut m, eps);
        let plus = m.eval_loss(&tokens, &words);
        shift(&mut m, -2.0 * eps);
        let minus = m.eval_loss(&tokens, &words);
        let fd = (plus - minus) / (2.0 * eps);
        assert!((fd - norm).abs() <= 0.3 * norm, "‖G‖ {norm} vs fd {fd}");
    }

    /// A fully masked window trains on nothing: zero loss, every gradient zero.
    #[test]
    fn full_mask_zeros_the_gradient() {
        let mut m = model();
        let (tokens, words) = window("fn main() { println!(\"hi\"); }");
        zero_grads(&mut m);
        let loss = m.forward_backward_masked(&tokens, &words, &vec![false; words.len() - 1]);
        assert_eq!(loss, 0.0);
        assert!(grads(&mut m).iter().all(|&g| g == 0.0));
    }

    /// Masking is additive and isolating: the full-window gradient is the sum of
    /// the one-word-at-a-time gradients, each weighted by its share of the rows
    /// (the loss is a mean over the scored rows).
    #[test]
    fn per_word_masks_sum_to_the_full_gradient() {
        let mut m = model();
        let (tokens, words) = window("let cat = sat(on, the).mat();");
        let dw = words.len() - 1;
        zero_grads(&mut m);
        m.forward_backward(&tokens, &words);
        let full = grads(&mut m);
        let total = m.last_rows() as f32;

        let mut summed = vec![0.0; full.len()];
        for w in 0..dw {
            let mut mask = vec![false; dw];
            mask[w] = true;
            zero_grads(&mut m);
            m.forward_backward_masked(&tokens, &words, &mask);
            let share = m.last_rows() as f32 / total;
            for (a, g) in summed.iter_mut().zip(grads(&mut m)) {
                *a += share * g;
            }
        }
        for (i, (&a, &b)) in full.iter().zip(&summed).enumerate() {
            assert!((a - b).abs() <= 1e-4 * a.abs().max(1e-2), "grad {i}: full {a} vs summed {b}");
        }
    }

    /// A packed document decodes exactly as it would in a window of its own: the
    /// backbone restarts at its first word, which carries no loss.
    #[test]
    fn packed_document_decodes_like_its_own_window() {
        let tok = Utf8Tokenizer::new();
        let a = tok.to_tokens("let a = 1; let b = a;");
        let b = tok.to_tokens("fn main() { run(2); }");
        let mut packed = a.clone();
        packed.push(END_TOKEN);
        packed.extend_from_slice(&b);
        let (tokens, words) = window_of(&packed);
        let (own_tokens, own_words) = window_of(&b);
        let d = words.len() - own_words.len();
        assert_eq!(words[d].start, a.len() + 1, "document B starts at word {d}");

        let mut m = model();
        let own = m.eval_loss(&own_tokens, &own_words);
        // Only document B's decodes; the border mask removes the first of them.
        let mask: Vec<bool> = (0..words.len() - 1).map(|w| w + 1 > d).collect();
        m.set_doc_starts(&[d]);
        let got = m.run(&tokens, &words, Some(&mask), false);
        assert!((got - own).abs() <= 1e-5 * own, "packed {got} vs own window {own}");

        let leaked = m.run(&tokens, &words, Some(&mask), false);
        assert!((leaked - own).abs() > 1e-4, "without the reset the document must see the one before");
    }

    /// Save → load reproduces every parameter and the loss bit for bit (every dqk
    /// is 4, so the 1/√dqk folded into k is exact).
    #[test]
    fn v2_round_trip_is_exact() {
        let path = std::env::temp_dir().join("cpu_hier_v2_round_trip");
        let path = path.to_str().unwrap();
        let mut m = model();
        m.step_count = 42;
        m.seen.add_pretrain(100, 20);
        m.save(path).unwrap();
        let mut back = Hierarchical::load(path).unwrap();
        assert_eq!(back.step_count, 42);
        assert_eq!(back.seen, m.seen);
        assert_eq!(back.layout, m.layout);
        let (tokens, words) = window("impl Foo for Bar { fn baz(&self) {} }");
        assert_eq!(m.eval_loss(&tokens, &words).to_bits(), back.eval_loss(&tokens, &words).to_bits());
        back.save(&format!("{path}.2")).unwrap();
        assert_eq!(std::fs::read(path).unwrap(), std::fs::read(format!("{path}.2")).unwrap());
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(format!("{path}.2"));
    }

    /// The CPU model's v2 file is the one the CPU inference path reads.
    #[test]
    fn infer_reads_what_the_trainer_writes() {
        let path = std::env::temp_dir().join("cpu_hier_v2_infer");
        let path = path.to_str().unwrap();
        model().save(path).unwrap();
        crate::infer::Hierarchical::load(path).unwrap();
        let _ = std::fs::remove_file(path);
    }
}
