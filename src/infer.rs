//! CPU inference for the hierarchical model, loaded from a v2 checkpoint.
//!
//! Weights only, no gradients or optimizer state, kept in the checkpoint's own
//! (GPU) layouts: the mLSTM's fused `q‖k‖v‖o` and `i‖f` projections with 1/√dqk
//! already folded into `k`, the sLSTM's gate-concatenated `wx`/`whr`/`bcat`.
//! Everything runs one step at a time: the backbone carries its state across
//! words, the encoder and the decoder restart per word.

use std::io;

use rayon::prelude::*;

use crate::checkpoint::{self, BlockKind, ModelCfg, Seen};
use crate::config::up_of;
use crate::segment;
use crate::tensor::gemm::gemm_nn;
use crate::tokenizer_utf8::{BYTE_TOKENS, TOOL_CLOSE_TOKEN, TOOL_OPEN_TOKEN, Utf8Tokenizer};

const EPS: f32 = 1e-6;

/// Words encoded together; bounds the encoder's per-row state.
const ENCODE_BATCH: usize = 64;

/// Consecutive empty words (`[W]` straight after a word start) before sampling
/// gives up.
const MAX_EMPTY_WORDS: usize = 16;

#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
fn log_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        -(1.0 + (-x).exp()).ln()
    } else {
        x - (1.0 + x.exp()).ln()
    }
}

/// `x / rms(x) · γ` over groups of `group` features, in place on every row.
fn rms_norm(x: &mut [f32], gamma: &[f32], group: usize) {
    for row in x.chunks_mut(gamma.len()) {
        for (g, chunk) in row.chunks_mut(group).enumerate() {
            let ss: f32 = chunk.iter().map(|v| v * v).sum();
            let inv = 1.0 / (ss / group as f32 + EPS).sqrt();
            let gm = &gamma[g * group..(g + 1) * group];
            for (v, &w) in chunk.iter_mut().zip(gm) {
                *v *= inv * w;
            }
        }
    }
}

struct Linear {
    inp: usize,
    out: usize,
    /// `[inp, out]`
    w: Vec<f32>,
    b: Vec<f32>,
}

/// Fewest weight rows a task of a single-row product takes.
const SLAB_MIN: usize = 16;
/// Below this many weights a product runs on the calling thread.
const PAR_MIN: usize = 1 << 16;

/// `y = x · W` for `x` `[rows, k]` and `W` `[k, n]`, overwriting `y`.
fn matmul(x: &[f32], rows: usize, w: &[f32], k: usize, n: usize, y: &mut [f32]) {
    debug_assert_eq!(x.len(), rows * k);
    debug_assert_eq!(w.len(), k * n);
    debug_assert_eq!(y.len(), rows * n);
    if k * n < PAR_MIN {
        gemm_nn(rows, k, n, x, w, y, 0.0);
    } else if rows == 1 {
        // Each task streams a contiguous slab of W's rows into a partial sum:
        // the product is bandwidth-bound and this reads W strictly in order.
        let per = k.div_ceil(rayon::current_num_threads()).max(SLAB_MIN);
        let partial = x
            .par_chunks(per)
            .zip(w.par_chunks(per * n))
            .map(|(xs, ws)| {
                let mut acc = vec![0.0; n];
                for (&a, w) in xs.iter().zip(ws.chunks_exact(n)) {
                    for (o, &wv) in acc.iter_mut().zip(w) {
                        *o = a.mul_add(wv, *o);
                    }
                }
                acc
            })
            .reduce_with(|mut a, b| {
                a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                a
            })
            .unwrap();
        y.copy_from_slice(&partial);
    } else {
        let per = rows.div_ceil(rayon::current_num_threads()).max(1);
        y.par_chunks_mut(per * n)
            .zip(x.par_chunks(per * k))
            .for_each(|(yc, xc)| gemm_nn(xc.len() / k, k, n, xc, w, yc, 0.0));
    }
}

impl Linear {
    /// `y = x · W + b` over `rows` rows.
    fn forward(&self, x: &[f32], rows: usize, y: &mut [f32]) {
        matmul(x, rows, &self.w, self.inp, self.out, y);
        for row in y.chunks_mut(self.out) {
            for (o, &b) in row.iter_mut().zip(&self.b) {
                *o += b;
            }
        }
    }
}

struct SLstm {
    h: usize,
    /// `[in, 4H]`, gates z, i, f, o as column blocks.
    wx: Vec<f32>,
    /// `[H, 4H]`
    whr: Vec<f32>,
    /// `[4H]`
    b: Vec<f32>,
    post_gamma: Vec<f32>,
}

struct MLstm {
    heads: usize,
    dqk: usize,
    dhv: usize,
    /// `→ q ‖ k/√dqk ‖ v ‖ o`
    qkvo: Linear,
    /// `→ ĩ ‖ f̃`, one per head
    gates: Linear,
    out: Linear,
    /// head-wise norm scale `[d]`, groups of `dhv`
    gamma: Vec<f32>,
}

enum Cell {
    S(SLstm),
    M(MLstm),
}

/// One block's recurrent state for `rows` independent sequences.
enum State {
    S {
        h: Vec<f32>,
        c: Vec<f32>,
        n: Vec<f32>,
        m: Vec<f32>,
    },
    M {
        c: Vec<f32>,
        n: Vec<f32>,
        m: Vec<f32>,
    },
}

impl SLstm {
    fn step(&self, x: &[f32], rows: usize, st: &mut State, y: &mut [f32]) {
        let State::S { h, c, n, m } = st else { unreachable!() };
        let (hd, h4) = (self.h, 4 * self.h);
        let inp = x.len() / rows;
        let mut g = vec![0.0; rows * h4];
        let mut gh = vec![0.0; rows * h4];
        matmul(x, rows, &self.wx, inp, h4, &mut g);
        matmul(h, rows, &self.whr, hd, h4, &mut gh);
        for (a, b) in g.iter_mut().zip(&gh) {
            *a += b;
        }
        for r in 0..rows {
            let gr = &g[r * h4..(r + 1) * h4];
            for j in 0..hd {
                let k = r * hd + j;
                let z = (gr[j] + self.b[j]).tanh();
                let i_pre = gr[hd + j] + self.b[hd + j];
                let f_pre = gr[2 * hd + j] + self.b[2 * hd + j];
                let o = sigmoid(gr[3 * hd + j] + self.b[3 * hd + j]);
                let fm = log_sigmoid(f_pre) + m[k];
                // n == 0 is a sequence's first step: m = ĩ makes i' exactly 1.
                let mn = if n[k] == 0.0 { i_pre } else { fm.max(i_pre) };
                let ip = (i_pre - mn).exp();
                let fp = (fm - mn).exp();
                c[k] = fp * c[k] + ip * z;
                n[k] = fp * n[k] + ip;
                m[k] = mn;
                h[k] = o * c[k] / n[k];
            }
        }
        y.copy_from_slice(h);
        rms_norm(y, &self.post_gamma, hd);
    }
}

impl MLstm {
    fn d(&self) -> usize {
        self.heads * self.dhv
    }

    fn step(&self, x: &[f32], rows: usize, st: &mut State, y: &mut [f32]) {
        let State::M { c, n, m } = st else { unreachable!() };
        let (heads, dqk, dhv, d) = (self.heads, self.dqk, self.dhv, self.d());
        let dq = heads * dqk;
        let w = self.qkvo.out;
        let mut proj = vec![0.0; rows * w];
        self.qkvo.forward(x, rows, &mut proj);
        let mut gates = vec![0.0; rows * 2 * heads];
        self.gates.forward(x, rows, &mut gates);

        let mut hcat = vec![0.0; rows * d];
        for r in 0..rows {
            let p = &proj[r * w..(r + 1) * w];
            let (q, rest) = p.split_at(dq);
            let (k, rest) = rest.split_at(dq);
            let (v, o) = rest.split_at(d);
            let gr = &gates[r * 2 * heads..(r + 1) * 2 * heads];
            for hh in 0..heads {
                let bh = r * heads + hh;
                let (i_pre, f_pre) = (gr[hh], gr[heads + hh]);
                let lf = log_sigmoid(f_pre) + m[bh];
                let mh = lf.max(i_pre);
                let ip = (i_pre - mh).exp();
                let fp = (lf - mh).exp();
                m[bh] = mh;
                let qh = &q[hh * dqk..(hh + 1) * dqk];
                let kh = &k[hh * dqk..(hh + 1) * dqk];
                let vh = &v[hh * dhv..(hh + 1) * dhv];
                let nh = &mut n[bh * dqk..(bh + 1) * dqk];
                let mut nq = 0.0;
                for j in 0..dqk {
                    nh[j] = fp.mul_add(nh[j], ip * kh[j]);
                    nq += nh[j] * qh[j];
                }
                // max(|n_trueᵀq|, 1) in stabilized terms
                let inv_psi = 1.0 / nq.abs().max((-mh).exp());
                let out = &mut hcat[r * d + hh * dhv..r * d + (hh + 1) * dhv];
                let ch = &mut c[bh * dhv * dqk..(bh + 1) * dhv * dqk];
                for i in 0..dhv {
                    let iv = ip * vh[i];
                    let row = &mut ch[i * dqk..(i + 1) * dqk];
                    let mut s = 0.0;
                    for j in 0..dqk {
                        row[j] = fp.mul_add(row[j], iv * kh[j]);
                        s += row[j] * qh[j];
                    }
                    out[i] = s * inv_psi;
                }
            }
            let hr = &mut hcat[r * d..(r + 1) * d];
            rms_norm(hr, &self.gamma, dhv);
            for (hv, &ov) in hr.iter_mut().zip(o) {
                *hv *= sigmoid(ov);
            }
        }
        self.out.forward(&hcat, rows, y);
    }
}

/// Pre-norm residual block: `z = x + cell(norm1(x))`, `y = z + SwiGLU(norm2(z))`.
/// The sLSTM's post-cell norm is part of its [`SLstm::step`].
struct Block {
    norm1: Vec<f32>,
    cell: Cell,
    norm2: Vec<f32>,
    gate: Linear,
    value: Linear,
    down: Linear,
}

impl Block {
    fn hidden(&self) -> usize {
        self.norm1.len()
    }

    fn state(&self, rows: usize) -> State {
        let z = |n: usize| vec![0.0; rows * n];
        match &self.cell {
            Cell::S(s) => State::S {
                h: z(s.h),
                c: z(s.h),
                n: z(s.h),
                m: z(s.h),
            },
            Cell::M(c) => State::M {
                c: z(c.heads * c.dhv * c.dqk),
                n: z(c.heads * c.dqk),
                m: z(c.heads),
            },
        }
    }

    /// `x` `[rows, H]` in, the block's output written back into it.
    fn step(&self, x: &mut [f32], rows: usize, st: &mut State) {
        let h = self.hidden();
        let mut xn = x.to_vec();
        rms_norm(&mut xn, &self.norm1, h);
        let mut cell = vec![0.0; rows * h];
        match &self.cell {
            Cell::S(s) => s.step(&xn, rows, st, &mut cell),
            Cell::M(m) => m.step(&xn, rows, st, &mut cell),
        }
        for (z, c) in x.iter_mut().zip(&cell) {
            *z += c;
        }
        let mut zn = x.to_vec();
        rms_norm(&mut zn, &self.norm2, h);
        let u = self.gate.out;
        let mut gate = vec![0.0; rows * u];
        let mut value = vec![0.0; rows * u];
        self.gate.forward(&zn, rows, &mut gate);
        self.value.forward(&zn, rows, &mut value);
        for (g, v) in gate.iter_mut().zip(&value) {
            *g = *g * sigmoid(*g) * v;
        }
        self.down.forward(&gate, rows, &mut cell);
        for (z, d) in x.iter_mut().zip(&cell) {
            *z += d;
        }
    }
}

fn step_stack(blocks: &[Block], x: &mut [f32], rows: usize, states: &mut [State]) {
    for (b, s) in blocks.iter().zip(states) {
        b.step(x, rows, s);
    }
}

/// Reads tensors in `param_slots` order, checking each one's shape.
struct Load {
    r: checkpoint::Reader,
    path: String,
}

impl Load {
    fn next(&mut self, dims: &[usize]) -> io::Result<Vec<f32>> {
        let mut v = Vec::new();
        self.r
            .next_into(dims, &mut v)
            .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", self.path)))?;
        Ok(v)
    }

    fn linear(&mut self, inp: usize, out: usize) -> io::Result<Linear> {
        Ok(Linear {
            inp,
            out,
            w: self.next(&[inp, out])?,
            b: self.next(&[out])?,
        })
    }

    fn block(&mut self, kind: BlockKind, h: usize) -> io::Result<Block> {
        let up = up_of(h);
        let norm1 = self.next(&[h])?;
        let cell = match kind {
            BlockKind::SLstm => Cell::S(SLstm {
                h,
                wx: self.next(&[h, 4 * h])?,
                whr: self.next(&[h, 4 * h])?,
                b: self.next(&[4 * h])?,
                post_gamma: self.next(&[h])?,
            }),
            BlockKind::MLstm { heads, dqk } => Cell::M(MLstm {
                heads,
                dqk,
                dhv: h / heads,
                qkvo: self.linear(h, 2 * heads * dqk + 2 * h)?,
                gates: self.linear(h, 2 * heads)?,
                out: self.linear(h, h)?,
                gamma: self.next(&[h])?,
            }),
        };
        Ok(Block {
            norm1,
            cell,
            norm2: self.next(&[h])?,
            gate: self.linear(h, up)?,
            value: self.linear(h, up)?,
            down: self.linear(up, h)?,
        })
    }

    fn stage(&mut self, kinds: &[BlockKind], h: usize) -> io::Result<Vec<Block>> {
        kinds.iter().map(|&k| self.block(k, h)).collect()
    }
}

/// The hierarchical model plus the conversation its backbone has read.
pub struct Hierarchical {
    pub cfg: ModelCfg,
    pub step: usize,
    pub seen: Seen,
    pub vocab_size: usize,
    tokenizer: Utf8Tokenizer,
    /// tied char table `[vocab, hc]`
    table: Vec<f32>,
    encoder: Vec<Block>,
    bb_front: Linear,
    backbone: Vec<Block>,
    bb_back: Linear,
    decoder: Vec<Block>,
    dec_norm: Vec<f32>,
    head: Linear,
    bb_state: Vec<State>,
    /// The next word's first decoder input: the backbone's output for the last
    /// word it read.
    ctx: Vec<f32>,
}

impl Hierarchical {
    pub fn load(path: &str) -> io::Result<Self> {
        let r = checkpoint::Reader::open(path)
            .map_err(|e| io::Error::new(e.kind(), format!("{path}: {e}")))?;
        let checkpoint::Header {
            cfg,
            layout,
            step,
            seen,
        } = r.header.clone();
        let tokenizer = Utf8Tokenizer::new();
        if cfg.w_token != tokenizer.w_token() as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{path}: checkpoint [W] token {} differs from the tokenizer's {}",
                    cfg.w_token,
                    tokenizer.w_token()
                ),
            ));
        }
        let mut l = Load {
            r,
            path: path.to_string(),
        };
        let (hc, wh) = (cfg.hc, cfg.wh);
        let table = l.next(&[cfg.vocab, hc])?;
        let encoder = l.stage(&layout.encoder, hc)?;
        let bb_front = l.linear(hc, wh)?;
        let backbone = l.stage(&layout.backbone, wh)?;
        let bb_back = l.linear(wh, hc)?;
        let decoder = l.stage(&layout.decoder, hc)?;
        let dec_norm = l.next(&[hc])?;
        let head = l.linear(hc, cfg.vocab)?;
        l.r.finish()
            .map_err(|e| io::Error::new(e.kind(), format!("{path}: {e}")))?;
        let bb_state = backbone.iter().map(|b| b.state(1)).collect();
        Ok(Self {
            cfg,
            step: step as usize,
            seen,
            vocab_size: cfg.vocab,
            tokenizer,
            table,
            encoder,
            bb_front,
            backbone,
            bb_back,
            decoder,
            dec_norm,
            head,
            bb_state,
            ctx: vec![0.0; hc],
        })
    }

    fn embed(&self, tok: usize) -> &[f32] {
        &self.table[tok * self.cfg.hc..(tok + 1) * self.cfg.hc]
    }

    /// Forget everything the backbone has read.
    pub fn reset(&mut self) {
        self.bb_state = self.backbone.iter().map(|b| b.state(1)).collect();
        self.ctx.fill(0.0);
    }

    /// Every word's `e_w` (`[n, hc]`): its chars then `[W]`, read out at `[W]`.
    pub fn encode(&self, words: &[&[u16]]) -> Vec<f32> {
        let hc = self.cfg.hc;
        let w_tok = self.cfg.w_token;
        let mut out = vec![0.0; words.len() * hc];
        for (batch, dst) in words.chunks(ENCODE_BATCH).zip(out.chunks_mut(ENCODE_BATCH * hc)) {
            let rows = batch.len();
            let steps = batch.iter().map(|w| w.len()).max().unwrap_or(0) + 1;
            let mut states: Vec<State> = self.encoder.iter().map(|b| b.state(rows)).collect();
            let mut x = vec![0.0; rows * hc];
            for t in 0..steps {
                for (r, w) in batch.iter().enumerate() {
                    let tok = w.get(t).map_or(w_tok, |&c| c as usize);
                    x[r * hc..(r + 1) * hc].copy_from_slice(self.embed(tok));
                }
                step_stack(&self.encoder, &mut x, rows, &mut states);
                for (r, w) in batch.iter().enumerate() {
                    if w.len() == t {
                        dst[r * hc..(r + 1) * hc].copy_from_slice(&x[r * hc..(r + 1) * hc]);
                    }
                }
            }
        }
        out
    }

    /// Step the backbone on one word embedding, leaving its output in `ctx`.
    fn advance(&mut self, e_w: &[f32]) {
        let mut x = vec![0.0; self.cfg.wh];
        self.bb_front.forward(e_w, 1, &mut x);
        step_stack(&self.backbone, &mut x, 1, &mut self.bb_state);
        let mut ctx = vec![0.0; self.cfg.hc];
        self.bb_back.forward(&x, 1, &mut ctx);
        self.ctx = ctx;
    }

    fn encode_advance(&mut self, word: &[u16]) {
        let e = self.encode(&[word]);
        self.advance(&e);
    }

    /// Encode `tokens` into the backbone without resetting it. They must start on a
    /// word border, which every turn border of a formatted chat is.
    pub fn feed(&mut self, tokens: &[u16]) {
        let mut words = Vec::new();
        let mut start = 0;
        for e in segment::word_ends(tokens) {
            let w = &tokens[start..e as usize];
            start = e as usize;
            if !w.is_empty() {
                words.push(w);
            }
        }
        self.feed_words(&words);
    }

    fn feed_words(&mut self, words: &[&[u16]]) {
        let e = self.encode(words);
        for e_w in e.chunks_exact(self.cfg.hc) {
            self.advance(e_w);
        }
    }

    /// [`feed`](Self::feed) for words given as embeddings (`[n, hc]`, e.g. heard
    /// through the audio encoder): every row steps the backbone once.
    pub fn feed_embeddings(&mut self, embeds: &[f32]) {
        for e_w in embeds.chunks_exact(self.cfg.hc) {
            self.advance(e_w);
        }
    }

    /// One decoder step from `x`, returning the capped logits.
    fn decode_step(&self, x: &[f32], states: &mut [State]) -> Vec<f32> {
        let mut h = x.to_vec();
        step_stack(&self.decoder, &mut h, 1, states);
        rms_norm(&mut h, &self.dec_norm, self.cfg.hc);
        let mut logits = vec![0.0; self.cfg.vocab];
        self.head.forward(&h, 1, &mut logits);
        let cap = self.cfg.cap;
        for v in logits.iter_mut() {
            *v = cap * (*v / cap).tanh();
        }
        logits
    }

    fn dec_states(&self) -> Vec<State> {
        self.decoder.iter().map(|b| b.state(1)).collect()
    }

    /// Continue `prefix`. Its last word may be cut off (`fn ma`), so it is
    /// teacher-forced as the start of the first generated word rather than encoded.
    pub fn sample(
        &mut self,
        prefix: &[u16],
        max_len: usize,
        temperature: f32,
        top_p: f32,
        callback: impl FnMut(u16) -> bool,
    ) -> Vec<u16> {
        self.reset();
        let (complete, tail) = segment::split_prefix(prefix);
        let mut tail = tail.to_vec();
        if complete.is_empty() {
            // Seed a word context from the tail, or a random byte.
            if tail.is_empty() {
                tail.push(rand::random_range(0..BYTE_TOKENS) as u16);
            }
            let seed = std::mem::take(&mut tail);
            self.encode_advance(&seed);
        } else {
            self.feed_words(&complete);
        }
        self.generate(tail, max_len, temperature, top_p, callback)
    }

    /// Continue the words given as embeddings (`[n, hc]`).
    pub fn sample_from_embeddings(
        &mut self,
        embeds: &[f32],
        max_len: usize,
        temperature: f32,
        top_p: f32,
        callback: impl FnMut(u16) -> bool,
    ) -> Vec<u16> {
        self.reset();
        self.feed_embeddings(embeds);
        self.generate(Vec::new(), max_len, temperature, top_p, callback)
    }

    /// Sample words from `ctx`, teacher-forcing `forced` as the first one's start.
    fn generate(
        &mut self,
        forced: Vec<u16>,
        max_len: usize,
        temperature: f32,
        top_p: f32,
        mut callback: impl FnMut(u16) -> bool,
    ) -> Vec<u16> {
        let w_tok = self.cfg.w_token;
        let mut forced = forced.into_iter();
        let mut out = Vec::with_capacity(max_len);
        let mut word: Vec<u16> = Vec::new();
        let mut empty_words = 0;
        let mut states = self.dec_states();
        let mut x = self.ctx.clone();
        loop {
            let logits = self.decode_step(&x, &mut states);
            let (tok, was_forced) = match forced.next() {
                Some(t) => (t, true),
                None => (sample_top_p(&logits, temperature, top_p) as u16, false),
            };
            if tok as usize == w_tok {
                if word.is_empty() {
                    empty_words += 1;
                    if empty_words > MAX_EMPTY_WORDS {
                        break;
                    }
                } else {
                    self.encode_advance(&word);
                    word.clear();
                    empty_words = 0;
                }
                states = self.dec_states();
                x.copy_from_slice(&self.ctx);
                continue;
            }
            if !was_forced {
                out.push(tok);
                if out.len() >= max_len || !callback(tok) {
                    break;
                }
            }
            word.push(tok);
            x.copy_from_slice(self.embed(tok as usize));
        }
        out
    }

    /// Reset, read the formatted `prompt` (its trailing `<SEP>` included) and
    /// generate the reply.
    pub fn sample_chat(
        &mut self,
        prompt: &[u16],
        max_len: usize,
        temperature: f32,
        tool_temperature: f32,
        top_p: f32,
        callback: impl FnMut(u16) -> bool,
    ) -> Vec<u16> {
        self.reset();
        self.feed(prompt);
        if prompt.is_empty() {
            self.encode_advance(&[rand::random_range(0..BYTE_TOKENS) as u16]);
        }
        self.continue_chat(max_len, temperature, tool_temperature, top_p, callback)
    }

    /// Generate one reply from what the backbone has read, until `<END>`. The
    /// reply's words are encoded as the model closes them with `[W]`, and the
    /// closing `<END>` too, so the state afterwards has read the whole exchange.
    /// Between `<tool>` and `</tool>` it samples at `tool_temperature`.
    pub fn continue_chat(
        &mut self,
        max_len: usize,
        temperature: f32,
        tool_temperature: f32,
        top_p: f32,
        mut callback: impl FnMut(u16) -> bool,
    ) -> Vec<u16> {
        let w_tok = self.cfg.w_token;
        let end_tok = self.tokenizer.end_token();
        let mut out = Vec::with_capacity(max_len);
        let mut word: Vec<u16> = Vec::new();
        let mut empty_words = 0;
        let mut in_tool = false;
        let mut states = self.dec_states();
        let mut x = self.ctx.clone();
        loop {
            let logits = self.decode_step(&x, &mut states);
            let temp = if in_tool { tool_temperature } else { temperature };
            let tok = sample_top_p(&logits, temp, top_p) as u16;
            match tok {
                TOOL_OPEN_TOKEN => in_tool = true,
                TOOL_CLOSE_TOKEN => in_tool = false,
                _ => {}
            }
            if tok == end_tok {
                break;
            }
            if tok as usize == w_tok {
                if word.is_empty() {
                    empty_words += 1;
                    if empty_words > MAX_EMPTY_WORDS {
                        break;
                    }
                } else {
                    self.encode_advance(&word);
                    word.clear();
                    empty_words = 0;
                }
                states = self.dec_states();
                x.copy_from_slice(&self.ctx);
                continue;
            }
            out.push(tok);
            if out.len() >= max_len || !callback(tok) {
                break;
            }
            word.push(tok);
            x.copy_from_slice(self.embed(tok as usize));
        }
        if !word.is_empty() {
            self.encode_advance(&word);
        }
        self.encode_advance(&[end_tok]);
        out
    }
}

/// Nucleus sampling at `temperature`; `temperature <= 0` is argmax.
pub fn sample_top_p(logits: &[f32], temperature: f32, top_p: f32) -> usize {
    let argmax = || {
        (0..logits.len())
            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
            .unwrap()
    };
    if temperature <= 0.0 {
        return argmax();
    }
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut q: Vec<f32> = logits.iter().map(|&v| ((v - max) / temperature).exp()).collect();
    let sum: f32 = q.iter().sum();
    q.iter_mut().for_each(|v| *v /= sum);

    let mut idx: Vec<usize> = (0..q.len()).collect();
    idx.sort_unstable_by(|&a, &b| q[b].total_cmp(&q[a]));
    let mut cum = 0.0;
    let mut kept = 0;
    while kept < idx.len() && cum < top_p {
        cum += q[idx[kept]];
        kept += 1;
    }
    let r = rand::random_range(0.0..cum);
    let mut acc = 0.0;
    for &i in &idx[..kept] {
        acc += q[i];
        if acc >= r {
            return i;
        }
    }
    idx[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu;
    use crate::tensor::Tensor;

    fn lin(l: &cpu::Linear) -> Linear {
        Linear {
            inp: l.w.rows(),
            out: l.w.cols(),
            w: l.w.data.clone(),
            b: l.b.data.clone(),
        }
    }

    /// The cpu cell's four `[in+H, H]` gate matrices packed the way the GPU stores them.
    fn slstm(c: &cpu::SLstm, post: &[f32]) -> SLstm {
        let (inp, h) = (c.input_size, c.hidden_size);
        let h4 = 4 * h;
        let mut wx = vec![0.0; inp * h4];
        let mut whr = vec![0.0; h * h4];
        let mut b = vec![0.0; h4];
        for (g, (w, bias)) in [(&c.wz, &c.bz), (&c.wi, &c.bi), (&c.wf, &c.bf), (&c.wo, &c.bo)]
            .into_iter()
            .enumerate()
        {
            for r in 0..inp + h {
                let (dst, dr) = if r < inp { (&mut wx, r) } else { (&mut whr, r - inp) };
                dst[dr * h4 + g * h..dr * h4 + (g + 1) * h].copy_from_slice(&w.data[r * h..(r + 1) * h]);
            }
            b[g * h..(g + 1) * h].copy_from_slice(&bias.data);
        }
        SLstm {
            h,
            wx,
            whr,
            b,
            post_gamma: post.to_vec(),
        }
    }

    fn cat(parts: &[&Tensor], rows: usize) -> Vec<f32> {
        let mut out = Vec::new();
        for r in 0..rows {
            for p in parts {
                let c = p.data.len() / rows;
                out.extend_from_slice(&p.data[r * c..(r + 1) * c]);
            }
        }
        out
    }

    fn mlstm(c: &cpu::MLstm) -> MLstm {
        let s = 1.0 / (c.dqk as f32).sqrt();
        let scaled = |t: &Tensor| Tensor::new(t.dims(), t.data.iter().map(|v| v * s).collect());
        let (wk, bk) = (scaled(&c.wk), scaled(&c.bk));
        let inp = c.input_size;
        MLstm {
            heads: c.heads,
            dqk: c.dqk,
            dhv: c.dhv,
            qkvo: Linear {
                inp,
                out: 2 * c.heads * c.dqk + 2 * c.d,
                w: cat(&[&c.wq, &wk, &c.wv, &c.wo], inp),
                b: cat(&[&c.bq, &bk, &c.bv, &c.bo], 1),
            },
            gates: Linear {
                inp,
                out: 2 * c.heads,
                w: cat(&[&c.wi, &c.wf], inp),
                b: cat(&[&c.bi, &c.bf], 1),
            },
            out: Linear {
                inp: c.d,
                out: c.d,
                w: c.w_out.data.clone(),
                b: c.b_out.data.clone(),
            },
            gamma: c.gamma.data.clone(),
        }
    }

    fn block<C: cpu::block::Cell>(b: &cpu::Block<C>, cell: Cell) -> Block {
        Block {
            norm1: b.pre_norm1.gamma.data.clone(),
            cell,
            norm2: b.pre_norm2.gamma.data.clone(),
            gate: lin(&b.lin_gate),
            value: lin(&b.lin_value),
            down: lin(&b.lin_down),
        }
    }

    fn randomize(t: &mut Tensor, scale: f32) {
        let r = Tensor::random(t.dims(), scale);
        t.data.iter_mut().zip(&r.data).for_each(|(a, b)| *a += b);
    }

    /// Stepping a block one position at a time reproduces cpu's whole-sequence
    /// forward over `[B, T, H]`.
    fn check(mut cpu: impl FnMut(&Tensor) -> Tensor, blk: &Block) {
        let (b, t, h) = (3, 7, blk.hidden());
        let x = Tensor::random(&[b, t, h], 1.0);
        let want = cpu(&x);
        let mut st = blk.state(b);
        let mut got = vec![0.0; b * t * h];
        for s in 0..t {
            let mut row: Vec<f32> = (0..b)
                .flat_map(|r| x.data[(r * t + s) * h..(r * t + s + 1) * h].to_vec())
                .collect();
            blk.step(&mut row, b, &mut st);
            for r in 0..b {
                got[(r * t + s) * h..(r * t + s + 1) * h].copy_from_slice(&row[r * h..(r + 1) * h]);
            }
        }
        let err = got.iter().zip(&want.data).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(err < 1e-4, "max |step - sequence| = {err}");
    }

    #[test]
    fn slstm_block_steps_like_cpu() {
        let (h, up) = (16, up_of(16));
        let mut b = cpu::SLstmBlock::new_slstm(h, up);
        for t in [&mut b.cell.wz, &mut b.cell.wi, &mut b.cell.wf, &mut b.cell.wo] {
            randomize(t, 0.3);
        }
        let post = b.post_cell_norm.as_mut().unwrap();
        randomize(&mut post.gamma, 0.2);
        randomize(&mut b.pre_norm1.gamma, 0.2);
        let cell = Cell::S(slstm(&b.cell, &b.post_cell_norm.as_ref().unwrap().gamma.data));
        let blk = block(&b, cell);
        check(|x| b.forward(x), &blk);
    }

    #[test]
    fn mlstm_block_steps_like_cpu() {
        let (h, up) = (16, up_of(16));
        let mut b = cpu::MLstmBlock::new_mlstm(h, up, 4, 4);
        randomize(&mut b.cell.wi, 0.5);
        randomize(&mut b.cell.wf, 0.5);
        randomize(&mut b.cell.bi, 2.0);
        randomize(&mut b.cell.gamma, 0.2);
        let blk = block(&b, Cell::M(mlstm(&b.cell)));
        check(|x| b.forward(x), &blk);
    }

    #[test]
    fn greedy_is_argmax() {
        let logits = [0.1, 2.0, -1.0, 1.9];
        assert_eq!(sample_top_p(&logits, 0.0, 0.9), 1);
        assert_eq!(sample_top_p(&logits, 1e-4, 0.9), 1);
    }
}
