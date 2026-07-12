//! Device-resident hierarchical (HAT-style) model — GPU counterpart of
//! [`nn2::hierarchical::Hierarchical`](crate::nn2::hierarchical::Hierarchical).
//!
//! Three coupled stages, run phase-by-phase over a window of words:
//!
//!   1. encoder  — per word, `Embedding → sLSTM block×N`, read out `e_w` at the
//!                 closing `[W]` step. Words are the batch axis.
//!   2. backbone — `Linear → (sLSTM/mLSTM block)×N → Linear`, autoregressing over
//!                 the word embeddings as one sequence (batch 1, length = words).
//!   3. decoder  — per word, slot 0 is the injected backbone context, later slots
//!                 feed the previous char through the **tied** char table;
//!                 `sLSTM block×N → RMSNorm → head → SoftCap`.
//!
//! The decoder's pre-head RMSNorm is the **only** stage-level norm, matching
//! `model.rs::build_hierarchical_model` (the blocks keep their internal norms).
//!
//! Everything — the tied char table, every block, the projections, the norm and
//! the head, plus all gradients and AdamW moments — lives in `DTensor`s. Index
//! bookkeeping (which row is a `[W]` step, which slot is a char) is computed on
//! the host and uploaded as id lists; only tensor *data* stays on the device.
//!
//! Checkpoints: `save`/`load` write a `GHIR` blob — the config header followed by
//! every parameter in `params_mut` order.

use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};

use super::block::{Block, BlockLike};
use super::{DTensor, Gpu, linear::Linear, mlstm::MLstm, ops, rms_norm::RmsNorm, slstm::SLstm};
use crate::nn2::optim::AdamCfg;
use crate::tensor::Tensor;

const MAGIC: u32 = 0x4748_4952; // "GHIR"
const VERSION: u32 = 1;

/// Config for the hierarchical stack (mirrors `nn2::HierCfg`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HierCfg {
    pub vocab: usize,
    pub hc: usize, // char/context hidden (tied embedding + decoder width)
    pub wh: usize, // backbone width
    pub enc_blocks: usize,
    pub bb_blocks: usize, // alternates sLSTM (even) / mLSTM (odd)
    pub dec_blocks: usize,
    pub heads: usize, // mLSTM heads
    pub dqk: usize,
    pub w_token: usize,
    pub cap: f32,
}

/// SwiGLU inner width, derived per block from its own hidden width — `8·h/3`,
/// the paper default, exactly as `SequentialBuilder::{slstm,mlstm}_block` does.
/// It therefore differs between stages (e.g. 128 → 341, 384 → 1024).
#[inline]
pub fn up_of(hidden: usize) -> usize {
    hidden * 8 / 3
}

/// Per-word encoder: sLSTM blocks only, `e_w` read out at the `[W]` step.
pub struct WordEncoder {
    pub blocks: Vec<Box<dyn BlockLike>>,
}

impl WordEncoder {
    fn new(gpu: &Gpu, hc: usize, n: usize) -> Self {
        Self {
            blocks: (0..n)
                .map(|_| {
                    Box::new(Block::from_cell(gpu, hc, up_of(hc), SLstm::new_rand(gpu, hc, hc)))
                        as Box<dyn BlockLike>
                })
                .collect(),
        }
    }
}

pub struct Hierarchical {
    pub cfg: HierCfg,

    // Tied char table (encoder input + decoder char slots) + grad/moments.
    pub table: DTensor,
    dtable: DTensor,
    m_tbl: DTensor,
    v_tbl: DTensor,

    pub encoder: WordEncoder,

    pub bb_front: Linear,               // HC → WH
    pub bb_blocks: Vec<Box<dyn BlockLike>>, // WH
    pub bb_back: Linear,                // WH → HC (context)

    pub dec_blocks: Vec<Box<dyn BlockLike>>, // HC
    pub dec_norm: RmsNorm,                   // HC — the only stage-level norm
    pub dec_head: Linear,                    // HC → vocab

    /// Optimizer step count, persisted with the checkpoint so training resumes.
    pub step_count: usize,
}

impl Hierarchical {
    pub fn new(gpu: &Gpu, cfg: &HierCfg) -> Self {
        let bb_blocks: Vec<Box<dyn BlockLike>> = (0..cfg.bb_blocks)
            .map(|i| {
                if i % 2 == 0 {
                    Box::new(Block::from_cell(
                        gpu, cfg.wh, up_of(cfg.wh), SLstm::new_rand(gpu, cfg.wh, cfg.wh),
                    )) as Box<dyn BlockLike>
                } else {
                    Box::new(Block::from_cell(
                        gpu, cfg.wh, up_of(cfg.wh),
                        MLstm::new_rand(gpu, cfg.wh, cfg.wh, cfg.heads, cfg.dqk),
                    )) as Box<dyn BlockLike>
                }
            })
            .collect();
        let dec_blocks: Vec<Box<dyn BlockLike>> = (0..cfg.dec_blocks)
            .map(|_| {
                Box::new(Block::from_cell(
                    gpu, cfg.hc, up_of(cfg.hc), SLstm::new_rand(gpu, cfg.hc, cfg.hc),
                )) as Box<dyn BlockLike>
            })
            .collect();
        Self {
            cfg: *cfg,
            table: DTensor::from_host(gpu, &Tensor::random(&[cfg.vocab, cfg.hc], 0.02)),
            dtable: DTensor::zeros(gpu, &[cfg.vocab, cfg.hc]),
            m_tbl: DTensor::zeros(gpu, &[cfg.vocab, cfg.hc]),
            v_tbl: DTensor::zeros(gpu, &[cfg.vocab, cfg.hc]),
            encoder: WordEncoder::new(gpu, cfg.hc, cfg.enc_blocks),
            bb_front: Linear::new_rand(gpu, cfg.hc, cfg.wh),
            bb_blocks,
            bb_back: Linear::new_rand(gpu, cfg.wh, cfg.hc),
            dec_blocks,
            dec_norm: RmsNorm::new(gpu, cfg.hc),
            dec_head: Linear::new_rand(gpu, cfg.hc, cfg.vocab),
            step_count: 0,
        }
    }

    /// Every learnable tensor, in a fixed order. Save and load both walk this, so
    /// the order defines the checkpoint layout.
    fn params_mut(&mut self) -> Vec<&mut DTensor> {
        let mut v: Vec<&mut DTensor> = vec![&mut self.table];
        for b in self.encoder.blocks.iter_mut() {
            v.extend(b.params_mut());
        }
        v.extend(self.bb_front.params_mut());
        for b in self.bb_blocks.iter_mut() {
            v.extend(b.params_mut());
        }
        v.extend(self.bb_back.params_mut());
        for b in self.dec_blocks.iter_mut() {
            v.extend(b.params_mut());
        }
        v.extend(self.dec_norm.params_mut());
        v.extend(self.dec_head.params_mut());
        v
    }

    /// Forward + backward over one window; accumulates all grads and returns the
    /// mean decode cross-entropy. `tokens` are char ids; `words` are `(start,
    /// end)` char ranges. Word 0 is encode-only; words 1..n are decoded.
    pub fn forward_backward(&mut self, gpu: &Gpu, tokens: &[usize], words: &[(usize, usize)]) -> f32 {
        let loss = self.forward_backward_window(gpu, tokens, words);
        // The window's temporaries have dropped by now, so their `cuMemFreeAsync`
        // frees are queued on the stream. CUDA's stream-ordered pool only hands
        // that memory back at a synchronization point — without one it just keeps
        // reserving fresh blocks for every new window shape and grows without
        // bound. One sync per window is noise next to the window's own kernels.
        gpu.stream.synchronize().expect("stream sync");
        loss
    }

    fn forward_backward_window(
        &mut self,
        gpu: &Gpu,
        tokens: &[usize],
        words: &[(usize, usize)],
    ) -> f32 {
        // Phase timing, off unless GPU_PROF is set (each mark syncs the stream).
        let prof = std::env::var("GPU_PROF").is_ok();
        let mut t0 = std::time::Instant::now();
        let mut mark = |name: &str| {
            if prof {
                gpu.stream.synchronize().expect("sync");
                println!("  {name:<22} {:>8.1?}", t0.elapsed());
                t0 = std::time::Instant::now();
            }
        };
        let n = words.len();
        if n < 2 {
            return 0.0;
        }
        let dw = n - 1;
        let (hc, wh) = (self.cfg.hc, self.cfg.wh);
        let w_token = self.cfg.w_token;

        // ---- PHASE 1: ENCODER ----------------------------------------------
        let enc_lens: Vec<usize> = (0..dw).map(|w| words[w].1 - words[w].0).collect();
        let enc_tmax = enc_lens.iter().map(|&l| l + 1).max().unwrap();
        let mut enc_ids = vec![0usize; dw * enc_tmax];
        let mut readout_rows = vec![0usize; dw]; // row index of each word's [W] step
        for w in 0..dw {
            let (s, _) = words[w];
            for k in 0..enc_lens[w] {
                enc_ids[w * enc_tmax + k] = tokens[s + k];
            }
            enc_ids[w * enc_tmax + enc_lens[w]] = w_token;
            readout_rows[w] = w * enc_tmax + enc_lens[w];
        }

        let embedded = ops::embedding_gather(gpu, &self.table, &enc_ids, hc); // [dw*T, HC]
        let mut h = embedded.reshaped(&[dw, enc_tmax, hc]);
        for blk in self.encoder.blocks.iter_mut() {
            h = blk.forward(gpu, &h);
        }
        let h_flat = h.reshaped(&[dw * enc_tmax, hc]);
        // e_w = the [W]-step row of each word (a row gather from the flat matrix).
        let e_w = ops::embedding_gather(gpu, &h_flat, &readout_rows, hc); // [dw, HC]
        mark("encoder fwd");

        // ---- PHASE 2: BACKBONE ---------------------------------------------
        let bb_in = self.bb_front.forward(gpu, &e_w); // [dw, WH]
        let mut hb = bb_in.reshaped(&[1, dw, wh]);
        for (i, blk) in self.bb_blocks.iter_mut().enumerate() {
            hb = blk.forward(gpu, &hb);
            mark(if i % 2 == 0 { "  bb sLSTM fwd" } else { "  bb mLSTM fwd" });
        }
        let o = self.bb_back.forward(gpu, &hb.reshaped(&[dw, wh])); // [dw, HC]
        mark("backbone fwd");

        // ---- PHASE 3: DECODER ----------------------------------------------
        // Word w's decode target is word w+1. Slot 0 = o[w]; slot k = embed(prev char).
        let dec_lens: Vec<usize> = (0..dw).map(|w| words[w + 1].1 - words[w + 1].0).collect();
        let dec_tmax = dec_lens.iter().map(|&l| l + 1).max().unwrap();
        let rows = dw * dec_tmax;

        let mut o_rows = vec![0usize; dw]; // dest row of each word's slot 0
        let mut char_rows = Vec::new(); // dest rows of the char slots
        let mut char_ids = Vec::new(); // the char id feeding each of those slots
        let mut targets = vec![0usize; rows];
        let mut mask = vec![false; rows];
        for w in 0..dw {
            let m = dec_lens[w];
            let (s, _) = words[w + 1];
            o_rows[w] = w * dec_tmax;
            for k in 1..=m {
                char_rows.push(w * dec_tmax + k);
                char_ids.push(tokens[s + k - 1]);
            }
            for k in 0..m {
                targets[w * dec_tmax + k] = tokens[s + k];
                mask[w * dec_tmax + k] = true;
            }
            targets[w * dec_tmax + m] = w_token;
            mask[w * dec_tmax + m] = true;
        }

        // Build the decoder input: zeros, then scatter the context and char rows.
        let mut dec_in = DTensor::zeros(gpu, &[rows, hc]);
        ops::scatter_rows(gpu, &mut dec_in, &o, &o_rows);
        let char_vecs = ops::embedding_gather(gpu, &self.table, &char_ids, hc);
        ops::scatter_rows(gpu, &mut dec_in, &char_vecs, &char_rows);

        let mut hd = dec_in.reshaped(&[dw, dec_tmax, hc]);
        for blk in self.dec_blocks.iter_mut() {
            hd = blk.forward(gpu, &hd);
        }
        let hdn = self.dec_norm.forward(gpu, &hd.reshaped(&[rows, hc]));
        let logits = self.dec_head.forward(gpu, &hdn);
        let capped = ops::softcap_forward(gpu, &logits, self.cfg.cap);

        let (loss, d_capped) = ops::masked_softmax_cross_entropy(gpu, &capped, &targets, &mask);
        mark("decoder fwd + loss");

        // ---- BACKWARD ------------------------------------------------------
        let d_logits = ops::softcap_backward(gpu, &d_capped, &capped, self.cfg.cap);
        let d_hdn = self.dec_head.backward(gpu, &d_logits);
        let d_hd_flat = self.dec_norm.backward(gpu, &d_hdn);
        let mut d_hd = d_hd_flat.reshaped(&[dw, dec_tmax, hc]);
        for blk in self.dec_blocks.iter_mut().rev() {
            d_hd = blk.backward(gpu, &d_hd);
        }
        mark("decoder bwd");
        let d_dec_in = d_hd.reshaped(&[rows, hc]);
        // Slot 0 rows → d_o; char-slot rows → tied table (gather then scatter-add).
        let d_o = ops::embedding_gather(gpu, &d_dec_in, &o_rows, hc); // [dw, HC]
        let d_char = ops::embedding_gather(gpu, &d_dec_in, &char_rows, hc);
        ops::embedding_scatter_add(gpu, &mut self.dtable, &char_ids, &d_char, hc);

        // Backbone backward.
        let d_bb_out = self.bb_back.backward(gpu, &d_o); // [dw, WH]
        let mut d_hb = d_bb_out.reshaped(&[1, dw, wh]);
        for (i, blk) in self.bb_blocks.iter_mut().enumerate().rev() {
            d_hb = blk.backward(gpu, &d_hb);
            mark(if i % 2 == 0 { "  bb sLSTM bwd" } else { "  bb mLSTM bwd" });
        }
        let d_e_w = self.bb_front.backward(gpu, &d_hb.reshaped(&[dw, wh])); // [dw, HC]
        mark("backbone bwd");

        // Encoder backward: scatter d_e_w onto the [W]-step rows, rest zero.
        let mut d_h = DTensor::zeros(gpu, &[dw * enc_tmax, hc]);
        ops::scatter_rows(gpu, &mut d_h, &d_e_w, &readout_rows);
        let mut d_h = d_h.reshaped(&[dw, enc_tmax, hc]);
        for blk in self.encoder.blocks.iter_mut().rev() {
            d_h = blk.backward(gpu, &d_h);
        }
        let d_embedded = d_h.reshaped(&[dw * enc_tmax, hc]);
        ops::embedding_scatter_add(gpu, &mut self.dtable, &enc_ids, &d_embedded, hc);
        mark("encoder bwd");

        loss
    }

    /// AdamW across every stage. Tied table and the logit head are undecayed;
    /// interior projections decay (matching the project's optimizer convention).
    pub fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        ops::adamw(gpu, &mut self.table, &self.dtable, &mut self.m_tbl, &mut self.v_tbl, cfg, false);
        self.dtable.zero_(gpu);
        for b in self.encoder.blocks.iter_mut() {
            b.step(gpu, cfg);
        }
        self.bb_front.step(gpu, cfg);
        for b in self.bb_blocks.iter_mut() {
            b.step(gpu, cfg);
        }
        self.bb_back.step(gpu, cfg);
        for b in self.dec_blocks.iter_mut() {
            b.step(gpu, cfg);
        }
        self.dec_norm.step(gpu, cfg);
        self.dec_head.step_wd(gpu, cfg, false); // logit head: no weight decay
        self.step_count += 1;
    }

    // --- checkpointing ------------------------------------------------------

    /// Write a `GHIR` checkpoint: config header, step count, then every parameter
    /// in `params_mut` order. Weights only (Adam moments are not persisted, so a
    /// resumed run restarts the moment estimates — same as the CPU system).
    pub fn save(&mut self, gpu: &Gpu, path: &str) -> io::Result<()> {
        if let Some(dir) = std::path::Path::new(path).parent() {
            fs::create_dir_all(dir)?;
        }
        let tmp = format!("{path}.tmp");
        {
            let mut w = BufWriter::new(File::create(&tmp)?);
            w.write_all(&MAGIC.to_le_bytes())?;
            w.write_all(&VERSION.to_le_bytes())?;
            let c = self.cfg;
            for v in [
                c.vocab, c.hc, c.wh, c.enc_blocks, c.bb_blocks, c.dec_blocks, c.heads,
                c.dqk, c.w_token,
            ] {
                w.write_all(&(v as u32).to_le_bytes())?;
            }
            w.write_all(&c.cap.to_le_bytes())?;
            w.write_all(&(self.step_count as u64).to_le_bytes())?;

            let params = self.params_mut();
            w.write_all(&(params.len() as u32).to_le_bytes())?;
            for p in params {
                let host = p.to_host(gpu);
                w.write_all(&(host.rank as u32).to_le_bytes())?;
                for d in host.dims() {
                    w.write_all(&(*d as u32).to_le_bytes())?;
                }
                for v in &host.data {
                    w.write_all(&v.to_le_bytes())?;
                }
            }
            w.flush()?;
        }
        // Rename only after a complete write, so a crash can't leave a torn file.
        fs::rename(&tmp, path)
    }

    /// Load a `GHIR` checkpoint, rebuilding the model from its stored config.
    pub fn load(gpu: &Gpu, path: &str) -> io::Result<Self> {
        let mut r = BufReader::new(File::open(path)?);
        let mut u32b = [0u8; 4];
        let mut rd_u32 = |r: &mut BufReader<File>| -> io::Result<u32> {
            r.read_exact(&mut u32b)?;
            Ok(u32::from_le_bytes(u32b))
        };
        if rd_u32(&mut r)? != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not a GHIR checkpoint"));
        }
        let ver = rd_u32(&mut r)?;
        if ver != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("GHIR version {ver} != {VERSION}"),
            ));
        }
        let mut f = [0usize; 9];
        for slot in f.iter_mut() {
            *slot = rd_u32(&mut r)? as usize;
        }
        let cap = f32::from_le_bytes({
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            b
        });
        let step_count = {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            u64::from_le_bytes(b) as usize
        };
        let cfg = HierCfg {
            vocab: f[0], hc: f[1], wh: f[2], enc_blocks: f[3], bb_blocks: f[4],
            dec_blocks: f[5], heads: f[6], dqk: f[7], w_token: f[8], cap,
        };

        let mut model = Hierarchical::new(gpu, &cfg);
        model.step_count = step_count;

        let count = rd_u32(&mut r)? as usize;
        let mut params = model.params_mut();
        if count != params.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("checkpoint has {count} params, model expects {}", params.len()),
            ));
        }
        for p in params.iter_mut() {
            let rank = rd_u32(&mut r)? as usize;
            let mut dims = Vec::with_capacity(rank);
            for _ in 0..rank {
                dims.push(rd_u32(&mut r)? as usize);
            }
            if dims != p.dims() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("param shape {dims:?} != model {:?}", p.dims()),
                ));
            }
            let n: usize = dims.iter().product();
            let mut data = vec![0f32; n];
            let mut buf = [0u8; 4];
            for v in data.iter_mut() {
                r.read_exact(&mut buf)?;
                *v = f32::from_le_bytes(buf);
            }
            **p = DTensor::from_host(gpu, &Tensor::new(&dims, data));
        }
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The GPU hierarchical stack must actually learn: memorize one tiny window,
    /// driving the decode loss down. Exercises the full wiring — tied char table
    /// (encoder input + decoder char slots), backbone context injection at the
    /// decoder's slot 0, the [W]-step readout, masked CE and AdamW across stages.
    /// Then round-trip a checkpoint and confirm the loss is unchanged.
    #[test]
    fn hierarchical_memorizes_and_checkpoints() {
        let Some(gpu) = super::super::test_gpu() else { return };
        let cfg = HierCfg {
            vocab: 9, hc: 16, wh: 24,
            enc_blocks: 1, bb_blocks: 2, dec_blocks: 1,
            heads: 2, dqk: 8, w_token: 8, cap: 30.0,
        };
        let mut model = Hierarchical::new(&gpu, &cfg);

        let tokens = vec![1usize, 2, 3, 4, 5, 6, 7, 1, 2, 3];
        let words = vec![(0, 3), (3, 5), (5, 8), (8, 10)];

        let mut opt = AdamCfg::new(5e-3, 0.0);
        let first = model.forward_backward(&gpu, &tokens, &words);
        for _ in 0..250 {
            let _ = model.forward_backward(&gpu, &tokens, &words);
            opt.t += 1;
            model.step(&gpu, &opt);
        }
        let last = model.forward_backward(&gpu, &tokens, &words);
        assert!(last < first * 0.4, "decode loss did not fall: {first} -> {last}");

        // Checkpoint round-trip: reloading must reproduce the exact same loss.
        let path = std::env::temp_dir().join("gpu_hier_test.ghir");
        let path = path.to_str().unwrap();
        model.save(&gpu, path).expect("save");
        let mut back = Hierarchical::load(&gpu, path).expect("load");
        assert_eq!(back.cfg, cfg, "config did not survive the round-trip");
        assert_eq!(back.step_count, model.step_count, "step count lost");
        let reloaded = back.forward_backward(&gpu, &tokens, &words);
        assert!(
            (reloaded - last).abs() < 1e-4,
            "reloaded model gives a different loss: {last} -> {reloaded}"
        );
        let _ = std::fs::remove_file(path);
    }
}
