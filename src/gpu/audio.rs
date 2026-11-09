//! Device-side audio encoder: speech features in, one embedding per spoken word
//! out, in the space the backbone reads text words in.
//!
//!   1. stem     — `stack` feature frames gathered per encoder frame, every
//!                 `stride` frames, `→ Linear`: a causal strided convolution over
//!                 time with a full-frequency kernel.
//!   2. frames   — blocks over each utterance (mLSTM, every fourth an sLSTM),
//!                 causal, utterances as the batch axis.
//!   3. boundary — `Linear → delay + 1` logits per encoder frame
//!                 (see [`BoundaryTargets`](crate::audio::segments::BoundaryTargets)):
//!                 it may wait up to `delay` frames to decide where a word ended.
//!   4. ctc      — `Linear → per_frame · alphabet` character logits per encoder
//!                 frame (`per_frame` CTC steps each, since speech has more
//!                 characters per second than 25 Hz has frames). Trains on the
//!                 transcript alone; how much of its gradient reaches the frames is
//!                 the caller's choice, down to none (a probe).
//!   5. segments — every word's own frames, nothing after them, as one row of a
//!                 batch of words, from a zero state, followed by a learned `[W]`
//!                 step like the text encoder's (absent in pre-AEN4 checkpoints,
//!                 which read the last frame); that step `→ Linear` plus a
//!                 `Linear` of the word's timing,
//!                 `→ RMSNorm` with a frozen scale, is its embedding. Frames past
//!                 the word would carry the next word into the text model, which
//!                 is asked to predict it. The scale is the text encoder's own
//!                 per-dimension RMS: left free, the embeddings grow ~10x past
//!                 anything the backbone read in training, which still predicts
//!                 the next word but derails it once text follows.
//!
//! The output projection and its bias are the audio path's own, which is all a
//! modality marker in front of the backbone would add.
//!
//! Host-side layout (stem ids, segment rows, targets) is in [`crate::audio`].

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};

use cudarc::driver::CudaSlice;

use super::arena::{self, ParamKind, ParamSlot, TrainingCache};
use super::block::{Block, BlockLike, Resets};
use super::{
    GTensor, Gpu, linear::Linear, mlstm::MLstm, ops, rms_norm::RmsNorm, slstm::SLstm, temp,
};
use crate::audio::ctc::CtcAlphabet;
use crate::audio::frontend::FeatureStats;
use crate::audio::segments::{SegmentRows, StemLayout, WordSegments};
use crate::config::{
    AUDIO_BLOCKS, AUDIO_DQK, AUDIO_FEATS, AUDIO_HEADS, AUDIO_HIDDEN, AUDIO_SEG_MAX_FRAMES, AUDIO_SEG_MAX_ROWS,
    AUDIO_SEG_STACK, AUDIO_STACK, AUDIO_STRIDE, BOUNDARY_DELAY, CTC_PER_FRAME, up_of,
};
use crate::bytes::{read_f32_vec, read_u32, read_u64, write_f32_slice, write_u32, write_u64};
use crate::cpu::optim::AdamCfg;
use crate::tensor::Tensor;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AudioCfg {
    /// Features per input frame.
    pub feats: usize,
    /// Input frames per stem row, and how many it advances.
    pub stack: usize,
    pub stride: usize,
    pub hidden: usize,
    pub blocks: usize,
    pub heads: usize,
    pub dqk: usize,
    pub seg_blocks: usize,
    /// Encoder frames the boundary head may wait before deciding a word ended.
    pub delay: usize,
    /// Width of the embeddings produced: the text encoder's `hc`.
    pub out: usize,
    /// Character logits per encoder frame: CTC steps per frame × alphabet.
    pub ctc: usize,
    /// Bit `i` set: segment block `i` is an sLSTM, else an mLSTM.
    pub seg_slstm: usize,
    /// The segment encoder reads a word at a `[W]` step after its frames.
    pub seg_word: bool,
    /// Frames of a word the segment encoder reads, counted back from its end.
    pub seg_max: usize,
}

impl AudioCfg {
    /// The configured encoder, producing embeddings `out` wide.
    /// Frames of each word the segment encoder reads in a batch of `words`:
    /// `seg_max`, unless that would pad the batch past `AUDIO_SEG_MAX_ROWS`.
    pub fn frames_per_word(&self, words: usize) -> usize {
        let steps = AUDIO_SEG_MAX_ROWS / words.max(1);
        self.seg_max.min(steps.saturating_sub(self.seg_word as usize)).max(1)
    }

    pub fn from_config(out: usize) -> Self {
        let seg_slstm = AUDIO_SEG_STACK
            .bytes()
            .enumerate()
            .map(|(i, c)| match c {
                b's' => 1 << i,
                b'm' => 0,
                _ => panic!("AUDIO_SEG_STACK: '{}' is neither s nor m", c as char),
            })
            .sum();
        Self {
            feats: AUDIO_FEATS,
            stack: AUDIO_STACK,
            stride: AUDIO_STRIDE,
            hidden: AUDIO_HIDDEN,
            blocks: AUDIO_BLOCKS,
            heads: AUDIO_HEADS,
            dqk: AUDIO_DQK,
            seg_blocks: AUDIO_SEG_STACK.len(),
            delay: BOUNDARY_DELAY,
            out,
            ctc: CTC_PER_FRAME * CtcAlphabet::SIZE,
            seg_slstm,
            seg_word: true,
            seg_max: AUDIO_SEG_MAX_FRAMES,
        }
    }

    pub fn stem(&self) -> StemLayout {
        StemLayout {
            stack: self.stack,
            stride: self.stride,
        }
    }
}

/// "AEN4": an audio encoder checkpoint.
const MAGIC: u32 = 0x4145_4E34;
/// "AEN3": one from before the segment layout fields, whose segment encoder is
/// mLSTMs but every fourth, reads all of a word's frames and has no `[W]`.
const MAGIC_NO_SEG_FIELDS: u32 = 0x4145_4E33;
/// "AEN2": one from before the output scale, which loads at γ = 1.
const MAGIC_WITH_CTC: u32 = 0x4145_4E32;
/// "AEN1": one from before the CTC head too, which loads fresh.
const MAGIC_NO_CTC: u32 = 0x4145_4E31;

/// What a checkpoint carries besides the weights.
#[derive(Clone, Debug)]
pub struct AudioCheckpoint {
    /// Optimizer steps taken.
    pub step: u64,
    /// Seconds of audio trained on.
    pub seconds: f64,
    /// The feature normalization the weights were trained under — part of the
    /// model, since other statistics would feed it other inputs.
    pub stats: FeatureStats,
}

impl AudioCfg {
    fn fields(&self) -> [usize; 14] {
        [
            self.feats,
            self.stack,
            self.stride,
            self.hidden,
            self.blocks,
            self.heads,
            self.dqk,
            self.seg_blocks,
            self.delay,
            self.out,
            self.ctc,
            self.seg_slstm,
            self.seg_word as usize,
            self.seg_max,
        ]
    }

    fn from_fields(f: [usize; 14]) -> Self {
        let [
            feats,
            stack,
            stride,
            hidden,
            blocks,
            heads,
            dqk,
            seg_blocks,
            delay,
            out,
            ctc,
            seg_slstm,
            seg_word,
            seg_max,
        ] = f;
        Self {
            feats,
            stack,
            stride,
            hidden,
            blocks,
            heads,
            dqk,
            seg_blocks,
            delay,
            out,
            ctc,
            seg_slstm,
            seg_word: seg_word != 0,
            seg_max,
        }
    }
}

/// What one forward hands back.
pub struct AudioOut {
    /// Boundary logits, `[batch · t_out, delay + 1]`.
    pub boundary: GTensor<f32>,
    /// Character logits, `[batch · t_out, ctc]`: per utterance, row-major, the
    /// `[t_out · per_frame, alphabet]` CTC input.
    pub ctc: GTensor<f32>,
    /// One row per segment, utterance by utterance in segment order,
    /// `[segments, out]`. `None` for a batch without segments.
    pub embeddings: Option<GTensor<f32>>,
    /// Encoder frames per utterance.
    pub t_out: usize,
}

/// The batch's layout, kept on the device from forward to backward.
struct Saved {
    batch: usize,
    t_out: usize,
    seg: Option<SegSaved>,
}

struct SegSaved {
    /// The embeddings, which the output norm's backward reads.
    out: GTensor<f32>,
    rows: usize,
    steps: usize,
    n_seg: usize,
    /// Row of the frame table every segment step reads.
    ids: CudaSlice<u32>,
    /// Segment step every embedding is read at.
    last: CudaSlice<u32>,
}

pub struct AudioEncoder {
    pub cfg: AudioCfg,
    pub stem: Linear,
    pub frame_blocks: Vec<Box<dyn BlockLike>>,
    pub boundary: Linear,
    pub seg_blocks: Vec<Box<dyn BlockLike>>,
    /// The `[W]` step's input, `[hidden]`; unused without `cfg.seg_word`.
    pub word: GTensor<f32>,
    pub d_word: GTensor<f32>,
    word_m: GTensor<f32>,
    word_v: GTensor<f32>,
    pub seg_out: Linear,
    pub seg_timing: Linear,
    pub ctc: Linear,
    pub out_norm: RmsNorm,
    /// Parts the loaded checkpoint did not have, at their initialization.
    pub missing: Vec<&'static str>,
    saved: Option<Saved>,
}

impl AudioEncoder {
    pub fn new(gpu: &Gpu, cfg: AudioCfg) -> Self {
        let h = cfg.hidden;
        let block = |slstm: bool| -> Box<dyn BlockLike> {
            if slstm {
                Box::new(Block::from_cell(
                    gpu,
                    h,
                    up_of(h),
                    SLstm::new_rand(gpu, h, h),
                ))
            } else {
                Box::new(Block::from_cell(
                    gpu,
                    h,
                    up_of(h),
                    MLstm::new_rand(gpu, h, h, cfg.heads, cfg.dqk),
                ))
            }
        };
        let mut boundary = Linear::new_rand(gpu, h, cfg.delay + 1);
        boundary.set_no_decay();
        let mut ctc = Linear::new_rand(gpu, h, cfg.ctc);
        ctc.set_no_decay();
        ctc.set_fp32();
        Self {
            cfg,
            stem: Linear::new_rand(gpu, cfg.stack * cfg.feats, h),
            frame_blocks: (0..cfg.blocks).map(|i| block(i % 4 == 3)).collect(),
            boundary,
            seg_blocks: (0..cfg.seg_blocks)
                .map(|i| block(cfg.seg_slstm >> i & 1 == 1))
                .collect(),
            word: GTensor::from_host(gpu, &Tensor::random(&[h], 1.0)),
            d_word: GTensor::zeros(gpu, &[h]),
            word_m: arena::unbacked(gpu),
            word_v: arena::unbacked(gpu),
            seg_out: Linear::new_rand(gpu, h, cfg.out),
            seg_timing: Linear::new_rand(gpu, 2, cfg.out),
            ctc,
            out_norm: {
                let mut n = RmsNorm::new(gpu, cfg.out);
                n.freeze();
                n
            },
            missing: Vec::new(),
            saved: None,
        }
    }

    /// Window scratch for batches of at most `frame_rows` encoder frames
    /// (`batch · t_out`) and `t_out` frames per utterance, with at most `words`
    /// segments.
    pub fn training_cache(
        &self,
        gpu: &Gpu,
        frame_rows: usize,
        t_out: usize,
        words: usize,
    ) -> TrainingCache {
        let c = self.cfg;
        let l = crate::config::MLSTM_CHUNK.min(ops::FUSED_MAX_L);
        let dhv = c.hidden / c.heads;
        let (seg_rows, steps) = SegmentRows::bound(words, t_out, c.seg_max, c.seg_word);
        let seg_rows = seg_rows.min(AUDIO_SEG_MAX_ROWS.max(steps));
        // The fullest batch: every word at the fewest steps the cap leaves.
        let fewest = c.frames_per_word(words).min(t_out) + c.seg_word as usize;
        TrainingCache::new(
            gpu,
            temp::widest(
                frame_rows.max(seg_rows),
                c.hidden,
                c.heads,
                c.dqk,
                c.stack * c.feats,
            ),
            temp::widest_small(frame_rows.max(seg_rows), c.heads),
            temp::widest_chunk(frame_rows, t_out, c.heads, c.dqk, dhv, l)
                .max(temp::widest_chunk(seg_rows, steps, c.heads, c.dqk, dhv, l))
                .max(temp::widest_chunk(words * fewest, fewest, c.heads, c.dqk, dhv, l)),
        )
    }

    /// Forward over `batch` utterances of `t_in` feature frames each
    /// (`feats` is `[batch, t_in, cfg.feats]` row-major, already normalized), with
    /// each utterance's word segments in encoder frames. Saves what
    /// [`backward`](Self::backward) needs; call [`discard`](Self::discard)
    /// instead when there will be none.
    pub fn forward(
        &mut self,
        gpu: &Gpu,
        feats: &[f32],
        batch: usize,
        t_in: usize,
        segments: &[WordSegments],
        cache: &TrainingCache,
    ) -> AudioOut {
        self.forward_aligned(gpu, feats, batch, t_in, cache, |_, _, _| segments.to_vec()).0
    }

    /// [`forward`](Self::forward) with the word segments chosen from this
    /// forward's own heads: `segments` gets the character logits
    /// (`[batch · t_out, ctc]`) and the boundary logits (`[batch · t_out,
    /// delay + 1]`) on the host with `t_out`, before the words are pooled. The
    /// character logits come back beside the output.
    pub fn forward_aligned(
        &mut self,
        gpu: &Gpu,
        feats: &[f32],
        batch: usize,
        t_in: usize,
        cache: &TrainingCache,
        segments: impl FnOnce(&[f32], &[f32], usize) -> Vec<WordSegments>,
    ) -> (AudioOut, Vec<f32>) {
        let c = self.cfg;
        assert_eq!(
            feats.len(),
            batch * t_in * c.feats,
            "AudioEncoder::forward — feature shape"
        );
        let t_out = c.stem().frames(t_in);
        assert!(t_out > 0, "AudioEncoder::forward — shorter than one stride");
        let n = batch * t_out;

        // The stem's padding row sits after the last real one and stays zero.
        let mut table = Vec::with_capacity((batch * t_in + 1) * c.feats);
        table.extend_from_slice(feats);
        table.resize((batch * t_in + 1) * c.feats, 0.0);
        let table = GTensor::from_host(gpu, &Tensor::new(&[batch * t_in + 1, c.feats], table));
        let ids = c.stem().ids(batch, t_in, (batch * t_in) as u32);
        let dids = ops::upload_ids_u32(gpu, &ids);
        let mut stacked =
            ops::embedding_gather_u32(gpu, &table, &dids.slice(..), ids.len(), c.feats);
        stacked.reshape_to(&[n, c.stack * c.feats]);

        let mut x = GTensor::uninit(gpu, &[n, c.hidden]);
        self.stem.forward(gpu, &stacked, &mut x);
        x.reshape_to(&[batch, t_out, c.hidden]);
        let whole = Resets::default();
        for blk in self.frame_blocks.iter_mut() {
            blk.set_carry(false);
            blk.reset_state(gpu);
            blk.set_resets(&whole);
            let mut y = GTensor::uninit(gpu, &[batch, t_out, c.hidden]);
            blk.forward(gpu, &x, &mut y, cache);
            x = y;
        }
        x.reshape_to(&[n, c.hidden]);

        let mut boundary = GTensor::uninit(gpu, &[n, c.delay + 1]);
        self.boundary.forward(gpu, &x, &mut boundary);
        let mut ctc = GTensor::uninit(gpu, &[n, c.ctc]);
        self.ctc.forward(gpu, &x, &mut ctc);
        let ctc_host = ctc.to_host(gpu).data;
        let boundary_host = boundary.to_host(gpu).data;
        let segments = segments(&ctc_host, &boundary_host, t_out);
        assert_eq!(
            segments.len(),
            batch,
            "AudioEncoder::forward — one segment list per utterance"
        );
        let segments = &segments[..];

        let words: usize = segments.iter().map(|u| u.len()).sum();
        let rows = SegmentRows::new(segments, t_out, c.frames_per_word(words), c.seg_word);
        let (embeddings, seg) = if rows.segments() == 0 {
            (None, None)
        } else {
            let (e, s) = self.segments_forward(gpu, &x, n, &rows, segments, cache);
            (Some(e), Some(s))
        };
        self.saved = Some(Saved { batch, t_out, seg });
        let out = AudioOut {
            boundary,
            ctc,
            embeddings,
            t_out,
        };
        (out, ctc_host)
    }

    /// `frames` is `[n, hidden]`; the segment steps read it and, at row `n`, the
    /// `[W]` input (zero without one, where only padding reads it).
    fn segments_forward(
        &mut self,
        gpu: &Gpu,
        frames: &GTensor<f32>,
        n: usize,
        rows: &SegmentRows,
        segments: &[WordSegments],
        cache: &TrainingCache,
    ) -> (GTensor<f32>, SegSaved) {
        let c = self.cfg;
        let (r, steps) = (rows.ids.len(), rows.steps);
        let n_seg = rows.segments();
        let table = GTensor::uninit(gpu, &[n + 1, c.hidden]);
        GTensor::view(gpu, &table.buf, 0, &[n, c.hidden]).copy_from(gpu, frames);
        let mut pad = GTensor::view(gpu, &table.buf, n * c.hidden, &[c.hidden]);
        if c.seg_word {
            pad.copy_from(gpu, &self.word);
        } else {
            pad.zero_(gpu);
        }
        let ids = ops::upload_ids_u32(gpu, &rows.ids);
        let mut x = ops::embedding_gather_u32(gpu, &table, &ids.slice(..), r, c.hidden);
        x.reshape_to(&[n_seg, steps, c.hidden]);

        let whole = Resets::default();
        for blk in self.seg_blocks.iter_mut() {
            blk.set_carry(false);
            blk.reset_state(gpu);
            blk.set_resets(&whole);
            let mut y = GTensor::uninit(gpu, &[n_seg, steps, c.hidden]);
            blk.forward(gpu, &x, &mut y, cache);
            x = y;
        }
        x.reshape_to(&[r, c.hidden]);

        let last = ops::upload_ids_u32(gpu, &rows.last);
        let ends = ops::embedding_gather_u32(gpu, &x, &last.slice(..), n_seg, c.hidden);
        let mut e = GTensor::uninit(gpu, &[n_seg, c.out]);
        self.seg_out.forward(gpu, &ends, &mut e);
        let timing: Vec<f32> = segments.iter().flat_map(WordSegments::timing).collect();
        let timing = GTensor::from_host(gpu, &Tensor::new(&[n_seg, 2], timing));
        let mut te = GTensor::uninit(gpu, &[n_seg, c.out]);
        self.seg_timing.forward(gpu, &timing, &mut te);
        ops::add_assign(gpu, &mut e, &te);
        let mut y = GTensor::uninit(gpu, &[n_seg, c.out]);
        self.out_norm.forward(gpu, &e, &mut y);

        let saved = SegSaved {
            out: y.dup(gpu),
            rows: r,
            steps,
            n_seg,
            ids,
            last,
        };
        (y, saved)
    }

    /// Backward of the last [`forward`](Self::forward): `d_boundary` is
    /// `[batch · t_out, delay + 1]`, `d_embeddings` `[segments, out]` (`None`
    /// trains the boundary head and frame stack alone), `d_ctc` the character
    /// logits' gradient with the share of it the frames receive — the head
    /// itself always learns from all of it. Accumulates into every parameter's
    /// gradient.
    pub fn backward(
        &mut self,
        gpu: &Gpu,
        d_boundary: &GTensor<f32>,
        d_embeddings: Option<&GTensor<f32>>,
        d_ctc: Option<(&GTensor<f32>, f32)>,
        cache: &TrainingCache,
    ) {
        let Saved { batch, t_out, seg } = self
            .saved
            .take()
            .expect("AudioEncoder::backward without a forward");
        let c = self.cfg;
        let n = batch * t_out;

        let mut dx = GTensor::uninit(gpu, &[n, c.hidden]);
        self.boundary.backward(gpu, d_boundary, &mut dx, cache);
        if let Some((d, share)) = d_ctc {
            let mut d_frames = GTensor::uninit(gpu, &[n, c.hidden]);
            self.ctc.backward(gpu, d, &mut d_frames, cache);
            if share != 0.0 {
                ops::scale_(gpu, &mut d_frames, share);
                ops::add_assign(gpu, &mut dx, &d_frames);
            }
        }
        match (seg, d_embeddings) {
            (Some(s), Some(de)) => {
                let d_rows = self.segments_backward(gpu, &s, de, cache);
                let mut d_table = GTensor::zeros(gpu, &[n + 1, c.hidden]);
                ops::embedding_scatter_add_u32(
                    gpu,
                    &mut d_table,
                    &s.ids.slice(..),
                    s.rows,
                    &d_rows,
                    c.hidden,
                );
                ops::add_assign(gpu, &mut dx, &GTensor::view(gpu, &d_table.buf, 0, &[n, c.hidden]));
                if c.seg_word {
                    let d_word = GTensor::view(gpu, &d_table.buf, n * c.hidden, &[c.hidden]);
                    ops::add_assign(gpu, &mut self.d_word, &d_word);
                }
            }
            (Some(_), None) => {
                for blk in self.seg_blocks.iter_mut() {
                    blk.drop_saved_act();
                }
            }
            (None, _) => {}
        }

        dx.reshape_to(&[batch, t_out, c.hidden]);
        for blk in self.frame_blocks.iter_mut().rev() {
            let mut d_in = GTensor::uninit(gpu, &[batch, t_out, c.hidden]);
            blk.backward(gpu, &dx, &mut d_in, cache);
            dx = d_in;
        }
        dx.reshape_to(&[n, c.hidden]);
        // The stem's input is data; its gradient is computed and dropped.
        let mut d_stacked = GTensor::uninit(gpu, &[n, c.stack * c.feats]);
        self.stem.backward(gpu, &dx, &mut d_stacked, cache);
    }

    /// Gradient with respect to the gathered segment rows, `[rows, hidden]`.
    fn segments_backward(
        &mut self,
        gpu: &Gpu,
        s: &SegSaved,
        de: &GTensor<f32>,
        cache: &TrainingCache,
    ) -> GTensor<f32> {
        let c = self.cfg;
        let mut d_raw = GTensor::uninit(gpu, &[s.n_seg, c.out]);
        self.out_norm.backward(gpu, de, &s.out, &mut d_raw, cache);
        let mut d_timing = GTensor::uninit(gpu, &[s.n_seg, 2]);
        self.seg_timing.backward(gpu, &d_raw, &mut d_timing, cache);
        let mut d_ends = GTensor::uninit(gpu, &[s.n_seg, c.hidden]);
        self.seg_out.backward(gpu, &d_raw, &mut d_ends, cache);

        let mut dx = GTensor::zeros(gpu, &[s.rows, c.hidden]);
        ops::embedding_scatter_add_u32(gpu, &mut dx, &s.last.slice(..), s.n_seg, &d_ends, c.hidden);
        dx.reshape_to(&[s.n_seg, s.steps, c.hidden]);
        let whole = Resets::default();
        for blk in self.seg_blocks.iter_mut().rev() {
            blk.set_resets(&whole);
            blk.reset_bptt(gpu);
            let mut d_in = GTensor::uninit(gpu, &[s.n_seg, s.steps, c.hidden]);
            blk.backward(gpu, &dx, &mut d_in, cache);
            dx = d_in;
        }
        dx.reshape_to(&[s.rows, c.hidden]);
        dx
    }

    /// Drop what the last forward saved, for a forward that gets no backward.
    pub fn discard(&mut self) {
        if let Some(saved) = self.saved.take() {
            for blk in self.frame_blocks.iter_mut() {
                blk.drop_saved_act();
            }
            if saved.seg.is_some() {
                for blk in self.seg_blocks.iter_mut() {
                    blk.drop_saved_act();
                }
            }
        }
    }

    /// Every parameter with its gradient and AdamW moments, stem to output.
    pub fn param_slots(&mut self) -> Vec<ParamSlot<'_>> {
        let mut slots = self.stem.param_slots();
        for blk in self.frame_blocks.iter_mut() {
            slots.extend(blk.param_slots());
        }
        slots.extend(self.boundary.param_slots());
        for blk in self.seg_blocks.iter_mut() {
            slots.extend(blk.param_slots());
        }
        slots.extend(self.seg_out.param_slots());
        slots.extend(self.seg_timing.param_slots());
        // Last, newest last, so an older checkpoint fills everything before them.
        slots.extend(self.ctc.param_slots());
        slots.extend(self.out_norm.param_slots());
        if self.cfg.seg_word {
            slots.push(ParamSlot::new(
                &mut self.word,
                &mut self.d_word,
                &mut self.word_m,
                &mut self.word_v,
                ParamKind::NoDecay,
            ));
        }
        slots
    }

    pub fn zero_grad(&mut self, gpu: &Gpu) {
        self.stem.zero_grad(gpu);
        for blk in self
            .frame_blocks
            .iter_mut()
            .chain(self.seg_blocks.iter_mut())
        {
            blk.zero_grad(gpu);
        }
        self.boundary.zero_grad(gpu);
        self.seg_out.zero_grad(gpu);
        self.seg_timing.zero_grad(gpu);
        self.ctc.zero_grad(gpu);
        self.out_norm.zero_grad(gpu);
        self.d_word.zero_(gpu);
    }

    /// AdamW over this encoder on its own, then clear its gradients.
    pub fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        arena::step_slots(gpu, &mut self.param_slots(), cfg);
    }

    /// Writes the weights in [`param_slots`](Self::param_slots) order, through a
    /// temporary file so an interrupted save leaves the previous one intact.
    /// The optimizer moments are not saved.
    pub fn save(&mut self, gpu: &Gpu, path: &str, ck: &AudioCheckpoint) -> io::Result<()> {
        let tmp = format!("{path}.tmp");
        {
            let mut w = BufWriter::new(File::create(&tmp)?);
            write_u32(&mut w, MAGIC)?;
            for f in self.cfg.fields() {
                write_u32(&mut w, f as u32)?;
            }
            write_u64(&mut w, ck.step)?;
            write_u64(&mut w, ck.seconds.to_bits())?;
            write_f32_slice(&mut w, &ck.stats.mean)?;
            write_f32_slice(&mut w, &ck.stats.std)?;
            let slots = self.param_slots();
            write_u32(&mut w, slots.len() as u32)?;
            for slot in slots {
                write_f32_slice(&mut w, &slot.param.to_host(gpu).data)?;
            }
            w.flush()?;
        }
        std::fs::rename(tmp, path)
    }

    pub fn load(gpu: &Gpu, path: &str) -> io::Result<(Self, AudioCheckpoint)> {
        let bad = |m: String| io::Error::new(io::ErrorKind::InvalidData, format!("{path}: {m}"));
        let mut r = BufReader::new(File::open(path)?);
        let r: &mut dyn Read = &mut r;
        let magic = read_u32(r)?;
        if ![MAGIC, MAGIC_NO_SEG_FIELDS, MAGIC_WITH_CTC, MAGIC_NO_CTC].contains(&magic) {
            return Err(bad("not an audio encoder checkpoint".into()));
        }
        let mut fields = [0usize; 14];
        fields[10] = CTC_PER_FRAME * CtcAlphabet::SIZE;
        let stored = match magic {
            MAGIC => 14,
            MAGIC_NO_CTC => 10,
            _ => 11,
        };
        for f in fields.iter_mut().take(stored) {
            *f = read_u32(r)? as usize;
        }
        if magic != MAGIC {
            fields[11] = (0..fields[7]).filter(|i| i % 4 == 3).map(|i| 1 << i).sum();
            fields[12] = 0;
            fields[13] = usize::MAX;
        }
        let cfg = AudioCfg::from_fields(fields);
        let step = read_u64(r)?;
        let seconds = f64::from_bits(read_u64(r)?);
        let stats = FeatureStats {
            mean: read_f32_vec(r)?.into_vec(),
            std: read_f32_vec(r)?.into_vec(),
        };
        let mut enc = Self::new(gpu, cfg);
        // What each format lacks of the tail of `param_slots`.
        let tail: &[(&'static str, usize)] = &[("character head", 2), ("output scale", 1)];
        let missing: &[(&'static str, usize)] = match magic {
            MAGIC | MAGIC_NO_SEG_FIELDS => &[],
            MAGIC_WITH_CTC => &tail[1..],
            _ => tail,
        };
        let mut slots = enc.param_slots();
        let n = read_u32(r)? as usize;
        let expected = slots.len() - missing.iter().map(|m| m.1).sum::<usize>();
        if n != expected {
            return Err(bad(format!(
                "{n} tensors, the configured encoder has {}",
                slots.len()
            )));
        }
        for (i, slot) in slots.iter_mut().take(n).enumerate() {
            let data = read_f32_vec(r)?.into_vec();
            if data.len() != slot.param.len() {
                return Err(bad(format!(
                    "tensor {i}: {} values, expected {}",
                    data.len(),
                    slot.param.len()
                )));
            }
            let host = GTensor::from_host(gpu, &Tensor::new(slot.param.dims(), data));
            slot.param.copy_from(gpu, &host);
        }
        drop(slots);
        enc.missing = missing.iter().map(|m| m.0).collect();
        Ok((
            enc,
            AudioCheckpoint {
                step,
                seconds,
                stats,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::ctc::Ctc;
    use crate::audio::segments::{Bce, BoundaryTargets, Segment};

    fn tiny() -> AudioCfg {
        AudioCfg {
            feats: 6,
            stack: 3,
            stride: 2,
            hidden: 32,
            blocks: 4,
            heads: 2,
            dqk: 16,
            seg_blocks: 2,
            delay: 2,
            out: 16,
            ctc: 2 * CtcAlphabet::SIZE,
            seg_slstm: 0b01,
            seg_word: true,
            seg_max: 6,
        }
    }

    #[test]
    fn checkpoint_round_trips() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let mut enc = AudioEncoder::new(&gpu, tiny());
        let path = std::env::temp_dir().join(format!("aen1-{}", std::process::id()));
        let path = path.to_str().unwrap();
        let ck = AudioCheckpoint {
            step: 7,
            seconds: 12.5,
            stats: FeatureStats {
                mean: vec![1.0, 2.0],
                std: vec![3.0, 4.0],
            },
        };
        enc.save(&gpu, path, &ck).unwrap();
        let (mut back, ck2) = AudioEncoder::load(&gpu, path).unwrap();
        std::fs::remove_file(path).ok();
        assert_eq!(
            (ck2.step, ck2.seconds, ck2.stats.std),
            (7, 12.5, vec![3.0, 4.0])
        );
        let a: Vec<Vec<f32>> = enc
            .param_slots()
            .iter()
            .map(|s| s.param.to_host(&gpu).data)
            .collect();
        let b: Vec<Vec<f32>> = back
            .param_slots()
            .iter()
            .map(|s| s.param.to_host(&gpu).data)
            .collect();
        assert_eq!(a, b);
    }

    /// A word's embedding is the same whatever else is in the batch: no state
    /// crosses from one word to the next and padding to a longer neighbour
    /// changes nothing, with and without `[W]`, and a word longer than
    /// `seg_max` keeps only its last frames.
    #[test]
    fn a_word_embeds_as_it_does_alone() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let seg = |s: usize, e: usize| Segment { start: s, end: e };
        for seg_word in [true, false] {
            let cfg = AudioCfg { seg_word, ..tiny() };
            let (batch, t_in) = (2, 64);
            let t_out = cfg.stem().frames(t_in);
            let segments = vec![
                WordSegments(vec![seg(0, 2), seg(2, 11), seg(14, 15)]),
                WordSegments(vec![seg(3, 7), seg(7, 9)]),
            ];
            let feats = Tensor::random_seeded(&[batch, t_in, cfg.feats], 0.5, 7).data;
            let mut enc = AudioEncoder::new(&gpu, cfg);
            let cache = enc.training_cache(&gpu, batch * t_out, t_out, 8);
            let embed = |enc: &mut AudioEncoder, segs: &[WordSegments]| {
                let out = enc.forward(&gpu, &feats, batch, t_in, segs, &cache);
                let e = out.embeddings.as_ref().unwrap().to_host(&gpu).data;
                enc.discard();
                e
            };
            let all = embed(&mut enc, &segments);
            let mut k = 0;
            for (b, utt) in segments.iter().enumerate() {
                for (i, s) in utt.0.iter().enumerate() {
                    // With the word before it, so the timing input sees the
                    // same pause.
                    let mut alone = vec![WordSegments::default(); batch];
                    alone[b] = WordSegments(utt.0[i.saturating_sub(1)..=i].to_vec());
                    let one = embed(&mut enc, &alone);
                    let one = &one[one.len() - cfg.out..];
                    let got = &all[k * cfg.out..(k + 1) * cfg.out];
                    let diff = got.iter().zip(one).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
                    assert!(diff < 1e-4, "seg_word {seg_word}: word {k} differs by {diff}");
                    k += 1;
                }
            }
        }
    }

    /// End to end on a fixed batch: both losses must fall, so every gradient
    /// reaches its parameters with the right sign. Feature 0 marks each word's
    /// last encoder frame, which makes the boundaries learnable from the input.
    #[test]
    fn fits_boundaries_and_embeddings_on_a_fixed_batch() {
        let Some(gpu) = super::super::test_gpu() else {
            return;
        };
        let cfg = tiny();
        let (batch, t_in) = (2, 64);
        let t_out = cfg.stem().frames(t_in);
        let seg = |s: usize, e: usize| Segment { start: s, end: e };
        let segments = vec![
            WordSegments(vec![
                seg(0, 4),
                seg(4, 9),
                seg(12, 15),
                seg(15, 22),
                seg(25, 30),
            ]),
            WordSegments(vec![seg(2, 6), seg(6, 8), seg(10, 18), seg(20, 31)]),
        ];
        let n_seg: usize = segments.iter().map(WordSegments::len).sum();

        let mut feats = Tensor::random(&[batch, t_in, cfg.feats], 0.5).data;
        for (b, segs) in segments.iter().enumerate() {
            for s in &segs.0 {
                for i in s.last() * cfg.stride..s.end * cfg.stride {
                    feats[(b * t_in + i) * cfg.feats] = 3.0;
                }
            }
        }
        let targets = BoundaryTargets::concat(
            segments
                .iter()
                .map(|s| BoundaryTargets::new(s, t_out, cfg.delay, 1)),
        );
        let bce = Bce { pos_weight: 4.0 };
        let want = Tensor::random(&[n_seg, cfg.out], 1.0).data;
        let ctc = Ctc {
            vocab: CtcAlphabet::SIZE,
        };
        let ctc_targets = [
            CtcAlphabet::encode("a cab").unwrap(),
            CtcAlphabet::encode("bee").unwrap(),
        ];

        let cache = TrainingCache::for_shape(&gpu, 256, cfg.hidden);
        let mut enc = AudioEncoder::new(&gpu, cfg);
        let mut adam = AdamCfg::new(3e-3, 0.0);
        let mut losses = Vec::new();
        for step in 0..150 {
            let out = enc.forward(&gpu, &feats, batch, t_in, &segments, &cache);
            let logits = out.boundary.to_host(&gpu).data;
            let (b_loss, d_logits) = bce.loss_and_grad(&logits, &targets);
            let e = out.embeddings.as_ref().unwrap().to_host(&gpu).data;
            let mse = e
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                / e.len() as f32;
            let d_e: Vec<f32> = e
                .iter()
                .zip(&want)
                .map(|(a, b)| 2.0 * (a - b) / e.len() as f32)
                .collect();
            let chars = out.ctc.to_host(&gpu).data;
            let mut d_ctc = vec![0.0; chars.len()];
            let mut c_loss = 0.0;
            for (b, target) in ctc_targets.iter().enumerate() {
                let rows = b * t_out * cfg.ctc..(b + 1) * t_out * cfg.ctc;
                let (l, g) = ctc.loss_and_grad(&chars[rows.clone()], target).unwrap();
                c_loss += l;
                d_ctc[rows].copy_from_slice(&g);
            }
            losses.push((b_loss, mse, c_loss));

            let d_logits = GTensor::from_host(
                &gpu,
                &Tensor::new(&[batch * t_out, cfg.delay + 1], d_logits),
            );
            let d_e = GTensor::from_host(&gpu, &Tensor::new(&[n_seg, cfg.out], d_e));
            let d_ctc = GTensor::from_host(&gpu, &Tensor::new(&[batch * t_out, cfg.ctc], d_ctc));
            enc.backward(&gpu, &d_logits, Some(&d_e), Some((&d_ctc, 1.0)), &cache);
            if step == 0 {
                let stem_grad: f32 = enc.stem.dw.to_host(&gpu).data.iter().map(|g| g.abs()).sum();
                assert!(stem_grad > 0.0, "no gradient reached the stem");
                let word_grad: f32 = enc.d_word.to_host(&gpu).data.iter().map(|g| g.abs()).sum();
                assert!(word_grad > 0.0, "no gradient reached [W]");
            }
            adam.t += 1;
            enc.step(&gpu, &adam);
        }
        let (first, last) = (losses[0], *losses.last().unwrap());
        assert!(
            last.0 < 0.5 * first.0,
            "boundary loss {first:?} -> {last:?}"
        );
        assert!(
            last.1 < 0.5 * first.1,
            "embedding loss {first:?} -> {last:?}"
        );
    }
}
