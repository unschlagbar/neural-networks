//! Device-side audio encoder: speech features in, one embedding per spoken word
//! out, in the space the backbone reads text words in.
//!
//!   1. stem     — `stack` feature frames gathered per encoder frame, every
//!                 `stride` frames, `→ Linear`: a causal strided convolution over
//!                 time with a full-frequency kernel.
//!   2. frames   — blocks over each utterance (mLSTM, every fourth an sLSTM),
//!                 causal, utterances as the batch axis.
//!   3. boundary — `Linear → delay + 1` logits per encoder frame
//!                 (see [`boundary_targets`](crate::audio::segments::boundary_targets)).
//!   4. segments — every word's frames plus `delay` after it, laid end to end in
//!                 one sequence with the cells restarting at each word; the last
//!                 step `→ Linear` plus a `Linear` of the word's timing is its
//!                 embedding.
//!
//! The output projection and its bias are the audio path's own, which is all a
//! modality marker in front of the backbone would add.
//!
//! Host-side layout (stem ids, segment rows, targets) is in [`crate::audio`].

use cudarc::driver::{CudaSlice, DevicePtr};

use super::arena::{self, ParamSlot, TrainingCache};
use super::block::{Block, BlockLike, Resets};
use super::{GTensor, Gpu, linear::Linear, mlstm::MLstm, ops, slstm::SLstm};
use crate::audio::segments::{SegmentRows, StemLayout, WordSegments};
use crate::config::{
    AUDIO_BLOCKS, AUDIO_DQK, AUDIO_FEATS, AUDIO_HEADS, AUDIO_HIDDEN, AUDIO_SEG_BLOCKS,
    AUDIO_STACK, AUDIO_STRIDE, BOUNDARY_DELAY, up_of,
};
use crate::nn2::optim::AdamCfg;
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
    /// Encoder frames the boundary head may decide late, and the segment encoder
    /// reads past a word's end.
    pub delay: usize,
    /// Width of the embeddings produced: the text encoder's `hc`.
    pub out: usize,
}

impl AudioCfg {
    /// The configured encoder, producing embeddings `out` wide.
    pub fn from_config(out: usize) -> Self {
        Self {
            feats: AUDIO_FEATS,
            stack: AUDIO_STACK,
            stride: AUDIO_STRIDE,
            hidden: AUDIO_HIDDEN,
            blocks: AUDIO_BLOCKS,
            heads: AUDIO_HEADS,
            dqk: AUDIO_DQK,
            seg_blocks: AUDIO_SEG_BLOCKS,
            delay: BOUNDARY_DELAY,
            out,
        }
    }

    pub fn stem(&self) -> StemLayout {
        StemLayout {
            stack: self.stack,
            stride: self.stride,
        }
    }
}

/// What one forward hands back.
pub struct AudioOut {
    /// Boundary logits, `[batch · t_out, delay + 1]`.
    pub boundary: GTensor<f32>,
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
    rows: usize,
    n_seg: usize,
    /// Encoder-frame row of every segment-sequence row.
    ids: CudaSlice<u32>,
    /// Segment-sequence row of every segment's last step.
    last: CudaSlice<u32>,
    /// Backs `resets.mask`.
    _mask: CudaSlice<i32>,
    resets: Resets,
}

pub struct AudioEncoder {
    pub cfg: AudioCfg,
    pub stem: Linear,
    pub frame_blocks: Vec<Box<dyn BlockLike>>,
    pub boundary: Linear,
    pub seg_blocks: Vec<Box<dyn BlockLike>>,
    pub seg_out: Linear,
    pub seg_timing: Linear,
    saved: Option<Saved>,
}

impl AudioEncoder {
    pub fn new(gpu: &Gpu, cfg: AudioCfg) -> Self {
        let h = cfg.hidden;
        let block = |i: usize| -> Box<dyn BlockLike> {
            if i % 4 == 3 {
                Box::new(Block::from_cell(gpu, h, up_of(h), SLstm::new_rand(gpu, h, h)))
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
        Self {
            cfg,
            stem: Linear::new_rand(gpu, cfg.stack * cfg.feats, h),
            frame_blocks: (0..cfg.blocks).map(block).collect(),
            boundary,
            seg_blocks: (0..cfg.seg_blocks).map(block).collect(),
            seg_out: Linear::new_rand(gpu, h, cfg.out),
            seg_timing: Linear::new_rand(gpu, 2, cfg.out),
            saved: None,
        }
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
        let c = self.cfg;
        assert_eq!(feats.len(), batch * t_in * c.feats, "AudioEncoder::forward — feature shape");
        assert_eq!(segments.len(), batch, "AudioEncoder::forward — one segment list per utterance");
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
        let mut stacked = ops::embedding_gather_u32(gpu, &table, &dids.slice(..), ids.len(), c.feats);
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

        let rows = SegmentRows::new(segments, t_out, c.delay);
        let (embeddings, seg) = if rows.segments() == 0 {
            (None, None)
        } else {
            let (e, s) = self.segments_forward(gpu, &x, &rows, segments, cache);
            (Some(e), Some(s))
        };
        self.saved = Some(Saved { batch, t_out, seg });
        AudioOut {
            boundary,
            embeddings,
            t_out,
        }
    }

    fn segments_forward(
        &mut self,
        gpu: &Gpu,
        frames: &GTensor<f32>,
        rows: &SegmentRows,
        segments: &[WordSegments],
        cache: &TrainingCache,
    ) -> (GTensor<f32>, SegSaved) {
        let c = self.cfg;
        let r = rows.ids.len();
        let n_seg = rows.segments();
        let ids = ops::upload_ids_u32(gpu, &rows.ids);
        let mut x = ops::embedding_gather_u32(gpu, frames, &ids.slice(..), r, c.hidden);
        x.reshape_to(&[1, r, c.hidden]);

        let mut mask_host = vec![0i32; r];
        for &i in &rows.resets {
            mask_host[i] = 1;
        }
        let mask = gpu.stream.clone_htod(&mask_host).expect("segment reset mask");
        let resets = Resets {
            rows: rows.resets.clone(),
            mask: if rows.resets.is_empty() {
                0
            } else {
                mask.device_ptr(&gpu.stream).0
            },
        };
        for blk in self.seg_blocks.iter_mut() {
            blk.set_carry(false);
            blk.reset_state(gpu);
            blk.set_resets(&resets);
            let mut y = GTensor::uninit(gpu, &[1, r, c.hidden]);
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

        let saved = SegSaved {
            rows: r,
            n_seg,
            ids,
            last,
            _mask: mask,
            resets,
        };
        (e, saved)
    }

    /// Backward of the last [`forward`](Self::forward): `d_boundary` is
    /// `[batch · t_out, delay + 1]`, `d_embeddings` `[segments, out]` (`None`
    /// trains the boundary head and frame stack alone). Accumulates into every
    /// parameter's gradient.
    pub fn backward(
        &mut self,
        gpu: &Gpu,
        d_boundary: &GTensor<f32>,
        d_embeddings: Option<&GTensor<f32>>,
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
        match (seg, d_embeddings) {
            (Some(s), Some(de)) => {
                let d_rows = self.segments_backward(gpu, &s, de, cache);
                ops::embedding_scatter_add_u32(gpu, &mut dx, &s.ids.slice(..), s.rows, &d_rows, c.hidden);
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
        let mut d_timing = GTensor::uninit(gpu, &[s.n_seg, 2]);
        self.seg_timing.backward(gpu, de, &mut d_timing, cache);
        let mut d_ends = GTensor::uninit(gpu, &[s.n_seg, c.hidden]);
        self.seg_out.backward(gpu, de, &mut d_ends, cache);

        let mut dx = GTensor::zeros(gpu, &[s.rows, c.hidden]);
        ops::embedding_scatter_add_u32(gpu, &mut dx, &s.last.slice(..), s.n_seg, &d_ends, c.hidden);
        dx.reshape_to(&[1, s.rows, c.hidden]);
        for blk in self.seg_blocks.iter_mut().rev() {
            blk.set_resets(&s.resets);
            blk.reset_bptt(gpu);
            let mut d_in = GTensor::uninit(gpu, &[1, s.rows, c.hidden]);
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
        slots
    }

    pub fn zero_grad(&mut self, gpu: &Gpu) {
        self.stem.zero_grad(gpu);
        for blk in self.frame_blocks.iter_mut().chain(self.seg_blocks.iter_mut()) {
            blk.zero_grad(gpu);
        }
        self.boundary.zero_grad(gpu);
        self.seg_out.zero_grad(gpu);
        self.seg_timing.zero_grad(gpu);
    }

    /// AdamW over this encoder on its own, then clear its gradients.
    pub fn step(&mut self, gpu: &Gpu, cfg: &AdamCfg) {
        arena::step_slots(gpu, &mut self.param_slots(), cfg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            seg_blocks: 1,
            delay: 2,
            out: 16,
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
            WordSegments(vec![seg(0, 4), seg(4, 9), seg(12, 15), seg(15, 22), seg(25, 30)]),
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

        let cache = TrainingCache::for_shape(&gpu, 256, cfg.hidden);
        let mut enc = AudioEncoder::new(&gpu, cfg);
        let mut adam = AdamCfg::new(3e-3, 0.0);
        let mut losses = Vec::new();
        for step in 0..150 {
            let out = enc.forward(&gpu, &feats, batch, t_in, &segments, &cache);
            let logits = out.boundary.to_host(&gpu).data;
            let (b_loss, d_logits) = bce.loss_and_grad(&logits, &targets);
            let e = out.embeddings.as_ref().unwrap().to_host(&gpu).data;
            let mse = e.iter().zip(&want).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / e.len() as f32;
            let d_e: Vec<f32> = e
                .iter()
                .zip(&want)
                .map(|(a, b)| 2.0 * (a - b) / e.len() as f32)
                .collect();
            losses.push((b_loss, mse));

            let d_logits = GTensor::from_host(&gpu, &Tensor::new(&[batch * t_out, cfg.delay + 1], d_logits));
            let d_e = GTensor::from_host(&gpu, &Tensor::new(&[n_seg, cfg.out], d_e));
            enc.backward(&gpu, &d_logits, Some(&d_e), &cache);
            if step == 0 {
                let stem_grad: f32 = enc.stem.dw.to_host(&gpu).data.iter().map(|g| g.abs()).sum();
                assert!(stem_grad > 0.0, "no gradient reached the stem");
            }
            adam.t += 1;
            enc.step(&gpu, &adam);
        }
        let (first, last) = (losses[0], *losses.last().unwrap());
        assert!(last.0 < 0.5 * first.0, "boundary loss {first:?} -> {last:?}");
        assert!(last.1 < 0.5 * first.1, "embedding loss {first:?} -> {last:?}");
    }
}
