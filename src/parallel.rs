// parallel.rs

use std::{any::Any, io};

use crate::{
    nn_layer::{DynCache, NnLayer},
    saving::{write_f32, write_u8, write_u32},
};

/// Cache für den ParallelLayer
pub struct ParallelCache {
    pub sub1: Box<dyn DynCache>,
    pub sub2: Box<dyn DynCache>,
    pub output: Vec<f32>, // concat( branch1.output + branch2.output )
    pub dx: Vec<f32>,
}

impl DynCache for ParallelCache {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn output(&self) -> &[f32] {
        &self.output
    }
    fn input_grad(&self) -> &[f32] {
        &self.dx
    }
}

// ── ParallelLayer ─────────────────────────────────────────────────────────────

pub struct ParallelLayer {
    branch1: Box<dyn NnLayer>, // z. B. Char-Head (Dense + Softmax)
    branch2: Box<dyn NnLayer>, // z. B. High-Level-Head (Dense + Tanh)
    branch1_size: usize,
    bptt_grad: Vec<f32>,
    pub lr1: f32,
    pub lr2: f32,
}

impl ParallelLayer {
    pub fn new(branch1: Box<dyn NnLayer>, branch2: Box<dyn NnLayer>, lr1: f32, lr2: f32) -> Self {
        Self {
            branch1_size: branch1.output_size(),
            bptt_grad: vec![0.0; branch1.output_size() + branch2.output_size()],
            branch1,
            branch2,
            lr1,
            lr2,
        }
    }
}

impl NnLayer for ParallelLayer {
    fn forward(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        let c = cache.as_any_mut().downcast_mut::<ParallelCache>().unwrap();

        self.branch1.forward(input, &mut *c.sub1);
        self.branch2.forward(input, &mut *c.sub2);

        let o1 = c.sub1.output();
        let o2 = c.sub2.output();
        c.output.resize(o1.len() + o2.len(), 0.0);
        c.output[..o1.len()].copy_from_slice(o1);
        c.output[o1.len()..].copy_from_slice(o2);
    }

    fn forward_sample(&mut self, input: &[f32], cache: &mut dyn DynCache) {
        let c = cache.as_any_mut().downcast_mut::<ParallelCache>().unwrap();
        self.branch1.forward_sample(input, &mut *c.sub1);
        self.branch2.forward_sample(input, &mut *c.sub2);

        let o1 = c.sub1.output();
        let o2 = c.sub2.output();
        c.output.resize(o1.len() + o2.len(), 0.0);
        c.output[..o1.len()].copy_from_slice(o1);
        c.output[o1.len()..].copy_from_slice(o2);
    }

    fn backward(&mut self, delta: &mut [f32], cache: &mut dyn DynCache) {
        let c = cache.as_any_mut().downcast_mut::<ParallelCache>().unwrap();

        let len1 = c.sub1.output().len();
        let (delta1, delta2) = delta.split_at_mut(len1);

        let mut d1 = delta1.to_vec();
        let mut d2 = delta2.to_vec();

        self.branch1.backward(&mut d1, &mut *c.sub1);
        self.branch2.backward(&mut d2, &mut *c.sub2);

        // dx = dx1 + dx2 (beide Branches haben denselben Input)
        let dx1 = c.sub1.input_grad();
        let dx2 = c.sub2.input_grad();
        c.dx.resize(dx1.len(), 0.0);
        for i in 0..dx1.len() {
            c.dx[i] = dx1[i] + dx2[i];
        }
    }

    fn layer_tag(&self) -> u8 {
        7
    } // TAG_PARALLEL

    fn save(&self, w: &mut dyn io::Write) -> io::Result<()> {
        write_u8(w, self.branch1.layer_tag())?;
        write_u32(w, self.branch1.input_size() as u32)?;
        write_u32(w, self.branch1.output_size() as u32)?;
        write_f32(w, self.lr1)?;
        self.branch1.save(w)?;
        write_u8(w, self.branch2.layer_tag())?;
        write_u32(w, self.branch2.input_size() as u32)?;
        write_u32(w, self.branch2.output_size() as u32)?;
        write_f32(w, self.lr2)?;
        self.branch2.save(w)
    }

    fn make_cache(&self) -> Box<dyn DynCache> {
        Box::new(ParallelCache {
            sub1: self.branch1.make_cache(),
            sub2: self.branch2.make_cache(),
            output: vec![],
            dx: vec![],
        })
    }

    fn input_size(&self) -> usize {
        self.branch1.input_size() // beide Branches haben gleichen Input
    }
    fn output_size(&self) -> usize {
        self.branch1.output_size() + self.branch2.output_size()
    }

    fn apply_grads(&mut self, lr: f32) {
        self.branch1.apply_grads(lr * self.lr1);
        self.branch2.apply_grads(lr * self.lr2);
    }
    fn clear_grads(&mut self) {
        self.branch1.clear_grads();
        self.branch2.clear_grads();
    }
    fn scale_grads(&mut self, scale: f32) {
        self.branch1.scale_grads(scale);
        self.branch2.scale_grads(scale);
    }
    fn clip_grads(&mut self) {
        self.branch1.clip_grads();
        self.branch2.clip_grads();
    }
    fn reset_state(&mut self) {
        self.branch1.reset_state();
        self.branch2.reset_state();
    }

    fn bptt_hidden_grad(&mut self) -> Option<&[f32]> {
        let hidden1 = self.branch1.bptt_hidden_grad();
        let hidden2 = self.branch2.bptt_hidden_grad();

        if hidden1.is_none() && hidden2.is_none() {
            return None;
        } else {
            if let Some(hidden) = hidden1 {
                self.bptt_grad[..self.branch1_size].copy_from_slice(hidden);
            }

            if let Some(hidden) = hidden2 {
                self.bptt_grad[self.branch1_size..].copy_from_slice(hidden);
            }

            Some(&self.bptt_grad)
        }
    }
}
