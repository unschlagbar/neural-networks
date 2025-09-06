use iron_oxide::collections::Matrix;

use crate::{
    layer::{Activation, DenseLayer},
    lstm::{DH, LSTMLayer, LSTMLayerGrads, sub_in_place, sub_vec_in_place},
    mlp::DenseLayerGrads,
};

pub struct RnntConfig {
    pub feat_dim: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub enc_layers: usize,
    pub pred_layers: usize,
    pub blank_id: u16,
}

pub struct Encoder {
    lstms: Vec<LSTMLayer>,
    proj: DenseLayer,
    grads_lstms: Vec<LSTMLayerGrads>,
    grads_proj: DenseLayerGrads,
}

impl Encoder {
    pub fn new(cfg: &RnntConfig) -> Self {
        let mut lstms = Vec::with_capacity(cfg.enc_layers);
        let mut grads_lstms = Vec::with_capacity(cfg.enc_layers);
        for i in 0..cfg.enc_layers {
            let in_size = if i == 0 {
                cfg.feat_dim
            } else {
                cfg.hidden_size
            };
            let layer = LSTMLayer::random(in_size, cfg.hidden_size);
            grads_lstms.push(LSTMLayerGrads {
                wf: Matrix::zeros(layer.wf.rows(), layer.wf.cols()),
                wi: Matrix::zeros(layer.wi.rows(), layer.wi.cols()),
                wc: Matrix::zeros(layer.wc.rows(), layer.wc.cols()),
                wo: Matrix::zeros(layer.wo.rows(), layer.wo.cols()),
                b: Matrix::zeros(4, layer.hidden_size),
            });
            lstms.push(layer);
        }
        let proj = DenseLayer::random(cfg.hidden_size, cfg.hidden_size, Activation::Linear);
        let grads_proj = DenseLayerGrads {
            weights: Matrix::zeros(proj.weights.rows(), proj.weights.cols()),
            biases: vec![0.0; proj.biases.len()],
        };
        Self {
            lstms,
            proj,
            grads_lstms,
            grads_proj,
        }
    }

    pub fn forward(&mut self, feats: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.clear_mem();
        let mut outputs = Vec::with_capacity(feats.len());
        for feat in feats {
            let mut h = feat.clone();
            for lstm in &mut self.lstms {
                let cache = lstm.forward(&h);
                h = cache.states[DH].to_vec();
            }
            let proj_h = self.proj.forward_no_activation(&h).to_vec();
            outputs.push(proj_h);
        }
        outputs
    }

    // Placeholder for backwards; RNNT requires custom loss propagation
    pub fn backwards(&mut self, _d_enc: &[Vec<f32>]) {
        // Implement RNNT-specific backprop here if training is needed
        // For now, stubbed as training logic is not fully adapted
    }

    pub fn clear_mem(&mut self) {
        for lstm in &mut self.lstms {
            lstm.d.clear();
        }
    }

    pub fn apply_grads(&mut self, lr: f32) {
        for (lstm, grads) in self.lstms.iter_mut().zip(&mut self.grads_lstms) {
            sub_in_place(&mut lstm.wf, &grads.wf, lr);
            sub_in_place(&mut lstm.wi, &grads.wi, lr);
            sub_in_place(&mut lstm.wc, &grads.wc, lr);
            sub_in_place(&mut lstm.wo, &grads.wo, lr);
            sub_vec_in_place(lstm.b.as_slice_mut(), grads.b.as_slice(), lr);
        }
        sub_in_place(&mut self.proj.weights, &self.grads_proj.weights, lr);
        sub_vec_in_place(&mut self.proj.biases, &self.grads_proj.biases, lr);
    }

    pub fn clear_grads(&mut self) {
        for grads in &mut self.grads_lstms {
            grads.wf.clear();
            grads.wi.clear();
            grads.wc.clear();
            grads.wo.clear();
            grads.b.clear();
        }
        self.grads_proj.weights.clear();
        self.grads_proj.biases.fill(0.0);
    }
}

pub struct Predictor {
    embed: Matrix, // vocab_size x hidden_size
    lstms: Vec<LSTMLayer>,
    proj: DenseLayer,
    blank_id: u16,
    grads_embed: Matrix,
    grads_lstms: Vec<LSTMLayerGrads>,
    grads_proj: DenseLayerGrads,
}

impl Predictor {
    pub fn new(cfg: &RnntConfig) -> Self {
        let embed = Matrix::random(cfg.vocab_size, cfg.hidden_size, 1.0); // Assume random init similar to layers
        let grads_embed = Matrix::zeros(cfg.vocab_size, cfg.hidden_size);
        let mut lstms = Vec::with_capacity(cfg.pred_layers);
        let mut grads_lstms = Vec::with_capacity(cfg.pred_layers);
        for _ in 0..cfg.pred_layers {
            let layer = LSTMLayer::random(cfg.hidden_size, cfg.hidden_size);
            grads_lstms.push(LSTMLayerGrads {
                wf: Matrix::zeros(layer.wf.rows(), layer.wf.cols()),
                wi: Matrix::zeros(layer.wi.rows(), layer.wi.cols()),
                wc: Matrix::zeros(layer.wc.rows(), layer.wc.cols()),
                wo: Matrix::zeros(layer.wo.rows(), layer.wo.cols()),
                b: Matrix::zeros(4, layer.hidden_size),
            });
            lstms.push(layer);
        }
        let proj = DenseLayer::random(cfg.hidden_size, cfg.hidden_size, Activation::Linear);
        let grads_proj = DenseLayerGrads {
            weights: Matrix::zeros(proj.weights.rows(), proj.weights.cols()),
            biases: vec![0.0; proj.biases.len()],
        };
        Self {
            embed,
            lstms,
            proj,
            blank_id: cfg.blank_id,
            grads_embed,
            grads_lstms,
            grads_proj,
        }
    }

    pub fn forward(&mut self, targets: &[u16]) -> Vec<Vec<f32>> {
        self.clear_mem();
        let mut outputs = Vec::with_capacity(targets.len() + 1);
        // BOS
        let emb = self.embed[self.blank_id as usize].to_vec();
        let mut h = emb;
        for lstm in &mut self.lstms {
            let cache = lstm.forward(&h);
            h = cache.states[DH].to_vec();
        }
        let proj_h = self.proj.forward_no_activation(&h);
        outputs.push(proj_h.to_vec());

        for &token in targets {
            let emb = self.embed[token as usize].to_vec();
            let mut h = emb;
            for lstm in &mut self.lstms {
                let cache = lstm.forward(&h);
                h = cache.states[DH].to_vec();
            }
            let proj_h = self.proj.forward_no_activation(&h);
            outputs.push(proj_h.to_vec());
        }
        outputs.to_vec()
    }

    pub fn step(&mut self, token: u16) -> Vec<f32> {
        let emb = self.embed[token as usize].to_vec();
        let mut h = emb;
        for lstm in &mut self.lstms {
            let cache = lstm.forward(&h);
            h = cache.states[DH].to_vec();
        }
        self.proj.forward_no_activation(&h).to_vec()
    }

    // Placeholder for backwards
    pub fn backwards(&mut self, _d_pred: &[Vec<f32>]) {
        // Implement RNNT-specific backprop here if needed
    }

    pub fn clear_mem(&mut self) {
        for lstm in &mut self.lstms {
            lstm.d.clear();
        }
    }

    pub fn apply_grads(&mut self, lr: f32) {
        sub_in_place(&mut self.embed, &self.grads_embed, lr);
        for (lstm, grads) in self.lstms.iter_mut().zip(&mut self.grads_lstms) {
            sub_in_place(&mut lstm.wf, &grads.wf, lr);
            sub_in_place(&mut lstm.wi, &grads.wi, lr);
            sub_in_place(&mut lstm.wc, &grads.wc, lr);
            sub_in_place(&mut lstm.wo, &grads.wo, lr);
            sub_vec_in_place(lstm.b.as_slice_mut(), grads.b.as_slice(), lr);
        }
        sub_in_place(&mut self.proj.weights, &self.grads_proj.weights, lr);
        sub_vec_in_place(&mut self.proj.biases, &self.grads_proj.biases, lr);
    }

    pub fn clear_grads(&mut self) {
        self.grads_embed.clear();
        for grads in &mut self.grads_lstms {
            grads.wf.clear();
            grads.wi.clear();
            grads.wc.clear();
            grads.wo.clear();
            grads.b.clear();
        }
        self.grads_proj.weights.clear();
        self.grads_proj.biases.fill(0.0);
    }
}

pub struct Joiner {
    out: DenseLayer,
    grads_out: DenseLayerGrads,
}

impl Joiner {
    pub fn new(cfg: &RnntConfig) -> Self {
        let out = DenseLayer::random(cfg.hidden_size, cfg.vocab_size, Activation::Linear);
        let grads_out = DenseLayerGrads {
            weights: Matrix::zeros(out.weights.rows(), out.weights.cols()),
            biases: vec![0.0; out.biases.len()],
        };
        Self { out, grads_out }
    }

    pub fn forward(&mut self, enc: &[Vec<f32>], pred: &[Vec<f32>]) -> Vec<Vec<Vec<f32>>> {
        let mut joint = Vec::with_capacity(enc.len());
        for enc_t in enc {
            let mut joint_t = Vec::with_capacity(pred.len());
            for pred_u in pred {
                let mut x = enc_t.clone();
                for (i, &p) in pred_u.iter().enumerate() {
                    x[i] += p;
                }
                x.iter_mut().for_each(|v| *v = v.tanh());
                let logit = self.out.forward_no_activation(&x).to_vec();
                joint_t.push(logit);
            }
            joint.push(joint_t);
        }
        joint
    }

    // Placeholder for backwards
    pub fn backwards(&mut self, _d_joint: &Vec<Vec<Vec<f32>>>) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        // Return d_enc and d_pred; implement if training needed
        (vec![], vec![])
    }

    pub fn apply_grads(&mut self, lr: f32) {
        sub_in_place(&mut self.out.weights, &self.grads_out.weights, lr);
        sub_vec_in_place(&mut self.out.biases, &self.grads_out.biases, lr);
    }

    pub fn clear_grads(&mut self) {
        self.grads_out.weights.clear();
        self.grads_out.biases.fill(0.0);
    }
}

pub struct Rnnt {
    encoder: Encoder,
    predictor: Predictor,
    joiner: Joiner,
    cfg: RnntConfig,
}

impl Rnnt {
    pub fn new(cfg: RnntConfig) -> Self {
        let encoder = Encoder::new(&cfg);
        let predictor = Predictor::new(&cfg);
        let joiner = Joiner::new(&cfg);
        Self {
            encoder,
            predictor,
            joiner,
            cfg,
        }
    }

    pub fn forward(&mut self, feats: &[Vec<f32>], targets: &[u16]) -> Vec<Vec<Vec<f32>>> {
        let enc = self.encoder.forward(feats);
        let pred = self.predictor.forward(targets);
        self.joiner.forward(&enc, &pred)
    }

    pub fn greedy_decode(&mut self, feats: &[Vec<f32>]) -> Vec<u16> {
        let enc = self.encoder.forward(feats);
        self.predictor.clear_mem();
        let mut tokens = Vec::new();
        let mut pred = self.predictor.step(self.cfg.blank_id);
        for enc_t in enc {
            let mut x = enc_t.clone();
            for (i, &p) in pred.iter().enumerate() {
                x[i] += p;
            }
            x.iter_mut().for_each(|v| *v = v.tanh());
            let logits = self.joiner.out.forward_no_activation(&x);
            let max_val = logits.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            let token = logits.iter().position(|&v| v == max_val).unwrap() as u16;
            if token != self.cfg.blank_id {
                tokens.push(token);
                pred = self.predictor.step(token);
            }
        }
        tokens
    }

    // Training would require implementing RNNT loss and full backprop
    // For example:
    // pub fn train(&mut self, feats_batches: Batches<Vec<f32>>, target_batches: Batches<u16>, lr: f32) {
    //     // Compute forward, RNNT loss, backwards through joiner/predictor/encoder
    //     // Not implemented here as it requires custom loss function
    // }
}
