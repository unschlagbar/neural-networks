use iron_oxide::collections::Matrix;
use rand::random_range;

use crate::{
    batches::Batches,
    layer::{softmax, Activation, DenseLayer, DenseLayerGrads},
    lstm::{
        one_hot, sub_in_place, sub_vec_in_place, LSTMCache, LSTMLayer, LSTMLayerGrads, CLIP, DH
    },
};

pub struct Sequential {
    pub layers: Vec<Layer>,
    pub grads: Vec<LayerGrads>,
    pub dh_next: Vec<Vec<f32>>,
    pub dc_next: Vec<Vec<f32>>,
    pub cache: Vec<Vec<Cache>>,
}

impl Sequential {
    pub fn new(layout: Vec<LayerBuilder>, mut input_size: usize) -> Self {
        let mut layers = Vec::with_capacity(layout.len());

        for layout in layout {
            layers.push(layout.layer_random_init(&mut input_size));
        }

        let grads = layers.iter().map(LayerGrads::from_layer).collect();

        let mut dh_next = Vec::with_capacity(layers.len());
        let mut dc_next = Vec::with_capacity(layers.len());

        for lay in &layers {
            dh_next.push(vec![0.0; lay.hidden_size()]);
            if lay.is_recurrent() {
                dc_next.push(vec![0.0; lay.hidden_size()]);
            } else {
                dc_next.push(Vec::with_capacity(0));
            }
        }
        Self {
            layers,
            grads,
            dh_next,
            dc_next,
            cache: Vec::new(),
        }
    }

    pub fn make_cache(&mut self, len: usize) {
        let mut caches = Vec::with_capacity(len);

        for i in 0..len {
            caches.push(Vec::with_capacity(self.layers.len()));

            for layer in &mut self.layers {
                caches[i].push(layer.make_cache());
            }
        }
        self.cache = caches;
    }

    pub fn forward_over(&mut self, input: &[u16]) {
        let t = input.len();

        for t in 0..t {
            let input_vec: &[f32] = &one_hot(input[t] as _, self.layers[0].input_size());

            for (l, layer) in self.layers.iter_mut().enumerate() {
                let (input, cache) = if l == 0 {
                    (input_vec, &mut self.cache[t][l])
                } else {
                    let (left, right) = self.cache[t].split_at_mut(l);
                    (left[l - 1].output(), &mut right[0])
                };
                layer.forward(input, cache);
            }
        }
    }

    pub fn forward_sample(&mut self, input: &[u16]) -> Matrix {
        let t = input.len();
        let mut logist = Matrix::uninit(t, self.out_size());
        let end = self.layers.len() - 1;

        for t in 0..t {
            let input_vec: &[f32] = &one_hot(input[t] as _, self.in_size());

            for (l, layer) in self.layers[..end].iter_mut().enumerate() {
                let (input, cache) = if l == 0 {
                    (input_vec, &mut self.cache[0][l])
                } else {
                    let (left, right) = self.cache[0].split_at_mut(l);
                    (left[l - 1].output(), &mut right[0])
                };
                layer.forward(input, cache);
            }
            let last_layer = self.layers.last_mut().unwrap();
            let (left, right) = self.cache[0].split_at_mut(end);

            let cache = if let Cache::Dense(cache) = &mut right[0] {
                cache
            } else {
                panic!()
            };

            let input = match &mut left[end - 1] {
                Cache::Lstm(cache) => &cache.states[DH],
                Cache::Dense(cache) => &cache.0,
            };
            match last_layer {
                Layer::Lstm(_) => panic!("not supported output layer"),
                Layer::Dense(layer) => layer.forward_no_activation(input, cache),
            }

            logist[t].copy_from_slice(&cache.1);
        }
        logist
    }

    fn backwards_sequence(&mut self, seq: &[u16], targets: &[u16]) {
        for layer in &mut self.dh_next {
            layer.fill(0.0);
        }

        for layer in &mut self.dc_next {
            layer.fill(0.0);
        }
        for t in (0..seq.len()).rev() {
            let mut delta = self.cache[t].last().unwrap().output().to_vec();
            delta[targets[t] as usize] -= 1.0;

            let mut dh_layer = delta; // dh entering top layer at time t (including future contribution)

            // Propagate through layers top->bottom at this time step
            for l in (0..self.layers.len()).rev() {
                let layer = &mut self.layers[l];
                let cache = &mut self.cache[t][l];

                // call the new per-layer backward that returns dh_prev, dc_prev, dx
                // note: pass mutable refs to dh_layer and dc_next[l] so function can read them
                match layer {
                    Layer::Lstm(layer) => {
                        let (dh_prev, dx) = layer.backwards(
                            cache.lstm(),
                            &dh_layer,
                            &mut self.dc_next[l],
                            self.grads[l].lstm(),
                        );

                        // prepare dh_layer for next (lower) layer: combine dh_next from same time-step and dx
                        if l > 0 {
                            // dh_next[l-1] already contains dh coming from future time-step for layer l-1
                            dh_layer = self.dh_next[l - 1].clone();
                            for i in 0..layer.input_size {
                                dh_layer[i] += dx[i];
                            }
                        }

                        // set dh_next for this layer (to be used when processing previous t)
                        self.dh_next[l] = dh_prev;
                    }
                    Layer::Dense(layer) => {
                        let cache_dense = cache.dense();

                        // For dense, we use dh_layer as incoming delta (dL/da or dL/dz based on activation)
                        // Prepare delta_next as temp for dx
                        let mut dx_temp = vec![0.0; layer.input_size()];
                        layer.backwards(
                            cache_dense,
                            &mut dh_layer,
                            self.grads[l].dense(),
                            Some(&mut dx_temp),
                        );

                        // Set dh_layer to dx for lower layer
                        dh_layer = dx_temp;
                    }
                }
            }
        }
    }

    pub fn train(
        &mut self,
        data: Batches<u16>,
        lr: f32,
        iteration: &mut usize,
        j: &mut usize,
        batch_size: usize,
    ) {
        for layer in &mut self.layers {
            layer.clear_mem();
        }

        let mut total_loss = 0.0;
        let mut steps = 0;

        for (inputs, targets) in data {
            self.forward_over(inputs);

            let l = self.seq_loss(targets);

            total_loss += l;
            steps += 1;

            self.backwards_sequence(inputs, targets);

            *iteration += 1;

            if *iteration % batch_size == 0 {
                self.sgd_step(lr / batch_size as f32);
                self.scale_grads(0.6);
                *iteration = 1;
                *j += 1;
            }
        }

        println!("{j} Average loss = {:.4}", total_loss / steps.max(1) as f32);
    }

    pub fn sample(
        &mut self,
        prefix: &[u16],
        max_len: usize,
        temperature: f32,
        mut callback: impl FnMut(u16) -> bool,
    ) -> Vec<u16> {
        for layer in &mut self.layers {
            layer.clear_mem();
        }

        let mut last_token;

        if prefix.is_empty() {
            last_token = random_range(0..self.layers.last().unwrap().hidden_size()) as u16;
        } else {
            let _ = self.forward_sample(&prefix[0..prefix.len() - 1]);
            last_token = prefix[prefix.len() - 1];
        }

        let mut out = Vec::with_capacity(max_len);
        for _ in 0..max_len {
            let logits = self.forward_sample(&[last_token]);
            let p = &logits[0];

            let logits: Vec<f32> = p.iter().map(|&p| p / temperature.max(1e-8)).collect();
            let q = softmax(&logits);

            let mut idx: Vec<usize> = (0..logits.len()).collect();
            idx.sort_unstable_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap());

            let r = random_range(0.0..1.0);
            let mut cum = 0.0;
            let mut next = 0;
            for i in idx {
                cum += q[i];
                if cum >= r {
                    next = i as u16;
                    break;
                }
            }
            out.push(next);
            if !callback(next) {
                break;
            }
            last_token = next;
        }

        out
    }

    fn sgd_step(&mut self, lr: f32) {
        for (layer, grads) in self.layers.iter_mut().zip(&mut self.grads) {
            grads.clip();
            layer.apply_grads(grads, lr);
        }
    }

    pub fn clear_grads(&mut self) {
        for g in &mut self.grads {
            g.clear();
        }
    }

    pub fn scale_grads(&mut self, scale: f32) {
        for g in &mut self.grads {
            g.scale(scale);
        }
    }

    pub fn out_size(&self) -> usize {
        self.layers.last().unwrap().hidden_size()
    }

    pub fn in_size(&self) -> usize {
        self.layers.first().unwrap().input_size()
    }

        fn seq_loss(&self, targets: &[u16]) -> f32 {
    let mut l = 0.0;
    for t in 0..targets.len() {
        let p = self.cache[t][self.layers.len() - 1].output()[targets[t] as usize] + 1e-12;
        l += -p.ln();
    }
    l / targets.len() as f32
}
}

pub enum Layer {
    Lstm(LSTMLayer),
    Dense(DenseLayer),
}

impl Layer {
    pub fn lstm(input_size: usize, hidden_size: usize) -> Self {
        Self::Lstm(LSTMLayer::random(input_size, hidden_size))
    }

    pub fn dense(input_size: usize, hidden_size: usize, activation: Activation) -> Self {
        Self::Dense(DenseLayer::random(input_size, hidden_size, activation))
    }

    pub fn input_size(&self) -> usize {
        match self {
            Layer::Lstm(lstm) => lstm.input_size,
            Layer::Dense(dense) => dense.input_size(),
        }
    }

    pub fn hidden_size(&self) -> usize {
        match self {
            Layer::Lstm(lstm) => lstm.hidden_size,
            Layer::Dense(dense) => dense.hidden_size(),
        }
    }

    pub fn forward(&mut self, input: &[f32], cache: &mut Cache) {
        match self {
            Self::Lstm(lstm) => {
                if let Cache::Lstm(cache) = cache {
                    lstm.forward(input, cache);
                } else {
                    panic!();
                }
            }
            Self::Dense(dense) => {
                if let Cache::Dense(cache) = cache {
                    dense.forward(input, cache);
                } else {
                    panic!();
                }
            }
        }
    }

    pub fn is_recurrent(&self) -> bool {
        match self {
            Self::Lstm(_) => true,
            Self::Dense(_) => false,
        }
    }

    pub fn clear_mem(&mut self) {
        match self {
            Self::Lstm(layer) => layer.d.clear(),
            Self::Dense(_) => (),
        }
    }

    pub fn apply_grads(&mut self, grads: &mut LayerGrads, lr: f32) {
        match self {
            Self::Lstm(layer) => {
                let grads = grads.lstm();

                sub_in_place(&mut layer.wf, &grads.wf, lr);
                sub_in_place(&mut layer.wi, &grads.wi, lr);
                sub_in_place(&mut layer.wc, &grads.wc, lr);
                sub_in_place(&mut layer.wo, &grads.wo, lr);
                sub_vec_in_place(layer.b.as_slice_mut(), grads.b.as_slice(), lr);
            }
            Self::Dense(layer) => {
                let grads = grads.dense();

                sub_in_place(&mut layer.weights, &grads.weights, lr);
                sub_vec_in_place(&mut layer.biases, &grads.biases, lr);
            }
        }
    }

    pub fn make_cache(&self) -> Cache {
        match self {
            Layer::Lstm(lstm_layer) => Cache::Lstm(lstm_layer.make_cache()),
            Layer::Dense(dense_layer) => Cache::Dense((
                vec![0.0; dense_layer.input_size()],
                vec![0.0; dense_layer.hidden_size()],
            )),
        }
    }
}

#[derive(Debug)]
pub enum Cache {
    Lstm(LSTMCache),
    Dense((Vec<f32>, Vec<f32>)),
}

impl Cache {
    pub fn output(&self) -> &[f32] {
        match self {
            Self::Lstm(lstmcache) => &lstmcache.states[DH],
            Self::Dense(items) => &items.1,
        }
    }

    pub fn lstm(&mut self) -> &mut LSTMCache {
        if let Self::Lstm(lstm) = self {
            lstm
        } else {
            unreachable!()
        }
    }

    pub fn dense(&mut self) -> &mut (Vec<f32>, Vec<f32>) {
        if let Self::Dense(dense) = self {
            dense
        } else {
            unreachable!()
        }
    }
}

pub enum LayerGrads {
    Lstm(LSTMLayerGrads),
    Dense(DenseLayerGrads),
}

impl LayerGrads {
    pub fn from_layer(layer: &Layer) -> Self {
        match layer {
            Layer::Lstm(lstm) => {
                let rows = lstm.wc.rows();
                let cols = lstm.wc.cols();
                Self::Lstm(LSTMLayerGrads {
                    wf: Matrix::zeros(rows, cols),
                    wi: Matrix::zeros(rows, cols),
                    wc: Matrix::zeros(rows, cols),
                    wo: Matrix::zeros(rows, cols),
                    b: Matrix::zeros(4, cols),
                })
            }
            Layer::Dense(dense) => {
                let rows = dense.weights.rows();
                let cols = dense.weights.cols();
                Self::Dense(DenseLayerGrads {
                    weights: Matrix::zeros(rows, cols),
                    biases: vec![0.0; cols],
                })
            }
        }
    }

    pub fn lstm(&mut self) -> &mut LSTMLayerGrads {
        if let Self::Lstm(lstm) = self {
            lstm
        } else {
            unreachable!()
        }
    }

    pub fn dense(&mut self) -> &mut DenseLayerGrads {
        if let Self::Dense(dense) = self {
            dense
        } else {
            unreachable!()
        }
    }

    pub fn clear(&mut self) {
        match self {
            Self::Lstm(grads) => {
                grads.wc.clear();
                grads.wf.clear();
                grads.wi.clear();
                grads.wo.clear();
                grads.b.clear();
            }
            Self::Dense(grads) => {
                grads.weights.clear();
                grads.biases.fill(0.0);
            }
        }
    }

    pub fn scale(&mut self, scale: f32) {
        match self {
            Self::Lstm(grads) => {
                grads.wc.scale(scale);
                grads.wf.scale(scale);
                grads.wi.scale(scale);
                grads.wo.scale(scale);
                grads.b.scale(scale);
            }
            Self::Dense(grads) => {
                grads.weights.scale(scale);
                grads.biases.iter_mut().for_each(|x| *x *= scale);
            }
        }
    }

    pub fn clip(&mut self) {
        match self {
            Self::Lstm(grads) => {
                grads.wc.clip(-CLIP, CLIP);
                grads.wf.clip(-CLIP, CLIP);
                grads.wi.clip(-CLIP, CLIP);
                grads.wo.clip(-CLIP, CLIP);
                grads.b.clip(-CLIP, CLIP);
            }
            Self::Dense(grads) => {
                grads.weights.clip(-CLIP, CLIP);
                grads
                    .biases
                    .iter_mut()
                    .for_each(|x| *x = x.clamp(-CLIP, CLIP));
            }
        }
    }
}

pub enum LayerBuilder {
    LSTM(usize),
    Dense(usize, Activation),
}

impl LayerBuilder {
    pub fn layer_random_init(self, input_size: &mut usize) -> Layer {
        match self {
            LayerBuilder::LSTM(hidden_size) => {
                let l = Layer::Lstm(LSTMLayer::random(*input_size, hidden_size));
                *input_size = hidden_size;
                l
            }
            LayerBuilder::Dense(hidden_size, activation) => {
                let l = Layer::Dense(DenseLayer::random(*input_size, hidden_size, activation));
                *input_size = hidden_size;
                l
            }
        }
    }
}
