use std::{fs::File, io::{self, BufReader, BufWriter, Read, Write}, ops::Range, time::{Duration, Instant}};

use iron_oxide::collections::Matrix;
use rand::random_range;

use crate::{
    batches::Batches,
    layer::{ActivationFn, LearningLayer, sigmoid, softmax},
};

const CLIP: f32 = 1.0;

pub struct LSTMLayer {
    pub input_size: usize,
    pub hidden_size: usize,

    pub wf: Matrix,
    pub wi: Matrix,
    pub wc: Matrix,
    pub wo: Matrix,

    pub b: Matrix,

    pub dh: Box<[f32]>,
    pub dc: Box<[f32]>,
}

impl LSTMLayer {
    pub fn random(input_size: usize, hidden_size: usize) -> Self {
        let rows = input_size + hidden_size;
        Self {
            input_size,
            hidden_size,

            wf: Matrix::random(rows, hidden_size, 0.08),
            wi: Matrix::random(rows, hidden_size, 0.08),
            wc: Matrix::random(rows, hidden_size, 0.08),
            wo: Matrix::random(rows, hidden_size, 0.08),

            b: Matrix::zeros(4, hidden_size),

            dh: vec![0.0; hidden_size].into(),
            dc: vec![0.0; hidden_size].into(),
        }
    }
}

pub struct LSTM {
    pub layers: Box<[LSTMLayer]>,
    pub output_layer: LearningLayer,
    pub grads: LSTMGrads,
}

impl LSTM {
    pub fn new(layout: &[usize], output_size: usize) -> Self {
        let layers: Box<[LSTMLayer]> = layout
            .windows(2)
            .map(|window| LSTMLayer::random(window[0], window[1]))
            .collect();

        let output_layer =
            LearningLayer::random(layout[layout.len() - 1], output_size, ActivationFn::Softmax);

        let mut grads = LSTMGrads {
            layers: Vec::with_capacity(layers.len()),
            wy: Matrix::uninit(
                output_layer.weights.rows(),
                output_layer.weights.cols(),
            ),
            by: vec![0.0; output_layer.biases.len()].into(),
            dh_prev: Vec::with_capacity(layers.len()),
            dc_prev: Vec::with_capacity(layers.len()),
        };

        for layer in &layers {
            let rows = layer.input_size + layer.hidden_size;
            grads.layers.push(LSTMLayerGrads {
                wf: Matrix::zeros(rows, layer.hidden_size),
                wi: Matrix::zeros(rows, layer.hidden_size),
                wc: Matrix::zeros(rows, layer.hidden_size),
                wo: Matrix::zeros(rows, layer.hidden_size),

                b: Matrix::zeros(4, layer.hidden_size),
            });
            grads.dh_prev.push(vec![0.0; layer.hidden_size].into());
            grads.dc_prev.push(vec![0.0; layer.hidden_size].into());
        }

        Self {
            layers,
            output_layer,
            grads,
        }
    }

    pub fn train(&mut self, input: &[u16], seq_len: Range<usize>, epochs: usize) {
        for layer in &mut self.layers {
            layer.dh.fill(0.0);
            layer.dc.fill(0.0);
        }
        for epoch in 0..epochs {
            let mut forward_time = Duration::new(0, 0);
            let mut backwards_time = Duration::new(0, 0);

            let mut total_loss = 0.0;
            let mut steps = 0;

            for (inputs, targets) in Batches::new(input, seq_len.clone()) {

                let forward_start_time = Instant::now();
                let (caches, probs_list) =
                    self.forward_sequence(inputs);
                forward_time += forward_start_time.elapsed();

                let l = seq_loss(&probs_list, targets);

                total_loss += l;
                steps += 1;

                let backwards_start_time = Instant::now();
                self.backwards_sequence(inputs, targets, caches, probs_list);
                backwards_time += backwards_start_time.elapsed();


                self.sgd_step(0.001);
                //self.clear_grads();
                self.scale_grads(0.7);
            }

            println!(
                "Epoch {epoch} average loss = {:.4}",
                total_loss / steps.max(1) as f32
            );

            self.save("Jarvis").unwrap();

            //println!("forward time: {:?}", forward_time / steps.max(1));
            //println!("backward time: {:?}", backwards_time / steps.max(1));
        }
        self.clear_grads();
    }

    pub fn forward_sequence(
        &mut self,
        input: &[u16],
    ) -> (Vec<Vec<LSTMCache>>, Matrix) {
        let t = input.len();
        let mut caches = Vec::with_capacity(t);
        let mut props_list = Matrix::uninit(t, self.output_layer.weights.cols());
        
        for t in 0..t {
            caches.push(Vec::with_capacity(self.layers.len()));
            let mut input_vec = &one_hot(input[t] as _, self.layers.first().unwrap().input_size);

            for layer in &mut self.layers {
                let h_prev = layer.dh.clone();
                let c_prev = layer.dc.clone();

                let xh = [input_vec.to_vec().into_boxed_slice(), h_prev.clone()]
                    .concat()
                    .into_boxed_slice();

                let ft: Box<[f32]> = layer
                    .wf
                    .row_mul(&xh)
                    .into_iter()
                    .zip(&layer.b[0])
                    .map(|(x, b)| sigmoid(x + b))
                    .collect();
                let it: Box<[f32]> = layer
                    .wi
                    .row_mul(&xh)
                    .into_iter()
                    .zip(&layer.b[1])
                    .map(|(x, b)| sigmoid(x + b))
                    .collect();
                let ct: Box<[f32]> = layer
                    .wc
                    .row_mul(&xh)
                    .into_iter()
                    .zip(&layer.b[2])
                    .map(|(x, b)| (x + b).tanh())
                    .collect();
                layer.dc = add_vec(&mul_vec(&ft, &c_prev), &mul_vec(&it, &ct)).into_boxed_slice();
                let ot: Box<[f32]> = layer
                    .wo
                    .row_mul(&xh)
                    .into_iter()
                    .zip(&layer.b[3])
                    .map(|(x, b)| sigmoid(x + b))
                    .collect();
                
                layer.dh = mul_vec(
                    &ot,
                    &layer.dc
                        .clone()
                        .into_iter()
                        .map(|x| x.tanh())
                        .collect::<Box<[f32]>>(),
                )
                .into_boxed_slice();

                caches[t].push(LSTMCache {
                    xh,
                    h_prev,
                    c_prev,
                    ft,
                    it,
                    ct,
                    ot,
                    c: layer.dc.clone(),
                    h: layer.dh.clone(),
                });

                input_vec = &layer.dh;
            }

            props_list[t].copy_from_slice(&self.output_layer.forward(&input_vec));
        }
        (caches, props_list)
    }

    pub fn forward_sample(
        &mut self,
        input: &[u16],
    ) -> Matrix {
        let t = input.len();
        let mut logits_list = Matrix::uninit(t, self.output_layer.weights.cols());
        
        for t in 0..t {
            let mut input_vec = &one_hot(input[t] as _, self.layers.first().unwrap().input_size);

            for layer in &mut self.layers {
                let h_prev = layer.dh.clone();
                let c_prev = layer.dc.clone();

                let xh = [input_vec.to_vec().into_boxed_slice(), h_prev.clone()]
                    .concat()
                    .into_boxed_slice();

                let ft: Box<[f32]> = layer
                    .wf
                    .row_mul(&xh)
                    .into_iter()
                    .zip(&layer.b[0])
                    .map(|(x, b)| sigmoid(x + b))
                    .collect();
                let it: Box<[f32]> = layer
                    .wi
                    .row_mul(&xh)
                    .into_iter()
                    .zip(&layer.b[1])
                    .map(|(x, b)| sigmoid(x + b))
                    .collect();
                let ct: Box<[f32]> = layer
                    .wc
                    .row_mul(&xh)
                    .into_iter()
                    .zip(&layer.b[2])
                    .map(|(x, b)| (x + b).tanh())
                    .collect();
                layer.dc = add_vec(&mul_vec(&ft, &c_prev), &mul_vec(&it, &ct)).into_boxed_slice();
                let ot: Box<[f32]> = layer
                    .wo
                    .row_mul(&xh)
                    .into_iter()
                    .zip(&layer.b[3])
                    .map(|(x, b)| sigmoid(x + b))
                    .collect();
                
                layer.dh = mul_vec(
                    &ot,
                    &layer.dc
                        .clone()
                        .into_iter()
                        .map(|x| x.tanh())
                        .collect::<Box<[f32]>>(),
                )
                .into_boxed_slice();

                input_vec = &layer.dh;
            }

            logits_list[t].copy_from_slice(&self.output_layer.forward_no_activation(&input_vec));
        }
        logits_list
    }

    fn backwards_sequence(
        &mut self,
        seq: &[u16],
        targets: &[u16],
        caches: Vec<Vec<LSTMCache>>,
        mut probs_list: Matrix,
    ) {
        let mut dh_next = Vec::with_capacity(self.layers.len());
        let mut dc_next = Vec::with_capacity(self.layers.len());

        for lay in &self.layers {
            dh_next.push(vec![0.0; lay.hidden_size].into_boxed_slice());
            dc_next.push(vec![0.0; lay.hidden_size].into_boxed_slice());
        }

        for t in (0..seq.len()).rev() {
            let dy = &mut probs_list[t];
            dy[targets[t] as usize] -= 1.0;

            // add Wy, by grads
            // dWy += h_top^T ⊗ dy
            let top_cache = caches[t].last().unwrap();
            self.grads.wy.add_inplace(&outer(&top_cache.h, &dy));
            add_vec_in_place(&mut self.grads.by, &dy);

            // dh for top layer receives Wy^T * dy plus dhNext from future
            let mut dh_top = dh_next.last().unwrap().clone();
            for i in 0..self.output_layer.input_size() {
                let mut s = 0.0;
                for j in 0..self.output_layer.hidden_size() {
                    s += dy[j] * self.output_layer.weights[i][j];
                }
                dh_top[i] += s;
            }

            // Propagate through layers top->bottom at this time step
            let mut dh_layer = dh_top; // dh entering top layer at time t (including future contribution)
            for l in (0..self.layers.len()).rev() {
                let layer = &mut self.layers[l];
                let cache = &caches[t][l];

                // compute gradients for gates (elementwise)
                // do = dh ⊙ tanh(c) * σ'(o)
                let tanh_c = ActivationFn::Tanh.activate(cache.c.clone());
                let mut do_ = vec![0.0; layer.hidden_size];
                for i in 0..do_.len() {
                    do_[i] = dh_layer[i] * tanh_c[i] * dsigmoid_from_y(cache.ot[i])
                }

                // dc = dh ⊙ o * (1 - tanh(c)^2) + dcNext
                let mut dc = dc_next[l].clone();
                for i in 0..do_.len() {
                    dc[i] += dh_layer[i] * cache.ot[i] * dtanh_from_y(tanh_c[i])
                }

                // df = dc ⊙ cPrev * σ'(f)
                let mut df = vec![0.0; layer.hidden_size];
                for i in 0..df.len() {
                    df[i] = dc[i] * cache.c_prev[i] * dsigmoid_from_y(cache.ft[i]);
                }

                // di = dc ⊙ cT * σ'(i)
                let mut di = vec![0.0; layer.hidden_size];
                for i in 0..di.len() {
                    di[i] = dc[i] * cache.ct[i] * dsigmoid_from_y(cache.it[i])
                }

                // dcT = dc ⊙ i * (1 - cT^2)
                let mut dct = vec![0.0; layer.hidden_size];
                for i in 0..dct.len() {
                    dct[i] = dc[i] * cache.it[i] * dtanh_from_y(cache.ct[i])
                }

                self.grads.layers[l].wf.add_inplace(&outer(&cache.xh, &df));
                self.grads.layers[l].wi.add_inplace(&outer(&cache.xh, &di));
                self.grads.layers[l].wc.add_inplace(&outer(&cache.xh, &dct));
                self.grads.layers[l].wo.add_inplace(&outer(&cache.xh, &do_));
                add_vec_in_place(&mut self.grads.layers[l].b[0], &df);
                add_vec_in_place(&mut self.grads.layers[l].b[1], &di);
                add_vec_in_place(&mut self.grads.layers[l].b[2], &dct);
                add_vec_in_place(&mut self.grads.layers[l].b[3], &do_);

                let mut dconcat = vec![0.0; layer.input_size + layer.hidden_size];
                let mut accum_back = |d_gate: &[f32], w: &Matrix| {
                    for i in 0..dconcat.len() {
                        let mut s = 0.0;
                        for j in 0..layer.hidden_size {
                            s += d_gate[j] * w[i][j];
                        }
                        dconcat[i] += s;
                    }
                };
                accum_back(&df, &layer.wf);
                accum_back(&di, &layer.wi);
                accum_back(&dct, &layer.wc);
                accum_back(&do_, &layer.wo);

                let dx = &dconcat[0..layer.input_size];
                let dh_prev = &dconcat[layer.input_size..];

                if l > 0 {
                    dh_layer = dh_next[l - 1].clone();
                    for i in 0..layer.input_size {
                        dh_layer[i] += dx[i];
                    }
                } else {
                    dh_layer = vec![0.0; layer.hidden_size].into();
                }

                // update dcNext and dhNext for this layer to be used at previous time-step (t-1)
                // dcNext for layer l becomes dc ⊙ f_t (as in single-layer derivation)
                let mut dc_prev_next = vec![0.0; layer.hidden_size].into_boxed_slice();
                for i in 0..layer.hidden_size {
                    dc_prev_next[i] = dc[i] * cache.ft[i];
                }
                dc_next[l] = dc_prev_next;

                // dhNext for this layer (for previous time step) is dhPrev (the portion from dconcat corresponding to h_{t-1})
                dh_next[l] = dh_prev.to_vec().into();
            }
        }
    }

    fn sgd_step(&mut self, lr: f32) {
        let grads = &mut self.grads;
        for g in &mut grads.layers {
            g.wf.clip(-CLIP, CLIP);
            g.wi.clip(-CLIP, CLIP);
            g.wc.clip(-CLIP, CLIP);
            g.wo.clip(-CLIP, CLIP);

            g.b.as_slice_mut().iter_mut().for_each(|x| *x = x.clamp(-CLIP, CLIP));
        }

        grads.wy.clip(-CLIP, CLIP);
        grads.by.iter_mut().for_each(|x| *x = x.clamp(-CLIP, CLIP));

        for (layer, g) in self.layers.iter_mut().zip(&grads.layers) {
            sub_in_place(&mut layer.wf, &g.wf, lr);
            sub_in_place(&mut layer.wi, &g.wi, lr);
            sub_in_place(&mut layer.wc, &g.wc, lr);
            sub_in_place(&mut layer.wo, &g.wo, lr);
            sub_vec_in_player(&mut layer.b[0], &g.b[0], lr);
            sub_vec_in_player(&mut layer.b[1], &g.b[1], lr);
            sub_vec_in_player(&mut layer.b[2], &g.b[2], lr);
            sub_vec_in_player(&mut layer.b[3], &g.b[3], lr);
        }

        self.output_layer.weights.add_inplace_scaled(&grads.wy, -lr);
        self.output_layer
            .biases
            .iter_mut()
            .zip(grads.by.iter())
            .for_each(|(w, g)| *w -= lr * g);
    }

    pub fn clear_grads(&mut self) {
        let grads = &mut self.grads;
        for g in &mut grads.layers {
            g.wf.clear();
            g.wi.clear();
            g.wc.clear();
            g.wo.clear();

            g.b.clear();
        }

        grads.wy.clear();
        grads.by.fill(0.0);
    }

    pub fn scale_grads(&mut self, scale: f32) {
        let grads = &mut self.grads;
        for g in &mut grads.layers {
            g.wf.scale(scale);
            g.wi.scale(scale);
            g.wc.scale(scale);
            g.wo.scale(scale);

            g.b.as_slice_mut().iter_mut().for_each(|x| *x *= scale);
        }

        grads.wy.scale(scale);
        grads.by.iter_mut().for_each(|x| *x *= scale);
    }

    pub fn sample(&mut self, prefix: &[u16], max_len: usize, temperature: f32) -> Vec<u16> {
        for layer in &mut self.layers {
            layer.dh.fill(0.0);
            layer.dc.fill(0.0);
        }

        let mut last_token;

        if prefix.is_empty() {
            last_token = random_range(0..self.output_layer.hidden_size()) as u16;
        } else {
            let _ = self.forward_sample(&prefix[0..prefix.len() - 1]);
            last_token = prefix[prefix.len() - 1];
        }

        
        let mut out = Vec::new();
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
            last_token = next;
        }

        out
    }

    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);

        write_u32(&mut writer, self.layers.len() as u32)?;
        if !self.layers.is_empty() {
            write_u32(&mut writer, self.layers[0].input_size as u32)?;
        }
        for layer in &self.layers {
            write_u32(&mut writer, layer.hidden_size as u32)?;
        }
        write_u32(&mut writer, self.output_layer.weights.cols() as u32)?;

        for layer in &self.layers {
            write_matrix(&mut writer, &layer.wf)?;
            write_matrix(&mut writer, &layer.wi)?;
            write_matrix(&mut writer, &layer.wc)?;
            write_matrix(&mut writer, &layer.wo)?;
            
            write_matrix(&mut writer, &layer.b)?;
        }

        write_matrix(&mut writer, &self.output_layer.weights)?;
        write_vec(&mut writer, &self.output_layer.biases)?;

        Ok(())
    }

    pub fn load(path: &str) -> io::Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);

        let num_layers = read_u32(&mut reader)? as usize;
        let input_size = if num_layers > 0 { read_u32(&mut reader)? as usize } else { 0 };
        let mut hidden_sizes = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            hidden_sizes.push(read_u32(&mut reader)? as usize);
        }
        let output_size = read_u32(&mut reader)? as usize;

        let mut layout = vec![input_size];
        layout.extend(hidden_sizes.iter().cloned());

        let mut lstm = Self::new(&layout, output_size);

        for layer in lstm.layers.iter_mut() {
            let rows = layer.input_size + layer.hidden_size;
            layer.wf = read_matrix(&mut reader, rows, layer.hidden_size)?;
            layer.wi = read_matrix(&mut reader, rows, layer.hidden_size)?;
            layer.wc = read_matrix(&mut reader, rows, layer.hidden_size)?;
            layer.wo = read_matrix(&mut reader, rows, layer.hidden_size)?;
            layer.b = read_matrix(&mut reader, 4, layer.hidden_size)?;
        }

        let output_input_size = if let Some(last_hidden) = layout.last() { *last_hidden } else { 0 };
        lstm.output_layer.weights = read_matrix(&mut reader, output_input_size, output_size)?;
        lstm.output_layer.biases = read_vec(&mut reader, output_size)?;

        Ok(lstm)
    }
}

pub struct LSTMCache {
    pub xh: Box<[f32]>,

    pub h_prev: Box<[f32]>,
    pub c_prev: Box<[f32]>,

    pub ft: Box<[f32]>,
    pub it: Box<[f32]>,
    pub ct: Box<[f32]>,
    pub ot: Box<[f32]>,

    pub c: Box<[f32]>,
    pub h: Box<[f32]>,
}

pub struct LSTMGrads {
    pub layers: Vec<LSTMLayerGrads>,
    pub wy: Matrix,
    pub by: Box<[f32]>,
    pub dh_prev: Vec<Box<[f32]>>,
    pub dc_prev: Vec<Box<[f32]>>,
}

pub struct LSTMLayerGrads {
    pub wf: Matrix,
    pub wi: Matrix,
    pub wc: Matrix,
    pub wo: Matrix,

    pub b: Matrix,
}

fn one_hot(index: usize, size: usize) -> Box<[f32]> {
    let mut out = vec![0.0; size].into_boxed_slice();
    out[index] = 1.0;
    out
}

fn mul_vec(x: &[f32], y: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), y.len());
    x.iter().zip(y).map(|(x, y)| x * y).collect()
}

fn add_vec(x: &[f32], y: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), y.len());
    x.iter().zip(y).map(|(x, y)| x + y).collect()
}

fn add_vec_in_place(x: &mut [f32], y: &[f32]) {
    assert_eq!(x.len(), y.len());
    x.iter_mut().zip(y).for_each(|(x, y)| *x += y);
}

fn seq_loss(probs_list: &Matrix, targets: &[u16]) -> f32 {
    let mut l = 0.0;
    for t in 0..targets.len() {
        let p = probs_list[t][targets[t] as usize] + 1e-12;
        l += -p.ln();
    }
    return l / targets.len() as f32;
}

fn outer(a: &[f32], b: &[f32]) -> Matrix {
    let mut mtx = Matrix::uninit(a.len(), b.len());
    for (x, &ai) in a.iter().enumerate() {
        for (y, &bj) in b.iter().enumerate() {
            mtx[x][y] = ai * bj;
        }
    }

    mtx
}

fn dsigmoid_from_y(y: f32) -> f32 {
    y * (1.0 - y)
}

fn dtanh_from_y(y: f32) -> f32 {
    1.0 - y * y
}

fn sub_in_place(a: &mut Matrix, b: &Matrix, lr: f32) {
    assert_eq!(
        a.rows(),
        b.rows(),
        "rows do not match, {} to {}",
        a.rows(),
        b.rows()
    );
    assert_eq!(
        a.cols(),
        b.cols(),
        "cols do not match, {} to {}",
        a.cols(),
        b.cols()
    );

    a.as_slice_mut()
        .iter_mut()
        .zip(b.as_slice())
        .for_each(|(a, b)| *a -= lr * b);
}

fn sub_vec_in_player(a: &mut [f32], b: &[f32], lr: f32) {
    a.iter_mut().zip(b).for_each(|(a, b)| *a -= lr * b);
}

fn write_u32<W: Write>(writer: &mut W, val: u32) -> io::Result<()> {
    writer.write_all(&val.to_le_bytes())
}

fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn write_matrix<W: Write>(writer: &mut W, matrix: &Matrix) -> io::Result<()> {
    for i in matrix.as_slice() {
        writer.write_all(&i.to_le_bytes())?;
    }
    Ok(())
}

fn read_matrix<R: Read>(reader: &mut R, rows: usize, cols: usize) -> io::Result<Matrix> {
    let mut matrix = Matrix::uninit(rows, cols);
    let mut buf = [0u8; 4];
    for row in 0..rows {
        for col in 0..cols {
            reader.read_exact(&mut buf)?;
            matrix[row][col] = f32::from_le_bytes(buf);
        }
    }
    Ok(matrix)
}

fn write_vec<W: Write>(writer: &mut W, vec: &[f32]) -> io::Result<()> {
    for &val in vec {
        writer.write_all(&val.to_le_bytes())?;
    }
    Ok(())
}

fn read_vec<R: Read>(reader: &mut R, len: usize) -> io::Result<Box<[f32]>> {
    let mut vec = vec![0.0; len].into_boxed_slice();
    let mut buf = [0u8; 4];
    for i in 0..vec.len() {
        reader.read_exact(&mut buf)?;
        vec[i] = f32::from_le_bytes(buf);
    }
    Ok(vec)
}
