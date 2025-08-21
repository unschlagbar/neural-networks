use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Read, Write},
    ops::Range,
    time::{Duration, Instant},
};

use iron_oxide::collections::Matrix;
use rand::random_range;

use crate::{
    batches::Batches,
    layer::{Activation, LearningLayer, sigmoid, softmax},
};

const CLIP: f32 = 1.0;
const F: usize = 0;
const I: usize = 1;
const C: usize = 2;
const O: usize = 3;

const DH: usize = 0;
const DC: usize = 1;

pub struct LSTMLayer {
    pub input_size: usize,
    pub hidden_size: usize,

    pub wf: Matrix,
    pub wi: Matrix,
    pub wc: Matrix,
    pub wo: Matrix,

    pub b: Matrix,
    pub d: Matrix,
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
            d: Matrix::zeros(2, hidden_size),
        }
    }

    pub fn forward(&mut self, input: &[f32]) -> LSTMCache {
        let states_prev = self.d.clone();

        // Combine input + previous hidden state
        let mut xh = Vec::with_capacity(self.input_size + self.hidden_size);
        xh.extend_from_slice(input);
        xh.extend_from_slice(&states_prev[DH]);

        // Compute gates with minimal allocations
        let mut ft = self.wf.row_mul(&xh);
        let mut it = self.wi.row_mul(&xh);
        let mut ct = self.wc.row_mul(&xh);
        let mut ot = self.wo.row_mul(&xh);

        let h = self.hidden_size;

        for j in 0..h {
            ft[j] = sigmoid(ft[j] + self.b[F][j]);
            it[j] = sigmoid(it[j] + self.b[I][j]);
            ct[j] = (ct[j] + self.b[C][j]).tanh();
            ot[j] = sigmoid(ot[j] + self.b[O][j]);
        }

        // Update cell state: c_t = f_t * c_{t-1} + i_t * c~_t
        let prev_c = &states_prev[DC];
        for j in 0..h {
            self.d[DC][j] = ft[j] * prev_c[j] + it[j] * ct[j];
        }

        // Compute hidden state: h_t = o_t * tanh(c_t)
        for j in 0..h {
            self.d[DH][j] = ot[j] * self.d[DC][j].tanh();
        }

        LSTMCache {
            xh: xh.into_boxed_slice(),
            states_prev,
            ft: ft.into_boxed_slice(),
            it: it.into_boxed_slice(),
            ct: ct.into_boxed_slice(),
            ot: ot.into_boxed_slice(),
            states: self.d.clone(),
        }
    }

    pub fn backwards(
        &mut self,
        mut cache: LSTMCache,
        dh: &mut [f32],
        dc_next: &mut [f32],
        grads: &mut LSTMLayerGrads,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let h = self.hidden_size;
        let rows = self.input_size + self.hidden_size;

        // 1) prepare: compute tanh(c_t) once (same as original code)
        // Note: caller already used Activation::Tanh.activate in original; keep same order:
        Activation::Tanh.activate(&mut cache.states[DC]);

        // 2) do = dh ⊙ tanh(c) * σ'(o)
        let mut do_ = vec![0.0f32; h];
        for i in 0..h {
            do_[i] = dh[i] * cache.states[DC][i] * dsigmoid_from_y(cache.ot[i]);
        }

        // 3) dc = dh ⊙ o * (1 - tanh(c)^2) + dc_next
        let mut dc = dc_next.to_vec(); // small allocation: size = hidden_size
        for i in 0..h {
            dc[i] += dh[i] * cache.ot[i] * dtanh_from_y(cache.states[DC][i]);
        }

        // 4) gate grads: df, di, dct
        let mut df = vec![0.0f32; h];
        let mut di = vec![0.0f32; h];
        let mut dct = vec![0.0f32; h];
        for i in 0..h {
            df[i] = dc[i] * cache.states_prev[DC][i] * dsigmoid_from_y(cache.ft[i]);
            di[i] = dc[i] * cache.ct[i] * dsigmoid_from_y(cache.it[i]);
            dct[i] = dc[i] * cache.it[i] * dtanh_from_y(cache.ct[i]);
        }

        // 5) accumulate grads WITHOUT allocating outer products:
        // grads.w*.rows = rows, cols = h
        // cache.xh.len() == rows
        // Update weight matrices: grads.wf[i][j] += xh[i] * df[j]
        for i in 0..rows {
            let xi = cache.xh[i];
            let wf_row = &mut grads.wf[i];
            let wi_row = &mut grads.wi[i];
            let wc_row = &mut grads.wc[i];
            let wo_row = &mut grads.wo[i];
            for j in 0..h {
                // inline multiply-add beats forming an outer-product Matrix then add_inplace
                wf_row[j] += xi * df[j];
                wi_row[j] += xi * di[j];
                wc_row[j] += xi * dct[j];
                wo_row[j] += xi * do_[j];
            }
        }

        // biases
        for j in 0..h {
            grads.b[0][j] += df[j];
            grads.b[1][j] += di[j];
            grads.b[2][j] += dct[j];
            grads.b[3][j] += do_[j];
        }

        // 6) backprop to inputs+prev-hidden: dconcat = Wf^T * df + Wi^T * di + ...
        // compute row-wise accumulation to keep contiguous memory access on W rows
        let mut dconcat = vec![0.0f32; rows];
        for i in 0..rows {
            let wf_row = &self.wf[i];
            let wi_row = &self.wi[i];
            let wc_row = &self.wc[i];
            let wo_row = &self.wo[i];
            let mut s = 0.0f32;
            // sum over columns (hidden units)
            for j in 0..h {
                s += wf_row[j] * df[j];
                s += wi_row[j] * di[j];
                s += wc_row[j] * dct[j];
                s += wo_row[j] * do_[j];
            }
            dconcat[i] = s;
        }

        // split dx and dh_prev
        let dh_prev = dconcat.split_off(self.input_size);
        let dx = dconcat;

        // dc_prev (for previous time-step) = dc ⊙ f_t
        let dc_prev: Vec<f32> = dc.iter().zip(&cache.ft).map(|(a, b)| a * b).collect();

        (dh_prev, dc_prev, dx)
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
            LearningLayer::random(layout[layout.len() - 1], output_size, Activation::Softmax);

        let mut grads = LSTMGrads {
            layers: Vec::with_capacity(layers.len()),
            wy: Matrix::zeros(output_layer.weights.rows(), output_layer.weights.cols()),
            by: vec![0.0; output_layer.biases.len()].into(),
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
        }

        Self {
            layers,
            output_layer,
            grads,
        }
    }

    pub fn train(&mut self, input: &[u16], seq_len: Range<usize>, epochs: usize) {
        for layer in &mut self.layers {
            layer.d.clear();
        }
        for epoch in 0..epochs {
            let mut forward_time = Duration::new(0, 0);
            let mut backwards_time = Duration::new(0, 0);

            let mut total_loss = 0.0;
            let mut steps = 0;

            for (inputs, targets) in Batches::new(input, seq_len.clone()) {
                let forward_start_time = Instant::now();
                let (caches, probs_list) = self.forward_sequence(inputs);
                forward_time += forward_start_time.elapsed();

                let l = seq_loss(&probs_list, targets);

                total_loss += l;
                steps += 1;

                let backwards_start_time = Instant::now();
                self.backwards_sequence(inputs, targets, caches, probs_list);
                backwards_time += backwards_start_time.elapsed();

                self.sgd_step(0.001);
                self.scale_grads(0.7);
            }

            println!(
                "Epoch {epoch} average loss = {:.4}",
                total_loss / steps.max(1) as f32
            );

            if total_loss.is_nan() {
                *self = Self::load("Jarvis").unwrap();
            } else {
                self.save("Jarvis").unwrap();
            }

            println!("forward time: {:?}", forward_time / steps.max(1));
            println!("backward time: {:?}", backwards_time / steps.max(1));
        }
        self.scale_grads(0.5);
    }

    pub fn forward_sequence(&mut self, input: &[u16]) -> (Vec<Vec<LSTMCache>>, Matrix) {
        let t = input.len();
        let mut caches = Vec::with_capacity(t);
        let mut props_list = Matrix::uninit(t, self.output_layer.weights.cols());

        for t in 0..t {
            caches.push(Vec::with_capacity(self.layers.len()));
            let mut input_vec: &[f32] = &one_hot(input[t] as _, self.layers.first().unwrap().input_size);

            for layer in &mut self.layers {
                caches[t].push(layer.forward(input_vec));
                input_vec = &layer.d[DH];
            }

            props_list[t].copy_from_slice(&self.output_layer.forward(&input_vec));
        }
        (caches, props_list)
    }

    pub fn forward_sample(&mut self, input: &[u16]) -> Matrix {
        let t = input.len();
        let mut logits_list = Matrix::uninit(t, self.output_layer.weights.cols());

        for t in 0..t {
            let mut input_vec: &[f32] = &one_hot(input[t] as _, self.layers.first().unwrap().input_size);

            for layer in &mut self.layers {
                layer.forward(input_vec);
                input_vec = &layer.d[DH];
            }

            logits_list[t].copy_from_slice(&self.output_layer.forward_no_activation(&input_vec));
        }
        logits_list
    }

    fn backwards_sequence(
        &mut self,
        seq: &[u16],
        targets: &[u16],
        mut caches: Vec<Vec<LSTMCache>>,
        mut probs_list: Matrix,
    ) {
        let mut dh_next = Vec::with_capacity(self.layers.len());
        let mut dc_next = Vec::with_capacity(self.layers.len());

        for lay in &self.layers {
            dh_next.push(vec![0.0; lay.hidden_size]);
            dc_next.push(vec![0.0; lay.hidden_size]);
        }

        for t in (0..seq.len()).rev() {
            let dy = &mut probs_list[t];
            dy[targets[t] as usize] -= 1.0;

            // add Wy, by grads
            // dWy += h_top^T ⊗ dy
            let top_cache = caches[t].last().unwrap();
            self.grads.wy.add_inplace(&outer(&top_cache.states[DH], dy));
            add_vec_in_place(&mut self.grads.by, dy);

            // dh for top layer receives Wy^T * dy plus dhNext from future
            let mut dh_top = dh_next.last().unwrap().clone();
            for i in 0..self.output_layer.input_size() {
                let mut s = 0.0;
                for (&dy, &weight) in dy.iter().zip(&self.output_layer.weights[i]) {
                    s += dy * weight;
                }
                dh_top[i] += s;
            }

            // Propagate through layers top->bottom at this time step
            let mut dh_layer = dh_top; // dh entering top layer at time t (including future contribution)
            for l in (0..self.layers.len()).rev() {
                let layer = &mut self.layers[l];
                let cache = caches[t].pop().unwrap(); // ownership moved into call

                // call the new per-layer backward that returns dh_prev, dc_prev, dx
                // note: pass mutable refs to dh_layer and dc_next[l] so function can read them
                let (dh_prev, dc_prev, dx) = layer.backwards(
                    cache,
                    &mut dh_layer,
                    &mut dc_next[l],
                    &mut self.grads.layers[l],
                );

                // prepare dh_layer for next (lower) layer: combine dh_next from same time-step and dx
                if l > 0 {
                    // dh_next[l-1] already contains dh coming from future time-step for layer l-1
                    dh_layer = dh_next[l - 1].clone();
                    for i in 0..layer.input_size {
                        dh_layer[i] += dx[i];
                    }
                }

                // set dc_next for previous time-step: dc_prev
                dc_next[l] = dc_prev;
                // set dh_next for this layer (to be used when processing previous t)
                dh_next[l] = dh_prev;
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

            g.b.as_slice_mut()
                .iter_mut()
                .for_each(|x| *x = x.clamp(-CLIP, CLIP));
        }

        grads.wy.clip(-CLIP, CLIP);
        grads.by.iter_mut().for_each(|x| *x = x.clamp(-CLIP, CLIP));

        for (layer, g) in self.layers.iter_mut().zip(&grads.layers) {
            sub_in_place(&mut layer.wf, &g.wf, lr);
            sub_in_place(&mut layer.wi, &g.wi, lr);
            sub_in_place(&mut layer.wc, &g.wc, lr);
            sub_in_place(&mut layer.wo, &g.wo, lr);
            sub_vec_in_player(&mut layer.b.as_slice_mut(), &g.b.as_slice(), lr);
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

            g.b.scale(scale);
        }

        grads.wy.scale(scale);
        grads.by.iter_mut().for_each(|x| *x *= scale);
    }

    pub fn sample(&mut self, prefix: &[u16], max_len: usize, temperature: f32) -> Vec<u16> {
        for layer in &mut self.layers {
            layer.d.clear();
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
        let input_size = if num_layers > 0 {
            read_u32(&mut reader)? as usize
        } else {
            0
        };
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

        let output_input_size = if let Some(last_hidden) = layout.last() {
            *last_hidden
        } else {
            0
        };
        lstm.output_layer.weights = read_matrix(&mut reader, output_input_size, output_size)?;
        lstm.output_layer.biases = read_vec(&mut reader, output_size)?;

        Ok(lstm)
    }
}

pub struct LSTMCache {
    pub xh: Box<[f32]>,

    pub states_prev: Matrix,

    pub ft: Box<[f32]>,
    pub it: Box<[f32]>,
    pub ct: Box<[f32]>,
    pub ot: Box<[f32]>,

    pub states: Matrix,
}

pub struct LSTMGrads {
    pub layers: Vec<LSTMLayerGrads>,
    pub wy: Matrix,
    pub by: Box<[f32]>,
}

pub struct LSTMLayerGrads {
    pub wf: Matrix,
    pub wi: Matrix,
    pub wc: Matrix,
    pub wo: Matrix,

    pub b: Matrix,
}

fn one_hot(index: usize, size: usize) -> Vec<f32> {
    let mut out = vec![0.0; size];
    out[index] = 1.0;
    out
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
    l / targets.len() as f32
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
