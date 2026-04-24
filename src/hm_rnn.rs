// hm_rnn.rs
//
// Hierarchical Multiscale RNN — Chung, Ahn, Bengio (arXiv:1609.03777v2).
//
// Two-level architecture, both levels share the same timestep axis:
//
//   Level 1 (char): runs LSTM at every t.  After a boundary (z_{t-1}=1),
//     the LSTM state is reset to h=0,c=0 (FLUSH in the paper).
//
//   Level 2 (high): COPY when z_t=0, UPDATE (run LSTM) when z_t=1.
//     There is no FLUSH at the high level in a 2-level model.
//
// Boundary detector (level 1 → level 2):
//   logit_t  = w_z · h_t^rep + b_z          (h_t^rep = output of word_rep_layer)
//   z_soft_t = σ(slope · logit_t)
//   z_t      = (z_soft_t ≥ 0.5)             ← hard during forward
//
// Efficient forward: COPY steps skip the high-LSTM computation entirely.
//
// Correct BPTT:
//   • COPY  → dh^2 passes through unchanged
//   • UPDATE → standard LSTM BPTT inside the high model; initial delta is the
//              accumulated context-gradient from chars since the last update
//   • FLUSH (char) → char BPTT zeroed before processing the boundary char's
//              backward (see zero_bptt_state introduced in the previous fix)
//
// STE gradient for the boundary detector (UPDATE steps only):
//   d_z     = Σ dh_t^2 · (h_t^2 − h_{t−1}^2)
//   d_logit = d_z · slope · z_soft · (1 − z_soft)
//   d_w_z  += d_logit · h_t^rep
//   d_h_t^rep += d_logit · w_z    (injected into the char backward)

use std::{
    fs::File,
    io::{self, Cursor, Read, Write},
};

use crate::{
    activations::sigmoid,
    loading::{read_f32, read_f32_vec, read_u32},
    lstm::add_vec_in_place,
    nn_layer::{DynCache, NnLayer},
    saving::{HM_RNN_MAGIC, write_f32, write_f32_slice, write_u32},
    sequential::Sequential,
    softmax::softmax,
};

// ── HmRnn ─────────────────────────────────────────────────────────────────────

pub struct HmRnn {
    pub char_model: Sequential,
    pub high_model: Sequential,

    pub vocab_size: usize,
    pub context_size: usize,  // = high_model.output_size
    pub char_rep_size: usize, // = char_model.layers[word_rep_layer].output_size()
    pub word_rep_layer: usize,

    // ── Boundary detector ─────────────────────────────────────────────────────
    pub z_w: Box<[f32]>, // [char_rep_size]
    pub z_b: f32,
    z_w_grad: Box<[f32]>,
    z_b_grad: f32,
    pub slope: f32, // annealing slope (1.0 initially)

    // ── Forward running state ─────────────────────────────────────────────────
    high_context: Box<[f32]>, // current h^2 (top-down context fed to char at next t)
    z_prev: bool,             // was z_{t-1} = 1? (triggers FLUSH at char level for next t)

    // ── Per-sequence caches ───────────────────────────────────────────────────
    // Char: one full cache per timestep (always runs)
    char_cache: Vec<Vec<Box<dyn DynCache>>>,
    // High: one cache per UPDATE event (sparse; indexed by update number 0..n_updates)
    high_cache: Vec<Vec<Box<dyn DynCache>>>,

    // Per-timestep metadata (valid for 0..seq_len after forward_over)
    is_update: Vec<bool>, // true → UPDATE at t (z_t = 1)
    z_soft: Vec<f32>,     // σ(slope·logit_t) stored for STE backward
    // For UPDATE step at t: high_cache_idx[t] = j (index into high_cache)
    high_cache_idx: Vec<usize>,
    // h_{t−1}^2 at each UPDATE event (needed for STE; indexed same as high_cache)
    high_h_prev: Vec<Box<[f32]>>,

    seq_len: usize, // filled by last forward_over call

    // ── Scratch buffers (no alloc in hot path) ────────────────────────────────
    char_delta_buf: Vec<f32>,
    high_delta_buf: Vec<f32>,
    full_input_buf: Vec<f32>,
    ste_extra: Vec<f32>, // d_logit * w_z — injected at word_rep_layer

    pub last_boundary_rate: f32,
    pub last_high_grad_signal: f32,
}

impl HmRnn {
    pub fn new(
        char_model: Sequential,
        high_model: Sequential,
        vocab_size: usize,
        word_rep_layer: usize,
        slope: f32,
    ) -> Self {
        let context_size = high_model.output_size;
        let char_rep_size = char_model.layers[word_rep_layer].output_size();

        assert_eq!(
            char_model.input_size,
            vocab_size + context_size,
            "char_model input must be vocab_size + context_size"
        );
        assert_eq!(
            high_model.input_size, char_rep_size,
            "high_model input must equal char_rep_size"
        );

        let char_max = char_model
            .layers
            .iter()
            .map(|l| l.output_size())
            .max()
            .unwrap_or(0);
        let high_max = high_model
            .layers
            .iter()
            .map(|l| l.output_size())
            .max()
            .unwrap_or(0);

        // Xavier-init for boundary detector
        let scale = (2.0 / char_rep_size as f32).sqrt();
        let z_w: Box<[f32]> = (0..char_rep_size)
            .map(|_| (rand::random::<f32>() * 2.0 - 1.0) * scale)
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            vocab_size,
            context_size,
            char_rep_size,
            word_rep_layer,
            z_w,
            z_b: 0.0,
            z_w_grad: vec![0.0; char_rep_size].into_boxed_slice(),
            z_b_grad: 0.0,
            slope,
            high_context: vec![0.0; context_size].into_boxed_slice(),
            z_prev: false,
            char_cache: vec![],
            high_cache: vec![],
            is_update: vec![],
            z_soft: vec![],
            high_cache_idx: vec![],
            high_h_prev: vec![],
            seq_len: 0,
            char_delta_buf: vec![0.0; char_max],
            high_delta_buf: vec![0.0; high_max],
            full_input_buf: vec![0.0; vocab_size + context_size],
            ste_extra: vec![0.0; char_rep_size],
            last_boundary_rate: 0.0,
            last_high_grad_signal: 0.0,
            char_model,
            high_model,
        }
    }

    // ── Cache allocation ──────────────────────────────────────────────────────

    pub fn make_cache(&mut self, max_seq_len: usize) {
        self.char_cache = (0..max_seq_len)
            .map(|_| {
                self.char_model
                    .layers
                    .iter()
                    .map(|l| l.make_cache())
                    .collect()
            })
            .collect();
        self.high_cache = (0..max_seq_len)
            .map(|_| {
                self.high_model
                    .layers
                    .iter()
                    .map(|l| l.make_cache())
                    .collect()
            })
            .collect();
        self.is_update = vec![false; max_seq_len];
        self.z_soft = vec![0.0; max_seq_len];
        self.high_cache_idx = vec![0; max_seq_len];
        self.high_h_prev = (0..max_seq_len)
            .map(|_| vec![0.0f32; self.context_size].into_boxed_slice())
            .collect();
    }

    // ── State reset ───────────────────────────────────────────────────────────

    pub fn reset(&mut self) {
        for layer in &mut self.char_model.layers {
            layer.reset_state();
        }
        for layer in &mut self.high_model.layers {
            layer.reset_state();
        }
        self.high_context.fill(0.0);
        self.z_prev = false;
    }

    // ── Forward helpers ───────────────────────────────────────────────────────

    fn forward_step(
        layers: &mut [Box<dyn NnLayer>],
        cache_t: &mut [Box<dyn DynCache>],
        input: &[f32],
    ) {
        for l in 0..layers.len() {
            let (left, right) = cache_t.split_at_mut(l);
            let inp = if l == 0 { input } else { left[l - 1].output() };
            layers[l].forward(inp, right[0].as_mut());
        }
    }

    fn forward_sample_step(
        layers: &mut [Box<dyn NnLayer>],
        cache_t: &mut [Box<dyn DynCache>],
        input: &[f32],
    ) {
        for l in 0..layers.len() {
            let (left, right) = cache_t.split_at_mut(l);
            let inp = if l == 0 { input } else { left[l - 1].output() };
            layers[l].forward_sample(inp, right[0].as_mut());
        }
    }

    #[inline]
    fn fill_input_buf(&mut self, token: u16) {
        self.full_input_buf[..self.vocab_size].fill(0.0);
        self.full_input_buf[token as usize] = 1.0;
        self.full_input_buf[self.vocab_size..].copy_from_slice(&self.high_context);
    }

    // ── Boundary detector ─────────────────────────────────────────────────────

    #[inline]
    fn boundary_forward(&self, h_rep: &[f32]) -> (bool, f32) {
        let logit: f32 = self.z_w.iter().zip(h_rep).map(|(w, h)| w * h).sum::<f32>() + self.z_b;
        let z_soft = sigmoid(self.slope * logit);
        (z_soft >= 0.5, z_soft)
    }

    // ── Training forward ──────────────────────────────────────────────────────

    pub fn forward_over(&mut self, input: &[u16]) {
        let seq_len = input.len();
        self.seq_len = seq_len;
        let mut n_updates: usize = 0;

        for t in 0..seq_len {
            // FLUSH at char level if z_{t−1} = 1
            if self.z_prev {
                for layer in &mut self.char_model.layers {
                    layer.reset_state();
                }
            }

            // Char forward
            self.fill_input_buf(input[t]);
            Self::forward_step(
                &mut self.char_model.layers,
                &mut self.char_cache[t],
                &self.full_input_buf,
            );

            // Boundary detection on char rep
            let h_rep = self.char_cache[t][self.word_rep_layer].output();
            let (z, z_soft) = self.boundary_forward(h_rep);
            self.is_update[t] = z;
            self.z_soft[t] = z_soft;

            // High level: UPDATE or COPY
            if z {
                // Store h_{t−1}^2 for STE backward
                self.high_h_prev[n_updates].copy_from_slice(&self.high_context);
                self.high_cache_idx[t] = n_updates;

                let h_rep = self.char_cache[t][self.word_rep_layer].output();
                Self::forward_step(
                    &mut self.high_model.layers,
                    &mut self.high_cache[n_updates],
                    h_rep,
                );
                let high_out = self.high_cache[n_updates].last().unwrap().output();
                self.high_context.copy_from_slice(high_out);
                n_updates += 1;
            }
            // COPY: high_context unchanged

            self.z_prev = z;
        }

        self.last_boundary_rate = n_updates as f32 / seq_len as f32;
    }

    // ── Loss (reads char_cache directly) ─────────────────────────────────────

    pub fn seq_loss(&self, targets: &[u16]) -> f32 {
        let last = self.char_model.layers.len() - 1;
        let mut loss = 0.0f32;
        for t in 0..targets.len() {
            let p = self.char_cache[t][last].output()[targets[t] as usize] + 1e-12;
            loss -= p.ln();
        }
        loss / targets.len() as f32
    }

    // ── Backward ─────────────────────────────────────────────────────────────

    pub fn backwards_sequence(&mut self, _inputs: &[u16], targets: &[u16]) {
        let seq_len = self.seq_len;
        if seq_len == 0 {
            return;
        }

        // Destructure to satisfy borrow checker
        let HmRnn {
            char_model,
            high_model,
            char_cache,
            high_cache,
            is_update,
            z_soft,
            high_cache_idx,
            high_h_prev,
            char_delta_buf,
            high_delta_buf,
            ste_extra,
            vocab_size,
            context_size,
            char_rep_size,
            word_rep_layer,
            z_w,
            z_w_grad,
            z_b_grad,
            slope,
            ..
        } = self;

        let n_char = char_model.layers.len();
        let n_high = high_model.layers.len();
        let char_layers = &mut char_model.layers;
        let high_layers = &mut high_model.layers;

        // dh_future: accumulated gradient w.r.t. the current high context h^2.
        // Chars between two UPDATE steps all use the same h^2, so their
        // d_context values are summed here and flushed when the UPDATE is reached
        // in the backward pass.
        let mut dh_future = vec![0.0f32; *context_size];
        let mut high_grad_norm_sum = 0.0f32;
        let mut n_updates_seen = 0usize;

        for t in (0..seq_len).rev() {
            // ── Char FLUSH: zero BPTT before this boundary char's backward ────
            // z_t = 1 means the char model will FLUSH at t+1.  In backward we
            // are at the last char of a segment; the accumulated dh_bptt from
            // the NEXT segment (t+1 .. next_boundary) must not leak back.
            if is_update[t] {
                for layer in char_layers.iter_mut() {
                    layer.zero_bptt_state();
                }
            }

            // ── High UPDATE at t ──────────────────────────────────────────────
            let d_h_t1_from_high: Option<Box<[f32]>>;

            if is_update[t] {
                let j = high_cache_idx[t];

                // STE: compute d_z before modifying dh_future
                let h_t2 = high_cache[j].last().unwrap().output();
                let h_prev2 = &high_h_prev[j];
                let d_z_ste: f32 = dh_future
                    .iter()
                    .zip(h_t2.iter())
                    .zip(h_prev2.iter())
                    .map(|((dh, h), hp)| dh * (h - hp))
                    .sum();
                let zs = z_soft[t];
                let d_logit = d_z_ste * *slope * zs * (1.0 - zs);

                // Accumulate boundary detector weight gradients
                let h_rep = char_cache[t][*word_rep_layer].output();
                for (wg, h) in z_w_grad.iter_mut().zip(h_rep.iter()) {
                    *wg += d_logit * h;
                }
                *z_b_grad += d_logit;

                // Extra gradient flowing into h_t^rep from boundary detector (STE)
                for (e, w) in ste_extra.iter_mut().zip(z_w.iter()) {
                    *e = d_logit * w;
                }

                // Run high backward with dh_future as initial delta
                high_delta_buf[..*context_size].copy_from_slice(&dh_future);
                let mut high_delta_len = *context_size;

                for hl in (0..n_high).rev() {
                    high_layers[hl].backward(
                        &mut high_delta_buf[..high_delta_len],
                        high_cache[j][hl].as_mut(),
                    );
                    if hl == 0 {
                        break;
                    }
                    let dx = high_cache[j][hl].input_grad();
                    let new_len = dx.len();
                    high_delta_buf[..new_len].copy_from_slice(dx);
                    if let Some(bptt) = high_layers[hl - 1].bptt_hidden_grad() {
                        add_vec_in_place(&mut high_delta_buf[..new_len], bptt);
                    }
                    high_delta_len = new_len;
                }

                // Gradient for h_t^1 from high model (= input_grad of high layer 0)
                let grad_from_high = high_cache[j][0].input_grad();
                let boxed: Box<[f32]> = grad_from_high.into();
                d_h_t1_from_high = Some(boxed);

                // Stats
                let norm: f32 = dh_future.iter().map(|x| x * x).sum::<f32>().sqrt();
                high_grad_norm_sum += norm;
                n_updates_seen += 1;

                // Reset dh_future: start accumulating for the segment BEFORE t
                dh_future.fill(0.0);
            } else {
                d_h_t1_from_high = None;
                ste_extra.fill(0.0);
            }

            // ── Char backward at t ────────────────────────────────────────────

            // Initial delta: cross-entropy gradient (softmax output − one-hot)
            let out = char_cache[t].last().unwrap().output();
            char_delta_buf[..out.len()].copy_from_slice(out);
            char_delta_buf[targets[t] as usize] -= 1.0;
            let mut delta_len = out.len();

            for l in (0..n_char).rev() {
                // Inject high + STE gradient at word_rep_layer
                if l == *word_rep_layer && is_update[t] {
                    debug_assert_eq!(delta_len, *char_rep_size);
                    if let Some(ref dh) = d_h_t1_from_high {
                        add_vec_in_place(&mut char_delta_buf[..delta_len], dh);
                    }
                    add_vec_in_place(&mut char_delta_buf[..delta_len], ste_extra);
                }

                char_layers[l]
                    .backward(&mut char_delta_buf[..delta_len], char_cache[t][l].as_mut());

                if l == 0 {
                    break;
                }

                let dx = char_cache[t][l].input_grad();
                let new_len = dx.len();
                char_delta_buf[..new_len].copy_from_slice(dx);
                if let Some(bptt) = char_layers[l - 1].bptt_hidden_grad() {
                    add_vec_in_place(&mut char_delta_buf[..new_len], bptt);
                }
                delta_len = new_len;
            }

            // ── Accumulate context gradient (d_h^2 from this char's input) ───
            // char_cache[t][0].input_grad() = gradient w.r.t. [one_hot | context]
            // The context part is [vocab..] which is d_h_{context_at_t}^2.
            let full_input_grad = char_cache[t][0].input_grad();
            let d_context = &full_input_grad[*vocab_size..];
            add_vec_in_place(&mut dh_future, d_context);
        }

        // ── Stats and finalize ────────────────────────────────────────────────

        self.last_high_grad_signal = if n_updates_seen > 0 {
            high_grad_norm_sum / n_updates_seen as f32
        } else {
            0.0
        };

        for layer in &mut self.char_model.layers {
            layer.accumulate_init_grad();
        }
        for layer in &mut self.high_model.layers {
            layer.accumulate_init_grad();
        }
    }

    // ── Training loop ─────────────────────────────────────────────────────────

    pub fn train<'a, I: Iterator<Item = (&'a [u16], &'a [u16])>>(
        &mut self,
        data: I,
        lr: f32,
        iteration: &mut usize,
        j: &mut usize,
        batch_size: usize,
    ) {
        let mut total_loss = 0.0f32;
        let mut steps = 0usize;

        for (inputs, targets) in data {
            self.reset();
            self.forward_over(inputs);
            total_loss += self.seq_loss(targets);
            steps += 1;

            self.backwards_sequence(inputs, targets);

            *iteration += 1;
            if *iteration % batch_size == 0 {
                for layer in &mut self.char_model.layers {
                    layer.apply_grads(lr);
                }
                self.char_model.clear_grads();

                for layer in &mut self.high_model.layers {
                    layer.apply_grads(lr);
                }
                self.high_model.clear_grads();

                // Boundary detector
                for (w, g) in self.z_w.iter_mut().zip(self.z_w_grad.iter()) {
                    *w -= lr * g;
                }
                self.z_b -= lr * self.z_b_grad;
                self.z_w_grad.fill(0.0);
                self.z_b_grad = 0.0;

                *iteration = 0;
                *j += 1;
            }
        }

        let avg = total_loss / steps.max(1) as f32;
        println!(
            "{j} | loss={:.4} | pp={:.1} | z={:.1}% | h∇={:.4} | slope={:.1}",
            avg,
            avg.exp(),
            self.last_boundary_rate * 100.0,
            self.last_high_grad_signal,
            self.slope,
        );
    }

    // ── Sampling ──────────────────────────────────────────────────────────────

    pub fn sample(
        &mut self,
        prefix: &[u16],
        max_len: usize,
        temperature: f32,
        mut callback: impl FnMut(u16) -> bool,
    ) -> Vec<u16> {
        self.reset();

        let mut last_token: u16 = if prefix.is_empty() {
            rand::random_range(0..self.vocab_size) as u16
        } else {
            if prefix.len() > 1 {
                self.sample_prefix(&prefix[..prefix.len() - 1]);
            }
            prefix[prefix.len() - 1]
        };

        let mut out = Vec::with_capacity(max_len);

        for _ in 0..max_len {
            // FLUSH if previous token triggered a boundary
            if self.z_prev {
                for layer in &mut self.char_model.layers {
                    layer.reset_state();
                }
            }

            self.fill_input_buf(last_token);
            Self::forward_sample_step(
                &mut self.char_model.layers,
                &mut self.char_cache[0],
                &self.full_input_buf,
            );

            // Boundary detection
            let h_rep = self.char_cache[0][self.word_rep_layer].output();
            let (z, _) = self.boundary_forward(h_rep);
            self.is_update[0] = z;

            // Sample next token
            let logits = self.char_cache[0].last().unwrap().output();
            let t = temperature.max(1e-8);
            let scaled: Vec<f32> = logits.iter().map(|&v| v / t).collect();
            let probs = softmax(&scaled);
            let next = self.sample_from_probs(&probs);
            out.push(next);
            if !callback(next) {
                break;
            }

            // High UPDATE if boundary detected
            if z {
                let h_rep = self.char_cache[0][self.word_rep_layer].output();
                Self::forward_sample_step(
                    &mut self.high_model.layers,
                    &mut self.high_cache[0],
                    h_rep,
                );
                let high_out = self.high_cache[0].last().unwrap().output();
                self.high_context.copy_from_slice(high_out);
            }

            self.z_prev = z;
            last_token = next;
        }

        out
    }

    fn sample_prefix(&mut self, prefix: &[u16]) {
        for &token in prefix {
            if self.z_prev {
                for layer in &mut self.char_model.layers {
                    layer.reset_state();
                }
            }
            self.fill_input_buf(token);
            Self::forward_sample_step(
                &mut self.char_model.layers,
                &mut self.char_cache[0],
                &self.full_input_buf,
            );
            let h_rep = self.char_cache[0][self.word_rep_layer].output();
            let (z, _) = self.boundary_forward(h_rep);
            if z {
                let h_rep = self.char_cache[0][self.word_rep_layer].output();
                Self::forward_sample_step(
                    &mut self.high_model.layers,
                    &mut self.high_cache[0],
                    h_rep,
                );
                let high_out = self.high_cache[0].last().unwrap().output();
                self.high_context.copy_from_slice(high_out);
            }
            self.z_prev = z;
        }
    }

    fn sample_from_probs(&self, probs: &[f32]) -> u16 {
        let r = rand::random_range(0.0..1.0f32);
        let mut cum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= r {
                return i as u16;
            }
        }
        (probs.len() - 1) as u16
    }

    // ── Persistence ───────────────────────────────────────────────────────────
    //
    // Format:
    //   HM_RNN_MAGIC   u32
    //   vocab_size     u32
    //   context_size   u32
    //   word_rep_layer u32
    //   slope          f32
    //   z_w            f32_slice (char_rep_size)
    //   z_b            f32
    //   char_model     <Sequential blob>
    //   high_model     <Sequential blob>

    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut buf = Cursor::new(Vec::<u8>::new());
        let w = &mut buf as &mut dyn Write;

        write_u32(w, HM_RNN_MAGIC)?;
        write_u32(w, self.vocab_size as u32)?;
        write_u32(w, self.context_size as u32)?;
        write_u32(w, self.word_rep_layer as u32)?;
        write_f32(w, self.slope)?;
        write_f32_slice(w, &self.z_w)?;
        write_f32(w, self.z_b)?;

        self.char_model.write_to(w)?;
        self.high_model.write_to(w)?;

        File::create(path)?.write_all(&buf.into_inner())
    }

    pub fn load(path: &str) -> io::Result<Self> {
        let r = &mut File::open(path)? as &mut dyn Read;

        let magic = crate::loading::read_u32(r)?;
        if magic != HM_RNN_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Expected HM_RNN magic 0x{HM_RNN_MAGIC:08X}, got 0x{magic:08X}"),
            ));
        }

        let vocab_size = read_u32(r)? as usize;
        let _context_size = read_u32(r)? as usize;
        let word_rep_layer = read_u32(r)? as usize;
        let slope = read_f32(r)?;
        let z_w: Box<[f32]> = read_f32_vec(r)?.into_boxed_slice();
        let z_b = read_f32(r)?;

        let char_model = Sequential::load_from(r)?;
        let high_model = Sequential::load_from(r)?;

        let char_rep_size = char_model.layers[word_rep_layer].output_size();
        let context_size = high_model.output_size;
        let char_max = char_model
            .layers
            .iter()
            .map(|l| l.output_size())
            .max()
            .unwrap_or(0);
        let high_max = high_model
            .layers
            .iter()
            .map(|l| l.output_size())
            .max()
            .unwrap_or(0);

        Ok(Self {
            vocab_size,
            context_size,
            char_rep_size,
            word_rep_layer,
            z_w,
            z_b,
            z_w_grad: vec![0.0; char_rep_size].into_boxed_slice(),
            z_b_grad: 0.0,
            slope,
            high_context: vec![0.0; context_size].into_boxed_slice(),
            z_prev: false,
            char_cache: vec![],
            high_cache: vec![],
            is_update: vec![],
            z_soft: vec![],
            high_cache_idx: vec![],
            high_h_prev: vec![],
            seq_len: 0,
            char_delta_buf: vec![0.0; char_max],
            high_delta_buf: vec![0.0; high_max],
            full_input_buf: vec![0.0; vocab_size + context_size],
            ste_extra: vec![0.0; char_rep_size],
            last_boundary_rate: 0.0,
            last_high_grad_signal: 0.0,
            char_model,
            high_model,
        })
    }
}
