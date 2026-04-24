// hierarchical_sequential.rs
//
// Fixes vs original:
//   1. sample() — forward_step_char used `forward` (applies softmax), then called
//      softmax(x/T) again → double-softmax.  Now uses `forward_sample_step_char`
//      which calls `layer.forward_sample()` so SoftmaxLayer passes raw logits.
//
//   2. backwards_sequence() — BPTT between word boundaries was dropped.
//      `dh_bptt` from the high model's first LSTM was never fed back into the
//      delta for the preceding word.  Now captured and added each iteration.
//
//   3. forward_over() / sample() — one_hot() + full_input Vec were re-allocated
//      every timestep.  Replaced with a pre-allocated scratch buffer.
//
//   4. save() / load() — new: model can now be persisted and reloaded.

use std::{
    fs::File,
    io::{self, Cursor, Read, Write},
    u32,
};

use crate::{
    loading::read_u16,
    loading::read_u32,
    lstm::add_vec_in_place,
    nn_layer::NnLayer,
    saving::{HIER_MAGIC, write_u16, write_u32},
    sequential::Sequential,
    softmax::softmax,
};

pub struct HierarchicalSequential {
    pub char_model: Sequential,
    pub char2_model: Sequential,
    pub high_model: Sequential,

    pub vocab_size: usize,
    pub context_size: usize,

    pub boundary_token_ids: Vec<u16>,

    word_context: Box<[f32]>,
    boundary_timesteps: Vec<usize>,

    char_input_buf: Vec<f32>,

    pub last_high_grad_signal: f32,
}

impl HierarchicalSequential {
    /// `boundary_token_ids` — from `tokenizer.boundary_token_ids()`.
    pub fn new(
        char_model: Sequential,
        char2_model: Sequential,
        high_model: Sequential,
        vocab_size: usize,
        boundary_token_ids: Vec<u16>,
    ) -> Self {
        let context_size = high_model.output_size;
        assert_eq!(
            char_model.input_size, vocab_size,
            "char_model.input_size must equal vocab_size"
        );

        assert_eq!(
            char2_model.output_size, vocab_size,
            "char2_model.output_size must equal vocab_size"
        );

        assert_eq!(char_model.output_size, high_model.input_size);

        assert_eq!(
            char2_model.input_size,
            high_model.output_size + char_model.output_size
        );

        let char_input_buf = vec![0.0; vocab_size];
        let word_context = vec![0.0; char_model.output_size + context_size];

        Self {
            char_model,
            char2_model,
            high_model,
            vocab_size,
            context_size,
            boundary_token_ids,
            word_context: word_context.into(),
            boundary_timesteps: vec![],
            char_input_buf,
            last_high_grad_signal: 0.0,
        }
    }

    // ── Cache allocation ──────────────────────────────────────────────────────

    pub fn make_cache(&mut self, seq_len: usize) {
        self.char_model.make_cache(seq_len);
        self.char2_model.make_cache(seq_len);
        self.high_model.make_cache(seq_len);
    }

    // ── State reset ───────────────────────────────────────────────────────────

    pub fn reset(&mut self) {
        for layer in &mut self.char_model.layers {
            layer.reset_state();
        }
        for layer in &mut self.char2_model.layers {
            layer.reset_state();
        }
        for layer in &mut self.high_model.layers {
            layer.reset_state();
        }
        self.boundary_timesteps.clear();
    }

    // ── Private forward helpers ───────────────────────────────────────────────

    fn forward_step(
        layers: &mut [Box<dyn NnLayer>],
        caches_t: &mut [Box<dyn crate::nn_layer::DynCache>],
        input: &[f32],
    ) {
        for l in 0..layers.len() {
            let (left, right) = caches_t.split_at_mut(l);
            let inp = if l == 0 { input } else { left[l - 1].output() };
            layers[l].forward(inp, right[0].as_mut());
        }
    }

    /// Like `forward_step_char` but calls `forward_sample` so that the
    /// SoftmaxLayer passes through raw logits instead of applying softmax.
    fn forward_sample_step(
        layers: &mut [Box<dyn NnLayer>],
        caches_t: &mut [Box<dyn crate::nn_layer::DynCache>],
        input: &[f32],
    ) {
        for l in 0..layers.len() {
            let (left, right) = caches_t.split_at_mut(l);
            let inp = if l == 0 { input } else { left[l - 1].output() };
            layers[l].forward_sample(inp, right[0].as_mut());
        }
    }

    // ── Build the combined [one_hot | context] input in the scratch buffer ────

    #[inline]
    fn fill_input_buf(&mut self, token: u16) {
        self.char_input_buf[..self.vocab_size].fill(0.0);
        self.char_input_buf[token as usize] = 1.0;
    }

    fn assert_no_nan(&self, slice: &[f32], name: &str, t: usize) {
        if let Some(pos) = slice.iter().position(|x| !x.is_finite()) {
            panic!("NaN/Inf detected in {name} at t={t}, idx={pos}");
        }
    }

    // ── Forward (training) ───────────────────────────────────────────────────

    pub fn forward_over(&mut self, input: &[u16]) {
        self.word_context.fill(0.0);
        let char_output = self.char_model.output_size;
        for t in 0..input.len() {
            let token = input[t];

            self.fill_input_buf(token);

            Self::forward_step(
                &mut self.char_model.layers,
                &mut self.char_model.cache[t],
                &self.char_input_buf,
            );

            let char1_output = self.char_model.cache[t].last().unwrap().output();
            self.assert_no_nan(char1_output, "char_output", t);
            self.word_context[0..char_output].copy_from_slice(char1_output);

            Self::forward_step(
                &mut self.char2_model.layers,
                &mut self.char2_model.cache[t],
                &self.word_context,
            );

            let is_boundary = self.boundary_token_ids.contains(&token);

            if is_boundary {
                self.boundary_timesteps.push(t);
                let word_idx = self.boundary_timesteps.len() - 1;

                let word_rep = self.char_model.cache[t].last().unwrap().output();

                Self::forward_step(
                    &mut self.high_model.layers,
                    &mut self.high_model.cache[word_idx],
                    word_rep,
                );

                for layer in &mut self.char_model.layers {
                    layer.reset_state();
                }
                for layer in &mut self.char2_model.layers {
                    layer.reset_state();
                }

                let high_out = self.high_model.cache[word_idx].last().unwrap().output();
                self.assert_no_nan(high_out, "high_output", word_idx);
                self.word_context[char_output..].copy_from_slice(high_out);
            }
        }
    }

    // ── Backward ─────────────────────────────────────────────────────────────
    //
    // Struktur der Rückwärtspropagation:
    //
    //   Für jedes t rückwärts von len-1 bis 0:
    //     1. Flush — wenn boundary_timesteps[current_word-1] >= t, lasse den
    //        bisher akkumulierten d_context durch high_model[current_word-1]
    //        laufen; das Ergebnis ist dL/d(word_rep) und wird als „pending" grad
    //        für char_model an t = boundary zurückgelegt.
    //     2. char2_model rückwärts mit Cross-Entropy-Start-Delta.
    //        cache[t][0].input_grad() = dL/d([char_out_t, context_t]).
    //        d_char_output → startet char_delta,  d_context → d_context_accum.
    //     3. Falls pending_target_t == t:  char_delta += pending_word_rep_grad.
    //     4. char_model rückwärts mit char_delta.
    //     5. Word-Start (t==0 oder boundary bei t-1): accumulate_init_grad +
    //        reset_backwards auf char_model, weil der Forward bei t-1 ein
    //        reset_state durchgeführt hat — h_{t-1} war h_init.  BPTT darf
    //        nicht über die Wortgrenze hinaus fließen.
    //
    //   Nach der Schleife: accumulate_init_grad für char2_model und high_model
    //   (diese werden innerhalb der Sequenz nie resettet).
    //
    // Cross-word BPTT für high_model passiert automatisch: zwischen Wort W und
    // W-1 überdauert dh_bptt in der Zelle und wird beim nächsten Backward
    // wieder eingefüttert — entweder direkt von Sequential (für nackte sLSTMs)
    // oder block-intern (für SLSTMBlock, siehe slstm_block.rs).

    pub fn backwards_sequence(&mut self, _inputs: &[u16], targets: &[u16]) {
        if targets.is_empty() {
            return;
        }

        let char_output = self.char_model.output_size;
        let context_size = self.context_size;
        let vocab = self.vocab_size;
        let n_boundaries = self.boundary_timesteps.len();

        // Destructure so we can borrow each sub-model independently of the others.
        let HierarchicalSequential {
            char_model,
            char2_model,
            high_model,
            boundary_timesteps,
            ..
        } = self;

        // ── BPTT state defensiv zurücksetzen (letzte Backward-Sequenz hat
        //    dh_bptt / dc_bptt in den Layern hinterlassen, die sonst beim
        //    ersten Backward-Schritt mitlaufen würden). ───────────────────────
        for l in &mut char_model.layers {
            l.zero_bptt_state();
        }
        for l in &mut char2_model.layers {
            l.zero_bptt_state();
        }
        for l in &mut high_model.layers {
            l.zero_bptt_state();
        }

        // ── Zwischenpuffer ────────────────────────────────────────────────────
        let mut d_context_accum = vec![0.0; context_size];
        let mut pending_word_rep_grad = vec![0.0; char_output];
        let mut pending_target_t = None;

        // w_ptr zeigt auf „das nächste noch NICHT geflushte Wort + 1".  Wir
        // starten oberhalb des letzten Wortes und wandern rückwärts, wobei wir
        // pro Boundary einmal high_model-Backward aufrufen.
        let mut w_ptr = n_boundaries;

        let mut total_grad_sq = 0.0;

        for t in (0..targets.len()).rev() {
            // ── A. Boundaries flushen, deren t bereits „hinter uns" liegt ────
            //
            // `boundary_timesteps[w_ptr - 1] >= t` heißt: alle t' > boundary_ts
            // haben wir verarbeitet und d_context_accum ist komplett für dieses
            // Wort.  Jetzt durch high_model backward jagen.
            while w_ptr > 0 && boundary_timesteps[w_ptr - 1] >= t {
                let word_idx = w_ptr - 1;

                // delta_buf für high_model befüllen mit d_context_accum
                high_model.delta_buf[..context_size].copy_from_slice(&d_context_accum);

                backward_through_layers(
                    &mut high_model.layers,
                    &mut high_model.cache[word_idx],
                    &mut high_model.delta_buf,
                    context_size,
                );

                // high_model's erster Layer hat als Input char_model[boundary_t].output,
                // also ist input_grad() jetzt dL/d(word_rep).
                let g = high_model.cache[word_idx][0].input_grad();
                pending_word_rep_grad.copy_from_slice(&g[..char_output]);

                // Für die Grad-Norm-Telemetry
                for &v in &pending_word_rep_grad[..] {
                    total_grad_sq += v * v;
                }

                pending_target_t = Some(boundary_timesteps[word_idx]);
                d_context_accum.fill(0.0);
                w_ptr -= 1;
            }

            // ── B. char2_model rückwärts (Cross-Entropy-Start) ───────────────
            let last_out = char2_model.cache[t].last().unwrap().output();
            char2_model.delta_buf[..vocab].copy_from_slice(last_out);
            char2_model.delta_buf[targets[t] as usize] -= 1.0;

            backward_through_layers(
                &mut char2_model.layers,
                &mut char2_model.cache[t],
                &mut char2_model.delta_buf,
                vocab,
            );

            // ── C. input_grad von char2_model splitten ───────────────────────
            //   char2_model.cache[t][0] ist die erste (nicht-rekurrente) Input-
            //   Projektion — ihr Input war [char_out_t, context_t].
            {
                let grad_in = char2_model.cache[t][0].input_grad();

                // char_model-Delta vorbereiten
                char_model.delta_buf[..char_output].copy_from_slice(&grad_in[..char_output]);

                // d_context akkumulieren (geht an das Wort, dessen Boundary
                // VOR t liegt — flush erfolgt in A bei der nächsten Runde).
                for i in 0..context_size {
                    d_context_accum[i] += grad_in[char_output + i];
                }
            }

            // ── D. Pending word_rep-Grad an Boundary-t einspeisen ────────────
            if pending_target_t == Some(t) {
                for i in 0..char_output {
                    char_model.delta_buf[i] += pending_word_rep_grad[i];
                }
                pending_target_t = None;
            }

            // ── E. char_model rückwärts ──────────────────────────────────────
            backward_through_layers(
                &mut char_model.layers,
                &mut char_model.cache[t],
                &mut char_model.delta_buf,
                char_output,
            );

            // ── F. Word-Start?  (t==0 oder boundary bei t-1) ─────────────────
            //
            //   Nach `reset_state` im Forward war h_{t-1} = h_init, c_{t-1} =
            //   c_init.  Also sind die aktuellen dh_bptt / dc_bptt gerade die
            //   Grads für h_init / c_init.  -> accumulate + reset, sonst
            //   würden sie in den backward bei t-1 (letztes Zeichen des
            //   Vorgängerworts) hineinsickern und dort FALSCH als Cross-Word-
            //   BPTT wirken.
            let is_word_start =
                t == 0 || (w_ptr > 0 && boundary_timesteps[w_ptr.saturating_sub(1)] == t - 1);

            if is_word_start {
                for layer in &mut char_model.layers {
                    layer.accumulate_init_grad();
                    layer.zero_bptt_state();
                }

                for layer in &mut char2_model.layers {
                    layer.accumulate_init_grad();
                    layer.zero_bptt_state();
                }
            }
        }

        for layer in &mut high_model.layers {
            layer.accumulate_init_grad();
        }

        // Kleine Telemetrie für das train()-Logging.
        self.last_high_grad_signal = total_grad_sq.sqrt();
    }
    // ── Training ─────────────────────────────────────────────────────────────

    pub fn train<'a, I: Iterator<Item = (&'a [u16], &'a [u16])>>(
        &mut self,
        data: I,
        lr: f32,
        iteration: &mut usize,
        j: &mut usize,
        batch_size: usize,
    ) {
        let mut total_loss = 0.0;
        let mut steps = 0;

        for (inputs, targets) in data {
            self.reset();
            self.forward_over(inputs);
            total_loss += self.char2_model.seq_loss(targets);
            steps += 1;

            self.backwards_sequence(inputs, targets);

            *iteration += 1;
            if *iteration % batch_size == 0 {
                let char_lr = self.boundary_timesteps.len() as f32 / inputs.len() as f32 * lr;

                for layer in &mut self.char_model.layers {
                    layer.apply_grads(char_lr);
                }
                self.char_model.clear_grads();

                for layer in &mut self.char2_model.layers {
                    layer.apply_grads(char_lr);
                }
                self.char2_model.clear_grads();

                for layer in &mut self.high_model.layers {
                    layer.apply_grads(lr);
                }
                self.high_model.clear_grads();

                *iteration = 0;
                *j += 1;
            }
        }

        println!(
            "{j} | char loss = {:.4} | high ∇ = {:.4} | pp = {:.4}",
            total_loss / steps.max(1) as f32,
            self.last_high_grad_signal,
            (total_loss / steps.max(1) as f32).exp()
        );
    }

    pub fn sample(
        &mut self,
        prefix: &[u16],
        max_len: usize,
        temperature: f32,
        mut callback: impl FnMut(u16) -> bool,
    ) -> Vec<u16> {
        self.reset();

        // Prefix (bis auf das letzte Token) durch das Netz jagen, damit der
        // rekurrente Zustand aufgebaut wird.  Das letzte Prefix-Token wird
        // below als erstes „last_token" fürs Sampling benutzt.
        let mut last_token = if prefix.is_empty() {
            rand::random_range(0..self.vocab_size) as u16
        } else {
            self.forward_sample_prefix(&prefix[..prefix.len() - 1]);
            prefix[prefix.len() - 1]
        };

        let char_output = self.char_model.output_size;
        let mut out = Vec::with_capacity(max_len);

        for _ in 0..max_len {
            // ── 1. char_model forward (ein Schritt, cache[0]) ────────────────
            self.fill_input_buf(last_token);
            Self::forward_sample_step(
                &mut self.char_model.layers,
                &mut self.char_model.cache[0],
                &self.char_input_buf,
            );
            {
                let char_out_slice = self.char_model.cache[0].last().unwrap().output();
                self.word_context[0..char_output].copy_from_slice(char_out_slice);
            }

            // ── 2. char2_model forward (mit [char_out, context]) ─────────────
            Self::forward_sample_step(
                &mut self.char2_model.layers,
                &mut self.char2_model.cache[0],
                &self.word_context,
            );

            // ── 3. Boundary?  → high_model updaten, char_model resetten ──────
            if self.boundary_token_ids.contains(&last_token) {
                // Word-Repräsentation kopieren, damit char_model als &mut
                // verwendet werden kann.
                let mut word_rep = vec![0.0f32; char_output];
                word_rep.copy_from_slice(self.char_model.cache[0].last().unwrap().output());

                Self::forward_sample_step(
                    &mut self.high_model.layers,
                    &mut self.high_model.cache[0],
                    &word_rep,
                );

                for layer in &mut self.char_model.layers {
                    layer.reset_state();
                }

                let high_out_slice = self.high_model.cache[0].last().unwrap().output();
                self.word_context[char_output..].copy_from_slice(high_out_slice);
            }

            // ── 4. Sampling aus den Logits ───────────────────────────────────
            //   forward_sample hat die Softmax übersprungen → letzte Ausgabe
            //   des char2_model enthält Roh-Logits; wir wenden Temperature +
            //   Softmax einmal extern an.
            let probs: Box<[f32]> = {
                let logits = self.char2_model.cache[0].last().unwrap().output();
                let temp = temperature.max(1e-8);
                let scaled: Vec<f32> = logits.iter().map(|&v| v / temp).collect();
                softmax(&scaled)
            };

            let next = self.sample_from_probs(&probs);
            out.push(next);
            if !callback(next) {
                break;
            }
            last_token = next;
        }

        out
    }

    /// Prefix-Feed fürs Sampling — identisch zu `forward_over`, aber
    /// (a) nutzt forward_sample_step (keine Softmax) und
    /// (b) arbeitet auf cache[0] (nur 1 Timestep gespeichert).
    fn forward_sample_prefix(&mut self, prefix: &[u16]) {
        let char_output = self.char_model.output_size;

        for &token in prefix {
            self.fill_input_buf(token);

            Self::forward_sample_step(
                &mut self.char_model.layers,
                &mut self.char_model.cache[0],
                &self.char_input_buf,
            );
            {
                let char_out_slice = self.char_model.cache[0].last().unwrap().output();
                self.word_context[0..char_output].copy_from_slice(char_out_slice);
            }

            Self::forward_sample_step(
                &mut self.char2_model.layers,
                &mut self.char2_model.cache[0],
                &self.word_context,
            );

            if self.boundary_token_ids.contains(&token) {
                let mut word_rep = vec![0.0f32; char_output];
                word_rep.copy_from_slice(self.char_model.cache[0].last().unwrap().output());

                Self::forward_sample_step(
                    &mut self.high_model.layers,
                    &mut self.high_model.cache[0],
                    &word_rep,
                );

                for layer in &mut self.char_model.layers {
                    layer.reset_state();
                }

                let high_out_slice = self.high_model.cache[0].last().unwrap().output();
                self.word_context[char_output..].copy_from_slice(high_out_slice);
            }
        }
    }

    fn sample_from_probs(&self, probs: &[f32]) -> u16 {
        let r = rand::random_range(0.0..1.0);
        let mut cum = 0.0;
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
    //   HIER_MAGIC    u32
    //   vocab_size    u32
    //   context_size  u32
    //   n_boundaries  u32
    //   boundary_ids  [u16 × n_boundaries]
    //   char_model    <Sequential blob>
    //   high_model    <Sequential blob>

    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut buf = Cursor::new(Vec::<u8>::new());
        let w = &mut buf as &mut dyn Write;

        // Header
        crate::saving::write_u32(w, HIER_MAGIC)?;
        write_u32(w, self.vocab_size as u32)?;
        write_u32(w, self.context_size as u32)?;
        write_u32(w, self.boundary_token_ids.len() as u32)?;
        for &id in &self.boundary_token_ids {
            write_u16(w, id)?;
        }

        // Sub-models
        self.char_model.write_to(w)?;
        self.char2_model.write_to(w)?;
        self.high_model.write_to(w)?;

        File::create(path)?.write_all(&buf.into_inner())
    }

    pub fn load(path: &str) -> io::Result<Self> {
        let r = &mut File::open(path)? as &mut dyn Read;

        let magic = read_u32(r)?;
        if magic != HIER_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Expected HIER magic 0x{HIER_MAGIC:08X}, got 0x{magic:08X}"),
            ));
        }

        let vocab_size = read_u32(r)? as usize;
        let context_size = read_u32(r)? as usize;
        let n_boundaries = read_u32(r)? as usize;
        let mut boundary_token_ids = Vec::with_capacity(n_boundaries);
        for _ in 0..n_boundaries {
            boundary_token_ids.push(read_u16(r)?);
        }

        let char_model = Sequential::load_from(r)?;
        let char2_model = Sequential::load_from(r)?;
        let high_model = Sequential::load_from(r)?;

        let char_input_buf = vec![0.0; vocab_size];
        let word_context = vec![0.0; char_model.output_size + context_size];

        Ok(Self {
            vocab_size,
            context_size,
            boundary_token_ids,
            char_model,
            char2_model,
            high_model,
            word_context: word_context.into_boxed_slice(),
            boundary_timesteps: vec![],
            char_input_buf,
            last_high_grad_signal: 0.0,
        })
    }
}

// ── Freie Helper ─────────────────────────────────────────────────────────────

/// Ein Layer-Stack rückwärts durchlaufen — exakt dieselbe Mechanik wie in
/// `Sequential::backwards_sequence`, nur ohne den Cross-Entropy-Start-Delta.
/// Der Aufrufer legt das Start-Delta in `delta_buf[..delta_len]` ab; die
/// Funktion schreibt dann `input_grad()` in jeden Cache und — für rekurrente
/// Layer — `dh_bptt` in den Layer selbst.
fn backward_through_layers(
    layers: &mut [Box<dyn NnLayer>],
    caches_t: &mut [Box<dyn crate::nn_layer::DynCache>],
    delta_buf: &mut [f32],
    mut delta_len: usize,
) {
    let n = layers.len();
    for l in (0..n).rev() {
        layers[l].backward(&mut delta_buf[..delta_len], caches_t[l].as_mut());

        if l == 0 {
            break;
        }

        let dx = caches_t[l].input_grad();
        let new_len = dx.len();
        delta_buf[..new_len].copy_from_slice(dx);

        if let Some(bptt) = layers[l - 1].bptt_hidden_grad() {
            add_vec_in_place(&mut delta_buf[..new_len], bptt);
        }

        delta_len = new_len;
    }
}
