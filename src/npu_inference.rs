// NPU inference for the flat Sequential model via OpenVINO.
//
// Requirements:
//   sudo pacman -S openvino          (or Intel-Repo / AUR)
//   intel-npu-acceleration-library   (NPU driver, AUR: intel-npu-driver)
//
// Flow: NpuSampler::new loads the ONNX file, compiles it for "NPU" and
// keeps a single InferRequest ready. step() runs one token step
// and returns new states. sample() wraps the full sampling
// loop (temperature + top-p, identical to Sequential::sample).

use std::io::{Write, stdin, stdout};

use openvino::{Core, ElementType, Shape, Tensor};
use rand::random_range;

use crate::{
    config::{C_SIZE, CHARSET, MAX_LEN, N_SIZE, NUM_HEADS, SEQ_LOC, TEMPERATURE, TOP_P},
    nn::softmax::softmax,
    onnx_export::export_flat_model,
    sequential::Sequential,
    tokenizer::Tokenizer,
};

const ONNX_PATH: &str = "model.onnx";

type BoxErr = Box<dyn std::error::Error + Send + Sync>;

fn f32s_to_raw(dst: &mut [u8], src: &[f32]) {
    for (chunk, &v) in dst.chunks_exact_mut(4).zip(src.iter()) {
        chunk.copy_from_slice(&v.to_le_bytes());
    }
}

fn raw_to_f32s(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// ─── State vector ────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct NpuState {
    pub c: Vec<f32>, // H·dhv·dqk
    pub n: Vec<f32>, // H·dqk
    pub m: Vec<f32>, // H  (stabiliser log)
}

impl NpuState {
    pub fn zero() -> Self {
        Self {
            c: vec![0.0; C_SIZE],
            n: vec![0.0; N_SIZE],
            m: vec![0.0; NUM_HEADS],
        }
    }
}

// ─── NPU sampler ─────────────────────────────────────────────────────────────

pub struct NpuSampler {
    request: openvino::InferRequest,
    vocab: usize,
}

impl NpuSampler {
    /// Loads `onnx_path` via OpenVINO and compiles for the NPU.
    /// `device` is typically `"NPU"`, `"CPU"` for testing or
    /// `"AUTO:NPU,CPU"` for automatic fallback.
    pub fn new(onnx_path: &str, vocab: usize, device: &str) -> Result<Self, BoxErr> {
        let mut core = Core::new()?;
        let onnx_bytes = std::fs::read(onnx_path)?;
        let model = core.read_model_from_buffer(&onnx_bytes, None)?;
        let mut compiled = core.compile_model(&model, device.into())?;
        let request = compiled.create_infer_request()?;
        Ok(Self { request, vocab })
    }

    /// Runs a single token step and returns (logits, new_state).
    pub fn step(&mut self, token: i64, state: &NpuState) -> Result<(Vec<f32>, NpuState), BoxErr> {
        // get_data/get_data_mut panic on alignment mismatch → always use get_raw_data*
        let mut tok = Tensor::new(ElementType::I64, &Shape::new(&[1])?)?;
        tok.get_raw_data_mut()?
            .copy_from_slice(&token.to_le_bytes());

        let mut c_t = Tensor::new(ElementType::F32, &Shape::new(&[C_SIZE as i64])?)?;
        f32s_to_raw(c_t.get_raw_data_mut()?, &state.c);

        let mut n_t = Tensor::new(ElementType::F32, &Shape::new(&[N_SIZE as i64])?)?;
        f32s_to_raw(n_t.get_raw_data_mut()?, &state.n);

        let mut m_t = Tensor::new(ElementType::F32, &Shape::new(&[NUM_HEADS as i64])?)?;
        f32s_to_raw(m_t.get_raw_data_mut()?, &state.m);

        self.request.set_tensor("token_id", &tok)?;
        self.request.set_tensor("c_state", &c_t)?;
        self.request.set_tensor("n_state", &n_t)?;
        self.request.set_tensor("m_state", &m_t)?;

        self.request.infer()?;

        let logits = raw_to_f32s(self.request.get_tensor("logits")?.get_raw_data()?);
        let new_state = NpuState {
            c: raw_to_f32s(self.request.get_tensor("c_new")?.get_raw_data()?),
            n: raw_to_f32s(self.request.get_tensor("n_new")?.get_raw_data()?),
            m: raw_to_f32s(self.request.get_tensor("m_new")?.get_raw_data()?),
        };

        println!("{:?}", &new_state);

        Ok((logits, new_state))
    }

    /// Full sampling loop (identical to Sequential::sample).
    /// Returns the generated tokens; stops when `callback` returns false.
    pub fn sample(
        &mut self,
        prefix: &[u16],
        max_len: usize,
        temperature: f32,
        top_p: f32,
        mut callback: impl FnMut(u16) -> bool,
    ) -> Result<Vec<u16>, BoxErr> {
        let mut state = NpuState::zero();

        let start_token = if prefix.is_empty() {
            random_range(0..self.vocab) as u16
        } else {
            for &tok in &prefix[..prefix.len().saturating_sub(1)] {
                let (_, s) = self.step(tok as i64, &state)?;
                state = s;
            }
            prefix[prefix.len() - 1]
        };

        let mut last = start_token;
        let mut out = Vec::with_capacity(max_len);

        for _ in 0..max_len {
            let (logits, new_state) = self.step(last as i64, &state)?;
            state = new_state;

            // Temperature scaling + softmax
            let scaled: Vec<f32> = logits.iter().map(|&v| v / temperature.max(1e-8)).collect();
            println!(
                "Logits (scaled): {:?}",
                scaled.iter().take(10).collect::<Vec<_>>()
            );
            let probs = softmax(&scaled);

            // Top-p sampling
            let mut idx: Vec<usize> = (0..probs.len()).collect();
            idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

            let mut cum = 0.0;
            let candidates: Vec<usize> = idx
                .iter()
                .copied()
                .take_while(|&i| {
                    if cum >= top_p {
                        return false;
                    }
                    cum += probs[i];
                    true
                })
                .collect();

            let total: f32 = candidates.iter().map(|&i| probs[i]).sum();
            let r = random_range(0.0..total);
            let mut cum = 0.0;
            let mut next = candidates[0] as u16;
            for &i in &candidates {
                cum += probs[i];
                if cum >= r {
                    next = i as u16;
                    break;
                }
            }

            out.push(next);
            if !callback(next) {
                break;
            }
            last = next;
        }

        Ok(out)
    }
}

// ─── Entry point ─────────────────────────────────────────────────────────────

/// Interactive NPU sampling mode.
/// Automatically exports the model to ONNX if not already present,
/// then runs the loop like `sample_normal()` but on the NPU.
pub fn sample_npu() {
    let tokenizer = Tokenizer::new(CHARSET, false);
    let vocab = tokenizer.vocab_size();

    // Generate ONNX if needed
    if !std::path::Path::new(ONNX_PATH).exists() {
        println!("Exportiere Modell nach {ONNX_PATH} ...");
        let model = match Sequential::load(SEQ_LOC) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Laden von '{SEQ_LOC}' fehlgeschlagen: {e}");
                std::process::exit(1);
            }
        };
        if let Err(e) = export_flat_model(&model, ONNX_PATH) {
            eprintln!("ONNX-Export fehlgeschlagen: {e}");
            std::process::exit(1);
        }
        println!("Fertig.");
    }

    // NPU sampler — tries NPU first, falls back to CPU
    let device = std::env::var("OV_DEVICE").unwrap_or_else(|_| "NPU".into());
    let mut sampler = match NpuSampler::new(ONNX_PATH, vocab, &device) {
        Ok(s) => {
            println!("OpenVINO-Device: {device}");
            s
        }
        Err(e) => {
            eprintln!("NPU-Init fehlgeschlagen ({device}): {e}");
            eprintln!(
                "Prüfe:\n  • OpenVINO ≥ 2024.3 installiert?\n  \
                 • NPU-Treiber geladen (intel-npu-driver)?\n  \
                 • OV_DEVICE=CPU setzen zum Testen ohne NPU."
            );
            std::process::exit(1);
        }
    };

    loop {
        println!("\nNPU-Sample — Präfix eingeben (leer = Zufall, Ctrl+D = Ende):");
        let mut input = String::new();
        if stdin().read_line(&mut input).unwrap() == 0 {
            println!();
            return;
        }

        let prefix: Vec<u16> = if !input.trim().is_empty() {
            tokenizer.to_tokens(input.trim())
        } else {
            Vec::new()
        };

        print!(">>> ");
        stdout().flush().unwrap();

        let result = sampler.sample(&prefix, MAX_LEN, TEMPERATURE, TOP_P, |token| {
            let s = tokenizer.get_char(token);
            if s == "<END>" {
                return false;
            }
            print!("{s}");
            stdout().flush().unwrap();
            true
        });

        if let Err(e) = result {
            eprintln!("\nNPU-Inferenz-Fehler: {e}");
        }

        println!();
    }
}
