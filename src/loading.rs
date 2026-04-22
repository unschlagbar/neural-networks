// ── loading.rs ────────────────────────────────────────────────────────────────

use std::{
    fs::File,
    io::{self, Read},
};

use iron_oxide::collections::Matrix;

use crate::{
    activations::{LeakyRelu, Linear, Relu, Sigmoid, Tanh},
    dense::DenseLayer,
    dropout::DropoutLayer,
    linear::LinearLayer,
    lstm::LSTMLayer,
    nn_layer::NnLayer,
    parallel::ParallelLayer,
    projection::Projection,
    rms_norm::{RMSNorm, RMSNormResidual},
    saving::{MAGIC, VERSION},
    sequential::Sequential,
    silu_dense::SiluDenseLayer,
    slstm::SLSTMLayer,
    slstm_block::SLSTMBlock,
    softmax::SoftmaxLayer,
};

// Bring in IndRNN only if the module exists in this crate.

//use crate::indrnn::IndRNNLayer;

// ── Primitive readers ─────────────────────────────────────────────────────────

pub fn read_u8(r: &mut dyn Read) -> io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}
pub fn read_u16(r: &mut dyn Read) -> io::Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}
pub fn read_u32(r: &mut dyn Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
pub fn read_f32(r: &mut dyn Read) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_bits(u32::from_le_bytes(b)))
}
pub fn read_f32_vec(r: &mut dyn Read) -> io::Result<Vec<f32>> {
    let len = read_u32(r)? as usize;
    (0..len).map(|_| read_f32(r)).collect()
}
pub fn read_matrix(r: &mut dyn Read) -> io::Result<Matrix> {
    let rows = read_u32(r)? as usize;
    let cols = read_u32(r)? as usize;
    let flat = read_f32_vec(r)?;
    Ok(Matrix::from_vec(flat, rows, cols))
}

// ── Layer load helpers ────────────────────────────────────────────────────────

pub struct LoadCtx {
    pub input_size: usize,
    pub output_size: usize,
}

pub fn load_dense(r: &mut dyn Read) -> io::Result<Box<dyn NnLayer>> {
    let input = read_u32(r)? as usize;
    let output = read_u32(r)? as usize;
    let act_id = read_u8(r)?;
    let weights = read_matrix(r)?;
    let biases: Box<[f32]> = read_f32_vec(r)?.into_boxed_slice();

    let layer: Box<dyn NnLayer> = match act_id {
        0 => Box::new(DenseLayer::from_loaded(
            input, output, Linear, weights, biases,
        )),
        1 => Box::new(DenseLayer::from_loaded(
            input, output, Relu, weights, biases,
        )),
        2 => Box::new(DenseLayer::from_loaded(
            input, output, Tanh, weights, biases,
        )),
        3 => Box::new(DenseLayer::from_loaded(
            input, output, Sigmoid, weights, biases,
        )),
        4 => Box::new(DenseLayer::from_loaded(
            input, output, LeakyRelu, weights, biases,
        )),
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Unknown activation for Dense",
            ));
        }
    };
    Ok(layer)
}

pub fn load_linear(r: &mut dyn Read) -> io::Result<Box<dyn NnLayer>> {
    let input = read_u32(r)? as usize;
    let output = read_u32(r)? as usize;
    let weights = read_matrix(r)?;
    let biases: Box<[f32]> = read_f32_vec(r)?.into_boxed_slice();

    let layer: Box<dyn NnLayer> =
        Box::new(LinearLayer::from_loaded(input, output, weights, biases));
    Ok(layer)
}

pub fn load_projection(r: &mut dyn Read, ctx: LoadCtx) -> io::Result<Box<dyn NnLayer>> {
    let act_id = read_u8(r)?;
    let weights = read_matrix(r)?;
    let input = ctx.input_size;
    let output = ctx.output_size;

    let layer: Box<dyn NnLayer> = match act_id {
        0 => Box::new(Projection::from_loaded(input, output, Linear, weights)),
        1 => Box::new(Projection::from_loaded(input, output, Relu, weights)),
        2 => Box::new(Projection::from_loaded(input, output, Tanh, weights)),
        3 => Box::new(Projection::from_loaded(input, output, Sigmoid, weights)),
        4 => Box::new(Projection::from_loaded(input, output, LeakyRelu, weights)),
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Unknown activation for Projection",
            ));
        }
    };
    Ok(layer)
}

pub fn load_softmax(r: &mut dyn Read) -> io::Result<Box<dyn NnLayer>> {
    let size = read_u32(r)? as usize;
    Ok(Box::new(SoftmaxLayer::new(size)))
}

pub fn load_dropout(r: &mut dyn Read, ctx: LoadCtx) -> io::Result<Box<dyn NnLayer>> {
    let rate = read_f32(r)?;
    Ok(Box::new(DropoutLayer::new(ctx.input_size, rate)))
}

pub fn load_indrnn(r: &mut dyn Read, ctx: LoadCtx) -> io::Result<Box<dyn NnLayer>> {
    let input = ctx.input_size;
    let hidden = ctx.output_size;
    let act_id = read_u8(r)?;
    let w: Matrix = read_matrix(r)?;
    let u: Box<[f32]> = read_f32_vec(r)?.into_boxed_slice();
    let b: Box<[f32]> = read_f32_vec(r)?.into_boxed_slice();
    let h_init: Box<[f32]> = read_f32_vec(r)?.into_boxed_slice();

    use crate::indrnn::IndRNNLayer;
    let layer: Box<dyn NnLayer> = match act_id {
        0 => Box::new(IndRNNLayer::from_loaded(
            input, hidden, Linear, w, u, b, h_init,
        )),
        1 => Box::new(IndRNNLayer::from_loaded(
            input, hidden, Relu, w, u, b, h_init,
        )),
        2 => Box::new(IndRNNLayer::from_loaded(
            input, hidden, Tanh, w, u, b, h_init,
        )),
        3 => Box::new(IndRNNLayer::from_loaded(
            input, hidden, Sigmoid, w, u, b, h_init,
        )),
        4 => Box::new(IndRNNLayer::from_loaded(
            input, hidden, LeakyRelu, w, u, b, h_init,
        )),
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Unknown activation for IndRNN",
            ));
        }
    };
    Ok(layer)
}

pub fn load_lstm(r: &mut dyn Read, ctx: LoadCtx) -> io::Result<Box<dyn NnLayer>> {
    let wf = read_matrix(r)?;
    let wi = read_matrix(r)?;
    let wc = read_matrix(r)?;
    let wo = read_matrix(r)?;
    let b = read_matrix(r)?;
    let h_init = read_f32_vec(r)?.into_boxed_slice();
    let c_init = read_f32_vec(r)?.into_boxed_slice();
    Ok(Box::new(LSTMLayer::from_loaded(
        ctx.input_size,
        ctx.output_size,
        wf,
        wi,
        wc,
        wo,
        b,
        h_init,
        c_init,
    )))
}

pub fn load_parallel(r: &mut dyn Read, _ctx: LoadCtx) -> io::Result<Box<dyn NnLayer>> {
    let tag1 = read_u8(r)?;
    let input_size1 = read_u32(r)? as usize;
    let out_size1 = read_u32(r)? as usize;
    let lr1 = read_f32(r)?;
    let branch1 = new_layer(
        r,
        tag1,
        LoadCtx {
            input_size: input_size1,
            output_size: out_size1,
        },
    )?;

    let tag2 = read_u8(r)?;
    let input_size2 = read_u32(r)? as usize;
    let out_size2 = read_u32(r)? as usize;
    let lr2 = read_f32(r)?;
    let branch2 = new_layer(
        r,
        tag2,
        LoadCtx {
            input_size: input_size2,
            output_size: out_size2,
        },
    )?;

    Ok(Box::new(ParallelLayer::new(branch1, branch2, lr1, lr2)))
}

pub fn load_res_norm(r: &mut dyn Read, _ctx: LoadCtx) -> io::Result<Box<dyn NnLayer>> {
    let gamma = read_f32_vec(r)?.into_boxed_slice();

    let inner_tag = read_u8(r)?;
    let inner_input = read_u32(r)? as usize;
    let inner_output = read_u32(r)? as usize;

    let ctx = LoadCtx {
        input_size: inner_input,
        output_size: inner_output,
    };
    let inner = new_layer(r, inner_tag, ctx)?;

    let mut wrapper = RMSNormResidual::new(inner);
    wrapper.gamma = gamma;
    Ok(Box::new(wrapper))
}

pub fn load_norm(r: &mut dyn Read, ctx: LoadCtx) -> io::Result<Box<dyn NnLayer>> {
    let gamma = read_f32_vec(r)?.into_boxed_slice();
    let mut wrapper = RMSNorm::new(ctx.input_size);
    wrapper.gamma = gamma;
    Ok(Box::new(wrapper))
}

pub fn load_slstm(r: &mut dyn Read, ctx: LoadCtx) -> io::Result<Box<dyn NnLayer>> {
    let wz = read_matrix(r)?;
    let wi = read_matrix(r)?;
    let wf = read_matrix(r)?;
    let wo = read_matrix(r)?;
    let b = read_matrix(r)?;
    let h_init = read_f32_vec(r)?.into_boxed_slice();
    let c_init = read_f32_vec(r)?.into_boxed_slice();
    Ok(Box::new(SLSTMLayer::from_loaded(
        ctx.input_size,
        ctx.output_size,
        wz,
        wi,
        wf,
        wo,
        b,
        h_init,
        c_init,
    )))
}

pub fn load_silu_dense(r: &mut dyn Read, ctx: LoadCtx) -> io::Result<Box<dyn NnLayer>> {
    let input = ctx.input_size;
    let output = ctx.output_size;
    let weights = read_matrix(r)?;
    let biases: Box<[f32]> = read_f32_vec(r)?.into_boxed_slice();
    Ok(Box::new(SiluDenseLayer::from_loaded(
        input, output, weights, biases,
    )))
}

/// xLSTM-style sLSTM-Block. Layout muss 1:1 zu `SLSTMBlock::save` passen.
///
/// Sequential-Header liefert uns hidden_size (= ctx.input_size == ctx.output_size).
/// Danach in dieser Reihenfolge:
///   - up_size: u32
///   - pre_gamma, post_gamma: f32_vec(H)
///   - cell-Matrizen: wz, wi, wf, wo, b, h_init, c_init  (wie `SLSTMLayer::save`)
///   - SwiGLU: w_gate, b_gate, w_value, b_value, w_down, b_down
pub fn load_slstm_block(r: &mut dyn Read, ctx: LoadCtx) -> io::Result<Box<dyn NnLayer>> {
    if ctx.input_size != ctx.output_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "SLSTMBlock erwartet input_size == output_size, bekam {} != {}",
                ctx.input_size, ctx.output_size,
            ),
        ));
    }
    let hidden_size = ctx.input_size;
    let up_size = read_u32(r)? as usize;

    let pre_gamma: Box<[f32]> = read_f32_vec(r)?.into_boxed_slice();
    let post_gamma: Box<[f32]> = read_f32_vec(r)?.into_boxed_slice();

    // Zell-Gewichte — identische Reihenfolge wie in load_slstm.
    let wz = read_matrix(r)?;
    let wi = read_matrix(r)?;
    let wf = read_matrix(r)?;
    let wo = read_matrix(r)?;
    let b = read_matrix(r)?;
    let h_init: Box<[f32]> = read_f32_vec(r)?.into_boxed_slice();
    let c_init: Box<[f32]> = read_f32_vec(r)?.into_boxed_slice();
    let cell = SLSTMLayer::from_loaded(hidden_size, hidden_size, wz, wi, wf, wo, b, h_init, c_init);

    // SwiGLU.
    let w_gate = read_matrix(r)?;
    let b_gate: Box<[f32]> = read_f32_vec(r)?.into_boxed_slice();
    let w_value = read_matrix(r)?;
    let b_value: Box<[f32]> = read_f32_vec(r)?.into_boxed_slice();
    let w_down = read_matrix(r)?;
    let b_down: Box<[f32]> = read_f32_vec(r)?.into_boxed_slice();

    Ok(Box::new(SLSTMBlock::from_loaded(
        hidden_size,
        up_size,
        pre_gamma,
        post_gamma,
        cell,
        w_gate,
        b_gate,
        w_value,
        b_value,
        w_down,
        b_down,
    )))
}

// ── Layer factory ─────────────────────────────────────────────────────────────

fn new_layer(r: &mut dyn Read, tag: u8, ctx: LoadCtx) -> io::Result<Box<dyn NnLayer>> {
    match tag {
        0 => load_lstm(r, ctx),
        1 => load_dense(r),
        2 => load_indrnn(r, ctx),
        3 => load_projection(r, ctx),
        4 => load_softmax(r),
        5 => load_slstm(r, ctx),
        6 => load_dropout(r, ctx),
        7 => load_parallel(r, ctx),
        8 => load_res_norm(r, ctx),
        9 => load_norm(r, ctx),
        10 => load_silu_dense(r, ctx),
        11 => load_slstm_block(r, ctx),
        12 => load_linear(r),

        o => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unknown layer tag {o}"),
        )),
    }
}

// ── Sequential::load ──────────────────────────────────────────────────────────

impl Sequential {
    /// Load from any `Read` source.
    /// Used by both `load()` and `HierarchicalSequential::load()`.
    pub fn load_from(r: &mut dyn Read) -> io::Result<Self> {
        if read_u32(r)? != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Wrong magic number",
            ));
        }
        if read_u8(r)? != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Unknown version",
            ));
        }
        let n = read_u32(r)? as usize;

        struct ArchEntry {
            tag: u8,
            input_size: usize,
            out_size: usize,
        }

        let arch: Vec<ArchEntry> = (0..n)
            .map(|_| {
                let tag = read_u8(r)?;
                let input_size = read_u32(r)? as usize;
                let out_size = read_u32(r)? as usize;
                Ok(ArchEntry {
                    tag,
                    input_size,
                    out_size,
                })
            })
            .collect::<io::Result<_>>()?;

        let mut layers: Vec<Box<dyn NnLayer>> = Vec::with_capacity(n);
        for e in arch {
            let ctx = LoadCtx {
                input_size: e.input_size,
                output_size: e.out_size,
            };
            layers.push(new_layer(r, e.tag, ctx)?);
        }

        let input_size = layers.first().map(|l| l.input_size()).unwrap_or(0);
        let output_size = layers.last().map(|l| l.output_size()).unwrap_or(0);
        let max_size = layers.iter().map(|l| l.output_size()).max().unwrap_or(0);

        Ok(Sequential {
            input_size,
            output_size,
            layers,
            cache: Vec::new(),
            delta_buf: vec![0.0; max_size],
        })
    }

    /// Load from a file path.
    pub fn load(path: &str) -> io::Result<Self> {
        Self::load_from(&mut File::open(path)?)
    }
}
