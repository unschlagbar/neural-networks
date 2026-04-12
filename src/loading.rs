// ── loading.rs ────────────────────────────────────────────────────────────────

use std::{
    fs::File,
    io::{self, Read},
};

use iron_oxide::collections::Matrix;

use crate::{
    activations::{LeakyRelu, Linear, Relu, Sigmoid, Tanh},
    dropout::DropoutLayer,
    layer::DenseLayer,
    lstm::LSTMLayer,
    nn_layer::NnLayer,
    parallel::ParallelLayer,
    projection::Projection,
    saving::{MAGIC, VERSION},
    sequential::Sequential,
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

    use crate::indrnn::IndRNNLayer;
    let layer: Box<dyn NnLayer> = match act_id {
        0 => Box::new(IndRNNLayer::from_loaded(input, hidden, Linear, w, u, b)),
        1 => Box::new(IndRNNLayer::from_loaded(input, hidden, Relu, w, u, b)),
        2 => Box::new(IndRNNLayer::from_loaded(input, hidden, Tanh, w, u, b)),
        3 => Box::new(IndRNNLayer::from_loaded(input, hidden, Sigmoid, w, u, b)),
        4 => Box::new(IndRNNLayer::from_loaded(input, hidden, LeakyRelu, w, u, b)),
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
    Ok(Box::new(LSTMLayer::from_loaded(
        ctx.input_size,
        ctx.output_size,
        wf,
        wi,
        wc,
        wo,
        b,
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

// ── Layer factory ─────────────────────────────────────────────────────────────

fn new_layer(r: &mut dyn Read, tag: u8, ctx: LoadCtx) -> io::Result<Box<dyn NnLayer>> {
    match tag {
        0 => load_lstm(r, ctx),
        1 => load_dense(r),
        2 => load_indrnn(r, ctx),
        3 => load_projection(r, ctx),
        4 => load_softmax(r),
        6 => load_dropout(r, ctx),
        7 => load_parallel(r, ctx),
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
