// ── loading.rs ────────────────────────────────────────────────────────────────

use std::{
    fs::File,
    io::{self, Read},
};

use iron_oxide::collections::Matrix;

use crate::{
    nn::{
        dropout::DropoutLayer,
        embedding::EmbeddingLayer,
        linear::LinearLayer,
        linear_nb::LinearNBLayer,
        lstm::LSTMLayer,
        mlstm::MLSTMLayer,
        mlstm_block::MLSTMBlock,
        rms_norm::{RMSNorm, RMSNormResidual},
        silu_dense::SiluDenseLayer,
        slstm::SLSTMLayer,
        slstm_block::SLSTMBlock,
        softmax::SoftmaxLayer,
    },
    nn_layer::NnLayer,
    saving::{MAGIC, VERSION},
    sequential::Sequential,
};

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
    Ok(f32::from_le_bytes(b))
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

pub fn load_linear(r: &mut dyn Read) -> io::Result<Box<dyn NnLayer>> {
    let input = read_u32(r)? as usize;
    let output = read_u32(r)? as usize;
    let weights = read_matrix(r)?;
    let biases: Box<[f32]> = read_f32_vec(r)?.into();

    let layer: Box<dyn NnLayer> =
        Box::new(LinearLayer::from_loaded(input, output, weights, biases));
    Ok(layer)
}

pub fn load_embedding(r: &mut dyn Read) -> io::Result<Box<dyn NnLayer>> {
    let weights = read_matrix(r)?;
    let input = weights.rows();
    let output = weights.cols();

    let layer: Box<dyn NnLayer> = Box::new(EmbeddingLayer::from_loaded(input, output, weights));
    Ok(layer)
}

pub fn load_linear_nb(r: &mut dyn Read, ctx: LoadCtx) -> io::Result<Box<dyn NnLayer>> {
    let weights = read_matrix(r)?;
    let input = ctx.input_size;
    let output = ctx.output_size;

    let layer: Box<dyn NnLayer> = Box::new(LinearNBLayer::from_loaded(input, output, weights));
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

pub fn load_lstm(r: &mut dyn Read, ctx: LoadCtx) -> io::Result<Box<dyn NnLayer>> {
    let wf = read_matrix(r)?;
    let wi = read_matrix(r)?;
    let wc = read_matrix(r)?;
    let wo = read_matrix(r)?;
    let b = read_matrix(r)?;
    let h_init = read_f32_vec(r)?.into();
    let c_init = read_f32_vec(r)?.into();
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

pub fn load_res_norm(r: &mut dyn Read, _ctx: LoadCtx) -> io::Result<Box<dyn NnLayer>> {
    let gamma = read_f32_vec(r)?.into();

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
    let gamma = read_f32_vec(r)?.into();
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
    let h_init = read_f32_vec(r)?.into();
    let c_init = read_f32_vec(r)?.into();
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
    let biases: Box<[f32]> = read_f32_vec(r)?.into();
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

    // Norm-Gewichte
    let pre_gamma: Box<[f32]> = read_f32_vec(r)?.into();
    let post_gamma: Box<[f32]> = read_f32_vec(r)?.into();
    let pre_norm = RMSNorm::from_loaded(hidden_size, pre_gamma);
    let post_norm = RMSNorm::from_loaded(hidden_size, post_gamma);

    // Zell-Gewichte (identische Reihenfolge wie in load_slstm)
    let wz = read_matrix(r)?;
    let wi = read_matrix(r)?;
    let wf = read_matrix(r)?;
    let wo = read_matrix(r)?;
    let b = read_matrix(r)?;
    let h_init: Box<[f32]> = read_f32_vec(r)?.into();
    let c_init: Box<[f32]> = read_f32_vec(r)?.into();
    let cell = SLSTMLayer::from_loaded(hidden_size, hidden_size, wz, wi, wf, wo, b, h_init, c_init);

    // SwiGLU-Projektionen
    let lin_gate = LinearLayer::from_loaded(
        hidden_size,
        up_size,
        read_matrix(r)?,
        read_f32_vec(r)?.into(),
    );
    let lin_value = LinearLayer::from_loaded(
        hidden_size,
        up_size,
        read_matrix(r)?,
        read_f32_vec(r)?.into(),
    );
    let lin_down = LinearLayer::from_loaded(
        up_size,
        hidden_size,
        read_matrix(r)?,
        read_f32_vec(r)?.into(),
    );

    Ok(Box::new(SLSTMBlock::from_loaded(
        hidden_size,
        up_size,
        pre_norm,
        post_norm,
        cell,
        lin_gate,
        lin_value,
        lin_down,
    )))
}

pub fn load_mlstm(r: &mut dyn Read, ctx: LoadCtx) -> io::Result<Box<dyn NnLayer>> {
    let wq = read_matrix(r)?;
    let wk = read_matrix(r)?;
    let wv = read_matrix(r)?;
    let wo = read_matrix(r)?;
    let wi: Box<[f32]> = read_f32_vec(r)?.into();
    let wf: Box<[f32]> = read_f32_vec(r)?.into();
    let bq: Box<[f32]> = read_f32_vec(r)?.into();
    let bk: Box<[f32]> = read_f32_vec(r)?.into();
    let bv: Box<[f32]> = read_f32_vec(r)?.into();
    let bo: Box<[f32]> = read_f32_vec(r)?.into();
    let bi_bf = read_f32_vec(r)?;
    Ok(Box::new(MLSTMLayer::from_loaded(
        ctx.input_size,
        ctx.output_size,
        wq,
        wk,
        wv,
        wo,
        wi,
        wf,
        bq,
        bk,
        bv,
        bo,
        bi_bf[0],
        bi_bf[1],
    )))
}

pub fn load_mlstm_block(r: &mut dyn Read, ctx: LoadCtx) -> io::Result<Box<dyn NnLayer>> {
    let d = ctx.input_size;
    let num_heads = read_u32(r)? as usize;
    let d_qk = read_u32(r)? as usize;
    let d_hv = read_u32(r)? as usize;
    let up_size = read_u32(r)? as usize;
    let mut block = MLSTMBlock::new(d, num_heads, d_qk, d_hv, up_size);
    block.norm1.gamma = read_f32_vec(r)?.into();
    block.norm2.gamma = read_f32_vec(r)?.into();
    let ml = &mut block.mlstm;
    ml.wq = read_matrix(r)?;
    ml.wk = read_matrix(r)?;
    ml.wv = read_matrix(r)?;
    ml.wo = read_matrix(r)?;
    ml.wi = read_matrix(r)?;
    ml.wf = read_matrix(r)?;
    ml.bq = read_f32_vec(r)?.into();
    ml.bk = read_f32_vec(r)?.into();
    ml.bv = read_f32_vec(r)?.into();
    ml.bo = read_f32_vec(r)?.into();
    ml.bi = read_f32_vec(r)?.into();
    ml.bf = read_f32_vec(r)?.into();
    ml.w_out.weights = read_matrix(r)?;
    ml.w_out.biases = read_f32_vec(r)?.into();
    block.lin_gate.weights = read_matrix(r)?;
    block.lin_gate.biases = read_f32_vec(r)?.into();
    block.lin_value.weights = read_matrix(r)?;
    block.lin_value.biases = read_f32_vec(r)?.into();
    block.lin_down.weights = read_matrix(r)?;
    block.lin_down.biases = read_f32_vec(r)?.into();
    Ok(Box::new(block))
}

// ── Layer factory ─────────────────────────────────────────────────────────────

fn new_layer(r: &mut dyn Read, tag: u8, ctx: LoadCtx) -> io::Result<Box<dyn NnLayer>> {
    match tag {
        0 => load_lstm(r, ctx),
        3 => load_linear_nb(r, ctx),
        4 => load_softmax(r),
        5 => load_slstm(r, ctx),
        6 => load_dropout(r, ctx),
        7 => load_embedding(r),
        8 => load_res_norm(r, ctx),
        9 => load_norm(r, ctx),
        10 => load_silu_dense(r, ctx),
        11 => load_slstm_block(r, ctx),
        12 => load_linear(r),
        13 => load_mlstm(r, ctx),
        14 => load_mlstm_block(r, ctx),

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
            delta_buf: vec![0.0; max_size].into(),
        })
    }

    /// Load from a file path.
    pub fn load(path: &str) -> io::Result<Self> {
        Self::load_from(&mut File::open(path)?)
    }
}
