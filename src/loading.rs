use std::io::{self, Read};

use iron_oxide::collections::Matrix;

use crate::{
    nn::{
        causal_conv1d::CausalConv1dLayer, dropout::DropoutLayer, embedding::EmbeddingLayer,
        linear::LinearLayer, linear_nb::LinearNBLayer, lstm::LSTMLayer, mlstm::MLSTMLayer,
        mlstm_block::MLSTMBlock, rms_norm::RMSNorm, silu_dense::SiluDenseLayer, slstm::SLSTMLayer,
        slstm_block::SLSTMBlock, soft_cap::SoftCapLayer,
    },
    nn_layer::NnLayer,
    saving::{STACK_MAGIC, STACK_VERSION},
    sequential::Sequential,
};

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
pub fn read_u64(r: &mut dyn Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
pub fn read_f32(r: &mut dyn Read) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}
pub fn read_f32_vec(r: &mut dyn Read) -> io::Result<Box<[f32]>> {
    let len = read_u32(r)? as usize;
    (0..len).map(|_| read_f32(r)).collect()
}
pub fn read_matrix(r: &mut dyn Read) -> io::Result<Matrix> {
    let rows = read_u32(r)? as usize;
    let cols = read_u32(r)? as usize;
    let flat = read_f32_vec(r)?;
    Ok(Matrix::from_box(flat, rows, cols))
}
/// Length-prefixed UTF-8 string (u32 byte count + bytes).
pub fn read_string(r: &mut dyn Read) -> io::Result<String> {
    let len = read_u32(r)? as usize;
    let mut bytes = vec![0u8; len];
    r.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

pub struct LoadCtx {
    pub input_size: usize,
    pub output_size: usize,
}

pub fn load_linear(r: &mut dyn Read) -> io::Result<Box<dyn NnLayer>> {
    let input = read_u32(r)? as usize;
    let output = read_u32(r)? as usize;
    let weights = read_matrix(r)?;
    let biases: Box<[f32]> = read_f32_vec(r)?;

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

pub fn load_dropout(r: &mut dyn Read, ctx: LoadCtx) -> io::Result<Box<dyn NnLayer>> {
    let rate = read_f32(r)?;
    Ok(Box::new(DropoutLayer::new(ctx.input_size, rate)))
}

pub fn load_soft_cap(r: &mut dyn Read, ctx: LoadCtx) -> io::Result<Box<dyn NnLayer>> {
    let cap = read_f32(r)?;
    Ok(Box::new(SoftCapLayer::new(ctx.input_size, cap)))
}

pub fn load_lstm(r: &mut dyn Read, ctx: LoadCtx) -> io::Result<Box<dyn NnLayer>> {
    let wf = read_matrix(r)?;
    let wi = read_matrix(r)?;
    let wc = read_matrix(r)?;
    let wo = read_matrix(r)?;
    let b = read_matrix(r)?;
    let h_init = read_f32_vec(r)?;
    let c_init = read_f32_vec(r)?;
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

pub fn load_norm(r: &mut dyn Read, ctx: LoadCtx) -> io::Result<Box<dyn NnLayer>> {
    let gamma = read_f32_vec(r)?;
    let mut wrapper = RMSNorm::new(ctx.input_size);
    wrapper.gamma = gamma;
    Ok(Box::new(wrapper))
}

pub fn load_slstm(r: &mut dyn Read, ctx: LoadCtx) -> io::Result<Box<dyn NnLayer>> {
    let wz = read_matrix(r)?;
    let wi = read_matrix(r)?;
    let wf = read_matrix(r)?;
    let wo = read_matrix(r)?;
    let bz = read_f32_vec(r)?;
    let bi = read_f32_vec(r)?;
    let bf = read_f32_vec(r)?;
    let bo = read_f32_vec(r)?;
    let h_init = read_f32_vec(r)?;
    let c_init = read_f32_vec(r)?;
    Ok(Box::new(SLSTMLayer::from_loaded(
        ctx.input_size,
        ctx.output_size,
        wz,
        wi,
        wf,
        wo,
        bz,
        bi,
        bf,
        bo,
        h_init,
        c_init,
    )))
}

pub fn load_silu_dense(r: &mut dyn Read, ctx: LoadCtx) -> io::Result<Box<dyn NnLayer>> {
    let input = ctx.input_size;
    let output = ctx.output_size;
    let weights = read_matrix(r)?;
    let biases: Box<[f32]> = read_f32_vec(r)?;
    Ok(Box::new(SiluDenseLayer::from_loaded(
        input, output, weights, biases,
    )))
}

/// xLSTM-style sLSTM-Block. Layout must match `SLSTMBlock::save` exactly.
///
/// Sequential-Header liefert uns hidden_size (= ctx.input_size == ctx.output_size).
/// Danach in dieser Reihenfolge:
///   - up_size: u32
///   - pre_norm1.gamma, post_cell_norm.gamma, pre_norm2.gamma: f32_vec(H)
///   - cell weights: wz, wi, wf, wo, bz, bi, bf, bo, h_init, c_init  (as in `SLSTMLayer::save`)
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

    // Norm weights
    let pre_norm1 = RMSNorm::from_loaded(hidden_size, read_f32_vec(r)?);
    let post_cell_norm = RMSNorm::from_loaded(hidden_size, read_f32_vec(r)?);
    let pre_norm2 = RMSNorm::from_loaded(hidden_size, read_f32_vec(r)?);

    // Cell weights (identical order to load_slstm)
    let wz = read_matrix(r)?;
    let wi = read_matrix(r)?;
    let wf = read_matrix(r)?;
    let wo = read_matrix(r)?;
    let bz: Box<[f32]> = read_f32_vec(r)?;
    let bi: Box<[f32]> = read_f32_vec(r)?;
    let bf: Box<[f32]> = read_f32_vec(r)?;
    let bo: Box<[f32]> = read_f32_vec(r)?;
    let h_init: Box<[f32]> = read_f32_vec(r)?;
    let c_init: Box<[f32]> = read_f32_vec(r)?;
    let cell = SLSTMLayer::from_loaded(
        hidden_size,
        hidden_size,
        wz,
        wi,
        wf,
        wo,
        bz,
        bi,
        bf,
        bo,
        h_init,
        c_init,
    );

    // SwiGLU-Projektionen
    let lin_gate =
        LinearLayer::from_loaded(hidden_size, up_size, read_matrix(r)?, read_f32_vec(r)?);
    let lin_value =
        LinearLayer::from_loaded(hidden_size, up_size, read_matrix(r)?, read_f32_vec(r)?);
    let lin_down =
        LinearLayer::from_loaded(up_size, hidden_size, read_matrix(r)?, read_f32_vec(r)?);

    Ok(Box::new(SLSTMBlock::from_loaded(
        hidden_size,
        up_size,
        pre_norm1,
        post_cell_norm,
        pre_norm2,
        cell,
        lin_gate,
        lin_value,
        lin_down,
    )))
}

pub fn load_mlstm(
    r: &mut dyn Read,
    input_size: usize,
    output_size: usize,
) -> std::io::Result<MLSTMLayer> {
    let num_heads = read_u32(r)? as usize;
    let dqk = read_u32(r)? as usize;

    let wq = read_matrix(r)?;
    let wk = read_matrix(r)?;
    let wv = read_matrix(r)?;
    let wo = read_matrix(r)?;
    let wi = read_matrix(r)?;
    let wf = read_matrix(r)?;

    let bq: Box<[f32]> = read_f32_vec(r)?.into();
    let bk: Box<[f32]> = read_f32_vec(r)?.into();
    let bv: Box<[f32]> = read_f32_vec(r)?.into();
    let bo: Box<[f32]> = read_f32_vec(r)?.into();
    let bi: Box<[f32]> = read_f32_vec(r)?.into();
    let bf: Box<[f32]> = read_f32_vec(r)?.into();

    let w_out_weights = read_matrix(r)?;
    let w_out_biases: Box<[f32]> = read_f32_vec(r)?.into();
    let w_out = LinearLayer::from_loaded(output_size, output_size, w_out_weights, w_out_biases);
    let head_norm_gamma: Box<[f32]> = read_f32_vec(r)?.into();

    Ok(MLSTMLayer::from_loaded(
        input_size,
        output_size,
        num_heads,
        dqk,
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
        bi,
        bf,
        w_out,
        head_norm_gamma,
    ))
}

pub fn load_mlstm_block(r: &mut dyn Read, hidden_size: usize) -> std::io::Result<Box<dyn NnLayer>> {
    let up_size = read_u32(r)? as usize;

    let pre_gamma: Box<[f32]> = read_f32_vec(r)?.into();
    let post_gamma: Box<[f32]> = read_f32_vec(r)?.into();
    let pre_norm = RMSNorm::from_loaded(hidden_size, pre_gamma);
    let post_norm = RMSNorm::from_loaded(hidden_size, post_gamma);

    let cell = load_mlstm(r, hidden_size, hidden_size)?;

    let make_lin = |r: &mut dyn Read, rows, cols| -> std::io::Result<LinearLayer> {
        let w = read_matrix(r)?;
        let b: Box<[f32]> = read_f32_vec(r)?.into();
        Ok(LinearLayer::from_loaded(rows, cols, w, b))
    };

    let lin_gate = make_lin(r, hidden_size, up_size)?;
    let lin_value = make_lin(r, hidden_size, up_size)?;
    let lin_down = make_lin(r, up_size, hidden_size)?;

    Ok(Box::new(MLSTMBlock::from_loaded(
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

pub fn load_causal_conv1d(r: &mut dyn Read, ctx: LoadCtx) -> io::Result<Box<dyn NnLayer>> {
    let channels = ctx.input_size;
    let kernel_size = read_u32(r)? as usize;
    let weights = read_matrix(r)?;
    let bias = read_f32_vec(r)?;
    Ok(Box::new(CausalConv1dLayer::from_loaded(
        channels,
        kernel_size,
        weights,
        bias,
    )))
}

fn new_layer(r: &mut dyn Read, tag: u8, ctx: LoadCtx) -> io::Result<Option<Box<dyn NnLayer>>> {
    if tag == 4 {
        read_u32(r)?; // legacy softmax — consume size field and skip
        return Ok(None);
    }
    let layer: Box<dyn NnLayer> = match tag {
        0 => load_lstm(r, ctx)?,
        3 => load_linear_nb(r, ctx)?,
        5 => load_slstm(r, ctx)?,
        6 => load_dropout(r, ctx)?,
        7 => load_embedding(r)?,
        9 => load_norm(r, ctx)?,
        10 => load_silu_dense(r, ctx)?,
        11 => load_slstm_block(r, ctx)?,
        12 => load_linear(r)?,
        13 => Box::new(load_mlstm(r, ctx.input_size, ctx.output_size).unwrap()),
        14 => load_mlstm_block(r, ctx.output_size)?,
        15 => load_causal_conv1d(r, ctx)?,
        17 => load_soft_cap(r, ctx)?,
        o => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unknown layer tag {o}"),
            ));
        }
    };
    Ok(Some(layer))
}

/// Load a layer stack (arch header + weights) from a blob. The primitive behind
/// [`Sequential::load_stack`]; used directly for single-layer stages.
pub fn load_layers(r: &mut dyn Read) -> io::Result<Vec<Box<dyn NnLayer>>> {
    if read_u32(r)? != STACK_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Wrong stack magic number",
        ));
    }
    if read_u8(r)? != STACK_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Unknown stack version",
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
        if let Some(layer) = new_layer(r, e.tag, ctx)? {
            layers.push(layer);
        }
    }
    Ok(layers)
}

impl Sequential {
    /// Load a stack blob (arch header + weights) from any `Read` source. The
    /// counterpart to [`Sequential::write_stack`]; the container in
    /// `src/format.rs` calls this per section.
    pub fn load_stack(r: &mut dyn Read) -> io::Result<Self> {
        let layers = load_layers(r)?;
        let input_size = layers.first().map(|l| l.input_size()).unwrap_or(0);
        let output_size = layers.last().map(|l| l.output_size()).unwrap_or(0);
        let max_size = layers.iter().map(|l| l.output_size()).max().unwrap_or(0);

        Ok(Sequential {
            input_size,
            output_size,
            layers,
            cache: Vec::new(),
            delta_buf: vec![0.0; max_size].into(),
            input_buf: vec![0.0; input_size].into(),
        })
    }
}
