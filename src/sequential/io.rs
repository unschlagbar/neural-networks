//! The `Sequential` file format: an `NNM1` container holding one stack blob.
//!
//! ```text
//! container  MAGIC u32 "NNM1" | VERSION u8 | KIND u8 (0 = flat)
//!            N_SECTIONS u32 | per section: name <string>, stack blob
//! stack blob STACK_MAGIC u32 "NNFW" | STACK_VERSION u8 | N_LAYERS u32
//!            per layer: tag u8, input u32, output u32
//!            per layer, same order: layer.save(w), weights only
//! ```
//!
//! Kind 1 (the old hierarchical model) is refused: that model is stored as a v2
//! checkpoint (`src/checkpoint.rs`).

use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Read, Write},
};

use super::Sequential;
use super::layer::NnLayer;
use super::layers::{
    causal_conv1d::CausalConv1dLayer, dropout::DropoutLayer, embedding::EmbeddingLayer,
    linear::LinearLayer, linear_nb::LinearNBLayer, lstm::LSTMLayer, rms_norm::RMSNorm,
    silu_dense::SiluDenseLayer, slstm::SLSTMLayer, soft_cap::SoftCapLayer,
};
use crate::bytes::{
    read_f32, read_f32_vec, read_matrix, read_string, read_u8, read_u32, write_string, write_u8,
    write_u32,
};

pub const MAGIC: u32 = 0x4E4E_4D31;
pub const VERSION: u8 = 2;
const MIN_VERSION: u8 = 1;
const KIND_FLAT: u8 = 0;
const KIND_HIERARCHICAL: u8 = 1;

pub const STACK_MAGIC: u32 = 0x4E4E_4657;
pub const STACK_VERSION: u8 = 2;

const SECTION: &str = "model";

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

impl Sequential {
    /// Write the container atomically: temp file, then rename.
    pub fn save(&self, path: &str) -> io::Result<()> {
        if let Some(dir) = std::path::Path::new(path).parent()
            && !dir.as_os_str().is_empty()
        {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = format!("{path}.tmp");
        let mut w = BufWriter::new(File::create(&tmp)?);
        self.write_to(&mut w)?;
        w.flush()?;
        drop(w);
        std::fs::rename(&tmp, path)
    }

    pub fn load(path: &str) -> io::Result<Self> {
        Self::read_from(&mut BufReader::new(File::open(path)?))
    }

    pub fn write_to(&self, w: &mut dyn Write) -> io::Result<()> {
        write_u32(w, MAGIC)?;
        write_u8(w, VERSION)?;
        write_u8(w, KIND_FLAT)?;
        write_u32(w, 1)?;
        write_string(w, SECTION)?;
        self.write_stack(w)
    }

    pub fn read_from(r: &mut dyn Read) -> io::Result<Self> {
        if read_u32(r)? != MAGIC {
            return Err(invalid("not an NNM1 model file (wrong magic)".into()));
        }
        let version = read_u8(r)?;
        if !(MIN_VERSION..=VERSION).contains(&version) {
            return Err(invalid(format!(
                "unsupported NNM1 version {version} (this build reads {MIN_VERSION}..={VERSION})"
            )));
        }
        match read_u8(r)? {
            KIND_FLAT => {}
            KIND_HIERARCHICAL => {
                return Err(invalid(
                    "a hierarchical NNM1 model; this build reads that model as v2 only".into(),
                ));
            }
            o => return Err(invalid(format!("unknown model kind tag {o}"))),
        }
        let n = read_u32(r)? as usize;
        let mut found = None;
        for _ in 0..n {
            let name = read_string(r)?;
            let stack = Self::load_stack(r)?;
            if name == SECTION {
                found = Some(stack);
            }
        }
        found.ok_or_else(|| invalid(format!("missing section {SECTION:?}")))
    }

    /// Write the layers as a stack blob (no container).
    pub fn write_stack(&self, w: &mut dyn Write) -> io::Result<()> {
        write_u32(w, STACK_MAGIC)?;
        write_u8(w, STACK_VERSION)?;
        write_u32(w, self.layers.len() as u32)?;
        for layer in &self.layers {
            write_u8(w, layer.layer_tag())?;
            write_u32(w, layer.input_size() as u32)?;
            write_u32(w, layer.output_size() as u32)?;
        }
        for layer in &self.layers {
            layer.save(w)?;
        }
        Ok(())
    }

    /// Read a stack blob written by [`write_stack`](Self::write_stack).
    pub fn load_stack(r: &mut dyn Read) -> io::Result<Self> {
        if read_u32(r)? != STACK_MAGIC {
            return Err(invalid("wrong stack magic number".into()));
        }
        if read_u8(r)? != STACK_VERSION {
            return Err(invalid("unknown stack version".into()));
        }
        let n = read_u32(r)? as usize;
        let arch = (0..n)
            .map(|_| Ok((read_u8(r)?, read_u32(r)? as usize, read_u32(r)? as usize)))
            .collect::<io::Result<Vec<_>>>()?;
        let mut layers = Vec::with_capacity(n);
        for (tag, input, output) in arch {
            if let Some(layer) = load_layer(r, tag, input, output)? {
                layers.push(layer);
            }
        }
        Ok(Self::from_layers(layers))
    }
}

fn load_layer(
    r: &mut dyn Read,
    tag: u8,
    input: usize,
    output: usize,
) -> io::Result<Option<Box<dyn NnLayer>>> {
    let layer: Box<dyn NnLayer> = match tag {
        0 => {
            let (wf, wi, wc, wo, b) = (
                read_matrix(r)?,
                read_matrix(r)?,
                read_matrix(r)?,
                read_matrix(r)?,
                read_matrix(r)?,
            );
            let (h_init, c_init) = (read_f32_vec(r)?, read_f32_vec(r)?);
            Box::new(LSTMLayer::from_loaded(
                input, output, wf, wi, wc, wo, b, h_init, c_init,
            ))
        }
        3 => Box::new(LinearNBLayer::from_loaded(input, output, read_matrix(r)?)),
        // A legacy softmax layer: its size field is consumed and the layer dropped.
        4 => {
            read_u32(r)?;
            return Ok(None);
        }
        5 => {
            let (wz, wi, wf, wo) = (
                read_matrix(r)?,
                read_matrix(r)?,
                read_matrix(r)?,
                read_matrix(r)?,
            );
            let (bz, bi, bf, bo) = (
                read_f32_vec(r)?,
                read_f32_vec(r)?,
                read_f32_vec(r)?,
                read_f32_vec(r)?,
            );
            let (h_init, c_init) = (read_f32_vec(r)?, read_f32_vec(r)?);
            Box::new(SLSTMLayer::from_loaded(
                input, output, wz, wi, wf, wo, bz, bi, bf, bo, h_init, c_init,
            ))
        }
        6 => Box::new(DropoutLayer::new(input, read_f32(r)?)),
        7 => {
            let w = read_matrix(r)?;
            Box::new(EmbeddingLayer::from_loaded(w.rows(), w.cols(), w))
        }
        9 => Box::new(RMSNorm::from_loaded(input, read_f32_vec(r)?)),
        10 => Box::new(SiluDenseLayer::from_loaded(
            input,
            output,
            read_matrix(r)?,
            read_f32_vec(r)?,
        )),
        12 => {
            let (i, o) = (read_u32(r)? as usize, read_u32(r)? as usize);
            Box::new(LinearLayer::from_loaded(
                i,
                o,
                read_matrix(r)?,
                read_f32_vec(r)?,
            ))
        }
        15 => {
            let kernel = read_u32(r)? as usize;
            Box::new(CausalConv1dLayer::from_loaded(
                input,
                kernel,
                read_matrix(r)?,
                read_f32_vec(r)?,
            ))
        }
        17 => Box::new(SoftCapLayer::new(input, read_f32(r)?)),
        o => return Err(invalid(format!("unknown layer tag {o}"))),
    };
    Ok(Some(layer))
}

#[cfg(test)]
mod tests {
    use super::super::SequentialBuilder;
    use super::*;
    use std::io::Cursor;

    /// Weights survive a round trip: a re-save of the reloaded model is the same bytes.
    #[test]
    fn container_roundtrips() {
        let model = SequentialBuilder::new(8)
            .embedding(16)
            .rms_norm()
            .silu_dense(12)
            .slstm(10)
            .lstm(6)
            .causal_conv1d(3)
            .soft_cap(30.0)
            .linear(4)
            .linear_no_bias(3)
            .build();
        let mut bytes = Vec::new();
        model.write_to(&mut bytes).unwrap();
        let back = Sequential::read_from(&mut Cursor::new(bytes.as_slice())).unwrap();
        let mut again = Vec::new();
        back.write_to(&mut again).unwrap();
        assert_eq!(bytes, again);
    }

    #[test]
    fn hierarchical_kind_is_refused() {
        let mut bytes = Vec::new();
        write_u32(&mut bytes, MAGIC).unwrap();
        write_u8(&mut bytes, VERSION).unwrap();
        write_u8(&mut bytes, KIND_HIERARCHICAL).unwrap();
        assert!(Sequential::read_from(&mut Cursor::new(bytes.as_slice())).is_err());
    }
}
