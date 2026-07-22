// Stack blob codec — the reusable per-`Sequential` serialization.
//
// A "stack blob" is one `Sequential`'s architecture header plus its weights. It
// is the building block the unified container (`src/format.rs`) frames as a
// named section, and it is also used in-memory for the replica round-trip and
// for building GPU models. It is NOT a standalone file format — files are
// always the `NNM1` container.
//
//  ┌─────────────────────────────────────────────────────┐
//  │  STACK_MAGIC  u32   0x4E4E_4657  ("NNFW")           │
//  │  STACK_VERSION u8                                   │
//  │  N_LAYERS     u32                                   │
//  ├─────────────────────────────────────────────────────┤  ← Architecture header
//  │  for each layer:                                    │
//  │    tag        u8                                    │
//  │    input_sz   u32                                   │
//  │    output_sz  u32                                   │
//  ├─────────────────────────────────────────────────────┤
//  │  for each layer (same order):                       │  ← Weights
//  │    layer.save(w)  — weights only, no shapes         │
//  └─────────────────────────────────────────────────────┘
//
// The byte primitives below (`write_u*`, `write_matrix`, `write_string`, …) are
// shared by both this codec and the container.

use std::io::{self, Write};

use iron_oxide::collections::Matrix;

use crate::sequential::Sequential;

/// Magic and version for a single stack blob (see module docs). Distinct from
/// the container magic in `src/format.rs`.
pub const STACK_MAGIC: u32 = 0x4E4E_4657;
pub const STACK_VERSION: u8 = 2;

#[inline]
pub fn write_u8(w: &mut dyn Write, v: u8) -> io::Result<()> {
    w.write_all(&[v])
}
#[inline]
pub fn write_u16(w: &mut dyn Write, v: u16) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
#[inline]
pub fn write_u32(w: &mut dyn Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
#[inline]
pub fn write_u64(w: &mut dyn Write, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
#[inline]
pub fn write_f32(w: &mut dyn Write, v: f32) -> io::Result<()> {
    w.write_all(&v.to_bits().to_le_bytes())
}
pub fn write_f32_slice(w: &mut dyn Write, s: &[f32]) -> io::Result<()> {
    write_u32(w, s.len() as u32)?;
    for &v in s {
        write_f32(w, v)?;
    }
    Ok(())
}
pub fn write_matrix(w: &mut dyn Write, m: &Matrix) -> io::Result<()> {
    write_u32(w, m.rows() as u32)?;
    write_u32(w, m.cols() as u32)?;
    write_f32_slice(w, m.as_slice())
}
/// Length-prefixed UTF-8 string (u32 byte count + bytes).
pub fn write_string(w: &mut dyn Write, s: &str) -> io::Result<()> {
    write_u32(w, s.len() as u32)?;
    w.write_all(s.as_bytes())
}

impl Sequential {
    /// Write this stack as a blob (arch header + weights) to any `Write`. The
    /// unified container frames this as a named section; it is also used for the
    /// in-memory replica round-trip and GPU model building. Not a file format on
    /// its own — files are the `NNM1` container (`src/format.rs`).
    pub fn write_stack(&self, w: &mut dyn Write) -> io::Result<()> {
        write_layers(w, &self.layers)
    }
}

/// Write a layer stack as a blob (arch header + weights). The primitive behind
/// [`Sequential::write_stack`]; used directly for multi-layer stages.
pub fn write_layers(
    w: &mut dyn Write,
    layers: &[Box<dyn crate::nn_layer::NnLayer>],
) -> io::Result<()> {
    write_stack_blob(w, layers.iter().map(|l| l.as_ref()))
}

/// Write a single standalone layer as a one-layer stack blob (e.g. the encoder
/// combine projection, which is not wrapped in a `Sequential`).
pub fn write_one_layer(w: &mut dyn Write, layer: &dyn crate::nn_layer::NnLayer) -> io::Result<()> {
    write_stack_blob(w, std::iter::once(layer))
}

/// Shared stack-blob writer: `STACK_MAGIC`, version, count, arch header, then
/// per-layer weights. Iterated twice, so takes a `Clone` iterator.
fn write_stack_blob<'a, I>(w: &mut dyn Write, layers: I) -> io::Result<()>
where
    I: Iterator<Item = &'a dyn crate::nn_layer::NnLayer> + Clone,
{
    write_u32(w, STACK_MAGIC)?;
    write_u8(w, STACK_VERSION)?;
    write_u32(w, layers.clone().count() as u32)?;

    // Architecture header
    for layer in layers.clone() {
        write_u8(w, layer.layer_tag())?;
        write_u32(w, layer.input_size() as u32)?;
        write_u32(w, layer.output_size() as u32)?;
    }

    // Weights (same order)
    for layer in layers {
        layer.save(w)?;
    }
    Ok(())
}
