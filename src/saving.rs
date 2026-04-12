// ── saving.rs ─────────────────────────────────────────────────────────────────
//
// File format v2
// ──────────────
//
//  ┌─────────────────────────────────────────────────────┐
//  │  MAGIC      u32   0x4E4E_4657  ("NNFW")             │
//  │  VERSION    u8    2                                  │
//  │  N_LAYERS   u32                                      │
//  ├─────────────────────────────────────────────────────┤  ← Architecture header
//  │  for each layer:                                     │
//  │    tag        u8                                     │
//  │    input_sz   u32                                    │
//  │    output_sz  u32                                    │
//  ├─────────────────────────────────────────────────────┤
//  │  for each layer (same order):                        │  ← Weights
//  │    layer.save(w)  — weights only, no shapes          │
//  └─────────────────────────────────────────────────────┘
//
// Hierarchical format ("HIER")  — written by HierarchicalSequential::save
// ──────────────────────────────
//
//  ┌─────────────────────────────────────────────────────┐
//  │  HIER_MAGIC   u32   0x4849_4552  ("HIER")           │
//  │  vocab_size   u32                                    │
//  │  context_size u32                                    │
//  │  n_boundaries u32                                    │
//  │  boundary_ids [u16 × n_boundaries]                  │
//  │  char_model   <Sequential blob>                     │
//  │  high_model   <Sequential blob>                     │
//  └─────────────────────────────────────────────────────┘

use std::{
    fs::File,
    io::{self, Cursor, Write},
};

use iron_oxide::collections::Matrix;

use crate::sequential::Sequential;

pub const MAGIC: u32 = 0x4E4E_4657;
pub const HIER_MAGIC: u32 = 0x4849_4552;
pub const VERSION: u8 = 2;

// ── Primitive writers ─────────────────────────────────────────────────────────

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

// ── Sequential::save ──────────────────────────────────────────────────────────

impl Sequential {
    /// Write the full model (header + weights) to any `Write`.
    /// Used both by `save()` and by `HierarchicalSequential::save()`.
    pub fn write_to(&self, w: &mut dyn Write) -> io::Result<()> {
        write_u32(w, MAGIC)?;
        write_u8(w, VERSION)?;
        write_u32(w, self.layers.len() as u32)?;

        // Architecture header
        for layer in &self.layers {
            write_u8(w, layer.layer_tag())?;
            write_u32(w, layer.input_size() as u32)?;
            write_u32(w, layer.output_size() as u32)?;
        }

        // Weights (same order)
        for layer in &self.layers {
            layer.save(w)?;
        }
        Ok(())
    }

    /// Save to a file at `path`.
    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut buf = Cursor::new(Vec::new());
        self.write_to(&mut buf)?;
        File::create(path)?.write_all(&buf.into_inner())
    }
}
