//! Little-endian byte primitives shared by the `Sequential` file format and the
//! audio encoder checkpoint.

use std::io::{self, Read, Write};

use iron_oxide::collections::Matrix;

pub fn read_u8(r: &mut dyn Read) -> io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
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
/// u32 length, then that many f32.
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
/// u32 byte count, then UTF-8.
pub fn read_string(r: &mut dyn Read) -> io::Result<String> {
    let len = read_u32(r)? as usize;
    let mut bytes = vec![0u8; len];
    r.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

pub fn write_u8(w: &mut dyn Write, v: u8) -> io::Result<()> {
    w.write_all(&[v])
}
pub fn write_u32(w: &mut dyn Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
pub fn write_u64(w: &mut dyn Write, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
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
pub fn write_string(w: &mut dyn Write, s: &str) -> io::Result<()> {
    write_u32(w, s.len() as u32)?;
    w.write_all(s.as_bytes())
}
