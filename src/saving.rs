use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};

use iron_oxide::collections::Matrix;

use crate::layer::{Activation, DenseLayer};
use crate::lstm::LSTMLayer;
use crate::sequential::{Layer, LayerGrads, Sequential};

impl Sequential {
    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut w = BufWriter::new(File::create(path)?);

        // Header (kleine Versionskennung)
        write_u32(&mut w, 0x53455131)?; // "SEQ1"
        write_u32(&mut w, 1)?; // version = 1

        // Anzahl Layer
        write_u32(&mut w, self.layers.len() as u32)?;

        for layer in &self.layers {
            match layer {
                Layer::Lstm(l) => {
                    write_u8(&mut w, 0)?; // Tag: 0 = LSTM
                    write_u32(&mut w, l.input_size as u32)?;
                    write_u32(&mut w, l.hidden_size as u32)?;
                    write_matrix(&mut w, &l.wf)?;
                    write_matrix(&mut w, &l.wi)?;
                    write_matrix(&mut w, &l.wc)?;
                    write_matrix(&mut w, &l.wo)?;
                    write_matrix(&mut w, &l.b)?;
                }
                Layer::Dense(l) => {
                    write_u8(&mut w, 1)?; // Tag: 1 = Dense
                    // input_size/hidden_size sind aus weights ableitbar, aber wir schreiben sie explizit mit
                    write_u32(&mut w, l.input_size() as u32)?;
                    write_u32(&mut w, l.hidden_size() as u32)?;
                    write_activation(&mut w, &l.activation)?;
                    write_matrix(&mut w, &l.weights)?;
                    write_f32_slice(&mut w, &l.biases)?;
                }
            }
        }

        w.flush()
    }

    pub fn load(path: &str) -> io::Result<Self> {
        let mut r = BufReader::new(File::open(path)?);

        // Header prüfen
        let magic = read_u32(&mut r)?;
        if magic != 0x53455131 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Bad model header",
            ));
        }
        let _version = read_u32(&mut r)?; // aktuell ungenutzt, reserviert für Upgrades

        let n_layers = read_u32(&mut r)? as usize;
        let mut layers = Vec::with_capacity(n_layers);

        for _ in 0..n_layers {
            let tag = read_u8(&mut r)?;
            match tag {
                0 => {
                    // LSTM
                    let input_size = read_u32(&mut r)? as usize;
                    let hidden_size = read_u32(&mut r)? as usize;

                    let wf = read_matrix(&mut r)?;
                    let wi = read_matrix(&mut r)?;
                    let wc = read_matrix(&mut r)?;
                    let wo = read_matrix(&mut r)?;
                    let b = read_matrix(&mut r)?;

                    let layer = LSTMLayer {
                        input_size,
                        hidden_size,
                        wf,
                        wi,
                        wc,
                        wo,
                        b,
                        d: Matrix::zeros(2, hidden_size),
                    };
                    layers.push(Layer::Lstm(layer));
                }
                1 => {
                    // Dense
                    let input_size = read_u32(&mut r)? as usize;
                    let hidden_size = read_u32(&mut r)? as usize;
                    let activation = read_activation(&mut r)?;
                    let weights = read_matrix(&mut r)?;
                    let biases = read_f32_vec(&mut r)?;

                    // defensive checks (optional)
                    debug_assert_eq!(weights.rows(), input_size);
                    debug_assert_eq!(weights.cols(), hidden_size);
                    debug_assert_eq!(biases.len(), hidden_size);

                    let layer = DenseLayer {
                        weights,
                        biases,
                        activation,
                        // Forward-Cache leer initialisieren
                        last_z: vec![0.0; hidden_size],
                    };
                    layers.push(Layer::Dense(layer));
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Unknown layer tag",
                    ));
                }
            }
        }

        // Grads passend zu den geladenen Layern initialisieren
        let grads = layers.iter().map(LayerGrads::from_layer).collect();

        let mut dh_next = Vec::with_capacity(layers.len());
        let mut dc_next = Vec::with_capacity(layers.len());

        for lay in &layers {
            dh_next.push(vec![0.0; lay.hidden_size()]);
            if lay.is_recurrent() {
                dc_next.push(vec![0.0; lay.hidden_size()]);
            } else {
                dc_next.push(Vec::with_capacity(0));
            }
        }

        Ok(Sequential {
            layers,
            grads,
            dh_next,
            dc_next,
        })
    }
}

/* ---------- IO-Helper ---------- */

fn write_u8<W: Write>(w: &mut W, v: u8) -> io::Result<()> {
    w.write_all(&[v])
}
fn read_u8<R: Read>(r: &mut R) -> io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

fn write_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn write_f32<W: Write>(w: &mut W, v: f32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn read_f32<R: Read>(r: &mut R) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn write_f32_slice<W: Write>(w: &mut W, xs: &[f32]) -> io::Result<()> {
    write_u32(w, xs.len() as u32)?;
    for &v in xs {
        write_f32(w, v)?;
    }
    Ok(())
}
fn read_f32_vec<R: Read>(r: &mut R) -> io::Result<Vec<f32>> {
    let len = read_u32(r)? as usize;
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        v.push(read_f32(r)?);
    }
    Ok(v)
}

fn write_matrix<W: Write>(w: &mut W, m: &Matrix) -> io::Result<()> {
    let rows = m.rows() as u32;
    let cols = m.cols() as u32;
    write_u32(w, rows)?;
    write_u32(w, cols)?;
    // Flattened row-major Annahme – entspricht as_slice()
    let flat = m.as_slice();
    debug_assert_eq!(flat.len(), (rows as usize) * (cols as usize));
    for &val in flat {
        write_f32(w, val)?;
    }
    Ok(())
}

fn read_matrix<R: Read>(r: &mut R) -> io::Result<Matrix> {
    let rows = read_u32(r)? as usize;
    let cols = read_u32(r)? as usize;
    let len = rows * cols;
    let mut data = Vec::with_capacity(len);
    for _ in 0..len {
        data.push(read_f32(r)?);
    }
    let m = Matrix::from_vec(data, rows, cols);
    Ok(m)
}

/* ---------- Activation <-> u8 Mapping ---------- */

fn write_activation<W: Write>(w: &mut W, a: &Activation) -> io::Result<()> {
    write_u8(w, activation_to_u8(a))
}
fn read_activation<R: Read>(r: &mut R) -> io::Result<Activation> {
    let code = read_u8(r)?;
    u8_to_activation(code)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Bad activation code"))
}

// ⚠️ PASSE DIESES MAPPING an deine tatsächlichen Activation-Varianten an!
fn activation_to_u8(a: &Activation) -> u8 {
    match a {
        Activation::Linear => 0,
        Activation::Relu => 1,
        Activation::Tanh => 2,
        Activation::Sigmoid => 3,
        Activation::Softmax => 4,
        // Activation::Gelu  => 5, // Beispiel – falls vorhanden
    }
}
fn u8_to_activation(code: u8) -> Option<Activation> {
    Some(match code {
        0 => Activation::Linear,
        1 => Activation::Relu,
        2 => Activation::Tanh,
        3 => Activation::Sigmoid,
        4 => Activation::Softmax,
        _ => return None,
    })
}
