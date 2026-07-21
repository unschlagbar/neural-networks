// Unified on-disk model format ("NNM1").
//
// One container for every model this crate saves. Replaces the old per-model
// magics (NNFW flat / HIE4 / HIE5 hierarchical): there is a single magic, a
// single version, and a `ModelKind` tag, followed by a small typed metadata
// head and a count-prefixed list of NAMED SECTIONS. Each section is a name
// string plus one layer-stack blob (arch header + weights, via
// `saving::write_layers`). A single standalone layer is just a one-layer
// stack, so every section is uniform.
//
// Layout
// ┌──────────────────────────────────────────────────────────┐
// │ MAGIC     u32   0x4E4E_4D31  ("NNM1")                     │
// │ VERSION   u8                                              │
// │ KIND      u8    0 = Flat, 1 = Hierarchical                │
// ├──────────────────────────────────────────────────────────┤ ← typed head
// │ (Flat)          — no head                                 │
// │ (Hierarchical)  vocab u32, context u32, step u64          │
// ├──────────────────────────────────────────────────────────┤ ← sections
// │ N_SECTIONS  u32                                           │
// │ for each: name <string>, stack <layer blob>              │
// └──────────────────────────────────────────────────────────┘
//
// The per-layer codecs (matrices, gates, block layouts) live in `saving.rs` /
// `loading.rs` and are reused unchanged — this module only owns the container.

use std::{
    fs::File, io::{self, BufReader, Cursor, Read, Write},
};

use crate::{
    loading::{load_layers, read_string, read_u8, read_u32, read_u64},
    nn_layer::NnLayer,
    saving::{write_layers, write_string, write_u8, write_u32, write_u64},
};

/// "NNM1" — the one and only container magic.
pub const MAGIC: u32 = 0x4E4E_4D31;
/// Container version. Bump when the container framing (not a layer codec) changes.
pub const VERSION: u8 = 1;

/// What kind of model a container holds. The tag is stored so a single reader
/// (and `inspect`) can dispatch without guessing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelKind {
    /// A flat `Sequential`.
    Flat,
    /// The hierarchical (HAT) model: encoder + backbone + decoder.
    Hierarchical,
}

impl ModelKind {
    fn tag(self) -> u8 {
        match self {
            ModelKind::Flat => 0,
            ModelKind::Hierarchical => 1,
        }
    }

    fn from_tag(tag: u8) -> io::Result<Self> {
        match tag {
            0 => Ok(ModelKind::Flat),
            1 => Ok(ModelKind::Hierarchical),
            o => Err(invalid(format!("unknown model kind tag {o}"))),
        }
    }
}

/// Metadata carried alongside the sections. Flat models carry none; the
/// hierarchical model carries what `Hierarchical::new` cannot recompute.
#[derive(Clone, Copy, Debug, Default)]
pub struct Meta {
    pub vocab_size: u32,
    pub context_size: u32,
    pub step: u64,
}

/// One named model stage to write: a stable name plus its layer stack. A stage
/// may be a full `Sequential`'s layers or a single standalone layer, so it holds
/// a layer slice either way.
pub struct Section<'a> {
    pub name: &'a str,
    pub layers: SectionLayers<'a>,
}

/// The layers of a section: either a borrowed slice (a `Sequential`) or a single
/// standalone layer that is not stored inside a `Sequential`.
pub enum SectionLayers<'a> {
    Slice(&'a [Box<dyn NnLayer>]),
    One(&'a dyn NnLayer),
}

/// A model container ready to write: kind + metadata + ordered named sections.
pub struct Writer<'a> {
    pub kind: ModelKind,
    pub meta: Meta,
    pub sections: Vec<Section<'a>>,
}

impl<'a> Writer<'a> {
    pub fn new(kind: ModelKind, meta: Meta) -> Self {
        Self {
            kind,
            meta,
            sections: Vec::new(),
        }
    }

    /// Append a named section from a layer stack (e.g. a `Sequential`'s layers).
    /// Section order is preserved and is what the reader sees.
    pub fn section(mut self, name: &'a str, layers: &'a [Box<dyn NnLayer>]) -> Self {
        self.sections.push(Section {
            name,
            layers: SectionLayers::Slice(layers),
        });
        self
    }

    /// Append a named section holding a single standalone layer.
    pub fn section_layer(mut self, name: &'a str, layer: &'a dyn NnLayer) -> Self {
        self.sections.push(Section {
            name,
            layers: SectionLayers::One(layer),
        });
        self
    }

    /// Serialize the container to any writer.
    pub fn write_to(&self, w: &mut dyn Write) -> io::Result<()> {
        write_u32(w, MAGIC)?;
        write_u8(w, VERSION)?;
        write_u8(w, self.kind.tag())?;

        // Typed head.
        if self.kind == ModelKind::Hierarchical {
            write_u32(w, self.meta.vocab_size)?;
            write_u32(w, self.meta.context_size)?;
            write_u64(w, self.meta.step)?;
        }

        // Named sections.
        write_u32(w, self.sections.len() as u32)?;
        for s in &self.sections {
            write_string(w, s.name)?;
            match s.layers {
                SectionLayers::Slice(layers) => write_layers(w, layers)?,
                SectionLayers::One(layer) => crate::saving::write_one_layer(w, layer)?,
            }
        }
        Ok(())
    }

    /// Serialize the container to a file (atomically: temp file then rename).
    pub fn save(&self, path: &str) -> io::Result<()> {
        if let Some(dir) = std::path::Path::new(path).parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir)?;
            }
        }
        let mut buf = Cursor::new(Vec::<u8>::new());
        self.write_to(&mut buf)?;
        let tmp = format!("{path}.tmp");
        File::create(&tmp)?.write_all(&buf.into_inner())?;
        std::fs::rename(&tmp, path)
    }
}

/// A model container read back from disk: kind, metadata, and its named layer
/// stacks in file order. Callers pull sections by name via [`take`](Self::take).
pub struct Reader {
    pub kind: ModelKind,
    pub meta: Meta,
    /// (name, layers) in file order.
    pub sections: Vec<(String, Vec<Box<dyn NnLayer>>)>,
}

impl Reader {
    /// Peek at just the header (magic, version, kind) without decoding sections.
    /// Used by `inspect` to label a file before deciding how to read it.
    pub fn peek_kind(path: &str) -> io::Result<ModelKind> {
        let r = &mut File::open(path)? as &mut dyn Read;
        read_header(r)
    }

    /// Read a whole container from any reader.
    pub fn read_from(r: &mut dyn Read) -> io::Result<Self> {
        let kind = read_header(r)?;

        let meta = if kind == ModelKind::Hierarchical {
            Meta {
                vocab_size: read_u32(r)?,
                context_size: read_u32(r)?,
                step: read_u64(r)?,
            }
        } else {
            Meta::default()
        };

        let n = read_u32(r)? as usize;
        let mut sections = Vec::with_capacity(n);
        for _ in 0..n {
            let name = read_string(r)?;
            let layers = load_layers(r)?;
            sections.push((name, layers));
        }

        Ok(Self {
            kind,
            meta,
            sections,
        })
    }

    /// Read a whole container from a file path.
    pub fn load(path: &str) -> io::Result<Self> {
        Self::read_from(&mut BufReader::new(File::open(path)?))
    }

    /// Remove and return the layers of section `name`. Section order is fixed by
    /// construction (writers append in a stable order), so this is a small
    /// linear scan. Errors if the section is missing.
    pub fn take(&mut self, name: &str) -> io::Result<Vec<Box<dyn NnLayer>>> {
        let idx = self
            .sections
            .iter()
            .position(|(n, _)| n == name)
            .ok_or_else(|| invalid(format!("missing section {name:?}")))?;
        Ok(self.sections.remove(idx).1)
    }

    /// Like [`take`](Self::take) but wraps the section as a `Sequential`.
    pub fn take_stack(&mut self, name: &str) -> io::Result<crate::sequential::Sequential> {
        Ok(crate::sequential::Sequential::from_layers(self.take(name)?))
    }
}

/// Read and validate the fixed header (magic + version), returning the kind.
fn read_header(r: &mut dyn Read) -> io::Result<ModelKind> {
    if read_u32(r)? != MAGIC {
        return Err(invalid("not an NNM1 model file (wrong magic)".into()));
    }
    if read_u8(r)? != VERSION {
        return Err(invalid("unsupported NNM1 version".into()));
    }
    ModelKind::from_tag(read_u8(r)?)
}

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::linear::LinearLayer;
    use crate::nn_layer::SequentialBuilder;

    /// A flat container round-trips: same kind, one "model" section, weights
    /// preserved (checked via a re-save producing identical bytes).
    #[test]
    fn flat_container_roundtrips() {
        let model = SequentialBuilder::new(8).embedding(16).rms_norm().linear(4).build();
        let mut buf = Cursor::new(Vec::new());
        Writer::new(ModelKind::Flat, Meta::default())
            .section("model", &model.layers)
            .write_to(&mut buf)
            .unwrap();
        let bytes = buf.into_inner();

        let mut reader = Reader::read_from(&mut Cursor::new(bytes.as_slice())).unwrap();
        assert_eq!(reader.kind, ModelKind::Flat);
        assert_eq!(reader.sections.len(), 1);
        let back = reader.take_stack("model").unwrap();

        // Re-serialize the reloaded stack; the section bytes must match.
        let mut buf2 = Cursor::new(Vec::new());
        Writer::new(ModelKind::Flat, Meta::default())
            .section("model", &back.layers)
            .write_to(&mut buf2)
            .unwrap();
        assert_eq!(bytes, buf2.into_inner());
    }

    /// A hierarchical container preserves metadata and every named section,
    /// including one written from a single standalone layer.
    #[test]
    fn hierarchical_container_roundtrips() {
        let fwd = SequentialBuilder::new(8).embedding(16).rms_norm().build();
        let combine: Box<dyn NnLayer> = Box::new(LinearLayer::new(32, 16));
        let wm = SequentialBuilder::new(16).rms_norm().linear(16).build();
        let dec = SequentialBuilder::new(16).rms_norm().linear(8).build();

        let meta = Meta {
            vocab_size: 8,
            context_size: 16,
            step: 4242,
        };
        let mut buf = Cursor::new(Vec::new());
        Writer::new(ModelKind::Hierarchical, meta)
            .section("encoder", &fwd.layers)
            .section_layer("extra", &*combine)
            .section("word_model", &wm.layers)
            .section("char2_model", &dec.layers)
            .write_to(&mut buf)
            .unwrap();

        let mut reader = Reader::read_from(&mut Cursor::new(buf.into_inner())).unwrap();
        assert_eq!(reader.kind, ModelKind::Hierarchical);
        assert_eq!(reader.meta.vocab_size, 8);
        assert_eq!(reader.meta.context_size, 16);
        assert_eq!(reader.meta.step, 4242);

        // Every section present and pullable by name.
        for name in ["encoder", "word_model", "char2_model"] {
            assert!(reader.take_stack(name).is_ok(), "missing {name}");
        }
        let one = reader.take("extra").unwrap();
        assert_eq!(one.len(), 1, "single-layer section round-trips as one layer");
        assert!(reader.take("nope").is_err(), "missing section must error");
    }

    /// The header carries the kind, and reading with the wrong expectation is a
    /// clean error at the call site (peek_kind never mislabels).
    #[test]
    fn kind_tag_is_readable_without_sections() {
        let model = SequentialBuilder::new(4).linear(2).build();
        let mut buf = Cursor::new(Vec::new());
        Writer::new(ModelKind::Flat, Meta::default())
            .section("model", &model.layers)
            .write_to(&mut buf)
            .unwrap();
        let bytes = buf.into_inner();
        let kind = read_header(&mut Cursor::new(bytes.as_slice())).unwrap();
        assert_eq!(kind, ModelKind::Flat);
    }
}
