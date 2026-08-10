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
// │                 + since v2: pretrain chars/words u64,     │
// │                             sft chars/words u64           │
// ├──────────────────────────────────────────────────────────┤ ← sections
// │ N_SECTIONS  u32                                           │
// │ for each: name <string>, stack <layer blob>              │
// └──────────────────────────────────────────────────────────┘
//
// The per-layer codecs (matrices, gates, block layouts) live in `saving.rs` /
// `loading.rs` and are reused unchanged — this module only owns the container.

use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Read, Write},
};

use crate::{
    loading::{load_layers, read_string, read_u8, read_u32, read_u64},
    nn_layer::NnLayer,
    saving::{write_layers, write_string, write_u8, write_u32, write_u64},
};

/// "NNM1" — the one and only container magic.
pub const MAGIC: u32 = 0x4E4E_4D31;
/// Container version. Bump when the container framing (not a layer codec) changes.
/// v1 files still load; their hierarchical head simply stops after `step` and the
/// data counters read back as zero.
pub const VERSION: u8 = 2;
/// Oldest container version this reader accepts.
pub const MIN_VERSION: u8 = 1;

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
    /// How much data the weights have actually seen, split by training phase.
    pub seen: Seen,
}

/// Cumulative data the checkpoint has been trained on, counted separately for
/// pretraining and SFT. A "char" is one tokenizer token of the training window
/// (byte-level, so a UTF-8 char can be several); a "word" is one
/// `segment::word_ends` unit. Both count every window fed to the model,
/// including re-visits across epochs — this is exposure, not corpus size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Seen {
    pub pretrain_chars: u64,
    pub pretrain_words: u64,
    /// SFT counts the whole formatted example, prompt included (the backbone
    /// reads all of it); [`sft_resp_chars`](Self::sft_resp_chars) is the
    /// loss-carrying subset.
    pub sft_chars: u64,
    pub sft_words: u64,
    /// The masked-in part of the SFT counts: tokens/words the loss came from.
    pub sft_resp_chars: u64,
    pub sft_resp_words: u64,
}

impl Seen {
    /// Add a pretraining window.
    pub fn add_pretrain(&mut self, chars: usize, words: usize) {
        self.pretrain_chars += chars as u64;
        self.pretrain_words += words as u64;
    }

    /// Add one SFT example: the full window plus the response-only subset.
    pub fn add_sft(&mut self, chars: usize, words: usize, resp_chars: usize, resp_words: usize) {
        self.sft_chars += chars as u64;
        self.sft_words += words as u64;
        self.sft_resp_chars += resp_chars as u64;
        self.sft_resp_words += resp_words as u64;
    }

    /// Whether this checkpoint has been fine-tuned at all.
    pub fn has_sft(&self) -> bool {
        self.sft_chars > 0 || self.sft_words > 0
    }

    /// Multi-line human report of both phases, one indented line per figure.
    /// Ends with a newline, so it prints with `print!`.
    pub fn report(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "  pretrain : {} chars, {} words\n",
            group(self.pretrain_chars),
            group(self.pretrain_words),
        ));
        if self.has_sft() {
            s.push_str(&format!(
                "  sft      : {} chars, {} words  (response only: {} chars, {} words)\n",
                group(self.sft_chars),
                group(self.sft_words),
                group(self.sft_resp_chars),
                group(self.sft_resp_words),
            ));
        } else {
            s.push_str("  sft      : none\n");
        }
        s.push_str(&format!(
            "  total    : {} chars, {} words\n",
            group(self.pretrain_chars + self.sft_chars),
            group(self.pretrain_words + self.sft_words),
        ));
        s
    }

    /// One-line summary for a training banner.
    pub fn summary(&self) -> String {
        format!(
            "pretrain {} chars / {} words, sft {} chars / {} words",
            group(self.pretrain_chars),
            group(self.pretrain_words),
            group(self.sft_chars),
            group(self.sft_words),
        )
    }

    /// Compact one-liner printed next to every checkpoint save. Shows only the
    /// phase that has data, and both phases once the model has been fine-tuned.
    pub fn save_line(&self) -> String {
        let pre = format!(
            "pretrain {} chars / {} words",
            compact(self.pretrain_chars),
            compact(self.pretrain_words),
        );
        if !self.has_sft() {
            return pre;
        }
        format!(
            "{pre} | sft {} chars / {} words (resp {} / {})",
            compact(self.sft_chars),
            compact(self.sft_words),
            compact(self.sft_resp_chars),
            compact(self.sft_resp_words),
        )
    }
}

/// Compact magnitude: `12_500_000` → `12.5m`. Three significant digits at most,
/// and a trailing `.0` is dropped (`2000` → `2k`, not `2.0k`).
pub fn compact(n: u64) -> String {
    const UNITS: [(u64, char); 4] = [
        (1_000_000_000_000, 't'),
        (1_000_000_000, 'b'),
        (1_000_000, 'm'),
        (1_000, 'k'),
    ];
    // Largest unit first, so 999_999 rounds into "1m" rather than "1000k".
    for (scale, suffix) in UNITS {
        // Round at this unit's precision before comparing, so a value just under
        // the boundary (999_999) is caught by the unit it rounds *into*.
        let v = n as f64 / scale as f64;
        if v < 0.9995 {
            continue;
        }
        // Keep the number three digits wide: 12.5m, 125m, 1.25b.
        let decimals = if v < 9.9995 {
            2
        } else if v < 99.995 {
            1
        } else {
            0
        };
        let mut s = format!("{v:.decimals$}");
        if s.contains('.') {
            s = s.trim_end_matches('0').trim_end_matches('.').to_string();
        }
        s.push(suffix);
        return s;
    }
    n.to_string()
}

/// Group a count into thousands (`1234567` → `1_234_567`) — these run to the
/// billions and are unreadable otherwise.
fn group(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push('_');
        }
        out.push(c);
    }
    out
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
            let s = &self.meta.seen;
            write_u64(w, s.pretrain_chars)?;
            write_u64(w, s.pretrain_words)?;
            write_u64(w, s.sft_chars)?;
            write_u64(w, s.sft_words)?;
            write_u64(w, s.sft_resp_chars)?;
            write_u64(w, s.sft_resp_words)?;
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
        let tmp = format!("{path}.tmp");
        let mut model = BufWriter::new(File::create(&tmp)?);
        self.write_to(&mut model)?;
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
        read_header(r).map(|(_, kind)| kind)
    }

    /// Read a whole container from any reader.
    pub fn read_from(r: &mut dyn Read) -> io::Result<Self> {
        let (version, kind) = read_header(r)?;

        let meta = if kind == ModelKind::Hierarchical {
            let mut meta = Meta {
                vocab_size: read_u32(r)?,
                context_size: read_u32(r)?,
                step: read_u64(r)?,
                seen: Seen::default(),
            };
            // The data counters were added in v2; a v1 head ends at `step` and
            // leaves them zero.
            if version >= 2 {
                meta.seen = Seen {
                    pretrain_chars: read_u64(r)?,
                    pretrain_words: read_u64(r)?,
                    sft_chars: read_u64(r)?,
                    sft_words: read_u64(r)?,
                    sft_resp_chars: read_u64(r)?,
                    sft_resp_words: read_u64(r)?,
                };
            }
            meta
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

/// Read and validate the fixed header (magic + version), returning the file's
/// version and kind. Older versions in `MIN_VERSION..=VERSION` are accepted; the
/// caller uses the version to decide which head fields are present.
fn read_header(r: &mut dyn Read) -> io::Result<(u8, ModelKind)> {
    if read_u32(r)? != MAGIC {
        return Err(invalid("not an NNM1 model file (wrong magic)".into()));
    }
    let version = read_u8(r)?;
    if !(MIN_VERSION..=VERSION).contains(&version) {
        return Err(invalid(format!(
            "unsupported NNM1 version {version} (this build reads {MIN_VERSION}..={VERSION})"
        )));
    }
    Ok((version, ModelKind::from_tag(read_u8(r)?)?))
}

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::linear::LinearLayer;
    use crate::nn_layer::SequentialBuilder;
    use std::io::Cursor;

    /// A flat container round-trips: same kind, one "model" section, weights
    /// preserved (checked via a re-save producing identical bytes).
    #[test]
    fn flat_container_roundtrips() {
        let model = SequentialBuilder::new(8)
            .embedding(16)
            .rms_norm()
            .linear(4)
            .build();
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
            seen: Seen {
                pretrain_chars: 12_345,
                pretrain_words: 3_456,
                sft_chars: 789,
                sft_words: 210,
                sft_resp_chars: 456,
                sft_resp_words: 120,
            },
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
        assert_eq!(reader.meta.seen, meta.seen, "data counters lost");

        // Every section present and pullable by name.
        for name in ["encoder", "word_model", "char2_model"] {
            assert!(reader.take_stack(name).is_ok(), "missing {name}");
        }
        let one = reader.take("extra").unwrap();
        assert_eq!(
            one.len(),
            1,
            "single-layer section round-trips as one layer"
        );
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
        let (version, kind) = read_header(&mut Cursor::new(bytes.as_slice())).unwrap();
        assert_eq!(version, VERSION);
        assert_eq!(kind, ModelKind::Flat);
    }

    /// Compact magnitudes stay three digits wide and never keep a dead `.0`.
    #[test]
    fn compact_magnitudes() {
        assert_eq!(compact(0), "0");
        assert_eq!(compact(999), "999");
        assert_eq!(compact(1_000), "1k");
        assert_eq!(compact(12_500), "12.5k");
        // Rounding at a unit boundary steps up rather than printing "1000k".
        assert_eq!(compact(999_999), "1m");
        assert_eq!(compact(999_499), "999k");
        assert_eq!(compact(1_000_000), "1m");
        assert_eq!(compact(12_500_000), "12.5m");
        assert_eq!(compact(125_000_000), "125m");
        assert_eq!(compact(1_250_000_000), "1.25b");
        assert_eq!(compact(3_000_000_000_000), "3t");
    }

    /// The save line names both phases only once the model has SFT data.
    #[test]
    fn save_line_shows_sft_only_when_present() {
        let mut seen = Seen::default();
        seen.add_pretrain(12_500_000, 3_400_000);
        assert_eq!(seen.save_line(), "pretrain 12.5m chars / 3.4m words");

        seen.add_sft(2_000_000, 500_000, 750_000, 200_000);
        assert_eq!(
            seen.save_line(),
            "pretrain 12.5m chars / 3.4m words | sft 2m chars / 500k words (resp 750k / 200k)"
        );
    }

    /// A v1 hierarchical head stops after `step`; it must still load, with the
    /// data counters reading back as zero rather than eating section bytes.
    #[test]
    fn v1_header_loads_with_zero_counters() {
        let fwd = SequentialBuilder::new(8).embedding(16).build();
        let wm = SequentialBuilder::new(16).linear(16).build();
        let dec = SequentialBuilder::new(16).linear(8).build();

        // Hand-build a v1 container: same framing, short head, no counters.
        let mut buf = Cursor::new(Vec::new());
        {
            let w = &mut buf as &mut dyn Write;
            write_u32(w, MAGIC).unwrap();
            write_u8(w, 1).unwrap();
            write_u8(w, ModelKind::Hierarchical.tag()).unwrap();
            write_u32(w, 8).unwrap();
            write_u32(w, 16).unwrap();
            write_u64(w, 99).unwrap();
            write_u32(w, 3).unwrap();
            for (name, layers) in [
                ("encoder", &fwd.layers),
                ("word_model", &wm.layers),
                ("char2_model", &dec.layers),
            ] {
                write_string(w, name).unwrap();
                write_layers(w, layers).unwrap();
            }
        }

        let mut reader = Reader::read_from(&mut Cursor::new(buf.into_inner())).unwrap();
        assert_eq!(reader.meta.step, 99);
        assert_eq!(reader.meta.seen, Seen::default());
        // Sections still parse — proof the head length was read correctly.
        for name in ["encoder", "word_model", "char2_model"] {
            assert!(reader.take_stack(name).is_ok(), "missing {name}");
        }
    }
}
