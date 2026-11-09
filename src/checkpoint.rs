//! Checkpoint format v2.
//!
//! A header ([`Header`]: the [`ModelCfg`] widths, the [`Layout`] — every block's
//! kind as a tag plus arguments — the step count and the [`Seen`] counters), then every parameter tensor in the model's `param_slots` order, each
//! as `rank u32, dims u64 × rank, data f32 × len`. Everything is little-endian.
//!
//! The tensors are stored in the GPU model's layouts (fused projections, the
//! sLSTM's gate-concatenated weights), so a reader maps them by position. Nothing
//! is named: the config and layout fix the layer sequence, and the per-tensor shape
//! is checked against it on load.
//!
//! Reading and writing stream one tensor at a time, so the host never holds more
//! than the largest tensor.

use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Read, Write},
    path::Path,
};

use crate::tensor::Tensor;

pub const MAGIC: [u8; 4] = *b"HCK2";
pub const VERSION: u32 = 2;

/// The hierarchical model's widths and block counts. The kind of every block is
/// [`Layout`]'s; [`Layout::standard`] is the pattern a new model is built with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelCfg {
    pub vocab: usize,
    /// char/context hidden (tied embedding + decoder width)
    pub hc: usize,
    /// backbone width
    pub wh: usize,
    pub enc_blocks: usize,
    pub bb_blocks: usize,
    pub dec_blocks: usize,
    /// backbone mLSTM heads
    pub heads: usize,
    pub dqk: usize,
    pub w_token: usize,
    pub cap: f32,
}

impl ModelCfg {
    /// The model `config.rs` describes, for a new run.
    pub fn configured(vocab: usize, w_token: usize) -> Self {
        use crate::config::{CHAR_HIDDEN, LOGIT_SOFTCAP, WORD_BLOCKS, WORD_HIDDEN};
        let heads = 8;
        Self {
            vocab,
            hc: CHAR_HIDDEN,
            wh: WORD_HIDDEN,
            enc_blocks: 4,
            bb_blocks: WORD_BLOCKS,
            dec_blocks: 4,
            heads,
            dqk: WORD_HIDDEN / heads,
            w_token,
            cap: LOGIT_SOFTCAP,
        }
    }
}

/// mLSTM heads in the encoder and decoder; their `dqk` is `hc / CHAR_HEADS`.
pub const CHAR_HEADS: usize = 16;

/// One residual block's cell. Stored as a one-byte tag plus its arguments, so a new
/// cell is a new tag, not a new format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind {
    SLstm,
    MLstm { heads: usize, dqk: usize },
}

impl BlockKind {
    fn write(self, w: &mut impl Write) -> io::Result<()> {
        let (tag, args): (u8, &[usize]) = match self {
            BlockKind::SLstm => (0, &[]),
            BlockKind::MLstm { heads, dqk } => (1, &[heads, dqk]),
        };
        w.write_all(&[tag, args.len() as u8])?;
        for &a in args {
            put_u64(w, a as u64)?;
        }
        Ok(())
    }

    fn read(r: &mut impl Read) -> io::Result<Self> {
        let mut head = [0; 2];
        r.read_exact(&mut head)?;
        let [tag, n] = head;
        let args = (0..n).map(|_| get_usize(r)).collect::<io::Result<Vec<_>>>()?;
        match (tag, args.as_slice()) {
            (0, []) => Ok(BlockKind::SLstm),
            (1, &[heads, dqk]) => Ok(BlockKind::MLstm { heads, dqk }),
            _ => Err(invalid(format!("unknown block kind {tag} with {n} arguments"))),
        }
    }
}

/// The cell of every block, per stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub encoder: Vec<BlockKind>,
    pub backbone: Vec<BlockKind>,
    pub decoder: Vec<BlockKind>,
}

impl Layout {
    /// Encoder and decoder alternate sLSTM / mLSTM starting with sLSTM; the backbone
    /// is all mLSTM.
    pub fn standard(cfg: &ModelCfg) -> Self {
        let char_m = BlockKind::MLstm {
            heads: CHAR_HEADS,
            dqk: cfg.hc / CHAR_HEADS,
        };
        let alternating =
            |n: usize| (0..n).map(|i| if i % 2 == 0 { BlockKind::SLstm } else { char_m }).collect();
        Self {
            encoder: alternating(cfg.enc_blocks),
            backbone: vec![
                BlockKind::MLstm {
                    heads: cfg.heads,
                    dqk: cfg.dqk,
                };
                cfg.bb_blocks
            ],
            decoder: alternating(cfg.dec_blocks),
        }
    }

    /// Whether the model can run this layout under `cfg`: the block counts agree,
    /// and every mLSTM has the heads and `dqk` its stage's scratch is sized for.
    pub fn check(&self, cfg: &ModelCfg) -> Result<(), String> {
        let char_m = (CHAR_HEADS, cfg.hc / CHAR_HEADS);
        for (name, blocks, n, m) in [
            ("encoder", &self.encoder, cfg.enc_blocks, char_m),
            ("backbone", &self.backbone, cfg.bb_blocks, (cfg.heads, cfg.dqk)),
            ("decoder", &self.decoder, cfg.dec_blocks, char_m),
        ] {
            if blocks.len() != n {
                return Err(format!("{name}: {} blocks in the layout, {n} in the config", blocks.len()));
            }
            for (i, b) in blocks.iter().enumerate() {
                if let BlockKind::MLstm { heads, dqk } = *b
                    && (heads, dqk) != m
                {
                    return Err(format!(
                        "{name} block {i}: mLSTM with {heads} heads x {dqk}, the stage runs {} x {}",
                        m.0, m.1
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    pub cfg: ModelCfg,
    pub layout: Layout,
    pub step: u64,
    pub seen: Seen,
}

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

fn put_u32(w: &mut impl Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn put_u64(w: &mut impl Write, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn get_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn get_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn get_usize(r: &mut impl Read) -> io::Result<usize> {
    usize::try_from(get_u64(r)?).map_err(|_| invalid("size does not fit in usize".into()))
}

impl Header {
    fn write(&self, w: &mut impl Write, tensors: usize) -> io::Result<()> {
        let c = &self.cfg;
        w.write_all(&MAGIC)?;
        put_u32(w, VERSION)?;
        for v in [c.vocab, c.hc, c.wh, c.heads, c.dqk, c.w_token] {
            put_u64(w, v as u64)?;
        }
        put_u32(w, c.cap.to_bits())?;
        for stage in [&self.layout.encoder, &self.layout.backbone, &self.layout.decoder] {
            put_u64(w, stage.len() as u64)?;
            for b in stage {
                b.write(w)?;
            }
        }
        put_u64(w, self.step)?;
        let s = &self.seen;
        for v in [
            s.pretrain_chars, s.pretrain_words, s.sft_chars, s.sft_words, s.sft_resp_chars,
            s.sft_resp_words,
        ] {
            put_u64(w, v)?;
        }
        put_u64(w, tensors as u64)
    }

    /// The header and the tensor count that follows it.
    fn read(r: &mut impl Read) -> io::Result<(Self, usize)> {
        let mut magic = [0; 4];
        r.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(invalid(format!("not a v2 checkpoint (magic {magic:?})")));
        }
        let version = get_u32(r)?;
        if version != VERSION {
            return Err(invalid(format!("checkpoint version {version}, expected {VERSION}")));
        }
        let (vocab, hc, wh) = (get_usize(r)?, get_usize(r)?, get_usize(r)?);
        let (heads, dqk, w_token) = (get_usize(r)?, get_usize(r)?, get_usize(r)?);
        let cap = f32::from_bits(get_u32(r)?);
        let mut stage = || -> io::Result<Vec<BlockKind>> {
            let n = get_usize(r)?;
            (0..n).map(|_| BlockKind::read(r)).collect()
        };
        let layout = Layout {
            encoder: stage()?,
            backbone: stage()?,
            decoder: stage()?,
        };
        let cfg = ModelCfg {
            vocab,
            hc,
            wh,
            enc_blocks: layout.encoder.len(),
            bb_blocks: layout.backbone.len(),
            dec_blocks: layout.decoder.len(),
            heads,
            dqk,
            w_token,
            cap,
        };
        layout.check(&cfg).map_err(invalid)?;
        let step = get_u64(r)?;
        let seen = Seen {
            pretrain_chars: get_u64(r)?,
            pretrain_words: get_u64(r)?,
            sft_chars: get_u64(r)?,
            sft_words: get_u64(r)?,
            sft_resp_chars: get_u64(r)?,
            sft_resp_words: get_u64(r)?,
        };
        let tensors = get_usize(r)?;
        Ok((
            Self {
                cfg,
                layout,
                step,
                seen,
            },
            tensors,
        ))
    }
}

/// Whether `path` starts with the v2 magic. A file too short to hold one is not v2.
pub fn is_v2(path: impl AsRef<Path>) -> io::Result<bool> {
    let mut magic = [0; 4];
    match File::open(path)?.read_exact(&mut magic) {
        Ok(()) => Ok(magic == MAGIC),
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(false),
        Err(e) => Err(e),
    }
}

/// Elements converted per write or read call.
const CHUNK: usize = 1 << 18;

/// Writes a checkpoint to `<path>.tmp` and renames it over `path` in
/// [`finish`](Self::finish), so an interrupted save leaves the old file intact.
pub struct Writer {
    w: BufWriter<File>,
    tmp: String,
    path: String,
    left: usize,
    bytes: Vec<u8>,
}

impl Writer {
    pub fn create(path: &str, header: &Header, tensors: usize) -> io::Result<Self> {
        let tmp = format!("{path}.tmp");
        let mut w = BufWriter::new(File::create(&tmp)?);
        header.write(&mut w, tensors)?;
        Ok(Self {
            w,
            tmp,
            path: path.to_string(),
            left: tensors,
            bytes: Vec::with_capacity(CHUNK * 4),
        })
    }

    pub fn tensor(&mut self, dims: &[usize], data: &[f32]) -> io::Result<()> {
        if self.left == 0 {
            return Err(invalid("more tensors written than the header declares".into()));
        }
        assert_eq!(dims.iter().product::<usize>(), data.len(), "tensor shape/data mismatch");
        self.left -= 1;
        put_u32(&mut self.w, dims.len() as u32)?;
        for &d in dims {
            put_u64(&mut self.w, d as u64)?;
        }
        for chunk in data.chunks(CHUNK) {
            self.bytes.clear();
            self.bytes.extend(chunk.iter().flat_map(|x| x.to_le_bytes()));
            self.w.write_all(&self.bytes)?;
        }
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<()> {
        if self.left != 0 {
            return Err(invalid(format!("{} declared tensors were never written", self.left)));
        }
        self.w.flush()?;
        self.w.get_ref().sync_all()?;
        drop(self.w);
        std::fs::rename(&self.tmp, &self.path)
    }
}

pub struct Reader {
    r: BufReader<File>,
    pub header: Header,
    left: usize,
    bytes: Vec<u8>,
}

impl Reader {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let mut r = BufReader::new(File::open(path)?);
        let (header, left) = Header::read(&mut r)?;
        Ok(Self {
            r,
            header,
            left,
            bytes: Vec::new(),
        })
    }

    /// Tensors not read yet.
    pub fn remaining(&self) -> usize {
        self.left
    }

    /// The next tensor's shape, then its data into `out` (resized to fit).
    fn next_raw(&mut self, out: &mut Vec<f32>) -> io::Result<Vec<usize>> {
        if self.left == 0 {
            return Err(invalid("checkpoint has fewer tensors than the model".into()));
        }
        self.left -= 1;
        let rank = get_u32(&mut self.r)? as usize;
        if rank > crate::tensor::MAX_RANK {
            return Err(invalid(format!("tensor rank {rank}")));
        }
        let dims = (0..rank).map(|_| get_usize(&mut self.r)).collect::<io::Result<Vec<_>>>()?;
        let n: usize = dims.iter().product();
        out.clear();
        out.reserve(n);
        let mut done = 0;
        while done < n {
            let k = (n - done).min(CHUNK);
            self.bytes.resize(k * 4, 0);
            self.r.read_exact(&mut self.bytes)?;
            out.extend(self.bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())));
            done += k;
        }
        Ok(dims)
    }

    /// The next tensor into `out`, which must be shaped `dims`.
    pub fn next_into(&mut self, dims: &[usize], out: &mut Vec<f32>) -> io::Result<()> {
        let index = self.left;
        let got = self.next_raw(out)?;
        if got != dims {
            return Err(invalid(format!(
                "tensor {} from the end is {got:?} in the checkpoint, the model expects {dims:?}",
                index
            )));
        }
        Ok(())
    }

    /// The next tensor's shape, its data skipped.
    pub fn skip(&mut self) -> io::Result<Vec<usize>> {
        if self.left == 0 {
            return Err(invalid("no tensors left".into()));
        }
        self.left -= 1;
        let rank = get_u32(&mut self.r)? as usize;
        if rank > crate::tensor::MAX_RANK {
            return Err(invalid(format!("tensor rank {rank}")));
        }
        let dims = (0..rank).map(|_| get_usize(&mut self.r)).collect::<io::Result<Vec<_>>>()?;
        let bytes = dims.iter().product::<usize>() * 4;
        self.r.seek_relative(bytes as i64)?;
        Ok(dims)
    }

    pub fn next_tensor(&mut self) -> io::Result<Tensor> {
        let mut data = Vec::new();
        let dims = self.next_raw(&mut data)?;
        Ok(Tensor::new(&dims, data))
    }

    /// Errors unless every tensor was read and nothing follows the last one.
    pub fn finish(mut self) -> io::Result<()> {
        if self.left != 0 {
            return Err(invalid(format!("{} checkpoint tensors left unread", self.left)));
        }
        let mut probe = [0u8; 1];
        match self.r.read(&mut probe)? {
            0 => Ok(()),
            _ => Err(invalid("trailing bytes after the last tensor".into())),
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ModelCfg {
        ModelCfg {
            vocab: 9,
            hc: 16,
            wh: 24,
            enc_blocks: 1,
            bb_blocks: 2,
            dec_blocks: 1,
            heads: 2,
            dqk: 8,
            w_token: 8,
            cap: 30.0,
        }
    }

    /// Header and tensors come back bit-exact, NaN payloads and signed zeros included.
    #[test]
    fn round_trip_is_bit_exact() {
        let path = std::env::temp_dir().join("checkpoint_v2_round_trip");
        let path = path.to_str().unwrap();
        let mut seen = Seen::default();
        seen.add_pretrain(123, 45);
        seen.add_sft(10, 4, 6, 2);
        let mut layout = Layout::standard(&cfg());
        layout.backbone[1] = BlockKind::SLstm;
        let header = Header {
            cfg: cfg(),
            layout,
            step: 77,
            seen,
        };
        let a: Vec<f32> = (0..CHUNK + 5).map(|i| (i as f32).sin()).collect();
        let b = vec![-0.0, f32::from_bits(0x7fc0_1234), f32::MIN_POSITIVE / 2.0];

        let mut w = Writer::create(path, &header, 2).unwrap();
        w.tensor(&[CHUNK + 5], &a).unwrap();
        w.tensor(&[1, 3], &b).unwrap();
        w.finish().unwrap();

        assert!(is_v2(path).unwrap());
        let mut r = Reader::open(path).unwrap();
        assert_eq!(r.header, header);
        assert_eq!(r.remaining(), 2);
        let mut buf = Vec::new();
        r.next_into(&[CHUNK + 5], &mut buf).unwrap();
        assert!(buf.iter().zip(&a).all(|(x, y)| x.to_bits() == y.to_bits()));
        let t = r.next_tensor().unwrap();
        assert_eq!(t.dims(), &[1, 3]);
        assert!(t.data.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
        r.finish().unwrap();
        let _ = std::fs::remove_file(path);
    }

    /// A layout that disagrees with its config, or a tag this build does not know,
    /// is refused in the header rather than surfacing as a tensor-shape error.
    #[test]
    fn bad_layouts_are_refused() {
        let c = cfg();
        let mut layout = Layout::standard(&c);
        assert!(layout.check(&c).is_ok());
        layout.backbone[0] = BlockKind::MLstm { heads: 4, dqk: 6 };
        assert!(layout.check(&c).is_err());
        layout.backbone.pop();
        assert!(layout.check(&c).is_err());

        let mut bytes = Vec::new();
        bytes.extend([7u8, 0]);
        assert!(BlockKind::read(&mut bytes.as_slice()).is_err());
        bytes.clear();
        BlockKind::MLstm { heads: 3, dqk: 5 }.write(&mut bytes).unwrap();
        assert_eq!(
            BlockKind::read(&mut bytes.as_slice()).unwrap(),
            BlockKind::MLstm { heads: 3, dqk: 5 }
        );
    }

    #[test]
    fn shape_mismatch_and_short_files_are_errors() {
        let path = std::env::temp_dir().join("checkpoint_v2_mismatch");
        let path = path.to_str().unwrap();
        let header = Header {
            cfg: cfg(),
            layout: Layout::standard(&cfg()),
            step: 0,
            seen: Seen::default(),
        };
        let mut w = Writer::create(path, &header, 2).unwrap();
        w.tensor(&[2, 3], &[0.0; 6]).unwrap();
        assert!(w.finish().is_err(), "a missing declared tensor must fail the save");

        let mut w = Writer::create(path, &header, 1).unwrap();
        w.tensor(&[2, 3], &[0.0; 6]).unwrap();
        w.finish().unwrap();
        let mut r = Reader::open(path).unwrap();
        assert!(r.next_into(&[3, 2], &mut Vec::new()).is_err());
        let r = Reader::open(path).unwrap();
        assert!(r.finish().is_err(), "unread tensors must be reported");
        let _ = std::fs::remove_file(path);
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
}
