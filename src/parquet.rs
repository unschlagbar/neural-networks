// Minimal parquet reader — just enough to stream one BYTE_ARRAY column out of a
// corpus file, one row group at a time.
//
// This is deliberately not a general parquet implementation. The training
// corpora we consume (FineWeb-style dumps: a `text` column plus metadata) use a
// narrow slice of the format, and supporting exactly that slice costs a few
// hundred lines with no dependencies — where pulling in the `parquet` crate
// would drag arrow + thrift + a large tree into a project that has almost none.
//
// What is supported:
//   - flat (non-nested) schema, one or more requested columns, BYTE_ARRAY type
//   - OPTIONAL or REQUIRED repetition (max definition level 0 or 1); nulls are
//     skipped, not surfaced
//   - UNCOMPRESSED and SNAPPY page compression
//   - data page v1 and v2, PLAIN and dictionary (PLAIN_DICTIONARY / RLE_DICT)
//     value encodings, RLE definition levels
//
// Anything outside that returns an `Err` naming what it hit, rather than
// silently producing wrong text.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
};

pub type Result<T> = std::result::Result<T, String>;

// ── Thrift compact protocol ────────────────────────────────────────────────
//
// Parquet metadata is thrift-compact-encoded. We only ever read structs whose
// fields we either recognise or skip, so a reader that can decode the primitive
// types and skip anything else is sufficient.

const T_BOOL_TRUE: u8 = 1;
const T_BOOL_FALSE: u8 = 2;
const T_I8: u8 = 3;
const T_I16: u8 = 4;
const T_I32: u8 = 5;
const T_I64: u8 = 6;
const T_DOUBLE: u8 = 7;
const T_BINARY: u8 = 8;
const T_LIST: u8 = 9;
const T_SET: u8 = 10;
const T_MAP: u8 = 11;
const T_STRUCT: u8 = 12;

struct Thrift<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Thrift<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn byte(&mut self) -> Result<u8> {
        let b = *self
            .buf
            .get(self.pos)
            .ok_or_else(|| "truncated thrift metadata".to_string())?;
        self.pos += 1;
        Ok(b)
    }

    fn varint(&mut self) -> Result<u64> {
        let mut out = 0u64;
        let mut shift = 0;
        loop {
            let b = self.byte()?;
            out |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(out);
            }
            shift += 7;
            if shift > 63 {
                return Err("varint too long".into());
            }
        }
    }

    fn zigzag(&mut self) -> Result<i64> {
        let n = self.varint()?;
        Ok(((n >> 1) as i64) ^ -((n & 1) as i64))
    }

    fn bytes(&mut self) -> Result<&'a [u8]> {
        let n = self.varint()? as usize;
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| "truncated thrift string".to_string())?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    /// List/set header: `(element_count, element_type)`.
    fn list_header(&mut self) -> Result<(usize, u8)> {
        let h = self.byte()?;
        let ty = h & 0x0f;
        let mut size = (h >> 4) as usize;
        if size == 15 {
            size = self.varint()? as usize;
        }
        Ok((size, ty))
    }

    /// Next field header inside a struct. Returns `None` at the stop field.
    /// `last_id` carries the running field id for delta-encoded headers.
    fn field(&mut self, last_id: &mut i16) -> Result<Option<(i16, u8)>> {
        let h = self.byte()?;
        if h == 0 {
            return Ok(None);
        }
        let ty = h & 0x0f;
        let delta = (h >> 4) as i16;
        let id = if delta == 0 {
            self.zigzag()? as i16
        } else {
            *last_id + delta
        };
        *last_id = id;
        Ok(Some((id, ty)))
    }

    fn skip(&mut self, ty: u8) -> Result<()> {
        match ty {
            T_BOOL_TRUE | T_BOOL_FALSE => {}
            T_I8 => {
                self.byte()?;
            }
            T_I16 | T_I32 | T_I64 => {
                self.zigzag()?;
            }
            T_DOUBLE => self.pos += 8,
            T_BINARY => {
                self.bytes()?;
            }
            T_LIST | T_SET => {
                let (n, et) = self.list_header()?;
                for _ in 0..n {
                    self.skip(et)?;
                }
            }
            T_MAP => {
                let n = self.varint()? as usize;
                if n > 0 {
                    let kv = self.byte()?;
                    let (kt, vt) = (kv >> 4, kv & 0x0f);
                    for _ in 0..n {
                        self.skip(kt)?;
                        self.skip(vt)?;
                    }
                }
            }
            T_STRUCT => self.skip_struct()?,
            other => return Err(format!("unknown thrift type {other}")),
        }
        Ok(())
    }

    fn skip_struct(&mut self) -> Result<()> {
        let mut id = 0;
        while let Some((_, ty)) = self.field(&mut id)? {
            self.skip(ty)?;
        }
        Ok(())
    }
}

// ── Snappy (raw block format) ──────────────────────────────────────────────

/// Decompress a snappy raw block. Parquet stores pages in the raw (non-framed)
/// format, so the input starts with the varint uncompressed length.
fn snappy_decompress(src: &[u8]) -> Result<Vec<u8>> {
    let mut pos = 0;

    // Preamble: uncompressed length as a varint.
    let mut expect = 0usize;
    let mut shift = 0;
    loop {
        let b = *src
            .get(pos)
            .ok_or_else(|| "snappy: truncated length".to_string())?;
        pos += 1;
        expect |= ((b & 0x7f) as usize) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift > 35 {
            return Err("snappy: bad length varint".into());
        }
    }

    let mut out: Vec<u8> = Vec::with_capacity(expect);

    while pos < src.len() {
        let tag = src[pos];
        pos += 1;
        match tag & 0x03 {
            // Literal: length is either inline (tag >> 2) or in the next 1..4 bytes.
            0 => {
                let mut len = (tag >> 2) as usize;
                if len >= 60 {
                    let extra = len - 59;
                    if pos + extra > src.len() {
                        return Err("snappy: truncated literal length".into());
                    }
                    let mut v = 0usize;
                    for i in 0..extra {
                        v |= (src[pos + i] as usize) << (8 * i);
                    }
                    pos += extra;
                    len = v;
                }
                len += 1;
                if pos + len > src.len() {
                    return Err("snappy: truncated literal".into());
                }
                out.extend_from_slice(&src[pos..pos + len]);
                pos += len;
            }
            // Copy with a 1-byte offset extension (length 4..11).
            1 => {
                if pos >= src.len() {
                    return Err("snappy: truncated copy1".into());
                }
                let len = 4 + ((tag >> 2) & 0x07) as usize;
                let off = (((tag >> 5) as usize) << 8) | src[pos] as usize;
                pos += 1;
                copy_from_back(&mut out, off, len)?;
            }
            // Copy with a 2-byte offset.
            2 => {
                if pos + 2 > src.len() {
                    return Err("snappy: truncated copy2".into());
                }
                let len = ((tag >> 2) as usize) + 1;
                let off = u16::from_le_bytes([src[pos], src[pos + 1]]) as usize;
                pos += 2;
                copy_from_back(&mut out, off, len)?;
            }
            // Copy with a 4-byte offset.
            _ => {
                if pos + 4 > src.len() {
                    return Err("snappy: truncated copy4".into());
                }
                let len = ((tag >> 2) as usize) + 1;
                let off = u32::from_le_bytes([src[pos], src[pos + 1], src[pos + 2], src[pos + 3]])
                    as usize;
                pos += 4;
                copy_from_back(&mut out, off, len)?;
            }
        }
    }

    if out.len() != expect {
        return Err(format!(
            "snappy: decompressed {} bytes, header said {expect}",
            out.len()
        ));
    }
    Ok(out)
}

/// Back-reference copy. Overlapping copies are legal and common (that is how
/// snappy encodes runs), so this copies byte by byte rather than by slice.
fn copy_from_back(out: &mut Vec<u8>, offset: usize, len: usize) -> Result<()> {
    if offset == 0 || offset > out.len() {
        return Err("snappy: copy offset out of range".into());
    }
    let start = out.len() - offset;
    for i in 0..len {
        let b = out[start + i];
        out.push(b);
    }
    Ok(())
}

// ── RLE / bit-packed hybrid ────────────────────────────────────────────────

/// Decoder for parquet's RLE/bit-packing hybrid, used for definition levels and
/// for dictionary indices.
struct RleDecoder<'a> {
    buf: &'a [u8],
    pos: usize,
    bit_width: u8,
    /// Values buffered from the current run.
    run: Vec<u32>,
    run_pos: usize,
}

impl<'a> RleDecoder<'a> {
    fn new(buf: &'a [u8], bit_width: u8) -> Self {
        Self {
            buf,
            pos: 0,
            bit_width,
            run: Vec::new(),
            run_pos: 0,
        }
    }

    fn varint(&mut self) -> Result<u32> {
        let mut out = 0u32;
        let mut shift = 0;
        loop {
            let b = *self
                .buf
                .get(self.pos)
                .ok_or_else(|| "rle: truncated varint".to_string())?;
            self.pos += 1;
            out |= ((b & 0x7f) as u32) << shift;
            if b & 0x80 == 0 {
                return Ok(out);
            }
            shift += 7;
            if shift > 31 {
                return Err("rle: varint too long".into());
            }
        }
    }

    fn next(&mut self) -> Result<Option<u32>> {
        if self.run_pos < self.run.len() {
            let v = self.run[self.run_pos];
            self.run_pos += 1;
            return Ok(Some(v));
        }
        if self.bit_width == 0 {
            // Width 0 means every value is 0; the stream carries no data, so
            // the caller's count is what bounds it.
            return Ok(Some(0));
        }
        if self.pos >= self.buf.len() {
            return Ok(None);
        }

        let header = self.varint()?;
        self.run.clear();
        self.run_pos = 0;

        if header & 1 == 0 {
            // RLE run: `header >> 1` repeats of one little-endian value whose
            // width is bit_width rounded up to whole bytes.
            let count = (header >> 1) as usize;
            let nbytes = (self.bit_width as usize).div_ceil(8);
            if self.pos + nbytes > self.buf.len() {
                return Err("rle: truncated repeated value".into());
            }
            let mut v = 0u32;
            for i in 0..nbytes {
                v |= (self.buf[self.pos + i] as u32) << (8 * i);
            }
            self.pos += nbytes;
            self.run.resize(count, v);
        } else {
            // Bit-packed run: `header >> 1` groups of 8 values each.
            let groups = (header >> 1) as usize;
            let count = groups * 8;
            let nbytes = groups * self.bit_width as usize;
            if self.pos + nbytes > self.buf.len() {
                return Err("rle: truncated bit-packed run".into());
            }
            let data = &self.buf[self.pos..self.pos + nbytes];
            self.pos += nbytes;
            self.run.reserve(count);
            let width = self.bit_width as usize;
            let mask = if width == 32 {
                u32::MAX
            } else {
                (1u32 << width) - 1
            };
            for i in 0..count {
                let bit = i * width;
                let mut v = 0u64;
                // Values are LSB-first and may straddle byte boundaries.
                for k in 0..((width + (bit % 8) + 7) / 8) {
                    let idx = bit / 8 + k;
                    if idx < data.len() {
                        v |= (data[idx] as u64) << (8 * k);
                    }
                }
                self.run.push(((v >> (bit % 8)) as u32) & mask);
            }
        }

        if self.run.is_empty() {
            return Ok(None);
        }
        let v = self.run[0];
        self.run_pos = 1;
        Ok(Some(v))
    }
}

// ── File metadata ──────────────────────────────────────────────────────────

const CODEC_UNCOMPRESSED: i64 = 0;
const CODEC_SNAPPY: i64 = 1;

const PAGE_DICTIONARY: i64 = 2;
const PAGE_DATA_V1: i64 = 0;
const PAGE_DATA_V2: i64 = 3;

const ENC_PLAIN: i64 = 0;
const ENC_PLAIN_DICTIONARY: i64 = 2;
const ENC_RLE_DICTIONARY: i64 = 8;

/// Where one column's pages live inside one row group.
#[derive(Debug, Clone)]
struct ColumnChunk {
    codec: i64,
    num_values: i64,
    data_page_offset: i64,
    dictionary_page_offset: Option<i64>,
    total_compressed_size: i64,
}

impl ColumnChunk {
    /// Byte offset of the first page: the dictionary page when present,
    /// otherwise the first data page.
    fn start(&self) -> i64 {
        match self.dictionary_page_offset {
            Some(d) if d > 0 && d < self.data_page_offset => d,
            _ => self.data_page_offset,
        }
    }
}

/// Everything we keep from the file footer.
struct FileMeta {
    /// Column chunks per row group: `chunks[group][col]`, in the order the
    /// columns were requested.
    chunks: Vec<Vec<ColumnChunk>>,
    num_rows: i64,
    /// Max definition level per requested column: 0 = REQUIRED, 1 = OPTIONAL.
    max_def_levels: Vec<u8>,
}

/// One schema element we care about (flat schema only).
struct SchemaElement {
    name: String,
    ty: Option<i64>,
    repetition: i64,
    num_children: i64,
}

fn parse_schema(t: &mut Thrift) -> Result<Vec<SchemaElement>> {
    let (n, _) = t.list_header()?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut id = 0;
        let mut el = SchemaElement {
            name: String::new(),
            ty: None,
            repetition: 0,
            num_children: 0,
        };
        while let Some((fid, ty)) = t.field(&mut id)? {
            match fid {
                1 => el.ty = Some(t.zigzag()?),
                3 => el.repetition = t.zigzag()?,
                4 => el.name = String::from_utf8_lossy(t.bytes()?).into_owned(),
                5 => el.num_children = t.zigzag()?,
                _ => t.skip(ty)?,
            }
        }
        out.push(el);
    }
    Ok(out)
}

/// Parse the column metadata of one column chunk. Returns it together with its
/// index in `columns`, or `None` if it is a column we were not asked for.
fn parse_column_chunk(t: &mut Thrift, columns: &[&str]) -> Result<Option<(usize, ColumnChunk)>> {
    let mut id = 0;
    let mut found = None;
    while let Some((fid, ty)) = t.field(&mut id)? {
        if fid == 3 && ty == T_STRUCT {
            // ColumnMetaData
            let mut mid = 0;
            let mut which: Option<usize> = None;
            let mut chunk = ColumnChunk {
                codec: CODEC_UNCOMPRESSED,
                num_values: 0,
                data_page_offset: 0,
                dictionary_page_offset: None,
                total_compressed_size: 0,
            };
            while let Some((mfid, mty)) = t.field(&mut mid)? {
                match mfid {
                    3 => {
                        // path_in_schema: list<string>. Flat schema, so a
                        // single-element path equal to the column name.
                        let (n, _) = t.list_header()?;
                        let mut parts = Vec::with_capacity(n);
                        for _ in 0..n {
                            parts.push(String::from_utf8_lossy(t.bytes()?).into_owned());
                        }
                        if parts.len() == 1 {
                            which = columns.iter().position(|c| *c == parts[0]);
                        }
                    }
                    4 => chunk.codec = t.zigzag()?,
                    5 => chunk.num_values = t.zigzag()?,
                    9 => chunk.data_page_offset = t.zigzag()?,
                    11 => chunk.dictionary_page_offset = Some(t.zigzag()?),
                    7 => chunk.total_compressed_size = t.zigzag()?,
                    _ => t.skip(mty)?,
                }
            }
            if let Some(i) = which {
                found = Some((i, chunk));
            }
        } else {
            t.skip(ty)?;
        }
    }
    Ok(found)
}

fn parse_footer(meta: &[u8], columns: &[&str]) -> Result<FileMeta> {
    let mut t = Thrift::new(meta);
    let mut id = 0;
    let mut num_rows = 0;
    let mut chunks = Vec::new();
    let mut schema: Vec<SchemaElement> = Vec::new();

    while let Some((fid, ty)) = t.field(&mut id)? {
        match fid {
            2 if ty == T_LIST => schema = parse_schema(&mut t)?,
            3 => num_rows = t.zigzag()?,
            4 if ty == T_LIST => {
                let (ngroups, _) = t.list_header()?;
                chunks.reserve(ngroups);
                for _ in 0..ngroups {
                    // RowGroup
                    let mut gid = 0;
                    while let Some((gfid, gty)) = t.field(&mut gid)? {
                        if gfid == 1 && gty == T_LIST {
                            let (ncols, _) = t.list_header()?;
                            // Slot by requested-column index, so a group's
                            // chunks come back in the caller's order regardless
                            // of how they are laid out in the file.
                            let mut hits: Vec<Option<ColumnChunk>> = vec![None; columns.len()];
                            for _ in 0..ncols {
                                if let Some((i, c)) = parse_column_chunk(&mut t, columns)? {
                                    hits[i] = Some(c);
                                }
                            }
                            if hits.iter().all(|h| h.is_some()) {
                                chunks.push(hits.into_iter().map(|h| h.unwrap()).collect());
                            }
                        } else {
                            t.skip(gty)?;
                        }
                    }
                }
            }
            _ => t.skip(ty)?,
        }
    }

    // Locate each column in the schema to learn its type and nullability.
    // Element 0 is the root; the rest are the flat leaf columns.
    let mut max_def_levels = Vec::with_capacity(columns.len());
    for column in columns {
        let leaf = schema
            .iter()
            .skip(1)
            .find(|e| &e.name == column)
            .ok_or_else(|| {
                let names: Vec<&str> = schema.iter().skip(1).map(|e| e.name.as_str()).collect();
                format!("column {column:?} not found; file has {names:?}")
            })?;
        if leaf.num_children != 0 {
            return Err(format!(
                "column {column:?} is nested, which is not supported"
            ));
        }
        // Type 6 = BYTE_ARRAY.
        if leaf.ty != Some(6) {
            return Err(format!(
                "column {column:?} has parquet type {:?}, expected BYTE_ARRAY",
                leaf.ty
            ));
        }
        // Repetition 0 = REQUIRED, 1 = OPTIONAL, 2 = REPEATED.
        max_def_levels.push(match leaf.repetition {
            0 => 0,
            1 => 1,
            _ => {
                return Err(format!(
                    "column {column:?} is REPEATED, which is not supported"
                ));
            }
        });
    }

    if chunks.is_empty() {
        return Err(format!("no row group contains all of {columns:?}"));
    }

    Ok(FileMeta {
        chunks,
        num_rows,
        max_def_levels,
    })
}

// ── Page headers ───────────────────────────────────────────────────────────

#[derive(Default, Debug)]
struct PageHeader {
    page_type: i64,
    uncompressed_size: i32,
    compressed_size: i32,
    /// v1 data page: (num_values, encoding, def_level_encoding)
    v1: Option<(i32, i64, i64)>,
    /// v2 data page: (num_values, num_nulls, encoding, def_levels_byte_len, is_compressed)
    v2: Option<(i32, i32, i64, i32, bool)>,
    /// dictionary page: (num_values, encoding)
    dict: Option<(i32, i64)>,
}

/// Parse a page header from the front of `buf`, returning it and how many bytes
/// it consumed.
fn parse_page_header(buf: &[u8]) -> Result<(PageHeader, usize)> {
    let mut t = Thrift::new(buf);
    let mut id = 0;
    let mut ph = PageHeader::default();

    while let Some((fid, ty)) = t.field(&mut id)? {
        match fid {
            1 => ph.page_type = t.zigzag()?,
            2 => ph.uncompressed_size = t.zigzag()? as i32,
            3 => ph.compressed_size = t.zigzag()? as i32,
            5 if ty == T_STRUCT => {
                // DataPageHeader (v1)
                let mut did = 0;
                let (mut nv, mut enc, mut dle) = (0i32, ENC_PLAIN, 3i64);
                while let Some((dfid, dty)) = t.field(&mut did)? {
                    match dfid {
                        1 => nv = t.zigzag()? as i32,
                        2 => enc = t.zigzag()?,
                        3 => dle = t.zigzag()?,
                        _ => t.skip(dty)?,
                    }
                }
                ph.v1 = Some((nv, enc, dle));
            }
            7 if ty == T_STRUCT => {
                // DictionaryPageHeader
                let mut did = 0;
                let (mut nv, mut enc) = (0i32, ENC_PLAIN);
                while let Some((dfid, dty)) = t.field(&mut did)? {
                    match dfid {
                        1 => nv = t.zigzag()? as i32,
                        2 => enc = t.zigzag()?,
                        _ => t.skip(dty)?,
                    }
                }
                ph.dict = Some((nv, enc));
            }
            8 if ty == T_STRUCT => {
                // DataPageHeaderV2
                let mut did = 0;
                let (mut nv, mut nn, mut enc, mut dbl) = (0i32, 0i32, ENC_PLAIN, 0i32);
                // `is_compressed` defaults to true when absent.
                let mut compressed = true;
                while let Some((dfid, dty)) = t.field(&mut did)? {
                    match dfid {
                        1 => nv = t.zigzag()? as i32,
                        2 => nn = t.zigzag()? as i32,
                        4 => enc = t.zigzag()?,
                        5 => dbl = t.zigzag()? as i32,
                        7 => compressed = dty == T_BOOL_TRUE,
                        _ => t.skip(dty)?,
                    }
                }
                ph.v2 = Some((nv, nn, enc, dbl, compressed));
            }
            _ => t.skip(ty)?,
        }
    }

    Ok((ph, t.pos))
}

// ── PLAIN byte-array decoding ──────────────────────────────────────────────

/// Decode a PLAIN-encoded BYTE_ARRAY block: a 4-byte little-endian length
/// followed by that many bytes, repeated. Pushes offsets into `buf`.
fn plain_byte_arrays(data: &[u8], limit: usize, out: &mut Vec<Vec<u8>>) -> Result<()> {
    let mut pos = 0;
    while pos + 4 <= data.len() && out.len() < limit {
        let len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if pos + len > data.len() {
            return Err("plain: byte array runs past end of page".into());
        }
        out.push(data[pos..pos + len].to_vec());
        pos += len;
    }
    Ok(())
}

// ── Public reader ──────────────────────────────────────────────────────────

/// Streams one or more BYTE_ARRAY columns out of a parquet file, one row group
/// at a time. Row groups are the natural unit: each is independently encoded
/// and, in the corpora we train on, is a few MB — small enough to hold, large
/// enough that per-group overhead is irrelevant.
///
/// Reading several columns at once is what makes row-level filtering possible:
/// a group's columns decode in lockstep, so index `i` of each is the same row.
pub struct ParquetColumnReader {
    file: File,
    meta: FileMeta,
    next_group: usize,
}

impl ParquetColumnReader {
    /// Open `path` and prepare to read `column`. Fails if the column is missing,
    /// nested, or not a BYTE_ARRAY.
    pub fn open(path: &str, column: &str) -> Result<Self> {
        Self::open_columns(path, &[column])
    }

    /// Open `path` and prepare to read several columns together. Each call to
    /// `next_row_group_columns` then returns one `Vec` per column, aligned by
    /// row.
    pub fn open_columns(path: &str, columns: &[&str]) -> Result<Self> {
        assert!(!columns.is_empty(), "need at least one column");
        let mut file = File::open(path).map_err(|e| format!("could not open {path:?}: {e}"))?;
        let size = file
            .metadata()
            .map_err(|e| format!("could not stat {path:?}: {e}"))?
            .len();
        if size < 12 {
            return Err(format!("{path:?} is too small to be a parquet file"));
        }

        // Header and footer both carry the magic; the footer is
        // `<metadata> <metadata_len u32> PAR1`.
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)
            .map_err(|e| format!("could not read {path:?}: {e}"))?;
        if &magic != b"PAR1" {
            return Err(format!("{path:?} is not a parquet file (bad header magic)"));
        }

        let mut tail = [0u8; 8];
        file.seek(SeekFrom::End(-8))
            .and_then(|_| file.read_exact(&mut tail))
            .map_err(|e| format!("could not read {path:?} footer: {e}"))?;
        if &tail[4..] != b"PAR1" {
            return Err(format!("{path:?} is not a parquet file (bad footer magic)"));
        }
        let meta_len = u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]) as u64;
        if meta_len + 8 > size {
            return Err(format!(
                "{path:?}: footer length {meta_len} exceeds file size"
            ));
        }

        let mut meta_buf = vec![0u8; meta_len as usize];
        file.seek(SeekFrom::Start(size - 8 - meta_len))
            .and_then(|_| file.read_exact(&mut meta_buf))
            .map_err(|e| format!("could not read {path:?} metadata: {e}"))?;

        let meta = parse_footer(&meta_buf, columns).map_err(|e| format!("{path:?}: {e}"))?;

        Ok(Self {
            file,
            meta,
            next_group: 0,
        })
    }

    pub fn num_rows(&self) -> i64 {
        self.meta.num_rows
    }

    pub fn num_row_groups(&self) -> usize {
        self.meta.chunks.len()
    }

    /// Restart from the first row group.
    pub fn rewind(&mut self) {
        self.next_group = 0;
    }

    /// Decode the next row group's values for the first requested column.
    /// Returns `None` at the end of the file. Null entries are skipped, so the
    /// result may be shorter than the group's row count.
    pub fn next_row_group(&mut self) -> Result<Option<Vec<Vec<u8>>>> {
        Ok(self
            .next_row_group_columns()?
            .map(|mut cols| cols.swap_remove(0)))
    }

    /// Decode the next row group for every requested column, in the order they
    /// were passed to `open_columns`. Returns `None` at the end of the file.
    ///
    /// The columns are row-aligned only when none of them contains nulls (the
    /// case in the corpora we read, and what row-level filtering relies on):
    /// nulls are skipped rather than surfaced, so a null in one column would
    /// shift it against the others.
    pub fn next_row_group_columns(&mut self) -> Result<Option<Vec<Vec<Vec<u8>>>>> {
        let Some(group) = self.meta.chunks.get(self.next_group).cloned() else {
            return Ok(None);
        };
        let index = self.next_group;
        self.next_group += 1;

        let mut out = Vec::with_capacity(group.len());
        for (col, chunk) in group.into_iter().enumerate() {
            out.push(self.decode_chunk(&chunk, self.meta.max_def_levels[col], index)?);
        }
        Ok(Some(out))
    }

    /// Decode every page of one column chunk into its values.
    fn decode_chunk(
        &mut self,
        chunk: &ColumnChunk,
        max_def_level: u8,
        group_index: usize,
    ) -> Result<Vec<Vec<u8>>> {
        // One read for the whole chunk: pages are laid out contiguously from
        // the dictionary page (or first data page) onward.
        let start = chunk.start() as u64;
        let len = chunk.total_compressed_size as usize;
        let mut raw = vec![0u8; len];
        self.file
            .seek(SeekFrom::Start(start))
            .and_then(|_| self.file.read_exact(&mut raw))
            .map_err(|e| format!("could not read row group {group_index}: {e}"))?;

        let mut dictionary: Vec<Vec<u8>> = Vec::new();
        let mut values: Vec<Vec<u8>> = Vec::with_capacity(chunk.num_values.max(0) as usize);
        let mut pos = 0usize;

        while pos < raw.len() && (values.len() as i64) < chunk.num_values {
            let (ph, hdr_len) = parse_page_header(&raw[pos..])?;
            pos += hdr_len;
            let clen = ph.compressed_size as usize;
            if pos + clen > raw.len() {
                return Err("page runs past end of row group".into());
            }
            let page = &raw[pos..pos + clen];
            pos += clen;

            match ph.page_type {
                PAGE_DICTIONARY => {
                    let (nv, enc) = ph
                        .dict
                        .ok_or_else(|| "dictionary page without header".to_string())?;
                    if enc != ENC_PLAIN && enc != ENC_PLAIN_DICTIONARY {
                        return Err(format!("dictionary encoding {enc} is not supported"));
                    }
                    let data = decompress(chunk.codec, page, ph.uncompressed_size as usize)?;
                    dictionary.clear();
                    plain_byte_arrays(&data, nv.max(0) as usize, &mut dictionary)?;
                }
                PAGE_DATA_V1 => {
                    let (nv, enc, dle) = ph
                        .v1
                        .ok_or_else(|| "v1 data page without header".to_string())?;
                    let data = decompress(chunk.codec, page, ph.uncompressed_size as usize)?;

                    // v1 puts the definition levels inside the compressed
                    // payload, length-prefixed when RLE-encoded.
                    let mut body = &data[..];
                    let mut non_null = nv as usize;
                    if max_def_level > 0 {
                        if dle != 3 {
                            return Err(format!(
                                "definition level encoding {dle} is not supported (want RLE)"
                            ));
                        }
                        if body.len() < 4 {
                            return Err("v1 page: truncated definition levels".into());
                        }
                        let n = u32::from_le_bytes([body[0], body[1], body[2], body[3]]) as usize;
                        if 4 + n > body.len() {
                            return Err("v1 page: definition levels run past page".into());
                        }
                        non_null = count_non_null(&body[4..4 + n], nv as usize)?;
                        body = &body[4 + n..];
                    }

                    decode_values(enc, body, non_null, &dictionary, &mut values)?;
                }
                PAGE_DATA_V2 => {
                    let (nv, nn, enc, dbl, compressed) = ph
                        .v2
                        .ok_or_else(|| "v2 data page without header".to_string())?;
                    // v2 keeps the level sections uncompressed at the front of
                    // the page; only the value section is compressed.
                    let dbl = dbl as usize;
                    if dbl > page.len() {
                        return Err("v2 page: level section runs past page".into());
                    }
                    let value_bytes = &page[dbl..];
                    let data = if compressed {
                        decompress(
                            chunk.codec,
                            value_bytes,
                            (ph.uncompressed_size as usize).saturating_sub(dbl),
                        )?
                    } else {
                        value_bytes.to_vec()
                    };
                    let non_null = (nv - nn).max(0) as usize;
                    decode_values(enc, &data, non_null, &dictionary, &mut values)?;
                }
                // Index pages carry no values.
                _ => {}
            }
        }

        Ok(values)
    }
}

fn decompress(codec: i64, page: &[u8], uncompressed: usize) -> Result<Vec<u8>> {
    match codec {
        CODEC_UNCOMPRESSED => Ok(page.to_vec()),
        CODEC_SNAPPY => {
            let out = snappy_decompress(page)?;
            if uncompressed != 0 && out.len() != uncompressed {
                return Err(format!(
                    "snappy: got {} bytes, page header said {uncompressed}",
                    out.len()
                ));
            }
            Ok(out)
        }
        other => Err(format!(
            "compression codec {other} is not supported (only UNCOMPRESSED and SNAPPY)"
        )),
    }
}

/// Count how many of `total` definition levels are at the max level, i.e. how
/// many values the page actually stores.
fn count_non_null(levels: &[u8], total: usize) -> Result<usize> {
    let mut dec = RleDecoder::new(levels, 1);
    let mut n = 0;
    for _ in 0..total {
        match dec.next()? {
            Some(1) => n += 1,
            Some(_) => {}
            None => break,
        }
    }
    Ok(n)
}

/// Decode `count` values from a page body under `encoding`, appending to `out`.
fn decode_values(
    encoding: i64,
    body: &[u8],
    count: usize,
    dictionary: &[Vec<u8>],
    out: &mut Vec<Vec<u8>>,
) -> Result<()> {
    match encoding {
        ENC_PLAIN => plain_byte_arrays(body, count, out),
        ENC_PLAIN_DICTIONARY | ENC_RLE_DICTIONARY => {
            if dictionary.is_empty() {
                return Err("dictionary-encoded page with no dictionary page".into());
            }
            // Dictionary indices: a leading byte holds the bit width, then an
            // RLE/bit-packed stream of indices.
            let Some((&width, rest)) = body.split_first() else {
                return Err("dictionary page body is empty".into());
            };
            let mut dec = RleDecoder::new(rest, width);
            for _ in 0..count {
                let Some(idx) = dec.next()? else { break };
                let v = dictionary
                    .get(idx as usize)
                    .ok_or_else(|| format!("dictionary index {idx} out of range"))?;
                out.push(v.clone());
            }
            Ok(())
        }
        other => Err(format!(
            "value encoding {other} is not supported (only PLAIN and dictionary)"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Snappy round-trip against literals and back-references, including the
    /// overlapping-copy case that encodes runs.
    #[test]
    fn snappy_decodes_literals_and_copies() {
        // "abcabcabc": literal "abc" then a copy of length 6 at offset 3.
        let mut src = Vec::new();
        src.push(9); // uncompressed length varint
        src.push((3 - 1) << 2); // literal, len 3
        src.extend_from_slice(b"abc");
        // copy1: len 6 -> (6-4)=2 in bits 2..5, offset 3
        src.push(0x01 | (2 << 2) | ((3 >> 8) << 5) as u8);
        src.push(3);
        assert_eq!(snappy_decompress(&src).unwrap(), b"abcabcabc");
    }

    /// Bit-packed RLE runs must unpack LSB-first across byte boundaries.
    #[test]
    fn rle_decodes_bit_packed_run() {
        // One group of 8 three-bit values 0..7 => 24 bits.
        let vals: [u32; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        let mut packed = [0u8; 3];
        for (i, v) in vals.iter().enumerate() {
            let bit = i * 3;
            let x = (*v as u32) << (bit % 8);
            packed[bit / 8] |= x as u8;
            if bit % 8 > 5 {
                packed[bit / 8 + 1] |= (x >> 8) as u8;
            }
        }
        let mut buf = vec![(1 << 1) | 1]; // 1 group, bit-packed
        buf.extend_from_slice(&packed);
        let mut dec = RleDecoder::new(&buf, 3);
        for want in vals {
            assert_eq!(dec.next().unwrap(), Some(want));
        }
    }

    /// An RLE run repeats one value `count` times.
    #[test]
    fn rle_decodes_repeated_run() {
        let buf = vec![5 << 1, 1]; // 5 repeats of the value 1, one byte wide
        let mut dec = RleDecoder::new(&buf, 1);
        for _ in 0..5 {
            assert_eq!(dec.next().unwrap(), Some(1));
        }
    }

    #[test]
    fn plain_byte_arrays_round_trip() {
        let mut data = Vec::new();
        for s in ["hello", "", "world!"] {
            data.extend_from_slice(&(s.len() as u32).to_le_bytes());
            data.extend_from_slice(s.as_bytes());
        }
        let mut out = Vec::new();
        plain_byte_arrays(&data, 3, &mut out).unwrap();
        assert_eq!(
            out,
            vec![b"hello".to_vec(), b"".to_vec(), b"world!".to_vec()]
        );
    }
}
