// Staging shards: filtered records parked on disk, one file per source.
//
// The mixer needs two things the input readers cannot give it — random access
// (to shuffle) and the final token count of a source (to size its share) — and
// neither may cost the corpus in RAM. So pass one streams every source through
// the filters into a shard and records where each record landed; pass two
// reads records back by index.

use std::fs::File;
use std::io::{BufReader, BufWriter, Seek, SeekFrom, Write};

use crate::config::Result;
use crate::record::Record;

pub struct Shard {
    path: String,
    out: BufWriter<File>,
    offsets: Vec<u64>,
    at: u64,
    pub tokens: usize,
}

impl Shard {
    pub fn create(dir: &str, name: &str) -> Result<Self> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{dir}: {e}"))?;
        let path = format!("{dir}/{name}.shard");
        let file = File::create(&path).map_err(|e| format!("{path}: {e}"))?;
        Ok(Self {
            path,
            out: BufWriter::new(file),
            offsets: Vec::new(),
            at: 0,
            tokens: 0,
        })
    }

    pub fn push(&mut self, rec: &Record) -> Result<()> {
        let mut buf = Vec::new();
        rec.write_to(&mut buf).map_err(|e| e.to_string())?;
        self.out
            .write_all(&buf)
            .map_err(|e| format!("{}: {e}", self.path))?;
        self.offsets.push(self.at);
        self.at += buf.len() as u64;
        self.tokens += rec.tokens();
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    pub fn into_reader(mut self) -> Result<ShardReader> {
        self.out.flush().map_err(|e| e.to_string())?;
        let file = File::open(&self.path).map_err(|e| format!("{}: {e}", self.path))?;
        Ok(ShardReader {
            path: self.path,
            file: BufReader::new(file),
            offsets: self.offsets,
        })
    }
}

pub struct ShardReader {
    path: String,
    file: BufReader<File>,
    offsets: Vec<u64>,
}

impl ShardReader {
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    pub fn get(&mut self, i: usize) -> Result<Record> {
        self.file
            .seek(SeekFrom::Start(self.offsets[i]))
            .map_err(|e| format!("{}: {e}", self.path))?;
        Record::read_from(&mut self.file).map_err(|e| format!("{}: {e}", self.path))
    }

    pub fn remove(self) {
        let _ = std::fs::remove_file(&self.path);
    }
}
