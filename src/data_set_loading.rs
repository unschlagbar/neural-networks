use std::{fs, path::PathBuf};

use rand::{rng, seq::SliceRandom};

use crate::tokenizer::Tokenizer;

pub struct DataSet {
    files: Vec<PathBuf>,
    tokenizer: Tokenizer,
}

impl DataSet {
    pub fn load_rust_files(tokenizer: &Tokenizer) -> Self {
        let mut files: Vec<PathBuf> = fs::read_dir("cargo_rust_files/")
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();

        files.shuffle(&mut rng());

        Self {
            files,
            tokenizer: tokenizer.clone(),
        }
    }
}

pub struct DataSetIter {
    files: Vec<PathBuf>,
    tokenizer: Tokenizer,
    idx: usize,
}

impl IntoIterator for DataSet {
    type Item = Vec<u16>;
    type IntoIter = DataSetIter;

    fn into_iter(self) -> Self::IntoIter {
        DataSetIter {
            files: self.files,
            tokenizer: self.tokenizer,
            idx: 0,
        }
    }
}

impl Iterator for DataSetIter {
    type Item = Vec<u16>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.files.len() {
            return None;
        }
        let path = &self.files[self.idx];
        self.idx += 1;
        let content = fs::read_to_string(path).ok()?;
        let data: Vec<u16> = self.tokenizer.to_tokens(&content);
        Some(data)
    }
}