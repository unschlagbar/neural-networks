use std::{fs, path::PathBuf, rc::Rc};

use rand::{rng, seq::SliceRandom};

use crate::tokenizer::Tokenizer;

pub struct DataSet {
    files: Vec<PathBuf>,
    tokenizer: Rc<Tokenizer>,
}

impl DataSet {
    pub fn load_rust_files(tokenizer: Rc<Tokenizer>) -> Self {
        let mut files: Vec<PathBuf> = fs::read_dir("cargo_rust_files/")
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();

        files.shuffle(&mut rng());
        files.shuffle(&mut rng());
        files.shuffle(&mut rng());
        files.shuffle(&mut rng());
        files.shuffle(&mut rng());

        Self { files, tokenizer }
    }

    pub fn load_pumpkin_files(tokenizer: Rc<Tokenizer>, root: &str) -> Self {
        let mut files: Vec<PathBuf> = collect_files_with_extension(root.into(), "rs", 110_000);

        files.shuffle(&mut rng());
        files.shuffle(&mut rng());
        files.shuffle(&mut rng());
        files.shuffle(&mut rng());
        files.shuffle(&mut rng());

        Self { files, tokenizer }
    }

    pub fn load_file(tokenizer: Rc<Tokenizer>, path: &str) -> Self {
        Self {
            files: vec![path.into()],
            tokenizer,
        }
    }
}

pub struct DataSetIter {
    files: Vec<PathBuf>,
    tokenizer: Rc<Tokenizer>,
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

pub fn collect_files_with_extension(root: PathBuf, extension: &str, max_size: u64) -> Vec<PathBuf> {
    let mut collected = Vec::new();
    if root.is_dir() {
        collect_recursively(root, extension, max_size, &mut collected);
    } else {
        panic!("not a dir!");
    }
    if collected.is_empty() {
        panic!("invalid dir!");
    }
    collected
}

fn collect_recursively(dir: PathBuf, extension: &str, max_size: u64, collected: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_recursively(path, extension, max_size, collected);
            } else if let Some(ext) = path.extension() {
                if ext == extension {
                    if let Ok(metadata) = fs::metadata(&path) {
                        if metadata.len() <= max_size {
                            collected.push(path);
                        }
                    }
                }
            }
        }
    }
}
