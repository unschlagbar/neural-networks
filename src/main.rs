use rand::rng;
use rand::seq::SliceRandom;

use crate::lstm::LSTM;
use crate::tokenizer::Tokenizer;

pub mod batches;
pub mod jarvis;
pub mod layer;
pub mod lstm;
pub mod mlp;
pub mod tokenizer;

use std::fs;
use std::io::stdin;
use std::path::{Path, PathBuf};

fn main() {
    #[allow(unused)]
    let tokenizer = Tokenizer::new("charset.txt");

    //println!("{:?}", tokenizer.itos);

    // 5. Model initialisieren
    let mut model = if let Ok(model) = LSTM::load("Jarvis") {
        model
    } else {
        let model = LSTM::new(&[tokenizer.vocab_size(), 512, 512], tokenizer.vocab_size());
        model.save("Jarvis").unwrap();
        model
    };

    //test(&mut model, &tokenizer);

    let mut files: Vec<PathBuf> = fs::read_dir("rust_files/")
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();

    files.shuffle(&mut rng());
    files.shuffle(&mut rng());
    files.shuffle(&mut rng());

    for (i, entry) in files.iter().enumerate() {
        let content = fs::read_to_string(entry).unwrap();

        let data: Vec<u16> = tokenizer.to_tokens(&content);
        model.train(&data, 200..250, &[], 1, 0.001);

        println!("completed data {i}")
    }

    println!("Training abgeschlossen!");
}

#[test]
fn build_trainings_set() {
    // Datei einlesen
    let xml = fs::read_to_string("input.xml").expect("Datei konnte nicht gelesen werden");

    fs::create_dir("training_files").unwrap();

    let mut search_start = 0;

    let start_tag = "<rohtext>";
    let end_tag = "</rohtext>";

    let mut i = 0;

    while let Some(start) = xml[search_start..].find(start_tag) {
        let start_index = search_start + start + start_tag.len() + 2;

        if let Some(end) = xml[start_index..].find(end_tag) {
            let end_index = start_index + end - 2;
            let mut content = xml[start_index..end_index].to_string();
            content = content.replace("&quot;", "");
            content = content.replace('\u{00A0}', "");
            content = content.replace('\u{00AD}', "");
            content = content.replace('\u{2013}', "\u{002d}");
            content = content.replace('\u{2018}', "\u{0060}");
            fs::write(format!("training_files/{i}",), &content).unwrap();

            search_start = end_index + end_tag.len();
        } else {
            break; // kein schließendes Tag mehr gefunden
        }
        i += 1;
    }
}

#[test]
fn build_trainings_set2() {
    // Datei einlesen
    let mut files = Vec::new();
    collect_files_recursively(Path::new("C:/Users/e7438/Desktop/Pumpkin"), &mut files);

    let target = "rust_files";

    fs::create_dir(target).unwrap();

    for (i, content) in files.iter().enumerate() {
        fs::write(format!("{target}/{i}.txt",), content.trim()).unwrap();
    }
}

pub fn test(model: &mut LSTM, tokenizer: &Tokenizer) {
    loop {
        let mut input = String::new();
        stdin().read_line(&mut input).unwrap();
        let prefix = tokenizer.to_tokens(&input.trim());

        let output = model.sample(&prefix, 1000, 0.4);
        let output = tokenizer.to_text(&output);
        println!("response: {output}");
    }
}

pub fn collect_files_recursively(dir: &Path, files: &mut Vec<String>) {
    if dir.is_dir() {
        let entries =
            fs::read_dir(dir).unwrap_or_else(|_| panic!("Kann Verzeichnis {:?} nicht lesen", dir));

        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.is_dir() {
                    // Rekursiver Aufruf für Unterverzeichnisse
                    collect_files_recursively(&path, files);
                } else if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("rs")
                {
                    files.push(fs::read_to_string(path).unwrap());
                }
            }
        }
    }
}
