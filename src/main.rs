use rand::rng;
use rand::seq::SliceRandom;

use crate::lstm::LSTM;

pub mod batches;
pub mod layer;
pub mod lstm;
pub mod mlp;

use std::collections::HashMap;
use std::fs;
use std::io::stdin;
use std::path::PathBuf; // passe den Pfad an, je nachdem wie dein Modul heißt

fn main() {
    // 1. Trainingsdaten laden (z.B. aus alice.txt)
    let text = get_all_text();
    let chars: Vec<char> = text.chars().collect();

    // 2. Alphabet bauen
    let mut vocab: Vec<char> = chars.clone();
    vocab.sort();
    vocab.dedup();
    let vocab_size = vocab.len();

    println!("Vocab size: {vocab_size}");

    // Mappings char->id und id->char
    let mut stoi = HashMap::new();
    let mut itos = HashMap::new();
    for (i, c) in vocab.iter().enumerate() {
        stoi.insert(*c, i as u16);
        itos.insert(i as u16, *c);
    }

    drop(text);

    // 5. Model initialisieren
    let mut model = if let Ok(model) = LSTM::load("Jarvis") {
        model
    } else {
        LSTM::new(&[vocab_size, 128, 128], vocab_size)
    };

    //test(&mut model, &stoi, &itos);

    let mut files: Vec<PathBuf> = fs::read_dir("training_files/").unwrap().into_iter().map(|e| e.unwrap().path()).collect();
    files.shuffle(&mut rng());

    for (i, entry) in files.iter().enumerate() {
        let content = fs::read_to_string(&entry).unwrap();
        //println!("content: {}", &content);

        // 3. Text in Indizes umwandeln
        let data: Vec<u16> = content.chars().filter_map(|ch| stoi.get(&ch).map(|&val| val as u16)).collect();
    
        model.train(&data, 100..500, 1);
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

    let start_tag = "<rohtext>\n";
    let end_tag = "\n</rohtext>";

    let mut i = 0;

    while let Some(start) = xml[search_start..].find(start_tag) {
        let start_index = search_start + start + start_tag.len();

        if let Some(end) = xml[start_index..].find(end_tag) {
            let end_index = start_index + end;
            let mut content = xml[start_index..end_index].to_string().replace("&quot;", "");
            content = content.replace('\u{00A0}', "");
            fs::write(format!("training_files/{i}", ),&content).unwrap();

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
    let xml = fs::read_to_string("deu_mixed-typical_2011_10K-sentences.txt").expect("Datei konnte nicht gelesen werden");

    fs::create_dir("training_files2").unwrap();

    let mut i = 0;

    for sent in xml.lines() {
        let content = sent.split_once(' ').unwrap().1;
        fs::write(format!("training_files2/{i}", ),&content).unwrap();
        i += 1;
    }
}

fn get_all_text() -> String {
        // Datei einlesen
    let xml = fs::read_to_string("input.xml").expect("Datei konnte nicht gelesen werden");

    let mut search_start = 0;

    let start_tag = "<rohtext>\n";
    let end_tag = "\n</rohtext>";

    let mut i = 0;
    let mut contents = String::new();

    while let Some(start) = xml[search_start..].find(start_tag) {
        let start_index = search_start + start + start_tag.len();

        if let Some(end) = xml[start_index..].find(end_tag) {
            let end_index = start_index + end;
            let content = &xml[start_index..end_index];
            fs::write(format!("training_files/{i}", ),content).unwrap();
            contents.push_str(content);

            search_start = end_index + end_tag.len();
        } else {
            break; // kein schließendes Tag mehr gefunden
        }
        i += 1;
    }

    contents
}

pub fn test(model: &mut LSTM, stoi: &HashMap<char, u16>, itos: &HashMap<u16, char>) {
    loop {
        let mut input = String::new();
        stdin().read_line(&mut input).unwrap();
        let prefix: Box<[u16]> = input.trim()
            .chars()
            .filter_map(|ch| stoi.get(&ch).map(|&val| val as u16))
            .collect();

        let output = model.sample(&prefix, 1000, 0.5);
        let output: String = output.into_iter().map(|i| itos.get(&i).unwrap()).collect();
        println!("response: {output}");
    }
}
