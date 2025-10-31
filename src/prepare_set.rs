#![allow(unused)]
use std::{fs, io::Write};

const DIR: &str = "political_speeches";

#[test]
fn political() {
    let corpus = fs::read_to_string("political_speeches.xml").unwrap();

    let mut speeches = Vec::new();

    for seq in corpus.split("<rohtext>\n").skip(1) {
        let speech = seq.split_once("\n</rohtext>").unwrap().0;
        speeches.push(speech);
    }

    fs::create_dir(DIR).unwrap();

    for (i, speech) in speeches.into_iter().enumerate() {
        let mut file = fs::File::create(format!("{DIR}/{i}")).unwrap();

        for part in speech.split("&amp;") {
            for part in part.split("apos;") {
                for part in part.split("&quot;") {
                    file.write_all(part.as_bytes()).unwrap();
                }
            }
        }
    }
}