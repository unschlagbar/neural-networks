//! Only for data set extraction
#![allow(unused)]

use std::{
    fs::{self, File},
    io::Write,
};

const FILE: &str = "data/political_speeches.txt";

#[test]
fn political() {
    let corpus = fs::read_to_string("political_speeches.xml").unwrap();

    let mut speeches = Vec::new();

    for seq in corpus.split("<rohtext>\n").skip(1) {
        let speech = seq.split_once("\n</rohtext>").unwrap().0;
        speeches.push(speech);
    }

    let mut file = File::create(FILE).unwrap();
    let mut first = true;

    for (i, speech) in speeches.into_iter().enumerate() {
        if !first {
            file.write_all(b"\n\n---FILE---\n\n").unwrap();
        } else {
            first = false;
        }

        for part in speech.split("&amp;") {
            for part in part.split("apos;") {
                for part in part.split("&quot;") {
                    file.write_all(part.as_bytes()).unwrap();
                }
            }
        }
    }
}

#[test]
fn collect_rs_files_for_training() {
    let src_dir = std::path::Path::new("/home/unschlagbar/Downloads/rust");
    let dst_dir = std::path::Path::new("/home/unschlagbar/Dev/neural-networks/data/rust-lib");

    std::fs::create_dir_all(dst_dir).unwrap();

    let mut stack = vec![src_dir.to_path_buf()];
    let mut count = 0u32;

    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();

            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                let file_name = path.file_name().unwrap().to_string_lossy().to_string();

                // Resolve name collisions
                let mut dst = dst_dir.join(&file_name);
                if dst.exists() {
                    let stem = path.file_stem().unwrap().to_string_lossy().to_string();
                    let mut i = 1u32;
                    loop {
                        dst = dst_dir.join(format!("{}_{}.rs", stem, i));
                        if !dst.exists() {
                            break;
                        }
                        i += 1;
                    }
                }

                std::fs::copy(&path, &dst).unwrap();
                count += 1;
            }
        }
    }

    println!("{} .rs-Dateien kopiert nach {:?}", count, dst_dir);
}
