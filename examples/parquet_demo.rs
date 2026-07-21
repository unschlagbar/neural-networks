// Dump what a parquet corpus looks like to the trainer: row/group counts, the
// first few documents, and a streaming pass reporting throughput.
//
//   cargo run --release --example parquet_demo -- <file.parquet> [column] [--all]
//
// `--all` streams the whole file (a full decode of every row group) instead of
// stopping after the first few — use it to check a new corpus end to end.

use std::{env, time::Instant};

use neural_networks::parquet::ParquetColumnReader;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let all = args.iter().any(|a| a == "--all");
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();

    let Some(path) = positional.first() else {
        eprintln!("usage: parquet_demo <file.parquet> [column] [--all]");
        std::process::exit(2);
    };
    let column = positional.get(1).map(|s| s.as_str()).unwrap_or("text");

    let mut r = match ParquetColumnReader::open(path, column) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "{path}: {} rows, {} row groups, column {column:?}",
        r.num_rows(),
        r.num_row_groups()
    );

    let start = Instant::now();
    let mut groups = 0;
    let mut docs = 0usize;
    let mut bytes = 0usize;

    loop {
        let group = match r.next_row_group() {
            Ok(Some(g)) => g,
            Ok(None) => break,
            Err(e) => {
                eprintln!("error in row group {groups}: {e}");
                std::process::exit(1);
            }
        };

        if groups == 0 {
            for (i, v) in group.iter().take(3).enumerate() {
                let text = String::from_utf8_lossy(v);
                let head: String = text.chars().take(200).collect();
                println!("\n--- doc {i} ({} bytes) ---\n{head}", v.len());
            }
            println!();
        }

        groups += 1;
        docs += group.len();
        bytes += group.iter().map(|v| v.len()).sum::<usize>();

        if !all && groups >= 4 {
            break;
        }
    }

    let secs = start.elapsed().as_secs_f64();
    println!(
        "decoded {groups} row groups, {docs} docs, {:.1} MB in {secs:.2}s ({:.0} MB/s)",
        bytes as f64 / 1e6,
        bytes as f64 / 1e6 / secs.max(1e-9),
    );
}
