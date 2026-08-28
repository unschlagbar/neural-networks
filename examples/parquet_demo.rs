// Dump what a parquet corpus looks like to the trainer: row/group counts, the
// first few documents, and a streaming pass reporting throughput.
//
//   cargo run --release --example parquet_demo -- <file.parquet> [column] [--all]
//   cargo run --release --example parquet_demo -- <file.parquet> --chat [--all]
//
// `--all` streams the whole file (a full decode of every row group) instead of
// stopping after the first few — use it to check a new corpus end to end.
// `--chat` reads a `messages: list<struct<role, content>>` column instead of a
// flat one, and prints the conversations turn by turn.

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

    if args.iter().any(|a| a == "--chat") {
        chat(path, all);
        return;
    }

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

/// Dump a chat corpus: one `list<struct<role, content>>` column, printed as
/// conversations. This is the shape `datamix` reads for multi-turn SFT.
fn chat(path: &str, all: bool) {
    let mut r = match ParquetColumnReader::open_columns(path, &["role", "content"]) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "{path}: {} rows, {} row groups, chat columns role+content",
        r.num_rows(),
        r.num_row_groups()
    );

    let start = Instant::now();
    let (mut groups, mut rows, mut turns, mut bytes) = (0usize, 0usize, 0usize, 0usize);
    loop {
        let group = match r.next_row_group_lists() {
            Ok(Some(g)) => g,
            Ok(None) => break,
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        };
        for row in &group {
            if rows < 2 && !all {
                println!("\n--- conversation {rows} ({} turns) ---", row.len());
                for turn in row {
                    let role = String::from_utf8_lossy(&turn[0]);
                    let content = String::from_utf8_lossy(&turn[1]);
                    let head: String = content.chars().take(160).collect();
                    println!("[{role}] {head}");
                }
            }
            rows += 1;
            turns += row.len();
            bytes += row.iter().map(|t| t[1].len()).sum::<usize>();
        }
        groups += 1;
        if !all && groups >= 4 {
            break;
        }
    }
    let secs = start.elapsed().as_secs_f64();
    println!(
        "\ndecoded {groups} row groups, {rows} conversations, {turns} turns, \
         {:.1} MB in {secs:.2}s ({:.0} MB/s)",
        bytes as f64 / 1e6,
        bytes as f64 / 1e6 / secs
    );
}
