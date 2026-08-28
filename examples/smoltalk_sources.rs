// Tally SmolTalk's `source` column: which subset each conversation came from,
// how long its assistant turns are, and how much it repeats itself. The subset
// is the only handle on answer length, and `distinct` is the one that says
// whether "short" means varied or means one canned line: a source can pass
// `epochs = 1` and still show the model the same reply thousands of times,
// because record-level dedup sees two different user turns.
use neural_networks::parquet::ParquetColumnReader;
use std::collections::{HashMap, HashSet};

fn main() {
    let dir = std::env::args().nth(1).unwrap();
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "parquet"))
        .collect();
    files.sort();

    // rows, total assistant bytes, assistant turns
    let mut tally: HashMap<String, (usize, usize, usize)> = HashMap::new();
    // distinct assistant turns, and a count of the most repeated one
    let mut turns_seen: HashMap<String, HashMap<String, usize>> = HashMap::new();
    for f in &files {
        let path = f.to_str().unwrap();
        let mut src = ParquetColumnReader::open(path, "source").unwrap();
        let mut msg = ParquetColumnReader::open_columns(
            path,
            &["messages.list.element.role", "messages.list.element.content"],
        )
        .unwrap();
        while let (Some(names), Some(rows)) =
            (src.next_row_group().unwrap(), msg.next_row_group_lists().unwrap())
        {
            for (name, row) in names.iter().zip(rows) {
                let key = String::from_utf8_lossy(name).to_string();
                let e = tally.entry(key.clone()).or_default();
                e.0 += 1;
                for element in row {
                    if element[0] == b"assistant" {
                        e.1 += element[1].len();
                        e.2 += 1;
                        *turns_seen
                            .entry(key.clone())
                            .or_default()
                            .entry(String::from_utf8_lossy(&element[1]).into_owned())
                            .or_insert(0) += 1;
                    }
                }
            }
        }
    }
    let mut rows: Vec<_> = tally.into_iter().collect();
    rows.sort_by_key(|(_, v)| std::cmp::Reverse(v.0));
    println!(
        "{:<28} {:>8} {:>8} {:>11} {:>9} {:>8}",
        "source", "convs", "turns", "mean asst B", "distinct", "top rep"
    );
    for (name, (n, bytes, turns)) in rows {
        let seen = &turns_seen[&name];
        let top = seen.values().max().copied().unwrap_or(0);
        let distinct: HashSet<&String> = seen.keys().collect();
        println!(
            "{name:<28} {n:>8} {turns:>8} {:>11} {:>9} {top:>8}",
            bytes / turns.max(1),
            distinct.len(),
        );
    }
}
