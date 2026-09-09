// A hand-editing UI for SFT data: a loopback HTTP server that serves one page
// and reads/writes the JSONL that page edits. The corpus it produces is a
// plain `kind = "jsonl"` source, so a hand-written file mixes in like any other.
//
// The page holds the whole file as text and does its own JSON escaping through
// `JSON.stringify`, so nothing here parses records — this side only moves bytes
// and runs the training loader over what was saved.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};

use crate::config::Result;

pub fn serve(path: &str, port: u16) -> Result<()> {
    let listener = TcpListener::bind(("127.0.0.1", port))
        .map_err(|e| format!("bind 127.0.0.1:{port}: {e}"))?;
    if !std::path::Path::new(path).exists() {
        if let Some(dir) = std::path::Path::new(path).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        std::fs::write(path, "").map_err(|e| format!("{path}: {e}"))?;
    }
    println!("editing {path}");
    println!("open http://127.0.0.1:{port}/   (ctrl-c to stop)");
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                if let Err(e) = handle(s, path) {
                    eprintln!("datamix edit: {e}");
                }
            }
            Err(e) => eprintln!("datamix edit: accept: {e}"),
        }
    }
    Ok(())
}

fn handle(mut stream: TcpStream, path: &str) -> Result<()> {
    let mut reader = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);
    let mut request = String::new();
    reader
        .read_line(&mut request)
        .map_err(|e| format!("read request: {e}"))?;
    let mut parts = request.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("/").to_string();

    let mut len = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
            break;
        }
        if line.trim().is_empty() {
            break;
        }
        if let Some(v) = header(&line, "content-length") {
            len = v.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0u8; len];
    if len > 0 {
        reader
            .read_exact(&mut body)
            .map_err(|e| format!("read body: {e}"))?;
    }

    match (method.as_str(), target.as_str()) {
        ("GET", "/") | ("GET", "/index.html") => {
            reply(&mut stream, "200 OK", "text/html; charset=utf-8", PAGE.as_bytes())
        }
        ("GET", "/api/file") => {
            let text = std::fs::read(path).unwrap_or_default();
            reply(&mut stream, "200 OK", "text/plain; charset=utf-8", &text)
        }
        ("POST", "/api/file") => {
            let msg = match save(path, &body) {
                Ok(m) => m,
                Err(e) => format!("save failed: {e}"),
            };
            reply(&mut stream, "200 OK", "text/plain; charset=utf-8", msg.as_bytes())
        }
        _ => reply(&mut stream, "404 Not Found", "text/plain", b"not found"),
    }
}

/// Write through a sibling temp file so an interrupted save cannot leave a
/// half-written corpus, then report what the *training* loader makes of it.
fn save(path: &str, body: &[u8]) -> Result<String> {
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, body).map_err(|e| format!("{tmp}: {e}"))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{path}: {e}"))?;
    let saved = body.iter().filter(|&&b| b == b'\n').count();
    match crate::verify_report(path) {
        Ok(stats) => Ok(format!("saved {saved} records — {stats}")),
        Err(e) => Ok(format!("saved {saved} records — loader: {e}")),
    }
}

fn header(line: &str, name: &str) -> Option<String> {
    let (k, v) = line.split_once(':')?;
    k.trim()
        .eq_ignore_ascii_case(name)
        .then(|| v.trim().to_string())
}

fn reply(stream: &mut TcpStream, status: &str, mime: &str, body: &[u8]) -> Result<()> {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {mime}\r\nContent-Length: {}\r\n\
         Cache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
    stream.write_all(body).map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())
}

const PAGE: &str = include_str!("editor.html");
