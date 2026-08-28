// Talking to a local OpenAI-compatible server (LM Studio, llama.cpp, Ollama's
// compat endpoint). Two jobs: generating synthetic examples that templates
// cannot phrase, and judging records against a prompt.
//
// Hand-rolled HTTP/1.1 over a TCP socket — the repo carries no HTTP client, and
// what a local chat completion needs is one POST with a `Connection: close`, so
// a real client would be a dependency for nothing. Only `http://` is supported:
// this is a server on your own machine.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

use crate::config::{Llm, Result};
use crate::json;

pub struct Client {
    pub cfg: Llm,
    host: String,
    port: u16,
    /// Path prefix from the endpoint, e.g. `/v1`.
    base: String,
    cache: Option<String>,
    pub calls: usize,
    pub cached: usize,
}

impl Client {
    pub fn new(cfg: &Llm) -> Result<Self> {
        let (host, port, base) = split_endpoint(&cfg.endpoint)?;
        let cache = if cfg.cache.is_empty() {
            None
        } else {
            std::fs::create_dir_all(&cfg.cache).map_err(|e| format!("{}: {e}", cfg.cache))?;
            Some(cfg.cache.clone())
        };
        Ok(Self {
            cfg: cfg.clone(),
            host,
            port,
            base,
            cache,
            calls: 0,
            cached: 0,
        })
    }

    /// Models the server has loaded — the reachability check, and the reason
    /// `model` may be left empty (a single-model server answers regardless).
    pub fn models(&mut self) -> Result<Vec<String>> {
        let body = self.request("GET", &format!("{}/models", self.base), None)?;
        let mut out = Vec::new();
        let mut rest = body.as_str();
        // The listing is `{"data":[{"id":"...","object":"model"},...]}`; walk the
        // `"id"` keys rather than parsing the array.
        while let Some(at) = rest.find("\"id\"") {
            rest = &rest[at + 4..];
            if let Some(colon) = rest.find(':')
                && let Some(q) = rest[colon..].find('"')
                && let Some(id) = json::decode(&rest[colon + q + 1..])
            {
                out.push(id);
            }
        }
        Ok(out)
    }

    /// One chat completion. Identical `(model, system, user, nonce)` tuples are
    /// answered from the cache — a rebuild of a mixture must not re-generate
    /// what it already generated. `nonce` is part of the key but not of the
    /// request: it is what lets a generator ask the same prompt N times and
    /// still cache each answer separately.
    pub fn chat(&mut self, system: &str, user: &str, nonce: u64) -> Result<String> {
        self.chat_as(system, user, nonce, "", -1.0)
    }

    /// [`chat`](Self::chat) with a per-source model and temperature. An empty
    /// `model` or a negative `temperature` inherits `[llm]`. Both are part of
    /// the cache key, so switching model invalidates nothing else.
    pub fn chat_as(
        &mut self,
        system: &str,
        user: &str,
        nonce: u64,
        model: &str,
        temperature: f32,
    ) -> Result<String> {
        let model = if model.is_empty() {
            self.cfg.model.clone()
        } else {
            model.to_string()
        };
        let temperature = if temperature < 0.0 {
            self.cfg.temperature
        } else {
            temperature
        };
        let key = hash(&format!(
            "{model}\u{1}{temperature}\u{1}{nonce}\u{1}{system}\u{1}{user}"
        ));
        if let Some(dir) = &self.cache
            && let Ok(hit) = std::fs::read_to_string(format!("{dir}/{key:016x}.txt"))
        {
            self.cached += 1;
            return Ok(hit);
        }

        let mut messages = String::from("[");
        if !system.trim().is_empty() {
            messages.push_str(&format!(
                "{{\"role\":\"system\",\"content\":\"{}\"}},",
                json::escape(system)
            ));
        }
        messages.push_str(&format!(
            "{{\"role\":\"user\",\"content\":\"{}\"}}]",
            json::escape(user)
        ));
        let body = format!(
            "{{\"model\":\"{}\",\"messages\":{messages},\"temperature\":{},\
             \"max_tokens\":{},\"stream\":false}}",
            json::escape(&model),
            temperature,
            self.cfg.max_tokens,
        );

        let mut last = String::new();
        for attempt in 0..=self.cfg.retries {
            match self.request("POST", &format!("{}/chat/completions", self.base), Some(&body)) {
                Ok(resp) => {
                    let content = json::field(&resp, "content").ok_or_else(|| {
                        format!("no message content in the reply: {}", head(&resp))
                    })?;
                    // Reasoning models fence their thinking; the corpus wants
                    // the answer only.
                    let content = strip_think(&content).trim().to_string();
                    self.calls += 1;
                    // An empty completion is a failure, not an answer: caching
                    // it would make every rerun reproduce the failure for free.
                    if let Some(dir) = &self.cache
                        && !content.is_empty()
                    {
                        let _ = std::fs::write(format!("{dir}/{key:016x}.txt"), &content);
                    }
                    return Ok(content);
                }
                Err(e) => {
                    last = e;
                    if attempt < self.cfg.retries {
                        std::thread::sleep(Duration::from_millis(400 * (attempt as u64 + 1)));
                    }
                }
            }
        }
        Err(format!("{} after {} tries: {last}", self.cfg.endpoint, self.cfg.retries + 1))
    }

    fn request(&self, method: &str, path: &str, body: Option<&str>) -> Result<String> {
        // Every resolved address, not just the first: `localhost` resolves to
        // ::1 before 127.0.0.1, and a server bound to IPv4 only would look
        // unreachable.
        let addrs: Vec<_> = (self.host.as_str(), self.port)
            .to_socket_addrs()
            .map_err(|e| format!("{}:{}: {e}", self.host, self.port))?
            .collect();
        if addrs.is_empty() {
            return Err(format!("{}:{} did not resolve", self.host, self.port));
        }
        let timeout = Duration::from_secs(self.cfg.timeout as u64);
        let mut sock = None;
        let mut connect_err = String::new();
        for addr in &addrs {
            match TcpStream::connect_timeout(addr, Duration::from_secs(5)) {
                Ok(s) => {
                    sock = Some(s);
                    break;
                }
                Err(e) => connect_err = e.to_string(),
            }
        }
        let mut sock = sock.ok_or_else(|| {
            format!(
                "cannot reach {} ({connect_err}) — is the LM Studio server started?",
                self.cfg.endpoint
            )
        })?;
        sock.set_read_timeout(Some(timeout)).ok();
        sock.set_write_timeout(Some(timeout)).ok();

        let mut req = format!(
            "{method} {path} HTTP/1.1\r\nHost: {}:{}\r\nAccept: application/json\r\n\
             Connection: close\r\n",
            self.host, self.port
        );
        if !self.cfg.api_key.is_empty() {
            req.push_str(&format!("Authorization: Bearer {}\r\n", self.cfg.api_key));
        }
        match body {
            Some(b) => req.push_str(&format!(
                "Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{b}",
                b.len()
            )),
            None => req.push_str("\r\n"),
        }
        sock.write_all(req.as_bytes())
            .map_err(|e| format!("send failed: {e}"))?;

        let mut raw = Vec::new();
        sock.read_to_end(&mut raw)
            .map_err(|e| format!("read failed (timeout is {}s): {e}", self.cfg.timeout))?;
        let text = String::from_utf8_lossy(&raw).into_owned();
        let (head_txt, body_txt) = text
            .split_once("\r\n\r\n")
            .ok_or_else(|| format!("malformed reply: {}", head(&text)))?;
        let status: u16 = head_txt
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let body_txt = if head_txt.to_lowercase().contains("transfer-encoding: chunked") {
            dechunk(body_txt)
        } else {
            body_txt.to_string()
        };
        if status != 200 {
            return Err(format!("HTTP {status}: {}", head(&body_txt)));
        }
        Ok(body_txt)
    }
}

/// `http://localhost:1234/v1` -> `("localhost", 1234, "/v1")`.
fn split_endpoint(ep: &str) -> Result<(String, u16, String)> {
    let rest = match ep.strip_prefix("http://") {
        Some(r) => r,
        None if ep.starts_with("https://") => {
            return Err(format!(
                "{ep}: only http:// is supported (this is a local server; \
                 put a proxy in front if you need TLS)"
            ));
        }
        None => ep,
    };
    let (hostport, path) = match rest.find('/') {
        Some(i) => (&rest[..i], rest[i..].trim_end_matches('/')),
        None => (rest, ""),
    };
    let (host, port) = match hostport.rsplit_once(':') {
        Some((h, p)) => (
            h.to_string(),
            p.parse().map_err(|_| format!("{ep}: bad port {p:?}"))?,
        ),
        None => (hostport.to_string(), 80u16),
    };
    if host.is_empty() {
        return Err(format!("{ep}: no host"));
    }
    Ok((host, port, path.to_string()))
}

/// Undo `Transfer-Encoding: chunked`: `<hex len>\r\n<data>\r\n` until a 0 chunk.
fn dechunk(body: &str) -> String {
    let mut out = String::with_capacity(body.len());
    let mut rest = body;
    loop {
        let Some((len_line, after)) = rest.split_once("\r\n") else {
            break;
        };
        let len = usize::from_str_radix(len_line.trim().split(';').next().unwrap_or(""), 16);
        match len {
            Ok(0) | Err(_) => break,
            Ok(n) if n <= after.len() => {
                out.push_str(&after[..n]);
                rest = after[n..].trim_start_matches("\r\n");
            }
            Ok(_) => {
                out.push_str(after);
                break;
            }
        }
    }
    out
}

/// Drop a `<think>...</think>` prelude (reasoning models emit one even when the
/// request asks for JSON).
fn strip_think(s: &str) -> &str {
    match s.find("</think>") {
        Some(i) => &s[i + "</think>".len()..],
        None => s,
    }
}

fn head(s: &str) -> String {
    s.chars().take(300).collect()
}

fn hash(s: &str) -> u64 {
    s.bytes().fold(0xcbf29ce484222325u64, |a, b| {
        (a ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_endpoints() {
        assert_eq!(
            split_endpoint("http://localhost:1234/v1").unwrap(),
            ("localhost".into(), 1234, "/v1".into())
        );
        assert_eq!(
            split_endpoint("127.0.0.1:8080").unwrap(),
            ("127.0.0.1".into(), 8080, "".into())
        );
        assert!(split_endpoint("https://api.example.com/v1").is_err());
    }

    #[test]
    fn dechunks_a_body() {
        assert_eq!(dechunk("4\r\n{\"a\"\r\n3\r\n:1}\r\n0\r\n\r\n"), "{\"a\":1}");
    }

    #[test]
    fn strips_reasoning() {
        assert_eq!(strip_think("<think>hmm</think>\nanswer").trim(), "answer");
        assert_eq!(strip_think("answer"), "answer");
    }
}
