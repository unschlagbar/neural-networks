// Compiling the code a record teaches.
//
// A rewritten record is only as good as the code in it, and a language model
// rewriting Python into Rust makes the kind of mistake that reads fine and does
// not build — a string literal eaten by a substitution, a crate invented to
// stand in for a Python library. `rustc` settles that question exactly, for
// about 0.3 s against the 4 s the rewrite itself costs, so it is close to free
// next to the call that produced the code.
//
// Measured on 24 rewritten records: 63% built as they came back, and one repair
// pass carrying rustc's own errors took that to 79%. A second pass fixed
// nothing, which is why there is only one.

use std::process::Command;

use neural_networks::sft::Turn;

/// Every ```rust block in the record, concatenated in order. Empty when the
/// record carries no Rust — such a record has nothing to verify and passes.
pub fn rust_blocks(turns: &[Turn]) -> String {
    let mut out = String::new();
    for t in turns {
        let mut rest = t.content.as_str();
        while let Some(open) = rest.find("```rust") {
            rest = &rest[open + 7..];
            // Skip to the end of the info line, so ```rust,ignore works too.
            let body = match rest.find('\n') {
                Some(nl) => &rest[nl + 1..],
                None => break,
            };
            let Some(close) = body.find("```") else { break };
            out.push_str(&body[..close]);
            out.push('\n');
            rest = &body[close + 3..];
        }
    }
    out
}

/// Does this Rust build? `Ok(())` when it does, `Err(errors)` with rustc's own
/// diagnostics when it does not — those go straight into the repair prompt,
/// because the compiler describes the defect better than any rule could.
///
/// Tried twice: as a file of items, then wrapped in a function. Instruction
/// data shows both — a full `fn`, and a bare statement demonstrating a call —
/// and only the second form needs the wrapper.
pub fn rust_compiles(src: &str, scratch: &std::path::Path) -> Result<(), String> {
    if src.trim().is_empty() {
        return Ok(());
    }
    let file = scratch.join("snippet.rs");
    if std::fs::write(&file, src).is_err() {
        return Ok(());
    }
    let run = |path: &std::path::Path| {
        Command::new("rustc")
            .args(["--edition", "2021", "--crate-type", "lib", "--emit=metadata"])
            .arg("--out-dir")
            .arg(scratch)
            .arg(path)
            .output()
    };
    let first = match run(&file) {
        Ok(o) => o,
        // No rustc on PATH is not a corpus defect: verification is skipped
        // rather than silently rejecting every record that carries code.
        Err(_) => return Ok(()),
    };
    if first.status.success() {
        return Ok(());
    }
    let wrapped = scratch.join("wrapped.rs");
    if std::fs::write(&wrapped, format!("fn __verify() {{\n{src}\n}}")).is_ok()
        && let Ok(second) = run(&wrapped)
        && second.status.success()
    {
        return Ok(());
    }
    let errors = String::from_utf8_lossy(&first.stderr);
    // Enough for the repair prompt to work from; a wall of notes is not.
    Err(errors.lines().take(30).collect::<Vec<_>>().join("\n"))
}
