#!/usr/bin/env python3
"""Dump the Python->Rust rewrites the build has produced so far.

The build only writes its corpus at the very end, but every rewrite is already
on disk in the LLM cache the moment it comes back. This reads that cache and
writes what is readable out of it, so the translation can be judged while the
run is still going.

    python3 scripts/dump_rustified.py [cache_dir] [out_prefix]

Writes <out_prefix>.jsonl (one rewritten conversation per line, same shape as
the corpus) and <out_prefix>.txt (the same thing laid out to read).
"""
import json
import pathlib
import sys

cache = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/llm-cache")
out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "data/old/rustified_preview")


def messages(text):
    """The rewrite reply, if this cache entry is one."""
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j < i:
        return None
    try:
        d = json.loads(text[i : j + 1])
    except json.JSONDecodeError:
        return None
    ms = d.get("messages")
    if not isinstance(ms, list) or not ms:
        return None
    if not all(isinstance(m, dict) and "role" in m and "content" in m for m in ms):
        return None
    return ms


files = sorted(cache.glob("*.txt"))
convos, skipped = [], 0
for f in files:
    ms = messages(f.read_text(encoding="utf-8", errors="replace"))
    if ms is None:
        skipped += 1
        continue
    convos.append((f.name, ms))

out.parent.mkdir(parents=True, exist_ok=True)
with open(f"{out}.jsonl", "w", encoding="utf-8") as jl, open(
    f"{out}.txt", "w", encoding="utf-8"
) as tx:
    for name, ms in convos:
        jl.write(json.dumps({"messages": ms}, ensure_ascii=False) + "\n")
        tx.write(f"{'=' * 78}\n{name}\n{'=' * 78}\n")
        for m in ms:
            tx.write(f"\n--- {m['role']} ---\n{m['content']}\n")
        tx.write("\n")

# Whatever is still Python is the thing worth looking at first.
leaked = sum(
    1
    for _, ms in convos
    if any(
        s in m["content"] for m in ms for s in ("```python", "```py\n", "Python", "python")
    )
)
rust = sum(1 for _, ms in convos if any("```rust" in m["content"] for m in ms))
print(f"cache entries: {len(files)}  (skipped {skipped} that are not rewrites)")
print(f"rewritten conversations: {len(convos)}")
print(f"  containing ```rust : {rust}")
print(f"  still mentioning Python anywhere: {leaked}")
print(f"\nwrote {out}.jsonl and {out}.txt")
