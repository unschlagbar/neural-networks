#!/usr/bin/env python3
"""Convert allenai/WildChat-1M parquet shards into `{"messages": [...]}` JSONL.

Only the metadata cuts the repo's parquet reader cannot make live here: the
`toxic` and `redacted` flags are BOOLEAN columns, and each turn carries its own
`language`. Kept: English conversations (every turn), not toxic, not redacted.
Every text-level filter stays in datamix, so this runs once per download —
except one: a first user turn whose opening repeats is a script driving the
API (AutoGPT loops, SEO content-farm templates), not a person, so at most
PREFIX_CAP conversations are kept per opening. datamix dedups whole records
and cannot see that.

Output is split by the answering model, because GPT-4's replies are the better
half and a mixture may want to weight them differently:
  <out_dir>/wildchat_gpt4.jsonl, <out_dir>/wildchat_gpt35.jsonl

Needs pyarrow (python -m venv .venv && .venv/bin/pip install pyarrow).
usage: wildchat_to_jsonl.py <shard_dir> <out_dir>
"""
import glob, json, os, sys

import pyarrow.parquet as pq

PREFIX_CAP = 2
PREFIX_BYTES = 60
COLS = ["conversation", "model", "language", "toxic", "redacted"]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    outs = {
        "gpt4": open(os.path.join(dst, "wildchat_gpt4.jsonl"), "w"),
        "gpt35": open(os.path.join(dst, "wildchat_gpt35.jsonl"), "w"),
    }
    read = kept = 0
    openings = {}
    for path in sorted(glob.glob(os.path.join(src, "*.parquet"))):
        f = pq.ParquetFile(path)
        for batch in f.iter_batches(columns=COLS, batch_size=4096):
            for r in batch.to_pylist():
                read += 1
                if r["language"] != "English" or r["toxic"] or r["redacted"]:
                    continue
                conv = r["conversation"]
                if any(t.get("language") not in (None, "English") for t in conv):
                    continue
                turns = [{"role": t["role"], "content": t["content"]} for t in conv]
                while turns and turns[-1]["role"] != "assistant":
                    turns.pop()
                if len(turns) < 2:
                    continue
                opening = " ".join(turns[0]["content"].split())[:PREFIX_BYTES].lower()
                openings[opening] = openings.get(opening, 0) + 1
                if openings[opening] > PREFIX_CAP:
                    continue
                key = "gpt4" if r["model"].startswith("gpt-4") else "gpt35"
                outs[key].write(json.dumps({"messages": turns, "category": "wildchat_" + key}) + "\n")
                kept += 1
        print(f"{os.path.basename(path)}: {kept} of {read} kept so far", flush=True)
    for o in outs.values():
        o.close()


if __name__ == "__main__":
    main()
