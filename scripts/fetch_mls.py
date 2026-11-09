#!/usr/bin/env python3
"""Downloads part of Multilingual LibriSpeech English (parler-tts/mls_eng on
HuggingFace, 1416 train shards of ~0.5 GB, ~31 h each) and unpacks it into one
Ogg Opus file per utterance plus one transcript file per shard:

    <out>/audio/<speaker>/<book>/<id>.opus
    <out>/transcripts/<shard>.tsv     id \t seconds \t speaker \t transcript \t begin

`begin` is where the utterance starts in its book recording, in seconds, so
consecutive utterances of a book can be read in order.

Shards are taken spread evenly over the whole train split, so the speakers are
too. Resumable: a shard whose .tsv exists is done. Stops before the disk holds
less than --keep-free GB.

    python -I scripts/fetch_mls.py --out DIR --shards 500 [--workers 2]

Needs pyarrow.
"""

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq

BASE = "https://huggingface.co/datasets/parler-tts/mls_eng/resolve/main/data"
TRAIN_SHARDS = 1416


def shard_names(count):
    picks = sorted({round(k * TRAIN_SHARDS / count) for k in range(count)})
    names = ["dev-00000-of-00001"]
    names += [f"train-{i:05d}-of-{TRAIN_SHARDS:05d}" for i in picks if i < TRAIN_SHARDS]
    return names


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def fetch(name, out, keep_free):
    tsv = os.path.join(out, "transcripts", f"{name}.tsv")
    if os.path.exists(tsv):
        return f"{name}: done before"
    if free_gb(out) < keep_free + 1.0:
        return f"{name}: skipped, disk nearly full"
    part = os.path.join(out, "tmp", f"{name}.parquet")
    r = subprocess.run(
        ["curl", "-sSfL", "--retry", "8", "--retry-delay", "10", "-C", "-", "-o", part, f"{BASE}/{name}.parquet"]
    )
    if r.returncode != 0:
        return f"{name}: download failed ({r.returncode})"
    rows, seconds = 0, 0.0
    lines = []
    pf = pq.ParquetFile(part)
    for batch in pf.iter_batches(batch_size=512):
        for row in batch.to_pylist():
            audio = row["audio"]
            uid = os.path.splitext(audio["path"])[0]
            spk, book = row["speaker_id"], row["book_id"]
            d = os.path.join(out, "audio", spk, book)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{uid}.opus"), "wb") as f:
                f.write(audio["bytes"])
            text = " ".join(row["transcript"].split())
            lines.append(f"{uid}\t{row['audio_duration']:.3f}\t{spk}\t{text}\t{row['begin_time']:.2f}\n")
            rows += 1
            seconds += row["audio_duration"]
    with open(tsv + ".part", "w") as f:
        f.writelines(lines)
    os.replace(tsv + ".part", tsv)
    os.remove(part)
    return f"{name}: {rows} utterances, {seconds / 3600:.1f} h, {free_gb(out):.0f} GB free"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", type=int, default=500)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--keep-free", type=float, default=40.0)
    a = ap.parse_args()
    for sub in ("audio", "transcripts", "tmp"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    names = shard_names(a.shards)
    print(f"{len(names)} shards to {a.out}", flush=True)
    with ThreadPoolExecutor(a.workers) as pool:
        for msg in pool.map(lambda n: fetch(n, a.out, a.keep_free), names):
            print(msg, flush=True)


if __name__ == "__main__":
    sys.exit(main())
