#!/usr/bin/env python3
"""Downloads Common Voice 17 English (crowd-sourced read sentences on people's
own microphones, many accents; fixie-ai's parquet copy, 90 train shards of
~0.5 GB, each sentence with an LLM-written `continuation`) and unpacks it into
takes in the voice recorder's layout:

    <out>/takes/metadata.jsonl   text = the sentence, spoken = normalized
    <out>/audio/<speaker>/<id>.flac   16 kHz mono
    <out>/docs.jsonl             the sentence followed by its continuation

Each take is read out of its own document, so training scores the
continuation after the audio as it does for the user's takes. Only sentences
whose normalized form a CTC alphabet of letters, apostrophe and space spells
(no digits) are kept. A take's session is its speaker. Resumable: a shard
listed in `<out>/done.txt` is skipped; at most `2 · workers` clips are being
converted at a time.

    python -I scripts/fetch_common_voice.py --out DIR [--shards 90] [--workers 8]

Needs pyarrow and ffmpeg.
"""

import argparse
import collections
import concurrent.futures as cf
import json
import os
import re
import subprocess

import pyarrow.parquet as pq

BASE = "https://huggingface.co/datasets/fixie-ai/common_voice_17_0/resolve/main/en"
SPELLABLE = re.compile(r"^[a-z' ]+$")
DROPPED = re.compile(r"[.,!?;:\"()\[\]“”„«»…]")


def spoken_form(sentence):
    s = sentence.lower().replace("’", "'").replace("‘", "'").replace("-", " ").replace("—", " ").replace("–", " ")
    s = " ".join(DROPPED.sub(" ", s).split())
    # Quote marks around words, not apostrophes inside them.
    s = " ".join(w.strip("'") for w in s.split())
    s = " ".join(s.split())
    return s if s and SPELLABLE.match(s) else None


def to_flac(data, path):
    tmp = path + ".tmp.flac"
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", "pipe:0", "-ac", "1", "-ar", "16000", tmp],
        input=data,
        capture_output=True,
    )
    if r.returncode != 0:
        return False
    os.replace(tmp, path)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", type=int, default=90)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    for sub in ("takes", "audio", "tmp"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    done_path = os.path.join(a.out, "done.txt")
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    for k in range(a.shards):
        name = f"train-{k:05d}-of-00090.parquet"
        if name in done:
            continue
        part = os.path.join(a.out, "tmp", name)
        r = subprocess.run(["curl", "-sSfL", "--retry", "20", "--retry-delay", "15", "-C", "-", "-o", part, f"{BASE}/{name}"])
        if r.returncode != 0:
            print(f"{name}: download failed ({r.returncode})", flush=True)
            continue
        kept, rejected = 0, collections.Counter()
        meta = open(os.path.join(a.out, "takes", "metadata.jsonl"), "a", buffering=1)
        docs = open(os.path.join(a.out, "docs.jsonl"), "a", buffering=1)
        with cf.ThreadPoolExecutor(a.workers) as pool:
            pending = collections.deque()

            def settle(limit):
                nonlocal kept
                while len(pending) > limit:
                    fut, record, doc = pending.popleft()
                    if fut.result():
                        meta.write(json.dumps(record) + "\n")
                        docs.write(json.dumps(doc) + "\n")
                        kept += 1
                    else:
                        rejected["undecodable"] += 1

            for batch in pq.ParquetFile(part).iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    sentence = " ".join((row.get("sentence") or "").split())
                    spoken = spoken_form(sentence)
                    if spoken is None:
                        rejected["unspellable"] += 1
                        continue
                    if (row.get("down_votes") or 0) > (row.get("up_votes") or 0):
                        rejected["voted down"] += 1
                        continue
                    audio = (row.get("audio") or {}).get("bytes")
                    if not audio:
                        rejected["no audio"] += 1
                        continue
                    spk = (row.get("client_id") or "unknown")[:16]
                    uid = os.path.splitext(os.path.basename(row.get("path") or ""))[0] or f"{k}_{kept}"
                    d = os.path.join(a.out, "audio", spk)
                    os.makedirs(d, exist_ok=True)
                    record = {
                        "file_name": os.path.join("..", "audio", spk, f"{uid}.flac"),
                        "id": f"{spk}/{uid}",
                        "source": "common_voice",
                        "text": sentence,
                        "spoken": spoken,
                        "speaker": "common_voice",
                        "session": spk,
                        "doc": uid,
                        "accent": row.get("accent") or "",
                        "gender": row.get("gender") or "",
                    }
                    cont = " ".join((row.get("continuation") or "").split())
                    doc = {"doc": uid, "text": f"{sentence} {cont}".strip()}
                    pending.append((pool.submit(to_flac, audio, os.path.join(d, f"{uid}.flac")), record, doc))
                    settle(2 * a.workers)
            settle(0)
        meta.close()
        docs.close()
        os.remove(part)
        with open(done_path, "a") as f:
            f.write(name + "\n")
        print(f"{name}: {kept} takes, rejected {dict(rejected)}", flush=True)


if __name__ == "__main__":
    main()
