#!/usr/bin/env python3
"""Downloads VoxPopuli English (European Parliament speeches, many non-native
speakers; the parquet conversion of facebook/voxpopuli on HuggingFace, 30 train
files of ~3.2 GB) and unpacks it into takes in the voice recorder's layout:

    <out>/takes/metadata.jsonl      text = raw transcript, spoken = normalized one
    <out>/audio/<speaker>/<id>.<ext>

Only rows whose normalized transcript, punctuation dropped and hyphens read as
spaces, a CTC alphabet of letters, apostrophe and space spells; the normalized
one stands in for a missing raw transcript. A take's session is its speaker, so holding out sessions holds out
speakers. Resumable: a file listed in `<out>/done.txt` is skipped.

    python -I scripts/fetch_voxpopuli.py --out DIR [--files 30]

Needs pyarrow.
"""

import argparse
import json
import os
import re
import subprocess

import pyarrow.parquet as pq

BASE = "https://huggingface.co/datasets/facebook/voxpopuli/resolve/refs%2Fconvert%2Fparquet/en/train"
SPELLABLE = re.compile(r"^[a-z' ]+$")
PUNCTUATION = re.compile(r"[.,!?;:\"()]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--files", type=int, default=30)
    a = ap.parse_args()
    for sub in ("takes", "audio", "tmp"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    done_path = os.path.join(a.out, "done.txt")
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    for k in range(a.files):
        name = f"{k:04d}.parquet"
        if name in done:
            continue
        part = os.path.join(a.out, "tmp", name)
        r = subprocess.run(["curl", "-sSfL", "--retry", "20", "--retry-delay", "15", "-C", "-", "-o", part, f"{BASE}/{name}"])
        if r.returncode != 0:
            print(f"{name}: download failed ({r.returncode})", flush=True)
            continue
        kept = 0
        rejected = {}
        with open(os.path.join(a.out, "takes", "metadata.jsonl"), "a", buffering=1) as meta:
            for batch in pq.ParquetFile(part).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    normalized = PUNCTUATION.sub(" ", (row.get("normalized_text") or "").lower().replace("-", " "))
                    spoken = " ".join(normalized.split())
                    text = " ".join((row.get("raw_text") or "").split()) or spoken[:1].upper() + spoken[1:]
                    reason = None if spoken and text else "empty"
                    if reason is None and not SPELLABLE.match(spoken):
                        reason = "unspellable"
                    if reason:
                        rejected[reason] = rejected.get(reason, 0) + 1
                        if rejected[reason] <= 3:
                            print(f"  {reason}: {row.get('normalized_text')!r:.120} | {text!r:.80}", flush=True)
                        continue
                    audio = row["audio"]
                    ext = os.path.splitext(audio.get("path") or "")[1] or ".wav"
                    spk = str(row.get("speaker_id") or "unknown")
                    uid = row["audio_id"]
                    d = os.path.join(a.out, "audio", spk)
                    os.makedirs(d, exist_ok=True)
                    with open(os.path.join(d, f"{uid}{ext}"), "wb") as f:
                        f.write(audio["bytes"])
                    meta.write(json.dumps({
                        "file_name": os.path.join("..", "audio", spk, f"{uid}{ext}"),
                        "id": f"{spk}/{uid}",
                        "source": "voxpopuli",
                        "text": text,
                        "spoken": spoken,
                        "speaker": "voxpopuli",
                        "session": spk,
                        "doc": "",
                        "accent": row.get("accent") or "",
                        "gender": row.get("gender") or "",
                    }) + "\n")
                    kept += 1
        os.remove(part)
        with open(done_path, "a") as f:
            f.write(name + "\n")
        print(f"{name}: {kept} takes, rejected {rejected}", flush=True)


if __name__ == "__main__":
    main()
