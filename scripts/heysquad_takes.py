#!/usr/bin/env python3
"""Unpacks HeySQuAD's human-read questions (76k SQuAD questions read aloud by
crowd workers on their own microphones) into takes in the voice recorder's
layout:

    <dir>/takes/metadata.jsonl   text = the question as written, spoken = normalized
    <dir>/audio/<article>/<id>.wav   16 kHz mono

The text is the `question` the speaker read, not the `transcription` column
(an ASR output: "beyond a" for Beyonce). Questions with digits are dropped,
as for Common Voice; every clip goes through ffmpeg (some are stereo,
some end mid-sample). Takes carry no document. A take's session is its SQuAD
article, which is the closest thing to a speaker the set has. Resumable: a
parquet file listed in `<dir>/done.txt` is skipped, and so is a question
already in the metadata; at most `2 · workers` clips are being converted at
a time.

    python -I scripts/heysquad_takes.py --dir ~/training_data/heysquad_human

Needs pyarrow (`~/training_data/venv`).
"""

import argparse
import collections
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import importlib.util
import subprocess

import pyarrow.parquet as pq

_spec = importlib.util.spec_from_file_location("fetch_common_voice", os.path.join(os.path.dirname(__file__), "fetch_common_voice.py"))
_cv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cv)
spoken_form = _cv.spoken_form


def write_take(audio, path):
    """Writes the clip as a 16 kHz mono 16-bit WAV through ffmpeg, which also
    repairs the shipped files whose data chunk ends mid-sample; False if it
    cannot be decoded. ffmpeg writes a file, not a pipe: on a pipe it cannot
    seek back to fill in the header's lengths."""
    tmp = path + ".tmp.wav"
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", tmp],
        input=audio,
        capture_output=True,
    )
    if r.returncode != 0:
        return False
    os.replace(tmp, path)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.dir, "takes"), exist_ok=True)
    done_path = os.path.join(a.dir, "done.txt")
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    meta_path = os.path.join(a.dir, "takes", "metadata.jsonl")
    seen = set()
    if os.path.exists(meta_path):
        seen = {json.loads(l)["id"].split("/")[-1] for l in open(meta_path) if l.strip()}
    for part in sorted(glob.glob(os.path.join(a.dir, "data", "*.parquet"))):
        name = os.path.basename(part)
        if name in done:
            continue
        kept, rejected = 0, collections.Counter()
        with open(meta_path, "a", buffering=1) as meta, cf.ThreadPoolExecutor(a.workers) as pool:
            pending = collections.deque()

            def settle(limit):
                nonlocal kept
                while len(pending) > limit:
                    fut, record = pending.popleft()
                    if fut.result():
                        meta.write(json.dumps(record) + "\n")
                        kept += 1
                    else:
                        rejected["undecodable"] += 1

            for batch in pq.ParquetFile(part).iter_batches(batch_size=64, columns=["audio", "question", "id", "context"]):
                for row in batch.to_pylist():
                    question = " ".join((row.get("question") or "").split())
                    spoken = spoken_form(question)
                    if spoken is None:
                        rejected["unspellable"] += 1
                        continue
                    audio = (row.get("audio") or {}).get("bytes")
                    if not audio:
                        rejected["no audio"] += 1
                        continue
                    uid = row.get("id") or f"{name}_{kept}"
                    if uid in seen:
                        continue
                    article = hashlib.md5((row.get("context") or "").encode()).hexdigest()[:10]
                    d = os.path.join(a.dir, "audio", article)
                    os.makedirs(d, exist_ok=True)
                    record = {
                        "file_name": os.path.join("..", "audio", article, f"{uid}.wav"),
                        "id": f"{article}/{uid}",
                        "source": "heysquad_human",
                        "text": question,
                        "spoken": spoken,
                        "speaker": "heysquad",
                        "session": article,
                        "doc": "",
                    }
                    pending.append((pool.submit(write_take, audio, os.path.join(d, f"{uid}.wav")), record))
                    settle(2 * a.workers)
            settle(0)
        with open(done_path, "a") as f:
            f.write(name + "\n")
        print(f"{name}: {kept} takes, rejected {dict(rejected)}", flush=True)


if __name__ == "__main__":
    main()
