#!/usr/bin/env python3
"""Reads sentences of pretraining documents in the user's own voice, cloned by
Chatterbox from their recorded takes, in the layout `tts_questions.py --text`
writes (`<out>/clone/metadata.jsonl`, `<out>/docs.jsonl`).

The reference for each take is a random one of the user's takes outside the
held-out sessions, at least `--min-ref` seconds long, so the held-out session
stays an honest test of the real voice. Needs the venv with torch and
chatterbox-tts (`~/training_data/venv-tts`).

    python -I scripts/tts_clone.py --voice-dir ~/training_data/voice/unschlagbar \
        --held-out 2026-10-17_1534 --text 'data/mix/pretrain_v2/shard.*.txt' \
        --out ~/training_data/tts_clone -n 20
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_questions import sentences, spoken, take_id, to_wav16k  # noqa: E402


def references(voice_dir, held_out, min_seconds):
    refs = []
    for line in open(os.path.join(voice_dir, "metadata.jsonl")):
        t = json.loads(line)
        if t.get("session") in held_out or t.get("seconds", 0) < min_seconds:
            continue
        refs.append(os.path.join(voice_dir, t["file_name"]))
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice-dir", required=True)
    ap.add_argument("--held-out", nargs="*", default=[])
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--min-ref", type=float, default=4.0)
    ap.add_argument("--min-words", type=int, default=6)
    ap.add_argument("--max-words", type=int, default=40)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()

    from chatterbox.tts import ChatterboxTTS

    rng = random.Random(a.seed)
    refs = references(os.path.expanduser(a.voice_dir), set(a.held_out), a.min_ref)
    print(f"{len(refs)} reference takes", flush=True)
    root = os.path.join(a.out, "clone")
    os.makedirs(root, exist_ok=True)
    meta_path = os.path.join(root, "metadata.jsonl")
    done = set()
    if os.path.exists(meta_path):
        done = {json.loads(l)["id"].split("/")[-1] for l in open(meta_path)}
    pool = sentences(a.text, rng, a.n * 2, a.min_words, a.max_words)
    model = ChatterboxTTS.from_pretrained(device="cuda")
    made = 0
    with open(meta_path, "a", buffering=1) as meta, open(os.path.join(a.out, "docs.jsonl"), "a", buffering=1) as docs:
        for text, doc in pool:
            if made >= a.n:
                break
            tid = take_id(text)
            if tid in done:
                continue
            ref = rng.choice(refs)
            with torch.inference_mode():
                wav = model.generate(text, audio_prompt_path=ref)
            audio = wav.squeeze(0).float().cpu().numpy()
            session = "clone"
            os.makedirs(os.path.join(root, session), exist_ok=True)
            file_name = f"{session}/{tid}.wav"
            to_wav16k(audio, model.sr, os.path.join(root, file_name))
            meta.write(json.dumps({
                "file_name": file_name,
                "id": f"{session}/{tid}",
                "source": "tts_clone",
                "text": text,
                "spoken": spoken(text),
                "seconds": round(len(audio) / model.sr, 2),
                "speaker": "clone",
                "session": session,
                "doc": tid,
                "reference": os.path.basename(ref),
            }) + "\n")
            docs.write(json.dumps({"doc": tid, "text": doc}) + "\n")
            done.add(tid)
            made += 1
            print(f"{made}: {text[:80]}", flush=True)


if __name__ == "__main__":
    main()
