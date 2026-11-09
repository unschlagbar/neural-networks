#!/usr/bin/env python3
"""Reads assistant questions aloud with Piper's multi-speaker LibriTTS-R voice,
so the speech encoder hears the kind of thing Jarvis is asked, not only books.

Questions are the first user turn of conversations in an SFT JSONL (the
`{"messages": [...]}` shape), kept when they are short spoken-style plain text
a speaker reads one way: letters and basic punctuation only, no digits, code,
hyphens or line breaks. With `--text` instead, one such sentence is drawn from
each of random documents of `<|endoftext|>`-separated pretraining shards, and
the document is its context. Each take gets a random speaker and tempo.

Writes the voice recorder's layout, so `audio::voice::load` reads it:

    <out>/piper/metadata.jsonl   one line per take (id, file_name, text, spoken, session, doc)
    <out>/piper/<session>/<id>.wav   16 kHz mono
    <out>/docs.jsonl             {"doc", "text"}: the question, a blank line, the
                                 answer; or the document the sentence came from

`session` is `spk<speaker id>`, so holding out sessions holds out voices.
Resumable: ids already in metadata.jsonl are skipped.

    python -I scripts/tts_questions.py --sft data/mix/assistant_full.jsonl \
        --voice ~/training_data/tts/en_US-libritts_r-medium.onnx --out DIR -n 100
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re
import subprocess

import numpy as np

PLAIN = re.compile(r"^[A-Za-z ,.?!']+$")
MAX_ANSWER_WORDS = 400


def questions(path, scan, min_words, max_words):
    """(question, answer) pairs from the first `scan` records of `path`."""
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= scan:
                break
            try:
                msgs = json.loads(line).get("messages") or []
            except json.JSONDecodeError:
                continue
            turns = [m for m in msgs if m.get("role") in ("user", "assistant")]
            if len(turns) < 2 or turns[0]["role"] != "user" or turns[1]["role"] != "assistant":
                continue
            q = turns[0]["content"].strip()
            if "\n" in q or not PLAIN.match(q):
                continue
            if not min_words <= len(q.split()) <= max_words:
                continue
            a = " ".join(turns[1]["content"].split(" ")[:MAX_ANSWER_WORDS]).strip()
            out.append((q, a))
    return out


SENTENCE_END = re.compile(r"(?<=[.?!])\s+(?=[A-Z])")
TYPOGRAPHIC = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"'})
MAX_DOC_CHARS = 20_000


def sentences(pattern, rng, want, min_words, max_words):
    """(sentence, document) pairs, one per random document of the shards."""
    paths = sorted(glob.glob(pattern))
    out = []
    while len(out) < want:
        path = rng.choice(paths)
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(rng.randrange(max(1, size - (1 << 20))))
            chunk = f.read(1 << 20).decode("utf-8", "ignore")
        for doc in chunk.split("<|endoftext|>")[1:-1]:
            doc = doc.strip()[:MAX_DOC_CHARS]
            flat = " ".join(doc.translate(TYPOGRAPHIC).split())
            picks = [
                s for s in SENTENCE_END.split(flat)
                if PLAIN.match(s) and min_words <= len(s.split()) <= max_words
            ]
            if picks:
                out.append((rng.choice(picks), doc))
    return out


def spoken(text):
    return " ".join(re.sub(r"[^a-z' ]", " ", text.lower()).split())


def take_id(text):
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def to_wav16k(audio, rate, path):
    pcm = np.clip(audio, -1.0, 1.0).astype(np.float32).tobytes()
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(rate), "-ac", "1", "-i", "-",
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", path],
        input=pcm,
        check=True,
    )


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sft", help="SFT JSONL: read first user turns")
    src.add_argument("--text", help="glob of pretraining shards: read one sentence per document")
    ap.add_argument("--voice", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--scan", type=int, default=200_000)
    ap.add_argument("--min-words", type=int, default=4)
    ap.add_argument("--max-words", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    root = os.path.join(a.out, "piper")
    os.makedirs(root, exist_ok=True)
    meta_path = os.path.join(root, "metadata.jsonl")
    done = set()
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            done = {json.loads(l)["id"].split("/")[-1] for l in f}

    rng = random.Random(a.seed)
    if a.sft:
        pool = questions(a.sft, a.scan, a.min_words, a.max_words)
        rng.shuffle(pool)
        source = "tts_question"
        print(f"{len(pool)} usable questions in the first {a.scan} records", flush=True)
    else:
        pool = sentences(a.text, rng, a.n * 2, a.min_words, a.max_words)
        source = "tts_text"
        print(f"{len(pool)} sentences drawn from {a.text}", flush=True)

    from piper import PiperVoice, SynthesisConfig

    voice = PiperVoice.load(a.voice)
    speakers = voice.config.num_speakers
    rate = voice.config.sample_rate
    made = 0
    with open(meta_path, "a") as meta, open(os.path.join(a.out, "docs.jsonl"), "a") as docs:
        for q, ans in pool:
            if made >= a.n:
                break
            tid = take_id(q)
            if tid in done:
                continue
            spk = rng.randrange(speakers)
            cfg = SynthesisConfig(speaker_id=spk, length_scale=rng.uniform(0.9, 1.15))
            audio = np.concatenate([c.audio_float_array for c in voice.synthesize(q, syn_config=cfg)])
            session = f"spk{spk}"
            os.makedirs(os.path.join(root, session), exist_ok=True)
            file_name = f"{session}/{tid}.wav"
            to_wav16k(audio, rate, os.path.join(root, file_name))
            meta.write(json.dumps({
                "file_name": file_name,
                "id": f"{session}/{tid}",
                "source": source,
                "text": q,
                "spoken": spoken(q),
                "seconds": round(len(audio) / rate, 2),
                "speaker": "piper",
                "session": session,
                "doc": tid,
            }) + "\n")
            context = f"{q}\n\n{ans}" if a.sft else ans
            docs.write(json.dumps({"doc": tid, "text": context}) + "\n")
            done.add(tid)
            made += 1
            if made % 50 == 0:
                print(f"{made} takes", flush=True)
    print(f"{made} new takes in {root}", flush=True)


if __name__ == "__main__":
    main()
