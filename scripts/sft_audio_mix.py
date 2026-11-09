#!/usr/bin/env python3
"""Writes the audio SFT corpus: the text SFT corpus with every user turn that
`scripts/sft_speech.py` spoke and `scripts/mms_align.py` aligned marked with
its take (`"audio": "<session>/<id>"`). Everything else stays as it was, so a
loader without the takes reads the corpus as the text corpus it came from.

    python -I scripts/sft_audio_mix.py --sft data/mix/assistant_full.jsonl \
        --speech ~/training_data/sft_speech --out data/mix/assistant_audio.jsonl
"""

import argparse
import hashlib
import json
import os


def ids(path):
    if not os.path.exists(path):
        return set()
    return {json.loads(l)["id"] for l in open(path) if l.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--speech", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    root = os.path.join(a.speech, "piper")
    rendered = ids(os.path.join(root, "metadata.jsonl"))
    aligned = ids(os.path.join(root, "alignment.jsonl"))
    ready = rendered & aligned
    seconds = {}
    for l in open(os.path.join(root, "metadata.jsonl")):
        if l.strip():
            m = json.loads(l)
            seconds[m["id"]] = m["seconds"]
    plan = {}
    for l in open(os.path.join(a.speech, "records.jsonl")):
        r = json.loads(l)
        plan[r["line"]] = r

    total = spoken = turns = stale = 0
    hours = 0.0
    tmp = a.out + ".tmp"
    with open(a.sft) as src, open(tmp, "w") as out:
        for i, line in enumerate(src):
            if not line.strip():
                continue
            total += 1
            r = plan.get(i)
            if r is None:
                out.write(line if line.endswith("\n") else line + "\n")
                continue
            if hashlib.md5(line.encode()).hexdigest()[:12] != r["hash"]:
                stale += 1
                out.write(line if line.endswith("\n") else line + "\n")
                continue
            record = json.loads(line)
            heard = 0
            for k, tid in r["turns"]:
                if tid in ready:
                    record["messages"][k]["audio"] = tid
                    heard += 1
                    hours += seconds[tid] / 3600
            spoken += heard > 0
            turns += heard
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp, a.out)
    print(f"{total} conversations, {spoken} with a heard user turn ({spoken / max(1, total):.1%}), "
          f"{turns} heard turns, {hours:.0f} h of speech; {len(rendered)} takes rendered, {len(aligned)} aligned"
          + (f"; {stale} drawn lines changed since the draw, left as text" if stale else ""))


if __name__ == "__main__":
    main()
