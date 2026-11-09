#!/usr/bin/env python3
"""Writes SLURP's real recordings (people talking to a home assistant, near and
far microphone) as takes in the voice recorder's layout, so `audio::voice::load`
reads them:

    <slurp>/takes/metadata.jsonl   file_name points into <slurp>/slurp_real

Only recordings SLURP marks `correct`, and only sentences a CTC alphabet of
letters, apostrophe and space spells (no digits). A recording's session is its
timestamp, shared by the headset and far-field copy of the same utterance, so
holding out sessions holds out both.

    python -I scripts/slurp_takes.py ~/training_data/slurp
"""

import json
import os
import re
import sys

SPELLABLE = re.compile(r"^[a-z' ]+$")


def written(sentence):
    words = ["I" + w[1:] if w == "i" or w.startswith("i'") else w for w in sentence.split()]
    text = " ".join(words)
    return text[:1].upper() + text[1:]


def main():
    root = os.path.expanduser(sys.argv[1])
    out_dir = os.path.join(root, "takes")
    os.makedirs(out_dir, exist_ok=True)
    kept = 0
    with open(os.path.join(out_dir, "metadata.jsonl"), "w") as out:
        for split in ("train", "devel", "test"):
            for line in open(os.path.join(root, f"{split}.jsonl")):
                d = json.loads(line)
                sentence = " ".join(d["sentence"].lower().split())
                if not SPELLABLE.match(sentence):
                    continue
                for r in d["recordings"]:
                    if r["status"] != "correct":
                        continue
                    path = os.path.join(root, "slurp_real", r["file"])
                    if not os.path.exists(path):
                        continue
                    stem = os.path.splitext(r["file"])[0]
                    session = stem.removesuffix("-headset")
                    out.write(json.dumps({
                        "file_name": os.path.join("..", "slurp_real", r["file"]),
                        "id": f"{session}/{stem}",
                        "source": f"slurp_{split}",
                        "text": written(sentence),
                        "spoken": sentence,
                        "speaker": "slurp",
                        "session": session,
                        "doc": "",
                    }) + "\n")
                    kept += 1
    print(f"{kept} takes in {out_dir}")


if __name__ == "__main__":
    main()
