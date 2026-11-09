#!/usr/bin/env python3
"""Word ends by forced alignment with torchaudio's MMS_FA (wav2vec2 trained for
alignment), in place of our own encoder's CTC alignment.

  # how close it comes to LibriSpeech's MFA word ends (from the repo root: the
  # index holds paths relative to it)
  python -I scripts/mms_align.py --librispeech ~/training_data/librispeech/index/dev-clean.tsv -n 500

  # every take of a set in the voice recorder layout; the alignment it replaces
  # is kept as alignment.<tag>.jsonl
  python -I scripts/mms_align.py --takes ~/training_data/tts_text --keep j5

  # new takes only, their lengths added to the trainer's cache so it need not
  # open every file again at start-up
  python -I scripts/mms_align.py --takes ~/training_data/common_voice --only-new \
      --lengths ~/training_data/librispeech/lengths.tsv --lengths-dir ../../training_data/common_voice

Writes `<dir>/<speaker>/alignment.jsonl` as `{"id", "ends"}` lines, the format
`audio::voice` reads. A take whose words cannot be spelled in MMS's alphabet
keeps the alignment it had. Needs `~/training_data/venv-tts` (torch, torchaudio).
"""

import argparse
import collections
import concurrent.futures as cf
import json
import os
import itertools
import shutil
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torchaudio.pipelines import MMS_FA

SR = 16_000
LABELS = set(MMS_FA.get_labels()) - {"-", "*"}
# Longer takes are skipped: training drops those past `AUDIO_MAX_TAKE_SECONDS` (25).
MAX_SECONDS = 30


def spellable(words):
    out = []
    for w in words:
        w = "".join(c for c in w.lower() if c in LABELS)
        if not w:
            return None
        out.append(w)
    return out


# (samples, rate) of every file read, as recorded, for the trainer's length cache.
LENGTHS = {}


def read(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    LENGTHS[path] = (len(audio), sr)
    audio = audio.mean(axis=1)
    if sr != SR:
        audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, SR).numpy()
    return audio


class Aligner:
    def __init__(self, shift=0.0):
        self.shift = shift
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            # Next to a training run: the allocator's cache grew to 2.7 GB per
            # process on its own and pushed the trainer out of memory.
            torch.cuda.set_per_process_memory_fraction(0.25)
        self.model = MMS_FA.get_model(with_star=False).to(self.device).half().eval()
        self.tokenizer = MMS_FA.get_tokenizer()
        self.aligner = MMS_FA.get_aligner()

    @torch.inference_mode()
    def ends(self, audio, words):
        """Each word's end in seconds, or None."""
        if len(audio) > MAX_SECONDS * SR:
            return None
        wave = torch.from_numpy(audio).to(self.device).half()[None]
        try:
            emission, _ = self.model(wave)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None
        emission = emission.float()
        frames = emission.size(1)
        tokens = self.tokenizer(words)
        if sum(len(t) for t in tokens) > frames:
            return None
        try:
            spans = self.aligner(emission[0], tokens)
        except Exception:
            return None
        ratio = len(audio) / frames / SR
        seconds = len(audio) / SR
        ends = [min(s[-1].end * ratio + self.shift, seconds) for s in spans]
        # Shifted ends stay in order: no word may end before the next starts.
        starts = [s[0].start * ratio for s in spans]
        return [min(e, starts[i + 1]) if i + 1 < len(ends) else e for i, e in enumerate(ends)]


def librispeech(args, al):
    lines = [l.rstrip("\n").split("\t") for l in open(args.librispeech) if not l.startswith("#")]
    lines = lines[: args.n]
    errs = []
    t0 = time.time()
    for _, audio, _, words in lines:
        toks = words.split(" ")
        spoken = toks[0::3]
        mfa = [float(e) for e in toks[2::3]]
        w = spellable(spoken)
        if w is None:
            continue
        e = al.ends(read(audio), w)
        if e is None:
            continue
        errs.extend(a - b for a, b in zip(e, mfa))
    signed = np.array(errs)
    errs = np.abs(signed)
    frame = 0.04
    print(f"{len(lines)} utterances, {len(errs)} words, {time.time() - t0:.0f}s")
    print(f"word end vs MFA: mean {errs.mean() * 1000:.0f} ms, median {np.median(errs) * 1000:.0f} ms, "
          f"bias {signed.mean() * 1000:+.0f} ms (median {np.median(signed) * 1000:+.0f} ms)")
    for k in [0.5, 1, 2, 3, 5]:
        print(f"  within {k:g} encoder frames ({k * frame * 1000:.0f} ms): {100 * (errs <= k * frame).mean():.1f}%")


def takes(args, al):
    for speaker in sorted(os.listdir(args.takes)):
        root = os.path.join(args.takes, speaker)
        meta = os.path.join(root, "metadata.jsonl")
        if not os.path.isfile(meta):
            continue
        rows = []
        # A downloader may still be appending: a cut last line is left for the next run.
        for l in open(meta):
            try:
                rows.append(json.loads(l))
            except json.JSONDecodeError:
                pass
        out = os.path.join(root, "alignment.jsonl")
        old = {}
        if os.path.isfile(out):
            for l in open(out):
                r = json.loads(l)
                old[r["id"]] = r["ends"]
            kept_as = os.path.join(root, f"alignment.{args.keep}.jsonl")
            # A rerun must not overwrite the original with its own output.
            if args.keep and not os.path.exists(kept_as):
                shutil.copy(out, kept_as)
        # Results so far, appended as they come, so an interrupted run resumes.
        partial = os.path.join(root, "alignment.partial.jsonl")
        new = {}
        if os.path.isfile(partial):
            for l in open(partial):
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue
                new[r["id"]] = r["ends"]
        todo = [r for r in rows if r["id"] not in new and not (args.only_new and r["id"] in old)]
        done, t0 = len(rows) - len(todo), time.time()
        if done:
            print(f"  {speaker}: resuming, {done} already aligned", flush=True)

        def load(r):
            w = spellable(r["spoken"].split())
            if w is None:
                return r, None, None
            try:
                return r, read(os.path.join(root, r["file_name"])), w
            except Exception:
                return r, None, None

        # At most `2 · workers` decoded takes wait for the GPU at any time:
        # `Executor.map` would submit every take at once and hold their audio.
        with cf.ThreadPoolExecutor(args.workers) as pool, open(partial, "a") as part:
            pending = collections.deque()
            it = iter(todo)
            for r in itertools.islice(it, 2 * args.workers):
                pending.append(pool.submit(load, r))
            while pending:
                r, audio, w = pending.popleft().result()
                nxt = next(it, None)
                if nxt is not None:
                    pending.append(pool.submit(load, nxt))
                e = al.ends(audio, w) if audio is not None else None
                del audio
                if e is not None:
                    new[r["id"]] = " ".join(f"{x:.2f}" for x in e)
                    part.write(json.dumps({"id": r["id"], "ends": new[r["id"]]}) + "\n")
                done += 1
                if done % 2000 == 0:
                    part.flush()
                    print(f"  {speaker}: {done}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
        kept = 0
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            for r in rows:
                ends = new.get(r["id"])
                if ends is None:
                    ends = old.get(r["id"])
                    kept += ends is not None
                if ends is not None:
                    f.write(json.dumps({"id": r["id"], "ends": ends}) + "\n")
        os.replace(tmp, out)
        os.remove(partial)
        if args.lengths:
            # Keyed the way the trainer joins its take dir, speaker and file name.
            prefix = os.path.join(args.lengths_dir or args.takes, speaker)
            with open(args.lengths, "a") as f:
                for r in rows:
                    l = LENGTHS.pop(os.path.join(root, r["file_name"]), None)
                    if l:
                        f.write(f"{os.path.join(prefix, r['file_name'])}\t{l[0]}\t{l[1]}\n")
        print(f"{root}: {len(new)} of {len(rows)} aligned by MMS, {kept} kept their old alignment "
              f"({time.time() - t0:.0f}s)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--librispeech")
    p.add_argument("-n", type=int, default=500)
    p.add_argument("--takes")
    p.add_argument("--keep", help="keep the replaced alignment as alignment.<keep>.jsonl")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lengths", help="append each file's (samples, rate) to this length cache")
    p.add_argument("--lengths-dir", help="the take dir as the trainer names it (e.g. ../../training_data/common_voice)")
    p.add_argument("--only-new", action="store_true",
                   help="leave takes that already have an alignment as they are")
    p.add_argument("--shift", type=float, default=0.0,
                   help="seconds added to every word end (MMS ends ~30 ms before MFA), "
                        "never past the next word's start")
    args = p.parse_args()
    al = Aligner(args.shift)
    if args.librispeech:
        librispeech(args, al)
    if args.takes:
        takes(args, al)


if __name__ == "__main__":
    sys.exit(main())
