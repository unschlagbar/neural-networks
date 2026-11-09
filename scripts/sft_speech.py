#!/usr/bin/env python3
"""Speaks the user turns of a share of an SFT corpus's conversations with
Piper's multi-speaker LibriTTS-R voice, so SFT can train the assistant on
heard questions with the same answers the text corpus has.

A conversation is drawn (by a hash of its line, so the draw is the same on
every run) among those with at least one speakable user turn, with the
probability that makes `--share` of all conversations spoken. Every speakable
user turn of a drawn conversation is read, all by the same speaker; the other
turns stay typed. Speakable is one paragraph of at most `--max-words` words of
plain prose: no code, markup or symbols a speaker would have to choose a
reading for. Numbers, percentages, amounts, ordinals and years are written out
first (`say_form`), so the transcript is what was said.

Writes the voice recorder's layout, so `audio::voice::load` reads it, plus the
map from conversations to takes that `scripts/sft_audio_mix.py` reads:

    <out>/piper/metadata.jsonl   text = the turn as read, spoken = normalized
    <out>/piper/<session>/<id>.flac   16 kHz mono
    <out>/records.jsonl          {"line", "hash", "turns": [[message index, take id], ...]}

Resumable: takes already in metadata.jsonl are skipped. At most `4 · workers`
takes are in flight.

    python -I scripts/sft_speech.py --sft data/mix/assistant_full.jsonl \
        --voice ~/training_data/tts/en_US-libritts_r-medium.onnx --out ~/training_data/sft_speech
"""

import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import os
import re
import subprocess

import numpy as np

PLAIN = re.compile(r"^[A-Za-z0-9 ,.?!'\"%$&:;()-]+$")
TYPOGRAPHIC = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
SPELLABLE = re.compile(r"^[a-z' ]+$")
DROPPED = re.compile(r"[.,!?;:\"()]")

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
SCALES = [(10**9, "billion"), (10**6, "million"), (1000, "thousand")]
ORDINAL = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}


def below_thousand(n):
    words = []
    if n >= 100:
        words += [ONES[n // 100], "hundred"]
        n %= 100
    if n >= 20:
        words.append(TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10]))
    elif n > 0 or not words:
        words.append(ONES[n])
    return " ".join(words)


def cardinal(n):
    if n >= 10**12:
        return None
    words = []
    for scale, name in SCALES:
        if n >= scale:
            words += [below_thousand(n // scale), name]
            n %= scale
    if n or not words:
        words.append(below_thousand(n))
    return " ".join(words)


def year(n):
    """A four-digit number as a year is read, or None."""
    if 2000 <= n <= 2009:
        return cardinal(n)
    if not 1100 <= n <= 2099:
        return None
    hi, lo = divmod(n, 100)
    if lo == 0:
        return f"{cardinal(hi)} hundred"
    return f"{cardinal(hi)} {'oh ' + ONES[lo] if lo < 10 else cardinal(lo)}"


def ordinal(n):
    words = cardinal(n).split()
    last = words[-1]
    if last in ORDINAL:
        last = ORDINAL[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return " ".join(words[:-1] + [last])


NUMBER = re.compile(r"(\$?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(%|st|nd|rd|th)?(?![\w])")


def say_form(text):
    """`text` with every number written out as it is read, or None when a
    number has no single reading (a fraction, a time, digits inside a word)."""

    def number(m):
        dollar, whole, frac, suffix = m.group(1), m.group(2), m.group(3), m.group(4)
        n = int(whole.replace(",", ""))
        if suffix in ("st", "nd", "rd", "th"):
            if dollar or frac:
                raise ValueError
            return ordinal(n)
        if frac and dollar:
            cents = frac[1:]
            if len(cents) != 2:
                raise ValueError
            return f"{cardinal(n)} dollars and {cardinal(int(cents))} cents"
        if frac:
            words = f"{cardinal(n)} point " + " ".join(ONES[int(d)] for d in frac[1:])
        elif not dollar and suffix is None and len(whole) == 4 and year(n):
            words = year(n)
        else:
            words = cardinal(n)
        if words is None:
            raise ValueError
        if dollar:
            words += " dollar" if n == 1 else " dollars"
        if suffix == "%":
            words += " percent"
        return words

    if re.search(r"\d[:/]\d", text):
        return None
    try:
        out = NUMBER.sub(number, text)
    except ValueError:
        return None
    out = out.replace("&", " and ").replace("%", " percent").replace("$", "")
    out = " ".join(out.split())
    return None if any(c.isdigit() for c in out) else out


def spoken_form(said):
    s = said.lower().replace("-", " ")
    s = " ".join(DROPPED.sub(" ", s).split())
    s = " ".join(w.strip("'") for w in s.split())
    s = " ".join(s.split())
    return s if s and SPELLABLE.match(s) else None


def speakable(content, min_words, max_words):
    """(as read, spoken form) of a user turn, or None."""
    t = content.strip().translate(TYPOGRAPHIC)
    if "\n" in t or not PLAIN.match(t) or not min_words <= len(t.split()) <= max_words:
        return None
    said = say_form(t)
    spoken = said and spoken_form(said)
    return (said, spoken) if spoken else None


def unit(text):
    """A number in [0, 1) from `text`, the same on every run."""
    return int(hashlib.md5(text.encode()).hexdigest()[:12], 16) / 16**12


def plan(path, share, min_words, max_words, speakers):
    """The drawn conversations: (line, hash, speaker, [(message index, said, spoken)])."""
    candidates, total = [], 0
    with open(path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            total += 1
            try:
                msgs = json.loads(line)["messages"]
            except (json.JSONDecodeError, KeyError):
                continue
            turns = []
            for k, m in enumerate(msgs):
                if m.get("role") != "user":
                    continue
                s = speakable(m.get("content", ""), min_words, max_words)
                if s:
                    turns.append((k, *s))
            if turns:
                h = hashlib.md5(line.encode()).hexdigest()[:12]
                candidates.append((i, h, turns))
    p = min(1.0, share * total / max(1, len(candidates)))
    drawn = [(i, h, int(h, 16) % speakers, turns) for i, h, turns in candidates if unit("draw" + h) < p]
    print(f"{total} conversations, {len(candidates)} with a speakable user turn; drawing {p:.3f} of those: "
          f"{len(drawn)} conversations ({len(drawn) / total:.1%}), {sum(len(t) for *_, t in drawn)} turns", flush=True)
    return drawn


VOICE = None


def init(voice_path):
    global VOICE
    import onnxruntime
    from piper import PiperVoice

    VOICE = PiperVoice.load(voice_path)
    # One thread per worker: Piper's own session takes every core, and a pool
    # of them oversubscribes the CPU many times over.
    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    VOICE.session = onnxruntime.InferenceSession(voice_path, sess_options=opts, providers=["CPUExecutionProvider"])


def render(job):
    """Synthesizes one take to FLAC; (job, seconds) or (job, None)."""
    from piper import SynthesisConfig

    tid, said, spk, tempo, path = job
    try:
        cfg = SynthesisConfig(speaker_id=spk, length_scale=tempo)
        audio = np.concatenate([c.audio_float_array for c in VOICE.synthesize(said, syn_config=cfg)])
        rate = VOICE.config.sample_rate
        pcm = np.clip(audio, -1.0, 1.0).astype(np.float32).tobytes()
        tmp = path + ".tmp.flac"
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(rate), "-ac", "1", "-i", "-",
             "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", "-c:a", "flac", tmp],
            input=pcm,
            capture_output=True,
        )
        if r.returncode != 0:
            return job, None
        os.replace(tmp, path)
        return job, round(len(audio) / rate, 2)
    except Exception:
        return job, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--share", type=float, default=0.2)
    ap.add_argument("--min-words", type=int, default=2)
    ap.add_argument("--max-words", type=int, default=55)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--speakers", type=int, default=904)
    a = ap.parse_args()

    root = os.path.join(a.out, "piper")
    os.makedirs(root, exist_ok=True)
    drawn = plan(a.sft, a.share, a.min_words, a.max_words, a.speakers)

    jobs, records = {}, []
    for line, h, spk, turns in drawn:
        ids = []
        for k, said, spoken in turns:
            tid = f"spk{spk}/{hashlib.sha1(said.encode()).hexdigest()[:16]}"
            ids.append([k, tid])
            if tid not in jobs:
                tempo = 0.88 + 0.3 * unit("tempo" + tid)
                jobs[tid] = (tid, said, spk, round(tempo, 3), os.path.join(root, tid + ".flac"), spoken)
        records.append({"line": line, "hash": h, "turns": ids})
    tmp = os.path.join(a.out, "records.jsonl.tmp")
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, os.path.join(a.out, "records.jsonl"))

    meta_path = os.path.join(root, "metadata.jsonl")
    done = set()
    if os.path.exists(meta_path):
        done = {json.loads(l)["id"] for l in open(meta_path) if l.strip()}
    # A random order, so a part rendered is a random part.
    todo = sorted((j for t, j in jobs.items() if t not in done), key=lambda j: unit("order" + j[0]))
    print(f"{len(jobs)} takes, {len(done)} already rendered, {len(todo)} to go", flush=True)
    for spk in {j[2] for j in todo}:
        os.makedirs(os.path.join(root, f"spk{spk}"), exist_ok=True)

    os.environ["OMP_NUM_THREADS"] = "1"
    made, failed, seconds = 0, 0, 0.0
    ctx = mp.get_context("spawn")
    with open(meta_path, "a", buffering=1) as meta, ctx.Pool(a.workers, initializer=init, initargs=(a.voice,)) as pool:
        pending = collections.deque()
        it = iter(todo)

        def feed():
            while len(pending) < 4 * a.workers:
                j = next(it, None)
                if j is None:
                    return
                pending.append(pool.apply_async(render, (j[:5],)))

        feed()
        while pending:
            (tid, said, spk, _, _), secs = pending.popleft().get()
            feed()
            if secs is None:
                failed += 1
                continue
            meta.write(json.dumps({
                "file_name": tid + ".flac",
                "id": tid,
                "source": "sft_speech",
                "text": said,
                "spoken": jobs[tid][5],
                "seconds": secs,
                "speaker": "piper",
                "session": f"spk{spk}",
                "doc": "",
            }) + "\n")
            made += 1
            seconds += secs
            if made % 1000 == 0:
                print(f"{made} takes, {seconds / 3600:.1f} h, {failed} failed", flush=True)
    print(f"done: {made} new takes ({seconds / 3600:.1f} h), {failed} failed", flush=True)


if __name__ == "__main__":
    main()
