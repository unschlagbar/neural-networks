#!/usr/bin/env python3
"""Moves the recording prompts the encoder would mishear most to the front of
the voice prompt pool, so the next sessions record where the errors are
(error-selected sentences personalise far better than random ones: Awasthi et
al., arXiv 2103.03142).

The input is `ae2e`'s per-word table of the user's takes (`<encoder>.misheard.tsv`:
word, times heard wrong, times said). A word's error rate is smoothed towards
a prior: halfway from the overall rate to the mean pooled rate of the words
sharing its letter pairs ("th" in "they", "their", "three"). Prompts are picked greedily
by the excess error per word, each word's weight shrinking by `--decay` every
time a picked prompt has contained it, so the front is not one word over and
over. The picked prompts lead in that order; the rest keep their order. The
harness reads the pool from the top and skips what was recorded.

    python -I scripts/error_prompts.py models/s4_R.ae2.misheard.tsv [--pick 1500]

Keeps the pool as it was as `prompts.random.jsonl` the first time.
"""

import argparse
import collections
import heapq
import json
import os
import shutil

POOL = "data/voice/prompts.jsonl"


def pairs(word):
    return {word[i : i + 2] for i in range(len(word) - 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("--pool", default=POOL)
    ap.add_argument("--pick", type=int, default=1500)
    ap.add_argument("--decay", type=float, default=0.6)
    ap.add_argument("--min-words", type=int, default=6)
    a = ap.parse_args()

    wrong, said = {}, {}
    for line in open(a.table):
        w, n_wrong, n_said = line.rstrip("\n").split("\t")
        wrong[w], said[w] = int(n_wrong), int(n_said)
    overall = sum(wrong.values()) / max(1, sum(said.values()))
    pair_wrong, pair_said = collections.Counter(), collections.Counter()
    for w in said:
        for p in pairs(w):
            pair_wrong[p] += wrong[w]
            pair_said[p] += said[w]

    def prior(w):
        rates = [pair_wrong[p] / pair_said[p] for p in pairs(w) if pair_said[p] >= 30]
        if not rates:
            return overall
        return overall + 0.5 * (sum(rates) / len(rates) - overall)

    rate_cache = {}

    def excess(w):
        if w not in rate_cache:
            k = 3.0
            rate = (wrong.get(w, 0) + k * prior(w)) / (said.get(w, 0) + k)
            rate_cache[w] = max(0.0, rate - overall)
        return rate_cache[w]

    backup = os.path.join(os.path.dirname(a.pool), "prompts.random.jsonl")
    if not os.path.exists(backup):
        shutil.copyfile(a.pool, backup)
    lines = open(backup).read().splitlines()
    words = []
    for line in lines:
        record = json.loads(line)
        # A number has several spoken forms, so a take may not say the one the
        # prompt's transcript holds; such prompts are never moved forward.
        readable = not any(c.isdigit() for c in record.get("text", ""))
        words.append(record.get("spoken", "").split() if readable else [])

    covered = collections.Counter()

    def gain(i):
        ws = words[i]
        if len(ws) < a.min_words:
            return 0.0
        return sum(excess(w) * a.decay ** covered[w] for w in ws) / len(ws)

    # Lazy greedy: a prompt's gain only falls as words get covered.
    heap = [(-gain(i), i) for i in range(len(lines))]
    heapq.heapify(heap)
    picked = []
    while heap and len(picked) < a.pick:
        g, i = heapq.heappop(heap)
        fresh = gain(i)
        if heap and fresh < -heap[0][0] - 1e-12:
            heapq.heappush(heap, (-fresh, i))
            continue
        picked.append(i)
        covered.update(words[i])

    chosen = set(picked)
    order = picked + [i for i in range(len(lines)) if i not in chosen]
    tmp = a.pool + ".tmp"
    with open(tmp, "w") as f:
        for i in order:
            f.write(lines[i] + "\n")
    os.replace(tmp, a.pool)

    top = sorted(said, key=lambda w: -excess(w))[:15]
    print(f"overall {overall:.1%} of {sum(said.values())} said words; hardest: " + ", ".join(f"{w} {excess(w) + overall:.0%}" for w in top))
    print(f"{len(picked)} prompts moved to the front of {a.pool}; first five:")
    for i in picked[:5]:
        print("  " + json.loads(lines[i])["text"])
    hits = collections.Counter(w for i in picked for w in words[i] if excess(w) > 0.1)
    print("most covered hard words: " + ", ".join(f"{w} {n}" for w, n in hits.most_common(15)))


if __name__ == "__main__":
    main()
