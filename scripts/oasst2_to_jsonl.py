#!/usr/bin/env python3
"""Flatten OpenAssistant oasst2 message trees into `{"messages": [...]}` JSONL.

One conversation per tree: from the root prompt, follow the best-ranked child
(rank 0, falling back to the most upvoted) until a leaf. English only; deleted
or review-failed messages end the thread there.

usage: oasst2_to_jsonl.py <ready.trees.jsonl.gz> <out.jsonl>
"""
import gzip, json, sys

ROLE = {"prompter": "user", "assistant": "assistant"}


def ok(m):
    return m.get("lang") == "en" and not m.get("deleted") and m.get("review_result", True) is not False


def best(children):
    kids = [c for c in children if ok(c)]
    if not kids:
        return None
    ranked = [c for c in kids if c.get("rank") is not None]
    if ranked:
        return min(ranked, key=lambda c: c["rank"])
    return max(kids, key=lambda c: (c.get("emojis") or {}).get("+1", 0))


def thread(root):
    turns, m = [], root
    while m is not None and ok(m):
        turns.append({"role": ROLE[m["role"]], "content": m["text"]})
        m = best(m.get("replies") or [])
    # A thread must end on an answer.
    while turns and turns[-1]["role"] != "assistant":
        turns.pop()
    return turns


def main():
    src, dst = sys.argv[1], sys.argv[2]
    n = kept = 0
    with gzip.open(src, "rt") as f, open(dst, "w") as out:
        for line in f:
            n += 1
            tree = json.loads(line)
            turns = thread(tree["prompt"])
            if len(turns) >= 2:
                kept += 1
                out.write(json.dumps({"messages": turns, "category": "oasst2"}) + "\n")
    print(f"{kept} of {n} trees kept")


if __name__ == "__main__":
    main()
