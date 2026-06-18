#!/usr/bin/env python3
"""Char-level n-gram (Markov) baseline on the political-speeches data.

Builds interpolated n-gram models of orders 0..N (Jelinek-Mercer, lambdas
tuned by EM on a held-out split) and reports per-char cross-entropy in nats
plus perplexity. This is the bar the hierarchical model's big word backbone
has to beat to justify its long-range context.

Split is along document boundaries (---FILE---) so no speech leaks across
train/val.
"""
import sys, math, random
from collections import defaultdict, Counter

PATH = "/home/unschlagbar/training_data/political_speeches.txt"
MAX_ORDER = 8            # markov context length in characters
TRAIN_FRAC = 0.9
EM_ITERS = 12
random.seed(0)

def load():
    with open(PATH, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    docs = text.split("---FILE---")
    docs = [d for d in docs if d.strip()]
    random.shuffle(docs)
    cut = int(len(docs) * TRAIN_FRAC)
    train = "\n".join(docs[:cut])
    val   = "\n".join(docs[cut:])
    # cap for speed of pure-python build/EM; still representative
    if len(train) > 4_000_000:
        train = train[:4_000_000]
    if len(val) > 300_000:
        val = val[:300_000]
    return train, val

def build_counts(text, order):
    # counts[k][context] = Counter(next_char) for context length k = 0..order
    counts = [defaultdict(Counter) for _ in range(order + 1)]
    n = len(text)
    for i in range(n):
        c = text[i]
        for k in range(order + 1):
            if i - k < 0:
                break
            ctx = text[i - k:i]
            counts[k][ctx][c] += 1
    return counts

def mle_probs(counts, vocab_size):
    # returns function p_k(c|ctx) for each order using raw MLE, plus uniform p_-1
    def p(k, ctx, c):
        cnt = counts[k].get(ctx)
        if not cnt:
            return None
        total = sum(cnt.values())
        x = cnt.get(c, 0)
        if x == 0:
            return None
        return x / total
    return p

def interp_prob(counts, lambdas, vocab_size, ctx_full, c):
    """Jelinek-Mercer interpolation over orders 0..order."""
    order = len(lambdas) - 1  # last index = uniform fallback weight slot
    # build per-order MLE estimates (None if unseen)
    probs = []
    for k in range(order + 1):
        ctx = ctx_full[len(ctx_full) - k:] if k > 0 else ""
        cnt = counts[k].get(ctx)
        if cnt:
            total = sum(cnt.values())
            probs.append(cnt.get(c, 0) / total)
        else:
            probs.append(0.0)
    uniform = 1.0 / vocab_size
    mix = lambdas[-1] * uniform
    for k in range(order + 1):
        mix += lambdas[k] * probs[k]
    return mix

def main():
    train, val = load()
    vocab = set(train) | set(val)
    V = len(vocab)
    print(f"train chars={len(train):,}  val chars={len(val):,}  vocab={V}")

    results = {}
    for order in range(0, MAX_ORDER + 1):
        counts = build_counts(train, order)
        # EM-tune interpolation weights on val (held-out), orders 0..order + uniform
        n_comp = order + 2  # orders 0..order plus uniform
        lambdas = [1.0 / n_comp] * n_comp

        # precompute per-position per-order MLE for val (cache)
        valn = len(val)
        for it in range(EM_ITERS):
            exp_counts = [0.0] * n_comp
            ll = 0.0
            for i in range(valn):
                c = val[i]
                comps = []
                for k in range(order + 1):
                    ctx = val[max(0, i - k):i]
                    cnt = counts[k].get(ctx)
                    comps.append((cnt.get(c, 0) / sum(cnt.values())) if cnt else 0.0)
                comps.append(1.0 / V)  # uniform
                mix = sum(lambdas[j] * comps[j] for j in range(n_comp))
                if mix <= 0:
                    mix = 1e-12
                ll += math.log(mix)
                for j in range(n_comp):
                    exp_counts[j] += lambdas[j] * comps[j] / mix
            s = sum(exp_counts)
            lambdas = [e / s for e in exp_counts]
        ce = -ll / valn
        ppl = math.exp(ce)
        results[order] = (ce, ppl, lambdas)
        print(f"order {order}: val CE = {ce:.4f} nats ({ce/math.log(2):.3f} bits)  ppl = {ppl:.3f}")

    print("\nbest interpolated lambdas (highest order):", [f"{x:.3f}" for x in results[MAX_ORDER][2]])

if __name__ == "__main__":
    main()
