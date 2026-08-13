"""Measure the null distribution. Setter tool.

The whole benchmark rests on one empirical claim: the same weights sampled in
two sessions look more alike than two different engines do. If those two
distributions overlap, no threshold separates them and the corpus is not
measuring what it says it measures.

This prints both, so the claim is a number rather than an assumption.
"""
from __future__ import annotations

import difflib
import os
import re
import statistics
import sys

sys.path.insert(0, os.getcwd())
from harness import corpus, execute, run  # noqa: E402


def sim(a, b):
    n = lambda s: re.sub(r"\s+", " ", s.strip().lower())  # noqa: E731
    ps = [(x, y) for x, y in zip(a, b) if not x.startswith(("<ERR", "<EMPTY"))]
    return statistics.mean(
        difflib.SequenceMatcher(None, n(x), n(y)).ratio() for x, y in ps
    ) if ps else 0.0


def main(split="dev"):
    pairs = corpus.load(split)
    probes = run._call("probes", {"seed": int(run.detector_digest()[:8], 16),
                                  "k": run.PROBES_PER_ROUND * run.ROUNDS, "history": []})
    print("%d probes, %d pairs\n" % (len(probes), len(pairs)))
    by_tier = {}
    for p in pairs:
        served, ref = execute.run_pair(p, probes)
        s = sim(served, ref)
        by_tier.setdefault(p["tier"], []).append((p["id"], s))
        print("  %-8s tier %d  sim %.3f  %s" % (p["id"], p["tier"], s, p["label"]))

    print("\n%-22s %8s %8s %8s %8s" % ("", "n", "mean", "min", "max"))
    for tier in sorted(by_tier):
        v = [s for _, s in by_tier[tier]]
        name = "negatives (same)" if tier == 0 else "tier %d swaps" % tier
        print("%-22s %8d %8.3f %8.3f %8.3f" % (name, len(v), statistics.mean(v), min(v), max(v)))

    neg = [s for _, s in by_tier.get(0, [])]
    swaps = [s for t in by_tier if t for _, s in by_tier[t]]
    if neg and swaps:
        print("\nnull floor (worst honest pair):   %.3f" % min(neg))
        print("swap ceiling (easiest to miss):   %.3f" % max(swaps))
        gap = min(neg) - max(swaps)
        print("separable by a single threshold:  %s (gap %+.3f)"
              % ("YES" if gap > 0 else "NO — the distributions overlap", gap))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
