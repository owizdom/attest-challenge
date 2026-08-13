"""Prove the benchmark is sound. Run this before opening it to solvers.

Each check is a property the benchmark claims in its README. A claim nobody
tested is a claim that is false later, in public, when a solver finds it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.getcwd())
from harness import corpus, evaluate, execute, run  # noqa: E402

# Held-out is not in the repo, so a solver running this locally gets dev. The
# properties under test — the gate, fail-closed, the seed commitment — are
# properties of the scorer, not of a particular split.
try:
    corpus.load("heldout")
    SPLIT = "heldout"
except SystemExit:
    SPLIT = "dev"


def _sim(a, b):
    """Local copy — the scorer must never import detector code, not even to
    measure the corpus."""
    import difflib
    import re
    import statistics
    n = lambda s: re.sub(r"\s+", " ", s.strip().lower())  # noqa: E731
    pairs = [(x, y) for x, y in zip(a, b) if not x.startswith(("<ERR", "<EMPTY"))]
    if not pairs:
        return 0.0
    return statistics.mean(
        difflib.SequenceMatcher(None, n(x), n(y)).ratio() for x, y in pairs
    )

OK, FAIL = "  PASS  ", "  FAIL  "
results = []


def check(name, passed, detail=""):
    results.append((name, passed, detail))
    print("%s %-42s %s" % (OK if passed else FAIL, name, detail))


# --- 3. the false-positive gate rejects a flag-everything detector -----------
# Tested against evaluate.py directly: it is the thing under test, and
# synthesising the decisions costs nothing while a full run costs 320
# generations. The end-to-end version is tools/flag_everything/.
def gate_check():
    pairs = corpus.load(SPLIT)
    with tempfile.TemporaryDirectory() as d:
        cwd = os.getcwd()
        os.chdir(d)
        try:
            with open("decisions.json", "w") as f:
                json.dump({
                    "split": SPLIT, "detector": "flag-everything" + "0" * 20,
                    "seed": 1, "sandboxed": True, "rounds": 2, "probes_per_round": 8,
                    "decisions": [{"id": p["id"], "tier": p["tier"],
                                   "substituted": True, "probes_used": 16,
                                   "merkle": "sha256:0"} for p in pairs],
                }, f)
            rc = evaluate.main()
            body = json.load(open("score.json"))
        finally:
            os.chdir(cwd)
    check("flag-everything is REJECTED", rc != 0 and body["score"] == 0.0,
          "tpr=%.0f, %d false accusations, budget %d, score=%.0f"
          % (body["metrics"]["true_positive_rate"], body["metrics"]["false_accusations"],
             body["metrics"]["false_accusation_budget"], body["score"]))
    check("  ...and it had a perfect TPR", body["metrics"]["true_positive_rate"] == 100.0)


# --- an unconfined run is never scored ---------------------------------------
def unconfined_check():
    with tempfile.TemporaryDirectory() as d:
        cwd = os.getcwd()
        os.chdir(d)
        try:
            with open("decisions.json", "w") as f:
                json.dump({"split": SPLIT, "detector": "x" * 40, "seed": 1,
                           "sandboxed": False, "rounds": 2, "probes_per_round": 8,
                           "decisions": []}, f)
            rc = evaluate.main()
            wrote = os.path.exists("score.json")
        finally:
            os.chdir(cwd)
    check("unconfined run is REJECTED", rc != 0 and not wrote, "no score.json written")


# --- a crashed detector fails closed -----------------------------------------
def missing_decision_check():
    pairs = corpus.load(SPLIT)
    with tempfile.TemporaryDirectory() as d:
        cwd = os.getcwd()
        os.chdir(d)
        try:
            with open("decisions.json", "w") as f:
                json.dump({"split": SPLIT, "detector": "x" * 40, "seed": 1,
                           "sandboxed": True, "rounds": 2, "probes_per_round": 8,
                           "decisions": [{"id": p["id"], "tier": p["tier"],
                                          "substituted": True, "probes_used": 16,
                                          "merkle": "sha256:0"} for p in pairs[:-1]]}, f)
            rc = evaluate.main()
            wrote = os.path.exists("score.json")
        finally:
            os.chdir(cwd)
    check("a missing decision fails CLOSED", rc != 0 and not wrote,
          "one pair omitted -> no score")


# --- 4. the seed commitment defeats pre-computation ---------------------------
def seed_commitment_check():
    base = run.detector_digest()
    with tempfile.TemporaryDirectory() as d:
        alt = os.path.join(d, "detector")
        shutil.copytree("detector", alt)
        # a solver adds a memorised lookup table
        with open(os.path.join(alt, "statistic.py"), "a") as f:
            f.write("\n_MEMORISED = {'abc': True}\n")
        moved = run.detector_digest(alt)
    check("adding a lookup table moves the seed", base != moved,
          "%s -> %s" % (base[:8], moved[:8]))

    with tempfile.TemporaryDirectory() as d:
        alt = os.path.join(d, "detector")
        shutil.copytree("detector", alt)
        os.makedirs(os.path.join(alt, "memory"), exist_ok=True)
        with open(os.path.join(alt, "memory", "99-new.md"), "w") as f:
            f.write("# notes\n")
        same = run.detector_digest(alt)
    check("adding a memory note does NOT", base == same, "notes are for humans")


# --- 5. the corpus is not trivially separable --------------------------------
def null_distribution_check():
    """The negatives must be genuinely noisy. If the same weights returned the
    same bytes, 'not identical -> swap' would score 100 and measure nothing.

    The probes come from the SANDBOXED detector, not from `import detector`.
    Importing a submission into the setter's own process — which is what this
    used to do, at full privilege, in the repo — is setter-side RCE from a
    contestant."""
    ps = run._call("probes", {"seed": 7, "k": 6, "history": []})
    m = corpus.M["llama1b_q8"]
    ref = [execute.generate(m, p["prompt"], execute.REF_SEED) for p in ps]
    srv = [execute.generate(m, p["prompt"], execute.SERVED_SEED) for p in ps]
    check("negatives are NOT byte-identical", srv != ref,
          "same weights, sim=%.3f across two sessions" % _sim(srv, ref))


# --- 6. determinism -----------------------------------------------------------
def determinism_check():
    if not os.path.exists("results.tsv"):
        check("determinism", False, "no results.tsv yet — run ./benchmark.sh first")
        return
    rows = [l.split("\t") for l in open("results.tsv").read().strip().split("\n")[1:]]
    by = {}
    for r in rows:
        if len(r) < 4:
            continue
        by.setdefault((r[0], r[1], r[2]), set()).add(r[3])   # detector, config, split
    repeats = {k: v for k, v in by.items() if len(v) > 1}
    seen_twice = sum(1 for k in by if len(by[k]) == 1
                     and sum(1 for r in rows if tuple(r[:3]) == k) > 1)
    check("same detector+config+split -> same score", not repeats,
          "%d combos, %d run more than once, %d disagreed"
          % (len(by), seen_twice, len(repeats)))


if __name__ == "__main__":
    print("\n== attest-challenge: benchmark soundness ==\n")
    print("split under test: %s%s\n"
          % (SPLIT, "" if SPLIT == "heldout" else "  (held-out not present here — expected)"))
    gate_check()
    unconfined_check()
    missing_decision_check()
    seed_commitment_check()
    null_distribution_check()
    determinism_check()
    bad = [n for n, p, _ in results if not p]
    print("\n%d/%d checks passed" % (len(results) - len(bad), len(results)))
    for n in bad:
        print("  FAILED: %s" % n)
    raise SystemExit(1 if bad else 0)
