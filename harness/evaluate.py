"""Score a run. TRUSTED, and it never imports detector code.

Reads `decisions.json` (written by the sandboxed run) plus the corpus labels,
which live only here. Writes the canonical `score.json`.

    score = true-positive rate on swap pairs, 0-100.

with one gate, and it is not a penalty:

    false positives above the pinned budget REJECT the run.

A false positive is this project accusing an honest provider of fraud, which
DESIGN.md §5 calls defamation in as many words. So a detector that fires on
everything scores nothing at all, in the same spirit as ecdsafail rejecting a
Toffoli win that came from skipping uncomputation: an invalid run fails, it does
not merely rank lower.
"""
from __future__ import annotations

import json
import os
import sys

from harness import corpus

DECISIONS_PATH = "decisions.json"
SCORE_PATH = "score.json"
RESULTS_TSV = "results.tsv"

# The budget is a RATE, so it does not silently loosen or tighten as the corpus
# grows. 12.5% of the negatives, rounded down, with a floor of 1 — enough to
# absorb one unlucky draw from the null distribution and nothing like enough to
# let a trigger-happy detector through.
#
# Say plainly what this can and cannot establish: on a corpus of eight negatives
# it is a sanity floor, not the 0.1% false-positive rate a public accusation
# needs. Growing the negative set is the first infrastructure task here.
FP_BUDGET_RATE = 0.125


def fp_budget(negatives: int) -> int:
    return max(1, int(negatives * FP_BUDGET_RATE))


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0


def main() -> int:
    if not os.path.exists(DECISIONS_PATH):
        print("no %s — the run did not complete" % DECISIONS_PATH, file=sys.stderr)
        return 1

    with open(DECISIONS_PATH, encoding="utf-8") as f:
        run = json.load(f)

    if not run.get("sandboxed"):
        print(
            "REJECTED: the detector ran unconfined. Install bubblewrap (Linux) "
            "or use macOS sandbox-exec; an unconfined run is for local "
            "iteration and is never scored.",
            file=sys.stderr,
        )
        return 1

    pairs = corpus.load(run["split"])
    if len({p["id"] for p in pairs}) != len(pairs):
        print("REJECTED: the corpus has duplicate pair ids", file=sys.stderr)
        return 1

    # A decisions.json from a different corpus revision would otherwise be
    # scored against today's labels by id.
    from harness import run as runner
    expect = runner.config_digest(pairs)
    if run.get("config") and run["config"] != expect:
        print("REJECTED: decisions were produced under a different corpus/probe "
              "configuration (%s, expected %s)" % (run["config"][:8], expect[:8]),
              file=sys.stderr)
        return 1

    labels = {p["id"]: p for p in pairs}
    got = {d["id"]: d for d in run["decisions"]}
    if len(got) != len(run["decisions"]):
        print("REJECTED: duplicate ids in decisions.json", file=sys.stderr)
        return 1

    missing = sorted(set(labels) - set(got))
    extra = sorted(set(got) - set(labels))
    if missing:
        print("REJECTED: no decision for %s" % ", ".join(missing), file=sys.stderr)
        return 1
    if extra:
        print("REJECTED: decisions for unknown pairs: %s" % ", ".join(extra), file=sys.stderr)
        return 1

    tp = fn = fp = tn = 0
    per_tier: dict[int, dict[str, int]] = {}
    for pid, meta in labels.items():
        fired = got[pid]["substituted"]
        swap = meta["label"] == "swap"
        if swap and fired:
            tp += 1
        elif swap and not fired:
            fn += 1
        elif not swap and fired:
            fp += 1
        else:
            tn += 1
        if swap:
            t = per_tier.setdefault(meta["tier"], {"caught": 0, "total": 0})
            t["total"] += 1
            t["caught"] += int(fired)

    swaps = tp + fn
    tpr = (tp / swaps * 100.0) if swaps else 0.0
    fpr = (fp / (fp + tn) * 100.0) if (fp + tn) else 0.0
    budget = fp_budget(fp + tn)
    rejected = fp > budget
    score = 0.0 if rejected else round(tpr, 2)

    body = {
        "score": score,
        "metrics": {
            "true_positive_rate": round(tpr, 2),
            "false_positive_rate": round(fpr, 2),
            "swaps_caught": tp,
            "swaps_missed": fn,
            "false_accusations": fp,
            "honest_cleared": tn,
            "false_accusation_budget": budget,
            "negatives": fp + tn,
            "rejected": rejected,
            "by_tier": {
                str(k): "%d/%d" % (v["caught"], v["total"])
                for k, v in sorted(per_tier.items())
            },
            "split": run["split"],
            # Two runs are only comparable under the same corpus and probe
            # budget. results.tsv used to mix configurations silently, which
            # makes a determinism check meaningless.
            "config": run.get("config"),
            "detector": run["detector"][:16],
            # what was actually spent, not the budget. A detector that answered
            # from 2 probes should not be reported as having used 16.
            "probes_per_pair_budget": run["rounds"] * run["probes_per_round"],
            "probes_used_median": _median([d.get("probes_used", 0) for d in run["decisions"]]),
            "sandbox": run.get("sandbox"),
        },
    }

    with open(SCORE_PATH, "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2)
        f.write("\n")

    new = not os.path.exists(RESULTS_TSV)
    with open(RESULTS_TSV, "a", encoding="utf-8") as f:
        if new:
            f.write("detector\tconfig\tsplit\tscore\ttpr\tfpr\tcaught\tmissed\tfalse_acc\n")
        f.write(
            "%s\t%s\t%s\t%.2f\t%.2f\t%.2f\t%d\t%d\t%d\n"
            % (run["detector"][:16], (run.get("config") or "-")[:8], run["split"],
               score, tpr, fpr, tp, fn, fp)
        )

    print(json.dumps(body, indent=2))
    if rejected:
        print(
            "\nREJECTED: %d false accusations, budget is %d. A detector that "
            "cannot clear an honest provider is not a detector."
            % (fp, budget),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
