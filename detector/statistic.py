"""The verdict. EDITABLE — this is the other half of your submission.

`substituted(history)` gets every probe from every round with both answers:

    [{"probe": {...}, "served": "...", "reference": "..."}, ...]

`served` came from the endpoint under audit. `reference` came from the model it
*claims* to be. Return True if you believe they are different engines.

Getting this wrong in one direction misses a fraud. Getting it wrong in the
other accuses an honest company, which is why false positives above the budget
reject the run outright rather than costing you a few points.

The baseline is attest-fyi's shipped rule (`scoring/verdict.py`): mean
`SequenceMatcher` similarity across probes, and call it a swap below 0.45. Those
constants were measured against real providers serving qwen-2.5-7b, where
similarity to the true weights landed in [0.59, 0.69] and to a 1B decoy in
[0.35, 0.40].

That rule separates model families cleanly. It has no answer at all for the
cases that matter most:

  * a requantised copy of the same weights, which stays well above 0.45
  * the endpoint that recognises audit-shaped traffic and answers honestly

Anything you can measure is fair: per-probe distributions instead of a single
mean, agreement structure across categories, response length, formatting tells,
which probes disagree rather than how much. The transcripts are all you get,
and they contain more than one number.
"""
from __future__ import annotations

import difflib
import re
import statistics

# attest-fyi scoring/verdict.py, measured thresholds.
SIM_DIVERGE = 0.45


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _is_error(s: str) -> bool:
    return s.startswith(("<ERR", "<EMPTY"))


def sim_rate(a: list[str], b: list[str]) -> float:
    """Mean similarity over aligned outputs. Ported from scoring/metrics.py."""
    pairs = [(x, y) for x, y in zip(a, b) if not (_is_error(x) or _is_error(y))]
    if not pairs:
        return 0.0
    return statistics.mean(
        difflib.SequenceMatcher(None, _norm(x), _norm(y)).ratio() for x, y in pairs
    )


def substituted(history: list[dict]) -> bool:
    """True if the served engine is not the claimed one.

    Baseline: one mean, one threshold. Everything interesting is in the shape of
    the disagreement, and this throws all of it away.
    """
    if not history:
        return False
    served = [h["served"] for h in history]
    reference = [h["reference"] for h in history]
    return sim_rate(served, reference) < SIM_DIVERGE
