"""The verdict. EDITABLE, and the other half of your submission.

`substituted(history)` gets every probe from every round with both answers:

    [{"probe": {...}, "served": "...", "reference": "..."}, ...]

`served` came from the endpoint under audit. `reference` came from the model it
*claims* to be. Return True if you believe they are different engines.

Getting this wrong in one direction misses a fraud. Getting it wrong in the
other accuses an honest company, which is why false accusations above the budget
reject the run outright rather than costing a few points.

WHY SELF-CONSISTENCY, AND NOT A SIMILARITY SCORE

Two earlier attempts, both measured, both dead ends:

1. Mean `difflib` similarity, swap below 0.45. Honest pairs landed at
   0.420-0.551 and swaps at 0.330-0.490. Overlapping, so no threshold works.
2. Tighter prompts with one correct answer, counting how many disagreed.
   Honest 1B pairs still disagreed on 46-54% of probes, right on top of the
   swaps. But qwen-7b against itself disagreed on 12.5% and llama-8b on 16.7%.

That second result is the whole clue. The noise is not the prompt and it is not
the metric, it is the MODEL: a 1B engine at temperature 0.7 is not confident on
a question like 47 * 89, so it samples a different wrong answer every time. Any
statistic that reads "these two answers differ" as evidence of a swap will
convict every small model of being someone else.

So measure that uncertainty instead of ignoring it. Each question arrives as
three paraphrases (see probes.py). Within one group we can compute:

    self_served    how much the served endpoint agrees with ITSELF
    self_reference how much the claimed model agrees with itself
    cross          how much the two agree with each other

If they are the same weights, cross should be about as high as the two selves.
If they are different weights, cross falls BELOW both. The excess

    excess = mean(self_served, self_reference) - cross

is near zero for an honest pair however noisy the model is, and positive for a
real swap. A model being bad at arithmetic no longer looks like fraud.

Untried from here: weighting groups by how confident they turned out to be,
per-category excess rather than one pooled number, and using answer length or
formatting as a second, independent channel.
"""
from __future__ import annotations

import difflib
import re
import statistics

# Two answers carry the same content. Short answers make this cheap; the fuzzy
# arm only catches wrappers like "42" against "The answer is 42".
NEAR = 0.86

# How far cross-agreement must fall below the endpoints' own self-agreement
# before this is called a swap. Set on the dev split only, and deliberately
# loose: a false accusation rejects the whole run, so a missed swap is the
# cheaper mistake.
EXCESS_MAX = 0.01

# Second, independent channel. Self-agreement is itself a capability
# fingerprint: measured on dev, the 7B and 8B models agree with themselves about
# 71% of the time and the 1B models about 33%. So when the two endpoints have
# very different self-consistency, they are very different engines, whatever
# their answers look like side by side. This is what catches a family or size
# swap that the excess alone misses.
SELF_GAP_MAX = 0.29


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^(the answer is|answer:|result:|it is|that is|sure[,!]?)\s*", "", s)
    s = re.sub(r"[.,!?;:'\"`*_()\[\]]+", "", s)
    return s.strip()


def _is_error(s: str) -> bool:
    return s.startswith(("<ERR", "<EMPTY"))


def _keys(s: str) -> list[str]:
    """The load-bearing tokens: numbers if there are any, else words."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    return nums if nums else s.split()


def agrees(a: str, b: str) -> bool:
    """Did these two answers land in the same place?"""
    if _is_error(a) or _is_error(b):
        return True  # a transport failure is not evidence of a swap
    na, nb = _norm(a), _norm(b)
    if na == nb:
        return True
    ka, kb = _keys(na), _keys(nb)
    if ka and ka == kb:
        return True
    if na and nb and (na in nb or nb in na):
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= NEAR


def _self_rate(answers: list[str]) -> float:
    """How often an endpoint agreed with itself across the paraphrases."""
    pairs = [(answers[i], answers[j])
             for i in range(len(answers)) for j in range(i + 1, len(answers))]
    if not pairs:
        return 1.0
    return sum(1 for a, b in pairs if agrees(a, b)) / len(pairs)


def _cross_rate(served: list[str], reference: list[str]) -> float:
    """How often the two endpoints agreed, every phrasing against every other.

    All pairs rather than position-by-position: the question is whether the two
    engines reach the same answer, not whether they reach it in the same order.
    """
    pairs = [(s, r) for s in served for r in reference]
    if not pairs:
        return 1.0
    return sum(1 for s, r in pairs if agrees(s, r)) / len(pairs)


def _group_of(probe: dict) -> str:
    """Which question group a probe belongs to.

    The group has to be recovered from the id, because the harness only carries
    `id`, `category` and `prompt` across the sandbox boundary
    (harness/untrusted.py:39-45) and drops every other key. Passing a `group`
    field silently loses it, every group arrives as a single member, and the
    self-consistency this whole detector rests on quietly evaluates to nothing.
    So probes.py encodes the group in the id as `<group>-<n>`.
    """
    pid = str(probe.get("id", ""))
    return re.sub(r"-\d+$", "", pid) or pid


def group_scores(history: list[dict]) -> dict[str, dict]:
    """Per question group: each side's self-agreement, the cross-agreement, and
    the excess. Also used by probes.py to pick round two."""
    groups: dict[str, dict[str, list[str]]] = {}
    for h in history:
        g = _group_of(h["probe"])
        b = groups.setdefault(g, {"served": [], "reference": []})
        b["served"].append(h["served"])
        b["reference"].append(h["reference"])

    out: dict[str, dict] = {}
    for g, b in groups.items():
        if len(b["served"]) < 2:
            continue  # a lone probe carries no self-consistency
        ss, sr = _self_rate(b["served"]), _self_rate(b["reference"])
        cross = _cross_rate(b["served"], b["reference"])
        out[g] = {
            "self_served": ss,
            "self_reference": sr,
            "cross": cross,
            "excess": (ss + sr) / 2 - cross,
        }
    return out


def summary(history: list[dict]) -> dict:
    """The two numbers the verdict rests on."""
    scored = group_scores(history)
    if not scored:
        return {"excess": 0.0, "self_gap": 0.0}
    return {
        # cross-agreement falling below the endpoints' own noise
        "excess": statistics.mean(s["excess"] for s in scored.values()),
        # the two endpoints having different amounts of noise in the first place
        "self_gap": abs(
            statistics.mean(s["self_served"] for s in scored.values())
            - statistics.mean(s["self_reference"] for s in scored.values())
        ),
    }


def substituted(history: list[dict]) -> bool:
    """True if the served engine is not the claimed one.

    Either channel is enough. They fail in different places: the excess catches
    engines that answer differently, the self gap catches engines of different
    capability, and a swap only has to trip one.
    """
    if not history:
        return False
    m = summary(history)
    return m["excess"] > EXCESS_MAX or m["self_gap"] > SELF_GAP_MAX
