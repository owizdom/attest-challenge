"""Probe generation. EDITABLE, and half of your submission.

`generate` returns the prompts the harness sends to both endpoints. You get
called once per round; `history` holds every probe and both answers from the
rounds before it, so you can follow up on what you have already seen.

You never touch the network. The harness runs your probes against the served
endpoint and against the model it claims to be, then hands you the transcripts.

WHY THESE PROMPTS

The first baseline asked creative questions: a metaphor for trust, a six-word
story, the opening line of a poem. Measured, that was the worst possible choice.
A 1B model at temperature 0.7 disagrees with ITSELF on those about as much as
two different models do, so honest pairs and swapped pairs landed on top of each
other (0.420-0.551 against 0.330-0.490) and no threshold could separate them.

The variance was coming from the probes, not from the weights.

So every prompt here has one short correct answer and a format that leaves no
room to wander. A model confident in an answer emits it again even when
sampling, which drags the honest-pair distribution up toward total agreement.
Two engines that differ still differ, because the questions sit where a small
change in the weights changes the answer:

  * arithmetic wide enough that a 1B model is not fluent at it
  * counting inside a specific word
  * recall that is either right or wrong, never approximately right
  * strict output shapes, where a formatting habit is itself a fingerprint

Ideas this still does not use: prompts tuned so the top-2 tokens are nearly
tied, long-context recall, refusal boundaries, tokenizer edge cases, and asking
the SAME probe under a different framing to separate a real difference from a
lucky sample.
"""
from __future__ import annotations

import random

# Probes come in GROUPS: several phrasings of the same underlying question.
#
# This is the whole idea. Measured, a 1B model asked "47 * 89" at temperature
# 0.7 samples a different wrong answer every time, so an honest pair of 1B
# sessions disagreed on 46-54% of single probes, right on top of where real
# swaps landed. The noise is the model, not the prompt, and no amount of
# prompt-tightening fixes that.
#
# Asking one question three ways gives each endpoint a self-consistency score of
# its own. An uncertain model contradicts ITSELF across the phrasings, so the
# statistic can subtract that off instead of counting it as evidence. What is
# left is the part where the two endpoints disagree with each other MORE than
# either disagrees with itself, and that is a real difference in the weights.
#
# Re-asking the identical prompt would not work: the harness caches a transcript
# per (model, prompt, seed), so the same question returns the same bytes.
# Paraphrases are distinct cache keys probing the same fact.
GROUPS: list[tuple[str, str, list[str]]] = [
    ("g-mul", "arith", [
        "What is 47 * 89? Reply with the number and nothing else.",
        "Compute 47 times 89. Give only the number.",
        "47 x 89 = ? Answer with digits only.",
    ]),
    ("g-sub", "arith", [
        "What is 823 - 467? Reply with the number and nothing else.",
        "Subtract 467 from 823. Give only the number.",
        "823 minus 467 equals what? Digits only.",
    ]),
    ("g-pct", "arith", [
        "What is 15% of 240? Reply with the number and nothing else.",
        "Fifteen percent of 240 is what? Number only.",
        "Calculate 0.15 * 240. Digits only.",
    ]),
    ("g-cnt", "count", [
        "How many times does the letter r appear in 'strawberry'? Number only.",
        "Count the r characters in the word strawberry. Give the count only.",
        "In 'strawberry', how many r's are there? Answer with a digit.",
    ]),
    ("g-len", "count", [
        "How many letters are in the word 'unconscionable'? Number only.",
        "Give the letter count of the word unconscionable. Number only.",
        "Length in letters of 'unconscionable'? Digits only.",
    ]),
    ("g-cap", "fact", [
        "What is the capital of Australia? One word, nothing else.",
        "Name Australia's capital city. One word only.",
        "Australia's capital is which city? Answer with the city name only.",
    ]),
    ("g-elem", "fact", [
        "What is the chemical symbol for tungsten? One word, nothing else.",
        "Give the periodic-table symbol for tungsten. Symbol only.",
        "Tungsten's element symbol is? Reply with the symbol alone.",
    ]),
    ("g-sec", "convert", [
        "How many seconds are in one day? Number only.",
        "Give the number of seconds in 24 hours. Digits only.",
        "One day equals how many seconds? Answer with the number.",
    ]),
    ("g-rev", "manip", [
        "Spell the word 'necessary' backwards. One word, nothing else.",
        "Write necessary in reverse order of its letters. One word only.",
        "Reverse the letters of 'necessary'. Reply with just that string.",
    ]),
    ("g-prime", "format", [
        "List the first 8 prime numbers, comma separated, nothing else.",
        "Give the 8 smallest primes as a comma separated list, nothing else.",
        "Write out the first eight prime numbers separated by commas.",
    ]),
    ("g-sqrt", "fact", [
        "What is the square root of 169? Number only.",
        "Give sqrt(169). Digits only.",
        "Which number squared equals 169? Number only.",
    ]),
    ("g-sort", "format", [
        "Sort alphabetically, comma separated: pear, apple, mango, fig",
        "Put these in alphabetical order, comma separated: pear, apple, mango, fig",
        "Alphabetise and return comma separated: pear, apple, mango, fig",
    ]),
]

# Flattened, each probe tagged with the group it belongs to.
POOL: list[tuple[str, str, str, str]] = [
    ("%s-%d" % (gid, i), cat, gid, prompt)
    for gid, cat, prompts in GROUPS
    for i, prompt in enumerate(prompts)
]

CATEGORIES = sorted({c for _, c, _, _ in POOL})


def _as_probe(t: tuple[str, str, str, str]) -> dict:
    pid, cat, gid, prompt = t
    # `group` is what the statistic keys self-consistency on.
    return {"id": pid, "category": cat, "group": gid, "prompt": prompt}


def _groups_with_a_gap(history: list[dict]) -> list[str]:
    """Groups where the endpoints differed by more than their own noise.

    Re-asking an identical prompt buys nothing: the harness caches a transcript
    per (model, prompt, seed), so the same question returns the same bytes.
    Spending round two on the groups that already showed a gap does buy
    something.
    """
    from detector.statistic import group_scores

    scored = group_scores(history)
    hot = [(g, sc["excess"]) for g, sc in scored.items() if sc["excess"] > 0]
    return [g for g, _ in sorted(hot, key=lambda kv: -kv[1])]


def generate(seed: int, k: int, history: list[dict] | None = None) -> list[dict]:
    """Return up to `k` probes, always in whole groups.

    A partial group is worthless, because self-consistency needs every phrasing
    of a question in the same run. So this packs whole groups and stops rather
    than sending a fragment.

    Round one spreads across categories. Round two prefers groups that already
    showed a gap.
    """
    history = history or []
    asked = {h["probe"]["prompt"] for h in history}
    rng = random.Random(seed)

    fresh = [g for g in GROUPS if not any(p in asked for p in g[2])]
    if not fresh:
        return []

    if history:
        rank = {g: i for i, g in enumerate(_groups_with_a_gap(history))}
        rng.shuffle(fresh)
        fresh.sort(key=lambda g: rank.get(g[0], len(rank)))
    else:
        # round-robin the categories so no single kind dominates round one
        by_cat: dict[str, list] = {}
        for g in fresh:
            by_cat.setdefault(g[1], []).append(g)
        for v in by_cat.values():
            rng.shuffle(v)
        cats = sorted(by_cat)
        rng.shuffle(cats)
        fresh = []
        while any(by_cat[c] for c in cats):
            for c in cats:
                if by_cat[c]:
                    fresh.append(by_cat[c].pop())

    out: list[dict] = []
    for gid, cat, prompts in fresh:
        if len(out) + len(prompts) > k:
            continue
        for i, prompt in enumerate(prompts):
            out.append({"id": "%s-%d" % (gid, i), "category": cat, "group": gid, "prompt": prompt})
    return out
