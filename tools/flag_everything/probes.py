"""Probe generation. EDITABLE — this is half of your submission.

`generate` returns the prompts the harness will send to both endpoints. You get
called once per round; `history` holds every probe and both answers from the
rounds before it, so you can follow up on what you have already seen.

You never touch the network. The harness runs your probes, against both the
served endpoint and the model it claims to be, and hands you the transcripts.

The baseline below is attest-fyi's shipped probe pool (`probes/suite.py`): short,
tightly-constrained prompts at temperature 0, chosen so that a different engine
diverges on the first few tokens rather than somewhere in a paragraph. It is a
reasonable start and a long way from the best you can do.

Ideas the baseline does not touch:

  * knife-edge prompts, where the reference's top-1 and top-2 tokens are nearly
    tied, so a requantised engine flips the argmax. DESIGN.md calls this the
    most sensitive precision probe available without logprobs.
  * arithmetic and counting at the exact width where reduced precision starts
    to slip.
  * long-context recall, where a smaller model degrades first.
  * refusal boundaries — the precise point a model declines.
  * tokenizer tells: rare unicode, leading spaces, repeated punctuation.
  * round two spent entirely on whichever probes disagreed in round one.
"""
from __future__ import annotations

import random

# attest-fyi probes/suite.py FIXED, unchanged.
FIXED = [
    ("fx01", "style", "Continue with exactly the next eight words, nothing else: "
                      "'In the quiet hours before the storm'"),
    ("fx02", "style", "Write the opening line of a poem about an empty train station. One line only."),
    ("fx03", "metaphor", "Give a single metaphor for trust, in under six words."),
    ("fx04", "knowledge", "Name a colour that is not in the rainbow. One word only."),
    ("fx05", "naming", "Invent a one-word name for a startup that audits AI. One word, no explanation."),
    ("fx06", "completion", "Finish this sentence in five words exactly: 'The seal was valid but'"),
    ("fx07", "rhyme", "Give one word that rhymes with 'silver'."),
    ("fx08", "explain", "In one short sentence, explain why the sky is blue."),
    ("fx09", "analogy", "Complete the analogy with one word: key is to lock as password is to ____"),
    ("fx10", "summary", "Summarise the idea of trust in exactly three words."),
    ("fx11", "story", "Write a six-word story about a broken promise."),
    ("fx12", "adjective", "Pick an adjective to describe the colour of rust. One word."),
]

_CONCEPTS = ["trust", "secrecy", "decay", "memory", "distance", "silence",
             "speed", "doubt", "loyalty", "ruin", "dawn", "machinery"]
_THINGS = ["a locked door", "an empty harbour", "a dead battery", "a cold engine",
           "a sealed letter", "a broken clock", "a foggy mirror", "a quiet server"]
_CONCEPT_T = [
    ("metaphor", "Give a single metaphor for %s, in under six words."),
    ("summary", "Summarise the idea of %s in exactly three words."),
]
_THING_T = [
    ("style", "Write the opening line of a poem about %s. One line only."),
    ("describe", "Describe %s in exactly five words."),
    ("story", "Give a six-word story involving %s."),
]


def _pool(seed: int) -> list[dict]:
    pool = [{"id": i, "category": c, "prompt": p} for i, c, p in FIXED]
    rng = random.Random(seed)
    concepts, things = _CONCEPTS[:], _THINGS[:]
    rng.shuffle(concepts)
    rng.shuffle(things)
    for c in concepts:
        for cat, tpl in _CONCEPT_T:
            pool.append({"id": "c-%s-%s" % (cat, c.replace(" ", "_")),
                         "category": cat, "prompt": tpl % c})
    for i, t in enumerate(things):
        for cat, tpl in _THING_T:
            pool.append({"id": "t-%s-%d" % (cat, i), "category": cat, "prompt": tpl % t})
    return pool


def generate(seed: int, k: int, history: list[dict] | None = None) -> list[dict]:
    """Return up to `k` probes.

    Baseline: sample from the pool, skipping anything already asked. It does not
    read `history` beyond that, which is the most obvious thing to improve —
    round two currently learns nothing from round one.
    """
    asked = {h["probe"]["prompt"] for h in (history or [])}
    pool = [p for p in _pool(seed) if p["prompt"] not in asked]
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:k]
