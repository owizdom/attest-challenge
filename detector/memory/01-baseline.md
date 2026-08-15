# Where the baseline stands, and the two dead ends behind it

The detector shipped today scores **57.14 on held-out, valid**: 8 of 14 swaps
caught, 1 false accusation of 8 (the budget is 1). Per tier:
**t1 3/3 · t2 2/3 · t3 2/3 · t4 0/3 · t5 1/2**. That is the CI number, from
Linux under bubblewrap, which is the scoring path. The same detector scored
50.00 on macOS, so treat a few points as machine noise. On dev it scores
**54.55** with zero false accusations.

The detector before it scored **0.00, rejected**. It caught more swaps (12 of
14) and accused half the honest providers doing it, which voids the run. Catching
swaps is the easy half.

## Attempt 1: mean similarity. Dead.

attest-fyi's production rule, ported unchanged: mean `difflib` similarity across
probes, swap called below 0.45. Constants measured against real 7B providers.

`tools/null.py` on that version:

```
negatives (same weights)      0.420 .. 0.551
swaps                         0.330 .. 0.490
separable by a threshold:     NO (gap -0.070)
```

The worst honest pair is *less* similar than the hardest swap, so every
threshold either misses that swap or accuses that provider. Not a tuning
problem. Moving 0.45 trades one failure for the other.

## Attempt 2: tighter prompts. Also dead, but it produced the clue.

If creative prompts are noisy, ask questions with one correct answer:
arithmetic, letter counts, factual recall, strict output shapes. Then count how
many probes disagreed instead of averaging how much.

Measured on dev, disagreement rate per pair:

```
qwen-7b   vs itself   0.125
llama-8b  vs itself   0.167
llama-1b  vs itself   0.458 .. 0.542     <-- the problem
swaps                 0.375 .. 0.542
```

Honest 1B pairs sat right on top of the swaps. **The noise is the model, not the
prompt.** A 1B engine at temperature 0.7 is not confident about 47 * 89, so it
samples a different wrong answer each time. Any statistic that reads "these two
answers differ" as evidence will convict every small model of being an impostor.

Tightening prompts further cannot fix this. Do not spend a week there.

## What works: measure each side's uncertainty and subtract it

Probes come in **groups of three paraphrases of one question**. Within a group:

```
self_served     how much the served endpoint agrees with ITSELF
self_reference  how much the claimed model agrees with itself
cross           how much the two agree with each other

excess = mean(self_served, self_reference) - cross
```

An uncertain model contradicts itself across the phrasings too, so that cancels.
What survives is the part where the two endpoints disagree with each other *more
than either disagrees with itself*, which is a real difference in the weights.

Measured on dev, two rounds of 12:

```
honest pairs    excess -0.056 .. -0.021
tier 1/2 swaps  excess +0.021 .. +0.146
tier 3/4/5      excess -0.090 .. +0.000
```

Honest pairs land tightly at or below zero regardless of how noisy the model is.
That is the property the first two attempts never had.

**Re-asking the identical prompt does not work.** The harness caches a transcript
per `(model, prompt, seed)`, so the same question returns the same bytes.
Paraphrases are distinct cache keys probing the same fact.

### Second channel, free from the same data

Self-consistency is itself a capability fingerprint: 7B/8B models agree with
themselves about 71% of the time, 1B models about 33%. So a large gap between
the two sides' self-consistency is a family or size swap, even when the answers
look alike. `self_gap > 0.29` catches tier 1 and 2 on its own.

Either channel firing is enough. They fail in different places.

## 🛑 The trap that will cost you an afternoon

**The harness passes only `id`, `category` and `prompt` across the sandbox
boundary** (`harness/untrusted.py:39-45`). Every other key you put on a probe is
silently dropped.

I attached a `group` field. It vanished. Every group arrived as a single member,
self-consistency evaluated to nothing, and the scored run returned **zero
detections with no error at all** while my local measurement said six. Nothing
warns you.

Encode anything you need in the `id` and parse it back. This detector uses
`<group>-<n>` and recovers it with a regex in `statistic.py::_group_of`.

## Where the remaining points are

- **Tier 4 is 0/3.** One precision step on identical weights. Untouched by both
  channels, exactly as DESIGN.md §10 predicted. Nobody knows if it is possible.
  Every point here is unclaimed.
- **Tier 3 is 2/3.** The old rule scored 3/3, but only by sitting at a threshold
  that also condemned half the honest providers. Real requantisation detection
  is still open.
- **Tier 5 is 1/2.** The adversarial endpoint answers honestly to anything that
  looks like a probe. Every prompt in this pool is visibly a test: short, "number
  only", "nothing else". Probes that read like ordinary traffic are untouched
  ground.
- **Round two is barely used.** It currently asks more questions from groups that
  already showed a gap. It could instead spend the whole budget on one group to
  get a much tighter self-consistency estimate.
- **Only one summary number.** Per-category excess, weighting groups by how
  confident they turned out to be, and answer length or formatting as an
  independent channel are all unexplored.

## Corpus caveat, unchanged

Tiers 3-5 use `llama3.2:1b` variants, the only family here with fp16/q8/q4 builds
that fit the machine that generated the corpus. 1B models are far noisier between
sessions than the 7B-70B models real providers serve: attest-fyi measured
0.59-0.69 similarity to true weights against live 7B endpoints. **This corpus is
harder than production.** Adding quantisation variants of a larger model is the
highest-value infrastructure contribution available, and it does not require
beating the detector.
