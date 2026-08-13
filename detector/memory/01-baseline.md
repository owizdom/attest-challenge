# The baseline, and exactly where it fails

The shipped detector is attest-fyi's production rule, ported unchanged:
mean `difflib` similarity across probes, swap called below 0.45.

Those constants are not arbitrary. They were measured against three real
providers serving qwen-2.5-7b: similarity to the true weights landed in
[0.59, 0.69], to a 1B decoy in [0.35, 0.40]. Clean separation, for that case.

## What it catches

Family swaps and large size swaps. Two different model families answering the
same tightly-constrained prompt diverge fast, and a mean over a dozen probes is
enough to see it.

## What it cannot catch, structurally

**Requantisation.** q4 and q8 of the same weights agree on most short prompts.
The mean similarity stays far above 0.45, so the rule says "bound" and the swap
walks. This is tier 3 and tier 4 of the corpus, and tier 4 (fp16 vs q8) is the
one DESIGN.md §10 openly calls possibly undetectable from behaviour alone.

The mean is the problem. A requantised engine agrees with the reference on most
prompts and disagrees sharply on a few, the ones sitting near a decision
boundary. Averaging deletes exactly the signal that matters. Look at the
distribution, or at *which* probes disagree, rather than how much on average.

**The adversarial endpoint** (tier 5) answers honestly whenever a prompt looks
like an audit and cheaply otherwise. Every probe in the baseline pool is
visibly a test, short, constrained, "one word only". A probe set that reads
like ordinary traffic is a different attack surface entirely, and the baseline
has none.

## Cheapest next moves

1. Replace the mean with a per-probe disagreement count. A swap should show a
   handful of hard disagreements, not a uniform drift.
2. Generate probes whose answers sit on a knife edge, so a precision change
   flips the argmax. DESIGN.md calls this the most sensitive precision probe
   available without logprobs.
3. Use round two. `history` is passed to `generate` and the baseline ignores it.
   Spend round one broad, round two only where round one disagreed.
4. Write probes that do not look like probes.

## Measured, 2026-08

Held-out: **33.33** (t1 1/1 · t2 1/1 · t3 0/2 · t4 0/1 · t5 0/1), with **1 false
accusation**, which is the entire budget. Dev: 83.33 with 0 false accusations.

The dev/held-out gap is the interesting part. Dev has one tier-3 pair and the
baseline catches it; held-out has two and it catches neither. Same rule, same
tier, opposite result, so the baseline is not "solving tier 3", it is landing
on the right side of a threshold by luck on one pair.

## The false positive is the same bug as the misses

The baseline's one false accusation is `llama3.2:1b-instruct-q4_K_M` judged
against itself. Two sessions of the *same* q4 weights disagree about as much as
q4 disagrees with q8. So the noise floor of the small quantised model is the
size of the signal you are hunting.

This is why lowering the threshold is not a strategy, it converts misses into
rejections. The budget is 1 and the baseline already spends it.

What it suggests instead: the baseline's probes are *creative* ("a metaphor for
doubt", "a six-word story"), and creative prompts have enormous session-to-
session variance in a 1B model. The variance is coming from the probes, not
from the weights. Prompts with a narrow correct answer, arithmetic at an
awkward width, a specific factual recall, a strictly-formatted output, should
collapse the null distribution while keeping the precision signal. Untested, but
it is the first thing to try, and it would improve the false-positive rate and
the detection rate at the same time.

## The score moved 50 points on a seed change, read this before trusting a run

An earlier version of this benchmark had six swap pairs and four negatives. The
baseline scored **33.3** on one probe seed and, on a different seed, caught five
of six but made two false accusations and was **rejected**. Same detector, same
corpus, same rule. The only thing that changed was which probes got drawn.

That is not the detector being unstable so much as the measurement being too
small to see it: at six swaps, one pair is 16.7 points, and at four negatives one
unlucky draw is the whole budget.

The corpus is bigger now (14 swaps, 8 negatives held out; 11 and 6 on dev) and
the budget is a rate rather than a fixed count. It is still small. **If you get a
result you like, re-run it.** A two-point improvement on this corpus is noise; a
tier that goes from 0/3 to 3/3 is a finding.

The negatives now include pairs where *both* sides are non-default sessions
(`served_seed=2024, ref_seed=9001`), not just "session 1 vs the reference run".
If your statistic implicitly assumes the reference stream is always the same
draw, those will catch you out.

## The measurement that defines this challenge

`python3 tools/null.py dev`, 24 probes, 17 pairs:

```
                              n     mean      min      max
negatives (same weights)      6    0.472    0.420    0.551
tier 1 swaps                  3    0.357    0.330    0.390
tier 2 swaps                  3    0.390    0.358    0.407
tier 3 swaps                  2    0.432    0.423    0.441
tier 4 swaps                  2    0.455    0.420    0.490
tier 5 swaps                  1    0.431    0.431    0.431

null floor (worst honest pair):   0.420
swap ceiling (easiest to miss):   0.490
separable by a single threshold:  NO (gap -0.070)
```

Read the last line carefully. **No threshold on mean similarity separates these
two distributions.** Not 0.45, not any other number. The worst honest pair
(0.420) is less similar than the hardest swap (0.490), so every threshold either
misses that swap or accuses that honest provider.

This is not a tuning problem and you cannot fix it by moving 0.45. The statistic
throws away the information that distinguishes the cases. Anything you do with
one mean is already at its ceiling.

Where the separation still exists: tiers 1 and 2 sit at 0.33-0.41, clear of the
null floor at 0.420, narrowly. Tiers 3, 4 and 5 sit inside the null entirely.

### What this implies for where to look

The mean is a single number summarising 24 comparisons, and it is the *wrong*
single number. Concretely untried:

- **The distribution, not its mean.** Two sessions of one model disagree
  *uniformly*, every probe a bit different. A requantised engine agrees closely
  on most probes and diverges hard on a few. Same mean, very different shape.
  Variance, skew, or a count of probes below some per-probe cut all preserve
  what the mean destroys.
- **Which probes disagree.** If the same 3 of 24 probes diverge on every
  session, that is a property of the weights. If a different 3 diverge each
  time, that is sampling noise. The baseline cannot tell these apart; you have
  two rounds and can ask the same probe twice.
- **Anything other than `difflib`.** Length, formatting, token-level agreement,
  refusal boundaries, numeric answers. Text similarity on creative prompts is
  close to the noisiest measurement available.

### Corpus caveat, stated plainly

Tiers 3-5 use `llama3.2:1b` variants, because it is the only family here with
fp16/q8/q4 builds that fit on the machine that generated the corpus. 1B models
at temperature 0.7 are far noisier session-to-session than the 7B-70B models a
real provider serves, attest-fyi measured 0.59-0.69 for true weights against
live 7B endpoints, against 0.42-0.55 here. So this corpus is *harder* than
production, and a rejected run here does not by itself mean a rule is broken in
production. Adding quantisation variants of a larger model is the other high-
value infrastructure contribution.
