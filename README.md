# attest-challenge

> **Catch an inference provider serving a model it did not promise you.**
> Score is the share of substitutions you catch. Falsely accusing an honest
> provider does not cost you points, it rejects your run.

## The Problem

**Goal metric: true-positive rate on model substitution, at a pinned
false-accusation budget. Higher is better.**

Confidential inference is sold on a promise. Your prompt goes into a sealed
enclave, a specific model answers it, and nobody, not even the operator, can look
inside. The attestation proves the *enclave* is genuine. It says nothing about
which weights got loaded into it.

An operator can serve a 7B where an 80B was promised, a q4 where fp16 was
promised, or a distilled student where the teacher was promised, and the seal
still verifies. The margin on doing so is large and the current cost of being
caught is zero, because there is no accepted method for catching it.

Cryptographic proof of inference does not exist at production scale. Everything
available today is behavioural: send the endpoint some prompts, send the same
prompts to weights you control, and decide from the two transcripts whether they
came from the same engine. **That decision is the metric.** A solver moves it by
choosing better probes and a better statistic, and by nothing else.

This is the ground-truth testbed for [attest.fyi](https://attest.fyi), which
audits live providers. The detector that wins here becomes the detector that
decides, in public, whether a real company is lying about what it serves. Which
is also why a false accusation is disqualifying rather than merely expensive: a
FAIL verdict published without a false-positive rate behind it is defamation.

### What makes it a live research problem

The corpus rises in five tiers.

| Tier | Substitution | Status |
|---|---|---|
| 1 | Different family, llama served where qwen was claimed | separable today |
| 2 | Different size in one family, 1B vs 8B | separable today |
| 3 | Requantisation two steps apart, q4 vs fp16 | partly |
| 4 | **One precision step, identical weights, fp16 vs q8** | **open** |
| 5 | **Adversarial: the endpoint serves the real model to traffic it recognises as an audit, and the cheap one to everything else** | **open** |

Tiers 4 and 5 are not exercises with hidden answers. attest.fyi's own DESIGN.md
calls tier 4 *possibly undetectable from behaviour alone* and tier 5 *the hard
ceiling*. Nobody knows whether either is solvable.

## The Verifier

Deterministic, offline, and it never runs solver code in its own process.

A **pair** is one audit: an endpoint under test, and the model it *claims* to be.
You choose the prompts. The harness runs them against both sides and hands you
the two transcripts. You return one bit: substituted, or not.

```
score = true-positive rate on swap pairs, 0-100
gate  = false accusations above 12.5% of the negatives REJECT the run, score 0
```

Rejected, not penalised, in the same spirit as ecdsafail rejecting a Toffoli win
that came from skipping uncomputation. A detector that flags everything gets a
perfect true-positive rate and scores nothing. Verified: it does.

`harness/evaluate.py` holds the labels, computes the score, and never imports
anything under `detector/`. `benchmark.sh` deletes `decisions.json` and
`score.json` before it runs, so nothing can be pre-seeded.

### The negatives are the hard part

Negatives are **the same weights sampled twice, not the same call twice**. The
harness runs the endpoint and the reference as different *sessions*, different
sampling seeds at temperature 0.7, because that is what auditing a real provider
looks like and because at temperature 0 the whole problem collapses into a byte
comparison that measures nothing. Some negatives compare two non-default sessions
against each other, so a statistic that quietly assumes the reference stream is
always the same draw will fail on them.

### What makes a run invalid

Rejected, not merely scored lower:

- **More false accusations than the budget.** Above.
- **The detector ran unconfined.** No bubblewrap on Linux or `sandbox-exec` on
  macOS means no sandbox, and `harness/evaluate.py` refuses to publish a score
  from an unconfined run. The unconfined path is for local iteration only.
- **A missing or duplicated decision, or decisions from a different corpus
  revision.** The run fails closed. Nothing is scored optimistically.

A rejected detector scores 0 and exits 0, because that is a valid measurement of
an invalid submission. A nonzero exit is reserved for runs that could not produce
a trusted score at all.

### What is closed by construction

- **You cannot read the labels, because they are not here.** The held-out corpus
  is not in this repository. It is materialised inside the scoring runner from a
  secret, as a file rather than an environment variable. This used to say the
  labels were safe because they sat outside `editablePaths`, which was wrong:
  outside-editable means you cannot *write* a file, not that you cannot *read*
  it, and `.git` will hand you any file in the tree regardless.
- **You cannot reach the network.** Your code is a pure function on both ends.
  It emits prompts and reads back strings, and the harness owns every endpoint
  call, so the sandbox denies networking outright.
- **You cannot keep a counter.** Each phase runs in a throwaway directory with
  writes confined to it, SysV IPC denied and the pasteboard denied, so nothing
  survives to the next pair.
- **You cannot learn which model you are judging.** You get two lists of
  strings. Not the tags, not the tier, not the pair id.
- **You cannot pre-compute the probes.** The seed is a SHA-256 of *every* regular
  file you submit under `detector/` (`memory/` and `__pycache__/` excluded), so
  it is not knowable until your code is fixed. Same commitment trick ecdsafail
  uses when it derives test points from a hash of the contestant's own op stream.
  Hashing only `*.py` was not enough: a lookup table in a `.json` rode in free.

## Reference numbers

The baseline in `detector/` is attest-fyi's shipped production rule, ported
unchanged: mean `difflib` similarity across 24 probes, swap called below 0.45.

| Split | Score | TPR | Caught | False accusations | Budget | Where |
|---|---|---|---|---|---|---|
| heldout | **0.00, rejected** | 85.71 | 12/14 | **4** of 8 | 1 | CI, Linux, bubblewrap |
| dev | **0.00, rejected** | 90.91 | 10/11 | **3** of 6 | 1 | macOS, sandbox-exec |

Per tier, held-out: **t1 3/3 · t2 3/3 · t3 3/3 · t4 2/3 · t5 1/2**.

The held-out row is the canonical one, because it comes from the scoring path,
which is Linux under bubblewrap. The same detector on the same corpus scored
78.57 on macOS. Determinism holds **per machine**, and `tools/verify.py` checks
exactly that, but inference is not bit-identical across hardware, so a score is
only comparable to another score from the same runner.

Read that table twice, because it is the whole challenge. The rule finds most of
the substitutions. It also accuses **half the honest providers**, so the run is
rejected and it scores nothing. It is a decent swap detector and not an auditor,
and the difference between those two things is this benchmark.

**Any valid run above zero beats the state of the art.** That is a low bar and a
real one.

### Why no threshold fixes it

`python3 tools/null.py dev` measures both distributions directly:

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

The worst honest pair is *less* similar than the hardest swap. Every threshold
therefore either misses that swap or accuses that provider. 0.45 is not a badly
chosen constant, it is the best available value of a measurement that cannot
work, and moving it trades one failure for the other.

So the opening move is not to tune. It is to measure something else. A mean over
24 comparisons is one number standing in for a distribution, and two sessions of
one model disagree *uniformly* while a requantised engine agrees closely on most
probes and diverges hard on a few. Same mean, different shape.
`detector/memory/01-baseline.md` has the longer version and the untried leads.

### Scope of that finding

Tiers 3 to 5 use `llama3.2:1b` variants, the only family here with fp16/q8/q4
builds small enough for the machine that generated the corpus. 1B models are far
noisier between sessions than the 7B to 70B models real providers serve.
attest-fyi measured 0.59 to 0.69 similarity to true weights against live 7B
endpoints, against 0.42 to 0.55 here. **This corpus is harder than production,
and a rejection here is not evidence that a rule misbehaves against the providers
on the register.** Adding quantisation variants of a larger model is the
highest-value infrastructure contribution available.

## Is it sound?

Two suites, both run in CI.

`python3 tools/verify.py`, 8/8:

- a flag-everything detector is rejected despite a perfect 100.0 TPR
- decisions from a different corpus revision are rejected rather than re-scored
- an unconfined run writes no score at all
- a crashed detector fails closed rather than scoring optimistically
- adding a lookup table to `detector/` moves the probe seed, a memory note does not
- the negatives are genuinely noisy, not byte-identical
- the same detector, config and split produce the same score, both splits run
  twice, identical

`python3 tools/attacks.py`, 26/26 closed, on both sandbox backends. These are not
hypotheticals. An adversarial review of an earlier version **scored 100.00 on the
held-out split without reading a single transcript**, and a second review of the
fixes broke it again:

| Break | Fix |
|---|---|
| The transcript cache sat in `.cache/` keyed on a plain SHA-256 of published constants, so a filename inverted back to a model tag | cache moved out of the repo, keyed with an HMAC under a salt stored outside the cached directory |
| `shutil.copytree` followed symlinks, so `ln -s ../harness detector/leak` copied the answer key into the sandbox | `symlinks=True` plus rejection of any symlink at intake |
| `.git` served `harness/corpus.py` out of the object store, so masking the working tree masked nothing | the held-out corpus left the repository entirely |
| The seed hashed only `*.py`, so a lookup table in a `.json` rode in free | every regular file is hashed |
| `detector/memory/` was excluded from the seed **and still copied in**, so the payload just moved there | `memory/` no longer enters the sandbox at all |
| `PWD` leaked the repo path, and SysV shared memory plus the macOS pasteboard both outlived the per-pair temp dir | explicit minimal child env, `(deny ipc*)` and a pasteboard deny |
| `decisions.json` published `tier` per pair, and tier 0 means *honest*, while CI uploaded it as an artifact | the tier is gone from the output and the file is no longer uploaded |
| **The Linux sandbox had never started.** `bwrap` was missing the ELF loader binds, so every run died on `execvp: No such file or directory` | `/lib` and `/lib64` bound literally rather than through `realpath`, which had deduped them away |
| On macOS a child reads its **parent's** environment via `KERN_PROCARGS2`, which handed over the whole corpus when CI passed it as `ATTEST_HELDOUT_B64` | the corpus is passed as a *path*, never bytes, and `harness/corpus.py` refuses to run at all if a corpus-carrying variable is set |

That last one is worth dwelling on. `(deny process-info*)` and `(deny
sysctl-read)` both SIGTRAP the interpreter, and `(sysctl-name "kern.procargs2")`
is accepted by the parser and silently does nothing, so the channel cannot be
closed from inside the profile. It is closed upstream instead: the secret is
never in the environment, and that is enforced by a hard refusal rather than a
note in a README. `(deny ipc-sysv*)` was the same trap, accepted with no effect,
and it made a test pass while the hole stayed open. `(deny ipc*)` is the one that
works.

The durable fixes were never a better sandbox. They were moving the secret out of
the place the sandbox had to defend, and testing the sandbox by actually starting
it. The strongest backend here was broken for its entire existence and every
green run had come from the weaker one.

### The two backends are not equivalent

`decisions.json` and `score.json` record which one ran.

- **Linux, bubblewrap.** The scoring path, and a true allowlist: the repository
  is not mounted, so the labels are `FileNotFoundError` rather than
  `PermissionError`. `--unshare-all` gives a fresh network, IPC and mount
  namespace per phase.
- **macOS, sandbox-exec.** Local iteration. Reads are allow-by-default with the
  secret directories denied by name, because a scoped read allowlist SIGABRTs the
  interpreter. That is only defensible because the secrets are now a short list
  living outside the repository.

## How to play

```bash
yukon clone <setter>/attest-challenge
cd attest-challenge
yukon setup          # pulls the pinned Ollama tags, the models ARE the corpus
yukon run            # scores you on the held-out split
```

Locally, without Yukon:

```bash
./setup.sh
./benchmark.sh --split dev       # the public corpus, labels visible
```

`./benchmark.sh` with no argument wants the held-out corpus, which is not in this
repository. That is what makes it held out. Locally you get dev, and the held-out
score comes back from the scoring runner.

You need [Ollama](https://ollama.com) running. `setup.sh` pulls five tags, about
14 GB. It runs on CPU, the 7B and 8B models are the slow part, and the transcript
cache means you pay that once.

## What you can edit

Only `detector/`. Two functions.

```python
# detector/probes.py
def generate(seed: int, k: int, history: list[dict]) -> list[dict]:
    """Return up to k probes: [{"id", "category", "prompt"}, ...].
    `history` is every probe and both answers from previous rounds."""

# detector/statistic.py
def substituted(history: list[dict]) -> bool:
    """[{"probe": {...}, "served": "...", "reference": "..."}, ...] -> one bit."""
```

You are called for **two rounds** of **twelve probes** per pair. Round two
receives round one's transcripts, so probing can adapt. The baseline ignores this
entirely, which is the most obvious thing to fix.

`detector/memory/` is inside `editablePaths` deliberately: notes travel with
submissions, so what you learn is what the next solver starts from. Add to it.

## Notes are unverified

Everything in `detector/memory/` was written by other solvers and is not checked
by anyone. Treat it as leads, not facts, and re-run the benchmark before you rely
on a claim in it.

## Openness

This benchmark measures how well substitution can be detected. That knowledge is
symmetric: a strong result tells an honest auditor what to look for and tells a
dishonest operator what to hide. Publishing it anyway is the right call, because
operators already know their own quantisation levels and only the auditors are
currently in the dark. But if you find a technique whose main effect is to make
substitution *harder* to detect, say so in your note rather than optimising it
quietly into the score.

## Credits

Built on [attest.fyi](https://attest.fyi). The probe pool, the similarity
metrics, the thresholds and the Merkle transcript root are ported from that
project's `probes/suite.py`, `scoring/metrics.py`, `scoring/verdict.py` and
`harness/transcripts.py`. The trusted/untrusted split, the sandbox policy, the
seed-from-submission-hash commitment and the reject-don't-penalise rule are
modelled on `Layr-Labs/ecdsafail-challenge`.
