"""The labelled corpus. TRUSTED — never imported by detector code.

Each entry is one audit: a `served` endpoint and the model it *claims* to be.
Sometimes the claim is true and sometimes the served engine has been swapped,
distilled or requantised. The detector is shown only the two output streams and
must decide. Only this module knows which is which.

Difficulty rises in tiers, and the top two are open problems rather than
exercises:

  1  family swap        llama vs qwen                       solved today
  2  size swap          1b vs 8b in one family              solved today
  3  quantisation       q4 vs q8                            partly solved
  4  precision          fp16 vs q8, IDENTICAL weights       DESIGN.md §10 calls
                                                            this possibly
                                                            undetectable
  5  adversarial        serves the real model to traffic    DESIGN.md §6 calls
                        it recognises as an audit           this the hard ceiling

Negatives are not "the same call twice". They are the same weights under the
nuisance variation a real audit meets — a second session, a different decoding
seed — because a detector that fires on ordinary sampling noise is worse than
no detector at all. That is the null distribution the thresholds in attest-fyi's
`scoring/verdict.py` were measured against.
"""
from __future__ import annotations

import json
import os

# Ollama tags, pinned. Anything not present fails setup rather than silently
# scoring a smaller corpus.
M = {
    "llama1b_fp16": "llama3.2:1b-instruct-fp16",
    "llama1b_q8": "llama3.2:1b-instruct-q8_0",
    "llama1b_q4": "llama3.2:1b-instruct-q4_K_M",
    "llama8b_q4": "llama3.1:8b-instruct-q4_K_M",
    "qwen7b": "qwen2.5:7b-instruct",
}

REQUIRED_TAGS = sorted(set(M.values()))


# Session seeds. A pair is a comparison between two *sessions*, and varying
# which sessions lets one set of weights produce several genuinely distinct
# negatives instead of the same draw over and over. This is what stops the
# score from being decided by four coin flips.
SESSIONS = [1337, 2024, 7717, 9001]


def _pair(pid, tier, served, claimed, label, note, adversarial=False,
          served_seed=None, ref_seed=None):
    return {
        "id": pid,
        "tier": tier,
        "served": served,
        "claimed": claimed,
        "served_seed": served_seed,
        "ref_seed": ref_seed,
        # "swap"  -> the detector MUST fire   (missing it is a false negative)
        # "same"  -> the detector MUST NOT    (firing is a false accusation)
        "label": label,
        "note": note,
        "adversarial": adversarial,
    }


# ---------------------------------------------------------------------------
# Public development corpus. Solvers see these and may tune against them.
# ---------------------------------------------------------------------------
DEV = [
    # tier 1 — different family
    _pair("d-t1-a", 1, M["qwen7b"], M["llama1b_q8"], "swap", "qwen served, llama claimed"),
    _pair("d-t1-b", 1, M["llama1b_q8"], M["qwen7b"], "swap", "llama served, qwen claimed"),
    _pair("d-t1-c", 1, M["qwen7b"], M["llama1b_fp16"], "swap", "qwen served, llama fp16 claimed"),
    # tier 2 — size swap inside a family
    _pair("d-t2-a", 2, M["llama1b_q8"], M["llama8b_q4"], "swap", "1b served, 8b claimed"),
    _pair("d-t2-b", 2, M["llama8b_q4"], M["llama1b_q8"], "swap", "8b served, 1b claimed"),
    _pair("d-t2-c", 2, M["llama1b_fp16"], M["llama8b_q4"], "swap", "1b fp16 served, 8b claimed"),
    # tier 3 — quantisation, two steps apart
    _pair("d-t3-a", 3, M["llama1b_q4"], M["llama1b_fp16"], "swap", "q4 served, fp16 claimed"),
    _pair("d-t3-b", 3, M["llama1b_fp16"], M["llama1b_q4"], "swap", "fp16 served, q4 claimed",
          served_seed=2024),
    # tier 4 — identical weights, one precision step
    _pair("d-t4-a", 4, M["llama1b_fp16"], M["llama1b_q8"], "swap", "fp16 vs q8, same weights"),
    _pair("d-t4-b", 4, M["llama1b_q8"], M["llama1b_q4"], "swap", "q8 vs q4, one step",
          served_seed=2024),
    # tier 5 — the endpoint recognises audit-shaped traffic
    _pair("d-t5-a", 5, M["llama1b_q4"], M["llama1b_fp16"], "swap",
          "cheap engine, honest answers to anything that looks like a probe",
          adversarial=True),
    # negatives — the same weights, different sessions. Several draws per model,
    # because one draw per model is four coin flips deciding the whole gate.
    _pair("d-n-a", 0, M["llama1b_q8"], M["llama1b_q8"], "same", "same weights, s1 vs ref"),
    _pair("d-n-b", 0, M["qwen7b"], M["qwen7b"], "same", "same weights, s1 vs ref"),
    _pair("d-n-c", 0, M["llama1b_q4"], M["llama1b_q4"], "same", "same weights, s2 vs ref",
          served_seed=2024),
    _pair("d-n-d", 0, M["llama1b_fp16"], M["llama1b_fp16"], "same", "same weights, s3 vs ref",
          served_seed=7717),
    _pair("d-n-e", 0, M["llama1b_q8"], M["llama1b_q8"], "same", "same weights, s2 vs s3",
          served_seed=2024, ref_seed=7717),
    _pair("d-n-f", 0, M["llama8b_q4"], M["llama8b_q4"], "same", "same weights, s1 vs ref"),
]

# ---------------------------------------------------------------------------
# The held-out corpus is NOT in this repository.
#
# It used to be, one file away from the detector, and that was the single
# biggest hole in this benchmark: "outside editablePaths" means a submission
# cannot WRITE it, not that it cannot READ it. A contestant could read the
# labels out of their own checkout — or out of `.git`, which no sandbox that
# leaves the repo visible can prevent.
#
# So held-out now means what the word means: absent at authoring time. It is
# resolved at scoring time from a FILE whose path is given in ATTEST_HELDOUT.
#
# A path, never the corpus itself. It used to also accept ATTEST_HELDOUT_B64,
# and on macOS a sandboxed child can read its PARENT's environment through
# KERN_PROCARGS2 — so the corpus was handed to the detector by a sandbox that
# correctly denied every file the corpus lived in. That channel cannot be closed
# from inside the profile (the working deny crashes the interpreter, and the
# targeted sysctl filter is a silent no-op), so the secret must simply never be
# in the environment. CI writes the file first, then runs.
#
# The path itself still leaks, and that is fine: its directory is denied by
# `harness/run.py::_secret_dirs`.
# ---------------------------------------------------------------------------
HELDOUT_ENV = "ATTEST_HELDOUT"

# Anything that could carry corpus BYTES rather than a path. Refuse to run at
# all if one is set, so the property is enforced instead of documented.
_BANNED_ENV = ("ATTEST_HELDOUT_B64", "ATTEST_CORPUS_B64", "ATTEST_HELDOUT_JSON")


def assert_no_secrets_in_env():
    bad = [k for k in _BANNED_ENV if os.environ.get(k)]
    if bad:
        raise SystemExit(
            "REFUSING TO RUN: %s is set.\n"
            "The corpus must never be in the environment — on macOS a sandboxed "
            "detector can read the parent's environment via KERN_PROCARGS2, "
            "which hands it the answer key.\n"
            "Write the corpus to a file and pass %s=<path> instead."
            % (", ".join(bad), HELDOUT_ENV)
        )


def _load_heldout():
    assert_no_secrets_in_env()
    path = os.environ.get(HELDOUT_ENV)
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)["pairs"]
    raise SystemExit(
        "The held-out corpus is not available here, which is the point: it is "
        "not in this repository.\n"
        "  Solvers:  ./benchmark.sh --split dev\n"
        "  Setters:  %s=/path/to/corpus-heldout.json ./benchmark.sh\n"
        "  CI:       decode the secret to a file, then set %s to it"
        % (HELDOUT_ENV, HELDOUT_ENV)
    )


def load(split: str):
    if split == "dev":
        return list(DEV)
    if split == "heldout":
        return list(_load_heldout())
    raise ValueError("split must be 'dev' or 'heldout'")


def counts(pairs):
    return {
        "pairs": len(pairs),
        "swap": sum(1 for p in pairs if p["label"] == "swap"),
        "same": sum(1 for p in pairs if p["label"] == "same"),
    }
