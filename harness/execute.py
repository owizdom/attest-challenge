"""Run probes against a model and record what came back. TRUSTED.

Ported from attest-fyi `harness/runner.py` and `models/ollama.py`, with two
additions the benchmark needs:

  * a transcript cache, because at temperature 0 with a pinned seed a
    (model, prompt) pair is a pure function. Caching makes a rerun cheap and
    makes determinism testable rather than merely hoped for.
  * the adversarial endpoint, which is the whole of tier 5.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.request

# Outside the repository, and salted.
#
# This used to be `.cache/transcripts` next to the detector, keyed on a plain
# sha256 of (model, prompt, seed, DECODING) — every input of which is published
# in fixtures/thresholds.json. A detector could emit a nonce prompt, then test
# which of the five model tags produced an existing filename, and read the
# served and claimed models straight off the disk. It scored 100.0 without ever
# looking at a transcript.
#
# The sandbox no longer mounts the repo at all, so this is defence in depth
# rather than the fix. Both halves matter: keep trusted state out of the
# contestant's tree, and do not let a filename be a preimage oracle.
CACHE_DIR = os.environ.get(
    "ATTEST_CACHE", os.path.expanduser("~/.cache/attest-challenge/transcripts")
)
# The salt lives OUTSIDE the cache directory, because CI caches that directory
# and a salt inside it travels with the thing it is supposed to protect.
SALT_PATH = os.environ.get(
    "ATTEST_SALT_PATH", os.path.expanduser("~/.attest-challenge/cache-salt"))
_SALT = os.environ.get("ATTEST_CACHE_SALT", "")
if not _SALT:
    try:
        with open(SALT_PATH) as _f:
            _SALT = _f.read().strip()
    except OSError:
        _SALT = hashlib.sha256(os.urandom(32)).hexdigest()
        os.makedirs(os.path.dirname(SALT_PATH), exist_ok=True)
        with open(SALT_PATH, "w") as _f:
            _f.write(_SALT)
        os.chmod(SALT_PATH, 0o600)
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

# Sampling is ON, and this is the single most important decision in the harness.
#
# At temperature 0 the same weights return the same bytes, so "the two streams
# are not identical" would be a perfect detector and the benchmark would measure
# nothing. It would also be a lie about the problem: no provider serves greedy
# decoding, and attest-fyi's thresholds were measured against real endpoints
# where similarity to the TRUE weights sat at 0.59-0.69, not 1.0.
#
# So every audit compares two different sessions. The served endpoint samples
# under its own seed and the auditor's reference run under theirs — for honest
# and swapped pairs alike. A negative is therefore the same weights with real
# sampling noise between them, which is the null distribution a detector has to
# survive, and a detector that fires on ordinary noise is rejected on it.
DECODING = {"temperature": 0.7, "top_p": 0.95, "num_predict": 96}
SERVED_SEED = 1337  # the endpoint's session
REF_SEED = 42  # the auditor's own reference run


def _post(url, body, timeout=600):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _cache_key(model, prompt, seed):
    blob = json.dumps([model, prompt, seed, DECODING], sort_keys=True).encode()
    return hmac.new(_SALT.encode(), blob, hashlib.sha256).hexdigest()


def _cached(key):
    p = os.path.join(CACHE_DIR, key + ".txt")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return f.read()
    return None


def _store(key, value):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, key + ".txt"), "w", encoding="utf-8") as f:
        f.write(value)


def generate(model: str, prompt: str, seed: int, tries: int = 3) -> str:
    """One completion, cached. Errors are returned as sentinels rather than
    raised: a flaky endpoint must not crash a scoring run, and a detector that
    cannot tell an error from a substitution deserves to lose points for it."""
    key = _cache_key(model, prompt, seed)
    hit = _cached(key)
    if hit is not None:
        return hit

    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": dict(DECODING, seed=seed),
    }
    for attempt in range(tries):
        try:
            out = (_post(OLLAMA + "/api/generate", body).get("response") or "").strip()
            out = out or "<EMPTY>"
            _store(key, out)
            return out
        except Exception as e:  # noqa: BLE001 - any transport failure is the same to us
            if attempt == tries - 1:
                return "<ERR %s>" % type(e).__name__
            time.sleep(2)
    return "<ERR unreachable>"


# ---------------------------------------------------------------------------
# Tier 5: the endpoint that recognises the audit
# ---------------------------------------------------------------------------

# Words that make a prompt *look* like a probe rather than ordinary traffic.
# A real evader would use something subtler; this is a deliberately legible
# stand-in for the behaviour, so the benchmark can score it at all.
# The tells live in the held-out corpus, not here. Shipping them in the
# contestant's checkout turned tier 5 into "avoid these nine substrings",
# which is not the same problem as detecting an endpoint that evades you.
_FALLBACK_TELLS = ("one word only", "one line only", "nothing else", "no explanation")


def _looks_like_an_audit(prompt: str, tells=None) -> bool:
    p = prompt.lower()
    return any(t in p for t in (tells or _FALLBACK_TELLS))


def answering_model(pair, prompt: str) -> str:
    """Which model's weights actually produced the served output for `prompt`.
    Usually pair["served"]; for an adversarial endpoint that recognised the
    probe, the model it claims to be."""
    if pair.get("adversarial") and _looks_like_an_audit(prompt, pair.get("audit_tells")):
        return pair["claimed"]
    return pair["served"]


def run_pair(pair, probes):
    """Return (served_outputs, reference_outputs) aligned to `probes`.

    The reference is the model the endpoint *claims* to be — exactly what a real
    auditor holds. The detector never learns which of the two it is looking at.
    """
    served, claimed = pair["served"], pair["claimed"]
    s_seed = pair.get("served_seed") or SERVED_SEED
    r_seed = pair.get("ref_seed") or REF_SEED
    served_out, ref_out = [], []
    for pr in probes:
        prompt = pr["prompt"]
        if pair.get("adversarial") and _looks_like_an_audit(prompt, pair.get("audit_tells")):
            # Caught us looking. Answer honestly, from the model we claim — but
            # from the served session's seed, because an endpoint that returned
            # the auditor's own bytes back would give itself away immediately.
            served_out.append(generate(claimed, prompt, s_seed))
        else:
            served_out.append(generate(served, prompt, s_seed))
        ref_out.append(generate(claimed, prompt, r_seed))
    return served_out, ref_out


def transcript_hash(model, prompt, output):
    blob = json.dumps([model, prompt, output], sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def merkle_root(hashes):
    """Ported from attest-fyi harness/transcripts.py. One root per run so a
    result can be disputed probe-by-probe without republishing the whole set."""
    if not hashes:
        return "sha256:" + hashlib.sha256(b"").hexdigest()
    layer = list(hashes)
    while len(layer) > 1:
        if len(layer) % 2:
            layer.append(layer[-1])
        layer = [
            hashlib.sha256((layer[i] + layer[i + 1]).encode()).hexdigest()
            for i in range(0, len(layer), 2)
        ]
    return "sha256:" + layer[0]
