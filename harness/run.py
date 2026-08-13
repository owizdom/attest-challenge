"""Orchestrate one benchmark run. TRUSTED — owns all I/O and all labels.

Sequence, per audit pair:

    1. ask the detector for probes           (sandboxed subprocess, no network)
    2. run them against served and claimed   (here, with the network)
    3. repeat for ROUNDS so probing can adapt
    4. ask the detector for a verdict        (sandboxed subprocess, no network)

The detector is handed two lists of strings and nothing else. It never learns
which endpoint it is judging, never sees the label, and never opens a socket.

Writes `decisions.json`. It does NOT compute the score — that is
`harness/evaluate.py`, which never imports detector code, exactly as
ecdsafail-challenge separates `build_circuit` from `eval_circuit`.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile

from harness import corpus, execute

# Two rounds so a detector can adapt: round two sees round one's transcripts.
#
# Twelve per round = 24 probes, which is not arbitrary: attest-fyi's shipped
# thresholds were measured on a 24-probe suite, and scoring that rule at 16
# probes tests it outside the regime it was calibrated for. The separation
# between "same weights, different session" and a real swap is a mean, and the
# width of the null shrinks with the probe count — so the probe budget is part
# of the problem statement, not a knob.
ROUNDS = 2
PROBES_PER_ROUND = 12
DECISIONS_PATH = "decisions.json"


def detector_digest(root: str = "detector") -> str:
    """A stable hash of the submitted detector.

    The probe seed is derived from this, so a solver cannot know which probes
    will run until after their code is fixed — the same commitment trick
    ecdsafail uses when it derives test points from a hash of the contestant's
    own op stream.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for dirpath, dirnames, filenames in os.walk(root):
            # notes are for humans and must not move the seed; __pycache__ is
            # the interpreter's own scratch and moves on its own, which would
            # make the seed — and therefore the score — nondeterministic.
            dirnames[:] = sorted(d for d in dirnames if d not in ("memory", "__pycache__"))
            if os.path.basename(dirpath) in ("memory", "__pycache__"):
                continue
            for name in sorted(filenames):
                # EVERY file, not just *.py. Hashing only Python let a solver
                # ship a lookup table in a .json without moving the seed —
                # which, with the seed in hand, made the pair order knowable.
                p = os.path.join(dirpath, name)
                if os.path.islink(p) or not os.path.isfile(p):
                    continue
                with open(p, "rb") as f:
                    data = f.read()
                info = tarfile.TarInfo(os.path.relpath(p, root))
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _prepare_scratch(tmp: str) -> str:
    """A throwaway working directory holding the detector and nothing else.

    The detector does NOT run in the repository. It runs here, with a copy of
    `detector/` and the phase shim, because otherwise `from harness import
    corpus` reads the answer key straight off disk — the detector never learns
    which pair it is judging, but the labels themselves must also be out of
    reach, and a fresh directory per call means no state survives between pairs
    either.
    """
    work = os.path.join(tmp, "work")
    os.makedirs(work)
    # symlinks=True, and then reject them: `ln -s ../harness detector/leak` used
    # to copy the answer key into the one directory the sandbox does bind.
    # memory/ is excluded from the seed digest because notes are for humans —
    # which meant an arbitrary payload could ride in there without moving the
    # seed. It is not copied into the sandbox at all now, so the exclusion and
    # the runtime view agree.
    shutil.copytree("detector", os.path.join(work, "detector"), symlinks=True,
                    ignore=shutil.ignore_patterns("memory", "__pycache__"))
    for dirpath, _, filenames in os.walk(os.path.join(work, "detector")):
        for name in filenames + os.listdir(dirpath):
            f = os.path.join(dirpath, name)
            if os.path.islink(f):
                raise SystemExit(
                    "REJECTED: detector/ contains a symlink (%s). Submissions "
                    "must be regular files." % os.path.relpath(f, work)
                )
    shutil.copyfile(
        os.path.join("harness", "untrusted.py"), os.path.join(work, "_shim.py")
    )
    return work


# What the interpreter genuinely needs to start. Everything else stays invisible.
def _runtime_roots() -> list[str]:
    """What the interpreter needs, and nothing that resolves back into the repo.

    Both the symlinked and the resolved path: Homebrew execs python through
    /opt/homebrew/opt/... while realpath() lands in .../Cellar/..., and an
    allowlist that knows only the second one cannot start the interpreter.
    """
    repo = os.path.realpath(os.getcwd())
    cand = {sys.prefix, sys.base_prefix,
            os.path.dirname(sys.executable),
            os.path.dirname(os.path.realpath(sys.executable))}
    for p in list(sys.path) + [sys.prefix, sys.base_prefix]:
        if p and os.path.isdir(p):
            cand.add(p)
            cand.add(os.path.realpath(p))
    forbidden = [repo] + [os.path.realpath(d) for d in _secret_dirs_raw()]
    roots = set()
    for r in cand:
        if not r or r == "/" or not os.path.isdir(r):
            continue
        real = os.path.realpath(r)
        # A root must not BE, CONTAIN, or be contained by anything secret.
        # sys.path holds "" and "." when run as -m from the repo, and a stray
        # PYTHONPATH of the parent directory would remount the whole repo.
        if any(real == f or real.startswith(f + os.sep) or f.startswith(real + os.sep)
               for f in forbidden):
            continue
        roots.add(real)
    return sorted(roots)


# The only genuinely secret state on this machine. Everything in the repository
# is public by construction now — the held-out labels moved out, and the
# transcript cache moved out and is salted — so this list is short and
# enumerable rather than "every secret you happen to own".
def _secret_dirs_raw() -> list[str]:
    d = [os.path.expanduser("~/.attest-challenge"),
         os.path.dirname(execute.CACHE_DIR.rstrip("/")),
         os.path.dirname(execute.SALT_PATH)]
    # ATTEST_HELDOUT can point anywhere. Denying the default location while the
    # corpus actually sits in /tmp protects nothing.
    hp = os.environ.get(corpus.HELDOUT_ENV)
    if hp:
        d.append(os.path.dirname(os.path.realpath(hp)))
    return [x for x in d if x]


def _secret_dirs() -> list[str]:
    return sorted(set(_secret_dirs_raw() + [os.path.realpath(os.getcwd())]))


# The dynamic loader, bound under the names the ELF headers actually use.
#
# These are deliberately NOT run through realpath. On Ubuntu /lib is a symlink
# into /usr/lib, so resolving it deduped /lib and /lib64 away — and without
# /lib64/ld-linux-x86-64.so.2 bwrap cannot exec the interpreter at all. The
# whole Linux path failed with `execvp: No such file or directory` on every run
# until someone started it in a container. A sandbox nobody has run is not a
# sandbox, so tools/attacks.py now checks this.
_LOADER_PATHS = ("/lib", "/lib64", "/usr/lib", "/usr/lib64", "/usr/libexec",
                 "/bin", "/sbin", "/usr/bin", "/usr/sbin", "/etc/alternatives",
                 "/usr/local/lib", "/usr/share/ca-certificates")


def _sandbox_cmd(phase: str, work: str) -> list[str]:
    """Run the untrusted phase with nothing visible but Python and the scratch dir.

    This was originally the other way round — mount everything, then deny two
    directories — and a review broke it in three independent ways within the
    hour: the transcript cache was a model-keyed oracle sitting in `.cache/`,
    `.git` served `harness/corpus.py` straight out of the object store, and the
    repo path leaked through the inherited environment so all of it was easy to
    find. A blocklist on a machine full of secrets is not a sandbox.

    So it is an allowlist now. The repository is not mounted at all. The labels,
    the cache, the git objects and the fixtures are not "denied" — they are not
    on the filesystem this process can see, which is the only version of that
    claim worth writing in a README.

    Linux gets bubblewrap, macOS gets sandbox-exec. When neither exists we fall
    back to an unconfined run and SAY SO — `evaluate.py` refuses to publish a
    score from an unconfined run, so local convenience can never become a
    published number.
    """
    inner = [sys.executable, "_shim.py", phase]

    if sys.platform.startswith("linux") and _which("bwrap"):
        cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
               "--clearenv", "--setenv", "HOME", work, "--setenv", "PATH", "/usr/bin:/bin"]
        repo = os.path.realpath(os.getcwd())
        forbidden = [repo] + [os.path.realpath(d) for d in _secret_dirs_raw()]
        for r in list(_LOADER_PATHS) + _runtime_roots():
            real = os.path.realpath(r)
            if any(real == f or real.startswith(f + os.sep) or f.startswith(real + os.sep)
                   for f in forbidden):
                continue
            cmd += ["--ro-bind-try", r, r]
        cmd += ["--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
                "--bind", work, work, "--chdir", work, *inner]
        return cmd

    if sys.platform == "darwin" and _which("sandbox-exec"):
        # Weaker than the Linux path, and deliberately so rather than
        # accidentally. A scoped read allowlist SIGABRTs the interpreter on
        # macOS (dyld needs paths that cannot be enumerated stably), so reads
        # stay open and the secrets are denied by name instead. That is only
        # defensible because the secrets are now a two-item list living outside
        # the repository. Scoring runs on Linux under bubblewrap, where the
        # repository is not mounted at all; this profile is for local iteration
        # and `decisions.json` records which of the two actually ran.
        deny_read = "".join('(subpath "%s")' % d for d in _secret_dirs())
        profile = (
            "(version 1)"
            "(allow default)"
            "(deny network*)"
            # NOTE: KERN_PROCARGS2 still hands a child the PARENT's environment
            # here. `(deny process-info*)` and `(deny sysctl-read)` both SIGTRAP
            # the interpreter, and `(sysctl-name "kern.procargs2")` is accepted
            # and does nothing. So this is closed upstream instead:
            # `corpus.assert_no_secrets_in_env` refuses to run if the corpus is
            # in the environment at all. Only a path leaks, and its directory is
            # denied below.
            # The pasteboard survived the per-pair temp dir and carried a
            # counter between invocations. `ipc*` does not cover it; mach does.
            "(deny mach-priv*)"
            '(deny mach-lookup (global-name "com.apple.pasteboard.1")'
            ' (global-name "com.apple.pbs.fetch_services")'
            ' (global-name "com.apple.cfprefsd.daemon")'
            ' (global-name "com.apple.distributed_notifications@Uid")'
            ' (global-name "com.apple.system.notification_center"))'
            # SysV shared memory outlived the per-pair TemporaryDirectory and
            # handed a detector a pair counter across invocations.
            # `ipc-sysv*` is accepted by the parser and does nothing — a silent
            # no-op that made this test pass while the hole stayed open.
            # `ipc*` is the one that actually denies shmget.
            "(deny ipc*)"
            "(deny file-read* %s)"
            "(deny file-write*)"
            '(allow file-write* (subpath "%s") (literal "/dev/null") (subpath "/dev/tty"))'
            % (deny_read, work)
        )
        return ["sandbox-exec", "-p", profile, *inner]

    return inner


# The child inherits nothing. PWD alone used to hand over the repository path,
# which is all an exploit needed to go looking for the cache.
def _child_env(work: str) -> dict:
    return {"HOME": work, "PATH": "/usr/bin:/bin", "TMPDIR": work,
            "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"}


def _which(name: str) -> bool:
    from shutil import which

    return which(name) is not None


def sandbox_flavour() -> str | None:
    """'bwrap' | 'sandbox-exec' | None. The two are not equivalent and the
    result records which one, so a number can never be quietly attributed to a
    stronger sandbox than the one that produced it."""
    head = _sandbox_cmd("probes", os.getcwd())[0]
    return head if head in ("bwrap", "sandbox-exec") else None


def sandboxed() -> bool:
    return sandbox_flavour() is not None


def _call(phase: str, payload: dict) -> object:
    """One untrusted phase, in a fresh throwaway directory that is destroyed on
    the way out. Nothing a detector writes survives to the next pair."""
    with tempfile.TemporaryDirectory(prefix="attest-detector-") as tmp:
        work = _prepare_scratch(tmp)
        proc = subprocess.run(
            _sandbox_cmd(phase, work),
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            cwd=work,
            env=_child_env(work),
            timeout=300,
            # own process group, so a forked child cannot outlive the phase
            start_new_session=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            "detector phase %r failed (rc=%d): %s"
            % (phase, proc.returncode, proc.stderr.strip()[:400])
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(
            "detector phase %r returned no usable JSON: %r"
            % (phase, proc.stdout[:200])
        ) from None


def audit_one(pair: dict, seed: int) -> dict:
    history: list[dict] = []
    hashes: list[str] = []

    for rnd in range(ROUNDS):
        probes = _call(
            "probes",
            {"seed": seed + rnd, "k": PROBES_PER_ROUND, "history": history},
        )
        if not probes:
            break
        served, reference = execute.run_pair(pair, probes)
        for pr, s, r in zip(probes, served, reference):
            history.append({"probe": pr, "served": s, "reference": r})
            # An adversarial endpoint answers tell-prompts from the model it
            # CLAIMS. Committing those bytes to pair["served"] would put a false
            # provenance in the root, which defeats the point of publishing one.
            answered = execute.answering_model(pair, pr["prompt"])
            hashes.append(execute.transcript_hash(answered, pr["prompt"], s))
            hashes.append(execute.transcript_hash(pair["claimed"], pr["prompt"], r))

    verdict = _call("decide", {"history": history})
    return {
        "id": pair["id"],
        # NOT the tier. tier 0 means "honest", so publishing it per decision
        # published the answer key — and CI uploaded the file as an artifact.
        # evaluate.py holds the labels and does not need them echoed back.
        "substituted": bool(verdict["substituted"]),
        "probes_used": len(history),
        "merkle": execute.merkle_root(hashes),
    }


def config_digest(pairs) -> str:
    """Fingerprint of everything that decides what a score MEANS: the corpus,
    the probe budget and the decoding. Change any of them and two runs are not
    comparable, so they must not sit in one column of results.tsv."""
    blob = json.dumps({
        "pairs": [(p["id"], p["tier"], p["served"], p["claimed"], p["label"],
                   p.get("served_seed"), p.get("ref_seed")) for p in pairs],
        "rounds": ROUNDS, "probes_per_round": PROBES_PER_ROUND,
        "decoding": execute.DECODING,
    }, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="heldout", choices=["dev", "heldout"])
    args = ap.parse_args()

    corpus.assert_no_secrets_in_env()
    pairs = corpus.load(args.split)
    config = config_digest(pairs)
    digest = detector_digest()
    seed = int(digest[:8], 16)
    # Defence in depth. The detector cannot see the corpus, but if it ever
    # could, a fixed pair order plus a leaked label list would be an index
    # lookup. Shuffling on the submission-derived seed means the order is not
    # knowable before the code is fixed.
    random.Random(seed).shuffle(pairs)

    print("detector  %s" % digest[:16])
    print("cache     %s" % execute.CACHE_DIR)
    print("seed      %d  (derived from the detector, not chosen by it)" % seed)
    print("sandbox   %s" % (sandbox_flavour() or "OFF — unconfined, will not score"))
    print("corpus    %s %s  config %s" % (args.split, corpus.counts(pairs), config[:8]))
    print()

    decisions = []
    for i, pair in enumerate(pairs, 1):
        d = audit_one(pair, seed)
        decisions.append(d)
        print(
            "  [%2d/%d] %-8s  ->  %s"
            % (i, len(pairs), d["id"],
               "SUBSTITUTED" if d["substituted"] else "bound")
        )

    with open(DECISIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": args.split,
                "detector": digest,
                "seed": seed,
                "sandboxed": sandboxed(),
                "sandbox": sandbox_flavour(),
                "config": config,
                "rounds": ROUNDS,
                "probes_per_round": PROBES_PER_ROUND,
                "decisions": decisions,
            },
            f,
            indent=2,
        )
    print("\nwrote %s" % DECISIONS_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
