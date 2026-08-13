"""The exploits that broke this benchmark, kept as regression tests.

An adversarial review scored **100.00 on the held-out split without reading a
single transcript**, three different ways. Each of those is a test now, because
a sandbox claim that is not re-run every time is a sandbox claim that quietly
stops being true.

    python3 tools/attacks.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.getcwd())
from harness import execute, run  # noqa: E402

results = []


def check(name, blocked, detail=""):
    results.append((name, blocked, detail))
    print("%s %-46s %s" % ("  BLOCKED " if blocked else "  OPEN    ", name, detail))


PROBE = r'''
import glob, json, os, socket, subprocess, sys
# Absolute paths passed in from the parent. Using os.path.expanduser("~") here
# resolves to the SCRATCH dir (HOME is rewritten), so the probe checked paths
# that never exist and reported BLOCKED for the wrong reason.
REPO, CACHE, HELDOUT, SALT = "@@REPO@@", "@@CACHE@@", "@@HELDOUT@@", "@@SALT@@"
out = {}
def t(k, fn):
    try: out[k] = ("LEAK", fn())
    except Exception as e: out[k] = ("BLOCKED", type(e).__name__)

t("labels_file", lambda: len(open(os.path.join(REPO,"harness","corpus.py")).read()))
t("labels_import", lambda: (sys.path.insert(0,REPO), __import__("harness.corpus"))[0])
t("repo_listdir", lambda: len(os.listdir(REPO)))
t("git_objects", lambda: len(os.listdir(os.path.join(REPO,".git"))))
t("git_catfile", lambda: len(subprocess.run(
    ["git","-C",REPO,"cat-file","-p","HEAD:harness/corpus.py"],
    capture_output=True, timeout=20).stdout) or (_ for _ in ()).throw(OSError()))
t("cache_in_repo", lambda: len(glob.glob(os.path.join(REPO,".cache","transcripts","*")))
                           or (_ for _ in ()).throw(OSError()))
t("cache_outside", lambda: len(glob.glob(os.path.join(CACHE,"*")))
                           or (_ for _ in ()).throw(OSError()))
t("heldout_file", lambda: len(open(HELDOUT).read()))
t("cache_salt", lambda: len(open(SALT).read()))
# Exact path components, not a substring: a scratch dir like
# /tmp/attest-detector-x/work contains "/work", which matched a repo at /work.
def _env_repo():
    hits = [k for k, v in os.environ.items()
            if os.path.realpath(str(v)) == REPO
            or str(v).startswith(REPO + os.sep)]
    return hits or (_ for _ in ()).throw(OSError("no env var points at the repo"))
t("repo_path_in_env", _env_repo)
# The PARENT's environment, via KERN_PROCARGS2 / /proc. This is where the
# held-out corpus escaped a sandbox that denied every file it lived in.
def _parent_env():
    hits = []
    try:
        with open("/proc/1/environ", "rb") as f:
            blob = f.read().decode("utf-8", "ignore")
        hits = [x for x in blob.split("\0") if x.startswith(("ATTEST_", "PWD="))]
    except OSError:
        import ctypes, ctypes.util
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        mib = (ctypes.c_int * 3)(1, 49, os.getppid())   # CTL_KERN, KERN_PROCARGS2
        n = ctypes.c_size_t(1 << 20); buf = ctypes.create_string_buffer(n.value)
        if libc.sysctl(mib, 3, buf, ctypes.byref(n), None, 0) != 0:
            raise OSError(ctypes.get_errno(), "procargs denied")
        blob = buf.raw[:n.value].decode("utf-8", "ignore")
        hits = [x for x in blob.split("\0") if x.startswith(("ATTEST_", "PWD="))]
    # The parent env IS readable on macOS and cannot be closed from inside the
    # profile. The invariant is therefore narrower and enforced upstream: it may
    # contain PATHS (whose directories are denied) but never corpus BYTES.
    # corpus.assert_no_secrets_in_env refuses to run otherwise.
    leaked = []
    for h in hits:
        k, _, v = h.partition("=")
        if k == "PWD":
            continue                      # the repo path; the repo is denied
        if k == "ATTEST_HELDOUT":
            if not os.path.isdir(v) and not v.endswith(".json"):
                leaked.append(k)
            continue                      # a path; its directory is denied
        if k == "ATTEST_CACHE" or k == "ATTEST_SALT_PATH":
            continue                      # paths, both denied
        leaked.append(k + "=" + v[:24])
    if leaked:
        return leaked
    raise OSError("parent env carries paths only, no corpus bytes")
t("parent_env_carries_no_secret", _parent_env)
# The pasteboard outlived the per-pair temp dir on macOS.
def _pb():
    subprocess.run(["pbcopy"], input=b"ATTEST-COUNTER", timeout=10, check=True)
    return subprocess.run(["pbpaste"], capture_output=True, timeout=10).stdout.decode()[:20]
t("pasteboard", _pb)
t("network", lambda: socket.create_connection(("127.0.0.1",11434), timeout=3).close())
# MARKER is written on pass 1 and read back on pass 2, from two separate
# sandboxed invocations. Writability alone proves nothing — under bwrap /tmp is
# a fresh tmpfs per call, so the write succeeds and the bytes are gone.
def _persist(path):
    prior = ""
    try:
        with open(path) as f:
            prior = f.read()
    except OSError:
        pass
    with open(path, "a") as f:
        f.write("M")
    if prior:
        return "SURVIVED %d prior marks" % len(prior)
    raise OSError("wrote, but nothing survived from the previous invocation")
t("persist_tmp", lambda: _persist("/tmp/attest-ctr"))
t("persist_home", lambda: _persist(os.path.expanduser("~/attest-ctr")))
t("persist_repo", lambda: _persist(os.path.join(REPO,"attest-ctr")))
def _shm():
    # The real syscall, and the real property: does a segment created by the
    # PREVIOUS invocation still exist? Under bwrap --unshare-all the IPC
    # namespace is fresh, so shmget succeeds and the segment is useless.
    import ctypes, ctypes.util
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    existing = libc.shmget(0x4154541, 32, 0o600)          # no IPC_CREAT
    created = libc.shmget(0x4154541, 32, 0o1000 | 0o600)  # IPC_CREAT | 0600
    if created < 0:
        raise OSError(ctypes.get_errno(), "shmget denied outright")
    if existing >= 0:
        return "SURVIVED, shmid %d" % existing
    raise OSError("created, but nothing survived from the previous invocation")
t("sysv_shm", _shm)
print(json.dumps(out), file=sys.stderr)
'''


def sandbox_probes():
    """Findings 1, 3, 5, 6, 7: what a detector can see from inside.

    Anything written into detector/ here MUST be removed: a stray file changes
    the submission digest and therefore the probe seed and the score.
    """
    src = "from detector import _atk  # noqa\n"
    heldout = os.environ.get("ATTEST_HELDOUT",
                             os.path.expanduser("~/.attest-challenge/corpus-heldout.json"))
    body = (PROBE
            .replace("@@REPO@@", os.path.realpath(os.getcwd()))
            .replace("@@CACHE@@", execute.CACHE_DIR)
            .replace("@@HELDOUT@@", heldout)
            .replace("@@SALT@@", execute.SALT_PATH))
    with open("detector/_atk.py", "w") as f:
        f.write(body)
    orig = open("detector/probes.py").read()
    passes = []
    try:
        open("detector/probes.py", "w").write(
            orig.replace("import random", "import random\n" + src, 1))
        # Twice, in two separate throwaway directories, exactly as two
        # consecutive pairs would run. The second pass is what makes the
        # persistence checks meaningful.
        for _ in range(2):
            with tempfile.TemporaryDirectory(prefix="attest-detector-") as tmp:
                work = run._prepare_scratch(tmp)
                p = subprocess.run(run._sandbox_cmd("probes", work),
                                   input=json.dumps({"seed": 1, "k": 2, "history": []}),
                                   capture_output=True, text=True, cwd=work,
                                   env=run._child_env(work), timeout=300,
                                   start_new_session=True)
                passes.append(p)
        p = passes[-1]
    finally:
        open("detector/probes.py", "w").write(orig)
        if os.path.exists("detector/_atk.py"):
            os.remove("detector/_atk.py")

    line = [l for l in p.stderr.splitlines() if l.startswith("{")]
    if not line:
        check("sandbox probe ran", False, "no output: %s" % p.stderr.strip()[:160])
        return
    got = json.loads(line[-1])
    for k, (state, detail) in sorted(got.items()):
        check(k, state == "BLOCKED", str(detail)[:60])


def symlink_escape():
    """Finding 2: `ln -s ../harness detector/leak` copied the answer key in."""
    link = "detector/_leak"
    os.symlink("../harness", link)
    try:
        with tempfile.TemporaryDirectory() as t:
            run._prepare_scratch(t)
        check("symlink into detector/", False, "copytree followed it")
    except SystemExit:
        check("symlink into detector/", True, "rejected at intake")
    finally:
        os.remove(link)


def non_py_payload():
    """Finding 4: a lookup table in .json rode in without moving the seed."""
    before = run.detector_digest()
    with open("detector/_payload.json", "w") as f:
        f.write("x" * 3200)
    try:
        after = run.detector_digest()
    finally:
        os.remove("detector/_payload.json")
    check("non-.py payload moves the seed", before != after,
          "%s -> %s" % (before[:8], after[:8]))


def memory_payload():
    """memory/ is excluded from the seed digest, so a payload there would ride
    in for free. It must therefore not reach the sandbox at all."""
    os.makedirs("detector/memory", exist_ok=True)
    with open("detector/memory/_payload.json", "w") as f:
        f.write("x" * 4096)
    try:
        with tempfile.TemporaryDirectory() as t:
            work = run._prepare_scratch(t)
            leaked = os.path.exists(os.path.join(work, "detector", "memory", "_payload.json"))
    finally:
        os.remove("detector/memory/_payload.json")
    check("memory/ payload cannot reach the sandbox", not leaked,
          "excluded from the digest AND from the copy")


def cache_is_not_an_oracle():
    """Finding 1: the key was sha256 over published constants, so a filename
    could be inverted back to a model tag."""
    k1 = execute._cache_key("qwen2.5:7b-instruct", "hello", 1337)
    import hashlib
    naive = hashlib.sha256(json.dumps(
        ["qwen2.5:7b-instruct", "hello", 1337, execute.DECODING], sort_keys=True
    ).encode()).hexdigest()
    check("cache key is not invertible", k1 != naive, "salted, not plain sha256")
    check("salt lives outside the cached dir",
          not os.path.realpath(execute.SALT_PATH).startswith(
              os.path.realpath(execute.CACHE_DIR)),
          execute.SALT_PATH)
    check("cache lives outside the repo",
          not os.path.realpath(execute.CACHE_DIR).startswith(os.path.realpath(os.getcwd())),
          execute.CACHE_DIR)


NEEDLE = "h-" + "t5" + "-a"   # split so this file is not itself a match


def decisions_leak_no_labels():
    """tier 0 means honest, so a tier per decision is the answer key — and CI
    uploads decisions.json as an artifact."""
    if not os.path.exists("decisions.json"):
        check("decisions.json carries no label", True, "no run artifact present")
        return
    d = json.load(open("decisions.json"))
    keys = set().union(*[set(x) for x in d["decisions"]]) if d["decisions"] else set()
    check("decisions.json carries no label", "tier" not in keys,
          "keys: %s" % ", ".join(sorted(keys)))


def heldout_absent():
    """Finding 3, the fundamental half: held-out must mean absent, not unwritable."""
    src = open("harness/corpus.py").read()
    check("held-out labels are not in the repo", "HELDOUT = [" not in src,
          "resolved from %s at scoring time" % run.corpus.HELDOUT_ENV)
    # Only TRACKED files matter. decisions.json and score.json are gitignored
    # run artifacts that exist on the scoring host after a run, and the sandbox
    # cannot read the repo anyway.
    tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
    # tools/ is NOT exempt: exempting the only tracked directory that contains
    # a held-out id is writing the test around the failure. The probe id below
    # is built at runtime so this file does not contain it literally.
    files = tracked.stdout.split()
    leaked = []
    for f in files:
        try:
            if NEEDLE in open(f, encoding="utf-8", errors="ignore").read():
                leaked.append(f)
        except OSError:
            pass
    check("no held-out pair id in any tracked file",
          not leaked, ", ".join(leaked)[:80] or ("%d files scanned" % len(files)))


if __name__ == "__main__":
    print("\n== attest-challenge: the exploits, re-run ==\n")
    print("sandbox: %s\n" % (run.sandbox_flavour() or "NONE — unconfined"))
    sandbox_probes()
    print()
    symlink_escape()
    non_py_payload()
    memory_payload()
    cache_is_not_an_oracle()
    decisions_leak_no_labels()
    heldout_absent()
    open_ = [n for n, b, _ in results if not b]
    print("\n%d/%d closed" % (len(results) - len(open_), len(results)))
    for n in open_:
        print("  STILL OPEN: %s" % n)
    raise SystemExit(1 if open_ else 0)
