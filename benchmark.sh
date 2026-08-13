#!/usr/bin/env bash
# Score a submission.
#
#   1. Wipe stale decisions.json / score.json so a contestant cannot pre-seed
#      them.
#   2. Run the detector (UNTRUSTED — imports detector/) inside a sandbox with
#      no network and a read-only filesystem, in its own process group. It is a
#      pure function on both ends: it emits probes, and it reads back two lists
#      of strings. Every endpoint call belongs to the harness, so a detector
#      cannot phone home, cannot discover which model it is judging, and cannot
#      read the labels.
#   3. Verify decisions.json exists. If the detector crashed, it is missing and
#      we fail closed.
#   4. Run the evaluator (TRUSTED — never imports detector code), which holds
#      the labels and writes the canonical score.json.
#
# Arguments are forwarded to the run step (e.g. --split dev).
set -euo pipefail

rm -f decisions.json score.json

python3 -m harness.run "$@"

[ -f decisions.json ] || { echo "decisions.json missing — the run failed" >&2; exit 1; }

python3 -m harness.evaluate
