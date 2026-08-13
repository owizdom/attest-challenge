"""The only place detector code is ever imported. Runs sandboxed, no network.

Two phases, both pure functions over JSON on stdin/stdout:

    probes   {seed, k, history}  ->  [{id, category, prompt}, ...]
    decide   {history}           ->  {"substituted": bool}

Keeping the detector pure is what lets the sandbox deny the network outright.
The harness owns every endpoint call, so a detector cannot phone home, cannot
look up which model it is judging, and cannot read the labels — it only ever
sees two lists of strings.

`history` carries prior rounds so a detector can probe adaptively: round two
may follow up on what round one revealed, without the detector ever holding a
socket.
"""
from __future__ import annotations

import json
import sys


def main() -> int:
    phase = sys.argv[1]
    payload = json.loads(sys.stdin.read())

    if phase == "probes":
        from detector import probes

        out = probes.generate(
            seed=payload["seed"], k=payload["k"], history=payload.get("history") or []
        )
        if not isinstance(out, list):
            raise SystemExit("detector.probes.generate must return a list")
        cleaned = []
        for i, p in enumerate(out[: payload["k"]]):
            if not isinstance(p, dict) or "prompt" not in p:
                raise SystemExit("probe %d is missing 'prompt'" % i)
            cleaned.append(
                {
                    "id": str(p.get("id", "p%d" % i)),
                    "category": str(p.get("category", "generic")),
                    "prompt": str(p["prompt"])[:2000],
                }
            )
        json.dump(cleaned, sys.stdout)
        return 0

    if phase == "decide":
        from detector import statistic

        verdict = statistic.substituted(payload["history"])
        json.dump({"substituted": bool(verdict)}, sys.stdout)
        return 0

    raise SystemExit("unknown phase %r" % phase)


if __name__ == "__main__":
    raise SystemExit(main())
