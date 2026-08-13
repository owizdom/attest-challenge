#!/usr/bin/env bash
# Prepare the corpus. The models ARE the benchmark: without them there is
# nothing to detect, so a missing tag fails here rather than silently scoring
# against a smaller corpus.
set -euo pipefail

echo "== attest-challenge setup =="

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama not found. Install from https://ollama.com and re-run." >&2
  exit 1
fi

if ! curl -sf --max-time 5 "${OLLAMA_HOST:-http://localhost:11434}/api/tags" >/dev/null; then
  echo "ollama is installed but not serving. Start it with 'ollama serve'." >&2
  exit 1
fi

TAGS=$(python3 - <<'PY'
from harness import corpus
print("\n".join(corpus.REQUIRED_TAGS))
PY
)

missing=0
while read -r tag; do
  [ -z "$tag" ] && continue
  if ollama list | awk '{print $1}' | grep -qx "$tag"; then
    echo "  ok      $tag"
  else
    echo "  pulling $tag"
    ollama pull "$tag" </dev/null || { echo "  FAILED  $tag" >&2; missing=1; }
  fi
done <<< "$TAGS"

[ "$missing" -eq 0 ] || { echo "corpus incomplete" >&2; exit 1; }

if ! command -v bwrap >/dev/null 2>&1 && [ "$(uname)" != "Darwin" ]; then
  echo
  echo "WARNING: bubblewrap not found. The detector would run unconfined and" >&2
  echo "the run will be REJECTED at scoring. Install bubblewrap." >&2
fi

echo
echo "setup complete."
