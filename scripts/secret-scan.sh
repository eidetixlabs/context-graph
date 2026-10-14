#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

patterns=(
  '523271873094'
  'license\.eidetix\.net'
  'hf_[A-Za-z0-9]{20,}'
  'AKIA[0-9A-Z]{16}'
)

failed=0
for pattern in "${patterns[@]}"; do
  if grep -RInE --exclude-dir=.git --exclude-dir=.venv --exclude-dir=__pycache__ \
      --exclude='LICENSE' --exclude='*.md' --exclude='secret-scan.sh' "$pattern" . ; then
    echo "secret-scan: matched $pattern" >&2
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi
echo "secret-scan ok"
