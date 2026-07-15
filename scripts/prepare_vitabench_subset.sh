#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PAHF virtual environment is missing: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" "$ROOT/scripts/prepare_vitabench_subset.py"
