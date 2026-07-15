#!/usr/bin/env bash
set -euo pipefail

WITH_MEMORY_DEPS=0
if [[ "${1:-}" == "--with-memory-deps" ]]; then
  WITH_MEMORY_DEPS=1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
PYTHON_BIN="$VENV/bin/python"

run_checked() {
  local label="$1"
  shift
  "$PYTHON_BIN" "$@"
  local code=$?
  if [[ $code -ne 0 ]]; then
    echo "Command failed during ${label} (exit code ${code}): $PYTHON_BIN $*" >&2
    exit $code
  fi
}

if [[ ! -x "$PYTHON_BIN" ]]; then
  python3 -m venv "$VENV"
fi

run_checked "pip upgrade" -m pip install --upgrade pip
run_checked "native dependency install" -m pip install -r "$ROOT/requirements.native.txt"

if [[ $WITH_MEMORY_DEPS -eq 1 ]]; then
  run_checked "embedding dependency install" -m pip install -r "$ROOT/requirements.memory.txt"
fi

echo "PAHF virtual environment ready at: $VENV"
echo "Fill API config in: $ROOT/.env.pahf"
echo "You can copy from: $ROOT/.env.pahf.example"
