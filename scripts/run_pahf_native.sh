#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_pahf_native.sh <shopping|embodied> [--mem-style sql|faiss] [--no-memory] [--model MODEL] [--human-model MODEL]" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$ROOT/.venv/bin/python"
ENV_FILE="$ROOT/.env.pahf"

if [[ ! -x "$VENV_PY" ]]; then
  echo "PAHF virtual environment is missing: $VENV_PY" >&2
  exit 1
fi

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

AGENT="$1"
shift

cd "$ROOT"
"$VENV_PY" run_agent.py --agent "$AGENT" "$@"
