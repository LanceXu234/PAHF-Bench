#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_VITABENCH_COMMIT="0c481ae1f97b973fe0fa0d30e78ab463c4d59193"

if [[ ! -f "$ROOT/.env.pahf" ]]; then
  cp "$ROOT/.env.pahf.example" "$ROOT/.env.pahf"
fi

set -a
# shellcheck disable=SC1090
source "$ROOT/.env.pahf"
set +a

VITABENCH_ROOT="${PAHF_VITABENCH_ROOT:-${VITABENCH_ROOT:-$ROOT/external/vitabench-2.0}}"
VITABENCH_COMMIT="${PAHF_VITABENCH_COMMIT:-$DEFAULT_VITABENCH_COMMIT}"

if [[ "$VITABENCH_ROOT" != /* ]]; then
  VITABENCH_ROOT="$ROOT/${VITABENCH_ROOT#./}"
fi

if [[ ! -d "$VITABENCH_ROOT/.git" ]]; then
  git clone https://github.com/meituan-longcat/vitabench-2.0 "$VITABENCH_ROOT"
fi

git -C "$VITABENCH_ROOT" fetch origin
git -C "$VITABENCH_ROOT" checkout "$VITABENCH_COMMIT"

bash "$ROOT/scripts/setup_pahf_env.sh" --with-memory-deps
"$ROOT/.venv/bin/python" -m pip install -e "$VITABENCH_ROOT"
"$ROOT/.venv/bin/python" "$ROOT/scripts/sync_vitabench_models.py" --vitabench-root "$VITABENCH_ROOT"
"$ROOT/.venv/bin/python" "$ROOT/scripts/preflight_pahf.py"

cat <<EOF

VitaBench bridge bootstrap finished.

Next:
1. Check/confirm API config in: $ROOT/.env.pahf
2. Run the 28-task PAHF benchmark:
   bash $ROOT/scripts/run_vitabench_pahf_28.sh

Optional:
- Override VitaBench root with PAHF_VITABENCH_ROOT in .env.pahf
- Override pinned VitaBench commit with PAHF_VITABENCH_COMMIT in .env.pahf
EOF
