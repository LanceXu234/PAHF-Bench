#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WITH_MEMORY_DEPS=0

for arg in "$@"; do
  case "$arg" in
    --with-memory-deps)
      WITH_MEMORY_DEPS=1
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$ROOT/.env.pahf" ]]; then
  cp "$ROOT/.env.pahf.example" "$ROOT/.env.pahf"
fi

if [[ $WITH_MEMORY_DEPS -eq 1 ]]; then
  bash "$ROOT/scripts/setup_pahf_env.sh" --with-memory-deps
else
  bash "$ROOT/scripts/setup_pahf_env.sh"
fi

bash "$ROOT/scripts/prepare_vitabench_subset.sh"
"$ROOT/.venv/bin/python" "$ROOT/scripts/preflight_pahf.py" || true

cat <<EOF

Cloud bootstrap finished.

Next:
1. Edit $ROOT/.env.pahf and fill PAHF_OPENAI_API_KEY / PAHF_OPENAI_BASE_URL.
2. Re-run preflight:
   $ROOT/.venv/bin/python $ROOT/scripts/preflight_pahf.py
3. Native PAHF smoke:
   bash $ROOT/scripts/run_pahf_native.sh shopping --no-memory

Note:
- Without --with-memory-deps, this setup is for native no-memory / baseline execution.
- Full PAHF memory mode requires local embedding dependencies and should be bootstrapped with --with-memory-deps.
- The VitaBench 28-task subset has been prepared locally, but a real VitaBench result still requires the PAHF->VitaBench bridge.
EOF
