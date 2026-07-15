#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "$ROOT/.env.pahf" ]]; then
  echo "Missing $ROOT/.env.pahf" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ROOT/.env.pahf"
set +a

VITABENCH_ROOT="${PAHF_VITABENCH_ROOT:-${VITABENCH_ROOT:-$ROOT/external/vitabench-2.0}}"
TASK_FILE="${PAHF_VITA_TASK_FILE:-$ROOT/data/vitabench/dynamic_delivery_28.tasks.json}"

AGENT_LLM="${PAHF_VITA_AGENT_LLM:-${PAHF_AGENT_MODEL:-gpt-4o-mini}}"
USER_LLM="${PAHF_VITA_USER_LLM:-${PAHF_HUMAN_MODEL:-$AGENT_LLM}}"
EVAL_LLM="${PAHF_VITA_EVALUATOR_LLM:-$AGENT_LLM}"
NUM_TRIALS="${PAHF_VITA_NUM_TRIALS:-4}"
MAX_STEPS="${PAHF_VITA_MAX_STEPS:-300}"
MAX_ERRORS="${PAHF_VITA_MAX_ERRORS:-10}"
MAX_CONCURRENCY="${PAHF_VITA_MAX_CONCURRENCY:-2}"
SEED="${PAHF_VITA_SEED:-300}"
MEMORY_BANK_TYPE="${PAHF_VITA_MEMORY_BACKEND:-sql}"
RUN_NAME="${PAHF_VITA_RUN_NAME:-}"
RUN_ROOT="${PAHF_VITA_RUN_ROOT:-$ROOT/runs/vitabench_pahf}"
LIMIT_TASKS="${PAHF_VITA_LIMIT_TASKS:-}"
LANGUAGE="${PAHF_VITA_LANGUAGE:-chinese}"
EVALUATION_TYPE="${PAHF_VITA_EVALUATION_TYPE:-trajectory}"
LOG_LEVEL="${PAHF_VITA_LOG_LEVEL:-INFO}"

if [[ "$VITABENCH_ROOT" != /* ]]; then
  VITABENCH_ROOT="$ROOT/${VITABENCH_ROOT#./}"
fi

if [[ "$TASK_FILE" != /* ]]; then
  TASK_FILE="$ROOT/${TASK_FILE#./}"
fi

if [[ "$RUN_ROOT" != /* ]]; then
  RUN_ROOT="$ROOT/${RUN_ROOT#./}"
fi

if [[ ! -d "$VITABENCH_ROOT" ]]; then
  echo "Missing VitaBench checkout at: $VITABENCH_ROOT" >&2
  echo "Run: bash $ROOT/scripts/bootstrap_vitabench_pahf.sh" >&2
  exit 1
fi

if [[ ! -f "$TASK_FILE" ]]; then
  echo "Missing task file: $TASK_FILE" >&2
  exit 1
fi

"$ROOT/.venv/bin/python" "$ROOT/scripts/sync_vitabench_models.py" --vitabench-root "$VITABENCH_ROOT"

ARGS=(
  --vitabench-root "$VITABENCH_ROOT"
  --task-file "$TASK_FILE"
  --agent-llm "$AGENT_LLM"
  --user-llm "$USER_LLM"
  --evaluator-llm "$EVAL_LLM"
  --num-trials "$NUM_TRIALS"
  --max-steps "$MAX_STEPS"
  --max-errors "$MAX_ERRORS"
  --max-concurrency "$MAX_CONCURRENCY"
  --seed "$SEED"
  --language "$LANGUAGE"
  --evaluation-type "$EVALUATION_TYPE"
  --log-level "$LOG_LEVEL"
  --memory-bank-type "$MEMORY_BANK_TYPE"
  --run-root "$RUN_ROOT"
)

if [[ -n "$RUN_NAME" ]]; then
  ARGS+=(--run-name "$RUN_NAME")
fi

if [[ -n "$LIMIT_TASKS" ]]; then
  ARGS+=(--limit-tasks "$LIMIT_TASKS")
fi

if [[ "${PAHF_VITA_DISABLE_LLM_EXTRACTION:-0}" == "1" ]]; then
  ARGS+=(--disable-llm-extraction)
fi

if [[ "${PAHF_VITA_DISABLE_LLM_QUESTIONS:-${PAHF_VITA_DISABLE_LLM_QUESTION:-0}}" == "1" ]]; then
  ARGS+=(--disable-llm-questions)
fi

if [[ "${PAHF_VITA_ENABLE_OUTCOME_REWARD:-0}" == "1" ]]; then
  ARGS+=(--enable-outcome-reward)
fi

"$ROOT/.venv/bin/python" "$ROOT/scripts/run_vitabench_pahf.py" "${ARGS[@]}"
