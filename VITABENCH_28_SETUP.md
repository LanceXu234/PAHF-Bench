# PAHF -> VitaBench Native 28-Task Setup

## Current Status

This repository now contains a native VitaBench bridge for PAHF.

The bridge is designed to keep VitaBench's official evaluation path intact:

- VitaBench personalization workflow
- VitaBench evaluator
- VitaBench metrics
- VitaBench raw results structure

and replace only the memory backend with a PAHF-style memory module.

The prepared surfaces are:

- isolated Python environment: `baseline/PAHF/.venv`
- dedicated API config file: `baseline/PAHF/.env.pahf`
- OpenAI-compatible endpoint support in `utils/llm.py`
- fixed 28-task manifest: `baseline/PAHF/configs/vitabench_28task_manifest.json`
- extracted local 28-task slice target: `baseline/PAHF/data/vitabench/dynamic_delivery_28.tasks.json`
- comparison contract: `baseline/PAHF/json/metric_contract.json`
- preflight check script: `baseline/PAHF/scripts/preflight_pahf.py`
- native VitaBench runner: `baseline/PAHF/scripts/run_vitabench_pahf.py`
- one-command bridge launcher: `baseline/PAHF/scripts/run_vitabench_pahf_28.sh`

## Dependency Split

There are now two runtime layers:

- `requirements.native.txt`
  - enough for PAHF package import and native no-memory / baseline execution
  - includes `faiss-cpu` because the released agents import `faiss` at module import time
- `requirements.memory.txt`
  - extra local embedding stack for full PAHF memory
  - includes `torch` and `transformers`

Important:

- `torch` is **not** required just to call an API
- `torch` is required when you want PAHF's local embedding-based memory stack
- the VitaBench bridge uses PAHF memory, so it also requires `requirements.memory.txt`
- a CPU-only server can still run the VitaBench bridge, but the first run will be slower

## Where To Fill The API

Copy:

```powershell
Copy-Item baseline\PAHF\.env.pahf.example baseline\PAHF\.env.pahf
```

Then fill:

- `PAHF_OPENAI_API_KEY`
- `PAHF_OPENAI_BASE_URL` if you use a third-party OpenAI-compatible service

Optional:

- `PAHF_AGENT_MODEL`
- `PAHF_HUMAN_MODEL`
- `PAHF_OPENAI_TIMEOUT`
- `PAHF_OPENAI_MAX_RETRIES`
- `PAHF_VITA_*` bridge variables if you want to override defaults

## Clean Environment

The isolated environment is:

- `baseline/PAHF/.venv`

Setup command:

```powershell
powershell -ExecutionPolicy Bypass -File baseline\PAHF\scripts\setup_pahf_env.ps1
```

If you later need the full PAHF memory stack:

```powershell
powershell -ExecutionPolicy Bypass -File baseline\PAHF\scripts\setup_pahf_env.ps1 -WithMemoryDeps
```

For the Linux cloud path, use:

```bash
bash scripts/bootstrap_vitabench_pahf.sh
```

## Preflight

Run:

```powershell
baseline\PAHF\.venv\Scripts\python.exe baseline\PAHF\scripts\preflight_pahf.py
```

This checks:

- API config presence
- core package availability
- optional memory-package availability
- 28-task manifest presence

## Fixed 28-Task Slice

The fixed task slice is already stored locally at:

- `baseline/PAHF/data/vitabench/dynamic_delivery_28.tasks.json`

So the 28-task run does **not** require downloading the full personalization dataset just to launch this benchmark.

## Native VitaBench Smoke Run

Minimal smoke run:

```bash
PAHF_VITA_LIMIT_TASKS=1 PAHF_VITA_NUM_TRIALS=1 PAHF_VITA_MAX_STEPS=20 bash scripts/run_vitabench_pahf_28.sh
```

Full 28-task run:

```bash
bash scripts/run_vitabench_pahf_28.sh
```

## Output Files

Each run writes a timestamped directory under:

- `baseline/PAHF/runs/vitabench_pahf/`

with:

- `official_results.json`
- `official_metrics.json`
- `subtask_records.jsonl`
- `subtask_records.csv`
- `skill_tag_metrics.json`
- `run_manifest.json`
- `SUMMARY.md`

This keeps both the official benchmark metrics and the fine-grained subtask-level traces for later analysis.
