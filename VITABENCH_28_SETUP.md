# PAHF -> VitaBench 28-Task Setup

## Current Status

`PAHF` is **not** a native VitaBench runner.

Its released codebase is built around:

- embodied household tasks
- shopping multiple-choice recommendation tasks

So running `python run_agent.py --agent shopping` is **not** equivalent to running the VitaBench 28-task personalization benchmark.

What has been prepared here is the clean baseline package and evaluation surface needed for a rigorous port:

- isolated Python environment: `baseline/PAHF/.venv`
- dedicated API config file: `baseline/PAHF/.env.pahf`
- OpenAI-compatible endpoint support in `utils/llm.py`
- fixed 28-task manifest: `baseline/PAHF/configs/vitabench_28task_manifest.json`
- extracted local 28-task slice target: `baseline/PAHF/data/vitabench/dynamic_delivery_28.tasks.json`
- comparison contract: `baseline/PAHF/json/metric_contract.json`
- preflight check script: `baseline/PAHF/scripts/preflight_pahf.py`

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
- `torch` is required only when you want PAHF's local embedding-based memory stack
- a CPU-only server can still run PAHF native baseline and can also run full memory mode, but full memory mode will be slower

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

## Prepare The Fixed 28-Task Slice

This copies the official VitaBench 28-task subset into the PAHF workspace without modifying the official benchmark files:

```powershell
powershell -ExecutionPolicy Bypass -File baseline\PAHF\scripts\prepare_vitabench_subset.ps1
```

Outputs:

- `baseline/PAHF/data/vitabench/dynamic_delivery_28.tasks.json`
- `baseline/PAHF/data/vitabench/dynamic_delivery_28.summary.json`

## Native PAHF Smoke Run

This does **not** run VitaBench. It only verifies the PAHF package itself:

```powershell
powershell -ExecutionPolicy Bypass -File baseline\PAHF\scripts\run_pahf_native.ps1 -Agent shopping -NoMemory
```

## Evaluation Chain Meaning

For the 28-task benchmark, the scientifically clean interpretation is:

1. Keep PAHF isolated as its own baseline package.
2. Freeze the target 28-task subset with the manifest file.
3. Port or bridge the PAHF method into the VitaBench runner only after the package-level environment and API path are verified.

This avoids the incorrect shortcut of pretending PAHF's native shopping benchmark is already a VitaBench result.
