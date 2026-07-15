# PAHF Cloud Server Quickstart

This package is prepared so you can clone it on a Linux server and bootstrap it with one command.

## What You Can Run Today

- native PAHF package verification
- native PAHF no-memory baseline
- full PAHF native memory mode, if the server can support local embedding inference
- native VitaBench 2.0 personalization evaluation with a PAHF memory bridge on the fixed 28-task slice

## Recommended Path For VitaBench

If your goal is the benchmark result we discussed, use the VitaBench bridge route instead of `run_agent.py`.

The fixed task slice already shipped in this repository is:

- `configs/vitabench_28task_manifest.json`
- `data/vitabench/dynamic_delivery_28.tasks.json`

The bridge keeps VitaBench's own:

- personalization workflow
- evaluator
- metrics
- raw result format

and only swaps the memory backend to PAHF-style retrieval and preference extraction.

## One-Command Bootstrap

No-memory / baseline-ready environment:

```bash
bash scripts/bootstrap_cloud_pahf.sh
```

Full PAHF memory environment:

```bash
bash scripts/bootstrap_cloud_pahf.sh --with-memory-deps
```

Native VitaBench bridge environment:

```bash
bash scripts/bootstrap_vitabench_pahf.sh
```

## Fill API

Edit:

```bash
.env.pahf
```

Fill:

- `PAHF_OPENAI_API_KEY`
- `PAHF_OPENAI_BASE_URL` if needed
- optionally `PAHF_AGENT_MODEL` / `PAHF_HUMAN_MODEL`
- optionally the `PAHF_VITA_*` variables for the bridge run

## Preflight

```bash
.venv/bin/python scripts/preflight_pahf.py
```

Expected:

- `core_packages`: true
- `native_packages`: true
- `memory_packages`: only required for full memory mode
- `required_env.PAHF_OPENAI_API_KEY`: true

## Native VitaBench 28-Task Run

```bash
bash scripts/run_vitabench_pahf_28.sh
```

Optional smoke run before the full benchmark:

```bash
PAHF_VITA_LIMIT_TASKS=1 PAHF_VITA_NUM_TRIALS=1 PAHF_VITA_MAX_STEPS=20 bash scripts/run_vitabench_pahf_28.sh
```

## Native Smoke Run

```bash
bash scripts/run_pahf_native.sh shopping --no-memory
```

## Dependency Meaning

- `requirements.native.txt`
  - `openai`, `numpy`, `tqdm`, `faiss-cpu`
  - enough for native PAHF import and no-memory baseline execution
- `requirements.memory.txt`
  - `torch`, `transformers`
  - needed for PAHF's local DragonPlus embedding memory, including the VitaBench bridge

`torch` is not for API calls. It is only for local embedding inference.

CPU-only servers can still run the bridge, but the first run will be slower because it needs to download and initialize the DragonPlus encoders.

## VitaBench Outputs

Each bridge run writes a timestamped folder under:

- `runs/vitabench_pahf/`

with:

- `official_results.json`
- `official_metrics.json`
- `subtask_records.jsonl`
- `subtask_records.csv`
- `skill_tag_metrics.json`
- `run_manifest.json`
- `SUMMARY.md`
