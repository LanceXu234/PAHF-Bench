# PAHF -> VitaBench Native Bridge

This repository now contains a native VitaBench bridge for PAHF.

The bridge keeps VitaBench's own:

- multi-subtask personalization workflow
- official evaluator
- official metrics
- raw results format

and swaps in a PAHF-style memory backend:

- PAHF retrieval bank (`sql` or `faiss`)
- PAHF-style preference extraction from interaction history
- PAHF-style clarification trigger and write-back loop

## Simplest Server Flow

```bash
git clone https://github.com/LanceXu234/PAHF-Bench.git
cd PAHF-Bench
bash scripts/bootstrap_vitabench_pahf.sh
bash scripts/run_vitabench_pahf_28.sh
```

Before the final run, edit `.env.pahf` once and fill:

- `PAHF_OPENAI_API_KEY`
- `PAHF_OPENAI_BASE_URL`

## What Bootstrap Does

`scripts/bootstrap_vitabench_pahf.sh` will:

1. create / reuse `PAHF-Bench/.venv`
2. install PAHF native + memory dependencies
3. clone official VitaBench 2.0 into `external/vitabench-2.0`
4. checkout the pinned VitaBench commit `0c481ae1f97b973fe0fa0d30e78ab463c4d59193`
5. install VitaBench into the same virtualenv
6. generate `external/vitabench-2.0/src/vita/models.yaml` from `.env.pahf`

## Configuration

Only `.env.pahf` needs to be edited.

Required:

- `PAHF_OPENAI_API_KEY`
- `PAHF_OPENAI_BASE_URL`

Optional VitaBench bridge knobs:

- `PAHF_VITA_AGENT_LLM`
- `PAHF_VITA_USER_LLM`
- `PAHF_VITA_EVALUATOR_LLM`
- `PAHF_VITA_NUM_TRIALS`
- `PAHF_VITA_MAX_STEPS`
- `PAHF_VITA_MAX_CONCURRENCY`
- `PAHF_VITA_MEMORY_BACKEND`

## Default 28-task Run

The default run uses the fixed task slice already stored in this repository:

- `data/vitabench/dynamic_delivery_28.tasks.json`

So you do **not** need to separately download the full personalization dataset just to run this 28-task benchmark.

## Smoke Run

Use this before the full benchmark if you want a cheap end-to-end check:

```bash
PAHF_VITA_LIMIT_TASKS=1 PAHF_VITA_NUM_TRIALS=1 PAHF_VITA_MAX_STEPS=20 bash scripts/run_vitabench_pahf_28.sh
```

## Runtime Notes

- The bridge uses PAHF's DragonPlus embedding memory, so `torch` and `transformers` are required.
- A GPU is not mandatory. CPU-only servers can run it too, just slower.
- The first bridge run may spend extra time downloading the DragonPlus encoders from Hugging Face.

## Outputs

Each run writes a timestamped directory under:

- `runs/vitabench_pahf/`

with:

- `official_results.json`
- `official_metrics.json`
- `subtask_records.jsonl`
- `subtask_records.csv`
- `skill_tag_metrics.json`
- `run_manifest.json`
- `SUMMARY.md`

This preserves both official benchmark metrics and the highest-granularity subtask traces needed for later analysis.
