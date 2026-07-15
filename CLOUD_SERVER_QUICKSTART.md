# PAHF Cloud Server Quickstart

This package is prepared so you can clone it on a Linux server and bootstrap it with one command.

## What You Can Run Today

- native PAHF package verification
- native PAHF no-memory baseline
- full PAHF native memory mode, if the server can support local embedding inference

## What Is Not Yet Native

This repository is **not** a native VitaBench runner.

The fixed VitaBench 28-task slice is already prepared here:

- `configs/vitabench_28task_manifest.json`
- `data/vitabench/dynamic_delivery_28.tasks.json`

But a scientifically valid VitaBench result still requires a PAHF-to-VitaBench bridge.

## One-Command Bootstrap

No-memory / baseline-ready environment:

```bash
bash scripts/bootstrap_cloud_pahf.sh
```

Full PAHF memory environment:

```bash
bash scripts/bootstrap_cloud_pahf.sh --with-memory-deps
```

## Fill API

Edit:

```bash
.env.pahf
```

Fill:

- `PAHF_OPENAI_API_KEY`
- `PAHF_OPENAI_BASE_URL` if needed

## Preflight

```bash
.venv/bin/python scripts/preflight_pahf.py
```

Expected:

- `core_packages`: true
- `native_packages`: true
- `memory_packages`: only required for full memory mode
- `required_env.PAHF_OPENAI_API_KEY`: true

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
  - only needed for PAHF's local embedding-based memory

`torch` is not for API calls. It is only for local embedding inference.
