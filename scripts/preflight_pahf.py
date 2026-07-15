from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
ENV_FILE = ROOT / ".env.pahf"
MANIFEST = ROOT / "configs" / "vitabench_28task_manifest.json"
PREPARED_SUBSET = ROOT / "data" / "vitabench" / "dynamic_delivery_28.tasks.json"


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def pkg_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    env_values = parse_env_file(ENV_FILE)
    merged = {**env_values, **os.environ}

    required_env = ["PAHF_OPENAI_API_KEY"]
    optional_env = [
        "PAHF_OPENAI_BASE_URL",
        "PAHF_AGENT_MODEL",
        "PAHF_HUMAN_MODEL",
        "PAHF_OPENAI_TIMEOUT",
        "PAHF_OPENAI_MAX_RETRIES",
    ]
    core_packages = ["openai", "numpy", "tqdm"]
    native_packages = ["faiss"]
    memory_packages = ["torch", "transformers"]

    report = {
        "root": str(ROOT),
        "python": sys.executable,
        "env_file_exists": ENV_FILE.exists(),
        "required_env": {key: bool(merged.get(key)) for key in required_env},
        "optional_env": {key: bool(merged.get(key)) for key in optional_env},
        "core_packages": {name: pkg_available(name) for name in core_packages},
        "native_packages": {name: pkg_available(name) for name in native_packages},
        "memory_packages": {name: pkg_available(name) for name in memory_packages},
        "manifest_exists": MANIFEST.exists(),
        "manifest_path": str(MANIFEST),
        "prepared_subset_exists": PREPARED_SUBSET.exists(),
        "prepared_subset_path": str(PREPARED_SUBSET),
        "benchmark_root_exists": (WORKSPACE / "benchmark" / "vitabench-2.0").exists(),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    failures: list[str] = []
    if not all(report["required_env"].values()):
        failures.append("missing_api")
    if not all(report["core_packages"].values()):
        failures.append("missing_core_packages")
    if not all(report["native_packages"].values()):
        failures.append("missing_native_packages")
    if not report["manifest_exists"]:
        failures.append("missing_manifest")

    if failures:
        print(f"PRECHECK_STATUS=failed:{','.join(failures)}")
        return 1

    print("PRECHECK_STATUS=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
