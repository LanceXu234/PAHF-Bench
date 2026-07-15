from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "vitabench_28task_manifest.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "vitabench"


@dataclass
class PreparedSubset:
    benchmark: str
    subset_name: str
    source_meta: str
    source_tasks: str
    task_ids: list[str]
    num_tasks: int
    num_subtasks: int
    prepared_at_utc: str
    tasks_path: str


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    manifest = load_json(DEFAULT_MANIFEST)
    source_meta = Path(manifest["source_meta"])
    if not source_meta.exists():
        raise FileNotFoundError(f"Source meta file not found: {source_meta}")

    meta = load_json(source_meta)
    source_tasks = Path(meta["source"])
    if not source_tasks.exists():
        raise FileNotFoundError(f"Source tasks file not found: {source_tasks}")

    all_tasks = load_json(source_tasks)
    task_ids = list(manifest["task_ids"])
    selected = [task for task in all_tasks if task.get("id") in task_ids]
    selected_ids = {task["id"] for task in selected}
    missing = [task_id for task_id in task_ids if task_id not in selected_ids]
    if missing:
        raise ValueError(f"Missing task ids in source tasks: {missing}")

    selected.sort(key=lambda task: task_ids.index(task["id"]))

    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    subset_name = manifest["task_subset_name"]
    tasks_path = output_dir / f"{subset_name}.tasks.json"
    summary_path = output_dir / f"{subset_name}.summary.json"

    tasks_path.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    prepared = PreparedSubset(
        benchmark=manifest["benchmark"],
        subset_name=subset_name,
        source_meta=str(source_meta),
        source_tasks=str(source_tasks),
        task_ids=task_ids,
        num_tasks=len(selected),
        num_subtasks=sum(len(task.get("subtasks", [])) for task in selected),
        prepared_at_utc=datetime.now(timezone.utc).isoformat(),
        tasks_path=str(tasks_path),
    )
    summary_path.write_text(
        json.dumps(asdict(prepared), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(asdict(prepared), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
