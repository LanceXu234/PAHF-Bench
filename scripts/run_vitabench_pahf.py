from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_FILE = REPO_ROOT / "data" / "vitabench" / "dynamic_delivery_28.tasks.json"
DEFAULT_RUN_ROOT = REPO_ROOT / "runs" / "vitabench_pahf"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _ensure_vitabench_imports(vitabench_root: Path) -> None:
    src_root = vitabench_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _load_tasks(task_file: Path):
    from vita.data_model.personalization_task import PersonalizationTask

    raw = json.loads(task_file.read_text(encoding="utf-8"))
    return [PersonalizationTask.model_validate(item) for item in raw]


def _make_run_dir(base_root: Path, run_name: str | None) -> Path:
    if run_name:
        run_dir = base_root / run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base_root / f"pahf_vitabench_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _metrics_to_dict(metrics: Any) -> Dict[str, Any]:
    data = metrics.as_dict()
    data["avg_reward"] = metrics.avg_reward
    data["avg_agent_cost"] = metrics.avg_agent_cost
    if metrics.pass_hat_ks:
        data["pass_hat_ks"] = metrics.pass_hat_ks
    if metrics.pass_at_n:
        data["pass_at_n"] = metrics.pass_at_n
    if metrics.average_at_n:
        data["average_at_n"] = metrics.average_at_n
    if metrics.avg_reward_breakdown:
        data["avg_reward_breakdown"] = metrics.avg_reward_breakdown
    if metrics.skill_split_metrics:
        data["skill_split_metrics"] = metrics.skill_split_metrics
    if metrics.subtask_pass_hat_ks:
        data["subtask_pass_hat_ks"] = metrics.subtask_pass_hat_ks
    if metrics.subtask_pass_at_n:
        data["subtask_pass_at_n"] = metrics.subtask_pass_at_n
    if metrics.subtask_average_at_n:
        data["subtask_average_at_n"] = metrics.subtask_average_at_n
    if metrics.subtask_num_units is not None:
        data["subtask_num_units"] = metrics.subtask_num_units
    return data


def _export_subtask_records(tasks, results, run_dir: Path) -> Dict[str, Dict[str, Any]]:
    task_lookup = {task.id: task for task in tasks}
    rows: List[Dict[str, Any]] = []
    skill_buckets: Dict[str, List[float]] = defaultdict(list)

    for sim in results.simulations:
        task = task_lookup.get(sim.task_id)
        if task is None:
            continue
        reward_info = sim.reward_info.info if sim.reward_info and sim.reward_info.info else {}
        subtask_rewards = reward_info.get("subtask_rewards", {}) or {}

        for idx, subtask in enumerate(task.subtasks):
            reward = subtask_rewards.get(f"subtask_{idx}_reward")
            skill_tags = list(subtask.skill_tested or [])
            if not skill_tags:
                skill_tags = ["personalize"]
            row = {
                "simulation_id": sim.id,
                "task_id": task.id,
                "trial": sim.trial,
                "subtask_idx": idx,
                "subtask_id": subtask.subtask_id,
                "domain": subtask.domain,
                "instruction": subtask.instruction,
                "reward": reward,
                "success": reward == 1.0,
                "skill_tags": "|".join(skill_tags),
                "termination_reason": str(sim.termination_reason),
                "agent_cost": sim.agent_cost,
                "user_cost": sim.user_cost,
            }
            rows.append(row)
            for tag in skill_tags:
                if reward is not None:
                    skill_buckets[tag].append(float(reward))

    jsonl_path = run_dir / "subtask_records.jsonl"
    csv_path = run_dir / "subtask_records.csv"
    with jsonl_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "simulation_id",
                "task_id",
                "trial",
                "subtask_idx",
                "subtask_id",
                "domain",
                "instruction",
                "reward",
                "success",
                "skill_tags",
                "termination_reason",
                "agent_cost",
                "user_cost",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    skill_metrics = {}
    for tag, rewards in skill_buckets.items():
        if not rewards:
            continue
        skill_metrics[tag] = {
            "count": len(rewards),
            "avg_reward": sum(rewards) / len(rewards),
            "success_rate": sum(1.0 for value in rewards if value == 1.0) / len(rewards),
        }
    _save_json(run_dir / "skill_tag_metrics.json", skill_metrics)
    return skill_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PAHF natively on VitaBench personalization tasks.")
    parser.add_argument("--vitabench-root", type=str, required=True, help="Path to VitaBench-2.0 repository root.")
    parser.add_argument("--task-file", type=str, default=str(DEFAULT_TASK_FILE), help="Path to the local personalization task JSON file.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional fixed run directory name.")
    parser.add_argument("--run-root", type=str, default=str(DEFAULT_RUN_ROOT), help="Directory where run outputs will be written.")
    parser.add_argument("--limit-tasks", type=int, default=None, help="Optional smoke-test limit on number of user tasks.")
    parser.add_argument("--num-trials", type=int, default=4, help="Number of trials per task. Use 4 for official Avg@4 / Pass@4 metrics.")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per subtask conversation.")
    parser.add_argument("--max-errors", type=int, default=10, help="Max consecutive tool errors.")
    parser.add_argument("--max-concurrency", type=int, default=2, help="Parallel task concurrency.")
    parser.add_argument("--seed", type=int, default=300, help="Base random seed.")
    parser.add_argument("--log-level", type=str, default="INFO", help="VitaBench log level.")
    parser.add_argument("--language", type=str, default="chinese", choices=["chinese", "english"], help="Benchmark language.")
    parser.add_argument("--evaluation-type", type=str, default="trajectory", help="VitaBench evaluation type.")
    parser.add_argument("--agent-llm", type=str, default=None, help="Agent LLM name from VitaBench models.yaml.")
    parser.add_argument("--user-llm", type=str, default=None, help="User-simulator LLM name from VitaBench models.yaml.")
    parser.add_argument("--evaluator-llm", type=str, default=None, help="Evaluator LLM name from VitaBench models.yaml.")
    parser.add_argument("--memory-bank-type", type=str, default="sql", choices=["sql", "faiss"], help="PAHF memory bank backend.")
    parser.add_argument("--disable-llm-extraction", action="store_true", help="Disable LLM extraction inside PAHFMemory and use heuristic extraction only.")
    parser.add_argument("--disable-llm-questions", action="store_true", help="Disable LLM-generated clarification questions and use heuristic templates only.")
    parser.add_argument("--enable-outcome-reward", action="store_true", help="Enable VitaBench outcome reward combination.")
    return parser


def main() -> int:
    _load_dotenv(REPO_ROOT / ".env.pahf")
    args = build_parser().parse_args()

    vitabench_root = Path(args.vitabench_root).resolve()
    task_file = Path(args.task_file).resolve()
    run_root = Path(args.run_root).resolve()
    _ensure_vitabench_imports(vitabench_root)

    from vita.config import DEFAULT_LLM_AGENT, DEFAULT_LLM_EVALUATOR, DEFAULT_LLM_USER, models
    from vita.metrics.agent_metrics import compute_metrics
    from vita.run import run_tasks

    tasks = _load_tasks(task_file)
    if args.limit_tasks is not None:
        tasks = tasks[: args.limit_tasks]

    agent_llm = args.agent_llm or os.environ.get("PAHF_VITA_AGENT_LLM") or os.environ.get("PAHF_AGENT_MODEL") or DEFAULT_LLM_AGENT
    user_llm = args.user_llm or os.environ.get("PAHF_VITA_USER_LLM") or os.environ.get("PAHF_HUMAN_MODEL") or DEFAULT_LLM_USER
    evaluator_llm = args.evaluator_llm or os.environ.get("PAHF_VITA_EVALUATOR_LLM") or agent_llm or DEFAULT_LLM_EVALUATOR

    run_dir = _make_run_dir(run_root, args.run_name)
    raw_results_path = run_dir / "official_results.json"

    run_manifest = {
        "vitabench_root": str(vitabench_root),
        "task_file": str(task_file),
        "num_tasks": len(tasks),
        "num_trials": args.num_trials,
        "max_steps": args.max_steps,
        "max_errors": args.max_errors,
        "max_concurrency": args.max_concurrency,
        "seed": args.seed,
        "agent_llm": agent_llm,
        "user_llm": user_llm,
        "evaluator_llm": evaluator_llm,
        "memory_class": "pahf_bench.vitabench_bridge.PAHFMemory",
        "memory_bank_type": args.memory_bank_type,
        "enable_llm_extraction": not args.disable_llm_extraction,
        "enable_llm_questions": not args.disable_llm_questions,
        "enable_outcome_reward": args.enable_outcome_reward,
        "language": args.language,
        "evaluation_type": args.evaluation_type,
        "output_dir": str(run_dir),
    }
    _save_json(run_dir / "run_manifest.json", run_manifest)

    memory_llm_kwargs = {
        "bank_type": args.memory_bank_type,
        "enable_llm_extraction": not args.disable_llm_extraction,
        "enable_llm_questions": not args.disable_llm_questions,
    }

    results = run_tasks(
        domain="personalization",
        tasks=tasks,
        agent="personalization_agent",
        user="personalization_user",
        llm_agent=agent_llm,
        llm_args_agent=dict(models.get(agent_llm, models.get("default", {}))),
        llm_user=user_llm,
        llm_args_user=dict(models.get(user_llm, models.get("default", {}))),
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        save_to=raw_results_path,
        console_display=True,
        evaluation_type=args.evaluation_type,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        log_level=args.log_level,
        enable_think=False,
        llm_evaluator=evaluator_llm,
        llm_args_evaluator=dict(models.get(evaluator_llm, models.get("default", {}))),
        language=args.language,
        memory_class="pahf_bench.vitabench_bridge.PAHFMemory",
        enable_outcome_reward=args.enable_outcome_reward,
        memory_type="rewrite",
        **memory_llm_kwargs,
    )

    metrics = compute_metrics(results)
    metrics_payload = _metrics_to_dict(metrics)
    _save_json(run_dir / "official_metrics.json", metrics_payload)
    skill_metrics = _export_subtask_records(tasks, results, run_dir)

    summary_md = "\n".join(
        [
            "# PAHF on VitaBench",
            "",
            f"- Run dir: `{run_dir}`",
            f"- Tasks: `{len(tasks)}`",
            f"- Trials: `{args.num_trials}`",
            f"- Agent LLM: `{agent_llm}`",
            f"- User LLM: `{user_llm}`",
            f"- Evaluator LLM: `{evaluator_llm}`",
            f"- Avg reward: `{metrics.avg_reward:.4f}`",
            "",
            "## Key files",
            f"- Official results: `{raw_results_path.name}`",
            "- Official metrics: `official_metrics.json`",
            "- Subtask records: `subtask_records.jsonl` / `subtask_records.csv`",
            "- Skill tag metrics: `skill_tag_metrics.json`",
            "",
            "## Skill tags",
            *[f"- `{name}`: avg_reward={payload['avg_reward']:.4f}, success_rate={payload['success_rate']:.4f}, count={payload['count']}" for name, payload in sorted(skill_metrics.items())],
        ]
    )
    (run_dir / "SUMMARY.md").write_text(summary_md, encoding="utf-8")

    print(json.dumps({"run_dir": str(run_dir), "avg_reward": metrics.avg_reward}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
