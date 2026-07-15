from __future__ import annotations

import argparse
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def collect_model_names(env_values: dict[str, str]) -> list[str]:
    raw_names = [
        env_values.get("PAHF_VITA_AGENT_LLM", "").strip(),
        env_values.get("PAHF_VITA_USER_LLM", "").strip(),
        env_values.get("PAHF_VITA_EVALUATOR_LLM", "").strip(),
        env_values.get("PAHF_AGENT_MODEL", "gpt-4o-mini").strip(),
        env_values.get("PAHF_HUMAN_MODEL", "gpt-4o-mini").strip(),
        "gpt-4o-mini",
        "gpt-4.1",
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for name in raw_names:
        if not name or name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate VitaBench models.yaml from .env.pahf")
    parser.add_argument("--vitabench-root", required=True, help="Path to VitaBench-2.0 repository root")
    args = parser.parse_args()

    env_values = parse_env_file(REPO_ROOT / ".env.pahf")
    base_url = env_values.get("PAHF_OPENAI_BASE_URL", "").strip()
    api_key = env_values.get("PAHF_OPENAI_API_KEY", "").strip()
    timeout = env_values.get("PAHF_OPENAI_TIMEOUT", "180").strip() or "180"
    max_retries = env_values.get("PAHF_OPENAI_MAX_RETRIES", "5").strip() or "5"
    model_names = collect_model_names(env_values)

    vitabench_root = Path(args.vitabench_root).resolve()
    target = vitabench_root / "src" / "vita" / "models.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "default": {
            "base_url": base_url,
            "api_key": api_key,
            "temperature": 0.0,
            "max_input_tokens": 128000,
            "request_timeout": int(timeout),
            "max_retries": int(max_retries),
        },
        "models": [],
    }
    for name in model_names:
        payload["models"].append(
            {
                "name": name,
                "max_tokens": 8192 if "gpt-4.1" in name else 4096,
                "cost_1m_token_dollar": {
                    "prompt_price": 0.0,
                    "completion_price": 0.0,
                },
            }
        )

    target.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
