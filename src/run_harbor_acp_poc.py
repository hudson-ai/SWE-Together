#!/usr/bin/env python3
"""Run one SWE-Together task through Harbor's simulated-user ACP bridge."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HARBOR_ROOT = Path(
    os.environ.get("HARBOR_REPO", REPO_ROOT.parent / "harbor")
).expanduser()

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HARBOR_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402
from harbor.models.bridge import BridgeConfig, BridgeKind  # noqa: E402
from harbor.models.trial.config import (  # noqa: E402
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    UserAgentConfig,
)
from harbor.trial.trial import Trial  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

CONTROLLER_IMPORT_PATH = (
    "user_agent.agents.harbor_acp_controller:SweTogetherACPController"
)
CONTROLLER_PROMPT_PATH = REPO_ROOT / "src/user_agent/harbor_acp_prompt.j2"
OPENCODE_NPX_REGISTRY_ENTRY = {
    "id": "opencode",
    "name": "OpenCode",
    "version": "1.18.30",
    "description": "OpenCode ACP server installed from its official npm package.",
    "distribution": {
        "npx": {
            "package": "opencode-ai@1.18.30",
            "args": ["acp"],
        }
    },
}
TARGET_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENROUTER_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "COPILOT_GITHUB_TOKEN",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "NPM_CONFIG_REGISTRY",
)


def _target_env() -> dict[str, str]:
    return {key: value for key in TARGET_ENV_KEYS if (value := os.environ.get(key))}


def _target_kwargs(target_agent: str) -> dict[str, object]:
    if target_agent in {"acp:github-copilot", "acp:github-copilot-cli"}:
        return {
            "auth_policy": "explicit",
            "authenticate_method_id": "copilot-login",
        }
    if target_agent.startswith("acp:"):
        return {
            "auth_policy": "disabled",
        }
    return {}


def _target_config(args: argparse.Namespace) -> AgentConfig:
    target_name = args.target_agent
    kwargs = _target_kwargs(target_name)
    if target_name == "acp:opencode":
        target_name = "acp"
        kwargs["registry_entry"] = OPENCODE_NPX_REGISTRY_ENTRY

    return AgentConfig(
        name=target_name,
        model_name=args.target_model,
        env=_target_env(),
        kwargs=kwargs,
        override_timeout_sec=args.timeout_sec,
    )


async def run(args: argparse.Namespace) -> None:
    task_dir = (REPO_ROOT / "tasks" / args.task).resolve()
    if not (task_dir / "task.toml").is_file():
        raise FileNotFoundError(f"SWE-Together task not found: {task_dir}")

    trial = await Trial.create(
        TrialConfig(
            task=TaskConfig(path=task_dir),
            trials_dir=Path(args.trials_dir),
            agent=_target_config(args),
            user_agent=UserAgentConfig(
                import_path=CONTROLLER_IMPORT_PATH,
                model_name=args.user_model,
                user_prompt_template_path=CONTROLLER_PROMPT_PATH,
                bridge=BridgeConfig(kind=BridgeKind.ACP),
                kwargs={
                    "task_dir": str(task_dir),
                    "max_turns": args.max_turns,
                    "max_consecutive_noops": args.max_consecutive_noops,
                    "turn_timeout_sec": args.turn_timeout_sec,
                },
                override_timeout_sec=args.timeout_sec,
            ),
            environment=EnvironmentConfig(
                type=args.environment,
                delete=not args.keep,
            ),
        )
    )
    result = await trial.run()
    print(result.model_dump_json(indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="Task directory name under tasks/")
    parser.add_argument(
        "--target-agent",
        default="acp:opencode",
        help="Harbor target agent, including acp:<registry-id> shorthand",
    )
    parser.add_argument("--target-model", required=True)
    parser.add_argument(
        "--user-model",
        default="openrouter/google/gemini-3.1-pro-preview",
    )
    parser.add_argument("--environment", default="docker")
    parser.add_argument("--trials-dir", default="trials/harbor-acp-poc")
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--max-consecutive-noops", type=int, default=4)
    parser.add_argument("--turn-timeout-sec", type=float, default=3600)
    parser.add_argument("--timeout-sec", type=float, default=4800)
    parser.add_argument("--keep", action="store_true")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
