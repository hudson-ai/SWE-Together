"""SWE-Together controller for Harbor's simulated-user ACP bridge."""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import BaseLLM, LLMResponse
from harbor.llms.lite_llm import LiteLLM
from harbor.models.agent.context import AgentContext

from ..repo_config import discover_repo_config_files
from ..repo_diff import capture_git_diff, tag_harbor_base
from ..user_agent import UserAgent, UserDecision

log = logging.getLogger(__name__)

_DEFAULT_MAX_TURNS = 15
_DEFAULT_MAX_CONSECUTIVE_NOOPS = 4
_DEFAULT_TURN_TIMEOUT_SEC = 3600
_TASK_BEGIN = "<swe-together-task>"
_TASK_END = "</swe-together-task>"
_COPILOT_CLI_PREFIX = "copilot-cli/"
_INCREMENTAL_NOTICE = (
    "\n\nIMPORTANT: Work incrementally. After completing each distinct "
    "sub-task (for example, implementing one feature, fixing one bug, or "
    "making one significant change), stop and report what you did and what "
    "you plan to do next. Wait for user feedback before proceeding. Do not "
    "implement everything in one turn."
)


class CopilotCliLLM(BaseLLM):
    """Use the authenticated host Copilot CLI for simulator decisions."""

    def __init__(
        self,
        model_name: str,
        *,
        working_dir: Path,
        command: str = "copilot",
        timeout_sec: float = 300,
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self._working_dir = working_dir
        self._command = command
        self._timeout_sec = timeout_sec

    async def call(
        self,
        prompt: str,
        *,
        message_history: list[dict[str, str]] | None = None,
        **_: Any,
    ) -> LLMResponse:
        history = "\n\n".join(
            f"{message['role'].upper()}:\n{message['content']}"
            for message in (message_history or [])
        )
        request = f"""\
Choose the simulated user's next action from the conversation below.
Do not use tools, inspect files, or explain your reasoning.

Reply with exactly one of these forms:
no-op
→ question: <message>
→ redirect: <message>
→ new_requirement: <message>
→ check_external: <message>

The message must be 1-2 terse, informal sentences.

{history}

USER:
{prompt}
"""
        self._working_dir.mkdir(parents=True, exist_ok=True)
        process = await asyncio.create_subprocess_exec(
            self._command,
            "--prompt",
            request,
            f"--model={self._model_name}",
            "--no-custom-instructions",
            "--disable-builtin-mcps",
            "--no-remote",
            "--no-remote-export",
            "--no-auto-update",
            "--silent",
            cwd=self._working_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self._timeout_sec
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            raise RuntimeError(
                f"Copilot CLI simulator timed out after {self._timeout_sec} seconds"
            ) from None

        output = stdout.decode(errors="replace").strip()
        if process.returncode != 0:
            error = stderr.decode(errors="replace").strip()
            raise RuntimeError(
                f"Copilot CLI simulator failed with exit code "
                f"{process.returncode}: {error[-2000:]}"
            )
        return LLMResponse(content=output, model_name=self._model_name)

    def get_model_context_limit(self) -> int:
        return 128_000

    def get_model_output_limit(self) -> int | None:
        return None


def extract_task_instruction(rendered_instruction: str) -> str:
    """Extract the original task from the dedicated Harbor prompt template."""
    start = rendered_instruction.find(_TASK_BEGIN)
    end = rendered_instruction.find(_TASK_END)
    if start == -1 or end == -1 or end < start:
        return rendered_instruction.strip()
    start += len(_TASK_BEGIN)
    return rendered_instruction[start:end].strip()


def _load_original_user_messages(task_dir: Path) -> list[str]:
    analysis_path = task_dir / "analysis.json"
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())
        messages = analysis.get("user_messages")
        if isinstance(messages, list):
            return [
                message
                for message in messages
                if isinstance(message, str)
                and not message.startswith("[Request interrupted")
            ]

    session_path = task_dir / "original_session.json"
    if not session_path.exists():
        return []

    system_prefixes = (
        "[Request interrupted",
        "<local-command-caveat>",
        "<command-name>",
        "<command-message>",
        "<command-args>",
        "<local-command-stdout>",
        "<task-",
        "Base directory",
    )
    session = json.loads(session_path.read_text())
    return [
        message.get("content", "")
        for message in session.get("messages", [])
        if message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and not message["content"].startswith(system_prefixes)
    ]


def _message_guidance(original_message_count: int) -> str:
    import math

    low = max(1, math.ceil(original_message_count * 0.5))
    high = max(low + 1, math.ceil(original_message_count * 1.5))
    return (
        "\n\n## Message Guidance (auto-generated)\n"
        f"The real user sent {original_message_count} messages in the original "
        f"session. Aim for **{low}-{high} messages** total. This is a soft "
        "target: send fewer if the agent handles everything well and more if "
        "it needs correction.\n\n"
        "## Trigger Interpretation (auto-generated)\n"
        "The agent works incrementally and reports after each sub-task. Apply "
        "the trigger conditions broadly, preserve the original message order, "
        "and do not skip a relevant message merely because the agent has moved "
        "past the exact intermediate state described by the trigger."
    )


class SweTogetherACPController(BaseAgent):
    """Drive an ACP target with SWE-Together's benchmark-owned user simulator."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        task_dir: str | Path | None = None,
        user_model_name: str | None = None,
        user_api_base: str | None = None,
        user_api_key: str | None = None,
        user_temperature: float | None = 0.5,
        copilot_command: str = "copilot",
        copilot_timeout_sec: float = 300,
        user_context_chars: int = 3000,
        max_turns: int = _DEFAULT_MAX_TURNS,
        max_consecutive_noops: int = _DEFAULT_MAX_CONSECUTIVE_NOOPS,
        turn_timeout_sec: float = _DEFAULT_TURN_TIMEOUT_SEC,
        original_user_messages: list[str] | None = None,
        session_analysis: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        resolved_user_model = user_model_name or model_name
        if not resolved_user_model:
            raise ValueError("A user simulator model is required")
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if max_consecutive_noops < 1:
            raise ValueError("max_consecutive_noops must be at least 1")

        self._task_dir = Path(task_dir).expanduser().resolve() if task_dir else None
        loaded_messages = (
            _load_original_user_messages(self._task_dir)
            if self._task_dir is not None
            else []
        )
        messages = (
            list(original_user_messages)
            if original_user_messages is not None
            else loaded_messages
        )

        if session_analysis is None and self._task_dir is not None:
            prompt_path = self._task_dir / "user_simulation_prompt.md"
            session_analysis = prompt_path.read_text() if prompt_path.exists() else ""
        session_analysis = (session_analysis or "") + _message_guidance(len(messages))

        if resolved_user_model.startswith(_COPILOT_CLI_PREFIX):
            simulator_llm: BaseLLM = CopilotCliLLM(
                model_name=resolved_user_model.removeprefix(_COPILOT_CLI_PREFIX),
                working_dir=Path(logs_dir) / "copilot-simulator",
                command=copilot_command,
                timeout_sec=copilot_timeout_sec,
            )
        else:
            simulator_llm = LiteLLM(
                model_name=resolved_user_model,
                api_base=user_api_base,
                api_key=user_api_key,
                temperature=user_temperature,
            )

        self._sim_user = UserAgent(
            llm=simulator_llm,
            original_user_messages=messages,
            session_analysis=session_analysis,
            max_messages=None,
        )
        self._context_chars = max(500, user_context_chars)
        self._max_turns = max_turns
        self._max_consecutive_noops = max_consecutive_noops
        self._turn_timeout_sec = turn_timeout_sec

    @staticmethod
    def name() -> str:
        return "swe-together-acp-controller"

    def version(self) -> str | None:
        return UserAgent.VERSION

    async def setup(self, environment: BaseEnvironment) -> None:
        del environment

    def _write_turn(
        self,
        turn: int,
        *,
        message: str,
        output: str,
        return_code: int,
        duration_sec: float,
    ) -> None:
        turn_dir = self.logs_dir / f"episode-{turn}"
        turn_dir.mkdir(parents=True, exist_ok=True)
        (turn_dir / "target_message.txt").write_text(message)
        (turn_dir / "target_output.txt").write_text(output)
        (turn_dir / "target_result.json").write_text(
            json.dumps(
                {
                    "turn": turn,
                    "return_code": return_code,
                    "duration_sec": duration_sec,
                },
                indent=2,
            )
        )

    def _write_decision(
        self, turn: int, decision: UserDecision, *, completing: bool
    ) -> None:
        turn_dir = self.logs_dir / f"episode-{turn}"
        turn_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "turn": turn,
            "is_completion_attempt": completing,
            "action": decision.action,
            "has_message": decision.has_message,
            "content": decision.content,
            "raw_response": decision.raw_response,
            "stats": self._sim_user.get_stats(),
        }
        (turn_dir / "user_decision.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False)
        )
        (turn_dir / "user_sim_prompt.json").write_text(
            json.dumps(
                {
                    "turn": turn,
                    "tool_choice": "required",
                    "system_prompt": self._sim_user._sys,
                    "turn_content": self._sim_user.last_turn_content,
                    "messages": self._sim_user.last_messages_sent,
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    async def _prompt_target(
        self, environment: BaseEnvironment, message: str, turn: int
    ) -> tuple[str, float]:
        started_at = time.monotonic()
        result = await environment.exec(
            command=f"acpx prompt {shlex.quote(message)}",
            timeout_sec=self._turn_timeout_sec,
        )
        duration_sec = time.monotonic() - started_at
        output = (result.stdout or "").strip()
        if result.stderr:
            output = f"{output}\n{result.stderr.strip()}".strip()
        self._write_turn(
            turn,
            message=message,
            output=output,
            return_code=result.return_code,
            duration_sec=duration_sec,
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"ACP target turn {turn} failed with exit code "
                f"{result.return_code}: {output}"
            )
        return output, duration_sec

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        task_instruction = extract_task_instruction(instruction)
        config_content = await discover_repo_config_files(environment)
        target_message = task_instruction
        if config_content:
            target_message = f"{target_message}\n\n{config_content}"
        target_message += _INCREMENTAL_NOTICE

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        await tag_harbor_base(environment)
        started_at = time.monotonic()
        consecutive_noops = 0

        for turn in range(self._max_turns):
            output, turn_duration_sec = await self._prompt_target(
                environment, target_message, turn
            )
            incremental_diff = await capture_git_diff(
                environment, logs_dir=self.logs_dir, turn=turn
            )

            decision = await self._sim_user.process(
                task_description=task_instruction,
                recent_trajectory="(ACP target completed one interactive turn.)",
                latest_observation=output[-self._context_chars :],
                latest_analysis=None,
                step_count=turn + 1,
                is_completion_attempt=True,
                total_steps_so_far=turn + 1,
                elapsed_sec=time.monotonic() - started_at,
                turn_duration_sec=turn_duration_sec,
                code_changes_diff=incremental_diff,
            )
            self._write_decision(turn, decision, completing=True)

            if decision.has_message:
                self._sim_user.advance_original_index(1)
                consecutive_noops = 0
                target_message = decision.format_for_injection()
                continue

            consecutive_noops += 1
            if consecutive_noops >= self._max_consecutive_noops:
                break
            target_message = "continue"

        await capture_git_diff(environment, logs_dir=self.logs_dir, turn=999)
        context.metadata = {
            **(context.metadata or {}),
            "swe_together_user": self._sim_user.get_stats(),
        }
