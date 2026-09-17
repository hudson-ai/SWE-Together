from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from harbor.models.agent.context import AgentContext
from user_agent.agents.harbor_acp_controller import (
    CopilotCliLLM,
    SweTogetherACPController,
    extract_task_instruction,
)
from user_agent.user_agent import UserDecision


def test_extract_task_instruction_from_harbor_template() -> None:
    rendered = """\
<swe-together-task>
Fix the scheduler.
</swe-together-task>

Run acpx prompt to talk to the target.
"""

    assert extract_task_instruction(rendered) == "Fix the scheduler."


def test_controller_can_use_local_copilot_simulator(tmp_path: Path) -> None:
    controller = SweTogetherACPController(
        logs_dir=tmp_path / "logs",
        model_name="copilot-cli/gpt-5.4",
        original_user_messages=[],
        session_analysis="",
    )

    assert isinstance(controller._sim_user._llm, CopilotCliLLM)


@pytest.mark.asyncio
async def test_controller_drives_one_persistent_acp_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = SweTogetherACPController(
        logs_dir=tmp_path / "logs",
        model_name="test/user-model",
        original_user_messages=["please add a regression test"],
        session_analysis="Send the follow-up after the first implementation.",
        max_turns=3,
        max_consecutive_noops=2,
    )
    controller._sim_user.process = AsyncMock(
        side_effect=[
            UserDecision(
                action="new_requirement", content="please add a regression test"
            ),
            UserDecision(action="no-op"),
            UserDecision(action="no-op"),
        ]
    )

    outputs = iter(["implemented it", "added the test", "all done"])

    async def exec_command(**kwargs):
        assert kwargs["command"].startswith("acpx prompt ")
        return SimpleNamespace(
            stdout=next(outputs),
            stderr="",
            return_code=0,
        )

    environment = SimpleNamespace(exec=AsyncMock(side_effect=exec_command))
    monkeypatch.setattr(
        "user_agent.agents.harbor_acp_controller.discover_repo_config_files",
        AsyncMock(return_value=""),
    )
    monkeypatch.setattr(
        "user_agent.agents.harbor_acp_controller.tag_harbor_base",
        AsyncMock(),
    )
    monkeypatch.setattr(
        "user_agent.agents.harbor_acp_controller.capture_git_diff",
        AsyncMock(return_value=""),
    )

    context = AgentContext()
    await controller.run(
        instruction=(
            "<swe-together-task>\nFix the scheduler.\n</swe-together-task>\n"
            "Use acpx prompt."
        ),
        environment=environment,
        context=context,
    )

    commands = [call.kwargs["command"] for call in environment.exec.await_args_list]
    assert len(commands) == 3
    assert "Fix the scheduler." in commands[0]
    assert "please add a regression test" in commands[1]
    assert commands[2] == "acpx prompt continue"
    assert controller._sim_user.process.await_count == 3
    assert "swe_together_user" in context.metadata
    assert (tmp_path / "logs" / "episode-0" / "target_output.txt").read_text() == (
        "implemented it"
    )
