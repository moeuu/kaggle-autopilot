from __future__ import annotations

from pathlib import Path

import pytest

from kagglebot.agents.identity import (
    resolve_oracle_implementation_agent,
    resolve_oracle_model,
    resolve_primary_agent,
    resolve_repository_implementation_agent,
)


def test_resolve_primary_agent_reads_project_configuration(tmp_path: Path) -> None:
    config_path = tmp_path / "pyproject.toml"
    config_path.write_text(
        """
[tool.kagglebot.agent]
model = "configured-model"
reasoning_effort = "high"
oracle_model = "configured-pro"
""".strip(),
        encoding="utf-8",
    )

    agent = resolve_primary_agent(config_path=config_path, environ={})

    assert agent.model == "configured-model"
    assert agent.reasoning_effort == "high"
    assert resolve_oracle_model(config_path=config_path, environ={}) == "configured-pro"


def test_resolve_primary_agent_environment_overrides_project_configuration(tmp_path: Path) -> None:
    config_path = tmp_path / "pyproject.toml"
    config_path.write_text(
        """
[tool.kagglebot.agent]
model = "configured-model"
reasoning_effort = "high"
""".strip(),
        encoding="utf-8",
    )

    agent = resolve_primary_agent(
        config_path=config_path,
        environ={
            "KAGGLEBOT_PRIMARY_MODEL": "environment-model",
            "KAGGLEBOT_PRIMARY_REASONING_EFFORT": "xhigh",
        },
    )

    assert agent.model == "environment-model"
    assert agent.reasoning_effort == "xhigh"


def test_resolve_primary_agent_rejects_invalid_project_configuration(tmp_path: Path) -> None:
    config_path = tmp_path / "pyproject.toml"
    config_path.write_text(
        """
[tool.kagglebot.agent]
model = 56
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="tool.kagglebot.agent.model must be a non-empty string"):
        resolve_primary_agent(config_path=config_path, environ={})


def test_resolve_oracle_model_environment_overrides_project_configuration(tmp_path: Path) -> None:
    config_path = tmp_path / "pyproject.toml"
    config_path.write_text(
        """
[tool.kagglebot.agent]
oracle_model = "configured-pro"
""".strip(),
        encoding="utf-8",
    )

    model = resolve_oracle_model(
        config_path=config_path,
        environ={"KAGGLEBOT_ORACLE_MODEL": "fixed-pro"},
    )

    assert model == "fixed-pro"


def test_resolve_repository_implementation_agent_uses_sol_xhigh_profile(tmp_path: Path) -> None:
    config_path = tmp_path / "pyproject.toml"
    config_path.write_text(
        """
[tool.kagglebot.agent]
repository_implementation_model = "gpt-5.6-sol"
repository_implementation_reasoning_effort = "xhigh"
repository_implementation_profile = "sol-xhigh"
repository_implementation_reasoning_profile = "xhigh"
""".strip(),
        encoding="utf-8",
    )

    agent = resolve_repository_implementation_agent(config_path=config_path, environ={})

    assert agent.model == "gpt-5.6-sol"
    assert agent.reasoning_effort == "xhigh"
    assert agent.cli_profile == "sol-xhigh"
    assert agent.reasoning_profile == "xhigh"


def test_resolve_oracle_implementation_agent_centralizes_all_oracle_followup_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "pyproject.toml"
    config_path.write_text(
        """
[tool.kagglebot.agent]
oracle_implementation_model = "gpt-5.6-sol"
oracle_implementation_reasoning_effort = "xhigh"
oracle_implementation_profile = "sol-xhigh"
oracle_implementation_reasoning_profile = "xhigh"
""".strip(),
        encoding="utf-8",
    )

    agent = resolve_oracle_implementation_agent(config_path=config_path, environ={})
    repository_agent = resolve_repository_implementation_agent(config_path=config_path, environ={})

    assert agent.model == "gpt-5.6-sol"
    assert agent.reasoning_effort == "xhigh"
    assert agent.cli_profile == "sol-xhigh"
    assert agent.reasoning_profile == "xhigh"
    assert repository_agent == agent
