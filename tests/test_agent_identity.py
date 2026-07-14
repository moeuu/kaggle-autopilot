from __future__ import annotations

from pathlib import Path

import pytest

from kagglebot.agents.identity import resolve_oracle_model, resolve_primary_agent


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
