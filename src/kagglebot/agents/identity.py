from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_MODEL = "gpt-5.6-sol"
_DEFAULT_REASONING_EFFORT = "xhigh"
_DEFAULT_ORACLE_MODEL = "gpt-5-pro"
_DEFAULT_ORACLE_IMPLEMENTATION_PROFILE = "sol-ultra"
_DEFAULT_ORACLE_REASONING_PROFILE = "ultra"


@dataclass(frozen=True)
class AgentIdentity:
    model: str
    reasoning_effort: str
    cli_command: str = "codex"
    log_alias: str = "gpt"
    display_family: str = "GPT"
    cli_profile: str | None = None
    reasoning_profile: str | None = None

    @property
    def version(self) -> str:
        match = re.search(r"(\d+(?:\.\d+)?)", self.model)
        return match.group(1) if match else self.model

    @property
    def display_name(self) -> str:
        version = self.version
        if re.fullmatch(r"\d+(?:\.\d+)?", version):
            return f"{self.display_family} {version}"
        return self.model

    @property
    def flow_token(self) -> str:
        version = self.version
        if re.fullmatch(r"\d+(?:\.\d+)?", version):
            return f"{self.log_alias}({version})"
        return f"{self.log_alias}({self.model})"


def resolve_primary_agent(
    *,
    config_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> AgentIdentity:
    env = os.environ if environ is None else environ
    settings = _load_project_agent_settings(config_path or _find_project_config())
    return AgentIdentity(
        model=env.get("KAGGLEBOT_PRIMARY_MODEL", settings.get("model", _DEFAULT_MODEL)),
        reasoning_effort=env.get(
            "KAGGLEBOT_PRIMARY_REASONING_EFFORT",
            settings.get("reasoning_effort", _DEFAULT_REASONING_EFFORT),
        ),
        cli_command=env.get("KAGGLEBOT_AGENT_CLI_COMMAND", "codex"),
        log_alias=env.get("KAGGLEBOT_AGENT_LOG_ALIAS", "gpt"),
        display_family=env.get("KAGGLEBOT_AGENT_DISPLAY_FAMILY", "GPT"),
    )


def resolve_oracle_model(
    *,
    config_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    env = os.environ if environ is None else environ
    settings = _load_project_agent_settings(config_path or _find_project_config())
    configured = settings.get("oracle_model", _DEFAULT_ORACLE_MODEL)
    return env.get("KAGGLEBOT_ORACLE_MODEL", configured).strip() or configured


def resolve_oracle_implementation_agent(
    *,
    config_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> AgentIdentity:
    env = os.environ if environ is None else environ
    settings = _load_project_agent_settings(config_path or _find_project_config())
    model = settings.get("oracle_implementation_model", settings.get("repository_implementation_model", _DEFAULT_MODEL))
    reasoning_effort = settings.get(
        "oracle_implementation_reasoning_effort",
        settings.get("repository_implementation_reasoning_effort", _DEFAULT_REASONING_EFFORT),
    )
    cli_profile = settings.get(
        "oracle_implementation_profile",
        settings.get("repository_implementation_profile", _DEFAULT_ORACLE_IMPLEMENTATION_PROFILE),
    )
    reasoning_profile = settings.get(
        "oracle_implementation_reasoning_profile",
        settings.get("repository_implementation_reasoning_profile", _DEFAULT_ORACLE_REASONING_PROFILE),
    )
    return AgentIdentity(
        model=env.get(
            "KAGGLEBOT_ORACLE_IMPLEMENTATION_MODEL",
            env.get("KAGGLEBOT_REPO_IMPLEMENTATION_MODEL", model),
        ),
        reasoning_effort=env.get(
            "KAGGLEBOT_ORACLE_IMPLEMENTATION_REASONING_EFFORT",
            env.get("KAGGLEBOT_REPO_IMPLEMENTATION_REASONING_EFFORT", reasoning_effort),
        ),
        cli_command=env.get("KAGGLEBOT_AGENT_CLI_COMMAND", "codex"),
        log_alias=env.get("KAGGLEBOT_AGENT_LOG_ALIAS", "gpt"),
        display_family=env.get("KAGGLEBOT_AGENT_DISPLAY_FAMILY", "GPT"),
        cli_profile=env.get(
            "KAGGLEBOT_ORACLE_IMPLEMENTATION_PROFILE",
            env.get("KAGGLEBOT_REPO_IMPLEMENTATION_PROFILE", cli_profile),
        ),
        reasoning_profile=env.get(
            "KAGGLEBOT_ORACLE_IMPLEMENTATION_REASONING_PROFILE",
            env.get("KAGGLEBOT_REPO_IMPLEMENTATION_REASONING_PROFILE", reasoning_profile),
        ),
    )


def resolve_repository_implementation_agent(
    *,
    config_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> AgentIdentity:
    """Backward-compatible alias for the shared Oracle-mediated implementer."""
    return resolve_oracle_implementation_agent(config_path=config_path, environ=environ)


def oracle_flow_token() -> str:
    model = resolve_oracle_model()
    label = "latest-pro" if model == _DEFAULT_ORACLE_MODEL else model
    return f"oracle({label})"


def _find_project_config(start_dir: Path | None = None) -> Path | None:
    current = (start_dir or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / "pyproject.toml"
        if candidate.is_file():
            return candidate

    source_checkout = Path(__file__).resolve().parents[3] / "pyproject.toml"
    return source_checkout if source_checkout.is_file() else None


def _load_project_agent_settings(config_path: Path | None) -> dict[str, str]:
    if config_path is None:
        return {}
    try:
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(f"Unable to read agent configuration from {config_path}: {exc}") from exc

    section: object = payload.get("tool", {})
    for key in ("kagglebot", "agent"):
        section = section.get(key, {}) if isinstance(section, dict) else {}
    if not isinstance(section, dict):
        return {}

    settings: dict[str, str] = {}
    for key in (
        "model",
        "reasoning_effort",
        "oracle_model",
        "oracle_implementation_model",
        "oracle_implementation_reasoning_effort",
        "oracle_implementation_profile",
        "oracle_implementation_reasoning_profile",
        "repository_implementation_model",
        "repository_implementation_reasoning_effort",
        "repository_implementation_profile",
        "repository_implementation_reasoning_profile",
    ):
        value = section.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"tool.kagglebot.agent.{key} must be a non-empty string in {config_path}")
        settings[key] = value.strip()
    return settings


PRIMARY_AGENT = resolve_primary_agent()
BRIEF_AGENT = PRIMARY_AGENT
STRATEGY_AGENT = PRIMARY_AGENT
IMPLEMENTATION_AGENT = PRIMARY_AGENT
ORACLE_IMPLEMENTATION_AGENT = resolve_oracle_implementation_agent()
REPOSITORY_IMPLEMENTATION_AGENT = ORACLE_IMPLEMENTATION_AGENT


def planning_flow_summary(*, strategy_token: str | None = None) -> str:
    return " -> ".join(
        (
            BRIEF_AGENT.flow_token,
            strategy_token or STRATEGY_AGENT.flow_token,
            IMPLEMENTATION_AGENT.flow_token,
        )
    )


def prompt_identity_mapping() -> dict[str, str]:
    return {
        "primary_agent_name": PRIMARY_AGENT.display_name,
        "primary_agent_model": PRIMARY_AGENT.model,
        "brief_agent_name": BRIEF_AGENT.display_name,
        "brief_agent_model": BRIEF_AGENT.model,
        "strategy_agent_name": STRATEGY_AGENT.display_name,
        "strategy_agent_model": STRATEGY_AGENT.model,
        "implementation_agent_name": IMPLEMENTATION_AGENT.display_name,
        "implementation_agent_model": IMPLEMENTATION_AGENT.model,
    }


def prompt_identity_format_args() -> dict[str, str]:
    return prompt_identity_mapping().copy()


def normalize_legacy_agent_text(text: str) -> str:
    replacements = {
        "# Codex Brief Extraction": f"# {BRIEF_AGENT.display_name} Brief Extraction",
        "You are Codex. Produce": f"You are {BRIEF_AGENT.display_name}. Produce",
        "# Codex Kernel Implementation": f"# {IMPLEMENTATION_AGENT.display_name} Kernel Implementation",
        "You are Codex. Implement": f"You are {IMPLEMENTATION_AGENT.display_name}. Implement",
        "# Codex Implementation (from Strategy)": (
            f"# {IMPLEMENTATION_AGENT.display_name} Implementation (from Strategy)"
        ),
        "You are Codex. Follow": f"You are {IMPLEMENTATION_AGENT.display_name}. Follow",
        "Give Codex step-by-step": f"Give {IMPLEMENTATION_AGENT.display_name} step-by-step",
        "Give Codex a step-by-step": f"Give {IMPLEMENTATION_AGENT.display_name} a step-by-step",
        "# Kagglebot Codex: Plan + Initial Model Implementation": (
            f"# Kagglebot {IMPLEMENTATION_AGENT.display_name}: Plan + Initial Model Implementation"
        ),
        "# Kagglebot Codex: Improvement Iteration": (
            f"# Kagglebot {IMPLEMENTATION_AGENT.display_name}: Improvement Iteration"
        ),
        "# Kagglebot Codex: Kernel Failure Fix": f"# Kagglebot {IMPLEMENTATION_AGENT.display_name}: Kernel Failure Fix",
        "# Kagglebot Codex: Postmortem & Knowledge Base Update": (
            f"# Kagglebot {IMPLEMENTATION_AGENT.display_name}: Postmortem & Knowledge Base Update"
        ),
    }
    rendered = text
    for source, target in replacements.items():
        rendered = rendered.replace(source, target)
    return rendered


def render_prompt_identity(template: str) -> str:
    rendered = normalize_legacy_agent_text(template)
    for key, value in prompt_identity_mapping().items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered
