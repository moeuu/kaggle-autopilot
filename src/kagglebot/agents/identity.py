from __future__ import annotations

import os
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentIdentity:
    model: str
    reasoning_effort: str
    cli_command: str = "codex"
    log_alias: str = "gpt"
    display_family: str = "GPT"

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


PRIMARY_AGENT = AgentIdentity(
    model=os.environ.get("KAGGLEBOT_PRIMARY_MODEL", "gpt-5.5"),
    reasoning_effort=os.environ.get("KAGGLEBOT_PRIMARY_REASONING_EFFORT", "xhigh"),
    cli_command=os.environ.get("KAGGLEBOT_AGENT_CLI_COMMAND", "codex"),
    log_alias=os.environ.get("KAGGLEBOT_AGENT_LOG_ALIAS", "gpt"),
    display_family=os.environ.get("KAGGLEBOT_AGENT_DISPLAY_FAMILY", "GPT"),
)
BRIEF_AGENT = PRIMARY_AGENT
STRATEGY_AGENT = PRIMARY_AGENT
IMPLEMENTATION_AGENT = PRIMARY_AGENT


def planning_flow_summary() -> str:
    return " -> ".join(
        (
            BRIEF_AGENT.flow_token,
            STRATEGY_AGENT.flow_token,
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
