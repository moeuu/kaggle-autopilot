from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RulesInfo:
    url: str
    source: str
    file: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BootstrapMeta:
    slug: str
    created_at: str
    rules: RulesInfo

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["rules"] = self.rules.to_dict()
        return payload


@dataclass
class PlanConfig:
    time_budget_min: int | None = None
    seed: int | None = None
    kernel_name: str | None = None
    internet: str | None = None
    accelerator: str | None = None
    kaggle_username: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AgentResult:
    transcript_path: str
    last_message_path: str
    returncode: int
    stdout: str
    stderr: str
