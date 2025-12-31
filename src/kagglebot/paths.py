from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CompetitionPaths:
    slug: str
    repo_root: Path

    @property
    def data_raw(self) -> Path:
        return self.repo_root / "data" / self.slug / "raw"

    @property
    def artifacts(self) -> Path:
        return self.repo_root / "artifacts" / self.slug

    @property
    def models_dir(self) -> Path:
        return self.artifacts / "models"

    @property
    def submissions_dir(self) -> Path:
        return self.artifacts / "submissions"

    @property
    def runs_dir(self) -> Path:
        return self.artifacts / "runs"

    @property
    def submission_csv(self) -> Path:
        return self.submissions_dir / "submission.csv"

    @property
    def submission_ledger(self) -> Path:
        return self.submissions_dir / "history.jsonl"

    @property
    def config_file(self) -> Path:
        return self.artifacts / "config.json"


def repo_root() -> Path:
    # Strict repo-root detection is possible, but MVP uses CWD.
    return Path.cwd()
