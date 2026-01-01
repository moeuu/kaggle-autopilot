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
    def data_dir(self) -> Path:
        return self.artifacts / "data"

    @property
    def context_dir(self) -> Path:
        return self.artifacts / "context"

    @property
    def prompts_dir(self) -> Path:
        return self.artifacts / "prompts"

    @property
    def artifacts(self) -> Path:
        return self.repo_root / "artifacts" / self.slug

    @property
    def models_dir(self) -> Path:
        return self.artifacts / "models"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts / "reports"

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
    def analysis_path(self) -> Path:
        return self.reports_dir / "competition_analysis.json"

    @property
    def training_report_path(self) -> Path:
        return self.reports_dir / "training_report.json"

    @property
    def model_info_path(self) -> Path:
        return self.models_dir / "model_info.json"

    @property
    def config_file(self) -> Path:
        return self.artifacts / "config.json"

    @property
    def meta_path(self) -> Path:
        return self.context_dir / "meta.json"

    @property
    def plan_path(self) -> Path:
        return self.context_dir / "plan.json"

    @property
    def rules_url_path(self) -> Path:
        return self.context_dir / "rules_url.txt"

    @property
    def dataset_summary_path(self) -> Path:
        return self.context_dir / "dataset_summary.txt"


def repo_root() -> Path:
    # Strict repo-root detection is possible, but MVP uses CWD.
    return Path.cwd()
