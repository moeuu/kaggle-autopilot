from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def resolve_artifacts_dir(workdir: Path, artifacts_dir: Path) -> Path:
    """
    Resolve artifacts directory relative to workdir unless absolute.

    Args:
        workdir: Base working directory.
        artifacts_dir: User-provided artifacts path.

    Returns:
        Absolute Path to artifacts directory.
    """
    if artifacts_dir.is_absolute():
        return artifacts_dir
    return (workdir / artifacts_dir).resolve()


@dataclass(frozen=True)
class CompetitionPaths:
    """
    Paths for a competition's artifacts.
    """

    slug: str
    artifacts_dir: Path | None = None
    repo_root: Path | None = None

    def __post_init__(self) -> None:
        if self.artifacts_dir is None:
            if self.repo_root is None:
                raise ValueError("Either artifacts_dir or repo_root must be provided.")
            object.__setattr__(self, "artifacts_dir", Path(self.repo_root) / "artifacts")
        if self.repo_root is None:
            object.__setattr__(self, "repo_root", self.artifacts_dir.parent)

    @property
    def base_dir(self) -> Path:
        return self.artifacts_dir / self.slug

    @property
    def meta_path(self) -> Path:
        return self.base_dir / "meta.json"

    @property
    def plan_path(self) -> Path:
        return self.base_dir / "plan.json"

    @property
    def context_dir(self) -> Path:
        return self.base_dir / "context"

    @property
    def context_agent_dir(self) -> Path:
        return self.context_dir / "agent"

    @property
    def rules_url_path(self) -> Path:
        return self.context_dir / "rules_url.txt"

    @property
    def rules_html_path(self) -> Path:
        return self.context_dir / "rules.html"

    @property
    def rules_md_path(self) -> Path:
        return self.context_dir / "rules.md"

    @property
    def overview_md_path(self) -> Path:
        return self.context_dir / "overview.md"

    @property
    def data_md_path(self) -> Path:
        return self.context_dir / "data.md"

    @property
    def submission_format_md_path(self) -> Path:
        return self.context_dir / "submission_format.md"

    @property
    def code_md_path(self) -> Path:
        return self.context_dir / "code.md"

    @property
    def code_notebooks_dir(self) -> Path:
        return self.context_dir / "code_notebooks"

    @property
    def code_notebooks_index_path(self) -> Path:
        return self.context_dir / "code_notebooks_index.json"

    @property
    def models_md_path(self) -> Path:
        return self.context_dir / "models.md"

    @property
    def discussion_md_path(self) -> Path:
        return self.context_dir / "discussion.md"

    @property
    def discussion_threads_dir(self) -> Path:
        return self.context_dir / "discussion_threads"

    @property
    def discussion_threads_index_path(self) -> Path:
        return self.context_dir / "discussion_threads_index.json"

    @property
    def reference_inputs_dir(self) -> Path:
        return self.context_dir / "reference_inputs"

    @property
    def reference_inputs_manifest_path(self) -> Path:
        return self.context_dir / "reference_inputs_manifest.json"

    @property
    def knowledge_hints_path(self) -> Path:
        return self.context_dir / "knowledge_hints.txt"

    @property
    def dataset_profile_path(self) -> Path:
        return self.context_dir / "dataset_profile.json"

    @property
    def sample_submission_path(self) -> Path:
        return self.context_dir / "sample_submission.csv"

    @property
    def sample_submission_head_path(self) -> Path:
        return self.context_dir / "sample_submission_head.csv"

    @property
    def top1_public_path(self) -> Path:
        return self.context_dir / "top1_public.json"

    @property
    def prompts_dir(self) -> Path:
        return self.base_dir / "prompts"

    @property
    def codex_plan_and_implement_prompt(self) -> Path:
        return self.prompts_dir / "codex_plan_and_implement.md"

    @property
    def codex_improve_template(self) -> Path:
        return self.prompts_dir / "codex_improve.md"

    @property
    def codex_kernel_fix_template(self) -> Path:
        return self.prompts_dir / "codex_kernel_fix.md"

    @property
    def runs_dir(self) -> Path:
        return self.base_dir / "runs"

    @property
    def submissions_dir(self) -> Path:
        return self.base_dir / "submissions"

    @property
    def kernels_dir(self) -> Path:
        return self.base_dir / "kernels"

    @property
    def kernel_source_dir(self) -> Path:
        return self.base_dir / "kernel"

    @property
    def submission_ledger_path(self) -> Path:
        return self.submissions_dir / "ledger.jsonl"

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def iter_dir(self, run_id: str, iteration: int) -> Path:
        return self.run_dir(run_id) / f"iter-{iteration}"

    def kernel_run_dir(self, run_id: str) -> Path:
        return self.kernels_dir / run_id


@dataclass(frozen=True)
class KnowledgePaths:
    """
    Paths for persistent knowledge base storage.
    """

    workdir: Path

    @property
    def knowledge_dir(self) -> Path:
        return self.workdir / "knowledge"

    @property
    def taxonomy_path(self) -> Path:
        return self.knowledge_dir / "taxonomy.yml"

    @property
    def kb_path(self) -> Path:
        return self.knowledge_dir / "kb.sqlite"


def repo_root() -> Path:
    """Backward-compatible repo root helper (current working directory)."""
    return Path.cwd()
