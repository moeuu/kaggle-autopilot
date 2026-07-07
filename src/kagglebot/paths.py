from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kagglebot.compression_suffixes import strip_compression_suffix
from kagglebot.submission_sample_discovery import (
    TABULAR_INPUT_SUFFIXES,
    TABULAR_INPUT_SUFFIXES_ORDERED,
    TABULAR_TEXT_SUFFIXES,
    sample_name_score,
    tabular_stem,
    tabular_suffix,
)

_SAMPLE_SUBMISSION_SUFFIXES = TABULAR_INPUT_SUFFIXES_ORDERED
_SAMPLE_SUBMISSION_SUFFIX_SET = set(TABULAR_INPUT_SUFFIXES)
_SAMPLE_SUBMISSION_HEAD_SUFFIX_SET = {strip_compression_suffix(suffix) for suffix in TABULAR_TEXT_SUFFIXES}


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
    def competition_policy_path(self) -> Path:
        return self.context_dir / "competition_policy.json"

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
    def analysis_path(self) -> Path:
        return self.context_dir / "analysis.json"

    @property
    def sample_submission_path(self) -> Path:
        return self._existing_context_sample_submission_artifact() or self.context_dir / "sample_submission.csv"

    @property
    def sample_submission_head_path(self) -> Path:
        existing = self._existing_context_tabular_artifact("sample_submission_head")
        if existing is not None:
            return existing
        return self.context_dir / "sample_submission_head.csv"

    def context_sample_submission_path_for_suffix(self, suffix: str) -> Path:
        normalized = suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        if normalized not in _SAMPLE_SUBMISSION_SUFFIX_SET:
            normalized = ".csv"
        return self.context_dir / f"sample_submission{normalized}"

    def context_sample_submission_head_path_for_suffix(self, suffix: str) -> Path:
        normalized = suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        normalized = strip_compression_suffix(normalized)
        if normalized not in _SAMPLE_SUBMISSION_HEAD_SUFFIX_SET:
            normalized = ".csv"
        return self.context_dir / f"sample_submission_head{normalized}"

    def _existing_context_tabular_artifact(self, stem: str) -> Path | None:
        if not self.context_dir.exists():
            return None
        candidates = [
            path
            for path in self.context_dir.glob(f"{stem}.*")
            if path.is_file() and tabular_suffix(path) in _SAMPLE_SUBMISSION_SUFFIX_SET
        ]
        if not candidates:
            return None

        def _key(path: Path) -> tuple[int, int, str]:
            try:
                mtime_ns = path.stat().st_mtime_ns
            except OSError:
                mtime_ns = 0
            suffix_rank = _SAMPLE_SUBMISSION_SUFFIXES.index(tabular_suffix(path))
            return (mtime_ns, suffix_rank, path.name)

        return max(candidates, key=_key)

    def _existing_context_sample_submission_artifact(self) -> Path | None:
        canonical = self._existing_context_tabular_artifact("sample_submission")
        if canonical is not None:
            return canonical
        if not self.context_dir.exists():
            return None
        candidates = [
            path
            for path in self.context_dir.iterdir()
            if path.is_file()
            and tabular_suffix(path) in _SAMPLE_SUBMISSION_SUFFIX_SET
            and tabular_stem(path).lower() != "sample_submission_head"
            and sample_name_score(path) >= 2
        ]
        if not candidates:
            return None

        def _key(path: Path) -> tuple[int, int, int, str]:
            try:
                mtime_ns = path.stat().st_mtime_ns
            except OSError:
                mtime_ns = 0
            suffix_rank = _SAMPLE_SUBMISSION_SUFFIXES.index(tabular_suffix(path))
            return (sample_name_score(path), mtime_ns, suffix_rank, path.name)

        return max(candidates, key=_key)

    @property
    def top1_public_path(self) -> Path:
        return self.context_dir / "top1_public.json"

    @property
    def method_registry_path(self) -> Path:
        return self.context_dir / "method_registry.json"

    @property
    def method_scout_queries_path(self) -> Path:
        return self.context_dir / "method_scout_queries.json"

    @property
    def source_registry_path(self) -> Path:
        return self.context_dir / "source_registry.json"

    @property
    def validation_registry_path(self) -> Path:
        return self.context_dir / "validation_registry.json"

    @property
    def validation_lab_report_path(self) -> Path:
        return self.context_dir / "validation_lab_report.json"

    @property
    def win_contract_path(self) -> Path:
        return self.context_dir / "win_contract.json"

    @property
    def private_robustness_report_path(self) -> Path:
        return self.context_dir / "private_robustness_report.json"

    @property
    def top1_exhaustion_report_path(self) -> Path:
        return self.context_dir / "top1_exhaustion_report.json"

    @property
    def experiment_graph_path(self) -> Path:
        return self.context_dir / "experiment_graph.json"

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

    @property
    def data_raw(self) -> Path:
        return self.data_dir

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
