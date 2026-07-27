from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from kagglebot.json_utils import load_json_object

_OUTPUT_FILENAME_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*\.[A-Za-z0-9][A-Za-z0-9.]*)",
    re.IGNORECASE,
)
_OUTPUT_REQUIREMENT_MARKERS = (
    "file must",
    "file named",
    "file name",
    "filename",
    "final file",
    "must be named",
    "must contain",
    "must output",
    "output a file",
    "outputs a file",
    "produce a file",
    "produces a file",
    "required artifact",
    "required file",
    "required output",
    "save as",
    "saved as",
)
_NOTEBOOK_REQUIREMENT_PATTERNS = (
    re.compile(r"\b(?:one|a|the)\s+(?:kaggle\s+)?notebook\s*\(required\)", re.IGNORECASE),
    re.compile(r"\brequires?\s+(?:one|a|the)?\s*(?:kaggle\s+)?notebook\b", re.IGNORECASE),
    re.compile(r"\battached\s+(?:public\s+)?(?:kaggle\s+)?notebook\b", re.IGNORECASE),
    re.compile(r"\bsubmissions?\s+must\s+be\s+made\s+through\s+notebooks?\b", re.IGNORECASE),
    re.compile(r"\bprivate\s+notebook\s+submission\b", re.IGNORECASE),
)
_EXCLUDED_REQUIRED_OUTPUT_NAMES = {
    "evaluation_report.json",
    "metrics.json",
    "plan.json",
    "sample_submission.csv",
    "submission_manifest.json",
}
_EXCLUDED_REQUIRED_OUTPUT_PREFIXES = (
    "sample_",
    "test.",
    "train.",
)
_EXPLICIT_OUTPUT_KEYS = {
    "artifact_filename",
    "artifact_filenames",
    "notebook_output",
    "notebook_output_file",
    "notebook_output_files",
    "output_filename",
    "output_filenames",
    "required_artifact_filename",
    "required_artifact_filenames",
    "required_artifact",
    "required_output",
    "required_output_file",
    "required_output_files",
    "required_output_filename",
    "required_output_filenames",
}
_CONTEXT_FILENAMES = (
    "overview.md",
    "rules.md",
    "submission_format.md",
    "data.md",
    "code.md",
)


@dataclass(frozen=True)
class DeliverableArtifactContract:
    deliverable_mode: str
    submit_mode: str
    required_output_names: tuple[str, ...]
    requires_notebook: bool
    evidence_paths: tuple[Path, ...]

    @property
    def primary_output_name(self) -> str | None:
        return self.required_output_names[0] if self.required_output_names else None


def resolve_deliverable_artifact_contract(
    competition_dir: Path,
    *,
    deliverable_mode: object = None,
    submit_mode: object = None,
) -> DeliverableArtifactContract:
    """Resolve notebook/writeup output requirements from the persisted competition contract."""
    plan = load_json_object(competition_dir / "plan.json") or {}
    evaluation_spec = load_json_object(competition_dir / "context" / "evaluation_spec.json") or {}
    mode = _normalize_deliverable_mode(
        deliverable_mode or plan.get("deliverable_mode") or evaluation_spec.get("deliverable_mode")
    )
    normalized_submit_mode = _normalize_submit_mode(
        submit_mode or plan.get("submit_mode") or evaluation_spec.get("submit_mode")
    )

    output_names: list[str] = []
    for payload in (evaluation_spec, plan):
        output_names.extend(_explicit_output_names(payload))

    evidence_paths: list[Path] = []
    context_dir = competition_dir / "context"
    context_texts: list[str] = []
    for filename in _CONTEXT_FILENAMES:
        path = context_dir / filename
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        evidence_paths.append(path)
        context_texts.append(text)
        output_names.extend(_required_output_names_from_text(text))

    requires_notebook = normalized_submit_mode == "notebook" or any(
        pattern.search(text) for text in context_texts for pattern in _NOTEBOOK_REQUIREMENT_PATTERNS
    )
    normalized_output_names = [normalized for name in output_names if (normalized := _normalize_output_name(name))]
    return DeliverableArtifactContract(
        deliverable_mode=mode,
        submit_mode=normalized_submit_mode,
        required_output_names=tuple(dict.fromkeys(normalized_output_names)),
        requires_notebook=requires_notebook,
        evidence_paths=tuple(evidence_paths),
    )


def _normalize_deliverable_mode(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"writeup", "writeups", "judged", "hackathon", "report"}:
        return "writeup"
    return "leaderboard"


def _normalize_submit_mode(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"kernel", "notebook"}:
        return "notebook"
    return "file"


def _explicit_output_names(payload: dict[str, object]) -> list[str]:
    names: list[str] = []

    def visit(value: object, *, key: str = "") -> None:
        normalized_key = key.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized_key in _EXPLICIT_OUTPUT_KEYS:
            if isinstance(value, str):
                names.extend(_filenames(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        names.extend(_filenames(item))
            return
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                visit(child_value, key=str(child_key))

    visit(payload)
    return names


def _required_output_names_from_text(text: str) -> list[str]:
    names: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if not line or not any(marker in lowered for marker in _OUTPUT_REQUIREMENT_MARKERS):
            continue
        names.extend(_filenames(line))
    return names


def _filenames(text: str) -> list[str]:
    return [
        normalized
        for match in _OUTPUT_FILENAME_RE.finditer(text)
        if (normalized := _normalize_output_name(match.group("name")))
    ]


def _normalize_output_name(value: str) -> str:
    name = Path(str(value or "").strip().strip("`*'\".,:;()[]{}<>")).name
    lowered = name.lower()
    if not name or lowered in _EXCLUDED_REQUIRED_OUTPUT_NAMES:
        return ""
    if lowered.startswith(_EXCLUDED_REQUIRED_OUTPUT_PREFIXES):
        return ""
    return name
