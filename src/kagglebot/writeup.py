from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from kagglebot.json_utils import load_json_object, write_json_object
from kagglebot.submission_sample_discovery import (
    is_tabular_data_path,
    path_mentions_role,
    select_sample_submission_path,
    tabular_data_row_count_capped,
    tabular_suffix,
)

if TYPE_CHECKING:
    from kagglebot.paths import CompetitionPaths
    from kagglebot.solver.evaluate import EvaluationResult

_WRITEUP_HINTS = (
    "writeup",
    "judged",
    "rubric",
    "panel",
    "manual grading",
    "manual review",
    "hackathon",
    "documentation and writeup",
)
_MANUAL_SUBMIT_HINTS = (
    "final submission",
    "submit 1 submission only",
    "manual grading",
    "manual review",
    "notebook",
    "writeup",
)
_WRITEUP_NEGATIVE_PATTERNS = (
    re.compile(r"\bnot a judged/writeup competition\b"),
    re.compile(r"\bnot (?:a )?judged(?:/writeup)? competition\b"),
    re.compile(r"\brather than `?writeup`?\b"),
    re.compile(r"\binstead of `?writeup`?\b"),
    re.compile(r"\bwithout (?:an? )?writeup\b"),
    re.compile(r"\bdeliverable[_ ]mode\s*[:=]\s*csv\b"),
    re.compile(r"\bdeliverable[_ ]mode\s*[:=]\s*leaderboard\b"),
)
_WRITEUP_STRONG_PATTERNS = (
    re.compile(r"\brequires? (?:an? )?writeup\b"),
    re.compile(r"\bdeliverable[_ ]mode\s*[:=]\s*writeup\b"),
    re.compile(r"\bdeliverable mode\s*[:=]\s*writeup\b"),
    re.compile(r"\bwriteup[- ]based\b"),
    re.compile(r"\bmanual (?:grading|review)\b"),
)
_LEADERBOARD_STRONG_PATTERNS = (
    re.compile(r"\bdeliverable[_ ]mode\s*[:=]\s*csv\b"),
    re.compile(r"\bdeliverable[_ ]mode\s*[:=]\s*leaderboard\b"),
    re.compile(r"\bdeliverable mode\s*[:=]\s*csv\b"),
    re.compile(r"\bdeliverable mode\s*[:=]\s*leaderboard\b"),
    re.compile(r"\bnormal leaderboard csv competition\b"),
    re.compile(r"\bnot a judged/writeup competition\b"),
    re.compile(r"\brather than `?writeup`?\b"),
    re.compile(r"\bsubmission(?:s)? must contain\b"),
    re.compile(r"\bsample submission\b"),
)
_LEADERBOARD_SUPPORT_PATTERNS = (
    re.compile(r"\bleaderboard\b"),
    re.compile(r"\bsubmission\.csv\b"),
    re.compile(r"\bprediction file\b"),
    re.compile(r"\bprobability predictions\b"),
)
_NOTEBOOK_SUBMIT_PATTERNS = (
    re.compile(r"submissions?\s+to\s+this\s+competition\s+must\s+be\s+made\s+through\s+notebooks?"),
    re.compile(r"submissions?\s+must\s+be\s+made\s+through\s+notebooks?"),
    re.compile(r"only\s+accepts?\s+submissions?\s+from\s+notebooks?"),
    re.compile(r"notebook[- ]only submissions?"),
    re.compile(r"submit(?:ted)?\s+through\s+notebooks?"),
)
_CODE_COMPETITION_PATTERNS = (
    re.compile(r"\bcode competition\b"),
    re.compile(r"\bcode competition faq\b"),
    re.compile(r"\bkernel submissions only\b"),
    re.compile(r"\bnotebook-only competition\b"),
    re.compile(r"\bhidden(?:/| or )full test\b"),
    re.compile(r"\bdummy\s+(?:public\s+)?test\b"),
    re.compile(r"\bpublic\s+test\s+(?:set\s+)?(?:is|contains)\s+(?:dummy|placeholder)\b"),
)


def normalize_deliverable_mode(value: object, *, default: str = "leaderboard") -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"writeup", "judged", "hackathon", "report"}:
        return "writeup"
    if normalized in {"csv", "submission", "leaderboard"}:
        return "leaderboard"
    return default


def normalize_submit_mode(value: object, *, default: str = "file") -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"notebook", "kernel"}:
        return "notebook"
    if normalized in {"file", "artifact", "csv"}:
        return "file"
    if normalized == "auto":
        return default
    return default


def _line_writeup_score(line: str) -> int:
    if any(pattern.search(line) for pattern in _WRITEUP_NEGATIVE_PATTERNS):
        return 0
    score = 0
    if any(pattern.search(line) for pattern in _WRITEUP_STRONG_PATTERNS):
        score += 2
    support_hits = sum(1 for marker in _WRITEUP_HINTS if marker in line)
    return score + min(support_hits, 2)


def _line_csv_score(line: str) -> int:
    score = 0
    if any(pattern.search(line) for pattern in _LEADERBOARD_STRONG_PATTERNS):
        score += 2
    if any(pattern.search(line) for pattern in _LEADERBOARD_SUPPORT_PATTERNS):
        score += 1
    return score


def infer_deliverable_mode(*texts: str, default: str = "leaderboard") -> str:
    lines = [line.strip().lower() for text in texts for line in text.splitlines() if line.strip()]
    if not lines:
        return default
    writeup_score = sum(_line_writeup_score(line) for line in lines)
    leaderboard_score = sum(_line_csv_score(line) for line in lines)
    return "writeup" if writeup_score >= 3 and writeup_score > leaderboard_score else default


def infer_submit_mode(*texts: str, default: str = "file") -> str:
    lines = [line.strip().lower() for text in texts for line in text.splitlines() if line.strip()]
    if not lines:
        return default
    if any(pattern.search(line) for line in lines for pattern in _NOTEBOOK_SUBMIT_PATTERNS):
        return "notebook"
    return default


def infer_code_competition(*texts: str, default: bool = False) -> bool:
    lines = [line.strip().lower() for text in texts for line in text.splitlines() if line.strip()]
    if not lines:
        return default
    return any(pattern.search(line) for line in lines for pattern in _CODE_COMPETITION_PATTERNS)


def infer_deliverable_mode_from_paths(
    paths: CompetitionPaths,
    *,
    explicit: object = None,
    default: str = "leaderboard",
) -> str:
    explicit_mode = normalize_deliverable_mode(explicit, default="")
    if explicit_mode:
        return explicit_mode
    texts = []
    for path in (
        paths.rules_md_path,
        paths.overview_md_path,
        paths.submission_format_md_path,
        paths.context_dir / "eval_advisor" / "sources_summary.md",
    ):
        try:
            texts.append(path.read_text(encoding="utf-8"))
        except OSError:
            continue
    return infer_deliverable_mode(*texts, default=default)


def infer_submit_mode_from_paths(
    paths: CompetitionPaths,
    *,
    explicit: object = None,
    default: str = "file",
) -> str:
    explicit_mode = normalize_submit_mode(explicit, default="")
    if explicit_mode:
        return explicit_mode
    texts = []
    for path in (
        paths.rules_md_path,
        paths.overview_md_path,
        paths.submission_format_md_path,
        paths.context_dir / "eval_advisor" / "sources_summary.md",
    ):
        try:
            texts.append(path.read_text(encoding="utf-8"))
        except OSError:
            continue
    return infer_submit_mode(*texts, default=default)


def infer_code_competition_from_paths(
    paths: CompetitionPaths,
    *,
    explicit: object = None,
    default: bool = False,
) -> bool:
    if explicit is not None:
        return bool(explicit)
    texts = []
    for path in (
        paths.rules_md_path,
        paths.overview_md_path,
        paths.data_md_path,
        paths.submission_format_md_path,
        paths.context_dir / "eval_advisor" / "sources_summary.md",
    ):
        try:
            texts.append(path.read_text(encoding="utf-8"))
        except OSError:
            continue
    if infer_code_competition(*texts, default=default):
        return True
    return _looks_like_notebook_hidden_test_contract(paths)


def _looks_like_notebook_hidden_test_contract(paths: CompetitionPaths) -> bool:
    """Infer code/notebook rerun submissions from a tiny public test contract.

    Some Kaggle code competitions only state `submit_mode: notebook` in the local
    evaluation spec and ship a tiny public test/sample pair. Embedding that local
    submission in a wrapper kernel passes local validation but fails Kaggle scoring
    once the notebook is evaluated against the real/full test contract.
    """
    if not _evaluation_spec_submit_mode_is_notebook(paths):
        return False
    test_path = _find_named_tabular_file(paths.data_dir, stem="test")
    sample_path = _find_sample_submission_path(paths.data_dir) or _find_sample_submission_path(paths.context_dir)
    if sample_path is None and paths.sample_submission_path.is_file():
        sample_path = paths.sample_submission_path
    if test_path is None or sample_path is None:
        return False
    test_rows = _tabular_data_row_count_at_most(test_path, limit=10)
    sample_rows = _tabular_data_row_count_at_most(sample_path, limit=10)
    return test_rows is True and sample_rows is True


def _evaluation_spec_submit_mode_is_notebook(paths: CompetitionPaths) -> bool:
    for path in (paths.context_dir / "evaluation_spec.json", paths.plan_path):
        payload = load_json_object(path)
        if payload is not None and normalize_submit_mode(payload.get("submit_mode"), default="") == "notebook":
            return True
    return False


def _tabular_data_row_count_at_most(path: Path, *, limit: int) -> bool | None:
    row_count = tabular_data_row_count_capped(path, cap=limit)
    if row_count is None:
        return None
    return row_count <= limit


def _find_named_tabular_file(data_dir: Path, *, stem: str) -> Path | None:
    lowered_stem = stem.lower()
    if not data_dir.exists():
        return None
    try:
        candidates = [
            path
            for path in data_dir.rglob("*")
            if path.is_file()
            and is_tabular_data_path(path)
            and (_tabular_stem(path).lower() == lowered_stem or path_mentions_role(path, lowered_stem))
            and ".kagglebot_cache" not in {part.lower() for part in path.parts}
        ]
    except OSError:
        return None
    if not candidates:
        return None
    return min(candidates, key=lambda path: (len(path.relative_to(data_dir).parts), str(path).lower()))


def _find_sample_submission_path(data_dir: Path) -> Path | None:
    if not data_dir.exists():
        return None
    try:
        files = [
            path
            for path in data_dir.rglob("*")
            if path.is_file()
            and is_tabular_data_path(path)
            and ".kagglebot_cache" not in {part.lower() for part in path.parts}
        ]
    except OSError:
        return None
    return select_sample_submission_path(files)


def _tabular_stem(path: Path) -> str:
    suffix = tabular_suffix(path)
    name = path.name
    if suffix and name.lower().endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def summarize_writeup_requirements(paths: CompetitionPaths, *, max_lines: int = 8) -> str:
    lines: list[str] = []
    for path in (
        paths.overview_md_path,
        paths.rules_md_path,
        paths.context_dir / "eval_advisor" / "sources_summary.md",
    ):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for raw_line in text.splitlines():
            line = raw_line.strip()
            lowered = line.lower()
            if not line:
                continue
            if _line_writeup_score(lowered) > _line_csv_score(lowered) and any(
                marker in lowered for marker in _WRITEUP_HINTS + _MANUAL_SUBMIT_HINTS
            ):
                if line not in lines:
                    lines.append(line)
            if len(lines) >= max_lines:
                break
        if len(lines) >= max_lines:
            break
    return "\n".join(f"- {line}" for line in lines)


def extract_rubric_sections(paths: CompetitionPaths) -> list[str]:
    names: list[str] = []
    pattern = re.compile(r"([A-Z][A-Za-z0-9 /&'_-]{3,80})\s*\((\d{1,3})\)")
    for path in (paths.overview_md_path, paths.rules_md_path):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for match in pattern.finditer(text):
            name = re.sub(r"\s+", " ", match.group(1)).strip(" .:-")
            if name and name not in names:
                names.append(name)
    return names[:8]


def build_writeup_bundle(
    *,
    paths: CompetitionPaths,
    run_id: str,
    iteration: int,
    resolved: dict[str, object],
    evaluation: EvaluationResult,
    metrics_payload: dict[str, object],
    top1_info: dict[str, object] | None,
) -> dict[str, object]:
    run_dir = paths.run_dir(run_id)
    writeup_dir = run_dir / "writeup"
    appendix_dir = writeup_dir / "appendix"
    writeup_dir.mkdir(parents=True, exist_ok=True)
    appendix_dir.mkdir(parents=True, exist_ok=True)

    rubric_sections = extract_rubric_sections(paths)
    section_titles = rubric_sections or [
        "Competition Context",
        "Approach",
        "Offline Evidence",
        "Limitations",
        "Submission Plan",
    ]
    requirements_summary = summarize_writeup_requirements(paths)

    report_path = writeup_dir / "report.md"
    checklist_path = writeup_dir / "submission_checklist.md"
    evidence_path = appendix_dir / "proxy_metrics.json"
    metadata_path = writeup_dir / "writeup_metadata.json"

    report_lines = [
        f"# {paths.slug} writeup",
        "",
        f"- Run ID: `{run_id}`",
        f"- Iteration: `{iteration}`",
        "- Official deliverable mode: `writeup`",
        f"- Proxy metric: `{evaluation.metric}` = `{evaluation.value:.6f}` ({evaluation.direction})",
        (
            f"- Selected pipeline: `{metrics_payload.get('chosen_pipeline')}`"
            if metrics_payload.get("chosen_pipeline")
            else "- Selected pipeline: unavailable"
        ),
        "- Note: offline metrics below are supporting evidence only, not the official judged score.",
        "",
        "## Competition Requirements",
        requirements_summary or "- No explicit writeup requirements were extracted from local context.",
        "",
    ]
    for title in section_titles:
        report_lines.extend(
            [
                f"## {title}",
                "",
                (
                    "Use this section to address the competition-specific criterion and tie claims back to the "
                    "offline evidence, methodology, and artifacts in the appendix."
                ),
                "",
            ]
        )
    report_lines.extend(
        [
            "## Appendix",
            "",
            f"- Proxy metrics JSON: `{evidence_path.relative_to(run_dir)}`",
            f"- Submission checklist: `{checklist_path.relative_to(run_dir)}`",
            "",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    checklist_lines = [
        "# Manual Submission Checklist",
        "",
        "- Confirm the competition accepts judged/writeup-style submissions.",
        "- Review `report.md` and replace any placeholder text with final narrative.",
        "- Verify notebook/writeup page requirements from Kaggle rules/overview.",
        "- Ensure any linked notebook or artifacts referenced by the writeup are published and accessible.",
        "- Perform the remaining Kaggle UI submission steps manually if no judged-submit API path exists.",
    ]
    checklist_path.write_text("\n".join(checklist_lines), encoding="utf-8")

    evidence_payload = {
        "deliverable_mode": "writeup",
        "proxy_metric": evaluation.metric,
        "proxy_value": evaluation.value,
        "proxy_direction": evaluation.direction,
        "run_id": run_id,
        "iteration": iteration,
        "chosen_pipeline": metrics_payload.get("chosen_pipeline"),
        "top1_public": top1_info or {},
        "resolved_plan": {
            "target_metric": resolved.get("target_metric"),
            "target_score": resolved.get("target_score"),
            "split_strategy": resolved.get("split_strategy"),
            "deliverable_mode": resolved.get("deliverable_mode"),
        },
    }
    write_json_object(evidence_path, evidence_payload)

    metadata = {
        "deliverable_mode": "writeup",
        "status": "manual_finalization_required",
        "run_id": run_id,
        "iteration": iteration,
        "report_path": str(report_path),
        "checklist_path": str(checklist_path),
        "evidence_path": str(evidence_path),
        "rubric_sections": section_titles,
        "requirements_summary": requirements_summary.splitlines() if requirements_summary else [],
    }
    write_json_object(metadata_path, metadata)
    return metadata
