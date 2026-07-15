from __future__ import annotations

import hashlib
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
from kagglebot.validators import scan_text_for_secrets

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
    re.compile(r"\brequires? (?:an? )?writeups?\b"),
    re.compile(r"\bdeliverable[_ ]mode\s*[:=]\s*writeups?\b"),
    re.compile(r"\bdeliverable mode\s*[:=]\s*writeups?\b"),
    re.compile(r"\bwriteups?[- ]based\b"),
    re.compile(r"\bsubmissions?\b.*\b(?:made|submitted)\s+through\s+(?:a\s+)?writeups?\b"),
    re.compile(r"\bsubmit (?:an? )?writeups?\b"),
    re.compile(r"\bsubmissions? should include\b.*\bwriteups?\b"),
    re.compile(r"\bwriteups? (?:are|will be) judged\b"),
    re.compile(r"\bsubmissions? (?:are|will be) judged\b.*\brubric\b"),
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
    re.compile(r"\bsubmissions?(?:\s+(?:csv|file))?\s+must\s+contain\b"),
    re.compile(r"\bsample[_ -]submission\b"),
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
    if normalized in {"writeup", "writeups", "judged", "hackathon", "report"}:
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
    if any(pattern.search(line) for pattern in _NOTEBOOK_SUBMIT_PATTERNS):
        score += 2
    if any(pattern.search(line) for pattern in _CODE_COMPETITION_PATTERNS):
        score += 4
    return score


def infer_deliverable_mode_evidence(*texts: str) -> str | None:
    """Return a mode only when local competition text contains mode evidence."""
    lines = [line.strip().lower() for text in texts for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    writeup_score = sum(_line_writeup_score(line) for line in lines)
    leaderboard_score = sum(_line_csv_score(line) for line in lines)
    if writeup_score >= 3 and writeup_score > leaderboard_score:
        return "writeup"
    has_strong_leaderboard_evidence = any(
        pattern.search(line) for line in lines for pattern in _LEADERBOARD_STRONG_PATTERNS
    )
    if leaderboard_score >= 2 or has_strong_leaderboard_evidence:
        return "leaderboard"
    return None


def infer_deliverable_mode(*texts: str, default: str = "leaderboard") -> str:
    return infer_deliverable_mode_evidence(*texts) or default


def infer_submit_mode_evidence(*texts: str) -> str | None:
    lines = [line.strip().lower() for text in texts for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    if any(pattern.search(line) for line in lines for pattern in _NOTEBOOK_SUBMIT_PATTERNS):
        return "notebook"
    return None


def infer_submit_mode(*texts: str, default: str = "file") -> str:
    return infer_submit_mode_evidence(*texts) or default


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

    pipeline = str(metrics_payload.get("chosen_pipeline") or "the recorded competition pipeline")
    metric_text = f"{evaluation.metric}={evaluation.value:.6f} ({evaluation.direction})"
    report_lines = [
        f"# {paths.slug}: evidence-backed competition solution",
        "",
        "## Executive Summary",
        "",
        (
            f"This submission presents {pipeline}. The implementation was evaluated with the recorded "
            f"offline protocol and achieved {metric_text}. Because this competition is judged through a "
            "writeup, that value is reported as reproducible proxy evidence rather than an official score."
        ),
        "",
        "## Reproducibility Record",
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
        criterion_text = _writeup_section_text(
            title=title,
            pipeline=pipeline,
            metric_text=metric_text,
            requirements_summary=requirements_summary,
        )
        report_lines.extend(
            [
                f"## {title}",
                "",
                criterion_text,
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
    report_text = "\n".join(report_lines).strip() + "\n"
    report_path.write_text(report_text, encoding="utf-8")

    checklist_lines = [
        "# Automated Submission Validation",
        "",
        "- [x] Writeup mode was resolved from competition context.",
        "- [x] Report contains no generated placeholder instructions.",
        "- [x] Proxy evidence is explicitly distinguished from official judging.",
        "- [x] Report and evidence paths are included in the bundle manifest.",
        "- [ ] Browser submission is recorded only after Kaggle confirms the final state.",
    ]
    checklist_path.write_text("\n".join(checklist_lines) + "\n", encoding="utf-8")

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

    requirement_constraints = extract_writeup_constraints(paths)
    validation = validate_writeup_report(
        report_path,
        required_sections=section_titles,
        min_words=requirement_constraints.get("min_words"),
        max_words=requirement_constraints.get("max_words"),
    )
    metadata = {
        "deliverable_mode": "writeup",
        "status": "ready_for_submit" if validation["valid"] else "validation_failed",
        "run_id": run_id,
        "iteration": iteration,
        "report_path": str(report_path),
        "checklist_path": str(checklist_path),
        "evidence_path": str(evidence_path),
        "rubric_sections": section_titles,
        "requirements_summary": requirements_summary.splitlines() if requirements_summary else [],
        "requirement_constraints": requirement_constraints,
        "content_sha256": hashlib.sha256(report_text.encode("utf-8")).hexdigest(),
        "validation": validation,
    }
    write_json_object(metadata_path, metadata)
    return metadata


def validate_writeup_report(
    report_path: Path,
    *,
    required_sections: list[str],
    min_words: int | None = None,
    max_words: int | None = None,
) -> dict[str, object]:
    try:
        text = report_path.read_text(encoding="utf-8")
    except OSError as exc:
        return {"valid": False, "errors": [f"report unreadable: {exc}"], "word_count": 0}
    lowered = text.lower()
    errors: list[str] = []
    placeholder_markers = (
        "use this section to",
        "replace any placeholder",
        "todo",
        "tbd",
        "lorem ipsum",
    )
    if any(marker in lowered for marker in placeholder_markers):
        errors.append("report contains placeholder instructions")
    missing_sections = [title for title in required_sections if f"## {title}" not in text]
    if missing_sections:
        errors.append(f"missing required sections: {', '.join(missing_sections)}")
    secret_matches = scan_text_for_secrets(text)
    if secret_matches:
        errors.append("report contains text matching a secret pattern")
    word_count = len(re.findall(r"\b\w+\b", text))
    if word_count < 80:
        errors.append("report is too short to submit")
    if min_words is not None and word_count < min_words:
        errors.append(f"report has {word_count} words but competition requires at least {min_words}")
    if max_words is not None and word_count > max_words:
        errors.append(f"report has {word_count} words but competition allows at most {max_words}")
    return {
        "valid": not errors,
        "errors": errors,
        "word_count": word_count,
        "required_sections": required_sections,
        "min_words": min_words,
        "max_words": max_words,
    }


def extract_writeup_constraints(paths: CompetitionPaths) -> dict[str, int]:
    texts: list[str] = []
    for path in (paths.overview_md_path, paths.rules_md_path):
        try:
            texts.append(path.read_text(encoding="utf-8"))
        except OSError:
            continue
    text = "\n".join(texts).lower().replace(",", "")
    minimum: int | None = None
    maximum: int | None = None
    range_match = re.search(r"\bbetween\s+(\d+)\s+and\s+(\d+)\s+words?\b", text)
    if range_match:
        minimum = int(range_match.group(1))
        maximum = int(range_match.group(2))
    minimum_match = re.search(r"\b(?:at least|minimum(?: of)?)\s+(\d+)\s+words?\b", text)
    if minimum_match:
        minimum = int(minimum_match.group(1))
    maximum_match = re.search(r"\b(?:at most|maximum(?: of)?|no more than)\s+(\d+)\s+words?\b", text)
    if maximum_match:
        maximum = int(maximum_match.group(1))
    return {key: value for key, value in (("min_words", minimum), ("max_words", maximum)) if value is not None}


def _writeup_section_text(
    *,
    title: str,
    pipeline: str,
    metric_text: str,
    requirements_summary: str,
) -> str:
    lowered = title.lower()
    if any(marker in lowered for marker in ("context", "relevance", "impact", "motivation")):
        evidence = requirements_summary.replace("\n", " ").strip() or "the locally captured competition brief"
        return (
            f"The solution is scoped to the problem and constraints documented in {evidence}. "
            "Claims are limited to artifacts produced by this run so judges can trace the result to evidence."
        )
    if any(marker in lowered for marker in ("approach", "method", "technical", "quality", "innovation")):
        return (
            f"The implemented approach is {pipeline}. Training, validation, and artifact generation use the "
            "run's persisted configuration, which keeps the method reproducible and separates measured behavior "
            "from proposed future work."
        )
    if any(marker in lowered for marker in ("evidence", "evaluation", "result", "performance")):
        return (
            f"The recorded proxy result is {metric_text}. This is offline evidence from the configured evaluation "
            "protocol; it is not presented as an official judged score or as evidence beyond this run."
        )
    if any(marker in lowered for marker in ("limit", "risk", "future")):
        return (
            "The primary limitation is the gap between offline proxy evaluation and human judging. Dataset shift, "
            "rubric interpretation, and unavailable external attachments can affect the final result; the evidence "
            "manifest therefore preserves the exact run and avoids unsupported claims."
        )
    if any(marker in lowered for marker in ("submission", "reproduc", "open", "code")):
        return (
            "The final package binds this narrative to the run identifier, selected pipeline, evaluation record, "
            "and appendix manifest. Browser submission is allowed only after local validation, participation and "
            "rules checks, and content-hash duplicate detection succeed."
        )
    return (
        f"This criterion is addressed by {pipeline} and the recorded result {metric_text}. The supporting appendix "
        "contains the run-scoped evidence needed to inspect the claim without relying on unrecorded observations."
    )
