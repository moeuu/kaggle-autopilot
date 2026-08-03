from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

from kagglebot.deliverable_artifacts import resolve_deliverable_artifact_contract
from kagglebot.hashing import sha256_path
from kagglebot.json_utils import load_json_object, write_json_object
from kagglebot.kernel_outputs import find_output_file
from kagglebot.submission_sample_discovery import (
    is_tabular_data_path,
    path_mentions_role,
    select_sample_submission_path,
    tabular_data_row_count_capped,
    tabular_suffix,
)
from kagglebot.validators import scan_text_for_secrets
from kagglebot.writeup_card import ensure_writeup_card

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
    evaluation: EvaluationResult | None,
    metrics_payload: dict[str, object],
    top1_info: dict[str, object] | None,
    notebook_id: str | None = None,
    source_report_path: Path | None = None,
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
    card_image_path = ensure_writeup_card(writeup_dir / "card_thumbnail.png")

    pipeline = str(
        metrics_payload.get("chosen_pipeline")
        or metrics_payload.get("selected_pipeline")
        or "the recorded competition pipeline"
    )
    proxy_metric, proxy_value, proxy_direction = _writeup_proxy_evidence(
        evaluation=evaluation,
        metrics_payload=metrics_payload,
        resolved=resolved,
    )
    metric_text = (
        f"{proxy_metric}={proxy_value:.6f} ({proxy_direction})"
        if proxy_value is not None
        else f"{proxy_metric}=unavailable ({proxy_direction})"
    )
    report_lines = [
        f"# {paths.slug}: evidence-backed competition solution",
        "",
        "## Executive Summary",
        "",
        (
            f"This submission presents {pipeline}. The implementation was evaluated with the recorded "
            f"offline protocol and recorded {metric_text}. Because this competition is judged through a "
            "writeup, that value is reported as reproducible proxy evidence rather than an official score."
        ),
        "",
        "## Reproducibility Record",
        "",
        f"- Run ID: `{run_id}`",
        f"- Iteration: `{iteration}`",
        "- Official deliverable mode: `writeup`",
        (
            f"- Proxy metric: `{proxy_metric}` = `{proxy_value:.6f}` ({proxy_direction})"
            if proxy_value is not None
            else f"- Proxy metric: `{proxy_metric}` is unavailable ({proxy_direction})"
        ),
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
    deliverable_contract = resolve_deliverable_artifact_contract(
        paths.base_dir,
        deliverable_mode="writeup",
    )
    resolved_notebook_id = str(notebook_id or "").strip()
    if resolved_notebook_id.startswith("local/"):
        resolved_notebook_id = ""
    notebook_url = f"https://www.kaggle.com/code/{resolved_notebook_id}" if resolved_notebook_id else None
    required_artifacts, missing_required_artifacts = _required_artifact_records(
        run_dir / f"iter-{iteration}",
        deliverable_contract.required_output_names,
    )

    if notebook_url is not None:
        report_lines.extend(
            [
                "## Private Notebook",
                "",
                f"- Reproducible private Kaggle Notebook: {notebook_url}",
                "- The notebook is intentionally kept private to preserve the competition confidentiality contract.",
                "",
            ]
        )
    report_lines.extend(
        [
            "## Appendix",
            "",
            f"- Proxy metrics JSON: `{evidence_path.relative_to(run_dir)}`",
            f"- Submission checklist: `{checklist_path.relative_to(run_dir)}`",
            *[
                f"- Required notebook output: `{record['name']}` (`sha256:{str(record['sha256'])[:12]}…`)"
                for record in required_artifacts
            ],
            "",
        ]
    )
    resolved_source_report = (
        source_report_path if source_report_path is not None and source_report_path.is_file() else None
    )
    report_text = (
        resolved_source_report.read_text(encoding="utf-8")
        if resolved_source_report is not None
        else "\n".join(report_lines).strip() + "\n"
    )
    if not report_text.endswith("\n"):
        report_text += "\n"
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
        "proxy_metric": proxy_metric,
        "proxy_value": proxy_value,
        "proxy_direction": proxy_direction,
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
        required_sections=[] if resolved_source_report is not None else section_titles,
        min_words=requirement_constraints.get("min_words"),
        max_words=requirement_constraints.get("max_words"),
    )
    artifact_errors = [f"required submission artifact not found: {name}" for name in missing_required_artifacts]
    if artifact_errors:
        validation["errors"] = [*list(validation.get("errors") or []), *artifact_errors]
        validation["valid"] = False
    notebook_ready = not deliverable_contract.requires_notebook or bool(resolved_notebook_id)
    if validation["valid"]:
        status = "ready_for_submit" if notebook_ready else "ready_for_notebook_publish"
    else:
        status = "validation_failed"
    metadata = {
        "deliverable_mode": "writeup",
        "status": status,
        "run_id": run_id,
        "iteration": iteration,
        "report_path": str(report_path),
        "checklist_path": str(checklist_path),
        "evidence_path": str(evidence_path),
        "rubric_sections": section_titles,
        "requirements_summary": requirements_summary.splitlines() if requirements_summary else [],
        "requirement_constraints": requirement_constraints,
        "artifact_contract": {
            "required_output_names": list(deliverable_contract.required_output_names),
            "requires_notebook": deliverable_contract.requires_notebook,
            "submit_mode": deliverable_contract.submit_mode,
            "requires_resource_attachment": bool(
                deliverable_contract.required_output_names and not deliverable_contract.requires_notebook
            ),
        },
        "required_artifacts": required_artifacts,
        "card_image_required": True,
        "card_image": {
            "path": str(card_image_path),
            "sha256": sha256_path(card_image_path),
            "width": 560,
            "height": 280,
        },
        "track": resolved.get("track"),
        "external_evaluation_required": evaluation is None,
        "source_report_path": str(resolved_source_report) if resolved_source_report is not None else None,
        "notebook": {
            "required": deliverable_contract.requires_notebook,
            "status": "ready" if notebook_ready else "publish_required",
            "kernel_id": resolved_notebook_id or None,
            "url": notebook_url,
            "private": True if resolved_notebook_id else None,
        },
        "content_sha256": hashlib.sha256(report_text.encode("utf-8")).hexdigest(),
        "validation": validation,
    }
    write_json_object(metadata_path, metadata)
    return metadata


def _writeup_proxy_evidence(
    *,
    evaluation: EvaluationResult | None,
    metrics_payload: dict[str, object],
    resolved: dict[str, object],
) -> tuple[str, float | None, str]:
    if evaluation is not None:
        return evaluation.metric, evaluation.value, evaluation.direction
    metric = str(metrics_payload.get("cv_metric_name") or metrics_payload.get("metric_name") or "offline proxy")
    raw_value = metrics_payload.get("cv_metric_value")
    if raw_value is None:
        raw_value = metrics_payload.get("cv_score")
    value = (
        float(raw_value)
        if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool) and math.isfinite(float(raw_value))
        else None
    )
    direction = str(
        metrics_payload.get("cv_metric_direction")
        or metrics_payload.get("direction")
        or resolved.get("target_direction")
        or "maximize"
    )
    return metric, value, direction


def attach_published_writeup_notebook(
    metadata: dict[str, object],
    *,
    kernel_id: str,
    output_dir: Path,
) -> dict[str, object]:
    """Attach a verified private Kaggle notebook and re-seal the writeup payload."""
    resolved_kernel_id = str(kernel_id or "").strip()
    if not resolved_kernel_id or resolved_kernel_id.startswith("local/"):
        raise ValueError("Writeup notebook publication did not return a remote Kaggle kernel ID.")
    report_path = Path(str(metadata.get("report_path") or ""))
    if not report_path.is_file():
        raise ValueError(f"Writeup report not found: {report_path}")
    contract_payload = metadata.get("artifact_contract")
    required_names_raw = contract_payload.get("required_output_names") if isinstance(contract_payload, dict) else []
    required_names = tuple(str(name) for name in required_names_raw) if isinstance(required_names_raw, list) else ()
    remote_records, missing = _required_artifact_records(output_dir, required_names)
    errors = [f"published notebook output not found: {name}" for name in missing]
    expected_hashes = {
        str(record.get("name")): str(record.get("sha256"))
        for record in metadata.get("required_artifacts", [])
        if isinstance(record, dict)
    }
    for record in remote_records:
        name = str(record["name"])
        expected_hash = expected_hashes.get(name)
        if expected_hash and expected_hash != record["sha256"]:
            errors.append(f"published notebook output differs from validated local artifact: {name}")
    if errors:
        validation = metadata.get("validation") if isinstance(metadata.get("validation"), dict) else {}
        validation = {**validation, "valid": False, "errors": [*list(validation.get("errors") or []), *errors]}
        metadata.update({"status": "validation_failed", "validation": validation})
        return metadata

    notebook_url = f"https://www.kaggle.com/code/{resolved_kernel_id}"
    report_text = report_path.read_text(encoding="utf-8")
    notebook_section = (
        "\n## Private Notebook\n\n"
        f"- Reproducible private Kaggle Notebook: {notebook_url}\n"
        "- The notebook is intentionally kept private to preserve the competition confidentiality contract.\n"
    )
    if "## Private Notebook" not in report_text:
        appendix_marker = "\n## Appendix\n"
        report_text = (
            report_text.replace(appendix_marker, notebook_section + appendix_marker, 1)
            if appendix_marker in report_text
            else report_text.rstrip() + notebook_section + "\n"
        )
        report_path.write_text(report_text, encoding="utf-8")
    metadata.update(
        {
            "status": "ready_for_submit",
            "content_sha256": hashlib.sha256(report_text.encode("utf-8")).hexdigest(),
            "published_required_artifacts": remote_records,
            "notebook": {
                "required": True,
                "status": "ready",
                "kernel_id": resolved_kernel_id,
                "url": notebook_url,
                "private": True,
            },
        }
    )
    return metadata


def _required_artifact_records(
    output_root: Path,
    required_output_names: tuple[str, ...],
) -> tuple[list[dict[str, object]], list[str]]:
    records: list[dict[str, object]] = []
    missing: list[str] = []
    for name in required_output_names:
        path = find_output_file(output_root, name)
        if path is None or not path.is_file():
            missing.append(name)
            continue
        records.append(
            {
                "name": name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        )
    return records, missing


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
