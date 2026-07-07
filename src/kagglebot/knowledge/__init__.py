from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from kagglebot.agents.identity import IMPLEMENTATION_AGENT
from kagglebot.asset_modality import (
    ARRAY_SUFFIXES,
    AUDIO_SUFFIXES,
    BIO_STRUCTURE_SUFFIXES,
    DOCUMENT_SUFFIXES,
    GEOSPATIAL_SUFFIXES,
    GRAPH_SUFFIXES,
    IMAGE_SUFFIXES,
    MEDICAL_IMAGE_SUFFIXES,
    POINT_CLOUD_SUFFIXES,
    SCIENTIFIC_ARRAY_SUFFIXES,
    SIGNAL_SUFFIXES,
    VIDEO_SUFFIXES,
    asset_suffix,
    infer_asset_modality,
    infer_asset_modality_from_extensions,
)
from kagglebot.baseline_tokens import ID_LIKE_COLUMN_NAMES
from kagglebot.compression_suffixes import open_compressed_text, strip_compression_suffix
from kagglebot.env_utils import parse_int_value
from kagglebot.json_utils import parse_json_array_text
from kagglebot.knowledge.classification import (
    classify_cause_category,
    classify_error_category,
    classify_fix_category,
)
from kagglebot.knowledge.profile_utils import safe_nunique
from kagglebot.knowledge.repositories import InsightRepository, TaxonomyRepository
from kagglebot.paths import KnowledgePaths
from kagglebot.plan_policy import apply_competition_eval_override
from kagglebot.rna_structure import detect_rna_structure_task, extract_target_id, load_rna_structure_task
from kagglebot.solver.io import (
    ensure_sample_submission,
    find_competition_files,
    infer_prediction_kind,
    infer_submission_layout,
    infer_task,
    materialize_sqlite_tables,
    read_table,
    task_for_prediction_kind,
)
from kagglebot.submission_sample_discovery import (
    TABULAR_STRUCTURED_SUFFIXES,
    TABULAR_TEXT_SUFFIXES,
    is_json_lines_tabular_suffix,
    is_tabular_data_path,
    sample_name_score,
    tabular_suffix,
)
from kagglebot.validators import extract_data_archives

_PROFILE_MAX_TABLE_BYTES_DEFAULT = 256 * 1024 * 1024
_PROFILE_SAMPLE_ROWS = 200_000


def _dataset_slug(data_dir: Path) -> str:
    return str(data_dir.parent.name).strip().lower()


def _apply_dataset_profile_override(data_dir: Path, profile: dict[str, object]) -> dict[str, object]:
    return apply_competition_eval_override(slug=_dataset_slug(data_dir), payload=profile)


@dataclass(frozen=True)
class Taxonomy:
    tags: set[str]
    aliases: dict[str, str]

    def normalize(self, tags: Iterable[str]) -> list[str]:
        normalized: list[str] = []
        for tag in tags:
            key = self.aliases.get(tag, tag)
            if key in self.tags and key not in normalized:
                normalized.append(key)
        return normalized

    def to_dict(self) -> dict[str, object]:
        return {"tags": sorted(self.tags), "aliases": dict(self.aliases)}


def ensure_taxonomy(paths: KnowledgePaths) -> dict[str, object]:
    return TaxonomyRepository(paths).ensure()


def load_taxonomy(path: Path) -> dict[str, object]:
    return TaxonomyRepository.load(path)


def _parse_yaml_taxonomy(text: str) -> dict[str, object]:
    tags: set[str] = set()
    aliases: dict[str, str] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if " #" in line:
            line = line.split(" #", 1)[0].rstrip()
            stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0 and stripped.endswith(":"):
            current = stripped[:-1]
            continue
        if current is None:
            continue
        if current == "aliases":
            if ":" in stripped:
                alias_key, alias_val = stripped.split(":", 1)
                alias_key = alias_key.strip()
                alias_val = alias_val.strip().strip('"').strip("'")
                if alias_key and alias_val:
                    aliases[alias_key] = alias_val
            continue
        if current == "inference_rules":
            continue
        if stripped.startswith("-"):
            item = stripped[1:].strip().strip('"').strip("'")
            if item:
                tags.add(item)
    return {"tags": sorted(tags), "aliases": aliases}


def _taxonomy_from_dict(payload: dict[str, object]) -> Taxonomy:
    tags = set(payload.get("tags", []) or [])
    aliases = payload.get("aliases", {}) or {}
    return Taxonomy(tags=tags, aliases=dict(aliases))


def derive_problem_types(profile: dict[str, object]) -> list[str]:
    modality = str(profile.get("modality") or "").strip().lower()
    task = str(profile.get("task") or "").strip().lower()
    raw_tags = profile.get("tags", [])
    tags = [str(tag).strip().lower() for tag in raw_tags if isinstance(tag, str) and str(tag).strip()]

    problem_types: list[str] = []
    if modality and task:
        problem_types.append(f"{modality}:{task}")
    if modality:
        problem_types.append(modality)
    if task:
        problem_types.append(task)

    allowed_tags = {
        "tabular",
        "text",
        "image",
        "multimodal",
        "grouped",
        "sample_weighted",
        "timeseries",
        "geospatial",
        "rna_structure",
        "rna",
        "sequence",
        "structure",
        "survival",
        "pairwise",
        "learning_to_rank",
        "unsupervised",
        "unsupervised_prediction",
        "anomaly_detection",
        "quantile_regression",
        "prediction_interval",
        "count_regression",
        "bounded_regression",
        "positive_skew_regression",
        "ordinal_classification",
        "recommender",
        "ctr",
        "forecasting",
        "object_detection",
        "segmentation",
        "coordinate_regression",
        "residue_level_output",
        "regression",
        "binary",
        "multiclass",
        "multi_label",
        "multi_output",
        "multi_target",
        "multitask",
        "missingness_high",
        "high_cardinality_cats",
    }
    for tag in tags:
        if tag in allowed_tags:
            problem_types.append(tag)

    if not problem_types:
        return ["unknown"]

    seen: set[str] = set()
    ordered: list[str] = []
    for item in problem_types:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def resolve_problem_type_insights(
    knowledge_paths: KnowledgePaths,
    problem_types: Iterable[str],
    *,
    limit: int = 5,
) -> list[dict[str, object]]:
    _ensure_db(knowledge_paths)
    normalized = [str(item).strip().lower() for item in problem_types if str(item).strip()]
    if not normalized:
        return []
    with _connect(knowledge_paths.kb_path) as conn:
        placeholders = ",".join("?" for _ in normalized)
        rows = conn.execute(
            f"""
            SELECT
                slug,
                run_id,
                iter,
                problem_type,
                cause_category,
                fix_category,
                why_poor,
                how_improved,
                delta_offline,
                outcome_bucket,
                submission_score,
                created_at
            FROM problem_type_insights
            WHERE problem_type IN ({placeholders})
            ORDER BY
                CASE outcome_bucket WHEN 'good' THEN 0 WHEN 'low' THEN 1 ELSE 2 END,
                created_at DESC
            LIMIT ?
            """,
            [*normalized, limit],
        ).fetchall()
    return [dict(row) for row in rows]


def format_problem_type_insights(insights: list[dict[str, object]], *, limit: int = 5) -> str:
    if not insights:
        return "No prior problem-type insights available."
    lines = ["Problem-type knowledge (past failures and successful fixes):", ""]
    for item in insights[:limit]:
        problem_type = item.get("problem_type", "unknown")
        cause = item.get("cause_category", "unknown")
        fix = item.get("fix_category", "unknown")
        delta = item.get("delta_offline")
        outcome_bucket = str(item.get("outcome_bucket") or "unknown")
        submission_score = item.get("submission_score")
        slug = item.get("slug", "unknown")
        why = _shorten_text(str(item.get("why_poor") or ""), 220)
        how = _shorten_text(str(item.get("how_improved") or ""), 220)
        delta_text = "n/a" if delta is None else f"{float(delta):+.6f}"
        score_text = "n/a" if submission_score is None else f"{float(submission_score):.6f}"
        lines.append(
            f"- [{problem_type}] outcome={outcome_bucket} cause={cause} -> fix={fix} "
            f"(online={score_text}, delta={delta_text}, slug={slug})"
        )
        if why:
            lines.append(f"  why: {why}")
        if how:
            lines.append(f"  fix: {how}")
    return "\n".join(lines)


def record_problem_type_insight(
    *,
    knowledge_paths: KnowledgePaths,
    slug: str,
    run_id: str,
    iteration: int,
    problem_types: Iterable[str],
    why_poor: str,
    how_improved: str,
    delta_offline: float | None,
    outcome_bucket: str | None = None,
    submission_score: float | None = None,
) -> None:
    _ensure_db(knowledge_paths)
    types = [str(item).strip().lower() for item in problem_types if str(item).strip()]
    if not types:
        types = ["unknown"]

    cause_category = classify_cause_category(why_poor)
    fix_category = classify_fix_category(how_improved)
    if cause_category == "unknown":
        cause_category = classify_cause_category(f"{why_poor}\n{how_improved}")
    if fix_category == "unknown":
        fix_category = classify_fix_category(f"{how_improved}\n{why_poor}")
    normalized_outcome = _normalize_outcome_bucket(outcome_bucket)

    now = int(time.time())
    why_text = _shorten_text(why_poor, 2000)
    fix_text = _shorten_text(how_improved, 2000)
    with _connect(knowledge_paths.kb_path) as conn:
        for problem_type in types:
            conn.execute(
                """
                INSERT INTO problem_type_insights (
                    slug,
                    run_id,
                    iter,
                    problem_type,
                    cause_category,
                    fix_category,
                    why_poor,
                    how_improved,
                    delta_offline,
                    outcome_bucket,
                    submission_score,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, iter, problem_type) DO UPDATE SET
                    slug=excluded.slug,
                    cause_category=excluded.cause_category,
                    fix_category=excluded.fix_category,
                    why_poor=excluded.why_poor,
                    how_improved=excluded.how_improved,
                    delta_offline=excluded.delta_offline,
                    outcome_bucket=excluded.outcome_bucket,
                    submission_score=excluded.submission_score,
                    created_at=excluded.created_at
                """,
                (
                    slug,
                    run_id,
                    iteration,
                    problem_type,
                    cause_category,
                    fix_category,
                    why_text,
                    fix_text,
                    delta_offline,
                    normalized_outcome,
                    submission_score,
                    now,
                ),
            )


def resolve_error_fix_insights(
    knowledge_paths: KnowledgePaths,
    problem_types: Iterable[str],
    *,
    limit: int = 5,
) -> list[dict[str, object]]:
    _ensure_db(knowledge_paths)
    normalized = [str(item).strip().lower() for item in problem_types if str(item).strip()]
    if not normalized:
        return []
    with _connect(knowledge_paths.kb_path) as conn:
        placeholders = ",".join("?" for _ in normalized)
        rows = conn.execute(
            f"""
            SELECT
                slug,
                run_id,
                iter,
                problem_type,
                error_category,
                error_message,
                fix_summary,
                resolved,
                outcome_bucket,
                submission_score,
                created_at
            FROM error_fix_insights
            WHERE problem_type IN ({placeholders})
            ORDER BY
                CASE outcome_bucket WHEN 'good' THEN 0 WHEN 'low' THEN 1 ELSE 2 END,
                CASE resolved WHEN 1 THEN 0 ELSE 1 END,
                created_at DESC
            LIMIT ?
            """,
            [*normalized, limit],
        ).fetchall()
    return [dict(row) for row in rows]


def format_error_fix_insights(insights: list[dict[str, object]], *, limit: int = 5) -> str:
    if not insights:
        return "No prior error-fix insights available."
    lines = ["Error-fix knowledge (errors and how they were fixed):", ""]
    for item in insights[:limit]:
        problem_type = item.get("problem_type", "unknown")
        category = item.get("error_category", "unknown")
        resolved = bool(item.get("resolved"))
        outcome_bucket = str(item.get("outcome_bucket") or "unknown")
        submission_score = item.get("submission_score")
        error_text = _shorten_text(str(item.get("error_message") or ""), 180)
        fix_text = _shorten_text(str(item.get("fix_summary") or ""), 220)
        score_text = "n/a" if submission_score is None else f"{float(submission_score):.6f}"
        lines.append(
            f"- [{problem_type}] outcome={outcome_bucket} resolved={resolved} error={category} (online={score_text})"
        )
        if error_text:
            lines.append(f"  issue: {error_text}")
        if fix_text:
            lines.append(f"  fix: {fix_text}")
    return "\n".join(lines)


def record_error_fix_insight(
    *,
    knowledge_paths: KnowledgePaths,
    slug: str,
    run_id: str,
    iteration: int,
    problem_types: Iterable[str],
    error_message: str,
    fix_summary: str,
    resolved: bool,
    outcome_bucket: str | None = None,
    submission_score: float | None = None,
) -> None:
    _ensure_db(knowledge_paths)
    types = [str(item).strip().lower() for item in problem_types if str(item).strip()]
    if not types:
        types = ["unknown"]

    normalized_outcome = _normalize_outcome_bucket(outcome_bucket)
    category = classify_error_category(error_message)
    error_text = _shorten_text(error_message, 2000)
    fix_text = _shorten_text(fix_summary, 2000)
    fingerprint = hashlib.sha256(" ".join(error_text.split()).encode("utf-8")).hexdigest()[:16]
    now = int(time.time())
    with _connect(knowledge_paths.kb_path) as conn:
        for problem_type in types:
            conn.execute(
                """
                INSERT INTO error_fix_insights (
                    slug,
                    run_id,
                    iter,
                    problem_type,
                    error_fingerprint,
                    error_category,
                    error_message,
                    fix_summary,
                    resolved,
                    outcome_bucket,
                    submission_score,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, iter, problem_type, error_fingerprint) DO UPDATE SET
                    slug=excluded.slug,
                    error_category=excluded.error_category,
                    error_message=excluded.error_message,
                    fix_summary=excluded.fix_summary,
                    resolved=excluded.resolved,
                    outcome_bucket=excluded.outcome_bucket,
                    submission_score=excluded.submission_score,
                    created_at=excluded.created_at
                """,
                (
                    slug,
                    run_id,
                    iteration,
                    problem_type,
                    fingerprint,
                    category,
                    error_text,
                    fix_text,
                    int(resolved),
                    normalized_outcome,
                    submission_score,
                    now,
                ),
            )


def _shorten_text(text: str, max_chars: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _normalize_outcome_bucket(value: str | None) -> str:
    if not value:
        return "unknown"
    lowered = str(value).strip().lower()
    if lowered in {"good", "low", "unknown"}:
        return lowered
    return "unknown"


def build_dataset_profile(data_dir: Path) -> dict[str, object]:
    extract_data_archives(data_dir, overwrite=False)
    materialize_sqlite_tables(data_dir)
    profile: dict[str, object] = {"data_dir": str(data_dir)}
    file_count, extension_counts, file_samples = _summarize_files(data_dir)
    profile["file_count"] = file_count
    profile["file_extension_counts"] = dict(sorted(extension_counts.items()))
    profile["file_samples"] = file_samples

    tabular_files = _find_tabular_files(data_dir)
    if not tabular_files:
        modality = infer_asset_modality(data_dir, include_code_artifact=True)
        tags = [] if modality == "unknown" else [modality]
        profile["status"] = "non_tabular_data"
        profile["modality"] = modality
        profile["task"] = modality
        profile["metric"] = "unknown"
        profile["tags"] = tags
        return profile

    if detect_rna_structure_task(data_dir):
        try:
            task = load_rna_structure_task(data_dir)
        except Exception as exc:  # noqa: BLE001
            profile["status"] = "invalid_rna_structure_data"
            profile["error"] = str(exc)
            profile["tags"] = ["rna_structure", "rna", "sequence", "structure"]
            return profile
        profile.update(_build_rna_structure_profile(task))
        return _apply_dataset_profile_override(data_dir, profile)

    try:
        train_path, test_path, sample_path = find_competition_files(data_dir)
    except FileNotFoundError as exc:
        submission_only_profile = _build_submission_only_pairwise_profile(
            data_dir=data_dir,
            file_samples=file_samples,
            max_table_bytes=_profile_max_table_bytes(),
        )
        if submission_only_profile is not None:
            profile.update(submission_only_profile)
            return profile
        modality = infer_asset_modality(data_dir, include_code_artifact=True)
        if modality not in {"unknown", "tabular"}:
            tags = [] if modality == "unknown" else [modality]
            profile["status"] = "non_tabular_data"
            profile["modality"] = modality
            profile["task"] = modality
            profile["metric"] = "unknown"
            profile["tags"] = tags
            return profile
        profile["status"] = "missing_required_files"
        profile["error"] = str(exc)
        profile["tags"] = []
        return profile

    max_table_bytes = _profile_max_table_bytes()
    try:
        train, train_row_count, train_sampled = _read_table_for_profile(train_path, max_table_bytes=max_table_bytes)
        test, test_row_count, test_sampled = _read_table_for_profile(test_path, max_table_bytes=max_table_bytes)
        sample, _, sample_sampled = _read_table_for_profile(sample_path, max_table_bytes=max_table_bytes)
    except Exception as exc:  # noqa: BLE001
        profile["status"] = "unreadable_tabular"
        profile["error"] = str(exc)
        profile["tags"] = []
        return profile

    id_col, target_cols, feature_cols = infer_submission_layout(train=train, test=test, sample=sample)
    if not target_cols:
        unlabeled_profile = _build_unlabeled_prediction_profile(
            train=train,
            test=test,
            sample=sample,
            train_path=train_path,
            test_path=test_path,
            sample_path=sample_path,
            train_row_count=train_row_count,
            test_row_count=test_row_count,
            train_sampled=train_sampled,
            test_sampled=test_sampled,
            sample_sampled=sample_sampled,
            max_table_bytes=max_table_bytes,
            id_col=id_col,
            feature_cols=feature_cols,
            data_dir=data_dir,
        )
        if unlabeled_profile is not None:
            profile.update(unlabeled_profile)
            return _apply_dataset_profile_override(data_dir, profile)
        profile["status"] = "missing_target_columns"
        profile["tags"] = []
        return profile
    sample_weight_column_hint = _infer_sample_weight_column_hint(
        train=train,
        test=test,
        feature_cols=feature_cols,
        target_cols=target_cols,
        id_col=id_col,
    )
    if sample_weight_column_hint:
        feature_cols = [col for col in feature_cols if col != sample_weight_column_hint]
    target_col = target_cols[0]
    task_by_target = {
        col: "regression"
        if _looks_like_bounded_regression_target(train[col], column_name=col)
        or _looks_like_positive_skew_regression_target(train[col], column_name=col)
        else infer_task(train[col])
        for col in target_cols
    }
    prediction_kind_by_target = {
        col: _infer_profile_prediction_kind(sample=sample, id_col=id_col, target_col=col, target_cols=target_cols)
        for col in target_cols
    }
    task_by_target = {
        col: task_for_prediction_kind(task_by_target[col], prediction_kind_by_target[col]) for col in target_cols
    }
    unique_tasks = sorted(set(task_by_target.values()))
    task = unique_tasks[0] if len(unique_tasks) == 1 else "mixed"
    target_semantics_by_target = {
        col: _infer_target_semantics(train[col], column_name=col, task=task_by_target[col]) for col in target_cols
    }
    target_semantics = _aggregate_target_semantics(
        target_semantics_by_target=target_semantics_by_target,
        task=task,
        target_cols=target_cols,
    )
    if _looks_like_coordinate_regression_targets(
        train=train,
        target_cols=target_cols,
        task_by_target=task_by_target,
    ):
        target_semantics = "coordinate_regression"
        target_semantics_by_target = {col: "coordinate_regression" for col in target_cols}
    if _has_pairwise_feature_signal(feature_cols):
        target_semantics = "pairwise"
    elif _looks_like_multi_label_indicator_targets(
        train=train,
        sample=sample,
        target_cols=target_cols,
        task_by_target=task_by_target,
    ):
        target_semantics = "multi_label"
        target_semantics_by_target = {col: "multi_label" for col in target_cols}
    if _infer_learning_to_rank_target_semantics(
        feature_cols=feature_cols,
        target_col=target_col,
        target_semantics=target_semantics,
    ):
        target_semantics = "learning_to_rank"
        if len(target_cols) == 1:
            target_semantics_by_target[target_col] = "learning_to_rank"
    recommender_semantics = _infer_recommender_target_semantics(
        feature_cols=feature_cols,
        target_col=target_col,
        target_semantics=target_semantics,
        prediction_kind=prediction_kind_by_target[target_col],
    )
    if recommender_semantics is not None:
        target_semantics = recommender_semantics
        if len(target_cols) == 1:
            target_semantics_by_target[target_col] = recommender_semantics
    if _looks_like_forecasting_layout(train=train, test=test, feature_cols=feature_cols):
        target_semantics = "forecasting"
        if len(target_cols) == 1:
            target_semantics_by_target[target_col] = "forecasting"
    submission_semantics = _infer_submission_target_semantics(sample=sample, target_cols=target_cols)
    if submission_semantics is not None:
        target_semantics = submission_semantics
        if len(target_cols) == 1:
            target_semantics_by_target[target_cols[0]] = submission_semantics
    n_rows = train_row_count if train_row_count is not None else len(train)
    n_cols = len(train.columns)
    missingness = float(train.isna().mean().mean())
    missingness_by_column = {col: float(val) for col, val in train.isna().mean().items()}
    dtype_by_column = {col: str(dtype) for col, dtype in train.dtypes.items()}
    cat_cols = [c for c in feature_cols if train[c].dtype == "object"]
    high_cardinality = [c for c in cat_cols if safe_nunique(train[c]) > 50]

    modality = _infer_modality(data_dir, train, test, feature_cols=feature_cols)
    if prediction_kind_by_target[target_col] == "text" and target_semantics not in {"object_detection", "segmentation"}:
        modality = "text"
    group_column_hint = _infer_group_column_hint(train, feature_cols=feature_cols)
    task_tag = _task_tag(task, train[target_col])
    size_tag = _size_tag(n_rows)

    tags = []
    for tag in (modality, task_tag, size_tag):
        if tag not in tags:
            tags.append(tag)
    if group_column_hint and "grouped" not in tags:
        tags.append("grouped")
    if sample_weight_column_hint and "sample_weighted" not in tags:
        tags.append("sample_weighted")
    if missingness > 0.2:
        tags.append("missingness_high")
    if high_cardinality:
        tags.append("high_cardinality_cats")
    semantic_tag = _target_semantics_tag(target_semantics)
    if semantic_tag and semantic_tag not in tags:
        tags.append(semantic_tag)

    if target_semantics == "ctr":
        metric = "logloss"
    elif target_semantics == "learning_to_rank":
        metric = "ndcg"
    elif target_semantics == "recommender":
        metric = "rmse"
    elif target_semantics == "forecasting":
        metric = "rmse"
    elif target_semantics == "quantile_regression":
        metric = "pinball_loss"
    elif target_semantics == "prediction_interval":
        metric = "interval_score"
    elif target_semantics == "coordinate_regression":
        metric = "rmse"
    elif target_semantics == "count_regression":
        metric = "rmsle"
    elif target_semantics == "bounded_regression":
        metric = "rmse"
    elif target_semantics == "positive_skew_regression":
        metric = "rmsle"
    elif target_semantics == "ordinal_classification":
        metric = "quadratic_weighted_kappa"
    elif task == "regression":
        metric = "rmse"
    elif target_semantics == "survival":
        metric = "concordance_index"
    elif target_semantics == "object_detection":
        metric = "map"
    elif target_semantics == "segmentation":
        metric = "dice"
    elif target_semantics == "multi_label":
        metric = "f1"
    elif target_semantics == "text_generation":
        metric = "text_similarity"
    elif prediction_kind_by_target[target_col] in {"probability", "probability_columns"}:
        metric = "logloss"
    elif prediction_kind_by_target[target_col] == "text":
        metric = "text_similarity"
    else:
        metric = "accuracy"
    target_stats = _target_stats(train[target_col], task)

    train_only = [c for c in train.columns if c not in test.columns and c not in target_cols]
    test_only = [c for c in test.columns if c not in train.columns]

    profile.update(
        {
            "status": "ok",
            "train_file": train_path.name,
            "test_file": test_path.name,
            "sample_submission_file": sample_path.name,
            "train_rows": n_rows,
            "train_cols": n_cols,
            "test_rows": test_row_count if test_row_count is not None else len(test),
            "test_cols": len(test.columns),
            "id_column": id_col,
            "target_column": target_col,
            "target_columns": target_cols,
            "task": task,
            "task_by_target": task_by_target,
            "target_semantics": target_semantics,
            "target_semantics_by_target": target_semantics_by_target,
            "prediction_kind_by_target": prediction_kind_by_target,
            "metric": metric,
            "missingness": missingness,
            "missingness_by_column": missingness_by_column,
            "dtype_by_column": dtype_by_column,
            "categorical_columns": cat_cols,
            "numeric_columns": [c for c in feature_cols if c not in cat_cols],
            "high_cardinality_columns": high_cardinality,
            "modality": modality,
            "tags": tags,
            "target_stats": target_stats,
            "train_only_columns": train_only,
            "test_only_columns": test_only,
            "profile_sampling": {
                "enabled": bool(train_sampled or test_sampled or sample_sampled),
                "max_table_bytes": max_table_bytes,
                "max_rows": _PROFILE_SAMPLE_ROWS,
                "train": train_sampled,
                "test": test_sampled,
                "sample_submission": sample_sampled,
            },
        }
    )
    if modality == "timeseries":
        profile["split_strategy_hint"] = "timeseries_split"
    elif group_column_hint:
        profile["split_strategy_hint"] = "group_kfold"
    if group_column_hint:
        profile["group_column_hint"] = group_column_hint
    if sample_weight_column_hint:
        profile["sample_weight_column_hint"] = sample_weight_column_hint
        profile["sample_weight_summary"] = _sample_weight_summary(train[sample_weight_column_hint])
    return _apply_dataset_profile_override(data_dir, profile)


def _infer_profile_prediction_kind(
    *,
    sample: pd.DataFrame,
    id_col: str | None,
    target_col: str,
    target_cols: list[str],
) -> str:
    if target_col in sample.columns:
        return infer_prediction_kind(sample[target_col], column_name=target_col)
    prediction_cols = [col for col in sample.columns if col != id_col]
    if (
        len(target_cols) == 1
        and len(prediction_cols) >= 2
        and all(pd.api.types.is_numeric_dtype(sample[col]) for col in prediction_cols)
    ):
        return "probability_columns"
    return "continuous"


def _build_unlabeled_prediction_profile(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    train_path: Path,
    test_path: Path,
    sample_path: Path,
    train_row_count: int | None,
    test_row_count: int | None,
    train_sampled: bool,
    test_sampled: bool,
    sample_sampled: bool,
    max_table_bytes: int,
    id_col: str | None,
    feature_cols: list[str],
    data_dir: Path,
) -> dict[str, object] | None:
    prediction_cols = [str(col) for col in sample.columns if str(col) != str(id_col)]
    if not prediction_cols:
        return None
    if not feature_cols:
        return None

    target_col = prediction_cols[0]
    target_semantics = (
        "anomaly_detection" if _looks_like_anomaly_prediction_column(target_col) else "unsupervised_prediction"
    )
    prediction_kind_by_target = {
        col: _infer_profile_prediction_kind(sample=sample, id_col=id_col, target_col=col, target_cols=prediction_cols)
        for col in prediction_cols
    }
    n_rows = train_row_count if train_row_count is not None else len(train)
    missingness = float(train.isna().mean().mean())
    missingness_by_column = {col: float(val) for col, val in train.isna().mean().items()}
    dtype_by_column = {col: str(dtype) for col, dtype in train.dtypes.items()}
    cat_cols = [c for c in feature_cols if c in train.columns and train[c].dtype == "object"]
    high_cardinality = [c for c in cat_cols if safe_nunique(train[c]) > 50]
    modality = _infer_modality(data_dir, train, test, feature_cols=feature_cols)
    tags = [modality, "unsupervised", _target_semantics_tag(target_semantics)]
    if missingness > 0.2:
        tags.append("missingness_high")
    if high_cardinality:
        tags.append("high_cardinality_cats")
    deduped_tags = [tag for index, tag in enumerate(tags) if tag and tag not in tags[:index]]
    metric = "auc" if target_semantics == "anomaly_detection" else "unknown"
    return {
        "status": "ok",
        "train_file": train_path.name,
        "test_file": test_path.name,
        "sample_submission_file": sample_path.name,
        "train_rows": n_rows,
        "train_cols": len(train.columns),
        "test_rows": test_row_count if test_row_count is not None else len(test),
        "test_cols": len(test.columns),
        "id_column": id_col,
        "target_column": target_col,
        "target_columns": prediction_cols,
        "task": "unsupervised",
        "task_by_target": {col: "unsupervised" for col in prediction_cols},
        "target_semantics": target_semantics,
        "target_semantics_by_target": {col: target_semantics for col in prediction_cols},
        "prediction_kind_by_target": prediction_kind_by_target,
        "metric": metric,
        "missingness": missingness,
        "missingness_by_column": missingness_by_column,
        "dtype_by_column": dtype_by_column,
        "categorical_columns": cat_cols,
        "numeric_columns": [c for c in feature_cols if c in train.columns and c not in cat_cols],
        "high_cardinality_columns": high_cardinality,
        "modality": modality,
        "tags": deduped_tags,
        "target_stats": {},
        "train_only_columns": [c for c in train.columns if c not in test.columns],
        "test_only_columns": [c for c in test.columns if c not in train.columns],
        "profile_sampling": {
            "enabled": bool(train_sampled or test_sampled or sample_sampled),
            "max_table_bytes": max_table_bytes,
            "max_rows": _PROFILE_SAMPLE_ROWS,
            "train": train_sampled,
            "test": test_sampled,
            "sample_submission": sample_sampled,
        },
    }


def _build_rna_structure_profile(task) -> dict[str, object]:
    train_sequences = task.train_sequences
    test_sequences = task.test_sequences
    train_labels = task.train_labels
    sample = task.sample_submission
    sequence_lengths = train_sequences[task.sequence_column].astype(str).str.len()
    sample_target_ids = sample[task.sample_id_column].astype(str).map(extract_target_id)
    train_target_ids = train_labels[task.label_id_column].astype(str).map(extract_target_id)

    missingness_by_column = {col: float(val) for col, val in train_sequences.isna().mean().items()}
    dtype_by_column = {col: str(dtype) for col, dtype in train_sequences.dtypes.items()}
    categorical_columns = [
        str(col)
        for col in train_sequences.columns
        if train_sequences[col].dtype == "object" and str(col) not in {task.sequence_column, task.sequence_id_column}
    ]
    numeric_columns = [
        str(col)
        for col in train_sequences.columns
        if pd.api.types.is_numeric_dtype(train_sequences[col]) and str(col) != task.sequence_column
    ]
    target_columns = list(task.sample_coordinate_columns)
    target_column = target_columns[0]
    coordinate_triplets = [triplet.copy_index for triplet in task.sample_coordinate_triplets]
    return {
        "status": "ok",
        "train_file": task.files.train_sequences_path.name,
        "test_file": task.files.test_sequences_path.name,
        "labels_file": task.files.train_labels_path.name,
        "sample_submission_file": task.files.sample_submission_path.name,
        "train_rows": len(train_sequences),
        "train_cols": len(train_sequences.columns),
        "test_rows": len(test_sequences),
        "test_cols": len(test_sequences.columns),
        "id_column": task.sample_id_column,
        "sequence_id_column": task.sequence_id_column,
        "target_column": target_column,
        "target_columns": target_columns,
        "task": "regression",
        "task_by_target": {column: "regression" for column in target_columns},
        "prediction_kind_by_target": {column: "continuous" for column in target_columns},
        "metric": "coord_rmse",
        "missingness": float(train_sequences.isna().mean().mean()),
        "missingness_by_column": missingness_by_column,
        "dtype_by_column": dtype_by_column,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "high_cardinality_columns": [],
        "modality": "rna_structure",
        "target_kind": task.target_kind,
        "sample_anchor_columns": list(task.sample_anchor_columns),
        "sequence_column": task.sequence_column,
        "sequence_length_stats": {
            "min": int(sequence_lengths.min()) if not sequence_lengths.empty else 0,
            "median": float(sequence_lengths.median()) if not sequence_lengths.empty else 0.0,
            "max": int(sequence_lengths.max()) if not sequence_lengths.empty else 0,
        },
        "residue_rows_train": int(len(train_labels)),
        "residue_rows_test": int(len(sample)),
        "coordinate_triplets": coordinate_triplets,
        "tags": [
            "rna_structure",
            "rna",
            "sequence",
            "structure",
            "coordinate_regression",
            "residue_level_output",
            _size_tag(len(train_sequences)),
        ],
        "target_stats": {
            "n_targets_train": int(train_target_ids.nunique()),
            "n_targets_test": int(sample_target_ids.nunique()),
            "n_coordinate_triplets": len(task.sample_coordinate_triplets),
        },
        "train_only_columns": [str(col) for col in train_sequences.columns if col not in test_sequences.columns],
        "test_only_columns": [str(col) for col in test_sequences.columns if col not in train_sequences.columns],
        "profile_sampling": {
            "enabled": False,
            "max_table_bytes": _profile_max_table_bytes(),
            "max_rows": _PROFILE_SAMPLE_ROWS,
            "train": False,
            "test": False,
            "sample_submission": False,
        },
    }


def _build_submission_only_pairwise_profile(
    *,
    data_dir: Path,
    file_samples: list[str],
    max_table_bytes: int,
) -> dict[str, object] | None:
    sample_path = ensure_sample_submission(data_dir)
    if sample_path is None:
        return None

    try:
        sample, sample_row_count, sample_sampled = _read_table_for_profile(sample_path, max_table_bytes=max_table_bytes)
    except Exception:  # noqa: BLE001
        return None

    if sample.empty or len(sample.columns) < 2:
        return None

    id_col = str(sample.columns[0])
    if not _is_id_like_column(id_col):
        return None
    target_cols = [str(col) for col in sample.columns[1:] if str(col).strip()]
    if not target_cols:
        return None
    target_col = target_cols[0]
    prediction_kind = infer_prediction_kind(sample[target_col])
    if prediction_kind not in {"probability", "continuous"}:
        return None

    file_names = {path.name for path in _find_tabular_files(data_dir)}
    looks_like_march_mania = _looks_like_march_mania_pairwise_competition(
        file_names=file_names,
        sample_path=sample_path,
        sample=sample,
        data_dir=data_dir,
    )
    if not looks_like_march_mania:
        return None

    return {
        "status": "ok",
        "sample_submission_file": sample_path.name,
        "train_rows": None,
        "train_cols": None,
        "test_rows": sample_row_count if sample_row_count is not None else len(sample),
        "test_cols": len(sample.columns),
        "id_column": id_col,
        "target_column": target_col,
        "target_columns": target_cols,
        "task": "classification",
        "task_by_target": {col: "classification" for col in target_cols},
        "target_semantics": "pairwise",
        "target_semantics_by_target": {col: "pairwise" for col in target_cols},
        "prediction_kind_by_target": {col: prediction_kind for col in target_cols},
        "metric": "brier_score",
        "missingness": 0.0,
        "missingness_by_column": {col: float(val) for col, val in sample.isna().mean().items()},
        "dtype_by_column": {col: str(dtype) for col, dtype in sample.dtypes.items()},
        "categorical_columns": [],
        "numeric_columns": [],
        "high_cardinality_columns": [],
        "modality": "tabular",
        "tags": ["tabular", "binary", "pairwise"],
        "competition_structure": "submission_only_pairwise_probability",
        "split_strategy_hint": "group_kfold",
        "group_column_hint": "Season",
        "note": (
            "Competition does not ship canonical train/test tables. "
            "Build a pregame matchup table from season history and submit pairwise win probabilities."
        ),
        "profile_sampling": {
            "enabled": bool(sample_sampled),
            "max_table_bytes": max_table_bytes,
            "max_rows": _PROFILE_SAMPLE_ROWS,
            "train": False,
            "test": False,
            "sample_submission": sample_sampled,
        },
        "file_samples": file_samples,
    }


def _looks_like_march_mania_pairwise_competition(
    *,
    file_names: set[str],
    sample_path: Path,
    sample: pd.DataFrame,
    data_dir: Path,
) -> bool:
    required_any = {
        "MRegularSeasonCompactResults.csv",
        "WRegularSeasonCompactResults.csv",
        "MNCAATourneyCompactResults.csv",
        "WNCAATourneyCompactResults.csv",
    }
    if not required_any.issubset(file_names):
        return False
    if tuple(str(col) for col in sample.columns[:2]) != ("ID", "Pred"):
        return False
    first_id = str(sample.iloc[0]["ID"]) if len(sample) else ""
    if not re.fullmatch(r"\d{4}_\d+_\d+", first_id):
        return False
    if sample_name_score(sample_path) >= 2:
        return True
    return "march-machine-learning-mania" in str(data_dir).lower()


def _is_id_like_column(column: object) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
    compact = normalized.replace("_", "")
    return normalized in ID_LIKE_COLUMN_NAMES or compact in ID_LIKE_COLUMN_NAMES


def _profile_max_table_bytes() -> int:
    value = parse_int_value(os.getenv("KAGGLEBOT_PROFILE_MAX_TABLE_BYTES"))
    if value is None:
        return _PROFILE_MAX_TABLE_BYTES_DEFAULT
    if value <= 0:
        return _PROFILE_MAX_TABLE_BYTES_DEFAULT
    return value


def _read_table_for_profile(path: Path, *, max_table_bytes: int) -> tuple[pd.DataFrame, int | None, bool]:
    size_bytes = _safe_file_size(path)
    suffix = tabular_suffix(path)
    oversized = size_bytes is not None and size_bytes > max_table_bytes
    if oversized and suffix in TABULAR_TEXT_SUFFIXES | TABULAR_STRUCTURED_SUFFIXES:
        if _is_json_lines_suffix(suffix):
            frame = _read_table(path, nrows=_PROFILE_SAMPLE_ROWS)
            row_count = _count_text_rows(path, has_header=False)
            return frame, row_count, True
        frame = _read_table(path, nrows=_PROFILE_SAMPLE_ROWS)
        row_count = _count_text_rows(path, has_header=True) if suffix in TABULAR_TEXT_SUFFIXES else len(frame)
        return frame, row_count, True
    frame = _read_table(path)
    return frame, len(frame), False


def _is_json_lines_suffix(suffix: str) -> bool:
    return is_json_lines_tabular_suffix(suffix)


def _safe_file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _count_text_rows(path: Path, *, has_header: bool) -> int | None:
    rows = 0
    try:
        with open_compressed_text(path, suffix=tabular_suffix(path), encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.strip():
                    rows += 1
    except OSError:
        return None
    if has_header and rows > 0:
        rows -= 1
    return rows


def _find_tabular_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and is_tabular_data_path(p)]


def _read_table(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    return read_table(path, nrows=nrows)


def _summarize_files(
    data_dir: Path,
    *,
    max_files: int = 20000,
    max_samples: int = 30,
) -> tuple[int, dict[str, int], list[str]]:
    counts: dict[str, int] = {}
    samples: list[str] = []
    total = 0
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        total += 1
        ext = tabular_suffix(path) or "<none>"
        counts[ext] = counts.get(ext, 0) + 1
        if len(samples) < max_samples:
            try:
                samples.append(str(path.relative_to(data_dir)))
            except ValueError:
                samples.append(path.name)
        if total >= max_files:
            break
    return total, counts, samples


def _target_stats(target: pd.Series, task: str) -> dict[str, object]:
    stats: dict[str, object] = {}
    if task == "regression":
        values = pd.to_numeric(target, errors="coerce").dropna()
        if not values.empty:
            stats["min"] = float(values.min())
            stats["max"] = float(values.max())
            stats["mean"] = float(values.mean())
            stats["std"] = float(values.std(ddof=0))
            skew = float(values.skew())
            stats["skew"] = None if pd.isna(skew) else skew
    else:
        counts = target.value_counts(dropna=False)
        stats["unique"] = int(counts.size)
        if counts.sum() > 0:
            stats["top_class_ratio"] = float(counts.iloc[0] / counts.sum())
    return stats


def _infer_target_semantics(target: pd.Series, *, column_name: str, task: str) -> str:
    if task == "text" and _is_text_generation_target_column(column_name):
        return "text_generation"
    if task == "regression" and _looks_like_count_regression_target(target, column_name=column_name):
        return "count_regression"
    if task == "regression" and _looks_like_bounded_regression_target(target, column_name=column_name):
        return "bounded_regression"
    if task == "regression" and _looks_like_positive_skew_regression_target(target, column_name=column_name):
        return "positive_skew_regression"
    if task == "classification" and _looks_like_multi_label_target(target, column_name=column_name):
        return "multi_label"
    if task == "classification" and _looks_like_ordinal_target(target, column_name=column_name):
        return "ordinal_classification"
    return task


def _is_text_generation_target_column(name: str) -> bool:
    compact = re.sub(r"[^a-z0-9]+", "", str(name).lower())
    tokens = set(_column_tokens(name))
    return compact in {
        "answer",
        "answers",
        "caption",
        "captions",
        "completion",
        "completions",
        "description",
        "descriptions",
        "essay",
        "essays",
        "explanation",
        "explanations",
        "generatedtext",
        "response",
        "responses",
        "summary",
        "summaries",
        "targettext",
        "textanswer",
        "transcription",
        "transcriptions",
        "translation",
        "translations",
    } or bool(
        tokens
        & {
            "answer",
            "caption",
            "completion",
            "description",
            "essay",
            "explanation",
            "response",
            "summary",
            "transcription",
            "translation",
        }
    )


def _aggregate_target_semantics(
    *,
    target_semantics_by_target: dict[str, str],
    task: str,
    target_cols: list[str],
) -> str:
    if len(target_cols) <= 1:
        return target_semantics_by_target[target_cols[0]]
    if _looks_like_survival_target_columns(target_cols):
        return "survival"
    semantics = set(target_semantics_by_target.values())
    if "multi_label" in semantics:
        return "multi_label"
    if task == "regression":
        return "multi_output_regression"
    if task == "classification":
        return "multi_target_classification"
    if task == "mixed":
        return "multi_task"
    return f"multi_output_{task}"


def _looks_like_coordinate_regression_targets(
    *,
    train: pd.DataFrame,
    target_cols: list[str],
    task_by_target: dict[str, str],
) -> bool:
    if len(target_cols) < 2:
        return False
    if not all(col in train.columns for col in target_cols):
        return False
    if not all(task_by_target.get(col) == "regression" for col in target_cols):
        return False
    if not all(pd.api.types.is_numeric_dtype(train[col]) for col in target_cols):
        return False
    axes = {_coordinate_axis_for_column(str(col)) for col in target_cols}
    axes.discard(None)
    if {"lat", "lon"}.issubset(axes):
        return True
    return {"x", "y"}.issubset(axes)


def _looks_like_multi_label_indicator_targets(
    *,
    train: pd.DataFrame,
    sample: pd.DataFrame,
    target_cols: list[str],
    task_by_target: dict[str, str],
) -> bool:
    if len(target_cols) < 3:
        return False
    if _looks_like_survival_target_columns(target_cols):
        return False
    if not all(col in train.columns and col in sample.columns for col in target_cols):
        return False
    if not all(task_by_target.get(col) == "classification" for col in target_cols):
        return False
    if not all(_looks_like_binary_indicator_column(train[col]) for col in target_cols):
        return False
    return all(pd.api.types.is_numeric_dtype(sample[col]) for col in target_cols)


def _looks_like_binary_indicator_column(series: pd.Series) -> bool:
    non_null = series.dropna()
    values = pd.to_numeric(non_null, errors="coerce").dropna()
    if values.empty or len(values) != len(non_null):
        return False
    unique = set(values.unique().tolist())
    return bool(unique) and unique <= {0, 1, 0.0, 1.0}


def _looks_like_count_regression_target(target: pd.Series, *, column_name: str) -> bool:
    if not pd.api.types.is_numeric_dtype(target):
        return False
    tokens = set(_column_tokens(column_name))
    compact = "".join(_column_tokens(column_name))
    if not (
        tokens
        & {
            "count",
            "counts",
            "demand",
            "quantity",
            "qty",
            "unit",
            "units",
            "trip",
            "trips",
            "ride",
            "rides",
            "rental",
            "rentals",
            "order",
            "orders",
            "booking",
            "bookings",
            "visitor",
            "visitors",
            "passenger",
            "passengers",
        }
        or compact in {"itemcount", "unitcount", "numorders", "numberoforders", "tripcount", "ridecount"}
        or compact.startswith("num")
    ):
        return False
    values = pd.to_numeric(target.dropna(), errors="coerce").dropna()
    if values.empty or bool((values < 0).any()):
        return False
    integer_like = ((values % 1).abs() < 1e-9).mean()
    return bool(float(integer_like) >= 0.95 and int(values.nunique(dropna=True)) >= 3)


def _looks_like_bounded_regression_target(target: pd.Series, *, column_name: str) -> bool:
    if not pd.api.types.is_numeric_dtype(target):
        return False
    tokens = set(_column_tokens(column_name))
    compact = "".join(_column_tokens(column_name))
    bounded_names = {
        "rate",
        "ratio",
        "percent",
        "percentage",
        "pct",
        "share",
        "fraction",
        "proportion",
        "probability",
        "prob",
    }
    bounded_compacts = {
        "conversionrate",
        "clickthroughrate",
        "defaultprobability",
        "winprobability",
        "targetrate",
        "targetratio",
    }
    if not (tokens & bounded_names or compact in bounded_compacts):
        return False
    values = pd.to_numeric(target.dropna(), errors="coerce").dropna()
    if values.empty or int(values.nunique(dropna=True)) < 3:
        return False
    if float(values.min()) < 0.0:
        return False
    max_value = float(values.max())
    if max_value <= 1.0:
        return True
    percent_names = {"percent", "percentage", "pct"}
    return bool((tokens & percent_names or "percent" in compact or "pct" in compact) and max_value <= 100.0)


def _looks_like_positive_skew_regression_target(target: pd.Series, *, column_name: str) -> bool:
    if not pd.api.types.is_numeric_dtype(target):
        return False
    tokens = set(_column_tokens(column_name))
    compact = "".join(_column_tokens(column_name))
    skew_names = {
        "amount",
        "cost",
        "fare",
        "income",
        "price",
        "profit",
        "revenue",
        "sale",
        "sales",
        "spend",
        "value",
    }
    skew_compacts = {
        "saleprice",
        "salesprice",
        "transactionamount",
        "purchaseamount",
        "targetvalue",
    }
    if not (tokens & skew_names or compact in skew_compacts):
        return False
    values = pd.to_numeric(target.dropna(), errors="coerce").dropna()
    if len(values) < 8 or bool((values < 0).any()) or int(values.nunique(dropna=True)) < 5:
        return False
    median = float(values.median())
    if median <= 0.0:
        return False
    skew = float(values.skew())
    if pd.isna(skew):
        return False
    return bool(skew >= 1.0 and float(values.max()) / median >= 5.0)


def _coordinate_axis_for_column(name: str) -> str | None:
    tokens = _column_tokens(name)
    compact = "".join(tokens)
    if not tokens:
        return None
    if tokens[0] in {"x", "y", "z"}:
        return tokens[0]
    if tokens[-1] in {"x", "y", "z"} and any(
        token in {"coord", "coordinate", "coords", "position", "pos"} for token in tokens
    ):
        return tokens[-1]
    if compact in {"x", "xcoord", "xcoordinate", "coordx", "coordinatex", "positionx", "posx"}:
        return "x"
    if compact in {"y", "ycoord", "ycoordinate", "coordy", "coordinatey", "positiony", "posy"}:
        return "y"
    if compact in {"z", "zcoord", "zcoordinate", "coordz", "coordinatez", "positionz", "posz"}:
        return "z"
    if "latitude" in tokens or "lat" in tokens or compact in {"latitude", "lat"}:
        return "lat"
    if any(token in {"longitude", "lon", "lng"} for token in tokens) or compact in {"longitude", "lon", "lng"}:
        return "lon"
    return None


def _target_semantics_tag(target_semantics: str) -> str | None:
    if target_semantics == "multi_label":
        return "multi_label"
    if target_semantics.startswith("multi_output"):
        return "multi_output"
    if target_semantics.startswith("multi_target"):
        return "multi_target"
    if target_semantics == "multi_task":
        return "multitask"
    if target_semantics == "survival":
        return "survival"
    if target_semantics == "pairwise":
        return "pairwise"
    if target_semantics == "learning_to_rank":
        return "learning_to_rank"
    if target_semantics == "anomaly_detection":
        return "anomaly_detection"
    if target_semantics == "unsupervised_prediction":
        return "unsupervised"
    if target_semantics == "quantile_regression":
        return "quantile_regression"
    if target_semantics == "prediction_interval":
        return "prediction_interval"
    if target_semantics == "coordinate_regression":
        return "coordinate_regression"
    if target_semantics == "count_regression":
        return "count_regression"
    if target_semantics == "bounded_regression":
        return "bounded_regression"
    if target_semantics == "positive_skew_regression":
        return "positive_skew_regression"
    if target_semantics == "ordinal_classification":
        return "ordinal_classification"
    if target_semantics == "recommender":
        return "recommender"
    if target_semantics == "ctr":
        return "ctr"
    if target_semantics == "forecasting":
        return "forecasting"
    if target_semantics == "text_generation":
        return "text_generation"
    if target_semantics == "object_detection":
        return "object_detection"
    if target_semantics == "segmentation":
        return "segmentation"
    return None


def _infer_submission_target_semantics(*, sample: pd.DataFrame, target_cols: list[str]) -> str | None:
    prediction_cols = [str(col) for col in sample.columns if str(col) in {str(target) for target in target_cols}]
    if not prediction_cols:
        prediction_cols = [str(col) for col in sample.columns[1:]]
    if _looks_like_prediction_interval_columns(prediction_cols):
        return "prediction_interval"
    if _looks_like_quantile_prediction_columns(prediction_cols):
        return "quantile_regression"
    compact_cols = {re.sub(r"[^a-z0-9]+", "", col.lower()) for col in prediction_cols}
    if compact_cols & {
        "encodedpixels",
        "rle",
        "runlengthencoding",
        "mask",
        "masks",
        "segmentation",
        "segmentationmask",
        "maskrle",
    }:
        return "segmentation"
    if compact_cols & {"predictionstring", "predstring", "detections", "detectionstring", "bbox", "bboxes", "boxes"}:
        return "object_detection"
    return None


def _looks_like_prediction_interval_columns(prediction_cols: list[str]) -> bool:
    compact_cols = {re.sub(r"[^a-z0-9]+", "", col.lower()) for col in prediction_cols}
    lower_tokens = {"lower", "lo", "low", "lwr", "lowerbound", "lowerci", "lowerlimit"}
    upper_tokens = {"upper", "hi", "high", "upr", "upperbound", "upperci", "upperlimit"}
    has_lower = bool(compact_cols & lower_tokens)
    has_upper = bool(compact_cols & upper_tokens)
    return has_lower and has_upper


def _looks_like_quantile_prediction_columns(prediction_cols: list[str]) -> bool:
    quantile_count = sum(1 for col in prediction_cols if _is_quantile_prediction_column(col))
    return quantile_count >= 2


def _is_quantile_prediction_column(name: str) -> bool:
    lower = name.lower().strip()
    compact = re.sub(r"[^a-z0-9]+", "", lower)
    if compact in {"median", "p50", "q50", "quantile50"}:
        return True
    if re.search(r"(?:^|[_\-.])(?:p|q)(?:0?[1-9]|[1-9][0-9])(?:$|[_\-.])", lower):
        return True
    if re.search(r"(?:quantile|percentile)[_\-.]?(?:0?\.\d+|0?[1-9]|[1-9][0-9])", lower):
        return True
    return bool(re.search(r"(?:^|[_\-.])0?\.\d+(?:$|[_\-.])", lower))


def _has_pairwise_feature_signal(feature_cols: list[str]) -> bool:
    compact_cols = {re.sub(r"[^a-z0-9]+", "", str(col).lower()) for col in feature_cols}
    pair_groups = (
        ("team1", "team2"),
        ("team_a", "team_b"),
        ("teama", "teamb"),
        ("home_team", "away_team"),
        ("hometeam", "awayteam"),
        ("player1", "player2"),
        ("playera", "playerb"),
        ("item1", "item2"),
        ("itema", "itemb"),
        ("modela", "modelb"),
        ("model_a", "model_b"),
        ("user1", "user2"),
        ("usera", "userb"),
    )
    for left, right in pair_groups:
        left_compact = re.sub(r"[^a-z0-9]+", "", left)
        right_compact = re.sub(r"[^a-z0-9]+", "", right)
        if left_compact in compact_cols and right_compact in compact_cols:
            return True
    prefixes: dict[str, set[str]] = {}
    for compact in compact_cols:
        match = re.match(r"(.+?)(?:id)?([12ab])$", compact)
        if not match:
            continue
        prefix, side = match.groups()
        if prefix in {"team", "player", "item", "model", "user", "entity", "candidate"}:
            prefixes.setdefault(prefix, set()).add(side)
    return any({"1", "2"}.issubset(sides) or {"a", "b"}.issubset(sides) for sides in prefixes.values())


def _infer_recommender_target_semantics(
    *,
    feature_cols: list[str],
    target_col: str,
    target_semantics: str,
    prediction_kind: str,
) -> str | None:
    if target_semantics not in {"classification", "regression", "ordinal_classification", "bounded_regression"}:
        return None
    if not _has_user_item_feature_signal(feature_cols):
        return None
    if _is_ctr_target_column(target_col):
        return "ctr"
    if target_semantics == "bounded_regression":
        return None
    if _is_recommender_score_target_column(target_col):
        return "recommender"
    if target_semantics == "classification" or prediction_kind == "probability":
        return "ctr"
    return "recommender"


def _infer_learning_to_rank_target_semantics(
    *,
    feature_cols: list[str],
    target_col: str,
    target_semantics: str,
) -> bool:
    if target_semantics not in {"classification", "regression"}:
        return False
    if not _has_query_candidate_feature_signal(feature_cols):
        return False
    return _is_learning_to_rank_target_column(target_col)


def _has_query_candidate_feature_signal(feature_cols: list[str]) -> bool:
    has_query = any(_is_query_entity_column(str(col)) for col in feature_cols)
    has_candidate = any(_is_ranking_candidate_column(str(col)) for col in feature_cols)
    return has_query and has_candidate


def _is_query_entity_column(name: str) -> bool:
    return _column_matches_entity_terms(
        name,
        {
            "query",
            "search",
            "request",
            "question",
            "prompt",
            "topic",
            "keyword",
            "qid",
        },
    )


def _is_ranking_candidate_column(name: str) -> bool:
    return _column_matches_entity_terms(
        name,
        {
            "document",
            "doc",
            "passage",
            "candidate",
            "result",
            "page",
            "url",
            "answer",
            "listing",
            "item",
        },
    )


def _is_learning_to_rank_target_column(name: str) -> bool:
    tokens = set(_column_tokens(name))
    compact = "".join(tokens)
    return bool(tokens & {"relevance", "rank", "ranking", "grade", "gain"}) or compact in {
        "relevancescore",
        "relevancegrade",
        "searchrank",
        "target",
        "label",
    }


def _looks_like_anomaly_prediction_column(name: str) -> bool:
    tokens = set(_column_tokens(name))
    compact = "".join(tokens)
    return bool(
        tokens
        & {
            "anomaly",
            "outlier",
            "fraud",
            "attack",
            "intrusion",
            "defect",
            "failure",
            "fault",
            "risk",
        }
    ) or compact in {"anomalyscore", "outlierscore", "fraudscore", "isfraud", "isanomaly"}


def _has_user_item_feature_signal(feature_cols: list[str]) -> bool:
    has_user = any(_is_user_entity_column(str(col)) for col in feature_cols)
    has_item = any(_is_item_entity_column(str(col)) for col in feature_cols)
    return has_user and has_item


def _is_user_entity_column(name: str) -> bool:
    return _column_matches_entity_terms(
        name,
        {
            "user",
            "customer",
            "client",
            "member",
            "account",
            "person",
            "profile",
            "visitor",
            "session",
        },
    )


def _is_item_entity_column(name: str) -> bool:
    return _column_matches_entity_terms(
        name,
        {
            "item",
            "product",
            "sku",
            "listing",
            "ad",
            "creative",
            "campaign",
            "content",
            "article",
            "movie",
            "book",
            "game",
            "song",
            "track",
            "merchant",
            "restaurant",
            "coupon",
        },
    )


def _column_matches_entity_terms(name: str, terms: set[str]) -> bool:
    tokens = _column_tokens(name)
    compact = "".join(tokens)
    if any(token in terms for token in tokens):
        return True
    id_suffixes = ("id", "idx", "uuid", "code", "key")
    return any(compact.startswith(term) and compact.endswith(id_suffixes) for term in terms)


def _is_ctr_target_column(name: str) -> bool:
    tokens = set(_column_tokens(name))
    compact = "".join(tokens)
    return bool(
        tokens
        & {
            "click",
            "clicked",
            "clicks",
            "ctr",
            "conversion",
            "converted",
            "purchase",
            "purchased",
            "booked",
            "install",
            "installed",
            "opened",
        }
    ) or compact in {"isclick", "isclicked", "hasclicked", "clickthroughrate", "target"}


def _is_recommender_score_target_column(name: str) -> bool:
    tokens = set(_column_tokens(name))
    compact = "".join(tokens)
    return bool(tokens & {"rating", "ratings", "score", "stars", "relevance", "preference"}) or compact in {
        "reviewscore",
        "userscore",
        "itemscore",
    }


def _looks_like_forecasting_layout(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> bool:
    temporal_cols = [
        col
        for col in feature_cols
        if col in train.columns and col in test.columns and _column_name_has_temporal_token(str(col))
    ]
    if not temporal_cols:
        return False
    return any(_has_future_temporal_holdout(train[col], test[col]) for col in temporal_cols)


def _has_future_temporal_holdout(train_series: pd.Series, test_series: pd.Series) -> bool:
    if _has_future_ordinal_holdout(train_series, test_series):
        return True
    train_dates = _parse_temporal_series(train_series)
    test_dates = _parse_temporal_series(test_series)
    if train_dates.empty or test_dates.empty:
        return False
    return bool(test_dates.max() > train_dates.max())


def _parse_temporal_series(series: pd.Series) -> pd.Series:
    sample = series.dropna().astype(str).head(500)
    if sample.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")
    parsed = pd.to_datetime(sample, errors="coerce", utc=True, format="mixed")
    if float(parsed.notna().mean()) < 0.8:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return parsed.dropna()


def _looks_like_survival_target_columns(target_cols: list[str]) -> bool:
    has_event = any(_is_survival_event_column(str(col)) for col in target_cols)
    has_time = any(_is_survival_time_column(str(col)) for col in target_cols)
    return has_event and has_time


def _is_survival_event_column(name: str) -> bool:
    compact = re.sub(r"[^a-z0-9]+", "", name.lower())
    return compact in {"event", "eventobserved", "observed", "status", "efs", "censor", "censored", "death", "dead"}


def _is_survival_time_column(name: str) -> bool:
    compact = re.sub(r"[^a-z0-9]+", "", name.lower())
    return compact in {
        "time",
        "duration",
        "survivaltime",
        "timeevent",
        "timetoevent",
        "eventtime",
        "efstime",
        "os",
        "ostime",
        "dfs",
        "dfstime",
    }


def _looks_like_multi_label_target(target: pd.Series, *, column_name: str) -> bool:
    if not (pd.api.types.is_object_dtype(target) or pd.api.types.is_string_dtype(target)):
        return False
    tokens = set(_column_tokens(column_name))
    compact = "".join(_column_tokens(column_name))
    strong_name = bool(tokens & {"labels", "tags", "classes", "categories"}) or "multilabel" in compact
    generic_name = strong_name or bool(tokens & {"label", "target", "class", "category"})
    if not generic_name:
        return False
    sample = target.dropna().astype(str).str.strip().head(500)
    sample = sample[sample != ""]
    if sample.empty:
        return False

    multi_count = 0
    atomic_labels: set[str] = set()
    for value in sample:
        labels = _split_multi_label_value(value, allow_whitespace=strong_name)
        if len(labels) < 2:
            continue
        multi_count += 1
        atomic_labels.update(labels)
    if float(multi_count / len(sample)) < 0.6:
        return False
    return len(atomic_labels) >= 2


def _looks_like_ordinal_target(target: pd.Series, *, column_name: str) -> bool:
    tokens = set(_column_tokens(column_name))
    compact = "".join(_column_tokens(column_name))
    ordinal_name = bool(
        tokens
        & {
            "severity",
            "grade",
            "stage",
            "level",
            "rating",
            "risk",
            "quality",
            "ordinal",
            "class",
            "label",
        }
    ) or compact in {"risklevel", "severitygrade", "qualitygrade", "ordinaltarget"}
    if not ordinal_name:
        return False
    values = target.dropna()
    if values.empty:
        return False
    unique = safe_nunique(values)
    if unique < 3 or unique > 20:
        return False
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty or len(numeric) != len(values):
            return False
        integer_like = bool(((numeric % 1).abs() < 1e-9).all())
        return integer_like and safe_nunique(numeric) == unique
    return _looks_like_ordered_category_values(values)


def _looks_like_ordered_category_values(values: pd.Series) -> bool:
    ordered_vocab = {
        "verylow",
        "low",
        "medium",
        "moderate",
        "high",
        "veryhigh",
        "none",
        "mild",
        "severe",
        "critical",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
    }
    normalized = {re.sub(r"[^a-z0-9]+", "", str(value).strip().lower()) for value in values}
    normalized.discard("")
    if not normalized:
        return False
    return normalized.issubset(ordered_vocab)


def _split_multi_label_value(value: str, *, allow_whitespace: bool) -> list[str]:
    raw = value.strip()
    if not raw:
        return []
    if any(sep in raw for sep in ("|", ";", ",")):
        parts = re.split(r"[|;,]+", raw)
    elif allow_whitespace:
        parts = re.split(r"\s+", raw)
    else:
        return []
    labels = [part.strip() for part in parts if part.strip()]
    if len(labels) < 2:
        return []
    if any(len(label) > 48 for label in labels):
        return []
    if any(not re.fullmatch(r"[A-Za-z0-9_.:+-]+", label) for label in labels):
        return []
    return labels


def _infer_modality(
    data_dir: Path,
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    *,
    feature_cols: list[str] | None = None,
) -> str:
    asset_modality = infer_asset_modality(data_dir)
    if asset_modality not in {"unknown", "tabular"}:
        return asset_modality
    asset_reference_modality = _infer_asset_reference_column_modality(train, feature_cols=feature_cols)
    has_text_signal = _has_text_column_signal(train, feature_cols=feature_cols)
    if asset_reference_modality is not None:
        if has_text_signal:
            return "multimodal"
        return asset_reference_modality
    bio_modality = _infer_bio_column_modality(train, feature_cols=feature_cols)
    if bio_modality is not None:
        return bio_modality
    if _has_graph_column_signal(train, feature_cols=feature_cols):
        return "graph"
    if _has_geospatial_column_signal(train, feature_cols=feature_cols):
        return "geospatial"
    if has_text_signal:
        return "text"
    columns = [col for col in (feature_cols or list(train.columns)) if col in train.columns]
    text_cols = [c for c in columns if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
    if text_cols:
        avg_len = 0.0
        sample = train[text_cols].astype(str).head(200)
        if not sample.empty:
            avg_len = sample.apply(lambda col: col.map(len)).mean().mean()
        if avg_len >= 30:
            return "text"
    if _has_temporal_signal(train, test):
        return "timeseries"
    return "tabular"


def _infer_group_column_hint(train: pd.DataFrame, *, feature_cols: list[str] | None = None) -> str | None:
    columns = [col for col in (feature_cols or list(train.columns)) if col in train.columns]
    scored: list[tuple[float, int, str]] = []
    for index, col in enumerate(columns):
        name = str(col)
        name_score = _group_column_name_score(name)
        if name_score <= 0:
            continue
        values_score = _group_column_values_score(train[col])
        if values_score <= 0:
            continue
        scored.append((name_score + values_score, -index, name))
    if not scored:
        return None
    return max(scored)[2]


def _group_column_name_score(name: str) -> float:
    tokens = set(_column_tokens(name))
    compact = "".join(_column_tokens(name))
    if compact in {"id", "rowid", "index", "targetid", "predictionid"}:
        return 0.0
    entity_tokens = {
        "account",
        "author",
        "case",
        "center",
        "customer",
        "device",
        "document",
        "donor",
        "entity",
        "group",
        "household",
        "installation",
        "patient",
        "participant",
        "session",
        "site",
        "source",
        "study",
        "subject",
        "user",
        "visit",
    }
    if tokens & entity_tokens:
        score = 0.8
        if "id" in tokens or compact.endswith("id"):
            score += 0.2
        return score
    if compact in {
        "patientid",
        "subjectid",
        "participantid",
        "sessionid",
        "visitid",
        "caseid",
        "studyid",
        "siteid",
        "userid",
        "customerid",
        "accountid",
        "householdid",
        "donorid",
        "authorid",
        "deviceid",
        "sourceid",
        "groupid",
    }:
        return 1.0
    return 0.0


def _group_column_values_score(series: pd.Series) -> float:
    sample = series.dropna().head(1000)
    if len(sample) < 4:
        return 0.0
    unique_count = safe_nunique(sample)
    if unique_count < 2 or unique_count >= len(sample):
        return 0.0
    value_counts = sample.astype(str).value_counts(dropna=True)
    if value_counts.empty or int(value_counts.max()) < 2:
        return 0.0
    repeat_ratio = 1.0 - (float(unique_count) / float(len(sample)))
    if repeat_ratio < 0.05:
        return 0.0
    return min(0.5, repeat_ratio)


def _infer_sample_weight_column_hint(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    id_col: str | None,
) -> str | None:
    excluded = {str(col) for col in target_cols}
    if id_col is not None:
        excluded.add(str(id_col))
    candidates = [str(col) for col in train.columns if str(col) not in excluded]
    scored: list[tuple[float, int, str]] = []
    for index, col in enumerate(candidates):
        if col not in train.columns:
            continue
        name_score = _sample_weight_column_name_score(col, train_only=col not in test.columns)
        if name_score <= 0:
            continue
        values_score = _sample_weight_values_score(train[col])
        if values_score <= 0:
            continue
        feature_penalty = 0.15 if col in feature_cols and col in test.columns else 0.0
        scored.append((name_score + values_score - feature_penalty, -index, col))
    if not scored:
        return None
    return max(scored)[2]


def _sample_weight_column_name_score(name: str, *, train_only: bool) -> float:
    tokens = set(_column_tokens(name))
    compact = "".join(_column_tokens(name))
    explicit_names = {
        "sampleweight",
        "rowweight",
        "observationweight",
        "instanceweight",
        "exampleweight",
        "evalweight",
        "evaluationweight",
        "metricweight",
        "targetweight",
    }
    if compact in explicit_names:
        return 1.0
    if "weight" in tokens and tokens & {
        "sample",
        "row",
        "observation",
        "instance",
        "example",
        "eval",
        "evaluation",
        "metric",
    }:
        return 0.95
    if compact in {"weight", "weights"} and train_only:
        return 0.72
    return 0.0


def _sample_weight_values_score(series: pd.Series) -> float:
    numeric = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    if len(numeric) < 2:
        return 0.0
    finite = numeric[numeric.map(lambda value: pd.notna(value))]
    if finite.empty:
        return 0.0
    if float((finite < 0).mean()) > 0.0:
        return 0.0
    if float((finite > 0).mean()) < 0.8:
        return 0.0
    return 0.35 if safe_nunique(finite) > 1 else 0.2


def _sample_weight_summary(series: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    if numeric.empty:
        return {"non_null": 0}
    return {
        "non_null": int(len(numeric)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "mean": float(numeric.mean()),
    }


def _has_text_column_signal(train: pd.DataFrame, *, feature_cols: list[str] | None = None) -> bool:
    columns = [col for col in (feature_cols or list(train.columns)) if col in train.columns]
    for col in columns:
        if not _is_text_feature_column(str(col)):
            continue
        series = train[col]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        if _looks_like_text_values(series):
            return True
    return False


def _is_text_feature_column(name: str) -> bool:
    tokens = set(_column_tokens(name))
    compact = "".join(_column_tokens(name))
    if tokens & {"path", "file", "filename", "filepath", "uri", "url"} or compact.endswith(
        ("path", "file", "filename", "filepath", "uri", "url")
    ):
        return False
    return bool(
        tokens
        & {
            "abstract",
            "body",
            "comment",
            "content",
            "description",
            "document",
            "essay",
            "message",
            "post",
            "prompt",
            "question",
            "review",
            "sentence",
            "text",
            "title",
            "transcript",
            "tweet",
        }
    ) or compact in {"questiontext", "reviewtext", "tweettext", "messagetext", "documenttext"}


def _looks_like_text_values(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).str.strip().head(200)
    if sample.empty:
        return False
    lengths = sample.map(len)
    if float((lengths >= 8).mean()) < 0.6:
        return False
    avg_len = float(lengths.mean())
    if avg_len < 10:
        return False
    word_like_ratio = float(sample.map(lambda value: bool(re.search(r"\s|[.!?,;:]", value))).mean())
    unique_ratio = float(safe_nunique(sample) / len(sample))
    return word_like_ratio >= 0.5 and (avg_len >= 12 or unique_ratio >= 0.8)


def _infer_asset_reference_column_modality(train: pd.DataFrame, *, feature_cols: list[str] | None = None) -> str | None:
    columns = [col for col in (feature_cols or list(train.columns)) if col in train.columns]
    suffixes: set[str] = set()
    for col in columns:
        suffixes.update(_asset_suffixes_from_values(train[col]))
    if suffixes:
        modality = infer_asset_modality_from_extensions(suffixes)
        if modality not in {"unknown", "tabular"}:
            return modality

    for col in columns:
        modality = _infer_asset_reference_modality_from_column_name(str(col))
        if modality is not None:
            return modality
    return None


def _asset_suffixes_from_values(series: pd.Series) -> set[str]:
    sample = series.dropna().astype(str).str.strip().head(200)
    if sample.empty:
        return set()
    suffixes: set[str] = set()
    matched = 0
    for value in sample:
        suffix = _asset_suffix_from_text(value)
        if suffix is None:
            continue
        matched += 1
        suffixes.add(suffix)
    if float(matched / len(sample)) < 0.6:
        return set()
    return suffixes


def _asset_suffix_from_text(value: str) -> str | None:
    candidate = value.strip().strip("\"'")
    if not candidate or any(char.isspace() for char in candidate):
        return None
    candidate = candidate.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    if not candidate:
        return None
    suffix = asset_suffix(Path(candidate))
    known_suffixes = (
        IMAGE_SUFFIXES
        | MEDICAL_IMAGE_SUFFIXES
        | AUDIO_SUFFIXES
        | VIDEO_SUFFIXES
        | SIGNAL_SUFFIXES
        | DOCUMENT_SUFFIXES
        | ARRAY_SUFFIXES
        | SCIENTIFIC_ARRAY_SUFFIXES
        | POINT_CLOUD_SUFFIXES
        | GEOSPATIAL_SUFFIXES
        | BIO_STRUCTURE_SUFFIXES
        | GRAPH_SUFFIXES
    )
    return suffix if suffix in known_suffixes else None


def _infer_asset_reference_modality_from_column_name(name: str) -> str | None:
    tokens = set(_column_tokens(name))
    compact = "".join(_column_tokens(name))
    if tokens & {"dicom", "nifti", "scan", "mri", "ct"} or compact in {"scanpath", "dicompath", "niftipath"}:
        return "medical_imaging"
    if tokens & {"signal", "signals", "waveform", "waveforms", "ecg", "eeg"} or compact in {
        "signalpath",
        "signalfile",
        "waveformpath",
        "waveformfile",
    }:
        return "signal"
    if tokens & {"audio", "sound", "speech"} or compact in {"audiopath", "audiofile", "soundpath"}:
        return "audio"
    if tokens & {"video", "frame"} or compact in {"videopath", "videofile", "clipfilename", "clippath"}:
        return "video"
    if tokens & {"document", "documents", "doc", "docs", "pdf", "report"} or compact in {
        "documentpath",
        "documentfile",
        "docpath",
        "docfile",
        "pdffile",
        "pdfpath",
        "reportpath",
    }:
        return "text"
    if tokens & {
        "annotation",
        "annotations",
        "bbox",
        "bboxes",
        "mask",
        "masks",
        "labelme",
        "coco",
        "yolo",
    } or compact in {
        "annotationpath",
        "annotationfile",
        "annotationspath",
        "annotationsfile",
        "bboxpath",
        "maskpath",
        "labelmepath",
        "cocopath",
        "yolopath",
    }:
        return "annotation"
    if tokens & {"lidar", "pointcloud"} or compact in {"lidarfile", "lidarpath", "pointcloudpath"}:
        return "point_cloud"
    if tokens & {"array", "netcdf", "grib", "fits", "zarr", "h5ad", "loom"}:
        return "array"
    if tokens & {"image", "photo", "picture"} or compact in {"imagepath", "imagefile", "filepathimage"}:
        return "image"
    return None


def _has_temporal_signal(train: pd.DataFrame, test: pd.DataFrame | None = None) -> bool:
    """Return True when the table contains a plausible temporal feature column."""
    for col in train.columns:
        series = train[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        if not _column_name_has_temporal_token(str(col)):
            continue
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_numeric_dtype(series)
        ):
            continue
        sample = series.dropna().astype(str).head(200)
        if sample.empty:
            continue
        parsed = pd.to_datetime(sample, errors="coerce", utc=True, format="mixed")
        if float(parsed.notna().mean()) >= 0.8:
            return True
        if test is not None and col in test.columns and _has_future_ordinal_holdout(series, test[col]):
            return True
    return False


def _column_name_has_temporal_token(name: str) -> bool:
    tokens = [token for token in re.split(r"[^A-Za-z0-9]+", name.lower()) if token]
    if any(
        token in {"date", "datetime", "timestamp", "time", "day", "daynum", "week", "month", "year"} for token in tokens
    ):
        return True
    compact = "".join(tokens)
    return compact in {"dateblocknum", "daynum", "weekofyear"}


def _has_future_ordinal_holdout(train_series: pd.Series, test_series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(train_series):
        return False
    train_values = pd.to_numeric(train_series, errors="coerce").dropna()
    test_values = pd.to_numeric(test_series, errors="coerce").dropna()
    if train_values.empty or test_values.empty:
        return False
    if safe_nunique(train_values) < 3:
        return False
    return float(test_values.min()) > float(train_values.max())


def _infer_bio_column_modality(train: pd.DataFrame, *, feature_cols: list[str] | None = None) -> str | None:
    columns = [col for col in (feature_cols or list(train.columns)) if col in train.columns]
    rna_columns = [
        col
        for col in columns
        if _is_rna_column(str(col)) or _looks_like_sequence_values(train[col], alphabet=set("ACGUTN"))
    ]
    if rna_columns:
        return "rna"
    for col in columns:
        if (
            _is_bio_column(str(col))
            or _looks_like_smiles_values(train[col])
            or _looks_like_sequence_values(train[col], alphabet=set("ACDEFGHIKLMNPQRSTVWYXBZUO"))
        ):
            return "bio"
    return None


def _is_rna_column(name: str) -> bool:
    tokens = set(_column_tokens(name))
    compact = "".join(_column_tokens(name))
    return bool(tokens & {"rna", "transcript", "nucleotide"}) or compact in {
        "rnasequence",
        "nucleotidesequence",
    }


def _is_bio_column(name: str) -> bool:
    tokens = set(_column_tokens(name))
    compact = "".join(_column_tokens(name))
    return bool(
        tokens & {"smiles", "molecule", "molecular", "protein", "peptide", "sequence", "fasta", "fastq"}
    ) or compact in {
        "proteinpath",
        "proteinid",
        "proteinsequence",
        "aminoacidsequence",
        "moleculeid",
        "molecularformula",
        "canonicalsmiles",
        "isosmiles",
    }


def _looks_like_smiles_values(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).str.strip().head(200)
    if sample.empty:
        return False
    smiles_re = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+$")

    def is_smiles(value: str) -> bool:
        if len(value) < 3 or " " in value:
            return False
        if not smiles_re.fullmatch(value):
            return False
        return any(marker in value for marker in ("C", "N", "O", "S", "P", "Cl", "Br", "=", "#", "(", "["))

    return float(sample.map(is_smiles).mean()) >= 0.6


def _looks_like_sequence_values(series: pd.Series, *, alphabet: set[str]) -> bool:
    sample = series.dropna().astype(str).str.strip().str.upper().head(200)
    if sample.empty:
        return False

    def is_sequence(value: str) -> bool:
        if len(value) < 8:
            return False
        chars = set(value)
        return chars.issubset(alphabet) and len(chars) >= 2

    return float(sample.map(is_sequence).mean()) >= 0.6


def _has_graph_column_signal(train: pd.DataFrame, *, feature_cols: list[str] | None = None) -> bool:
    columns = [str(col) for col in (feature_cols or list(train.columns)) if col in train.columns]
    if any(_is_edge_index_column(column) for column in columns):
        return True
    return _has_source_destination_columns(columns)


def _is_edge_index_column(name: str) -> bool:
    tokens = _column_tokens(name)
    compact = "".join(tokens)
    return any(token in {"edge", "edges", "edgelist", "adjacency"} for token in tokens) or compact in {
        "edgeindex",
        "edgeindices",
        "edgelist",
        "adjacencylist",
        "adjacencymatrix",
    }


def _has_source_destination_columns(columns: list[str]) -> bool:
    return any(_is_source_node_column(column) for column in columns) and any(
        _is_destination_node_column(column) for column in columns
    )


def _is_source_node_column(name: str) -> bool:
    tokens = set(_column_tokens(name))
    compact = "".join(_column_tokens(name))
    return bool(tokens & {"src", "source"}) or compact in {"sourcenode", "sourceid", "fromnode", "fromid", "node1"}


def _is_destination_node_column(name: str) -> bool:
    tokens = set(_column_tokens(name))
    compact = "".join(_column_tokens(name))
    return bool(tokens & {"dst", "destination"}) or compact in {
        "destinationnode",
        "destinationid",
        "targetnode",
        "targetid",
        "tonode",
        "toid",
        "node2",
    }


def _has_geospatial_column_signal(train: pd.DataFrame, *, feature_cols: list[str] | None = None) -> bool:
    columns = [col for col in (feature_cols or list(train.columns)) if col in train.columns]
    lat_columns = [
        col for col in columns if _is_latitude_column(str(col)) and _numeric_values_in_range(train[col], -90.0, 90.0)
    ]
    lon_columns = [
        col for col in columns if _is_longitude_column(str(col)) and _numeric_values_in_range(train[col], -180.0, 180.0)
    ]
    if lat_columns and lon_columns:
        return True
    return any(_is_geometry_column(str(col)) and _looks_like_geometry_values(train[col]) for col in columns)


def _column_tokens(name: str) -> list[str]:
    return [token for token in re.split(r"[^A-Za-z0-9]+", name.lower()) if token]


def _is_latitude_column(name: str) -> bool:
    tokens = _column_tokens(name)
    compact = "".join(tokens)
    return "latitude" in tokens or "lat" in tokens or compact in {"latitude", "lat"}


def _is_longitude_column(name: str) -> bool:
    tokens = _column_tokens(name)
    compact = "".join(tokens)
    return any(token in {"longitude", "lon", "lng"} for token in tokens) or compact in {"longitude", "lon", "lng"}


def _is_geometry_column(name: str) -> bool:
    tokens = _column_tokens(name)
    compact = "".join(tokens)
    return any(token in {"geometry", "geom", "wkt", "geohash"} for token in tokens) or compact in {
        "geometry",
        "geom",
        "wkt",
        "geohash",
    }


def _numeric_values_in_range(series: pd.Series, lower: float, upper: float) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return False
    return float(((values >= lower) & (values <= upper)).mean()) >= 0.8


def _looks_like_geometry_values(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).str.strip().head(200)
    if sample.empty:
        return False
    geometry_re = re.compile(
        r"^(?:POINT|LINESTRING|POLYGON|MULTI(?:POINT|LINESTRING|POLYGON)|GEOMETRYCOLLECTION)\s*\(", re.I
    )
    return float(sample.map(lambda value: bool(geometry_re.match(value))).mean()) >= 0.5


def _task_tag(task: str, target: pd.Series) -> str:
    if task == "mixed":
        return "multitask"
    if task in {"text", "translation", "text_generation"}:
        return "text"
    if task == "regression":
        return "regression"
    unique = safe_nunique(target)
    return "binary" if unique <= 2 else "multiclass"


def _size_tag(n_rows: int) -> str:
    if n_rows < 10_000:
        return "n_rows_small"
    if n_rows < 100_000:
        return "n_rows_medium"
    return "n_rows_large"


def _format_dataset_dimensions(profile: dict[str, object]) -> str:
    train_rows = profile.get("train_rows")
    train_cols = profile.get("train_cols")
    if isinstance(train_rows, int) and isinstance(train_cols, int):
        return f"{train_rows:,} rows × {train_cols} columns"

    test_rows = profile.get("test_rows")
    test_cols = profile.get("test_cols")
    if isinstance(test_rows, int) and isinstance(test_cols, int):
        return f"train table unavailable; sample/test view: {test_rows:,} rows × {test_cols} columns"

    return "unknown"


def build_plan_and_initial_prompt(
    *,
    slug: str,
    rules_url: str,
    profile: dict[str, object],
    taxonomy: dict[str, object],
    similar_improvements: list[dict[str, object]],
    self_improvement_context: str = "",
) -> str:
    tags = profile.get("tags", [])
    task = profile.get("task", "unknown")
    target_semantics = str(profile.get("target_semantics") or "").strip()
    metric = profile.get("metric", "rmse")
    dataset_dimensions = _format_dataset_dimensions(profile)
    raw_sample_submission_file = profile.get("sample_submission_file")
    sample_submission_file = (
        str(raw_sample_submission_file).strip()
        if isinstance(raw_sample_submission_file, str) and str(raw_sample_submission_file).strip()
        else "sample_submission.* or the detected sample-submission alias"
    )
    sample_submission_head_file = _sample_submission_head_file_for_prompt(sample_submission_file)

    lines = [
        f"# Kagglebot {IMPLEMENTATION_AGENT.display_name}: Plan + Implement (Iteration 1)",
        "",
        "## Competition Overview",
        "",
        f"**Slug**: {slug}",
        "**Competition URL**: {{competition_url}}",
        f"**Rules URL**: {rules_url}",
        f"**Task**: {task}",
        *([f"**Target semantics**: {target_semantics}"] if target_semantics else []),
        f"**Metric (confirm via rules)**: {metric}",
        f"**Dataset**: {dataset_dimensions}",
        f"**Tags**: {', '.join(tags) if tags else 'None'}",
        "",
        "## Compute Context",
        "",
        "- Compute mode: {{compute_mode}}",
        "- Accelerator: {{accelerator}}",
        "- Internet: {{internet}}",
        "- Run ID: {{run_id}}",
        "",
        "## Context Files (read these)",
        "",
        f"- artifacts/{slug}/context/dataset_profile.json",
        f"- artifacts/{slug}/context/{sample_submission_file}",
        f"- artifacts/{slug}/context/{sample_submission_head_file}",
        f"- artifacts/{slug}/context/top1_public.json",
        f"- artifacts/{slug}/context/rules_url.txt",
        f"- artifacts/{slug}/context/rules.md (if present)",
        f"- artifacts/{slug}/context/rules.html (if present)",
        f"- artifacts/{slug}/context/overview.md (if present)",
        f"- artifacts/{slug}/context/data.md (if present)",
        f"- artifacts/{slug}/context/submission_format.md (if present)",
        f"- artifacts/{slug}/kernel/kernel.py (authoritative for all compute modes)",
        f"- artifacts/{slug}/context/knowledge_hints.txt",
        "",
        "## Knowledge Base: Similar Competitions",
        "",
    ]

    if similar_improvements:
        lines.append("We found past runs on similar competitions. Learn from what worked:")
        lines.append("")
        for item in similar_improvements:
            overlap = item.get("overlap", 0)
            summary = item.get("summary", "No summary")
            comp_slug = item.get("slug", "unknown")
            lines.append(f"- {comp_slug} ({overlap} tag overlap): {summary}")
    else:
        lines.append("No similar competitions found in knowledge base (this is the first run).")

    lines.extend(["", "## System Self-Improvement Directives", ""])
    if self_improvement_context.strip():
        lines.append(self_improvement_context.strip())
    else:
        lines.append("No system self-improvement report is available yet.")

    lines += [
        "",
        "---",
        "",
        "## Your Task",
        "",
        "### 1) Decide the plan (required)",
        "",
        f"Update artifacts/{slug}/plan.json with:",
        "",
        "```json",
        "{",
        '  "target_metric": "<derive_from_rules>",',
        '  "target_score": 0.0,',
        '  "target_direction": "<minimize|maximize>",',
        '  "score_source": "cv",',
        '  "holdout_frac": 0.2,',
        '  "cv_folds": 5,',
        '  "seed": 42,',
        '  "target_medal": "winner",',
        '  "target_rank_percentile": 0.001,',
        '  "internet": "on",',
        '  "max_iterations": 5,',
        '  "submit_policy": "always"',
        "}",
        "```",
        "",
        "Guidance:",
        "- Derive target_metric and direction from rules.md/rules.html and the sample submission file.",
        "- Read overview.md/data.md for problem framing and data caveats.",
        "- If sample_submission is ambiguous, use submission_format.md for the required columns.",
        "- Use overview.md/data.md plus dataset_profile.json (file_extension_counts/file_samples)",
        "  to identify data format.",
        "- Use web search to choose the strongest initial approach; prefer official docs and competition discussions.",
        "- Use top1_public.json to set a realistic target_score; avoid generic metric heuristics.",
        "- Default to winner-mode search: target_medal=winner and target_rank_percentile=0.001 unless rules/runtime "
        "make that impossible.",
        "- Prefer CV by default for stronger model ranking; use holdout only when CV is infeasible.",
        "- Actively use preinstalled dependencies before introducing new ones:",
        "  torch/timm/torchvision/opencv, xgboost/lightgbm/catboost, transformers/tabicl, sklearn.",
        "- If one backend import fails, disable only that path and keep other high-capacity backends active.",
        "- When a required package is truly missing, add it with `uv add <package>` so",
        "  `pyproject.toml` and `uv.lock` stay consistent.",
        "- Do NOT change submit_policy; autopilot controls submission behavior.",
        "",
        "### 2) Implement the strongest initial model",
        "",
        "Implement the best initial approach based on overview/data/rules + web research:",
        "- Loads train/test and sample_submission correctly.",
        "- Handles missing values and encodes categoricals.",
        "- Uses a GPU-optimized supervised model if available.",
        "- Avoid simplistic starters; aim for competition-appropriate strength from the start.",
        "- Do NOT leave the default starter in place; replace it with a competition-specific model.",
        "- Evaluates with the score_source from plan.json.",
        "- Writes the required submission artifact matching the sample/format exactly.",
        "- Cite the key sources you used (short notes).",
        "",
        "Implementation notes:",
        "- For local_gpu/kaggle_gpu/kaggle_tpu: use artifacts/{slug}/kernel/kernel.py",
        "  as the only training implementation.",
        "- local_gpu and kaggle_gpu must keep the same algorithm/pipeline; only execution location may differ.",
        "- Create artifacts/{slug}/kernel/kernel.py from scratch if missing.",
        "- If data is non-tabular or requires custom parsing, implement custom_main() in kernel.py.",
        "- If pretrained checkpoints are likely beneficial, implement download + cache logic in kernel.py",
        "  and provide a fallback path when internet is unavailable.",
        "- When reusable text/translation helpers emerge, move them into `src/kagglebot/kernel_runtime/`",
        "  and keep competition-specific metadata joins / dictionaries inside `kernel.py`.",
        "- For vision/multimodal tasks, prefer strong prepared backbones (e.g., timm/ConvNeXt)",
        "  and avoid silent downgrade to weak fallback models when imports succeed.",
        "- If GPU runs finish in under ~1 minute, increase model iterations/trees or use CV to better utilize GPU.",
        "- Do NOT implement competition-specific logic in src local trainers.",
        "",
        "### 3) Safety + verification",
        "",
        "- NEVER accept Kaggle rules via API.",
        "- NEVER include secrets in code/logs.",
        "- NO interactive prompts.",
        "",
        "Run tests before finishing:",
        "```bash",
        "uv run pytest -q",
        "```",
        "",
        "The autopilot will iterate up to max_iterations (default 5),",
        "submit according to submission gate policy, and use readiness score as the primary loop decision signal.",
    ]
    return "\n".join(lines) + "\n"


def build_improve_template() -> str:
    """Build the improvement prompt template for iterations 1-N.

    This template has placeholders that get filled with .format() at runtime:
    - {slug}, {iteration}, {plan_path}, {run_path}, {metrics_path}, {diagnostics_path}, {logs_dir}
    - {code_md}, {code_index}, {code_reference_score}, {code_reference_source},
      {code_reference_delta}, {code_reference_status}
    """
    return """\
# Kagglebot {implementation_agent_name}: Improvement Iteration

## Context

**Competition**: `{slug}`
**Iteration**: {iteration}
**Goal**: Improve loop-decision score (readiness primary; submission/rank as guardrails) toward top1-tier
or best possible within the max_iterations budget (default 5)
**Compute**: {compute} ({accelerator})
**Top1 gap**: {top1_gap}
**Delta vs previous best**: {delta_offline}
**Improvement mode**: {improvement_mode}
**Next iteration**: {next_iteration}

## Current Score Context

- **Metric**: {metric} ({direction})
- **Current score**: {current_score} (source: {current_score_source})
- **Target score**: {target_score}
- **Top1 public**: {top1_score} (source: {top1_source})

## Input Files

- **Plan**: `{plan_path}` - Target metric/score/direction, evaluation strategy
- **Run Config**: `{run_path}` - Current run settings and history
- **Current Metrics**: `{metrics_path}` - Latest loop-decision + offline-by-source results
- **Diagnostics**: `{diagnostics_path}` - Agent-readable analysis of current performance
- **Logs**: `{logs_dir}` - Training logs and error messages (if any)
- **Knowledge Hints**: `{knowledge_hints}`
- **Rules URL**: `{rules_url}`
- **Rules Markdown**: `{rules_md}` (preferred)
- **Rules HTML**: `{rules_html}` (fallback)
- **Overview Markdown**: `{overview_md}` (read this)
- **Data Markdown**: `{data_md}` (read this)
- **Submission Format**: `{submission_format}` (read this if present)
- **Dataset Profile**: `{dataset_profile}`
- **Sample Submission**: `{sample_submission}`
- **Code Snapshot**: `{code_md}` (read this; includes Required Reference Notebook baseline)
- **Code Notebook Index**: `{code_index}` (use this for structured notebook metadata)
- **Code Reference Score**: `{code_reference_score}`
  (source: {code_reference_source}, delta_vs_current: {code_reference_delta}, status: {code_reference_status})
- **Kernel Main**: `{kernel_main}` (edit this for all compute modes)

## Your Task

### Step 1: Analyze Current Performance

Read the diagnostics and metrics files to understand:

1. **What is the current score?**
   - Check `metrics.json` for `loop_decision.source` and `loop_decision.value`
   - Also check offline-by-source metrics for model-quality signal
   - Compare to `target_score` in `plan.json`

2. **Why is the score not meeting target?**
   - Underfitting? (high train + val error → model too simple)
   - Overfitting? (low train error, high val error → model too complex or bad CV)
   - Data leakage? (unrealistically good scores)
   - Poor feature engineering? (missing important signals)
   - Wrong hyperparameters? (learning rate, regularization, tree depth)
   - Class imbalance? (classification only)
   - Bad splits? (e.g., time series shuffled incorrectly)

3. **What has been tried before?**
   - Check `{run_path}` for previous iteration summaries
   - Don't repeat failed experiments

### Step 2: Implement Improvements

Make **targeted, incremental changes** to improve the current loop-decision score
(submission preferred; offline fallback).
Prefer the highest realistic score ceiling over making the current iteration immediately submittable.
If forced to choose, keep a stronger high-potential path instead of collapsing to a weak submit-ready fallback.

Before changing the model, read overview.md/data.md and respect any constraints or data caveats.

**Where to implement**:
- Always edit `{kernel_main}` (create if missing).
- `local_gpu` and `kaggle_gpu` must use the same algorithm/pipeline.
- For non-tabular inputs, implement `custom_main()` in `{kernel_main}`.

**Modeling policy**:
- Use rules.md + dataset_profile.json + overview/data to decide the best supervised approach.
- Use dataset_profile.json (file_extension_counts/file_samples) to confirm the data format.
- Avoid weak starter implementations.
- Avoid generic metric heuristics; derive metric choices from the rules.
- Use web search each iteration to validate model/feature choices; prefer official docs or top Kaggle discussions.
- Prefer already-available dependencies before adding new ones:
  torch/timm/torchvision/opencv, xgboost/lightgbm/catboost, transformers/tabicl, sklearn.
- Do not silently downgrade model capacity when prepared libraries import successfully.
- If one dependency is missing, isolate that path only and keep other strong pipelines enabled.
- If a dependency is genuinely required and missing, you may add it via `uv add <package>`;
  keep `pyproject.toml` and `uv.lock` in sync.
- Use one consistent evaluation path for training-time selection and final offline scoring.
  If you print `val_*` during training, it must be computed under the same split/rollout/aggregation
  assumptions as the final reported metric.
- If the previous run failed because of timeout or local GPU budget, keep the highest-ceiling pipeline but
  reduce multiplicative runtime first: one full fine-tune seed, fewer full-training folds for heavy backbones,
  cached embeddings or lightweight heads for extra seeds, earlier stopping, and checkpoint reuse. Do not replace
  a working strong model with a weak baseline only to finish faster.
- local_gpu has no default wall-clock limit unless a stricter competition runtime cap applies.
- Evaluate at least one simple baseline (mean/majority/persistence as appropriate) with the same
  validation protocol, and do not select/submit a learned pipeline that underperforms that baseline.
- If `code_reference_status` is `underperforming_code_reference`, you MUST inspect `{code_md}` and `{code_index}`
  and treat `Required Reference Notebook (Execution baseline)` as mandatory context before proposing new variants.

**Mode guidance**:
- `major_overhaul`: switch model family or core feature strategy; remove dead code paths.
- `moderate_update`: add meaningful features + tune key hyperparameters.
- `minor_tuning`: small hyperparameter/feature tweaks, calibration, or ensembling.

**Pretrained model guidance**:
- If transfer learning can improve score, add checkpoint download + cache logic in `kernel.py`.
- Respect rules and internet settings, and include an internet-off fallback model path.

If the score did not improve (delta <= 0), treat it as `major_overhaul` even if the default mode is weaker.

**Recommended strategies** (pick 1-2 per iteration):

**For Underfitting**:
- Add more features (interactions, polynomials, domain-specific)
- Use a more complex model (Linear → Tree Ensemble → Neural Network)
- Reduce regularization (lower L1/L2 penalties, increase max_depth)
- Increase training time (more epochs, trees)

**For Overfitting**:
- Add regularization (L1/L2, dropout, early stopping)
- Reduce model complexity (fewer trees, smaller max_depth, fewer layers)
- Improve cross-validation (more folds, stratified splits, grouped CV)
- Feature selection (drop noisy/redundant features)
- Data augmentation (if applicable)

**For Feature Engineering**:
- Handle missing values better (indicators, model-based imputation)
- Encode categoricals differently (target encoding, embeddings)
- Create domain-specific features (e.g., for time series: lags, rolling stats)
- Remove highly correlated features
- Log/sqrt transform skewed features

**For Hyperparameter Tuning**:
- Grid search or random search over key parameters
- Tune learning rate, max_depth, n_estimators, regularization
- Use appropriate loss function for the metric

**For Classification**:
- Handle class imbalance (SMOTE, class weights, stratified sampling)
- Adjust decision threshold (precision-recall tradeoff)
- Try different metrics during training (focal loss for imbalance)

**For Ensembling** (later iterations only):
- Blend predictions from multiple models
- Stack models with a meta-learner
- Average across CV folds

### Step 3: Validate Changes

Before finalizing:

**Agent runtime budget**:
- Do not launch or monitor a full local GPU/autopilot run from inside the implementation agent.
- Run only bounded validation: py_compile, fast-dev smoke, config checks, and unit tests.
- If full training is needed, exit after reporting the exact command; the autopilot orchestrator will run it.
- Do not tail files under `kernels/` or poll `nvidia-smi` after the fix is validated.

1. **Run offline evaluation (supporting signal)**:
   - Run `kernel.py` via autopilot/train and use kernel metrics/logs for evaluation.
   - Keep one implementation path in `kernel.py` across local and kaggle execution.

2. **Check metrics**:
   - Is the score improving in the right direction?
   - Is train-val gap reasonable? (not too large = overfitting)

3. **Run tests**:
   ```bash
   uv run pytest -q
   ```

### Step 4: Safety Checks

**CRITICAL SAFETY RULES**:
- ❌ NEVER automate rules acceptance or submission
- ❌ NEVER commit secrets (kaggle.json, API keys)
- ❌ NEVER add interactive prompts
- ❌ NEVER bypass safety guardrails (duplicate checks, rate limits)
- ✅ DO keep changes incremental and testable
- ✅ DO preserve backward compatibility with existing CLI
- ✅ DO document significant changes in code comments

**Quality Checklist**:
- [ ] Changes address root cause from diagnostics
- [ ] Loop-decision score improves, or offline diagnostics provide actionable learning
- [ ] Submission artifact format still matches the required sample/format
- [ ] Tests pass: `uv run pytest -q`
- [ ] No secrets leaked into code/logs

---

## Tips for Effective Improvements

1. **Make one change at a time**: Easier to debug and understand what worked
2. **Optimize loop decision**: prioritize submission score when available; use offline as fallback
3. **Read diagnostics carefully**: The autopilot generates actionable hints
4. **Check for data leakage**: If score is "too good to be true", it probably is
5. **Don't over-optimize**: Diminishing returns after 3-5 iterations are normal
6. **Document what didn't work**: Failed experiments provide valuable learning

Good luck improving the model! 🔧
"""


def resolve_similar_improvements(
    *,
    knowledge_paths: KnowledgePaths,
    taxonomy: dict[str, object],
    tags: Iterable[str],
    limit: int = 3,
) -> list[dict[str, object]]:
    tax = _taxonomy_from_dict(taxonomy)
    normalized = tax.normalize(tags)
    if not normalized:
        return []
    _ensure_db(knowledge_paths)
    with _connect(knowledge_paths.kb_path) as conn:
        placeholders = ",".join("?" for _ in normalized)
        rows = conn.execute(
            f"""
            SELECT slug, COUNT(*) as overlap
            FROM competition_tags
            WHERE tag IN ({placeholders})
            GROUP BY slug
            ORDER BY overlap DESC
            LIMIT ?
            """,
            [*normalized, limit],
        ).fetchall()

        results: list[dict[str, object]] = []
        for row in rows:
            slug = row["slug"]
            overlap = row["overlap"]
            improvement = conn.execute(
                """
                SELECT summary, delta_offline
                FROM improvements
                JOIN runs ON improvements.run_id = runs.run_id
                WHERE runs.slug = ?
                ORDER BY delta_offline DESC, created_at DESC
                LIMIT 1
                """,
                (slug,),
            ).fetchone()
            summary = improvement["summary"] if improvement else "No improvement summary recorded."
            results.append({"slug": slug, "overlap": overlap, "summary": summary})
        return results


def record_competition_profile(
    *,
    knowledge_paths: KnowledgePaths,
    taxonomy: dict[str, object],
    slug: str,
    competition_url: str | None,
    profile: dict[str, object],
) -> None:
    tax = _taxonomy_from_dict(taxonomy)
    tags = tax.normalize(profile.get("tags", []))
    _ensure_db(knowledge_paths)
    now = int(time.time())
    with _connect(knowledge_paths.kb_path) as conn:
        conn.execute(
            """
            INSERT INTO competitions (slug, url, metric, task_type, created_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(slug) DO UPDATE SET
                url=excluded.url,
                metric=excluded.metric,
                task_type=excluded.task_type,
                last_seen_at=excluded.last_seen_at
            """,
            (
                slug,
                competition_url,
                profile.get("metric"),
                profile.get("task"),
                now,
                now,
            ),
        )
        for tag in tags:
            conn.execute("INSERT OR IGNORE INTO tags (tag) VALUES (?)", (tag,))
            conn.execute(
                "INSERT OR IGNORE INTO competition_tags (slug, tag) VALUES (?, ?)",
                (slug, tag),
            )


def record_run(
    *,
    knowledge_paths: KnowledgePaths,
    run_id: str,
    slug: str,
    compute: str,
    goal_metric: str,
    goal_score: float,
    direction: str,
) -> None:
    InsightRepository(knowledge_paths, ensure_db=_ensure_db, connect=_connect).record_run(
        run_id=run_id,
        slug=slug,
        compute=compute,
        goal_metric=goal_metric,
        goal_score=goal_score,
        direction=direction,
    )


def record_iteration(
    *,
    knowledge_paths: KnowledgePaths,
    run_id: str,
    iteration: int,
    score_source: str,
    offline_value: float,
    offline_std: float | None,
    top1_public_score: float | None,
    met_target: bool,
    git_commit: str | None,
) -> None:
    InsightRepository(knowledge_paths, ensure_db=_ensure_db, connect=_connect).record_iteration(
        run_id=run_id,
        iteration=iteration,
        score_source=score_source,
        offline_value=offline_value,
        offline_std=offline_std,
        top1_public_score=top1_public_score,
        met_target=met_target,
        git_commit=git_commit,
    )


def record_improvement(
    *,
    knowledge_paths: KnowledgePaths,
    run_id: str,
    iteration: int,
    summary: str,
    delta_offline: float | None,
) -> None:
    InsightRepository(knowledge_paths, ensure_db=_ensure_db, connect=_connect).record_improvement(
        run_id=run_id,
        iteration=iteration,
        summary=summary,
        delta_offline=delta_offline,
    )


def knowledge_show(knowledge_paths: KnowledgePaths, slug: str) -> dict[str, object]:
    return InsightRepository(knowledge_paths, ensure_db=_ensure_db, connect=_connect).show(slug)


def knowledge_search(
    knowledge_paths: KnowledgePaths,
    tags: Iterable[str],
    limit: int,
) -> list[dict[str, object]]:
    return InsightRepository(knowledge_paths, ensure_db=_ensure_db, connect=_connect).search(tags, limit)


def record_research_artifacts(
    *,
    knowledge_paths: KnowledgePaths,
    slug: str,
    problem_types: Iterable[str],
    research_sources_jsonl: str,
    research_summary_md: str,
) -> dict[str, str]:
    _ensure_db(knowledge_paths)
    normalized_types = _normalize_problem_types(problem_types)
    primary_type = _primary_problem_type(normalized_types)
    safe_slug = _safe_label(slug)
    safe_primary = _safe_label(primary_type)
    base_dir = knowledge_paths.knowledge_dir / "research" / safe_primary / safe_slug
    base_dir.mkdir(parents=True, exist_ok=True)

    sources_path = base_dir / "research_sources.jsonl"
    summary_path = base_dir / "research_summary.md"
    sources_path.write_text(research_sources_jsonl.strip() + "\n", encoding="utf-8")
    summary_path.write_text(research_summary_md.strip() + "\n", encoding="utf-8")

    now = int(time.time())
    sources_rel = str(sources_path.relative_to(knowledge_paths.knowledge_dir))
    summary_rel = str(summary_path.relative_to(knowledge_paths.knowledge_dir))
    problem_types_json = json.dumps(normalized_types, ensure_ascii=True)
    with _connect(knowledge_paths.kb_path) as conn:
        existing = conn.execute(
            "SELECT slug FROM competition_research WHERE slug = ?",
            (slug,),
        ).fetchone()
        if existing is None:
            conn.execute(
                """
                INSERT INTO competition_research (
                    slug, primary_problem_type, problem_types_json,
                    sources_path, summary_path, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    slug,
                    primary_type,
                    problem_types_json,
                    sources_rel,
                    summary_rel,
                    now,
                    now,
                ),
            )
        else:
            conn.execute(
                """
                UPDATE competition_research
                SET primary_problem_type = ?,
                    problem_types_json = ?,
                    sources_path = ?,
                    summary_path = ?,
                    updated_at = ?
                WHERE slug = ?
                """,
                (
                    primary_type,
                    problem_types_json,
                    sources_rel,
                    summary_rel,
                    now,
                    slug,
                ),
            )
        conn.execute("DELETE FROM research_problem_types WHERE slug = ?", (slug,))
        for problem_type in normalized_types:
            conn.execute(
                """
                INSERT OR REPLACE INTO research_problem_types (slug, problem_type)
                VALUES (?, ?)
                """,
                (slug, problem_type),
            )
    return {
        "sources_path": str(sources_path),
        "summary_path": str(summary_path),
        "primary_problem_type": primary_type,
    }


def resolve_research_artifacts(
    *,
    knowledge_paths: KnowledgePaths,
    problem_types: Iterable[str],
    limit: int = 5,
) -> list[dict[str, object]]:
    _ensure_db(knowledge_paths)
    normalized = _normalize_problem_types(problem_types)
    if not normalized:
        return []
    placeholders = ",".join("?" for _ in normalized)
    with _connect(knowledge_paths.kb_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                cr.slug,
                cr.primary_problem_type,
                cr.problem_types_json,
                cr.sources_path,
                cr.summary_path,
                cr.updated_at,
                COUNT(*) AS overlap
            FROM competition_research AS cr
            JOIN research_problem_types AS rpt ON rpt.slug = cr.slug
            WHERE rpt.problem_type IN ({placeholders})
            GROUP BY cr.slug
            ORDER BY overlap DESC, cr.updated_at DESC
            LIMIT ?
            """,
            [*normalized, limit],
        ).fetchall()

    results: list[dict[str, object]] = []
    for row in rows:
        record = dict(row)
        sources_path = knowledge_paths.knowledge_dir / str(record.get("sources_path") or "")
        summary_path = knowledge_paths.knowledge_dir / str(record.get("summary_path") or "")
        problem_types_json = str(record.get("problem_types_json") or "[]")
        record["problem_types"] = parse_json_array_text(problem_types_json) or []
        record["sources_path"] = str(sources_path)
        record["summary_path"] = str(summary_path)
        results.append(record)
    return results


def resolve_research_paths_for_slug(
    *,
    knowledge_paths: KnowledgePaths,
    slug: str,
) -> tuple[Path | None, Path | None]:
    _ensure_db(knowledge_paths)
    with _connect(knowledge_paths.kb_path) as conn:
        row = conn.execute(
            "SELECT sources_path, summary_path FROM competition_research WHERE slug = ?",
            (slug,),
        ).fetchone()
    if row is None:
        return None, None
    sources_rel = str(row["sources_path"])
    summary_rel = str(row["summary_path"])
    sources_path = knowledge_paths.knowledge_dir / sources_rel if sources_rel else None
    summary_path = knowledge_paths.knowledge_dir / summary_rel if summary_rel else None
    return sources_path, summary_path


def format_research_artifacts(research_rows: list[dict[str, object]], *, limit: int = 5) -> str:
    if not research_rows:
        return "No cross-competition research artifacts available."
    lines = ["Cross-competition research artifacts:", ""]
    for row in research_rows[:limit]:
        slug = str(row.get("slug") or "unknown")
        primary = str(row.get("primary_problem_type") or "unknown")
        overlap = row.get("overlap")
        summary_path = str(row.get("summary_path") or "")
        sources_path = str(row.get("sources_path") or "")
        problem_types = row.get("problem_types")
        if not isinstance(problem_types, list):
            problem_types = []
        overlap_text = f", overlap={overlap}" if isinstance(overlap, int) else ""
        lines.append(f"- slug={slug}, class={primary}{overlap_text}, tags={problem_types[:6]}")
        if summary_path:
            lines.append(f"  summary: {summary_path}")
        if sources_path:
            lines.append(f"  sources: {sources_path}")
    return "\n".join(lines)


def _sample_submission_head_file_for_prompt(sample_submission_file: str) -> str:
    name = str(sample_submission_file or "").strip()
    if not name or "*" in name:
        return "sample_submission_head.* or the detected sample-submission preview"
    suffix = strip_compression_suffix(tabular_suffix(Path(name)))
    if suffix in {strip_compression_suffix(candidate) for candidate in TABULAR_TEXT_SUFFIXES}:
        return f"sample_submission_head{suffix}"
    return "sample_submission_head.csv"


def _ensure_db(paths: KnowledgePaths) -> None:
    paths.knowledge_dir.mkdir(parents=True, exist_ok=True)
    with _connect(paths.kb_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS competitions (
                slug TEXT PRIMARY KEY,
                url TEXT,
                metric TEXT,
                task_type TEXT,
                created_at INTEGER,
                last_seen_at INTEGER
            );
            CREATE TABLE IF NOT EXISTS tags (
                tag TEXT PRIMARY KEY
            );
            CREATE TABLE IF NOT EXISTS competition_tags (
                slug TEXT NOT NULL,
                tag TEXT NOT NULL,
                PRIMARY KEY (slug, tag)
            );
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                slug TEXT NOT NULL,
                started_at INTEGER,
                compute TEXT,
                goal_metric TEXT,
                goal_score REAL,
                direction TEXT
            );
            CREATE TABLE IF NOT EXISTS iterations (
                run_id TEXT NOT NULL,
                iter INTEGER NOT NULL,
                score_source TEXT,
                offline_value REAL,
                offline_std REAL,
                top1_public_score REAL,
                met_target INTEGER,
                git_commit TEXT,
                created_at INTEGER,
                PRIMARY KEY (run_id, iter)
            );
            CREATE TABLE IF NOT EXISTS improvements (
                run_id TEXT NOT NULL,
                iter INTEGER NOT NULL,
                summary TEXT,
                delta_offline REAL,
                created_at INTEGER,
                PRIMARY KEY (run_id, iter)
            );
            CREATE TABLE IF NOT EXISTS problem_type_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT NOT NULL,
                run_id TEXT NOT NULL,
                iter INTEGER NOT NULL,
                problem_type TEXT NOT NULL,
                cause_category TEXT NOT NULL,
                fix_category TEXT NOT NULL,
                why_poor TEXT,
                how_improved TEXT,
                delta_offline REAL,
                outcome_bucket TEXT,
                submission_score REAL,
                created_at INTEGER,
                UNIQUE (run_id, iter, problem_type)
            );
            CREATE INDEX IF NOT EXISTS idx_problem_type_insights_problem_type
                ON problem_type_insights(problem_type, created_at DESC);
            CREATE TABLE IF NOT EXISTS error_fix_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT NOT NULL,
                run_id TEXT NOT NULL,
                iter INTEGER NOT NULL,
                problem_type TEXT NOT NULL,
                error_fingerprint TEXT NOT NULL,
                error_category TEXT NOT NULL,
                error_message TEXT,
                fix_summary TEXT,
                resolved INTEGER NOT NULL DEFAULT 0,
                outcome_bucket TEXT,
                submission_score REAL,
                created_at INTEGER,
                UNIQUE (run_id, iter, problem_type, error_fingerprint)
            );
            CREATE INDEX IF NOT EXISTS idx_error_fix_insights_problem_type
                ON error_fix_insights(problem_type, created_at DESC);
            CREATE TABLE IF NOT EXISTS competition_research (
                slug TEXT PRIMARY KEY,
                primary_problem_type TEXT NOT NULL,
                problem_types_json TEXT NOT NULL,
                sources_path TEXT NOT NULL,
                summary_path TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS research_problem_types (
                slug TEXT NOT NULL,
                problem_type TEXT NOT NULL,
                PRIMARY KEY (slug, problem_type)
            );
            CREATE INDEX IF NOT EXISTS idx_research_problem_types_problem_type
                ON research_problem_types(problem_type, slug);
            """
        )
        _ensure_table_column(conn, "problem_type_insights", "outcome_bucket", "TEXT")
        _ensure_table_column(conn, "problem_type_insights", "submission_score", "REAL")
        _ensure_table_column(conn, "error_fix_insights", "resolved", "INTEGER NOT NULL DEFAULT 0")
        _ensure_table_column(conn, "error_fix_insights", "outcome_bucket", "TEXT")
        _ensure_table_column(conn, "error_fix_insights", "submission_score", "REAL")


def _normalize_problem_types(problem_types: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for item in problem_types:
        value = str(item).strip().lower()
        if not value:
            continue
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        normalized.append("unknown")
    return normalized


def _primary_problem_type(problem_types: list[str]) -> str:
    for item in problem_types:
        if item != "unknown":
            return item
    return "unknown"


def _safe_label(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.strip().lower())
    clean = clean.strip("-")
    return clean or "unknown"


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_table_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    existing = {str(row["name"]) for row in rows}
    if column in existing:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def build_kernel_fix_template() -> str:
    """Build the kernel-failure fix prompt template."""
    return """\
# Kagglebot {implementation_agent_name}: Kernel Failure Fix

## Context

**Competition**: `{slug}`
**Run ID**: {run_id}
**Iteration**: {iteration}
**Compute**: {compute} ({accelerator})

## Known Missing Modules (avoid import unless installed)

{blocked_modules}

If one of these modules is required for a strong solution, you may install it with
`uv add <package>` and keep `pyproject.toml` + `uv.lock` updated.

## Error Summary

```
{error_message}
```

## Input Files

- **Kernel Main (authoritative)**: `{kernel_main}`
- **Kernel Script (generated copy; read-only)**: `{kernel_script}`
  Do **NOT** edit this file. It is regenerated on each run and any edits will be discarded.
- **Kernel Logs**: `{logs_dir}` (check for runtime traces)
- **Rules URL**: `{rules_url}`
- **Rules Markdown**: `{rules_md}` (preferred)
- **Overview Markdown**: `{overview_md}` (read this)
- **Data Markdown**: `{data_md}` (read this)
- **Submission Format**: `{submission_format}` (read this if present)
- **Dataset Profile**: `{dataset_profile}`
- **Sample Submission**: `{sample_submission}`

## Your Task

1) Identify the root cause of the kernel failure from logs and traceback.
2) Fix the issue with minimal, targeted changes.
   - Edit **only** the authoritative `kernel.py` (`Kernel Main` above) for competition-specific fixes.
   - Do **NOT** edit anything under `artifacts/<slug>/kernels/` (generated staging directory).
   - Do **NOT** edit the generated `Kernel Script` copy (it will be overwritten).
   - Prefer already-installed dependencies first (torch/timm/torchvision/opencv,
     xgboost/lightgbm/catboost, transformers/tabicl, ultralytics, sklearn).
   - If a required dependency is missing, add it via `uv add <package>` and keep
     `pyproject.toml` and `uv.lock` updated together.
   - If one dependency is missing, guard/disable only that path; do not globally downgrade
     all pipelines to a weak fallback.
   - If the failure is in the **runner/validators/CLI plumbing** (e.g., kernel packaging,
     path injection, validation, or Kaggle CLI status/output handling), you may edit **src/**
     to fix the root cause.
   - Use `custom_main()` when data is non-CSV or default loader is inadequate.
   - If the error mentions `decoder_input_ids` or `decoder_inputs_embeds`, you likely loaded a T5/ProtT5
     encoder-decoder with `AutoModel`. Use `T5EncoderModel` or `model.get_encoder()` and pass only
     `input_ids`/`attention_mask`.
3) Do NOT introduce secrets or network side-effects.
4) Keep the change small and explain what you changed in your response.

The autopilot will re-push the kernel after your fix.
"""
