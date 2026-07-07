from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from kagglebot.analyzer.schema import infer_schema, load_schema_frames
from kagglebot.analyzer.strategy import build_strategy
from kagglebot.analyzer.types import CompetitionMetadata, CompetitionSchema
from kagglebot.asset_modality import infer_asset_modality
from kagglebot.json_utils import write_json_object
from kagglebot.knowledge import build_dataset_profile
from kagglebot.paths import CompetitionPaths
from kagglebot.rna_structure import detect_rna_structure_task, load_rna_structure_task
from kagglebot.solver.io import (
    CompetitionData,
    find_competition_files,
    load_competition_data,
    looks_like_natural_language_text_target,
    task_for_prediction_kind,
)
from kagglebot.submission_sample_discovery import is_tabular_data_path, select_sample_submission_path


class UnsupportedCompetitionError(RuntimeError):
    pass


@dataclass(frozen=True)
class AnalysisResult:
    metadata: CompetitionMetadata
    analysis_path: Path


def analyze_competition(
    *,
    slug: str,
    paths: CompetitionPaths,
    time_budget_minutes: int,
    cv_folds: int,
    models: list[str] | None,
    use_stacking: bool,
) -> AnalysisResult:
    if detect_rna_structure_task(paths.data_raw):
        return _analyze_rna_structure_task(
            slug=slug,
            paths=paths,
            time_budget_minutes=time_budget_minutes,
            cv_folds=cv_folds,
            models=models,
            use_stacking=use_stacking,
        )
    try:
        train_path, test_path, sample_path = _find_required(paths)
    except UnsupportedCompetitionError as exc:
        return _analyze_non_tabular_fallback(
            slug=slug,
            paths=paths,
            reason=str(exc),
            time_budget_minutes=time_budget_minutes,
            cv_folds=cv_folds,
            models=models,
            use_stacking=use_stacking,
        )
    solver_data = load_competition_data(paths.data_raw)
    frames = load_schema_frames(train_path, test_path, sample_path)
    schema_fallback_reason: str | None = None
    try:
        schema = infer_schema(
            frames=frames,
            train_path=train_path,
            test_path=test_path,
            sample_path=sample_path,
        )
    except ValueError as exc:
        if not _solver_data_supports_schema_fallback(solver_data):
            raise
        schema = _schema_from_solver_data(
            data=solver_data,
            train_path=train_path,
            test_path=test_path,
            sample_path=sample_path,
        )
        schema_fallback_reason = str(exc)
    dataset_profile = _safe_dataset_profile(paths.data_raw)

    target_col = solver_data.target_column
    raw_task = solver_data.task_by_target.get(target_col, solver_data.task)
    prediction_kind = solver_data.prediction_kind_by_target.get(target_col, solver_data.prediction_kind)
    task = (
        solver_data.task
        if solver_data.target_column == target_col
        else task_for_prediction_kind(raw_task, prediction_kind)
    )
    if _looks_like_survival_single_score_layout(solver_data):
        task = "survival"
        prediction_kind = "risk_score"
    elif _looks_like_multi_label_layout(solver_data):
        task = "multi_label"
    elif _looks_like_segmentation_submission(solver_data):
        task = "segmentation"
        prediction_kind = "rle"
    elif _looks_like_object_detection_submission(solver_data):
        task = "object_detection"
        prediction_kind = "prediction_string"
    elif _looks_like_learning_to_rank_layout(solver_data):
        task = "learning_to_rank"
        prediction_kind = "ranking_score"
    elif _looks_like_forecasting_layout(solver_data):
        task = "forecasting"
        prediction_kind = "continuous"
    elif _looks_like_count_regression_layout(solver_data):
        task = "count_regression"
        prediction_kind = "continuous"
    elif _looks_like_bounded_regression_layout(solver_data):
        task = "bounded_regression"
        prediction_kind = "continuous"
    elif _looks_like_positive_skew_regression_layout(solver_data):
        task = "positive_skew_regression"
        prediction_kind = "continuous"
    elif _looks_like_pairwise_layout(solver_data):
        task = "pairwise"
        if len(solver_data.target_columns) > 1:
            prediction_kind = "probability_columns"
    elif _looks_like_multi_label_indicator_layout(solver_data):
        task = "multi_label"
        prediction_kind = "multi_label_columns"
    elif _looks_like_recommender_layout(solver_data):
        task = "recommender"
        prediction_kind = "continuous"
    elif _looks_like_ctr_layout(solver_data):
        task = "ctr"
        prediction_kind = "probability"
    elif _looks_like_coordinate_regression_layout(solver_data):
        task = "coordinate_regression"
        prediction_kind = "coordinate_columns"
    elif (multi_target_layout := _multi_target_layout(solver_data)) is not None:
        task, prediction_kind = multi_target_layout
    elif task != "text" and (domain_layout := _domain_modality_layout(dataset_profile)) is not None:
        task, prediction_kind = domain_layout
    metric, metric_direction = _default_metric(task, prediction_kind=prediction_kind)

    strategy = build_strategy(
        task,
        prediction_kind=prediction_kind,
        time_budget_minutes=time_budget_minutes,
        cv_folds=cv_folds,
        models=models,
        use_stacking=use_stacking,
    )

    assumptions = ["tabular competition inferred from local train/test/sample files"]
    if len(schema.target_columns) == 1:
        assumptions.append("single target column inferred from sample submission")
    else:
        assumptions.append(
            f"multi-target submission detected ({len(schema.target_columns)} targets); "
            f"primary target for baseline is '{target_col}'"
        )
    if schema.id_column is None:
        assumptions.append("id column not clearly detected; submission alignment defaults to row order")
    if schema_fallback_reason is not None:
        fallback_kind = _schema_fallback_kind(solver_data)
        assumptions.append(
            f"schema inferred from solver {fallback_kind} layout after schema fallback: {schema_fallback_reason}"
        )
    if task == "survival":
        assumptions.append("survival event/time targets inferred and mapped to single risk-score submission")
    elif task == "multi_label":
        if prediction_kind == "multi_label_columns" and len(solver_data.target_columns) > 1:
            assumptions.append("binary indicator multi-label target columns inferred for one-vs-rest label prediction")
        else:
            assumptions.append("delimiter-based multi-label target inferred for label-set prediction")
    elif task == "segmentation":
        assumptions.append("segmentation mask/RLE submission inferred from sample_submission columns")
    elif task == "object_detection":
        assumptions.append("object-detection prediction-string submission inferred from sample_submission columns")
    elif task == "learning_to_rank":
        assumptions.append("query/document relevance layout inferred for grouped learning-to-rank scoring")
    elif task == "forecasting":
        assumptions.append(f"future temporal holdout inferred from time column '{solver_data.time_column}'")
    elif task == "count_regression":
        assumptions.append("non-negative integer count target inferred for RMSLE/Poisson-style regression planning")
    elif task == "bounded_regression":
        assumptions.append("bounded rate/ratio target inferred for clipped continuous regression planning")
    elif task == "positive_skew_regression":
        assumptions.append("positive skew non-negative target inferred for RMSLE/log1p regression planning")
    elif task == "pairwise":
        assumptions.append("paired entity matchup layout inferred for pairwise probability/ranking calibration")
    elif task == "ctr":
        assumptions.append("user-item click/conversion layout inferred for calibrated CTR scoring")
    elif task == "recommender":
        assumptions.append("user-item rating/relevance layout inferred for recommender-style scoring")
    elif task == "coordinate_regression":
        assumptions.append("coordinate target columns inferred for structured numeric coordinate regression")
    elif task == "multi_output_regression":
        assumptions.append("multi-output regression layout inferred from multiple numeric target columns")
    elif task == "multi_target_classification":
        assumptions.append("multi-target classification layout inferred from multiple class target columns")
    elif task == "multi_task":
        assumptions.append("mixed multi-task target layout inferred from heterogeneous target columns")
    elif task in _DOMAIN_MODALITY_TASKS:
        assumptions.append(f"{task} modality inferred from dataset profile and preserved for domain-specific planning")
    elif task in _TEXT_FEATURE_TASKS:
        assumptions.append("text feature modality inferred from dataset profile and preserved for NLP planning")
    elif prediction_kind == "probability":
        assumptions.append("probability submission inferred from solver target/sample metadata")
    elif prediction_kind == "probability_columns":
        assumptions.append("class-probability submission inferred from multiple numeric sample_submission columns")
    elif prediction_kind == "text":
        assumptions.append("text submission inferred from sample_submission target column")
    elif prediction_kind == "quantile_columns":
        assumptions.append(
            "quantile submission columns inferred from sample_submission and mapped to the training target"
        )
    elif prediction_kind == "prediction_interval_columns":
        assumptions.append(
            "prediction-interval submission columns inferred from sample_submission and mapped to the training target"
        )

    metadata = CompetitionMetadata(
        slug=slug,
        competition_type="tabular",
        task=task,
        metric=metric,
        metric_direction=metric_direction,
        prediction_kind=prediction_kind,
        schema=schema,
        strategy=strategy,
        assumptions=assumptions,
    )

    analysis_path = paths.analysis_path
    payload = metadata.to_dict()
    payload["generated_at"] = datetime.now(UTC).isoformat()
    write_json_object(analysis_path, payload)

    return AnalysisResult(metadata=metadata, analysis_path=analysis_path)


def _analyze_rna_structure_task(
    *,
    slug: str,
    paths: CompetitionPaths,
    time_budget_minutes: int,
    cv_folds: int,
    models: list[str] | None,
    use_stacking: bool,
) -> AnalysisResult:
    task = load_rna_structure_task(paths.data_raw)
    feature_columns = [str(col) for col in task.train_sequences.columns if col in task.test_sequences.columns]
    numeric_columns = [col for col in feature_columns if pd.api.types.is_numeric_dtype(task.train_sequences[col])]
    categorical_columns = [col for col in feature_columns if col not in numeric_columns]
    schema = CompetitionSchema(
        train_path=task.files.train_sequences_path,
        test_path=task.files.test_sequences_path,
        sample_submission_path=task.files.sample_submission_path,
        id_column=task.sample_id_column,
        target_columns=task.sample_coordinate_columns,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=[],
    )
    strategy = build_strategy(
        "rna_structure",
        prediction_kind=task.target_kind,
        time_budget_minutes=time_budget_minutes,
        cv_folds=cv_folds,
        models=models,
        use_stacking=use_stacking,
    )
    metadata = CompetitionMetadata(
        slug=slug,
        competition_type="rna_structure",
        task="rna_structure",
        metric="rmse",
        metric_direction="minimize",
        prediction_kind=task.target_kind,
        schema=schema,
        strategy=strategy,
        assumptions=[
            "RNA sequence/structure layout inferred from train/test sequence tables and coordinate label tables",
            f"sequence id column resolved as '{task.sequence_id_column}'",
            f"sample residue id column resolved as '{task.sample_id_column}'",
            f"coordinate submission uses {len(task.sample_coordinate_triplets)} x/y/z triplet(s)",
            "residue anchor columns preserved for coordinate submission validation",
        ],
    )
    analysis_path = paths.analysis_path
    payload = metadata.to_dict()
    payload["generated_at"] = datetime.now(UTC).isoformat()
    payload["rna_structure"] = {
        "train_labels_path": str(task.files.train_labels_path),
        "sequence_id_column": task.sequence_id_column,
        "sequence_column": task.sequence_column,
        "label_id_column": task.label_id_column,
        "sample_anchor_columns": task.sample_anchor_columns,
        "label_anchor_columns": task.label_anchor_columns,
        "coordinate_triplets": len(task.sample_coordinate_triplets),
    }
    write_json_object(analysis_path, payload)
    return AnalysisResult(metadata=metadata, analysis_path=analysis_path)


def _analyze_non_tabular_fallback(
    *,
    slug: str,
    paths: CompetitionPaths,
    reason: str,
    time_budget_minutes: int,
    cv_folds: int,
    models: list[str] | None,
    use_stacking: bool,
) -> AnalysisResult:
    modality = infer_asset_modality(paths.data_raw, include_code_artifact=True)
    task = modality if modality != "unknown" else "unknown"
    strategy = build_strategy(
        task,
        prediction_kind="artifact",
        time_budget_minutes=time_budget_minutes,
        cv_folds=cv_folds,
        models=models,
        use_stacking=use_stacking,
    )
    sample_submission_path = _resolve_non_tabular_sample_submission_path(paths)
    schema = CompetitionSchema(
        train_path=paths.data_raw,
        test_path=paths.data_raw,
        sample_submission_path=sample_submission_path,
        id_column=None,
        target_columns=[],
        feature_columns=[],
        numeric_columns=[],
        categorical_columns=[],
        datetime_columns=[],
    )
    metadata = CompetitionMetadata(
        slug=slug,
        competition_type=modality,
        task=task,
        metric="unknown",
        metric_direction="maximize",
        prediction_kind="artifact",
        schema=schema,
        strategy=strategy,
        assumptions=[
            "tabular train/test/sample files were not available; using non-tabular asset fallback analysis",
            f"required tabular discovery failed: {reason}",
            f"asset modality inferred as '{modality}' from local files",
            f"sample submission/template path resolved as '{sample_submission_path.name}'",
        ],
    )
    analysis_path = paths.analysis_path
    payload = metadata.to_dict()
    payload["generated_at"] = datetime.now(UTC).isoformat()
    write_json_object(analysis_path, payload)
    return AnalysisResult(metadata=metadata, analysis_path=analysis_path)


def _find_required(paths: CompetitionPaths) -> tuple[Path, Path, Path]:
    try:
        return find_competition_files(paths.data_raw)
    except FileNotFoundError as exc:
        raise UnsupportedCompetitionError(str(exc)) from exc


def _resolve_non_tabular_sample_submission_path(paths: CompetitionPaths) -> Path:
    """Return the best available sample/template path for non-tabular fallback analysis."""
    data_root = paths.data_raw
    if data_root.exists():
        try:
            files = [path for path in data_root.rglob("*") if path.is_file() and is_tabular_data_path(path)]
        except OSError:
            files = []
        candidate = select_sample_submission_path(files)
        if candidate is not None:
            return candidate
    return paths.sample_submission_path


def _safe_dataset_profile(data_dir: Path) -> dict[str, object]:
    try:
        profile = build_dataset_profile(data_dir)
    except Exception:  # noqa: BLE001
        return {}
    return profile if isinstance(profile, dict) else {}


_DOMAIN_MODALITY_TASKS = {
    "image",
    "medical_imaging",
    "audio",
    "video",
    "signal",
    "array",
    "point_cloud",
    "geospatial",
    "graph",
    "annotation",
    "bio",
    "rna",
    "multimodal",
}
_TEXT_FEATURE_TASKS = {"text_classification", "text_regression"}


def _domain_modality_layout(profile: dict[str, object]) -> tuple[str, str] | None:
    modality = str(profile.get("modality") or "").strip().lower()
    if modality == "text":
        return _text_feature_modality_layout(profile)
    if modality not in _DOMAIN_MODALITY_TASKS:
        return None
    target_semantics = str(profile.get("target_semantics") or "").strip().lower()
    if target_semantics not in {
        "",
        "classification",
        "regression",
        "ordinal_classification",
        "text_generation",
    }:
        return None
    prediction_kinds = profile.get("prediction_kind_by_target")
    prediction_kind = ""
    if isinstance(prediction_kinds, dict):
        target_column = str(profile.get("target_column") or "")
        value = prediction_kinds.get(target_column) if target_column else None
        if value is None and prediction_kinds:
            value = next(iter(prediction_kinds.values()))
        prediction_kind = str(value or "").strip().lower()
    if not prediction_kind:
        if target_semantics == "text_generation":
            prediction_kind = "text"
        elif str(profile.get("task") or "").strip().lower() == "regression":
            prediction_kind = "continuous"
        else:
            prediction_kind = "class"
    return modality, prediction_kind


def _text_feature_modality_layout(profile: dict[str, object]) -> tuple[str, str] | None:
    target_semantics = str(profile.get("target_semantics") or "").strip().lower()
    if target_semantics not in {"", "classification", "regression", "ordinal_classification"}:
        return None
    prediction_kind = _profile_primary_prediction_kind(profile)
    profile_task = str(profile.get("task") or "").strip().lower()
    if profile_task == "regression" and prediction_kind == "probability":
        prediction_kind = "continuous"
    if not prediction_kind:
        prediction_kind = "continuous" if profile_task == "regression" else "class"
    task = "text_regression" if prediction_kind == "continuous" else "text_classification"
    return task, prediction_kind


def _profile_primary_prediction_kind(profile: dict[str, object]) -> str:
    prediction_kinds = profile.get("prediction_kind_by_target")
    if not isinstance(prediction_kinds, dict):
        return ""
    target_column = str(profile.get("target_column") or "")
    value = prediction_kinds.get(target_column) if target_column else None
    if value is None and prediction_kinds:
        value = next(iter(prediction_kinds.values()))
    return str(value or "").strip().lower()


def _solver_data_supports_schema_fallback(data: CompetitionData) -> bool:
    if not (data.target_columns and data.feature_columns):
        return False
    if "unsupervised" in set(data.task_by_target.values()):
        return True
    return data.prediction_kind in {"quantile_columns", "prediction_interval_columns", "continuous_columns"}


def _schema_fallback_kind(data: CompetitionData) -> str:
    if "unsupervised" in set(data.task_by_target.values()):
        return "no-label"
    if data.prediction_kind == "quantile_columns":
        return "quantile-submission"
    if data.prediction_kind == "prediction_interval_columns":
        return "prediction-interval"
    if data.prediction_kind == "continuous_columns":
        return "expanded-continuous-submission"
    return "resolved"


def _schema_from_solver_data(
    *,
    data: CompetitionData,
    train_path: Path,
    test_path: Path,
    sample_path: Path,
) -> CompetitionSchema:
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    datetime_columns: list[str] = []
    for col in data.feature_columns:
        if col not in data.train.columns:
            continue
        series = data.train[col]
        if _looks_datetime(series):
            datetime_columns.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(col)
        else:
            categorical_columns.append(col)
    return CompetitionSchema(
        train_path=train_path,
        test_path=test_path,
        sample_submission_path=sample_path,
        id_column=data.id_column,
        target_columns=list(data.target_columns),
        feature_columns=list(data.feature_columns),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
    )


def _looks_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    values = series.dropna()
    if values.empty:
        return False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)
        parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return bool(parsed.notna().mean() >= 0.8)


def _looks_like_survival_single_score_layout(data: CompetitionData) -> bool:
    if not _survival_event_time_columns(data.target_columns):
        return False
    prediction_cols = [str(col) for col in data.sample.columns if col != data.id_column]
    if len(prediction_cols) != 1:
        return False
    score_col = prediction_cols[0]
    return score_col not in data.target_columns and pd.api.types.is_numeric_dtype(data.sample[score_col])


def _looks_like_multi_label_layout(data: CompetitionData) -> bool:
    target_col = data.target_column
    if target_col not in data.train.columns:
        return False
    prediction_kind = data.prediction_kind_by_target.get(target_col, data.prediction_kind)
    if prediction_kind == "multi_label_columns":
        return True
    if prediction_kind != "text":
        return False
    return _looks_like_multi_label_target(data.train[target_col], column_name=target_col)


def _looks_like_multi_label_target(target: pd.Series, *, column_name: str) -> bool:
    if not (pd.api.types.is_object_dtype(target) or pd.api.types.is_string_dtype(target)):
        return False
    tokens = set(_column_tokens(column_name))
    compact = "".join(_column_tokens(column_name))
    strong_name = bool(tokens & {"labels", "tags", "classes", "categories"}) or "multilabel" in compact
    generic_name = strong_name or bool(tokens & {"label", "target", "class", "category"})
    if not generic_name:
        return False
    values = target.dropna().astype(str).str.strip().head(500)
    values = values[values != ""]
    if values.empty:
        return False

    multi_count = 0
    atomic_labels: set[str] = set()
    for value in values:
        labels = _split_multi_label_value(value, allow_whitespace=strong_name)
        if len(labels) < 2:
            continue
        multi_count += 1
        atomic_labels.update(labels)
    if float(multi_count / len(values)) < 0.6:
        return False
    return len(atomic_labels) >= 2


def _split_multi_label_value(value: str, *, allow_whitespace: bool) -> list[str]:
    raw = value.strip()
    if not raw:
        return []
    if any(sep in raw for sep in ("|", ";", ",")):
        parts = re.split(r"[|;,]+", raw)
    elif allow_whitespace:
        parts = raw.split()
    else:
        return [raw]
    labels = [part.strip().lower() for part in parts if part.strip()]
    return [label for label in labels if re.fullmatch(r"[a-z0-9_.:+-]+", label)]


def _looks_like_segmentation_submission(data: CompetitionData) -> bool:
    prediction_compacts = _sample_prediction_compacts(data)
    return bool(
        prediction_compacts
        & {
            "encodedpixels",
            "rle",
            "runlengthencoding",
            "mask",
            "masks",
            "segmentation",
            "segmentationmask",
            "maskrle",
        }
    )


def _looks_like_object_detection_submission(data: CompetitionData) -> bool:
    prediction_compacts = _sample_prediction_compacts(data)
    return bool(
        prediction_compacts
        & {
            "predictionstring",
            "predstring",
            "detections",
            "detectionstring",
            "bbox",
            "bboxes",
            "boxes",
        }
    )


def _sample_prediction_compacts(data: CompetitionData) -> set[str]:
    return {_compact_name(col) for col in data.sample.columns if col != data.id_column}


def _looks_like_learning_to_rank_layout(data: CompetitionData) -> bool:
    target_col = data.target_column
    compact_target = _compact_name(target_col)
    if compact_target not in {"relevance", "relevancescore", "rank", "ranking", "score", "rankscore"}:
        return False
    if target_col not in data.train.columns or not pd.api.types.is_numeric_dtype(data.train[target_col]):
        return False

    feature_compacts = {_compact_name(col) for col in data.feature_columns}
    has_query_feature = bool(feature_compacts & {"queryid", "qid", "searchid", "requestid", "sessionid"})
    has_item_feature = bool(
        feature_compacts & {"documentid", "docid", "candidateid", "itemid", "productid", "passageid"}
    )
    if has_query_feature and has_item_feature:
        return True

    train_cols = {_compact_name(col) for col in data.train.columns}
    test_cols = {_compact_name(col) for col in data.test.columns}
    common_cols = train_cols & test_cols
    return bool(
        common_cols & {"queryid", "qid", "searchid", "requestid"}
        and common_cols & {"documentid", "docid", "candidateid", "itemid", "passageid"}
    )


def _looks_like_forecasting_layout(data: CompetitionData) -> bool:
    if not data.time_column:
        return False
    target_col = data.target_column
    if target_col not in data.train.columns:
        return False
    if not pd.api.types.is_numeric_dtype(data.train[target_col]):
        return False
    return data.task_by_target.get(target_col, data.task) == "regression"


def _looks_like_count_regression_layout(data: CompetitionData) -> bool:
    if len(data.target_columns) != 1:
        return False
    target_col = data.target_column
    if target_col not in data.train.columns:
        return False
    if _has_user_item_feature_signal(data.feature_columns) and _is_ctr_target_column(target_col):
        return False
    if data.task_by_target.get(target_col, data.task) != "regression":
        return False
    prediction_kind = data.prediction_kind_by_target.get(target_col, data.prediction_kind)
    if prediction_kind != "continuous":
        return False
    return _looks_like_count_regression_target(data.train[target_col], column_name=target_col)


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
    if values.empty or (values < 0).any():
        return False
    integer_like = ((values % 1).abs() < 1e-9).mean()
    return bool(float(integer_like) >= 0.95 and int(values.nunique(dropna=True)) >= 3)


def _looks_like_bounded_regression_layout(data: CompetitionData) -> bool:
    if len(data.target_columns) != 1:
        return False
    target_col = data.target_column
    if target_col not in data.train.columns:
        return False
    if data.task_by_target.get(target_col, data.task) != "regression":
        return False
    prediction_kind = data.prediction_kind_by_target.get(target_col, data.prediction_kind)
    if prediction_kind != "continuous":
        return False
    return _looks_like_bounded_regression_target(data.train[target_col], column_name=target_col)


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


def _looks_like_positive_skew_regression_layout(data: CompetitionData) -> bool:
    if len(data.target_columns) != 1:
        return False
    target_col = data.target_column
    if target_col not in data.train.columns:
        return False
    if data.task_by_target.get(target_col, data.task) != "regression":
        return False
    prediction_kind = data.prediction_kind_by_target.get(target_col, data.prediction_kind)
    if prediction_kind != "continuous":
        return False
    return _looks_like_positive_skew_regression_target(data.train[target_col], column_name=target_col)


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
    if len(values) < 8 or (values < 0).any() or int(values.nunique(dropna=True)) < 5:
        return False
    median = float(values.median())
    if median <= 0.0:
        return False
    skew = float(values.skew())
    if pd.isna(skew):
        return False
    return bool(skew >= 1.0 and float(values.max()) / median >= 5.0)


def _looks_like_pairwise_layout(data: CompetitionData) -> bool:
    feature_compacts = {_compact_name(col) for col in data.feature_columns}
    pair_groups = (
        ("team1", "team2"),
        ("teama", "teamb"),
        ("hometeam", "awayteam"),
        ("player1", "player2"),
        ("playera", "playerb"),
        ("item1", "item2"),
        ("itema", "itemb"),
        ("modela", "modelb"),
        ("user1", "user2"),
        ("usera", "userb"),
    )
    if any(left in feature_compacts and right in feature_compacts for left, right in pair_groups):
        return True
    prefixes: dict[str, set[str]] = {}
    for compact in feature_compacts:
        match = re.match(r"(.+?)(?:id)?([12ab])$", compact)
        if not match:
            continue
        prefix, side = match.groups()
        if prefix in {"team", "player", "item", "model", "user", "entity", "candidate"}:
            prefixes.setdefault(prefix, set()).add(side)
    return any({"1", "2"}.issubset(sides) or {"a", "b"}.issubset(sides) for sides in prefixes.values())


def _looks_like_ctr_layout(data: CompetitionData) -> bool:
    if not _has_user_item_feature_signal(data.feature_columns):
        return False
    target_col = data.target_column
    if _is_recommender_score_target_column(target_col):
        return False
    raw_task = data.task_by_target.get(target_col, data.task)
    prediction_kind = data.prediction_kind_by_target.get(target_col, data.prediction_kind)
    if _is_ctr_target_column(target_col):
        return True
    return raw_task == "classification" or prediction_kind == "probability"


def _looks_like_recommender_layout(data: CompetitionData) -> bool:
    if not _has_user_item_feature_signal(data.feature_columns):
        return False
    target_col = data.target_column
    if _is_recommender_score_target_column(target_col):
        return True
    raw_task = data.task_by_target.get(target_col, data.task)
    prediction_kind = data.prediction_kind_by_target.get(target_col, data.prediction_kind)
    return raw_task == "regression" and prediction_kind in {"continuous", "ordinal"}


def _looks_like_multi_label_indicator_layout(data: CompetitionData) -> bool:
    target_columns = [str(col) for col in data.target_columns if col in data.train.columns]
    if len(target_columns) < 3 or len(target_columns) != len(data.target_columns):
        return False
    if _survival_event_time_columns(target_columns):
        return False
    if not all(data.task_by_target.get(col, data.task) == "classification" for col in target_columns):
        return False
    if not all(_looks_like_binary_indicator_column(data.train[col]) for col in target_columns):
        return False
    sample_targets = [col for col in target_columns if col in data.sample.columns]
    if len(sample_targets) != len(target_columns):
        return False
    return all(pd.api.types.is_numeric_dtype(data.sample[col]) for col in sample_targets)


def _looks_like_binary_indicator_column(series: pd.Series) -> bool:
    values = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    if values.empty:
        return False
    if len(values) != len(series.dropna()):
        return False
    unique = set(values.unique().tolist())
    return bool(unique) and unique <= {0, 1, 0.0, 1.0}


def _looks_like_coordinate_regression_layout(data: CompetitionData) -> bool:
    target_columns = [str(col) for col in data.target_columns if col in data.train.columns]
    if len(target_columns) < 2 or len(target_columns) != len(data.target_columns):
        return False
    if not all(data.task_by_target.get(col, data.task) == "regression" for col in target_columns):
        return False
    if not all(pd.api.types.is_numeric_dtype(data.train[col]) for col in target_columns):
        return False
    axes = {_coordinate_axis_for_column(col) for col in target_columns}
    axes.discard(None)
    if {"lat", "lon"}.issubset(axes):
        return True
    return {"x", "y"}.issubset(axes)


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


def _multi_target_layout(data: CompetitionData) -> tuple[str, str] | None:
    target_columns = [col for col in data.target_columns if col in data.train.columns]
    if len(target_columns) <= 1 or len(target_columns) != len(data.target_columns):
        return None
    raw_tasks = {data.task_by_target.get(col, data.task) for col in target_columns}
    prediction_kinds = {data.prediction_kind_by_target.get(col, data.prediction_kind) for col in target_columns}
    if raw_tasks == {"regression"}:
        return "multi_output_regression", "continuous_columns"
    if raw_tasks == {"classification"}:
        if prediction_kinds <= {"class", "probability"}:
            return "multi_target_classification", "target_columns"
        if prediction_kinds <= {"class", "probability", "probability_columns"}:
            return "multi_target_classification", "probability_columns"
    return "multi_task", "target_columns"


def _has_user_item_feature_signal(feature_columns: list[str]) -> bool:
    has_user = any(_is_user_entity_column(str(col)) for col in feature_columns)
    has_item = any(_is_item_entity_column(str(col)) for col in feature_columns)
    return has_user and has_item


def _is_user_entity_column(name: str) -> bool:
    return _column_matches_entity_terms(
        name,
        {"user", "customer", "member", "client", "account", "visitor", "viewer", "shopper", "subscriber"},
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
    return any(compact.startswith(term) and compact.endswith(("id", "idx", "uuid", "code", "key")) for term in terms)


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


def _column_tokens(name: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", name.lower()) if token]


def _survival_event_time_columns(target_columns: list[str]) -> bool:
    has_event = any(_is_survival_event_column(col) for col in target_columns)
    has_time = any(_is_survival_time_column(col) for col in target_columns)
    return has_event and has_time


def _is_survival_event_column(name: str) -> bool:
    compact = _compact_name(name)
    return compact in {"event", "eventobserved", "observed", "status", "efs", "censor", "censored", "death", "dead"}


def _is_survival_time_column(name: str) -> bool:
    compact = _compact_name(name)
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


def _compact_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _infer_task(target: pd.Series) -> str:
    if pd.api.types.is_object_dtype(target) or isinstance(target.dtype, pd.CategoricalDtype):
        return "classification"
    if not pd.api.types.is_numeric_dtype(target):
        return "classification"
    nunique = target.nunique(dropna=True)
    if nunique <= 20:
        return "classification"
    if nunique / max(len(target), 1) <= 0.05:
        return "classification"
    return "regression"


def _default_metric(task: str, *, prediction_kind: str) -> tuple[str, str]:
    if task == "coordinate_regression":
        return "rmse", "minimize"
    if task == "count_regression":
        return "rmsle", "minimize"
    if task == "bounded_regression":
        return "rmse", "minimize"
    if task == "positive_skew_regression":
        return "rmsle", "minimize"
    if task == "multi_output_regression":
        return "mcrmse", "minimize"
    if task == "multi_target_classification":
        return "f1", "maximize"
    if task == "multi_task":
        return "mean_target_metric", "maximize"
    if task == "multi_label":
        return "f1", "maximize"
    if task == "text_classification":
        if prediction_kind in {"probability", "probability_columns"}:
            return "logloss", "minimize"
        return "accuracy", "maximize"
    if task == "text_regression":
        return "rmse", "minimize"
    if task in _DOMAIN_MODALITY_TASKS:
        if prediction_kind in {"probability", "probability_columns"}:
            return "logloss", "minimize"
        if prediction_kind == "text":
            return "text_similarity", "maximize"
        if prediction_kind == "ordinal":
            return "quadratic_weighted_kappa", "maximize"
        if prediction_kind == "class":
            return "accuracy", "maximize"
        return "rmse", "minimize"
    if prediction_kind == "text" or task == "text":
        return "text_similarity", "maximize"
    if task == "survival":
        return "concordance_index", "maximize"
    if task == "segmentation":
        return "dice", "maximize"
    if task == "object_detection":
        return "map", "maximize"
    if task == "learning_to_rank":
        return "ndcg", "maximize"
    if task == "forecasting":
        return "rmse", "minimize"
    if task == "pairwise":
        if prediction_kind in {"probability", "probability_columns"}:
            return "logloss", "minimize"
        return "accuracy", "maximize"
    if task == "ctr":
        return "logloss", "minimize"
    if task == "recommender":
        return "rmse", "minimize"
    if prediction_kind == "quantile_columns":
        return "pinball_loss", "minimize"
    if prediction_kind == "prediction_interval_columns":
        return "interval_score", "minimize"
    if prediction_kind == "ordinal":
        return "quadratic_weighted_kappa", "maximize"
    if task == "unsupervised":
        return "auc", "maximize"
    if task == "classification":
        if prediction_kind in {"probability", "probability_columns"}:
            return "logloss", "minimize"
        return "accuracy", "maximize"
    return "rmse", "minimize"


def _infer_prediction_kind(
    task: str,
    *,
    sample: pd.DataFrame,
    target_series: pd.Series,
    id_column: str | None,
    target_column: str,
) -> str:
    if target_column not in sample.columns:
        prediction_columns = [col for col in sample.columns if col != id_column]
        if (
            task == "classification"
            and len(prediction_columns) >= 2
            and all(pd.api.types.is_numeric_dtype(sample[col]) for col in prediction_columns)
        ):
            return "probability_columns"
        return "continuous"
    sample_target = sample[target_column]
    if task != "classification":
        return "continuous"
    from kagglebot.solver.io import infer_prediction_kind

    inferred = infer_prediction_kind(sample_target, column_name=target_column)
    if inferred == "text":
        return "text"
    if inferred == "class" and looks_like_natural_language_text_target(target_series):
        return "text"
    if pd.api.types.is_float_dtype(sample_target):
        return "probability"
    return "class"
