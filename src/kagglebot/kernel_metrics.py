from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from statistics import stdev

from kagglebot.json_utils import load_json_object, write_json_object
from kagglebot.metric_matching import normalize_metric_name
from kagglebot.scalar_utils import tolerant_finite_float
from kagglebot.score_sources import (
    is_trusted_offline_score_source,
    normalize_score_source_name,
    validate_grouped_oof_contract,
)
from kagglebot.solver.evaluate import EvaluationResult
from kagglebot.solver.io import read_table
from kagglebot.solver.metrics import canonical_metric, compute_metric, infer_direction, metric_requires_proba

_METRIC_ASSIGNMENT_NAME_SUFFIXES = (
    "quadratic_weighted_kappa",
    "weighted_kappa",
    "concordance_index",
    "interval_score",
    "pinball_loss",
    "brier_score",
    "log_loss",
    "r2_score",
    "accuracy",
    "precision",
    "spearman",
    "pearson",
    "recall",
    "logloss",
    "mcrmse",
    "rmsle",
    "smape",
    "kappa",
    "score",
    "brier",
    "cindex",
    "gini",
    "loss",
    "ndcg",
    "mape",
    "rmse",
    "mae",
    "mse",
    "aurc",
    "auc",
    "qwk",
    "map",
    "acc",
    "r2",
    "f1",
)
_METRIC_ASSIGNMENT_NAME_SUFFIX_RE = "|".join(re.escape(suffix) for suffix in _METRIC_ASSIGNMENT_NAME_SUFFIXES)
_BASELINE_SCORE_ASSIGNMENT_RE = re.compile(
    rf"\b(?P<name>[a-z_][a-z0-9_]*?(?:{_METRIC_ASSIGNMENT_NAME_SUFFIX_RE}))\s*=\s*"
    r"(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    flags=re.IGNORECASE,
)


def extract_trusted_cv_value_from_metrics_payload(payload: dict[str, object]) -> float | None:
    """Extract a CV-based fallback score from metrics payload when reported source is untrusted."""
    for key in (
        "cv_brier",
        "cv_score",
        "cv_mean",
        "selected_cv_mean",
        "best_cv",
        "oof_score",
        "oof_metric",
        "oof_brier",
    ):
        parsed = tolerant_finite_float(payload.get(key))
        if parsed is not None:
            return float(parsed)

    fold_scores_raw = payload.get("fold_scores")
    if isinstance(fold_scores_raw, list):
        fold_scores = [float(parsed) for item in fold_scores_raw if (parsed := tolerant_finite_float(item)) is not None]
        if fold_scores:
            return float(sum(fold_scores) / len(fold_scores))
    return None


def pick_oof_target_column(frame) -> str | None:  # type: ignore[no-untyped-def]
    """Return the target column name from an OOF prediction table."""
    columns = [str(col) for col in frame.columns]
    normalized = {col.lower().strip(): col for col in columns}
    for key in ("y", "target", "label", "y_true", "isdefault", "is_default"):
        if key in normalized:
            return normalized[key]
    return None


def pick_oof_prediction_column(frame, *, metric: str) -> str | None:  # type: ignore[no-untyped-def]
    """Return the most suitable prediction column for the requested metric."""
    columns = [str(col) for col in frame.columns]
    normalized = {col.lower().strip(): col for col in columns}
    is_prob_metric = bool(metric_requires_proba(metric))

    if is_prob_metric:
        for key in ("oof_proba", "pred_proba", "prediction_proba", "probability", "proba", "score"):
            if key in normalized:
                return normalized[key]
        for col in columns:
            lowered = col.lower()
            if "proba" in lowered or "prob" in lowered or "score" in lowered:
                return col
    for key in ("oof_pred", "prediction", "pred", "y_pred"):
        if key in normalized:
            return normalized[key]
    if is_prob_metric:
        return None
    for col in columns:
        lowered = col.lower()
        if any(token in lowered for token in ("pred", "score", "proba", "prob")):
            return col
    return None


def extract_numeric_list(value: object) -> list[float] | None:
    """Return parsed numeric list or None when payload value is not a numeric list."""
    if not isinstance(value, list):
        return None
    parsed = [float(item) for item in value if isinstance(item, (int, float))]
    return parsed or None


def persist_metric_recheck_payload(*, iter_dir: Path, resolved_metrics_path: Path, payload: dict[str, object]) -> None:
    """Persist recomputed metric payload to canonical iteration metrics artifacts."""
    candidates = [resolved_metrics_path, iter_dir / "metrics.json", iter_dir / "output" / "metrics.json"]
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        write_json_object(path, payload)


def recompute_metric_from_oof_artifact(
    *,
    iter_dir: Path,
    payload: dict[str, object] | None,
    target_metric: str | None,
    metric_direction: str,
    resolve_iteration_artifact: Callable[[Path, str], Path | None],
) -> tuple[EvaluationResult, dict[str, object]] | None:
    """Recompute target metric from cached OOF predictions without rerunning training."""
    if not target_metric:
        return None
    oof_path = resolve_iteration_artifact(iter_dir, "oof_predictions.csv")
    if oof_path is None or not oof_path.exists():
        return None
    try:
        oof = read_table(oof_path)
    except Exception:
        return None
    if oof.empty:
        return None
    try:
        import pandas as pd
    except Exception:
        return None

    y_col = pick_oof_target_column(oof)
    pred_col = pick_oof_prediction_column(oof, metric=target_metric)
    if y_col is None or pred_col is None:
        return None

    y_series = pd.to_numeric(oof[y_col], errors="coerce")
    pred_series = pd.to_numeric(oof[pred_col], errors="coerce")
    valid_mask = y_series.notna() & pred_series.notna()
    if int(valid_mask.sum()) < 2:
        return None
    y_values = y_series[valid_mask].to_numpy()
    pred_values = pred_series[valid_mask].to_numpy()

    try:
        metric_value = float(compute_metric(target_metric, y_values, pred_values))
    except Exception:
        return None

    metric_name = canonical_metric(target_metric)
    direction = infer_direction(metric_name, metric_direction)
    score_source_raw = payload.get("score_source") if isinstance(payload, dict) else None
    score_source = (
        str(score_source_raw).strip() if isinstance(score_source_raw, str) and str(score_source_raw).strip() else "cv"
    )
    std_value = tolerant_finite_float(payload.get("offline_std")) if isinstance(payload, dict) else None
    train_score = tolerant_finite_float(payload.get("train_score")) if isinstance(payload, dict) else None
    val_score = tolerant_finite_float(payload.get("val_score")) if isinstance(payload, dict) else None
    fold_scores = extract_numeric_list(payload.get("fold_scores")) if isinstance(payload, dict) else None

    evaluation = EvaluationResult(
        score_source=score_source,
        metric=metric_name,
        direction=direction,  # type: ignore[arg-type]
        value=metric_value,
        std=std_value,
        train_score=train_score,
        val_score=val_score,
        fold_scores=fold_scores,
    )
    updated_payload = dict(payload) if isinstance(payload, dict) else {}
    updated_payload["metric"] = metric_name
    updated_payload["direction"] = direction
    updated_payload["score_source"] = score_source
    updated_payload["offline_value"] = metric_value
    updated_payload["value"] = metric_value
    updated_payload["metric_recheck_source"] = f"oof_predictions:{oof_path.name}"
    updated_payload["metric_recheck_without_retrain"] = True
    loop_decision = updated_payload.get("loop_decision")
    rubric_loop = bool(
        isinstance(loop_decision, dict)
        and (
            loop_decision.get("metric") == "rubric_readiness_score_0_100"
            or loop_decision.get("source") == "offline_artifact_rubric"
        )
    )
    if rubric_loop:
        model_selection = updated_payload.get("model_selection_decision")
        model_selection = dict(model_selection) if isinstance(model_selection, dict) else {}
        model_selection.update(
            {
                "metric": metric_name,
                "direction": direction,
                "source": score_source,
                "value": metric_value,
                "metric_recheck_without_retrain": True,
            }
        )
        updated_payload["model_selection_decision"] = model_selection
    elif isinstance(loop_decision, dict):
        loop_decision["source"] = score_source
        loop_decision["value"] = metric_value
    else:
        updated_payload["loop_decision"] = {"source": score_source, "value": metric_value}
    return evaluation, updated_payload


def extract_kernel_metric(payload: dict[str, object], target_metric: str | None) -> tuple[str | None, float | None]:
    def as_number(value: object) -> float | None:
        return tolerant_finite_float(value)

    def normalize(text: str) -> str:
        return "".join(ch for ch in text.lower() if ch.isalnum())

    def metric_hint() -> str | None:
        metric_raw = payload.get("metric")
        if isinstance(metric_raw, str) and metric_raw.strip():
            return metric_raw.strip()
        primary_raw = payload.get("primary_metric")
        if isinstance(primary_raw, str) and primary_raw.strip():
            return primary_raw.strip()
        return target_metric.strip() if isinstance(target_metric, str) and target_metric.strip() else None

    def pick_selected_metric(selected: dict[str, object]) -> tuple[str | None, float | None]:
        hint = metric_hint()
        hint_norm = normalize(hint) if hint else ""
        if "map" in hint_norm and "f1" in hint_norm:
            value = as_number(selected.get("combined_score"))
            if value is not None:
                return (hint, value)
        if "map" in hint_norm:
            value = as_number(selected.get("mean_map"))
            if value is not None:
                return (hint or "mean_map", value)
        if "f1" in hint_norm:
            value = as_number(selected.get("oof_f1"))
            if value is not None:
                return (hint or "f1", value)
        for key, name in (
            ("offline_value", hint),
            ("value", hint),
            ("score", hint),
            ("cv_mean", hint),
            ("combined_score", hint or "combined_score"),
            ("mean_map", hint or "mean_map"),
            ("oof_f1", hint or "f1"),
        ):
            value = as_number(selected.get(key))
            if value is not None:
                return (name, value)
        return (None, None)

    def strip_prefixes(text: str) -> str:
        lowered = text.lower()
        for prefix in (
            "val_",
            "train_",
            "test_",
            "oof_",
            "cv_",
            "holdout_",
            "offline_",
            "online_",
            "public_",
            "private_",
        ):
            if lowered.startswith(prefix):
                return text[len(prefix) :]
        return text

    def prefers_lower(metric: str) -> bool:
        return canonical_metric(metric) in {
            "aurc",
            "rmse",
            "rmsle",
            "mae",
            "mape",
            "mse",
            "logloss",
            "mcrmse",
            "smape",
            "pinball_loss",
            "interval_score",
        } or normalize(metric) in {"logloss", "loss"}

    def pick_from_dict(metric_key: str, values: dict[str, object]) -> float | None:
        selection = payload.get("selection")
        if isinstance(selection, str) and selection in values:
            selected_value = as_number(values.get(selection))
            if selected_value is not None:
                return selected_value
        for key in ("selected", "average", "stacked", "best", "val", "oof", "score"):
            value = as_number(values.get(key))
            if value is not None:
                return value
        numeric = [parsed for raw in values.values() if (parsed := as_number(raw)) is not None]
        if not numeric:
            return None
        return min(numeric) if prefers_lower(metric_key) else max(numeric)

    selected_raw = payload.get("selected")
    if isinstance(selected_raw, dict):
        metric, value = pick_selected_metric(selected_raw)
        if value is not None:
            return (metric, value)

    selected_cv_mean = as_number(payload.get("selected_cv_mean"))
    if selected_cv_mean is not None:
        return (metric_hint(), selected_cv_mean)

    offline_value = as_number(payload.get("offline_value"))
    if offline_value is not None:
        return (str(payload.get("metric") or target_metric or "unknown"), offline_value)
    payload_value = as_number(payload.get("value"))
    if payload_value is not None:
        return (str(payload.get("metric") or target_metric or "unknown"), payload_value)
    score_value = as_number(payload.get("score"))
    if score_value is not None:
        return (str(payload.get("metric") or target_metric or "unknown"), score_value)

    hinted_metric = metric_hint()
    hinted_key = normalize(hinted_metric) if hinted_metric else ""
    if hinted_key:
        for key, val in payload.items():
            parsed = as_number(val)
            if parsed is None:
                continue
            normalized_key = normalize(str(key))
            normalized_base = normalize(strip_prefixes(str(key)))
            if normalized_key == hinted_key or normalized_base == hinted_key:
                return (hinted_metric, parsed)

    leaderboard_raw = payload.get("leaderboard")
    if isinstance(leaderboard_raw, list):
        selected_pipeline = payload.get("selected_pipeline")
        selected_name = selected_pipeline.strip() if isinstance(selected_pipeline, str) else ""
        best_value: float | None = None
        best_metric = metric_hint()
        for item in leaderboard_raw:
            if not isinstance(item, dict):
                continue
            if selected_name:
                pipeline = item.get("pipeline")
                if isinstance(pipeline, str) and pipeline.strip() != selected_name:
                    continue
            cv_mean = as_number(item.get("cv_mean"))
            if cv_mean is None:
                continue
            if best_value is None:
                best_value = cv_mean
            elif best_metric and prefers_lower(best_metric):
                best_value = min(best_value, cv_mean)
            else:
                best_value = max(best_value, cv_mean)
        if best_value is not None:
            return (best_metric, best_value)

    pipelines_raw = payload.get("pipelines")
    if isinstance(pipelines_raw, list):
        selected_name = None
        if isinstance(selected_raw, dict):
            maybe_name = selected_raw.get("name")
            if isinstance(maybe_name, str) and maybe_name.strip():
                selected_name = maybe_name.strip()

        best_value: float | None = None
        best_metric: str | None = None
        for item in pipelines_raw:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if selected_name and isinstance(name, str) and name.strip() != selected_name:
                continue
            metric, value = pick_selected_metric(item)
            if value is None:
                continue
            if best_value is None:
                best_value = value
                best_metric = metric
            elif best_metric and prefers_lower(best_metric):
                best_value = min(best_value, value)
            else:
                best_value = max(best_value, value)
        if best_value is not None:
            return (best_metric, best_value)

    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        metric_key = None
        if isinstance(key, str) and key.lower().startswith("oof_"):
            metric_key = key[4:]
        elif isinstance(key, str):
            metric_key = key
        if metric_key is None:
            continue
        if target_metric is not None and normalize(metric_key) != normalize(target_metric):
            continue
        picked = pick_from_dict(metric_key, value)
        if picked is not None:
            return (metric_key, picked)

    aliases: dict[str, tuple[str, ...]] = {
        "accuracy": ("accuracy", "acc"),
        "aurc": ("aurc", "risk_coverage_auc", "area_under_risk_coverage_curve"),
        "auc": ("auc", "rocauc", "roc_auc"),
        "brier_score": ("brier", "brier_score", "brierscore"),
        "concordance_index": ("cindex", "concordance", "concordance_index", "concordanceindex"),
        "fmax": ("fmax", "proxyfmax"),
        "f1": ("f1", "f1score"),
        "interval_score": ("interval_score", "intervalscore", "prediction_interval_score"),
        "logloss": ("logloss", "log_loss"),
        "mae": ("mae",),
        "mape": ("mape",),
        "mcrmse": ("mcrmse", "mean_columnwise_rmse", "columnwise_rmse"),
        "ndcg": ("ndcg", "normalized_discounted_cumulative_gain"),
        "pearson": ("pearson", "pearsonr"),
        "pinball_loss": ("pinball", "pinball_loss", "quantile_loss"),
        "quadratic_weighted_kappa": ("qwk", "weighted_kappa", "quadratic_weighted_kappa", "cohen_kappa"),
        "rmse": ("rmse",),
        "rmsle": ("rmsle",),
        "r2": ("r2", "r2score"),
        "smape": ("smape",),
        "spearman": ("spearman", "spearmanr"),
    }

    target_key = normalize(target_metric) if target_metric else ""
    if target_key:
        wanted = set()
        for key, values in aliases.items():
            if target_key == normalize(key) or target_key in {normalize(v) for v in values}:
                wanted.update({normalize(v) for v in values})
                wanted.add(normalize(key))
                break
        if wanted:
            for key, val in payload.items():
                parsed = as_number(val)
                if parsed is None:
                    continue
                normalized_key = normalize(str(key))
                normalized_base = normalize(strip_prefixes(str(key)))
                if normalized_key in wanted or normalized_base in wanted:
                    return (str(target_metric), parsed)

    for key, val in payload.items():
        parsed = as_number(val)
        if parsed is None:
            continue
        normalized_key = normalize(str(key))
        normalized_base = normalize(strip_prefixes(str(key)))
        for metric_name, values in aliases.items():
            normalized_aliases = {normalize(v) for v in values}
            normalized_aliases.add(normalize(metric_name))
            if normalized_key in normalized_aliases or normalized_base in normalized_aliases:
                return (metric_name, parsed)

    return (str(target_metric) if target_metric else None, None)


def evaluation_from_kernel_metrics_payload(
    payload: dict[str, object],
    *,
    direction: str,
    target_metric: str | None,
) -> EvaluationResult | None:
    """Build an evaluation result from kernel metrics payload with trust-aware source fallback."""
    metric_name, value = extract_kernel_metric(payload, target_metric)
    if value is None:
        return None
    payload_direction_raw = payload.get("direction")
    if payload_direction_raw is None:
        payload_direction_raw = payload.get("target_direction")
    payload_direction = str(payload_direction_raw).strip().lower() if payload_direction_raw is not None else ""
    resolved_direction = direction
    if payload_direction in {"minimize", "maximize"}:
        resolved_direction = payload_direction

    std = payload.get("offline_std")
    if std is None:
        std = payload.get("std")
    if std is None:
        std = payload.get("selected_cv_std")
    std_value = tolerant_finite_float(std)

    fold_scores_raw = payload.get("fold_scores")
    fold_scores: list[float] | None = None
    if isinstance(fold_scores_raw, list):
        parsed_fold_scores = [float(item) for item in fold_scores_raw if isinstance(item, (int, float))]
        if parsed_fold_scores:
            fold_scores = parsed_fold_scores
            if std_value is None and len(parsed_fold_scores) > 1:
                std_value = float(stdev(parsed_fold_scores))

    raw_score_source = str(payload.get("score_source", "holdout")).strip().lower()
    raw_score_source = raw_score_source.replace("-", "_").replace(" ", "_")
    score_source = normalize_score_source_name(raw_score_source)
    grouped_contract_valid = True
    if raw_score_source == "grouped_oof_cv":
        provenance = payload.get("model_selection_decision")
        grouped_contract_valid, _ = validate_grouped_oof_contract(provenance)
        if not grouped_contract_valid:
            score_source = "grouped_oof_cv_invalid_contract"
    if score_source == "holdout":
        for key in payload.keys():
            if isinstance(key, str) and key.lower().startswith("oof_"):
                score_source = "cv"
                break
    trusted_fallback_value = None
    if not grouped_contract_valid:
        trusted_fallback_value = None
    elif not is_trusted_offline_score_source(score_source):
        trusted_fallback_value = extract_trusted_cv_value_from_metrics_payload(payload)
        if trusted_fallback_value is not None:
            value = trusted_fallback_value
            score_source = "cv"

    return EvaluationResult(
        score_source=score_source,
        metric=metric_name or target_metric or "unknown",
        direction=resolved_direction,  # type: ignore[arg-type]
        value=float(value),
        std=std_value,
        train_score=None,
        val_score=None,
        fold_scores=fold_scores,
    )


def load_kernel_metrics(metrics_path: Path, direction: str, target_metric: str | None) -> EvaluationResult | None:
    """Load kernel metrics from disk into a normalized evaluation result."""
    payload = load_json_object(metrics_path)
    if payload is None:
        return None
    return evaluation_from_kernel_metrics_payload(
        payload,
        direction=direction,
        target_metric=target_metric,
    )


def metric_value_from_payload_item(item: dict[str, object]) -> float | None:
    for key in (
        "offline_value",
        "selected_cv_mean",
        "cv_mean",
        "score",
        "value",
        "combined_score",
        "mean_map",
        "oof_f1",
    ):
        value = tolerant_finite_float(item.get(key))
        if value is not None:
            return value
    return None


def extract_baseline_candidates_from_metrics_payload(payload: dict[str, object]) -> list[tuple[str, float]]:
    candidates: list[tuple[str, float]] = []

    pipelines_raw = payload.get("pipelines")
    if isinstance(pipelines_raw, list):
        for item in pipelines_raw:
            if not isinstance(item, dict):
                continue
            name_raw = item.get("name") or item.get("pipeline")
            name = str(name_raw).strip() if isinstance(name_raw, str) else ""
            lowered = name.lower()
            if "baseline" not in lowered and "persistence" not in lowered:
                continue
            score = metric_value_from_payload_item(item)
            if score is not None:
                candidates.append((f"pipelines:{name or 'unnamed'}", float(score)))

    for key, value in payload.items():
        if not isinstance(key, str):
            continue
        lowered = key.lower()
        if "baseline" not in lowered and "persistence" not in lowered:
            continue
        if isinstance(value, dict):
            score = metric_value_from_payload_item(value)
            if score is not None:
                candidates.append((f"metrics:{key}", float(score)))
                continue
            for nested_key, nested_value in value.items():
                nested_score = tolerant_finite_float(nested_value)
                if nested_score is None:
                    continue
                candidates.append((f"metrics:{key}.{nested_key}", float(nested_score)))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    score = metric_value_from_payload_item(item)
                    if score is not None:
                        candidates.append((f"metrics:{key}[{index}]", float(score)))
                        continue
                parsed = tolerant_finite_float(item)
                if parsed is not None:
                    candidates.append((f"metrics:{key}[{index}]", float(parsed)))
        else:
            parsed = tolerant_finite_float(value)
            if parsed is not None:
                candidates.append((f"metrics:{key}", float(parsed)))

    return candidates


def collect_kernel_log_text(logs_dir: Path | None) -> str:
    if logs_dir is None or not logs_dir.exists():
        return ""
    texts: list[str] = []
    for path in sorted(logs_dir.glob("*.log")):
        name = path.name.lower()
        if "stdout" not in name and "kernel" not in name:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not text:
            continue
        texts.append(text[-250_000:])
    if not texts:
        return ""
    return "\n".join(texts)


def extract_validation_scores_from_log_text(log_text: str, metric_name: str | None) -> list[float]:
    if not log_text:
        return []
    pattern = re.compile(
        r"val_([^=\n]{1,80})\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        flags=re.IGNORECASE,
    )
    target_norm = normalize_metric_name(metric_name)
    scores: list[float] = []
    for match in pattern.finditer(log_text):
        metric_label = match.group(1).strip()
        parsed = tolerant_finite_float(match.group(2))
        if parsed is None:
            continue
        if target_norm:
            label_norm = normalize_metric_name(metric_label)
            if label_norm and (target_norm not in label_norm and label_norm not in target_norm):
                continue
        scores.append(float(parsed))
    return scores


def extract_baseline_scores_from_log_text(log_text: str) -> list[float]:
    if not log_text:
        return []
    scores: list[float] = []
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if "baseline" not in lowered and "persistence" not in lowered:
            continue
        if "fold=" in lowered:
            continue
        for match in _BASELINE_SCORE_ASSIGNMENT_RE.finditer(line):
            parsed = tolerant_finite_float(match.group("value"))
            if parsed is not None:
                scores.append(float(parsed))
    return scores
