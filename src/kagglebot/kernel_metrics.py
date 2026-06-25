from __future__ import annotations

import re
from pathlib import Path
from statistics import stdev

from kagglebot.json_utils import load_json_object
from kagglebot.metric_matching import normalize_metric_name
from kagglebot.scalar_utils import parse_finite_float
from kagglebot.score_sources import is_trusted_offline_score_source, normalize_score_source_name
from kagglebot.solver.evaluate import EvaluationResult


def _to_float(value: object) -> float | None:
    return parse_finite_float(value, allow_commas=True)


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
        parsed = _to_float(payload.get(key))
        if parsed is not None:
            return float(parsed)

    fold_scores_raw = payload.get("fold_scores")
    if isinstance(fold_scores_raw, list):
        fold_scores = [float(parsed) for item in fold_scores_raw if (parsed := _to_float(item)) is not None]
        if fold_scores:
            return float(sum(fold_scores) / len(fold_scores))
    return None


def extract_kernel_metric(payload: dict[str, object], target_metric: str | None) -> tuple[str | None, float | None]:
    def as_number(value: object) -> float | None:
        return _to_float(value)

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
        return normalize(metric) in {"rmse", "rmsle", "mae", "mape", "logloss", "loss"}

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
        "auc": ("auc", "rocauc", "roc_auc"),
        "brier_score": ("brier", "brier_score", "brierscore"),
        "fmax": ("fmax", "proxyfmax"),
        "f1": ("f1", "f1score"),
        "logloss": ("logloss", "log_loss"),
        "mae": ("mae",),
        "mape": ("mape",),
        "rmse": ("rmse",),
        "rmsle": ("rmsle",),
        "r2": ("r2", "r2score"),
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
    std_value = _to_float(std)

    fold_scores_raw = payload.get("fold_scores")
    fold_scores: list[float] | None = None
    if isinstance(fold_scores_raw, list):
        parsed_fold_scores = [float(item) for item in fold_scores_raw if isinstance(item, (int, float))]
        if parsed_fold_scores:
            fold_scores = parsed_fold_scores
            if std_value is None and len(parsed_fold_scores) > 1:
                std_value = float(stdev(parsed_fold_scores))

    score_source = normalize_score_source_name(payload.get("score_source", "holdout"))
    if score_source == "holdout":
        for key in payload.keys():
            if isinstance(key, str) and key.lower().startswith("oof_"):
                score_source = "cv"
                break
    trusted_fallback_value = None
    if not is_trusted_offline_score_source(score_source):
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
        value = _to_float(item.get(key))
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
                nested_score = _to_float(nested_value)
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
                parsed = _to_float(item)
                if parsed is not None:
                    candidates.append((f"metrics:{key}[{index}]", float(parsed)))
        else:
            parsed = _to_float(value)
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
        parsed = _to_float(match.group(2))
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
    pattern = re.compile(
        r"\b(?P<name>[a-z_][a-z0-9_]*?(?:score|auc|rmse|mae|mse|f1|loss|accuracy|acc|precision|recall|map|ndcg|logloss|brier|gini))\s*=\s*"
        r"(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        flags=re.IGNORECASE,
    )
    scores: list[float] = []
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if "baseline" not in lowered and "persistence" not in lowered:
            continue
        if "fold=" in lowered:
            continue
        for match in pattern.finditer(line):
            parsed = _to_float(match.group("value"))
            if parsed is not None:
                scores.append(float(parsed))
    return scores
