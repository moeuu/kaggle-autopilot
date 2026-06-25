from __future__ import annotations

import re

_PIPELINE_SEED_FOLD_RE = re.compile(r"(?P<pipeline>[A-Za-z0-9][A-Za-z0-9_.-]*)_seed(?P<seed>\d+)_fold(?P<fold>\d+)")
_PIPELINE_SEED_FOLD_INLINE_RE = re.compile(
    r"\b(?P<pipeline>[A-Za-z0-9][A-Za-z0-9_.-]*)\s*:\s*seed=(?P<seed>\d+)\s+fold=(?P<fold>\d+)\b"
)
_PIPELINE_START_RE = re.compile(r"\b(?:Running|Training)\s+pipeline:\s*(?P<pipeline>[A-Za-z0-9][A-Za-z0-9_.-]*)")
_PIPELINE_DONE_RE = re.compile(r"\bPipeline\s+(?P<pipeline>[A-Za-z0-9][A-Za-z0-9_.-]*)\s*:")
_PIPELINE_SUITE_RE = re.compile(r"\bSuite:\s*(?P<suite>[A-Za-z0-9][A-Za-z0-9_.-]*)")
_TRAIN_MODEL_START_RE = re.compile(r"\btrain start:\s*model=(?P<model>[A-Za-z0-9][A-Za-z0-9_.-]*)")
_CATBOOST_FALLBACK_RE = re.compile(r"\bCatBoost GPU failed; retrying on CPU:\s*(?P<reason>.+)")


def extract_training_stage_from_line(line: str) -> tuple[str, int, int] | None:
    inline_match = _PIPELINE_SEED_FOLD_INLINE_RE.search(line)
    if inline_match:
        return _match_to_stage_tuple(inline_match)
    path_match = _PIPELINE_SEED_FOLD_RE.search(line)
    if path_match:
        return _match_to_stage_tuple(path_match)
    return None


def _match_to_stage_tuple(match: re.Match[str]) -> tuple[str, int, int] | None:
    try:
        pipeline = str(match.group("pipeline")).strip()
        seed = int(match.group("seed"))
        fold = int(match.group("fold"))
    except Exception:  # noqa: BLE001
        return None
    if not pipeline:
        return None
    return pipeline, seed, fold


def extract_pipeline_start_from_line(line: str) -> str | None:
    match = _PIPELINE_START_RE.search(line)
    if not match:
        return None
    pipeline = str(match.group("pipeline")).strip()
    return pipeline or None


def extract_pipeline_suite_from_line(line: str) -> str | None:
    match = _PIPELINE_SUITE_RE.search(line)
    if not match:
        return None
    suite = str(match.group("suite")).strip()
    return suite or None


def extract_pipeline_done_from_line(line: str) -> str | None:
    match = _PIPELINE_DONE_RE.search(line)
    if not match:
        return None
    pipeline = str(match.group("pipeline")).strip()
    return pipeline or None


def extract_train_model_start_from_line(line: str) -> str | None:
    match = _TRAIN_MODEL_START_RE.search(line)
    if not match:
        return None
    model = str(match.group("model")).strip()
    return model or None


def extract_catboost_fallback_reason_from_line(line: str) -> str | None:
    match = _CATBOOST_FALLBACK_RE.search(line)
    if not match:
        return None
    reason = str(match.group("reason")).strip()
    return reason or None


def resolve_seed_current(*, seed: int, expected_seeds: list[int]) -> int | None:
    if not expected_seeds:
        return None
    try:
        return expected_seeds.index(seed) + 1
    except ValueError:
        return None


def resolve_fold_current(*, fold_raw: int, expected_folds: int | None, zero_based: bool) -> int | None:
    if expected_folds is None:
        return None
    if zero_based:
        value = fold_raw + 1
        if 1 <= value <= expected_folds:
            return value
    if 1 <= fold_raw <= expected_folds:
        return fold_raw
    if 0 <= fold_raw < expected_folds:
        return fold_raw + 1
    return None
