from __future__ import annotations

import re
from pathlib import Path

from kagglebot.artifact_io import copy_artifact_if_needed
from kagglebot.baseline_tokens import ID_LIKE_COLUMN_NAMES
from kagglebot.submission_sample_discovery import (
    TABULAR_INPUT_SUFFIXES,
    TABULAR_INPUT_SUFFIXES_ORDERED,
    TABULAR_SUBMISSION_SUFFIXES,
    TABULAR_TEXT_SUFFIXES,
    default_delimited_text_separator,
    find_usable_sample_submissions,
    is_tabular_data_path,
    path_mentions_role,
    sample_candidate_key,
    sample_name_score,
    tabular_file_has_data_rows,
    tabular_file_has_two_or_more_columns,
    tabular_stem,
    tabular_suffix,
)
from kagglebot.test_table_aliases import (
    STRONG_TEST_TABLE_TOKENS,
    TEST_TABLE_COMPACT_ALIASES,
    TEST_TABLE_EXCLUDE_TOKENS,
    TEST_TABLE_STEMS,
    WEAK_TEST_TABLE_TOKENS,
)

_SAMPLE_MIRROR_SUFFIXES = set(TABULAR_INPUT_SUFFIXES)
_EXPANDABLE_SAMPLE_SUFFIXES = set(TABULAR_SUBMISSION_SUFFIXES)
_TABULAR_DATA_SUFFIXES = TABULAR_INPUT_SUFFIXES_ORDERED


def ensure_local_sample_submission_file(*, base_dir: Path, slug: str) -> Path | None:
    """Ensure local sample submission artifacts exist for legacy and suffix-aware kernels."""
    competition_dir = base_dir / slug
    data_dir = competition_dir / "data"
    canonical_path = data_dir / "sample_submission.csv"
    if canonical_path.exists():
        if tabular_file_has_data_rows(canonical_path) and tabular_file_has_two_or_more_columns(canonical_path):
            expand_placeholder_sample_submission(canonical_path=canonical_path, data_dir=data_dir)
            mirrored_path = mirror_missing_sample_submission_source(
                context_dir=competition_dir / "context",
                data_dir=data_dir,
                canonical_path=canonical_path,
            )
            if mirrored_path is not None and tabular_suffix(mirrored_path) in _EXPANDABLE_SAMPLE_SUFFIXES:
                expand_placeholder_sample_submission(canonical_path=mirrored_path, data_dir=data_dir)
            for sibling in _existing_suffixed_sample_submissions(data_dir=data_dir, exclude=canonical_path):
                if tabular_suffix(sibling) in _EXPANDABLE_SAMPLE_SUFFIXES:
                    expand_placeholder_sample_submission(canonical_path=sibling, data_dir=data_dir)
            return canonical_path
    source_path = resolve_sample_submission_source(
        context_dir=competition_dir / "context",
        data_dir=data_dir,
    )
    if source_path is None:
        source_path = synthesize_sample_submission_source(data_dir=data_dir)
    if source_path is None:
        return canonical_path if canonical_path.exists() else None
    data_dir.mkdir(parents=True, exist_ok=True)
    mirrored_path = mirror_sample_submission_with_suffix(source=source_path, data_dir=data_dir)
    canonical_written = copy_or_convert_sample_submission(source=source_path, destination=canonical_path)
    if canonical_written:
        expand_placeholder_sample_submission(canonical_path=canonical_path, data_dir=data_dir)
    if mirrored_path is not None and tabular_suffix(mirrored_path) in _EXPANDABLE_SAMPLE_SUFFIXES:
        expand_placeholder_sample_submission(canonical_path=mirrored_path, data_dir=data_dir)
    if canonical_written or canonical_path.exists():
        return canonical_path
    return mirrored_path


def synthesize_sample_submission_source(*, data_dir: Path) -> Path | None:
    try:
        from kagglebot.solver.io import ensure_sample_submission
    except Exception:
        return None
    try:
        return ensure_sample_submission(data_dir)
    except Exception:
        return None


def resolve_sample_submission_source(*, context_dir: Path, data_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for root in (context_dir, data_dir):
        if not root.exists():
            continue
        try:
            paths = sorted(root.rglob("*"))
        except OSError:
            continue
        for path in paths:
            if not path.is_file():
                continue
            if not is_tabular_data_path(path):
                continue
            if sample_name_score(path) < 2:
                continue
            if not tabular_file_has_data_rows(path):
                continue
            if not tabular_file_has_two_or_more_columns(path):
                continue
            candidates.append(path)
    if candidates:
        return max(candidates, key=sample_candidate_key)
    if not data_dir.exists():
        return None
    return None


def copy_or_convert_sample_submission(*, source: Path, destination: Path) -> bool:
    if tabular_suffix(source) == ".csv":
        copy_artifact_if_needed(source=source, destination=destination)
        return True
    try:
        from kagglebot.solver.io import read_table, write_table
    except Exception:
        return False

    try:
        frame = read_table(source)
    except Exception:
        return False
    try:
        write_table(frame, destination)
    except Exception:
        return False
    return True


def mirror_sample_submission_with_suffix(*, source: Path, data_dir: Path) -> Path | None:
    suffix = tabular_suffix(source)
    if suffix not in _SAMPLE_MIRROR_SUFFIXES:
        return None
    destination = data_dir / f"sample_submission{suffix}"
    if _same_path(source, destination):
        return destination
    copy_artifact_if_needed(source=source, destination=destination)
    return destination


def mirror_missing_sample_submission_source(*, context_dir: Path, data_dir: Path, canonical_path: Path) -> Path | None:
    source = _resolve_missing_mirror_sample_source(
        context_dir=context_dir,
        data_dir=data_dir,
        canonical_path=canonical_path,
    )
    if source is None or _same_path(source, canonical_path):
        return None
    suffix = tabular_suffix(source)
    if suffix in {"", ".csv"} or suffix not in _SAMPLE_MIRROR_SUFFIXES:
        return None
    destination = data_dir / f"sample_submission{suffix}"
    if destination.exists():
        return destination
    return mirror_sample_submission_with_suffix(source=source, data_dir=data_dir)


def _resolve_missing_mirror_sample_source(*, context_dir: Path, data_dir: Path, canonical_path: Path) -> Path | None:
    for root in (context_dir, data_dir):
        if not root.exists():
            continue
        for candidate in find_usable_sample_submissions(root):
            if _same_path(candidate, canonical_path):
                continue
            suffix = tabular_suffix(candidate)
            if suffix and suffix != ".csv":
                return candidate
    return None


def _existing_suffixed_sample_submissions(*, data_dir: Path, exclude: Path) -> list[Path]:
    candidates: list[Path] = []
    for suffix in _SAMPLE_MIRROR_SUFFIXES:
        candidate = data_dir / f"sample_submission{suffix}"
        if candidate == exclude or not candidate.is_file():
            continue
        candidates.append(candidate)
    return sorted(candidates, key=lambda path: str(path).lower())


def expand_placeholder_sample_submission(*, canonical_path: Path, data_dir: Path) -> None:
    """Expand tiny sample_submission templates to full test ids when confidently detected."""
    try:
        import pandas as pd
    except Exception:
        return

    test_path = _find_test_tabular_file(data_dir=data_dir)
    if not canonical_path.exists() or test_path is None or not test_path.exists():
        return
    try:
        from kagglebot.solver.io import read_table, write_table

        sample = read_table(canonical_path)
    except Exception:
        return
    if sample.empty or len(sample.columns) < 2:
        return

    id_col = str(sample.columns[0])
    if not _is_id_like_column(id_col):
        return
    pred_cols = [str(col) for col in sample.columns if str(col) != id_col]
    if not pred_cols:
        return
    if len(sample) > 10 or sample[id_col].duplicated().any():
        return

    try:
        test = read_table(test_path)
    except Exception:
        return
    if id_col not in test.columns:
        return

    test_ids = test[id_col].astype(str).tolist()
    sample_ids = sample[id_col].astype(str).tolist()
    if len(test_ids) <= max(len(sample_ids) * 3, len(sample_ids) + 10):
        return
    if sample_ids and test_ids[: len(sample_ids)] != sample_ids:
        return

    defaults = placeholder_prediction_defaults(
        sample=sample,
        data_dir=data_dir,
        id_col=id_col,
        prediction_columns=pred_cols,
    )
    expanded = pd.DataFrame({id_col: test_ids})
    for col in pred_cols:
        expanded[col] = defaults.get(col, 0.0)
    canonical_columns = [str(col) for col in sample.columns]
    for col in canonical_columns:
        if col not in expanded.columns:
            expanded[col] = ""
    expanded = expanded[canonical_columns]
    try:
        write_table(expanded, canonical_path)
    except Exception:
        write_expanded_placeholder_fallback(expanded, canonical_path)


def write_expanded_placeholder_fallback(frame, path: Path) -> None:
    suffix = tabular_suffix(path)
    if suffix not in TABULAR_TEXT_SUFFIXES:
        return
    frame.to_csv(path, index=False, sep=default_delimited_text_separator(suffix))


def placeholder_prediction_defaults(
    *,
    sample,
    data_dir: Path,
    id_col: str,
    prediction_columns: list[str],
) -> dict[str, float]:
    """Estimate stable default values for expanded placeholder prediction columns."""
    try:
        import pandas as pd
    except Exception:
        return {col: 0.0 for col in prediction_columns}

    defaults: dict[str, float] = {}
    for col in prediction_columns:
        sample_series = pd.to_numeric(sample[col], errors="coerce").dropna()
        defaults[col] = float(sample_series.mean()) if not sample_series.empty else 0.0

    train_path = _find_named_tabular_file(data_dir=data_dir, stem="train")
    if train_path is None or not train_path.exists():
        return defaults
    train_cols = [col for col in prediction_columns if col != id_col]
    if not train_cols:
        return defaults
    try:
        from kagglebot.solver.io import read_table

        train = read_table(train_path)
    except Exception:
        return defaults
    for col in train_cols:
        if col not in train.columns:
            continue
        train_series = pd.to_numeric(train[col], errors="coerce").dropna()
        if not train_series.empty:
            defaults[col] = float(train_series.mean())
    return defaults


def _find_named_tabular_file(*, data_dir: Path, stem: str) -> Path | None:
    lowered_stem = stem.lower()
    for suffix in _TABULAR_DATA_SUFFIXES:
        direct = data_dir / f"{stem}{suffix}"
        if direct.is_file():
            return direct
    try:
        candidates = [
            path
            for path in data_dir.rglob("*")
            if path.is_file()
            and is_tabular_data_path(path)
            and (tabular_stem(path).lower() == lowered_stem or path_mentions_role(path, lowered_stem))
            and ".kagglebot_cache" not in {part.lower() for part in path.parts}
        ]
    except OSError:
        return None
    if not candidates:
        return None
    return min(candidates, key=lambda path: (len(path.relative_to(data_dir).parts), str(path).lower()))


def _find_test_tabular_file(*, data_dir: Path) -> Path | None:
    for stem in TEST_TABLE_STEMS:
        for suffix in _TABULAR_DATA_SUFFIXES:
            direct = data_dir / f"{stem}{suffix}"
            if direct.is_file():
                return direct
    try:
        candidates = [
            (_test_table_name_score(path), path)
            for path in data_dir.rglob("*")
            if path.is_file()
            and is_tabular_data_path(path)
            and ".kagglebot_cache" not in {part.lower() for part in path.parts}
            and _test_table_name_score(path) > 0
        ]
    except OSError:
        return None
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item[0],
            -len(item[1].relative_to(data_dir).parts),
            str(item[1]).lower(),
        ),
    )[1]


def _test_table_name_score(path: Path) -> int:
    stem = tabular_stem(path).lower().replace("-", "_").replace(" ", "_")
    compact = stem.replace("_", "").replace(".", "")
    if stem in TEST_TABLE_STEMS or compact in TEST_TABLE_COMPACT_ALIASES:
        return 4
    tokens = {token for token in stem.replace(".", "_").split("_") if token}
    if tokens & TEST_TABLE_EXCLUDE_TOKENS:
        return 0
    if path_mentions_role(path, "test"):
        return 3
    if tokens & STRONG_TEST_TABLE_TOKENS:
        return 2
    if tokens & WEAK_TEST_TABLE_TOKENS:
        return 1
    return 0


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left == right


def _is_id_like_column(column: object) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
    compact = normalized.replace("_", "")
    return normalized in ID_LIKE_COLUMN_NAMES or compact in ID_LIKE_COLUMN_NAMES


def _tabular_stem(path: Path) -> str:
    return tabular_stem(path)
