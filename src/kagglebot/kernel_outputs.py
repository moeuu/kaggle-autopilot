from __future__ import annotations

import re
from pathlib import Path

from kagglebot.artifact_io import copy_artifact_if_needed as _copy_artifact_if_needed
from kagglebot.submission_artifacts import find_submission_manifest, resolve_manifest_references

_INTERMEDIATE_SUBMISSION_RE = re.compile(r"^submission(?:_[A-Za-z0-9_.-]+)?_fold(?P<fold>\d+)\.csv$", re.IGNORECASE)
LOCAL_KERNEL_OPTIONAL_ARTIFACTS: tuple[str, ...] = (
    "oof_predictions.csv",
    "split_diagnostics.json",
    "feature_suspects.csv",
    "submission_manifest.json",
    "metrics_summary.json",
    "cv_results.json",
    "cv_summary.json",
    "pipeline_diagnostics.json",
)


def find_submission_file(output_dir: Path) -> Path | None:
    manifest_path = find_submission_manifest(output_dir)
    if manifest_path is not None:
        _, submission_path, staging_dir, members = resolve_manifest_references(manifest_path)
        if submission_path is not None and submission_path.exists() and submission_path.is_file():
            return submission_path
        if staging_dir is not None or members:
            return manifest_path
    candidate = find_output_file(output_dir, "submission.csv")
    if candidate:
        return candidate
    candidate = find_intermediate_submission_file(output_dir)
    if candidate:
        return candidate
    return _find_submission_by_extension(output_dir)


def find_output_file(output_dir: Path, filename: str) -> Path | None:
    """Find the newest matching artifact within an output tree.

    Local kernels can be executed repeatedly for the same run/iteration while
    iterating on fixes. In that scenario, stale artifacts may exist alongside
    fresh ones (or nested under additional run directories). Prefer the most
    recently modified match to avoid accidentally reusing stale outputs.
    """

    candidates: list[Path] = []
    direct = output_dir / filename
    if direct.exists():
        candidates.append(direct)
    try:
        candidates.extend(path for path in output_dir.rglob(filename) if path.exists())
    except OSError:
        # Best-effort discovery; callers handle missing artifacts.
        pass
    files = [path for path in candidates if path.is_file()]
    if not files:
        return None
    # Deterministic tie-breaker: path string.
    return max(files, key=lambda path: (path.stat().st_mtime, str(path)))


def find_intermediate_submission_file(output_dir: Path) -> Path | None:
    """Find the newest fold-level submission emitted by an interrupted run."""

    candidates: list[tuple[int, Path]] = []
    try:
        paths = output_dir.rglob("submission*_fold*.csv")
    except OSError:
        return None
    for path in paths:
        if not path.is_file():
            continue
        match = _INTERMEDIATE_SUBMISSION_RE.match(path.name)
        if match is None:
            continue
        try:
            fold = int(match.group("fold"))
        except ValueError:
            continue
        candidates.append((fold, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[1].stat().st_mtime, item[0], str(item[1])))[1]


def resolve_local_kernel_artifacts(
    *,
    kernel_dir: Path,
    output_dir: Path,
    started_at: float,
) -> tuple[Path | None, Path | None]:
    candidates = local_kernel_artifact_roots(kernel_dir=kernel_dir, output_dir=output_dir)
    submission_candidates: list[Path] = []
    metrics_candidates: list[Path] = []
    for root in candidates:
        if not root.exists():
            continue
        sub = find_submission_file(root)
        if sub is not None and sub.exists():
            submission_candidates.append(sub)
        metric_path = find_output_file(root, "metrics.json")
        if metric_path is not None and metric_path.exists():
            metrics_candidates.append(metric_path)

    min_mtime = started_at - 1.0
    submission_path = pick_latest_artifact(submission_candidates, min_mtime=min_mtime)
    metrics_path = pick_latest_artifact(metrics_candidates, min_mtime=min_mtime)
    return submission_path, metrics_path


def resolve_local_kernel_artifact_file(
    *,
    kernel_dir: Path,
    output_dir: Path,
    started_at: float,
    filename: str,
) -> Path | None:
    file_candidates: list[Path] = []
    for root in local_kernel_artifact_roots(kernel_dir=kernel_dir, output_dir=output_dir):
        if not root.exists():
            continue
        match = find_output_file(root, filename)
        if match is not None and match.exists():
            file_candidates.append(match)
    min_mtime = started_at - 1.0
    return pick_latest_artifact(file_candidates, min_mtime=min_mtime)


def local_kernel_artifact_roots(*, kernel_dir: Path, output_dir: Path) -> list[Path]:
    return [
        output_dir,
        # Legacy generated kernels may write to the slug-level kernel_output
        # directory instead of the per-run output dir.
        kernel_dir.parents[2] / "kernel_output",
        # Many kernels treat the parent of the staged copy (run_dir) as the
        # "challenge dir" and write artifacts under run_dir/outputs.
        kernel_dir.parent / "outputs",
        kernel_dir.parent,
        kernel_dir / "outputs",
        Path("/kaggle/working"),
        kernel_dir,
    ]


def find_newest_existing_path(paths: list[Path], *, min_mtime: float | None = None) -> Path | None:
    candidates: list[tuple[float, int, str, Path]] = []
    for path in paths:
        try:
            if not path.exists():
                continue
            stat = path.stat()
        except OSError:
            continue
        if min_mtime is not None and stat.st_mtime < min_mtime:
            continue
        candidates.append((float(stat.st_mtime), int(stat.st_size), str(path), path))
    if not candidates:
        return None
    return max(candidates)[3]


def pick_latest_artifact(paths: list[Path], *, min_mtime: float) -> Path | None:
    return find_newest_existing_path(paths, min_mtime=min_mtime)


def copy_artifact_if_needed(*, source: Path, destination: Path) -> Path:
    return _copy_artifact_if_needed(source=source, destination=destination)


def copy_local_kernel_primary_artifacts(
    *,
    submission_path: Path,
    metrics_path: Path | None,
    output_dir: Path,
) -> tuple[Path, Path | None]:
    submission_dst = copy_artifact_if_needed(
        source=submission_path,
        destination=output_dir / submission_path.name,
    )
    metrics_dst = None
    if metrics_path is not None:
        metrics_dst = copy_artifact_if_needed(
            source=metrics_path,
            destination=output_dir / "metrics.json",
        )
    return submission_dst, metrics_dst


def copy_optional_local_kernel_artifacts(
    *,
    kernel_dir: Path,
    output_dir: Path,
    started_at: float,
    filenames: tuple[str, ...] = LOCAL_KERNEL_OPTIONAL_ARTIFACTS,
) -> list[Path]:
    copied: list[Path] = []
    for filename in filenames:
        optional_src = resolve_local_kernel_artifact_file(
            kernel_dir=kernel_dir,
            output_dir=output_dir,
            started_at=started_at,
            filename=filename,
        )
        if optional_src is None:
            continue
        copied.append(
            copy_artifact_if_needed(
                source=optional_src,
                destination=output_dir / filename,
            )
        )
    return copied


def _find_submission_by_extension(output_dir: Path) -> Path | None:
    suffixes = {".csv", ".tsv", ".txt", ".parquet", ".json", ".jsonl", ".zip"}
    compound_names = {"submission.tar.gz", "submission.tgz"}
    candidates: list[Path] = []
    for name in sorted(compound_names):
        candidate = output_dir / name
        if candidate.is_file():
            candidates.append(candidate)
    for suffix in sorted(suffixes):
        candidate = output_dir / f"submission{suffix}"
        if candidate.is_file():
            candidates.append(candidate)
    for path in output_dir.rglob("submission.*"):
        if not path.is_file():
            continue
        if path.name.lower() in compound_names:
            candidates.append(path)
            continue
        if path.suffix.lower() not in suffixes:
            continue
        candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, str(path)))
