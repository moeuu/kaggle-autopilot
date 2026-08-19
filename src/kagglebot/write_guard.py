from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from kagglebot.exceptions import KaggleBotError
from kagglebot.submission_sample_discovery import TABULAR_SUBMISSION_SUFFIXES, tabular_suffix
from kagglebot.validators import scan_text_for_secrets

_MAX_GUARD_FILE_BYTES = 2_000_000
_MAX_GUARD_TOTAL_BACKUP_BYTES = 16 * 1024 * 1024
_PROTECTED_PATHS = (
    "src/",
    "tests/",
    "docs/",
    "README.md",
    "pyproject.toml",
    "uv.lock",
    ".gitignore",
    "AGENTS.md",
    "STRATEGY.md",
    "SECURITY.md",
    "knowledge/research/",
    "knowledge/kb.sqlite",
)
_NOISE_PREFIXES = (
    ".pytest_cache/",
    ".ruff_cache/",
    ".mypy_cache/",
    ".cache/",
    ".kagglebot_cache/",
    ".venv/",
)
_NOISE_SUFFIXES = (".pyc", ".pyo", ".DS_Store")
_EVENT_DELIVERY_STATE_FILENAME = "event_delivery_state.json"
_SAMPLE_SUBMISSION_SUFFIXES = set(TABULAR_SUBMISSION_SUFFIXES)


@dataclass(frozen=True)
class GuardSnapshot:
    backup: dict[str, bytes]
    oversized: set[str]
    external_backup: dict[str, bytes | None] = field(default_factory=dict)


@dataclass(frozen=True)
class WriteGuardPolicy:
    allowed_prefixes: tuple[Path, ...]
    denied_prefixes: tuple[Path, ...] = ()
    external_guard_paths: tuple[Path, ...] = ()
    snapshot_prefixes: tuple[Path, ...] = ()
    max_backup_bytes: int | None = None


def _snapshot_tree(root: Path, policy: WriteGuardPolicy | None = None) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    for path in _iter_snapshot_files(root, policy.snapshot_prefixes if policy is not None else ()):
        rel = path.relative_to(root).as_posix()
        try:
            stat = path.stat()
        except OSError:
            continue
        snapshot[rel] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def _iter_snapshot_files(root: Path, prefixes: Sequence[Path]) -> Sequence[Path]:
    roots = _snapshot_roots(root, prefixes)
    files: list[Path] = []
    seen: set[str] = set()
    for scan_root in roots:
        if scan_root.is_file():
            candidates = [scan_root]
        else:
            candidates = []
            for dirpath, dirnames, filenames in os.walk(scan_root):
                directory = Path(dirpath)
                rel_dir = directory.relative_to(root)
                dirnames[:] = [name for name in dirnames if not _prune_snapshot_directory((rel_dir / name).as_posix())]
                candidates.extend(directory / filename for filename in filenames)
        for path in candidates:
            rel = path.relative_to(root).as_posix()
            if rel in seen or any(rel.endswith(suffix) for suffix in _NOISE_SUFFIXES):
                continue
            seen.add(rel)
            files.append(path)
    return files


def _snapshot_roots(root: Path, prefixes: Sequence[Path]) -> list[Path]:
    if not prefixes:
        return [root]
    roots: list[Path] = []
    for prefix in prefixes:
        try:
            prefix.relative_to(root)
        except ValueError:
            continue
        if prefix.exists() and prefix not in roots:
            roots.append(prefix)
    for path in root.glob(".env*"):
        if path.exists() and path not in roots:
            roots.append(path)
    return roots


def _prune_snapshot_directory(path: str) -> bool:
    normalized = path.strip("/")
    if normalized == ".git" or normalized.startswith(".git/"):
        return True
    parts = normalized.split("/")
    if "__pycache__" in parts:
        return True
    return any(prefix.strip("/") in parts for prefix in _NOISE_PREFIXES)


def _default_external_guard_paths(repo_root: Path) -> tuple[Path, ...]:
    return (
        repo_root / ".git" / "HEAD",
        repo_root / ".git" / "index",
        repo_root / ".git" / "config",
        repo_root / ".git" / "packed-refs",
        repo_root / ".git" / "refs",
        repo_root / ".git" / "logs",
        Path.home() / ".kaggle" / "kaggle.json",
    )


def _repo_root_write_policy(
    *,
    repo_root: Path,
    denied_prefixes: list[Path],
    extra_allowed_prefixes: list[Path] | None = None,
    extra_external_guard_paths: list[Path] | None = None,
    snapshot_prefixes: list[Path] | None = None,
) -> WriteGuardPolicy:
    allowed_prefixes = [repo_root]
    if extra_allowed_prefixes:
        allowed_prefixes.extend(extra_allowed_prefixes)
    external_guard_paths = list(_default_external_guard_paths(repo_root))
    if extra_external_guard_paths:
        external_guard_paths.extend(extra_external_guard_paths)
    return WriteGuardPolicy(
        allowed_prefixes=tuple(allowed_prefixes),
        denied_prefixes=tuple(denied_prefixes),
        external_guard_paths=tuple(dict.fromkeys(external_guard_paths)),
        snapshot_prefixes=tuple(dict.fromkeys(snapshot_prefixes or [])),
        max_backup_bytes=_MAX_GUARD_TOTAL_BACKUP_BYTES,
    )


def build_repair_write_policy(
    *,
    repo_root: Path,
    data_dir: Path,
    kernels_dir: Path,
    module_file: Path,
    extra_allowed_prefixes: list[Path] | None = None,
) -> WriteGuardPolicy:
    extra_allowed: list[Path] = []
    if extra_allowed_prefixes:
        extra_allowed.extend(extra_allowed_prefixes)
    module_src_root = module_file.resolve().parents[1]
    if module_src_root.name == "src":
        extra_allowed.append(module_src_root)
    return _repo_root_write_policy(
        repo_root=repo_root,
        denied_prefixes=[data_dir, kernels_dir],
        extra_allowed_prefixes=extra_allowed,
        # Always include the loaded source tree.  Repair artifacts commonly live
        # outside the repository (for example under /data), and a policy made
        # only from those paths produced an empty repository snapshot.  That let
        # an agent edit src/ without the parent process noticing that its loaded
        # code was stale.
        snapshot_prefixes=[module_src_root, data_dir.parent / "kernel", data_dir, kernels_dir],
    )


def _coerce_write_policy(
    root: Path,
    allowed_prefixes: list[Path] | WriteGuardPolicy,
    denied_prefixes: list[Path] | None = None,
    external_guard_paths: list[Path] | None = None,
) -> WriteGuardPolicy:
    del root
    if isinstance(allowed_prefixes, WriteGuardPolicy):
        return allowed_prefixes
    return WriteGuardPolicy(
        allowed_prefixes=tuple(allowed_prefixes),
        denied_prefixes=tuple(denied_prefixes or []),
        external_guard_paths=tuple(external_guard_paths or []),
    )


def _backup_guarded_files(
    root: Path,
    allowed_prefixes: list[Path] | WriteGuardPolicy,
    denied_prefixes: list[Path] | None = None,
    external_guard_paths: list[Path] | None = None,
) -> GuardSnapshot:
    policy = _coerce_write_policy(
        root,
        allowed_prefixes,
        denied_prefixes=denied_prefixes,
        external_guard_paths=external_guard_paths,
    )
    allowed = _allowed_prefixes(root, policy.allowed_prefixes)
    denied = _allowed_prefixes(root, list(policy.denied_prefixes))
    backup: dict[str, bytes] = {}
    oversized: set[str] = set()
    backup_bytes = 0
    backup_limit = None if policy.max_backup_bytes is None else max(0, int(policy.max_backup_bytes))
    for path in _iter_snapshot_files(root, policy.snapshot_prefixes):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(".git/"):
            continue
        if _is_noise_path(rel, denied):
            continue
        if not (_is_protected_path(rel) or _is_denied(rel, denied)):
            continue
        if _is_allowed(rel, allowed, denied):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > _MAX_GUARD_FILE_BYTES:
            oversized.add(rel)
            continue
        if backup_limit is not None and backup_bytes + size > backup_limit:
            continue
        try:
            content = path.read_bytes()
        except OSError:
            continue
        if backup_limit is not None and backup_bytes + len(content) > backup_limit:
            continue
        backup[rel] = content
        backup_bytes += len(content)
    return GuardSnapshot(
        backup=backup,
        oversized=oversized,
        external_backup=_snapshot_external_guard_paths(policy.external_guard_paths),
    )


def _enforce_allowlist_changes(
    *,
    root: Path,
    before: dict[str, tuple[int, int]],
    after: dict[str, tuple[int, int]],
    allowed_prefixes: list[Path] | WriteGuardPolicy,
    stage: str,
    guard_snapshot: GuardSnapshot | None = None,
    auto_repair: bool = False,
    denied_prefixes: list[Path] | None = None,
    external_guard_paths: list[Path] | None = None,
) -> None:
    policy = _coerce_write_policy(
        root,
        allowed_prefixes,
        denied_prefixes=denied_prefixes,
        external_guard_paths=external_guard_paths,
    )
    _verify_external_guard_paths(stage=stage, guard_snapshot=guard_snapshot, paths=policy.external_guard_paths)
    allowed = _allowed_prefixes(root, policy.allowed_prefixes)
    denied = _allowed_prefixes(root, list(policy.denied_prefixes))
    changed = _diff_snapshots(before, after)
    unauthorized = [
        path for path in changed if not _is_allowed(path, allowed, denied) and not _is_noise_path(path, denied)
    ]
    if not unauthorized:
        return
    if auto_repair and guard_snapshot is not None:
        errors = _repair_unauthorized_changes(root, unauthorized, guard_snapshot, before, denied)
        after_repair = _snapshot_tree(root, policy)
        changed = _diff_snapshots(before, after_repair)
        unauthorized = [
            path for path in changed if not _is_allowed(path, allowed, denied) and not _is_noise_path(path, denied)
        ]
        unauthorized = _filter_restored_paths(root, unauthorized, guard_snapshot)
        if not unauthorized:
            return
        if errors:
            issue_text = "\n".join(f"- {error}" for error in errors)
            raise KaggleBotError(
                f"Agent write-guard failed in {stage} after repair:\n{issue_text}\nRemaining: {unauthorized}"
            )
    raise KaggleBotError(f"Agent write-guard failed in {stage}: {unauthorized}")


def _diff_snapshots(before: dict[str, tuple[int, int]], after: dict[str, tuple[int, int]]) -> list[str]:
    changed: list[str] = []
    for path, meta in after.items():
        if before.get(path) != meta:
            changed.append(path)
    for path in before:
        if path not in after:
            changed.append(path)
    return sorted(set(changed))


def _allowed_prefixes(root: Path, allowed_prefixes: list[Path]) -> list[str]:
    allowed: list[str] = []
    for prefix in allowed_prefixes:
        try:
            rel = prefix.relative_to(root).as_posix()
        except ValueError:
            continue
        rel_clean = rel.rstrip("/")
        if rel_clean in {"", "."}:
            allowed.append("/")
            continue
        if prefix.exists():
            if prefix.is_dir():
                allowed.append(rel_clean + "/")
            else:
                allowed.append(rel_clean)
            continue
        # Heuristic for not-yet-existing paths: treat dotted names as files.
        if Path(rel_clean).suffix:
            allowed.append(rel_clean)
        else:
            allowed.append(rel_clean + "/")
    return list(dict.fromkeys(allowed))


def _is_noise_path(path: str, denied_prefixes: list[str] | None = None) -> bool:
    denied = denied_prefixes or []
    if _is_denied(path, denied):
        return False
    if _is_event_delivery_state_path(path):
        return True
    if path.startswith("artifacts/") and "/kernels/" in path:
        return True
    if _is_volatile_run_submission_output(path):
        return True
    for prefix in _NOISE_PREFIXES:
        if path.startswith(prefix):
            return True
        if f"/{prefix.strip('/')}/" in path:
            return True
    if "/__pycache__/" in path or path.endswith("/__pycache__") or path == "__pycache__":
        return True
    for suffix in _NOISE_SUFFIXES:
        if path.endswith(suffix):
            return True
    return False


def _is_event_delivery_state_path(path: str) -> bool:
    parts = path.split("/")
    return len(parts) >= 3 and parts[-1] == _EVENT_DELIVERY_STATE_FILENAME and "_watch" in parts[:-1]


def _snapshot_external_guard_paths(paths: Sequence[Path]) -> dict[str, bytes | None]:
    snapshot: dict[str, bytes | None] = {}
    for path in paths:
        key = str(path)
        if key in snapshot:
            continue
        try:
            if path.is_file():
                snapshot[key] = path.read_bytes()
            elif path.is_dir():
                lines: list[str] = []
                for child in sorted(path.rglob("*")):
                    if not child.is_file():
                        continue
                    stat = child.stat()
                    rel = child.relative_to(path).as_posix()
                    lines.append(f"{rel}\t{stat.st_mtime_ns}\t{stat.st_size}")
                snapshot[key] = "\n".join(lines).encode("utf-8")
            else:
                snapshot[key] = None
        except OSError:
            snapshot[key] = None
    return snapshot


def _verify_external_guard_paths(*, stage: str, guard_snapshot: GuardSnapshot | None, paths: Sequence[Path]) -> None:
    if guard_snapshot is None or not guard_snapshot.external_backup:
        return
    current = _snapshot_external_guard_paths(paths)
    changed: list[str] = []
    for key, original in guard_snapshot.external_backup.items():
        if current.get(key) != original:
            changed.append(key)
    if changed:
        raise KaggleBotError(f"Agent write-guard failed in {stage}: forbidden external path edited: {sorted(changed)}")


def _is_volatile_run_submission_output(path: str) -> bool:
    parts = path.split("/")
    if len(parts) < 5:
        return False
    if parts[0] != "artifacts" or parts[2] != "runs":
        return False
    filename = Path(parts[-1])
    suffix = tabular_suffix(filename)
    if suffix not in _SAMPLE_SUBMISSION_SUFFIXES:
        return False
    return filename.name[: -len(suffix)] == "submission.compact"


def _is_protected_path(path: str) -> bool:
    if path.startswith("artifacts/"):
        # Protect competition-scoped control files so stray edits can be restored by the guard.
        parts = path.split("/")
        # artifacts/<slug>/meta.json or artifacts/<slug>/plan.json
        if len(parts) == 3 and parts[2] in {"meta.json", "plan.json"}:
            return True
        # artifacts/<slug>/data/sample_submission.*
        if _is_artifact_data_sample_submission(path):
            return True
        # artifacts/<slug>/submissions/ledger.jsonl
        if len(parts) == 4 and parts[2] == "submissions" and parts[3] == "ledger.jsonl":
            return True
        # artifacts/<slug>/{kernel,prompts}/...
        if len(parts) >= 4 and parts[2] in {"kernel", "prompts"}:
            return True
        return False
    for entry in _PROTECTED_PATHS:
        if entry.endswith("/"):
            if path.startswith(entry):
                return True
            continue
        if path == entry:
            return True
    return False


def _is_sensitive_repo_path(path: str) -> bool:
    name = Path(path).name.lower()
    if name == "kaggle.json":
        return True
    if name.startswith(".env"):
        return True
    return False


def _filter_restored_paths(root: Path, unauthorized: list[str], guard_snapshot: GuardSnapshot | None) -> list[str]:
    if guard_snapshot is None:
        return unauthorized
    filtered: list[str] = []
    for rel in unauthorized:
        path = root / rel
        if not path.exists():
            continue
        original = guard_snapshot.backup.get(rel)
        if original is None:
            if _matches_artifact_data_sample_submission_context(root, rel):
                continue
            filtered.append(rel)
            continue
        try:
            if path.read_bytes() == original:
                continue
        except OSError:
            filtered.append(rel)
            continue
        filtered.append(rel)
    return filtered


def _is_artifact_data_sample_submission(path: str) -> bool:
    parts = path.split("/")
    if len(parts) != 4 or parts[0] != "artifacts" or parts[2] != "data":
        return False
    sample_path = Path(parts[3])
    suffix = tabular_suffix(sample_path)
    if suffix not in _SAMPLE_SUBMISSION_SUFFIXES:
        return False
    return sample_path.name[: -len(suffix)] == "sample_submission"


def _restore_artifact_data_sample_submission(root: Path, rel: str) -> bool:
    if not _is_artifact_data_sample_submission(rel):
        return False
    parts = rel.split("/")
    source = _artifact_context_sample_submission_source(root, parts[1], tabular_suffix(Path(parts[3])))
    if not source.is_file():
        return False
    target = root / rel
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    except OSError:
        return False
    return True


def _matches_artifact_data_sample_submission_context(root: Path, rel: str) -> bool:
    if not _is_artifact_data_sample_submission(rel):
        return False
    parts = rel.split("/")
    source = _artifact_context_sample_submission_source(root, parts[1], tabular_suffix(Path(parts[3])))
    target = root / rel
    if not source.is_file() or not target.is_file():
        return False
    try:
        return target.read_bytes() == source.read_bytes()
    except OSError:
        return False


def _artifact_context_sample_submission_source(root: Path, slug: str, suffix: str) -> Path:
    normalized = suffix.lower()
    if normalized not in _SAMPLE_SUBMISSION_SUFFIXES:
        normalized = ".csv"
    return root / "artifacts" / slug / "context" / f"sample_submission{normalized}"


def _repair_unauthorized_changes(
    root: Path,
    unauthorized: list[str],
    guard_snapshot: GuardSnapshot,
    before: dict[str, tuple[int, int]],
    denied_prefixes: list[str],
) -> list[str]:
    errors: list[str] = []
    for rel in unauthorized:
        if _is_noise_path(rel, denied_prefixes):
            _remove_path(root / rel)
            continue
        if rel in guard_snapshot.oversized:
            if _restore_artifact_data_sample_submission(root, rel):
                continue
            errors.append(f"Cannot restore oversized file: {rel}")
            continue
        if rel in guard_snapshot.backup:
            try:
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(guard_snapshot.backup[rel])
            except OSError as exc:
                errors.append(f"Failed to restore {rel}: {exc}")
            continue
        if rel not in before:
            _remove_path(root / rel)
            continue
        if _is_protected_path(rel):
            if _restore_artifact_data_sample_submission(root, rel):
                continue
            errors.append(f"Cannot restore protected file: {rel}")
            continue
        errors.append(f"Cannot auto-repair changed file: {rel}")
    return errors


def _remove_path(path: Path) -> None:
    try:
        if path.is_file() or path.is_symlink():
            path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def _matches_prefix(path: str, prefixes: list[str]) -> bool:
    for prefix in prefixes:
        if prefix == "/":
            return True
        if prefix.endswith("/"):
            if path.startswith(prefix):
                return True
            continue
        if path == prefix:
            return True
    return False


def _is_denied(path: str, denied_prefixes: list[str]) -> bool:
    return _matches_prefix(path, denied_prefixes) or _is_sensitive_repo_path(path)


def _is_allowed(path: str, allowed_prefixes: list[str], denied_prefixes: list[str]) -> bool:
    return _matches_prefix(path, allowed_prefixes) and not _is_denied(path, denied_prefixes)


def _assert_no_secrets(text: str) -> None:
    matches = scan_text_for_secrets(text)
    if matches:
        raise KaggleBotError(f"Secret pattern detected in prompt text: {matches}")
