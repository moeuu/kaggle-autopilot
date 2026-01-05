from __future__ import annotations

import ast
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich import print

from kagglebot.exceptions import (
    KaggleCliError,
    KaggleNetworkError,
    KernelFailedError,
    KernelTimeoutError,
    RulesNotAcceptedError,
)
from kagglebot.kaggle_api import (
    check_rules_accepted,
    kernel_exists,
    kernel_id_by_title,
    kernels_init,
    kernels_output,
    kernels_push,
    kernels_status,
)
from kagglebot.logging_utils import truncate_lines
from kagglebot.solution_guard import ensure_solution_path_allowed
from kagglebot.validators import ensure_kernel_sources_valid, validate_kernel_package

_COLUMN_MAP_FILENAME = "column_map.json"
_COLUMN_MAP_SHIM_MARKER = "# kagglebot: column-map-shim"


@dataclass(frozen=True)
class KernelRunResult:
    kernel_id: str
    output_dir: Path
    submission_path: Path | None
    metrics_path: Path | None


def sanitize_kernel_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    return cleaned[:50]


_KERNEL_URL_RE = re.compile(
    r"https?://(?:www\.)?kaggle\.com/(?:code/)?(?P<user>[A-Za-z0-9_-]+)/(?P<slug>[A-Za-z0-9_.-]+)"
)
_KERNEL_ID_RE = re.compile(r"(?P<user>[A-Za-z0-9_-]+)/(?P<slug>[A-Za-z0-9_.-]+)")


def _extract_kernel_id_from_push(output: str) -> str | None:
    if not output:
        return None
    match = _KERNEL_URL_RE.search(output)
    if match:
        return f"{match.group('user')}/{match.group('slug')}"
    for line in output.splitlines():
        if "kernel" not in line.lower():
            continue
        match = _KERNEL_ID_RE.search(line)
        if match:
            return f"{match.group('user')}/{match.group('slug')}"
    return None


def find_submission_file(output_dir: Path) -> Path | None:
    candidate = _find_output_file(output_dir, "submission.csv")
    if candidate:
        return candidate
    return _find_submission_by_extension(output_dir)


def resolve_kaggle_username(explicit: str | None) -> str:
    if explicit:
        return explicit
    env_user = os.getenv("KAGGLE_USERNAME")
    if env_user:
        return env_user
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        data = json.loads(kaggle_json.read_text(encoding="utf-8"))
        if "username" in data:
            return str(data["username"])
    raise ValueError("Kaggle username not found. Provide --kaggle-username or set KAGGLE_USERNAME.")


def run_kernel(
    *,
    slug: str,
    run_id: str,
    iteration: int,
    base_dir: Path,
    kaggle_username: str,
    kernel_name: str | None,
    accelerator: str,
    enable_internet: bool,
    score_source: str,
    metric: str,
    direction: str,
    holdout_frac: float,
    cv_folds: int,
    seed: int,
    dry_run: bool,
    timeout_minutes: int | None,
) -> KernelRunResult:
    kernel_dir = base_dir / slug / "kernels" / run_id
    output_dir = base_dir / slug / "runs" / run_id / f"iter-{iteration}" / "output"
    logs_dir = base_dir / slug / "runs" / run_id / f"iter-{iteration}" / "logs"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not dry_run and not check_rules_accepted(slug, dry_run=False):
        raise RulesNotAcceptedError("Competition rules not accepted.")

    if not dry_run:
        print(f"[cyan]kernel init[/cyan]: {kernel_dir}")
        kernels_init(kernel_dir, dry_run=False)

    kernel_slug = _resolve_kernel_slug(kernel_name, slug, run_id, iteration)
    kernel_id = f"{kaggle_username}/{kernel_slug}"
    custom_kernel_dir = base_dir / slug / "kernel"
    ensure_solution_path_allowed(custom_kernel_dir, artifacts_dir=base_dir, slug=slug)
    if custom_kernel_dir.exists():
        _copy_kernel_sources(custom_kernel_dir, kernel_dir)
        _ensure_kernel_import_path(kernel_dir)
        _inline_kernel_modules(kernel_dir)
        _inject_data_dir_resolver(kernel_dir)
        _inject_column_map_shim(kernel_dir, base_dir / slug / "context")
        ensure_kernel_sources_valid(kernel_dir)
    else:
        _write_kernel_script(
            kernel_dir=kernel_dir,
            slug=slug,
            accelerator=accelerator,
            score_source=score_source,
            metric=metric,
            direction=direction,
            holdout_frac=holdout_frac,
            cv_folds=cv_folds,
            seed=seed,
            run_id=run_id,
            iteration=iteration,
        )
    _write_kernel_metadata(
        kernel_dir=kernel_dir,
        kernel_id=kernel_id,
        title=kernel_slug,
        code_file="kernel.py",
        accelerator=accelerator,
        enable_internet=enable_internet,
        competition_slug=slug,
    )
    validate_kernel_package(kernel_dir)

    if dry_run:
        return KernelRunResult(kernel_id=kernel_id, output_dir=output_dir, submission_path=None, metrics_path=None)

    print(f"[cyan]kernel push[/cyan]: {kernel_dir}")
    push_attempt = 1
    push_output = kernels_push(kernel_dir, slug=slug, dry_run=False)
    _write_push_log(logs_dir, push_attempt, push_output)
    pushed_kernel_id = _extract_kernel_id_from_push(push_output)
    if pushed_kernel_id and pushed_kernel_id != kernel_id:
        print(f"[cyan]kernel id[/cyan]: {pushed_kernel_id}")
        kernel_id = pushed_kernel_id
    kernel_id = _resolve_kernel_id(kernel_id, kernel_slug)
    resolved_id = _wait_for_kernel_registration(kernel_id, kernel_slug)
    if not resolved_id:
        print("[yellow]kernel not found after push[/yellow]: retrying once")
        push_attempt += 1
        push_output = kernels_push(kernel_dir, slug=slug, dry_run=False)
        _write_push_log(logs_dir, push_attempt, push_output)
        pushed_kernel_id = _extract_kernel_id_from_push(push_output)
        if pushed_kernel_id and pushed_kernel_id != kernel_id:
            print(f"[cyan]kernel id[/cyan]: {pushed_kernel_id}")
            kernel_id = pushed_kernel_id
        kernel_id = _resolve_kernel_id(kernel_id, kernel_slug)
        resolved_id = _wait_for_kernel_registration(kernel_id, kernel_slug)
        if not resolved_id:
            raise KernelFailedError("Kaggle kernel not found after push; aborting.")
        kernel_id = resolved_id
    else:
        kernel_id = resolved_id
    print(f"[cyan]kernel status[/cyan]: {kernel_id}")
    _wait_for_kernel(kernel_id, slug, timeout_minutes, output_dir=output_dir)
    print(f"[cyan]kernel output[/cyan]: {output_dir}")
    kernels_output(kernel_id, output_dir, slug=slug, dry_run=False)

    submission_path = find_submission_file(output_dir)
    metrics_path = _find_output_file(output_dir, "metrics.json")
    return KernelRunResult(
        kernel_id=kernel_id, output_dir=output_dir, submission_path=submission_path, metrics_path=metrics_path
    )


def _resolve_kernel_slug(kernel_name: str | None, slug: str, run_id: str, iteration: int) -> str:
    if kernel_name:
        return sanitize_kernel_slug(kernel_name)
    suffix = f"{run_id[-6:]}-i{iteration}"
    prefix = f"kagglebot-{slug}"
    max_len = 50
    allowed_prefix_len = max_len - len(suffix) - 1
    if allowed_prefix_len < 1:
        prefix = "kagglebot"
    else:
        prefix = prefix[:allowed_prefix_len].rstrip("-")
    base = f"{prefix}-{suffix}"
    return sanitize_kernel_slug(base)


def _write_kernel_metadata(
    *,
    kernel_dir: Path,
    kernel_id: str,
    title: str,
    code_file: str,
    accelerator: str,
    enable_internet: bool,
    competition_slug: str,
) -> None:
    meta_path = kernel_dir / "kernel-metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {}
    meta.update(
        {
            "id": kernel_id,
            "title": title,
            "code_file": code_file,
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": accelerator == "gpu",
            "enable_tpu": accelerator == "tpu",
            "enable_internet": bool(enable_internet),
            "competition_sources": [competition_slug],
            "dataset_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
    )
    if meta["enable_gpu"] and meta["enable_tpu"]:
        raise ValueError("kernel-metadata.json cannot enable both GPU and TPU.")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _write_kernel_script(
    *,
    kernel_dir: Path,
    slug: str,
    accelerator: str,
    score_source: str,
    metric: str,
    direction: str,
    holdout_frac: float,
    cv_folds: int,
    seed: int,
    run_id: str,
    iteration: int,
) -> None:
    script = _render_kernel_main(
        slug=slug,
        accelerator=accelerator,
        score_source=score_source,
        metric=metric,
        direction=direction,
        holdout_frac=holdout_frac,
        cv_folds=cv_folds,
        seed=seed,
        run_id=run_id,
        iteration=iteration,
    )
    (kernel_dir / "kernel.py").write_text(script, encoding="utf-8")


def _copy_kernel_sources(source_dir: Path, dest_dir: Path) -> None:
    for path in source_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, dest_dir / path.name)


_KERNEL_BOOTSTRAP_MARKER = "# kagglebot:kernel_sys_path"
_KERNEL_BOOTSTRAP_END = "del _os, _sys, _KROOT, _KWORK"
_KERNEL_DATA_RESOLVER_MARKER = "# kagglebot:data_resolver"
_DATA_DIR_JOIN_RE = re.compile(r"(\bdata_dir\s*/\s*)(['\"])([^'\"]+)\2")


def _strip_kernel_bootstrap(lines: list[str]) -> list[str]:
    stripped = lines
    while _KERNEL_BOOTSTRAP_MARKER in stripped:
        start = stripped.index(_KERNEL_BOOTSTRAP_MARKER)
        end = None
        search_end = min(start + 20, len(stripped))
        for idx in range(start + 1, search_end):
            if stripped[idx].strip() == _KERNEL_BOOTSTRAP_END:
                end = idx + 1
                break
        if end is None:
            stripped = stripped[:start] + stripped[start + 1 :]
        else:
            stripped = stripped[:start] + stripped[end:]
    return stripped


def _ensure_kernel_import_path(kernel_dir: Path) -> None:
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    bootstrap = (
        f"{_KERNEL_BOOTSTRAP_MARKER}\n"
        "import os as _os\n"
        "import sys as _sys\n"
        "try:\n"
        "    _KROOT = _os.path.dirname(_os.path.abspath(__file__))\n"
        "except NameError:\n"
        "    _KROOT = _os.getcwd()\n"
        "if _KROOT not in _sys.path:\n"
        "    _sys.path.insert(0, _KROOT)\n"
        "_KWORK = '/kaggle/working'\n"
        "if _KWORK not in _sys.path:\n"
        "    _sys.path.insert(0, _KWORK)\n"
        "del _os, _sys, _KROOT, _KWORK\n"
    )
    lines = _strip_kernel_bootstrap(text.splitlines())
    insert_at = _find_bootstrap_insertion_index(lines)
    bootstrap_lines = bootstrap.splitlines()
    new_lines = lines[:insert_at] + bootstrap_lines + lines[insert_at:]
    new_text = "\n".join(new_lines)
    if text.endswith("\n"):
        new_text += "\n"
    kernel_path.write_text(new_text, encoding="utf-8")


def _inject_data_dir_resolver(kernel_dir: Path) -> None:
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    if _KERNEL_DATA_RESOLVER_MARKER in text:
        return
    if not _DATA_DIR_JOIN_RE.search(text):
        return
    resolver_block = [
        _KERNEL_DATA_RESOLVER_MARKER,
        "from pathlib import Path as _KBPath",
        "",
        "def _kb_find_file(base: _KBPath, name: str) -> _KBPath:",
        "    candidate = base / name",
        "    if candidate.exists():",
        "        return candidate",
        "    try:",
        "        matches = list(base.rglob(name))",
        "    except Exception:",
        "        matches = []",
        "    if matches:",
        "        return matches[0]",
        "    return candidate",
        "",
    ]
    lines = text.splitlines()
    insert_at = _find_bootstrap_block_end(lines)
    if insert_at is None:
        insert_at = _find_bootstrap_insertion_index(lines)
    lines = lines[:insert_at] + resolver_block + lines[insert_at:]
    updated = "\n".join(lines)
    updated = _DATA_DIR_JOIN_RE.sub(r"_kb_find_file(data_dir, '\3')", updated)
    if text.endswith("\n"):
        updated += "\n"
    kernel_path.write_text(updated, encoding="utf-8")


def _inject_column_map_shim(kernel_dir: Path, context_dir: Path) -> None:
    map_path = context_dir / _COLUMN_MAP_FILENAME
    if not map_path.exists():
        return
    kernel_map_path = kernel_dir / _COLUMN_MAP_FILENAME
    shutil.copy2(map_path, kernel_map_path)
    site_path = kernel_dir / "sitecustomize.py"
    shim = [
        _COLUMN_MAP_SHIM_MARKER,
        "import json",
        "from pathlib import Path",
        "",
        "def _kb_load_map() -> dict:",
        "    candidates = [",
        f"        Path(__file__).with_name('{_COLUMN_MAP_FILENAME}'),",
        f"        Path('/kaggle/working/{_COLUMN_MAP_FILENAME}'),",
        "    ]",
        "    for path in candidates:",
        "        if path.exists():",
        "            try:",
        "                payload = json.loads(path.read_text(encoding='utf-8'))",
        "            except Exception:",
        "                continue",
        "            mapping = payload.get('mapping') if isinstance(payload, dict) else None",
        "            if isinstance(mapping, dict) and mapping:",
        "                return mapping",
        "    return {}",
        "",
        "def _kb_patch_pandas() -> None:",
        "    try:",
        "        import pandas as _pd",
        "    except Exception:",
        "        return",
        "    mapping = _kb_load_map()",
        "    if not mapping:",
        "        return",
        "    _orig = _pd.read_csv",
        "    def _patched(*args, **kwargs):",
        "        df = _orig(*args, **kwargs)",
        "        try:",
        "            return df.rename(columns=mapping)",
        "        except Exception:",
        "            return df",
        "    _pd.read_csv = _patched",
        "",
        "_kb_patch_pandas()",
        "",
    ]
    if site_path.exists():
        text = site_path.read_text(encoding="utf-8", errors="ignore")
        if _COLUMN_MAP_SHIM_MARKER in text:
            return
        site_path.write_text(text.rstrip("\n") + "\n\n" + "\n".join(shim), encoding="utf-8")
        return
    site_path.write_text("\n".join(shim), encoding="utf-8")


def _find_bootstrap_block_end(lines: list[str]) -> int | None:
    if _KERNEL_BOOTSTRAP_MARKER not in lines:
        return None
    start = lines.index(_KERNEL_BOOTSTRAP_MARKER)
    search_end = min(start + 30, len(lines))
    for idx in range(start + 1, search_end):
        if lines[idx].strip() == _KERNEL_BOOTSTRAP_END:
            return idx + 1
    return None


def _find_bootstrap_insertion_index(lines: list[str]) -> int:
    idx = 0
    if idx < len(lines) and lines[idx].startswith("#!"):
        idx += 1
    for _ in range(2):
        if idx < len(lines) and re.match(r"^#.*coding[:=]\s*[-\w.]+", lines[idx]):
            idx += 1
    while idx < len(lines) and (lines[idx].strip() == "" or lines[idx].lstrip().startswith("#")):
        idx += 1
    if idx < len(lines):
        stripped = lines[idx].lstrip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            quote = '"""' if stripped.startswith('"""') else "'''"
            if stripped.count(quote) >= 2:
                idx += 1
            else:
                idx += 1
                while idx < len(lines) and quote not in lines[idx]:
                    idx += 1
                if idx < len(lines):
                    idx += 1
    while idx < len(lines) and (lines[idx].strip() == "" or lines[idx].lstrip().startswith("#")):
        idx += 1
    while idx < len(lines) and re.match(r"^\s*from\s+__future__\s+import\s+", lines[idx]):
        idx += 1
    return idx


def _inline_kernel_modules(kernel_dir: Path, modules: tuple[str, ...] | None = None) -> None:
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    if modules is None:
        modules = _discover_inline_modules(kernel_dir, lines)
    if not modules or not _kernel_imports_local_modules(lines, modules):
        return
    alias_modules = _modules_with_alias_imports(lines, modules)
    if alias_modules:
        modules = tuple(module for module in modules if module not in alias_modules)
        if not modules:
            return

    stripped = lines
    for module in modules:
        stripped = _strip_module_import(stripped, module)

    module_blocks: list[str] = []
    for module in modules:
        module_path = kernel_dir / f"{module}.py"
        if not module_path.exists():
            continue
        module_lines = module_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        cleaned = _strip_module_headers(module_lines)
        cleaned = _strip_local_module_imports(cleaned, modules)
        if not cleaned:
            continue
        module_blocks.append(f"# --- Begin inlined module: {module}.py ---")
        module_blocks.extend(cleaned)
        module_blocks.append(f"# --- End inlined module: {module}.py ---")

    if not module_blocks:
        return

    insert_at = _find_main_guard_index(stripped)
    new_lines = stripped[:insert_at] + [""] + module_blocks + [""] + stripped[insert_at:]
    new_text = "\n".join(new_lines)
    if text.endswith("\n"):
        new_text += "\n"
    kernel_path.write_text(new_text, encoding="utf-8")


def _kernel_imports_local_modules(lines: list[str], modules: tuple[str, ...]) -> bool:
    for line in lines:
        for module in modules:
            if re.match(rf"^\s*from\s+\.?{re.escape(module)}\s+import\b", line):
                return True
            if re.match(rf"^\s*import\s+{re.escape(module)}\b", line):
                return True
    return False


def _modules_with_alias_imports(lines: list[str], modules: tuple[str, ...]) -> set[str]:
    if not modules:
        return set()
    text = "\n".join(lines)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _modules_with_alias_imports_fallback(lines, modules)

    alias_modules: set[str] = set()
    module_set = set(modules)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base = alias.name.split(".", 1)[0]
                if base in module_set and alias.asname:
                    alias_modules.add(base)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            base = node.module.split(".", 1)[0]
            if base not in module_set:
                continue
            for alias in node.names:
                if alias.asname:
                    alias_modules.add(base)
                    break
    return alias_modules


def _modules_with_alias_imports_fallback(lines: list[str], modules: tuple[str, ...]) -> set[str]:
    alias_modules: set[str] = set()
    for line in lines:
        for module in modules:
            if re.match(rf"^\s*import\s+{re.escape(module)}\s+as\s+\w+", line):
                alias_modules.add(module)
                continue
            if re.match(rf"^\s*from\s+\.?{re.escape(module)}\s+import\b", line):
                if " as " in line:
                    alias_modules.add(module)
    return alias_modules


def _strip_module_import(lines: list[str], module: str) -> list[str]:
    output: list[str] = []
    skipping = False
    paren_depth = 0
    for line in lines:
        if not skipping:
            if re.match(rf"^\s*from\s+\.?{re.escape(module)}\s+import\b", line):
                skipping = True
                paren_depth = line.count("(") - line.count(")")
                if paren_depth <= 0 and not line.rstrip().endswith("\\"):
                    skipping = False
                continue
            if re.match(rf"^\s*import\s+{re.escape(module)}\b", line):
                continue
            output.append(line)
            continue
        paren_depth += line.count("(") - line.count(")")
        if paren_depth <= 0 and not line.rstrip().endswith("\\"):
            skipping = False
        continue
    return output


def _discover_inline_modules(kernel_dir: Path, lines: list[str]) -> tuple[str, ...]:
    module_names: list[str] = []
    for path in kernel_dir.glob("*.py"):
        if path.name == "kernel.py":
            continue
        name = path.stem
        if name.isidentifier():
            module_names.append(name)
    if not module_names:
        return ()
    used: list[str] = []
    for name in module_names:
        if _kernel_imports_local_modules(lines, (name,)):
            used.append(name)
    return tuple(used)


def _strip_module_headers(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    for line in lines:
        if not cleaned and line.startswith("#!"):
            continue
        if not cleaned and re.match(r"^#.*coding[:=]\s*[-\w.]+", line):
            continue
        if re.match(r"^\s*from\s+__future__\s+import\s+", line):
            continue
        cleaned.append(line)
    while cleaned and cleaned[0].strip() == "":
        cleaned.pop(0)
    return cleaned


def _strip_local_module_imports(lines: list[str], modules: tuple[str, ...]) -> list[str]:
    cleaned = lines
    for module in modules:
        cleaned = _strip_module_import(cleaned, module)
    return cleaned


def _find_main_guard_index(lines: list[str]) -> int:
    for idx, line in enumerate(lines):
        if re.match(r"^\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", line):
            return idx
    return len(lines)


LOG_POLL_INTERVAL = 2.0
HEARTBEAT_INTERVAL = 30.0
STATUS_ERROR_SLEEP = 10.0
MAX_STATUS_ERRORS = 6
KERNEL_REGISTER_RETRIES = 24
KERNEL_REGISTER_SLEEP = 5.0


def _wait_for_kernel(kernel_id: str, slug: str, timeout_minutes: int | None, *, output_dir: Path) -> None:
    deadline = None
    if timeout_minutes is not None:
        deadline = time.monotonic() + max(timeout_minutes, 1) * 60
    last_status = None
    last_log_fetch = 0.0
    log_state = _KernelLogState()
    status_errors = 0
    while True:
        try:
            output = kernels_status(kernel_id, slug=slug, dry_run=False)
            status_errors = 0
        except KaggleCliError as exc:
            status_errors += 1
            detail = (exc.output or str(exc)).strip()
            if detail:
                detail = detail.replace("\n", " ")
            if isinstance(exc, KaggleNetworkError):
                message = (
                    f"[yellow]kernel status network error[/yellow]: {detail or 'unknown error'} "
                    f"(attempt {status_errors})"
                )
                print(message)
                if deadline is not None and time.monotonic() > deadline:
                    raise KernelTimeoutError("Kaggle kernel did not complete within timeout.") from exc
                if MAX_STATUS_ERRORS is not None and status_errors >= MAX_STATUS_ERRORS:
                    kernel_url = f"https://www.kaggle.com/code/{kernel_id}"
                    raise KaggleNetworkError(
                        "Kaggle API unreachable while polling kernel status. "
                        f"Check network/DNS and monitor the kernel at {kernel_url}.",
                        getattr(exc, "command", None),
                        exit_code=getattr(exc, "exit_code", None),
                        output=getattr(exc, "output", ""),
                    ) from exc
                time.sleep(STATUS_ERROR_SLEEP)
                continue
            message = f"[yellow]kernel status failed[/yellow]: {detail or 'unknown error'} (attempt {status_errors})"
            print(message)
            if deadline is not None and time.monotonic() > deadline:
                raise KernelTimeoutError("Kaggle kernel did not complete within timeout.") from exc
            if MAX_STATUS_ERRORS is not None and status_errors >= MAX_STATUS_ERRORS:
                raise KernelFailedError(
                    f"Kaggle kernel status failed {status_errors} times. Last error: {detail or 'unknown error'}"
                ) from exc
            time.sleep(STATUS_ERROR_SLEEP)
            continue
        status = _parse_kernel_status(output).lower()
        if status != last_status:
            print(f"[cyan]kernel status[/cyan]: {status}")
            last_status = status
        now = time.monotonic()
        if now - last_log_fetch >= LOG_POLL_INTERVAL:
            _try_fetch_kernel_output(kernel_id, output_dir=output_dir, slug=slug)
            had_logs = _print_kernel_logs(output_dir, log_state)
            if had_logs:
                log_state.last_log_at = now
            last_log_fetch = now
            log_failure = _detect_failure_in_logs(output_dir)
            if log_failure:
                log_failure = truncate_lines(log_failure, max_lines=5)
                message = f"Kaggle kernel error detected in logs.\n\n--- kernel log tail ---\n{log_failure}"
                raise KernelFailedError(message)
        if status in {"running", "queued"}:
            if log_state.last_heartbeat == 0.0 or now - log_state.last_heartbeat >= HEARTBEAT_INTERVAL:
                since = now - log_state.last_log_at if log_state.last_log_at is not None else None
                if since is None:
                    print("[cyan]kernel[/cyan]: still running (no logs yet)")
                else:
                    print(f"[cyan]kernel[/cyan]: still running (no new logs for {since:.0f}s)")
                log_state.last_heartbeat = now
        if "complete" in status:
            return
        if "error" in status or "fail" in status:
            _try_fetch_kernel_output(kernel_id, output_dir=output_dir, slug=slug)
            log_tail = _collect_log_tail(output_dir)
            message = f"Kaggle kernel failed: {output}"
            if log_tail:
                log_tail = truncate_lines(log_tail, max_lines=5)
                message = f"{message}\n\n--- kernel log tail ---\n{log_tail}"
            raise KernelFailedError(message)
        time.sleep(STATUS_ERROR_SLEEP)
        if deadline is not None and time.monotonic() > deadline:
            raise KernelTimeoutError("Kaggle kernel did not complete within timeout.")


@dataclass
class _KernelLogState:
    seen_lines: dict[Path, int] = field(default_factory=dict)
    seen_json: dict[Path, int] = field(default_factory=dict)
    seen_size: dict[Path, int] = field(default_factory=dict)
    last_log_at: float | None = None
    last_heartbeat: float = 0.0


def _wait_for_kernel_registration(kernel_id: str, kernel_slug: str) -> str | None:
    for attempt in range(1, KERNEL_REGISTER_RETRIES + 1):
        try:
            kernels_status(kernel_id, dry_run=False)
            return kernel_id
        except KaggleCliError as exc:
            detail = (exc.output or str(exc)).strip().replace("\n", " ")
            if detail:
                print(f"[yellow]kernel status unavailable[/yellow]: {detail} (attempt {attempt})")
        try:
            if kernel_exists(kernel_id):
                return kernel_id
            resolved = kernel_id_by_title(kernel_slug)
            if resolved:
                return resolved
        except KaggleCliError as exc:
            detail = (exc.output or str(exc)).strip().replace("\n", " ")
            if detail:
                print(f"[yellow]kernel list failed[/yellow]: {detail} (attempt {attempt})")
        time.sleep(KERNEL_REGISTER_SLEEP)
    return None


def _resolve_kernel_id(kernel_id: str, kernel_slug: str) -> str:
    try:
        resolved = kernel_id_by_title(kernel_slug)
    except KaggleCliError:
        return kernel_id
    if resolved and resolved != kernel_id:
        print(f"[cyan]kernel id[/cyan]: {resolved}")
        return resolved
    return kernel_id


def _write_push_log(logs_dir: Path, attempt: int, output: str) -> None:
    path = logs_dir / f"kernel_push-{attempt:02d}.txt"
    path.write_text(output.strip() + "\n", encoding="utf-8")


def _parse_kernel_status(output: str) -> str:
    match = re.search(r"status\\s+\\\"?([A-Za-z0-9_.-]+)\\\"?", output)
    if match:
        return match.group(1)
    return output.strip() or "unknown"


def _try_fetch_kernel_output(kernel_id: str, *, output_dir: Path, slug: str) -> None:
    try:
        kernels_output(kernel_id, output_dir, slug=slug, dry_run=False, force=True, quiet=True)
    except KaggleCliError:
        return


def _log_candidates(output_dir: Path) -> list[Path]:
    candidates = []
    for name in ("stdout.txt", "stderr.txt", "output.log", "log.txt", "logs.txt"):
        path = output_dir / name
        if path.exists():
            candidates.append(path)
    candidates.extend(sorted(output_dir.rglob("*.log")))
    return candidates


def _print_kernel_logs(output_dir: Path, state: _KernelLogState) -> bool:
    printed = False
    for path in _log_candidates(output_dir):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        size = len(text)
        prev_size = state.seen_size.get(path, 0)
        if size < prev_size:
            state.seen_lines[path] = 0
            state.seen_json[path] = 0
        state.seen_size[path] = size

        json_events = _parse_json_log(text)
        if json_events is not None:
            last = state.seen_json.get(path, 0)
            if len(json_events) <= last:
                continue
            new_events = json_events[last:]
            state.seen_json[path] = len(json_events)
            formatted = _format_log_events(new_events)
            if not formatted:
                continue
            print(f"[cyan]kernel log[/cyan]: {path.name}")
            print(truncate_lines("\n".join(formatted), max_lines=5))
            printed = True
            continue

        lines = text.splitlines()
        last = state.seen_lines.get(path, 0)
        if len(lines) <= last:
            continue
        new_lines = lines[last:]
        state.seen_lines[path] = len(lines)
        print(f"[cyan]kernel log[/cyan]: {path.name}")
        print(truncate_lines("\n".join(new_lines), max_lines=5))
        printed = True
    return printed


def _detect_failure_in_logs(output_dir: Path) -> str | None:
    for path in _log_candidates(output_dir):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "Traceback (most recent call last)" not in text:
            continue
        tail = _collect_log_tail(output_dir)
        if tail:
            return tail
        return f"{path.name}\nTraceback detected"
    return None


def _collect_log_tail(output_dir: Path, max_lines: int = 50) -> str | None:
    for path in _log_candidates(output_dir):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        json_events = _parse_json_log(text)
        if json_events is not None:
            formatted = _format_log_events(json_events)
            if not formatted:
                continue
            tail = "\n".join(formatted[-max_lines:])
            return f"{path.name}\n{tail}".strip()
        lines = text.splitlines()
        if not lines:
            continue
        tail = "\n".join(lines[-max_lines:])
        return f"{path.name}\n{tail}".strip()
    return None


def _parse_json_log(text: str) -> list[dict[str, object]] | None:
    stripped = text.lstrip()
    if not stripped:
        return None
    if stripped.startswith("["):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return None
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict) and isinstance(payload.get("logs"), list):
            return [item for item in payload["logs"] if isinstance(item, dict)]
        return None
    return None


def _format_log_events(events: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for event in events:
        data = event.get("data")
        if not isinstance(data, str) or not data:
            continue
        stream = event.get("stream_name")
        prefix = f"[{stream}] " if isinstance(stream, str) and stream else ""
        for line in data.splitlines():
            lines.append(f"{prefix}{line}")
    return lines


def _find_output_file(output_dir: Path, filename: str) -> Path | None:
    candidate = output_dir / filename
    if candidate.exists():
        return candidate
    matches = list(output_dir.rglob(filename))
    if matches:
        return matches[0]
    return None


def _find_submission_by_extension(output_dir: Path) -> Path | None:
    suffixes = [".csv", ".tsv", ".txt", ".parquet", ".json", ".jsonl"]
    for suffix in suffixes:
        candidate = output_dir / f"submission{suffix}"
        if candidate.exists():
            return candidate
    for path in output_dir.rglob("submission.*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in suffixes:
            return path
    return None


def _render_kernel_main(
    *,
    slug: str,
    accelerator: str,
    score_source: str,
    metric: str,
    direction: str,
    holdout_frac: float,
    cv_folds: int,
    seed: int,
    run_id: str,
    iteration: int,
) -> str:
    return f'''import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

CONFIG = {{
    "slug": "{slug}",
    "accelerator": "{accelerator}",
    "score_source": "{score_source}",
    "metric": "{metric}",
    "direction": "{direction}",
    "holdout_frac": {holdout_frac},
    "cv_folds": {cv_folds},
    "seed": {seed},
    "run_id": "{run_id}",
    "iteration": {iteration},
}}

INPUT_ROOT = Path("/kaggle/input") / CONFIG["slug"]
WORKING = Path("/kaggle/working")
SUBMISSION_PATH = WORKING / "submission.csv"
METRICS_PATH = WORKING / "metrics.json"


def find_tabular_files(root: Path) -> list[Path]:
    suffixes = {".csv", ".tsv", ".txt", ".parquet", ".json", ".jsonl"}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes]


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\\t")
    return pd.read_csv(path)


def pick_files(files: list[Path]) -> tuple[Path, Path, Path]:
    if not files:
        raise FileNotFoundError("No tabular files found.")
    def score_sample(path: Path) -> int:
        name = path.name.lower()
        if "sample_submission" in name:
            return 3
        if "submission" in name:
            return 1
        return 0
    sample = sorted(files, key=score_sample, reverse=True)[0]
    train = next((p for p in files if "train" in p.name.lower()), None)
    test = next((p for p in files if "test" in p.name.lower()), None)
    if train is None or test is None:
        raise FileNotFoundError("train/test files not found.")
    return train, test, sample


def infer_target(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> tuple[str, str, list[str]]:
    id_col = sample.columns[0]
    candidates = [c for c in train.columns if c not in test.columns and c in sample.columns]
    target_cols = candidates or list(sample.columns[1:])
    if len(target_cols) != 1:
        raise ValueError("Only single-target competitions supported.")
    target = target_cols[0]
    features = [c for c in train.columns if c != target]
    if id_col in features:
        features.remove(id_col)
    return id_col, target, features


def infer_task(y: pd.Series) -> str:
    if y.dtype == "object":
        return "classification"
    nunique = y.nunique(dropna=True)
    if nunique <= 20 or nunique / max(len(y), 1) <= 0.05:
        return "classification"
    return "regression"


def metric_requires_proba(metric: str) -> bool:
    metric = metric.lower()
    return "logloss" in metric or "auc" in metric


def compute_metric(metric: str, y_true, y_pred) -> float:
    metric = metric.lower()
    if metric == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if metric == "rmsle":
        y_true = np.clip(np.asarray(y_true, dtype=float), 0, None)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
        return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred))))
    if metric in ("logloss", "log_loss"):
        return float(log_loss(y_true, y_pred))
    if metric == "auc":
        return float(roc_auc_score(y_true, y_pred))
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _as_numpy(x):
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x)


class TorchMLP:
    def __init__(self, task: str, hidden: int = 128, epochs: int = 20, lr: float = 1e-3, batch_size: int = 256):
        self.task = task
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model = None
        self.num_classes = None
        self.device = None

    def _init_model(self, input_dim: int, num_classes: int):
        if torch is None or nn is None:
            raise RuntimeError("torch is not available")
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        output_dim = 1 if (self.task == "regression" or num_classes <= 2) else num_classes
        self.model = nn.Sequential(
            nn.Linear(input_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden, output_dim),
        ).to(self.device)

    def fit(self, x, y):
        if torch is None or DataLoader is None or TensorDataset is None:
            raise RuntimeError("torch is not available")
        x_np = _as_numpy(x).astype(np.float32)
        y_np = np.asarray(y)
        num_classes = int(np.unique(y_np).size) if self.task != "regression" else 1
        if self.model is None:
            self._init_model(x_np.shape[1], num_classes)
        torch.manual_seed(CONFIG["seed"])
        if self.task == "regression":
            y_tensor = torch.tensor(y_np.astype(np.float32).reshape(-1, 1))
            loss_fn = nn.MSELoss()
        elif num_classes <= 2:
            y_tensor = torch.tensor(y_np.astype(np.float32).reshape(-1, 1))
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            y_tensor = torch.tensor(y_np.astype(np.int64))
            loss_fn = nn.CrossEntropyLoss()
        dataset = TensorDataset(torch.tensor(x_np), y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, x):
        if torch is None:
            raise RuntimeError("torch is not available")
        x_np = _as_numpy(x).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(x_np).to(self.device))
            if self.task == "regression":
                return logits.cpu().numpy().reshape(-1)
            if self.num_classes is None or self.num_classes <= 2:
                probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
                return probs
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

    def predict(self, x):
        if self.task == "regression":
            return self.predict_proba(x)
        probs = self.predict_proba(x)
        if self.num_classes is None or self.num_classes <= 2:
            return (probs >= 0.5).astype(int)
        return probs.argmax(axis=1)


def feature_engineering(
    train: pd.DataFrame,
    test: pd.DataFrame,
    id_col: str,
    target_col: str,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    return train, test, features


def build_preprocessor(features: list[str], train: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in features if train[c].dtype == "object"]
    num_cols = [c for c in features if c not in cat_cols]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


def find_label_file(root: Path) -> Path | None:
    for name in ["test_labels.csv", "labels.csv", "y_test.csv"]:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def select_score_source(test: pd.DataFrame, target_col: str, id_col: str) -> tuple[str, pd.Series | None]:
    source = CONFIG["score_source"]
    if source in ("auto", "test"):
        if target_col in test.columns:
            return "test", test[target_col]
        label_path = find_label_file(INPUT_ROOT)
        if label_path is not None:
            labels = pd.read_csv(label_path)
            if target_col in labels.columns and id_col in labels.columns:
                merged = test.merge(labels[[id_col, target_col]], on=id_col, how="inner")
                if not merged.empty:
                    return "test", merged[target_col]
        if source == "test":
            raise RuntimeError("score_source=test requested but no labeled test found.")
        return "holdout", None
    return source, None


def predict_for_metric(model, x, task: str, metric: str):
    if task == "classification" and metric_requires_proba(metric):
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x)
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            return proba
    return model.predict(x)


def predict_for_submission(model, x, task: str, metric: str, prediction_kind: str):
    if task == "classification" and (metric_requires_proba(metric) or prediction_kind == "probability"):
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x)
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            return proba
    return model.predict(x)


def _slice_rows(values, idx):
    if hasattr(values, "iloc"):
        return values.iloc[idx]
    return values[idx]


def evaluate_holdout(model, pre, x, y, task: str, metric: str, prediction_kind: str):
    stratify = y if task == "classification" else None
    x_tr, x_val, y_tr, y_val = train_test_split(
        x, y, test_size=CONFIG["holdout_frac"], random_state=CONFIG["seed"], stratify=stratify
    )
    x_tr_p = pre.fit_transform(x_tr)
    x_val_p = pre.transform(x_val)
    model.fit(x_tr_p, y_tr)
    preds = predict_for_metric(model, x_val_p, task, metric)
    return compute_metric(metric, y_val, preds), None


def evaluate_cv(model_builder, pre, x, y, task: str, metric: str, prediction_kind: str):
    splitter = (
        StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=CONFIG["seed"])
        if task == "classification"
        else KFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=CONFIG["seed"])
    )
    scores = []
    for train_idx, val_idx in splitter.split(x, y):
        x_tr, x_val = _slice_rows(x, train_idx), _slice_rows(x, val_idx)
        y_tr, y_val = _slice_rows(y, train_idx), _slice_rows(y, val_idx)
        x_tr_p = pre.fit_transform(x_tr)
        x_val_p = pre.transform(x_val)
        model = model_builder(task)
        model.fit(x_tr_p, y_tr)
        preds = predict_for_metric(model, x_val_p, task, metric)
        scores.append(compute_metric(metric, y_val, preds))
    return float(np.mean(scores)), float(np.std(scores))


def build_model(task: str):
    if torch is not None:
        return TorchMLP(task)
    if CONFIG["accelerator"] == "gpu":
        try:
            import xgboost as xgb
            if task == "classification":
                return xgb.XGBClassifier(tree_method="gpu_hist", max_depth=6, n_estimators=200, learning_rate=0.1)
            return xgb.XGBRegressor(tree_method="gpu_hist", max_depth=6, n_estimators=200, learning_rate=0.1)
        except Exception:
            pass
    if task == "classification":
        return HistGradientBoostingClassifier()
    return HistGradientBoostingRegressor()


def train_tpu(x_train, y_train, x_eval, task: str):
    import tensorflow as tf
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)

    with strategy.scope():
        output_units = 1 if task == "regression" else int(np.unique(y_train).size)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(x_train.shape[1],)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(output_units),
            ]
        )
        if task == "classification":
            if output_units > 2:
                model.add(tf.keras.layers.Activation("softmax"))
                model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            else:
                model.add(tf.keras.layers.Activation("sigmoid"))
                model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        else:
            model.compile(optimizer="adam", loss="mse")

    model.fit(x_train, y_train, epochs=5, batch_size=256, verbose=0)
    outputs = model.predict(x_eval, batch_size=256, verbose=0)
    return outputs


def main() -> None:
    if "custom_main" in globals():
        custom_main = globals()["custom_main"]
        if callable(custom_main):
            custom_main()
            return
    train_path, test_path, sample_path = pick_files(find_tabular_files(INPUT_ROOT))
    train = read_table(train_path)
    test = read_table(test_path)
    sample = read_table(sample_path)

    id_col, target_col, features = infer_target(train, test, sample)
    train, test, features = feature_engineering(train, test, id_col, target_col, features)
    task = infer_task(train[target_col])
    prediction_kind = "probability" if sample[target_col].dtype.kind in {{"f", "c"}} else "class"

    label_encoder = None
    y = train[target_col]
    if task == "classification":
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), index=train.index, name=target_col)

    x = train[features]
    pre = build_preprocessor(features, train)
    score_source, test_labels = select_score_source(test, target_col, id_col)

    std = None
    if CONFIG["accelerator"] == "tpu":
        x_full = pre.fit_transform(x)
        if hasattr(x_full, "toarray"):
            x_full = x_full.toarray()
        if score_source == "cv":
            scores = []
            splitter = (
                StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=CONFIG["seed"])
                if task == "classification"
                else KFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=CONFIG["seed"])
            )
            for train_idx, val_idx in splitter.split(x_full, y):
                preds = train_tpu(x_full[train_idx], y[train_idx], x_full[val_idx], task)
                scores.append(compute_metric(CONFIG["metric"], y[val_idx], preds))
            score = float(np.mean(scores))
            std = float(np.std(scores))
        else:
            preds = train_tpu(x_full, y, x_full, task)
            score = compute_metric(CONFIG["metric"], y, preds)
    else:
        if score_source == "cv":
            score, std = evaluate_cv(build_model, pre, x, y, task, CONFIG["metric"], prediction_kind)
        elif score_source == "test" and test_labels is not None:
            model = build_model(task)
            x_train_p = pre.fit_transform(x)
            x_test_p = pre.transform(test[features])
            model.fit(x_train_p, y)
            preds = predict_for_metric(model, x_test_p, task, CONFIG["metric"])
            score = compute_metric(CONFIG["metric"], test_labels, preds)
        else:
            model = build_model(task)
            score, std = evaluate_holdout(model, pre, x, y, task, CONFIG["metric"], prediction_kind)

    x_full = pre.fit_transform(x)
    if hasattr(x_full, "toarray"):
        x_full = x_full.toarray()
    if CONFIG["accelerator"] == "tpu":
        test_features = pre.transform(test[features])
        if hasattr(test_features, "toarray"):
            test_features = test_features.toarray()
        preds = train_tpu(x_full, y, test_features, task)
    else:
        model = build_model(task)
        model.fit(x_full, y)
        test_x = pre.transform(test[features])
        preds = predict_for_submission(model, test_x, task, CONFIG["metric"], prediction_kind)
    if task == "classification" and prediction_kind == "class" and label_encoder is not None:
        if preds.ndim > 1:
            preds = preds.argmax(axis=1)
        preds = label_encoder.inverse_transform(np.asarray(preds, dtype=int))

    submission = sample.copy()
    submission[target_col] = preds
    submission.to_csv(SUBMISSION_PATH, index=False)

    metrics = {{
        "run_id": CONFIG["run_id"],
        "iter": CONFIG["iteration"],
        "score_source": score_source,
        "metric": CONFIG["metric"],
        "direction": CONFIG["direction"],
        "offline_value": float(score),
        "offline_std": float(std) if std is not None else None,
        "folds": CONFIG["cv_folds"] if score_source == "cv" else None,
        "holdout_frac": CONFIG["holdout_frac"] if score_source == "holdout" else None,
        "seed": CONFIG["seed"],
        "target_score": None,
        "met_target": False,
        "top1_public_score": None,
        "top1_public_timestamp": None,
        "compare_to_top1_note": "heuristic; not directly comparable",
        "compute": "kaggle_{accelerator}",
        "accelerator": "{accelerator}",
        "git_commit": None,
        "timestamp": int(datetime.utcnow().timestamp()),
    }}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
'''
