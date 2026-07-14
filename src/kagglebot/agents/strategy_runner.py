from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import mkdtemp
from urllib.error import URLError
from urllib.request import urlopen

from kagglebot.agents.identity import STRATEGY_AGENT, render_prompt_identity, resolve_oracle_model
from kagglebot.agents.sandbox_fallback import (
    append_sandbox_args,
    detect_sandbox_startup_failure,
    resolve_agent_sandbox_mode,
)
from kagglebot.exec_utils import CommandResult, run_command

_DEFAULT_MODEL = STRATEGY_AGENT.model
_DEFAULT_REASONING_EFFORT = STRATEGY_AGENT.reasoning_effort
_DEFAULT_TIMEOUT_SEC = 600.0
_PYTEST_TIMEOUT_SEC = 2.0
_RUNNER_LABEL = STRATEGY_AGENT.log_alias
_DEFAULT_ORACLE_BROWSER_PORT = 9222
_ORACLE_BROWSER_READY_TIMEOUT_SEC = 15.0
_DEFAULT_ORACLE_BROWSER_INPUT_TIMEOUT = "600s"
_DEFAULT_ORACLE_BROWSER_TIMEOUT = "24h"
_DEFAULT_ORACLE_BROWSER_THINKING_TIME = "extended"
_DEFAULT_ORACLE_DATA_ATTACHMENT_MAX_BYTES = 100 * 1024 * 1024
_DEFAULT_ORACLE_FILE_SIZE_LIMIT_BYTES = 1024 * 1024
_ORACLE_RUNTIME_CONTEXT_FILE_MAX_BYTES = 4 * 1024 * 1024
_ORACLE_RUNTIME_CONTEXT_TOTAL_MAX_BYTES = 16 * 1024 * 1024
_ORACLE_REMOTE_DATA_PART_BYTES = 15 * 1024 * 1024
_ORACLE_CANONICAL_CONTEXT_FILES = (
    "rules_url.txt",
    "rules.md",
    "overview.md",
    "data.md",
    "submission_format.md",
    "dataset_profile.json",
    "competition_policy.json",
    "evaluation_spec.json",
    "top1_public.json",
    "reference_inputs_manifest.json",
    "knowledge_hints.txt",
    "code.md",
    "models.md",
    "discussion.md",
    "code_notebooks_index.json",
    "discussion_threads_index.json",
    "research_sources.jsonl",
    "research_summary.md",
    "method_registry.json",
    "source_registry.json",
    "validation_registry.json",
    "validation_lab_report.json",
    "win_contract.json",
    "private_robustness_report.json",
    "top1_exhaustion_report.json",
    "top1_exhaustion_report.md",
)
_ORACLE_DATA_EGRESS_BLOCK_MARKERS = (
    "not to transmit",
    "do not transmit",
    "not transmit",
    "party not participating",
    "persons who have not formally agreed",
    "private sharing outside of teams",
    "privately sharing code or data outside of teams",
)
_ORACLE_BENIGN_USE_PREAMBLE = (
    "Authorized benign use: this is offline data-science work for a Kaggle competition the operator has joined. "
    "Analyze only supplied competition artifacts and public research. Limit recommendations to competition "
    "modeling, validation, and submission generation; do not propose interacting with external systems or "
    "real-world targets. "
)
_ORACLE_CHROME_PROFILE_ROOT_EXCLUDES = (
    "DevToolsActivePort",
    "SingletonCookie",
    "SingletonLock",
    "SingletonSocket",
)


@dataclass(frozen=True)
class StrategyResult:
    transcript_path: Path
    last_message_path: Path
    returncode: int
    stdout: str
    stderr: str
    sandbox_policy_mode: str = "permissive"
    used_sandbox_fallback: bool = False
    sandbox_failure_excerpt: str | None = None
    engine: str = "codex"


@dataclass
class OracleBrowserBootstrap:
    args: list[str]
    process: subprocess.Popen[bytes] | None = None
    temp_profile_dir: Path | None = None

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.temp_profile_dir is not None:
            shutil.rmtree(self.temp_profile_dir, ignore_errors=True)


@dataclass
class OracleBrowserCompatibility:
    process: subprocess.Popen[bytes] | None = None

    def close(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


@dataclass(frozen=True)
class OracleAttachmentPlan:
    paths: tuple[Path, ...]
    data_paths: tuple[Path, ...]
    data_decision: str


@dataclass(frozen=True)
class OracleDataDelivery:
    source_path: Path
    paths: tuple[Path, ...]
    sha256: str


def run_strategy(prompt_path: Path, output_dir: Path, *, dry_run: bool = False, engine: str = "auto") -> StrategyResult:
    resolved_engine = _resolve_strategy_engine(engine)
    if resolved_engine == "oracle":
        return _run_oracle_strategy(prompt_path, output_dir, dry_run=dry_run)
    return _run_codex_strategy(prompt_path, output_dir, dry_run=dry_run)


def _run_codex_strategy(prompt_path: Path, output_dir: Path, *, dry_run: bool = False) -> StrategyResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_text = render_prompt_identity(prompt_path.read_text(encoding="utf-8"))
    transcript_path = output_dir / "strategy_exec.txt"
    last_message_path = output_dir / "strategy_last_message.txt"

    if dry_run:
        transcript_path.write_text("", encoding="utf-8")
        last_message_path.write_text("DRY RUN: strategy not executed.\n", encoding="utf-8")
        return StrategyResult(
            transcript_path=transcript_path,
            last_message_path=last_message_path,
            returncode=0,
            stdout="",
            stderr="",
            sandbox_policy_mode=resolve_agent_sandbox_mode(),
            engine="codex",
        )

    normalized_effort = _normalize_reasoning_effort(_DEFAULT_REASONING_EFFORT)
    timeout = float(os.environ.get("KAGGLEBOT_STRATEGY_TIMEOUT_SEC", str(_DEFAULT_TIMEOUT_SEC)))
    if os.environ.get("PYTEST_CURRENT_TEST"):
        timeout = float(os.environ.get("KAGGLEBOT_PYTEST_STRATEGY_TIMEOUT_SEC", str(_PYTEST_TIMEOUT_SEC)))
    args = [
        STRATEGY_AGENT.cli_command,
        "exec",
        "-m",
        _DEFAULT_MODEL,
        "-c",
        f'model_reasoning_effort="{normalized_effort}"',
    ]
    supported = _supported_flags()
    sandbox_policy_mode = resolve_agent_sandbox_mode()
    sandbox_mode = "workspace-write" if sandbox_policy_mode in {"fallback", "workspace-write"} else "danger-full-access"
    dangerously_bypass = sandbox_policy_mode == "permissive"
    append_sandbox_args(
        args,
        supported,
        sandbox_mode=sandbox_mode,
        dangerously_bypass=dangerously_bypass,
        include_full_auto=not dangerously_bypass,
    )
    if "--search" in supported:
        args.append("--search")
    args += [
        "--output-last-message",
        str(last_message_path),
        "-",
    ]
    stop_event = threading.Event()
    start_time = time.monotonic()
    print(f"{_RUNNER_LABEL}: sandbox mode {sandbox_policy_mode}", flush=True)
    print(f"{_RUNNER_LABEL} running... (0s total)", flush=True)
    heartbeat = threading.Thread(target=_heartbeat, args=(stop_event, start_time), daemon=True)
    heartbeat.start()
    try:
        result = run_command(args, input_text=prompt_text, timeout=timeout)
        sandbox_failure_excerpt = detect_sandbox_startup_failure(
            result.stdout,
            result.stderr,
            last_message_path.read_text(encoding="utf-8", errors="ignore") if last_message_path.exists() else "",
        )
        used_sandbox_fallback = False
        if sandbox_policy_mode == "fallback" and result.returncode != 0 and sandbox_failure_excerpt is not None:
            used_sandbox_fallback = True
            print(f"{_RUNNER_LABEL}: sandbox startup failed; retrying without sandbox", flush=True)
            retry_args = [
                STRATEGY_AGENT.cli_command,
                "exec",
                "-m",
                _DEFAULT_MODEL,
                "-c",
                f'model_reasoning_effort="{normalized_effort}"',
            ]
            append_sandbox_args(
                retry_args,
                supported,
                sandbox_mode="danger-full-access",
                dangerously_bypass=True,
                include_full_auto=False,
            )
            if "--search" in supported:
                retry_args.append("--search")
            retry_args += [
                "--output-last-message",
                str(last_message_path),
                "-",
            ]
            retry_result = run_command(retry_args, input_text=prompt_text, timeout=timeout)
            stdout_text = result.stdout + retry_result.stdout
            stderr_chunks = [chunk for chunk in (result.stderr, retry_result.stderr) if chunk.strip()]
            stderr_text = "\n\n".join(stderr_chunks)
            if sandbox_failure_excerpt:
                if stderr_text:
                    stderr_text = f"{sandbox_failure_excerpt}\n\n{stderr_text}"
                else:
                    stderr_text = sandbox_failure_excerpt
            result = CommandResult(
                args=retry_result.args,
                returncode=retry_result.returncode,
                stdout=stdout_text,
                stderr=stderr_text,
                duration_sec=retry_result.duration_sec,
            )
        else:
            used_sandbox_fallback = False
    except subprocess.TimeoutExpired:
        total_elapsed = int(time.monotonic() - start_time)
        stop_event.set()
        heartbeat.join(timeout=1.0)
        message = f"Strategy runner timed out after {int(timeout)}s (elapsed={total_elapsed}s)."
        transcript_path.write_text(message + "\n", encoding="utf-8")
        last_message_path.write_text(message + "\n", encoding="utf-8")
        return StrategyResult(
            transcript_path=transcript_path,
            last_message_path=last_message_path,
            returncode=124,
            stdout=message,
            stderr=message,
            sandbox_policy_mode=sandbox_policy_mode,
            engine="codex",
        )
    finally:
        stop_event.set()
        heartbeat.join(timeout=1.0)
    total_elapsed = int(time.monotonic() - start_time)
    print(f"{_RUNNER_LABEL} done... ({total_elapsed}s total, exit={result.returncode})", flush=True)
    transcript_path.write_text(result.stdout, encoding="utf-8")
    last_message = last_message_path.read_text(encoding="utf-8").strip() if last_message_path.exists() else ""
    if not last_message:
        last_message = result.stdout.strip()
        last_message_path.write_text((last_message + "\n") if last_message else "", encoding="utf-8")
    return StrategyResult(
        transcript_path=transcript_path,
        last_message_path=last_message_path,
        returncode=result.returncode,
        stdout=last_message,
        stderr=result.stderr,
        sandbox_policy_mode=sandbox_policy_mode,
        used_sandbox_fallback=used_sandbox_fallback,
        sandbox_failure_excerpt=sandbox_failure_excerpt if used_sandbox_fallback else None,
        engine="codex",
    )


def _run_oracle_strategy(prompt_path: Path, output_dir: Path, *, dry_run: bool = False) -> StrategyResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_prompt = render_prompt_identity(prompt_path.read_text(encoding="utf-8"))
    oracle_prompt_path = output_dir / "oracle_strategy_prompt.md"
    oracle_prompt_path.write_text(rendered_prompt, encoding="utf-8")
    transcript_path = output_dir / "strategy_exec.txt"
    last_message_path = output_dir / "strategy_last_message.txt"

    if dry_run:
        transcript_path.write_text("", encoding="utf-8")
        last_message_path.write_text("DRY RUN: oracle strategy not executed.\n", encoding="utf-8")
        return StrategyResult(
            transcript_path=transcript_path,
            last_message_path=last_message_path,
            returncode=0,
            stdout="",
            stderr="",
            sandbox_policy_mode="external",
            engine="oracle",
        )

    command = _oracle_command()
    if not command:
        message = "Oracle strategy runner unavailable: empty KAGGLEBOT_ORACLE_COMMAND."
        transcript_path.write_text(message + "\n", encoding="utf-8")
        last_message_path.write_text(message + "\n", encoding="utf-8")
        return StrategyResult(
            transcript_path=transcript_path,
            last_message_path=last_message_path,
            returncode=127,
            stdout=message,
            stderr=message,
            sandbox_policy_mode="external",
            engine="oracle",
        )

    model = resolve_oracle_model()
    consult_prompt = (
        _ORACLE_BENIGN_USE_PREAMBLE
        + "Read the attached Kagglebot strategy prompt file and any attached Kagglebot context bundle files. "
        "Treat this as a single-turn consultation with no prior session memory; all required context is attached. "
        "Read oracle_context_manifest.md first and treat full canonical context attachments as authoritative. "
        "If the manifest lists split competition-data parts, concatenate them in order and verify the listed SHA-256 "
        "before inspecting the reconstructed archive. "
        "Return exactly the delimiter sections requested by the attached strategy prompt; do not invent sections. "
        "Do not omit source evidence when the prompt requires it."
    )
    extra_args = _oracle_extra_args()
    inline_prompt = _oracle_inline_prompt_enabled(extra_args)
    if inline_prompt:
        consult_prompt = (
            _ORACLE_BENIGN_USE_PREAMBLE
            + "Use the Kagglebot strategy prompt below together with every attached canonical context file. "
            "Read oracle_context_manifest.md first; attached full files override trimmed inline excerpts. "
            "If competition data is split into parts, reconstruct it exactly as directed by the manifest. "
            "Return only the delimiter sections requested inside it; do not invent sections that its contract "
            "does not request.\n\n"
            f"{rendered_prompt}"
        )
    attachment_plan = _build_oracle_attachment_plan(
        prompt_path=prompt_path,
        oracle_prompt_path=oracle_prompt_path,
        output_dir=output_dir,
        inline_prompt=inline_prompt,
        split_large_data=_oracle_browser_engine_requested(extra_args),
    )
    attachment_paths = list(attachment_plan.paths)
    browser_bootstrap = _maybe_start_oracle_browser(extra_args)
    args = [
        *command,
        *_oracle_engine_args(extra_args),
        *extra_args,
        *browser_bootstrap.args,
        *_oracle_wait_args(extra_args),
        *_oracle_force_args(extra_args),
        *_oracle_max_file_size_args(extra_args, attachment_paths),
        "--model",
        model,
        "--write-output",
        str(transcript_path),
        "-p",
        consult_prompt,
    ]
    if attachment_paths:
        args += ["--file", *[str(path) for path in attachment_paths]]
    timeout = _oracle_strategy_timeout()

    stop_event = threading.Event()
    start_time = time.monotonic()
    print("oracle strategy running... (0s total)", flush=True)
    heartbeat = threading.Thread(
        target=_heartbeat,
        args=(stop_event, start_time),
        kwargs={"label": "oracle strategy"},
        daemon=True,
    )
    heartbeat.start()
    browser_compatibility = _start_oracle_browser_attachment_compatibility(
        args=[*extra_args, *browser_bootstrap.args],
        attachment_paths=attachment_paths,
    )
    try:
        archive_report: dict[str, object] | None = None
        try:
            transcript_path.unlink(missing_ok=True)
            last_message_path.unlink(missing_ok=True)
            (output_dir / "oracle_archive.json").unlink(missing_ok=True)
            result = run_command(args, timeout=timeout)
            if _oracle_browser_engine_requested(extra_args):
                archive_report = _ensure_oracle_conversation_archived(
                    transcript_path=transcript_path,
                    output_dir=output_dir,
                    browser_bootstrap=browser_bootstrap,
                    extra_args=extra_args,
                )
        except FileNotFoundError:
            message = f"Oracle strategy runner unavailable: executable not found: {command[0]}"
            transcript_path.write_text(message + "\n", encoding="utf-8")
            last_message_path.write_text(message + "\n", encoding="utf-8")
            return StrategyResult(
                transcript_path=transcript_path,
                last_message_path=last_message_path,
                returncode=127,
                stdout=message,
                stderr=message,
                sandbox_policy_mode="external",
                engine="oracle",
            )
    except subprocess.TimeoutExpired:
        total_elapsed = int(time.monotonic() - start_time)
        stop_event.set()
        heartbeat.join(timeout=1.0)
        message = f"Oracle strategy runner timed out after {int(timeout)}s (elapsed={total_elapsed}s)."
        transcript_path.write_text(message + "\n", encoding="utf-8")
        last_message_path.write_text(message + "\n", encoding="utf-8")
        return StrategyResult(
            transcript_path=transcript_path,
            last_message_path=last_message_path,
            returncode=124,
            stdout=message,
            stderr=message,
            sandbox_policy_mode="external",
            engine="oracle",
        )
    finally:
        stop_event.set()
        heartbeat.join(timeout=1.0)
        browser_compatibility.close()
        browser_bootstrap.close()

    total_elapsed = int(time.monotonic() - start_time)
    transcript_text = transcript_path.read_text(encoding="utf-8", errors="ignore") if transcript_path.exists() else ""
    returncode = result.returncode
    stderr = result.stderr
    if returncode != 0 and transcript_text.strip():
        returncode = 0
        stderr = "\n".join(
            part
            for part in (
                stderr,
                f"Oracle command exited {result.returncode} after writing a response; validating the response content.",
            )
            if part
        )
    if archive_report is not None and archive_report.get("archived") is not True:
        archive_error = str(archive_report.get("fallbackReason") or archive_report.get("reason") or "unknown")
        stderr = "\n".join(
            part for part in (stderr, f"Oracle conversation archive verification warning: {archive_error}") if part
        )
    print(f"oracle strategy done... ({total_elapsed}s total, exit={returncode})", flush=True)
    stdout_text = transcript_text.strip() or result.stdout.strip()
    if not transcript_text:
        transcript_path.write_text((stdout_text + "\n") if stdout_text else "", encoding="utf-8")
    last_message_path.write_text((stdout_text + "\n") if stdout_text else "", encoding="utf-8")
    return StrategyResult(
        transcript_path=transcript_path,
        last_message_path=last_message_path,
        returncode=returncode,
        stdout=stdout_text,
        stderr=stderr,
        sandbox_policy_mode="external",
        engine="oracle",
    )


def _resolve_strategy_engine(engine: str | None) -> str:
    return resolve_strategy_engine(engine)


def resolve_strategy_engine(engine: str | None = None) -> str:
    requested = (engine or os.environ.get("KAGGLEBOT_STRATEGY_ENGINE") or "auto").strip().lower()
    if requested == "auto":
        return "oracle"
    if requested in {"oracle", "codex"}:
        return requested
    raise ValueError(f"Unsupported strategy engine: {requested}")


def _oracle_command() -> list[str]:
    raw = os.environ.get("KAGGLEBOT_ORACLE_COMMAND", "oracle").strip()
    return shlex.split(raw) if raw else []


def _oracle_extra_args() -> list[str]:
    raw = os.environ.get("KAGGLEBOT_ORACLE_ARGS", "").strip()
    return shlex.split(raw) if raw else []


def _oracle_engine_args(extra_args: list[str]) -> list[str]:
    if _oracle_args_include_option(extra_args, "--engine", "-e"):
        return []
    requested = os.environ.get("KAGGLEBOT_ORACLE_ENGINE", "browser").strip().lower()
    if requested in {"api", "browser"}:
        return ["--engine", requested]
    if requested == "auto":
        return []
    return ["--engine", "browser"]


def _oracle_wait_args(extra_args: list[str]) -> list[str]:
    if _oracle_args_include_option(extra_args, "--wait"):
        return []
    return ["--wait"]


def _oracle_strategy_timeout() -> float | None:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return float(os.environ.get("KAGGLEBOT_PYTEST_STRATEGY_TIMEOUT_SEC", str(_PYTEST_TIMEOUT_SEC)))
    raw = os.environ.get("KAGGLEBOT_ORACLE_STRATEGY_TIMEOUT_SEC")
    if raw is None:
        raw = os.environ.get("KAGGLEBOT_STRATEGY_TIMEOUT_SEC")
    return None if raw is None or not raw.strip() else float(raw)


def _oracle_force_args(extra_args: list[str]) -> list[str]:
    if not _env_flag("KAGGLEBOT_ORACLE_FORCE", default=True):
        return []
    if _oracle_args_include_option(extra_args, "--force"):
        return []
    return ["--force"]


def _oracle_inline_prompt_enabled(extra_args: list[str]) -> bool:
    if _oracle_args_include_option(extra_args, "--file", "-f"):
        return False
    return _env_flag("KAGGLEBOT_ORACLE_INLINE_PROMPT", default=True)


def _build_oracle_attachment_plan(
    *,
    prompt_path: Path,
    oracle_prompt_path: Path,
    output_dir: Path,
    inline_prompt: bool,
    split_large_data: bool = False,
) -> OracleAttachmentPlan:
    context_paths: list[Path] = [] if inline_prompt else [oracle_prompt_path]
    for candidate in _oracle_context_bundle_candidates(prompt_path):
        if candidate.exists() and candidate.is_file() and candidate not in context_paths:
            context_paths.append(candidate)

    context_dir = _oracle_context_dir(prompt_path)
    data_paths: list[Path] = []
    data_decision = "not evaluated: no competition context directory was found"
    if context_dir is not None:
        for name in _ORACLE_CANONICAL_CONTEXT_FILES:
            _append_existing_file(context_paths, context_dir / name)
        for pattern in ("sample_submission_head.*", "sample_submission.*"):
            for candidate in sorted(context_dir.glob(pattern)):
                _append_existing_file(context_paths, candidate)
        _append_existing_file(context_paths, context_dir / "agent" / "brief_for_strategy.md")
        data_paths, data_decision = _oracle_competition_data_attachments(context_dir)

    if not context_paths and context_dir is None:
        return OracleAttachmentPlan(
            paths=tuple(context_paths),
            data_paths=tuple(data_paths),
            data_decision=data_decision,
        )

    context_bundle_path = output_dir / "oracle_canonical_context.md"
    _write_oracle_canonical_context_bundle(context_bundle_path, context_paths)
    data_deliveries = _prepare_oracle_data_deliveries(
        data_paths=data_paths,
        output_dir=output_dir,
        split_large_data=split_large_data,
    )

    manifest_path = output_dir / "oracle_context_manifest.md"
    manifest_path.write_text(
        _render_oracle_context_manifest(
            inline_prompt=inline_prompt,
            context_paths=context_paths,
            context_bundle_path=context_bundle_path,
            data_paths=data_paths,
            data_deliveries=data_deliveries,
            data_decision=data_decision,
        ),
        encoding="utf-8",
    )
    delivery_paths = list(context_paths) if context_dir is None else []
    delivery_paths.append(context_bundle_path)
    for delivery in data_deliveries:
        delivery_paths.extend(delivery.paths)
    delivery_paths.append(manifest_path)
    return OracleAttachmentPlan(
        paths=tuple(delivery_paths),
        data_paths=tuple(data_paths),
        data_decision=data_decision,
    )


def _write_oracle_canonical_context_bundle(bundle_path: Path, context_paths: list[Path]) -> None:
    lines = [
        "# Oracle Canonical Context Bundle",
        "",
        "This file contains the complete text of every canonical context source listed below.",
    ]
    for path in context_paths:
        content = path.read_text(encoding="utf-8", errors="replace")
        lines.extend(
            [
                "",
                f"===== BEGIN FILE: {path} ({path.stat().st_size} bytes) =====",
                content.rstrip("\n"),
                f"===== END FILE: {path} =====",
            ]
        )
    bundle_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_oracle_data_deliveries(
    *,
    data_paths: list[Path],
    output_dir: Path,
    split_large_data: bool,
) -> list[OracleDataDelivery]:
    parts_dir = output_dir / "oracle_data_parts"
    shutil.rmtree(parts_dir, ignore_errors=True)
    deliveries: list[OracleDataDelivery] = []
    for source_path in data_paths:
        digest = _sha256_file(source_path)
        if not split_large_data or source_path.stat().st_size <= _ORACLE_REMOTE_DATA_PART_BYTES:
            deliveries.append(OracleDataDelivery(source_path=source_path, paths=(source_path,), sha256=digest))
            continue

        parts_dir.mkdir(parents=True, exist_ok=True)
        source_size = source_path.stat().st_size
        part_count = (source_size + _ORACLE_REMOTE_DATA_PART_BYTES - 1) // _ORACLE_REMOTE_DATA_PART_BYTES
        part_paths: list[Path] = []
        with source_path.open("rb") as source:
            for index in range(1, part_count + 1):
                part_path = parts_dir / f"{source_path.name}.part-{index:03d}-of-{part_count:03d}.zip"
                part_path.write_bytes(source.read(_ORACLE_REMOTE_DATA_PART_BYTES))
                part_paths.append(part_path)
        deliveries.append(OracleDataDelivery(source_path=source_path, paths=tuple(part_paths), sha256=digest))
    return deliveries


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _oracle_context_dir(prompt_path: Path) -> Path | None:
    for candidate in prompt_path.parents:
        context_candidate = candidate if candidate.name == "context" else candidate / "context"
        if any((context_candidate / name).exists() for name in ("rules.md", "overview.md", "dataset_profile.json")):
            return context_candidate
    return None


def _append_existing_file(paths: list[Path], candidate: Path) -> None:
    if candidate.exists() and candidate.is_file() and candidate not in paths:
        paths.append(candidate)


def _oracle_competition_data_attachments(context_dir: Path) -> tuple[list[Path], str]:
    mode = os.environ.get("KAGGLEBOT_ORACLE_COMPETITION_DATA", "auto").strip().lower()
    if mode not in {"auto", "never", "owner-authorized"}:
        mode = "auto"
    if mode == "never":
        return [], "omitted: KAGGLEBOT_ORACLE_COMPETITION_DATA=never"

    if mode == "auto":
        rules_path = context_dir / "rules.md"
        rules_text = rules_path.read_text(encoding="utf-8", errors="ignore").lower() if rules_path.exists() else ""
        if not rules_text.strip():
            return [], "omitted: competition rules are unavailable, so data egress cannot be verified"
        marker = next((item for item in _ORACLE_DATA_EGRESS_BLOCK_MARKERS if item in rules_text), None)
        if marker is not None:
            return [], f'omitted: competition rules restrict third-party data transmission (matched "{marker}")'

    data_dir = context_dir.parent / "data"
    candidates = _oracle_competition_package_candidates(data_dir=data_dir, slug=context_dir.parent.name)
    if not candidates:
        return [], "omitted: no canonical downloaded competition package was found"

    max_bytes = _oracle_data_attachment_max_bytes()
    total_bytes = sum(path.stat().st_size for path in candidates)
    if total_bytes > max_bytes:
        return [], f"omitted: canonical package size {total_bytes} bytes exceeds limit {max_bytes} bytes"
    authorization = "; owner-authorized processing" if mode == "owner-authorized" else ""
    return candidates, (
        f"attached: {len(candidates)} canonical package file(s), {total_bytes} bytes total{authorization}"
    )


def _oracle_competition_package_candidates(*, data_dir: Path, slug: str) -> list[Path]:
    if not data_dir.exists():
        return []
    exact_zip = data_dir / f"{slug}.zip"
    if exact_zip.exists() and exact_zip.is_file():
        return [exact_zip]
    archive_suffixes = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".7z")
    return sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file() and any(path.name.lower().endswith(suffix) for suffix in archive_suffixes)
    )


def _oracle_data_attachment_max_bytes() -> int:
    raw = os.environ.get(
        "KAGGLEBOT_ORACLE_DATA_ATTACHMENT_MAX_BYTES",
        str(_DEFAULT_ORACLE_DATA_ATTACHMENT_MAX_BYTES),
    ).strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_ORACLE_DATA_ATTACHMENT_MAX_BYTES


def _render_oracle_context_manifest(
    *,
    inline_prompt: bool,
    context_paths: list[Path],
    context_bundle_path: Path,
    data_paths: list[Path],
    data_deliveries: list[OracleDataDelivery],
    data_decision: str,
) -> str:
    lines = [
        "# Oracle Context Manifest",
        "",
        f"- rendered_prompt_delivery: {'inline' if inline_prompt else 'file attachment'}",
        f"- canonical_context_source_count: {len(context_paths)}",
        f"- canonical_context_delivery: {context_bundle_path} ({context_bundle_path.stat().st_size} bytes)",
        f"- competition_data: {data_decision}",
        "",
        "## Canonical Context Sources",
    ]
    lines.extend(f"- {path} ({path.stat().st_size} bytes)" for path in context_paths)
    if data_paths:
        lines.extend(["", "## Competition Data Packages"])
        for delivery in data_deliveries:
            lines.append(
                f"- source: {delivery.source_path} ({delivery.source_path.stat().st_size} bytes; "
                f"sha256={delivery.sha256})"
            )
            if delivery.paths == (delivery.source_path,):
                lines.append(f"  - attachment: {delivery.source_path}")
                continue
            lines.append("  - reconstruction: concatenate the following parts in listed order as raw bytes")
            lines.extend(f"  - part: {path} ({path.stat().st_size} bytes)" for path in delivery.paths)
            lines.append("  - verify: reconstructed file SHA-256 must match the source SHA-256 above")
    lines.extend(
        [
            "",
            "Read oracle_canonical_context.md as the authoritative full text when an inline excerpt is trimmed.",
            "When a package is split, concatenate its parts before opening the reconstructed archive.",
            "Do not assume access to any competition data package listed as omitted.",
        ]
    )
    return "\n".join(lines) + "\n"


def _oracle_max_file_size_args(extra_args: list[str], attachment_paths: list[Path]) -> list[str]:
    if not attachment_paths or _oracle_args_include_option(extra_args, "--max-file-size-bytes"):
        return []
    largest = max(path.stat().st_size for path in attachment_paths)
    if largest <= _DEFAULT_ORACLE_FILE_SIZE_LIMIT_BYTES:
        return []
    return ["--max-file-size-bytes", str(largest)]


def _oracle_context_bundle_candidates(prompt_path: Path) -> list[Path]:
    candidates = [
        prompt_path.parent / "strategy_context_bundle.md",
        prompt_path.parent.parent / "strategy_context_bundle.md",
    ]
    context_dir = _oracle_context_dir(prompt_path)
    if context_dir is not None:
        competition_dir = context_dir.parent
        candidates.extend(
            [
                competition_dir / "plan.json",
                competition_dir / "kernel" / "kernel.py",
            ]
        )
        run_dir = _oracle_run_dir(prompt_path)
        if run_dir is not None:
            candidates.extend(
                [
                    run_dir / "run.json",
                    run_dir / "run_state.json",
                    run_dir / "compute_handoff.json",
                    run_dir / "submit_failure_context.json",
                ]
            )
        iter_dir = _oracle_iteration_dir(prompt_path)
        if iter_dir is not None:
            candidates.extend(_oracle_iteration_context_candidates(iter_dir))
            logs_dir = iter_dir / "logs"
            if logs_dir.exists():
                candidates.extend(
                    path
                    for path in sorted(logs_dir.rglob("*"))
                    if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".log", ".md", ".txt"}
                )
        for parent in prompt_path.parents:
            if parent == run_dir:
                break
            candidates.extend(sorted(parent.glob("error*.txt")))
            candidates.extend(sorted(parent.glob("kernel_error*.txt")))
            candidates.extend(sorted(parent.glob("submit_failure*.json")))

    self_improvement_dir = _oracle_self_improvement_dir(prompt_path)
    if self_improvement_dir is not None:
        candidates.extend(
            self_improvement_dir / name
            for name in (
                "latest.json",
                "latest.md",
                "strategy_context.md",
                "experiment_backlog.json",
                "skill_candidates.json",
                "outcomes.jsonl",
            )
        )
    return _bounded_unique_context_paths(candidates)


def _oracle_run_dir(prompt_path: Path) -> Path | None:
    for candidate in prompt_path.parents:
        if candidate.parent.name == "runs":
            return candidate
    return None


def _oracle_iteration_dir(prompt_path: Path) -> Path | None:
    run_dir = _oracle_run_dir(prompt_path)
    if run_dir is None:
        return None
    for candidate in prompt_path.parents:
        if candidate.parent == run_dir and candidate.name.startswith("iter-"):
            return candidate
    return None


def _oracle_iteration_context_candidates(iter_dir: Path) -> list[Path]:
    return [
        iter_dir / "metrics.json",
        iter_dir / "diagnostics.md",
        iter_dir / "evaluation_report.json",
        iter_dir / "iteration_state.json",
        iter_dir / "compute_handoff.json",
        iter_dir / "experiment_graph.json",
        iter_dir / "allocator_decision.json",
        iter_dir / "graph_execution_report.json",
        iter_dir / "validation_lab_report.json",
        iter_dir / "private_robustness_report.json",
        iter_dir / "portfolio_optimizer_report.json",
        iter_dir / "top1_exhaustion_report.json",
        iter_dir / "blend_report.json",
        iter_dir / "portfolio_plan.json",
        iter_dir / "output" / "metrics.json",
    ]


def _oracle_self_improvement_dir(prompt_path: Path) -> Path | None:
    return next((candidate for candidate in prompt_path.parents if candidate.name == "_self_improvement"), None)


def _bounded_unique_context_paths(candidates: list[Path]) -> list[Path]:
    selected: list[Path] = []
    selected_bytes = 0
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_file() or candidate in selected:
            continue
        try:
            size = candidate.stat().st_size
        except OSError:
            continue
        if size > _ORACLE_RUNTIME_CONTEXT_FILE_MAX_BYTES:
            continue
        if selected_bytes + size > _ORACLE_RUNTIME_CONTEXT_TOTAL_MAX_BYTES:
            continue
        selected.append(candidate)
        selected_bytes += size
    return selected


def _maybe_start_oracle_browser(extra_args: list[str]) -> OracleBrowserBootstrap:
    if not _oracle_browser_bootstrap_enabled():
        return OracleBrowserBootstrap(args=[])
    if not _oracle_browser_engine_requested(extra_args):
        return OracleBrowserBootstrap(args=[])
    if _oracle_args_include_option(
        extra_args,
        "--remote-chrome",
        "--browser-attach-running",
        "--copy-profile",
        "--browser-manual-login",
        "--remote-host",
        "--browser-cookie-path",
    ):
        return OracleBrowserBootstrap(args=[])

    port = _oracle_browser_port()
    remote_arg = f"127.0.0.1:{port}"
    oracle_args = [
        "--remote-chrome",
        remote_arg,
        *_oracle_browser_model_strategy_args(extra_args),
        *_oracle_browser_thinking_time_args(extra_args),
        *_oracle_browser_attachments_args(extra_args),
        *_oracle_browser_timeout_args(extra_args),
        *_oracle_browser_archive_args(extra_args),
    ]
    if _oracle_remote_chrome_ready(port):
        return OracleBrowserBootstrap(args=oracle_args)

    chrome_command = _oracle_chrome_command()
    if not chrome_command:
        return OracleBrowserBootstrap(args=[])

    profile_dir, temp_profile_dir = _prepare_oracle_chrome_profile()
    display_env = _oracle_browser_display_env()
    headless = not display_env and _env_flag("KAGGLEBOT_ORACLE_BROWSER_HEADLESS", default=True)
    launch_args = [
        *chrome_command,
        "--remote-debugging-address=127.0.0.1",
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile_dir}",
        f"--profile-directory={os.environ.get('KAGGLEBOT_ORACLE_CHROME_PROFILE', 'Default').strip() or 'Default'}",
        "--disable-gpu",
        "--lang=en-US",
        "--no-first-run",
        "--no-default-browser-check",
    ]
    if headless:
        launch_args.append("--headless=new")
    launch_args.append(os.environ.get("KAGGLEBOT_ORACLE_CHATGPT_URL", "https://chatgpt.com/"))
    env = os.environ.copy()
    env.update(display_env)
    process = subprocess.Popen(launch_args, env=env)  # noqa: S603
    bootstrap = OracleBrowserBootstrap(
        args=oracle_args,
        process=process,
        temp_profile_dir=temp_profile_dir,
    )
    if not _wait_for_oracle_remote_chrome(port, timeout_sec=_ORACLE_BROWSER_READY_TIMEOUT_SEC):
        bootstrap.close()
        return OracleBrowserBootstrap(args=[])
    return bootstrap


def _oracle_browser_bootstrap_enabled() -> bool:
    return _env_flag("KAGGLEBOT_ORACLE_BROWSER_BOOTSTRAP", default=True)


def _oracle_browser_model_strategy_args(extra_args: list[str]) -> list[str]:
    if _oracle_args_include_option(extra_args, "--browser-model-strategy"):
        return []
    return ["--browser-model-strategy", "select"]


def _oracle_browser_thinking_time_args(extra_args: list[str]) -> list[str]:
    if _oracle_args_include_option(extra_args, "--browser-thinking-time"):
        return []
    return ["--browser-thinking-time", _DEFAULT_ORACLE_BROWSER_THINKING_TIME]


def _oracle_browser_archive_args(extra_args: list[str]) -> list[str]:
    if _oracle_args_include_option(extra_args, "--browser-archive"):
        return []
    return ["--browser-archive", "always"]


def _ensure_oracle_conversation_archived(
    *,
    transcript_path: Path,
    output_dir: Path,
    browser_bootstrap: OracleBrowserBootstrap,
    extra_args: list[str],
) -> dict[str, object]:
    if _oracle_archive_mode(extra_args) == "never":
        return {"archived": True, "mode": "never", "verification": "explicitly_disabled"}
    report_path = output_dir / "oracle_archive.json"
    status = _find_oracle_session_archive_status(transcript_path)
    if status.get("archived") is True:
        report_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return status
    conversation_url = str(status.get("conversationUrl") or "").strip()
    remote = _oracle_remote_chrome_endpoint(browser_bootstrap.args or extra_args)
    if not conversation_url or remote is None:
        report = {
            **status,
            "archived": False,
            "fallbackAttempted": False,
            "fallbackReason": "missing-conversation-url-or-remote-chrome",
        }
    else:
        host, port = remote
        report = dict(status)
        for attempt in range(1, 4):
            report = {
                **report,
                **_archive_oracle_conversation_via_cdp(
                    conversation_url=conversation_url,
                    host=host,
                    port=port,
                ),
                "fallbackAttempt": attempt,
            }
            if report.get("archived") is True:
                break
            time.sleep(float(attempt))
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report.get("archived") is True:
        print("oracle strategy: archived ChatGPT conversation", flush=True)
    else:
        print(
            "oracle strategy: warning: ChatGPT conversation archive could not be verified "
            f"({report.get('fallbackReason') or report.get('reason') or 'unknown'})",
            flush=True,
        )
    return report


def _oracle_archive_mode(extra_args: list[str]) -> str:
    for index, value in enumerate(extra_args):
        if value == "--browser-archive" and index + 1 < len(extra_args):
            return extra_args[index + 1].strip().lower()
        if value.startswith("--browser-archive="):
            return value.partition("=")[2].strip().lower()
    return "always"


def _find_oracle_session_archive_status(transcript_path: Path) -> dict[str, object]:
    sessions_root = Path(os.environ.get("ORACLE_HOME", str(Path.home() / ".oracle"))).expanduser() / "sessions"
    if not sessions_root.is_dir():
        return {}
    expected = str(transcript_path.resolve())
    meta_paths = sorted(
        sessions_root.glob("*/meta.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for meta_path in meta_paths[:100]:
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        options = payload.get("options") if isinstance(payload, dict) else None
        if not isinstance(options, dict) or str(options.get("writeOutputPath") or "") != expected:
            continue
        browser = payload.get("browser")
        archive = browser.get("archive") if isinstance(browser, dict) else None
        report = dict(archive) if isinstance(archive, dict) else {}
        report["oracleSession"] = str(meta_path.parent)
        return report
    return {}


def _oracle_remote_chrome_endpoint(args: list[str]) -> tuple[str, int] | None:
    for index, value in enumerate(args):
        if value == "--remote-chrome" and index + 1 < len(args):
            raw = args[index + 1]
        elif value.startswith("--remote-chrome="):
            raw = value.partition("=")[2]
        else:
            continue
        host, separator, port_text = raw.rpartition(":")
        if not separator:
            return None
        try:
            return host or "127.0.0.1", int(port_text)
        except ValueError:
            return None
    return None


def _start_oracle_browser_attachment_compatibility(
    *,
    args: list[str],
    attachment_paths: list[Path],
) -> OracleBrowserCompatibility:
    remote = _oracle_remote_chrome_endpoint(args)
    cdp_module = _oracle_cdp_module_path()
    node = _oracle_node_command()
    if remote is None or not attachment_paths or cdp_module is None or node is None:
        return OracleBrowserCompatibility()
    host, port = remote
    names = list(dict.fromkeys(path.name for path in attachment_paths))
    try:
        process = subprocess.Popen(  # noqa: S603
            [
                node,
                "-e",
                _ORACLE_ATTACHMENT_COMPATIBILITY_CDP_SCRIPT,
                str(cdp_module),
                host,
                str(port),
                json.dumps(names),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return OracleBrowserCompatibility()
    return OracleBrowserCompatibility(process=process)


def _archive_oracle_conversation_via_cdp(
    *,
    conversation_url: str,
    host: str,
    port: int,
) -> dict[str, object]:
    cdp_module = _oracle_cdp_module_path()
    node = _oracle_node_command()
    if cdp_module is None or node is None:
        return {
            "archived": False,
            "fallbackAttempted": False,
            "fallbackReason": "chrome-remote-interface-unavailable",
        }
    try:
        result = run_command(
            [
                node,
                "-e",
                _ORACLE_ARCHIVE_CDP_SCRIPT,
                str(cdp_module),
                host,
                str(port),
                conversation_url,
            ],
            timeout=30.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {
            "archived": False,
            "fallbackAttempted": True,
            "fallbackReason": type(exc).__name__,
        }
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        payload = {}
    if result.returncode != 0 or not isinstance(payload, dict):
        return {
            "archived": False,
            "fallbackAttempted": True,
            "fallbackReason": (result.stderr or result.stdout or "cdp-archive-failed")[-500:],
        }
    payload["fallbackAttempted"] = True
    return payload


def _oracle_node_command() -> str | None:
    configured = os.environ.get("KAGGLEBOT_ORACLE_NODE_COMMAND", "").strip()
    if configured:
        return configured
    return shutil.which("node")


def _oracle_cdp_module_path() -> Path | None:
    configured = os.environ.get("KAGGLEBOT_ORACLE_CDP_MODULE", "").strip()
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path.home() / ".local/oracle-node24/lib/node_modules/@steipete/oracle/node_modules/chrome-remote-interface",
        Path("/usr/local/lib/node_modules/@steipete/oracle/node_modules/chrome-remote-interface"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    npx_root = Path.home() / ".npm/_npx"
    if npx_root.is_dir():
        matches = sorted(
            npx_root.glob("*/node_modules/@steipete/oracle/node_modules/chrome-remote-interface"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if matches:
            return matches[0]
    return None


_ORACLE_ARCHIVE_CDP_SCRIPT = r"""
const CDP = require(process.argv[1]);
const host = process.argv[2];
const port = Number(process.argv[3]);
const conversationUrl = process.argv[4];
(async () => {
  let client;
  let target;
  try {
    const conversationId = new URL(conversationUrl).pathname.split('/').filter(Boolean).pop();
    target = await CDP.New({host, port, url: conversationUrl});
    client = await CDP({host, port, target});
    await client.Page.enable();
    await client.Page.navigate({url: conversationUrl});
    await client.Page.loadEventFired();
    const expression = `(async () => {
      const sessionResponse = await fetch('/api/auth/session', {credentials: 'include'});
      const session = await sessionResponse.json();
      const accessToken = session && session.accessToken;
      if (!accessToken) {
        return {archived: false, status: sessionResponse.status, fallbackReason: 'access-token-unavailable'};
      }
      const response = await fetch('/backend-api/conversation/${conversationId}', {
        method: 'PATCH',
        credentials: 'include',
        headers: {'authorization': 'Bearer ' + accessToken, 'content-type': 'application/json'},
        body: JSON.stringify({is_archived: true}),
      });
      const body = await response.text();
      if (!response.ok) {
        return {archived: false, status: response.status, response: body.slice(0, 500)};
      }
      const verifyResponse = await fetch('/backend-api/conversation/${conversationId}', {
        credentials: 'include',
        headers: {'authorization': 'Bearer ' + accessToken},
      });
      const verified = await verifyResponse.json().catch(() => ({}));
      return {
        archived: verifyResponse.ok && verified && verified.is_archived === true,
        status: response.status,
        verificationStatus: verifyResponse.status,
        response: body.slice(0, 500),
      };
    })()`;
    const evaluated = await client.Runtime.evaluate({expression, awaitPromise: true, returnByValue: true});
    const value = evaluated.result && evaluated.result.value;
    console.log(JSON.stringify(value || {archived: false, fallbackReason: 'empty-cdp-result'}));
  } catch (error) {
    console.log(JSON.stringify({archived: false, fallbackReason: String(error)}));
    process.exitCode = 1;
  } finally {
    if (client) await client.close().catch(() => {});
    if (target) await CDP.Close({host, port, id: target.id}).catch(() => {});
  }
})();
"""


_ORACLE_ATTACHMENT_COMPATIBILITY_CDP_SCRIPT = r"""
const CDP = require(process.argv[1]);
const host = process.argv[2];
const port = Number(process.argv[3]);
const expectedNames = JSON.parse(process.argv[4]);
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const patchTarget = async (target) => {
  let client;
  try {
    client = await CDP({host, port, target});
    const names = JSON.stringify(expectedNames);
    const expression = `(() => {
      const expectedNames = ${names};
      let changed = 0;
      for (const button of document.querySelectorAll('button[aria-label]')) {
        const label = button.getAttribute('aria-label') || '';
        if (!label || /^remove\\b/i.test(label)) continue;
        const name = expectedNames.find((candidate) => {
          const stem = candidate.replace(/\\.[a-z0-9]{1,10}$/i, '');
          return label.includes(candidate) || (stem.length >= 6 && label.includes(stem));
        });
        if (!name || label.trim() === name) continue;
        button.setAttribute('aria-label', 'Remove ' + name);
        changed += 1;
      }
      return changed;
    })()`;
    await client.Runtime.evaluate({expression, returnByValue: true});
  } finally {
    if (client) await client.close().catch(() => {});
  }
};

(async () => {
  while (true) {
    const targets = await CDP.List({host, port}).catch(() => []);
    const pages = targets.filter(
      (target) => target.type === 'page' && /^https:\/\/chatgpt\.com\//.test(target.url || ''),
    );
    for (const target of pages) {
      await patchTarget(target).catch(() => {});
    }
    await delay(250);
  }
})().catch(() => process.exit(0));
"""


def _oracle_browser_attachments_args(extra_args: list[str]) -> list[str]:
    if _oracle_args_include_option(extra_args, "--browser-attachments"):
        return []
    mode = os.environ.get("KAGGLEBOT_ORACLE_BROWSER_ATTACHMENTS", "auto").strip().lower()
    if mode not in {"auto", "never", "always"}:
        mode = "auto"
    return ["--browser-attachments", mode]


def _oracle_browser_timeout_args(extra_args: list[str]) -> list[str]:
    args: list[str] = []
    if not _oracle_args_include_option(extra_args, "--browser-input-timeout"):
        args += [
            "--browser-input-timeout",
            os.environ.get("KAGGLEBOT_ORACLE_BROWSER_INPUT_TIMEOUT", _DEFAULT_ORACLE_BROWSER_INPUT_TIMEOUT),
        ]
    if not _oracle_args_include_option(extra_args, "--browser-timeout"):
        args += [
            "--browser-timeout",
            os.environ.get("KAGGLEBOT_ORACLE_BROWSER_TIMEOUT", _DEFAULT_ORACLE_BROWSER_TIMEOUT),
        ]
    return args


def _oracle_browser_engine_requested(extra_args: list[str]) -> bool:
    explicit = _option_value(extra_args, "--engine", "-e")
    if explicit is not None:
        return explicit.strip().lower() == "browser"
    requested = os.environ.get("KAGGLEBOT_ORACLE_ENGINE", "browser").strip().lower()
    return requested not in {"api", "auto"}


def _oracle_browser_port() -> int:
    raw = os.environ.get("KAGGLEBOT_ORACLE_BROWSER_PORT", str(_DEFAULT_ORACLE_BROWSER_PORT)).strip()
    try:
        port = int(raw)
    except ValueError:
        return _DEFAULT_ORACLE_BROWSER_PORT
    if 1 <= port <= 65535:
        return port
    return _DEFAULT_ORACLE_BROWSER_PORT


def _oracle_chrome_command() -> list[str]:
    raw = os.environ.get("KAGGLEBOT_ORACLE_CHROME_COMMAND", "").strip()
    if raw:
        return shlex.split(raw)
    for candidate in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        path = shutil.which(candidate)
        if path:
            return [path]
    return []


def _prepare_oracle_chrome_profile() -> tuple[Path, Path | None]:
    explicit = os.environ.get("KAGGLEBOT_ORACLE_CHROME_USER_DATA_DIR", "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path, None

    temp_profile_dir = Path(mkdtemp(prefix="kagglebot-oracle-chrome-"))
    temp_profile_dir.chmod(0o700)
    source = Path(os.environ.get("KAGGLEBOT_ORACLE_CHROME_COPY_PROFILE", "~/.config/google-chrome")).expanduser()
    if source.exists():
        rsync = shutil.which("rsync")
        if rsync:
            root_excludes = [f"--exclude=/{name}" for name in _ORACLE_CHROME_PROFILE_ROOT_EXCLUDES]
            subprocess.run(  # noqa: S603
                [
                    rsync,
                    "-a",
                    *root_excludes,
                    "--exclude=*/Cache/*",
                    "--exclude=*/Code Cache/*",
                    "--exclude=*/GPUCache/*",
                    "--exclude=*/Service Worker/CacheStorage/*",
                    "--exclude=*/GrShaderCache/*",
                    "--exclude=*/ShaderCache/*",
                    f"{source}/",
                    f"{temp_profile_dir}/",
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            shutil.copytree(
                source,
                temp_profile_dir,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    "Cache",
                    "Code Cache",
                    "GPUCache",
                    "CacheStorage",
                    "GrShaderCache",
                    "ShaderCache",
                    *_ORACLE_CHROME_PROFILE_ROOT_EXCLUDES,
                ),
            )
    return temp_profile_dir, temp_profile_dir


def _oracle_browser_display_env() -> dict[str, str]:
    explicit_display = os.environ.get("KAGGLEBOT_ORACLE_DISPLAY", "").strip()
    display = explicit_display or os.environ.get("DISPLAY", "").strip()
    candidates = [display] if display else _discover_x_displays()
    for candidate in candidates:
        env = {"DISPLAY": candidate}
        xauthority = _oracle_xauthority()
        if xauthority:
            env["XAUTHORITY"] = str(xauthority)
        if _display_usable(env):
            return env
    return {}


def _discover_x_displays() -> list[str]:
    x11_dir = Path("/tmp/.X11-unix")
    if not x11_dir.exists():
        return []
    displays: list[str] = []
    for path in sorted(x11_dir.glob("X*")):
        suffix = path.name.removeprefix("X")
        if suffix.isdigit():
            displays.append(f":{suffix}")
    return displays


def _oracle_xauthority() -> Path | None:
    for raw in (
        os.environ.get("KAGGLEBOT_ORACLE_XAUTHORITY"),
        os.environ.get("XAUTHORITY"),
        f"/run/user/{os.getuid()}/gdm/Xauthority",
        "~/.Xauthority",
    ):
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.exists():
            return path
    return None


def _display_usable(env: dict[str, str]) -> bool:
    xset = shutil.which("xset")
    if not xset:
        return True
    check_env = os.environ.copy()
    check_env.update(env)
    result = subprocess.run(  # noqa: S603
        [xset, "q"],
        env=check_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=3,
        check=False,
    )
    return result.returncode == 0


def _wait_for_oracle_remote_chrome(port: int, *, timeout_sec: float) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if _oracle_remote_chrome_ready(port):
            return True
        time.sleep(0.25)
    return False


def _oracle_remote_chrome_ready(port: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/json/version", timeout=1.0) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _oracle_args_include_option(args: list[str], *options: str) -> bool:
    option_set = set(options)
    for arg in args:
        if arg in option_set:
            return True
        if any(arg.startswith(f"{option}=") for option in option_set if option.startswith("--")):
            return True
    return False


def _option_value(args: list[str], *options: str) -> str | None:
    option_set = set(options)
    for index, arg in enumerate(args):
        if arg in option_set:
            if index + 1 < len(args):
                return args[index + 1]
            return ""
        for option in option_set:
            if option.startswith("--") and arg.startswith(f"{option}="):
                return arg.split("=", 1)[1]
    return None


def _oracle_available() -> bool:
    command = _oracle_command()
    return bool(command and shutil.which(command[0]))


def _normalize_reasoning_effort(effort: str) -> str:
    normalized = effort.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"extra_high", "xhgih"}:
        return "xhigh"
    return normalized


def _heartbeat(
    stop_event: threading.Event,
    start_time: float,
    interval: float = 30.0,
    *,
    label: str = _RUNNER_LABEL,
) -> None:
    while not stop_event.wait(interval):
        elapsed = int(time.monotonic() - start_time)
        print(f"{label} running... ({elapsed}s total)", flush=True)


@lru_cache(maxsize=1)
def _codex_help() -> str:
    try:
        result = run_command([STRATEGY_AGENT.cli_command, "exec", "--help"])
    except FileNotFoundError:
        return ""
    if result.returncode != 0:
        return ""
    return result.output


def _supported_flags() -> set[str]:
    text = _codex_help()
    flags: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        for token in line.split():
            if not token.startswith("-"):
                break
            flags.add(token.rstrip(","))
    return flags
