from __future__ import annotations

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

from kagglebot.agents.identity import STRATEGY_AGENT, render_prompt_identity
from kagglebot.agents.sandbox_fallback import (
    append_sandbox_args,
    detect_sandbox_startup_failure,
    resolve_agent_sandbox_mode,
)
from kagglebot.exec_utils import CommandResult, run_command

_DEFAULT_MODEL = STRATEGY_AGENT.model
_DEFAULT_REASONING_EFFORT = STRATEGY_AGENT.reasoning_effort
_DEFAULT_TIMEOUT_SEC = 600.0
_DEFAULT_ORACLE_TIMEOUT_SEC = 3900.0
_PYTEST_TIMEOUT_SEC = 2.0
_RUNNER_LABEL = STRATEGY_AGENT.log_alias
_DEFAULT_ORACLE_MODEL = "gpt-5.5-pro"
_DEFAULT_ORACLE_BROWSER_PORT = 9222
_ORACLE_BROWSER_READY_TIMEOUT_SEC = 15.0
_DEFAULT_ORACLE_BROWSER_INPUT_TIMEOUT = "600s"
_DEFAULT_ORACLE_BROWSER_TIMEOUT = "60m"


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

    model = os.environ.get("KAGGLEBOT_ORACLE_MODEL", _DEFAULT_ORACLE_MODEL).strip() or _DEFAULT_ORACLE_MODEL
    consult_prompt = (
        "Read the attached Kagglebot strategy prompt file and any attached Kagglebot context bundle files. "
        "Treat this as a single-turn consultation with no prior session memory; all required context is attached. "
        "Return exactly the requested delimiter sections, including PLAN_JSON and CODEX_INSTRUCTIONS. "
        "Do not omit source evidence when the prompt requires it."
    )
    extra_args = _oracle_extra_args()
    browser_bootstrap = _maybe_start_oracle_browser(extra_args)
    inline_prompt = _oracle_inline_prompt_enabled(extra_args)
    if inline_prompt:
        consult_prompt = (
            "Use the Kagglebot strategy prompt below as the complete context. "
            "Return only the delimiter sections requested inside it, especially STRATEGY, PLAN_JSON, "
            "and CODEX_INSTRUCTIONS.\n\n"
            f"{rendered_prompt}"
        )
    attachment_paths = _oracle_attachment_paths(prompt_path=prompt_path, oracle_prompt_path=oracle_prompt_path)
    args = [
        *command,
        *_oracle_engine_args(extra_args),
        *extra_args,
        *browser_bootstrap.args,
        *_oracle_wait_args(extra_args),
        *_oracle_force_args(extra_args),
        "--model",
        model,
        "--write-output",
        str(transcript_path),
        "-p",
        consult_prompt,
    ]
    if not inline_prompt:
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
    try:
        try:
            result = run_command(args, timeout=timeout)
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
        browser_bootstrap.close()

    total_elapsed = int(time.monotonic() - start_time)
    print(f"oracle strategy done... ({total_elapsed}s total, exit={result.returncode})", flush=True)
    transcript_text = transcript_path.read_text(encoding="utf-8", errors="ignore") if transcript_path.exists() else ""
    stdout_text = transcript_text.strip() or result.stdout.strip()
    if not transcript_text:
        transcript_path.write_text((stdout_text + "\n") if stdout_text else "", encoding="utf-8")
    last_message_path.write_text((stdout_text + "\n") if stdout_text else "", encoding="utf-8")
    return StrategyResult(
        transcript_path=transcript_path,
        last_message_path=last_message_path,
        returncode=result.returncode,
        stdout=stdout_text,
        stderr=result.stderr,
        sandbox_policy_mode="external",
        engine="oracle",
    )


def _resolve_strategy_engine(engine: str | None) -> str:
    return resolve_strategy_engine(engine)


def resolve_strategy_engine(engine: str | None = None) -> str:
    requested = (engine or os.environ.get("KAGGLEBOT_STRATEGY_ENGINE") or "auto").strip().lower()
    if requested == "auto":
        return "oracle" if _oracle_available() else "codex"
    if requested in {"oracle", "codex"}:
        return requested
    return "codex"


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


def _oracle_strategy_timeout() -> float:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return float(os.environ.get("KAGGLEBOT_PYTEST_STRATEGY_TIMEOUT_SEC", str(_PYTEST_TIMEOUT_SEC)))
    raw = os.environ.get("KAGGLEBOT_ORACLE_STRATEGY_TIMEOUT_SEC")
    if raw is None:
        raw = os.environ.get("KAGGLEBOT_STRATEGY_TIMEOUT_SEC", str(_DEFAULT_ORACLE_TIMEOUT_SEC))
    return float(raw)


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


def _oracle_attachment_paths(*, prompt_path: Path, oracle_prompt_path: Path) -> list[Path]:
    paths: list[Path] = [oracle_prompt_path]
    for candidate in _oracle_context_bundle_candidates(prompt_path):
        if candidate.exists() and candidate.is_file() and candidate not in paths:
            paths.append(candidate)
    return paths


def _oracle_context_bundle_candidates(prompt_path: Path) -> list[Path]:
    return [
        prompt_path.parent / "strategy_context_bundle.md",
        prompt_path.parent.parent / "strategy_context_bundle.md",
    ]


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
        *_oracle_browser_attachments_args(extra_args),
        *_oracle_browser_timeout_args(extra_args),
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
    strategy = os.environ.get("KAGGLEBOT_ORACLE_BROWSER_MODEL_STRATEGY", "ignore").strip().lower()
    if strategy not in {"select", "current", "ignore"}:
        strategy = "ignore"
    return ["--browser-model-strategy", strategy]


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
            subprocess.run(  # noqa: S603
                [
                    rsync,
                    "-a",
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
