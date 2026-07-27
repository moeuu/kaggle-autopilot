from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_ORACLE_PACKAGE = "@steipete/oracle"
_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")


class OracleUpdateError(RuntimeError):
    """Raised when the installed Oracle CLI cannot be checked or updated."""


@dataclass(frozen=True)
class OracleUpdateConfig:
    prefix: Path
    node_prefix: Path
    proc_root: Path
    enabled: bool

    @property
    def npm_bin(self) -> Path:
        return self.node_prefix / "bin" / "npm"

    @property
    def package_json(self) -> Path:
        return self.prefix / "lib" / "node_modules" / "@steipete" / "oracle" / "package.json"

    @property
    def lock_path(self) -> Path:
        return self.prefix / ".oracle-update.lock"

    @property
    def oracle_bins(self) -> tuple[Path, ...]:
        return (self.prefix / "bin" / "oracle", self.prefix / "bin" / "oracle-mcp")


@dataclass(frozen=True)
class OracleUpdateResult:
    status: str
    installed_version: str | None
    latest_version: str | None


def resolve_update_config(environ: Mapping[str, str] | None = None) -> OracleUpdateConfig:
    env = os.environ if environ is None else environ
    home = Path(env.get("HOME", str(Path.home()))).expanduser()
    return OracleUpdateConfig(
        prefix=Path(env.get("KAGGLEBOT_ORACLE_UPDATE_PREFIX", str(home / ".local" / "oracle-node24"))).expanduser(),
        node_prefix=Path(
            env.get("KAGGLEBOT_ORACLE_NODE_PREFIX", str(home / ".local" / "opt" / "node-v24"))
        ).expanduser(),
        proc_root=Path(env.get("KAGGLEBOT_ORACLE_PROC_ROOT", "/proc")),
        enabled=_env_flag(env.get("KAGGLEBOT_ORACLE_AUTO_UPDATE"), default=True),
    )


def update_oracle(
    *,
    config: OracleUpdateConfig | None = None,
    environ: Mapping[str, str] | None = None,
) -> OracleUpdateResult:
    resolved = config or resolve_update_config(environ)
    if not resolved.enabled:
        return OracleUpdateResult(
            status="disabled", installed_version=_installed_version(resolved), latest_version=None
        )
    if not resolved.npm_bin.is_file():
        raise OracleUpdateError(f"Oracle updater npm executable not found: {resolved.npm_bin}")

    resolved.prefix.mkdir(parents=True, exist_ok=True)
    with resolved.lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return OracleUpdateResult(
                status="deferred_locked",
                installed_version=_installed_version(resolved),
                latest_version=None,
            )

        installed = _installed_version(resolved)
        latest = _latest_version(resolved, environ=environ)
        if installed == latest:
            return OracleUpdateResult(status="current", installed_version=installed, latest_version=latest)
        if _oracle_process_running(resolved):
            return OracleUpdateResult(status="deferred_active", installed_version=installed, latest_version=latest)

        _install_version(resolved, latest, environ=environ)
        verified = _installed_version(resolved)
        if verified != latest:
            raise OracleUpdateError(
                f"Oracle update verification failed: expected {latest}, found {verified or 'missing'}"
            )
        return OracleUpdateResult(status="updated", installed_version=verified, latest_version=latest)


def _installed_version(config: OracleUpdateConfig) -> str | None:
    try:
        payload = json.loads(config.package_json.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    version = payload.get("version") if isinstance(payload, dict) else None
    return version.strip() if isinstance(version, str) and version.strip() else None


def _latest_version(
    config: OracleUpdateConfig,
    *,
    environ: Mapping[str, str] | None,
) -> str:
    result = _run_npm(
        config,
        ["view", _ORACLE_PACKAGE, "dist-tags.latest", "--json"],
        environ=environ,
        timeout=60,
    )
    try:
        latest = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise OracleUpdateError(f"npm returned invalid Oracle version JSON: {result.stdout.strip()}") from exc
    if not isinstance(latest, str) or not _VERSION_PATTERN.fullmatch(latest.strip()):
        raise OracleUpdateError(f"npm returned an invalid Oracle latest version: {latest!r}")
    return latest.strip()


def _install_version(
    config: OracleUpdateConfig,
    version: str,
    *,
    environ: Mapping[str, str] | None,
) -> None:
    _run_npm(
        config,
        [
            "install",
            "--global",
            "--prefix",
            str(config.prefix),
            f"{_ORACLE_PACKAGE}@{version}",
            "--no-audit",
            "--no-fund",
        ],
        environ=environ,
        timeout=15 * 60,
    )


def _run_npm(
    config: OracleUpdateConfig,
    args: list[str],
    *,
    environ: Mapping[str, str] | None,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ if environ is None else environ)
    current_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(
        part for part in (str(config.node_prefix / "bin"), current_path, "/usr/bin", "/bin") if part
    )
    command = [str(config.npm_bin), *args]
    try:
        result = subprocess.run(  # noqa: S603
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OracleUpdateError(f"Oracle npm command failed to start: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise OracleUpdateError(f"Oracle npm command failed with exit {result.returncode}: {detail}")
    return result


def _oracle_process_running(config: OracleUpdateConfig) -> bool:
    expected = {str(path) for path in config.oracle_bins}
    try:
        process_dirs = list(config.proc_root.iterdir())
    except OSError:
        return False
    for process_dir in process_dirs:
        if not process_dir.name.isdigit():
            continue
        try:
            args = (process_dir / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        decoded = {arg.decode("utf-8", errors="ignore") for arg in args if arg}
        if expected & decoded:
            return True
    return False


def _env_flag(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def main() -> int:
    try:
        result = update_oracle()
    except OracleUpdateError as exc:
        print(f"Oracle auto-update failed: {exc}", file=sys.stderr)
        return 1

    installed = result.installed_version or "missing"
    latest = result.latest_version or "unknown"
    messages = {
        "disabled": "Oracle auto-update is disabled.",
        "current": f"Oracle is current: {installed}.",
        "deferred_active": f"Oracle update deferred while a session is active: installed={installed}, latest={latest}.",
        "deferred_locked": f"Oracle update deferred because another updater is running: installed={installed}.",
        "updated": f"Oracle updated successfully: {installed}.",
    }
    print(messages[result.status])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
