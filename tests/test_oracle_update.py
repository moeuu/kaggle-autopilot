from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from kagglebot import oracle_update


def _config(tmp_path: Path, *, enabled: bool = True) -> oracle_update.OracleUpdateConfig:
    prefix = tmp_path / "oracle"
    node_prefix = tmp_path / "node"
    (node_prefix / "bin").mkdir(parents=True)
    (node_prefix / "bin" / "npm").write_text("#!/bin/sh\n", encoding="utf-8")
    return oracle_update.OracleUpdateConfig(
        prefix=prefix,
        node_prefix=node_prefix,
        proc_root=tmp_path / "proc",
        enabled=enabled,
    )


def _write_installed(config: oracle_update.OracleUpdateConfig, version: str) -> None:
    config.package_json.parent.mkdir(parents=True, exist_ok=True)
    config.package_json.write_text(json.dumps({"version": version}), encoding="utf-8")


def test_update_oracle_installs_and_verifies_latest(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_installed(config, "0.16.0")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        if "view" in args:
            return subprocess.CompletedProcess(args, 0, stdout='"0.16.1"\n', stderr="")
        _write_installed(config, "0.16.1")
        return subprocess.CompletedProcess(args, 0, stdout="updated\n", stderr="")

    monkeypatch.setattr(oracle_update.subprocess, "run", fake_run)

    result = oracle_update.update_oracle(config=config, environ={"PATH": "/usr/bin"})

    assert result.status == "updated"
    assert result.installed_version == "0.16.1"
    assert calls[0][0][1:] == ["view", "@steipete/oracle", "dist-tags.latest", "--json"]
    assert calls[1][0][1:6] == [
        "install",
        "--global",
        "--prefix",
        str(config.prefix),
        "@steipete/oracle@0.16.1",
    ]
    assert all("shell" not in kwargs for _, kwargs in calls)


def test_update_oracle_skips_install_when_current(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_installed(config, "0.16.1")
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout='"0.16.1"\n', stderr="")

    monkeypatch.setattr(oracle_update.subprocess, "run", fake_run)

    result = oracle_update.update_oracle(config=config, environ={})

    assert result.status == "current"
    assert len(calls) == 1


def test_update_oracle_defers_while_oracle_process_is_active(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_installed(config, "0.16.0")
    process_dir = config.proc_root / "123"
    process_dir.mkdir(parents=True)
    process_dir.joinpath("cmdline").write_bytes(
        b"node\0" + str(config.oracle_bins[0]).encode("utf-8") + b"\0--engine\0browser\0"
    )

    monkeypatch.setattr(
        oracle_update.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, stdout='"0.16.1"\n', stderr=""),
    )

    result = oracle_update.update_oracle(config=config, environ={})

    assert result.status == "deferred_active"
    assert result.installed_version == "0.16.0"
    assert result.latest_version == "0.16.1"


def test_update_oracle_rejects_invalid_registry_version(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_installed(config, "0.16.0")
    monkeypatch.setattr(
        oracle_update.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, stdout='"latest; bad"\n', stderr=""),
    )

    with pytest.raises(oracle_update.OracleUpdateError, match="invalid Oracle latest version"):
        oracle_update.update_oracle(config=config, environ={})


def test_update_oracle_can_be_disabled_without_npm_call(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path, enabled=False)
    _write_installed(config, "0.16.0")

    def fail_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("npm must not run when auto-update is disabled")

    monkeypatch.setattr(oracle_update.subprocess, "run", fail_run)

    result = oracle_update.update_oracle(config=config, environ={})

    assert result.status == "disabled"
