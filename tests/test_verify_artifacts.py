from __future__ import annotations

import os
from pathlib import Path

from kagglebot.exec_utils import CommandResult
from kagglebot.verify_artifacts import (
    ensure_pytest_plugin_loaded,
    is_pytest_invocation,
    is_pytest_xdist_unrecognized_args,
    mirror_verify_artifacts,
    run_repo_verify,
    run_verify,
    serial_pytest_retry_args,
    verify_compat_shim,
)


def test_verify_compat_shim_resolves_known_competition_files() -> None:
    deep_past = verify_compat_shim(slug="deep-past-initiative-machine-translation", filename="kernel.py")
    deep_past_normalized = verify_compat_shim(
        slug=" Deep-Past-Initiative-Machine-Translation ",
        filename="/tmp/kernel.py",
    )
    s6e3 = verify_compat_shim(slug="playground-series-s6e3", filename="runtime.py")

    assert "KAGGLEBOT_VERIFY_COMPAT_SHIM" in deep_past
    assert "_prepare_reference_baseline_cfg" in deep_past
    assert deep_past_normalized == deep_past
    assert "KAGGLEBOT_VERIFY_COMPAT_SHIM" in s6e3
    assert "build_suite_specs" in s6e3
    assert verify_compat_shim(slug="demo", filename="kernel.py") == ""


def test_is_pytest_invocation_detects_direct_and_module_forms() -> None:
    assert is_pytest_invocation(["pytest", "-q"])
    assert is_pytest_invocation(["uv", "run", "pytest", "-q"])
    assert is_pytest_invocation(["python", "-m", "pytest", "-q"])
    assert not is_pytest_invocation(["uv", "run", "ruff", "check", "."])


def test_ensure_pytest_plugin_loaded_handles_common_invocations() -> None:
    assert ensure_pytest_plugin_loaded(["pytest", "-q"], "xdist.plugin") == [
        "pytest",
        "-p",
        "xdist.plugin",
        "-q",
    ]
    assert ensure_pytest_plugin_loaded(["uv", "run", "pytest", "-q"], "xdist.plugin") == [
        "uv",
        "run",
        "pytest",
        "-p",
        "xdist.plugin",
        "-q",
    ]
    assert ensure_pytest_plugin_loaded(["python", "-m", "pytest", "-q"], "xdist.plugin") == [
        "python",
        "-m",
        "pytest",
        "-p",
        "xdist.plugin",
        "-q",
    ]
    assert ensure_pytest_plugin_loaded(["pytest", "-p", "xdist", "-q"], "xdist.plugin") == [
        "pytest",
        "-p",
        "xdist",
        "-q",
    ]


def test_serial_pytest_retry_args_strips_xdist_from_command_and_addopts(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "\n".join(
            [
                "[tool.pytest.ini_options]",
                "addopts = \"-p no:launch_testing -n auto -m 'not slow'\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    retry_args = serial_pytest_retry_args(
        ["uv", "run", "pytest", "-p", "xdist.plugin", "-n", "auto", "-q"],
        repo_root=tmp_path,
    )

    assert retry_args == [
        "uv",
        "run",
        "pytest",
        "-o",
        "addopts=-p no:launch_testing -m 'not slow'",
        "-q",
    ]


def test_is_pytest_xdist_unrecognized_args_detects_missing_plugin_shapes() -> None:
    assert is_pytest_xdist_unrecognized_args("pytest: error: unrecognized arguments: -n")
    assert is_pytest_xdist_unrecognized_args("ERROR: Error importing plugin 'xdist.plugin': No module named 'xdist'")


def test_run_verify_retries_pytest_without_xdist_when_n_is_unrecognized(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "\n".join(
            [
                "[tool.pytest.ini_options]",
                "addopts = \"-p no:launch_testing -n auto -m 'not slow'\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run_command(args, **kwargs):  # noqa: ANN003, ARG001
        calls.append(args)
        if len(calls) == 1:
            return CommandResult(
                args=args,
                returncode=4,
                stdout="",
                stderr="pytest: error: unrecognized arguments: -n",
                duration_sec=0.0,
            )
        return CommandResult(args=args, returncode=0, stdout="", stderr="", duration_sec=0.0)

    monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", raising=False)

    run_verify(
        "uv run pytest -q",
        dry_run=False,
        repo_root=tmp_path,
        run_command_fn=fake_run_command,
    )

    assert calls == [
        ["uv", "run", "pytest", "-p", "xdist.plugin", "-q"],
        [
            "uv",
            "run",
            "pytest",
            "-o",
            "addopts=-p no:launch_testing -m 'not slow'",
            "-q",
        ],
    ]


def test_run_verify_sets_pytest_env_and_mirrors_artifacts(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    artifacts_dir = tmp_path / "external"
    source_kernel = artifacts_dir / "demo" / "kernel"
    source_kernel.mkdir(parents=True)
    (source_kernel / "kernel.py").write_text("VALUE = 1\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run_command(args, **kwargs):  # noqa: ANN003
        captured["args"] = args
        captured["env"] = kwargs.get("env")
        return CommandResult(args=args, returncode=0, stdout="", stderr="", duration_sec=0.0)

    monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", raising=False)

    run_verify(
        "uv run pytest -q",
        dry_run=False,
        artifacts_dir=artifacts_dir,
        repo_root=repo_root,
        run_command_fn=fake_run_command,
    )

    assert captured["args"] == ["uv", "run", "pytest", "-p", "xdist.plugin", "-q"]
    env = captured.get("env")
    assert isinstance(env, dict)
    assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert (repo_root / "artifacts" / "demo" / "kernel" / "kernel.py").exists()


def test_run_repo_verify_uses_current_directory_as_repo_root(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    artifacts_dir = tmp_path / "external"
    source_kernel = artifacts_dir / "demo" / "kernel"
    source_kernel.mkdir(parents=True)
    (source_kernel / "kernel.py").write_text("VALUE = 1\n", encoding="utf-8")

    def fake_run_command(args, **kwargs):  # noqa: ANN003, ARG001
        assert (repo_root / "artifacts" / "demo" / "kernel" / "kernel.py").exists()
        return CommandResult(args=args, returncode=0, stdout="", stderr="", duration_sec=0.0)

    monkeypatch.chdir(repo_root)

    run_repo_verify(
        "pytest -q",
        dry_run=False,
        artifacts_dir=artifacts_dir,
        run_command_fn=fake_run_command,
    )


def test_mirror_verify_artifacts_copies_kernel_tree_and_excludes_outputs(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    artifacts_dir = tmp_path / "external"
    source_kernel = artifacts_dir / "demo" / "kernel"
    source_kernel.mkdir(parents=True)
    (source_kernel / "kernel.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source_kernel / "output").mkdir()
    (source_kernel / "output" / "submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")

    mirror_verify_artifacts(artifacts_dir, repo_root=repo_root)

    mirrored_kernel = repo_root / "artifacts" / "demo" / "kernel" / "kernel.py"
    assert mirrored_kernel.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert not (repo_root / "artifacts" / "demo" / "kernel" / "output").exists()


def test_mirror_verify_artifacts_prefers_latest_kernel_version_and_appends_shim(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    artifacts_dir = tmp_path / "external"
    old_kernel = artifacts_dir / "playground-series-s6e3" / "kernels" / "run-a" / "local-iter-1" / "runtime.py"
    new_kernel = artifacts_dir / "playground-series-s6e3" / "kernels" / "run-b" / "local-iter-2" / "runtime.py"
    old_kernel.parent.mkdir(parents=True)
    new_kernel.parent.mkdir(parents=True)
    old_kernel.write_text("VALUE = 'old'\n", encoding="utf-8")
    new_kernel.write_text("VALUE = 'new'\n", encoding="utf-8")
    (new_kernel.parent / "plan.json").write_text('{"run": "b"}\n', encoding="utf-8")
    (new_kernel.parent.parent / "plan.json").write_text('{"artifact": "b"}\n', encoding="utf-8")
    old_time = 1_000_000_000
    new_time = 2_000_000_000
    old_kernel.touch()
    new_kernel.touch()
    os.utime(old_kernel, (old_time, old_time))
    os.utime(new_kernel, (new_time, new_time))

    mirror_verify_artifacts(artifacts_dir, repo_root=repo_root)

    mirrored_runtime = repo_root / "artifacts" / "playground-series-s6e3" / "kernel" / "runtime.py"
    text = mirrored_runtime.read_text(encoding="utf-8")
    assert "VALUE = 'new'" in text
    assert "KAGGLEBOT_VERIFY_COMPAT_SHIM" in text
    assert (repo_root / "artifacts" / "playground-series-s6e3" / "kernel" / "plan.json").read_text(
        encoding="utf-8"
    ) == '{"run": "b"}\n'
    assert (repo_root / "artifacts" / "playground-series-s6e3" / "plan.json").read_text(
        encoding="utf-8"
    ) == '{"artifact": "b"}\n'
