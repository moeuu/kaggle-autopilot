"""Tests for Kaggle CLI kernel wrappers."""

from __future__ import annotations

from types import SimpleNamespace

from kagglebot import kaggle_cli


def test_kernels_push_status_output(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run(args, check, capture_output, text):  # noqa: ARG001
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(kaggle_cli.subprocess, "run", fake_run)

    kaggle_cli.kernels_push(tmp_path, slug="demo")
    kaggle_cli.kernels_status("user/kernel", slug="demo")
    kaggle_cli.kernels_output("user/kernel", tmp_path / "out", slug="demo")
    kaggle_cli.competitions_files("demo")

    assert calls[0][:3] == ["kaggle", "kernels", "push"]
    assert calls[1][:3] == ["kaggle", "kernels", "status"]
    assert calls[2][:3] == ["kaggle", "kernels", "output"]
    assert calls[3][:3] == ["kaggle", "competitions", "files"]
