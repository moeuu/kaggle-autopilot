"""Tests for kaggle_api helpers."""

from __future__ import annotations

from types import SimpleNamespace

from kagglebot import kaggle_api


def test_check_rules_accepted(monkeypatch) -> None:
    def fake_run(args, **kwargs):  # noqa: ARG001
        stdout = "ref,title\nfoo,Example\nbar,Other\n"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="", output=stdout)

    monkeypatch.setattr(kaggle_api, "run_command", fake_run)
    assert kaggle_api.check_rules_accepted("foo") is True
    assert kaggle_api.check_rules_accepted("missing") is False
