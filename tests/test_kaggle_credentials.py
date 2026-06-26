from __future__ import annotations

import json
from pathlib import Path

import pytest

from kagglebot.kaggle_credentials import kaggle_json_candidates, resolve_kaggle_username


def test_kaggle_json_candidates_accepts_config_dir_and_file_path(tmp_path: Path) -> None:
    config_file = tmp_path / "custom-kaggle.json"
    assert kaggle_json_candidates(config_dir_env=str(config_file), home=tmp_path / "home")[0] == config_file

    config_dir = tmp_path / "cfg"
    candidates = kaggle_json_candidates(config_dir_env=str(config_dir), home=tmp_path / "home")

    assert candidates[:2] == [config_dir / "kaggle.json", config_dir / "kaggle" / "kaggle.json"]
    assert candidates[-2:] == [
        tmp_path / "home" / ".kaggle" / "kaggle.json",
        tmp_path / "home" / ".config" / "kaggle" / "kaggle.json",
    ]


def test_resolve_kaggle_username_prefers_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_CONFIG_DIR", raising=False)

    assert resolve_kaggle_username("explicit-user") == "explicit-user"


def test_resolve_kaggle_username_prefers_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "env-user")

    assert resolve_kaggle_username(None) == "env-user"


def test_resolve_kaggle_username_reads_config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    (tmp_path / "kaggle.json").write_text(json.dumps({"username": "cfg-user"}), encoding="utf-8")

    assert resolve_kaggle_username(None) == "cfg-user"


def test_resolve_kaggle_username_reads_config_file_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    config_path = tmp_path / "custom-kaggle.json"
    config_path.write_text(json.dumps({"username": "file-user"}), encoding="utf-8")
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(config_path))

    assert resolve_kaggle_username(None) == "file-user"


def test_resolve_kaggle_username_skips_invalid_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "cfg").mkdir()
    (tmp_path / "cfg" / "kaggle.json").write_text("{invalid", encoding="utf-8")
    (tmp_path / "cfg" / "kaggle").mkdir()
    (tmp_path / "cfg" / "kaggle" / "kaggle.json").write_text("[]", encoding="utf-8")
    (tmp_path / "home" / ".config" / "kaggle").mkdir(parents=True)
    (tmp_path / "home" / ".config" / "kaggle" / "kaggle.json").write_text(
        json.dumps({"username": "home-config-user"}),
        encoding="utf-8",
    )

    assert resolve_kaggle_username(None) == "home-config-user"


def test_resolve_kaggle_username_errors_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "missing"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    with pytest.raises(ValueError, match="Kaggle username is required"):
        resolve_kaggle_username(None)
