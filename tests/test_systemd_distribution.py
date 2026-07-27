from __future__ import annotations

import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEMD_DIR = ROOT / "deploy" / "systemd"


def test_systemd_units_launch_repository_commands_without_model_overrides() -> None:
    watch = (SYSTEMD_DIR / "kagglebot-watch.service").read_text(encoding="utf-8")
    notifier = (SYSTEMD_DIR / "kagglebot-discord-notifier.service").read_text(encoding="utf-8")
    oracle_update = (SYSTEMD_DIR / "kagglebot-oracle-update.service").read_text(encoding="utf-8")
    oracle_update_timer = (SYSTEMD_DIR / "kagglebot-oracle-update.timer").read_text(encoding="utf-8")

    assert "uv run kagglebot --force watch" in watch
    assert "uv run kagglebot discord-notifier" in notifier
    assert "python -m kagglebot.oracle_update" in oracle_update
    assert "OnUnitActiveSec=15min" in oracle_update_timer
    assert "kagglebot-oracle-update.service" in watch
    assert "WorkingDirectory=%h/.local/share/kagglebot-autopilot/current" in watch
    assert "KAGGLEBOT_PRIMARY_MODEL" not in watch
    assert "KAGGLEBOT_ORACLE_MODEL" not in watch
    assert "/home/morita" not in watch + notifier + oracle_update + oracle_update_timer


def test_systemd_installer_registers_versioned_units_from_the_clone() -> None:
    installer = ROOT / "scripts" / "kagglebot-systemd"
    content = installer.read_text(encoding="utf-8")

    assert installer.stat().st_mode & stat.S_IXUSR
    assert "uv sync --frozen" in content
    assert 'ln -sfn "${repo_root}" "${current_link}"' in content
    assert 'systemctl --user enable "${service_names[@]}" "${timer_names[@]}"' in content
    assert "kagglebot-oracle-update.service" in content
    assert "kagglebot-oracle-update.timer" in content
    assert 'systemctl --user restart "${service_names[@]}"' in content
