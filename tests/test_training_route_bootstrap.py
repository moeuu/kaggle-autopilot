from __future__ import annotations

from pathlib import Path

from kagglebot.kernel_bootstrap import ensure_kernel_force_train_env, ensure_kernel_non_training_env


def test_non_training_bootstrap_replaces_force_training(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text("print('run')\n", encoding="utf-8")
    ensure_kernel_force_train_env(kernel_dir)

    ensure_kernel_non_training_env(kernel_dir)

    source = kernel_path.read_text(encoding="utf-8")
    assert "# kagglebot:force_train" not in source
    assert "# kagglebot:non_training_submission" in source
    assert "KAGGLEBOT_EXECUTION_MODE'] = 'non_training_submission'" in source
    assert "KAGGLEBOT_DO_TRAIN'] = '0'" in source
    assert "KAGGLEBOT_NON_TRAINING_VALIDATION_REQUIRED'] = '1'" in source


def test_force_training_bootstrap_replaces_non_training_mode(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text("print('run')\n", encoding="utf-8")
    ensure_kernel_non_training_env(kernel_dir)

    ensure_kernel_force_train_env(kernel_dir)

    source = kernel_path.read_text(encoding="utf-8")
    assert "# kagglebot:non_training_submission" not in source
    assert "# kagglebot:force_train" in source
    assert "KAGGLEBOT_DO_TRAIN'] = '1'" in source
