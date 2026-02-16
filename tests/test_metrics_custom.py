from __future__ import annotations

from pathlib import Path

import numpy as np

from kagglebot.solver.metrics import compute_metric


def test_custom_metric_loaded_from_python_file(tmp_path: Path) -> None:
    module_path = tmp_path / "custom_metric.py"
    module_path.write_text(
        "\n".join(
            [
                "def compute_metric(y_true, y_pred, metric_name=None):",
                "    return float(sum(y_pred) - sum(y_true))",
            ]
        ),
        encoding="utf-8",
    )
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 4.0])
    score = compute_metric(f"custom:{module_path}:compute_metric", y_true, y_pred)
    assert score == 2.0


def test_custom_metric_loaded_from_module_function(tmp_path: Path, monkeypatch) -> None:
    package_dir = tmp_path / "custompkg"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "metrics_impl.py").write_text(
        "\n".join(
            [
                "def score(y_true, y_pred):",
                "    return 0.123",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8])
    score = compute_metric("custom:custompkg.metrics_impl:score", y_true, y_pred)
    assert score == 0.123
