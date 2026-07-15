from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from kagglebot.kernel_runtime.submit_runtime_fidelity import (
    EXPECTED_FILE_NAME,
    RUNTIME_FILE_NAME,
    package_source_fingerprint,
    record_runtime_fidelity,
)


def _write_expected(package_dir: Path, *, output_file: str = "submission.csv") -> None:
    payload = {
        "schema_version": 1,
        "run_id": "run-1",
        "iteration": 2,
        "kernel": {"id": "user/demo"},
        "output": {"filename": output_file},
        "accelerator": {"requested": "cpu", "executed": "cpu", "machine_shape": ""},
        "package_fingerprint": package_source_fingerprint(package_dir),
    }
    (package_dir / EXPECTED_FILE_NAME).write_text(json.dumps(payload), encoding="utf-8")


def test_package_fingerprint_covers_source_but_skips_weights_and_expected_contract(tmp_path: Path) -> None:
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    kernel_path = package_dir / "kernel.py"
    kernel_path.write_text("print('v1')\n", encoding="utf-8")
    weights_path = package_dir / "model.safetensors"
    weights_path.write_bytes(b"weights-v1")

    first = package_source_fingerprint(package_dir)
    weights_path.write_bytes(b"weights-v2")
    assert package_source_fingerprint(package_dir) == first

    (package_dir / EXPECTED_FILE_NAME).write_text('{"changed": true}', encoding="utf-8")
    assert package_source_fingerprint(package_dir) == first

    kernel_path.write_text("print('v2')\n", encoding="utf-8")
    assert package_source_fingerprint(package_dir) != first


def test_runtime_recorder_captures_bounded_relative_inputs_output_and_tabular_stats(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package_dir = tmp_path / "private-package-root"
    output_dir = tmp_path / "working"
    input_dir = tmp_path / "private-input-root"
    package_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "competition").mkdir(parents=True)
    (package_dir / "kernel.py").write_text("print('ok')\n", encoding="utf-8")
    _write_expected(package_dir)
    (input_dir / "competition" / "test.csv").write_text("id\n2\n1\n", encoding="utf-8")
    (input_dir / "reference-model" / "config.json").parent.mkdir()
    (input_dir / "reference-model" / "config.json").write_text("{}", encoding="utf-8")
    (output_dir / "submission.csv").write_text("id,target\n2,0.8\n1,0.2\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "chosen_pipeline": "model",
                "active_model_source": "owner/model/1",
                "submission_output_file": "submission.csv",
                "test_prediction_distribution": {"source_top10": [["model", 2]]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("KAGGLEBOT_FIDELITY_REQUESTED_ACCELERATOR", "cpu")
    monkeypatch.setenv("KAGGLEBOT_FIDELITY_EXECUTED_ACCELERATOR", "cpu")
    monkeypatch.setenv("PRIVATE_TOKEN", "must-not-be-recorded")

    report = record_runtime_fidelity(
        package_root=package_dir,
        output_root=output_dir,
        input_root=input_dir,
    )

    assert (
        report["package"]["source_sha256"]
        == json.loads((package_dir / EXPECTED_FILE_NAME).read_text(encoding="utf-8"))["package_fingerprint"]
    )
    assert report["input_inventory"]["test_like_inputs"] == ["competition/test.csv"]
    assert "reference-model" in report["input_inventory"]["model_or_reference_roots"]
    assert report["outputs"]["selected"]["filename"] == "submission.csv"
    assert report["tabular_prediction"]["row_count"] == 2
    assert report["tabular_prediction"]["identifier"]["unique"] is True
    assert report["tabular_prediction"]["identifier"]["order_digests"]["id"]
    assert report["tabular_prediction"]["numeric_dispersion"][0]["stddev"] > 0
    serialized = (output_dir / RUNTIME_FILE_NAME).read_text(encoding="utf-8")
    assert str(input_dir) not in serialized
    assert "must-not-be-recorded" not in serialized


def test_installed_recorder_chains_unhandled_exception_and_writes_evidence(tmp_path: Path) -> None:
    package_dir = tmp_path / "package"
    output_dir = tmp_path / "working"
    input_dir = tmp_path / "input"
    package_dir.mkdir()
    output_dir.mkdir()
    input_dir.mkdir()
    (package_dir / "kernel.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    _write_expected(package_dir, output_file="submission.json")
    script = tmp_path / "raise_after_install.py"
    script.write_text(
        "\n".join(
            [
                "from kagglebot.kernel_runtime.submit_runtime_fidelity import install",
                "install("
                f"package_root={str(package_dir)!r}, "
                f"output_root={str(output_dir)!r}, "
                f"input_root={str(input_dir)!r})",
                "raise RuntimeError('boom')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "RuntimeError: boom" in completed.stderr
    report = json.loads((output_dir / RUNTIME_FILE_NAME).read_text(encoding="utf-8"))
    assert report["errors"]["unhandled_exception"]["type"] == "RuntimeError"
    assert "RuntimeError: boom" in report["errors"]["unhandled_exception"]["traceback"]
    assert str(tmp_path) not in json.dumps(report)
