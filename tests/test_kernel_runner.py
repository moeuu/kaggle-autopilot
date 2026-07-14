"""Tests for kernel runner helpers."""

from __future__ import annotations

import base64
import gzip
import io
import json
import lzma
import os
import re
import runpy
import sqlite3
import time
import zipfile
from pathlib import Path

import duckdb
import pandas as pd
import pyreadr
import pytest
import zstandard as zstd

from kagglebot.exceptions import KernelCapacityError, KernelFailedError, KernelStillRunningError, KernelTimeoutError
from kagglebot.kernel_package_files import (
    copy_competition_external_assets,
    copy_kernel_sources,
    copy_shared_kernel_runtime_modules,
    sync_plan_snapshot,
)
from kagglebot.kernel_runner import (
    resolve_kaggle_username,
    run_kernel,
    run_kernel_local,
    run_submit_kernel,
)
from kagglebot.local_kernel_process import (
    LocalKernelLogFilterState,
    run_local_kernel_once,
    should_suppress_local_kernel_log_line,
)
from kagglebot.local_kernel_progress import build_local_kernel_progress_tracker
from kagglebot.local_kernel_shims import (
    ensure_training_progress_shim,
    inject_column_fill_shim,
    inject_column_map_shim,
    inject_context_io_shims,
    inject_device_coerce_shim,
    inject_local_runtime_shims,
    inject_object_coerce_shim,
    inject_pandas_tabular_read_shim,
    inject_training_compat_shims,
    inject_training_progress_shim,
    inject_transformers_eval_strategy_shim,
    inject_zero_overlap_drift_shim,
)
from kagglebot.submission_sample_discovery import TABULAR_INPUT_SUFFIXES

pytestmark = pytest.mark.slow


def test_resolve_kaggle_username_prefers_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_CONFIG_DIR", raising=False)
    assert resolve_kaggle_username("explicit-user") == "explicit-user"


def test_resolve_kaggle_username_reads_kaggle_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    (tmp_path / "kaggle.json").write_text(json.dumps({"username": "cfg-user", "key": "x"}), encoding="utf-8")
    assert resolve_kaggle_username(None) == "cfg-user"


def test_resolve_kaggle_username_reads_kaggle_config_file_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    kaggle_json = tmp_path / "custom-kaggle.json"
    kaggle_json.write_text(json.dumps({"username": "cfg-file-user", "key": "x"}), encoding="utf-8")
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(kaggle_json))
    assert resolve_kaggle_username(None) == "cfg-file-user"


def test_resolve_kaggle_username_skips_invalid_or_non_object_json_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    (tmp_path / "kaggle.json").write_text("{invalid", encoding="utf-8")
    (tmp_path / "kaggle" / "kaggle.json").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "kaggle" / "kaggle.json").write_text("[]", encoding="utf-8")
    home = tmp_path / "home"
    (home / ".kaggle").mkdir(parents=True, exist_ok=True)
    (home / ".kaggle" / "kaggle.json").write_text(json.dumps({"username": "home-user", "key": "x"}), encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    assert resolve_kaggle_username(None) == "home-user"


def test_resolve_kaggle_username_errors_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(ValueError, match="Kaggle username"):
        resolve_kaggle_username(None)


def test_run_kernel_dry_run(tmp_path: Path) -> None:
    # Ensure dry-run avoids Kaggle CLI calls.
    from kagglebot import kernel_runner

    kernel_runner.kernels_init = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("should not run"))
    kernel_path = tmp_path / "demo" / "kernel" / "kernel.py"
    kernel_path.parent.mkdir(parents=True, exist_ok=True)
    kernel_path.write_text(
        "\n".join(
            [
                "DATA = '/kaggle/input/demo/train.csv'",
                "OUT1 = '/kaggle/working/submission.csv'",
                "OUT2 = '/kaggle/working/metrics.json'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    plan_payload = {
        "toggles": {"USE_MODEL": True},
        "kaggle_kernel_sources": {
            "dataset_sources": ["alice/demo-dataset"],
            "kernel_sources": ["bob/demo-kernel"],
            "model_sources": ["carol/demo-model/PyTorch/default/1"],
        },
    }
    plan_path = tmp_path / "demo" / "plan.json"
    plan_path.write_text(json.dumps(plan_payload, indent=2), encoding="utf-8")
    stale_local_output = tmp_path / "demo" / "kernels" / "run-1" / "local-iter-1" / "outputs" / "package.py"
    stale_local_output.parent.mkdir(parents=True)
    stale_local_output.write_text("password = 'third-party-example'\n", encoding="utf-8")
    run_kernel(
        slug="demo",
        run_id="run-1",
        iteration=1,
        base_dir=tmp_path,
        kaggle_username="user",
        kernel_name=None,
        accelerator="gpu",
        enable_internet=False,
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        seed=42,
        dry_run=True,
        timeout_minutes=None,
    )
    meta_path = tmp_path / "demo" / "kernels" / "run-1" / "kernel-metadata.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["enable_gpu"] is True
    assert payload["enable_tpu"] is False
    assert payload["kernel_type"] == "script"
    assert payload["code_file"] == "kernel.py"
    assert payload["dataset_sources"] == ["alice/demo-dataset"]
    assert payload["kernel_sources"] == ["bob/demo-kernel"]
    assert payload["model_sources"] == ["carol/demo-model/PyTorch/default/1"]
    assert not stale_local_output.exists()
    kernel_text = (tmp_path / "demo" / "kernels" / "run-1" / "kernel.py").read_text(encoding="utf-8")
    assert "# kagglebot:competition_slug" in kernel_text
    assert "_kb_os.environ['KAGGLEBOT_COMPETITION_SLUG'] = \"demo\"" in kernel_text
    assert "_kb_os.environ['KAGGLEBOT_SLUG'] = \"demo\"" in kernel_text
    assert "# kagglebot:force_train" in kernel_text
    assert "_kb_os.environ['KAGGLEBOT_DO_TRAIN'] = '1'" in kernel_text
    assert "_kb_os.environ['KAGGLEBOT_FORCE_TRAIN'] = '1'" in kernel_text
    assert "demo" in kernel_text
    wrapper_path = tmp_path / "demo" / "kernels" / "run-1" / "kernel.ipynb"
    assert not wrapper_path.exists()
    staged_plan = tmp_path / "demo" / "kernels" / "run-1" / "plan.json"
    assert staged_plan.exists()
    assert json.loads(staged_plan.read_text(encoding="utf-8")) == plan_payload


def test_run_submit_kernel_dry_run_embeds_submission(tmp_path: Path) -> None:
    # Ensure dry-run avoids Kaggle CLI calls and stages a submit-only kernel.
    from kagglebot import kernel_runner

    kernel_runner.kernels_init = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("should not run"))
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    run_submit_kernel(
        slug="demo",
        run_id="run-1",
        iteration=1,
        base_dir=tmp_path,
        kaggle_username="user",
        kernel_name=None,
        accelerator="gpu",
        enable_internet=False,
        submission_path=submission_path,
        dry_run=True,
        timeout_minutes=None,
    )
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1" / "submit-iter-1"
    kernel_text = (kernel_dir / "kernel.py").read_text(encoding="utf-8")
    assert "__SUBMISSION_GZIP_B64__" not in kernel_text
    assert "SUBMISSION_GZIP_B64 = " in kernel_text
    assert not (kernel_dir / "submission_source.csv").exists()
    assert '"kind": "submit_only"' not in kernel_text
    assert "metrics_path.write_text" not in kernel_text
    assert "Training metrics.json is preserved" in kernel_text

    payload_match = re.search(r"SUBMISSION_GZIP_B64 = \"([A-Za-z0-9+/=]+)\"", kernel_text)
    assert payload_match is not None
    encoded = payload_match.group(1)
    decoded = gzip.decompress(base64.b64decode(encoded.encode("ascii"))).decode("utf-8")
    assert decoded == submission_path.read_text(encoding="utf-8")

    payload = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert payload["competition_sources"] == ["demo"]
    assert payload["code_file"] == "kernel.py"
    assert payload["enable_gpu"] is True
    assert payload["enable_tpu"] is False


def test_run_submit_kernel_dry_run_accepts_directory_submission(tmp_path: Path) -> None:
    from kagglebot import kernel_runner

    kernel_runner.kernels_init = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("should not run"))
    submission_path = tmp_path / "submission.zarr"
    (submission_path / "arrays").mkdir(parents=True)
    (submission_path / ".zgroup").write_text("{}", encoding="utf-8")
    (submission_path / "arrays" / "0").write_bytes(b"chunk")

    run_submit_kernel(
        slug="demo",
        run_id="run-1",
        iteration=1,
        base_dir=tmp_path,
        kaggle_username="user",
        kernel_name=None,
        accelerator="gpu",
        enable_internet=False,
        submission_path=submission_path,
        dry_run=True,
        timeout_minutes=None,
    )

    kernel_dir = tmp_path / "demo" / "kernels" / "run-1" / "submit-iter-1"
    kernel_text = (kernel_dir / "kernel.py").read_text(encoding="utf-8")
    assert 'SUBMISSION_OUTPUT_NAME = "submission.zarr.zip"' in kernel_text
    assert 'SUBMISSION_INPUT_SUFFIX = ".zip"' in kernel_text


def test_run_submit_kernel_wrapper_rejects_tiny_code_competition_submission(tmp_path: Path) -> None:
    from kagglebot import kernel_runner

    kernel_runner.kernels_init = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("should not run"))
    slug = "demo"
    context_dir = tmp_path / slug / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "overview.md").write_text(
        "This is a Code Competition. The public test set is dummy data and hidden/full test runs in Kaggle.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n3,0.3\n", encoding="utf-8")

    with pytest.raises(KernelFailedError, match="static wrapper submit kernel"):
        run_submit_kernel(
            slug=slug,
            run_id="run-1",
            iteration=1,
            base_dir=tmp_path,
            kaggle_username="user",
            kernel_name=None,
            accelerator="gpu",
            enable_internet=False,
            submission_path=submission_path,
            mode="wrapper",
            dry_run=True,
            timeout_minutes=None,
        )


def test_run_submit_kernel_wrapper_rejects_tiny_tsv_code_competition_submission(tmp_path: Path) -> None:
    from kagglebot import kernel_runner

    kernel_runner.kernels_init = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("should not run"))
    slug = "demo"
    context_dir = tmp_path / slug / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "overview.md").write_text(
        "This is a Code Competition. The public test set is dummy data and hidden/full test runs in Kaggle.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.tsv"
    submission_path.write_text("id\ttarget\n1\t0.1\n2\t0.2\n3\t0.3\n", encoding="utf-8")

    with pytest.raises(KernelFailedError, match="static wrapper submit kernel"):
        run_submit_kernel(
            slug=slug,
            run_id="run-1",
            iteration=1,
            base_dir=tmp_path,
            kaggle_username="user",
            kernel_name=None,
            accelerator="gpu",
            enable_internet=False,
            submission_path=submission_path,
            mode="wrapper",
            dry_run=True,
            timeout_minutes=None,
        )


def test_run_submit_kernel_wrapper_aligns_to_runtime_sample_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kagglebot import kernel_runner

    pd = pytest.importorskip("pandas")
    kernel_runner.kernels_init = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("should not run"))
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text(
        "\n".join(
            [
                "id,winner_model_a,winner_model_b,winner_tie",
                "1,0.2,0.3,0.5",
                "2,0.6,0.2,0.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_submit_kernel(
        slug="demo",
        run_id="run-1",
        iteration=1,
        base_dir=tmp_path,
        kaggle_username="user",
        kernel_name=None,
        accelerator="gpu",
        enable_internet=False,
        submission_path=submission_path,
        dry_run=True,
        timeout_minutes=None,
    )

    input_dir = tmp_path / "input" / "demo"
    input_dir.mkdir(parents=True)
    (input_dir / "AnswerTemplate.csv").write_text(
        "\n".join(
            [
                "id,winner_model_a,winner_model_b,winner_tie",
                "2,0,0,0",
                "3,0,0,0",
                "1,0,0,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    working_dir = tmp_path / "working"
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(tmp_path / "input"))
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    kernel_dir = tmp_path / "demo" / "kernels" / "run-1" / "submit-iter-1"
    runpy.run_path(str(kernel_dir / "kernel.py"), run_name="__main__")

    out = pd.read_csv(working_dir / "submission.csv")
    assert out["id"].tolist() == [2, 3, 1]
    assert list(out.columns) == ["id", "winner_model_a", "winner_model_b", "winner_tie"]
    assert out.loc[0, ["winner_model_a", "winner_model_b", "winner_tie"]].tolist() == pytest.approx([0.6, 0.2, 0.2])
    assert out.loc[2, ["winner_model_a", "winner_model_b", "winner_tie"]].tolist() == pytest.approx([0.2, 0.3, 0.5])
    assert out.loc[1, ["winner_model_a", "winner_model_b", "winner_tie"]].tolist() == pytest.approx([0.4, 0.25, 0.35])
    assert out[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1).tolist() == pytest.approx(
        [1.0, 1.0, 1.0]
    )


def test_run_submit_kernel_wrapper_runtime_validation_rejects_column_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kagglebot import kernel_runner

    kernel_runner.kernels_init = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("should not run"))
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,prediction\n1,0.2\n2,0.8\n", encoding="utf-8")
    run_submit_kernel(
        slug="demo",
        run_id="run-1",
        iteration=1,
        base_dir=tmp_path,
        kaggle_username="user",
        kernel_name=None,
        accelerator="gpu",
        enable_internet=False,
        submission_path=submission_path,
        dry_run=True,
        timeout_minutes=None,
    )

    input_dir = tmp_path / "input" / "demo"
    input_dir.mkdir(parents=True)
    (input_dir / "sample_submission.csv").write_text("id,target\n1,0.0\n2,0.0\n", encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(tmp_path / "input"))
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(tmp_path / "working"))

    kernel_dir = tmp_path / "demo" / "kernels" / "run-1" / "submit-iter-1"
    with pytest.raises(RuntimeError, match="columns mismatch"):
        runpy.run_path(str(kernel_dir / "kernel.py"), run_name="__main__")


def test_run_submit_kernel_wrapper_expands_tiny_runtime_sample_to_hidden_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kagglebot import kernel_runner

    pd = pytest.importorskip("pandas")
    kernel_runner.kernels_init = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("should not run"))
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text(
        "\n".join(
            [
                "id,winner_model_a,winner_model_b,winner_tie",
                "1,0.2,0.3,0.5",
                "2,0.6,0.2,0.2",
                "3,0.1,0.7,0.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_submit_kernel(
        slug="demo",
        run_id="run-1",
        iteration=1,
        base_dir=tmp_path,
        kaggle_username="user",
        kernel_name=None,
        accelerator="gpu",
        enable_internet=False,
        submission_path=submission_path,
        dry_run=True,
        timeout_minutes=None,
    )

    input_dir = tmp_path / "input" / "competitions" / "demo"
    input_dir.mkdir(parents=True)
    (input_dir / "sample_submission.csv").write_text(
        "\n".join(
            [
                "id,winner_model_a,winner_model_b,winner_tie",
                "1,0.333333,0.333333,0.333334",
                "2,0.333333,0.333333,0.333334",
                "3,0.333333,0.333333,0.333334",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (input_dir / "test.csv").write_text(
        "\n".join(
            [
                "id,prompt,response_a,response_b",
                "2,p,a,b",
                "4,p,a,b",
                "1,p,a,b",
                "5,p,a,b",
                "3,p,a,b",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    working_dir = tmp_path / "working"
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(tmp_path / "input"))
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    kernel_dir = tmp_path / "demo" / "kernels" / "run-1" / "submit-iter-1"
    runpy.run_path(str(kernel_dir / "kernel.py"), run_name="__main__")

    out = pd.read_csv(working_dir / "submission.csv")
    assert out["id"].tolist() == [2, 4, 1, 5, 3]
    assert list(out.columns) == ["id", "winner_model_a", "winner_model_b", "winner_tie"]
    assert out.loc[0, ["winner_model_a", "winner_model_b", "winner_tie"]].tolist() == pytest.approx([0.6, 0.2, 0.2])
    assert out.loc[2, ["winner_model_a", "winner_model_b", "winner_tie"]].tolist() == pytest.approx([0.2, 0.3, 0.5])
    assert out.loc[4, ["winner_model_a", "winner_model_b", "winner_tie"]].tolist() == pytest.approx([0.1, 0.7, 0.2])
    assert out.loc[1, ["winner_model_a", "winner_model_b", "winner_tie"]].tolist() == pytest.approx([0.3, 0.4, 0.3])
    assert out.loc[3, ["winner_model_a", "winner_model_b", "winner_tie"]].tolist() == pytest.approx([0.3, 0.4, 0.3])
    assert out[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1).tolist() == pytest.approx(
        [1.0, 1.0, 1.0, 1.0, 1.0]
    )


def test_run_submit_kernel_wrapper_aligns_tsv_runtime_sample_and_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kagglebot import kernel_runner

    pd = pytest.importorskip("pandas")
    kernel_runner.kernels_init = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("should not run"))
    submission_path = tmp_path / "submission.tsv"
    submission_path.write_text("id\ttarget\n1\t0.1\n2\t0.9\n", encoding="utf-8")

    run_submit_kernel(
        slug="demo",
        run_id="run-1",
        iteration=1,
        base_dir=tmp_path,
        kaggle_username="user",
        kernel_name=None,
        accelerator="gpu",
        enable_internet=False,
        submission_path=submission_path,
        dry_run=True,
        timeout_minutes=None,
    )

    input_dir = tmp_path / "input" / "demo"
    input_dir.mkdir(parents=True)
    (input_dir / "sample_submission.tsv").write_text("id\ttarget\n1\t0.0\n2\t0.0\n", encoding="utf-8")
    (input_dir / "test.tsv").write_text("id\tfeature\n2\t20\n3\t30\n1\t10\n", encoding="utf-8")
    working_dir = tmp_path / "working"
    monkeypatch.setenv("KAGGLEBOT_INPUT_ROOT", str(tmp_path / "input"))
    monkeypatch.setenv("KAGGLEBOT_WORKING_DIR", str(working_dir))

    kernel_dir = tmp_path / "demo" / "kernels" / "run-1" / "submit-iter-1"
    runpy.run_path(str(kernel_dir / "kernel.py"), run_name="__main__")

    out_path = working_dir / "submission.tsv"
    assert out_path.exists()
    out = pd.read_csv(out_path, sep="\t")
    assert out["id"].tolist() == [2, 3, 1]
    assert out["target"].tolist() == pytest.approx([0.9, 0.5, 0.1])


def test_run_submit_kernel_dry_run_inference_mode_stages_authoritative_kernel(tmp_path: Path) -> None:
    from kagglebot import kernel_runner

    kernel_runner.kernels_init = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("should not run"))
    slug = "demo"
    source_kernel_dir = tmp_path / slug / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    (source_kernel_dir / "output").mkdir(parents=True, exist_ok=True)
    (source_kernel_dir / "output" / "stale-submission.csv").write_text("id,target\n1,0.1\n", encoding="utf-8")
    (source_kernel_dir / "kernel.py").write_text(
        "\n".join(
            [
                "import os",
                "from pathlib import Path",
                "KERNEL_DIR = Path(__file__).resolve().parent",
                "ARTIFACT_DIR = KERNEL_DIR.parent",
                "ARTIFACT_ROOT = KERNEL_DIR.parent",
                "DATA = Path('/kaggle/input/demo/test.csv')",
                "LOCAL_OUTPUT_DIR = Path(os.environ.get('KAGGLEBOT_LOCAL_OUTPUT_DIR', str(KERNEL_DIR / 'outputs')))",
                "KAGGLE_WORKING_DIR = Path('/kaggle/working')",
                "LOCAL_OUT = KERNEL_DIR / 'outputs'",
                "ARTIFACT_OUT = ARTIFACT_DIR.joinpath('outputs')",
                "ROOT_OUT = ARTIFACT_ROOT / 'output'",
                "OUT = Path('/kaggle/working/submission.csv')",
                "MET = Path('/kaggle/working/metrics.json')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / slug / "plan.json").write_text(json.dumps({"toggles": {"USE_MODEL": True}}, indent=2), encoding="utf-8")
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n", encoding="utf-8")

    run_submit_kernel(
        slug=slug,
        run_id="run-1",
        iteration=1,
        base_dir=tmp_path,
        kaggle_username="user",
        kernel_name=None,
        accelerator="gpu",
        enable_internet=False,
        submission_path=submission_path,
        mode="inference",
        dry_run=True,
        timeout_minutes=None,
    )

    kernel_dir = tmp_path / slug / "kernels" / "run-1" / "submit-iter-1"
    kernel_text = (kernel_dir / "kernel.py").read_text(encoding="utf-8")
    assert "SUBMISSION_GZIP_B64" not in kernel_text
    assert "# kagglebot:submit_inference" in kernel_text
    assert "_kb_os.environ['KAGGLEBOT_DO_TRAIN'] = '0'" in kernel_text
    assert "_kb_os.environ['KAGGLEBOT_DO_INFER'] = '1'" in kernel_text
    assert "_kb_os.environ['KAGGLEBOT_SUBMIT_NOTEBOOK'] = '1'" in kernel_text
    assert "_kb_os.environ['KAGGLEBOT_SUBMIT_SKIP_CV'] = '1'" in kernel_text
    assert "/kaggle/working/submission.csv" in kernel_text
    assert "/kaggle/working/metrics.json" in kernel_text
    assert "LOCAL_OUTPUT_DIR = KERNEL_DIR / 'outputs'" not in kernel_text
    assert "KERNEL_DIR / 'outputs'" not in kernel_text
    assert "ARTIFACT_DIR.joinpath('outputs')" not in kernel_text
    assert "ARTIFACT_ROOT / 'output'" not in kernel_text
    assert "str(KAGGLE_WORKING_DIR)" not in kernel_text
    assert "LOCAL_OUTPUT_DIR = KAGGLE_WORKING_DIR" not in kernel_text
    assert "Path('/kaggle/working')" in kernel_text
    assert not (kernel_dir / "output").exists()
    payload = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert payload["enable_gpu"] is True
    assert payload["enable_tpu"] is False


def test_run_submit_kernel_allows_submit_accelerator_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kagglebot import kernel_runner

    kernel_runner.kernels_init = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("should not run"))
    monkeypatch.setenv("KAGGLEBOT_SUBMIT_KERNEL_ACCELERATOR", "gpu")
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n", encoding="utf-8")

    run_submit_kernel(
        slug="demo",
        run_id="run-1",
        iteration=1,
        base_dir=tmp_path,
        kaggle_username="user",
        kernel_name=None,
        accelerator="cpu",
        enable_internet=False,
        submission_path=submission_path,
        dry_run=True,
        timeout_minutes=None,
    )

    payload = json.loads(
        (tmp_path / "demo" / "kernels" / "run-1" / "submit-iter-1" / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    assert payload["enable_gpu"] is True
    assert payload["enable_tpu"] is False


@pytest.mark.parametrize(
    "output_expr",
    [
        "OUT = Path('/kaggle/src/output')",
        "OUT = Path('/kaggle/src/outputs')",
        "OUT = Path('/kaggle/src').joinpath('outputs')",
    ],
)
def test_run_submit_kernel_inference_mode_rejects_read_only_output_patterns(
    tmp_path: Path,
    output_expr: str,
) -> None:
    from kagglebot import kernel_runner

    kernel_runner.kernels_init = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("should not run"))
    slug = "demo"
    source_kernel_dir = tmp_path / slug / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    (source_kernel_dir / "kernel.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "DATA = Path('/kaggle/input/demo/test.csv')",
                output_expr,
                "SUB = Path('/kaggle/working/submission.csv')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / slug / "plan.json").write_text(json.dumps({"toggles": {"USE_MODEL": True}}, indent=2), encoding="utf-8")
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.1\n", encoding="utf-8")

    with pytest.raises(KernelFailedError, match="Invalid notebook submit artifact"):
        run_submit_kernel(
            slug=slug,
            run_id="run-1",
            iteration=1,
            base_dir=tmp_path,
            kaggle_username="user",
            kernel_name=None,
            accelerator="gpu",
            enable_internet=False,
            submission_path=submission_path,
            mode="inference",
            dry_run=True,
            timeout_minutes=None,
        )


def test_kernel_push_injects_competition_slug_before_push(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kagglebot import kernel_runner

    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    output_dir = tmp_path / "demo" / "runs" / "run-1" / "iter-1" / "output"
    logs_dir = tmp_path / "demo" / "runs" / "run-1" / "iter-1" / "logs"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    (kernel_dir / "kernel.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "DATA = '/kaggle/input/demo/train.csv'",
                "OUT1 = '/kaggle/working/submission.csv'",
                "OUT2 = '/kaggle/working/metrics.json'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_kernels_push(package_dir: Path, *, slug: str, dry_run: bool) -> str:
        assert dry_run is False
        assert slug == "demo"
        kernel_text = (package_dir / "kernel.py").read_text(encoding="utf-8")
        assert "# kagglebot:competition_slug" in kernel_text
        assert "_kb_os.environ['KAGGLEBOT_COMPETITION_SLUG'] = \"demo\"" in kernel_text
        assert "# kagglebot:force_train" in kernel_text
        assert "_kb_os.environ['KAGGLEBOT_DO_TRAIN'] = '1'" in kernel_text
        assert "_kb_os.environ['KAGGLEBOT_FORCE_TRAIN'] = '1'" in kernel_text
        return "https://www.kaggle.com/code/user/kernel-slug"

    monkeypatch.setattr(kernel_runner, "kernels_push", fake_kernels_push)
    monkeypatch.setattr(kernel_runner, "kernel_id_by_title", lambda *args, **kwargs: None)
    monkeypatch.setattr(kernel_runner, "_wait_for_kernel_registration", lambda *args, **kwargs: "user/kernel-slug")
    monkeypatch.setattr(kernel_runner, "_wait_for_kernel", lambda *args, **kwargs: None)
    monkeypatch.setattr(kernel_runner, "kernels_output", lambda *args, **kwargs: "")

    preparation = kernel_runner.KernelPreparation(
        kernel_dir=kernel_dir,
        output_dir=output_dir,
        logs_dir=logs_dir,
        kernel_slug="kernel-slug",
        kernel_id="user/kernel-slug",
    )
    started: list[str] = []
    kernel_id = kernel_runner.KernelJobMonitor().push_and_wait(
        preparation=preparation,
        slug="demo",
        timeout_minutes=1,
        on_remote_started=started.append,
    )
    assert kernel_id == "user/kernel-slug"
    assert started == ["user/kernel-slug"]


def test_kernel_push_clears_stale_output_before_wait(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kagglebot import kernel_runner

    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    output_dir = tmp_path / "demo" / "runs" / "run-1" / "iter-1" / "output"
    logs_dir = tmp_path / "demo" / "runs" / "run-1" / "iter-1" / "logs"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    stale_log = output_dir / "old-error.log"
    stale_log.write_text("Traceback (most recent call last):\nboom\n", encoding="utf-8")
    stale_submission = output_dir / "submission.csv"
    stale_submission.write_text("id,target\n1,0\n", encoding="utf-8")

    (kernel_dir / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    monkeypatch.setattr(
        kernel_runner, "kernels_push", lambda *args, **kwargs: "https://www.kaggle.com/code/user/kernel-slug"
    )
    monkeypatch.setattr(kernel_runner, "kernel_id_by_title", lambda *args, **kwargs: None)
    monkeypatch.setattr(kernel_runner, "_wait_for_kernel_registration", lambda *args, **kwargs: "user/kernel-slug")

    observed: dict[str, bool] = {}

    def fake_wait(*args, **kwargs) -> None:
        observed["stale_log_exists"] = stale_log.exists()
        observed["stale_submission_exists"] = stale_submission.exists()

    monkeypatch.setattr(kernel_runner, "_wait_for_kernel", fake_wait)
    monkeypatch.setattr(kernel_runner, "kernels_output", lambda *args, **kwargs: "")

    preparation = kernel_runner.KernelPreparation(
        kernel_dir=kernel_dir,
        output_dir=output_dir,
        logs_dir=logs_dir,
        kernel_slug="kernel-slug",
        kernel_id="user/kernel-slug",
    )
    kernel_runner.KernelJobMonitor().push_and_wait(
        preparation=preparation,
        slug="demo",
        timeout_minutes=1,
    )
    assert observed["stale_log_exists"] is False
    assert observed["stale_submission_exists"] is False


def test_kernel_push_resumes_prior_running_kernel_without_new_push(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kagglebot import kernel_runner

    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    output_dir = tmp_path / "demo" / "runs" / "run-1" / "iter-1" / "output"
    logs_dir = tmp_path / "demo" / "runs" / "run-1" / "iter-1" / "logs"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text("print('ok')\n", encoding="utf-8")
    (logs_dir / "kernel_push-01.txt").write_text("Kernel version 1 successfully pushed.\n", encoding="utf-8")

    statuses = iter(
        [
            'user/kernel-slug has status "KernelWorkerStatus.RUNNING"',
            'user/kernel-slug has status "KernelWorkerStatus.COMPLETE"',
        ]
    )
    monkeypatch.setattr(kernel_runner, "kernels_status", lambda *args, **kwargs: next(statuses))
    monkeypatch.setattr(kernel_runner.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(kernel_runner, "_try_fetch_kernel_output", lambda *args, **kwargs: None)
    monkeypatch.setattr(kernel_runner._kernel_logs, "print_kernel_logs", lambda *args, **kwargs: False)
    monkeypatch.setattr(kernel_runner, "kernels_push", lambda *args, **kwargs: pytest.fail("unexpected push"))
    output_calls: list[str] = []
    monkeypatch.setattr(
        kernel_runner, "kernels_output", lambda kernel_id, *args, **kwargs: output_calls.append(kernel_id)
    )

    preparation = kernel_runner.KernelPreparation(
        kernel_dir=kernel_dir,
        output_dir=output_dir,
        logs_dir=logs_dir,
        kernel_slug="kernel-slug",
        kernel_id="user/kernel-slug",
    )
    kernel_id = kernel_runner.KernelJobMonitor().push_and_wait(
        preparation=preparation,
        slug="demo",
        timeout_minutes=1,
    )

    assert kernel_id == "user/kernel-slug"
    assert output_calls == ["user/kernel-slug"]


def test_submit_kernel_resume_supersedes_stale_queued_kernel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kagglebot import kernel_runner

    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    output_dir = tmp_path / "demo" / "runs" / "run-1" / "iter-1" / "output"
    logs_dir = tmp_path / "demo" / "runs" / "run-1" / "iter-1" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    push_log = logs_dir / "kernel_push-01.txt"
    push_log.write_text("Kernel version 1 successfully pushed.\n", encoding="utf-8")
    os.utime(push_log, (100.0, 100.0))
    monkeypatch.setenv("KAGGLEBOT_KERNEL_QUEUED_TIMEOUT_SEC", "30")
    monkeypatch.setattr(kernel_runner.time, "time", lambda: 1000.0)
    monkeypatch.setattr(kernel_runner.time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(
        kernel_runner,
        "kernels_status",
        lambda *args, **kwargs: 'user/kernel-slug has status "KernelWorkerStatus.QUEUED"',
    )

    preparation = kernel_runner.KernelPreparation(
        kernel_dir=kernel_dir,
        output_dir=output_dir,
        logs_dir=logs_dir,
        kernel_slug="kernel-slug",
        kernel_id="user/kernel-slug",
        supersede_stale_queued=True,
    )

    assert (
        kernel_runner._resume_prior_kernel_if_active(  # noqa: SLF001
            preparation=preparation,
            kernel_id="user/kernel-slug",
            slug="demo",
            timeout_minutes=1,
        )
        is None
    )


@pytest.mark.parametrize("status", ["KernelWorkerStatus.RUNNING", "KernelWorkerStatus.QUEUED"])
def test_wait_for_kernel_timeout_marks_remote_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    from kagglebot import kernel_runner

    times = iter([0.0, 0.0, 61.0, 62.0])
    monkeypatch.setattr(kernel_runner.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(
        kernel_runner,
        "kernels_status",
        lambda *args, **kwargs: f'user/kernel-slug has status "{status}"',
    )
    monkeypatch.setattr(kernel_runner, "_try_fetch_kernel_output", lambda *args, **kwargs: None)
    monkeypatch.setattr(kernel_runner._kernel_logs, "print_kernel_logs", lambda *args, **kwargs: False)
    monkeypatch.setattr(kernel_runner.time, "sleep", lambda _seconds: None)

    expected_status = "queued" if status.endswith("QUEUED") else "running"
    with pytest.raises(KernelStillRunningError, match=f"still {expected_status}"):
        kernel_runner._wait_for_kernel("user/kernel-slug", "demo", 1, output_dir=tmp_path)  # noqa: SLF001


def test_wait_for_kernel_queued_timeout_raises_capacity_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kagglebot import kernel_runner

    monkeypatch.setenv("KAGGLEBOT_KERNEL_QUEUED_TIMEOUT_SEC", "30")
    monkeypatch.setattr(kernel_runner.time, "monotonic", lambda: 31.0)
    monkeypatch.setattr(
        kernel_runner,
        "kernels_status",
        lambda *args, **kwargs: 'user/kernel-slug has status "KernelWorkerStatus.QUEUED"',
    )

    with pytest.raises(KernelCapacityError, match="stayed queued"):
        kernel_runner._wait_for_kernel(  # noqa: SLF001
            "user/kernel-slug",
            "demo",
            None,
            output_dir=tmp_path,
            initial_queued_since=0.0,
        )


def test_wait_for_kernel_timeout_on_unknown_status_raises_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kagglebot import kernel_runner

    times = iter([0.0, 0.0, 61.0, 62.0])
    monkeypatch.setattr(kernel_runner.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(
        kernel_runner,
        "kernels_status",
        lambda *args, **kwargs: 'user/kernel-slug has status "KernelWorkerStatus.UNKNOWN"',
    )
    monkeypatch.setattr(kernel_runner, "_try_fetch_kernel_output", lambda *args, **kwargs: None)
    monkeypatch.setattr(kernel_runner._kernel_logs, "print_kernel_logs", lambda *args, **kwargs: False)
    monkeypatch.setattr(kernel_runner.time, "sleep", lambda _seconds: None)

    with pytest.raises(KernelTimeoutError, match="last status was unknown"):
        kernel_runner._wait_for_kernel("user/kernel-slug", "demo", 1, output_dir=tmp_path)  # noqa: SLF001


def test_kernel_push_aborts_immediately_on_invalid_attached_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kagglebot import kernel_runner

    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    output_dir = tmp_path / "demo" / "runs" / "run-1" / "iter-1" / "output"
    logs_dir = tmp_path / "demo" / "runs" / "run-1" / "iter-1" / "logs"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text("print('ok')\n", encoding="utf-8")
    (kernel_dir / "kernel-metadata.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        kernel_runner,
        "kernels_push",
        lambda *args, **kwargs: (
            "The following are not valid dataset sources and could not be added to the kernel: "
            "['alice/missing-dataset']\n"
            "Kernel version 1 successfully pushed."
        ),
    )

    observed = {"wait_called": False, "output_called": False}

    def fake_wait(*args, **kwargs) -> None:
        observed["wait_called"] = True

    def fake_output(*args, **kwargs) -> None:
        observed["output_called"] = True

    monkeypatch.setattr(kernel_runner, "_wait_for_kernel_registration", lambda *args, **kwargs: "user/kernel-slug")
    monkeypatch.setattr(kernel_runner, "_wait_for_kernel", fake_wait)
    monkeypatch.setattr(kernel_runner, "kernels_output", fake_output)

    preparation = kernel_runner.KernelPreparation(
        kernel_dir=kernel_dir,
        output_dir=output_dir,
        logs_dir=logs_dir,
        kernel_slug="kernel-slug",
        kernel_id="user/kernel-slug",
    )

    with pytest.raises(KernelFailedError, match="alice/missing-dataset"):
        kernel_runner.KernelJobMonitor().push_and_wait(
            preparation=preparation,
            slug="demo",
            timeout_minutes=1,
        )

    assert observed["wait_called"] is False
    assert observed["output_called"] is False


def test_ensure_kernel_competition_slug_env_rewrites_stale_slug(tmp_path: Path) -> None:
    from kagglebot.kernel_bootstrap import ensure_kernel_competition_slug_env

    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "# kagglebot:competition_slug",
                "import os as _kb_os",
                "_kb_os.environ['KAGGLEBOT_COMPETITION_SLUG'] = \"kaggle\"",
                "_kb_os.environ['KAGGLEBOT_SLUG'] = \"kaggle\"",
                "del _kb_os",
                "",
                "print('ok')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    ensure_kernel_competition_slug_env(kernel_dir, "demo")
    updated = kernel_path.read_text(encoding="utf-8")
    assert "_kb_os.environ['KAGGLEBOT_COMPETITION_SLUG'] = \"demo\"" in updated
    assert "_kb_os.environ['KAGGLEBOT_SLUG'] = \"demo\"" in updated
    assert "_kb_os.environ['KAGGLEBOT_COMPETITION_SLUG'] = \"kaggle\"" not in updated


def test_copy_kernel_sources_skips_output_dirs_and_copy_external_assets(tmp_path: Path) -> None:
    base_dir = tmp_path / "artifacts"
    slug = "playground-series-s6e3"
    kernel_source_dir = base_dir / slug / "kernel"
    (kernel_source_dir / "output").mkdir(parents=True, exist_ok=True)
    (kernel_source_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (kernel_source_dir / "__pycache__").mkdir(parents=True, exist_ok=True)
    (kernel_source_dir / "kernel.py").write_text("from runtime import main\n", encoding="utf-8")
    (kernel_source_dir / "runtime.py").write_text("def main():\n    pass\n", encoding="utf-8")
    (kernel_source_dir / "output" / "submission.csv").write_text("id,target\n", encoding="utf-8")
    (kernel_source_dir / "outputs" / "submission.csv").write_text("id,target\n", encoding="utf-8")
    (kernel_source_dir / "__pycache__" / "kernel.pyc").write_bytes(b"pyc")

    external_dir = base_dir / slug / "external"
    external_dir.mkdir(parents=True, exist_ok=True)
    (external_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv").write_text("customerID,Churn\nx,No\n", encoding="utf-8")
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)

    copy_kernel_sources(kernel_source_dir, kernel_dir)
    copy_shared_kernel_runtime_modules(kernel_dir)
    copy_competition_external_assets(base_dir=base_dir, slug=slug, kernel_dir=kernel_dir)

    assert (kernel_dir / "kernel.py").exists()
    assert (kernel_dir / "runtime.py").exists()
    assert (kernel_dir / "tabular_ensemble.py").exists()
    assert (kernel_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv").exists()
    assert not (kernel_dir / "output").exists()
    assert not (kernel_dir / "outputs").exists()
    assert not (kernel_dir / "__pycache__").exists()


def test_sync_plan_snapshot_skips_self_copy_and_writes_targets(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text('{"name": "demo"}\n', encoding="utf-8")
    target = tmp_path / "kernel" / "plan.json"

    sync_plan_snapshot(plan_path=plan_path, targets=[plan_path, target])

    assert target.read_text(encoding="utf-8") == plan_path.read_text(encoding="utf-8")


def test_run_local_kernel_once_does_not_wait_for_inherited_stdout_holders(tmp_path: Path) -> None:
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(
        (
            "import subprocess\n"
            "import sys\n"
            "\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])\n"
            "print('kernel parent exited', flush=True)\n"
        ),
        encoding="utf-8",
    )

    started = time.monotonic()
    result = run_local_kernel_once(
        kernel_path=kernel_path,
        kernel_stage_dir=tmp_path,
        current_env=os.environ.copy(),
        timeout_sec=5,
        line_callback=None,
        progress_tracker=None,
    )
    elapsed = time.monotonic() - started

    assert result.command_result.returncode == 0
    assert result.command_result.args[1] == "-u"
    assert "kernel parent exited" in result.command_result.stdout
    assert elapsed < 5


def test_should_suppress_local_kernel_log_line_filters_fragmentation_and_catboost_noise() -> None:
    state = LocalKernelLogFilterState()
    lines = [
        "/tmp/kernel.py:1036: PerformanceWarning: DataFrame is highly fragmented.\n",
        "  out[ratio_col] = out[t1] / (out[t2].abs() + 1e-6)\n",
        "Default metric period is 5 because BrierScore is/are not implemented for GPU\n",
        "training fold=1\n",
    ]

    suppressed = [should_suppress_local_kernel_log_line(line, state=state) for line in lines]

    assert suppressed == [True, True, True, False]


def test_run_local_kernel_once_suppresses_known_warning_noise(tmp_path: Path) -> None:
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(
        (
            "print('/tmp/kernel.py:1036: PerformanceWarning: DataFrame is highly fragmented.', flush=True)\n"
            "print('  out[ratio_col] = out[t1] / (out[t2].abs() + 1e-6)', flush=True)\n"
            "print('Default metric period is 5 because BrierScore is/are not implemented for GPU', flush=True)\n"
            "print('training fold=1', flush=True)\n"
        ),
        encoding="utf-8",
    )

    result = run_local_kernel_once(
        kernel_path=kernel_path,
        kernel_stage_dir=tmp_path,
        current_env=os.environ.copy(),
        timeout_sec=5,
        line_callback=None,
        progress_tracker=None,
    )

    assert result.command_result.returncode == 0
    assert "training fold=1" in result.command_result.stdout
    assert "PerformanceWarning" not in result.command_result.stdout
    assert "BrierScore is/are not implemented for GPU" not in result.command_result.stdout


def test_run_local_kernel_once_counts_partial_stdout_as_activity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAGGLEBOT_LOCAL_KERNEL_STALL_SEC", "1")
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(
        (
            "import sys\n"
            "import time\n"
            "for _ in range(8):\n"
            "    sys.stdout.write('.')\n"
            "    sys.stdout.flush()\n"
            "    time.sleep(0.25)\n"
            "print('done', flush=True)\n"
        ),
        encoding="utf-8",
    )
    tracker = build_local_kernel_progress_tracker(base_dir=tmp_path, slug="demo", watch_dirs=[])

    result = run_local_kernel_once(
        kernel_path=kernel_path,
        kernel_stage_dir=tmp_path,
        current_env=os.environ.copy(),
        timeout_sec=5,
        line_callback=tracker.observe_line,
        progress_tracker=tracker,
    )

    assert result.command_result.returncode == 0
    assert result.killed_for_stall is False
    assert "done" in result.command_result.stdout


def test_kernel_metadata_tpu(tmp_path: Path) -> None:
    kernel_path = tmp_path / "demo" / "kernel" / "kernel.py"
    kernel_path.parent.mkdir(parents=True, exist_ok=True)
    kernel_path.write_text(
        "\n".join(
            [
                "DATA = '/kaggle/input/demo/train.csv'",
                "OUT1 = '/kaggle/working/submission.csv'",
                "OUT2 = '/kaggle/working/metrics.json'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_kernel(
        slug="demo",
        run_id="run-2",
        iteration=1,
        base_dir=tmp_path,
        kaggle_username="user",
        kernel_name=None,
        accelerator="tpu",
        enable_internet=False,
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        seed=42,
        dry_run=True,
        timeout_minutes=None,
    )
    meta_path = tmp_path / "demo" / "kernels" / "run-2" / "kernel-metadata.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["enable_tpu"] is True
    assert payload["enable_gpu"] is False
    assert payload["kernel_type"] == "script"
    assert payload["code_file"] == "kernel.py"


def test_inject_column_fill_shim(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text("print('ok')\n", encoding="utf-8")
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    payload = {"files": {"test.csv": ["A", "B"]}}
    (context_dir / "column_fill.json").write_text(json.dumps(payload), encoding="utf-8")

    inject_column_fill_shim(kernel_dir, context_dir)

    site_path = kernel_dir / "sitecustomize.py"
    assert site_path.exists()
    text = site_path.read_text(encoding="utf-8")
    assert "column-fill-shim" in text
    assert "column_fill.json" in text
    assert "_pd.DataFrame.__getitem__" in text
    assert "'read_sas'" in text
    assert "'read_spss'" in text
    assert "'read_html'" in text
    assert "float('nan')" in text
    assert "_pd.NA" not in text
    assert (kernel_dir / "column_fill.json").exists()


def test_column_fill_shim_wraps_non_csv_pandas_readers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    payload = {"files": {"table.parquet": ["missing_feature"]}}
    (context_dir / "column_fill.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(pd, "read_parquet", lambda path, *args, **kwargs: pd.DataFrame({"id": [1, 2]}))
    inject_column_fill_shim(kernel_dir, context_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    frame = pd.read_parquet(kernel_dir / "table.parquet")

    assert frame["id"].tolist() == [1, 2]
    assert "missing_feature" in frame.columns
    assert frame["missing_feature"].isna().all()


@pytest.mark.parametrize(
    ("reader_name", "suffix"),
    [
        ("read_orc", ".orc"),
        ("read_hdf", ".hdf5"),
        ("read_sas", ".sas7bdat"),
        ("read_spss", ".sav"),
    ],
)
def test_column_fill_shim_wraps_binary_pandas_readers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader_name: str,
    suffix: str,
) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    payload = {"files": {f"table{suffix}": ["missing_feature"]}}
    (context_dir / "column_fill.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(pd, reader_name, lambda path, *args, **kwargs: pd.DataFrame({"id": [1, 2]}))
    inject_column_fill_shim(kernel_dir, context_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    frame = getattr(pd, reader_name)(kernel_dir / f"table{suffix}")

    assert frame["id"].tolist() == [1, 2]
    assert "missing_feature" in frame.columns
    assert frame["missing_feature"].isna().all()


def test_column_fill_shim_wraps_html_reader_lists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    payload = {"files": {"sample_submission.html": ["missing_feature"]}}
    (context_dir / "column_fill.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(pd, "read_html", lambda path, *args, **kwargs: [pd.DataFrame({"id": [1, 2]})])
    inject_column_fill_shim(kernel_dir, context_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    tables = pd.read_html(kernel_dir / "sample_submission.html")

    assert len(tables) == 1
    assert tables[0]["id"].tolist() == [1, 2]
    assert "missing_feature" in tables[0].columns
    assert tables[0]["missing_feature"].isna().all()


def test_column_fill_shim_matches_same_stem_non_csv_pandas_readers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    payload = {"files": {"test.csv": ["missing_feature"]}}
    (context_dir / "column_fill.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(pd, "read_parquet", lambda path, *args, **kwargs: pd.DataFrame({"id": [1, 2]}))
    inject_column_fill_shim(kernel_dir, context_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    frame = pd.read_parquet(kernel_dir / "test.parquet")

    assert frame["id"].tolist() == [1, 2]
    assert "missing_feature" in frame.columns
    assert frame["missing_feature"].isna().all()


def test_inject_column_map_shim(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text("print('ok')\n", encoding="utf-8")
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    payload = {"mapping": {"old": "new"}}
    (context_dir / "column_map.json").write_text(json.dumps(payload), encoding="utf-8")

    inject_column_map_shim(kernel_dir, context_dir)
    inject_column_map_shim(kernel_dir, context_dir)

    site_path = kernel_dir / "sitecustomize.py"
    assert site_path.exists()
    text = site_path.read_text(encoding="utf-8")
    assert text.count("column-map-shim") == 1
    assert "column_map.json" in text
    assert "_kb_apply_column_map" in text
    assert "item.rename(columns=mapping)" in text
    assert "result.rename(columns=mapping)" in text
    assert "'read_sas'" in text
    assert "'read_spss'" in text
    assert "'read_html'" in text
    assert (kernel_dir / "column_map.json").exists()


def test_column_map_shim_wraps_non_csv_pandas_readers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    payload = {"mapping": {"old_name": "new_name"}}
    (context_dir / "column_map.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(pd, "read_parquet", lambda path, *args, **kwargs: pd.DataFrame({"old_name": [1, 2]}))
    inject_column_map_shim(kernel_dir, context_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    frame = pd.read_parquet(kernel_dir / "table.parquet")

    assert frame.columns.tolist() == ["new_name"]
    assert frame["new_name"].tolist() == [1, 2]


@pytest.mark.parametrize("reader_name", ["read_orc", "read_hdf", "read_sas", "read_spss"])
def test_column_map_shim_wraps_binary_pandas_readers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader_name: str,
) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    payload = {"mapping": {"old_name": "new_name"}}
    (context_dir / "column_map.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(pd, reader_name, lambda path, *args, **kwargs: pd.DataFrame({"old_name": [1, 2]}))
    inject_column_map_shim(kernel_dir, context_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    frame = getattr(pd, reader_name)(kernel_dir / "table")

    assert frame.columns.tolist() == ["new_name"]
    assert frame["new_name"].tolist() == [1, 2]


def test_column_map_shim_wraps_html_reader_lists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    payload = {"mapping": {"old_name": "new_name"}}
    (context_dir / "column_map.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(pd, "read_html", lambda path, *args, **kwargs: [pd.DataFrame({"old_name": [1, 2]})])
    inject_column_map_shim(kernel_dir, context_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    tables = pd.read_html(kernel_dir / "table.html")

    assert len(tables) == 1
    assert tables[0].columns.tolist() == ["new_name"]
    assert tables[0]["new_name"].tolist() == [1, 2]


def test_prepare_zero_overlap_drift_guard_detects_high_risk_zero_overlap_feature(tmp_path: Path) -> None:
    from kagglebot.local_kernel_drift_guard import prepare_zero_overlap_drift_guard

    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "\n".join(
            [
                "id,risk_cat,safe_cat,target",
                "A,x,same,1",
                "B,x,same,1",
                "C,x,same,1",
                "D,y,same,0",
                "E,y,same,0",
                "F,y,same,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text(
        "\n".join(
            [
                "id,risk_cat,safe_cat",
                "T1,u,same",
                "T2,u,same",
                "T3,v,same",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    context_dir = tmp_path / "demo" / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "dataset_profile.json").write_text(
        json.dumps({"target_column": "target", "id_column": "id"}, indent=2),
        encoding="utf-8",
    )

    guard_path = prepare_zero_overlap_drift_guard(
        base_dir=tmp_path,
        slug="demo",
        context_dir=context_dir,
    )

    assert guard_path is not None and guard_path.exists()
    payload = json.loads(guard_path.read_text(encoding="utf-8"))
    assert payload["enabled"] is True
    assert "risk_cat" in payload["drop_columns"]
    assert "id" not in payload["drop_columns"]
    assert payload["reason"] == "zero_overlap_high_drift_detected"


def test_inject_zero_overlap_drift_shim(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text("print('ok')\n", encoding="utf-8")
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    payload = {"enabled": True, "drop_columns": ["risk_cat"]}
    (context_dir / "zero_overlap_drift_guard.json").write_text(json.dumps(payload), encoding="utf-8")

    inject_zero_overlap_drift_shim(kernel_dir, context_dir)
    inject_zero_overlap_drift_shim(kernel_dir, context_dir)

    site_path = kernel_dir / "sitecustomize.py"
    assert site_path.exists()
    text = site_path.read_text(encoding="utf-8")
    assert "zero-overlap-drift-shim" in text
    assert text.count("zero-overlap-drift-shim") == 1
    assert "zero_overlap_drift_guard.json" in text
    assert "'.sqlite3'" in text
    assert "'.duckdb'" in text
    assert "'.rds'" in text
    assert "'.pkl.zst'" in text
    assert "'read_sas'" in text
    assert "'read_spss'" in text
    assert "'read_html'" in text
    assert "_KB_ROLE_ALIASES" in text
    assert "_KB_ROLE_SUFFIXES" in text
    assert "_kb_tabular_stem(path).lower() in {'train', 'test'}" not in text
    assert ".sqlite3" in TABULAR_INPUT_SUFFIXES
    assert ".duckdb" in TABULAR_INPUT_SUFFIXES
    assert ".rds" in TABULAR_INPUT_SUFFIXES
    assert ".pkl.zst" in TABULAR_INPUT_SUFFIXES
    assert (kernel_dir / "zero_overlap_drift_guard.json").exists()


def test_zero_overlap_drift_shim_wraps_non_csv_pandas_readers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    payload = {"enabled": True, "drop_columns": ["risk_cat"]}
    (context_dir / "zero_overlap_drift_guard.json").write_text(json.dumps(payload), encoding="utf-8")

    def fake_read_parquet(path, *args, **kwargs):
        return pd.DataFrame({"id": [1, 2], "risk_cat": ["x", "y"], "safe_cat": ["a", "b"]})

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    inject_zero_overlap_drift_shim(kernel_dir, context_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    train = pd.read_parquet(kernel_dir / "train.parquet")
    leaderboard = pd.read_parquet(kernel_dir / "leaderboard_features.parquet")
    metadata = pd.read_parquet(kernel_dir / "metadata.parquet")

    assert train.columns.tolist() == ["id", "safe_cat"]
    assert leaderboard.columns.tolist() == ["id", "safe_cat"]
    assert metadata.columns.tolist() == ["id", "risk_cat", "safe_cat"]


@pytest.mark.parametrize(
    ("reader_name", "suffix"),
    [
        ("read_orc", ".orc"),
        ("read_hdf", ".hdf5"),
        ("read_sas", ".sas7bdat"),
        ("read_spss", ".sav"),
    ],
)
def test_zero_overlap_drift_shim_wraps_binary_pandas_readers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader_name: str,
    suffix: str,
) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    payload = {"enabled": True, "drop_columns": ["risk_cat"]}
    (context_dir / "zero_overlap_drift_guard.json").write_text(json.dumps(payload), encoding="utf-8")

    def fake_reader(path, *args, **kwargs):
        return pd.DataFrame({"id": [1, 2], "risk_cat": ["x", "y"], "safe_cat": ["a", "b"]})

    monkeypatch.setattr(pd, reader_name, fake_reader)
    inject_zero_overlap_drift_shim(kernel_dir, context_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    train = getattr(pd, reader_name)(kernel_dir / f"train{suffix}")
    training_set = getattr(pd, reader_name)(kernel_dir / f"TrainingSet{suffix}")
    metadata = getattr(pd, reader_name)(kernel_dir / f"metadata{suffix}")

    assert train.columns.tolist() == ["id", "safe_cat"]
    assert training_set.columns.tolist() == ["id", "safe_cat"]
    assert metadata.columns.tolist() == ["id", "risk_cat", "safe_cat"]


def test_zero_overlap_drift_shim_wraps_html_reader_lists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    payload = {"enabled": True, "drop_columns": ["risk_cat"]}
    (context_dir / "zero_overlap_drift_guard.json").write_text(json.dumps(payload), encoding="utf-8")

    def fake_read_html(path, *args, **kwargs):
        return [pd.DataFrame({"id": [1, 2], "risk_cat": ["x", "y"], "safe_cat": ["a", "b"]})]

    monkeypatch.setattr(pd, "read_html", fake_read_html)
    inject_zero_overlap_drift_shim(kernel_dir, context_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    train_tables = pd.read_html(kernel_dir / "train.html")
    metadata_tables = pd.read_html(kernel_dir / "metadata.html")

    assert train_tables[0].columns.tolist() == ["id", "safe_cat"]
    assert metadata_tables[0].columns.tolist() == ["id", "risk_cat", "safe_cat"]


def test_inject_pandas_tabular_read_shim(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)

    inject_pandas_tabular_read_shim(kernel_dir)
    inject_pandas_tabular_read_shim(kernel_dir)

    site_path = kernel_dir / "sitecustomize.py"
    assert site_path.exists()
    text = site_path.read_text(encoding="utf-8")
    assert "pandas-tabular-read-shim" in text
    assert text.count("pandas-tabular-read-shim") == 1
    assert "_KB_TABULAR_TEXT_SUFFIXES" in text
    assert "_KB_TABULAR_STRUCTURED_SUFFIXES" in text
    assert "_KB_TABULAR_PICKLE_SUFFIXES" in text
    assert "_KB_TABULAR_ARROW_IPC_SUFFIXES" in text
    assert "_KB_TABULAR_PARQUET_SUFFIXES" in text
    assert "_KB_TABULAR_EXCEL_INPUT_ONLY_SUFFIXES" in text
    assert "_KB_TABULAR_EXCEL_SUFFIXES" in text
    assert "_KB_TABULAR_GEOPACKAGE_SUFFIXES" in text
    assert "_KB_TABULAR_HDF_SUFFIXES" in text
    assert "_KB_TABULAR_JSON_LINES_SUFFIX_PREFIXES" in text
    assert "_KB_TABULAR_KML_SUFFIXES" in text
    assert "_KB_TABULAR_STATA_SUFFIXES" in text
    assert "_KB_TABULAR_SAS_SUFFIXES" in text
    assert "_KB_TABULAR_SHAPEFILE_SUFFIXES" in text
    assert "_KB_TABULAR_SPSS_SUFFIXES" in text
    assert "_KB_TABULAR_MATLAB_SUFFIXES" in text
    assert "_KB_TABULAR_ARFF_SUFFIXES" in text
    assert "_KB_TABULAR_HTML_SUFFIX_PREFIXES" in text
    assert "_KB_TABULAR_SVMLIGHT_SUFFIX_PREFIXES" in text
    assert "_KB_TABULAR_FIXED_WIDTH_SUFFIX_PREFIXES" in text
    assert "_pd.read_pickle(_kb_open_binary_sample(resolved_path, suffix))" in text
    assert "_pd.read_json(StringIO(_kb_read_text(resolved_path, suffix)), lines=True)" in text
    assert "_pd.read_excel(_kb_open_binary_sample(resolved_path, suffix))" in text
    assert "_pd.read_orc(_kb_open_binary_sample(resolved_path, suffix))" in text
    assert "_pd.read_parquet = _patched_read_parquet" in text
    assert "_pd.read_feather = _patched_read_feather" in text
    assert "_pd.read_pickle = _patched_read_pickle" in text
    assert "_pd.read_json = _patched_read_json" in text
    assert "_pd.read_excel = _patched_read_excel" in text
    assert "_kb_zip_base_suffix(suffix)" in text
    assert "_kb_open_binary_sample(resolved_path, suffix)" in text
    assert "_kb_read_geopackage_table(resolved_path)" in text
    assert "_kb_read_kml_tabular_frame(resolved_path)" in text
    assert "_kb_read_shapefile_table(resolved_path)" in text
    assert "_kb_read_hdf_table(resolved_path)" in text
    assert "_pd.read_stata(_kb_open_binary_sample(resolved_path, suffix))" in text
    assert (
        "_pd.read_sas(_kb_open_binary_sample(resolved_path, suffix), format=_kb_sas_format_for_suffix(suffix))" in text
    )
    assert "_pd.read_spss(_kb_open_binary_sample(resolved_path, suffix))" in text
    assert "_pd.read_stata = _patched_read_stata" in text
    assert "_pd.read_sas = _patched_read_sas" in text
    assert "_pd.read_spss = _patched_read_spss" in text
    assert "_kb_read_mat_tabular_frame(resolved_path)" in text
    assert "_kb_read_arff_tabular_frame(resolved_path)" in text
    assert "_kb_read_html_tabular_frame(resolved_path, suffix)" in text
    assert "_kb_read_svmlight_tabular_frame(resolved_path, suffix)" in text
    assert "_kb_read_fixed_width_tabular_frame(resolved_path, suffix)" in text
    assert "table.shape[1] > 0" in text
    assert "not table.empty and table.shape[1] > 0" not in text
    assert "return _orig(StringIO(_kb_read_text(resolved_path, suffix)), *args, **kwargs)" in text
    assert "'.csv.gz'" in text
    assert "'.csv.zip'" in text
    assert "'.parquet.zip'" in text
    assert "'.tab'" in text
    assert "'.psv'" in text
    assert "_kb_select_zip_tabular_member" in text
    assert "'.ndjson'" in text
    assert "'.sqlite3'" in text
    assert "'.duckdb'" in text
    assert "'.rds'" in text
    assert "'.pkl.zst'" in text
    assert ".sqlite3" in TABULAR_INPUT_SUFFIXES
    assert ".duckdb" in TABULAR_INPUT_SUFFIXES
    assert ".rds" in TABULAR_INPUT_SUFFIXES
    assert ".pkl.zst" in TABULAR_INPUT_SUFFIXES
    assert "_KB_ASSET_COMPRESSION_SUFFIXES = (" in text
    assert "def _kb_compression_suffix_for" in text
    assert "def _kb_open_compressed_text" in text
    assert "return _kb_open_compressed_text(path, suffix)" in text
    assert "gzip.open(path, 'rt'" in text
    assert "_kb_sniff_sep(resolved_path)" in text
    assert "_kb_resolve_existing_tabular_path(filepath_or_buffer)" in text


def test_pandas_tabular_read_shim_resolves_missing_csv_to_same_stem_non_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.parquet").write_bytes(b"placeholder")
    seen: list[Path] = []

    def fake_read_parquet(path, *args, **kwargs):
        seen.append(Path(path))
        return pd.DataFrame({"id": [1], "feature": [2], "target": [0]})

    monkeypatch.setenv("KAGGLEBOT_LOCAL_KERNEL", "1")
    monkeypatch.setattr(pd, "read_csv", pd.read_csv)
    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    inject_pandas_tabular_read_shim(kernel_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    frame = pd.read_csv(data_dir / "train.csv")

    assert seen == [data_dir / "train.parquet"]
    assert frame.to_dict("records") == [{"id": 1, "feature": 2, "target": 0}]


def test_pandas_tabular_read_shim_resolves_missing_csv_to_zip_wrapped_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(data_dir / "train.csv.zip", "w") as archive:
        archive.writestr("train.csv", "id,feature,target\n1,2,0\n")

    monkeypatch.setenv("KAGGLEBOT_LOCAL_KERNEL", "1")
    monkeypatch.setattr(pd, "read_csv", pd.read_csv)
    inject_pandas_tabular_read_shim(kernel_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    frame = pd.read_csv(data_dir / "train.csv")

    assert frame.to_dict("records") == [{"id": 1, "feature": 2, "target": 0}]


def test_pandas_tabular_read_shim_resolves_missing_csv_to_zip_wrapped_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = io.BytesIO()
    pd.DataFrame({"id": [1], "feature": [2], "target": [0]}).to_parquet(payload, index=False)
    with zipfile.ZipFile(data_dir / "train.parquet.zip", "w") as archive:
        archive.writestr("nested/train.parquet", payload.getvalue())

    monkeypatch.setenv("KAGGLEBOT_LOCAL_KERNEL", "1")
    monkeypatch.setattr(pd, "read_csv", pd.read_csv)
    inject_pandas_tabular_read_shim(kernel_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    frame = pd.read_csv(data_dir / "train.csv")

    assert frame.to_dict("records") == [{"id": 1, "feature": 2, "target": 0}]


@pytest.mark.parametrize(
    ("reader_name", "suffix"),
    [
        ("read_parquet", ".parquet"),
        ("read_feather", ".feather"),
    ],
)
def test_pandas_tabular_read_shim_resolves_native_binary_reader_to_zip_wrapped_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader_name: str,
    suffix: str,
) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"id": [1], "feature": [2], "target": [0]})
    payload = io.BytesIO()
    if suffix == ".parquet":
        frame.to_parquet(payload, index=False)
    else:
        frame.to_feather(payload)
    with zipfile.ZipFile(data_dir / f"train{suffix}.zip", "w") as archive:
        archive.writestr(f"nested/train{suffix}", payload.getvalue())

    monkeypatch.setenv("KAGGLEBOT_LOCAL_KERNEL", "1")
    monkeypatch.setattr(pd, reader_name, getattr(pd, reader_name))
    inject_pandas_tabular_read_shim(kernel_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    loaded = getattr(pd, reader_name)(data_dir / f"train{suffix}")

    assert loaded.to_dict("records") == [{"id": 1, "feature": 2, "target": 0}]


def test_pandas_tabular_read_shim_resolves_native_json_reader_to_zip_wrapped_jsonl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(data_dir / "sample_submission.jsonl.zip", "w") as archive:
        archive.writestr("nested/sample_submission.jsonl", '{"id":1,"target":0.1}\n')

    monkeypatch.setenv("KAGGLEBOT_LOCAL_KERNEL", "1")
    monkeypatch.setattr(pd, "read_json", pd.read_json)
    inject_pandas_tabular_read_shim(kernel_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    loaded = pd.read_json(data_dir / "sample_submission.jsonl")

    assert loaded.to_dict("records") == [{"id": 1, "target": 0.1}]


def test_pandas_tabular_read_shim_resolves_native_pickle_reader_to_zip_wrapped_pickle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = io.BytesIO()
    pd.DataFrame({"id": [1], "feature": [2], "target": [0]}).to_pickle(payload)
    with zipfile.ZipFile(data_dir / "train.pkl.zip", "w") as archive:
        archive.writestr("nested/train.pkl", payload.getvalue())

    monkeypatch.setenv("KAGGLEBOT_LOCAL_KERNEL", "1")
    monkeypatch.setattr(pd, "read_pickle", pd.read_pickle)
    inject_pandas_tabular_read_shim(kernel_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    loaded = pd.read_pickle(data_dir / "train.pkl")

    assert loaded.to_dict("records") == [{"id": 1, "feature": 2, "target": 0}]


def test_pandas_tabular_read_shim_resolves_native_excel_reader_to_zip_wrapped_workbook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = io.BytesIO()
    pd.DataFrame({"id": [1], "feature": [2], "target": [0]}).to_excel(payload, index=False)
    with zipfile.ZipFile(data_dir / "train.xlsx.zip", "w") as archive:
        archive.writestr("nested/train.xlsx", payload.getvalue())

    monkeypatch.setenv("KAGGLEBOT_LOCAL_KERNEL", "1")
    monkeypatch.setattr(pd, "read_excel", pd.read_excel)
    inject_pandas_tabular_read_shim(kernel_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    loaded = pd.read_excel(data_dir / "train.xlsx")

    assert loaded.to_dict("records") == [{"id": 1, "feature": 2, "target": 0}]


def test_pandas_tabular_read_shim_resolves_native_stata_reader_to_zip_wrapped_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = io.BytesIO()
    pd.DataFrame({"id": [1], "feature": [2], "target": [0]}).to_stata(payload, write_index=False)
    with zipfile.ZipFile(data_dir / "train.dta.zip", "w") as archive:
        archive.writestr("nested/train.dta", payload.getvalue())

    monkeypatch.setenv("KAGGLEBOT_LOCAL_KERNEL", "1")
    monkeypatch.setattr(pd, "read_stata", pd.read_stata)
    inject_pandas_tabular_read_shim(kernel_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    loaded = pd.read_stata(data_dir / "train.dta")

    assert loaded.to_dict("records") == [{"id": 1, "feature": 2, "target": 0}]


@pytest.mark.parametrize("suffix", [".orc", ".hdf5"])
def test_pandas_tabular_read_shim_resolves_missing_csv_to_binary_tabular(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    actual = data_dir / f"sample_submission{suffix}"
    frame = pd.DataFrame({"id": [1, 2], "target": [0, 1]})
    if suffix == ".orc":
        frame.to_orc(actual, index=False)
    else:
        frame.to_hdf(actual, key="submission", mode="w", format="table", index=False)

    monkeypatch.setenv("KAGGLEBOT_LOCAL_KERNEL", "1")
    monkeypatch.setattr(pd, "read_csv", pd.read_csv)
    inject_pandas_tabular_read_shim(kernel_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    loaded = pd.read_csv(data_dir / "sample_submission.csv")

    assert loaded.to_dict("list") == {"id": [1, 2], "target": [0, 1]}


def test_pandas_tabular_read_shim_resolves_missing_csv_to_html_tabular(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"id": [1, 2], "target": [0, 1]})
    frame.to_html(data_dir / "sample_submission.html", index=False)

    monkeypatch.setenv("KAGGLEBOT_LOCAL_KERNEL", "1")
    monkeypatch.setattr(pd, "read_csv", pd.read_csv)
    inject_pandas_tabular_read_shim(kernel_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    loaded = pd.read_csv(data_dir / "sample_submission.csv")

    assert loaded.to_dict("list") == {"id": [1, 2], "target": [0, 1]}


def test_pandas_tabular_read_shim_resolves_missing_csv_to_compressed_arff_tabular(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(data_dir / "train.arff.gz", "wt", encoding="utf-8") as handle:
        handle.write(
            """
@RELATION train
@ATTRIBUTE id NUMERIC
@ATTRIBUTE feature NUMERIC
@ATTRIBUTE target {no,yes}
@DATA
1,10,no
2,20,yes
""".strip()
        )

    monkeypatch.setenv("KAGGLEBOT_LOCAL_KERNEL", "1")
    monkeypatch.setattr(pd, "read_csv", pd.read_csv)
    inject_pandas_tabular_read_shim(kernel_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    loaded = pd.read_csv(data_dir / "train.csv")

    assert loaded.to_dict("list") == {"id": [1.0, 2.0], "feature": [10.0, 20.0], "target": ["no", "yes"]}


def test_pandas_tabular_read_shim_reads_same_stem_sqlite_from_csv_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = data_dir / "train.sqlite"
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute("CREATE TABLE train (id INTEGER, feature INTEGER, target INTEGER)")
        conn.executemany("INSERT INTO train VALUES (?, ?, ?)", [(1, 10, 0), (2, 20, 1)])

    monkeypatch.setenv("KAGGLEBOT_LOCAL_KERNEL", "1")
    monkeypatch.setattr(pd, "read_csv", pd.read_csv)
    inject_pandas_tabular_read_shim(kernel_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    frame = pd.read_csv(data_dir / "train.csv")

    assert frame.to_dict("records") == [
        {"id": 1, "feature": 10, "target": 0},
        {"id": 2, "feature": 20, "target": 1},
    ]


def test_pandas_tabular_read_shim_reads_same_stem_duckdb_from_csv_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "train.duckdb"
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE train (id INTEGER, feature INTEGER, target INTEGER)")
        conn.execute("INSERT INTO train VALUES (1, 10, 0), (2, 20, 1)")
    finally:
        conn.close()

    monkeypatch.setenv("KAGGLEBOT_LOCAL_KERNEL", "1")
    monkeypatch.setattr(pd, "read_csv", pd.read_csv)
    inject_pandas_tabular_read_shim(kernel_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    frame = pd.read_csv(data_dir / "train.csv")

    assert frame.to_dict("records") == [
        {"id": 1, "feature": 10, "target": 0},
        {"id": 2, "feature": 20, "target": 1},
    ]


def test_pandas_tabular_read_shim_reads_same_stem_rds_from_csv_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pyreadr.write_rds(data_dir / "train.rds", pd.DataFrame({"id": [1, 2], "feature": [10, 20], "target": [0, 1]}))

    monkeypatch.setenv("KAGGLEBOT_LOCAL_KERNEL", "1")
    monkeypatch.setattr(pd, "read_csv", pd.read_csv)
    inject_pandas_tabular_read_shim(kernel_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    frame = pd.read_csv(data_dir / "train.csv")

    assert frame.to_dict("records") == [
        {"id": 1, "feature": 10, "target": 0},
        {"id": 2, "feature": 20, "target": 1},
    ]


def test_pandas_tabular_read_shim_reads_compressed_tsv_from_csv_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(data_dir / "train.tsv.gz", "wt", encoding="utf-8") as handle:
        handle.write("id\tfeature\ttarget\n1\t10\t0\n2\t20\t1\n")

    monkeypatch.setenv("KAGGLEBOT_LOCAL_KERNEL", "1")
    monkeypatch.setattr(pd, "read_csv", pd.read_csv)
    inject_pandas_tabular_read_shim(kernel_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    frame = pd.read_csv(data_dir / "train.csv")

    assert frame.to_dict("records") == [
        {"id": 1, "feature": 10, "target": 0},
        {"id": 2, "feature": 20, "target": 1},
    ]


def test_pandas_tabular_read_shim_reads_zstd_jsonl_from_csv_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = b'{"id":1,"target":0.1}\n{"id":2,"target":0.2}\n'
    (data_dir / "sample_submission.jsonl.zst").write_bytes(zstd.ZstdCompressor().compress(payload))

    monkeypatch.setenv("KAGGLEBOT_LOCAL_KERNEL", "1")
    monkeypatch.setattr(pd, "read_csv", pd.read_csv)
    inject_pandas_tabular_read_shim(kernel_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    frame = pd.read_csv(data_dir / "sample_submission.csv")

    assert frame.to_dict("records") == [
        {"id": 1, "target": 0.1},
        {"id": 2, "target": 0.2},
    ]


def test_pandas_tabular_read_shim_reads_compressed_yaml_from_csv_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    yaml = pytest.importorskip("yaml")
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(
        [{"id": 1, "target": 0.1}, {"id": 2, "target": 0.2}],
        sort_keys=False,
    )
    with lzma.open(data_dir / "sample_submission.yaml.xz", "wt", encoding="utf-8") as handle:
        handle.write(payload)

    monkeypatch.setenv("KAGGLEBOT_LOCAL_KERNEL", "1")
    monkeypatch.setattr(pd, "read_csv", pd.read_csv)
    inject_pandas_tabular_read_shim(kernel_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    frame = pd.read_csv(data_dir / "sample_submission.csv")

    assert frame.to_dict("records") == [
        {"id": 1, "target": 0.1},
        {"id": 2, "target": 0.2},
    ]


def test_pandas_tabular_read_shim_resolves_missing_csv_to_fixed_width_tabular(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.fwf").write_text("id feature target\n1  10      0\n2  20      1\n", encoding="utf-8")

    monkeypatch.setenv("KAGGLEBOT_LOCAL_KERNEL", "1")
    monkeypatch.setattr(pd, "read_csv", pd.read_csv)
    inject_pandas_tabular_read_shim(kernel_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    frame = pd.read_csv(data_dir / "train.csv")

    assert frame.to_dict("records") == [
        {"id": 1, "feature": 10, "target": 0},
        {"id": 2, "feature": 20, "target": 1},
    ]


def test_pandas_tabular_read_shim_resolves_missing_csv_to_svmlight_tabular(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.svmlight").write_text("1 1:0.5 2:1.5\n0 1:2.0 2:0.0\n", encoding="utf-8")

    monkeypatch.setenv("KAGGLEBOT_LOCAL_KERNEL", "1")
    monkeypatch.setattr(pd, "read_csv", pd.read_csv)
    inject_pandas_tabular_read_shim(kernel_dir)
    runpy.run_path(str(kernel_dir / "sitecustomize.py"))

    frame = pd.read_csv(data_dir / "train.csv")
    dense = frame.copy()
    for column in dense.columns:
        if isinstance(dense[column].dtype, pd.SparseDtype):
            dense[column] = dense[column].sparse.to_dense()

    assert dense.to_dict("records") == [
        {"target": 1.0, "feature_1": 0.5, "feature_2": 1.5},
        {"target": 0.0, "feature_1": 2.0, "feature_2": 0.0},
    ]


def test_inject_object_coerce_shim(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text("print('ok')\n", encoding="utf-8")
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    payload = {"enabled": True}
    (context_dir / "object_coerce.json").write_text(json.dumps(payload), encoding="utf-8")

    inject_object_coerce_shim(kernel_dir, context_dir)

    site_path = kernel_dir / "sitecustomize.py"
    assert site_path.exists()
    text = site_path.read_text(encoding="utf-8")
    assert "object-coerce-shim" in text
    assert "object_coerce.json" in text
    assert (kernel_dir / "object_coerce.json").exists()


def test_inject_device_coerce_shim(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text("print('ok')\n", encoding="utf-8")
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    payload = {"enabled": True}
    (context_dir / "device_coerce.json").write_text(json.dumps(payload), encoding="utf-8")

    inject_device_coerce_shim(kernel_dir, context_dir)

    site_path = kernel_dir / "sitecustomize.py"
    assert site_path.exists()
    text = site_path.read_text(encoding="utf-8")
    assert "device-coerce-shim" in text
    assert "device_coerce.json" in text
    assert (kernel_dir / "device_coerce.json").exists()


def test_inject_context_io_shims_groups_context_driven_shims(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text("print('ok')\n", encoding="utf-8")
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "column_map.json").write_text(json.dumps({"mapping": {"old": "new"}}), encoding="utf-8")
    (context_dir / "column_fill.json").write_text(json.dumps({"files": {"test.csv": ["A"]}}), encoding="utf-8")
    (context_dir / "object_coerce.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")
    (context_dir / "device_coerce.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")

    inject_context_io_shims(kernel_dir, context_dir)
    inject_context_io_shims(kernel_dir, context_dir)

    text = (kernel_dir / "sitecustomize.py").read_text(encoding="utf-8")
    for marker in ("column-map-shim", "column-fill-shim", "object-coerce-shim", "device-coerce-shim"):
        assert marker in text
        assert text.count(marker) == 1


def test_inject_local_runtime_shims(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    inject_local_runtime_shims(kernel_dir)
    inject_local_runtime_shims(kernel_dir)

    site_path = kernel_dir / "sitecustomize.py"
    assert site_path.exists()
    text = site_path.read_text(encoding="utf-8")
    assert "kaggle-working-redirect-shim" in text
    assert text.count("kaggle-working-redirect-shim") == 1
    assert "lgbm-gpu-guard-shim" in text
    assert text.count("lgbm-gpu-guard-shim") == 1
    assert "torch-runtime-guard-shim" in text
    assert text.count("torch-runtime-guard-shim") == 1


def test_inject_transformers_eval_strategy_shim(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    inject_transformers_eval_strategy_shim(kernel_dir)

    site_path = kernel_dir / "sitecustomize.py"
    assert site_path.exists()
    text = site_path.read_text(encoding="utf-8")
    assert "transformers-eval-strategy-shim" in text
    assert "evaluation_strategy" in text
    assert "eval_strategy" in text
    assert "Seq2SeqTrainingArguments" in text


def test_inject_training_progress_shim(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)

    inject_training_progress_shim(kernel_dir)
    inject_training_progress_shim(kernel_dir)

    site_path = kernel_dir / "sitecustomize.py"
    assert site_path.exists()
    text = site_path.read_text(encoding="utf-8")
    assert text.count("kagglebot: train-progress-shim") == 1
    assert "train watchdog" in text
    assert "cv fold start:" in text
    assert "train start:" in text
    assert "train done:" in text


def test_inject_training_compat_shims_groups_training_shims(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    inject_training_compat_shims(kernel_dir)
    inject_training_compat_shims(kernel_dir)

    text = (kernel_dir / "sitecustomize.py").read_text(encoding="utf-8")
    assert "kagglebot: train-progress-shim" in text
    assert text.count("kagglebot: train-progress-shim") == 1
    assert "transformers-eval-strategy-shim" in text
    assert text.count("transformers-eval-strategy-shim") == 1


def test_kernel_bootstrap_preserves_future_import(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python",
                "# -*- coding: utf-8 -*-",
                '"""docstring"""',
                "from __future__ import annotations",
                "",
                "print('ok')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    from kagglebot.kernel_bootstrap import ensure_kernel_import_path

    ensure_kernel_import_path(kernel_dir)
    lines = kernel_path.read_text(encoding="utf-8").splitlines()
    future_idx = next(i for i, line in enumerate(lines) if "from __future__ import annotations" in line)
    marker_idx = next(i for i, line in enumerate(lines) if "kagglebot:kernel_sys_path" in line)
    assert marker_idx > future_idx


def test_run_kernel_uses_custom_kernel(tmp_path: Path) -> None:
    custom_kernel = tmp_path / "demo" / "kernel" / "kernel.py"
    custom_kernel.parent.mkdir(parents=True, exist_ok=True)
    custom_kernel.write_text(
        "\n".join(
            [
                "DATA = '/kaggle/input/demo/train.csv'",
                "OUT1 = '/kaggle/working/submission.csv'",
                "OUT2 = '/kaggle/working/metrics.json'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_kernel(
        slug="demo",
        run_id="run-3",
        iteration=1,
        base_dir=tmp_path,
        kaggle_username="user",
        kernel_name=None,
        accelerator="gpu",
        enable_internet=False,
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        seed=42,
        dry_run=True,
        timeout_minutes=None,
    )
    kernel_path = tmp_path / "demo" / "kernels" / "run-3" / "kernel.py"
    content = kernel_path.read_text(encoding="utf-8")
    assert "/kaggle/input/" in content
    assert "submission.csv" in content
    assert "metrics.json" in content
    sitecustomize_path = tmp_path / "demo" / "kernels" / "run-3" / "sitecustomize.py"
    assert sitecustomize_path.exists()
    sitecustomize = sitecustomize_path.read_text(encoding="utf-8")
    assert "kagglebot: train-progress-shim" in sitecustomize
    assert "train watchdog" in sitecustomize
    assert "cv fold start:" in sitecustomize
    assert "train start:" in sitecustomize
    assert "train done:" in sitecustomize
    assert "log_evaluation(period=log_every)" in sitecustomize


def test_run_kernel_requires_authoritative_kernel(tmp_path: Path) -> None:
    with pytest.raises(KernelFailedError, match="Authoritative kernel entrypoint is missing"):
        run_kernel(
            slug="demo",
            run_id="run-4",
            iteration=1,
            base_dir=tmp_path,
            kaggle_username="user",
            kernel_name=None,
            accelerator="gpu",
            enable_internet=False,
            score_source="holdout",
            metric="rmse",
            direction="minimize",
            holdout_frac=0.2,
            cv_folds=3,
            seed=42,
            dry_run=True,
            timeout_minutes=None,
        )


def test_run_kernel_local_executes_staged_copy(tmp_path: Path) -> None:
    source_kernel_dir = tmp_path / "demo" / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    source_kernel_path = source_kernel_dir / "kernel.py"
    source_text = "\n".join(
        [
            "from pathlib import Path",
            "",
            "Path('submission.csv').write_text('id,target\\n1,0.1\\n', encoding='utf-8')",
            "Path('metrics.json').write_text('{\"metric\":\"rmse\",\"offline_value\":0.1}', encoding='utf-8')",
        ]
    )
    source_kernel_path.write_text(source_text + "\n", encoding="utf-8")
    plan_path = tmp_path / "demo" / "plan.json"
    plan_path.write_text(json.dumps({"toggles": {"USE_MODEL": True}}, indent=2), encoding="utf-8")

    result = run_kernel_local(
        slug="demo",
        run_id="run-5",
        iteration=1,
        base_dir=tmp_path,
        accelerator="gpu",
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        seed=42,
        dry_run=False,
        timeout_minutes=1,
        strict_accelerator=False,
    )

    assert result.submission_path is not None and result.submission_path.exists()
    assert result.metrics_path is not None and result.metrics_path.exists()
    assert source_kernel_path.read_text(encoding="utf-8") == source_text + "\n"
    staged_kernel = tmp_path / "demo" / "kernels" / "run-5" / "local-iter-1" / "kernel.py"
    assert staged_kernel.exists()
    staged_text = staged_kernel.read_text(encoding="utf-8")
    assert "# kagglebot:competition_slug" in staged_text
    assert "_kb_os.environ['KAGGLEBOT_COMPETITION_SLUG'] = \"demo\"" in staged_text
    assert "_kb_os.environ['KAGGLEBOT_SLUG'] = \"demo\"" in staged_text
    assert "# kagglebot:force_train" in staged_text
    assert "_kb_os.environ['KAGGLEBOT_DO_TRAIN'] = '1'" in staged_text
    assert "_kb_os.environ['KAGGLEBOT_FORCE_TRAIN'] = '1'" in staged_text
    assert "demo" in staged_text
    staged_sitecustomize = tmp_path / "demo" / "kernels" / "run-5" / "local-iter-1" / "sitecustomize.py"
    assert staged_sitecustomize.exists()
    staged_sitecustomize_text = staged_sitecustomize.read_text(encoding="utf-8")
    assert "kagglebot: train-progress-shim" in staged_sitecustomize_text
    assert "kagglebot: torch-runtime-guard-shim" in staged_sitecustomize_text
    staged_plan_local = tmp_path / "demo" / "kernels" / "run-5" / "local-iter-1" / "plan.json"
    staged_plan_parent = tmp_path / "demo" / "kernels" / "run-5" / "plan.json"
    assert staged_plan_local.exists()
    assert staged_plan_parent.exists()
    assert json.loads(staged_plan_local.read_text(encoding="utf-8")) == {"toggles": {"USE_MODEL": True}}
    assert json.loads(staged_plan_parent.read_text(encoding="utf-8")) == {"toggles": {"USE_MODEL": True}}


def test_run_kernel_local_fails_fast_when_local_kernel_stalls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_kernel_dir = tmp_path / "demo" / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    (source_kernel_dir / "kernel.py").write_text(
        "\n".join(
            [
                "import time",
                "print('submission.csv', flush=True)",
                "print('metrics.json', flush=True)",
                "print('kernel start', flush=True)",
                "time.sleep(10)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "demo" / "plan.json").write_text(
        json.dumps({"toggles": {"USE_MODEL": True}}, indent=2), encoding="utf-8"
    )
    monkeypatch.setenv("KAGGLEBOT_LOCAL_KERNEL_STALL_SEC", "5")

    with pytest.raises(KernelFailedError, match="Local kernel stalled"):
        run_kernel_local(
            slug="demo",
            run_id="run-stall",
            iteration=1,
            base_dir=tmp_path,
            accelerator="gpu",
            score_source="holdout",
            metric="rmse",
            direction="minimize",
            holdout_frac=0.2,
            cv_folds=3,
            seed=42,
            dry_run=False,
            timeout_minutes=1,
            strict_accelerator=False,
        )

    stdout_log = tmp_path / "demo" / "runs" / "run-stall" / "iter-1" / "logs" / "local_kernel_stdout.log"
    assert stdout_log.exists()
    assert "kernel start" in stdout_log.read_text(encoding="utf-8")


def test_run_kernel_local_ignores_stale_output_artifacts_for_stall_watchdog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_kernel_dir = tmp_path / "demo" / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    (source_kernel_dir / "kernel.py").write_text(
        "\n".join(
            [
                "import time",
                "from pathlib import Path",
                "time.sleep(0.5)",
                "print('kernel start', flush=True)",
                "out = Path('outputs')",
                "out.mkdir(exist_ok=True)",
                "out.joinpath('submission.csv').write_text('id,target\\n1,0.1\\n', encoding='utf-8')",
                'metrics = \'{"metric":"rmse","offline_value":0.1}\'',
                "out.joinpath('metrics.json').write_text(metrics, encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "demo" / "plan.json").write_text(
        json.dumps({"toggles": {"USE_MODEL": True}}, indent=2), encoding="utf-8"
    )
    stale_output_dir = tmp_path / "demo" / "runs" / "run-stale" / "iter-1" / "output"
    stale_output_dir.mkdir(parents=True, exist_ok=True)
    stale_files = [
        stale_output_dir / "submission.csv",
        stale_output_dir / "metrics.json",
    ]
    stale_files[0].write_text("id,target\n1,0.0\n", encoding="utf-8")
    stale_files[1].write_text('{"metric":"rmse","offline_value":9.9}\n', encoding="utf-8")
    stale_mtime = time.time() - 60.0
    for stale_file in stale_files:
        os.utime(stale_file, (stale_mtime, stale_mtime))
    monkeypatch.setenv("KAGGLEBOT_LOCAL_KERNEL_STALL_SEC", "5")

    result = run_kernel_local(
        slug="demo",
        run_id="run-stale",
        iteration=1,
        base_dir=tmp_path,
        accelerator="gpu",
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        seed=42,
        dry_run=False,
        timeout_minutes=1,
        strict_accelerator=False,
    )

    assert result.submission_path is not None and result.submission_path.exists()
    stdout_log = tmp_path / "demo" / "runs" / "run-stale" / "iter-1" / "logs" / "local_kernel_stdout.log"
    assert "kernel start" in stdout_log.read_text(encoding="utf-8")


def test_run_kernel_local_does_not_reuse_stale_output_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_kernel_dir = tmp_path / "demo" / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    (source_kernel_dir / "kernel.py").write_text(
        "\n".join(
            [
                "print('kernel start', flush=True)",
                "# submission.csv",
                "# metrics.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "demo" / "plan.json").write_text(
        json.dumps({"toggles": {"USE_MODEL": True}}, indent=2),
        encoding="utf-8",
    )
    stale_output_dir = tmp_path / "demo" / "runs" / "run-stale" / "iter-1" / "output"
    stale_output_dir.mkdir(parents=True, exist_ok=True)
    stale_submission = stale_output_dir / "submission.csv"
    stale_metrics = stale_output_dir / "metrics.json"
    stale_submission.write_text("id,target\n1,0.0\n", encoding="utf-8")
    stale_metrics.write_text('{"metric":"rmse","offline_value":9.9}\n', encoding="utf-8")
    stale_mtime = time.time() - 60.0
    os.utime(stale_submission, (stale_mtime, stale_mtime))
    os.utime(stale_metrics, (stale_mtime, stale_mtime))
    monkeypatch.setenv("KAGGLEBOT_LOCAL_KERNEL_STALL_SEC", "5")

    with pytest.raises(KernelFailedError, match="submission output was not found"):
        run_kernel_local(
            slug="demo",
            run_id="run-stale",
            iteration=1,
            base_dir=tmp_path,
            accelerator="gpu",
            score_source="holdout",
            metric="rmse",
            direction="minimize",
            holdout_frac=0.2,
            cv_folds=3,
            seed=42,
            dry_run=False,
            timeout_minutes=1,
            strict_accelerator=False,
        )

    stdout_log = tmp_path / "demo" / "runs" / "run-stale" / "iter-1" / "logs" / "local_kernel_stdout.log"
    assert "kernel start" in stdout_log.read_text(encoding="utf-8")


def test_run_kernel_local_dry_run_stages_required_seq2seq_models(tmp_path: Path) -> None:
    slug = "deep-past-initiative-machine-translation"
    source_kernel_dir = tmp_path / slug / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    (source_kernel_dir / "kernel.py").write_text(
        "\n".join(
            [
                "print('/kaggle/input/demo/train.csv')",
                "print('submission.csv')",
                "print('metrics.json')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    plan_payload = {
        "toggles": {"USE_MODEL": True},
        "kaggle_kernel_sources": {
            "pipeline_model_hints": {
                "pooled_multi_byt5_mbr": ["google/byt5-base"],
            },
            "required_local_seq2seq_pipelines": ["pooled_multi_byt5_mbr"],
        },
    }
    (tmp_path / slug / "plan.json").write_text(json.dumps(plan_payload, indent=2), encoding="utf-8")
    model_dir = tmp_path / slug / "kernels" / "old-run" / "local-iter-1" / "models" / "google--byt5-base"
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("config.json", "tokenizer_config.json", "pytorch_model.bin"):
        (model_dir / filename).write_text("x", encoding="utf-8")

    result = run_kernel_local(
        slug=slug,
        run_id="run-local-models",
        iteration=1,
        base_dir=tmp_path,
        accelerator="gpu",
        score_source="cv",
        metric="gmean",
        direction="maximize",
        holdout_frac=0.2,
        cv_folds=5,
        seed=42,
        dry_run=True,
        timeout_minutes=1,
        strict_accelerator=False,
    )

    assert result.kernel_id == f"local/{slug}"
    staged_model_dir = tmp_path / slug / "kernels" / "run-local-models" / "local-iter-1" / "models" / "google_byt5_base"
    assert staged_model_dir.exists()
    assert (staged_model_dir / "config.json").exists()


def test_stage_resolved_model_hints_rejects_artem_alias_pointing_to_google_large(tmp_path: Path) -> None:
    from kagglebot.local_kernel_models import stage_resolved_model_hints

    google_dir = tmp_path / "models--google--byt5-large" / "snapshots" / "abc123"
    google_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("config.json", "tokenizer_config.json", "model.safetensors"):
        (google_dir / filename).write_text("x", encoding="utf-8")
    alias_dir = tmp_path / "artemgoncarov_dpc_byt5_large"
    alias_dir.symlink_to(google_dir, target_is_directory=True)

    staged = stage_resolved_model_hints(
        hints=["artemgoncarov/dpc-byt5-large"],
        candidate_dirs=[alias_dir, google_dir],
        staged_root=tmp_path / "staged-models",
    )

    assert staged == []


def test_stage_resolved_model_hints_rejects_mattia_alias_pointing_to_assiaben(tmp_path: Path) -> None:
    from kagglebot.local_kernel_models import stage_resolved_model_hints

    assiaben_dir = tmp_path / "dataset__assiaben__final-byt5" / "byt5-akkadian-optimized-34x"
    assiaben_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("config.json", "tokenizer_config.json", "model.safetensors"):
        (assiaben_dir / filename).write_text("x", encoding="utf-8")
    alias_dir = tmp_path / "mattiaangeli_byt5_akkadian_mbr_pytorch_default_6"
    alias_dir.symlink_to(assiaben_dir, target_is_directory=True)

    staged = stage_resolved_model_hints(
        hints=["mattiaangeli/byt5-akkadian-mbr/PyTorch/default/6"],
        candidate_dirs=[alias_dir, assiaben_dir],
        staged_root=tmp_path / "staged-models",
    )

    assert staged == []


def test_run_kernel_local_dry_run_fails_when_required_seq2seq_models_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slug = "deep-past-initiative-machine-translation"
    source_kernel_dir = tmp_path / slug / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    (source_kernel_dir / "kernel.py").write_text(
        "\n".join(
            [
                "print('/kaggle/input/demo/train.csv')",
                "print('submission.csv')",
                "print('metrics.json')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    plan_payload = {
        "toggles": {"USE_MODEL": True},
        "kaggle_kernel_sources": {
            "pipeline_model_hints": {
                "pooled_multi_byt5_mbr": ["mattiaangeli/byt5-akkadian-mbr/PyTorch/default/6"],
            },
            "required_local_seq2seq_pipelines": ["pooled_multi_byt5_mbr"],
        },
    }
    (tmp_path / slug / "plan.json").write_text(json.dumps(plan_payload, indent=2), encoding="utf-8")

    home = tmp_path / "fake-home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))

    with pytest.raises(KernelFailedError, match="Required local seq2seq model sources could not be resolved"):
        run_kernel_local(
            slug=slug,
            run_id="run-missing-models",
            iteration=1,
            base_dir=tmp_path,
            accelerator="gpu",
            score_source="cv",
            metric="gmean",
            direction="maximize",
            holdout_frac=0.2,
            cv_folds=5,
            seed=42,
            dry_run=True,
            timeout_minutes=1,
            strict_accelerator=False,
        )


def test_run_kernel_local_dry_run_stages_text_runtime_aux_inputs(tmp_path: Path) -> None:
    slug = "demo-translation"
    source_kernel_dir = tmp_path / slug / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    (source_kernel_dir / "kernel.py").write_text("print('submission.csv')\nprint('metrics.json')\n", encoding="utf-8")
    (tmp_path / slug / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / slug / "context").mkdir(parents=True, exist_ok=True)
    (tmp_path / slug / "data" / "lexicon.csv").write_text("token,norm\n", encoding="utf-8")
    (tmp_path / slug / "context" / "metadata.csv").write_text("id,value\n", encoding="utf-8")
    plan_payload = {
        "text_runtime": {
            "required_aux_inputs": ["data/lexicon.csv", "context/metadata.csv"],
            "metadata_supervision": "high_precision",
            "constraint_rewrite_mode": "soft",
            "group_key_columns": ["document_id"],
        }
    }
    (tmp_path / slug / "plan.json").write_text(json.dumps(plan_payload, indent=2), encoding="utf-8")

    result = run_kernel_local(
        slug=slug,
        run_id="run-text-runtime",
        iteration=1,
        base_dir=tmp_path,
        accelerator="gpu",
        score_source="cv",
        metric="gmean",
        direction="maximize",
        holdout_frac=0.2,
        cv_folds=5,
        seed=42,
        dry_run=True,
        timeout_minutes=1,
        strict_accelerator=False,
    )

    assert result.kernel_id == f"local/{slug}"
    staged_kernel_dir = tmp_path / slug / "kernels" / "run-text-runtime" / "local-iter-1"
    assert (staged_kernel_dir / "text_translation.py").exists()
    assert (staged_kernel_dir / "aux_inputs" / "data" / "lexicon.csv").exists()
    assert (staged_kernel_dir / "aux_inputs" / "context" / "metadata.csv").exists()


def test_run_kernel_local_dry_run_fails_when_required_text_aux_inputs_missing(tmp_path: Path) -> None:
    slug = "demo-translation"
    source_kernel_dir = tmp_path / slug / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    (source_kernel_dir / "kernel.py").write_text("print('submission.csv')\nprint('metrics.json')\n", encoding="utf-8")
    (tmp_path / slug / "plan.json").write_text(
        json.dumps({"text_runtime": {"required_aux_inputs": ["data/missing_lexicon.csv"]}}, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(KernelFailedError, match="Required text runtime aux inputs could not be resolved"):
        run_kernel_local(
            slug=slug,
            run_id="run-missing-aux",
            iteration=1,
            base_dir=tmp_path,
            accelerator="gpu",
            score_source="cv",
            metric="gmean",
            direction="maximize",
            holdout_frac=0.2,
            cv_folds=5,
            seed=42,
            dry_run=True,
            timeout_minutes=1,
            strict_accelerator=False,
        )


def test_run_kernel_local_enforces_bvs_contract_rejects_regressed_kernel(tmp_path: Path) -> None:
    slug = "beyond-visible-spectrum-ai-for-agriculture-2026p2"
    source_kernel_dir = tmp_path / slug / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    source_kernel_path = source_kernel_dir / "kernel.py"
    source_kernel_path.write_text(
        "\n".join(
            [
                "import json",
                "from pathlib import Path",
                "",
                "print('tri_branch_convnext_spectral cfg: load_size=64 crop_size=64')",
                "Path('submission.csv').write_text('Id,Category\\nval_1.tif,Health\\n', encoding='utf-8')",
                "Path('metrics.json').write_text(",
                "    json.dumps({",
                "        'model_name': 'resnet50',",
                "        'chosen_pipeline': 'tri_branch_convnext_spectral',",
                "        'pipelines': [",
                "            {'name': 'tri_branch_convnext_spectral', 'score': 0.68},",
                "        ],",
                "    }),",
                "    encoding='utf-8',",
                ")",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(KernelFailedError, match="BVS kernel contract failed"):
        run_kernel_local(
            slug=slug,
            run_id="run-bvs-reject",
            iteration=1,
            base_dir=tmp_path,
            accelerator="gpu",
            score_source="cv",
            metric="accuracy",
            direction="maximize",
            holdout_frac=0.2,
            cv_folds=5,
            seed=42,
            dry_run=False,
            timeout_minutes=1,
            strict_accelerator=False,
        )


def test_run_kernel_local_enforces_policy_kernel_contract_for_generic_slug(tmp_path: Path) -> None:
    slug = "demo-vision-contract"
    source_kernel_dir = tmp_path / slug / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    source_kernel_path = source_kernel_dir / "kernel.py"
    source_kernel_path.write_text(
        "\n".join(
            [
                "import json",
                "from pathlib import Path",
                "",
                "print('tri_branch_convnext_spectral cfg: load_size=64 crop_size=64')",
                "Path('submission.csv').write_text('Id,Category\\nval_1.tif,Health\\n', encoding='utf-8')",
                "Path('metrics.json').write_text(",
                "    json.dumps({",
                "        'model_name': 'resnet50',",
                "        'chosen_pipeline': 'tri_branch_convnext_spectral',",
                "        'pipelines': [{'name': 'tri_branch_convnext_spectral', 'score': 0.68}],",
                "    }),",
                "    encoding='utf-8',",
                ")",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    policy_path = tmp_path / slug / "context" / "competition_policy.json"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(
        json.dumps({"execution_hints": {"kernel_contract": "bvs"}}),
        encoding="utf-8",
    )

    with pytest.raises(KernelFailedError, match="BVS kernel contract failed"):
        run_kernel_local(
            slug=slug,
            run_id="run-policy-contract-reject",
            iteration=1,
            base_dir=tmp_path,
            accelerator="gpu",
            score_source="cv",
            metric="accuracy",
            direction="maximize",
            holdout_frac=0.2,
            cv_folds=5,
            seed=42,
            dry_run=False,
            timeout_minutes=1,
            strict_accelerator=False,
        )


def test_run_kernel_local_enforces_bvs_contract_allows_ensemble_kernel(tmp_path: Path) -> None:
    slug = "beyond-visible-spectrum-ai-for-agriculture-2026p2"
    source_kernel_dir = tmp_path / slug / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    source_kernel_path = source_kernel_dir / "kernel.py"
    source_kernel_path.write_text(
        "\n".join(
            [
                "import json",
                "from pathlib import Path",
                "",
                "print('tri_branch_timm_gated cfg: load_size=224 crop_size=64')",
                "Path('submission.csv').write_text('Id,Category\\nval_1.tif,Health\\n', encoding='utf-8')",
                "Path('metrics.json').write_text(",
                "    json.dumps({",
                "        'model_name': 'convnext_tiny',",
                "        'chosen_pipeline': 'ensemble_tri_branch__tabular',",
                "        'pipelines': [",
                "            {'name': 'tri_branch_timm_gated', 'score': 0.70},",
                "            {'name': 'tabular_fallback', 'score': 0.66},",
                "            {'name': 'ensemble_tri_branch__tabular', 'score': 0.72},",
                "        ],",
                "    }),",
                "    encoding='utf-8',",
                ")",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_kernel_local(
        slug=slug,
        run_id="run-bvs-allow",
        iteration=1,
        base_dir=tmp_path,
        accelerator="gpu",
        score_source="cv",
        metric="accuracy",
        direction="maximize",
        holdout_frac=0.2,
        cv_folds=5,
        seed=42,
        dry_run=False,
        timeout_minutes=1,
        strict_accelerator=False,
    )

    assert result.submission_path is not None and result.submission_path.exists()
    assert result.metrics_path is not None and result.metrics_path.exists()


def test_run_kernel_local_applies_zero_overlap_drift_drop_shim(tmp_path: Path) -> None:
    source_kernel_dir = tmp_path / "demo" / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    source_kernel_path = source_kernel_dir / "kernel.py"
    source_kernel_path.write_text(
        "\n".join(
            [
                "import json",
                "from pathlib import Path",
                "import pandas as pd",
                "",
                "data_root = Path(__file__).resolve().parents[3] / 'data'",
                "train_df = pd.read_csv(data_root / 'train.csv')",
                "test_df = pd.read_csv(data_root / 'test.csv')",
                "dropped = float('risk_cat' not in train_df.columns and 'risk_cat' not in test_df.columns)",
                "Path('submission.csv').write_text('id,target\\n1,0.1\\n', encoding='utf-8')",
                "Path('metrics.json').write_text(",
                "    json.dumps({'metric': 'auc', 'offline_value': dropped}),",
                "    encoding='utf-8',",
                ")",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    plan_path = tmp_path / "demo" / "plan.json"
    plan_path.write_text(json.dumps({"toggles": {"USE_MODEL": True}}, indent=2), encoding="utf-8")
    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text(
        "\n".join(
            [
                "id,risk_cat,target",
                "A,x,1",
                "B,x,1",
                "C,x,1",
                "D,y,0",
                "E,y,0",
                "F,y,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text(
        "\n".join(
            [
                "id,risk_cat",
                "T1,u",
                "T2,u",
                "T3,v",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0.0\n2,0.0\n3,0.0\n", encoding="utf-8")
    context_dir = tmp_path / "demo" / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "dataset_profile.json").write_text(
        json.dumps({"target_column": "target", "id_column": "id"}, indent=2),
        encoding="utf-8",
    )

    result = run_kernel_local(
        slug="demo",
        run_id="run-zod",
        iteration=1,
        base_dir=tmp_path,
        accelerator="gpu",
        score_source="cv",
        metric="auc",
        direction="maximize",
        holdout_frac=0.2,
        cv_folds=5,
        seed=42,
        dry_run=False,
        timeout_minutes=1,
        strict_accelerator=False,
    )

    assert result.metrics_path is not None
    metrics_payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload.get("offline_value") == 1.0
    staged_sitecustomize = tmp_path / "demo" / "kernels" / "run-zod" / "local-iter-1" / "sitecustomize.py"
    assert staged_sitecustomize.exists()
    assert "zero-overlap-drift-shim" in staged_sitecustomize.read_text(encoding="utf-8")


def test_run_kernel_local_reads_non_csv_data_through_csv_references(tmp_path: Path) -> None:
    source_kernel_dir = tmp_path / "demo" / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    source_kernel_path = source_kernel_dir / "kernel.py"
    source_kernel_path.write_text(
        "\n".join(
            [
                "import json",
                "from pathlib import Path",
                "import pandas as pd",
                "",
                "data_root = Path(__file__).resolve().parents[3] / 'data'",
                "train_df = pd.read_csv(data_root / 'train.csv')",
                "test_df = pd.read_csv(data_root / 'test.csv')",
                "ok = float(",
                "    list(train_df.columns) == ['id', 'feature', 'target']",
                "    and list(test_df.columns) == ['id', 'feature']",
                "    and len(train_df) == 2",
                "    and len(test_df) == 2",
                ")",
                "Path('submission.csv').write_text('id,target\\n3,0.1\\n4,0.2\\n', encoding='utf-8')",
                "Path('metrics.json').write_text(",
                "    json.dumps({'metric': 'auc', 'offline_value': ok}),",
                "    encoding='utf-8',",
                ")",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    plan_path = tmp_path / "demo" / "plan.json"
    plan_path.write_text(json.dumps({"toggles": {"USE_MODEL": True}}, indent=2), encoding="utf-8")
    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.jsonl").write_text(
        "\n".join(
            [
                '{"id":1,"feature":10,"target":0}',
                '{"id":2,"feature":20,"target":1}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "test.jsonl").write_text(
        "\n".join(
            [
                '{"id":3,"feature":30}',
                '{"id":4,"feature":40}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text("id,target\n3,0.0\n4,0.0\n", encoding="utf-8")

    result = run_kernel_local(
        slug="demo",
        run_id="run-jsonl-csv-ref",
        iteration=1,
        base_dir=tmp_path,
        accelerator="gpu",
        score_source="cv",
        metric="auc",
        direction="maximize",
        holdout_frac=0.2,
        cv_folds=5,
        seed=42,
        dry_run=False,
        timeout_minutes=1,
        strict_accelerator=False,
    )

    assert result.metrics_path is not None
    metrics_payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload.get("offline_value") == 1.0
    staged_kernel = tmp_path / "demo" / "kernels" / "run-jsonl-csv-ref" / "local-iter-1" / "kernel.py"
    staged_sitecustomize = tmp_path / "demo" / "kernels" / "run-jsonl-csv-ref" / "local-iter-1" / "sitecustomize.py"
    assert "_kb_find_file(data_root, 'train.csv')" in staged_kernel.read_text(encoding="utf-8")
    assert "pandas-tabular-read-shim" in staged_sitecustomize.read_text(encoding="utf-8")


def test_run_kernel_local_copies_optional_oof_artifacts(tmp_path: Path) -> None:
    source_kernel_dir = tmp_path / "demo" / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    source_kernel_path = source_kernel_dir / "kernel.py"
    source_kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "out = Path('outputs')",
                "out.mkdir(parents=True, exist_ok=True)",
                "out.joinpath('submission.csv').write_text('id,target\\n1,0.1\\n', encoding='utf-8')",
                "out.joinpath('metrics.json').write_text(",
                '    \'{"metric":"accuracy","offline_value":0.5}\',',
                "    encoding='utf-8',",
                ")",
                "out.joinpath('oof_predictions.tsv').write_text(",
                "    'row_id\\ty\\toof_pred\\toof_proba\\tfold\\n0\\t0\\t0\\t0.1\\t1\\n1\\t1\\t1\\t0.9\\t1\\n',",
                "    encoding='utf-8',",
                ")",
                "out.joinpath('split_diagnostics.json').write_text('{\"ok\": true}', encoding='utf-8')",
                "out.joinpath('feature_suspects.csv').write_text('col,score\\na,0.1\\n', encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    plan_path = tmp_path / "demo" / "plan.json"
    plan_path.write_text(json.dumps({"toggles": {"USE_MODEL": True}}, indent=2), encoding="utf-8")

    result = run_kernel_local(
        slug="demo",
        run_id="run-oof",
        iteration=1,
        base_dir=tmp_path,
        accelerator="gpu",
        score_source="cv",
        metric="auc",
        direction="maximize",
        holdout_frac=0.2,
        cv_folds=5,
        seed=42,
        dry_run=False,
        timeout_minutes=1,
        strict_accelerator=False,
    )

    assert (result.output_dir / "oof_predictions.tsv").exists()
    assert (result.output_dir / "split_diagnostics.json").exists()
    assert (result.output_dir / "feature_suspects.csv").exists()


def test_run_kernel_local_retries_cuda_oom_by_disabling_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENABLE_LLM", raising=False)
    monkeypatch.delenv("PIPELINE_NAME", raising=False)

    source_kernel_dir = tmp_path / "demo" / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    source_kernel_path = source_kernel_dir / "kernel.py"
    source_kernel_path.write_text(
        "\n".join(
            [
                "import os",
                "import sys",
                "from pathlib import Path",
                "",
                "if os.getenv('ENABLE_LLM', '1') != '0':",
                "    sys.stderr.write('torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.00 MiB\\n')",
                "    raise SystemExit(1)",
                "",
                "Path('submission.csv').write_text('id,target\\n1,0.1\\n', encoding='utf-8')",
                "Path('metrics.json').write_text('{\"metric\":\"rmse\",\"offline_value\":0.1}', encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_kernel_local(
        slug="demo",
        run_id="run-oom",
        iteration=1,
        base_dir=tmp_path,
        accelerator="gpu",
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        seed=42,
        dry_run=False,
        timeout_minutes=1,
        strict_accelerator=False,
    )

    assert result.submission_path is not None and result.submission_path.exists()
    assert result.metrics_path is not None and result.metrics_path.exists()
    logs_dir = tmp_path / "demo" / "runs" / "run-oom" / "iter-1" / "logs"
    assert (logs_dir / "local_kernel_stdout_oom_retry.log").exists()


def test_run_kernel_local_rejects_staged_plan_with_sequence_hyperparameters(tmp_path: Path) -> None:
    source_kernel_dir = tmp_path / "demo" / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    (source_kernel_dir / "kernel.py").write_text(
        "from pathlib import Path\nPath('submission.csv').write_text('id,target\\n1,0.1\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    plan_path = tmp_path / "demo" / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "toggles": {"USE_MODEL": True},
                "pipelines": [
                    {
                        "name": "pipe_a",
                        "key_hyperparameters": {"dropout": [0.05, 0.1]},
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(KernelFailedError, match="unresolved hyperparameter sequences"):
        run_kernel_local(
            slug="demo",
            run_id="run-bad-plan",
            iteration=1,
            base_dir=tmp_path,
            accelerator="gpu",
            score_source="holdout",
            metric="rmse",
            direction="minimize",
            holdout_frac=0.2,
            cv_folds=3,
            seed=42,
            dry_run=False,
            timeout_minutes=1,
            strict_accelerator=False,
        )


def test_run_kernel_local_host_memory_watchdog_kills_memory_hog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAGGLEBOT_LOCAL_KERNEL_MAX_RSS_MB", "32")

    source_kernel_dir = tmp_path / "demo" / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    source_kernel_path = source_kernel_dir / "kernel.py"
    source_kernel_path.write_text(
        "\n".join(
            [
                "import time",
                "",
                "SUBMISSION_NAME = 'submission.csv'",
                "METRICS_NAME = 'metrics.json'",
                "print('allocating memory', flush=True)",
                "blob = bytearray(96 * 1024 * 1024)",
                "print(len(blob), flush=True)",
                "time.sleep(10)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    plan_path = tmp_path / "demo" / "plan.json"
    plan_path.write_text(json.dumps({"toggles": {"USE_MODEL": True}}, indent=2), encoding="utf-8")

    with pytest.raises(KernelFailedError, match="exceeded host memory guard"):
        run_kernel_local(
            slug="demo",
            run_id="run-memguard",
            iteration=1,
            base_dir=tmp_path,
            accelerator="gpu",
            score_source="holdout",
            metric="rmse",
            direction="minimize",
            holdout_frac=0.2,
            cv_folds=3,
            seed=42,
            dry_run=False,
            timeout_minutes=1,
            strict_accelerator=False,
        )

    logs_dir = tmp_path / "demo" / "runs" / "run-memguard" / "iter-1" / "logs"
    stdout_log = logs_dir / "local_kernel_stdout.log"
    assert stdout_log.exists()


def test_run_kernel_local_finds_artifacts_in_parent_outputs(tmp_path: Path) -> None:
    source_kernel_dir = tmp_path / "demo" / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    source_kernel_path = source_kernel_dir / "kernel.py"
    source_kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "stage_dir = Path(__file__).resolve().parent",
                "challenge_dir = stage_dir.parent",
                "out_dir = challenge_dir / 'outputs'",
                "out_dir.mkdir(parents=True, exist_ok=True)",
                "sub = out_dir / 'submission.csv'",
                "met = out_dir / 'metrics.json'",
                "sub.write_text('id,target\\n1,0.1\\n', encoding='utf-8')",
                'met.write_text(\'{"metric":"rmse","offline_value":0.1}\', encoding=\'utf-8\')',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_kernel_local(
        slug="demo",
        run_id="run-5b",
        iteration=1,
        base_dir=tmp_path,
        accelerator="gpu",
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        seed=42,
        dry_run=False,
        timeout_minutes=1,
        strict_accelerator=False,
    )

    expected_output_dir = tmp_path / "demo" / "runs" / "run-5b" / "iter-1" / "output"
    assert result.submission_path == expected_output_dir / "submission.csv"
    assert result.metrics_path == expected_output_dir / "metrics.json"
    assert result.submission_path.exists()
    assert result.metrics_path.exists()


def test_run_kernel_local_exports_output_dir_env(tmp_path: Path) -> None:
    source_kernel_dir = tmp_path / "demo" / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    source_kernel_path = source_kernel_dir / "kernel.py"
    source_kernel_path.write_text(
        "\n".join(
            [
                "import os",
                "from pathlib import Path",
                "",
                "out_dir = Path(os.environ['KAGGLEBOT_OUTPUT_DIR'])",
                "out_dir.mkdir(parents=True, exist_ok=True)",
                "sub = out_dir / 'submission.csv'",
                "met = out_dir / 'metrics.json'",
                "sub.write_text('id,target\\n1,0.1\\n', encoding='utf-8')",
                'met.write_text(\'{"metric":"rmse","offline_value":0.1}\', encoding=\'utf-8\')',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_kernel_local(
        slug="demo",
        run_id="run-env-output",
        iteration=1,
        base_dir=tmp_path,
        accelerator="gpu",
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        seed=42,
        dry_run=False,
        timeout_minutes=1,
        strict_accelerator=False,
    )

    expected_output_dir = tmp_path / "demo" / "runs" / "run-env-output" / "iter-1" / "output"
    assert result.submission_path == expected_output_dir / "submission.csv"
    assert result.metrics_path == expected_output_dir / "metrics.json"
    assert result.submission_path.exists()
    assert result.metrics_path.exists()


@pytest.mark.parametrize("suffix", [".jsonl", ".ndjson", ".tsv.gz", ".pkl", ".dta", ".xml", ".orc", ".hdf5"])
def test_run_kernel_local_sets_submission_filename_from_sample_suffix(tmp_path: Path, suffix: str) -> None:
    source_kernel_dir = tmp_path / "demo" / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    source_kernel_path = source_kernel_dir / "kernel.py"
    source_kernel_path.write_text(
        "\n".join(
            [
                "import os",
                "from pathlib import Path",
                "",
                "name = os.environ['KAGGLEBOT_SUBMISSION_FILENAME']",
                f"if name != 'submission{suffix}':",
                "    raise AssertionError(name)",
                "if name.endswith('.pkl'):",
                "    import pandas as pd",
                "    pd.DataFrame({'id': [1], 'target': [0.1]}).to_pickle(name)",
                "elif name.endswith('.dta'):",
                "    import pandas as pd",
                "    pd.DataFrame({'id': [1], 'target': [0.1]}).to_stata(name, write_index=False)",
                "elif name.endswith('.xml'):",
                "    import pandas as pd",
                "    pd.DataFrame({'id': [1], 'target': [0.1]}).to_xml(name, index=False, parser='etree')",
                "elif name.endswith('.orc'):",
                "    import pandas as pd",
                "    pd.DataFrame({'id': [1], 'target': [0.1]}).to_orc(name, index=False)",
                "elif name.endswith('.hdf5'):",
                "    import pandas as pd",
                "    pd.DataFrame({'id': [1], 'target': [0.1]}).to_hdf(",
                "        name, key='submission', mode='w', format='table', index=False",
                "    )",
                "elif name.endswith('.tsv.gz'):",
                "    import gzip",
                "    with gzip.open(name, 'wt', encoding='utf-8') as handle:",
                "        handle.write('id\\ttarget\\n1\\t0.1\\n')",
                "else:",
                "    Path(name).write_text('{\"id\":1,\"target\":0.1}\\n', encoding='utf-8')",
                "Path('metrics.json').write_text('{\"metric\":\"rmse\",\"offline_value\":0.1}', encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    context_sample = tmp_path / "demo" / "context" / f"SampleSubmission{suffix}"
    context_sample.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".pkl":
        pd.DataFrame({"id": [1], "target": [0.0]}).to_pickle(context_sample)
    elif suffix == ".dta":
        pd.DataFrame({"id": [1], "target": [0.0]}).to_stata(context_sample, write_index=False)
    elif suffix == ".xml":
        pd.DataFrame({"id": [1], "target": [0.0]}).to_xml(context_sample, index=False, parser="etree")
    elif suffix == ".orc":
        pd.DataFrame({"id": [1], "target": [0.0]}).to_orc(context_sample, index=False)
    elif suffix == ".hdf5":
        pd.DataFrame({"id": [1], "target": [0.0]}).to_hdf(
            context_sample,
            key="submission",
            mode="w",
            format="table",
            index=False,
        )
    elif suffix == ".tsv.gz":
        with gzip.open(context_sample, "wt", encoding="utf-8") as handle:
            handle.write("id\ttarget\n1\t0.0\n")
    else:
        context_sample.write_text('{"id": 1, "target": 0.0}\n', encoding="utf-8")

    result = run_kernel_local(
        slug="demo",
        run_id="run-env-submission-name",
        iteration=1,
        base_dir=tmp_path,
        accelerator="gpu",
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        seed=42,
        dry_run=False,
        timeout_minutes=1,
        strict_accelerator=False,
    )

    expected_output_dir = tmp_path / "demo" / "runs" / "run-env-submission-name" / "iter-1" / "output"
    assert result.submission_path == expected_output_dir / f"submission{suffix}"
    if suffix == ".pkl":
        assert pd.read_pickle(result.submission_path).to_dict("list") == {"id": [1], "target": [0.1]}
    elif suffix == ".dta":
        assert pd.read_stata(result.submission_path).to_dict("list") == {"id": [1], "target": [0.1]}
    elif suffix == ".xml":
        assert pd.read_xml(result.submission_path, parser="etree").to_dict("list") == {"id": [1], "target": [0.1]}
    elif suffix == ".orc":
        assert pd.read_orc(result.submission_path).to_dict("list") == {"id": [1], "target": [0.1]}
    elif suffix == ".hdf5":
        assert pd.read_hdf(result.submission_path).to_dict("list") == {"id": [1], "target": [0.1]}
    elif suffix == ".tsv.gz":
        with gzip.open(result.submission_path, "rt", encoding="utf-8") as handle:
            assert handle.read() == "id\ttarget\n1\t0.1\n"
    else:
        assert result.submission_path.read_text(encoding="utf-8") == '{"id":1,"target":0.1}\n'


@pytest.mark.parametrize(
    ("suffix", "format_text"),
    [
        (".tar.xz", "Submit a submission.tar.xz archive containing model weights and inference code."),
        (".onnx", "Submit a single ONNX file named `submission.onnx`."),
    ],
)
def test_run_kernel_local_sets_submission_filename_from_submission_format(
    tmp_path: Path,
    suffix: str,
    format_text: str,
) -> None:
    source_kernel_dir = tmp_path / "demo" / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    source_kernel_path = source_kernel_dir / "kernel.py"
    source_kernel_path.write_text(
        "\n".join(
            [
                "import os",
                "from pathlib import Path",
                "",
                "name = os.environ['KAGGLEBOT_SUBMISSION_FILENAME']",
                f"if name != 'submission{suffix}':",
                "    raise AssertionError(name)",
                "Path(name).write_bytes(b'artifact')",
                "Path('metrics.json').write_text('{\"metric\":\"rmse\",\"offline_value\":0.1}', encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    context_dir = tmp_path / "demo" / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(f"## Submission Format\n{format_text}\n", encoding="utf-8")
    run_id = f"run-format-submission-name-{suffix.replace('.', '-').strip('-')}"

    result = run_kernel_local(
        slug="demo",
        run_id=run_id,
        iteration=1,
        base_dir=tmp_path,
        accelerator="gpu",
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        seed=42,
        dry_run=False,
        timeout_minutes=1,
        strict_accelerator=False,
    )

    expected_output_dir = tmp_path / "demo" / "runs" / run_id / "iter-1" / "output"
    assert result.submission_path == expected_output_dir / f"submission{suffix}"
    assert result.submission_path.read_bytes() == b"artifact"


def test_run_kernel_local_uses_explicit_submission_filename_from_submission_format(tmp_path: Path) -> None:
    source_kernel_dir = tmp_path / "demo" / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    source_kernel_path = source_kernel_dir / "kernel.py"
    source_kernel_path.write_text(
        "\n".join(
            [
                "import os",
                "from pathlib import Path",
                "",
                "name = os.environ['KAGGLEBOT_SUBMISSION_FILENAME']",
                "if name != 'answers.nii.gz':",
                "    raise AssertionError(name)",
                "Path(name).write_bytes(b'artifact')",
                "Path('metrics.json').write_text('{\"metric\":\"rmse\",\"offline_value\":0.1}', encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    context_dir = tmp_path / "demo" / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload a single file named `answers.nii.gz`.\n",
        encoding="utf-8",
    )

    result = run_kernel_local(
        slug="demo",
        run_id="run-format-explicit-submission-name",
        iteration=1,
        base_dir=tmp_path,
        accelerator="gpu",
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        seed=42,
        dry_run=False,
        timeout_minutes=1,
        strict_accelerator=False,
    )

    expected_output_dir = tmp_path / "demo" / "runs" / "run-format-explicit-submission-name" / "iter-1" / "output"
    assert result.submission_path == expected_output_dir / "answers.nii.gz"
    assert result.submission_path.read_bytes() == b"artifact"


def test_run_kernel_local_does_not_override_explicit_submission_filename_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_kernel_dir = tmp_path / "demo" / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    source_kernel_path = source_kernel_dir / "kernel.py"
    source_kernel_path.write_text(
        "\n".join(
            [
                "import os",
                "from pathlib import Path",
                "",
                "name = os.environ['KAGGLEBOT_SUBMISSION_FILENAME']",
                "if name != 'custom.tsv':",
                "    raise AssertionError(name)",
                "Path(name).write_text('id\\ttarget\\n1\\t0.1\\n', encoding='utf-8')",
                "Path('metrics.json').write_text('{\"metric\":\"rmse\",\"offline_value\":0.1}', encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    context_sample = tmp_path / "demo" / "context" / "sample_submission.jsonl"
    context_sample.parent.mkdir(parents=True, exist_ok=True)
    context_sample.write_text('{"id": 1, "target": 0.0}\n', encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_FILENAME", "custom.tsv")

    result = run_kernel_local(
        slug="demo",
        run_id="run-explicit-submission-name",
        iteration=1,
        base_dir=tmp_path,
        accelerator="gpu",
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        seed=42,
        dry_run=False,
        timeout_minutes=1,
        strict_accelerator=False,
    )

    expected_output_dir = tmp_path / "demo" / "runs" / "run-explicit-submission-name" / "iter-1" / "output"
    assert result.submission_path == expected_output_dir / "custom.tsv"
    assert result.submission_path.read_text(encoding="utf-8") == "id\ttarget\n1\t0.1\n"


def test_run_kernel_local_finds_artifacts_in_legacy_kernel_output(tmp_path: Path) -> None:
    source_kernel_dir = tmp_path / "demo" / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    source_kernel_path = source_kernel_dir / "kernel.py"
    source_kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "out_dir = Path(__file__).resolve().parent.parents[2] / 'kernel_output'",
                "out_dir.mkdir(parents=True, exist_ok=True)",
                "sub = out_dir / 'submission.csv'",
                "met = out_dir / 'metrics.json'",
                "sub.write_text('id,target\\n1,0.1\\n', encoding='utf-8')",
                'met.write_text(\'{"metric":"rmse","offline_value":0.1}\', encoding=\'utf-8\')',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_kernel_local(
        slug="demo",
        run_id="run-legacy-output",
        iteration=1,
        base_dir=tmp_path,
        accelerator="gpu",
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        seed=42,
        dry_run=False,
        timeout_minutes=1,
        strict_accelerator=False,
    )

    expected_output_dir = tmp_path / "demo" / "runs" / "run-legacy-output" / "iter-1" / "output"
    assert result.submission_path == expected_output_dir / "submission.csv"
    assert result.metrics_path == expected_output_dir / "metrics.json"
    assert result.submission_path.exists()
    assert result.metrics_path.exists()


def test_run_kernel_local_mirrors_context_sample_submission(tmp_path: Path) -> None:
    source_kernel_dir = tmp_path / "demo" / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    source_kernel_path = source_kernel_dir / "kernel.py"
    source_kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "data_root = Path(__file__).resolve().parents[3] / 'data'",
                "sample = data_root / 'sample_submission.csv'",
                "if not sample.exists():",
                "    raise FileNotFoundError(f'sample missing at {sample}')",
                "Path('submission.csv').write_text('id,target\\n1,0.1\\n', encoding='utf-8')",
                "Path('metrics.json').write_text('{\"metric\":\"rmse\",\"offline_value\":0.1}', encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    context_sample = tmp_path / "demo" / "context" / "sample_submission.csv"
    context_sample.parent.mkdir(parents=True, exist_ok=True)
    context_sample.write_text("id,target\n1,0.0\n", encoding="utf-8")

    result = run_kernel_local(
        slug="demo",
        run_id="run-6",
        iteration=1,
        base_dir=tmp_path,
        accelerator="gpu",
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        seed=42,
        dry_run=False,
        timeout_minutes=1,
        strict_accelerator=False,
    )

    mirrored = tmp_path / "demo" / "data" / "sample_submission.csv"
    assert mirrored.exists()
    assert mirrored.read_text(encoding="utf-8") == "id,target\n1,0.0\n"
    assert result.submission_path is not None and result.submission_path.exists()
    assert result.metrics_path is not None and result.metrics_path.exists()


def test_run_kernel_local_mirrors_context_dataset_profile(tmp_path: Path) -> None:
    source_kernel_dir = tmp_path / "demo" / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    source_kernel_path = source_kernel_dir / "kernel.py"
    source_kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "Path('submission.csv').write_text('id,target\\n1,0.1\\n', encoding='utf-8')",
                "Path('metrics.json').write_text('{\"metric\":\"rmse\",\"offline_value\":0.1}', encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    context_dir = tmp_path / "demo" / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "sample_submission.csv").write_text("id,target\n1,0.0\n", encoding="utf-8")
    profile_payload = {"modality": "tabular", "task": "regression", "target_column": "target"}
    (context_dir / "dataset_profile.json").write_text(json.dumps(profile_payload), encoding="utf-8")

    result = run_kernel_local(
        slug="demo",
        run_id="run-6-profile",
        iteration=1,
        base_dir=tmp_path,
        accelerator="gpu",
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        seed=42,
        dry_run=False,
        timeout_minutes=1,
        strict_accelerator=False,
    )

    staged_profile = tmp_path / "demo" / "kernels" / "run-6-profile" / "context" / "dataset_profile.json"
    assert staged_profile.exists()
    assert json.loads(staged_profile.read_text(encoding="utf-8")) == profile_payload
    assert result.submission_path is not None and result.submission_path.exists()
    assert result.metrics_path is not None and result.metrics_path.exists()


def test_run_kernel_local_stages_competition_data_dir(tmp_path: Path) -> None:
    source_kernel_dir = tmp_path / "demo" / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    source_kernel_path = source_kernel_dir / "kernel.py"
    source_kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "Path('submission.csv').write_text('id,target\\n1,0.1\\n', encoding='utf-8')",
                "Path('metrics.json').write_text('{\"metric\":\"rmse\",\"offline_value\":0.1}', encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text("id,target\n1,0\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("id\n1\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")

    result = run_kernel_local(
        slug="demo",
        run_id="run-6b",
        iteration=1,
        base_dir=tmp_path,
        accelerator="gpu",
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        seed=42,
        dry_run=False,
        timeout_minutes=1,
        strict_accelerator=False,
    )

    staged_data_dir = tmp_path / "demo" / "kernels" / "run-6b" / "data"
    assert staged_data_dir.exists()
    assert (staged_data_dir / "train.csv").exists()
    assert (staged_data_dir / "test.csv").exists()
    assert (staged_data_dir / "sample_submission.csv").exists()
    assert result.submission_path is not None and result.submission_path.exists()
    assert result.metrics_path is not None and result.metrics_path.exists()


def test_run_kernel_local_supports_legacy_artifacts_data_dir_layout(tmp_path: Path) -> None:
    source_kernel_dir = tmp_path / "demo" / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    source_kernel_path = source_kernel_dir / "kernel.py"
    source_kernel_path.write_text(
        "\n".join(
            [
                "import os",
                "from pathlib import Path",
                "",
                "this_file = Path(__file__).resolve()",
                "slug = os.getenv('KAGGLEBOT_COMPETITION_SLUG', 'demo')",
                "repo_root = this_file.parents[3]",
                "legacy_data_dir = repo_root / 'artifacts' / slug / 'data'",
                "if not legacy_data_dir.exists():",
                "    raise FileNotFoundError(f'Data directory not found: {legacy_data_dir}')",
                "Path('submission.csv').write_text('id,target\\n1,0.1\\n', encoding='utf-8')",
                "Path('metrics.json').write_text('{\"metric\":\"rmse\",\"offline_value\":0.1}', encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text("id,target\n1,0\n", encoding="utf-8")

    result = run_kernel_local(
        slug="demo",
        run_id="run-legacy-path",
        iteration=1,
        base_dir=tmp_path,
        accelerator="gpu",
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        seed=42,
        dry_run=False,
        timeout_minutes=1,
        strict_accelerator=False,
    )

    assert result.submission_path is not None and result.submission_path.exists()
    assert result.metrics_path is not None and result.metrics_path.exists()


def test_run_kernel_local_stages_non_tabular_data_tree(tmp_path: Path) -> None:
    source_kernel_dir = tmp_path / "demo" / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    source_kernel_path = source_kernel_dir / "kernel.py"
    source_kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "data_root = Path(__file__).resolve().parents[1] / 'data'",
                "assert (data_root / 'images' / 'a.jpg').exists()",
                "assert (data_root / 'labels' / 'a.txt').exists()",
                "assert (data_root / 'sample_submission.csv').exists()",
                "Path('submission.csv').write_text('id,target\\n1,0.1\\n', encoding='utf-8')",
                "Path('metrics.json').write_text('{\"metric\":\"rmse\",\"offline_value\":0.1}', encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    data_dir = tmp_path / "demo" / "data"
    (data_dir / "images").mkdir(parents=True, exist_ok=True)
    (data_dir / "labels").mkdir(parents=True, exist_ok=True)
    (data_dir / "images" / "a.jpg").write_bytes(b"img")
    (data_dir / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")

    result = run_kernel_local(
        slug="demo",
        run_id="run-6c",
        iteration=1,
        base_dir=tmp_path,
        accelerator="gpu",
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        seed=42,
        dry_run=False,
        timeout_minutes=1,
        strict_accelerator=False,
    )

    staged_data_dir = tmp_path / "demo" / "kernels" / "run-6c" / "data"
    assert staged_data_dir.exists()
    assert (staged_data_dir / "images" / "a.jpg").exists()
    assert (staged_data_dir / "labels" / "a.txt").exists()
    assert (staged_data_dir / "sample_submission.csv").exists()
    assert result.submission_path is not None and result.submission_path.exists()
    assert result.metrics_path is not None and result.metrics_path.exists()


def test_run_kernel_local_records_duration_history(tmp_path: Path) -> None:
    source_kernel_dir = tmp_path / "demo" / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    source_kernel_path = source_kernel_dir / "kernel.py"
    source_kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "Path('submission.csv').write_text('id,target\\n1,0.1\\n', encoding='utf-8')",
                "Path('metrics.json').write_text('{\"metric\":\"rmse\",\"offline_value\":0.1}', encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    context_sample = tmp_path / "demo" / "context" / "sample_submission.csv"
    context_sample.parent.mkdir(parents=True, exist_ok=True)
    context_sample.write_text("id,target\n1,0.0\n", encoding="utf-8")

    _ = run_kernel_local(
        slug="demo",
        run_id="run-7",
        iteration=1,
        base_dir=tmp_path,
        accelerator="gpu",
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        seed=42,
        dry_run=False,
        timeout_minutes=1,
        strict_accelerator=False,
    )
    from kagglebot.local_kernel_duration import estimate_local_kernel_duration_seconds

    estimate, samples = estimate_local_kernel_duration_seconds(base_dir=tmp_path, slug="demo")
    assert samples == 1
    assert estimate is not None and estimate > 0.0


def test_ensure_training_progress_shim_requires_marker(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    site_path = kernel_dir / "sitecustomize.py"
    site_path.write_text("# no marker\n", encoding="utf-8")

    with pytest.raises(KernelFailedError, match="mandatory progress logging"):
        ensure_training_progress_shim(kernel_dir)
