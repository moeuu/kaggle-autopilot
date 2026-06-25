"""Tests for kernel runner helpers."""

from __future__ import annotations

import base64
import gzip
import json
import os
import re
import runpy
import time
from pathlib import Path

import pytest

from kagglebot.exceptions import KernelCapacityError, KernelFailedError, KernelStillRunningError, KernelTimeoutError
from kagglebot.kernel_runner import (
    _build_local_kernel_progress_tracker,
    _copy_competition_external_assets,
    _copy_kernel_sources,
    _copy_shared_kernel_runtime_modules,
    _ensure_training_progress_shim,
    _format_local_gpu_activity_suffix,
    _format_local_kernel_activity_suffix,
    _load_dataset_profile_identity,
    _LocalKernelLogFilterState,
    _resolve_kernel_slug,
    _resolve_submit_kernel_slug,
    _run_local_kernel_once,
    _should_suppress_local_kernel_log_line,
    resolve_kaggle_username,
    run_kernel,
    run_kernel_local,
    run_submit_kernel,
    sanitize_kernel_slug,
)

pytestmark = pytest.mark.slow


def test_sanitize_kernel_slug() -> None:
    assert sanitize_kernel_slug("KaggleBot Titan! 2024") == "kagglebot-titan-2024"


def test_submit_kernel_slug_stays_distinct_when_long_slug_is_truncated() -> None:
    slug = "deep-past-initiative-machine-translation"
    run_id = "20260318T153341Z-52de3f45"
    train_slug = _resolve_kernel_slug(None, slug, run_id, 1)
    submit_slug = _resolve_submit_kernel_slug(None, slug, run_id, 1)

    assert train_slug == "kagglebot-deep-past-initiative-machine-t-de3f45-i1"
    assert submit_slug.startswith("kagglebot-submit-")
    assert submit_slug != train_slug
    assert len(submit_slug) <= 50


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
    assert payload["enable_gpu"] is False
    assert payload["enable_tpu"] is False


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
    (input_dir / "sample_submission.csv").write_text(
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
    assert payload["enable_gpu"] is False
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
    kernel_id = kernel_runner.KernelJobMonitor().push_and_wait(
        preparation=preparation,
        slug="demo",
        timeout_minutes=1,
    )
    assert kernel_id == "user/kernel-slug"


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
    from kagglebot import kernel_runner

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

    kernel_runner._ensure_kernel_competition_slug_env(kernel_dir, "demo")
    updated = kernel_path.read_text(encoding="utf-8")
    assert "_kb_os.environ['KAGGLEBOT_COMPETITION_SLUG'] = \"demo\"" in updated
    assert "_kb_os.environ['KAGGLEBOT_SLUG'] = \"demo\"" in updated
    assert "_kb_os.environ['KAGGLEBOT_COMPETITION_SLUG'] = \"kaggle\"" not in updated


def test_inject_data_dir_resolver_rewrites_candidate_presence_check(tmp_path: Path) -> None:
    from kagglebot import kernel_runner

    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "def locate_data_dir(slug: str) -> Path:",
                "    required = ('train.csv', 'test.csv', 'sample_submission.csv')",
                "    for cand in [Path(f'/kaggle/input/{slug}')]:",
                "        if all((cand / name).exists() for name in required):",
                "            return cand",
                "    raise FileNotFoundError(f\"Could not find required csv files for slug='{slug}'\")",
                "",
                "def load_competition_frames(data_dir: Path):",
                "    return data_dir / 'train.csv', data_dir / 'test.csv'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    kernel_runner._inject_data_dir_resolver(kernel_dir)
    updated = kernel_path.read_text(encoding="utf-8")
    assert "# kagglebot:data_resolver" in updated
    assert "all(_kb_find_file(cand, name).exists() for name in required)" in updated
    assert "_kb_find_file(data_dir, 'train.csv')" in updated
    assert "_kb_find_file(data_dir, 'test.csv')" in updated
    assert "# kagglebot:data-dir-fallback-scan" in updated
    assert "for cand in sorted(input_root.iterdir(), key=lambda p: p.name):" in updated


def test_inject_data_dir_resolver_upgrades_existing_marker(tmp_path: Path) -> None:
    from kagglebot import kernel_runner

    kernel_dir = tmp_path / "demo" / "kernels" / "run-1"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "# kagglebot:data_resolver",
                "from pathlib import Path as _KBPath",
                "def _kb_find_file(base: _KBPath, name: str) -> _KBPath:",
                "    return base / name",
                "",
                "def locate_data_dir(slug: str):",
                "    required = ('train.csv', 'test.csv', 'sample_submission.csv')",
                "    for cand in [Path(f'/kaggle/input/{slug}')]:",
                "        if all((cand / name).exists() for name in required):",
                "            return cand",
                "    raise FileNotFoundError(f\"Could not find required csv files for slug='{slug}'\")",
                "",
                "def load_competition_frames(data_dir):",
                "    return data_dir / 'train.csv'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    kernel_runner._inject_data_dir_resolver(kernel_dir)
    updated = kernel_path.read_text(encoding="utf-8")
    assert updated.count("# kagglebot:data_resolver") == 1
    assert "all(_kb_find_file(cand, name).exists() for name in required)" in updated
    assert "_kb_find_file(data_dir, 'train.csv')" in updated
    assert "# kagglebot:data-dir-fallback-scan" in updated


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

    _copy_kernel_sources(kernel_source_dir, kernel_dir)
    _copy_shared_kernel_runtime_modules(kernel_dir)
    _copy_competition_external_assets(base_dir=base_dir, slug=slug, kernel_dir=kernel_dir)

    assert (kernel_dir / "kernel.py").exists()
    assert (kernel_dir / "runtime.py").exists()
    assert (kernel_dir / "tabular_ensemble.py").exists()
    assert (kernel_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv").exists()
    assert not (kernel_dir / "output").exists()
    assert not (kernel_dir / "outputs").exists()
    assert not (kernel_dir / "__pycache__").exists()


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
    result = _run_local_kernel_once(
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
    state = _LocalKernelLogFilterState()
    lines = [
        "/tmp/kernel.py:1036: PerformanceWarning: DataFrame is highly fragmented.\n",
        "  out[ratio_col] = out[t1] / (out[t2].abs() + 1e-6)\n",
        "Default metric period is 5 because BrierScore is/are not implemented for GPU\n",
        "training fold=1\n",
    ]

    suppressed = [_should_suppress_local_kernel_log_line(line, state=state) for line in lines]

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

    result = _run_local_kernel_once(
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
    tracker = _build_local_kernel_progress_tracker(base_dir=tmp_path, slug="demo", watch_dirs=[])

    result = _run_local_kernel_once(
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


def test_write_kernel_metadata_ignores_invalid_existing_metadata(tmp_path: Path) -> None:
    from kagglebot import kernel_runner

    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "kernel-metadata.json").write_text("{", encoding="utf-8")

    kernel_runner._write_kernel_metadata(
        kernel_dir=kernel_dir,
        kernel_id="user/demo",
        title="demo",
        code_file="kernel.py",
        kernel_type="script",
        accelerator="gpu",
        enable_internet=False,
        competition_slug="competition",
    )

    payload = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert payload["id"] == "user/demo"
    assert payload["competition_sources"] == ["competition"]
    assert payload["dataset_sources"] == []
    assert payload["kernel_sources"] == []
    assert payload["model_sources"] == []


def test_inject_column_fill_shim(tmp_path: Path) -> None:
    from kagglebot import kernel_runner

    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text("print('ok')\n", encoding="utf-8")
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    payload = {"files": {"test.csv": ["A", "B"]}}
    (context_dir / "column_fill.json").write_text(json.dumps(payload), encoding="utf-8")

    kernel_runner._inject_column_fill_shim(kernel_dir, context_dir)

    site_path = kernel_dir / "sitecustomize.py"
    assert site_path.exists()
    text = site_path.read_text(encoding="utf-8")
    assert "column-fill-shim" in text
    assert "column_fill.json" in text
    assert "_pd.DataFrame.__getitem__" in text
    assert "float('nan')" in text
    assert "_pd.NA" not in text
    assert (kernel_dir / "column_fill.json").exists()


def test_prepare_zero_overlap_drift_guard_detects_high_risk_zero_overlap_feature(tmp_path: Path) -> None:
    from kagglebot import kernel_runner

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

    guard_path = kernel_runner._prepare_zero_overlap_drift_guard(
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
    from kagglebot import kernel_runner

    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text("print('ok')\n", encoding="utf-8")
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    payload = {"enabled": True, "drop_columns": ["risk_cat"]}
    (context_dir / "zero_overlap_drift_guard.json").write_text(json.dumps(payload), encoding="utf-8")

    kernel_runner._inject_zero_overlap_drift_shim(kernel_dir, context_dir)
    kernel_runner._inject_zero_overlap_drift_shim(kernel_dir, context_dir)

    site_path = kernel_dir / "sitecustomize.py"
    assert site_path.exists()
    text = site_path.read_text(encoding="utf-8")
    assert "zero-overlap-drift-shim" in text
    assert text.count("zero-overlap-drift-shim") == 1
    assert "zero_overlap_drift_guard.json" in text
    assert (kernel_dir / "zero_overlap_drift_guard.json").exists()


def test_inject_object_coerce_shim(tmp_path: Path) -> None:
    from kagglebot import kernel_runner

    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text("print('ok')\n", encoding="utf-8")
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    payload = {"enabled": True}
    (context_dir / "object_coerce.json").write_text(json.dumps(payload), encoding="utf-8")

    kernel_runner._inject_object_coerce_shim(kernel_dir, context_dir)

    site_path = kernel_dir / "sitecustomize.py"
    assert site_path.exists()
    text = site_path.read_text(encoding="utf-8")
    assert "object-coerce-shim" in text
    assert "object_coerce.json" in text
    assert (kernel_dir / "object_coerce.json").exists()


def test_inject_device_coerce_shim(tmp_path: Path) -> None:
    from kagglebot import kernel_runner

    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text("print('ok')\n", encoding="utf-8")
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    payload = {"enabled": True}
    (context_dir / "device_coerce.json").write_text(json.dumps(payload), encoding="utf-8")

    kernel_runner._inject_device_coerce_shim(kernel_dir, context_dir)

    site_path = kernel_dir / "sitecustomize.py"
    assert site_path.exists()
    text = site_path.read_text(encoding="utf-8")
    assert "device-coerce-shim" in text
    assert "device_coerce.json" in text
    assert (kernel_dir / "device_coerce.json").exists()


def test_inject_local_runtime_shims(tmp_path: Path) -> None:
    from kagglebot import kernel_runner

    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    kernel_runner._inject_kaggle_working_redirect_shim(kernel_dir)
    kernel_runner._inject_lgbm_gpu_guard_shim(kernel_dir)

    site_path = kernel_dir / "sitecustomize.py"
    assert site_path.exists()
    text = site_path.read_text(encoding="utf-8")
    assert "kaggle-working-redirect-shim" in text
    assert "lgbm-gpu-guard-shim" in text


def test_inject_transformers_eval_strategy_shim(tmp_path: Path) -> None:
    from kagglebot import kernel_runner

    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    kernel_runner._inject_transformers_eval_strategy_shim(kernel_dir)

    site_path = kernel_dir / "sitecustomize.py"
    assert site_path.exists()
    text = site_path.read_text(encoding="utf-8")
    assert "transformers-eval-strategy-shim" in text
    assert "evaluation_strategy" in text
    assert "eval_strategy" in text
    assert "Seq2SeqTrainingArguments" in text


def test_apply_local_runtime_env_defaults_sets_optional_backend_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kagglebot import kernel_runner

    monkeypatch.setattr(
        kernel_runner,
        "_module_available",
        lambda name: False if name == "xgboost" else True,
    )
    monkeypatch.setattr(kernel_runner, "_local_lightgbm_gpu_probe_usable", lambda: False)
    monkeypatch.delenv("KAGGLEBOT_FORCE_LGBM_GPU", raising=False)

    env: dict[str, str] = {}
    notes = kernel_runner._apply_local_runtime_env_defaults(
        env=env,
        accelerator="gpu",
        local_working_dir=tmp_path / "local-working",
    )

    assert env["KAGGLEBOT_DISABLE_KAGGLE_WORKING_WRITES"] == "1"
    assert env["KAGGLEBOT_LOCAL_WORKING_DIR"] == str(tmp_path / "local-working")
    assert env["KAGGLEBOT_NUM_WORKERS"] == "0"
    assert env["KAGGLEBOT_TORCH_SHARING_STRATEGY"] == "file_system"
    assert env["KAGGLEBOT_LOCAL_NOFILE"] == "4096"
    assert env["KAGGLEBOT_LOCAL_KERNEL_STALL_SEC"] == "900"
    assert env["KAGGLEBOT_DO_TRAIN"] == "1"
    assert env["KAGGLEBOT_FORCE_TRAIN"] == "1"
    assert env["KAGGLEBOT_ALLOW_MODEL_DOWNLOAD"] == "1"
    assert env["USE_XGB"] == "0"
    assert env["KAGGLEBOT_DISABLE_XGBOOST"] == "1"
    assert env["USE_LGBM_GPU"] == "0"
    assert env["KAGGLEBOT_DISABLE_LGBM_GPU"] == "1"
    assert any("xgboost unavailable" in note for note in notes)
    assert any("LightGBM GPU probe failed" in note for note in notes)
    assert any("KAGGLEBOT_ALLOW_MODEL_DOWNLOAD=1" in note for note in notes)
    assert any("KAGGLEBOT_NUM_WORKERS=0" in note for note in notes)
    assert any("KAGGLEBOT_TORCH_SHARING_STRATEGY=file_system" in note for note in notes)


def test_inject_pipeline_cfg_fallback_replaces_keyerror(tmp_path: Path) -> None:
    from kagglebot import kernel_runner

    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "def get_pipeline_cfg(plan, name):",
                "    for p in plan.get('pipelines', []):",
                "        if p.get('name') == name:",
                "            return p",
                '    raise KeyError(f"Pipeline not found in plan: {name}")',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    kernel_runner._inject_pipeline_cfg_fallback(kernel_dir)
    text = kernel_path.read_text(encoding="utf-8")
    assert "kagglebot:pipeline_cfg_fallback" in text
    assert "raise KeyError" not in text
    assert "missing_pipeline_in_plan" in text


def test_kernel_bootstrap_preserves_future_import(tmp_path: Path) -> None:
    from kagglebot import kernel_runner

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

    kernel_runner._ensure_kernel_import_path(kernel_dir)
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
    from kagglebot import kernel_runner

    google_dir = tmp_path / "models--google--byt5-large" / "snapshots" / "abc123"
    google_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("config.json", "tokenizer_config.json", "model.safetensors"):
        (google_dir / filename).write_text("x", encoding="utf-8")
    alias_dir = tmp_path / "artemgoncarov_dpc_byt5_large"
    alias_dir.symlink_to(google_dir, target_is_directory=True)

    staged = kernel_runner._stage_resolved_model_hints(
        hints=["artemgoncarov/dpc-byt5-large"],
        candidate_dirs=[alias_dir, google_dir],
        staged_root=tmp_path / "staged-models",
    )

    assert staged == []


def test_stage_resolved_model_hints_rejects_mattia_alias_pointing_to_assiaben(tmp_path: Path) -> None:
    from kagglebot import kernel_runner

    assiaben_dir = tmp_path / "dataset__assiaben__final-byt5" / "byt5-akkadian-optimized-34x"
    assiaben_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("config.json", "tokenizer_config.json", "model.safetensors"):
        (assiaben_dir / filename).write_text("x", encoding="utf-8")
    alias_dir = tmp_path / "mattiaangeli_byt5_akkadian_mbr_pytorch_default_6"
    alias_dir.symlink_to(assiaben_dir, target_is_directory=True)

    staged = kernel_runner._stage_resolved_model_hints(
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
                "out.joinpath('oof_predictions.csv').write_text(",
                "    'row_id,y,oof_pred,oof_proba,fold\\n0,0,0,0.1,1\\n1,1,1,0.9,1\\n',",
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

    assert (result.output_dir / "oof_predictions.csv").exists()
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


def test_local_kernel_memory_cap_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from kagglebot import kernel_runner

    monkeypatch.setenv("KAGGLEBOT_LOCAL_KERNEL_MAX_RSS_MB", "64")
    cap_bytes = kernel_runner._resolve_local_kernel_memory_cap_bytes(dict(os.environ))  # noqa: SLF001
    assert cap_bytes == 64 * 1024 * 1024


def test_local_kernel_memory_cap_env_rejects_invalid_override() -> None:
    from kagglebot import kernel_runner

    with pytest.raises(KernelFailedError, match="positive integer number of MiB"):
        kernel_runner._resolve_local_kernel_memory_cap_bytes({"KAGGLEBOT_LOCAL_KERNEL_MAX_RSS_MB": "64.5"})  # noqa: SLF001


def test_local_kernel_stall_timeout_env_uses_minimum_and_disable() -> None:
    from kagglebot import kernel_runner

    assert kernel_runner._resolve_local_kernel_stall_timeout_sec({"KAGGLEBOT_LOCAL_KERNEL_STALL_SEC": "1"}) == 5.0  # noqa: SLF001
    assert kernel_runner._resolve_local_kernel_stall_timeout_sec({"KAGGLEBOT_LOCAL_KERNEL_STALL_SEC": "0"}) is None  # noqa: SLF001


def test_local_kernel_stall_timeout_env_rejects_invalid_override() -> None:
    from kagglebot import kernel_runner

    with pytest.raises(KernelFailedError, match="positive number of seconds"):
        kernel_runner._resolve_local_kernel_stall_timeout_sec({"KAGGLEBOT_LOCAL_KERNEL_STALL_SEC": "nan"})  # noqa: SLF001


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


def test_load_dataset_profile_identity_ignores_missing_invalid_or_non_object_payload(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir()

    assert _load_dataset_profile_identity(context_dir=context_dir) == (None, None)

    profile_path = context_dir / "dataset_profile.json"
    profile_path.write_text("{", encoding="utf-8")
    assert _load_dataset_profile_identity(context_dir=context_dir) == (None, None)

    profile_path.write_text("[]", encoding="utf-8")
    assert _load_dataset_profile_identity(context_dir=context_dir) == (None, None)

    profile_path.write_text(json.dumps({"target_column": "target", "id_column": "id"}), encoding="utf-8")
    assert _load_dataset_profile_identity(context_dir=context_dir) == ("target", "id")


def test_validate_local_kernel_plan_runtime_hyperparameters_rejects_non_object_payload(tmp_path: Path) -> None:
    from kagglebot import kernel_runner

    plan_path = tmp_path / "plan.json"
    plan_path.write_text("[]", encoding="utf-8")

    with pytest.raises(KernelFailedError, match="must be a JSON object"):
        kernel_runner._validate_local_kernel_plan_runtime_hyperparameters(plan_path)


def test_ensure_local_sample_submission_file_expands_placeholder_template(tmp_path: Path) -> None:
    from kagglebot import kernel_runner

    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text("id,target\n1,0\n2,1\n3,0\n", encoding="utf-8")
    (data_dir / "test.csv").write_text(
        "id,feature\n1,10\n2,20\n3,30\n4,40\n5,50\n6,60\n7,70\n8,80\n9,90\n10,100\n11,110\n12,120\n13,130\n14,140\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0\n2,0\n3,0\n", encoding="utf-8")

    resolved = kernel_runner._ensure_local_sample_submission_file(base_dir=tmp_path, slug="demo")

    assert resolved == data_dir / "sample_submission.csv"
    lines = (data_dir / "sample_submission.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 15  # header + 14 test ids
    assert lines[0] == "id,target"
    assert lines[1].startswith("1,")
    assert lines[14].startswith("14,")


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


def test_stage_local_kernel_data_dir_replaces_stale_file_target(tmp_path: Path) -> None:
    from kagglebot import kernel_runner

    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "sample_submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
    (data_dir / "images").mkdir(exist_ok=True)
    (data_dir / "images" / "a.jpg").write_bytes(b"img")

    run_dir = tmp_path / "demo" / "kernels" / "run-stale"
    run_dir.mkdir(parents=True, exist_ok=True)
    stale_target = run_dir / "data"
    stale_target.write_text("stale", encoding="utf-8")

    kernel_runner._stage_local_kernel_data_dir(base_dir=tmp_path, slug="demo", run_dir=run_dir)

    assert stale_target.exists()
    assert stale_target.is_dir() or stale_target.is_symlink()
    assert (stale_target / "sample_submission.csv").exists()
    assert (stale_target / "images" / "a.jpg").exists()
    compat_target = tmp_path / "demo" / "artifacts" / "demo" / "data"
    assert compat_target.exists()
    assert compat_target.is_dir() or compat_target.is_symlink()
    assert (compat_target / "sample_submission.csv").exists()
    assert (compat_target / "images" / "a.jpg").exists()


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


def test_normalize_local_kernel_metrics_promotes_urban_flood_flat_full_data(tmp_path: Path) -> None:
    from kagglebot import kernel_runner

    data_dir = tmp_path / "urban-flood-modelling" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in kernel_runner._URBAN_FLOOD_FLAT_FULL_REQUIRED_FILES:
        (data_dir / name).write_text("stub\n", encoding="utf-8")

    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "metric": "rmse",
                "offline_value": 0.1,
                "score_source": "sample_diagnostic",
            }
        ),
        encoding="utf-8",
    )

    normalized = kernel_runner._normalize_local_kernel_metrics(
        slug="urban-flood-modelling",
        data_dir=data_dir,
        metrics_path=metrics_path,
        score_source="cv",
    )

    assert normalized == metrics_path
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["score_source"] == "cv"
    assert payload["dataset_kind"] == "full"
    assert payload["dataset_mode"] == "full"
    assert payload["full_dataset_resolved"] is True
    assert payload["metrics_normalized_by"] == "kernel_runner.local_full_data_guard"


def test_normalize_local_kernel_metrics_keeps_other_slugs_unchanged(tmp_path: Path) -> None:
    from kagglebot import kernel_runner

    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in kernel_runner._URBAN_FLOOD_FLAT_FULL_REQUIRED_FILES:
        (data_dir / name).write_text("stub\n", encoding="utf-8")

    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "metric": "rmse",
                "offline_value": 0.1,
                "score_source": "sample_diagnostic",
            }
        ),
        encoding="utf-8",
    )

    kernel_runner._normalize_local_kernel_metrics(
        slug="demo",
        data_dir=data_dir,
        metrics_path=metrics_path,
        score_source="cv",
    )

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["score_source"] == "sample_diagnostic"
    assert "full_dataset_resolved" not in payload


def test_normalize_local_kernel_metrics_ignores_invalid_or_non_object_metrics(tmp_path: Path) -> None:
    from kagglebot import kernel_runner

    data_dir = tmp_path / "urban-flood-modelling" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in kernel_runner._URBAN_FLOOD_FLAT_FULL_REQUIRED_FILES:
        (data_dir / name).write_text("stub\n", encoding="utf-8")

    invalid = tmp_path / "invalid_metrics.json"
    invalid.write_text("{", encoding="utf-8")
    assert (
        kernel_runner._normalize_local_kernel_metrics(
            slug="urban-flood-modelling",
            data_dir=data_dir,
            metrics_path=invalid,
            score_source="cv",
        )
        == invalid
    )
    assert invalid.read_text(encoding="utf-8") == "{"

    array_payload = tmp_path / "array_metrics.json"
    array_payload.write_text("[]", encoding="utf-8")
    assert (
        kernel_runner._normalize_local_kernel_metrics(
            slug="urban-flood-modelling",
            data_dir=data_dir,
            metrics_path=array_payload,
            score_source="cv",
        )
        == array_payload
    )
    assert array_payload.read_text(encoding="utf-8") == "[]"


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


def test_build_local_kernel_progress_tracker_reads_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "demo" / "plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps({"cv_folds": 5, "eval_seeds": [42, 2024, 777]}, indent=2),
        encoding="utf-8",
    )

    tracker = _build_local_kernel_progress_tracker(base_dir=tmp_path, slug="demo")
    assert tracker.expected_folds == 5
    assert tracker.expected_seeds == [42, 2024, 777]


def test_build_local_kernel_progress_tracker_ignores_invalid_or_non_object_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "demo" / "plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)

    plan_path.write_text("{", encoding="utf-8")
    invalid_tracker = _build_local_kernel_progress_tracker(base_dir=tmp_path, slug="demo")
    assert invalid_tracker.expected_folds is None
    assert invalid_tracker.expected_seeds == []

    plan_path.write_text("[]", encoding="utf-8")
    array_tracker = _build_local_kernel_progress_tracker(base_dir=tmp_path, slug="demo")
    assert array_tracker.expected_folds is None
    assert array_tracker.expected_seeds == []


def test_progress_tracker_reports_generic_activity(tmp_path: Path) -> None:
    plan_path = tmp_path / "demo" / "plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps({"cv_folds": 3, "eval_seeds": [42]}, indent=2), encoding="utf-8")
    tracker = _build_local_kernel_progress_tracker(base_dir=tmp_path, slug="demo")

    tracker.observe_line("[kernel] Running pipeline: tri_blend_stack")
    tracker.observe_line("[kernel] Pipeline tri_blend_stack: CV=0.123 method=weighted_mean_log")

    snapshot = tracker.snapshot()
    assert snapshot["lines_seen"] == 2
    assert snapshot["current_pipeline"] == "tri_blend_stack"
    assert snapshot["completed_pipeline_count"] == 1
    assert isinstance(snapshot["last_log_age_sec"], (int, float))
    assert "artifact_count" in snapshot
    assert "last_artifact_age_sec" in snapshot

    suffix = _format_local_kernel_activity_suffix(tracker)
    assert "logs=2" in suffix
    assert "pipeline=tri_blend_stack" in suffix
    assert "pipelines_done=1" in suffix
    assert "artifacts=" in suffix
    assert "last_artifact=" in suffix


def test_progress_tracker_reports_runtime_pipeline_suite_and_model(tmp_path: Path) -> None:
    tracker = _build_local_kernel_progress_tracker(base_dir=tmp_path, slug="demo")

    tracker.observe_line("Training pipeline: catboost_origstats_multiseed_fast__origw0800")
    tracker.observe_line("Suite: comp_plus_orig")
    tracker.observe_line("[kernel] train start: model=catboost.CatBoostClassifier rows=123 cols=45")
    tracker.observe_line("[kernel] CatBoost GPU failed; retrying on CPU: RuntimeError: CUDA out of memory")

    snapshot = tracker.snapshot()
    assert snapshot["current_pipeline"] == "catboost_origstats_multiseed_fast__origw0800"
    assert snapshot["current_suite"] == "comp_plus_orig"
    assert snapshot["current_model"] == "catboost.CatBoostClassifier"
    assert snapshot["last_fallback_reason"] == "RuntimeError: CUDA out of memory"

    suffix = _format_local_kernel_activity_suffix(tracker)
    assert "pipeline=catboost_origstats_multiseed_fast__origw0800" in suffix
    assert "suite=comp_plus_orig" in suffix
    assert "model=catboost.CatBoostClassifier" in suffix
    assert "fallback=RuntimeError: CUDA out of memory" in suffix


def test_progress_tracker_reports_artifact_activity(tmp_path: Path) -> None:
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir(parents=True, exist_ok=True)
    artifact = watch_dir / "metrics.json"
    artifact.write_text('{"ok":true}\n', encoding="utf-8")
    tracker = _build_local_kernel_progress_tracker(base_dir=tmp_path, slug="demo", watch_dirs=[watch_dir])

    snapshot = tracker.snapshot()
    assert int(snapshot["artifact_count"]) >= 1
    assert isinstance(snapshot["last_artifact_age_sec"], (int, float))

    suffix = _format_local_kernel_activity_suffix(tracker)
    assert "artifacts=" in suffix
    assert "last_artifact=" in suffix


def test_progress_tracker_ignores_stale_artifacts_then_counts_new_activity(tmp_path: Path) -> None:
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir(parents=True, exist_ok=True)
    stale_artifact = watch_dir / "submission.csv"
    stale_artifact.write_text("id,target\n1,0.0\n", encoding="utf-8")
    stale_mtime = time.time() - 60.0
    os.utime(stale_artifact, (stale_mtime, stale_mtime))

    tracker = _build_local_kernel_progress_tracker(base_dir=tmp_path, slug="demo", watch_dirs=[watch_dir])
    stale_snapshot = tracker.snapshot()
    assert stale_snapshot["artifact_count"] == 0
    assert stale_snapshot["last_artifact_age_sec"] is None

    fresh_artifact = watch_dir / "metrics.json"
    fresh_artifact.write_text('{"ok":true}\n', encoding="utf-8")

    fresh_snapshot = tracker.snapshot()
    assert int(fresh_snapshot["artifact_count"]) == 1
    assert isinstance(fresh_snapshot["last_artifact_age_sec"], (int, float))


def test_format_local_gpu_activity_suffix_handles_missing_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kagglebot.kernel_runner.shutil.which", lambda name: None)
    assert _format_local_gpu_activity_suffix(accelerator="gpu") == ""


def test_ensure_training_progress_shim_requires_marker(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    site_path = kernel_dir / "sitecustomize.py"
    site_path.write_text("# no marker\n", encoding="utf-8")

    with pytest.raises(KernelFailedError, match="mandatory progress logging"):
        _ensure_training_progress_shim(kernel_dir)
