"""Tests for kernel runner helpers."""

from __future__ import annotations

import base64
import gzip
import json
import os
import re
from pathlib import Path

import pytest

from kagglebot.exceptions import KernelFailedError
from kagglebot.kernel_runner import (
    _append_local_kernel_duration_history,
    _build_local_kernel_progress_tracker,
    _ensure_training_progress_shim,
    _estimate_local_kernel_duration_seconds,
    _extract_pipeline_done_from_line,
    _extract_pipeline_start_from_line,
    _extract_training_stage_from_line,
    _find_output_file,
    _format_local_gpu_activity_suffix,
    _format_local_kernel_activity_suffix,
    _resolve_fold_current,
    _resolve_seed_current,
    find_submission_file,
    resolve_kaggle_username,
    run_kernel,
    run_kernel_local,
    run_submit_kernel,
    sanitize_kernel_slug,
)


def test_sanitize_kernel_slug() -> None:
    assert sanitize_kernel_slug("KaggleBot Titan! 2024") == "kagglebot-titan-2024"


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


def test_resolve_kaggle_username_skips_invalid_json_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    (tmp_path / "kaggle.json").write_text("{invalid", encoding="utf-8")
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
    plan_path = tmp_path / "demo" / "plan.json"
    plan_path.write_text(json.dumps({"toggles": {"USE_MODEL": True}}, indent=2), encoding="utf-8")
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
    assert json.loads(staged_plan.read_text(encoding="utf-8")) == {"toggles": {"USE_MODEL": True}}


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

    payload_match = re.search(r"SUBMISSION_GZIP_B64 = \"([A-Za-z0-9+/=]+)\"", kernel_text)
    assert payload_match is not None
    encoded = payload_match.group(1)
    decoded = gzip.decompress(base64.b64decode(encoded.encode("ascii"))).decode("utf-8")
    assert decoded == submission_path.read_text(encoding="utf-8")

    payload = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert payload["competition_sources"] == ["demo"]
    assert payload["code_file"] == "kernel.py"


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


def test_find_submission_file(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    nested = output_dir / "nested"
    nested.mkdir()
    submission = nested / "submission.csv"
    submission.write_text("id,target\n1,0.1\n", encoding="utf-8")
    assert find_submission_file(output_dir) == submission


def test_find_submission_file_supports_zip_submission(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    submission = output_dir / "submission.zip"
    submission.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    assert find_submission_file(output_dir) == submission


def test_find_output_file_picks_newest_match(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    direct = output_dir / "metrics.json"
    direct.write_text('{"metric":"rmse"}\n', encoding="utf-8")
    nested = output_dir / "nested"
    nested.mkdir()
    newest = nested / "metrics.json"
    newest.write_text('{"metric":"rmse","offline_value":0.1}\n', encoding="utf-8")

    os.utime(direct, (1000, 1000))
    os.utime(newest, (2000, 2000))

    assert _find_output_file(output_dir, "metrics.json") == newest


def test_find_output_file_prefers_newest_under_run_tree(tmp_path: Path) -> None:
    root = tmp_path / "kernel-run"
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    (root / "runs" / "run_2").mkdir(parents=True, exist_ok=True)
    older = root / "outputs" / "metrics.json"
    newer = root / "runs" / "run_2" / "metrics.json"
    older.write_text('{"metric":"rmse"}\n', encoding="utf-8")
    newer.write_text('{"metric":"rmse","offline_value":0.1}\n', encoding="utf-8")

    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))

    assert _find_output_file(root, "metrics.json") == newer


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
    assert "kagglebot: train-progress-shim" in staged_sitecustomize.read_text(encoding="utf-8")
    staged_plan_local = tmp_path / "demo" / "kernels" / "run-5" / "local-iter-1" / "plan.json"
    staged_plan_parent = tmp_path / "demo" / "kernels" / "run-5" / "plan.json"
    assert staged_plan_local.exists()
    assert staged_plan_parent.exists()
    assert json.loads(staged_plan_local.read_text(encoding="utf-8")) == {"toggles": {"USE_MODEL": True}}
    assert json.loads(staged_plan_parent.read_text(encoding="utf-8")) == {"toggles": {"USE_MODEL": True}}


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


def test_local_kernel_duration_history_estimate_uses_recent_median(tmp_path: Path) -> None:
    for idx, duration in enumerate([100.0, 120.0, 80.0, 110.0], start=1):
        _append_local_kernel_duration_history(
            base_dir=tmp_path,
            slug="demo",
            run_id="run-a",
            iteration=idx,
            duration_sec=duration,
        )

    estimate, samples = _estimate_local_kernel_duration_seconds(base_dir=tmp_path, slug="demo")
    assert samples == 4
    assert estimate == 105.0


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
    estimate, samples = _estimate_local_kernel_duration_seconds(base_dir=tmp_path, slug="demo")
    assert samples == 1
    assert estimate is not None and estimate > 0.0


def test_extract_training_stage_from_line() -> None:
    inline = "[kernel] yolo_ensemble_wbf_geometry: seed=2024 fold=0 imgsz=768 epochs=250"
    assert _extract_training_stage_from_line(inline) == ("yolo_ensemble_wbf_geometry", 2024, 0)

    path_line = "/tmp/runs/yolo_ensemble_wbf_geometry_seed2024_fold1/weights/best.pt saved"
    assert _extract_training_stage_from_line(path_line) == ("yolo_ensemble_wbf_geometry", 2024, 1)


def test_extract_pipeline_progress_from_line() -> None:
    assert _extract_pipeline_start_from_line("[kernel] Running pipeline: tri_blend_stack") == "tri_blend_stack"
    assert (
        _extract_pipeline_done_from_line("[kernel] Pipeline tri_blend_stack: CV=0.125 method=weighted_mean_log")
        == "tri_blend_stack"
    )


def test_progress_helpers() -> None:
    assert _resolve_seed_current(seed=2024, expected_seeds=[42, 2024, 777]) == 2
    assert _resolve_seed_current(seed=999, expected_seeds=[42, 2024, 777]) is None
    assert _resolve_fold_current(fold_raw=0, expected_folds=5, zero_based=True) == 1
    assert _resolve_fold_current(fold_raw=2, expected_folds=5, zero_based=True) == 3
    assert _resolve_fold_current(fold_raw=2, expected_folds=5, zero_based=False) == 2


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
