import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from kagglebot import write_guard
from kagglebot.agents.identity import IMPLEMENTATION_AGENT, ORACLE_IMPLEMENTATION_AGENT
from kagglebot.exceptions import KaggleBotError
from kagglebot.orchestrator import agent_pipeline
from kagglebot.orchestrator.agent_pipeline import AgentPipelineConfig
from kagglebot.paths import CompetitionPaths


def test_oracle_strategy_uses_sol_xhigh_only_for_followup_implementation() -> None:
    assert agent_pipeline._implementation_agent_for_strategy_engine("oracle") is ORACLE_IMPLEMENTATION_AGENT
    assert agent_pipeline._implementation_agent_for_strategy_engine("auto") is ORACLE_IMPLEMENTATION_AGENT
    assert agent_pipeline._implementation_agent_for_strategy_engine("codex") is IMPLEMENTATION_AGENT


def test_implementation_failure_diagnostics_include_stdout_when_stderr_is_empty() -> None:
    result = SimpleNamespace(returncode=7, stdout="useful agent failure", stderr="")
    signaled_result = SimpleNamespace(returncode=-9, stdout="", stderr="")

    detail = agent_pipeline._format_agent_failure(result)
    signaled_detail = agent_pipeline._format_agent_failure(signaled_result)

    assert "useful agent failure" in detail
    assert "returncode=7" in detail
    assert detail.strip() != ""
    assert "returncode=-9 (SIGKILL)" in signaled_detail


def test_implementation_result_diagnostics_are_persisted(tmp_path: Path) -> None:
    result = SimpleNamespace(returncode=124, stdout="captured stdout", stderr="timed out after 30s")

    agent_pipeline._persist_implementation_result(tmp_path, result)

    assert (tmp_path / "implementation_stdout.txt").read_text(encoding="utf-8") == "captured stdout"
    assert (tmp_path / "implementation_stderr.txt").read_text(encoding="utf-8") == "timed out after 30s"
    status = json.loads((tmp_path / "implementation_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "timed_out"
    assert status["returncode"] == 124
    assert status["timed_out"] is True


def test_setup_plus_four_actions_is_rejected_and_clamped_to_three() -> None:
    issue = agent_pipeline._candidate_message_budget_issue(
        setup_message_count=1,
        action_message_count=4,
        max_messages_per_candidate=4,
    )
    effective = agent_pipeline._effective_action_message_count(
        requested_action_messages=4,
        setup_message_count=1,
        max_messages_per_candidate=4,
    )

    assert issue is not None
    assert "5 total" in issue
    assert effective == 3


def test_kernel_repair_prompt_requires_contract_safe_multipost_width(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.base_dir.mkdir(parents=True)
    paths.plan_path.write_text(
        '{"runtime_budget":{"max_messages_per_candidate":4},"pipelines":[]}',
        encoding="utf-8",
    )
    smoke = agent_pipeline.KernelContractSmokeResult(
        compile_returncode=0,
        compile_stdout="",
        compile_stderr="",
        smoke_returncode=1,
        smoke_stdout="",
        smoke_stderr="candidate 360 exceeds frozen plan message cap",
    )

    prompt = agent_pipeline._build_kernel_repair_prompt(
        paths=paths,
        original_prompt_path=tmp_path / "prompt.md",
        initial_failure="transport failed",
        smoke_result=smoke,
    )

    assert 'available_posts = max(0, int(CONFIG["max_messages_per_candidate"]) - 1)' in prompt
    assert "must emit only 3 action messages" in prompt
    assert "every selectable profile" in prompt
    assert "candidate 360 exceeds frozen plan message cap" in prompt


def test_missing_frozen_plan_pipeline_lookup_is_diagnosed(tmp_path: Path) -> None:
    kernel_path = tmp_path / "kernel.py"
    plan_path = tmp_path / "plan.json"
    kernel_path.write_text(
        '# _pipeline_hyperparameters("commented_out_pipeline")\ncfg = _pipeline_hyperparameters("obsolete_pipeline")\n',
        encoding="utf-8",
    )
    plan_path.write_text(
        '{"pipelines":[{"name":"actual_pipeline","key_hyperparameters":{}}]}',
        encoding="utf-8",
    )

    issues = agent_pipeline._diagnose_missing_pipeline_lookups(kernel_path, plan_path)

    assert len(issues) == 1
    assert "obsolete_pipeline" in issues[0]
    assert "commented_out_pipeline" not in issues[0]
    assert "actual_pipeline" in issues[0]


def test_inference_server_submit_runtime_repair_avoids_local_gateway(tmp_path: Path) -> None:
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(
        """\
import os

def _submit_notebook_mode():
    return True

def write_fallback_submission():
    return "submission.csv"

def validate_code_competition_submission(path):
    return None

def maybe_start_inference_server(sdk_root=None):
    submit_mode = _submit_notebook_mode()
    if not os.getenv("KAGGLE_IS_COMPETITION_RERUN") and not submit_mode:
        submission_path = write_fallback_submission()
        validate_code_competition_submission(submission_path)
        return

    import kaggle_evaluation.demo.inference_server as server

    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        server.DemoInferenceServer().serve()
        return

    if submit_mode:
        server.DemoInferenceServer().run_local_gateway(data_paths=(str(sdk_root),))
        validate_code_competition_submission("submission.csv")
        return
""",
        encoding="utf-8",
    )

    assert agent_pipeline._apply_inference_server_submit_runtime_repair(kernel_path) is True

    repaired = kernel_path.read_text(encoding="utf-8")
    assert "run_local_gateway" not in repaired
    assert repaired.count("server.DemoInferenceServer().serve()") == 1
    assert "submission_path = write_fallback_submission()" in repaired
    assert repaired.index("import kaggle_evaluation") < repaired.index("server.DemoInferenceServer().serve()")
    assert repaired.index("server.DemoInferenceServer().serve()") < repaired.index(
        'if os.getenv("KAGGLE_IS_COMPETITION_RERUN") is None'
    )
    assert agent_pipeline._apply_inference_server_submit_runtime_repair(kernel_path) is False


def test_inference_server_submit_runtime_repair_upgrades_legacy_hidden_only_server(tmp_path: Path) -> None:
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(
        """\
import os

def write_fallback_submission():
    return "submission.csv"

def validate_code_competition_submission(path):
    return None

def maybe_start_inference_server(sdk_root=None):
    if not os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        submission_path = write_fallback_submission()
        validate_code_competition_submission(submission_path)
        return

    import kaggle_evaluation.demo.inference_server as server

    server.DemoInferenceServer().serve()
    return
""",
        encoding="utf-8",
    )

    assert agent_pipeline._apply_inference_server_submit_runtime_repair(kernel_path) is True

    repaired = kernel_path.read_text(encoding="utf-8")
    assert repaired.index("server.DemoInferenceServer().serve()") < repaired.index(
        "submission_path = write_fallback_submission()"
    )
    assert 'if os.getenv("KAGGLE_IS_COMPETITION_RERUN") is None' in repaired
    assert agent_pipeline._apply_inference_server_submit_runtime_repair(kernel_path) is False


def test_embedded_attack_profile_dispatch_drift_is_diagnosed(tmp_path: Path) -> None:
    kernel_path = tmp_path / "kernel.py"
    plan_path = tmp_path / "plan.json"
    kernel_path.write_text(
        '''\
ATTACK_SOURCE = r"""
def build_profile_messages(profile=None):
    resolved_profile = profile or "actual_pipeline"
    if resolved_profile == "actual_pipeline":
        return []
    if resolved_profile == "obsolete_pipeline":
        return []
    return build_profile_messages("obsolete_fallback")
"""
''',
        encoding="utf-8",
    )
    plan_path.write_text(
        '{"pipelines":[{"name":"actual_pipeline","key_hyperparameters":{}}]}',
        encoding="utf-8",
    )

    issues = agent_pipeline._diagnose_missing_pipeline_lookups(kernel_path, plan_path)

    assert len(issues) == 1
    assert "generated attack dispatch" in issues[0]
    assert "obsolete_pipeline" in issues[0]
    assert "obsolete_fallback" in issues[0]
    assert "actual_pipeline" in issues[0]


def test_kernel_contract_smoke_injects_data_free_self_test_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = CompetitionPaths(
        slug="arc-prize-2026-arc-agi-2",
        artifacts_dir=tmp_path / "artifacts",
    )
    paths.kernel_source_dir.mkdir(parents=True)
    paths.data_dir.mkdir(parents=True)
    (paths.data_dir / "dataset.json").write_text("not for contract smoke", encoding="utf-8")
    kernel_path = paths.kernel_source_dir / "kernel.py"
    kernel_path.write_text(
        """\
import json
import os
from pathlib import Path

if os.environ.get("ARC_SELF_TEST") != "1":
    raise FileNotFoundError("normal entrypoint requires competition data")
assert os.environ.get("FAST_DEV") == "0"
assert "ARC_DATA_DIR" not in os.environ
assert not (Path.cwd().parent / "plan.json").exists()
assert not any((Path.cwd().parent / "data").iterdir())
output_dir = Path("outputs")
output_dir.mkdir()
(output_dir / "self_test_results.json").write_text(
    json.dumps({"status": "passed"}), encoding="utf-8"
)
print("isolated contract self-test passed")
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(agent_pipeline, "_diagnose_missing_pipeline_lookups", lambda *args: ())
    monkeypatch.setenv("ARC_SELF_TEST", "parent-value")
    monkeypatch.setenv("FAST_DEV", "parent-value")
    monkeypatch.setenv("ARC_DATA_DIR", str(paths.data_dir))

    results = [
        agent_pipeline._run_kernel_contract_smoke(paths=paths, kernel_path=kernel_path) for _ in ("initial", "repaired")
    ]

    assert all(result.passed for result in results)
    assert all("isolated contract self-test passed" in result.smoke_stdout for result in results)
    assert (paths.data_dir / "dataset.json").read_text(encoding="utf-8") == "not for contract smoke"
    assert os.environ["ARC_SELF_TEST"] == "parent-value"
    assert os.environ["FAST_DEV"] == "parent-value"
    assert os.environ["ARC_DATA_DIR"] == str(paths.data_dir)


def _write_structured_contract_kernel(
    kernel_path: Path,
    *,
    backward_finite: bool = True,
    contract_exception: str | None = None,
    contract_output_arg: bool = False,
    data_free: bool = True,
    malformed_report: bool = False,
    normal_mode: str = "missing_data",
    logit_classes: int = 3,
    training_performed: bool = False,
    require_smoke_env_aliases: bool = False,
) -> None:
    exception_line = f'raise RuntimeError("{contract_exception}")' if contract_exception else ""
    contract_signature = "output_dir=None" if contract_output_arg else ""
    output_resolution = (
        'output_dir = Path(output_dir or os.environ["KAGGLEBOT_OUTPUT_DIR"])'
        if contract_output_arg
        else 'output_dir = Path(os.environ["KAGGLEBOT_OUTPUT_DIR"])'
    )
    normal_lines = {
        "missing_data": [
            'print(json.dumps({"status": "blocked", "error_type": "DataDiscoveryError", '
            '"reason_code": "missing_raw_training_assets", '
            '"message": "Raw labeled training assets were not found. Supply the competition data root through '
            'KAGGLE_INPUT_DIR, DATA_ROOT, or CUHKX_DATA_ROOT.", "submission_written": False}), file=sys.stderr)',
            "raise SystemExit(2)",
        ],
        "observed_data_contract_missing": [
            'print(json.dumps({"status": "blocked", "error_type": "DataContractError", '
            '"message": "No competition data root found. Supply test.csv plus raw HAR training data through '
            'KAGGLE_INPUT_DIR, DATA_ROOT, or CUHKX_DATA_ROOT.", "submission_written": False}), file=sys.stderr)',
            "raise SystemExit(2)",
        ],
        "runtime_error": ['raise RuntimeError("unrelated normal-entrypoint failure")'],
        "spoofed_runtime_error": [
            'raise RuntimeError("DataDiscoveryError: Raw labeled training assets were not found")'
        ],
        "success": ["return"],
        "sample_copy": [
            'output_dir = Path(os.environ["KAGGLEBOT_OUTPUT_DIR"])',
            'output_dir.joinpath("submission.csv").write_text("id,prediction\\n1,0\\n", encoding="utf-8")',
        ],
    }[normal_mode]
    if require_smoke_env_aliases:
        normal_lines = [
            'assert os.environ["KAGGLEBOT_FAST_DEV"] == "1"',
            'assert os.environ["FAST_DEV"] == "1"',
            'assert os.environ["KAGGLEBOT_VALIDATION_MAX_SAMPLES"] == "2"',
            'assert os.environ["VALIDATION_MAX_SAMPLES"] == "2"',
            'assert os.environ["HF_HUB_OFFLINE"] == "1"',
            'assert os.environ["HF_DATASETS_OFFLINE"] == "1"',
            'assert os.environ["TRANSFORMERS_OFFLINE"] == "1"',
            'assert os.environ["KB_ALLOW_MODEL_DOWNLOAD"] == "0"',
            'assert os.environ["KAGGLEBOT_IO_SCHEMA_SMOKE"] == "1"',
            *normal_lines,
        ]
    normal_body = "\n    ".join(normal_lines)
    report_payload = '"not valid json"' if malformed_report else "json.dumps(report)"
    kernel_path.write_text(
        f"""\
import json
import os
import sys
from pathlib import Path

class DataDiscoveryError(RuntimeError):
    pass

def contract_smoke({contract_signature}):
    {exception_line or "pass"}
    {output_resolution}
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {{
        "status": "passed",
        "data_free": {data_free!r},
        "training_performed": {training_performed!r},
        "score_reported": False,
        "profiles": {{
            "planned_pipeline": {{
                "forward_finite": True,
                "backward_finite": {backward_finite!r},
                "deploy_bytes": 1024,
                "logit_shape": [2, {logit_classes}],
            }}
        }},
    }}
    (output_dir / "contract_smoke.json").write_text({report_payload}, encoding="utf-8")

def main():
    {normal_body}

if __name__ == "__main__":
    main()
""",
        encoding="utf-8",
    )


def _structured_contract_paths(
    tmp_path: Path,
    *,
    data_ready: bool,
    deliverable_mode: str = "leaderboard",
    unavailable_profile_status: str = "missing_required_files",
) -> tuple[CompetitionPaths, Path]:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.kernel_source_dir.mkdir(parents=True)
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.data_dir.mkdir(parents=True)
    paths.plan_path.write_text(
        json.dumps(
            {
                "deliverable_mode": deliverable_mode,
                "pipelines": [{"name": "planned_pipeline"}],
                "toggles": {"STRICT_MODEL_SIZE_MB": 100},
            }
        ),
        encoding="utf-8",
    )
    profile = {"status": unavailable_profile_status}
    if data_ready:
        (paths.data_dir / "train.csv").write_text("feature,target\n1,0\n", encoding="utf-8")
        profile = {"status": "ok", "train_file": "train.csv"}
    paths.dataset_profile_path.write_text(json.dumps(profile), encoding="utf-8")
    return paths, paths.kernel_source_dir / "kernel.py"


def test_missing_data_accepts_contract_and_expected_full_entrypoint_block(tmp_path: Path) -> None:
    paths, kernel_path = _structured_contract_paths(
        tmp_path,
        data_ready=False,
        deliverable_mode="writeup",
    )
    _write_structured_contract_kernel(kernel_path)

    result = agent_pipeline._run_kernel_contract_smoke(paths=paths, kernel_path=kernel_path)

    assert result.passed is True
    assert result.contract_only is True
    assert result.normal_smoke_required is True
    assert result.normal_smoke_returncode == 2
    assert "DataDiscoveryError" in result.normal_smoke_stderr
    assert "Raw labeled training assets were not found" in result.normal_smoke_stderr
    assert not result.normal_smoke_issues
    assert not list(paths.base_dir.rglob("submission.csv"))
    assert not list(paths.base_dir.rglob("metrics.json"))
    assert not list(paths.base_dir.rglob("oof_*.npy"))
    assert not list(paths.base_dir.rglob("*.pth"))


def test_truthy_readiness_result_with_unavailable_status_accepts_expected_block(
    tmp_path: Path,
) -> None:
    paths, kernel_path = _structured_contract_paths(
        tmp_path,
        data_ready=False,
        deliverable_mode="writeup",
        unavailable_profile_status="non_tabular_data",
    )
    _write_structured_contract_kernel(kernel_path)

    readiness = agent_pipeline.assess_local_training_data(paths)
    result = agent_pipeline._run_kernel_contract_smoke(paths=paths, kernel_path=kernel_path)

    assert bool(readiness) is True
    assert readiness.ready is False
    assert readiness.reason == "labeled_training_source_missing"
    assert result.passed is True
    assert result.contract_only is True
    assert result.data_ready is False
    assert result.data_readiness_reason == "labeled_training_source_missing"
    assert not result.normal_smoke_issues
    formatted = agent_pipeline._format_kernel_contract_smoke(result)
    assert "training data readiness: unavailable (labeled_training_source_missing)" in formatted
    assert "blocked_missing_competition_training_data (expected)" in formatted


def test_observed_data_contract_block_is_rejected_when_training_data_is_unavailable(
    tmp_path: Path,
) -> None:
    paths, kernel_path = _structured_contract_paths(
        tmp_path,
        data_ready=False,
        deliverable_mode="writeup",
    )
    _write_structured_contract_kernel(
        kernel_path,
        normal_mode="observed_data_contract_missing",
    )

    result = agent_pipeline._run_kernel_contract_smoke(paths=paths, kernel_path=kernel_path)

    assert result.passed is False
    assert result.contract_only is False
    assert result.normal_smoke_returncode == 2
    issues = "\n".join(result.normal_smoke_issues)
    assert "error_type must be DataDiscoveryError" in issues
    assert "reason_code must be 'missing_raw_training_assets'" in issues


def test_missing_data_probe_rejects_observed_structured_data_contract_block(tmp_path: Path) -> None:
    payload = {
        "status": "blocked",
        "error_type": "DataContractError",
        "message": (
            "No competition data root found. Supply test.csv plus raw HAR training data through "
            "KAGGLE_INPUT_DIR, DATA_ROOT, or CUHKX_DATA_ROOT."
        ),
        "submission_written": False,
    }
    stderr = "diagnostic emitted before the final structured block\n" + json.dumps(payload)

    issues = agent_pipeline._missing_data_probe_issues(
        returncode=2,
        stderr=stderr,
        staging_root=tmp_path,
        training_data_unavailable=True,
    )

    rendered = "\n".join(issues)
    assert "error_type must be DataDiscoveryError" in rendered
    assert "reason_code must be 'missing_raw_training_assets'" in rendered


@pytest.mark.parametrize(
    ("stderr", "expected_issue"),
    [
        (
            "Transparent stop: raw labeled training assets were not found\n",
            "does not end with a readable JSON object",
        ),
        (
            '{"status":"blocked","error_type":"DataContractError",'
            '"message":"HAR/data label schema mismatch","submission_written":false}\n',
            "does not identify unavailable training input",
        ),
        (
            '{"status":"blocked","error_type":"DataContractError",'
            '"message":"Training labels are unavailable because schema validation failed",'
            '"submission_written":false}\n',
            "does not identify unavailable training input",
        ),
    ],
)
def test_missing_data_probe_rejects_unstructured_or_unrelated_blocks(
    tmp_path: Path,
    stderr: str,
    expected_issue: str,
) -> None:
    issues = agent_pipeline._missing_data_probe_issues(
        returncode=2,
        stderr=stderr,
        staging_root=tmp_path,
        training_data_unavailable=True,
    )

    assert expected_issue in "\n".join(issues)


def test_missing_data_probe_rejects_incomplete_final_stderr_json(tmp_path: Path) -> None:
    issues = agent_pipeline._missing_data_probe_issues(
        returncode=2,
        stderr='"message":"raw labeled training assets were not found","submission_written":false}',
        staging_root=tmp_path,
        training_data_unavailable=True,
    )

    assert "readable JSON object" in "\n".join(issues)


@pytest.mark.parametrize("artifact_name", ["submission.csv", "submission.zip"])
def test_missing_data_probe_rejects_submission_claim_or_artifact(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    payload = {
        "status": "blocked",
        "error_type": "DataDiscoveryError",
        "reason_code": "missing_raw_training_assets",
        "message": "Raw labeled training assets were not found.",
        "submission_written": True,
    }

    claimed_issues = agent_pipeline._missing_data_probe_issues(
        returncode=2,
        stderr=json.dumps(payload),
        staging_root=tmp_path,
        training_data_unavailable=True,
    )

    assert "submission_written=false" in "\n".join(claimed_issues)

    payload["submission_written"] = False
    (tmp_path / artifact_name).write_text("probe artifact", encoding="utf-8")
    artifact_issues = agent_pipeline._missing_data_probe_issues(
        returncode=2,
        stderr=json.dumps(payload),
        staging_root=tmp_path,
        training_data_unavailable=True,
    )

    assert "created prohibited" in "\n".join(artifact_issues)
    assert artifact_name in "\n".join(artifact_issues)


@pytest.mark.parametrize(
    ("returncode", "stdout", "reason_code", "expected_issue"),
    [
        (1, "", "missing_raw_training_assets", "expected return code 2"),
        (2, "unexpected output\n", "missing_raw_training_assets", "must not emit stdout"),
        (2, "", "wrong_reason", "reason_code must be 'missing_raw_training_assets'"),
    ],
)
def test_missing_data_probe_rejects_adversarial_variants(
    tmp_path: Path,
    returncode: int,
    stdout: str,
    reason_code: str,
    expected_issue: str,
) -> None:
    payload = {
        "status": "blocked",
        "error_type": "DataDiscoveryError",
        "reason_code": reason_code,
        "message": "Raw labeled training assets were not found.",
        "submission_written": False,
    }

    issues = agent_pipeline._missing_data_probe_issues(
        returncode=returncode,
        stdout=stdout,
        stderr=json.dumps(payload),
        staging_root=tmp_path,
        training_data_unavailable=True,
    )

    assert expected_issue in "\n".join(issues)


def test_missing_data_probe_rejects_block_when_training_data_is_ready(tmp_path: Path) -> None:
    payload = {
        "status": "blocked",
        "error_type": "DataDiscoveryError",
        "reason_code": "missing_raw_training_assets",
        "message": "Raw labeled training assets were not found.",
        "submission_written": False,
    }

    issues = agent_pipeline._missing_data_probe_issues(
        returncode=2,
        stderr=json.dumps(payload),
        staging_root=tmp_path,
        training_data_unavailable=False,
    )

    assert "training data readiness is available" in "\n".join(issues)


def test_required_local_training_missing_data_is_contract_only(
    tmp_path: Path,
) -> None:
    paths, kernel_path = _structured_contract_paths(
        tmp_path,
        data_ready=False,
        deliverable_mode="leaderboard",
    )
    plan = json.loads(paths.plan_path.read_text(encoding="utf-8"))
    plan["runtime_budget"] = {"local_training_required": True}
    paths.plan_path.write_text(json.dumps(plan), encoding="utf-8")
    _write_structured_contract_kernel(kernel_path)

    result = agent_pipeline._run_kernel_contract_smoke(
        paths=paths,
        kernel_path=kernel_path,
    )

    assert result.passed is True
    assert result.contract_only is True
    assert result.allow_missing_training_data is True
    assert result.data_readiness_reason == ("dataset_profile_missing_required_files")


def test_contract_only_acceptance_promotes_kernel_and_records_environmental_blocker(tmp_path: Path) -> None:
    paths, kernel_path = _structured_contract_paths(tmp_path, data_ready=False)
    candidate_path = tmp_path / "candidate.py"
    candidate_path.write_text("print('validated candidate')\n", encoding="utf-8")
    smoke = agent_pipeline.KernelContractSmokeResult(
        compile_returncode=0,
        compile_stdout="",
        compile_stderr="",
        smoke_returncode=0,
        smoke_stdout="contract passed",
        smoke_stderr="",
        contract_report={
            "status": "passed",
            "data_free": True,
            "training_performed": False,
            "score_reported": False,
        },
        data_ready=False,
        data_readiness_reason="dataset_profile_missing_required_files",
        allow_missing_training_data=True,
        normal_smoke_required=True,
        normal_smoke_returncode=2,
        normal_smoke_stderr=(
            '{"status":"blocked","error_type":"DataDiscoveryError",'
            '"reason_code":"missing_raw_training_assets",'
            '"message":"Raw labeled training assets were not found.",'
            '"submission_written":false}'
        ),
    )

    agent_pipeline._accept_validated_kernel(
        paths=paths,
        run_id="run-1",
        candidate_path=candidate_path,
        kernel_path=kernel_path,
        smoke_result=smoke,
    )

    assert kernel_path.read_text(encoding="utf-8") == "print('validated candidate')\n"
    verification = json.loads((paths.run_dir("run-1") / "implementation_verification.json").read_text(encoding="utf-8"))
    assert verification["status"] == "passed_contract_only"
    assert verification["blocked_reason"] == "missing_competition_data"
    assert verification["normal_smoke_performed"] is True
    assert verification["full_run_ok"] is False
    assert verification["data_available"] is False
    assert verification["training_performed"] is False
    assert verification["score_reported"] is False


def test_missing_data_does_not_hide_failing_contract_report(tmp_path: Path) -> None:
    paths, kernel_path = _structured_contract_paths(tmp_path, data_ready=False)
    _write_structured_contract_kernel(kernel_path, backward_finite=False)

    result = agent_pipeline._run_kernel_contract_smoke(paths=paths, kernel_path=kernel_path)

    assert result.passed is False
    assert "finite_backward=true" in "\n".join(result.contract_report_issues)


def test_missing_data_remains_fatal_for_conventional_prediction_route(tmp_path: Path) -> None:
    paths, kernel_path = _structured_contract_paths(tmp_path, data_ready=False)
    _write_structured_contract_kernel(kernel_path)

    result = agent_pipeline._run_kernel_contract_smoke(paths=paths, kernel_path=kernel_path)

    assert result.passed is False
    assert result.contract_only is False
    assert result.allow_missing_training_data is False
    assert result.data_readiness_reason == "dataset_profile_missing_required_files"


@pytest.mark.parametrize(
    ("contract_kwargs", "expected_issue"),
    [
        ({"data_free": False}, "data_free=true"),
        ({"training_performed": True}, "training_performed=false"),
    ],
)
def test_missing_data_remains_blocking_for_non_data_free_or_training_contract(
    tmp_path: Path,
    contract_kwargs: dict[str, bool],
    expected_issue: str,
) -> None:
    paths, kernel_path = _structured_contract_paths(tmp_path, data_ready=False)
    _write_structured_contract_kernel(kernel_path, **contract_kwargs)

    result = agent_pipeline._run_kernel_contract_smoke(paths=paths, kernel_path=kernel_path)

    assert result.passed is False
    assert result.contract_only is False
    assert result.normal_smoke_required is False
    assert expected_issue in "\n".join(result.contract_report_issues)


def test_contract_report_requires_exact_profiles_and_plan_class_count(tmp_path: Path) -> None:
    paths, _ = _structured_contract_paths(tmp_path, data_ready=False)
    paths.plan_path.write_text(
        json.dumps(
            {
                "pipelines": [
                    {
                        "name": "planned_pipeline",
                        "models": ["reference encoder with a 40-class linear head"],
                    }
                ],
                "toggles": {"STRICT_MODEL_SIZE_MB": 100},
            }
        ),
        encoding="utf-8",
    )
    report = {
        "status": "passed",
        "data_free": True,
        "training_performed": False,
        "score_reported": False,
        "profiles": {
            "planned_pipeline": {
                "forward_finite": True,
                "backward_finite": True,
                "deploy_bytes": 1024,
                "logit_shape": [2, 3],
            },
            "unexpected_pipeline": {
                "forward_finite": True,
                "backward_finite": True,
                "deploy_bytes": 1024,
                "logit_shape": [2, 40],
            },
        },
    }

    issues = agent_pipeline._validate_contract_smoke_report(
        report,
        plan_path=paths.plan_path,
        pipeline_names=("planned_pipeline",),
    )

    assert "unexpected_pipeline" in "\n".join(issues)
    assert "40 output classes" in "\n".join(issues)


def _profiled_contract_plan(plan_path: Path) -> None:
    plan_path.write_text(
        json.dumps(
            {
                "deliverable_mode": "writeup",
                "hardware_profile": "rtx3060",
                "runtime_budget": {
                    "hardware_profile": "rtx3060",
                    "scale_profiles": {"rtx3060": {}, "rtx5090": {}},
                },
                "pipelines": [
                    {
                        "name": "planned_pipeline",
                        "models": ["reference encoder with a 40-class linear head"],
                    }
                ],
                "toggles": {"STRICT_MODEL_SIZE_MB": 100},
            }
        ),
        encoding="utf-8",
    )


def _canonical_contract_report(*, entry_profile: bool = False) -> dict[str, object]:
    pipeline = {
        "finite_forward": True,
        "finite_backward": True,
        "loss": 3.5,
        "logits_shape": [2, 40],
        "deploy_bytes": 1024,
    }
    if entry_profile:
        pipeline["profile"] = "rtx3060"
    return {
        "status": "passed",
        "profile": "local_gpu",
        "top_level_knobs": {"HARDWARE_PROFILE": "rtx3060"},
        "data_free": True,
        "training_performed": False,
        "score_reported": False,
        "pipelines": {"planned_pipeline": pipeline},
    }


@pytest.mark.parametrize("compute_profile", ["local_gpu", "kaggle_gpu", "kaggle_tpu"])
@pytest.mark.parametrize("entry_profile", [False, True])
def test_contract_report_accepts_independent_compute_and_hardware_profiles(
    tmp_path: Path,
    compute_profile: str,
    entry_profile: bool,
) -> None:
    plan_path = tmp_path / "plan.json"
    _profiled_contract_plan(plan_path)
    report = _canonical_contract_report(entry_profile=entry_profile)
    report["profile"] = compute_profile

    warnings: list[str] = []
    issues = agent_pipeline._validate_contract_smoke_report(
        report,
        plan_path=plan_path,
        pipeline_names=("planned_pipeline",),
        expected_compute_profile=compute_profile,
        warnings=warnings,
    )

    assert issues == ()
    assert "interpreted as compute_mode" in "\n".join(warnings)
    if entry_profile:
        assert "interpreted as hardware_profile" in "\n".join(warnings)


def test_contract_report_rejects_hardware_environment_that_conflicts_with_frozen_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _profiled_contract_plan(plan_path)
    report = _canonical_contract_report(entry_profile=True)
    report["top_level_knobs"]["HARDWARE_PROFILE"] = "rtx5090"
    report["pipelines"]["planned_pipeline"]["hardware_profile"] = "rtx5090"

    issues = agent_pipeline._validate_contract_smoke_report(
        report,
        plan_path=plan_path,
        pipeline_names=("planned_pipeline",),
        expected_compute_profile="local_gpu",
        smoke_environment={"KAGGLEBOT_HARDWARE_PROFILE": "rtx3060"},
    )

    assert "contract_smoke.json hardware_profile='rtx5090'; expected='rtx3060'" in issues
    assert "pipelines.planned_pipeline.hardware_profile='rtx5090'; expected='rtx3060'" in issues


def test_contract_smoke_runner_injects_and_validates_independent_profiles(tmp_path: Path) -> None:
    paths, kernel_path = _structured_contract_paths(
        tmp_path,
        data_ready=False,
        deliverable_mode="writeup",
    )
    _profiled_contract_plan(paths.plan_path)
    kernel_path.write_text(
        """\
import json
import os
import sys
from pathlib import Path

class DataDiscoveryError(RuntimeError):
    pass

def contract_smoke(output_dir=None):
    output_dir = Path(output_dir or os.environ["KAGGLEBOT_OUTPUT_DIR"])
    compute_profile = os.environ["KAGGLEBOT_COMPUTE_PROFILE"]
    hardware_profile = os.environ["KAGGLEBOT_HARDWARE_PROFILE"]
    report = {
        "status": "passed",
        "profile": compute_profile,
        "top_level_knobs": {"HARDWARE_PROFILE": hardware_profile},
        "data_free": True,
        "training_performed": False,
        "score_reported": False,
        "pipelines": {
            "planned_pipeline": {
                "profile": hardware_profile,
                "finite_forward": True,
                "finite_backward": True,
                "logits_shape": [2, 40],
                "deploy_bytes": 1024,
            }
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("contract_smoke.json").write_text(json.dumps(report), encoding="utf-8")

def main():
    print(
        json.dumps({
            "status": "blocked",
            "error_type": "DataDiscoveryError",
            "reason_code": "missing_raw_training_assets",
            "message": "Raw labeled training assets were not found.",
            "submission_written": False,
        }),
        file=sys.stderr,
    )
    raise SystemExit(2)

if __name__ == "__main__":
    main()
""",
        encoding="utf-8",
    )

    result = agent_pipeline._run_kernel_contract_smoke(
        paths=paths,
        kernel_path=kernel_path,
        expected_compute_profile="kaggle_gpu",
        expected_hardware_profile="rtx3060",
    )

    assert result.passed is True
    assert result.contract_only is True
    assert result.contract_report_issues == ()
    assert result.contract_report["profile"] == "kaggle_gpu"
    assert result.contract_report["top_level_knobs"]["HARDWARE_PROFILE"] == "rtx3060"
    assert result.contract_report["pipelines"]["planned_pipeline"]["profile"] == "rtx3060"
    assert "permits for contract-only verification" in "\n".join(result.warnings)


def test_contract_report_rejects_missing_conflicting_nonfinite_and_oversized_pipeline(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    _profiled_contract_plan(plan_path)
    reports = []

    missing = _canonical_contract_report()
    missing["pipelines"] = {}
    reports.append((missing, "missing=['planned_pipeline']"))

    conflicting = _canonical_contract_report(entry_profile=True)
    conflicting["pipelines"]["planned_pipeline"]["profile"] = "rtx5090"
    reports.append((conflicting, "hardware_profile='rtx5090'; expected='rtx3060'"))

    nonfinite = _canonical_contract_report()
    nonfinite["pipelines"]["planned_pipeline"]["finite_backward"] = False
    reports.append((nonfinite, "finite_backward=true"))

    oversized = _canonical_contract_report()
    oversized["pipelines"]["planned_pipeline"]["deploy_bytes"] = 100 * 1024 * 1024
    reports.append((oversized, "exceeds frozen limit"))

    zero_deploy = _canonical_contract_report()
    zero_deploy["pipelines"]["planned_pipeline"]["deploy_bytes"] = 0
    reports.append((zero_deploy, "invalid deploy_bytes"))

    wrong_logit_shape = _canonical_contract_report()
    wrong_logit_shape["pipelines"]["planned_pipeline"]["logits_shape"] = [1, 40]
    reports.append((wrong_logit_shape, "invalid logits_shape"))

    not_data_free = _canonical_contract_report()
    not_data_free.pop("data_free")
    reports.append((not_data_free, "data_free=true"))

    training_performed = _canonical_contract_report()
    training_performed["training_performed"] = True
    reports.append((training_performed, "training_performed=false"))

    score_reported = _canonical_contract_report()
    score_reported["score_reported"] = True
    reports.append((score_reported, "score_reported=false"))

    for report, expected_issue in reports:
        issues = agent_pipeline._validate_contract_smoke_report(
            report,
            plan_path=plan_path,
            pipeline_names=("planned_pipeline",),
        )
        assert expected_issue in "\n".join(issues)


def test_contract_report_accepts_legacy_pipeline_compute_and_inherits_hardware(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _profiled_contract_plan(plan_path)
    report = _canonical_contract_report(entry_profile=True)
    report["pipelines"]["planned_pipeline"]["profile"] = "local_gpu"
    warnings: list[str] = []

    issues = agent_pipeline._validate_contract_smoke_report(
        report,
        plan_path=plan_path,
        pipeline_names=("planned_pipeline",),
        expected_compute_profile="local_gpu",
        warnings=warnings,
    )

    assert issues == ()
    assert "pipelines.planned_pipeline.profile='local_gpu' is legacy; interpreted as compute_mode" in warnings


def test_legacy_pipeline_compute_requires_reported_top_level_hardware(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _profiled_contract_plan(plan_path)
    report = _canonical_contract_report(entry_profile=True)
    report.pop("top_level_knobs")
    report["pipelines"]["planned_pipeline"]["profile"] = "local_gpu"

    issues = agent_pipeline._validate_contract_smoke_report(
        report,
        plan_path=plan_path,
        pipeline_names=("planned_pipeline",),
        expected_compute_profile="local_gpu",
    )

    assert "pipelines.planned_pipeline.hardware_profile is missing; expected='rtx3060'" in issues


def test_contract_report_accepts_legacy_pipeline_hardware_and_inherits_compute(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _profiled_contract_plan(plan_path)
    report = _canonical_contract_report(entry_profile=True)
    warnings: list[str] = []

    issues = agent_pipeline._validate_contract_smoke_report(
        report,
        plan_path=plan_path,
        pipeline_names=("planned_pipeline",),
        expected_compute_profile="local_gpu",
        warnings=warnings,
    )

    assert issues == ()
    assert "pipelines.planned_pipeline.profile='rtx3060' is legacy; interpreted as hardware_profile" in warnings


def test_contract_report_accepts_explicit_profile_schema_without_legacy_warning(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _profiled_contract_plan(plan_path)
    report = _canonical_contract_report()
    report.pop("profile")
    report["compute_mode"] = "local_gpu"
    report["hardware_profile"] = "rtx3060"
    report.pop("top_level_knobs")
    report["pipelines"]["planned_pipeline"].update(
        compute_mode="local_gpu",
        hardware_profile="rtx3060",
    )
    warnings: list[str] = []

    issues = agent_pipeline._validate_contract_smoke_report(
        report,
        plan_path=plan_path,
        pipeline_names=("planned_pipeline",),
        expected_compute_profile="local_gpu",
        warnings=warnings,
    )

    assert issues == ()
    assert warnings == []


def test_contract_report_rejects_mismatched_compute_mode(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _profiled_contract_plan(plan_path)
    report = _canonical_contract_report()
    report["profile"] = "kaggle_gpu"

    issues = agent_pipeline._validate_contract_smoke_report(
        report,
        plan_path=plan_path,
        pipeline_names=("planned_pipeline",),
        expected_compute_profile="local_gpu",
    )

    assert "contract_smoke.json compute_mode='kaggle_gpu'; expected='local_gpu'" in issues


def test_contract_report_rejects_unknown_legacy_profile(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _profiled_contract_plan(plan_path)
    report = _canonical_contract_report(entry_profile=True)
    report["pipelines"]["planned_pipeline"]["profile"] = "mystery_profile"

    issues = agent_pipeline._validate_contract_smoke_report(
        report,
        plan_path=plan_path,
        pipeline_names=("planned_pipeline",),
        expected_compute_profile="local_gpu",
    )

    assert "pipelines.planned_pipeline.profile='mystery_profile' is unknown" in issues


def test_contract_report_normalizes_legacy_alias_and_rejects_disagreement(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _profiled_contract_plan(plan_path)
    report = _canonical_contract_report()
    report["profiles"] = {
        "planned_pipeline": {
            "profile": "rtx3060",
            "forward_finite": True,
            "backward_finite": True,
            "loss": 3.5,
            "logit_shape": [2, 40],
            "deploy_bytes": 1024,
        }
    }

    matching_issues = agent_pipeline._validate_contract_smoke_report(
        report,
        plan_path=plan_path,
        pipeline_names=("planned_pipeline",),
    )
    report["profiles"]["planned_pipeline"]["deploy_bytes"] = 2048
    conflicting_issues = agent_pipeline._validate_contract_smoke_report(
        report,
        plan_path=plan_path,
        pipeline_names=("planned_pipeline",),
    )

    assert matching_issues == ()
    assert "pipelines and profiles disagree" in "\n".join(conflicting_issues)


@pytest.mark.parametrize(
    ("forward_key", "backward_key"),
    [
        ("finite_forward", "finite_backward"),
        ("forward_finite", "backward_finite"),
    ],
)
def test_contract_report_accepts_canonical_and_finiteness_aliases(
    tmp_path: Path,
    forward_key: str,
    backward_key: str,
) -> None:
    plan_path = tmp_path / "plan.json"
    _profiled_contract_plan(plan_path)
    report = _canonical_contract_report()
    pipeline = report["pipelines"]["planned_pipeline"]
    pipeline[forward_key] = pipeline.pop("finite_forward")
    pipeline[backward_key] = pipeline.pop("finite_backward")

    issues = agent_pipeline._validate_contract_smoke_report(
        report,
        plan_path=plan_path,
        pipeline_names=("planned_pipeline",),
    )

    assert issues == ()


def test_kernel_candidate_instructions_specify_exact_contract_profile_schema(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")

    instructions = agent_pipeline._kernel_candidate_contract_instructions(paths)

    assert "`data_free=true`" in instructions
    assert '"compute_mode": "<local_gpu, kaggle_gpu, or kaggle_tpu>"' in instructions
    assert '"hardware_profile": "<hardware profile>"' in instructions
    assert '"finite_forward": true' in instructions
    assert '"finite_backward": true' in instructions
    assert '"logits_shape": [2, <class count>]' in instructions
    assert '"deploy_bytes": <bytes>' in instructions
    assert '"error_type":"DataDiscoveryError"' in instructions
    assert '"reason_code":"missing_raw_training_assets"' in instructions
    assert '"submission_written":false' in instructions


@pytest.mark.parametrize(
    ("mutations", "expected_issue"),
    [
        ({"finite_forward": None}, "finite_forward=true"),
        ({"finite_forward": False}, "finite_forward=true"),
        ({"finite_forward": 1}, "finite_forward=true"),
        ({"finite_backward": None}, "finite_backward=true"),
        ({"finite_backward": False}, "finite_backward=true"),
        ({"finite_backward": 1}, "finite_backward=true"),
        (
            {"finite_forward": True, "forward_finite": False},
            "aliases for finite_forward disagree",
        ),
        (
            {"finite_forward": True, "forward_finite": 1},
            "aliases for finite_forward disagree",
        ),
        (
            {"finite_backward": True, "backward_finite": False},
            "aliases for finite_backward disagree",
        ),
        (
            {"finite_backward": True, "backward_finite": 1},
            "aliases for finite_backward disagree",
        ),
    ],
)
def test_contract_report_rejects_missing_false_nonboolean_and_conflicting_finiteness(
    tmp_path: Path,
    mutations: dict[str, object],
    expected_issue: str,
) -> None:
    plan_path = tmp_path / "plan.json"
    _profiled_contract_plan(plan_path)
    report = _canonical_contract_report()
    pipeline = report["pipelines"]["planned_pipeline"]
    for key, value in mutations.items():
        if value is None:
            pipeline.pop(key)
        else:
            pipeline[key] = value

    issues = agent_pipeline._validate_contract_smoke_report(
        report,
        plan_path=plan_path,
        pipeline_names=("planned_pipeline",),
    )

    assert expected_issue in "\n".join(issues)


def test_contract_report_revalidation_does_not_reuse_initial_diagnostics(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _profiled_contract_plan(plan_path)
    initial_report = _canonical_contract_report()
    initial_report["pipelines"] = {}
    repaired_report = _canonical_contract_report(entry_profile=True)

    initial_issues = agent_pipeline._validate_contract_smoke_report(
        initial_report,
        plan_path=plan_path,
        pipeline_names=("planned_pipeline",),
    )
    repaired_issues = agent_pipeline._validate_contract_smoke_report(
        repaired_report,
        plan_path=plan_path,
        pipeline_names=("planned_pipeline",),
    )

    assert initial_issues
    assert repaired_issues == ()


def test_missing_data_does_not_hide_unrelated_contract_exception(tmp_path: Path) -> None:
    paths, kernel_path = _structured_contract_paths(tmp_path, data_ready=False)
    _write_structured_contract_kernel(kernel_path, contract_exception="unrelated runtime failure")

    result = agent_pipeline._run_kernel_contract_smoke(paths=paths, kernel_path=kernel_path)

    assert result.passed is False
    assert result.smoke_returncode == 1
    assert "unrelated runtime failure" in result.smoke_stderr


def test_contract_smoke_supports_single_output_directory_argument(tmp_path: Path) -> None:
    paths, kernel_path = _structured_contract_paths(
        tmp_path,
        data_ready=False,
        deliverable_mode="writeup",
    )
    _write_structured_contract_kernel(kernel_path, contract_output_arg=True)

    result = agent_pipeline._run_kernel_contract_smoke(paths=paths, kernel_path=kernel_path)

    assert result.passed is True
    assert result.contract_only is True


def test_missing_or_malformed_contract_is_rejected(tmp_path: Path) -> None:
    paths, kernel_path = _structured_contract_paths(tmp_path, data_ready=False)
    kernel_path.write_text("def main():\n    return\n", encoding="utf-8")

    missing = agent_pipeline._run_kernel_contract_smoke(paths=paths, kernel_path=kernel_path)

    assert missing.passed is False
    assert "must export callable" in "\n".join(missing.contract_report_issues)

    _write_structured_contract_kernel(kernel_path, malformed_report=True)
    malformed = agent_pipeline._run_kernel_contract_smoke(paths=paths, kernel_path=kernel_path)

    assert malformed.passed is False
    assert "readable contract_smoke.json" in "\n".join(malformed.contract_report_issues)


@pytest.mark.parametrize(
    ("normal_mode", "expected_issue"),
    [
        ("runtime_error", "does not end with a readable JSON object"),
        ("spoofed_runtime_error", "expected return code 2"),
    ],
)
def test_missing_profile_does_not_hide_unrelated_normal_runtime_error(
    tmp_path: Path,
    normal_mode: str,
    expected_issue: str,
) -> None:
    paths, kernel_path = _structured_contract_paths(tmp_path, data_ready=False)
    _write_structured_contract_kernel(kernel_path, normal_mode=normal_mode)

    result = agent_pipeline._run_kernel_contract_smoke(paths=paths, kernel_path=kernel_path)

    assert result.passed is False
    assert "RuntimeError" in result.normal_smoke_stderr
    assert expected_issue in "\n".join(result.normal_smoke_issues)


@pytest.mark.parametrize("normal_mode", ["success", "sample_copy"])
def test_missing_profile_rejects_silent_non_training_success(tmp_path: Path, normal_mode: str) -> None:
    paths, kernel_path = _structured_contract_paths(tmp_path, data_ready=False)
    _write_structured_contract_kernel(kernel_path, normal_mode=normal_mode)

    result = agent_pipeline._run_kernel_contract_smoke(paths=paths, kernel_path=kernel_path)

    assert result.passed is False
    assert "exited zero" in "\n".join(result.normal_smoke_issues)
    if normal_mode == "sample_copy":
        assert "submission.csv" in "\n".join(result.normal_smoke_issues)


def test_ready_data_keeps_normal_data_discovery_failure_fatal(tmp_path: Path) -> None:
    paths, kernel_path = _structured_contract_paths(tmp_path, data_ready=True)
    _write_structured_contract_kernel(kernel_path)

    result = agent_pipeline._run_kernel_contract_smoke(paths=paths, kernel_path=kernel_path)

    assert result.passed is False
    assert result.normal_smoke_required is True
    assert result.normal_smoke_returncode == 2
    assert "DataDiscoveryError" in result.normal_smoke_stderr
    assert "training data readiness is available" in "\n".join(result.normal_smoke_issues)


def test_ready_data_accepts_successful_normal_probe(tmp_path: Path) -> None:
    paths, kernel_path = _structured_contract_paths(tmp_path, data_ready=True)
    _write_structured_contract_kernel(kernel_path, normal_mode="success")

    result = agent_pipeline._run_kernel_contract_smoke(paths=paths, kernel_path=kernel_path)

    assert result.passed is True
    assert result.contract_only is False
    assert result.data_ready is True
    assert result.normal_smoke_returncode == 0
    assert not result.normal_smoke_issues


def test_ready_data_normal_probe_receives_prefixed_and_legacy_fast_dev_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths, kernel_path = _structured_contract_paths(tmp_path, data_ready=True)
    _write_structured_contract_kernel(
        kernel_path,
        normal_mode="success",
        require_smoke_env_aliases=True,
    )
    monkeypatch.setenv("FAST_DEV", "0")
    monkeypatch.setenv("VALIDATION_MAX_SAMPLES", "999")

    result = agent_pipeline._run_kernel_contract_smoke(paths=paths, kernel_path=kernel_path)

    assert result.passed is True
    assert result.normal_smoke_returncode == 0


@pytest.mark.parametrize(
    ("repair_returncode", "expect_error"),
    [(0, False), (-9, True)],
)
def test_failed_implementation_gets_one_bounded_repair_and_contract_resmoke(
    monkeypatch,
    tmp_path: Path,
    repair_returncode: int,
    expect_error: bool,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    paths = CompetitionPaths(
        slug="demo",
        artifacts_dir=tmp_path / "artifacts",
        repo_root=repo_root,
    )
    paths.context_agent_dir.mkdir(parents=True)
    paths.kernel_source_dir.mkdir(parents=True)
    paths.plan_path.write_text(
        '{"runtime_budget":{"max_messages_per_candidate":4},"pipelines":[]}',
        encoding="utf-8",
    )
    (paths.context_agent_dir / "strategy_plan.md").write_text("strategy", encoding="utf-8")
    instructions_path = paths.context_agent_dir / "instructions.md"
    instructions_path.write_text("instructions", encoding="utf-8")
    (paths.kernel_source_dir / "kernel.py").write_text("print('generated')\n", encoding="utf-8")
    output_dir = paths.context_agent_dir / "implement"
    output_dir.mkdir()

    agent_calls: list[Path] = []

    def fake_agent(**kwargs):
        agent_calls.append(kwargs["prompt_path"])
        last_message_path = kwargs["output_dir"] / "last-message.txt"
        last_message_path.write_text("repair complete", encoding="utf-8")
        if len(agent_calls) == 1:
            return SimpleNamespace(
                returncode=1,
                stdout='{"type":"turn.failed","error":{"message":"transport denied"}}',
                stderr="",
                last_message_path=last_message_path,
            )
        return SimpleNamespace(
            returncode=repair_returncode,
            stdout="repair output",
            stderr="",
            last_message_path=last_message_path,
        )

    failed_smoke = agent_pipeline.KernelContractSmokeResult(
        compile_returncode=0,
        compile_stdout="",
        compile_stderr="",
        smoke_returncode=1,
        smoke_stdout="",
        smoke_stderr="candidate 360 exceeds frozen plan message cap",
    )
    passed_smoke = agent_pipeline.KernelContractSmokeResult(
        compile_returncode=0,
        compile_stdout="",
        compile_stderr="",
        smoke_returncode=0,
        smoke_stdout="all profiles valid",
        smoke_stderr="",
    )
    smoke_results = iter((failed_smoke, passed_smoke))
    monkeypatch.setattr(agent_pipeline, "_run_guarded_kernel_implementation_agent", fake_agent)
    monkeypatch.setattr(agent_pipeline, "_run_kernel_contract_smoke", lambda **kwargs: next(smoke_results))

    config = AgentPipelineConfig(
        slug="demo",
        competition_url=None,
        compute="local_gpu",
        accelerator="gpu",
        internet="off",
        run_id="run-1",
        dry_run=False,
        repo_root=repo_root,
    )

    if expect_error:
        with pytest.raises(KaggleBotError, match="failed after one repair attempt") as exc_info:
            agent_pipeline._run_codex_kernel_implementation(paths, config, output_dir, instructions_path)
        assert "returncode=-9 (SIGKILL)" in str(exc_info.value)
    else:
        agent_pipeline._run_codex_kernel_implementation(paths, config, output_dir, instructions_path)

    assert len(agent_calls) == 2
    assert agent_calls[1] == output_dir / "repair-1" / "prompt.md"
    repair_prompt = agent_calls[1].read_text(encoding="utf-8")
    assert "transport denied" in repair_prompt
    assert "candidate 360 exceeds frozen plan message cap" in repair_prompt


def test_successful_implementation_is_contract_smoked(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts", repo_root=repo_root)
    paths.context_agent_dir.mkdir(parents=True)
    paths.kernel_source_dir.mkdir(parents=True)
    paths.plan_path.write_text('{"pipelines":[]}', encoding="utf-8")
    (paths.context_agent_dir / "strategy_plan.md").write_text("strategy", encoding="utf-8")
    instructions_path = paths.context_agent_dir / "instructions.md"
    instructions_path.write_text("instructions", encoding="utf-8")
    (paths.kernel_source_dir / "kernel.py").write_text("print('generated')\n", encoding="utf-8")
    output_dir = paths.context_agent_dir / "implement"
    output_dir.mkdir()
    last_message_path = output_dir / "last-message.txt"
    last_message_path.write_text("complete", encoding="utf-8")
    result = SimpleNamespace(
        returncode=0,
        stdout="complete",
        stderr="",
        last_message_path=last_message_path,
    )
    smoke = agent_pipeline.KernelContractSmokeResult(
        compile_returncode=0,
        compile_stdout="",
        compile_stderr="",
        smoke_returncode=0,
        smoke_stdout="valid",
        smoke_stderr="",
    )
    smoke_calls: list[dict[str, object]] = []
    monkeypatch.setattr(agent_pipeline, "_run_guarded_kernel_implementation_agent", lambda **kwargs: result)
    monkeypatch.setattr(
        agent_pipeline,
        "_run_kernel_contract_smoke",
        lambda **kwargs: smoke_calls.append(kwargs) or smoke,
    )
    config = AgentPipelineConfig(
        slug="demo",
        competition_url=None,
        compute="local_gpu",
        accelerator="gpu",
        internet="off",
        run_id="run-1",
        dry_run=False,
        repo_root=repo_root,
        hardware_profile="rtx5090",
    )

    agent_pipeline._run_codex_kernel_implementation(paths, config, output_dir, instructions_path)

    assert len(smoke_calls) == 1
    assert smoke_calls[0]["kernel_path"].name == "kernel.py"
    assert smoke_calls[0]["kernel_path"] != paths.kernel_source_dir / "kernel.py"
    assert smoke_calls[0]["expected_compute_profile"] == "local_gpu"
    assert smoke_calls[0]["expected_hardware_profile"] == "rtx5090"
    assert (paths.kernel_source_dir / "kernel.py").read_text(encoding="utf-8") == "print('generated')\n"


def test_classifier_block_discards_partial_kernel_and_uses_sanitized_retry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    paths = CompetitionPaths(slug="security-demo", artifacts_dir=tmp_path / "artifacts", repo_root=repo_root)
    paths.context_agent_dir.mkdir(parents=True)
    paths.kernel_source_dir.mkdir(parents=True)
    paths.plan_path.write_text(
        json.dumps(
            {
                "runtime_budget": {
                    "max_return_candidates": 700,
                    "max_messages_per_candidate": 4,
                    "max_message_chars": 900,
                    "max_posts_per_candidate": 3,
                    "archive_cap": 64,
                    "enable_validation": True,
                    "enable_training": False,
                },
                "pipelines": [
                    {"name": "adaptive_k1_conditional_multipost_fill"},
                    {"name": "bounded_trace_archive_mutation_scout"},
                    {"name": "reference_k1_boundary_anchor_644"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (paths.context_agent_dir / "strategy_plan.md").write_text(
        "strategy with candidate-payload-sentinel",
        encoding="utf-8",
    )
    instructions_path = paths.context_agent_dir / "instructions.md"
    instructions_path.write_text("instructions with candidate-payload-sentinel", encoding="utf-8")
    live_kernel_path = paths.kernel_source_dir / "kernel.py"
    live_kernel_path.write_text("print('clean baseline')\n", encoding="utf-8")
    output_dir = paths.context_agent_dir / "implement"
    output_dir.mkdir()

    prompts: list[str] = []
    attempted_paths: list[Path] = []

    def staged_entrypoint(prompt_text: str) -> Path:
        for line in prompt_text.splitlines():
            if "Primary entrypoint:" in line or line.startswith("Staged entrypoint:"):
                return Path(line.split(":", maxsplit=1)[1].strip().strip("`"))
        raise AssertionError("staged entrypoint missing from implementation prompt")

    def fake_agent(**kwargs):
        prompt_text = kwargs["prompt_path"].read_text(encoding="utf-8")
        prompts.append(prompt_text)
        candidate_path = staged_entrypoint(prompt_text)
        attempted_paths.append(candidate_path)
        last_message_path = kwargs["output_dir"] / "last-message.txt"
        if len(prompts) == 1:
            candidate_path.write_text("print('partial blocked edit')\n", encoding="utf-8")
            last_message_path.write_text("blocked", encoding="utf-8")
            return SimpleNamespace(
                returncode=1,
                stdout=(
                    '{"type":"turn.failed","error":{"message":"This content was flagged for possible '
                    'cybersecurity risk."}}'
                ),
                stderr="",
                last_message_path=last_message_path,
            )
        assert candidate_path.read_text(encoding="utf-8") == "print('clean baseline')\n"
        assert live_kernel_path.read_text(encoding="utf-8") == "print('clean baseline')\n"
        candidate_path.write_text("print('validated repair')\n", encoding="utf-8")
        last_message_path.write_text("repair complete", encoding="utf-8")
        return SimpleNamespace(
            returncode=0,
            stdout="repair complete",
            stderr="",
            last_message_path=last_message_path,
        )

    def fake_smoke(**kwargs):
        text = kwargs["kernel_path"].read_text(encoding="utf-8")
        if "partial blocked edit" in text:
            return agent_pipeline.KernelContractSmokeResult(
                compile_returncode=0,
                compile_stdout="candidate payload must not enter retry prompt",
                compile_stderr="",
                smoke_returncode=1,
                smoke_stdout="candidate payload must not enter retry prompt",
                smoke_stderr="KeyError: 'hybrid'",
            )
        return agent_pipeline.KernelContractSmokeResult(
            compile_returncode=0,
            compile_stdout="",
            compile_stderr="",
            smoke_returncode=0,
            smoke_stdout="all profiles valid",
            smoke_stderr="",
        )

    monkeypatch.setattr(agent_pipeline, "_run_guarded_kernel_implementation_agent", fake_agent)
    monkeypatch.setattr(agent_pipeline, "_run_kernel_contract_smoke", fake_smoke)
    config = AgentPipelineConfig(
        slug="security-demo",
        competition_url=None,
        compute="local_gpu",
        accelerator="gpu",
        internet="off",
        run_id="run-1",
        dry_run=False,
        repo_root=repo_root,
    )

    agent_pipeline._run_codex_kernel_implementation(paths, config, output_dir, instructions_path)

    assert len(prompts) == 2
    assert attempted_paths[0] != attempted_paths[1]
    assert "candidate-payload-sentinel" in prompts[0]
    assert "candidate-payload-sentinel" not in prompts[1]
    assert "deterministic offline Kaggle SDK" in prompts[1]
    assert "adaptive_k1_conditional_multipost_fill" in prompts[1]
    assert "max_messages_per_candidate=4" in prompts[1]
    assert "KeyError: 'hybrid'" in prompts[1]
    assert "candidate payload must not enter retry prompt" not in prompts[1]
    assert live_kernel_path.read_text(encoding="utf-8") == "print('validated repair')\n"


def test_agent_pipeline_publishes_codex_oracle_codex_phases(monkeypatch, tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    config = AgentPipelineConfig(
        slug="demo",
        competition_url=None,
        compute="local_gpu",
        accelerator="gpu",
        internet="on",
        run_id="run-1",
        dry_run=False,
        repo_root=tmp_path,
    )
    brief_path = tmp_path / "brief.md"
    instructions_path = tmp_path / "instructions.md"
    phases: list[tuple[str, str | None]] = []

    monkeypatch.setattr(agent_pipeline, "_ensure_context_materials", lambda paths: None)
    monkeypatch.setattr(agent_pipeline, "refresh_kaggle_discovery", lambda **kwargs: {})
    monkeypatch.setattr(agent_pipeline.CodexBriefStage, "run", lambda self: brief_path)
    monkeypatch.setattr(
        agent_pipeline.StrategyStage,
        "run",
        lambda self, *, brief_path: instructions_path,
    )
    monkeypatch.setattr(
        agent_pipeline.CodexImplementationStage,
        "run",
        lambda self, *, instructions_path: None,
    )
    monkeypatch.setattr(
        agent_pipeline,
        "update_watch_phase",
        lambda config, run_id, phase, *, detail=None, iteration=None: phases.append((phase, detail)),
    )

    agent_pipeline.AgentPipeline(paths=paths, config=config).run()

    assert [phase for phase, _ in phases] == [
        "kaggle_discovery",
        "codex_brief",
        "oracle_strategy",
        "codex_implementation",
    ]
    assert "Oracle Pro" in str(phases[2][1])
    assert "sol-xhigh" in str(phases[3][1])


def test_fallback_strategy_instructions_cover_general_tabular_formats(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text("{}", encoding="utf-8")
    config = AgentPipelineConfig(
        slug="demo",
        competition_url=None,
        compute="local_gpu",
        accelerator="gpu",
        internet="off",
        run_id="run-1",
        dry_run=False,
        repo_root=tmp_path,
    )

    strategy, instructions, _, research_sources_text, _ = agent_pipeline._build_fallback_strategy(
        paths=paths,
        config=config,
        brief_content="",
        error_text="strategy unavailable",
    )

    assert "point-cloud/3D" in strategy
    assert "annotation" in strategy
    assert "artifact outputs" in strategy
    assert "tabular/text/image/sequence" not in strategy
    assert "Excel" in instructions
    assert "Feather" in instructions
    assert "Stata" in instructions
    assert "XML" in instructions
    assert "SQLite" in instructions
    assert "compressed tabular variants" in instructions
    assert "non-tabular artifacts" in instructions
    assert "point-cloud/3D" in instructions
    assert "model files" in instructions
    assert "forced to CSV" in instructions
    assert "annotation" in instructions
    assert "model-artifact specific paths" in instructions
    assert "CNN/transformer/sequence models where useful" in instructions
    assert "repo `read_table` helper" in instructions
    assert "submission format file schema artifact" in research_sources_text
    assert "submission format csv columns" not in research_sources_text


def test_strategy_prompt_includes_ranked_kaggle_discovery(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True)
    paths.kaggle_discovery_md_path.write_text(
        "# Kaggle Discovery Snapshot\n\n## Models\n- Relevant pretrained model\n",
        encoding="utf-8",
    )
    config = AgentPipelineConfig(
        slug="demo",
        competition_url=None,
        compute="local_gpu",
        accelerator="gpu",
        internet="on",
        run_id="run-1",
        dry_run=False,
        repo_root=tmp_path,
    )

    prompt = agent_pipeline._build_strategy_prompt(  # noqa: SLF001
        template=agent_pipeline._load_template("strategy_plan.md"),  # noqa: SLF001
        config=config,
        paths=paths,
        brief_content="competition brief",
        compact=False,
    )

    assert "Ranked Kaggle ecosystem discovery" in prompt
    assert "Relevant pretrained model" in prompt
    assert "{{kaggle_discovery_snapshot}}" not in prompt
    bundle = (paths.context_agent_dir / "strategy_context_bundle.md").read_text(encoding="utf-8")
    assert "## Ranked Kaggle Ecosystem Discovery" in bundle


def test_codex_brief_guard_ignores_sibling_competition_artifact_churn(monkeypatch, tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "kaggle-autopilot-artifacts")
    paths.context_agent_dir.mkdir(parents=True, exist_ok=True)
    output_dir = paths.context_agent_dir / "brief"
    prompt_path = output_dir / "prompt.md"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("brief prompt\n", encoding="utf-8")

    other_context = paths.artifacts_dir / "other-slug" / "context"
    other_context.mkdir(parents=True, exist_ok=True)
    other_guard = other_context / "zero_overlap_drift_guard.json"
    other_guard.write_text('{"status":"original"}\n', encoding="utf-8")

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs):  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        other_guard.write_text('{"status":"changed"}\n', encoding="utf-8")
        last_message_path = output_dir / "last_message.txt"
        last_message_path.write_text("Brief text\n", encoding="utf-8")
        return SimpleNamespace(last_message_path=last_message_path, returncode=0, stderr="")

    monkeypatch.setattr(agent_pipeline, "run_codex", fake_run_codex)

    config = AgentPipelineConfig(
        slug="demo",
        competition_url=None,
        compute="local_gpu",
        accelerator="gpu",
        internet="off",
        run_id="run-1",
        dry_run=False,
        repo_root=tmp_path,
    )

    brief_text, error_text = agent_pipeline._run_codex_brief_with_retry(
        prompt_path=prompt_path,
        output_dir=output_dir,
        paths=paths,
        config=config,
    )

    assert brief_text == "Brief text"
    assert error_text == ""
    assert other_guard.read_text(encoding="utf-8") == '{"status":"changed"}\n'


def test_codex_brief_guard_rejects_active_context_edits_outside_agent(monkeypatch, tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "kaggle-autopilot-artifacts")
    paths.context_agent_dir.mkdir(parents=True, exist_ok=True)
    output_dir = paths.context_agent_dir / "brief"
    prompt_path = output_dir / "prompt.md"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("brief prompt\n", encoding="utf-8")

    guard_path = paths.context_dir / "zero_overlap_drift_guard.json"
    guard_path.write_text('{"status":"original"}\n', encoding="utf-8")

    def fake_run_codex(prompt_path: Path, output_dir: Path, dry_run: bool, **kwargs):  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        guard_path.write_text('{"status":"changed"}\n', encoding="utf-8")
        last_message_path = output_dir / "last_message.txt"
        last_message_path.write_text("Brief text\n", encoding="utf-8")
        return SimpleNamespace(last_message_path=last_message_path, returncode=0, stderr="")

    monkeypatch.setattr(agent_pipeline, "run_codex", fake_run_codex)

    config = AgentPipelineConfig(
        slug="demo",
        competition_url=None,
        compute="local_gpu",
        accelerator="gpu",
        internet="off",
        run_id="run-1",
        dry_run=False,
        repo_root=tmp_path,
    )

    try:
        agent_pipeline._run_codex_brief_with_retry(
            prompt_path=prompt_path,
            output_dir=output_dir,
            paths=paths,
            config=config,
        )
    except KaggleBotError as exc:
        message = str(exc)
        assert "Agent write-guard failed in codex_brief" in message
        assert "context/zero_overlap_drift_guard.json" in message
    else:
        raise AssertionError("expected active competition context edit to be rejected")


def test_guard_restores_other_competition_kernel(tmp_path: Path) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts"

    allowed_kernel = artifacts_dir / "demo" / "kernel"
    allowed_kernel.mkdir(parents=True, exist_ok=True)
    (allowed_kernel / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    other_kernel = artifacts_dir / "other" / "kernel"
    other_kernel.mkdir(parents=True, exist_ok=True)
    other_kernel_path = other_kernel / "kernel.py"
    original = "print('original')\n"
    other_kernel_path.write_text(original, encoding="utf-8")

    allowed_prefixes = [allowed_kernel]
    guard_snapshot = write_guard._backup_guarded_files(repo_root, allowed_prefixes)
    before = write_guard._snapshot_tree(repo_root)

    # Simulate an unauthorized edit outside the allowlist.
    other_kernel_path.write_text("print('changed')\n", encoding="utf-8")
    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
        root=repo_root,
        before=before,
        after=after,
        allowed_prefixes=allowed_prefixes,
        stage="test_guard",
        guard_snapshot=guard_snapshot,
        auto_repair=True,
    )

    assert other_kernel_path.read_text(encoding="utf-8") == original


def test_guard_restores_other_competition_submission_ledger(tmp_path: Path) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts"

    allowed_kernel = artifacts_dir / "demo" / "kernel"
    allowed_kernel.mkdir(parents=True, exist_ok=True)
    (allowed_kernel / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    other_submissions = artifacts_dir / "other" / "submissions"
    other_submissions.mkdir(parents=True, exist_ok=True)
    ledger_path = other_submissions / "ledger.jsonl"
    original = '{"run_id":"r1","hash":"abc"}\n'
    ledger_path.write_text(original, encoding="utf-8")

    allowed_prefixes = [allowed_kernel]
    guard_snapshot = write_guard._backup_guarded_files(repo_root, allowed_prefixes)
    before = write_guard._snapshot_tree(repo_root)

    ledger_path.write_text('{"run_id":"r2","hash":"def"}\n', encoding="utf-8")
    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
        root=repo_root,
        before=before,
        after=after,
        allowed_prefixes=allowed_prefixes,
        stage="test_guard",
        guard_snapshot=guard_snapshot,
        auto_repair=True,
    )

    assert ledger_path.read_text(encoding="utf-8") == original


def test_guard_restores_other_competition_data_sample_submission(tmp_path: Path) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts"

    allowed_kernel = artifacts_dir / "demo" / "kernel"
    allowed_kernel.mkdir(parents=True, exist_ok=True)
    (allowed_kernel / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    other_data = artifacts_dir / "other" / "data"
    other_data.mkdir(parents=True, exist_ok=True)
    sample_path = other_data / "sample_submission.csv"
    original = "Id,Category\nval_1.tif,Health\n"
    sample_path.write_text(original, encoding="utf-8")

    allowed_prefixes = [allowed_kernel]
    guard_snapshot = write_guard._backup_guarded_files(repo_root, allowed_prefixes)
    before = write_guard._snapshot_tree(repo_root)

    sample_path.write_text("Id,Category\nval_1.tif,Rust\n", encoding="utf-8")
    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
        root=repo_root,
        before=before,
        after=after,
        allowed_prefixes=allowed_prefixes,
        stage="test_guard",
        guard_snapshot=guard_snapshot,
        auto_repair=True,
    )

    assert sample_path.read_text(encoding="utf-8") == original


def test_guard_restores_other_competition_jsonl_sample_submission(tmp_path: Path) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts"

    allowed_kernel = artifacts_dir / "demo" / "kernel"
    allowed_kernel.mkdir(parents=True, exist_ok=True)
    (allowed_kernel / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    other_data = artifacts_dir / "other" / "data"
    other_data.mkdir(parents=True, exist_ok=True)
    sample_path = other_data / "sample_submission.jsonl"
    original = '{"id":1,"target":0.1}\n'
    sample_path.write_text(original, encoding="utf-8")

    allowed_prefixes = [allowed_kernel]
    guard_snapshot = write_guard._backup_guarded_files(repo_root, allowed_prefixes)
    before = write_guard._snapshot_tree(repo_root)

    sample_path.write_text('{"id":1,"target":0.9}\n', encoding="utf-8")
    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
        root=repo_root,
        before=before,
        after=after,
        allowed_prefixes=allowed_prefixes,
        stage="test_guard",
        guard_snapshot=guard_snapshot,
        auto_repair=True,
    )

    assert sample_path.read_text(encoding="utf-8") == original


def test_guard_restores_oversized_data_sample_submission_from_context(tmp_path: Path) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts"

    allowed_kernel = artifacts_dir / "demo" / "kernel"
    allowed_kernel.mkdir(parents=True, exist_ok=True)
    (allowed_kernel / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    slug = "beyond-visible-spectrum-ai-for-agriculture-2026"
    context_sample = artifacts_dir / slug / "context" / "sample_submission.csv"
    data_sample = artifacts_dir / slug / "data" / "sample_submission.csv"
    context_sample.parent.mkdir(parents=True, exist_ok=True)
    data_sample.parent.mkdir(parents=True, exist_ok=True)

    context_content = "Id,Category\nval_1.tif,Healthy\n"
    context_sample.write_text(context_content, encoding="utf-8")

    row = "val_1.tif,Healthy\n"
    repeat = (write_guard._MAX_GUARD_FILE_BYTES // len(row)) + 1_000
    oversized_content = "Id,Category\n" + (row * repeat)
    data_sample.write_text(oversized_content, encoding="utf-8")
    assert data_sample.stat().st_size > write_guard._MAX_GUARD_FILE_BYTES

    allowed_prefixes = [allowed_kernel]
    guard_snapshot = write_guard._backup_guarded_files(repo_root, allowed_prefixes)
    sample_rel = data_sample.relative_to(repo_root).as_posix()
    assert sample_rel in guard_snapshot.oversized
    before = write_guard._snapshot_tree(repo_root)

    data_sample.write_text("Id,Category\nval_1.tif,Rust\n", encoding="utf-8")
    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
        root=repo_root,
        before=before,
        after=after,
        allowed_prefixes=allowed_prefixes,
        stage="test_guard",
        guard_snapshot=guard_snapshot,
        auto_repair=True,
    )

    assert data_sample.read_text(encoding="utf-8") == context_content


def test_guard_restores_oversized_jsonl_data_sample_submission_from_context(tmp_path: Path) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts"

    allowed_kernel = artifacts_dir / "demo" / "kernel"
    allowed_kernel.mkdir(parents=True, exist_ok=True)
    (allowed_kernel / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    slug = "jsonl-demo"
    context_sample = artifacts_dir / slug / "context" / "sample_submission.jsonl"
    data_sample = artifacts_dir / slug / "data" / "sample_submission.jsonl"
    context_sample.parent.mkdir(parents=True, exist_ok=True)
    data_sample.parent.mkdir(parents=True, exist_ok=True)

    context_content = '{"id":1,"target":0.1}\n'
    context_sample.write_text(context_content, encoding="utf-8")

    row = '{"id":1,"target":0.1}\n'
    repeat = (write_guard._MAX_GUARD_FILE_BYTES // len(row)) + 1_000
    data_sample.write_text(row * repeat, encoding="utf-8")
    assert data_sample.stat().st_size > write_guard._MAX_GUARD_FILE_BYTES

    allowed_prefixes = [allowed_kernel]
    guard_snapshot = write_guard._backup_guarded_files(repo_root, allowed_prefixes)
    sample_rel = data_sample.relative_to(repo_root).as_posix()
    assert sample_rel in guard_snapshot.oversized
    before = write_guard._snapshot_tree(repo_root)

    data_sample.write_text('{"id":1,"target":0.9}\n', encoding="utf-8")
    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
        root=repo_root,
        before=before,
        after=after,
        allowed_prefixes=allowed_prefixes,
        stage="test_guard",
        guard_snapshot=guard_snapshot,
        auto_repair=True,
    )

    assert data_sample.read_text(encoding="utf-8") == context_content


def test_guard_recognizes_compressed_data_sample_submission() -> None:
    assert write_guard._is_artifact_data_sample_submission("artifacts/demo/data/sample_submission.csv.gz")
    assert write_guard._is_artifact_data_sample_submission("artifacts/demo/data/sample_submission.xlsx")
    assert write_guard._is_artifact_data_sample_submission("artifacts/demo/data/sample_submission.jsonl.xz")
    assert not write_guard._is_artifact_data_sample_submission("artifacts/demo/data/sample_submission.zip")


def test_guard_matches_compressed_data_sample_submission_context(tmp_path: Path) -> None:
    repo_root = tmp_path
    context_sample = repo_root / "artifacts" / "demo" / "context" / "sample_submission.csv.gz"
    data_sample = repo_root / "artifacts" / "demo" / "data" / "sample_submission.csv.gz"
    context_sample.parent.mkdir(parents=True, exist_ok=True)
    data_sample.parent.mkdir(parents=True, exist_ok=True)
    payload = b"\x1f\x8bcompressed-sample-bytes"
    context_sample.write_bytes(payload)
    data_sample.write_bytes(payload)

    assert write_guard._matches_artifact_data_sample_submission_context(
        repo_root,
        "artifacts/demo/data/sample_submission.csv.gz",
    )


def test_guard_matches_excel_data_sample_submission_context(tmp_path: Path) -> None:
    repo_root = tmp_path
    context_sample = repo_root / "artifacts" / "demo" / "context" / "sample_submission.xlsx"
    data_sample = repo_root / "artifacts" / "demo" / "data" / "sample_submission.xlsx"
    context_sample.parent.mkdir(parents=True, exist_ok=True)
    data_sample.parent.mkdir(parents=True, exist_ok=True)
    payload = b"excel-sample-bytes"
    context_sample.write_bytes(payload)
    data_sample.write_bytes(payload)

    assert write_guard._matches_artifact_data_sample_submission_context(
        repo_root,
        "artifacts/demo/data/sample_submission.xlsx",
    )


def test_guard_ignores_kagglebot_cache_churn(tmp_path: Path) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts"

    allowed_kernel = artifacts_dir / "demo" / "kernel"
    allowed_kernel.mkdir(parents=True, exist_ok=True)
    (allowed_kernel / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    cache_file = artifacts_dir / "demo" / "data" / ".kagglebot_cache" / "sample_submission_synth.csv"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("id,target\n1,0\n", encoding="utf-8")

    allowed_prefixes = [allowed_kernel]
    guard_snapshot = write_guard._backup_guarded_files(repo_root, allowed_prefixes)
    before = write_guard._snapshot_tree(repo_root)

    cache_file.write_text("id,target\n1,1\n", encoding="utf-8")
    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
        root=repo_root,
        before=before,
        after=after,
        allowed_prefixes=allowed_prefixes,
        stage="test_guard",
        guard_snapshot=guard_snapshot,
        auto_repair=True,
    )

    assert cache_file.read_text(encoding="utf-8").strip().endswith(",1")


def test_guard_restores_competition_control_files(tmp_path: Path) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts"

    allowed_kernel = artifacts_dir / "demo" / "kernel"
    allowed_kernel.mkdir(parents=True, exist_ok=True)
    (allowed_kernel / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    meta_path = artifacts_dir / "demo" / "meta.json"
    plan_path = artifacts_dir / "demo" / "plan.json"
    prompts_path = artifacts_dir / "demo" / "prompts" / "codex_kernel_fix.md"
    kb_path = repo_root / "knowledge" / "kb.sqlite"

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    (prompts_path.parent).mkdir(parents=True, exist_ok=True)
    kb_path.parent.mkdir(parents=True, exist_ok=True)

    meta_original = '{"slug":"demo"}\n'
    plan_original = '{"pipelines":[]}\n'
    prompts_original = "# original prompt\n"
    kb_original = b"sqlite-bytes"

    meta_path.write_text(meta_original, encoding="utf-8")
    plan_path.write_text(plan_original, encoding="utf-8")
    prompts_path.write_text(prompts_original, encoding="utf-8")
    kb_path.write_bytes(kb_original)

    allowed_prefixes = [allowed_kernel]
    guard_snapshot = write_guard._backup_guarded_files(repo_root, allowed_prefixes)
    before = write_guard._snapshot_tree(repo_root)

    meta_path.write_text('{"slug":"changed"}\n', encoding="utf-8")
    plan_path.write_text('{"pipelines":["oops"]}\n', encoding="utf-8")
    prompts_path.write_text("# changed prompt\n", encoding="utf-8")
    kb_path.write_bytes(b"changed-bytes")

    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
        root=repo_root,
        before=before,
        after=after,
        allowed_prefixes=allowed_prefixes,
        stage="test_guard",
        guard_snapshot=guard_snapshot,
        auto_repair=True,
    )

    assert meta_path.read_text(encoding="utf-8") == meta_original
    assert plan_path.read_text(encoding="utf-8") == plan_original
    assert prompts_path.read_text(encoding="utf-8") == prompts_original
    assert kb_path.read_bytes() == kb_original


def test_guard_restores_knowledge_research_artifacts(tmp_path: Path) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts"

    allowed_kernel = artifacts_dir / "demo" / "kernel"
    allowed_kernel.mkdir(parents=True, exist_ok=True)
    (allowed_kernel / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    research_dir = repo_root / "knowledge" / "research" / "unknown" / "demo"
    research_dir.mkdir(parents=True, exist_ok=True)
    sources_path = research_dir / "research_sources.jsonl"
    summary_path = research_dir / "research_summary.md"

    sources_original = '{"url":"https://example.com","title":"Example"}\n'
    summary_original = "# Research\n\nOriginal summary.\n"
    sources_path.write_text(sources_original, encoding="utf-8")
    summary_path.write_text(summary_original, encoding="utf-8")

    allowed_prefixes = [allowed_kernel]
    guard_snapshot = write_guard._backup_guarded_files(repo_root, allowed_prefixes)
    before = write_guard._snapshot_tree(repo_root)

    sources_path.write_text('{"url":"https://bad.example","title":"Changed"}\n', encoding="utf-8")
    summary_path.write_text("# Research\n\nChanged summary.\n", encoding="utf-8")

    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
        root=repo_root,
        before=before,
        after=after,
        allowed_prefixes=allowed_prefixes,
        stage="test_guard",
        guard_snapshot=guard_snapshot,
        auto_repair=True,
    )

    assert sources_path.read_text(encoding="utf-8") == sources_original
    assert summary_path.read_text(encoding="utf-8") == summary_original


def test_guard_ignores_generated_kernel_staging_tree(tmp_path: Path) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts"

    allowed_kernel = artifacts_dir / "demo" / "kernel"
    allowed_kernel.mkdir(parents=True, exist_ok=True)
    (allowed_kernel / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    staged_kernel = artifacts_dir / "demo" / "kernels" / "run123" / "local-iter-1" / "kernel.py"
    staged_kernel.parent.mkdir(parents=True, exist_ok=True)
    staged_kernel.write_text("print('original')\n", encoding="utf-8")

    allowed_prefixes = [allowed_kernel]
    guard_snapshot = write_guard._backup_guarded_files(repo_root, allowed_prefixes)
    before = write_guard._snapshot_tree(repo_root)

    staged_kernel.write_text("print('changed')\n", encoding="utf-8")
    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
        root=repo_root,
        before=before,
        after=after,
        allowed_prefixes=allowed_prefixes,
        stage="test_guard",
        guard_snapshot=guard_snapshot,
        auto_repair=True,
    )

    assert staged_kernel.read_text(encoding="utf-8") == "print('changed')\n"


@pytest.mark.parametrize(
    ("filename", "payload", "updated_payload"),
    [
        ("submission.compact.csv", b"id,target\n1,0\n", b"id,target\n1,1\n"),
        ("submission.compact.csv.gz", b"compressed-before", b"compressed-after"),
        ("submission.compact.tsv", b"id\ttarget\n1\t0\n", b"id\ttarget\n1\t1\n"),
        ("submission.compact.tsv.zst", b"tsv-zst-before", b"tsv-zst-after"),
        ("submission.compact.jsonl.zst", b"jsonl-zst-before", b"jsonl-zst-after"),
    ],
)
def test_guard_ignores_historical_run_submission_compact_churn(
    tmp_path: Path,
    filename: str,
    payload: bytes,
    updated_payload: bytes,
) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts"

    allowed_kernel = artifacts_dir / "demo" / "kernel"
    allowed_kernel.mkdir(parents=True, exist_ok=True)
    (allowed_kernel / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    compact_submission = artifacts_dir / "other" / "runs" / "run123" / "iter-3" / filename
    compact_submission.parent.mkdir(parents=True, exist_ok=True)
    compact_submission.write_bytes(payload)

    allowed_prefixes = [allowed_kernel]
    guard_snapshot = write_guard._backup_guarded_files(repo_root, allowed_prefixes)
    before = write_guard._snapshot_tree(repo_root)

    compact_submission.write_bytes(updated_payload)
    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
        root=repo_root,
        before=before,
        after=after,
        allowed_prefixes=allowed_prefixes,
        stage="test_guard",
        guard_snapshot=guard_snapshot,
        auto_repair=True,
    )

    assert compact_submission.read_bytes() == updated_payload


def test_guard_ignores_venv_churn_and_restores_uv_lock(tmp_path: Path) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts"

    allowed_kernel = artifacts_dir / "demo" / "kernel"
    allowed_kernel.mkdir(parents=True, exist_ok=True)
    (allowed_kernel / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    uv_lock = repo_root / "uv.lock"
    uv_lock_original = "lock-version = 1\n"
    uv_lock.write_text(uv_lock_original, encoding="utf-8")

    venv_entrypoint = repo_root / ".venv" / "bin" / "kagglebot"
    venv_entrypoint.parent.mkdir(parents=True, exist_ok=True)
    venv_entrypoint.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    allowed_prefixes = [allowed_kernel]
    guard_snapshot = write_guard._backup_guarded_files(repo_root, allowed_prefixes)
    before = write_guard._snapshot_tree(repo_root)

    uv_lock.write_text("lock-version = 999\n", encoding="utf-8")
    venv_entrypoint.write_text("# changed\n", encoding="utf-8")
    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
        root=repo_root,
        before=before,
        after=after,
        allowed_prefixes=allowed_prefixes,
        stage="test_guard",
        guard_snapshot=guard_snapshot,
        auto_repair=True,
    )

    assert uv_lock.read_text(encoding="utf-8") == uv_lock_original
    assert venv_entrypoint.read_text(encoding="utf-8") == "# changed\n"


def test_guard_allows_explicit_dependency_file_edits(tmp_path: Path) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts"

    allowed_kernel = artifacts_dir / "demo" / "kernel"
    allowed_kernel.mkdir(parents=True, exist_ok=True)
    (allowed_kernel / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

    pyproject_path = repo_root / "pyproject.toml"
    uv_lock_path = repo_root / "uv.lock"
    pyproject_path.write_text("[project]\nname='demo'\n", encoding="utf-8")
    uv_lock_path.write_text("lock-version = 1\n", encoding="utf-8")

    allowed_prefixes = [allowed_kernel, pyproject_path, uv_lock_path]
    guard_snapshot = write_guard._backup_guarded_files(repo_root, allowed_prefixes)
    before = write_guard._snapshot_tree(repo_root)

    pyproject_path.write_text("[project]\nname='demo'\ndependencies=['albumentations']\n", encoding="utf-8")
    uv_lock_path.write_text("lock-version = 2\n", encoding="utf-8")
    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
        root=repo_root,
        before=before,
        after=after,
        allowed_prefixes=allowed_prefixes,
        stage="test_guard",
        guard_snapshot=guard_snapshot,
        auto_repair=True,
    )

    assert "albumentations" in pyproject_path.read_text(encoding="utf-8")
    assert uv_lock_path.read_text(encoding="utf-8") == "lock-version = 2\n"


def test_repo_root_write_policy_allows_src_but_restores_data_and_generated_kernels(tmp_path: Path) -> None:
    repo_root = tmp_path
    src_path = repo_root / "src" / "demo.py"
    data_path = repo_root / "artifacts" / "demo" / "data" / "train.csv"
    staged_kernel_path = repo_root / "artifacts" / "demo" / "kernels" / "run123" / "local-iter-1" / "kernel.py"
    src_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    staged_kernel_path.parent.mkdir(parents=True, exist_ok=True)
    src_path.write_text("VALUE = 1\n", encoding="utf-8")
    data_path.write_text("id,target\n1,0\n", encoding="utf-8")
    staged_kernel_path.write_text("print('original')\n", encoding="utf-8")

    policy = write_guard._repo_root_write_policy(
        repo_root=repo_root,
        denied_prefixes=[repo_root / "artifacts" / "demo" / "data", repo_root / "artifacts" / "demo" / "kernels"],
    )
    guard_snapshot = write_guard._backup_guarded_files(repo_root, policy)
    before = write_guard._snapshot_tree(repo_root)

    src_path.write_text("VALUE = 2\n", encoding="utf-8")
    data_path.write_text("id,target\n1,1\n", encoding="utf-8")
    staged_kernel_path.write_text("print('changed')\n", encoding="utf-8")
    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
        root=repo_root,
        before=before,
        after=after,
        allowed_prefixes=policy,
        stage="test_broad_policy",
        guard_snapshot=guard_snapshot,
        auto_repair=True,
    )

    assert src_path.read_text(encoding="utf-8") == "VALUE = 2\n"
    assert data_path.read_text(encoding="utf-8") == "id,target\n1,0\n"
    assert staged_kernel_path.read_text(encoding="utf-8") == "print('original')\n"


def test_write_guard_caps_repo_policy_backup_and_rejects_unbacked_denied_edits(tmp_path: Path) -> None:
    repo_root = tmp_path
    denied_dir = repo_root / "artifacts" / "demo" / "kernels"
    denied_dir.mkdir(parents=True)
    first = denied_dir / "first.py"
    second = denied_dir / "second.py"
    first.write_bytes(b"a" * 6)
    second.write_bytes(b"b" * 6)
    policy = write_guard.WriteGuardPolicy(
        allowed_prefixes=(repo_root,),
        denied_prefixes=(denied_dir,),
        max_backup_bytes=6,
    )

    guard_snapshot = write_guard._backup_guarded_files(repo_root, policy)

    assert sum(len(content) for content in guard_snapshot.backup.values()) <= 6
    unbacked = next(
        path for path in (first, second) if path.relative_to(repo_root).as_posix() not in guard_snapshot.backup
    )
    before = write_guard._snapshot_tree(repo_root)
    unbacked.write_text("changed\n", encoding="utf-8")
    after = write_guard._snapshot_tree(repo_root)

    with pytest.raises(KaggleBotError, match="Cannot auto-repair changed file"):
        write_guard._enforce_allowlist_changes(
            root=repo_root,
            before=before,
            after=after,
            allowed_prefixes=policy,
            stage="test_backup_budget",
            guard_snapshot=guard_snapshot,
            auto_repair=True,
        )


def test_build_repair_write_policy_allows_src_and_denies_data_and_kernels(tmp_path: Path) -> None:
    repo_root = tmp_path
    module_file = repo_root / "src" / "kagglebot" / "autopilot.py"
    agent_dir = repo_root / "artifacts" / "demo" / "runs" / "run-1" / "iter-1" / "agent"
    module_file.parent.mkdir(parents=True, exist_ok=True)

    policy = write_guard.build_repair_write_policy(
        repo_root=repo_root,
        data_dir=repo_root / "artifacts" / "demo" / "data",
        kernels_dir=repo_root / "artifacts" / "demo" / "kernels",
        module_file=module_file,
        extra_allowed_prefixes=[agent_dir],
    )

    assert repo_root in policy.allowed_prefixes
    assert repo_root / "src" in policy.allowed_prefixes
    assert agent_dir in policy.allowed_prefixes
    assert repo_root / "artifacts" / "demo" / "data" in policy.denied_prefixes
    assert repo_root / "artifacts" / "demo" / "kernels" in policy.denied_prefixes
    assert policy.snapshot_prefixes == (
        repo_root / "src",
        repo_root / "artifacts" / "demo" / "kernel",
        repo_root / "artifacts" / "demo" / "data",
        repo_root / "artifacts" / "demo" / "kernels",
    )
    assert policy.max_backup_bytes == write_guard._MAX_GUARD_TOTAL_BACKUP_BYTES


def test_repo_root_write_policy_rejects_sensitive_repo_files(tmp_path: Path) -> None:
    repo_root = tmp_path
    env_path = repo_root / ".env.local"
    env_path.write_text("TOKEN=old\n", encoding="utf-8")
    policy = write_guard._repo_root_write_policy(
        repo_root=repo_root,
        denied_prefixes=[repo_root / "artifacts" / "demo" / "data", repo_root / "artifacts" / "demo" / "kernels"],
    )
    guard_snapshot = write_guard._backup_guarded_files(repo_root, policy)
    before = write_guard._snapshot_tree(repo_root)

    env_path.write_text("TOKEN=new\n", encoding="utf-8")
    after = write_guard._snapshot_tree(repo_root)

    write_guard._enforce_allowlist_changes(
        root=repo_root,
        before=before,
        after=after,
        allowed_prefixes=policy,
        stage="test_sensitive_repo_path",
        guard_snapshot=guard_snapshot,
        auto_repair=True,
    )

    assert env_path.read_text(encoding="utf-8") == "TOKEN=old\n"


def test_guard_rejects_external_sensitive_path_edits(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    tracked = repo_root / "src" / "demo.py"
    tracked.parent.mkdir(parents=True, exist_ok=True)
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    external_path = tmp_path / "outside" / "kaggle.json"
    external_path.parent.mkdir(parents=True, exist_ok=True)
    external_path.write_text('{"username":"demo"}\n', encoding="utf-8")

    policy = write_guard.WriteGuardPolicy(
        allowed_prefixes=(repo_root,),
        external_guard_paths=(external_path,),
    )
    guard_snapshot = write_guard._backup_guarded_files(repo_root, policy)
    before = write_guard._snapshot_tree(repo_root)

    external_path.write_text('{"username":"changed"}\n', encoding="utf-8")
    after = write_guard._snapshot_tree(repo_root)

    try:
        write_guard._enforce_allowlist_changes(
            root=repo_root,
            before=before,
            after=after,
            allowed_prefixes=policy,
            stage="test_external_guard",
            guard_snapshot=guard_snapshot,
            auto_repair=True,
        )
    except KaggleBotError as exc:
        assert "forbidden external path edited" in str(exc)
        assert str(external_path) in str(exc)
    else:
        raise AssertionError("expected forbidden external path edit to be rejected")


def test_auto_strategy_engine_requires_oracle() -> None:
    config = agent_pipeline.AgentPipelineConfig(
        slug="demo",
        competition_url=None,
        compute="local_gpu",
        accelerator="gpu",
        internet="off",
        run_id="run-1",
        dry_run=False,
        repo_root=Path("."),
        strategy_engine="auto",
    )

    assert agent_pipeline._strategy_engine_is_required(config, "oracle") is True  # noqa: SLF001


def test_snapshot_tree_prunes_noise_directories(tmp_path: Path) -> None:
    tracked = tmp_path / "src" / "tracked.py"
    noise_files = [
        tmp_path / ".git" / "index",
        tmp_path / ".venv" / "lib" / "package.py",
        tmp_path / "nested" / ".venv" / "package.py",
        tmp_path / "tests" / "__pycache__" / "cached.pyc",
        tmp_path / ".pytest_cache" / "state",
    ]
    tracked.parent.mkdir(parents=True)
    tracked.write_text("tracked\n", encoding="utf-8")
    for path in noise_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("noise\n", encoding="utf-8")

    snapshot = write_guard._snapshot_tree(tmp_path)

    assert snapshot == {"src/tracked.py": (tracked.stat().st_mtime_ns, tracked.stat().st_size)}


def test_snapshot_tree_scopes_repair_policy_to_current_competition_paths(tmp_path: Path) -> None:
    source = tmp_path / "src" / "kagglebot" / "autopilot.py"
    kernel = tmp_path / "artifacts" / "demo" / "kernel" / "kernel.py"
    data = tmp_path / "artifacts" / "demo" / "data" / "train.csv"
    staged = tmp_path / "artifacts" / "demo" / "kernels" / "run-1" / "kernel.py"
    historical = tmp_path / "artifacts" / "demo" / "kernels" / "old-run" / "kernel.py"
    unrelated = tmp_path / "artifacts" / "other" / "kernel" / "kernel.py"
    for path in (source, kernel, data, staged, historical, unrelated):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")
    policy = write_guard.build_repair_write_policy(
        repo_root=tmp_path,
        data_dir=data.parent,
        kernels_dir=staged.parent,
        module_file=tmp_path / "src" / "kagglebot" / "autopilot.py",
    )

    snapshot = write_guard._snapshot_tree(tmp_path, policy)

    assert source.relative_to(tmp_path).as_posix() in snapshot
    assert kernel.relative_to(tmp_path).as_posix() in snapshot
    assert data.relative_to(tmp_path).as_posix() in snapshot
    assert staged.relative_to(tmp_path).as_posix() in snapshot
    assert historical.relative_to(tmp_path).as_posix() not in snapshot
    assert unrelated.relative_to(tmp_path).as_posix() not in snapshot
