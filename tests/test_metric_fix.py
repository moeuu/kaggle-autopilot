from __future__ import annotations

from kagglebot.metric_fix import run_metric_only_competition_metric_fix


def test_run_metric_only_competition_metric_fix_builds_restricted_kernel_fix_request() -> None:
    calls: list[dict[str, object]] = []

    run_metric_only_competition_metric_fix(
        mismatch_reason="target=auc/maximize, kernel=accuracy/maximize",
        attempt=2,
        codex_model="gpt-test",
        codex_reasoning_effort="high",
        max_codex_passes=4,
        run_kernel_fix=lambda **kwargs: calls.append(kwargs),
    )

    assert len(calls) == 1
    request = calls[0]
    assert request["attempt"] == 2
    assert request["use_gpt_strategy"] is True
    assert request["codex_model"] == "gpt-test"
    assert request["codex_reasoning_effort"] == "high"
    assert request["max_codex_passes"] == 4
    assert "Competition metric mismatch detected" in str(request["error_message"])
    assert "target=auc/maximize" in str(request["error_message"])
    assert "Edit ONLY competition metric" in str(request["prompt_prefix"])
    assert "Do NOT change model architecture" in str(request["prompt_prefix"])
