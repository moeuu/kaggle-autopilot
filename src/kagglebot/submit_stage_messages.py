from __future__ import annotations


def format_iteration_submit_status_message(
    *,
    iteration: int,
    max_iterations: int,
    submit_enabled: bool,
    submit_allowed_by_gate: bool,
    submit_phase_state: str,
    quality_reasons: list[str],
    competition_faithfulness: dict[str, object] | None = None,
) -> str | None:
    if not submit_enabled:
        return None
    if submit_allowed_by_gate:
        return f"[cyan]submit[/cyan]: iter {iteration}/{max_iterations} attempting submission now."
    detail = ""
    if quality_reasons and submit_phase_state == "blocked_quality_guard":
        detail = f" reasons={','.join(quality_reasons)}"
    if isinstance(competition_faithfulness, dict):
        detail = f"{detail}{format_competition_faithfulness_detail(competition_faithfulness)}"
    return (
        "[cyan]submit[/cyan]: "
        f"iter {iteration}/{max_iterations} not attempted yet "
        f"(state={submit_phase_state}{detail})."
    )


def format_competition_faithfulness_detail(competition_faithfulness: dict[str, object]) -> str:
    metric_detail = ""
    expected_metric = str(competition_faithfulness.get("expected_metric") or "").strip()
    actual_metric = str(competition_faithfulness.get("actual_metric") or "").strip()
    if expected_metric or actual_metric:
        metric_detail = f" metric={actual_metric or 'unknown'}/{expected_metric or 'unknown'}"

    split_detail = ""
    expected_split = str(competition_faithfulness.get("expected_split_strategy") or "").strip()
    actual_split = str(competition_faithfulness.get("actual_split_strategy") or "").strip()
    if expected_split or actual_split:
        split_detail = f" split={actual_split or 'unknown'}/{expected_split or 'unknown'}"

    dataset_mode = str(competition_faithfulness.get("dataset_mode") or "").strip()
    dataset_detail = f" dataset_mode={dataset_mode}" if dataset_mode else ""
    return f"{metric_detail}{split_detail}{dataset_detail}"


def extract_submission_row_message(row: dict[str, object]) -> str:
    for key in (
        "errorDescription",
        "error_description",
        "failureReason",
        "failure_reason",
        "error",
        "message",
        "comments",
        "comment",
        "description",
    ):
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""
