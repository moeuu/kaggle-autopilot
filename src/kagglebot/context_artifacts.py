from __future__ import annotations

from pathlib import Path

from rich import print

from kagglebot.eval import validate_evaluation_spec
from kagglebot.json_utils import load_json_object
from kagglebot.plan_policy import apply_competition_eval_override
from kagglebot.submission_sample_discovery import tabular_data_row_count_capped


def count_tabular_data_rows_capped(path: Path, *, cap: int = 10) -> int | None:
    """Count data rows for supported tabular submission artifacts, capped at cap + 1."""
    return tabular_data_row_count_capped(path, cap=cap)


def count_csv_data_rows_capped(path: Path, *, cap: int = 10) -> int | None:
    """Backward-compatible alias for count_tabular_data_rows_capped."""
    return count_tabular_data_rows_capped(path, cap=cap)


def load_dataset_profile(*, slug: str, dataset_profile_path: Path) -> dict[str, object]:
    payload = load_json_object(dataset_profile_path)
    if payload is None:
        return {}
    return apply_competition_eval_override(slug=slug, payload=payload)


def load_evaluation_spec(*, slug: str, evaluation_spec_path: Path) -> dict[str, object]:
    payload = load_json_object(evaluation_spec_path)
    if payload is None:
        return {}
    spec, issues = validate_evaluation_spec(payload)
    if issues:
        issue_text = "; ".join(issues)
        print(f"[yellow]evaluation spec ignored[/yellow]: {issue_text}")
        return apply_competition_eval_override(slug=slug, payload={}, include_spec_keys=True)
    return apply_competition_eval_override(
        slug=slug,
        payload=spec or {},
        include_spec_keys=True,
    )
