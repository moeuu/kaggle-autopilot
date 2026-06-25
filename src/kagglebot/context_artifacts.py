from __future__ import annotations

from pathlib import Path

from rich import print

from kagglebot.eval import validate_evaluation_spec
from kagglebot.json_utils import load_json_object
from kagglebot.plan_policy import apply_competition_eval_override


def count_csv_data_rows_capped(path: Path, *, cap: int = 10) -> int | None:
    data_rows = 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            for index, _line in enumerate(handle):
                if index > cap:
                    return cap + 1
                data_rows = index
    except OSError:
        return None
    return data_rows


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
