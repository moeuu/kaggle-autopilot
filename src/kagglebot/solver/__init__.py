from __future__ import annotations

from kagglebot.solver.io import CompetitionData, load_competition_data, write_submission

__all__ = [
    "CompetitionData",
    "load_competition_data",
    "write_submission",
    "validate_submission_file",
]


def __getattr__(name: str):
    if name == "validate_submission_file":
        from kagglebot.solver.validate import validate_submission_file

        return validate_submission_file
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
