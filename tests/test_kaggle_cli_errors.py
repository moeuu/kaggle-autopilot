from __future__ import annotations

from kagglebot.exceptions import KaggleCliError
from kagglebot.kaggle_cli_errors import is_missing_kaggle_credentials_error


def test_missing_kaggle_credentials_detects_missing_kaggle_json_output() -> None:
    error = KaggleCliError(
        "Kaggle CLI failed",
        output="OSError: Could not find kaggle.json. Or use the environment method.",
    )

    assert is_missing_kaggle_credentials_error(error)


def test_missing_kaggle_credentials_detects_authenticate_traceback_in_stderr() -> None:
    error = KaggleCliError(
        "Kaggle CLI failed",
        stderr="RuntimeError from api.authenticate: missing kaggle.json",
    )

    assert is_missing_kaggle_credentials_error(error)


def test_missing_kaggle_credentials_detects_generic_api_credentials_message() -> None:
    error = KaggleCliError(
        "Kaggle CLI failed",
        stderr="Kaggle API credentials not found for streaming download.",
    )

    assert is_missing_kaggle_credentials_error(error)


def test_missing_kaggle_credentials_rejects_unrelated_cli_errors() -> None:
    error = KaggleCliError("Kaggle CLI failed", stderr="404 competition not found")

    assert not is_missing_kaggle_credentials_error(error)
