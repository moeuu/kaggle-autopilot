from __future__ import annotations

from kaggle.api.kaggle_api_extended import KaggleApi


def get_kaggle_api() -> KaggleApi:
    """
    Get authenticated Kaggle API instance.
    Uses OAuth token from ~/.kaggle/access_token automatically.
    """
    api = KaggleApi()
    api.authenticate()
    return api


def kaggle_submit(slug: str, submission_file: str, message: str) -> None:
    """
    Submit a file to a Kaggle competition using the Python API.

    Args:
        slug: Competition slug (e.g., 'titanic')
        submission_file: Path to submission CSV file
        message: Submission description message

    Raises:
        RuntimeError: If submission fails
    """
    api = get_kaggle_api()
    try:
        result = api.competition_submit(submission_file, message, slug)
        # The API returns a SubmitResult object, but may raise on error
        print(f"Submission successful: {result}")
    except Exception as e:
        raise RuntimeError(f"Kaggle submission failed: {e}") from e
